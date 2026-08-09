"""Single-file and folder-batch speech-enhancement scene capture."""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .audio import AudioBackend
from .config import ExperimentConfig, SceneConfig
from .labels import write_label_files
from .quality import channel_metrics
from .signals import load_audio, route_outputs, scale_dbfs
from .storage import RunStore, sha256

Log = Callable[[str], None]


@dataclass(frozen=True)
class SourcePair:
    target: Path | None
    interferer: Path | None


def discover_source_pairs(scene: SceneConfig) -> list[SourcePair]:
    """Resolve stable target/interferer pairs from files or folders."""
    target_required = any(item in {"target_only", "mixture"} for item in scene.items)
    interferer_required = any(item in {"interferer_only", "mixture"} for item in scene.items)
    if scene.source_mode == "single":
        target = Path(scene.target_file) if target_required else None
        interferer = Path(scene.interferer_file) if interferer_required else None
        _require_file(target, "目标语音")
        _require_file(interferer, "干扰声音")
        return [SourcePair(target, interferer)]

    extensions = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in scene.file_extensions
    }
    targets = _scan_audio_folder(Path(scene.target_folder), extensions) if target_required else [None]
    interferers = (
        _scan_audio_folder(Path(scene.interferer_folder), extensions) if interferer_required else [None]
    )
    if target_required and not targets:
        raise FileNotFoundError(f"目标语音文件夹中没有匹配音频：{scene.target_folder}")
    if interferer_required and not interferers:
        raise FileNotFoundError(f"干扰声音文件夹中没有匹配音频：{scene.interferer_folder}")
    if scene.pairing_mode == "cartesian":
        return [SourcePair(target, interferer) for target, interferer in itertools.product(targets, interferers)]
    count = max(len(targets), len(interferers))
    return [SourcePair(targets[index % len(targets)], interferers[index % len(interferers)]) for index in range(count)]


def _require_file(path: Path | None, label: str) -> None:
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"{label}文件不存在：{path}")


def _scan_audio_folder(folder: Path, extensions: set[str]) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"音频文件夹不存在：{folder}")
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: str(path.relative_to(folder)).casefold(),
    )


def _fit_once(signal: np.ndarray, samples: int) -> np.ndarray:
    """Trim long clips and zero-pad short clips; never repeat speech content."""
    output = np.zeros(samples, dtype=np.float32)
    count = min(samples, len(signal))
    output[:count] = signal[:count]
    return output


def _pair_sample_count(pair: SourcePair, config: ExperimentConfig) -> int:
    scene, fs = config.scene, config.audio.sample_rate
    if scene.duration_s is not None:
        return round(scene.duration_s * fs)
    lengths = []
    for path in (pair.target, pair.interferer):
        if path is not None:
            info = sf.info(path)
            lengths.append(round(info.frames * fs / info.samplerate))
    return min(lengths) if lengths else fs


def _load_pair(pair: SourcePair, config: ExperimentConfig, samples: int) -> tuple[np.ndarray, np.ndarray]:
    scene, fs = config.scene, config.audio.sample_rate
    target = (
        scale_dbfs(_fit_once(load_audio(pair.target, fs), samples), scene.target_level_dbfs)
        if pair.target is not None
        else np.zeros(samples, dtype=np.float32)
    )
    interferer = (
        scale_dbfs(_fit_once(load_audio(pair.interferer, fs), samples), scene.interferer_level_dbfs)
        if pair.interferer is not None
        else np.zeros(samples, dtype=np.float32)
    )
    return target, interferer


def _safe_label(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip())
    return value.strip("_") or "sample"


def _automatic_label(scene: SceneConfig, pair: SourcePair, index: int) -> str:
    parts = [scene.label_prefix, f"{index:04d}"]
    if pair.target is not None:
        parts.append(pair.target.stem)
    if pair.interferer is not None:
        parts.append(pair.interferer.stem)
    return _safe_label("_".join(part for part in parts if part))


def _relative(path: Path | None, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/") if path is not None else ""


def _quality_flag(metrics_by_item: dict[str, dict]) -> str:
    if any(bool(metrics.get("backend_status", {}).get("xrun")) for metrics in metrics_by_item.values()):
        return "音频丢帧"
    channels = [
        channel
        for item, metrics in metrics_by_item.items()
        if item != "ambient"
        for channel in metrics.get("channels", [])
    ]
    if any(bool(channel.get("clipped")) for channel in channels):
        return "削波"
    if channels and max(float(channel.get("peak", 0.0)) for channel in channels) <= 1e-12:
        return "全零录音"
    return "通过"


def capture_scene_block(
    config: ExperimentConfig,
    backend: AudioBackend,
    log: Log = print,
) -> RunStore:
    fs, scene = config.audio.sample_rate, config.scene
    pairs = discover_source_pairs(scene)
    store = RunStore.create(config, "scene")
    output_channels = max(
        config.audio.target_output_channel,
        config.audio.interferer_output_channel,
    )
    source_info = []
    for index, pair in enumerate(pairs, 1):
        source_info.append(
            {
                "sample_index": index,
                "automatic_label": _automatic_label(scene, pair, index),
                "target": {
                    "path": str(pair.target) if pair.target else None,
                    "sha256": sha256(pair.target) if pair.target else None,
                },
                "interferer": {
                    "path": str(pair.interferer) if pair.interferer else None,
                    "sha256": sha256(pair.interferer) if pair.interferer else None,
                },
            }
        )
    store.write_json("references/sources.json", {"mode": scene.source_mode, "pairs": source_info})
    summary: dict = {
        "source_mode": scene.source_mode,
        "pairing_mode": scene.pairing_mode,
        "pair_count": len(pairs),
        "items": scene.items,
        "repetitions": scene.repetitions,
        "captures": [],
    }
    label_rows: list[dict] = []

    try:
        for repetition in range(1, scene.repetitions + 1):
            ambient_path: Path | None = None
            if "ambient" in scene.items:
                log(f"场景：环境底噪，重复 {repetition}/{scene.repetitions}")
                if scene.countdown_s:
                    log(f"  将在 {scene.countdown_s:g} 秒后开始")
                    time.sleep(scene.countdown_s)
                capture = backend.record(round(scene.ambient_duration_s * fs))
                ambient_path = store.write_audio(
                    f"raw/rep_{repetition:03d}_ambient_mics.wav",
                    capture.microphones,
                    fs,
                )
                ambient_metrics = {
                    "item": "ambient",
                    "repetition": repetition,
                    "file": _relative(ambient_path, store.root),
                    "channels": channel_metrics(capture.microphones),
                    "backend_status": capture.status,
                }
                store.write_json(f"metrics/rep_{repetition:03d}_ambient.json", ambient_metrics)
                summary["captures"].append(ambient_metrics)

            for pair_index, pair in enumerate(pairs, 1):
                samples = _pair_sample_count(pair, config)
                target, interferer = _load_pair(pair, config, samples)
                automatic_label = _automatic_label(scene, pair, pair_index)
                log(
                    f"样本 {pair_index}/{len(pairs)}：{automatic_label}，"
                    f"时长 {samples / fs:.3f} 秒"
                )
                single_legacy = scene.source_mode == "single" and len(pairs) == 1
                reference_prefix = "" if single_legacy else f"sample_{pair_index:04d}_"
                target_emitted: Path | None = None
                interferer_emitted: Path | None = None
                if config.storage.save_playback_reference:
                    if pair.target is not None:
                        target_emitted = store.path(f"references/{reference_prefix}target_emitted.wav")
                        if not target_emitted.exists():
                            target_emitted = store.write_audio(
                                f"references/{reference_prefix}target_emitted.wav", target, fs
                            )
                    if pair.interferer is not None:
                        interferer_emitted = store.path(f"references/{reference_prefix}interferer_emitted.wav")
                        if not interferer_emitted.exists():
                            interferer_emitted = store.write_audio(
                                f"references/{reference_prefix}interferer_emitted.wav", interferer, fs
                            )

                recording_paths: dict[str, Path] = {}
                playback_paths: dict[str, Path] = {}
                metrics_paths: dict[str, Path] = {}
                metrics_by_item: dict[str, dict] = {}
                for item in [value for value in scene.items if value != "ambient"]:
                    item_name = {
                        "target_only": "仅目标声源",
                        "interferer_only": "仅干扰声源",
                        "mixture": "目标与干扰同时播放",
                    }[item]
                    log(f"  场景：{item_name}，重复 {repetition}/{scene.repetitions}")
                    if scene.countdown_s:
                        log(f"    将在 {scene.countdown_s:g} 秒后开始")
                        time.sleep(scene.countdown_s)
                    routed: dict[int, np.ndarray] = {}
                    if item in {"target_only", "mixture"}:
                        routed[config.audio.target_output_channel] = target
                    if item in {"interferer_only", "mixture"}:
                        routed[config.audio.interferer_output_channel] = interferer
                    playback = route_outputs(routed, samples, output_channels=output_channels)
                    capture = backend.play_record(playback)
                    stem = (
                        f"rep_{repetition:03d}_{item}"
                        if single_legacy
                        else f"sample_{pair_index:04d}_rep_{repetition:03d}_{item}"
                    )
                    raw_path = store.write_audio(f"raw/{stem}_mics.wav", capture.microphones, fs)
                    recording_paths[item] = raw_path
                    if config.storage.save_playback_reference:
                        playback_paths[item] = store.write_audio(
                            f"references/{stem}_playback.wav", playback, fs
                        )
                    metrics = {
                        "sample_index": pair_index,
                        "automatic_label": automatic_label,
                        "item": item,
                        "repetition": repetition,
                        "file": _relative(raw_path, store.root),
                        "channels": channel_metrics(capture.microphones),
                        "backend_status": capture.status,
                    }
                    metrics_path = store.write_json(f"metrics/{stem}.json", metrics)
                    metrics_paths[item] = metrics_path
                    metrics_by_item[item] = metrics
                    summary["captures"].append(metrics)
                    if scene.gap_s:
                        time.sleep(scene.gap_s)

                label_rows.append(
                    {
                        "sample_id": f"{automatic_label}_rep{repetition:03d}",
                        "manual_label": "",
                        "automatic_label": automatic_label,
                        "dataset_split": scene.dataset_split,
                        "valid": "是",
                        "quality_flag": _quality_flag(metrics_by_item),
                        "repetition": repetition,
                        "target_source": str(pair.target) if pair.target else "",
                        "interferer_source": str(pair.interferer) if pair.interferer else "",
                        "duration_s": samples / fs,
                        "sample_rate_hz": fs,
                        "microphone_channels": ",".join(map(str, config.audio.input_channels)),
                        "target_output_channel": config.audio.target_output_channel,
                        "interferer_output_channel": config.audio.interferer_output_channel,
                        "target_level_dbfs": scene.target_level_dbfs,
                        "interferer_level_dbfs": scene.interferer_level_dbfs,
                        "ambient_recording": _relative(ambient_path, store.root),
                        "target_recording": _relative(recording_paths.get("target_only"), store.root),
                        "interferer_recording": _relative(recording_paths.get("interferer_only"), store.root),
                        "mixture_recording": _relative(recording_paths.get("mixture"), store.root),
                        "target_playback": _relative(playback_paths.get("target_only"), store.root),
                        "interferer_playback": _relative(playback_paths.get("interferer_only"), store.root),
                        "mixture_playback": _relative(playback_paths.get("mixture"), store.root),
                        "metrics_files": {
                            item: _relative(path, store.root) for item, path in metrics_paths.items()
                        },
                        "metadata_json": config.metadata,
                        "notes": "",
                    }
                )

        label_files = write_label_files(store.root, label_rows, config.metadata)
        for path in label_files.values():
            store.add_artifact(path.name)
        summary["labels"] = {name: path.name for name, path in label_files.items()}
        summary["label_rows"] = len(label_rows)
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        log(f"语音增强场景结果与标签表已保存到：{store.root}")
        return store
    except Exception as exc:
        store.finish({"error": str(exc), **summary}, status="failed")
        raise
