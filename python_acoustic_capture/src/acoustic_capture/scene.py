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
from .labels import flatten_experiment_metadata, write_label_files
from .quality import (
    channel_metrics,
    evaluate_supervision_quality_gate,
    mixture_additivity_metrics,
    multichannel_health_metrics,
)
from .signals import load_audio, route_outputs, scale_dbfs
from .storage import RunStore, sha256

Log = Callable[[str], None]
StopRequested = Callable[[], bool]
Progress = Callable[[dict], None]
PAIRED_ITEM_ORDER = ("target_only", "interferer_only", "mixture")


class _CaptureCancelled(Exception):
    pass


def _check_cancelled(stop_requested: StopRequested | None) -> None:
    if stop_requested is not None and stop_requested():
        raise _CaptureCancelled


def _wait_or_cancel(seconds: float, stop_requested: StopRequested | None) -> None:
    if stop_requested is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _check_cancelled(stop_requested)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    _check_cancelled(stop_requested)


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
    value = re.sub(r"[^\w-]+", "_", value.strip(), flags=re.UNICODE)
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


def build_paired_sequence(
    target: np.ndarray,
    interferer: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, dict]]:
    """Place all requested speech scenes in one continuous output stream.

    The target-only and mixture blocks reuse the exact same target samples.
    Because the backend is opened only once, every block shares one hardware
    input/output latency and can be split on a common sample grid.
    """
    if len(target) != len(interferer):
        raise ValueError("target and interferer must have equal lengths")
    samples = len(target)
    fs = config.audio.sample_rate
    gap_samples = round(config.scene.gap_s * fs)
    items = [item for item in PAIRED_ITEM_ORDER if item in config.scene.items]
    output_channels = max(
        config.audio.target_output_channel,
        config.audio.interferer_output_channel,
    )
    total_samples = gap_samples + len(items) * (samples + gap_samples)
    sequence = np.zeros((total_samples, output_channels), dtype=np.float32)
    segments: dict[str, dict] = {}
    cursor = gap_samples
    for item in items:
        routed: dict[int, np.ndarray] = {}
        if item in {"target_only", "mixture"}:
            routed[config.audio.target_output_channel] = target
        if item in {"interferer_only", "mixture"}:
            routed[config.audio.interferer_output_channel] = interferer
        playback = route_outputs(routed, samples, output_channels=output_channels)
        start, end = cursor, cursor + samples
        sequence[start:end] = playback
        segments[item] = {
            "start_sample": start,
            "end_sample": end,
            "sample_count": samples,
            "playback": playback,
        }
        cursor = end + gap_samples
    return sequence, segments


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
    if any(
        metrics.get("array_health", {}).get("exact_duplicate_channel_pairs")
        for item, metrics in metrics_by_item.items()
        if item != "ambient"
    ):
        return "录制通道完全重复"
    if channels and max(float(channel.get("peak", 0.0)) for channel in channels) <= 1e-12:
        return "全零录音"
    return "通过"


def _capture_type(items: set[str]) -> str:
    if {"target_only", "mixture"}.issubset(items):
        return "supervised_pair"
    if items == {"target_only"}:
        return "target_only"
    if items == {"interferer_only"}:
        return "interferer_only"
    return "+".join(sorted(items))


def _shared_hardware_clock(config: ExperimentConfig) -> bool:
    """Return true only when a common duplex device clock is verifiable."""
    if config.audio.backend == "simulated":
        return True
    input_device = config.audio.input_device
    output_device = config.audio.output_device
    if input_device in (None, "") or output_device in (None, ""):
        return False
    return str(input_device).strip().casefold() == str(output_device).strip().casefold()


def _finish_scene(
    store: RunStore,
    label_rows: list[dict],
    summary: dict,
    metadata: dict,
    *,
    status: str,
) -> None:
    label_files = write_label_files(store.root, label_rows, metadata)
    for path in label_files.values():
        store.add_artifact(path.name)
    summary["labels"] = {name: path.name for name, path in label_files.items()}
    summary["label_rows"] = len(label_rows)
    summary["supervision_ready_rows"] = sum(
        row.get("supervision_ready") == "是" for row in label_rows
    )
    summary["cancelled"] = status == "cancelled"
    store.write_json("metrics/summary.json", summary)
    store.finish(summary, status=status)


def capture_scene_block(
    config: ExperimentConfig,
    backend: AudioBackend,
    log: Log = print,
    stop_requested: StopRequested | None = None,
    progress: Progress | None = None,
) -> RunStore:
    fs, scene = config.audio.sample_rate, config.scene
    if (
        "mixture" in scene.items
        and config.audio.target_output_channel == config.audio.interferer_output_channel
    ):
        raise ValueError("mixture capture requires different target and interferer output channels")
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
        "capture_strategy": scene.capture_strategy,
        "sample_alignment": "single_continuous_full_duplex_stream",
        "captures": [],
        "paired_sequences": [],
    }
    label_rows: list[dict] = []
    metadata_columns = flatten_experiment_metadata(config.metadata)
    shared_hardware_clock = _shared_hardware_clock(config)

    try:
        for repetition in range(1, scene.repetitions + 1):
            _check_cancelled(stop_requested)
            ambient_path: Path | None = None
            if "ambient" in scene.items:
                log(f"场景：环境底噪，重复 {repetition}/{scene.repetitions}")
                if scene.countdown_s:
                    log(f"  将在 {scene.countdown_s:g} 秒后开始")
                    _wait_or_cancel(scene.countdown_s, stop_requested)
                capture = backend.record(round(scene.ambient_duration_s * fs))
                _check_cancelled(stop_requested)
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

            if not any(item != "ambient" for item in scene.items):
                continue

            for pair_index, pair in enumerate(pairs, 1):
                _check_cancelled(stop_requested)
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
                microphones_by_item: dict[str, np.ndarray] = {}
                paired_playback, segments = build_paired_sequence(target, interferer, config)
                sequence_stem = (
                    f"rep_{repetition:03d}_paired_sequence"
                    if single_legacy
                    else f"sample_{pair_index:04d}_rep_{repetition:03d}_paired_sequence"
                )
                if progress is not None:
                    progress(
                        {
                            "event": "pair_prepared",
                            "pair_index": pair_index,
                            "pair_count": len(pairs),
                            "repetition": repetition,
                            "target_name": pair.target.name if pair.target else "(not used)",
                            "interferer_name": pair.interferer.name if pair.interferer else "(not used)",
                            "target": target,
                            "interferer": interferer,
                            "sample_rate": fs,
                            "stream_samples": len(paired_playback),
                            "segments": {
                                item: {
                                    "start_sample": segment["start_sample"],
                                    "end_sample": segment["end_sample"],
                                }
                                for item, segment in segments.items()
                            },
                        }
                    )
                log(
                    "  配对序列："
                    + " → ".join(
                        {
                            "target_only": "仅目标",
                            "interferer_only": "仅干扰",
                            "mixture": "同时发声",
                        }[item]
                        for item in segments
                    )
                )
                if scene.countdown_s:
                    log(f"    将在 {scene.countdown_s:g} 秒后开始一次连续播录")
                    _wait_or_cancel(scene.countdown_s, stop_requested)
                capture = backend.play_record(paired_playback)
                _check_cancelled(stop_requested)
                paired_recording_path = store.write_audio(
                    f"raw/{sequence_stem}_mics.wav", capture.microphones, fs
                )
                paired_playback_path: Path | None = None
                if config.storage.save_playback_reference:
                    paired_playback_path = store.write_audio(
                        f"references/{sequence_stem}_playback.wav", paired_playback, fs
                    )
                layout = {
                    "sample_index": pair_index,
                    "repetition": repetition,
                    "sample_rate_hz": fs,
                    "stream_sample_count": len(paired_playback),
                    "gap_samples": round(scene.gap_s * fs),
                    "gap_s": scene.gap_s,
                    "alignment_method": "single_continuous_full_duplex_stream",
                    "segments": {
                        item: {
                            "start_sample": segment["start_sample"],
                            "end_sample": segment["end_sample"],
                            "sample_count": segment["sample_count"],
                        }
                        for item, segment in segments.items()
                    },
                }
                layout_path = store.write_json(
                    f"metrics/{sequence_stem}_layout.json", layout
                )
                sequence_summary = {
                    **layout,
                    "recording": _relative(paired_recording_path, store.root),
                    "playback": _relative(paired_playback_path, store.root),
                    "backend_status": capture.status,
                }
                summary["paired_sequences"].append(sequence_summary)

                for item, segment in segments.items():
                    start, end = segment["start_sample"], segment["end_sample"]
                    playback = segment["playback"]
                    microphones = capture.microphones[start:end]
                    stem = (
                        f"rep_{repetition:03d}_{item}"
                        if single_legacy
                        else f"sample_{pair_index:04d}_rep_{repetition:03d}_{item}"
                    )
                    raw_path = store.write_audio(f"raw/{stem}_mics.wav", microphones, fs)
                    recording_paths[item] = raw_path
                    microphones_by_item[item] = microphones
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
                        "paired_stream": sequence_stem,
                        "segment_start_sample": start,
                        "segment_end_sample": end,
                        "channels": channel_metrics(microphones),
                        "array_health": multichannel_health_metrics(microphones),
                        "backend_status": capture.status,
                    }
                    metrics_path = store.write_json(f"metrics/{stem}.json", metrics)
                    metrics_paths[item] = metrics_path
                    metrics_by_item[item] = metrics
                    summary["captures"].append(metrics)

                target_mixture_aligned = bool(
                    "target_only" in segments
                    and "mixture" in segments
                    and np.array_equal(
                        segments["target_only"]["playback"][:, config.audio.target_output_channel - 1],
                        segments["mixture"]["playback"][:, config.audio.target_output_channel - 1],
                    )
                    and segments["target_only"]["sample_count"]
                    == segments["mixture"]["sample_count"]
                )
                interferer_mixture_aligned = bool(
                    "interferer_only" in segments
                    and "mixture" in segments
                    and np.array_equal(
                        segments["interferer_only"]["playback"][:, config.audio.interferer_output_channel - 1],
                        segments["mixture"]["playback"][:, config.audio.interferer_output_channel - 1],
                    )
                    and segments["interferer_only"]["sample_count"]
                    == segments["mixture"]["sample_count"]
                )
                additivity: dict[str, object] = {}
                if {"target_only", "interferer_only", "mixture"}.issubset(
                    microphones_by_item
                ):
                    additivity = mixture_additivity_metrics(
                        microphones_by_item["target_only"],
                        microphones_by_item["interferer_only"],
                        microphones_by_item["mixture"],
                    )
                quality_gate = evaluate_supervision_quality_gate(
                    additivity, config.metadata
                )
                if quality_gate["enabled"] and not additivity:
                    quality_gate["passed"] = False
                    quality_gate["reasons"] = [
                        "target_only, interferer_only and mixture are all required for additivity QC"
                    ]
                supervision_metrics_path: Path | None = None
                if additivity or quality_gate["enabled"]:
                    supervision_metrics_path = store.write_json(
                        f"metrics/{sequence_stem}_supervision_quality.json",
                        {"additivity": additivity, "quality_gate": quality_gate},
                    )
                    summary.setdefault("supervision_quality", []).append(
                        {
                            "sample_index": pair_index,
                            "repetition": repetition,
                            "metrics_file": _relative(
                                supervision_metrics_path, store.root
                            ),
                            "residual_db_max": additivity.get(
                                "residual_db_max", ""
                            ),
                            "correlation_min": additivity.get(
                                "correlation_min", ""
                            ),
                            "quality_gate": quality_gate,
                        }
                    )
                quality_flag = _quality_flag(metrics_by_item)
                if quality_gate["enabled"] and not quality_gate["passed"]:
                    quality_flag = "监督一致性未通过"
                supervision_ready = (
                    target_mixture_aligned
                    and shared_hardware_clock
                    and quality_flag == "通过"
                    and bool(quality_gate["passed"])
                )
                scene_id = str(config.metadata.get("scene_id") or config.storage.session_name)
                sample_id = (
                    f"{_safe_label(scene_id)}__{automatic_label}_rep{repetition:03d}"
                )
                segment_items = set(segments)
                source_hashes = source_info[pair_index - 1]

                label_rows.append(
                    {
                        **metadata_columns,
                        "run_id": store.root.name,
                        "sample_id": sample_id,
                        "supervision_pair_id": sample_id if target_mixture_aligned else "",
                        "capture_type": _capture_type(segment_items),
                        "scene_id": scene_id,
                        "manual_label": "",
                        "automatic_label": automatic_label,
                        "dataset_split": scene.dataset_split,
                        "valid": "是",
                        "quality_flag": quality_flag,
                        "supervision_ready": "是" if supervision_ready else "否",
                        "capture_strategy": "single_stream_paired_sequence",
                        "shared_hardware_clock": "是" if shared_hardware_clock else "否",
                        "target_mixture_sample_aligned": "是" if target_mixture_aligned else "否",
                        "interferer_mixture_sample_aligned": (
                            "是" if interferer_mixture_aligned else "否"
                        ),
                        "supervision_contract": (
                            "same_source_samples+single_continuous_full_duplex_stream"
                            if target_mixture_aligned and shared_hardware_clock
                            else ""
                        ),
                        "quality_gate_enabled": (
                            "是" if quality_gate["enabled"] else "否"
                        ),
                        "quality_gate_passed": (
                            "未启用"
                            if not quality_gate["enabled"]
                            else "是"
                            if quality_gate["passed"]
                            else "否"
                        ),
                        "mixture_consistency_residual_db_max": additivity.get(
                            "residual_db_max", ""
                        ),
                        "mixture_consistency_correlation_min": additivity.get(
                            "correlation_min", ""
                        ),
                        "mixture_consistency_metrics_json": additivity,
                        "supervision_quality_metrics": _relative(
                            supervision_metrics_path, store.root
                        ),
                        "repetition": repetition,
                        "target_source": str(pair.target) if pair.target else "",
                        "interferer_source": str(pair.interferer) if pair.interferer else "",
                        "target_source_sha256": source_hashes["target"]["sha256"] or "",
                        "interferer_source_sha256": (
                            source_hashes["interferer"]["sha256"] or ""
                        ),
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
                        "paired_sequence_recording": _relative(
                            paired_recording_path, store.root
                        ),
                        "paired_sequence_playback": _relative(
                            paired_playback_path, store.root
                        ),
                        "segment_layout": _relative(layout_path, store.root),
                        "target_segment_start_sample": (
                            segments.get("target_only", {}).get("start_sample", "")
                        ),
                        "mixture_segment_start_sample": (
                            segments.get("mixture", {}).get("start_sample", "")
                        ),
                        "metrics_files": {
                            item: _relative(path, store.root) for item, path in metrics_paths.items()
                        },
                        "metadata_json": config.metadata,
                        "notes": "",
                    }
                )

        _finish_scene(store, label_rows, summary, config.metadata, status="completed")
        log(f"语音增强场景结果与标签表已保存到：{store.root}")
        return store
    except _CaptureCancelled:
        _finish_scene(store, label_rows, summary, config.metadata, status="cancelled")
        log(f"语音增强采集已停止；已完成的数据保存在：{store.root}")
        return store
    except Exception as exc:
        if stop_requested is not None and stop_requested():
            _finish_scene(store, label_rows, summary, config.metadata, status="cancelled")
            log(f"语音增强采集已停止；已完成的数据保存在：{store.root}")
            return store
        store.finish({"error": str(exc), **summary}, status="failed")
        raise
