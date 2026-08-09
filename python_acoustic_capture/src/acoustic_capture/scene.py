"""Selectable target/interferer scene-block capture."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics
from .signals import fit_duration, load_audio, route_outputs, scale_dbfs
from .storage import RunStore, sha256

Log = Callable[[str], None]


def _source(config: ExperimentConfig, kind: str, samples: int) -> np.ndarray:
    path = config.scene.target_file if kind == "target" else config.scene.interferer_file
    level = config.scene.target_level_dbfs if kind == "target" else config.scene.interferer_level_dbfs
    return scale_dbfs(fit_duration(load_audio(path, config.audio.sample_rate), samples), level)


def capture_scene_block(
    config: ExperimentConfig,
    backend: AudioBackend,
    log: Log = print,
) -> RunStore:
    fs, scene = config.audio.sample_rate, config.scene
    paths = [Path(scene.target_file), Path(scene.interferer_file)]
    for item in scene.items:
        if item in {"target_only", "mixture"} and not paths[0].is_file():
            raise FileNotFoundError(paths[0])
        if item in {"interferer_only", "mixture"} and not paths[1].is_file():
            raise FileNotFoundError(paths[1])

    lengths = []
    for path in paths:
        if path.is_file():
            import soundfile as sf

            lengths.append(round(len(sf.SoundFile(path)) * fs / sf.info(path).samplerate))
    samples = round(scene.duration_s * fs) if scene.duration_s else (min(lengths) if lengths else fs)
    target = _source(config, "target", samples) if paths[0].is_file() else np.zeros(samples, np.float32)
    interferer = _source(config, "interferer", samples) if paths[1].is_file() else np.zeros(samples, np.float32)
    store = RunStore.create(config, "scene")
    if config.storage.save_playback_reference:
        if paths[0].is_file():
            store.write_audio("references/target_emitted.wav", target, fs)
        if paths[1].is_file():
            store.write_audio("references/interferer_emitted.wav", interferer, fs)

    source_info = {
        "target": {"path": str(paths[0]), "sha256": sha256(paths[0]) if paths[0].is_file() else None},
        "interferer": {"path": str(paths[1]), "sha256": sha256(paths[1]) if paths[1].is_file() else None},
    }
    store.write_json("references/sources.json", source_info)
    summary: dict = {"items": scene.items, "repetitions": scene.repetitions, "captures": []}
    try:
        for repetition in range(1, scene.repetitions + 1):
            for item in scene.items:
                stem = f"rep_{repetition:03d}_{item}"
                item_name = {
                    "ambient": "环境底噪",
                    "target_only": "仅目标声源",
                    "interferer_only": "仅干扰声源",
                    "mixture": "目标与干扰同时播放",
                }[item]
                log(f"场景：{item_name}，重复 {repetition}/{scene.repetitions}")
                if scene.countdown_s:
                    log(f"  将在 {scene.countdown_s:g} 秒后开始")
                    time.sleep(scene.countdown_s)
                if item == "ambient":
                    capture = backend.record(round(scene.ambient_duration_s * fs))
                    playback = None
                else:
                    routed: dict[int, np.ndarray] = {}
                    if item in {"target_only", "mixture"}:
                        routed[config.audio.target_output_channel] = target
                    if item in {"interferer_only", "mixture"}:
                        routed[config.audio.interferer_output_channel] = interferer
                    playback = route_outputs(
                        routed,
                        samples,
                        output_channels=max(
                            config.audio.target_output_channel,
                            config.audio.interferer_output_channel,
                        ),
                    )
                    capture = backend.play_record(playback)
                raw_path = store.write_audio(f"raw/{stem}_mics.wav", capture.microphones, fs)
                if playback is not None and config.storage.save_playback_reference:
                    store.write_audio(f"references/{stem}_playback.wav", playback, fs)
                metrics = {
                    "item": item,
                    "repetition": repetition,
                    "file": str(raw_path.relative_to(store.root)),
                    "channels": channel_metrics(capture.microphones),
                    "backend_status": capture.status,
                }
                store.write_json(f"metrics/{stem}.json", metrics)
                summary["captures"].append(metrics)
                if scene.gap_s:
                    time.sleep(scene.gap_s)
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        log(f"语音增强场景结果已保存到：{store.root}")
        return store
    except Exception as exc:
        store.finish({"error": str(exc), **summary}, status="failed")
        raise
