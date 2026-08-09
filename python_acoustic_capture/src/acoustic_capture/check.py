"""Non-playing microphone sanity check."""

from __future__ import annotations

import numpy as np

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics
from .storage import RunStore


def capture_input_check(config: ExperimentConfig, backend: AudioBackend, duration_s: float = 5.0) -> RunStore:
    frames = round(duration_s * config.audio.sample_rate)
    store = RunStore.create(config, "input_check")
    try:
        result = backend.record(frames)
        store.write_audio("raw/input_check_mics.wav", result.microphones, config.audio.sample_rate)
        channels = channel_metrics(result.microphones, config.repeats.clip_threshold)
        warnings = []
        if channels and max(float(item["peak"]) for item in channels) <= 1e-12:
            warnings.append("所有录制通道均为全零，请检查声卡输入路由、麦克风增益和系统权限")
        if result.status.get("xrun"):
            warnings.append("录制期间发生输入溢出或欠载，请调整声卡缓冲区后重试")
        summary = {
            "duration_s": duration_s,
            "channels": channels,
            "backend_status": result.status,
            "warnings": warnings,
        }
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        return store
    except Exception as exc:
        store.finish({"error": str(exc)}, status="failed")
        raise


def capture_silent_duplex_check(
    config: ExperimentConfig, backend: AudioBackend, duration_s: float = 1.0
) -> RunStore:
    """Open the real full-duplex path while emitting digital silence."""
    frames = round(duration_s * config.audio.sample_rate)
    channels = max(config.audio.target_output_channel, config.audio.interferer_output_channel)
    silence = np.zeros((frames, channels), dtype=np.float32)
    store = RunStore.create(config, "silent_duplex_check")
    try:
        result = backend.play_record(silence)
        store.write_audio("raw/silent_duplex_mics.wav", result.microphones, config.audio.sample_rate)
        warnings = []
        if result.status.get("xrun"):
            warnings.append("静音双工检查发生输入溢出或输出欠载，请调整声卡缓冲区后重试")
        summary = {
            "duration_s": duration_s,
            "output_channels_opened": channels,
            "recorded_channels": channel_metrics(result.microphones, config.repeats.clip_threshold),
            "backend_status": result.status,
            "warnings": warnings,
        }
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        return store
    except Exception as exc:
        store.finish({"error": str(exc)}, status="failed")
        raise
