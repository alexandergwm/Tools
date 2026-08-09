"""Non-playing microphone sanity check."""

from __future__ import annotations

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
