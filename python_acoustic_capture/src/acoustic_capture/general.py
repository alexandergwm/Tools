"""Basic play-only, record-only, and simultaneous play/record workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics
from .signals import fit_duration, load_audio, route_outputs, scale_dbfs
from .storage import RunStore, sha256

Log = Callable[[str], None]
StopRequested = Callable[[], bool]


def capture_general_io(
    config: ExperimentConfig,
    backend: AudioBackend,
    log: Log = print,
    stop_requested: StopRequested | None = None,
) -> RunStore:
    """Run one basic I/O operation and archive exactly what was used."""
    io = config.general
    sample_rate = config.audio.sample_rate
    frames = round(io.duration_s * sample_rate)
    needs_playback = io.action in {"play", "play_record"}
    needs_recording = io.action in {"record", "play_record"}
    source_path = Path(io.source_file)
    if needs_playback and not source_path.is_file():
        raise FileNotFoundError(source_path)

    output = None
    if needs_playback:
        source = scale_dbfs(fit_duration(load_audio(source_path, sample_rate), frames), io.level_dbfs)
        output = route_outputs({io.output_channel: source}, frames)

    store = RunStore.create(config, f"io_{io.action}")
    summary: dict = {
        "action": io.action,
        "duration_s": io.duration_s,
        "output_channel": io.output_channel if needs_playback else None,
    }
    try:
        if stop_requested is not None and stop_requested():
            store.finish(summary, status="cancelled")
            return store
        if output is not None:
            store.write_audio("references/played.wav", output, sample_rate)
            store.write_json(
                "references/source.json",
                {"path": str(source_path), "sha256": sha256(source_path)},
            )
        action_name = {"play": "仅播放", "record": "仅录制", "play_record": "同步播放并录制"}[io.action]
        log(f"基础播录：{action_name}")
        if io.action == "play":
            summary["backend_status"] = backend.play(output)
        elif io.action == "record":
            result = backend.record(frames)
            store.write_audio("raw/recording.wav", result.microphones, sample_rate)
            summary["channels"] = channel_metrics(result.microphones)
            summary["backend_status"] = result.status
        else:
            result = backend.play_record(output)
            store.write_audio("raw/recording.wav", result.microphones, sample_rate)
            summary["channels"] = channel_metrics(result.microphones)
            summary["backend_status"] = result.status
        if stop_requested is not None and stop_requested():
            summary["cancelled"] = True
            store.write_json("metrics/summary.json", summary)
            store.finish(summary, status="cancelled")
            log(f"基础播录已停止；已完成的数据保存在：{store.root}")
            return store
        if needs_recording and all(item["peak"] <= 1e-12 for item in summary["channels"]):
            warning = "录制信号所有通道均为全零，请检查麦克风权限、输入设备、通道路由或硬件静音状态。"
            summary["warnings"] = [warning]
            log(f"警告：{warning}")
        if summary.get("backend_status", {}).get("xrun"):
            warning = "本次音频流发生输入溢出或输出欠载，数据可能包含丢帧，不建议用于正式数据集。"
            summary.setdefault("warnings", []).append(warning)
            log(f"警告：{warning}")
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        log(f"基础播录结果已保存到：{store.root}")
        return store
    except Exception as exc:
        if stop_requested is not None and stop_requested():
            summary["cancelled"] = True
            store.finish(summary, status="cancelled")
            log(f"基础播录已停止；已完成的数据保存在：{store.root}")
            return store
        store.finish({"error": str(exc), **summary}, status="failed")
        raise
