"""Repeated ESS room impulse response measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics, normalized_correlation
from .signals import exponential_sweep, measurement_signal, route_outputs
from .storage import RunStore

Log = Callable[[str], None]


@dataclass
class RIRTake:
    index: int
    rir: np.ndarray
    peak_samples: list[int]
    accepted: bool
    metrics: dict


def make_inverse(sweep: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    """Farina-style inverse for an exponential sine sweep."""
    # The reversed sweep runs from high to low frequency. Exponential decay
    # compensates for the longer energy density of the low-frequency portion.
    correction = np.exp(np.linspace(0.0, -np.log(end_hz / start_hz), len(sweep)))
    inverse = sweep[::-1].astype(np.float64) * correction
    response = fftconvolve(sweep.astype(np.float64), inverse, mode="full")
    inverse /= max(abs(response[len(sweep) - 1]), 1e-12)
    return inverse.astype(np.float32)


def extract_rir(
    recording: np.ndarray,
    inverse: np.ndarray,
    sample_rate: int,
    expected_peak: int,
    duration_s: float,
    pre_peak_s: float,
) -> tuple[np.ndarray, list[int]]:
    output_samples = round(duration_s * sample_rate)
    pre_samples = round(pre_peak_s * sample_rate)
    channels, peaks = [], []
    search_radius = round(0.25 * sample_rate)
    for channel in range(recording.shape[1]):
        deconvolved = fftconvolve(recording[:, channel], inverse, mode="full")
        low = max(0, expected_peak - search_radius)
        high = min(len(deconvolved), expected_peak + search_radius)
        peak = low + int(np.argmax(np.abs(deconvolved[low:high])))
        start = peak - pre_samples
        segment = np.zeros(output_samples, dtype=np.float32)
        src_start, dst_start = max(start, 0), max(-start, 0)
        count = min(len(deconvolved) - src_start, output_samples - dst_start)
        if count > 0:
            segment[dst_start : dst_start + count] = deconvolved[src_start : src_start + count]
        channels.append(segment)
        peaks.append(peak)
    return np.column_stack(channels), peaks


def align_rir(rir: np.ndarray, peak_samples: list[int], reference_peaks: list[int]) -> np.ndarray:
    aligned = np.zeros_like(rir)
    for channel in range(rir.shape[1]):
        shift = reference_peaks[channel] - peak_samples[channel]
        if shift >= 0:
            aligned[shift:, channel] = rir[: len(rir) - shift, channel]
        else:
            aligned[:shift, channel] = rir[-shift:, channel]
    return aligned


def capture_rir(
    config: ExperimentConfig,
    backend: AudioBackend,
    output_channel: int | None = None,
    log: Log = print,
) -> RunStore:
    fs, sweep_cfg, repeat_cfg = config.audio.sample_rate, config.sweep, config.repeats
    output_channel = output_channel or config.audio.target_output_channel
    sweep = exponential_sweep(
        fs,
        sweep_cfg.start_hz,
        sweep_cfg.end_hz,
        sweep_cfg.duration_s,
        sweep_cfg.fade_s,
        sweep_cfg.level_dbfs,
    )
    played = measurement_signal(sweep, fs, sweep_cfg.pre_silence_s, sweep_cfg.post_silence_s)
    inverse = make_inverse(sweep, sweep_cfg.start_hz, sweep_cfg.end_hz)
    output = route_outputs({output_channel: played})
    expected_peak = round(sweep_cfg.pre_silence_s * fs) + len(sweep) - 1

    store = RunStore.create(config, "rir")
    store.write_audio("references/sweep.wav", sweep, fs)
    store.write_audio("references/played.wav", played, fs)
    store.write_audio("references/inverse_filter.wav", inverse, fs)
    accepted: list[RIRTake] = []
    all_metrics: list[dict] = []
    stable_count = 0

    try:
        for take_index in range(1, repeat_cfg.maximum + 1):
            log(f"脉冲响应采集 {take_index}/{repeat_cfg.maximum}：正在播放扫频信号")
            capture = backend.play_record(output)
            raw = capture.microphones
            raw_metrics = channel_metrics(raw, repeat_cfg.clip_threshold)
            rir, peaks = extract_rir(
                raw,
                inverse,
                fs,
                expected_peak,
                sweep_cfg.rir_duration_s,
                sweep_cfg.pre_peak_s,
            )
            channel_count = rir.shape[1]
            correlations = [1.0] * channel_count
            drift = [0] * channel_count
            aligned = rir
            if accepted:
                reference = np.mean([take.rir for take in accepted], axis=0)
                reference_peaks = accepted[0].peak_samples
                # extract_rir already places every direct peak at pre_peak_s;
                # absolute peak positions are retained only as a clock/latency check.
                aligned = rir
                correlations = [
                    normalized_correlation(aligned[:, ch], reference[:, ch])
                    for ch in range(channel_count)
                ]
                drift = [peaks[ch] - reference_peaks[ch] for ch in range(channel_count)]
            clipped = any(bool(item["clipped"]) for item in raw_metrics)
            xrun = bool(capture.status.get("xrun"))
            accepted_now = not xrun and not (repeat_cfg.reject_clipped and clipped)
            if accepted and min(correlations) < repeat_cfg.correlation_threshold:
                accepted_now = False
            metrics = {
                "take": take_index,
                "accepted": accepted_now,
                "raw_channels": raw_metrics,
                "peak_samples": peaks,
                "peak_drift_samples": drift,
                "correlation_to_running_average": correlations,
                "backend_status": capture.status,
                "audio_xrun": xrun,
            }
            store.write_audio(f"raw/take_{take_index:03d}.wav", raw, fs)
            store.write_audio(f"processed/take_{take_index:03d}_rir.wav", aligned, fs)
            store.write_json(f"metrics/take_{take_index:03d}.json", metrics)
            all_metrics.append(metrics)
            if accepted_now:
                accepted.append(RIRTake(take_index, aligned, peaks, True, metrics))
                drift_ok = max(abs(value) for value in drift) <= repeat_cfg.peak_drift_samples
                corr_ok = min(correlations) >= repeat_cfg.correlation_threshold
                stable_count = stable_count + 1 if drift_ok and corr_ok else 0
                log(f"  已接受；相关性={min(correlations):.4f}，峰值漂移={drift}")
            else:
                stable_count = 0
                log(f"  已拒绝；音频丢帧={xrun}，削波={clipped}，相关性={min(correlations):.4f}")

            enough = len(accepted) >= repeat_cfg.minimum
            stable = stable_count >= repeat_cfg.required_stable_takes
            if enough and stable:
                log("已达到自适应重复停止条件")
                break
            if repeat_cfg.pause_s:
                time.sleep(repeat_cfg.pause_s)

        if len(accepted) < repeat_cfg.minimum:
            raise RuntimeError(
                f"只有 {len(accepted)} 次有效脉冲响应，最少需要 {repeat_cfg.minimum} 次"
            )
        stack = np.stack([take.rir for take in accepted])
        average = np.mean(stack, axis=0).astype(np.float32)
        median = np.median(stack, axis=0).astype(np.float32)
        store.write_audio("processed/average_rir.wav", average, fs)
        store.write_audio("processed/median_rir.wav", median, fs)
        summary = {
            "output_channel": output_channel,
            "attempted_takes": len(all_metrics),
            "accepted_takes": [take.index for take in accepted],
            "rejected_takes": [item["take"] for item in all_metrics if not item["accepted"]],
            "sample_rate": fs,
            "rir_samples": len(average),
        }
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        log(f"脉冲响应结果已保存到：{store.root}")
        return store
    except Exception as exc:
        store.finish({"error": str(exc)}, status="failed")
        raise
