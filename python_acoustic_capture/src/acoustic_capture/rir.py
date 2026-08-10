"""Repeated ESS room impulse response measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics, normalized_correlation
from .signals import exponential_sweep, measurement_signal, route_outputs
from .storage import RunStore

Log = Callable[[str], None]
Progress = Callable[[Path, int], None]


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


def align_rir_to_reference(
    rir: np.ndarray,
    reference: np.ndarray,
    max_shift_samples: int = 32,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Remove small residual integer delays left after direct-peak cropping.

    ``extract_rir`` aligns the largest deconvolution peak, but a band-limited
    direct arrival can have several nearly equal samples.  Independent Windows
    input/output devices can also leave a few samples of residual delay.  The
    residual must be removed before repeat correlation and averaging.
    """
    if rir.shape != reference.shape:
        raise ValueError("RIR and reference must have the same shape")
    if max_shift_samples < 0:
        raise ValueError("max_shift_samples must be non-negative")

    aligned = np.zeros_like(rir)
    shifts: list[int] = []
    correlations: list[float] = []
    for channel in range(rir.shape[1]):
        best_shift = 0
        best_signal = rir[:, channel]
        best_correlation = normalized_correlation(best_signal, reference[:, channel])
        for shift in range(-max_shift_samples, max_shift_samples + 1):
            candidate = _shift_with_zeros(rir[:, channel], shift)
            correlation = normalized_correlation(candidate, reference[:, channel])
            if correlation > best_correlation:
                best_shift = shift
                best_signal = candidate
                best_correlation = correlation
        aligned[:, channel] = best_signal
        shifts.append(best_shift)
        correlations.append(best_correlation)
    return aligned, shifts, correlations


def _shift_with_zeros(signal: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    if shift == 0:
        shifted[:] = signal
    elif shift > 0 and shift < len(signal):
        shifted[shift:] = signal[:-shift]
    elif shift < 0 and -shift < len(signal):
        shifted[:shift] = signal[-shift:]
    return shifted


def sweep_snr_db(
    recording: np.ndarray,
    sample_rate: int,
    pre_silence_s: float,
    sweep_samples: int,
) -> list[float]:
    """Compare each microphone's sweep-window RMS with its leading noise RMS."""
    pre_samples = max(1, round(pre_silence_s * sample_rate))
    active_start = pre_samples
    active_end = min(len(recording), active_start + sweep_samples)
    if active_end <= active_start:
        return [float("-inf")] * recording.shape[1]
    values = []
    for channel in range(recording.shape[1]):
        noise = recording[:pre_samples, channel].astype(np.float64)
        active = recording[active_start:active_end, channel].astype(np.float64)
        noise_rms = np.sqrt(np.mean(noise * noise)) if len(noise) else 0.0
        active_rms = np.sqrt(np.mean(active * active)) if len(active) else 0.0
        values.append(
            float(20 * np.log10(max(active_rms, 1e-12) / max(noise_rms, 1e-12)))
        )
    return values


def capture_rir(
    config: ExperimentConfig,
    backend: AudioBackend,
    output_channel: int | None = None,
    log: Log = print,
    progress: Progress | None = None,
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
            sweep_snr = sweep_snr_db(
                raw,
                fs,
                sweep_cfg.pre_silence_s,
                len(sweep),
            )
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
            residual_alignment = [0] * channel_count
            drift = [0] * channel_count
            aligned = rir
            if accepted:
                reference = np.mean([take.rir for take in accepted], axis=0)
                reference_peaks = accepted[0].peak_samples
                aligned, residual_alignment, correlations = align_rir_to_reference(
                    rir, reference
                )
                drift = [peaks[ch] - reference_peaks[ch] for ch in range(channel_count)]
            clipped = any(bool(item["clipped"]) for item in raw_metrics)
            xrun = bool(capture.status.get("xrun"))
            low_sweep_snr = min(sweep_snr) < repeat_cfg.minimum_sweep_snr_db
            drift_ok = not accepted or max(abs(value) for value in drift) <= repeat_cfg.peak_drift_samples
            correlation_ok = not accepted or min(correlations) >= repeat_cfg.correlation_threshold
            rejection_reasons = []
            if xrun:
                rejection_reasons.append("音频丢帧")
            if repeat_cfg.reject_clipped and clipped:
                rejection_reasons.append("削波")
            if low_sweep_snr:
                rejection_reasons.append("扫频信噪比不足")
            if not correlation_ok:
                rejection_reasons.append("重复相关性不足")
            if not drift_ok:
                rejection_reasons.append("峰值漂移超限")
            accepted_now = not rejection_reasons
            metrics = {
                "take": take_index,
                "accepted": accepted_now,
                "raw_channels": raw_metrics,
                "sweep_snr_db": sweep_snr,
                "minimum_sweep_snr_db": repeat_cfg.minimum_sweep_snr_db,
                "peak_samples": peaks,
                "peak_drift_samples": drift,
                "residual_alignment_samples": residual_alignment,
                "correlation_to_running_average": correlations,
                "backend_status": capture.status,
                "audio_xrun": xrun,
                "rejection_reasons": rejection_reasons,
            }
            store.write_audio(f"raw/take_{take_index:03d}.wav", raw, fs)
            store.write_audio(f"processed/take_{take_index:03d}_rir.wav", aligned, fs)
            store.write_json(f"metrics/take_{take_index:03d}.json", metrics)
            all_metrics.append(metrics)
            if accepted_now:
                accepted.append(RIRTake(take_index, aligned, peaks, True, metrics))
                stable_count += 1
                log(
                    f"  已接受；扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，峰值漂移={drift}"
                )
            else:
                stable_count = 0
                log(
                    f"  已拒绝：{', '.join(rejection_reasons)}；"
                    f"扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，峰值漂移={drift}"
                )

            if progress is not None:
                progress(store.root, take_index)

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
        mean_rir_files = []
        for channel in range(average.shape[1]):
            relative = f"processed/average_rir_mic_{channel + 1:02d}.wav"
            store.write_audio(relative, average[:, channel], fs)
            mean_rir_files.append(relative)
        summary = {
            "output_channel": output_channel,
            "attempted_takes": len(all_metrics),
            "accepted_takes": [take.index for take in accepted],
            "rejected_takes": [item["take"] for item in all_metrics if not item["accepted"]],
            "sample_rate": fs,
            "rir_samples": len(average),
            "mean_rir_2ch": "processed/average_rir.wav",
            "mean_rir_per_microphone": mean_rir_files,
        }
        store.write_json("metrics/summary.json", summary)
        store.finish(summary)
        log(f"脉冲响应结果已保存到：{store.root}")
        return store
    except Exception as exc:
        store.finish({"error": str(exc)}, status="failed")
        raise
