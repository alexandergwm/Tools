"""Repeated ESS room impulse response measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.fft import next_fast_len

from .audio import AudioBackend
from .config import ExperimentConfig
from .quality import channel_metrics, multichannel_health_metrics, normalized_correlation
from .signals import exponential_sweep, measurement_signal, route_outputs
from .storage import RunStore

Log = Callable[[str], None]
Progress = Callable[[Path, int], None]
StopRequested = Callable[[], bool]


@dataclass
class RIRTake:
    index: int
    rir: np.ndarray
    peak_samples: list[int]
    accepted: bool
    metrics: dict


def _moving_mean_asymmetric(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """Match MATLAB ``movmean(x, [before, after])`` endpoint behaviour."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - before)
    stops = np.minimum(len(values), indices + after + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (cumulative[stops] - cumulative[starts]) / (stops - starts)


def kirkeby_inverse_spectrum(excitation: np.ndarray, response_samples: int) -> np.ndarray:
    """Build the regularised inverse used by MATLAB ``impzest`` for ESS.

    The frequency-dependent Kirkeby regularisation suppresses inverse-filter
    gain outside the useful sweep band.  It is substantially more robust than
    a plain time-reversed sweep when the recording contains noise.
    """
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    if len(excitation) < 2 or response_samples < len(excitation):
        raise ValueError("response must contain the complete ESS excitation")
    fft_length = next_fast_len(2 * response_samples)
    if fft_length % 2:
        fft_length = next_fast_len(fft_length + 1)
    spectrum = np.fft.fft(excitation, fft_length)
    half = fft_length // 2
    frequency_index = np.arange(half + 1, dtype=np.float64)
    flattened = np.abs(spectrum[: half + 1]) * np.sqrt(frequency_index)
    flattened = np.minimum(
        _moving_mean_asymmetric(flattened, 1000, 10),
        _moving_mean_asymmetric(flattened, 10, 1000),
    )
    maximum = max(float(np.max(flattened)), np.finfo(np.float64).eps)
    regularisation = np.maximum(1e-4 * maximum, 0.38 * maximum - flattened)
    positive = np.conj(spectrum[: half + 1]) / (
        np.abs(spectrum[: half + 1]) ** 2 + regularisation
    )
    inverse = np.empty(fft_length, dtype=np.complex128)
    inverse[: half + 1] = positive
    inverse[half + 1 :] = np.conj(positive[1:half][::-1])
    return inverse


def estimate_impulse_response(
    excitation: np.ndarray,
    response: np.ndarray,
    output_samples: int,
) -> np.ndarray:
    """Estimate a multi-channel IR with MATLAB-compatible ESS deconvolution."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, None]
    if output_samples < 1:
        raise ValueError("output_samples must be positive")
    if output_samples > len(response) - len(excitation):
        raise ValueError("the response does not contain enough trailing silence")
    inverse = kirkeby_inverse_spectrum(excitation, len(response))
    fft_length = len(inverse)
    estimate = np.fft.ifft(
        np.fft.fft(response, fft_length, axis=0) * inverse[:, None], axis=0
    ).real
    estimate = np.fft.ifftshift(estimate, axes=0)
    center = fft_length // 2
    return estimate[center : center + output_samples].astype(np.float32)


def extract_rir(
    recording: np.ndarray,
    excitation: np.ndarray,
    sample_rate: int,
    pre_silence_s: float,
    post_silence_s: float,
    duration_s: float,
    pre_peak_s: float,
) -> tuple[np.ndarray, list[int], int, list[int], np.ndarray]:
    """Deconvolve and crop every microphone on one common time grid.

    Microphone 1 supplies the common direct-arrival reference.  All microphone
    channels receive the same crop, so real inter-microphone time differences
    remain present in the saved RIR.
    """
    recording = np.asarray(recording)
    if recording.ndim != 2 or recording.shape[1] < 1:
        raise ValueError("recording must have at least one microphone channel")
    pre_samples = round(pre_silence_s * sample_rate)
    response = recording[pre_samples:]
    full_samples = round(post_silence_s * sample_rate)
    full_rir = estimate_impulse_response(excitation, response, full_samples)

    search_samples = min(len(full_rir), max(1, round(0.5 * sample_rate)))
    reference_peak = int(np.argmax(np.abs(full_rir[:search_samples, 0])))
    local_radius = max(1, round(0.01 * sample_rate))
    low = max(0, reference_peak - local_radius)
    high = min(len(full_rir), reference_peak + local_radius + 1)
    peaks = [
        low + int(np.argmax(np.abs(full_rir[low:high, channel])))
        for channel in range(full_rir.shape[1])
    ]
    offsets = [peak - reference_peak for peak in peaks]

    output_samples = round(duration_s * sample_rate)
    before_peak = round(pre_peak_s * sample_rate)
    start = reference_peak - before_peak
    cropped = np.zeros((output_samples, full_rir.shape[1]), dtype=np.float32)
    source_start = max(start, 0)
    destination_start = max(-start, 0)
    count = min(len(full_rir) - source_start, output_samples - destination_start)
    if count > 0:
        cropped[destination_start : destination_start + count] = full_rir[
            source_start : source_start + count
        ]
    return cropped, peaks, reference_peak, offsets, full_rir


def align_rir_to_reference(
    rir: np.ndarray,
    reference: np.ndarray,
    max_shift_samples: int = 32,
    reference_channel: int = 0,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Align repeated takes with one common shift while preserving stereo ITD."""
    if rir.shape != reference.shape:
        raise ValueError("RIR and reference must have the same shape")
    if max_shift_samples < 0:
        raise ValueError("max_shift_samples must be non-negative")
    if not 0 <= reference_channel < rir.shape[1]:
        raise ValueError("reference_channel is outside the RIR")

    best_shift = 0
    best_correlation = normalized_correlation(
        rir[:, reference_channel], reference[:, reference_channel]
    )
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        candidate = _shift_with_zeros(rir[:, reference_channel], shift)
        correlation = normalized_correlation(candidate, reference[:, reference_channel])
        if correlation > best_correlation:
            best_shift = shift
            best_correlation = correlation
    aligned = _shift_with_zeros(rir, best_shift)
    correlations = [
        normalized_correlation(aligned[:, channel], reference[:, channel])
        for channel in range(rir.shape[1])
    ]
    return aligned, [best_shift] * rir.shape[1], correlations


def _shift_with_zeros(signal: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    if shift == 0:
        shifted[...] = signal
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


def _finalize_average(
    store: RunStore,
    accepted: list[RIRTake],
    all_metrics: list[dict],
    sample_rate: int,
    output_channel: int,
    *,
    status: str = "completed",
) -> dict:
    summary: dict = {
        "output_channel": output_channel,
        "attempted_takes": len(all_metrics),
        "accepted_takes": [take.index for take in accepted],
        "rejected_takes": [item["take"] for item in all_metrics if not item["accepted"]],
        "sample_rate": sample_rate,
        "deconvolution": "regularized_kirkeby_matlab_impzest_compatible",
        "alignment": "common_shift_from_microphone_1_preserves_inter_microphone_delay",
    }
    if accepted:
        stack = np.stack([take.rir for take in accepted])
        average = np.mean(stack, axis=0).astype(np.float32)
        median = np.median(stack, axis=0).astype(np.float32)
        store.write_audio("processed/average_rir.wav", average, sample_rate)
        store.write_audio("processed/median_rir.wav", median, sample_rate)
        mean_rir_files = []
        for channel in range(average.shape[1]):
            relative = f"processed/average_rir_mic_{channel + 1:02d}.wav"
            store.write_audio(relative, average[:, channel], sample_rate)
            mean_rir_files.append(relative)
        summary.update(
            {
                "rir_samples": len(average),
                "mean_rir_2ch": "processed/average_rir.wav",
                "mean_rir_per_microphone": mean_rir_files,
                "partial_average": status == "cancelled",
            }
        )
    store.write_json("metrics/summary.json", summary)
    store.finish(summary, status=status)
    return summary


def capture_rir(
    config: ExperimentConfig,
    backend: AudioBackend,
    output_channel: int | None = None,
    log: Log = print,
    progress: Progress | None = None,
    stop_requested: StopRequested | None = None,
) -> RunStore:
    fs, sweep_cfg, repeat_cfg = config.audio.sample_rate, config.sweep, config.repeats
    output_channel = output_channel or config.audio.target_output_channel
    sweep = exponential_sweep(
        fs,
        sweep_cfg.start_hz,
        sweep_cfg.end_hz,
        sweep_cfg.duration_s,
        sweep_cfg.level_dbfs,
        sweep_cfg.fade_in_s,
        sweep_cfg.fade_out_s,
    )
    played = measurement_signal(sweep, fs, sweep_cfg.pre_silence_s, sweep_cfg.post_silence_s)
    output = route_outputs({output_channel: played})

    store = RunStore.create(config, "rir")
    store.write_audio("references/sweep.wav", sweep, fs)
    store.write_audio("references/played.wav", played, fs)
    inverse_spectrum = kirkeby_inverse_spectrum(
        sweep, len(sweep) + round(sweep_cfg.post_silence_s * fs)
    )
    store.write_audio(
        "references/regularized_inverse_filter.wav",
        np.fft.ifft(inverse_spectrum).real.astype(np.float32),
        fs,
    )
    accepted: list[RIRTake] = []
    all_metrics: list[dict] = []
    stable_count = 0
    cancelled = False

    try:
        for take_index in range(1, repeat_cfg.maximum + 1):
            if stop_requested is not None and stop_requested():
                cancelled = True
                break
            log(f"脉冲响应采集 {take_index}/{repeat_cfg.maximum}：正在播放扫频信号")
            capture = backend.play_record(output)
            if stop_requested is not None and stop_requested():
                cancelled = True
                log("已停止：当前未完成的扫频不会加入平均")
                break
            raw = capture.microphones
            raw_metrics = channel_metrics(raw, repeat_cfg.clip_threshold)
            array_health = multichannel_health_metrics(raw)
            sweep_snr = sweep_snr_db(raw, fs, sweep_cfg.pre_silence_s, len(sweep))
            rir, peaks, reference_peak, microphone_offsets, full_rir = extract_rir(
                raw,
                sweep,
                fs,
                sweep_cfg.pre_silence_s,
                sweep_cfg.post_silence_s,
                sweep_cfg.rir_duration_s,
                sweep_cfg.pre_peak_s,
            )
            channel_count = rir.shape[1]
            correlations = [1.0] * channel_count
            residual_alignment = [0] * channel_count
            drift = 0
            aligned = rir
            if accepted:
                reference = np.mean([take.rir for take in accepted], axis=0)
                reference_peak_first = accepted[0].metrics["reference_peak_sample"]
                aligned, residual_alignment, correlations = align_rir_to_reference(rir, reference)
                drift = reference_peak - reference_peak_first
            clipped = any(bool(item["clipped"]) for item in raw_metrics)
            xrun = bool(capture.status.get("xrun"))
            low_sweep_snr = min(sweep_snr) < repeat_cfg.minimum_sweep_snr_db
            drift_ok = not accepted or abs(drift) <= repeat_cfg.peak_drift_samples
            correlation_ok = not accepted or min(correlations) >= repeat_cfg.correlation_threshold
            rejection_reasons = []
            if xrun:
                rejection_reasons.append("音频丢帧")
            if array_health["has_nonfinite_samples"]:
                rejection_reasons.append("录音包含非有限数值")
            if array_health["has_silent_channel"]:
                rejection_reasons.append("存在静音通道")
            if repeat_cfg.reject_clipped and clipped:
                rejection_reasons.append("削波")
            if low_sweep_snr:
                rejection_reasons.append("扫频信噪比不足")
            if array_health["exact_duplicate_channel_pairs"]:
                rejection_reasons.append("录制通道完全重复")
            if not correlation_ok:
                rejection_reasons.append("重复相关性不足")
            if not drift_ok:
                rejection_reasons.append("公共峰值漂移超限")
            accepted_now = not rejection_reasons
            metrics = {
                "take": take_index,
                "accepted": accepted_now,
                "raw_channels": raw_metrics,
                "array_health": array_health,
                "sweep_snr_db": sweep_snr,
                "minimum_sweep_snr_db": repeat_cfg.minimum_sweep_snr_db,
                "peak_samples": peaks,
                "reference_peak_sample": reference_peak,
                "microphone_peak_offsets_from_mic_1_samples": microphone_offsets,
                "reference_peak_drift_samples": drift,
                "residual_common_alignment_samples": residual_alignment[0],
                "correlation_to_running_average": correlations,
                "backend_status": capture.status,
                "audio_xrun": xrun,
                "rejection_reasons": rejection_reasons,
            }
            store.write_audio(f"raw/take_{take_index:03d}.wav", raw, fs)
            store.write_audio(f"processed/take_{take_index:03d}_full_ir.wav", full_rir, fs)
            store.write_audio(f"processed/take_{take_index:03d}_rir.wav", aligned, fs)
            store.write_json(f"metrics/take_{take_index:03d}.json", metrics)
            store.checkpoint()
            all_metrics.append(metrics)
            if accepted_now:
                accepted.append(RIRTake(take_index, aligned, peaks, True, metrics))
                stable_count += 1
                log(
                    f"  已接受；扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，公共峰值漂移={drift}，"
                    f"双麦峰值偏移={microphone_offsets}"
                )
            else:
                stable_count = 0
                log(
                    f"  已拒绝：{', '.join(rejection_reasons)}；"
                    f"扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，公共峰值漂移={drift}"
                )

            if progress is not None:
                progress(store.root, take_index)

            enough = len(accepted) >= repeat_cfg.minimum
            stable = stable_count >= repeat_cfg.required_stable_takes
            if enough and stable:
                log("已达到自适应重复停止条件")
                break
            if repeat_cfg.pause_s:
                if stop_requested is None:
                    time.sleep(repeat_cfg.pause_s)
                elif _interruptible_wait(repeat_cfg.pause_s, stop_requested):
                    cancelled = True
                    break

        if cancelled:
            _finalize_average(
                store, accepted, all_metrics, fs, output_channel, status="cancelled"
            )
            log(f"采集已停止；已完成的数据保存在：{store.root}")
            return store
        if len(accepted) < repeat_cfg.minimum:
            raise RuntimeError(
                f"只有 {len(accepted)} 次有效脉冲响应，最少需要 {repeat_cfg.minimum} 次"
            )
        _finalize_average(store, accepted, all_metrics, fs, output_channel)
        log(f"脉冲响应结果已保存到：{store.root}")
        return store
    except Exception as exc:
        if stop_requested is not None and stop_requested():
            _finalize_average(
                store, accepted, all_metrics, fs, output_channel, status="cancelled"
            )
            log(f"采集已停止；已完成的数据保存在：{store.root}")
            return store
        store.finish({"error": str(exc)}, status="failed")
        raise


def _interruptible_wait(seconds: float, stop_requested: StopRequested) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_requested():
            return True
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return stop_requested()
