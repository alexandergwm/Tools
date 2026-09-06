"""Regularized ESS deconvolution and measured common-window extraction."""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len

from .timing import detect_direct_arrival


class RIRWindowError(ValueError):
    """A requested crop omits a measured direct arrival or invents a late tail.

    The partial result is retained for the capture workflow to save diagnostics
    and reject the take without discarding its raw recording.
    """

    def __init__(self, issues: list[str], result: tuple, details: dict):
        super().__init__("; ".join(issues))
        self.issues = issues
        self.result = result
        self.details = details


def _moving_mean_asymmetric(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """Match MATLAB ``movmean(x, [before, after])`` endpoint behaviour."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - before)
    stops = np.minimum(len(values), indices + after + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (cumulative[stops] - cumulative[starts]) / (stops - starts)


def kirkeby_inverse_spectrum(excitation: np.ndarray, response_samples: int) -> np.ndarray:
    """Build the regularised inverse used by MATLAB R2024b ``impzest`` for ESS.

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
    """Estimate a multi-channel IR matching MATLAB R2024b ESS deconvolution."""
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
    *,
    strict: bool = True,
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

    _reference_onset, reference_peak, _arrival = detect_direct_arrival(
        full_rir[:, 0], sample_rate
    )
    peaks = [
        detect_direct_arrival(full_rir[:, channel], sample_rate)[1]
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
    result = cropped, peaks, reference_peak, offsets, full_rir
    missing_tail = max(0, start + output_samples - len(full_rir))
    omitted_channels = [
        channel + 1 for channel, peak in enumerate(peaks)
        if peak < source_start or peak >= start + output_samples
    ]
    issues = []
    if missing_tail:
        issues.append(
            f"RIR 尾部超出实测范围 {missing_tail} 样点；请增加 post_silence_s "
            "以覆盖公共播录延迟，或缩短 rir_duration_s"
        )
    if omitted_channels:
        issues.append(
            f"RIR 裁剪遗漏麦克风 {omitted_channels} 的直达声；请增加 pre_peak_s "
            "或调整共同裁剪窗口"
        )
    if strict and issues:
        raise RIRWindowError(issues, result, {
            "valid": False,
            "requested_start_sample": start,
            "requested_stop_sample": start + output_samples,
            "available_ir_samples": len(full_rir),
            "unmeasured_tail_samples": missing_tail,
            "omitted_direct_arrival_channels": omitted_channels,
        })
    return result

