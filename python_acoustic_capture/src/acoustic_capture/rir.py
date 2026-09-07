"""Repeated ESS room impulse response measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

from .audio import AudioBackend
from .config import ExperimentConfig, RepeatConfig
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
    validation_response: np.ndarray | None = None


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


def detect_direct_arrival(
    impulse_response: np.ndarray,
    sample_rate: int,
    *,
    search_duration_s: float = 0.5,
    direct_peak_window_s: float = 0.0025,
    relative_threshold_db: float = -30.0,
) -> tuple[int, int, dict]:
    """Locate the first significant arrival and its nearby direct-path peak.

    The largest value in an RIR is not necessarily the direct sound.  Headset
    structures and nearby surfaces can create a stronger early reflection.
    We therefore detect the first persistent rise above both the deconvolution
    noise floor and a level relative to the strongest early response, then
    search only a short window after that onset for the direct-path peak.
    """
    values = np.asarray(impulse_response, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or len(values) == 0:
        raise ValueError("impulse response and sample rate must be non-empty and positive")
    search_samples = min(len(values), max(1, round(search_duration_s * sample_rate)))
    magnitude = np.abs(values[:search_samples])
    smoothing_samples = min(
        search_samples,
        max(1, round(0.00025 * sample_rate)),
    )
    kernel = np.full(smoothing_samples, 1.0 / smoothing_samples)
    envelope = np.sqrt(np.convolve(magnitude * magnitude, kernel, mode="same"))
    envelope_peak = max(float(np.max(envelope)), np.finfo(np.float64).eps)
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    robust_sigma = 1.4826 * mad
    noise_threshold = median + 8.0 * robust_sigma
    relative_threshold = envelope_peak * 10.0 ** (relative_threshold_db / 20.0)
    threshold = max(noise_threshold, relative_threshold)

    minimum_run = min(search_samples, max(1, round(0.0001 * sample_rate)))
    above = envelope >= threshold
    persistent = np.convolve(
        above.astype(np.int16), np.ones(minimum_run, dtype=np.int16), mode="valid"
    )
    starts = np.flatnonzero(persistent >= minimum_run)
    if len(starts):
        onset = max(0, int(starts[0]) - smoothing_samples // 2)
        method = "first_persistent_energy_rise"
    else:
        onset = int(np.argmax(magnitude))
        method = "fallback_early_global_peak"

    peak_end = min(
        search_samples,
        onset + max(1, round(direct_peak_window_s * sample_rate)),
    )
    if peak_end <= onset:
        direct_peak = onset
    else:
        direct_peak = onset + int(np.argmax(magnitude[onset:peak_end]))
    noise_floor = max(median + robust_sigma, np.finfo(np.float64).eps)
    diagnostics = {
        "method": method,
        "onset_sample": onset,
        "direct_peak_sample": direct_peak,
        "search_samples": search_samples,
        "smoothing_samples": smoothing_samples,
        "threshold": threshold,
        "relative_threshold_db": relative_threshold_db,
        "onset_confidence_db": float(
            20.0 * np.log10(max(float(envelope[direct_peak]), noise_floor) / noise_floor)
        ),
    }
    return onset, direct_peak, diagnostics


def gcc_phat_delay_samples(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    max_delay_s: float = 0.002,
    frequency_band_hz: tuple[float, float] = (300.0, 8_000.0),
) -> tuple[float, dict]:
    """Estimate target-minus-reference delay with band-limited GCC-PHAT.

    Positive delay means that ``target`` arrives later than ``reference``.
    A parabolic interpolation around the correlation maximum provides a
    fractional-sample diagnostic while the integer peak remains available in
    the returned metadata.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    count = min(len(reference), len(target))
    if sample_rate <= 0 or count < 4:
        raise ValueError("GCC-PHAT requires two non-empty signals and a positive sample rate")
    reference = reference[:count] - np.mean(reference[:count])
    target = target[:count] - np.mean(target[:count])
    window = np.hanning(count)
    fft_length = next_fast_len(max(2 * count, 32))
    reference_spectrum = np.fft.rfft(reference * window, fft_length)
    target_spectrum = np.fft.rfft(target * window, fft_length)
    cross = target_spectrum * np.conj(reference_spectrum)
    frequencies = np.fft.rfftfreq(fft_length, 1.0 / sample_rate)
    low_hz = max(0.0, float(frequency_band_hz[0]))
    high_hz = min(float(frequency_band_hz[1]), sample_rate / 2.0)
    usable = (frequencies >= low_hz) & (frequencies <= high_hz)
    phat = np.zeros_like(cross)
    phat[usable] = cross[usable] / np.maximum(np.abs(cross[usable]), 1e-15)
    correlation = np.fft.irfft(phat, fft_length)
    max_delay_samples = min(
        fft_length // 2 - 1,
        max(1, round(max_delay_s * sample_rate)),
    )
    correlation = np.concatenate(
        (correlation[-max_delay_samples:], correlation[: max_delay_samples + 1])
    )
    lags = np.arange(-max_delay_samples, max_delay_samples + 1)
    magnitudes = np.abs(correlation)
    index = int(np.argmax(magnitudes))
    integer_delay = int(lags[index])
    fractional_offset = 0.0
    if 0 < index < len(magnitudes) - 1:
        left, center, right = magnitudes[index - 1 : index + 2]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-15:
            fractional_offset = float(0.5 * (left - right) / denominator)
            fractional_offset = float(np.clip(fractional_offset, -0.5, 0.5))
    delay = float(integer_delay + fractional_offset)
    competing = magnitudes.copy()
    competing[max(0, index - 2) : min(len(competing), index + 3)] = 0.0
    second_peak = float(np.max(competing)) if len(competing) > 1 else 0.0
    diagnostics = {
        "integer_delay_samples": integer_delay,
        "fractional_delay_samples": delay,
        "delay_seconds": delay / sample_rate,
        "frequency_band_hz": [low_hz, high_hz],
        "maximum_delay_samples": max_delay_samples,
        "peak_to_second_peak_ratio": float(
            magnitudes[index] / max(second_peak, np.finfo(np.float64).eps)
        ),
    }
    return delay, diagnostics


def low_frequency_group_delay_samples(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    frequency_band_hz: tuple[float, float] = (100.0, 800.0),
) -> tuple[float | None, dict]:
    """Estimate relative low-frequency delay from cross-spectrum phase slope.

    This estimator is intentionally diagnostic: room reflections and unequal
    microphone phase responses can make a single group-delay value unreliable.
    The weighted fit quality is returned so callers can compare it with
    GCC-PHAT instead of treating it as ground truth.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    count = min(len(reference), len(target))
    if sample_rate <= 0 or count < 8:
        raise ValueError("group delay requires two non-empty signals and a positive sample rate")
    reference = reference[:count]
    target = target[:count]
    fft_length = next_fast_len(max(8 * count, 4096))
    reference_spectrum = np.fft.rfft(reference, fft_length)
    target_spectrum = np.fft.rfft(target, fft_length)
    cross = target_spectrum * np.conj(reference_spectrum)
    frequencies = np.fft.rfftfreq(fft_length, 1.0 / sample_rate)
    low_hz = max(0.0, float(frequency_band_hz[0]))
    high_hz = min(float(frequency_band_hz[1]), sample_rate / 2.0)
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_magnitude = np.abs(cross[in_band])
    if np.count_nonzero(in_band) < 3 or not np.any(band_magnitude > 0):
        return None, {
            "frequency_band_hz": [low_hz, high_hz],
            "fit_r_squared": 0.0,
            "reliable": False,
        }
    magnitude_floor = float(np.percentile(band_magnitude, 25.0))
    usable = in_band & (np.abs(cross) >= magnitude_floor)
    selected_frequencies = frequencies[usable]
    phase = np.unwrap(np.angle(cross[usable]))
    weights = np.sqrt(np.abs(cross[usable]))
    weights /= max(float(np.max(weights)), np.finfo(np.float64).eps)
    design = np.column_stack((selected_frequencies, np.ones_like(selected_frequencies)))
    weighted_design = design * weights[:, None]
    weighted_phase = phase * weights
    slope, intercept = np.linalg.lstsq(weighted_design, weighted_phase, rcond=None)[0]
    fitted = slope * selected_frequencies + intercept
    weighted_mean = float(np.average(phase, weights=np.maximum(weights, 1e-12)))
    residual_energy = float(np.sum(weights * (phase - fitted) ** 2))
    total_energy = float(np.sum(weights * (phase - weighted_mean) ** 2))
    fit_r_squared = 1.0 - residual_energy / max(total_energy, 1e-24)
    delay_samples = float(-slope * sample_rate / (2.0 * np.pi))
    reliable = bool(fit_r_squared >= 0.8 and np.isfinite(delay_samples))
    return delay_samples, {
        "frequency_band_hz": [low_hz, high_hz],
        "fit_r_squared": fit_r_squared,
        "bins_used": int(np.count_nonzero(usable)),
        "reliable": reliable,
    }


def rir_timing_metrics(
    full_rir: np.ndarray,
    sample_rate: int,
    reference_channel: int = 0,
) -> dict:
    """Return direct-arrival and inter-microphone delay diagnostics."""
    full_rir = np.asarray(full_rir, dtype=np.float64)
    if full_rir.ndim == 1:
        full_rir = full_rir[:, None]
    if not 0 <= reference_channel < full_rir.shape[1]:
        raise ValueError("reference channel is outside the RIR")
    onset, direct_peak, arrival = detect_direct_arrival(
        full_rir[:, reference_channel], sample_rate
    )
    before = max(1, round(0.002 * sample_rate))
    after = max(1, round(0.012 * sample_rate))
    start = max(0, onset - before)
    stop = min(len(full_rir), onset + after)
    direct_window = full_rir[start:stop]
    group_stop = min(len(full_rir), onset + max(after, round(0.08 * sample_rate)))
    group_window = full_rir[start:group_stop]
    per_channel = []
    speed_of_sound_m_s = 343.0
    for channel in range(full_rir.shape[1]):
        if channel == reference_channel:
            gcc_delay = 0.0
            gcc_details = {
                "integer_delay_samples": 0,
                "fractional_delay_samples": 0.0,
                "delay_seconds": 0.0,
                "frequency_band_hz": [300.0, min(8_000.0, sample_rate / 2.0)],
                "maximum_delay_samples": round(0.002 * sample_rate),
                "peak_to_second_peak_ratio": None,
            }
            group_delay = 0.0
            group_details = {
                "frequency_band_hz": [100.0, min(800.0, sample_rate / 2.0)],
                "fit_r_squared": 1.0,
                "bins_used": None,
                "reliable": True,
            }
        else:
            gcc_delay, gcc_details = gcc_phat_delay_samples(
                direct_window[:, reference_channel],
                direct_window[:, channel],
                sample_rate,
            )
            group_delay, group_details = low_frequency_group_delay_samples(
                group_window[:, reference_channel],
                group_window[:, channel],
                sample_rate,
            )
        agreement = (
            abs(float(gcc_delay) - float(group_delay))
            if group_delay is not None
            else None
        )
        per_channel.append(
            {
                "microphone_channel": channel + 1,
                "gcc_phat_delay_samples": float(gcc_delay),
                "gcc_phat_delay_microseconds": float(gcc_delay) / sample_rate * 1e6,
                "equivalent_path_difference_m": (
                    float(gcc_delay) / sample_rate * speed_of_sound_m_s
                ),
                "low_frequency_group_delay_samples": (
                    float(group_delay) if group_delay is not None else None
                ),
                "estimator_agreement_samples": agreement,
                "estimators_agree_within_one_sample": (
                    bool(agreement <= 1.0) if agreement is not None else False
                ),
                "gcc_phat": gcc_details,
                "low_frequency_group_delay": group_details,
            }
        )
    return {
        "reference_microphone_channel": reference_channel + 1,
        "reference_arrival": arrival,
        "direct_analysis_window_samples": [start, stop],
        "per_channel": per_channel,
    }


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

    _reference_onset, reference_peak, _arrival = detect_direct_arrival(
        full_rir[:, 0], sample_rate
    )
    local_radius = max(1, round(0.002 * sample_rate))
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
    comparison_samples: int | None = None,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Align repeated takes with one common shift while preserving microphone TDOA.

    Only the early part of an RIR should normally drive repeat alignment.  The
    late reverberant tail has little timing information and can make otherwise
    valid repeated measurements look dissimilar because it is dominated by
    noise.  ``comparison_samples=None`` retains the previous full-RIR API.
    """
    if rir.shape != reference.shape:
        raise ValueError("RIR and reference must have the same shape")
    if max_shift_samples < 0:
        raise ValueError("max_shift_samples must be non-negative")
    if not 0 <= reference_channel < rir.shape[1]:
        raise ValueError("reference_channel is outside the RIR")
    if comparison_samples is not None and comparison_samples < 1:
        raise ValueError("comparison_samples must be positive")

    count = (
        rir.shape[0]
        if comparison_samples is None
        else min(rir.shape[0], int(comparison_samples))
    )

    best_shift = 0
    best_correlation = normalized_correlation(
        rir[:count, reference_channel], reference[:count, reference_channel]
    )
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        candidate = _shift_with_zeros(rir[:, reference_channel], shift)
        correlation = normalized_correlation(
            candidate[:count], reference[:count, reference_channel]
        )
        if correlation > best_correlation:
            best_shift = shift
            best_correlation = correlation
    aligned = _shift_with_zeros(rir, best_shift)
    correlations = [
        normalized_correlation(aligned[:count, channel], reference[:count, channel])
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


def estimate_sweep_clock_drift_ppm(
    recording: np.ndarray,
    excitation: np.ndarray,
    sample_rate: int,
    pre_silence_s: float,
    approximate_delay_samples: int,
) -> dict:
    """Estimate relative playback/record clock drift from ESS segment timing.

    A fixed driver/acoustic latency moves every segment equally.  Independent
    playback and capture clocks instead make later sweep segments arrive
    progressively earlier or later.  This diagnostic fits that timing slope;
    it does not resample or otherwise alter the recorded data.
    """
    recording = np.asarray(recording, dtype=np.float64).reshape(-1)
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or len(excitation) < round(0.5 * sample_rate):
        return {
            "estimated_drift_ppm": None,
            "accumulated_drift_samples_over_sweep": None,
            "fit_r_squared": 0.0,
            "reliable": False,
            "reason": "sweep_too_short",
        }
    pre_samples = round(pre_silence_s * sample_rate)
    segment_samples = min(round(0.35 * sample_rate), len(excitation) // 5)
    margin_samples = max(round(0.015 * sample_rate), 64)
    observations = []
    for fraction in (0.15, 0.325, 0.5, 0.675, 0.85):
        center = round(fraction * (len(excitation) - 1))
        segment_start = max(0, min(len(excitation) - segment_samples, center - segment_samples // 2))
        segment = excitation[segment_start : segment_start + segment_samples]
        predicted = pre_samples + approximate_delay_samples + segment_start
        search_start = predicted - margin_samples
        search_stop = predicted + segment_samples + margin_samples
        if search_start < 0 or search_stop > len(recording):
            continue
        search = recording[search_start:search_stop]
        correlation = fftconvolve(search, segment[::-1], mode="valid")
        magnitude = np.abs(correlation)
        peak_index = int(np.argmax(magnitude))
        fractional_offset = 0.0
        if 0 < peak_index < len(magnitude) - 1:
            left, center_value, right = magnitude[peak_index - 1 : peak_index + 2]
            denominator = left - 2.0 * center_value + right
            if abs(denominator) > 1e-15:
                fractional_offset = float(
                    np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
                )
        observed_start = search_start + peak_index + fractional_offset
        baseline = float(np.median(magnitude))
        confidence = float(
            magnitude[peak_index] / max(baseline, np.finfo(np.float64).eps)
        )
        observations.append(
            {
                "sweep_sample": segment_start + segment_samples // 2,
                "residual_delay_samples": observed_start - predicted,
                "peak_to_median_ratio": confidence,
            }
        )
    if len(observations) < 3:
        return {
            "estimated_drift_ppm": None,
            "accumulated_drift_samples_over_sweep": None,
            "fit_r_squared": 0.0,
            "reliable": False,
            "reason": "insufficient_segments",
            "segments": observations,
        }
    positions = np.asarray([item["sweep_sample"] for item in observations], dtype=np.float64)
    delays = np.asarray(
        [item["residual_delay_samples"] for item in observations], dtype=np.float64
    )
    slope, intercept = np.polyfit(positions, delays, 1)
    fitted = slope * positions + intercept
    residual_energy = float(np.sum((delays - fitted) ** 2))
    total_energy = float(np.sum((delays - np.mean(delays)) ** 2))
    fit_r_squared = (
        1.0 - residual_energy / total_energy if total_energy > 1e-12 else 1.0
    )
    drift_ppm = float(slope * 1e6)
    accumulated = float(slope * len(excitation))
    minimum_confidence = min(item["peak_to_median_ratio"] for item in observations)
    near_constant_delay = float(np.ptp(delays)) <= 0.5
    reliable = bool(
        np.isfinite(drift_ppm)
        and (fit_r_squared >= 0.8 or near_constant_delay)
        and minimum_confidence >= 3.0
    )
    return {
        "estimated_drift_ppm": drift_ppm,
        "accumulated_drift_samples_over_sweep": accumulated,
        "fit_r_squared": fit_r_squared,
        "minimum_peak_to_median_ratio": minimum_confidence,
        "reliable": reliable,
        "sign_convention": "positive_means_later_segments_arrive_progressively_later",
        "segments": observations,
    }


def _validation_response(
    recording: np.ndarray,
    sweep_samples: int,
    rir_samples: int,
    sample_rate: int,
    pre_silence_s: float,
    pre_peak_s: float,
    reference_peak: int,
    alignment_shift: int,
) -> np.ndarray:
    """Extract the real sweep response on the cropped RIR's time grid."""
    pre_samples = round(pre_silence_s * sample_rate)
    before_peak = round(pre_peak_s * sample_rate)
    # ``alignment_shift`` was applied to the RIR. Move the real-response
    # window by the opposite amount so that both direct arrivals stay aligned.
    start = pre_samples + reference_peak - before_peak - alignment_shift
    start = max(0, min(int(start), len(recording)))
    wanted = sweep_samples + rir_samples - 1
    return np.asarray(recording[start : start + wanted], dtype=np.float32)


def normalized_reconstruction_error_db(
    rir: np.ndarray,
    excitation: np.ndarray,
    measured_responses: list[np.ndarray],
) -> float:
    """Return normalized reconvolution residual against real recordings.

    The candidate RIR is convolved with the exact ESS that was played.  A
    least-squares gain is fitted independently for every recording/channel so
    that the metric judges time/frequency structure rather than sound-card
    gain. Lower (more negative) values are better.
    """
    rir = np.asarray(rir, dtype=np.float64)
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    if rir.ndim == 1:
        rir = rir[:, None]
    if not measured_responses:
        return float("inf")
    predictions = [
        fftconvolve(excitation, rir[:, channel], mode="full")
        for channel in range(rir.shape[1])
    ]
    return _reconstruction_error_from_predictions_db(predictions, measured_responses)


def reconstruct_recording(rir: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    """Convolve the original ESS with every RIR channel."""
    rir = np.asarray(rir, dtype=np.float64)
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    if rir.ndim == 1:
        rir = rir[:, None]
    return np.column_stack(
        [
            fftconvolve(excitation, rir[:, channel], mode="full")
            for channel in range(rir.shape[1])
        ]
    ).astype(np.float32)


def reconstruction_metrics(
    rir: np.ndarray,
    excitation: np.ndarray,
    measured_response: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Compare ``ESS * RIR`` with the corresponding real microphone sweep.

    Absolute MSE/NMSE are reported without gain fitting.  A second
    scale-invariant NMSE fits one scalar gain per microphone and is included as
    a diagnostic for transfer-function shape.  The RIR itself is never
    normalized before averaging.
    """
    reconstructed = reconstruct_recording(rir, excitation)
    measured = np.asarray(measured_response, dtype=np.float64)
    if measured.ndim == 1:
        measured = measured[:, None]
    if measured.shape[1] != reconstructed.shape[1]:
        raise ValueError("RIR and measured response channel counts differ")
    count = min(len(measured), len(reconstructed))
    if count < 1:
        raise ValueError("reconstruction comparison window is empty")

    per_channel = []
    for channel in range(reconstructed.shape[1]):
        actual = measured[:count, channel]
        predicted = reconstructed[:count, channel].astype(np.float64)
        residual = actual - predicted
        mse = float(np.mean(residual * residual))
        signal_mean_square = float(np.mean(actual * actual))
        nmse = mse / max(signal_mean_square, 1e-24)
        denominator = float(np.dot(predicted, predicted))
        fitted_gain = (
            float(np.dot(actual, predicted) / denominator)
            if denominator > 1e-24
            else 0.0
        )
        fitted_residual = actual - fitted_gain * predicted
        fitted_mse = float(np.mean(fitted_residual * fitted_residual))
        fitted_nmse = fitted_mse / max(signal_mean_square, 1e-24)
        per_channel.append(
            {
                "microphone_channel": channel + 1,
                "samples_compared": count,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "nmse": nmse,
                "nmse_db": float(10.0 * np.log10(max(nmse, 1e-24))),
                "correlation": normalized_correlation(actual, predicted),
                "fitted_gain": fitted_gain,
                "scale_invariant_mse": fitted_mse,
                "scale_invariant_nmse_db": float(
                    10.0 * np.log10(max(fitted_nmse, 1e-24))
                ),
            }
        )
    return (
        {
            "comparison": "recorded_sweep_vs_original_ess_convolved_with_rir",
            "gain_fitted_for_absolute_mse": False,
            "per_channel": per_channel,
            "mean_mse": float(np.mean([item["mse"] for item in per_channel])),
            "worst_nmse_db": max(item["nmse_db"] for item in per_channel),
            "minimum_correlation": min(
                item["correlation"] for item in per_channel
            ),
            "worst_scale_invariant_nmse_db": max(
                item["scale_invariant_nmse_db"] for item in per_channel
            ),
        },
        reconstructed,
    )


def _reconstruction_error_from_predictions_db(
    predictions: list[np.ndarray], measured_responses: list[np.ndarray]
) -> float:
    """Score cached reconvolutions without repeating their FFTs."""
    if not measured_responses:
        return float("inf")
    channel_count = len(predictions)
    channel_errors: list[list[float]] = [[] for _ in range(channel_count)]
    for measured in measured_responses:
        measured = np.asarray(measured, dtype=np.float64)
        if measured.ndim == 1:
            measured = measured[:, None]
        if measured.shape[1] != channel_count:
            raise ValueError("RIR and measured response channel counts differ")
        for channel, prediction in enumerate(predictions):
            count = min(len(prediction), len(measured))
            if count < 1:
                continue
            actual = measured[:count, channel]
            predicted = prediction[:count]
            denominator = float(np.dot(predicted, predicted))
            gain = float(np.dot(actual, predicted) / denominator) if denominator > 1e-24 else 0.0
            residual = actual - gain * predicted
            measured_energy = float(np.dot(actual, actual))
            if measured_energy <= 1e-24:
                channel_errors[channel].append(float("inf"))
            else:
                ratio = np.sqrt(float(np.dot(residual, residual)) / measured_energy)
                channel_errors[channel].append(
                    float(20.0 * np.log10(max(ratio, 1e-12)))
                )
    per_channel = [
        float(np.median(values)) if values else float("inf")
        for values in channel_errors
    ]
    # One bad microphone must not be hidden by the other microphones. All
    # channels therefore share one selection, scored by the worst channel.
    return max(per_channel)


def normalized_rir_change_db(current: np.ndarray, previous: np.ndarray) -> float:
    """Scale-invariant normalized change between two aligned multi-mic RIRs."""
    current = np.asarray(current, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if current.shape != previous.shape:
        raise ValueError("RIR shapes differ")
    denominator = float(np.sum(current * current))
    gain = float(np.sum(previous * current) / denominator) if denominator > 1e-24 else 0.0
    residual = previous - gain * current
    reference_energy = float(np.sum(previous * previous))
    if reference_energy <= 1e-24:
        return float("inf")
    ratio = np.sqrt(float(np.sum(residual * residual)) / reference_energy)
    return float(20.0 * np.log10(max(ratio, 1e-12)))


def _consensus_takes(accepted: list[RIRTake], threshold: float) -> list[RIRTake]:
    if len(accepted) <= 2:
        return list(accepted)
    similarities = np.eye(len(accepted), dtype=np.float64)
    for left in range(len(accepted)):
        for right in range(left + 1, len(accepted)):
            value = min(
                normalized_correlation(
                    accepted[left].rir[:, channel], accepted[right].rir[:, channel]
                )
                for channel in range(accepted[left].rir.shape[1])
            )
            similarities[left, right] = similarities[right, left] = value
    medoid = max(
        range(len(accepted)),
        key=lambda index: (float(np.median(similarities[index])), -accepted[index].index),
    )
    members = [
        take
        for index, take in enumerate(accepted)
        if similarities[medoid, index] >= threshold
    ]
    return members or [accepted[medoid]]


def select_rir_ensemble(
    accepted: list[RIRTake],
    excitation: np.ndarray,
    correlation_threshold: float = 0.98,
) -> dict:
    """Choose best-single, aligned-mean, or consensus-mean with LOO error."""
    if not accepted:
        raise ValueError("at least one accepted RIR is required")

    excitation64 = np.asarray(excitation, dtype=np.float64).reshape(-1)
    take_by_index = {take.index: take for take in accepted}
    prediction_cache: dict[tuple[int, ...], tuple[np.ndarray, list[np.ndarray]]] = {}

    def group_model(indices: tuple[int, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
        key = tuple(sorted(indices))
        cached = prediction_cache.get(key)
        if cached is not None:
            return cached
        model = np.mean([take_by_index[index].rir for index in key], axis=0).astype(
            np.float32
        )
        predictions = [
            fftconvolve(excitation64, model[:, channel], mode="full")
            for channel in range(model.shape[1])
        ]
        prediction_cache[key] = (model, predictions)
        return model, predictions

    pairwise_errors: dict[tuple[int, int], float] = {}
    for candidate in accepted:
        _model, predictions = group_model((candidate.index,))
        for validation in accepted:
            pairwise_errors[(candidate.index, validation.index)] = (
                _reconstruction_error_from_predictions_db(
                    predictions, [validation.validation_response]
                )
            )

    def best_single(training: list[RIRTake]) -> RIRTake:
        if len(training) == 1:
            return training[0]
        scored = []
        for candidate in training:
            errors = [
                pairwise_errors[(candidate.index, validation.index)]
                for validation in training
                if validation.index != candidate.index
            ]
            scored.append((float(np.median(errors)), candidate.index, candidate))
        return min(scored, key=lambda item: (item[0], item[1]))[2]

    def score_group(indices: tuple[int, ...], validation: RIRTake) -> float:
        _model, predictions = group_model(indices)
        return _reconstruction_error_from_predictions_db(
            predictions, [validation.validation_response]
        )

    method_fold_errors: dict[str, list[float]] = {
        "best_single": [],
        "aligned_mean": [],
        "consensus_mean": [],
    }
    if len(accepted) == 1:
        score = pairwise_errors[(accepted[0].index, accepted[0].index)]
        for values in method_fold_errors.values():
            values.append(score)
    else:
        for validation in accepted:
            training = [take for take in accepted if take.index != validation.index]
            best = best_single(training)
            consensus = _consensus_takes(training, correlation_threshold)
            groups = {
                "best_single": (best.index,),
                "aligned_mean": tuple(take.index for take in training),
                "consensus_mean": tuple(take.index for take in consensus),
            }
            for method, indices in groups.items():
                method_fold_errors[method].append(score_group(indices, validation))

    method_scores = {
        method: float(np.median(values))
        for method, values in method_fold_errors.items()
    }
    best_single_take = best_single(accepted)
    consensus = _consensus_takes(accepted, correlation_threshold)
    all_indices = tuple(take.index for take in accepted)
    consensus_indices = tuple(take.index for take in consensus)
    models = {
        "best_single": (
            group_model((best_single_take.index,))[0],
            [best_single_take.index],
        ),
        "aligned_mean": (
            group_model(all_indices)[0],
            [take.index for take in accepted],
        ),
        "consensus_mean": (
            group_model(consensus_indices)[0],
            [take.index for take in consensus],
        ),
    }
    selected_method = min(
        models,
        key=lambda method: (method_scores[method], -len(models[method][1]), method),
    )
    selected_rir, selected_indices = models[selected_method]
    all_mean = models["aligned_mean"][0]
    return {
        "selection_method": selected_method,
        "selected_rir": selected_rir,
        "selected_take_indices": selected_indices,
        "selected_reconstruction_error_db": method_scores[selected_method],
        "best_single_rir": best_single_take.rir,
        "best_single_take": best_single_take.index,
        "best_single_reconstruction_error_db": method_scores["best_single"],
        "all_accepted_mean_rir": all_mean,
        "all_accepted_mean_reconstruction_error_db": method_scores["aligned_mean"],
        "consensus_take_indices": models["consensus_mean"][1],
        "consensus_mean_reconstruction_error_db": method_scores["consensus_mean"],
        "candidate_methods": [
            {
                "method": method,
                "take_indices": models[method][1],
                "leave_one_out_reconstruction_error_db": method_scores[method],
                "fold_errors_db": method_fold_errors[method],
            }
            for method in ("best_single", "aligned_mean", "consensus_mean")
        ],
    }


def _align_and_filter_repeat_consensus(
    store: RunStore,
    candidates: list[RIRTake],
    sample_rate: int,
    excitation: np.ndarray,
    repeat_config: RepeatConfig,
) -> dict:
    """Align basic-QC candidates to a medoid and reject inconsistent repeats.

    Selection is deliberately deferred until every requested attempt has been
    recorded.  This avoids anchoring an experiment to the first take.  One
    common shift is applied to every microphone channel and to the matching
    validation response, so neither inter-microphone delay nor reconstruction
    timing is destroyed.
    """
    if not candidates:
        return {
            "reference_take": None,
            "comparison_samples": 0,
            "correlation_threshold": repeat_config.correlation_threshold,
            "candidate_takes": [],
            "accepted_takes": [],
        }

    comparison_samples = min(
        candidates[0].rir.shape[0],
        max(1, round(0.12 * sample_rate)),
    )
    maximum_shift = min(
        max(32, round(0.001 * sample_rate)),
        max(0, candidates[0].rir.shape[0] - 1),
    )
    similarities = np.eye(len(candidates), dtype=np.float64)
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            _aligned, _shifts, correlations = align_rir_to_reference(
                candidates[left].rir,
                candidates[right].rir,
                maximum_shift,
                comparison_samples=comparison_samples,
            )
            similarities[left, right] = similarities[right, left] = min(correlations)

    medoid_index = max(
        range(len(candidates)),
        key=lambda index: (
            float(np.median(similarities[index])),
            -candidates[index].index,
        ),
    )
    reference = candidates[medoid_index]
    retained: list[RIRTake] = []
    for candidate in candidates:
        aligned, shifts, correlations = align_rir_to_reference(
            candidate.rir,
            reference.rir,
            maximum_shift,
            comparison_samples=comparison_samples,
        )
        common_shift = shifts[0]
        candidate.rir = aligned.astype(np.float32, copy=False)
        if candidate.validation_response is not None:
            candidate.validation_response = _shift_with_zeros(
                candidate.validation_response,
                common_shift,
            )
        consistent = (
            len(candidates) <= 2
            or candidate is reference
            or min(correlations) >= repeat_config.correlation_threshold
        )
        metrics = candidate.metrics
        metrics["accepted_by_recording_qc"] = True
        metrics["repeat_consensus_reference_take"] = reference.index
        metrics["repeat_consensus_comparison_samples"] = comparison_samples
        metrics["residual_common_alignment_samples"] = common_shift
        metrics["correlation_to_repeat_consensus"] = correlations
        metrics["repeat_consistency_pass"] = consistent
        metrics.pop("correlation_to_running_average", None)
        if not consistent:
            metrics["accepted"] = False
            metrics.setdefault("rejection_reasons", []).append(
                "早期脉冲响应与重复测量共识不一致"
            )
        else:
            metrics["accepted"] = True
            retained.append(candidate)

        if candidate.validation_response is not None:
            reconstruction, reconstructed = reconstruction_metrics(
                candidate.rir,
                excitation,
                candidate.validation_response,
            )
            metrics["self_reconstruction"] = reconstruction
            store.write_audio(
                f"processed/recon_{candidate.index:03d}.wav",
                reconstructed,
                sample_rate,
            )
        store.write_audio(
            f"processed/take_{candidate.index:03d}_rir.wav",
            candidate.rir,
            sample_rate,
        )
        store.write_json(f"metrics/take_{candidate.index:03d}.json", metrics)

    candidates[:] = retained
    return {
        "reference_take": reference.index,
        "method": "early_rir_medoid_common_shift",
        "comparison_samples": comparison_samples,
        "maximum_common_shift_samples": maximum_shift,
        "correlation_threshold": repeat_config.correlation_threshold,
        "candidate_takes": [],
        "accepted_takes": [take.index for take in retained],
        "similarity_matrix": similarities.tolist(),
    }


def _finalize_average(
    store: RunStore,
    accepted: list[RIRTake],
    all_metrics: list[dict],
    sample_rate: int,
    output_channel: int,
    excitation: np.ndarray,
    repeat_config: RepeatConfig,
    *,
    status: str = "completed",
) -> dict:
    candidate_indices = [take.index for take in accepted]
    repeat_consensus = _align_and_filter_repeat_consensus(
        store,
        accepted,
        sample_rate,
        excitation,
        repeat_config,
    )
    repeat_consensus["candidate_takes"] = candidate_indices
    reliable_clock_drifts = [
        {
            "take": item["take"],
            **item["sweep_clock_drift"],
        }
        for item in all_metrics
        if (item.get("sweep_clock_drift") or {}).get("reliable")
    ]
    backend_warnings = sorted(
        {
            str(warning)
            for item in all_metrics
            for warning in (item.get("backend_status") or {}).get("warnings", [])
        }
    )
    quality_issues = []
    quality_warnings = list(backend_warnings)
    if any("不代表共用硬件时钟" in warning for warning in backend_warnings):
        quality_issues.append(
            "录制和播放不是同一 ASIO 双工设备，无法保证输入/输出采样时钟同步"
        )
    if any("Windows/驱动可能进行重采样" in warning for warning in backend_warnings):
        quality_issues.append(
            "设备默认采样率与实验采样率不一致，Windows/驱动可能对扫频重采样"
        )
    high_clock_drift = [
        item
        for item in reliable_clock_drifts
        if abs(float(item["estimated_drift_ppm"])) >= 30.0
    ]
    if high_clock_drift:
        quality_issues.append(
            "检测到输入/输出采样时钟偏差，RIR 可能发生时间拉伸"
        )
    if len(accepted) < min(2, max(1, repeat_config.fixed_count)):
        quality_issues.append("最终只有一次有效 RIR，无法验证重复性")
    repeat_rejected = [
        item["take"]
        for item in all_metrics
        if item.get("accepted_by_recording_qc") and not item.get("accepted")
    ]
    if repeat_rejected:
        quality_warnings.append(
            "以下 take 未通过重复一致性验收："
            + ", ".join(map(str, repeat_rejected))
        )
    accepted_metric_rows = [item for item in all_metrics if item.get("accepted")]
    delay_stability = []
    if accepted_metric_rows:
        timing_rows = [
            (item.get("rir_timing") or {}).get("per_channel") or []
            for item in accepted_metric_rows
        ]
        channel_count = min((len(row) for row in timing_rows), default=0)
        for channel_index in range(1, channel_count):
            values = np.asarray(
                [
                    float(row[channel_index]["gcc_phat_delay_samples"])
                    for row in timing_rows
                ],
                dtype=np.float64,
            )
            peak_to_peak = float(np.ptp(values)) if len(values) else 0.0
            delay_stability.append(
                {
                    "microphone_channel": channel_index + 1,
                    "relative_to_microphone_channel": 1,
                    "take_count": len(values),
                    "median_gcc_phat_delay_samples": float(np.median(values)),
                    "minimum_gcc_phat_delay_samples": float(np.min(values)),
                    "maximum_gcc_phat_delay_samples": float(np.max(values)),
                    "peak_to_peak_samples": peak_to_peak,
                    "stable_within_two_samples": peak_to_peak <= 2.0,
                }
            )
            if len(values) >= 3 and peak_to_peak > 2.0:
                quality_issues.append(
                    f"麦克风 {channel_index + 1} 相对麦克风 1 的 GCC-PHAT "
                    f"延迟跨 take 波动 {peak_to_peak:.2f} 个采样点"
                )
    summary: dict = {
        "output_channel": output_channel,
        "capture_strategy": repeat_config.strategy,
        "requested_fixed_attempts": repeat_config.fixed_count,
        "attempted_takes": len(all_metrics),
        "accepted_takes": [take.index for take in accepted],
        "rejected_takes": [item["take"] for item in all_metrics if not item["accepted"]],
        "sample_rate": sample_rate,
        "deconvolution": "regularized_inverse_matched_to_matlab_r2024b_impzest",
        "matlab_compatibility": {
            "reference_release": "R2024b",
            "sweeptone_and_impzest_numeric_comparison": "passed",
            "note": (
                "The regularized result is intentionally not an amplitude-perfect "
                "least-squares inverse; this matches MATLAB impzest and suppresses "
                "out-of-band noise amplification."
            ),
        },
        "alignment": "early_rir_medoid_common_shift_preserves_inter_microphone_delay",
        "repeat_consensus": repeat_consensus,
        "sweep_clock_drift": {
            "reliable_estimates": reliable_clock_drifts,
            "warning_threshold_ppm": 30.0,
        },
        "quality": {
            "status": "pass" if not quality_issues else "review_required",
            "recommended_for_training": not quality_issues,
            "issues": quality_issues,
            "warnings": quality_warnings,
        },
        "intermicrophone_delay_stability": delay_stability,
        "offline_reselection": {
            "available": True,
            "selection_deferred": repeat_config.strategy == "fixed_count",
            "takes": [
                {
                    "take": item["take"],
                    "accepted_by_qc": item["accepted"],
                    "rejection_reasons": item.get("rejection_reasons", []),
                    "raw_recording": f"raw/take_{item['take']:03d}.wav",
                    "full_ir": f"processed/take_{item['take']:03d}_full_ir.wav",
                    "aligned_rir": f"processed/take_{item['take']:03d}_rir.wav",
                    "reconstructed_sweep": (
                        f"processed/recon_{item['take']:03d}.wav"
                        if item.get("self_reconstruction") is not None
                        else None
                    ),
                    "metrics": f"metrics/take_{item['take']:03d}.json",
                }
                for item in all_metrics
            ],
        },
    }
    if status == "cancelled":
        stop_reason = "cancelled_by_operator"
    else:
        stop_reason = "fixed_attempt_count_completed"
    summary["completion"] = {
        "stop_reason": stop_reason,
        "requested_attempts": repeat_config.fixed_count,
        "completed_attempts": len(all_metrics),
    }
    # Retained as a compatibility field for existing dataset spreadsheets.
    # This workflow no longer uses convergence or adaptive stopping.
    summary["convergence"] = {
        "converged": None,
        "stop_reason": stop_reason,
        "disabled": True,
    }
    if accepted:
        stack = np.stack([take.rir for take in accepted])
        average = np.mean(stack, axis=0).astype(np.float32)
        median = np.median(stack, axis=0).astype(np.float32)
        store.write_audio("processed/average_rir.wav", average, sample_rate)
        store.write_audio("processed/selected_rir.wav", average, sample_rate)
        store.write_audio(
            "processed/all_accepted_mean_rir.wav", average, sample_rate
        )
        store.write_audio("processed/median_rir.wav", median, sample_rate)
        average_reconstruction_per_take = []
        reconstructed_average = None
        for take in accepted:
            reconstruction, reconstructed = reconstruction_metrics(
                average, excitation, take.validation_response
            )
            average_reconstruction_per_take.append(
                {"take": take.index, **reconstruction}
            )
            if reconstructed_average is None:
                reconstructed_average = reconstructed
        if reconstructed_average is not None:
            store.write_audio(
                "processed/mean_recon.wav",
                reconstructed_average,
                sample_rate,
            )
        channel_reconstruction = []
        for channel in range(average.shape[1]):
            values = [
                item["per_channel"][channel]
                for item in average_reconstruction_per_take
            ]
            channel_reconstruction.append(
                {
                    "microphone_channel": channel + 1,
                    "median_mse": float(np.median([item["mse"] for item in values])),
                    "median_nmse_db": float(
                        np.median([item["nmse_db"] for item in values])
                    ),
                    "minimum_correlation": min(
                        item["correlation"] for item in values
                    ),
                    "worst_scale_invariant_nmse_db": max(
                        item["scale_invariant_nmse_db"] for item in values
                    ),
                }
            )
        scale_invariant_error_db = normalized_reconstruction_error_db(
            average,
            excitation,
            [take.validation_response for take in accepted],
        )
        mean_rir_files = []
        for channel in range(average.shape[1]):
            relative = f"processed/average_rir_mic_{channel + 1:02d}.wav"
            store.write_audio(relative, average[:, channel], sample_rate)
            mean_rir_files.append(relative)
        summary.update(
            {
                "rir_samples": len(average),
                "mean_rir_multichannel": "processed/average_rir.wav",
                "mean_rir_2ch": (
                    "processed/average_rir.wav" if average.shape[1] == 2 else None
                ),
                "mean_rir_per_microphone": mean_rir_files,
                "average_rir_timing": rir_timing_metrics(average, sample_rate),
                "timing_interpretation": (
                    "Inter-microphone delays are physical relative delays. The absolute "
                    "reference peak also contains playback, driver, converter and acoustic "
                    "latency and is not a propagation-time measurement without a wired "
                    "loopback/reference channel."
                ),
                "partial_average": status == "cancelled",
                "selection_method": "all_accepted_aligned_mean",
                "selected_take_ids": [take.index for take in accepted],
                "selected_reconstruction_error_db": scale_invariant_error_db,
                "all_accepted_mean_reconstruction_error_db": (
                    scale_invariant_error_db
                ),
                "average_rir_reconstruction": {
                    "comparison": (
                        "each_recorded_sweep_vs_original_ess_convolved_with_"
                        "all_accepted_aligned_mean_rir"
                    ),
                    "per_take": average_reconstruction_per_take,
                    "per_channel_summary": channel_reconstruction,
                    "worst_median_nmse_db": max(
                        item["median_nmse_db"] for item in channel_reconstruction
                    ),
                    "minimum_correlation": min(
                        item["minimum_correlation"]
                        for item in channel_reconstruction
                    ),
                },
                "selection_note": (
                    "The final RIR is the aligned arithmetic mean of every take "
                    "accepted by recording QC and repeat-consensus QC. A single common "
                    "time shift is applied to all microphones in a take; no per-channel "
                    "alignment and no per-take peak/RMS normalization are applied."
                ),
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
    cancelled = False
    attempt_limit = repeat_cfg.fixed_count
    reference_peak_baseline: int | None = None

    try:
        for take_index in range(1, attempt_limit + 1):
            if stop_requested is not None and stop_requested():
                cancelled = True
                break
            log(f"脉冲响应采集 {take_index}/{attempt_limit}：正在播放扫频信号")
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
            residual_alignment = [0] * channel_count
            aligned = rir
            drift = (
                reference_peak - reference_peak_baseline
                if reference_peak_baseline is not None
                else None
            )
            timing = rir_timing_metrics(full_rir, fs)
            clock_drift = estimate_sweep_clock_drift_ppm(
                raw[:, 0],
                sweep,
                fs,
                sweep_cfg.pre_silence_s,
                reference_peak,
            )
            clipped = any(bool(item["clipped"]) for item in raw_metrics)
            xrun = bool(capture.status.get("xrun"))
            low_sweep_snr = min(sweep_snr) < repeat_cfg.minimum_sweep_snr_db
            rejection_reasons = []
            warnings = []
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
            if drift is not None and abs(drift) > repeat_cfg.peak_drift_samples:
                warnings.append(
                    "整套播录链路的公共延迟发生变化；这不会改变麦克风间时差，"
                    "但说明当前输入/输出可能没有共用硬件时钟"
                )
            if (
                clock_drift.get("reliable")
                and abs(float(clock_drift["estimated_drift_ppm"])) >= 30.0
            ):
                warnings.append(
                    f"扫频段时序估计到约 {clock_drift['estimated_drift_ppm']:.1f} ppm "
                    "的输入/输出采样时钟偏差；本次 RIR 可能被时间拉伸，"
                    "不建议与其他 take 平均"
                )
            accepted_now = not rejection_reasons
            if accepted_now and reference_peak_baseline is None:
                reference_peak_baseline = reference_peak
                drift = 0
            validation_response = _validation_response(
                raw,
                len(sweep),
                len(aligned),
                fs,
                sweep_cfg.pre_silence_s,
                sweep_cfg.pre_peak_s,
                reference_peak,
                residual_alignment[0],
            )
            reconstruction = None
            reconstructed = None
            if not array_health["has_nonfinite_samples"]:
                reconstruction, reconstructed = reconstruction_metrics(
                    aligned, sweep, validation_response
                )
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
                "repeat_consistency_pass": None if accepted_now else False,
                "rir_timing": timing,
                "sweep_clock_drift": clock_drift,
                "backend_status": capture.status,
                "audio_xrun": xrun,
                "rejection_reasons": rejection_reasons,
                "warnings": warnings,
                "accepted_by_recording_qc": accepted_now,
                "self_reconstruction": reconstruction,
            }
            store.write_audio(f"raw/take_{take_index:03d}.wav", raw, fs)
            store.write_audio(f"processed/take_{take_index:03d}_full_ir.wav", full_rir, fs)
            store.write_audio(f"processed/take_{take_index:03d}_rir.wav", aligned, fs)
            if reconstructed is not None:
                store.write_audio(
                    f"processed/recon_{take_index:03d}.wav",
                    reconstructed,
                    fs,
                )
            all_metrics.append(metrics)
            if accepted_now:
                accepted.append(
                    RIRTake(
                        take_index,
                        aligned,
                        peaks,
                        True,
                        metrics,
                        validation_response,
                    )
                )
                reconstruction_log = (
                    f"，重构 MSE={reconstruction['mean_mse']:.3e}，"
                    f"最差 NMSE={reconstruction['worst_nmse_db']:.2f} dB，"
                    f"最低相关={reconstruction['minimum_correlation']:.4f}"
                    if reconstruction is not None
                    else ""
                )
                log(
                    f"  基础质检通过；扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"公共链路延迟漂移={drift}，双麦局部峰值偏移="
                    f"{microphone_offsets}{reconstruction_log}；"
                    "全部测量结束后再做重复一致性验收"
                )
            else:
                log(
                    f"  已拒绝：{', '.join(rejection_reasons)}；"
                    f"扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"公共链路延迟漂移={drift}"
                )

            store.write_json(f"metrics/take_{take_index:03d}.json", metrics)
            store.checkpoint()

            if progress is not None:
                progress(store.root, take_index)

            if repeat_cfg.pause_s:
                if stop_requested is None:
                    time.sleep(repeat_cfg.pause_s)
                elif _interruptible_wait(repeat_cfg.pause_s, stop_requested):
                    cancelled = True
                    break

        if cancelled:
            _finalize_average(
                store,
                accepted,
                all_metrics,
                fs,
                output_channel,
                sweep,
                repeat_cfg,
                status="cancelled",
            )
            log(f"采集已停止；已完成的数据保存在：{store.root}")
            return store
        required_accepted = 1
        if len(accepted) < required_accepted:
            raise RuntimeError(
                f"只有 {len(accepted)} 次有效脉冲响应，最少需要 {required_accepted} 次"
            )
        _finalize_average(
            store,
            accepted,
            all_metrics,
            fs,
            output_channel,
            sweep,
            repeat_cfg,
        )
        log(f"脉冲响应结果已保存到：{store.root}")
        return store
    except Exception as exc:
        if stop_requested is not None and stop_requested():
            _finalize_average(
                store,
                accepted,
                all_metrics,
                fs,
                output_channel,
                sweep,
                repeat_cfg,
                status="cancelled",
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
