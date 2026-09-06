"""Arrival, relative-delay, and acoustic sweep timing diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

def _frequency_band(frequency_band_hz: tuple[float, float], sample_rate: int) -> tuple[float, float]:
    low = max(0.0, float(frequency_band_hz[0]))
    high = min(float(frequency_band_hz[1]), sample_rate / 2.0)
    if not np.isfinite(low + high) or not 0 <= low < high:
        raise ValueError("delay frequency band must overlap positive frequencies below Nyquist")
    return low, high


def gcc_phat_delay_samples(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    max_delay_s: float = 0.002,
    frequency_band_hz: tuple[float, float] = (300.0, 8_000.0),
) -> tuple[float, dict]:
    """Estimate target-minus-reference delay; inspect ``valid`` before use.

    The float return is retained for compatibility. Invalid inputs return zero
    and an explicit invalid diagnostic, never the artificial negative search
    boundary produced by argmax of an all-zero correlation.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    count = min(len(reference), len(target))
    if sample_rate <= 0 or count < 4:
        raise ValueError("GCC-PHAT requires two non-empty signals and a positive sample rate")
    if not np.isfinite(max_delay_s) or max_delay_s <= 0:
        raise ValueError("max_delay_s must be finite and positive")
    low, high = _frequency_band(frequency_band_hz, sample_rate)
    max_delay = max(1, round(max_delay_s * sample_rate))
    details = {
        "integer_delay_samples": 0,
        "fractional_delay_samples": 0.0,
        "delay_seconds": 0.0,
        "frequency_band_hz": [low, high],
        "maximum_delay_samples": max_delay,
        "peak_to_second_peak_ratio": 0.0,
        "valid": False,
        "reliable": False,
    }
    if not np.all(np.isfinite(reference[:count])) or not np.all(np.isfinite(target[:count])):
        return 0.0, {**details, "reason": "nonfinite_samples"}
    reference = reference[:count] - np.mean(reference[:count])
    target = target[:count] - np.mean(target[:count])
    if not np.any(reference) or not np.any(target):
        return 0.0, {**details, "reason": "no_signal_energy"}
    fft_length = next_fast_len(max(2 * count, 32))
    window = np.hanning(count)
    cross = np.fft.rfft(target * window, fft_length) * np.conj(
        np.fft.rfft(reference * window, fft_length)
    )
    frequencies = np.fft.rfftfreq(fft_length, 1.0 / sample_rate)
    in_band = (frequencies >= low) & (frequencies <= high)
    magnitude = np.abs(cross)
    band_peak = float(np.max(magnitude[in_band])) if np.any(in_band) else 0.0
    if band_peak <= np.finfo(float).tiny:
        return 0.0, {**details, "reason": "no_in_band_energy"}
    usable = in_band & (magnitude > band_peak * 1e-12)
    if np.count_nonzero(usable) < 3:
        return 0.0, {**details, "reason": "insufficient_frequency_support"}
    phat = np.zeros_like(cross)
    phat[usable] = cross[usable] / magnitude[usable]
    circular = np.fft.irfft(phat, fft_length)
    # Inspect all finite-record lags before restricting the physical search.
    extent = min(count - 1, fft_length // 2 - 1)
    correlations = np.concatenate((circular[-extent:], circular[:extent + 1]))
    lags = np.arange(-extent, extent + 1)
    magnitudes = np.abs(correlations)
    global_lag = int(lags[int(np.argmax(magnitudes))])
    if abs(global_lag) > max_delay:
        return 0.0, {
            **details, "reason": "peak_outside_search_range",
            "unconstrained_peak_delay_samples": global_lag,
        }
    allowed = np.abs(lags) <= max_delay
    candidates = np.flatnonzero(allowed)
    index = int(candidates[np.argmax(magnitudes[allowed])])
    integer_delay = int(lags[index])
    fraction = 0.0
    if 0 < index < len(magnitudes) - 1:
        left, center, right = magnitudes[index - 1:index + 2]
        denominator = left - 2 * center + right
        if abs(denominator) > np.finfo(float).eps:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    delay = float(integer_delay + fraction)
    if abs(delay) > max_delay + 0.5:
        return 0.0, {**details, "reason": "peak_outside_search_range"}
    # Exclude the band-limited main lobe rather than a fixed two samples.
    main_lobe = max(2, int(np.ceil(sample_rate / (high - low))))
    competitors = magnitudes.copy()
    competitors[np.abs(lags - integer_delay) <= main_lobe] = 0.0
    second_peak = float(np.max(competitors))
    ratio = float(magnitudes[index] / max(second_peak, np.finfo(float).eps))
    reliable = bool(ratio >= 1.2)
    return delay, {
        **details,
        "integer_delay_samples": integer_delay,
        "fractional_delay_samples": delay,
        "delay_seconds": delay / sample_rate,
        "peak_to_second_peak_ratio": ratio,
        "valid": reliable,
        "reliable": reliable,
        "reason": "ok" if reliable else "ambiguous_correlation_peak",
        "at_search_boundary": abs(integer_delay) == max_delay,
    }


def low_frequency_group_delay_samples(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    frequency_band_hz: tuple[float, float] = (100.0, 800.0),
    max_delay_s: float | None = None,
    coarse_delay_samples: float | None = None,
) -> tuple[float | None, dict]:
    """Fit relative phase slope separately across contiguous reliable bands.

    Each band has its own phase intercept. Thus a removed spectral notch never
    forces an arbitrary 2-pi connection between disconnected observations.
    This is a frequency-band delay diagnostic, not geometric direct-path TDOA.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    count = min(len(reference), len(target))
    if sample_rate <= 0 or count < 8:
        raise ValueError("group delay requires two non-empty signals and a positive sample rate")
    if max_delay_s is not None and (not np.isfinite(max_delay_s) or max_delay_s <= 0):
        raise ValueError("max_delay_s must be finite and positive")
    if coarse_delay_samples is not None and not np.isfinite(coarse_delay_samples):
        raise ValueError("coarse_delay_samples must be finite")
    low, high = _frequency_band(frequency_band_hz, sample_rate)
    details = {
        "frequency_band_hz": [low, high], "fit_r_squared": 0.0,
        "bins_used": 0, "reliable": False, "valid": False,
        "phase_unwrap_method": "contiguous_bands_with_independent_intercepts",
    }
    if not np.all(np.isfinite(reference[:count])) or not np.all(np.isfinite(target[:count])):
        return None, {**details, "reason": "nonfinite_samples"}
    fft_length = next_fast_len(max(8 * count, 4096))
    cross = np.fft.rfft(target[:count], fft_length) * np.conj(
        np.fft.rfft(reference[:count], fft_length)
    )
    frequencies = np.fft.rfftfreq(fft_length, 1.0 / sample_rate)
    in_band = (frequencies >= low) & (frequencies <= high)
    magnitude = np.abs(cross)
    if np.count_nonzero(in_band) < 3 or not np.any(magnitude[in_band] > 0):
        return None, {**details, "reason": "no_in_band_energy"}
    floor = max(float(np.percentile(magnitude[in_band], 25)),
                float(np.max(magnitude[in_band])) * 1e-12)
    indices = np.flatnonzero(in_band & (magnitude >= floor) & (magnitude > 0))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    groups = [group for group in groups if len(group) >= 3]
    if not groups:
        return None, {**details, "reason": "insufficient_contiguous_frequency_support"}
    prior = 0.0 if coarse_delay_samples is None else float(coarse_delay_samples)
    residual_cross = cross * np.exp(2j * np.pi * frequencies * prior / sample_rate)
    centered_frequency, centered_phase, fit_weights = [], [], []
    usable_span = 0.0
    for group in groups:
        frequency = frequencies[group]
        phase = np.unwrap(np.angle(residual_cross[group]))
        # Weight in the squared-error objective, consistently in fit and R².
        weights = magnitude[group] / float(np.max(magnitude[in_band]))
        centered_frequency.append(frequency - np.average(frequency, weights=weights))
        centered_phase.append(phase - np.average(phase, weights=weights))
        fit_weights.append(weights)
        usable_span += float(frequency[-1] - frequency[0])
    frequency = np.concatenate(centered_frequency)
    phase = np.concatenate(centered_phase)
    weights = np.concatenate(fit_weights)
    denominator = float(np.sum(weights * frequency ** 2))
    independent_bins = usable_span / (sample_rate / count)
    details.update({"bins_used": len(frequency), "contiguous_bands": len(groups),
                    "independent_frequency_bins": independent_bins,
                    "coarse_delay_prior_samples": coarse_delay_samples})
    if denominator <= np.finfo(float).tiny or independent_bins < 2:
        return None, {**details, "reason": "insufficient_frequency_resolution"}
    slope = float(np.sum(weights * frequency * phase) / denominator)
    residual_energy = float(np.sum(weights * (phase - slope * frequency) ** 2))
    total_energy = float(np.sum(weights * phase ** 2))
    r_squared = 1.0 - residual_energy / max(total_energy, 1e-24)
    phase_rms = float(np.sqrt(residual_energy / np.sum(weights)))
    delay = float(prior - slope * sample_rate / (2 * np.pi))
    phase_variation = float(np.sqrt(total_energy / np.sum(weights)))
    reliable = bool(np.isfinite(delay) and phase_rms <= 0.35
                    and (r_squared >= 0.8 or phase_variation <= 0.05))
    reason = "ok" if reliable else "nonlinear_or_noisy_phase"
    if max_delay_s is not None and abs(delay) > max_delay_s * sample_rate + 0.5:
        reliable = False
        reason = "outside_configured_delay_range"
    return (delay if reliable else None), {
        **details, "fit_r_squared": r_squared,
        "phase_fit_residual_rms_rad": phase_rms,
        "estimated_delay_samples": delay, "valid": reliable,
        "reliable": reliable, "reason": reason,
    }


def rir_timing_metrics(
    full_rir: np.ndarray,
    sample_rate: int,
    reference_channel: int = 0,
    *,
    repeat_config=None,
) -> dict:
    """Report relative delays using configured bands, bounds and agreement.

    ``repeat_config=None`` preserves the legacy standalone frequency defaults.
    Capture callers should pass their RepeatConfig so saved settings control
    both estimators. Group delay remains a frequency-dependent diagnostic.
    """
    full_rir = np.asarray(full_rir, dtype=np.float64)
    if full_rir.ndim == 1:
        full_rir = full_rir[:, None]
    if full_rir.ndim != 2 or len(full_rir) < 8 or sample_rate <= 0:
        raise ValueError("RIR must contain at least eight samples and a positive sample rate")
    if not 0 <= reference_channel < full_rir.shape[1]:
        raise ValueError("reference channel is outside the RIR")
    if repeat_config is None:
        maximum_delay_s, agreement_limit = 0.002, 1.0
        gcc_band, group_band = (300.0, 8000.0), (100.0, 800.0)
    else:
        maximum_delay_s = float(repeat_config.delay_max_ms) / 1000.0
        agreement_limit = float(repeat_config.delay_agreement_samples)
        gcc_band = group_band = (float(repeat_config.delay_low_hz),
                                 float(repeat_config.delay_high_hz))
    # Sanitize only the onset locator; estimators must see and flag invalid data.
    onset, _peak, arrival = detect_direct_arrival(
        np.nan_to_num(full_rir[:, reference_channel], nan=0, posinf=0, neginf=0), sample_rate
    )
    # Include the full permitted negative lag plus a taper guard, so an earlier
    # microphone's arrival cannot be cut out before delay estimation even begins.
    before = max(1, round((maximum_delay_s + 0.002) * sample_rate))
    after = max(1, round(max(0.012, maximum_delay_s + 0.008) * sample_rate))
    start = max(0, onset - before)
    stop = min(len(full_rir), onset + after)
    group_stop = min(len(full_rir), onset + max(after, round(0.08 * sample_rate)))
    direct_window, group_window = full_rir[start:stop], full_rir[start:group_stop]
    per_channel = []
    for channel in range(full_rir.shape[1]):
        gcc_delay, gcc_details = gcc_phat_delay_samples(
            direct_window[:, reference_channel], direct_window[:, channel], sample_rate,
            max_delay_s=maximum_delay_s, frequency_band_hz=gcc_band,
        )
        group_delay, group_details = low_frequency_group_delay_samples(
            group_window[:, reference_channel], group_window[:, channel], sample_rate,
            frequency_band_hz=group_band, max_delay_s=maximum_delay_s,
            coarse_delay_samples=gcc_delay if gcc_details["valid"] else None,
        )
        valid = gcc_details["valid"] and group_details["valid"]
        agreement = abs(gcc_delay - group_delay) if valid and group_delay is not None else None
        per_channel.append({
            "microphone_channel": channel + 1,
            "gcc_phat_delay_samples": gcc_delay,
            "gcc_phat_delay_microseconds": gcc_delay / sample_rate * 1e6 if gcc_details["valid"] else None,
            "equivalent_path_difference_m": gcc_delay / sample_rate * 343.0 if gcc_details["valid"] else None,
            "low_frequency_group_delay_samples": group_delay,
            "estimator_agreement_samples": agreement,
            "delay_agreement_samples": agreement_limit,
            "estimators_agree": bool(agreement is not None and agreement <= agreement_limit),
            "estimators_agree_within_configured_tolerance": bool(agreement is not None and agreement <= agreement_limit),
            # Preserve the legacy field's literal meaning; configured callers
            # should use estimators_agree instead of this fixed-one-sample key.
            "estimators_agree_within_one_sample": bool(agreement is not None and agreement <= 1.0),
            "gcc_phat": gcc_details, "low_frequency_group_delay": group_details,
        })
    return {
        "reference_microphone_channel": reference_channel + 1,
        "reference_arrival": arrival,
        "direct_analysis_window_samples": [start, stop],
        "group_analysis_window_samples": [start, group_stop],
        "maximum_delay_samples": round(maximum_delay_s * sample_rate),
        "agreement_tolerance_samples": agreement_limit,
        "per_channel": per_channel,
    }


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


def _timing_trend_metadata(details: dict) -> dict:
    """Keep legacy keys, but never infer hardware clock drift from acoustics."""
    return {
        **details,
        "estimated_timing_slope_ppm": details.get("estimated_drift_ppm"),
        "timing_trend_reliable": bool(details.get("reliable", False)),
        "clock_drift_confirmed": False,
        "diagnostic_type": "acoustic_sweep_timing_trend",
        "interpretation": (
            "A fitted acoustic timing trend can arise from fixed frequency-dependent "
            "system phase as well as clock scaling. Without an independent reference "
            "it does not confirm hardware clock drift or justify rejecting a take."
        ),
    }


def estimate_sweep_clock_drift_ppm(
    recording: np.ndarray,
    excitation: np.ndarray,
    sample_rate: int,
    pre_silence_s: float,
    approximate_delay_samples: int,
) -> dict:
    """Fit an acoustic ESS timing trend, not a confirmed hardware clock error.

    Different sweep segments occupy different frequencies. A fixed system's
    phase response can therefore mimic a trend even with perfectly shared
    clocks. ``reliable`` describes the trend fit only; the compatibility key
    ``estimated_drift_ppm`` must not be used as a hardware-clock diagnosis.
    """
    recording = np.asarray(recording, dtype=np.float64).reshape(-1)
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or len(excitation) < round(0.5 * sample_rate):
        return _timing_trend_metadata({
            "estimated_drift_ppm": None,
            "accumulated_drift_samples_over_sweep": None,
            "fit_r_squared": 0.0,
            "reliable": False,
            "reason": "sweep_too_short",
        })
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
        return _timing_trend_metadata({
            "estimated_drift_ppm": None,
            "accumulated_drift_samples_over_sweep": None,
            "fit_r_squared": 0.0,
            "reliable": False,
            "reason": "insufficient_segments",
            "segments": observations,
        })
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
    return _timing_trend_metadata({
        "estimated_drift_ppm": drift_ppm,
        "accumulated_drift_samples_over_sweep": accumulated,
        "fit_r_squared": fit_r_squared,
        "minimum_peak_to_median_ratio": minimum_confidence,
        "reliable": reliable,
        "sign_convention": "positive_means_later_segments_arrive_progressively_later",
        "segments": observations,
    })



