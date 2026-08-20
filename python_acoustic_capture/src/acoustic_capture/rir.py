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
    summary: dict = {
        "output_channel": output_channel,
        "capture_strategy": repeat_config.strategy,
        "requested_fixed_attempts": repeat_config.fixed_count,
        "attempted_takes": len(all_metrics),
        "accepted_takes": [take.index for take in accepted],
        "rejected_takes": [item["take"] for item in all_metrics if not item["accepted"]],
        "sample_rate": sample_rate,
        "deconvolution": "regularized_kirkeby_matlab_impzest_compatible",
        "alignment": "common_shift_from_microphone_1_preserves_inter_microphone_delay",
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
                "mean_rir_2ch": "processed/average_rir.wav",
                "mean_rir_per_microphone": mean_rir_files,
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
                    "accepted by recording QC. No best-single or consensus selection "
                    "and no per-take peak/RMS normalization are applied."
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
                "correlation_to_running_average": correlations,
                "backend_status": capture.status,
                "audio_xrun": xrun,
                "rejection_reasons": rejection_reasons,
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
                    f"  已接受；扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，公共峰值漂移={drift}，"
                    f"双麦峰值偏移={microphone_offsets}{reconstruction_log}"
                )
            else:
                log(
                    f"  已拒绝：{', '.join(rejection_reasons)}；"
                    f"扫频信噪比={min(sweep_snr):.1f} dB，"
                    f"相关性={min(correlations):.4f}，公共峰值漂移={drift}"
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
