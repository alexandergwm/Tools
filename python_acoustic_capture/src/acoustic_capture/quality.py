"""Metrics shared by RIR and speech capture."""

from __future__ import annotations

import numpy as np


def channel_metrics(signal: np.ndarray, clip_threshold: float = 0.999) -> list[dict[str, float | int | bool]]:
    signal = np.atleast_2d(signal) if signal.ndim == 1 else signal
    result = []
    for channel in range(signal.shape[1]):
        x = signal[:, channel].astype(np.float64)
        finite = np.isfinite(x)
        safe = np.where(finite, x, 0.0)
        peak = float(np.max(np.abs(safe))) if len(safe) else 0.0
        rms = float(np.sqrt(np.mean(safe * safe))) if len(safe) else 0.0
        result.append(
            {
                "channel": channel + 1,
                "peak": peak,
                "peak_dbfs": float(20 * np.log10(max(peak, 1e-12))),
                "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
                "clipped_samples": int(np.count_nonzero(np.abs(x) >= clip_threshold)),
                "clipped": bool(peak >= clip_threshold),
                "nonfinite_samples": int(np.count_nonzero(~finite)),
                "zero_fraction": float(np.mean(safe == 0.0)) if len(safe) else 1.0,
                "dc_offset": float(np.mean(safe)) if len(safe) else 0.0,
            }
        )
    return result


def multichannel_health_metrics(signal: np.ndarray) -> dict[str, object]:
    """Detect obvious channel-routing failures without judging real acoustics."""
    value = np.asarray(signal)
    value = value[:, None] if value.ndim == 1 else value
    if value.ndim != 2:
        raise ValueError("signal must be a 1-D or 2-D array")
    channels = channel_metrics(value)
    duplicate_pairs: list[list[int]] = []
    for first in range(value.shape[1]):
        for second in range(first + 1, value.shape[1]):
            if np.array_equal(value[:, first], value[:, second]):
                duplicate_pairs.append([first + 1, second + 1])
    rms_values = [float(item["rms_dbfs"]) for item in channels]
    return {
        "channel_count": value.shape[1],
        "exact_duplicate_channel_pairs": duplicate_pairs,
        "rms_spread_db": max(rms_values) - min(rms_values) if rms_values else 0.0,
        "has_nonfinite_samples": any(
            int(item["nonfinite_samples"]) > 0 for item in channels
        ),
        "has_silent_channel": any(float(item["peak"]) <= 1e-12 for item in channels),
    }


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def mixture_additivity_metrics(
    target_only: np.ndarray,
    interferer_only: np.ndarray,
    mixture: np.ndarray,
) -> dict[str, object]:
    """Measure how closely a real mixture follows target + interferer.

    Sequential acoustic recordings are not expected to add exactly because of
    ambient noise and small physical changes.  The metric is therefore a QC
    signal rather than an assertion that the target-only recording is the
    mathematical clean component inside the mixture.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in (target_only, interferer_only, mixture)]
    arrays = [value[:, None] if value.ndim == 1 else value for value in arrays]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("target_only, interferer_only and mixture must be 1-D or 2-D arrays")
    shapes = {value.shape for value in arrays}
    if len(shapes) != 1:
        raise ValueError(f"supervised recordings must have equal shapes, got {sorted(shapes)}")

    target, interferer, mixed = arrays
    channels: list[dict[str, float | int]] = []
    eps = 1e-12
    for index in range(mixed.shape[1]):
        target_ch = target[:, index]
        interferer_ch = interferer[:, index]
        mixture_ch = mixed[:, index]
        predicted = target_ch + interferer_ch
        residual = mixture_ch - predicted
        target_rms = float(np.sqrt(np.mean(target_ch * target_ch))) if len(target_ch) else 0.0
        interferer_rms = (
            float(np.sqrt(np.mean(interferer_ch * interferer_ch))) if len(interferer_ch) else 0.0
        )
        mixture_rms = float(np.sqrt(np.mean(mixture_ch * mixture_ch))) if len(mixture_ch) else 0.0
        residual_rms = float(np.sqrt(np.mean(residual * residual))) if len(residual) else 0.0
        channels.append(
            {
                "channel": index + 1,
                "mixture_consistency_residual_db": float(
                    20 * np.log10(max(residual_rms, eps) / max(mixture_rms, eps))
                ),
                "mixture_consistency_correlation": normalized_correlation(
                    mixture_ch, predicted
                ),
                "estimated_sir_db": float(
                    20 * np.log10(max(target_rms, eps) / max(interferer_rms, eps))
                ),
                "target_rms_dbfs": float(20 * np.log10(max(target_rms, eps))),
                "interferer_rms_dbfs": float(20 * np.log10(max(interferer_rms, eps))),
                "mixture_rms_dbfs": float(20 * np.log10(max(mixture_rms, eps))),
            }
        )
    return {
        "definition": "mixture_recording - (target_only_recording + interferer_only_recording)",
        "interpretation": "quality-control only; real sequential acoustic captures are not exact source images",
        "channels": channels,
        "residual_db_max": max(
            (float(item["mixture_consistency_residual_db"]) for item in channels),
            default=0.0,
        ),
        "correlation_min": min(
            (float(item["mixture_consistency_correlation"]) for item in channels),
            default=0.0,
        ),
    }


def evaluate_supervision_quality_gate(
    metrics: dict[str, object], metadata: dict[str, object]
) -> dict[str, object]:
    """Evaluate optional, explicit production thresholds for additivity QC."""
    raw = metadata.get("quality_gate")
    settings = raw if isinstance(raw, dict) else {}
    enabled = bool(settings.get("enabled", False))
    max_residual_db = float(settings.get("max_additivity_residual_db", -6.0))
    min_correlation = float(settings.get("min_additivity_correlation", 0.70))
    reasons: list[str] = []
    if enabled:
        residual = float(metrics.get("residual_db_max", 0.0))
        correlation = float(metrics.get("correlation_min", 0.0))
        if residual > max_residual_db:
            reasons.append(
                f"residual_db_max={residual:.2f} > {max_residual_db:.2f}"
            )
        if correlation < min_correlation:
            reasons.append(
                f"correlation_min={correlation:.3f} < {min_correlation:.3f}"
            )
    return {
        "enabled": enabled,
        "passed": not reasons,
        "max_additivity_residual_db": max_residual_db,
        "min_additivity_correlation": min_correlation,
        "reasons": reasons,
    }
