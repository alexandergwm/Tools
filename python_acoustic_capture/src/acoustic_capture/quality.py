"""Metrics shared by RIR and speech capture."""

from __future__ import annotations

import numpy as np


def channel_metrics(signal: np.ndarray, clip_threshold: float = 0.999) -> list[dict[str, float | int | bool]]:
    signal = np.atleast_2d(signal) if signal.ndim == 1 else signal
    result = []
    for channel in range(signal.shape[1]):
        x = signal[:, channel].astype(np.float64)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        result.append(
            {
                "channel": channel + 1,
                "peak": peak,
                "peak_dbfs": float(20 * np.log10(max(peak, 1e-12))),
                "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
                "clipped_samples": int(np.count_nonzero(np.abs(x) >= clip_threshold)),
                "clipped": bool(peak >= clip_threshold),
            }
        )
    return result


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0

