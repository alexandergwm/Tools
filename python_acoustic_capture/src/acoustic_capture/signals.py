"""Signal generation, loading and routing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp, resample_poly


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def exponential_sweep(
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    duration_s: float,
    fade_s: float,
    level_dbfs: float,
) -> np.ndarray:
    count = round(sample_rate * duration_s)
    time = np.arange(count, dtype=np.float64) / sample_rate
    signal = chirp(time, f0=start_hz, f1=end_hz, t1=duration_s, method="logarithmic")
    fade_n = min(round(fade_s * sample_rate), count // 2)
    if fade_n:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade_n, endpoint=True)) ** 2
        signal[:fade_n] *= ramp
        signal[-fade_n:] *= ramp[::-1]
    signal *= db_to_gain(level_dbfs) / max(np.max(np.abs(signal)), 1e-12)
    return signal.astype(np.float32)


def measurement_signal(sweep: np.ndarray, sample_rate: int, pre_s: float, post_s: float) -> np.ndarray:
    return np.pad(sweep, (round(pre_s * sample_rate), round(post_s * sample_rate))).astype(np.float32)


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    data, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(data, axis=1)
    if source_rate != sample_rate:
        from math import gcd

        divisor = gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 1.0:
        mono = mono / peak
    return mono.astype(np.float32)


def fit_duration(signal: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        raise ValueError("duration must contain at least one sample")
    if len(signal) == 0:
        return np.zeros(samples, dtype=np.float32)
    repeats = int(np.ceil(samples / len(signal)))
    return np.tile(signal, repeats)[:samples].astype(np.float32)


def scale_dbfs(signal: np.ndarray, level_dbfs: float) -> np.ndarray:
    peak = max(float(np.max(np.abs(signal))), 1e-12)
    return (signal * (db_to_gain(level_dbfs) / peak)).astype(np.float32)


def route_outputs(
    signals: dict[int, np.ndarray],
    sample_count: int | None = None,
    output_channels: int | None = None,
) -> np.ndarray:
    """Create a dense output matrix from one-based hardware channel numbers."""
    if not signals:
        if sample_count is None:
            raise ValueError("sample_count is required when no signal is routed")
        return np.zeros((sample_count, 1), dtype=np.float32)
    length = sample_count if sample_count is not None else max(len(x) for x in signals.values())
    required_channels = max(signals)
    channel_count = output_channels or required_channels
    if channel_count < required_channels:
        raise ValueError("output_channels cannot be smaller than the highest routed channel")
    output = np.zeros((length, channel_count), dtype=np.float32)
    for channel, signal in signals.items():
        output[: min(length, len(signal)), channel - 1] = signal[:length]
    return output
