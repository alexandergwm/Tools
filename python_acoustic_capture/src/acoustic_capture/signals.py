"""Signal generation, loading and routing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def exponential_sweep(
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    duration_s: float,
    level_dbfs: float,
    fade_in_s: float = 0.08,
    fade_out_s: float = 0.005,
) -> np.ndarray:
    """Generate an upward ESS compatible with MATLAB ``sweeptone``.

    For start frequency f0, end frequency f1 and duration T:

        f(t)   = f0 * (f1 / f0) ** (t / T)
        phi(t) = 2*pi*f0*T/log(f1/f0) * ((f1/f0)**(t/T) - 1)
        s(t)   = 10**(level_dbfs/20) * window(t) * sin(phi(t))

    MATLAB R2024b uses an 80 ms sine fade-in and a 5 ms sine fade-out for
    upward sweeps. ``measurement_signal`` additionally adds configurable
    leading and trailing silence for the physical measurement workflow.
    """
    # Compatibility with configs/tests from releases where the last two
    # positional arguments were (fade_s, level_dbfs).
    if fade_in_s < 0 <= level_dbfs:
        legacy_fade_s = level_dbfs
        level_dbfs = fade_in_s
        fade_in_s = legacy_fade_s
        fade_out_s = legacy_fade_s
    count = round(sample_rate * duration_s)
    time = np.arange(count, dtype=np.float64) / sample_rate
    log_ratio = np.log(end_hz / start_hz)
    phase = (
        2.0
        * np.pi
        * start_hz
        * duration_s
        / log_ratio
        * (np.exp(log_ratio * time / duration_s) - 1.0)
    )
    signal = np.sin(phase)
    fade_in_n = min(int(np.ceil(fade_in_s * sample_rate)), count)
    if fade_in_n:
        fade_in = np.sin(
            np.pi / 2 * np.linspace(0.0, 1.0 - 1.0 / fade_in_n, fade_in_n)
        )
        signal[:fade_in_n] *= fade_in
    fade_out_n = min(int(np.ceil(fade_out_s * sample_rate)), count)
    if fade_out_n:
        fade_out = np.sin(
            np.pi / 2 * np.linspace(1.0 - 1.0 / fade_out_n, 0.0, fade_out_n)
        )
        signal[-fade_out_n:] *= fade_out
    signal *= db_to_gain(level_dbfs)
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
