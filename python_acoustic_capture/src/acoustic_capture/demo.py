"""Generate copyright-free signals for workflow rehearsal.

These files are deliberately synthetic. They verify routing, recording and
archiving, but must be replaced by clean speech/noise material for a real
speech-enhancement dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp


def _fade(signal: np.ndarray, sample_rate: int, duration_s: float = 0.03) -> np.ndarray:
    count = min(round(sample_rate * duration_s), len(signal) // 2)
    if count:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, count)) ** 2
        signal[:count] *= ramp
        signal[-count:] *= ramp[::-1]
    return signal


def generate_demo_audio(
    output_dir: str | Path,
    sample_rate: int = 48_000,
    duration_s: float = 4.0,
) -> dict[str, Path]:
    """Write deterministic target, interferer and basic-I/O demonstration WAVs."""
    if sample_rate < 8_000 or duration_s <= 0:
        raise ValueError("演示音频的采样率和时长必须为正数")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_count = round(sample_rate * duration_s)
    times = np.arange(sample_count, dtype=np.float64) / sample_rate

    # A speech-like harmonic signal with a slowly changing fundamental and
    # syllabic envelope. It is synthetic and contains no spoken content.
    fundamental = 150.0 + 28.0 * np.sin(2.0 * np.pi * 0.35 * times)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / sample_rate
    target = sum((1.0 / harmonic) * np.sin(harmonic * phase) for harmonic in range(1, 7))
    envelope = 0.18 + 0.82 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.2 * times)) ** 1.5
    target = _fade((target * envelope).astype(np.float64), sample_rate)

    # A distinct, clean interferer: two moving tones plus deterministic noise.
    rng = np.random.default_rng(20260809)
    interferer = (
        0.65 * chirp(times, f0=320.0, f1=2_600.0, t1=duration_s, method="linear")
        + 0.25 * np.sin(2.0 * np.pi * 730.0 * times)
        + 0.08 * rng.standard_normal(sample_count)
    )
    interferer = _fade(interferer.astype(np.float64), sample_rate)

    def normalized(signal: np.ndarray, peak: float = 0.5) -> np.ndarray:
        return (signal * peak / max(float(np.max(np.abs(signal))), 1e-12)).astype(np.float32)

    files = {
        "target": root / "target.wav",
        "interferer": root / "interferer.wav",
        "source": root / "source.wav",
    }
    target = normalized(target)
    interferer = normalized(interferer)
    sf.write(files["target"], target, sample_rate, subtype="FLOAT")
    sf.write(files["interferer"], interferer, sample_rate, subtype="FLOAT")
    sf.write(files["source"], target, sample_rate, subtype="FLOAT")
    return files
