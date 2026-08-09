from pathlib import Path

import soundfile as sf

from acoustic_capture.demo import generate_demo_audio


def test_generate_demo_audio_writes_three_distinct_signals(tmp_path: Path):
    files = generate_demo_audio(tmp_path, sample_rate=16_000, duration_s=0.2)
    assert set(files) == {"target", "interferer", "source"}
    target, sample_rate = sf.read(files["target"])
    interferer, interferer_rate = sf.read(files["interferer"])
    assert sample_rate == interferer_rate == 16_000
    assert len(target) == len(interferer) == 3_200
    assert not (target == interferer).all()
