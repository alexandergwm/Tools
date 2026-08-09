from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_capture.audio import CaptureResult, SimulatedBackend
from acoustic_capture.check import capture_input_check
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.general import capture_general_io
from acoustic_capture.rir import capture_rir
from acoustic_capture.scene import capture_scene_block


def test_rir_end_to_end(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.sweep.duration_s = 0.25
    cfg.sweep.pre_silence_s = 0.05
    cfg.sweep.post_silence_s = 0.1
    cfg.sweep.rir_duration_s = 0.1
    cfg.repeats.minimum = 2
    cfg.repeats.maximum = 3
    cfg.repeats.required_stable_takes = 1
    cfg.repeats.correlation_threshold = 0.9
    cfg.repeats.pause_s = 0
    cfg.storage.root = str(tmp_path)
    cfg.storage.compute_sha256 = False
    store = capture_rir(cfg, SimulatedBackend(cfg.audio), log=lambda _: None)
    average, fs = sf.read(store.path("processed/average_rir.wav"), always_2d=True)
    assert fs == cfg.audio.sample_rate
    assert average.shape == (round(cfg.sweep.rir_duration_s * fs), 2)
    assert store.manifest["status"] == "completed"


def test_rir_supports_more_than_two_microphones(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.audio.input_channels = [1, 2, 3, 4]
    cfg.sweep.duration_s = 0.2
    cfg.sweep.pre_silence_s = 0.05
    cfg.sweep.post_silence_s = 0.1
    cfg.sweep.rir_duration_s = 0.08
    cfg.repeats.minimum = 2
    cfg.repeats.maximum = 2
    cfg.repeats.required_stable_takes = 1
    cfg.repeats.correlation_threshold = 0.85
    cfg.repeats.pause_s = 0
    cfg.storage.root = str(tmp_path)
    cfg.storage.compute_sha256 = False
    cfg.validate()
    store = capture_rir(cfg, SimulatedBackend(cfg.audio), log=lambda _: None)
    average, _ = sf.read(store.path("processed/average_rir.wav"), always_2d=True)
    assert average.shape[1] == 4


def test_selectable_scene_block(tmp_path: Path):
    fs = 48_000
    tone = np.sin(2 * np.pi * 440 * np.arange(fs // 10) / fs).astype(np.float32)
    target, interferer = tmp_path / "target.wav", tmp_path / "interferer.wav"
    sf.write(target, tone, fs)
    sf.write(interferer, tone, fs)
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.scene.items = ["target_only", "mixture"]
    cfg.scene.duration_s = 0.1
    cfg.scene.target_file = str(target)
    cfg.scene.interferer_file = str(interferer)
    cfg.scene.gap_s = 0
    cfg.scene.countdown_s = 0
    cfg.storage.root = str(tmp_path / "runs")
    cfg.storage.compute_sha256 = False
    store = capture_scene_block(cfg, SimulatedBackend(cfg.audio), log=lambda _: None)
    assert store.path("raw/rep_001_target_only_mics.wav").is_file()
    assert store.path("raw/rep_001_mixture_mics.wav").is_file()
    assert not store.path("raw/rep_001_ambient_mics.wav").exists()
    target_playback, _ = sf.read(
        store.path("references/rep_001_target_only_playback.wav"), always_2d=True
    )
    mixture_playback, _ = sf.read(
        store.path("references/rep_001_mixture_playback.wav"), always_2d=True
    )
    assert target_playback.shape[1] == mixture_playback.shape[1] == 2
    assert np.allclose(target_playback[:, 1], 0)


def test_basic_io_actions(tmp_path: Path):
    fs = 48_000
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(fs // 20, dtype=np.float32), fs)
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.general.source_file = str(source)
    cfg.general.duration_s = 0.05
    cfg.storage.root = str(tmp_path / "runs")
    cfg.storage.compute_sha256 = False
    for action in ("play", "record", "play_record"):
        cfg.general.action = action
        store = capture_general_io(cfg, SimulatedBackend(cfg.audio), log=lambda _: None)
        assert store.manifest["status"] == "completed"
        assert store.path("references/played.wav").exists() is (action != "record")
        assert store.path("raw/recording.wav").exists() is (action != "play")


def test_basic_io_warns_when_recording_is_all_zero(tmp_path: Path):
    class SilentBackend(SimulatedBackend):
        def record(self, frames: int) -> CaptureResult:
            return CaptureResult(np.zeros((frames, 2), dtype=np.float32), {"backend": "silent-test"})

    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.general.action = "record"
    cfg.general.duration_s = 0.01
    cfg.storage.root = str(tmp_path)
    cfg.storage.compute_sha256 = False
    store = capture_general_io(cfg, SilentBackend(cfg.audio), log=lambda _: None)
    assert store.manifest["summary"]["warnings"]


def test_input_check_warns_when_all_microphones_are_zero(tmp_path: Path):
    class SilentBackend(SimulatedBackend):
        def record(self, frames: int) -> CaptureResult:
            return CaptureResult(np.zeros((frames, 2), dtype=np.float32), {"backend": "silent-test"})

    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.storage.root = str(tmp_path)
    cfg.storage.compute_sha256 = False
    store = capture_input_check(cfg, SilentBackend(cfg.audio), duration_s=0.01)
    assert store.manifest["summary"]["warnings"]
