import numpy as np
import pytest

from acoustic_capture.audio import CaptureResult, SimulatedBackend
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.general import capture_general_io
from acoustic_capture import storage


@pytest.mark.parametrize("stop_after_record", [False, True])
def test_record_saves_audio_before_optional_provenance(tmp_path, monkeypatch, stop_after_record):
    cfg = ExperimentConfig()
    cfg.storage.root = str(tmp_path)
    cfg.general.action = "record"
    cfg.general.duration_s = .01
    calls = []

    class Backend(SimulatedBackend):
        def record(self, frames):
            assert not calls
            calls.append("record")
            return CaptureResult(np.ones((frames, 2), np.float32) * .1, {})

    def provenance():
        assert calls[0] == "record"
        assert list(tmp_path.glob("*/raw/recording.wav"))
        calls.append("provenance")
        return "test"

    monkeypatch.setattr(storage.platform, "platform", provenance)
    monkeypatch.setattr(storage, "_git_commit", provenance)
    store = capture_general_io(cfg, Backend(cfg.audio), log=lambda _: None,
        stop_requested=lambda: stop_after_record and bool(calls))
    assert store.manifest["status"] == ("cancelled" if stop_after_record else "completed")
    assert not store.manifest["provenance_pending"]
    assert store.manifest["summary"]["preparation_seconds"] >= 0
