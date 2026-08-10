import numpy as np
import sys
from types import SimpleNamespace

from acoustic_capture.audio import SoundDeviceBackend, check_hardware_settings, device_choices
from acoustic_capture.config import AudioConfig


class FakeSoundDevice:
    class _Default:
        device = (4, 5)

    default = _Default()

    def __init__(self):
        self.input_checks = []
        self.output_checks = []

    def check_input_settings(self, **kwargs):
        self.input_checks.append(kwargs)

    def check_output_settings(self, **kwargs):
        self.output_checks.append(kwargs)

    def query_devices(self, device, kind=None):
        return {
            "index": 7 if kind == "input" else 8,
            "name": str(device),
            "hostapi": 2,
            "max_input_channels": 8,
            "max_output_channels": 8,
            "default_samplerate": 48_000.0,
        }

    def get_status(self):
        class Status:
            input_underflow = False
            input_overflow = True
            output_underflow = False
            output_overflow = False

            def __str__(self):
                return "input overflow"

        return Status()


def test_hardware_check_uses_highest_configured_rme_channels(monkeypatch):
    fake = FakeSoundDevice()
    monkeypatch.setattr(SoundDeviceBackend, "_module", lambda self: fake)
    config = AudioConfig(
        backend="sounddevice",
        input_device="RME 输入",
        output_device="RME 输出",
        input_channels=[1, 2],
        target_output_channel=1,
        interferer_output_channel=2,
    )
    status = check_hardware_settings(config)
    assert fake.input_checks[0]["channels"] == 2
    assert fake.output_checks[0]["channels"] == 2
    assert fake.input_checks[0]["samplerate"] == 48_000
    assert status["input_device"]["name"] == "RME 输入"
    assert status["output_device"]["name"] == "RME 输出"


def test_play_record_keeps_only_selected_input_channels(monkeypatch):
    fake = FakeSoundDevice()

    def playrec(output, *, channels, **kwargs):
        assert output.shape == (16, 2)
        assert channels == 3
        return np.column_stack(
            [np.full(16, 1.0), np.full(16, 2.0), np.full(16, 3.0)]
        ).astype(np.float32)

    fake.playrec = playrec
    monkeypatch.setattr(SoundDeviceBackend, "_module", lambda self: fake)
    config = AudioConfig(input_channels=[1, 3])
    result = SoundDeviceBackend(config).play_record(np.zeros((16, 2), dtype=np.float32))
    assert result.microphones.shape == (16, 2)
    assert np.allclose(result.microphones[:, 0], 1.0)
    assert np.allclose(result.microphones[:, 1], 3.0)
    assert result.status["xrun"] is True
    assert result.status["callback_status"]["input_overflow"] is True


def test_gui_device_choices_filter_input_and_output(monkeypatch):
    fake_module = SimpleNamespace(
        query_hostapis=lambda: [{"name": "Test API"}],
        query_devices=lambda: [
            {"name": "input only", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
            {"name": "output only", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
            {"name": "duplex", "hostapi": 0, "max_input_channels": 8, "max_output_channels": 8},
        ],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)

    inputs = device_choices("input")
    outputs = device_choices("output")

    assert any("input only" in choice for choice in inputs)
    assert not any("output only" in choice for choice in inputs)
    assert any("output only" in choice for choice in outputs)
    assert not any("input only" in choice for choice in outputs)
    assert any("duplex" in choice for choice in inputs)
    assert any("duplex" in choice for choice in outputs)
