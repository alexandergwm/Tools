import numpy as np
import pytest
import sys
from types import SimpleNamespace

from acoustic_capture.audio import (
    SoundDeviceBackend,
    check_hardware_settings,
    device_choices,
    host_api_choices,
)
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

    def query_hostapis(self, index=None):
        values = [{"name": "MME"}, {"name": "WASAPI"}, {"name": "Test API"}]
        return values if index is None else values[index]

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


def test_hardware_check_rejects_device_outside_selected_protocol(monkeypatch):
    fake = FakeSoundDevice()
    monkeypatch.setattr(SoundDeviceBackend, "_module", lambda self: fake)
    config = AudioConfig(
        backend="sounddevice",
        host_api="MME",
        input_device="RME input",
        output_device="RME output",
    )
    with pytest.raises(RuntimeError, match="不属于所选音频协议 MME"):
        check_hardware_settings(config)


def test_play_record_keeps_only_selected_input_channels(monkeypatch):
    fake = FakeSoundDevice()
    captured = {}

    class CallbackStop(Exception):
        pass

    class Stream:
        def __init__(self, *, channels, callback, **kwargs):
            assert channels == (3, 2)
            self.callback = callback
            self.active = False

        def start(self):
            indata = np.column_stack(
                [np.full(16, 1.0), np.full(16, 2.0), np.full(16, 3.0)]
            ).astype(np.float32)
            outdata = np.zeros((16, 2), dtype=np.float32)
            try:
                self.callback(indata, outdata, 16, None, None)
            except CallbackStop:
                pass
            captured["outdata"] = outdata
            self.active = False

        def abort(self):
            self.active = False

        def close(self):
            pass

    fake.Stream = Stream
    fake.CallbackStop = CallbackStop
    monkeypatch.setattr(SoundDeviceBackend, "_module", lambda self: fake)
    config = AudioConfig(input_channels=[1, 3])
    result = SoundDeviceBackend(config).play_record(np.zeros((16, 2), dtype=np.float32))
    assert result.microphones.shape == (16, 2)
    assert np.allclose(result.microphones[:, 0], 1.0)
    assert np.allclose(result.microphones[:, 1], 3.0)
    assert result.status["xrun"] is False
    assert np.allclose(captured["outdata"], 0)


def test_stop_requests_abort_without_calling_global_sounddevice_stop(monkeypatch):
    backend = SoundDeviceBackend(AudioConfig())
    calls: list[str] = []

    class Stream:
        def abort(self):
            calls.append("abort")

    backend._set_active_stream(Stream())
    backend.stop()
    deadline = __import__("time").monotonic() + 1.0
    while not calls and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert backend._abort_requested.is_set()
    assert calls == ["abort"]


def test_device_argument_converts_numeric_gui_prefix_to_portaudio_index():
    assert SoundDeviceBackend._device_argument("24: Laptop microphone") == 24
    assert SoundDeviceBackend._device_argument(" 4 ") == 4
    assert SoundDeviceBackend._device_argument("ASIO Fireface USB") == "ASIO Fireface USB"
    assert SoundDeviceBackend._device_argument(None) is None


def test_gui_device_choices_filter_input_and_output(monkeypatch):
    fake_module = SimpleNamespace(
        query_hostapis=lambda: [{"name": "MME"}, {"name": "ASIO"}],
        query_devices=lambda: [
            {"name": "input only", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
            {"name": "output only", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
            {"name": "duplex", "hostapi": 1, "max_input_channels": 8, "max_output_channels": 8},
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
    assert host_api_choices() == ["MME", "ASIO"]

    mme_inputs = device_choices("input", "MME")
    mme_outputs = device_choices("output", "MME")
    asio_inputs = device_choices("input", "ASIO")
    assert any("input only" in choice for choice in mme_inputs)
    assert not any("duplex" in choice for choice in mme_inputs)
    assert any("output only" in choice for choice in mme_outputs)
    assert not any("input only" in choice for choice in mme_outputs)
    assert len(asio_inputs) == 1 and "duplex" in asio_inputs[0]
