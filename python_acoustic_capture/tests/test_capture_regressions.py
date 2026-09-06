"""Regression coverage for acquisition failures and editable GUI configuration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from acoustic_capture.audio import SoundDeviceBackend
from acoustic_capture.config import AudioConfig, ExperimentConfig
import acoustic_capture.gui as gui_module
from acoustic_capture.gui import CaptureGUI, FIELDS, _display, _get


class FakeDevice:
    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self, blocks, after_block=None):
        self.blocks = blocks
        self.after_block = after_block
        self.outputs = []
        self.closed = False

    def _stream(self, kind, **kwargs):
        owner = self

        class Stream:
            active = False
            latency = (0.01, 0.01) if kind == "duplex" else 0.01

            def start(self):
                self.active = True
                for count in owner.blocks:
                    channels = kwargs["channels"]
                    input_count = channels[0] if kind == "duplex" else channels
                    output_count = channels[1] if kind == "duplex" else channels
                    indata = np.tile(np.arange(1, input_count + 1) * .125, (count, 1)).astype(np.float32)
                    outdata = np.full((count, output_count), np.nan, np.float32)
                    try:
                        if kind == "duplex":
                            kwargs["callback"](indata, outdata, count, None, None)
                        elif kind == "input":
                            kwargs["callback"](indata, count, None, None)
                        else:
                            kwargs["callback"](outdata, count, None, None)
                    except (owner.CallbackStop, owner.CallbackAbort):
                        if kind != "input":
                            owner.outputs.append(outdata.copy())
                        break
                    if kind != "input":
                        owner.outputs.append(outdata.copy())
                    if owner.after_block:
                        owner.after_block()
                self.active = False

            def abort(self):
                self.active = False

            def close(self):
                owner.closed = True

        return Stream()

    def Stream(self, **kwargs):
        return self._stream("duplex", **kwargs)

    def InputStream(self, **kwargs):
        return self._stream("input", **kwargs)

    def OutputStream(self, **kwargs):
        return self._stream("output", **kwargs)


def exercise(backend, device, kind, frames=64):
    output = np.full((frames, 1), .2, np.float32)
    if kind == "duplex":
        return backend._duplex_stream(output, device)
    if kind == "input":
        return backend._input_stream(frames, device)
    return backend._output_stream(output, device)


@pytest.mark.parametrize("kind", ["duplex", "input", "output"])
def test_early_inactive_stream_is_an_error_not_zero_padded_success(kind):
    backend = SoundDeviceBackend(AudioConfig())
    updates = []
    backend.set_progress_callback(updates.append)
    device = FakeDevice([16])
    with pytest.raises(RuntimeError, match="16/64"):
        exercise(backend, device, kind)
    assert device.closed
    assert updates[-1]["phase"] == "failed"
    assert updates[-1]["frames"] == 16


@pytest.mark.parametrize("kind", ["duplex", "input", "output"])
def test_callback_exception_is_propagated_on_owner_thread(kind, monkeypatch):
    backend = SoundDeviceBackend(AudioConfig())
    original = ValueError("synthetic callback processing failure")

    def fail(*args):
        raise original

    monkeypatch.setattr(backend, "_merge_callback_status", fail)
    device = FakeDevice([16])
    with pytest.raises(RuntimeError, match="synthetic callback") as error:
        exercise(backend, device, kind)
    assert error.value.__cause__ is original
    assert device.closed
    if kind != "input":
        assert np.all(device.outputs[-1] == 0)


@pytest.mark.parametrize("kind", ["duplex", "input", "output"])
def test_cancelled_stream_reports_actual_frames_without_padding(kind):
    backend = SoundDeviceBackend(AudioConfig())
    device = FakeDevice([16], after_block=backend.stop)
    result = exercise(backend, device, kind)
    status, cancelled = result[-2:]
    assert cancelled and status["cancelled"]
    assert status["processed_frames"] == 16
    assert status["expected_frames"] == 64
    assert not status["complete"]
    if kind != "output":
        assert result[0].shape == (16, 2)
        assert np.all(result[0] != 0)


def test_partial_final_callback_completes_exact_requested_length():
    backend = SoundDeviceBackend(AudioConfig())
    device = FakeDevice([16])
    recording, status, cancelled = exercise(backend, device, "duplex", frames=10)
    assert recording.shape == (10, 2)
    assert status["complete"] and not cancelled
    assert status["processed_frames"] == 10
    assert np.all(device.outputs[-1][10:] == 0)


def test_backend_snapshot_preserves_channel_identity_after_config_edit(monkeypatch):
    config = AudioConfig(input_channels=[1, 2])
    backend = SoundDeviceBackend(config)
    device = FakeDevice([16], after_block=lambda: setattr(config, "input_channels", [2, 1]))
    monkeypatch.setattr(backend, "_module", lambda: device)
    monkeypatch.setattr(backend, "check_settings", lambda **kwargs: {})
    monkeypatch.setattr(backend, "_operation_status", lambda sd, status: status)
    # Edit occurs before play_record selects its returned microphone columns.
    config.input_channels[:] = [2, 1]
    config.sample_rate = 96000
    result = backend.play_record(np.zeros((16, 1), np.float32))
    assert result.microphones[0].tolist() == [.125, .25]
    assert backend.config.sample_rate == 48000


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint8", "float64"])
def test_integer_and_unsupported_float_stream_formats_are_rejected(dtype):
    with pytest.raises(ValueError, match="float32"):
        SoundDeviceBackend(AudioConfig(dtype=dtype))


def test_record_to_file_propagates_callback_failure(tmp_path, monkeypatch):
    backend = SoundDeviceBackend(AudioConfig())
    device = FakeDevice([16])
    monkeypatch.setattr(backend, "_module", lambda: device)
    monkeypatch.setattr(backend, "check_settings", lambda **kwargs: {})
    def fail(*args):
        raise ValueError("write callback failed")

    monkeypatch.setattr(backend, "_merge_callback_status", fail)
    with pytest.raises(RuntimeError, match="write callback failed"):
        backend.record_to_file(tmp_path / "partial.wav", stop_requested=lambda: False)
    assert device.closed


@pytest.mark.parametrize("user_stopped", [True, False])
def test_record_to_file_distinguishes_user_stop_from_host_failure(tmp_path, monkeypatch, user_stopped):
    import soundfile as sf

    backend = SoundDeviceBackend(AudioConfig())
    device = FakeDevice([16], after_block=backend.stop if user_stopped else None)
    monkeypatch.setattr(backend, "_module", lambda: device)
    monkeypatch.setattr(backend, "check_settings", lambda **kwargs: {})
    monkeypatch.setattr(backend, "_operation_status", lambda sd, status: status)
    path = tmp_path / "partial.wav"
    if user_stopped:
        result = backend.record_to_file(path, stop_requested=lambda: False)
        assert result["stopped_by_user"] and result["frames"] == 16
    else:
        with pytest.raises(RuntimeError, match="录音流意外结束"):
            backend.record_to_file(path, stop_requested=lambda: False)
    assert sf.info(path).frames == 16
    assert device.closed


class Var:
    def __init__(self, value):
        self.value = value

    def get(self, *args):
        return self.value


@pytest.mark.parametrize("checklist", [None, {"id": "test"}])
def test_gui_save_and_validate_cannot_change_pending_measurement(tmp_path, monkeypatch, checklist):
    config = ExperimentConfig()
    queued = []
    used = []
    gui = SimpleNamespace(
        checklist_row=checklist,
        config_data=config,
        config_path=tmp_path / "gui.yaml",
        variables={name: Var(_display(_get(config, name))) for name, *_ in FIELDS},
        item_vars={name: Var(True) for name in config.scene.items},
        metadata=Var("{}"),
        mode_var=Var("rir"),
        _update_metadata_summary=lambda: None,
        _append=lambda *_: None,
        _stop_event=SimpleNamespace(clear=lambda: None, is_set=lambda: False),
        viewer=SimpleNamespace(stop_audio=lambda: None, set_capture_mode=lambda **_: None),
        _set_busy=lambda _: None,
        events=SimpleNamespace(put=lambda _: None),
    )
    gui._apply_values = lambda: CaptureGUI._apply_values(gui)
    gui.save = lambda: CaptureGUI.save(gui)
    monkeypatch.setattr(gui_module.threading, "Thread", lambda *, target, **_: SimpleNamespace(start=lambda: queued.append(target)))
    monkeypatch.setattr(gui_module, "assert_capture_ready", lambda *_: SimpleNamespace(warnings=[]))
    monkeypatch.setattr(gui_module, "build_preflight_report", lambda *_: SimpleNamespace(can_start=True, warnings=[]))
    monkeypatch.setattr(gui_module, "format_preflight_report", lambda *_: "ok")
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *_: None)
    monkeypatch.setattr(gui_module, "create_backend", lambda cfg: SimpleNamespace(config=deepcopy(cfg), set_progress_callback=lambda _: None))

    def capture(snapshot, backend, **kwargs):
        used.append((snapshot, backend.config))
        return SimpleNamespace(root=tmp_path)

    monkeypatch.setattr(gui_module, "capture_rir", capture)
    CaptureGUI.run(gui, "rir")
    gui.variables["audio.input_channels"].value = "2,1"
    gui.variables["audio.sample_rate"].value = "96000"
    gui.variables["sweep.duration_s"].value = "4"
    assert gui.save()
    CaptureGUI.validate_config(gui)
    queued[0]()
    snapshot, backend_config = used[0]
    assert snapshot.audio.input_channels == backend_config.input_channels == [1, 2]
    assert snapshot.audio.sample_rate == backend_config.sample_rate == 48000
    assert snapshot.sweep.duration_s == 8
    assert config.audio.input_channels == [2, 1]
    assert config.audio.sample_rate == 96000


def test_gui_constructs_scene_estimate_before_building_widgets(monkeypatch):
    # Exercise the real constructor without creating a display or querying devices.
    methods = ("__init__", "title", "geometry", "minsize", "state", "protocol", "after")
    for name in methods:
        monkeypatch.setattr(gui_module.tk.Tk, name, lambda *args, **kwargs: None)

    class TclVar(Var):
        def __init__(self, value=None, **kwargs):
            super().__init__(value)

        def trace_add(self, *args):
            pass

    monkeypatch.setattr(gui_module.tk, "StringVar", TclVar)
    monkeypatch.setattr(gui_module.tk, "BooleanVar", TclVar)
    monkeypatch.setattr(CaptureGUI, "_build_menu", lambda _: None)
    monkeypatch.setattr(CaptureGUI, "_load_values", lambda _: None)
    observed = []
    monkeypatch.setattr(CaptureGUI, "_build", lambda self: observed.append(self.scene_estimate_var))
    app = CaptureGUI()
    assert observed == [app.scene_estimate_var]
