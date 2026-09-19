import ctypes
import json
import os
import threading

import numpy as np
import pytest
from scipy.signal import fftconvolve

from acoustic_capture.audio import _audio_com_apartment, CaptureResult, SimulatedBackend
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.rir import capture_rir, extract_rir
from acoustic_capture.rir_quality import timing_issues
from acoustic_capture.signals import exponential_sweep, measurement_signal
from acoustic_capture.timing import rir_timing_metrics


def test_gui_exposes_nested_rir_quality_issues():
    from acoustic_capture.gui import _run_quality_warnings
    assert _run_quality_warnings({"summary": {"quality": {
        "status": "review_required", "issues": ["delay mismatch"],
        "warnings": ["clock unknown"],
    }, "warnings": ["clock unknown"]}}) == ["clock unknown", "delay mismatch"]
    assert _run_quality_warnings({"summary": {"quality": {"status": "pass"}}}) == []


def test_early_reflection_does_not_replace_first_packet_and_is_flagged():
    fs = 48000
    sweep = exponential_sweep(fs, 40, 22000, .5, -18)
    played = measurement_signal(sweep, fs, .1, .3)
    responses = []
    for direct, reflection in [(173, 221), (180, 260)]:
        impulse = np.zeros(5000)
        impulse[direct], impulse[reflection] = .2, 1
        responses.append(fftconvolve(played, impulse)[:len(played)])
    _, peaks, _, offsets, full = extract_rir(
        np.column_stack(responses), sweep, fs, .1, .3, .2, .01
    )
    assert peaks == [173, 180]
    assert offsets == [0, 7]
    # GCC on a reverberant window can follow the strong reflection. Retain the
    # measurement but do not certify that correlation peak as geometric TDOA.
    timing = rir_timing_metrics(full, fs, repeat_config=ExperimentConfig().repeats)
    assert timing["per_channel"][1]["first_packet_peak_delay_samples"] == 7
    assert timing_issues(timing)


def test_default_reconstruction_rejects_repeatable_nonlinearity(tmp_path):
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.sweep.duration_s = .5
    cfg.sweep.pre_silence_s = .1
    cfg.sweep.post_silence_s = .3
    cfg.sweep.rir_duration_s = .2
    cfg.repeats.fixed_count = 2
    cfg.repeats.pause_s = 0
    cfg.storage.root = str(tmp_path)
    cfg.storage.compute_sha256 = False

    class Nonlinear(SimulatedBackend):
        def play_record(self, output):
            x = output[:, 0]
            y = x + 20 * x * x
            channels = []
            for delay, gain in [(173, .8), (180, .65)]:
                signal = np.zeros(len(y))
                signal[delay:] = gain * y[:-delay]
                channels.append(signal)
            return CaptureResult(np.column_stack(channels), {"xrun": False})

    with pytest.raises(RuntimeError, match="有效脉冲响应"):
        capture_rir(cfg, Nonlinear(cfg.audio), log=lambda _: None)
    metrics = json.loads(next(tmp_path.glob("*/metrics/take_001.json")).read_text("utf-8"))
    assert not metrics["accepted"]
    assert any("重构" in reason for reason in metrics["rejection_reasons"])
    assert next(tmp_path.glob("*/raw/take_001.wav")).is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows COM")
def test_audio_com_is_initialized_and_balanced_on_worker_even_after_exception():
    results = []
    def worker():
        dll = ctypes.WinDLL("ole32")
        def apartment():
            a, q = ctypes.c_int(), ctypes.c_int()
            return dll.CoGetApartmentType(ctypes.byref(a), ctypes.byref(q)) & 0xFFFFFFFF
        results.append(apartment())
        try:
            with _audio_com_apartment():
                results.append(apartment())
                with _audio_com_apartment():
                    results.append(apartment())
                results.append(apartment())
                raise ValueError("driver open failed")
        except ValueError:
            pass
        results.append(apartment())
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    # Other Windows libraries can establish an implicit MTA for the process.
    # Both an initially uninitialized thread and an existing apartment are valid;
    # this context must preserve that initial state and work while nested.
    assert results[0] in (0x800401F0, 0)
    assert results[1:4] == [0, 0, 0]
    assert results[-1] == results[0]
