"""Counterexamples from the RIR audit, exercised through the real pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acoustic_capture.audio import CaptureResult, SimulatedBackend
from acoustic_capture.config import ExperimentConfig, load_config
from acoustic_capture.deconvolution import RIRWindowError, extract_rir
from acoustic_capture.rir import RIRTake, _finalize_average, capture_rir, reconstruction_metrics
from acoustic_capture.signals import exponential_sweep
from acoustic_capture.storage import RunStore


def make_config(tmp_path, count=3):
    config = ExperimentConfig()
    config.audio.backend = "simulated"
    config.sweep.duration_s = 0.5
    config.sweep.pre_silence_s = 0.05
    config.sweep.post_silence_s = 0.6
    config.sweep.rir_duration_s = 0.08
    config.sweep.level_dbfs = -30
    config.repeats.fixed_count = count
    config.repeats.pause_s = 0
    config.storage.root = str(tmp_path)
    config.storage.compute_sha256 = False
    config.validate()
    return config


class InconsistentBackend(SimulatedBackend):
    def __init__(self, config):
        super().__init__(config)
        self.take = 0

    def play_record(self, output):
        result = super().play_record(output)
        self.take += 1
        if self.take == 1:
            result.microphones[23:, 1] = result.microphones[:-23, 1].copy()
            result.microphones[:23, 1] = 0
        return result


class LateEchoBackend(SimulatedBackend):
    def _paths(self, gain):
        paths = []
        for channel in range(len(self.config.input_channels)):
            response = np.zeros(round(0.5 * self.config.sample_rate), dtype=np.float32)
            response[170 + 7 * channel] = gain
            response[170 + 7 * channel + round(0.18 * self.config.sample_rate)] = 3 * gain
            paths.append(response)
        return paths


class HighGainBackend(SimulatedBackend):
    def _paths(self, gain):
        return [10 * response for response in super()._paths(gain)]


class FixedLowpassBackend(SimulatedBackend):
    def _paths(self, gain):
        fs = self.config.sample_rate
        a = np.exp(-2 * np.pi * 1000 / fs)
        lowpass = gain * (1 - a) * a ** np.arange(fs)
        return [np.pad(lowpass, (173 + 7 * channel, 0)).astype(np.float32)
                for channel in range(len(self.config.input_channels))]


def test_two_conflicting_takes_cannot_certify_either_member(tmp_path):
    config = make_config(tmp_path, count=2)
    store = capture_rir(config, InconsistentBackend(config.audio), log=lambda _: None)
    summary = store.manifest["summary"]
    assert summary["repeat_consensus"]["similarity_matrix"][0][1] < 0.1
    assert summary["accepted_takes"] == []
    assert summary["quality"]["recommended_for_training"] is False
    assert not store.path("processed/average_rir.wav").exists()
    for take in (1, 2):
        assert store.path(f"raw/take_{take:03d}.wav").is_file()
        metrics = json.loads(store.path(f"metrics/take_{take:03d}.json").read_text("utf-8"))
        assert metrics["repeat_consistency_pass"] is False


def test_two_consistent_takes_are_still_accepted(tmp_path):
    config = make_config(tmp_path, count=2)
    store = capture_rir(config, SimulatedBackend(config.audio), log=lambda _: None)
    assert store.manifest["summary"]["accepted_takes"] == [1, 2]
    assert store.manifest["summary"]["quality"]["status"] == "pass"


def test_pcm_recording_preference_preserves_float_rir_and_inverse(tmp_path):
    config = make_config(tmp_path)
    config.storage.wav_subtype = "PCM_24"
    store = capture_rir(config, HighGainBackend(config.audio), log=lambda _: None)
    assert sf.info(store.path("raw/take_001.wav")).subtype == "PCM_24"
    for relative in (
        "processed/average_rir.wav",
        "processed/take_001_full_ir.wav",
        "references/regularized_inverse_filter.wav",
    ):
        assert sf.info(store.path(relative)).subtype == "FLOAT"
    rir, _ = sf.read(store.path("processed/average_rir.wav"), always_2d=True)
    assert np.max(np.abs(rir)) > 5.0
    # Independent readback reconvolution must describe the actual saved RIR.
    sweep, fs = sf.read(store.path("references/sweep.wav"))
    raw, _ = sf.read(store.path("raw/take_001.wav"), always_2d=True)
    metrics = json.loads(store.path("metrics/take_001.json").read_text("utf-8"))
    start = round(config.sweep.pre_silence_s * fs) + metrics["reference_peak_sample"] - round(config.sweep.pre_peak_s * fs)
    if start < 0:
        raw = np.pad(raw, ((-start, 0), (0, 0)))
        start = 0
    measured = raw[start:start + len(sweep) + len(rir) - 1]
    restored, _ = reconstruction_metrics(rir, sweep, measured)
    reported = store.manifest["summary"]["average_rir_reconstruction"]["per_take"][0]
    assert restored["minimum_correlation"] > 0.99
    assert abs(restored["worst_nmse_db"] - reported["worst_nmse_db"]) < 0.1


def test_early_rir_diagnostic_policy_states_limited_training_scope(tmp_path):
    config = make_config(tmp_path)
    store = capture_rir(config, LateEchoBackend(config.audio), log=lambda _: None)
    summary = store.manifest["summary"]
    assert summary["average_rir_reconstruction"]["minimum_correlation"] < 0.5
    assert summary["quality"]["reconstruction_policy"] == "diagnostic"
    assert summary["quality"]["training_recommendation_scope"] == "recording_and_repeat_consistency_only"


def test_full_response_policy_rejects_missing_late_echo_and_keeps_raw(tmp_path):
    config = make_config(tmp_path)
    config.repeats.reconstruction_policy = "full_response"
    with pytest.raises(RuntimeError, match="有效脉冲响应"):
        capture_rir(config, LateEchoBackend(config.audio), log=lambda _: None)
    run, = tmp_path.iterdir()
    assert (run / "raw/take_001.wav").is_file()
    metrics = json.loads((run / "metrics/take_001.json").read_text("utf-8"))
    assert metrics["accepted"] is False
    assert any("完整响应重构" in issue for issue in metrics["rejection_reasons"])


def test_final_average_must_pass_full_response_policy_even_when_each_take_fits(tmp_path):
    config = make_config(tmp_path, count=2)
    config.repeats.reconstruction_policy = "full_response"
    store = RunStore.create(config, "rir")
    excitation = np.random.default_rng(42).normal(0, 0.01, 4096)
    takes = []
    for index, gain in enumerate((1.0, 3.0), start=1):
        rir = np.zeros((1024, 1), np.float32)
        rir[100, 0] = gain
        response = np.convolve(excitation, rir[:, 0])[:, None]
        self_metrics, _ = reconstruction_metrics(rir, excitation, response)
        assert self_metrics["worst_nmse_db"] < -100
        metrics = {"take": index, "accepted": True, "rejection_reasons": [],
                   "accepted_by_recording_qc": True, "self_reconstruction": self_metrics}
        takes.append(RIRTake(index, rir, [100], True, metrics, response))
    summary = _finalize_average(store, takes, [take.metrics for take in takes],
                                48000, 1, excitation, config.repeats)
    assert summary["accepted_takes"] == [1, 2]
    assert summary["quality"]["recommended_for_training"] is False
    assert any("平均 RIR" in issue for issue in summary["quality"]["issues"])


def test_fixed_filter_timing_trend_does_not_become_a_clock_failure(tmp_path):
    config = make_config(tmp_path, count=2)
    config.sweep.duration_s = 2.0
    store = capture_rir(config, FixedLowpassBackend(config.audio), log=lambda _: None)
    summary = store.manifest["summary"]
    trends = summary["sweep_clock_drift"]["reliable_estimates"]
    assert any(abs(trend["estimated_drift_ppm"]) > 30 for trend in trends)
    assert summary["sweep_clock_drift"]["clock_drift_confirmed"] is False
    assert summary["quality"]["status"] == "pass"


def test_backend_cancel_status_is_respected_without_external_stop_callback(tmp_path):
    class CancelBackend(SimulatedBackend):
        def play_record(self, output):
            return CaptureResult(np.zeros((16, 2), np.float32), {"cancelled": True})

    config = make_config(tmp_path)
    store = capture_rir(config, CancelBackend(config.audio), log=lambda _: None)
    assert store.manifest["status"] == "cancelled"
    assert store.manifest["summary"]["attempted_takes"] == 0
    assert not store.path("processed/average_rir.wav").exists()


def recorded_impulses(fs, delays, *, pre=0.05, post=0.2):
    sweep = exponential_sweep(fs, 40, min(22000, fs / 2 - 100), 0.25, -12)
    post_samples = round(post * fs)
    columns = []
    for delay in delays:
        impulse = np.zeros(post_samples)
        impulse[delay] = 0.5
        columns.append(np.pad(np.convolve(sweep, impulse), (0, 1))[:len(sweep) + post_samples])
    recording = np.pad(np.column_stack(columns), ((round(pre * fs), 0), (0, 0)))
    return sweep, recording


def test_crop_rejects_window_that_removes_earlier_microphone():
    fs = 48000
    sweep, recording = recorded_impulses(fs, [173, 100])
    with pytest.raises(RIRWindowError) as error:
        extract_rir(recording, sweep, fs, 0.05, 0.2, 0.08, 0)
    assert error.value.details["omitted_direct_arrival_channels"] == [2]
    _, _, _, _, full = error.value.result
    assert np.argmax(np.abs(full[:, 1])) == 100


def test_crop_rejects_tail_padding_due_to_common_latency():
    fs = 8000
    sweep, recording = recorded_impulses(fs, [400, 407])
    with pytest.raises(RIRWindowError) as error:
        extract_rir(recording, sweep, fs, 0.05, 0.2, 0.21, 0.01)
    assert error.value.details["unmeasured_tail_samples"] == 400


def test_crop_failure_retains_raw_and_diagnostic_ir_in_real_pipeline(tmp_path):
    config = make_config(tmp_path, count=1)
    config.sweep.post_silence_s = 0.08
    config.sweep.rir_duration_s = 0.09
    # Equality is allowed by config, but measured common latency consumes margin.
    config.validate()
    with pytest.raises(RuntimeError, match="有效脉冲响应"):
        capture_rir(config, SimulatedBackend(config.audio), log=lambda _: None)
    run, = tmp_path.iterdir()
    metrics = json.loads((run / "metrics/take_001.json").read_text("utf-8"))
    assert metrics["crop_validation"]["valid"] is False
    assert (run / "raw/take_001.wav").is_file()
    assert (run / "processed/take_001_full_ir.wav").is_file()


def test_run_store_snapshots_configuration_and_metadata(tmp_path):
    config = make_config(tmp_path)
    config.metadata["operator"] = "initial"
    store = RunStore.create(config, "rir")
    config.audio.input_channels.reverse()
    config.metadata["operator"] = "changed"
    config.storage.wav_subtype = "PCM_16"
    assert store.config.audio.input_channels == [1, 2]
    assert store.manifest["metadata"]["operator"] == "initial"
    assert store.config.storage.wav_subtype == "FLOAT"


@pytest.mark.parametrize("dtype", ["int16", "int32", "uint8", "float64"])
def test_config_rejects_non_normalized_stream_types(tmp_path, dtype):
    config = make_config(tmp_path)
    config.audio.dtype = dtype
    with pytest.raises(ValueError, match="float32"):
        config.validate()


def test_shipped_configs_resolve_paths_within_unpacked_project():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.yaml"):
        config = load_config(path)
        assert Path(config.storage.root).is_relative_to(root)
        assert config.audio.dtype == "float32"
