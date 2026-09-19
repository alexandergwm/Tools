"""Independent delay oracles for the confirmed RIR audit counterexamples."""

import numpy as np
import pytest
from scipy.signal import fftconvolve, firwin

from acoustic_capture.config import RepeatConfig
from acoustic_capture.signals import exponential_sweep
from acoustic_capture.timing import (
    estimate_sweep_clock_drift_ppm,
    gcc_phat_delay_samples,
    low_frequency_group_delay_samples,
    rir_timing_metrics,
)


@pytest.mark.parametrize("delay", [-144, -120, 120, 144])
def test_configured_timing_covers_earlier_and_later_microphone(delay):
    fs = 48_000
    rir = np.zeros((8192, 2))
    rir[1000, 0], rir[1000 + delay, 1] = 1.0, 0.8
    config = RepeatConfig(delay_low_hz=100, delay_high_hz=1000,
                          delay_max_ms=3.0, delay_agreement_samples=2.0)

    timing = rir_timing_metrics(rir, fs, repeat_config=config)
    result = timing["per_channel"][1]

    assert timing["direct_analysis_window_samples"][0] < 1000 + min(delay, 0)
    assert result["gcc_phat"]["frequency_band_hz"] == [100.0, 1000.0]
    assert result["low_frequency_group_delay"]["frequency_band_hz"] == [100.0, 1000.0]
    assert result["gcc_phat"]["maximum_delay_samples"] == 144
    assert result["gcc_phat"]["valid"] is True
    assert result["gcc_phat_delay_samples"] == pytest.approx(delay, abs=0.1)
    assert result["low_frequency_group_delay_samples"] == pytest.approx(delay, abs=0.1)
    assert result["equivalent_path_difference_m"] == pytest.approx(delay / fs * 343, abs=0.001)
    assert result["estimators_agree_within_configured_tolerance"] is True
    assert result["delay_agreement_samples"] == 2.0


def test_configured_agreement_threshold_changes_decision_without_changing_estimates(monkeypatch):
    # Isolate the agreement policy from estimator accuracy. A 1.5-sample
    # disagreement is permitted by a 2-sample setting but not a 1-sample one.
    import acoustic_capture.timing as timing_module

    monkeypatch.setattr(timing_module, "gcc_phat_delay_samples", lambda *a, **k: (
        7.0, {"valid": True, "reliable": True}))
    monkeypatch.setattr(timing_module, "low_frequency_group_delay_samples", lambda *a, **k: (
        8.5, {"valid": True, "reliable": True}))
    rir = np.zeros((4096, 2))
    rir[120, :] = [1.0, 0.8]
    relaxed = rir_timing_metrics(rir, 48_000, repeat_config=RepeatConfig(delay_agreement_samples=2))
    strict = rir_timing_metrics(rir, 48_000, repeat_config=RepeatConfig(delay_agreement_samples=1))
    assert relaxed["per_channel"][1]["estimators_agree"] is True
    assert strict["per_channel"][1]["estimators_agree"] is False


def test_silence_is_invalid_and_does_not_become_a_physical_delay():
    silence = np.zeros(4096)
    delay, details = gcc_phat_delay_samples(silence, silence, 48_000)
    assert delay == 0.0
    assert details["valid"] is False
    assert details["reliable"] is False
    timing = rir_timing_metrics(np.column_stack((silence, silence)), 48_000)
    for channel in timing["per_channel"]:
        assert channel["gcc_phat"]["valid"] is False
        assert channel["low_frequency_group_delay_samples"] is None
        assert channel["equivalent_path_difference_m"] is None
        assert channel["estimators_agree"] is False


@pytest.mark.parametrize("delay", [-180, 180])
def test_out_of_range_delay_is_not_replaced_with_an_in_range_side_lobe(delay):
    reference, target = np.zeros(4096), np.zeros(4096)
    reference[1000], target[1000 + delay] = 1.0, 0.8
    estimate, details = gcc_phat_delay_samples(reference, target, 48_000, max_delay_s=0.003)
    assert estimate == 0.0
    assert details["valid"] is False
    assert details["reason"] == "peak_outside_search_range"
    group_delay, group_details = low_frequency_group_delay_samples(
        reference, target, 48_000, max_delay_s=0.003)
    assert group_delay is None
    assert group_details["reliable"] is False
    assert group_details["estimated_delay_samples"] == pytest.approx(delay, abs=1e-6)


@pytest.mark.parametrize("coarse_prior", [None, 120.0])
def test_spectral_gap_does_not_change_known_group_delay_branch(coarse_prior):
    fs, actual_delay = 48_000, 144
    h = firwin(1025, [300, 600], pass_zero="bandstop", fs=fs)
    reference = np.pad(h, (100, 1000))
    target = np.pad(reference, (actual_delay, 0))[:len(reference)]

    estimate, details = low_frequency_group_delay_samples(
        reference, target, fs, max_delay_s=0.003, coarse_delay_samples=coarse_prior)

    assert details["contiguous_bands"] >= 2
    assert details["reliable"] is True
    assert estimate == pytest.approx(actual_delay, abs=1e-5)


def test_fixed_lti_phase_does_not_confirm_hardware_clock_drift():
    fs = 48_000
    sweep = exponential_sweep(fs, 40, 22000, 2.0, -12)
    alpha = np.exp(-2 * np.pi * 1000 / fs)
    impulse_response = (1 - alpha) * alpha ** np.arange(fs)
    response = fftconvolve(sweep, impulse_response)
    recording = np.pad(response, (round(0.2 * fs) + 173, 1000))

    details = estimate_sweep_clock_drift_ppm(recording, sweep, fs, 0.2, 173)

    # The original counterexample still has a well-fitted acoustic timing
    # trend. Its interpretation must not become a hardware clock assertion.
    assert details["estimated_timing_slope_ppm"] == pytest.approx(-98.34259, abs=0.01)
    assert details["timing_trend_reliable"] is True
    assert details["clock_drift_confirmed"] is False
    assert details["diagnostic_type"] == "acoustic_sweep_timing_trend"
    assert "fixed frequency-dependent" in details["interpretation"]
