import numpy as np

from acoustic_capture.quality import normalized_correlation
from acoustic_capture.rir import align_rir_to_reference
from acoustic_capture.signals import exponential_sweep, route_outputs


def test_sweep_level_and_length():
    sweep = exponential_sweep(48_000, 40, 22_000, 1.0, 0.01, -12)
    assert len(sweep) == 48_000
    assert np.isclose(np.max(np.abs(sweep)), 10 ** (-12 / 20), rtol=1e-4)


def test_ess_analytic_phase_starts_at_zero_and_reaches_requested_band():
    sample_rate = 48_000
    start_hz, end_hz, duration_s = 80.0, 12_000.0, 1.0
    sweep = exponential_sweep(sample_rate, start_hz, end_hz, duration_s, 0.0, 0.0)
    assert sweep[0] == 0.0
    # Zero crossings in the final window must be much denser than at the start.
    early = np.count_nonzero(np.diff(np.signbit(sweep[:4_800])))
    late = np.count_nonzero(np.diff(np.signbit(sweep[-4_800:])))
    assert late > early * 20


def test_output_routing_is_one_based():
    signal = np.ones(10, dtype=np.float32)
    routed = route_outputs({2: signal})
    assert routed.shape == (10, 2)
    assert np.all(routed[:, 0] == 0)
    assert np.all(routed[:, 1] == 1)


def test_output_routing_can_keep_a_fixed_hardware_width():
    signal = np.ones(8, dtype=np.float32)
    routed = route_outputs({1: signal}, output_channels=4)
    assert routed.shape == (8, 4)
    assert np.allclose(routed[:, 0], signal)
    assert np.allclose(routed[:, 1:], 0)


def test_correlation():
    data = np.arange(20, dtype=np.float32)
    assert normalized_correlation(data, data) > 0.9999


def test_rir_residual_alignment_handles_each_microphone_independently():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(2048, 2)).astype(np.float32)
    shifted = np.zeros_like(reference)
    shifted[7:, 0] = reference[:-7, 0]
    shifted[:-11, 1] = reference[11:, 1]

    aligned, shifts, correlations = align_rir_to_reference(shifted, reference, 16)

    assert shifts == [-7, 11]
    assert min(correlations) > 0.99
    assert np.allclose(aligned[16:-16], reference[16:-16])
