import numpy as np

from acoustic_capture.quality import normalized_correlation
from acoustic_capture.signals import exponential_sweep, route_outputs


def test_sweep_level_and_length():
    sweep = exponential_sweep(48_000, 40, 22_000, 1.0, 0.01, -12)
    assert len(sweep) == 48_000
    assert np.isclose(np.max(np.abs(sweep)), 10 ** (-12 / 20), rtol=1e-4)


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
