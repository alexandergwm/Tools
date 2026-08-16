import numpy as np

from acoustic_capture.quality import normalized_correlation
from acoustic_capture.rir import (
    RIRTake,
    align_rir_to_reference,
    estimate_impulse_response,
    normalized_reconstruction_error_db,
    normalized_rir_change_db,
    select_rir_ensemble,
)
from acoustic_capture.signals import exponential_sweep, route_outputs


def test_sweep_level_and_length():
    sweep = exponential_sweep(48_000, 40, 22_000, 1.0, -12, 0.01, 0.01)
    assert len(sweep) == 48_000
    assert np.isclose(np.max(np.abs(sweep)), 10 ** (-12 / 20), rtol=1e-4)


def test_ess_analytic_phase_starts_at_zero_and_reaches_requested_band():
    sample_rate = 48_000
    start_hz, end_hz, duration_s = 80.0, 12_000.0, 1.0
    sweep = exponential_sweep(sample_rate, start_hz, end_hz, duration_s, 0.0, 0.0, 0.0)
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


def test_rir_residual_alignment_uses_common_shift_and_preserves_microphone_delay():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(2048, 2)).astype(np.float32)
    shifted = np.zeros_like(reference)
    shifted[7:, 0] = reference[:-7, 0]
    shifted[7:, 1] = reference[:-7, 1]

    aligned, shifts, correlations = align_rir_to_reference(shifted, reference, 16)

    assert shifts == [-7, -7]
    assert min(correlations) > 0.99
    assert np.allclose(aligned[16:-16], reference[16:-16])


def test_regularized_ess_deconvolution_recovers_known_two_channel_fir():
    sample_rate = 48_000
    sweep = exponential_sweep(sample_rate, 40, 22_000, 0.5, -12)
    silence = np.zeros(round(0.2 * sample_rate), dtype=np.float32)
    excitation = np.concatenate((sweep, silence))
    responses = []
    expected_peaks = [173, 180]
    for peak in expected_peaks:
        impulse = np.zeros(1200, dtype=np.float32)
        impulse[peak] = 0.8
        impulse[peak + 121] = 0.25
        impulse[peak + 367] = -0.12
        responses.append(np.convolve(excitation, impulse)[: len(excitation)])
    estimate = estimate_impulse_response(
        sweep, np.column_stack(responses), len(silence)
    )
    assert np.argmax(np.abs(estimate), axis=0).tolist() == expected_peaks
    assert np.all(np.isfinite(estimate))


def test_reconstruction_error_prefers_the_rir_that_generated_real_recording():
    rng = np.random.default_rng(18)
    excitation = rng.normal(0, 0.1, 2048).astype(np.float32)
    true_rir = np.zeros((256, 2), dtype=np.float32)
    true_rir[40, 0], true_rir[47, 1] = 0.8, 0.7
    true_rir[121, 0], true_rir[128, 1] = -0.2, -0.18
    measured = np.column_stack(
        [np.convolve(excitation, true_rir[:, channel]) for channel in range(2)]
    ).astype(np.float32)
    measured += rng.normal(0, 1e-4, measured.shape).astype(np.float32)
    wrong_rir = np.roll(true_rir, 17, axis=0)

    true_error = normalized_reconstruction_error_db(true_rir, excitation, [measured])
    wrong_error = normalized_reconstruction_error_db(wrong_rir, excitation, [measured])

    assert true_error < wrong_error - 20.0


def test_rir_selection_keeps_one_common_consensus_for_all_microphones():
    rng = np.random.default_rng(29)
    excitation = rng.normal(0, 0.1, 1024).astype(np.float32)
    true_rir = np.zeros((192, 2), dtype=np.float32)
    true_rir[30, 0], true_rir[37, 1] = 0.9, 0.75
    true_rir[88, 0], true_rir[95, 1] = 0.22, 0.18
    accepted = []
    for index in range(1, 6):
        rir = true_rir + rng.normal(0, 2e-4, true_rir.shape).astype(np.float32)
        if index == 5:
            rir = np.roll(true_rir, 25, axis=0)
        measured = np.column_stack(
            [np.convolve(excitation, true_rir[:, channel]) for channel in range(2)]
        ).astype(np.float32)
        measured += rng.normal(0, 2e-4, measured.shape).astype(np.float32)
        accepted.append(RIRTake(index, rir, [30, 37], True, {}, measured))

    result = select_rir_ensemble(accepted, excitation, correlation_threshold=0.98)

    assert result["consensus_take_indices"] == [1, 2, 3, 4]
    assert result["selected_take_indices"] in ([1, 2, 3, 4], [1], [2], [3], [4])
    assert 5 not in result["selected_take_indices"]
    assert result["all_accepted_mean_reconstruction_error_db"] > result[
        "consensus_mean_reconstruction_error_db"
    ]


def test_normalized_rir_change_reports_one_percent_as_minus_40_db():
    previous = np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32)
    current = previous.copy()
    current[1] = [0.01, 0.005]
    assert np.isclose(normalized_rir_change_db(current, previous), -40.0, atol=0.01)
