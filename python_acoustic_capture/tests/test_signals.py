import numpy as np

from acoustic_capture.quality import normalized_correlation
from acoustic_capture.rir import (
    RIRTake,
    align_rir_to_reference,
    detect_direct_arrival,
    estimate_sweep_clock_drift_ppm,
    estimate_impulse_response,
    extract_rir,
    gcc_phat_delay_samples,
    low_frequency_group_delay_samples,
    normalized_reconstruction_error_db,
    normalized_rir_change_db,
    reconstruction_metrics,
    rir_timing_metrics,
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


def test_direct_arrival_is_not_replaced_by_a_stronger_late_reflection():
    sample_rate = 48_000
    impulse = np.zeros(2048, dtype=np.float32)
    impulse[120] = 0.3
    impulse[400] = 1.0

    onset, direct_peak, details = detect_direct_arrival(impulse, sample_rate)

    assert onset < 120
    assert direct_peak == 120
    assert details["method"] == "first_persistent_energy_rise"


def test_relative_delay_estimators_use_target_minus_reference_sign():
    sample_rate = 48_000
    rng = np.random.default_rng(91)
    reference = rng.normal(size=4096)
    target = np.zeros_like(reference)
    target[7:] = reference[:-7]

    gcc_delay, gcc_details = gcc_phat_delay_samples(reference, target, sample_rate)
    group_delay, group_details = low_frequency_group_delay_samples(
        reference, target, sample_rate
    )

    assert np.isclose(gcc_delay, 7.0, atol=0.1)
    assert gcc_details["integer_delay_samples"] == 7
    assert group_delay is not None
    assert np.isclose(group_delay, 7.0, atol=0.3)
    assert group_details["reliable"] is True


def test_sweep_segment_timing_detects_independent_clock_drift():
    sample_rate = 48_000
    sweep = exponential_sweep(sample_rate, 40, 22_000, 2.0, -12)
    expected_ppm = 50.0
    stretch = 1.0 + expected_ppm * 1e-6
    stretched_count = round(len(sweep) * stretch)
    stretched = np.interp(
        np.arange(stretched_count) / stretch,
        np.arange(len(sweep)),
        sweep,
        left=0.0,
        right=0.0,
    )
    pre_silence_s = 0.2
    delay = 173
    recording = np.zeros(
        round(pre_silence_s * sample_rate) + delay + stretched_count + 1000
    )
    start = round(pre_silence_s * sample_rate) + delay
    recording[start : start + stretched_count] = stretched

    metrics = estimate_sweep_clock_drift_ppm(
        recording,
        sweep,
        sample_rate,
        pre_silence_s,
        delay,
    )

    assert metrics["reliable"] is True
    assert np.isclose(metrics["estimated_drift_ppm"], expected_ppm, atol=5.0)


def test_rir_timing_reports_physical_inter_microphone_delay():
    sample_rate = 48_000
    rir = np.zeros((4096, 2), dtype=np.float32)
    rir[120, 0], rir[127, 1] = 1.0, 0.8
    rir[300, 0], rir[307, 1] = 0.2, 0.16

    timing = rir_timing_metrics(rir, sample_rate)
    microphone_2 = timing["per_channel"][1]

    assert np.isclose(microphone_2["gcc_phat_delay_samples"], 7.0, atol=0.1)
    assert np.isclose(
        microphone_2["low_frequency_group_delay_samples"], 7.0, atol=0.1
    )
    assert microphone_2["estimators_agree_within_one_sample"] is True


def test_extract_rir_uses_first_arrival_and_one_common_crop_for_all_microphones():
    sample_rate = 48_000
    sweep = exponential_sweep(sample_rate, 40, 22_000, 0.25, -12)
    pre_silence_s = 0.05
    post_silence_s = 0.12
    pre = np.zeros(round(pre_silence_s * sample_rate), dtype=np.float32)
    post_samples = round(post_silence_s * sample_rate)
    responses = []
    for direct_delay in (173, 180):
        impulse = np.zeros(post_samples, dtype=np.float32)
        impulse[direct_delay] = 0.25
        impulse[direct_delay + 300] = 1.0
        convolved = np.convolve(sweep, impulse)
        responses.append(np.pad(convolved, (0, 1))[: len(sweep) + post_samples])
    recording = np.vstack(
        (pre[:, None].repeat(2, axis=1), np.column_stack(responses))
    )

    cropped, peaks, reference_peak, offsets, _full = extract_rir(
        recording,
        sweep,
        sample_rate,
        pre_silence_s,
        post_silence_s,
        0.08,
        0.01,
    )

    assert abs(reference_peak - 173) <= 1
    assert offsets == [0, 7]
    assert peaks[1] - peaks[0] == 7
    assert int(np.argmax(np.abs(cropped[:, 1]))) - int(
        np.argmax(np.abs(cropped[:, 0]))
    ) == 7


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


def test_regularized_ess_deconvolution_matches_matlab_r2024b_golden_value():
    """Golden value measured from sweeptone/impzest in MATLAB R2024b."""
    sample_rate = 48_000
    sweep = exponential_sweep(sample_rate, 40, 22_000, 0.5, -12)
    trailing = np.zeros(round(0.2 * sample_rate), dtype=np.float32)
    excitation = np.concatenate((sweep, trailing))
    impulse = np.zeros(1200, dtype=np.float32)
    impulse[173], impulse[294], impulse[540] = 0.8, 0.25, -0.12
    response = np.convolve(excitation, impulse)[: len(excitation)]

    estimate = estimate_impulse_response(sweep, response, len(trailing))[:, 0]

    assert int(np.argmax(np.abs(estimate))) == 173
    assert np.isclose(estimate[173], 0.71001269893, atol=3e-7)


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


def test_reconstruction_metrics_report_mse_nmse_and_correlation_per_channel():
    rng = np.random.default_rng(73)
    excitation = rng.normal(0, 0.1, 1024).astype(np.float32)
    rir = np.zeros((96, 2), dtype=np.float32)
    rir[12, 0], rir[19, 1] = 0.8, 0.65
    measured = np.column_stack(
        [np.convolve(excitation, rir[:, channel]) for channel in range(2)]
    ).astype(np.float32)

    metrics, reconstructed = reconstruction_metrics(rir, excitation, measured)

    assert reconstructed.shape == measured.shape
    assert metrics["mean_mse"] < 1e-14
    assert metrics["worst_nmse_db"] < -100
    assert metrics["minimum_correlation"] > 0.999999
    assert [item["microphone_channel"] for item in metrics["per_channel"]] == [1, 2]


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
