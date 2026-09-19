"""Fail-closed quality checks for laboratory capture."""
import csv
import numpy as np
import pytest
from acoustic_capture.quality import channel_metrics, evaluate_supervision_quality_gate
from acoustic_capture.audio import SimulatedBackend, CaptureResult
from acoustic_capture.human import capture_human
from test_human_capture import configuration


def test_mono_metrics_are_one_channel():
    result = channel_metrics(np.array([0., .2, -.3], np.float32))
    assert len(result) == 1
    assert result[0]['peak'] == pytest.approx(.3)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_supervision_gate_rejects_nonfinite(value):
    result = evaluate_supervision_quality_gate(
        {'residual_db_max': value, 'correlation_min': 1.},
        {'quality_gate': {'enabled': True}})
    assert not result['passed']


@pytest.mark.parametrize('fault', ['silent', 'clipped', 'duplicate', 'nonfinite'])
def test_human_invalid_audio_is_saved_but_never_passed(tmp_path, fault):
    cfg = configuration(tmp_path)
    class Faulty(SimulatedBackend):
        def record(self, frames):
            data = np.random.default_rng(8).normal(0, .01, (frames, 3)).astype(np.float32)
            if fault == 'silent': data[:, 1] = 0
            if fault == 'clipped': data[10, 1] = 1.
            if fault == 'duplicate': data[:, 1] = data[:, 0]
            if fault == 'nonfinite': data[10, 1] = np.nan
            return CaptureResult(data, {'xrun': False})
    store = capture_human(cfg, Faulty(cfg.audio), log=lambda _: None)
    with store.path('raw/labels.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    target = [row for row in rows if row['stage'] == 'target_only']
    assert target and all(row['quality_passed'] == 'False' for row in target)
    assert all(row['quality_issues'] for row in target)
    assert store.path(target[0]['recording']).is_file()


def test_three_stage_default_rejects_acoustically_inconsistent_mixture(tmp_path):
    from acoustic_capture.scene import capture_scene_block
    cfg = configuration(tmp_path)
    cfg.scene.target_file = cfg.scene.interferer_file = str(tmp_path / 'noise/test.wav')
    cfg.scene.duration_s = .1
    cfg.scene.gap_s = .2
    cfg.scene.countdown_s = 0
    class ChangedGain(SimulatedBackend):
        def play_record(self, output):
            result = super().play_record(output)
            # Third stage has an unrequested gain change, without an xrun.
            result.microphones[-round(.3 * cfg.audio.sample_rate):] *= .01
            return result
    store = capture_scene_block(cfg, ChangedGain(cfg.audio), log=lambda _: None)
    with store.path('raw/labels.csv').open(encoding='utf-8-sig') as handle:
        row = next(csv.DictReader(handle))
    assert row['quality_gate_enabled'] == '是'
    assert row['quality_gate_passed'] == '否'
    assert row['supervision_ready'] == '否'
