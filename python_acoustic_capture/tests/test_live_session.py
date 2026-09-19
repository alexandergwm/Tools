import threading
from types import SimpleNamespace
import numpy as np
import pytest
from acoustic_capture.audio import SoundDeviceBackend
from acoustic_capture.config import AudioConfig
from acoustic_capture.live_session import LiveSession


def make_session(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('acoustic_capture.live_session.time.monotonic', lambda: clock[0])
    backend = SoundDeviceBackend(AudioConfig(sample_rate=1000))
    live = LiveSession(backend, 2)
    return live, backend, clock


def tick(live, clock, now, flags=None):
    clock[0] = now
    output = np.ones((20, 2), np.float32)
    live.callback(np.ones((20, 2), np.float32)*.2, output, 20, SimpleNamespace(), flags)
    return output


def test_idle_discards_audio_and_requires_sustained_stability(monkeypatch):
    live, backend, clock = make_session(monkeypatch)
    assert not tick(live, clock, 0).any()
    tick(live, clock, .02)
    tick(live, clock, 3.35)
    assert not live.ready and live.job is None
    for i in range(1, 52):
        tick(live, clock, 3.35+i*.02)
    assert live.ready


def test_measurement_gap_is_flagged_without_driver_xrun(monkeypatch):
    live, backend, clock = make_session(monkeypatch)
    status = backend._new_callback_status()
    status.update(timing_discontinuity=False, max_callback_gap_seconds=0)
    live.job = dict(recording=np.zeros((40,2),np.float32), output=np.ones((40,1),np.float32)*.1,
                    cursor=0,status=status,done=threading.Event(),requested_at=0.)
    live.stable_since=0.
    first=tick(live,clock,1.)
    assert np.allclose(first[:,0],.1) and not first[:,1].any()
    tick(live,clock,4.3)
    assert status['timing_discontinuity'] and not status['xrun']
    assert live.job['done'].is_set()
    assert np.allclose(live.job['recording'],.2)


def test_job_does_not_start_on_stale_ready_flag(monkeypatch):
    live, backend, clock=make_session(monkeypatch)
    live.ready=True
    live.stable_since=0.
    live.last_time=1.
    live.job=dict(recording=np.zeros((40,2),np.float32),output=np.ones((40,1),np.float32),
                  cursor=0,status=backend._new_callback_status(),done=threading.Event())
    assert not tick(live,clock,4.).any()
    assert live.job['cursor']==0


def test_ready_resets_after_a_gap(monkeypatch):
    live, _, clock=make_session(monkeypatch)
    for i in range(60):tick(live,clock,i*.02)
    assert live.ready
    tick(live,clock,5.)
    assert not live.ready


def test_scene_qc_rejects_timing_discontinuity():
    from acoustic_capture.scene import _quality_flag
    assert _quality_flag({'target_only':{'backend_status':{'timing_discontinuity':True}}}) != '通过'


def test_gui_warns_for_scene_timing_failure_without_driver_xrun():
    from acoustic_capture.gui import _run_quality_warnings
    warnings=_run_quality_warnings({'summary':{'captures':[{'backend_status':{'timing_discontinuity':True}}]}})
    assert any('时间连续性' in warning for warning in warnings)


def test_session_cleanup_runs_on_exception(monkeypatch):
    closed=[]
    monkeypatch.setattr(LiveSession, 'close', lambda self: closed.append(threading.get_ident()))
    backend=SoundDeviceBackend(AudioConfig())
    with pytest.raises(ValueError):
        with backend.experiment_session(2):
            assert backend._experiment_live is not None
            raise ValueError('test')
    assert closed==[threading.get_ident()]
    assert backend._experiment_live is None


def test_scene_partial_capture_keeps_invalid_label_and_can_resume(tmp_path):
    import json
    from acoustic_capture.audio import SimulatedBackend, CaptureResult
    from acoustic_capture.config import ExperimentConfig
    from acoustic_capture.demo import generate_demo_audio
    from acoustic_capture.scene import capture_scene_block
    cfg=ExperimentConfig()
    files=generate_demo_audio(tmp_path/'corpus',duration_s=.05)
    cfg.scene.source_mode='single'
    cfg.scene.target_file=str(files['target'])
    cfg.scene.interferer_file=str(files['interferer'])
    cfg.scene.duration_s=.05
    cfg.scene.countdown_s=0
    cfg.scene.gap_s=0
    cfg.scene.items=['target_only','interferer_only','mixture']
    cfg.storage.root=str(tmp_path/'runs')
    stopped=[False]
    class Partial(SimulatedBackend):
        def play_record(self, output):
            result=super().play_record(output)
            stopped[0]=True
            return CaptureResult(result.microphones[:100], {'cancelled':True})
    store=capture_scene_block(cfg,Partial(cfg.audio),log=lambda _:None,stop_requested=lambda:stopped[0])
    assert store.manifest['status']=='cancelled'
    rows=[json.loads(s) for s in store.path('raw/labels.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==1 and rows[0]['valid']=='否'
    assert store.path(rows[0]['paired_sequence_recording']).is_file()
    cfg.scene.resume_run=str(store.root)
    resumed=capture_scene_block(cfg,SimulatedBackend(cfg.audio),log=lambda _:None)
    assert resumed.manifest['status']=='completed'
    assert resumed.manifest['summary']['label_rows']==1
