"""Explicit opt-in hardware smoke test; DOES play sound and record a microphone."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import numpy as np
import soundfile as sf

from acoustic_capture.audio import create_backend
from acoustic_capture.config import ExperimentConfig, save_config
from acoustic_capture.demo import generate_demo_audio
from acoustic_capture.general import capture_general_io
from acoustic_capture.human import capture_human
from acoustic_capture.rir import capture_rir
from acoustic_capture.scene import capture_scene_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=int, required=True)
    parser.add_argument('--output', type=int, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--sample-rate', type=int, default=48000)
    parser.add_argument('--block-size', type=int, default=1024)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = generate_demo_audio(root / 'stimuli', sample_rate=args.sample_rate, duration_s=4)
    cfg = ExperimentConfig()
    cfg.audio.backend = 'sounddevice'
    cfg.audio.host_api = 'MME'
    cfg.audio.input_device = args.input
    cfg.audio.output_device = args.output
    cfg.audio.input_channels = [1]
    cfg.audio.block_size = args.block_size
    cfg.audio.sample_rate = args.sample_rate
    cfg.storage.root = str(root / 'runs')
    cfg.metadata = {'experiment_purpose': 'MME hardware smoke test, synthetic stimuli',
                    'not_training_data': True}
    cfg.general.action = 'record'
    cfg.general.duration_s = 4
    cfg.sweep.level_dbfs = -24
    cfg.sweep.duration_s = 2
    cfg.sweep.start_hz = 100
    cfg.sweep.end_hz = 8000
    cfg.sweep.pre_silence_s = .5
    cfg.sweep.post_silence_s = 1
    cfg.sweep.rir_duration_s = .5
    cfg.repeats.fixed_count = 2
    cfg.scene.source_mode = 'folders'
    cfg.scene.target_folder = str(files['target_folder_1'].parent)
    cfg.scene.interferer_folder = str(files['interferer_folder_1'].parent)
    cfg.scene.target_file = str(files['target'])
    cfg.scene.interferer_file = str(files['interferer'])
    cfg.scene.measurement_count = 2
    cfg.scene.duration_s = 4
    cfg.scene.items = ['target_only', 'interferer_only', 'mixture']
    cfg.scene.target_level_dbfs = -24
    cfg.scene.interferer_level_dbfs = -24
    cfg.scene.countdown_s = 0
    cfg.scene.gap_s = .3
    cfg.human.duration_s = 5
    cfg.human.preparation_s = 1
    save_config(cfg, root / 'mme_test.yaml')
    if not args.run:
        return
    reports = []
    for name, operation in [('simple', capture_general_io), ('S1', capture_rir),
                            ('S23', capture_scene_block), ('S4', capture_human),
                            ('simple_stop', capture_general_io)]:
        config = deepcopy(cfg)
        config.storage.session_name = name
        backend = create_backend(config.audio)
        stop = threading.Event()
        began = time.perf_counter()
        first = []
        timer = None
        def progress(update):
            nonlocal timer
            if update.get('phase') in ('recording', 'playing') and not first:
                first.append(time.perf_counter() - began)
                print('FIRST_AUDIO', name, first[-1], flush=True)
                if name == 'simple_stop':
                    def cancel():
                        stop.set()
                        backend.stop()
                    timer = threading.Timer(.8, cancel)
                    timer.start()
        backend.set_progress_callback(progress)
        report = {'mode': name}
        print('START', name, flush=True)
        try:
            store = operation(config, backend, stop_requested=stop.is_set)
            report.update(root=str(store.root), status=store.manifest['status'],
                          summary=store.manifest.get('summary'))
            report['raw_audio'] = []
            for path in store.root.glob('raw/*.wav'):
                data, fs = sf.read(path, always_2d=True)
                report['raw_audio'].append(dict(file=path.name, frames=len(data),
                    channels=data.shape[1], seconds=len(data)/fs,
                    peak=float(np.max(np.abs(data))) if len(data) else 0,
                    rms=float(np.sqrt(np.mean(data**2))) if len(data) else 0))
        except Exception as exc:
            report['error'] = repr(exc)
        finally:
            if timer:
                timer.cancel()
            report['elapsed_seconds'] = time.perf_counter()-began
            report['first_progress_seconds'] = first
            reports.append(report)
            (root / 'audit.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
            print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
