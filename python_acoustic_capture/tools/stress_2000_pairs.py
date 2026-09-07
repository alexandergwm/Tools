"""Low-duration 2000-pair stress test for scheduling, checkpoints and labels.

This deliberately keeps audio clips tiny so it measures application overhead
rather than spending gigabytes during CI.  It still creates 4000 valid source
files, performs 2000 complete paired captures, writes every checkpoint and
compiles the final training index.
"""

from __future__ import annotations

import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_capture.audio import SimulatedBackend
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.scene import capture_scene_block
from acoustic_capture.speech_dataset import compile_speech_dataset


PAIR_COUNT = 2_000
SAMPLE_RATE = 8_000
FRAMES = 80


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="acoustic_capture_stress_") as temp:
        root = Path(temp)
        targets, interferers = root / "targets", root / "interferers"
        targets.mkdir()
        interferers.mkdir()
        signal = (0.2 * np.sin(2 * np.pi * 440 * np.arange(FRAMES) / SAMPLE_RATE)).astype(
            np.float32
        )
        started = time.perf_counter()
        for index in range(PAIR_COUNT):
            sf.write(targets / f"target_{index:04d}.wav", signal, SAMPLE_RATE)
            sf.write(interferers / f"noise_{index:04d}.wav", signal, SAMPLE_RATE)
        source_seconds = time.perf_counter() - started

        config = ExperimentConfig()
        config.audio.backend = "simulated"
        config.audio.sample_rate = SAMPLE_RATE
        config.sweep.end_hz = 3_500
        config.scene.items = ["target_only", "interferer_only", "mixture"]
        config.scene.source_mode = "folders"
        config.scene.target_folder = str(targets)
        config.scene.interferer_folder = str(interferers)
        config.scene.duration_s = FRAMES / SAMPLE_RATE
        config.scene.countdown_s = 0
        config.scene.gap_s = 0
        config.storage.root = str(root / "runs")
        config.storage.compute_sha256 = False
        config.storage.save_playback_reference = False
        config.metadata = {
            "project_id": "stress_2000_pairs",
            "room_id": "simulated_room",
            "artificial_head_id": "simulated_head",
            "headset_model_id": "simulated_model",
            "headset_unit_id": "simulated_unit",
            "wearing_id": "fixed",
            "boom_pose_id": "fixed",
            "microphone_1": "left",
            "microphone_2": "right",
            "target": {
                "source_id": "simulated_mouth",
                "position_id": "front",
                "azimuth_deg": 0,
                "elevation_deg": 0,
                "height_m": 1.4,
                "distance_m": 0.05,
            },
            "interferer": {
                "source_id": "simulated_interferer",
                "position_id": "right",
                "azimuth_deg": 90,
                "elevation_deg": 0,
                "height_m": 1.4,
                "distance_m": 1.0,
            },
        }

        tracemalloc.start()
        capture_started = time.perf_counter()
        store = capture_scene_block(
            config, SimulatedBackend(config.audio), log=lambda _message: None
        )
        capture_seconds = time.perf_counter() - capture_started
        _current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        compile_started = time.perf_counter()
        dataset = compile_speech_dataset(
            config.storage.root, root / "dataset", copy_audio=False
        )
        compile_seconds = time.perf_counter() - compile_started
        labels = store.path("labels.jsonl").read_text(encoding="utf-8").splitlines()
        manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
        assert len(labels) == PAIR_COUNT
        assert manifest["supervised_pair_count"] == PAIR_COUNT
        assert store.manifest["status"] == "completed"
        print(
            json.dumps(
                {
                    "status": "passed",
                    "source_files": PAIR_COUNT * 2,
                    "pairs": PAIR_COUNT,
                    "source_generation_s": round(source_seconds, 3),
                    "capture_and_checkpoint_s": round(capture_seconds, 3),
                    "dataset_compile_s": round(compile_seconds, 3),
                    "python_peak_memory_mib": round(peak_memory / (1024**2), 2),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
