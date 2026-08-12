"""End-to-end simulated verification for ten paired speech samples.

This script produces ten deliberately distinct target/noise source files,
captures target-only plus mixture through the deterministic simulated room,
and verifies the resulting training index.  It is a workflow verification,
not a substitute for real loudspeaker/microphone acceptance testing.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_capture.audio import SimulatedBackend
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.scene import capture_scene_block
from acoustic_capture.speech_dataset import compile_speech_dataset


SAMPLE_RATE = 16_000
DURATION_S = 0.32
PAIR_COUNT = 10


def _write_sources(root: Path) -> tuple[Path, Path]:
    """Create ten target and ten interferer WAVs with distinct signatures."""
    targets, interferers = root / "target", root / "interferer"
    targets.mkdir(parents=True)
    interferers.mkdir(parents=True)
    time = np.arange(round(SAMPLE_RATE * DURATION_S), dtype=np.float32) / SAMPLE_RATE
    for index in range(1, PAIR_COUNT + 1):
        # Harmonic voiced-like target: each file has a different fundamental.
        f0 = 135.0 + 19.0 * index
        target = (
            np.sin(2 * np.pi * f0 * time)
            + 0.42 * np.sin(2 * np.pi * 2 * f0 * time + 0.11 * index)
            + 0.21 * np.sin(2 * np.pi * 3 * f0 * time)
        )
        target *= 0.35 + 0.65 * np.sin(np.pi * time / DURATION_S) ** 2

        # Different deterministic noise/chirp combination for every pair.
        rng = np.random.default_rng(10_000 + index)
        interferer = (
            0.62 * np.sin(2 * np.pi * (500.0 + 71.0 * index) * time)
            + 0.24 * np.sin(2 * np.pi * (1_200.0 + 41.0 * index) * time)
            + 0.10 * rng.standard_normal(len(time))
        )
        fade = np.sin(np.pi * time / DURATION_S) ** 2
        interferer *= fade
        sf.write(targets / f"target_{index:02d}.wav", target.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        sf.write(interferers / f"interferer_{index:02d}.wav", interferer.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
    return targets, interferers


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "verification_10_pairs"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    target_folder, interferer_folder = _write_sources(root / "sources")

    config = ExperimentConfig()
    config.audio.backend = "simulated"
    config.audio.sample_rate = SAMPLE_RATE
    config.audio.input_channels = [1, 2]
    config.audio.target_output_channel = 1
    config.audio.interferer_output_channel = 2
    # Keep the otherwise unused RIR section valid at this lower verification
    # sampling rate.
    config.sweep.end_hz = 7_000
    config.scene.items = ["target_only", "mixture"]
    config.scene.source_mode = "folders"
    config.scene.target_folder = str(target_folder)
    config.scene.interferer_folder = str(interferer_folder)
    config.scene.pairing_mode = "cycle"
    config.scene.duration_s = DURATION_S
    config.scene.target_level_dbfs = -18.0
    config.scene.interferer_level_dbfs = -18.0
    config.scene.repetitions = 1
    config.scene.countdown_s = 0
    config.scene.gap_s = 0.025
    config.scene.label_prefix = "verify10"
    config.storage.root = str(root / "runs")
    config.storage.session_name = "ten_pair_closed_loop"
    config.storage.compute_sha256 = True
    config.metadata = {
        "project_id": "verify_ten_supervised_pairs",
        "scene_id": "simulated_closed_loop_10_pairs",
        "experiment_id": "simulated_closed_loop_10_pairs",
        "artificial_head_id": "simulator",
        "headset_unit_id": "simulator_headset",
        "wearing_id": "fixed",
        "boom_pose_id": "fixed",
        "target": {"source_id": "simulated_mouth", "position_id": "front"},
        "interferer": {"source_id": "simulated_interferer", "position_id": "right"},
    }
    config.validate()

    store = capture_scene_block(config, SimulatedBackend(config.audio), log=print)
    labels = _read_rows(store.path("labels.csv"))
    assert len(labels) == PAIR_COUNT, f"expected {PAIR_COUNT} label rows, got {len(labels)}"
    assert all(row["capture_type"] == "supervised_pair" for row in labels)
    assert all(row["supervision_ready"] == "是" for row in labels)
    assert all(row["target_mixture_sample_aligned"] == "是" for row in labels)

    for index, row in enumerate(labels, 1):
        assert Path(row["target_source"]).name == f"target_{index:02d}.wav"
        assert Path(row["interferer_source"]).name == f"interferer_{index:02d}.wav"
        target, target_fs = sf.read(store.path(row["target_recording"]), always_2d=True)
        mixture, mixture_fs = sf.read(store.path(row["mixture_recording"]), always_2d=True)
        target_playback, _ = sf.read(store.path(row["target_playback"]), always_2d=True)
        mixture_playback, _ = sf.read(store.path(row["mixture_playback"]), always_2d=True)
        assert target_fs == mixture_fs == SAMPLE_RATE
        assert target.shape == mixture.shape == (round(SAMPLE_RATE * DURATION_S), 2)
        assert np.array_equal(target_playback[:, 0], mixture_playback[:, 0])
        assert np.allclose(target_playback[:, 1], 0.0)
        assert float(np.sqrt(np.mean((mixture - target) ** 2))) > 1e-3

    dataset = compile_speech_dataset(
        config.storage.root,
        root / "dataset",
        project_id="verify_ten_supervised_pairs",
    )
    supervised = _read_rows(dataset / "indexes" / "supervised_pairs.csv")
    assert len(supervised) == PAIR_COUNT
    for row in supervised:
        mixture = dataset / row["mixture_recording"]
        target = dataset / row["target_recording"]
        assert mixture.is_file() and target.is_file()
        assert sf.info(mixture).frames == sf.info(target).frames
        assert sf.info(mixture).channels == sf.info(target).channels == 2

    report = {
        "status": "passed",
        "source_pairs": PAIR_COUNT,
        "label_rows": len(labels),
        "supervised_pairs": len(supervised),
        "run_dir": str(store.root),
        "dataset_dir": str(dataset),
        "workbook": str(dataset / "indexes" / "speech_dataset.xlsx"),
    }
    (root / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
