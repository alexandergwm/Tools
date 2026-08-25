from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import soundfile as sf

from acoustic_capture.acqua_simple import (
    finish_acqua_recording,
    generate_acqua_mixed_target_program,
    prepare_acqua_recording,
)
from acoustic_capture.campaign import create_campaign, package_campaign
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.labels import write_label_files


def _write(path: Path, value: str | bytes = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _manifest(run: Path, kind: str, status: str = "completed") -> None:
    _write(
        run / "manifest.json",
        json.dumps(
            {
                "kind": kind,
                "status": status,
                "metadata": {"experiment_name": run.name},
                "summary": {
                    "selected_take_ids": [1, 2],
                    "label_rows": 1,
                    "supervision_ready_rows": 0,
                },
            }
        ),
    )


def test_big_experiment_makes_one_selective_zip(tmp_path: Path):
    root = create_campaign(tmp_path, "headset_A_room_1")
    rir = root / "runs" / "rir_angle_090"
    _manifest(rir, "rir")
    _write(rir / "raw" / "take_001.wav", b"raw sweep")
    _write(rir / "references" / "played.wav", b"reference")
    _write(rir / "processed" / "average_rir.wav", b"final rir")
    _write(rir / "processed" / "average_rir_mic_01.wav", b"mic 1")
    _write(rir / "processed" / "take_001_rir.wav", b"intermediate")
    _write(rir / "metrics" / "summary.json", "{}")

    scene = root / "runs" / "speech_scene_01"
    _manifest(scene, "scene", status="failed")
    _write(scene / "raw" / "pair_001_mixture_mics.wav", b"mixture")
    _write(scene / "raw" / "pair_001_target_only_mics.wav", b"target")
    _write(scene / "raw" / "pair_001_paired_sequence_mics.wav", b"long")
    _write(scene / "raw" / "labels.csv", "sample_id,supervision_ready\na,否\n")
    _write(scene / "raw" / "supervised_pairs.csv", "sample_id,supervision_ready\na,否\n")
    _write(scene / "references" / "pair_001_playback.wav", b"reference")

    result = package_campaign(root)

    assert Path(result["zip"]).is_file()
    assert result["run_count"] == 2
    with zipfile.ZipFile(result["zip"]) as archive:
        names = set(archive.namelist())
    prefix = f"{root.name}/"
    assert prefix + "big_experiment_runs.csv" in names
    assert prefix + "runs/rir_angle_090/processed/average_rir.wav" in names
    assert prefix + "runs/rir_angle_090/processed/average_rir_mic_01.wav" in names
    assert prefix + "runs/speech_scene_01/raw/pair_001_mixture_mics.wav" in names
    assert prefix + "runs/speech_scene_01/raw/labels.csv" in names
    assert prefix + "runs/speech_scene_01/raw/supervised_pairs.csv" in names
    assert not any("references/" in name for name in names)
    assert not any("rir_angle_090/raw/" in name for name in names)
    assert not any(name.endswith("take_001_rir.wav") for name in names)
    assert not any(name.endswith("paired_sequence_mics.wav") for name in names)
    rows = list(
        csv.DictReader(
            (root / "big_experiment_runs.csv").open(encoding="utf-8-sig")
        )
    )
    assert [row["status"] for row in rows] == ["completed", "failed"]
    assert rows[0]["accepted_rir_takes"] == "2"


def test_failed_supervised_pair_is_kept_in_pairing_table(tmp_path: Path):
    row = {
        "sample_id": "pair_failed_001",
        "supervision_pair_id": "pair_failed_001",
        "valid": "否",
        "quality_flag": "存在静音通道",
        "supervision_ready": "否",
        "mixture_recording": "raw/mix.wav",
        "target_recording": "raw/target.wav",
    }

    files = write_label_files(tmp_path, [row], {})

    with files["supervised_csv"].open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 1
    assert pairs[0]["sample_id"] == "pair_failed_001"
    assert pairs[0]["supervision_ready"] == "否"
    assert pairs[0]["quality_flag"] == "存在静音通道"


def _acqua_config(tmp_path: Path) -> ExperimentConfig:
    target_root = tmp_path / "targets"
    interferer_root = tmp_path / "interferers"
    target_root.mkdir()
    interferer_root.mkdir()
    fs = 48_000
    samples = np.arange(960, dtype=np.float32)
    for index, frequency in enumerate((300, 500), 1):
        signal = 0.2 * np.sin(2 * np.pi * frequency * samples / fs)
        sf.write(target_root / f"target_{index}.wav", signal, fs)
    for index, frequency in enumerate((700, 900), 1):
        signal = 0.2 * np.sin(2 * np.pi * frequency * samples / fs)
        sf.write(interferer_root / f"noise_{index}.wav", signal, fs)
    config = ExperimentConfig()
    config.audio.sample_rate = fs
    config.scene.source_mode = "folders"
    config.scene.target_folder = str(target_root)
    config.scene.interferer_folder = str(interferer_root)
    config.scene.items = ["target_only", "mixture"]
    config.acqua.segment_duration_s = 0.01
    config.acqua.gap_s = 0.002
    config.acqua.pairing_seed = 73
    config.acqua.wav_subtype = "FLOAT"
    return config


def test_acqua_program_is_mixed_target_alternation_with_mapping(tmp_path: Path):
    config = _acqua_config(tmp_path)
    progress: list[dict] = []

    result = generate_acqua_mixed_target_program(
        config, tmp_path / "programs", "lab_sequence", progress=progress.append
    )

    program = Path(result["program"])
    data, fs = sf.read(program, always_2d=True, dtype="float32")
    with (Path(result["root"]) / "sequence_mapping.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert fs == config.audio.sample_rate
    assert data.shape[1] == 2
    assert [row["stage"] for row in rows] == [
        "mixed",
        "target_only",
        "mixed",
        "target_only",
    ]
    for row in rows:
        start, end = int(row["start_sample"]), int(row["end_sample"])
        block = data[start:end]
        assert len(block) == round(config.acqua.segment_duration_s * fs)
        if row["stage"] == "mixed":
            assert np.max(np.abs(block[:, 0])) > 0
            assert np.max(np.abs(block[:, 1])) > 0
        else:
            assert np.max(np.abs(block[:, 0])) > 0
            assert np.array_equal(block[:, 1], np.zeros(len(block)))
    assert len(progress) == 2
    manifest = json.loads(
        (Path(result["root"]) / "program_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["order"] == "mixed,target_only,mixed,target_only,..."
    assert manifest["pairing_seed"] == 73


def test_acqua_early_recording_keeps_prefix_and_mapping(tmp_path: Path):
    config = _acqua_config(tmp_path)
    result = generate_acqua_mixed_target_program(
        config, tmp_path / "programs", "partial_sequence"
    )
    prepared = prepare_acqua_recording(
        tmp_path / "recordings", "stopped_after_first_pairs", result["program"]
    )
    frames = 320
    sf.write(
        prepared["output"],
        np.zeros((frames, 2), dtype=np.float32),
        config.audio.sample_rate,
        subtype="FLOAT",
    )
    manifest_path = finish_acqua_recording(
        prepared,
        {
            "frames": frames,
            "channels": 2,
            "sample_rate": config.audio.sample_rate,
            "duration_s": frames / config.audio.sample_rate,
        },
        stopped_early=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped_early"
    assert manifest["usable_prefix_preserved"] is True
    assert manifest["complete_mapped_segments"] == 0
    assert (Path(prepared["root"]) / "sequence_mapping.csv").is_file()
    assert (Path(prepared["root"]) / "recorded_prefix_mapping.csv").is_file()
    assert Path(prepared["output"]).is_file()
