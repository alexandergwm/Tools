import csv
import json
from pathlib import Path

import soundfile as sf
import yaml

from acoustic_capture.audio import SimulatedBackend
from acoustic_capture.config import ExperimentConfig, load_config, save_config
from acoustic_capture.experiment import (
    compile_completed_rir_runs,
    compile_rir_dataset,
    experiment_id,
    expand_experiment_plan,
    load_experiment_plan,
)
from acoustic_capture.rir import capture_rir


def _experiment(wearing_id: str, split: str) -> dict:
    return {
        "artificial_head_id": "head_a",
        "headset_model_id": "model_a",
        "headset_unit_id": "hs01",
        "wearing_id": wearing_id,
        "boom_pose_id": "b00_nominal",
        "source_role": "mouth",
        "source_id": "mouth01",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "source_height_cm": 140,
        "distance_cm": 5,
        "dataset_split": split,
        "output_channel": 1,
    }


def _write_plan(tmp_path: Path) -> Path:
    base = ExperimentConfig()
    base.audio.backend = "simulated"
    base.sweep.duration_s = 0.08
    base.sweep.pre_silence_s = 0.02
    base.sweep.post_silence_s = 0.03
    base.sweep.rir_duration_s = 0.04
    base.repeats.fixed_count = 1
    base.repeats.minimum = 1
    base.repeats.maximum = 1
    base.repeats.required_stable_takes = 1
    base.repeats.pause_s = 0
    base.storage.compute_sha256 = False
    save_config(base, tmp_path / "base.yaml")
    plan = {
        "schema_version": 1,
        "project": {"project_id": "test_project", "dataset_version": "v1"},
        "paths": {
            "base_config": "base.yaml",
            "generated_configs": "generated",
            "runs_root": "runs",
            "dataset_root": "dataset",
        },
        "experiments": [_experiment("w01", "train"), _experiment("w02", "test")],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def test_experiment_id_is_stable_and_windows_safe():
    first = experiment_id(_experiment("W 01", "train"))
    second = experiment_id(_experiment("W 01", "train"))
    assert first == second
    assert " " not in first
    assert "azp000" in first
    assert "h140" in first


def test_experiment_id_distinguishes_same_angle_at_different_heights():
    low = _experiment("w01", "train")
    high = {**low, "source_height_cm": 170}
    assert experiment_id(low) != experiment_id(high)


def test_manual_experiment_id_keeps_chinese_name_and_removes_path_punctuation():
    assert experiment_id({"experiment_id": "耳机A / 角度 90°"}) == "耳机a-角度-90"


def test_matrix_plan_expands_to_experiment_configs(tmp_path: Path):
    plan_path = _write_plan(tmp_path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["matrix"] = {
        "artificial_heads": [
            {"artificial_head_id": "head_a"},
            {"artificial_head_id": "head_b"},
        ],
        "headset_units": [{"headset_model_id": "m", "headset_unit_id": "h1"}],
        "wearings": [{"wearing_id": "w1", "dataset_split": "train"}],
        "source_sets": [
            {
                "source_role": "mouth",
                "source_id": "mouth",
                "output_channel": 1,
                "boom_poses": ["nominal", "up"],
                "poses": [
                    {
                        "azimuth_deg": 0,
                        "elevation_deg": 0,
                        "source_height_cm": 140,
                        "distance_cm": 5,
                    }
                ],
            }
        ],
    }
    plan.pop("experiments")
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    loaded = load_experiment_plan(plan_path)
    written = expand_experiment_plan(plan_path)

    assert len(loaded["experiments"]) == len(written) == 4
    generated = load_config(written[0])
    assert generated.metadata["project_id"] == "test_project"
    assert generated.metadata["experiment_id"] == generated.storage.session_name
    assert generated.metadata["artificial_head_id"] == "head_a"
    assert generated.metadata["experiment_id"].startswith("ah-head-a__")


def test_interferer_rir_plan_can_use_output_channel_two(tmp_path: Path):
    plan_path = _write_plan(tmp_path)
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["experiments"] = [
        {
            **_experiment("w01", "train"),
            "source_role": "interferer",
            "source_id": "noise_speaker",
            "output_channel": 2,
        }
    ]
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    [generated_path] = expand_experiment_plan(plan_path)
    generated = load_config(generated_path)
    assert generated.audio.target_output_channel == 2
    assert generated.metadata["source_role"] == "interferer"


def test_compile_rir_dataset_exports_one_mean_ir_per_microphone_without_cross_experiment_mean(
    tmp_path: Path,
):
    plan_path = _write_plan(tmp_path)
    generated = expand_experiment_plan(plan_path)
    for config_path in generated:
        config = load_config(config_path)
        store = capture_rir(config, SimulatedBackend(config.audio), log=lambda _: None)
        assert store.path("processed/average_rir_mic_01.wav").is_file()
        assert store.path("processed/average_rir_mic_02.wav").is_file()

    dataset_root = compile_rir_dataset(plan_path)
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    with (dataset_root / "indexes/rir_experiments.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert manifest["planned_experiments"] == manifest["completed_experiments"] == 2
    assert manifest["cross_experiment_averaging"] is False
    assert all(row["status"] == "completed" for row in rows)
    assert not (dataset_root / "rir/groups").exists()
    assert (dataset_root / "indexes/rir_dataset.xlsx").is_file()
    for row in rows:
        dual, sample_rate = sf.read(
            dataset_root / row["mean_ir_multichannel"], always_2d=True
        )
        mic_1, _ = sf.read(dataset_root / row["mean_ir_mic_01"], always_2d=True)
        mic_2, _ = sf.read(dataset_root / row["mean_ir_mic_02"], always_2d=True)
        assert dual.shape[1] == 2
        assert mic_1.shape[1] == mic_2.shape[1] == 1
        assert sample_rate == 48_000


def test_compile_manual_rir_runs_keeps_runs_independent_and_all_microphones(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.audio.input_channels = [1, 3, 5]
    cfg.sweep.duration_s = 0.08
    cfg.sweep.pre_silence_s = 0.02
    cfg.sweep.post_silence_s = 0.03
    cfg.sweep.rir_duration_s = 0.04
    cfg.repeats.fixed_count = 1
    cfg.repeats.minimum = 1
    cfg.repeats.maximum = 1
    cfg.repeats.required_stable_takes = 1
    cfg.repeats.pause_s = 0
    cfg.storage.root = str(tmp_path / "runs")
    cfg.storage.compute_sha256 = False
    cfg.storage.session_name = "manual_az090_h170"
    cfg.metadata = {
        "project_id": "manual_project",
        "experiment_id": "manual_az090_h170",
        "artificial_head_id": "head01",
        "room_id": "lab_a",
        "headset_model_id": "model_a",
        "headset_unit_id": "hs01",
        "wearing_id": "w01",
        "boom_pose_id": "b00",
        "source_role": "interferer",
        "source_id": "speaker01",
        "azimuth_deg": 90,
        "elevation_deg": 0,
        "source_height_cm": 170,
        "distance_cm": 100,
        "microphone_1": "left",
        "microphone_3": "right",
        "microphone_5": "reference",
    }
    for _ in range(2):
        capture_rir(cfg, SimulatedBackend(cfg.audio), log=lambda _: None)

    dataset = compile_completed_rir_runs(
        tmp_path / "runs",
        tmp_path / "manual_dataset",
        project_id="manual_project",
    )
    with (dataset / "indexes/rir_experiments.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == manifest["completed_experiments"] == 2
    assert manifest["cross_experiment_averaging"] is False
    assert {row["duplicate_manual_name_count"] for row in rows} == {"2"}
    assert all(row["microphone_channels"] == "3" for row in rows)
    assert all(row["recording_channels"] == "1,3,5" for row in rows)
    channel_map = json.loads(rows[0]["mean_ir_channel_map_json"])
    assert [item["recording_channel"] for item in channel_map] == [1, 3, 5]
    assert [item["microphone_id"] for item in channel_map] == ["left", "right", "reference"]
    assert all(row["mean_ir_mic_03"] for row in rows)
    assert (dataset / "indexes/rir_dataset.xlsx").is_file()


def test_plan_compiler_rejects_same_name_with_changed_physical_condition(tmp_path: Path):
    plan_path = _write_plan(tmp_path)
    generated = expand_experiment_plan(plan_path)
    config = load_config(generated[0])
    config.metadata["source_height_cm"] = 999
    capture_rir(config, SimulatedBackend(config.audio), log=lambda _: None)

    dataset = compile_rir_dataset(plan_path)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    mismatches = (dataset / "indexes/condition_mismatches.jsonl").read_text(encoding="utf-8")
    assert manifest["condition_mismatch_count"] == 1
    assert not manifest["training_ready"]
    assert "999" in mismatches
