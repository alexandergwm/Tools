from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_capture.checklist import (
    apply_checklist_row,
    create_checklist,
    read_checklist,
    update_checklist_row,
)
from acoustic_capture.config import ExperimentConfig
from acoustic_capture.professional import build_preflight_report
from acoustic_capture.quality import mixture_additivity_metrics
from acoustic_capture.speech_dataset import audit_split_leakage


def test_checklist_round_trip_and_apply(tmp_path: Path):
    path = create_checklist(
        tmp_path / "capture.xlsx",
        [
            {
                "status": "待采集",
                "workflow": "rir",
                "experiment_name": "hs01_w02_target_mouth",
                "dataset_split": "train",
                "project_id": "beamformer_v1",
                "room_id": "lab_a",
                "artificial_head_id": "head01",
                "headset_model_id": "model_a",
                "headset_unit_id": "hs01",
                "wearing_id": "w02",
                "boom_pose_id": "b00",
                "source_role": "mouth",
                "source_id": "mouth01",
                "azimuth_deg": 0,
                "elevation_deg": 0,
            }
        ],
    )
    rows = read_checklist(path)
    assert len(rows) == 1
    assert rows[0]["_row_number"] == 2

    config = ExperimentConfig()
    kind = apply_checklist_row(config, rows[0], path)
    assert kind == "rir"
    assert config.storage.session_name == "hs01_w02_target_mouth"
    assert config.metadata["wearing_id"] == "w02"
    assert config.metadata["checklist_row"] == 2

    update_checklist_row(
        path,
        2,
        status="已完成",
        completed_run="runs/example",
    )
    updated = read_checklist(path)[0]
    assert updated["status"] == "已完成"
    assert updated["completed_run"] == "runs/example"
    assert updated["completed_at"]


def test_standard_profile_warns_but_does_not_block_legacy_channel_map(tmp_path: Path):
    config = ExperimentConfig()
    config.storage.root = str(tmp_path)
    config.metadata = {
        "capture_profile": "standard",
        "microphone_1": "left",
        "microphone_2": "right",
    }
    report = build_preflight_report(config, "rir")
    assert report.can_start
    assert report.warnings


def test_production_supervised_capture_blocks_separate_device_clocks(tmp_path: Path):
    config = ExperimentConfig()
    config.storage.root = str(tmp_path)
    config.audio.backend = "sounddevice"
    config.audio.input_device = "RME input"
    config.audio.output_device = "Other output"
    config.scene.items = ["target_only", "mixture"]
    config.metadata = {
        "capture_profile": "production",
        "project_id": "p1",
        "room_id": "lab",
        "artificial_head_id": "head1",
        "headset_model_id": "m1",
        "headset_unit_id": "u1",
        "wearing_id": "w1",
        "boom_pose_id": "b1",
        "microphone_1": "left",
        "microphone_2": "right",
        "target": {"source_id": "mouth1", "position_id": "fixed"},
    }
    report = build_preflight_report(
        config, "scene", check_source_paths=False
    )
    assert not report.can_start
    assert any(check.check_id == "shared_hardware_clock" for check in report.errors)


def test_mixture_additivity_metrics_are_channelwise():
    rng = np.random.default_rng(4)
    target = rng.normal(0, 0.1, (2000, 3))
    interferer = rng.normal(0, 0.03, (2000, 3))
    mixture = target + interferer
    metrics = mixture_additivity_metrics(target, interferer, mixture)
    assert len(metrics["channels"]) == 3
    assert metrics["correlation_min"] > 0.999999
    assert metrics["residual_db_max"] < -150


def test_split_leakage_audit_finds_content_and_physical_group_overlap():
    samples = [
        {
            "dataset_sample_id": "a",
            "dataset_split": "train",
            "target_source_sha256": "same-target",
            "split_group_id": "same-wearing",
        },
        {
            "dataset_sample_id": "b",
            "dataset_split": "test",
            "target_source_sha256": "same-target",
            "split_group_id": "same-wearing",
        },
    ]
    report = audit_split_leakage(samples)
    assert report["blocking_issue_count"] == 2
    assert {item["group_key"] for item in report["issues"]} == {
        "target_source_sha256",
        "split_group_id",
    }
