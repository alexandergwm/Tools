"""Experiment-plan expansion and per-experiment RIR dataset packaging."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from .config import load_config, save_config


EXPERIMENT_FIELDS = (
    "headset_model_id",
    "headset_unit_id",
    "wearing_id",
    "boom_pose_id",
    "source_role",
    "source_id",
    "azimuth_deg",
    "elevation_deg",
    "source_height_cm",
    "distance_cm",
    "dataset_split",
    "output_channel",
)

def _token(value: Any, fallback: str = "na") -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def _signed_token(prefix: str, value: Any) -> str:
    number = int(round(float(value)))
    return f"{prefix}{'p' if number >= 0 else 'm'}{abs(number):03d}"


def experiment_id(metadata: dict[str, Any]) -> str:
    """Build a stable identifier for one independently measured experiment."""
    identifier = metadata.get("experiment_id") or metadata.get("condition_id")
    if identifier:
        text = str(identifier).strip().lower()
        # Manual GUI names may contain Chinese characters.  Keep Unicode word
        # characters while still replacing path punctuation and whitespace.
        text = re.sub(r"[^\w-]+", "-", text, flags=re.UNICODE).strip("-_")
        return text or "experiment"
    required = (
        "headset_model_id",
        "headset_unit_id",
        "wearing_id",
        "boom_pose_id",
        "source_role",
        "source_id",
        "azimuth_deg",
        "elevation_deg",
        "distance_cm",
    )
    missing = [name for name in required if metadata.get(name) in (None, "")]
    if missing:
        raise ValueError(f"experiment metadata is missing: {', '.join(missing)}")
    parts = [
        f"hm-{_token(metadata['headset_model_id'])}",
        f"hu-{_token(metadata['headset_unit_id'])}",
        f"w-{_token(metadata['wearing_id'])}",
        f"b-{_token(metadata['boom_pose_id'])}",
        f"src-{_token(metadata['source_role'])}-{_token(metadata['source_id'])}",
        _signed_token("az", metadata["azimuth_deg"]),
        _signed_token("el", metadata["elevation_deg"]),
        f"d{int(round(float(metadata['distance_cm']))):03d}",
    ]
    return "__".join(parts)


def condition_id(metadata: dict[str, Any]) -> str:
    """Backward-compatible alias for older callers."""
    return experiment_id(metadata)


def load_experiment_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle) or {}
    if plan.get("schema_version") != 1:
        raise ValueError("experiment plan schema_version must be 1")
    project = plan.get("project") or {}
    if not project.get("project_id"):
        raise ValueError("experiment plan requires project.project_id")
    if not plan.get("experiments") and plan.get("conditions"):
        plan["experiments"] = plan.pop("conditions")
    if not plan.get("experiments") and plan.get("matrix"):
        matrix = plan["matrix"]
        expanded = []
        for headset in matrix.get("headset_units", []):
            for wearing in matrix.get("wearings", []):
                for source_set in matrix.get("source_sets", []):
                    source_common = {
                        key: value
                        for key, value in source_set.items()
                        if key not in {"boom_poses", "poses"}
                    }
                    for boom in source_set.get("boom_poses", []):
                        boom_values = boom if isinstance(boom, dict) else {"boom_pose_id": boom}
                        for pose in source_set.get("poses", []):
                            expanded.append(
                                {
                                    **headset,
                                    **wearing,
                                    **source_common,
                                    **boom_values,
                                    **pose,
                                }
                            )
        plan["experiments"] = expanded
    if not plan.get("experiments"):
        raise ValueError("experiment plan requires experiments or a non-empty matrix")

    paths = plan.setdefault("paths", {})
    for key, default in (
        ("base_config", "lab_rir_base.yaml"),
        ("generated_configs", "generated_rir_configs"),
        ("runs_root", "../runs"),
        ("dataset_root", "../datasets/rir_dataset"),
    ):
        value = Path(paths.get(key, default))
        paths[key] = str(value if value.is_absolute() else (plan_path.parent / value).resolve())

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in plan["experiments"]:
        item = dict(raw)
        item.setdefault("dataset_split", "train")
        item.setdefault("source_height_cm", "")
        item["experiment_id"] = experiment_id(item)
        if item["experiment_id"] in seen:
            raise ValueError(f"duplicate experiment_id: {item['experiment_id']}")
        if item["source_role"] not in {"mouth", "interferer"}:
            raise ValueError("source_role must be mouth or interferer")
        if item["dataset_split"] not in {"train", "valid", "test"}:
            raise ValueError("dataset_split must be train, valid, or test")
        seen.add(item["experiment_id"])
        normalized.append(item)
    plan["experiments"] = normalized
    plan["_plan_path"] = str(plan_path)
    return plan


def expand_experiment_plan(path: str | Path) -> list[Path]:
    """Create one ordinary capture YAML per independent experiment."""
    plan = load_experiment_plan(path)
    base_path = Path(plan["paths"]["base_config"])
    output_dir = Path(plan["paths"]["generated_configs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    project_metadata = dict(plan["project"])
    written: list[Path] = []
    for order, experiment in enumerate(plan["experiments"], 1):
        config = load_config(base_path)
        metadata = dict(config.metadata)
        metadata.update(project_metadata)
        metadata.update(experiment)
        metadata["experiment_order"] = order
        config.metadata = metadata
        config.audio.target_output_channel = int(experiment["output_channel"])
        config.storage.root = plan["paths"]["runs_root"]
        config.storage.session_name = experiment["experiment_id"]
        config.validate()
        destination = output_dir / f"{order:03d}_{experiment['experiment_id']}.yaml"
        save_config(config, destination)
        written.append(destination)
    return written


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compile_rir_dataset(path: str | Path) -> Path:
    """Package one within-experiment mean IR per microphone, without cross-experiment averaging."""
    plan = load_experiment_plan(path)
    runs_root = Path(plan["paths"]["runs_root"])
    dataset_root = Path(plan["paths"]["dataset_root"])
    project_id = str(plan["project"]["project_id"])
    planned = {item["experiment_id"]: item for item in plan["experiments"]}

    candidates: dict[str, list[tuple[str, Path, dict[str, Any]]]] = defaultdict(list)
    for manifest_path in runs_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest.get("metadata") or {}
            if (
                manifest.get("kind") != "rir"
                or manifest.get("status") != "completed"
                or str(metadata.get("project_id")) != project_id
            ):
                continue
            eid = experiment_id(metadata)
            if eid in planned and (manifest_path.parent / "processed/average_rir.wav").is_file():
                candidates[eid].append((str(manifest.get("created_at", "")), manifest_path.parent, manifest))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    experiment_rows: list[dict[str, Any]] = []
    for item in plan["experiments"]:
        eid = item["experiment_id"]
        row = {"experiment_order": len(experiment_rows) + 1, "experiment_id": eid, **item}
        matches = sorted(candidates.get(eid, []), key=lambda entry: entry[0])
        row["duplicate_completed_runs"] = max(0, len(matches) - 1)
        if not matches:
            row.update(
                {
                    "status": "missing",
                    "run_id": "",
                    "mean_ir_2ch": "",
                    "mean_ir_mic_01": "",
                    "mean_ir_mic_02": "",
                }
            )
            experiment_rows.append(row)
            continue
        _created, run_dir, manifest = matches[-1]
        source = run_dir / "processed/average_rir.wav"
        split = item["dataset_split"]
        target_dir = dataset_root / "rir" / "experiments" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{eid}__mean-ir-2ch.wav"
        shutil.copy2(source, target)
        audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
        microphone_paths = []
        for channel in range(audio.shape[1]):
            microphone_path = target_dir / f"{eid}__mean-ir-mic-{channel + 1:02d}.wav"
            sf.write(microphone_path, audio[:, channel], sample_rate, subtype="FLOAT")
            microphone_paths.append(microphone_path.relative_to(dataset_root).as_posix())
        summary = manifest.get("summary") or {}
        row.update(
            {
                "status": "completed",
                "run_id": run_dir.name,
                "created_at": manifest.get("created_at", ""),
                "accepted_take_count": len(summary.get("accepted_takes", [])),
                "rejected_take_count": len(summary.get("rejected_takes", [])),
                "sample_rate_hz": sample_rate,
                "rir_samples": len(audio),
                "microphone_channels": audio.shape[1],
                "mean_ir_2ch": target.relative_to(dataset_root).as_posix(),
                "mean_ir_mic_01": microphone_paths[0] if microphone_paths else "",
                "mean_ir_mic_02": microphone_paths[1] if len(microphone_paths) > 1 else "",
                "mean_ir_files_json": json.dumps(microphone_paths, ensure_ascii=False),
                "source_run": str(run_dir),
            }
        )
        experiment_rows.append(row)

    _write_rows(dataset_root / "indexes" / "rir_experiments", experiment_rows)
    summary = {
        "schema_version": 1,
        "project": plan["project"],
        "planned_experiments": len(plan["experiments"]),
        "completed_experiments": sum(row["status"] == "completed" for row in experiment_rows),
        "missing_experiments": [
            row["experiment_id"] for row in experiment_rows if row["status"] == "missing"
        ],
        "training_unit": "mean_ir_per_microphone",
        "averaging_scope": "accepted_takes_within_one_experiment_only",
        "cross_experiment_averaging": False,
    }
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dataset_root
