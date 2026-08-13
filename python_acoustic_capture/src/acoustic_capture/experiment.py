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
from .professional import array_geometry_sha256, array_metadata


EXPERIMENT_FIELDS = (
    "artificial_head_id",
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
        *(
            [f"ah-{_token(metadata['artificial_head_id'])}"]
            if metadata.get("artificial_head_id") not in (None, "")
            else []
        ),
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


def _physical_group_id(metadata: dict[str, Any]) -> str:
    parts = [
        str(metadata.get(key) or "")
        for key in (
            "room_id",
            "artificial_head_id",
            "headset_unit_id",
            "wearing_id",
            "boom_pose_id",
        )
    ]
    return "|".join(parts) if any(parts) else ""


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
        artificial_heads = matrix.get("artificial_heads") or [{}]
        for artificial_head in artificial_heads:
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
                                        **artificial_head,
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
    # This directory is compiler output.  Remove only numbered YAMLs created
    # by previous expansions so renamed metadata fields cannot leave stale,
    # duplicate experiments behind.
    for old_path in output_dir.glob("*.yaml"):
        if re.match(r"^\d{3}_.*\.yaml$", old_path.name):
            old_path.unlink()
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


def _write_rir_workbook(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        import xlsxwriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 XlsxWriter，无法生成 RIR 数据集汇总表") from exc
    workbook = xlsxwriter.Workbook(path)
    header = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"}
    )
    wrapped = workbook.add_format({"valign": "top", "text_wrap": True})
    section = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#548235"}
    )
    sheet = workbook.add_worksheet("RIR实验")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    for column, key in enumerate(keys):
        sheet.write(0, column, key, header)
    for row_index, row in enumerate(rows, 1):
        for column, key in enumerate(keys):
            value = row.get(key, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            sheet.write(row_index, column, value, wrapped)
    if keys:
        sheet.autofilter(0, 0, max(1, len(rows)), len(keys) - 1)
        sheet.freeze_panes(1, 2)
        sheet.set_column(0, len(keys) - 1, 20)
        for key in (
            "experiment_id",
            "dataset_experiment_id",
            "mean_ir_multichannel",
            "source_run",
        ):
            if key in keys:
                column = keys.index(key)
                sheet.set_column(column, column, 38)
    summary_sheet = workbook.add_worksheet("汇总")
    summary_sheet.set_column("A:A", 32)
    summary_sheet.set_column("B:B", 24)
    summary_sheet.write("A1", "RIR 数据集汇总", section)
    for row_index, (key, value) in enumerate(summary.items(), 3):
        summary_sheet.write(row_index - 1, 0, key)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        summary_sheet.write(row_index - 1, 1, value, wrapped)
    workbook.close()


def compile_completed_rir_runs(
    runs_root: str | Path,
    dataset_root: str | Path,
    *,
    project_id: str | None = None,
) -> Path:
    """Package every completed manual RIR experiment without using a plan.

    Each completed run remains an independent experiment.  No IR samples are
    ever averaged across runs, even when two manual experiment names match.
    """
    runs_root = Path(runs_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    rows: list[dict[str, Any]] = []
    names: dict[str, int] = defaultdict(int)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        metadata = manifest.get("metadata") or {}
        if manifest.get("kind") != "rir" or manifest.get("status") != "completed":
            continue
        if project_id is not None and str(metadata.get("project_id", "")) != project_id:
            continue
        if not (manifest_path.parent / "processed" / "average_rir.wav").is_file():
            continue
        candidates.append((manifest_path.parent, manifest))
        candidate_name = str(
            metadata.get("experiment_id")
            or metadata.get("experiment_name")
            or manifest_path.parent.name
        )
        names[candidate_name] += 1

    for run_dir, manifest in candidates:
        metadata = manifest.get("metadata") or {}
        eid = str(
            metadata.get("experiment_id")
            or metadata.get("experiment_name")
            or run_dir.name
        )
        run_id = run_dir.name
        dataset_eid = f"{run_id}__{eid}"
        split = str(metadata.get("dataset_split") or "train")
        if split not in {"train", "valid", "test"}:
            split = "train"
        destination_dir = dataset_root / "rir" / "experiments" / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = run_dir / "processed" / "average_rir.wav"
        target = destination_dir / f"{run_id}__mean-ir-multichannel.wav"
        shutil.copy2(source, target)
        audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
        microphone_paths: list[str] = []
        for channel in range(audio.shape[1]):
            microphone_path = destination_dir / f"{run_id}__mean-ir-mic-{channel + 1:02d}.wav"
            sf.write(microphone_path, audio[:, channel], sample_rate, subtype="FLOAT")
            microphone_paths.append(microphone_path.relative_to(dataset_root).as_posix())
        summary = manifest.get("summary") or {}
        row = {
            "dataset_experiment_id": dataset_eid,
            "experiment_id": eid,
            "duplicate_manual_name_count": names[eid],
            "run_id": run_id,
            "status": "completed",
            "created_at": manifest.get("created_at", ""),
            "dataset_split": split,
            **{field: metadata.get(field, "") for field in EXPERIMENT_FIELDS},
            "project_id": metadata.get("project_id", ""),
            "run_uuid": manifest.get("run_uuid", ""),
            "config_sha256": manifest.get("config_sha256", ""),
            "capture_profile": metadata.get("capture_profile", "standard"),
            "task_type": metadata.get("task_type", metadata.get("experiment_type", "")),
            "array_id": array_metadata(metadata).get("array_id", ""),
            "array_geometry_sha256": array_geometry_sha256(metadata),
            "array_geometry_json": json.dumps(array_metadata(metadata), ensure_ascii=False),
            "physical_capture_group_id": _physical_group_id(metadata),
            "split_group_id": metadata.get(
                "split_group_id", _physical_group_id(metadata)
            ),
            "accepted_take_count": len(summary.get("accepted_takes", [])),
            "rejected_take_count": len(summary.get("rejected_takes", [])),
            "sample_rate_hz": sample_rate,
            "rir_samples": len(audio),
            "microphone_channels": audio.shape[1],
            "mean_ir_multichannel": target.relative_to(dataset_root).as_posix(),
            "mean_ir_files_json": json.dumps(microphone_paths, ensure_ascii=False),
            "source_run": str(run_dir),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }
        for index, microphone_path in enumerate(microphone_paths, 1):
            row[f"mean_ir_mic_{index:02d}"] = microphone_path
        rows.append(row)

    index_base = dataset_root / "indexes" / "rir_experiments"
    _write_rows(index_base, rows)
    dataset_summary = {
        "schema_version": 2,
        "kind": "rir_dataset_from_completed_runs",
        "project_id_filter": project_id,
        "source_runs_root": str(runs_root),
        "completed_experiments": len(rows),
        "training_unit": "within_experiment_mean_ir_per_microphone",
        "averaging_scope": "accepted_takes_within_one_run_only",
        "cross_experiment_averaging": False,
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_rir_workbook(dataset_root / "indexes" / "rir_dataset.xlsx", rows, dataset_summary)
    return dataset_root


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
                    "mean_ir_multichannel": "",
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
        target = target_dir / f"{eid}__mean-ir-multichannel.wav"
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
                "run_uuid": manifest.get("run_uuid", ""),
                "config_sha256": manifest.get("config_sha256", ""),
                "capture_profile": (manifest.get("metadata") or {}).get(
                    "capture_profile", "standard"
                ),
                "task_type": (manifest.get("metadata") or {}).get(
                    "task_type", (manifest.get("metadata") or {}).get("experiment_type", "")
                ),
                "array_id": array_metadata(manifest.get("metadata") or {}).get(
                    "array_id", ""
                ),
                "array_geometry_sha256": array_geometry_sha256(
                    manifest.get("metadata") or {}
                ),
                "array_geometry_json": json.dumps(
                    array_metadata(manifest.get("metadata") or {}), ensure_ascii=False
                ),
                "physical_capture_group_id": _physical_group_id(
                    manifest.get("metadata") or {}
                ),
                "split_group_id": (manifest.get("metadata") or {}).get(
                    "split_group_id",
                    _physical_group_id(manifest.get("metadata") or {}),
                ),
                "created_at": manifest.get("created_at", ""),
                "accepted_take_count": len(summary.get("accepted_takes", [])),
                "rejected_take_count": len(summary.get("rejected_takes", [])),
                "sample_rate_hz": sample_rate,
                "rir_samples": len(audio),
                "microphone_channels": audio.shape[1],
                "mean_ir_multichannel": target.relative_to(dataset_root).as_posix(),
                "mean_ir_2ch": (
                    target.relative_to(dataset_root).as_posix()
                    if audio.shape[1] == 2
                    else ""
                ),
                "mean_ir_mic_01": microphone_paths[0] if microphone_paths else "",
                "mean_ir_mic_02": microphone_paths[1] if len(microphone_paths) > 1 else "",
                "mean_ir_files_json": json.dumps(microphone_paths, ensure_ascii=False),
                "source_run": str(run_dir),
            }
        )
        experiment_rows.append(row)

    _write_rows(dataset_root / "indexes" / "rir_experiments", experiment_rows)
    summary = {
        "schema_version": 2,
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
    _write_rir_workbook(
        dataset_root / "indexes" / "rir_dataset.xlsx", experiment_rows, summary
    )
    return dataset_root
