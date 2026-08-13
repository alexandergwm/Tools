"""Package completed scene runs into a portable speech-enhancement dataset."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf


RECORDING_KEYS = (
    "ambient_recording",
    "target_recording",
    "interferer_recording",
    "mixture_recording",
)

DEFAULT_SPLIT_GROUP_KEYS = (
    "target_source_sha256",
    "split_group_id",
    "speaker_id",
    "utterance_id",
)

TRAINING_INDEX_KEYS = (
    "dataset_sample_id",
    "supervision_pair_id",
    "dataset_split",
    "dataset_supervision_ready",
    "quality_flag",
    "mixture_recording",
    "target_recording",
    "interferer_recording",
    "sample_rate_hz",
    "microphone_channels",
    "duration_s",
    "scene_id",
    "project_id",
    "room_id",
    "artificial_head_id",
    "headset_model_id",
    "headset_unit_id",
    "wearing_id",
    "boom_pose_id",
    "array_id",
    "array_geometry_sha256",
    "split_group_id",
    "speaker_id",
    "utterance_id",
    "target_source_sha256",
    "interferer_source_sha256",
    "mixture_consistency_residual_db_max",
    "mixture_consistency_correlation_min",
    "source_run",
)


def _safe_token(value: Any) -> str:
    text = re.sub(r"[^\w.-]+", "_", str(value).strip(), flags=re.UNICODE).strip("._")
    return text or "sample"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


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
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def _resolve_run_file(run_dir: Path, relative: Any) -> Path | None:
    if relative in (None, ""):
        return None
    root = run_dir.resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"recording path escapes run directory: {relative}")
    return path


def _audio_signature(path: Path | None) -> tuple[int, int, int] | None:
    if path is None or not path.is_file():
        return None
    info = sf.info(path)
    return info.samplerate, info.frames, info.channels


def _supervision_error(row: dict[str, Any], run_dir: Path) -> str:
    if row.get("supervision_ready") != "是":
        return "source row is not supervision-ready"
    if row.get("shared_hardware_clock") != "是":
        return "input/output shared hardware clock was not verified"
    target = _resolve_run_file(run_dir, row.get("target_recording"))
    mixture = _resolve_run_file(run_dir, row.get("mixture_recording"))
    target_signature = _audio_signature(target)
    mixture_signature = _audio_signature(mixture)
    if target_signature is None or mixture_signature is None:
        return "target or mixture recording is missing"
    if target_signature != mixture_signature:
        return (
            "target/mixture shape mismatch: "
            f"target={target_signature}, mixture={mixture_signature}"
        )
    return ""


def _package_recordings(
    row: dict[str, Any], run_dir: Path, dataset_root: Path, run_id: str
) -> None:
    for key in RECORDING_KEYS:
        original = row.get(key, "")
        row[f"source_{key}"] = original
        source = _resolve_run_file(run_dir, original)
        if source is None:
            continue
        if not source.is_file():
            row["dataset_validation_error"] = (
                row.get("dataset_validation_error") or f"missing file: {original}"
            )
            continue
        relative = Path("audio") / run_id / Path(str(original))
        destination = dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            not destination.exists()
            or destination.stat().st_size != source.stat().st_size
            or destination.stat().st_mtime_ns < source.stat().st_mtime_ns
        ):
            shutil.copy2(source, destination)
        row[key] = relative.as_posix()


def audit_split_leakage(
    samples: list[dict[str, Any]],
    group_keys: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Report identities that occur in more than one dataset split."""
    keys = tuple(group_keys or DEFAULT_SPLIT_GROUP_KEYS)
    issues: list[dict[str, Any]] = []
    for key in keys:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in samples:
            value = str(row.get(key) or "").strip()
            split = str(row.get("dataset_split") or "").strip().lower()
            if value and split in {"train", "valid", "test"}:
                groups[value].append(row)
        for value, members in groups.items():
            splits = sorted({str(row.get("dataset_split")) for row in members})
            if len(splits) < 2:
                continue
            issues.append(
                {
                    "group_key": key,
                    "group_value": value,
                    "splits": ",".join(splits),
                    "sample_count": len(members),
                    "sample_ids": [
                        row.get("dataset_sample_id", row.get("sample_id", ""))
                        for row in members
                    ],
                    "severity": (
                        "error"
                        if key in {"target_source_sha256", "split_group_id"}
                        else "warning"
                    ),
                    "explanation": (
                        "相同目标素材内容跨数据划分"
                        if key == "target_source_sha256"
                        else "相同物理佩戴/阵列条件跨数据划分"
                        if key == "split_group_id"
                        else f"相同 {key} 跨数据划分；是否允许取决于训练协议"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "group_keys": list(keys),
        "issue_count": len(issues),
        "blocking_issue_count": sum(issue["severity"] == "error" for issue in issues),
        "issues": issues,
    }


def _write_workbook(
    path: Path,
    samples: list[dict[str, Any]],
    supervised: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
) -> None:
    try:
        import xlsxwriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 XlsxWriter，无法生成语音数据集汇总表") from exc

    workbook = xlsxwriter.Workbook(path)
    header = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"}
    )
    wrapped = workbook.add_format({"valign": "top", "text_wrap": True})
    section = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#548235"}
    )

    def add_table(
        name: str,
        rows: list[dict[str, Any]],
        preferred_keys: tuple[str, ...] | None = None,
    ) -> None:
        sheet = workbook.add_worksheet(name)
        sheet.hide_gridlines(2)
        keys: list[str] = (
            [key for key in preferred_keys if any(key in row for row in rows)]
            if preferred_keys
            else []
        )
        if not preferred_keys:
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
                "dataset_sample_id",
                "sample_id",
                "mixture_recording",
                "target_recording",
                "interferer_recording",
            ):
                if key in keys:
                    column = keys.index(key)
                    sheet.set_column(column, column, 36)

    add_table("训练索引", supervised, TRAINING_INDEX_KEYS)
    add_table("全部样本（高级）", samples)
    add_table("运行记录", run_rows)
    add_table(
        "划分泄漏检查",
        leakage_rows
        or [{"status": "通过", "message": "未发现目标内容或物理采集分组跨 train/valid/test。"}],
    )
    summary = workbook.add_worksheet("汇总")
    summary.hide_gridlines(2)
    summary.set_column("A:A", 30)
    summary.set_column("B:B", 18)
    summary.write("A1", "语音增强数据集汇总", section)
    values = (
        ("完成运行数", len(run_rows)),
        ("样本总数", len(samples)),
        ("合格监督配对数", len(supervised)),
        ("纯目标样本数", sum(row.get("capture_type") == "target_only" for row in samples)),
        ("纯干扰样本数", sum(row.get("capture_type") == "interferer_only" for row in samples)),
        ("校验异常样本数", sum(bool(row.get("dataset_validation_error")) for row in samples)),
        ("数据划分泄漏项", len(leakage_rows)),
        (
            "阻止训练的泄漏项",
            sum(row.get("severity") == "error" for row in leakage_rows),
        ),
    )
    for row_index, (label, value) in enumerate(values, 3):
        summary.write(row_index - 1, 0, label)
        summary.write_number(row_index - 1, 1, value)
    workbook.close()


def compile_speech_dataset(
    runs_root: str | Path,
    dataset_root: str | Path,
    *,
    project_id: str | None = None,
    copy_audio: bool = True,
    split_group_keys: tuple[str, ...] | list[str] | None = None,
    fail_on_split_leakage: bool = False,
) -> Path:
    """Combine completed speech runs and validate every supervised pair.

    Audio is copied once under ``audio/<run_id>/`` so paths remain portable
    without duplicating ambient recordings for every source pair.
    """
    runs_root = Path(runs_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        metadata = manifest.get("metadata") or {}
        if manifest.get("kind") != "scene" or manifest.get("status") != "completed":
            continue
        if project_id is not None and str(metadata.get("project_id", "")) != project_id:
            continue
        run_dir = manifest_path.parent
        reviewed_label_path = run_dir / "labels_reviewed.jsonl"
        label_path = reviewed_label_path if reviewed_label_path.is_file() else run_dir / "labels.jsonl"
        if not label_path.is_file():
            continue
        run_id = _safe_token(run_dir.name)
        rows = _read_jsonl(label_path)
        run_rows.append(
            {
                "run_id": run_id,
                "created_at": manifest.get("created_at", ""),
                "scene_id": metadata.get("scene_id", ""),
                "project_id": metadata.get("project_id", ""),
                "status": manifest.get("status", ""),
                "sample_count": len(rows),
                "source_run": str(run_dir),
            }
        )
        for row in rows:
            row = dict(row)
            row["run_id"] = run_id
            row["dataset_sample_id"] = f"{run_id}__{_safe_token(row.get('sample_id'))}"
            row["source_run"] = str(run_dir)
            error = _supervision_error(row, run_dir) if row.get("mixture_recording") else ""
            row["dataset_validation_error"] = error
            row["dataset_supervision_ready"] = (
                "是"
                if not error
                and row.get("supervision_ready") == "是"
                and row.get("valid") == "是"
                else "否"
            )
            if copy_audio:
                _package_recordings(row, run_dir, dataset_root, run_id)
            samples.append(row)

    leakage_report = audit_split_leakage(samples, split_group_keys)
    supervised = [
        row for row in samples if row.get("dataset_supervision_ready") == "是"
    ]
    indexes = dataset_root / "indexes"
    _write_rows(indexes / "speech_samples", samples)
    _write_rows(indexes / "supervised_pairs", supervised)
    _write_rows(indexes / "speech_runs", run_rows)
    _write_rows(indexes / "split_leakage", leakage_report["issues"])
    (indexes / "split_leakage_report.json").write_text(
        json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_workbook(
        indexes / "speech_dataset.xlsx",
        samples,
        supervised,
        run_rows,
        leakage_report["issues"],
    )
    validation_error_count = sum(
        bool(row.get("dataset_validation_error")) for row in samples
    )
    manifest = {
        "schema_version": 2,
        "kind": "speech_enhancement_dataset",
        "project_id_filter": project_id,
        "source_runs_root": str(runs_root),
        "audio_packaged": copy_audio,
        "completed_runs": len(run_rows),
        "sample_count": len(samples),
        "supervised_pair_count": len(supervised),
        "validation_error_count": validation_error_count,
        "split_leakage_issue_count": leakage_report["issue_count"],
        "split_leakage_blocking_count": leakage_report["blocking_issue_count"],
        "training_ready": (
            validation_error_count == 0
            and leakage_report["blocking_issue_count"] == 0
            and bool(supervised)
        ),
        "supervised_input": "mixture_recording",
        "supervised_target": "target_recording",
        "microphone_storage": "one_multichannel_wav_per_recording",
        "split_leakage_rule": "group by exact target content and physical capture group before train/valid/test split",
    }
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if fail_on_split_leakage and leakage_report["blocking_issue_count"]:
        raise ValueError(
            "dataset split leakage detected; see indexes/split_leakage_report.json"
        )
    return dataset_root
