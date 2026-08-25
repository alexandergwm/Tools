"""Group named RIR/speech runs into one operator-controlled experiment.

A campaign keeps the full child runs on disk for traceability.  Its final ZIP
is deliberately smaller: RIR raw sweeps, playback references and continuous
speech paired-sequence recordings are reproducible/intermediate artifacts and
are therefore excluded from the transfer package.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Callable
import zipfile

from .storage import _safe_name


StopRequested = Callable[[], bool]
Progress = Callable[[dict], None]


class CampaignPackagingCancelled(Exception):
    pass


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def create_campaign(storage_root: str | Path, name: str) -> Path:
    """Create one large experiment whose ``runs`` hold named child tests."""
    clean_name = _safe_name(name)
    if not name.strip():
        raise ValueError("大实验名称不能为空")
    parent = Path(storage_root).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    root = parent / f"{stamp}_{clean_name}_big_experiment"
    suffix = 2
    while root.exists():
        root = parent / f"{stamp}_{clean_name}_big_experiment_{suffix:02d}"
        suffix += 1
    (root / "runs").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "kind": "acoustic_capture_big_experiment",
        "name": name.strip(),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_directory": "runs",
        "package_policy": {
            "one_zip_for_entire_experiment": True,
            "rir_raw_audio_included": False,
            "playback_references_included": False,
            "speech_paired_sequence_mics_included": False,
        },
    }
    _write_json(root / "big_experiment.json", manifest)
    return root


def load_campaign(path: str | Path) -> dict:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "big_experiment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"不是大实验目录：{root}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("kind") != "acoustic_capture_big_experiment":
        raise ValueError(f"大实验清单类型无效：{root}")
    return value


def _run_rows(campaign_root: Path) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in sorted((campaign_root / "runs").glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            continue
        summary = manifest.get("summary") or {}
        metadata = manifest.get("metadata") or {}
        run_root = manifest_path.parent
        label_count = int(summary.get("label_rows") or 0)
        rows.append(
            {
                "run_name": run_root.name,
                "test_name": metadata.get("experiment_name", run_root.name),
                "kind": manifest.get("kind", ""),
                "status": manifest.get("status", ""),
                "created_at": manifest.get("created_at", ""),
                "finished_at": manifest.get("finished_at", ""),
                "input_channels": ",".join(
                    map(str, manifest.get("audio_input_channels") or [])
                ),
                "label_rows": label_count,
                "supervision_ready_rows": int(
                    summary.get("supervision_ready_rows") or 0
                ),
                "accepted_rir_takes": len(
                    summary.get("accepted_take_ids")
                    or summary.get("selected_take_ids")
                    or []
                ),
                "final_rir": summary.get("mean_rir_2ch", ""),
                "headset_unit_id": metadata.get("headset_unit_id", ""),
                "wearing_id": metadata.get("wearing_id", ""),
                "boom_pose_id": metadata.get("boom_pose_id", ""),
                "source_role": metadata.get("source_role", ""),
                "azimuth_deg": metadata.get("azimuth_deg", ""),
                "source_height_cm": metadata.get("source_height_cm", ""),
            }
        )
    return rows


def _write_campaign_index(root: Path, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else ["run_name", "test_name", "kind", "status"]
    with (root / "big_experiment_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "big_experiment_runs.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        import xlsxwriter

        workbook = xlsxwriter.Workbook(root / "big_experiment_runs.xlsx")
        sheet = workbook.add_worksheet("测试汇总")
        header = workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"}
        )
        for column, key in enumerate(keys):
            sheet.write(0, column, key, header)
        for row_index, row in enumerate(rows, 1):
            for column, key in enumerate(keys):
                sheet.write(row_index, column, row.get(key, ""))
        if keys:
            sheet.autofilter(0, 0, max(1, len(rows)), len(keys) - 1)
            sheet.freeze_panes(1, 2)
            sheet.set_column(0, len(keys) - 1, 20)
        workbook.close()
    except ImportError:  # pragma: no cover - normal install includes XlsxWriter
        pass


def _include_run_file(run_root: Path, path: Path, kind: str) -> tuple[bool, str]:
    relative = path.relative_to(run_root)
    parts = relative.parts
    if not parts:
        return False, "empty_path"
    if parts[0] == "references":
        return False, "playback_or_source_reference"
    if kind == "rir":
        if parts[0] == "raw":
            return False, "rir_raw_audio"
        if parts[0] == "processed":
            name = path.name
            keep = name == "average_rir.wav" or (
                name.startswith("average_rir_mic_") and name.endswith(".wav")
            )
            return keep, "final_rir_only" if not keep else ""
    if kind == "scene" and parts[0] == "raw":
        if path.name.endswith("_paired_sequence_mics.wav"):
            return False, "continuous_paired_sequence_recording"
    return True, ""


def package_campaign(
    campaign_root: str | Path,
    *,
    stop_requested: StopRequested | None = None,
    progress: Progress | None = None,
) -> dict:
    """Finish a campaign and create exactly one selective ZIP beside it."""
    root = Path(campaign_root).expanduser().resolve()
    manifest = load_campaign(root)
    rows = _run_rows(root)
    if not rows:
        raise ValueError("大实验中还没有任何已开始的 RIR 或语音增强测试")
    _write_campaign_index(root, rows)
    candidates: list[tuple[Path, str]] = []
    excluded_counts: dict[str, int] = {}
    for run_row in rows:
        run_root = root / "runs" / str(run_row["run_name"])
        kind = str(run_row["kind"])
        for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
            included, reason = _include_run_file(run_root, path, kind)
            if included:
                candidates.append((path, path.relative_to(root).as_posix()))
            else:
                excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
    campaign_files = [
        root / "big_experiment.json",
        root / "big_experiment_runs.csv",
        root / "big_experiment_runs.jsonl",
    ]
    xlsx = root / "big_experiment_runs.xlsx"
    if xlsx.is_file():
        campaign_files.append(xlsx)
    total_files = len(campaign_files) + len(candidates)
    final_zip = root.parent / f"{root.name}.zip"
    partial_zip = root.parent / f"{root.name}.zip.partial"
    if final_zip.exists():
        raise FileExistsError(f"大实验压缩包已经存在：{final_zip}")
    source_bytes = sum(path.stat().st_size for path, _ in candidates)
    required_free = source_bytes + max(64 * 1024**2, source_bytes // 50)
    disk_free = shutil.disk_usage(root.parent).free
    if disk_free < required_free:
        raise OSError(
            "大实验打包空间不足："
            f"可用 {disk_free / 1024**3:.2f} GiB，至少需要约 "
            f"{required_free / 1024**3:.2f} GiB"
        )
    manifest.update(
        {
            "status": "packaged",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "run_count": len(rows),
            "run_status_counts": {
                status: sum(row["status"] == status for row in rows)
                for status in sorted({str(row["status"]) for row in rows})
            },
            "included_file_count": total_files,
            "excluded_file_counts": excluded_counts,
        }
    )
    _write_json(root / "big_experiment.json", manifest)
    try:
        with zipfile.ZipFile(
            partial_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            entries = [
                (path, path.relative_to(root).as_posix()) for path in campaign_files
            ] + candidates
            for index, (path, relative) in enumerate(entries, 1):
                if stop_requested is not None and stop_requested():
                    raise CampaignPackagingCancelled("用户停止了大实验打包")
                archive.write(
                    path,
                    f"{root.name}/{relative}",
                    compress_type=(
                        zipfile.ZIP_STORED
                        if path.suffix.casefold() in {".wav", ".rf64"}
                        else zipfile.ZIP_DEFLATED
                    ),
                )
                if progress is not None:
                    progress(
                        {
                            "file_index": index,
                            "file_count": len(entries),
                            "relative_path": relative,
                        }
                    )
        partial_zip.replace(final_zip)
    except Exception:
        if partial_zip.exists():
            partial_zip.unlink()
        manifest["status"] = "active"
        manifest.pop("finished_at", None)
        _write_json(root / "big_experiment.json", manifest)
        raise
    return {
        "root": str(root),
        "zip": str(final_zip),
        "run_count": len(rows),
        "included_file_count": total_files,
        "excluded_file_counts": excluded_counts,
        "zip_bytes": final_zip.stat().st_size,
        "source_bytes": source_bytes,
        "disk_free_bytes": shutil.disk_usage(final_zip.parent).free,
    }
