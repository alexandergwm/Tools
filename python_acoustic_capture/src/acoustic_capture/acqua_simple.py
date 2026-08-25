"""Minimal ACQUA workflow: generate one program, then only record it.

The program alternates ``mixed`` and the matching ``target_only`` block.  No
marker detection or external-control state machine is involved.  The mapping
table describes the source program timeline; the long microphone WAV remains
usable when recording is stopped early.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import soundfile as sf

from .config import ExperimentConfig
from .scene import (
    PAIRING_STRATEGY,
    SourcePair,
    _load_pair,
    _scan_audio_folder,
    _seeded_repeat,
    validate_source_audio,
)
from .storage import _safe_name


StopRequested = Callable[[], bool]
Progress = Callable[[dict], None]


def discover_acqua_pairs(config: ExperimentConfig) -> list[SourcePair]:
    extensions = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in config.scene.file_extensions
    }
    target_root = Path(config.scene.target_folder).expanduser().resolve()
    interferer_root = Path(config.scene.interferer_folder).expanduser().resolve()
    targets = _scan_audio_folder(target_root, extensions)
    interferers = _scan_audio_folder(interferer_root, extensions)
    if not targets:
        raise FileNotFoundError(f"目标语料文件夹中没有音频：{target_root}")
    if not interferers:
        raise FileNotFoundError(f"干扰语料文件夹中没有音频：{interferer_root}")
    count = max(len(targets), len(interferers))
    interferer_plan = _seeded_repeat(
        interferers, count, interferer_root, config.acqua.pairing_seed
    )
    return [
        SourcePair(targets[index % len(targets)], interferer_plan[index])
        for index in range(count)
    ]


def _write_rows(
    root: Path, rows: list[dict], stem: str = "sequence_mapping"
) -> dict[str, Path | None]:
    keys = list(rows[0]) if rows else []
    csv_path = root / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)
    jsonl_path = root / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    xlsx_path = root / f"{stem}.xlsx"
    try:
        import xlsxwriter

        workbook = xlsxwriter.Workbook(xlsx_path)
        sheet = workbook.add_worksheet("ACQUA序列映射")
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
            sheet.freeze_panes(1, 3)
            sheet.set_column(0, len(keys) - 1, 22)
        workbook.close()
    except ImportError:  # pragma: no cover
        xlsx_path = None
    return {"csv": csv_path, "jsonl": jsonl_path, "xlsx": xlsx_path}


def generate_acqua_mixed_target_program(
    config: ExperimentConfig,
    output_parent: str | Path,
    sequence_name: str,
    *,
    stop_requested: StopRequested | None = None,
    progress: Progress | None = None,
) -> dict:
    """Stream a two-channel mixed/target/mixed/target ACQUA program to disk."""
    if not sequence_name.strip():
        raise ValueError("ACQUA 序列名称不能为空")
    config.validate()
    pairs = discover_acqua_pairs(config)
    validate_source_audio(pairs)
    fs = config.audio.sample_rate
    segment_samples = round(config.acqua.segment_duration_s * fs)
    gap_samples = round(config.acqua.gap_s * fs)
    # Initial/final gaps make manual ACQUA start/stop less abrupt without
    # introducing a marker or any marker-dependent processing.
    total_frames = gap_samples + len(pairs) * (2 * segment_samples + 2 * gap_samples)
    bytes_per_sample = {"PCM_16": 2, "PCM_24": 3, "FLOAT": 4}[
        config.acqua.wav_subtype
    ]
    estimated_bytes = total_frames * 2 * bytes_per_sample + 65_536
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < estimated_bytes * 1.1:
        raise OSError(
            f"磁盘空间不足：预计需要 {estimated_bytes / 1024**3:.2f} GiB"
        )
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    package_name = f"{stamp}_{_safe_name(sequence_name)}_acqua_program"
    root = parent / package_name
    suffix = 2
    while root.exists():
        root = parent / f"{package_name}_{suffix:02d}"
        suffix += 1
    root.mkdir(parents=True)
    extension = ".rf64.wav" if estimated_bytes >= 0xFFFFFFFF else ".wav"
    program = root / f"{_safe_name(sequence_name)}_mixed_target{extension}"
    format_name = "RF64" if estimated_bytes >= 0xFFFFFFFF else "WAV"
    rows: list[dict] = []
    cursor = gap_samples
    zero_gap = np.zeros((gap_samples, 2), dtype=np.float32)
    status = "completed"
    with sf.SoundFile(
        program,
        mode="w",
        samplerate=fs,
        channels=2,
        subtype=config.acqua.wav_subtype,
        format=format_name,
    ) as output:
        if gap_samples:
            output.write(zero_gap)
        for pair_index, pair in enumerate(pairs, 1):
            if stop_requested is not None and stop_requested():
                status = "cancelled"
                break
            target, interferer = _load_pair(pair, config, segment_samples)
            mixed = np.column_stack((target, interferer)).astype(np.float32)
            target_only = np.column_stack(
                (target, np.zeros(segment_samples, dtype=np.float32))
            )
            target_path = pair.target.resolve()
            interferer_path = pair.interferer.resolve()
            for stage, block in (("mixed", mixed), ("target_only", target_only)):
                start = cursor
                output.write(block)
                cursor += len(block)
                rows.append(
                    {
                        "sequence_index": len(rows) + 1,
                        "pair_index": pair_index,
                        "stage": stage,
                        "start_sample": start,
                        "end_sample": cursor,
                        "sample_count": len(block),
                        "start_s": start / fs,
                        "end_s": cursor / fs,
                        "target_source": str(target_path),
                        "target_relative_path": target_path.relative_to(
                            Path(config.scene.target_folder).resolve()
                        ).as_posix(),
                        "interferer_source": str(interferer_path),
                        "interferer_relative_path": interferer_path.relative_to(
                            Path(config.scene.interferer_folder).resolve()
                        ).as_posix(),
                        "pairing_seed": config.acqua.pairing_seed,
                        "pairing_strategy": PAIRING_STRATEGY,
                        "sample_rate_hz": fs,
                        "logical_channel_1": "target",
                        "logical_channel_2": (
                            "interferer" if stage == "mixed" else "silence"
                        ),
                    }
                )
                if gap_samples:
                    output.write(zero_gap)
                    cursor += gap_samples
            if progress is not None:
                progress(
                    {
                        "pair_index": pair_index,
                        "pair_count": len(pairs),
                        "target_name": pair.target.name,
                        "interferer_name": pair.interferer.name,
                    }
                )
    mapping_files = _write_rows(root, rows)
    manifest = {
        "schema_version": 1,
        "kind": "acqua_mixed_target_program",
        "status": status,
        "sequence_name": sequence_name.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "program_file": program.name,
        "program_format": format_name,
        "wav_subtype": config.acqua.wav_subtype,
        "sample_rate_hz": fs,
        "channels": 2,
        "logical_channel_map": {"1": "target", "2": "interferer_or_silence"},
        "order": "mixed,target_only,mixed,target_only,...",
        "segment_duration_s": config.acqua.segment_duration_s,
        "gap_s": config.acqua.gap_s,
        "pairing_seed": config.acqua.pairing_seed,
        "pairing_strategy": PAIRING_STRATEGY,
        "planned_pair_count": len(pairs),
        "completed_pair_count": len(rows) // 2,
        "mapping_row_count": len(rows),
        "program_frames": int(sf.info(program).frames),
        "program_duration_s": float(sf.info(program).duration),
        "mapping_files": {
            key: path.name for key, path in mapping_files.items() if path is not None
        },
        "timing_note": (
            "Mapping is the ACQUA source-file timeline. External playback start "
            "offset is not marker-corrected in the simple record-only workflow."
        ),
    }
    (root / "program_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"root": str(root), "program": str(program), **manifest}


def prepare_acqua_recording(
    storage_root: str | Path,
    recording_name: str,
    program_file: str | Path,
    *,
    expected_sample_rate: int | None = None,
) -> dict:
    if not recording_name.strip():
        raise ValueError("ACQUA 录制名称不能为空")
    program = Path(program_file).expanduser().resolve()
    if not program.is_file():
        raise FileNotFoundError(f"找不到 ACQUA 长播放音频：{program}")
    info = sf.info(program)
    if expected_sample_rate is not None and info.samplerate != expected_sample_rate:
        raise ValueError(
            f"长音频采样率是 {info.samplerate} Hz，但录制设置是 "
            f"{expected_sample_rate} Hz；请保持一致，以便按映射表切割。"
        )
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    root = Path(storage_root).expanduser().resolve() / (
        f"{stamp}_{_safe_name(recording_name)}_acqua_recording"
    )
    suffix = 2
    candidate = root
    while candidate.exists():
        candidate = Path(f"{root}_{suffix:02d}")
        suffix += 1
    root = candidate
    (root / "raw").mkdir(parents=True)
    output = root / "raw" / f"{_safe_name(recording_name)}_mics.wav"
    for name in (
        "program_manifest.json",
        "sequence_mapping.csv",
        "sequence_mapping.jsonl",
        "sequence_mapping.xlsx",
    ):
        source = program.parent / name
        if source.is_file():
            shutil.copy2(source, root / name)
    return {
        "root": root,
        "output": output,
        "program": program,
        "program_frames": info.frames,
        "program_duration_s": info.duration,
        "sample_rate_hz": info.samplerate,
    }


def finish_acqua_recording(prepared: dict, status: dict, *, stopped_early: bool) -> Path:
    root = Path(prepared["root"])
    recorded_frames = int(status.get("frames", 0))
    prefix_rows: list[dict] = []
    mapping_path = root / "sequence_mapping.csv"
    if mapping_path.is_file():
        with mapping_path.open(encoding="utf-8-sig", newline="") as handle:
            prefix_rows = [
                row
                for row in csv.DictReader(handle)
                if int(row.get("end_sample") or 0) <= recorded_frames
            ]
    prefix_files = _write_rows(root, prefix_rows, "recorded_prefix_mapping")
    manifest = {
        "schema_version": 1,
        "kind": "acqua_simple_recording",
        "status": "stopped_early" if stopped_early else "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "program_file": str(prepared["program"]),
        "recording_file": Path(prepared["output"]).relative_to(root).as_posix(),
        "program_frames": int(prepared["program_frames"]),
        "program_duration_s": float(prepared["program_duration_s"]),
        "recorded_frames": recorded_frames,
        "recorded_duration_s": float(status.get("duration_s", 0.0)),
        "recording_channels": int(status.get("channels", 0)),
        "sample_rate_hz": int(status.get("sample_rate", 0)),
        "usable_prefix_preserved": recorded_frames > 0,
        "complete_mapped_segments": len(prefix_rows),
        "recorded_prefix_mapping_files": {
            key: path.name for key, path in prefix_files.items() if path is not None
        },
        "alignment_warning": (
            "No marker correction is applied. Segment offsets are the ACQUA "
            "program timeline and assume playback/recording start alignment."
        ),
    }
    path = root / "recording_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
