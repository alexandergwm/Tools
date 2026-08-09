"""Dataset label exports for speech-enhancement scene captures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    ("sample_id", "样本编号", "一次播录样本的稳定唯一编号"),
    ("manual_label", "人工标签", "留给人工填写或修订的标签"),
    ("automatic_label", "自动标签", "由标签前缀和源文件名自动生成"),
    ("dataset_split", "数据集划分", "train、valid 或 test"),
    ("valid", "是否有效", "人工质检开关：是或否"),
    ("quality_flag", "质量标记", "自动质检结果"),
    ("repetition", "重复编号", "同一物理条件下的重复编号"),
    ("target_source", "目标源文件", "目标干净语音源文件"),
    ("interferer_source", "干扰源文件", "干扰声音源文件"),
    ("duration_s", "时长（秒）", "本条播录的实际时长"),
    ("sample_rate_hz", "采样率（赫兹）", "录制采样率"),
    ("microphone_channels", "麦克风通道", "一基编号，逗号分隔"),
    ("target_output_channel", "目标输出通道", "目标声源所用声卡输出"),
    ("interferer_output_channel", "干扰输出通道", "干扰声源所用声卡输出"),
    ("target_level_dbfs", "目标电平（dBFS）", "目标数字播放峰值电平"),
    ("interferer_level_dbfs", "干扰电平（dBFS）", "干扰数字播放峰值电平"),
    ("ambient_recording", "环境底噪录音", "相对于本次运行目录的路径"),
    ("target_recording", "仅目标录音", "相对于本次运行目录的路径"),
    ("interferer_recording", "仅干扰录音", "相对于本次运行目录的路径"),
    ("mixture_recording", "混合录音", "相对于本次运行目录的路径"),
    ("target_playback", "目标播放参考", "两路输出矩阵，未使用通道为零"),
    ("interferer_playback", "干扰播放参考", "两路输出矩阵，未使用通道为零"),
    ("mixture_playback", "混合播放参考", "目标与干扰播放矩阵之和"),
    ("metrics_files", "指标文件", "各场景指标 JSON 路径"),
    ("metadata_json", "实验元数据", "人工头、佩戴、位置等 JSON"),
    ("notes", "备注", "可人工补充"),
]

CORE_LABEL_KEYS = [
    "sample_id",
    "manual_label",
    "automatic_label",
    "dataset_split",
    "valid",
    "quality_flag",
    "repetition",
    "duration_s",
    "notes",
]

CAPTURE_PARAMETER_KEYS = [
    "sample_id",
    "sample_rate_hz",
    "microphone_channels",
    "target_output_channel",
    "interferer_output_channel",
    "target_level_dbfs",
    "interferer_level_dbfs",
]

FILE_INDEX_KEYS = [
    "sample_id",
    "target_source",
    "interferer_source",
    "ambient_recording",
    "target_recording",
    "interferer_recording",
    "mixture_recording",
    "target_playback",
    "interferer_playback",
    "mixture_playback",
    "metrics_files",
    "metadata_json",
]


def write_label_files(
    run_dir: str | Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Write JSONL, CSV and a formatted/editable XLSX workbook."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    normalized = [
        {
            key: _cell_value(row.get(key, ""))
            for key, _header, _description in LABEL_COLUMNS
        }
        for row in rows
    ]

    jsonl_path = root / "labels.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = root / "labels.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _header, _description in LABEL_COLUMNS])
        writer.writeheader()
        writer.writerows(normalized)

    xlsx_path = root / "labels.xlsx"
    _write_xlsx(xlsx_path, normalized, metadata)
    return {"jsonl": jsonl_path, "csv": csv_path, "xlsx": xlsx_path}


def _cell_value(value: Any):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_xlsx(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    try:
        import xlsxwriter
    except Exception as exc:  # pragma: no cover - installation failure
        raise RuntimeError("缺少 XlsxWriter，无法生成 labels.xlsx") from exc

    workbook = xlsxwriter.Workbook(path)
    workbook.set_properties(
        {
            "title": "语音增强采集标签",
            "subject": "声学采集工具自动生成的数据集索引",
            "author": "声学采集工具",
            "comments": "人工标签、是否有效和备注列可在 Excel 中继续编辑。",
        }
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "border": 0,
        }
    )
    text_format = workbook.add_format({"valign": "top"})
    wrapped = workbook.add_format({"valign": "top", "text_wrap": True})
    number = workbook.add_format({"num_format": "0.000", "valign": "top"})
    section = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#548235", "align": "left"}
    )
    note = workbook.add_format({"font_color": "#595959", "text_wrap": True, "valign": "top"})
    bad = workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#9C0006"})
    warning = workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"})

    sheet = workbook.add_worksheet("标签")
    sheet.hide_gridlines(2)
    sheet.freeze_panes(1, 3)
    column_info = {key: (header_text, description) for key, header_text, description in LABEL_COLUMNS}
    keys = CORE_LABEL_KEYS
    headers = [column_info[key][0] for key in keys]
    for column, header_text in enumerate(headers):
        sheet.write(0, column, header_text, header)
    sheet.set_row(0, 30)

    for row_index, row in enumerate(rows, 1):
        for column, key in enumerate(keys):
            value = row.get(key, "")
            cell_format = number if key in {"duration_s", "target_level_dbfs", "interferer_level_dbfs"} else text_format
            if key == "notes":
                cell_format = wrapped
            sheet.write(row_index, column, value, cell_format)

    last_row = max(1, len(rows))
    last_column = len(keys) - 1
    sheet.autofilter(0, 0, last_row, last_column)
    sheet.data_validation(1, 3, max(last_row, 1000), 3, {"validate": "list", "source": ["train", "valid", "test"]})
    sheet.data_validation(1, 4, max(last_row, 1000), 4, {"validate": "list", "source": ["是", "否"]})
    sheet.conditional_format(1, 4, max(last_row, 1000), 4, {"type": "text", "criteria": "containing", "value": "否", "format": bad})
    sheet.conditional_format(1, 5, max(last_row, 1000), 5, {"type": "text", "criteria": "not containing", "value": "通过", "format": warning})

    widths = {
        0: 34,
        1: 18,
        2: 30,
        3: 12,
        4: 10,
        5: 18,
        6: 10,
        7: 12,
        8: 24,
    }
    for column in range(len(keys)):
        sheet.set_column(column, column, widths.get(column, 18))

    parameter_sheet = workbook.add_worksheet("采集参数")
    parameter_sheet.hide_gridlines(2)
    parameter_sheet.freeze_panes(1, 1)
    for column, key in enumerate(CAPTURE_PARAMETER_KEYS):
        parameter_sheet.write(0, column, column_info[key][0], header)
    parameter_sheet.set_row(0, 30)
    for row_index, row in enumerate(rows, 1):
        for column, key in enumerate(CAPTURE_PARAMETER_KEYS):
            cell_format = number if key in {"target_level_dbfs", "interferer_level_dbfs"} else text_format
            parameter_sheet.write(row_index, column, row.get(key, ""), cell_format)
    parameter_sheet.autofilter(0, 0, max(1, len(rows)), len(CAPTURE_PARAMETER_KEYS) - 1)
    parameter_sheet.set_column(0, 0, 34)
    parameter_sheet.set_column(1, len(CAPTURE_PARAMETER_KEYS) - 1, 18)

    index_sheet = workbook.add_worksheet("文件索引")
    index_sheet.hide_gridlines(2)
    index_sheet.freeze_panes(1, 1)
    for column, key in enumerate(FILE_INDEX_KEYS):
        index_sheet.write(0, column, column_info[key][0], header)
    index_sheet.set_row(0, 30)
    for row_index, row in enumerate(rows, 1):
        for column, key in enumerate(FILE_INDEX_KEYS):
            index_sheet.write(row_index, column, row.get(key, ""), wrapped)
        index_sheet.set_row(row_index, 48)
    index_sheet.autofilter(0, 0, max(1, len(rows)), len(FILE_INDEX_KEYS) - 1)
    index_sheet.set_column(0, 0, 34)
    index_sheet.set_column(1, len(FILE_INDEX_KEYS) - 1, 30)

    summary = workbook.add_worksheet("汇总")
    summary.hide_gridlines(2)
    summary.set_column("A:A", 28)
    summary.set_column("B:B", 18)
    summary.write("A1", "数据集采集汇总", section)
    summary.write("A3", "样本总数")
    summary.write_formula("B3", f"=COUNTA('标签'!A2:A{len(rows) + 1})", None, len(rows))
    summary.write("A4", "标记有效样本")
    summary.write_formula("B4", f'=COUNTIF(\'标签\'!E2:E{len(rows) + 1},"是")', None, len(rows))
    summary.write("A5", "训练集样本")
    summary.write_formula("B5", f'=COUNTIF(\'标签\'!D2:D{len(rows) + 1},"train")', None, sum(row.get("dataset_split") == "train" for row in rows))
    summary.write("A7", "实验元数据", section)
    summary.write("A8", json.dumps(metadata, ensure_ascii=False, indent=2), note)
    summary.set_row(7, 120)

    guide = workbook.add_worksheet("字段说明")
    guide.hide_gridlines(2)
    guide.freeze_panes(1, 0)
    guide.write_row(0, 0, ["字段名", "Excel 列名", "说明"], header)
    for row_index, (key, header_text, description) in enumerate(LABEL_COLUMNS, 1):
        guide.write_row(row_index, 0, [key, header_text, description])
    guide.set_column("A:A", 28)
    guide.set_column("B:B", 24)
    guide.set_column("C:C", 60)
    workbook.close()
