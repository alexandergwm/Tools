"""Low-friction XLSX capture checklist support.

The checklist intentionally stays flat: one row is one physical experiment.
It can be edited in Excel, selected from the GUI, and updated in place after a
successful capture.  The implementation reads and patches the small subset of
OOXML used by ordinary Excel tables so the Portable build needs no extra Excel
runtime or heavyweight workbook reader.
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .config import ExperimentConfig


CHECKLIST_SHEET = "采集清单"
CHECKLIST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "待采集/已完成/跳过/失败"),
    ("workflow", "rir / supervised / supervised_pair / target_only / interferer_only / audio"),
    ("experiment_name", "现场唯一实验名；一行对应一次物理实验"),
    ("dataset_split", "train / valid / test"),
    ("project_id", "项目编号"),
    ("room_id", "房间编号"),
    ("artificial_head_id", "人工头编号"),
    ("headset_model_id", "耳机型号"),
    ("headset_unit_id", "耳机个体编号"),
    ("wearing_id", "本次重新佩戴编号"),
    ("boom_pose_id", "麦杆姿态编号"),
    ("source_role", "RIR: mouth / interferer"),
    ("source_id", "RIR 声源编号"),
    ("azimuth_deg", "RIR 声源方位角"),
    ("elevation_deg", "RIR 声源俯仰角"),
    ("source_height_cm", "RIR 声源高度 cm"),
    ("distance_cm", "RIR 声源距离 cm"),
    ("target_source_id", "语音目标源编号"),
    ("target_position_id", "语音目标源位置编号"),
    ("target_azimuth_deg", "语音目标源方位角"),
    ("target_elevation_deg", "语音目标源俯仰角"),
    ("target_height_m", "语音目标源高度 m"),
    ("target_distance_m", "语音目标源距离 m"),
    ("interferer_source_id", "语音干扰源编号"),
    ("interferer_position_id", "语音干扰源位置编号"),
    ("interferer_azimuth_deg", "语音干扰源方位角"),
    ("interferer_elevation_deg", "语音干扰源俯仰角"),
    ("interferer_height_m", "语音干扰源高度 m"),
    ("interferer_distance_m", "语音干扰源距离 m"),
    ("notes", "现场备注"),
    ("completed_run", "工具自动填写结果目录"),
    ("completed_at", "工具自动填写完成时间"),
    ("last_error", "工具自动填写最近错误"),
)

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_DOC_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL_PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("", _MAIN_NS)


def create_checklist(path: str | Path, rows: list[dict[str, Any]] | None = None) -> Path:
    """Create a practical checklist template using the app's XLSX writer."""
    try:
        import xlsxwriter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 XlsxWriter，无法生成测试清单") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows or []
    workbook = xlsxwriter.Workbook(path)
    workbook.set_properties(
        {
            "title": "声学数据采集测试清单",
            "subject": "一行对应一次独立物理实验",
            "author": "声学采集工具",
        }
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
        }
    )
    text = workbook.add_format({"valign": "top"})
    warning = workbook.add_format({"bg_color": "#FFF2CC"})
    success = workbook.add_format({"bg_color": "#E2F0D9"})

    sheet = workbook.add_worksheet(CHECKLIST_SHEET)
    sheet.hide_gridlines(2)
    sheet.freeze_panes(1, 3)
    keys = [key for key, _description in CHECKLIST_COLUMNS]
    sheet.write_row(0, 0, keys, header)
    sheet.set_row(0, 28)
    for row_index, row in enumerate(rows, 1):
        for column, key in enumerate(keys):
            sheet.write(row_index, column, row.get(key, ""), text)
    last_row = max(len(rows), 200)
    status_col = keys.index("status")
    workflow_col = keys.index("workflow")
    split_col = keys.index("dataset_split")
    sheet.data_validation(
        1,
        status_col,
        last_row,
        status_col,
        {"validate": "list", "source": ["待采集", "已完成", "跳过", "失败"]},
    )
    sheet.data_validation(
        1,
        workflow_col,
        last_row,
        workflow_col,
        {
            "validate": "list",
            "source": [
                "rir",
                "supervised",
                "supervised_pair",
                "target_only",
                "interferer_only",
                "audio",
            ],
        },
    )
    sheet.data_validation(
        1,
        split_col,
        last_row,
        split_col,
        {"validate": "list", "source": ["train", "valid", "test"]},
    )
    sheet.conditional_format(
        1,
        status_col,
        last_row,
        status_col,
        {"type": "text", "criteria": "containing", "value": "待采集", "format": warning},
    )
    sheet.conditional_format(
        1,
        status_col,
        last_row,
        status_col,
        {"type": "text", "criteria": "containing", "value": "已完成", "format": success},
    )
    sheet.autofilter(0, 0, max(1, len(rows)), len(keys) - 1)
    sheet.set_column(0, 0, 12)
    sheet.set_column(1, 1, 16)
    sheet.set_column(2, 2, 38)
    sheet.set_column(3, len(keys) - 1, 19)
    sheet.set_column(keys.index("notes"), keys.index("notes"), 34)
    sheet.set_column(keys.index("completed_run"), keys.index("last_error"), 38)
    # Project-wide values normally come from the YAML template and progress
    # details are written by the app.  Keep them available for specialists,
    # but hide them in the default field view so the lab checklist stays fast.
    for key in (
        "project_id",
        "room_id",
        "artificial_head_id",
        "headset_model_id",
        "target_source_id",
        "target_position_id",
        "interferer_source_id",
        "completed_at",
        "last_error",
    ):
        column = keys.index(key)
        sheet.set_column(column, column, 18, None, {"hidden": True})

    guide = workbook.add_worksheet("字段说明")
    guide.hide_gridlines(2)
    guide.freeze_panes(1, 0)
    guide.write_row(0, 0, ["字段名", "说明"], header)
    for row_index, item in enumerate(CHECKLIST_COLUMNS, 1):
        guide.write_row(row_index, 0, item, text)
    guide.set_column(0, 0, 30)
    guide.set_column(1, 1, 70)
    workbook.close()
    return path


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    value = 0
    for character in letters.group(0):
        value = value * 26 + ord(character) - 64
    return value - 1


def _column_letters(index: int) -> str:
    value = index + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(65 + remainder) + output
    return output


def _sheet_part(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(f"{{{_MAIN_NS}}}sheets")
    if sheets is None or not list(sheets):
        raise ValueError("XLSX 中没有工作表")
    selected = next(
        (sheet for sheet in sheets if sheet.get("name") == CHECKLIST_SHEET),
        list(sheets)[0],
    )
    relationship_id = selected.get(f"{{{_REL_DOC_NS}}}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship = next(
        (
            item
            for item in relationships.findall(f"{{{_REL_PKG_NS}}}Relationship")
            if item.get("Id") == relationship_id
        ),
        None,
    )
    if relationship is None:
        raise ValueError("无法解析测试清单工作表")
    target = str(relationship.get("Target") or "").replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")
    return str((Path("xl") / target).as_posix())


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    result = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        result.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return result


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        return "" if inline is None else "".join(
            node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t")
        )
    value_node = cell.find(f"{{{_MAIN_NS}}}v")
    value = "" if value_node is None else value_node.text or ""
    if cell_type == "s" and value:
        return shared[int(value)]
    if cell_type in {"str", "e"}:
        return value
    if cell_type == "b":
        return value == "1"
    if value == "":
        return ""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def read_checklist(path: str | Path) -> list[dict[str, Any]]:
    """Read checklist rows and attach the real Excel row as ``_row_number``."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        part = _sheet_part(archive)
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(part))
    sheet_data = root.find(f"{{{_MAIN_NS}}}sheetData")
    if sheet_data is None:
        return []
    raw_rows: list[tuple[int, dict[int, Any]]] = []
    for row in sheet_data.findall(f"{{{_MAIN_NS}}}row"):
        row_number = int(row.get("r") or 0)
        values = {
            _column_index(str(cell.get("r"))): _cell_value(cell, shared)
            for cell in row.findall(f"{{{_MAIN_NS}}}c")
            if cell.get("r")
        }
        raw_rows.append((row_number, values))
    if not raw_rows:
        return []
    headers = {
        column: str(value).strip()
        for column, value in raw_rows[0][1].items()
        if str(value).strip()
    }
    required = {"workflow", "experiment_name"}
    if not required.issubset(headers.values()):
        raise ValueError("测试清单首行必须包含 workflow 和 experiment_name 列")
    rows = []
    for row_number, values in raw_rows[1:]:
        item = {
            key: values.get(column, "")
            for column, key in headers.items()
        }
        if not any(value not in (None, "") for value in item.values()):
            continue
        item["_row_number"] = row_number
        rows.append(item)
    return rows


def _set_inline_cell(row: ET.Element, column: int, row_number: int, value: Any) -> None:
    reference = f"{_column_letters(column)}{row_number}"
    cell = next(
        (
            item
            for item in row.findall(f"{{{_MAIN_NS}}}c")
            if item.get("r") == reference
        ),
        None,
    )
    if cell is None:
        cell = ET.Element(f"{{{_MAIN_NS}}}c", {"r": reference})
        existing = list(row.findall(f"{{{_MAIN_NS}}}c"))
        insertion = next(
            (
                index
                for index, item in enumerate(existing)
                if _column_index(str(item.get("r"))) > column
            ),
            len(existing),
        )
        row.insert(insertion, cell)
    for child in list(cell):
        cell.remove(child)
    cell.set("t", "inlineStr")
    inline = ET.SubElement(cell, f"{{{_MAIN_NS}}}is")
    text = ET.SubElement(inline, f"{{{_MAIN_NS}}}t")
    text.text = "" if value is None else str(value)


def update_checklist_row(
    path: str | Path,
    row_number: int,
    *,
    status: str,
    completed_run: str = "",
    last_error: str = "",
) -> None:
    """Patch progress cells in place while preserving the rest of the workbook."""
    path = Path(path).resolve()
    with zipfile.ZipFile(path, "r") as source:
        part = _sheet_part(source)
        shared = _shared_strings(source)
        root = ET.fromstring(source.read(part))
        sheet_data = root.find(f"{{{_MAIN_NS}}}sheetData")
        if sheet_data is None:
            raise ValueError("测试清单没有数据区域")
        rows = sheet_data.findall(f"{{{_MAIN_NS}}}row")
        header_row = next((row for row in rows if int(row.get("r") or 0) == 1), None)
        target_row = next(
            (row for row in rows if int(row.get("r") or 0) == int(row_number)), None
        )
        if header_row is None or target_row is None:
            raise ValueError(f"测试清单中不存在第 {row_number} 行")
        headers = {
            str(_cell_value(cell, shared)).strip(): _column_index(str(cell.get("r")))
            for cell in header_row.findall(f"{{{_MAIN_NS}}}c")
            if cell.get("r")
        }
        updates = {
            "status": status,
            "completed_run": completed_run,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "last_error": last_error,
        }
        missing = [key for key in updates if key not in headers]
        if missing:
            raise ValueError("测试清单缺少进度列：" + ", ".join(missing))
        for key, value in updates.items():
            _set_inline_cell(target_row, headers[key], row_number, value)
        replacement = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        entries = [(item, source.read(item.filename)) for item in source.infolist()]

    descriptor, temporary = tempfile.mkstemp(suffix=".xlsx", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for info, data in entries:
                destination.writestr(info, replacement if info.filename == part else data)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _convert_like(value: Any, current: Any) -> Any:
    if value in (None, ""):
        return current
    if isinstance(current, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "是"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return [item.strip() for item in str(value).split(",") if item.strip()]
    return str(value).strip()


def _metadata_set(metadata: dict[str, Any], dotted: str, value: Any) -> None:
    target = metadata
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def apply_checklist_row(
    config: ExperimentConfig, row: dict[str, Any], checklist_path: str | Path
) -> str:
    """Apply one flat checklist row and return the internal workflow kind."""
    workflow = str(row.get("workflow") or "").strip().lower()
    workflow_aliases = {
        "rir": "rir",
        "supervised": "scene",
        "supervised_pair": "scene",
        "target_mixed": "scene",
        "speech": "scene",
        "target_only": "scene",
        "interferer_only": "scene",
        "audio": "io",
        "io": "io",
    }
    if workflow not in workflow_aliases:
        raise ValueError(f"不支持的清单 workflow：{workflow}")
    name = str(row.get("experiment_name") or "").strip()
    if not name:
        raise ValueError("清单 experiment_name 不能为空")

    if workflow == "supervised" or workflow == "speech":
        config.scene.items = ["ambient", "target_only", "interferer_only", "mixture"]
    elif workflow in {"supervised_pair", "target_mixed"}:
        config.scene.items = ["target_only", "interferer_only", "mixture"]
    elif workflow == "target_only":
        config.scene.items = ["target_only"]
    elif workflow == "interferer_only":
        config.scene.items = ["interferer_only"]

    metadata = dict(config.metadata)
    direct_metadata = {
        "project_id",
        "room_id",
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
        "notes",
    }
    numeric_metadata = {
        "azimuth_deg",
        "elevation_deg",
        "source_height_cm",
        "distance_cm",
    }
    # A checklist row is a complete physical experiment, not a partial patch.
    # Clear values owned by the previous row so a blank cell cannot silently
    # inherit the preceding wearing, head, source or geometry.
    for key in direct_metadata:
        metadata.pop(key, None)
    metadata.pop("target", None)
    metadata.pop("interferer", None)
    for key in direct_metadata:
        value = row.get(key)
        if value not in (None, ""):
            metadata[key] = float(value) if key in numeric_metadata else str(value).strip()
    source_mapping = {
        "target_source_id": "target.source_id",
        "target_position_id": "target.position_id",
        "target_azimuth_deg": "target.azimuth_deg",
        "target_elevation_deg": "target.elevation_deg",
        "target_height_m": "target.height_m",
        "target_distance_m": "target.distance_m",
        "interferer_source_id": "interferer.source_id",
        "interferer_position_id": "interferer.position_id",
        "interferer_azimuth_deg": "interferer.azimuth_deg",
        "interferer_elevation_deg": "interferer.elevation_deg",
        "interferer_height_m": "interferer.height_m",
        "interferer_distance_m": "interferer.distance_m",
    }
    numeric_source = {
        "target_azimuth_deg",
        "target_elevation_deg",
        "target_height_m",
        "target_distance_m",
        "interferer_azimuth_deg",
        "interferer_elevation_deg",
        "interferer_height_m",
        "interferer_distance_m",
    }
    for key, destination in source_mapping.items():
        value = row.get(key)
        if value not in (None, ""):
            _metadata_set(
                metadata,
                destination,
                float(value) if key in numeric_source else str(value).strip(),
            )

    reserved = {key for key, _description in CHECKLIST_COLUMNS} | {"_row_number"}
    for key, value in row.items():
        if key in reserved or value in (None, ""):
            continue
        if key.startswith("metadata."):
            _metadata_set(metadata, key.removeprefix("metadata."), value)
            continue
        if "." not in key:
            continue
        section_name, field_name = key.split(".", 1)
        section = getattr(config, section_name, None)
        if section is not None and hasattr(section, field_name):
            current = getattr(section, field_name)
            setattr(section, field_name, _convert_like(value, current))

    split = str(row.get("dataset_split") or config.scene.dataset_split).strip().lower()
    if split:
        config.scene.dataset_split = split
        metadata["dataset_split"] = split
    metadata.update(
        {
            "experiment_name": name,
            "experiment_id": name,
            "scene_id": name,
            "checklist_file": str(Path(checklist_path).resolve()),
            "checklist_row": int(row["_row_number"]),
        }
    )
    config.metadata = metadata
    config.storage.session_name = name
    config.validate()
    return workflow_aliases[workflow]
