"""Professional capture metadata validation and preflight reporting.

The ordinary GUI remains permissive.  Setting ``metadata.capture_profile`` to
``production`` turns missing training-critical metadata into blocking errors.
This lets the same application serve quick engineering trials and controlled
dataset production without maintaining two separate capture programs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


PRODUCTION_PROFILE = "production"
SUPPORTED_PROFILES = {"standard", "development", PRODUCTION_PROFILE}


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    status: str  # pass | warning | error
    title: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    workflow: str
    capture_profile: str
    checks: tuple[PreflightCheck, ...]

    @property
    def errors(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.status == "error"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [check for check in self.checks if check.status == "warning"]

    @property
    def can_start(self) -> bool:
        return not self.errors

    @property
    def production_ready(self) -> bool:
        return self.can_start and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": self.workflow,
            "capture_profile": self.capture_profile,
            "status": (
                "blocked"
                if self.errors
                else "warning"
                if self.warnings
                else "ready"
            ),
            "can_start": self.can_start,
            "production_ready": self.production_ready,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": [asdict(check) for check in self.checks],
        }


def capture_profile(config: ExperimentConfig) -> str:
    value = config.metadata.get(
        "capture_profile", config.metadata.get("professional_profile", "standard")
    )
    return str(value or "standard").strip().lower()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    value = metadata.get("array")
    return value if isinstance(value, dict) else {}


def array_geometry_sha256(metadata: dict[str, Any]) -> str:
    array = array_metadata(metadata)
    return canonical_sha256(array) if array else ""


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _add_required_metadata_checks(
    checks: list[PreflightCheck], config: ExperimentConfig, production: bool
) -> None:
    required = (
        "project_id",
        "room_id",
        "artificial_head_id",
        "headset_model_id",
        "headset_unit_id",
        "wearing_id",
        "boom_pose_id",
    )
    missing = [key for key in required if _missing(config.metadata.get(key))]
    if not missing:
        checks.append(
            PreflightCheck(
                "traceability",
                "pass",
                "实验身份与物理条件",
                "项目、房间、人工头、耳机个体、佩戴和麦杆姿态均已记录。",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "traceability",
                "error" if production else "warning",
                "实验身份与物理条件",
                "缺少字段：" + ", ".join(missing),
            )
        )


def _array_checks(
    checks: list[PreflightCheck], config: ExperimentConfig, production: bool
) -> None:
    array = array_metadata(config.metadata)
    if not array:
        legacy_mapping = {
            channel: config.metadata.get(f"microphone_{channel}")
            for channel in config.audio.input_channels
        }
        mapped = all(not _missing(value) for value in legacy_mapping.values())
        checks.append(
            PreflightCheck(
                "array_geometry",
                "warning" if mapped else "error" if production else "warning",
                "麦克风阵列定义",
                (
                    "已使用 microphone_N 保存稳定通道身份；若模型显式使用阵列几何，再在项目模板中补 metadata.array。"
                    if mapped
                    else "未提供 metadata.array 或完整 microphone_N 映射，无法可靠还原通道身份。"
                ),
            )
        )
        return

    errors: list[str] = []
    warnings: list[str] = []
    for key in ("array_id", "coordinate_system", "reference_point"):
        if _missing(array.get(key)):
            errors.append(f"array.{key} 为空")
    channels = array.get("channels")
    if not isinstance(channels, list) or not channels:
        errors.append("array.channels 必须是非空列表")
        channels = []

    recording_channels: list[int] = []
    microphone_ids: list[str] = []
    for index, channel in enumerate(channels, 1):
        if not isinstance(channel, dict):
            errors.append(f"array.channels[{index}] 不是对象")
            continue
        recording_channel = channel.get("recording_channel")
        if not isinstance(recording_channel, int) or recording_channel < 1:
            errors.append(f"array.channels[{index}].recording_channel 无效")
        else:
            recording_channels.append(recording_channel)
        microphone_id = str(channel.get("microphone_id") or "").strip()
        if not microphone_id:
            errors.append(f"array.channels[{index}].microphone_id 为空")
        else:
            microphone_ids.append(microphone_id)
        coordinates = []
        for key in ("x_m", "y_m", "z_m"):
            value = channel.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                coordinates.append(key)
        geometry_required = bool(array.get("geometry_required", False))
        if coordinates:
            message = (
                f"array.channels[{index}] 缺少有限坐标：{', '.join(coordinates)}"
            )
            (errors if geometry_required else warnings).append(message)
        if channel.get("polarity", 1) not in (-1, 1):
            errors.append(f"array.channels[{index}].polarity 只能是 -1 或 1")

    expected = list(config.audio.input_channels)
    if sorted(recording_channels) != sorted(expected):
        errors.append(
            f"阵列 recording_channel={recording_channels} 与录制通道={expected} 不一致"
        )
    if len(recording_channels) != len(set(recording_channels)):
        errors.append("array.channels 中存在重复 recording_channel")
    if len(microphone_ids) != len(set(microphone_ids)):
        errors.append("array.channels 中存在重复 microphone_id")
    reference_channel = array.get("reference_channel")
    if reference_channel not in recording_channels:
        errors.append("array.reference_channel 必须对应一个录制通道")

    if errors:
        checks.append(
            PreflightCheck(
                "array_geometry", "error", "麦克风阵列定义", "；".join(errors)
            )
        )
    elif warnings:
        checks.append(
            PreflightCheck(
                "array_geometry", "warning", "麦克风阵列定义", "；".join(warnings)
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "array_geometry",
                "pass",
                "麦克风阵列定义",
                f"{array.get('array_id')}：{len(channels)} 个通道，映射与坐标完整。",
            )
        )


def _source_checks(
    checks: list[PreflightCheck], config: ExperimentConfig, workflow: str, production: bool
) -> None:
    metadata = config.metadata
    if workflow == "rir":
        missing = [
            key
            for key in (
                "source_role",
                "source_id",
                "azimuth_deg",
                "elevation_deg",
                "source_height_cm",
                "distance_cm",
            )
            if _missing(metadata.get(key))
        ]
        invalid: list[str] = []
        numeric_ranges = {
            "azimuth_deg": (None, None),
            "elevation_deg": (-90.0, 90.0),
            "source_height_cm": (0.0, None),
            "distance_cm": (0.0, None),
        }
        for key, (minimum, maximum) in numeric_ranges.items():
            if key in missing:
                continue
            try:
                value = float(metadata[key])
            except (TypeError, ValueError):
                invalid.append(f"{key} 不是数字")
                continue
            if not math.isfinite(value):
                invalid.append(f"{key} 不是有限值")
            elif minimum is not None and value <= minimum:
                invalid.append(f"{key} 必须大于 {minimum:g}")
            elif maximum is not None and value > maximum:
                invalid.append(f"{key} 必须不大于 {maximum:g}")
        problems = [*(f"缺少 {key}" for key in missing), *invalid]
        checks.append(
            PreflightCheck(
                "rir_source_geometry",
                "error" if problems and production else "warning" if problems else "pass",
                "RIR 声源与几何",
                "；".join(problems)
                if problems
                else "声源角色、编号、角度、高度和距离均已记录。",
            )
        )
        return

    if workflow not in {"scene", "speech", "speech_enhancement"}:
        return
    items = set(config.scene.items)
    missing: list[str] = []
    if items & {"target_only", "mixture"}:
        target = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
        for key in (
            "source_id",
            "position_id",
            "azimuth_deg",
            "elevation_deg",
            "height_m",
            "distance_m",
        ):
            if _missing(target.get(key)):
                missing.append(f"target.{key}")
    if items & {"interferer_only", "mixture"}:
        interferer = (
            metadata.get("interferer")
            if isinstance(metadata.get("interferer"), dict)
            else {}
        )
        for key in (
            "source_id",
            "position_id",
            "azimuth_deg",
            "elevation_deg",
            "height_m",
            "distance_m",
        ):
            if _missing(interferer.get(key)):
                missing.append(f"interferer.{key}")
    checks.append(
        PreflightCheck(
            "speech_source_geometry",
            "error" if missing and production else "warning" if missing else "pass",
            "语音声源与几何",
            "缺少字段：" + ", ".join(missing)
            if missing
            else "当前采集场景所需的目标/干扰声源身份与位置已记录。",
        )
    )


def _source_file_checks(checks: list[PreflightCheck], config: ExperimentConfig) -> None:
    if not set(config.scene.items) & {"target_only", "interferer_only", "mixture"}:
        return
    missing: list[str] = []
    if config.scene.source_mode == "single":
        if set(config.scene.items) & {"target_only", "mixture"} and not Path(
            config.scene.target_file
        ).is_file():
            missing.append(f"target_file={config.scene.target_file}")
        if set(config.scene.items) & {"interferer_only", "mixture"} and not Path(
            config.scene.interferer_file
        ).is_file():
            missing.append(f"interferer_file={config.scene.interferer_file}")
    else:
        extensions = {
            value.lower() if value.startswith(".") else f".{value.lower()}"
            for value in config.scene.file_extensions
        }

        def has_audio(folder_value: str) -> bool:
            folder = Path(folder_value)
            return folder.is_dir() and any(
                path.is_file() and path.suffix.lower() in extensions
                for path in folder.rglob("*")
            )

        if set(config.scene.items) & {"target_only", "mixture"} and not has_audio(
            config.scene.target_folder
        ):
            missing.append(f"target_folder 无匹配音频={config.scene.target_folder}")
        if set(config.scene.items) & {"interferer_only", "mixture"} and not has_audio(
            config.scene.interferer_folder
        ):
            missing.append(
                f"interferer_folder 无匹配音频={config.scene.interferer_folder}"
            )
    checks.append(
        PreflightCheck(
            "source_files",
            "error" if missing else "pass",
            "播放素材",
            "不存在：" + "；".join(missing) if missing else "所需播放文件或文件夹存在。",
        )
    )


def _clock_check(checks: list[PreflightCheck], config: ExperimentConfig) -> None:
    if "mixture" not in config.scene.items:
        return
    same_device = config.audio.backend == "simulated" or (
        config.audio.input_device not in (None, "")
        and config.audio.output_device not in (None, "")
        and str(config.audio.input_device).strip().casefold()
        == str(config.audio.output_device).strip().casefold()
    )
    checks.append(
        PreflightCheck(
            "shared_hardware_clock",
            "pass" if same_device else "error" if capture_profile(config) == PRODUCTION_PROFILE else "warning",
            "监督配对硬件时钟",
            "输入与输出使用同一可验证双工设备。"
            if same_device
            else "输入与输出不是同一可验证双工设备；只能做流程试采，不能标记为严格监督可用。",
        )
    )
    if config.audio.target_output_channel == config.audio.interferer_output_channel:
        checks.append(
            PreflightCheck(
                "source_output_routing",
                "error",
                "目标/干扰输出路由",
                "mixed 采集要求目标与干扰使用不同输出通道。",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "source_output_routing",
                "pass",
                "目标/干扰输出路由",
                f"目标输出 {config.audio.target_output_channel}，干扰输出 {config.audio.interferer_output_channel}。",
            )
        )


def _calibration_check(
    checks: list[PreflightCheck], config: ExperimentConfig, production: bool
) -> None:
    calibration = config.metadata.get("calibration")
    calibration = calibration if isinstance(calibration, dict) else {}
    useful = any(
        not _missing(calibration.get(key))
        for key in (
            "date",
            "microphone_calibration_date",
            "playback_spl_db",
            "playback_calibration_spl_db",
            "calibrator_id",
        )
    )
    required = bool(calibration.get("required", False))
    checks.append(
        PreflightCheck(
            "calibration",
            "pass" if useful else "error" if required else "warning",
            "校准记录",
            "已记录校准信息。"
            if useful
            else "未记录麦克风灵敏度/输入增益/播放 SPL 校准；相对训练可用，但跨批次电平不可严格比较。",
        )
    )


def _scene_pair_count(config: ExperimentConfig) -> int | None:
    scene = config.scene
    if scene.source_mode == "single":
        return 1
    extensions = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in scene.file_extensions
    }
    items = set(scene.items)
    counts: list[int] = []
    for required, folder_value in (
        (bool(items & {"target_only", "mixture"}), scene.target_folder),
        (bool(items & {"interferer_only", "mixture"}), scene.interferer_folder),
    ):
        if not required:
            counts.append(1)
            continue
        folder = Path(folder_value)
        if not folder.is_dir():
            return None
        counts.append(
            sum(
                path.is_file() and path.suffix.lower() in extensions
                for path in folder.rglob("*")
            )
        )
    if not counts or not all(counts):
        return None
    return math.prod(counts) if scene.pairing_mode == "cartesian" else max(counts)


def _estimated_scene_bytes(config: ExperimentConfig) -> int | None:
    pairs = _scene_pair_count(config)
    duration = config.scene.duration_s
    if pairs is None or duration is None:
        return None
    scene, fs = config.scene, config.audio.sample_rate
    repetitions = scene.repetitions
    mic_count = len(config.audio.input_channels)
    output_count = max(
        config.audio.target_output_channel, config.audio.interferer_output_channel
    )
    item_count = len([item for item in scene.items if item != "ambient"])
    stream_s = scene.gap_s + item_count * (duration + scene.gap_s)
    # Complete multichannel stream plus extracted item WAVs.
    channel_seconds = pairs * repetitions * mic_count * (
        stream_s + item_count * duration
    )
    if "ambient" in scene.items:
        channel_seconds += repetitions * mic_count * scene.ambient_duration_s
    if config.storage.save_playback_reference:
        # Full playback, extracted item playbacks and one mono emitted copy per source.
        channel_seconds += pairs * repetitions * output_count * (
            stream_s + item_count * duration
        )
        channel_seconds += pairs * duration * 2
    bytes_per_sample = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4}.get(
        config.storage.wav_subtype, 4
    )
    return math.ceil(channel_seconds * fs * bytes_per_sample * 1.15)


def _storage_check(
    checks: list[PreflightCheck], config: ExperimentConfig, workflow: str
) -> None:
    root = Path(config.storage.root)
    parent = _existing_parent(root)
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    detail = f"保存目录：{root}"
    if writable:
        try:
            free_gib = shutil.disk_usage(parent).free / (1024**3)
            detail += f"；可用空间 {free_gib:.1f} GiB"
            estimate = (
                _estimated_scene_bytes(config)
                if workflow in {"scene", "speech", "speech_enhancement"}
                else None
            )
            if estimate is not None:
                estimate_gib = estimate / (1024**3)
                detail += f"；预计本批最多约 {estimate_gib:.1f} GiB（含 15% 余量）"
                status = "error" if estimate > shutil.disk_usage(parent).free else "pass"
                if status == "error":
                    detail += "（空间不足，已阻止开始）"
            else:
                status = "warning" if free_gib < 2.0 else "pass"
            if estimate is None and free_gib < 2.0:
                detail += "（建议至少保留 2 GiB）"
        except OSError:
            status = "warning"
            detail += "；无法读取剩余空间"
    else:
        status = "error"
        detail += f"；现有父目录 {parent} 不可写"
    checks.append(PreflightCheck("storage", status, "保存位置", detail))


def build_preflight_report(
    config: ExperimentConfig,
    workflow: str,
    *,
    check_source_paths: bool = True,
) -> PreflightReport:
    """Build a non-destructive, machine-readable professional preflight."""
    profile = capture_profile(config)
    production = profile == PRODUCTION_PROFILE
    checks: list[PreflightCheck] = []
    if profile not in SUPPORTED_PROFILES:
        checks.append(
            PreflightCheck(
                "capture_profile",
                "error",
                "采集配置级别",
                f"不支持 capture_profile={profile!r}；可用 standard/development/production。",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "capture_profile",
                "pass",
                "采集配置级别",
                f"当前为 {profile}；production 模式会阻止缺少关键元数据的正式采集。",
            )
        )
    _add_required_metadata_checks(checks, config, production)
    _array_checks(checks, config, production)
    _source_checks(checks, config, workflow, production)
    if workflow in {"scene", "speech", "speech_enhancement"}:
        if check_source_paths:
            _source_file_checks(checks, config)
        _clock_check(checks, config)
    _calibration_check(checks, config, production)
    _storage_check(checks, config, workflow)
    return PreflightReport(workflow, profile, tuple(checks))


def format_preflight_report(report: PreflightReport) -> str:
    marker = {"pass": "[通过]", "warning": "[提醒]", "error": "[阻止]"}
    lines = [
        f"工作流：{report.workflow}",
        f"采集级别：{report.capture_profile}",
        f"结论：{'可以开始' if report.can_start else '不能开始'}",
        "",
    ]
    lines.extend(
        f"{marker.get(check.status, '[?]')} {check.title}：{check.detail}"
        for check in report.checks
    )
    return "\n".join(lines)


def assert_capture_ready(config: ExperimentConfig, workflow: str) -> PreflightReport:
    """Raise only for blocking preflight errors; standard warnings stay non-blocking."""
    report = build_preflight_report(config, workflow)
    if not report.can_start:
        details = "；".join(check.detail for check in report.errors)
        raise ValueError(f"专业预检未通过：{details}")
    return report
