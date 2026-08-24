"""Small Tk GUI for editing common parameters and starting captures."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .audio import (
    check_hardware_settings,
    create_backend,
    device_choices,
    format_hardware_status,
    host_api_choices,
    list_devices,
)
from .config import ExperimentConfig, load_config, save_config
from .labels import import_reviewed_labels
from .checklist import apply_checklist_row, create_checklist, read_checklist, update_checklist_row
from .general import capture_general_io
from .professional import assert_capture_ready, build_preflight_report, format_preflight_report
from .rir import capture_rir
from .scene import capture_scene_block, discover_source_pairs
from .signals import exponential_sweep, measurement_signal
from .viewer import ResultsViewer


FIELDS = [
    ("audio.backend", "音频后端", str, "common"),
    ("audio.host_api", "音频协议 / 主机接口", str, "common"),
    ("audio.input_device", "录制设备名称或编号", str, "common"),
    ("audio.output_device", "播放设备名称或编号", str, "common"),
    ("audio.sample_rate", "采样率", int, "common"),
    ("audio.block_size", "缓冲区帧数", int, "common"),
    ("audio.input_channels", "录制通道（逗号分隔）", lambda x: [int(v) for v in x.split(",")], "common"),
    ("general.action", "播录操作", str, "basic"),
    ("general.source_file", "要播放的音频文件", str, "basic"),
    ("general.duration_s", "持续时间（秒）", float, "basic"),
    ("general.level_dbfs", "播放电平（满刻度分贝）", float, "basic"),
    ("general.output_channel", "播放输出通道", int, "basic"),
    ("audio.target_output_channel", "人工嘴 / 目标源输出通道", int, "rir_speech"),
    ("audio.interferer_output_channel", "干扰源输出通道", int, "speech"),
    ("sweep.start_hz", "扫频起始频率（赫兹）", float, "rir"),
    ("sweep.end_hz", "扫频终止频率（赫兹）", float, "rir"),
    ("sweep.duration_s", "扫频持续时间（秒）", float, "rir"),
    ("sweep.pre_silence_s", "扫频前静音（秒）", float, "rir"),
    ("sweep.post_silence_s", "扫频后静音（秒）", float, "rir"),
    ("sweep.fade_in_s", "扫频淡入（秒，MATLAB 默认 0.08）", float, "rir"),
    ("sweep.fade_out_s", "扫频淡出（秒，MATLAB 默认 0.005）", float, "rir"),
    ("sweep.level_dbfs", "扫频播放电平（dBFS）", float, "rir"),
    ("sweep.rir_duration_s", "脉冲响应长度（秒）", float, "rir"),
    ("sweep.pre_peak_s", "脉冲峰值前保留时间（秒）", float, "rir"),
    ("repeats.strategy", "RIR 重复采集方式", str, "rir"),
    ("repeats.fixed_count", "每次实验的录制次数", int, "rir"),
    ("repeats.correlation_threshold", "脉冲响应相关性阈值", float, "rir"),
    ("repeats.minimum_sweep_snr_db", "扫频相对底噪最低信噪比（分贝）", float, "rir"),
    ("scene.source_mode", "素材选择方式", str, "speech"),
    ("scene.duration_s", "每条片段时长（秒，推荐 4）", lambda x: None if not x else float(x), "speech"),
    ("scene.target_file", "干净目标音频文件", str, "speech"),
    ("scene.interferer_file", "干扰音频文件", str, "speech"),
    ("scene.target_folder", "干净目标音频文件夹", str, "speech"),
    ("scene.interferer_folder", "干扰音频文件夹", str, "speech"),
    ("scene.target_index_csv", "目标素材元数据 CSV（可选）", str, "speech"),
    ("scene.interferer_index_csv", "干扰素材元数据 CSV（可选）", str, "speech"),
    ("scene.pairing_mode", "目标与干扰配对方式", str, "speech"),
    ("scene.file_extensions", "扫描扩展名（逗号分隔）", lambda x: [v.strip() for v in x.split(",") if v.strip()], "speech"),
    ("scene.label_prefix", "标签前缀", str, "speech"),
    ("scene.dataset_split", "数据集划分标签", str, "speech"),
    ("scene.target_level_dbfs", "干净目标播放电平（dBFS）", float, "speech"),
    ("scene.interferer_level_dbfs", "干扰播放电平（dBFS）", float, "speech"),
    ("scene.repetitions", "整组采集重复次数", int, "speech"),
    ("scene.capture_strategy", "配对采集策略", str, "speech"),
    (
        "scene.require_supervised_pair",
        "混合场景强制包含仅目标监督（true/false）",
        lambda x: str(x).strip().lower() in {"1", "true", "yes", "是"},
        "speech",
    ),
    ("scene.ambient_duration_s", "环境底噪时长（秒）", float, "speech"),
    ("scene.countdown_s", "每组配对序列开始前倒计时（秒）", float, "speech"),
    ("scene.gap_s", "配对片段前后及片段间静音（秒）", float, "speech"),
    ("storage.root", "结果保存目录", str, "common"),
    ("storage.session_name", "测试名称", str, "common"),
]

ACTION_TO_LABEL = {"play_record": "同步播放并录制", "play": "仅播放", "record": "仅录制"}
LABEL_TO_ACTION = {label: action for action, label in ACTION_TO_LABEL.items()}
BACKEND_TO_LABEL = {"sounddevice": "真实声卡", "simulated": "模拟声卡"}
LABEL_TO_BACKEND = {label: backend for backend, label in BACKEND_TO_LABEL.items()}
SOURCE_MODE_TO_LABEL = {"single": "单个文件", "folders": "文件夹批量"}
LABEL_TO_SOURCE_MODE = {label: value for value, label in SOURCE_MODE_TO_LABEL.items()}
PAIRING_MODE_TO_LABEL = {"cycle": "按顺序循环配对", "cartesian": "全部组合配对"}
LABEL_TO_PAIRING_MODE = {label: value for value, label in PAIRING_MODE_TO_LABEL.items()}
RIR_STRATEGY_TO_LABEL = {
    "reconstruct_average": "固定次数 + 重构质检 + 全部有效平均（推荐）",
    "fixed_count": "固定次数 + 保留原始 take，之后再选",
}
LABEL_TO_RIR_STRATEGY = {
    label: value for value, label in RIR_STRATEGY_TO_LABEL.items()
}
FILE_PATH_FIELDS = {
    "general.source_file",
    "scene.target_file",
    "scene.interferer_file",
    "scene.target_index_csv",
    "scene.interferer_index_csv",
}
FOLDER_PATH_FIELDS = {"scene.target_folder", "scene.interferer_folder", "storage.root"}
SCENE_LABELS = {
    "ambient": "环境底噪",
    "target_only": "仅目标声源",
    "interferer_only": "仅干扰声源",
    "mixture": "目标与干扰同时播放",
}
SECTION_STARTS = {
    "audio.backend": ("声卡与录制通道", "common"),
    "general.action": ("基础播录设置", "basic"),
    "audio.target_output_channel": ("声源输出路由", "rir_speech"),
    "sweep.start_hz": ("ESS 扫频信号设置", "rir"),
    "repeats.strategy": ("RIR 重复与质量判定", "rir"),
    "scene.source_mode": ("语音增强素材与采集序列", "speech"),
    "storage.root": ("保存位置与实验标识", "common"),
}
RIR_PREVIEW_FIELDS = {
    "audio.sample_rate",
    "sweep.start_hz",
    "sweep.end_hz",
    "sweep.duration_s",
    "sweep.pre_silence_s",
    "sweep.post_silence_s",
    "sweep.fade_in_s",
    "sweep.fade_out_s",
    "sweep.level_dbfs",
}

AUDIO_PRESETS = {
    "标准监督：目标 + 干扰 + MIXED（推荐）": {
        "kind": "scene",
        "items": ["target_only", "interferer_only", "mixture"],
        "help": "一次连续播录依次采 target_only、interferer_only 和 mixed；mixed 与 target_only 复用完全相同的目标语音，额外的纯干扰可用于离线混合。",
    },
    "标准监督采集（推荐）": {
        "kind": "scene",
        "items": ["ambient", "target_only", "interferer_only", "mixture"],
        "help": "自动先录底噪，再用一次连续播录完成纯净目标、纯干扰和同时发声；纯净目标可作为混合录音的监督参考。",
    },
    "只采干净目标": {
        "kind": "scene",
        "items": ["target_only"],
        "help": "只让人工嘴/目标扬声器发声并录制麦克风。",
    },
    "只采干扰": {
        "kind": "scene",
        "items": ["interferer_only"],
        "help": "人工嘴不发声，只播放干扰源并录制麦克风。",
    },
    "普通单文件同步播录": {
        "kind": "io",
        "action": "play_record",
        "help": "从一个输出通道播放单个音频文件，同时录制麦克风。",
    },
    "仅录制麦克风": {
        "kind": "io",
        "action": "record",
        "help": "不播放声音，只录制所选麦克风通道。",
    },
    "仅播放音频": {
        "kind": "io",
        "action": "play",
        "help": "只播放单个音频文件，不录制麦克风。",
    },
    "自定义语音场景（高级）": {
        "kind": "scene",
        "items": None,
        "help": "打开高级设置后，可自行勾选底噪、纯净目标、纯干扰和同时发声。",
    },
}

ADVANCED_FIELDS = {
    "audio.backend",
    "audio.sample_rate",
    "audio.block_size",
    "general.action",
    "sweep.start_hz",
    "sweep.end_hz",
    "sweep.duration_s",
    "sweep.pre_silence_s",
    "sweep.post_silence_s",
    "sweep.fade_in_s",
    "sweep.fade_out_s",
    "sweep.rir_duration_s",
    "sweep.pre_peak_s",
    "repeats.correlation_threshold",
    "repeats.minimum_sweep_snr_db",
    "scene.pairing_mode",
    "scene.file_extensions",
    "scene.label_prefix",
    "scene.dataset_split",
    "scene.capture_strategy",
    "scene.require_supervised_pair",
    "scene.ambient_duration_s",
    "scene.countdown_s",
    "scene.gap_s",
    "storage.session_name",
}

COMMON_METADATA_FIELDS = [
    ("project_id", "项目编号", str),
    ("room_id", "房间编号", str),
    ("artificial_head_id", "人工头编号", str),
    ("headset_model_id", "耳机型号", str),
    ("headset_unit_id", "耳机个体编号", str),
    ("wearing_id", "本次佩戴编号", str),
    ("boom_pose_id", "麦杆姿态编号", str),
]
RIR_METADATA_FIELDS = [
    ("source_role", "RIR 声源类型（mouth/interferer）", str),
    ("source_id", "RIR 声源编号", str),
    ("azimuth_deg", "声源方位角（度）", float),
    ("elevation_deg", "声源俯仰角（度）", float),
    ("source_height_cm", "声源高度（厘米）", float),
    ("distance_cm", "声源距离（厘米）", float),
]
SPEECH_METADATA_FIELDS = [
    ("target.source_id", "目标声源编号", str),
    ("target.position_id", "目标位置编号", str),
    ("target.azimuth_deg", "目标方位角（度）", float),
    ("target.elevation_deg", "目标俯仰角（度）", float),
    ("target.height_m", "目标高度（米）", float),
    ("target.distance_m", "目标距离（米）", float),
    ("interferer.source_id", "干扰声源编号", str),
    ("interferer.position_id", "干扰位置编号", str),
    ("interferer.azimuth_deg", "干扰方位角（度）", float),
    ("interferer.elevation_deg", "干扰俯仰角（度）", float),
    ("interferer.height_m", "干扰高度（米）", float),
    ("interferer.distance_m", "干扰距离（米）", float),
]


def _metadata_get(metadata: dict, dotted: str):
    value = metadata
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part, "")
    return value


def _metadata_set(metadata: dict, dotted: str, value) -> None:
    target = metadata
    parts = dotted.split(".")
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = {}
            target[part] = nested
        target = nested
    target[parts[-1]] = value

def _get(config: ExperimentConfig, dotted: str):
    section, name = dotted.split(".")
    return getattr(getattr(config, section), name)


def _set(config: ExperimentConfig, dotted: str, value):
    section, name = dotted.split(".")
    setattr(getattr(config, section), name, value)


def _display(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


class CaptureGUI(tk.Tk):
    def __init__(
        self,
        initial_config: str | Path | None = None,
        run_once: str | None = None,
    ):
        super().__init__()
        self.title("声学采集工具")
        self.geometry("1500x900")
        self.minsize(1100, 700)
        self.config_path: Path | None = Path(initial_config).resolve() if initial_config else None
        self.config_data = load_config(self.config_path) if self.config_path else ExperimentConfig()
        self.variables = {name: tk.StringVar() for name, _, _, _ in FIELDS}
        self.mode_var = tk.StringVar(value="rir")
        self.audio_preset_var = tk.StringVar(value=next(iter(AUDIO_PRESETS)))
        self.advanced_var = tk.BooleanVar(value=False)
        self.field_rows: dict[str, tuple[tk.Widget, tk.Widget]] = {}
        self.section_widgets: list[tuple[tk.Widget, str]] = []
        self.device_boxes: dict[str, ttk.Combobox] = {}
        self.host_api_box: ttk.Combobox | None = None
        self.audio_preset_widgets: list[tk.Widget] = []
        self.item_vars = {
            name: tk.BooleanVar(value=True)
            for name in ("ambient", "target_only", "interferer_only", "mixture")
        }
        self.metadata_summary_var = tk.StringVar()
        self.events: queue.Queue[object] = queue.Queue()
        self._busy = False
        self._stop_event = threading.Event()
        self._active_backend = None
        self._worker_thread: threading.Thread | None = None
        self._busy_kind: str | None = None
        self._scene_scan_thread: threading.Thread | None = None
        self._cancel_watch_after_id: str | None = None
        self.checklist_path: Path | None = None
        self.checklist_row: dict | None = None
        self.checklist_kind: str | None = None
        self._logged_preflight_warning_ids: set[str] = set()
        self._preview_after_id: str | None = None
        self._build_menu()
        self._build()
        self._load_values()
        for name in RIR_PREVIEW_FIELDS:
            self.variables[name].trace_add("write", self._schedule_rir_preview)
        # A fixed 1500x900 logical size can extend beyond the work area on
        # Windows machines using 125%/150% display scaling.  Starting maximized
        # keeps the action buttons and log visible on laboratory laptops.
        if sys.platform == "win32":
            self.state("zoomed")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll_events)
        if run_once:
            self.after(800, lambda: self.run(run_once))

    def _build(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        for text, command in (
            ("打开配置", self.open_config),
            ("保存", self.save),
            ("另存为", self.save_as),
            ("查看音频设备", self.show_devices),
            ("检查声卡", self.check_hardware),
            ("检查配置", self.validate_config),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=3)
        self.quick_record_button = ttk.Button(
            toolbar,
            text="● 快速录音",
            command=self.start_quick_recording,
        )
        self.quick_record_button.pack(side="left", padx=(12, 3))
        ttk.Button(toolbar, text="测试清单 (.xlsx)", command=self.open_checklist).pack(
            side="left", padx=3
        )

        mode_bar = ttk.Frame(self, padding=(10, 0, 10, 6))
        mode_bar.pack(fill="x")
        ttk.Label(mode_bar, text="测试模式", font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(0, 8))
        for value, label in (
            ("rir", "RIR 采集"),
            ("audio", "音频 / 语音采集"),
        ):
            ttk.Radiobutton(
                mode_bar, text=label, value=value, variable=self.mode_var, command=self._set_mode
            ).pack(side="left", padx=8)
        advanced = ttk.Checkbutton(
            mode_bar,
            text="显示高级设置",
            variable=self.advanced_var,
            command=self._set_mode,
        )
        advanced.pack(side="right")

        preset_bar = ttk.Frame(self, padding=(10, 0, 10, 6))
        preset_label = ttk.Label(preset_bar, text="采集方案")
        preset_box = ttk.Combobox(
            preset_bar,
            textvariable=self.audio_preset_var,
            values=list(AUDIO_PRESETS),
            state="readonly",
            width=26,
        )
        preset_box.bind("<<ComboboxSelected>>", self._audio_preset_changed)
        self.audio_preset_help = ttk.Label(
            preset_bar,
            foreground="#24415c",
            wraplength=760,
        )
        preset_label.pack(side="left")
        preset_box.pack(side="left", padx=8)
        self.audio_preset_help.pack(side="left", fill="x", expand=True)
        self.audio_preset_widgets = [preset_bar]
        preset_bar.pack(fill="x")

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
        self.main_paned = paned
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned)
        self.viewer = ResultsViewer(paned)
        paned.add(left, minsize=440, width=570)
        paned.add(self.viewer, minsize=560)

        parameter_area = ttk.Frame(left)
        parameter_area.pack(fill="both", expand=True)
        canvas = tk.Canvas(parameter_area, highlightthickness=0)
        scroll = ttk.Scrollbar(parameter_area, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=12)
        body.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.bind_all(
            "<MouseWheel>",
            lambda event: self._scroll_parameter_area(event, canvas, parameter_area),
            add="+",
        )

        row = 0
        for name, label, _, _ in FIELDS:
            if name in SECTION_STARTS:
                title, group = SECTION_STARTS[name]
                section = ttk.Frame(body, padding=(0, 8, 0, 3))
                ttk.Label(section, text=title, font=("TkDefaultFont", 10, "bold")).pack(
                    side="left", padx=(0, 8)
                )
                ttk.Separator(section, orient="horizontal").pack(side="left", fill="x", expand=True)
                section.grid(row=row, column=0, columnspan=2, sticky="ew")
                self.section_widgets.append((section, group))
                row += 1
            label_widget = ttk.Label(body, text=label, width=31)
            label_widget.grid(row=row, column=0, sticky="w", pady=2)
            if name == "audio.backend":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(BACKEND_TO_LABEL.values()),
                    state="readonly",
                )
            elif name == "audio.host_api":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    state="readonly",
                )
                input_widget.bind("<<ComboboxSelected>>", self._host_api_changed)
                self.host_api_box = input_widget
            elif name == "general.action":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(ACTION_TO_LABEL.values()),
                    state="readonly",
                )
            elif name == "scene.source_mode":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(SOURCE_MODE_TO_LABEL.values()),
                    state="readonly",
                )
                input_widget.bind("<<ComboboxSelected>>", lambda _: self._update_scene_source_fields())
            elif name == "scene.pairing_mode":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(PAIRING_MODE_TO_LABEL.values()),
                    state="readonly",
                )
            elif name == "repeats.strategy":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(RIR_STRATEGY_TO_LABEL.values()),
                    state="readonly",
                )
                input_widget.bind("<<ComboboxSelected>>", lambda _: self._set_mode())
            elif name in {"audio.input_device", "audio.output_device"}:
                input_widget = ttk.Combobox(body, textvariable=self.variables[name])
                self.device_boxes[name] = input_widget
            elif name in FILE_PATH_FIELDS | FOLDER_PATH_FIELDS:
                input_widget = ttk.Frame(body)
                ttk.Button(
                    input_widget,
                    text="选择文件夹" if name in FOLDER_PATH_FIELDS else "选择文件",
                    command=lambda field=name: self._browse_path(field),
                ).pack(side="right", padx=(5, 0))
                ttk.Entry(input_widget, textvariable=self.variables[name], width=20).pack(
                    side="left", fill="x", expand=True
                )
            else:
                input_widget = ttk.Entry(body, textvariable=self.variables[name], width=65)
            input_widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.field_rows[name] = (label_widget, input_widget)
            row += 1
        ttk.Label(body, text="场景采集项目").grid(row=row, column=0, sticky="nw", pady=6)
        checks = ttk.Frame(body)
        checks.grid(row=row, column=1, sticky="w")
        self.scene_widgets = (body.grid_slaves(row=row, column=0)[0], checks)
        for name, variable in self.item_vars.items():
            ttk.Checkbutton(
                checks,
                text=SCENE_LABELS[name],
                variable=variable,
                command=self._set_mode,
            ).pack(side="left", padx=4)
        row += 1
        self.metadata_summary_panel = ttk.Frame(body, padding=(0, 8, 0, 4))
        self.metadata_summary_panel.grid(row=row, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            self.metadata_summary_panel,
            text="实验标签",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="left")
        ttk.Label(
            self.metadata_summary_panel,
            textvariable=self.metadata_summary_var,
            foreground="#24415c",
            wraplength=360,
        ).pack(side="left", fill="x", expand=True, padx=10)
        self.edit_metadata_button = ttk.Button(
            self.metadata_summary_panel,
            text="编辑实验标签",
            command=self.edit_experiment_labels,
        )
        self.edit_metadata_button.pack(side="right")
        row += 1
        self.metadata_label = ttk.Label(body, text="实验元数据（JSON）")
        self.metadata_label.grid(row=row, column=0, sticky="nw", pady=4)
        self.metadata = tk.Text(body, height=8, width=65)
        self.metadata.grid(row=row, column=1, sticky="ew")
        controls = ttk.Frame(left, padding=(8, 5))
        controls.pack(fill="x", side="bottom", before=parameter_area)
        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(0, 5))
        self.start_button = ttk.Button(actions, command=self.start_current_workflow)
        self.scene_scan_button = ttk.Button(actions, text="扫描并预览文件", command=self.scan_scene_sources)
        self.stop_button = ttk.Button(
            actions,
            text="停止录制 / 测试",
            command=self.stop_capture,
            state="disabled",
        )
        self.stop_button.pack(side="right", padx=4)
        self.actions = actions
        self.experiment_status = ttk.Label(
            controls,
            text="RIR 使用默认 ESS 即可；只有需要修改扫频或质量阈值时才打开高级设置。每次改变佩戴、角度、高度或麦杆姿态后，请开始一个新实验并重新命名。",
            foreground="#24415c",
            wraplength=540,
        )
        self.experiment_status.pack(fill="x", pady=(0, 5))
        self.log = tk.Text(controls, height=9, state="disabled", bg="#151515", fg="#dddddd")
        self.log.pack(fill="x")
        body.columnconfigure(1, weight=1)
        self._set_mode()

    def _build_menu(self):
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="打开配置", command=self.open_config)
        file_menu.add_command(label="保存配置", command=self.save)
        file_menu.add_command(label="配置另存为", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="打开测试清单 (.xlsx)", command=self.open_checklist)
        file_menu.add_command(label="新建测试清单模板", command=self.new_checklist_template)
        file_menu.add_command(label="导入人工质检 labels.xlsx", command=self.import_label_review)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._close)
        menu.add_cascade(label="文件", menu=file_menu)
        device_menu = tk.Menu(menu, tearoff=False)
        device_menu.add_command(label="查看音频设备", command=self.show_devices)
        device_menu.add_command(label="检查声卡通道", command=self.check_hardware)
        device_menu.add_command(label="检查当前配置", command=self.validate_config)
        menu.add_cascade(label="工具", menu=device_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(
            label="关于",
            command=lambda: messagebox.showinfo("关于", "声学采集工具\n多通道播录、脉冲响应与语音增强数据采集"),
        )
        menu.add_cascade(label="帮助", menu=help_menu)
        self.configure(menu=menu)

    @staticmethod
    def _scroll_parameter_area(event, canvas: tk.Canvas, parameter_area: tk.Widget):
        """Scroll parameters only when the pointer is over the left pane."""
        if event.state & 0x0004:  # Ctrl is reserved for waveform zooming.
            return None
        widget = parameter_area.winfo_containing(event.x_root, event.y_root)
        current = widget
        while current is not None:
            if current is parameter_area:
                direction = -1 if event.delta > 0 else 1
                canvas.yview_scroll(direction, "units")
                return "break"
            current = getattr(current, "master", None)
        return None

    def _load_values(self):
        for name, _, _, _ in FIELDS:
            value = _get(self.config_data, name)
            if name == "general.action":
                shown = ACTION_TO_LABEL.get(value, _display(value))
            elif name == "audio.backend":
                shown = BACKEND_TO_LABEL.get(value, _display(value))
            elif name == "scene.source_mode":
                shown = SOURCE_MODE_TO_LABEL.get(value, _display(value))
            elif name == "scene.pairing_mode":
                shown = PAIRING_MODE_TO_LABEL.get(value, _display(value))
            elif name == "repeats.strategy":
                if value == "adaptive_select":
                    value = "reconstruct_average"
                shown = RIR_STRATEGY_TO_LABEL.get(value, _display(value))
            else:
                shown = _display(value)
            self.variables[name].set(shown)
        for name, variable in self.item_vars.items():
            variable.set(name in self.config_data.scene.items)
        item_set = set(self.config_data.scene.items)
        inferred = "自定义语音场景（高级）"
        for label, preset in AUDIO_PRESETS.items():
            if preset.get("kind") == "scene" and preset.get("items") is not None:
                if item_set == set(preset["items"]):
                    inferred = label
                    break
        self.audio_preset_var.set(inferred)
        self.metadata.delete("1.0", "end")
        self.metadata.insert("1.0", json.dumps(self.config_data.metadata, ensure_ascii=False, indent=2))
        self._update_metadata_summary()
        self._update_scene_source_fields()
        self._refresh_device_choices()

    def _apply_values(self) -> ExperimentConfig:
        for name, _, converter, _ in FIELDS:
            raw = self.variables[name].get().strip()
            if name == "general.action":
                value = LABEL_TO_ACTION.get(raw, raw)
            elif name == "audio.backend":
                value = LABEL_TO_BACKEND.get(raw, raw)
            elif name == "scene.source_mode":
                value = LABEL_TO_SOURCE_MODE.get(raw, raw)
            elif name == "scene.pairing_mode":
                value = LABEL_TO_PAIRING_MODE.get(raw, raw)
            elif name == "repeats.strategy":
                value = LABEL_TO_RIR_STRATEGY.get(raw, raw)
            else:
                value = converter(raw)
            # Numeric strings are allowed as sounddevice indices.
            if name in {"audio.input_device", "audio.output_device"}:
                prefix = raw.partition(":")[0]
                if prefix.isdigit():
                    value = int(prefix)
            if name in {"audio.input_device", "audio.output_device"} and raw == "":
                value = None
            if name == "audio.host_api" and raw == "":
                value = None
            _set(self.config_data, name, value)
        self.config_data.scene.items = [name for name, var in self.item_vars.items() if var.get()]
        self.config_data.metadata = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
        self._update_metadata_summary()
        self.config_data.validate()
        return self.config_data

    def _update_metadata_summary(self):
        try:
            metadata = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
        except json.JSONDecodeError:
            self.metadata_summary_var.set("JSON 无效，请打开高级设置修正")
            return
        parts = [
            f"人工头={metadata.get('artificial_head_id') or '未填'}",
            f"耳机={metadata.get('headset_unit_id') or '未填'}",
            f"佩戴={metadata.get('wearing_id') or '未填'}",
            f"麦杆={metadata.get('boom_pose_id') or '未填'}",
        ]
        if self.checklist_row is not None:
            parts.insert(0, f"清单#{self.checklist_row.get('_row_number')}")
        if self.mode_var.get() == "rir":
            parts.extend(
                [
                    f"声源={metadata.get('source_role') or '未填'}",
                    f"az={metadata.get('azimuth_deg', '未填')}",
                    f"高度={metadata.get('source_height_cm', '未填')}cm",
                ]
            )
        else:
            interferer = metadata.get("interferer") or {}
            parts.extend(
                [
                    f"干扰az={interferer.get('azimuth_deg', '未填')}",
                    f"高度={interferer.get('height_m', '未填')}m",
                ]
            )
        self.metadata_summary_var.set("  |  ".join(parts))

    def edit_experiment_labels(self):
        """Edit structured labels without requiring users to write JSON."""
        try:
            metadata = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
        except json.JSONDecodeError as exc:
            messagebox.showerror("实验元数据无效", str(exc))
            return
        fields = list(COMMON_METADATA_FIELDS)
        try:
            microphone_channels = [
                int(value.strip())
                for value in self.variables["audio.input_channels"].get().split(",")
                if value.strip()
            ]
        except (KeyError, ValueError):
            microphone_channels = list(self.config_data.audio.input_channels)
        fields.extend(
            (f"microphone_{channel}", f"录制通道 {channel} 含义", str)
            for channel in microphone_channels
        )
        fields += (
            RIR_METADATA_FIELDS if self.mode_var.get() == "rir" else SPEECH_METADATA_FIELDS
        )
        window = tk.Toplevel(self)
        window.title("编辑实验标签")
        window.transient(self)
        window.geometry("600x720")
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        variables: dict[str, tk.StringVar] = {}
        for row, (key, label, _converter) in enumerate(fields):
            ttk.Label(body, text=label, width=30).grid(row=row, column=0, sticky="w", pady=3)
            variable = tk.StringVar(value=_display(_metadata_get(metadata, key)))
            variables[key] = variable
            if key == "source_role":
                widget = ttk.Combobox(
                    body,
                    textvariable=variable,
                    values=["mouth", "interferer"],
                    state="readonly",
                )
            else:
                widget = ttk.Entry(body, textvariable=variable, width=40)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
        body.columnconfigure(1, weight=1)

        def apply_labels():
            try:
                for key, _label, converter in fields:
                    raw = variables[key].get().strip()
                    _metadata_set(metadata, key, "" if not raw else converter(raw))
            except ValueError as exc:
                messagebox.showerror("实验标签无效", str(exc), parent=window)
                return
            if self.mode_var.get() != "rir":
                preset = AUDIO_PRESETS[self.audio_preset_var.get()]
                items = set(preset.get("items") or self.config_data.scene.items)
                metadata.setdefault("target", {})["active"] = bool(
                    items & {"target_only", "mixture"}
                )
                metadata.setdefault("interferer", {})["active"] = bool(
                    items & {"interferer_only", "mixture"}
                )
            self.metadata.delete("1.0", "end")
            self.metadata.insert("1.0", json.dumps(metadata, ensure_ascii=False, indent=2))
            self._update_metadata_summary()
            window.destroy()

        actions = ttk.Frame(window, padding=10)
        actions.pack(fill="x")
        ttk.Button(actions, text="取消", command=window.destroy).pack(side="right", padx=4)
        ttk.Button(actions, text="保存标签", command=apply_labels).pack(side="right", padx=4)
        window.grab_set()

    def new_checklist_template(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 测试清单", "*.xlsx")],
            initialfile="acoustic_capture_checklist.xlsx",
        )
        if not path:
            return
        try:
            create_checklist(path)
            messagebox.showinfo(
                "测试清单已创建",
                "已创建空白测试清单。\n\n每一行代表一次独立物理实验；现场只需选择行并开始采集。",
            )
        except Exception as exc:
            messagebox.showerror("创建测试清单失败", str(exc))

    def open_checklist(self):
        path = filedialog.askopenfilename(
            filetypes=[("Excel 测试清单", "*.xlsx"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            rows = read_checklist(path)
        except Exception as exc:
            messagebox.showerror("打开测试清单失败", str(exc))
            return
        self.checklist_path = Path(path).resolve()
        window = tk.Toplevel(self)
        window.title("选择测试清单实验")
        window.transient(self)
        window.geometry("1080x560")
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="选择一行后，实验名和条件会自动带入；采集完成后 status 会自动更新。也可关闭窗口继续手工命名。",
            foreground="#24415c",
        ).pack(fill="x", pady=(0, 8))
        columns = (
            "row",
            "status",
            "workflow",
            "experiment_name",
            "wearing_id",
            "boom_pose_id",
            "source",
        )
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "row": "行",
            "status": "状态",
            "workflow": "类型",
            "experiment_name": "实验名",
            "wearing_id": "佩戴",
            "boom_pose_id": "麦杆",
            "source": "声源/位置",
        }
        widths = {
            "row": 55,
            "status": 85,
            "workflow": 110,
            "experiment_name": 280,
            "wearing_id": 90,
            "boom_pose_id": 110,
            "source": 240,
        }
        row_by_item: dict[str, dict] = {}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], stretch=column in {"experiment_name", "source"})
        for index, row in enumerate(rows):
            status = str(row.get("status") or "待采集")
            source = row.get("source_id") or row.get("interferer_position_id") or ""
            item = tree.insert(
                "",
                "end",
                values=(
                    row.get("_row_number"),
                    status,
                    row.get("workflow", ""),
                    row.get("experiment_name", ""),
                    row.get("wearing_id", ""),
                    row.get("boom_pose_id", ""),
                    source,
                ),
                tags=("completed",) if status == "已完成" else (),
            )
            row_by_item[item] = row
            if index == 0 and status != "已完成":
                tree.selection_set(item)
                tree.focus(item)
        tree.tag_configure("completed", foreground="#777777")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def choose():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("尚未选择实验", "请先选择测试清单中的一行。", parent=window)
                return
            row = row_by_item[selection[0]]
            try:
                kind = apply_checklist_row(self.config_data, row, self.checklist_path)
            except Exception as exc:
                messagebox.showerror("清单行无效", str(exc), parent=window)
                return
            self.checklist_row = row
            self.checklist_kind = kind
            self.mode_var.set("rir" if kind == "rir" else "audio")
            self._load_values()
            if kind == "io":
                io_label = next(
                    label
                    for label, preset in AUDIO_PRESETS.items()
                    if preset.get("kind") == "io"
                    and preset.get("action") == self.config_data.general.action
                )
                self.audio_preset_var.set(io_label)
            self._set_mode()
            name = str(row.get("experiment_name") or "")
            self.experiment_status.configure(
                text=f"已选择测试清单第 {row['_row_number']} 行：{name}。确认设备后直接点击开始；完成后清单会自动回写。"
            )
            self._append(f"已载入测试清单第 {row['_row_number']} 行：{name}")
            window.destroy()

        actions = ttk.Frame(window, padding=10)
        actions.pack(fill="x", side="bottom")
        ttk.Button(actions, text="取消", command=window.destroy).pack(side="right", padx=4)
        ttk.Button(actions, text="使用所选实验", command=choose).pack(side="right", padx=4)
        tree.bind("<Double-1>", lambda _event: choose())
        if not rows:
            messagebox.showinfo(
                "测试清单为空",
                "清单中还没有实验行。请在 Excel 的“采集清单”工作表添加实验后重新打开。",
                parent=window,
            )

    def _update_checklist_progress(
        self, status: str, *, completed_run: str = "", last_error: str = ""
    ) -> None:
        if self.checklist_path is None or self.checklist_row is None:
            return
        try:
            update_checklist_row(
                self.checklist_path,
                int(self.checklist_row["_row_number"]),
                status=status,
                completed_run=completed_run,
                last_error=last_error,
            )
            self._append(
                f"测试清单第 {self.checklist_row['_row_number']} 行已更新为：{status}"
            )
        except Exception as exc:
            self._append(f"测试清单自动回写失败（采集文件不受影响）：{exc}")

    def open_config(self):
        path = filedialog.askopenfilename(filetypes=[("配置文件", "*.yaml *.yml"), ("所有文件", "*.*")])
        if path:
            try:
                self.config_path = Path(path)
                self.config_data = load_config(path)
                self._load_values()
                self._append(f"已打开配置：{path}")
            except Exception as exc:
                messagebox.showerror("打开失败", str(exc))

    def import_label_review(self):
        path = filedialog.askopenfilename(
            title="选择已人工编辑的 labels.xlsx",
            filetypes=[("Excel 标签表", "*.xlsx")],
        )
        if not path:
            return
        try:
            outputs = import_reviewed_labels(Path(path).parent, path)
            messagebox.showinfo(
                "标签质检已导入",
                "已生成 labels_reviewed.jsonl；以后汇总训练数据时会优先使用它。\n\n"
                + str(outputs["jsonl"]),
            )
        except Exception as exc:
            messagebox.showerror("导入标签质检失败", str(exc))

    def save(self) -> bool:
        if self.config_path is None:
            return self.save_as()
        try:
            config = self._apply_values()
            resume_run = config.scene.resume_run
            config.scene.resume_run = ""
            try:
                save_config(config, self.config_path)
            finally:
                config.scene.resume_run = resume_run
            self._append(f"已保存配置：{self.config_path}")
            return True
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return False

    def save_as(self) -> bool:
        path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("配置文件", "*.yaml")])
        if not path:
            return False
        self.config_path = Path(path)
        return self.save()

    def validate_config(self):
        try:
            config = self._apply_values()
            if self.mode_var.get() == "rir":
                workflow = "rir"
            else:
                workflow = str(AUDIO_PRESETS[self.audio_preset_var.get()]["kind"])
            report = build_preflight_report(config, workflow)
            details = format_preflight_report(report)
            if not report.can_start:
                messagebox.showerror("专业预检未通过", details)
            elif report.warnings:
                messagebox.showwarning("配置可用（有自动提醒）", details)
            else:
                messagebox.showinfo("专业预检通过", details)
        except Exception as exc:
            messagebox.showerror("配置无效", str(exc))

    def show_devices(self):
        self._refresh_device_choices()
        window = tk.Toplevel(self)
        window.title("音频设备")
        text_widget = tk.Text(window, width=110, height=30)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", list_devices())

    def _browse_path(self, field: str):
        current = self.variables[field].get().strip()
        current_path = Path(current).expanduser() if current else Path.cwd()
        initial = str(current_path if current_path.is_dir() else current_path.parent)
        if field in FOLDER_PATH_FIELDS:
            selected = filedialog.askdirectory(title="选择文件夹", initialdir=initial)
        else:
            filetypes = (
                [("CSV 素材索引", "*.csv"), ("所有文件", "*.*")]
                if field.endswith("index_csv")
                else [("音频文件", "*.wav *.flac *.aiff *.aif"), ("所有文件", "*.*")]
            )
            selected = filedialog.askopenfilename(
                title="选择素材索引" if field.endswith("index_csv") else "选择音频文件",
                initialdir=initial,
                filetypes=filetypes,
            )
        if selected:
            self.variables[field].set(selected)

    def scan_scene_sources(self):
        try:
            config = self._apply_values()
        except Exception as exc:
            messagebox.showerror("扫描失败", str(exc))
            return

        if self._scene_scan_thread is not None and self._scene_scan_thread.is_alive():
            return
        self.scene_scan_button.configure(state="disabled")
        self._append("正在后台扫描语音素材文件夹；窗口仍可正常操作……")

        def worker():
            try:
                pairs = discover_source_pairs(config.scene)
                preview = [
                    (
                        f"{index:04d}  target="
                        f"{pair.target.name if pair.target else '(不使用)'}"
                        f"  |  interferer="
                        f"{pair.interferer.name if pair.interferer else '(不使用)'}"
                    )
                    for index, pair in enumerate(pairs[:30], 1)
                ]
                self.events.put(("SCENE_SCAN_DONE", len(pairs), preview))
            except Exception as exc:
                self.events.put(("SCENE_SCAN_ERROR", str(exc)))

        self._scene_scan_thread = threading.Thread(
            target=worker, daemon=True, name="acoustic-source-scan"
        )
        self._scene_scan_thread.start()

    def check_hardware(self):
        try:
            config = self._apply_values()
            result = check_hardware_settings(config.audio)
            details = format_hardware_status(result)
            if result.get("warnings"):
                messagebox.showwarning("声卡参数可用，但存在风险", details)
            else:
                messagebox.showinfo("声卡检查通过", details)
        except Exception as exc:
            messagebox.showerror("声卡检查失败", str(exc))

    def start_named_experiment(self, kind: str):
        """Require a fresh manual name before starting one physical experiment."""
        kind_label = {"rir": "RIR", "scene": "语音增强", "io": "音频播录"}[kind]
        checklist_selected = self.checklist_row is not None
        if checklist_selected:
            if self.checklist_kind != kind:
                messagebox.showerror(
                    "清单类型不匹配",
                    f"所选清单行要求 {self.checklist_kind}，当前按钮将运行 {kind}。请重新选择清单行。",
                )
                return
            name = str(self.checklist_row.get("experiment_name") or "").strip()
        else:
            name = simpledialog.askstring(
                f"开始新的 {kind_label} 实验",
                "请输入本次实验的唯一名称。\n"
                "换角度、高度、佩戴或麦杆姿态后，请使用一个新名称。\n\n"
                "示例：hs01_w01_b00_interferer_az090_h170",
                parent=self,
            )
            if name is None:
                return
            name = name.strip()
        if not name:
            messagebox.showwarning("实验名称不能为空", "没有开始采集，请重新点击开始并输入名称。")
            return
        existing_runs = []
        resumable_runs: list[Path] = []
        runs_root = Path(self.variables["storage.root"].get().strip()).expanduser()
        if runs_root.is_dir():
            for manifest_path in runs_root.glob("*/manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    old_metadata = manifest.get("metadata") or {}
                    old_name = old_metadata.get("experiment_name") or old_metadata.get("experiment_id")
                    manifest_kind = str(manifest.get("kind") or "")
                    same_kind = manifest_kind == kind or (
                        kind == "io" and manifest_kind.startswith("io_")
                    )
                    if (
                        same_kind
                        and manifest.get("status") == "completed"
                        and str(old_name) == name
                    ):
                        existing_runs.append(manifest_path.parent.name)
                    if (
                        kind == "scene"
                        and same_kind
                        and manifest.get("status") in {"running", "cancelled", "failed"}
                        and str(old_name) == name
                        and (manifest_path.parent / "metrics" / "scene_checkpoint.json").is_file()
                    ):
                        resumable_runs.append(manifest_path.parent)
                except (OSError, TypeError, json.JSONDecodeError):
                    continue
        resume_run = ""
        if resumable_runs:
            latest = max(resumable_runs, key=lambda path: path.stat().st_mtime)
            if messagebox.askyesno(
                "发现未完成实验",
                f"名称“{name}”有可续采结果：\n{latest}\n\n"
                "选择“是”会跳过已完成素材，从下一条继续；选择“否”会新建实验。",
                parent=self,
            ):
                resume_run = str(latest)
        if existing_runs and not messagebox.askyesno(
            "实验名称已经使用",
            f"名称“{name}”已有 {len(existing_runs)} 个完成结果。\n"
            "继续会生成一个重复实验名（文件不会覆盖）。是否继续？",
            parent=self,
        ):
            return
        try:
            metadata = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
        except json.JSONDecodeError as exc:
            messagebox.showerror("实验元数据无效", str(exc))
            return
        metadata["experiment_name"] = name
        metadata["experiment_id"] = name
        metadata["scene_id"] = name
        self.metadata.delete("1.0", "end")
        self.metadata.insert("1.0", json.dumps(metadata, ensure_ascii=False, indent=2))
        self.variables["storage.session_name"].set(name)
        self.config_data.scene.resume_run = resume_run
        self._append(f"本次实验名称：{name}")
        self._update_metadata_summary()
        self.run(kind)

    def _audio_preset_changed(self, _event=None):
        preset = AUDIO_PRESETS[self.audio_preset_var.get()]
        if preset.get("kind") == "io":
            self.variables["general.action"].set(
                ACTION_TO_LABEL[str(preset["action"])]
            )
        elif preset.get("items") is not None:
            selected = set(preset["items"])
            for name, variable in self.item_vars.items():
                variable.set(name in selected)
        self._set_mode()

    def start_current_workflow(self):
        if self.mode_var.get() == "rir":
            self.start_named_experiment("rir")
            return
        preset = AUDIO_PRESETS[self.audio_preset_var.get()]
        if preset["kind"] == "io":
            self.variables["general.action"].set(
                ACTION_TO_LABEL[str(preset["action"])]
            )
        elif preset.get("items") is not None:
            selected = set(preset["items"])
            for name, variable in self.item_vars.items():
                variable.set(name in selected)
        selected_items = {
            name for name, variable in self.item_vars.items() if variable.get()
        }
        real_backend = (
            LABEL_TO_BACKEND.get(
                self.variables["audio.backend"].get(),
                self.variables["audio.backend"].get(),
            )
            != "simulated"
        )
        input_device = self.variables["audio.input_device"].get().strip().casefold()
        output_device = self.variables["audio.output_device"].get().strip().casefold()
        if (
            preset["kind"] == "scene"
            and "mixture" in selected_items
            and real_backend
            and (not input_device or input_device != output_device)
            and not messagebox.askyesno(
                "输入输出未共享同一声卡时钟",
                "当前录制设备与播放设备不是同一个可验证的双工设备。\n\n"
                "可以继续做流程试采，但本次 mixed/target-only 不会标记为严格监督可用。\n"
                "正式训练数据应把人工嘴和干扰扬声器都接到 RME，并让输入输出都选择同一个 RME ASIO 设备。\n\n"
                "是否仍继续？",
                parent=self,
            )
        ):
            return
        self.start_named_experiment(str(preset["kind"]))

    def _host_api_changed(self, _event=None):
        self._refresh_device_choices(clear_devices=True)

    def _refresh_device_choices(self, clear_devices: bool = False):
        protocols = host_api_choices()
        if self.host_api_box is not None:
            self.host_api_box.configure(values=protocols)
        selected_protocol = self.variables["audio.host_api"].get().strip() or None
        self.device_boxes["audio.input_device"].configure(
            values=device_choices("input", selected_protocol)
        )
        self.device_boxes["audio.output_device"].configure(
            values=device_choices("output", selected_protocol)
        )
        if clear_devices:
            self.variables["audio.input_device"].set("")
            self.variables["audio.output_device"].set("")

    def run(self, kind: str):
        if self.checklist_row is None:
            if not self.save():
                return
            config = self.config_data
        else:
            try:
                config = self._apply_values()
            except Exception as exc:
                messagebox.showerror("清单实验配置无效", str(exc))
                return
        if kind != "scene":
            config.scene.resume_run = ""
        self._stop_event.clear()
        self._busy_kind = kind
        self._set_busy(True)
        self._append("正在后台预检、扫描素材并准备声卡；界面保持可操作。")

        def worker():
            try:
                # Folder recursion, availability checks and device opening can
                # block on OneDrive/network drives or a faulty audio driver.
                # None of these are allowed on Tk's event thread.
                report = assert_capture_ready(config, kind)
                for warning in report.warnings:
                    self.events.put(("PREFLIGHT_WARNING", warning.check_id, warning.title, warning.detail))
                self.events.put(("STATUS", "预检完成；正在打开声卡流…"))
                backend = create_backend(config.audio)
                self._active_backend = backend
                logger = lambda message: self.events.put(str(message))
                backend.set_progress_callback(
                    lambda update: self.events.put(("AUDIO_PROGRESS", update))
                )
                if kind == "rir":
                    store = capture_rir(
                        config,
                        backend,
                        log=logger,
                        progress=lambda path, take: self.events.put(
                            ("RIR_PROGRESS", str(path), take)
                        ),
                        stop_requested=self._stop_event.is_set,
                    )
                else:
                    operation = {
                        "io": capture_general_io,
                        "scene": capture_scene_block,
                    }[kind]
                    if kind == "scene":
                        store = capture_scene_block(
                            config,
                            backend,
                            log=logger,
                            stop_requested=self._stop_event.is_set,
                            progress=lambda update: self.events.put(("SCENE_PROGRESS", update)),
                        )
                    else:
                        store = operation(
                            config,
                            backend,
                            log=logger,
                            stop_requested=self._stop_event.is_set,
                        )
                self.events.put(f"__DONE__{store.root}")
            except Exception as exc:
                self.events.put(f"__ERROR__{exc}")
            finally:
                self._active_backend = None

        self._worker_thread = threading.Thread(
            target=worker, daemon=True, name="acoustic-capture-worker"
        )
        self._worker_thread.start()

    def _quick_recording_audio_config(self):
        """Read only the input settings needed by standalone recording.

        This intentionally avoids validating sweep, playback, metadata and
        dataset fields: none of them should be able to block a plain recording.
        """
        audio = deepcopy(self.config_data.audio)
        backend_text = self.variables["audio.backend"].get().strip()
        audio.backend = LABEL_TO_BACKEND.get(backend_text, backend_text)
        audio.host_api = self.variables["audio.host_api"].get().strip() or None
        device_text = self.variables["audio.input_device"].get().strip()
        prefix = device_text.partition(":")[0]
        audio.input_device = int(prefix) if prefix.isdigit() else (device_text or None)
        audio.sample_rate = int(self.variables["audio.sample_rate"].get().strip())
        audio.block_size = int(self.variables["audio.block_size"].get().strip())
        audio.input_channels = [
            int(value.strip())
            for value in self.variables["audio.input_channels"].get().split(",")
            if value.strip()
        ]
        if audio.backend not in {"sounddevice", "simulated"}:
            raise ValueError("请选择真实声卡或模拟声卡")
        if audio.sample_rate < 8_000:
            raise ValueError("采样率不能低于 8000 Hz")
        if audio.block_size != 0 and audio.block_size < 16:
            raise ValueError("缓冲区帧数必须为 0，或至少为 16")
        if not audio.input_channels:
            raise ValueError("至少选择一个录制通道")
        if any(channel < 1 for channel in audio.input_channels):
            raise ValueError("录制通道从 1 开始，必须为正整数")
        if len(set(audio.input_channels)) != len(audio.input_channels):
            raise ValueError("录制通道不能重复")
        return audio

    def start_quick_recording(self):
        """Choose one WAV path and immediately start a standalone recording."""
        if self._busy:
            messagebox.showinfo("正在录制", "请先停止当前录制或测试。")
            return
        initial_root = Path(self.variables["storage.root"].get().strip() or ".").expanduser()
        if not initial_root.is_dir():
            initial_root = Path.cwd()
        path_text = filedialog.asksaveasfilename(
            title="选择录音保存位置（选择后立即开始录制）",
            initialdir=str(initial_root.resolve()),
            initialfile=f"recording_{datetime.now():%Y%m%d_%H%M%S}.wav",
            defaultextension=".wav",
            filetypes=[("WAV 音频", "*.wav")],
        )
        if not path_text:
            return
        output_path = Path(path_text).expanduser().resolve()
        try:
            audio = self._quick_recording_audio_config()
        except Exception as exc:
            messagebox.showerror("录音设置无效", str(exc))
            return
        self._stop_event.clear()
        self._busy_kind = "quick_recording"
        self._set_busy(True)
        self.experiment_status.configure(
            text=f"快速录音中：{output_path.name}  |  点击右侧“停止录制 / 测试”即可保存"
        )
        self._append(
            f"快速录音准备中：{output_path}；采样率 {audio.sample_rate} Hz，"
            f"录制通道 {','.join(map(str, audio.input_channels))}"
        )

        def worker():
            try:
                backend = create_backend(audio)
                self._active_backend = backend
                backend.set_progress_callback(
                    lambda update: self.events.put(("QUICK_RECORD_PROGRESS", update))
                )
                status = backend.record_to_file(
                    output_path,
                    stop_requested=self._stop_event.is_set,
                    subtype=self.config_data.storage.wav_subtype,
                )
                self.events.put(("QUICK_RECORD_DONE", str(output_path), status))
            except Exception as exc:
                self.events.put(("QUICK_RECORD_ERROR", str(output_path), str(exc)))
            finally:
                self._active_backend = None

        self._worker_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="acoustic-quick-recording-worker",
        )
        self._worker_thread.start()

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy and self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except tk.TclError:
                pass
            self._preview_after_id = None
        state = "disabled" if busy else "normal"
        self.start_button.configure(state=state)
        self.scene_scan_button.configure(state=state)
        self.edit_metadata_button.configure(state=state)
        self.quick_record_button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")

    def stop_capture(self):
        if not self._busy or self._stop_event.is_set():
            return
        self._stop_event.set()
        self.stop_button.configure(state="disabled")
        self._append(
            "正在停止录音并写完 WAV 文件，请稍候……"
            if self._busy_kind == "quick_recording"
            else "正在停止当前测试，请稍候……"
        )
        backend = self._active_backend
        if backend is not None:
            try:
                backend.stop()
            except Exception as exc:
                self._append(f"停止音频流时收到提示：{exc}")
        self._schedule_cancel_watch()

    def _schedule_cancel_watch(self):
        """Keep the GUI usable even if a third-party audio driver ignores stop."""
        if self._cancel_watch_after_id is not None:
            return
        self._cancel_watch_after_id = self.after(250, self._check_cancel_watch)

    def _check_cancel_watch(self):
        self._cancel_watch_after_id = None
        worker = self._worker_thread
        if not self._busy or not self._stop_event.is_set() or worker is None:
            return
        if worker.is_alive():
            self._append("仍在等待声卡驱动停止；界面可继续响应。若超过超时限制，本次会自动报错退出。")
            self._cancel_watch_after_id = self.after(2_000, self._check_cancel_watch)

    def _set_mode(self):
        mode = self.mode_var.get()
        preset = AUDIO_PRESETS[self.audio_preset_var.get()]
        is_scene = mode == "audio" and preset["kind"] == "scene"
        is_basic = mode == "audio" and preset["kind"] == "io"
        visible_groups = {"common"}
        if mode == "rir":
            visible_groups.update({"rir", "rir_speech"})
        elif is_scene:
            visible_groups.update({"speech", "rir_speech"})
        elif is_basic:
            visible_groups.add("basic")

        advanced = self.advanced_var.get()
        action = str(preset.get("action") or "")
        if preset.get("items") is None:
            scene_items = {name for name, variable in self.item_vars.items() if variable.get()}
        else:
            scene_items = set(preset.get("items") or [])
        target_needed = bool(scene_items & {"target_only", "mixture"})
        interferer_needed = bool(scene_items & {"interferer_only", "mixture"})
        playback_needed = mode == "rir" or is_scene and (target_needed or interferer_needed) or (
            is_basic and action in {"play", "play_record"}
        )
        recording_needed = mode == "rir" or is_scene or (
            is_basic and action in {"record", "play_record"}
        )
        source_mode = LABEL_TO_SOURCE_MODE.get(
            self.variables.get("scene.source_mode", tk.StringVar(value="单个文件")).get(),
            "single",
        )
        for name, _, _, group in FIELDS:
            visible = group in visible_groups
            if name in ADVANCED_FIELDS and not advanced:
                visible = False
            if name == "general.action":
                visible = False
            if name == "audio.input_device" or name == "audio.input_channels":
                visible = visible and recording_needed
            elif name == "audio.output_device":
                visible = visible and playback_needed
            elif is_basic and name in {
                "general.source_file",
                "general.level_dbfs",
                "general.output_channel",
            }:
                visible = visible and playback_needed
            elif is_scene and name in {"audio.target_output_channel"}:
                visible = visible and target_needed
            elif is_scene and name in {
                "audio.interferer_output_channel",
                "scene.interferer_file",
                "scene.interferer_folder",
                "scene.interferer_level_dbfs",
            }:
                visible = visible and interferer_needed
            elif is_scene and name in {
                "scene.target_file",
                "scene.target_folder",
                "scene.target_level_dbfs",
            }:
                visible = visible and target_needed
            if name in {"scene.target_file", "scene.interferer_file"}:
                visible = visible and source_mode == "single"
            elif name in {"scene.target_folder", "scene.interferer_folder"}:
                visible = visible and source_mode == "folders"
            for widget in self.field_rows.get(name, ()):
                widget.grid() if visible else widget.grid_remove()
        for widget, group in self.section_widgets:
            widget.grid() if group in visible_groups else widget.grid_remove()
        for widget in self.scene_widgets:
            widget.grid() if is_scene and advanced else widget.grid_remove()
        if advanced:
            self.metadata_label.grid()
            self.metadata.grid()
        else:
            self.metadata_label.grid_remove()
            self.metadata.grid_remove()

        if mode == "rir" or is_scene:
            self.metadata_summary_panel.grid()
        else:
            self.metadata_summary_panel.grid_remove()
        self._update_metadata_summary()

        for widget in self.audio_preset_widgets:
            if mode == "audio":
                widget.pack(fill="x", before=self.main_paned)
            else:
                widget.pack_forget()
        self.audio_preset_help.configure(text=str(preset["help"]))
        self.start_button.pack_forget()
        self.scene_scan_button.pack_forget()
        if mode == "rir":
            self.start_button.configure(text="开始新的 RIR 实验")
        elif is_scene:
            self.start_button.configure(text="开始新的语音增强实验")
        else:
            self.start_button.configure(text="开始新的音频播录")
        self.start_button.pack(side="left", padx=4)
        if is_scene:
            self.scene_scan_button.pack(side="left", padx=4)
        if mode == "rir":
            self._schedule_rir_preview()
        elif is_scene and self.viewer.run_dir is None:
            self.viewer.show_speech_workflow_guide()

    def _schedule_rir_preview(self, *_):
        if self._busy or self.mode_var.get() != "rir":
            return
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except tk.TclError:
                pass
        self._preview_after_id = self.after(180, self._refresh_rir_preview)

    def _refresh_rir_preview(self):
        self._preview_after_id = None
        if self._busy or self.mode_var.get() != "rir":
            return
        try:
            sample_rate = int(self.variables["audio.sample_rate"].get().strip())
            start_hz = float(self.variables["sweep.start_hz"].get().strip())
            end_hz = float(self.variables["sweep.end_hz"].get().strip())
            duration_s = float(self.variables["sweep.duration_s"].get().strip())
            pre_silence_s = float(self.variables["sweep.pre_silence_s"].get().strip())
            post_silence_s = float(self.variables["sweep.post_silence_s"].get().strip())
            fade_in_s = float(self.variables["sweep.fade_in_s"].get().strip())
            fade_out_s = float(self.variables["sweep.fade_out_s"].get().strip())
            level_dbfs = float(self.variables["sweep.level_dbfs"].get().strip())
            if sample_rate < 8_000:
                raise ValueError("采样率不能低于 8000 Hz")
            if not 0 < start_hz < end_hz < sample_rate / 2:
                raise ValueError("扫频范围必须满足 0 < 起始频率 < 终止频率 < Nyquist")
            if duration_s <= 0 or pre_silence_s < 0 or post_silence_s < 0:
                raise ValueError("扫频时长必须为正，前后静音不能为负")
            if min(fade_in_s, fade_out_s) < 0 or max(fade_in_s, fade_out_s) > duration_s:
                raise ValueError("淡入淡出不能为负，且每一项不能超过扫频时长")
            if not -100 <= level_dbfs <= 0:
                raise ValueError("播放电平必须在 -100 到 0 dBFS 之间")
            sweep = exponential_sweep(
                sample_rate,
                start_hz,
                end_hz,
                duration_s,
                level_dbfs,
                fade_in_s,
                fade_out_s,
            )
            played = measurement_signal(sweep, sample_rate, pre_silence_s, post_silence_s)
            self.viewer.show_sweep_preview(
                played,
                sweep,
                sample_rate,
                start_hz,
                end_hz,
                duration_s,
                pre_silence_s,
                post_silence_s,
                level_dbfs,
                fade_in_s,
                fade_out_s,
            )
        except Exception as exc:
            self.viewer.summary_var.set(f"扫频参数暂时无法预览：{exc}")

    def _update_scene_source_fields(self):
        self._set_mode()

    def _append(self, message: str):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, tuple) and event and event[0] == "PREFLIGHT_WARNING":
                _, check_id, title, detail = event
                if check_id not in self._logged_preflight_warning_ids:
                    self._append(f"预检提醒：{title}：{detail}")
                    self._logged_preflight_warning_ids.add(check_id)
            elif isinstance(event, tuple) and event and event[0] == "SCENE_SCAN_DONE":
                _, pair_count, preview = event
                self._scene_scan_thread = None
                if not self._busy:
                    self.scene_scan_button.configure(state="normal")
                suffix = "" if pair_count <= 30 else f"\n……另有 {pair_count - 30} 组未显示"
                messagebox.showinfo(
                    "语音文件扫描结果",
                    f"共形成 {pair_count} 组采集任务。\n\n" + "\n".join(preview) + suffix,
                )
            elif isinstance(event, tuple) and event and event[0] == "SCENE_SCAN_ERROR":
                _, error = event
                self._scene_scan_thread = None
                if not self._busy:
                    self.scene_scan_button.configure(state="normal")
                messagebox.showerror("扫描失败", error)
            elif isinstance(event, tuple) and event and event[0] == "STATUS":
                self._append(str(event[1]))
            elif isinstance(event, tuple) and event and event[0] == "SCENE_PROGRESS":
                _, update = event
                if update.get("event") == "pair_loading":
                    self._append(
                        f"正在读取素材 {update['pair_index']}/{update['pair_count']}："
                        f"target={update['target_name']} | interferer={update['interferer_name']}"
                    )
                    self.viewer.run_label.configure(
                        text=(
                            f"Loading pair {update['pair_index']}/{update['pair_count']}: "
                            f"target={update['target_name']} | interferer={update['interferer_name']}"
                        )
                    )
                    self.viewer.summary_var.set("正在读取并重采样当前两条素材；完成后立即开始播录。")
                elif update.get("event") == "pair_prepared":
                    self._append(
                        f"素材对 {update['pair_index']}/{update['pair_count']}："
                        f"target={update['target_name']} | interferer={update['interferer_name']}"
                    )
                    self.viewer.show_live_speech_pair(
                        update["target"],
                        update["interferer"],
                        update["sample_rate"],
                        update["segments"],
                        target_name=update["target_name"],
                        interferer_name=update["interferer_name"],
                        pair_index=update["pair_index"],
                        pair_count=update["pair_count"],
                        stream_samples=update["stream_samples"],
                    )
            elif isinstance(event, tuple) and event and event[0] == "AUDIO_PROGRESS":
                _, update = event
                self.viewer.update_live_progress(
                    update.get("frames", 0),
                    update.get("total_frames", 1),
                    update.get("sample_rate", 1),
                    str(update.get("phase", "")),
                )
            elif isinstance(event, tuple) and event and event[0] == "QUICK_RECORD_PROGRESS":
                _, update = event
                frames = int(update.get("frames", 0))
                sample_rate = max(1, int(update.get("sample_rate", 1)))
                elapsed = frames / sample_rate
                self.experiment_status.configure(
                    text=(
                        f"快速录音中：已录 {elapsed:.1f} 秒  |  "
                        "点击“停止录制 / 测试”完成保存"
                    )
                )
            elif isinstance(event, tuple) and event and event[0] == "QUICK_RECORD_DONE":
                _, path, status = event
                self._set_busy(False)
                self._busy_kind = None
                self._worker_thread = None
                duration = float(status.get("duration_s", 0.0))
                channels = int(status.get("channels", 0))
                self.experiment_status.configure(
                    text="快速录音已保存。可继续录下一条，或切换到 RIR / 语音增强实验。"
                )
                self._append(
                    f"快速录音已保存：{path}（{channels} 通道，{duration:.2f} 秒）"
                )
                self.viewer.load_recording_file(path)
                messagebox.showinfo(
                    "录音已保存",
                    f"{channels} 通道，{duration:.2f} 秒\n\n保存到：\n{path}",
                )
            elif isinstance(event, tuple) and event and event[0] == "QUICK_RECORD_ERROR":
                _, path, error = event
                self._set_busy(False)
                self._busy_kind = None
                self._worker_thread = None
                self.experiment_status.configure(text="快速录音失败；请检查录制设备和通道。")
                self._append(f"快速录音失败：{error}")
                messagebox.showerror(
                    "快速录音失败",
                    f"{error}\n\n如果文件已经产生，其中可能保留了停止前的音频：\n{path}",
                )
            elif isinstance(event, tuple) and event and event[0] == "RIR_PROGRESS":
                _, path, take = event
                self._append(f"第 {take} 次 RIR 已完成，正在刷新右侧图形：{path}")
                self.viewer.load_run(path)
            elif isinstance(event, str) and event.startswith("__DONE__"):
                self._set_busy(False)
                self._busy_kind = None
                self._worker_thread = None
                path = event.removeprefix("__DONE__")
                self._append(f"测试完成：{path}")
                self.viewer.load_run(path)
                manifest_path = Path(path) / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "cancelled":
                    self._update_checklist_progress(
                        "待采集", completed_run=path, last_error="用户停止；可重新采集"
                    )
                    self._append(f"测试已停止，已完成的数据已保存：{path}")
                    messagebox.showinfo("测试已停止", f"已完成的数据已保存到：\n{path}")
                    continue
                self._update_checklist_progress("已完成", completed_run=path)
                warnings = manifest.get("summary", {}).get("warnings", [])
                if warnings:
                    messagebox.showwarning("测试完成，但存在质量警告", "\n".join(warnings) + f"\n\n结果已保存到：\n{path}")
                else:
                    messagebox.showinfo("测试完成", f"结果已保存到：\n{path}")
            elif isinstance(event, str) and event.startswith("__ERROR__"):
                self._set_busy(False)
                self._busy_kind = None
                self._worker_thread = None
                error = event.removeprefix("__ERROR__")
                self._update_checklist_progress("失败", last_error=error)
                self._append(f"错误：{error}")
                messagebox.showerror("测试失败", error)
            else:
                self._append(str(event))
        self.after(100, self._poll_events)

    def _close(self):
        if self._busy:
            self.stop_capture()
        self.viewer.stop_audio()
        self.destroy()


def main(config_path: str | Path | None = None, run_once: str | None = None):
    CaptureGUI(config_path, run_once=run_once).mainloop()


if __name__ == "__main__":
    main()
