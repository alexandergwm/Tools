"""Small Tk GUI for editing common parameters and starting captures."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .audio import (
    check_hardware_settings,
    create_backend,
    device_choices,
    format_hardware_status,
    list_devices,
)
from .config import ExperimentConfig, load_config, save_config
from .general import capture_general_io
from .rir import capture_rir
from .scene import capture_scene_block, discover_source_pairs
from .signals import ESS_FORMULA
from .viewer import ResultsViewer


FIELDS = [
    ("audio.backend", "音频后端", str, "common"),
    ("audio.input_device", "录制设备名称或编号", str, "common"),
    ("audio.output_device", "播放设备名称或编号", str, "common"),
    ("audio.sample_rate", "采样率", int, "common"),
    ("audio.block_size", "缓冲区帧数", int, "common"),
    ("audio.input_channels", "录制通道（逗号分隔）", lambda x: [int(v) for v in x.split(",")], "common"),
    ("general.action", "播录操作", str, "basic"),
    ("general.source_file", "播放文件", str, "basic"),
    ("general.duration_s", "持续时间（秒）", float, "basic"),
    ("general.level_dbfs", "播放电平（满刻度分贝）", float, "basic"),
    ("general.output_channel", "播放输出通道", int, "basic"),
    ("audio.target_output_channel", "目标声源或扫频输出通道", int, "rir_speech"),
    ("audio.interferer_output_channel", "干扰声源输出通道", int, "speech"),
    ("sweep.start_hz", "扫频起始频率（赫兹）", float, "rir"),
    ("sweep.end_hz", "扫频终止频率（赫兹）", float, "rir"),
    ("sweep.duration_s", "扫频持续时间（秒）", float, "rir"),
    ("sweep.pre_silence_s", "扫频前静音（秒）", float, "rir"),
    ("sweep.post_silence_s", "扫频后静音（秒）", float, "rir"),
    ("sweep.fade_s", "扫频淡入淡出（秒）", float, "rir"),
    ("sweep.level_dbfs", "扫频播放电平（满刻度分贝）", float, "rir"),
    ("sweep.rir_duration_s", "脉冲响应长度（秒）", float, "rir"),
    ("sweep.pre_peak_s", "脉冲峰值前保留时间（秒）", float, "rir"),
    ("repeats.minimum", "最少脉冲响应采集次数", int, "rir"),
    ("repeats.maximum", "最多脉冲响应采集次数", int, "rir"),
    ("repeats.correlation_threshold", "脉冲响应相关性阈值", float, "rir"),
    ("scene.source_mode", "语音来源模式", str, "speech"),
    ("scene.duration_s", "每条片段时长（秒，推荐 4）", lambda x: None if not x else float(x), "speech"),
    ("scene.target_file", "单文件模式：目标语音", str, "speech"),
    ("scene.interferer_file", "单文件模式：干扰声音", str, "speech"),
    ("scene.target_folder", "文件夹模式：目标语音目录", str, "speech"),
    ("scene.interferer_folder", "文件夹模式：干扰声音目录", str, "speech"),
    ("scene.pairing_mode", "目标与干扰配对方式", str, "speech"),
    ("scene.file_extensions", "扫描扩展名（逗号分隔）", lambda x: [v.strip() for v in x.split(",") if v.strip()], "speech"),
    ("scene.label_prefix", "标签前缀", str, "speech"),
    ("scene.dataset_split", "数据集划分标签", str, "speech"),
    ("scene.target_level_dbfs", "目标播放电平（满刻度分贝）", float, "speech"),
    ("scene.interferer_level_dbfs", "干扰播放电平（满刻度分贝）", float, "speech"),
    ("scene.repetitions", "场景重复次数", int, "speech"),
    ("scene.ambient_duration_s", "环境底噪时长（秒）", float, "speech"),
    ("scene.countdown_s", "每项开始前倒计时（秒）", float, "speech"),
    ("scene.gap_s", "每项结束后间隔（秒）", float, "speech"),
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
FILE_PATH_FIELDS = {"general.source_file", "scene.target_file", "scene.interferer_file"}
FOLDER_PATH_FIELDS = {"scene.target_folder", "scene.interferer_folder", "storage.root"}
SCENE_LABELS = {
    "ambient": "环境底噪",
    "target_only": "仅目标声源",
    "interferer_only": "仅干扰声源",
    "mixture": "目标与干扰同时播放",
}


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
        self.mode_var = tk.StringVar(value="basic")
        self.field_rows: dict[str, tuple[tk.Widget, tk.Widget]] = {}
        self.device_boxes: list[ttk.Combobox] = []
        self.item_vars = {
            name: tk.BooleanVar(value=True)
            for name in ("ambient", "target_only", "interferer_only", "mixture")
        }
        self.events: queue.Queue[str] = queue.Queue()
        self._build_menu()
        self._build()
        self._load_values()
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

        mode_bar = ttk.Frame(self, padding=(10, 0, 10, 6))
        mode_bar.pack(fill="x")
        ttk.Label(mode_bar, text="测试模式", font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(0, 8))
        for value, label in (
            ("basic", "基础播录"),
            ("rir", "房间脉冲响应采集"),
            ("speech", "语音增强数据采集"),
        ):
            ttk.Radiobutton(
                mode_bar, text=label, value=value, variable=self.mode_var, command=self._set_mode
            ).pack(side="left", padx=8)

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
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

        for row, (name, label, _, _) in enumerate(FIELDS):
            label_widget = ttk.Label(body, text=label, width=31)
            label_widget.grid(row=row, column=0, sticky="w", pady=2)
            if name == "audio.backend":
                input_widget = ttk.Combobox(
                    body,
                    textvariable=self.variables[name],
                    values=list(BACKEND_TO_LABEL.values()),
                    state="readonly",
                )
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
            elif name in {"audio.input_device", "audio.output_device"}:
                input_widget = ttk.Combobox(body, textvariable=self.variables[name])
                self.device_boxes.append(input_widget)
            elif name in FILE_PATH_FIELDS | FOLDER_PATH_FIELDS:
                input_widget = ttk.Frame(body)
                ttk.Entry(input_widget, textvariable=self.variables[name], width=54).pack(
                    side="left", fill="x", expand=True
                )
                ttk.Button(
                    input_widget,
                    text="选择文件夹" if name in FOLDER_PATH_FIELDS else "选择文件",
                    command=lambda field=name: self._browse_path(field),
                ).pack(side="left", padx=(5, 0))
            else:
                input_widget = ttk.Entry(body, textvariable=self.variables[name], width=65)
            input_widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.field_rows[name] = (label_widget, input_widget)
        row = len(FIELDS)
        ess_title = ttk.Label(body, text="ESS 生成公式", font=("TkDefaultFont", 10, "bold"))
        ess_title.grid(row=row, column=0, sticky="nw", pady=(8, 2))
        ess_formula = ttk.Label(
            body,
            text=ESS_FORMULA,
            justify="left",
            foreground="#24415c",
            wraplength=720,
        )
        ess_formula.grid(row=row, column=1, sticky="ew", pady=(8, 2))
        self.ess_widgets = (ess_title, ess_formula)
        row += 1
        ttk.Label(body, text="场景采集项目").grid(row=row, column=0, sticky="nw", pady=6)
        checks = ttk.Frame(body)
        checks.grid(row=row, column=1, sticky="w")
        self.scene_widgets = (body.grid_slaves(row=row, column=0)[0], checks)
        for name, variable in self.item_vars.items():
            ttk.Checkbutton(checks, text=SCENE_LABELS[name], variable=variable).pack(side="left", padx=4)
        row += 1
        ttk.Label(body, text="实验元数据（JSON）").grid(row=row, column=0, sticky="nw", pady=4)
        self.metadata = tk.Text(body, height=8, width=65)
        self.metadata.grid(row=row, column=1, sticky="ew")
        controls = ttk.Frame(left, padding=(8, 5))
        controls.pack(fill="x", side="bottom", before=parameter_area)
        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(0, 5))
        self.rir_button = ttk.Button(actions, text="开始脉冲响应采集", command=lambda: self.run("rir"))
        self.scene_button = ttk.Button(actions, text="开始语音增强数据采集", command=lambda: self.run("scene"))
        self.scene_scan_button = ttk.Button(actions, text="扫描并预览文件", command=self.scan_scene_sources)
        self.basic_button = ttk.Button(actions, text="开始基础播录", command=lambda: self.run("io"))
        self.actions = actions
        self.log = tk.Text(controls, height=9, state="disabled", bg="#151515", fg="#dddddd")
        self.log.pack(fill="x")
        body.columnconfigure(1, weight=1)
        self._refresh_device_choices()
        self._set_mode()

    def _build_menu(self):
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="打开配置", command=self.open_config)
        file_menu.add_command(label="保存配置", command=self.save)
        file_menu.add_command(label="配置另存为", command=self.save_as)
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
            else:
                shown = _display(value)
            self.variables[name].set(shown)
        for name, variable in self.item_vars.items():
            variable.set(name in self.config_data.scene.items)
        self.metadata.delete("1.0", "end")
        self.metadata.insert("1.0", json.dumps(self.config_data.metadata, ensure_ascii=False, indent=2))
        self._update_scene_source_fields()

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
            else:
                value = converter(raw)
            # Numeric strings are allowed as sounddevice indices.
            if name in {"audio.input_device", "audio.output_device"}:
                prefix = raw.partition(":")[0]
                if prefix.isdigit():
                    value = int(prefix)
            if name in {"audio.input_device", "audio.output_device"} and raw == "":
                value = None
            _set(self.config_data, name, value)
        self.config_data.scene.items = [name for name, var in self.item_vars.items() if var.get()]
        self.config_data.metadata = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
        self.config_data.validate()
        return self.config_data

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

    def save(self) -> bool:
        if self.config_path is None:
            return self.save_as()
        try:
            save_config(self._apply_values(), self.config_path)
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
            self._apply_values()
            messagebox.showinfo("配置检查", "配置有效，可以开始测试。")
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
            selected = filedialog.askopenfilename(
                title="选择音频文件",
                initialdir=initial,
                filetypes=[("音频文件", "*.wav *.flac *.aiff *.aif"), ("所有文件", "*.*")],
            )
        if selected:
            self.variables[field].set(selected)

    def scan_scene_sources(self):
        try:
            config = self._apply_values()
            pairs = discover_source_pairs(config.scene)
            preview = [
                f"{index:04d}  目标：{pair.target.name if pair.target else '不使用'}"
                f"  |  干扰：{pair.interferer.name if pair.interferer else '不使用'}"
                for index, pair in enumerate(pairs[:30], 1)
            ]
            suffix = "" if len(pairs) <= 30 else f"\n……另有 {len(pairs) - 30} 组未显示"
            messagebox.showinfo(
                "语音文件扫描结果",
                f"共形成 {len(pairs)} 组采集任务，每条按配置时长处理。\n\n"
                + "\n".join(preview)
                + suffix,
            )
        except Exception as exc:
            messagebox.showerror("扫描失败", str(exc))

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

    def _refresh_device_choices(self):
        choices = device_choices()
        for box in self.device_boxes:
            box.configure(values=choices)

    def run(self, kind: str):
        if not self.save():
            return
        config = self.config_data
        self._set_busy(True)

        def worker():
            try:
                backend = create_backend(config.audio)
                operation = {
                    "io": capture_general_io,
                    "rir": capture_rir,
                    "scene": capture_scene_block,
                }[kind]
                store = operation(config, backend, log=lambda message: self.events.put(str(message)))
                self.events.put(f"__DONE__{store.root}")
            except Exception as exc:
                self.events.put(f"__ERROR__{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.rir_button.configure(state=state)
        self.scene_button.configure(state=state)
        self.scene_scan_button.configure(state=state)
        self.basic_button.configure(state=state)

    def _set_mode(self):
        mode = self.mode_var.get()
        visible_groups = {"common", mode}
        if mode in {"rir", "speech"}:
            visible_groups.add("rir_speech")
        for name, _, _, group in FIELDS:
            for widget in self.field_rows.get(name, ()):
                widget.grid() if group in visible_groups else widget.grid_remove()
        for widget in self.scene_widgets:
            widget.grid() if mode == "speech" else widget.grid_remove()
        for widget in self.ess_widgets:
            widget.grid() if mode == "rir" else widget.grid_remove()
        for button in (self.basic_button, self.rir_button, self.scene_button, self.scene_scan_button):
            button.pack_forget()
        selected = {"basic": self.basic_button, "rir": self.rir_button, "speech": self.scene_button}[mode]
        selected.pack(side="left", padx=4)
        if mode == "speech":
            self.scene_scan_button.pack(side="left", padx=4)
        self._update_scene_source_fields()

    def _update_scene_source_fields(self):
        source_mode = LABEL_TO_SOURCE_MODE.get(
            self.variables.get("scene.source_mode", tk.StringVar(value="单个文件")).get(),
            "single",
        )
        single_fields = {"scene.target_file", "scene.interferer_file"}
        folder_fields = {
            "scene.target_folder",
            "scene.interferer_folder",
            "scene.pairing_mode",
            "scene.file_extensions",
        }
        for field in single_fields | folder_fields:
            for widget in self.field_rows.get(field, ()):
                if self.mode_var.get() != "speech":
                    widget.grid_remove()
                elif (field in single_fields) == (source_mode == "single"):
                    widget.grid()
                else:
                    widget.grid_remove()

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
            if event.startswith("__DONE__"):
                self._set_busy(False)
                path = event.removeprefix("__DONE__")
                self._append(f"测试完成：{path}")
                self.viewer.load_run(path)
                manifest_path = Path(path) / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                warnings = manifest.get("summary", {}).get("warnings", [])
                if warnings:
                    messagebox.showwarning("测试完成，但存在质量警告", "\n".join(warnings) + f"\n\n结果已保存到：\n{path}")
                else:
                    messagebox.showinfo("测试完成", f"结果已保存到：\n{path}")
            elif event.startswith("__ERROR__"):
                self._set_busy(False)
                error = event.removeprefix("__ERROR__")
                self._append(f"错误：{error}")
                messagebox.showerror("测试失败", error)
            else:
                self._append(event)
        self.after(100, self._poll_events)

    def _close(self):
        self.viewer.stop_audio()
        self.destroy()


def main(config_path: str | Path | None = None, run_once: str | None = None):
    CaptureGUI(config_path, run_once=run_once).mainloop()


if __name__ == "__main__":
    main()
