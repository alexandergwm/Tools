"""Reusable result browser embedded in the Tk GUI.

This module only reads saved run artifacts. It never changes measurement data.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy.signal import spectrogram

rcParams["font.family"] = "DejaVu Sans"
rcParams["font.sans-serif"] = ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def discover_audio_files(run_dir: str | Path) -> dict[str, list[Path]]:
    """Classify saved WAV files for selectors in the result viewer."""
    root = Path(run_dir)
    playback = sorted((root / "references").glob("*.wav"))
    recordings = sorted((root / "raw").glob("*.wav"))
    rirs = sorted((root / "processed").glob("*rir.wav"))
    return {"playback": playback, "recording": recordings, "rir": rirs}


def display_points(data: np.ndarray, sample_rate: int, limit: int = 16_000):
    """Return a lightweight view of long audio while preserving all channels."""
    step = max(1, int(np.ceil(len(data) / limit)))
    indices = np.arange(0, len(data), step)
    return indices / sample_rate, data[indices]


def minmax_envelope(
    signal: np.ndarray, sample_rate: int, bins: int = 4_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a waveform for display without hiding fast oscillations.

    A fixed-step plot aliases an ESS once its instantaneous frequency approaches
    the plotting sample rate.  Keeping the minimum and maximum from each time
    bucket preserves the visible amplitude envelope instead.
    """
    values = np.asarray(signal).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if len(values) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    bucket_count = min(int(bins), len(values))
    edges = np.linspace(0, len(values), bucket_count + 1, dtype=np.int64)
    centers = np.empty(bucket_count, dtype=np.float64)
    lows = np.empty(bucket_count, dtype=np.float64)
    highs = np.empty(bucket_count, dtype=np.float64)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        chunk = values[start:stop]
        centers[index] = (start + stop - 1) / (2.0 * sample_rate)
        lows[index] = float(np.min(chunk))
        highs[index] = float(np.max(chunk))
    return centers, lows, highs


def select_audio_channel(data: np.ndarray, selection: str) -> tuple[np.ndarray, str]:
    """Select one signal for listening or spectrogram display."""
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("音频数据中没有可用通道")
    if selection.startswith("通道"):
        channel = int(selection.split()[-1]) - 1
        if channel < 0 or channel >= data.shape[1]:
            raise ValueError(f"音频只有 {data.shape[1]} 个通道")
        return data[:, channel], f"通道 {channel + 1}"
    if selection == "混合全部通道":
        return np.mean(data, axis=1), "全部通道混合"
    if len(data) == 0:
        return data[:, 0], "通道 1"
    channel = int(np.argmax(np.max(np.abs(data), axis=0)))
    return data[:, channel], f"通道 {channel + 1}"


def zoom_interval(
    current: tuple[float, float],
    bounds: tuple[float, float],
    center: float,
    step: float,
) -> tuple[float, float]:
    """Zoom a time interval around the cursor while remaining in data bounds."""
    bound_left, bound_right = bounds
    full_span = max(bound_right - bound_left, np.finfo(float).eps)
    left, right = current
    current_span = max(right - left, np.finfo(float).eps)
    factor = 0.8 ** float(np.clip(step, -8.0, 8.0))
    new_span = float(np.clip(current_span * factor, full_span / 100_000.0, full_span))
    ratio = float(np.clip((center - left) / current_span, 0.0, 1.0))
    new_left = center - ratio * new_span
    new_right = new_left + new_span
    if new_left < bound_left:
        new_right += bound_left - new_left
        new_left = bound_left
    if new_right > bound_right:
        new_left -= new_right - bound_right
        new_right = bound_right
    return max(bound_left, new_left), min(bound_right, new_right)


def pan_interval(
    current: tuple[float, float], bounds: tuple[float, float], delta: float
) -> tuple[float, float]:
    """Shift a time interval and clamp it to the available signal duration."""
    bound_left, bound_right = bounds
    left, right = current
    span = right - left
    if span >= bound_right - bound_left:
        return bounds
    left += delta
    right += delta
    if left < bound_left:
        right += bound_left - left
        left = bound_left
    if right > bound_right:
        left -= right - bound_right
        right = bound_right
    return left, right


def _preferred(paths: list[Path], names: tuple[str, ...]) -> Path | None:
    for name in names:
        matches = [path for path in paths if path.name == name]
        if matches:
            return matches[-1]
    return paths[-1] if paths else None


class ResultsViewer(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=(8, 4))
        self.run_dir: Path | None = None
        self.files: dict[str, list[Path]] = {"playback": [], "recording": [], "rir": []}
        self.path_by_label: dict[str, Path] = {}
        self.playback_var = tk.StringVar()
        self.recording_var = tk.StringVar()
        self.rir_var = tk.StringVar()
        self.listen_channel = tk.StringVar(value="自动选择电平最高的通道")
        self.listen_device = tk.StringVar(value="")
        self.monitor_db = tk.DoubleVar(value=-12.0)
        self.show_spectrogram = tk.BooleanVar(value=False)
        self.summary_var = tk.StringVar(value="完成一次测试后，这里会显示本次结果。")
        self._x_bounds: dict[object, tuple[float, float]] = {}
        self._drag_state: tuple[object, float, tuple[float, float]] | None = None
        self.preview_spec: dict[str, float] | None = None
        self._build()

    def _build(self):
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 5))
        ttk.Label(header, text="测试结果", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Button(header, text="打开历史结果", command=self.open_run).pack(side="right")

        self.run_label = ttk.Label(self, text="尚未加载结果", foreground="#555555")
        self.run_label.pack(fill="x", pady=(0, 6))

        selectors = ttk.LabelFrame(self, text="已保存信号", padding=6)
        selectors.pack(fill="x")
        self._selector(selectors, 0, "播放或参考信号", self.playback_var)
        self._selector(selectors, 1, "麦克风录制信号", self.recording_var)
        self._selector(selectors, 2, "计算得到的脉冲响应", self.rir_var)
        selectors.columnconfigure(1, weight=1)

        plot_options = ttk.Frame(self, padding=(0, 6, 0, 0))
        plot_options.pack(fill="x")
        ttk.Checkbutton(
            plot_options,
            text="显示语谱图（关闭时显示波形）",
            variable=self.show_spectrogram,
            command=self.refresh_plots,
        ).pack(side="left")
        ttk.Button(plot_options, text="恢复完整时间范围", command=self.reset_view).pack(side="left", padx=8)
        ttk.Label(
            plot_options,
            text="提示：鼠标移到图上，按住 Ctrl 滚轮缩放；按住左键水平拖动",
            foreground="#555555",
        ).pack(side="right")

        listen = ttk.Frame(self, padding=(0, 6))
        listen.pack(fill="x")
        ttk.Button(listen, text="▶ 试听播放信号", command=lambda: self.play_selected("playback")).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(listen, text="▶ 试听录制信号", command=lambda: self.play_selected("recording")).pack(
            side="left", padx=4
        )
        ttk.Button(listen, text="■ 停止试听", command=self.stop_audio).pack(side="left", padx=4)
        self.channel_box = ttk.Combobox(
            listen,
            textvariable=self.listen_channel,
            values=["自动选择电平最高的通道", "混合全部通道"],
            state="readonly",
            width=22,
        )
        self.channel_box.pack(side="left", padx=(12, 4))
        self.channel_box.bind("<<ComboboxSelected>>", self._channel_changed)
        ttk.Label(listen, text="监听/语谱图通道").pack(side="left", before=self.channel_box, padx=(12, 0))

        listen_options = ttk.Frame(self)
        listen_options.pack(fill="x", pady=(0, 5))
        ttk.Label(listen_options, text="试听设备").pack(side="left")
        ttk.Entry(listen_options, textvariable=self.listen_device, width=24).pack(side="left", padx=4)
        ttk.Label(listen_options, text="试听电平").pack(side="left", padx=(10, 2))
        ttk.Scale(listen_options, variable=self.monitor_db, from_=-40, to=0, length=120).pack(side="left")
        self.monitor_label = ttk.Label(listen_options, text="-12 分贝")
        self.monitor_label.pack(side="left", padx=3)
        self.monitor_db.trace_add(
            "write", lambda *_: self.monitor_label.configure(text=f"{self.monitor_db.get():.0f} 分贝")
        )

        self.figure = Figure(figsize=(7.2, 7.2), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(3, 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        for axis, title in zip(
            self.axes, ("Playback / reference", "Microphone recording", "Estimated impulse response")
        ):
            axis.set_title(title)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Amplitude")
            axis.grid(True, alpha=0.25)
        self.canvas.draw_idle()
        ttk.Label(self, textvariable=self.summary_var, anchor="w", justify="left").pack(fill="x", pady=4)

    def _channel_changed(self, _event=None):
        if self.show_spectrogram.get():
            self.refresh_plots()

    def reset_view(self):
        for axis, bounds in self._x_bounds.items():
            axis.set_xlim(bounds)
        self._drag_state = None
        self.canvas.draw_idle()

    def _selector(self, parent, row: int, label: str, variable: tk.StringVar):
        ttk.Label(parent, text=label, width=21).grid(row=row, column=0, sticky="w", pady=2)
        box = ttk.Combobox(parent, textvariable=variable, state="readonly")
        box.grid(row=row, column=1, sticky="ew", pady=2)
        box.bind("<<ComboboxSelected>>", lambda _: self.refresh_plots())
        setattr(self, f"_{('playback', 'recording', 'rir')[row]}_box", box)

    def open_run(self):
        path = filedialog.askdirectory(title="选择一次声学测试结果")
        if path:
            self.load_run(path)

    def show_sweep_preview(
        self,
        played: np.ndarray,
        sweep: np.ndarray,
        sample_rate: int,
        start_hz: float,
        end_hz: float,
        duration_s: float,
        pre_silence_s: float,
        post_silence_s: float,
        level_dbfs: float,
        fade_in_s: float,
        fade_out_s: float,
    ) -> None:
        """Show the ESS that would be played before a measurement starts."""
        self.run_dir = None
        self.preview_spec = {
            "sample_rate": float(sample_rate),
            "start_hz": float(start_hz),
            "end_hz": float(end_hz),
            "duration_s": float(duration_s),
            "pre_silence_s": float(pre_silence_s),
            "post_silence_s": float(post_silence_s),
            "level_dbfs": float(level_dbfs),
            "fade_in_s": float(fade_in_s),
            "fade_out_s": float(fade_out_s),
        }
        self.files = {"playback": [], "recording": [], "rir": []}
        self.path_by_label.clear()
        for category, variable in (
            ("playback", self.playback_var),
            ("recording", self.recording_var),
            ("rir", self.rir_var),
        ):
            variable.set("")
            getattr(self, f"_{category}_box").configure(values=[])
        self.run_label.configure(text="ESS 扫频实时预览（修改左侧参数后自动更新）")
        self._x_bounds.clear()
        self._drag_state = None

        detail_duration = min(duration_s, max(0.02, 8.0 / start_hz))
        detail_samples = min(len(sweep), max(2, int(round(detail_duration * sample_rate))))
        detail_times = pre_silence_s + np.arange(detail_samples) / sample_rate
        frequency_times = np.linspace(0.0, duration_s, min(2000, max(2, len(sweep))))
        frequencies = start_hz * (end_hz / start_hz) ** (frequency_times / duration_s)

        for axis in self.axes:
            axis.clear()
            axis.grid(True, alpha=0.25)

        # A compressed high-frequency waveform (or a peak envelope) looks like
        # dropouts even when the ESS is perfectly continuous.  The first panel
        # therefore shows only the playback *schedule*.  The next two panels
        # carry the actual waveform and frequency information.
        full_duration = len(played) / sample_rate
        sweep_start = pre_silence_s
        sweep_end = pre_silence_s + duration_s
        timeline = (
            (0.0, pre_silence_s, "#d9d9d9", f"Pre-silence\n{pre_silence_s:g} s"),
            (
                sweep_start,
                duration_s,
                "#4c9bd6",
                f"ESS sweep\n{duration_s:g} s  |  {start_hz:g} -> {end_hz:g} Hz  |  {level_dbfs:g} dBFS",
            ),
            (sweep_end, post_silence_s, "#d9d9d9", f"Post-silence\n{post_silence_s:g} s"),
        )
        self.axes[0].grid(False)
        for left, width, color, label in timeline:
            if width <= 0:
                continue
            self.axes[0].barh(
                0.5,
                width,
                left=left,
                height=0.48,
                color=color,
                edgecolor="#4a4a4a",
                linewidth=0.8,
            )
            self.axes[0].text(
                left + width / 2.0,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="#202020",
            )
        self.axes[0].axvline(sweep_start, color="#235f8f", linewidth=1.0, linestyle="--")
        self.axes[0].axvline(sweep_end, color="#235f8f", linewidth=1.0, linestyle="--")
        self.axes[0].set_title("Playback sequence (timeline only - not a waveform)")
        self.axes[0].set_yticks([])
        self.axes[0].set_ylim(0.0, 1.0)
        self.axes[0].set_xlabel("Playback time (s)")
        self.axes[0].set_xlim(0.0, max(full_duration, 1.0 / sample_rate))
        self._x_bounds[self.axes[0]] = self.axes[0].get_xlim()

        self.axes[1].plot(detail_times, sweep[:detail_samples], linewidth=1.0, color="#ff7f0e")
        self.axes[1].axhline(0.0, linewidth=0.5, color="#555555", alpha=0.5)
        self.axes[1].set_title(f"Actual ESS start: first {detail_duration * 1000.0:.0f} ms")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].set_xlabel("Playback time (s)")
        detail_bounds = (
            pre_silence_s,
            pre_silence_s + max(detail_samples - 1, 1) / sample_rate,
        )
        self.axes[1].set_xlim(detail_bounds)
        self._x_bounds[self.axes[1]] = detail_bounds

        absolute_frequency_times = frequency_times + pre_silence_s
        self.axes[2].semilogy(absolute_frequency_times, frequencies, linewidth=1.6, color="#2ca02c")
        self.axes[2].set_title("ESS instantaneous frequency")
        self.axes[2].set_ylabel("Frequency (Hz, log scale)")
        self.axes[2].set_xlabel("Playback time (s)")
        sweep_bounds = (sweep_start, sweep_end)
        self.axes[2].set_xlim(sweep_bounds)
        self.axes[2].set_ylim(max(1.0, start_hz * 0.8), end_hz * 1.2)
        self._x_bounds[self.axes[2]] = sweep_bounds

        peak = float(np.max(np.abs(sweep))) if len(sweep) else 0.0
        self.summary_var.set(
            f"扫频预览：{start_hz:g} → {end_hz:g} Hz，扫频 {duration_s:g} 秒，"
            f"淡入 {fade_in_s:g} 秒，淡出 {fade_out_s:g} 秒，总播放 {full_duration:g} 秒，"
            f"电平 {level_dbfs:g} dBFS，峰值 {peak:.4f}；"
            "第一图只表示前静音 / ESS / 后静音的播放顺序，不是波形，也不用于判断掉音"
        )
        self.canvas.draw_idle()

    def show_speech_workflow_guide(self) -> None:
        """Replace a stale ESS preview with the supervised speech workflow."""
        self.run_dir = None
        self.preview_spec = None
        self.files = {"playback": [], "recording": [], "rir": []}
        self.path_by_label.clear()
        for category, variable in (
            ("playback", self.playback_var),
            ("recording", self.recording_var),
            ("rir", self.rir_var),
        ):
            variable.set("")
            getattr(self, f"_{category}_box").configure(values=[])
        self.run_label.configure(text="语音增强采集：完成后自动显示本次多通道录音")
        self._x_bounds.clear()
        self._drag_state = None
        for axis in self.axes:
            axis.clear()
            axis.set_axis_off()
        self.axes[0].text(
            0.5,
            0.5,
            "TARGET ONLY  >  MIXED",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#1F4E78",
            transform=self.axes[0].transAxes,
        )
        self.axes[1].text(
            0.5,
            0.55,
            "One continuous full-duplex audio stream",
            ha="center",
            va="center",
            fontsize=13,
            transform=self.axes[1].transAxes,
        )
        self.axes[1].text(
            0.5,
            0.35,
            "Same target source samples and one shared full-duplex stream",
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
            transform=self.axes[1].transAxes,
        )
        self.axes[2].text(
            0.5,
            0.5,
            "TRAINING INPUT: mixture     |     LABEL: target only",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color="#548235",
            transform=self.axes[2].transAxes,
        )
        self.summary_var.set(
            "配对监督采集会在一次连续声卡流中录制 target-only 与 mixed；"
            "两段复用完全相同的目标语音。训练输入是 mixed，监督标签是同一行的 target-only。"
        )
        self.canvas.draw_idle()

    def load_run(self, run_dir: str | Path):
        root = Path(run_dir).resolve()
        if not (root / "manifest.json").is_file():
            messagebox.showerror("结果目录无效", "所选目录中没有 manifest.json。")
            return
        self.run_dir = root
        self.preview_spec = None
        self.files = discover_audio_files(root)
        self.path_by_label.clear()
        for category, paths in self.files.items():
            labels = [str(path.relative_to(root)) for path in paths]
            for label, path in zip(labels, paths):
                self.path_by_label[f"{category}:{label}"] = path
            box = getattr(self, f"_{category}_box")
            box.configure(values=labels)

        playback = _preferred(self.files["playback"], ("played.wav", "target_emitted.wav"))
        recording = self.files["recording"][-1] if self.files["recording"] else None
        rir = _preferred(self.files["rir"], ("average_rir.wav",))
        self.playback_var.set(str(playback.relative_to(root)) if playback else "")
        self.recording_var.set(str(recording.relative_to(root)) if recording else "")
        self.rir_var.set(str(rir.relative_to(root)) if rir else "")
        self.run_label.configure(text=str(root))
        self.refresh_plots()

    def _selected_path(self, category: str) -> Path | None:
        variable = {
            "playback": self.playback_var,
            "recording": self.recording_var,
            "rir": self.rir_var,
        }[category]
        label = variable.get()
        return self.path_by_label.get(f"{category}:{label}") if label else None

    def refresh_plots(self):
        summaries = []
        channel_max = 1
        self._x_bounds.clear()
        self._drag_state = None
        for axis, category, title in zip(
            self.axes,
            ("playback", "recording", "rir"),
            ("Playback / reference", "Microphone recording", "Estimated impulse response"),
        ):
            axis.clear()
            path = self._selected_path(category)
            if path is None:
                axis.text(0.5, 0.5, "No file selected", ha="center", va="center", transform=axis.transAxes)
                axis.set_title(title)
                continue
            data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            channel_max = max(channel_max, data.shape[1])
            duration = len(data) / sample_rate
            self._x_bounds[axis] = (0.0, max(duration, 1.0 / sample_rate))
            if self.show_spectrogram.get():
                signal, selected_label = select_audio_channel(data, self.listen_channel.get())
                self._plot_spectrogram(axis, signal, sample_rate)
            else:
                times, shown = display_points(data, sample_rate)
                for channel in range(shown.shape[1]):
                    axis.plot(times, shown[:, channel], linewidth=0.8, label=f"Channel {channel + 1}")
            axis.set_title(title)
            axis.set_xlabel("Time (s)")
            if self.show_spectrogram.get():
                axis.set_ylabel("Frequency (kHz)")
            else:
                axis.set_ylabel("Amplitude")
                axis.grid(True, alpha=0.25)
                if shown.shape[1] <= 12:
                    axis.legend(loc="upper right", ncol=min(4, shown.shape[1]), fontsize=8)
            axis.set_xlim(self._x_bounds[axis])
            peaks = np.max(np.abs(data), axis=0) if len(data) else np.zeros(data.shape[1])
            category_name = {"playback": "播放", "recording": "录制", "rir": "脉冲响应"}[category]
            summary = (
                f"{category_name}："
                f"{data.shape[1]} 通道，{len(data) / sample_rate:.3f} 秒，"
                f"峰值 {float(np.max(peaks)):.4f}"
            )
            if self.show_spectrogram.get():
                summary += f"，语谱图使用{selected_label}"
            summaries.append(summary)
        self.channel_box.configure(
            values=["自动选择电平最高的通道", "混合全部通道"]
            + [f"通道 {channel}" for channel in range(1, channel_max + 1)]
        )
        self.summary_var.set("  |  ".join(summaries) if summaries else "没有找到音频结果文件")
        self.canvas.draw_idle()

    @staticmethod
    def _plot_spectrogram(axis, signal: np.ndarray, sample_rate: int):
        if len(signal) < 8:
            axis.text(0.5, 0.5, "Signal is too short for a spectrogram", ha="center", va="center", transform=axis.transAxes)
            return
        if float(np.max(np.abs(signal))) <= 1e-10:
            axis.set_facecolor("#111111")
            axis.text(
                0.5,
                0.5,
                "The signal is silent",
                color="white",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            return
        nperseg = min(2048, len(signal))
        nperseg = max(8, 2 ** int(np.floor(np.log2(nperseg))))
        frequencies, times, magnitude = spectrogram(
            signal,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg * 3 // 4,
            scaling="spectrum",
            mode="magnitude",
        )
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
        maximum = float(np.max(db)) if db.size else -200.0
        axis.pcolormesh(
            times,
            frequencies / 1000.0,
            db,
            shading="auto",
            cmap="magma",
            vmin=maximum - 80.0,
            vmax=maximum,
        )

    @staticmethod
    def _modifier_pressed(event) -> bool:
        key = str(event.key or "").lower()
        modifiers = {str(item).lower() for item in (getattr(event, "modifiers", None) or ())}
        names = ("control", "ctrl")
        return any(name in key for name in names) or any(name in modifiers for name in names)

    def _on_scroll(self, event):
        axis = event.inaxes
        if axis not in self._x_bounds or event.xdata is None or not self._modifier_pressed(event):
            return
        axis.set_xlim(zoom_interval(axis.get_xlim(), self._x_bounds[axis], event.xdata, event.step))
        self.canvas.draw_idle()

    def _on_button_press(self, event):
        if event.button != 1 or event.inaxes not in self._x_bounds or event.x is None:
            return
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        self._drag_state = (event.inaxes, float(event.x), event.inaxes.get_xlim())

    def _on_motion(self, event):
        if self._drag_state is None or event.x is None:
            return
        axis, start_pixel, start_xlim = self._drag_state
        if event.inaxes is not axis:
            return
        width = max(float(axis.bbox.width), 1.0)
        span = start_xlim[1] - start_xlim[0]
        delta = -(float(event.x) - start_pixel) * span / width
        axis.set_xlim(pan_interval(start_xlim, self._x_bounds[axis], delta))
        self.canvas.draw_idle()

    def _on_button_release(self, event):
        if event.button == 1:
            self._drag_state = None

    def play_selected(self, category: str):
        path = self._selected_path(category)
        if path is None:
            messagebox.showinfo("试听", "尚未选择对应的音频文件。")
            return
        try:
            import sounddevice as sd

            data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            selection = self.listen_channel.get()
            monitor, resolved_selection = select_audio_channel(data, selection)
            monitor = monitor * (10.0 ** (self.monitor_db.get() / 20.0))
            device_text = self.listen_device.get().strip()
            device = int(device_text) if device_text.isdigit() else (device_text or None)
            sd.stop()
            sd.play(monitor, sample_rate, device=device, blocking=False)
            self.summary_var.set(
                f"正在试听：{path.name}（{resolved_selection}），试听电平 {self.monitor_db.get():.0f} 分贝"
            )
        except Exception as exc:
            messagebox.showerror("试听失败", str(exc))

    def stop_audio(self):
        try:
            import sounddevice as sd

            sd.stop()
            self.summary_var.set("试听已停止")
        except Exception as exc:
            messagebox.showerror("停止试听失败", str(exc))
