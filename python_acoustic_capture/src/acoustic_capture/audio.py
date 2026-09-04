"""Audio I/O backends.

SoundDeviceBackend addresses hardware channels using one-based values in the
configuration. SimulatedBackend is deterministic enough for tests and demos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .config import AudioConfig


@dataclass
class CaptureResult:
    microphones: np.ndarray
    status: dict[str, Any]


class AudioBackend(ABC):
    def __init__(self, config: AudioConfig):
        self.config = config

    @abstractmethod
    def play_record(self, output: np.ndarray) -> CaptureResult:
        """Play all output channels and record the configured microphones."""

    @abstractmethod
    def record(self, frames: int) -> CaptureResult:
        """Record without playback."""

    @abstractmethod
    def play(self, output: np.ndarray) -> dict[str, Any]:
        """Play without recording."""

    def stop(self) -> None:
        """Request that an active audio operation stop as soon as possible."""

    def set_progress_callback(self, callback) -> None:
        """Receive lightweight playback progress dictionaries when supported.

        Backends deliberately do not know about Tk.  The GUI installs a
        thread-safe queue writer here, while command-line callers can simply
        leave the callback unset.
        """


class SoundDeviceBackend(AudioBackend):
    """PortAudio backend with bounded, explicitly cancellable streams.

    ``sounddevice.playrec(..., blocking=True)`` is concise, but a driver that
    never reaches its callback leaves the calling worker blocked forever.  This
    matters on laptops where the Windows microphone and speakers are separate
    devices.  Owning the stream explicitly lets the GUI request cancellation
    and lets us surface a useful timeout instead of appearing to hang
    indefinitely.

    PortAudio/ASIO stream control is deliberately thread-affine: the thread
    that constructs and starts a stream is also the only thread that aborts
    and closes it.  The Tk thread merely sets ``_abort_requested``.  Several
    Windows ASIO drivers become unstable when a second helper thread invokes
    ``abort()`` or ``close()``.
    """

    def __init__(self, config: AudioConfig):
        super().__init__(config)
        self._stream_lock = threading.Lock()
        self._active_stream: Any | None = None
        self._stream_owner_thread_id: int | None = None
        self._abort_requested = threading.Event()
        self._progress_callback = None
        self._last_progress_at = 0.0
        self._last_check_warnings: list[str] = []

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def _emit_progress(self, phase: str, frames: int, total_frames: int, *, force: bool = False) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        now = time.monotonic()
        if not force and now - self._last_progress_at < 0.08:
            return
        self._last_progress_at = now
        try:
            callback(
                {
                    "phase": phase,
                    "frames": int(frames),
                    "total_frames": int(total_frames),
                    "sample_rate": int(self.config.sample_rate),
                }
            )
        except Exception:
            # Progress reporting must never interrupt an audio callback.
            pass

    def _module(self):
        _enable_windows_asio()
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - depends on local driver
            raise RuntimeError("无法加载 sounddevice/PortAudio，请检查安装和声卡驱动") from exc
        return sd

    @staticmethod
    def _callback_status(sd) -> dict[str, Any]:
        """Return PortAudio callback faults from the just-finished operation."""
        try:
            flags = sd.get_status()
        except Exception:
            return {"text": "", "xrun": False}
        names = ("input_underflow", "input_overflow", "output_underflow", "output_overflow")
        values = {name: bool(getattr(flags, name, False)) for name in names}
        return {
            "text": str(flags),
            "xrun": any(values.values()),
            **values,
        }

    def _devices(self):
        input_device = self.config.input_device if self.config.input_device is not None else self.config.device
        output_device = self.config.output_device if self.config.output_device is not None else self.config.device
        return input_device, output_device

    @staticmethod
    def _device_argument(device: str | int | None) -> str | int | None:
        """Normalize a GUI selection such as ``"24: Microphone ..."`` to 24.

        A bare numeric string is treated by sounddevice as a name search and
        can match the same device exposed by several Windows host APIs.  The
        integer index is unambiguous and is what the GUI displays.
        """
        if isinstance(device, str):
            prefix = device.partition(":")[0].strip()
            if prefix.isdigit():
                return int(prefix)
        return device

    def _device_status(self, sd) -> dict[str, Any]:
        def describe(device, kind: str):
            try:
                info = dict(sd.query_devices(device, kind=kind))
                host_api = info.get("hostapi")
                try:
                    host_api_name = sd.query_hostapis(host_api)["name"]
                except Exception:
                    host_api_name = None
                return {
                    "index": info.get("index"),
                    "name": info.get("name"),
                    "host_api": info.get("hostapi"),
                    "host_api_name": host_api_name,
                    "max_input_channels": info.get("max_input_channels"),
                    "max_output_channels": info.get("max_output_channels"),
                    "default_sample_rate": info.get("default_samplerate"),
                    "default_low_input_latency": info.get("default_low_input_latency"),
                    "default_high_input_latency": info.get("default_high_input_latency"),
                    "default_low_output_latency": info.get("default_low_output_latency"),
                    "default_high_output_latency": info.get("default_high_output_latency"),
                }
            except Exception as exc:
                return {"error": str(exc), "configured": device}

        input_device, output_device = self._devices()
        input_device = self._device_argument(input_device)
        output_device = self._device_argument(output_device)
        try:
            portaudio_version = sd.get_portaudio_version()[1]
        except Exception:
            portaudio_version = None
        return {
            "backend": "sounddevice",
            "requested_host_api": self.config.host_api,
            "requested_sample_rate": self.config.sample_rate,
            "sounddevice_version": getattr(sd, "__version__", None),
            "portaudio_version": portaudio_version,
            "windows_asio_requested": bool(os.environ.get("SD_ENABLE_ASIO")),
            "default_devices": list(sd.default.device),
            "input_device": describe(input_device, "input"),
            "output_device": describe(output_device, "output"),
        }

    def _operation_status(
        self, sd, callback_status: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        status = self._device_status(sd)
        status["callback_status"] = callback_status or self._callback_status(sd)
        status["xrun"] = status["callback_status"]["xrun"]
        status["warnings"] = list(self._last_check_warnings)
        return status

    @staticmethod
    def _new_callback_status() -> dict[str, Any]:
        return {
            "text": "",
            "xrun": False,
            "input_underflow": False,
            "input_overflow": False,
            "output_underflow": False,
            "output_overflow": False,
            "callback_count": 0,
            "first_callback_time": None,
            "last_callback_time": None,
        }

    @staticmethod
    def _record_callback_time(target: dict[str, Any], time_info: Any) -> None:
        """Keep PortAudio ADC/DAC timestamps for later clock/latency diagnosis."""
        timing = {}
        for name in ("inputBufferAdcTime", "currentTime", "outputBufferDacTime"):
            try:
                timing[name] = float(getattr(time_info, name))
            except Exception:
                continue
        target["callback_count"] = int(target.get("callback_count", 0)) + 1
        if target.get("first_callback_time") is None:
            target["first_callback_time"] = timing
        target["last_callback_time"] = timing

    @staticmethod
    def _merge_callback_status(
        target: dict[str, Any], status: Any
    ) -> None:
        if not status:
            return
        names = ("input_underflow", "input_overflow", "output_underflow", "output_overflow")
        for name in names:
            target[name] = bool(target[name] or getattr(status, name, False))
        target["xrun"] = any(bool(target[name]) for name in names)
        message = str(status).strip()
        if message and message not in str(target["text"]):
            target["text"] = "; ".join(part for part in (target["text"], message) if part)

    def _set_active_stream(self, stream: Any | None) -> None:
        with self._stream_lock:
            self._active_stream = stream
            self._stream_owner_thread_id = (
                threading.get_ident() if stream is not None else None
            )

    def _run_stream(
        self, stream: Any, expected_frames: int, sd, *, active_phase: str = "playing"
    ) -> tuple[bool, bool]:
        """Start, observe and close a stream.

        Returns ``(cancelled, timed_out)``.  The limit includes ample room for
        high-latency MME devices, while still preventing an endless wait caused
        by an unavailable laptop audio endpoint.
        """
        self._set_active_stream(stream)
        expected_s = expected_frames / max(1, self.config.sample_rate)
        deadline = time.monotonic() + max(15.0, expected_s * 2.0 + 8.0)
        cancelled = False
        timed_out = False
        try:
            self._emit_progress("opening_audio_stream", 0, expected_frames, force=True)
            if self._abort_requested.is_set():
                cancelled = True
            else:
                stream.start()
                self._emit_progress(active_phase, 0, expected_frames, force=True)
            while bool(getattr(stream, "active", False)):
                if self._abort_requested.is_set():
                    cancelled = True
                    try:
                        stream.abort()
                    except Exception:
                        pass
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    try:
                        stream.abort()
                    except Exception:
                        pass
                    break
                time.sleep(0.02)
            if self._abort_requested.is_set() and not timed_out:
                # The PortAudio callback may have consumed the stop request
                # just before the owner loop observed the inactive stream.
                cancelled = True
        finally:
            try:
                if bool(getattr(stream, "active", False)):
                    stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self._set_active_stream(None)
            self._emit_progress(
                "cancelled" if cancelled else "timed_out" if timed_out else "completed",
                expected_frames if not cancelled and not timed_out else 0,
                expected_frames,
                force=True,
            )
        return cancelled, timed_out

    def _duplex_stream(self, output: np.ndarray, sd) -> tuple[np.ndarray, dict[str, Any], bool]:
        """Return a fixed-length recording from one cancellable full-duplex stream."""
        input_count = max(self.config.input_channels)
        frames = len(output)
        recording = np.zeros((frames, input_count), dtype=np.float32)
        callback_status = self._new_callback_status()
        cursor = 0

        def callback(indata, outdata, callback_frames, time_info, status):
            nonlocal cursor
            self._merge_callback_status(callback_status, status)
            self._record_callback_time(callback_status, time_info)
            available = max(0, frames - cursor)
            copied = min(callback_frames, available)
            outdata.fill(0)
            if copied:
                recording[cursor : cursor + copied] = indata[:copied]
                outdata[:copied] = output[cursor : cursor + copied]
                cursor += copied
                self._emit_progress("playing", cursor, frames)
            if cursor >= frames or self._abort_requested.is_set():
                raise sd.CallbackStop()

        input_device, output_device = self._devices()
        input_device = self._device_argument(input_device)
        output_device = self._device_argument(output_device)
        stream = sd.Stream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=(input_device, output_device),
            channels=(input_count, output.shape[1]),
            dtype=(self.config.dtype, self.config.dtype),
            latency=self.config.latency,
            callback=callback,
            never_drop_input=False,
            prime_output_buffers_using_stream_callback=False,
        )
        try:
            latency = stream.latency
            callback_status["stream_latency_seconds"] = [float(value) for value in latency]
        except Exception:
            callback_status["stream_latency_seconds"] = None
        cancelled, timed_out = self._run_stream(stream, frames, sd)
        if timed_out:
            raise TimeoutError(
                "同步播录超时：音频驱动没有在预期时间内完成。"
                "请改选 MME 或 Windows WASAPI 的实际麦克风/扬声器，"
                "并先点击“检查声卡”。"
            )
        callback_status["cancelled"] = cancelled
        return recording, callback_status, cancelled

    def _input_stream(self, frames: int, sd) -> tuple[np.ndarray, dict[str, Any], bool]:
        input_count = max(self.config.input_channels)
        recording = np.zeros((frames, input_count), dtype=np.float32)
        callback_status = self._new_callback_status()
        cursor = 0

        def callback(indata, callback_frames, time_info, status):
            nonlocal cursor
            self._merge_callback_status(callback_status, status)
            self._record_callback_time(callback_status, time_info)
            copied = min(callback_frames, max(0, frames - cursor))
            if copied:
                recording[cursor : cursor + copied] = indata[:copied]
                cursor += copied
                self._emit_progress("recording", cursor, frames)
            if cursor >= frames or self._abort_requested.is_set():
                raise sd.CallbackStop()

        input_device, _ = self._devices()
        input_device = self._device_argument(input_device)
        stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=input_device,
            channels=input_count,
            dtype=self.config.dtype,
            latency=self.config.latency,
            callback=callback,
        )
        try:
            callback_status["stream_latency_seconds"] = float(stream.latency)
        except Exception:
            callback_status["stream_latency_seconds"] = None
        cancelled, timed_out = self._run_stream(
            stream, frames, sd, active_phase="recording"
        )
        if timed_out:
            raise TimeoutError(
                "录制超时：麦克风驱动没有返回音频。请检查 Windows 麦克风权限和所选设备。"
            )
        callback_status["cancelled"] = cancelled
        return recording, callback_status, cancelled

    def _output_stream(self, output: np.ndarray, sd) -> tuple[dict[str, Any], bool]:
        frames = len(output)
        callback_status = self._new_callback_status()
        cursor = 0

        def callback(outdata, callback_frames, time_info, status):
            nonlocal cursor
            self._merge_callback_status(callback_status, status)
            self._record_callback_time(callback_status, time_info)
            copied = min(callback_frames, max(0, frames - cursor))
            outdata.fill(0)
            if copied:
                outdata[:copied] = output[cursor : cursor + copied]
                cursor += copied
                self._emit_progress("playing", cursor, frames)
            if cursor >= frames or self._abort_requested.is_set():
                raise sd.CallbackStop()

        _, output_device = self._devices()
        output_device = self._device_argument(output_device)
        stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=output_device,
            channels=output.shape[1],
            dtype=self.config.dtype,
            latency=self.config.latency,
            callback=callback,
            prime_output_buffers_using_stream_callback=False,
        )
        try:
            callback_status["stream_latency_seconds"] = float(stream.latency)
        except Exception:
            callback_status["stream_latency_seconds"] = None
        cancelled, timed_out = self._run_stream(stream, frames, sd)
        if timed_out:
            raise TimeoutError(
                "播放超时：扬声器驱动没有在预期时间内完成。请检查所选播放设备。"
            )
        callback_status["cancelled"] = cancelled
        return callback_status, cancelled

    def check_settings(
        self,
        *,
        input_required: bool,
        output_channels: int | None,
    ) -> dict[str, Any]:
        """Ask PortAudio to validate devices, channels, format and sample rate."""
        sd = self._module()
        input_device, output_device = self._devices()
        input_device = self._device_argument(input_device)
        output_device = self._device_argument(output_device)
        if input_required:
            sd.check_input_settings(
                device=input_device,
                channels=max(self.config.input_channels),
                dtype=self.config.dtype,
                samplerate=self.config.sample_rate,
            )
        if output_channels is not None:
            sd.check_output_settings(
                device=output_device,
                channels=output_channels,
                dtype=self.config.dtype,
                samplerate=self.config.sample_rate,
            )
        status = self._device_status(sd)
        input_info = status.get("input_device", {})
        output_info = status.get("output_device", {})
        input_host = input_info.get("host_api")
        output_host = output_info.get("host_api")
        requested_host = str(self.config.host_api or "").strip()
        if requested_host:
            for required, info, label in (
                (input_required, input_info, "录制设备"),
                (output_channels is not None, output_info, "播放设备"),
            ):
                actual = str(info.get("host_api_name") or "")
                if required and actual.casefold() != requested_host.casefold():
                    raise RuntimeError(
                        f"{label}不属于所选音频协议 {requested_host}；"
                        f"实际协议为 {actual or '未知'}，请按协议重新选择设备"
                    )
        if input_required and output_channels is not None and input_host != output_host:
            raise RuntimeError(
                "同步播录要求录制设备和播放设备属于同一主机接口；"
                f"当前分别为 {input_info.get('host_api_name')} 和 {output_info.get('host_api_name')}"
            )
        if (
            input_required
            and output_channels is not None
            and input_info.get("host_api_name") == "ASIO"
            and input_info.get("index") != output_info.get("index")
        ):
            raise RuntimeError("ASIO 同步播录应为输入和输出选择同一个 RME 双工设备")
        warnings = []
        for required, info, label in (
            (input_required, input_info, "录制设备"),
            (output_channels is not None, output_info, "播放设备"),
        ):
            default_rate = info.get("default_sample_rate")
            if required and default_rate is not None:
                try:
                    differs = abs(float(default_rate) - self.config.sample_rate) > 0.5
                except (TypeError, ValueError):
                    differs = False
                if differs:
                    warnings.append(
                        f"{label}默认采样率为 {float(default_rate):g} Hz，实验请求 "
                        f"{self.config.sample_rate} Hz；Windows/驱动可能进行重采样。"
                        "正式 RIR 采集请在声卡和 Windows 中统一采样率"
                    )
        if sys.platform == "win32" and input_required and output_channels is not None:
            host_name = str(input_info.get("host_api_name") or "")
            if host_name != "ASIO":
                warnings.append(
                    f"当前同步播录使用 {host_name or '未知接口'}，不是 ASIO；"
                    "正式 RME 数据采集建议选择设备列表中主机接口为 ASIO 的设备"
                )
                if input_info.get("index") != output_info.get("index"):
                    warnings.append(
                        "录制和播放选择了两个不同的非 ASIO 端点。相同音频协议不代表"
                        "共用硬件时钟；公共延迟可能变化，长扫频还可能产生采样时钟漂移。"
                        "RIR 的绝对到达时间需增加有线回环参考后才可信"
                    )
        status["warnings"] = warnings
        self._last_check_warnings = list(warnings)
        return status

    def play_record(self, output: np.ndarray) -> CaptureResult:
        # Clear before validation/opening.  A Stop click received while a slow
        # ASIO driver is opening then remains visible to the owner loop.
        self._abort_requested.clear()
        sd = self._module()
        self.check_settings(input_required=True, output_channels=output.shape[1])
        recording, callback_status, _ = self._duplex_stream(output, sd)
        selected = recording[:, np.asarray(self.config.input_channels) - 1]
        return CaptureResult(
            np.asarray(selected, dtype=np.float32),
            self._operation_status(sd, callback_status),
        )

    def record(self, frames: int) -> CaptureResult:
        self._abort_requested.clear()
        sd = self._module()
        self.check_settings(input_required=True, output_channels=None)
        recording, callback_status, _ = self._input_stream(frames, sd)
        selected = recording[:, np.asarray(self.config.input_channels) - 1]
        return CaptureResult(
            np.asarray(selected, dtype=np.float32),
            self._operation_status(sd, callback_status),
        )

    def play(self, output: np.ndarray) -> dict[str, Any]:
        self._abort_requested.clear()
        sd = self._module()
        self.check_settings(input_required=False, output_channels=output.shape[1])
        callback_status, _ = self._output_stream(output, sd)
        return self._operation_status(sd, callback_status)

    def stop(self) -> None:
        # Do not invoke stream.abort(), stream.close() or sounddevice.stop()
        # from this caller (normally Tk's main thread).  The owner loop in
        # _run_stream observes this event within 20 ms and performs all ASIO
        # control calls on the same thread that opened and started the stream.
        self._abort_requested.set()


class SimulatedBackend(AudioBackend):
    """A small multi-microphone room simulator for CI and workflow rehearsal."""

    def __init__(self, config: AudioConfig, seed: int = 7):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        self.paths = {
            config.target_output_channel: self._paths(1.0),
            config.interferer_output_channel: self._paths(0.72),
        }
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def _emit_progress(self, phase: str, frames: int, total_frames: int) -> None:
        if self._progress_callback is not None:
            self._progress_callback(
                {"phase": phase, "frames": frames, "total_frames": total_frames, "sample_rate": self.config.sample_rate}
            )

    def _paths(self, base_gain: float) -> list[np.ndarray]:
        delay = 170
        paths = []
        for channel in range(len(self.config.input_channels)):
            gain = base_gain * (0.86 + 0.14 * np.cos(channel * 0.8))
            offset = channel * 7
            impulse = np.zeros(1500, dtype=np.float32)
            impulse[delay + offset] = gain
            impulse[delay + 121 + offset] = gain * 0.32
            impulse[delay + 367 + offset] = gain * -0.17
            paths.append(impulse)
        return paths

    def play_record(self, output: np.ndarray) -> CaptureResult:
        self._emit_progress("playing", 0, len(output))
        microphones = np.zeros((len(output), len(self.config.input_channels)), dtype=np.float32)
        for one_based_channel, paths in self.paths.items():
            if one_based_channel <= output.shape[1]:
                emitted = output[:, one_based_channel - 1]
                for mic, impulse in enumerate(paths):
                    microphones[:, mic] += fftconvolve(emitted, impulse, mode="full")[: len(output)]
        microphones += self.rng.normal(0, 2e-5, microphones.shape).astype(np.float32)
        self._emit_progress("completed", len(output), len(output))
        return CaptureResult(microphones, {"backend": "simulated", "overflow": False})

    def record(self, frames: int) -> CaptureResult:
        self._emit_progress("recording", 0, frames)
        microphones = self.rng.normal(
            0, 2e-5, (frames, len(self.config.input_channels))
        ).astype(np.float32)
        self._emit_progress("completed", frames, frames)
        return CaptureResult(microphones, {"backend": "simulated", "overflow": False})

    def play(self, output: np.ndarray) -> dict[str, Any]:
        return {"backend": "simulated", "frames": len(output)}

    def stop(self) -> None:
        return None


def create_backend(config: AudioConfig) -> AudioBackend:
    if config.backend == "sounddevice":
        return SoundDeviceBackend(config)
    if config.backend == "simulated":
        return SimulatedBackend(config)
    raise ValueError(f"unknown audio backend: {config.backend}")


def check_hardware_settings(config: AudioConfig) -> dict[str, Any]:
    """Validate a real sound card without playing or recording audio."""
    if config.backend != "sounddevice":
        return {"backend": config.backend, "message": "当前使用模拟声卡，不需要硬件检查"}
    return SoundDeviceBackend(config).check_settings(
        input_required=True,
        output_channels=max(config.target_output_channel, config.interferer_output_channel),
    )


def format_hardware_status(status: dict[str, Any]) -> str:
    """Format hardware diagnostics for the Chinese GUI and command line."""
    if status.get("message"):
        return str(status["message"])
    lines = [f"音频后端：{status.get('backend', '未知')}"]
    if status.get("requested_host_api"):
        lines.append(f"配置选择的音频协议：{status['requested_host_api']}")
    if status.get("requested_sample_rate"):
        lines.append(f"实验采样率：{status['requested_sample_rate']} Hz")
    if status.get("sounddevice_version"):
        lines.append(f"sounddevice 版本：{status['sounddevice_version']}")
    if status.get("portaudio_version"):
        lines.append(f"PortAudio：{status['portaudio_version']}")
    if "default_devices" in status:
        lines.append(f"系统默认输入/输出设备编号：{status['default_devices']}")
    for key, label in (("input_device", "录制设备"), ("output_device", "播放设备")):
        device = status.get(key, {})
        if "error" in device:
            lines.append(f"{label}：查询失败：{device['error']}")
            continue
        lines.extend(
            [
                f"{label}：{device.get('name', '未知')}（编号 {device.get('index', '未知')}）",
                f"  主机接口：{device.get('host_api_name') or device.get('host_api', '未知')}",
                f"  最大输入通道：{device.get('max_input_channels', '未知')}",
                f"  最大输出通道：{device.get('max_output_channels', '未知')}",
                f"  默认采样率：{device.get('default_sample_rate', '未知')}",
                f"  默认低/高延迟（输入）：{device.get('default_low_input_latency', '未知')} / "
                f"{device.get('default_high_input_latency', '未知')} 秒",
                f"  默认低/高延迟（输出）：{device.get('default_low_output_latency', '未知')} / "
                f"{device.get('default_high_output_latency', '未知')} 秒",
            ]
        )
    for warning in status.get("warnings", []):
        lines.append(f"警告：{warning}")
    return "\n".join(lines)


def list_devices() -> str:
    try:
        _enable_windows_asio()
        import sounddevice as sd

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        lines = ["编号 | 主机接口 | 输入通道 | 输出通道 | 设备名称"]
        for index, item in enumerate(devices):
            host_name = host_apis[item["hostapi"]]["name"]
            lines.append(
                f"{index:>4} | {host_name} | {item['max_input_channels']:>8} | "
                f"{item['max_output_channels']:>8} | {item['name']}"
            )
        lines.append(f"\n系统默认输入/输出设备编号：{list(sd.default.device)}")
        return "\n".join(lines)
    except Exception as exc:
        return f"查询音频设备失败：{exc}"


def host_api_choices() -> list[str]:
    """Return host APIs that currently own at least one usable audio device."""
    try:
        _enable_windows_asio()
        import sounddevice as sd

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        used = {
            int(item["hostapi"])
            for item in devices
            if item["max_input_channels"] > 0 or item["max_output_channels"] > 0
        }
        return [
            str(item["name"])
            for index, item in enumerate(host_apis)
            if index in used
        ]
    except Exception:
        return []


def device_choices(
    kind: str | None = None,
    host_api: str | None = None,
) -> list[str]:
    """Return GUI device choices, optionally filtered for input or output.

    Full-duplex devices are present in both filtered lists.  The numeric prefix
    is accepted by sounddevice and avoids ambiguous device names.
    """
    if kind not in {None, "input", "output"}:
        raise ValueError("device choice kind must be input, output or None")
    try:
        _enable_windows_asio()
        import sounddevice as sd
        host_apis = sd.query_hostapis()
        choices = []
        requested_host = str(host_api or "").strip().casefold()
        for index, item in enumerate(sd.query_devices()):
            api_name = str(host_apis[item["hostapi"]]["name"])
            if requested_host and api_name.casefold() != requested_host:
                continue
            if kind == "input" and item["max_input_channels"] <= 0:
                continue
            if kind == "output" and item["max_output_channels"] <= 0:
                continue
            if kind is None and item["max_input_channels"] <= 0 and item["max_output_channels"] <= 0:
                continue
            api_suffix = "" if requested_host else f" [{api_name}]"
            if kind == "input":
                capacity = f"（最多 {item['max_input_channels']} 个输入通道）"
            elif kind == "output":
                capacity = f"（最多 {item['max_output_channels']} 个输出通道）"
            else:
                capacity = (
                    f"（输入 {item['max_input_channels']}，输出 {item['max_output_channels']}）"
                )
            choices.append(f"{index}: {item['name']}{api_suffix} {capacity}")
        return choices
    except Exception:
        return []


def _enable_windows_asio() -> None:
    """Ask pip's sounddevice package to load its ASIO-enabled DLL on Windows.

    Set ``ACOUSTIC_CAPTURE_ENABLE_ASIO=0`` before launching to opt out when
    diagnosing a machine-specific PortAudio problem.
    """
    if (
        sys.platform == "win32"
        and os.environ.get("ACOUSTIC_CAPTURE_ENABLE_ASIO", "1") != "0"
    ):
        os.environ.setdefault("SD_ENABLE_ASIO", "1")
