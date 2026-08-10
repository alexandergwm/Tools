"""Audio I/O backends.

SoundDeviceBackend addresses hardware channels using one-based values in the
configuration. SimulatedBackend is deterministic enough for tests and demos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import sys
from typing import Any

import numpy as np
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


class SoundDeviceBackend(AudioBackend):
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
                }
            except Exception as exc:
                return {"error": str(exc), "configured": device}

        input_device, output_device = self._devices()
        try:
            portaudio_version = sd.get_portaudio_version()[1]
        except Exception:
            portaudio_version = None
        return {
            "backend": "sounddevice",
            "sounddevice_version": getattr(sd, "__version__", None),
            "portaudio_version": portaudio_version,
            "windows_asio_requested": bool(os.environ.get("SD_ENABLE_ASIO")),
            "default_devices": list(sd.default.device),
            "input_device": describe(input_device, "input"),
            "output_device": describe(output_device, "output"),
        }

    def _operation_status(self, sd) -> dict[str, Any]:
        status = self._device_status(sd)
        status["callback_status"] = self._callback_status(sd)
        status["xrun"] = status["callback_status"]["xrun"]
        return status

    def check_settings(
        self,
        *,
        input_required: bool,
        output_channels: int | None,
    ) -> dict[str, Any]:
        """Ask PortAudio to validate devices, channels, format and sample rate."""
        sd = self._module()
        input_device, output_device = self._devices()
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
        if sys.platform == "win32" and input_required and output_channels is not None:
            host_name = str(input_info.get("host_api_name") or "")
            if host_name != "ASIO":
                warnings.append(
                    f"当前同步播录使用 {host_name or '未知接口'}，不是 ASIO；"
                    "正式 RME 数据采集建议选择设备列表中主机接口为 ASIO 的设备"
                )
        status["warnings"] = warnings
        return status

    def play_record(self, output: np.ndarray) -> CaptureResult:
        sd = self._module()
        self.check_settings(input_required=True, output_channels=output.shape[1])
        input_count = max(self.config.input_channels)
        recording = sd.playrec(
            output,
            samplerate=self.config.sample_rate,
            channels=input_count,
            dtype=self.config.dtype,
            device=(
                self.config.input_device if self.config.input_device is not None else self.config.device,
                self.config.output_device if self.config.output_device is not None else self.config.device,
            ),
            blocksize=self.config.block_size,
            latency=self.config.latency,
            blocking=True,
        )
        selected = recording[:, np.asarray(self.config.input_channels) - 1]
        return CaptureResult(np.asarray(selected, dtype=np.float32), self._operation_status(sd))

    def record(self, frames: int) -> CaptureResult:
        sd = self._module()
        self.check_settings(input_required=True, output_channels=None)
        input_count = max(self.config.input_channels)
        recording = sd.rec(
            frames,
            samplerate=self.config.sample_rate,
            channels=input_count,
            dtype=self.config.dtype,
            device=self.config.input_device if self.config.input_device is not None else self.config.device,
            blocksize=self.config.block_size,
            latency=self.config.latency,
            blocking=True,
        )
        selected = recording[:, np.asarray(self.config.input_channels) - 1]
        return CaptureResult(np.asarray(selected, dtype=np.float32), self._operation_status(sd))

    def play(self, output: np.ndarray) -> dict[str, Any]:
        sd = self._module()
        self.check_settings(input_required=False, output_channels=output.shape[1])
        sd.play(
            output,
            samplerate=self.config.sample_rate,
            device=self.config.output_device if self.config.output_device is not None else self.config.device,
            blocksize=self.config.block_size,
            latency=self.config.latency,
            blocking=True,
        )
        return self._operation_status(sd)


class SimulatedBackend(AudioBackend):
    """A small multi-microphone room simulator for CI and workflow rehearsal."""

    def __init__(self, config: AudioConfig, seed: int = 7):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
        self.paths = {
            config.target_output_channel: self._paths(1.0),
            config.interferer_output_channel: self._paths(0.72),
        }

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
        microphones = np.zeros((len(output), len(self.config.input_channels)), dtype=np.float32)
        for one_based_channel, paths in self.paths.items():
            if one_based_channel <= output.shape[1]:
                emitted = output[:, one_based_channel - 1]
                for mic, impulse in enumerate(paths):
                    microphones[:, mic] += fftconvolve(emitted, impulse, mode="full")[: len(output)]
        microphones += self.rng.normal(0, 2e-5, microphones.shape).astype(np.float32)
        return CaptureResult(microphones, {"backend": "simulated", "overflow": False})

    def record(self, frames: int) -> CaptureResult:
        microphones = self.rng.normal(
            0, 2e-5, (frames, len(self.config.input_channels))
        ).astype(np.float32)
        return CaptureResult(microphones, {"backend": "simulated", "overflow": False})

    def play(self, output: np.ndarray) -> dict[str, Any]:
        return {"backend": "simulated", "frames": len(output)}


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


def device_choices(kind: str | None = None) -> list[str]:
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
        return [
            f"{index}: {item['name']} [{host_apis[item['hostapi']]['name']}] "
            f"（输入 {item['max_input_channels']}，输出 {item['max_output_channels']}）"
            for index, item in enumerate(sd.query_devices())
            if (
                kind is None
                or (kind == "input" and item["max_input_channels"] > 0)
                or (kind == "output" and item["max_output_channels"] > 0)
            )
        ]
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
