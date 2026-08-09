"""Audio I/O backends.

SoundDeviceBackend addresses hardware channels using one-based values in the
configuration. SimulatedBackend is deterministic enough for tests and demos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - depends on local driver
            raise RuntimeError("无法加载 sounddevice/PortAudio，请检查安装和声卡驱动") from exc
        return sd

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
        return {
            "backend": "sounddevice",
            "default_devices": list(sd.default.device),
            "input_device": describe(input_device, "input"),
            "output_device": describe(output_device, "output"),
        }

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
        return self._device_status(sd)

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
        return CaptureResult(np.asarray(selected, dtype=np.float32), self._device_status(sd))

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
        return CaptureResult(np.asarray(selected, dtype=np.float32), self._device_status(sd))

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
        return self._device_status(sd)


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
    return "\n".join(lines)


def list_devices() -> str:
    try:
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


def device_choices() -> list[str]:
    """Compact GUI choices; the numeric prefix is accepted by sounddevice."""
    try:
        import sounddevice as sd
        host_apis = sd.query_hostapis()
        return [
            f"{index}: {item['name']} [{host_apis[item['hostapi']]['name']}] "
            f"（输入 {item['max_input_channels']}，输出 {item['max_output_channels']}）"
            for index, item in enumerate(sd.query_devices())
        ]
    except Exception:
        return []
