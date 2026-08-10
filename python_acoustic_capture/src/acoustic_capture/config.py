"""Configuration loading and validation.

The schema deliberately uses dataclasses instead of a large framework so that
the configuration rules remain easy to read and modify.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    backend: str = "sounddevice"  # sounddevice | simulated
    device: str | int | None = None  # legacy duplex fallback
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 48_000
    block_size: int = 1024
    input_channels: list[int] = field(default_factory=lambda: [1, 2])
    target_output_channel: int = 1
    interferer_output_channel: int = 2
    latency: str | float = "high"
    dtype: str = "float32"


@dataclass
class GeneralIOConfig:
    action: str = "play_record"  # play | record | play_record
    source_file: str = "audio/source.wav"
    duration_s: float = 5.0
    level_dbfs: float = -24.0
    output_channel: int = 1


@dataclass
class SweepConfig:
    start_hz: float = 40.0
    end_hz: float = 22_000.0
    duration_s: float = 8.0
    pre_silence_s: float = 1.0
    post_silence_s: float = 3.0
    fade_s: float = 0.02
    level_dbfs: float = -18.0
    rir_duration_s: float = 2.0
    pre_peak_s: float = 0.01


@dataclass
class RepeatConfig:
    minimum: int = 4
    maximum: int = 10
    correlation_threshold: float = 0.98
    peak_drift_samples: int = 2
    minimum_sweep_snr_db: float = 6.0
    required_stable_takes: int = 3
    reject_clipped: bool = True
    clip_threshold: float = 0.999
    pause_s: float = 0.5


@dataclass
class SceneConfig:
    items: list[str] = field(
        default_factory=lambda: [
            "ambient",
            "target_only",
            "interferer_only",
            "mixture",
        ]
    )
    source_mode: str = "single"  # single | folders
    duration_s: float | None = 4.0
    ambient_duration_s: float = 10.0
    target_file: str = "audio/target.wav"
    interferer_file: str = "audio/interferer.wav"
    target_folder: str = "audio/targets"
    interferer_folder: str = "audio/interferers"
    pairing_mode: str = "cycle"  # cycle | cartesian
    file_extensions: list[str] = field(default_factory=lambda: [".wav", ".flac"])
    label_prefix: str = ""
    dataset_split: str = "train"
    target_level_dbfs: float = -20.0
    interferer_level_dbfs: float = -20.0
    repetitions: int = 1
    capture_strategy: str = "paired_sequence"
    require_supervised_pair: bool = True
    countdown_s: float = 3.0
    gap_s: float = 1.0


@dataclass
class StorageConfig:
    root: str = "runs"
    session_name: str = "measurement"
    wav_subtype: str = "FLOAT"
    save_playback_reference: bool = True
    compute_sha256: bool = True


@dataclass
class ExperimentConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    general: GeneralIOConfig = field(default_factory=GeneralIOConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    repeats: RepeatConfig = field(default_factory=RepeatConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        a, s, r = self.audio, self.sweep, self.repeats
        if not a.input_channels:
            raise ValueError("audio.input_channels must contain at least one microphone channel")
        if len(set(a.input_channels)) != len(a.input_channels):
            raise ValueError("audio.input_channels cannot contain duplicate channels")
        channels = a.input_channels + [a.target_output_channel, a.interferer_output_channel]
        if any((not isinstance(ch, int)) or ch < 1 for ch in channels):
            raise ValueError("channel numbers are one-based positive integers")
        if a.sample_rate < 8_000 or (a.block_size != 0 and a.block_size < 16):
            raise ValueError("sample_rate or block_size is unrealistically small")
        if self.general.action not in {"play", "record", "play_record"}:
            raise ValueError("general.action must be play, record, or play_record")
        if self.general.duration_s <= 0:
            raise ValueError("general.duration_s must be positive")
        if self.general.output_channel < 1:
            raise ValueError("general.output_channel must be a positive one-based channel")
        if not 0 < s.start_hz < s.end_hz < a.sample_rate / 2:
            raise ValueError("sweep frequencies must satisfy 0 < start < end < Nyquist")
        if min(s.duration_s, s.pre_silence_s, s.post_silence_s, s.rir_duration_s) <= 0:
            raise ValueError("sweep durations must be positive")
        if s.fade_s < 0 or s.fade_s * 2 > s.duration_s:
            raise ValueError("sweep.fade_s must be non-negative and no longer than half the sweep")
        if s.pre_peak_s < 0 or s.pre_peak_s >= s.rir_duration_s:
            raise ValueError("sweep.pre_peak_s must be within the saved RIR duration")
        if not 1 <= r.minimum <= r.maximum:
            raise ValueError("repeats must satisfy 1 <= minimum <= maximum")
        if r.minimum_sweep_snr_db < 0:
            raise ValueError("repeats.minimum_sweep_snr_db must be non-negative")
        if not -100 <= s.level_dbfs <= 0:
            raise ValueError("sweep.level_dbfs must be between -100 and 0")
        allowed = {"ambient", "target_only", "interferer_only", "mixture"}
        if not self.scene.items:
            raise ValueError("scene.items must contain at least one capture item")
        unknown = set(self.scene.items) - allowed
        if unknown:
            raise ValueError(f"unknown scene items: {sorted(unknown)}")
        if self.scene.repetitions < 1:
            raise ValueError("scene.repetitions must be at least one")
        if self.scene.capture_strategy != "paired_sequence":
            raise ValueError("scene.capture_strategy must be paired_sequence")
        if (
            self.scene.require_supervised_pair
            and "mixture" in self.scene.items
            and "target_only" not in self.scene.items
        ):
            raise ValueError(
                "supervised mixture capture requires target_only in scene.items"
            )
        if (
            "mixture" in self.scene.items
            and a.target_output_channel == a.interferer_output_channel
        ):
            raise ValueError("mixture capture requires different target and interferer output channels")
        if self.scene.source_mode not in {"single", "folders"}:
            raise ValueError("scene.source_mode must be single or folders")
        if self.scene.pairing_mode not in {"cycle", "cartesian"}:
            raise ValueError("scene.pairing_mode must be cycle or cartesian")
        if not self.scene.file_extensions:
            raise ValueError("scene.file_extensions cannot be empty")
        if self.scene.duration_s is not None and self.scene.duration_s <= 0:
            raise ValueError("scene.duration_s must be positive or null")
        if self.scene.countdown_s < 0 or self.scene.gap_s < 0:
            raise ValueError("scene countdown and gap cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ExperimentConfig(
        audio=_make(AudioConfig, raw.get("audio")),
        general=_make(GeneralIOConfig, raw.get("general")),
        sweep=_make(SweepConfig, raw.get("sweep")),
        repeats=_make(RepeatConfig, raw.get("repeats")),
        scene=_make(SceneConfig, raw.get("scene")),
        storage=_make(StorageConfig, raw.get("storage")),
        metadata=raw.get("metadata", {}),
    )
    # Relative paths are interpreted relative to the YAML file, not the shell.
    for section, attr in (
        (cfg.general, "source_file"),
        (cfg.scene, "target_file"),
        (cfg.scene, "interferer_file"),
        (cfg.scene, "target_folder"),
        (cfg.scene, "interferer_folder"),
    ):
        value = Path(getattr(section, attr))
        if not value.is_absolute():
            setattr(section, attr, str((path.parent / value).resolve()))
    root = Path(cfg.storage.root)
    if not root.is_absolute():
        cfg.storage.root = str((path.parent / root).resolve())
    cfg.validate()
    return cfg


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
