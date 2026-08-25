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
    host_api: str | None = None  # ASIO | MME | Windows WASAPI | ...
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
    fade_in_s: float = 0.08
    fade_out_s: float = 0.005
    level_dbfs: float = -18.0
    rir_duration_s: float = 2.0
    pre_peak_s: float = 0.01


@dataclass
class RepeatConfig:
    strategy: str = "reconstruct_average"  # reconstruct_average | fixed_count
    fixed_count: int = 5
    correlation_threshold: float = 0.98
    peak_drift_samples: int = 2
    minimum_sweep_snr_db: float = 6.0
    reject_clipped: bool = True
    clip_threshold: float = 0.999
    pause_s: float = 0.5
    delay_low_hz: float = 100.0
    delay_high_hz: float = 1_000.0
    delay_max_ms: float = 3.0
    delay_agreement_samples: float = 2.0


@dataclass
class AcquaConfig:
    """Simple external ACQUA program generation and record-only capture."""

    program_file: str = ""
    segment_duration_s: float = 4.0
    gap_s: float = 0.5
    pairing_seed: int = 0
    wav_subtype: str = "PCM_24"  # PCM_16 | PCM_24 | FLOAT
    recording_margin_s: float = 0.0


@dataclass
class SceneConfig:
    items: list[str] = field(
        default_factory=lambda: [
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
    target_index_csv: str = ""
    interferer_index_csv: str = ""
    resume_run: str = ""
    pairing_seed: int = 0
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
    acqua: AcquaConfig = field(default_factory=AcquaConfig)
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
        if min(s.fade_in_s, s.fade_out_s) < 0:
            raise ValueError("sweep fade durations must be non-negative")
        if max(s.fade_in_s, s.fade_out_s) > s.duration_s:
            raise ValueError("each sweep fade duration cannot exceed the sweep duration")
        if s.pre_peak_s < 0 or s.pre_peak_s >= s.rir_duration_s:
            raise ValueError("sweep.pre_peak_s must be within the saved RIR duration")
        if s.rir_duration_s - s.pre_peak_s > s.post_silence_s:
            raise ValueError(
                "the RIR tail after pre_peak_s cannot exceed sweep.post_silence_s; "
                "otherwise the saved RIR tail was never recorded"
            )
        if r.strategy not in {"reconstruct_average", "fixed_count"}:
            raise ValueError(
                "repeats.strategy must be reconstruct_average or fixed_count"
            )
        if not 1 <= r.fixed_count <= 100:
            raise ValueError("repeats.fixed_count must be between 1 and 100")
        if not 0.0 <= r.correlation_threshold <= 1.0:
            raise ValueError("repeats.correlation_threshold must be between 0 and 1")
        if r.peak_drift_samples < 0:
            raise ValueError("repeats.peak_drift_samples must be non-negative")
        if r.minimum_sweep_snr_db < 0:
            raise ValueError("repeats.minimum_sweep_snr_db must be non-negative")
        if r.pause_s < 0:
            raise ValueError("repeats.pause_s must be non-negative")
        if not 0 < r.delay_low_hz < r.delay_high_hz < a.sample_rate / 2:
            raise ValueError("RIR delay frequency band must be within Nyquist")
        if r.delay_max_ms <= 0 or r.delay_agreement_samples < 0:
            raise ValueError("RIR delay limits must be positive")
        q = self.acqua
        if q.segment_duration_s <= 0 or q.gap_s < 0 or q.recording_margin_s < 0:
            raise ValueError("ACQUA segment/gap/margin durations are invalid")
        if isinstance(q.pairing_seed, bool) or not isinstance(q.pairing_seed, int):
            raise ValueError("acqua.pairing_seed must be an integer")
        if q.wav_subtype not in {"PCM_16", "PCM_24", "FLOAT"}:
            raise ValueError("acqua.wav_subtype must be PCM_16, PCM_24, or FLOAT")
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
        if self.scene.source_mode not in {"single", "folders"}:
            raise ValueError("scene.source_mode must be single or folders")
        if isinstance(self.scene.pairing_seed, bool) or not isinstance(
            self.scene.pairing_seed, int
        ):
            raise ValueError("scene.pairing_seed must be an integer")
        if self.scene.dataset_split not in {"train", "valid", "test"}:
            raise ValueError("scene.dataset_split must be train, valid, or test")
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
    sweep_values = dict(raw.get("sweep") or {})
    repeat_values = dict(raw.get("repeats") or {})
    scene_values = dict(raw.get("scene") or {})
    # Older releases exposed cycle/cartesian.  Folder pairing is now always
    # bounded and seed-deterministic; silently migrate those YAML files.
    scene_values.pop("pairing_mode", None)
    for legacy_field in (
        "minimum",
        "maximum",
        "required_stable_takes",
        "aggregate_change_threshold_db",
        "reconstruction_change_threshold_db",
    ):
        repeat_values.pop(legacy_field, None)
    legacy_fade = sweep_values.pop("fade_s", None)
    if legacy_fade is not None:
        # Older files used one symmetric fade.  Preserve their intent while
        # allowing new configs to match MATLAB sweeptone's asymmetric fades.
        sweep_values.setdefault("fade_in_s", legacy_fade)
        sweep_values.setdefault("fade_out_s", legacy_fade)
    cfg = ExperimentConfig(
        audio=_make(AudioConfig, raw.get("audio")),
        general=_make(GeneralIOConfig, raw.get("general")),
        sweep=_make(SweepConfig, sweep_values),
        repeats=_make(RepeatConfig, repeat_values),
        acqua=_make(AcquaConfig, raw.get("acqua")),
        scene=_make(SceneConfig, scene_values),
        storage=_make(StorageConfig, raw.get("storage")),
        metadata=raw.get("metadata", {}),
    )
    # Version 0.2.7 used an adaptive TrajectoRIR-inspired strategy.  Preserve
    # old configuration files while migrating them to the fixed-count,
    # reconvolution-QC-and-mean workflow.
    if cfg.repeats.strategy == "adaptive_select":
        cfg.repeats.strategy = "reconstruct_average"
    # Relative paths are interpreted relative to the YAML file, not the shell.
    for section, attr in (
        (cfg.general, "source_file"),
        (cfg.scene, "target_file"),
        (cfg.scene, "interferer_file"),
        (cfg.scene, "target_folder"),
        (cfg.scene, "interferer_folder"),
        (cfg.scene, "target_index_csv"),
        (cfg.scene, "interferer_index_csv"),
        (cfg.scene, "resume_run"),
        (cfg.acqua, "program_file"),
    ):
        raw_value = getattr(section, attr)
        if not raw_value:
            continue
        value = Path(raw_value)
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
