"""Run directory creation and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from . import __version__
from .config import ExperimentConfig
from .professional import array_geometry_sha256, build_preflight_report, canonical_sha256


def _replace_with_windows_retry(temporary: Path, destination: Path) -> None:
    """Replace atomically even while an antivirus/viewer briefly opens the file."""
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01 * (attempt + 1))


def _safe_name(value: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", value.strip())
    text = "_".join(text.split()).strip(" .")
    if not text:
        return "measurement"
    # Leave room for timestamp/kind and nested artifact names on Windows.
    text = text[:120].rstrip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    return f"_{text}" if text.upper() in reserved else text


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RunStore:
    root: Path
    config: ExperimentConfig
    manifest: dict[str, Any]
    _artifact_positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._artifact_positions = {
            str(entry.get("path")): index
            for index, entry in enumerate(self.manifest.get("artifacts", []))
            if entry.get("path")
        }

    @classmethod
    def create(cls, config: ExperimentConfig, kind: str) -> "RunStore":
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        root = Path(config.storage.root) / f"{stamp}_{_safe_name(config.storage.session_name)}_{kind}"
        suffix = 1
        candidate = root
        while candidate.exists():
            candidate = Path(f"{root}_{suffix:02d}")
            suffix += 1
        root = candidate
        for folder in ("raw", "processed", "references", "metrics", "logs"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        preflight = build_preflight_report(config, kind)
        manifest = {
            "schema_version": 2,
            "run_uuid": str(uuid.uuid4()),
            "software_version": __version__,
            "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
            "config_sha256": canonical_sha256(config.to_dict()),
            "array_geometry_sha256": array_geometry_sha256(config.metadata),
            "audio_backend": config.audio.backend,
            "audio_host_api": config.audio.host_api,
            "audio_input_device": config.audio.input_device,
            "audio_output_device": config.audio.output_device,
            "audio_input_channels": config.audio.input_channels,
            "status": "running",
            "metadata": config.metadata,
            "preflight": preflight.to_dict(),
            "artifacts": [],
        }
        store = cls(root, config, manifest)
        with (root / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
        store.write_json("manifest.json", manifest)
        store.add_artifact("config_resolved.yaml")
        store.write_json("metrics/preflight_report.json", preflight.to_dict())
        store.checkpoint()
        return store

    @classmethod
    def resume(cls, root: str | Path, config: ExperimentConfig, kind: str) -> "RunStore":
        """Re-open an interrupted run without discarding completed artifacts."""
        run_root = Path(root).resolve()
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"续采目录缺少 manifest.json：{run_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != kind:
            raise ValueError(f"续采目录类型不是 {kind}：{run_root}")
        if manifest.get("status") == "completed":
            raise ValueError("该实验已经完成，不能续采；请开始一个新实验")
        store = cls(run_root, config, manifest)
        manifest["status"] = "running"
        manifest.pop("finished_at", None)
        manifest.setdefault("resume_history", []).append(
            datetime.now(timezone.utc).isoformat()
        )
        store.checkpoint()
        return store

    def path(self, relative: str) -> Path:
        return self.root / relative

    def write_audio(self, relative: str, data, sample_rate: int) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, sample_rate, subtype=self.config.storage.wav_subtype)
        self.add_artifact(relative)
        return path

    def write_json(self, relative: str, data: Any) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        _replace_with_windows_retry(temporary, path)
        if relative != "manifest.json":
            self.add_artifact(relative)
        return path

    def add_artifact(self, relative: str) -> None:
        entry: dict[str, Any] = {"path": relative}
        path = self.path(relative)
        if path.exists():
            entry["bytes"] = path.stat().st_size
            if self.config.storage.compute_sha256:
                entry["sha256"] = sha256(path)
        artifacts = self.manifest["artifacts"]
        existing_index = self._artifact_positions.get(relative)
        if existing_index is not None:
            artifacts[existing_index] = entry
        else:
            self._artifact_positions[relative] = len(artifacts)
            artifacts.append(entry)

    def checkpoint(self) -> None:
        """Durably save the manifest at a logical capture boundary."""
        self._flush()

    def finish(self, summary: dict[str, Any], status: str = "completed") -> None:
        report = [
            f"# Acoustic capture report: {self.config.storage.session_name}",
            "",
            f"- Status: `{status}`",
            f"- Type: `{self.manifest['kind']}`",
            f"- Started: `{self.manifest['created_at']}`",
            f"- Sample rate: `{self.config.audio.sample_rate}` Hz",
            f"- Microphone inputs: `{self.config.audio.input_channels}`",
            "",
            "## Experiment metadata",
            "",
            "```json",
            json.dumps(self.config.metadata, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        report_path = self.path("report.md")
        report_path.write_text("\n".join(report), encoding="utf-8")
        self.add_artifact("report.md")
        self.manifest["status"] = status
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["summary"] = summary
        self._flush()

    def _flush(self) -> None:
        path = self.path("manifest.json")
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)
        _replace_with_windows_retry(temporary, path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
