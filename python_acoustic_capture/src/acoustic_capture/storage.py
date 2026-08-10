"""Run directory creation and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml

from . import __version__
from .config import ExperimentConfig


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
        manifest = {
            "schema_version": 1,
            "software_version": __version__,
            "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": _git_commit(),
            "status": "running",
            "metadata": config.metadata,
            "artifacts": [],
        }
        store = cls(root, config, manifest)
        with (root / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
        store.write_json("manifest.json", manifest)
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
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
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
        self.manifest["artifacts"].append(entry)
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
        with self.path("manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
