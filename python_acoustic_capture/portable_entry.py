"""Windows frozen-application entry point for Acoustic Capture."""

from __future__ import annotations

import os
import json
import shutil
import sys
import traceback
from pathlib import Path

PORTABLE_CONFIG_VERSION = "0.2.11"


def _portable_config_path(root: Path) -> Path:
    return root / "configs" / f"portable_default_{PORTABLE_CONFIG_VERSION}.yaml"


def _bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _preferred_portable_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _writable_portable_root() -> Path:
    preferred = _preferred_portable_root()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".acoustic_capture_write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AcousticCapturePortable"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _install_assets(root: Path) -> None:
    source_root = _bundle_root() / "portable_assets"
    if not source_root.is_dir():
        raise RuntimeError(f"Portable assets are missing: {source_root}")
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = (
            _portable_config_path(root)
            if relative.as_posix() == "configs/portable_default.yaml"
            else root / relative
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        if relative.as_posix() == "configs/portable_default.yaml":
            legacy = root / relative
            if not legacy.exists():
                shutil.copy2(source, legacy)
    for folder in ("runs", "datasets", "logs"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def _show_fatal_error(root: Path, exc: BaseException) -> None:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path = root / "logs" / "portable_startup_error.txt"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(details, encoding="utf-8")
    except OSError:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        window = tk.Tk()
        window.withdraw()
        messagebox.showerror(
            "Acoustic Capture failed to start",
            f"{exc}\n\nDetails were written to:\n{log_path}",
            parent=window,
        )
        window.destroy()
    except Exception:
        pass


def _portable_smoke_test(root: Path) -> None:
    """Exercise frozen DLLs and lazy imports without opening the full GUI."""
    from acoustic_capture.audio import _enable_windows_asio
    from acoustic_capture.checklist import read_checklist
    from acoustic_capture.config import load_config
    from acoustic_capture.labels import import_reviewed_labels, write_label_files
    from acoustic_capture.professional import build_preflight_report
    from acoustic_capture.rir import estimate_impulse_response
    from acoustic_capture.signals import exponential_sweep

    _enable_windows_asio()
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import tkinter as tk

    config = load_config(_portable_config_path(root))
    checklist_path = root / "acoustic_capture_checklist_template.xlsx"
    checklist_rows = read_checklist(checklist_path)
    preflight = build_preflight_report(config, "scene")
    sweep = exponential_sweep(
        config.audio.sample_rate,
        config.sweep.start_hz,
        config.sweep.end_hz,
        0.05,
        config.sweep.level_dbfs,
        0.01,
        0.005,
    )
    synthetic_response = np.pad(np.convolve(sweep, [1.0, 0.35]), (0, 128))
    estimated_ir = estimate_impulse_response(sweep, synthetic_response, 64)
    smoke_root = root / "logs" / "portable_smoke_artifacts"
    smoke_root.mkdir(parents=True, exist_ok=True)
    sf.write(smoke_root / "sweep.wav", sweep, config.audio.sample_rate)
    label_files = write_label_files(smoke_root, [], {"smoke_test": True})
    reviewed_label_files = import_reviewed_labels(smoke_root)
    tcl = tk.Tcl()
    tcl_version = tcl.eval("info patchlevel")
    result = {
        "status": "passed",
        "python_frozen": bool(getattr(sys, "frozen", False)),
        "portable_root": str(root),
        "config": str(_portable_config_path(root)),
        "checklist": str(checklist_path),
        "checklist_rows": len(checklist_rows),
        "preflight_status": preflight.to_dict()["status"],
        "sample_rate_hz": config.audio.sample_rate,
        "sweep_samples": len(sweep),
        "rir_smoke_peak_sample": int(abs(estimated_ir[:, 0]).argmax()),
        "sounddevice_version": getattr(sd, "__version__", ""),
        "portaudio_version": sd.get_portaudio_version()[1],
        "host_apis": [str(item["name"]) for item in sd.query_hostapis()],
        "soundfile_formats": sorted(sf.available_formats()),
        "tcl_version": tcl_version,
        "label_files": {key: str(path) for key, path in label_files.items()},
        "reviewed_label_files": {
            key: str(path) for key, path in reviewed_label_files.items()
        },
    }
    (root / "logs" / "portable_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    root = _writable_portable_root()
    try:
        _install_assets(root)
        os.chdir(root)
        config = _portable_config_path(root)
        if os.environ.get("ACOUSTIC_CAPTURE_PORTABLE_SMOKE") == "1":
            _portable_smoke_test(root)
            return 0
        # Select sounddevice's ASIO-enabled PortAudio DLL before any GUI or
        # plotting imports can indirectly import sounddevice.
        from acoustic_capture.audio import _enable_windows_asio

        _enable_windows_asio()
        from acoustic_capture.gui import main as gui_main

        gui_main(config)
        return 0
    except BaseException as exc:
        _show_fatal_error(root, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
