from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path

from acoustic_capture.config import load_config, save_config
import acoustic_capture.gui as gui_module
from acoustic_capture.gui import CaptureGUI


def _wait_for_run(app: CaptureGUI, root: Path, kind: str, timeout_s: float = 15.0) -> Path:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.update()
        for manifest_path in root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # RunStore updates the manifest atomically from the GUI user's
                # perspective, but this tight polling loop can observe the file
                # between truncate and write.
                continue
            if manifest.get("kind") == kind and manifest.get("status") == "completed":
                # The worker marks the manifest completed before its __DONE__
                # message is consumed by Tk's 100 ms event poll.  Wait for the
                # viewer itself instead of relying on a fixed sleep, otherwise
                # a fast simulated run can make this test race the GUI callback.
                run_dir = manifest_path.parent
                while time.monotonic() < deadline:
                    app.update()
                    if app.viewer.run_dir == run_dir:
                        return run_dir
                    time.sleep(0.01)
                raise AssertionError(
                    f"GUI completed {kind} but viewer stayed at {app.viewer.run_dir}; "
                    f"busy={app._busy}"
                )
        time.sleep(0.01)
    raise AssertionError(f"GUI did not complete {kind} within {timeout_s} seconds")


def test_gui_buttons_run_all_simulated_workflows(tmp_path: Path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    config = load_config(project / "configs" / "simulated.yaml")
    config.audio.input_device = None
    config.audio.output_device = None
    config.storage.root = str(tmp_path / "runs")
    config.storage.compute_sha256 = False
    config.general.duration_s = 0.05
    config.sweep.duration_s = 0.1
    config.sweep.pre_silence_s = 0.02
    config.sweep.post_silence_s = 0.05
    config.sweep.rir_duration_s = 0.05
    config.repeats.minimum = 2
    config.repeats.maximum = 2
    config.repeats.required_stable_takes = 1
    config.repeats.pause_s = 0
    config.scene.duration_s = 0.05
    config.scene.ambient_duration_s = 0.05
    config.scene.countdown_s = 0
    config.scene.gap_s = 0
    config.scene.repetitions = 1
    config_path = tmp_path / "gui_test.yaml"
    save_config(config, config_path)

    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui_module,
        "device_choices",
        lambda kind=None: [f"30: Test ASIO {kind or 'duplex'} device"],
    )
    monkeypatch.setattr(gui_module, "list_devices", lambda: "30 | ASIO | 2 | 2 | Test device")
    monkeypatch.setattr(
        gui_module.messagebox,
        "showinfo",
        lambda title, message: dialogs.append((str(title), str(message))),
    )
    monkeypatch.setattr(
        gui_module.messagebox,
        "showwarning",
        lambda title, message: dialogs.append((str(title), str(message))),
    )
    monkeypatch.setattr(
        gui_module.messagebox,
        "showerror",
        lambda title, message: dialogs.append((str(title), str(message))),
    )
    experiment_names = iter(("gui_rir_az090", "gui_speech_scene01"))
    monkeypatch.setattr(
        gui_module.simpledialog,
        "askstring",
        lambda *args, **kwargs: next(experiment_names),
    )

    app = CaptureGUI(config_path)
    app.withdraw()
    try:
        assert not hasattr(app, "ess_widgets")
        assert "自动多次采集" in str(app.rir_button.cget("text"))
        loaded_runs: list[Path] = []
        original_load_run = app.viewer.load_run

        def track_load_run(path):
            loaded_runs.append(Path(path).resolve())
            return original_load_run(path)

        app.viewer.load_run = track_load_run

        # Selecting RIR and editing a sweep field updates the right-side preview
        # without saving or starting an audio stream.
        app.mode_var.set("rir")
        app._set_mode()
        app.variables["sweep.duration_s"].set("0.12")
        deadline = time.monotonic() + 2.0
        while app.viewer.preview_spec is None and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
        assert app.viewer.preview_spec is not None
        assert app.viewer.preview_spec["duration_s"] == 0.12
        assert any(
            group == "rir" and widget.winfo_manager() == "grid"
            for widget, group in app.section_widgets
        )

        # Toolbar/menu command paths.
        app.validate_config()
        app.check_hardware()
        app.show_devices()
        app.scan_scene_sources()
        for child in app.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()

        # The following .invoke() calls are the same Tk command path used by a
        # real mouse click, including saving, worker threads, event polling and
        # loading the completed run into ResultsViewer.
        app.mode_var.set("basic")
        app._set_mode()
        assert app.basic_button.winfo_manager() == "pack"
        app.basic_button.invoke()
        io_run = _wait_for_run(app, Path(config.storage.root), "io_play_record")

        app.mode_var.set("rir")
        app._set_mode()
        assert app.rir_button.winfo_manager() == "pack"
        app.rir_button.invoke()
        rir_run = _wait_for_run(app, Path(config.storage.root), "rir")
        assert sum(path == rir_run.resolve() for path in loaded_runs) >= 3

        app.mode_var.set("speech")
        app._set_mode()
        assert app.scene_button.winfo_manager() == "pack"
        assert app.scene_scan_button.winfo_manager() == "pack"
        app.scene_button.invoke()
        scene_run = _wait_for_run(app, Path(config.storage.root), "scene")

        assert (io_run / "raw" / "recording.wav").is_file()
        assert (rir_run / "processed" / "average_rir.wav").is_file()
        assert "gui_rir_az090" in rir_run.name
        rir_manifest = json.loads((rir_run / "manifest.json").read_text(encoding="utf-8"))
        assert rir_manifest["metadata"]["experiment_id"] == "gui_rir_az090"
        assert (scene_run / "raw" / "rep_001_mixture_mics.wav").is_file()
        assert (scene_run / "labels.xlsx").is_file()
        assert "gui_speech_scene01" in scene_run.name
        assert app.viewer.run_dir == scene_run
        assert all(str(button.cget("state")) == "normal" for button in (
            app.basic_button,
            app.rir_button,
            app.scene_button,
            app.scene_scan_button,
        ))
        assert not [title for title, _ in dialogs if "失败" in title or "无效" in title]
        assert sum(title == "测试完成" for title, _ in dialogs) == 3
    finally:
        app.destroy()
