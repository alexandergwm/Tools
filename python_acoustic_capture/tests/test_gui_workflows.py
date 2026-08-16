from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path

from acoustic_capture.config import load_config, save_config
from acoustic_capture.checklist import create_checklist
import acoustic_capture.gui as gui_module
from acoustic_capture.gui import CaptureGUI


def _descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from _descendants(child)


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
    checklist = create_checklist(
        tmp_path / "checklist.xlsx",
        [
            {
                "status": "待采集",
                "workflow": "rir",
                "experiment_name": "checklist_rir_001",
                "project_id": "p1",
                "wearing_id": "w04",
                "source_role": "mouth",
                "source_id": "mouth01",
                "azimuth_deg": 0,
                "elevation_deg": 0,
            }
        ],
    )

    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gui_module,
        "device_choices",
        lambda kind=None, host_api=None: [
            f"30: Test {host_api or 'ASIO'} {kind or 'duplex'} device"
        ],
    )
    monkeypatch.setattr(gui_module, "host_api_choices", lambda: ["MME", "ASIO"])
    monkeypatch.setattr(
        gui_module.filedialog, "askopenfilename", lambda **_kwargs: str(checklist)
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
    experiment_names = iter(("gui_basic_io", "gui_rir_az090", "gui_speech_scene01"))
    monkeypatch.setattr(
        gui_module.simpledialog,
        "askstring",
        lambda *args, **kwargs: next(experiment_names),
    )

    app = CaptureGUI(config_path)
    app.withdraw()
    try:
        assert not hasattr(app, "ess_widgets")
        assert "RIR" in str(app.start_button.cget("text"))
        assert app.mode_var.get() == "rir"
        assert not app.advanced_var.get()
        assert app.variables["repeats.strategy"].get().startswith("自动选优")
        assert app.field_rows["repeats.strategy"][1].winfo_manager() == "grid"
        assert app.field_rows["repeats.fixed_count"][1].winfo_manager() == ""
        app.variables["repeats.strategy"].set(
            gui_module.RIR_STRATEGY_TO_LABEL["fixed_count"]
        )
        app._set_mode()
        assert app.field_rows["repeats.fixed_count"][1].winfo_manager() == "grid"
        app.variables["repeats.strategy"].set(
            gui_module.RIR_STRATEGY_TO_LABEL["adaptive_select"]
        )
        app._set_mode()
        assert app.field_rows["sweep.start_hz"][1].winfo_manager() == ""
        assert app.field_rows["sweep.level_dbfs"][1].winfo_manager() == "grid"
        assert app.metadata_summary_panel.winfo_manager() == "grid"
        assert "人工头=" in app.metadata_summary_var.get()
        app.edit_experiment_labels()
        editor = next(
            child
            for child in app.winfo_children()
            if isinstance(child, tk.Toplevel) and child.title() == "编辑实验标签"
        )
        editor.destroy()
        app.open_checklist()
        checklist_window = next(
            child
            for child in app.winfo_children()
            if isinstance(child, tk.Toplevel)
            and child.title() == "选择测试清单实验"
        )
        tree = next(
            child
            for child in _descendants(checklist_window)
            if isinstance(child, gui_module.ttk.Treeview)
        )
        tree.selection_set(tree.get_children()[0])
        choose = next(
            child
            for child in _descendants(checklist_window)
            if isinstance(child, gui_module.ttk.Button)
            and str(child.cget("text")) == "使用所选实验"
        )
        choose.invoke()
        assert app.checklist_kind == "rir"
        assert app.config_data.metadata["wearing_id"] == "w04"
        assert "清单#2" in app.metadata_summary_var.get()
        # Continue the remainder of this test through the manual-name path.
        app.checklist_path = None
        app.checklist_row = None
        app.checklist_kind = None
        app.mode_var.set("audio")
        app.audio_preset_var.set("标准监督：目标 + 干扰 + MIXED（推荐）")
        app._audio_preset_changed()
        assert app.viewer.preview_spec is None
        assert "训练输入是 mixed" in app.viewer.summary_var.get()
        assert app.item_vars["target_only"].get()
        assert app.item_vars["mixture"].get()
        assert app.item_vars["interferer_only"].get()
        assert "人工嘴 / 目标源输出通道" in str(
            app.field_rows["audio.target_output_channel"][0].cget("text")
        )
        assert "干扰源输出通道" in str(
            app.field_rows["audio.interferer_output_channel"][0].cget("text")
        )
        app.mode_var.set("rir")
        app._set_mode()
        app.advanced_var.set(True)
        app._set_mode()
        assert app.field_rows["sweep.start_hz"][1].winfo_manager() == "grid"
        app.advanced_var.set(False)
        app._set_mode()
        assert str(app.stop_button.cget("state")) == "disabled"
        app.variables["audio.input_device"].set("old input")
        app.variables["audio.output_device"].set("old output")
        app.variables["audio.host_api"].set("MME")
        app._host_api_changed()
        assert app.variables["audio.input_device"].get() == ""
        assert app.variables["audio.output_device"].get() == ""
        assert all("MME input" in value for value in app.device_boxes["audio.input_device"].cget("values"))
        assert all("MME output" in value for value in app.device_boxes["audio.output_device"].cget("values"))

        class StopProbe:
            called = False

            def stop(self):
                self.called = True

        probe = StopProbe()
        app._active_backend = probe
        app._stop_event.clear()
        app._set_busy(True)
        app.stop_button.invoke()
        assert probe.called
        assert app._stop_event.is_set()
        app._active_backend = None
        app._stop_event.clear()
        app._set_busy(False)
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
        assert app.viewer.axes[0].get_title() == (
            "Playback sequence (timeline only - not a waveform)"
        )
        assert app.viewer.axes[0].get_ylabel() == ""
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
        app.mode_var.set("audio")
        app.audio_preset_var.set("普通单文件同步播录")
        app._audio_preset_changed()
        app._set_mode()
        assert app.start_button.winfo_manager() == "pack"
        assert "音频播录" in str(app.start_button.cget("text"))
        app.start_button.invoke()
        io_run = _wait_for_run(app, Path(config.storage.root), "io_play_record")

        app.mode_var.set("rir")
        app._set_mode()
        assert app.start_button.winfo_manager() == "pack"
        app.start_button.invoke()
        rir_run = _wait_for_run(app, Path(config.storage.root), "rir")
        assert sum(path == rir_run.resolve() for path in loaded_runs) >= 3

        app.mode_var.set("audio")
        app.audio_preset_var.set("标准监督：目标 + 干扰 + MIXED（推荐）")
        app._audio_preset_changed()
        assert app.start_button.winfo_manager() == "pack"
        assert app.scene_scan_button.winfo_manager() == "pack"
        assert "target_only" in str(app.audio_preset_help.cget("text"))
        app.start_button.invoke()
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
            app.start_button,
            app.scene_scan_button,
        ))
        assert not [title for title, _ in dialogs if "失败" in title or "无效" in title]
        assert sum(title == "测试完成" for title, _ in dialogs) == 3
    finally:
        app.destroy()
