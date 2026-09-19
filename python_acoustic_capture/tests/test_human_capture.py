import csv
import json
import numpy as np
import soundfile as sf
from acoustic_capture.audio import SimulatedBackend, CaptureResult
from acoustic_capture.config import ExperimentConfig, save_config, load_config
from acoustic_capture.human import capture_human, read_corpus


def configuration(tmp_path):
    cfg = ExperimentConfig()
    cfg.audio.backend = "simulated"
    cfg.audio.input_channels = [1, 2, 3]
    cfg.human.duration_s = 4
    cfg.human.preparation_s = 0
    cfg.storage.root = str(tmp_path / "runs")
    cfg.storage.compute_sha256 = False
    folder = tmp_path / "noise"
    folder.mkdir()
    sf.write(folder / "test.wav", .1 * np.sin(np.arange(4800) * .05), 48000)
    cfg.scene.interferer_folder = str(folder)
    return cfg


def test_s4_three_stages_keep_multichannel_and_do_not_claim_pairs(tmp_path):
    cfg = configuration(tmp_path)
    updates = []
    store = capture_human(cfg, SimulatedBackend(cfg.audio), progress=updates.append)
    rows = list(csv.DictReader((store.root / "raw/labels.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 6
    assert {r["stage"] for r in rows} == {"target_only", "interferer_only", "mixture"}
    assert all(r["strict_supervised_pair"] == "False" for r in rows)
    for row in rows:
        info = sf.info(store.root / (row["clip"] or row["recording"]))
        assert info.channels == 3
        assert info.frames == 4 * 48000
    assert [u["stage"] for u in updates] == ["target_only", "interferer_only", "mixture"]
    assert store.manifest["summary"]["completed_stages"] == 3


def test_s4_stop_keeps_short_stage_and_labels(tmp_path):
    cfg = configuration(tmp_path)
    stopped = [False]
    class Partial(SimulatedBackend):
        def record(self, frames):
            stopped[0] = True
            return CaptureResult(np.zeros((500, 3), dtype=np.float32), {"xrun": False})
    store = capture_human(cfg, Partial(cfg.audio), stop_requested=lambda: stopped[0])
    rows = list(csv.DictReader((store.root / "raw/labels.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 1 and rows[0]["status"] == "partial"
    assert rows[0]["end_sample"] == "500"
    assert store.manifest["status"] == "cancelled"


def test_s4_one_line_per_round_shared_by_stages_and_labels(tmp_path):
    cfg = configuration(tmp_path)
    corpus = tmp_path / "sentences.txt"
    sentences = ["太阳从东方升起来", "主楼位于校园的东北侧"]
    corpus.write_text("\n".join(sentences), encoding="utf-8-sig")
    cfg.human.corpus_file = str(corpus)
    cfg.human.rounds = 3  # Also verify corpus wrap-around.
    updates = []
    store = capture_human(cfg, SimulatedBackend(cfg.audio), progress=updates.append)
    assert [u["prompt"] for u in updates] == [sentences[0]] * 3 + [sentences[1]] * 3 + [sentences[0]] * 3
    with (store.root / "raw/labels.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        expected = "" if row["stage"] == "interferer_only" else sentences[(int(row["round"]) - 1) % 2]
        assert row["prompt_text"] == expected
        assert row["strict_supervised_pair"] == "False"


def test_corpus_roundtrip_and_default_sentences(tmp_path):
    cfg = configuration(tmp_path)
    corpus = tmp_path / "read.txt"
    corpus.write_text("太阳从东方升起来\nHello world\n", encoding="utf-8")
    cfg.human.corpus_file = str(corpus)
    path = tmp_path / "test.yaml"
    save_config(cfg, path)
    assert read_corpus(load_config(path).human.corpus_file) == ["太阳从东方升起来", "Hello world"]
    assert len(read_corpus()) == 19


def test_gui_four_entries_and_human_worker(tmp_path, monkeypatch):
    import time
    import acoustic_capture.gui as gui
    cfg = configuration(tmp_path)
    cfg.human.rounds = 2
    path = tmp_path / "gui.yaml"
    save_config(cfg, path)
    monkeypatch.setattr(gui, "host_api_choices", lambda: [])
    monkeypatch.setattr(gui, "device_choices", lambda *args: [])
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda *args, **kwargs: None)
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    app = gui.CaptureGUI(path)
    app.withdraw()
    try:
        app.mode_var.set("record")
        app._select_mode()
        assert not app.field_rows["audio.output_device"][1].winfo_manager()
        app.mode_var.set("human")
        app._select_mode()
        assert app.field_rows["audio.interferer_output_channel"][1].winfo_manager() == "grid"
        assert not app.field_rows["scene.target_folder"][1].winfo_manager()
        assert app.viewer.human_prompt_panel.winfo_manager() == "pack"
        from tkinter.font import Font
        prompt_font = Font(root=app, font=app.viewer.human_prompt_label.cget("font"))
        assert prompt_font.actual("size") == 24
        assert prompt_font.actual("weight") == "bold"
        assert all(not w.winfo_manager() for w in app.scene_widgets)
        app.viewer.prepare_live_animation()
        draws = []
        original_draw_idle = app.viewer.canvas.draw_idle
        app.viewer.canvas.draw_idle = lambda: draws.append(True)
        app.viewer.update_live_progress(48000, 192000, 48000, "recording")
        assert not draws  # advancing a cursor must not redraw the full plot
        app.viewer.canvas.draw_idle = original_draw_idle
        shown = []
        original_show = app.viewer.show_human_stage
        def observe_stage(stage, prompt, ordinal, rounds, frames, sample_rate):
            original_show(stage, prompt, ordinal, rounds, frames, sample_rate)
            shown.append((stage, ordinal, app.viewer.human_prompt.get(), app.viewer.human_instruction.get()))
        monkeypatch.setattr(app.viewer, "show_human_stage", observe_stage)
        app.run("human")
        deadline = time.monotonic() + 15
        while app._busy and time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        assert not app._busy
        assert not errors
        manifest = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text("utf-8"))
        assert manifest["kind"] == "human" and manifest["status"] == "completed"
        assert manifest["summary"]["completed_stages"] == 6
        assert len(shown) == 6
        for offset, sentence in ((0, read_corpus()[0]), (3, read_corpus()[1])):
            assert shown[offset][2] == shown[offset + 2][2] == sentence
            assert "请不要说话" in shown[offset + 1][2]
            assert "同一句" in shown[offset + 2][3]
    finally:
        app.destroy()
