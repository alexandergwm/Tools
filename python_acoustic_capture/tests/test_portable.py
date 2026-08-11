from pathlib import Path

from acoustic_capture.config import load_config
import portable_entry


def test_portable_assets_install_with_relative_paths(tmp_path: Path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(portable_entry, "_bundle_root", lambda: project)

    portable_entry._install_assets(tmp_path)
    config = load_config(tmp_path / "configs" / "portable_default.yaml")

    assert Path(config.storage.root) == (tmp_path / "runs").resolve()
    assert Path(config.scene.target_folder) == (tmp_path / "audio" / "targets").resolve()
    assert Path(config.scene.interferer_folder) == (
        tmp_path / "audio" / "interferers"
    ).resolve()
    assert (tmp_path / "audio" / "source.wav").is_file()
    assert (tmp_path / "acoustic_capture_checklist_template.xlsx").is_file()
    assert (tmp_path / "datasets").is_dir()


def test_portable_asset_install_does_not_overwrite_user_config(tmp_path: Path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(portable_entry, "_bundle_root", lambda: project)
    portable_entry._install_assets(tmp_path)
    config_path = tmp_path / "configs" / "portable_default.yaml"
    config_path.write_text("user-edited", encoding="utf-8")

    portable_entry._install_assets(tmp_path)

    assert config_path.read_text(encoding="utf-8") == "user-edited"
