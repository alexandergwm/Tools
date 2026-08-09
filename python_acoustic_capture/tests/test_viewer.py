from pathlib import Path
from types import SimpleNamespace

import numpy as np

from acoustic_capture.viewer import (
    ResultsViewer,
    discover_audio_files,
    display_points,
    pan_interval,
    select_audio_channel,
    zoom_interval,
)


def test_discover_audio_files(tmp_path: Path):
    for folder in ("references", "raw", "processed"):
        (tmp_path / folder).mkdir()
    (tmp_path / "references/played.wav").touch()
    (tmp_path / "raw/take_001.wav").touch()
    (tmp_path / "processed/average_rir.wav").touch()
    files = discover_audio_files(tmp_path)
    assert files["playback"][0].name == "played.wav"
    assert files["recording"][0].name == "take_001.wav"
    assert files["rir"][0].name == "average_rir.wav"


def test_display_points_keeps_multichannel_shape():
    data = np.zeros((100_000, 6), dtype=np.float32)
    times, shown = display_points(data, 48_000, limit=1_000)
    assert len(times) <= 1_000
    assert shown.shape[1] == 6


def test_select_audio_channel_supports_auto_mix_and_explicit_channel():
    data = np.column_stack(
        [np.full(32, 0.1, dtype=np.float32), np.full(32, 0.8, dtype=np.float32)]
    )
    automatic, automatic_label = select_audio_channel(data, "自动选择电平最高的通道")
    mixed, mixed_label = select_audio_channel(data, "混合全部通道")
    explicit, explicit_label = select_audio_channel(data, "通道 1")
    assert np.allclose(automatic, 0.8)
    assert automatic_label == "通道 2"
    assert np.allclose(mixed, 0.45)
    assert mixed_label == "全部通道混合"
    assert np.allclose(explicit, 0.1)
    assert explicit_label == "通道 1"


def test_zoom_and_pan_intervals_stay_inside_signal_bounds():
    zoomed = zoom_interval((0.0, 10.0), (0.0, 10.0), center=2.0, step=1.0)
    assert np.allclose(zoomed, (0.4, 8.4))
    assert pan_interval(zoomed, (0.0, 10.0), delta=5.0) == (2.0, 10.0)
    assert pan_interval(zoomed, (0.0, 10.0), delta=-5.0) == (0.0, 8.0)


def test_zoom_modifier_accepts_matplotlib_modifier_collection():
    event = SimpleNamespace(key=None, modifiers=frozenset({"ctrl"}))
    assert ResultsViewer._modifier_pressed(event)
