import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_video_utils():
    module_name = "video_utils_for_pyav_fallback_test"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "groot"
        / "vla"
        / "common"
        / "utils"
        / "misc"
        / "video_utils.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


video_utils = _load_video_utils()


def test_decord_timestamp_reader_falls_back_to_pyav(monkeypatch):
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    frame_ts = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    def fake_read_all_frames(video_path):
        assert video_path == "sample.mp4"
        return frames, frame_ts

    monkeypatch.setattr(video_utils, "DECORD_AVAILABLE", False)
    monkeypatch.setattr(video_utils, "_read_all_frames_pyav", fake_read_all_frames)

    out = video_utils.get_frames_by_timestamps(
        "sample.mp4",
        [0.49, 0.99],
        video_backend="decord",
    )

    np.testing.assert_array_equal(out, frames[[1, 2]])


def test_decord_index_reader_falls_back_to_pyav(monkeypatch):
    frames = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)

    def fake_read_all_frames(video_path):
        assert video_path == "sample.mp4"
        return frames, np.arange(len(frames), dtype=np.float64)

    monkeypatch.setattr(video_utils, "DECORD_AVAILABLE", False)
    monkeypatch.setattr(video_utils, "_read_all_frames_pyav", fake_read_all_frames)

    out = video_utils.get_frames_by_indices(
        "sample.mp4",
        [0, 3],
        video_backend="decord",
    )

    np.testing.assert_array_equal(out, frames[[0, 3]])
