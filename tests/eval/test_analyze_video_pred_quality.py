import importlib.util
from pathlib import Path

import numpy as np


def _load_video_quality_module():
    path = Path(__file__).resolve().parents[2] / "scripts/eval/analyze_video_pred_quality.py"
    spec = importlib.util.spec_from_file_location("analyze_video_pred_quality_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_mp4(path: Path, frames: np.ndarray) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=20)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_analyze_video_pred_dir_categorizes_and_summarizes(tmp_path):
    mod = _load_video_quality_module()
    base = np.zeros((4, 16, 20, 3), dtype=np.uint8)
    for i in range(base.shape[0]):
        base[i, :, :, :] = 32 + i * 16
        base[i, 4:12, 5:15, 1] = 180

    _write_mp4(tmp_path / "session_abcd" / "infer0000_env0000_agent0.mp4", base)
    _write_mp4(
        tmp_path / "session_abcd" / "observed" / "infer0000_env0000_observed_agent0.mp4",
        base,
    )
    _write_mp4(
        tmp_path / "session_abcd" / "comparison" / "infer0000_env0000_agent0_compare.mp4",
        np.concatenate([base, base, base], axis=2),
    )

    report = mod.analyze_video_pred_dir(tmp_path)

    by_category = report["summary"]["by_category"]
    assert by_category["pred"]["count"] == 1
    assert by_category["observed"]["count"] == 1
    assert by_category["comparison"]["count"] == 1
    assert report["summary"]["num_files"] == 3
    assert all(item["frames_read"] == 4 for item in report["files"])


def test_contact_sheet_writer_creates_image(tmp_path):
    mod = _load_video_quality_module()
    frames = np.full((3, 12, 16, 3), 96, dtype=np.uint8)
    _write_mp4(tmp_path / "session_abcd" / "infer0000_env0000_agent0.mp4", frames)
    report = mod.analyze_video_pred_dir(tmp_path)

    out = tmp_path / "contact.jpg"
    mod.write_contact_sheet(report["files"], out)

    assert out.exists()
    assert out.stat().st_size > 0
