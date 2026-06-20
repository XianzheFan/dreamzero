import numpy as np

from scripts.eval.analyze_video_pred_quality import (
    _risk_flags,
    compare_videos,
    video_metrics,
)


def test_video_metrics_flags_frozen_low_dynamic_clip():
    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)

    metrics = video_metrics(frames)

    assert metrics["temporal_freeze_frac"] == 1.0
    assert metrics["temporal_absdiff"]["mean"] == 0.0
    assert "mostly_frozen" in _risk_flags(metrics)
    assert "low_dynamic_range" in _risk_flags(metrics)


def test_video_metrics_measures_temporal_change():
    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        frames[i, :, :, :] = i * 20

    metrics = video_metrics(frames)

    assert metrics["temporal_absdiff"]["mean"] == 20.0
    assert metrics["temporal_freeze_frac"] == 0.0


def test_compare_videos_reports_rgb_mae():
    pred = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    observed = np.full((3, 8, 8, 3), 10, dtype=np.uint8)

    metrics = compare_videos(pred, observed)

    assert metrics["frame_count"] == 3
    assert metrics["mae_rgb"] == 10.0
    assert metrics["first_frame_mae_rgb"] == 10.0
    assert metrics["last_frame_mae_rgb"] == 10.0
