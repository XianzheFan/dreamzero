import numpy as np

from scripts.eval.analyze_video_pred_quality import (
    _aggregate,
    _risk_flags,
    compare_conditioning_frame_to_current_observation,
    compare_pred_to_future_trace,
    compare_videos,
    video_metrics,
    write_text_report,
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
    assert metrics["mae_rgb_by_frame"] == [10.0, 10.0, 10.0]
    assert metrics["mae_luma_by_frame"] == [10.0, 10.0, 10.0]
    assert metrics["mae_rgb_first_to_last_delta"] == 0.0


def test_compare_conditioning_frame_uses_current_observed_frame():
    pred = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    pred[0] = 30
    observed = np.stack(
        [
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ],
        axis=0,
    )

    chronological = compare_conditioning_frame_to_current_observation(
        pred,
        observed,
        includes_conditioning_frame=True,
        observed_window_mode="history-chronological",
    )
    current_first = compare_conditioning_frame_to_current_observation(
        pred,
        observed,
        includes_conditioning_frame=True,
        observed_window_mode="history-current-first",
    )
    unavailable = compare_conditioning_frame_to_current_observation(
        pred,
        observed,
        includes_conditioning_frame=False,
        observed_window_mode="history-chronological",
    )

    assert chronological["observed_frame_index"] == 2
    assert chronological["mae_rgb"] == 0.0
    assert current_first["observed_frame_index"] == 0
    assert current_first["mae_rgb"] == 20.0
    assert unavailable is None


def test_compare_pred_to_future_trace_aligns_after_conditioning_frame():
    pred = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    future = np.stack(
        [
            np.full((4, 4, 3), 99, dtype=np.uint8),
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ],
        axis=0,
    )
    trace = {
        "path": "episode_1000.npz",
        "rgb_trace_step": np.asarray([10, 11, 12, 13], dtype=np.int32),
        "left_rgb": future,
        "right_rgb": future + 1,
    }

    metrics = compare_pred_to_future_trace(
        pred,
        trace,
        agent_id=0,
        env_step=10,
        includes_conditioning_frame=False,
    )

    assert metrics["matched_frame_count"] == 3
    assert metrics["first_matched_step"] == 11
    assert metrics["last_matched_step"] == 13
    assert metrics["view_key"] == "left_rgb"
    assert metrics["mae_rgb"] == 20.0
    assert metrics["first_frame_mae_rgb"] == 10.0
    assert metrics["last_frame_mae_rgb"] == 30.0
    assert metrics["mae_rgb_by_frame"] == [10.0, 20.0, 30.0]
    assert metrics["mae_rgb_first_to_last_delta"] == 20.0
    assert metrics["best_alignment_offset"] == 0
    assert metrics["best_alignment_mae_rgb"] == 20.0


def test_compare_pred_to_future_trace_skips_conditioning_frame():
    pred = np.stack(
        [
            np.full((4, 4, 3), 99, dtype=np.uint8),
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
        ],
        axis=0,
    )
    future = np.stack(
        [
            np.full((4, 4, 3), 99, dtype=np.uint8),
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
        ],
        axis=0,
    )
    trace = {
        "path": "episode_1000.npz",
        "rgb_trace_step": np.asarray([10, 11, 12], dtype=np.int32),
        "left_rgb": future,
    }

    metrics = compare_pred_to_future_trace(
        pred,
        trace,
        agent_id=0,
        env_step=10,
        includes_conditioning_frame=True,
    )

    assert metrics["matched_frame_count"] == 2
    assert metrics["first_matched_step"] == 11
    assert metrics["last_matched_step"] == 12
    assert metrics["pred_start_index"] == 1
    assert metrics["skipped_conditioning_frame"] is True
    assert metrics["mae_rgb"] == 0.0


def test_compare_pred_to_future_trace_scans_alignment_offsets():
    pred = np.stack(
        [
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
            np.full((4, 4, 3), 40, dtype=np.uint8),
        ],
        axis=0,
    )
    future = np.stack(
        [
            np.full((4, 4, 3), 0, dtype=np.uint8),
            np.full((4, 4, 3), 10, dtype=np.uint8),
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
            np.full((4, 4, 3), 40, dtype=np.uint8),
        ],
        axis=0,
    )
    trace = {
        "path": "episode_1000.npz",
        "rgb_trace_step": np.asarray([10, 11, 12, 13, 14], dtype=np.int32),
        "left_rgb": future,
    }

    metrics = compare_pred_to_future_trace(
        pred,
        trace,
        agent_id=0,
        env_step=10,
        includes_conditioning_frame=False,
        offset_radius=2,
    )

    assert metrics["mae_rgb"] == 10.0
    assert metrics["best_alignment_offset"] == 1
    assert metrics["best_alignment_mae_rgb"] == 0.0
    assert metrics["best_alignment_improvement_rgb"] == 10.0
    assert metrics["best_alignment_matched_frame_count"] == 3


def test_video_quality_summary_names_condition_window_metric(tmp_path):
    payload = {
        "root": str(tmp_path),
        "videos": [
            {
                "path": "session/infer0000_agent0.mp4",
                "agent_id": 0,
                "env_step": 0,
                "pred_latent_start_frame": 0,
                "pred_latent_end_frame": 5,
                "pred_latent_includes_conditioning_frame": True,
                "current_start_frame_after_infer": 5,
                "cached_until_frame": 5,
                "shared_global_wrist_window_mode": "history-current-first",
                "video_pred_wrist_window_mode": "history-chronological",
                "reset_causal_state_each_infer": True,
                "video_pred_rollout_mode": "noncausal",
                "last_video_pred_rollout_mode": "noncausal",
                "metrics": {
                    "temporal_absdiff": {"mean": 2.0, "p95": 4.0},
                    "temporal_freeze_frac": 0.0,
                    "laplacian_var": {"mean": 30.0},
                    "saturation_frac": {"mean": 0.0},
                },
                "pred_vs_condition_window": {"mae_rgb": 12.5},
                "pred_vs_observed": {"mae_rgb": 12.5},
                "pred_conditioning_frame_vs_current_observation": {
                    "mae_rgb": 4.5,
                    "mae_luma": 3.5,
                    "observed_frame_index": 0,
                },
                "pred_vs_future": {
                    "mae_rgb": 8.5,
                    "mae_luma": 7.5,
                    "first_frame_mae_rgb": 5.0,
                    "last_frame_mae_rgb": 11.0,
                    "mae_rgb_by_frame": [5.0, 8.0, 10.0, 11.0],
                    "mae_luma_by_frame": [4.0, 7.0, 8.0, 11.0],
                    "mae_rgb_first_to_last_delta": 6.0,
                    "best_alignment_offset": 1,
                    "best_alignment_mae_rgb": 6.5,
                    "best_alignment_improvement_rgb": 2.0,
                    "matched_frame_count": 4,
                    "first_matched_step": 1,
                    "last_matched_step": 4,
                },
                "risk_flags": [],
            }
        ],
        "failures": [],
    }
    payload["summary"] = _aggregate(payload["videos"])

    assert payload["summary"]["pred_vs_condition_window_mae_rgb_mean"] == 12.5
    assert payload["summary"]["pred_vs_observed_mae_rgb_mean"] == 12.5
    assert payload["summary"]["pred_conditioning_frame_mae_rgb_mean"] == 4.5
    assert payload["summary"]["pred_conditioning_frame_mae_luma_mean"] == 3.5
    assert payload["summary"]["pred_conditioning_frame_available_count"] == 1
    assert payload["summary"]["pred_vs_future_mae_rgb_mean"] == 8.5
    assert payload["summary"]["pred_vs_future_mae_luma_mean"] == 7.5
    assert payload["summary"]["pred_vs_future_matched_frame_count_mean"] == 4.0
    assert payload["summary"]["pred_vs_future_mae_rgb_first_to_last_delta_mean"] == 6.0
    assert payload["summary"]["pred_vs_future_mae_rgb_by_frame_mean"] == [
        5.0,
        8.0,
        10.0,
        11.0,
    ]
    assert payload["summary"]["pred_vs_future_mae_luma_by_frame_mean"] == [
        4.0,
        7.0,
        8.0,
        11.0,
    ]
    assert payload["summary"]["pred_vs_future_best_alignment_offset_counts"] == {"1": 1}
    assert payload["summary"]["pred_vs_future_best_alignment_offset_abs_mean"] == 1.0
    assert payload["summary"]["pred_vs_future_best_alignment_mae_rgb_mean"] == 6.5
    assert payload["summary"]["pred_vs_future_best_alignment_improvement_rgb_mean"] == 2.0
    assert payload["summary"]["video_pred_rollout_mode_counts"] == {"noncausal": 1}
    assert payload["summary"]["pred_latent_includes_conditioning_frame_counts"] == {
        "True": 1
    }
    assert payload["summary"]["shared_global_wrist_window_mode_counts"] == {
        "history-current-first": 1
    }
    assert payload["summary"]["video_pred_wrist_window_mode_counts"] == {
        "history-chronological": 1
    }

    report = tmp_path / "report.txt"
    write_text_report(payload, report)
    text = report.read_text(encoding="utf-8")
    assert "video_pred_rollout_mode_counts: {'noncausal': 1}" in text
    assert "pred_latent_includes_conditioning_frame_counts" in text
    assert "shared_global_wrist_window_mode_counts" in text
    assert "video_pred_wrist_window_mode_counts" in text
    assert "rollout=noncausal/noncausal" in text
    assert "action_wrist_window=history-current-first" in text
    assert "video_wrist_window=history-chronological" in text
    assert "pred_vs_condition_window_mae_rgb_mean" in text
    assert "pred_conditioning_frame_mae_rgb_mean" in text
    assert "pred_vs_future_mae_rgb_mean" in text
    assert "pred_vs_future_mae_rgb_by_frame_mean" in text
    assert "pred_vs_future_best_alignment_offset_counts" in text
    assert "condition_window_mae=12.5" in text
    assert "conditioning_t0_mae=4.5" in text
    assert "conditioning_t0_obs_idx=0" in text
    assert "includes_conditioning=True" in text
    assert "future_mae=8.5" in text
    assert "future_best_offset=1" in text
    assert "future_best_mae=6.5" in text
    assert "future_delta=6.0" in text
    assert "future_steps=1:4" in text
    assert "latent_frames=0:5" in text
