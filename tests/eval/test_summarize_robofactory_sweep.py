import json
from pathlib import Path

from scripts.eval.summarize_robofactory_sweep import summarize_sweep


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_sweep_extracts_trace_metrics(tmp_path):
    scale_1 = tmp_path / "jscale_1p0"
    scale_2 = tmp_path / "jscale_1p5"
    _write_json(
        scale_2 / "results.json",
        {
            "success_rate": 1.0,
            "results": [{"seed": 1000, "success": True}],
            "eval_config": {
                "replan_every": 12,
                "action_representation": "absolute_qpos",
                "joint_target_scale": 1.5,
                "joint_target_scale_reference": "auto",
                "joint_target_scale_clip": 0.25,
                "joint_target_slew_rate": 0.35,
                "joint_target_accel_limit": 0.08,
                "replan_boundary_blend_steps": 4,
                "temporal_action_ensemble_decay": 0.6,
            },
        },
    )
    _write_json(
        scale_2 / "action_dump_summary.json",
        {
            "summary": {
                "all_grippers_decisive_close": 1,
                "action_representation_counts": {"absolute_qpos": 1},
                "any_gripper_never_closes": 0,
                "mean_joint_step_delta": 0.04,
                "max_joint_step_delta": 0.2,
                "mean_joint_step_accel": 0.03,
                "max_joint_step_accel": 0.12,
                "mean_joint_step_jerk": 0.025,
                "max_joint_step_jerk": 0.11,
                "mean_joint_accel_to_delta_ratio": 0.75,
                "max_joint_accel_to_delta_ratio": 0.6,
                "joint_delta_sign_flip_frac": 0.5,
                "mean_joint_delta_sign_flip_frac": 0.4,
                "joint_delta_sign_flip_count": 6,
                "joint_delta_active_pair_count": 12,
                "mean_pred_chunk_joint_step_delta": 0.05,
                "max_pred_chunk_joint_step_delta": 0.21,
                "mean_pred_chunk_joint_step_accel": 0.04,
                "max_pred_chunk_joint_step_accel": 0.16,
                "mean_pred_chunk_joint_step_jerk": 0.035,
                "max_pred_chunk_joint_step_jerk": 0.14,
                "mean_pred_chunk_joint_accel_to_delta_ratio": 0.8,
                "max_pred_chunk_joint_accel_to_delta_ratio": 0.76,
                "pred_chunk_joint_delta_sign_flip_frac": 0.45,
                "mean_model_replan_boundary_joint_jump": 0.035,
                "max_model_replan_boundary_joint_jump": 0.11,
                "mean_replan_boundary_joint_jump": 0.025,
                "max_replan_boundary_joint_jump": 0.09,
                "env_trace": {
                    "left_tcp_to_grasp_target_min_mean": 0.07,
                    "right_tcp_to_grasp_target_min_mean": 0.08,
                    "success_margin_max_best": -0.02,
                    "success_margin_max_mean": -0.03,
                    "left_grasp_episodes": 1,
                    "right_grasp_episodes": 1,
                },
            },
            "episodes": [
                {
                    "first_cmd_delta_mean": 0.08,
                    "first_cmd_delta_max": 0.18,
                    "norm_debug": {
                        "raw_joint_saturation_frac": 0.04,
                        "raw_gripper_saturation_frac": 0.5,
                        "joint_clamp_delta_mean": 0.03,
                        "joint_clamp_delta_max": 0.12,
                        "raw_joint_saturation_frac_first_step": 0.02,
                        "raw_joint_saturation_frac_late_steps": 0.05,
                        "raw_joint_saturation_frac_by_arm": {
                            "left": 0.06,
                            "right": 0.02,
                        },
                        "joint_clamp_delta_mean_by_arm": {
                            "left": 0.04,
                            "right": 0.02,
                        },
                    },
                    "joint_debug": {
                        "pre_blend_replan_boundary_joint_jump": {
                            "mean_abs": 0.08,
                            "max_abs": 0.22,
                        },
                        "pre_ensemble_replan_boundary_joint_jump": {
                            "mean_abs": 0.07,
                            "max_abs": 0.2,
                        },
                        "temporal_ensemble_correction_joint": {
                            "mean_abs": 0.025,
                            "max_abs": 0.11,
                        },
                        "accel_limiter_correction_joint": {
                            "mean_abs": 0.012,
                            "max_abs": 0.05,
                        },
                        "pre_slew_joint_step_delta": {
                            "mean_abs": 0.05,
                            "max_abs": 0.19,
                        },
                        "slew_correction_joint": {
                            "mean_abs": 0.01,
                            "max_abs": 0.04,
                        },
                    },
                }
            ],
        },
    )
    _write_json(
        scale_1 / "results.json",
        {
            "success_rate": 0.0,
            "results": [{"seed": 1000, "success": False}],
            "eval_config": {
                "replan_every": 24,
                "action_representation": "absolute_qpos",
                "joint_delta_scale": 1.0,
                "joint_delta_scale_reference": "auto",
                "joint_delta_scale_clip": 0.25,
                "joint_target_slew_rate": 0.35,
                "joint_target_accel_limit": 0.0,
                "replan_boundary_blend_steps": 0,
                "temporal_action_ensemble_decay": 0.0,
            },
        },
    )
    _write_json(
        scale_1 / "action_dump_summary.json",
        {
            "summary": {
                "all_grippers_decisive_close": 1,
                "action_representation_counts": {"absolute_qpos": 1},
                "any_gripper_never_closes": 0,
                "mean_joint_step_delta": 0.02,
                "max_joint_step_delta": 0.1,
                "mean_joint_step_accel": 0.01,
                "max_joint_step_accel": 0.04,
                "mean_joint_step_jerk": 0.008,
                "max_joint_step_jerk": 0.03,
                "mean_joint_accel_to_delta_ratio": 0.5,
                "max_joint_accel_to_delta_ratio": 0.4,
                "joint_delta_sign_flip_frac": 0.25,
                "mean_joint_delta_sign_flip_frac": 0.2,
                "joint_delta_sign_flip_count": 2,
                "joint_delta_active_pair_count": 8,
                "mean_pred_chunk_joint_step_delta": 0.025,
                "max_pred_chunk_joint_step_delta": 0.11,
                "mean_pred_chunk_joint_step_accel": 0.012,
                "max_pred_chunk_joint_step_accel": 0.045,
                "mean_pred_chunk_joint_step_jerk": 0.009,
                "max_pred_chunk_joint_step_jerk": 0.035,
                "mean_pred_chunk_joint_accel_to_delta_ratio": 0.48,
                "max_pred_chunk_joint_accel_to_delta_ratio": 0.41,
                "pred_chunk_joint_delta_sign_flip_frac": 0.15,
                "mean_model_replan_boundary_joint_jump": 0.018,
                "max_model_replan_boundary_joint_jump": 0.055,
                "mean_replan_boundary_joint_jump": 0.015,
                "max_replan_boundary_joint_jump": 0.05,
                "env_trace": {
                    "left_tcp_to_grasp_target_min_mean": 0.13,
                    "right_tcp_to_grasp_target_min_mean": 0.12,
                    "success_margin_max_best": -0.14,
                    "success_margin_max_mean": -0.14,
                    "left_grasp_episodes": 0,
                    "right_grasp_episodes": 0,
                },
            },
            "episodes": [
                {
                    "first_cmd_delta_mean": 0.02,
                    "first_cmd_delta_max": 0.06,
                    "norm_debug": {
                        "raw_joint_saturation_frac": 0.0,
                        "raw_gripper_saturation_frac": 0.0,
                        "joint_clamp_delta_mean": 0.0,
                        "joint_clamp_delta_max": 0.0,
                        "raw_joint_saturation_frac_first_step": 0.0,
                        "raw_joint_saturation_frac_late_steps": 0.0,
                        "raw_joint_saturation_frac_by_arm": {
                            "left": 0.0,
                            "right": 0.0,
                        },
                        "joint_clamp_delta_mean_by_arm": {
                            "left": 0.0,
                            "right": 0.0,
                        },
                    },
                    "joint_debug": {
                        "pre_blend_replan_boundary_joint_jump": {
                            "mean_abs": 0.015,
                            "max_abs": 0.05,
                        },
                        "pre_ensemble_replan_boundary_joint_jump": {
                            "mean_abs": 0.015,
                            "max_abs": 0.05,
                        },
                        "temporal_ensemble_correction_joint": {
                            "mean_abs": 0.0,
                            "max_abs": 0.0,
                        },
                        "accel_limiter_correction_joint": {
                            "mean_abs": 0.0,
                            "max_abs": 0.0,
                        },
                        "pre_slew_joint_step_delta": {
                            "mean_abs": 0.02,
                            "max_abs": 0.1,
                        },
                        "slew_correction_joint": {
                            "mean_abs": 0.0,
                            "max_abs": 0.0,
                        },
                    },
                }
            ],
        },
    )

    rows = summarize_sweep(str(tmp_path))

    assert [row["scale"] for row in rows] == [1.0, 1.5]
    assert rows[0]["success_count"] == 0
    assert rows[0]["video_pred_rollout_mode"] is None
    assert rows[0]["replan_every"] == 24.0
    assert rows[0]["action_representation"] == "absolute_qpos"
    assert rows[0]["action_representation_counts"] == {"absolute_qpos": 1}
    assert rows[0]["target_slew_rate"] == 0.35
    assert rows[0]["target_accel_limit"] == 0.0
    assert rows[0]["replan_boundary_blend_steps"] == 0.0
    assert rows[0]["temporal_action_ensemble_decay"] == 0.0
    assert rows[0]["first_cmd_delta_mean"] == 0.02
    assert rows[0]["first_cmd_delta_max"] == 0.06
    assert rows[0]["target_min_mean_left"] == 0.13
    assert rows[0]["raw_joint_saturation_frac"] == 0.0
    assert rows[0]["raw_joint_saturation_frac_left"] == 0.0
    assert rows[0]["raw_joint_saturation_frac_right"] == 0.0
    assert rows[0]["raw_joint_saturation_frac_first_step"] == 0.0
    assert rows[0]["raw_joint_saturation_frac_late_steps"] == 0.0
    assert rows[0]["raw_gripper_saturation_frac"] == 0.0
    assert rows[0]["joint_clamp_delta_mean"] == 0.0
    assert rows[0]["joint_clamp_delta_mean_left"] == 0.0
    assert rows[0]["joint_clamp_delta_mean_right"] == 0.0
    assert rows[0]["joint_clamp_delta_max"] == 0.0
    assert rows[0]["mean_joint_accel_to_delta_ratio"] == 0.5
    assert rows[0]["max_joint_accel_to_delta_ratio"] == 0.4
    assert rows[0]["joint_delta_sign_flip_frac"] == 0.25
    assert rows[0]["mean_joint_delta_sign_flip_frac"] == 0.2
    assert rows[0]["joint_delta_sign_flip_count"] == 2
    assert rows[0]["joint_delta_active_pair_count"] == 8
    assert rows[0]["mean_pred_chunk_joint_step_delta"] == 0.025
    assert rows[0]["max_pred_chunk_joint_step_accel"] == 0.045
    assert rows[0]["mean_joint_step_jerk"] == 0.008
    assert rows[0]["max_joint_step_jerk"] == 0.03
    assert rows[0]["mean_pred_chunk_joint_step_jerk"] == 0.009
    assert rows[0]["max_pred_chunk_joint_step_jerk"] == 0.035
    assert rows[0]["mean_pred_chunk_joint_accel_to_delta_ratio"] == 0.48
    assert rows[0]["pred_chunk_joint_delta_sign_flip_frac"] == 0.15
    assert rows[0]["max_model_replan_boundary_joint_jump"] == 0.055
    assert rows[0]["max_pre_blend_replan_boundary_joint_jump"] == 0.05
    assert rows[0]["max_pre_ensemble_replan_boundary_joint_jump"] == 0.05
    assert rows[0]["mean_temporal_ensemble_correction_joint"] == 0.0
    assert rows[0]["mean_accel_limiter_correction_joint"] == 0.0
    assert rows[0]["mean_slew_correction_joint"] == 0.0
    assert rows[0]["right_grasp_episodes"] == 0
    assert rows[1]["success_count"] == 1
    assert rows[1]["video_pred_rollout_mode"] is None
    assert rows[1]["replan_every"] == 12.0
    assert rows[1]["action_representation"] == "absolute_qpos"
    assert rows[1]["scale_clip"] == 0.25
    assert rows[1]["target_slew_rate"] == 0.35
    assert rows[1]["target_accel_limit"] == 0.08
    assert rows[1]["replan_boundary_blend_steps"] == 4.0
    assert rows[1]["temporal_action_ensemble_decay"] == 0.6
    assert rows[1]["target_min_mean_right"] == 0.08
    assert rows[1]["barrier_margin_best"] == -0.02
    assert rows[1]["mean_joint_step_accel"] == 0.03
    assert rows[1]["mean_joint_step_jerk"] == 0.025
    assert rows[1]["mean_joint_accel_to_delta_ratio"] == 0.75
    assert rows[1]["max_joint_accel_to_delta_ratio"] == 0.6
    assert rows[1]["joint_delta_sign_flip_frac"] == 0.5
    assert rows[1]["mean_joint_delta_sign_flip_frac"] == 0.4
    assert rows[1]["joint_delta_sign_flip_count"] == 6
    assert rows[1]["joint_delta_active_pair_count"] == 12
    assert rows[1]["mean_pred_chunk_joint_step_delta"] == 0.05
    assert rows[1]["max_pred_chunk_joint_step_accel"] == 0.16
    assert rows[1]["max_pred_chunk_joint_step_jerk"] == 0.14
    assert rows[1]["mean_pred_chunk_joint_accel_to_delta_ratio"] == 0.8
    assert rows[1]["pred_chunk_joint_delta_sign_flip_frac"] == 0.45
    assert rows[1]["max_model_replan_boundary_joint_jump"] == 0.11
    assert rows[1]["max_replan_boundary_joint_jump"] == 0.09
    assert rows[1]["mean_pre_blend_replan_boundary_joint_jump"] == 0.08
    assert rows[1]["max_pre_blend_replan_boundary_joint_jump"] == 0.22
    assert rows[1]["mean_pre_ensemble_replan_boundary_joint_jump"] == 0.07
    assert rows[1]["max_pre_ensemble_replan_boundary_joint_jump"] == 0.2
    assert rows[1]["mean_temporal_ensemble_correction_joint"] == 0.025
    assert rows[1]["max_temporal_ensemble_correction_joint"] == 0.11
    assert rows[1]["mean_accel_limiter_correction_joint"] == 0.012
    assert rows[1]["max_accel_limiter_correction_joint"] == 0.05
    assert rows[1]["mean_pre_slew_joint_step_delta"] == 0.05
    assert rows[1]["max_pre_slew_joint_step_delta"] == 0.19
    assert rows[1]["mean_slew_correction_joint"] == 0.01
    assert rows[1]["max_slew_correction_joint"] == 0.04
    assert rows[1]["first_cmd_delta_mean"] == 0.08
    assert rows[1]["first_cmd_delta_max"] == 0.18
    assert rows[1]["raw_joint_saturation_frac"] == 0.04
    assert rows[1]["raw_joint_saturation_frac_left"] == 0.06
    assert rows[1]["raw_joint_saturation_frac_right"] == 0.02
    assert rows[1]["raw_joint_saturation_frac_first_step"] == 0.02
    assert rows[1]["raw_joint_saturation_frac_late_steps"] == 0.05
    assert rows[1]["raw_gripper_saturation_frac"] == 0.5
    assert rows[1]["joint_clamp_delta_mean"] == 0.03
    assert rows[1]["joint_clamp_delta_mean_left"] == 0.04
    assert rows[1]["joint_clamp_delta_mean_right"] == 0.02
    assert rows[1]["joint_clamp_delta_max"] == 0.12


def test_summarize_sweep_tracks_video_pred_rollout_mode(tmp_path):
    action = tmp_path / "causal_flowmatch_resetcache_vpred_action_rp12_jscale_1p0"
    noncausal = tmp_path / "causal_flowmatch_resetcache_vpred_noncausal_rp12_jscale_1p0"
    for path, mode in ((action, "action"), (noncausal, "noncausal")):
        _write_json(
            path / "results.json",
            {
                "success_rate": 0.0,
                "results": [{"seed": 1000, "success": False}],
                "server": {"meta": {"video_pred_rollout_mode": mode}},
                "eval_config": {
                    "replan_every": 12,
                    "joint_target_scale": 1.0,
                },
            },
        )

    rows = summarize_sweep(str(tmp_path), pattern="*_jscale_*")

    assert {row["setting_dir"]: row["video_pred_rollout_mode"] for row in rows} == {
        action.name: "action",
        noncausal.name: "noncausal",
    }
