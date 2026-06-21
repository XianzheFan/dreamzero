import json
from pathlib import Path

from scripts.eval.summarize_robofactory_checkpoint_grid import (
    find_eval_roots,
    summarize_roots,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_eval(
    root: Path,
    step: int,
    *,
    success_count: int,
    raw_sat: float,
    future_delta: float,
) -> None:
    _write_json(
        root / "checkpoint_eval_manifest.json",
        {
            "ckpt_setting": f"checkpoint-{step}",
            "eval_status": "0",
            "run_name": f"eval-c{step}",
            "dreamzero_git_commit": "abc123",
            "checkpoint_code_commit": "trainabc",
            "checkpoint_expected_code_commit": "trainabc",
            "checkpoint_stage_label": "droidwidth-teacher-style",
            "checkpoint_action_dim": 32,
            "checkpoint_diffusion_action_dim": 32,
            "checkpoint_num_agents": 2,
            "checkpoint_agent_dim": "gamma",
            "checkpoint_global_video_attention_mode": "bidirectional",
            "checkpoint_global_video_timestep_mode": "clean",
            "checkpoint_use_sparse_hub_attention": False,
            "checkpoint_train_architecture": "lora",
        },
    )
    _write_json(
        root / "scale_sweep_summary.json",
        {
            "rows": [
                {
                    "setting_dir": "rp24_bad",
                    "replan_every": 24,
                    "scale": 1.0,
                    "replan_boundary_blend_steps": 0,
                    "temporal_action_ensemble_decay": 0.0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "target_min_mean_left": 0.16,
                    "target_min_mean_right": 0.15,
                    "barrier_margin_best": -0.15,
                    "raw_joint_saturation_frac": 0.8,
                    "max_replan_boundary_joint_jump": 0.12,
                },
                {
                    "setting_dir": "rp12_best",
                    "replan_every": 12,
                    "scale": 1.0,
                    "target_slew_rate": 0.35,
                    "replan_boundary_blend_steps": 4,
                    "temporal_action_ensemble_decay": 0.6,
                    "success_count": success_count,
                    "success_rate": float(success_count),
                    "target_min_mean_left": 0.08,
                    "target_min_mean_right": 0.09,
                    "barrier_margin_best": -0.02,
                    "left_grasp_episodes": success_count,
                    "right_grasp_episodes": success_count,
                    "mean_joint_step_delta": 0.03,
                    "max_joint_step_delta": 0.14,
                    "mean_joint_step_accel": 0.02,
                    "mean_joint_accel_to_delta_ratio": 0.67,
                    "max_joint_accel_to_delta_ratio": 0.5,
                    "joint_delta_sign_flip_frac": 0.25,
                    "mean_joint_delta_sign_flip_frac": 0.2,
                    "max_replan_boundary_joint_jump": 0.06,
                    "raw_joint_saturation_frac": raw_sat,
                    "raw_joint_saturation_frac_left": raw_sat + 0.01,
                    "raw_joint_saturation_frac_right": raw_sat - 0.01,
                    "raw_joint_saturation_frac_first_step": raw_sat + 0.02,
                    "raw_joint_saturation_frac_late_steps": raw_sat - 0.02,
                    "raw_gripper_saturation_frac": 0.7,
                    "joint_clamp_delta_mean": 0.05,
                    "joint_clamp_delta_max": 0.2,
                },
            ]
        },
    )
    _write_json(
        root / "video_pred_quality.json",
        {
            "summary": {
                "video_pred_rollout_mode_counts": {"action": 8},
                "video_pred_wrist_window_mode_counts": {"action": 8},
                "pred_vs_future_mae_rgb_mean": 20.0,
                "pred_vs_future_first_frame_mae_rgb_mean": 12.0,
                "pred_vs_future_last_frame_mae_rgb_mean": 12.0 + future_delta,
                "pred_vs_future_mae_rgb_first_to_last_delta_mean": future_delta,
                "pred_vs_future_best_alignment_offset_counts": {"1": 2},
                "pred_vs_future_best_alignment_offset_abs_mean": 1.0,
                "pred_vs_future_best_alignment_mae_rgb_mean": 10.0,
                "pred_vs_future_best_alignment_improvement_rgb_mean": 4.0,
                "temporal_absdiff_mean": 18.0,
                "temporal_freeze_frac_mean": 0.0,
                "laplacian_var_mean": 40.0,
            }
        },
    )


def test_find_eval_roots_discovers_nested_artifacts(tmp_path):
    eval_2000 = tmp_path / "runs" / "c2000" / "eval_outputs"
    eval_4000 = tmp_path / "runs" / "c4000" / "eval_outputs"
    _make_eval(eval_2000, 2000, success_count=0, raw_sat=0.7, future_delta=8.0)
    _make_eval(eval_4000, 4000, success_count=1, raw_sat=0.2, future_delta=3.0)

    roots = find_eval_roots([tmp_path])

    assert roots == [eval_2000.resolve(), eval_4000.resolve()]


def test_summarize_roots_orders_by_checkpoint_and_selects_best_setting(tmp_path):
    eval_2000 = tmp_path / "c2000" / "eval_outputs"
    eval_4000 = tmp_path / "c4000" / "eval_outputs"
    _make_eval(eval_4000, 4000, success_count=1, raw_sat=0.2, future_delta=3.0)
    _make_eval(eval_2000, 2000, success_count=0, raw_sat=0.7, future_delta=8.0)

    rows = summarize_roots([tmp_path])

    assert [row["checkpoint_step"] for row in rows] == [2000, 4000]
    assert rows[0]["ckpt_setting"] == "checkpoint-2000"
    assert rows[0]["checkpoint_code_commit"] == "trainabc"
    assert rows[0]["checkpoint_stage_label"] == "droidwidth-teacher-style"
    assert rows[0]["checkpoint_action_dim"] == 32
    assert rows[0]["checkpoint_global_video_attention_mode"] == "bidirectional"
    assert rows[0]["best_setting_dir"] == "rp12_best"
    assert rows[0]["raw_joint_saturation_frac"] == 0.7
    assert rows[0]["mean_joint_accel_to_delta_ratio"] == 0.67
    assert rows[0]["max_joint_accel_to_delta_ratio"] == 0.5
    assert rows[0]["joint_delta_sign_flip_frac"] == 0.25
    assert rows[0]["mean_joint_delta_sign_flip_frac"] == 0.2
    assert rows[0]["pred_vs_future_mae_rgb_first_to_last_delta_mean"] == 8.0
    assert rows[0]["pred_vs_future_best_alignment_offset_counts"] == {"1": 2}
    assert rows[0]["pred_vs_future_best_alignment_mae_rgb_mean"] == 10.0
    assert rows[0]["pred_vs_future_best_alignment_improvement_rgb_mean"] == 4.0
    assert rows[1]["success_count"] == 1
    assert rows[1]["raw_joint_saturation_frac_left"] == 0.21000000000000002
    assert rows[1]["pred_vs_future_mae_rgb_first_to_last_delta_mean"] == 3.0
