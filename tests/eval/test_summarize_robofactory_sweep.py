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
                "joint_target_scale": 1.5,
                "joint_target_scale_reference": "auto",
                "joint_target_scale_clip": 0.25,
            },
        },
    )
    _write_json(
        scale_2 / "action_dump_summary.json",
        {
            "summary": {
                "all_grippers_decisive_close": 1,
                "any_gripper_never_closes": 0,
                "mean_joint_step_delta": 0.04,
                "max_joint_step_delta": 0.2,
                "mean_joint_step_accel": 0.03,
                "max_joint_step_accel": 0.12,
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
            }
        },
    )
    _write_json(
        scale_1 / "results.json",
        {
            "success_rate": 0.0,
            "results": [{"seed": 1000, "success": False}],
            "eval_config": {
                "joint_delta_scale": 1.0,
                "joint_delta_scale_reference": "auto",
                "joint_delta_scale_clip": 0.25,
            },
        },
    )
    _write_json(
        scale_1 / "action_dump_summary.json",
        {
            "summary": {
                "all_grippers_decisive_close": 1,
                "any_gripper_never_closes": 0,
                "mean_joint_step_delta": 0.02,
                "max_joint_step_delta": 0.1,
                "mean_joint_step_accel": 0.01,
                "max_joint_step_accel": 0.04,
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
            }
        },
    )

    rows = summarize_sweep(str(tmp_path))

    assert [row["scale"] for row in rows] == [1.0, 1.5]
    assert rows[0]["success_count"] == 0
    assert rows[0]["target_min_mean_left"] == 0.13
    assert rows[0]["right_grasp_episodes"] == 0
    assert rows[1]["success_count"] == 1
    assert rows[1]["scale_clip"] == 0.25
    assert rows[1]["target_min_mean_right"] == 0.08
    assert rows[1]["barrier_margin_best"] == -0.02
    assert rows[1]["mean_joint_step_accel"] == 0.03
    assert rows[1]["max_replan_boundary_joint_jump"] == 0.09
