import numpy as np

from scripts.eval.analyze_gripper_dump import _aggregate, analyze_episode


def test_analyze_episode_reports_joint_jitter_metrics(tmp_path):
    path = tmp_path / "episode_1000.npz"
    exec_action = np.zeros((5, 16), dtype=np.float32)
    exec_action[:, :7] = np.asarray(
        [
            [0.0] * 7,
            [0.1] * 7,
            [0.2] * 7,
            [0.8] * 7,
            [0.9] * 7,
        ],
        dtype=np.float32,
    )
    exec_action[:, 8:15] = exec_action[:, :7]
    exec_action_pre_blend = exec_action.copy()
    exec_action_pre_blend[3, :7] = 1.2
    exec_action_pre_blend[3, 8:15] = 1.2
    exec_action_pre_accel = exec_action.copy()
    exec_action_pre_accel[3, :7] = 1.1
    exec_action_pre_accel[3, 8:15] = 1.1
    exec_action_pre_ensemble = exec_action_pre_blend.copy()
    exec_action_pre_ensemble[3, :7] = 1.5
    exec_action_pre_ensemble[3, 8:15] = 1.5
    exec_action_pre_slew = exec_action.copy()
    exec_action_pre_slew[3, :7] = 1.0
    exec_action_pre_slew[3, 8:15] = 1.0
    pred_chunk = np.zeros((2, 3, 16), dtype=np.float32)
    pred_chunk[0, :, :7] = np.asarray([0.0, 0.2, 0.0], dtype=np.float32)[:, None]
    pred_chunk[0, :, 8:15] = pred_chunk[0, :, :7]
    pred_chunk[1, :, :7] = np.asarray([0.4, 0.1, 0.5], dtype=np.float32)[:, None]
    pred_chunk[1, :, 8:15] = pred_chunk[1, :, :7]
    obs_qpos = np.zeros((2, 16), dtype=np.float32)
    action_norm_raw = np.zeros((2, 3, 16), dtype=np.float32)
    action_norm_raw[:, :, 0] = 1.5
    action_norm_raw[:, 1:, 8] = -2.0
    action_norm_clipped = np.clip(action_norm_raw, -1.0, 1.0)
    np.savez(
        path,
        seed=np.asarray(1000),
        success=np.asarray(False),
        action_representation=np.asarray("absolute_qpos", dtype="<U32"),
        exec_action=exec_action,
        exec_action_pre_blend=exec_action_pre_blend,
        exec_action_pre_accel=exec_action_pre_accel,
        exec_action_pre_ensemble=exec_action_pre_ensemble,
        exec_action_pre_slew=exec_action_pre_slew,
        pred_chunk=pred_chunk,
        action_norm_raw=action_norm_raw,
        action_norm_clipped=action_norm_clipped,
        infer_step=np.asarray([0, 3], dtype=np.int64),
        obs_qpos=obs_qpos,
    )

    episode = analyze_episode(
        str(path),
        close_threshold=0.0,
        decisive_threshold=-0.5,
        print_profiles=False,
        num_arms=2,
        arm_dim=8,
        gripper_offset=7,
        explicit_gripper_dims=None,
        custom_arm_labels=None,
    )

    assert episode["mean_joint_step_delta"] > 0.0
    assert episode["action_representation"] == "absolute_qpos"
    assert episode["max_joint_step_accel"] > 0.0
    assert episode["mean_joint_accel_to_delta_ratio"] > 0.0
    assert episode["max_joint_accel_to_delta_ratio"] > 0.0
    assert episode["joint_delta_sign_flip_frac"] == 0.0
    assert episode["joint_delta_sign_flip_count"] == 0
    assert episode["joint_delta_active_pair_count"] > 0
    assert np.isclose(episode["max_replan_boundary_joint_jump"], 0.6)
    assert np.isclose(episode["max_model_replan_boundary_joint_jump"], 0.2)
    assert episode["mean_pred_chunk_joint_step_delta"] > 0.0
    assert episode["max_pred_chunk_joint_step_accel"] > 0.0
    assert episode["mean_pred_chunk_joint_accel_to_delta_ratio"] > 0.0
    assert episode["pred_chunk_joint_delta_sign_flip_frac"] == 1.0
    assert episode["pred_chunk_joint_delta_sign_flip_count"] == 28
    assert episode["pred_chunk_joint_delta_active_pair_count"] == 28
    assert "exec_joint_step_accel" in episode["joint_debug"]
    assert "replan_boundary_joint_jump" in episode["joint_debug"]
    assert "model_replan_boundary_joint_jump" in episode["joint_debug"]
    assert "pred_chunk_joint_step_delta" in episode["joint_debug"]
    assert "pred_chunk_joint_step_accel" in episode["joint_debug"]
    assert "pre_blend_replan_boundary_joint_jump" in episode["joint_debug"]
    assert "pre_ensemble_replan_boundary_joint_jump" in episode["joint_debug"]
    assert "temporal_ensemble_correction_joint" in episode["joint_debug"]
    assert np.isclose(
        episode["joint_debug"]["temporal_ensemble_correction_joint"]["max_abs"],
        0.3,
    )
    assert "accel_limiter_correction_joint" in episode["joint_debug"]
    assert np.isclose(
        episode["joint_debug"]["accel_limiter_correction_joint"]["max_abs"],
        0.1,
    )
    assert "pre_slew_joint_step_delta" in episode["joint_debug"]
    assert "boundary_blend_correction_joint" in episode["joint_debug"]
    assert "slew_correction_joint" in episode["joint_debug"]
    assert episode["norm_debug"] is not None
    norm_debug = episode["norm_debug"]
    assert len(norm_debug["raw_joint_saturation_frac_per_dim"]) == 14
    assert norm_debug["raw_joint_pos_saturation_frac_per_dim"][0] == 1.0
    assert np.isclose(
        norm_debug["raw_joint_neg_saturation_frac_per_dim"][7],
        2 / 3,
        atol=1e-4,
    )
    assert np.isclose(
        norm_debug["raw_joint_saturation_frac_by_arm"]["left"],
        1 / 7,
    )
    assert np.isclose(
        norm_debug["raw_joint_saturation_frac_by_arm"]["right"],
        2 / 21,
    )
    assert np.isclose(
        norm_debug["raw_joint_saturation_frac_first_step"],
        1 / 14,
    )
    assert np.isclose(
        norm_debug["raw_joint_saturation_frac_late_steps"],
        1 / 7,
    )
    assert norm_debug["joint_clamp_delta_mean_by_arm"]["left"] > 0.0
    assert norm_debug["joint_clamp_delta_mean_by_arm"]["right"] > 0.0
    assert norm_debug["raw_joint_top_saturated_dims"][0]["dim"] == 0
    assert norm_debug["raw_joint_top_saturated_dims"][1]["dim"] == 8

    summary = _aggregate([episode])
    assert summary["action_representation_counts"] == {"absolute_qpos": 1}


def test_analyze_episode_reports_joint_delta_sign_flips(tmp_path):
    path = tmp_path / "episode_1001.npz"
    exec_action = np.zeros((5, 16), dtype=np.float32)
    oscillating = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    exec_action[:, :7] = oscillating[:, None]
    exec_action[:, 8:15] = oscillating[:, None]
    pred_chunk = np.zeros((1, 3, 16), dtype=np.float32)
    np.savez(
        path,
        seed=np.asarray(1001),
        success=np.asarray(False),
        exec_action=exec_action,
        pred_chunk=pred_chunk,
    )

    episode = analyze_episode(
        str(path),
        close_threshold=0.0,
        decisive_threshold=-0.5,
        print_profiles=False,
        num_arms=2,
        arm_dim=8,
        gripper_offset=7,
        explicit_gripper_dims=None,
        custom_arm_labels=None,
    )

    assert episode["joint_delta_active_pair_count"] == 42
    assert episode["joint_delta_sign_flip_count"] == 42
    assert episode["joint_delta_sign_flip_frac"] == 1.0

    summary = _aggregate([episode])
    assert summary["joint_delta_active_pair_count"] == 42
    assert summary["joint_delta_sign_flip_count"] == 42
    assert summary["joint_delta_sign_flip_frac"] == 1.0
