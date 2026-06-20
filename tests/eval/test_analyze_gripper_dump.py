import numpy as np

from scripts.eval.analyze_gripper_dump import analyze_episode


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
    pred_chunk = np.zeros((2, 3, 16), dtype=np.float32)
    obs_qpos = np.zeros((2, 16), dtype=np.float32)
    np.savez(
        path,
        seed=np.asarray(1000),
        success=np.asarray(False),
        exec_action=exec_action,
        pred_chunk=pred_chunk,
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
    assert episode["max_joint_step_accel"] > 0.0
    assert np.isclose(episode["max_replan_boundary_joint_jump"], 0.6)
    assert "exec_joint_step_accel" in episode["joint_debug"]
    assert "replan_boundary_joint_jump" in episode["joint_debug"]
