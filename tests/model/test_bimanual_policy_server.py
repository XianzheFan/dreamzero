import numpy as np
import pytest


def _load_server_module():
    return pytest.importorskip("eval_utils.bimanual_policy_server")


def _stats(q01, q99):
    return {"q01": list(q01), "q99": list(q99)}


def _make_policy(metadata):
    mod = _load_server_module()
    policy = mod.BimanualPolicy.__new__(mod.BimanualPolicy)
    policy.action_horizon = 2
    policy.action_dim = 16
    policy.num_frames = 3
    policy._metadata = metadata
    policy._relative_action = True
    policy._relative_action_per_horizon = False
    policy._relative_action_keys = {"panda0_joint_pos", "panda1_joint_pos"}
    policy._last_action_debug = {}
    policy._sessions = {}
    policy.prompt_override = ""
    policy.gripper_binarize_threshold = None
    return policy


def _metadata_with_action_stats():
    return {
        "robofactory": {
            "statistics": {
                "action": {
                    "panda0_joint_pos": _stats(np.zeros(7), np.full(7, 0.1)),
                    "panda0_gripper_pos": _stats([0.0], [1.0]),
                    "panda1_joint_pos": _stats(np.zeros(7), np.full(7, 0.2)),
                    "panda1_gripper_pos": _stats([0.0], [1.0]),
                }
            }
        }
    }


def test_denorm_action_uses_metadata_and_adds_relative_joint_reference():
    policy = _make_policy(_metadata_with_action_stats())
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, :7] = 1.0
    pred[0, 0, :, 7] = -1.0
    pred[0, 1, :, :7] = -1.0
    pred[0, 1, :, 7] = 1.0
    qpos = np.arange(16, dtype=np.float32)

    out = policy._denorm_action({"action_pred": pred}, qpos)

    assert out.shape == (2, 16)
    np.testing.assert_allclose(
        out[:, :7],
        np.repeat((qpos[:7] + 0.1)[None], out.shape[0], axis=0),
    )
    np.testing.assert_allclose(out[:, 7], 0.0)
    np.testing.assert_allclose(
        out[:, 8:15],
        np.repeat(qpos[8:15][None], out.shape[0], axis=0),
    )
    np.testing.assert_allclose(out[:, 15], 1.0)
    assert "action_norm_raw" in policy._last_action_debug
    assert "action_norm_clipped" in policy._last_action_debug


def test_denorm_action_accepts_full_action_prefixed_metadata_keys():
    metadata = _metadata_with_action_stats()
    action_stats = metadata["robofactory"]["statistics"]["action"]
    metadata["robofactory"]["statistics"]["action"] = {
        f"action.{key}": value for key, value in action_stats.items()
    }
    policy = _make_policy(metadata)
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)

    out = policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))

    np.testing.assert_allclose(out[:, :7], np.full((2, 7), 0.05))
    np.testing.assert_allclose(out[:, 8:15], np.full((2, 7), 0.1))
    np.testing.assert_allclose(out[:, [7, 15]], np.full((2, 2), 0.5))


def test_denorm_action_uses_prefixed_negative_gripper_stats():
    metadata = _metadata_with_action_stats()
    action_stats = metadata["robofactory"]["statistics"]["action"]
    action_stats["panda0_gripper_pos"] = _stats([-1.0], [1.0])
    action_stats["panda1_gripper_pos"] = _stats([-1.0], [1.0])
    metadata["robofactory"]["statistics"]["action"] = {
        f"action.{key}": value for key, value in action_stats.items()
    }
    policy = _make_policy(metadata)
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, :, :, 7] = -0.5

    out = policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))

    np.testing.assert_allclose(out[:, [7, 15]], np.full((2, 2), -0.5))


def test_denorm_action_raises_when_action_stats_are_missing():
    metadata = _metadata_with_action_stats()
    del metadata["robofactory"]["statistics"]["action"]["panda1_gripper_pos"]
    policy = _make_policy(metadata)
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)

    with pytest.raises(KeyError, match="Missing action normalization stats"):
        policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))


def test_prompt_override_wins_over_client_prompt():
    policy = _make_policy(_metadata_with_action_stats())
    policy.prompt_override = "stack the two blocks"

    assert policy._effective_prompt("Move red block and green block") == "stack the two blocks"
    assert policy.reset({"session_id": "episode-0", "prompt": "client prompt"}) == "reset successful"
    assert policy._sessions["episode-0"]["prompt"] == "stack the two blocks"
