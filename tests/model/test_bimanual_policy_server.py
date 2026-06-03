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
    policy.gripper_override = "none"
    policy.gripper_close_after_infer = 0
    policy.gripper_close_value = 0.0
    policy.gripper_force_open_until_infer = 0
    return policy


class _DummyTransform:
    def __init__(self, training=True, transforms=None):
        self.training = training
        self.transforms = transforms or []


class _DummyDreamTransform(_DummyTransform):
    def _prepare_action(self):
        raise NotImplementedError

    def _prepare_state(self):
        raise NotImplementedError

    def _prepare_video(self):
        raise NotImplementedError


def _metadata_with_action_stats(tag="robofactory"):
    return {
        tag: {
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


def test_metadata_tag_prefers_robotwin_over_legacy_robofactory():
    metadata = {
        **_metadata_with_action_stats("robofactory"),
        **_metadata_with_action_stats("robotwin"),
    }
    metadata["robotwin"]["statistics"]["action"]["panda0_joint_pos"] = _stats(
        np.zeros(7),
        np.full(7, 0.3),
    )
    policy = _make_policy(metadata)
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, :7] = 1.0

    out = policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))

    assert policy._metadata_tag() == "robotwin"
    np.testing.assert_allclose(out[:, :7], np.full((2, 7), 0.3))


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


def test_relative_action_keys_accept_action_prefix():
    policy = _make_policy(_metadata_with_action_stats())
    policy._relative_action_keys = {
        "action.panda0_joint_pos",
        "action.panda1_joint_pos",
    }
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, :7] = 1.0
    pred[0, 1, :, :7] = -1.0
    qpos = np.arange(16, dtype=np.float32)

    out = policy._denorm_action({"action_pred": pred}, qpos)

    np.testing.assert_allclose(
        out[:, :7],
        np.repeat((qpos[:7] + 0.1)[None], out.shape[0], axis=0),
    )
    np.testing.assert_allclose(
        out[:, 8:15],
        np.repeat(qpos[8:15][None], out.shape[0], axis=0),
    )


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


def test_denorm_action_can_binarize_physical_gripper_targets():
    policy = _make_policy(_metadata_with_action_stats())
    policy.gripper_binarize_threshold = 0.5
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, 7] = [-0.25, 0.25]  # physical 0.375, 0.625
    pred[0, 1, :, 7] = [0.25, -0.25]  # physical 0.625, 0.375

    out = policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))

    np.testing.assert_allclose(out[:, 7], [0.0, 1.0])
    np.testing.assert_allclose(out[:, 15], [1.0, 0.0])
    assert "action_physical_pre_binarize" in policy._last_action_debug
    assert "action_physical_final" in policy._last_action_debug


def test_gripper_override_can_force_close_after_infer():
    policy = _make_policy(_metadata_with_action_stats())
    policy.gripper_override = "close-after-infer"
    policy.gripper_close_after_infer = 3
    policy.gripper_close_value = 0.0
    action = np.ones((2, 16), dtype=np.float32)

    policy._apply_gripper_override({"infer_idx": 2}, action)
    np.testing.assert_allclose(action[:, [7, 15]], np.ones((2, 2)))

    policy._apply_gripper_override({"infer_idx": 3}, action)
    np.testing.assert_allclose(action[:, [7, 15]], np.zeros((2, 2)))
    assert "action_physical_after_override" in policy._last_action_debug


def test_gripper_force_open_until_infer_suppresses_initial_close_targets():
    policy = _make_policy(_metadata_with_action_stats())
    policy.gripper_force_open_until_infer = 6
    action = np.zeros((2, 16), dtype=np.float32)

    policy._apply_gripper_force_open({"infer_idx": 5}, action)
    np.testing.assert_allclose(action[:, [7, 15]], np.ones((2, 2)))
    assert "action_physical_after_force_open" in policy._last_action_debug

    action = np.zeros((2, 16), dtype=np.float32)
    policy._apply_gripper_force_open({"infer_idx": 6}, action)
    np.testing.assert_allclose(action[:, [7, 15]], np.zeros((2, 2)))


def test_action_summary_logs_final_and_debug_gripper_values(caplog):
    policy = _make_policy(_metadata_with_action_stats())
    policy.action_representation = "absolute_qpos"
    policy.gripper_binarize_threshold = 0.5
    action = np.zeros((2, 16), dtype=np.float32)
    action[:, 7] = [0.0, 1.0]
    action[:, 15] = [1.0, 0.0]
    policy._last_action_debug = {
        "action_norm_raw": np.zeros((2, 16), dtype=np.float32),
        "action_physical_pre_binarize": np.full((2, 16), 0.6, dtype=np.float32),
    }

    with caplog.at_level("INFO"):
        policy._log_action_summary(
            {"prompt": "stack the two blocks", "infer_idx": 3},
            "episode-0",
            action,
        )

    assert "Action summary session=episode-0 infer_idx=3" in caplog.text
    assert "gripper_binarize_threshold=0.5" in caplog.text
    assert "gripper_force_open_until_infer=0" in caplog.text
    assert "gripper_override=none" in caplog.text
    assert "left_gripper[min=0.000 max=1.000 mean=0.500 close_lt_0p5=1/2" in caplog.text
    assert "norm_raw_left_gripper[min=0.000 max=0.000 mean=0.000 close_lt_0p5=2/2" in caplog.text
    assert "physical_pre_binarize_left_gripper[min=0.600 max=0.600 mean=0.600 close_lt_0p5=0/2" in caplog.text


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


def test_inference_transform_modes_only_enable_dream_action_path():
    policy = _make_policy(_metadata_with_action_stats())
    video_transform = _DummyTransform(training=True)
    state_transform = _DummyTransform(training=True)
    dream_transform = _DummyDreamTransform(training=False)
    policy._transform = _DummyTransform(
        training=True,
        transforms=[video_transform, state_transform, dream_transform],
    )

    saved = policy._set_eval_inference_transform_modes()

    assert policy._transform.training is False
    assert video_transform.training is False
    assert state_transform.training is False
    assert dream_transform.training is True

    policy._restore_transform_modes(saved)

    assert policy._transform.training is True
    assert video_transform.training is True
    assert state_transform.training is True
    assert dream_transform.training is False


def _constant_rgb(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def test_shared_global_video_window_repeats_current_frame_for_all_streams():
    policy = _make_policy(_metadata_with_action_stats())
    policy.num_frames = 3
    shared_global_transform = _DummyTransform()
    shared_global_transform.global_views = [0]
    policy._transform = _DummyTransform(transforms=[shared_global_transform])
    history = [
        (_constant_rgb(0), _constant_rgb(10), _constant_rgb(20)),
        (_constant_rgb(1), _constant_rgb(11), _constant_rgb(21)),
        (_constant_rgb(2), _constant_rgb(12), _constant_rgb(22)),
    ]

    global_video, agent0_video, agent1_video = policy._build_video_windows(history)

    np.testing.assert_array_equal(global_video[:, 0, 0, 0], [2, 2, 2])
    np.testing.assert_array_equal(agent0_video[:, 0, 0, 0], [12, 12, 12])
    np.testing.assert_array_equal(agent1_video[:, 0, 0, 0], [22, 22, 22])


def test_legacy_video_window_preserves_all_camera_histories():
    policy = _make_policy(_metadata_with_action_stats())
    policy.num_frames = 3
    policy._transform = _DummyTransform()
    history = [
        (_constant_rgb(0), _constant_rgb(10), _constant_rgb(20)),
        (_constant_rgb(1), _constant_rgb(11), _constant_rgb(21)),
        (_constant_rgb(2), _constant_rgb(12), _constant_rgb(22)),
    ]

    global_video, agent0_video, agent1_video = policy._build_video_windows(history)

    np.testing.assert_array_equal(global_video[:, 0, 0, 0], [0, 1, 2])
    np.testing.assert_array_equal(agent0_video[:, 0, 0, 0], [10, 11, 12])
    np.testing.assert_array_equal(agent1_video[:, 0, 0, 0], [20, 21, 22])
