import numpy as np
import pytest
from types import SimpleNamespace


def _load_server_module():
    return pytest.importorskip("eval_utils.bimanual_policy_server")


def _stats(q01, q99):
    return {"q01": list(q01), "q99": list(q99)}


def _make_policy(metadata):
    mod = _load_server_module()
    policy = mod.BimanualPolicy.__new__(mod.BimanualPolicy)
    policy.image_h = 240
    policy.image_w = 320
    policy.model_image_h = None
    policy.model_image_w = None
    policy.action_horizon = 2
    policy.action_dim = 16
    policy.num_frames = 3
    policy._metadata = metadata
    policy.gripper_convention = "auto"
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


class _FakeDist:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent = []

    def broadcast_object_list(self, obj_list, src=0):
        assert src == 0
        if obj_list[0] is None:
            if not self.incoming:
                raise AssertionError("no fake command available")
            obj_list[0] = self.incoming.pop(0)
        else:
            self.sent.append(obj_list[0])


class _ProxyPolicy:
    def __init__(self):
        self.calls = []

    def reset(self, payload):
        self.calls.append(("reset", payload))
        return "reset successful"

    def infer(self, payload):
        self.calls.append(("infer", payload))
        return {"action_chunk": payload["qpos"]}


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


def test_resolve_inference_parallel_size_rejects_unsupported_world_size():
    mod = _load_server_module()

    assert mod._resolve_inference_parallel_size(0, 1) == 1
    assert mod._resolve_inference_parallel_size(0, 2) == 2
    assert mod._resolve_inference_parallel_size(2, 2) == 2

    with pytest.raises(ValueError, match="supports inference_parallel_size 1 or 2"):
        mod._resolve_inference_parallel_size(0, 8)
    with pytest.raises(ValueError, match="Distributed serving launch mismatch"):
        mod._resolve_inference_parallel_size(2, 1)


def test_distributed_policy_proxy_broadcasts_leader_commands():
    mod = _load_server_module()
    policy = _ProxyPolicy()
    fake_dist = _FakeDist()
    proxy = mod.DistributedPolicyProxy(
        policy,
        rank=0,
        world_size=2,
        dist_module=fake_dist,
    )

    assert proxy.reset({"session_id": "abc"}) == "reset successful"
    assert proxy.infer({"qpos": [1, 2, 3]}) == {"action_chunk": [1, 2, 3]}

    assert fake_dist.sent == [
        {"endpoint": "reset", "payload": {"session_id": "abc"}},
        {"endpoint": "infer", "payload": {"qpos": [1, 2, 3]}},
    ]
    assert policy.calls == [
        ("reset", {"session_id": "abc"}),
        ("infer", {"qpos": [1, 2, 3]}),
    ]


def test_distributed_policy_proxy_worker_executes_received_commands():
    mod = _load_server_module()
    policy = _ProxyPolicy()
    fake_dist = _FakeDist(
        [
            {"endpoint": "reset", "payload": {"session_id": "abc"}},
            {"endpoint": "infer", "payload": {"qpos": [4, 5, 6]}},
            {"endpoint": "shutdown", "payload": {}},
        ]
    )
    proxy = mod.DistributedPolicyProxy(
        policy,
        rank=1,
        world_size=2,
        dist_module=fake_dist,
    )

    proxy.worker_loop()

    assert policy.calls == [
        ("reset", {"session_id": "abc"}),
        ("infer", {"qpos": [4, 5, 6]}),
    ]


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


def test_denorm_action_accepts_droid_width_per_agent_output():
    policy = _make_policy(_metadata_with_action_stats())
    pred = np.zeros((1, 2, 2, 32), dtype=np.float32)
    pred[0, 0, :, :7] = 1.0
    pred[0, 0, :, 7] = -1.0
    pred[0, 1, :, :7] = -1.0
    pred[0, 1, :, 7] = 1.0
    pred[..., 8:] = 99.0
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
    assert policy._last_action_debug["action_norm_raw"].shape == (2, 16)
    assert policy._last_action_debug["action_norm_raw_model_full"].shape == (2, 64)


def test_denorm_action_keeps_constant_gripper_stats_constant():
    metadata = _metadata_with_action_stats()
    action_stats = metadata["robofactory"]["statistics"]["action"]
    action_stats["panda0_gripper_pos"] = _stats([1.0], [1.0])
    action_stats["panda1_gripper_pos"] = _stats([-1.0], [-1.0])
    policy = _make_policy(metadata)
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, 7] = np.array([-1.0, 1.0])
    pred[0, 1, :, 7] = np.array([1.0, -1.0])

    out = policy._denorm_action({"action_pred": pred}, np.zeros(16, dtype=np.float32))

    np.testing.assert_allclose(out[:, 7], 1.0)
    np.testing.assert_allclose(out[:, 15], -1.0)
    np.testing.assert_allclose(
        policy._last_action_debug["action_physical_pre_binarize"][:, [7, 15]],
        np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32),
    )


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


def test_relative_action_keys_accept_state_prefix():
    policy = _make_policy(_metadata_with_action_stats())
    policy._relative_action_keys = {
        "state.panda0_joint_pos",
        "state.panda1_joint_pos",
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
    np.testing.assert_allclose(out[:, [7, 15]], np.full((2, 2), 0.5))


def test_denorm_action_does_not_add_reference_when_relative_is_disabled():
    policy = _make_policy(_metadata_with_action_stats())
    policy._relative_action = False
    policy._relative_action_per_horizon = False
    policy._relative_action_keys = {"state.panda0_joint_pos", "state.panda1_joint_pos"}
    pred = np.zeros((1, 2, 2, 8), dtype=np.float32)
    pred[0, 0, :, :7] = 1.0
    pred[0, 1, :, :7] = -1.0
    qpos = np.arange(16, dtype=np.float32)

    out = policy._denorm_action({"action_pred": pred}, qpos)

    np.testing.assert_allclose(out[:, :7], np.full((2, 7), 0.1))
    np.testing.assert_allclose(out[:, 8:15], np.zeros((2, 7)))
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


def test_robotwin_auto_gripper_convention_uses_zero_close():
    policy = _make_policy(_metadata_with_action_stats("robotwin"))
    policy.gripper_close_value = None

    assert policy._resolved_gripper_convention() == "robotwin"
    assert policy._gripper_close_target() == 0.0
    assert policy._gripper_open_target() == 1.0


def test_robofactory_auto_gripper_convention_uses_negative_close_for_override():
    policy = _make_policy(_metadata_with_action_stats("robofactory"))
    policy.gripper_close_value = None
    policy.gripper_override = "close-all"
    action = np.ones((2, 16), dtype=np.float32)

    policy._apply_gripper_override({"infer_idx": 0}, action)

    assert policy._resolved_gripper_convention() == "robofactory"
    assert policy._gripper_close_target() == -1.0
    np.testing.assert_allclose(action[:, [7, 15]], -np.ones((2, 2)))


def test_robofactory_auto_gripper_binarize_outputs_negative_positive_commands():
    policy = _make_policy(_metadata_with_action_stats("robofactory"))
    policy.gripper_close_value = None
    policy.gripper_binarize_threshold = 0.0
    action = np.zeros((2, 16), dtype=np.float32)
    action[:, 7] = [-0.25, 0.25]
    action[:, 15] = [0.25, -0.25]

    policy._binarize_gripper_targets(action)

    np.testing.assert_allclose(action[:, 7], [-1.0, 1.0])
    np.testing.assert_allclose(action[:, 15], [1.0, -1.0])


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


def test_runtime_action_shapes_follow_checkpoint_config(caplog):
    policy = _make_policy(_metadata_with_action_stats())
    policy.action_horizon = 8
    policy.num_frames = 9
    policy._cfg = {"action_horizon": 24, "num_frames": 33}

    with caplog.at_level("WARNING"):
        policy._sync_runtime_shape_from_config()

    assert policy.action_horizon == 24
    assert policy.num_frames == 33
    assert "Overriding eval action_horizon=8" in caplog.text
    assert "Overriding eval num_frames=9" in caplog.text


def test_default_eval_resolution_keeps_checkpoint_config():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    policy = _make_policy(_metadata_with_action_stats())
    policy._cfg = OmegaConf.create(
        {
            "image_resolution_height": 176,
            "image_resolution_width": 320,
            "action_head_cfg": {"config": {"target_video_height": None, "target_video_width": None}},
        }
    )

    policy._apply_model_resolution_overrides()

    assert policy._cfg.image_resolution_height == 176
    assert policy._cfg.image_resolution_width == 320
    assert policy._cfg.action_head_cfg.config.target_video_height is None
    assert policy._cfg.action_head_cfg.config.target_video_width is None


def test_model_resolution_override_updates_config_and_resize_transform(caplog):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    policy = _make_policy(_metadata_with_action_stats())
    policy.model_image_h = 160
    policy.model_image_w = 320
    policy._cfg = OmegaConf.create(
        {
            "image_resolution_height": 176,
            "image_resolution_width": 320,
            "action_head_cfg": {"config": {"target_video_height": 176, "target_video_width": 320}},
            "model": {
                "config": {
                    "action_head_cfg": {
                        "config": {
                            "target_video_height": None,
                            "target_video_width": None,
                        }
                    }
                }
            },
        }
    )

    with caplog.at_level("WARNING"):
        policy._apply_model_resolution_overrides()

    assert policy._cfg.image_resolution_height == 160
    assert policy._cfg.image_resolution_width == 320
    assert policy._cfg.action_head_cfg.config.target_video_height == 160
    assert policy._cfg.action_head_cfg.config.target_video_width == 320
    assert policy._cfg.model.config.action_head_cfg.config.target_video_height is None
    assert "Applying model-side eval resize override HxW=160x320" in caplog.text

    resize = type("VideoResize", (), {"height": 176, "width": 320, "transforms": []})()

    with caplog.at_level("WARNING"):
        policy._set_transform_resize_resolution(SimpleNamespace(transforms=[resize]))

    assert resize.height == 160
    assert resize.width == 320
    assert "Overriding 1 VideoResize transform(s) to HxW=160x320" in caplog.text


def test_eval_diffusion_structure_override_updates_all_config_copies(monkeypatch, caplog):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    policy = _make_policy(_metadata_with_action_stats())
    policy._cfg = OmegaConf.create(
        {
            "action_head_cfg": {
                "config": {
                    "diffusion_model_cfg": {
                        "in_dim": 16,
                        "concat_first_frame_latent": False,
                    }
                }
            },
            "model": {
                "config": {
                    "action_head_cfg": {
                        "config": {
                            "diffusion_model_cfg": {
                                "in_dim": 16,
                                "concat_first_frame_latent": False,
                            }
                        }
                    }
                }
            },
        }
    )
    monkeypatch.setenv("DREAMZERO_EVAL_DIFFUSION_IN_DIM", "36")
    monkeypatch.setenv("DREAMZERO_EVAL_CONCAT_FIRST_FRAME_LATENT", "true")
    monkeypatch.delenv("DREAMZERO_DISABLE_MULTI_AGENT_HUB", raising=False)

    with caplog.at_level("WARNING"):
        policy._apply_eval_config_overrides()

    assert policy._cfg.action_head_cfg.config.diffusion_model_cfg.in_dim == 36
    assert (
        policy._cfg.action_head_cfg.config.diffusion_model_cfg.concat_first_frame_latent
        is True
    )
    embedded = policy._cfg.model.config.action_head_cfg.config.diffusion_model_cfg
    assert embedded.in_dim == 36
    assert embedded.concat_first_frame_latent is True
    assert "DreamZero eval diffusion-structure override" in caplog.text


def test_shape_mismatch_filter_can_drop_finetune_patch_embedding():
    torch = pytest.importorskip("torch")
    mod = _load_server_module()
    key = "action_head.model.patch_embedding.weight"
    ckpt = torch.ones((4, 16, 1, 2, 2))
    model = torch.zeros((4, 36, 1, 2, 2))

    kept, sliced, dropped = mod._filter_shape_mismatches_for_load(
        {key: ckpt},
        {key: model},
    )

    assert key in kept
    assert sliced == {key: ((4, 16, 1, 2, 2), (4, 36, 1, 2, 2))}
    assert dropped == {}
    assert tuple(kept[key].shape) == tuple(model.shape)
    torch.testing.assert_close(kept[key][:, :16], torch.ones_like(kept[key][:, :16]))
    torch.testing.assert_close(kept[key][:, 16:], torch.zeros_like(kept[key][:, 16:]))

    kept, sliced, dropped = mod._filter_shape_mismatches_for_load(
        {key: ckpt},
        {key: model},
        drop_mismatched_keys=mod.FINETUNE_PATCH_EMBEDDING_KEYS,
    )

    assert kept == {}
    assert sliced == {}
    assert dropped == {key: ((4, 16, 1, 2, 2), (4, 36, 1, 2, 2))}


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
