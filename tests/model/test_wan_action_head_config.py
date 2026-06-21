from pathlib import Path
import types

import pytest


def test_wan_action_head_yaml_parses_and_contains_gripper_clean_defaults():
    yaml = pytest.importorskip("yaml")

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (
        repo_root
        / "groot"
        / "vla"
        / "configs"
        / "model"
        / "dreamzero"
        / "action_head"
        / "wan_flow_matching_action_tf.yaml"
    )

    cfg = yaml.safe_load(cfg_path.read_text())
    head_cfg = cfg["action_head_cfg"]["config"]

    assert head_cfg["dynamics_loss_weight"] == 1.0
    assert head_cfg["gripper_clean_action_loss_weight"] == 0.0
    assert head_cfg["gripper_clean_close_action_loss_weight"] == 1.0
    assert head_cfg["gripper_clean_max_sigma"] == 1.0
    assert head_cfg["gripper_binary_action_loss_weight"] == 0.0
    assert head_cfg["gripper_binary_close_action_loss_weight"] == 1.0
    assert head_cfg["gripper_binary_logit_scale"] == 4.0
    assert head_cfg["gripper_binary_max_sigma"] == 1.0
    assert head_cfg["action_delta_loss_weight"] == 0.0
    assert head_cfg["action_jerk_loss_weight"] == 0.0
    assert head_cfg["action_delta_max_sigma"] == 1.0
    assert head_cfg["action_delta_exclude_gripper"] is True
    assert head_cfg["first_close_joint_loss_weight"] == 1.0
    assert head_cfg["first_close_joint_loss_window_before"] == 0
    assert head_cfg["first_close_joint_loss_window_after"] == 0
    assert head_cfg["joint_prefix_loss_weight"] == 1.0
    assert head_cfg["joint_prefix_loss_len"] == 0
    assert head_cfg["pre_close_joint_loss_weight"] == 1.0
    assert head_cfg["pre_close_joint_loss_window_before"] == 0
    assert head_cfg["open_phase_joint_loss_weight"] == 1.0
    assert head_cfg["multi_agent_shuffle_agents"] is False
    assert head_cfg["multi_agent_sample_agent_pool"] is False
    assert head_cfg["global_video_dropout_prob"] == 0.0
    assert head_cfg["self_forcing_train"] is False
    assert head_cfg["self_forcing_warmup_steps"] == 0
    assert head_cfg["self_forcing_fast_writeback"] is False
    assert "gripper_clean_action_loss_weight" not in cfg
    assert "gripper_binary_action_loss_weight" not in cfg


def test_wan_action_head_config_imports_with_gripper_defaults():
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    pytest.importorskip("hydra")
    pytest.importorskip("peft")

    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
        WANPolicyHeadConfig,
    )

    cfg = WANPolicyHeadConfig()

    assert cfg.dynamics_loss_weight == 1.0
    assert tuple(cfg.gripper_action_dims) == (7,)
    assert cfg.action_delta_loss_weight == 0.0
    assert cfg.action_jerk_loss_weight == 0.0
    assert cfg.action_delta_max_sigma == 1.0
    assert cfg.action_delta_exclude_gripper is True
    assert cfg.first_close_joint_loss_weight == 1.0
    assert cfg.first_close_joint_loss_window_before == 0
    assert cfg.first_close_joint_loss_window_after == 0
    assert cfg.joint_prefix_loss_weight == 1.0
    assert cfg.joint_prefix_loss_len == 0
    assert cfg.pre_close_joint_loss_weight == 1.0
    assert cfg.pre_close_joint_loss_window_before == 0
    assert cfg.open_phase_joint_loss_weight == 1.0
    assert cfg.multi_agent_shuffle_agents is False
    assert cfg.multi_agent_sample_agent_pool is False
    assert cfg.global_video_dropout_prob == 0.0
    assert cfg.self_forcing_train is False
    assert cfg.self_forcing_warmup_steps == 0
    assert cfg.self_forcing_fast_writeback is False


def test_self_forcing_train_enabled_respects_warmup_steps(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    pytest.importorskip("hydra")
    pytest.importorskip("peft")

    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
        WANPolicyHead,
    )

    monkeypatch.delenv("MAI_SELF_FORCING_TRAIN", raising=False)
    head = WANPolicyHead.__new__(WANPolicyHead)
    head.config = types.SimpleNamespace(
        self_forcing_train=True,
        self_forcing_warmup_steps=5,
    )

    head.global_step = 4
    assert head._self_forcing_train_enabled() is False

    head.global_step = 5
    assert head._self_forcing_train_enabled() is True

    monkeypatch.setenv("MAI_SELF_FORCING_TRAIN", "0")
    assert head._self_forcing_train_enabled() is False

    monkeypatch.setenv("MAI_SELF_FORCING_TRAIN", "1")
    head.global_step = 4
    assert head._self_forcing_train_enabled() is False
