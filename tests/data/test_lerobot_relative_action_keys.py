import pytest


def _load_helpers():
    for dep in ("numpy", "pandas", "pydantic", "torch", "yaml"):
        pytest.importorskip(dep)

    from groot.vla.data.dataset.lerobot import (
        _normalize_relative_action_key,
        _normalize_relative_action_keys,
        _relative_action_key_matches,
    )

    return (
        _normalize_relative_action_key,
        _normalize_relative_action_keys,
        _relative_action_key_matches,
    )


def test_relative_action_key_normalization_accepts_modality_prefixes():
    normalize, normalize_keys, matches = _load_helpers()

    assert normalize("panda0_joint_pos") == "panda0_joint_pos"
    assert normalize("action.panda0_joint_pos") == "panda0_joint_pos"
    assert normalize("state.panda0_joint_pos") == "panda0_joint_pos"
    assert normalize_keys([
        "action.panda0_joint_pos",
        "state.panda1_joint_pos",
    ]) == [
        "panda0_joint_pos",
        "panda1_joint_pos",
    ]

    configured = ["action.panda0_joint_pos", "state.panda1_joint_pos"]
    assert matches("panda0_joint_pos", configured)
    assert matches("action.panda0_joint_pos", configured)
    assert matches("state.panda1_joint_pos", configured)
    assert not matches("panda0_gripper_pos", configured)


def test_relative_action_key_none_matches_all_keys():
    _, _, matches = _load_helpers()

    assert matches("panda0_joint_pos", None)
    assert matches("action.panda0_gripper_pos", None)
