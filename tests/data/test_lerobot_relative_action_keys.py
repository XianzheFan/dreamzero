import ast
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_NAMES = {
    "_normalize_relative_action_key",
    "_normalize_relative_action_keys",
    "_relative_action_key_matches",
}


def _load_helpers():
    module_path = _REPO_ROOT / "groot" / "vla" / "data" / "dataset" / "lerobot.py"
    tree = ast.parse(module_path.read_text())
    helper_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES
    ]
    assert {node.name for node in helper_defs} == _HELPER_NAMES

    namespace = {"Sequence": Sequence}
    helper_module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(helper_module)
    exec(compile(helper_module, str(module_path), "exec"), namespace)
    return (
        namespace["_normalize_relative_action_key"],
        namespace["_normalize_relative_action_keys"],
        namespace["_relative_action_key_matches"],
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
