import ast
from pathlib import Path


def test_loss_logger_callback_tracks_auxiliary_loss_keys():
    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "groot/vla/experiment/base.py"
    tree = ast.parse(source_path.read_text())
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LossLoggerCallback"
    )
    on_log = next(
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and node.name == "on_log"
    )
    logged_keys = {
        node.value
        for node in ast.walk(on_log)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "gripper_clean_action_loss_avg" in logged_keys
    assert "gripper_binary_action_loss_avg" in logged_keys
    assert "action_delta_loss_avg" in logged_keys
    assert "action_jerk_loss_avg" in logged_keys
