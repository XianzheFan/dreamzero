from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "osmo_workflows/robotwin/eval_stack_blocks_two_ckpt7000_l40.yaml"
CODE_CACHE_URI = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_gripperconv_20260604"


def _python_heredocs(script: str) -> list[str]:
    lines = script.splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i] == "python - <<'PY'":
            start = i + 1
            end = start
            while end < len(lines) and lines[end] != "PY":
                end += 1
            blocks.append("\n".join(lines[start:end]))
            i = end
        i += 1
    return blocks


def _python_heredoc_with(script: str, marker: str) -> str:
    matches = [block for block in _python_heredocs(script) if marker in block]
    assert len(matches) == 1
    return matches[0]


def test_stack_eval_workflow_gripper_hold_patch_is_valid():
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)

    defaults = workflow["default-values"]
    assert defaults["workflow_name"].endswith(
        "gripperfix-data500-ckpt50000-l40-xianzhef-20260603"
    )
    assert defaults["source_train_run_name"] == (
        "dreamzero-rt-stack-blocks-two-train-shared-global-gripperfix-data500-xianzhef-20260603"
    )
    assert defaults["ckpt_setting"] == "checkpoint-50000"
    assert defaults["num_episodes"] == 1
    assert defaults["client_timeout_seconds"] == 14400
    assert defaults["osmo_cache_root"] == "/mnt/amlfs-04/home/xianzhef/osmo_cache"
    assert defaults["eval_ckpt_root"].endswith(
        "/eval_ckpts/robotwin_stack_blocks_two_dreamzero-rt-stack-blocks-two-train-shared-global-gripperfix-data500-xianzhef-20260603_checkpoint-50000"
    )
    assert defaults["gripper_binarize_threshold"] == "0.5"
    assert defaults["gripper_close_hold_steps"] == "0"
    assert defaults["gripper_close_hold_min_infer"] == "0"
    assert defaults["gripper_force_open_until_infer"] == "0"

    env = workflow["workflow"]["tasks"][0]["environment"]
    assert env["OSMO_CACHE_ROOT"] == "{{osmo_cache_root}}"
    assert env["EVAL_CKPT_ROOT"] == "{{eval_ckpt_root}}"

    script = workflow["workflow"]["tasks"][0]["files"][0]["contents"]
    assert f'CODE_S3_URI="{CODE_CACHE_URI}"' in script
    assert "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/*" in script
    assert "Refusing non-xianzhef data URI" in script
    assert "DreamZero code cache commit" in script
    assert "OSMO_CODE_COMMIT" in script
    assert "DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS" in script
    assert "DREAMZERO_GRIPPER_CLOSE_HOLD_MIN_INFER" in script
    assert "DREAMZERO_GRIPPER_FORCE_OPEN_UNTIL_INFER" in script
    assert "--gripper_close_hold_steps" in script
    assert "--gripper_close_hold_min_infer" in script
    assert "--gripper_force_open_until_infer" in script
    assert "Trying direct eval checkpoint path" in script
    assert '"${src_uri}/${SOURCE_TRAIN_RUN_NAME}/${CKPT_SETTING}"' in script
    assert '"${src_uri}/${CKPT_SETTING}"' in script
    assert "Downloading eval checkpoint candidate from ${src_uri}/" in script
    assert "Acquired eval checkpoint restore lock" in script
    assert "Waiting for eval checkpoint cache lock" in script
    hold_assignment_idx = script.index(
        'DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS="${DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS:-{{gripper_close_hold_steps}}}"'
    )
    hold_export_idx = script.index(
        "export DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS\n",
        hold_assignment_idx,
    )
    patch_idx = script.index("python - <<'PY'")
    assert hold_assignment_idx < hold_export_idx < patch_idx
    assert "/mnt/amlfs-0[1-9]/home/xianzhef/*" in script
    assert "compile(deploy_s, str(deploy), \"exec\")" in script
    assert "require_gripperfix_checkpoint_config" in script
    for marker in (
        "gripper_clean_action_loss_weight: 2.0",
        "gripper_clean_close_action_loss_weight: 4.0",
        "gripper_clean_max_sigma: 0.75",
        "action_prefix_loss_len: 8",
    ):
        assert marker in script

    heredocs = _python_heredocs(script)
    assert len(heredocs) == 3
    for block in heredocs:
        compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec")
    gripper_convention_block = _python_heredoc_with(
        script,
        "RoboTwin gripper convention self-check passed",
    )
    for marker in (
        "range=[0,1], open=1.0, close=0.0",
        "np.clip(gripper_val, 0, 1)",
        "def close_gripper(self, arm_tag: ArmTag, pos: float = 0.0)",
        "def open_gripper(self, arm_tag: ArmTag, pos: float = 1.0)",
        "left_gripper_val < 0.2",
        "left_gripper_val > 0.8",
    ):
        assert marker in gripper_convention_block
    shared_global_block = _python_heredoc_with(
        script,
        "Patched shared-global eval video windows",
    )
    assert "Repeat all camera streams from the current observation" in shared_global_block
    assert "current_global, current_agent0, current_agent1 = history[-1]" in shared_global_block
    assert "Patched eval server to prefer robotwin metadata with robofactory fallback." in shared_global_block
    assert "Patched eval server relative action key matching for state./action. prefixes." in shared_global_block
    assert "def _metadata_tag(self)" in shared_global_block
    assert 'if \\"robotwin\\" in self._metadata' in shared_global_block
    assert 'candidates.add(f"state.{subkey}")' in shared_global_block

    block = _python_heredoc_with(script, "_apply_gripper_close_hold")
    for marker in (
        "_apply_gripper_close_hold",
        "_force_gripper_open_until_infer",
        "DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS",
        "DREAMZERO_GRIPPER_FORCE_OPEN_UNTIL_INFER",
        "_gripper_hold_remaining",
        "gripper_close_hold_min_infer",
        "gripper_force_open_until_infer",
        "hold=(",
    ):
        assert marker in block


def test_stack_eval_workflow_gripper_hold_patch_matches_refreshed_client(
    monkeypatch, tmp_path
):
    with WORKFLOW_PATH.open() as f:
        workflow = yaml.safe_load(f)
    script = workflow["workflow"]["tasks"][0]["files"][0]["contents"]
    block = _python_heredoc_with(script, "_apply_gripper_close_hold")
    block = block.replace(
        'deploy = Path("/workspace/robotwin/policy/DreamZero/deploy_policy.py")',
        'deploy = Path(os.environ["TEST_ROBOTWIN_DEPLOY_POLICY"])',
    )
    block = block.replace(
        'p = Path("/workspace/robotwin/script/eval_policy.py")',
        'p = Path(os.environ["TEST_ROBOTWIN_EVAL_POLICY"])',
    )

    deploy = tmp_path / "deploy_policy.py"
    deploy.write_text(
        '''from __future__ import annotations

import os
import uuid

import numpy as np


class DreamZeroBimanualPolicy:
    def __init__(self, usr_args):
        self.action_mode = str(usr_args.get("action_mode", "auto"))
        self.debug_interval = int(usr_args.get("debug_interval", 0))
        self.session_id: str = uuid.uuid4().hex
        self._needs_reset: bool = True
        self._infer_count: int = 0
        self.action_representation = "absolute_qpos"

    def reset(self) -> None:
        self.session_id = uuid.uuid4().hex
        self._needs_reset = True
        self._infer_count = 0

    def _resolved_action_mode(self) -> str:
        return "absolute"

    def _clip_target_inplace(self, target: np.ndarray) -> None:
        target[7] = np.clip(target[7], 0.0, 1.0)
        target[15] = np.clip(target[15], 0.0, 1.0)

    def _as_abs_action_chunk(self, action_chunk: np.ndarray, qpos: np.ndarray) -> list[np.ndarray]:
        chunk = action_chunk[:8]
        mode = self._resolved_action_mode()
        if mode in ("absolute", "abs_qpos", "qpos"):
            targets = [a.astype(np.float32, copy=True) for a in chunk]
            for target in targets:
                self._clip_target_inplace(target)
            return targets

        targets: list[np.ndarray] = []
        cur = qpos.copy()
        for delta in chunk:
            target = cur.copy()
            targets.append(target.astype(np.float32, copy=True))
            cur = target
        return targets

    def _debug_action(self, qpos: np.ndarray, raw_chunk: np.ndarray, targets: list[np.ndarray]) -> None:
        print(
            "[DreamZero/RoboTwin] "
            f"infer={self._infer_count} mode={self.action_mode}->{self._resolved_action_mode()} "
            f"grip=({targets[0][7]:.3f},{targets[0][15]:.3f}) "
        )
'''
    )

    eval_policy = tmp_path / "eval_policy.py"
    eval_policy.write_text(
        '''import os


def patched_eval(args, TASK_ENV):
    test_num = 100
    task_name = args["task_name"]
    now_id = 0
    if True:
        if True:
            episode_info_list = [TASK_ENV.info.get("info", {})]
    if True:
        instruction = (
            "stack the two blocks"
            if args["task_name"] == "stack_blocks_two"
            else "use the robot arm to pick up the hammer and hit the block"
        )
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction
    return test_num, task_name
'''
    )

    monkeypatch.setenv("TEST_ROBOTWIN_DEPLOY_POLICY", str(deploy))
    monkeypatch.setenv("TEST_ROBOTWIN_EVAL_POLICY", str(eval_policy))
    monkeypatch.setenv("DREAMZERO_GRIPPER_CLOSE_HOLD_STEPS", "64")
    monkeypatch.setenv("DREAMZERO_GRIPPER_CLOSE_HOLD_MIN_INFER", "6")
    monkeypatch.setenv("DREAMZERO_GRIPPER_FORCE_OPEN_UNTIL_INFER", "6")

    exec(compile(block, f"{WORKFLOW_PATH}:embedded-python", "exec"), {})

    patched_deploy = deploy.read_text()
    assert "_apply_gripper_close_hold" in patched_deploy
    assert "_force_gripper_open_until_infer" in patched_deploy
    assert patched_deploy.count("self._apply_gripper_close_hold(targets)") == 2
    assert "gripper_close_hold_min_infer" in patched_deploy
    assert "gripper_force_open_until_infer" in patched_deploy
    assert "hold=(" in patched_deploy
    compile(patched_deploy, str(deploy), "exec")
    ns: dict[str, object] = {}
    exec(compile(patched_deploy, str(deploy), "exec"), ns)
    policy_cls = ns["DreamZeroBimanualPolicy"]
    policy = policy_cls(
        {
            "gripper_close_hold_steps": "64",
            "gripper_close_hold_min_infer": "0",
            "gripper_force_open_until_infer": "6",
        }
    )
    action_chunk = np.zeros((8, 16), dtype=np.float32)
    qpos = np.zeros(16, dtype=np.float32)
    policy._infer_count = 1
    targets = policy._as_abs_action_chunk(action_chunk, qpos)
    assert targets[0][7] == 1.0
    assert targets[0][15] == 1.0
    policy._infer_count = 6
    targets = policy._as_abs_action_chunk(action_chunk, qpos)
    assert targets[0][7] == 0.0
    assert targets[0][15] == 0.0

    patched_eval = eval_policy.read_text()
    assert 'ROBOTWIN_EVAL_EPISODES", "100"' in patched_eval
    assert "prompt_override" in patched_eval
    compile(patched_eval, str(eval_policy), "exec")
