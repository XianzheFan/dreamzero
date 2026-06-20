"""RoboFactory rollout for the DreamZero bimanual policy via WebSocket.

Mirrors ``RoboTwin/policy/DreamZero/deploy_policy.py`` (the RoboTwin
client for the same server) but targets RoboFactory's ManiSkill envs.
Runs in the *RoboFactory* conda env (it imports mani_skill / sapien);
the policy itself lives in a separate dreamzero-env process that this
script reaches over ws://host:port.

Usage (after starting the policy server)::

    /lustre/.../miniconda3/envs/RoboFactory/bin/python \\
        scripts/eval/eval_robofactory_ws.py \\
        --num-episodes 10 --seed-start 1000 \\
        --log /tmp/eval_ckpt2500.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import numpy as np

# msgpack-numpy patch BEFORE we pack any np.ndarray.
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

ENV_TRACE_COLUMNS = np.asarray(
    [
        "step",
        "barrier_x",
        "barrier_y",
        "barrier_z",
        "success_margin",
        "success",
        "left_tcp_x",
        "left_tcp_y",
        "left_tcp_z",
        "right_tcp_x",
        "right_tcp_y",
        "right_tcp_z",
        "left_tcp_to_barrier",
        "right_tcp_to_barrier",
        "left_grasping",
        "right_grasping",
        "cmd_left_gripper",
        "cmd_right_gripper",
        "left_grasp_target_x",
        "left_grasp_target_y",
        "left_grasp_target_z",
        "right_grasp_target_x",
        "right_grasp_target_y",
        "right_grasp_target_z",
        "left_tcp_to_grasp_target",
        "right_tcp_to_grasp_target",
    ],
    dtype="<U32",
)

# Module-level wildcard import for env registration -- Python rejects
# ``from x import *`` inside a function. Only ever imported from the
# RoboFactory conda env (the dreamzero policy server doesn't import
# this client script), so the dependency on ``robofactory`` is safe.
from robofactory.tasks import *  # noqa: F401,F403


def _ws_connect(uri: str, timeout_s: float = 600.0, retry_every: float = 2.0):
    import websockets.sync.client as wsc

    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            return wsc.connect(
                uri,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
            )
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(retry_every)
    raise RuntimeError(f"ws connect to {uri} failed: {last_err}")


def _to_uint8_rgb(sensor_field):
    x = sensor_field["rgb"]
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    return x.astype(np.uint8)


def _to_np(arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    return np.asarray(arr).reshape(-1)


def _first_xyz(value) -> np.ndarray:
    arr = _to_np(value).astype(np.float32)
    if arr.size < 3:
        raise ValueError(f"expected at least 3 values, got shape {arr.shape}")
    return arr[:3]


def _nan_xyz() -> np.ndarray:
    return np.full(3, np.nan, dtype=np.float32)


def _maybe_pose_p(obj) -> np.ndarray:
    try:
        return _first_xyz(obj.pose.p)
    except Exception:
        return _nan_xyz()


def _first_matrix(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4 matrix, got shape {arr.shape}")
    return arr


def _maybe_liftbarrier_grasp_targets(root) -> tuple[np.ndarray, np.ndarray]:
    barrier = getattr(root, "barrier", None)
    annotation_data = getattr(root, "annotation_data", {}) or {}
    actor_data = annotation_data.get("barrier") if isinstance(annotation_data, dict) else None
    if barrier is None or not actor_data:
        return _nan_xyz(), _nan_xyz()

    try:
        actor_matrix = _first_matrix(barrier.pose.to_transformation_matrix())
        contact_poses = actor_data["contact_points_pose"]
        scale = np.asarray(actor_data.get("scale", 1.0), dtype=np.float32)
        convert_matrix = np.asarray(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        targets = []
        for idx in (1, 2):
            local = np.asarray(contact_poses[idx], dtype=np.float32).copy()
            local[:3, 3] *= scale
            global_pose = actor_matrix @ local @ convert_matrix
            targets.append(global_pose[:3, 3].astype(np.float32))
        return targets[0], targets[1]
    except Exception:
        return _nan_xyz(), _nan_xyz()


def _env_root(env):
    return getattr(env, "unwrapped", env)


def _maybe_success(info) -> float:
    if info is None or "success" not in info:
        return np.nan
    return float(_bool_from(info["success"]))


def _maybe_grasping(agent, actor) -> float:
    if agent is None or actor is None or not hasattr(agent, "is_grasping"):
        return np.nan
    try:
        return float(_bool_from(agent.is_grasping(actor)))
    except Exception:
        return np.nan


def collect_env_trace(env, step: int, action16: np.ndarray | None, info=None) -> np.ndarray:
    """Best-effort per-step physical trace for closed-loop diagnostics."""
    root = _env_root(env)
    barrier = getattr(root, "barrier", None)
    barrier_p = _maybe_pose_p(barrier) if barrier is not None else _nan_xyz()

    agents_root = getattr(root, "agent", None)
    agents = list(getattr(agents_root, "agents", []) or [])
    left = agents[0] if len(agents) > 0 else None
    right = agents[1] if len(agents) > 1 else None

    base_p = _maybe_pose_p(left.robot) if left is not None and hasattr(left, "robot") else _nan_xyz()
    margin = barrier_p[2] - (base_p[2] + 0.15) if np.isfinite(barrier_p[2]) and np.isfinite(base_p[2]) else np.nan

    left_tcp = _maybe_pose_p(left.tcp) if left is not None and hasattr(left, "tcp") else _nan_xyz()
    right_tcp = _maybe_pose_p(right.tcp) if right is not None and hasattr(right, "tcp") else _nan_xyz()
    left_dist = (
        float(np.linalg.norm(left_tcp - barrier_p))
        if np.isfinite(left_tcp).all() and np.isfinite(barrier_p).all()
        else np.nan
    )
    right_dist = (
        float(np.linalg.norm(right_tcp - barrier_p))
        if np.isfinite(right_tcp).all() and np.isfinite(barrier_p).all()
        else np.nan
    )
    left_target, right_target = _maybe_liftbarrier_grasp_targets(root)
    left_target_dist = (
        float(np.linalg.norm(left_tcp - left_target))
        if np.isfinite(left_tcp).all() and np.isfinite(left_target).all()
        else np.nan
    )
    right_target_dist = (
        float(np.linalg.norm(right_tcp - right_target))
        if np.isfinite(right_tcp).all() and np.isfinite(right_target).all()
        else np.nan
    )

    action = np.asarray(action16, dtype=np.float32).reshape(-1) if action16 is not None else None
    cmd_left = float(action[7]) if action is not None and action.size > 7 else np.nan
    cmd_right = float(action[15]) if action is not None and action.size > 15 else np.nan

    return np.asarray(
        [
            float(step),
            float(barrier_p[0]),
            float(barrier_p[1]),
            float(barrier_p[2]),
            float(margin),
            _maybe_success(info),
            float(left_tcp[0]),
            float(left_tcp[1]),
            float(left_tcp[2]),
            float(right_tcp[0]),
            float(right_tcp[1]),
            float(right_tcp[2]),
            left_dist,
            right_dist,
            _maybe_grasping(left, barrier),
            _maybe_grasping(right, barrier),
            cmd_left,
            cmd_right,
            float(left_target[0]),
            float(left_target[1]),
            float(left_target[2]),
            float(right_target[0]),
            float(right_target[1]),
            float(right_target[2]),
            left_target_dist,
            right_target_dist,
        ],
        dtype=np.float32,
    )


def extract_obs(obs):
    """RoboFactory raw obs -> (head_rgb, left_rgb, right_rgb, qpos[16])."""
    sd = obs["sensor_data"]
    head = _to_uint8_rgb(sd["head_camera_global"])
    left = _to_uint8_rgb(sd["head_camera_agent0"])
    right = _to_uint8_rgb(sd["head_camera_agent1"])
    q0 = _to_np(obs["agent"]["panda-0"]["qpos"]).astype(np.float32)
    q1 = _to_np(obs["agent"]["panda-1"]["qpos"]).astype(np.float32)
    # Training used qpos[:8] per arm (7 arm joints + 1 finger joint).
    qpos = np.concatenate([q0[:8], q1[:8]]).astype(np.float32)
    assert qpos.shape == (16,)
    return head, left, right, qpos


def integrate_action(
    action16: np.ndarray,
    qpos16: np.ndarray,
    action_representation: str,
) -> np.ndarray:
    """Convert policy output to ManiSkill ``pd_joint_pos`` action.

    Checkpoints trained through DreamZero's ``relative_action`` path are
    converted back to absolute qpos targets by the policy server. Legacy
    checkpoints that stored joint deltas directly still need integration
    against the latest observed qpos here. Gripper commands are always
    forwarded verbatim.
    """
    action16 = np.asarray(action16, dtype=np.float32)
    qpos16 = np.asarray(qpos16, dtype=np.float32)
    if action_representation == "absolute_qpos":
        return action16.astype(np.float32, copy=True)

    out = qpos16.astype(np.float32, copy=True)
    out[0:7] = qpos16[0:7] + action16[0:7]
    out[7] = action16[7]
    out[8:15] = qpos16[8:15] + action16[8:15]
    out[15] = action16[15]
    return out


def env_action_dict(abs16: np.ndarray) -> dict:
    return {
        "panda-0": np.asarray(abs16[0:8], dtype=np.float32),
        "panda-1": np.asarray(abs16[8:16], dtype=np.float32),
    }


def scale_joint_targets(
    action16: np.ndarray,
    reference_qpos16: np.ndarray,
    scale: float,
    clip: float | None = None,
    left_scale: float | None = None,
    right_scale: float | None = None,
) -> np.ndarray:
    """Scale joint target displacement from the current 16-D qpos.

    This is an eval-only diagnostic for separating under-sized joint
    commands from gripper timing. Gripper commands are copied verbatim.
    """
    effective_left = scale if left_scale is None else left_scale
    effective_right = scale if right_scale is None else right_scale
    out = np.asarray(action16, dtype=np.float32).copy()
    if effective_left == 1.0 and effective_right == 1.0 and (clip is None or clip <= 0.0):
        return out
    for label, value in (("left", effective_left), ("right", effective_right)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} joint target scale must be positive and finite, got {value}")
    if clip is not None and clip > 0.0 and not np.isfinite(clip):
        raise ValueError(f"joint target scale clip must be finite when enabled, got {clip}")

    qpos16 = np.asarray(reference_qpos16, dtype=np.float32)
    left_delta = effective_left * (out[0:7] - qpos16[0:7])
    right_delta = effective_right * (out[8:15] - qpos16[8:15])
    if clip is not None and clip > 0.0:
        left_delta = np.clip(left_delta, -clip, clip)
        right_delta = np.clip(right_delta, -clip, clip)
    out[0:7] = qpos16[0:7] + left_delta
    out[8:15] = qpos16[8:15] + right_delta
    return out


def prepare_env_action_target(
    action16: np.ndarray,
    rolling_qpos16: np.ndarray,
    infer_qpos16: np.ndarray,
    action_representation: str,
    joint_target_scale: float,
    joint_target_scale_reference: str = "auto",
    joint_target_scale_clip: float | None = None,
    left_joint_target_scale: float | None = None,
    right_joint_target_scale: float | None = None,
) -> np.ndarray:
    """Convert one policy action row to an env target.

    ``absolute_qpos`` chunks are anchored to the qpos observed at inference time.
    Diagnostic scaling must use that same anchor for the whole chunk; otherwise
    scaling compounds against the previous command and can explode.
    """
    abs16 = integrate_action(action16, rolling_qpos16, action_representation)
    reference_mode = joint_target_scale_reference.replace("-", "_")
    if reference_mode == "auto":
        scale_reference = (
            np.asarray(infer_qpos16, dtype=np.float32)
            if action_representation == "absolute_qpos"
            else np.asarray(rolling_qpos16, dtype=np.float32)
        )
    elif reference_mode in ("infer", "inference", "anchor", "observed", "obs"):
        scale_reference = np.asarray(infer_qpos16, dtype=np.float32)
    elif reference_mode in ("rolling", "previous", "current"):
        scale_reference = np.asarray(rolling_qpos16, dtype=np.float32)
    else:
        raise ValueError(
            f"unknown joint target scale reference: {joint_target_scale_reference!r}"
        )
    return scale_joint_targets(
        abs16,
        scale_reference,
        joint_target_scale,
        joint_target_scale_clip,
        left_scale=left_joint_target_scale,
        right_scale=right_joint_target_scale,
    )


def limit_joint_target_slew(
    action16: np.ndarray,
    previous_action16: np.ndarray | None,
    max_joint_delta: float | None,
) -> np.ndarray:
    """Limit per-env-step joint target jumps while preserving gripper commands."""
    out = np.asarray(action16, dtype=np.float32).copy()
    if previous_action16 is None or max_joint_delta is None or max_joint_delta <= 0.0:
        return out
    if not np.isfinite(max_joint_delta):
        raise ValueError(f"joint target slew rate must be finite, got {max_joint_delta}")

    prev = np.asarray(previous_action16, dtype=np.float32)
    for start, end in ((0, 7), (8, 15)):
        delta = np.clip(out[start:end] - prev[start:end], -max_joint_delta, max_joint_delta)
        out[start:end] = prev[start:end] + delta
    return out


def blend_replan_boundary_target(
    action16: np.ndarray,
    boundary_anchor16: np.ndarray | None,
    chunk_offset: int,
    blend_steps: int,
) -> np.ndarray:
    """Linearly bridge joint targets at the start of a replanned chunk.

    This is an eval-only diagnostic for separating policy jitter from chunk
    boundary discontinuity. Gripper commands are intentionally preserved.
    """
    out = np.asarray(action16, dtype=np.float32).copy()
    if chunk_offset < 0:
        raise ValueError(f"chunk_offset must be >= 0, got {chunk_offset}")
    if blend_steps < 0:
        raise ValueError(f"boundary blend steps must be >= 0, got {blend_steps}")
    if boundary_anchor16 is None or blend_steps == 0 or chunk_offset >= blend_steps:
        return out

    anchor = np.asarray(boundary_anchor16, dtype=np.float32)
    alpha = float(chunk_offset + 1) / float(blend_steps)
    for start, end in ((0, 7), (8, 15)):
        out[start:end] = anchor[start:end] + alpha * (out[start:end] - anchor[start:end])
    return out


def apply_gripper_override(
    action16: np.ndarray,
    step: int,
    mode: str,
    close_after_step: int,
    open_value: float,
    close_value: float,
    latch_state: dict[str, bool] | None = None,
    policy_close_threshold: float = 0.0,
    policy_close_min_step: int = 0,
    left_close_after_step: int | None = None,
    right_close_after_step: int | None = None,
) -> np.ndarray:
    """Optionally override RoboFactory gripper commands.

    RoboFactory's motion-planner convention is +1=open, -1=close. The
    policy output is left untouched by default; explicit overrides are
    diagnostics for separating arm trajectory quality from gripper
    command failures.
    """
    out = np.asarray(action16, dtype=np.float32).copy()
    if mode == "none":
        return out
    if mode == "open":
        target = open_value
    elif mode == "close":
        target = close_value
    elif mode == "close-after-step":
        for dim, threshold in (
            (7, close_after_step if left_close_after_step is None else left_close_after_step),
            (15, close_after_step if right_close_after_step is None else right_close_after_step),
        ):
            if step >= threshold:
                out[dim] = close_value
        return out
    elif mode == "open-until-step":
        for dim, threshold in (
            (7, close_after_step if left_close_after_step is None else left_close_after_step),
            (15, close_after_step if right_close_after_step is None else right_close_after_step),
        ):
            if step < threshold:
                out[dim] = open_value
        return out
    elif mode == "open-then-close-after-step":
        for dim, threshold in (
            (7, close_after_step if left_close_after_step is None else left_close_after_step),
            (15, close_after_step if right_close_after_step is None else right_close_after_step),
        ):
            out[dim] = open_value if step < threshold else close_value
        return out
    elif mode == "policy-close-latch":
        if latch_state is None:
            latch_state = {}
        for key, dim in (("left", 7), ("right", 15)):
            if step >= policy_close_min_step and out[dim] <= policy_close_threshold:
                latch_state[key] = True
            if latch_state.get(key, False):
                out[dim] = close_value
        return out
    else:
        raise ValueError(f"unknown gripper override mode: {mode}")

    out[7] = target
    out[15] = target
    return out


def _bool_from(info_val) -> bool:
    if info_val is None:
        return False
    if hasattr(info_val, "item"):
        return bool(info_val.item())
    if hasattr(info_val, "any"):
        return bool(np.any(info_val))
    return bool(info_val)


def _current_grasp_count(env) -> int:
    root = _env_root(env)
    barrier = getattr(root, "barrier", None)
    agents_root = getattr(root, "agent", None)
    agents = list(getattr(agents_root, "agents", []) or [])
    count = 0
    for agent in agents[:2]:
        if agent is not None and barrier is not None and hasattr(agent, "is_grasping"):
            try:
                count += int(_bool_from(agent.is_grasping(barrier)))
            except Exception:
                pass
    return count


def episode_success(
    env,
    info,
    mode: str,
    strict_success_min_grasp_count: int,
) -> bool:
    env_success = _bool_from(info.get("success", False)) if isinstance(info, dict) else _bool_from(info)
    normalized_mode = mode.replace("_", "-")
    if normalized_mode in ("env", "default"):
        return env_success
    if normalized_mode == "strict-lift":
        return env_success and _current_grasp_count(env) >= strict_success_min_grasp_count
    raise ValueError(f"unknown success mode: {mode!r}")


def run_episode(
    env,
    ws,
    seed: int,
    prompt: str,
    replan_every: int,
    max_steps: int,
    action_representation: str,
    joint_target_scale: float,
    joint_target_scale_reference: str,
    joint_target_scale_clip: float | None,
    left_joint_target_scale: float | None,
    right_joint_target_scale: float | None,
    gripper_override: str,
    gripper_close_after_step: int,
    left_gripper_close_after_step: int | None,
    right_gripper_close_after_step: int | None,
    gripper_open_value: float,
    gripper_close_value: float,
    gripper_policy_close_threshold: float,
    gripper_policy_close_min_step: int,
    joint_target_slew_rate: float,
    replan_boundary_blend_steps: int,
    success_mode: str,
    strict_success_min_grasp_count: int,
    dump: dict | None = None,
):
    # RoboFactory's RFSceneBuilder samples object poses with global np.random
    # rather than ManiSkill's episode RNG. Seed it explicitly so seed labels
    # correspond to reproducible object poses across independent eval jobs.
    np.random.seed(seed)
    raw_obs, _ = env.reset(seed=seed)
    if dump is not None:
        dump["env_trace"].append(collect_env_trace(env, 0, None, None))
    session_id = uuid.uuid4().hex
    ws.send(
        msgpack.packb(
            {"endpoint": "reset", "session_id": session_id, "prompt": prompt},
            use_bin_type=True,
        )
    )
    _ = ws.recv()  # ack

    steps = 0
    last_commanded: np.ndarray | None = None
    gripper_latch_state = {"left": False, "right": False}
    while steps < max_steps:
        head, left, right, qpos = extract_obs(raw_obs)
        ws.send(
            msgpack.packb(
                {
                    "endpoint": "infer",
                    "session_id": session_id,
                    "qpos": qpos,
                    "head_rgb": head,
                    "left_rgb": left,
                    "right_rgb": right,
                    "step": int(steps),
                    "replan_every": int(replan_every),
                    "chunk_start_index": int(steps),
                    "prompt": prompt,
                },
                use_bin_type=True,
            )
        )
        reply_raw = ws.recv()
        if isinstance(reply_raw, str):
            raise RuntimeError(f"server error: {reply_raw}")
        reply = msgpack.unpackb(reply_raw, raw=False)
        actions = np.asarray(reply["action_chunk"], dtype=np.float32)
        assert actions.ndim == 2 and actions.shape[1] == 16

        if dump is not None:
            # Record the full denormalized predicted chunk, optional
            # normalized raw/clipped chunks, the request step, and the qpos
            # the model conditioned on. This is read-only bookkeeping; the
            # replan/open-loop logic below is untouched.
            dump["infer_step"].append(int(steps))
            dump["pred_chunk"].append(actions.copy())
            dump["obs_qpos"].append(qpos.copy())
            if "action_norm_raw" in reply:
                dump["action_norm_raw"].append(
                    np.asarray(reply["action_norm_raw"], dtype=np.float32).copy()
                )
            if "action_norm_clipped" in reply:
                dump["action_norm_clipped"].append(
                    np.asarray(reply["action_norm_clipped"], dtype=np.float32).copy()
                )

        infer_qpos = qpos.copy()
        cur = infer_qpos.copy()
        boundary_anchor = last_commanded.copy() if last_commanded is not None else None
        for chunk_offset, da in enumerate(actions[:replan_every]):
            abs16 = prepare_env_action_target(
                da,
                cur,
                infer_qpos,
                action_representation,
                joint_target_scale,
                joint_target_scale_reference,
                joint_target_scale_clip,
                left_joint_target_scale=left_joint_target_scale,
                right_joint_target_scale=right_joint_target_scale,
            )
            abs16 = apply_gripper_override(
                abs16,
                steps,
                gripper_override,
                gripper_close_after_step,
                gripper_open_value,
                gripper_close_value,
                latch_state=gripper_latch_state,
                policy_close_threshold=gripper_policy_close_threshold,
                policy_close_min_step=gripper_policy_close_min_step,
                left_close_after_step=left_gripper_close_after_step,
                right_close_after_step=right_gripper_close_after_step,
            )
            pre_blend_abs16 = abs16.copy()
            abs16 = blend_replan_boundary_target(
                abs16,
                boundary_anchor,
                chunk_offset,
                replan_boundary_blend_steps,
            )
            pre_slew_abs16 = abs16.copy()
            abs16 = limit_joint_target_slew(abs16, last_commanded, joint_target_slew_rate)
            raw_obs, reward, term, trunc, info = env.step(env_action_dict(abs16))
            last_commanded = abs16.copy()
            cur = abs16
            steps += 1
            if dump is not None:
                dump["exec_action_pre_blend"].append(pre_blend_abs16.copy())
                dump["exec_action_pre_slew"].append(pre_slew_abs16.copy())
                dump["exec_action"].append(abs16.copy())
                dump["env_trace"].append(collect_env_trace(env, steps, abs16, info))
            if episode_success(
                env,
                info,
                success_mode,
                strict_success_min_grasp_count,
            ):
                return True, steps
            if _bool_from(term) or _bool_from(trunc):
                return False, steps
            if steps >= max_steps:
                break
    return False, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=5001, type=int)
    ap.add_argument("--seed-start", default=1000, type=int)
    ap.add_argument("--num-episodes", default=10, type=int)
    ap.add_argument("--max-steps", default=300, type=int)
    ap.add_argument("--replan-every", default=8, type=int)
    ap.add_argument("--task", default="LiftBarrier-rf")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--prompt",
        default="the two robot arms lift the steel barrier together off the table",
    )
    ap.add_argument("--log", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-setting", default=None)
    ap.add_argument(
        "--gripper-override",
        choices=(
            "none",
            "open",
            "close",
            "close-after-step",
            "open-until-step",
            "open-then-close-after-step",
            "policy-close-latch",
        ),
        default="none",
        help="Diagnostic RoboFactory gripper override. RoboFactory uses +1=open, -1=close.",
    )
    ap.add_argument(
        "--gripper-close-after-step",
        type=int,
        default=40,
        help=(
            "First env step to force close when --gripper-override=close-after-step "
            "or open-then-close-after-step. The latter also forces open before "
            "this step."
        ),
    )
    ap.add_argument(
        "--left-gripper-close-after-step",
        type=int,
        default=None,
        help=(
            "Optional left-arm override for --gripper-close-after-step. "
            "Defaults to the shared schedule step."
        ),
    )
    ap.add_argument(
        "--right-gripper-close-after-step",
        type=int,
        default=None,
        help=(
            "Optional right-arm override for --gripper-close-after-step. "
            "Defaults to the shared schedule step."
        ),
    )
    ap.add_argument("--gripper-open-value", type=float, default=1.0)
    ap.add_argument("--gripper-close-value", type=float, default=-1.0)
    ap.add_argument(
        "--gripper-policy-close-threshold",
        type=float,
        default=0.0,
        help=(
            "For --gripper-override=policy-close-latch, latch each gripper closed "
            "once the policy command is <= this threshold."
        ),
    )
    ap.add_argument(
        "--gripper-policy-close-min-step",
        type=int,
        default=0,
        help=(
            "Earliest env step where policy-close-latch may latch a gripper closed."
        ),
    )
    ap.add_argument(
        "--joint-target-scale",
        "--joint-delta-scale",
        dest="joint_target_scale",
        type=float,
        default=1.0,
        help=(
            "Eval-only diagnostic: scale joint target displacement relative to "
            "the current commanded qpos before gripper overrides. 1.0 preserves "
            "policy output."
        ),
    )
    ap.add_argument(
        "--joint-target-scale-reference",
        "--joint-delta-scale-reference",
        dest="joint_target_scale_reference",
        default="auto",
        choices=(
            "auto",
            "infer",
            "inference",
            "anchor",
            "observed",
            "obs",
            "rolling",
            "previous",
            "current",
        ),
        help=(
            "Reference qpos for --joint-target-scale. auto preserves the "
            "safe default: absolute_qpos chunks use the inference-time qpos "
            "anchor, while legacy delta chunks use the rolling commanded qpos."
        ),
    )
    ap.add_argument(
        "--joint-target-scale-clip",
        "--joint-delta-scale-clip",
        "--joint-delta-output-clip",
        dest="joint_target_scale_clip",
        type=float,
        default=0.0,
        help=(
            "Optional eval-only per-joint displacement clip in radians after "
            "scaling. 0 disables clipping."
        ),
    )
    ap.add_argument(
        "--joint-target-slew-rate",
        type=float,
        default=0.0,
        help=(
            "Optional eval-only per-step joint target slew limit in radians. "
            "0 disables smoothing. Gripper commands are not smoothed."
        ),
    )
    ap.add_argument(
        "--replan-boundary-blend-steps",
        type=int,
        default=0,
        help=(
            "Optional eval-only number of steps at the start of each replanned "
            "chunk to linearly bridge joint targets from the last executed "
            "command. 0 disables boundary blending. Gripper commands are not "
            "blended."
        ),
    )
    ap.add_argument(
        "--left-joint-target-scale",
        "--left-joint-delta-scale",
        dest="left_joint_target_scale",
        type=float,
        default=None,
        help=(
            "Optional left-arm-only override for --joint-target-scale. "
            "Useful for diagnosing asymmetric two-arm contact failures."
        ),
    )
    ap.add_argument(
        "--right-joint-target-scale",
        "--right-joint-delta-scale",
        dest="right_joint_target_scale",
        type=float,
        default=None,
        help=(
            "Optional right-arm-only override for --joint-target-scale. "
            "Useful for diagnosing asymmetric two-arm contact failures."
        ),
    )
    ap.add_argument(
        "--video-dir",
        default=None,
        help="If set, wrap env with RecordEpisode and write one mp4 per seed.",
    )
    ap.add_argument(
        "--dump-actions",
        default=None,
        help="If set, write one episode_<seed>.npz per episode containing the "
        "full denormalized predicted action chunks, optional raw/clipped "
        "normalized chunks, the qpos the model saw, and executed actions. "
        "Gripper dims are 7 (left) and 15 (right): >0=open, <0=close.",
    )
    ap.add_argument(
        "--success-mode",
        choices=("env", "default", "strict-lift"),
        default="env",
        help=(
            "Success predicate. strict-lift requires env success and at least "
            "--strict-success-min-grasp-count current grasping arms."
        ),
    )
    ap.add_argument(
        "--strict-success-min-grasp-count",
        type=int,
        default=1,
        help="Minimum current grasping arms required for --success-mode=strict-lift.",
    )
    args = ap.parse_args()
    if args.replan_boundary_blend_steps < 0:
        ap.error("--replan-boundary-blend-steps must be >= 0")

    # Late imports so a server-side schema mismatch fails before the heavy
    # sapien init. (``from x import *`` is illegal inside a function, so
    # the env-registration wildcard import lives at module level below.)
    import gymnasium as gym
    from robofactory import CONFIG_DIR

    if args.config is None:
        args.config = os.path.join(CONFIG_DIR, "table", "lift_barrier.yaml")
    print(f"Config:        {args.config}")
    print(f"Task:          {args.task}")
    print(f"Server:        ws://{args.host}:{args.port}")
    print(f"Seeds:         {args.seed_start}..{args.seed_start + args.num_episodes - 1}")
    print(f"Max steps:     {args.max_steps}")
    print(f"Replan every:  {args.replan_every}")
    print(f"Joint scale:   {args.joint_target_scale}")
    print(f"Joint ref:     {args.joint_target_scale_reference}")
    print(f"Joint clip:    {args.joint_target_scale_clip}")
    print(f"Joint slew:    {args.joint_target_slew_rate}")
    print(f"Boundary blend: {args.replan_boundary_blend_steps}")
    if args.left_joint_target_scale is not None or args.right_joint_target_scale is not None:
        effective_left = (
            args.joint_target_scale
            if args.left_joint_target_scale is None
            else args.left_joint_target_scale
        )
        effective_right = (
            args.joint_target_scale
            if args.right_joint_target_scale is None
            else args.right_joint_target_scale
        )
        print(f"Arm scales:    left={effective_left} right={effective_right}")
    print(f"Gripper mode:  {args.gripper_override}")
    if args.gripper_override in (
        "close-after-step",
        "open-until-step",
        "open-then-close-after-step",
    ):
        left_close_after = (
            args.gripper_close_after_step
            if args.left_gripper_close_after_step is None
            else args.left_gripper_close_after_step
        )
        right_close_after = (
            args.gripper_close_after_step
            if args.right_gripper_close_after_step is None
            else args.right_gripper_close_after_step
        )
        print(
            f"Close after:   {args.gripper_close_after_step} "
            f"(left={left_close_after} right={right_close_after})"
        )
    if args.gripper_override == "policy-close-latch":
        print(
            "Policy close latch: "
            f"threshold={args.gripper_policy_close_threshold} "
            f"min_step={args.gripper_policy_close_min_step}"
        )
    print(
        f"Success mode:  {args.success_mode} "
        f"(min_grasp={args.strict_success_min_grasp_count})"
    )

    # Build env FIRST (sapien init takes ~30-60s); only then open the
    # ws connection. The sync ws client doesn't service pings while
    # ``gym.make`` blocks the thread, so opening ws before this
    # heavy step lets the server's default 20s/20s ping/timeout close
    # the connection idle, and the first ws.send afterwards either
    # hangs or fails silently.
    print("Building env…", flush=True)
    # Let sapien render at its task-default resolution; the policy
    # server's ``infer`` does ``cv2.resize`` to its own ``image_h /
    # image_w`` (=240/320, matching LeRobot v2 mp4) before feeding
    # the transform chain. Forcing sensor 240px here was triggering
    # a sapien-side hang during ``gym.make``.
    env = gym.make(
        args.task,
        config=args.config,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="cpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
    )
    if args.video_dir:
        from mani_skill.utils.wrappers import RecordEpisode

        os.makedirs(args.video_dir, exist_ok=True)
        # RecordEpisode wraps reset()/step() and writes a single mp4 per
        # episode boundary; trajectory_name picks up the seed via the
        # reset-time options dict (set below in run_episode).
        env = RecordEpisode(
            env,
            output_dir=args.video_dir,
            save_trajectory=False,
            save_video=True,
            info_on_video=True,
            video_fps=30,
        )
        print(f"Video recording -> {args.video_dir}", flush=True)
    print("Env ready.", flush=True)

    print("Connecting to policy server (waits up to 600s for warm-up)…", flush=True)
    ws = _ws_connect(f"ws://{args.host}:{args.port}", timeout_s=600.0)
    meta_raw = ws.recv()
    meta = msgpack.unpackb(meta_raw, raw=False) if isinstance(meta_raw, (bytes, bytearray)) else json.loads(meta_raw)
    print(f"Server meta: {meta}", flush=True)
    assert meta.get("num_agents") == 2, f"server reports num_agents={meta.get('num_agents')}"
    action_representation = meta.get("action_representation", "robotwin_delta")

    results = []
    def _write_partial() -> None:
        if not args.log:
            return
        ok = sum(1 for r in results if r.get("success"))
        rate = ok / len(results) if results else 0.0
        with open(args.log, "w") as f:
            json.dump(
                {
                    "results": results,
                    "success_rate": rate,
                    "ckpt": args.ckpt_setting or args.host,
                    "ckpt_dir": args.ckpt_dir,
                    "ckpt_setting": args.ckpt_setting,
                    "server": {"host": args.host, "port": args.port, "meta": meta},
                    "eval_config": {
                        "task": args.task,
                        "seed_start": args.seed_start,
                        "num_episodes": args.num_episodes,
                        "max_steps": args.max_steps,
                        "replan_every": args.replan_every,
                        "joint_target_scale": args.joint_target_scale,
                        "joint_delta_scale": args.joint_target_scale,
                        "left_joint_target_scale": args.left_joint_target_scale,
                        "right_joint_target_scale": args.right_joint_target_scale,
                        "joint_target_scale_reference": args.joint_target_scale_reference,
                        "joint_delta_scale_reference": args.joint_target_scale_reference,
                        "joint_target_scale_clip": args.joint_target_scale_clip,
                        "joint_delta_output_clip": args.joint_target_scale_clip,
                        "joint_target_slew_rate": args.joint_target_slew_rate,
                        "replan_boundary_blend_steps": args.replan_boundary_blend_steps,
                        "effective_joint_delta_scale": args.joint_target_scale,
                        "effective_joint_delta_output_clip": args.joint_target_scale_clip,
                        "allow_absolute_joint_delta_scale": True,
                        "prompt": args.prompt,
                        "gripper_override": args.gripper_override,
                        "gripper_close_after_step": args.gripper_close_after_step,
                        "left_gripper_close_after_step": args.left_gripper_close_after_step,
                        "right_gripper_close_after_step": args.right_gripper_close_after_step,
                        "gripper_open_value": args.gripper_open_value,
                        "gripper_close_value": args.gripper_close_value,
                        "gripper_policy_close_threshold": args.gripper_policy_close_threshold,
                        "gripper_policy_close_min_step": args.gripper_policy_close_min_step,
                        "success_mode": args.success_mode,
                        "strict_success_min_grasp_count": args.strict_success_min_grasp_count,
                    },
                    "n_completed": len(results),
                    "n_target": args.num_episodes,
                },
                f, indent=2,
            )

    if args.dump_actions:
        os.makedirs(args.dump_actions, exist_ok=True)
        print(f"Dumping action chunks -> {args.dump_actions}", flush=True)

    def _save_dump(seed: int, dump: dict | None, success: bool) -> None:
        if dump is None or not dump["pred_chunk"]:
            return
        path = os.path.join(args.dump_actions, f"episode_{seed}.npz")
        payload = {
            "seed": seed,
            "success": bool(success),
            "infer_step": np.asarray(dump["infer_step"], dtype=np.int32),
            "pred_chunk": np.stack(dump["pred_chunk"]),       # denorm [n_infer, chunk_len, 16]
            "obs_qpos": np.stack(dump["obs_qpos"]),           # [n_infer, 16]
            "exec_action_pre_blend": np.stack(dump["exec_action_pre_blend"]),
            "exec_action_pre_slew": np.stack(dump["exec_action_pre_slew"]),
            "exec_action": np.stack(dump["exec_action"]),     # [n_steps, 16]
            "env_trace": np.stack(dump["env_trace"]),         # [n_steps + 1, len(ENV_TRACE_COLUMNS)]
            "env_trace_columns": ENV_TRACE_COLUMNS,
        }
        if dump.get("action_norm_raw"):
            payload["action_norm_raw"] = np.stack(dump["action_norm_raw"])
        if dump.get("action_norm_clipped"):
            payload["action_norm_clipped"] = np.stack(dump["action_norm_clipped"])
        np.savez_compressed(path, **payload)

    for i in range(args.num_episodes):
        seed = args.seed_start + i
        t0 = time.time()
        dump = (
            {
                "infer_step": [],
                "pred_chunk": [],
                "obs_qpos": [],
                "exec_action_pre_blend": [],
                "exec_action_pre_slew": [],
                "exec_action": [],
                "env_trace": [],
                "action_norm_raw": [],
                "action_norm_clipped": [],
            }
            if args.dump_actions
            else None
        )
        try:
            success, steps = run_episode(
                env, ws, seed, args.prompt, args.replan_every, args.max_steps,
                action_representation=action_representation,
                joint_target_scale=args.joint_target_scale,
                joint_target_scale_reference=args.joint_target_scale_reference,
                joint_target_scale_clip=args.joint_target_scale_clip,
                left_joint_target_scale=args.left_joint_target_scale,
                right_joint_target_scale=args.right_joint_target_scale,
                gripper_override=args.gripper_override,
                gripper_close_after_step=args.gripper_close_after_step,
                left_gripper_close_after_step=args.left_gripper_close_after_step,
                right_gripper_close_after_step=args.right_gripper_close_after_step,
                gripper_open_value=args.gripper_open_value,
                gripper_close_value=args.gripper_close_value,
                gripper_policy_close_threshold=args.gripper_policy_close_threshold,
                gripper_policy_close_min_step=args.gripper_policy_close_min_step,
                joint_target_slew_rate=args.joint_target_slew_rate,
                replan_boundary_blend_steps=args.replan_boundary_blend_steps,
                success_mode=args.success_mode,
                strict_success_min_grasp_count=args.strict_success_min_grasp_count,
                dump=dump,
            )
        except Exception as e:
            print(f"seed={seed} ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({"seed": seed, "success": False, "steps": -1, "wall_s": 0.0, "error": str(e)})
            _save_dump(seed, dump, False)   # persist whatever chunks we got before the error
            _write_partial()    # persist partial so a slurm preempt doesn't lose finished seeds
            continue
        dt = time.time() - t0
        results.append({"seed": seed, "success": bool(success), "steps": int(steps), "wall_s": round(dt, 1)})
        print(f"seed={seed} success={success} steps={steps} wall={dt:.1f}s", flush=True)
        _save_dump(seed, dump, success)
        _write_partial()

    ok = sum(1 for r in results if r.get("success"))
    rate = ok / len(results) if results else 0.0
    print(f"\nTotal: {ok}/{len(results)} = {rate * 100:.1f}%", flush=True)

    if args.log:
        _write_partial()
        print(f"Wrote {args.log}", flush=True)

    env.close()
    ws.close()


if __name__ == "__main__":
    main()
