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
ENV_TRACE_COLUMN_INDEX = {str(name): i for i, name in enumerate(ENV_TRACE_COLUMNS)}

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


def extract_qpos(obs) -> np.ndarray:
    """RoboFactory raw obs -> qpos[16]."""
    q0 = _to_np(obs["agent"]["panda-0"]["qpos"]).astype(np.float32)
    q1 = _to_np(obs["agent"]["panda-1"]["qpos"]).astype(np.float32)
    # Training used qpos[:8] per arm (7 arm joints + 1 finger joint).
    qpos = np.concatenate([q0[:8], q1[:8]]).astype(np.float32)
    assert qpos.shape == (16,)
    return qpos


def extract_obs(obs):
    """RoboFactory raw obs -> (head_rgb, left_rgb, right_rgb, qpos[16])."""
    sd = obs["sensor_data"]
    head = _to_uint8_rgb(sd["head_camera_global"])
    left = _to_uint8_rgb(sd["head_camera_agent0"])
    right = _to_uint8_rgb(sd["head_camera_agent1"])
    qpos = extract_qpos(obs)
    return head, left, right, qpos


def update_loop_qpos_after_step(
    obs_after,
    commanded16: np.ndarray,
    joint_delta_scale_reference: str,
) -> np.ndarray:
    """Choose the qpos reference for the next substep in a replan window."""
    if joint_delta_scale_reference == "observed":
        return extract_qpos(obs_after)
    return np.asarray(commanded16, dtype=np.float32).copy()


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


def scale_joint_target_delta(
    action16: np.ndarray,
    reference16: np.ndarray,
    scale: float,
    clip: float | None,
    output_clip: float | None = None,
    left_scale: float | None = None,
    right_scale: float | None = None,
) -> np.ndarray:
    """Scale per-step joint-target changes while leaving grippers untouched."""
    arm_scales = (
        scale if left_scale is None else left_scale,
        scale if right_scale is None else right_scale,
    )
    if (
        arm_scales == (1.0, 1.0)
        and (clip is None or clip <= 0.0)
        and (output_clip is None or output_clip <= 0.0)
    ):
        return action16

    out = np.asarray(action16, dtype=np.float32).copy()
    reference16 = np.asarray(reference16, dtype=np.float32)
    for lo, arm_scale in zip((0, 8), arm_scales):
        delta = out[lo:lo + 7] - reference16[lo:lo + 7]
        if clip is not None and clip > 0.0:
            delta = np.clip(delta, -clip, clip)
        scaled_delta = arm_scale * delta
        if output_clip is not None and output_clip > 0.0:
            scaled_delta = np.clip(scaled_delta, -output_clip, output_clip)
        out[lo:lo + 7] = reference16[lo:lo + 7] + scaled_delta
    return out


def limit_joint_target_slew(
    action16: np.ndarray,
    previous16: np.ndarray,
    max_delta: float | None,
) -> np.ndarray:
    """Limit per-joint target changes between consecutive executed actions."""
    if max_delta is None or max_delta <= 0.0:
        return action16

    out = np.asarray(action16, dtype=np.float32).copy()
    previous16 = np.asarray(previous16, dtype=np.float32)
    for lo in (0, 8):
        delta = out[lo:lo + 7] - previous16[lo:lo + 7]
        out[lo:lo + 7] = previous16[lo:lo + 7] + np.clip(delta, -max_delta, max_delta)
    return out


def resolve_joint_delta_controls(
    action_representation: str,
    *,
    scale: float,
    clip: float | None,
    output_clip: float | None,
    left_scale: float | None,
    right_scale: float | None,
    allow_absolute_scale: bool,
) -> tuple[float, float | None, float | None, float | None, float | None, bool]:
    """Return joint-delta controls that match the server action semantics.

    The bimanual policy server reports ``absolute_qpos`` after it has already
    converted DreamZero relative outputs back to absolute joint targets. Applying
    the diagnostic delta-scale knob again at that point turns small absolute
    target corrections into large per-step jumps. Keep the legacy knob available
    for explicit diagnostics, but make the safe behavior the default.
    """
    if action_representation != "absolute_qpos" or allow_absolute_scale:
        return scale, clip, output_clip, left_scale, right_scale, False

    has_non_default = (
        scale != 1.0
        or (clip is not None and clip > 0.0)
        or (output_clip is not None and output_clip > 0.0)
        or left_scale is not None
        or right_scale is not None
    )
    if not has_non_default:
        return scale, clip, output_clip, left_scale, right_scale, False

    return 1.0, None, None, None, None, True


def env_action_dict(abs16: np.ndarray) -> dict:
    return {
        "panda-0": np.asarray(abs16[0:8], dtype=np.float32),
        "panda-1": np.asarray(abs16[8:16], dtype=np.float32),
    }


def apply_gripper_override(
    action16: np.ndarray,
    step: int,
    mode: str,
    close_after_step: int,
    open_value: float,
    close_value: float,
    *,
    latch_state: dict[str, bool] | None = None,
    policy_close_threshold: float = 0.0,
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
    if mode == "policy-close-latch":
        if latch_state is None:
            raise ValueError("policy-close-latch requires latch_state")
        for label, dim in (("left", 7), ("right", 15)):
            if out[dim] < policy_close_threshold:
                latch_state[label] = True
            if latch_state.get(label, False):
                out[dim] = close_value
        return out
    if mode == "open":
        target = open_value
    elif mode == "close":
        target = close_value
    elif mode == "close-after-step":
        if step < close_after_step:
            return out
        target = close_value
    elif mode == "open-until-step":
        if step >= close_after_step:
            return out
        target = open_value
    elif mode == "open-then-close-after-step":
        target = open_value if step < close_after_step else close_value
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


def _trace_array(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


def _trace_float(x):
    arr = _trace_array(x)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def _trace_xyz(actor):
    pose = getattr(actor, "pose", None)
    p = getattr(pose, "p", None)
    arr = _trace_array(p)
    if arr is None or arr.size == 0:
        return [None, None, None]
    arr = np.asarray(arr, dtype=np.float64)
    row = arr if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])[0]
    return [float(row[i]) if i < row.shape[0] else None for i in range(3)]


def _trace_xyz_array(actor) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in _trace_xyz(actor)],
        dtype=np.float32,
    )


def _nan_xyz() -> np.ndarray:
    return np.full(3, np.nan, dtype=np.float32)


def _first_matrix(value) -> np.ndarray:
    arr = _trace_array(value)
    if arr is None:
        raise ValueError("missing matrix")
    arr = np.asarray(arr, dtype=np.float32)
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


def _maybe_grasping(agent, actor) -> float:
    if agent is None or actor is None or not hasattr(agent, "is_grasping"):
        return np.nan
    try:
        return float(_bool_from(agent.is_grasping(actor)))
    except Exception:
        return np.nan


def _maybe_success(info) -> float:
    if not isinstance(info, dict) or "success" not in info:
        return np.nan
    return float(_bool_from(info["success"]))


def collect_env_trace(env, step: int, action16: np.ndarray | None, info=None) -> np.ndarray:
    """Best-effort LiftBarrier physical trace for closed-loop diagnostics."""
    root = getattr(env, "unwrapped", env)
    barrier = getattr(root, "barrier", None)
    barrier_p = _trace_xyz_array(barrier) if barrier is not None else _nan_xyz()

    agents_root = getattr(root, "agent", None)
    agents = list(getattr(agents_root, "agents", []) or [])
    left = agents[0] if len(agents) > 0 else None
    right = agents[1] if len(agents) > 1 else None

    base_p = _trace_xyz_array(getattr(left, "robot", None)) if left is not None else _nan_xyz()
    margin = (
        float(barrier_p[2] - (base_p[2] + 0.15))
        if np.isfinite(barrier_p[2]) and np.isfinite(base_p[2])
        else np.nan
    )

    left_tcp = _trace_xyz_array(getattr(left, "tcp", None)) if left is not None else _nan_xyz()
    right_tcp = _trace_xyz_array(getattr(right, "tcp", None)) if right is not None else _nan_xyz()
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
            margin,
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


def _trace_value(row: np.ndarray, column: str) -> float:
    return float(np.asarray(row).reshape(-1)[ENV_TRACE_COLUMN_INDEX[column]])


def update_liftbarrier_grasp_counts(
    counts: dict[str, int],
    trace_row: np.ndarray,
) -> dict[str, int]:
    """Accumulate RoboFactory LiftBarrier grasp detections from env trace."""
    left = _trace_value(trace_row, "left_grasping")
    right = _trace_value(trace_row, "right_grasping")
    if np.isfinite(left) and left > 0.5:
        counts["left"] = counts.get("left", 0) + 1
    if np.isfinite(right) and right > 0.5:
        counts["right"] = counts.get("right", 0) + 1
    return counts


def liftbarrier_strict_success(
    trace_row: np.ndarray,
    grasp_counts: dict[str, int],
    min_grasp_count: int,
) -> bool:
    """Return true only for simulator success with bilateral grasp evidence."""
    sim_success = _trace_value(trace_row, "success")
    if not (np.isfinite(sim_success) and sim_success > 0.5):
        return False
    required = max(1, int(min_grasp_count))
    return (
        grasp_counts.get("left", 0) >= required
        and grasp_counts.get("right", 0) >= required
    )


def _trace_obs_qpos(obs):
    try:
        q0 = _to_np(obs["agent"]["panda-0"]["qpos"]).astype(np.float32)
        q1 = _to_np(obs["agent"]["panda-1"]["qpos"]).astype(np.float32)
        return np.concatenate([q0[:8], q1[:8]]).astype(float).tolist()
    except Exception:
        return None


def _xy_dist(a, b):
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 2 or len(b) < 2:
        return None
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return None
    return float(np.linalg.norm(np.asarray(a[:2], dtype=np.float64) - np.asarray(b[:2], dtype=np.float64)))


def _less_than(a, b):
    if a is None or b is None:
        return False
    return bool(a < b)


def _trace_state(env, info, reward, term, trunc, seed, step_before, qpos_before, action, obs_after):
    e = getattr(env, "unwrapped", env)
    meat_xyz = _trace_xyz(getattr(e, "meat", None))
    pot_xyz = _trace_xyz(getattr(e, "pot", None))
    agent_root = getattr(e, "agent", None)
    agents = list(getattr(agent_root, "agents", []) or [])
    base_xyz = _trace_xyz(getattr(agents[0], "robot", None)) if agents else [None, None, None]
    tcp_xyz = [_trace_xyz(getattr(agent, "tcp", None)) for agent in agents]

    base_z = base_xyz[2] if len(base_xyz) >= 3 else None
    meat_z = meat_xyz[2] if len(meat_xyz) >= 3 else None
    meat_above_base = None if base_z is None or meat_z is None else float(meat_z - base_z)
    meat_drop_target_z = None if base_z is None else float(base_z + 0.1)
    meat_pot_xy_dist = _xy_dist(meat_xyz, pot_xyz)
    place_success_like = (
        _less_than(meat_z, meat_drop_target_z)
        and meat_pot_xy_dist is not None
        and meat_pot_xy_dist < 0.1
    )

    qpos_before_arr = np.asarray(qpos_before, dtype=np.float32).reshape(-1)
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    joint_delta = []
    for arm in range(2):
        lo = 8 * arm
        if action_arr.size >= lo + 7 and qpos_before_arr.size >= lo + 7:
            joint_delta.extend((action_arr[lo:lo + 7] - qpos_before_arr[lo:lo + 7]).astype(float).tolist())
    gripper_dims = [d for d in (7, 15) if d < action_arr.size]
    info_success = _bool_from(info.get("success", False)) if isinstance(info, dict) else _bool_from(info)

    return {
        "seed": int(seed),
        "step_before": int(step_before),
        "step_after": int(step_before + 1),
        "reward": _trace_float(reward),
        "term": _bool_from(term),
        "trunc": _bool_from(trunc),
        "info_success": bool(info_success),
        "place_success_like": bool(place_success_like),
        "base_xyz": base_xyz,
        "meat_xyz": meat_xyz,
        "pot_xyz": pot_xyz,
        "tcp_xyz": tcp_xyz,
        "meat_above_base": meat_above_base,
        "meat_drop_target_z": meat_drop_target_z,
        "meat_pot_xy_dist": meat_pot_xy_dist,
        "qpos_before": qpos_before_arr.astype(float).tolist(),
        "qpos_after": _trace_obs_qpos(obs_after),
        "exec_action": action_arr.astype(float).tolist(),
        "exec_action_abs_max": float(np.max(np.abs(action_arr))) if action_arr.size else None,
        "exec_action_mean_abs": float(np.mean(np.abs(action_arr))) if action_arr.size else None,
        "joint_delta_abs_max": float(np.max(np.abs(joint_delta))) if joint_delta else None,
        "joint_delta_mean_abs": float(np.mean(np.abs(joint_delta))) if joint_delta else None,
        "grippers": [float(action_arr[d]) for d in gripper_dims],
        "info": {k: _trace_float(v) for k, v in info.items()} if isinstance(info, dict) else {},
    }


def run_episode(
    env,
    ws,
    seed: int,
    prompt: str,
    replan_every: int,
    max_steps: int,
    action_representation: str,
    gripper_override: str,
    gripper_close_after_step: int,
    gripper_open_value: float,
    gripper_close_value: float,
    gripper_policy_close_threshold: float,
    joint_delta_scale: float,
    joint_delta_clip: float | None,
    joint_delta_output_clip: float | None,
    joint_delta_scale_reference: str,
    joint_target_slew_rate: float | None,
    success_mode: str,
    strict_success_min_grasp_count: int,
    left_joint_delta_scale: float | None = None,
    right_joint_delta_scale: float | None = None,
    dump: dict | None = None,
):
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

    last_commanded = extract_qpos(raw_obs)
    grasp_counts = {"left": 0, "right": 0}
    gripper_latch_state = {"left": False, "right": False}
    steps = 0
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

        cur = qpos.copy()
        chunk_reference = qpos.copy()
        for da in actions[:replan_every]:
            abs16 = integrate_action(da, cur, action_representation)
            scale_reference = chunk_reference if joint_delta_scale_reference == "chunk" else cur
            abs16 = scale_joint_target_delta(
                abs16,
                scale_reference,
                joint_delta_scale,
                joint_delta_clip,
                output_clip=joint_delta_output_clip,
                left_scale=left_joint_delta_scale,
                right_scale=right_joint_delta_scale,
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
            )
            abs16 = limit_joint_target_slew(abs16, last_commanded, joint_target_slew_rate)
            qpos_before_step = cur.copy()
            raw_obs, reward, term, trunc, info = env.step(env_action_dict(abs16))
            steps += 1
            last_commanded = abs16.copy()
            trace_row = (
                collect_env_trace(env, steps, abs16, info)
                if dump is not None or success_mode == "strict-lift"
                else None
            )
            if trace_row is not None:
                update_liftbarrier_grasp_counts(grasp_counts, trace_row)
            if dump is not None:
                dump["exec_action"].append(abs16.copy())
                dump["env_trace"].append(trace_row)
                dump["trace"].append(
                    _trace_state(
                        env,
                        info,
                        reward,
                        term,
                        trunc,
                        seed,
                        steps - 1,
                        qpos_before_step,
                        abs16,
                        raw_obs,
                    )
                )
            cur = update_loop_qpos_after_step(
                raw_obs,
                abs16,
                joint_delta_scale_reference,
            )
            if _bool_from(info.get("success", False)):
                if success_mode == "strict-lift":
                    return (
                        liftbarrier_strict_success(
                            trace_row,
                            grasp_counts,
                            strict_success_min_grasp_count,
                        ),
                        steps,
                    )
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
        help=(
            "Diagnostic RoboFactory gripper override. RoboFactory uses +1=open, -1=close. "
            "policy-close-latch leaves policy commands untouched until each arm first "
            "predicts close, then keeps that arm closed for the rest of the episode."
        ),
    )
    ap.add_argument(
        "--gripper-close-after-step",
        type=int,
        default=40,
        help=(
            "Step threshold for gripper schedule overrides. For close-after-step this is "
            "the first step to force close; for open-until-step and "
            "open-then-close-after-step this is the first step where forced open ends."
        ),
    )
    ap.add_argument("--gripper-open-value", type=float, default=1.0)
    ap.add_argument("--gripper-close-value", type=float, default=-1.0)
    ap.add_argument(
        "--gripper-policy-close-threshold",
        type=float,
        default=0.0,
        help=(
            "For --gripper-override=policy-close-latch, latch an arm closed once "
            "its policy gripper command is below this threshold."
        ),
    )
    ap.add_argument(
        "--joint-delta-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic closed-loop control knob. Multiplies joint target "
            "changes relative to --joint-delta-scale-reference; gripper "
            "dimensions are unchanged."
        ),
    )
    ap.add_argument(
        "--joint-delta-clip",
        type=float,
        default=None,
        help=(
            "Optional per-step absolute joint delta clip applied before "
            "--joint-delta-scale. Values <=0 disable clipping."
        ),
    )
    ap.add_argument(
        "--joint-delta-output-clip",
        type=float,
        default=None,
        help=(
            "Optional per-step absolute joint delta clip applied after "
            "--joint-delta-scale. Values <=0 disable clipping. This is useful "
            "for bounded scale sweeps where --joint-delta-scale-reference=previous "
            "can otherwise compound inside a replan window."
        ),
    )
    ap.add_argument(
        "--left-joint-delta-scale",
        type=float,
        default=None,
        help=(
            "Optional left-arm-only override for --joint-delta-scale. "
            "Useful for diagnosing asymmetric two-arm contact failures."
        ),
    )
    ap.add_argument(
        "--right-joint-delta-scale",
        type=float,
        default=None,
        help=(
            "Optional right-arm-only override for --joint-delta-scale. "
            "Useful for diagnosing asymmetric two-arm contact failures."
        ),
    )
    ap.add_argument(
        "--joint-delta-scale-reference",
        choices=("previous", "chunk", "observed"),
        default="previous",
        help=(
            "Reference for --joint-delta-scale. 'previous' scales each target "
            "relative to the previous commanded target; 'chunk' scales every "
            "target in a replan window relative to the qpos observed at that "
            "replan; 'observed' scales each target relative to the latest "
            "post-step observed qpos."
        ),
    )
    ap.add_argument(
        "--joint-target-slew-rate",
        type=float,
        default=None,
        help=(
            "Optional per-step per-joint target slew-rate limit applied to arm "
            "joints after joint-delta diagnostics. Values <=0 disable it. "
            "Unlike --joint-delta-output-clip, this limits changes between "
            "consecutive executed targets, so it can smooth oscillation while "
            "still allowing cumulative motion."
        ),
    )
    ap.add_argument(
        "--success-mode",
        choices=("sim", "strict-lift"),
        default="sim",
        help=(
            "Success accounting mode. 'sim' uses RoboFactory info['success']; "
            "'strict-lift' only counts LiftBarrier success when both arms have "
            "positive is_grasping(barrier) evidence before/at simulator success."
        ),
    )
    ap.add_argument(
        "--strict-success-min-grasp-count",
        type=int,
        default=1,
        help=(
            "With --success-mode=strict-lift, each arm must be observed grasping "
            "the barrier at least this many env-trace frames before/at success."
        ),
    )
    ap.add_argument(
        "--allow-absolute-joint-delta-scale",
        action="store_true",
        help=(
            "Allow --joint-delta-* controls even when the policy server reports "
            "absolute_qpos. This preserves the old diagnostic behavior, but it "
            "can create very large absolute joint-target jumps."
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
    args = ap.parse_args()

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
    print(f"Gripper mode:  {args.gripper_override}")
    print(
        f"Joint scale:   {args.joint_delta_scale} "
        f"clip={args.joint_delta_clip} output_clip={args.joint_delta_output_clip} "
        f"ref={args.joint_delta_scale_reference}"
    )
    print(f"Target slew:   {args.joint_target_slew_rate}", flush=True)
    print(
        f"Success mode:  {args.success_mode} "
        f"min_grasp_count={args.strict_success_min_grasp_count}",
        flush=True,
    )
    if args.left_joint_delta_scale is not None or args.right_joint_delta_scale is not None:
        effective_left = (
            args.joint_delta_scale
            if args.left_joint_delta_scale is None
            else args.left_joint_delta_scale
        )
        effective_right = (
            args.joint_delta_scale
            if args.right_joint_delta_scale is None
            else args.right_joint_delta_scale
        )
        print(f"Arm scales:    left={effective_left} right={effective_right}", flush=True)
    if args.gripper_override in {
        "close-after-step",
        "open-until-step",
        "open-then-close-after-step",
    }:
        print(f"Schedule step: {args.gripper_close_after_step}")
    if args.gripper_override == "policy-close-latch":
        print(
            f"Policy close latch threshold: {args.gripper_policy_close_threshold}",
            flush=True,
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
    (
        effective_joint_delta_scale,
        effective_joint_delta_clip,
        effective_joint_delta_output_clip,
        effective_left_joint_delta_scale,
        effective_right_joint_delta_scale,
        ignored_absolute_delta_controls,
    ) = resolve_joint_delta_controls(
        action_representation,
        scale=args.joint_delta_scale,
        clip=args.joint_delta_clip,
        output_clip=args.joint_delta_output_clip,
        left_scale=args.left_joint_delta_scale,
        right_scale=args.right_joint_delta_scale,
        allow_absolute_scale=args.allow_absolute_joint_delta_scale,
    )
    if ignored_absolute_delta_controls:
        print(
            "Ignoring --joint-delta-* controls because the policy server reports "
            "absolute_qpos. Pass --allow-absolute-joint-delta-scale only for "
            "intentional diagnostics.",
            flush=True,
        )

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
                        "prompt": args.prompt,
                        "gripper_override": args.gripper_override,
                        "gripper_close_after_step": args.gripper_close_after_step,
                        "gripper_open_value": args.gripper_open_value,
                        "gripper_close_value": args.gripper_close_value,
                        "gripper_policy_close_threshold": args.gripper_policy_close_threshold,
                        "joint_delta_scale": args.joint_delta_scale,
                        "left_joint_delta_scale": args.left_joint_delta_scale,
                        "right_joint_delta_scale": args.right_joint_delta_scale,
                        "joint_delta_clip": args.joint_delta_clip,
                        "joint_delta_output_clip": args.joint_delta_output_clip,
                        "joint_delta_scale_reference": args.joint_delta_scale_reference,
                        "joint_target_slew_rate": args.joint_target_slew_rate,
                        "success_mode": args.success_mode,
                        "strict_success_min_grasp_count": args.strict_success_min_grasp_count,
                        "effective_joint_delta_scale": effective_joint_delta_scale,
                        "effective_left_joint_delta_scale": effective_left_joint_delta_scale,
                        "effective_right_joint_delta_scale": effective_right_joint_delta_scale,
                        "effective_joint_delta_clip": effective_joint_delta_clip,
                        "effective_joint_delta_output_clip": effective_joint_delta_output_clip,
                        "ignored_absolute_delta_controls": ignored_absolute_delta_controls,
                        "allow_absolute_joint_delta_scale": args.allow_absolute_joint_delta_scale,
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
            "exec_action": np.stack(dump["exec_action"]),     # [n_steps, 16]
            "env_trace": np.stack(dump["env_trace"]),         # [n_steps + 1, len(ENV_TRACE_COLUMNS)]
            "env_trace_columns": ENV_TRACE_COLUMNS,
        }
        if dump.get("action_norm_raw"):
            payload["action_norm_raw"] = np.stack(dump["action_norm_raw"])
        if dump.get("action_norm_clipped"):
            payload["action_norm_clipped"] = np.stack(dump["action_norm_clipped"])
        np.savez_compressed(path, **payload)
        if dump.get("trace"):
            trace_path = os.path.join(args.dump_actions, f"episode_{seed}_trace.jsonl")
            with open(trace_path, "w", encoding="utf-8") as f:
                for row in dump["trace"]:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

    for i in range(args.num_episodes):
        seed = args.seed_start + i
        t0 = time.time()
        dump = (
            {
                "infer_step": [],
                "pred_chunk": [],
                "obs_qpos": [],
                "exec_action": [],
                "env_trace": [],
                "trace": [],
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
                gripper_override=args.gripper_override,
                gripper_close_after_step=args.gripper_close_after_step,
                gripper_open_value=args.gripper_open_value,
                gripper_close_value=args.gripper_close_value,
                gripper_policy_close_threshold=args.gripper_policy_close_threshold,
                joint_delta_scale=effective_joint_delta_scale,
                joint_delta_clip=effective_joint_delta_clip,
                joint_delta_output_clip=effective_joint_delta_output_clip,
                joint_delta_scale_reference=args.joint_delta_scale_reference,
                joint_target_slew_rate=args.joint_target_slew_rate,
                success_mode=args.success_mode,
                strict_success_min_grasp_count=args.strict_success_min_grasp_count,
                left_joint_delta_scale=effective_left_joint_delta_scale,
                right_joint_delta_scale=effective_right_joint_delta_scale,
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
