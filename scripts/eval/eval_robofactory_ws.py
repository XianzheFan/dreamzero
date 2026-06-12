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


def scale_joint_target_delta(
    action16: np.ndarray,
    reference16: np.ndarray,
    scale: float,
    clip: float | None,
) -> np.ndarray:
    """Scale per-step joint-target changes while leaving grippers untouched."""
    if scale == 1.0 and (clip is None or clip <= 0.0):
        return action16

    out = np.asarray(action16, dtype=np.float32).copy()
    reference16 = np.asarray(reference16, dtype=np.float32)
    for lo in (0, 8):
        delta = out[lo:lo + 7] - reference16[lo:lo + 7]
        if clip is not None and clip > 0.0:
            delta = np.clip(delta, -clip, clip)
        out[lo:lo + 7] = reference16[lo:lo + 7] + (scale * delta)
    return out


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
    joint_delta_scale: float,
    joint_delta_clip: float | None,
    dump: dict | None = None,
):
    raw_obs, _ = env.reset(seed=seed)
    session_id = uuid.uuid4().hex
    ws.send(
        msgpack.packb(
            {"endpoint": "reset", "session_id": session_id, "prompt": prompt},
            use_bin_type=True,
        )
    )
    _ = ws.recv()  # ack

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
        for da in actions[:replan_every]:
            abs16 = integrate_action(da, cur, action_representation)
            abs16 = scale_joint_target_delta(
                abs16,
                cur,
                joint_delta_scale,
                joint_delta_clip,
            )
            abs16 = apply_gripper_override(
                abs16,
                steps,
                gripper_override,
                gripper_close_after_step,
                gripper_open_value,
                gripper_close_value,
            )
            qpos_before_step = cur.copy()
            raw_obs, reward, term, trunc, info = env.step(env_action_dict(abs16))
            cur = abs16
            steps += 1
            if dump is not None:
                dump["exec_action"].append(abs16.copy())
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
            if _bool_from(info.get("success", False)):
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
        ),
        default="none",
        help="Diagnostic RoboFactory gripper override. RoboFactory uses +1=open, -1=close.",
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
        "--joint-delta-scale",
        type=float,
        default=1.0,
        help=(
            "Diagnostic closed-loop control knob. Multiplies joint target "
            "changes relative to the previous commanded target; gripper "
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
    print(f"Joint scale:   {args.joint_delta_scale} clip={args.joint_delta_clip}")
    if args.gripper_override in {
        "close-after-step",
        "open-until-step",
        "open-then-close-after-step",
    }:
        print(f"Schedule step: {args.gripper_close_after_step}")

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
                        "prompt": args.prompt,
                        "gripper_override": args.gripper_override,
                        "gripper_close_after_step": args.gripper_close_after_step,
                        "gripper_open_value": args.gripper_open_value,
                        "gripper_close_value": args.gripper_close_value,
                        "joint_delta_scale": args.joint_delta_scale,
                        "joint_delta_clip": args.joint_delta_clip,
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
                joint_delta_scale=args.joint_delta_scale,
                joint_delta_clip=args.joint_delta_clip,
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
