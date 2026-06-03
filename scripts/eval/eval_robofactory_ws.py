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

from eval_utils.gripper_convention import (
    gripper_values_mismatch_message,
    resolve_gripper_values_from_server_meta,
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
        target = close_value if step >= close_after_step else open_value
    else:
        raise ValueError(f"unknown gripper override mode: {mode}")

    out[7] = target
    out[15] = target
    return out


def apply_client_gripper_binarize(
    action16: np.ndarray,
    threshold: float | None,
    open_value: float,
    close_value: float,
) -> np.ndarray:
    """Discretize model gripper outputs by value, not by time.

    This is a calibration diagnostic for continuous gripper predictions. It
    keeps the model's open/close timing and only changes the sign threshold.
    """
    out = np.asarray(action16, dtype=np.float32).copy()
    if threshold is None:
        return out
    out[7] = open_value if out[7] >= threshold else close_value
    out[15] = open_value if out[15] >= threshold else close_value
    return out


def warn_if_gripper_values_mismatch(
    meta: dict,
    client_open: float,
    client_close: float,
) -> None:
    try:
        message = gripper_values_mismatch_message(
            meta,
            client_open=client_open,
            client_close=client_close,
        )
    except Exception as exc:
        print(f"WARNING: cannot validate server gripper values: {exc}", flush=True)
        return
    if message:
        print(f"WARNING: {message}", flush=True)


def _bool_from(info_val) -> bool:
    if info_val is None:
        return False
    if hasattr(info_val, "item"):
        return bool(info_val.item())
    if hasattr(info_val, "any"):
        return bool(np.any(info_val))
    return bool(info_val)


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
    gripper_chunk_lookahead: int,
    gripper_exec_mode: str,
    client_gripper_binarize_threshold: float | None,
    dump: dict | None = None,
):
    raw_obs, _ = env.reset(seed=seed)
    session_id = uuid.uuid4().hex
    # Absolute env step -> (left_cmd, right_cmd, source_infer_step, source_chunk_idx).
    # Queue mode preserves the model's own future gripper predictions until
    # their intended env timestep instead of discarding them at each replan.
    gripper_schedule: dict[int, tuple[float, float, int, int]] = {}
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
        infer_step = int(steps)

        if gripper_exec_mode == "queue":
            for stale_step in [s for s in gripper_schedule if s < infer_step]:
                del gripper_schedule[stale_step]
            for chunk_idx in range(actions.shape[0]):
                env_step = infer_step + chunk_idx
                gripper_schedule.setdefault(
                    env_step,
                    (
                        float(actions[chunk_idx, 7]),
                        float(actions[chunk_idx, 15]),
                        infer_step,
                        chunk_idx,
                    ),
                )

        if dump is not None:
            # Record the full denormalized predicted chunk, optional
            # normalized raw/clipped chunks, the request step, and the qpos
            # the model conditioned on. This is read-only bookkeeping; the
            # replan/open-loop logic below is untouched.
            dump["infer_step"].append(infer_step)
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
            for key in ("action_physical_pre_binarize", "action_physical_final"):
                if key in reply:
                    dump[key].append(
                        np.asarray(reply[key], dtype=np.float32).copy()
                    )

        cur = qpos.copy()
        for chunk_idx, da in enumerate(actions[:replan_every]):
            exec_step = int(steps)
            gripper_chunk_idx = min(
                chunk_idx + gripper_chunk_lookahead,
                actions.shape[0] - 1,
            )
            gripper_source_infer_step = infer_step
            abs16 = integrate_action(da, cur, action_representation)
            if gripper_exec_mode == "queue":
                queued = gripper_schedule.get(exec_step)
                if queued is not None:
                    left_g, right_g, gripper_source_infer_step, gripper_chunk_idx = queued
                    abs16[7] = left_g
                    abs16[15] = right_g
            elif gripper_chunk_lookahead > 0:
                # Use the model's own future gripper prediction while keeping
                # short-horizon receding control for the arm joints.
                abs16[7] = actions[gripper_chunk_idx, 7]
                abs16[15] = actions[gripper_chunk_idx, 15]
            pre_client_binarize = abs16.copy()
            abs16 = apply_client_gripper_binarize(
                abs16,
                client_gripper_binarize_threshold,
                gripper_open_value,
                gripper_close_value,
            )
            abs16 = apply_gripper_override(
                abs16,
                steps,
                gripper_override,
                gripper_close_after_step,
                gripper_open_value,
                gripper_close_value,
            )
            raw_obs, reward, term, trunc, info = env.step(env_action_dict(abs16))
            cur = abs16
            steps += 1
            if dump is not None:
                dump["exec_action"].append(abs16.copy())
                dump["exec_action_pre_client_binarize"].append(
                    pre_client_binarize.copy()
                )
                dump["exec_chunk_index"].append(int(chunk_idx))
                dump["exec_gripper_chunk_index"].append(int(gripper_chunk_idx))
                dump["exec_gripper_source_infer_step"].append(
                    int(gripper_source_infer_step)
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
        choices=("none", "open", "close", "close-after-step"),
        default="none",
        help="Diagnostic RoboFactory gripper override. RoboFactory uses +1=open, -1=close.",
    )
    ap.add_argument(
        "--gripper-close-after-step",
        type=int,
        default=40,
        help=(
            "First env step to force close when --gripper-override=close-after-step; "
            "earlier steps are forced open."
        ),
    )
    ap.add_argument("--gripper-open-value", type=float, default=1.0)
    ap.add_argument("--gripper-close-value", type=float, default=-1.0)
    ap.add_argument(
        "--client-gripper-values-from-server-metadata",
        action="store_true",
        help=(
            "Use the policy server's q99/q01 gripper action statistics as "
            "the client-side binarize open/close values. By default eval keeps "
            "RoboFactory's +1=open, -1=close convention."
        ),
    )
    ap.add_argument(
        "--client-gripper-binarize-threshold",
        type=float,
        default=None,
        help=(
            "Optional client-side gripper threshold in physical action units. "
            "When set, gripper commands >= threshold become --gripper-open-value "
            "and lower commands become --gripper-close-value. This calibrates "
            "model gripper sign without a time-based forced schedule."
        ),
    )
    ap.add_argument(
        "--gripper-chunk-lookahead",
        type=int,
        default=0,
        help=(
            "Use the model-predicted gripper command from this many future "
            "chunk positions while executing joints from the current chunk "
            "position. 0 preserves the original behavior; this is not a "
            "forced open/close schedule."
        ),
    )
    ap.add_argument(
        "--gripper-exec-mode",
        choices=("chunk", "queue"),
        default="chunk",
        help=(
            "How to execute model gripper predictions. 'chunk' preserves the "
            "original receding-horizon behavior (optionally with "
            "--gripper-chunk-lookahead). 'queue' keeps each model-predicted "
            "future gripper command and executes it at its corresponding env "
            "step; arm joints still use the latest chunk."
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
    print(f"Client grip bin:{args.client_gripper_binarize_threshold}")
    print(f"Grip exec:     {args.gripper_exec_mode}")
    print(f"Grip lookahead:{args.gripper_chunk_lookahead}")
    if args.gripper_override == "close-after-step":
        print(f"Close after:   {args.gripper_close_after_step}")
    if args.gripper_chunk_lookahead < 0:
        raise ValueError("--gripper-chunk-lookahead must be >= 0")

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
    print(
        f"Server gripper action values: {meta.get('gripper_action_values')}",
        flush=True,
    )
    if args.client_gripper_values_from_server_metadata:
        if not isinstance(meta.get("gripper_action_values"), dict):
            print(
                "Server meta has no gripper_action_values; using CLI gripper "
                f"values open={args.gripper_open_value} "
                f"close={args.gripper_close_value}",
                flush=True,
            )
        args.gripper_open_value, args.gripper_close_value = (
            resolve_gripper_values_from_server_meta(
                meta,
                fallback_open=args.gripper_open_value,
                fallback_close=args.gripper_close_value,
            )
        )
        print(
            "Using server metadata gripper values for client binarize: "
            f"open={args.gripper_open_value} close={args.gripper_close_value}",
            flush=True,
        )
    else:
        warn_if_gripper_values_mismatch(
            meta,
            client_open=args.gripper_open_value,
            client_close=args.gripper_close_value,
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
                        "client_gripper_values_from_server_metadata": (
                            args.client_gripper_values_from_server_metadata
                        ),
                        "client_gripper_binarize_threshold": args.client_gripper_binarize_threshold,
                        "gripper_chunk_lookahead": args.gripper_chunk_lookahead,
                        "gripper_exec_mode": args.gripper_exec_mode,
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
            "exec_action_pre_client_binarize": np.stack(
                dump["exec_action_pre_client_binarize"]
            ),
            "exec_chunk_index": np.asarray(dump["exec_chunk_index"], dtype=np.int32),
            "exec_gripper_chunk_index": np.asarray(
                dump["exec_gripper_chunk_index"], dtype=np.int32
            ),
            "exec_gripper_source_infer_step": np.asarray(
                dump["exec_gripper_source_infer_step"], dtype=np.int32
            ),
        }
        if dump.get("action_norm_raw"):
            payload["action_norm_raw"] = np.stack(dump["action_norm_raw"])
        if dump.get("action_norm_clipped"):
            payload["action_norm_clipped"] = np.stack(dump["action_norm_clipped"])
        for key in ("action_physical_pre_binarize", "action_physical_final"):
            if dump.get(key):
                payload[key] = np.stack(dump[key])
        np.savez_compressed(path, **payload)

    for i in range(args.num_episodes):
        seed = args.seed_start + i
        t0 = time.time()
        dump = (
            {
                "infer_step": [],
                "pred_chunk": [],
                "obs_qpos": [],
                "exec_action": [],
                "exec_action_pre_client_binarize": [],
                "exec_chunk_index": [],
                "exec_gripper_chunk_index": [],
                "exec_gripper_source_infer_step": [],
                "action_norm_raw": [],
                "action_norm_clipped": [],
                "action_physical_pre_binarize": [],
                "action_physical_final": [],
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
                gripper_chunk_lookahead=args.gripper_chunk_lookahead,
                gripper_exec_mode=args.gripper_exec_mode,
                client_gripper_binarize_threshold=args.client_gripper_binarize_threshold,
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
