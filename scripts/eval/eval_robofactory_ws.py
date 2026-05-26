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


def integrate_action(delta16: np.ndarray, qpos16: np.ndarray) -> np.ndarray:
    """Server output convention (see bimanual_policy_server.py docstring):
      [0:7]  panda0 joint deltas
      [7]    panda0 gripper (absolute)
      [8:15] panda1 joint deltas
      [15]   panda1 gripper (absolute)
    """
    nxt = qpos16.copy()
    nxt[0:7] = qpos16[0:7] + delta16[0:7]
    nxt[7] = delta16[7]
    nxt[8:15] = qpos16[8:15] + delta16[8:15]
    nxt[15] = delta16[15]
    return nxt.astype(np.float32)


def env_action_dict(abs16: np.ndarray) -> dict:
    return {
        "panda-0": np.asarray(abs16[0:8], dtype=np.float32),
        "panda-1": np.asarray(abs16[8:16], dtype=np.float32),
    }


def _bool_from(info_val) -> bool:
    if info_val is None:
        return False
    if hasattr(info_val, "item"):
        return bool(info_val.item())
    if hasattr(info_val, "any"):
        return bool(np.any(info_val))
    return bool(info_val)


def run_episode(env, ws, seed: int, prompt: str, replan_every: int, max_steps: int):
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

        cur = qpos.copy()
        for da in actions[:replan_every]:
            abs16 = integrate_action(da, cur)
            raw_obs, reward, term, trunc, info = env.step(env_action_dict(abs16))
            cur = abs16
            steps += 1
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
    print("Env ready.", flush=True)

    print("Connecting to policy server (waits up to 600s for warm-up)…", flush=True)
    ws = _ws_connect(f"ws://{args.host}:{args.port}", timeout_s=600.0)
    meta_raw = ws.recv()
    meta = msgpack.unpackb(meta_raw, raw=False) if isinstance(meta_raw, (bytes, bytearray)) else json.loads(meta_raw)
    print(f"Server meta: {meta}", flush=True)
    assert meta.get("num_agents") == 2, f"server reports num_agents={meta.get('num_agents')}"

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
                    "ckpt": args.host,
                    "n_completed": len(results),
                    "n_target": args.num_episodes,
                },
                f, indent=2,
            )

    for i in range(args.num_episodes):
        seed = args.seed_start + i
        t0 = time.time()
        try:
            success, steps = run_episode(
                env, ws, seed, args.prompt, args.replan_every, args.max_steps
            )
        except Exception as e:
            print(f"seed={seed} ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({"seed": seed, "success": False, "steps": -1, "wall_s": 0.0, "error": str(e)})
            _write_partial()    # persist partial so a slurm preempt doesn't lose finished seeds
            continue
        dt = time.time() - t0
        results.append({"seed": seed, "success": bool(success), "steps": int(steps), "wall_s": round(dt, 1)})
        print(f"seed={seed} success={success} steps={steps} wall={dt:.1f}s", flush=True)
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
