"""One-shot smoke for ``BimanualPolicy.infer()``: load v3_droid ckpt,
fabricate a synthetic ``obs`` (16-dim qpos + random RGB triples),
call ``infer()`` once, and print the returned 16-dim action chunk's
shape / range. Confirms the full transform chain + 4-step load + PR 6
multi-agent inference all wire together. End-to-end sim rollout still
needs Vulkan + RoboFactory eval client.

Usage::

    python -m eval_utils.smoke_bimanual_server \\
        --ckpt-dir /lustre/.../checkpoints/robofactory_bimanual_liftbarrier_v3_droid \\
        --ckpt-setting checkpoint-9000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch._dynamo  # noqa: F401

torch._dynamo.config.cache_size_limit = 64

from eval_utils.bimanual_policy_server import BimanualPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--ckpt-setting", type=str, default="checkpoint-9000")
    ap.add_argument("--image-h", type=int, default=240,
                    help="Raw per-camera height fed to the transform chain "
                         "(must match the LeRobot v2 mp4 height).")
    ap.add_argument("--image-w", type=int, default=320)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print(f"[smoke] loading policy from {args.ckpt_dir}/{args.ckpt_setting} …")
    policy = BimanualPolicy(
        ckpt_dir=args.ckpt_dir,
        ckpt_setting=args.ckpt_setting,
        image_h=args.image_h,
        image_w=args.image_w,
    )
    print("[smoke] policy loaded; resetting session …")
    policy.reset({
        "session_id": "smoke",
        "prompt": "the two robot arms lift the barrier together",
    })

    rng = np.random.default_rng(0)
    qpos = rng.standard_normal(16).astype(np.float32) * 0.1
    head = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    left = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    right = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    obs = {
        "session_id": "smoke",
        "qpos": qpos,
        "head_rgb": head,
        "left_rgb": left,
        "right_rgb": right,
    }

    print("[smoke] calling infer() …")
    import time
    t0 = time.time()
    result = policy.infer(obs)
    dt = time.time() - t0
    chunk = result["action_chunk"]
    print(
        f"[smoke] infer ok in {dt:.1f}s; action_chunk "
        f"shape={chunk.shape} dtype={chunk.dtype} "
        f"min={chunk.min():.3f} max={chunk.max():.3f} mean={chunk.mean():.3f}"
    )
    # Sanity: shape must be (T_a=24, 16)
    assert chunk.shape == (policy.action_horizon, policy.action_dim), (
        f"unexpected chunk shape {chunk.shape}"
    )
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()
