"""Offline action-prediction eval for the bimanual (multi-agent) DreamZero VLA.

Loads a saved checkpoint, samples N batches from its training-time
LeRobot dataset (configured via the dumped Hydra ``experiment_cfg``),
runs ``model.get_action(batch)``, and compares the predicted action
chunk against the ground-truth action chunk that lives in the same
batch under the per-arm action keys.

This is *not* a closed-loop sim eval — there is no env, no reset, no
chained policy rollout. It answers the question "for a given observation
in a real RoboFactory episode, how close is the policy's action chunk
to what the demonstrator actually did". Lower L1/L2 + higher
threshold-pass rate => better.

Usage::

    python -m eval_utils.offline_eval_bimanual \\
        --ckpt-dir /lustre/.../checkpoints/robofactory_bimanual_liftbarrier_v3_droid \\
        --ckpt-setting checkpoint-5500 \\
        --num-batches 8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401 -- needed before config access

# UniPC scheduler is torch.compile'd and accumulates per-step model_outputs
# history. With num_inference_steps=16 and two separate streams (video + action)
# that is up to ~32 distinct shapes/lengths, well past the default cache of 8.
torch._dynamo.config.cache_size_limit = 64

from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader


# Per-arm action layout for RoboFactory bimanual (matches data config:
# agent_action_dims=[[0,8],[8,16]] with 7 absolute joint targets + 1 gripper).
PER_ARM_JOINT_DIMS = list(range(0, 7))      # [0..6]
PER_ARM_GRIPPER_DIMS = [7]
ARM_SLICES = [(0, 8), (8, 16)]              # left, right
JOINT_THRESHOLDS = [0.02, 0.05, 0.1]        # normalized-action units
GRIPPER_THRESHOLDS = [0.05, 0.1, 0.2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Top-level checkpoint dir (contains checkpoint-N/ subdirs).",
    )
    p.add_argument(
        "--ckpt-setting",
        type=str,
        default="checkpoint-5500",
        help="Which sub-dir to load LoRA weights from.",
    )
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_resolved_cfg(ckpt_dir: Path, ckpt_setting: str):
    exp_cfg_dir = ckpt_dir / ckpt_setting / "experiment_cfg"
    if not (exp_cfg_dir / "conf.yaml").is_file():
        exp_cfg_dir = ckpt_dir / "experiment_cfg"
    cfg_path = exp_cfg_dir / "conf.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No experiment_cfg/conf.yaml at {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))

    meta_path = exp_cfg_dir / "metadata.json"
    metadata = json.load(open(meta_path)) if meta_path.is_file() else {}
    return cfg, metadata


def load_model(ckpt_dir: Path, ckpt_setting: str, cfg, device: str):
    """Mirror groot/vla/experiment/base.py::create_model's load sequence.

    Three-step load (skip any of these and the model is silently random):
      1. ``instantiate(cfg.model)`` -- architecture only, no weights.
      2. Load ``cfg.pretrained_model_path`` shards (DROID base body +
         text encoder + VAE + image encoder + all the original DreamZero
         weights). Shape-filter so the bimanual deltas don't crash on
         strict=False.
      3. ``inject_lora_after_loading()`` to wrap the DiT body with LoRA
         adapters under the PEFT ``base_model.model.`` prefix.
      4. Load the LoRA fine-tune ckpt (``ckpt-N/model.safetensors``).
         Now the wrapped keys match and the trained LoRA weights apply.
    """
    import gc
    import json as _json

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[load_model] step 1: instantiate(cfg.model)")
    model = instantiate(cfg.model)

    pretrained_path = cfg.get("pretrained_model_path", None)
    if pretrained_path is None:
        raise ValueError(
            "cfg.pretrained_model_path is required (need DROID base body / "
            "text encoder weights). Offline eval cannot proceed without it."
        )
    pretrained_dir = Path(pretrained_path)
    print(f"[load_model] step 2: loading pretrained base from {pretrained_dir}")

    model_state = model.state_dict()
    dropped_mismatched: dict[str, tuple] = {}

    def _filter_shape_mismatches(sd: dict) -> dict:
        kept = {}
        for k, v in sd.items():
            ref = model_state.get(k)
            if ref is not None and tuple(ref.shape) != tuple(v.shape):
                dropped_mismatched[k] = (tuple(v.shape), tuple(ref.shape))
                continue
            kept[k] = v
        return kept

    index_path = pretrained_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            index = _json.load(f)
        n_loaded = 0
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = pretrained_dir / shard_file
            shard_sd = load_file(str(shard_path))
            shard_sd = _filter_shape_mismatches(shard_sd)
            model.load_state_dict(shard_sd, strict=False)
            n_loaded += len(shard_sd)
            del shard_sd
            gc.collect()
        print(f"[load_model]   loaded {n_loaded} pretrained tensors across shards")
    else:
        pretrained_safe = pretrained_dir / "model.safetensors"
        if not pretrained_safe.is_file():
            raise FileNotFoundError(
                f"No model.safetensors[.index.json] under {pretrained_dir}"
            )
        sd = _filter_shape_mismatches(load_file(str(pretrained_safe)))
        model.load_state_dict(sd, strict=False)
        print(f"[load_model]   loaded {len(sd)} pretrained tensors (single file)")
    if dropped_mismatched:
        print(
            f"[load_model]   dropped {len(dropped_mismatched)} shape-mismatched tensor(s); "
            "they re-init from scratch (expected for multi-agent deltas vs DROID)."
        )

    # Step 3: LoRA inject BEFORE loading the fine-tune ckpt so its wrapped
    # keys (``base_model.model.*``) land on the new wrapper.
    if (
        hasattr(model, "action_head")
        and hasattr(model.action_head, "inject_lora_after_loading")
        and getattr(model.action_head.config, "defer_lora_injection", False)
    ):
        print(f"[load_model] step 3: inject_lora_after_loading()")
        model.action_head.inject_lora_after_loading()

    # Step 4: load the fine-tune LoRA + new heads.
    weight_path = ckpt_dir / ckpt_setting / "model.safetensors"
    print(f"[load_model] step 4: loading fine-tune ckpt from {weight_path}")
    state_dict = _filter_shape_mismatches(load_file(str(weight_path)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[load_model]   loaded {len(state_dict)} tensors "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[load_model] model on {device} in {dtype}")
    return model, dtype


def build_dataloader(cfg, batch_size: int, num_workers: int, metadata: dict):
    print("[data] instantiating train_dataset …")
    dataset = instantiate(cfg.train_dataset)
    print(f"[data] dataset type: {type(dataset).__name__}")
    collator = instantiate(cfg.data_collator)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=False,
    )
    return loader


def move_to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def find_gt_action(batch: dict) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return (action, action_mask) from the post-transform batch.

    Bimanual layout: action is [B, P=2, T_a, D_per_arm=8] (7 absolute joint
    targets + 1 gripper per arm). action_mask is the same shape (bool).
    """
    if "action" not in batch:
        return None, None
    action = batch["action"]
    mask = batch.get("action_mask")
    if mask is None:
        mask = torch.ones_like(action, dtype=torch.bool)
    return action, mask


def find_pred_action(outputs) -> torch.Tensor:
    """Extract the predicted action chunk from get_action() output.

    Bimanual training emits ``action_pred`` shaped [B, P, T_a, D_per_arm].
    """
    if hasattr(outputs, "data"):
        outputs = outputs.data
    if isinstance(outputs, dict):
        if "action_pred" in outputs:
            return outputs["action_pred"]
    raise RuntimeError(
        f"Could not extract action_pred from outputs of type {type(outputs)} "
        f"with keys {list(outputs.keys()) if isinstance(outputs, dict) else 'n/a'}"
    )


def collect_dim_subset(
    all_abs_err: list[tuple[np.ndarray, np.ndarray]], dims
) -> np.ndarray:
    """Concatenate per-batch errors at the given per-arm dim subset,
    masked by ``valid``. Each batch may have a different T_a (e.g. 96
    vs 72 for episodes that don't fill the full 4-chunk grid) so we
    flatten to 1D per batch *before* concatenation.
    """
    pieces = []
    for err, valid in all_abs_err:
        e = err[..., dims]
        v = valid[..., dims]
        pieces.append(e[v])
    if not pieces:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def summarize(flat: np.ndarray, thresholds, label: str) -> None:
    if flat.size == 0:
        print(f"  {label}: no valid entries")
        return
    print(f"  {label} (n={flat.size}):")
    print(
        f"    L1 mean   = {flat.mean():.4f}"
        f"   median = {np.median(flat):.4f}"
        f"   p95 = {np.quantile(flat, 0.95):.4f}"
        f"   max = {flat.max():.4f}"
    )
    for thr in thresholds:
        rate = (flat < thr).mean() * 100
        print(f"    |err| < {thr:<6}: {rate:5.1f}%")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, metadata = load_resolved_cfg(args.ckpt_dir, args.ckpt_setting)
    model, dtype = load_model(args.ckpt_dir, args.ckpt_setting, cfg, args.device)
    loader = build_dataloader(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        metadata=metadata,
    )

    all_abs_err: list[np.ndarray] = []
    inference_secs: list[float] = []

    print(f"\n=== Running offline eval on {args.num_batches} batches ===\n")
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            if i == 0:
                print(f"[debug] batch keys: {sorted(batch.keys())}")
                for k in list(batch.keys())[:30]:
                    v = batch[k]
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
                    else:
                        print(f"  {k}: {type(v).__name__}")

            gt, gt_mask = find_gt_action(batch)
            if gt is None:
                print(
                    "[skip] batch missing 'action' key; got: "
                    f"{[k for k in batch.keys() if 'action' in k]}"
                )
                continue

            batch_gpu = move_to_device(batch, args.device, dtype)

            # Sanity: training-mode forward should reproduce the loss-log
            # action_loss for this checkpoint. If THIS number is way off
            # 0.02-0.03, the model isn't loaded correctly. If it's right
            # but inference still gives garbage, the rollout is broken.
            with torch.no_grad():
                train_out = model.forward(batch_gpu)
            train_data = train_out.data if hasattr(train_out, "data") else train_out
            print(
                f"[batch {i}] train-mode forward "
                f"loss={float(train_data['loss']):.4f} "
                f"dynamics_loss={float(train_data['dynamics_loss']):.4f} "
                f"action_loss={float(train_data['action_loss']):.4f}"
            )

            t0 = time.time()
            outputs = model.get_action(batch_gpu)
            inference_secs.append(time.time() - t0)

            if i == 0:
                outdata = outputs.data if hasattr(outputs, "data") else outputs
                print(f"[debug] get_action output keys: {sorted(outdata.keys())}")
                for k, v in outdata.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  out {k}: tensor {tuple(v.shape)} {v.dtype}")
                    else:
                        print(f"  out {k}: {type(v).__name__}")

            pred = find_pred_action(outputs)

            # Match shapes: slice GT to the prediction horizon if longer
            # (training writes 96-step action; model predicts action_horizon=24).
            T = min(pred.shape[-2], gt.shape[-2])
            D = min(pred.shape[-1], gt.shape[-1])
            pred = pred[..., :T, :D].detach().float().cpu()
            gt = gt[..., :T, :D].float()
            gt_mask = gt_mask[..., :T, :D]

            err = (pred - gt).abs()
            valid = gt_mask.bool()
            err_masked = err.masked_select(valid).numpy()
            valid_count = int(valid.sum())
            total_count = int(valid.numel())

            # Store the full-shaped err for per-dim breakdown; mask aggregated
            # to the same shape as err.
            all_abs_err.append((err.numpy(), valid.numpy()))
            print(
                f"[batch {i}] pred {tuple(pred.shape)} gt {tuple(gt.shape)} "
                f"valid {valid_count}/{total_count} "
                f"L1 mean = {err_masked.mean():.4f} (took {inference_secs[-1]:.1f}s)"
            )

    if not all_abs_err:
        print("\nNo batches produced comparable predictions. Aborting.")
        return

    # Per-batch T_a can differ (96 vs 72 for short episodes), so we
    # collect per-dim 1D arrays instead of concatenating the full
    # [B, P, T_a, D] tensors. action_mask drops the padding so partial
    # episodes contribute only their real timesteps.
    joints_flat = collect_dim_subset(all_abs_err, PER_ARM_JOINT_DIMS)
    grippers_flat = collect_dim_subset(all_abs_err, PER_ARM_GRIPPER_DIMS)
    all_flat = np.concatenate([joints_flat, grippers_flat], axis=0)

    n_batches = len(all_abs_err)
    t_a_seen = sorted({e.shape[2] for e, _ in all_abs_err})
    print(f"\n=== Aggregated over {n_batches} batches (T_a values seen: {t_a_seen}) ===")
    print(f"Avg inference time / batch: {np.mean(inference_secs):.1f}s")
    print(
        f"Overall (valid only, n={all_flat.size}): "
        f"L1 mean={all_flat.mean():.4f} median={np.median(all_flat):.4f}\n"
    )

    summarize(joints_flat, JOINT_THRESHOLDS, "Joints (per-arm dims 0..6)")
    summarize(grippers_flat, GRIPPER_THRESHOLDS, "Gripper (per-arm dim 7)")


if __name__ == "__main__":
    main()
