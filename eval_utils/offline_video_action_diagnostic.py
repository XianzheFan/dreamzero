"""Model-only bimanual action and predicted-video diagnostic.

This runs the DreamZero policy on batches from the checkpoint's training
dataset. It does not import RoboFactory, ManiSkill, or SAPIEN, so it can run on
GB200/aarch64 where the closed-loop simulator stack is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_utils.offline_eval_bimanual import (
    PER_ARM_GRIPPER_DIMS,
    PER_ARM_JOINT_DIMS,
    build_dataloader,
    find_gt_action,
    find_pred_action,
    gripper_open_close_metrics,
    load_model,
    load_resolved_cfg,
    move_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-setting", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--inference-mode",
        choices=("causal_flowmatch", "noncausal_flowmatch"),
        default="noncausal_flowmatch",
    )
    parser.add_argument("--gripper-class-threshold", type=float, default=0.0)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--skip-forward-loss",
        action="store_true",
        help="Skip model.forward() loss sanity check before get_action().",
    )
    return parser.parse_args()


def configure_inference_mode(mode: str) -> None:
    os.environ["MAI_CAUSAL_SCHEDULER"] = "flowmatch"
    if mode == "causal_flowmatch":
        os.environ["MAI_USE_CAUSAL_INFERENCE"] = "1"
    elif mode == "noncausal_flowmatch":
        os.environ["MAI_USE_CAUSAL_INFERENCE"] = "0"
    else:
        raise ValueError(f"unsupported inference mode: {mode}")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    return value


def _masked_flat(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    valid = np.asarray(valid).astype(bool)
    if values.shape != valid.shape:
        raise ValueError(f"shape mismatch: {values.shape} vs {valid.shape}")
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32)
    return values[valid].astype(np.float32)


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def action_diagnostic_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    gripper_threshold: float,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.asarray(valid).astype(bool)

    err = np.abs(pred - gt)
    joint_dims = PER_ARM_JOINT_DIMS
    gripper_dim = PER_ARM_GRIPPER_DIMS[0]
    joint_valid = valid[..., joint_dims]
    gripper_valid = valid[..., gripper_dim]

    pred_joint_delta = np.diff(pred[..., joint_dims], axis=2)
    gt_joint_delta = np.diff(gt[..., joint_dims], axis=2)
    delta_valid = valid[..., 1:, joint_dims] & valid[..., :-1, joint_dims]
    pred_joint_jerk = np.diff(pred_joint_delta, axis=2)
    gt_joint_jerk = np.diff(gt_joint_delta, axis=2)
    jerk_valid = delta_valid[..., 1:, :] & delta_valid[..., :-1, :]

    pred_gripper_close = pred[..., gripper_dim] < gripper_threshold
    gt_gripper_close = gt[..., gripper_dim] < gripper_threshold
    gripper_switch_valid = gripper_valid[..., 1:] & gripper_valid[..., :-1]
    pred_gripper_switch = np.diff(pred_gripper_close.astype(np.int8), axis=2) != 0
    gt_gripper_switch = np.diff(gt_gripper_close.astype(np.int8), axis=2) != 0

    pred_delta_l2 = np.linalg.norm(pred_joint_delta, axis=-1)
    gt_delta_l2 = np.linalg.norm(gt_joint_delta, axis=-1)
    delta_l2_valid = np.all(delta_valid, axis=-1)
    pred_jerk_l2 = np.linalg.norm(pred_joint_jerk, axis=-1)
    gt_jerk_l2 = np.linalg.norm(gt_joint_jerk, axis=-1)
    jerk_l2_valid = np.all(jerk_valid, axis=-1)

    gripper_pred = pred[..., gripper_dim][gripper_valid]
    gripper_gt = gt[..., gripper_dim][gripper_valid]

    metrics = {
        "joint_l1": _summary_stats(_masked_flat(err[..., joint_dims], joint_valid)),
        "gripper_l1": _summary_stats(
            _masked_flat(err[..., gripper_dim], gripper_valid)
        ),
        "pred_joint_delta_l2": _summary_stats(
            _masked_flat(pred_delta_l2, delta_l2_valid)
        ),
        "gt_joint_delta_l2": _summary_stats(_masked_flat(gt_delta_l2, delta_l2_valid)),
        "pred_joint_jerk_l2": _summary_stats(_masked_flat(pred_jerk_l2, jerk_l2_valid)),
        "gt_joint_jerk_l2": _summary_stats(_masked_flat(gt_jerk_l2, jerk_l2_valid)),
        "pred_gripper_switch_rate": float(
            pred_gripper_switch[gripper_switch_valid].mean()
        )
        if np.any(gripper_switch_valid)
        else None,
        "gt_gripper_switch_rate": float(gt_gripper_switch[gripper_switch_valid].mean())
        if np.any(gripper_switch_valid)
        else None,
        "gripper_open_close": gripper_open_close_metrics(
            gripper_pred,
            gripper_gt,
            threshold=gripper_threshold,
        )
        if gripper_pred.size
        else None,
    }

    pred_delta_mean = metrics["pred_joint_delta_l2"]["mean"]
    gt_delta_mean = metrics["gt_joint_delta_l2"]["mean"]
    pred_jerk_mean = metrics["pred_joint_jerk_l2"]["mean"]
    gt_jerk_mean = metrics["gt_joint_jerk_l2"]["mean"]
    metrics["pred_to_gt_delta_l2_mean_ratio"] = (
        float(pred_delta_mean / gt_delta_mean)
        if pred_delta_mean is not None and gt_delta_mean not in (None, 0.0)
        else None
    )
    metrics["pred_to_gt_jerk_l2_mean_ratio"] = (
        float(pred_jerk_mean / gt_jerk_mean)
        if pred_jerk_mean is not None and gt_jerk_mean not in (None, 0.0)
        else None
    )
    return metrics


def decode_latent_video(model, latents: torch.Tensor, device: str, dtype) -> np.ndarray:
    """Decode ``[B, P, C, F, H, W]`` latents to ``[P, T, H, W, 3]`` uint8."""
    action_head = model.action_head
    bsz, num_agents, channels, frames, height, width = latents.shape
    latents_bp = latents.reshape(
        bsz * num_agents, channels, frames, height, width
    ).to(device=device, dtype=dtype)
    with torch.inference_mode():
        decoded = action_head.vae.decode(
            latents_bp,
            tiled=getattr(action_head, "tiled", False),
            tile_size=(
                getattr(action_head, "tile_size_height", 34),
                getattr(action_head, "tile_size_width", 34),
            ),
            tile_stride=(
                getattr(action_head, "tile_stride_height", 18),
                getattr(action_head, "tile_stride_width", 16),
            ),
        )
    decoded = ((decoded.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    arr = decoded.cpu().numpy()
    arr = arr.transpose(0, 2, 3, 4, 1)
    return arr.reshape(bsz, num_agents, -1, arr.shape[2], arr.shape[3], 3)[0]


def write_video_set(frames: np.ndarray, out_dir: Path, prefix: str, fps: int) -> list[str]:
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for agent_idx in range(frames.shape[0]):
        name = f"{prefix}_agent{agent_idx}.mp4"
        path = out_dir / name
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("h264", rate=fps)
            stream.width = int(frames.shape[3])
            stream.height = int(frames.shape[2])
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23"}
            for frame_idx in range(frames.shape[1]):
                frame = av.VideoFrame.from_ndarray(
                    frames[agent_idx, frame_idx],
                    format="rgb24",
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        written.append(name)
    return written


def video_metrics(frames: np.ndarray) -> dict[str, Any]:
    frames_f = frames.astype(np.float32)
    diffs = np.abs(np.diff(frames_f, axis=1)) if frames.shape[1] > 1 else None
    return {
        "decoded_shape": [int(x) for x in frames.shape],
        "pixel_mean": float(frames_f.mean()),
        "pixel_std": float(frames_f.std()),
        "black_fraction": float((frames <= 5).mean()),
        "white_fraction": float((frames >= 250).mean()),
        "frame_absdiff_mean": float(diffs.mean()) if diffs is not None else None,
        "frame_absdiff_p95": float(np.quantile(diffs, 0.95))
        if diffs is not None
        else None,
    }


def write_text_summary(summary: dict[str, Any], path: Path) -> None:
    def metric_mean(row: dict[str, Any], key: str) -> float:
        value = row["action_metrics"][key]["mean"]
        return float("nan") if value is None else float(value)

    lines = [
        f"checkpoint: {summary['ckpt_dir']}/{summary['ckpt_setting']}",
        f"inference_mode: {summary['inference_mode']}",
        f"num_batches: {summary['num_batches']}",
        "",
    ]
    for row in summary["batches"]:
        lines.append(
            "batch {batch}: get_action={get_action_seconds:.1f}s "
            "joint_l1_mean={joint_l1:.4f} "
            "pred_delta_mean={pred_delta:.4f} "
            "gt_delta_mean={gt_delta:.4f} "
            "pred_jerk_mean={pred_jerk:.4f} "
            "gt_jerk_mean={gt_jerk:.4f} "
            "video_frame_diff={video_diff}".format(
                batch=row["batch"],
                get_action_seconds=row["get_action_seconds"],
                joint_l1=metric_mean(row, "joint_l1"),
                pred_delta=metric_mean(row, "pred_joint_delta_l2"),
                gt_delta=metric_mean(row, "gt_joint_delta_l2"),
                pred_jerk=metric_mean(row, "pred_joint_jerk_l2"),
                gt_jerk=metric_mean(row, "gt_joint_jerk_l2"),
                video_diff=(
                    "n/a"
                    if row.get("video_metrics") is None
                    else f"{row['video_metrics']['frame_absdiff_mean']:.2f}"
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configure_inference_mode(args.inference_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg, metadata = load_resolved_cfg(args.ckpt_dir, args.ckpt_setting)
    model, dtype = load_model(args.ckpt_dir, args.ckpt_setting, cfg, args.device)
    loader = build_dataloader(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        metadata=metadata,
    )

    summary: dict[str, Any] = {
        "ckpt_dir": str(args.ckpt_dir),
        "ckpt_setting": args.ckpt_setting,
        "inference_mode": args.inference_mode,
        "num_batches": args.num_batches,
        "env": {
            "MAI_USE_CAUSAL_INFERENCE": os.environ.get("MAI_USE_CAUSAL_INFERENCE"),
            "MAI_CAUSAL_SCHEDULER": os.environ.get("MAI_CAUSAL_SCHEDULER"),
            "MAI_WRITE_DENOISED_CONTEXT_CACHE": os.environ.get(
                "MAI_WRITE_DENOISED_CONTEXT_CACHE"
            ),
            "MAI_ROLLING_NOISE": os.environ.get("MAI_ROLLING_NOISE"),
        },
        "batches": [],
    }

    manifest_dir = args.out_dir / "video_pred"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.jsonl"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            gt, mask = find_gt_action(batch)
            if gt is None:
                print(f"[batch {batch_idx}] missing action key; skipping", flush=True)
                continue

            batch_gpu = move_to_device(batch, args.device, dtype)
            forward_losses = None
            if not args.skip_forward_loss:
                train_out = model.forward(batch_gpu)
                train_data = train_out.data if hasattr(train_out, "data") else train_out
                forward_losses = {
                    k: float(train_data[k])
                    for k in ("loss", "dynamics_loss", "action_loss")
                    if k in train_data
                }
                print(
                    f"[batch {batch_idx}] forward_losses={forward_losses}",
                    flush=True,
                )

            start = time.time()
            outputs = model.get_action(batch_gpu)
            get_action_seconds = time.time() - start
            pred = find_pred_action(outputs)

            horizon = min(pred.shape[-2], gt.shape[-2])
            dims = min(pred.shape[-1], gt.shape[-1])
            pred_np = pred[..., :horizon, :dims].detach().float().cpu().numpy()
            gt_np = gt[..., :horizon, :dims].float().cpu().numpy()
            mask_np = mask[..., :horizon, :dims].bool().cpu().numpy()
            action_metrics = action_diagnostic_metrics(
                pred_np,
                gt_np,
                mask_np,
                args.gripper_class_threshold,
            )

            row: dict[str, Any] = {
                "batch": batch_idx,
                "get_action_seconds": get_action_seconds,
                "pred_shape": list(pred_np.shape),
                "gt_shape": list(gt_np.shape),
                "forward_losses": forward_losses,
                "action_metrics": action_metrics,
                "video_metrics": None,
                "pred_files": [],
            }

            latents = getattr(model.action_head, "_last_video_pred", None)
            if latents is None:
                print(f"[batch {batch_idx}] no _last_video_pred latents", flush=True)
            else:
                print(
                    f"[batch {batch_idx}] decoding video latents "
                    f"shape={tuple(latents.shape)}",
                    flush=True,
                )
                frames = decode_latent_video(model, latents.detach(), args.device, dtype)
                prefix = f"batch{batch_idx:03d}_{args.inference_mode}"
                pred_files = write_video_set(
                    frames,
                    manifest_dir,
                    prefix=prefix,
                    fps=args.video_fps,
                )
                row["video_metrics"] = video_metrics(frames)
                row["pred_files"] = pred_files
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(
                        json.dumps(
                            {
                                "infer_idx": batch_idx,
                                "env_step": None,
                                "session_id_prefix": "offline",
                                "latent_shape": list(latents.shape),
                                "decoded_shape": row["video_metrics"]["decoded_shape"],
                                "pred_files": pred_files,
                                "observed_files": [],
                                "comparison_files": [],
                                "video_pred_rollout_mode": args.inference_mode,
                                "last_video_pred_rollout_mode": getattr(
                                    model.action_head,
                                    "_last_video_pred_rollout_mode",
                                    None,
                                ),
                                "pred_latent_start_frame": getattr(
                                    model.action_head,
                                    "_last_video_pred_start_frame",
                                    None,
                                ),
                                "pred_latent_end_frame": getattr(
                                    model.action_head,
                                    "_last_video_pred_end_frame",
                                    None,
                                ),
                                "pred_video_semantics": (
                                    "offline dataset batch decoded denoised "
                                    "future video latents"
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

            print(
                f"[batch {batch_idx}] get_action={get_action_seconds:.1f}s "
                f"metrics={json.dumps(jsonable(action_metrics), sort_keys=True)}",
                flush=True,
            )
            summary["batches"].append(row)

    (args.out_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_text_summary(summary, args.out_dir / "summary.txt")
    print(f"Wrote diagnostic summary to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
