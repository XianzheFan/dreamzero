# RoboFactory DreamZero OSMO Workflows

Reusable OSMO workflow specs for the DreamZero RoboFactory LiftBarrier runs.

## Training Code Cache

The gamma droidwidth teacher and staged training workflows intentionally leave
`code_s3_uri` and `expected_code_commit` empty by default. Upload the current
branch source and pass both values explicitly when submitting; the container
fails fast if either value is missing, which prevents accidentally training an
old cached code snapshot.

```bash
osmo workflow submit osmo_workflows/robofactory/train_liftbarrier_gamma_droidwidth_teacher.yaml \
  --pool groot-h100-01 \
  --priority LOW \
  --set-string \
  code_s3_uri=swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/<current-code-cache> \
  expected_code_commit=$(git rev-parse HEAD)
```

## Staged Gamma Curriculum

The staged LiftBarrier workflow follows the Gamma-World teacher/student split
more closely than the standalone teacher workflow: stage1 uses dense attention
with `GLOBAL_VIDEO_ATTENTION_MODE=bidirectional`, while both sparse student
stages use `GLOBAL_VIDEO_ATTENTION_MODE=read_only` so global video remains
causal-safe context during policy training.

This is a sparse read-only/global-context student, not a literal Gamma-World
block-causal sparse-hub student. DreamZero keeps denoised chunks bidirectional
inside the training path because the extra block-causal sparse-hub time mask
previously produced strong periodic artifacts in predicted video; model tests
cover that invariant.

The standalone droidwidth teacher workflow also defaults
`GLOBAL_VIDEO_ATTENTION_MODE=bidirectional` so new 32D teacher runs match the
dense-teacher side of this curriculum rather than the sparse read-only student
side.
It also keeps `GLOBAL_VIDEO_DROPOUT_PROB=0.0` by default, matching the
teacher-style objective where the available shared global observation should
remain clean context for video planning.

The standalone droidwidth teacher default is a 50k bidirectional-teacher run:
`run_name=dz-rf-sg-gamma-dwteacher-bidir-nodrop-lb500-50k-xz-20260622`, with the
actual training stage saved under the `-teacher` suffix. The H100 slim eval
templates and 2k checkpoint grid point at that stage run, so the default train
workflow must produce checkpoints through `checkpoint-50000`.

Gamma-World's released bidirectional teacher trains on a much longer video
window than the current DreamZero RoboFactory default. The current droidwidth
teacher keeps the DreamZero-DROID-compatible short window
(`train_num_frames=33`, `train_max_chunk_size=4`) so it can start from the
released DROID backbone without changing the window contract mid-run. For an
explicit long-window pred-video ablation, override these OSMO template values
together:

```bash
--set-string train_num_frames=65 train_max_chunk_size=8
```

When evaluating a long-window checkpoint, pass the matching server window to
the 2k grid helper as well:

```bash
python scripts/eval/submit_robofactory_droidwidth_eval_grid.py \
  --tag 20260621-longwin \
  --only-ready \
  --skip-existing \
  --eval-num-frames 65 \
  --eval-action-horizon 24 \
  --submit
```

In this loader, `train_num_frames` should remain `8 * train_max_chunk_size + 1`;
larger values such as `97/12` are closer to the Gamma-World teacher objective
but should be treated as a separate memory/runtime ablation. The droidwidth
teacher defaults to `train_max_grad_norm=0.1`, matching Gamma-World's
causal-cosmos2 teacher grad-clip setting; override it only for explicit
DreamZero-DROID fine-tuning baseline ablations.

## 2k Checkpoint Eval Grid

The gamma training workflows save checkpoints every 2000 optimizer steps by
default. Keep closed-loop evals on the same 2k cadence by generating one H100
eval workflow per checkpoint instead of running all checkpoints inside one long
workflow.

Dry-run the full `checkpoint-2000` through `checkpoint-50000` grid:

```bash
python scripts/eval/submit_robofactory_droidwidth_eval_grid.py --tag 20260621
```

Submit a subset once the corresponding checkpoints are visible:

```bash
python scripts/eval/submit_robofactory_droidwidth_eval_grid.py \
  --tag 20260621 \
  --steps 2000,4000,6000 \
  --submit
```

For periodic polling, let the helper skip checkpoints whose S3 checkpoint
directory is not ready yet:

```bash
python scripts/eval/submit_robofactory_droidwidth_eval_grid.py \
  --tag 20260621 \
  --steps 2000,4000,6000 \
  --only-ready \
  --skip-existing \
  --submit
```

Add `--fail-if-none-ready` when a scheduler or wrapper should treat "nothing to
submit yet" as a nonzero exit instead of a clean no-op. Keep a stable `--tag`
for a polling run so `--skip-existing` can recognize already-submitted eval
workflows.

Explicit `--steps` values are still checked against the same 2k cadence by
default, so off-grid checkpoints such as `checkpoint-1000`,
`checkpoint-1500`, or `checkpoint-3500` are rejected. Use
`--allow-off-grid-steps` only for isolated diagnostics that should not become
part of the standing checkpoint curve.

The helper uses `osmo workflow submit` with `--set-string` overrides for
`workflow_name`, `run_name`, `ckpt_run_name`, `ckpt_s3_base`,
`ckpt_amlfs_base`, `ckpt_setting`, and `local_eval_ckpt_root`, plus optional
`eval_num_frames` / `eval_action_horizon` overrides for long-window ablations,
so each checkpoint writes to an isolated eval run while still pointing at the
current bidirectional teacher checkpoint tree. The default cadence is:

```text
checkpoint-2000, checkpoint-4000, ..., checkpoint-50000
```

Do not pack the whole grid into one eval workflow. A single H100 closed-loop
eval already runs model load, action diagnostics, video prediction diagnostics,
and a small physical sweep; separate OSMO workflows make retries and queueing
cleaner.

The H100 droidwidth eval template saves both action-path and noncausal
diagnostic predicted videos by default (`VIDEO_PRED_ROLLOUT_MODES="action
noncausal"`). The standing 2k checkpoint curve intentionally does not amplify
actions by default: `replan=24/12`, `scale=1.0`, `accel_limit=0/0.08`,
`blend=4`, and `temporal_ensemble=0.6` for 8 total settings across the two
video rollout modes. This keeps the routine curve focused on model quality and
low-jitter execution rather than hiding errors behind action scaling. Override
`VIDEO_PRED_ROLLOUT_MODE=action` or `VIDEO_PRED_ROLLOUT_MODES=action` when you
only want the control-path video diagnostic and need to cut runtime. Override
the sweep defaults with `--set-string replan_everys=...
joint_delta_scales=... joint_target_accel_limits=...
replan_boundary_blend_steps=... temporal_action_ensemble_decays=...` for
deeper one-off diagnostics; use `joint_delta_scales=2.0` only as an explicit
action-amplification diagnostic, not as the standing eval default. The
video-quality analyzer keeps these rollout modes separated in
`by_video_pred_rollout_mode`, and the checkpoint grid summary exposes
action/noncausal future-MAE columns directly. Physical best-setting selection
uses the action-path row when both modes are present, because noncausal rollout
mode is a video-only diagnostic.

After downloading eval artifacts, summarize the checkpoint grid:

```bash
python scripts/eval/summarize_robofactory_checkpoint_grid.py \
  /path/to/downloaded/eval/runs \
  --csv-out checkpoint_grid.csv \
  --json-out checkpoint_grid.json
```

The summary table compares the best physical setting per checkpoint, including
success, reach/grasp margins, action saturation/clamping, boundary jumps, and
predicted-video future MAE/drift. The video-quality report also scans a small
future-frame alignment window and reports the best offset; a consistent nonzero
best offset is evidence of a pred-video/eval time-window mismatch rather than
only poor visual modeling. The `cond_t0` column compares predicted frame 0 with
the current observed wrist frame when the decoded video includes the conditioning
frame; high `cond_t0` points to a conditioning/VAE/window issue before future
dynamics quality is even evaluated. Future MAE starts from the first actual
future frame, so this conditioning frame is not mixed into the future-quality
score.

The H100 droidwidth eval template also sweeps `--joint-target-accel-limit`
over `0` and `0.08` by default. This is eval-only second-order smoothing for
testing whether action amplification is causing high-frequency target reversals;
the action dump summary reports both acceleration/delta ratios and the
acceleration-limiter correction magnitude. The default keeps boundary blending
and temporal ensembling enabled so the 2k checkpoint grid measures a practical
low-jitter execution setting; raw model-output jitter is still reported
separately. The analyzer separates raw model-output jitter from execution
smoothing: `pred_accel_ratio` and `pred_flip` are computed inside predicted
chunks, while `model_boundary` compares each new chunk's first joint target to
the last command from the previous chunk.

The RoboFactory client now requires the policy server to declare
`action_representation` (`absolute_qpos` or `robotwin_delta`) in its handshake.
This prevents a missing server field from silently falling back to legacy delta
integration, which would make relative-action checkpoints look over-amplified
and jittery. The value is copied into `results.json`, action dumps, sweep
summaries, and checkpoint-grid summaries.

New gamma training checkpoints include `experiment_cfg/runtime_provenance.json`,
and the droidwidth closed-loop eval manifests surface checkpoint code commit,
stage label, action dimension, and global-video attention mode. Use those fields
to avoid mixing old-code evals with the bidirectional 32D teacher runs.
