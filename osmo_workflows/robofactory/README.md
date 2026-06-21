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
  --pool groot-h100-02 \
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

## 2k Checkpoint Eval Grid

The droidwidth teacher training workflow saves checkpoints every 2000 optimizer
steps. Keep closed-loop evals on the same 2k cadence by generating one H100 eval
workflow per checkpoint instead of running all checkpoints inside one long
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

The helper uses `osmo workflow submit` with `--set-string` overrides for
`workflow_name`, `run_name`, `ckpt_setting`, and `local_eval_ckpt_root`, so each
checkpoint writes to an isolated eval run. The default cadence is:

```text
checkpoint-2000, checkpoint-4000, ..., checkpoint-50000
```

Do not pack the whole grid into one eval workflow. A single H100 closed-loop
eval already runs model load, action diagnostics, video prediction diagnostics,
and a small physical sweep; separate OSMO workflows make retries and queueing
cleaner.

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
only poor visual modeling.

New gamma training checkpoints include `experiment_cfg/runtime_provenance.json`,
and the droidwidth closed-loop eval manifests surface checkpoint code commit,
stage label, action dimension, and global-video attention mode. Use those fields
to avoid mixing old-code evals with the bidirectional 32D teacher runs.
