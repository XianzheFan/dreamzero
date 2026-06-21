# RoboFactory DreamZero OSMO Workflows

Reusable OSMO workflow specs for the DreamZero RoboFactory LiftBarrier runs.

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
predicted-video future MAE/drift.
