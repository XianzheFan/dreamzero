# LiftBarrier Teacher Action Diagnostics

Date: 2026-06-25

This note records the first diagnostic pass for the 0% LiftBarrier teacher
failure. It intentionally does not change training or eval code. The immediate
goal is to pin down whether the failure is action-magnitude under-commanding,
normalization/denormalization mismatch, temporal/window mismatch, or contact
planning.

## Runs Checked

Primary same-config evals:

| checkpoint | workflow | status | eval |
| --- | --- | --- | --- |
| `checkpoint-12000` | `dz-rf-gamma-dw-rofix2-lb500-c12000-eval-h100-s1000-xz-readonlyfix2-20260623-latest-h10002-1` | `COMPLETED` | one seed, wider scale/profile sweep |
| `checkpoint-20000` | `dz-rf-gamma-dw-rofix2-lb500-c20000-eval-h100-s1000-xz-readonlyfix2-20260624-latest-1` | `COMPLETED` | one seed, wider scale/profile sweep |
| `checkpoint-32000` | `dz-rf-gamma-dw-readonlyfix2-lb500-50k-c32000-eval-h100-s1000-teacher-xz-20260624teacherb-1` | `COMPLETED` | one seed, wider scale/profile sweep |
| `checkpoint-48000` | `dz-rf-gamma-dw-readonlyfix2-lb500-50k-c48000-eval-h100-s1000-s10-best-full-xz-20260625-1` | `COMPLETED` | seeds `1000..1009`, `replan=24,12`, `scale=1.0` |
| `checkpoint-50000` | `dz-rf-gamma-dw-readonlyfix2-lb500-50k-c50000-eval-h100-s1000-s10-best-full-xz-20260625-1` | `COMPLETED` | seeds `1000..1009`, `replan=24,12`, `scale=1.0` |
| dataset target inspection | `dz-rf-liftbarrier-action-magnitude-cpu-xz-20260625-v2-1` | `COMPLETED` | OSMO CPU scan of all 500 LiftBarrier episodes |
| model-vs-data comparison | `dz-rf-liftbarrier-model-data-action-compare-cpu-xz-20260625-1` | `COMPLETED` | OSMO CPU comparison of 50k action dumps vs dataset target stats |
| full-finetune teacher | `dz-rf-sg-gamma-dwteacher-fullft-lb500-30k-xz-20260625-1` | `RUNNING` | 8xH100 full-finetune teacher, 30k max steps, 2k checkpoint cadence |

The `checkpoint-50000` eval completed with exit code 0 and uploaded outputs to:

```text
s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/dz-rf-gamma-dw-readonlyfix2-lb500-50k-c50000-eval-h100-s1000-s10-best-full-xz-20260625/
```

Local AWS/OSMO data access does not currently have credentials for direct S3
download, so the 10-seed rows below are taken from the OSMO eval logs, where the
workflow prints the same physical summary table written to
`action_dump_summary.json`.

## Dataset Action Target Distribution

The OSMO CPU workflow
`dz-rf-liftbarrier-action-magnitude-cpu-xz-20260625-v2-1` completed on
`groot-l40-01` with exit code 0. It read all 500 authoritative RoboFactory
LiftBarrier episodes from:

```text
s3://GearHome/users/xianzhef/oci-migration/data/robofactory_lerobot_v2/LiftBarrier-rf-500
```

The uploaded diagnostic artifact is:

```text
s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/dz-rf-liftbarrier-action-magnitude-cpu-xz-20260625-v2/action_magnitude/
```

Key results:

| metric | value | read |
| --- | ---: | --- |
| episodes read | `500` | full dataset scan |
| action dim / state dim | `16 / 16` | two 8-D arms, grippers at 7 and 15 |
| joint dims | `[0..6, 8..14]` | 14 joint dimensions |
| one-step target-current p50 / p95 | `0.000404 / 0.048662` | data's immediate next action can be small |
| horizon-24 target-current p50 / p95 | `0.006510 / 0.259733` | chunk-level target is much larger |
| left/right horizon-24 p95 | `0.274633 / 0.245864` | both arms require large chunk deltas |
| horizon offset 23 p50 / p95 | `0.014662 / 0.360589` | later chunk targets are very large |
| first close step p50 / p95 | `33 / 37` | dataset closes earlier than current eval override step 52 |
| absolute action q99 clip fraction | `0.017495` | stats cover almost all absolute actions |
| absolute action round-trip p99 error | `0.002396` | q99 clip creates only small tail error |
| `relative_stats_dreamzero.json` | absent | relative-action round-trip not testable from this dataset artifact |

Against the 48k/50k teacher logs and the direct 50k action-dump comparison, the
model is clearly under-commanding at the chunk level. This confirms the
action-magnitude gap as a real data-vs-model mismatch, not just a success-rate
artifact. It also explains why inference-time scaling can make the task look
less impossible while causing jerk: scale is compensating for a learned
under-commanding bias instead of fixing the policy.

## Model-Vs-Data Action Comparison

The OSMO CPU workflow
`dz-rf-liftbarrier-model-data-action-compare-cpu-xz-20260625-1` completed on
`groot-l40-01` with exit code 0. It downloaded 20 action-dump episodes from the
50k eval artifact and compared them against the dataset target distribution
above. The uploaded diagnostic artifact is:

```text
s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/dz-rf-liftbarrier-model-data-action-compare-cpu-xz-20260625/model_data_action_compare/
```

Key model-vs-data results:

| variant | replan | episodes | first p50 / p95 | chunk p50 / p95 | last-offset p50 / p95 | exec step p50 / p95 | chunk p95 / data horizon p95 | last p95 / data last p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rp12_jscale_clip_smooth_smooth` | `12` | `10` | `0.008829 / 0.085735` | `0.009807 / 0.094490` | `0.020978 / 0.171876` | `0.004092 / 0.034929` | `0.364` | `0.477` |
| `rp24_jscale_clip_smooth_smooth` | `24` | `10` | `0.007666 / 0.077071` | `0.007367 / 0.057789` | `0.015635 / 0.116865` | `0.004027 / 0.030534` | `0.222` | `0.324` |

Read: even at the 95th percentile, the model's full chunk target-current
magnitude is only `22-36%` of the dataset horizon target p95, and the final
chunk offset is only `32-48%` of the dataset's final-offset p95. This is the
clearest evidence so far that the teacher has learned a conservative/shrunken
action distribution. The executed step p95 is also small (`0.030-0.035`),
which matches the qualitative failure where both wrists get closer but do not
commit enough motion before the close phase.

## Stage1 Trend From Action Quality

These rows use the same inference variant where available:
`vpred_action`, `scale=1.0`, `clip=0.35`, `slew=0.35`, `accel=0.08`,
`profile=smooth`, `blend=4`, `ensemble=0.6`, strict lift. The 12k/20k/32k rows
are one-seed sweeps, while 48k/50k are 10-seed evals, so the absolute numbers
are not perfectly comparable. The trend is still useful for deciding whether
stage1 should blindly run to 50k.

`replan=24`:

| checkpoint | episodes | success | L/R target min | joint mean/max | pred joint | raw sat | clamp mean | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12k | 1 | `0/1` | `0.138 / 0.123` | `0.019 / 0.201` | `0.024` | `0.188` | `0.183` | far from target, heavy clamp/saturation |
| 20k | 1 | `0/1` | `0.057 / 0.057` | `0.009 / 0.167` | `0.011` | `0.015` | `0.007` | big action-quality improvement |
| 32k | 1 | `0/1` | `0.066 / 0.049` | `0.010 / 0.122` | `0.012` | `0.015` | `0.006` | similar to 20k |
| 48k | 10 | `0/10` | `0.070 / 0.082` | `0.008 / 0.131` | `0.010` | `0.009` | `0.003` | no success, plateau |
| 50k | 10 | `0/10` | `0.070 / 0.081` | `0.008 / 0.130` | `0.010` | `0.010` | `0.003` | flat vs 48k |

`replan=12`:

| checkpoint | episodes | success | L/R target min | joint mean/max | pred joint | raw sat | clamp mean | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12k | 1 | `0/1` | `0.137 / 0.123` | `0.015 / 0.121` | `0.025` | `0.203` | `0.207` | saturated/clamped and still far |
| 20k | 1 | `0/1` | `0.061 / 0.086` | `0.010 / 0.115` | `0.017` | `0.041` | `0.021` | much cleaner than 12k |
| 32k | 1 | `0/1` | `0.061 / 0.067` | `0.009 / 0.100` | `0.015` | `0.034` | `0.016` | similar to 20k |
| 48k | 10 | `0/10` | `0.083 / 0.095` | `0.009 / 0.122` | `0.014` | `0.030` | `0.012` | no contact improvement |
| 50k | 10 | `0/10` | `0.083 / 0.094` | `0.009 / 0.102` | `0.014` | `0.031` | `0.012` | flat vs 48k |

Read: the model gets much cleaner between 12k and 20k/32k, especially on
raw normalized saturation and clamp. After that, more stage1 steps do not
translate into better contact, grasp, lift, or lower TCP target distance. Based
on the current evidence, stage1 should be eval-gated around 20k/30k rather than
assumed to need 50k. If the 20k/30k 10-seed gate is still 0% with the same
plateau, the right next move is to change the objective/training setup instead
of spending more teacher steps.

## 48k To 50k Same-Config Comparison

`replan=24`:

| metric | 48k | 50k | read |
| --- | ---: | ---: | --- |
| success | `0/10` | `0/10` | no improvement |
| episode length | `300` each | `300` each | no early success |
| L/R target min | `0.070 / 0.082` | `0.070 / 0.081` | unchanged, still ~7-8 cm away |
| success margin best | `-0.143` | `-0.143` | unchanged |
| L/R grasp episodes | `0 / 0` | `0 / 0` | no grasp |
| joint step delta mean/max | `0.008 / 0.131` | `0.008 / 0.130` | unchanged |
| pred joint delta mean | `0.010` | `0.010` | unchanged |
| jerk mean | `0.023` | `0.024` | unchanged/slightly higher |
| accel ratio | `1.609` | `1.609` | unchanged |
| replan boundary max | `0.051` | `0.051` | unchanged |
| model boundary max | `0.183` | `0.205` | slightly worse |
| raw saturation | `0.009` | `0.010` | unchanged/slightly higher |
| clamp mean/max | `0.003 / 1.375` | `0.003 / 1.562` | max slightly worse |

`replan=12`:

| metric | 48k | 50k | read |
| --- | ---: | ---: | --- |
| success | `0/10` | `0/10` | no improvement |
| episode length | `300` each | `300` each | no early success |
| L/R target min | `0.083 / 0.095` | `0.083 / 0.094` | unchanged, worse than `replan=24` |
| success margin best | `-0.143` | `-0.143` | unchanged |
| L/R grasp episodes | `0 / 0` | `0 / 0` | no grasp |
| joint step delta mean/max | `0.009 / 0.122` | `0.009 / 0.102` | max lower, not useful |
| pred joint delta mean | `0.014` | `0.014` | unchanged |
| jerk mean | `0.025` | `0.025` | unchanged |
| accel ratio | `1.623` | `1.625` | unchanged |
| replan boundary max | `0.057` | `0.056` | unchanged |
| model boundary max | `0.340` | `0.333` | slightly lower |
| raw saturation | `0.030` | `0.031` | unchanged/slightly higher |
| clamp mean/max | `0.012 / 3.125` | `0.012 / 3.375` | max slightly worse |

Result: there is no evidence that continuing the same LoRA teacher from 48k to
50k improves either task outcome or action quality. The failure signature is
flat across those checkpoints.

## Failure Signature

The current teacher:

- gets both arms closer than the reset pose, but plateaus around `0.07-0.09 m`
  from the grasp targets;
- closes both grippers deterministically at step `52`, but `left_grasp_eps` and
  `right_grasp_eps` stay `0`;
- never lifts the barrier: `success_margin_max_best = -0.143`;
- outputs small executed joint deltas: `joint_mean ~= 0.008-0.009`;
- is not being dominated by output clipping: raw saturation and clamp means are
  low;
- does not become better with more stage1 steps.

This supports "cannot reach/contact robustly before close" more strongly than
"teacher has not trained long enough".

## Scale Sweep Context

Older local one-seed scale-sweep artifacts show the diagnostic role of action
amplification. In the `scale=1.0` reset-cache sample:

```text
first_cmd_delta_abs p95 ~= 0.0309
exec_joint_step_delta p95 ~= 0.0144
left target min ~= 0.061
right target min ~= 0.096
left/right first decisive close = 52
left/right grasp count = 0
barrier z max = 0.007
```

Raising scale increases joint movement, but those runs are single-seed
diagnostics and should not define the main metric. If scale helps, it should be
treated as evidence of an action-magnitude gap, not as a deployment fix.

## Normalization / Denormalization Audit

The absolute q99 normalization round-trip is proven from the dataset
`meta/stats.json`: p99 error is `0.002396`, with `1.75%` clip fraction. What is
not proven by that dataset artifact alone is relative-action normalization,
because `meta/relative_stats_dreamzero.json` is not present in the raw
downloaded LiftBarrier dataset.

The train/eval code path closes that gap for actual checkpoints:

- training config uses `relative_action=true`,
  `relative_action_keys=[panda0_joint_pos, panda1_joint_pos]`, and
  `use_global_metadata=false`;
- `LeRobotSingleDataset` computes missing relative stats locally from
  target-current joint offsets, inserts those stats into
  `train_dataset.merged_metadata`, and `BaseExperiment` writes that merged
  metadata to `experiment_cfg/metadata.json`;
- the bimanual eval server loads the checkpoint's `metadata.json`, denormalizes
  the model's q99-normalized action output with those stats, and raises if
  required `q01/q99` stats are missing or shape-mismatched;
- for `relative_action` checkpoints, the server declares
  `action_representation=absolute_qpos` and adds the latest observed qpos back
  to the denormalized joint offsets for both arms before the eval client applies
  any scale/blend/slew logic;
- the action dumps therefore contain `action_norm_raw` /
  `action_norm_clipped` in model-normalized space, while `pred_chunk` is the
  physical 16-D denormalized action chunk that the eval client consumes.

Read: a silent eval-side denormalization shrink from missing or mismatched
LiftBarrier stats is now unlikely. If stats are absent, the server should fail
instead of producing small actions. The under-commanding evidence is therefore
more consistent with the learned action distribution than with an eval
normalization bug.

For droid-width checkpoints, the model action head is padded to 32 dimensions
per agent (`MODEL_ACTION_DIM=32`, `AGENT_ACTION_PAD_DIM=32`), but the physical
RoboFactory action used by eval is still the first 8 dimensions per arm:
7 joints plus 1 gripper, flattened to a 16-D environment action. The eval tests
cover this path and preserve the full 64-D normalized prediction only as debug
metadata.

## What Is Still Not Proven

The model-side p50/p95 action magnitude comparison is now complete for the 50k
eval action dumps. Remaining uncertainty is no longer "is there an action-size
gap"; it is "which part of training caused the action-size gap".

The next unresolved measurement from the original checklist is the
TCP-to-grasp-target trajectory and gripper close timing. A CPU-only offline
trace analyzer has been added for the existing eval dumps:

```text
scripts/eval/analyze_liftbarrier_trace_diagnostics.py
osmo_workflows/robofactory/analyze_liftbarrier_trace_diagnostics_cpu.yaml
```

It reads `env_trace` from `action_dump/episode_*.npz` and reports:

- target-distance curves at fixed steps;
- start/min/final/last-window TCP-to-grasp-target distances;
- the TCP-to-target distance immediately before and after the decisive close
  command;
- how many episodes close while still farther than the contact threshold;
- actual grasp counts.

The workflow has passed local static tests and `osmo workflow validate` on
`groot-l40-01`, but has not yet been submitted.

## Current Diagnosis

Most likely primary issue:

```text
action-magnitude / closed-loop commitment gap
```

Evidence:

- same checkpoint family stays at 0% through the completed 48k/50k 10-seed
  evals;
- earlier completed 12k/20k/32k one-seed sweeps show action quality improves
  early, then plateaus rather than improving monotonically toward 50k;
- authoritative dataset horizon-24 target-current p95 is `0.260` overall
  (`0.361` at offset 23), while the 50k model action dumps are only
  `0.058-0.094` at chunk p95 and `0.117-0.172` at final-offset p95;
- 48k and 50k action-quality metrics are flat;
- gripper close is not missing, but close happens while TCPs are still outside
  the contact/grasp region;
- executed joint deltas are small and not saturation-limited;
- scale amplification is a plausible diagnostic, but it also increases jerk and
  should not become the metric.

Not yet excluded:

- temporal/horizon mismatch in the actual action target distribution;
- contact geometry/planning error despite adequate action magnitude.

## Next Concrete Step

The dataset-side and model-vs-data diagnostics have both been run with
OSMO-side data credentials:

```text
dz-rf-liftbarrier-action-magnitude-cpu-xz-20260625-v2-1
dz-rf-liftbarrier-model-data-action-compare-cpu-xz-20260625-1
```

The exact model p50/p95 matches the qualitative log diagnosis: the 50k teacher
is under-commanding by a large margin. Prioritize the fixes in this order:

1. Full-finetune teacher at `scale=1.0`, not inference-time action scaling.
2. Action objective/loss weighting review, especially anything that rewards
   small deltas.
3. Time-window/contact-phase eval: the dataset first close p50 is `33`, while
   the current eval override closes at step `52`. The trace diagnostics
   workflow above is the prepared measurement for this item.

The droidwidth teacher training path now exposes the necessary full-finetune
switches instead of hard-coding LoRA:

```text
TRAIN_ARCHITECTURE=full
SAVE_LORA_ONLY=false
DEFER_LORA_INJECTION=false
SKIP_COMPONENT_LOADING=true
GRAD_CKPT=true
ACTION_DELTA_LOSS_WEIGHT=0.0
```

The default workflow remains LoRA for continuity, but a full-finetune teacher
has now been submitted through:

```text
osmo_workflows/robofactory/train_liftbarrier_gamma_droidwidth_teacher.yaml
```

Run:

```text
dz-rf-sg-gamma-dwteacher-fullft-lb500-30k-xz-20260625-1
```

Submission details:

```text
pool=groot-h100-02
priority=LOW
code_commit=72659ca34e5cf7a4c5a3c9604473dc2bcbe36073
code_s3_uri=swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/dreamzero_code_gamma_72659ca3_20260625
stage1_max_steps=30000
train_architecture=full
save_lora_only=false
defer_lora_injection=false
skip_component_loading=true
grad_ckpt=true
action_delta_loss_weight=0.0
save_total_limit=6
```

Early logs confirm the code cache commit self-check passed, the job started
from DreamZero-DROID with no restore run, `MODEL_ACTION_DIM=32`,
`AGENT_ACTION_PAD_DIM=32`, `save_steps=2000`, and 8 distributed ranks reached
`Run name: teacher`. This is the cleanest next training experiment: compare it
to the LoRA teacher using the same 10-seed `scale=1.0` eval, then rerun the
model-vs-data action comparison to see whether chunk p95 moves toward the
dataset horizon p95. Keeping `action_delta_loss_weight=0.0` avoids adding an
adjacent-action smoothing loss while the main diagnosis is under-commanding.
If memory is tight, add
`deepspeed_cfg=groot/vla/configs/deepspeed/zero2_offload.json` as a throughput
tradeoff rather than changing the modeling setup.

Use the fixed eval preset for the first full-finetune gate:

```text
python scripts/eval/submit_robofactory_droidwidth_eval_grid.py \
  --preset fullft-gate \
  --tag YYYYMMDD \
  --only-ready \
  --skip-existing \
  --ready-check-source train-workflow \
  --ready-train-workflow dz-rf-sg-gamma-dwteacher-fullft-lb500-30k-xz-YYYYMMDD-1 \
  --submit
```

The preset evaluates `checkpoint-10000`, `checkpoint-20000`, and
`checkpoint-30000` with 10 seeds, `joint_delta_scales=1.0`, action rollout
mode, `replan=24/12`, and raw/smooth profiles. It disables the extra accel
limit sweep so the first decision is about whether the teacher learned usable
actions at native scale, not whether an execution regularizer can hide a weak
action distribution.

Close timing should stay out of that primary metric. The H100 eval workflow now
exposes `gripper_close_pairs` as a template override while keeping the default
at `52:52`. For a one-off contact-phase diagnostic after a checkpoint reaches
near the target, run an explicit step subset with:

```text
--close-timing-diagnostic
```

That adds `gripper_close_pairs="33:33 52:52"` so the dataset-like close timing
(`33:33`, near the dataset first-close p50) can be compared against the current
standing eval close timing (`52:52`) without redefining the main 10-seed gate.

Before submitting that workflow, commit the intended source state and create a
matching OSMO code cache:

```text
python scripts/osmo/upload_code_cache.py
```

The helper refuses dirty worktrees and prints the `code_s3_uri` plus
`expected_code_commit` values required by the training workflow.

The Stage3 self-forcing warmup footgun has already been fixed in the current
code path: `BaseTrainer.training_step` pushes `state.global_step` into the
unwrapped action head before forward, and `_self_forcing_train_enabled()`
returns `False` while `global_step < self_forcing_warmup_steps`. Coverage:

```text
tests/experiment/test_action_head_global_step.py
tests/model/test_wan_action_head_config.py::test_self_forcing_train_enabled_respects_warmup_steps
```

The reusable dataset workflow is:

```text
osmo_workflows/robofactory/inspect_liftbarrier_action_magnitude_cpu.yaml
```

It has passed YAML parsing, embedded bash/Python syntax checks, and
`osmo workflow validate`. Submit it on `groot-l40-01` or another pool that
accepts `gpu: 0`; `groot-h100-02` rejects `gpu: 0` and requires `gpu: 8`.

The model-vs-data comparison workflow is:

```text
osmo_workflows/robofactory/compare_liftbarrier_model_data_action_magnitude_cpu.yaml
```

It downloads the 50k eval `action_dump/episode_*.npz` files plus the dataset
stats artifact above, then reports model `first_cmd`, full-horizon
`pred_chunk-current`, final-offset, and executed step p50/p95 next to the
dataset p50/p95. It has passed YAML parsing, embedded bash/Python syntax checks,
`osmo workflow validate`, and a local synthetic `.npz` smoke test. The submitted
workflow completed successfully as
`dz-rf-liftbarrier-model-data-action-compare-cpu-xz-20260625-1`.

The same comparison logic is also available locally as:

```text
scripts/eval/compare_liftbarrier_model_data_action_magnitude.py
```

with focused coverage in:

```text
tests/eval/test_compare_liftbarrier_model_data_action_magnitude.py
tests/data/test_liftbarrier_action_diagnostics_workflows.py
tests/eval/test_analyze_liftbarrier_trace_diagnostics.py
```

The local test currently verifies variant grouping, model p50/p95 extraction,
dataset ratio calculation, the missing-action-dump failure path, and static
workflow invariants for the CPU diagnostic workflows. The trace test also
verifies plateau/min/final distance extraction, close-before-contact counts,
grasp counts, and step-curve summary formatting on synthetic dumps.
