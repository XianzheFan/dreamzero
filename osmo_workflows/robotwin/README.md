# RoboTwin DreamZero OSMO Workflows

Reusable OSMO workflow specs for the DreamZero RoboTwin `stack_blocks_two`
Franka bimanual run.

## Files

- `prepare_stack_blocks_two_dataset.yaml`: prepares/persists the RoboTwin
  `stack_blocks_two` LeRobot v2 dataset cache.
- `train_stack_blocks_two_shared_global.yaml`: trains DreamZero from the DROID
  checkpoint with the multi-agent shared-global configuration and W&B reporting
  through the OSMO `wandb-api-key` credential.
- `eval_stack_blocks_two_ckpt7000_l40.yaml`: evaluates `checkpoint-7000` on L40,
  using persistent AMLFS caches and a runtime patch that fills the official
  RoboTwin `{A}/{B}/{a}/{b}` instruction placeholders from the live
  `block1`/`block2` poses.
- `probe_stack_blocks_two_metadata.yaml`: lightweight probe used to inspect the
  local RoboTwin task metadata and instruction schema.

## Submit

```bash
osmo workflow submit osmo_workflows/robotwin/train_stack_blocks_two_shared_global.yaml --pool groot-h100-02
osmo workflow submit osmo_workflows/robotwin/eval_stack_blocks_two_ckpt7000_l40.yaml --pool groot-l40-04
```

The eval workflow intentionally uses L40 because the RoboTwin/SAPIEN runtime and
AMLFS cache are already validated there. H100 remains the better choice for the
training workflow.
