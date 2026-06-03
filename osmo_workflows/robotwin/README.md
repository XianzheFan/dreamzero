# RoboTwin DreamZero OSMO Workflows

Reusable OSMO workflow specs for the DreamZero RoboTwin `stack_blocks_two`
Franka bimanual run.

## Files

- `prepare_stack_blocks_two_dataset.yaml`: prepares/persists the RoboTwin
  `stack_blocks_two` LeRobot v2 dataset cache.
- `train_stack_blocks_two_shared_global.yaml`: trains DreamZero from the DROID
  checkpoint with the multi-agent shared-global configuration and W&B reporting
  through the OSMO `wandb-api-key` credential.
- `eval_stack_blocks_two_ckpt7000_l40.yaml`: evaluates the data500 gripperfix
  training run's `checkpoint-50000` on L40 by default, using persistent AMLFS
  caches and runtime patches for RoboTwin instruction placeholders and gripper
  action handling.
- `probe_stack_blocks_two_metadata.yaml`: lightweight probe used to inspect the
  local RoboTwin task metadata and instruction schema.

## Submit

```bash
osmo workflow submit osmo_workflows/robotwin/prepare_stack_blocks_two_dataset.yaml --pool groot-l40-04
osmo workflow submit osmo_workflows/robotwin/train_stack_blocks_two_shared_global.yaml --pool groot-h100-02
osmo workflow submit osmo_workflows/robotwin/eval_stack_blocks_two_ckpt7000_l40.yaml --pool groot-l40-04
```

To expand the post-training set without overwriting the validated 500-episode
converted dataset, submit the same prepare workflow with isolated raw,
converted, and workflow-local RoboTwin cache prefixes:

```bash
osmo workflow submit osmo_workflows/robotwin/prepare_stack_blocks_two_dataset.yaml \
  --pool groot-l40-04 \
  --set workflow_name=robotwin-stack-blocks-two-dataset1000-l40-xianzhef-20260603 \
  data_variant=stack_blocks_two-rt-1000 \
  target_episodes=1000 \
  raw_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/RoboTwin/data/stack_blocks_two/demo_full_franka_1000 \
  raw_bootstrap_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/RoboTwin/data/stack_blocks_two/demo_full_franka \
  converted_data_s3_uri=s3://GearHome/users/xianzhef/oci-migration/data/robotwin_lerobot_v2/stack_blocks_two-rt-1000
```

The eval workflow intentionally uses L40 because the RoboTwin/SAPIEN runtime and
AMLFS cache are already validated there. H100 remains the better choice for the
training workflow.
