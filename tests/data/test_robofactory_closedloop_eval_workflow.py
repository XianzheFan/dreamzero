from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_GAIN_WORKFLOW = (
    REPO_ROOT
    / "osmo_workflows/robofactory/"
    / "closedloop_liftbarrier_motionw4_c1500_safegain_pred_3seed_20260618.yaml"
)


def _workflow_script(path: Path) -> str:
    workflow = yaml.safe_load(path.read_text())
    task = workflow["workflow"]["groups"][0]["tasks"][0]
    return task["files"][0]["contents"]


def test_liftbarrier_safe_gain_workflow_keeps_slew_limited_absolute_gain():
    script = _workflow_script(SAFE_GAIN_WORKFLOW)

    assert 'DREAMZERO_GIT_COMMIT="2b112162f2dcdcd2b30bf91d48132e08bdd2ebe8"' in script
    assert '--reset-causal-state-each-infer \\' in script
    assert 'JOINT_DELTA_SCALES="4.0 8.0 12.0"' in script
    assert 'JOINT_DELTA_OUTPUT_CLIP="0.35"' in script
    assert 'JOINT_TARGET_SLEW_RATES="0.04"' in script
    assert '--joint-delta-output-clip "$JOINT_DELTA_OUTPUT_CLIP" \\' in script
    assert "--unsafe-absolute-joint-delta-scale \\" in script
