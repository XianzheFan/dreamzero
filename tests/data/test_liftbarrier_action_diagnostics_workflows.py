from pathlib import Path
import subprocess

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
INSPECT_WORKFLOW_PATH = (
    REPO_ROOT / "osmo_workflows/robofactory/inspect_liftbarrier_action_magnitude_cpu.yaml"
)
COMPARE_WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/compare_liftbarrier_model_data_action_magnitude_cpu.yaml"
)
TRACE_WORKFLOW_PATH = (
    REPO_ROOT
    / "osmo_workflows/robofactory/analyze_liftbarrier_trace_diagnostics_cpu.yaml"
)


def _workflow_and_script(path: Path):
    with path.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["tasks"][0]
    return workflow, task, task["files"][0]["contents"]


def _workflow_task_and_files(path: Path):
    with path.open() as f:
        workflow = yaml.safe_load(f)
    task = workflow["workflow"]["tasks"][0]
    return workflow, task, {item["path"]: item["contents"] for item in task["files"]}


def _python_heredocs(script: str) -> list[str]:
    lines = script.splitlines()
    blocks: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip().endswith("<<'PY'"):
            continue
        start = i + 1
        end = start
        while end < len(lines) and lines[end].strip() != "PY":
            end += 1
        assert end < len(lines), f"missing PY heredoc terminator after line {i + 1}"
        blocks.append("\n".join(lines[start:end]))
    return blocks


def _assert_embedded_scripts_compile(script: str) -> None:
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    heredocs = _python_heredocs(script)
    assert heredocs
    for source in heredocs:
        compile(source, "<embedded workflow python>", "exec")


def test_inspect_action_magnitude_workflow_is_cpu_only_and_targets_liftbarrier_data():
    workflow, task, script = _workflow_and_script(INSPECT_WORKFLOW_PATH)

    defaults = workflow["default-values"]
    resources = workflow["workflow"]["resources"]["default"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert resources["gpu"] == 0
    assert resources["cpu"] == "{{resource_cpu}}"
    assert defaults["resource_cpu"] == "8"
    assert defaults["resource_memory"] == "64Gi"
    assert defaults["resource_storage"] == "250Gi"
    assert defaults["data_variant"] == "LiftBarrier-rf-500"
    assert defaults["data_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/data/robofactory_lerobot_v2/"
        "LiftBarrier-rf-500"
    )
    assert defaults["download_regex"] == r".*(meta/.*|data/.*\.parquet)$"
    assert task["name"] == "inspect-action-magnitude"
    assert task["image"] == "nvcr.io/nvidian/gear-h100-train:latest"

    assert "Refusing non-xianzhef URI" in script
    assert 'osmo data download --resume --regex "$DOWNLOAD_REGEX"' in script
    assert "liftbarrier_action_magnitude.json" in script
    assert "summary.txt" in script
    assert "horizon_p95=" in script
    assert "absolute_action_roundtrip_p99_error=" in script
    assert 'osmo data upload "${OUT_S3_URI}/" /workspace/action_magnitude_outputs' in script
    assert "> /workspace/action_magnitude_outputs/liftbarrier_action_magnitude.stdout.json" in script

    _assert_embedded_scripts_compile(script)


def test_compare_model_data_workflow_downloads_eval_npz_and_dataset_stats():
    workflow, task, script = _workflow_and_script(COMPARE_WORKFLOW_PATH)

    defaults = workflow["default-values"]
    resources = workflow["workflow"]["resources"]["default"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert resources["gpu"] == 0
    assert resources["cpu"] == "{{resource_cpu}}"
    assert defaults["resource_cpu"] == "8"
    assert defaults["resource_memory"] == "32Gi"
    assert defaults["resource_storage"] == "200Gi"
    assert defaults["eval_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-gamma-dw-readonlyfix2-lb500-50k-c50000-eval-h100-s1000-s10-best-full-xz-20260625"
    )
    assert defaults["data_stats_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-liftbarrier-action-magnitude-cpu-xz-20260625-v2/action_magnitude"
    )
    assert defaults["eval_download_regex"] == (
        r".*(action_dump_summary\.json|scale_sweep_summary\.json|action_dump/episode_[0-9]+\.npz)$"
    )
    assert defaults["data_stats_download_regex"] == (
        r".*(liftbarrier_action_magnitude\.json|summary\.txt)$"
    )
    assert task["name"] == "compare-model-data-action"

    assert "Refusing non-xianzhef URI" in script
    assert 'osmo data download --resume --regex "$EVAL_DOWNLOAD_REGEX"' in script
    assert 'osmo data download --resume --regex "$DATA_STATS_DOWNLOAD_REGEX"' in script
    assert 'find "$EVAL_PARENT" -path "*/action_dump/episode_*.npz" -type f | wc -l' in script
    assert "model_data_action_compare.json" in script
    assert "chunk_p95_over_data" in script
    assert 'osmo data upload "${OUT_S3_URI}/" /workspace/model_data_action_compare_outputs' in script

    _assert_embedded_scripts_compile(script)


def test_trace_diagnostics_workflow_downloads_eval_npz_and_is_cpu_only():
    workflow, task, files = _workflow_task_and_files(TRACE_WORKFLOW_PATH)
    script = files["/tmp/analyze_liftbarrier_trace.sh"]
    embedded_python = files["/tmp/analyze_liftbarrier_trace_diagnostics.py"]

    defaults = workflow["default-values"]
    resources = workflow["workflow"]["resources"]["default"]
    assert workflow["workflow"]["name"] == "{{workflow_name}}"
    assert resources["gpu"] == 0
    assert resources["cpu"] == "{{resource_cpu}}"
    assert defaults["resource_cpu"] == "8"
    assert defaults["resource_memory"] == "32Gi"
    assert defaults["resource_storage"] == "200Gi"
    assert defaults["eval_s3_uri"] == (
        "s3://GearHome/users/xianzhef/oci-migration/dreamzero_runs/"
        "dz-rf-gamma-dw-readonlyfix2-lb500-50k-c50000-eval-h100-s1000-s10-best-full-xz-20260625"
    )
    assert defaults["eval_download_regex"] == (
        r".*(action_dump/episode_[0-9]+\.npz|action_dump_summary\.json|scale_sweep_summary\.json)$"
    )
    assert defaults["contact_threshold"] == "0.05"
    assert defaults["decisive_threshold"] == "-0.5"
    assert task["name"] == "analyze-liftbarrier-trace"

    assert "Refusing non-xianzhef URI" in script
    assert 'osmo data download --resume --regex "$EVAL_DOWNLOAD_REGEX"' in script
    assert 'find "$EVAL_PARENT" -path "*/action_dump/episode_*.npz" -type f | wc -l' in script
    assert "liftbarrier_trace_diagnostics.json" in script
    assert "trace_diagnostics" in script
    assert "left_before_close_p50" in embedded_python
    assert "decisive_close_before_contact_count" in embedded_python
    assert "curve_step" in embedded_python

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    compile(embedded_python, "<embedded trace diagnostics python>", "exec")
