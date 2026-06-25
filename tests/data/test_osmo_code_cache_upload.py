import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/osmo/upload_code_cache.py"


def _load_module():
    module_name = "upload_code_cache_for_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(args, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (repo / "script.py").write_text("print('ok')\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)
    return repo


def test_stage_code_writes_commit_marker_and_tracked_files(tmp_path):
    upload = _load_module()
    repo = _make_repo(tmp_path)
    commit = upload.git_output(repo, "rev-parse", "HEAD")
    stage = tmp_path / "stage"
    stage.mkdir()

    upload.stage_code(repo, stage, commit)

    assert (stage / "pyproject.toml").is_file()
    assert (stage / "script.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert (stage / "OSMO_CODE_COMMIT").read_text(encoding="utf-8") == commit + "\n"


def test_upload_code_cache_dry_run_refuses_dirty_worktree(tmp_path):
    upload = _load_module()
    repo = _make_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to upload a dirty worktree"):
        upload.upload_code_cache(
            repo,
            "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration/demo",
            dry_run=True,
        )


def test_upload_code_cache_refuses_non_xianzhef_uri(tmp_path):
    upload = _load_module()
    repo = _make_repo(tmp_path)

    with pytest.raises(SystemExit, match="Refusing non-xianzhef upload URI"):
        upload.upload_code_cache(
            repo,
            "s3://GearHome/users/someone_else/demo",
            dry_run=True,
        )


def test_upload_code_cache_script_compiles():
    compile(SCRIPT_PATH.read_text(encoding="utf-8"), str(SCRIPT_PATH), "exec")
