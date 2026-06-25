"""Upload a committed DreamZero source snapshot for OSMO workflows.

Training workflows intentionally require both ``code_s3_uri`` and
``expected_code_commit``. This helper creates the matching code cache from a
clean git commit, writes an ``OSMO_CODE_COMMIT`` marker into the uploaded root,
and prints the exact submission values.

The script refuses dirty worktrees by default. That keeps the code marker
honest: the training container can compare ``expected_code_commit`` to the
uploaded marker and fail fast if the wrong cache is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys
import tempfile


DEFAULT_PREFIX = "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/oci-migration"


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def git_output(repo: Path, *args: str) -> str:
    return _run(["git", *args], cwd=repo, capture=True).stdout.strip()


def repo_root(path: Path) -> Path:
    return Path(git_output(path, "rev-parse", "--show-toplevel"))


def require_clean_worktree(repo: Path) -> None:
    status = git_output(repo, "status", "--porcelain")
    if status:
        raise SystemExit(
            "Refusing to upload a dirty worktree. Commit the intended source "
            "state first so OSMO_CODE_COMMIT matches the uploaded code.\n\n"
            + status
        )


def validate_xianzhef_uri(uri: str) -> None:
    allowed = (
        "s3://GearHome/users/xianzhef/",
        "swift://pdx.s8k.io/AUTH_team-gear/datasets/users/xianzhef/",
    )
    if not uri.startswith(allowed):
        raise SystemExit(f"Refusing non-xianzhef upload URI: {uri}")


def stage_code(repo: Path, stage_dir: Path, commit: str) -> None:
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=str(repo),
        stdout=subprocess.PIPE,
    )
    if archive.stdout is None:
        raise RuntimeError("git archive stdout was not captured")
    tar = subprocess.run(
        ["tar", "-xf", "-", "-C", str(stage_dir)],
        stdin=archive.stdout,
        check=True,
    )
    archive.stdout.close()
    archive_status = archive.wait()
    if archive_status != 0 or tar.returncode != 0:
        raise subprocess.CalledProcessError(archive_status, ["git", "archive", commit])

    (stage_dir / "OSMO_CODE_COMMIT").write_text(commit + "\n", encoding="utf-8")


def upload_code_cache(repo: Path, uri: str, *, dry_run: bool) -> tuple[str, str]:
    commit = git_output(repo, "rev-parse", "HEAD")
    require_clean_worktree(repo)
    validate_xianzhef_uri(uri)

    with tempfile.TemporaryDirectory(prefix="dreamzero_code_cache_") as tmp:
        stage_dir = Path(tmp) / "source"
        stage_dir.mkdir()
        stage_code(repo, stage_dir, commit)
        if dry_run:
            file_count = sum(1 for p in stage_dir.rglob("*") if p.is_file())
            print(f"dry_run=true")
            print(f"staged_dir={stage_dir}")
            print(f"staged_files={file_count}")
        else:
            _run(["osmo", "data", "upload", uri.rstrip("/") + "/", str(stage_dir)])

    return uri.rstrip("/"), commit


def default_cache_name(repo: Path, prefix: str) -> str:
    branch = git_output(repo, "rev-parse", "--abbrev-ref", "HEAD")
    short = git_output(repo, "rev-parse", "--short=8", "HEAD")
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    safe_branch = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in branch)
    return f"{prefix}_{safe_branch}_{short}_{today}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository path. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Destination code cache URI. Defaults under the xianzhef OSMO Swift prefix.",
    )
    parser.add_argument(
        "--name-prefix",
        default="dreamzero_code",
        help="Name prefix used when --uri is omitted.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = repo_root(args.repo)
    name = default_cache_name(repo, args.name_prefix)
    uri = args.uri or f"{DEFAULT_PREFIX}/{name}"
    code_s3_uri, expected_code_commit = upload_code_cache(repo, uri, dry_run=args.dry_run)
    print(f"code_s3_uri={code_s3_uri}")
    print(f"expected_code_commit={expected_code_commit}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {' '.join(map(str, exc.cmd))}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
