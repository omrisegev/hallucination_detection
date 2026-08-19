#!/usr/bin/env python3
"""Build an auditable opening-state manifest for a research consolidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def git_text(root: Path, *args: str) -> str:
    return git_bytes(root, *args).decode("utf-8", errors="surrogateescape").rstrip("\n")


def nul_paths(root: Path, *args: str) -> list[str]:
    raw = git_bytes(root, *args)
    return [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    stat = path.stat()
    return {
        "path": relative,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def parse_status(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw in git_bytes(root, "status", "--porcelain=v1", "-z").split(b"\0"):
        if not raw:
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        records.append({"xy": text[:2], "path": text[3:]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(git_text(Path.cwd(), "rev-parse", "--show-toplevel"))
    output = (root / args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {output}")

    generated_tool = "scripts/build_research_consolidation_manifest.py"
    excluded_paths = {generated_tool, output.relative_to(root).as_posix()}
    status = [entry for entry in parse_status(root) if entry["path"] not in excluded_paths]
    untracked = [
        path
        for path in nul_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
        if path not in excluded_paths
    ]
    modified = nul_paths(root, "diff", "--name-only", "-z")
    staged = nul_paths(root, "diff", "--cached", "--name-only", "-z")
    refs = {}
    for ref in (
        "refs/heads/master",
        "refs/heads/codex/consolidate-research-2026-08-19",
        "refs/heads/codex/whitebox-layer-fusion",
        "refs/remotes/origin/master",
        "refs/remotes/origin/paper-exact/acquisition-v1",
        "refs/remotes/origin/whitebox/per-layer-views",
    ):
        try:
            refs[ref] = git_text(root, "rev-parse", ref)
        except subprocess.CalledProcessError:
            refs[ref] = None

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(root),
        "branch": git_text(root, "branch", "--show-current"),
        "head": git_text(root, "rev-parse", "HEAD"),
        "refs": refs,
        "remote": git_text(root, "remote", "get-url", "origin"),
        "status": status,
        "excluded_generated_paths": sorted(excluded_paths),
        "counts": {
            "status_entries": len(status),
            "modified_files": len(modified),
            "staged_files": len(staged),
            "untracked_files": len(untracked),
        },
        "modified_files": [file_record(root, path) for path in modified],
        "staged_files": [file_record(root, path) for path in staged],
        "untracked_files": [file_record(root, path) for path in untracked],
        "branches": git_text(
            root,
            "for-each-ref",
            "--format=%(refname)%09%(objectname)%09%(upstream:short)",
            "refs/heads",
            "refs/remotes/origin",
        ).splitlines(),
        "stashes": git_text(root, "stash", "list", "--format=%gd%x09%H%x09%gs").splitlines(),
        "worktrees_porcelain": git_text(root, "worktree", "list", "--porcelain").splitlines(),
        "lfs_files": git_text(root, "lfs", "ls-files", "--long").splitlines(),
        "environment": {
            "python": os.sys.version.split()[0],
            "git": subprocess.check_output(["git", "--version"], text=True).strip(),
            "git_lfs": subprocess.check_output(["git", "lfs", "version"], text=True).strip(),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
