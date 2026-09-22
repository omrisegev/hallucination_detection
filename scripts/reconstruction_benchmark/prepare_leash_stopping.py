#!/usr/bin/env python3
"""Audit sources or prepare one independent LEASH stopping build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.leash_preparation import (  # noqa: E402
    audit_leash_sources,
    prepare_leash_build,
)
from spectral_utils.reconstruction_benchmark.leash_contract import validate_safe_component  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-id")
    parser.add_argument("--build", choices=("A", "B"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--release-root", type=Path,
        default=REPO / "results/reconstruction_benchmark_v1/releases",
    )
    parser.add_argument(
        "--private-root", type=Path,
        default=REPO / "results/reconstruction_benchmark_v1/private_control",
    )
    parser.add_argument(
        "--registry", type=Path,
        default=REPO / "configs/reconstruction_benchmark_v1/leash_stopping.json",
    )
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_leash_sources(
            source_root=args.source_root, registry_path=args.registry
        ), indent=2, sort_keys=True))
        return
    if not args.release_id or not args.build:
        parser.error("--release-id and --build are required unless --audit-only is used")
    release_id = validate_safe_component(args.release_id, name="LEASH release ID")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO,
        check=True, capture_output=True, text=True,
    ).stdout
    if status.strip() and not args.allow_dirty_debug:
        raise RuntimeError("scientific LEASH preparation requires a clean worktree")
    public = args.release_root / release_id / "leash" / args.build / "preparation"
    private = args.private_root / release_id / "leash" / args.build / "outcomes"
    result = prepare_leash_build(
        source_root=args.source_root, registry_path=args.registry,
        public_output=public, private_output=private,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
