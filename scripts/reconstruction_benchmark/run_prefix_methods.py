#!/usr/bin/env python3
"""CPU-recompute the frozen three-system causal-prefix roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.prefix_fit import run_prefix_methods  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", choices=("A", "B"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
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
        default=REPO / "configs/reconstruction_benchmark_v1/prefix.json",
    )
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip() and not args.allow_dirty_debug:
        raise RuntimeError("scientific prefix fitting requires a clean worktree")
    manifest = run_prefix_methods(
        repo=REPO,
        registry_path=args.registry,
        release_root=args.release_root,
        private_root=args.private_root,
        release_id=args.release_id,
        build_id=args.build,
        source_root=args.source_root,
        scientific_full=not args.allow_dirty_debug,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "build": args.build,
        "status": manifest["execution_status"],
        "observations": manifest["score_artifact"]["observations"],
        "method_scores": manifest["score_artifact"]["method_scores"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
