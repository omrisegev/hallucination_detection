#!/usr/bin/env python3
"""Evaluate one certified LEASH stopping build after target isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.leash_evaluation import evaluate_leash_build  # noqa: E402
from spectral_utils.reconstruction_benchmark.leash_contract import validate_safe_component  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", choices=("A", "B"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/releases")
    parser.add_argument("--private-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/private_control")
    parser.add_argument("--registry", type=Path, default=REPO / "configs/reconstruction_benchmark_v1/leash_stopping.json")
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()
    release_id = validate_safe_component(args.release_id, name="LEASH release ID")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO,
        check=True, capture_output=True, text=True,
    ).stdout
    if status.strip() and not args.allow_dirty_debug:
        raise RuntimeError("scientific LEASH evaluation requires a clean worktree")
    base = args.release_root / release_id / "leash"
    private = args.private_root / release_id / "leash"
    result = evaluate_leash_build(
        source_root=args.source_root,
        preparation_dir=base / args.build / "preparation",
        fit_dir=base / args.build / "fit",
        private_dir=private / args.build / "outcomes",
        preparation_ab_certificate=base / "PREPARATION_AB_VERIFICATION.json",
        fit_ab_certificate=base / "FIT_AB_VERIFICATION.json",
        registry_path=args.registry,
        output_dir=base / args.build / "evaluation",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
