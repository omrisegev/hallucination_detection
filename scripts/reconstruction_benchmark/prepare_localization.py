#!/usr/bin/env python3
"""Prepare one independent target-free localization build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.localization_preparation import (  # noqa: E402
    prepare_localization_build,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_SOURCE_ROOT = REPO / "results/reconstruction_benchmark_v1/source_overlays/external_final_answer_v1"
DEFAULT_LOCALIZATION_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/localization.json"
DEFAULT_EXTERNAL_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True, help="Localization release ID")
    parser.add_argument("--external-release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_LOCALIZATION_REGISTRY)
    parser.add_argument("--external-registry", type=Path, default=DEFAULT_EXTERNAL_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument("--identity-key", type=Path)
    parser.add_argument("--allow-dirty-debug", action="store_true")
    args = parser.parse_args()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip() and not args.allow_dirty_debug:
        raise RuntimeError("scientific localization preparation requires a clean worktree")
    identity_key = args.identity_key or (
        args.release_root.parent / "private_control" / args.external_release_id
        / "external_final_answer" / "external-id-v2.key"
    )
    manifest = prepare_localization_build(
        repo=REPO,
        localization_registry_path=args.registry,
        external_registry_path=args.external_registry,
        population_registry_path=args.populations,
        release_root=args.release_root,
        localization_release_id=args.release_id,
        external_release_id=args.external_release_id,
        build_id=args.build,
        source_root=args.source_root,
        identity_key_path=identity_key,
        scientific_full=not args.allow_dirty_debug,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "external_release_id": args.external_release_id,
        "build": args.build,
        "n_cells": manifest["n_cells"],
        "external_certificate_sha256": manifest["external_certificate_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
