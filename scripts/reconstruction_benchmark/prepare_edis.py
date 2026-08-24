#!/usr/bin/env python3
"""Prepare one independent target-free EDIS A/B input build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.edis_identity import load_edis_identity_controller  # noqa: E402
from spectral_utils.reconstruction_benchmark.edis_preparation import (  # noqa: E402
    PREPARATION_SOURCE_PATHS,
    load_preparation_registry,
    prepare_build,
)
from spectral_utils.reconstruction_benchmark.io import canonical_json_bytes, sha256_bytes, sha256_file  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_PRIVATE_CONTROL = REPO / "results/reconstruction_benchmark_v1/private_control"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/edis_target_free.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--private-control-root", type=Path, default=DEFAULT_PRIVATE_CONTROL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source-root", type=Path, default=REPO)
    args = parser.parse_args()

    registry = load_preparation_registry(args.registry)
    identity = load_edis_identity_controller(
        private_control_root=args.private_control_root,
        release_id=args.release_id,
        create=True,
        release_root=args.release_root,
        repo=REPO,
    )
    snapshot = {
        "files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in PREPARATION_SOURCE_PATHS
        ]
    }
    snapshot["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    manifest = prepare_build(
        release_id=args.release_id,
        build_id=args.build,
        registry=registry,
        identity=identity,
        source_root=args.source_root,
        release_root=args.release_root,
        private_control_root=args.private_control_root,
        preparation_source_snapshot=snapshot,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "build_id": args.build,
        "n_cells": len(manifest["cells"]),
        "fit_registry": str(
            args.release_root / args.release_id / f"build_{args.build}"
            / "edis/inputs/FIT_REGISTRY.json"
        ),
        "labels_opened": False,
        "group_structure_fit_visible": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
