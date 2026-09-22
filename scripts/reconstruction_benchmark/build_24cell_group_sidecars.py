#!/usr/bin/env python3
"""Build the fail-closed grouped-bootstrap sidecars for one 24-cell release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.reconstruction_benchmark.group_sidecars import (  # noqa: E402
    build_group_sidecars,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct all 24 source-group vectors without indexing or deriving "
            "from labels. Raw cache objects may contain labels; this post-fit "
            "provenance builder never uses them. Repeated cells have no IID fallback."
        )
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--label-bundle",
        type=Path,
        default=REPO_ROOT / "results" / "dependency_fusion_raw" / "cells.npz",
        help=(
            "The frozen source/label bundle. It contains label members, but only "
            "V, pool, and hand_signs are indexed."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Defaults to <release-root>/group_sidecars.",
    )
    parser.add_argument("--build-id", choices=("A", "B"), default="A")
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=REPO_ROOT / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json",
    )
    parser.add_argument(
        "--cell-registry",
        type=Path,
        default=REPO_ROOT / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json",
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=REPO_ROOT / "configs" / "residual_graph_deem_24cell_v1_registry.json",
    )
    args = parser.parse_args()

    release = args.release_root.resolve()
    output = (args.out_root or (release / "group_sidecars")).resolve()
    manifest = build_group_sidecars(
        repo_root=REPO_ROOT,
        release_root=release,
        out_root=output,
        raw_root=args.raw_root,
        label_bundle=args.label_bundle,
        feature_config_path=args.feature_config,
        cell_registry_path=args.cell_registry,
        source_registry_path=args.source_registry,
        build_id=args.build_id,
    )
    summary = {
        "manifest": str(output / "GROUP_SIDECARS.json"),
        "all_verified": manifest["all_verified"],
        "n_verified": manifest["n_verified"],
        "n_failed": manifest["n_failed"],
        "failed_cells": [
            {
                "cell_id": row["cell_id"],
                "failure_code": row.get("failure_code"),
                "failure_detail": row.get("failure_detail"),
            }
            for row in manifest["cells"]
            if row["verification_status"] != "VERIFIED"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if manifest["all_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
