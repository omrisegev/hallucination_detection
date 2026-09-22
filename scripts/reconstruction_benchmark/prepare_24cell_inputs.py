#!/usr/bin/env python3
"""Build independent target-free mixed-v2 input trees for the frozen 24 cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.preparation import (  # noqa: E402
    compare_prepared_builds,
    prepare_build,
)


DEFAULT_SOURCE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_RELEASE_ROOT = REPO / "results" / "reconstruction_benchmark_v1" / "releases"
FEATURE_CONFIG = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"
CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"


def _load_and_validate_static_contracts() -> tuple[dict, dict]:
    config = json.loads(FEATURE_CONFIG.read_text())
    registry = json.loads(CELL_REGISTRY.read_text())
    if config.get("schema_version") != "reconstruction-feature-contract-v1":
        raise RuntimeError("unexpected feature-contract schema")
    if registry.get("schema_version") != "reconstruction-frozen24-cell-registry-v1":
        raise RuntimeError("unexpected frozen-24 registry schema")
    for path_key, hash_key in (
        ("input_manifest", "input_manifest_sha256"),
        ("transform_source", "transform_source_sha256"),
        ("orientation_source", "orientation_source_sha256"),
        ("roster_source", "roster_source_sha256"),
    ):
        path = REPO / config[path_key]
        observed = sha256_file(path)
        if observed != config[hash_key]:
            raise RuntimeError(
                f"frozen feature input drifted for {path_key}: "
                f"expected {config[hash_key]}, got {observed}"
            )
    cells = registry.get("cells", [])
    registry_ids = tuple(item["cell_id"] for item in cells)
    if len(registry_ids) != 24 or len(set(registry_ids)) != 24:
        raise RuntimeError("frozen registry must contain 24 unique cells")
    if registry_ids != tuple(INSCOPE):
        raise RuntimeError("frozen registry order disagrees with scripts.inscope_cells")
    registry_domains = {item["cell_id"]: item["domain"] for item in cells}
    if registry_domains != {cell_id: GROUP[cell_id] for cell_id in INSCOPE}:
        raise RuntimeError("frozen registry domains disagree with scripts.inscope_cells")
    if registry.get("population_id") != "frozen24_response_v1":
        raise RuntimeError("unexpected frozen-24 population ID")
    return config, registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", choices=("A", "B", "both"), default="both")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, registry = _load_and_validate_static_contracts()
    release = args.release_root / args.release_id
    builds = ("A", "B") if args.build == "both" else (args.build,)
    for build_id in builds:
        prepare_build(
            source_bundle=args.source,
            out_dir=release / f"build_{build_id}" / "inputs",
            roster=INSCOPE,
            domains=GROUP,
            expected_source_sha256=config["input_sha256"],
            feature_contract_config_sha256=sha256_file(FEATURE_CONFIG),
            transform_source_sha256=config["transform_source_sha256"],
            orientation_source_sha256=config["orientation_source_sha256"],
            roster_source_sha256=config["roster_source_sha256"],
            build_id=build_id,
        )
    if args.build == "both":
        result = compare_prepared_builds(
            release / "build_A" / "inputs", release / "build_B" / "inputs"
        )
        atomic_write_json(release / "PREPARED_AB_VERIFICATION.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
