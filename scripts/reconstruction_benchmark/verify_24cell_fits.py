#!/usr/bin/env python3
"""Issue the mandatory independent A/B score-equality certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark import PRIMARY_METHOD_SPECS  # noqa: E402
from spectral_utils.reconstruction_benchmark.fit_validation import (  # noqa: E402
    compare_frozen_builds,
    validate_prepared_manifest,
)
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results" / "reconstruction_benchmark_v1" / "releases"
FEATURE_CONFIG = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"
CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = args.release_root / args.release_id
    if (release / "SCORE_AB_VERIFICATION.json").exists():
        raise FileExistsError("A/B score certificate already exists; use a new release")
    feature_config = json.loads(FEATURE_CONFIG.read_text())
    cell_registry = json.loads(CELL_REGISTRY.read_text())
    inputs = {
        build_id: validate_prepared_manifest(
            input_root=release / f"build_{build_id}" / "inputs",
            build_id=build_id,
            repo=REPO,
            feature_config=feature_config,
            cell_registry=cell_registry,
        )
        for build_id in ("A", "B")
    }
    result = compare_frozen_builds(
        release_root=release,
        input_manifests=inputs,
        method_specs=PRIMARY_METHOD_SPECS,
        population_id=cell_registry["population_id"],
    )
    atomic_write_json(release / "SCORE_AB_VERIFICATION.json", result)
    print(json.dumps({
        "pass": result["pass"],
        "n_pairs": result["n_pairs"],
        "payload_sha256": result["payload_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
