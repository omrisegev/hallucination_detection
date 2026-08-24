#!/usr/bin/env python3
"""Build immutable reporting-schema inputs from one strict frozen-24 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.reporting_bridge import (  # noqa: E402
    build_bridge_inputs,
    publish_bridge_inputs,
)


DEFAULT_CELL_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "frozen24_cells.json"
DEFAULT_METHOD_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "methods.json"
DEFAULT_FEATURE_REGISTRY = REPO / "configs" / "reconstruction_benchmark_v1" / "feature_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True, help="Reporting release ID; normally the scientific release directory name.")
    parser.add_argument("--evaluation-dir", type=Path, required=True, help="Directory containing EVALUATION_MANIFEST.json and its three hashed artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New immutable directory for reporting-builder inputs.")
    parser.add_argument("--cell-registry", type=Path, default=DEFAULT_CELL_REGISTRY)
    parser.add_argument("--method-registry", type=Path, default=DEFAULT_METHOD_REGISTRY)
    parser.add_argument("--feature-registry", type=Path, default=DEFAULT_FEATURE_REGISTRY)
    graph = parser.add_mutually_exclusive_group(required=True)
    graph.add_argument(
        "--graph-diagnostics-dir",
        type=Path,
        help="Signed graph_diagnostics directory from the same scientific release.",
    )
    graph.add_argument(
        "--allow-empty-graph-diagnostics",
        action="store_true",
        help="Explicit non-publication mode for bridge/unit checks before diagnostics exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = build_bridge_inputs(
        evaluation_dir=args.evaluation_dir,
        release_id=args.release_id,
        cell_registry_path=args.cell_registry,
        method_registry_path=args.method_registry,
        feature_registry_path=args.feature_registry,
        graph_diagnostics_dir=args.graph_diagnostics_dir,
        allow_empty_graph_diagnostics=args.allow_empty_graph_diagnostics,
    )
    output = publish_bridge_inputs(args.output_dir, inputs)
    print(json.dumps({
        "status": "BUILT",
        "release_id": inputs.registry["release_id"],
        "registry_sha256": inputs.registry["registry_sha256"],
        "row_counts": {table: len(rows) for table, rows in inputs.rows.items()},
        "output_dir": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
