#!/usr/bin/env python3
"""Audit external final-answer applicability without downloading or scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_final_answer import (  # noqa: E402
    audit_external_registry,
    load_external_registry,
)
from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402


DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument(
        "--source-root", type=Path, default=REPO,
        help="Root containing the hash-frozen relative telemetry paths.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Open verified telemetry and audit all nominal views plus the exact present-feature subset; may be expensive.",
    )
    args = parser.parse_args()
    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    manifest = audit_external_registry(
        registry=registry, repo=args.source_root, deep=args.deep
    )
    if args.output:
        atomic_write_json(args.output, manifest)
    print(json.dumps({
        "deep": args.deep,
        "n_cells": len(manifest["cells"]),
        "status_counts": manifest["status_counts"],
        "payload_sha256": manifest["payload_sha256"],
        "output": None if args.output is None else str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
