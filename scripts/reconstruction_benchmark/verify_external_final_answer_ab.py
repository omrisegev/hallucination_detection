#!/usr/bin/env python3
"""Issue the required exact A/B certificate for external final-answer scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_ab import verify_external_ab  # noqa: E402
from spectral_utils.reconstruction_benchmark.external_final_answer import load_external_registry  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    args = parser.parse_args()
    registry = load_external_registry(
        repo=REPO,
        registry_path=args.registry,
        population_registry_path=args.populations,
    )
    certificate = verify_external_ab(
        release_id=args.release_id,
        release_root=args.release_root,
        registry=registry,
        repo=REPO,
    )
    print(json.dumps({
        "status": certificate["status"],
        "certificate_sha256": certificate["certificate_sha256"],
        "n_cells": len(certificate["cell_ids"]),
        "n_method_comparisons": certificate["n_method_comparisons"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
