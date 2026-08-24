#!/usr/bin/env python3
"""Certify exact post-label A/B identity for external final-answer evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_evaluation_ab import (  # noqa: E402
    verify_external_evaluation_ab,
)


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/external_final_answer.json"
DEFAULT_POPULATIONS = REPO / "configs/reconstruction_benchmark_v1/populations.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATIONS)
    parser.add_argument("--score-ab-certificate", type=Path)
    parser.add_argument(
        "--identity-key", type=Path,
        help="Controller-only release identity key; defaults outside releases/.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_external_evaluation_ab(
        release_id=args.release_id,
        release_root=args.release_root,
        registry_path=args.registry,
        population_registry_path=args.populations,
        repo=REPO,
        score_certificate_path=args.score_ab_certificate,
        identity_key_path=args.identity_key,
        output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "certificate_sha256": result["certificate_sha256"],
        "n_cells": len(result["cell_ids"]),
        "n_metric_rows": result["n_metric_rows"],
        "n_contrast_rows": result["n_contrast_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
