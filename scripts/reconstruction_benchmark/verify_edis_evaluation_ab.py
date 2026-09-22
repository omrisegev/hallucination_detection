#!/usr/bin/env python3
"""Certify exact A/B equivalence of post-freeze EDIS evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.edis_ab import verify_evaluation_ab  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_PRIVATE_CONTROL = REPO / "results/reconstruction_benchmark_v1/private_control"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/edis_target_free.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--private-control-root", type=Path, default=DEFAULT_PRIVATE_CONTROL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--score-ab-certificate", type=Path)
    args = parser.parse_args()
    certificate = verify_evaluation_ab(
        release_id=args.release_id,
        release_root=args.release_root,
        private_control_root=args.private_control_root,
        preparation_registry_path=args.registry,
        repo=REPO,
        score_certificate_path=args.score_ab_certificate,
    )
    print(json.dumps({
        "status": certificate["status"],
        "certificate_sha256": certificate["certificate_sha256"],
        "bootstrap_draws": certificate["bootstrap_draws"],
        "tables": certificate["canonical_table_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
