#!/usr/bin/env python3
"""Verify source-rederived byte identity of LEASH preparation A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.leash_ab import verify_leash_preparation_ab  # noqa: E402
from spectral_utils.reconstruction_benchmark.leash_contract import validate_safe_component  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/releases")
    parser.add_argument("--private-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/private_control")
    parser.add_argument("--registry", type=Path, default=REPO / "configs/reconstruction_benchmark_v1/leash_stopping.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    release_id = validate_safe_component(args.release_id, name="LEASH release ID")
    base = args.release_root / release_id / "leash"
    private = args.private_root / release_id / "leash"
    certificate = verify_leash_preparation_ab(
        source_root=args.source_root, registry_path=args.registry,
        public_a=base / "A/preparation", private_a=private / "A/outcomes",
        public_b=base / "B/preparation", private_b=private / "B/outcomes",
        certificate_path=args.output or base / "PREPARATION_AB_VERIFICATION.json",
    )
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
