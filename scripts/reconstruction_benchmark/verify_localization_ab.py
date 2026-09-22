#!/usr/bin/env python3
"""Verify byte-exact A/B localization preparation, scores, and projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.localization_ab import verify_localization_ab  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--release-root", type=Path,
        default=REPO / "results/reconstruction_benchmark_v1/releases",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    certificate = verify_localization_ab(
        release_id=args.release_id,
        release_root=args.release_root,
        output_path=args.output,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "status": certificate["status"],
        "certificate_sha256": certificate["certificate_sha256"],
        "n_cells": certificate["n_cells"],
        "n_core_systems": certificate["n_core_systems"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
