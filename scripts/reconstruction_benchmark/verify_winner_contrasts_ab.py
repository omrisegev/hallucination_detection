#!/usr/bin/env python3
"""Verify independently built winner-contrast artifacts and issue a PASS cert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.winner_contrasts import (  # noqa: E402
    load_external_source,
    load_frozen24_source,
    verify_winner_contrasts_ab,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-a", required=True, type=Path)
    parser.add_argument("--build-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="source", required=True)
    frozen = subparsers.add_parser(
        "frozen24", help="Rederive both artifacts from one certified frozen24 archive.",
    )
    frozen.add_argument("--evaluation-dir", required=True, type=Path)
    external = subparsers.add_parser(
        "external", help="Rederive from independently certified external A/B builds.",
    )
    external.add_argument("--evaluation-dir-a", required=True, type=Path)
    external.add_argument("--evaluation-dir-b", required=True, type=Path)
    external.add_argument("--evaluation-ab-certificate", required=True, type=Path)
    args = parser.parse_args()
    if args.source == "frozen24":
        source_a = source_b = load_frozen24_source(args.evaluation_dir)
    else:
        source_a = load_external_source(
            args.evaluation_dir_a, args.evaluation_ab_certificate,
        )
        source_b = load_external_source(
            args.evaluation_dir_b, args.evaluation_ab_certificate,
        )
    result = verify_winner_contrasts_ab(
        args.build_a, args.build_b, source_a=source_a, source_b=source_b,
        output_path=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "lane_id": result["lane_id"],
        "certificate_sha256": result["certificate_sha256"],
        "normalized_manifest_sha256": result["normalized_manifest_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
