#!/usr/bin/env python3
"""Build immutable downstream all-pairs and point-winner contrast artifacts."""

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
    publish_winner_contrasts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replica-id", required=True)
    subparsers = parser.add_subparsers(dest="source", required=True)
    frozen = subparsers.add_parser(
        "frozen24", help="Use the certified frozen24 BOOTSTRAP_DRAWS.npz archive.",
    )
    frozen.add_argument("--evaluation-dir", required=True, type=Path)
    external = subparsers.add_parser(
        "external", help="Recompute draws from certified signed external scores/labels.",
    )
    external.add_argument("--evaluation-dir", required=True, type=Path)
    external.add_argument("--evaluation-ab-certificate", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "frozen24":
        source = load_frozen24_source(args.evaluation_dir)
    else:
        source = load_external_source(
            args.evaluation_dir, args.evaluation_ab_certificate,
        )
    result = publish_winner_contrasts(
        source, output_dir=args.output_dir, replica_id=args.replica_id,
    )
    print(json.dumps({
        "status": result["status"],
        "source_type": result["source_type"],
        "replica_id": result["replica_id"],
        "n_comparison_scopes": result["n_comparison_scopes"],
        "row_counts": result["row_counts"],
        "payload_sha256": result["payload_sha256"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
