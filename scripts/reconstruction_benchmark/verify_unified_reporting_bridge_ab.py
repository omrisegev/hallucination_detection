#!/usr/bin/env python3
"""Verify two complete unified-reporting builds and issue a no-clobber PASS cert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_reporting.unified_reporting_publish import (  # noqa: E402
    verify_unified_release_ab,
)
from spectral_utils.reconstruction_reporting.unified_reporting_schemas import (  # noqa: E402
    UnifiedReportingError,
)


DEFAULT_CONTRACT = REPO / "configs/reconstruction_benchmark_v1/unified_reporting_v1.json"
DEFAULT_SOURCE_LOCK = (
    REPO / "configs/reconstruction_benchmark_v1/unified_reporting_source_lock_v1.json"
)


def _source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        root_id, separator, raw_path = value.partition("=")
        if not separator or not root_id or not raw_path:
            raise UnifiedReportingError("--source-root must use ROOT_ID=PATH")
        if root_id in roots:
            raise UnifiedReportingError(f"duplicate runtime source root: {root_id}")
        roots[root_id] = Path(raw_path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-a", required=True, type=Path)
    parser.add_argument("--build-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument(
        "--source-root", action="append", required=True, metavar="ROOT_ID=PATH",
        help="Runtime root alias required by the reviewed lock; repeat for each root.",
    )
    args = parser.parse_args()
    result = verify_unified_release_ab(
        build_a=args.build_a, build_b=args.build_b, certificate_path=args.output,
        contract_path=args.contract, source_lock_path=args.source_lock,
        source_roots=_source_roots(args.source_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
