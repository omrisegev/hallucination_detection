#!/usr/bin/env python3
"""Build, verify, or snapshot the Reasoning Localization 0.3662 report."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "spectral_utils" / "reasoning_localization_reporting.py"
SPEC = importlib.util.spec_from_file_location("reasoning_localization_reporting", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"cannot load report module: {MODULE_PATH}")
REPORTING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORTING
SPEC.loader.exec_module(REPORTING)


DEFAULT_REPORT_DIR = REPO / "results" / "reasoning_localization_03662_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--check", action="store_true", help="fail unless generated outputs are byte-identical to a fresh build")
    parser.add_argument("--snapshot", metavar="LABEL", help="create or verify an immutable reporting/phase_N/amendment snapshot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    try:
        build = REPORTING.prepare_build(report_dir, REPO)
        if args.check:
            REPORTING.check_build(report_dir, build)
            action = "checked"
        else:
            REPORTING.write_build(report_dir, build)
            action = "built"
        snapshot = None
        if args.snapshot:
            snapshot = REPORTING.create_immutable_snapshot(report_dir, args.snapshot, build, REPO)
        print(f"{action}: {report_dir / 'REPORT.html'}")
        print(f"sha256: {build.manifest['output']['sha256']}")
        if snapshot is not None:
            print(f"snapshot: {snapshot}")
        return 0
    except REPORTING.ReportingValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
