#!/usr/bin/env python3
"""Query a reconstruction benchmark DuckDB by task/dataset/cell/slice/method."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.reconstruction_reporting.query import (  # noqa: E402
    VIEW_NAMES,
    query_results,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--view", choices=VIEW_NAMES, default="v_atomic_leaderboard")
    for field in (
        "lane_id",
        "task_id",
        "dataset_id",
        "population_id",
        "cell_id",
        "slice_id",
        "cohort_id",
        "method_id",
        "method_version_id",
        "adapter_id",
        "system_id",
        "comparison_group_id",
        "aggregation_id",
        "aggregation_level",
        "metric_id",
        "status",
        "evidence_grade",
        "fidelity",
        "access_contract_id",
        "feature_contract_id",
        "evaluator_id",
    ):
        parser.add_argument("--" + field.replace("_", "-"), dest=field)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--output", type=Path, help="Default is stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    filter_fields = (
        "lane_id",
        "task_id",
        "dataset_id",
        "population_id",
        "cell_id",
        "slice_id",
        "cohort_id",
        "method_id",
        "method_version_id",
        "adapter_id",
        "system_id",
        "comparison_group_id",
        "aggregation_id",
        "aggregation_level",
        "metric_id",
        "status",
        "evidence_grade",
        "fidelity",
        "access_contract_id",
        "feature_contract_id",
        "evaluator_id",
    )
    filters = {field: getattr(args, field) for field in filter_fields if getattr(args, field) is not None}
    columns, rows = query_results(
        args.database,
        view=args.view,
        filters=filters,
        limit=args.limit,
    )
    if args.format == "json":
        payload = json.dumps(
            [dict(zip(columns, row)) for row in rows],
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    else:
        from io import StringIO

        buffer = StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
        payload = buffer.getvalue()
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
