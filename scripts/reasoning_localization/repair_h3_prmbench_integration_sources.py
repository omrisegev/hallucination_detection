#!/usr/bin/env python3
"""Repair the partially integrated H3 PRMBench family metric sources.

The evaluator's raw ``BY_FAMILY.csv`` deliberately uses evaluator-local status
labels (``OK`` and ``METRIC_UNDEFINED_SINGLE_CLASS``).  The living-report schema
uses execution statuses instead.  Preserve the frozen evaluator artifact and
derive a report-only source with the two status namespaces kept separate.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic_v2 as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402

EXPERIMENT = "P2_H3_PRMBENCH_DIAGNOSTIC"
RAW = run.ROOT / "evaluation" / "BY_FAMILY.csv"
REPORT_SOURCE = run.ROOT / "evaluation" / "BY_FAMILY_REPORT.csv"
MANIFEST = run.ROOT / "ARTIFACT_MANIFEST.json"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    raw_fields, raw_rows = read_csv(RAW)
    report_rows: list[dict[str, str]] = []
    for raw in raw_rows:
        row = dict(raw)
        evaluator_status = row["status"]
        row["metric_status"] = evaluator_status
        if evaluator_status == "OK":
            row["status"] = "COMPLETE"
        elif evaluator_status == "METRIC_UNDEFINED_SINGLE_CLASS":
            row["status"] = "BLOCKED"
        else:
            raise RuntimeError(f"unknown evaluator status: {evaluator_status}")
        report_rows.append(row)
    report_fields = [*raw_fields, "metric_status"]
    write_csv(REPORT_SOURCE, report_fields, report_rows)
    report_sha = sha256_file(REPORT_SOURCE)

    metric_path = p1.PROGRAM_ROOT / "METRICS_LONG.csv"
    metric_fields, metrics = read_csv(metric_path)
    source_rel = str(REPORT_SOURCE.relative_to(REPO))
    changed = 0
    status_by_key = {
        (row["arm_id"], row["error_family"]): row["status"] for row in report_rows
    }
    for row in metrics:
        if row.get("experiment_id") != EXPERIMENT or not row.get("cell_id", "").startswith("prmbench::"):
            continue
        family = row["cell_id"].split("::", 1)[1]
        row["status"] = status_by_key[(row["variant_id"], family)]
        row["source_artifact"] = source_rel
        row["source_sha256"] = report_sha
        changed += 1
    if changed != 54:
        raise RuntimeError(f"expected 54 family metric rows, repaired {changed}")
    write_csv(metric_path, metric_fields, metrics)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["artifacts"] = [
        artifact for artifact in payload["artifacts"] if artifact["path"] != source_rel
    ]
    payload["artifacts"].append({"path": source_rel, "sha256": report_sha})
    payload["artifacts"].sort(key=lambda artifact: artifact["path"])
    atomic_write_json(MANIFEST, payload)
    print(json.dumps({"family_rows": len(report_rows), "metric_rows_repaired": changed, "report_source_sha256": report_sha}, indent=2))


if __name__ == "__main__":
    main()
