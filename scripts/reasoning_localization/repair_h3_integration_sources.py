#!/usr/bin/env python3
"""Repair source selectors after the first H3 import stopped before report build."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402

EXPERIMENT = "P2_H3_RELIABILITY_FUSION"
DEST = p1.PROGRAM_ROOT / "phase_2" / "diagnostic" / "h3_reliability_fusion_v1"


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    metric_path = p1.PROGRAM_ROOT / "METRICS_LONG.csv"
    fields, metrics = read(metric_path)
    changed_metrics = 0
    for row in metrics:
        if row.get("experiment_id") == EXPERIMENT:
            arm = row["source_row_selector"].split(";", 1)[0]
            row["source_row_selector"] = arm
            changed_metrics += 1
    write(metric_path, fields, metrics)

    gate_path = p1.PROGRAM_ROOT / "GATES_LONG.csv"
    fields, gates = read(gate_path)
    selected = [row for row in gates if row.get("experiment_id") == EXPERIMENT]
    gate_source = DEST / "INTEGRATION_GATES.csv"
    source_fields = ["gate_id", "metric_id", "observed", "threshold", "direction", "passed", "unit", "status", "evidence_status"]
    write(gate_source, source_fields, selected)
    gate_sha = sha256_file(gate_source)
    changed_gates = 0
    for row in gates:
        if row.get("experiment_id") == EXPERIMENT:
            row["source_artifact"] = str(gate_source.relative_to(REPO))
            row["source_sha256"] = gate_sha
            row["source_row_selector"] = f"gate_id={row['gate_id']}"
            row["source_value_field"] = "observed"
            changed_gates += 1
    write(gate_path, fields, gates)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print({"metrics_repaired": changed_metrics, "gates_repaired": changed_gates, "report_sha256": build.manifest["output"]["sha256"]})


if __name__ == "__main__":
    main()
