#!/usr/bin/env python3
"""Repair P3D reporting selectors after the fail-closed first integration."""

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
from scripts.reasoning_localization.run_phase3_deployed_upcr_prune_refit import (  # noqa: E402
    EXPERIMENT_ID,
    OUTPUT,
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    contrasts_path = p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv"
    contrasts = read(contrasts_path)
    repaired_contrasts = 0
    for row in contrasts:
        if row["experiment_id"] != EXPERIMENT_ID:
            continue
        row["source_row_selector"] = (
            f"left_variant_id={row['left_variant_id']};"
            f"right_variant_id={row['right_variant_id']};"
            f"metric_id={row['metric_id']}"
        )
        repaired_contrasts += 1
    write(contrasts_path, contrasts)

    gate_source = OUTPUT / "evaluation/REPORTING_GATES.csv"
    gates_source_rows = read(gate_source)
    for row in gates_source_rows:
        row["passed"] = row["passed"].lower()
    write(gate_source, gates_source_rows)
    gate_sha = sha256_file(gate_source)

    gates_path = p1.PROGRAM_ROOT / "GATES_LONG.csv"
    gates = read(gates_path)
    repaired_gates = 0
    for row in gates:
        if row["experiment_id"] != EXPERIMENT_ID:
            continue
        row["passed"] = row["passed"].lower()
        row["source_sha256"] = gate_sha
        repaired_gates += 1
    write(gates_path, gates)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print({
        "repaired_contrasts": repaired_contrasts,
        "repaired_gates": repaired_gates,
        "report_sha256": build.manifest["output"]["sha256"],
    })


if __name__ == "__main__":
    main()
