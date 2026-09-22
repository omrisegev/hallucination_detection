#!/usr/bin/env python3
"""Integrate one checksum-verified Phase-1 run into the living report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_bytes, atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT, VARIANTS  # noqa: E402


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, handle.getvalue().encode("utf-8"))


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def source_metric_rows(variant_id: str, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    _, panels = read_csv(output / "evaluation/PROCESSBENCH_PANELS.csv")
    for row in panels:
        rows.append({
            "source_id": f"pb_panel::{row['population_id']}::{row['metric_id']}",
            "variant_id": variant_id, "task_id": "processbench_first_error",
            "dataset_id": "processbench", "population_id": row["population_id"],
            "cell_id": "aggregate", "slice_id": "all",
            "metric_id": "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"],
            "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "notes": "paired whole-source-question bootstrap; model and dataset cells macro-averaged",
        })
    _, cells = read_csv(output / "evaluation/PROCESSBENCH_BY_CELL.csv")
    for row in cells:
        rows.append({
            "source_id": f"pb_cell::{row['cell_id']}::macro_f1",
            "variant_id": variant_id, "task_id": "processbench_first_error",
            "dataset_id": "processbench", "population_id": "current_full_twelve_cell",
            "cell_id": row["cell_id"], "slice_id": row["slice_id"], "metric_id": "macro_f1",
            "value": row["official_macro_f1"], "ci_low": "", "ci_high": "",
            "n_rows": row["n_examples"], "n_groups": row["n_examples"],
            "notes": "descriptive cell estimate; source-grouped inference remains at the registered panel level",
        })
    _, prm = read_csv(output / "evaluation/PRMBENCH_SLICES.csv")
    for row in prm:
        if row.get("status", "") not in {"", "OK"} or row["auroc"] == "":
            continue
        population = "prmbench_error_responses" if row["slice_type"] == "overall" else "prmbench_error_family"
        cell_id = "aggregate" if row["slice_type"] == "overall" else row["slice_id"]
        for metric in ("auroc", "auprc"):
            rows.append({
                "source_id": f"prm::{row['slice_type']}::{row['slice_id']}::{metric}",
                "variant_id": variant_id, "task_id": "prmbench_step_error",
                "dataset_id": "prmbench", "population_id": population,
                "cell_id": cell_id, "slice_id": row["slice_id"], "metric_id": metric,
                "value": row[metric], "ci_low": "", "ci_high": "",
                "n_rows": row["n_examples"], "n_groups": "6208" if row["slice_type"] == "overall" else "",
                "notes": "every-step error ranking; no ProcessBench aggregation",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()
    variant_id = args.variant
    output = PROGRAM_ROOT / "phase_1" / variant_id.lower()
    run_path = output / "RUN_MANIFEST.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    registry_path = PROGRAM_ROOT / "phase_1" / f"{variant_id}_EXECUTION_REGISTRY.json"
    if run.get("status") != "COMPLETE" or run.get("variant_id") != variant_id:
        raise RuntimeError("run manifest is not a completed matching variant")
    if run.get("execution_registry_sha256") != sha256_file(registry_path):
        raise RuntimeError("execution-registry checksum mismatch")

    source_rows = source_metric_rows(variant_id, output)
    source_path = output / "evaluation/REPORT_METRICS.csv"
    source_fields = list(source_rows[0])
    write_csv(source_path, source_fields, source_rows)
    source_sha = sha256_file(source_path)

    metric_path = PROGRAM_ROOT / "METRICS_LONG.csv"
    metric_fields, existing = read_csv(metric_path)
    existing = [row for row in existing if not (row["phase_id"] == "P1" and row["variant_id"] == variant_id)]
    display_base = 1000 + 100 * VARIANTS.index(variant_id)
    additions = []
    for position, row in enumerate(source_rows):
        comparison = f"p1::{row['task_id']}::{row['population_id']}::{row['metric_id']}"
        additions.append({
            "phase_id": "P1", "experiment_id": "P1_BASELINES", "variant_id": variant_id,
            "task_id": row["task_id"], "dataset_id": row["dataset_id"],
            "population_id": row["population_id"], "cell_id": row["cell_id"], "slice_id": row["slice_id"],
            "metric_id": row["metric_id"], "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"], "comparison_group_id": comparison,
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT", "display_order": display_base + position,
            "axis_value": "", "source_artifact": repo_relative(source_path), "source_sha256": source_sha,
            "source_row_selector": f"source_id={row['source_id']}", "source_value_field": "value", "notes": row["notes"],
        })
    write_csv(metric_path, metric_fields, existing + additions)

    gate_source_fields, raw_gates = read_csv(output / "evaluation/GATES.csv")
    del gate_source_fields
    report_gates = []
    for row in raw_gates:
        passed = "true" if row["status"] == "PASS" else ("false" if row["status"] == "HARD_FAIL" else "NA")
        report_gates.append({
            "gate_id": row["gate_id"], "observed": row["observed"], "threshold": row["required"],
            "direction": "contract", "passed": passed,
            "status": "HARD_FAIL" if row["status"] == "HARD_FAIL" else ("BLOCKED" if row["status"] == "BLOCKED" else "COMPLETE"),
            "evidence_status": "DEVELOPMENT", "notes": row["detail"],
        })
    gate_source_path = output / "evaluation/REPORT_GATES.csv"
    write_csv(gate_source_path, list(report_gates[0]), report_gates)
    gate_sha = sha256_file(gate_source_path)
    gates_path = PROGRAM_ROOT / "GATES_LONG.csv"
    gate_fields, existing_gates = read_csv(gates_path)
    existing_gates = [row for row in existing_gates if not (row["phase_id"] == "P1" and row["variant_id"] == variant_id)]
    gate_additions = [{
        "phase_id": "P1", "experiment_id": "P1_BASELINES", "variant_id": variant_id,
        "gate_id": row["gate_id"], "metric_id": "contract_audit", "observed": row["observed"],
        "threshold": row["threshold"], "direction": row["direction"], "passed": row["passed"], "unit": "contract",
        "status": row["status"], "evidence_status": row["evidence_status"],
        "source_artifact": repo_relative(gate_source_path), "source_sha256": gate_sha,
        "source_row_selector": f"gate_id={row['gate_id']}", "source_value_field": "observed", "notes": row["notes"],
    } for row in report_gates]
    write_csv(gates_path, gate_fields, existing_gates + gate_additions)

    variant_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))
    variant = next(row for row in variants["variants"] if row["variant_id"] == variant_id)
    variant["execution_status"] = "COMPLETE"
    variant["statistical_status"] = "NOT_EVALUATED"
    variant["decision_status"] = "PENDING"
    atomic_write_json(variant_path, variants)
    subprocess.run([sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")], cwd=REPO, check=True)
    print(json.dumps({"variant_id": variant_id, "integrated_metrics": len(additions), "report_sha256": sha256_file(PROGRAM_ROOT / "REPORT.html")}, indent=2))


if __name__ == "__main__":
    main()
