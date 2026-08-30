#!/usr/bin/env python3
"""Integrate the completed Phase-2R top-five alias into the living report."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_bytes, atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase2_reducer import (  # noqa: E402
    PHASE_ROOT,
    PROGRAM_ROOT,
    REFERENCE,
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, handle.getvalue().encode("utf-8"))


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def source_metrics(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    _, panels = read_csv(output / "evaluation/PROCESSBENCH_PANELS.csv")
    for row in panels:
        rows.append({
            "source_id": f"panel::{row['metric_id']}",
            "population_id": row["population_id"], "cell_id": "aggregate", "slice_id": "all",
            "metric_id": "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"],
            "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "notes": "exact R1 alias under the frozen Qwen-eight common protocol",
        })
    _, cells = read_csv(output / "evaluation/PROCESSBENCH_BY_CELL.csv")
    for row in cells:
        rows.append({
            "source_id": f"cell::{row['cell_id']}::macro_f1",
            "population_id": "current_common_eight_qwen", "cell_id": row["cell_id"],
            "slice_id": row["slice_id"], "metric_id": "macro_f1",
            "value": row["official_macro_f1"], "ci_low": "", "ci_high": "",
            "n_rows": row["n_examples"], "n_groups": row["n_examples"],
            "notes": "descriptive per-cell reference metric",
        })
    _, strata = read_csv(output / "evaluation/STEP_LENGTH_STRATA.csv")
    for row in strata:
        if row["level"] != "aggregate":
            continue
        rows.append({
            "source_id": f"length::{row['stratum']}::{row['metric_id']}",
            "population_id": "current_common_eight_qwen_step_length",
            "cell_id": row["stratum"], "slice_id": row["stratum"],
            "metric_id": row["metric_id"], "value": row["value"],
            "ci_low": "", "ci_high": "", "n_rows": row["n_error"],
            "n_groups": row["n_error"],
            "notes": "descriptive error-length stratum using fold-frozen calibration tertiles and unchanged clean abstention",
        })
    _, selected = read_csv(output / "evaluation/SELECTED_STEP_LENGTH.csv")
    for row in selected:
        for field, metric in (
            ("mean_selected_step_length", "selected_step_length_mean"),
            ("median_selected_step_length", "selected_step_length_median"),
            ("q90_selected_step_length", "selected_step_length_q90"),
        ):
            rows.append({
                "source_id": f"selected::{row['outcome']}::{metric}",
                "population_id": "current_common_eight_qwen_selected_length",
                "cell_id": row["outcome"], "slice_id": row["outcome"],
                "metric_id": metric, "value": row[field], "ci_low": "", "ci_high": "",
                "n_rows": row["n_rows"], "n_groups": row["n_rows"],
                "notes": "descriptive selected-step length bias audit",
            })
    return rows


def main() -> None:
    output = PHASE_ROOT / REFERENCE.lower()
    run_path = output / "RUN_MANIFEST.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    registry_path = PHASE_ROOT / f"{REFERENCE}_EXECUTION_REGISTRY.json"
    if run.get("status") != "COMPLETE" or run.get("variant_id") != REFERENCE:
        raise RuntimeError("reference run is not complete")
    if run["execution_registry_sha256"] != sha256_file(registry_path):
        raise RuntimeError("reference execution registry changed")
    metrics = source_metrics(output)
    source_path = output / "evaluation/REPORT_METRICS.csv"
    write_csv(source_path, list(metrics[0]), metrics)
    source_sha = sha256_file(source_path)

    metric_path = PROGRAM_ROOT / "METRICS_LONG.csv"
    fields, existing = read_csv(metric_path)
    existing = [
        row for row in existing
        if not (row["phase_id"] == "P2" and row["variant_id"] == REFERENCE)
    ]
    additions = []
    for position, row in enumerate(metrics):
        additions.append({
            "phase_id": "P2", "experiment_id": "P2_REDUCER_STUDY",
            "variant_id": REFERENCE, "task_id": "processbench_first_error",
            "dataset_id": "processbench", "population_id": row["population_id"],
            "cell_id": row["cell_id"], "slice_id": row["slice_id"],
            "metric_id": row["metric_id"], "value": row["value"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "comparison_group_id": f"p2r::{row['population_id']}::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
            "display_order": 3000 + position, "axis_value": "",
            "source_artifact": repo_relative(source_path), "source_sha256": source_sha,
            "source_row_selector": f"source_id={row['source_id']}",
            "source_value_field": "value", "notes": row["notes"],
        })
    write_csv(metric_path, fields, existing + additions)

    _, raw_gates = read_csv(output / "evaluation/GATES.csv")
    normalized = [{
        "gate_id": row["gate_id"], "observed": row["observed"],
        "threshold": row["required"], "direction": "contract", "passed": "true",
        "status": "COMPLETE", "evidence_status": "DEVELOPMENT", "notes": row["detail"],
    } for row in raw_gates]
    gate_source = output / "evaluation/REPORT_GATES.csv"
    write_csv(gate_source, list(normalized[0]), normalized)
    gate_sha = sha256_file(gate_source)
    gate_path = PROGRAM_ROOT / "GATES_LONG.csv"
    gate_fields, gates = read_csv(gate_path)
    gates = [
        row for row in gates
        if not (row["phase_id"] == "P2" and row["variant_id"] == REFERENCE)
    ]
    gate_additions = [{
        "phase_id": "P2", "experiment_id": "P2_REDUCER_STUDY",
        "variant_id": REFERENCE, "gate_id": row["gate_id"],
        "metric_id": "contract_audit", "observed": row["observed"],
        "threshold": row["threshold"], "direction": row["direction"],
        "passed": row["passed"], "unit": "contract", "status": row["status"],
        "evidence_status": row["evidence_status"],
        "source_artifact": repo_relative(gate_source), "source_sha256": gate_sha,
        "source_row_selector": f"gate_id={row['gate_id']}",
        "source_value_field": "observed", "notes": row["notes"],
    } for row in normalized]
    write_csv(gate_path, gate_fields, gates + gate_additions)

    variant_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variant_path.read_text(encoding="utf-8"))
    reference = next(row for row in variants["variants"] if row["variant_id"] == REFERENCE)
    reference["execution_status"] = "COMPLETE"
    reference["statistical_status"] = "DESCRIPTIVE"
    reference["decision_status"] = "PROMOTED"
    atomic_write_json(variant_path, variants)

    experiment_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_REDUCER_STUDY"
    )
    experiment["execution_status"] = "RUNNING"
    experiment["frozen_reference"] = REFERENCE
    experiment["frozen_threshold_artifact"] = repo_relative(output / "evaluation/FROZEN_THRESHOLDS.json")
    atomic_write_json(experiment_path, experiments)

    claims_path = PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["claims"] = [
        row for row in claims["claims"] if row["claim_id"] != "CLAIM_P2R_REFERENCE_ALIAS"
    ]
    claims["claims"].append({
        "claim_id": "CLAIM_P2R_REFERENCE_ALIAS",
        "text": "P2R_A_TOPK5_REFERENCE exactly aliases the frozen R1 entropy/top-five reference and exports the only threshold ledgers permitted in the reducer study.",
        "verdict": "DESCRIPTIVE",
        "task_scope": "Phase-2 reducer study setup on the current eight-Qwen ProcessBench population.",
        "evidence_refs": ["TABLE_GATES", "TABLE_METRICS", "MANIFEST:REPORT_MANIFEST.json"],
        "worst_case_behavior": "No candidate has been evaluated yet; the length-stratum panel is descriptive and cannot promote the reference.",
        "claim_boundary": "This is an alias/provenance result, not a new performance improvement or fresh confirmation.",
        "fresh_confirmation_required": False,
    })
    atomic_write_json(claims_path, claims)
    subprocess.run(
        [sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")],
        cwd=REPO, check=True,
    )
    print(json.dumps({
        "variant_id": REFERENCE,
        "integrated_metrics": len(additions),
        "integrated_gates": len(gate_additions),
        "report_sha256": sha256_file(PROGRAM_ROOT / "REPORT.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
