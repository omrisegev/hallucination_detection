#!/usr/bin/env python3
"""Integrate the frozen C1 run into the living evidence report."""

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

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
)
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402


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


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def add_metrics(output: Path) -> None:
    _, panels = read_csv(output / "evaluation/PROCESSBENCH_PANELS.csv")
    _, cells = read_csv(output / "evaluation/PROCESSBENCH_BY_CELL.csv")
    _, strata = read_csv(output / "evaluation/STEP_LENGTH_STRATA.csv")
    _, flips = read_csv(output / "evaluation/PREDICTION_FLIP_SUMMARY.csv")
    _, contrasts = read_csv(output / "evaluation/PAIRWISE_CONTRASTS.csv")
    report_rows: list[dict[str, object]] = []
    for row in panels:
        report_rows.append({
            "source_id": f"panel::{row['arm_id']}::{row['metric_id']}",
            "variant_id": row["arm_id"], "population_id": row["population_id"],
            "cell_id": "aggregate", "slice_id": "all",
            "metric_id": "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"],
            "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "notes": "absolute metric under the per-arm atomic grouped five-fold threshold contract",
        })
    for row in cells:
        report_rows.append({
            "source_id": f"cell::{row['arm_id']}::{row['cell_id']}::macro_f1",
            "variant_id": row["arm_id"], "population_id": "current_common_eight_qwen",
            "cell_id": row["cell_id"], "slice_id": row["slice_id"], "metric_id": "macro_f1",
            "value": row["official_macro_f1"], "ci_low": "", "ci_high": "",
            "n_rows": row["n_examples"], "n_groups": row["n_examples"],
            "notes": "descriptive per-cell atomic metric on the common population",
        })
    for row in strata:
        if row["level"] != "aggregate":
            continue
        report_rows.append({
            "source_id": f"length::{row['stratum']}::{row['metric_id']}",
            "variant_id": c1.CANDIDATE,
            "population_id": "current_common_eight_qwen_step_length",
            "cell_id": row["stratum"], "slice_id": row["stratum"],
            "metric_id": row["metric_id"], "value": row["value"],
            "ci_low": "", "ci_high": "", "n_rows": row["n_error"], "n_groups": row["n_error"],
            "notes": "descriptive C1 error-step length stratum using calibration-frozen cut points",
        })
    for row in flips:
        report_rows.append({
            "source_id": f"flip::{row['cell_id']}::{row['transition']}",
            "variant_id": c1.CANDIDATE,
            "population_id": "current_common_eight_qwen_prediction_flips",
            "cell_id": row["cell_id"], "slice_id": row["transition"],
            "metric_id": "prediction_flip_count", "value": row["count"],
            "ci_low": "", "ci_high": "", "n_rows": row["count"], "n_groups": row["count"],
            "notes": "exact deterministic C1-versus-atomic-top-ten prediction transition count",
        })
    for row in contrasts:
        if row["right_variant_id"] != c1.REFERENCE:
            continue
        mapped = {
            "first_error_exact": "first_error_exact_delta",
            "clean_abstention_accuracy": "clean_abstention_delta",
        }.get(row["metric_id"])
        if mapped:
            report_rows.append({
                "source_id": f"scatter::{mapped}", "variant_id": c1.CANDIDATE,
                "population_id": "current_common_eight_qwen", "cell_id": "aggregate",
                "slice_id": "versus_atomic_top10", "metric_id": mapped,
                "value": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "n_rows": 6400, "n_groups": 3400,
                "notes": "paired component delta versus P2A_TOPK10_REFERENCE",
            })

    source = output / "evaluation/REPORT_METRICS.csv"
    write_csv(source, list(report_rows[0]), report_rows)
    source_sha = sha256_file(source)
    path = p1.PROGRAM_ROOT / "METRICS_LONG.csv"
    fields, existing = read_csv(path)
    atomic_ids = {c1.REFERENCE, c1.CANDIDATE}
    existing = [
        row for row in existing
        if not (row["experiment_id"] == "P2_ATOMIC" and row["variant_id"] in atomic_ids)
    ]
    additions = []
    for index, row in enumerate(report_rows):
        additions.append({
            "phase_id": "P2", "experiment_id": "P2_ATOMIC", "variant_id": row["variant_id"],
            "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": row["population_id"], "cell_id": row["cell_id"], "slice_id": row["slice_id"],
            "metric_id": row["metric_id"], "value": row["value"], "ci_low": row["ci_low"],
            "ci_high": row["ci_high"], "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "comparison_group_id": f"p2a::{row['population_id']}::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
            "display_order": 5000 + index, "axis_value": "", "source_artifact": relative(source),
            "source_sha256": source_sha, "source_row_selector": f"source_id={row['source_id']}",
            "source_value_field": "value", "notes": row["notes"],
        })
    write_csv(path, fields, existing + additions)


def add_contrasts(output: Path) -> dict[tuple[str, str], dict[str, str]]:
    source = output / "evaluation/PAIRWISE_CONTRASTS.csv"
    _, rows = read_csv(source)
    source_sha = sha256_file(source)
    path = p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_ATOMIC"]
    additions = []
    for row in rows:
        additions.append({
            "phase_id": "P2", "experiment_id": "P2_ATOMIC",
            "left_variant_id": row["left_variant_id"], "right_variant_id": row["right_variant_id"],
            "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen", "metric_id": row["metric_id"],
            "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "p_adjusted": "", "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
            "worst_unit_delta": row["worst_unit_delta"],
            "comparison_group_id": f"p2a::processbench::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
            "source_artifact": relative(source), "source_sha256": source_sha,
            "source_row_selector": f"contrast_id={row['contrast_id']}",
            "notes": (
                f"{row['inference']}; worst cell {row['worst_unit_id']}; family W/T/L "
                f"{row['family_wins']}/{row['family_ties']}/{row['family_losses']}; "
                f"worst family {row['worst_family_id']} {float(row['worst_family_delta']):+.6f}; "
                "practical bounds +0.005/-0.005"
            ),
        })
    write_csv(path, fields, existing + additions)
    return {(row["right_variant_id"], row["metric_id"]): row for row in rows}


def add_gates(output: Path) -> None:
    _, rows = read_csv(output / "evaluation/GATES.csv")
    normalized = [{
        "gate_id": row["gate_id"], "observed": row["observed"],
        "threshold": row["required"], "direction": "contract",
        "passed": str(row["status"] == "PASS").lower(), "status": "COMPLETE",
        "evidence_status": "DEVELOPMENT",
        "notes": f"{row['detail']}; raw gate status={row['status']}",
    } for row in rows]
    source = output / "evaluation/REPORT_GATES.csv"
    write_csv(source, list(normalized[0]), normalized)
    source_sha = sha256_file(source)
    path = p1.PROGRAM_ROOT / "GATES_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_ATOMIC"]
    additions = [{
        "phase_id": "P2", "experiment_id": "P2_ATOMIC", "variant_id": c1.CANDIDATE,
        "gate_id": row["gate_id"], "metric_id": "contract_or_promotion_gate",
        "observed": row["observed"], "threshold": row["threshold"], "direction": row["direction"],
        "passed": row["passed"], "unit": "contract", "status": row["status"],
        "evidence_status": row["evidence_status"], "source_artifact": relative(source),
        "source_sha256": source_sha, "source_row_selector": f"gate_id={row['gate_id']}",
        "source_value_field": "observed", "notes": row["notes"],
    } for row in normalized]
    write_csv(path, fields, existing + additions)


def update_registries(output: Path, contrast_map: Mapping[tuple[str, str], Mapping[str, str]]) -> None:
    run = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    variant_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(variant_path.read_text(encoding="utf-8"))
    for row in registry["variants"]:
        if row["variant_id"] == c1.REFERENCE:
            row.update({
                "execution_status": "COMPLETE", "decision_status": "PROMOTED",
                "evidence_status": "DEVELOPMENT", "statistical_status": "DESCRIPTIVE",
                "rankable": True,
            })
        elif row["variant_id"] == c1.CANDIDATE:
            row.update({
                "execution_status": run["status"], "decision_status": "REJECTED",
                "evidence_status": "DEVELOPMENT", "statistical_status": "HARD_FAILURE",
                "rankable": False,
            })
    atomic_write_json(variant_path, registry)

    experiment_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P2_ATOMIC")
    experiment.update({
        "execution_status": "RUNNING", "opened_variants": [c1.CANDIDATE],
        "latest_variant": c1.CANDIDATE, "opened_primary_comparisons": 2,
        "multiplicity_family_size": 2,
        "c1_premise_gate": "HARD_FAIL_WORST_CELL",
        "swvar_reducer_template_status": "NOT_OPENED_BY_GATE",
    })
    atomic_write_json(experiment_path, experiments)

    primary = contrast_map[(c1.REFERENCE, "macro_f1")]
    top5 = contrast_map[("R1_ENTROPY_TOP5", "macro_f1")]
    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["claims"] = [row for row in claims["claims"] if row["claim_id"] != "CLAIM_C1_ENT_SW16"]
    claims["claims"].append({
        "claim_id": "CLAIM_C1_ENT_SW16",
        "text": (
            "Equal rank fusion of entropy level and causal trailing SWVar16 did not pass the atomic "
            "ProcessBench premise gate under the frozen top-ten parent."
        ),
        "verdict": "BLOCKED",
        "task_scope": "Current eight-Qwen ProcessBench development population under the P2 atomic contract.",
        "evidence_refs": [
            "PLOT_P2_DELTA_FOREST", "PLOT_P2_ATOMIC_GATE_MATRIX",
            "PLOT_P2_EXACT_CLEAN", "PLOT_P2_ATOMIC_LENGTH_HEATMAP",
            f"CONTRAST:{c1.CANDIDATE}:{c1.REFERENCE}",
        ],
        "worst_case_behavior": (
            f"Against atomic top-ten, cell W/T/L was {primary['wins']}/{primary['ties']}/{primary['losses']}; "
            f"the worst cell was {primary['worst_unit_id']} at {float(primary['worst_unit_delta']):+.6f}, "
            "crossing the preregistered -0.030 hard bound."
        ),
        "claim_boundary": (
            f"The average delta versus top-ten was {float(primary['delta']):+.6f} with interval "
            f"[{float(primary['ci_low']):+.6f}, {float(primary['ci_high']):+.6f}], which is not a "
            "supported-harm claim because it crosses zero. Versus top-five the delta was "
            f"{float(top5['delta']):+.6f} [{float(top5['ci_low']):+.6f}, {float(top5['ci_high']):+.6f}]. "
            "Rejection is caused by the robustness hard gate, not by interpreting uncertainty as failure."
        ),
        "statistical_summary": {
            "metric": "macro_f1", "point_delta": float(primary["delta"]),
            "ci_low": float(primary["ci_low"]), "ci_high": float(primary["ci_high"]),
            "benefit_bound": c1.BENEFIT, "harm_bound": c1.HARM,
            "bound_basis": "P2 atomic ProcessBench practical bounds",
            "multiplicity": primary["inference"],
        },
        "fresh_confirmation_required": False,
    })
    atomic_write_json(claims_path, claims)


def update_plots() -> None:
    path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["plot_id"]: row for row in manifest["plots"]}
    additions = [
        {
            "bootstrap_definition": "Gate rows bind the frozen C1 execution and paired atomic contrasts.",
            "caption": "C1 contract, robustness, and promotion gates; the worst-cell hard failure is distinct from interval uncertainty.",
            "comparison_group": "P2 atomic C1 gate roster", "kind": "gate_matrix",
            "legend": ["Green = pass", "Red = fail or hard fail", "Gray = pending"],
            "phase": "P2", "plot_id": "PLOT_P2_ATOMIC_GATE_MATRIX",
            "selection": {"experiment_id": "P2_ATOMIC", "status": "COMPLETE"},
            "selection_rule": "Every emitted gate from the frozen C1 execution.",
            "series_field": "passed", "source_table": "GATES_LONG.csv",
            "title": "C1 atomic premise gates", "x_field": "gate_id", "y_field": "variant_id",
        },
        {
            "bootstrap_definition": "Length strata are descriptive; primary inference remains paired by whole source question.",
            "caption": "C1 macro F1 by calibration-frozen short, medium, and long true-error step spans.",
            "comparison_group": "P2 atomic C1 step-length strata", "kind": "heatmap",
            "legend": ["Columns = short / medium / long", "Color = descriptive macro F1"],
            "phase": "P2", "plot_id": "PLOT_P2_ATOMIC_LENGTH_HEATMAP",
            "selection": {
                "experiment_id": "P2_ATOMIC", "variant_id": c1.CANDIDATE,
                "population_id": "current_common_eight_qwen_step_length", "metric_id": "macro_f1",
                "status": "COMPLETE",
            },
            "selection_rule": "All aggregate C1 calibration-frozen step-length strata.",
            "series_field": "value", "source_table": "METRICS_LONG.csv",
            "title": "C1 performance by true-error step length", "x_field": "cell_id", "y_field": "variant_id",
        },
    ]
    for plot in additions:
        if plot["plot_id"] in by_id:
            manifest["plots"][manifest["plots"].index(by_id[plot["plot_id"]])] = plot
        else:
            manifest["plots"].append(plot)
    atomic_write_json(path, manifest)


def main() -> None:
    output = c1.OUTPUT_ROOT
    run_path = output / "RUN_MANIFEST.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("variant_id") != c1.CANDIDATE or run.get("status") not in {"COMPLETE", "HARD_FAIL"}:
        raise RuntimeError("C1 run is not complete")
    if run["execution_registry_sha256"] != sha256_file(c1.REGISTRY_PATH):
        raise RuntimeError("C1 execution registry changed after run")
    add_metrics(output)
    contrasts = add_contrasts(output)
    add_gates(output)
    update_registries(output, contrasts)
    update_plots()
    subprocess.run(
        [sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")],
        cwd=REPO, check=True,
    )
    print(json.dumps({
        "variant_id": c1.CANDIDATE, "execution_status": run["status"],
        "decision_status": "REJECTED", "statistical_status": "HARD_FAILURE",
        "macro_f1": run["summary"]["candidate_macro_f1"],
        "delta_vs_top10": float(contrasts[(c1.REFERENCE, "macro_f1")]["delta"]),
        "delta_vs_top5": float(contrasts[("R1_ENTROPY_TOP5", "macro_f1")]["delta"]),
        "report_sha256": sha256_file(p1.PROGRAM_ROOT / "REPORT.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
