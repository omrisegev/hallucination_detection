#!/usr/bin/env python3
"""Integrate one completed reducer and recompute opened-family inference."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    load_npz_no_pickle,
    sha256_file,
)
from scripts.reasoning_localization import run_phase2_reducer as base  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer_candidate as candidate  # noqa: E402


BENEFIT = 0.005
HARM = -0.005


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
            "population_id": row["population_id"], "cell_id": "aggregate",
            "slice_id": "all",
            "metric_id": "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"],
            "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "notes": "candidate absolute metric under frozen top-five detector thresholds; not candidate-rethresholded",
        })
    _, cells = read_csv(output / "evaluation/PROCESSBENCH_BY_CELL.csv")
    for row in cells:
        rows.append({
            "source_id": f"cell::{row['cell_id']}::macro_f1",
            "population_id": "current_common_eight_qwen", "cell_id": row["cell_id"],
            "slice_id": row["slice_id"], "metric_id": "macro_f1",
            "value": row["official_macro_f1"], "ci_low": "", "ci_high": "",
            "n_rows": row["n_examples"], "n_groups": row["n_examples"],
            "notes": "descriptive per-cell candidate metric under the frozen threshold",
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
            "notes": "descriptive error-length stratum using the reference fold-frozen cut points and candidate predictions",
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
                "notes": "descriptive candidate selected-step length bias audit",
            })
    _, flips = read_csv(output / "evaluation/PREDICTION_FLIP_SUMMARY.csv")
    for row in flips:
        rows.append({
            "source_id": f"flip::{row['cell_id']}::{row['transition']}",
            "population_id": "current_common_eight_qwen_prediction_flips",
            "cell_id": row["cell_id"], "slice_id": row["transition"],
            "metric_id": "prediction_flip_count", "value": row["count"],
            "ci_low": "", "ci_high": "", "n_rows": row["count"],
            "n_groups": row["count"],
            "notes": "exact deterministic candidate-versus-top-five prediction transition count",
        })
    return rows


def completed_candidates() -> list[str]:
    output: list[str] = []
    for variant_id in candidate.CANDIDATES:
        run_path = base.PHASE_ROOT / variant_id.lower() / "RUN_MANIFEST.json"
        registry_path = base.PHASE_ROOT / f"{variant_id}_EXECUTION_REGISTRY.json"
        if not run_path.is_file() or not registry_path.is_file():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("variant_id") != variant_id or run.get("status") not in {"COMPLETE", "HARD_FAIL"}:
            raise RuntimeError(f"invalid completed reducer run: {variant_id}")
        if run["execution_registry_sha256"] != sha256_file(registry_path):
            raise RuntimeError(f"execution registry changed after run: {variant_id}")
        output.append(variant_id)
    return output


def build_contrasts(variants: list[str]) -> tuple[Path, list[dict[str, object]]]:
    if not variants:
        raise RuntimeError("no opened candidate contrasts")
    reference_samples = load_npz_no_pickle(
        candidate.REFERENCE_ROOT / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"
    )
    confirmatory = [
        variant for variant in variants
        if variant not in base.EXPLORATORY_STAGE_A_VARIANTS
    ]
    exploratory = [
        variant for variant in variants
        if variant in base.EXPLORATORY_STAGE_A_VARIANTS
    ]
    rows: list[dict[str, object]] = []
    for variant_id in variants:
        analysis_tier = (
            "POST_HOC_EXPLORATORY"
            if variant_id in base.EXPLORATORY_STAGE_A_VARIANTS
            else "PREREGISTERED_DEVELOPMENT"
        )
        family_size = (
            len(exploratory)
            if analysis_tier == "POST_HOC_EXPLORATORY"
            else len(confirmatory)
        )
        primary_q = 0.025 / family_size
        output = base.PHASE_ROOT / variant_id.lower()
        samples = load_npz_no_pickle(
            output / "evaluation/PROCESSBENCH_BOOTSTRAP_SAMPLES.npz"
        )
        _, raw_rows = read_csv(output / "evaluation/PAIRWISE_CONTRASTS.csv")
        by_metric = {row["metric_id"]: row for row in raw_rows}
        for metric, raw in by_metric.items():
            source_metric = raw["source_metric_id"]
            delta_samples = np.asarray(samples[source_metric]) - np.asarray(
                reference_samples[source_metric]
            )
            q = primary_q if metric == "macro_f1" else 0.025
            rows.append({
                "contrast_id": f"pb::{variant_id}::{base.REFERENCE}::{metric}",
                "left_variant_id": variant_id,
                "right_variant_id": base.REFERENCE,
                "metric_id": metric,
                "delta": raw["delta"],
                "ci_low": float(np.quantile(delta_samples, q)),
                "ci_high": float(np.quantile(delta_samples, 1.0 - q)),
                "p_adjusted": "",
                "wins": raw["cell_wins"], "ties": raw["cell_ties"],
                "losses": raw["cell_losses"],
                "worst_unit_delta": raw["worst_cell_delta"],
                "worst_unit_id": raw["worst_cell_id"],
                "family_wins": raw["family_wins"], "family_ties": raw["family_ties"],
                "family_losses": raw["family_losses"],
                "worst_family_delta": raw["worst_family_delta"],
                "worst_family_id": raw["worst_family_id"],
                "multiplicity_family_size": family_size if metric == "macro_f1" else 1,
                "analysis_tier": analysis_tier,
                "inference": (
                    (
                        f"descriptive Bonferroni interval across {family_size} post-hoc tail-fraction amendments; promotion forbidden on this opened population"
                        if analysis_tier == "POST_HOC_EXPLORATORY"
                        else f"Bonferroni simultaneous percentile interval across {family_size} preregistered reducer primary contrasts"
                    )
                    if metric == "macro_f1"
                    else "unadjusted paired diagnostic percentile interval"
                ),
            })
    root = base.PHASE_ROOT / "reducer_interim"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "P2R_CONTRASTS.csv"
    write_csv(path, list(rows[0]), rows)
    return path, rows


def statistical_status(row: Mapping[str, object], hard_failure: bool) -> str:
    if hard_failure:
        return "HARD_FAILURE"
    delta = float(row["delta"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if low > BENEFIT:
        return "SUPPORTED_IMPROVEMENT"
    if high < HARM:
        return "SUPPORTED_HARM"
    if delta > 0.0 and low <= 0.0 <= high:
        return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def promotion_checks(row: Mapping[str, object], raw_by_metric: Mapping[str, Mapping[str, str]]) -> dict[str, tuple[object, str, str, bool, str]]:
    exact = float(raw_by_metric["first_error_exact"]["delta"])
    abstention = float(raw_by_metric["clean_abstention_accuracy"]["delta"])
    return {
        "P2R_POINT_BENEFIT": (row["delta"], f">= {BENEFIT}", "ge", float(row["delta"]) >= BENEFIT, "candidate macro-F1 point delta versus top-five"),
        "P2R_SIMULTANEOUS_CI_BENEFIT": (row["ci_low"], f"> {BENEFIT}", "gt", float(row["ci_low"]) > BENEFIT, "multiplicity-aware lower bound"),
        "P2R_NONNEGATIVE_CELLS": (row["wins"] + "/" + row["ties"], ">= 6/8 nonnegative", "ge", int(row["wins"]) + int(row["ties"]) >= 6, "cell W/T/L excludes no registered cell"),
        "P2R_WORST_CELL_PROMOTION": (row["worst_unit_delta"], ">= -0.020", "ge", float(row["worst_unit_delta"]) >= -0.020, f"worst cell {row['worst_unit_id']}"),
        "P2R_EXACT_ERROR_COMPONENT": (exact, ">= -0.010", "ge", exact >= -0.010, "paired exact-error point delta"),
        "P2R_CLEAN_ABSTENTION_COMPONENT": (abstention, ">= -0.010", "ge", abstention >= -0.010, "paired clean-abstention point delta"),
    }


def integrate_metrics(variant_id: str, output: Path) -> None:
    metrics = source_metrics(output)
    source_path = output / "evaluation/REPORT_METRICS.csv"
    write_csv(source_path, list(metrics[0]), metrics)
    source_sha = sha256_file(source_path)
    metric_path = base.PROGRAM_ROOT / "METRICS_LONG.csv"
    fields, existing = read_csv(metric_path)
    existing = [
        row for row in existing
        if not (row["phase_id"] == "P2" and row["variant_id"] == variant_id)
    ]
    additions = []
    for position, row in enumerate(metrics):
        additions.append({
            "phase_id": "P2", "experiment_id": "P2_REDUCER_STUDY",
            "variant_id": variant_id, "task_id": "processbench_first_error",
            "dataset_id": "processbench", "population_id": row["population_id"],
            "cell_id": row["cell_id"], "slice_id": row["slice_id"],
            "metric_id": row["metric_id"], "value": row["value"],
            "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "comparison_group_id": f"p2r::{row['population_id']}::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
            "display_order": 3100 + base.STAGE_A_VARIANTS.index(variant_id) * 100 + position,
            "axis_value": "", "source_artifact": repo_relative(source_path),
            "source_sha256": source_sha,
            "source_row_selector": f"source_id={row['source_id']}",
            "source_value_field": "value", "notes": row["notes"],
        })
    write_csv(metric_path, fields, existing + additions)


def integrate_contrasts(source_path: Path, rows: list[dict[str, object]]) -> None:
    source_sha = sha256_file(source_path)
    path = base.PROGRAM_ROOT / "CONTRASTS_LONG.csv"
    fields, existing = read_csv(path)
    existing = [row for row in existing if row["experiment_id"] != "P2_REDUCER_STUDY"]
    additions = [{
        "phase_id": "P2", "experiment_id": "P2_REDUCER_STUDY",
        "left_variant_id": row["left_variant_id"],
        "right_variant_id": row["right_variant_id"],
        "task_id": "processbench_first_error", "dataset_id": "processbench",
        "population_id": "current_common_eight_qwen", "metric_id": row["metric_id"],
        "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        "p_adjusted": row["p_adjusted"], "wins": row["wins"],
        "ties": row["ties"], "losses": row["losses"],
        "worst_unit_delta": row["worst_unit_delta"],
        "comparison_group_id": f"p2r::processbench::{row['metric_id']}",
        "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
        "source_artifact": repo_relative(source_path), "source_sha256": source_sha,
        "source_row_selector": f"contrast_id={row['contrast_id']}",
        "notes": f"{row['inference']}; family W/T/L {row['family_wins']}/{row['family_ties']}/{row['family_losses']}; worst family {row['worst_family_id']} {float(row['worst_family_delta']):+.6f}; practical bounds +0.005/-0.005",
    } for row in rows]
    write_csv(path, fields, existing + additions)


def integrate_gates(
    variant_id: str,
    output: Path,
    primary: Mapping[str, object],
    raw_by_metric: Mapping[str, Mapping[str, str]],
) -> dict[str, bool]:
    _, raw = read_csv(output / "evaluation/GATES.csv")
    normalized: list[dict[str, object]] = []
    for row in raw:
        passed = row["status"] == "PASS"
        normalized.append({
            "gate_id": row["gate_id"], "observed": row["observed"],
            "threshold": row["required"], "direction": "contract",
            "passed": str(passed).lower(), "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "notes": row["detail"] + f"; raw gate status={row['status']}",
        })
    checks = promotion_checks(primary, raw_by_metric)
    for gate_id, (observed, threshold, direction, passed, note) in checks.items():
        normalized.append({
            "gate_id": gate_id, "observed": observed, "threshold": threshold,
            "direction": direction, "passed": str(passed).lower(),
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT", "notes": note,
        })
    if variant_id in base.EXPLORATORY_STAGE_A_VARIANTS:
        normalized.append({
            "gate_id": "P2R_POSTHOC_PROMOTION_ELIGIBILITY",
            "observed": "post-hoc after ProcessBench reducer outcomes opened",
            "threshold": "fresh confirmation required",
            "direction": "contract",
            "passed": "false",
            "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "notes": "This amendment is descriptive on the current population and cannot promote regardless of its point estimate or interval.",
        })
    source = output / "evaluation/REPORT_GATES.csv"
    write_csv(source, list(normalized[0]), normalized)
    source_sha = sha256_file(source)
    gate_path = base.PROGRAM_ROOT / "GATES_LONG.csv"
    fields, existing = read_csv(gate_path)
    existing = [
        row for row in existing
        if not (row["phase_id"] == "P2" and row["variant_id"] == variant_id)
    ]
    additions = [{
        "phase_id": "P2", "experiment_id": "P2_REDUCER_STUDY",
        "variant_id": variant_id, "gate_id": row["gate_id"],
        "metric_id": "contract_or_promotion_gate", "observed": row["observed"],
        "threshold": row["threshold"], "direction": row["direction"],
        "passed": row["passed"], "unit": "contract", "status": row["status"],
        "evidence_status": row["evidence_status"],
        "source_artifact": repo_relative(source), "source_sha256": source_sha,
        "source_row_selector": f"gate_id={row['gate_id']}",
        "source_value_field": "observed", "notes": row["notes"],
    } for row in normalized]
    write_csv(gate_path, fields, existing + additions)
    return {str(row["gate_id"]): str(row["passed"]) == "true" for row in normalized}


def update_plot_contract() -> None:
    path = base.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    plot = next(row for row in manifest["plots"] if row["plot_id"] == "PLOT_P2_GATE_MATRIX")
    plot["selection"]["experiment_id"] = "P2_REDUCER_STUDY"
    plot["caption"] = "Reducer candidates promote only when every required gate passes; failed or pending gates remain explicit."
    plot["selection_rule"] = "All required reducer-study gates for every attempted candidate."
    atomic_write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=candidate.CANDIDATES)
    args = parser.parse_args()
    variant_id = args.variant
    output = base.PHASE_ROOT / variant_id.lower()
    run_path = output / "RUN_MANIFEST.json"
    registry_path = base.PHASE_ROOT / f"{variant_id}_EXECUTION_REGISTRY.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("variant_id") != variant_id or run.get("status") not in {"COMPLETE", "HARD_FAIL"}:
        raise RuntimeError("candidate run is not complete")
    if run["execution_registry_sha256"] != sha256_file(registry_path):
        raise RuntimeError("candidate execution registry changed")

    integrate_metrics(variant_id, output)
    opened = completed_candidates()
    contrast_source, contrast_rows = build_contrasts(opened)
    integrate_contrasts(contrast_source, contrast_rows)
    primary_by_variant = {
        str(row["left_variant_id"]): row for row in contrast_rows
        if row["metric_id"] == "macro_f1"
    }
    raw_by_variant: dict[str, dict[str, dict[str, str]]] = {}
    for opened_variant in opened:
        _, raw = read_csv(
            base.PHASE_ROOT / opened_variant.lower() / "evaluation/PAIRWISE_CONTRASTS.csv"
        )
        raw_by_variant[opened_variant] = {row["metric_id"]: row for row in raw}

    gate_passes: dict[str, dict[str, bool]] = {}
    for opened_variant in opened:
        gate_passes[opened_variant] = integrate_gates(
            opened_variant,
            base.PHASE_ROOT / opened_variant.lower(),
            primary_by_variant[opened_variant],
            raw_by_variant[opened_variant],
        )

    variant_path = base.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(variant_path.read_text(encoding="utf-8"))
    statuses: dict[str, str] = {}
    for row in registry["variants"]:
        opened_variant = row["variant_id"]
        if opened_variant not in opened:
            continue
        opened_run = json.loads(
            (base.PHASE_ROOT / opened_variant.lower() / "RUN_MANIFEST.json").read_text(encoding="utf-8")
        )
        hard_failure = opened_run["status"] == "HARD_FAIL"
        status = (
            "DESCRIPTIVE"
            if opened_variant in base.EXPLORATORY_STAGE_A_VARIANTS
            else statistical_status(primary_by_variant[opened_variant], hard_failure)
        )
        statuses[opened_variant] = status
        row["execution_status"] = opened_run["status"]
        row["statistical_status"] = status
        passes = gate_passes[opened_variant]
        promoted = all(passes.get(gate_id, False) for gate_id in (
            "P2R_POINT_BENEFIT", "P2R_SIMULTANEOUS_CI_BENEFIT",
            "P2R_NONNEGATIVE_CELLS", "P2R_WORST_CELL_PROMOTION",
            "P2R_EXACT_ERROR_COMPONENT", "P2R_CLEAN_ABSTENTION_COMPONENT",
        )) and not hard_failure
        if opened_variant in base.EXPLORATORY_STAGE_A_VARIANTS:
            row["decision_status"] = "NO_PROMOTION"
            row["rankable"] = False
        elif promoted:
            row["decision_status"] = "PROMOTED"
            row["rankable"] = True
        elif status in {"SUPPORTED_HARM", "HARD_FAILURE"}:
            row["decision_status"] = "REJECTED"
            row["rankable"] = False
        else:
            row["decision_status"] = "NO_PROMOTION"
            row["rankable"] = False
    atomic_write_json(variant_path, registry)

    experiment_path = base.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"]
        if row["experiment_id"] == "P2_REDUCER_STUDY"
    )
    experiment["execution_status"] = "RUNNING"
    experiment["opened_stage_a_variants"] = [base.REFERENCE, *opened]
    experiment["opened_primary_contrasts"] = len(opened)
    experiment["multiplicity_family_size"] = len([
        variant for variant in opened
        if variant not in base.EXPLORATORY_STAGE_A_VARIANTS
    ])
    experiment["posthoc_multiplicity_family_size"] = len([
        variant for variant in opened
        if variant in base.EXPLORATORY_STAGE_A_VARIANTS
    ])
    experiment["opened_posthoc_contrasts"] = len([
        variant for variant in opened
        if variant in base.EXPLORATORY_STAGE_A_VARIANTS
    ])
    experiment["latest_variant"] = variant_id
    atomic_write_json(experiment_path, experiments)

    summary = json.loads((output / "evaluation/SUMMARY.json").read_text(encoding="utf-8"))
    claims_path = base.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    opened_claim_ids = {f"CLAIM_{opened_variant}" for opened_variant in opened}
    claims["claims"] = [
        row for row in claims["claims"] if row["claim_id"] not in opened_claim_ids
    ]
    for opened_variant in opened:
        primary = primary_by_variant[opened_variant]
        status = statuses[opened_variant]
        claim_verdict = (
            "DESCRIPTIVE"
            if opened_variant in base.EXPLORATORY_STAGE_A_VARIANTS
            else (
                "SUPPORTED_HARM"
                if float(primary["ci_high"]) < HARM
                else ("BLOCKED" if status == "HARD_FAILURE" else status)
            )
        )
        claims["claims"].append({
            "claim_id": f"CLAIM_{opened_variant}",
            "text": f"{opened_variant} changes only the within-step reducer from top-five mean to the registered candidate and has a ProcessBench macro-F1 delta of {float(primary['delta']):+.6f} under the frozen reference thresholds.",
            "verdict": claim_verdict,
            "task_scope": "Current eight-Qwen ProcessBench development population under the Phase-2 reducer contract.",
            "evidence_refs": [
                "PLOT_P2_REDUCER_DELTA_FOREST", "PLOT_P2_REDUCER_LENGTH_HEATMAP",
                "PLOT_P2_GATE_MATRIX",
                f"CONTRAST:{opened_variant}:{base.REFERENCE}",
            ],
            "worst_case_behavior": f"Worst cell is {primary['worst_unit_id']} at {float(primary['worst_unit_delta']):+.6f}; family W/T/L is {primary['family_wins']}/{primary['family_ties']}/{primary['family_losses']} and worst family is {primary['worst_family_id']} at {float(primary['worst_family_delta']):+.6f}.",
            "claim_boundary": (
                "This reducer was requested after ProcessBench reducer outcomes were opened. Its separate descriptive multiplicity interval cannot support promotion; fresh confirmation is required."
                if opened_variant in base.EXPLORATORY_STAGE_A_VARIANTS
                else "This is selection-opened development evidence in the closed preregistered family; no PRMBench transfer or cross-task aggregate is implied."
            ),
            "statistical_summary": {
                "metric": "macro_f1", "point_delta": float(primary["delta"]),
                "ci_low": float(primary["ci_low"]), "ci_high": float(primary["ci_high"]),
                "benefit_bound": BENEFIT, "harm_bound": HARM,
                "bound_basis": "Phase-2 ProcessBench reducer practical bounds",
                "multiplicity": primary["inference"],
            },
            "fresh_confirmation_required": True,
        })
    atomic_write_json(claims_path, claims)
    update_plot_contract()
    subprocess.run(
        [sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")],
        cwd=REPO, check=True,
    )
    print(json.dumps({
        "variant_id": variant_id,
        "execution_status": run["status"],
        "statistical_status": statuses[variant_id],
        "decision_status": next(
            row["decision_status"] for row in registry["variants"]
            if row["variant_id"] == variant_id
        ),
        "processbench_qwen8_macro_f1": summary["processbench_qwen8_macro_f1"],
        "paired_delta_macro_f1": primary_by_variant[variant_id]["delta"],
        "simultaneous_ci": [
            primary_by_variant[variant_id]["ci_low"],
            primary_by_variant[variant_id]["ci_high"],
        ],
        "opened_primary_contrasts": len(opened),
        "report_sha256": sha256_file(base.PROGRAM_ROOT / "REPORT.html"),
    }, indent=2))


if __name__ == "__main__":
    main()
