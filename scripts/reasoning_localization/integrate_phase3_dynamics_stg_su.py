#!/usr/bin/env python3
"""Integrate the completed dynamics-only SU/STG-SU ladder into the living report."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization import run_phase3_dynamics_stg_su as run  # noqa: E402


EXPERIMENT = run.EXPERIMENT
EVAL = run.OUTPUT / "evaluation"
H0 = run.H0
H0_REPORT = "P2C_F6_TOP10_REFERENCE"
S0, S1, S2, S3, S4 = run.VARIANTS
ORDERS = {H0_REPORT: 169, S0: 188, S1: 189, S2: 190, S3: 191, S4: 192}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in columns} for row in rows)


def append(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    rows = read(path)
    fields = list(rows[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in rows}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"duplicate reporting key {key}")
        keys.add(key)
    write(path, [*rows, *additions], fields)


def alias(value: str) -> str:
    return H0_REPORT if value == H0 else value


def upsert(rows: list[dict[str, Any]], key: str, row: dict[str, Any]) -> None:
    hits = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)


def main() -> None:
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"
    panels = read(panels_path)
    metrics = []
    for source in panels:
        metric = "macro_f1" if source["metric_id"] == "official_macro_f1" else source["metric_id"]
        variant = alias(source["arm_id"])
        metrics.append({
            "phase_id": "P3", "experiment_id": EXPERIMENT, "variant_id": variant,
            "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen", "cell_id": "aggregate", "slice_id": "all",
            "metric_id": metric, "value": source["value"], "ci_low": source["ci_low"],
            "ci_high": source["ci_high"], "n_rows": source["n_rows"], "n_groups": source["n_groups"],
            "comparison_group_id": f"p3s_dynamics_stg::{metric}", "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT", "display_order": ORDERS[variant], "axis_value": "",
            "source_artifact": str(panels_path.relative_to(REPO)), "source_sha256": sha256_file(panels_path),
            "source_row_selector": f"arm_id={source['arm_id']};metric_id={source['metric_id']}",
            "source_value_field": "value",
            "notes": "Five outer donor folds; dynamics-only SU/STG-SU family expert; H0 and top-ten fixed.",
        })
    append(p1.PROGRAM_ROOT / "METRICS_LONG.csv", metrics, ("experiment_id", "variant_id", "metric_id", "cell_id"))

    contrasts_path = EVAL / "PAIRWISE_CONTRASTS.csv"
    raw_contrasts = read(contrasts_path)
    contrasts = []
    for source in raw_contrasts:
        contrasts.append({
            "phase_id": "P3", "experiment_id": EXPERIMENT,
            "left_variant_id": alias(source["left_variant_id"]), "right_variant_id": alias(source["right_variant_id"]),
            "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen", "metric_id": source["metric_id"],
            "delta": source["delta"], "ci_low": source["ci_low"], "ci_high": source["ci_high"],
            "p_adjusted": "", "wins": source["wins"], "ties": source["ties"], "losses": source["losses"],
            "worst_unit_delta": source["worst_unit_delta"],
            "comparison_group_id": f"p3s_dynamics_stg::{source['metric_id']}", "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT", "source_artifact": str(contrasts_path.relative_to(REPO)),
            "source_sha256": sha256_file(contrasts_path),
            "source_row_selector": f"left_variant_id={source['left_variant_id']};right_variant_id={source['right_variant_id']};metric_id={source['metric_id']}",
            "notes": f"{source['statistical_status']}; worst={source['worst_unit_id']}; Bonferroni family={source['multiplicity_family_size']}.",
        })
    append(p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv", contrasts, ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"))

    summary = json.loads((EVAL / "SUMMARY.json").read_text(encoding="utf-8"))
    macro = {(r["left_variant_id"], r["right_variant_id"]): r for r in raw_contrasts if r["metric_id"] == "macro_f1"}
    exact = {(r["left_variant_id"], r["right_variant_id"]): r for r in raw_contrasts if r["metric_id"] == "first_error_exact"}

    def gate(gate_id: str, variant: str, metric: str, observed: Any, threshold: Any, direction: str) -> dict[str, Any]:
        value = float(observed)
        passed = {"ge": value >= float(threshold), "gt": value > float(threshold), "le": value <= float(threshold), "eq": value == float(threshold)}[direction]
        return {"gate_id": gate_id, "variant_id": variant, "metric_id": metric, "observed": observed,
                "threshold": threshold, "direction": direction, "passed": str(passed).lower(),
                "unit": "fraction", "status": "PASS" if passed else "FAIL", "evidence_status": "DEVELOPMENT"}

    gate_rows = [
        gate("P3S_ALIAS_P3E", S0, "max_parent_alias_error", summary["p3e_parent_alias_max_error"], 1e-12, "le"),
        gate("P3S_ABSTENTION_ALIAS", S2, "abstention_mismatches", max(summary["abstention_mismatches"].values()), 0, "eq"),
        gate("P3S2_POINT_VS_IU", S2, "macro_f1", macro[(S2, S0)]["delta"], run.BENEFIT, "ge"),
        gate("P3S2_CI_VS_IU", S2, "macro_f1_ci_low", macro[(S2, S0)]["ci_low"], run.BENEFIT, "gt"),
        gate("P3S2_EXACT_VS_IU", S2, "first_error_exact", exact[(S2, S0)]["delta"], -0.010, "ge"),
        gate("P3S2_WORST_VS_IU", S2, "worst_cell_delta", macro[(S2, S0)]["worst_unit_delta"], -0.020, "ge"),
        gate("P3S2_VS_PERMUTATION", S2, "macro_f1_ci_low", macro[(S2, S3)]["ci_low"], 0.0, "gt"),
        gate("P3S2_VS_RANDOM", S2, "macro_f1_ci_low", macro[(S2, S4)]["ci_low"], 0.0, "gt"),
        gate("P3S_SUPPORT_MECHANISM", S2, "mechanism_supported", int(summary["support_mechanism_supported"]), 1, "eq"),
    ]
    gate_source = EVAL / "REPORTING_GATES.csv"
    write(gate_source, gate_rows)
    gates = [{
        "phase_id": "P3", "experiment_id": EXPERIMENT, **row,
        "source_artifact": str(gate_source.relative_to(REPO)), "source_sha256": sha256_file(gate_source),
        "source_row_selector": f"gate_id={row['gate_id']}", "source_value_field": "observed",
        "notes": "A crossing CI is inconclusive or promising-unconfirmed; it is not generic rejection.",
    } for row in gate_rows]
    append(p1.PROGRAM_ROOT / "GATES_LONG.csv", gates, ("experiment_id", "variant_id", "gate_id"))

    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    panel_f1 = {r["arm_id"]: float(r["value"]) for r in panels if r["metric_id"] == "official_macro_f1"}
    statuses = {
        S0: ("NO_PROMOTION", "INCONCLUSIVE", f"Exact P3E1 parent alias; F1={panel_f1[S0]:.6f}."),
        S1: ("NO_PROMOTION", macro[(S1, S0)]["statistical_status"], f"F1={panel_f1[S1]:.6f}; canonical SU delta {float(macro[(S1,S0)]['delta']):+.6f}; theorem flag is reported diagnostically."),
        S2: ("NO_PROMOTION", macro[(S2, S0)]["statistical_status"], f"F1={panel_f1[S2]:.6f}; STG-SU delta {float(macro[(S2,S0)]['delta']):+.6f}; support mechanism not supported."),
        S3: ("NO_PROMOTION", "INCONCLUSIVE", f"Permutation control F1={panel_f1[S3]:.6f}; no learned-support claim."),
        S4: ("NO_PROMOTION", "INCONCLUSIVE", f"Random-support control F1={panel_f1[S4]:.6f}; no learned-support claim."),
    }
    for variant, (decision, statistical, limitations) in statuses.items():
        row = next(item for item in variants["variants"] if item["variant_id"] == variant)
        row.update({"execution_status": "COMPLETE", "decision_status": decision, "statistical_status": statistical, "limitations": limitations})
    atomic_write_json(variants_path, variants)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = next(item for item in experiments["experiments"] if item["experiment_id"] == EXPERIMENT)
    experiment.update({"execution_status": "COMPLETE", "next_variant": None,
                       "verdict": "STG_PREMISE_INCONCLUSIVE__NO_PROMOTION",
                       "result_summary": "STG-SU is -0.000114 vs IU parent (CI [-0.002964,+0.002697]); learned support is not separated from controls."})
    atomic_write_json(experiments_path, experiments)

    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    primary = macro[(S2, S0)]
    mechanism_perm = macro[(S2, S3)]
    upsert(claims["claims"], "claim_id", {
        "claim_id": "CLAIM_P3S_DYNAMICS_STG_SU",
        "text": "On the opened eight-Qwen ProcessBench population, nested STG-SU did not establish an improvement over the dynamics-IU parent; the learned-support contrast is unresolved and the permutation/random controls do not support a mechanism claim.",
        "verdict": "INCONCLUSIVE",
        "task_scope": "Current common eight-Qwen ProcessBench first-error development population; five outer folds and nested five-fold donor-only STG.",
        "evidence_refs": ["PLOT_P3S_DYNAMICS_STG", f"CONTRAST:{S2}:{S0}", f"CONTRAST:{S2}:{S3}", f"CONTRAST:{S2}:{S4}", "TABLE_GATES"],
        "fresh_confirmation_required": True,
        "statistical_summary": {"metric": "macro_f1", "point_delta": float(primary["delta"]), "ci_low": float(primary["ci_low"]), "ci_high": float(primary["ci_high"]), "benefit_bound": run.BENEFIT, "harm_bound": run.HARM, "bound_basis": "Frozen P3S practical bounds; multiplicity family size five.", "multiplicity": "Bonferroni-simultaneous across five preregistered contrasts."},
        "worst_case_behavior": f"Worst STG-vs-IU cell {float(primary['worst_unit_delta']):+.6f}; STG-vs-permutation {float(mechanism_perm['delta']):+.6f} with CI [{float(mechanism_perm['ci_low']):+.6f},{float(mechanism_perm['ci_high']):+.6f}].",
        "claim_boundary": "Development-only opened population; canonical theorem validity is reported as a control diagnostic, while STG and controls passed finite/convergence/theorem checks. No PRMBench transfer opens.",
    })
    atomic_write_json(claims_path, claims)

    plots_path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(plots_path.read_text(encoding="utf-8"))
    upsert(plots["plots"], "plot_id", {
        "plot_id": "PLOT_P3S_DYNAMICS_STG", "title": "Dynamics IU, canonical SU and STG-SU contrasts", "phase": "P3",
        "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv",
        "selection": {"experiment_id": EXPERIMENT, "metric_id": "macro_f1", "status": "COMPLETE"},
        "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id",
        "comparison_group": "same eight-Qwen rows; five outer donor folds; H0 and top-ten fixed; one dynamics-family change",
        "bootstrap_definition": "20,000 paired whole-question draws; Bonferroni simultaneous intervals across five frozen primary contrasts.",
        "selection_rule": "All five preregistered SU/STG contrasts, including permutation and cardinality-matched random-support controls.",
        "legend": ["Canonical SU is a diagnostic control", "Positive CI crossing zero is promising-unconfirmed", "STG support mechanism requires both controls to clear zero"],
        "caption": "STG-SU is near parity with the IU parent and does not clear its mechanism controls; all rows preserve H0 abstention exactly.",
    })
    atomic_write_json(plots_path, plots)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"status": "INTEGRATED", "report_sha256": build.manifest["output"]["sha256"], "experiment": EXPERIMENT}, indent=2))


if __name__ == "__main__":
    main()
