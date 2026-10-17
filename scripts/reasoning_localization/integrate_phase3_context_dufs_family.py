#!/usr/bin/env python3
"""Integrate completed P3F contextual-DUFS family results."""

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
from scripts.reasoning_localization import run_phase3_context_dufs_family as run  # noqa: E402

EXPERIMENT = run.EXPERIMENT
BENEFIT = run.BENEFIT
ALIAS_TOLERANCE = run.ALIAS_TOLERANCE
F0, F1, F2, F3 = run.VARIANTS
EVAL = run.OUTPUT / "evaluation"
H0_REPORT = "P2C_F6_TOP10_REFERENCE"
ORDERS = {run.H0: 169, F0: 182, F1: 183, F2: 184, F3: 185}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in columns} for row in rows])


def append(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    rows = read(path)
    fields = list(rows[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in rows}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"duplicate {key}")
        keys.add(key)
    write(path, [*rows, *additions], fields)


def upsert(rows: list[dict], key: str, row: dict) -> None:
    hits = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate {key}")
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)


def alias(value: str) -> str:
    return H0_REPORT if value == run.H0 else value


def main() -> None:
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"
    panels = read(panels_path)
    metrics = []
    for row in panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metrics.append({
            "phase_id": "P3", "experiment_id": EXPERIMENT,
            "variant_id": alias(row["arm_id"]), "task_id": "processbench_first_error",
            "dataset_id": "processbench", "population_id": "current_common_eight_qwen",
            "cell_id": "aggregate", "slice_id": "all", "metric_id": metric,
            "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "n_rows": row["n_rows"], "n_groups": row["n_groups"],
            "comparison_group_id": f"p3f_context_dufs::{metric}", "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT", "display_order": ORDERS[row["arm_id"]],
            "axis_value": "", "source_artifact": str(panels_path.relative_to(REPO)),
            "source_sha256": sha256_file(panels_path),
            "source_row_selector": f"arm_id={row['arm_id']};metric_id={row['metric_id']}",
            "source_value_field": "value",
            "notes": "Five-fold donor-only DUFS/LIU; all-H2 context affects only the dynamics graph; H0 and top-ten fixed.",
        })
    append(p1.PROGRAM_ROOT / "METRICS_LONG.csv", metrics, ("experiment_id", "variant_id", "metric_id", "cell_id"))

    contrasts_path = EVAL / "PAIRWISE_CONTRASTS.csv"
    raw = read(contrasts_path)
    contrasts = []
    for row in raw:
        contrasts.append({
            "phase_id": "P3", "experiment_id": EXPERIMENT,
            "left_variant_id": alias(row["left_variant_id"]), "right_variant_id": alias(row["right_variant_id"]),
            "task_id": "processbench_first_error", "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen", "metric_id": row["metric_id"],
            "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "p_adjusted": "", "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
            "worst_unit_delta": row["worst_unit_delta"],
            "comparison_group_id": f"p3f_context_dufs::{row['metric_id']}", "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT", "source_artifact": str(contrasts_path.relative_to(REPO)),
            "source_sha256": sha256_file(contrasts_path),
            "source_row_selector": f"left_variant_id={row['left_variant_id']};right_variant_id={row['right_variant_id']};metric_id={row['metric_id']}",
            "notes": f"{row['statistical_status']}; worst={row['worst_unit_id']}; family={row['multiplicity_family_size']}.",
        })
    append(p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv", contrasts, ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"))

    summary = json.loads((EVAL / "SUMMARY.json").read_text())
    macro = {(row["left_variant_id"], row["right_variant_id"]): row for row in raw if row["metric_id"] == "macro_f1"}
    exact = {(row["left_variant_id"], row["right_variant_id"]): row for row in raw if row["metric_id"] == "first_error_exact"}
    gate_rows = []
    checks = [
        ("F1_POINT", F1, "macro_f1", macro[(F1, F0)]["delta"], BENEFIT, "ge"),
        ("F1_SIMULTANEOUS", F1, "macro_f1_ci_low", macro[(F1, F0)]["ci_low"], BENEFIT, "gt"),
        ("F2_POINT", F2, "macro_f1", macro[(F2, F0)]["delta"], BENEFIT, "ge"),
        ("F2_SIMULTANEOUS", F2, "macro_f1_ci_low", macro[(F2, F0)]["ci_low"], BENEFIT, "gt"),
        ("F2_EXACT", F2, "first_error_exact", exact[(F2, F0)]["delta"], -.010, "ge"),
        ("F2_WORST", F2, "worst_cell_delta", macro[(F2, F0)]["worst_unit_delta"], -.020, "ge"),
        ("CONTEXT_VS_LOCAL", F2, "macro_f1_ci_low", macro[(F2, F1)]["ci_low"], 0.0, "gt"),
        ("CONTEXT_VS_PERM", F2, "macro_f1_ci_low", macro[(F2, F3)]["ci_low"], 0.0, "gt"),
        ("ALIAS_PARENT", F0, "max_abs_error", summary["alias_max_errors"]["p3e_parent"], ALIAS_TOLERANCE, "le"),
        ("ALIAS_LAMBDA_ZERO", F2, "max_abs_error", max(summary["alias_max_errors"].values()), ALIAS_TOLERANCE, "le"),
        ("ABSTENTION_ALIAS", F2, "mismatches", max(summary["abstention_mismatches"].values()), 0.0, "eq"),
    ]
    for gate_id, variant, metric, observed, threshold, direction in checks:
        value = float(observed)
        if direction == "ge": passed = value >= threshold
        elif direction == "gt": passed = value > threshold
        elif direction == "le": passed = value <= threshold
        else: passed = value == threshold
        gate_rows.append({
            "gate_id": f"P3F_{gate_id}", "variant_id": variant, "metric_id": metric,
            "observed": observed, "threshold": threshold, "direction": direction,
            "passed": str(passed).lower(), "unit": "fraction", "status": "PASS" if passed else "FAIL",
            "evidence_status": "DEVELOPMENT",
        })
    gate_source = EVAL / "REPORTING_GATES.csv"
    write(gate_source, gate_rows)
    gates = [{
        "phase_id": "P3", "experiment_id": EXPERIMENT, **row,
        "source_artifact": str(gate_source.relative_to(REPO)), "source_sha256": sha256_file(gate_source),
        "source_row_selector": f"gate_id={row['gate_id']}", "source_value_field": "observed",
        "notes": "Failed improvement gate means no promotion; a crossing interval is not generic rejection.",
    } for row in gate_rows]
    append(p1.PROGRAM_ROOT / "GATES_LONG.csv", gates, ("experiment_id", "variant_id", "gate_id"))

    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(variants_path.read_text())
    panel_f1 = {row["arm_id"]: float(row["value"]) for row in panels if row["metric_id"] == "official_macro_f1"}
    status_rows = {
        F0: ("NO_PROMOTION", "INCONCLUSIVE", f"Exact P3E1 parent alias; F1={panel_f1[F0]:.6f}."),
        F1: ("NO_PROMOTION", macro[(F1, F0)]["statistical_status"], f"F1={panel_f1[F1]:.6f}; delta vs parent {float(macro[(F1,F0)]['delta']):+.6f}."),
        F2: ("NO_PROMOTION", macro[(F2, F0)]["statistical_status"], f"F1={panel_f1[F2]:.6f}; delta vs parent {float(macro[(F2,F0)]['delta']):+.6f}; context mechanism supported={summary['context_mechanism_supported']}."),
        F3: ("NO_PROMOTION", macro[(F3, F0)]["statistical_status"], f"Negative control F1={panel_f1[F3]:.6f}; delta vs parent {float(macro[(F3,F0)]['delta']):+.6f}."),
    }
    for variant, (decision, statistical, limitations) in status_rows.items():
        row = next(item for item in registry["variants"] if item["variant_id"] == variant)
        row.update({"execution_status": "COMPLETE", "decision_status": decision, "statistical_status": statistical, "limitations": limitations})
    atomic_write_json(variants_path, registry)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == EXPERIMENT)
    experiment.update({
        "execution_status": "COMPLETE", "next_variant": None,
        "verdict": "CONTEXT_MECHANISM_SUPPORTED" if summary["context_mechanism_supported"] else "CONTEXT_MECHANISM_NOT_SUPPORTED",
        "topk_secondary_control_eligible": summary["topk_secondary_control_eligible"],
    })
    next(row for row in experiments["experiments"] if row["experiment_id"] == "P3_FUSION")["next_variant"] = None
    atomic_write_json(experiments_path, experiments)

    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text())
    primary = macro[(F2, F0)]
    mechanism = macro[(F2, F3)]
    verdict = "SUPPORTED" if summary["context_mechanism_supported"] else (
        "PROMISING_UNCONFIRMED" if float(primary["delta"]) > 0 else "INCONCLUSIVE"
    )
    upsert(claims["claims"], "claim_id", {
        "claim_id": "CLAIM_P3F_CONTEXT_DUFS_FAMILY",
        "text": "The all-H2 contextual DUFS graph was tested as an indirect conditioner of the dynamics-only family expert, with local-DUFS and aligned-context controls.",
        "verdict": verdict,
        "task_scope": "Current common eight-Qwen ProcessBench first-error development population; five-fold donor cross-fit.",
        "evidence_refs": ["PLOT_P3F_CONTEXT_DUFS", f"CONTRAST:{F2}:{F0}", f"CONTRAST:{F2}:{F3}", "TABLE_GATES"],
        "fresh_confirmation_required": True,
        "statistical_summary": {
            "metric": "macro_f1", "point_delta": float(primary["delta"]),
            "ci_low": float(primary["ci_low"]), "ci_high": float(primary["ci_high"]),
            "benefit_bound": BENEFIT, "harm_bound": -BENEFIT,
            "bound_basis": "Frozen P3F practical bounds.",
            "multiplicity": f"Bonferroni simultaneous across {run.FAMILY_SIZE} frozen DUFS contrasts.",
        },
        "worst_case_behavior": f"Worst-cell net delta {float(primary['worst_unit_delta']):+.6f}; aligned-context minus permuted-control delta {float(mechanism['delta']):+.6f}.",
        "claim_boundary": "Opened development evidence; context may affect only the graph, not output coordinates. No PRMBench transfer or fresh confirmation follows automatically.",
    })
    atomic_write_json(claims_path, claims)

    plots_path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(plots_path.read_text())
    upsert(plots["plots"], "plot_id", {
        "plot_id": "PLOT_P3F_CONTEXT_DUFS", "title": "Dynamics family-local versus all-H2 context DUFS",
        "phase": "P3", "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv",
        "selection": {"experiment_id": EXPERIMENT, "metric_id": "macro_f1", "status": "COMPLETE"},
        "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id",
        "comparison_group": "same Qwen-eight rows; five-fold donor cross-fit; H0, outer mean and top-ten fixed",
        "bootstrap_definition": "20,000 paired whole-question draws; Bonferroni simultaneous across four frozen contrasts.",
        "selection_rule": "Every registered primary contrast plus the permuted-control parent contrast.",
        "legend": ["F1 uses dynamics-only geometry", "F2 uses all-H2 geometry but dynamics-only weights", "F3 circularly shifts outside-family donor context"],
        "caption": f"Net contextual delta {float(primary['delta']):+.5f}; aligned-context minus permuted control {float(mechanism['delta']):+.5f}.",
    })
    atomic_write_json(plots_path, plots)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({
        "status": "INTEGRATED", "context_supported": summary["context_mechanism_supported"],
        "topk_eligible": summary["topk_secondary_control_eligible"],
        "report_sha256": build.manifest["output"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
