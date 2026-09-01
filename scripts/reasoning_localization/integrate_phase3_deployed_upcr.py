#!/usr/bin/env python3
"""Integrate the completed P3D compact-view prune/refit ladder."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization.run_phase3_deployed_upcr_prune_refit import (  # noqa: E402
    CLEAN_FLOOR,
    EXACT_FLOOR,
    EXPERIMENT_ID,
    H0_REFERENCE,
    H2_PARENT,
    MASK_MEAN_JACCARD_FLOOR,
    MULTIPLICITY_FAMILY_SIZE,
    OUTPUT,
    P3D0,
    P3D1,
    P3D2,
    P3D3,
    PRACTICAL_BENEFIT,
    VARIANT_IDS,
    WORST_CELL_FLOOR,
)

EVAL = OUTPUT / "evaluation"
H0_REPORT_ID = "P2C_F6_TOP10_REFERENCE"
DISPLAY_ORDER = {
    H0_REFERENCE: 169,
    H2_PARENT: 170,
    P3D0: 173,
    P3D1: 174,
    P3D2: 175,
    P3D3: 176,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    selected = list(fields or rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in selected} for row in rows])


def _append_unique(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    rows = _read_csv(path)
    fields = list(rows[0].keys())
    keys = {tuple(row.get(field, "") for field in unique) for row in rows}
    for addition in additions:
        key = tuple(str(addition.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"duplicate reporting row: {key}")
        keys.add(key)
    _write_csv(path, [*rows, *additions], fields)


def _alias(variant_id: str) -> str:
    return H0_REPORT_ID if variant_id == H0_REFERENCE else variant_id


def _upsert(rows: list[dict[str, Any]], key: str, row: dict[str, Any]) -> None:
    matches = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if matches:
        rows[matches[0]] = row
    else:
        rows.append(row)


def main() -> None:
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"
    panels = _read_csv(panels_path)
    metric_rows = []
    for row in panels:
        metric = "macro_f1" if row["metric_id"] == "official_macro_f1" else row["metric_id"]
        metric_rows.append({
            "phase_id": "P3",
            "experiment_id": EXPERIMENT_ID,
            "variant_id": _alias(row["arm_id"]),
            "task_id": "processbench_first_error",
            "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen",
            "cell_id": "aggregate",
            "slice_id": "all",
            "metric_id": metric,
            "value": row["value"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "n_rows": row["n_rows"],
            "n_groups": row["n_groups"],
            "comparison_group_id": f"p3d_compact_view::{metric}",
            "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "display_order": DISPLAY_ORDER[row["arm_id"]],
            "axis_value": "",
            "source_artifact": str(panels_path.relative_to(REPO)),
            "source_sha256": sha256_file(panels_path),
            "source_row_selector": f"arm_id={row['arm_id']};metric_id={row['metric_id']}",
            "source_value_field": "value",
            "notes": "Five-fold donor cross-fit; H0 abstention and top-ten reducer frozen. P3D3 is the predeclared mean over 20 random-mask arms.",
        })
    _append_unique(
        p1.PROGRAM_ROOT / "METRICS_LONG.csv",
        metric_rows,
        ("experiment_id", "variant_id", "metric_id", "cell_id"),
    )

    contrasts_path = EVAL / "PAIRWISE_CONTRASTS.csv"
    contrasts = _read_csv(contrasts_path)
    contrast_rows = []
    for row in contrasts:
        contrast_rows.append({
            "phase_id": "P3",
            "experiment_id": EXPERIMENT_ID,
            "left_variant_id": _alias(row["left_variant_id"]),
            "right_variant_id": _alias(row["right_variant_id"]),
            "task_id": "processbench_first_error",
            "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen",
            "metric_id": row["metric_id"],
            "delta": row["delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "p_adjusted": "",
            "wins": row["wins"],
            "ties": row["ties"],
            "losses": row["losses"],
            "worst_unit_delta": row["worst_unit_delta"],
            "comparison_group_id": f"p3d_compact_view::{row['metric_id']}",
            "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "source_artifact": str(contrasts_path.relative_to(REPO)),
            "source_sha256": sha256_file(contrasts_path),
            "source_row_selector": f"left_variant_id={row['left_variant_id']};right_variant_id={row['right_variant_id']};metric_id={row['metric_id']}",
            "notes": f"{row['statistical_status']}; multiplicity family={row['multiplicity_family_size']}; worst={row['worst_unit_id']}.",
        })
    _append_unique(
        p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv",
        contrast_rows,
        ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"),
    )

    summary = json.loads((EVAL / "SUMMARY.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (OUTPUT / "score_freeze/SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8")
    )
    primary = summary["primary_contrast"]
    vs_h2 = summary["candidate_vs_h2"]
    macro = {
        (row["left_variant_id"], row["right_variant_id"]): row
        for row in contrasts if row["metric_id"] == "macro_f1"
    }
    exact = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D0
        and row["metric_id"] == "first_error_exact"
    )
    clean = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D0
        and row["metric_id"] == "clean_abstention_accuracy"
    )
    min_jaccard = min(summary["mask_mean_jaccard_by_cell"].values())
    gate_rows = [
        {"gate_id": "P3D_NO_PRUNE_ALIAS", "variant_id": P3D0, "metric_id": "max_abs_error", "observed": manifest["no_pruning_alias_max_abs_error"], "threshold": 1e-12, "direction": "le", "passed": manifest["no_pruning_alias_max_abs_error"] <= 1e-12},
        {"gate_id": "P3D_H0_ABSTENTION_ALIAS", "variant_id": P3D1, "metric_id": "mismatches", "observed": sum(summary["abstention_mismatches"].values()), "threshold": 0, "direction": "eq", "passed": sum(summary["abstention_mismatches"].values()) == 0},
        {"gate_id": "P3D_PRIMARY_PRACTICAL", "variant_id": P3D1, "metric_id": "macro_f1_ci_low", "observed": primary["ci_low"], "threshold": PRACTICAL_BENEFIT, "direction": "gt", "passed": primary["ci_low"] > PRACTICAL_BENEFIT},
        {"gate_id": "P3D_H2_SYSTEM_PRACTICAL", "variant_id": P3D1, "metric_id": "macro_f1_ci_low", "observed": vs_h2["ci_low"], "threshold": PRACTICAL_BENEFIT, "direction": "gt", "passed": vs_h2["ci_low"] > PRACTICAL_BENEFIT},
        {"gate_id": "P3D_EXACT_FLOOR", "variant_id": P3D1, "metric_id": "first_error_exact", "observed": exact["delta"], "threshold": EXACT_FLOOR, "direction": "ge", "passed": float(exact["delta"]) >= EXACT_FLOOR},
        {"gate_id": "P3D_CLEAN_FLOOR", "variant_id": P3D1, "metric_id": "clean_abstention_accuracy", "observed": clean["delta"], "threshold": CLEAN_FLOOR, "direction": "ge", "passed": float(clean["delta"]) >= CLEAN_FLOOR},
        {"gate_id": "P3D_WORST_CELL_FLOOR", "variant_id": P3D1, "metric_id": "worst_cell_delta", "observed": primary["worst_unit_delta"], "threshold": WORST_CELL_FLOOR, "direction": "ge", "passed": primary["worst_unit_delta"] >= WORST_CELL_FLOOR},
        {"gate_id": "P3D_MASK_STABILITY", "variant_id": P3D1, "metric_id": "minimum_cell_mean_pairwise_jaccard", "observed": min_jaccard, "threshold": MASK_MEAN_JACCARD_FLOOR, "direction": "ge", "passed": min_jaccard >= MASK_MEAN_JACCARD_FLOOR},
        {"gate_id": "P3D_REFIT_BEATS_EQUAL_MASK", "variant_id": P3D1, "metric_id": "macro_f1_ci_low", "observed": macro[(P3D1, P3D2)]["ci_low"], "threshold": 0.0, "direction": "gt", "passed": float(macro[(P3D1, P3D2)]["ci_low"]) > 0.0},
        {"gate_id": "P3D_RHO_MASK_BEATS_RANDOM", "variant_id": P3D2, "metric_id": "macro_f1_ci_low", "observed": macro[(P3D2, P3D3)]["ci_low"], "threshold": 0.0, "direction": "gt", "passed": float(macro[(P3D2, P3D3)]["ci_low"]) > 0.0},
    ]
    gate_source = EVAL / "REPORTING_GATES.csv"
    for row in gate_rows:
        row.update({
            "unit": "fraction",
            "status": "PASS" if row["passed"] else "FAIL",
            "evidence_status": "DEVELOPMENT",
        })
        row["passed"] = str(row["passed"]).lower()
    _write_csv(gate_source, gate_rows)
    reporting_gates = []
    for row in gate_rows:
        reporting_gates.append({
            "phase_id": "P3",
            "experiment_id": EXPERIMENT_ID,
            "variant_id": row["variant_id"],
            "gate_id": row["gate_id"],
            "metric_id": row["metric_id"],
            "observed": row["observed"],
            "threshold": row["threshold"],
            "direction": row["direction"],
            "passed": row["passed"],
            "unit": row["unit"],
            "status": row["status"],
            "evidence_status": row["evidence_status"],
            "source_artifact": str(gate_source.relative_to(REPO)),
            "source_sha256": sha256_file(gate_source),
            "source_row_selector": f"gate_id={row['gate_id']}",
            "source_value_field": "observed",
            "notes": "A failed promotion/mechanism gate is not generic rejection; see paired CI status.",
        })
    _append_unique(
        p1.PROGRAM_ROOT / "GATES_LONG.csv",
        reporting_gates,
        ("experiment_id", "variant_id", "gate_id"),
    )

    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    updates = {
        P3D0: {
            "execution_status": "COMPLETE",
            "decision_status": "NO_PROMOTION",
            "statistical_status": "INCONCLUSIVE",
            "limitations": "F1 0.354240; delta -0.009850 versus H2, simultaneous CI [-0.021086,+0.000962]. Same-matrix spectral parent only.",
        },
        P3D1: {
            "execution_status": "COMPLETE",
            "decision_status": "NO_PROMOTION",
            "statistical_status": "PROMISING_UNCONFIRMED",
            "limitations": "F1 0.356740; +0.002499 versus P3D0, simultaneous CI [-0.008683,+0.013781], but -0.007350 versus H2. Stable masks did not beat equal/random controls inferentially.",
        },
        P3D2: {
            "execution_status": "COMPLETE",
            "decision_status": "NO_PROMOTION",
            "statistical_status": "INCONCLUSIVE",
            "limitations": "F1 0.353551; the rho-derived survivor mask with equal weights is -0.000457 versus the 20-mask random reference, CI [-0.009545,+0.008547].",
        },
        P3D3: {
            "execution_status": "COMPLETE",
            "decision_status": "NO_PROMOTION",
            "statistical_status": "INCONCLUSIVE",
            "limitations": "Distributional control: mean metric and paired draw across 20 preregistered cardinality-matched masks; F1 0.354007.",
        },
    }
    for variant_id, update in updates.items():
        next(row for row in variants["variants"] if row["variant_id"] == variant_id).update(update)
    atomic_write_json(variants_path, variants)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = next(
        row for row in experiments["experiments"] if row["experiment_id"] == EXPERIMENT_ID
    )
    experiment.update({
        "execution_status": "COMPLETE",
        "next_variant": None,
        "verdict": "PROMISING_UNCONFIRMED_VS_FULLPOOL__NO_PROMOTION__MECHANISM_NOT_SUPPORTED",
        "result_summary": "P3D1 +0.002499 vs P3D0, CI [-0.008683,+0.013781]; -0.007350 vs H2. Masks stable, but refit/equal-mask/random-mask mechanism gates fail.",
    })
    atomic_write_json(experiments_path, experiments)

    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claim = {
        "claim_id": "CLAIM_P3D_DEPLOYED_UPCR",
        "text": "Deployed U-PCR weak-view exclusion is numerically better than matched full-pool IU on the same 24 H2 member views, but the increment is unconfirmed and the deployed arm remains below equal-family H2; the prune/refit mechanism is not supported by its matched controls.",
        "verdict": "PROMISING_UNCONFIRMED",
        "task_scope": "Current common eight-Qwen ProcessBench first-error development population; five-fold donor cross-fit.",
        "evidence_refs": [
            "PLOT_P3D_DEPLOYED_FOREST",
            f"CONTRAST:{P3D1}:{P3D0}",
            "TABLE_GATES",
        ],
        "fresh_confirmation_required": True,
        "statistical_summary": {
            "metric": "macro_f1",
            "point_delta": primary["delta"],
            "ci_low": primary["ci_low"],
            "ci_high": primary["ci_high"],
            "benefit_bound": PRACTICAL_BENEFIT,
            "harm_bound": -PRACTICAL_BENEFIT,
            "bound_basis": "Frozen Phase-3 practical bounds.",
            "multiplicity": f"Bonferroni simultaneous interval across {MULTIPLICITY_FAMILY_SIZE} registered macro-F1 contrasts.",
        },
        "worst_case_behavior": "Versus H2 the point delta is -0.007350 with 1/2/5 cell W/T/L and worst cell -0.019673. Equal-mask does not beat random masks.",
        "claim_boundary": "A CI crossing zero is not rejection or zero effect. This does not authorize PRMBench transfer or reopen failed full-pool methods.",
    }
    _upsert(claims["claims"], "claim_id", claim)
    atomic_write_json(claims_path, claims)

    plot_path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plot_manifest = json.loads(plot_path.read_text(encoding="utf-8"))
    forest = {
        "plot_id": "PLOT_P3D_DEPLOYED_FOREST",
        "title": "Deployed U-PCR prune/refit and matched controls",
        "phase": "P3",
        "kind": "contrast_forest",
        "source_table": "CONTRASTS_LONG.csv",
        "selection": {
            "experiment_id": EXPERIMENT_ID,
            "metric_id": "macro_f1",
            "status": "COMPLETE",
        },
        "x_field": "delta",
        "y_field": "left_variant_id",
        "series_field": "right_variant_id",
        "comparison_group": "same Qwen-eight rows; H0 abstention and top-ten reducer frozen; five-fold donor cross-fit",
        "bootstrap_definition": "20,000 paired whole-question draws; Bonferroni simultaneous interval across six registered macro-F1 contrasts.",
        "selection_rule": "All P3D full-pool, deployed, equal-mask, random-mask and H2 contrasts frozen before labels.",
        "legend": [
            "P3D3 is the predeclared mean across 20 cardinality-matched random-mask arms",
            "CI crossing zero is unresolved, not rejection",
        ],
        "caption": "P3D1 gains +0.00250 over same-matrix P3D0 but remains -0.00735 below H2. Stable masks alone do not establish useful pruning.",
    }
    gate_plot = {
        "plot_id": "PLOT_P3D_GATE_MATRIX",
        "title": "P3D promotion and mechanism gates",
        "phase": "P3",
        "kind": "gate_matrix",
        "source_table": "GATES_LONG.csv",
        "selection": {"experiment_id": EXPERIMENT_ID},
        "x_field": "gate_id",
        "y_field": "variant_id",
        "comparison_group": "P3D frozen execution and promotion contract",
        "bootstrap_definition": "Inferential gates use the same 20,000 paired whole-question draws; alias and mask-stability gates are deterministic contract checks.",
        "selection_rule": "Every registered P3D gate.",
        "legend": ["PASS = gate passed", "FAIL = no promotion through that gate; not generic rejection"],
        "caption": "Aliases, robustness floors and mask stability pass; practical improvement and matched mechanism controls do not.",
    }
    _upsert(plot_manifest["plots"], "plot_id", forest)
    _upsert(plot_manifest["plots"], "plot_id", gate_plot)
    atomic_write_json(plot_path, plot_manifest)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({
        "status": "INTEGRATED",
        "primary_delta": primary["delta"],
        "primary_ci": [primary["ci_low"], primary["ci_high"]],
        "promotion_passed": summary["promotion_passed"],
        "report_sha256": build.manifest["output"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
