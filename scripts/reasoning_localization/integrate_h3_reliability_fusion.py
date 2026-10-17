#!/usr/bin/env python3
"""Import the isolated H2/H3 reliability experiment into the living program."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402


SOURCE = Path(
    "/Users/osegev/Documents/Codex/2026-08-29/"
    "referenced-chatgpt-conversation-this-is-an/outputs/"
    "h3_reliability_fusion_v1"
)
DEST = p1.PROGRAM_ROOT / "phase_2" / "diagnostic" / "h3_reliability_fusion_v1"
EXPERIMENT = "P2_H3_RELIABILITY_FUSION"
PARENT = "P2C_F6_TOP10_REFERENCE"
ARM_MAP = {
    "H0_FAMILY6_TOP10": PARENT,
    "H2_CLEAN_C7": "P2D_H2_CLEAN_C7",
    "H3_EQUAL": "P2D_H3_EQUAL_C8_RERANK",
    "H3_RELIABILITY": "P2D_H3_RELIABILITY_C8_RERANK",
}
DISPLAY = {
    "P2D_H2_CLEAN_C7": 144,
    "P2D_H3_EQUAL_C8_RERANK": 145,
    "P2D_H3_RELIABILITY_C8_RERANK": 146,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def append_rows(path: Path, additions: list[dict[str, object]], unique: tuple[str, ...]) -> None:
    existing = read_csv(path)
    fields = list(existing[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in existing}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"already integrated into {path.name}: {key}")
        keys.add(key)
    write_csv(path, [*existing, *additions], fields)


def verify_and_copy() -> None:
    manifest = json.loads((SOURCE / "REPORT_MANIFEST.json").read_text())
    for row in manifest["artifacts"]:
        path = SOURCE / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source hash mismatch: {path}")
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, DEST)


def register_variants_and_experiment() -> None:
    vp = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(vp.read_text())
    if any(row["variant_id"] in set(ARM_MAP.values()) - {PARENT} for row in variants["variants"]):
        raise RuntimeError("H2/H3 variants already registered")
    common = {
        "phase": "P2D",
        "method_id": "fusion_selection",
        "role": "role_separated_reranker",
        "step_reducer": "top-ten frozen before H0 decision; candidate reranks H0 non-abstentions only",
        "detector": "exact H0 grouped cross-fitted threshold and clean/error decision",
        "supervision": "target-free score/weight fit; outcome-selected opened-population development experiment",
        "access_tier": "gray_box_single_pass",
        "causal_validity": "completed-trace localization; C8 residual is causal but full H0 pipeline is not early-certified",
        "evidence_status": "DEVELOPMENT",
        "execution_status": "COMPLETE",
        "decision_status": "NO_PROMOTION",
        "rankable": True,
        "task_ids": ["processbench_first_error"],
        "limitations": "Four choices were motivated after Phase-2C outcomes opened; fresh questions are required for promotion.",
    }
    variants["variants"].extend(
        [
            {
                **common,
                "variant_id": "P2D_H2_CLEAN_C7",
                "display_name": "H2: favorable family edits behind H0 detector",
                "display_order": DISPLAY["P2D_H2_CLEAN_C7"],
                "parent_variant_ids": [PARENT],
                "signals": ["parent without sampled-token energy", "partition energy without energy_series", "C7 inside entropy dynamics"],
                "transforms": ["fixed family/view removal", "frozen C7 insertion", "H0 role separation"],
                "fusion": "equal within-family and equal outer-family parent rule",
                "novelty": "Combines three individually positive point-estimate edits while preserving H0 abstention.",
                "failure_hypothesis": "Outcome-selected edits do not combine beyond uncertainty.",
                "prior_evidence": "Three individual Phase-2C point estimates were positive but unconfirmed.",
                "statistical_status": "PROMISING_UNCONFIRMED",
            },
            {
                **common,
                "variant_id": "P2D_H3_EQUAL_C8_RERANK",
                "display_name": "H3 equal: H2 plus C8 localization-only reranking",
                "display_order": DISPLAY["P2D_H3_EQUAL_C8_RERANK"],
                "parent_variant_ids": ["P2D_H2_CLEAN_C7", "P2C_F6_PLUS_C8_OUTER_EXPERT"],
                "signals": ["H2 step ranks", "frozen C8 step ranks"],
                "transforms": ["H0 role separation", "equal 50/50 step-rank fusion"],
                "fusion": "0.5 H2 donor rank plus 0.5 C8 donor rank on H0 non-abstentions",
                "novelty": "Uses innovation only for location, never for the clean/error decision.",
                "failure_hypothesis": "C8 adds no step-ranking value after H2 cleanup.",
                "prior_evidence": "C8 improved exact localization but harmed clean abstention when allowed into the full decision.",
                "statistical_status": "INCONCLUSIVE",
            },
            {
                **common,
                "variant_id": "P2D_H3_RELIABILITY_C8_RERANK",
                "display_name": "H3 reliability: donor-stability weighted H2/C8 reranking",
                "display_order": DISPLAY["P2D_H3_RELIABILITY_C8_RERANK"],
                "parent_variant_ids": ["P2D_H3_EQUAL_C8_RERANK"],
                "signals": ["H2 step ranks", "frozen C8 step ranks", "donor perturbation stability"],
                "transforms": ["twelve circular moving-block perturbations", "five label-free folds"],
                "fusion": "alpha_C8=R_C8/(R_H2+R_C8), no grid",
                "novelty": "Attempts label-free reliability weighting without task-metric optimization.",
                "failure_hypothesis": "Stability weights collapse to equal weighting or misestimate localization reliability.",
                "prior_evidence": "B3-style conditional reliability was promising historically but not localized or frozen here.",
                "statistical_status": "PROMISING_UNCONFIRMED",
            },
        ]
    )
    atomic_write_json(vp, variants)

    ep = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(ep.read_text())
    if any(row["experiment_id"] == EXPERIMENT for row in experiments["experiments"]):
        raise RuntimeError("H3 experiment already registered")
    experiments["experiments"].append(
        {
            "experiment_id": EXPERIMENT,
            "display_name": "H2/H3 role-separated reliability reranking",
            "phase": "P2D",
            "execution_status": "COMPLETE",
            "question": "Do favorable family edits and C8 combine when H0 retains exclusive detection authority, and does donor reliability beat equal weighting?",
            "prerequisite": "Phase 2C complete; all four candidate definitions frozen before labels opened in the isolated side run.",
            "population_ids": ["current_common_eight_qwen"],
            "task_ids": ["processbench_first_error"],
            "primary_metrics": ["paired_delta_macro_f1", "first_error_exact", "clean_abstention_accuracy"],
            "registered_comparators": [PARENT, "P2D_H3_EQUAL_C8_RERANK"],
            "promotion_gates": [
                "development-only because choices were outcome-selected on an opened population",
                "H0 detector and abstention decisions alias exactly",
                "Bonferroni simultaneous lower bound exceeds +0.003 practical benefit",
                "fresh questions required before Phase-3 eligibility",
            ],
            "report_sections": ["p2d_h3_absolute", "p2d_h3_forest"],
            "variant_order": ["P2D_H2_CLEAN_C7", "P2D_H3_EQUAL_C8_RERANK", "P2D_H3_RELIABILITY_C8_RERANK"],
            "bootstrap": "20,000 paired whole-source-question draws; Bonferroni simultaneous macro-F1 intervals across four primary contrasts",
            "verdict": "H3_EQUAL_DIRECTIONAL_BUT_BELOW_PRACTICAL_BOUND__NO_PROMOTION",
            "raw_best": "P2D_H3_EQUAL_C8_RERANK",
            "source_report_sha256": sha256_file(DEST / "H3_RELIABILITY_REPORT.html"),
        }
    )
    atomic_write_json(ep, experiments)


def integrate_tables() -> None:
    panel_path = DEST / "PANELS.csv"
    contrast_path = DEST / "CONTRASTS.csv"
    panel_sha = sha256_file(panel_path)
    contrast_sha = sha256_file(contrast_path)
    metrics: list[dict[str, object]] = []
    for row in read_csv(panel_path):
        if row["arm_id"] == "H0_FAMILY6_TOP10":
            continue
        variant = ARM_MAP[row["arm_id"]]
        for metric in p1.PB_METRICS:
            metrics.append(
                {
                    "phase_id": "P2D", "experiment_id": EXPERIMENT, "variant_id": variant,
                    "task_id": "processbench_first_error", "dataset_id": "processbench",
                    "population_id": "current_common_eight_qwen", "cell_id": "aggregate", "slice_id": "all",
                    "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                    "value": row[metric], "n_rows": 6800, "n_groups": 3400,
                    "comparison_group_id": f"p2d_h3_qwen8::{metric}", "status": "COMPLETE",
                    "evidence_status": "DEVELOPMENT", "display_order": DISPLAY[variant],
                    "source_artifact": str(panel_path.relative_to(REPO)), "source_sha256": panel_sha,
                    "source_row_selector": f"arm_id={row['arm_id']}", "source_value_field": metric,
                    "notes": "Outcome-selected opened-population role-separated reranker; no promotion.",
                }
            )
    append_rows(p1.PROGRAM_ROOT / "METRICS_LONG.csv", metrics, ("experiment_id", "variant_id", "metric_id", "cell_id"))

    contrasts: list[dict[str, object]] = []
    for row in read_csv(contrast_path):
        left, right = ARM_MAP[row["left"]], ARM_MAP[row["right"]]
        contrasts.append(
            {
                "phase_id": "P2D", "experiment_id": EXPERIMENT,
                "left_variant_id": left, "right_variant_id": right,
                "task_id": "processbench_first_error", "dataset_id": "processbench",
                "population_id": "current_common_eight_qwen",
                "metric_id": "macro_f1" if row["metric"] == "official_macro_f1" else row["metric"],
                "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
                "worst_unit_delta": row["worst_cell_delta"],
                "comparison_group_id": f"p2d_h3_qwen8::{row['metric']}",
                "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
                "source_artifact": str(contrast_path.relative_to(REPO)), "source_sha256": contrast_sha,
                "source_row_selector": f"contrast_id={row['contrast_id']}", "source_value_field": "delta",
                "notes": row["interval"] + "; outcome-selected opened-population diagnostic.",
            }
        )
    append_rows(p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv", contrasts, ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"))

    evaluation_path = DEST / "EVALUATION.json"
    evaluation = json.loads(evaluation_path.read_text())
    gates = [
        ("P2D_H3_H0_ALIAS", "h0_parent_alias_abs_error", 0.0, 0.0, "eq", True, "H0 parent reconstruction aliases exactly."),
        ("P2D_H3_ABSTENTION_ALIAS", "abstention_mismatches", 0, 0, "eq", True, "All H2/H3 abstentions alias H0 exactly."),
        ("P2D_H3_GROUP_FOLD", "max_group_fold_count", 1, 1, "le", True, "Every source group appears in at most one fit fold."),
        ("P2D_H3_EQUAL_DIRECTIONAL", "macro_f1_ci_low", 0.001768682486614345, 0.0, "gt", True, "Directional simultaneous interval is above zero."),
        ("P2D_H3_EQUAL_PRACTICAL", "macro_f1_ci_low", 0.001768682486614345, 0.003, "gt", False, "Lower bound does not clear the existing practical-benefit boundary."),
        ("P2D_H3_FRESH_CONFIRMATION", "fresh_population", False, True, "eq", False, "Current ProcessBench questions were already opened."),
    ]
    gate_source = DEST / "INTEGRATION_GATES.csv"
    write_csv(
        gate_source,
        [
            {
                "gate_id": gate_id,
                "metric_id": metric,
                "observed": str(observed).lower() if isinstance(observed, bool) else observed,
                "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
                "direction": direction,
                "passed": str(passed).lower(),
                "unit": "boolean" if isinstance(observed, bool) else "fraction",
                "status": "PASS" if passed else "FAIL",
                "evidence_status": "DEVELOPMENT",
            }
            for gate_id, metric, observed, threshold, direction, passed, _note in gates
        ],
        ["gate_id", "metric_id", "observed", "threshold", "direction", "passed", "unit", "status", "evidence_status"],
    )
    gate_source_sha = sha256_file(gate_source)
    gate_rows = []
    for gate_id, metric, observed, threshold, direction, passed, note in gates:
        gate_rows.append(
            {
                "phase_id": "P2D", "experiment_id": EXPERIMENT, "variant_id": "P2D_H3_EQUAL_C8_RERANK",
                "gate_id": gate_id, "metric_id": metric, "observed": str(observed).lower() if isinstance(observed, bool) else observed,
                "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
                "direction": direction, "passed": str(passed).lower(), "unit": "boolean" if isinstance(observed, bool) else "fraction",
                "status": "PASS" if passed else "FAIL", "evidence_status": "DEVELOPMENT",
                "source_artifact": str(gate_source.relative_to(REPO)), "source_sha256": gate_source_sha,
                "source_row_selector": f"gate_id={gate_id}", "source_value_field": "observed",
                "notes": note,
            }
        )
    append_rows(p1.PROGRAM_ROOT / "GATES_LONG.csv", gate_rows, ("experiment_id", "variant_id", "gate_id"))


def register_plot_and_claim() -> None:
    pp = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(pp.read_text())
    plots["plots"].append(
        {
            "plot_id": "PLOT_P2D_H3_RELIABILITY_FOREST",
            "title": "Role-separated H2/H3 reranking versus the frozen H0 detector",
            "phase": "P2D", "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv",
            "selection": {"experiment_id": EXPERIMENT, "metric_id": "macro_f1", "status": "COMPLETE"},
            "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id",
            "comparison_group": "same eight-Qwen rows, H0 abstention and grouped bootstrap stream",
            "bootstrap_definition": "20,000 paired whole-source-question draws; Bonferroni simultaneous macro-F1 intervals across four primary contrasts.",
            "selection_rule": "All four frozen macro-F1 contrasts: H2, H3 equal and H3 reliability versus H0, plus reliability versus equal.",
            "legend": ["Point and line = paired delta and simultaneous interval", "H3 equal is directional but below the +0.003 practical lower-bound gate"],
            "caption": "H3 equal is raw-best and preserves clean abstention exactly. Donor reliability learns nearly equal weights and does not improve the simple 50/50 rule.",
        }
    )
    atomic_write_json(pp, plots)

    cp = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(cp.read_text())
    claims["claims"].append(
        {
            "claim_id": "CLAIM_P2D_H3_EQUAL_ROLE_SEPARATION",
            "text": "On the opened eight-Qwen development population, combining the favorable family edits with C8 as a localization-only equal-rank expert improves macro F1 directionally while preserving H0 clean abstention exactly; the practical-bound and fresh-confirmation gates remain unmet.",
            "verdict": "INCONCLUSIVE",
            "task_scope": "Current common eight-Qwen ProcessBench first-error localization development population.",
            "evidence_refs": ["PLOT_P2D_H3_RELIABILITY_FOREST", "CONTRAST:P2D_H3_EQUAL_C8_RERANK:P2C_F6_TOP10_REFERENCE", "TABLE_GATES"],
            "worst_case_behavior": "Two of eight cells lose; worst-cell macro-F1 delta is -0.002764. Clean abstention is unchanged by construction.",
            "claim_boundary": "Directional CI excludes zero, but its +0.001769 lower bound does not clear the registered +0.003 practical-benefit boundary; outcome selection and opened questions forbid promotion.",
            "fresh_confirmation_required": True,
            "statistical_summary": {
                "metric": "macro_f1", "point_delta": 0.012392219310811137,
                "ci_low": 0.001768682486614345, "ci_high": 0.022807350265052286,
                "benefit_bound": 0.003, "harm_bound": -0.005,
                "bound_basis": "Existing Phase-3 parent practical-benefit semantics applied conservatively to the development diagnostic.",
                "multiplicity": "Bonferroni simultaneous across four frozen macro-F1 contrasts.",
            },
        }
    )
    atomic_write_json(cp, claims)


def main() -> None:
    verify_and_copy()
    register_variants_and_experiment()
    integrate_tables()
    register_plot_and_claim()
    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"experiment": EXPERIMENT, "report_sha256": build.manifest["output"]["sha256"], "source_report_sha256": sha256_file(DEST / "H3_RELIABILITY_REPORT.html")}, indent=2))


if __name__ == "__main__":
    main()
