#!/usr/bin/env python3
"""Integrate the completed ASTGI-Q1 ProcessBench rung into the living report."""
from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization import run_phase3_astgi_q1 as run  # noqa: E402


EVAL = run.OUTPUT / "evaluation"
EXPERIMENT = run.EXPERIMENT
VARIANT = run.VARIANT
PARENT = run.PARENT
PERMUTED = run.PERMUTED
NO_BOUNDARY = run.NO_BOUNDARY
ORDERS = {VARIANT: 193, PERMUTED: 194, NO_BOUNDARY: 195, PARENT: 170}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def append(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    existing = read(path)
    fields = list(existing[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in existing}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"duplicate reporting key: {key}")
        keys.add(key)
    write(path, [*existing, *additions])


def remove_experiment(path: Path, experiment_id: str) -> None:
    """Make a failed pre-label integration retryable without touching other experiments."""
    rows = read(path)
    if not rows or "experiment_id" not in rows[0]:
        raise RuntimeError(f"unexpected reporting table: {path}")
    kept = [row for row in rows if row.get("experiment_id") != experiment_id]
    if len(kept) != len(rows):
        write(path, kept)


def upsert(rows: list[dict[str, Any]], key: str, row: dict[str, Any]) -> None:
    hits = [i for i, current in enumerate(rows) if current.get(key) == row[key]]
    if len(hits) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if hits:
        rows[hits[0]] = row
    else:
        rows.append(row)


def main() -> None:
    # The first integration attempt may have stopped after appending metrics but
    # before registries were updated. Remove only this experiment's rows so a
    # retry is deterministic and cannot duplicate evidence.
    for filename in ("METRICS_LONG.csv", "CONTRASTS_LONG.csv", "GATES_LONG.csv"):
        remove_experiment(p1.PROGRAM_ROOT / filename, EXPERIMENT)
    panels_path = EVAL / "PROCESSBENCH_PANELS.csv"
    panels = read(panels_path)
    metric_rows = []
    for source in panels:
        arm = source["arm_id"]
        if arm == "P3_H0_REFERENCE":
            continue
        metric_rows.append({
            "phase_id": "P3",
            "experiment_id": EXPERIMENT,
            "variant_id": arm,
            "task_id": "processbench_first_error",
            "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen",
            "cell_id": "aggregate",
            "slice_id": "all",
            "metric_id": "macro_f1" if source["metric_id"] == "official_macro_f1" else source["metric_id"],
            "value": source["value"],
            "ci_low": source["ci_low"],
            "ci_high": source["ci_high"],
            "n_rows": source["n_rows"],
            "n_groups": source["n_groups"],
            "comparison_group_id": f"p3tq1::{source['metric_id']}",
            "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "display_order": ORDERS.get(arm, 193),
            "axis_value": "",
            "source_artifact": str(panels_path.relative_to(REPO)),
            "source_sha256": sha256_file(panels_path),
            "source_row_selector": f"arm_id={arm};metric_id={source['metric_id']}",
            "source_value_field": "value",
            "notes": "Fixed analytic query; H0 detector and top-ten reducer unchanged; controls are non-rankable diagnostics.",
        })
    append(p1.PROGRAM_ROOT / "METRICS_LONG.csv", metric_rows, ("experiment_id", "variant_id", "metric_id", "cell_id"))

    contrasts_path = EVAL / "PAIRWISE_CONTRASTS.csv"
    raw_contrasts = read(contrasts_path)
    contrast_rows = []
    for source in raw_contrasts:
        contrast_rows.append({
            "phase_id": "P3",
            "experiment_id": EXPERIMENT,
            "left_variant_id": source["left_variant_id"],
            "right_variant_id": source["right_variant_id"],
            "task_id": "processbench_first_error",
            "dataset_id": "processbench",
            "population_id": "current_common_eight_qwen",
            "metric_id": source["metric_id"],
            "delta": source["delta"],
            "ci_low": source["ci_low"],
            "ci_high": source["ci_high"],
            "p_adjusted": "",
            "wins": source["wins"],
            "ties": source["ties"],
            "losses": source["losses"],
            "worst_unit_delta": source["worst_unit_delta"],
            "comparison_group_id": f"p3tq1::{source['metric_id']}",
            "status": "COMPLETE",
            "evidence_status": "DEVELOPMENT",
            "source_artifact": str(contrasts_path.relative_to(REPO)),
            "source_sha256": sha256_file(contrasts_path),
            "source_row_selector": f"contrast_id={source['contrast_id']}",
            "notes": f"{source['statistical_status']}; worst={source['worst_unit_id']}; Bonferroni family={source['multiplicity_family_size']}.",
        })
    append(p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv", contrast_rows, ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"))

    summary = json.loads((EVAL / "SUMMARY.json").read_text(encoding="utf-8"))
    score_freeze = json.loads((run.OUTPUT / "score_freeze/SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    primary = summary["primary_contrast"]
    controls = {(row["left_variant_id"], row["right_variant_id"], row["metric_id"]): row for row in raw_contrasts}
    gate_rows = []

    def gate(gate_id: str, variant: str, metric: str, observed: Any, threshold: Any, direction: str, note: str) -> None:
        value = float(observed)
        target = float(threshold)
        passed = {"ge": value >= target, "gt": value > target, "le": value <= target, "eq": value == target}[direction]
        gate_rows.append({
            "gate_id": gate_id,
            "variant_id": variant,
            "metric_id": metric,
            "observed": observed,
            "threshold": threshold,
            "direction": direction,
            "passed": str(passed).lower(),
            "unit": "fraction",
            "status": "PASS" if passed else "FAIL",
            "evidence_status": "DEVELOPMENT",
            "notes": note,
        })

    gate("P3TQ1_PARENT_ALIAS", VARIANT, "max_parent_alias_error", score_freeze["parent_alias_max_abs_error"], 1e-12, "le", "Exact H2 parent alias before labels.")
    gate("P3TQ1_ABSTENTION_ALIAS", VARIANT, "abstention_mismatches", max(summary["abstention_mismatches"].values()), 0, "eq", "Every Q1 arm copies H0 clean/error authority.")
    gate("P3TQ1_POINT_VS_PARENT", VARIANT, "macro_f1", primary["delta"], run.BENEFIT, "ge", "Primary point estimate must clear practical benefit.")
    gate("P3TQ1_CI_VS_PARENT", VARIANT, "macro_f1_ci_low", primary["ci_low"], run.BENEFIT, "gt", "Primary Bonferroni lower bound must clear practical benefit.")
    gate("P3TQ1_EXACT_VS_PARENT", VARIANT, "first_error_exact", controls[(VARIANT, PARENT, "first_error_exact")]["delta"], -0.010, "ge", "No material exact-error regression.")
    gate("P3TQ1_WORST_CELL", VARIANT, "worst_cell_delta", primary["worst_unit_delta"], -0.020, "ge", "No material worst-cell regression.")
    gate("P3TQ1_BEATS_QUERY_PERMUTATION", VARIANT, "macro_f1_ci_low", controls[(VARIANT, PERMUTED, "macro_f1")]["ci_low"], 0.0, "gt", "Family-role query mechanism requires separation from permutation.")
    gate("P3TQ1_BEATS_NO_BOUNDARY", VARIANT, "macro_f1_ci_low", controls[(VARIANT, NO_BOUNDARY, "macro_f1")]["ci_low"], 0.0, "gt", "Boundary contribution requires separation from no-boundary control.")

    gate_source = EVAL / "REPORTING_GATES.csv"
    write(gate_source, gate_rows)
    append(p1.PROGRAM_ROOT / "GATES_LONG.csv", [{
        "phase_id": "P3",
        "experiment_id": EXPERIMENT,
        **row,
        "source_artifact": str(gate_source.relative_to(REPO)),
        "source_sha256": sha256_file(gate_source),
        "source_row_selector": f"gate_id={row['gate_id']}",
        "source_value_field": "observed",
    } for row in gate_rows], ("experiment_id", "variant_id", "gate_id"))

    variants_path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants_doc = json.loads(variants_path.read_text(encoding="utf-8"))
    variants = variants_doc["variants"]
    candidate = next(row for row in variants if row["variant_id"] == VARIANT)
    candidate.update({
        "execution_status": "COMPLETE",
        "decision_status": "NO_PROMOTION",
        "statistical_status": primary["statistical_status"],
        "rankable": False,
        "limitations": f"F1={float(next(r['value'] for r in panels if r['arm_id']==VARIANT and r['metric_id']=='official_macro_f1')):.6f}; vs H2 delta {float(primary['delta']):+.6f} [{float(primary['ci_low']):+.6f},{float(primary['ci_high']):+.6f}]; Q1 premise not established.",
        "execution_artifact": str((run.OUTPUT / "RUN_COMPLETE.json").relative_to(REPO)),
    })
    template = deepcopy(candidate)
    for control_id, display_name, role, parent_ids, transform in (
        (PERMUTED, "ASTGI-Q1 query-permuted control", "negative_control", [VARIANT], "reverse frozen q_onset family prior"),
        (NO_BOUNDARY, "ASTGI-Q1 no-boundary control", "negative_control", [VARIANT], "set boundary gamma to zero"),
    ):
        if not any(row["variant_id"] == control_id for row in variants):
            control = deepcopy(template)
            control.update({
                "variant_id": control_id,
                "display_name": display_name,
                "display_order": ORDERS[control_id],
                "parent_variant_ids": parent_ids,
                "role": role,
                "rankable": False,
                "decision_status": "NO_PROMOTION",
                "statistical_status": "INCONCLUSIVE",
                "fusion": transform,
                "novelty": "Predeclared mechanism control; not a candidate method.",
                "limitations": "Diagnostic control only; never enters a leaderboard.",
            })
            variants.append(control)
    atomic_write_json(variants_path, variants_doc)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments_doc = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = next(row for row in experiments_doc["experiments"] if row["experiment_id"] == EXPERIMENT)
    experiment.update({
        "execution_status": "COMPLETE",
        "next_variant": None,
        "control_variant_ids": [PARENT, PERMUTED, NO_BOUNDARY],
        "verdict": "Q1_INCONCLUSIVE_NO_PROMOTION__Q2_NOT_OPENED",
        "result_summary": f"Q1 point query is {float(primary['delta']):+.6f} vs H2 [{float(primary['ci_low']):+.6f},{float(primary['ci_high']):+.6f}]; candidate has no supported ProcessBench gain and controls do not establish a query mechanism.",
        "result_artifact": str((EVAL / "SUMMARY.json").relative_to(REPO)),
    })
    atomic_write_json(experiments_path, experiments_doc)

    claims_path = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims_doc = json.loads(claims_path.read_text(encoding="utf-8"))
    upsert(claims_doc["claims"], "claim_id", {
        "claim_id": "CLAIM_P3T_Q1_POINT_QUERY",
        "text": "On the opened eight-Qwen ProcessBench development population, the fixed ASTGI-Q1 point-query pooling rule did not establish an improvement over the exact H2 parent; its interval is inconclusive and the query-permutation/boundary controls do not support a mechanism claim.",
        "verdict": "INCONCLUSIVE",
        "task_scope": "Current common eight-Qwen ProcessBench first-error development population; H0 detector and top-ten reducer fixed.",
        "evidence_refs": ["PLOT_P3T_Q1_POINT_QUERY", f"CONTRAST:{VARIANT}:{PARENT}", f"CONTRAST:{VARIANT}:{PERMUTED}", f"CONTRAST:{VARIANT}:{NO_BOUNDARY}", "TABLE_GATES"],
        "fresh_confirmation_required": True,
        "statistical_summary": {
            "metric": "macro_f1",
            "point_delta": float(primary["delta"]),
            "ci_low": float(primary["ci_low"]),
            "ci_high": float(primary["ci_high"]),
            "benefit_bound": run.BENEFIT,
            "harm_bound": run.HARM,
            "bound_basis": "Frozen Q1 practical bounds; Bonferroni simultaneous family size three.",
            "multiplicity": "Three paired whole-question ProcessBench contrasts.",
        },
        "worst_case_behavior": f"Worst Q1-vs-H2 cell {float(primary['worst_unit_delta']):+.6f} ({primary['worst_unit_id']}); exact delta {float(controls[(VARIANT,PARENT,'first_error_exact')]['delta']):+.6f}.",
        "claim_boundary": "Development-only opened labels; no PRMBench transfer and no Q2 opening. A CI crossing zero is reported as inconclusive, not rejection.",
    })
    atomic_write_json(claims_path, claims_doc)

    plots_path = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots_doc = json.loads(plots_path.read_text(encoding="utf-8"))
    upsert(plots_doc["plots"], "plot_id", {
        "plot_id": "PLOT_P3T_Q1_POINT_QUERY",
        "title": "ASTGI-Q1 fixed point-query pooling",
        "phase": "P3",
        "kind": "contrast_forest",
        "source_table": "CONTRASTS_LONG.csv",
        "selection": {"experiment_id": EXPERIMENT, "metric_id": "macro_f1", "status": "COMPLETE"},
        "x_field": "delta",
        "y_field": "left_variant_id",
        "series_field": "right_variant_id",
        "comparison_group": "same eight-Qwen ProcessBench rows; H2 family representation and top-ten reducer fixed",
        "bootstrap_definition": "20,000 paired whole-question draws; Bonferroni simultaneous intervals across parent, query-permutation and no-boundary contrasts",
        "selection_rule": "Primary Q1-vs-H2 plus two preregistered controls; raw score and practical bounds remain visible",
        "legend": ["Crossing zero is inconclusive", "Controls are diagnostics and non-rankable", "No PRMBench transfer opened"],
        "caption": "The fixed query pool is below H2 in point estimate; its interval reaches parity but does not establish a mechanism against controls.",
        "source_artifact": str((EVAL / "P3T_Q1_RESULTS.svg").relative_to(REPO)),
        "source_sha256": sha256_file(EVAL / "P3T_Q1_RESULTS.svg"),
    })
    atomic_write_json(plots_path, plots_doc)

    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"status": "INTEGRATED", "report_sha256": build.manifest["output"]["sha256"], "experiment": EXPERIMENT}, indent=2))


if __name__ == "__main__":
    main()
