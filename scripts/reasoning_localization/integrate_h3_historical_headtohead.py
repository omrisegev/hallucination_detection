#!/usr/bin/env python3
"""Integrate the matched historical-regime H3 head-to-head into the report."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reasoning_localization import run_h3_historical_headtohead as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402


EVAL = run.ROOT / "evaluation"
END_PANELS = EVAL / "END_TO_END_PANELS.csv"
END_CELLS = EVAL / "END_TO_END_BY_CELL.csv"
END_CONTRASTS = EVAL / "END_TO_END_CONTRASTS.csv"
DIAG_PANELS = EVAL / "DIAGNOSTIC_PANELS.csv"
DIAG_CELLS = EVAL / "DIAGNOSTIC_BY_CELL.csv"
DIAG_CONTRASTS = EVAL / "DIAGNOSTIC_CONTRASTS.csv"
INTERACTIONS = EVAL / "INTERACTIONS.csv"
SUMMARY = EVAL / "SUMMARY.json"
GATE_SOURCE = EVAL / "GATES.csv"

VARIANTS = {
    run.ENTROPY: "P4H_ENTROPY_TOP5",
    run.FINALIST: "P4H_HIST_FINALIST",
    run.H0: "P4H_H0_FAMILY_TOP10",
    run.H2: "P4H_H2_CLEAN_C7",
    run.H3: "P4H_H3_EQUAL_C8",
}
DIAGNOSTIC_VARIANTS = {
    "HISTDET_HISTLOC": VARIANTS[run.FINALIST],
    "H0DET_H3LOC": VARIANTS[run.H3],
    "HISTDET_H0LOC": "P4H_HISTDET_H0LOC",
    "HISTDET_H2LOC": "P4H_HISTDET_H2LOC",
    "HISTDET_H3LOC": "P4H_HISTDET_H3LOC",
    "H0DET_HISTLOC": "P4H_H0DET_HISTLOC",
}
DISPLAY = {
    "P4H_ENTROPY_TOP5": 160,
    "P4H_HIST_FINALIST": 161,
    "P4H_H0_FAMILY_TOP10": 162,
    "P4H_H2_CLEAN_C7": 163,
    "P4H_H3_EQUAL_C8": 164,
    "P4H_HISTDET_H0LOC": 165,
    "P4H_HISTDET_H2LOC": 166,
    "P4H_HISTDET_H3LOC": 167,
    "P4H_H0DET_HISTLOC": 168,
}
METRIC_ID = {
    "f1": "macro_f1",
    "exact_error": "exact_error",
    "clean_abstention": "clean_abstention",
    "within_one": "within_one",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def append_rows(path: Path, additions: list[dict[str, Any]], unique: tuple[str, ...]) -> None:
    existing = read_csv(path)
    fields = list(existing[0])
    keys = {tuple(row.get(field, "") for field in unique) for row in existing}
    for row in additions:
        key = tuple(str(row.get(field, "")) for field in unique)
        if key in keys:
            raise RuntimeError(f"duplicate integration key in {path.name}: {key}")
        keys.add(key)
    write_csv(path, [*existing, *additions], fields)


def source_contrast(left: str, right: str, metric: str) -> dict[str, str]:
    rows = [
        row for row in read_csv(END_CONTRASTS)
        if row["left"] == left and row["right"] == right and row["metric"] == metric
    ]
    if len(rows) != 1:
        raise RuntimeError(f"missing contrast: {left}/{right}/{metric}")
    return rows[0]


def validate() -> dict[str, Any]:
    required = [
        run.REGISTRY, run.ROOT / "SCORE_FREEZE_MANIFEST.json", END_PANELS,
        END_CELLS, END_CONTRASTS, DIAG_PANELS, DIAG_CELLS,
        DIAG_CONTRASTS, INTERACTIONS, SUMMARY, run.ROOT / "ARTIFACT_MANIFEST.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("incomplete historical head-to-head artifacts")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETE" or summary["verdict"] != "NUMERICALLY_BETTER_UNRESOLVED":
        raise RuntimeError("unexpected primary result status")
    if abs(summary["historical_aliases"]["entropy"] - 0.3614213583669282) > 1e-15:
        raise RuntimeError("entropy anchor drift")
    if abs(summary["historical_aliases"]["finalist"] - 0.3662328341717007) > 1e-15:
        raise RuntimeError("finalist anchor drift")
    freeze = json.loads((run.ROOT / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_AUDIT_LABEL_OPEN" or freeze["labels_selected"] is not False:
        raise RuntimeError("score freeze boundary failed")
    if sum(row["role_counts"]["audit"] for row in freeze["cells"]) != 1270:
        raise RuntimeError("historical audit population drift")
    return summary


def variant(
    variant_id: str, name: str, parent_ids: list[str], signals: list[str],
    transforms: list[str], detector: str, reducer: str, novelty: str,
    limitation: str, statistical: str = "INCONCLUSIVE",
    decision: str = "NO_PROMOTION", role: str = "matched_historical_bridge",
) -> dict[str, Any]:
    return {
        "variant_id": variant_id, "display_name": name,
        "method_id": "fusion_selection", "phase": "P4", "role": role,
        "parent_variant_ids": parent_ids, "signals": signals,
        "transforms": transforms, "step_reducer": reducer,
        "detector": detector, "fusion": "as frozen in the named parent system",
        "novelty": novelty, "access_tier": "gray_box_single_pass",
        "supervision": "historical 40% calibration labels for detector threshold; fixed 20% audit only after score freeze",
        "causal_validity": "completed-trace retrospective localization; not an early-detection claim",
        "prior_evidence": "Frozen H2/H3 development and transfer artifacts plus checksum-exact Stage-4 replay.",
        "failure_hypothesis": "The apparent current-regime gain does not survive the exact historical population and evaluator.",
        "limitations": limitation, "execution_status": "COMPLETE",
        "decision_status": decision, "evidence_status": "RETROSPECTIVE",
        "statistical_status": statistical, "task_ids": ["processbench_first_error"],
        "rankable": True, "display_order": DISPLAY[variant_id],
    }


def update_registries(summary: dict[str, Any]) -> None:
    vp = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(vp.read_text(encoding="utf-8"))
    existing = {row["variant_id"] for row in payload["variants"]}
    rows = [
        variant("P4H_ENTROPY_TOP5", "Historical entropy/top-five — matched replay", ["H_STAGE4_ENTROPY"], ["primitive entropy"], ["historical calibration"], "historical entropy detector", "step top-five mean", "Rankable checksum-equivalent anchor inside the matched bridge.", "Context comparator; no claim of novelty.", "NOT_EVALUATED"),
        variant("P4H_HIST_FINALIST", "Historical Stage-4 finalist — matched replay", ["H_STAGE4_FINALIST"], ["historical family compression", "RegisteredGlobal response score"], ["historical family fusion"], "historical RegisteredGlobal", "step top-five mean", "Rankable checksum-equivalent 0.3662 system inside its original audit regime.", "Retrospective anchor; historical method was selected on opened development evidence.", "NOT_EVALUATED"),
        variant("P4H_H0_FAMILY_TOP10", "Current H0 — historical-regime application", ["P2C_F6_TOP10_REFERENCE"], ["five-family token curve", "H0 response detector"], ["equal-family fusion"], "H0 threshold fit on historical calibration", "step top-ten mean", "Applies the frozen current parent to the exact Stage-4 population.", "Positive point delta versus the historical finalist, but CI crosses zero.", "PROMISING_UNCONFIRMED"),
        variant("P4H_H2_CLEAN_C7", "H2 cleanup+C7 — historical-regime application", ["P2D_H2_CLEAN_C7"], ["H0 minus sampled energy", "partition without energy_series", "C7 in entropy dynamics"], ["fixed family cleanup", "C7 insertion"], "H0 threshold fit on historical calibration", "step top-ten mean", "Tests the compact cleanup and C7 bundle without changing its frozen definition.", "Raw-best in this bridge, but its paired interval versus the historical finalist crosses zero.", "PROMISING_UNCONFIRMED"),
        variant("P4H_H3_EQUAL_C8", "H3 equal+C8 — historical-regime application", ["P4H_H2_CLEAN_C7", "P2F_H3_EQUAL_C8_RERANK_PRM"], ["H2 step rank", "C8 self-innovation rank"], ["fixed 50/50 within-response rank fusion"], "H0 threshold fit on historical calibration", "step top-ten mean", "Direct matched-regime test of the cross-task H3 candidate against 0.3662.", "H3 is numerically above 0.3662 and retains supported PRMBench ranking evidence, but the ProcessBench CI crosses zero and worst cell loses 0.03461.", "PROMISING_UNCONFIRMED", "PRMBENCH_SPECIALIST", "dual_task_development_candidate"),
        variant("P4H_HISTDET_H0LOC", "Historical detector + H0 localizer", ["P4H_HIST_FINALIST", "P4H_H0_FAMILY_TOP10"], ["historical detector", "H0 localizer"], ["2x2 mechanism cross"], "historical RegisteredGlobal", "step top-ten mean", "Holds the detector fixed to isolate the H0 localizer.", "Mechanism diagnostic only; not a deployable-system ranking.", role="mechanism_diagnostic"),
        variant("P4H_HISTDET_H2LOC", "Historical detector + H2 localizer", ["P4H_HIST_FINALIST", "P4H_H2_CLEAN_C7"], ["historical detector", "H2 localizer"], ["shared-detector diagnostic"], "historical RegisteredGlobal", "step top-ten mean", "Holds the historical detector fixed while inserting H2 localization.", "Mechanism diagnostic only; CI is unresolved.", role="mechanism_diagnostic"),
        variant("P4H_HISTDET_H3LOC", "Historical detector + H3 localizer", ["P4H_HIST_FINALIST", "P4H_H3_EQUAL_C8"], ["historical detector", "H3 localizer"], ["2x2 mechanism cross"], "historical RegisteredGlobal", "step top-ten mean", "Isolates H3 localization under the historical clean/error decision.", "Point delta versus the historical localizer is negative but unresolved.", role="mechanism_diagnostic"),
        variant("P4H_H0DET_HISTLOC", "H0 detector + historical localizer", ["P4H_HIST_FINALIST", "P4H_H0_FAMILY_TOP10"], ["H0 detector", "historical localizer"], ["2x2 mechanism cross"], "H0 threshold fit on historical calibration", "step top-five mean", "Isolates the current detector with the historical localizer.", "Positive detector point contribution, but CI crosses zero.", "PROMISING_UNCONFIRMED", role="mechanism_diagnostic"),
    ]
    if existing.intersection(row["variant_id"] for row in rows):
        raise RuntimeError("historical head-to-head variants already integrated")
    payload["variants"].extend(rows)
    atomic_write_json(vp, payload)

    ep = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(ep.read_text(encoding="utf-8"))
    if any(row["experiment_id"] == run.EXPERIMENT for row in experiments["experiments"]):
        raise RuntimeError("historical head-to-head experiment already integrated")
    experiments["experiments"].append({
        "experiment_id": run.EXPERIMENT,
        "display_name": "Historical-regime H3 head-to-head against 0.3662",
        "phase": "P4", "execution_status": "COMPLETE",
        "question": "Does frozen H3 beat the original 0.3662 system on the exact Stage-4 rows, split, calibration and evaluator?",
        "population_ids": ["historical_stage4_eight_cell_audit_matched"],
        "task_ids": ["processbench_first_error"],
        "prerequisite": "Checksum-exact replay of entropy 0.3614213584 and finalist 0.3662328342, then candidate-score freeze before audit-label opening.",
        "variant_order": [VARIANTS[key] for key in run.END_TO_END],
        "registered_comparators": [VARIANTS[run.FINALIST], VARIANTS[run.H0], VARIANTS[run.H2]],
        "primary_metrics": ["paired_delta_macro_f1", "exact_error", "clean_abstention", "within_one"],
        "bootstrap": "2,000 paired historical source-question grouped draws using seed 20260816.",
        "promotion_gates": [
            "H3-minus-historical macro-F1 CI entirely above zero",
            "historical aliases exact and candidate scores frozen before audit labels",
            "PRMBench advantage retained as a separate estimand",
            "fresh-question confirmation remains required",
        ],
        "report_sections": ["p4h_absolute", "p4h_delta_forest", "p4h_tradeoff", "p4h_2x2"],
        "raw_best": "P4H_H2_CLEAN_C7", "survivors": ["P4H_H2_CLEAN_C7", "P4H_H3_EQUAL_C8"],
        "verdict": "H3_NUMERICALLY_ABOVE_03662_UNRESOLVED__H2_RAW_BEST__NO_DUAL_TASK_PROMOTION",
        "phase4_promotion": False, "fresh_confirmation": False,
        "result_summary_sha256": sha256_file(SUMMARY),
        "next_variant": None,
    })
    atomic_write_json(ep, experiments)


def metric_row(
    variant_id: str, row: dict[str, str], metric: str, source: Path,
    selector: str, *, axis: str = "", slice_id: str = "all",
) -> dict[str, Any]:
    return {
        "phase_id": "P4", "experiment_id": run.EXPERIMENT,
        "variant_id": variant_id, "task_id": "processbench_first_error",
        "dataset_id": "processbench", "population_id": "historical_stage4_eight_cell_audit_matched",
        "cell_id": row.get("cell_id", "aggregate"), "slice_id": slice_id,
        "metric_id": METRIC_ID[metric], "value": row[metric],
        "n_rows": row.get("n", row.get("n_scorer_rows", 1270)),
        "n_groups": row.get("n_groups", 635),
        "comparison_group_id": f"p4h_matched::{METRIC_ID[metric]}",
        "status": "COMPLETE", "evidence_status": "RETROSPECTIVE",
        "display_order": DISPLAY[variant_id], "axis_value": axis,
        "source_artifact": str(source.relative_to(REPO)), "source_sha256": sha256_file(source),
        "source_row_selector": selector, "source_value_field": metric,
        "notes": "Exact historical Stage-4 matched audit; retrospective, not fresh-question confirmation.",
    }


def integrate_metrics() -> None:
    additions: list[dict[str, Any]] = []
    cross_axes = {
        "P4H_HIST_FINALIST": ("historical_detector", "historical_localizer"),
        "P4H_HISTDET_H3LOC": ("historical_detector", "h3_localizer"),
        "P4H_H0DET_HISTLOC": ("h0_detector", "historical_localizer"),
        "P4H_H3_EQUAL_C8": ("h0_detector", "h3_localizer"),
    }
    for row in read_csv(END_PANELS):
        variant_id = VARIANTS[row["candidate"]]
        axis, slice_id = cross_axes.get(variant_id, ("", "all"))
        for metric in run.METRICS:
            additions.append(metric_row(variant_id, row, metric, END_PANELS, f"candidate={row['candidate']}", axis=axis, slice_id=slice_id))
    for row in read_csv(END_CELLS):
        variant_id = VARIANTS[row["candidate"]]
        for metric in run.METRICS:
            additions.append(metric_row(variant_id, row, metric, END_CELLS, f"candidate={row['candidate']};cell_id={row['cell_id']}"))
    for row in read_csv(DIAG_PANELS):
        if row["candidate"] not in DIAGNOSTIC_VARIANTS or row["candidate"] in {"HISTDET_HISTLOC", "H0DET_H3LOC"}:
            continue
        variant_id = DIAGNOSTIC_VARIANTS[row["candidate"]]
        axis, slice_id = cross_axes.get(variant_id, ("", "all"))
        for metric in run.METRICS:
            additions.append(metric_row(variant_id, row, metric, DIAG_PANELS, f"candidate={row['candidate']}", axis=axis, slice_id=slice_id))
    for row in read_csv(DIAG_CELLS):
        if row["candidate"] not in DIAGNOSTIC_VARIANTS or row["candidate"] in {"HISTDET_HISTLOC", "H0DET_H3LOC"}:
            continue
        variant_id = DIAGNOSTIC_VARIANTS[row["candidate"]]
        for metric in run.METRICS:
            additions.append(metric_row(variant_id, row, metric, DIAG_CELLS, f"candidate={row['candidate']};cell_id={row['cell_id']}"))
    append_rows(p1.PROGRAM_ROOT / "METRICS_LONG.csv", additions, ("experiment_id", "variant_id", "metric_id", "cell_id"))


def contrast_row(row: dict[str, str], source: Path, left: str, right: str, metric_id: str | None = None) -> dict[str, Any]:
    return {
        "phase_id": "P4", "experiment_id": run.EXPERIMENT,
        "left_variant_id": left, "right_variant_id": right,
        "task_id": "processbench_first_error", "dataset_id": "processbench",
        "population_id": "historical_stage4_eight_cell_audit_matched",
        "metric_id": metric_id or METRIC_ID[row["metric"]],
        "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
        "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
        "worst_unit_delta": row["worst_cell_delta"],
        "comparison_group_id": f"p4h_matched::{metric_id or METRIC_ID[row['metric']]}",
        "status": "COMPLETE", "evidence_status": "RETROSPECTIVE",
        "source_artifact": str(source.relative_to(REPO)), "source_sha256": sha256_file(source),
        "source_row_selector": f"contrast_id={row['contrast_id']}",
        "notes": row["interval"] + "; worst cell=" + row["worst_cell"],
    }


def integrate_contrasts() -> None:
    additions = [
        contrast_row(row, END_CONTRASTS, VARIANTS[row["left"]], VARIANTS[row["right"]])
        for row in read_csv(END_CONTRASTS)
    ]
    additions.extend(
        contrast_row(row, DIAG_CONTRASTS, DIAGNOSTIC_VARIANTS[row["left"]], DIAGNOSTIC_VARIANTS[row["right"]])
        for row in read_csv(DIAG_CONTRASTS)
    )
    additions.extend(
        contrast_row(
            row, INTERACTIONS, VARIANTS[run.H3], VARIANTS[run.FINALIST],
            f"detector_localizer_interaction_{METRIC_ID[row['metric']]}",
        )
        for row in read_csv(INTERACTIONS)
    )
    append_rows(
        p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv", additions,
        ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"),
    )


def prior_prm_contrast() -> dict[str, str]:
    rows = [
        row for row in read_csv(p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv")
        if row["experiment_id"] == "P2_H3_PRMBENCH_DIAGNOSTIC"
        and row["left_variant_id"] == "P2F_H3_EQUAL_C8_RERANK_PRM"
        and row["right_variant_id"] == "P2F_H0_FAMILY6_TOP10_PRM"
        and row["metric_id"] == "auroc"
    ]
    if len(rows) != 1:
        raise RuntimeError("missing frozen H3 PRMBench contrast")
    return rows[0]


def integrate_gates(summary: dict[str, Any]) -> None:
    primary = summary["primary_contrast"]
    prm = prior_prm_contrast()
    rows = [
        (VARIANTS[run.FINALIST], "P4H_HISTORICAL_ALIAS", "max_abs_error", 0.0, 1e-15, "le", True, "PASS", "Both historical F1 anchors replay exactly."),
        (VARIANTS[run.H3], "P4H_SCORE_FREEZE", "labels_selected_before_freeze", False, False, "eq", True, "PASS", "All candidate locators froze before audit labels opened."),
        (VARIANTS[run.H3], "P4H_POPULATION", "audit_rows", 1270, 1270, "eq", True, "PASS", "Exact 635-group, 1,270-scorer-row Stage-4 audit."),
        (VARIANTS[run.H3], "P4H_PRIMARY_DIRECTIONAL", "macro_f1_ci_low", primary["ci_low"], 0.0, "gt", False, "INCONCLUSIVE", "Positive point estimate, but the paired interval crosses zero; not rejected."),
        (VARIANTS[run.H3], "P4H_PRIMARY_PRACTICAL", "macro_f1_ci_low", primary["ci_low"], 0.003, "gt", False, "INCONCLUSIVE", "+0.003 is practical context, not the inferential null."),
        (VARIANTS[run.H3], "P4H_PRMBENCH_ADVANTAGE", "auroc_ci_low", prm["ci_low"], 0.003, "gt", float(prm["ci_low"]) > 0.003, "PASS", "Previously frozen PRMBench H3-vs-H0 advantage remains supported and separate."),
        (VARIANTS[run.H3], "P4H_DUAL_TASK_PROMOTION", "promotion_eligible", False, True, "eq", False, "INCONCLUSIVE", "PRMBench passes, but matched ProcessBench improvement is unresolved."),
        (VARIANTS[run.H3], "P4H_FRESH_CONFIRMATION", "fresh_population", False, True, "eq", False, "BLOCKED", "The exact questions were historically opened; independent questions remain required."),
    ]
    fields = ["gate_id", "variant_id", "metric_id", "observed", "threshold", "direction", "passed", "status", "evidence_status", "notes"]
    source_rows = [{
        "gate_id": gate, "variant_id": variant_id, "metric_id": metric,
        "observed": str(observed).lower() if isinstance(observed, bool) else observed,
        "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
        "direction": direction, "passed": str(passed).lower(), "status": status,
        "evidence_status": "RETROSPECTIVE", "notes": note,
    } for variant_id, gate, metric, observed, threshold, direction, passed, status, note in rows]
    write_csv(GATE_SOURCE, source_rows, fields)
    source_sha = sha256_file(GATE_SOURCE)
    additions = [{
        "phase_id": "P4", "experiment_id": run.EXPERIMENT,
        "variant_id": row["variant_id"], "gate_id": row["gate_id"], "metric_id": row["metric_id"],
        "observed": row["observed"], "threshold": row["threshold"], "direction": row["direction"],
        "passed": row["passed"], "unit": "boolean" if row["observed"] in {"true", "false"} else "fraction",
        "status": row["status"], "evidence_status": row["evidence_status"],
        "source_artifact": str(GATE_SOURCE.relative_to(REPO)), "source_sha256": source_sha,
        "source_row_selector": f"gate_id={row['gate_id']}", "source_value_field": "observed",
        "notes": row["notes"],
    } for row in source_rows]
    append_rows(p1.PROGRAM_ROOT / "GATES_LONG.csv", additions, ("experiment_id", "variant_id", "gate_id"))


def register_plots_and_claims(summary: dict[str, Any]) -> None:
    pp = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(pp.read_text(encoding="utf-8"))
    plots["plots"].extend([
        {"plot_id": "PLOT_P4H_ABSOLUTE_F1", "title": "Historical-regime absolute ProcessBench macro F1", "phase": "P4", "kind": "forest", "source_table": "METRICS_LONG.csv", "selection": {"experiment_id": run.EXPERIMENT, "metric_id": "macro_f1", "cell_id": "aggregate", "variant_id": list(VARIANTS.values())}, "x_field": "value", "y_field": "variant_id", "series_field": "metric_id", "comparison_group": "same eight Stage-4 cells, 1,270 scorer rows and 635 paired source questions", "bootstrap_definition": "Absolute points; paired uncertainty appears in the aligned contrast forest.", "selection_rule": "All five preregistered end-to-end systems in frozen order.", "legend": ["Point = historical-evaluator macro F1", "All rows are rankable only inside this matched bridge"], "caption": "H2 is raw-best at 0.37479, H0 reaches 0.37410, H3 reaches 0.37266, and the historical finalist is reproduced at 0.36623."},
        {"plot_id": "PLOT_P4H_DELTA_FOREST", "title": "Paired deltas versus the historical 0.3662 finalist", "phase": "P4", "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv", "selection": {"experiment_id": run.EXPERIMENT, "metric_id": "macro_f1", "right_variant_id": VARIANTS[run.FINALIST]}, "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id", "comparison_group": "same Stage-4 audit rows and whole-question pairing", "bootstrap_definition": "2,000 paired historical source-question grouped draws; seed 20260816.", "selection_rule": "Every preregistered end-to-end macro-F1 contrast against the historical finalist.", "legend": ["Point and line = paired delta and 95% interval", "Crossing zero means unresolved, not rejected"], "caption": "H3 is +0.00643 above the historical finalist, but its interval [-0.02689,+0.03947] crosses zero. H2 is the raw-best arm; none has a supported matched-regime improvement."},
        {"plot_id": "PLOT_P4H_EXACT_CLEAN", "title": "Exact-error versus clean-abstention change", "phase": "P4", "kind": "scatter", "source_table": "CONTRASTS_LONG.csv", "selection": {"experiment_id": run.EXPERIMENT, "right_variant_id": VARIANTS[run.FINALIST], "metric_id": ["exact_error", "clean_abstention"], "left_variant_id": [VARIANTS[run.H0], VARIANTS[run.H2], VARIANTS[run.H3]]}, "x_field": "processbench_delta_clean_abstention", "y_field": "processbench_delta_exact_error", "series_field": "left_variant_id", "comparison_group": "same Stage-4 audit rows", "bootstrap_definition": "Coordinates are paired point deltas; interval details remain in CONTRASTS_LONG.csv.", "selection_rule": "H0, H2 and H3, each relative to the historical finalist, on the two preregistered decomposition metrics.", "legend": ["Right = improved clean abstention", "Up = improved exact first-error localization"], "caption": "The current detector improves clean abstention, while exact first-error point estimates are slightly lower; H2/H3 share the H0 abstention decision exactly."},
        {"plot_id": "PLOT_P4H_2X2", "title": "Detector × localizer 2×2 macro-F1 cross", "phase": "P4", "kind": "heatmap", "source_table": "METRICS_LONG.csv", "selection": {"experiment_id": run.EXPERIMENT, "metric_id": "macro_f1", "cell_id": "aggregate", "variant_id": [VARIANTS[run.FINALIST], "P4H_HISTDET_H3LOC", "P4H_H0DET_HISTLOC", VARIANTS[run.H3]]}, "x_field": "axis_value", "y_field": "slice_id", "series_field": "variant_id", "comparison_group": "same four detector/localizer combinations on the Stage-4 audit", "bootstrap_definition": "2,000 paired grouped draws for each neighboring contrast and the difference-in-differences interaction.", "selection_rule": "The preregistered historical/H0 detector crossed with historical/H3 localizer.", "legend": ["Columns = detector", "Rows = localizer", "Interaction CI is reported in the contrast table"], "caption": "The current detector has the larger favorable point contribution; H3 localization does not show an isolated gain. The interaction is -0.00362 with CI [-0.01216,+0.00436]."},
    ])
    atomic_write_json(pp, plots)

    primary = summary["primary_contrast"]
    cp = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(cp.read_text(encoding="utf-8"))
    claims["claims"].extend([
        {"claim_id": "CLAIM_P4H_H3_MATCHED_03662", "text": "On the exact historical Stage-4 regime, frozen H3 is numerically above the 0.3662 finalist, while its already-frozen PRMBench advantage remains supported; the ProcessBench improvement claim is unresolved rather than rejected.", "verdict": "PROMISING_UNCONFIRMED", "task_scope": "Historical eight-cell ProcessBench audit, 1,270 scorer rows grouped as 635 source questions; PRMBench remains a separate prior transfer estimand.", "evidence_refs": ["PLOT_P4H_DELTA_FOREST", f"CONTRAST:{VARIANTS[run.H3]}:{VARIANTS[run.FINALIST]}", "TABLE_GATES"], "worst_case_behavior": "H3 wins 5/8 cells but loses 3/8; worst cell is Llama GSM8K at -0.03461 macro F1.", "claim_boundary": "The interval crosses zero, the questions are historically opened, and the result does not satisfy dual-task promotion or fresh confirmation.", "fresh_confirmation_required": True, "statistical_summary": {"metric": "macro_f1", "point_delta": primary["delta"], "ci_low": primary["ci_low"], "ci_high": primary["ci_high"], "benefit_bound": 0.003, "harm_bound": -0.005, "bound_basis": "+0.003 is practical context only; zero is the head-to-head inferential boundary.", "multiplicity": "Preregistered primary paired contrast; 2,000 grouped bootstrap draws."}},
        {"claim_id": "CLAIM_P4H_ATTRIBUTION", "text": "The matched bridge does not support an isolated H3-localizer gain or a detector-by-localizer interaction; the favorable end-to-end point difference is descriptively concentrated in the current detector's clean-abstention behavior.", "verdict": "INCONCLUSIVE", "task_scope": "Historical Stage-4 ProcessBench detector/localizer mechanism cross.", "evidence_refs": ["PLOT_P4H_2X2", "PLOT_P4H_EXACT_CLEAN", "TABLE_CONTRASTS"], "worst_case_behavior": "Under the historical detector, H3 localization is -0.00307 macro F1 versus the historical localizer; CI [-0.02845,+0.02062].", "claim_boundary": "All detector, localizer and interaction intervals cross zero; point attribution is descriptive, not causal proof.", "fresh_confirmation_required": True, "statistical_summary": {"metric": "detector_localizer_interaction_macro_f1", "point_delta": summary["detector_localizer_interaction"]["f1"]["delta"], "ci_low": summary["detector_localizer_interaction"]["f1"]["ci_low"], "ci_high": summary["detector_localizer_interaction"]["f1"]["ci_high"], "benefit_bound": 0.003, "harm_bound": -0.005, "bound_basis": "Same practical context bounds; no interaction promotion gate.", "multiplicity": "2,000 paired grouped draws for the preregistered 2x2 diagnostic."}},
    ])
    atomic_write_json(cp, claims)


def refresh_artifact_manifest() -> None:
    manifest = run.ROOT / "ARTIFACT_MANIFEST.json"
    artifacts = [
        {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for path in sorted(run.ROOT.rglob("*"))
        if path.is_file() and path != manifest
    ]
    atomic_write_json(manifest, {
        "schema": "reasoning-localization-h3-historical-artifacts-v1",
        "status": "COMPLETE", "experiment_id": run.EXPERIMENT,
        "artifacts": artifacts,
        "plot_rule": "Standalone SVGs are deterministic functions of the registered panel and contrast tables; the living report renders from long-form registries.",
    })


def main() -> None:
    summary = validate()
    update_registries(summary)
    integrate_metrics()
    integrate_contrasts()
    integrate_gates(summary)
    register_plots_and_claims(summary)
    refresh_artifact_manifest()
    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({
        "experiment": run.EXPERIMENT, "verdict": summary["verdict"],
        "h3_f1": summary["panels"][run.H3]["f1"],
        "historical_f1": summary["panels"][run.FINALIST]["f1"],
        "report_sha256": build.manifest["output"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
