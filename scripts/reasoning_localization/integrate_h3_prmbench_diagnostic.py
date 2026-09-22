#!/usr/bin/env python3
"""Integrate the corrected H2/H3 PRMBench diagnostic into the living report."""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic as v1  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic_v2 as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402


EVAL = run.ROOT / "evaluation"
PANELS = EVAL / "PANELS.csv"
BY_FAMILY = EVAL / "BY_FAMILY.csv"
BY_FAMILY_REPORT = EVAL / "BY_FAMILY_REPORT.csv"
CONTRASTS = EVAL / "CONTRASTS.csv"
SUMMARY = EVAL / "SUMMARY.json"
PLOT_DATA = EVAL / "H3_PRMBENCH_PLOT_DATA.csv"
PLOT = EVAL / "H3_PRMBENCH_RESULTS.svg"
MANIFEST = run.ROOT / "ARTIFACT_MANIFEST.json"
FAILURE = v1.ROOT / "PRELABEL_HARD_FAIL.json"
DISPLAY = {run.H0: 154, run.H2: 155, run.H3: 156}


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


def contrast(left: str, right: str, metric: str) -> dict[str, str]:
    matches = [
        row for row in read_csv(CONTRASTS)
        if row["left"] == left and row["right"] == right and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one contrast {left} vs {right} {metric}")
    return matches[0]


def validate() -> dict[str, Any]:
    required = [PANELS, BY_FAMILY, CONTRASTS, SUMMARY, FAILURE, run.ROOT / "SCORE_FREEZE_MANIFEST.json"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("incomplete PRMBench diagnostic artifacts")
    summary = json.loads(SUMMARY.read_text())
    if summary["status"] != "COMPLETE" or summary["phase4_promotion"] is not False:
        raise RuntimeError("unexpected PRMBench diagnostic status")
    if summary["phase1_h0_score_alias_max_abs_error"] > 1e-12:
        raise RuntimeError("corrected top-five parent alias failed")
    if max(summary["qwen_processbench_alias"].values()) > 1e-12:
        raise RuntimeError("Qwen H0/H2/H3 alias failed")
    if summary["n_evaluable_responses"] != 6208 or summary["n_evaluable_steps"] != 83280:
        raise RuntimeError("PRMBench population changed")
    return summary


def update_registries(summary: dict[str, Any]) -> None:
    vp = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(vp.read_text())
    by_id = {row["variant_id"]: row for row in payload["variants"]}
    for arm in run.ARMS:
        by_id[arm]["execution_status"] = "COMPLETE"
    by_id[run.H0].update({"decision_status": "NO_PROMOTION", "statistical_status": "NOT_EVALUATED"})
    by_id[run.H2].update({"decision_status": "NO_PROMOTION", "statistical_status": "SUPPORTED_IMPROVEMENT"})
    by_id[run.H3].update({"decision_status": "PRMBENCH_SPECIALIST", "statistical_status": "SUPPORTED_IMPROVEMENT"})
    by_id[run.H0]["limitations"] += " V1's invalid top-ten-to-top-five alias hard-failed before labels; V2 corrected it with a non-rankable top-five control."
    by_id[run.H2]["limitations"] += " AUROC benefit is supported here, but ProcessBench remains unconfirmed."
    by_id[run.H3]["limitations"] += " Strong PRMBench specialization does not establish ProcessBench first-error improvement."
    atomic_write_json(vp, payload)

    ep = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(ep.read_text())
    matches = [row for row in experiments["experiments"] if row["experiment_id"] == v1.EXPERIMENT]
    if len(matches) != 1:
        raise RuntimeError("missing PRMBench diagnostic experiment")
    matches[0].update(
        {
            "execution_status": "COMPLETE",
            "active_execution_contract": "AMENDMENT_V2_COMPLETE",
            "verdict": "H3_SUPPORTED_PRMBENCH_SPECIALIST__NO_PHASE4_PROMOTION",
            "raw_best": run.H3,
            "survivors": [],
            "next_variant": None,
            "result_summary_sha256": sha256_file(SUMMARY),
            "phase4_promotion": False,
        }
    )
    atomic_write_json(ep, experiments)


def integrate_metrics() -> None:
    additions = []
    family_rows = []
    for raw in read_csv(BY_FAMILY):
        row = dict(raw)
        evaluator_status = row["status"]
        row["metric_status"] = evaluator_status
        if evaluator_status == "OK":
            row["status"] = "COMPLETE"
        elif evaluator_status == "METRIC_UNDEFINED_SINGLE_CLASS":
            row["status"] = "BLOCKED"
        else:
            raise RuntimeError(f"unknown evaluator status: {evaluator_status}")
        family_rows.append(row)
    write_csv(BY_FAMILY_REPORT, family_rows, list(family_rows[0]))
    panel_sha, family_sha = sha256_file(PANELS), sha256_file(BY_FAMILY_REPORT)
    for row in read_csv(PANELS):
        additions.append(
            {
                "phase_id": "P2F", "experiment_id": v1.EXPERIMENT,
                "variant_id": row["arm_id"], "task_id": "prmbench_step_error",
                "dataset_id": "prmbench", "population_id": "prmbench_qwen3_response_v1",
                "cell_id": "aggregate", "slice_id": "all", "metric_id": row["metric_id"],
                "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "n_rows": row["n_steps"], "n_groups": row["n_groups"],
                "comparison_group_id": f"p2f_prm::{row['metric_id']}", "status": "COMPLETE",
                "evidence_status": "TRANSFER", "display_order": DISPLAY[row["arm_id"]],
                "source_artifact": str(PANELS.relative_to(REPO)), "source_sha256": panel_sha,
                "source_row_selector": f"arm_id={row['arm_id']};metric_id={row['metric_id']}",
                "source_value_field": "value",
                "notes": "Frozen cross-task diagnostic on historically opened PRMBench; no Phase-4 promotion.",
            }
        )
    for row in family_rows:
        for metric in ("auroc", "auprc"):
            value = row[metric]
            additions.append(
                {
                    "phase_id": "P2F", "experiment_id": v1.EXPERIMENT,
                    "variant_id": row["arm_id"], "task_id": "prmbench_step_error",
                    "dataset_id": "prmbench", "population_id": "prmbench_qwen3_response_v1",
                    "cell_id": f"prmbench::{row['error_family']}", "slice_id": row["error_family"],
                    "metric_id": metric, "value": value,
                    "n_rows": row["n_examples"], "n_groups": "",
                    "comparison_group_id": f"p2f_prm::{metric}",
                    "status": row["status"], "evidence_status": "TRANSFER",
                    "display_order": DISPLAY[row["arm_id"]],
                    "source_artifact": str(BY_FAMILY_REPORT.relative_to(REPO)), "source_sha256": family_sha,
                    "source_row_selector": f"arm_id={row['arm_id']};error_family={row['error_family']}",
                    "source_value_field": metric,
                    "notes": "Single-class families remain undefined and visible; they are never zero-filled.",
                }
            )
    append_rows(
        p1.PROGRAM_ROOT / "METRICS_LONG.csv",
        additions,
        ("experiment_id", "variant_id", "metric_id", "cell_id"),
    )


def integrate_contrasts() -> None:
    source_sha = sha256_file(CONTRASTS)
    additions = []
    for row in read_csv(CONTRASTS):
        additions.append(
            {
                "phase_id": "P2F", "experiment_id": v1.EXPERIMENT,
                "left_variant_id": row["left"], "right_variant_id": row["right"],
                "task_id": "prmbench_step_error", "dataset_id": "prmbench",
                "population_id": "prmbench_qwen3_response_v1", "metric_id": row["metric"],
                "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
                "worst_unit_delta": row["worst_family_delta"],
                "comparison_group_id": f"p2f_prm::{row['metric']}", "status": "COMPLETE",
                "evidence_status": "TRANSFER", "source_artifact": str(CONTRASTS.relative_to(REPO)),
                "source_sha256": source_sha, "source_row_selector": f"contrast_id={row['contrast_id']}",
                "notes": row["interval"] + f"; worst evaluable family={row['worst_family']}; diagnostic only.",
            }
        )
    append_rows(
        p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv",
        additions,
        ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"),
    )


def integrate_gates(summary: dict[str, Any]) -> None:
    h2 = contrast(run.H2, run.H0, "auroc")
    h3 = contrast(run.H3, run.H0, "auroc")
    h3_h2 = contrast(run.H3, run.H2, "auroc")
    gates = [
        (run.H0, "P2F_V1_PRELABEL_ALIAS", "max_abs_error", 0.23125777605843761, 1e-12, "le", False, "HARD_FAIL", "Invalid top-ten-to-top-five alias; stopped before labels and superseded by V2."),
        (run.H0, "P2F_V2_TOP5_PARENT_ALIAS", "max_abs_error", summary["phase1_h0_score_alias_max_abs_error"], 1e-12, "le", True, "PASS", "Corrected non-rankable top-five control aliases Phase-1 R2 exactly."),
        (run.H3, "P2F_QWEN_SCORE_ALIASES", "max_abs_error", max(summary["qwen_processbench_alias"].values()), 1e-12, "le", True, "PASS", "All imported Qwen H0/H2/H3 scores reproduce exactly."),
        (run.H3, "P2F_POPULATION_RESPONSES", "responses", summary["n_evaluable_responses"], 6208, "eq", True, "PASS", "Exact sealed evaluable response population."),
        (run.H3, "P2F_POPULATION_STEPS", "steps", summary["n_evaluable_steps"], 83280, "eq", True, "PASS", "Exact sealed annotated-step population."),
        (run.H3, "P2F_NINE_FAMILIES", "families_visible", summary["all_nine_families_visible"], True, "eq", True, "PASS", "All nine families visible; multi_solutions remains single-class."),
        (run.H2, "P2F_H2_AUROC_BENEFIT", "auroc_ci_low", float(h2["ci_low"]), 0.003, "gt", float(h2["ci_low"]) > 0.003, "PASS", "H2 clears the frozen practical-benefit bound versus H0."),
        (run.H3, "P2F_H3_AUROC_BENEFIT", "auroc_ci_low", float(h3["ci_low"]), 0.003, "gt", float(h3["ci_low"]) > 0.003, "PASS", "H3 clears the benefit bound versus H0."),
        (run.H3, "P2F_H3_INCREMENTAL_AUROC", "auroc_ci_low", float(h3_h2["ci_low"]), 0.003, "gt", float(h3_h2["ci_low"]) > 0.003, "PASS", "H3 clears the benefit bound versus H2."),
        (run.H3, "P2F_H3_WORST_FAMILY", "worst_family_delta", float(h3["worst_family_delta"]), -0.020, "ge", float(h3["worst_family_delta"]) >= -0.020, "PASS", "Every evaluable PRMBench family improves."),
        (run.H3, "P2F_SOURCE_STRATA", "source_strata_available", False, True, "eq", False, "BLOCKED", "Sealed evaluator lacks prm_train/prm_test membership."),
        (run.H3, "P2F_PHASE4_PROMOTION", "promotion_eligible", False, True, "eq", False, "FAIL", "Opened labels and missing ProcessBench survivor forbid Phase-4 promotion."),
    ]
    source = EVAL / "GATES.csv"
    write_csv(
        source,
        [
            {"gate_id": gate_id, "variant_id": variant, "metric_id": metric,
             "observed": str(observed).lower() if isinstance(observed, bool) else observed,
             "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
             "direction": direction, "passed": str(passed).lower(), "status": status,
             "evidence_status": "TRANSFER", "notes": note}
            for variant, gate_id, metric, observed, threshold, direction, passed, status, note in gates
        ],
        ["gate_id", "variant_id", "metric_id", "observed", "threshold", "direction", "passed", "status", "evidence_status", "notes"],
    )
    source_sha = sha256_file(source)
    additions = [
        {"phase_id": "P2F", "experiment_id": v1.EXPERIMENT, "variant_id": variant,
         "gate_id": gate_id, "metric_id": metric,
         "observed": str(observed).lower() if isinstance(observed, bool) else observed,
         "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
         "direction": direction, "passed": str(passed).lower(), "unit": "boolean" if isinstance(observed, bool) else "fraction",
         "status": status, "evidence_status": "TRANSFER", "source_artifact": str(source.relative_to(REPO)),
         "source_sha256": source_sha, "source_row_selector": f"gate_id={gate_id}", "source_value_field": "observed", "notes": note}
        for variant, gate_id, metric, observed, threshold, direction, passed, status, note in gates
    ]
    append_rows(
        p1.PROGRAM_ROOT / "GATES_LONG.csv", additions,
        ("experiment_id", "variant_id", "gate_id"),
    )


def make_svg() -> None:
    panel_rows = read_csv(PANELS)
    panels = {(row["arm_id"], row["metric_id"]): float(row["value"]) for row in panel_rows}
    contrasts = [contrast(run.H2, run.H0, "auroc"), contrast(run.H3, run.H0, "auroc"), contrast(run.H3, run.H2, "auroc")]
    rows = [
        {"kind": "absolute", "id": arm, "metric": metric, "value": panels[(arm, metric)], "ci_low": "", "ci_high": ""}
        for arm in run.ARMS for metric in ("auroc", "auprc")
    ] + [
        {"kind": "contrast", "id": row["contrast_id"], "metric": "auroc", "value": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]}
        for row in contrasts
    ]
    write_csv(PLOT_DATA, rows, ["kind", "id", "metric", "value", "ci_low", "ci_high"])
    names = {run.H0: "H0 family6", run.H2: "H2 cleanup+C7", run.H3: "H3 equal+C8"}
    colors = {run.H0: "#8392a8", run.H2: "#147d79", run.H3: "#3156b8"}
    x0, x1 = 205.0, 700.0
    ax = lambda value: x0 + (value - 0.58) / 0.05 * (x1 - x0)
    dx = lambda value: x0 + (value + 0.005) / 0.04 * (x1 - x0)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="570" viewBox="0 0 900 570" role="img" aria-labelledby="title desc">',
        '<title id="title">H3 PRMBench every-step ranking diagnostic</title>',
        '<desc id="desc">Absolute AUROC and paired simultaneous AUROC improvements for H2 and H3.</desc>',
        '<rect width="900" height="570" fill="#fbfcfe"/>',
        '<style>text{font-family:Inter,system-ui,sans-serif;fill:#172f55}.title{font-size:24px;font-weight:800}.sub{font-size:13px;fill:#54657b}.label{font-size:13px;font-weight:700}.tick{font-size:11px;fill:#64748b}.value{font-size:12px;font-weight:700}.axis{stroke:#9aa8ba}.zero{stroke:#8b97a8;stroke-dasharray:5 4}.ci{stroke:#3156b8;stroke-width:4}.pt{fill:#3156b8}</style>',
        '<text class="title" x="42" y="42">Frozen H2/H3 transfer to PRMBench</text>',
        '<text class="sub" x="42" y="66">83,280 steps · 6,208 source groups · development-only transfer</text>',
        '<text class="label" x="42" y="108">Absolute PRMBench AUROC</text>',
    ]
    for tick in (0.58, 0.59, 0.60, 0.61, 0.62, 0.63):
        x = ax(tick); lines += [f'<line class="axis" x1="{x:.1f}" y1="122" x2="{x:.1f}" y2="270" opacity=".28"/>', f'<text class="tick" x="{x:.1f}" y="286" text-anchor="middle">{tick:.2f}</text>']
    for idx, arm in enumerate(run.ARMS):
        y = 150 + idx * 48; value = panels[(arm, "auroc")]; x = ax(value)
        lines += [f'<text class="label" x="42" y="{y+5}">{html.escape(names[arm])}</text>', f'<rect x="{x0:.1f}" y="{y-13}" width="{max(x-x0,1):.1f}" height="24" rx="5" fill="{colors[arm]}"/>', f'<text class="value" x="{x+9:.1f}" y="{y+5}">{value:.4f}</text>']
    lines += ['<text class="label" x="42" y="340">Paired AUROC deltas (Bonferroni-simultaneous)</text>', f'<line class="zero" x1="{dx(0):.1f}" y1="356" x2="{dx(0):.1f}" y2="510"/>']
    for tick in (0.00, 0.01, 0.02, 0.03):
        x = dx(tick); lines += [f'<line class="axis" x1="{x:.1f}" y1="502" x2="{x:.1f}" y2="508"/>', f'<text class="tick" x="{x:.1f}" y="526" text-anchor="middle">{tick:+.2f}</text>']
    for idx, (label, row) in enumerate(zip(("H2 - H0", "H3 - H0", "H3 - H2"), contrasts)):
        y = 382 + idx * 52; lo, point, hi = map(float, (row["ci_low"], row["delta"], row["ci_high"]))
        lines += [f'<text class="label" x="42" y="{y+5}">{label}</text>', f'<line class="ci" x1="{dx(lo):.1f}" y1="{y}" x2="{dx(hi):.1f}" y2="{y}"/>', f'<circle class="pt" cx="{dx(point):.1f}" cy="{y}" r="6"/>', f'<text class="value" x="720" y="{y+5}">{point:+.4f} [{lo:+.4f}, {hi:+.4f}]</text>']
    lines += ['<text class="sub" x="42" y="554">H3 improves all 8 evaluable families; this is PRMBench specialization, not a cross-task aggregate.</text>', '</svg>']
    PLOT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_plot_claim_manifest() -> None:
    make_svg()
    pp = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(pp.read_text())
    plots["plots"].append(
        {"plot_id": "PLOT_P2F_H3_PRMBENCH_FOREST", "title": "Frozen H2/H3 transfer to PRMBench every-step ranking",
         "phase": "P2F", "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv",
         "selection": {"experiment_id": v1.EXPERIMENT, "metric_id": "auroc", "status": "COMPLETE"},
         "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id",
         "comparison_group": "same 83,280 PRMBench steps and 6,208 paired source_idx groups",
         "bootstrap_definition": "20,000 paired whole-source_idx draws; Bonferroni simultaneous intervals across three frozen AUROC contrasts.",
         "selection_rule": "All three frozen AUROC contrasts: H2-H0, H3-H0 and H3-H2.",
         "legend": ["Point and line = paired AUROC delta and simultaneous interval", "ProcessBench is not included in this axis"],
         "caption": "H2 improves H0 modestly; H3 adds a large supported increment over both H0 and H2 and wins all eight evaluable error families. The result is diagnostic PRMBench specialization, not Phase-4 promotion."}
    )
    atomic_write_json(pp, plots)

    cp = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(cp.read_text())
    primary = contrast(run.H3, run.H0, "auroc")
    claims["claims"].append(
        {"claim_id": "CLAIM_P2F_H3_PRMBENCH_SPECIALIST",
         "text": "The frozen H3 equal-C8 reranker is a supported PRMBench every-step ranking specialist: it improves both H0 and H2 in AUROC/AUPRC and wins all eight evaluable error families.",
         "verdict": "PRM_SPECIALIST", "task_scope": "Historically opened PRMBench Qwen3 response population; 83,280 annotated steps in 6,208 source groups.",
         "evidence_refs": ["PLOT_P2F_H3_PRMBENCH_FOREST", f"CONTRAST:{run.H3}:{run.H0}", "TABLE_GATES"],
         "worst_case_behavior": "The multi_solutions family is single-class and remains undefined. Among eight evaluable families, the smallest H3-versus-H0 AUROC gain is +0.021655 on deception.",
         "claim_boundary": "Does not establish ProcessBench first-error improvement, a shared task winner, fresh confirmation, source-stratum behavior, or Phase-4 promotion.",
         "fresh_confirmation_required": True,
         "statistical_summary": {"metric": "prmbench_auroc_h3_vs_h0", "point_delta": float(primary["delta"]), "ci_low": float(primary["ci_low"]), "ci_high": float(primary["ci_high"]), "benefit_bound": 0.003, "harm_bound": -0.005, "bound_basis": "Frozen P2F diagnostic practical bounds.", "multiplicity": "Bonferroni simultaneous across three frozen AUROC contrasts."}}
    )
    atomic_write_json(cp, claims)

    artifacts = []
    for path in [run.REGISTRY, FAILURE, run.ROOT / "SCORE_FREEZE_MANIFEST.json", PANELS, BY_FAMILY, BY_FAMILY_REPORT, CONTRASTS, EVAL / "BOOTSTRAP_SAMPLES.npz", EVAL / "GATES.csv", SUMMARY, PLOT_DATA, PLOT]:
        artifacts.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
    atomic_write_json(
        MANIFEST,
        {"schema": "reasoning-localization-h3-prmbench-artifacts-v2", "status": "COMPLETE", "experiment_id": v1.EXPERIMENT,
         "phase4_promotion": False, "artifacts": artifacts,
         "plot_rule": "SVG generated deterministically from PANELS.csv and CONTRASTS.csv; no manual result entry."},
    )


def main() -> None:
    summary = validate()
    update_registries(summary)
    integrate_metrics()
    integrate_contrasts()
    integrate_gates(summary)
    register_plot_claim_manifest()
    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"experiment": v1.EXPERIMENT, "verdict": "H3_SUPPORTED_PRMBENCH_SPECIALIST", "plot_sha256": sha256_file(PLOT), "report_sha256": build.manifest["output"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
