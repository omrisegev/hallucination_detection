#!/usr/bin/env python3
"""Integrate the frozen H3 Llama scorer-transfer result into the living report."""

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
from scripts.reasoning_localization import run_h3_llama_transfer as run  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402


EVAL = run.ROOT / "evaluation"
PANELS = EVAL / "PANELS.csv"
BY_CELL = EVAL / "BY_CELL.csv"
CONTRASTS = EVAL / "CONTRASTS.csv"
SUMMARY = EVAL / "SUMMARY.json"
PLOT_DATA = EVAL / "H3_LLAMA_TRANSFER_PLOT_DATA.csv"
PLOT = EVAL / "H3_LLAMA_TRANSFER_RESULTS.svg"
ARTIFACT_MANIFEST = run.ROOT / "ARTIFACT_MANIFEST.json"
DISPLAY = {run.H0: 147, run.H2: 148, run.H3: 149}


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


def validate_result() -> dict[str, Any]:
    required = [PANELS, BY_CELL, CONTRASTS, SUMMARY, run.ROOT / "SCORE_FREEZE_MANIFEST.json"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("incomplete H3 Llama transfer artifacts")
    summary = json.loads(SUMMARY.read_text())
    if summary.get("status") != "COMPLETE" or summary.get("fresh_confirmation") is not False:
        raise RuntimeError("unexpected transfer result status")
    if max(summary["qwen_alias"].values()) > 1e-12:
        raise RuntimeError("Qwen score alias failed")
    if any(summary["abstention_mismatches"].values()):
        raise RuntimeError("H2/H3 abstention alias failed")
    if len(read_csv(PANELS)) != 15 or len(read_csv(BY_CELL)) != 12 or len(read_csv(CONTRASTS)) != 15:
        raise RuntimeError("unexpected result table cardinality")
    return summary


def update_registries() -> None:
    vp = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(vp.read_text())
    by_id = {row["variant_id"]: row for row in variants["variants"]}
    for arm in run.ARMS:
        row = by_id[arm]
        row["execution_status"] = "COMPLETE"
        row["decision_status"] = "NO_PROMOTION"
        row["statistical_status"] = (
            "NOT_EVALUATED" if arm == run.H0 else "PROMISING_UNCONFIRMED"
        )
    by_id[run.H0]["limitations"] += " Reference arm; no directional claim is assigned to its absolute score."
    by_id[run.H2]["limitations"] += " Positive transfer point estimate, but the simultaneous interval crosses zero."
    by_id[run.H3]["limitations"] += (
        " Positive transfer point estimate versus H0, but the simultaneous interval crosses zero; "
        "H3 also has an uncertain negative point delta versus H2."
    )
    atomic_write_json(vp, variants)

    ep = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(ep.read_text())
    matches = [row for row in experiments["experiments"] if row["experiment_id"] == run.EXPERIMENT]
    if len(matches) != 1:
        raise RuntimeError("missing H3 Llama experiment")
    matches[0].update(
        {
            "execution_status": "COMPLETE",
            "verdict": "POSITIVE_POINTS_UNCONFIRMED__H2_RAW_BEST__NO_FRESH_CONFIRMATION",
            "raw_best": run.H2,
            "survivors": [],
            "next_variant": None,
            "result_summary_sha256": sha256_file(SUMMARY),
        }
    )
    atomic_write_json(ep, experiments)


def integrate_metrics() -> None:
    panel_sha, cell_sha = sha256_file(PANELS), sha256_file(BY_CELL)
    additions: list[dict[str, Any]] = []
    for row in read_csv(PANELS):
        metric = row["metric_id"]
        additions.append(
            {
                "phase_id": "P2E", "experiment_id": run.EXPERIMENT,
                "variant_id": row["arm_id"], "task_id": "processbench_first_error",
                "dataset_id": "processbench", "population_id": "current_llama4_scorer_transfer",
                "cell_id": "aggregate", "slice_id": "all",
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "value": row["value"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "n_rows": row["n_rows"], "n_groups": row["n_groups"],
                "comparison_group_id": f"p2e_h3_llama4::{metric}", "status": "COMPLETE",
                "evidence_status": "TRANSFER", "display_order": DISPLAY[row["arm_id"]],
                "source_artifact": str(PANELS.relative_to(REPO)), "source_sha256": panel_sha,
                "source_row_selector": f"arm_id={row['arm_id']};metric_id={metric}",
                "source_value_field": "value",
                "notes": "Scorer-family transfer on Phase-1-opened source questions; not fresh confirmation.",
            }
        )
    for row in read_csv(BY_CELL):
        for metric in p1.PB_METRICS:
            additions.append(
                {
                    "phase_id": "P2E", "experiment_id": run.EXPERIMENT,
                    "variant_id": row["arm_id"], "task_id": "processbench_first_error",
                    "dataset_id": "processbench", "population_id": "current_llama4_scorer_transfer",
                    "cell_id": row["cell_id"], "slice_id": row["slice_id"],
                    "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                    "value": row[metric], "n_rows": row["n_examples"], "n_groups": row["n_examples"],
                    "comparison_group_id": f"p2e_h3_llama4::{metric}", "status": "COMPLETE",
                    "evidence_status": "TRANSFER", "display_order": DISPLAY[row["arm_id"]],
                    "source_artifact": str(BY_CELL.relative_to(REPO)), "source_sha256": cell_sha,
                    "source_row_selector": f"arm_id={row['arm_id']};cell_id={row['cell_id']}",
                    "source_value_field": metric,
                    "notes": "Cell result on previously opened ProcessBench questions.",
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
        metric = row["metric"]
        additions.append(
            {
                "phase_id": "P2E", "experiment_id": run.EXPERIMENT,
                "left_variant_id": row["left"], "right_variant_id": row["right"],
                "task_id": "processbench_first_error", "dataset_id": "processbench",
                "population_id": "current_llama4_scorer_transfer",
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "wins": row["wins"], "ties": row["ties"], "losses": row["losses"],
                "worst_unit_delta": row["worst_cell_delta"],
                "comparison_group_id": f"p2e_h3_llama4::{metric}", "status": "COMPLETE",
                "evidence_status": "TRANSFER", "source_artifact": str(CONTRASTS.relative_to(REPO)),
                "source_sha256": source_sha, "source_row_selector": f"contrast_id={row['contrast_id']}",
                "notes": row["interval"] + "; same opened questions, new scorer-family traces.",
            }
        )
    append_rows(
        p1.PROGRAM_ROOT / "CONTRASTS_LONG.csv",
        additions,
        ("experiment_id", "left_variant_id", "right_variant_id", "metric_id"),
    )


def integrate_gates(summary: dict[str, Any]) -> None:
    h2 = contrast(run.H2, run.H0, "official_macro_f1")
    h3 = contrast(run.H3, run.H0, "official_macro_f1")
    h3_h2 = contrast(run.H3, run.H2, "official_macro_f1")
    decisions = read_csv(EVAL / "DECISIONS.csv")
    group_folds: dict[tuple[str, str], set[str]] = {}
    for row in decisions:
        group_folds.setdefault((row["arm_id"], row["group_id"]), set()).add(row["fold"])
    max_group_folds = max(map(len, group_folds.values()))
    gates = [
        (run.H0, "P2E_QWEN_H0_ALIAS", "max_abs_error", summary["qwen_alias"]["h0_combined_max_abs_error"], 1e-12, "le", True, "Exact H0 combined-score reconstruction."),
        (run.H2, "P2E_QWEN_H2_ALIAS", "max_abs_error", summary["qwen_alias"]["h2_max_abs_error"], 1e-12, "le", True, "Exact H2 Qwen score alias before Llama label import."),
        (run.H3, "P2E_QWEN_H3_ALIAS", "max_abs_error", summary["qwen_alias"]["h3_max_abs_error"], 1e-12, "le", True, "Exact H3 Qwen score alias before Llama label import."),
        (run.H3, "P2E_ABSTENTION_ALIAS", "mismatches", sum(summary["abstention_mismatches"].values()), 0, "eq", True, "H2/H3 copy H0 clean/error decisions exactly."),
        (run.H3, "P2E_GROUP_FOLD", "max_fold_count", max_group_folds, 1, "le", max_group_folds <= 1, "Every source group stays inside one fold per arm."),
        (run.H2, "P2E_H2_DIRECTIONAL", "macro_f1_ci_low", float(h2["ci_low"]), 0.0, "gt", float(h2["ci_low"]) > 0.0, "H2 interval crosses zero; positive-improvement claim unsupported, not rejected."),
        (run.H2, "P2E_H2_PRACTICAL", "macro_f1_ci_low", float(h2["ci_low"]), 0.003, "gt", float(h2["ci_low"]) > 0.003, "H2 does not clear the practical-benefit bound."),
        (run.H3, "P2E_H3_DIRECTIONAL", "macro_f1_ci_low", float(h3["ci_low"]), 0.0, "gt", float(h3["ci_low"]) > 0.0, "H3 interval crosses zero; positive-improvement claim unsupported, not rejected."),
        (run.H3, "P2E_H3_PRACTICAL", "macro_f1_ci_low", float(h3["ci_low"]), 0.003, "gt", float(h3["ci_low"]) > 0.003, "H3 does not clear the practical-benefit bound."),
        (run.H3, "P2E_H3_WORST_CELL", "worst_cell_delta", float(h3["worst_cell_delta"]), -0.020, "ge", float(h3["worst_cell_delta"]) >= -0.020, "Worst Llama family is slightly beyond the robustness boundary, but above the -0.030 hard-stop bound."),
        (run.H3, "P2E_H3_BEATS_H2", "macro_f1_ci_low", float(h3_h2["ci_low"]), 0.0, "gt", float(h3_h2["ci_low"]) > 0.0, "H3 does not improve over H2; its negative point delta is uncertain."),
        (run.H3, "P2E_FRESH_CONFIRMATION", "fresh_population", False, True, "eq", False, "The exact questions and labels were already opened in Phase 1."),
    ]
    gate_source = EVAL / "GATES.csv"
    write_csv(
        gate_source,
        [
            {
                "gate_id": gate_id, "variant_id": variant, "metric_id": metric,
                "observed": str(observed).lower() if isinstance(observed, bool) else observed,
                "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
                "direction": direction, "passed": str(passed).lower(),
                "status": "PASS" if passed else "FAIL", "evidence_status": "TRANSFER",
                "notes": note,
            }
            for variant, gate_id, metric, observed, threshold, direction, passed, note in gates
        ],
        ["gate_id", "variant_id", "metric_id", "observed", "threshold", "direction", "passed", "status", "evidence_status", "notes"],
    )
    source_sha = sha256_file(gate_source)
    additions = [
        {
            "phase_id": "P2E", "experiment_id": run.EXPERIMENT, "variant_id": variant,
            "gate_id": gate_id, "metric_id": metric,
            "observed": str(observed).lower() if isinstance(observed, bool) else observed,
            "threshold": str(threshold).lower() if isinstance(threshold, bool) else threshold,
            "direction": direction, "passed": str(passed).lower(),
            "unit": "boolean" if isinstance(observed, bool) else "fraction",
            "status": "PASS" if passed else "FAIL", "evidence_status": "TRANSFER",
            "source_artifact": str(gate_source.relative_to(REPO)), "source_sha256": source_sha,
            "source_row_selector": f"gate_id={gate_id}", "source_value_field": "observed",
            "notes": note,
        }
        for variant, gate_id, metric, observed, threshold, direction, passed, note in gates
    ]
    append_rows(
        p1.PROGRAM_ROOT / "GATES_LONG.csv",
        additions,
        ("experiment_id", "variant_id", "gate_id"),
    )


def make_svg() -> None:
    panels = {
        row["arm_id"]: float(row["value"])
        for row in read_csv(PANELS) if row["metric_id"] == "official_macro_f1"
    }
    contrasts = [
        contrast(run.H2, run.H0, "official_macro_f1"),
        contrast(run.H3, run.H0, "official_macro_f1"),
        contrast(run.H3, run.H2, "official_macro_f1"),
    ]
    plot_rows = [
        {"kind": "absolute", "id": arm, "value": panels[arm], "ci_low": "", "ci_high": ""}
        for arm in run.ARMS
    ] + [
        {"kind": "contrast", "id": row["contrast_id"], "value": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]}
        for row in contrasts
    ]
    write_csv(PLOT_DATA, plot_rows, ["kind", "id", "value", "ci_low", "ci_high"])

    names = {run.H0: "H0 family6", run.H2: "H2 cleanup+C7", run.H3: "H3 equal+C8"}
    colors = {run.H0: "#8392a8", run.H2: "#147d79", run.H3: "#3156b8"}
    abs_min, abs_max = 0.34, 0.36
    x0, x1 = 205.0, 700.0
    def abs_x(value: float) -> float:
        return x0 + (value - abs_min) / (abs_max - abs_min) * (x1 - x0)
    delta_min, delta_max = -0.025, 0.025
    def delta_x(value: float) -> float:
        return x0 + (value - delta_min) / (delta_max - delta_min) * (x1 - x0)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="570" viewBox="0 0 900 570" role="img" aria-labelledby="title desc">',
        '<title id="title">H3 Llama scorer-family transfer</title>',
        '<desc id="desc">Absolute macro F1 and paired simultaneous confidence intervals for H2 and H3 versus their frozen parents.</desc>',
        '<rect width="900" height="570" fill="#fbfcfe"/>',
        '<style>text{font-family:Inter,system-ui,sans-serif;fill:#172f55}.title{font-size:24px;font-weight:800}.sub{font-size:13px;fill:#54657b}.label{font-size:13px;font-weight:700}.tick{font-size:11px;fill:#64748b}.value{font-size:12px;font-weight:700}.axis{stroke:#9aa8ba}.zero{stroke:#8b97a8;stroke-dasharray:5 4}.ci{stroke:#3156b8;stroke-width:4}.pt{fill:#3156b8}</style>',
        '<text class="title" x="42" y="42">Frozen H3 transfer to four Llama scorer cells</text>',
        '<text class="sub" x="42" y="66">Same 3,400 opened questions; transfer evidence, not fresh confirmation</text>',
        '<text class="label" x="42" y="108">Absolute ProcessBench macro F1</text>',
    ]
    for tick in (0.34, 0.35, 0.36):
        x = abs_x(tick)
        lines += [f'<line class="axis" x1="{x:.1f}" y1="122" x2="{x:.1f}" y2="270" opacity=".28"/>', f'<text class="tick" x="{x:.1f}" y="286" text-anchor="middle">{tick:.3f}</text>']
    for idx, arm in enumerate(run.ARMS):
        y = 150 + idx * 48
        x = abs_x(panels[arm])
        lines += [
            f'<text class="label" x="42" y="{y+5}">{html.escape(names[arm])}</text>',
            f'<rect x="{x0:.1f}" y="{y-13}" width="{max(x-x0,1):.1f}" height="24" rx="5" fill="{colors[arm]}"/>',
            f'<text class="value" x="{x+9:.1f}" y="{y+5}">{panels[arm]:.4f}</text>',
        ]
    lines += [
        '<text class="label" x="42" y="340">Paired macro-F1 deltas (Bonferroni-simultaneous 95% family)</text>',
        f'<line class="zero" x1="{delta_x(0):.1f}" y1="356" x2="{delta_x(0):.1f}" y2="510"/>',
    ]
    for tick in (-0.02, -0.01, 0.0, 0.01, 0.02):
        x = delta_x(tick)
        lines += [f'<line class="axis" x1="{x:.1f}" y1="502" x2="{x:.1f}" y2="508"/>', f'<text class="tick" x="{x:.1f}" y="526" text-anchor="middle">{tick:+.2f}</text>']
    labels = ["H2 - H0", "H3 - H0", "H3 - H2"]
    for idx, (label, row) in enumerate(zip(labels, contrasts)):
        y = 382 + idx * 52
        lo, point, hi = map(float, (row["ci_low"], row["delta"], row["ci_high"]))
        lines += [
            f'<text class="label" x="42" y="{y+5}">{label}</text>',
            f'<line class="ci" x1="{delta_x(lo):.1f}" y1="{y}" x2="{delta_x(hi):.1f}" y2="{y}"/>',
            f'<circle class="pt" cx="{delta_x(point):.1f}" cy="{y}" r="6"/>',
            f'<text class="value" x="720" y="{y+5}">{point:+.4f} [{lo:+.4f}, {hi:+.4f}]</text>',
        ]
    lines += [
        '<text class="sub" x="42" y="554">Intervals crossing zero are uncertain, not rejected. H2 is raw-best; H3 does not beat H2 on Llama.</text>',
        '</svg>',
    ]
    PLOT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_plot_claim_and_manifest() -> None:
    make_svg()
    pp = p1.PROGRAM_ROOT / "PLOT_MANIFEST.json"
    plots = json.loads(pp.read_text())
    plots["plots"].append(
        {
            "plot_id": "PLOT_P2E_H3_LLAMA_TRANSFER_FOREST",
            "title": "Frozen H3 transfer to four Llama scorer cells",
            "phase": "P2E", "kind": "contrast_forest", "source_table": "CONTRASTS_LONG.csv",
            "selection": {"experiment_id": run.EXPERIMENT, "metric_id": "macro_f1", "status": "COMPLETE"},
            "x_field": "delta", "y_field": "left_variant_id", "series_field": "right_variant_id",
            "comparison_group": "same 3,400 source questions, four Llama scorer cells, identical H0 abstention stream",
            "bootstrap_definition": "20,000 paired whole-source-question draws; Bonferroni simultaneous intervals across three macro-F1 contrasts.",
            "selection_rule": "All three frozen primary contrasts: H2-H0, H3-H0 and H3-H2.",
            "legend": ["Point and line = paired delta and simultaneous interval", "Crossing zero = unresolved, not rejection"],
            "caption": "H2 and H3 have positive transfer point estimates versus H0, but both intervals cross zero. H2 is raw-best; H3-minus-H2 is negative but uncertain.",
        }
    )
    atomic_write_json(pp, plots)

    cp = p1.PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(cp.read_text())
    claims["claims"].append(
        {
            "claim_id": "CLAIM_P2E_H3_LLAMA_TRANSFER",
            "text": "On the four Llama scorer cells, the frozen H2 and H3 rerankers retain positive macro-F1 point estimates versus H0, but neither has a simultaneous interval excluding zero; H2, not H3, is the raw-best arm.",
            "verdict": "PROMISING_UNCONFIRMED",
            "task_scope": "Four Llama scorer cells over the same 3,400 ProcessBench questions previously opened in Phase 1.",
            "evidence_refs": ["PLOT_P2E_H3_LLAMA_TRANSFER_FOREST", f"CONTRAST:{run.H3}:{run.H0}", "TABLE_GATES"],
            "worst_case_behavior": "H3 wins two and loses two families versus H0; its worst family delta is -0.02160. H2 also splits 2/2 but its worst delta is -0.01002.",
            "claim_boundary": "This is scorer-family transfer, not fresh-question confirmation. It supports continued interest but cannot promote H2/H3 or open Phase 3.",
            "fresh_confirmation_required": True,
            "statistical_summary": {
                "metric": "macro_f1_h3_vs_h0",
                "point_delta": 0.004371881377572884,
                "ci_low": -0.009677021874356245,
                "ci_high": 0.0184524462729739,
                "benefit_bound": 0.003,
                "harm_bound": -0.005,
                "bound_basis": "Existing Phase-3 parent practical-benefit semantics; transfer cannot promote regardless because questions were opened.",
                "multiplicity": "Bonferroni simultaneous across three frozen macro-F1 contrasts.",
            },
        }
    )
    atomic_write_json(cp, claims)

    artifacts = []
    for path in [
        run.REGISTRY, run.ROOT / "SCORE_FREEZE_MANIFEST.json", PANELS, BY_CELL,
        CONTRASTS, EVAL / "DECISIONS.csv", EVAL / "GATES.csv", SUMMARY, PLOT_DATA, PLOT,
    ]:
        artifacts.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
    atomic_write_json(
        ARTIFACT_MANIFEST,
        {
            "schema": "reasoning-localization-h3-llama-transfer-artifacts-v1",
            "status": "COMPLETE", "experiment_id": run.EXPERIMENT,
            "fresh_confirmation": False, "artifacts": artifacts,
            "plot_rule": "SVG generated deterministically from PANELS.csv and CONTRASTS.csv; no manual result entry.",
        },
    )


def main() -> None:
    summary = validate_result()
    update_registries()
    integrate_metrics()
    integrate_contrasts()
    integrate_gates(summary)
    register_plot_claim_and_manifest()
    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO)
    REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(
        json.dumps(
            {
                "experiment": run.EXPERIMENT,
                "verdict": "POSITIVE_POINTS_UNCONFIRMED__H2_RAW_BEST",
                "plot_sha256": sha256_file(PLOT),
                "report_sha256": build.manifest["output"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
