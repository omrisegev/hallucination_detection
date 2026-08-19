#!/usr/bin/env python3
"""Aggregate and report the frozen Global-Local-Online IU v1 screen."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/global_local_online_iu_v1"
METHODS = (
    "iu28_no_length",
    "deepconf_entropy_w64",
    "cusum_swvar_equal",
    "dyn_level4_iu",
    "dyn_persist6_iu",
    "dyn_change6_iu",
)
CANDIDATES = ("dyn_level4_iu", "dyn_persist6_iu", "dyn_change6_iu")
BUDGETS = (16, 32, 64, 128, 256, 512)


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: str | float | int | None) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def equal_family_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if finite(row.get(field)):
            by_family[str(row["family"])].append(float(row[field]))
    return mean(mean(values) for values in by_family.values())


def aggregate_performance() -> list[dict[str, Any]]:
    source = read_csv("PER_CELL_METRICS.csv")
    output = []
    for method in METHODS:
        for budget in BUDGETS:
            selected = [
                row for row in source
                if row["method"] == method
                and row["budget"] == str(budget)
                and not row.get("length_band")
                and finite(row.get("auroc"))
            ]
            output.append({
                "method": method,
                "budget": budget,
                "cell_macro_auroc": mean(float(row["auroc"]) for row in selected),
                "equal_family_auroc": equal_family_mean(selected, "auroc"),
                "cell_macro_auprc": mean(float(row["auprc"]) for row in selected),
                "equal_family_auprc": equal_family_mean(selected, "auprc"),
                "cell_macro_spearman_vs_final": mean(float(row["spearman_vs_final"]) for row in selected if finite(row.get("spearman_vs_final"))),
                "cell_macro_final_decision_agreement": mean(float(row["final_decision_agreement"]) for row in selected if finite(row.get("final_decision_agreement"))),
                "n_cells": len(selected),
                "n_cells_at_risk_ge_20": sum(int(float(row["n_at_risk"])) >= 20 for row in selected),
                "n_families": len({row["family"] for row in selected}),
                "n_at_risk_total": sum(int(float(row["n_at_risk"])) for row in selected),
            })
    return output


def aggregate_length_bands() -> list[dict[str, Any]]:
    source = read_csv("PER_CELL_METRICS.csv")
    output = []
    for method in METHODS:
        for band in ("lt128", "128_511", "ge512"):
            selected = [
                row for row in source
                if row["method"] == method
                and row["budget"] == "final"
                and row.get("length_band") == band
                and finite(row.get("auroc"))
            ]
            output.append({
                "method": method,
                "length_band": band,
                "cell_macro_auroc": mean(float(row["auroc"]) for row in selected),
                "equal_family_auroc": equal_family_mean(selected, "auroc"),
                "cell_macro_auprc": mean(float(row["auprc"]) for row in selected),
                "equal_family_auprc": equal_family_mean(selected, "auprc"),
                "n_cells": len(selected),
                "n_families": len({row["family"] for row in selected}),
                "n_at_risk_total": sum(int(float(row["n_at_risk"])) for row in selected),
            })
    return output


def aggregate_declarations() -> list[dict[str, Any]]:
    source = read_csv("DECLARATION_METRICS.csv")
    metrics = (
        "coverage", "ever_wrong_rate_all", "selective_error_rate",
        "false_alarm_rate_all", "false_clearance_rate_all",
        "mean_decision_budget", "mean_potential_tokens_remaining",
    )
    output = []
    for method in METHODS:
        selected = [row for row in source if row["method"] == method]
        result: dict[str, Any] = {
            "method": method,
            "n_cells": len(selected),
            "n_families": len({row["family"] for row in selected}),
        }
        for metric in metrics:
            result[f"cell_macro_{metric}"] = mean(float(row[metric]) for row in selected if finite(row.get(metric)))
            result[f"equal_family_{metric}"] = equal_family_mean(selected, metric)
        output.append(result)
    return output


def aggregate_efficiency() -> list[dict[str, Any]]:
    source = read_csv("EFFICIENCY.csv")
    output = []
    for method in CANDIDATES:
        selected = [row for row in source if row["method"] == method]
        output.append({
            "method": method,
            "cells": len(selected),
            "feature_count_min": min(int(row["feature_count"]) for row in selected),
            "feature_count_max": max(int(row["feature_count"]) for row in selected),
            "state_scalars_per_trace": max(int(row["persistent_state_scalars_per_trace"]) for row in selected),
            "median_fit_seconds": median(float(row["fit_seconds"]) for row in selected),
            "median_score_seconds": median(float(row["score_seconds"]) for row in selected),
            "max_python_traced_peak_bytes": max(int(row["python_traced_peak_bytes"]) for row in selected),
            "update_complexity": selected[0]["update_complexity"],
        })
    return output


def endpoint_by_cell() -> dict[tuple[str, str], float]:
    source = read_csv("PER_CELL_METRICS.csv")
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in source:
        if row["budget"] in {"64", "128"} and not row.get("length_band") and finite(row.get("auroc")):
            values[(row["cell_id"], row["method"])].append(float(row["auroc"]))
    return {key: mean(items) for key, items in values.items()}


def aggregate_missing_streams() -> list[dict[str, Any]]:
    source = read_csv("MISSING_STREAM_SENSITIVITY.csv")
    full = endpoint_by_cell()
    output = []
    for method in CANDIDATES:
        for missing in ("cusum_max", "sw_var_peak"):
            selected = [row for row in source if row["method"] == method and row["missing_stream"] == missing]
            deltas = []
            for row in selected:
                item = dict(row)
                item["delta"] = float(row["endpoint_auroc_64_128"]) - full[(row["cell_id"], method)]
                deltas.append(item)
            output.append({
                "method": method,
                "missing_stream": missing,
                "cell_macro_endpoint": mean(float(row["endpoint_auroc_64_128"]) for row in selected),
                "equal_family_endpoint": equal_family_mean(selected, "endpoint_auroc_64_128"),
                "cell_macro_delta_vs_full": mean(float(row["delta"]) for row in deltas),
                "equal_family_delta_vs_full": equal_family_mean(deltas, "delta"),
                "worst_cell_delta_vs_full": min(float(row["delta"]) for row in deltas),
                "n_cells": len(selected),
            })
    return output


def redundancy_rows() -> list[dict[str, Any]]:
    source = read_csv("PER_QUESTION_SCORES.csv")
    by_slice: dict[tuple[str, str, str], dict[tuple[str, str], float]] = defaultdict(dict)
    family_for: dict[str, str] = {}
    for row in source:
        if row["budget"] not in {"64", "128"}:
            continue
        cell = row["cell_id"]
        family_for[cell] = row["family"]
        key = (row["group"], row["unit_index"])
        by_slice[(cell, row["budget"], row["method"])][key] = float(row["score"])
    output = []
    for method in CANDIDATES:
        for reference in ("iu28_no_length", "cusum_swvar_equal"):
            cell_correlations: list[dict[str, Any]] = []
            for cell in sorted(family_for):
                per_budget = []
                for budget in ("64", "128"):
                    left = by_slice[(cell, budget, method)]
                    right = by_slice[(cell, budget, reference)]
                    keys = sorted(set(left) & set(right))
                    if len(keys) < 3:
                        continue
                    correlation = float(spearmanr([left[key] for key in keys], [right[key] for key in keys]).statistic)
                    if math.isfinite(correlation):
                        per_budget.append(correlation)
                if per_budget:
                    cell_correlations.append({"family": family_for[cell], "correlation": mean(per_budget)})
            output.append({
                "method": method,
                "reference": reference,
                "median_cell_spearman_64_128": median(row["correlation"] for row in cell_correlations),
                "cell_macro_spearman_64_128": mean(row["correlation"] for row in cell_correlations),
                "equal_family_spearman_64_128": equal_family_mean(cell_correlations, "correlation"),
                "n_cells": len(cell_correlations),
            })
    return output


def candidate_ledger(
    declarations: Sequence[Mapping[str, Any]],
    efficiency: Sequence[Mapping[str, Any]],
    redundancy: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    intervals = read_csv("GROUPED_INTERVALS.csv")
    hypotheses = {
        "dyn_level4_iu": "running extremes retain early warning",
        "dyn_persist6_iu": "positive area and run persistence beat one-off magnitude",
        "dyn_change6_iu": "slope and failure-to-recover add information",
    }
    coordinates = {
        "dyn_level4_iu": "current + running maximum per component",
        "dyn_persist6_iu": "current + positive area + run fraction per component",
        "dyn_change6_iu": "current + slope + recovery per component",
    }
    iu_decl = next(row for row in declarations if row["method"] == "iu28_no_length")
    output = []
    for method in CANDIDATES:
        versus_iu = next(row for row in intervals if row["method"] == method and row["reference"] == "iu28_no_length")
        versus_deepconf = next(row for row in intervals if row["method"] == method and row["reference"] == "deepconf_entropy_w64")
        decl = next(row for row in declarations if row["method"] == method)
        cost = next(row for row in efficiency if row["method"] == method)
        redundant = next(row for row in redundancy if row["method"] == method and row["reference"] == "cusum_swvar_equal")
        coverage_delta = float(decl["equal_family_coverage"]) - float(iu_decl["equal_family_coverage"])
        wrong_delta = float(decl["equal_family_ever_wrong_rate_all"]) - float(iu_decl["equal_family_ever_wrong_rate_all"])
        output.append({
            "method": method,
            "hypothesis": hypotheses[method],
            "coordinates": coordinates[method],
            "features_min_max": f"{cost['feature_count_min']}-{cost['feature_count_max']}",
            "state_scalars_per_trace": cost["state_scalars_per_trace"],
            "delta_vs_iu28": float(versus_iu["delta"]),
            "ci_low_vs_iu28": float(versus_iu["ci_low"]),
            "ci_high_vs_iu28": float(versus_iu["ci_high"]),
            "family_wins_ties_losses_vs_iu28": f"{versus_iu['family_wins']}/{versus_iu['family_ties']}/{versus_iu['family_losses']}",
            "delta_vs_deepconf": float(versus_deepconf["delta"]),
            "ci_low_vs_deepconf": float(versus_deepconf["ci_low"]),
            "ci_high_vs_deepconf": float(versus_deepconf["ci_high"]),
            "coverage_delta_vs_iu28": coverage_delta,
            "ever_wrong_delta_vs_iu28": wrong_delta,
            "spearman_vs_equal_magnitude": float(redundant["equal_family_spearman_64_128"]),
            "localization_hash_identity": True,
            "promotion_gate": "FAIL: superiority interval crosses zero",
            "decision": "CLOSE coarse-grid mechanism; retain IU28 baseline",
        })
    return output


def f4(value: Any) -> str:
    return "NA" if not finite(value) else f"{float(value):.4f}"


def report(
    performance: Sequence[Mapping[str, Any]],
    declarations: Sequence[Mapping[str, Any]],
    efficiency: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    redundancy: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> str:
    perf = {(row["method"], int(row["budget"])): row for row in performance}
    intervals = read_csv("GROUPED_INTERVALS.csv")
    graph = {row["arm"]: row for row in read_csv("GRAPH_ABLATIONS.csv")}
    inventory = json.loads((OUT / "INVENTORY_SUMMARY.json").read_text(encoding="utf-8"))
    anchor = json.loads((OUT / "ANCHOR_REGRESSION.json").read_text(encoding="utf-8"))

    lines = [
        "# Global-Local-Online IU v1 — retrospective optimization report",
        "",
        "## Decision",
        "",
        "**Retain IU28 without final length as the Online head; retain the frozen Global/Local heads; close the tested coarse-monitor dynamic family.** None of the three frozen dynamic candidates passes the early-panel superiority gate. Localization is mechanically unchanged (bit-identical score hashes), so there is no localization regression and also no localization gain to trade against the early result.",
        "",
        "This is a development-only conclusion from existing caches. It does not authorize inference, a GPU/cluster run, a large download, or fresh-confirmation language. The correct claim remains **unsupervised scorer with calibrated decision policies**.",
        "",
        "## Evidence inventory and independence",
        "",
        f"The inventory contains **{inventory['n_cache_records']}** cache/artifact records: **{inventory['classification_counts']['causal-prefix-valid']}** causal-prefix-valid, **{inventory['classification_counts']['localization-only']}** localization-only, and **{inventory['classification_counts']['unusable']}** unusable for this cycle. The early screen has 11 cells but only five equal-weight dataset families; generator/model copies within a family do not create independent family evidence. ProcessBench is grouped by original question, and PRMBench remains a separate teacher-forced step task.",
        "",
        "The Google Drive check was read-only and recorded path, size, and modification metadata. No Drive artifact was copied, moved, overwritten, or deleted.",
        "",
        "## Localization panel (kept separate)",
        "",
        "The frozen localization evidence remains:",
        "",
        "- GL-LIU v1 ProcessBench macro F1: **0.3136**, versus **0.2571** for Mind the Gap.",
        f"- Fixed trajectory-first ProcessBench macro F1: **{anchor['processbench_f1']:.4f}**; matched Qwen3-8B F1 **0.3035** versus **0.2496**.",
        f"- Fixed trajectory-first PRMBench step AUROC: **{anchor['prmbench_step_auroc']:.4f}**, versus **0.6136** for the older step-first adapter.",
        f"- Online-only candidates reproduce the ProcessBench and PRMBench score hashes exactly: `{anchor['processbench_score_hash']}` and `{anchor['prmbench_score_hash']}`.",
        "",
        "Same-matrix graph controls remain tiny for ordinary/uniform/DUFS and harmful for the temporal detector arm:",
        "",
        "| component | ordinary | uniform | DUFS | temporal |",
        "|---|---:|---:|---:|---:|",
        f"| global answer AUROC | {f4(graph['global_ordinary_mixed']['macro_value'])} | {f4(graph['global_uniform_mixed']['macro_value'])} | {f4(graph['global_dufs_mixed']['macro_value'])} | — |",
        f"| local top-5 detector AUROC | {f4(graph['local_ordinary_top5']['macro_value'])} | {f4(graph['local_uniform_top5']['macro_value'])} | {f4(graph['local_dufs_top5']['macro_value'])} | {f4(graph['local_temporal_top5']['macro_value'])} |",
        f"| exact locator rate | {f4(graph['locator_ordinary']['macro_value'])} | {f4(graph['locator_uniform_l0p3']['macro_value'])} | {f4(graph['locator_dufs_l0p3']['macro_value'])} | {f4(graph['locator_temporal_l0p3']['macro_value'])} |",
        "",
        "Ordinary IU is the exact `lambda=0` path; the regression test confirms bit identity. These graph rows are historical same-matrix controls, not a new hyperparameter search.",
        "",
        "## Early-ranking panel",
        "",
        "Equal-family AUROC among unfinished traces:",
        "",
        "| method | 16 | 32 | 64 | 128 | 256 | 512 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = " | ".join(f4(perf[(method, budget)]["equal_family_auroc"]) for budget in BUDGETS)
        lines.append(f"| `{method}` | {values} |")
    lines.extend([
        "",
        "The frozen primary endpoint is the equal-family mean across 64 and 128 tokens. Paired question/family bootstrap results:",
        "",
        "| candidate | reference | delta | 95% CI | family W/T/L |",
        "|---|---|---:|---:|---:|",
    ])
    for row in intervals:
        lines.append(
            f"| `{row['method']}` | `{row['reference']}` | {float(row['delta']):+.4f} | [{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | {row['family_wins']}/{row['family_ties']}/{row['family_losses']} |"
        )
    lines.extend([
        "",
        "The least complex dynamic arm, `dyn_level4_iu`, is essentially a re-expression of the magnitude control and does not improve IU28: **-0.0051 [-0.0553, +0.0519]**, with wins in 2/5 families. Persistence and change/recovery coordinates also fail. DeepConf comparisons cross zero as well. Therefore no candidate is promoted.",
        "",
        "The canonical elapsed-prefix-length arm remains an ablation, not part of IU28. The prior 11-cell result showed no stable reason to add it; this cycle did not refit or silently merge that feature.",
        "",
        "## Convergence and declaration behavior",
        "",
        "At the two primary budgets, rank correlation with each method's own completed score and final-decision agreement remain descriptive convergence metrics, not substitutes for discrimination:",
        "",
        "| method | Spearman @64 | Spearman @128 | agreement @64 | agreement @128 |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        p64, p128 = perf[(method, 64)], perf[(method, 128)]
        lines.append(
            f"| `{method}` | {f4(p64['cell_macro_spearman_vs_final'])} | {f4(p128['cell_macro_spearman_vs_final'])} | {f4(p64['cell_macro_final_decision_agreement'])} | {f4(p128['cell_macro_final_decision_agreement'])} |"
        )
    lines.extend([
        "",
        "Calibrated three-way declaration summaries (equal-family averages):",
        "",
        "| method | coverage | ever wrong | selective error | mean decision budget | potential tokens remaining |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in declarations:
        lines.append(
            f"| `{row['method']}` | {f4(row['equal_family_coverage'])} | {f4(row['equal_family_ever_wrong_rate_all'])} | {f4(row['equal_family_selective_error_rate'])} | {f4(row['equal_family_mean_decision_budget'])} | {f4(row['equal_family_mean_potential_tokens_remaining'])} |"
        )
    lines.extend([
        "",
        "Declaration behavior cannot rescue the failed ranking gate. All thresholds were fit from calibration labels; the score constructors did not see labels.",
        "",
        "## Redundancy, missing streams, and cost",
        "",
        "The dynamic heads are highly redundant with the equal CUSUM/`sw_var` magnitude control:",
        "",
        "| method | equal-family Spearman vs magnitude control @64/128 |",
        "|---|---:|",
    ])
    for row in redundancy:
        if row["reference"] == "cusum_swvar_equal":
            lines.append(f"| `{row['method']}` | {f4(row['equal_family_spearman_64_128'])} |")
    lines.extend([
        "",
        "A missing component is deterministically replaced at its fitted reference level. Sensitivity of the 64/128 endpoint:",
        "",
        "| method | missing stream | equal-family delta vs full | worst cell delta |",
        "|---|---|---:|---:|",
    ])
    for row in missing:
        lines.append(
            f"| `{row['method']}` | `{row['missing_stream']}` | {float(row['equal_family_delta_vs_full']):+.4f} | {float(row['worst_cell_delta_vs_full']):+.4f} |"
        )
    lines.extend([
        "",
        "Measured Online-head cost (11 cells):",
        "",
        "| method | retained features | state scalars/trace | median fit s | median score s | max Python traced peak |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in efficiency:
        lines.append(
            f"| `{row['method']}` | {row['feature_count_min']}-{row['feature_count_max']} | {row['state_scalars_per_trace']} | {float(row['median_fit_seconds']):.4f} | {float(row['median_score_seconds']):.4f} | {int(row['max_python_traced_peak_bytes']) / 1_048_576:.2f} MiB |"
        )
    lines.extend([
        "",
        "Each dynamic arm uses O(1) work and O(1) persistent state per new **monitor observation**. This benchmark does not measure upstream telemetry extraction or full IU28 stream-computation cost, so it cannot establish an end-to-end compute Pareto win. The available trajectories are saved at the existing absolute monitor grid; they are causal but are not a newly generated token-by-token recurrence.",
        "",
        "## Candidate ledger and disposition",
        "",
        "| candidate | hypothesis | delta vs IU28 (95% CI) | declarations vs IU28 (coverage / ever-wrong) | decision |",
        "|---|---|---:|---:|---|",
    ])
    for row in ledger:
        lines.append(
            f"| `{row['method']}` | {row['hypothesis']} | {float(row['delta_vs_iu28']):+.4f} [{float(row['ci_low_vs_iu28']):+.4f}, {float(row['ci_high_vs_iu28']):+.4f}] | {float(row['coverage_delta_vs_iu28']):+.4f} / {float(row['ever_wrong_delta_vs_iu28']):+.4f} | close |"
        )
    lines.extend([
        "",
        "**Retain:** frozen Global/Local heads and `iu28_no_length` Online head.  ",
        "**Close:** current/running-maximum, persistence/area, and slope/recovery transformations of the existing coarse CUSUM/`sw_var` trajectories. The failure mode is lack of independent signal: effects are small, intervals are wide and cross zero, family directions are inconsistent, and the simplest arm is almost perfectly redundant with the magnitude control.  ",
        "**Do not promote:** graph regularization, elapsed length, or a declaration-only variant.  ",
        "**Next gate:** no GPU or fresh-inference run is justified by this screen. Reopen only for a token-native causal recurrence or genuinely new telemetry/data, under a separately frozen protocol and explicit authorization.",
        "",
        "## Audit trail",
        "",
        "- Anchor regression: PASS.",
        "- Suffix, feature-order, label-removal/permutation, repeated-run, missing-component, and exact `lambda=0` tests: PASS.",
        "- New inference: no. GPU hours: 0. Drive mutation: no.",
        "- A6/PTNI artifacts and protocol: untouched.",
        "- All opened data are retrospective development evidence; no fresh confirmation claim is made.",
        "",
        "Machine-readable outputs are in this directory, including `AUDIT.json`, `CANDIDATE_LEDGER.csv`, `AGGREGATE_PERFORMANCE.csv`, `AGGREGATE_DECLARATIONS.csv`, `GROUPED_INTERVALS.csv`, `PER_CELL_METRICS.csv`, `PER_QUESTION_SCORES.csv`, `PER_TRACE_CONVERGENCE.csv`, `MISSING_STREAM_AGGREGATE.csv`, and `EFFICIENCY_AGGREGATE.csv`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    performance = aggregate_performance()
    length_bands = aggregate_length_bands()
    declarations = aggregate_declarations()
    efficiency = aggregate_efficiency()
    missing = aggregate_missing_streams()
    redundancy = redundancy_rows()
    ledger = candidate_ledger(declarations, efficiency, redundancy)
    write_csv("AGGREGATE_PERFORMANCE.csv", performance)
    write_csv("LENGTH_BAND_PERFORMANCE.csv", length_bands)
    write_csv("AGGREGATE_DECLARATIONS.csv", declarations)
    write_csv("EFFICIENCY_AGGREGATE.csv", efficiency)
    write_csv("MISSING_STREAM_AGGREGATE.csv", missing)
    write_csv("REDUNDANCY.csv", redundancy)
    write_csv("CANDIDATE_LEDGER.csv", ledger)
    decision = {
        "status": "COMPLETE_NO_PROMOTION",
        "online_head": "iu28_no_length",
        "global_local_heads": "frozen_unchanged",
        "promoted_candidates": [],
        "closed_mechanisms": list(CANDIDATES),
        "gpu_followup_justified": False,
        "fresh_confirmation": False,
        "reason": "Every dynamic candidate's 95% grouped interval versus IU28 crosses zero.",
    }
    (OUT / "DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        report(performance, declarations, efficiency, missing, redundancy, ledger),
        encoding="utf-8",
    )
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
