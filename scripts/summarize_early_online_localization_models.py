#!/usr/bin/env python3
"""Aggregate the causal GL-LIU online follow-up with equal-family contrasts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results" / "early_online_localization_models_v1"
METHODS = (
    "global_gl_liu_no_length",
    "global_gl_liu_elapsed_length",
    "local_temporal_gl_liu_max",
    "local_dufs_gl_liu_top5",
    "fused_gl_liu",
    "cusum_max",
    "sw_var_peak",
    "cusum_swvar_equal",
    "iu28_no_length",
    "deepconf_entropy_w32",
    "deepconf_entropy_w64",
)
NEW_METHODS = METHODS[:8]
BASELINES = ("deepconf_entropy_w64", "iu28_no_length")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(
    values: Sequence[float], repeats: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(int(repeats))
    ])
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def dataset_family(cell: str) -> str:
    if cell.startswith("processbench_"):
        return cell.split("__", 1)[0]
    if "math500" in cell:
        return "phase15_math500"
    return cell.split("__", 1)[0]


def load_cells(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/result.json"))
    }


def contrast_rows(
    cell_values: Mapping[str, Mapping[str, float]],
    *,
    endpoint: str,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for method in NEW_METHODS:
        for baseline in BASELINES:
            deltas = {
                cell: float(values[method] - values[baseline])
                for cell, values in cell_values.items()
                if method in values and baseline in values
            }
            if not deltas:
                continue
            families: dict[str, list[float]] = {}
            for cell, delta in deltas.items():
                families.setdefault(dataset_family(cell), []).append(delta)
            family_values = [float(np.mean(values)) for values in families.values()]
            low, high = bootstrap_interval(
                family_values,
                repeats,
                seed + sum(map(ord, endpoint + method + baseline)),
            )
            output.append({
                "endpoint": endpoint,
                "method": method,
                "baseline": baseline,
                "n_cells": int(len(deltas)),
                "n_dataset_families": int(len(family_values)),
                "cell_macro_delta_auroc": float(np.mean(list(deltas.values()))),
                "equal_family_delta_auroc": float(np.mean(family_values)),
                "family_ci_low": low,
                "family_ci_high": high,
                "family_wins": int(sum(value > 0 for value in family_values)),
                "family_ties": int(sum(value == 0 for value in family_values)),
                "family_losses": int(sum(value < 0 for value in family_values)),
            })
    return output


def aggregate(
    root: Path, *, repeats: int = 10_000, seed: int = 20260816
) -> dict[str, Any]:
    cells = load_cells(root)
    if not cells:
        raise SystemExit("no cell results found")

    convergence_long = []
    by_budget_cell: dict[int, dict[str, dict[str, float]]] = {}
    for cell, result in cells.items():
        for row in result["convergence"]:
            convergence_long.append({"cell": cell, **row})
            if row["auroc"] is not None:
                by_budget_cell.setdefault(int(row["budget"]), {}).setdefault(
                    cell, {}
                )[row["method"]] = float(row["auroc"])

    macro_convergence = []
    for budget in sorted(by_budget_cell):
        for method in METHODS:
            selected = [
                row for row in convergence_long
                if int(row["budget"]) == budget and row["method"] == method
                and row["auroc"] is not None
            ]
            if not selected:
                continue
            correlations = [
                row["spearman_vs_final"] for row in selected
                if row["spearman_vs_final"] is not None
            ]
            macro_convergence.append({
                "budget": budget,
                "method": method,
                "n_cells": int(len(selected)),
                "n_cells_at_risk_ge_20": int(sum(
                    int(row["n_at_risk"]) >= 20 for row in selected
                )),
                "macro_auroc": float(np.mean([row["auroc"] for row in selected])),
                "macro_spearman_vs_final": (
                    float(np.mean(correlations)) if correlations else None
                ),
                "macro_final_decision_agreement": float(np.mean([
                    row["final_decision_agreement"] for row in selected
                ])),
            })

    final_by_cell: dict[str, dict[str, float]] = {}
    final_performance = []
    for cell, result in cells.items():
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in result["per_trace_convergence"]:
            grouped.setdefault(row["method"], []).append(row)
        for method, rows in grouped.items():
            labels = np.asarray([row["label_error"] for row in rows], dtype=int)
            scores = np.asarray([row["final_score"] for row in rows], dtype=float)
            if len(np.unique(labels)) == 2:
                final_by_cell.setdefault(cell, {})[method] = float(
                    roc_auc_score(labels, scores)
                )
    for method in METHODS:
        values = [scores[method] for scores in final_by_cell.values() if method in scores]
        if values:
            final_performance.append({
                "method": method,
                "n_cells": int(len(values)),
                "macro_final_auroc": float(np.mean(values)),
            })

    contrasts = []
    for budget, cell_values in sorted(by_budget_cell.items()):
        contrasts.extend(contrast_rows(
            cell_values,
            endpoint=str(budget),
            repeats=repeats,
            seed=seed + budget,
        ))
    contrasts.extend(contrast_rows(
        final_by_cell,
        endpoint="final",
        repeats=repeats,
        seed=seed + 9999,
    ))

    declarations = []
    for method in METHODS:
        selected = [
            result["declarations"][method]["evaluation_summary"]
            for result in cells.values() if method in result["declarations"]
        ]
        if not selected:
            continue
        declarations.append({
            "method": method,
            "n_cells": int(len(selected)),
            "macro_coverage": float(np.mean([row["coverage"] for row in selected])),
            "macro_ever_wrong_rate": float(np.mean([
                row["ever_wrong_rate_all"] for row in selected
            ])),
            "cells_meeting_10pct_ever_wrong": int(sum(
                row["ever_wrong_rate_all"] <= 0.10 + 1e-12 for row in selected
            )),
            "macro_selective_error": float(np.nanmean([
                np.nan if row["selective_error_rate"] is None
                else row["selective_error_rate"] for row in selected
            ])),
        })

    def contrast(method: str, baseline: str, endpoint: str) -> Mapping[str, Any]:
        return next(row for row in contrasts if row["method"] == method
                    and row["baseline"] == baseline and row["endpoint"] == endpoint)

    global_64 = contrast("global_gl_liu_no_length", "deepconf_entropy_w64", "64")
    fused_64 = contrast("fused_gl_liu", "deepconf_entropy_w64", "64")
    fused_vs_iu_64 = contrast("fused_gl_liu", "iu28_no_length", "64")
    global_final = contrast(
        "global_gl_liu_no_length", "deepconf_entropy_w64", "final"
    )
    fused_vs_global = {
        str(budget): float(
            next(row["macro_auroc"] for row in macro_convergence
                 if row["method"] == "fused_gl_liu" and row["budget"] == budget)
            - next(row["macro_auroc"] for row in macro_convergence
                   if row["method"] == "global_gl_liu_no_length"
                   and row["budget"] == budget)
        )
        for budget in (64, 128, 256, 512)
        if any(row["method"] == "fused_gl_liu" and row["budget"] == budget
               for row in macro_convergence)
    }
    decision = {
        "promising_parity_continues": True,
        "retrospective_superiority_established": bool(
            fused_64["family_ci_low"] > 0 or global_64["family_ci_low"] > 0
        ),
        "global_head_improves_final_over_deepconf_w64": bool(
            global_final["equal_family_delta_auroc"] > 0
        ),
        "local_fusion_improves_over_iu28_at_64": bool(
            fused_vs_iu_64["equal_family_delta_auroc"] > 0
            and fused_vs_iu_64["family_ci_low"] > 0
        ),
        "fused_minus_global_cell_macro": fused_vs_global,
        "summary": (
            "The causal global GL-LIU head is competitive and stronger on completed "
            "traces, but the selected local head does not materially improve 64–128 "
            "token detection when fused. CUSUM/sw_var remain strong individual "
            "mechanisms. No early equal-family interval establishes superiority."
        ),
    }
    return {
        "protocol": "EARLY_ONLINE_LOCALIZATION_MODELS_V1",
        "n_cells": int(len(cells)),
        "n_dataset_families": int(len({dataset_family(cell) for cell in cells})),
        "cell_ids": sorted(cells),
        "macro_convergence": macro_convergence,
        "final_performance": final_performance,
        "contrasts": contrasts,
        "declarations": declarations,
        "decision": decision,
        "bootstrap_repeats": int(repeats),
        "new_inference": False,
        "gpu_hours": 0,
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def report(summary: Mapping[str, Any]) -> str:
    macro = {(row["method"], row["budget"]): row
             for row in summary["macro_convergence"]}
    convergence_rows = []
    shown = (
        "global_gl_liu_no_length", "local_temporal_gl_liu_max",
        "local_dufs_gl_liu_top5", "fused_gl_liu", "cusum_max",
        "sw_var_peak", "cusum_swvar_equal", "iu28_no_length",
        "deepconf_entropy_w64",
    )
    for budget in (16, 32, 64, 128, 256, 512):
        for method in shown:
            row = macro.get((method, budget))
            if row:
                convergence_rows.append(
                    f"| {budget} | {method} | {row['n_cells']} | "
                    f"{row['n_cells_at_risk_ge_20']} | {fmt(row['macro_auroc'])} | "
                    f"{fmt(row['macro_spearman_vs_final'])} |"
                )
    final_rows = [
        f"| {row['method']} | {row['n_cells']} | {fmt(row['macro_final_auroc'])} |"
        for row in summary["final_performance"]
    ]
    contrast_rows = []
    for endpoint in ("64", "128", "512", "final"):
        for method in (
            "global_gl_liu_no_length", "local_temporal_gl_liu_max",
            "local_dufs_gl_liu_top5", "fused_gl_liu", "sw_var_peak",
            "cusum_swvar_equal",
        ):
            row = next((item for item in summary["contrasts"]
                        if item["endpoint"] == endpoint
                        and item["method"] == method
                        and item["baseline"] == "deepconf_entropy_w64"), None)
            if row:
                contrast_rows.append(
                    f"| {endpoint} | {method} | "
                    f"{fmt(row['equal_family_delta_auroc'])} | "
                    f"[{fmt(row['family_ci_low'])}, {fmt(row['family_ci_high'])}] | "
                    f"{row['family_wins']}/{row['family_ties']}/{row['family_losses']} |"
                )
    iu_rows = []
    for endpoint in ("64", "128", "512", "final"):
        for method in ("global_gl_liu_no_length", "fused_gl_liu",
                       "sw_var_peak", "cusum_swvar_equal"):
            row = next((item for item in summary["contrasts"]
                        if item["endpoint"] == endpoint
                        and item["method"] == method
                        and item["baseline"] == "iu28_no_length"), None)
            if row:
                iu_rows.append(
                    f"| {endpoint} | {method} | "
                    f"{fmt(row['equal_family_delta_auroc'])} | "
                    f"[{fmt(row['family_ci_low'])}, {fmt(row['family_ci_high'])}] |"
                )
    declaration_rows = [
        f"| {row['method']} | {fmt(row['macro_coverage'])} | "
        f"{fmt(row['macro_ever_wrong_rate'])} | "
        f"{row['cells_meeting_10pct_ever_wrong']}/{row['n_cells']} | "
        f"{fmt(row['macro_selective_error'])} |"
        for row in summary["declarations"] if row["method"] in shown
    ]
    return f"""# Aggregate causal localization-model online result

CPU-only retrospective replay over **{summary['n_cells']}** cells and
**{summary['n_dataset_families']}** equal-weight dataset families. No inference,
GPU job, Drive mutation, or raw-data mutation was performed.

## Completed-trace performance

| method | cells | macro final AUROC |
|---|---:|---:|
{chr(10).join(final_rows)}

## Fixed-budget performance

| budget | method | cells | cells with ≥20 at risk | macro AUROC | Spearman vs final |
|---:|---|---:|---:|---:|---:|
{chr(10).join(convergence_rows)}

## Equal-family delta versus DeepConf-w64

| endpoint | method | delta AUROC | family-bootstrap 95% interval | family W/T/L |
|---:|---|---:|---:|---:|
{chr(10).join(contrast_rows)}

## Equal-family delta versus IU28

| endpoint | method | delta AUROC | family-bootstrap 95% interval |
|---:|---|---:|---:|
{chr(10).join(iu_rows)}

## Held-out early declaration

| method | coverage | ever wrong | cells ≤10% ever wrong | selective error |
|---|---:|---:|---:|---:|
{chr(10).join(declaration_rows)}

## Interpretation

**Promising parity remains, but the localization head does not produce the
hoped-for early jump.** The causal global GL-LIU detector is competitive with
DeepConf and improves completed-trace scoring. The local temporal/DUFS heads
are substantially weaker as answer-level detectors, and equal-weight fusion
does not materially improve the 64–128 token result over the global head or
IU28. `sw_var_peak` and the fixed CUSUM/sw-var combination are the strongest
mechanism-level findings, including the best completed-trace macro AUROC.

No early equal-family 95% interval is wholly above zero, and the 10% held-out
declaration constraint still transfers inconsistently. This does not close the
comparison; it narrows the next step to a better causal global aggregation or
calibrated dynamic model rather than reusing the localization locator as-is.
"""


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    summary = aggregate(args.root, repeats=args.bootstrap, seed=args.seed)
    write_json(args.root / "AGGREGATE.json", summary)
    write_csv(args.root / "AGGREGATE_CONVERGENCE.csv", summary["macro_convergence"])
    write_csv(args.root / "AGGREGATE_FINAL_PERFORMANCE.csv", summary["final_performance"])
    write_csv(args.root / "AGGREGATE_CONTRASTS.csv", summary["contrasts"])
    write_csv(args.root / "AGGREGATE_DECLARATIONS.csv", summary["declarations"])
    (args.root / "REPORT.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps(summary["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
