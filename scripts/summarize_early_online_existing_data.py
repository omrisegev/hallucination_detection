#!/usr/bin/env python3
"""Build the equal-cell aggregate for EARLY_ONLINE_EXISTING_DATA_V1."""

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
DEFAULT_ROOT = REPO / "results" / "early_online_existing_data_v1"


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


def cell_bootstrap(values: Sequence[float], repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(int(repeats))
    ])
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def load_cells(root: Path) -> list[tuple[str, dict[str, Any]]]:
    output = []
    for path in sorted(root.glob("*/result.json")):
        output.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    return output


def dataset_family(cell: str) -> str:
    """Conservative family unit for the cross-cell robustness interval."""
    if cell.startswith("processbench_"):
        return cell.split("__", 1)[0]
    if "math500" in cell:
        return "phase15_math500"
    return cell.split("__", 1)[0]


def aggregate(root: Path, repeats: int = 10000, seed: int = 20260816) -> dict[str, Any]:
    cells = load_cells(root)
    if not cells:
        raise SystemExit("no cell result.json files found")
    convergence_rows = []
    for cell, result in cells:
        for row in result["convergence"]:
            convergence_rows.append({"cell": cell, **row})

    final_rows = []
    for cell, result in cells:
        by_method = {}
        for row in result["per_trace_convergence"]:
            by_method.setdefault(row["method"], []).append(row)
        for method, rows in by_method.items():
            labels = np.asarray([row["label_error"] for row in rows], dtype=int)
            scores = np.asarray([row["final_score"] for row in rows], dtype=float)
            final_rows.append({
                "cell": cell,
                "method": method,
                "n": int(len(rows)),
                "final_auroc": (
                    float(roc_auc_score(labels, scores))
                    if len(np.unique(labels)) == 2 else None
                ),
            })
    final_macro = []
    for method in sorted({row["method"] for row in final_rows}):
        selected = [
            row["final_auroc"] for row in final_rows
            if row["method"] == method and row["final_auroc"] is not None
        ]
        final_macro.append({
            "method": method,
            "n_cells": int(len(selected)),
            "macro_final_auroc": float(np.mean(selected)),
        })

    macro = []
    keys = sorted({(row["method"], row["budget"]) for row in convergence_rows})
    for method, budget in keys:
        selected = [
            row for row in convergence_rows
            if row["method"] == method and row["budget"] == budget
            and row["auroc"] is not None
        ]
        if not selected:
            continue
        macro.append({
            "method": method,
            "budget": int(budget),
            "n_cells": int(len(selected)),
            "n_cells_at_risk_ge_20": int(sum(row["n_at_risk"] >= 20 for row in selected)),
            "mean_n_at_risk": float(np.mean([row["n_at_risk"] for row in selected])),
            "macro_auroc": float(np.mean([row["auroc"] for row in selected])),
            "macro_spearman_vs_final": float(np.mean([
                row["spearman_vs_final"] for row in selected
                if row["spearman_vs_final"] is not None
            ])),
            "macro_final_decision_agreement": float(np.mean([
                row["final_decision_agreement"] for row in selected
            ])),
            "macro_auc_recovery": float(np.mean([
                row["above_chance_auc_recovery"] for row in selected
                if row["above_chance_auc_recovery"] is not None
            ])),
        })

    contrasts = []
    for budget in sorted({row["budget"] for row in convergence_rows}):
        by_cell = {}
        for row in convergence_rows:
            if row["budget"] == budget and row["auroc"] is not None:
                by_cell.setdefault(row["cell"], {})[row["method"]] = row
        for method, baseline in (
            ("iu28_no_length", "deepconf_entropy_w64"),
            ("iu29_elapsed_length", "iu28_no_length"),
        ):
            deltas, supported, cell_deltas = [], [], {}
            for cell, values in by_cell.items():
                if method not in values or baseline not in values:
                    continue
                delta = float(values[method]["auroc"] - values[baseline]["auroc"])
                deltas.append(delta)
                cell_deltas[cell] = delta
                if min(values[method]["n_at_risk"], values[baseline]["n_at_risk"]) >= 20:
                    supported.append(delta)
            if not deltas:
                continue
            low, high = cell_bootstrap(deltas, repeats, seed + int(budget))
            family_values = {}
            for cell, delta in cell_deltas.items():
                family_values.setdefault(dataset_family(cell), []).append(delta)
            family_means = [float(np.mean(values)) for values in family_values.values()]
            family_low, family_high = cell_bootstrap(
                family_means, repeats, seed + 1000 + int(budget)
            )
            contrasts.append({
                "method": method,
                "baseline": baseline,
                "budget": int(budget),
                "n_cells": int(len(deltas)),
                "n_cells_at_risk_ge_20": int(len(supported)),
                "mean_delta_auroc": float(np.mean(deltas)),
                "cell_bootstrap_ci_low": low,
                "cell_bootstrap_ci_high": high,
                "n_dataset_families": int(len(family_means)),
                "equal_family_mean_delta_auroc": float(np.mean(family_means)),
                "family_bootstrap_ci_low": family_low,
                "family_bootstrap_ci_high": family_high,
                "family_wins": int(sum(value > 0 for value in family_means)),
                "family_ties": int(sum(value == 0 for value in family_means)),
                "family_losses": int(sum(value < 0 for value in family_means)),
                "wins": int(sum(value > 0 for value in deltas)),
                "ties": int(sum(value == 0 for value in deltas)),
                "losses": int(sum(value < 0 for value in deltas)),
                "supported_mean_delta_auroc": (
                    float(np.mean(supported)) if supported else None
                ),
            })

    declarations = []
    methods = sorted({
        method for _, result in cells for method in result["declarations"]
    })
    for method in methods:
        selected = [
            result["declarations"][method]["evaluation_summary"]
            for _, result in cells if method in result["declarations"]
        ]
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
            "macro_potential_tokens_remaining": float(np.mean([
                row["mean_potential_tokens_remaining"] for row in selected
            ])),
        })

    primary = [
        row for row in contrasts
        if row["method"] == "iu28_no_length"
        and row["baseline"] == "deepconf_entropy_w64"
    ]
    positive_excluding_zero = [
        row for row in primary
        if row["equal_family_mean_delta_auroc"] > 0
        and row["family_bootstrap_ci_low"] > 0
    ]
    declaration_primary = next(
        row for row in declarations if row["method"] == "iu28_no_length"
    )
    gate = {
        "exact_paper_reproduction_authorized": False,
        "positive_paired_budget_with_ci_excluding_zero": bool(positive_excluding_zero),
        "positive_budgets": [row["budget"] for row in positive_excluding_zero],
        "declaration_10pct_transfer_cells": (
            f"{declaration_primary['cells_meeting_10pct_ever_wrong']}/"
            f"{declaration_primary['n_cells']}"
        ),
        "decision": "FAIL_EXISTING_DATA_PROMOTION_GATE",
        "reason": (
            "IU28 has no budget whose equal-dataset-family paired advantage over "
            "the same-access DeepConf entropy proxy has a 95% interval above zero; "
            "the calibration-constrained declaration policy also fails the 10% "
            "held-out ever-wrong target in multiple cells."
        ),
    }
    return {
        "protocol": "EARLY_ONLINE_EXISTING_DATA_V1",
        "n_cells": int(len(cells)),
        "cell_ids": [cell for cell, _ in cells],
        "macro_convergence": macro,
        "final_performance": final_macro,
        "cell_macro_contrasts": contrasts,
        "declaration_macro": declarations,
        "gate": gate,
        "bootstrap_repeats": int(repeats),
        "new_inference": False,
        "gpu_hours": 0,
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def report(summary: Mapping[str, Any]) -> str:
    macro_lookup = {
        (row["method"], row["budget"]): row for row in summary["macro_convergence"]
    }
    convergence = []
    for budget in (16, 32, 64, 128, 256, 512):
        for method in ("iu28_no_length", "iu29_elapsed_length", "deepconf_entropy_w64"):
            row = macro_lookup.get((method, budget))
            if row:
                convergence.append(
                    f"| {budget} | {method} | {row['n_cells']} | "
                    f"{row['n_cells_at_risk_ge_20']} | {fmt(row['macro_auroc'])} | "
                    f"{fmt(row['macro_spearman_vs_final'])} | "
                    f"{fmt(row['macro_final_decision_agreement'])} |"
                )
    contrasts = [
        row for row in summary["cell_macro_contrasts"]
        if row["method"] == "iu28_no_length"
        and row["baseline"] == "deepconf_entropy_w64"
    ]
    contrast_rows = [
        f"| {row['budget']} | {row['n_cells']} | {row['n_dataset_families']} | "
        f"{fmt(row['mean_delta_auroc'])} | "
        f"{fmt(row['equal_family_mean_delta_auroc'])} | "
        f"[{fmt(row['family_bootstrap_ci_low'])}, {fmt(row['family_bootstrap_ci_high'])}] | "
        f"{row['family_wins']}/{row['family_ties']}/{row['family_losses']} |"
        for row in contrasts
    ]
    declaration_rows = [
        f"| {row['method']} | {fmt(row['macro_coverage'])} | "
        f"{fmt(row['macro_ever_wrong_rate'])} | "
        f"{row['cells_meeting_10pct_ever_wrong']}/{row['n_cells']} | "
        f"{fmt(row['macro_selective_error'])} |"
        for row in summary["declaration_macro"]
    ]
    final_rows = [
        f"| {row['method']} | {row['n_cells']} | {fmt(row['macro_final_auroc'])} |"
        for row in summary["final_performance"]
    ]
    return f"""# Aggregate existing-cache early/online result

CPU-only retrospective screen over **{summary['n_cells']}** materialized
dataset×model/generator cells. Cells are weighted equally. No inference, GPU
job, Drive mutation, or raw-data mutation was performed.

## Score convergence

Completed-trace ranking first:

| method | cells | macro final AUROC |
|---|---:|---:|
{chr(10).join(final_rows)}

| budget | method | cells | cells with ≥20 at risk | macro AUROC | macro Spearman vs final | decision agreement |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(convergence)}

The score does converge with generation: correlation with the completed score
and final-decision agreement generally rise by 64–128 tokens. But convergence
is not the same as superiority over a simple same-access control.

## IU28 versus DeepConf entropy proxy

| budget | cells | dataset families | cell-macro delta | equal-family delta | family-bootstrap 95% interval | family W/T/L |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(contrast_rows)}

## Frozen early-declaration transfer

| method | macro coverage | macro ever-wrong | cells ≤10% ever-wrong | selective error |
|---|---:|---:|---:|---:|
{chr(10).join(declaration_rows)}

The 10% constraint was imposed only on calibration questions. Failure on a
held-out half is therefore a real transfer failure, not a threshold that may be
retuned after seeing evaluation labels.

## Decision

**{summary['gate']['decision']}.** {summary['gate']['reason']}

The evidence supports the scientific question — causal scores become more like
their final values over time — but does not currently support promoting the
frozen 28/29-stream maximum-risk adapter as a better online detector. Exact
native-paper reproductions or new GPU inference are therefore not authorized
by this gate. The next CPU-only diagnosis should separate the weak early score
into aggregation (maximum token risk), population fit, and feature-family
components before considering new data collection.
"""


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    summary = aggregate(root)
    write_json(root / "AGGREGATE.json", summary)
    write_csv(root / "AGGREGATE_FINAL_PERFORMANCE.csv", summary["final_performance"])
    write_csv(root / "AGGREGATE_CONVERGENCE.csv", summary["macro_convergence"])
    write_csv(root / "AGGREGATE_CONTRASTS.csv", summary["cell_macro_contrasts"])
    write_csv(root / "AGGREGATE_DECLARATIONS.csv", summary["declaration_macro"])
    (root / "REPORT.md").write_text(report(summary), encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
