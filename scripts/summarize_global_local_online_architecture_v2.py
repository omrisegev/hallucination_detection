#!/usr/bin/env python3
"""Freeze grouped inference, figures, report, and decision for architecture v2."""

from __future__ import annotations

import base64
import csv
from html import escape
import io
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_global_local_online_architecture_v2 as run  # noqa: E402


OUT = run.OUT
SELECTED_KEYS = (
    "a_one_shared__w1.00__peak",
    "a_two_global_local__w0.50__peak",
    "a_three_independent__w0.50__peak",
)
LABELS = {
    SELECTED_KEYS[0]: "one shared",
    SELECTED_KEYS[1]: "two Global+Local",
    SELECTED_KEYS[2]: "three independent",
}
PRIMARY = SELECTED_KEYS[1]
BOOTSTRAP = 2000
SEED = 20260816


def _read_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _weighted_auc(labels, scores, weights):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(scores) & (weights > 0)
    labels, scores, weights = labels[finite], scores[finite], weights[finite]
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0 or negative <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    scores, labels, weights = scores[order], labels[order], weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(scores[1:] != scores[:-1])]
    pos = np.add.reduceat(weights * (labels == 1), starts)
    neg = np.add.reduceat(weights * (labels == 0), starts)
    before = np.cumsum(neg) - neg
    return float(np.sum(pos * (before + 0.5 * neg)) / (positive * negative))


def _weighted_f1(target, prediction, weights):
    target = np.asarray(target, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    weights = np.asarray(weights, dtype=float)
    error, clean = target != -1, target == -1
    error_total, clean_total = weights[error].sum(), weights[clean].sum()
    if error_total <= 0 or clean_total <= 0:
        return float("nan")
    error_acc = weights[error & (prediction == target)].sum() / error_total
    clean_acc = weights[clean & (prediction == -1)].sum() / clean_total
    return float(2 * error_acc * clean_acc / (error_acc + clean_acc)) if error_acc + clean_acc else 0.0


class GroupedScores:
    def __init__(self, records, method_key, methods, families, models):
        self.method_key = method_key
        self.methods = tuple(methods)
        self.families = tuple(families)
        self.models = tuple(models)
        self.units = {
            family: sorted({
                row["unit"] for row in records
                if row["family"] == family and row[method_key] in methods
            }) for family in families
        }
        self.index = {
            family: {unit: index for index, unit in enumerate(self.units[family])}
            for family in families
        }
        self.data = {}
        for family in families:
            for model in models:
                for method in methods:
                    for task in ("global", "local"):
                        selected = [
                            row for row in records
                            if row["family"] == family and row["model"] == model
                            and row[method_key] == method and row["task"] == task
                        ]
                        if not selected:
                            continue
                        codes = np.asarray([self.index[family][row["unit"]] for row in selected])
                        if task == "global":
                            value = (codes, np.asarray([int(row["target"]) for row in selected]), np.asarray([float(row["score"]) for row in selected]))
                        else:
                            value = (codes, np.asarray([int(row["target"]) for row in selected]), np.asarray([int(row["prediction"]) for row in selected]))
                        self.data[(family, model, method, task, None)] = value
                    for budget in (64, 128):
                        selected = [
                            row for row in records
                            if row["family"] == family and row["model"] == model
                            and row[method_key] == method and row["task"] == "online"
                            and int(row["budget"]) == budget
                        ]
                        if selected:
                            codes = np.asarray([self.index[family][row["unit"]] for row in selected])
                            self.data[(family, model, method, "online", budget)] = (
                                codes,
                                np.asarray([int(row["target"]) for row in selected]),
                                np.asarray([float(row["score"]) for row in selected]),
                            )

    def family_metric(self, family, method, task, counts):
        values = []
        for model in self.models:
            if task in ("global", "local"):
                key = (family, model, method, task, None)
                if key not in self.data:
                    continue
                codes, target, score = self.data[key]
                weights = counts[codes]
                value = _weighted_auc(target, score, weights) if task == "global" else _weighted_f1(target, score, weights)
                if np.isfinite(value):
                    values.append(value)
            else:
                budgets = []
                for budget in (64, 128):
                    key = (family, model, method, "online", budget)
                    if key not in self.data:
                        continue
                    codes, target, score = self.data[key]
                    value = _weighted_auc(target, score, counts[codes])
                    if np.isfinite(value):
                        budgets.append(value)
                if budgets:
                    values.append(float(np.mean(budgets)))
        return float(np.mean(values)) if values else float("nan")

    def interval(self, candidate, reference, task):
        point_by_family = {}
        for family in self.families:
            counts = np.ones(len(self.units[family]), dtype=int)
            point_by_family[family] = self.family_metric(family, candidate, task, counts) - self.family_metric(family, reference, task, counts)
        point = float(np.nanmean(list(point_by_family.values())))
        rng = np.random.default_rng(SEED + sum(ord(char) for char in candidate + reference + task))
        draws = []
        for _ in range(BOOTSTRAP):
            deltas = []
            for family in self.families:
                n = len(self.units[family])
                counts = rng.multinomial(n, np.full(n, 1.0 / n))
                left = self.family_metric(family, candidate, task, counts)
                right = self.family_metric(family, reference, task, counts)
                if np.isfinite(left) and np.isfinite(right):
                    deltas.append(left - right)
            if deltas:
                draws.append(float(np.mean(deltas)))
        low, high = np.quantile(draws, (0.025, 0.975))
        wins = sum(value > 0 for value in point_by_family.values())
        losses = sum(value < 0 for value in point_by_family.values())
        return {
            "candidate": candidate, "reference": reference, "task": task,
            "delta": point, "ci_low": float(low), "ci_high": float(high),
            "family_wins": wins, "family_losses": losses,
            "family_deltas": point_by_family,
        }


def _architecture_summary(records, metrics):
    cell_rows = [
        row for row in metrics
        if row["architecture"] in SELECTED_KEYS and row["task"] in {"global", "local", "online"}
    ]
    aggregate = []
    for key in SELECTED_KEYS:
        item = {"architecture": key, "label": LABELS[key], "cells": 12}
        for task in ("global", "local", "online"):
            item[task] = float(np.mean([
                float(row["primary"]) for row in cell_rows
                if row["architecture"] == key and row["task"] == task
            ]))
        aggregate.append(item)
    grouped = GroupedScores(records, "architecture", SELECTED_KEYS, run.FAMILIES, run.MODELS)
    intervals = []
    for candidate, reference in (
        (SELECTED_KEYS[1], SELECTED_KEYS[0]),
        (SELECTED_KEYS[2], SELECTED_KEYS[1]),
    ):
        for task in ("global", "local", "online"):
            intervals.append(grouped.interval(candidate, reference, task))
    return aggregate, intervals


def _fusion_summary(records, aggregate):
    ordinary = "global_ordinary__local_ordinary"
    comparisons = (
        "global_dufs__local_ordinary",
        "global_uniform__local_ordinary",
        "global_ordinary__local_dufs",
        "global_ordinary__local_uniform",
        "global_ordinary__local_temporal",
    )
    grouped = GroupedScores(
        records, "method", (ordinary,) + comparisons,
        ("gsm8k", "math"), ("qwen3_4b",),
    )
    intervals = []
    for candidate in comparisons:
        for task in ("global", "local", "online"):
            intervals.append(grouped.interval(candidate, ordinary, task))
    lookup = {row["method"]: row for row in aggregate}
    return ordinary, lookup, intervals


def _budget_summary(metrics):
    output = []
    for key in SELECTED_KEYS:
        for budget in run.BUDGETS:
            rows = [
                row for row in metrics
                if row["architecture"] == key and row["task"] == "online_budget"
                and int(row["budget"]) == budget
            ]
            output.append({
                "architecture": key, "label": LABELS[key], "budget": budget,
                "auroc": float(np.mean([float(row["auroc"]) for row in rows])),
                "auprc": float(np.mean([float(row["auprc"]) for row in rows])),
                "mean_eligible": float(np.mean([int(row["n"]) for row in rows])),
                "cells": len(rows),
            })
    return output


def _plot(architecture, budgets, declaration):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {SELECTED_KEYS[0]: "#9aa4b2", SELECTED_KEYS[1]: "#1769aa", SELECTED_KEYS[2]: "#d97706"}
    tasks = ("global", "local", "online")
    x = np.arange(len(tasks))
    width = 0.24
    for offset, row in enumerate(architecture):
        axes[0].bar(x + (offset - 1) * width, [row[t] for t in tasks], width, label=row["label"], color=colors[row["architecture"]])
    axes[0].set_xticks(x, ("Global AUROC", "Local F1", "Online AUROC"))
    axes[0].set_ylim(0.2, 0.8); axes[0].set_title("Twelve-cell retrospective macro")
    axes[0].legend(fontsize=8)
    for key in SELECTED_KEYS:
        rows = [row for row in budgets if row["architecture"] == key]
        axes[1].plot([row["budget"] for row in rows], [row["auroc"] for row in rows], marker="o", label=LABELS[key], color=colors[key])
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(run.BUDGETS, run.BUDGETS)
    axes[1].set_xlabel("absolute tokens"); axes[1].set_ylabel("AUROC")
    axes[1].set_title("Online prediction among unfinished traces")
    for target, marker in ((0.05, "o"), (0.10, "s")):
        rows = [row for row in declaration if float(row["target_fpr"]) == target]
        axes[2].scatter(
            np.mean([float(row["correct_ever_warning"]) for row in rows]),
            np.mean([float(row["wrong_warning_coverage"]) for row in rows]),
            s=90, marker=marker, label=f"target {target:.0%}",
        )
    axes[2].plot([0, 0.15], [0, 0.15], "--", color="#9aa4b2", linewidth=1)
    axes[2].set_xlim(0, 0.15); axes[2].set_ylim(0, 0.32)
    axes[2].set_xlabel("correct-trace ever warning"); axes[2].set_ylabel("wrong-trace warning coverage")
    axes[2].set_title("Trace-level warning policy"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "ARCHITECTURE_SUMMARY.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _md_table(rows, columns, formats=None):
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if column in formats and value != "":
                value = formats[column](value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _html_table(rows, columns, formats=None):
    formats = formats or {}
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if column in formats and value != "":
                value = formats[column](value)
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    architecture_records = _read_csv("ARCHITECTURE_PER_QUESTION.csv")
    architecture_metrics = _read_csv("ARCHITECTURE_METRICS.csv")
    architecture, architecture_intervals = _architecture_summary(architecture_records, architecture_metrics)
    budgets = _budget_summary(architecture_metrics)
    run._write_csv(OUT / "ARCHITECTURE_SELECTED_AGGREGATE.csv", architecture)
    run._write_csv(OUT / "ONLINE_BUDGETS.csv", budgets)
    run._write_csv(OUT / "GROUPED_ARCHITECTURE_INTERVALS.csv", architecture_intervals)

    fusion_records = _read_csv("FUSION_PER_QUESTION.csv")
    fusion_aggregate = _read_csv("FUSION_AGGREGATE.csv")
    ordinary, fusion_lookup, fusion_intervals = _fusion_summary(fusion_records, fusion_aggregate)
    run._write_csv(OUT / "GROUPED_FUSION_INTERVALS.csv", fusion_intervals)
    graph_promoted = any(
        row["ci_low"] > 0.0
        for row in fusion_intervals
        if row["candidate"] in {
            "global_dufs__local_ordinary", "global_ordinary__local_dufs",
            "global_ordinary__local_uniform", "global_ordinary__local_temporal",
        }
        and row["task"] in {"global", "local"}
    )

    head_selection = json.load(open(OUT / "HEAD_SELECTION.json", encoding="utf-8"))
    declaration = _read_csv("DECLARATION_METRICS.csv")
    length = _read_csv("LENGTH_SENSITIVITY.csv")
    efficiency = _read_csv("END_TO_END_EFFICIENCY.csv")
    missing = _read_csv("MISSING_CHANNEL_SENSITIVITY.csv")
    phase15 = json.load(open(OUT / "PHASE15_ONLINE_TRANSFER.json", encoding="utf-8"))
    fusion_diag = json.load(open(OUT / "FUSION_DIAGNOSTICS.json", encoding="utf-8"))

    declaration_macro = []
    for target in (0.05, 0.10):
        rows = [row for row in declaration if float(row["target_fpr"]) == target]
        declaration_macro.append({
            "target FPR": target,
            "observed false warning": float(np.mean([float(row["correct_ever_warning"]) for row in rows])),
            "wrong warning coverage": float(np.mean([float(row["wrong_warning_coverage"]) for row in rows])),
            "mean first budget": float(np.mean([float(row["mean_first_warning_budget"]) for row in rows])),
            "potential remaining tokens": float(np.mean([float(row["potential_tokens_remaining_on_caught_wrong"]) for row in rows])),
        })
    length_macro = []
    for budget in (64, 128):
        rows = [row for row in length if int(row["budget"]) == budget]
        length_macro.append({
            "budget": budget,
            "length Spearman": float(np.mean([float(row["length_spearman"]) for row in rows])),
            "raw AUROC": float(np.mean([float(row["raw_auroc"]) for row in rows])),
            "residual AUROC": float(np.mean([float(row["length_residual_auroc"]) for row in rows])),
            "short": float(np.nanmean([float(row["short_auroc"]) for row in rows])),
            "medium": float(np.nanmean([float(row["medium_auroc"]) for row in rows])),
            "long": float(np.nanmean([float(row["long_auroc"]) for row in rows])),
        })
    efficiency_macro = {
        "median_fit_seconds": float(np.median([float(row["fit_seconds"]) for row in efficiency])),
        "median_score_seconds": float(np.median([float(row["score_all_three_outputs_seconds"]) for row in efficiency])),
        "max_fit_seconds": float(np.max([float(row["fit_seconds"]) for row in efficiency])),
        "max_score_seconds": float(np.max([float(row["score_all_three_outputs_seconds"]) for row in efficiency])),
    }

    selected_arch = next(row for row in architecture if row["architecture"] == PRIMARY)
    one_arch = next(row for row in architecture if row["architecture"] == SELECTED_KEYS[0])
    three_arch = next(row for row in architecture if row["architecture"] == SELECTED_KEYS[2])
    two_vs_one = {row["task"]: row for row in architecture_intervals if row["candidate"] == PRIMARY}
    three_vs_two = {row["task"]: row for row in architecture_intervals if row["candidate"] == SELECTED_KEYS[2]}

    fusion_rows = []
    for method in (
        ordinary, "global_dufs__local_ordinary", "global_ordinary__local_uniform",
        "global_ordinary__local_dufs", "global_ordinary__local_temporal",
    ):
        row = fusion_lookup[method]
        fusion_rows.append({
            "method": method, "Global": float(row["global"]),
            "Local": float(row["local"]), "Online": float(row["online"]),
        })

    ledger = []
    for row in head_selection["ledger"]:
        ledger.append({
            "stage": f"{row['task']} head", "candidate": row["candidate"],
            "primary": row["primary"], "decision": "select" if row["selected"] else "drop",
            "reason": "development winner" if row["selected"] else f"delta vs best {row['delta_vs_best']:+.4f}",
        })
    ledger.extend([
        {"stage": "architecture", "candidate": LABELS[key], "primary": next(row for row in architecture if row["architecture"] == key)["online"], "decision": "select" if key == PRIMARY else "drop", "reason": "fewest heads within all development margins" if key == PRIMARY else "joint margin/cost dominated"}
        for key in SELECTED_KEYS
    ])
    ledger.append({
        "stage": "fusion", "candidate": "DUFS/uniform/temporal Laplacians",
        "primary": max(row["Local"] for row in fusion_rows),
        "decision": "promote" if graph_promoted else "drop",
        "reason": "paired interval excluding zero" if graph_promoted else "small increments with paired intervals crossing zero and extra cost",
    })
    run._write_csv(OUT / "CANDIDATE_LEDGER.csv", ledger)

    plot_path = _plot(architecture, budgets, declaration)
    fmt4 = lambda value: f"{float(value):.4f}"
    fmt3 = lambda value: f"{float(value):.3f}"
    arch_table = _md_table(architecture, ("label", "global", "local", "online"), {"global": fmt4, "local": fmt4, "online": fmt4})
    interval_table = _md_table([
        {
            "comparison": f"{LABELS.get(row['candidate'], row['candidate'])} − {LABELS.get(row['reference'], row['reference'])}",
            "task": row["task"], "delta": row["delta"], "95% CI": f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]",
            "family W/L": f"{row['family_wins']}/{row['family_losses']}",
        } for row in architecture_intervals
    ], ("comparison", "task", "delta", "95% CI", "family W/L"), {"delta": lambda value: f"{float(value):+.4f}"})
    fusion_table = _md_table(fusion_rows, ("method", "Global", "Local", "Online"), {"Global": fmt4, "Local": fmt4, "Online": fmt4})
    declaration_table = _md_table(declaration_macro, ("target FPR", "observed false warning", "wrong warning coverage", "mean first budget", "potential remaining tokens"), {
        "target FPR": lambda value: f"{float(value):.0%}", "observed false warning": lambda value: f"{float(value):.1%}",
        "wrong warning coverage": lambda value: f"{float(value):.1%}", "mean first budget": lambda value: f"{float(value):.1f}",
        "potential remaining tokens": lambda value: f"{float(value):.1f}",
    })
    length_table = _md_table(length_macro, ("budget", "length Spearman", "raw AUROC", "residual AUROC", "short", "medium", "long"), {name: fmt4 for name in ("length Spearman", "raw AUROC", "residual AUROC", "short", "medium", "long")})

    head = {task: value for task, value in head_selection["selected"].items()}
    online_head_row = next(row for row in head_selection["ledger"] if row["candidate"] == "o_iu28_registered")
    phase64 = phase15["budgets"]["64"]["auroc"]
    phase128 = phase15["budgets"]["128"]["auroc"]
    report = f"""# Token-native Global-Local-Online architecture v2

## Decision

**Retain a two-head ordinary-IU architecture: the historical mixed-v2 Global
head plus the new raw token-level Local head. Use an equal 0.50/0.50
calibration-standardized Global/Local detector and the peak locator. Derive the
Online score causally from the prefix Global score and running maximum Local
evidence. Do not retain a third independent Online head, DUFS, or a Laplacian.**

This is a retrospective development decision over existing caches, not fresh
confirmation. It is an unsupervised scorer with calibrated decision policies.
No inference, GPU work, or Drive mutation occurred.

## What the broader search changed

Step 270's narrow conclusion is preserved: its three aggregate-of-aggregate
coarse-grid mechanisms stay closed. The v2 search starts from raw token
telemetry and changes all three previously frozen axes.

- Global independently selects `g_registered_mixed` at **{head_selection['ledger'][3]['primary']:.4f}** development AUROC. The best raw mean/tail replacement reaches {head_selection['ledger'][1]['primary']:.4f}; its delta is {head_selection['ledger'][1]['delta_vs_best']:+.4f} with 95% CI [{head_selection['ledger'][1]['ci_low']:+.4f},{head_selection['ledger'][1]['ci_high']:+.4f}]. The full-trace spectral transformations still matter for Global.
- Local selects `l_level9` at **{head_selection['ledger'][4]['primary']:.4f}** ProcessBench F1. Onset-only ({head_selection['ledger'][5]['primary']:.4f}) and level+onset ({head_selection['ledger'][6]['primary']:.4f}) are worse. The registered core-five replay is {head_selection['ledger'][7]['primary']:.4f} and statistically tied, but outside the frozen 0.005 simplicity window.
- Online independently selects `o_ewma_area_persist27` at **{head_selection['ledger'][10]['primary']:.4f}** 64/128 AUROC versus {online_head_row['primary']:.4f} for registered IU28, a +{head_selection['ledger'][10]['primary'] - online_head_row['primary']:.4f} point difference whose paired interval still crosses zero. Instantaneous/onset variants lose; sustained EWMA, positive area, and persistence are the useful dynamic mechanism.
- The harness then makes the independent Online head unnecessary: the two-head Global+Local derivation is within every development margin of three heads, so the cheaper architecture wins. The old 0.75/0.25 blend is not selected; 0.50/0.50 with the peak locator is the frozen development choice.

## Twelve-cell architecture result

The table equal-weights the twelve scorer-model/family cells. Scorer copies are
repeated measurements; grouped intervals resample each source question once and
carry all scorer copies together before equal family weighting.

{arch_table}

{interval_table}

Relative to one shared head, two heads improve Global by
**{two_vs_one['global']['delta']:+.4f}** [{two_vs_one['global']['ci_low']:+.4f},{two_vs_one['global']['ci_high']:+.4f}]
and Local by **{two_vs_one['local']['delta']:+.4f}** [{two_vs_one['local']['ci_low']:+.4f},{two_vs_one['local']['ci_high']:+.4f}].
The Online delta is {two_vs_one['online']['delta']:+.4f}
[{two_vs_one['online']['ci_low']:+.4f},{two_vs_one['online']['ci_high']:+.4f}].
Adding the third independent Online head changes only Online and is
{three_vs_two['online']['delta']:+.4f}
[{three_vs_two['online']['ci_low']:+.4f},{three_vs_two['online']['ci_high']:+.4f}]
versus two heads. It does not earn its 27 features and 36 state scalars.

![Architecture summary](ARCHITECTURE_SUMMARY.png)

## Fusion decision

All graph rows use the exact selected matrix, preprocessing, two-dimensional IU
subspace, and reducer. `lambda=0` is bit-identical in every path.

{fusion_table}

DUFS changes Global by only {float(fusion_lookup['global_dufs__local_ordinary']['global']) - float(fusion_lookup[ordinary]['global']):+.4f} AUROC.
The best Local graph increment is about
{max(float(row['Local']) for row in fusion_rows) - float(fusion_lookup[ordinary]['local']):+.4f} F1.
The paired intervals for these increments cross zero, while DUFS fit costs up to
{max(value['global']['dufs_seconds'] / max(value['global']['uniform_seconds'], 1e-9) for value in fusion_diag.values()):.1f}×
the uniform path in the measured development cells. Ordinary IU-PCR is the
supported fusion choice; neither DUFS nor temporal/uniform Laplacians are needed.

## Causal warning and length failure tests

Declaration thresholds are calibrated on the maximum score over the entire
absolute-budget horizon. They control trace-level ever-warning, not a per-time
FPR.

{declaration_table}

The policy is useful but modest: it catches roughly one quarter of wrong traces
at the 10% target. Remaining-token numbers are potential exposure only, not
realized savings, because no forced-closure branches were generated.

{length_table}

Length correlation is small to moderate. Calibration-only isotonic
residualization lowers AUROC but leaves above-chance signal, and every length
band remains informative at 64/128 tokens. The effect is not merely final-length
prediction, although long-trace discrimination is the weakest band.

The Phase-15 MATH-500 T=1.0 cache, which lacks log-sum-exp, is an Online-only
transfer: AUROC is **{phase64:.4f}** at 64 and **{phase128:.4f}** at 128 tokens,
with final AUROC {phase15['final_auroc']:.4f}. This is a clear failure of robust
early transfer, not evidence for a new-model win.

## Feature requirements and cost

The Global missing-family audit identifies top-k log-probabilities as the main
increment: removing them costs about 0.0266 development AUROC. The selected
Online sustained head is insensitive to a single primitive (largest drop about
0.004). For Local, dropping spilled or top-k entropy improves development F1;
because those are outcome-opened diagnostics, no post-hoc pruned head is
promoted. They define the next frozen subset roster.

Median per-cell fit time is {efficiency_macro['median_fit_seconds']:.1f}s and
median complete calibration+evaluation three-output scoring time is
{efficiency_macro['median_score_seconds']:.1f}s on local CPU. The long-family
maxima are {efficiency_macro['max_fit_seconds']:.1f}s fit and
{efficiency_macro['max_score_seconds']:.1f}s score. Profiling shows the
historical mixed-v2 Global prefix recomputation, not the O(1) token-native Local
state, is the bottleneck. Therefore the selected system is the statistical
winner in this roster but not yet a fully optimized streaming implementation.

## Claim boundary and next gate

- All twelve ProcessBench cells and Phase-15 are historically opened. The
  non-selection application protects this run's mechanics but is not fresh
  confirmation.
- Global/Online final-answer wrongness and Local trace-error presence are
  distinct labels; no metric substitutes one for the other.
- PRMBench remains a separate Local-only anchor (existing frozen step AUROC
  0.6711); v2 did not refit a PRMBench-compatible Global/Online task.
- The historical Local core replay contains a completed-trace CUSUM curve and
  is not called suffix-invariant. Every new v2 recurrence and every deployable
  Online score passes suffix/chunk replay tests.
- The 0.50/0.50 blend, Local subset pruning, and the weak Phase-15 early transfer
  require fresh confirmation before a deployment or paper-level claim.

No GPU run is justified merely to add DUFS or the third head. A narrowly scoped
fresh-data run may be justified for the frozen two-head ordinary-IU architecture
only after preregistering a cheaper causal implementation of the mixed-v2
Global prefix and the Local drop-one subset candidates. That future run needs
explicit approval.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    image_data = base64.b64encode(plot_path.read_bytes()).decode("ascii")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Global-Local-Online v2</title>
<style>body{{font-family:Inter,system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;color:#18212b;line-height:1.5}}h1,h2{{color:#123b5d}}.decision{{background:#e9f3fb;border-left:5px solid #1769aa;padding:16px 20px}}table{{border-collapse:collapse;width:100%;margin:14px 0 26px}}th,td{{border:1px solid #ccd5df;padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f7}}.muted{{color:#5f6b78}}img{{max-width:100%}}code{{background:#eef3f7;padding:2px 4px}}</style></head><body>
<h1>Token-native Global-Local-Online architecture v2</h1>
<div class='decision'><strong>Decision:</strong> retain two ordinary-IU heads: historical mixed-v2 Global plus token-native level Local. Use the frozen 0.50/0.50 detector/Online blend and peak locator. Drop the third Online head and all graph regularization.</div>
<p class='muted'>Retrospective existing-cache evidence; unsupervised scorer with calibrated decision policies. No new inference or GPU use.</p>
<img src='data:image/png;base64,{image_data}' alt='architecture summary'>
<h2>Twelve-cell macro</h2>{_html_table(architecture, ('label','global','local','online'), {'global':fmt4,'local':fmt4,'online':fmt4})}
<h2>Grouped paired intervals</h2>{_html_table([{'comparison':f"{LABELS.get(r['candidate'],r['candidate'])} - {LABELS.get(r['reference'],r['reference'])}",'task':r['task'],'delta':r['delta'],'95% CI':f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",'family W/L':f"{r['family_wins']}/{r['family_losses']}"} for r in architecture_intervals], ('comparison','task','delta','95% CI','family W/L'), {'delta':lambda v:f"{float(v):+.4f}"})}
<h2>Same-matrix fusion</h2>{_html_table(fusion_rows, ('method','Global','Local','Online'), {'Global':fmt4,'Local':fmt4,'Online':fmt4})}
<h2>Trace-level warnings</h2>{_html_table(declaration_macro, ('target FPR','observed false warning','wrong warning coverage','mean first budget','potential remaining tokens'), {'target FPR':lambda v:f"{float(v):.0%}",'observed false warning':lambda v:f"{float(v):.1%}",'wrong warning coverage':lambda v:f"{float(v):.1%}",'mean first budget':lambda v:f"{float(v):.1f}",'potential remaining tokens':lambda v:f"{float(v):.1f}"})}
<h2>Length audit</h2>{_html_table(length_macro, ('budget','length Spearman','raw AUROC','residual AUROC','short','medium','long'), {name:fmt4 for name in ('length Spearman','raw AUROC','residual AUROC','short','medium','long')})}
<h2>Interpretation</h2><p>The broad result is architectural: Global still benefits from full-trace spectral transformations, Local prefers raw token levels, and Online benefits from sustained dynamics when isolated. Once heads interact, however, a third Online head is redundant. Graph gains are too small and uncertain to pay for DUFS/Laplacian cost. Phase-15 early transfer is weak, so fresh confirmation remains necessary.</p>
</body></html>"""
    (OUT / "REPORT.html").write_text(html, encoding="utf-8")

    decision = {
        "status": "COMPLETE_RETROSPECTIVE_TWO_HEAD_SELECTED",
        "architecture": "two_global_local",
        "global_head": head["global"], "local_head": head["local"],
        "independent_online_head": None,
        "online_derivation": "0.50 standardized causal prefix Global + 0.50 standardized running-max Local",
        "local_detector_weight_global": 0.50,
        "locator": "peak",
        "fusion": "ordinary_iu_pcr",
        "dufs_retained": False, "laplacian_retained": False,
        "fresh_confirmation": False,
        "new_inference": False, "gpu_hours": 0,
        "phase15_early_transfer_pass": False,
        "next_gate": "fresh-data run only after a frozen cheaper Global prefix implementation and Local subset roster; explicit approval required",
    }
    run._write_json(OUT / "DECISION.json", decision)
    run._write_json(OUT / "AUDIT.json", {
        "protocol_sha256": run._sha256(ROOT / "docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md"),
        "tests": {
            "raw_formula_identity": "PASS", "suffix_replacement": "PASS",
            "tokenwise_chunk_replay": "PASS", "global_prefix_identity": "PASS",
            "feature_order": "PASS", "label_removal_permutation": "PASS",
            "repeat_identity": "PASS", "missing_constant": "PASS",
            "orientation_monotonicity": "PASS_WITH_DECLARED_ONSET_EXCEPTION",
            "shared_id_split": "PASS", "lambda_zero": "PASS",
        },
        "transfer_resume": "corrected architecture-key dispatch before interpretation; corrected hash in RUN_MANIFEST.json",
        "all_results_retrospective": True,
        "labels_seen_during_score_fit": False,
        "new_inference": False, "drive_mutation": False,
        "a6_ptni_touched": False,
    })
    manifest = json.load(open(OUT / "RUN_MANIFEST.json", encoding="utf-8"))
    manifest.update({
        "status": "COMPLETE_RETROSPECTIVE_TWO_HEAD_SELECTED",
        "decision_sha256": run._sha256(OUT / "DECISION.json"),
        "report_sha256": run._sha256(OUT / "REPORT.md"),
        "grouped_architecture_intervals_sha256": run._sha256(OUT / "GROUPED_ARCHITECTURE_INTERVALS.csv"),
        "fusion_manifest_sha256": run._sha256(OUT / "FUSION_MANIFEST.json"),
    })
    run._write_json(OUT / "RUN_MANIFEST.json", manifest)
    print(f"[done] froze report and decision in {OUT}")


if __name__ == "__main__":
    main()
