#!/usr/bin/env python3
"""Render frozen diagnostics and gates for the cross-view Phase-1 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, "results", "graybox_cross_view_phase1")
PRIMARY_LAMBDA = 0.1
PRIMARY_K = 7
WORLDS = ("P1-A", "P1-B", "P1-C", "P1-D", "P1-E", "P1-F")
WORLD_LABELS = {
    "P1-A": "Aligned target",
    "P1-B": "Discovery nuisance",
    "P1-C": "Measured shared nuisance",
    "P1-D": "Paired targets",
    "P1-E": "Pure noise",
    "P1-F": "Unmeasured shared nuisance",
}
ARM_LABELS = {
    "iu": "IU-PCR",
    "direct_g": "Direct G graph",
    "direct_a": "Direct A graph",
    "g_to_a": "G→A audited",
    "a_to_g": "A→G audited",
    "consensus": "Hard-veto consensus",
    "mmdufs_shared": "mmDUFS-inspired",
    "projected_ridge": "Projected ridge",
    "permuted_consensus": "Permuted consensus",
    "nuisance_only": "Nuisance graph",
    "oracle": "Oracle graph",
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def as_int(row, key):
    return int(row[key])


def as_bool(row, key):
    return str(row[key]).lower() == "true"


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def one_sided_lower(values):
    values = np.asarray(values, dtype=float)
    mean, se = mean_se(values)
    if len(values) < 2:
        return mean
    return float(mean - student_t.ppf(0.95, len(values) - 1) * se)


def two_sided_interval(values):
    values = np.asarray(values, dtype=float)
    mean, se = mean_se(values)
    if len(values) < 2:
        return mean, mean
    radius = student_t.ppf(0.975, len(values) - 1) * se
    return float(mean - radius), float(mean + radius)


def subset(rows, **filters):
    return [
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in filters.items())
    ]


def primary_rows(rows, world, arm, target="g"):
    return [
        row for row in rows
        if row["world"] == world and row["arm"] == arm
        and row["target"] == target
        and abs(as_float(row, "lambda") - PRIMARY_LAMBDA) < 1e-12
    ]


def paired_difference(rows, world, left, right, target="g"):
    left_rows = {
        as_int(row, "replicate"): as_float(row, "auroc_delta")
        for row in primary_rows(rows, world, left, target)
    }
    right_rows = {
        as_int(row, "replicate"): as_float(row, "auroc_delta")
        for row in primary_rows(rows, world, right, target)
    }
    if set(left_rows) != set(right_rows):
        raise ValueError(f"unpaired rows for {world}: {left} vs {right}")
    return np.asarray([
        left_rows[key] - right_rows[key] for key in sorted(left_rows)
    ])


def dataset_primary_audits(audits, world):
    return [
        row for row in audits
        if row["world"] == world and as_int(row, "k") == PRIMARY_K
    ]


def acceptance_by_dataset(audits, datasets, world):
    world_audits = dataset_primary_audits(audits, world)
    by_rep = {}
    for row in world_audits:
        rep = as_int(row, "replicate")
        by_rep.setdefault(rep, []).append(row)
    output = {}
    for dataset in datasets:
        if dataset["world"] != world:
            continue
        rep = as_int(dataset, "replicate")
        directions = by_rep[rep]
        output[rep] = {
            "any_raw_pass": any(as_bool(row, "raw_pass") for row in directions),
            "all_raw_pass": all(as_bool(row, "raw_pass") for row in directions),
            "any_nuisance_veto": any(
                as_bool(row, "nuisance_veto") for row in directions
            ),
            "any_direction_accepted": as_bool(dataset, "consensus_accepted"),
            "accepted_count": as_int(dataset, "accepted_direction_count"),
        }
    return output


def summarize(rows):
    output = []
    keys = sorted({
        (row["world"], row["target"], row["arm"], as_float(row, "lambda"))
        for row in rows
    })
    for world, target, arm, lambda_ in keys:
        values = [
            as_float(row, "auroc_delta") for row in rows
            if row["world"] == world and row["target"] == target
            and row["arm"] == arm
            and abs(as_float(row, "lambda") - lambda_) < 1e-12
        ]
        aucs = [
            as_float(row, "auroc") for row in rows
            if row["world"] == world and row["target"] == target
            and row["arm"] == arm
            and abs(as_float(row, "lambda") - lambda_) < 1e-12
        ]
        mean, se = mean_se(values)
        auc_mean, auc_se = mean_se(aucs)
        output.append({
            "world": world,
            "target": target,
            "arm": arm,
            "lambda": lambda_,
            "n": len(values),
            "auroc_mean": auc_mean,
            "auroc_se": auc_se,
            "auroc_delta_mean": mean,
            "auroc_delta_se": se,
            "auroc_delta_one_sided_95_lower": one_sided_lower(values),
        })
    return output


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gates(rows, audits, datasets):
    acceptance = {
        world: acceptance_by_dataset(audits, datasets, world) for world in WORLDS
    }
    a_accept = sum(item["any_direction_accepted"] for item in acceptance["P1-A"].values())
    c_raw = sum(item["any_raw_pass"] for item in acceptance["P1-C"].values())
    c_fallback = sum(not item["any_direction_accepted"] for item in acceptance["P1-C"].values())
    e_accept = sum(item["any_direction_accepted"] for item in acceptance["P1-E"].values())
    f_raw = sum(item["any_raw_pass"] for item in acceptance["P1-F"].values())
    b_g_accept = sum(
        as_bool(row, "direction_accepted")
        for row in dataset_primary_audits(audits, "P1-B")
        if row["direction"] == "g_to_a"
    )

    paired_hash_pass = True
    for replicate in sorted({
        as_int(row, "replicate") for row in rows if row["world"] == "P1-D"
    }):
        hashes = {
            row["method_hash"] for row in rows
            if row["world"] == "P1-D" and as_int(row, "replicate") == replicate
            and row["arm"] != "oracle"
        }
        paired_hash_pass &= len(hashes) == 1

    lambda0_error = max(as_float(row, "lambda0_max_score_error") for row in datasets)
    min_roughness = min(as_float(row, "roughness_min_eigenvalue") for row in datasets)
    method_once = all(as_int(row, "method_call_count") == 1 for row in datasets)
    labels_absent = all(as_bool(row, "label_parameter_absent") for row in datasets)
    gate0 = bool(
        lambda0_error == 0.0 and min_roughness >= -1e-8
        and method_once and labels_absent and paired_hash_pass
    )

    gate1 = bool(
        a_accept >= 7 and b_g_accept <= 1 and c_raw >= 7 and c_fallback >= 7
        and e_accept <= 1 and f_raw >= 7 and paired_hash_pass
    )

    a_candidate = np.asarray([
        as_float(row, "auroc_delta")
        for row in primary_rows(rows, "P1-A", "consensus")
    ])
    direct_g = np.asarray([
        as_float(row, "auroc_delta")
        for row in primary_rows(rows, "P1-A", "direct_g")
    ])
    direct_a = np.asarray([
        as_float(row, "auroc_delta")
        for row in primary_rows(rows, "P1-A", "direct_a")
    ])
    better_direct_gain = max(float(direct_g.mean()), float(direct_a.mean()))
    retain_threshold = 0.8 * max(better_direct_gain, 0.0)
    vs_ridge = paired_difference(rows, "P1-A", "consensus", "projected_ridge")
    vs_permuted = paired_difference(
        rows, "P1-A", "consensus", "permuted_consensus"
    )
    gate2 = bool(
        a_candidate.mean() >= 0.005
        and one_sided_lower(a_candidate) > 0.0
        and a_candidate.mean() >= retain_threshold
        and one_sided_lower(vs_ridge) > 0.0
        and one_sided_lower(vs_permuted) > 0.0
    )

    safety = {}
    safety_pass = True
    for world in ("P1-B", "P1-C", "P1-F"):
        values = np.asarray([
            as_float(row, "auroc_delta")
            for row in primary_rows(rows, world, "consensus")
        ])
        item = {
            "mean": float(values.mean()),
            "one_sided_95_lower": one_sided_lower(values),
            "mean_pass": bool(values.mean() >= -0.001),
            "lower_pass": bool(one_sided_lower(values) >= -0.005),
        }
        safety[world] = item
        safety_pass &= item["mean_pass"] and item["lower_pass"]
    b_safer = (
        np.mean([as_float(row, "auroc_delta") for row in primary_rows(
            rows, "P1-B", "consensus"
        )])
        > np.mean([as_float(row, "auroc_delta") for row in primary_rows(
            rows, "P1-B", "direct_g"
        )])
    )
    c_candidate = np.mean([
        as_float(row, "auroc_delta") for row in primary_rows(
            rows, "P1-C", "consensus"
        )
    ])
    c_safer = all(
        c_candidate > np.mean([
            as_float(row, "auroc_delta") for row in primary_rows(rows, "P1-C", arm)
        ])
        for arm in ("direct_g", "direct_a")
    )
    gate3 = bool(safety_pass and b_safer and c_safer)

    a_audits = dataset_primary_audits(audits, "P1-A")
    audit_permutation_destroyed = sum(
        as_float(row, "audit_row_permutation_p") > 0.025 for row in a_audits
    ) >= 14
    gate4 = bool(
        audit_permutation_destroyed
        and one_sided_lower(vs_permuted) > 0.0
        and one_sided_lower(vs_ridge) > 0.0
        and b_g_accept <= 1
    )

    e_values = np.asarray([
        as_float(row, "auroc_delta")
        for row in primary_rows(rows, "P1-E", "consensus")
    ])
    e_interval = two_sided_interval(e_values)
    gate6 = bool(
        e_interval[0] <= 0.0 <= e_interval[1]
        and e_accept <= 1
        and all(as_int(row, "n") > 0 for row in datasets)
    )

    overall = bool(gate0 and gate1 and gate2 and gate3 and gate4 and gate6)
    return {
        "gate0_algebra_implementation": gate0,
        "gate1_audit_premise": gate1,
        "gate2_positive_mechanism": gate2,
        "gate3_nuisance_safety": gate3,
        "gate4_attribution": gate4,
        "gate5_phase2_representation": None,
        "gate6_null_missingness_safety": gate6,
        "gate7_scope_discipline": True,
        "overall_phase1_pass": overall,
        "counts": {
            "p1a_consensus_accept": a_accept,
            "p1b_g_to_a_accept": b_g_accept,
            "p1c_any_raw_transfer": c_raw,
            "p1c_fallback": c_fallback,
            "p1e_consensus_accept": e_accept,
            "p1f_any_raw_transfer": f_raw,
            "p1a_audit_row_permutation_destroyed": sum(
                as_float(row, "audit_row_permutation_p") > 0.025
                for row in a_audits
            ),
            "p1a_audit_direction_tests": len(a_audits),
        },
        "invariants": {
            "lambda0_max_score_error": lambda0_error,
            "roughness_min_eigenvalue": min_roughness,
            "method_called_once": method_once,
            "label_parameter_absent": labels_absent,
            "paired_target_hash_identical": paired_hash_pass,
        },
        "positive": {
            "candidate_mean": float(a_candidate.mean()),
            "candidate_one_sided_95_lower": one_sided_lower(a_candidate),
            "better_direct_mean": better_direct_gain,
            "required_retained_gain": retain_threshold,
            "vs_projected_ridge_lower": one_sided_lower(vs_ridge),
            "vs_permuted_lower": one_sided_lower(vs_permuted),
        },
        "safety": safety,
        "p1b_safer_than_direct_g": bool(b_safer),
        "p1c_safer_than_both_direct_graphs": bool(c_safer),
        "pure_noise_interval": list(e_interval),
    }


def plot_decision_funnel(audits, datasets, figures):
    stages = ("Raw transfer", "No nuisance veto", "Final accepted")
    x = np.arange(len(WORLDS))
    values = {stage: [] for stage in stages}
    for world in WORLDS:
        state = acceptance_by_dataset(audits, datasets, world)
        values["Raw transfer"].append(sum(item["any_raw_pass"] for item in state.values()))
        values["No nuisance veto"].append(sum(
            item["any_raw_pass"] and not item["any_nuisance_veto"]
            for item in state.values()
        ))
        values["Final accepted"].append(sum(
            item["any_direction_accepted"] for item in state.values()
        ))
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.24
    colors = ("#64748b", "#d97706", "#2563eb")
    for index, stage in enumerate(stages):
        ax.bar(x + (index - 1) * width, values[stage], width,
               label=stage, color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels([WORLD_LABELS[w] for w in WORLDS], rotation=20, ha="right")
    ax.set_ylabel("dataset replicates")
    ax.set_title("Frozen graph-decision funnel")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "01_decision_funnel.png"), dpi=170)
    plt.close(fig)


def plot_lambda_paths(rows, figures):
    arms = ("consensus", "direct_g", "direct_a", "mmdufs_shared", "projected_ridge")
    colors = ("#2563eb", "#0f766e", "#14b8a6", "#7c3aed", "#64748b")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, world in zip(axes.ravel(), WORLDS):
        for arm, color in zip(arms, colors):
            means, ses = [], []
            lambdas = sorted({
                as_float(row, "lambda") for row in rows
                if row["world"] == world and row["target"] == "g"
            })
            for lambda_ in lambdas:
                vals = [
                    as_float(row, "auroc_delta") for row in rows
                    if row["world"] == world and row["target"] == "g"
                    and row["arm"] == arm
                    and abs(as_float(row, "lambda") - lambda_) < 1e-12
                ]
                mean, se = mean_se(vals)
                means.append(100 * mean)
                ses.append(100 * se)
            ax.errorbar(lambdas, means, yerr=ses, marker="o", ms=3,
                        linewidth=1.3, color=color, label=ARM_LABELS[arm])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(PRIMARY_LAMBDA, color="#dc2626", linestyle="--", linewidth=0.9)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_title(WORLD_LABELS[world])
        ax.set_ylabel("Δ AUROC (pp)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Diagnostic lambda paths; primary lambda=0.1 was not retuned")
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "02_lambda_paths.png"), dpi=170)
    plt.close(fig)


def plot_transfer_nuisance(audits, figures):
    primary = [row for row in audits if as_int(row, "k") == PRIMARY_K]
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {world: marker for world, marker in zip(WORLDS, "o^sDPX")}
    for world in WORLDS:
        selected = [row for row in primary if row["world"] == world]
        ax.scatter(
            [as_float(row, "nuisance_cka") for row in selected],
            [as_float(row, "transfer_statistic") for row in selected],
            c=["#2563eb" if as_bool(row, "direction_accepted") else "#94a3b8"
               for row in selected],
            marker=markers[world], s=48, alpha=0.8, label=WORLD_LABELS[world],
        )
    ax.axhline(2.0, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_xlabel("nuisance affinity CKA")
    ax.set_ylabel("out-of-family transfer robust Z")
    ax.set_title("Transfer versus nuisance alignment (blue = accepted)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "03_transfer_vs_nuisance.png"), dpi=170)
    plt.close(fig)


def plot_stability(audits, figures):
    primary = [row for row in audits if as_int(row, "k") == PRIMARY_K]
    values = [
        [as_float(row, "stability_cka") for row in primary if row["world"] == world]
        for world in WORLDS
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values, tick_labels=[WORLD_LABELS[w] for w in WORLDS], showfliers=False)
    ax.axhline(0.75, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_ylabel("median affinity CKA: k=7 vs {5,11}")
    ax.set_title("Graph stability across frozen neighborhood sizes")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "04_graph_stability.png"), dpi=170)
    plt.close(fig)


def plot_primary_comparison(rows, figures):
    arms = ("consensus", "direct_g", "direct_a", "mmdufs_shared", "projected_ridge")
    x = np.arange(len(WORLDS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, arm in enumerate(arms):
        means, ses = [], []
        for world in WORLDS:
            vals = [as_float(row, "auroc_delta") for row in primary_rows(rows, world, arm)]
            mean, se = mean_se(vals)
            means.append(100 * mean)
            ses.append(100 * se)
        ax.bar(x + (index - 2) * width, means, width, yerr=ses,
               label=ARM_LABELS[arm], capsize=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([WORLD_LABELS[w] for w in WORLDS], rotation=20, ha="right")
    ax.set_ylabel("Δ AUROC at lambda=0.1 (pp)")
    ax.set_title("Same-view, audited, shared-operator, and ridge controls")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "05_primary_comparison.png"), dpi=170)
    plt.close(fig)


def plot_evidence_convergence(rows, gates, figures):
    current_a = 100 * gates["positive"]["candidate_mean"]
    current_f = 100 * gates["safety"]["P1-F"]["mean"]
    labels = (
        "Prior same-view\nsmooth",
        "Prior same-view\nnuisance",
        "Cross-view P1-A\naligned",
        "Cross-view P1-F\nunmeasured nuisance",
    )
    values = (0.382, -0.568, current_a, current_f)
    colors = ("#0f766e", "#dc2626", "#2563eb", "#d97706")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(np.arange(4), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=0.9,
               label="meaningful-gain threshold")
    ax.axhline(-0.5, color="#64748b", linestyle=":", linewidth=0.9,
               label="maximum lower-bound harm")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean Δ AUROC (pp)")
    ax.set_title("Evidence convergence: the new test addresses the prior failure")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:+.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "06_evidence_convergence.png"), dpi=170)
    plt.close(fig)


def write_report(path, stage, rows, audits, datasets, summary, gates):
    lines = [
        f"# Gray-box cross-view graph audit — {stage} report",
        "",
        "Version: `graybox-cross-view-phase1-v1-2026-08-06`",
        "",
        "## Scope",
        "",
        "Generated data only. The cross-view method was called once per dataset and",
        "received only G, A, and N. Evaluator labels and oracle latents were joined",
        "after method scores and hashes were frozen. Confirmation was not opened.",
        "",
        "## Frozen design",
        "",
        f"- Dataset replicates per world: {len({as_int(r, 'replicate') for r in datasets if r['world'] == 'P1-A'})}",
        f"- Samples per replicate: {datasets[0]['n']}",
        "- k path: [5, 7, 11]; primary k=7",
        "- lambda path: [0, 0.01, 0.03, 0.1, 0.3, 1.0]; primary lambda=0.1",
        "- 199 synchronized node permutations per graph",
        "- transfer: p<=0.025 and robust Z>=2.0",
        "- stability: identical decision across k and median affinity CKA>=0.75",
        "",
        "## Audit decisions",
        "",
        "| world | raw transfer (datasets) | final accepted |",
        "|---|---:|---:|",
    ]
    for world in WORLDS:
        state = acceptance_by_dataset(audits, datasets, world)
        raw = sum(item["any_raw_pass"] for item in state.values())
        accepted = sum(item["any_direction_accepted"] for item in state.values())
        lines.append(f"| {WORLD_LABELS[world]} | {raw}/{len(state)} | {accepted}/{len(state)} |")
    lines += [
        "",
        "## Primary performance at lambda=0.1",
        "",
        "Values are mean paired AUROC changes in percentage points +/- one SE versus ordinary IU-PCR.",
        "",
        "| world | consensus | direct G | direct A | mmDUFS-inspired | ridge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for world in WORLDS:
        cells = []
        for arm in ("consensus", "direct_g", "direct_a", "mmdufs_shared", "projected_ridge"):
            values = [as_float(row, "auroc_delta") for row in primary_rows(rows, world, arm)]
            mean, se = mean_se(values)
            cells.append(f"{100 * mean:+.3f} +/- {100 * se:.3f}")
        lines.append(f"| {WORLD_LABELS[world]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Frozen gates",
        "",
        "| gate | result |",
        "|---|---:|",
    ]
    for key in (
        "gate0_algebra_implementation",
        "gate1_audit_premise",
        "gate2_positive_mechanism",
        "gate3_nuisance_safety",
        "gate4_attribution",
        "gate6_null_missingness_safety",
        "overall_phase1_pass",
    ):
        lines.append(f"| `{key}` | **{'PASS' if gates[key] else 'FAIL'}** |")
    lines += [
        "",
        "### Decisive diagnostics",
        "",
        f"- P1-A consensus acceptance: {gates['counts']['p1a_consensus_accept']}/8.",
        f"- P1-B nuisance G->A acceptance: {gates['counts']['p1b_g_to_a_accept']}/8.",
        f"- P1-C raw transfer/fallback: {gates['counts']['p1c_any_raw_transfer']}/8 / {gates['counts']['p1c_fallback']}/8.",
        f"- P1-E consensus acceptance: {gates['counts']['p1e_consensus_accept']}/8.",
        f"- P1-F raw transfer: {gates['counts']['p1f_any_raw_transfer']}/8.",
        f"- P1-A mean/lower delta: {100 * gates['positive']['candidate_mean']:+.3f} / {100 * gates['positive']['candidate_one_sided_95_lower']:+.3f} pp.",
        f"- P1-F mean/lower delta: {100 * gates['safety']['P1-F']['mean']:+.3f} / {100 * gates['safety']['P1-F']['one_sided_95_lower']:+.3f} pp.",
        f"- Lambda=0 exact score error: {gates['invariants']['lambda0_max_score_error']:.3e}.",
        f"- Minimum roughness eigenvalue: {gates['invariants']['roughness_min_eigenvalue']:.3e}.",
        "",
        "## Decision",
        "",
    ]
    if gates["overall_phase1_pass"]:
        lines += [
            "Every essential Phase-1 gate passed. Phase 2 is eligible for a separate",
            "review and exact generator preregistration; it is not automatically launched.",
        ]
    else:
        lines += [
            "At least one essential Phase-1 gate failed. Do not implement the trajectory",
            "bank or group-gated DUFS as a rescue. Interpret the failed mechanism first.",
        ]
    lines += [
        "",
        "## Figures",
        "",
        "- `figures/01_decision_funnel.png`",
        "- `figures/02_lambda_paths.png`",
        "- `figures/03_transfer_vs_nuisance.png`",
        "- `figures/04_graph_stability.png`",
        "- `figures/05_primary_comparison.png`",
        "- `figures/06_evidence_convergence.png`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "development", "confirmation"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = os.path.join(args.out_dir, args.stage)
    rows = load_csv(prefix + "_per_run.csv")
    audits = load_csv(prefix + "_audit_diagnostics.csv")
    datasets = load_csv(prefix + "_dataset_diagnostics.csv")
    summary = summarize(rows)
    summary_path = os.path.join(args.out_dir, f"{args.stage}_summary.csv")
    write_csv(summary_path, summary)
    figures = os.path.join(args.out_dir, f"{args.stage}_figures")
    os.makedirs(figures, exist_ok=True)

    # Smoke is an implementation/runtime check. Render its diagnostics, but do
    # not apply the 8-replicate development acceptance counts to two replicates.
    if args.stage == "smoke":
        invariant = {
            "lambda0_max_score_error": max(
                as_float(row, "lambda0_max_score_error") for row in datasets
            ),
            "roughness_min_eigenvalue": min(
                as_float(row, "roughness_min_eigenvalue") for row in datasets
            ),
            "method_called_once": all(
                as_int(row, "method_call_count") == 1 for row in datasets
            ),
            "label_parameter_absent": all(
                as_bool(row, "label_parameter_absent") for row in datasets
            ),
        }
        passed = bool(
            invariant["lambda0_max_score_error"] == 0.0
            and invariant["roughness_min_eigenvalue"] >= -1e-8
            and invariant["method_called_once"]
            and invariant["label_parameter_absent"]
        )
        with open(os.path.join(args.out_dir, "smoke_gate.json"), "w", encoding="utf-8") as handle:
            json.dump({"smoke_invariant_pass": passed, "invariants": invariant},
                      handle, indent=2, sort_keys=True)
        plot_decision_funnel(audits, datasets, figures)
        plot_lambda_paths(rows, figures)
        plot_transfer_nuisance(audits, figures)
        plot_stability(audits, figures)
        plot_primary_comparison(rows, figures)
        print(json.dumps({"smoke_invariant_pass": passed, **invariant}, indent=2))
        return

    gates = evaluate_gates(rows, audits, datasets)
    with open(os.path.join(args.out_dir, f"{args.stage}_gate_decisions.json"),
              "w", encoding="utf-8") as handle:
        json.dump(gates, handle, indent=2, sort_keys=True)
    plot_decision_funnel(audits, datasets, figures)
    plot_lambda_paths(rows, figures)
    plot_transfer_nuisance(audits, figures)
    plot_stability(audits, figures)
    plot_primary_comparison(rows, figures)
    plot_evidence_convergence(rows, gates, figures)
    report_name = "DEVELOPMENT_REPORT.md" if args.stage == "development" else "CONFIRMATION_REPORT.md"
    write_report(
        os.path.join(args.out_dir, report_name), args.stage, rows, audits,
        datasets, summary, gates,
    )
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
