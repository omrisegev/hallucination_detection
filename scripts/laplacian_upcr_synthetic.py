#!/usr/bin/env python3
"""Phase-1 synthetic falsification study for Laplacian-regularized IU-PCR.

This is deliberately not a real-data benchmark.  It asks whether the proposed
mechanism can work, whether a DUFS-derived graph contains the needed geometry,
and where it fails before any hallucination benchmark is opened.

The protocol has disjoint development and confirmation seeds.  A single
regularization strength is selected from the smooth-signal development world;
all primary and gated confirmation contrasts use that frozen value.  Labels
are used only for evaluation and for this offline synthetic choice.  The
candidate estimator accepts only the feature matrix.

Controls
--------
dufs
    Continuous DUFS gates build the sample graph; no feature is removed.
ungated
    The same graph construction with every gate fixed to one.
random_gates
    A feature permutation of the learned gates, preserving their distribution.
permuted_graph
    A node permutation of the DUFS graph, preserving graph spectrum and edge
    weights while destroying sample neighbourhood alignment.
projected_ridge
    Isotropic ridge inside the same IU-PCR two-component subspace, with the
    same trace scale.  This isolates graph geometry from generic shrinkage.
oracle_latent
    A graph built from the generating latent variable.  This is a synthetic
    mechanism ceiling and is never an implementable candidate.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.linalg import eigh
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
    permute_graph,
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)


VERSION = "laplacian-upcr-synthetic-v3-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "laplacian_upcr_synthetic")
CONFIRMATION_SEED_OFFSET = 1_700_000
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
WORLDS = (
    "smooth_signal",
    "correlated_errors",
    "nuisance_manifold",
    "subspace_exclusion",
    "disconnected_graph",
    "pure_noise",
)
CONTROLS = (
    "dufs",
    "ungated",
    "random_gates",
    "permuted_graph",
    "projected_ridge",
    "oracle_latent",
)
NON_ORACLE_CONTROLS = tuple(control for control in CONTROLS
                            if control not in ("dufs", "oracle_latent"))
CONTROL_LABELS = {
    "dufs": "DUFS graph",
    "ungated": "Ungated graph",
    "random_gates": "Shuffled gates",
    "permuted_graph": "Permuted graph",
    "projected_ridge": "Projected ridge",
    "oracle_latent": "Oracle latent graph",
}
CONTROL_COLORS = {
    "dufs": "#2563eb",
    "ungated": "#0f766e",
    "random_gates": "#d97706",
    "permuted_graph": "#dc2626",
    "projected_ridge": "#64748b",
    "oracle_latent": "#7c3aed",
}
WORLD_LABELS = {
    "smooth_signal": "Smooth signal",
    "correlated_errors": "Correlated errors",
    "nuisance_manifold": "Nuisance manifold",
    "subspace_exclusion": "Signal outside U2",
    "disconnected_graph": "Disconnected graph",
    "pure_noise": "Pure noise",
}
EPS = 1e-12


def zscore_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.where(scale > EPS, scale, 1.0)


def balanced_labels(latent, noise, rng):
    decision = np.asarray(latent) + float(noise) * rng.standard_normal(len(latent))
    return (decision > np.median(decision)).astype(int)


def make_world(world, seed, n=360):
    """Return oriented rows, labels, truth latent, and planted feature roles."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)

    if world == "smooth_signal":
        labels = balanced_labels(g, 0.75, rng)
        rows = [g + sigma * rng.standard_normal(n)
                for sigma in (0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0)]
        rows += [0.55 * g + 1.15 * rng.standard_normal(n) for _ in range(2)]
        rows += [rng.standard_normal(n) for _ in range(2)]
        roles = ["signal"] * 8 + ["weak_signal"] * 2 + ["noise"] * 2

    elif world == "correlated_errors":
        labels = balanced_labels(g, 0.75, rng)
        shared = rng.standard_normal(n)
        rows = [g + 0.45 * rng.standard_normal(n) for _ in range(4)]
        rows += [g + 1.15 * shared + 0.3 * rng.standard_normal(n) for _ in range(4)]
        rows += [g - 0.8 * shared + 0.55 * rng.standard_normal(n) for _ in range(4)]
        roles = (["clean_signal"] * 4 + ["shared_error_positive"] * 4
                 + ["shared_error_negative"] * 4)

    elif world == "nuisance_manifold":
        labels = balanced_labels(g, 0.75, rng)
        nuisance = rng.standard_normal(n)
        # Equal-size rank-one blocks keep the signal inside U2 and ordinary IU-PCR
        # useful. The nuisance block is locally cleaner, so graph construction is
        # tempted to organize samples by irrelevant geometry.
        rows = [g + 0.55 * rng.standard_normal(n) for _ in range(6)]
        rows += [nuisance + 0.25 * rng.standard_normal(n) for _ in range(6)]
        roles = ["signal"] * 6 + ["nuisance"] * 6

    elif world == "subspace_exclusion":
        labels = balanced_labels(g, 0.75, rng)
        nuisance = rng.standard_normal(n)
        rows = [g + 0.65 * rng.standard_normal(n) for _ in range(4)]
        rows += [1.5 * nuisance + 0.25 * rng.standard_normal(n) for _ in range(8)]
        roles = ["signal"] * 4 + ["nuisance"] * 8

    elif world == "disconnected_graph":
        labels = balanced_labels(g, 0.75, rng)
        group = rng.choice((-1.0, 1.0), size=n)
        rows = [g + 0.55 * rng.standard_normal(n) for _ in range(4)]
        rows += [g + 2.8 * group + 0.35 * rng.standard_normal(n) for _ in range(4)]
        rows += [1.8 * group + 0.45 * rng.standard_normal(n) for _ in range(4)]
        roles = ["signal"] * 4 + ["signal_plus_cluster"] * 4 + ["cluster_only"] * 4

    elif world == "pure_noise":
        labels = balanced_labels(g, 0.75, rng)
        rows = [rng.standard_normal(n) for _ in range(12)]
        roles = ["noise"] * 12

    else:
        raise ValueError(f"unknown world: {world}")

    F = zscore_rows(np.vstack(rows))
    latent = (g - g.mean()) / (g.std() + EPS)
    return F, labels, latent, roles


def oracle_projected_weight(F, latent):
    """Supervised-in-simulation comparator restricted to IU-PCR's same U2."""
    m, n = F.shape
    C = F @ F.T / n
    values, U = eigh(C, subset_by_index=[m - 2, m - 1])
    U = U[:, np.argsort(values)[::-1]]
    Cp = U.T @ C @ U
    rho = F @ latent / n
    return U @ np.linalg.solve(Cp, U.T @ rho)


def cosine(left, right):
    return float(np.dot(left, right) /
                 (np.linalg.norm(left) * np.linalg.norm(right) + EPS))


def pearson(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.corrcoef(left, right)[0, 1])


def make_graphs(F, latent, gates, rng, k):
    dufs_graph = build_graph_from_features(F, gates=gates, k=k)
    shuffled = gates[rng.permutation(len(gates))]
    return {
        "dufs": dufs_graph,
        "ungated": build_graph_from_features(F, k=k),
        "random_gates": build_graph_from_features(F, gates=shuffled, k=k),
        "permuted_graph": permute_graph(dufs_graph, rng.permutation(F.shape[1])),
        "oracle_latent": self_tuning_knn_graph(latent[:, None], k=k),
    }, shuffled


def projected_ridge_path(F, baseline, lambdas, evaluation_laplacian):
    """Ordinary isotropic ridge in IU-PCR's U2, trace-matched to C."""
    m, n = F.shape
    C = F @ F.T / n
    values, U = eigh(C, subset_by_index=[m - 2, m - 1])
    U = U[:, np.argsort(values)[::-1]]
    Cp = 0.5 * (U.T @ C @ U + (U.T @ C @ U).T)
    ridge = np.eye(2) * np.trace(Cp) / 2.0
    rhs = U.T @ baseline.rho_hat
    zero_w = U @ np.linalg.solve(Cp, rhs)
    output = {}
    for lambda_ in lambdas:
        system = Cp + lambda_ * ridge
        w = baseline.w.copy() if lambda_ == 0.0 else U @ np.linalg.solve(system, rhs)
        scores = w @ F
        output[lambda_] = {
            "w": w,
            "projected_condition_number": float(np.linalg.cond(system)),
            "weight_cosine_vs_iu": cosine(w, baseline.w),
            "weight_norm": float(np.linalg.norm(w)),
            "score_variance": float(np.var(scores)),
            # The DUFS Laplacian is an evaluation diagnostic here, not an input.
            "score_laplacian_energy": float(
                scores @ (evaluation_laplacian @ scores) / n
            ),
            "roughness_min_eigenvalue": 0.0,
            "roughness_effective_rank": 2.0,
            "zero_equation_weight_error": float(
                np.max(np.abs(zero_w - baseline.w))
            ),
        }
    return output


def run_dataset(split, replicate, seed, world, args):
    F, labels, latent, roles = make_world(world, seed, n=args.n)
    gates, gate_diag = dufs_soft_gates(
        F, seeds=(11, 23, 37), epochs=args.dufs_epochs,
    )
    control_rng = np.random.default_rng(seed + 731_291)
    graphs, shuffled = make_graphs(F, latent, gates, control_rng, args.k)
    oracle_w = oracle_projected_weight(F, latent)
    rows = []
    graph_rows = []
    gate_feature_rows = []
    exact_errors = []

    graph_controls = tuple(control for control in CONTROLS
                           if control != "projected_ridge")
    laplacians = {
        control: symmetric_normalized_laplacian(graphs[control])
        for control in graph_controls
    }
    ungated_norm = float(np.sqrt(np.sum(laplacians["ungated"].data ** 2)))
    paths = {
        control: laplacian_iu_path(F, LAMBDAS, graph=graphs[control], k=args.k)
        for control in graph_controls
    }
    baseline = paths["dufs"][0.0].baseline
    ridge_path = projected_ridge_path(F, baseline, LAMBDAS, laplacians["dufs"])

    baseline_scores = None
    for control in CONTROLS:
        path = ridge_path if control == "projected_ridge" else paths[control]
        base_w = baseline.w if control == "projected_ridge" else path[0.0].w
        scores0 = base_w @ F
        if baseline_scores is None:
            baseline_scores = scores0
        exact_errors.append(float(np.max(np.abs(scores0 - baseline_scores))))
        energies = []
        for lambda_ in LAMBDAS:
            result = path[lambda_]
            if control == "projected_ridge":
                w, diag = result["w"], result
            else:
                w, diag = result.w, result.diagnostics
            scores = w @ F
            energies.append(diag["score_laplacian_energy"])
            rows.append({
                "version": VERSION,
                "split": split,
                "replicate": replicate,
                "seed": seed,
                "world": world,
                "control": control,
                "lambda": lambda_,
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                "latent_correlation": pearson(scores, latent),
                "oracle_weight_cosine": cosine(w, oracle_w),
                "weight_cosine_vs_iu": diag["weight_cosine_vs_iu"],
                "weight_norm": diag["weight_norm"],
                "effective_weight_count": float(
                    np.sum(np.abs(w)) ** 2 / (np.sum(w ** 2) + EPS)
                ),
                "negative_weight_count": int(np.sum(w < -1e-10)),
                "score_variance": diag["score_variance"],
                "score_laplacian_energy": diag["score_laplacian_energy"],
                "projected_condition_number": diag["projected_condition_number"],
                "roughness_min_eigenvalue": diag["roughness_min_eigenvalue"],
                "roughness_effective_rank": diag["roughness_effective_rank"],
                "zero_equation_weight_error": diag["zero_equation_weight_error"],
                "baseline_additive_projection_residual": baseline.proj_residual,
                "baseline_g2_fraction": baseline.g2_frac_of_var_y,
            })
        if control != "projected_ridge":
            graph_diag = path[0.0].diagnostics
            difference = laplacians[control] - laplacians["ungated"]
            graph_rows.append({
                "split": split,
                "replicate": replicate,
                "seed": seed,
                "world": world,
                "control": control,
                "n_edges": graph_diag["n_edges"],
                "n_components": graph_diag["n_components"],
                "degree_min": graph_diag["degree_min"],
                "degree_mean": graph_diag["degree_mean"],
                "degree_max": graph_diag["degree_max"],
                "algebraic_connectivity": graph_diag["algebraic_connectivity"],
                "graph_symmetry_error": graph_diag["graph_symmetry_error"],
                "roughness_min_eigenvalue": graph_diag["roughness_min_eigenvalue"],
                "laplacian_relative_distance_vs_ungated": float(
                    np.sqrt(np.sum(difference.data ** 2)) / (ungated_norm + EPS)
                ),
                "energy_monotone": bool(np.all(np.diff(energies) <= 1e-9)),
            })

    raw = np.asarray(gate_diag["raw_probabilities"])
    per_seed = np.asarray(gate_diag["per_seed_probabilities"])
    gate_row = {
        "split": split,
        "replicate": replicate,
        "seed": seed,
        "world": world,
        "mean_probability": gate_diag["mean_probability"],
        "near_zero_fraction": gate_diag["near_zero_fraction"],
        "near_one_fraction": gate_diag["near_one_fraction"],
        "effective_feature_count": gate_diag["effective_feature_count"],
        "mean_seed_std": gate_diag["mean_seed_std"],
        "min_probability": float(raw.min()),
        "max_probability": float(raw.max()),
        "gate_order_stability": float(np.mean([
            np.corrcoef(per_seed[i], per_seed[j])[0, 1]
            for i in range(len(per_seed)) for j in range(i + 1, len(per_seed))
        ])),
        "shuffle_l2_distance": float(np.linalg.norm(gates - shuffled)),
        "lambda0_max_score_error": float(max(exact_errors)),
    }
    for feature, role in enumerate(roles):
        gate_feature_rows.append({
            "split": split,
            "replicate": replicate,
            "seed": seed,
            "world": world,
            "feature": feature,
            "planted_role": role,
            "gate_probability": float(raw[feature]),
            "relative_gate": float(gates[feature]),
            "seed_11_probability": float(per_seed[0, feature]),
            "seed_23_probability": float(per_seed[1, feature]),
            "seed_37_probability": float(per_seed[2, feature]),
        })
    return rows, graph_rows, gate_row, gate_feature_rows


def grouped(rows, **filters):
    values = []
    for row in rows:
        if all(row[key] == value for key, value in filters.items()):
            values.append(row)
    return values


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def one_sided_lower(mean, se, n, confidence=0.95):
    """Student-t lower confidence bound for an estimated mean."""
    if n < 2:
        return float(mean)
    return float(mean - student_t.ppf(confidence, df=n - 1) * se)


def add_baseline_deltas(rows):
    baselines = {}
    for row in rows:
        key = (row["split"], row["world"], row["replicate"])
        if row["control"] == "dufs" and row["lambda"] == 0.0:
            baselines[key] = row
    for row in rows:
        base = baselines[(row["split"], row["world"], row["replicate"])]
        row["auroc_delta"] = row["auroc"] - base["auroc"]
        row["auprc_delta"] = row["auprc"] - base["auprc"]
        row["cosine_delta"] = row["oracle_weight_cosine"] - base["oracle_weight_cosine"]
        row["energy_ratio"] = (
            row["score_laplacian_energy"] /
            (grouped(rows, split=row["split"], world=row["world"],
                     replicate=row["replicate"], control=row["control"],
                     **{"lambda": 0.0})[0]["score_laplacian_energy"] + EPS)
        )


def select_lambda(rows):
    """Smallest nonzero lambda within one SE of best smooth-world dev mean."""
    candidates = []
    for lambda_ in LAMBDAS:
        vals = [row["auroc_delta"] for row in grouped(
            rows, split="development", world="smooth_signal",
            control="dufs", **{"lambda": lambda_})]
        mean, se = mean_se(vals)
        candidates.append((lambda_, mean, se))
    best = max(candidates, key=lambda item: item[1])
    threshold = best[1] - best[2]
    eligible = [item[0] for item in candidates
                if item[0] > 0 and item[1] > 0 and item[1] >= threshold]
    return (min(eligible) if eligible else 0.0), candidates


def summaries(rows):
    output = []
    for split in ("development", "confirmation"):
        for world in WORLDS:
            for control in CONTROLS:
                for lambda_ in LAMBDAS:
                    subset = grouped(rows, split=split, world=world,
                                     control=control, **{"lambda": lambda_})
                    record = {"split": split, "world": world,
                              "control": control, "lambda": lambda_,
                              "n": len(subset)}
                    for metric in ("auroc", "auroc_delta", "auprc", "auprc_delta",
                                   "latent_correlation",
                                   "oracle_weight_cosine", "cosine_delta",
                                   "energy_ratio", "projected_condition_number"):
                        mean, se = mean_se([row[metric] for row in subset])
                        record[f"{metric}_mean"] = mean
                        record[f"{metric}_se"] = se
                    output.append(record)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_paths(summary, selected, out_dir):
    # Independent y-scales keep the small mechanism effects visible; the frozen
    # cross-world comparison with a shared scale is provided by plot 03.
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(23, 3.8), sharey=False)
    x = np.arange(len(LAMBDAS))
    for ax, world in zip(axes, WORLDS):
        for control in CONTROLS:
            points = grouped(summary, split="confirmation", world=world, control=control)
            points.sort(key=lambda row: LAMBDAS.index(row["lambda"]))
            mean = np.array([row["auroc_delta_mean"] for row in points]) * 100
            se = np.array([row["auroc_delta_se"] for row in points]) * 100
            ax.plot(x, mean, marker="o", ms=3, lw=1.5,
                    color=CONTROL_COLORS[control], label=CONTROL_LABELS[control])
            ax.fill_between(x, mean - se, mean + se,
                            color=CONTROL_COLORS[control], alpha=0.1)
        ax.axhline(0, color="#475569", lw=0.8)
        ax.axvline(LAMBDAS.index(selected), color="#111827", ls="--", lw=1)
        ax.set_title(WORLD_LABELS[world])
        ax.set_xticks(x)
        ax.set_xticklabels([str(value) for value in LAMBDAS], rotation=45)
        ax.set_xlabel("regularization λ")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("confirmation AUROC Δ (points)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Laplacian IU-PCR regularization paths (mean ± SE)", y=1.14,
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "01_confirmation_auroc_paths.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(summary, selected, out_dir):
    selected_controls = ("dufs", "projected_ridge", "permuted_graph", "oracle_latent")
    fig, axes = plt.subplots(2, len(WORLDS), figsize=(23, 7), sharex=True)
    x = np.arange(len(LAMBDAS))
    for column, world in enumerate(WORLDS):
        for control in selected_controls:
            points = grouped(summary, split="confirmation", world=world, control=control)
            points.sort(key=lambda row: LAMBDAS.index(row["lambda"]))
            axes[0, column].plot(
                x, [100 * row["cosine_delta_mean"] for row in points], marker="o",
                ms=3, color=CONTROL_COLORS[control], label=CONTROL_LABELS[control],
            )
            axes[1, column].plot(
                x, [row["energy_ratio_mean"] for row in points], marker="o",
                ms=3, color=CONTROL_COLORS[control], label=CONTROL_LABELS[control],
            )
        for row in range(2):
            axes[row, column].axvline(LAMBDAS.index(selected), color="#111827",
                                      ls="--", lw=1)
            axes[row, column].grid(alpha=0.2)
        axes[0, column].axhline(0, color="#475569", lw=0.8)
        axes[0, column].set_title(WORLD_LABELS[world])
        axes[1, column].set_xticks(x)
        axes[1, column].set_xticklabels([str(value) for value in LAMBDAS], rotation=45)
        axes[1, column].set_xlabel("λ")
    axes[0, 0].set_ylabel("oracle-weight cosine Δ (×100)")
    axes[1, 0].set_ylabel("score graph energy / λ=0")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Mechanism diagnosis: target alignment and enforced smoothness", y=1.075,
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_mechanism_diagnostics.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_selected(rows, selected, out_dir):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    width = 0.13
    x = np.arange(len(WORLDS))
    for index, control in enumerate(CONTROLS):
        means, errors = [], []
        for world in WORLDS:
            vals = [row["auroc_delta"] for row in grouped(
                rows, split="confirmation", world=world, control=control,
                **{"lambda": selected})]
            mean, se = mean_se(vals)
            means.append(100 * mean)
            errors.append(100 * se)
        center = (len(CONTROLS) - 1) / 2.0
        ax.bar(x + (index - center) * width, means, width, yerr=errors, capsize=2,
               color=CONTROL_COLORS[control], label=CONTROL_LABELS[control], alpha=0.9)
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([WORLD_LABELS[world] for world in WORLDS])
    ax.set_ylabel("confirmation AUROC Δ (points)")
    ax.set_title(f"Frozen synthetic choice λ={selected:g}: candidate and controls")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_frozen_lambda_confirmation.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_gate_diagnostics(gates, graphs, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    positions = np.arange(len(WORLDS))
    for position, world in zip(positions, WORLDS):
        subset = grouped(gates, split="confirmation", world=world)
        axes[0].scatter(np.full(len(subset), position),
                        [row["effective_feature_count"] for row in subset],
                        alpha=0.55, color="#2563eb", s=18)
        axes[1].scatter(np.full(len(subset), position),
                        [row["gate_order_stability"] for row in subset],
                        alpha=0.55, color="#0f766e", s=18)
        graph_subset = grouped(graphs, split="confirmation", world=world, control="dufs")
        axes[2].scatter(np.full(len(graph_subset), position),
                        [row["algebraic_connectivity"] for row in graph_subset],
                        alpha=0.55, color="#7c3aed", s=18)
        axes[3].scatter(np.full(len(graph_subset), position),
                        [row["laplacian_relative_distance_vs_ungated"]
                         for row in graph_subset],
                        alpha=0.55, color="#d97706", s=18)
    axes[0].set_ylabel("effective feature count (of 12)")
    axes[1].set_ylabel("mean cross-seed gate correlation")
    axes[2].set_ylabel("graph algebraic connectivity")
    axes[3].set_ylabel("||L_DUFS-L_1|| / ||L_1||")
    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels([WORLD_LABELS[world] for world in WORLDS], rotation=30,
                           ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("DUFS gate and graph diagnostics (confirmation seeds)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_gate_graph_diagnostics.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_gate_roles(gate_features, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, world in zip(axes.ravel(), WORLDS):
        subset = grouped(gate_features, split="confirmation", world=world)
        roles = list(dict.fromkeys(row["planted_role"] for row in subset))
        values = [[row["gate_probability"] for row in subset
                   if row["planted_role"] == role] for role in roles]
        ax.boxplot(values, tick_labels=[role.replace("_", "\n") for role in roles],
                   showfliers=False)
        ax.set_title(WORLD_LABELS[world])
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("continuous gate probability")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("What the adapted DUFS gates identify (confirmation seeds)",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "05_gate_probabilities_by_planted_role.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def evaluation(rows, graph_rows, gate_rows, selected):
    def contrast(world, control):
        vals = [row["auroc_delta"] for row in grouped(
            rows, split="confirmation", world=world, control=control,
            **{"lambda": selected})]
        return mean_se(vals)

    smooth_dufs = contrast("smooth_signal", "dufs")
    smooth_n = len(grouped(rows, split="confirmation", world="smooth_signal",
                           control="dufs", **{"lambda": selected}))
    noise_auc = [row["auroc"] for row in grouped(
        rows, split="confirmation", world="pure_noise", control="dufs",
        **{"lambda": selected})]
    nuisance = contrast("nuisance_manifold", "dufs")
    nuisance_baseline = mean_se([row["auroc"] for row in grouped(
        rows, split="confirmation", world="nuisance_manifold", control="dufs",
        **{"lambda": 0.0})])
    exact = max(row["lambda0_max_score_error"] for row in gate_rows)
    zero_equation = max(row["zero_equation_weight_error"] for row in rows)
    min_r = min(row["roughness_min_eigenvalue"] for row in graph_rows)
    monotone = all(row["energy_monotone"] for row in graph_rows)
    disconnected_connectivity = max(
        row["algebraic_connectivity"] for row in graph_rows
        if row["n_components"] > 1
    )
    smooth_by_control = {
        control: {row["replicate"]: row["auroc_delta"] for row in grouped(
            rows, split="confirmation", world="smooth_signal", control=control,
            **{"lambda": selected})}
        for control in CONTROLS
    }
    paired_specificity = {}
    for control in NON_ORACLE_CONTROLS:
        differences = [
            smooth_by_control["dufs"][replicate] - value
            for replicate, value in smooth_by_control[control].items()
        ]
        mean, se = mean_se(differences)
        paired_specificity[control] = {
            "mean": mean,
            "se": se,
            "one_sided_95_lower": one_sided_lower(mean, se, len(differences)),
        }
    smooth_lower = one_sided_lower(smooth_dufs[0], smooth_dufs[1], smooth_n)
    mechanism_pass = (
        selected > 0
        and smooth_dufs[0] > 0.005
        and smooth_lower > 0.0
        and all(item["one_sided_95_lower"] > 0.0
                for item in paired_specificity.values())
    )
    invariant_pass = (
        exact == 0.0 and zero_equation < 1e-10 and min_r >= -1e-8
        and monotone and disconnected_connectivity == 0.0
    )
    nuisance_n = len(grouped(rows, split="confirmation", world="nuisance_manifold",
                             control="dufs", **{"lambda": selected}))
    nuisance_lower = one_sided_lower(nuisance[0], nuisance[1], nuisance_n)
    nuisance_baseline_lower = one_sided_lower(
        nuisance_baseline[0], nuisance_baseline[1], nuisance_n
    )
    # The maximum tolerated harm is tied to (and cannot exceed) the predeclared
    # minimum meaningful positive-world benefit of 0.5 AUROC points.
    robustness_pass = nuisance_baseline_lower > 0.65 and nuisance_lower > -0.005
    return {
        "selected_lambda": selected,
        "lambda0_exact_max_score_error": exact,
        "lambda0_unforced_equation_max_weight_error": zero_equation,
        "roughness_min_eigenvalue": min_r,
        "all_energy_paths_monotone": monotone,
        "disconnected_graph_max_algebraic_connectivity": disconnected_connectivity,
        "smooth_dufs_delta_mean": smooth_dufs[0],
        "smooth_dufs_delta_se": smooth_dufs[1],
        "smooth_dufs_one_sided_95_lower": smooth_lower,
        "smooth_paired_specificity": paired_specificity,
        "nuisance_dufs_delta_mean": nuisance[0],
        "nuisance_dufs_delta_se": nuisance[1],
        "nuisance_dufs_one_sided_95_lower": nuisance_lower,
        "nuisance_baseline_auroc_mean": nuisance_baseline[0],
        "nuisance_baseline_auroc_se": nuisance_baseline[1],
        "nuisance_baseline_one_sided_95_lower": nuisance_baseline_lower,
        "pure_noise_auroc_mean": mean_se(noise_auc)[0],
        "invariant_gate_pass": bool(invariant_pass),
        "positive_mechanism_gate_pass": bool(mechanism_pass),
        "graph_identification_robustness_gate_pass": bool(robustness_pass),
        "overall_phase1_pass": bool(
            invariant_pass and mechanism_pass and robustness_pass
        ),
    }


def write_report(path, evaluation_result, selection_path, rows, args):
    selected = evaluation_result["selected_lambda"]
    lines = [
        "# Laplacian IU-PCR synthetic falsification study",
        "",
        f"Version: `{VERSION}`",
        "",
        "## Scope and leakage boundary",
        "",
        "This phase used generated data only; no hallucination artifact bundle or real",
        "benchmark was read. Development and confirmation were executed as separate",
        "commands. The first command persisted a frozen lambda and source/config hashes;",
        "the second refused mismatches before opening disjoint confirmation seeds.",
        "Labels are evaluation-only. `oracle_latent` is a synthetic mechanism ceiling,",
        "never a deployable method.",
        "",
        "## Frozen design",
        "",
        f"- Development replicates per world: {args.dev_replicates}",
        f"- Confirmation replicates per world: {args.confirm_replicates}",
        f"- Samples per replicate: {args.n}; features: 12; k-NN: {args.k}",
        f"- Lambda grid: {list(LAMBDAS)}",
        f"- Gate optimization: 3 seeds, {args.dufs_epochs} epochs each",
        "- Choice rule: smallest positive lambda within one SE of the best positive",
        "  smooth-signal development mean; otherwise lambda=0.",
        "- The gate learner is the repository's parameter-free adapted DUFS: its kernel,",
        "  optimizer, and CPU budget differ from paper-faithful DUFS. Its continuous gates",
        "  feed a separate symmetric sparse k-NN graph; no feature is deleted.",
        "- `projected_ridge` is trace-matched isotropic ridge in the identical U2 and",
        "  isolates graph geometry from generic regularization.",
        "",
        "## Development choice",
        "",
        "| lambda | mean AUROC delta | SE |",
        "|---:|---:|---:|",
    ]
    for lambda_, mean, se in selection_path:
        lines.append(f"| {lambda_:g} | {100 * mean:+.3f} pp | {100 * se:.3f} pp |")
    lines += [
        "",
        f"Frozen choice: **lambda={selected:g}**.",
        "",
        "## Confirmation result at the frozen lambda",
        "",
        "| world | " + " | ".join(CONTROL_LABELS[c] for c in CONTROLS) + " |",
        "|---|" + "---:|" * len(CONTROLS),
    ]
    for world in WORLDS:
        cells = []
        for control in CONTROLS:
            mean, se = mean_se([row["auroc_delta"] for row in grouped(
                rows, split="confirmation", world=world, control=control,
                **{"lambda": selected})])
            cells.append(f"{100 * mean:+.3f} +/- {100 * se:.3f}")
        lines.append(f"| {WORLD_LABELS[world]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Cells are paired AUROC changes in percentage points +/- one SE relative to",
        "ordinary IU-PCR. Repeated lambda=0 arms are not independent observations.",
        "",
        "### Absolute performance and secondary AUPRC",
        "",
        "| world | IU AUROC | candidate AUROC | IU AUPRC | candidate AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for world in WORLDS:
        base = grouped(rows, split="confirmation", world=world, control="dufs",
                       **{"lambda": 0.0})
        candidate = grouped(rows, split="confirmation", world=world, control="dufs",
                            **{"lambda": selected})
        stats = [mean_se([row[metric] for row in subset])
                 for metric, subset in (("auroc", base), ("auroc", candidate),
                                        ("auprc", base), ("auprc", candidate))]
        cells = [f"{mean:.4f} +/- {se:.4f}" for mean, se in stats]
        lines.append(f"| {WORLD_LABELS[world]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### DUFS specificity on the positive control",
        "",
        "| comparator | paired advantage | SE | one-sided 95% lower |",
        "|---|---:|---:|---:|",
    ]
    for control in NON_ORACLE_CONTROLS:
        item = evaluation_result["smooth_paired_specificity"][control]
        lines.append(
            f"| {CONTROL_LABELS[control]} | {100 * item['mean']:+.3f} pp | "
            f"{100 * item['se']:.3f} pp | {100 * item['one_sided_95_lower']:+.3f} pp |"
        )
    oracle = {control: {row["replicate"]: row["auroc_delta"] for row in grouped(
        rows, split="confirmation", world="smooth_signal", control=control,
        **{"lambda": selected})} for control in ("dufs", "oracle_latent")}
    oracle_gap = [oracle["dufs"][rep] - value
                  for rep, value in oracle["oracle_latent"].items()]
    oracle_mean, oracle_se = mean_se(oracle_gap)
    lines += [
        "",
        f"The paired DUFS-minus-oracle gap was {100 * oracle_mean:+.3f} +/- "
        f"{100 * oracle_se:.3f} pp; a negative value is the remaining graph-identification gap.",
        "",
        "## Review-revised gates frozen before this rerun",
        "",
        f"- Algebraic/invariant gate: **{'PASS' if evaluation_result['invariant_gate_pass'] else 'FAIL'}**",
        f"  (exact-copy score error {evaluation_result['lambda0_exact_max_score_error']:.3e};",
        f"  unforced equation weight error {evaluation_result['lambda0_unforced_equation_max_weight_error']:.3e};",
        f"  minimum R eigenvalue {evaluation_result['roughness_min_eigenvalue']:.3e};",
        f"  disconnected connectivity {evaluation_result['disconnected_graph_max_algebraic_connectivity']:.3e};",
        f"  all Laplacian energy paths monotone: {evaluation_result['all_energy_paths_monotone']}).",
        f"- Positive mechanism and DUFS-specificity gate: **{'PASS' if evaluation_result['positive_mechanism_gate_pass'] else 'FAIL'}**.",
        "  It requires nonzero lambda, >0.5 pp mean smooth-signal gain with a positive",
        "  one-sided 95% lower bound, and a positive paired lower bound versus every",
        "  non-oracle control, including projected ridge.",
        f"- Graph-identification robustness gate: **{'PASS' if evaluation_result['graph_identification_robustness_gate_pass'] else 'FAIL'}**",
        f"  (baseline lower bound {evaluation_result['nuisance_baseline_one_sided_95_lower']:.4f}",
        f"  must exceed 0.65; delta lower bound {100 * evaluation_result['nuisance_dufs_one_sided_95_lower']:+.3f} pp",
        "  must exceed -0.5 pp, the same magnitude as the minimum meaningful gain).",
        f"- Overall Phase-1 gate: **{'PASS' if evaluation_result['overall_phase1_pass'] else 'FAIL'}**.",
        "",
        "The `Signal outside U2` world keeps the earlier chance-level construction as a",
        "separate limitation test: a final penalty restricted to fixed U2 cannot recover",
        "correctness signal that ordinary covariance excluded from that subspace.",
        "",
        "## Diagnostic artifacts",
        "",
        "- `01_confirmation_auroc_paths.png`: full confirmation paths.",
        "- `02_mechanism_diagnostics.png`: target alignment and smoothness.",
        "- `03_frozen_lambda_confirmation.png`: shared-scale frozen comparison.",
        "- `04_gate_graph_diagnostics.png`: stability, connectivity, and graph separation.",
        "- `05_gate_probabilities_by_planted_role.png`: per-role gate identification.",
        "- Raw CSVs retain AUROC/AUPRC, weight diagnostics, conditioning, ordinary additive residual,",
        "  per-feature gates, planted roles, graph diagnostics, and every path point.",
        "- `run_metadata.json` records commands, dependency versions, config, Git HEAD,",
        "  and hashes of the exact uncommitted source files.",
        "",
        "## Interpretation discipline",
        "",
        "Passing this phase establishes only a synthetic mechanism and failure boundary.",
        "It does not establish improvement on hallucination detection. Phase 2 remains",
        "closed until we discuss these results; later internal validation is not a pristine",
        "publication test, and genuinely unseen external data remains necessary.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def experiment_config(args):
    return {
        "version": VERSION,
        "dev_replicates": args.dev_replicates,
        "confirm_replicates": args.confirm_replicates,
        "n": args.n,
        "k": args.k,
        "dufs_epochs": args.dufs_epochs,
        "lambdas": list(LAMBDAS),
        "worlds": list(WORLDS),
        "controls": list(CONTROLS),
        "gate_seeds": [11, 23, 37],
        "confirmation_seed_offset": CONFIRMATION_SEED_OFFSET,
    }


def source_hashes():
    relative_paths = (
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/selectors/a2_groupfs.py",
        "scripts/laplacian_upcr_synthetic.py",
        "scripts/test_laplacian_upcr.py",
    )
    output = {}
    for relative in relative_paths:
        with open(os.path.join(REPO, relative), "rb") as handle:
            output[relative] = hashlib.sha256(handle.read()).hexdigest()
    return output


def run_metadata(args, frozen):
    import torch
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        git_head = "unavailable"
    return {
        "version": VERSION,
        "development_command": frozen["development_command"],
        "confirmation_command": list(sys.argv),
        "config": experiment_config(args),
        "source_hashes": source_hashes(),
        "git_head": git_head,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "torch": torch.__version__,
        },
    }


def run_split(split, count, offset, args):
    rows, graph_rows, gate_rows, gate_features = [], [], [], []
    total = count * len(WORLDS)
    completed = 0
    for replicate in range(count):
        for world_index, world in enumerate(WORLDS):
            seed = offset + replicate * 101 + world_index * 10_003
            batch, graph_batch, gate_row, feature_batch = run_dataset(
                split, replicate, seed, world, args,
            )
            rows.extend(batch)
            graph_rows.extend(graph_batch)
            gate_rows.append(gate_row)
            gate_features.extend(feature_batch)
            completed += 1
            print(f"[{completed:03d}/{total:03d}] {split} r={replicate} {world}",
                  flush=True)
    return rows, graph_rows, gate_rows, gate_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("development", "confirmation"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--dev-replicates", type=int, default=8)
    parser.add_argument("--confirm-replicates", type=int, default=8)
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--dufs-epochs", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dev_replicates < 2 or args.confirm_replicates < 2:
        raise ValueError("at least two replicates are required in each split")
    os.makedirs(args.out_dir, exist_ok=True)
    frozen_path = os.path.join(args.out_dir, "frozen_choice.json")
    development_path = os.path.join(args.out_dir, "development_artifacts.json")
    started = time.time()

    if args.stage == "development":
        rows, graphs, gates, gate_features = run_split(
            "development", args.dev_replicates, 10_000, args,
        )
        add_baseline_deltas(rows)
        selected, selection_path = select_lambda(rows)
        frozen = {
            "version": VERSION,
            "selected_lambda": selected,
            "selection_path": selection_path,
            "config": experiment_config(args),
            "source_hashes": source_hashes(),
            "development_command": list(sys.argv),
        }
        with open(frozen_path, "w", encoding="utf-8") as handle:
            json.dump(frozen, handle, indent=2, sort_keys=True)
        with open(development_path, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "graphs": graphs, "gates": gates,
                       "gate_features": gate_features}, handle)
        write_csv(os.path.join(args.out_dir, "development_per_run.csv"), rows)
        write_csv(os.path.join(args.out_dir, "development_graph_diagnostics.csv"), graphs)
        write_csv(os.path.join(args.out_dir, "development_gate_diagnostics.csv"), gates)
        write_csv(os.path.join(args.out_dir, "development_gate_features.csv"), gate_features)
        print(json.dumps({
            "stage": "development",
            "selected_lambda": selected,
            "selection_path": selection_path,
            "elapsed_seconds": time.time() - started,
            "next_command_requires_unchanged_config_and_sources": True,
        }, indent=2))
        return

    if not os.path.exists(frozen_path) or not os.path.exists(development_path):
        raise FileNotFoundError("run --stage development before confirmation")
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen["config"] != experiment_config(args):
        raise ValueError("confirmation config differs from frozen development config")
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("source files changed after lambda was frozen")
    with open(development_path, encoding="utf-8") as handle:
        development = json.load(handle)
    confirmation = run_split(
        "confirmation", args.confirm_replicates, CONFIRMATION_SEED_OFFSET, args,
    )
    rows = development["rows"] + confirmation[0]
    graph_rows = development["graphs"] + confirmation[1]
    gate_rows = development["gates"] + confirmation[2]
    gate_features = development["gate_features"] + confirmation[3]
    add_baseline_deltas(rows)
    selected = float(frozen["selected_lambda"])
    selection_path = [tuple(item) for item in frozen["selection_path"]]
    summary = summaries(rows)
    result = evaluation(rows, graph_rows, gate_rows, selected)
    result["elapsed_seconds_confirmation"] = time.time() - started
    result["development_selection_path"] = [
        {"lambda": value, "auroc_delta_mean": mean, "auroc_delta_se": se}
        for value, mean, se in selection_path
    ]

    write_csv(os.path.join(args.out_dir, "per_run.csv"), rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, "graph_diagnostics.csv"), graph_rows)
    write_csv(os.path.join(args.out_dir, "gate_diagnostics.csv"), gate_rows)
    write_csv(os.path.join(args.out_dir, "gate_features.csv"), gate_features)
    with open(os.path.join(args.out_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    metadata = run_metadata(args, frozen)
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    plot_paths(summary, selected, args.out_dir)
    plot_mechanism(summary, selected, args.out_dir)
    plot_selected(rows, selected, args.out_dir)
    plot_gate_diagnostics(gate_rows, graph_rows, args.out_dir)
    plot_gate_roles(gate_features, args.out_dir)
    write_report(os.path.join(args.out_dir, "REPORT.md"), result, selection_path,
                 rows, args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
