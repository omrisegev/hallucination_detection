#!/usr/bin/env python3
"""Stage-gated synthetic mechanism study for SpecRaGE-LIU.

The candidate never receives labels or planted latent variables.  They remain
inside the evaluator and are joined only after every arm's score is frozen.

This runner supports a cheap smoke gate and a small registered development
grid.  Confirmation is intentionally absent: the development result must be
reviewed before additional seeds are opened.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types

import numpy as np
from scipy.linalg import eigh
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.laplacian_upcr import (                   # noqa: E402
    IU_FIT_DEFAULTS,
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)
from spectral_utils.specrage_laplacian import (               # noqa: E402
    SpecRaGEConfig,
    fit_specrage_graph,
    graph_for_control,
    weighted_multiview_graph,
)
from spectral_utils.upcr import upcr_fit                       # noqa: E402


VERSION = "specrage-liu-synthetic-v1-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "specrage_upcr_synthetic")
WORLDS = (
    "aligned_clean",
    "conditional_corruption",
    "global_corruption",
    "view_specific_nuisance",
    "shared_nuisance",
    "pure_noise",
)
WORLD_LABELS = {
    "aligned_clean": "Aligned clean views",
    "conditional_corruption": "Sample-specific corruption",
    "global_corruption": "Globally corrupted view",
    "view_specific_nuisance": "View-specific nuisance",
    "shared_nuisance": "Shared unmeasured nuisance",
    "pure_noise": "Pure noise",
}
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
PRIMARY_LAMBDA = 0.1
FROZEN_DUFS_K = 7
FROZEN_DUFS_LAMBDA = 0.1
CONTROLS = (
    "iu",
    "deployed_upcr",
    "dufs_liu",
    "specrage_sample",
    "specrage_global",
    "specrage_uniform",
    "specrage_permuted",
    "projected_ridge",
    "oracle_reliability",
    "oracle_target_graph",
)
DEVELOPMENT_CONFIGS = {
    "base": SpecRaGEConfig(),
    "local_graph": replace(SpecRaGEConfig(), n_neighbors=5),
    "broad_graph": replace(SpecRaGEConfig(), n_neighbors=11),
    "sharp_fusion": replace(SpecRaGEConfig(), temperature=1.0),
    "smooth_fusion": replace(SpecRaGEConfig(), temperature=100.0),
    "rank3": replace(SpecRaGEConfig(), output_dim=3),
}
DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}
EPS = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_metadata():
    import matplotlib
    import scipy
    import sklearn
    import torch

    sources = (
        "SPEC_SPECRAGE_LIU_V1.md",
        "spectral_utils/specrage_laplacian.py",
        "spectral_utils/specrage_views.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/selectors/a2_groupfs.py",
        "scripts/specrage_upcr_synthetic.py",
    )
    def git_output(*args):
        completed = subprocess.run(
            ("git", *args), cwd=REPO, text=True, capture_output=True, check=False
        )
        return completed.stdout.strip()
    return {
        "source_sha256": {
            path: sha256_file(os.path.join(REPO, path)) for path in sources
        },
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
    }


def zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centered / scale


def balanced_labels(latent, rng):
    decision = np.asarray(latent) + 0.70 * rng.standard_normal(len(latent))
    return (decision > np.median(decision)).astype(int)


def _signal_view(g, rng, noise=0.35):
    return np.column_stack([
        g + noise * rng.standard_normal(len(g)),
        np.tanh(g) + (noise + 0.05) * rng.standard_normal(len(g)),
    ])


def make_world(world, seed, n=320):
    """Return observed method inputs plus evaluator-only truth."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal(n)
    q = rng.standard_normal(n)
    labels = balanced_labels(g, rng)
    views = [_signal_view(g, rng, 0.30 + 0.05 * v) for v in range(4)]
    corruption = np.zeros((n, 4), dtype=bool)

    if world == "aligned_clean":
        pass
    elif world == "conditional_corruption":
        bad_view = rng.integers(0, 4, size=n)
        for i, view_index in enumerate(bad_view):
            views[view_index][i] = 3.0 * rng.standard_normal(2)
            corruption[i, view_index] = True
    elif world == "global_corruption":
        views[3] = np.column_stack([
            1.5 * q + 0.15 * rng.standard_normal(n),
            np.tanh(1.5 * q) + 0.15 * rng.standard_normal(n),
        ])
        corruption[:, 3] = True
    elif world == "view_specific_nuisance":
        affected = q > np.quantile(q, 0.35)
        views[0][affected] = np.column_stack([
            1.7 * q[affected] + 0.12 * rng.standard_normal(np.sum(affected)),
            np.tanh(1.7 * q[affected])
            + 0.12 * rng.standard_normal(np.sum(affected)),
        ])
        corruption[affected, 0] = True
    elif world == "shared_nuisance":
        views = [
            np.column_stack([
                1.8 * q + 0.25 * g + 0.15 * rng.standard_normal(n),
                np.tanh(1.8 * q) + 0.20 * g + 0.15 * rng.standard_normal(n),
            ])
            for _ in range(4)
        ]
        corruption[:] = True
    elif world == "pure_noise":
        views = [rng.standard_normal((n, 2)) for _ in range(4)]
        corruption[:] = True
    else:
        raise ValueError(f"unknown world: {world}")

    named_views = {
        f"view_{index}": zscore_columns(view)
        for index, view in enumerate(views)
    }
    # Every regressor belongs to exactly one declared view. This keeps the
    # SpecRaGE, DUFS and U-PCR feature contracts identical.
    fusion_matrix = np.column_stack(list(named_views.values()))
    fusion_matrix = zscore_columns(fusion_matrix).T

    oracle_alpha = np.ones((n, 4), dtype=float)
    if world not in ("shared_nuisance", "pure_noise"):
        oracle_alpha[corruption] = 0.0
    oracle_alpha /= oracle_alpha.sum(axis=1, keepdims=True)
    return {
        "views": named_views,
        "F": fusion_matrix,
        "labels": labels,
        "target_latent": g,
        "nuisance_latent": q,
        "corruption": corruption,
        "oracle_alpha": oracle_alpha,
    }


def cosine(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + EPS))


def pearson(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if np.std(left) < EPS or np.std(right) < EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def projected_ridge_path(F, baseline, evaluation_graph):
    m, n = F.shape
    covariance = F @ F.T / n
    values, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    basis = basis[:, np.argsort(values)[::-1]]
    projected = basis.T @ covariance @ basis
    projected = 0.5 * (projected + projected.T)
    ridge = np.eye(2) * np.trace(projected) / 2.0
    rhs = basis.T @ baseline.rho_hat
    laplacian = symmetric_normalized_laplacian(evaluation_graph)
    output = {}
    for lambda_ in LAMBDAS:
        if lambda_ == 0:
            weight = baseline.w.copy()
        else:
            weight = basis @ np.linalg.solve(projected + lambda_ * ridge, rhs)
        scores = weight @ F
        output[lambda_] = {
            "w": weight,
            "scores": scores,
            "condition_number": float(np.linalg.cond(projected + lambda_ * ridge)),
            "energy": float(scores @ (laplacian @ scores) / n),
        }
    return output


def _fit_graph_paths(F, graphs):
    return {
        name: laplacian_iu_path(F, LAMBDAS, graph=graph)
        for name, graph in graphs.items()
    }


def run_one(world, replicate, seed, config_name, config, args, dufs_cache):
    data = make_world(world, seed, n=args.n)
    F = data["F"]
    spec = fit_specrage_graph(
        data["views"], config=config, seeds=tuple(args.model_seeds)
    )
    graphs = {
        "specrage_sample": spec.graph,
        "specrage_global": graph_for_control(spec, "global"),
        "specrage_uniform": graph_for_control(spec, "uniform"),
        "specrage_permuted": graph_for_control(
            spec, "permuted", seed=seed + 700_001
        ),
        "oracle_reliability": weighted_multiview_graph(
            spec.base_graphs, data["oracle_alpha"]
        ),
        "oracle_target_graph": self_tuning_knn_graph(
            data["target_latent"][:, None], k=config.n_neighbors
        ),
    }
    cache_key = (world, replicate, seed)
    if cache_key not in dufs_cache:
        dufs_cache[cache_key] = dufs_soft_gates(
            F, seeds=tuple(args.dufs_seeds), epochs=args.dufs_epochs
        )
    gates, gate_diagnostics = dufs_cache[cache_key]
    graphs["dufs_liu"] = build_graph_from_features(
        F, gates=gates, k=FROZEN_DUFS_K
    )
    paths = _fit_graph_paths(F, graphs)
    baseline = paths["specrage_sample"][0.0].baseline
    ridge_path = projected_ridge_path(F, baseline, spec.graph)
    deployed = upcr_fit(F, **DEPLOYED_FIT)
    arm_scores = {
        "iu": {lambda_: baseline.w @ F for lambda_ in LAMBDAS},
        "deployed_upcr": {lambda_: deployed.w @ F for lambda_ in LAMBDAS},
        "projected_ridge": {
            lambda_: ridge_path[lambda_]["scores"] for lambda_ in LAMBDAS
        },
    }
    for arm, path in paths.items():
        arm_scores[arm] = {
            lambda_: path[lambda_].w @ F for lambda_ in LAMBDAS
        }

    rows = []
    iu_auc = roc_auc_score(data["labels"], baseline.w @ F)
    iu_scores = baseline.w @ F
    iu_ranks = rankdata(iu_scores) / len(iu_scores)
    for arm in CONTROLS:
        for lambda_ in LAMBDAS:
            scores = arm_scores[arm][lambda_]
            if arm in paths:
                fitted = paths[arm][lambda_]
                weight = fitted.w
                diagnostics = fitted.diagnostics
                condition = diagnostics["projected_condition_number"]
                energy = diagnostics["score_laplacian_energy"]
            elif arm == "projected_ridge":
                weight = ridge_path[lambda_]["w"]
                condition = ridge_path[lambda_]["condition_number"]
                energy = ridge_path[lambda_]["energy"]
            elif arm == "deployed_upcr":
                weight = deployed.w
                condition = float("nan")
                energy = float("nan")
            else:
                weight = baseline.w
                condition = float("nan")
                energy = float("nan")
            auc = float(roc_auc_score(data["labels"], scores))
            score_ranks = rankdata(scores) / len(scores)
            rows.append({
                "version": VERSION,
                "world": world,
                "replicate": replicate,
                "seed": seed,
                "config": config_name,
                "config_hash": config.fingerprint,
                "arm": arm,
                "lambda": lambda_,
                "auroc": auc,
                "delta_vs_iu_pp": 100 * (auc - iu_auc),
                "auprc": float(average_precision_score(data["labels"], scores)),
                "target_correlation": pearson(scores, data["target_latent"]),
                "nuisance_correlation": pearson(scores, data["nuisance_latent"]),
                "weight_cosine_vs_iu": cosine(weight, baseline.w),
                "weight_norm": float(np.linalg.norm(weight)),
                "score_variance": float(np.var(scores)),
                "score_spearman_vs_iu": pearson(score_ranks, iu_ranks),
                "mean_abs_percentile_rank_change_vs_iu": float(
                    np.mean(np.abs(score_ranks - iu_ranks))
                ),
                "condition_number": condition,
                "score_laplacian_energy": energy,
            })

    corruption = data["corruption"]
    corrupted_weights = spec.alpha[corruption]
    clean_weights = spec.alpha[~corruption]
    reliability_margin = float("nan")
    detection_auc = float("nan")
    if corrupted_weights.size and clean_weights.size:
        reliability_margin = float(clean_weights.mean() - corrupted_weights.mean())
        target = (~corruption).astype(int).ravel()
        if len(np.unique(target)) == 2:
            detection_auc = float(roc_auc_score(target, spec.alpha.ravel()))
    seed_scores = []
    for seed_result in spec.seed_results:
        seed_fit = laplacian_iu_path(
            F, (PRIMARY_LAMBDA,), graph=seed_result.graph
        )[PRIMARY_LAMBDA]
        seed_scores.append(seed_fit.w @ F)
    seed_score_correlations = [
        pearson(seed_scores[left], seed_scores[right])
        for left in range(len(seed_scores))
        for right in range(left + 1, len(seed_scores))
    ]
    diagnostic = {
        "world": world,
        "replicate": replicate,
        "seed": seed,
        "config": config_name,
        "config_hash": config.fingerprint,
        "alpha_entropy_normalized": spec.diagnostics["alpha_entropy_normalized"],
        "alpha_seed_mad": spec.diagnostics["alpha_seed_mad"],
        "alpha_seed_std_mean": spec.diagnostics["alpha_seed_std_mean"],
        "graph_seed_relative_distance_mean": spec.diagnostics[
            "graph_seed_relative_distance_mean"
        ],
        "score_seed_correlation_min": float(min(seed_score_correlations))
            if seed_score_correlations else 1.0,
        "base_edge_jaccard_mean": spec.diagnostics["base_edge_jaccard_mean"],
        "effective_edge_fraction": spec.diagnostics["effective_edge_fraction"],
        "degree_p05_over_mean": spec.diagnostics["degree_p05_over_mean"],
        "near_isolated_fraction": spec.diagnostics["near_isolated_fraction"],
        "total_affinity_vs_uniform": spec.diagnostics[
            "total_affinity_vs_uniform"
        ],
        "reliability_margin": reliability_margin,
        "reliability_detection_auc": detection_auc,
        "mean_view_weight": spec.diagnostics["mean_view_weight"],
        "dominant_view_fraction": spec.diagnostics["dominant_view_fraction"],
        "graph_edges": spec.diagnostics["n_edges"],
        "graph_components": spec.diagnostics["n_components"],
        "dufs_effective_feature_count": gate_diagnostics["effective_feature_count"],
        "dufs_seed_std": gate_diagnostics["mean_seed_std"],
        "seed_diagnostics": [result.diagnostics for result in spec.seed_results],
    }
    histories = []
    for result in spec.seed_results:
        for item in result.history:
            histories.append({
                "world": world,
                "replicate": replicate,
                "seed": seed,
                "config": config_name,
                "model_seed": result.seed,
                **item,
            })
    alpha_payload = {
        "alpha": spec.alpha,
        "per_seed_alpha": np.stack([
            result.alpha for result in spec.seed_results
        ], axis=0),
        "corruption": data["corruption"],
        "view_names": np.asarray(spec.view_names),
        "sample_score": paths["specrage_sample"][PRIMARY_LAMBDA].w @ F,
        "iu_score": baseline.w @ F,
    }
    return rows, diagnostic, histories, alpha_payload


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values, namespace, count=10000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(int(count), len(values)))
    means = values[index].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def summarize(rows):
    primary = [row for row in rows if float(row["lambda"]) == PRIMARY_LAMBDA]
    output = []
    for config in sorted({row["config"] for row in primary}):
        for world in WORLDS:
            for arm in CONTROLS:
                selected = [row for row in primary if row["config"] == config
                            and row["world"] == world and row["arm"] == arm]
                values = np.asarray([row["delta_vs_iu_pp"] for row in selected])
                lo, hi = bootstrap_ci(values, f"{config}-{world}-{arm}")
                output.append({
                    "config": config,
                    "world": world,
                    "arm": arm,
                    "n": len(selected),
                    "mean_auroc": float(np.mean([row["auroc"] for row in selected])),
                    "mean_delta_vs_iu_pp": float(np.mean(values)),
                    "ci95_low_pp": lo,
                    "ci95_high_pp": hi,
                    "wins_vs_iu": int(np.sum(values > 0)),
                    "losses_vs_iu": int(np.sum(values < 0)),
                })
    return output


def make_plots(rows, diagnostics, histories, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    primary_config = "base" if any(row["config"] == "base" for row in rows) \
        else sorted({row["config"] for row in rows})[0]
    selected_arms = (
        "deployed_upcr", "dufs_liu", "specrage_sample",
        "specrage_global", "specrage_uniform", "specrage_permuted",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, world in zip(axes.ravel(), WORLDS):
        for arm in selected_arms:
            means = []
            errors = []
            for lambda_ in LAMBDAS:
                values = [row["delta_vs_iu_pp"] for row in rows
                          if row["config"] == primary_config
                          and row["world"] == world and row["arm"] == arm
                          and float(row["lambda"]) == lambda_]
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1) / np.sqrt(len(values))
                              if len(values) > 1 else 0.0)
            axis.errorbar(LAMBDAS, means, yerr=errors, marker="o", label=arm)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(WORLD_LABELS[world])
        axis.set_xscale("symlog", linthresh=0.01)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Laplacian strength lambda")
    axes[1, 1].set_xlabel("Laplacian strength lambda")
    axes[1, 2].set_xlabel("Laplacian strength lambda")
    axes[0, 0].set_ylabel("AUROC change vs IU-PCR (pp)")
    axes[1, 0].set_ylabel("AUROC change vs IU-PCR (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("SpecRaGE-LIU mechanism and failure paths", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "lambda_paths.png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for world in WORLDS:
        chosen = [row for row in diagnostics
                  if row["config"] == primary_config and row["world"] == world]
        axes[0].scatter(
            [WORLD_LABELS[world]] * len(chosen),
            [row["alpha_entropy_normalized"] for row in chosen],
            alpha=0.7,
        )
        finite = [row["reliability_detection_auc"] for row in chosen
                  if np.isfinite(row["reliability_detection_auc"])]
        axes[1].scatter([WORLD_LABELS[world]] * len(finite), finite, alpha=0.7)
    axes[0].set_ylabel("Normalized view-weight entropy")
    axes[1].set_ylabel("Clean-vs-corrupt reliability AUROC")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("What SpecRaGE relies on")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "reliability_diagnostics.png"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for world in WORLDS:
        selected = [row for row in histories
                    if row["config"] == primary_config and row["world"] == world]
        epochs = sorted({int(row["epoch"]) for row in selected})
        means = [np.mean([row["loss"] for row in selected
                          if int(row["epoch"]) == epoch]) for epoch in epochs]
        axis.plot(epochs, means, label=WORLD_LABELS[world])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Self-supervised weighted Rayleigh loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("SpecRaGE training convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_convergence.png"), dpi=180)
    plt.close(fig)


def render_report(summary, diagnostics, metadata):
    configs = sorted({row["config"] for row in summary})
    primary = "base" if "base" in configs else configs[0]
    lookup = {(row["config"], row["world"], row["arm"]): row for row in summary}
    lines = [
        "# SpecRaGE-LIU synthetic mechanism study",
        "",
        f"Version: `{VERSION}`. Stage: `{metadata['stage']}`.",
        "",
        "The SpecRaGE learner receives only provenance views. Labels and planted latents are "
        "joined after every score is frozen. Values below are paired AUROC-point changes "
        f"versus ordinary IU-PCR at `lambda={PRIMARY_LAMBDA}`.",
        "",
        "![Regularization paths](figures/lambda_paths.png)",
        "",
        "## Primary configuration",
        "",
        "| world | deployed U-PCR | DUFS-LIU | SpecRaGE sample | global | uniform | permuted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    arms = (
        "deployed_upcr", "dufs_liu", "specrage_sample",
        "specrage_global", "specrage_uniform", "specrage_permuted",
    )
    for world in WORLDS:
        values = [lookup[primary, world, arm]["mean_delta_vs_iu_pp"] for arm in arms]
        lines.append(
            f"| {WORLD_LABELS[world]} | " + " | ".join(f"{value:+.3f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "## Mechanism diagnostics",
        "",
        "![Reliability diagnostics](figures/reliability_diagnostics.png)",
        "",
        "![Training convergence](figures/training_convergence.png)",
        "",
        "| world | alpha entropy | seed MAD | clean/corrupt reliability AUROC |",
        "|---|---:|---:|---:|",
    ])
    for world in WORLDS:
        selected = [row for row in diagnostics
                    if row["config"] == primary and row["world"] == world]
        detection = [row["reliability_detection_auc"] for row in selected
                     if np.isfinite(row["reliability_detection_auc"])]
        lines.append(
            f"| {WORLD_LABELS[world]} | "
            f"{np.mean([row['alpha_entropy_normalized'] for row in selected]):.3f} | "
            f"{np.mean([row['alpha_seed_mad'] for row in selected]):.4f} | "
            f"{np.mean(detection):.3f} |" if detection else
            f"| {WORLD_LABELS[world]} | "
            f"{np.mean([row['alpha_entropy_normalized'] for row in selected]):.3f} | "
            f"{np.mean([row['alpha_seed_mad'] for row in selected]):.4f} | n/a |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A gain is attributed to conditional reliability only if the sample-specific arm "
        "separates from global, uniform, and permuted controls and if learned weights identify "
        "the planted clean view. Shared-nuisance failure remains an explicit boundary.",
        "",
        "This report is generated mechanically. The registered Gate-E independent result "
        "review is stored separately and controls the final conclusion.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python scripts/specrage_upcr_synthetic.py --stage {metadata['stage']}",
        "```",
        "",
        f"Runtime: {metadata['runtime_seconds']:.1f}s.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--dufs-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.n = 128 if args.n is None else args.n
        args.replicates = 2 if args.replicates is None else args.replicates
        args.dufs_epochs = 20 if args.dufs_epochs is None else args.dufs_epochs
        args.model_seeds = (11,)
        args.dufs_seeds = (11,)
        configs = {
            "base": replace(
                DEVELOPMENT_CONFIGS["base"],
                batch_size=min(128, args.n),
                max_epochs=12,
                min_epochs=6,
                patience=5,
            )
        }
    else:
        args.n = 320 if args.n is None else args.n
        args.replicates = 6 if args.replicates is None else args.replicates
        args.dufs_epochs = 80 if args.dufs_epochs is None else args.dufs_epochs
        args.model_seeds = (11, 23)
        args.dufs_seeds = (11, 23, 37)
        configs = {
            name: replace(config, batch_size=min(config.batch_size, args.n))
            for name, config in DEVELOPMENT_CONFIGS.items()
        }

    started = time.time()
    rows, diagnostics, histories = [], [], []
    alpha_artifacts = {}
    dufs_cache = {}
    for config_name, config in configs.items():
        for world_index, world in enumerate(WORLDS):
            for replicate in range(args.replicates):
                seed = 4_100_000 + 10_000 * world_index + replicate
                result_rows, diagnostic, run_history, alpha_payload = run_one(
                    world, replicate, seed, config_name, config, args, dufs_cache
                )
                rows.extend(result_rows)
                diagnostics.append(diagnostic)
                histories.extend(run_history)
                prefix = f"{config_name}__{world}__replicate_{replicate}"
                for name, value in alpha_payload.items():
                    alpha_artifacts[f"{prefix}__{name}"] = value
                print({
                    "config": config_name,
                    "world": world,
                    "replicate": replicate,
                    "alpha_entropy": diagnostic["alpha_entropy_normalized"],
                    "reliability_auc": diagnostic["reliability_detection_auc"],
                }, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = summarize(rows)
    metadata = {
        "version": VERSION,
        "stage": args.stage,
        "runtime_seconds": time.time() - started,
        "n": args.n,
        "replicates": args.replicates,
        "model_seeds": list(args.model_seeds),
        "dufs_seeds": list(args.dufs_seeds),
        "dufs_epochs": args.dufs_epochs,
        "lambdas": list(LAMBDAS),
        "primary_lambda": PRIMARY_LAMBDA,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "provenance": provenance_metadata(),
    }
    write_csv(os.path.join(args.out_dir, f"{args.stage}_per_run.csv"), rows)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_history.csv"), histories)
    with open(os.path.join(args.out_dir, f"{args.stage}_diagnostics.json"),
              "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=True)
    with open(os.path.join(args.out_dir, f"{args.stage}_metadata.json"),
              "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.stage}_sample_reliance.npz"),
        **alpha_artifacts,
    )
    make_plots(rows, diagnostics, histories, args.out_dir)
    report = render_report(summary, diagnostics, metadata)
    with open(os.path.join(args.out_dir, f"{args.stage.upper()}_REPORT.md"),
              "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
