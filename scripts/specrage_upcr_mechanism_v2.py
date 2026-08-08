#!/usr/bin/env python3
"""Corrected, two-link SpecRaGE--IU-PCR synthetic mechanism study.

The study deliberately separates two questions that the v1 smoke run mixed:

Link A: can a label-free multi-view learner identify conditional view
        reliability and learn a useful joint representation?
Link B: if a useful graph is available, can it change the IU-PCR head in a
        target-helpful way?

Labels and planted latents are evaluator-only.  The candidate receives views,
feature scores, and registered numerical settings.  Calibration labels choose
one graph interface and one Laplacian strength on separate synthetic seeds;
the held-out seeds are evaluated once after that choice is frozen.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import types

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


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
    self_tuning_knn_graph,
    symmetric_normalized_laplacian,
)
from spectral_utils.specrage_laplacian import (  # noqa: E402
    SpecRaGEConfig,
    cross_view_agreement_targets,
    fit_specrage_graph,
    weighted_multiview_graph,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "specrage-liu-mechanism-v2-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "specrage_upcr_mechanism_v2")
LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SELECTABLE_LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SELECTABLE_INTERFACES = (
    "specrage_agreement_Y",
    "specrage_agreement_alpha",
)
EPS = 1e-12

# Fixed before held-out execution.  These coefficients create four regressor
# families with one target factor shared across families and a different
# correlated-error factor inside each family.
COUPLING_SIGNAL = np.array([0.369, 0.726, 0.260, 0.345, 0.955, 0.949, 0.397, 0.406])
COUPLING_NUISANCE = np.array([-0.899, 0.807, 2.663, -0.575, -2.888, -1.001, 2.422, 3.905])
COUPLING_NOISE = np.array([0.080, 0.165, 0.245, 0.106, 0.340, 0.225, 0.526, 0.336])

DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_metadata():
    import scipy
    import sklearn
    import torch

    sources = (
        "SPEC_SPECRAGE_LIU_V2.md",
        "spectral_utils/specrage_laplacian.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "scripts/specrage_upcr_mechanism_v2.py",
    )
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO, text=True,
        capture_output=True, check=False,
    )
    return {
        "source_sha256": {
            path: _sha256_file(os.path.join(REPO, path)) for path in sources
        },
        "git_head": completed.stdout.strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    }


def zscore_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    scale[scale < EPS] = 1.0
    return centered / scale


def _signal_view(g, rng, noise):
    return np.column_stack((
        g + noise * rng.standard_normal(len(g)),
        np.tanh(g) + (noise + 0.05) * rng.standard_normal(len(g)),
    ))


def make_reliability_world(seed, n):
    """Four-view majority world with one conditionally corrupted view/sample."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal(n)
    labels = (g + 0.70 * rng.standard_normal(n) > np.median(g)).astype(int)
    views = [_signal_view(g, rng, 0.25 + 0.04 * index) for index in range(4)]
    corruption = np.zeros((n, 4), dtype=bool)
    bad_view = rng.integers(0, 4, size=n)
    for sample, view in enumerate(bad_view):
        views[view][sample] = 3.0 * rng.standard_normal(2)
        corruption[sample, view] = True
    named = {f"view_{index}": value for index, value in enumerate(views)}
    return {
        "views": named,
        "g": g,
        "labels": labels,
        "corruption": corruption,
    }


def make_coupling_world(seed, n):
    """Four-family dependent-error world with known oracle LIU headroom."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal(n)
    family_nuisance = rng.standard_normal((4, n))
    labels = (g + 0.80 * rng.standard_normal(n) > np.median(g)).astype(int)
    features = np.vstack([
        COUPLING_SIGNAL[index] * g
        + COUPLING_NUISANCE[index] * family_nuisance[index // 2]
        + COUPLING_NOISE[index] * rng.standard_normal(n)
        for index in range(8)
    ])
    features = zscore_rows(features)
    views = {
        f"family_{family}": features[2 * family:2 * family + 2].T
        for family in range(4)
    }
    return {"F": features, "views": views, "g": g, "labels": labels}


def specrage_configs(epochs, batch_size):
    common = SpecRaGEConfig(
        output_dim=2,
        n_neighbors=15,
        temperature=90.0,
        learning_rate=1e-2,
        batch_size=batch_size,
        max_epochs=epochs,
        min_epochs=epochs,
        patience=epochs + 1,
        lr_patience=max(10, epochs // 3),
        encoder_hidden=(32,),
        fusion_hidden=(50,),
        checkpoint_mode="final",
        orthogonalization="svd_floor",
        orthogonal_floor=1e-3,
    )
    return {
        "plain": common,
        "agreement": replace(
            common,
            temperature=1.0,
            agreement_strength=2.0,
            agreement_temperature=0.08,
            edge_mass_strength=0.1,
        ),
        "uniform": replace(common, fusion_mode="uniform"),
    }


def _score_auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def _spearman(left, right):
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def _graph_energy(graph, signal):
    laplacian = symmetric_normalized_laplacian(graph)
    signal = np.asarray(signal, dtype=float)
    return float(signal @ (laplacian @ signal) / len(signal))


def _normalized_projected(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return matrix / max(float(np.trace(matrix)), EPS)


def _history_rows(result, *, link, partition, world_seed, arm):
    rows = []
    for seed_result in result.seed_results:
        for item in seed_result.history:
            rows.append({
                "link": link,
                "partition": partition,
                "world_seed": world_seed,
                "arm": arm,
                "model_seed": seed_result.seed,
                **item,
            })
    return rows


def run_reliability_gate(seeds, n, configs, model_seeds):
    rows, histories = [], []
    for world_seed in seeds:
        data = make_reliability_world(world_seed, n)
        for arm in ("plain", "agreement"):
            print(f"Link A seed={world_seed} arm={arm}", flush=True)
            result = fit_specrage_graph(
                data["views"], config=configs[arm], seeds=model_seeds
            )
            clean = (~data["corruption"]).astype(int).ravel()
            learned_auc = _score_auc(clean, result.alpha.ravel())
            target, _ = cross_view_agreement_targets(
                result.base_graphs,
                temperature=configs["agreement"].agreement_temperature,
            )
            target_auc = _score_auc(clean, target.ravel())
            embedding_r2 = np.mean([
                LinearRegression().fit(seed_result.embedding, data["g"]).score(
                    seed_result.embedding, data["g"]
                )
                for seed_result in result.seed_results
            ])
            rows.append({
                "world_seed": world_seed,
                "arm": arm,
                "reliability_auc": learned_auc,
                "agreement_target_auc": target_auc,
                "alpha_entropy": result.diagnostics["alpha_entropy_normalized"],
                "alpha_seed_mad": result.diagnostics["alpha_seed_mad"],
                "embedding_target_r2": float(embedding_r2),
                "embedding_graph_target_energy": _graph_energy(
                    result.embedding_graph, data["g"]
                ),
                "alpha_graph_target_energy": _graph_energy(result.graph, data["g"]),
                "optimizer_updates": result.seed_results[0].diagnostics[
                    "optimizer_updates"
                ],
                "orthogonal_condition_max": float(max(
                    max(item["orthogonal_condition_max"] for item in seed.history)
                    for seed in result.seed_results
                )),
                "orthogonal_clipped_fraction": float(np.mean([
                    item["orthogonal_clipped_fraction"]
                    for seed in result.seed_results for item in seed.history
                ])),
            })
            histories.extend(_history_rows(
                result, link="A", partition="mechanism", world_seed=world_seed,
                arm=arm,
            ))
    return rows, histories


def _fit_deployed(F):
    result = upcr_fit(F, **DEPLOYED_FIT)
    return result.w @ F


def run_coupling_partition(
    partition, seeds, n, configs, model_seeds, dufs_epochs
):
    rows, histories, diagnostics = [], [], []
    for world_seed in seeds:
        data = make_coupling_world(world_seed, n)
        F, labels, g = data["F"], data["labels"], data["g"]
        fits = {}
        for arm in ("plain", "agreement", "uniform"):
            print(f"Link B {partition} seed={world_seed} arm={arm}", flush=True)
            fits[arm] = fit_specrage_graph(
                data["views"], config=configs[arm], seeds=model_seeds
            )
            histories.extend(_history_rows(
                fits[arm], link="B", partition=partition,
                world_seed=world_seed, arm=arm,
            ))

        agreement = fits["agreement"]
        uniform_alpha = np.full_like(
            agreement.alpha, 1.0 / agreement.alpha.shape[1]
        )
        raw_uniform = weighted_multiview_graph(
            agreement.base_graphs, uniform_alpha
        )
        oracle_graph = self_tuning_knn_graph(g[:, None], k=15)
        gates, dufs_diagnostics = dufs_soft_gates(
            F, seeds=(0,), epochs=dufs_epochs
        )
        dufs_graph = build_graph_from_features(F, gates=gates, k=7)
        graphs = {
            "raw_uniform": raw_uniform,
            "specrage_plain_Y": fits["plain"].embedding_graph,
            "specrage_agreement_Y": agreement.embedding_graph,
            "specrage_agreement_alpha": agreement.graph,
            "specrage_uniform_Y": fits["uniform"].embedding_graph,
            "dufs_liu": dufs_graph,
            "oracle_target": oracle_graph,
        }
        paths = {
            name: laplacian_iu_path(F, LAMBDAS, graph=graph)
            for name, graph in graphs.items()
        }
        baseline_scores = next(iter(paths.values()))[0.0].baseline.w @ F
        baseline_auc = _score_auc(labels, baseline_scores)
        deployed_scores = _fit_deployed(F)
        deployed_auc = _score_auc(labels, deployed_scores)
        for lambda_ in LAMBDAS:
            rows.append({
                "partition": partition,
                "world_seed": world_seed,
                "arm": "iu",
                "lambda": lambda_,
                "auroc": baseline_auc,
                "delta_vs_iu_pp": 0.0,
                "score_spearman_vs_iu": 1.0,
            })
            rows.append({
                "partition": partition,
                "world_seed": world_seed,
                "arm": "deployed_upcr",
                "lambda": lambda_,
                "auroc": deployed_auc,
                "delta_vs_iu_pp": 100.0 * (deployed_auc - baseline_auc),
                "score_spearman_vs_iu": _spearman(
                    baseline_scores, deployed_scores
                ),
            })
            for arm, path in paths.items():
                result = path[lambda_]
                scores = result.w @ F
                auc = _score_auc(labels, scores)
                rows.append({
                    "partition": partition,
                    "world_seed": world_seed,
                    "arm": arm,
                    "lambda": lambda_,
                    "auroc": auc,
                    "delta_vs_iu_pp": 100.0 * (auc - baseline_auc),
                    "score_spearman_vs_iu": _spearman(
                        baseline_scores, scores
                    ),
                })

        uniform_path = paths["raw_uniform"]
        oracle_path = paths["oracle_target"]
        for lambda_ in SELECTABLE_LAMBDAS:
            uniform_scores = uniform_path[lambda_].w @ F
            oracle_scores = oracle_path[lambda_].w @ F
            projected_change = np.linalg.norm(
                _normalized_projected(
                    oracle_path[lambda_].projected_roughness_scaled
                )
                - _normalized_projected(
                    uniform_path[lambda_].projected_roughness_scaled
                )
            )
            diagnostics.append({
                "partition": partition,
                "world_seed": world_seed,
                "lambda": lambda_,
                "oracle_delta_vs_iu_pp": 100.0 * (
                    _score_auc(labels, oracle_scores) - baseline_auc
                ),
                "uniform_delta_vs_iu_pp": 100.0 * (
                    _score_auc(labels, uniform_scores) - baseline_auc
                ),
                "oracle_score_spearman_vs_iu": _spearman(
                    baseline_scores, oracle_scores
                ),
                "projected_roughness_orientation_change": float(projected_change),
                "oracle_target_energy": _graph_energy(oracle_graph, g),
                "agreement_Y_target_energy": _graph_energy(
                    agreement.embedding_graph, g
                ),
                "agreement_alpha_target_energy": _graph_energy(
                    agreement.graph, g
                ),
                "dufs_effective_feature_count": dufs_diagnostics[
                    "effective_feature_count"
                ],
            })
    return rows, histories, diagnostics


def choose_configuration(rows):
    calibration = [row for row in rows if row["partition"] == "calibration"]
    choices = []
    for arm in SELECTABLE_INTERFACES:
        for lambda_ in SELECTABLE_LAMBDAS:
            values = [
                row["delta_vs_iu_pp"] for row in calibration
                if row["arm"] == arm and float(row["lambda"]) == lambda_
            ]
            choices.append({
                "arm": arm,
                "lambda": lambda_,
                "mean_delta_vs_iu_pp": float(np.mean(values)),
                "worst_delta_vs_iu_pp": float(np.min(values)),
                "se_delta_vs_iu_pp": float(
                    np.std(values, ddof=1) / math.sqrt(len(values))
                ) if len(values) > 1 else 0.0,
            })
    # Use the registered one-standard-error rule: find the best calibration
    # mean, then choose the smallest lambda among candidates statistically
    # indistinguishable from it.  This prevents a monotone saturating path from
    # selecting an unnecessarily extreme value merely because it is last.
    choices.sort(
        key=lambda row: (
            row["mean_delta_vs_iu_pp"], row["worst_delta_vs_iu_pp"],
            -row["lambda"], row["arm"] == "specrage_agreement_Y",
        ),
        reverse=True,
    )
    best = choices[0]
    threshold = best["mean_delta_vs_iu_pp"] - best["se_delta_vs_iu_pp"]
    eligible = [
        row for row in choices if row["mean_delta_vs_iu_pp"] >= threshold
    ]
    eligible.sort(key=lambda row: (
        row["lambda"],
        row["arm"] != "specrage_agreement_Y",
        -row["worst_delta_vs_iu_pp"],
    ))
    chosen = dict(eligible[0])
    chosen["best_mean_delta_vs_iu_pp"] = best["mean_delta_vs_iu_pp"]
    chosen["one_se_threshold_pp"] = threshold
    return chosen, choices


def aggregate_primary(rows, chosen):
    selected = []
    definitions = (
        ("IU-PCR", "iu", 0.0),
        ("Deployed U-PCR", "deployed_upcr", 0.0),
        ("DUFS-LIU", "dufs_liu", 0.1),
        ("Raw-uniform LIU", "raw_uniform", chosen["lambda"]),
        ("Uniform SpecRaGE-Y LIU", "specrage_uniform_Y", chosen["lambda"]),
        ("Chosen SpecRaGE-derived LIU", chosen["arm"], chosen["lambda"]),
        ("Oracle target graph", "oracle_target", chosen["lambda"]),
    )
    for partition in ("calibration", "heldout"):
        for label, arm, lambda_ in definitions:
            values = [row for row in rows
                      if row["partition"] == partition
                      and row["arm"] == arm
                      and float(row["lambda"]) == float(lambda_)]
            selected.append({
                "partition": partition,
                "method": label,
                "arm": arm,
                "lambda": lambda_,
                "n": len(values),
                "mean_auroc": float(np.mean([row["auroc"] for row in values])),
                "mean_delta_vs_iu_pp": float(np.mean([
                    row["delta_vs_iu_pp"] for row in values
                ])),
                "std_delta_vs_iu_pp": float(np.std([
                    row["delta_vs_iu_pp"] for row in values
                ], ddof=1)) if len(values) > 1 else 0.0,
                "wins_vs_iu": int(np.sum([
                    row["delta_vs_iu_pp"] > 0 for row in values
                ])),
            })
    return selected


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(reliability, coupling, histories, primary, chosen, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, metric in enumerate(("reliability_auc", "embedding_target_r2")):
        arms = ("plain", "agreement")
        means = [np.mean([row[metric] for row in reliability if row["arm"] == arm])
                 for arm in arms]
        axes[index].bar(arms, means, color=("#9ca3af", "#2563eb"))
        for arm_index, arm in enumerate(arms):
            values = [row[metric] for row in reliability if row["arm"] == arm]
            axes[index].scatter([arm_index] * len(values), values, color="black", zorder=3)
        axes[index].grid(axis="y", alpha=0.2)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Clean-view reliability AUROC")
    axes[1].set_ylabel("Fused embedding linear R² with target latent")
    fig.suptitle("Link A: label-free reliability and representation")
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "link_a_reliability.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_arms = (
        "raw_uniform", "specrage_uniform_Y", "specrage_agreement_Y",
        "specrage_agreement_alpha", "dufs_liu", "oracle_target",
    )
    for axis, partition in zip(axes, ("calibration", "heldout")):
        for arm in plot_arms:
            means = [np.mean([
                row["delta_vs_iu_pp"] for row in coupling
                if row["partition"] == partition and row["arm"] == arm
                and float(row["lambda"]) == lambda_
            ]) for lambda_ in LAMBDAS]
            axis.plot(LAMBDAS, means, marker="o", label=arm)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(chosen["lambda"], color="#dc2626", linestyle="--", linewidth=1)
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set_title(partition.capitalize())
        axis.set_xlabel("Laplacian strength λ")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("AUROC change vs IU-PCR (points)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False)
    fig.suptitle("Link B: graph-to-IU-PCR actuation", y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    fig.savefig(os.path.join(figure_dir, "link_b_lambda_paths.png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    heldout = [row for row in primary if row["partition"] == "heldout"]
    fig, axis = plt.subplots(figsize=(10, 5))
    names = [row["method"] for row in heldout]
    values = [row["mean_delta_vs_iu_pp"] for row in heldout]
    errors = [row["std_delta_vs_iu_pp"] for row in heldout]
    colors = ["#2563eb" if "Chosen" in name else "#9ca3af" for name in names]
    axis.barh(names, values, xerr=errors, color=colors, alpha=0.9)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Held-out AUROC change vs IU-PCR (points)")
    axis.grid(axis="x", alpha=0.2)
    axis.set_title("Frozen calibration choice on held-out synthetic worlds")
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "heldout_comparison.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for arm, color in (("plain", "#9ca3af"), ("agreement", "#2563eb")):
        selected = [row for row in histories if row["link"] == "A" and row["arm"] == arm]
        updates = sorted({int(row["optimizer_updates"]) for row in selected})
        for metric, axis in (("training_loss", axes[0]),
                             ("orthogonal_condition_max", axes[1])):
            means = [np.mean([row[metric] for row in selected
                              if int(row["optimizer_updates"]) == update])
                     for update in updates]
            axis.plot(updates, means, label=arm, color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Training objective")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("QR/SVD condition number")
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Optimization convergence and numerical conditioning")
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "training_diagnostics.png"), dpi=180)
    plt.close(fig)


def render_report(reliability, primary, diagnostics, chosen, choices, metadata):
    rel = {}
    for arm in ("plain", "agreement"):
        selected = [row for row in reliability if row["arm"] == arm]
        rel[arm] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("reliability_auc", "agreement_target_auc",
                        "embedding_target_r2", "alpha_entropy")
        }
    heldout = [row for row in primary if row["partition"] == "heldout"]
    headroom = [row for row in diagnostics if row["partition"] == "heldout"
                and float(row["lambda"]) == float(chosen["lambda"])]
    lines = [
        "# Corrected SpecRaGE–IU-PCR mechanism study",
        "",
        f"Version: `{VERSION}`. Runtime: {metadata['runtime_seconds']:.1f}s.",
        "",
        "This study separates reliability learning (Link A) from downstream "
        "IU-PCR actuation (Link B). Calibration labels chose one interface and "
        "one value of lambda; held-out synthetic seeds were opened only afterward.",
        "",
        "## Link A — reliability",
        "",
        "![Reliability gate](figures/link_a_reliability.png)",
        "",
        "| arm | clean-view AUROC | agreement-target AUROC | embedding R² | alpha entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("plain", "agreement"):
        value = rel[arm]
        lines.append(
            f"| {arm} | {value['reliability_auc']:.3f} | "
            f"{value['agreement_target_auc']:.3f} | "
            f"{value['embedding_target_r2']:.3f} | {value['alpha_entropy']:.3f} |"
        )
    lines.extend([
        "",
        "## Link B — coupling",
        "",
        f"Calibration selected `{chosen['arm']}` with `lambda={chosen['lambda']}` "
        f"(mean calibration change {chosen['mean_delta_vs_iu_pp']:+.3f} points; "
        f"worst calibration replicate {chosen['worst_delta_vs_iu_pp']:+.3f}).",
        f"The grid was extended through `lambda=100`; the one-standard-error "
        f"threshold was {chosen['one_se_threshold_pp']:+.3f} points, so the "
        "smaller saturating value was retained instead of the boundary optimum.",
        "",
        "![Lambda paths](figures/link_b_lambda_paths.png)",
        "",
        "![Held-out comparison](figures/heldout_comparison.png)",
        "",
        "| held-out method | lambda | AUROC | change vs IU (pp) | wins |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in heldout:
        lines.append(
            f"| {row['method']} | {row['lambda']:.1f} | {row['mean_auroc']:.4f} | "
            f"{row['mean_delta_vs_iu_pp']:+.3f} | {row['wins_vs_iu']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## Oracle-headroom gate",
        "",
        f"At the frozen lambda, the oracle graph changes held-out AUROC by "
        f"{np.mean([row['oracle_delta_vs_iu_pp'] for row in headroom]):+.3f} points "
        f"and changes projected roughness orientation by "
        f"{np.mean([row['projected_roughness_orientation_change'] for row in headroom]):.3f} "
        "on average. This is the gate missing from v1: the learner is tested only "
        "where useful geometry can affect score ranks.",
        "",
        "## Numerical boundary",
        "",
        "The small-sample spectral networks still produce ill-conditioned raw "
        "outputs. Registered SVD singular-value flooring was therefore active and "
        "is reported in `reliability_gate.csv` and `training_history.csv`. This is "
        "a stabilization, not evidence that the released QR optimization is healthy.",
        "",
        "![Training diagnostics](figures/training_diagnostics.png)",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/specrage_upcr_mechanism_v2.py",
        "```",
        "",
        "The original v1 negative smoke artifacts remain unchanged.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dufs-epochs", type=int, default=40)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(11, 23))
    parser.add_argument("--reliability-seeds", type=int, nargs="+", default=(7100, 7101))
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=(5000, 5001, 5002))
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=(6000, 6001, 6002, 6003))
    args = parser.parse_args()
    if set(args.calibration_seeds) & set(args.heldout_seeds):
        raise ValueError("calibration and held-out seeds must be disjoint")
    started = time.time()
    configs = specrage_configs(args.epochs, args.batch_size)

    reliability, reliability_history = run_reliability_gate(
        args.reliability_seeds, args.n, configs, tuple(args.model_seeds)
    )
    calibration, calibration_history, calibration_diagnostics = \
        run_coupling_partition(
            "calibration", args.calibration_seeds, args.n, configs,
            tuple(args.model_seeds), args.dufs_epochs,
        )
    chosen, choices = choose_configuration(calibration)
    print(f"Frozen choice: {chosen}", flush=True)
    heldout, heldout_history, heldout_diagnostics = run_coupling_partition(
        "heldout", args.heldout_seeds, args.n, configs,
        tuple(args.model_seeds), args.dufs_epochs,
    )

    coupling = calibration + heldout
    histories = reliability_history + calibration_history + heldout_history
    diagnostics = calibration_diagnostics + heldout_diagnostics
    primary = aggregate_primary(coupling, chosen)
    metadata = {
        "version": VERSION,
        "runtime_seconds": time.time() - started,
        "n": args.n,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dufs_epochs": args.dufs_epochs,
        "model_seeds": list(args.model_seeds),
        "reliability_seeds": list(args.reliability_seeds),
        "calibration_seeds": list(args.calibration_seeds),
        "heldout_seeds": list(args.heldout_seeds),
        "lambdas": list(LAMBDAS),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "chosen": chosen,
        "provenance": provenance_metadata(),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "reliability_gate.csv"), reliability)
    write_csv(os.path.join(args.out_dir, "coupling_per_run.csv"), coupling)
    write_csv(os.path.join(args.out_dir, "oracle_headroom.csv"), diagnostics)
    write_csv(os.path.join(args.out_dir, "training_history.csv"), histories)
    write_csv(os.path.join(args.out_dir, "primary_summary.csv"), primary)
    write_csv(os.path.join(args.out_dir, "calibration_choices.csv"), choices)
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    make_plots(reliability, coupling, histories, primary, chosen, args.out_dir)
    report = render_report(
        reliability, primary, diagnostics, chosen, choices, metadata
    )
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
