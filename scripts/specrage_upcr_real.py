#!/usr/bin/env python3
"""Grouped real-artifact calibration for SpecRaGE-LIU.

The method fit is label-free.  Correctness labels are joined only after scores
for every registered configuration are frozen.  Scientific hyperparameters are
selected across dataset/model cells with leave-one-family-out cross-fitting;
random rows from the same cell are never treated as independent validation.
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
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
)
from spectral_utils.specrage_laplacian import (               # noqa: E402
    SpecRaGEConfig,
    fit_specrage_graph,
    graph_for_control,
)
from spectral_utils.specrage_views import (                   # noqa: E402
    VIEW_SCHEMA_VERSION,
    fixed_stable_from_bundle,
    provenance_views,
    view_members,
)
from spectral_utils.upcr import upcr_fit                       # noqa: E402


VERSION = "specrage-liu-real-calibration-v2-2026-08-06"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "specrage_upcr_real")
LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
FROZEN_DUFS_K = 7
FROZEN_DUFS_LAMBDA = 0.1
V2_BASE = SpecRaGEConfig(
    output_dim=2,
    n_neighbors=15,
    temperature=1.0,
    learning_rate=1e-2,
    batch_size=128,
    max_epochs=60,
    min_epochs=60,
    patience=61,
    lr_patience=20,
    encoder_hidden=(32,),
    fusion_hidden=(50,),
    checkpoint_mode="final",
    orthogonalization="svd_floor",
    orthogonal_floor=1e-3,
    agreement_strength=2.0,
    agreement_temperature=0.08,
    edge_mass_strength=0.1,
)
CONFIGS = {
    "agreement_k15": V2_BASE,
    "agreement_k7": replace(V2_BASE, n_neighbors=7),
    "paper_plain": replace(
        V2_BASE,
        temperature=90.0,
        agreement_strength=0.0,
        edge_mass_strength=0.0,
    ),
}
CONFIG_COMPLEXITY = {
    "agreement_k15": 0,
    "agreement_k7": 1,
    "paper_plain": 2,
}
ARM_COMPLEXITY = {"specrage_embedding": 0, "specrage_sample": 1}
CANDIDATE_ARMS = tuple(ARM_COMPLEXITY)
ARMS = (
    "deployed_upcr",
    "iu",
    "dufs_liu",
    "specrage_sample",
    "specrage_embedding",
    "specrage_uniform_embedding",
    "specrage_global",
    "specrage_uniform",
    "specrage_permuted",
)
DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}
KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)
EPS = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_metadata(bundle):
    import matplotlib
    import scipy
    import sklearn
    import torch

    sources = (
        "SPEC_SPECRAGE_LIU_V1.md",
        "SPEC_SPECRAGE_LIU_V2.md",
        "spectral_utils/specrage_laplacian.py",
        "spectral_utils/specrage_views.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/feature_contract.py",
        "spectral_utils/selectors/a2_groupfs.py",
        "scripts/specrage_upcr_real.py",
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
        "input_sha256": sha256_file(bundle),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output("status", "--porcelain"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
    }


def family(cell):
    return next((name for name in KNOWN_FAMILIES if name in cell), cell)


def domain(cell):
    return "math" if any(name in cell for name in
                         ("gsm8k", "math500", "gpqa", "humaneval")) else "QA"


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_row(cell, config_name, arm, lambda_, scores, labels, extra=None):
    output = {
        "version": VERSION,
        "cell": cell,
        "family": family(cell),
        "domain": domain(cell),
        "config": config_name,
        "arm": arm,
        "lambda": float(lambda_),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "orientation_failure": bool(roc_auc_score(labels, scores) < 0.5),
        "score_variance": float(np.var(scores)),
    }
    if extra:
        output.update(extra)
    return output


def run_cell(cell, data, config_name, config, args, dufs_cache):
    stored = np.asarray(data[f"{cell}__V"], dtype=float)
    labels = np.asarray(data[f"{cell}__labels"], dtype=int)
    names = tuple(str(name) for name in data[f"{cell}__pool"])
    legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
    matrix, stable_names = fixed_stable_from_bundle(stored, names, legacy)
    views = provenance_views(matrix, stable_names)
    F = matrix.T
    spec = fit_specrage_graph(
        views, config=config, seeds=tuple(args.model_seeds)
    )
    uniform_config = replace(
        config,
        fusion_mode="uniform",
        agreement_strength=0.0,
        edge_mass_strength=0.0,
    )
    uniform_spec = fit_specrage_graph(
        views, config=uniform_config, seeds=tuple(args.model_seeds)
    )
    graphs = {
        "specrage_sample": spec.graph,
        "specrage_embedding": spec.embedding_graph,
        "specrage_uniform_embedding": uniform_spec.embedding_graph,
        "specrage_global": graph_for_control(spec, "global"),
        "specrage_uniform": graph_for_control(spec, "uniform"),
        "specrage_permuted": graph_for_control(
            spec, "permuted", seed=args.permutation_seed +
            int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
        ),
    }
    if cell not in dufs_cache:
        gates, gate_diagnostics = dufs_soft_gates(
            F, seeds=tuple(args.dufs_seeds), epochs=args.dufs_epochs
        )
        dufs_cache[cell] = (gates, gate_diagnostics)
    gates, gate_diagnostics = dufs_cache[cell]
    graphs["dufs_liu"] = build_graph_from_features(
        F, gates=gates, k=FROZEN_DUFS_K
    )
    paths = {
        arm: laplacian_iu_path(F, LAMBDAS, graph=graph)
        for arm, graph in graphs.items()
    }
    baseline = paths["specrage_sample"][0.0].baseline
    deployed = upcr_fit(F, **DEPLOYED_FIT)
    rows = []
    iu_scores = baseline.w @ F
    iu_ranks = rankdata(iu_scores) / len(iu_scores)
    graph_collapsed = bool(
        spec.diagnostics["effective_edge_fraction"] < 0.10
        or spec.diagnostics["degree_p05_over_mean"] < 0.01
        or spec.diagnostics["near_isolated_fraction"] > 0.05
    )
    for lambda_ in LAMBDAS:
        rows.append(score_row(
            cell, config_name, "iu", lambda_, baseline.w @ F, labels,
            {"weight_norm": float(np.linalg.norm(baseline.w)),
             "weight_cosine_vs_iu": 1.0},
        ))
        rows.append(score_row(
            cell, config_name, "deployed_upcr", lambda_, deployed.w @ F, labels,
            {"weight_norm": float(np.linalg.norm(deployed.w)),
             "weight_cosine_vs_iu": float(np.dot(deployed.w, baseline.w) /
                 (np.linalg.norm(deployed.w) * np.linalg.norm(baseline.w) + EPS))},
        ))
        for arm, path in paths.items():
            fitted = path[lambda_]
            scores = fitted.w @ F
            ranks = rankdata(scores) / len(scores)
            algebra_valid = bool(
                np.isfinite(scores).all()
                and np.isfinite(fitted.w).all()
                and fitted.diagnostics["zero_equation_weight_error"] < 1e-8
                and fitted.diagnostics["roughness_min_eigenvalue"] >= -1e-8
                and np.isfinite(fitted.diagnostics["projected_condition_number"])
            )
            rows.append(score_row(
                cell, config_name, arm, lambda_, scores, labels,
                {
                    "weight_norm": fitted.diagnostics["weight_norm"],
                    "weight_cosine_vs_iu": fitted.diagnostics["weight_cosine_vs_iu"],
                    "condition_number": fitted.diagnostics["projected_condition_number"],
                    "score_laplacian_energy": fitted.diagnostics["score_laplacian_energy"],
                    "graph_components": fitted.diagnostics["n_components"],
                    "graph_edges": fitted.diagnostics["n_edges"],
                    "algebra_valid": algebra_valid,
                    "graph_collapsed": graph_collapsed
                        if arm == "specrage_sample" else bool(
                            fitted.diagnostics["degree_min"] <= 0
                        ) if arm == "specrage_embedding" else False,
                    "score_spearman_vs_iu": float(np.corrcoef(ranks, iu_ranks)[0, 1]),
                    "mean_abs_percentile_rank_change_vs_iu": float(
                        np.mean(np.abs(ranks - iu_ranks))
                    ),
                },
            ))
    seed_scores = []
    for seed_result in spec.seed_results:
        fitted = laplacian_iu_path(
            F, (10.0,), graph=seed_result.graph
        )[10.0]
        seed_scores.append(fitted.w @ F)
    score_seed_correlations = [
        float(np.corrcoef(seed_scores[left], seed_scores[right])[0, 1])
        for left in range(len(seed_scores))
        for right in range(left + 1, len(seed_scores))
    ]
    diagnostics = {
        "cell": cell,
        "family": family(cell),
        "domain": domain(cell),
        "config": config_name,
        "config_hash": config.fingerprint,
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "view_schema": VIEW_SCHEMA_VERSION,
        "view_members": view_members(stable_names),
        "mean_view_weight": spec.diagnostics["mean_view_weight"],
        "dominant_view_fraction": spec.diagnostics["dominant_view_fraction"],
        "alpha_entropy_normalized": spec.diagnostics["alpha_entropy_normalized"],
        "alpha_seed_mad": spec.diagnostics["alpha_seed_mad"],
        "alpha_seed_std_mean": spec.diagnostics["alpha_seed_std_mean"],
        "graph_seed_relative_distance_mean": spec.diagnostics[
            "graph_seed_relative_distance_mean"
        ],
        "score_seed_correlation_min": float(min(score_seed_correlations))
            if score_seed_correlations else 1.0,
        "base_edge_jaccard_mean": spec.diagnostics["base_edge_jaccard_mean"],
        "effective_edge_fraction": spec.diagnostics["effective_edge_fraction"],
        "degree_p05_over_mean": spec.diagnostics["degree_p05_over_mean"],
        "near_isolated_fraction": spec.diagnostics["near_isolated_fraction"],
        "total_affinity_vs_uniform": spec.diagnostics[
            "total_affinity_vs_uniform"
        ],
        "graph_collapsed": graph_collapsed,
        "dufs_effective_feature_count": gate_diagnostics["effective_feature_count"],
        "dufs_seed_std": gate_diagnostics["mean_seed_std"],
        "seed_diagnostics": [result.diagnostics for result in spec.seed_results],
        "uniform_seed_diagnostics": [
            result.diagnostics for result in uniform_spec.seed_results
        ],
    }
    histories = []
    for result in spec.seed_results:
        for item in result.history:
            histories.append({
                "cell": cell,
                "family": family(cell),
                "config": config_name,
                "training_arm": "sample",
                "model_seed": result.seed,
                **item,
            })
    for result in uniform_spec.seed_results:
        for item in result.history:
            histories.append({
                "cell": cell,
                "family": family(cell),
                "config": config_name,
                "training_arm": "uniform",
                "model_seed": result.seed,
                **item,
            })
    reliance = {
        "alpha": spec.alpha,
        "per_seed_alpha": np.stack([
            result.alpha for result in spec.seed_results
        ], axis=0),
        "view_names": np.asarray(spec.view_names),
        "stable_feature_names": np.asarray(stable_names),
        "sample_score": paths["specrage_sample"][10.0].w @ F,
        "embedding_score": paths["specrage_embedding"][10.0].w @ F,
        "uniform_embedding_score": paths[
            "specrage_uniform_embedding"
        ][10.0].w @ F,
        "iu_score": baseline.w @ F,
    }
    return rows, diagnostics, histories, reliance


def candidate_deltas(rows, cells, config, lambda_, *,
                     candidate_arm="specrage_sample",
                     reference="deployed_upcr", reference_lambda=None):
    lookup = {(row["cell"], row["config"], row["arm"], float(row["lambda"])):
              row for row in rows}
    reference_lambda = lambda_ if reference_lambda is None else reference_lambda
    return np.asarray([
        lookup[cell, config, candidate_arm, lambda_]["auroc"]
        - lookup[cell, config, reference, reference_lambda]["auroc"]
        for cell in cells
    ])


def family_macro_values(cells, values):
    cells = list(cells)
    values = np.asarray(values, dtype=float)
    families = sorted({family(cell) for cell in cells})
    return np.asarray([
        np.mean([value for cell, value in zip(cells, values)
                 if family(cell) == heldout])
        for heldout in families
    ], dtype=float)


def select_one_standard_error(rows, cells):
    """Fail-closed family-macro one-standard-error selection."""
    lookup = {(row["cell"], row["config"], row["arm"], float(row["lambda"])):
              row for row in rows}
    candidates = []
    for config in sorted({row["config"] for row in rows}):
        for candidate_arm in CANDIDATE_ARMS:
            for lambda_ in LAMBDAS:
                delta = candidate_deltas(
                    rows, cells, config, lambda_, candidate_arm=candidate_arm
                )
                delta_family = family_macro_values(cells, delta)
                versus_dufs = candidate_deltas(
                    rows,
                    cells,
                    config,
                    lambda_,
                    candidate_arm=candidate_arm,
                    reference="dufs_liu",
                    reference_lambda=FROZEN_DUFS_LAMBDA,
                )
                versus_dufs_family = family_macro_values(cells, versus_dufs)
                sample_rows = [
                    lookup[cell, config, candidate_arm, lambda_]
                    for cell in cells
                ]
                mean = float(np.mean(delta_family))
                se = float(
                    np.std(delta_family, ddof=1) / np.sqrt(len(delta_family))
                ) if len(delta_family) > 1 else 0.0
                finite = bool(
                    np.isfinite(delta_family).all()
                    and np.isfinite(versus_dufs_family).all()
                    and all(np.isfinite(row["auroc"]) for row in sample_rows)
                )
                algebra_valid = bool(all(
                    row.get("algebra_valid", False) for row in sample_rows
                ))
                graph_valid = bool(not any(
                    row.get("graph_collapsed", True) for row in sample_rows
                ))
                orientation_failures = int(sum(
                    bool(row["orientation_failure"]) for row in sample_rows
                ))
                median_vs_dufs = float(np.median(versus_dufs_family))
                candidates.append({
                    "config": config,
                    "arm": candidate_arm,
                    "lambda": lambda_,
                    "mean_delta": mean,
                    "se": se,
                    "n_families": len(delta_family),
                    "median_delta_vs_frozen_dufs": median_vs_dufs,
                    "finite": finite,
                    "algebra_valid": algebra_valid,
                    "graph_valid": graph_valid,
                    "orientation_failures": orientation_failures,
                    "eligible": bool(
                        finite and algebra_valid and graph_valid
                        and orientation_failures == 0 and median_vs_dufs >= 0.0
                    ),
                })
    valid = [item for item in candidates if item["eligible"]]
    if not valid:
        return None, candidates
    best = max(valid, key=lambda item: item["mean_delta"])
    cutoff = best["mean_delta"] - best["se"]
    eligible = [item for item in valid if item["mean_delta"] >= cutoff]
    chosen = min(
        eligible,
        key=lambda item: (
            CONFIG_COMPLEXITY[item["config"]],
            ARM_COMPLEXITY[item["arm"]],
            item["lambda"],
            -item["mean_delta"],
        ),
    )
    return chosen, candidates


def lofo_selection(rows):
    families = sorted({row["family"] for row in rows})
    decisions, predictions = [], []
    lookup = {(row["cell"], row["config"], row["arm"], float(row["lambda"])):
              row for row in rows}
    all_cells = sorted({row["cell"] for row in rows})
    for heldout in families:
        training = [cell for cell in all_cells if family(cell) != heldout]
        test = [cell for cell in all_cells if family(cell) == heldout]
        chosen, _ = select_one_standard_error(rows, training)
        if chosen is None:
            decisions.append({
                "heldout_family": heldout,
                "training_cells": len(training),
                "chosen_config": "",
                "chosen_arm": "",
                "chosen_lambda": float("nan"),
                "training_delta_pp": float("nan"),
                "training_se_pp": float("nan"),
                "selection_failed": True,
                "failure_reason": (
                    "no candidate passed finite/algebra/graph/orientation/DUFS guards"
                ),
            })
            continue
        decisions.append({
            "heldout_family": heldout,
            "training_cells": len(training),
            "chosen_config": chosen["config"],
            "chosen_arm": chosen["arm"],
            "chosen_lambda": chosen["lambda"],
            "training_delta_pp": 100 * chosen["mean_delta"],
            "training_se_pp": 100 * chosen["se"],
            "selection_failed": False,
            "failure_reason": "",
        })
        for cell in test:
            spec = lookup[
                cell, chosen["config"], chosen["arm"], chosen["lambda"]
            ]
            deployed = lookup[cell, chosen["config"], "deployed_upcr", chosen["lambda"]]
            dufs = lookup[
                cell, chosen["config"], "dufs_liu", FROZEN_DUFS_LAMBDA
            ]
            if chosen["arm"] == "specrage_embedding":
                uniform_control = lookup[
                    cell, chosen["config"], "specrage_uniform_embedding",
                    chosen["lambda"]
                ]
                global_control = permuted_control = None
            else:
                global_control = lookup[
                    cell, chosen["config"], "specrage_global", chosen["lambda"]
                ]
                uniform_control = lookup[
                    cell, chosen["config"], "specrage_uniform", chosen["lambda"]
                ]
                permuted_control = lookup[
                    cell, chosen["config"], "specrage_permuted", chosen["lambda"]
                ]
            predictions.append({
                "cell": cell,
                "family": heldout,
                "domain": domain(cell),
                "chosen_config": chosen["config"],
                "chosen_arm": chosen["arm"],
                "chosen_lambda": chosen["lambda"],
                "specrage_auroc": spec["auroc"],
                "deployed_auroc": deployed["auroc"],
                "dufs_auroc": dufs["auroc"],
                "delta_vs_deployed_pp": 100 * (spec["auroc"] - deployed["auroc"]),
                "delta_vs_dufs_pp": 100 * (spec["auroc"] - dufs["auroc"]),
                "delta_vs_global_pp": 100 * (
                    spec["auroc"] - global_control["auroc"]
                ) if global_control is not None else float("nan"),
                "delta_vs_uniform_pp": 100 * (
                    spec["auroc"] - uniform_control["auroc"]
                ),
                "delta_vs_permuted_pp": 100 * (
                    spec["auroc"] - permuted_control["auroc"]
                ) if permuted_control is not None else float("nan"),
            })
    return decisions, predictions


def bootstrap_ci(values, namespace, count=20000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(count), len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def make_plots(rows, predictions, diagnostics, histories, out_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "figures")
    os.makedirs(plot_dir, exist_ok=True)
    cells = [row["cell"] for row in predictions]
    x = np.arange(len(cells))
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.bar(x - 0.2, [row["delta_vs_deployed_pp"] for row in predictions],
             width=0.4, label="SpecRaGE-LIU vs deployed")
    axis.bar(x + 0.2, [row["delta_vs_dufs_pp"] for row in predictions],
             width=0.4, label="SpecRaGE-LIU vs DUFS-LIU")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, cells, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Cross-fitted AUROC change (pp)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_title("Leave-one-family-out SpecRaGE-LIU calibration")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "crossfit_deltas.png"), dpi=180)
    plt.close(fig)

    view_names = sorted({name for row in diagnostics for name in row["mean_view_weight"]})
    matrix = np.asarray([
        [row["mean_view_weight"].get(name, 0.0) for name in view_names]
        for row in diagnostics
    ])
    fig, axis = plt.subplots(figsize=(10, max(5, 0.18 * len(diagnostics))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(view_names)), view_names, rotation=35, ha="right")
    axis.set_yticks(range(len(diagnostics)),
                    [f"{row['cell']}:{row['config']}" for row in diagnostics], fontsize=5)
    axis.set_title("Mean SpecRaGE view reliance")
    fig.colorbar(image, ax=axis, label="Mean alpha")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "view_reliance.png"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for config in sorted({row["config"] for row in histories}):
        selected = [row for row in histories if row["config"] == config]
        epochs = sorted({int(row["epoch"]) for row in selected})
        means = [np.mean([row["loss"] for row in selected
                          if int(row["epoch"]) == epoch]) for epoch in epochs]
        axis.plot(epochs, means, label=config)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Self-supervised loss")
    axis.set_title("SpecRaGE convergence across real cells")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_convergence.png"), dpi=180)
    plt.close(fig)


def render_report(predictions, decisions, final_choice, diagnostics, metadata):
    if metadata["stage"] == "smoke":
        lines = [
            "# SpecRaGE-LIU real-artifact smoke run",
            "",
            f"Version: `{VERSION}`.",
            "",
            "This is a fixed-configuration execution diagnostic on a small cell subset. "
            "It is not cross-fitted calibration and is not scientific performance evidence.",
            "",
            "| cell | SpecRaGE vs deployed (pp) | SpecRaGE vs frozen DUFS-LIU (pp) |",
            "|---|---:|---:|",
        ]
        for row in predictions:
            lines.append(
                f"| `{row['cell']}` | {row['delta_vs_deployed_pp']:+.3f} | "
                f"{row['delta_vs_dufs_pp']:+.3f} |"
            )
        lines.extend([
            "",
            "![View reliance](figures/view_reliance.png)",
            "",
            "![Training convergence](figures/training_convergence.png)",
            "",
            "Passing smoke authorizes only the registered development run.",
        ])
        return "\n".join(lines) + "\n"

    if metadata.get("selection_failed", False):
        failed_folds = [row for row in decisions if row.get("selection_failed", False)]
        lines = [
            "# SpecRaGE-LIU grouped real-artifact calibration",
            "",
            f"Version: `{VERSION}`. Stage: `development`.",
            "",
            "## Decision: calibration failed closed",
            "",
            "No baseline is promoted. At least one leave-one-family-out fold or the final "
            "all-development selection had no candidate passing the registered finite, "
            "algebra, graph-collapse, orientation, and frozen-DUFS guards.",
            "",
            f"Failed LOFO folds: `{len(failed_folds)}/{len(decisions)}`. "
            f"Final all-development candidate available: `{final_choice is not None}`.",
            "",
            "All raw scores, candidate rejection fields, convergence histories, sample-level "
            "reliance weights, diagnostics, and provenance hashes were retained for Gate-E review.",
            "",
            "![View reliance](figures/view_reliance.png)",
            "",
            "![Training convergence](figures/training_convergence.png)",
            "",
            "The runner exits with status 2 only after writing these artifacts.",
        ]
        return "\n".join(lines) + "\n"

    delta_deployed = np.asarray([row["delta_vs_deployed_pp"] for row in predictions])
    delta_dufs = np.asarray([row["delta_vs_dufs_pp"] for row in predictions])
    cells = [row["cell"] for row in predictions]
    family_deployed = family_macro_values(cells, delta_deployed)
    family_dufs = family_macro_values(cells, delta_dufs)
    lo_dep, hi_dep = bootstrap_ci(family_deployed, "specrage-real-deployed-family")
    lo_dufs, hi_dufs = bootstrap_ci(family_dufs, "specrage-real-dufs-family")
    mechanism = {
        control: family_macro_values(
            cells, np.asarray([row[f"delta_vs_{control}_pp"] for row in predictions])
        )
        for control in ("global", "uniform", "permuted")
    }
    lines = [
        "# SpecRaGE-LIU grouped real-artifact calibration",
        "",
        f"Version: `{VERSION}`. Stage: `{metadata['stage']}`.",
        "",
        "The per-cell learner is label-free. Labels select scientific hyperparameters only "
        "across other dataset/model families. These are cross-fitted development estimates, "
        "not confirmation on a new family.",
        "",
        "![Cross-fitted deltas](figures/crossfit_deltas.png)",
        "",
        "## Cross-fitted result",
        "",
        f"- SpecRaGE-LIU versus deployed U-PCR: **{np.mean(family_deployed):+.3f}pp**, "
        f"95% family-bootstrap CI [{lo_dep:+.3f}, {hi_dep:+.3f}], "
        f"{np.sum(delta_deployed > 0)}/{np.sum(delta_deployed < 0)} wins/losses.",
        f"- SpecRaGE-LIU versus frozen DUFS-LIU (`k={FROZEN_DUFS_K}`, "
        f"`lambda={FROZEN_DUFS_LAMBDA}`): **{np.mean(family_dufs):+.3f}pp**, "
        f"95% family-bootstrap CI [{lo_dufs:+.3f}, {hi_dufs:+.3f}], "
        f"{np.sum(delta_dufs > 0)}/{np.sum(delta_dufs < 0)} wins/losses.",
        f"- Worst change versus deployed U-PCR: **{np.min(delta_deployed):+.3f}pp**.",
        f"- Sample-specific minus global/uniform/permuted controls (family macro): "
        f"**{np.nanmean(mechanism['global']):+.3f} / "
        f"{np.mean(mechanism['uniform']):+.3f} / "
        f"{np.nanmean(mechanism['permuted']):+.3f}pp**. "
        "Global/permuted are not applicable when the embedding interface is selected.",
        "",
        "## Configuration chosen on all development cells",
        "",
        f"- graph configuration: `{final_choice['config']}`",
        f"- graph interface: `{final_choice['arm']}`",
        f"- lambda: `{final_choice['lambda']}`",
        f"- development mean change: `{100 * final_choice['mean_delta']:+.3f}pp`",
        "",
        "This configuration may be frozen for new-family confirmation. Its all-development "
        "number is not an unbiased performance estimate.",
        "",
        "## Leave-one-family-out choices",
        "",
        "| held-out family | configuration | interface | lambda | training change +/- SE (pp) |",
        "|---|---|---|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['heldout_family']} | `{row['chosen_config']}` | "
            f"`{row['chosen_arm']}` | "
            f"{row['chosen_lambda']:.2g} | {row['training_delta_pp']:+.3f} +/- "
            f"{row['training_se_pp']:.3f} |"
        )
    lines.extend([
        "",
        "## Reliance and convergence",
        "",
        "![View reliance](figures/view_reliance.png)",
        "",
        "![Training convergence](figures/training_convergence.png)",
        "",
        "Mean normalized alpha entropy: "
        f"`{np.mean([row['alpha_entropy_normalized'] for row in diagnostics]):.3f}`; "
        "mean seed alpha MAD: "
        f"`{np.mean([row['alpha_seed_mad'] for row in diagnostics]):.4f}`.",
        "",
        "## Claim boundary",
        "",
        "This stage can establish a development baseline and whether sample-specific weighting "
        "separates from DUFS/global/uniform/permuted controls. A publishable generalization "
        "claim requires the frozen configuration on newly collected families.",
        "",
        "The independent Gate-E result review is required before updating the research conclusion.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dufs-epochs", type=int, default=None)
    parser.add_argument("--permutation-seed", type=int, default=8_104_113)
    args = parser.parse_args()
    if args.stage == "smoke":
        args.max_cells = 2 if args.max_cells is None else args.max_cells
        args.dufs_epochs = 20 if args.dufs_epochs is None else args.dufs_epochs
        args.model_seeds = (11,)
        args.dufs_seeds = (11,)
        configs = {
            "agreement_k15": replace(
                CONFIGS["agreement_k15"],
                max_epochs=30,
                min_epochs=30,
                patience=31,
            )
        }
    else:
        args.dufs_epochs = 80 if args.dufs_epochs is None else args.dufs_epochs
        args.model_seeds = (11, 23)
        args.dufs_seeds = (11, 23, 37)
        configs = CONFIGS

    started = time.time()
    data = np.load(args.bundle, allow_pickle=True)
    cells = sorted({name.rsplit("__", 1)[0] for name in data.files})
    if args.max_cells is not None:
        cells = cells[:args.max_cells]
    rows, diagnostics, histories = [], [], []
    reliance_artifacts = {}
    dufs_cache = {}
    for config_name, config in configs.items():
        for cell in cells:
            cell_rows, cell_diagnostics, cell_histories, cell_reliance = run_cell(
                cell, data, config_name, config, args, dufs_cache
            )
            rows.extend(cell_rows)
            diagnostics.append(cell_diagnostics)
            histories.extend(cell_histories)
            prefix = f"{config_name}__{cell}"
            for name, value in cell_reliance.items():
                reliance_artifacts[f"{prefix}__{name}"] = value
            print({
                "cell": cell,
                "config": config_name,
                "alpha_entropy": cell_diagnostics["alpha_entropy_normalized"],
                "alpha_seed_mad": cell_diagnostics["alpha_seed_mad"],
            }, flush=True)

    if args.stage == "development":
        decisions, predictions = lofo_selection(rows)
        final_choice, candidates = select_one_standard_error(rows, cells)
        selection_failed = bool(
            final_choice is None
            or any(row.get("selection_failed", False) for row in decisions)
        )
    else:
        # Smoke verifies execution only; it must not fail or promote a method
        # based on two cells. Use the registered base configuration verbatim.
        decisions = []
        candidates = []
        final_choice = {
            "config": "agreement_k15",
            "arm": "specrage_sample",
            "lambda": 10.0,
            "mean_delta": float("nan"),
            "se": float("nan"),
            "eligible": False,
            "smoke_only": True,
        }
        selection_failed = False
        lookup = {(row["cell"], row["arm"], float(row["lambda"])): row
                  for row in rows if row["config"] == "agreement_k15"}
        predictions = []
        for cell in cells:
            spec = lookup[cell, "specrage_sample", 10.0]
            deployed = lookup[cell, "deployed_upcr", 10.0]
            dufs = lookup[cell, "dufs_liu", FROZEN_DUFS_LAMBDA]
            global_control = lookup[cell, "specrage_global", 10.0]
            uniform_control = lookup[cell, "specrage_uniform", 10.0]
            permuted_control = lookup[cell, "specrage_permuted", 10.0]
            predictions.append({
                "cell": cell,
                "family": family(cell),
                "domain": domain(cell),
                "chosen_config": "agreement_k15",
                "chosen_arm": "specrage_sample",
                "chosen_lambda": 10.0,
                "specrage_auroc": spec["auroc"],
                "deployed_auroc": deployed["auroc"],
                "dufs_auroc": dufs["auroc"],
                "delta_vs_deployed_pp": 100 * (spec["auroc"] - deployed["auroc"]),
                "delta_vs_dufs_pp": 100 * (spec["auroc"] - dufs["auroc"]),
                "delta_vs_global_pp": 100 * (spec["auroc"] - global_control["auroc"]),
                "delta_vs_uniform_pp": 100 * (spec["auroc"] - uniform_control["auroc"]),
                "delta_vs_permuted_pp": 100 * (spec["auroc"] - permuted_control["auroc"]),
            })
    os.makedirs(args.out_dir, exist_ok=True)
    metadata = {
        "version": VERSION,
        "stage": args.stage,
        "runtime_seconds": time.time() - started,
        "bundle": os.path.relpath(args.bundle, REPO),
        "cells": cells,
        "model_seeds": list(args.model_seeds),
        "dufs_seeds": list(args.dufs_seeds),
        "dufs_epochs": args.dufs_epochs,
        "view_schema": VIEW_SCHEMA_VERSION,
        "lambdas": list(LAMBDAS),
        "configs": {name: asdict(config) for name, config in configs.items()},
        "final_choice": final_choice,
        "selection_failed": selection_failed,
        "provenance": provenance_metadata(args.bundle),
    }
    write_csv(os.path.join(args.out_dir, f"{args.stage}_per_run.csv"), rows)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_lofo_decisions.csv"), decisions)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_crossfit_predictions.csv"), predictions)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_candidate_selection.csv"), candidates)
    write_csv(os.path.join(args.out_dir, f"{args.stage}_history.csv"), histories)
    with open(os.path.join(args.out_dir, f"{args.stage}_diagnostics.json"),
              "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=True)
    with open(os.path.join(args.out_dir, f"{args.stage}_metadata.json"),
              "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.stage}_sample_reliance.npz"),
        **reliance_artifacts,
    )
    make_plots(rows, predictions, diagnostics, histories, args.out_dir)
    report = render_report(predictions, decisions, final_choice, diagnostics, metadata)
    with open(os.path.join(args.out_dir, f"{args.stage.upper()}_REPORT.md"),
              "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)
    if selection_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
