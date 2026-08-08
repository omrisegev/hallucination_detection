#!/usr/bin/env python3
"""Frozen-score pilot for repeated-measurement reliability fusion.

All feature filtering, generalized-eigenvalue truncation, DUFS fitting, score
orientation, and score hashing happen before this script reads an outcome
label.  The benchmark therefore tests a fixed label-free construction; labels
are used only to report final discrimination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repeated_measurement_reliability_pilot import (
    FixedMixedV2Transformer,
    feature_matrix,
    load_rows,
    run_configuration,
    trace_features,
)
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.laplacian_upcr import build_graph_from_features, laplacian_iu_path
from spectral_utils.repeated_measurement_reliability import psd_projection
from spectral_utils.streaming_utils import anchor_orient


BLOCK_FRACTION = 0.20
REPEATS = 12
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
GRAPH_K = 7
LAPLACIAN_LAMBDA = 0.1
MIN_TRACE_LENGTH = 64


def _zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return centered / scale


def _orient_coordinates(coordinates, anchor):
    output = np.asarray(coordinates, dtype=float).copy()
    flips = []
    for j in range(output.shape[1]):
        output[:, j], flipped = anchor_orient(output[:, j], anchor)
        flips.append(bool(flipped))
    return _zscore_columns(output), flips


def _orient_to_original_columns(filtered, original):
    output = np.asarray(filtered, dtype=float).copy()
    flips = []
    for j in range(output.shape[1]):
        output[:, j], flipped = anchor_orient(output[:, j], original[:, j])
        flips.append(bool(flipped))
    return _zscore_columns(output), flips


def _off_diagonal_fraction(matrix):
    covariance = np.cov(np.asarray(matrix, dtype=float), rowvar=False)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return float(
        np.linalg.norm(off_diagonal, ord="fro")
        / max(np.linalg.norm(covariance, ord="fro"), 1e-12)
    )


def _fit_pair(features):
    """Return IU-PCR and DUFS-LIU scores for one feature matrix."""
    F = np.asarray(features, dtype=float).T
    gates, gate_diagnostics = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graph = build_graph_from_features(F, gates=gates, k=GRAPH_K)
    path = laplacian_iu_path(F, (0.0, LAPLACIAN_LAMBDA), graph=graph)
    return {
        "iu_pcr": path[0.0].baseline.w @ F,
        "dufs_liu": path[LAPLACIAN_LAMBDA].w @ F,
    }, {
        "gate_effective_feature_count": gate_diagnostics["effective_feature_count"],
        "gate_mean_seed_std": gate_diagnostics["mean_seed_std"],
        "liu": path[LAPLACIAN_LAMBDA].diagnostics,
    }


def fit_label_free(rows, seed):
    """Fit every score without accepting or reading labels."""
    reliability = run_configuration(
        rows, fraction=BLOCK_FRACTION, repeats=REPEATS, seed=seed
    )
    if not reliability["summary"]["validity_passed"]:
        raise RuntimeError("repeated-measurement validity gate failed")

    # The comparison baseline is the current mixed-v2 pool, including length.
    full_raw, full_names, full_exclusions = feature_matrix(
        [trace_features(row) for row in rows], exclude_fixed_length=False
    )
    full_transformer = FixedMixedV2Transformer.fit(full_raw, full_names)
    full = full_transformer.training_output
    consensus = full.mean(axis=1)
    baseline_scores, baseline_diagnostics = _fit_pair(full)

    eligible = reliability["eligible_mask"]
    reliable_original = reliability["original_matrix"][:, eligible]
    eigenvalues = reliability["generalized_eigenvalues"]
    retained_k = int(np.sum(eigenvalues > 1.0))
    if retained_k < 3:
        raise RuntimeError(f"only {retained_k} reliable directions survived")
    vectors = reliability["generalized_vectors"][:, :retained_k]
    coordinates, coordinate_flips = _orient_coordinates(
        reliable_original @ vectors, consensus
    )
    latent_scores, latent_diagnostics = _fit_pair(coordinates)

    # A generalized eigenbasis is close to covariance-diagonal by design and
    # therefore is a poor set of "regressors" for U-PCR's off-diagonal moment
    # equations.  The corrected variants filter in that basis but return to the
    # original feature axes before U-PCR.
    orthonormal, _ = np.linalg.qr(vectors)
    projected, projected_flips = _orient_to_original_columns(
        reliable_original @ orthonormal @ orthonormal.T,
        reliable_original,
    )
    projected_scores, projected_diagnostics = _fit_pair(projected)

    signal_psd, _ = psd_projection(reliability["signal"])
    within = reliability["within"]
    ridge = float(reliability["summary"]["generalized_ridge"])
    wiener_operator = np.linalg.solve(
        signal_psd + within + ridge * np.eye(len(within)), signal_psd
    )
    wiener, wiener_flips = _orient_to_original_columns(
        reliable_original @ wiener_operator, reliable_original
    )
    wiener_scores, wiener_diagnostics = _fit_pair(wiener)

    scores = {
        "iu_pcr_mixed_v2": baseline_scores["iu_pcr"],
        "dufs_liu_mixed_v2": baseline_scores["dufs_liu"],
        "rm_latent_iu_pcr": latent_scores["iu_pcr"],
        "rm_latent_dufs_liu": latent_scores["dufs_liu"],
        "rm_projected_iu_pcr": projected_scores["iu_pcr"],
        "rm_projected_dufs_liu": projected_scores["dufs_liu"],
        "rm_wiener_iu_pcr": wiener_scores["iu_pcr"],
        "rm_wiener_dufs_liu": wiener_scores["dufs_liu"],
    }
    score_flips = {}
    for name in scores:
        scores[name], score_flips[name] = anchor_orient(scores[name], consensus)
    diagnostics = {
        "labels_seen_during_fit": False,
        "block_fraction": BLOCK_FRACTION,
        "repeats": REPEATS,
        "reliability_summary": reliability["summary"],
        "procedure_compatible_features": list(reliability["eligible_feature_names"]),
        "full_mixed_v2_features": list(full_names),
        "full_mixed_v2_exclusions": full_exclusions,
        "retained_generalized_directions": retained_k,
        "retained_generalized_eigenvalues": eigenvalues[:retained_k].tolist(),
        "coordinate_flips_to_consensus": coordinate_flips,
        "projected_feature_flips": projected_flips,
        "wiener_feature_flips": wiener_flips,
        "off_diagonal_covariance_fraction": {
            "full_mixed_v2": _off_diagonal_fraction(full),
            "latent_coordinates": _off_diagonal_fraction(coordinates),
            "hard_projected_features": _off_diagonal_fraction(projected),
            "wiener_filtered_features": _off_diagonal_fraction(wiener),
        },
        "score_flips_to_consensus": score_flips,
        "baseline": baseline_diagnostics,
        "latent_negative_control": latent_diagnostics,
        "hard_projected_candidate": projected_diagnostics,
        "wiener_candidate": wiener_diagnostics,
    }
    return scores, diagnostics


def metric(y, score, rng, bootstraps=500):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    bootstrap = []
    for _ in range(bootstraps):
        index = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        bootstrap.append(roc_auc_score(y[index], score[index]))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "auroc": auc,
        "auroc_low": float(low),
        "auroc_high": float(high),
        "auprc": ap,
    }


def plot_metrics(records, output_path):
    cells = list(dict.fromkeys(row["cell"] for row in records))
    methods = list(dict.fromkeys(row["method"] for row in records))
    figure, axes = plt.subplots(1, len(cells), figsize=(7 * len(cells), 5), squeeze=False)
    colors = ["#6c757d", "#2b8a3e", "#748ffc", "#e8590c"]
    for axis, cell in zip(axes[0], cells):
        selected = {row["method"]: row for row in records if row["cell"] == cell}
        values = [selected[name]["auroc"] for name in methods]
        low = [selected[name]["auroc"] - selected[name]["auroc_low"] for name in methods]
        high = [selected[name]["auroc_high"] - selected[name]["auroc"] for name in methods]
        axis.bar(np.arange(len(methods)), values, color=colors[:len(methods)])
        axis.errorbar(np.arange(len(methods)), values, yerr=[low, high], fmt="none", color="black", capsize=4)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(np.arange(len(methods)), methods, rotation=30, ha="right")
        axis.set_ylim(0.35, 0.85)
        axis.set_ylabel("AUROC (higher is better)")
        axis.set_title(cell)
    figure.suptitle("Repeated-measurement reliability fusion — frozen-score pilot")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, all_diagnostics = [], {}
    for path_text in args.paths:
        path = Path(path_text)
        rows = [
            row for row in load_rows(path)
            if len(row["token_entropies"]) >= MIN_TRACE_LENGTH
        ]
        if args.max_rows and len(rows) > args.max_rows:
            selection_rng = np.random.default_rng(args.seed)
            selected = np.sort(selection_rng.choice(len(rows), args.max_rows, replace=False))
            rows = [rows[index] for index in selected]
        cell = f"{path.parent.name}__{path.stem.removeprefix('processbench_')}"
        cell_seed = args.seed + (
            int.from_bytes(hashlib.sha256(cell.encode("utf-8")).digest()[:4], "little")
            % 1_000_000
        )
        scores, diagnostics = fit_label_free(rows, cell_seed)
        diagnostics["cell_seed"] = int(cell_seed)
        hashes = {
            name: hashlib.sha256(np.asarray(score, dtype="<f8").tobytes()).hexdigest()
            for name, score in scores.items()
        }
        diagnostics["score_hashes_before_labels"] = hashes
        diagnostics["scores_frozen_before_labels"] = True
        np.savez_compressed(
            output_dir / f"{cell}__scores_before_labels.npz", **scores
        )

        # Outcome labels are first accessed here, after all scores are frozen.
        y = np.asarray([row["final_answer_correct"] for row in rows], dtype=int)
        np.savez_compressed(
            output_dir / f"{cell}__evaluation_arrays.npz", y=y, **scores
        )
        rng = np.random.default_rng(args.seed)
        for method, score in scores.items():
            records.append({
                "cell": cell, "method": method, "n": len(y),
                "positive_rate": float(y.mean()), **metric(y, score, rng),
            })
        all_diagnostics[cell] = diagnostics
        print(f"\n{cell} (n={len(y)})")
        for record in records[-len(scores):]:
            print(
                f"  {record['method']:28s} {record['auroc']:.4f} "
                f"[{record['auroc_low']:.4f}, {record['auroc_high']:.4f}]"
            )
    with open(output_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    with open(output_dir / "diagnostics.json", "w") as handle:
        json.dump({
            "protocol": "scores frozen and hashed before final_answer_correct was read",
            "labels_used_for_model_selection": False,
            "cells": all_diagnostics,
        }, handle, indent=2, default=str)
    plot_metrics(records, output_dir / "auroc_by_cell.png")


if __name__ == "__main__":
    main()
