#!/usr/bin/env python3
"""Cross-dataset supervised diagnostic for a transferable hallucination manifold."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402


VERSION = "cross-dataset-hallucination-manifold-v1-2026-08-19"
DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_IU = REPO / "results" / "frozen_24cell_benchmark" / "scores"
DEFAULT_OUT = REPO / "results" / "cross_dataset_hallucination_manifold_v1"
PROTOCOL = REPO / "docs" / "experiments" / "CROSS_DATASET_HALLUCINATION_MANIFOLD_V1.md"
FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)
EXPECTED_COMMON = (
    "cusum_max", "cusum_max_energy", "cusum_max_spilled", "epr",
    "epr_energy", "epr_spilled", "logprob_margin", "mean_logprob_entropy",
    "mean_top1_logprob", "min_energy", "renyi_entropy_2", "sw_var_peak",
    "sw_var_peak_energy", "sw_var_peak_spilled", "topk_tail_mass", "varentropy",
)
MODEL_ORDER = (
    "epr_risk", "mean_confidence_risk", "iu_pcr_risk",
    "shared_direction", "balanced_logistic", "ppca_manifold_k4",
    "knn_manifold_k5",
)
BOOTSTRAPS = 5000
SIGN_FLIPS = 10000
PPCA_K = 4
PPCA_SHRINKAGE = 0.10
KNN_K = 5
KNN_SUPPORT_PER_CELL_CLASS = 64
SEED = 20260819


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2 ** 32)


def family(cell: str) -> str:
    found = [name for name in FAMILY_NAMES if name in cell]
    if len(found) != 1:
        raise ValueError(f"cannot assign exactly one dataset family to {cell}: {found}")
    return found[0]


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0])
    return np.asarray(matrix[indices], dtype=float)


def covariance(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    centered = rows - rows.mean(axis=0, keepdims=True)
    denominator = max(1, len(rows) - 1)
    return centered.T @ centered / denominator


def effective_rank(rows: np.ndarray) -> float:
    values = np.maximum(np.linalg.eigvalsh(covariance(rows)), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = values / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-30))))


def load_cells(bundle_path: Path, iu_dir: Path) -> tuple[dict, tuple[str, ...]]:
    cells = {}
    with np.load(bundle_path, allow_pickle=True) as bundle:
        observed = {key[:-3] for key in bundle.files if key.endswith("__V")}
        if observed != set(INSCOPE):
            raise RuntimeError(
                f"bundle roster mismatch: missing={sorted(set(INSCOPE)-observed)}, "
                f"extra={sorted(observed-set(INSCOPE))}"
            )
        prepared = {}
        common = None
        for cell in INSCOPE:
            matrix, names = fixed_stable_from_bundle(
                bundle[f"{cell}__V"],
                tuple(map(str, bundle[f"{cell}__pool"])),
                bundle[f"{cell}__hand_signs"],
            )
            prepared[cell] = (matrix, names)
            common = set(names) if common is None else common.intersection(names)
        common_names = tuple(sorted(common))
        if common_names != EXPECTED_COMMON:
            raise RuntimeError(
                f"common feature contract changed: {common_names} != {EXPECTED_COMMON}"
            )
        for cell in INSCOPE:
            matrix, names = prepared[cell]
            lookup = {name: index for index, name in enumerate(names)}
            X = np.asarray(matrix[:, [lookup[name] for name in common_names]], dtype=float)
            correctness = np.asarray(bundle[f"{cell}__labels"], dtype=int)
            if X.shape != (len(correctness), len(common_names)):
                raise RuntimeError(f"matrix/label mismatch in {cell}")
            if not np.isfinite(X).all() or not np.isin(correctness, [0, 1]).all():
                raise RuntimeError(f"non-finite or non-binary input in {cell}")
            y = 1 - correctness
            if len(np.unique(y)) != 2:
                raise RuntimeError(f"single-class cell: {cell}")
            score_path = iu_dir / f"{cell}.npz"
            with np.load(score_path, allow_pickle=False) as scores:
                iu = -np.asarray(scores["iu_pcr"], dtype=float)
                if not np.array_equal(scores["sample_index"], np.arange(len(y))):
                    raise RuntimeError(f"IU score order mismatch in {cell}")
            cells[cell] = {
                "X": X,
                "y": y,
                "family": family(cell),
                "domain": GROUP[cell],
                "iu_risk": iu,
            }
    return cells, common_names


def cell_fingerprints(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    X, y = cell["X"], cell["y"]
    error, correct = X[y == 1], X[y == 0]
    mean_fp = unit(error.mean(axis=0) - correct.mean(axis=0))
    shape_fp = unit(upper_triangle(covariance(error) - covariance(correct)))
    return mean_fp, shape_fp


def primary_fingerprint_transfer(
    fingerprints: dict[str, np.ndarray], signs: dict[str, float] | None = None,
) -> tuple[float, list[dict]]:
    signs = signs or {cell: 1.0 for cell in INSCOPE}
    rows = []
    for held_family in FAMILY_NAMES:
        train = [cell for cell in INSCOPE if family(cell) != held_family]
        test = [cell for cell in INSCOPE if family(cell) == held_family]
        donor = unit(np.mean([signs[cell] * fingerprints[cell] for cell in train], axis=0))
        values = [float(donor @ (signs[cell] * fingerprints[cell])) for cell in test]
        rows.append({
            "held_family": held_family,
            "n_cells": len(test),
            "mean_cosine": float(np.mean(values)),
        })
    return float(np.mean([row["mean_cosine"] for row in rows])), rows


def sign_flip_test(fingerprints: dict[str, np.ndarray], namespace: str) -> dict:
    observed, per_family = primary_fingerprint_transfer(fingerprints)
    rng = np.random.default_rng(stable_seed(VERSION, namespace, SEED))
    null = np.empty(SIGN_FLIPS, dtype=float)
    for index in range(SIGN_FLIPS):
        signs = dict(zip(INSCOPE, rng.choice((-1.0, 1.0), size=len(INSCOPE))))
        null[index] = primary_fingerprint_transfer(fingerprints, signs)[0]
    return {
        "observed_equal_family_cosine": observed,
        "null_mean": float(null.mean()),
        "null_95th_percentile": float(np.quantile(null, 0.95)),
        "one_sided_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
        "per_family": per_family,
    }


def training_weights(train_cells: list[str], cells: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices, labels, weights = [], [], []
    for cell in train_cells:
        X, y = cells[cell]["X"], cells[cell]["y"]
        matrices.append(X)
        labels.append(y)
        block = np.empty(len(y), dtype=float)
        for target in (0, 1):
            mask = y == target
            block[mask] = 1.0 / int(mask.sum())
        weights.append(block)
    X = np.vstack(matrices)
    y = np.concatenate(labels)
    weight = np.concatenate(weights)
    weight *= len(weight) / weight.sum()
    return X, y, weight


def shared_direction(train_cells: list[str], cells: dict) -> np.ndarray:
    directions = []
    for cell in train_cells:
        directions.append(cell_fingerprints(cells[cell])[0])
    return unit(np.mean(directions, axis=0))


def fit_ppca_class(train_cells: list[str], cells: dict, target: int) -> dict:
    means, covariances = [], []
    for cell in train_cells:
        rows = cells[cell]["X"][cells[cell]["y"] == target]
        means.append(rows.mean(axis=0))
        covariances.append(covariance(rows))
    mean = np.mean(means, axis=0)
    empirical = np.mean(covariances, axis=0)
    values, vectors = np.linalg.eigh(0.5 * (empirical + empirical.T))
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 1e-8), vectors[:, order]
    k = min(PPCA_K, len(values) - 1)
    residual = float(np.mean(values[k:])) if k < len(values) else 1e-4
    residual = max(residual, 1e-4)
    covariance_model = vectors[:, :k] @ np.diag(values[:k]) @ vectors[:, :k].T
    covariance_model += residual * (np.eye(len(values)) - vectors[:, :k] @ vectors[:, :k].T)
    covariance_model = (
        (1.0 - PPCA_SHRINKAGE) * covariance_model
        + PPCA_SHRINKAGE * np.eye(len(values))
    )
    sign, logdet = np.linalg.slogdet(covariance_model)
    if sign <= 0:
        raise RuntimeError("PPCA covariance is not positive definite")
    return {"mean": mean, "precision": np.linalg.inv(covariance_model), "logdet": float(logdet)}


def gaussian_log_density(X: np.ndarray, model: dict) -> np.ndarray:
    centered = X - model["mean"]
    return -0.5 * (
        np.einsum("ij,jk,ik->i", centered, model["precision"], centered)
        + model["logdet"]
    )


def deterministic_support(cell: str, target: int, cells: dict) -> np.ndarray:
    rows = cells[cell]["X"][cells[cell]["y"] == target]
    if len(rows) <= KNN_SUPPORT_PER_CELL_CLASS:
        return rows
    rng = np.random.default_rng(stable_seed(VERSION, "knn", cell, target, SEED))
    return rows[np.sort(rng.choice(len(rows), KNN_SUPPORT_PER_CELL_CLASS, replace=False))]


def knn_manifold_score(test: np.ndarray, train_cells: list[str], cells: dict) -> np.ndarray:
    donor_scores = []
    for cell in train_cells:
        distances = {}
        for target in (0, 1):
            support = deterministic_support(cell, target, cells)
            k = min(KNN_K, len(support))
            squared = cdist(test, support, metric="sqeuclidean")
            nearest = np.partition(squared, k - 1, axis=1)[:, :k]
            distances[target] = np.sqrt(np.maximum(nearest.mean(axis=1), 1e-12))
        donor_scores.append(np.log(distances[0] + 1e-9) - np.log(distances[1] + 1e-9))
    return np.mean(donor_scores, axis=0)


def safe_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def evaluate_split(scheme: str, cells: dict, common_names: tuple[str, ...]) -> list[dict]:
    if scheme == "family":
        folds = [(name, [cell for cell in INSCOPE if family(cell) == name]) for name in FAMILY_NAMES]
    elif scheme == "cell":
        folds = [(cell, [cell]) for cell in INSCOPE]
    else:
        raise ValueError(scheme)
    epr_index = common_names.index("epr")
    output = []
    for fold_index, (held_group, test_cells) in enumerate(folds, start=1):
        train_cells = [cell for cell in INSCOPE if cell not in test_cells]
        train_X, train_y, weights = training_weights(train_cells, cells)
        direction = shared_direction(train_cells, cells)
        logistic = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            random_state=stable_seed(VERSION, scheme, held_group),
        )
        logistic.fit(train_X, train_y, sample_weight=weights)
        ppca_correct = fit_ppca_class(train_cells, cells, 0)
        ppca_error = fit_ppca_class(train_cells, cells, 1)
        print(f"[{scheme}] fold {fold_index}/{len(folds)}: hold {held_group}", flush=True)
        for cell in test_cells:
            X, y = cells[cell]["X"], cells[cell]["y"]
            scores = {
                "epr_risk": -X[:, epr_index],
                "mean_confidence_risk": -X.mean(axis=1),
                "iu_pcr_risk": cells[cell]["iu_risk"],
                "shared_direction": X @ direction,
                "balanced_logistic": logistic.decision_function(X),
                "ppca_manifold_k4": gaussian_log_density(X, ppca_error) - gaussian_log_density(X, ppca_correct),
                "knn_manifold_k5": knn_manifold_score(X, train_cells, cells),
            }
            for method in MODEL_ORDER:
                auc, ap = safe_metrics(y, scores[method])
                output.append({
                    "split": scheme,
                    "held_group": held_group,
                    "cell": cell,
                    "family": cells[cell]["family"],
                    "domain": cells[cell]["domain"],
                    "n": len(y),
                    "error_rate": float(y.mean()),
                    "method": method,
                    "auroc": auc,
                    "error_auprc": ap,
                })
    return output


def family_bootstrap(values: dict[str, float], namespace: str) -> tuple[float, float, float]:
    names = tuple(FAMILY_NAMES)
    array = np.asarray([values[name] for name in names], dtype=float)
    rng = np.random.default_rng(stable_seed(VERSION, "bootstrap", namespace, SEED))
    indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    estimates = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for split in ("family", "cell"):
        subset = [row for row in rows if row["split"] == split]
        for method in MODEL_ORDER:
            selected = [row for row in subset if row["method"] == method]
            family_auc = {
                name: float(np.mean([row["auroc"] for row in selected if row["family"] == name]))
                for name in FAMILY_NAMES
            }
            mean, low, high = family_bootstrap(family_auc, f"{split}|{method}|auc")
            family_ap = {
                name: float(np.mean([row["error_auprc"] for row in selected if row["family"] == name]))
                for name in FAMILY_NAMES
            }
            ap_mean, ap_low, ap_high = family_bootstrap(family_ap, f"{split}|{method}|ap")
            summary.append({
                "split": split,
                "method": method,
                "cell_macro_auroc": float(np.mean([row["auroc"] for row in selected])),
                "family_macro_auroc": mean,
                "family_auc_ci_low": low,
                "family_auc_ci_high": high,
                "cell_macro_error_auprc": float(np.mean([row["error_auprc"] for row in selected])),
                "family_macro_error_auprc": ap_mean,
                "family_auprc_ci_low": ap_low,
                "family_auprc_ci_high": ap_high,
            })
    return summary


def method_delta(rows: list[dict], candidate: str, reference: str) -> dict:
    primary = [row for row in rows if row["split"] == "family"]
    lookup = {(row["cell"], row["method"]): row["auroc"] for row in primary}
    by_family = {}
    for name in FAMILY_NAMES:
        members = [cell for cell in INSCOPE if family(cell) == name]
        by_family[name] = float(np.mean([
            lookup[(cell, candidate)] - lookup[(cell, reference)] for cell in members
        ]))
    mean, low, high = family_bootstrap(by_family, f"delta|{candidate}|{reference}")
    return {"mean": mean, "low": low, "high": high, "by_family": by_family}


def feature_effects(cells: dict, names: tuple[str, ...]) -> list[dict]:
    effects = {}
    for cell in INSCOPE:
        X, y = cells[cell]["X"], cells[cell]["y"]
        effects[cell] = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    rows = []
    rng = np.random.default_rng(stable_seed(VERSION, "feature-effects", SEED))
    family_indices = rng.integers(0, len(FAMILY_NAMES), size=(BOOTSTRAPS, len(FAMILY_NAMES)))
    for index, name in enumerate(names):
        family_values = np.asarray([
            np.mean([effects[cell][index] for cell in INSCOPE if family(cell) == fam])
            for fam in FAMILY_NAMES
        ])
        boot = family_values[family_indices].mean(axis=1)
        values = np.asarray([effects[cell][index] for cell in INSCOPE])
        rows.append({
            "feature": name,
            "cell_macro_error_minus_correct": float(values.mean()),
            "family_macro_error_minus_correct": float(family_values.mean()),
            "family_ci_low": float(np.quantile(boot, 0.025)),
            "family_ci_high": float(np.quantile(boot, 0.975)),
            "positive_cells": int(np.sum(values > 0)),
            "negative_cells": int(np.sum(values < 0)),
        })
    rows.sort(key=lambda row: abs(row["family_macro_error_minus_correct"]), reverse=True)
    return rows


def geometry_summary(cells: dict) -> dict:
    mean_fps, shape_fps = {}, {}
    dimensions = []
    for cell in INSCOPE:
        mean_fps[cell], shape_fps[cell] = cell_fingerprints(cells[cell])
        X, y = cells[cell]["X"], cells[cell]["y"]
        dimensions.append({
            "cell": cell,
            "family": family(cell),
            "correct_effective_rank": effective_rank(X[y == 0]),
            "error_effective_rank": effective_rank(X[y == 1]),
        })
    return {
        "mean_fingerprint": sign_flip_test(mean_fps, "mean"),
        "shape_fingerprint": sign_flip_test(shape_fps, "shape"),
        "intrinsic_dimension": {
            "correct_cell_macro": float(np.mean([row["correct_effective_rank"] for row in dimensions])),
            "error_cell_macro": float(np.mean([row["error_effective_rank"] for row in dimensions])),
            "per_cell": dimensions,
        },
    }


def decision(summary: list[dict], geometry: dict, rows: list[dict]) -> tuple[str, dict]:
    primary = {row["method"]: row for row in summary if row["split"] == "family"}
    deltas = {
        method: method_delta(rows, method, "balanced_logistic")
        for method in ("ppca_manifold_k4", "knn_manifold_k5")
    }
    nonlinear = [method for method, delta in deltas.items()
                 if primary[method]["family_macro_auroc"] >= 0.60
                 and primary[method]["family_auc_ci_low"] > 0.50
                 and delta["mean"] >= 0.005 and delta["low"] > 0.0]
    mean_transfer = geometry["mean_fingerprint"]["one_sided_p"] <= 0.05
    shape_transfer = geometry["shape_fingerprint"]["one_sided_p"] <= 0.05
    linear_predictive = (
        primary["balanced_logistic"]["family_macro_auroc"] >= 0.60
        and primary["balanced_logistic"]["family_auc_ci_low"] > 0.50
    )
    if nonlinear:
        verdict = "TYPICAL_NONLINEAR_MANIFOLD_SUPPORTED"
    elif mean_transfer and linear_predictive:
        verdict = "SHARED_DIRECTION_NOT_DISTINCT_NONLINEAR_MANIFOLD"
    elif shape_transfer:
        verdict = "SHAPE_REGULARITY_ONLY"
    else:
        verdict = "NO_TRANSFERABLE_GEOMETRY"
    return verdict, {
        "nonlinear_methods_passing": nonlinear,
        "mean_fingerprint_transfers": mean_transfer,
        "shape_fingerprint_transfers": shape_transfer,
        "balanced_logistic_predictive": linear_predictive,
        "manifold_vs_logistic_deltas": deltas,
    }


def make_figures(out: Path, rows: list[dict], effects: list[dict], geometry: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row["split"] == "family"]
    cells_order = list(INSCOPE)
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.18
    shown = ("mean_confidence_risk", "balanced_logistic", "ppca_manifold_k4", "knn_manifold_k5")
    x = np.arange(len(cells_order))
    lookup = {(row["cell"], row["method"]): row["auroc"] for row in primary}
    for offset, method in enumerate(shown):
        ax.bar(x + (offset - 1.5) * width, [lookup[(cell, method)] for cell in cells_order], width, label=method)
    ax.axhline(0.5, color="black", linewidth=1)
    ax.set_ylabel("Held-family AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(cells_order, rotation=70, ha="right", fontsize=7)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "01_primary_per_cell_auroc.png", dpi=180)
    plt.close(fig)

    top = effects[:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    values = np.asarray([row["family_macro_error_minus_correct"] for row in top])
    low = np.asarray([row["family_ci_low"] for row in top])
    high = np.asarray([row["family_ci_high"] for row in top])
    ax.barh(range(len(top)), values, xerr=np.vstack([values-low, high-values]), color="#4c78a8", alpha=.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([row["feature"] for row in top])
    ax.set_xlabel("Error − correct standardized feature mean (equal-family)")
    fig.tight_layout()
    fig.savefig(out / "02_feature_effects.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["Mean shift", "Covariance shape"]
    observed = [geometry["mean_fingerprint"]["observed_equal_family_cosine"], geometry["shape_fingerprint"]["observed_equal_family_cosine"]]
    null95 = [geometry["mean_fingerprint"]["null_95th_percentile"], geometry["shape_fingerprint"]["null_95th_percentile"]]
    x = np.arange(2)
    ax.bar(x - .17, observed, .34, label="observed")
    ax.bar(x + .17, null95, .34, label="sign-flip null 95th")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-family fingerprint cosine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_fingerprint_transfer.png", dpi=180)
    plt.close(fig)


def build_report(out: Path, summary: list[dict], effects: list[dict], geometry: dict,
                 verdict: str, gates: dict) -> None:
    primary = [row for row in summary if row["split"] == "family"]
    secondary = [row for row in summary if row["split"] == "cell"]
    lines = [
        "# Cross-dataset hallucination manifold diagnostic v1", "",
        f"**Decision: `{verdict}`**", "",
        "## Short answer", "",
    ]
    if verdict == "TYPICAL_NONLINEAR_MANIFOLD_SUPPORTED":
        lines.append("At least one nonlinear geometry transfers across held-out dataset families and improves over the balanced shared linear head under the frozen gates.")
    elif verdict == "SHARED_DIRECTION_NOT_DISTINCT_NONLINEAR_MANIFOLD":
        lines.append("The features contain a transferable hallucination direction, but the nonlinear manifold models do not add a reliable advantage over a balanced shared linear head. The evidence supports a common direction, not a distinct curved hallucination manifold.")
    elif verdict == "SHAPE_REGULARITY_ONLY":
        lines.append("Some class-specific covariance shape repeats across families, but it does not produce a sufficiently predictive cross-family manifold detector.")
    else:
        lines.append("Neither the labelled geometry fingerprints nor the predictive geometry models show a transferable hallucination manifold under the frozen criteria.")
    lines += [
        "", "This is a supervised retrospective diagnostic. It does not make DUFS-LIU label-free geometry identifiable and it is not external confirmation.",
        "", "The inputs use the project's previously frozen confidence-orientation contract. That contract was informed by earlier labelled audits, so the fact that individual feature shifts mostly point in the confidence direction is not a fresh discovery. The held-family predictive comparison and the nonlinear-versus-linear contrast are the useful parts of this diagnostic.", "",
        "## Primary leave-one-dataset-family-out results", "",
        "| method | family AUROC [95% CI] | cell AUROC | family error AUPRC |",
        "|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| `{row['method']}` | {row['family_macro_auroc']:.4f} [{row['family_auc_ci_low']:.4f}, {row['family_auc_ci_high']:.4f}] | "
            f"{row['cell_macro_auroc']:.4f} | {row['family_macro_error_auprc']:.4f} |"
        )
    lines += ["", "## Secondary leave-one-cell-out results", "", "| method | family AUROC [95% CI] | cell AUROC |", "|---|---:|---:|"]
    for row in secondary:
        lines.append(f"| `{row['method']}` | {row['family_macro_auroc']:.4f} [{row['family_auc_ci_low']:.4f}, {row['family_auc_ci_high']:.4f}] | {row['cell_macro_auroc']:.4f} |")
    mean = geometry["mean_fingerprint"]
    shape = geometry["shape_fingerprint"]
    dims = geometry["intrinsic_dimension"]
    lines += [
        "", "## Does the geometry itself repeat?", "",
        "| fingerprint | held-family cosine | null 95th | one-sided p |",
        "|---|---:|---:|---:|",
        f"| error−correct mean direction | {mean['observed_equal_family_cosine']:.4f} | {mean['null_95th_percentile']:.4f} | {mean['one_sided_p']:.4g} |",
        f"| error−correct covariance shape | {shape['observed_equal_family_cosine']:.4f} | {shape['null_95th_percentile']:.4f} | {shape['one_sided_p']:.4g} |",
        "", f"The cell-macro covariance participation rank is {dims['error_cell_macro']:.2f} for errors and {dims['correct_cell_macro']:.2f} for correct answers, out of 16 available dimensions.",
        "", "## Strongest repeatable feature shifts", "",
        "Positive means the confidence-oriented feature is higher on errors; negative means lower on errors. Because the orientation contract defines higher values as more likely correct, the signs are expected; cross-family consistency and effect magnitude are descriptive, not an independent validation of the signs.", "",
        "| feature | equal-family error−correct | 95% CI | sign across cells |",
        "|---|---:|---:|---:|",
    ]
    for row in effects[:10]:
        lines.append(f"| `{row['feature']}` | {row['family_macro_error_minus_correct']:+.3f} | [{row['family_ci_low']:+.3f}, {row['family_ci_high']:+.3f}] | {row['positive_cells']}+ / {row['negative_cells']}− |")
    lines += [
        "", "## Interpretation boundary", "",
        "A shared supervised direction means donor labels identify a repeatable target axis in these already oriented features. It does not mean the unlabeled marginal geometry can tell DUFS which axis is hallucination. A common global sign reflection of feature axes cannot manufacture a nonlinear advantage, so the failure of PPCA/kNN to beat the linear control remains informative. Without that nonlinear win, 'shared direction' is the more accurate description.",
        "", "## Figures", "",
        "![Primary per-cell AUROC](01_primary_per_cell_auroc.png)", "",
        "![Feature effects](02_feature_effects.png)", "",
        "![Fingerprint transfer](03_fingerprint_transfer.png)", "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--iu-dir", type=Path, default=DEFAULT_IU)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells, common_names = load_cells(args.bundle, args.iu_dir)
    definition = {
        "version": VERSION,
        "bundle": str(args.bundle.relative_to(REPO)),
        "bundle_sha256": sha256_file(args.bundle),
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source": str(Path(__file__).resolve().relative_to(REPO)),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "cells": list(INSCOPE),
        "families": list(FAMILY_NAMES),
        "common_features": list(common_names),
        "models": list(MODEL_ORDER),
        "parameters": {
            "ppca_k": PPCA_K, "ppca_shrinkage": PPCA_SHRINKAGE,
            "knn_k": KNN_K, "knn_support_per_cell_class": KNN_SUPPORT_PER_CELL_CLASS,
            "bootstraps": BOOTSTRAPS, "sign_flips": SIGN_FLIPS, "seed": SEED,
        },
        "label_role": "supervised donor-cell target; held-cell labels evaluation only",
        "claim_boundary": "retrospective diagnostic, not label-free and not external confirmation",
    }
    write_json(args.out_dir / "RUN_DEFINITION.json", definition)

    geometry = geometry_summary(cells)
    effects = feature_effects(cells, common_names)
    primary_rows = evaluate_split("family", cells, common_names)
    secondary_rows = evaluate_split("cell", cells, common_names)
    metric_rows = primary_rows + secondary_rows
    summary = summarize(metric_rows)
    verdict, gates = decision(summary, geometry, metric_rows)

    write_csv(args.out_dir / "PER_CELL_METRICS.csv", metric_rows)
    write_csv(args.out_dir / "SUMMARY.csv", summary)
    write_csv(args.out_dir / "FEATURE_EFFECTS.csv", effects)
    write_json(args.out_dir / "GEOMETRY.json", geometry)
    write_json(args.out_dir / "DECISION.json", {"decision": verdict, **gates})
    make_figures(args.out_dir, metric_rows, effects, geometry)
    build_report(args.out_dir, summary, effects, geometry, verdict, gates)
    print(f"decision={verdict}")
    print(args.out_dir / "REPORT.md")


if __name__ == "__main__":
    main()
