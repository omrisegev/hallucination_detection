#!/usr/bin/env python3
"""Staged synthetic test of target-anchored Laplacian IU-PCR.

Development uses only the already-consumed 40,000 seed block.  Confirmation is
a separate command that refuses source, dependency, or configuration changes
before opening the reserved 2,600,000 block.  No real hallucination data is
read by this script.
"""

import argparse
from collections import defaultdict
import csv
import gzip
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
    laplacian_iu_fit,
    self_tuning_knn_graph,
)
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.target_anchored_laplacian import (  # noqa: E402
    fixed_logistic_scores,
    ordinary_u2_coordinates,
    projected_ridge_fit,
    pseudo_anchor_laplacian_fit,
    target_anchored_laplacian_fit,
)


VERSION = "target-anchored-liu-synthetic-v1-2026-08-06"
SPEC = "SPEC_TARGET_ANCHORED_LIU_V1.md"
DEFAULT_OUT = os.path.join(REPO, "results", "target_anchored_laplacian_synthetic")
DEVELOPMENT_SEED_OFFSET = 40_000
CONFIRMATION_SEED_OFFSET = 2_600_000
BUDGETS = (4, 8, 16, 32, 64)
PRIMARY_BUDGET = 16
LAMBDA = 0.1
TASKS = (
    "smooth_signal",
    "nuisance_manifold",
    "selective_target_signal",
    "selective_target_nuisance",
    "correlated_errors",
    "pure_noise",
)
METHODS = (
    "iu",
    "dufs_liu",
    "projected_ridge",
    "pseudo_anchor",
    "ta_liu",
    "u2_logistic",
    "full_logistic",
    "oracle_latent",
)
METHOD_LABELS = {
    "iu": "IU-PCR",
    "dufs_liu": "DUFS-LIU",
    "projected_ridge": "Projected ridge",
    "pseudo_anchor": "Pseudo-anchor",
    "ta_liu": "TA-LIU",
    "u2_logistic": "U2 logistic",
    "full_logistic": "Full logistic",
    "oracle_latent": "Oracle latent",
}
METHOD_COLORS = {
    "iu": "#111827",
    "dufs_liu": "#2563eb",
    "projected_ridge": "#64748b",
    "pseudo_anchor": "#d97706",
    "ta_liu": "#7c3aed",
    "u2_logistic": "#0f766e",
    "full_logistic": "#dc2626",
    "oracle_latent": "#ec4899",
}
TASK_LABELS = {
    "smooth_signal": "Smooth signal",
    "nuisance_manifold": "Broad nuisance",
    "selective_target_signal": "Paired target: g",
    "selective_target_nuisance": "Paired target: u",
    "correlated_errors": "Correlated errors",
    "pure_noise": "Pure noise",
}
EPS = 1e-12
CANONICAL_CONFIG = {
    "dev_replicates": 8,
    "confirm_replicates": 8,
    "calibration_draws": 16,
    "n": 360,
    "k": 7,
    "dufs_epochs": 120,
}


def zscore_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.where(scale > EPS, scale, 1.0)


def balanced_labels(latent, noise, rng):
    decision = np.asarray(latent) + float(noise) * rng.standard_normal(len(latent))
    return (decision > np.median(decision)).astype(int)


def standardize(vector):
    vector = np.asarray(vector, dtype=float)
    return (vector - vector.mean()) / (vector.std() + EPS)


def make_single_task(task, seed, n):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    if task == "smooth_signal":
        labels = balanced_labels(g, 0.75, rng)
        rows = [g + sigma * rng.standard_normal(n)
                for sigma in (0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0)]
        rows += [0.55 * g + 1.15 * rng.standard_normal(n) for _ in range(2)]
        rows += [rng.standard_normal(n) for _ in range(2)]
        roles = ["signal"] * 8 + ["weak_signal"] * 2 + ["noise"] * 2
    elif task == "nuisance_manifold":
        labels = balanced_labels(g, 0.75, rng)
        nuisance = rng.standard_normal(n)
        rows = [g + 0.55 * rng.standard_normal(n) for _ in range(6)]
        rows += [nuisance + 0.25 * rng.standard_normal(n) for _ in range(6)]
        roles = ["signal"] * 6 + ["nuisance"] * 6
    elif task == "correlated_errors":
        labels = balanced_labels(g, 0.75, rng)
        shared = rng.standard_normal(n)
        rows = [g + 0.45 * rng.standard_normal(n) for _ in range(4)]
        rows += [g + 1.15 * shared + 0.3 * rng.standard_normal(n) for _ in range(4)]
        rows += [g - 0.8 * shared + 0.55 * rng.standard_normal(n) for _ in range(4)]
        roles = (["clean_signal"] * 4 + ["shared_error_positive"] * 4
                 + ["shared_error_negative"] * 4)
    elif task == "pure_noise":
        labels = balanced_labels(g, 0.75, rng)
        rows = [rng.standard_normal(n) for _ in range(12)]
        roles = ["noise"] * 12
    else:
        raise ValueError(f"unsupported single task: {task}")
    return zscore_rows(np.vstack(rows)), labels, standardize(g), roles


def make_selective_pair(seed, n):
    """Generate one F with two exchangeable targets and return both views."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    u = rng.standard_normal(n)
    labels_g = balanced_labels(g, 0.75, rng)
    labels_u = balanced_labels(u, 0.75, rng)
    rows = [g + 0.75 * rng.standard_normal(n) for _ in range(6)]
    rows += [u + 0.10 * rng.standard_normal(n) for _ in range(6)]
    F = zscore_rows(np.vstack(rows))
    roles = ["g_block"] * 6 + ["u_block"] * 6
    return F, {
        "selective_target_signal": (labels_g, standardize(g), roles),
        "selective_target_nuisance": (labels_u, standardize(u), roles),
    }


def array_hash(array):
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def sparse_hash(matrix):
    matrix = matrix.copy().tocsr()
    matrix.sort_indices()
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(array_hash(array).encode("ascii"))
    return digest.hexdigest()


def prepare_unlabeled(F, args):
    """Compute every artifact that is forbidden from seeing target labels."""
    dufs_gates, dufs_diagnostic = adapted_dufs_soft_gates(
        F, seeds=(11, 23, 37), epochs=args.dufs_epochs
    )
    dufs_fit = laplacian_iu_fit(F, lambda_=LAMBDA, gates=dufs_gates, k=args.k)
    iu_weights = dufs_fit.baseline.w
    iu_scores = iu_weights @ F
    ridge = projected_ridge_fit(F, lambda_=LAMBDA)
    pseudo = pseudo_anchor_laplacian_fit(
        F, iu_scores, lambda_=LAMBDA, k=args.k
    )
    u2, u2_basis = ordinary_u2_coordinates(F)
    hashes = {
        "F": array_hash(F),
        "iu_weights": array_hash(iu_weights),
        "iu_scores": array_hash(iu_scores),
        "dufs_gates": array_hash(dufs_gates),
        "dufs_graph": sparse_hash(dufs_fit.graph),
        "dufs_liu_weights": array_hash(dufs_fit.w),
        "dufs_liu_scores": array_hash(dufs_fit.w @ F),
        "projected_ridge_weights": array_hash(ridge.w),
        "projected_ridge_scores": array_hash(ridge.w @ F),
        "pseudo_gates": array_hash(pseudo.gate_result.gates),
        "pseudo_weights": array_hash(pseudo.fit.w),
        "pseudo_scores": array_hash(pseudo.fit.w @ F),
    }
    return {
        "F": F,
        "dufs_gates": dufs_gates,
        "dufs_diagnostic": dufs_diagnostic,
        "dufs_fit": dufs_fit,
        "iu_scores": iu_scores,
        "ridge": ridge,
        "pseudo": pseudo,
        "u2": u2,
        "u2_basis": u2_basis,
        "hashes": hashes,
    }


def calibration_permutations(seed, n, draws):
    output = []
    for draw in range(draws):
        rng = np.random.default_rng(seed + 900_001 + draw * 7_919)
        output.append(rng.permutation(n))
    return output


def safe_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def run_target_view(split, replicate, seed, task, F, labels, latent, roles,
                    prepared, permutations, args):
    oracle_graph = self_tuning_knn_graph(latent[:, None], k=args.k)
    oracle = laplacian_iu_fit(F, lambda_=LAMBDA, graph=oracle_graph, k=args.k)
    fixed_scores = {
        "iu": prepared["iu_scores"],
        "dufs_liu": prepared["dufs_fit"].w @ F,
        "projected_ridge": prepared["ridge"].w @ F,
        "pseudo_anchor": prepared["pseudo"].fit.w @ F,
        "oracle_latent": oracle.w @ F,
    }
    rows = []
    gate_rows = []

    raw_probabilities = np.asarray(
        prepared["dufs_diagnostic"]["raw_probabilities"]
    )
    for feature, role in enumerate(roles):
        gate_rows.append({
            "version": VERSION, "split": split, "replicate": replicate,
            "seed": seed, "task": task, "draw": -1, "budget": 0,
            "gate_source": "dufs", "feature": feature,
            "planted_role": role, "gate_value": float(prepared["dufs_gates"][feature]),
            "raw_correlation": "", "fallback": False,
            "fallback_reason": "", "raw_probability": float(raw_probabilities[feature]),
        })
        gate_rows.append({
            "version": VERSION, "split": split, "replicate": replicate,
            "seed": seed, "task": task, "draw": -1, "budget": 0,
            "gate_source": "pseudo_anchor", "feature": feature,
            "planted_role": role,
            "gate_value": float(prepared["pseudo"].gate_result.gates[feature]),
            "raw_correlation": float(
                prepared["pseudo"].gate_result.correlations[feature]
            ),
            "fallback": prepared["pseudo"].gate_result.fallback,
            "fallback_reason": prepared["pseudo"].gate_result.fallback_reason,
            "raw_probability": "",
        })

    for draw, permutation in enumerate(permutations):
        for budget in BUDGETS:
            calibration = permutation[:budget]
            evaluation_mask = np.ones(F.shape[1], dtype=bool)
            evaluation_mask[calibration] = False
            evaluation = np.flatnonzero(evaluation_mask)
            index_hash = array_hash(calibration)

            anchored = target_anchored_laplacian_fit(
                F, labels, calibration, lambda_=LAMBDA, k=args.k
            )
            u2_scores, u2_diagnostic = fixed_logistic_scores(
                prepared["u2"], labels, calibration
            )
            full_scores, full_diagnostic = fixed_logistic_scores(
                F.T, labels, calibration
            )
            scores_by_method = {
                **fixed_scores,
                "ta_liu": anchored.fit.w @ F,
                "u2_logistic": u2_scores,
                "full_logistic": full_scores,
            }
            diagnostics = {
                "ta_liu": (
                    anchored.gate_result.fallback,
                    anchored.gate_result.fallback_reason,
                ),
                "u2_logistic": (
                    u2_diagnostic["fallback"],
                    u2_diagnostic["fallback_reason"],
                ),
                "full_logistic": (
                    full_diagnostic["fallback"],
                    full_diagnostic["fallback_reason"],
                ),
            }
            iu_auroc, iu_auprc = safe_metrics(
                labels[evaluation], scores_by_method["iu"][evaluation]
            )
            for method in METHODS:
                auroc, auprc = safe_metrics(
                    labels[evaluation], scores_by_method[method][evaluation]
                )
                fallback, reason = diagnostics.get(method, (False, ""))
                rows.append({
                    "version": VERSION,
                    "split": split,
                    "replicate": replicate,
                    "seed": seed,
                    "task": task,
                    "draw": draw,
                    "budget": budget,
                    "calibration_index_hash": index_hash,
                    "n_evaluation": int(len(evaluation)),
                    "method": method,
                    "auroc": auroc,
                    "auprc": auprc,
                    "auroc_delta_vs_iu": auroc - iu_auroc,
                    "auprc_delta_vs_iu": auprc - iu_auprc,
                    "fallback": bool(fallback),
                    "fallback_reason": reason,
                    "calibration_positive_rate": float(labels[calibration].mean()),
                })
            for feature, role in enumerate(roles):
                gate_rows.append({
                    "version": VERSION, "split": split, "replicate": replicate,
                    "seed": seed, "task": task, "draw": draw, "budget": budget,
                    "gate_source": "ta_liu", "feature": feature,
                    "planted_role": role,
                    "gate_value": float(anchored.gate_result.gates[feature]),
                    "raw_correlation": float(
                        anchored.gate_result.correlations[feature]
                    ),
                    "fallback": anchored.gate_result.fallback,
                    "fallback_reason": anchored.gate_result.fallback_reason,
                    "raw_probability": "",
                })
    return rows, gate_rows


def dataset_seed(offset, replicate, task_index):
    return int(offset + replicate * 1_009 + task_index * 100_003)


def run_split(split, count, offset, args):
    rows, gates, invariants = [], [], []
    completed = 0
    total = count * 5
    for replicate in range(count):
        for task_index, task in enumerate((
            "smooth_signal", "nuisance_manifold", "correlated_errors", "pure_noise"
        )):
            seed_index = (0, 1, 3, 4)[task_index]
            seed = dataset_seed(offset, replicate, seed_index)
            F, labels, latent, roles = make_single_task(task, seed, args.n)
            prepared = prepare_unlabeled(F, args)
            permutations = calibration_permutations(
                seed, args.n, args.calibration_draws
            )
            batch, gate_batch = run_target_view(
                split, replicate, seed, task, F, labels, latent, roles,
                prepared, permutations, args,
            )
            rows.extend(batch)
            gates.extend(gate_batch)
            completed += 1
            print(f"[{completed:03d}/{total:03d}] {split} r={replicate} {task}",
                  flush=True)

        pair_seed = dataset_seed(offset, replicate, 2)
        F, views = make_selective_pair(pair_seed, args.n)
        prepared_g = prepare_unlabeled(F, args)
        prepared_u = prepare_unlabeled(F.copy(), args)
        equality = {
            key: prepared_g["hashes"][key] == prepared_u["hashes"][key]
            for key in prepared_g["hashes"]
        }
        invariant = {
            "version": VERSION,
            "split": split,
            "replicate": replicate,
            "seed": pair_seed,
            "F_hash_signal": prepared_g["hashes"]["F"],
            "F_hash_nuisance": prepared_u["hashes"]["F"],
            **{f"equal_{key}": value for key, value in equality.items()},
            "all_declared_artifacts_equal": bool(all(equality.values())),
        }
        if not invariant["all_declared_artifacts_equal"]:
            failed = [key for key, value in equality.items() if not value]
            raise AssertionError(
                "label-swap invariant failed for: " + ", ".join(failed)
            )
        permutations = calibration_permutations(
            pair_seed, args.n, args.calibration_draws
        )
        pair_batches = {}
        for task, prepared in (
            ("selective_target_signal", prepared_g),
            ("selective_target_nuisance", prepared_u),
        ):
            labels, latent, roles = views[task]
            batch, gate_batch = run_target_view(
                split, replicate, pair_seed, task, F, labels, latent, roles,
                prepared, permutations, args,
            )
            rows.extend(batch)
            gates.extend(gate_batch)
            pair_batches[task] = batch
        calibration_maps = {}
        for task, batch in pair_batches.items():
            calibration_maps[task] = {
                (int(row["draw"]), int(row["budget"])): row["calibration_index_hash"]
                for row in batch if row["method"] == "iu"
            }
        calibration_equal = (
            calibration_maps["selective_target_signal"]
            == calibration_maps["selective_target_nuisance"]
            and len(calibration_maps["selective_target_signal"])
            == args.calibration_draws * len(BUDGETS)
        )
        invariant["calibration_hash_bundle_signal"] = array_hash(
            np.asarray([
                value for _, value in sorted(
                    calibration_maps["selective_target_signal"].items()
                )
            ], dtype="S64")
        )
        invariant["calibration_hash_bundle_nuisance"] = array_hash(
            np.asarray([
                value for _, value in sorted(
                    calibration_maps["selective_target_nuisance"].items()
                )
            ], dtype="S64")
        )
        invariant["equal_calibration_indices"] = bool(calibration_equal)
        invariant["all_declared_artifacts_equal"] = bool(
            invariant["all_declared_artifacts_equal"] and calibration_equal
        )
        if not calibration_equal:
            raise AssertionError("paired targets did not use identical calibration indices")
        invariants.append(invariant)
        completed += 1
        print(f"[{completed:03d}/{total:03d}] {split} r={replicate} selective_pair",
              flush=True)
    return {"rows": rows, "gates": gates, "invariants": invariants}


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def one_sided_lower(mean, se, n):
    return float(mean - student_t.ppf(0.95, n - 1) * se) if n > 1 else float(mean)


def two_sided_interval(mean, se, n):
    radius = student_t.ppf(0.975, n - 1) * se if n > 1 else 0.0
    return float(mean - radius), float(mean + radius)


def aggregate_datasets(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["split"], row["task"], row["replicate"], row["seed"],
            row["budget"], row["method"],
        )
        groups[key].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        split, task, replicate, seed, budget, method = key
        output.append({
            "version": VERSION,
            "split": split,
            "task": task,
            "replicate": replicate,
            "seed": seed,
            "budget": budget,
            "method": method,
            "n_calibration_draws": len(subset),
            "auroc": float(np.mean([row["auroc"] for row in subset])),
            "auprc": float(np.mean([row["auprc"] for row in subset])),
            "auroc_delta_vs_iu": float(np.mean([
                row["auroc_delta_vs_iu"] for row in subset
            ])),
            "auprc_delta_vs_iu": float(np.mean([
                row["auprc_delta_vs_iu"] for row in subset
            ])),
            "fallback_fraction": float(np.mean([row["fallback"] for row in subset])),
        })
    return output


def aggregate_summary(per_dataset):
    groups = defaultdict(list)
    for row in per_dataset:
        groups[(row["split"], row["task"], row["budget"], row["method"])].append(row)
    output = []
    for key, subset in sorted(groups.items()):
        split, task, budget, method = key
        record = {
            "version": VERSION, "split": split, "task": task,
            "budget": budget, "method": method, "n_datasets": len(subset),
        }
        for metric in ("auroc", "auprc", "auroc_delta_vs_iu",
                       "auprc_delta_vs_iu", "fallback_fraction"):
            mean, se = mean_se([row[metric] for row in subset])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_se"] = se
        output.append(record)
    return output


def paired_contrast(per_dataset, split, task, left, right, budget=PRIMARY_BUDGET):
    selected = [row for row in per_dataset
                if row["split"] == split and row["task"] == task
                and row["budget"] == budget]
    lookup = {(row["replicate"], row["method"]): row for row in selected}
    replicates = sorted({row["replicate"] for row in selected})
    values = [lookup[(replicate, left)]["auroc"]
              - lookup[(replicate, right)]["auroc"] for replicate in replicates]
    mean, se = mean_se(values)
    lower = one_sided_lower(mean, se, len(values))
    low_two, high_two = two_sided_interval(mean, se, len(values))
    leave_one_out = [float(np.mean(values[:index] + values[index + 1:]))
                     for index in range(len(values))] if len(values) > 1 else values
    return {
        "task": task, "left": left, "right": right, "budget": budget,
        "n_datasets": len(values), "values": values, "mean": mean, "se": se,
        "one_sided_95_lower": lower,
        "two_sided_95_low": low_two, "two_sided_95_high": high_two,
        "wins": int(np.sum(np.asarray(values) > 0)),
        "leave_one_out_min": float(np.min(leave_one_out)),
        "leave_one_out_max": float(np.max(leave_one_out)),
    }


def evaluate_gates(per_dataset, invariants, split):
    contrast = lambda task, left, right: paired_contrast(
        per_dataset, split, task, left, right, PRIMARY_BUDGET
    )
    rescue_iu = contrast("selective_target_signal", "ta_liu", "iu")
    rescue_pseudo = contrast("selective_target_signal", "ta_liu", "pseudo_anchor")
    swap_iu = contrast("selective_target_nuisance", "ta_liu", "iu")
    same_label = {}
    for task in ("selective_target_signal", "selective_target_nuisance"):
        same_label[task] = {
            method: contrast(task, "ta_liu", method)
            for method in ("u2_logistic", "full_logistic")
        }
    smooth = contrast("smooth_signal", "ta_liu", "dufs_liu")
    nuisance = contrast("nuisance_manifold", "ta_liu", "iu")
    correlated = contrast("correlated_errors", "ta_liu", "iu")
    noise_delta = contrast("pure_noise", "ta_liu", "iu")
    noise_rows = [row for row in per_dataset
                  if row["split"] == split and row["task"] == "pure_noise"
                  and row["budget"] == PRIMARY_BUDGET and row["method"] == "ta_liu"]
    noise_mean, noise_se = mean_se([row["auroc"] for row in noise_rows])
    noise_low, noise_high = two_sided_interval(
        noise_mean, noise_se, len(noise_rows)
    )

    meaningful = lambda item: (
        item["mean"] >= 0.005
        and item["one_sided_95_lower"] > 0.0
        and item["wins"] >= 6
    )
    noninferior = lambda item: item["one_sided_95_lower"] >= -0.005
    split_invariants = [row for row in invariants if row["split"] == split]
    gate_results = {
        "selective_nuisance_rescue": meaningful(rescue_iu) and meaningful(rescue_pseudo),
        "label_swap_consistency": meaningful(swap_iu),
        "same_label_attribution": all(
            noninferior(item) for task in same_label.values() for item in task.values()
        ),
        "smooth_preservation": noninferior(smooth),
        "existing_nuisance_safety": noninferior(nuisance),
        "correlated_error_safety": noninferior(correlated),
        "null_safety": (
            noise_low >= 0.45 and noise_high <= 0.55
            and noise_delta["two_sided_95_low"] >= -0.005
            and noise_delta["two_sided_95_high"] <= 0.005
            and noise_delta["one_sided_95_lower"] <= 0.0
        ),
        "identifiability_invariants": bool(split_invariants) and all(
            row["all_declared_artifacts_equal"] for row in split_invariants
        ),
    }
    return {
        "version": VERSION,
        "split": split,
        "primary_budget": PRIMARY_BUDGET,
        "contrasts": {
            "rescue_vs_iu": rescue_iu,
            "rescue_vs_pseudo": rescue_pseudo,
            "swap_vs_iu": swap_iu,
            "same_label": same_label,
            "smooth_vs_dufs": smooth,
            "nuisance_vs_iu": nuisance,
            "correlated_vs_iu": correlated,
            "pure_noise_delta_vs_iu": noise_delta,
            "pure_noise_ta_auroc": {
                "n_datasets": len(noise_rows), "mean": noise_mean, "se": noise_se,
                "two_sided_95_low": noise_low, "two_sided_95_high": noise_high,
            },
        },
        "gates": gate_results,
        "overall_pass": bool(all(gate_results.values())),
        "development_is_exploratory": split == "development",
    }


def select_rows(rows, **filters):
    return [row for row in rows
            if all(row.get(key) == value for key, value in filters.items())]


def plot_budget_curves(per_dataset, split, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    curve_methods = ("ta_liu", "u2_logistic", "full_logistic", "dufs_liu",
                     "pseudo_anchor")
    for ax, task in zip(axes.ravel(), TASKS):
        for method in curve_methods:
            means, errors = [], []
            for budget in BUDGETS:
                values = [100 * row["auroc_delta_vs_iu"] for row in select_rows(
                    per_dataset, split=split, task=task, budget=budget, method=method
                )]
                mean, se = mean_se(values)
                means.append(mean)
                errors.append(se)
            ax.errorbar(BUDGETS, means, yerr=errors, marker="o", ms=4,
                        lw=1.4, capsize=2, color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])
        ax.axhline(0, color="#111827", lw=0.8)
        ax.axvline(PRIMARY_BUDGET, color="#111827", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(value) for value in BUDGETS])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("calibration labels")
        ax.set_ylabel("AUROC change vs IU (points)")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{split.title()} sample-efficiency curves (dataset mean ± SE)",
                 y=1.06, fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"01_{split}_budget_curves.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_label_swap(per_dataset, split, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, (
        "selective_target_signal", "selective_target_nuisance"
    )):
        for method in ("ta_liu", "u2_logistic", "full_logistic", "dufs_liu",
                       "pseudo_anchor"):
            means, errors = [], []
            for budget in BUDGETS:
                values = [100 * row["auroc_delta_vs_iu"] for row in select_rows(
                    per_dataset, split=split, task=task, budget=budget, method=method
                )]
                mean, se = mean_se(values)
                means.append(mean)
                errors.append(se)
            ax.errorbar(BUDGETS, means, yerr=errors, marker="o", ms=4,
                        capsize=2, color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])
        ax.axhline(0, color="#111827", lw=0.8)
        ax.axvline(PRIMARY_BUDGET, color="#111827", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(value) for value in BUDGETS])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("calibration labels")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("AUROC change vs IU (points)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Same feature matrix, swapped target", y=1.13, fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"02_{split}_label_swap.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_role_gates(gate_rows, split, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, task in zip(axes, (
        "selective_target_signal", "selective_target_nuisance"
    )):
        selected = select_rows(
            gate_rows, split=split, task=task, gate_source="ta_liu"
        )
        for role, color in (("g_block", "#2563eb"), ("u_block", "#d97706")):
            means, errors = [], []
            for budget in BUDGETS:
                by_dataset = defaultdict(list)
                for row in selected:
                    if row["budget"] == budget and row["planted_role"] == role:
                        by_dataset[row["replicate"]].append(row["gate_value"])
                values = [float(np.mean(value)) for value in by_dataset.values()]
                mean, se = mean_se(values)
                means.append(mean)
                errors.append(se)
            ax.errorbar(BUDGETS, means, yerr=errors, marker="o", capsize=2,
                        color=color, label=role.replace("_", " "))
        ax.axvline(PRIMARY_BUDGET, color="#111827", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(value) for value in BUDGETS])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("calibration labels")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("TA graph gate (RMS normalized)")
    axes[1].legend(frameon=False)
    fig.suptitle("Does the target anchor switch the planted graph role?", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"03_{split}_role_gates.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_primary_controls(per_dataset, split, out_dir):
    methods = tuple(method for method in METHODS if method != "iu")
    fig, ax = plt.subplots(figsize=(15, 5.2))
    x = np.arange(len(TASKS))
    width = 0.11
    for index, method in enumerate(methods):
        means, errors = [], []
        for task in TASKS:
            values = [100 * row["auroc_delta_vs_iu"] for row in select_rows(
                per_dataset, split=split, task=task, budget=PRIMARY_BUDGET,
                method=method,
            )]
            mean, se = mean_se(values)
            means.append(mean)
            errors.append(se)
        center = (len(methods) - 1) / 2
        ax.bar(x + (index - center) * width, means, width, yerr=errors,
               capsize=2, color=METHOD_COLORS[method], alpha=0.9,
               label=METHOD_LABELS[method])
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in TASKS], rotation=15,
                       ha="right")
    ax.set_ylabel("AUROC change vs IU (points)")
    ax.set_title(f"Frozen k={PRIMARY_BUDGET} candidate against all controls")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, f"04_{split}_primary_controls.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def write_json_gz(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle)


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def dependency_versions():
    import torch
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:5]),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
    }


def source_hashes():
    relative_paths = (
        SPEC,
        "spectral_utils/adapted_dufs.py",
        "spectral_utils/target_anchored_laplacian.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/fusion_utils.py",
        "scripts/target_anchored_laplacian_synthetic.py",
        "scripts/test_target_anchored_laplacian.py",
    )
    output = {}
    for relative in relative_paths:
        with open(os.path.join(REPO, relative), "rb") as handle:
            output[relative] = hashlib.sha256(handle.read()).hexdigest()
    return output


def experiment_config(args):
    return {
        "version": VERSION,
        "development_seed_offset": DEVELOPMENT_SEED_OFFSET,
        "confirmation_seed_offset": CONFIRMATION_SEED_OFFSET,
        "dev_replicates": args.dev_replicates,
        "confirm_replicates": args.confirm_replicates,
        "calibration_draws": args.calibration_draws,
        "n": args.n,
        "k": args.k,
        "dufs_epochs": args.dufs_epochs,
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "lambda": LAMBDA,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "dufs_seeds": [11, 23, 37],
        "logistic": {
            "penalty": "l2", "C": 1.0, "fit_intercept": True,
            "class_weight": None, "solver": "lbfgs", "max_iter": 1000,
            "tol": 1e-8, "random_state": 0,
        },
    }


def is_canonical_config(args):
    return all(getattr(args, key) == value
               for key, value in CANONICAL_CONFIG.items())


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unavailable"


def write_report(path, split, per_dataset, evaluation, args):
    lines = [
        "# Target-anchored Laplacian IU-PCR synthetic study",
        "",
        f"Version: `{VERSION}`",
        "",
        "## Scope",
        "",
        f"This report contains the **{split}** split only. It used generated data; no",
        "real hallucination features or cached benchmark data were opened.",
        "Development uses an already-consumed seed block and is exploratory. It cannot",
        "establish the preregistered claim or change the frozen primary budget k=16.",
        "",
        "## Design",
        "",
        f"- Dataset replicates: {args.dev_replicates if split == 'development' else args.confirm_replicates}",
        f"- Calibration permutations per dataset: {args.calibration_draws}",
        f"- Samples per dataset: {args.n}; graph k-NN: {args.k}; lambda: {LAMBDA}",
        f"- Nested label budgets: {list(BUDGETS)}; frozen primary budget: {PRIMARY_BUDGET}",
        "- Calibration draws are averaged within each dataset. Dataset replicates are",
        "  the uncertainty units in every confidence interval.",
        "- The paired targets use identical F and identical calibration indices.",
        f"- Eligible for reserved confirmation: {'yes' if is_canonical_config(args) else 'no (debug configuration)'}.",
        "",
        "## Frozen k=16 results",
        "",
        "Values are AUROC changes in percentage points versus ordinary IU-PCR,",
        "reported as dataset mean +/- one SE.",
        "",
        "| task | " + " | ".join(METHOD_LABELS[m] for m in METHODS if m != "iu") + " |",
        "|---|" + "---:|" * (len(METHODS) - 1),
    ]
    for task in TASKS:
        cells = []
        for method in METHODS:
            if method == "iu":
                continue
            values = [100 * row["auroc_delta_vs_iu"] for row in select_rows(
                per_dataset, split=split, task=task, budget=PRIMARY_BUDGET,
                method=method,
            )]
            mean, se = mean_se(values)
            cells.append(f"{mean:+.3f} +/- {se:.3f}")
        lines.append(f"| {TASK_LABELS[task]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Preregistered-gate diagnostic",
        "",
        f"Because this is {split}, the result below is "
        + ("exploratory only." if split == "development" else "the confirmatory result."),
        "",
        "| gate | result |",
        "|---|---:|",
    ]
    for name, passed in evaluation["gates"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        f"Overall: **{'PASS' if evaluation['overall_pass'] else 'FAIL'}**",
        "",
        "The raw CSV retains every calibration draw. `per_dataset.csv` first averages",
        "those draws; `summary.csv`, confidence bounds, win counts, and leave-one-dataset-",
        "out diagnostics all use the dataset replicate as the effective sample size.",
        "",
        "## Plots",
        "",
        f"- `01_{split}_budget_curves.png`: sample efficiency on all tasks.",
        f"- `02_{split}_label_swap.png`: identical F under the two targets.",
        f"- `03_{split}_role_gates.png`: whether target gates switch planted roles.",
        f"- `04_{split}_primary_controls.png`: frozen k=16 against every control.",
        "",
        (
            "Stop for interpretation before opening the reserved confirmation split or any"
            if split == "development"
            else "Stop for independent audit and interpretation before opening any"
        ),
        "real hallucination data.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def persist_stage(stage_result, split, out_dir, args):
    per_dataset = aggregate_datasets(stage_result["rows"])
    summary = aggregate_summary(per_dataset)
    evaluation = evaluate_gates(per_dataset, stage_result["invariants"], split)
    write_csv(os.path.join(out_dir, f"{split}_per_draw.csv"), stage_result["rows"])
    write_csv(os.path.join(out_dir, f"{split}_gate_features.csv"), stage_result["gates"])
    write_csv(os.path.join(out_dir, f"{split}_invariants.csv"), stage_result["invariants"])
    write_csv(os.path.join(out_dir, f"{split}_per_dataset.csv"), per_dataset)
    write_csv(os.path.join(out_dir, f"{split}_summary.csv"), summary)
    write_json(os.path.join(out_dir, f"{split}_evaluation.json"), evaluation)
    plot_budget_curves(per_dataset, split, out_dir)
    plot_label_swap(per_dataset, split, out_dir)
    plot_role_gates(stage_result["gates"], split, out_dir)
    plot_primary_controls(per_dataset, split, out_dir)
    write_report(os.path.join(out_dir, f"{split.upper()}_REPORT.md"), split,
                 per_dataset, evaluation, args)
    return per_dataset, summary, evaluation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("development", "confirmation"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--dev-replicates", type=int, default=8)
    parser.add_argument("--confirm-replicates", type=int, default=8)
    parser.add_argument("--calibration-draws", type=int, default=16)
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--dufs-epochs", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dev_replicates < 2 or args.confirm_replicates < 2:
        raise ValueError("at least two dataset replicates are required")
    if args.calibration_draws < 1:
        raise ValueError("at least one calibration draw is required")
    if max(BUDGETS) >= args.n:
        raise ValueError("every calibration budget must leave evaluation samples")
    os.makedirs(args.out_dir, exist_ok=True)
    frozen_path = os.path.join(args.out_dir, "frozen_protocol.json")
    development_path = os.path.join(args.out_dir, "development_artifacts.json.gz")
    started = time.time()

    if args.stage == "development":
        result = run_split(
            "development", args.dev_replicates, DEVELOPMENT_SEED_OFFSET, args
        )
        _, _, evaluation = persist_stage(
            result, "development", args.out_dir, args
        )
        frozen = {
            "version": VERSION,
            "config": experiment_config(args),
            "source_hashes": source_hashes(),
            "dependencies": dependency_versions(),
            "development_command": list(sys.argv),
            "git_head": git_head(),
            "confirmation_eligible": is_canonical_config(args),
        }
        write_json(frozen_path, frozen)
        write_json_gz(development_path, result)
        write_json(os.path.join(args.out_dir, "development_run_metadata.json"), {
            **frozen,
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": time.time() - started,
            "confirmation_seed_generated": False,
        })
        print(json.dumps({
            "stage": "development",
            "overall_exploratory_pass": evaluation["overall_pass"],
            "elapsed_seconds": time.time() - started,
            "confirmation_seed_generated": False,
            "stop_for_discussion": True,
        }, indent=2))
        return

    if not os.path.exists(frozen_path) or not os.path.exists(development_path):
        raise FileNotFoundError("run --stage development before confirmation")
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    if not frozen.get("confirmation_eligible", False):
        raise ValueError(
            "debug/development configuration is not eligible for reserved confirmation"
        )
    if not is_canonical_config(args):
        raise ValueError("reserved confirmation requires the canonical preregistered config")
    if frozen["config"] != experiment_config(args):
        raise ValueError("confirmation config differs from the frozen protocol")
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("source or specification changed after development freeze")
    if frozen["dependencies"] != dependency_versions():
        raise ValueError("dependency versions changed after development freeze")

    confirmation = run_split(
        "confirmation", args.confirm_replicates, CONFIRMATION_SEED_OFFSET, args
    )
    _, _, evaluation = persist_stage(
        confirmation, "confirmation", args.out_dir, args
    )
    write_json_gz(
        os.path.join(args.out_dir, "confirmation_artifacts.json.gz"), confirmation
    )
    write_json(os.path.join(args.out_dir, "confirmation_run_metadata.json"), {
        "version": VERSION,
        "development_command": frozen["development_command"],
        "confirmation_command": list(sys.argv),
        "config": experiment_config(args),
        "source_hashes": source_hashes(),
        "dependencies": dependency_versions(),
        "git_head": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
    })
    print(json.dumps({
        "stage": "confirmation",
        "overall_pass": evaluation["overall_pass"],
        "elapsed_seconds": time.time() - started,
        "stop_for_independent_audit": True,
    }, indent=2))


if __name__ == "__main__":
    main()
