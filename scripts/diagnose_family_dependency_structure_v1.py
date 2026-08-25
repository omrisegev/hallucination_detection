#!/usr/bin/env python3
"""Audit within- and between-family dependence on the frozen 24-cell panel.

The ``fit`` phase is target-free and serializes every representation and
dependence statistic before labels can be imported.  The ``evaluate`` phase
first verifies the frozen fit manifest, then dynamically imports the label
sidecar module and measures dependence remaining after conditioning on the
true binary state.  This is a diagnostic, not a model-selection run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    family_index_map,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
)


SCHEMA = "family_dependency_structure_v1"
SEEDS = (0, 1, 2, 3, 4)


def _zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (X - mean) / scale


def _rank_columns(X: np.ndarray) -> np.ndarray:
    return _zscore(np.column_stack([rankdata(X[:, j], method="average") for j in range(X.shape[1])]))


def _corr(X: np.ndarray) -> np.ndarray:
    Z = _zscore(X)
    return np.clip((Z.T @ Z) / max(len(Z), 1), -1.0, 1.0)


def _residualize(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    B = np.column_stack([np.ones(len(X)), np.asarray(basis, dtype=float)])
    coef = np.linalg.lstsq(B, X, rcond=None)[0]
    return X - B @ coef


def _class_residual(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    output = np.asarray(X, dtype=float).copy()
    for value in (0, 1):
        mask = y == value
        if int(mask.sum()) < 2:
            raise ValueError("each class needs at least two rows")
        output[mask] -= output[mask].mean(axis=0, keepdims=True)
    return output


def _class_length_residual(X: np.ndarray, y: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    output = np.zeros_like(np.asarray(X, dtype=float))
    basis = _length_basis(lengths)
    for value in (0, 1):
        mask = y == value
        if int(mask.sum()) < basis.shape[1] + 2:
            raise ValueError("class too small for class-specific length residualization")
        output[mask] = _residualize(np.asarray(X, dtype=float)[mask], basis[mask])
    return output


def _effective_rank(X: np.ndarray) -> float:
    values = np.maximum(np.linalg.eigvalsh(_corr(X)), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    p = values / total
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def _distance_correlation(x: np.ndarray, y: np.ndarray, *, max_n: int = 512) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) > max_n:
        indices = np.linspace(0, len(x) - 1, max_n, dtype=int)
        x, y = x[indices], y[indices]
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])
    A -= A.mean(axis=0, keepdims=True)
    A -= A.mean(axis=1, keepdims=True)
    A += A.mean()
    B -= B.mean(axis=0, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    B += B.mean()
    dcov2 = float(np.mean(A * B))
    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))
    denom = np.sqrt(max(dvarx2 * dvary2, 0.0))
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 1e-15 else 0.0


def _group_splits(groups: Iterable[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(tuple(groups))
    unique = np.unique(groups)
    folds = min(5, len(unique))
    if folds < 2:
        raise ValueError("cross-fitting requires at least two source groups")
    dummy = np.zeros((len(groups), 1))
    return list(GroupKFold(n_splits=folds).split(dummy, groups=groups))


def _oof_r2(target: np.ndarray, predictors: np.ndarray, groups: Iterable[str]) -> float:
    target = np.asarray(target, dtype=float)
    predictors = np.asarray(predictors, dtype=float)
    if predictors.ndim != 2 or predictors.shape[1] == 0 or target.std() <= 1e-12:
        return float("nan")
    pred = np.zeros(len(target), dtype=float)
    for train, held in _group_splits(groups):
        model = Ridge(alpha=1.0).fit(_zscore(predictors[train]), target[train])
        mean = predictors[train].mean(axis=0)
        scale = predictors[train].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        pred[held] = model.predict((predictors[held] - mean) / scale)
    denom = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - np.sum((target - pred) ** 2) / denom) if denom > 1e-12 else float("nan")


def _crossfit_pair_excluded_residuals(
    values: np.ndarray,
    left: int,
    right: int,
    groups: Iterable[str],
    raw_trace_length: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.zeros((len(values), 2), dtype=float)
    splits = _group_splits(groups)
    predictors = [index for index in range(values.shape[1]) if index not in {left, right}]
    log_length = np.log1p(np.asarray(raw_trace_length, dtype=float))
    for train, held in splits:
        family_mean = values[train][:, predictors].mean(axis=0)
        family_scale = values[train][:, predictors].std(axis=0)
        family_scale = np.where(family_scale > 1e-12, family_scale, 1.0)
        length_mean = float(log_length[train].mean())
        length_scale = max(float(log_length[train].std()), 1e-12)
        donor_length = (log_length[train] - length_mean) / length_scale
        held_length = (log_length[held] - length_mean) / length_scale
        donor_design = np.column_stack([
            (values[train][:, predictors] - family_mean) / family_scale,
            donor_length,
            donor_length ** 2,
            donor_length ** 3,
        ])
        held_design = np.column_stack([
            (values[held][:, predictors] - family_mean) / family_scale,
            held_length,
            held_length ** 2,
            held_length ** 3,
        ])
        targets = values[train][:, [left, right]]
        model = Ridge(alpha=1.0).fit(donor_design, targets)
        prediction = model.predict(held_design)
        target_scale = targets.std(axis=0)
        target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
        output[held] = (values[held][:, [left, right]] - prediction) / target_scale
    return output


def _spectral_summary(values: np.ndarray) -> dict[str, float]:
    correlation = _corr(values)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = max(float(eigenvalues.sum()), 1e-12)
    rank2 = (eigenvectors[:, :2] * eigenvalues[:2]) @ eigenvectors[:, :2].T
    upper = np.triu_indices(len(correlation), 1)
    residual = correlation - rank2
    return {
        "top1_variance_fraction": float(eigenvalues[0] / total),
        "top2_variance_fraction": float(eigenvalues[:2].sum() / total),
        "rank2_offdiag_residual_abs_mean": float(np.mean(np.abs(residual[upper]))),
        "rank2_offdiag_residual_fraction_gt_0_10": float(np.mean(np.abs(residual[upper]) > 0.10)),
    }


def _stability_summary(records: list[tuple[str, tuple[str, ...], np.ndarray, np.ndarray]]) -> dict:
    if len(records) < 2:
        raise ValueError("stability summary needs at least two aligned cells")
    vectors, projectors, matrices, dataset_families = [], [], [], []
    for dataset_family, names, values, lengths in records:
        if len(names) != 6:
            continue
        adjusted = _zscore(_residualize(values, _length_basis(lengths)))
        correlation = _corr(adjusted)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        order = np.argsort(eigenvalues)[::-1]
        top = eigenvectors[:, order[0]]
        if top.sum() < 0:
            top = -top
        vectors.append(top)
        projectors.append(eigenvectors[:, order[:2]] @ eigenvectors[:, order[:2]].T)
        matrices.append(correlation)
        dataset_families.append(dataset_family)
    upper = np.triu_indices(6, 1)
    cosines, rank2_similarities, edge_correlations = [], [], []
    for left, right in itertools.combinations(range(len(vectors)), 2):
        cosines.append(abs(float(vectors[left] @ vectors[right])))
        rank2_similarities.append(float(np.trace(projectors[left] @ projectors[right]) / 2.0))
        edge_correlations.append(float(np.corrcoef(matrices[left][upper], matrices[right][upper])[0, 1]))
    family_mean_matrices = []
    for family in sorted(set(dataset_families)):
        family_mean_matrices.append(np.mean([
            matrix for matrix, label in zip(matrices, dataset_families) if label == family
        ], axis=0))
    family_edge_correlations = [
        float(np.corrcoef(left[upper], right[upper])[0, 1])
        for left, right in itertools.combinations(family_mean_matrices, 2)
    ]
    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {"mean": float(array.mean()), "median": float(np.median(array)), "min": float(array.min())}
    return {
        "n_complete_six_family_cells": len(vectors),
        "length_adjusted_top1_absolute_cosine": stats(cosines),
        "length_adjusted_rank2_projector_similarity": stats(rank2_similarities),
        "length_adjusted_edge_pattern_correlation": stats(edge_correlations),
        "dataset_family_mean_edge_pattern_correlation": stats(family_edge_correlations),
    }


def _load_b3(baseline_dir: Path, cell_id: str, n: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    logits = []
    family_rows: dict[str, list[np.ndarray]] = {}
    for seed in SEEDS:
        path = baseline_dir / "fits" / cell_id / f"B3__seed{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            logit = np.asarray(data["logit"], dtype=float)
            if logit.shape != (n,):
                raise ValueError(f"B3 row mismatch: {cell_id} seed {seed}")
            logits.append(logit)
            keys = sorted(name for name in data.files if name.startswith("family_contribution__"))
            for key in keys:
                family = key.split("__", 1)[1]
                family_rows.setdefault(family, []).append(np.asarray(data[key], dtype=float))
    families = tuple(sorted(family_rows))
    if any(len(family_rows[name]) != len(SEEDS) for name in families):
        raise ValueError(f"incomplete B3 family contributions: {cell_id}")
    return (
        np.mean(np.stack(logits), axis=0),
        np.column_stack([np.mean(np.stack(family_rows[name]), axis=0) for name in families]),
        families,
    )


def _quotient_entropy_duplicate(values: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    required = {"entropy_level", "topk_distribution"}
    if not required.issubset(names):
        raise ValueError("entropy duplicate quotient requires both source families")
    indices = {name: names.index(name) for name in names}
    merged = 0.5 * (_zscore(values[:, indices["entropy_level"]][:, None])[:, 0]
                    + _zscore(values[:, indices["topk_distribution"]][:, None])[:, 0])
    output_names = tuple(name for name in names if name not in required) + ("token_entropy_quotient",)
    output = np.column_stack([
        values[:, indices[name]] for name in names if name not in required
    ] + [merged])
    return output, output_names


def _length_basis(raw_trace_length: np.ndarray) -> np.ndarray:
    value = _zscore(np.log1p(np.asarray(raw_trace_length, dtype=float))[:, None])[:, 0]
    return np.column_stack([value, value ** 2, value ** 3])


def _pair_rows(cell_id: str, dataset_family: str, names: tuple[str, ...], values: np.ndarray,
               group_ids: Iterable[str], raw_trace_length: np.ndarray,
               representation: str) -> list[dict]:
    pearson = _corr(values)
    spearman = _corr(_rank_columns(values))
    length_residual_values = _residualize(values, _length_basis(raw_trace_length))
    length_residual_corr = _corr(length_residual_values)
    rows = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_residual = _crossfit_pair_excluded_residuals(
                values, left, right, group_ids, raw_trace_length
            )
            partial_corr = _corr(pair_residual)[0, 1]
            rows.append({
                "cell_id": cell_id,
                "dataset_family": dataset_family,
                "representation": representation,
                "left": names[left],
                "right": names[right],
                "pearson": float(pearson[left, right]),
                "spearman": float(spearman[left, right]),
                "cubic_log_length_residual_pearson": float(length_residual_corr[left, right]),
                "crossfit_pair_excluded_partial_pearson": float(partial_corr),
                "distance_correlation": _distance_correlation(values[:, left], values[:, right]),
                "cubic_log_length_residual_distance_correlation": _distance_correlation(
                    length_residual_values[:, left], length_residual_values[:, right]
                ),
                "crossfit_pair_excluded_partial_distance_correlation": _distance_correlation(
                    pair_residual[:, 0], pair_residual[:, 1]
                ),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    family_rows: list[dict] = []
    pair_rows: list[dict] = []
    cell_rows: list[dict] = []
    artifacts = []
    raw_stability = []
    b3_stability = []
    for record in registry["cells"]:
        cell_id = record["cell_id"]
        bundle_path = args.bundle_dir / f"{cell_id}.npz"
        bundle = load_target_free_bundle(bundle_path)
        X, _, _ = donor_risk_matrix(bundle.X_raw, bundle.X_raw, bundle.feature_names)
        groups = family_index_map(bundle.feature_names)
        family_names = tuple(groups)
        raw_family = np.column_stack([X[:, groups[name]].mean(axis=1) for name in family_names])
        b3_logit, b3_family_unsorted, b3_names_sorted = _load_b3(args.baseline_dir, cell_id, len(X))
        b3_lookup = {name: b3_family_unsorted[:, i] for i, name in enumerate(b3_names_sorted)}
        if set(b3_lookup) != set(family_names):
            raise ValueError(f"B3/bundle family mismatch: {cell_id}")
        b3_family = np.column_stack([b3_lookup[name] for name in family_names])
        raw_stability.append((bundle.dataset_family, family_names, raw_family, bundle.raw_trace_length))
        b3_stability.append((bundle.dataset_family, family_names, b3_family, bundle.raw_trace_length))

        pair_rows.extend(_pair_rows(cell_id, bundle.dataset_family, family_names,
                                    raw_family, bundle.group_ids, bundle.raw_trace_length,
                                    "raw_family_mean"))
        pair_rows.extend(_pair_rows(cell_id, bundle.dataset_family, family_names,
                                    b3_family, bundle.group_ids, bundle.raw_trace_length,
                                    "b3_family_contribution"))

        feature_corr = _corr(X)
        feature_rank_corr = _corr(_rank_columns(X))
        within_values, between_values = [], []
        within_rank, between_rank = [], []
        feature_family = {}
        for family, indices in groups.items():
            for index in indices:
                feature_family[index] = family
        for left in range(X.shape[1]):
            for right in range(left + 1, X.shape[1]):
                same = feature_family[left] == feature_family[right]
                (within_values if same else between_values).append(abs(feature_corr[left, right]))
                (within_rank if same else between_rank).append(abs(feature_rank_corr[left, right]))

        for family_index, family in enumerate(family_names):
            indices = groups[family]
            peers = [j for j in indices]
            feature_r2 = []
            for target in indices:
                predictors = [j for j in peers if j != target]
                if predictors:
                    feature_r2.append(_oof_r2(X[:, target], X[:, predictors], bundle.group_ids))
            others = [j for j in range(len(family_names)) if j != family_index]
            family_rows.append({
                "cell_id": cell_id,
                "dataset_family": bundle.dataset_family,
                "family": family,
                "n_features": len(indices),
                "within_abs_pearson_mean": float(np.mean(np.abs(_corr(X[:, indices])[np.triu_indices(len(indices), 1)]))) if len(indices) > 1 else float("nan"),
                "within_effective_rank": _effective_rank(X[:, indices]),
                "within_feature_oof_r2_mean": float(np.nanmean(feature_r2)) if feature_r2 else float("nan"),
                "raw_family_from_other_families_oof_r2": _oof_r2(raw_family[:, family_index], raw_family[:, others], bundle.group_ids),
                "b3_family_from_other_families_oof_r2": _oof_r2(b3_family[:, family_index], b3_family[:, others], bundle.group_ids),
            })

        raw_corr = _corr(raw_family)
        b3_corr = _corr(b3_family)
        same_family_spearman = [
            float(_corr(_rank_columns(np.column_stack([raw_family[:, index], b3_family[:, index]])))[0, 1])
            for index in range(len(family_names))
        ]
        raw_length_corr = _corr(_residualize(raw_family, _length_basis(bundle.raw_trace_length)))
        b3_length_corr = _corr(_residualize(b3_family, _length_basis(bundle.raw_trace_length)))
        upper = np.triu_indices(len(family_names), 1)
        raw_quotient, _ = _quotient_entropy_duplicate(raw_family, family_names)
        b3_quotient, _ = _quotient_entropy_duplicate(b3_family, family_names)
        cell_record = {
            "cell_id": cell_id,
            "dataset_family": bundle.dataset_family,
            "n_rows": len(X),
            "n_features": X.shape[1],
            "n_families": len(family_names),
            "within_feature_abs_pearson_mean": float(np.mean(within_values)),
            "between_feature_abs_pearson_mean": float(np.mean(between_values)),
            "within_feature_abs_spearman_mean": float(np.mean(within_rank)),
            "between_feature_abs_spearman_mean": float(np.mean(between_rank)),
            "raw_family_abs_pearson_mean": float(np.mean(np.abs(raw_corr[upper]))),
            "b3_family_abs_pearson_mean": float(np.mean(np.abs(b3_corr[upper]))),
            "raw_to_b3_same_family_spearman_mean": float(np.mean(same_family_spearman)),
            "raw_family_abs_pearson_after_length": float(np.mean(np.abs(raw_length_corr[upper]))),
            "b3_family_abs_pearson_after_length": float(np.mean(np.abs(b3_length_corr[upper]))),
            "raw_family_effective_rank": _effective_rank(raw_family),
            "b3_family_effective_rank": _effective_rank(b3_family),
        }
        cell_record.update({f"raw_family_{key}": value for key, value in _spectral_summary(raw_family).items()})
        cell_record.update({f"b3_family_{key}": value for key, value in _spectral_summary(b3_family).items()})
        cell_record.update({f"raw_length_residual_{key}": value for key, value in _spectral_summary(_residualize(raw_family, _length_basis(bundle.raw_trace_length))).items()})
        cell_record.update({f"b3_length_residual_{key}": value for key, value in _spectral_summary(_residualize(b3_family, _length_basis(bundle.raw_trace_length))).items()})
        cell_record.update({f"raw_entropy_quotient_{key}": value for key, value in _spectral_summary(raw_quotient).items()})
        cell_record.update({f"b3_entropy_quotient_{key}": value for key, value in _spectral_summary(b3_quotient).items()})
        cell_record.update({f"raw_entropy_quotient_length_residual_{key}": value for key, value in _spectral_summary(_residualize(raw_quotient, _length_basis(bundle.raw_trace_length))).items()})
        cell_record.update({f"b3_entropy_quotient_length_residual_{key}": value for key, value in _spectral_summary(_residualize(b3_quotient, _length_basis(bundle.raw_trace_length))).items()})
        cell_rows.append(cell_record)
        state_path = out / "states" / f"{cell_id}.npz"
        digest = atomic_save_npz(
            state_path,
            X_risk=X,
            feature_names=np.asarray(bundle.feature_names),
            family_names=np.asarray(family_names),
            raw_family=raw_family,
            b3_family=b3_family,
            b3_logit=b3_logit,
            row_id=np.asarray(bundle.row_ids),
            group_id=np.asarray(bundle.group_ids),
            raw_trace_length=np.asarray(bundle.raw_trace_length),
        )
        artifacts.append({
            "cell_id": cell_id,
            "dataset_family": bundle.dataset_family,
            "bundle_sha256": bundle.bundle_sha256,
            "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
            "state": state_path.relative_to(out).as_posix(),
            "state_sha256": digest,
        })

    _write_csv(out / "LABEL_FREE_CELL.csv", cell_rows)
    _write_csv(out / "LABEL_FREE_FAMILY.csv", family_rows)
    _write_csv(out / "LABEL_FREE_PAIRS.csv", pair_rows)
    numeric_cell_keys = [key for key in cell_rows[0] if key not in {"cell_id", "dataset_family"}]
    label_free_summary = {
        "schema": SCHEMA,
        "phase": "label_free_summary",
        "labels_accessed": False,
        "equal_cell_means": {
            key: float(np.mean([float(row[key]) for row in cell_rows]))
            for key in numeric_cell_keys
        },
        "equal_dataset_family_means": {
            key: _equal_family_mean(cell_rows, key) for key in numeric_cell_keys
        },
        "pair_excluded_crossfit_summary": {},
        "raw_family_stability": _stability_summary(raw_stability),
        "b3_family_stability": _stability_summary(b3_stability),
        "caveat": "spectral summaries describe correlation geometry, not target identification",
    }
    for representation in ("raw_family_mean", "b3_family_contribution"):
        selected = [row for row in pair_rows if row["representation"] == representation]
        without_duplicate = [
            row for row in selected
            if {row["left"], row["right"]} != {"entropy_level", "topk_distribution"}
        ]
        label_free_summary["pair_excluded_crossfit_summary"][representation] = {
            "mean_abs_partial_pearson": float(np.mean(np.abs([
                row["crossfit_pair_excluded_partial_pearson"] for row in selected
            ]))),
            "median_abs_partial_pearson": float(np.median(np.abs([
                row["crossfit_pair_excluded_partial_pearson"] for row in selected
            ]))),
            "excluding_entropy_duplicate_mean_abs_partial_pearson": float(np.mean(np.abs([
                row["crossfit_pair_excluded_partial_pearson"] for row in without_duplicate
            ]))),
            "excluding_entropy_duplicate_median_abs_partial_pearson": float(np.median(np.abs([
                row["crossfit_pair_excluded_partial_pearson"] for row in without_duplicate
            ]))),
        }
        dataset_families = sorted({row["dataset_family"] for row in without_duplicate})
        pairs = sorted({(row["left"], row["right"]) for row in without_duplicate})
        fold_means = []
        supports = []
        for omitted in dataset_families:
            means = {
                pair: float(np.mean([
                    row["crossfit_pair_excluded_partial_pearson"]
                    for row in without_duplicate
                    if (row["left"], row["right"]) == pair
                    and row["dataset_family"] != omitted
                ]))
                for pair in pairs
            }
            fold_means.append(means)
            supports.append({pair for pair, value in means.items() if abs(value) > 0.10})
        jaccards = [
            len(left & right) / len(left | right) if left | right else 1.0
            for left, right in itertools.combinations(supports, 2)
        ]
        stable_edges = []
        for pair in pairs:
            values = np.asarray([means[pair] for means in fold_means])
            stable_edges.append({
                "left": pair[0],
                "right": pair[1],
                "mean_abs_leave_family_out_partial": float(np.mean(np.abs(values))),
                "sign_agreement": float(abs(np.mean(np.sign(values)))),
                "support_frequency_at_abs_0_10": int(sum(pair in support for support in supports)),
            })
        label_free_summary["pair_excluded_crossfit_summary"][representation].update({
            "leave_dataset_family_out_support_threshold": 0.10,
            "leave_dataset_family_out_support_sizes": [len(value) for value in supports],
            "leave_dataset_family_out_support_jaccard_mean": float(np.mean(jaccards)),
            "leave_dataset_family_out_support_jaccard_min": float(np.min(jaccards)),
            "leave_dataset_family_out_edges": sorted(
                stable_edges,
                key=lambda row: (-row["support_frequency_at_abs_0_10"],
                                 -row["mean_abs_leave_family_out_partial"]),
            ),
        })
    atomic_write_json(out / "LABEL_FREE_SUMMARY.json", label_free_summary)
    manifest = {
        "schema": SCHEMA,
        "phase": "label_free_fit",
        "labels_accessed": False,
        "registry_sha256": sha256_file(args.registry),
        "script_sha256": sha256_file(Path(__file__)),
        "artifacts": artifacts,
        "tables": {
            name: sha256_file(out / name)
            for name in ("LABEL_FREE_CELL.csv", "LABEL_FREE_FAMILY.csv", "LABEL_FREE_PAIRS.csv", "LABEL_FREE_SUMMARY.json")
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(out / "LABEL_FREE_FREEZE.json", manifest)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _equal_family_mean(rows: list[dict], key: str) -> float:
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_family.setdefault(row["dataset_family"], []).append(float(row[key]))
    return float(np.mean([np.mean(values) for values in by_family.values()]))


def evaluate(args: argparse.Namespace) -> None:
    out = args.out_dir
    freeze_path = out / "LABEL_FREE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = freeze.pop("content_sha256")
    if canonical_sha256(freeze) != expected or freeze.get("labels_accessed") is not False:
        raise ValueError("label-free freeze hash/contract mismatch")
    freeze["content_sha256"] = expected
    if freeze.get("script_sha256") != sha256_file(Path(__file__)):
        raise ValueError("diagnostic script drift")
    if freeze.get("registry_sha256") != sha256_file(args.registry):
        raise ValueError("registry drift")
    for relative, digest in freeze.get("tables", {}).items():
        if sha256_file(out / relative) != digest:
            raise ValueError(f"label-free table drift: {relative}")
    registry = load_registry(args.registry)
    by_cell = {row["cell_id"]: row for row in freeze["artifacts"]}
    if set(by_cell) != {row["cell_id"] for row in registry["cells"]}:
        raise ValueError("freeze does not cover exact registry")
    for cell_id, record in by_cell.items():
        state = out / record["state"]
        if sha256_file(state) != record["state_sha256"]:
            raise ValueError(f"state hash mismatch: {cell_id}")
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell_id}.npz")
        if bundle.bundle_sha256 != record["bundle_sha256"]:
            raise ValueError(f"bundle drift: {cell_id}")
        if canonical_sha256(list(bundle.row_ids)) != record["ordered_row_id_sha256"]:
            raise ValueError(f"row-order drift: {cell_id}")

    # Deliberately imported only after every target-free artifact is verified.
    from spectral_utils.residual_graph_deem_labels import join_labels_by_id, load_label_sidecar

    conditional_rows: list[dict] = []
    pair_rows: list[dict] = []
    sidecar_hashes: dict[str, str] = {}
    for registry_row in registry["cells"]:
        cell_id = registry_row["cell_id"]
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell_id}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell_id}.npz")
        sidecar_hashes[cell_id] = sidecar.sidecar_sha256
        y = join_labels_by_id(bundle, sidecar)
        with np.load(out / by_cell[cell_id]["state"], allow_pickle=False) as data:
            X = np.asarray(data["X_risk"], dtype=float)
            raw_family = np.asarray(data["raw_family"], dtype=float)
            b3_family = np.asarray(data["b3_family"], dtype=float)
            b3_logit = np.asarray(data["b3_logit"], dtype=float)
            raw_trace_length = np.asarray(data["raw_trace_length"], dtype=float)
            family_names = tuple(str(value) for value in data["family_names"].tolist())
            feature_names = tuple(str(value) for value in data["feature_names"].tolist())
        groups = family_index_map(feature_names)
        marginal = _corr(X)
        conditional = _corr(_class_residual(X, y))
        within_marginal, between_marginal = [], []
        within_conditional, between_conditional = [], []
        feature_family = {index: family for family, indices in groups.items() for index in indices}
        for left in range(X.shape[1]):
            for right in range(left + 1, X.shape[1]):
                same = feature_family[left] == feature_family[right]
                (within_marginal if same else between_marginal).append(abs(marginal[left, right]))
                (within_conditional if same else between_conditional).append(abs(conditional[left, right]))

        raw_marginal = _corr(raw_family)
        raw_conditional = _corr(_class_residual(raw_family, y))
        raw_conditional_length = _corr(_class_length_residual(raw_family, y, raw_trace_length))
        b3_marginal = _corr(b3_family)
        b3_conditional = _corr(_class_residual(b3_family, y))
        b3_conditional_length = _corr(_class_length_residual(b3_family, y, raw_trace_length))
        upper = np.triu_indices(len(family_names), 1)
        conditional_rows.append({
            "cell_id": cell_id,
            "dataset_family": bundle.dataset_family,
            "positive_rate": float(y.mean()),
            "within_feature_abs_corr_marginal": float(np.mean(within_marginal)),
            "within_feature_abs_corr_conditional_y": float(np.mean(within_conditional)),
            "between_feature_abs_corr_marginal": float(np.mean(between_marginal)),
            "between_feature_abs_corr_conditional_y": float(np.mean(between_conditional)),
            "raw_family_abs_corr_marginal": float(np.mean(np.abs(raw_marginal[upper]))),
            "raw_family_abs_corr_conditional_y": float(np.mean(np.abs(raw_conditional[upper]))),
            "raw_family_abs_corr_conditional_y_and_length": float(np.mean(np.abs(raw_conditional_length[upper]))),
            "b3_family_abs_corr_marginal": float(np.mean(np.abs(b3_marginal[upper]))),
            "b3_family_abs_corr_conditional_y": float(np.mean(np.abs(b3_conditional[upper]))),
            "b3_family_abs_corr_conditional_y_and_length": float(np.mean(np.abs(b3_conditional_length[upper]))),
            "b3_logit_class_separation_sd": float((b3_logit[y == 1].mean() - b3_logit[y == 0].mean()) / max(b3_logit.std(), 1e-12)),
        })
        for left in range(len(family_names)):
            for right in range(left + 1, len(family_names)):
                pair_rows.append({
                    "cell_id": cell_id,
                    "dataset_family": bundle.dataset_family,
                    "left": family_names[left],
                    "right": family_names[right],
                    "raw_marginal_corr": float(raw_marginal[left, right]),
                    "raw_conditional_y_corr": float(raw_conditional[left, right]),
                    "b3_marginal_corr": float(b3_marginal[left, right]),
                    "b3_conditional_y_corr": float(b3_conditional[left, right]),
                })
    _write_csv(out / "CONDITIONAL_CELL.csv", conditional_rows)
    _write_csv(out / "CONDITIONAL_PAIRS.csv", pair_rows)
    keys = [key for key in conditional_rows[0] if key not in {"cell_id", "dataset_family"}]
    summary = {
        "schema": SCHEMA,
        "phase": "conditional_evaluation",
        "labels_accessed": True,
        "label_free_freeze_sha256": sha256_file(freeze_path),
        "label_sidecar_sha256": sidecar_hashes,
        "equal_dataset_family_means": {key: _equal_family_mean(conditional_rows, key) for key in keys},
        "tables": {
            "CONDITIONAL_CELL.csv": sha256_file(out / "CONDITIONAL_CELL.csv"),
            "CONDITIONAL_PAIRS.csv": sha256_file(out / "CONDITIONAL_PAIRS.csv"),
        },
        "interpretation_boundary": "retrospective diagnostic; labels may not select a deployed model",
    }
    atomic_write_json(out / "CONDITIONAL_SUMMARY.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/bundles")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_frozen")
    parser.add_argument("--sidecar-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/label_sidecars")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/family_dependency_structure_v1")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "fit":
        fit(arguments)
    else:
        evaluate(arguments)
