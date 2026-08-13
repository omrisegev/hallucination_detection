"""Paired scorer repeatability utilities for automatic group-free IU Phase A4.

The module is deliberately target-blind.  Raw ProcessBench dictionaries cross
exactly one sanitizer; all fit/evaluation functions below it accept numeric
arrays and structural metadata only.  A repeated component is not interpreted
as correctness or hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .feature_utils import extract_all_features
from .group_free_research import resolve_local_lfs_object
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)


EPS = 1e-12
VIEWS = ("qwen3_4b", "qwen3_8b", "llama31_8b")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
FEATURE_ROSTER = (
    "epr", "trace_length", "spectral_entropy", "low_band_power",
    "high_band_power", "hl_ratio", "dominant_freq", "spectral_centroid",
    "stft_max_high_power", "stft_spectral_entropy", "rpdi", "sw_var_peak",
    "pe_mean", "hurst_exponent", "cusum_max", "cusum_shift_idx",
    "epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "epr_energy",
    "min_energy", "sw_var_peak_energy", "cusum_max_energy",
    "mean_top1_logprob", "logprob_margin", "mean_logprob_entropy",
    "varentropy", "renyi_entropy_2", "topk_tail_mass",
)
TELEMETRY_KEYS = (
    "token_entropies", "token_spilled_energies", "token_logsumexp",
    "top_k_logprobs",
)
SANITIZED_KEYS = frozenset({"id", "problem", "steps", *TELEMETRY_KEYS})
PROHIBITED_KEYS = frozenset({
    "label", "labels", "first_error", "error_label",
    "final_answer_correct", "step_token_spans", "align_diag",
})
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@dataclass(frozen=True)
class PairedDataset:
    raw: np.ndarray
    covariates: np.ndarray
    subset: np.ndarray
    item_id: np.ndarray
    group_id: np.ndarray
    feature_names: tuple[str, ...]
    input_rows: tuple[dict, ...]


@dataclass
class RawZTransformer:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, raw, names):
        raw = np.asarray(raw, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("raw-z input must be finite")
        signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names])
        oriented = raw * signs[None, :]
        scale = oriented.std(axis=0)
        scale[scale < EPS] = 1.0
        return cls(tuple(names), oriented.mean(axis=0), scale)

    def transform(self, raw):
        raw = np.asarray(raw, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("raw-z input must be finite")
        signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in self.names])
        return (raw * signs[None, :] - self.mean[None, :]) / self.std[None, :]


@dataclass
class TextDesign:
    continuous_mean: np.ndarray
    continuous_std: np.ndarray

    @classmethod
    def fit(cls, covariates):
        values = np.asarray(covariates, dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < EPS] = 1.0
        return cls(mean, std)

    def transform(self, covariates, subsets, view_codes):
        covariates = np.asarray(covariates, dtype=float)
        z = (covariates - self.continuous_mean[None, :]) / self.continuous_std[None, :]
        subset = np.asarray(subsets, dtype=int)
        onehot = np.column_stack([subset == index for index in range(len(SUBSETS))])
        return np.column_stack([
            z, z * z, onehot.astype(float), np.asarray(view_codes, dtype=float),
        ])


@dataclass
class PreparedFold:
    train_residuals: np.ndarray
    train_full_residuals: np.ndarray
    held_residuals: np.ndarray
    train_design: np.ndarray
    held_design: np.ndarray
    transformer: object
    feature_model: Ridge
    pair_permutation: np.ndarray | None


def _content_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _word_count(value) -> int:
    return len(re.findall(r"\w+", _content_text(value), flags=re.UNICODE))


def sanitize_processbench_row(cache_key, row: Mapping):
    """Return a target-free row or ``None`` for a known alignment failure."""
    diagnostics = row.get("align_diag", {})
    if not isinstance(diagnostics, Mapping):
        raise TypeError("align_diag must be a mapping")
    if diagnostics.get("problems"):
        return None
    missing = [key for key in TELEMETRY_KEYS if row.get(key) is None]
    if missing:
        raise KeyError(f"missing telemetry keys: {missing}")
    output = {
        "id": str(row.get("id", cache_key)),
        "problem": row.get("problem"),
        "steps": row.get("steps"),
    }
    output.update({key: row[key] for key in TELEMETRY_KEYS})
    validate_sanitized_row(output)
    return output


def validate_sanitized_row(row: Mapping) -> None:
    unexpected = set(row) - SANITIZED_KEYS
    if unexpected:
        raise ValueError(f"non-sanitized keys reached A4 boundary: {sorted(unexpected)}")
    if PROHIBITED_KEYS.intersection(row):
        raise ValueError("target or raw diagnostic key reached A4 boundary")


def extract_feature_row(row: Mapping) -> np.ndarray:
    validate_sanitized_row(row)
    values = extract_all_features(
        row["token_entropies"],
        spilled_energies=row["token_spilled_energies"],
        allow_short=True,
    ) or {}
    values.update(energy_features_from_logsumexp(row["token_logsumexp"]))
    values.update(logprob_features(row["top_k_logprobs"]))
    values.update(logprob_features_extended(row["top_k_logprobs"]))
    output = np.asarray([values.get(name, np.nan) for name in FEATURE_ROSTER], dtype=float)
    if not np.isfinite(output).all():
        missing = [name for name, value in zip(FEATURE_ROSTER, output) if not np.isfinite(value)]
        raise ValueError(f"non-finite primary A4 feature(s): {missing}")
    return output


def row_covariates(row: Mapping) -> np.ndarray:
    validate_sanitized_row(row)
    response = row["steps"]
    problem = row["problem"]
    steps = response if isinstance(response, Sequence) and not isinstance(response, str) else [response]
    counts = (
        len(_content_text(response)), _word_count(response), len(steps),
        len(_content_text(problem)), _word_count(problem), len(row["token_entropies"]),
    )
    return np.log1p(np.asarray(counts, dtype=float))


def content_hash(row: Mapping) -> str:
    validate_sanitized_row(row)
    payload = json.dumps(
        {"problem": row["problem"], "steps": row["steps"]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_view_subset(path: Path, repo_root: Path):
    resolved, storage = resolve_local_lfs_object(path, repo_root)
    with resolved.open("rb") as handle:
        cache = pickle.load(handle)
    output = {}
    for key in sorted(cache):
        safe = sanitize_processbench_row(key, cache[key])
        if safe is not None:
            output[safe["id"]] = {
                "content_hash": content_hash(safe),
                "feature_row": extract_feature_row(safe),
                "covariates": row_covariates(safe),
            }
    del cache
    return output, storage


def load_paired_dataset(cache_root: str | Path, repo_root: str | Path) -> PairedDataset:
    cache_root, repo_root = Path(cache_root), Path(repo_root)
    matrices, covariates, subsets, item_ids, groups = [], [], [], [], []
    input_rows = []
    for subset_index, subset in enumerate(SUBSETS):
        by_view = {}
        for view in VIEWS:
            path = cache_root / f"pb_{view}" / f"processbench_{subset}.pkl"
            rows, storage = _load_view_subset(path, repo_root)
            by_view[view] = rows
            input_rows.append({
                "view": view, "subset": subset, "path": str(path.relative_to(repo_root)),
                "sha256": storage["sha256"], "size": int(storage["size"]),
                "row_count": len(rows), "labels_accessed": False,
            })
        ids = sorted(set.intersection(*(set(by_view[view]) for view in VIEWS)))
        if len(ids) != len(by_view[VIEWS[0]]) or any(len(rows) != len(ids) for rows in by_view.values()):
            raise RuntimeError(f"inexact A4 pairing for {subset}")
        for item_id in ids:
            safe = [by_view[view][item_id] for view in VIEWS]
            hashes = [row["content_hash"] for row in safe]
            if len(set(hashes)) != 1:
                raise RuntimeError(f"content mismatch for {subset}/{item_id}")
            matrices.append(np.stack([row["feature_row"] for row in safe]))
            covariates.append(np.stack([row["covariates"] for row in safe]))
            subsets.append(subset_index)
            item_ids.append(f"{subset}::{item_id}")
            groups.append(hashes[0])
        del by_view
    raw = np.stack(matrices)
    cov = np.stack(covariates)
    if raw.shape != (3400, 3, len(FEATURE_ROSTER)) or not np.isfinite(raw).all():
        raise RuntimeError(f"A4 paired tensor violates frozen shape: {raw.shape}")
    return PairedDataset(
        raw=raw, covariates=cov, subset=np.asarray(subsets, dtype=int),
        item_id=np.asarray(item_ids), group_id=np.asarray(groups),
        feature_names=FEATURE_ROSTER, input_rows=tuple(input_rows),
    )


def grouped_splits(subsets, groups, *, n_splits, seed):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    dummy = np.zeros(len(subsets))
    return list(splitter.split(dummy, np.asarray(subsets), np.asarray(groups)))


def _fit_transformer(raw_train, names, kind):
    pooled = np.vstack([raw_train[:, 0], raw_train[:, 1]])
    if not np.isfinite(pooled).all():
        raise ValueError("median imputation is prohibited in A4")
    if kind == "mixed_v2":
        return FixedMixedV2Transformer.fit(pooled, names)
    if kind == "raw_z":
        return RawZTransformer.fit(pooled, names)
    raise ValueError(f"unknown transform kind: {kind}")


def fit_length_edges(subsets, response_word_log):
    subsets = np.asarray(subsets, dtype=int)
    lengths = np.asarray(response_word_log, dtype=float)
    return {
        int(subset): np.unique(np.quantile(
            lengths[subsets == subset], np.linspace(0.0, 1.0, 11),
        ))
        for subset in sorted(set(subsets))
    }


def conditional_derangement(
    subsets, response_word_log, rng, *, edges_by_subset=None, forbidden_groups=None,
):
    subsets = np.asarray(subsets, dtype=int)
    lengths = np.asarray(response_word_log, dtype=float)
    edges_by_subset = (
        fit_length_edges(subsets, lengths) if edges_by_subset is None else edges_by_subset
    )
    forbidden_groups = (
        np.arange(len(subsets)) if forbidden_groups is None
        else np.asarray(forbidden_groups)
    )
    permutation = np.arange(len(subsets))
    diagnostics = []
    for subset in sorted(set(subsets)):
        indices = np.flatnonzero(subsets == subset)
        edges = np.asarray(edges_by_subset[int(subset)], dtype=float)
        bins = np.digitize(lengths[indices], edges[1:-1], right=True)
        for bin_id in sorted(set(bins)):
            members = indices[bins == bin_id]
            if len(members) < 2:
                raise ValueError(f"conditional shuffle singleton: subset={subset}, bin={bin_id}")
            for _ in range(1000):
                order = rng.permutation(members)
                proposed = np.roll(order, 1)
                if np.all(forbidden_groups[order] != forbidden_groups[proposed]):
                    permutation[order] = proposed
                    break
            else:
                raise ValueError(
                    f"cannot derange groups: subset={subset}, bin={bin_id}"
                )
            diagnostics.append({"subset": int(subset), "bin": int(bin_id), "n": len(members)})
    if np.any(forbidden_groups[permutation] == forbidden_groups):
        raise RuntimeError("conditional shuffle contains a fixed point")
    return permutation, diagnostics


def _stack_qwen_design(design, covariates, subsets):
    n = len(subsets)
    cov = np.vstack([covariates[:, 0], covariates[:, 1]])
    sub = np.tile(subsets, 2)
    code = np.concatenate([-np.ones(n), np.ones(n)])
    return design.transform(cov, sub, code)


def _view_design(design, covariates, subsets):
    output = []
    for view, code in enumerate((-1.0, 1.0, 0.0)):
        output.append(design.transform(covariates[:, view], subsets, np.full(len(subsets), code)))
    return np.stack(output, axis=1)


def prepare_fold(
    raw_train, cov_train, subset_train, group_train,
    raw_held, cov_held, subset_held, names,
    *, transform_kind="mixed_v2", split_seed=20260813,
):
    raw_train = np.asarray(raw_train).copy()
    cov_train = np.asarray(cov_train).copy()
    transformer = _fit_transformer(raw_train, names, transform_kind)
    transformed_train = np.stack([
        transformer.transform(raw_train[:, view]) for view in range(2)
    ], axis=1)
    transformed_held = np.stack([
        transformer.transform(raw_held[:, view]) for view in range(3)
    ], axis=1)

    residual_train = np.empty_like(transformed_train)
    nuisance_splits = grouped_splits(
        subset_train, group_train, n_splits=5, seed=int(split_seed) + 17,
    )
    for nuisance_train, nuisance_valid in nuisance_splits:
        design = TextDesign.fit(np.vstack([
            cov_train[nuisance_train, 0], cov_train[nuisance_train, 1],
        ]))
        x_train = _stack_qwen_design(
            design, cov_train[nuisance_train], subset_train[nuisance_train],
        )
        y_train = np.vstack([
            transformed_train[nuisance_train, 0], transformed_train[nuisance_train, 1],
        ])
        model = Ridge(alpha=1.0).fit(x_train, y_train)
        valid_design = _view_design(
            design, cov_train[nuisance_valid], subset_train[nuisance_valid],
        )
        for view in range(2):
            residual_train[nuisance_valid, view] = (
                transformed_train[nuisance_valid, view]
                - model.predict(valid_design[:, view])
            )

    design = TextDesign.fit(np.vstack([cov_train[:, 0], cov_train[:, 1]]))
    train_design = _stack_qwen_design(design, cov_train, subset_train)
    train_y = np.vstack([transformed_train[:, 0], transformed_train[:, 1]])
    feature_model = Ridge(alpha=1.0).fit(train_design, train_y)
    train_view_design = _view_design(design, cov_train, subset_train)
    full_residual_train = np.empty_like(transformed_train)
    for view in range(2):
        full_residual_train[:, view] = (
            transformed_train[:, view]
            - feature_model.predict(train_view_design[:, view])
        )
    held_design = _view_design(design, cov_held, subset_held)
    residual_held = np.empty_like(transformed_held)
    for view in range(3):
        residual_held[:, view] = (
            transformed_held[:, view] - feature_model.predict(held_design[:, view])
        )
    return PreparedFold(
        residual_train, full_residual_train, residual_held,
        train_design, held_design, transformer, feature_model, None,
    )


def _center_pair(x4, x8):
    return x4 - x4.mean(axis=0), x8 - x8.mean(axis=0)


def algebraic_sign(vector):
    vector = np.asarray(vector, dtype=float).copy()
    maximum = np.max(np.abs(vector))
    candidates = np.flatnonzero(np.isclose(np.abs(vector), maximum, rtol=0.0, atol=1e-14))
    index = int(candidates[0])
    if vector[index] < 0:
        vector *= -1
    return vector


def fit_corrca(x4, x8, ridge_fraction):
    x4, x8 = _center_pair(np.asarray(x4), np.asarray(x8))
    n, p = x4.shape
    within = (x4.T @ x4 + x8.T @ x8) / (2.0 * (n - 1))
    between = (x4.T @ x8 + x8.T @ x4) / (2.0 * (n - 1))
    regularized = within + float(ridge_fraction) * np.trace(within) / p * np.eye(p)
    values, vectors = eigh(between, regularized)
    vector = vectors[:, int(np.argmax(values))]
    vector /= np.linalg.norm(vector)
    return algebraic_sign(vector), float(np.max(values))


def _symmetric_inverse_sqrt(matrix):
    values, vectors = eigh(0.5 * (matrix + matrix.T))
    if values[0] <= 0:
        raise ValueError("CCA regularized covariance is not positive definite")
    return (vectors * (values ** -0.5)[None, :]) @ vectors.T


def fit_cca_common(x4, x8, ridge_fraction):
    x4, x8 = _center_pair(np.asarray(x4), np.asarray(x8))
    n, p = x4.shape
    c44, c88, c48 = x4.T @ x4 / (n - 1), x8.T @ x8 / (n - 1), x4.T @ x8 / (n - 1)
    p4 = _symmetric_inverse_sqrt(c44 + float(ridge_fraction) * np.trace(c44) / p * np.eye(p))
    p8 = _symmetric_inverse_sqrt(c88 + float(ridge_fraction) * np.trace(c88) / p * np.eye(p))
    u, _, vt = np.linalg.svd(p4 @ c48 @ p8, full_matrices=False)
    a4, a8 = p4 @ u[:, 0], p8 @ vt.T[:, 0]
    a4, a8 = a4 / np.linalg.norm(a4), a8 / np.linalg.norm(a8)
    if float(a4 @ a8) < 0:
        a8 *= -1
    common = a4 + a8
    if np.linalg.norm(common) < EPS:
        raise ValueError("CCA common loading has zero norm")
    return algebraic_sign(common / np.linalg.norm(common))


def fit_baseline_loading(method, x4, x8, *, ridge=None, feature_index=None):
    x4, x8 = _center_pair(np.asarray(x4), np.asarray(x8))
    p = x4.shape[1]
    if method == "cca":
        return fit_cca_common(x4, x8, ridge)
    if method == "diagonal":
        numerator = 2.0 * np.sum(x4 * x8, axis=0) / (len(x4) - 1)
        denominator = (
            np.sum(x4 * x4, axis=0) + np.sum(x8 * x8, axis=0)
        ) / (len(x4) - 1)
        weights = np.maximum(numerator / np.maximum(denominator, EPS), 0.0)
        if np.linalg.norm(weights) < EPS:
            raise ValueError("diagonal reliability weights are all zero")
        return weights / np.linalg.norm(weights)
    if method == "single":
        output = np.zeros(p)
        output[int(feature_index)] = 1.0
        return output
    if method == "pca":
        stacked = np.vstack([x4, x8])
        values, vectors = eigh(stacked.T @ stacked / (len(stacked) - 1))
        return algebraic_sign(vectors[:, int(np.argmax(values))])
    if method == "equal":
        return np.ones(p) / np.sqrt(p)
    raise ValueError(f"unknown baseline: {method}")


def safe_correlation(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(left) < 4 or np.std(left) < EPS or np.std(right) < EPS:
        raise ValueError("correlation cell is too small or constant")
    return float(np.corrcoef(left, right)[0, 1])


def fisher_macro(correlations):
    values = np.asarray(correlations, dtype=float)
    return float(np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))


def correlation_cells(scores_left, scores_right, folds, subsets):
    rows = []
    for fold in sorted(set(folds)):
        for subset in range(len(SUBSETS)):
            mask = (folds == fold) & (subsets == subset)
            rows.append({
                "fold": int(fold), "subset": SUBSETS[subset], "n": int(mask.sum()),
                "correlation": safe_correlation(scores_left[mask], scores_right[mask]),
            })
    return rows


def correlation_summary(scores, folds, subsets):
    qwen = correlation_cells(scores[:, 0], scores[:, 1], folds, subsets)
    llama = correlation_cells(scores[:, 2], scores[:, :2].mean(axis=1), folds, subsets)
    by_subset = {}
    for subset in SUBSETS:
        by_subset[subset] = {
            "qwen": fisher_macro([row["correlation"] for row in qwen if row["subset"] == subset]),
            "llama": fisher_macro([row["correlation"] for row in llama if row["subset"] == subset]),
        }
    return {
        "qwen_cells": qwen, "llama_cells": llama,
        "qwen_macro": fisher_macro([row["correlation"] for row in qwen]),
        "llama_macro": fisher_macro([row["correlation"] for row in llama]),
        "by_subset": by_subset,
    }


def candidate_key(method, value=None):
    if value is None:
        return method
    return f"{method}:{value}"


def loading_from_key(key, x4, x8):
    method, _, value = key.partition(":")
    if method == "corrca":
        return fit_corrca(x4, x8, float(value))[0]
    if method == "cca":
        return fit_cca_common(x4, x8, float(value))
    if method == "single":
        return fit_baseline_loading(method, x4, x8, feature_index=int(value))
    return fit_baseline_loading(method, x4, x8)


def select_nested(
    raw, covariates, subsets, groups, names, *, transform_kind,
    seed, pair_null_seed=None, baselines=True,
):
    raw = np.asarray(raw).copy()
    covariates = np.asarray(covariates).copy()
    splits = grouped_splits(subsets, groups, n_splits=4, seed=int(seed) + 31)
    prepared = []
    null_permutations = []
    for fold, (train, valid) in enumerate(splits):
        raw_train, raw_valid = raw[train].copy(), raw[valid].copy()
        cov_train, cov_valid = covariates[train].copy(), covariates[valid].copy()
        if pair_null_seed is not None:
            nested_edges = fit_length_edges(
                subsets[train], cov_train[:, 0, 1],
            )
            train_permutation, _ = conditional_derangement(
                subsets[train], cov_train[:, 0, 1],
                np.random.default_rng(int(pair_null_seed) + 1009 * (fold + 1)),
                edges_by_subset=nested_edges,
                forbidden_groups=np.asarray(groups)[train],
            )
            valid_permutation, _ = conditional_derangement(
                subsets[valid], cov_valid[:, 0, 1],
                np.random.default_rng(int(pair_null_seed) + 2003 * (fold + 1)),
                edges_by_subset=nested_edges,
                forbidden_groups=np.asarray(groups)[valid],
            )
            raw_train[:, 1], cov_train[:, 1] = (
                raw_train[train_permutation, 1], cov_train[train_permutation, 1]
            )
            raw_valid[:, 1], cov_valid[:, 1] = (
                raw_valid[valid_permutation, 1], cov_valid[valid_permutation, 1]
            )
            null_permutations.append({
                "fold": fold, "train": train_permutation, "valid": valid_permutation,
                "length_edges": nested_edges,
            })
        prepared.append((valid, prepare_fold(
            raw_train, cov_train, subsets[train], np.asarray(groups)[train],
            raw_valid, cov_valid, subsets[valid], names,
            transform_kind=transform_kind,
            split_seed=int(seed) + 101 * fold,
        )))
    corr_candidates = [candidate_key("corrca", ridge) for ridge in RIDGE_GRID]
    baseline_candidates = [candidate_key("cca", ridge) for ridge in RIDGE_GRID]
    baseline_candidates += ["diagonal", "pca", "equal"]
    baseline_candidates += [candidate_key("single", index) for index in range(len(names))]

    def score_key(key):
        correlations = []
        for valid, fold_data in prepared:
            loading = loading_from_key(
                key, fold_data.train_residuals[:, 0], fold_data.train_residuals[:, 1],
            )
            scores = fold_data.held_residuals @ loading
            for subset in sorted(set(np.asarray(subsets, dtype=int))):
                mask = subsets[valid] == subset
                correlations.append(safe_correlation(scores[mask, 0], scores[mask, 1]))
        return fisher_macro(correlations)

    corr_scores = {key: score_key(key) for key in corr_candidates}
    # The ascending key pass plus >= implements the registered larger-ridge tie.
    corr_selected = max(corr_candidates, key=lambda key: (corr_scores[key], float(key.split(":")[1])))
    result = {
        "corrca_selected": corr_selected,
        "corrca_scores": corr_scores,
        "pair_null_permutations": null_permutations,
    }
    if baselines:
        baseline_scores = {key: score_key(key) for key in baseline_candidates}
        baseline_selected = max(
            baseline_candidates, key=lambda key: (baseline_scores[key], key)
        )
        result.update({
            "baseline_selected": baseline_selected,
            "baseline_scores": baseline_scores,
        })
    return result


def scalar_confound_metrics(prepared, loading, subset_held):
    # Both train and held targets live in the residual coordinates produced by
    # the same full outer-training nuisance model. CorrCA itself remains fitted
    # on cross-fitted train residuals.
    train_scores = prepared.train_full_residuals @ loading
    model = Ridge(alpha=1.0).fit(prepared.train_design, np.concatenate([
        train_scores[:, 0], train_scores[:, 1],
    ]))
    held_scores = prepared.held_residuals @ loading
    rows = []
    for view in range(3):
        predicted = model.predict(prepared.held_design[:, view])
        actual = held_scores[:, view]
        for subset in range(len(SUBSETS)):
            mask = subset_held == subset
            y, p = actual[mask], predicted[mask]
            r2 = 1.0 - float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
            rows.append({
                "view": VIEWS[view], "subset": SUBSETS[subset], "n": int(mask.sum()),
                "r2": r2, "abs_correlation": abs(safe_correlation(y, p)),
            })
    return rows


__all__ = [
    "FEATURE_ROSTER", "PairedDataset", "PreparedFold", "RIDGE_GRID", "SUBSETS", "VIEWS",
    "algebraic_sign", "conditional_derangement", "content_hash", "correlation_summary",
    "extract_feature_row", "fisher_macro", "fit_baseline_loading", "fit_cca_common",
    "fit_corrca", "fit_length_edges", "grouped_splits", "load_paired_dataset", "loading_from_key",
    "prepare_fold", "row_covariates", "safe_correlation", "sanitize_processbench_row",
    "scalar_confound_metrics", "select_nested", "validate_sanitized_row",
]
