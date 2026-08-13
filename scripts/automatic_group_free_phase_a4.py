#!/usr/bin/env python3
"""Prepare and execute the frozen automatic group-free IU Phase A4 protocol.

``prepare`` sanitizes the exact ProcessBench triples, freezes their numeric
tensor, item-first folds, input/source hashes, and the no-label boundary.
``run`` refuses to continue unless that boundary still verifies, then executes
the shared-repeatability premise test.  Neither command reads correctness or
step-error targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.paired_repeatability import (  # noqa: E402
    FEATURE_ROSTER,
    RIDGE_GRID,
    SUBSETS,
    VIEWS,
    conditional_derangement,
    correlation_summary,
    fisher_macro,
    fit_length_edges,
    grouped_splits,
    load_paired_dataset,
    loading_from_key,
    prepare_fold,
    safe_correlation,
    scalar_confound_metrics,
    select_nested,
)


VERSION = "automatic-group-free-iu-a4-v1-2026-08-13"
DEFAULT_CACHE = REPO / "dataset_cache" / "repgrid"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a4_v1"
PROTOCOL = REPO / "docs" / "experiments" / "AUTOMATIC_GROUP_FREE_IU_PHASE_A4_V1.md"
SEED = 20260813
BOOTSTRAP_DRAWS = 5000
TRAIN_NULL_DRAWS = 200
HELD_NULL_DRAWS = 1000
SOURCE_FILES = (
    "scripts/automatic_group_free_phase_a4.py",
    "scripts/test_paired_repeatability.py",
    "spectral_utils/paired_repeatability.py",
    "spectral_utils/repeated_measurement_reliability.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/repgrid_scoring.py",
    "spectral_utils/group_free_research.py",
)
BOUNDARY_PRIORS = (
    "results/automatic_group_free_phase_a0_v1/processbench_cross_model_pairing.json",
    "results/leverage_balanced_processbench_transfer_v1/FIT_MANIFEST.json",
)
BASELINE_TIE_ORDER = {
    "equal": 5, "pca": 4, "diagonal": 3, "cca": 2, "single": 1,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    return {path: sha256_file(REPO / path) for path in SOURCE_FILES}


def prior_hashes():
    return {path: sha256_file(REPO / path) for path in BOUNDARY_PRIORS}


def _fold_assignment(subsets, groups):
    assignment = np.full(len(subsets), -1, dtype=int)
    rows = []
    splits = grouped_splits(subsets, groups, n_splits=5, seed=SEED)
    for fold, (train, held) in enumerate(splits):
        if np.any(assignment[held] >= 0):
            raise RuntimeError("outer held folds overlap")
        assignment[held] = fold
        train_groups, held_groups = set(groups[train]), set(groups[held])
        if train_groups.intersection(held_groups):
            raise RuntimeError("content group crossed an outer fold")
        for subset in range(len(SUBSETS)):
            rows.append({
                "fold": fold,
                "subset": SUBSETS[subset],
                "train_items": int(np.sum(np.asarray(subsets)[train] == subset)),
                "held_items": int(np.sum(np.asarray(subsets)[held] == subset)),
                "group_overlap": 0,
            })
    if np.any(assignment < 0):
        raise RuntimeError("outer folds do not cover every item")
    return assignment, rows


def prepare(cache_root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_paired_dataset(cache_root, REPO)
    folds, fold_rows = _fold_assignment(data.subset, data.group_id)
    tensor_path = out / "paired_no_label_tensor.npz"
    np.savez_compressed(
        tensor_path,
        raw=data.raw,
        covariates=data.covariates,
        subset=data.subset,
        item_id=data.item_id.astype(str),
        group_id=data.group_id.astype(str),
        outer_fold=folds,
        feature_names=np.asarray(data.feature_names),
        view_names=np.asarray(VIEWS),
        subset_names=np.asarray(SUBSETS),
    )
    fold_path = out / "outer_folds.csv"
    write_csv(fold_path, fold_rows)
    boundary = {
        "version": VERSION,
        "status": "FROZEN_BEFORE_HELD_LLAMA_STRUCTURAL_EVALUATION",
        "protocol": str(PROTOCOL.relative_to(REPO)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": source_hashes(),
        "prior_artifact_sha256": prior_hashes(),
        "input_rows": data.input_rows,
        "tensor": {
            "path": str(tensor_path.relative_to(REPO)),
            "sha256": sha256_file(tensor_path),
            "shape": list(data.raw.shape),
            "labels_accessed": False,
        },
        "fold_manifest": {
            "path": str(fold_path.relative_to(REPO)),
            "sha256": sha256_file(fold_path),
            "splitter": "StratifiedGroupKFold",
            "n_splits": 5,
            "seed": SEED,
            "groups": "sha256(problem,steps)",
        },
        "feature_roster": list(FEATURE_ROSTER),
        "view_names": list(VIEWS),
        "subset_names": list(SUBSETS),
        "ridge_grid": list(RIDGE_GRID),
        "baseline_tie_order": BASELINE_TIE_ORDER,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "train_null_draws": TRAIN_NULL_DRAWS,
        "held_null_draws": HELD_NULL_DRAWS,
        "detector_verdict_frozen": "CLOSE_NO_TARGET_CONTRAST",
        "correctness_or_step_targets_accessed": False,
    }
    write_json(out / "A4_BOUNDARY.json", boundary)
    (out / "BOUNDARY_REPORT.md").write_text(
        f"""# Automatic group-free IU — A4 frozen boundary

- Version: `{VERSION}`
- Exact item triples: **{len(data.raw)}**
- Views: **{', '.join(VIEWS)}**
- Features: **{len(FEATURE_ROSTER)}** (exact frozen intersection)
- Outer folds: **5**, grouped by exact content hash
- Correctness or step targets accessed: **no**
- Detector verdict frozen: **`CLOSE_NO_TARGET_CONTRAST`**

The no-label numeric tensor, exact folds, inputs, transitive source hashes,
ridge grid, null counts, and protocol are frozen before held-Llama structural
evaluation. Run `automatic_group_free_phase_a4.py run` only after this boundary
has been committed.
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": boundary["status"], "tensor_sha256": boundary["tensor"]["sha256"],
        "n_items": len(data.raw), "n_features": len(FEATURE_ROSTER),
    }, indent=2))


def load_and_verify_boundary(out):
    out = Path(out)
    boundary = json.load((out / "A4_BOUNDARY.json").open())
    if boundary["version"] != VERSION:
        raise RuntimeError("A4 boundary version mismatch")
    observed_sources = source_hashes()
    if observed_sources != boundary["source_sha256"]:
        raise RuntimeError("A4 source changed after boundary freeze")
    if prior_hashes() != boundary["prior_artifact_sha256"]:
        raise RuntimeError("A4 prior artifact changed after boundary freeze")
    if sha256_file(PROTOCOL) != boundary["protocol_sha256"]:
        raise RuntimeError("A4 protocol changed after boundary freeze")
    tensor_path = REPO / boundary["tensor"]["path"]
    if sha256_file(tensor_path) != boundary["tensor"]["sha256"]:
        raise RuntimeError("A4 no-label tensor changed after boundary freeze")
    with np.load(tensor_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_keys = {
        "raw", "covariates", "subset", "item_id", "group_id", "outer_fold",
        "feature_names", "view_names", "subset_names",
    }
    if set(arrays) != expected_keys:
        raise RuntimeError("A4 tensor keys violate the frozen schema")
    fold_path = REPO / boundary["fold_manifest"]["path"]
    if sha256_file(fold_path) != boundary["fold_manifest"]["sha256"]:
        raise RuntimeError("A4 fold manifest changed after boundary freeze")
    if boundary["feature_roster"] != list(FEATURE_ROSTER):
        raise RuntimeError("A4 boundary feature roster mismatch")
    if boundary["view_names"] != list(VIEWS) or boundary["subset_names"] != list(SUBSETS):
        raise RuntimeError("A4 boundary view/subset roster mismatch")
    if boundary["ridge_grid"] != list(RIDGE_GRID):
        raise RuntimeError("A4 boundary ridge grid mismatch")
    if boundary["baseline_tie_order"] != BASELINE_TIE_ORDER:
        raise RuntimeError("A4 boundary baseline tie rule mismatch")
    if (
        boundary["bootstrap_draws"] != BOOTSTRAP_DRAWS
        or boundary["train_null_draws"] != TRAIN_NULL_DRAWS
        or boundary["held_null_draws"] != HELD_NULL_DRAWS
        or boundary["detector_verdict_frozen"] != "CLOSE_NO_TARGET_CONTRAST"
    ):
        raise RuntimeError("A4 boundary execution constants mismatch")
    if tuple(arrays["feature_names"].tolist()) != FEATURE_ROSTER:
        raise RuntimeError("A4 feature roster mismatch")
    if tuple(arrays["view_names"].tolist()) != VIEWS:
        raise RuntimeError("A4 view roster mismatch")
    if arrays["raw"].shape != (3400, 3, len(FEATURE_ROSTER)):
        raise RuntimeError("A4 tensor shape mismatch")
    if not np.isfinite(arrays["raw"]).all() or not np.isfinite(arrays["covariates"]).all():
        raise RuntimeError("A4 tensor contains non-finite values")
    if arrays["covariates"].shape != (3400, 3, 6):
        raise RuntimeError("A4 covariate shape mismatch")
    for name in ("subset", "item_id", "group_id", "outer_fold"):
        if arrays[name].shape != (3400,):
            raise RuntimeError(f"A4 {name} shape mismatch")
    if set(arrays["subset"].tolist()) != set(range(len(SUBSETS))):
        raise RuntimeError("A4 subset indices mismatch")
    if set(arrays["outer_fold"].tolist()) != set(range(5)):
        raise RuntimeError("A4 outer fold indices mismatch")
    if len(set(arrays["item_id"].tolist())) != 3400:
        raise RuntimeError("A4 item IDs are not unique")
    recomputed_folds, _ = _fold_assignment(arrays["subset"], arrays["group_id"])
    if not np.array_equal(recomputed_folds, arrays["outer_fold"]):
        raise RuntimeError("A4 semantic fold assignment mismatch")
    return boundary, arrays


def _baseline_category_selections(scores):
    by_category = {}
    for key, value in scores.items():
        category = key.split(":", 1)[0]
        by_category.setdefault(category, []).append((key, value))

    def tie_value(key):
        category, _, suffix = key.partition(":")
        if category == "cca":
            return float(suffix)
        if category == "single":
            return -int(suffix)
        return 0.0

    selected = {
        category: max(rows, key=lambda row: (row[1], tie_value(row[0])))[0]
        for category, rows in by_category.items()
    }
    strongest = max(
        selected.values(),
        key=lambda key: (
            scores[key], BASELINE_TIE_ORDER[key.split(":", 1)[0]], tie_value(key),
        ),
    )
    return selected, strongest


def _outer_null_pairing(
    raw_train, cov_train, subset_train, group_train,
    raw_held, cov_held, subset_held, group_held, *, seed,
):
    """Break both train and held Qwen8 pairs using train-fitted strata."""
    raw_train, cov_train = np.asarray(raw_train).copy(), np.asarray(cov_train).copy()
    raw_held, cov_held = np.asarray(raw_held).copy(), np.asarray(cov_held).copy()
    edges = fit_length_edges(subset_train, cov_train[:, 0, 1])
    train_permutation, train_strata = conditional_derangement(
        subset_train, cov_train[:, 0, 1], np.random.default_rng(int(seed) + 3001),
        edges_by_subset=edges, forbidden_groups=group_train,
    )
    held_permutation, held_strata = conditional_derangement(
        subset_held, cov_held[:, 0, 1], np.random.default_rng(int(seed) + 4001),
        edges_by_subset=edges, forbidden_groups=group_held,
    )
    raw_train[:, 1], cov_train[:, 1] = (
        raw_train[train_permutation, 1], cov_train[train_permutation, 1]
    )
    raw_held[:, 1], cov_held[:, 1] = (
        raw_held[held_permutation, 1], cov_held[held_permutation, 1]
    )
    return raw_train, cov_train, raw_held, cov_held, {
        "length_edges": edges,
        "train_permutation": train_permutation,
        "held_permutation": held_permutation,
        "train_strata": train_strata,
        "held_strata": held_strata,
    }


def _outer_fit(arrays, *, transform_kind="mixed_v2", pair_null_seed=None, baselines=True):
    raw, cov = arrays["raw"], arrays["covariates"]
    subsets, groups = arrays["subset"], arrays["group_id"]
    folds = arrays["outer_fold"]
    candidate_scores = np.full((len(raw), 3), np.nan)
    strongest_scores = np.full((len(raw), 3), np.nan)
    baseline_scores = {
        category: np.full((len(raw), 3), np.nan)
        for category in ("cca", "diagonal", "single", "pca", "equal")
    } if baselines else {}
    selections, loadings, confounds = [], [], []
    for fold in range(5):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        null_seed = None if pair_null_seed is None else int(pair_null_seed) + 100003 * fold
        raw_train, raw_held = raw[train].copy(), raw[held].copy()
        cov_train, cov_held = cov[train].copy(), cov[held].copy()
        nested = select_nested(
            raw_train, cov_train, subsets[train], groups[train], FEATURE_ROSTER,
            transform_kind=transform_kind, seed=SEED + 1000 * fold,
            pair_null_seed=null_seed, baselines=baselines,
        )
        outer_null_diagnostics = None
        if pair_null_seed is not None:
            raw_train, cov_train, raw_held, cov_held, outer_null_diagnostics = (
                _outer_null_pairing(
                    raw_train, cov_train, subsets[train], groups[train],
                    raw_held, cov_held, subsets[held], groups[held], seed=null_seed,
                )
            )
        prepared = prepare_fold(
            raw_train, cov_train, subsets[train], groups[train],
            raw_held, cov_held, subsets[held], FEATURE_ROSTER,
            transform_kind=transform_kind,
            split_seed=SEED + 3000 * fold,
        )
        corr_loading = loading_from_key(
            nested["corrca_selected"],
            prepared.train_residuals[:, 0], prepared.train_residuals[:, 1],
        )
        candidate_scores[held] = prepared.held_residuals @ corr_loading
        loadings.append(corr_loading)
        row = {
            "fold": fold,
            "corrca_selected": nested["corrca_selected"],
            "corrca_nested_score": nested["corrca_scores"][nested["corrca_selected"]],
            "outer_null_diagnostics": outer_null_diagnostics,
            "nested_null_diagnostics": nested["pair_null_permutations"],
        }
        if pair_null_seed is None:
            for metric in scalar_confound_metrics(prepared, corr_loading, subsets[held]):
                confounds.append({"fold": fold, **metric})
        if baselines:
            category_selected, strongest = _baseline_category_selections(
                nested["baseline_scores"]
            )
            row.update({
                "strongest_baseline": strongest,
                "strongest_baseline_nested_score": nested["baseline_scores"][strongest],
                "baseline_category_selected": category_selected,
            })
            for category, key in category_selected.items():
                loading = loading_from_key(
                    key, prepared.train_residuals[:, 0], prepared.train_residuals[:, 1],
                )
                baseline_scores[category][held] = prepared.held_residuals @ loading
            strongest_scores[held] = baseline_scores[strongest.split(":", 1)[0]][held]
        selections.append(row)
    if not np.isfinite(candidate_scores).all():
        raise RuntimeError("outer candidate predictions are incomplete")
    result = {
        "candidate_scores": candidate_scores,
        "selections": selections,
        "loadings": np.asarray(loadings),
        "confounds": confounds,
        "candidate_summary": correlation_summary(candidate_scores, folds, subsets),
    }
    if baselines:
        if not np.isfinite(strongest_scores).all():
            raise RuntimeError("outer baseline predictions are incomplete")
        result.update({
            "strongest_scores": strongest_scores,
            "baseline_scores": baseline_scores,
            "strongest_summary": correlation_summary(strongest_scores, folds, subsets),
            "baseline_summaries": {
                key: correlation_summary(values, folds, subsets)
                for key, values in baseline_scores.items()
            },
        })
    return result


def _metric_from_indices(scores, folds, subsets, sampled_by_cell, kind):
    correlations = []
    for fold in range(5):
        for subset in range(len(SUBSETS)):
            indices = sampled_by_cell[fold, subset]
            if kind == "qwen":
                left, right = scores[indices, 0], scores[indices, 1]
            else:
                left, right = scores[indices, 2], scores[indices, :2].mean(axis=1)
            correlations.append(safe_correlation(left, right))
    return fisher_macro(correlations)


def paired_bootstrap(candidate, baseline, folds, subsets, groups):
    rng = np.random.default_rng(SEED + 7001)
    group_cells = {}
    for fold in range(5):
        for subset in range(len(SUBSETS)):
            mask = (folds == fold) & (subsets == subset)
            unique = np.unique(groups[mask])
            group_cells[fold, subset] = (
                unique,
                {group: np.flatnonzero(mask & (groups == group)) for group in unique},
            )
    qwen, llama, delta = [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_by_cell = {}
        for cell, (unique, lookup) in group_cells.items():
            sampled = rng.choice(unique, size=len(unique), replace=True)
            sampled_by_cell[cell] = np.concatenate([lookup[group] for group in sampled])
        qwen.append(_metric_from_indices(candidate, folds, subsets, sampled_by_cell, "qwen"))
        llama_candidate = _metric_from_indices(
            candidate, folds, subsets, sampled_by_cell, "llama"
        )
        llama_baseline = _metric_from_indices(
            baseline, folds, subsets, sampled_by_cell, "llama"
        )
        llama.append(llama_candidate)
        delta.append(llama_candidate - llama_baseline)
    return {
        "qwen": np.asarray(qwen), "llama": np.asarray(llama),
        "llama_delta_vs_strongest": np.asarray(delta),
    }


def _training_nulls(arrays):
    values, diagnostics = [], []
    for draw in range(TRAIN_NULL_DRAWS):
        fitted = _outer_fit(
            arrays, transform_kind="mixed_v2",
            pair_null_seed=SEED + 1000000 + draw, baselines=False,
        )
        values.append(fitted["candidate_summary"]["qwen_macro"])
        fold_diagnostics = []
        for row in fitted["selections"]:
            outer = row["outer_null_diagnostics"]
            nested = row["nested_null_diagnostics"]
            fold_diagnostics.append({
                "fold": row["fold"],
                "corrca_selected": row["corrca_selected"],
                "outer_train_stratum_sizes": [item["n"] for item in outer["train_strata"]],
                "outer_held_stratum_sizes": [item["n"] for item in outer["held_strata"]],
                "outer_minimum_stratum_size": min(
                    item["n"] for item in outer["train_strata"] + outer["held_strata"]
                ),
                "outer_train_fixed_points": int(np.sum(
                    np.asarray(outer["train_permutation"])
                    == np.arange(len(outer["train_permutation"]))
                )),
                "outer_held_fixed_points": int(np.sum(
                    np.asarray(outer["held_permutation"])
                    == np.arange(len(outer["held_permutation"]))
                )),
                "outer_train_permutation_sha256": hashlib.sha256(
                    np.asarray(outer["train_permutation"], dtype="<i8").tobytes()
                ).hexdigest(),
                "outer_held_permutation_sha256": hashlib.sha256(
                    np.asarray(outer["held_permutation"], dtype="<i8").tobytes()
                ).hexdigest(),
                "nested": [{
                    "fold": item["fold"],
                    "train_fixed_points": int(np.sum(
                        np.asarray(item["train"]) == np.arange(len(item["train"]))
                    )),
                    "valid_fixed_points": int(np.sum(
                        np.asarray(item["valid"]) == np.arange(len(item["valid"]))
                    )),
                    "train_permutation_sha256": hashlib.sha256(
                        np.asarray(item["train"], dtype="<i8").tobytes()
                    ).hexdigest(),
                    "valid_permutation_sha256": hashlib.sha256(
                        np.asarray(item["valid"], dtype="<i8").tobytes()
                    ).hexdigest(),
                    "length_edge_counts": {
                        str(key): len(value) for key, value in item["length_edges"].items()
                    },
                } for item in nested],
            })
        diagnostics.append({"draw": draw, "folds": fold_diagnostics})
        if (draw + 1) % 20 == 0:
            print(f"training-null {draw + 1}/{TRAIN_NULL_DRAWS}", flush=True)
    return np.asarray(values), diagnostics


def _held_nulls(arrays, candidate_scores):
    folds, subsets, groups = arrays["outer_fold"], arrays["subset"], arrays["group_id"]
    cov = arrays["covariates"]
    values, diagnostics = [], []
    for draw in range(HELD_NULL_DRAWS):
        shuffled = candidate_scores.copy()
        fold_diagnostics = []
        for fold in range(5):
            train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            edges = fit_length_edges(subsets[train], cov[train, 0, 1])
            permutation, strata = conditional_derangement(
                subsets[held], cov[held, 2, 1],
                np.random.default_rng(SEED + 2000000 + draw * 101 + fold),
                edges_by_subset=edges, forbidden_groups=groups[held],
            )
            shuffled[held, 2] = candidate_scores[held[permutation], 2]
            fold_diagnostics.append({
                "fold": fold,
                "stratum_sizes": [row["n"] for row in strata],
                "minimum_stratum_size": min(row["n"] for row in strata),
                "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
                "permutation_sha256": hashlib.sha256(
                    np.asarray(permutation, dtype="<i8").tobytes()
                ).hexdigest(),
                "length_edge_counts": {str(key): len(value) for key, value in edges.items()},
            })
        values.append(correlation_summary(shuffled, folds, subsets)["llama_macro"])
        diagnostics.append({"draw": draw, "folds": fold_diagnostics})
    return np.asarray(values), diagnostics


def _loso(arrays):
    raw, cov = arrays["raw"], arrays["covariates"]
    subsets, groups = arrays["subset"], arrays["group_id"]
    rows = []
    for held_subset in range(len(SUBSETS)):
        train = np.flatnonzero(subsets != held_subset)
        held = np.flatnonzero(subsets == held_subset)
        nested = select_nested(
            raw[train], cov[train], subsets[train], groups[train], FEATURE_ROSTER,
            transform_kind="mixed_v2", seed=SEED + 40000 + held_subset,
            pair_null_seed=None, baselines=False,
        )
        prepared = prepare_fold(
            raw[train], cov[train], subsets[train], groups[train],
            raw[held], cov[held], subsets[held], FEATURE_ROSTER,
            transform_kind="mixed_v2", split_seed=SEED + 50000 + held_subset,
        )
        loading = loading_from_key(
            nested["corrca_selected"], prepared.train_residuals[:, 0],
            prepared.train_residuals[:, 1],
        )
        scores = prepared.held_residuals @ loading
        rows.append({
            "held_subset": SUBSETS[held_subset],
            "selected": nested["corrca_selected"],
            "n": len(held),
            "qwen_correlation": safe_correlation(scores[:, 0], scores[:, 1]),
            "llama_correlation": safe_correlation(scores[:, 2], scores[:, :2].mean(axis=1)),
        })
    return rows


def _component_stability(loadings):
    loadings = np.asarray(loadings, dtype=float)
    rows = []
    for left in range(len(loadings)):
        for right in range(left + 1, len(loadings)):
            cosine = float(loadings[left] @ loadings[right] /
                           (np.linalg.norm(loadings[left]) * np.linalg.norm(loadings[right])))
            rows.append({"left_fold": left, "right_fold": right, "squared_cosine": cosine ** 2})
    return rows


def _confound_gate(confounds):
    fold_rows = []
    for fold in range(5):
        view_rows = []
        for view in VIEWS:
            selected = [row for row in confounds if row["fold"] == fold and row["view"] == view]
            view_rows.append({
                "view": view,
                "macro_r2": float(np.mean([row["r2"] for row in selected])),
                "macro_abs_correlation": float(np.mean([
                    row["abs_correlation"] for row in selected
                ])),
            })
        fold_rows.append({
            "fold": fold,
            "views": view_rows,
            "worst_macro_r2": max(row["macro_r2"] for row in view_rows),
            "worst_macro_abs_correlation": max(
                row["macro_abs_correlation"] for row in view_rows
            ),
        })
    return fold_rows


def _artifact_hashes(out):
    files = sorted(
        path for path in Path(out).iterdir()
        if path.is_file() and path.name != "ARTIFACT_HASHES.json"
    )
    return {path.name: sha256_file(path) for path in files}


def _report(payload):
    summary = payload["primary_summary"]
    gate = payload["gates"]
    return f"""# Automatic group-free IU — Phase A4

## Outcome

- Shared-repeatability premise: **{payload['premise_verdict']}**
- Detector/target verdict: **{payload['detector_verdict']}**
- Correctness or step labels accessed: **no**

The primary residualized CorrCA component has Qwen repeatability
**{summary['qwen_macro']:.4f}** and one-Llama external structural correlation
**{summary['llama_macro']:.4f}**. The preselected strongest paired baseline has
Llama correlation **{payload['strongest_baseline_summary']['llama_macro']:.4f}**;
the observed delta is **{payload['bootstrap']['observed_llama_delta']:+.4f}**
with 95% interval **[{payload['bootstrap']['llama_delta_ci'][0]:+.4f},
{payload['bootstrap']['llama_delta_ci'][1]:+.4f}]**.

The training-pair null 95th percentile is **{payload['nulls']['training_q95']:.4f}**
and the held-Llama conditional-pair null 95th percentile is
**{payload['nulls']['held_q95']:.4f}**. Minimum outer-fold loading squared
cosine is **{payload['stability']['minimum_squared_cosine']:.4f}**.

## Gates

| gate | pass |
|---|---:|
{chr(10).join(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in gate.items())}

## Interpretation boundary

This experiment tests one shared/repeatable telemetry component after
feature-level text/length removal. It does not identify a complementary
scorer-sensitive component, and neither shared nor residual variation is
identified as hallucination. Because the fixed responses contain no legal
target-changing contrast, `CLOSE_NO_TARGET_CONTRAST` was frozen before this
run and holds regardless of the structural premise result. A5 begins next.
"""


def run(out):
    out = Path(out)
    boundary, arrays = load_and_verify_boundary(out)
    print("boundary verified; starting real outer fits", flush=True)
    real = _outer_fit(arrays, transform_kind="mixed_v2", baselines=True)
    raw_z = _outer_fit(arrays, transform_kind="raw_z", baselines=False)
    stability_rows = _component_stability(real["loadings"])
    confound_folds = _confound_gate(real["confounds"])
    bootstrap = paired_bootstrap(
        real["candidate_scores"], real["strongest_scores"],
        arrays["outer_fold"], arrays["subset"], arrays["group_id"],
    )
    print("real fits and bootstrap complete; starting training-pair null", flush=True)
    training_null, null_selections = _training_nulls(arrays)
    print("training-pair null complete; starting held-pair null", flush=True)
    held_null, held_null_diagnostics = _held_nulls(arrays, real["candidate_scores"])
    loso = _loso(arrays)

    summary = real["candidate_summary"]
    strongest = real["strongest_summary"]
    qwen_ci = np.quantile(bootstrap["qwen"], [0.025, 0.975])
    llama_ci = np.quantile(bootstrap["llama"], [0.025, 0.975])
    delta_ci = np.quantile(bootstrap["llama_delta_vs_strongest"], [0.025, 0.975])
    observed_delta = summary["llama_macro"] - strongest["llama_macro"]
    train_q95, held_q95 = np.quantile(training_null, 0.95), np.quantile(held_null, 0.95)
    min_stability = min(row["squared_cosine"] for row in stability_rows)
    loso_llama = [row["llama_correlation"] for row in loso]

    gates = {
        "material_delta_vs_preselected_paired_baseline": bool(
            observed_delta >= 0.02 and delta_ci[0] > 0.0
        ),
        "positive_every_subset_and_macro_intervals": bool(
            all(row["qwen"] > 0 and row["llama"] > 0 for row in summary["by_subset"].values())
            and qwen_ci[0] > 0 and llama_ci[0] > 0
        ),
        "feature_level_text_length_confound_control": bool(all(
            row["worst_macro_r2"] <= 0.10
            and row["worst_macro_abs_correlation"] <= 0.35
            for row in confound_folds
        )),
        "conditional_pair_nulls": bool(
            summary["qwen_macro"] - train_q95 >= 0.02
            and summary["llama_macro"] - held_q95 >= 0.02
        ),
        "leave_one_subset_transfer": bool(
            sum(value > 0 for value in loso_llama) >= 3
            and fisher_macro(loso_llama) > 0
        ),
        "outer_loading_stability": bool(min_stability >= 0.70),
    }
    premise = (
        "PASS_SHARED_REPEATABLE_COMPONENT_PREMISE"
        if all(gates.values()) else "CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE"
    )

    np.savez_compressed(
        out / "held_predictions.npz",
        candidate=real["candidate_scores"],
        strongest_baseline=real["strongest_scores"],
        cca=real["baseline_scores"]["cca"],
        diagonal=real["baseline_scores"]["diagonal"],
        single=real["baseline_scores"]["single"],
        pca=real["baseline_scores"]["pca"],
        equal=real["baseline_scores"]["equal"],
        raw_z_candidate=raw_z["candidate_scores"],
        outer_fold=arrays["outer_fold"], subset=arrays["subset"],
        item_id=arrays["item_id"], group_id=arrays["group_id"],
    )
    np.savez_compressed(
        out / "resampling_distributions.npz",
        bootstrap_qwen=bootstrap["qwen"],
        bootstrap_llama=bootstrap["llama"],
        bootstrap_llama_delta=bootstrap["llama_delta_vs_strongest"],
        training_pair_null=training_null,
        held_pair_null=held_null,
    )
    write_csv(out / "confound_cells.csv", real["confounds"])
    write_csv(out / "stability.csv", stability_rows)
    write_json(out / "outer_selections.json", real["selections"])
    write_json(out / "training_null_selections.json", null_selections)
    write_json(out / "held_null_diagnostics.json", held_null_diagnostics)
    write_json(out / "leave_one_subset_out.json", loso)
    metrics = {
        "primary": real["candidate_summary"],
        "strongest_baseline": real["strongest_summary"],
        "baselines": real["baseline_summaries"],
        "raw_z_sensitivity": raw_z["candidate_summary"],
        "confound_folds": confound_folds,
    }
    write_json(out / "STRUCTURAL_METRICS.json", metrics)
    payload = {
        "version": VERSION,
        "boundary_sha256": sha256_file(out / "A4_BOUNDARY.json"),
        "premise_verdict": premise,
        "detector_verdict": "CLOSE_NO_TARGET_CONTRAST",
        "correctness_or_step_targets_accessed": False,
        "primary_summary": real["candidate_summary"],
        "strongest_baseline_summary": real["strongest_summary"],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "qwen_ci": qwen_ci,
            "llama_ci": llama_ci,
            "observed_llama_delta": observed_delta,
            "llama_delta_ci": delta_ci,
        },
        "nulls": {
            "training_draws": TRAIN_NULL_DRAWS,
            "training_q95": train_q95,
            "training_margin": summary["qwen_macro"] - train_q95,
            "held_draws": HELD_NULL_DRAWS,
            "held_q95": held_q95,
            "held_margin": summary["llama_macro"] - held_q95,
        },
        "stability": {
            "minimum_squared_cosine": min_stability,
            "pairs": stability_rows,
        },
        "confound_folds": confound_folds,
        "leave_one_subset_out": loso,
        "raw_z_sensitivity": raw_z["candidate_summary"],
        "gates": gates,
        "source_sha256_verified": boundary["source_sha256"],
    }
    write_json(out / "A4_COMPLETE.json", payload)
    (out / "REPORT.md").write_text(_report(payload), encoding="utf-8")
    write_json(out / "ARTIFACT_HASHES.json", _artifact_hashes(out))
    print(json.dumps({
        "premise_verdict": premise,
        "detector_verdict": payload["detector_verdict"],
        "qwen_macro": summary["qwen_macro"],
        "llama_macro": summary["llama_macro"],
        "delta_vs_baseline": observed_delta,
        "gates": gates,
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.cache, args.out)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
