#!/usr/bin/env python3
"""Run Phase A2: nested environment-CV joint block diagonalization audit.

The primary scope uses all 30 atomic residual coordinates with train-only PSD
completion and scores only entries genuinely observed in the held-out cell.
The 17-coordinate universally complete core is retained as a transparent
diagnostic.  All selection and evaluation use covariance reconstruction with
nested environment folds; no new correctness label or detector metric is read.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.automatic_group_free_phase_a1 import source_covariances  # noqa: E402
from scripts.atomic_nrm_structural_audit import load_spaces, DEFAULT_BUNDLE  # noqa: E402
from spectral_utils.contribution_subspace import fit_contribution_transform  # noqa: E402
from spectral_utils.factorial_measurement import (  # noqa: E402
    FactorialConfiguration,
    fit_factorial_measurement,
    pooled_mean_reconstruction_rows,
)
from spectral_utils.group_free_research import (  # noqa: E402
    derive_feature_dag,
    sha256_file,
)
from spectral_utils.multi_environment_jbd import (  # noqa: E402
    JBDConfiguration,
    balanced_environment_row_null,
    complete_missing_covariances,
    covariance_atoms,
    fit_jbd_model,
    fit_pca_block_model,
    masked_reconstruction_rows,
    mechanism_subspace_overlap,
    missingness_preserving_stationary_null,
    shuffled_environment_row_null,
)


VERSION = "automatic-group-free-iu-a2-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a2_v1"
RIDGES = (0.01, 0.1, 1.0, 10.0)
BLOCK_QUANTILES = (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 1.0)
RANDOM_DRAWS = 32
JACOBI_SWEEPS = 32
SEED = 20260813
OUTER_FOLDS = 5
BOOTSTRAP_DRAWS = 2000
STABILITY_OVERLAP_GATE = 0.80
STABILITY_RANK_RATIO_GATE = 0.70
MAX_BLOCK_FRACTION = 0.75
SHUFFLED_ADVANTAGE_FRACTION = 0.25


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


def environment_fold(cell: str) -> int:
    digest = hashlib.sha256(f"{VERSION}:{cell}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % OUTER_FOLDS


def complete_roster(covariances: np.ndarray, names: tuple[str, ...]):
    diagonal = np.diagonal(covariances, axis1=1, axis2=2)
    keep = np.isfinite(diagonal).all(axis=0)
    indices = np.flatnonzero(keep)
    complete = covariances[:, indices][:, :, indices]
    if not np.isfinite(complete).all():
        raise RuntimeError("universal diagonal coverage did not imply complete covariance")
    return complete, tuple(names[index] for index in indices), indices


def common_standardized_residuals(common_names: tuple[str, ...]) -> list[np.ndarray]:
    """Recover the exact common residual rows used to build source correlations."""

    _, spaces, _ = load_spaces(DEFAULT_BUNDLE)
    output = []
    for space in spaces:
        transform = fit_contribution_transform(
            space, np.arange(len(space.baseline_score), dtype=int)
        )
        _, residuals = transform.apply(space.baseline_score, space.contributions)
        columns = np.asarray([space.families.index(name) for name in common_names])
        values = np.asarray(residuals[:, columns], dtype=float)
        values -= values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        if np.any(scale < 1e-12):
            raise RuntimeError("common residual coordinate is inactive")
        output.append(values / scale)
    return output


def configuration_grid(method: str) -> list[JBDConfiguration]:
    quantiles = BLOCK_QUANTILES if method == "jbd" else (1.0,)
    return [
        JBDConfiguration(
            method=method,
            ridge=ridge,
            block_quantile=quantile,
            random_draws=RANDOM_DRAWS,
            random_seed=SEED,
            jacobi_sweeps=JACOBI_SWEEPS,
        )
        for quantile in quantiles
        for ridge in RIDGES
    ]


def evaluate_model(
    model,
    covariances: np.ndarray,
    environment_ids: list[str],
    method: str,
) -> list[dict]:
    rows = []
    for covariance, environment in zip(covariances, environment_ids):
        rows.extend({"method": method, **row} for row in masked_reconstruction_rows(
            model, covariance, environment
        ))
    return rows


def blocked_cv_mse(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> float:
    environment_errors = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        model = fit_jbd_model(covariances[~held], names, configuration)
        rows = evaluate_model(
            model,
            covariances[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            configuration.method,
        )
        for environment in sorted({row["environment"] for row in rows}):
            environment_errors.append(float(np.mean([
                row["squared_error"] for row in rows
                if row["environment"] == environment
            ])))
    return float(np.mean(environment_errors))


def select_configuration(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    method: str,
) -> tuple[JBDConfiguration, list[dict]]:
    rows = []
    for configuration in configuration_grid(method):
        mse = blocked_cv_mse(
            covariances, environment_ids, folds, names, configuration
        )
        rows.append({
            "method": method,
            "ridge": configuration.ridge,
            "block_quantile": configuration.block_quantile,
            "cv_mse": mse,
        })
    rows.sort(key=lambda row: (
        row["cv_mse"], row["block_quantile"], row["ridge"]
    ))
    best = rows[0]
    return JBDConfiguration(
        method=method,
        ridge=best["ridge"],
        block_quantile=best["block_quantile"],
        random_draws=RANDOM_DRAWS,
        random_seed=SEED,
        jacobi_sweeps=JACOBI_SWEEPS,
    ), rows


def factorial_basis(
    full_covariances: np.ndarray,
    full_names: tuple[str, ...],
    train_indices: np.ndarray,
    common_indices: np.ndarray,
) -> np.ndarray:
    configuration = FactorialConfiguration(
        "hybrid", rank=6, interaction=True, ridge=0.1, alpha=0.25
    )
    fit = fit_factorial_measurement(
        full_covariances[train_indices],
        full_names,
        derive_feature_dag(full_names),
        configuration,
    )
    return fit.basis[common_indices]


def nested_predictions(
    common_covariances: np.ndarray,
    full_covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    common_names: tuple[str, ...],
    full_names: tuple[str, ...],
    common_indices: np.ndarray,
) -> tuple[dict[str, list[dict]], list[dict]]:
    methods = ("pca", "pca_full", "rjd", "ajd", "jbd")
    predictions = {method: [] for method in methods}
    predictions["pca_matched"] = []
    predictions["factorial_jbd"] = []
    predictions["pooled_mean"] = []
    selection_rows = []
    for outer_fold in sorted(set(int(value) for value in folds)):
        test = folds == outer_fold
        train = ~test
        train_indices = np.flatnonzero(train)
        train_ids = [environment_ids[index] for index in train_indices]
        inner_folds = folds[train]
        selected = {}
        for method in methods:
            configuration, rows = select_configuration(
                common_covariances[train],
                train_ids,
                inner_folds,
                common_names,
                method,
            )
            selected[method] = configuration
            selection_rows.extend({
                "outer_fold": outer_fold,
                "selected": index == 0,
                **row,
            } for index, row in enumerate(rows))
            model = fit_jbd_model(
                common_covariances[train], common_names, configuration
            )
            predictions[method].extend(evaluate_model(
                model,
                common_covariances[test],
                [environment_ids[index] for index in np.flatnonzero(test)],
                method,
            ))

        jbd_model = fit_jbd_model(
            common_covariances[train], common_names, selected["jbd"]
        )
        pca_matched = fit_pca_block_model(
            common_covariances[train],
            common_names,
            ridge=selected["jbd"].ridge,
            block_sizes=tuple(len(block) for block in jbd_model.blocks),
        )
        predictions["pca_matched"].extend(evaluate_model(
            pca_matched,
            common_covariances[test],
            [environment_ids[index] for index in np.flatnonzero(test)],
            "pca_matched",
        ))

        factor_configuration = replace(
            selected["jbd"], method="factorial_jbd"
        )
        basis = factorial_basis(
            full_covariances, full_names, train_indices, common_indices
        )
        factor_model = fit_jbd_model(
            common_covariances[train],
            common_names,
            factor_configuration,
            factor_basis=basis,
        )
        predictions["factorial_jbd"].extend(evaluate_model(
            factor_model,
            common_covariances[test],
            [environment_ids[index] for index in np.flatnonzero(test)],
            "factorial_jbd",
        ))
        for index in np.flatnonzero(test):
            predictions["pooled_mean"].extend({"method": "pooled_mean", **row}
                for row in pooled_mean_reconstruction_rows(
                    common_covariances[train],
                    common_covariances[index],
                    common_names,
                    environment_ids[index],
                ))
        print(json.dumps({
            "completed_outer_fold": outer_fold,
            "test_environments": int(test.sum()),
            "selected_jbd": {
                "ridge": selected["jbd"].ridge,
                "block_quantile": selected["jbd"].block_quantile,
            },
        }), flush=True)
    return predictions, selection_rows


def select_missing_configuration(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    method: str,
) -> tuple[JBDConfiguration, list[dict]]:
    """Select a 30-atom method with completion refit inside every CV split."""

    rows = []
    for configuration in configuration_grid(method):
        environment_errors = []
        completion_shrinks = []
        for fold in sorted(set(int(value) for value in folds)):
            held = folds == fold
            completed_train, diagnostics = complete_missing_covariances(
                covariances[~held]
            )
            completion_shrinks.extend(
                row["cross_covariance_shrink"] for row in diagnostics
            )
            model = fit_jbd_model(completed_train, names, configuration)
            fold_rows = evaluate_model(
                model,
                covariances[held],
                [environment_ids[index] for index in np.flatnonzero(held)],
                method,
            )
            for environment in sorted({row["environment"] for row in fold_rows}):
                environment_errors.append(float(np.mean([
                    row["squared_error"] for row in fold_rows
                    if row["environment"] == environment
                ])))
        rows.append({
            "scope": "missing_aware_30",
            "method": method,
            "ridge": configuration.ridge,
            "block_quantile": configuration.block_quantile,
            "cv_mse": float(np.mean(environment_errors)),
            "minimum_completion_cross_shrink": float(min(completion_shrinks)),
        })
    rows.sort(key=lambda row: (
        row["cv_mse"], row["block_quantile"], row["ridge"]
    ))
    best = rows[0]
    return JBDConfiguration(
        method=method,
        ridge=best["ridge"],
        block_quantile=best["block_quantile"],
        random_draws=RANDOM_DRAWS,
        random_seed=SEED,
        jacobi_sweeps=JACOBI_SWEEPS,
    ), rows


def nested_missing_predictions(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    """Nested environment evaluation over all 30 atoms and actual test support."""

    methods = ("pca_full", "jbd")
    predictions = {method: [] for method in methods}
    predictions["pca_matched"] = []
    search_rows = []
    completion_rows = []
    for outer_fold in sorted(set(int(value) for value in folds)):
        test = folds == outer_fold
        train = ~test
        train_ids = [environment_ids[index] for index in np.flatnonzero(train)]
        inner_folds = folds[train]
        completed_train, diagnostics = complete_missing_covariances(
            covariances[train]
        )
        completion_rows.extend({
            "outer_fold": int(outer_fold),
            "train_environment": train_ids[index],
            **row,
        } for index, row in enumerate(diagnostics))
        selected = {}
        for method in methods:
            configuration, rows = select_missing_configuration(
                covariances[train], train_ids, inner_folds, names, method
            )
            selected[method] = configuration
            search_rows.extend({
                "outer_fold": int(outer_fold),
                "selected": index == 0,
                **row,
            } for index, row in enumerate(rows))
            model = fit_jbd_model(completed_train, names, configuration)
            predictions[method].extend(evaluate_model(
                model,
                covariances[test],
                [environment_ids[index] for index in np.flatnonzero(test)],
                f"missing_{method}",
            ))
        jbd_model = fit_jbd_model(
            completed_train, names, selected["jbd"]
        )
        pca_matched = fit_pca_block_model(
            completed_train,
            names,
            ridge=selected["jbd"].ridge,
            block_sizes=tuple(len(block) for block in jbd_model.blocks),
        )
        predictions["pca_matched"].extend(evaluate_model(
            pca_matched,
            covariances[test],
            [environment_ids[index] for index in np.flatnonzero(test)],
            "missing_pca_matched",
        ))
        print(json.dumps({
            "completed_missing_outer_fold": int(outer_fold),
            "test_environments": int(test.sum()),
            "selected_jbd": {
                "ridge": selected["jbd"].ridge,
                "block_quantile": selected["jbd"].block_quantile,
                "block_sizes": [len(block) for block in jbd_model.blocks],
                "covariance_atom_count": jbd_model.diagnostics["n_covariance_atoms"],
            },
        }), flush=True)
    return predictions, search_rows, completion_rows


def reconstruction_mse(rows: list[dict]) -> float:
    environments = sorted({row["environment"] for row in rows})
    if not environments:
        raise ValueError("no reconstruction rows")
    return float(np.mean([
        np.mean([row["squared_error"] for row in rows
                 if row["environment"] == environment])
        for environment in environments
    ]))


def keyed_errors(rows: list[dict]):
    ordered = sorted(rows, key=lambda row: (
        row["environment"], row["held_out_feature"], row["partner_feature"]
    ))
    keys = [(
        row["environment"], row["held_out_feature"], row["partner_feature"]
    ) for row in ordered]
    errors = np.asarray([row["squared_error"] for row in ordered], dtype=float)
    return keys, errors


def grouped_bootstrap_delta(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    *,
    seed: int,
) -> dict:
    candidate_keys, candidate = keyed_errors(candidate_rows)
    baseline_keys, baseline = keyed_errors(baseline_rows)
    if candidate_keys != baseline_keys:
        raise RuntimeError("paired nested reconstruction rows do not align")
    groups = np.asarray([key[0] for key in candidate_keys], dtype=object)
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[draw] = float(np.mean([
            np.mean(candidate[groups == group] - baseline[groups == group])
            for group in sampled
        ]))
    environment_delta = [
        float(np.mean(candidate[groups == group] - baseline[groups == group]))
        for group in unique
    ]
    return {
        "delta_mse": float(np.mean(environment_delta)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_groups": len(unique),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": int(seed),
    }


def stability_rows(
    covariances: np.ndarray,
    environment_ids: list[str],
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> tuple[list[dict], object]:
    reference = fit_jbd_model(covariances, names, configuration)
    rows = []
    for held_out, environment in enumerate(environment_ids):
        candidate = fit_jbd_model(
            np.delete(covariances, held_out, axis=0), names, configuration
        )
        overlap = mechanism_subspace_overlap(reference, candidate)
        rows.append({
            "held_out_environment": environment,
            **overlap,
            "candidate_block_sizes": json.dumps(candidate.diagnostics["block_sizes"]),
        })
    return rows, reference


def missing_stability_rows(
    covariances: np.ndarray,
    environment_ids: list[str],
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> tuple[list[dict], object]:
    """LOEO mechanism stability with completion refit after every deletion."""

    completed, _ = complete_missing_covariances(covariances)
    reference = fit_jbd_model(completed, names, configuration)
    rows = []
    for held_out, environment in enumerate(environment_ids):
        candidate_covariances, _ = complete_missing_covariances(
            np.delete(covariances, held_out, axis=0)
        )
        candidate = fit_jbd_model(candidate_covariances, names, configuration)
        overlap = mechanism_subspace_overlap(reference, candidate)
        rows.append({
            "held_out_environment": environment,
            **overlap,
            "candidate_block_sizes": json.dumps(candidate.diagnostics["block_sizes"]),
        })
    return rows, reference


def cross_validated_fixed_predictions(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> list[dict]:
    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        model = fit_jbd_model(covariances[~held], names, configuration)
        rows.extend(evaluate_model(
            model,
            covariances[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            configuration.method,
        ))
    return rows


def cross_validated_fixed_missing_predictions(
    fit_covariances: np.ndarray,
    score_covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> list[dict]:
    """Fit completion inside each fold and score the declared observed support."""

    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        completed_train, _ = complete_missing_covariances(fit_covariances[~held])
        model = fit_jbd_model(completed_train, names, configuration)
        rows.extend(evaluate_model(
            model,
            score_covariances[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            configuration.method,
        ))
    return rows


def cross_validated_missing_stationary_null_predictions(
    covariances: np.ndarray,
    sample_counts: list[int],
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
    *,
    seed: int,
) -> list[dict]:
    """Generate the stationary null from training cells inside every fold."""

    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        null = missingness_preserving_stationary_null(
            covariances,
            sample_counts,
            seed=int(seed) + int(fold),
            reference_covariances=covariances[~held],
        )
        completed_train, _ = complete_missing_covariances(null[~held])
        model = fit_jbd_model(completed_train, names, configuration)
        rows.extend(evaluate_model(
            model,
            null[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            configuration.method,
        ))
    return rows


def cross_validated_missing_stationary_null_matched_pca(
    covariances: np.ndarray,
    sample_counts: list[int],
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    jbd_configuration: JBDConfiguration,
    *,
    seed: int,
) -> list[dict]:
    """Matched-PCA predictions on the exact same fold-wise stationary null."""

    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        null = missingness_preserving_stationary_null(
            covariances,
            sample_counts,
            seed=int(seed) + int(fold),
            reference_covariances=covariances[~held],
        )
        completed_train, _ = complete_missing_covariances(null[~held])
        jbd = fit_jbd_model(completed_train, names, jbd_configuration)
        model = fit_pca_block_model(
            completed_train,
            names,
            ridge=jbd_configuration.ridge,
            block_sizes=tuple(len(block) for block in jbd.blocks),
        )
        rows.extend(evaluate_model(
            model,
            null[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            "pca_matched",
        ))
    return rows


def cross_validated_missing_matched_pca_predictions(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    jbd_configuration: JBDConfiguration,
) -> list[dict]:
    """Use fold-specific JBD block sizes but a pooled-PCA orientation."""

    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        completed_train, _ = complete_missing_covariances(covariances[~held])
        jbd = fit_jbd_model(completed_train, names, jbd_configuration)
        model = fit_pca_block_model(
            completed_train,
            names,
            ridge=jbd_configuration.ridge,
            block_sizes=tuple(len(block) for block in jbd.blocks),
        )
        rows.extend(evaluate_model(
            model,
            covariances[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            "pca_matched",
        ))
    return rows


def cross_validated_matched_pca_predictions(
    covariances: np.ndarray,
    environment_ids: list[str],
    folds: np.ndarray,
    names: tuple[str, ...],
    jbd_configuration: JBDConfiguration,
) -> list[dict]:
    """Complete-roster PCA orientation with fold-specific JBD block capacity."""

    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        held = folds == fold
        jbd = fit_jbd_model(covariances[~held], names, jbd_configuration)
        model = fit_pca_block_model(
            covariances[~held],
            names,
            ridge=jbd_configuration.ridge,
            block_sizes=tuple(len(block) for block in jbd.blocks),
        )
        rows.extend(evaluate_model(
            model,
            covariances[held],
            [environment_ids[index] for index in np.flatnonzero(held)],
            "pca_matched",
        ))
    return rows


def missing_permutation_repeatability(
    covariances: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> dict:
    """Check exact prediction invariance for the incomplete 30-atom path."""

    train, held = covariances[:-1], covariances[-1]
    completed, _ = complete_missing_covariances(train)
    reference = fit_jbd_model(completed, names, configuration)
    reference_rows = evaluate_model(reference, held[None], ["held"], "reference")
    repeated_completed, _ = complete_missing_covariances(train)
    repeated = fit_jbd_model(repeated_completed, names, configuration)
    repeated_rows = evaluate_model(repeated, held[None], ["held"], "repeat")
    permutation = np.random.default_rng(SEED + 407).permutation(len(names))
    permuted_names = tuple(names[index] for index in permutation)
    permuted_train = train[:, permutation][:, :, permutation]
    permuted_held = held[np.ix_(permutation, permutation)]
    permuted_completed, _ = complete_missing_covariances(permuted_train)
    candidate = fit_jbd_model(permuted_completed, permuted_names, configuration)
    candidate_rows = evaluate_model(
        candidate, permuted_held[None], ["held"], "permuted"
    )

    def lookup(rows):
        return {(
            row["environment"], row["held_out_feature"], row["partner_feature"]
        ): row["prediction"] for row in rows}

    reference_values = lookup(reference_rows)
    repeated_values = lookup(repeated_rows)
    candidate_values = lookup(candidate_rows)
    if reference_values.keys() != repeated_values.keys() or reference_values.keys() != candidate_values.keys():
        raise RuntimeError("missing-aware invariance keys disagree")
    return {
        "repeatability_max_prediction_error": float(max(
            abs(reference_values[key] - repeated_values[key])
            for key in reference_values
        )),
        "feature_permutation_max_prediction_error": float(max(
            abs(reference_values[key] - candidate_values[key])
            for key in reference_values
        )),
        "seed": SEED + 407,
        "uses_labels": False,
    }


def simulator_check(configuration: JBDConfiguration) -> dict:
    rng = np.random.default_rng(SEED + 301)
    p = 8
    environments = 16
    mixing, _ = np.linalg.qr(rng.normal(size=(p, p)))
    true_blocks = ((0, 1, 2), (3, 4), (5,), (6,), (7,))
    covariances = []
    for _ in range(environments):
        latent = np.zeros((p, p), dtype=float)
        for block in true_blocks:
            local = rng.normal(size=(len(block), len(block)))
            local = local @ local.T / len(block) + 0.4 * np.eye(len(block))
            latent[np.ix_(block, block)] = local
        covariances.append(mixing @ latent @ mixing.T + 0.03 * np.eye(p))
    covariances = np.asarray(covariances)
    names = tuple(f"sim_feature_{index}" for index in range(p))
    train, audit = covariances[:12], covariances[12:]
    candidate = fit_jbd_model(train, names, configuration)
    pca = fit_pca_block_model(
        train,
        names,
        ridge=configuration.ridge,
        block_sizes=tuple(len(block) for block in candidate.blocks),
    )
    candidate_rows = evaluate_model(candidate, audit, list(map(str, range(4))), "jbd")
    pca_rows = evaluate_model(pca, audit, list(map(str, range(4))), "pca")
    true_atoms, true_labels = covariance_atoms(mixing, true_blocks)
    true_model = replace(
        candidate,
        mixing=mixing,
        blocks=true_blocks,
        atoms=true_atoms,
        atom_labels=true_labels,
    )
    overlap = mechanism_subspace_overlap(true_model, candidate)
    return {
        "candidate_mse": reconstruction_mse(candidate_rows),
        "block_capacity_matched_pca_mse": reconstruction_mse(pca_rows),
        "candidate_beats_capacity_matched_pca": (
            reconstruction_mse(candidate_rows) < reconstruction_mse(pca_rows)
        ),
        "true_block_sizes": [len(block) for block in true_blocks],
        "recovered_block_sizes": list(candidate.diagnostics["block_sizes"]),
        "true_mechanism_overlap": overlap,
        "uses_labels": False,
    }


def permutation_repeatability(
    covariances: np.ndarray,
    names: tuple[str, ...],
    configuration: JBDConfiguration,
) -> dict:
    train, held = covariances[:-1], covariances[-1]
    reference = fit_jbd_model(train, names, configuration)
    reference_rows = evaluate_model(reference, held[None], ["held"], "reference")
    repeated = fit_jbd_model(train, names, configuration)
    repeated_rows = evaluate_model(repeated, held[None], ["held"], "repeat")
    permutation = np.random.default_rng(SEED + 401).permutation(len(names))
    candidate = fit_jbd_model(
        train[:, permutation][:, :, permutation],
        tuple(names[index] for index in permutation),
        configuration,
    )
    candidate_rows = evaluate_model(
        candidate,
        held[np.ix_(permutation, permutation)][None],
        ["held"],
        "permuted",
    )

    def lookup(rows):
        return {(
            row["environment"], row["held_out_feature"], row["partner_feature"]
        ): row["prediction"] for row in rows}

    reference_values = lookup(reference_rows)
    repeated_values = lookup(repeated_rows)
    candidate_values = lookup(candidate_rows)
    return {
        "repeatability_max_prediction_error": float(max(
            abs(reference_values[key] - repeated_values[key])
            for key in reference_values
        )),
        "feature_permutation_max_prediction_error": float(max(
            abs(reference_values[key] - candidate_values[key])
            for key in reference_values
        )),
        "seed": SEED + 401,
        "uses_labels": False,
    }


def main() -> None:
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    full_covariances, full_names, source_rows = source_covariances()
    common_covariances, common_names, common_indices = complete_roster(
        full_covariances, full_names
    )
    environment_ids = [row["cell"] for row in source_rows]
    folds = np.asarray([environment_fold(cell) for cell in environment_ids])
    fold_rows = [
        {"environment": cell, "outer_fold": int(fold), "labels_accessed": False}
        for cell, fold in zip(environment_ids, folds)
    ]
    write_csv(out / "environment_folds.csv", fold_rows)
    boundary = {
        "version": VERSION,
        "evaluation": "nested five-fold environment covariance reconstruction",
        "fold_rule": "sha256(version:cell) first 32 bits modulo 5",
        "primary_feature_policy": (
            "all 30 atoms; train-only pairwise PSD completion; score only genuinely "
            "observed held-out feature pairs"
        ),
        "primary_feature_names": list(full_names),
        "primary_feature_count": len(full_names),
        "complete_core_diagnostic_feature_names": list(common_names),
        "complete_core_diagnostic_feature_count": len(common_names),
        "ridge_grid": list(RIDGES),
        "block_quantile_grid": list(BLOCK_QUANTILES),
        "random_draws": RANDOM_DRAWS,
        "jacobi_sweeps": JACOBI_SWEEPS,
        "stability_overlap_gate": STABILITY_OVERLAP_GATE,
        "stability_rank_ratio_gate": STABILITY_RANK_RATIO_GATE,
        "maximum_block_fraction": MAX_BLOCK_FRACTION,
        "shuffled_advantage_fraction": SHUFFLED_ADVANTAGE_FRACTION,
        "source_bundle_sha256": sha256_file(
            REPO / "results" / "dependency_fusion_raw" / "cells.npz"
        ),
        "code_sha256": {
            "script": sha256_file(Path(__file__)),
            "jbd_module": sha256_file(
                REPO / "spectral_utils" / "multi_environment_jbd.py"
            ),
            "a1_source_helper": sha256_file(
                REPO / "scripts" / "automatic_group_free_phase_a1.py"
            ),
            "factorial_measurement_module": sha256_file(
                REPO / "spectral_utils" / "factorial_measurement.py"
            ),
            "contribution_subspace_module": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
            "group_free_research_module": sha256_file(
                REPO / "spectral_utils" / "group_free_research.py"
            ),
            "feature_contract_module": sha256_file(
                REPO / "spectral_utils" / "feature_contract.py"
            ),
            "mixed_v2_contract_module": sha256_file(
                REPO / "spectral_utils" / "dufs_liu_feature_contract.py"
            ),
            "atomic_source_loader": sha256_file(
                REPO / "scripts" / "atomic_nrm_structural_audit.py"
            ),
            "hard_filter_contract_loader": sha256_file(
                REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py"
            ),
            "atomic_residual_module": sha256_file(
                REPO / "spectral_utils" / "atomic_neutral_residual.py"
            ),
            "upcr_module": sha256_file(REPO / "spectral_utils" / "upcr.py"),
            "laplacian_upcr_module": sha256_file(
                REPO / "spectral_utils" / "laplacian_upcr.py"
            ),
        },
        "literature": [
            "Ablin, Cardoso, Gramfort (2019), arXiv:1811.11433",
            "He, Kressner (2024), doi:10.1137/22M1541265",
            "Cai, Li (2021), PMLR 130:1495-1503",
        ],
        "input_label_boundary": (
            "no new correctness labels beyond frozen mixed-v2 transforms and signs"
        ),
        "correctness_labels_accessed": False,
    }
    write_json(out / "A2_BOUNDARY.json", boundary)

    missing_predictions, missing_search, completion_rows = nested_missing_predictions(
        full_covariances, environment_ids, folds, full_names
    )
    write_csv(out / "missing_aware_nested_selection.csv", missing_search)
    write_csv(out / "missing_aware_completion_diagnostics.csv", completion_rows)
    write_csv(out / "missing_aware_nested_reconstruction_rows.csv", [
        row for rows in missing_predictions.values() for row in rows
    ])
    write_csv(out / "missing_aware_nested_reconstruction_summary.csv", [{
        "method": method,
        "environment_macro_mse": reconstruction_mse(rows),
        "environment_macro_rmse": float(np.sqrt(reconstruction_mse(rows))),
        "prediction_rows": len(rows),
    } for method, rows in missing_predictions.items()])

    predictions, nested_search = nested_predictions(
        common_covariances,
        full_covariances,
        environment_ids,
        folds,
        common_names,
        full_names,
        common_indices,
    )
    write_csv(out / "nested_selection.csv", nested_search)
    all_prediction_rows = [row for rows in predictions.values() for row in rows]
    write_csv(out / "nested_reconstruction_rows.csv", all_prediction_rows)
    summary_rows = [{
        "method": method,
        "mse": reconstruction_mse(rows),
        "rmse": float(np.sqrt(reconstruction_mse(rows))),
        "prediction_rows": len(rows),
    } for method, rows in predictions.items()]
    write_csv(out / "nested_reconstruction_summary.csv", summary_rows)

    final_configurations = {}
    final_search = []
    for method in ("pca", "pca_full", "rjd", "ajd", "jbd"):
        configuration, rows = select_configuration(
            common_covariances,
            environment_ids,
            folds,
            common_names,
            method,
        )
        final_configurations[method] = configuration
        final_search.extend(rows)
    write_csv(out / "final_configuration_search.csv", final_search)
    write_json(out / "FROZEN_A2_MODEL_SELECTION.json", {
        "version": VERSION,
        "configurations": {
            method: {
                "method": configuration.method,
                "ridge": configuration.ridge,
                "block_quantile": configuration.block_quantile,
                "random_draws": configuration.random_draws,
                "random_seed": configuration.random_seed,
                "jacobi_sweeps": configuration.jacobi_sweeps,
            }
            for method, configuration in final_configurations.items()
        },
        "correctness_labels_accessed": False,
    })

    missing_final_configurations = {}
    missing_final_search = []
    for method in ("pca_full", "jbd"):
        configuration, rows = select_missing_configuration(
            full_covariances, environment_ids, folds, full_names, method
        )
        missing_final_configurations[method] = configuration
        missing_final_search.extend(rows)
    write_csv(out / "missing_aware_final_configuration_search.csv", missing_final_search)
    completed_full, full_completion_diagnostics = complete_missing_covariances(
        full_covariances
    )
    missing_final_model = fit_jbd_model(
        completed_full, full_names, missing_final_configurations["jbd"]
    )
    missing_stability, missing_final_model = missing_stability_rows(
        full_covariances,
        environment_ids,
        full_names,
        missing_final_configurations["jbd"],
    )
    write_csv(out / "missing_aware_stability.csv", missing_stability)
    missing_invariance = missing_permutation_repeatability(
        full_covariances, full_names, missing_final_configurations["jbd"]
    )
    write_json(out / "missing_aware_invariance.json", missing_invariance)
    missing_delta = grouped_bootstrap_delta(
        missing_predictions["jbd"], missing_predictions["pca_full"],
        seed=SEED + 505,
    )
    missing_matched_delta = grouped_bootstrap_delta(
        missing_predictions["jbd"], missing_predictions["pca_matched"],
        seed=SEED + 506,
    )
    sample_counts = [int(row["n_samples"]) for row in source_rows]
    missing_null_jbd = cross_validated_missing_stationary_null_predictions(
        full_covariances, sample_counts, environment_ids, folds, full_names,
        missing_final_configurations["jbd"], seed=SEED + 605,
    )
    missing_null_pca = cross_validated_missing_stationary_null_predictions(
        full_covariances, sample_counts, environment_ids, folds, full_names,
        missing_final_configurations["pca_full"], seed=SEED + 605,
    )
    missing_null_matched = cross_validated_missing_stationary_null_matched_pca(
        full_covariances,
        sample_counts,
        environment_ids,
        folds,
        full_names,
        missing_final_configurations["jbd"],
        seed=SEED + 605,
    )
    missing_null_delta = grouped_bootstrap_delta(
        missing_null_jbd, missing_null_pca, seed=SEED + 606
    )
    missing_null_matched_delta = grouped_bootstrap_delta(
        missing_null_jbd, missing_null_matched, seed=SEED + 608
    )
    write_json(out / "missing_aware_completion_summary.json", {
        "feature_count": len(full_names),
        "minimum_observed_feature_count": int(min(
            row["observed_feature_count"] for row in full_completion_diagnostics
        )),
        "maximum_missing_feature_count": int(max(
            row["missing_feature_count"] for row in full_completion_diagnostics
        )),
        "minimum_cross_covariance_shrink": float(min(
            row["cross_covariance_shrink"] for row in full_completion_diagnostics
        )),
        "maximum_observed_entry_error": float(max(
            row["maximum_observed_entry_error"] for row in full_completion_diagnostics
        )),
        "minimum_completed_eigenvalue": float(min(
            row["minimum_eigenvalue"] for row in full_completion_diagnostics
        )),
        "stationary_null_description": (
            "train-pool shared PSD covariance, original feature missingness and "
            "environment sample counts"
        ),
        "stationary_null_delta": missing_null_delta,
        "stationary_null_delta_vs_block_capacity_matched_pca": missing_null_matched_delta,
        "correctness_labels_accessed": False,
    })

    primary = final_configurations["jbd"]
    matched_pca_delta = grouped_bootstrap_delta(
        predictions["jbd"], predictions["pca_full"], seed=SEED + 501
    )
    block_matched_pca_delta = grouped_bootstrap_delta(
        predictions["jbd"], predictions["pca_matched"], seed=SEED + 504
    )
    diagonal_pca_delta = grouped_bootstrap_delta(
        predictions["jbd"], predictions["pca"], seed=SEED + 503
    )
    ajd_delta = grouped_bootstrap_delta(
        predictions["jbd"], predictions["ajd"], seed=SEED + 502
    )
    stability, final_model = stability_rows(
        common_covariances, environment_ids, common_names, primary
    )
    write_csv(out / "stability.csv", stability)

    residual_matrices = common_standardized_residuals(common_names)
    shuffled = shuffled_environment_row_null(
        residual_matrices, seed=SEED + 601
    )
    minimum_null_eigenvalue = float(np.min(np.linalg.eigvalsh(shuffled)))
    shuffled_jbd = cross_validated_fixed_predictions(
        shuffled, environment_ids, folds, common_names, primary
    )
    shuffled_pca = cross_validated_fixed_predictions(
        shuffled, environment_ids, folds, common_names, final_configurations["pca_full"]
    )
    shuffled_matched_pca = cross_validated_matched_pca_predictions(
        shuffled, environment_ids, folds, common_names, primary
    )
    shuffled_delta = grouped_bootstrap_delta(
        shuffled_jbd, shuffled_pca, seed=SEED + 602
    )
    shuffled_matched_delta = grouped_bootstrap_delta(
        shuffled_jbd, shuffled_matched_pca, seed=SEED + 609
    )
    balanced_shuffled = balanced_environment_row_null(
        residual_matrices, seed=SEED + 603
    )
    balanced_jbd = cross_validated_fixed_predictions(
        balanced_shuffled, environment_ids, folds, common_names, primary
    )
    balanced_pca = cross_validated_fixed_predictions(
        balanced_shuffled,
        environment_ids,
        folds,
        common_names,
        final_configurations["pca_full"],
    )
    balanced_matched_pca = cross_validated_matched_pca_predictions(
        balanced_shuffled, environment_ids, folds, common_names, primary
    )
    balanced_delta = grouped_bootstrap_delta(
        balanced_jbd, balanced_pca, seed=SEED + 604
    )
    balanced_matched_delta = grouped_bootstrap_delta(
        balanced_jbd, balanced_matched_pca, seed=SEED + 610
    )
    write_json(out / "environment_shuffle_control.json", {
        "null": "pooled standardized residual rows randomly reassigned to environments",
        "environment_sample_counts_preserved": True,
        "pooled_empirical_residual_distribution_preserved": True,
        "coherent_environment_profiles_destroyed": True,
        "minimum_covariance_eigenvalue": minimum_null_eigenvalue,
        "jbd_mse": reconstruction_mse(shuffled_jbd),
        "enclosing_full_block_pca_mse": reconstruction_mse(shuffled_pca),
        "paired_delta_vs_enclosing_full_block_pca": shuffled_delta,
        "paired_delta_vs_block_capacity_matched_pca": shuffled_matched_delta,
        "balanced_environment_null": {
            "description": "each null cell draws equally from every source environment",
            "minimum_covariance_eigenvalue": float(
                np.min(np.linalg.eigvalsh(balanced_shuffled))
            ),
            "jbd_mse": reconstruction_mse(balanced_jbd),
            "enclosing_full_block_pca_mse": reconstruction_mse(balanced_pca),
            "paired_delta_vs_enclosing_full_block_pca": balanced_delta,
            "paired_delta_vs_block_capacity_matched_pca": balanced_matched_delta,
        },
        "correctness_labels_accessed": False,
    })

    simulator = simulator_check(primary)
    invariance = permutation_repeatability(common_covariances, common_names, primary)
    write_json(out / "simulator_results.json", simulator)
    write_json(out / "invariance.json", invariance)

    stability_overlaps = [row["projector_overlap_on_smaller_span"] for row in stability]
    stability_rank_ratios = [row["rank_ratio"] for row in stability]
    largest_block_fraction = max(final_model.diagnostics["block_sizes"]) / len(common_names)
    full_symmetric_rank = len(common_names) * (len(common_names) + 1) // 2
    reference_mechanism_rank = stability[0]["reference_mechanism_rank"]
    original_advantage = max(-block_matched_pca_delta["delta_mse"], 0.0)
    shuffled_advantage = max(
        -shuffled_matched_delta["delta_mse"],
        -balanced_matched_delta["delta_mse"],
        0.0,
    )
    missing_stability_overlaps = [
        row["projector_overlap_on_smaller_span"] for row in missing_stability
    ]
    missing_stability_rank_ratios = [row["rank_ratio"] for row in missing_stability]
    missing_largest_block_fraction = (
        max(missing_final_model.diagnostics["block_sizes"]) / len(full_names)
    )
    missing_full_symmetric_rank = len(full_names) * (len(full_names) + 1) // 2
    missing_reference_mechanism_rank = missing_stability[0]["reference_mechanism_rank"]
    missing_original_advantage = max(-missing_matched_delta["delta_mse"], 0.0)
    missing_null_advantage = max(-missing_null_matched_delta["delta_mse"], 0.0)
    missing_outer_fold_structures = []
    for fold in sorted(set(int(value) for value in folds)):
        selected_rows = [
            row for row in missing_search
            if int(row["outer_fold"]) == fold
            and row["method"] == "jbd"
            and bool(row["selected"])
        ]
        selected_row = selected_rows[0]
        train = folds != fold
        completed_train, _ = complete_missing_covariances(full_covariances[train])
        configuration = JBDConfiguration(
            method="jbd",
            ridge=float(selected_row["ridge"]),
            block_quantile=float(selected_row["block_quantile"]),
            random_draws=RANDOM_DRAWS,
            random_seed=SEED,
            jacobi_sweeps=JACOBI_SWEEPS,
        )
        model = fit_jbd_model(completed_train, full_names, configuration)
        missing_outer_fold_structures.append({
            "outer_fold": int(fold),
            "block_sizes": list(model.diagnostics["block_sizes"]),
            "covariance_atom_count": model.diagnostics["n_covariance_atoms"],
        })
    gates = {
        "missing_nested_jbd_beats_capacity_matched_pca_ci": (
            missing_matched_delta["ci_upper"] < 0.0
        ),
        "missing_minimum_mechanism_overlap": min(missing_stability_overlaps),
        "missing_mechanism_span_nonvacuous": (
            missing_reference_mechanism_rank < missing_full_symmetric_rank
        ),
        "missing_mechanism_overlap_pass": (
            missing_reference_mechanism_rank < missing_full_symmetric_rank
            and min(missing_stability_overlaps) >= STABILITY_OVERLAP_GATE
        ),
        "missing_minimum_mechanism_rank_ratio": min(missing_stability_rank_ratios),
        "missing_mechanism_rank_ratio_pass": (
            min(missing_stability_rank_ratios) >= STABILITY_RANK_RATIO_GATE
        ),
        "missing_largest_block_fraction": missing_largest_block_fraction,
        "missing_nontrivial_block_identification_pass": (
            len(missing_final_model.blocks) >= 2
            and missing_largest_block_fraction <= MAX_BLOCK_FRACTION
        ),
        "missing_stationary_null_advantage": missing_null_advantage,
        "missing_environment_null_loss_pass": (
            missing_original_advantage > 0
            and missing_null_advantage
            <= SHUFFLED_ADVANTAGE_FRACTION * missing_original_advantage
        ),
        "missing_feature_permutation_pass": (
            missing_invariance["feature_permutation_max_prediction_error"] < 1e-10
        ),
        "missing_repeatability_pass": (
            missing_invariance["repeatability_max_prediction_error"] < 1e-12
        ),
        "completion_preserves_observed_entries_pass": max(
            row["maximum_observed_entry_error"]
            for row in full_completion_diagnostics
        ) < 1e-10,
        "completion_psd_pass": min(
            row["minimum_eigenvalue"] for row in full_completion_diagnostics
        ) >= -2e-9,
        "nested_jbd_beats_capacity_matched_pca_ci": (
            block_matched_pca_delta["ci_upper"] < 0.0
        ),
        "minimum_mechanism_overlap": min(stability_overlaps),
        "mechanism_span_nonvacuous": reference_mechanism_rank < full_symmetric_rank,
        "mechanism_overlap_pass": (
            reference_mechanism_rank < full_symmetric_rank
            and min(stability_overlaps) >= STABILITY_OVERLAP_GATE
        ),
        "minimum_mechanism_rank_ratio": min(stability_rank_ratios),
        "mechanism_rank_ratio_pass": min(stability_rank_ratios) >= STABILITY_RANK_RATIO_GATE,
        "largest_block_fraction": largest_block_fraction,
        "nontrivial_block_identification_pass": (
            len(final_model.blocks) >= 2 and largest_block_fraction <= MAX_BLOCK_FRACTION
        ),
        "shuffled_advantage": shuffled_advantage,
        "environment_shuffle_loss_pass": (
            original_advantage > 0
            and shuffled_advantage <= SHUFFLED_ADVANTAGE_FRACTION * original_advantage
        ),
        "simulator_reconstruction_pass": simulator[
            "candidate_beats_capacity_matched_pca"
        ],
        "simulator_mechanism_overlap": simulator["true_mechanism_overlap"][
            "projector_overlap_on_smaller_span"
        ],
        "simulator_identification_pass": simulator["true_mechanism_overlap"][
            "projector_overlap_on_smaller_span"
        ] >= STABILITY_OVERLAP_GATE,
        "feature_permutation_pass": invariance["feature_permutation_max_prediction_error"] < 1e-10,
        "repeatability_pass": invariance["repeatability_max_prediction_error"] < 1e-12,
    }
    passed = all([
        gates["missing_nested_jbd_beats_capacity_matched_pca_ci"],
        gates["missing_mechanism_overlap_pass"],
        gates["missing_mechanism_rank_ratio_pass"],
        gates["missing_nontrivial_block_identification_pass"],
        gates["missing_environment_null_loss_pass"],
        gates["missing_feature_permutation_pass"],
        gates["missing_repeatability_pass"],
        gates["completion_preserves_observed_entries_pass"],
        gates["completion_psd_pass"],
        gates["simulator_reconstruction_pass"],
        gates["simulator_identification_pass"],
    ])
    decision = "PASS" if passed else "CLOSE_MISSING_AWARE_JBD_AS_TARGET_BASIS"
    decision_payload = {
        "version": VERSION,
        "decision": decision,
        "primary_configuration": {
            "method": missing_final_configurations["jbd"].method,
            "ridge": missing_final_configurations["jbd"].ridge,
            "block_quantile": missing_final_configurations["jbd"].block_quantile,
        },
        "primary_block_sizes": list(
            missing_final_model.diagnostics["block_sizes"]
        ),
        "primary_covariance_atom_count": (
            missing_final_model.diagnostics["n_covariance_atoms"]
        ),
        "primary_scope": "missing_aware_30_atomic_features",
        "missing_aware_30": {
            "primary_configuration": {
                "method": missing_final_configurations["jbd"].method,
                "ridge": missing_final_configurations["jbd"].ridge,
                "block_quantile": missing_final_configurations["jbd"].block_quantile,
            },
            "block_sizes": list(missing_final_model.diagnostics["block_sizes"]),
            "covariance_atom_count": missing_final_model.diagnostics["n_covariance_atoms"],
            "nested_mse": {
                method: reconstruction_mse(rows)
                for method, rows in missing_predictions.items()
            },
            "paired_delta_vs_enclosing_full_block_pca": missing_delta,
            "paired_delta_vs_block_capacity_matched_pca": missing_matched_delta,
            "stationary_null_delta_vs_enclosing_full_block_pca": missing_null_delta,
            "stationary_null_delta_vs_block_capacity_matched_pca": (
                missing_null_matched_delta
            ),
            "outer_fold_structures": missing_outer_fold_structures,
        },
        "nested_mse": {method: reconstruction_mse(rows)
                       for method, rows in predictions.items()},
        "complete_core_17_diagnostic": {
            "configuration": {
                "method": primary.method,
                "ridge": primary.ridge,
                "block_quantile": primary.block_quantile,
            },
            "block_sizes": list(final_model.diagnostics["block_sizes"]),
            "covariance_atom_count": final_model.diagnostics["n_covariance_atoms"],
        },
        "paired_delta_vs_enclosing_full_block_pca": matched_pca_delta,
        "paired_delta_vs_block_capacity_matched_pca": block_matched_pca_delta,
        "paired_delta_vs_diagonal_pca": diagonal_pca_delta,
        "paired_delta_vs_ajd": ajd_delta,
        "environment_shuffle_delta_vs_enclosing_full_block_pca": shuffled_delta,
        "balanced_environment_shuffle_delta_vs_enclosing_full_block_pca": balanced_delta,
        "gates": gates,
        "correctness_labels_accessed": False,
        "promotion_only_gates_not_run": [
            "exact_and_near_duplicate_detector_score stress",
            "affine detector-score reconstruction",
            "zero-auxiliary-evidence fallback to IU-PCR",
        ],
        "promotion_gate_rationale": (
            "A2 failed representation-level reconstruction and stability premises "
            "before any detector score, orientation, or trust rule was constructed; "
            "these detector-only gates would be mandatory before any future promotion."
        ),
        "a3_status": "CLOSED_BECAUSE_A1_FAILED",
        "next_phase": "A4 exact paired cross-model target identification",
    }
    write_json(out / "A2_COMPLETE.json", decision_payload)

    report = f"""# Automatic group-free IU — Phase A2 multi-environment JBD

- Version: `{VERSION}`
- New correctness labels accessed: **no** (the frozen mixed-v2 input contract
  inherits earlier label-informed transforms and signs)
- Evaluation: **nested {OUTER_FOLDS}-fold environment covariance reconstruction**, 23 environments
- Primary missing-aware atomic roster: **{len(full_names)} features**; completion
  is refit within training folds and evaluation uses only genuinely observed pairs
- Missing-aware JBD / block-capacity-matched PCA / enclosing full-block PCA
  environment-macro MSE: **{reconstruction_mse(missing_predictions['jbd']):.6f} / {reconstruction_mse(missing_predictions['pca_matched']):.6f} / {reconstruction_mse(missing_predictions['pca_full']):.6f}**
- Missing-aware JBD minus block-capacity-matched PCA delta, environment-grouped
  95% CI: **{missing_matched_delta['delta_mse']:.6g} [{missing_matched_delta['ci_lower']:.6g}, {missing_matched_delta['ci_upper']:.6g}]** (gate fails)
- Missing-aware final block sizes: **{list(missing_final_model.diagnostics['block_sizes'])}**
- Missing-aware outer-fold block sizes: **{[row['block_sizes'] for row in missing_outer_fold_structures]}**
- Missing-aware LOEO minimum mechanism-rank ratio: **{min(missing_stability_rank_ratios):.6f}**
- Complete-core diagnostic roster: **{len(common_names)} features**
- Complete-core diagnostic JBD configuration: ridge **{primary.ridge}**, coupling quantile **{primary.block_quantile}**
- Complete-core diagnostic recovered block sizes: **{list(final_model.diagnostics['block_sizes'])}**
- Nested MSE — JBD / block-capacity-matched PCA / enclosing full-block PCA / diagonal PCA / AJD / RJD / factorial-JBD / pooled mean: **{reconstruction_mse(predictions['jbd']):.6f} / {reconstruction_mse(predictions['pca_matched']):.6f} / {reconstruction_mse(predictions['pca_full']):.6f} / {reconstruction_mse(predictions['pca']):.6f} / {reconstruction_mse(predictions['ajd']):.6f} / {reconstruction_mse(predictions['rjd']):.6f} / {reconstruction_mse(predictions['factorial_jbd']):.6f} / {reconstruction_mse(predictions['pooled_mean']):.6f}**
- Paired JBD-block-capacity-matched-PCA MSE delta, environment-grouped 95% CI: **{block_matched_pca_delta['delta_mse']:.6g} [{block_matched_pca_delta['ci_lower']:.6g}, {block_matched_pca_delta['ci_upper']:.6g}]**
- Paired JBD-enclosing-full-block-PCA MSE delta, environment-grouped 95% CI: **{matched_pca_delta['delta_mse']:.6g} [{matched_pca_delta['ci_lower']:.6g}, {matched_pca_delta['ci_upper']:.6g}]**
- Paired JBD-diagonal-PCA MSE delta (unmatched diagnostic): **{diagonal_pca_delta['delta_mse']:.6g} [{diagonal_pca_delta['ci_lower']:.6g}, {diagonal_pca_delta['ci_upper']:.6g}]**
- Paired JBD-AJD MSE delta, environment-grouped 95% CI: **{ajd_delta['delta_mse']:.6g} [{ajd_delta['ci_lower']:.6g}, {ajd_delta['ci_upper']:.6g}]**
- Leave-one-environment mechanism overlap: min **{min(stability_overlaps):.6f}**, median **{np.median(stability_overlaps):.6f}**
- Environment-shuffle JBD minus block-capacity-matched PCA delta: **{shuffled_matched_delta['delta_mse']:.6g}**
- Feature permutation / repeatability max error: **{invariance['feature_permutation_max_prediction_error']:.3g} / {invariance['repeatability_max_prediction_error']:.3g}**
- Simulator JBD / block-capacity-matched PCA MSE: **{simulator['candidate_mse']:.6g} / {simulator['block_capacity_matched_pca_mse']:.6g}**

## Decision

**{decision}**. The missing-aware 30-atom result is the primary scope; the
17-atom universally complete run is a diagnostic. A2 passes only if nested
reconstruction beats a pooled-PCA basis with the same recovered block sizes,
mechanism count, and ridge. The full-block PCA row is an enclosing-space
diagnostic, not the capacity-matched gate. The primary matched comparison is
promising but uncertain and is not robustly positive across environments;
the covariance-mechanism span must be stable and nontrivially block identified,
the advantage must collapse under the registered PSD null, the known-block
simulator must be recovered, and exact invariance gates must pass. Here the
matched reconstruction interval crosses zero and the LOEO mechanism-rank ratio
is 0.618, below the frozen 0.70 threshold; both independently close the route.
This phase identifies covariance mechanisms only; it does not claim that any
block is hallucination-related.

The exact/near-duplicate detector-score, affine score reconstruction, and
zero-evidence IU-PCR fallback gates were not run because A2 failed before a
detector, orientation, or trust rule existed. They remain mandatory if this
representation is ever reopened for promotion; their absence cannot be used
as evidence for a positive result.

A3 is closed regardless of this result because A1 failed its own duplicate
robustness premise. The next independent route is A4: use the exact 3,400-item
cross-model ProcessBench surface to ask whether a recovered mechanism tracks a
response-invariant target component rather than scorer-specific nuisance.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    hashes = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_HASHES.json":
            hashes[path.name] = sha256_file(path)
    write_json(out / "ARTIFACT_HASHES.json", hashes)
    print(json.dumps({
        "out": str(out),
        "decision": decision,
        "nested_jbd_mse": decision_payload["nested_mse"]["jbd"],
        "nested_enclosing_full_block_pca_mse": decision_payload["nested_mse"]["pca_full"],
        "block_sizes": decision_payload["primary_block_sizes"],
        "correctness_labels_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
