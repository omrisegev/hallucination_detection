"""Label-free atomic neutral-residual correction for IU-PCR.

The frozen family NRM groups feature contributions by provenance before it
estimates a residual covariance.  This module keeps the same affine
contribution-space construction but treats every feature contribution as an
atom.  Cross-cell alignment uses feature identity only; feature order and
provenance-family names are irrelevant.

The calibration API deliberately accepts no correctness labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contribution_subspace import (
    ContributionSpace,
    fit_contribution_transform,
)


EPS = 1e-12
DEFAULT_NULL_DRAWS = 1000
DEFAULT_NULL_ALPHA = 0.05
DEFAULT_NULL_SEED = 20260813


@dataclass(frozen=True)
class AtomicNeutralCalibration:
    """One atomic, label-free neutral-subspace calibration."""

    feature_names: tuple[str, ...]
    direction: np.ndarray
    anchor: np.ndarray
    residual_covariance: np.ndarray
    pair_counts: np.ndarray
    eigenvalues: np.ndarray
    neutral_mask: np.ndarray
    null_lower: float
    null_upper: float
    null_draws: int
    null_alpha: float
    null_seed: int
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicNeutralScore:
    """Affine IU-PCR score plus a frozen atomic residual correction."""

    transform: object
    score: np.ndarray
    baseline_score: np.ndarray
    correction: np.ndarray
    delta: np.ndarray
    effective_weights: np.ndarray
    intercept: float
    diagnostics: dict = field(default_factory=dict)


def atomic_contribution_space(F, feature_names, weights):
    """Decompose an IU-PCR score into one contribution per feature."""
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if F.ndim != 2 or F.shape[0] != len(names):
        raise ValueError("F and feature_names disagree")
    if weights.shape != (F.shape[0],):
        raise ValueError("weights must provide one value per feature")
    if len(set(names)) != len(names):
        raise ValueError("atomic feature names must be unique")
    if not np.isfinite(F).all() or not np.isfinite(weights).all():
        raise ValueError("features and weights must be finite")
    contributions = (weights[:, None] * F).T
    baseline = weights @ F
    error = float(np.max(np.abs(contributions.sum(axis=1) - baseline)))
    tolerance = (
        64.0 * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(baseline))))
        * max(1, F.shape[0])
    )
    if error > tolerance:
        raise RuntimeError(
            f"atomic contributions do not reconstruct IU-PCR: {error:.3e}"
        )
    return ContributionSpace(
        families=names,
        members={name: np.asarray([index], dtype=int)
                 for index, name in enumerate(names)},
        baseline_score=baseline,
        contributions=contributions,
        diagnostics={
            "n_features": int(F.shape[0]),
            "n_samples": int(F.shape[1]),
            "n_atoms": int(F.shape[0]),
            "reconstruction_error": error,
        },
    )


def _subset_space(space, feature_names):
    feature_names = tuple(feature_names)
    lookup = {name: index for index, name in enumerate(space.families)}
    missing = [name for name in feature_names if name not in lookup]
    if missing:
        raise ValueError(
            "atomic space is missing frozen feature(s): " + ", ".join(missing)
        )
    columns = np.asarray([lookup[name] for name in feature_names], dtype=int)
    return ContributionSpace(
        families=feature_names,
        members={name: np.asarray(space.members[name], dtype=int)
                 for name in feature_names},
        baseline_score=np.asarray(space.baseline_score, dtype=float),
        contributions=np.asarray(space.contributions, dtype=float)[:, columns],
        diagnostics={**space.diagnostics, "atomic_subset": True},
    )


def _all_indices(space):
    return np.arange(len(space.baseline_score), dtype=int)


def _residuals(space):
    transform = fit_contribution_transform(space, _all_indices(space))
    _, values = transform.apply(space.baseline_score, space.contributions)
    return values


def _pairwise_covariance(residual_records, feature_names, permutations=None):
    p = len(feature_names)
    lookup = {name: index for index, name in enumerate(feature_names)}
    covariance_sum = np.zeros((p, p), dtype=float)
    pair_counts = np.zeros((p, p), dtype=int)
    for record_index, (names, residuals) in enumerate(residual_records):
        name_to_column = {name: index for index, name in enumerate(names)}
        present_names = tuple(name for name in feature_names if name in name_to_column)
        local = [lookup[name] for name in present_names]
        columns = [name_to_column[name] for name in present_names]
        values = np.asarray(residuals[:, columns], dtype=float)
        if permutations is not None:
            rng = permutations[record_index]
            values = np.column_stack([
                values[rng.permutation(len(values)), column]
                for column in range(values.shape[1])
            ])
        local_covariance = values.T @ values / len(values)
        covariance_sum[np.ix_(local, local)] += local_covariance
        pair_counts[np.ix_(local, local)] += 1
    if np.any(pair_counts == 0):
        missing = np.argwhere(pair_counts == 0)
        raise ValueError(
            "calibration does not jointly cover every atomic pair: "
            + ", ".join(
                f"{feature_names[i]}/{feature_names[j]}" for i, j in missing
            )
        )
    covariance = covariance_sum / pair_counts
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, pair_counts


def fit_atomic_neutral_calibration(
    spaces,
    *,
    minimum_cell_fraction=1.0,
    feature_names=None,
    null_draws=DEFAULT_NULL_DRAWS,
    null_alpha=DEFAULT_NULL_ALPHA,
    null_seed=DEFAULT_NULL_SEED,
):
    """Fit a neutral projector from atomic residual covariance, without labels.

    Eligible atoms are those present in at least ``minimum_cell_fraction`` of
    source cells.  The v1 audit freezes this at 1.0, which makes the covariance
    an ordinary average of positive-semidefinite correlation matrices and
    avoids target-time missing-coordinate decisions.

    Independent within-cell column permutations define a simultaneous null
    interval from the lower tail of the smallest eigenvalue and upper tail of
    the largest eigenvalue.  All observed eigenmodes inside that interval are
    retained.  To avoid choosing an arbitrary basis vector inside a potentially
    repeated eigenspace, the retained projector is applied to a symmetric
    inverse-absolute-dependence anchor.
    """
    spaces = tuple(spaces)
    if not spaces:
        raise ValueError("at least one atomic contribution space is required")
    if not 0 < float(minimum_cell_fraction) <= 1:
        raise ValueError("minimum_cell_fraction must lie in (0, 1]")
    if int(null_draws) < 1:
        raise ValueError("null_draws must be positive")
    if not 0 < float(null_alpha) < 1:
        raise ValueError("null_alpha must lie in (0, 1)")

    presence = {}
    residual_records = []
    total_samples = 0
    for space in spaces:
        if len(set(space.families)) != len(space.families):
            raise ValueError("atomic feature names must be unique within a cell")
        for name in space.families:
            presence[name] = presence.get(name, 0) + 1
        values = _residuals(space)
        active = np.std(values, axis=0) > EPS
        names = tuple(name for name, keep in zip(space.families, active) if keep)
        values = values[:, active]
        residual_records.append((names, values))
        total_samples += len(values)

    required = int(np.ceil(float(minimum_cell_fraction) * len(spaces) - EPS))
    active_presence = {
        name: sum(name in names for names, _ in residual_records)
        for name in presence
    }
    if feature_names is None:
        feature_names = tuple(sorted(
            name for name, count in active_presence.items() if count >= required
        ))
        eligibility_policy = "minimum_source_cell_fraction"
    else:
        feature_names = tuple(sorted(str(name) for name in feature_names))
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("frozen atomic feature names must be unique")
        missing = [
            name for name in feature_names
            if active_presence.get(name, 0) < required
        ]
        if missing:
            raise ValueError(
                "frozen atomic feature(s) fail source coverage: "
                + ", ".join(missing)
            )
        eligibility_policy = "frozen_feature_identity"
    if len(feature_names) < 3:
        raise ValueError("fewer than three atomic features meet source coverage")
    covariance, pair_counts = _pairwise_covariance(
        residual_records, feature_names
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[0]) < -1e-8:
        raise ValueError("pairwise atomic covariance is not positive semidefinite")

    root = np.random.SeedSequence(int(null_seed))
    draw_sequences = root.spawn(int(null_draws))
    null_minimum = np.empty(int(null_draws), dtype=float)
    null_maximum = np.empty(int(null_draws), dtype=float)
    for draw, sequence in enumerate(draw_sequences):
        cell_rngs = [np.random.default_rng(child)
                     for child in sequence.spawn(len(residual_records))]
        null_covariance, _ = _pairwise_covariance(
            residual_records, feature_names, permutations=cell_rngs
        )
        null_eigenvalues = np.linalg.eigvalsh(null_covariance)
        null_minimum[draw] = null_eigenvalues[0]
        null_maximum[draw] = null_eigenvalues[-1]
    null_lower = float(np.quantile(null_minimum, float(null_alpha) / 2.0))
    null_upper = float(np.quantile(null_maximum, 1.0 - float(null_alpha) / 2.0))
    neutral_mask = (eigenvalues >= null_lower) & (eigenvalues <= null_upper)
    if not np.any(neutral_mask):
        raise ValueError("the permutation null retained no neutral eigenmode")
    projector = (
        eigenvectors[:, neutral_mask] @ eigenvectors[:, neutral_mask].T
    )
    inverse_dependence = 1.0 / (
        np.sum(np.abs(covariance), axis=1) + EPS
    )
    anchor = inverse_dependence / np.linalg.norm(inverse_dependence)
    direction = projector @ anchor
    retained_norm = float(np.linalg.norm(direction))
    if retained_norm <= EPS:
        raise ValueError("the symmetric anchor has no neutral-subspace component")
    direction /= retained_norm
    if float(direction @ anchor) < 0:
        direction *= -1.0

    all_names = tuple(sorted(presence))
    excluded = tuple(name for name in all_names if name not in feature_names)
    return AtomicNeutralCalibration(
        feature_names=feature_names,
        direction=direction,
        anchor=anchor,
        residual_covariance=covariance,
        pair_counts=pair_counts,
        eigenvalues=eigenvalues,
        neutral_mask=neutral_mask,
        null_lower=null_lower,
        null_upper=null_upper,
        null_draws=int(null_draws),
        null_alpha=float(null_alpha),
        null_seed=int(null_seed),
        diagnostics={
            "n_calibration_cells": int(len(spaces)),
            "n_calibration_samples": int(total_samples),
            "minimum_cell_fraction": float(minimum_cell_fraction),
            "eligibility_policy": eligibility_policy,
            "required_cell_count": int(required),
            "n_seen_features": int(len(all_names)),
            "n_eligible_features": int(len(feature_names)),
            "excluded_features": excluded,
            "neutral_dimension": int(np.sum(neutral_mask)),
            "anchor_retained_norm": retained_norm,
            "direction_norm": float(np.linalg.norm(direction)),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "uses_labels": False,
        },
    )


def atomic_neutral_score(space, weights, calibration, fit_indices=None):
    """Apply a frozen atomic neutral direction as an affine IU correction."""
    selected = _subset_space(space, calibration.feature_names)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector")
    n = len(selected.baseline_score)
    indices = _all_indices(selected) if fit_indices is None else np.asarray(
        fit_indices, dtype=int
    )
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("fit_indices must be a non-empty vector")
    if np.any(indices < 0) or np.any(indices >= n):
        raise IndexError("fit_indices are out of range")

    transform = fit_contribution_transform(selected, indices)
    baseline, residuals = transform.apply(
        selected.baseline_score, selected.contributions
    )
    direction = np.asarray(calibration.direction, dtype=float)
    if direction.shape != (len(selected.families),):
        raise ValueError("calibration direction has the wrong dimension")
    raw_correction = residuals @ direction
    raw_scale = float(np.std(raw_correction[indices]))
    # Under the retained identity-covariance null, an equal atomic average has
    # standard deviation 1/sqrt(p).  This is the fixed, label-free trust scale.
    trust_ratio = 1.0 / np.sqrt(len(selected.families))
    if raw_scale <= EPS:
        delta = np.zeros_like(direction)
        correction = np.zeros(n, dtype=float)
    else:
        delta = trust_ratio * direction / raw_scale
        correction = residuals @ delta
    score = baseline + correction

    residual_coefficient = delta / transform.residual_scale
    baseline_coefficient = 1.0 - float(np.dot(
        residual_coefficient, transform.baseline_loadings
    ))
    effective_weights = baseline_coefficient * weights / transform.baseline_scale
    for atom_index, name in enumerate(selected.families):
        members = selected.members[name]
        effective_weights[members] += (
            residual_coefficient[atom_index]
            * weights[members]
            / transform.contribution_scale[atom_index]
        )
    intercept = (
        -baseline_coefficient
        * transform.baseline_mean
        / transform.baseline_scale
        - float(np.dot(
            residual_coefficient,
            transform.contribution_mean / transform.contribution_scale,
        ))
        - float(np.dot(
            delta,
            transform.residual_mean / transform.residual_scale,
        ))
    )
    # Reconstruct through the contribution representation.  This is equivalent
    # to ``effective_weights @ F + intercept`` and does not divide by tiny w_i.
    reconstructed = (
        baseline_coefficient
        * np.asarray(selected.baseline_score)
        / transform.baseline_scale
        + np.sum(
            np.asarray(selected.contributions)
            * (residual_coefficient / transform.contribution_scale)[None, :],
            axis=1,
        )
        + intercept
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - score)))
    centered_baseline = baseline[indices] - np.mean(baseline[indices])
    centered_correction = correction[indices] - np.mean(correction[indices])
    orthogonality = float(
        centered_baseline @ centered_correction / len(indices)
    )
    return AtomicNeutralScore(
        transform=transform,
        score=score,
        baseline_score=baseline,
        correction=correction,
        delta=delta,
        effective_weights=effective_weights,
        intercept=float(intercept),
        diagnostics={
            "n_fit": int(len(indices)),
            "n_atoms": int(len(selected.families)),
            "trust_ratio": float(trust_ratio),
            "raw_correction_scale": raw_scale,
            "correction_scale": float(np.std(correction[indices])),
            "baseline_correction_covariance": orthogonality,
            "weight_reconstruction_error": reconstruction_error,
            "neutral_dimension": int(np.sum(calibration.neutral_mask)),
            "uses_labels": False,
        },
    )


__all__ = [
    "AtomicNeutralCalibration",
    "AtomicNeutralScore",
    "DEFAULT_NULL_ALPHA",
    "DEFAULT_NULL_DRAWS",
    "DEFAULT_NULL_SEED",
    "atomic_contribution_space",
    "atomic_neutral_score",
    "fit_atomic_neutral_calibration",
]
