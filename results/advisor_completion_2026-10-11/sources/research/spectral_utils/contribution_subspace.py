"""HARP-inspired target correction in IU-PCR contribution space.

This module deliberately does *not* add observations or replace the feature
contract.  It starts from one ordinary IU-PCR weight vector and groups its
per-feature score contributions by the repository's frozen provenance
families.  A supervised proof-of-concept may then learn a small correction in
that contribution space while remaining exactly anchored to IU-PCR at zero
correction.

For feature ``i`` in family ``g`` and IU-PCR weight ``w_i``, define

    h_g(x) = sum_{i in g} w_i f_i(x).

The family contributions sum exactly to the IU-PCR score.  On a training
partition, each ``h_g`` is standardized and residualized against the
standardized IU score ``b``.  The corrected score is

    s_delta(x) = b(x) + r(x)^T delta.

Thus ``delta=0`` is an exact ranking identity with IU-PCR.  Fitting ``delta``
with correctness labels is a supervised research instrument, not the final
label-free algorithm.  Its purpose is to test whether a low-dimensional,
fusion-internal target correction exists before trying to estimate that
correction without labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .specrage_views import FEATURE_TO_VIEW, VIEW_ORDER


EPS = 1e-12


@dataclass(frozen=True)
class ContributionSpace:
    """IU-PCR score decomposed into fixed provenance-family contributions."""

    families: tuple[str, ...]
    members: dict[str, np.ndarray]
    baseline_score: np.ndarray
    contributions: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ContributionTransform:
    """Training-only centering, scaling, and IU-residualization parameters."""

    families: tuple[str, ...]
    baseline_mean: float
    baseline_scale: float
    contribution_mean: np.ndarray
    contribution_scale: np.ndarray
    baseline_loadings: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray

    def apply(self, baseline_score, contributions):
        """Return standardized IU score and orthogonal family residuals."""
        baseline = np.asarray(baseline_score, dtype=float)
        values = np.asarray(contributions, dtype=float)
        if baseline.ndim != 1 or values.shape != (len(baseline), len(self.families)):
            raise ValueError("baseline/contribution shapes disagree with the transform")
        standardized_baseline = (
            baseline - self.baseline_mean
        ) / self.baseline_scale
        standardized_contributions = (
            values - self.contribution_mean[None, :]
        ) / self.contribution_scale[None, :]
        residuals = (
            standardized_contributions
            - standardized_baseline[:, None] * self.baseline_loadings[None, :]
        )
        residuals = (
            residuals - self.residual_mean[None, :]
        ) / self.residual_scale[None, :]
        return standardized_baseline, residuals


@dataclass(frozen=True)
class AnchoredContributionHead:
    """One supervised residual correction anchored exactly to IU-PCR."""

    transform: ContributionTransform
    delta: np.ndarray
    prior_strength: float
    diagnostics: dict = field(default_factory=dict)

    def score(self, baseline_score, contributions):
        baseline, residuals = self.transform.apply(
            baseline_score, contributions
        )
        return baseline + residuals @ self.delta


@dataclass(frozen=True)
class LeverageBalancedContributionScore:
    """Label-free IU score with a provenance-family leverage correction."""

    transform: ContributionTransform
    score: np.ndarray
    baseline_score: np.ndarray
    correction: np.ndarray
    delta: np.ndarray
    family_leverage: np.ndarray
    effective_weights: np.ndarray
    intercept: float
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LeverageBalancedIUResult:
    """End-to-end ordinary IU-PCR plus label-free leverage balancing."""

    baseline: object
    contribution_space: ContributionSpace
    balanced: LeverageBalancedContributionScore

    @property
    def score(self):
        return self.balanced.score

    @property
    def effective_weights(self):
        return self.balanced.effective_weights

    @property
    def intercept(self):
        return self.balanced.intercept


@dataclass(frozen=True)
class CardinalityBalancedContributionScore:
    """Label-free IU score corrected for provenance-family multiplicity."""

    transform: ContributionTransform
    score: np.ndarray
    baseline_score: np.ndarray
    correction: np.ndarray
    delta: np.ndarray
    family_cardinality: np.ndarray
    effective_weights: np.ndarray
    intercept: float
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CardinalityBalancedIUResult:
    """End-to-end ordinary IU-PCR plus family-cardinality balancing."""

    baseline: object
    contribution_space: ContributionSpace
    balanced: CardinalityBalancedContributionScore

    @property
    def score(self):
        return self.balanced.score

    @property
    def effective_weights(self):
        return self.balanced.effective_weights

    @property
    def intercept(self):
        return self.balanced.intercept


@dataclass(frozen=True)
class NeutralResidualModeCalibration:
    """Label-free cross-cell calibration of the neutral residual mode."""

    families: tuple[str, ...]
    direction: np.ndarray
    residual_covariance: np.ndarray
    pair_counts: np.ndarray
    eigenvalues: np.ndarray
    selected_index: int
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NeutralResidualModeScore:
    """IU score plus a frozen neutral-mode contribution correction."""

    transform: ContributionTransform
    score: np.ndarray
    baseline_score: np.ndarray
    correction: np.ndarray
    delta: np.ndarray
    calibration_direction: np.ndarray
    effective_weights: np.ndarray
    intercept: float
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NeutralResidualModeIUResult:
    """End-to-end ordinary IU-PCR plus a neutral residual mode."""

    baseline: object
    contribution_space: ContributionSpace
    neutral: NeutralResidualModeScore

    @property
    def score(self):
        return self.neutral.score

    @property
    def effective_weights(self):
        return self.neutral.effective_weights

    @property
    def intercept(self):
        return self.neutral.intercept


def _finite_indices(indices, n):
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a nonempty vector")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices contain an out-of-range sample")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("indices contain duplicates")
    return indices


def iu_family_contributions(F, feature_names, weights):
    """Decompose an IU-PCR score into fixed provenance-family contributions.

    Parameters
    ----------
    F:
        Feature-by-sample matrix.
    feature_names:
        Name for every feature row.  Every name must already belong to the
        frozen provenance registry.
    weights:
        One ordinary IU-PCR weight per feature.

    Returns
    -------
    ContributionSpace
        ``contributions`` has shape ``(samples, present_families)`` and sums
        exactly to ``baseline_score`` up to floating-point summation order.
    """
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    names = tuple(str(name) for name in feature_names)
    if F.ndim != 2 or F.shape[0] != len(names):
        raise ValueError("F and feature_names disagree")
    if weights.shape != (F.shape[0],):
        raise ValueError("weights must provide one value per feature")
    if not np.isfinite(F).all() or not np.isfinite(weights).all():
        raise ValueError("features and weights must be finite")
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))

    families = tuple(
        family for family in VIEW_ORDER
        if any(FEATURE_TO_VIEW[name] == family for name in names)
    )
    members = {
        family: np.asarray([
            index for index, name in enumerate(names)
            if FEATURE_TO_VIEW[name] == family
        ], dtype=int)
        for family in families
    }
    contributions = np.column_stack([
        weights[indices] @ F[indices] for indices in members.values()
    ])
    baseline = weights @ F
    reconstruction = contributions.sum(axis=1)
    error = float(np.max(np.abs(reconstruction - baseline)))
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(baseline)))
    ) * max(1, F.shape[0])
    if error > tolerance:
        raise RuntimeError(
            f"family contributions do not reconstruct IU-PCR: {error:.3e}"
        )
    return ContributionSpace(
        families=families,
        members=members,
        baseline_score=baseline,
        contributions=contributions,
        diagnostics={
            "n_features": int(F.shape[0]),
            "n_samples": int(F.shape[1]),
            "n_families": int(len(families)),
            "reconstruction_error": error,
        },
    )


def fit_contribution_transform(space, training_indices):
    """Fit the contribution coordinate system without reading labels."""
    baseline = np.asarray(space.baseline_score, dtype=float)
    values = np.asarray(space.contributions, dtype=float)
    indices = _finite_indices(training_indices, len(baseline))

    baseline_mean = float(np.mean(baseline[indices]))
    baseline_scale = float(np.std(baseline[indices]))
    if baseline_scale <= EPS:
        raise ValueError("IU-PCR training score is constant")
    b = (baseline - baseline_mean) / baseline_scale

    contribution_mean = np.mean(values[indices], axis=0)
    contribution_scale = np.std(values[indices], axis=0)
    contribution_scale = np.where(contribution_scale > EPS, contribution_scale, 1.0)
    standardized = (
        values - contribution_mean[None, :]
    ) / contribution_scale[None, :]
    denominator = float(np.dot(b[indices], b[indices]))
    loadings = (
        b[indices] @ standardized[indices]
    ) / max(denominator, EPS)
    residuals = standardized - b[:, None] * loadings[None, :]
    residual_mean = np.mean(residuals[indices], axis=0)
    residual_scale = np.std(residuals[indices], axis=0)
    residual_scale = np.where(residual_scale > EPS, residual_scale, 1.0)

    return ContributionTransform(
        families=space.families,
        baseline_mean=baseline_mean,
        baseline_scale=baseline_scale,
        contribution_mean=contribution_mean,
        contribution_scale=contribution_scale,
        baseline_loadings=loadings,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
    )


def fit_anchored_contribution_head(
    space,
    labels,
    training_indices,
    *,
    transform_indices=None,
    prior_strength=0.3,
    class_balanced=True,
):
    """Fit a supervised family correction while keeping IU-PCR as the prior.

    ``training_indices`` identifies rows whose labels may be read.
    ``transform_indices`` may additionally include unlabeled training rows used
    to define the contribution coordinate system, but it must contain every
    labelled row.  Evaluation rows must be excluded by the caller.

    The empirical logistic loss is normalized to total weight one, making the
    fixed ``prior_strength`` comparable across training-set sizes.  With no
    fitted correction the score is the standardized IU-PCR score exactly.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (len(space.baseline_score),) or not np.all(
        np.isin(labels, (0, 1))
    ):
        raise ValueError("labels must be a binary vector over every sample")
    indices = _finite_indices(training_indices, len(labels))
    if transform_indices is None:
        transform_indices = indices
    transform_indices = _finite_indices(transform_indices, len(labels))
    if not np.all(np.isin(indices, transform_indices)):
        raise ValueError("transform_indices must contain every labelled row")
    if len(np.unique(labels[indices])) < 2:
        raise ValueError("training labels contain only one class")
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior_strength must be finite and positive")

    transform = fit_contribution_transform(space, transform_indices)
    baseline, residuals = transform.apply(
        space.baseline_score, space.contributions
    )
    y = labels[indices].astype(float)
    X = residuals[indices]
    b = baseline[indices]
    if class_balanced:
        positive = max(int(np.sum(y == 1)), 1)
        negative = max(int(np.sum(y == 0)), 1)
        sample_weight = np.where(y == 1, 0.5 / positive, 0.5 / negative)
    else:
        sample_weight = np.full(len(y), 1.0 / len(y), dtype=float)

    def objective(delta):
        score = b + X @ delta
        probability = expit(score)
        loss = float(np.sum(
            sample_weight * (np.logaddexp(0.0, score) - y * score)
        ))
        loss += 0.5 * prior_strength * float(np.dot(delta, delta))
        gradient = X.T @ (sample_weight * (probability - y))
        gradient += prior_strength * delta
        return loss, gradient

    initial = np.zeros(X.shape[1], dtype=float)
    fitted = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": 1000,
            "maxls": 100,
            "ftol": 1e-10,
            "gtol": 1e-7,
        },
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"anchored contribution fit failed: {fitted.message}")
    return AnchoredContributionHead(
        transform=transform,
        delta=np.asarray(fitted.x, dtype=float),
        prior_strength=prior_strength,
        diagnostics={
            "success": True,
            "n_iter": int(fitted.nit),
            "objective": float(fitted.fun),
            "delta_norm": float(np.linalg.norm(fitted.x)),
            "n_training": int(len(indices)),
            "n_transform": int(len(transform_indices)),
            "training_positive_rate": float(np.mean(y)),
            "class_balanced": bool(class_balanced),
        },
    )


def _balanced_direction_components(space, weights, fit_indices, direction):
    """Apply one fixed family direction in the IU-orthogonal residual space."""
    weights = np.asarray(weights, dtype=float)
    n = len(space.baseline_score)
    if fit_indices is None:
        fit_indices = np.arange(n, dtype=int)
    indices = _finite_indices(fit_indices, n)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector")
    if any(np.any(member >= len(weights)) for member in space.members.values()):
        raise ValueError("weights do not cover every contribution member")

    family_count = len(space.families)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (family_count,) or not np.isfinite(direction).all():
        raise ValueError("direction must be finite with one value per family")

    transform = fit_contribution_transform(space, indices)
    baseline, residuals = transform.apply(
        space.baseline_score, space.contributions
    )
    raw_correction = residuals @ direction
    raw_scale = float(np.std(raw_correction[indices]))
    trust_ratio = 1.0 / family_count
    if raw_scale <= EPS or float(np.linalg.norm(direction)) <= EPS:
        delta = np.zeros(family_count, dtype=float)
        correction = np.zeros(n, dtype=float)
    else:
        delta = trust_ratio * direction / raw_scale
        correction = residuals @ delta
    score = baseline + correction

    # The entire contribution-space operation is affine in the original
    # feature matrix.  Return that equivalent weight vector so this remains an
    # IU-PCR fusion rule rather than a detector stacked after fusion.
    residual_coefficient = delta / transform.residual_scale
    baseline_coefficient = 1.0 - float(np.dot(
        residual_coefficient, transform.baseline_loadings
    ))
    effective_weights = baseline_coefficient * weights / transform.baseline_scale
    for family_index, name in enumerate(space.families):
        members = space.members[name]
        effective_weights[members] += (
            residual_coefficient[family_index]
            * weights[members]
            / transform.contribution_scale[family_index]
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
    reconstructed_score = (
        baseline_coefficient
        * np.asarray(space.baseline_score)
        / transform.baseline_scale
        + np.sum(
            np.asarray(space.contributions)
            * (
                residual_coefficient
                / transform.contribution_scale
            )[None, :],
            axis=1,
        )
        + intercept
    )
    weight_reconstruction_error = float(np.max(np.abs(
        reconstructed_score - score
    )))
    orthogonality = float(np.dot(
        baseline[indices] - np.mean(baseline[indices]),
        correction[indices] - np.mean(correction[indices]),
    ) / len(indices))
    diagnostics = {
        "n_fit": int(len(indices)),
        "n_families": int(family_count),
        "trust_ratio": float(trust_ratio),
        "raw_correction_scale": raw_scale,
        "correction_scale": float(np.std(correction[indices])),
        "baseline_correction_covariance": orthogonality,
        "delta_norm": float(np.linalg.norm(delta)),
        "zero_correction": bool(not np.any(delta)),
        "baseline_effective_coefficient": baseline_coefficient,
        "weight_reconstruction_error": weight_reconstruction_error,
    }
    return (
        transform,
        score,
        baseline,
        correction,
        delta,
        effective_weights,
        float(intercept),
        diagnostics,
    )


def fit_neutral_residual_mode_calibration(spaces):
    """Estimate one cross-cell target direction without correctness labels.

    Every contribution residual is standardized, so independent residual
    variation has unit covariance.  Eigenmodes far above one capture shared
    dependence and modes near zero capture deterministic redundancy.  The
    neutral mode is therefore the eigenvector whose eigenvalue is closest to
    one.  Its otherwise arbitrary sign is anchored to the equal-family
    direction; all underlying feature rows have already been confidence
    oriented by the frozen feature contract.

    Covariance entries are averaged only over calibration cells in which both
    families are present.  This keeps a missing family from acting like an
    observed zero residual.  The function accepts only contribution spaces and
    never receives labels.
    """
    spaces = tuple(spaces)
    if not spaces:
        raise ValueError("at least one contribution space is required")
    family_count = len(VIEW_ORDER)
    covariance_sum = np.zeros((family_count, family_count), dtype=float)
    pair_counts = np.zeros((family_count, family_count), dtype=int)
    n_samples = 0
    for space in spaces:
        transform = fit_contribution_transform(
            space, np.arange(len(space.baseline_score), dtype=int)
        )
        _, residuals = transform.apply(
            space.baseline_score, space.contributions
        )
        present = np.asarray([
            VIEW_ORDER.index(name) for name in space.families
        ], dtype=int)
        local_covariance = residuals.T @ residuals / len(residuals)
        covariance_sum[np.ix_(present, present)] += local_covariance
        pair_counts[np.ix_(present, present)] += 1
        n_samples += len(residuals)
    if np.any(pair_counts == 0):
        missing = np.argwhere(pair_counts == 0)
        raise ValueError(
            "calibration does not jointly cover every family pair: "
            + ", ".join(
                f"{VIEW_ORDER[i]}/{VIEW_ORDER[j]}" for i, j in missing
            )
        )
    covariance = covariance_sum / pair_counts
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    distances = np.abs(eigenvalues - 1.0)
    selected = int(np.argmin(distances))
    direction = np.asarray(eigenvectors[:, selected], dtype=float)
    anchor_dot = float(np.sum(direction))
    if abs(anchor_dot) <= EPS:
        pivot = int(np.argmax(np.abs(direction)))
        sign = 1.0 if direction[pivot] >= 0 else -1.0
    else:
        sign = 1.0 if anchor_dot > 0 else -1.0
    direction *= sign
    alternatives = np.delete(distances, selected)
    return NeutralResidualModeCalibration(
        families=tuple(VIEW_ORDER),
        direction=direction,
        residual_covariance=covariance,
        pair_counts=pair_counts,
        eigenvalues=eigenvalues,
        selected_index=selected,
        diagnostics={
            "n_calibration_cells": int(len(spaces)),
            "n_calibration_samples": int(n_samples),
            "selected_eigenvalue": float(eigenvalues[selected]),
            "distance_from_unit": float(distances[selected]),
            "unit_distance_gap": float(
                np.min(alternatives) - distances[selected]
            ),
            "orientation_anchor_dot": float(np.sum(direction)),
            "direction_norm": float(np.linalg.norm(direction)),
            "uses_labels": False,
        },
    )


def neutral_residual_mode_score(
    space,
    weights,
    calibration,
    fit_indices=None,
):
    """Apply a frozen neutral residual mode inside IU contribution fusion."""
    if tuple(calibration.families) != tuple(VIEW_ORDER):
        raise ValueError("calibration family order does not match VIEW_ORDER")
    global_direction = np.asarray(calibration.direction, dtype=float)
    if global_direction.shape != (len(VIEW_ORDER),):
        raise ValueError("calibration direction has the wrong shape")
    direction = np.asarray([
        global_direction[VIEW_ORDER.index(name)] for name in space.families
    ], dtype=float)
    (
        transform,
        score,
        baseline,
        correction,
        delta,
        effective_weights,
        intercept,
        diagnostics,
    ) = _balanced_direction_components(
        space, weights, fit_indices, direction
    )
    diagnostics = {
        **diagnostics,
        "calibration_selected_eigenvalue": float(
            calibration.eigenvalues[calibration.selected_index]
        ),
        "calibration_direction_norm": float(
            np.linalg.norm(global_direction)
        ),
        "uses_labels": False,
    }
    return NeutralResidualModeScore(
        transform=transform,
        score=score,
        baseline_score=baseline,
        correction=correction,
        delta=delta,
        calibration_direction=global_direction,
        effective_weights=effective_weights,
        intercept=intercept,
        diagnostics=diagnostics,
    )


def leverage_balanced_contribution_score(
    space,
    weights,
    fit_indices=None,
):
    """Build a label-free, family-balanced correction to ordinary IU-PCR.

    IU-PCR can assign disproportionate total weight to a provenance family
    simply because that family contains several correlated measurements.  For
    family ``g`` define its observable leverage as

    ``ell_g = sum_{i in g} abs(w_i)``.

    The centered log-ratio direction

    ``d_g = mean_h(log ell_h) - log ell_g``

    moves away from high-leverage families and toward low-leverage families.
    It is invariant to a common rescaling of the IU weights.  The direction is
    applied only in the contribution residual space, which is orthogonal to
    the standardized IU score on ``fit_indices``.  Its sample standard
    deviation is limited to ``1 / G`` of the standardized baseline, where
    ``G`` is the number of present provenance families.  Thus the trust scale
    is determined by the observable family count rather than correctness
    labels or a tuned regularization path.

    This function reads no labels and uses no quantity outside the existing
    IU-PCR fusion calculation.  Fitting on all rows is a transductive,
    label-free use matching the ordinary IU-PCR batch setting.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite vector")
    if any(np.any(member >= len(weights)) for member in space.members.values()):
        raise ValueError("weights do not cover every contribution member")
    leverage = np.asarray([
        np.sum(np.abs(weights[space.members[name]]))
        for name in space.families
    ], dtype=float)
    if not np.isfinite(leverage).all() or np.any(leverage < 0):
        raise ValueError("family leverage is not finite and nonnegative")

    reference = max(float(np.max(leverage)), EPS)
    safe_leverage = np.maximum(leverage, EPS * reference)
    log_leverage = np.log(safe_leverage)
    direction = np.mean(log_leverage) - log_leverage
    (
        transform,
        score,
        baseline,
        correction,
        delta,
        effective_weights,
        intercept,
        diagnostics,
    ) = _balanced_direction_components(
        space, weights, fit_indices, direction
    )

    return LeverageBalancedContributionScore(
        transform=transform,
        score=score,
        baseline_score=baseline,
        correction=correction,
        delta=delta,
        family_leverage=leverage,
        effective_weights=effective_weights,
        intercept=intercept,
        diagnostics=diagnostics,
    )


def cardinality_balanced_contribution_score(
    space,
    weights,
    fit_indices=None,
):
    """Correct IU-PCR for feature-count multiplicity across families.

    For family ``g``, let ``m_g`` be its number of registered measurements.
    The fixed direction

    ``d_g = mean_h(log m_h) - log m_g``

    downweights families that received more coordinates merely because the
    feature engineering produced more variants from that measurement source.
    As in leverage balancing, the direction is applied only after removing the
    standardized IU score from every family contribution, and the correction
    scale is fixed to ``1 / G``.  Equal family sizes return ordinary IU-PCR
    exactly.  No correctness labels are read.
    """
    weights = np.asarray(weights, dtype=float)
    cardinality = np.asarray([
        len(space.members[name]) for name in space.families
    ], dtype=float)
    if np.any(cardinality <= 0):
        raise ValueError("every present family must contain a feature")
    log_cardinality = np.log(cardinality)
    direction = np.mean(log_cardinality) - log_cardinality
    (
        transform,
        score,
        baseline,
        correction,
        delta,
        effective_weights,
        intercept,
        diagnostics,
    ) = _balanced_direction_components(
        space, weights, fit_indices, direction
    )
    return CardinalityBalancedContributionScore(
        transform=transform,
        score=score,
        baseline_score=baseline,
        correction=correction,
        delta=delta,
        family_cardinality=cardinality,
        effective_weights=effective_weights,
        intercept=intercept,
        diagnostics=diagnostics,
    )


def leverage_balanced_iu_fit(
    F,
    feature_names,
    fit_indices=None,
    *,
    iu_kwargs=None,
):
    """Fit ordinary IU-PCR and return its leverage-balanced score and weights.

    By default this uses the repository's fixed two-component IU configuration
    used by DUFS-LIU.  ``iu_kwargs`` exists for controlled research ablations;
    changing it changes the registered v1 method.
    """
    from .laplacian_upcr import IU_FIT_DEFAULTS
    from .upcr import upcr_fit

    F = np.asarray(F, dtype=float)
    kwargs = dict(IU_FIT_DEFAULTS)
    if iu_kwargs:
        kwargs.update(iu_kwargs)
    baseline = upcr_fit(F, **kwargs)
    space = iu_family_contributions(F, feature_names, baseline.w)
    balanced = leverage_balanced_contribution_score(
        space, baseline.w, fit_indices=fit_indices
    )
    return LeverageBalancedIUResult(
        baseline=baseline,
        contribution_space=space,
        balanced=balanced,
    )


def cardinality_balanced_iu_fit(
    F,
    feature_names,
    fit_indices=None,
    *,
    iu_kwargs=None,
):
    """Fit ordinary IU-PCR and return cardinality-balanced score and weights."""
    from .laplacian_upcr import IU_FIT_DEFAULTS
    from .upcr import upcr_fit

    F = np.asarray(F, dtype=float)
    kwargs = dict(IU_FIT_DEFAULTS)
    if iu_kwargs:
        kwargs.update(iu_kwargs)
    baseline = upcr_fit(F, **kwargs)
    space = iu_family_contributions(F, feature_names, baseline.w)
    balanced = cardinality_balanced_contribution_score(
        space, baseline.w, fit_indices=fit_indices
    )
    return CardinalityBalancedIUResult(
        baseline=baseline,
        contribution_space=space,
        balanced=balanced,
    )


def neutral_residual_mode_iu_fit(
    F,
    feature_names,
    calibration,
    fit_indices=None,
    *,
    iu_kwargs=None,
):
    """Fit IU-PCR and apply one frozen label-free neutral-mode calibration."""
    from .laplacian_upcr import IU_FIT_DEFAULTS
    from .upcr import upcr_fit

    F = np.asarray(F, dtype=float)
    kwargs = dict(IU_FIT_DEFAULTS)
    if iu_kwargs:
        kwargs.update(iu_kwargs)
    baseline = upcr_fit(F, **kwargs)
    space = iu_family_contributions(F, feature_names, baseline.w)
    neutral = neutral_residual_mode_score(
        space, baseline.w, calibration, fit_indices=fit_indices
    )
    return NeutralResidualModeIUResult(
        baseline=baseline,
        contribution_space=space,
        neutral=neutral,
    )


__all__ = [
    "AnchoredContributionHead",
    "CardinalityBalancedContributionScore",
    "CardinalityBalancedIUResult",
    "ContributionSpace",
    "ContributionTransform",
    "LeverageBalancedContributionScore",
    "LeverageBalancedIUResult",
    "NeutralResidualModeCalibration",
    "NeutralResidualModeIUResult",
    "NeutralResidualModeScore",
    "fit_anchored_contribution_head",
    "fit_contribution_transform",
    "fit_neutral_residual_mode_calibration",
    "iu_family_contributions",
    "cardinality_balanced_contribution_score",
    "cardinality_balanced_iu_fit",
    "leverage_balanced_contribution_score",
    "leverage_balanced_iu_fit",
    "neutral_residual_mode_iu_fit",
    "neutral_residual_mode_score",
]
