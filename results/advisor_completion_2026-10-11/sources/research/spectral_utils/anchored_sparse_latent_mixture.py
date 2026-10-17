"""IU-anchored sparse equal-covariance latent mixtures for Phase A5.

This module contains only target-free numerical machinery.  It accepts dense
sample-by-feature arrays and never accepts labels, records, prompts, or raw
cache mappings.  The deployable score is always affine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Iterable
import warnings

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning


EPS = 1e-12
GRAPH_EDGE_THRESHOLD = 1e-2


def _matrix(values: np.ndarray, *, name: str = "X") -> np.ndarray:
    if isinstance(values, Mapping):
        raise TypeError(f"{name} must be a numeric array, not a mapping")
    output = np.asarray(values, dtype=float)
    if output.ndim != 2 or min(output.shape) < 2:
        raise ValueError(f"{name} must have shape (n>=2, p>=2)")
    if not np.isfinite(output).all():
        raise ValueError(f"{name} contains non-finite values")
    return output


def _vector(values: np.ndarray, p: int, *, name: str) -> np.ndarray:
    if isinstance(values, Mapping):
        raise TypeError(f"{name} must be a numeric array, not a mapping")
    output = np.asarray(values, dtype=float)
    if output.shape != (p,) or not np.isfinite(output).all():
        raise ValueError(f"{name} must be a finite vector of length {p}")
    return output


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    scale: np.ndarray
    signs: np.ndarray

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = _matrix(raw, name="raw")
        if raw.shape[1] != len(self.mean):
            raise ValueError("raw matrix has the wrong feature count")
        return (raw * self.signs - self.mean) / self.scale

    def fold_affine(self, weight: np.ndarray, intercept: float = 0.0):
        weight = _vector(weight, len(self.mean), name="weight")
        raw_weight = self.signs * weight / self.scale
        raw_intercept = float(intercept - np.sum(weight * self.mean / self.scale))
        return raw_weight, raw_intercept


def fit_standardization(raw: np.ndarray, signs: np.ndarray | None = None):
    raw = _matrix(raw, name="raw")
    p = raw.shape[1]
    if signs is None:
        signs = np.ones(p)
    signs = _vector(signs, p, name="signs")
    oriented = raw * signs
    mean = oriented.mean(axis=0)
    scale = oriented.std(axis=0)
    scale = np.where(scale < EPS, 1.0, scale)
    fitted = Standardization(mean=mean, scale=scale, signs=signs)
    return fitted, fitted.transform(raw)


def full_support(p: int) -> np.ndarray:
    output = np.ones((int(p), int(p)), dtype=bool)
    return output


def diagonal_support(p: int) -> np.ndarray:
    return np.eye(int(p), dtype=bool)


def validate_support(support: np.ndarray, p: int) -> np.ndarray:
    support = np.asarray(support, dtype=bool)
    if support.shape != (p, p):
        raise ValueError("support has the wrong shape")
    if not np.array_equal(support, support.T):
        raise ValueError("support must be symmetric")
    support = support.copy()
    np.fill_diagonal(support, True)
    return support


@dataclass(frozen=True)
class PrecisionFit:
    precision: np.ndarray
    covariance: np.ndarray
    support: np.ndarray
    converged: bool
    objective: float
    relative_gradient: float
    minimum_eigenvalue: float
    iterations: int
    message: str


@dataclass(frozen=True)
class GraphicalSupportFit:
    support: np.ndarray
    partial_correlation: np.ndarray
    converged: bool
    iterations: int
    final_cost: float
    final_dual_gap: float
    warning_messages: tuple[str, ...]


def _precision_parameterization(support: np.ndarray):
    p = support.shape[0]
    edges = [(i, j) for i in range(p) for j in range(i + 1, p) if support[i, j]]
    basis = []
    for i in range(p):
        value = np.zeros((p, p), dtype=float)
        value[i, i] = 1.0
        basis.append(value)
    for i, j in edges:
        value = np.zeros((p, p), dtype=float)
        value[i, j] = value[j, i] = 1.0
        basis.append(value)
    basis = np.asarray(basis)

    def unpack(theta: np.ndarray) -> np.ndarray:
        omega = np.zeros((p, p), dtype=float)
        omega[np.diag_indices(p)] = theta[:p]
        for value, (i, j) in zip(theta[p:], edges):
            omega[i, j] = omega[j, i] = value
        return omega

    return edges, basis, unpack


def fit_fixed_support_precision(
    empirical_covariance: np.ndarray,
    support: np.ndarray,
    *,
    tolerance: float = 1e-7,
    max_iter: int = 1000,
) -> PrecisionFit:
    """Fit the Gaussian precision MLE under an exact undirected support.

    The objective is convex on the positive-definite cone.  A damped Newton
    method works in the exact free-entry subspace and accepts only Cholesky-
    feasible Armijo steps, so forbidden entries remain zero and every iterate
    remains strictly positive definite.
    """
    covariance = np.asarray(empirical_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("empirical_covariance must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("empirical_covariance contains non-finite values")
    covariance = (covariance + covariance.T) / 2.0
    p = covariance.shape[0]
    support = validate_support(support, p)
    edges, basis, unpack = _precision_parameterization(support)
    diagonal = np.maximum(np.diag(covariance), 1e-6)
    theta = np.concatenate([1.0 / diagonal, np.zeros(len(edges))])

    def state(parameters: np.ndarray, *, hessian: bool):
        omega_value = unpack(parameters)
        try:
            cholesky = np.linalg.cholesky(omega_value)
        except np.linalg.LinAlgError:
            return None
        logdet = 2.0 * np.sum(np.log(np.diag(cholesky)))
        inverse = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.eye(p)))
        gradient_matrix = covariance - inverse
        gradient = np.einsum("aij,ij->a", basis, gradient_matrix)
        objective = float(np.sum(covariance * omega_value) - logdet)
        if not hessian:
            return objective, gradient, omega_value, inverse, None
        # H_ab = tr(Sigma B_a Sigma B_b), the exact Hessian in the
        # symmetric free-entry coordinates.
        transformed = np.einsum("ij,ajk->aik", inverse, basis)
        curvature = np.einsum("aij,bji->ab", transformed, transformed)
        curvature = (curvature + curvature.T) / 2.0
        return objective, gradient, omega_value, inverse, curvature

    previous_objective = None
    relative_objective_change = float("inf")
    message = "maximum iterations reached"
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        objective, gradient, omega, fitted_covariance, hessian = state(theta, hessian=True)
        relative_gradient = float(np.linalg.norm(gradient) / (1.0 + np.linalg.norm(theta)))
        if relative_gradient <= tolerance and (
            previous_objective is None or relative_objective_change <= tolerance
        ):
            if previous_objective is None:
                relative_objective_change = 0.0
            message = "free-entry KKT and objective tolerances satisfied"
            break
        # Cholesky solve is preferred; lstsq is a deterministic safety path for
        # nearly redundant legal supports.
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = -np.linalg.lstsq(hessian, gradient, rcond=1e-12)[0]
        directional_derivative = float(gradient @ direction)
        if not np.isfinite(directional_derivative) or directional_derivative >= 0:
            direction = -gradient
            directional_derivative = -float(gradient @ gradient)

        step = 1.0
        accepted = None
        for _ in range(100):
            proposal = theta + step * direction
            candidate = state(proposal, hessian=False)
            if candidate is not None and candidate[0] <= objective + 1e-4 * step * directional_derivative:
                accepted = (proposal, candidate)
                break
            step *= 0.5
        if accepted is None:
            message = "SPD-preserving Armijo line search failed"
            break
        theta, candidate = accepted
        next_objective = candidate[0]
        relative_objective_change = abs(next_objective - objective) / (1.0 + abs(objective))
        previous_objective = objective
    final = state(theta, hessian=False)
    if final is None:  # pragma: no cover - every accepted iterate is SPD
        raise RuntimeError("precision optimizer lost positive definiteness")
    value, gradient, omega, fitted_covariance, _ = final
    eigenvalues = np.linalg.eigvalsh(omega)
    minimum = float(eigenvalues[0])
    if minimum <= 0:
        raise RuntimeError("precision optimizer returned a non-SPD matrix")
    relative_gradient = float(np.linalg.norm(gradient) / (1.0 + np.linalg.norm(theta)))
    converged = bool(
        np.isfinite(value)
        and minimum > 1e-8
        and relative_gradient <= tolerance
        and relative_objective_change <= tolerance
    )
    return PrecisionFit(
        precision=omega,
        covariance=fitted_covariance,
        support=support,
        converged=converged,
        objective=float(value),
        relative_gradient=relative_gradient,
        minimum_eigenvalue=minimum,
        iterations=int(iterations),
        message=message,
    )


def gaussian_log_density(
    X: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    X = _matrix(X)
    p = X.shape[1]
    mean = _vector(mean, p, name="mean")
    precision = np.asarray(precision, dtype=float)
    if precision.shape != (p, p):
        raise ValueError("precision has the wrong shape")
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("precision must be positive definite")
    residual = X - mean
    quadratic = np.einsum("ni,ij,nj->n", residual, precision, residual)
    return 0.5 * (logdet - p * np.log(2.0 * np.pi) - quadratic)


def mixture_log_density(
    X: np.ndarray,
    centre: np.ndarray,
    delta: np.ndarray,
    precision: np.ndarray,
    prior: float,
) -> np.ndarray:
    prior = float(np.clip(prior, 0.05, 0.95))
    negative = gaussian_log_density(X, centre - delta / 2.0, precision) + np.log(1.0 - prior)
    positive = gaussian_log_density(X, centre + delta / 2.0, precision) + np.log(prior)
    return logsumexp(np.column_stack([negative, positive]), axis=1)


@dataclass(frozen=True)
class SparseMixtureFit:
    centre: np.ndarray
    delta: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    prior: float
    responsibilities: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int
    support: np.ndarray
    precision_fit: PrecisionFit
    history: tuple[float, ...] = field(default_factory=tuple)

    @property
    def discriminant(self) -> np.ndarray:
        return self.precision @ self.delta

    @property
    def effective_masses(self) -> tuple[float, float]:
        positive = float(np.sum(self.responsibilities))
        return float(len(self.responsibilities) - positive), positive


def _initial_responsibilities(X: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    score = X @ anchor
    scale = max(float(np.std(score)), EPS)
    return np.clip(expit((score - np.median(score)) / scale), 0.01, 0.99)


def fit_sparse_equal_covariance_mixture(
    X: np.ndarray,
    support: np.ndarray,
    anchor: np.ndarray,
    *,
    max_iter: int = 300,
    tolerance: float = 1e-7,
) -> SparseMixtureFit:
    """Fit a two-component Gaussian mixture with one fixed-support covariance."""
    X = _matrix(X)
    n, p = X.shape
    anchor = _vector(anchor, p, name="anchor")
    support = validate_support(support, p)
    responsibilities = _initial_responsibilities(X, anchor)
    history: list[float] = []
    previous_fit = None

    for iteration in range(int(max_iter)):
        positive_mass = max(float(np.sum(responsibilities)), EPS)
        negative_mass = max(float(n - positive_mass), EPS)
        mean_positive = np.sum(responsibilities[:, None] * X, axis=0) / positive_mass
        mean_negative = np.sum((1.0 - responsibilities)[:, None] * X, axis=0) / negative_mass
        delta = mean_positive - mean_negative
        centre = (mean_positive + mean_negative) / 2.0
        prior = float(np.clip(positive_mass / n, 0.05, 0.95))
        residual_positive = X - mean_positive
        residual_negative = X - mean_negative
        within = (
            (residual_positive * responsibilities[:, None]).T @ residual_positive
            + (residual_negative * (1.0 - responsibilities)[:, None]).T @ residual_negative
        ) / n
        within = (within + within.T) / 2.0
        within[np.diag_indices(p)] += 1e-8
        precision_fit = fit_fixed_support_precision(within, support, tolerance=tolerance)
        precision = precision_fit.precision
        covariance = precision_fit.covariance

        discriminant = precision @ delta
        if float(discriminant @ covariance @ anchor) < 0:
            delta = -delta
            prior = 1.0 - prior
        log_negative = (
            gaussian_log_density(X, centre - delta / 2.0, precision)
            + np.log(1.0 - prior)
        )
        log_positive = (
            gaussian_log_density(X, centre + delta / 2.0, precision)
            + np.log(prior)
        )
        normalization = logsumexp(np.column_stack([log_negative, log_positive]), axis=1)
        log_likelihood = float(np.mean(normalization))
        updated = np.exp(log_positive - normalization)
        history.append(log_likelihood)
        current_fit = (centre, delta, covariance, precision, prior, updated, precision_fit)

        if previous_fit is not None:
            relative = abs(history[-1] - history[-2]) / (1.0 + abs(history[-2]))
            if relative < tolerance:
                previous_fit = current_fit
                break
        responsibilities = np.clip(updated, 1e-8, 1.0 - 1e-8)
        previous_fit = current_fit

    if previous_fit is None:  # pragma: no cover - n>=2 always executes once
        raise RuntimeError("mixture did not execute")
    centre, delta, covariance, precision, prior, responsibilities, precision_fit = previous_fit
    monotone = all(b + 1e-7 >= a for a, b in zip(history, history[1:]))
    converged = bool(
        precision_fit.converged
        and len(history) < max_iter
        and monotone
        and min(np.sum(responsibilities), np.sum(1.0 - responsibilities)) >= 1.0
    )
    return SparseMixtureFit(
        centre=centre,
        delta=delta,
        covariance=covariance,
        precision=precision,
        prior=float(prior),
        responsibilities=responsibilities,
        log_likelihood=float(history[-1]),
        converged=converged,
        iterations=len(history),
        support=support,
        precision_fit=precision_fit,
        history=tuple(history),
    )


def support_from_residual_correlations(
    residual_correlations: Iterable[np.ndarray], penalty: float
) -> GraphicalSupportFit:
    matrices = [np.asarray(value, dtype=float) for value in residual_correlations]
    if not matrices:
        raise ValueError("at least one residual correlation is required")
    shape = matrices[0].shape
    if any(value.shape != shape for value in matrices):
        raise ValueError("residual correlations have inconsistent shapes")
    average = np.mean(matrices, axis=0)
    average = (average + average.T) / 2.0
    np.fill_diagonal(average, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, precision, costs, iterations = graphical_lasso(
            average, alpha=float(penalty), max_iter=5000, tol=1e-4,
            return_costs=True, return_n_iter=True,
        )
    diagonal = np.sqrt(np.maximum(np.diag(precision), EPS))
    partial = -precision / np.outer(diagonal, diagonal)
    np.fill_diagonal(partial, 1.0)
    support = np.abs(partial) > GRAPH_EDGE_THRESHOLD
    np.fill_diagonal(support, True)
    final_cost, final_dual_gap = costs[-1] if costs else (float("nan"), float("inf"))
    messages = tuple(str(value.message) for value in caught)
    converged = bool(
        not messages
        and int(iterations) < 5000
        and np.isfinite(final_cost)
        and np.isfinite(final_dual_gap)
        and abs(float(final_dual_gap)) <= 1e-4
    )
    return GraphicalSupportFit(
        support=support,
        partial_correlation=partial,
        converged=converged,
        iterations=int(iterations),
        final_cost=float(final_cost),
        final_dual_gap=float(final_dual_gap),
        warning_messages=messages,
    )


def within_component_correlation(fit: SparseMixtureFit, X: np.ndarray) -> np.ndarray:
    X = _matrix(X)
    r = fit.responsibilities
    mean_positive = fit.centre + fit.delta / 2.0
    mean_negative = fit.centre - fit.delta / 2.0
    rp = X - mean_positive
    rn = X - mean_negative
    covariance = (
        (rp * r[:, None]).T @ rp + (rn * (1.0 - r)[:, None]).T @ rn
    ) / len(X)
    scale = np.sqrt(np.maximum(np.diag(covariance), EPS))
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def anchored_direction(
    mixture: SparseMixtureFit, iu_weight: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    p = len(mixture.delta)
    iu_weight = _vector(iu_weight, p, name="iu_weight")
    covariance = mixture.covariance
    w_mix = mixture.discriminant.copy()
    orientation = float(w_mix @ covariance @ iu_weight)
    iu_norm = float(iu_weight @ covariance @ iu_weight)
    mix_norm = float(w_mix @ covariance @ w_mix)
    orientation_tolerance = 1e-10 * np.sqrt(max(iu_norm * mix_norm, 0.0))
    if (
        not np.isfinite(orientation)
        or iu_norm <= EPS
        or mix_norm <= EPS
        or abs(orientation) <= orientation_tolerance
    ):
        correction = np.zeros_like(iu_weight)
        return iu_weight.copy(), correction, {
            "orientation_inner_product": orientation,
            "degenerate_mixture_direction": True,
            "degeneracy_reason": "zero_or_unorientable_evidence",
            "orientation_tolerance": orientation_tolerance,
            "iu_correction_covariance": 0.0,
        }
    if orientation < 0:
        w_mix = -w_mix
    w_mix *= np.sqrt(iu_norm / mix_norm)
    projection = float(iu_weight @ covariance @ w_mix) / iu_norm
    correction = w_mix - projection * iu_weight
    weight = iu_weight + float(alpha) * correction
    return weight, correction, {
        "orientation_inner_product": orientation,
        "degenerate_mixture_direction": False,
        "orientation_tolerance": orientation_tolerance,
        "iu_correction_covariance": float(iu_weight @ covariance @ correction),
        "iu_norm": iu_norm,
        "mix_norm_after_scaling": float(w_mix @ covariance @ w_mix),
        "alpha": float(alpha),
    }


@dataclass(frozen=True)
class ConstrainedMixtureFit:
    centre: np.ndarray
    delta: np.ndarray
    beta: float
    prior: float
    mean_log_likelihood: float
    responsibilities: np.ndarray
    converged: bool
    iterations: int
    history: tuple[float, ...]


def fit_constrained_direction_mixture(
    X: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
    direction: np.ndarray,
    *,
    max_iter: int = 1000,
    tolerance: float = 1e-7,
) -> ConstrainedMixtureFit:
    """Fit centre/prior/nonnegative separation with fixed covariance/direction."""
    X = _matrix(X)
    n, p = X.shape
    direction = _vector(direction, p, name="direction")
    covariance = np.asarray(covariance, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if covariance.shape != (p, p) or precision.shape != (p, p):
        raise ValueError("covariance/precision has the wrong shape")
    delta_unit = covariance @ direction
    norm = float(delta_unit @ precision @ delta_unit)
    if norm <= EPS:
        delta_unit = np.zeros(p)
        direction_metric_scale = 1.0
    else:
        direction_metric_scale = np.sqrt(norm)
        delta_unit /= np.sqrt(norm)
    projected = X @ direction
    responsibilities = np.clip(
        expit((projected - np.median(projected)) / max(float(np.std(projected)), EPS)),
        0.01,
        0.99,
    )
    beta = max(float(np.std(projected)) / direction_metric_scale, 0.0)
    centre = X.mean(axis=0)
    history: list[float] = []

    for _ in range(int(max_iter)):
        t = 2.0 * responsibilities - 1.0
        centre = np.mean(X - t[:, None] * beta * delta_unit / 2.0, axis=0)
        denominator = float(n * delta_unit @ precision @ delta_unit)
        numerator = float(2.0 * np.sum(t * ((X - centre) @ precision @ delta_unit)))
        beta = max(0.0, numerator / max(denominator, EPS))
        delta = beta * delta_unit
        prior = float(np.clip(np.mean(responsibilities), 0.05, 0.95))
        log_negative = gaussian_log_density(X, centre - delta / 2.0, precision) + np.log(1-prior)
        log_positive = gaussian_log_density(X, centre + delta / 2.0, precision) + np.log(prior)
        normalizer = logsumexp(np.column_stack([log_negative, log_positive]), axis=1)
        updated = np.exp(log_positive - normalizer)
        history.append(float(np.mean(normalizer)))
        if len(history) > 1:
            relative = abs(history[-1] - history[-2]) / (1.0 + abs(history[-2]))
            if relative < tolerance:
                responsibilities = updated
                break
        responsibilities = np.clip(updated, 1e-8, 1.0 - 1e-8)
    delta = beta * delta_unit
    monotone = all(b + 1e-7 >= a for a, b in zip(history, history[1:]))
    return ConstrainedMixtureFit(
        centre=centre,
        delta=delta,
        beta=float(beta),
        prior=float(prior),
        mean_log_likelihood=float(history[-1]),
        responsibilities=responsibilities,
        converged=bool(len(history) < max_iter and monotone),
        iterations=len(history),
        history=tuple(history),
    )


def held_mean_log_likelihood(
    X: np.ndarray, fit: ConstrainedMixtureFit, precision: np.ndarray
) -> float:
    values = mixture_log_density(X, fit.centre, fit.delta, precision, fit.prior)
    return float(np.mean(values))


def posterior_log_odds(
    X: np.ndarray,
    centre: np.ndarray,
    delta: np.ndarray,
    precision: np.ndarray,
    prior: float,
) -> np.ndarray:
    """Exact affine posterior log odds for an equal-covariance mixture."""
    X = _matrix(X)
    weight = precision @ delta
    intercept = float(np.log(prior / (1.0 - prior)) - centre @ weight)
    return X @ weight + intercept


__all__ = [
    "ConstrainedMixtureFit",
    "GRAPH_EDGE_THRESHOLD",
    "GraphicalSupportFit",
    "PrecisionFit",
    "SparseMixtureFit",
    "Standardization",
    "anchored_direction",
    "diagonal_support",
    "fit_constrained_direction_mixture",
    "fit_fixed_support_precision",
    "fit_sparse_equal_covariance_mixture",
    "fit_standardization",
    "full_support",
    "gaussian_log_density",
    "held_mean_log_likelihood",
    "mixture_log_density",
    "posterior_log_odds",
    "support_from_residual_correlations",
    "validate_support",
    "within_component_correlation",
]
