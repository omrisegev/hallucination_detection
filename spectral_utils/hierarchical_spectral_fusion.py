"""Cross-cell spectral correction and label-blind active acquisition.

The shared representation keeps a cell's U-PCR score as its first coordinate
and expresses every transferable correction through stable, named features.
All transformations are fitted from the cell's unlabeled training matrix.  A
grouped logistic head can then share one coefficient vector across cells while
giving each donor cell its own nuisance intercept.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .semi_supervised_fusion import HeadResult, fit_logistic_head


__all__ = [
    "GroupedHeadResult",
    "SharedRepresentation",
    "fit_shared_representation",
    "fit_grouped_logistic_head",
    "fit_score_head",
    "fisher_d_optimal_order",
]


@dataclass
class SharedRepresentation:
    """Training-fitted map from cell features to aligned correction scores."""

    common_features: tuple
    common_indices: np.ndarray
    upcr_weight: np.ndarray
    upcr_center: float
    upcr_scale: float
    common_center: np.ndarray
    residual_projection: np.ndarray
    residual_scale: np.ndarray

    def transform(self, matrix):
        """Apply the frozen cell-specific map to standardized rows."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.upcr_weight):
            raise ValueError("matrix does not match the fitted representation")
        if not np.isfinite(matrix).all():
            raise ValueError("matrix contains non-finite values")
        upcr = (matrix @ self.upcr_weight - self.upcr_center) / self.upcr_scale
        common = matrix[:, self.common_indices] - self.common_center
        residual = common - upcr[:, None] * self.residual_projection[None, :]
        residual = residual / self.residual_scale
        return np.column_stack([upcr, residual])


@dataclass
class GroupedHeadResult:
    """Shared logistic coefficients with group-specific nuisance intercepts."""

    coefficients: np.ndarray
    group_intercepts: dict
    converged: bool
    objective: float
    n_iter: int
    meta: dict = field(default_factory=dict)


def fit_shared_representation(matrix, feature_names, common_features, upcr_weight):
    """Fit an aligned U-PCR-plus-residual representation without labels.

    Each common feature is residualized against the standardized U-PCR score.
    Residual columns are then variance-normalized inside the cell.  The named
    columns align semantics across cells while the within-cell normalization
    prevents scale differences from masquerading as transferable signal.
    """
    matrix = np.asarray(matrix, dtype=float)
    feature_names = tuple(str(name) for name in feature_names)
    common_features = tuple(str(name) for name in common_features)
    upcr_weight = np.asarray(upcr_weight, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("matrix and feature names disagree")
    if upcr_weight.shape != (matrix.shape[1],):
        raise ValueError("U-PCR weight has the wrong dimension")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be unique")
    lookup = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in common_features if name not in lookup]
    if missing:
        raise ValueError(f"missing common features: {missing}")

    indices = np.asarray([lookup[name] for name in common_features], dtype=int)
    raw_upcr = matrix @ upcr_weight
    upcr_center = float(raw_upcr.mean())
    upcr_scale = float(raw_upcr.std())
    if upcr_scale <= 1e-10:
        raise ValueError("U-PCR score is degenerate")
    upcr = (raw_upcr - upcr_center) / upcr_scale

    common = matrix[:, indices]
    common_center = common.mean(axis=0)
    centered = common - common_center
    projection = np.mean(upcr[:, None] * centered, axis=0)
    residual = centered - upcr[:, None] * projection[None, :]
    residual_scale = residual.std(axis=0)
    residual_scale = np.where(residual_scale > 1e-8, residual_scale, 1.0)

    representation = SharedRepresentation(
        common_features=common_features,
        common_indices=indices,
        upcr_weight=upcr_weight.copy(),
        upcr_center=upcr_center,
        upcr_scale=upcr_scale,
        common_center=common_center,
        residual_projection=projection,
        residual_scale=residual_scale,
    )
    transformed = representation.transform(matrix)
    if np.max(np.abs(transformed[:, 0].mean())) > 1e-10:
        raise RuntimeError("U-PCR representation is not centered")
    if np.max(np.abs(transformed[:, 1:].T @ transformed[:, 0] / len(matrix))) > 1e-8:
        raise RuntimeError("common residuals are not orthogonal to U-PCR")
    return representation


def _logistic_probability(linear):
    probability = np.empty_like(linear, dtype=float)
    positive = linear >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
    exp_linear = np.exp(linear[~positive])
    probability[~positive] = exp_linear / (1.0 + exp_linear)
    return probability


def fit_grouped_logistic_head(
    matrices,
    labels,
    group_ids,
    prior_coefficients,
    *,
    prior_strength=20.0,
    intercept_strength=1.0,
):
    """Fit shared coefficients and one regularized intercept per donor group."""
    matrix = np.asarray(matrices, dtype=float)
    labels = np.asarray(labels, dtype=float)
    group_ids = np.asarray(group_ids)
    prior = np.asarray(prior_coefficients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(prior):
        raise ValueError("matrix and coefficient prior disagree")
    if len(labels) != len(matrix) or len(group_ids) != len(matrix):
        raise ValueError("rows, labels, and groups must align")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")

    groups = tuple(sorted(set(group_ids.tolist()), key=str))
    if not groups:
        raise ValueError("at least one donor group is required")
    group_lookup = {group: index for index, group in enumerate(groups)}
    group_index = np.asarray([group_lookup[group] for group in group_ids], dtype=int)
    initial = np.concatenate([prior, np.zeros(len(groups), dtype=float)])

    def objective(parameters):
        coefficients = parameters[: matrix.shape[1]]
        intercepts = parameters[matrix.shape[1] :]
        linear = matrix @ coefficients + intercepts[group_index]
        probability = _logistic_probability(linear)
        loss = np.logaddexp(0.0, linear) - labels * linear
        difference = coefficients - prior
        value = float(loss.sum())
        value += 0.5 * float(prior_strength) * float(difference @ difference)
        value += 0.5 * float(intercept_strength) * float(intercepts @ intercepts)

        residual = probability - labels
        coefficient_gradient = matrix.T @ residual + float(prior_strength) * difference
        intercept_gradient = np.bincount(
            group_index, weights=residual, minlength=len(groups),
        ) + float(intercept_strength) * intercepts
        return value, np.concatenate([coefficient_gradient, intercept_gradient])

    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    coefficients = np.asarray(result.x[: matrix.shape[1]], dtype=float)
    intercepts = np.asarray(result.x[matrix.shape[1] :], dtype=float)
    return GroupedHeadResult(
        coefficients=coefficients,
        group_intercepts={group: float(intercepts[index]) for index, group in enumerate(groups)},
        converged=bool(result.success),
        objective=float(result.fun),
        n_iter=int(getattr(result, "nit", 0)),
        meta={"message": str(result.message), "n_groups": len(groups)},
    )


def fit_score_head(
    scores,
    labels,
    prior_coefficients,
    *,
    prior_strength=10.0,
    intercept_prior_strength=0.1,
):
    """Fit a logistic head directly over a small matrix of frozen scores."""
    scores = np.asarray(scores, dtype=float)
    prior = np.asarray(prior_coefficients, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(prior):
        raise ValueError("score matrix and prior disagree")
    return fit_logistic_head(
        scores,
        labels,
        np.eye(scores.shape[1]),
        prior,
        prior_strength=prior_strength,
        intercept_prior_strength=intercept_prior_strength,
    )


def fisher_d_optimal_order(
    scores,
    prior_coefficients,
    max_budget,
    *,
    prior_strength=10.0,
    intercept_prior_strength=0.1,
):
    """Return a nested, label-blind greedy Fisher/D-optimal acquisition order.

    Candidate information weights are frozen under the supplied prior.  The
    routine never reads correctness labels and therefore can be used before
    annotation.  Greedy determinant updates favor points near the prior's
    decision boundary while preserving diversity in the score coordinates.
    """
    scores = np.asarray(scores, dtype=float)
    prior = np.asarray(prior_coefficients, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(prior):
        raise ValueError("score matrix and prior disagree")
    if not np.isfinite(scores).all() or not np.isfinite(prior).all():
        raise ValueError("scores and prior must be finite")
    budget = min(max(0, int(max_budget)), len(scores))
    if budget == 0:
        return np.asarray([], dtype=int)

    design = np.column_stack([np.ones(len(scores)), scores])
    linear = scores @ prior
    probability = _logistic_probability(linear)
    fisher_weight = np.maximum(probability * (1.0 - probability), 1e-6)
    diagonal = np.r_[max(float(intercept_prior_strength), 1e-6),
                     np.full(scores.shape[1], max(float(prior_strength), 1e-6))]
    information = np.diag(diagonal)
    available = np.ones(len(scores), dtype=bool)
    selected = []

    for _ in range(budget):
        inverse = np.linalg.inv(information)
        leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
        gain = fisher_weight * leverage
        gain[~available] = -np.inf
        index = int(np.argmax(gain))
        selected.append(index)
        available[index] = False
        row = design[index]
        information += fisher_weight[index] * np.outer(row, row)
    return np.asarray(selected, dtype=int)
