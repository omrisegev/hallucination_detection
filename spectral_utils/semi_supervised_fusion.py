"""Label-efficient heads centred on an unlabeled U-PCR score.

The module separates two sources of information:

* an unlabeled feature matrix estimates U-PCR and a low-dimensional covariance
  basis;
* a small trusted subset estimates only the coefficients in that basis.

Correctness labels never enter ``spectral_score_basis``.  The fitted head is a
linear score so its effective feature weights remain inspectable.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


__all__ = [
    "HeadResult",
    "standardize_train_test",
    "orient_weight",
    "spectral_score_basis",
    "pca_score_basis",
    "fit_logistic_head",
    "fit_soft_logistic_head",
]


@dataclass
class HeadResult:
    """A fitted linear head over a supplied feature-space basis."""

    weight: np.ndarray
    intercept: float
    coefficients: np.ndarray
    converged: bool
    objective: float
    n_iter: int
    meta: dict = field(default_factory=dict)


def standardize_train_test(train, test):
    """Fit column standardisation on ``train`` and freeze it for ``test``."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train/test must be finite matrices with matching columns")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("train/test contain non-finite values")
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (train - center) / scale, (test - center) / scale, center, scale


def orient_weight(weight, matrix, anchor):
    """Resolve a weight vector's global sign against an unlabeled anchor."""
    weight = np.asarray(weight, dtype=float).copy()
    matrix = np.asarray(matrix, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    score = matrix @ weight
    if score.std() < 1e-12 or anchor.std() < 1e-12:
        raise ValueError("weight score and anchor must be non-constant")
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("global orientation correlation is non-finite")
    return -weight if correlation < 0 else weight


def _metric_orthonormalize(matrix, candidates, max_rank):
    """Orthonormalise feature vectors in the training-score covariance metric."""
    matrix = np.asarray(matrix, dtype=float)
    basis = []
    score_basis = []
    for candidate in candidates:
        vector = np.asarray(candidate, dtype=float).copy()
        score = matrix @ vector
        score = score - score.mean()
        for previous_vector, previous_score in zip(basis, score_basis):
            coefficient = float(np.mean(score * previous_score))
            vector = vector - coefficient * previous_vector
            score = score - coefficient * previous_score
        norm = float(np.sqrt(np.mean(score ** 2)))
        if norm <= 1e-9:
            continue
        vector /= norm
        score /= norm
        basis.append(vector)
        score_basis.append(score)
        if len(basis) >= int(max_rank):
            break
    if not basis:
        raise ValueError("no non-degenerate score direction")
    return np.column_stack(basis)


def spectral_score_basis(matrix, upcr_weight, rank=6):
    """Return an unlabeled score basis beginning exactly with the U-PCR score.

    The remaining candidates are feature-covariance eigenvectors.  The returned
    columns satisfy ``B.T @ C @ B ~= I`` for training covariance ``C``.
    """
    matrix = np.asarray(matrix, dtype=float)
    upcr_weight = np.asarray(upcr_weight, dtype=float)
    if matrix.ndim != 2 or upcr_weight.shape != (matrix.shape[1],):
        raise ValueError("matrix/upcr_weight dimensions disagree")
    covariance = (matrix.T @ matrix) / len(matrix)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    candidates = [upcr_weight] + [vectors[:, j] for j in order]
    return _metric_orthonormalize(matrix, candidates, min(rank, matrix.shape[1]))


def pca_score_basis(matrix, rank=6):
    """Return leading PCA directions standardised in training-score variance."""
    matrix = np.asarray(matrix, dtype=float)
    covariance = (matrix.T @ matrix) / len(matrix)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    candidates = [vectors[:, j] for j in order]
    return _metric_orthonormalize(matrix, candidates, min(rank, matrix.shape[1]))


def _logistic_terms(linear, target):
    loss = np.logaddexp(0.0, linear) - target * linear
    probability = np.empty_like(linear)
    positive = linear >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
    exp_linear = np.exp(linear[~positive])
    probability[~positive] = exp_linear / (1.0 + exp_linear)
    return loss, probability


def _fit_head(
    labelled_scores,
    labels,
    basis,
    prior_coefficients,
    *,
    prior_strength,
    intercept_prior_strength,
    soft_scores=None,
    soft_targets=None,
    soft_total_weight=0.0,
):
    labelled_scores = np.asarray(labelled_scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    basis = np.asarray(basis, dtype=float)
    prior = np.asarray(prior_coefficients, dtype=float)
    if labelled_scores.ndim != 2 or labelled_scores.shape[1] != len(prior):
        raise ValueError("labelled score matrix and prior dimensions disagree")
    if len(labels) != len(labelled_scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary and aligned")

    initial = np.concatenate([[0.0], prior])

    def objective(parameters):
        intercept, coefficients = parameters[0], parameters[1:]
        linear = intercept + labelled_scores @ coefficients
        labelled_loss, labelled_probability = _logistic_terms(linear, labels)
        value = float(labelled_loss.sum())
        gradient_intercept = float(np.sum(labelled_probability - labels))
        gradient_coefficients = labelled_scores.T @ (labelled_probability - labels)

        if soft_scores is not None and float(soft_total_weight) > 0:
            soft_linear = intercept + soft_scores @ coefficients
            soft_loss, soft_probability = _logistic_terms(soft_linear, soft_targets)
            scale = float(soft_total_weight) / max(1, len(soft_scores))
            value += scale * float(soft_loss.sum())
            soft_residual = soft_probability - soft_targets
            gradient_intercept += scale * float(np.sum(soft_residual))
            gradient_coefficients += scale * (soft_scores.T @ soft_residual)

        difference = coefficients - prior
        value += 0.5 * float(prior_strength) * float(difference @ difference)
        value += 0.5 * float(intercept_prior_strength) * float(intercept ** 2)
        gradient_coefficients += float(prior_strength) * difference
        gradient_intercept += float(intercept_prior_strength) * intercept
        gradient = np.concatenate([[gradient_intercept], gradient_coefficients])
        return value, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    coefficients = np.asarray(result.x[1:], dtype=float)
    return HeadResult(
        weight=basis @ coefficients,
        intercept=float(result.x[0]),
        coefficients=coefficients,
        converged=bool(result.success),
        objective=float(result.fun),
        n_iter=int(getattr(result, "nit", 0)),
        meta={"message": str(result.message)},
    )


def fit_logistic_head(
    matrix,
    labels,
    basis,
    prior_coefficients,
    *,
    prior_strength=10.0,
    intercept_prior_strength=0.1,
):
    """Fit a ridge/MAP logistic head over a fixed unlabeled basis."""
    scores = np.asarray(matrix, dtype=float) @ np.asarray(basis, dtype=float)
    return _fit_head(
        scores,
        labels,
        basis,
        prior_coefficients,
        prior_strength=prior_strength,
        intercept_prior_strength=intercept_prior_strength,
    )


def fit_soft_logistic_head(
    labelled_matrix,
    labels,
    unlabelled_matrix,
    soft_targets,
    basis,
    prior_coefficients,
    *,
    prior_strength=0.1,
    intercept_prior_strength=0.1,
    soft_total_weight=10.0,
):
    """Fit a head to trusted labels plus confidence-weighted soft pseudo-labels."""
    basis = np.asarray(basis, dtype=float)
    labelled_scores = np.asarray(labelled_matrix, dtype=float) @ basis
    soft_scores = np.asarray(unlabelled_matrix, dtype=float) @ basis
    soft_targets = np.asarray(soft_targets, dtype=float)
    if len(soft_targets) != len(soft_scores) or np.any((soft_targets < 0) | (soft_targets > 1)):
        raise ValueError("soft targets must be probabilities aligned to unlabelled rows")
    return _fit_head(
        labelled_scores,
        labels,
        basis,
        prior_coefficients,
        prior_strength=prior_strength,
        intercept_prior_strength=intercept_prior_strength,
        soft_scores=soft_scores,
        soft_targets=soft_targets,
        soft_total_weight=soft_total_weight,
    )
