"""Few-label target anchoring for Laplacian-regularized IU-PCR.

The label-free DUFS graph can identify a clean manifold without knowing
whether that manifold is relevant to the prediction target.  This module uses
a small calibration set only to choose the coordinates used in graph
distances.  It does not alter IU-PCR's covariance, moment estimator, U2
subspace, or feature pool.

For oriented feature row ``i`` and calibration indices ``A`` the gate is

    q_i = max(Corr(F_i[A], y[A]), 0) / RMS(max(r, 0)).

One-class targets and all-zero positive correlations fall back to an ungated
graph.  A zero gate removes a coordinate from graph distances only; all
features remain in the final fusion equation.
"""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression

from .laplacian_upcr import LaplacianIUResult, laplacian_iu_fit
from .upcr import UPCRResult, upcr_fit


_EPS = 1e-12


@dataclass
class CorrelationGateResult:
    """Positive-correlation graph gates and their audit information."""

    gates: np.ndarray
    correlations: np.ndarray
    positive_correlations: np.ndarray
    fallback: bool
    fallback_reason: str
    diagnostics: dict = field(default_factory=dict)


@dataclass
class TargetAnchoredIUResult:
    """TA-LIU fit together with the label-derived graph gates."""

    fit: LaplacianIUResult
    gate_result: CorrelationGateResult


@dataclass
class ProjectedRidgeResult:
    """Trace-matched isotropic ridge inside ordinary IU-PCR's U2."""

    w: np.ndarray
    baseline: UPCRResult
    projected_covariance: np.ndarray
    projected_ridge: np.ndarray
    lambda_: float
    diagnostics: dict = field(default_factory=dict)


def _validate_features(F):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must have shape (features, samples)")
    if F.shape[0] < 3 or F.shape[1] < 3:
        raise ValueError("at least three features and samples are required")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    return F


def _validate_indices(indices, n):
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a nonempty one-dimensional array")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices contain an out-of-range sample")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("indices must not contain duplicates")
    return indices


def positive_correlation_gates(
    F,
    target,
    *,
    indices=None,
    require_two_classes=True,
):
    """Return RMS-normalized positive Pearson-correlation graph gates.

    Parameters
    ----------
    F:
        Oriented feature matrix with shape ``(m_features, n_samples)``.
    target:
        A target value for every sample.  Binary labels are expected when
        ``require_two_classes`` is true; a continuous pseudo-target is allowed
        otherwise.
    indices:
        Samples allowed to influence the gates.  ``None`` uses all samples.
    require_two_classes:
        Enforce the declared TA-LIU one-class fallback.  Set false only for the
        preregistered continuous pseudo-anchor control.
    """
    F = _validate_features(F)
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or len(target) != F.shape[1]:
        raise ValueError("target must provide one finite value per sample")
    if not np.isfinite(target).all():
        raise ValueError("target contains non-finite values")
    if indices is None:
        indices = np.arange(F.shape[1], dtype=int)
    indices = _validate_indices(indices, F.shape[1])

    selected_target = target[indices]
    selected_features = F[:, indices]
    correlations = np.zeros(F.shape[0], dtype=float)

    if require_two_classes and len(np.unique(selected_target)) < 2:
        reason = "one_class"
        positive = np.zeros_like(correlations)
        gates = np.ones_like(correlations)
        fallback = True
    else:
        centered_target = selected_target - selected_target.mean()
        centered_features = (
            selected_features
            - selected_features.mean(axis=1, keepdims=True)
        )
        numerator = centered_features @ centered_target
        denominator = np.sqrt(
            np.sum(centered_features ** 2, axis=1)
            * np.sum(centered_target ** 2)
        )
        np.divide(
            numerator,
            denominator,
            out=correlations,
            where=denominator > _EPS,
        )
        correlations = np.clip(correlations, -1.0, 1.0)
        positive = np.maximum(correlations, 0.0)
        rms = float(np.sqrt(np.mean(positive ** 2)))
        if rms <= _EPS:
            reason = "no_positive_correlation"
            gates = np.ones_like(correlations)
            fallback = True
        else:
            reason = ""
            gates = positive / rms
            fallback = False

    diagnostics = {
        "n_gate_samples": int(len(indices)),
        "n_target_values": int(len(np.unique(selected_target))),
        "mean_correlation": float(np.mean(correlations)),
        "max_correlation": float(np.max(correlations)),
        "positive_gate_count": int(np.sum(positive > 0)),
        "zero_gate_fraction": float(np.mean(gates == 0.0)),
        "effective_feature_count": float(
            (np.sum(gates) ** 2) / (np.sum(gates ** 2) + _EPS)
        ),
    }
    return CorrelationGateResult(
        gates=gates,
        correlations=correlations,
        positive_correlations=positive,
        fallback=fallback,
        fallback_reason=reason,
        diagnostics=diagnostics,
    )


def target_anchored_laplacian_fit(
    F,
    labels,
    calibration_indices,
    *,
    lambda_=0.1,
    k=7,
    baseline_kwargs=None,
):
    """Fit TA-LIU while limiting all label access to calibration indices."""
    gate_result = positive_correlation_gates(
        F,
        labels,
        indices=calibration_indices,
        require_two_classes=True,
    )
    fit = laplacian_iu_fit(
        F,
        lambda_=lambda_,
        gates=gate_result.gates,
        k=k,
        baseline_kwargs=baseline_kwargs,
    )
    return TargetAnchoredIUResult(fit=fit, gate_result=gate_result)


def pseudo_anchor_laplacian_fit(
    F,
    pseudo_target,
    *,
    lambda_=0.1,
    k=7,
    baseline_kwargs=None,
):
    """Fit the full-data continuous pseudo-anchor negative control."""
    gate_result = positive_correlation_gates(
        F,
        pseudo_target,
        indices=None,
        require_two_classes=False,
    )
    fit = laplacian_iu_fit(
        F,
        lambda_=lambda_,
        gates=gate_result.gates,
        k=k,
        baseline_kwargs=baseline_kwargs,
    )
    return TargetAnchoredIUResult(fit=fit, gate_result=gate_result)


def projected_ridge_fit(F, *, lambda_=0.1, baseline_kwargs=None):
    """Fit the frozen trace-matched projected-ridge control.

    This is isotropic ridge in the same ordinary IU-PCR two-dimensional
    subspace.  It changes neither the rho estimator nor the feature pool.
    """
    F = _validate_features(F)
    lambda_ = float(lambda_)
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and nonnegative")

    kwargs = {
        "loss": "l2",
        "exclusion": False,
        "difficulty_gate": False,
        "simple_avg_fallback": False,
        "recompute_after_exclusion": False,
        "g2_projection_k": 1,
        "scale_ratio": 0.25,
        "n_components": 2,
        "auto_components": False,
    }
    if baseline_kwargs:
        kwargs.update(baseline_kwargs)
    baseline = upcr_fit(F, **kwargs)

    m, n = F.shape
    covariance = F @ F.T / n
    values, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    basis = basis[:, np.argsort(values)[::-1]]
    projected = basis.T @ covariance @ basis
    projected = 0.5 * (projected + projected.T)
    ridge = np.eye(2) * np.trace(projected) / 2.0
    rhs = basis.T @ baseline.rho_hat
    if lambda_ == 0.0:
        weights = baseline.w.copy()
    else:
        weights = basis @ np.linalg.solve(projected + lambda_ * ridge, rhs)
    zero_weights = basis @ np.linalg.solve(projected, rhs)
    return ProjectedRidgeResult(
        w=weights,
        baseline=baseline,
        projected_covariance=projected,
        projected_ridge=ridge,
        lambda_=lambda_,
        diagnostics={
            "projected_condition_number": float(
                np.linalg.cond(projected + lambda_ * ridge)
            ),
            "zero_equation_weight_error": float(
                np.max(np.abs(zero_weights - baseline.w))
            ),
            "weight_norm": float(np.linalg.norm(weights)),
            "score_variance": float(np.var(weights @ F)),
        },
    )


def ordinary_u2_coordinates(F):
    """Return label-free top-two PCR coordinates standardized over all samples."""
    F = _validate_features(F)
    m, n = F.shape
    covariance = F @ F.T / n
    values, basis = eigh(covariance, subset_by_index=[m - 2, m - 1])
    order = np.argsort(values)[::-1]
    basis = basis[:, order]
    coordinates = F.T @ basis
    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    scale = coordinates.std(axis=0, keepdims=True)
    coordinates = coordinates / np.where(scale > _EPS, scale, 1.0)
    return coordinates, basis


def fixed_logistic_scores(X, labels, calibration_indices):
    """Fit the preregistered fixed-L2 logistic control.

    The returned score covers all transductive samples.  A one-class
    calibration prefix produces its empirical class probability as a constant.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != len(labels):
        raise ValueError("X and labels must have shapes (samples, d) and (samples,)")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary")
    indices = _validate_indices(calibration_indices, len(labels))
    calibration_labels = labels[indices]
    classes = np.unique(calibration_labels)
    if len(classes) == 1:
        prior = float(calibration_labels.mean())
        return np.full(len(labels), prior, dtype=float), {
            "fallback": True,
            "fallback_reason": "one_class",
            "calibration_positive_rate": prior,
        }

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        class_weight=None,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-8,
        random_state=0,
    )
    with warnings.catch_warnings():
        # scikit-learn 1.8 deprecates the spelling but the preregistration fixes
        # this exact estimator configuration, including penalty="l2".
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(X[indices], calibration_labels)
    return model.predict_proba(X)[:, 1], {
        "fallback": False,
        "fallback_reason": "",
        "calibration_positive_rate": float(calibration_labels.mean()),
        "n_iter": int(model.n_iter_[0]),
    }


__all__ = [
    "CorrelationGateResult",
    "ProjectedRidgeResult",
    "TargetAnchoredIUResult",
    "fixed_logistic_scores",
    "ordinary_u2_coordinates",
    "positive_correlation_gates",
    "projected_ridge_fit",
    "pseudo_anchor_laplacian_fit",
    "target_anchored_laplacian_fit",
]
