"""Geometry and score controls for the U2-prior reconciliation checkpoint.

This module does not define a new fusion estimator.  It makes the relationship
between an IU-PCR-anchored two-direction head and ordinary U2 coordinates
explicit, and implements deliberately optimistic score-combination diagnostics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, subspace_angles
from sklearn.metrics import average_precision_score, roc_auc_score

from .semi_supervised_fusion import fit_logistic_head


CURRENT_UPCR_KWARGS = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "n_components": 2,
    "auto_components": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}

HISTORICAL_UPCR_KWARGS = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}

GEOMETRY_TOLERANCES = {
    "max_principal_angle_rad": 1e-7,
    "projector_fro": 1e-7,
    "relative_reconstruction": 1e-7,
    "covariance_orthonormality": 1e-8,
}

FIT_TOLERANCES = {
    "objective_abs": 1e-7,
    "weight_max_abs": 1e-7,
    "intercept_abs": 1e-7,
    "score_scaled_max_abs": 1e-7,
}


@dataclass
class BasisAlignment:
    """Measured map from a source score basis into a reference basis."""

    coordinate_map: np.ndarray
    source_prior_in_reference: np.ndarray
    max_principal_angle_rad: float
    projector_fro: float
    covariance_orthonormality_reference: float
    covariance_orthonormality_source: float
    coordinate_map_orthogonality: float
    relative_reconstruction: float
    geometrically_equivalent: bool

    def as_dict(self):
        return {
            "coordinate_map": self.coordinate_map.tolist(),
            "source_prior_in_reference": self.source_prior_in_reference.tolist(),
            "max_principal_angle_rad": self.max_principal_angle_rad,
            "projector_fro": self.projector_fro,
            "covariance_orthonormality_reference": (
                self.covariance_orthonormality_reference
            ),
            "covariance_orthonormality_source": (
                self.covariance_orthonormality_source
            ),
            "coordinate_map_orthogonality": self.coordinate_map_orthogonality,
            "relative_reconstruction": self.relative_reconstruction,
            "geometrically_equivalent": self.geometrically_equivalent,
        }


def _validate_feature_matrix(F):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or min(F.shape) < 3:
        raise ValueError("F must have shape (features, samples), both at least three")
    if not np.isfinite(F).all():
        raise ValueError("F contains non-finite values")
    return F


def covariance_normalized_u2_basis(F):
    """Return the leading covariance basis with unit transductive score variance.

    The reconciliation inputs are row-centred by their frozen feature contract.
    Refusing non-centred input avoids silently confusing a raw second-moment
    metric with centred covariance.
    """
    F = _validate_feature_matrix(F)
    row_mean = F.mean(axis=1)
    if float(np.max(np.abs(row_mean))) > 1e-10:
        raise ValueError("F must be row-centred before U2 reconciliation")
    m, n = F.shape
    covariance = F @ F.T / n
    values, vectors = eigh(covariance, subset_by_index=[m - 2, m - 1])
    order = np.argsort(values)[::-1]
    raw_basis = vectors[:, order]
    raw_scores = F.T @ raw_basis
    scale = raw_scores.std(axis=0)
    if np.any(scale <= 1e-12):
        raise ValueError("ordinary U2 contains a degenerate score direction")
    basis = raw_basis / scale[None, :]
    return basis, F.T @ basis, values[order]


def basis_alignment(F, reference_basis, source_basis, source_prior=(1.0, 0.0)):
    """Measure whether two covariance-normalized bases span the same U2 space."""
    F = _validate_feature_matrix(F)
    reference = np.asarray(reference_basis, dtype=float)
    source = np.asarray(source_basis, dtype=float)
    prior = np.asarray(source_prior, dtype=float)
    if reference.ndim != 2 or source.shape != reference.shape:
        raise ValueError("reference and source bases must have matching shapes")
    if reference.shape[0] != F.shape[0] or prior.shape != (source.shape[1],):
        raise ValueError("basis or prior dimensions disagree with F")

    covariance = F @ F.T / F.shape[1]
    coordinate_map = reference.T @ covariance @ source
    reference_gram = reference.T @ covariance @ reference
    source_gram = source.T @ covariance @ source
    reconstruction = reference @ coordinate_map
    q_reference, _ = np.linalg.qr(reference)
    q_source, _ = np.linalg.qr(source)
    projector_reference = q_reference @ q_reference.T
    projector_source = q_source @ q_source.T

    diagnostics = {
        "max_principal_angle_rad": float(
            np.max(subspace_angles(reference, source))
        ),
        "projector_fro": float(
            np.linalg.norm(projector_reference - projector_source, ord="fro")
        ),
        "covariance_orthonormality_reference": float(
            np.linalg.norm(reference_gram - np.eye(reference.shape[1]), ord="fro")
        ),
        "covariance_orthonormality_source": float(
            np.linalg.norm(source_gram - np.eye(source.shape[1]), ord="fro")
        ),
        "coordinate_map_orthogonality": float(
            np.linalg.norm(
                coordinate_map.T @ coordinate_map - np.eye(source.shape[1]),
                ord="fro",
            )
        ),
        "relative_reconstruction": float(
            np.linalg.norm(reconstruction - source, ord="fro")
            / max(np.linalg.norm(source, ord="fro"), 1e-15)
        ),
    }
    equivalent = (
        diagnostics["max_principal_angle_rad"]
        <= GEOMETRY_TOLERANCES["max_principal_angle_rad"]
        and diagnostics["projector_fro"]
        <= GEOMETRY_TOLERANCES["projector_fro"]
        and diagnostics["relative_reconstruction"]
        <= GEOMETRY_TOLERANCES["relative_reconstruction"]
        and diagnostics["covariance_orthonormality_reference"]
        <= GEOMETRY_TOLERANCES["covariance_orthonormality"]
        and diagnostics["covariance_orthonormality_source"]
        <= GEOMETRY_TOLERANCES["covariance_orthonormality"]
    )
    return BasisAlignment(
        coordinate_map=coordinate_map,
        source_prior_in_reference=coordinate_map @ prior,
        geometrically_equivalent=bool(equivalent),
        **diagnostics,
    )


def fit_prior_head(matrix, labels, calibration_indices, basis, prior):
    """Fit the frozen summed-logloss head and return logits for every sample."""
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibration = np.asarray(calibration_indices, dtype=int)
    if calibration.ndim != 1 or len(calibration) == 0:
        raise ValueError("calibration_indices must be a nonempty vector")
    head = fit_logistic_head(
        matrix[calibration],
        labels[calibration],
        basis,
        prior,
        prior_strength=10.0,
        intercept_prior_strength=0.1,
    )
    scores = head.intercept + matrix @ head.weight
    return head, scores


def fit_equivalence_diagnostics(source_head, source_scores, target_head, target_scores):
    """Compare two heads that should differ only by an orthogonal basis change."""
    source_scores = np.asarray(source_scores, dtype=float)
    target_scores = np.asarray(target_scores, dtype=float)
    score_scale = max(1.0, float(np.std(source_scores)))
    values = {
        "objective_abs": float(abs(source_head.objective - target_head.objective)),
        "weight_max_abs": float(
            np.max(np.abs(source_head.weight - target_head.weight))
        ),
        "intercept_abs": float(abs(source_head.intercept - target_head.intercept)),
        "score_max_abs": float(np.max(np.abs(source_scores - target_scores))),
        "score_scale": score_scale,
    }
    values["score_scaled_max_abs"] = values["score_max_abs"] / score_scale
    values["fit_equivalent"] = bool(
        values["objective_abs"] <= FIT_TOLERANCES["objective_abs"]
        and values["weight_max_abs"] <= FIT_TOLERANCES["weight_max_abs"]
        and values["intercept_abs"] <= FIT_TOLERANCES["intercept_abs"]
        and values["score_scaled_max_abs"]
        <= FIT_TOLERANCES["score_scaled_max_abs"]
    )
    return values


def _standardize_score(score):
    score = np.asarray(score, dtype=float)
    centered = score - score.mean()
    scale = float(centered.std())
    return centered / scale if scale > 1e-12 else np.zeros_like(centered)


def _metrics(labels, scores):
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def optimistic_endpoint_controls(
    labels,
    score_iu,
    score_u2,
    evaluation_indices,
    *,
    alphas=None,
):
    """Price endpoint switching and evaluation-selected score interpolation.

    The chosen endpoint and interpolation coefficient both see evaluation
    labels.  These are optimistic diagnostics, never deployable estimators.
    """
    labels = np.asarray(labels, dtype=int)
    evaluation = np.asarray(evaluation_indices, dtype=int)
    iu = _standardize_score(score_iu)
    u2 = _standardize_score(score_u2)
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 201)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or len(alphas) < 2 or np.any((alphas < 0) | (alphas > 1)):
        raise ValueError("alphas must be a vector in [0, 1]")

    auc_iu, ap_iu = _metrics(labels[evaluation], iu[evaluation])
    auc_u2, ap_u2 = _metrics(labels[evaluation], u2[evaluation])
    endpoint_is_u2 = auc_u2 > auc_iu
    endpoint = u2 if endpoint_is_u2 else iu
    endpoint_auc, endpoint_ap = (
        (auc_u2, ap_u2) if endpoint_is_u2 else (auc_iu, ap_iu)
    )
    average = 0.5 * iu + 0.5 * u2
    average_auc, average_ap = _metrics(labels[evaluation], average[evaluation])

    interpolation_aucs = []
    for alpha in alphas:
        combined = (1.0 - alpha) * iu + alpha * u2
        interpolation_aucs.append(
            roc_auc_score(labels[evaluation], combined[evaluation])
        )
    best_index = int(np.argmax(interpolation_aucs))
    best_alpha = float(alphas[best_index])
    interpolation = (1.0 - best_alpha) * iu + best_alpha * u2
    interpolation_auc, interpolation_ap = _metrics(
        labels[evaluation], interpolation[evaluation]
    )
    return {
        "scores": {
            "optimistic_endpoint_switch": endpoint,
            "fixed_average": average,
            "optimistic_interpolation": interpolation,
        },
        "metrics": {
            "iu": {"auroc": auc_iu, "auprc": ap_iu},
            "u2": {"auroc": auc_u2, "auprc": ap_u2},
            "optimistic_endpoint_switch": {
                "auroc": endpoint_auc,
                "auprc": endpoint_ap,
                "selected": "u2" if endpoint_is_u2 else "iu",
            },
            "fixed_average": {"auroc": average_auc, "auprc": average_ap},
            "optimistic_interpolation": {
                "auroc": interpolation_auc,
                "auprc": interpolation_ap,
                "alpha_u2": best_alpha,
            },
        },
    }


__all__ = [
    "BasisAlignment",
    "CURRENT_UPCR_KWARGS",
    "FIT_TOLERANCES",
    "GEOMETRY_TOLERANCES",
    "HISTORICAL_UPCR_KWARGS",
    "basis_alignment",
    "covariance_normalized_u2_basis",
    "fit_equivalence_diagnostics",
    "fit_prior_head",
    "optimistic_endpoint_controls",
]
