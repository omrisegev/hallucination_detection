"""One-stage B3-orthogonal boost from an IU-PGRD weak learner.

The frozen B3 posterior is the baseline.  Ordinary IU-PCR is used only to
construct the family-residual coordinate system in which historical PGRD had
its positive mechanism result.  All fits in this module are target-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

import numpy as np
from scipy.special import expit, logit

from .contribution_subspace import (
    fit_contribution_transform,
    iu_family_contributions,
)
from .graph_topology import self_safe_knn_graph
from .laplacian_upcr import (
    IU_FIT_DEFAULTS,
    symmetric_normalized_laplacian,
)
from .specrage_views import VIEW_ORDER
from .upcr import upcr_fit


EPS = 1e-12


@dataclass(frozen=True)
class GraphRoughnessMoment:
    A: np.ndarray
    c: np.ndarray
    presence: np.ndarray
    families: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _IUResidualState:
    baseline_fit: object
    contribution_space: object
    transform: object
    baseline: np.ndarray
    residuals: np.ndarray
    standardized_contributions: np.ndarray


def _fit_iu_residual_state(X_risk: np.ndarray, feature_names: Sequence[str]):
    """Local copy of the historical IU contribution-residual construction."""

    F = np.asarray(X_risk, dtype=float).T
    fitted = upcr_fit(F, **dict(IU_FIT_DEFAULTS))
    space = iu_family_contributions(F, feature_names, fitted.w)
    transform = fit_contribution_transform(space, np.arange(F.shape[1], dtype=int))
    baseline, residuals = transform.apply(space.baseline_score, space.contributions)
    standardized = (
        np.asarray(space.contributions, dtype=float)
        - transform.contribution_mean[None, :]
    ) / transform.contribution_scale[None, :]
    return _IUResidualState(
        baseline_fit=fitted,
        contribution_space=space,
        transform=transform,
        baseline=np.asarray(baseline, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        standardized_contributions=np.asarray(standardized, dtype=float),
    )


def _graph_roughness_moment(
    baseline: np.ndarray,
    residuals: np.ndarray,
    families: Sequence[str],
    graph: object,
) -> GraphRoughnessMoment:
    """Historical trace-normalized global-family moment, implemented locally."""

    b = np.asarray(baseline, dtype=float)
    R = np.asarray(residuals, dtype=float)
    local_families = tuple(str(value) for value in families)
    laplacian = symmetric_normalized_laplacian(graph)
    local_A = np.asarray(R.T @ (laplacian @ R) / len(b), dtype=float)
    local_A = 0.5 * (local_A + local_A.T)
    local_c = np.asarray(R.T @ (laplacian @ b) / len(b), dtype=float)
    raw_trace = float(np.trace(local_A))
    if not np.isfinite(raw_trace) or raw_trace <= EPS:
        raise ValueError("roughness trace is non-positive")
    trace_scale = float(len(local_families) / raw_trace)
    indices = np.asarray([VIEW_ORDER.index(name) for name in local_families], dtype=int)
    A = np.zeros((len(VIEW_ORDER), len(VIEW_ORDER)), dtype=float)
    c = np.zeros(len(VIEW_ORDER), dtype=float)
    A[np.ix_(indices, indices)] = trace_scale * local_A
    c[indices] = trace_scale * local_c
    presence = np.zeros(len(VIEW_ORDER), dtype=bool)
    presence[indices] = True
    return GraphRoughnessMoment(
        A=A,
        c=c,
        presence=presence,
        families=tuple(VIEW_ORDER),
        diagnostics={
            "n_samples": int(len(b)),
            "n_families_present": int(len(local_families)),
            "roughness_trace_raw": raw_trace,
            "trace_scale": trace_scale,
            "scaled_cross_norm": float(np.linalg.norm(c)),
        },
    )


def _pool_moments(
    moments: Sequence[GraphRoughnessMoment], group_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not moments or len(moments) != len(group_ids):
        raise ValueError("moments/group IDs must be nonempty and aligned")
    groups = tuple(sorted(set(str(value) for value in group_ids)))
    group_A, group_c = [], []
    for group in groups:
        selected = [
            moment
            for moment, group_id in zip(moments, group_ids)
            if str(group_id) == group
        ]
        group_A.append(np.mean([moment.A for moment in selected], axis=0))
        group_c.append(np.mean([moment.c for moment in selected], axis=0))
    A = np.mean(group_A, axis=0)
    A = 0.5 * (A + A.T)
    c = np.mean(group_c, axis=0)
    return np.asarray(A, dtype=float), np.asarray(c, dtype=float), groups


@dataclass(frozen=True)
class B3IUPGRDCell:
    """One target-free cell in the B3-aligned IU residual geometry."""

    cell_id: str
    baseline_score: np.ndarray
    baseline_logit: np.ndarray
    baseline_mean: float
    baseline_scale: float
    baseline_z: np.ndarray
    iu_score: np.ndarray
    iu_score_aligned: np.ndarray
    iu_orientation: int
    iu_weights: np.ndarray
    families: tuple[str, ...]
    raw_contributions: np.ndarray
    standardized_contributions: np.ndarray
    residuals: np.ndarray
    graph: object
    laplacian: object
    moment: GraphRoughnessMoment
    transform_arrays: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class B3IUPGRDBoostScore:
    """B3 plus one bounded residual correction."""

    score: np.ndarray
    logit: np.ndarray
    correction_z: np.ndarray
    raw_correction: np.ndarray
    projected_correction: np.ndarray
    projection_coefficients: np.ndarray
    direction: np.ndarray
    row_permutation: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    return logit(np.clip(values, 1e-12, 1.0 - 1e-12))


def _lexical_tie_keys(row_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([str(value) for value in row_ids], dtype=str)
    if values.ndim != 1 or len(values) != len(set(values.tolist())):
        raise ValueError("row IDs must be a unique vector")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def deterministic_row_permutation(
    row_ids: Sequence[str], *, salt: str
) -> np.ndarray:
    """Return a stable non-identity row permutation from row IDs and a salt."""

    values = tuple(str(value) for value in row_ids)
    if len(values) < 3 or len(values) != len(set(values)):
        raise ValueError("row permutation needs at least three unique row IDs")
    keys = np.asarray(
        [
            hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()
            for value in values
        ],
        dtype=str,
    )
    permutation = np.argsort(keys, kind="mergesort")
    if np.array_equal(permutation, np.arange(len(values))):
        permutation = np.roll(permutation, 1)
    return np.asarray(permutation, dtype=np.int64)


def fit_b3_iupgrd_cell(
    cell_id: str,
    X_risk: np.ndarray,
    feature_names: Sequence[str],
    baseline_score: np.ndarray,
    row_ids: Sequence[str],
    *,
    k: int = 7,
) -> B3IUPGRDCell:
    """Fit one B3-aligned IU residual state and union-kNN graph label-free."""

    X = np.asarray(X_risk, dtype=float)
    names = tuple(str(value) for value in feature_names)
    score = np.asarray(baseline_score, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X_risk/feature-name shape mismatch")
    if score.shape != (len(X),) or len(row_ids) != len(X):
        raise ValueError("B3 score/row alignment mismatch")
    if not np.isfinite(X).all() or not np.isfinite(score).all():
        raise ValueError("cell inputs must be finite")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("B3 score must contain probabilities")

    ell = _safe_logit(score)
    ell_mean = float(np.mean(ell))
    ell_scale = float(np.std(ell))
    if ell_scale <= EPS:
        raise ValueError("B3 logit is constant")
    baseline_z = (ell - ell_mean) / ell_scale

    iu = _fit_iu_residual_state(X, names)
    iu_score = np.asarray(iu.baseline, dtype=float)
    correlation = float(np.dot(baseline_z, iu_score) / len(iu_score))
    if not np.isfinite(correlation) or abs(correlation) <= 0.05:
        raise ValueError("IU/B3 orientation is not identifiable")
    orientation = 1 if correlation > 0.0 else -1
    aligned_iu = orientation * iu_score
    aligned_residuals = orientation * np.asarray(iu.residuals, dtype=float)
    residual_mean_error = float(np.max(np.abs(np.mean(aligned_residuals, axis=0))))
    residual_sd_error = float(
        np.max(np.abs(np.std(aligned_residuals, axis=0) - 1.0))
    )
    residual_iu_covariance_error = float(
        np.max(np.abs(aligned_residuals.T @ aligned_iu / len(aligned_iu)))
    )
    if max(
        residual_mean_error,
        residual_sd_error,
        residual_iu_covariance_error,
    ) > 1e-8:
        raise ValueError("IU family residual invariants failed")

    graph = self_safe_knn_graph(
        aligned_residuals,
        k=int(k),
        tie_keys=_lexical_tie_keys(row_ids),
    )
    laplacian = symmetric_normalized_laplacian(graph)
    moment = _graph_roughness_moment(
        aligned_iu,
        aligned_residuals,
        iu.contribution_space.families,
        graph,
    )
    transform = iu.transform
    transform_arrays = {
        "baseline_mean": np.asarray(transform.baseline_mean, dtype=np.float64),
        "baseline_scale": np.asarray(transform.baseline_scale, dtype=np.float64),
        "contribution_mean": np.asarray(transform.contribution_mean, dtype=np.float64),
        "contribution_scale": np.asarray(transform.contribution_scale, dtype=np.float64),
        "baseline_loadings": np.asarray(transform.baseline_loadings, dtype=np.float64),
        "residual_mean": np.asarray(transform.residual_mean, dtype=np.float64),
        "residual_scale": np.asarray(transform.residual_scale, dtype=np.float64),
    }
    return B3IUPGRDCell(
        cell_id=str(cell_id),
        baseline_score=score,
        baseline_logit=ell,
        baseline_mean=ell_mean,
        baseline_scale=ell_scale,
        baseline_z=baseline_z,
        iu_score=iu_score,
        iu_score_aligned=aligned_iu,
        iu_orientation=orientation,
        iu_weights=np.asarray(iu.baseline_fit.w, dtype=float),
        families=tuple(iu.contribution_space.families),
        raw_contributions=np.asarray(iu.contribution_space.contributions, dtype=float),
        standardized_contributions=np.asarray(
            iu.standardized_contributions, dtype=float
        ),
        residuals=aligned_residuals,
        graph=graph,
        laplacian=laplacian,
        moment=moment,
        transform_arrays=transform_arrays,
        diagnostics={
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_families": int(len(iu.contribution_space.families)),
            "graph_k": int(min(int(k), len(X) - 1)),
            "graph_nnz": int(graph.nnz),
            "iu_b3_correlation_before_orientation": correlation,
            "iu_orientation": orientation,
            "iu_b3_correlation_after_orientation": abs(correlation),
            "iu_reconstruction_error": float(
                iu.contribution_space.diagnostics["reconstruction_error"]
            ),
            "residual_mean_max_abs": residual_mean_error,
            "residual_sd_max_abs_error": residual_sd_error,
            "residual_iu_covariance_max_abs": residual_iu_covariance_error,
            "roughness_trace_raw": float(
                moment.diagnostics["roughness_trace_raw"]
            ),
            "uses_labels": False,
        },
    )


def pooled_cross_only_direction(
    cells: Sequence[B3IUPGRDCell], group_ids: Sequence[str]
) -> tuple[np.ndarray, dict]:
    """Equal-group pool of donor moments followed by the supported ``-c`` rule."""

    records = tuple(cells)
    if not records:
        raise ValueError("at least one donor cell is required")
    A, c, groups = _pool_moments(
        [record.moment for record in records],
        group_ids,
    )
    direction = -np.asarray(c, dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("pooled cross-gradient is negligible")
    return direction, {
        "n_donor_cells": int(len(records)),
        "n_donor_groups": int(len(groups)),
        "donor_groups": list(groups),
        "pooled_A_trace": float(np.trace(A)),
        "pooled_c_norm": float(np.linalg.norm(c)),
        "direction_norm": norm,
        "cross_only": True,
        "uses_labels": False,
    }


def permute_family_direction(
    direction: np.ndarray, permutation: Sequence[int]
) -> np.ndarray:
    values = np.asarray(direction, dtype=float)
    indices = np.asarray(permutation, dtype=int)
    if values.shape != (len(VIEW_ORDER),):
        raise ValueError("direction has the wrong global family shape")
    if sorted(indices.tolist()) != list(range(len(VIEW_ORDER))):
        raise ValueError("family permutation must be a bijection")
    return np.asarray(values[indices], dtype=float)


def score_b3_iupgrd_boost(
    cell: B3IUPGRDCell,
    direction: np.ndarray,
    *,
    trust_factor: float,
    project_against_b3: bool,
    row_ids: Sequence[str],
    row_permutation_salt: str | None = None,
) -> B3IUPGRDBoostScore:
    """Apply one fixed IU-PGRD weak learner as a B3 residual update."""

    global_direction = np.asarray(direction, dtype=float)
    if global_direction.shape != (len(VIEW_ORDER),):
        raise ValueError("direction has the wrong global family shape")
    trust_factor = float(trust_factor)
    if (
        not np.isfinite(global_direction).all()
        or not np.isfinite(trust_factor)
        or trust_factor < 0.0
    ):
        raise ValueError("invalid direction/trust factor")
    local_indices = np.asarray([VIEW_ORDER.index(name) for name in cell.families])
    raw = np.asarray(cell.residuals @ global_direction[local_indices], dtype=float)
    permutation = np.arange(len(raw), dtype=np.int64)
    if row_permutation_salt is not None:
        permutation = deterministic_row_permutation(
            row_ids, salt=str(row_permutation_salt)
        )
        raw = raw[permutation]

    if project_against_b3:
        design = np.column_stack([np.ones(len(raw)), cell.baseline_z])
        design_rank = int(np.linalg.matrix_rank(design))
        design_condition = float(np.linalg.cond(design))
        if design_rank != 2 or not np.isfinite(design_condition):
            raise ValueError("B3 projection basis is rank deficient")
        coefficients = np.linalg.lstsq(design, raw, rcond=None)[0]
        projected = raw - design @ coefficients
    else:
        design_rank = 0
        design_condition = 0.0
        coefficients = np.zeros(2, dtype=float)
        projected = raw.copy()

    if trust_factor == 0.0:
        return B3IUPGRDBoostScore(
            score=cell.baseline_score,
            logit=cell.baseline_logit,
            correction_z=np.zeros(len(raw), dtype=float),
            raw_correction=raw,
            projected_correction=projected,
            projection_coefficients=np.asarray(coefficients, dtype=float),
            direction=global_direction,
            row_permutation=permutation,
            diagnostics={
                "exact_b3_alias": True,
                "trust_factor": 0.0,
                "uses_labels": False,
            },
        )
    projected_scale = float(np.std(projected))
    if not np.isfinite(projected_scale) or projected_scale <= EPS:
        raise ValueError("residual correction is constant")
    requested_sd = float(trust_factor) / len(cell.families)
    correction_z = requested_sd * projected / projected_scale
    updated_logit = cell.baseline_logit + cell.baseline_scale * correction_z
    updated_score = expit(updated_logit)
    covariance = float(np.dot(cell.baseline_z, correction_z) / len(raw))
    projected_mean = float(np.mean(projected))
    projected_b3_covariance = float(
        np.dot(cell.baseline_z, projected) / len(projected)
    )
    if project_against_b3 and max(
        abs(projected_mean), abs(projected_b3_covariance)
    ) > 1e-9:
        raise ValueError("B3 residual projection invariants failed")
    reconstruction = float(
        np.max(
            np.abs(
                updated_logit
                - (cell.baseline_logit + cell.baseline_scale * correction_z)
            )
        )
    )
    return B3IUPGRDBoostScore(
        score=np.asarray(updated_score, dtype=float),
        logit=np.asarray(updated_logit, dtype=float),
        correction_z=np.asarray(correction_z, dtype=float),
        raw_correction=np.asarray(raw, dtype=float),
        projected_correction=np.asarray(projected, dtype=float),
        projection_coefficients=np.asarray(coefficients, dtype=float),
        direction=global_direction,
        row_permutation=permutation,
        diagnostics={
            "exact_b3_alias": False,
            "trust_factor": float(trust_factor),
            "n_present_families": int(len(cell.families)),
            "requested_correction_z_sd": requested_sd,
            "correction_z_sd": float(np.std(correction_z)),
            "correction_logit_sd": float(np.std(cell.baseline_scale * correction_z)),
            "project_against_b3": bool(project_against_b3),
            "projection_basis": ["intercept", "standardized_b3_logit"],
            "projection_basis_rank": design_rank,
            "projection_basis_condition": design_condition,
            "projected_mean": projected_mean,
            "projected_b3_covariance": projected_b3_covariance,
            "row_permuted": row_permutation_salt is not None,
            "raw_correction_sd": float(np.std(raw)),
            "projected_correction_sd": projected_scale,
            "baseline_correction_covariance": covariance,
            "logit_reconstruction_max_abs": reconstruction,
            "uses_labels": False,
        },
    )


__all__ = [
    "B3IUPGRDBoostScore",
    "B3IUPGRDCell",
    "deterministic_row_permutation",
    "fit_b3_iupgrd_cell",
    "permute_family_direction",
    "pooled_cross_only_direction",
    "score_b3_iupgrd_boost",
]
