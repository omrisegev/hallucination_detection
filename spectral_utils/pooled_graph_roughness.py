"""Cross-environment calibration from family-residual graph roughness.

The routines in this module are deliberately label-free.  A sample graph in
each environment supplies the local quadratic expansion

    E_e(delta) = (b_e + R_e delta)' L_e (b_e + R_e delta) / n_e,

whose Hessian and gradient terms are ``A_e = R_e' L_e R_e / n_e`` and
``c_e = R_e' L_e b_e / n_e``.  After hierarchical environment pooling, the
regularized roughness-descent direction is

    d = -lambda (I + lambda A_bar)^(-1) c_bar.

Unlike Family-NRM, this construction does not select a covariance eigenvector
or attach semantics to an eigenvalue near one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .laplacian_upcr import symmetric_normalized_laplacian
from .specrage_views import VIEW_ORDER


EPS = 1e-12


@dataclass(frozen=True)
class GraphRoughnessMoment:
    """One environment's trace-normalized roughness expansion."""

    A: np.ndarray
    c: np.ndarray
    presence: np.ndarray
    families: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PooledRoughnessCalibration:
    """A pooled label-free correction direction."""

    A: np.ndarray
    c: np.ndarray
    direction: np.ndarray
    lambda_: float
    families: tuple[str, ...]
    source_groups: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PooledRoughnessScore:
    """An IU score plus a bounded family-residual correction."""

    score: np.ndarray
    correction: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _family_indices(
    families: Sequence[str], global_families: Sequence[str]
) -> np.ndarray:
    families = tuple(str(value) for value in families)
    global_families = tuple(str(value) for value in global_families)
    if len(set(families)) != len(families):
        raise ValueError("families must be unique")
    if len(set(global_families)) != len(global_families):
        raise ValueError("global_families must be unique")
    unknown = sorted(set(families) - set(global_families))
    if unknown:
        raise ValueError(f"unknown families: {unknown}")
    return np.asarray([global_families.index(name) for name in families], dtype=int)


def align_family_matrix(
    residuals,
    families: Sequence[str],
    *,
    global_families: Sequence[str] = VIEW_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed local family columns into a fixed global family registry."""

    values = np.asarray(residuals, dtype=float)
    families = tuple(str(value) for value in families)
    global_families = tuple(str(value) for value in global_families)
    if values.ndim != 2 or values.shape[1] != len(families):
        raise ValueError("residuals/families shape mismatch")
    if not np.isfinite(values).all():
        raise ValueError("residuals contain non-finite values")
    indices = _family_indices(families, global_families)
    aligned = np.zeros((values.shape[0], len(global_families)), dtype=float)
    aligned[:, indices] = values
    presence = np.zeros(len(global_families), dtype=bool)
    presence[indices] = True
    return aligned, presence


def graph_roughness_moment(
    baseline,
    residuals,
    families: Sequence[str],
    graph: csr_matrix,
    *,
    global_families: Sequence[str] = VIEW_ORDER,
) -> GraphRoughnessMoment:
    """Return one trace-normalized, globally aligned ``(A, c)`` moment."""

    b = np.asarray(baseline, dtype=float)
    R = np.asarray(residuals, dtype=float)
    families = tuple(str(value) for value in families)
    global_families = tuple(str(value) for value in global_families)
    if b.ndim != 1 or R.ndim != 2 or R.shape != (len(b), len(families)):
        raise ValueError("baseline/residual/family shape mismatch")
    if len(b) < 3 or not np.isfinite(b).all() or not np.isfinite(R).all():
        raise ValueError("baseline/residuals must be finite with at least 3 rows")
    if graph.shape != (len(b), len(b)):
        raise ValueError("graph/sample shape mismatch")
    L = symmetric_normalized_laplacian(graph)
    local_A = np.asarray(R.T @ (L @ R) / len(b), dtype=float)
    local_A = 0.5 * (local_A + local_A.T)
    local_c = np.asarray(R.T @ (L @ b) / len(b), dtype=float)
    raw_trace = float(np.trace(local_A))
    if not np.isfinite(raw_trace) or raw_trace <= EPS:
        raise ValueError("roughness trace is non-positive")
    trace_scale = float(len(families) / raw_trace)
    indices = _family_indices(families, global_families)
    A = np.zeros((len(global_families), len(global_families)), dtype=float)
    c = np.zeros(len(global_families), dtype=float)
    A[np.ix_(indices, indices)] = trace_scale * local_A
    c[indices] = trace_scale * local_c
    presence = np.zeros(len(global_families), dtype=bool)
    presence[indices] = True
    return GraphRoughnessMoment(
        A=A,
        c=c,
        presence=presence,
        families=global_families,
        diagnostics={
            "n_samples": int(len(b)),
            "n_families_present": int(len(families)),
            "roughness_trace_raw": raw_trace,
            "trace_scale": trace_scale,
            "cross_norm": float(np.linalg.norm(local_c)),
            "scaled_cross_norm": float(np.linalg.norm(c)),
            "minimum_A_eigenvalue": float(np.min(np.linalg.eigvalsh(local_A))),
        },
    )


def pool_graph_roughness_moments(
    moments: Sequence[GraphRoughnessMoment],
    group_ids: Sequence[str],
    *,
    pooling: str = "equal_group",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Pool moments equally within group, then equally across groups.

    ``pooling='equal_cell'`` is retained as a named mechanism control.
    Missing family rows/columns have already been embedded as zeros; no
    pairwise-availability reweighting is performed.
    """

    moments = tuple(moments)
    group_ids = tuple(str(value) for value in group_ids)
    if not moments or len(moments) != len(group_ids):
        raise ValueError("moments/group_ids must be non-empty and aligned")
    families = moments[0].families
    shape = moments[0].A.shape
    for moment in moments:
        if moment.families != families or moment.A.shape != shape:
            raise ValueError("moment registries disagree")
        if moment.c.shape != (shape[0],):
            raise ValueError("invalid moment cross-vector shape")
    groups = tuple(sorted(set(group_ids)))
    if pooling == "equal_cell":
        A = np.mean([moment.A for moment in moments], axis=0)
        c = np.mean([moment.c for moment in moments], axis=0)
    elif pooling == "equal_group":
        group_A, group_c = [], []
        for group in groups:
            selected = [
                moment for moment, group_id in zip(moments, group_ids)
                if group_id == group
            ]
            group_A.append(np.mean([moment.A for moment in selected], axis=0))
            group_c.append(np.mean([moment.c for moment in selected], axis=0))
        A = np.mean(group_A, axis=0)
        c = np.mean(group_c, axis=0)
    else:
        raise ValueError("pooling must be equal_group or equal_cell")
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    c = np.asarray(c, dtype=float)
    if not np.isfinite(A).all() or not np.isfinite(c).all():
        raise ValueError("pooled moments are non-finite")
    return A, c, groups


def fit_pooled_roughness_calibration(
    moments: Sequence[GraphRoughnessMoment],
    group_ids: Sequence[str],
    lambda_: float,
    *,
    pooling: str = "equal_group",
    cross_only: bool = False,
) -> PooledRoughnessCalibration:
    """Fit a pooled regularized descent direction without using labels."""

    lambda_ = float(lambda_)
    if not np.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("lambda must be finite and positive")
    A, c, groups = pool_graph_roughness_moments(
        moments, group_ids, pooling=pooling
    )
    if cross_only:
        direction = -c
    else:
        direction = -lambda_ * np.linalg.solve(
            np.eye(len(c), dtype=float) + lambda_ * A, c
        )
    if not np.isfinite(direction).all():
        raise ValueError("calibration direction is non-finite")
    return PooledRoughnessCalibration(
        A=A,
        c=c,
        direction=np.asarray(direction, dtype=float),
        lambda_=lambda_,
        families=tuple(moments[0].families),
        source_groups=groups,
        diagnostics={
            "pooling": pooling,
            "cross_only": bool(cross_only),
            "n_source_cells": int(len(moments)),
            "n_source_groups": int(len(groups)),
            "direction_norm": float(np.linalg.norm(direction)),
            "descent_inner_product": float(np.dot(c, direction)),
            "solve_residual": float(np.linalg.norm(
                direction + lambda_ * (A @ direction + c)
            )) if not cross_only else 0.0,
        },
    )


def apply_pooled_roughness(
    baseline,
    residuals,
    families: Sequence[str],
    calibration: PooledRoughnessCalibration,
    trust_factor: float,
) -> PooledRoughnessScore:
    """Apply a fixed direction with correction SD ``trust/G_present``."""

    b = np.asarray(baseline, dtype=float)
    R = np.asarray(residuals, dtype=float)
    families = tuple(str(value) for value in families)
    trust_factor = float(trust_factor)
    if b.ndim != 1 or R.shape != (len(b), len(families)):
        raise ValueError("baseline/residual/family shape mismatch")
    if not np.isfinite(trust_factor) or trust_factor < 0:
        raise ValueError("trust_factor must be finite and nonnegative")
    indices = _family_indices(families, calibration.families)
    raw = R @ calibration.direction[indices]
    raw_sd = float(np.std(raw, ddof=0))
    requested_sd = trust_factor / len(families)
    if trust_factor == 0.0 or raw_sd <= EPS:
        correction = np.zeros_like(b)
    else:
        correction = requested_sd * raw / raw_sd
    score = b + correction
    return PooledRoughnessScore(
        score=np.asarray(score, dtype=float),
        correction=np.asarray(correction, dtype=float),
        diagnostics={
            "n_families_present": int(len(families)),
            "raw_correction_sd": raw_sd,
            "requested_correction_sd": requested_sd,
            "correction_sd": float(np.std(correction, ddof=0)),
            "baseline_correction_covariance": float(np.cov(
                b, correction, ddof=0
            )[0, 1]) if np.any(correction) else 0.0,
        },
    )


def direction_cosine(left, right) -> float:
    """Cosine helper with an explicit zero-vector guard."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(left, right) / denom)


__all__ = [
    "GraphRoughnessMoment",
    "PooledRoughnessCalibration",
    "PooledRoughnessScore",
    "align_family_matrix",
    "apply_pooled_roughness",
    "direction_cosine",
    "fit_pooled_roughness_calibration",
    "graph_roughness_moment",
    "pool_graph_roughness_moments",
]
