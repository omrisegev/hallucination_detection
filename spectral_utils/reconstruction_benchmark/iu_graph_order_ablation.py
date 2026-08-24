"""Label-free graph-order ablations around canonical mixed-v2 IU-PCR.

The module intentionally lives outside the 13-method roster.  It answers one
mechanism question with the same answer graph held fixed: either smooth the
feature matrix and refit IU-PCR, smooth the IU score itself, or restrict the
change to the canonical IU-orthogonal family-residual coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .contracts import PreparedCell, canonical_sha256
from .methods import (
    _iu_contribution_coordinates,
    _make_context,
    _orientation_multiplier,
    _row_tie_keys,
    _usable_family_residuals,
)
from ..graph_topology import self_safe_knn_graph
from ..laplacian_upcr import (
    IU_FIT_DEFAULTS,
    graph_diagnostics,
    symmetric_normalized_laplacian,
)
from ..upcr import upcr_fit


CONFIG_SCHEMA_VERSION = "iu-graph-order-ablation-config-v1"
FIT_SCHEMA_VERSION = "iu-graph-order-ablation-fit-v1"
_EPS = 1e-12


@dataclass(frozen=True)
class CellAblationResult:
    """One target-free fitted score bank."""

    cell_id: str
    row_ids: tuple[str, ...]
    prepared_matrix_sha256: str
    scores: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]


def lambda_token(value: float) -> str:
    """Stable arm-name token for one positive regularization strength."""

    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError("lambda must be finite and positive")
    return format(number, ".12g").replace(".", "p").replace("-", "m")


def arm_id(family: str, value: float) -> str:
    return f"{family}__lam_{lambda_token(value)}"


def expected_arm_ids(lambdas: Sequence[float]) -> tuple[str, ...]:
    values = tuple(float(value) for value in lambdas)
    if len(values) != len(set(values)) or any(value <= 0 for value in values):
        raise ValueError("lambda grid must contain unique positive values")
    output = ["iu_pcr", "equal_family_mean"]
    for value in values:
        output.extend(
            arm_id(family, value)
            for family in (
                "feature_smooth_residual_graph",
                "feature_smooth_raw_graph",
                "score_smooth_residual_graph",
                "residual_ridge_correction",
            )
        )
    return tuple(output)


def validate_config(config: Mapping[str, object]) -> tuple[float, ...]:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unexpected graph-order ablation config schema")
    if config.get("feature_contract_id") != (
        "dufs-liu-mixed-v2-development-2026-08-07"
    ):
        raise ValueError("graph-order ablation requires the canonical mixed-v2 contract")
    if config.get("runtime_labels_used") is not False:
        raise ValueError("fit config must explicitly forbid runtime targets")
    if int(config.get("graph_k", -1)) != 7:
        raise ValueError("v1 freezes graph_k=7")
    values = tuple(float(value) for value in config.get("lambda_grid", ()))
    expected_arm_ids(values)
    if float(config.get("primary_lambda", np.nan)) not in values:
        raise ValueError("primary lambda is absent from the frozen grid")
    if float(config.get("historical_mechanism_lambda", np.nan)) not in values:
        raise ValueError("historical mechanism lambda is absent from the frozen grid")
    return values


def config_sha256(config: Mapping[str, object]) -> str:
    validate_config(config)
    return canonical_sha256(config)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    output = values - means
    usable = scales > _EPS
    output[:, usable] /= scales[usable]
    output[:, ~usable] = 0.0
    if not np.isfinite(output).all():
        raise RuntimeError("column standardization produced non-finite values")
    return output


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    score = np.asarray(values, dtype=np.float64)
    scale = float(score.std(ddof=0))
    if not np.isfinite(scale) or scale <= _EPS:
        raise RuntimeError("score is constant after graph operation")
    output = (score - float(score.mean())) / scale
    if not np.isfinite(output).all():
        raise RuntimeError("score standardization produced non-finite values")
    return output


def _sparse_hash(matrix: sparse.spmatrix) -> str:
    value = sparse.csr_matrix(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"shape": list(value.shape), "dtype": "float64-le"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii"))
    for array in (value.indptr.astype("<i8"), value.indices.astype("<i8"), value.data.astype("<f8")):
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _smooth(matrix: np.ndarray, laplacian: sparse.spmatrix, value: float) -> np.ndarray:
    if value <= 0:
        return np.asarray(matrix, dtype=np.float64).copy()
    n = int(laplacian.shape[0])
    operator = sparse.identity(n, format="csc", dtype=np.float64) + float(value) * sparse.csc_matrix(laplacian)
    solver = splu(operator)
    source = np.asarray(matrix, dtype=np.float64)
    if source.ndim == 1:
        return np.asarray(solver.solve(source), dtype=np.float64)
    return np.column_stack([
        np.asarray(solver.solve(source[:, index]), dtype=np.float64)
        for index in range(source.shape[1])
    ])


def _iu_confidence(matrix: np.ndarray, confidence_anchor: np.ndarray) -> tuple[np.ndarray, dict]:
    values = _zscore_columns(matrix)
    fit = upcr_fit(values.T, **IU_FIT_DEFAULTS)
    raw = np.asarray(fit.w @ values.T, dtype=np.float64)
    multiplier, correlation = _orientation_multiplier(raw, confidence_anchor)
    aligned = _zscore_vector(multiplier * raw)
    return aligned, {
        "orientation_multiplier": multiplier,
        "orientation_correlation": correlation,
        "n_components_used": int(fit.n_components_used),
        "g2_hat": float(fit.g2_hat),
        "var_y": float(fit.var_y),
    }


def fit_cell(cell: PreparedCell, config: Mapping[str, object]) -> CellAblationResult:
    """Fit every frozen arm without reading targets or source groups."""

    lambdas = validate_config(config)
    context = _make_context(cell)
    coordinates = _iu_contribution_coordinates(context)
    families, residuals, _, dropped = _usable_family_residuals(coordinates)
    if residuals.shape[1] < 3:
        raise RuntimeError("matched ablation needs at least three non-degenerate families")

    tie_keys = _row_tie_keys(cell.row_ids)
    residual_graph = self_safe_knn_graph(
        residuals,
        k=int(config["graph_k"]),
        tie_keys=tie_keys,
    )
    raw_graph = self_safe_knn_graph(
        cell.matrix,
        k=int(config["graph_k"]),
        tie_keys=tie_keys,
    )
    residual_laplacian = symmetric_normalized_laplacian(residual_graph)
    raw_laplacian = symmetric_normalized_laplacian(raw_graph)

    baseline = _zscore_vector(np.asarray(coordinates.baseline, dtype=np.float64))
    equal_family = _zscore_vector(np.asarray(context.confidence_anchor, dtype=np.float64))
    recomputed_iu, iu_diagnostics = _iu_confidence(cell.matrix, context.confidence_anchor)
    if not np.allclose(recomputed_iu, baseline, atol=1e-9, rtol=1e-9):
        raise RuntimeError("canonical IU baseline disagrees with contribution-space IU")

    n = residuals.shape[0]
    moment_a = np.asarray(residuals.T @ (residual_laplacian @ residuals) / n, dtype=np.float64)
    moment_a = 0.5 * (moment_a + moment_a.T)
    moment_c = np.asarray(residuals.T @ (residual_laplacian @ baseline) / n, dtype=np.float64)

    confidence_scores: dict[str, np.ndarray] = {
        "iu_pcr": baseline,
        "equal_family_mean": equal_family,
    }
    per_arm: dict[str, object] = {
        "iu_pcr": iu_diagnostics,
        "equal_family_mean": {"n_families": len(context.family_members)},
    }
    eye = np.eye(residuals.shape[1], dtype=np.float64)
    for value in lambdas:
        # One factorization serves both the feature matrix and the direct
        # score-smoothing control at this lambda.
        residual_joint = _smooth(
            np.column_stack([cell.matrix, baseline]),
            residual_laplacian,
            value,
        )
        residual_smoothed = _zscore_columns(residual_joint[:, :-1])
        raw_smoothed = _zscore_columns(_smooth(cell.matrix, raw_laplacian, value))
        residual_iu, residual_iu_diag = _iu_confidence(
            residual_smoothed, context.confidence_anchor
        )
        raw_iu, raw_iu_diag = _iu_confidence(raw_smoothed, context.confidence_anchor)
        smoothed_score = _zscore_vector(residual_joint[:, -1])
        delta = -float(value) * np.linalg.solve(
            eye + float(value) * moment_a,
            moment_c,
        )
        corrected_raw = baseline + residuals @ delta
        corrected = _zscore_vector(corrected_raw)

        residual_feature_id = arm_id("feature_smooth_residual_graph", value)
        raw_feature_id = arm_id("feature_smooth_raw_graph", value)
        score_smooth_id = arm_id("score_smooth_residual_graph", value)
        correction_id = arm_id("residual_ridge_correction", value)
        confidence_scores[residual_feature_id] = residual_iu
        confidence_scores[raw_feature_id] = raw_iu
        confidence_scores[score_smooth_id] = smoothed_score
        confidence_scores[correction_id] = corrected
        per_arm[residual_feature_id] = residual_iu_diag
        per_arm[raw_feature_id] = raw_iu_diag
        per_arm[score_smooth_id] = {
            "roughness_before": float(baseline @ (residual_laplacian @ baseline) / n),
            "roughness_after": float(smoothed_score @ (residual_laplacian @ smoothed_score) / n),
        }
        per_arm[correction_id] = {
            "delta": delta.tolist(),
            "delta_norm": float(np.linalg.norm(delta)),
            "correction_sd": float(np.std(residuals @ delta, ddof=0)),
            "roughness_before": float(baseline @ (residual_laplacian @ baseline) / n),
            "roughness_after": float(corrected @ (residual_laplacian @ corrected) / n),
            "objective_before": float(value) / (2.0 * n) * float(
                baseline @ (residual_laplacian @ baseline)
            ),
            "objective_after": 0.5 * float(delta @ delta) + float(value) / (2.0 * n) * float(
                corrected_raw @ (residual_laplacian @ corrected_raw)
            ),
        }

    if tuple(confidence_scores) != expected_arm_ids(lambdas):
        raise RuntimeError("ablation score roster drifted")
    risk_scores = {
        name: np.asarray(-score, dtype=np.float64)
        for name, score in confidence_scores.items()
    }
    diagnostics = {
        "schema_version": FIT_SCHEMA_VERSION,
        "config_sha256": config_sha256(config),
        "families": list(families),
        "dropped_degenerate_families": list(dropped),
        "residual_graph": graph_diagnostics(residual_graph, residual_laplacian),
        "raw_graph": graph_diagnostics(raw_graph, raw_laplacian),
        "residual_graph_sha256": _sparse_hash(residual_graph),
        "residual_laplacian_sha256": _sparse_hash(residual_laplacian),
        "raw_graph_sha256": _sparse_hash(raw_graph),
        "raw_laplacian_sha256": _sparse_hash(raw_laplacian),
        "moment_a": moment_a.tolist(),
        "moment_c": moment_c.tolist(),
        "arms": per_arm,
        "runtime_labels_used": False,
    }
    return CellAblationResult(
        cell_id=cell.cell_id,
        row_ids=cell.row_ids,
        prepared_matrix_sha256=cell.matrix_sha256,
        scores=risk_scores,
        diagnostics=diagnostics,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "FIT_SCHEMA_VERSION",
    "CellAblationResult",
    "arm_id",
    "config_sha256",
    "expected_arm_ids",
    "fit_cell",
    "lambda_token",
    "validate_config",
]
