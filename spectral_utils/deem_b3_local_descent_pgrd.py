"""One-stage local-descent PGRD routing over an exactly frozen B3 ensemble.

This module is a new, target-free experiment.  It does not modify historical
B3 or the earlier residual-MoE/PGRD implementations.

For a cell, ``z`` is the standardized logit of the exact mean-of-five frozen
B3 posterior and ``r_ig`` is a true grouped-fold leave-one-family-out residual
of B3 family ``g``.  A pooled PGRD calibration, fitted after excluding the
target *dataset family*, supplies a signed family coefficient ``v_g``.  The
signed residual expert is

    t_ig = v_g r_ig.

On one fixed target-cell residual graph with normalized Laplacian ``L``, the
primary local router keeps only experts with a negative first-order local
roughness derivative,

    a_ig = stability_g [-t_ig (L z)_i]_+,
    pi_ig = a_ig / sum_h a_ih,
    d_raw_i = sum_g pi_ig t_ig.

Rows with no positive activation abstain (all ``pi_ig`` are zero).  The whole
router is frozen at the B3 base.  After centering and unit-SD normalizing
``d_raw``, a single exact quadratic line step is taken:

    alpha = clip(-d' L z / (d' L d), 0, 0.5 / G).

If the direction is degenerate, non-descent, or numerically invalid, the
method returns the frozen B3 score *verbatim*.  Static and deterministic
row-permuted gate modes are capacity-matched mechanism controls.  All modes
use the same signed expert terms and the same common graph; only row-to-gate
alignment changes.

The method is transductive with respect to target-cell feature geometry.  It
does not claim per-row or end-to-end self-free inference.  No target labels
are accepted anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

from .deem_b3_residual_moe import (
    EPS,
    FAMILY_ORDER,
    GraphRoughnessMoment,
    ResidualCell,
)
from .graph_topology import self_safe_knn_graph
from .laplacian_upcr import graph_diagnostics, symmetric_normalized_laplacian


GATE_MODES = ("alias", "local", "static", "row_permuted")


@dataclass(frozen=True)
class LocalDescentPGRDConfig:
    """Frozen scoring choices for the one-stage experiment."""

    gate_mode: str = "local"
    graph_k: int = 7
    tau_numerator: float = 0.5
    residual_source: str = "loo"
    laplacian_kind: str = "symmetric_normalized"
    direction_pooling: str = "equal_dataset_family"
    correction_centering: str = "mean"
    correction_scaling: str = "unit_sd"


@dataclass(frozen=True)
class LocalDescentCalibration:
    """Pooled PGRD direction fitted without the target dataset family."""

    target_dataset_family: str
    donor_dataset_families: tuple[str, ...]
    direction: np.ndarray
    stability: np.ndarray
    donor_group_directions: np.ndarray
    donor_group_presence: np.ndarray
    family_order: tuple[str, ...] = FAMILY_ORDER
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualGraph:
    """One common, fixed target-cell graph for gating and line search."""

    graph: sparse.csr_matrix
    laplacian: sparse.csr_matrix
    coordinates: np.ndarray
    present_mask: np.ndarray
    tie_keys: np.ndarray
    row_ids: tuple[str, ...]
    family_order: tuple[str, ...]
    binding_sha256: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalDescentPGRDResult:
    """Frozen B3 plus a one-stage graph-descent residual correction."""

    score: np.ndarray
    logit: np.ndarray
    baseline_z: np.ndarray
    local_gradient: np.ndarray
    expert_terms: np.ndarray
    activation: np.ndarray
    local_gate_probabilities: np.ndarray
    gate_probabilities: np.ndarray
    raw_direction: np.ndarray
    direction: np.ndarray
    correction_z: np.ndarray
    row_permutation: np.ndarray
    alpha: float
    tau: float
    cross_term: float
    quadratic_term: float
    roughness_before: float
    roughness_after: float
    family_direction: np.ndarray
    family_stability: np.ndarray
    present_mask: np.ndarray
    family_order: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def validate_config(config: LocalDescentPGRDConfig) -> None:
    if str(config.gate_mode) not in GATE_MODES:
        raise ValueError(f"gate_mode must be one of {GATE_MODES}")
    if int(config.graph_k) != 7:
        raise ValueError("local-descent PGRD v1 is frozen to graph_k=7")
    if float(config.tau_numerator) != 0.5:
        raise ValueError("local-descent PGRD v1 is frozen to tau_numerator=0.5")
    if str(config.residual_source) != "loo":
        raise ValueError("local-descent PGRD v1 requires true LOO residuals")
    if str(config.laplacian_kind) != "symmetric_normalized":
        raise ValueError("local-descent PGRD v1 uses a symmetric normalized Laplacian")
    if str(config.direction_pooling) != "equal_dataset_family":
        raise ValueError("direction pooling must be equal_dataset_family")
    if str(config.correction_centering) != "mean":
        raise ValueError("correction_centering must be mean")
    if str(config.correction_scaling) != "unit_sd":
        raise ValueError("correction_scaling must be unit_sd")


def _validate_moment(moment: GraphRoughnessMoment) -> None:
    if str(moment.residual_source) != "loo":
        raise ValueError("local-descent calibration accepts only LOO moments")
    if moment.a0.shape != (len(FAMILY_ORDER), len(FAMILY_ORDER)):
        raise ValueError("PGRD moment a0 has the wrong shape")
    if moment.c0.shape != (len(FAMILY_ORDER),):
        raise ValueError("PGRD moment c0 has the wrong shape")
    if moment.present_mask.shape != (len(FAMILY_ORDER),):
        raise ValueError("PGRD moment present mask has the wrong shape")
    if not np.isfinite(moment.a0).all() or not np.isfinite(moment.c0).all():
        raise ValueError("PGRD moment is non-finite")


def fit_leave_dataset_family_out_direction(
    moments: Sequence[GraphRoughnessMoment],
    dataset_families: Sequence[str],
    *,
    target_dataset_family: str,
) -> LocalDescentCalibration:
    """Fit the signed PGRD family direction after an exact family holdout.

    Every donor cell moment is trace-normalized by ``graph_roughness_moment``.
    Moments are averaged first within dataset family and then equally across
    donor dataset families.  Coordinate stability is the positive part of
    signed agreement between each donor-family direction and the pooled
    direction.  Missing coordinates do not count as disagreements.
    """

    records = tuple(moments)
    labels = tuple(str(value) for value in dataset_families)
    target = str(target_dataset_family)
    if not records or len(records) != len(labels):
        raise ValueError("moments and dataset_families must be nonempty and aligned")
    for record in records:
        _validate_moment(record)
    selected = [
        (record, label)
        for record, label in zip(records, labels)
        if label != target
    ]
    if not selected:
        raise ValueError("target-family exclusion leaves no donor moments")
    if any(label == target for _, label in selected):
        raise AssertionError("target dataset family crossed the donor boundary")
    donor_groups = tuple(sorted({label for _, label in selected}))
    if len(donor_groups) < 2:
        raise ValueError("at least two donor dataset families are required")

    group_directions = []
    group_presence = []
    group_cell_counts = []
    for group in donor_groups:
        group_records = [record for record, label in selected if label == group]
        group_c = np.mean([record.c0 for record in group_records], axis=0)
        group_direction = -np.asarray(group_c, dtype=np.float64)
        presence = np.any(
            np.row_stack([record.present_mask for record in group_records]), axis=0
        )
        group_direction[~presence] = 0.0
        group_directions.append(group_direction)
        group_presence.append(presence)
        group_cell_counts.append(len(group_records))
    group_directions_array = np.row_stack(group_directions)
    group_presence_array = np.row_stack(group_presence).astype(bool)
    direction = np.mean(group_directions_array, axis=0)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("pooled PGRD direction is negligible")

    stability = np.zeros(len(FAMILY_ORDER), dtype=np.float64)
    for family_index in range(len(FAMILY_ORDER)):
        available = group_presence_array[:, family_index] & (
            np.abs(group_directions_array[:, family_index]) > EPS
        )
        if not np.any(available) or abs(direction[family_index]) <= EPS:
            stability[family_index] = 0.0
            continue
        signed_agreement = np.sign(direction[family_index]) * np.sign(
            group_directions_array[available, family_index]
        )
        stability[family_index] = float(
            np.clip(np.mean(signed_agreement), 0.0, 1.0)
        )
    if not np.any(stability > 0.0):
        raise ValueError("all pooled PGRD coordinates are donor-unstable")

    return LocalDescentCalibration(
        target_dataset_family=target,
        donor_dataset_families=donor_groups,
        direction=np.asarray(direction, dtype=np.float64),
        stability=stability,
        donor_group_directions=group_directions_array,
        donor_group_presence=group_presence_array,
        diagnostics={
            "uses_labels": False,
            "target_dataset_family_excluded": True,
            "target_dataset_family": target,
            "donor_dataset_families": list(donor_groups),
            "n_donor_dataset_families": len(donor_groups),
            "n_donor_cells": len(selected),
            "donor_cell_count_by_dataset_family": dict(
                zip(donor_groups, group_cell_counts)
            ),
            "direction_norm": norm,
            "stable_family_count": int(np.sum(stability > 0.0)),
            "family_stability": stability.tolist(),
        },
    )


def _lexical_tie_keys(row_ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([str(value) for value in row_ids], dtype=str)
    if len(values) != len(set(values.tolist())):
        raise ValueError("row IDs must be unique")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _update_array_digest(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(name).encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _residual_graph_binding_sha256(
    *,
    graph: sparse.spmatrix,
    laplacian: sparse.spmatrix,
    coordinates: np.ndarray,
    present_mask: np.ndarray,
    tie_keys: np.ndarray,
    row_ids: Sequence[str],
    family_order: Sequence[str],
) -> str:
    """Bind graph geometry to ordered rows and residual-family coordinates."""

    graph_csr = sparse.csr_matrix(graph).copy()
    laplacian_csr = sparse.csr_matrix(laplacian).copy()
    graph_csr.sort_indices()
    laplacian_csr.sort_indices()
    digest = hashlib.sha256()
    digest.update(b"deem_b3_local_descent_pgrd_residual_graph_v1")
    for value in row_ids:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for value in family_order:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    _update_array_digest(digest, "coordinates", coordinates)
    _update_array_digest(digest, "present_mask", present_mask)
    _update_array_digest(digest, "tie_keys", tie_keys)
    for prefix, matrix in (("graph", graph_csr), ("laplacian", laplacian_csr)):
        _update_array_digest(digest, f"{prefix}_data", matrix.data)
        _update_array_digest(digest, f"{prefix}_indices", matrix.indices)
        _update_array_digest(digest, f"{prefix}_indptr", matrix.indptr)
        _update_array_digest(digest, f"{prefix}_shape", np.asarray(matrix.shape))
    return digest.hexdigest()


def build_common_residual_graph(
    cell: ResidualCell,
    row_ids: Sequence[str],
    *,
    k: int = 7,
) -> ResidualGraph:
    """Build the one common transductive graph used by every score arm."""

    if int(k) != 7:
        raise ValueError("local-descent PGRD v1 is frozen to k=7")
    present = np.asarray(cell.present_mask, dtype=bool)
    coordinates = np.asarray(cell.loo_residuals[:, present], dtype=np.float64)
    if coordinates.shape[0] != len(row_ids) or coordinates.shape[1] < 3:
        raise ValueError("LOO coordinates and row IDs are not aligned")
    if not np.isfinite(coordinates).all():
        raise ValueError("LOO graph coordinates are non-finite")
    tie_keys = _lexical_tie_keys(row_ids)
    graph = sparse.csr_matrix(
        self_safe_knn_graph(coordinates, k=int(k), tie_keys=tie_keys),
        dtype=np.float64,
    )
    laplacian = sparse.csr_matrix(
        symmetric_normalized_laplacian(graph), dtype=np.float64
    )
    graph.sort_indices()
    laplacian.sort_indices()
    delta = laplacian - laplacian.T
    symmetry_error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    diagnostics = graph_diagnostics(graph, laplacian)
    trace = float(np.sum(laplacian.diagonal()))
    if (
        graph.shape != (len(row_ids), len(row_ids))
        or laplacian.shape != graph.shape
        or diagnostics["n_edges"] <= 0
        or diagnostics["degree_min"] <= 0.0
        or symmetry_error > 1e-10
        or not np.isfinite(graph.data).all()
        or not np.isfinite(laplacian.data).all()
        or abs(trace / len(row_ids) - 1.0) > 1e-12
    ):
        raise ValueError("common residual graph failed health checks")
    present_order = tuple(
        family for family, keep in zip(cell.family_order, present) if keep
    )
    ordered_row_ids = tuple(str(value) for value in row_ids)
    binding_sha256 = _residual_graph_binding_sha256(
        graph=graph,
        laplacian=laplacian,
        coordinates=coordinates,
        present_mask=present,
        tie_keys=tie_keys,
        row_ids=ordered_row_ids,
        family_order=present_order,
    )
    return ResidualGraph(
        graph=graph,
        laplacian=laplacian,
        coordinates=coordinates,
        present_mask=present,
        tie_keys=tie_keys,
        row_ids=ordered_row_ids,
        family_order=present_order,
        binding_sha256=binding_sha256,
        diagnostics={
            **diagnostics,
            "uses_labels": False,
            "transductive_target_cell_geometry": True,
            "per_row_self_free_inference": False,
            "coordinate_source": "true_grouped_fold_loo_b3_family_residuals",
            "topology": "self_safe_self_tuning_union_knn",
            "graph_k": int(k),
            "laplacian": "symmetric_normalized",
            "laplacian_trace": trace,
            "laplacian_trace_per_row": trace / len(row_ids),
            "laplacian_symmetry_max_abs": symmetry_error,
            "family_order": list(present_order),
            "binding_sha256": binding_sha256,
        },
    )


def deterministic_row_permutation(row_ids: Sequence[str]) -> np.ndarray:
    """Return a content-addressed non-identity row permutation."""

    values = tuple(str(value) for value in row_ids)
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("row permutation requires at least two unique row IDs")
    keys = np.asarray(
        [
            hashlib.sha256(
                ("deem_b3_local_descent_pgrd_v1|" + value).encode("utf-8")
            ).hexdigest()
            for value in values
        ],
        dtype=str,
    )
    permutation = np.argsort(keys, kind="mergesort").astype(np.int64)
    if np.array_equal(permutation, np.arange(len(values), dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    return permutation


def _validate_laplacian(laplacian: sparse.spmatrix, n_rows: int) -> sparse.csr_matrix:
    value = sparse.csr_matrix(laplacian, dtype=np.float64).copy()
    value.sort_indices()
    if value.shape != (n_rows, n_rows) or not np.isfinite(value.data).all():
        raise ValueError("Laplacian is not finite/aligned")
    delta = value - value.T
    error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    if error > 1e-10:
        raise ValueError(f"Laplacian is not symmetric: {error:.3e}")
    return value


def _row_normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    denominator = np.sum(values, axis=1)
    active = denominator > EPS
    output[active] = values[active] / denominator[active, None]
    return output


def score_local_descent_pgrd(
    cell: ResidualCell,
    calibration: LocalDescentCalibration,
    residual_graph: ResidualGraph,
    *,
    config: LocalDescentPGRDConfig | None = None,
) -> LocalDescentPGRDResult:
    """Apply one frozen local-descent step or a matched mechanism control."""

    config = config or LocalDescentPGRDConfig()
    validate_config(config)
    if tuple(cell.family_order) != tuple(calibration.family_order):
        raise ValueError("cell and calibration family orders differ")
    expected_shape = (len(FAMILY_ORDER),)
    if (
        np.asarray(calibration.direction).shape != expected_shape
        or np.asarray(calibration.stability).shape != expected_shape
        or not np.isfinite(calibration.direction).all()
        or not np.isfinite(calibration.stability).all()
        or np.any(calibration.stability < 0.0)
        or np.any(calibration.stability > 1.0)
    ):
        raise ValueError("calibration direction/stability is invalid")
    if cell.diagnostics.get("n_present_families") is not None and int(
        cell.diagnostics["n_present_families"]
    ) < 3:
        raise ValueError("at least three present residual families are required")
    n_rows = len(cell.baseline_score)
    present = np.asarray(cell.present_mask, dtype=bool)
    present_order = tuple(
        family for family, keep in zip(cell.family_order, present) if keep
    )
    row_ids = tuple(str(value) for value in residual_graph.row_ids)
    expected_ties = _lexical_tie_keys(row_ids)
    expected_coordinates = np.asarray(
        cell.loo_residuals[:, present], dtype=np.float64
    )
    if (
        len(row_ids) != n_rows
        or not np.array_equal(residual_graph.present_mask, present)
        or tuple(residual_graph.family_order) != present_order
        or not np.array_equal(residual_graph.tie_keys, expected_ties)
        or residual_graph.coordinates.shape != expected_coordinates.shape
        or not np.array_equal(residual_graph.coordinates, expected_coordinates)
    ):
        raise ValueError("residual graph is not bound to this cell/row order")
    actual_graph_binding = _residual_graph_binding_sha256(
        graph=residual_graph.graph,
        laplacian=residual_graph.laplacian,
        coordinates=residual_graph.coordinates,
        present_mask=residual_graph.present_mask,
        tie_keys=residual_graph.tie_keys,
        row_ids=row_ids,
        family_order=residual_graph.family_order,
    )
    if (
        not isinstance(residual_graph.binding_sha256, str)
        or len(residual_graph.binding_sha256) != 64
        or actual_graph_binding != residual_graph.binding_sha256
        or residual_graph.diagnostics.get("binding_sha256")
        != residual_graph.binding_sha256
    ):
        raise ValueError("residual graph binding hash mismatch")
    L = _validate_laplacian(residual_graph.laplacian, n_rows)
    count = int(np.sum(present))
    tau = float(config.tau_numerator) / count
    baseline_z = np.asarray(cell.baseline_z, dtype=np.float64)
    if not np.isfinite(baseline_z).all():
        raise ValueError("baseline z coordinate is non-finite")

    family_direction = np.zeros(len(FAMILY_ORDER), dtype=np.float64)
    local_norm = float(np.linalg.norm(calibration.direction[present]))
    if not np.isfinite(local_norm) or local_norm <= EPS:
        raise ValueError("calibration has no nonzero direction on present families")
    family_direction[present] = calibration.direction[present] / local_norm
    family_stability = np.zeros(len(FAMILY_ORDER), dtype=np.float64)
    family_stability[present] = np.clip(
        calibration.stability[present], 0.0, 1.0
    )
    expert_terms = np.asarray(cell.loo_residuals, dtype=np.float64) * (
        family_direction[None, :]
    )
    local_gradient = np.asarray(L @ baseline_z, dtype=np.float64)
    activation = family_stability[None, :] * np.maximum(
        -local_gradient[:, None] * expert_terms, 0.0
    )
    activation[:, ~present] = 0.0
    local_probabilities = _row_normalize_nonnegative(activation)

    permutation = np.arange(n_rows, dtype=np.int64)
    if config.gate_mode == "alias":
        probabilities = np.zeros_like(local_probabilities)
    elif config.gate_mode == "local":
        probabilities = local_probabilities.copy()
    elif config.gate_mode == "static":
        mean_gate = np.mean(local_probabilities, axis=0)
        probabilities = np.repeat(mean_gate[None, :], n_rows, axis=0)
    elif config.gate_mode == "row_permuted":
        permutation = deterministic_row_permutation(row_ids)
        probabilities = local_probabilities[permutation]
    else:  # guarded by validate_config
        raise AssertionError(config.gate_mode)

    raw_direction = np.sum(probabilities * expert_terms, axis=1)
    centered_direction = raw_direction - float(np.mean(raw_direction))
    raw_sd = float(np.std(centered_direction))
    if raw_sd > EPS and config.gate_mode != "alias":
        direction = centered_direction / raw_sd
    else:
        direction = np.zeros(n_rows, dtype=np.float64)

    Lz = local_gradient
    Ld = np.asarray(L @ direction, dtype=np.float64)
    cross_term = float(np.dot(direction, Lz))
    quadratic_term = float(np.dot(direction, Ld))
    roughness_before = float(np.dot(baseline_z, Lz))
    alpha = 0.0
    if (
        config.gate_mode != "alias"
        and raw_sd > EPS
        and np.isfinite(cross_term)
        and np.isfinite(quadratic_term)
        and cross_term < 0.0
        and quadratic_term > EPS
    ):
        alpha = float(np.clip(-cross_term / quadratic_term, 0.0, tau))
    correction_z = alpha * direction
    updated_z = baseline_z + correction_z
    roughness_after = float(np.dot(updated_z, np.asarray(L @ updated_z)))
    tolerance = 1e-10 * max(1.0, abs(roughness_before))
    if alpha > 0.0 and roughness_after > roughness_before + tolerance:
        raise AssertionError("accepted line step increased common-graph roughness")

    updated_logit = np.asarray(cell.baseline_logit, dtype=np.float64) + (
        float(cell.baseline_scale) * correction_z
    )
    if alpha == 0.0:
        score = np.asarray(cell.baseline_score, dtype=np.float64).copy()
        updated_logit = np.asarray(cell.baseline_logit, dtype=np.float64).copy()
    else:
        score = expit(updated_logit)
    reconstruction_error = float(
        np.max(
            np.abs(
                updated_logit
                - (
                    np.asarray(cell.baseline_logit, dtype=np.float64)
                    + float(cell.baseline_scale) * correction_z
                )
            )
        )
    )
    row_mass = np.sum(probabilities, axis=1)
    local_row_mass = np.sum(local_probabilities, axis=1)
    result = LocalDescentPGRDResult(
        score=score,
        logit=updated_logit,
        baseline_z=baseline_z.copy(),
        local_gradient=local_gradient,
        expert_terms=expert_terms,
        activation=activation,
        local_gate_probabilities=local_probabilities,
        gate_probabilities=probabilities,
        raw_direction=raw_direction,
        direction=direction,
        correction_z=correction_z,
        row_permutation=permutation,
        alpha=alpha,
        tau=tau,
        cross_term=cross_term,
        quadratic_term=quadratic_term,
        roughness_before=roughness_before,
        roughness_after=roughness_after,
        family_direction=family_direction,
        family_stability=family_stability,
        present_mask=present,
        family_order=tuple(cell.family_order),
        diagnostics={
            "uses_labels": False,
            "gate_mode": str(config.gate_mode),
            "one_stage_only": True,
            "gates_frozen_at_baseline": True,
            "common_graph_for_gate_and_line_step": True,
            "transductive_target_cell_geometry": True,
            "per_row_self_free_inference": False,
            "exact_b3_alias": bool(alpha == 0.0),
            "n_rows": n_rows,
            "n_present_families": count,
            "tau": tau,
            "alpha": alpha,
            "alpha_within_bounds": bool(0.0 <= alpha <= tau),
            "raw_direction_sd": raw_sd,
            "direction_sd": float(np.std(direction)),
            "correction_sd": float(np.std(correction_z)),
            "cross_term": cross_term,
            "quadratic_term": quadratic_term,
            "roughness_before": roughness_before,
            "roughness_after": roughness_after,
            "roughness_delta": roughness_after - roughness_before,
            "roughness_nonincreasing": bool(
                roughness_after <= roughness_before + tolerance
            ),
            "local_active_row_fraction": float(np.mean(local_row_mass > 0.0)),
            "applied_active_row_fraction": float(np.mean(row_mass > 0.0)),
            "local_gate_zero_fraction_present": float(
                np.mean(local_probabilities[:, present] == 0.0)
            ),
            "applied_gate_zero_fraction_present": float(
                np.mean(probabilities[:, present] == 0.0)
            ),
            "gate_row_sum_min": float(np.min(row_mass)),
            "gate_row_sum_max": float(np.max(row_mass)),
            "logit_reconstruction_max_abs": reconstruction_error,
            "score_finite": bool(np.isfinite(score).all()),
        },
    )
    if not result.diagnostics["score_finite"]:
        raise ValueError("local-descent score is non-finite")
    if config.gate_mode == "alias" and not np.array_equal(
        result.score, cell.baseline_score
    ):
        raise AssertionError("alias mode did not preserve frozen B3 bytes")
    return result


__all__ = [
    "GATE_MODES",
    "LocalDescentCalibration",
    "LocalDescentPGRDConfig",
    "LocalDescentPGRDResult",
    "ResidualGraph",
    "build_common_residual_graph",
    "deterministic_row_permutation",
    "fit_leave_dataset_family_out_direction",
    "score_local_descent_pgrd",
    "validate_config",
]
