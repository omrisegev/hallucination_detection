"""Label-free graph geometry construction and intrinsic selection helpers.

This module is intentionally outcome blind.  It accepts feature matrices,
IU/family-residual states, and graph moments, but never correctness labels.
The geometry roster and the intrinsic selector are frozen here so that the
fit process can materialize and hash its choices before a reporting process
is allowed to open outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.covariance import LedoitWolf

from .family_residual_graph import (
    FamilyResidualState,
    graphs_from_coordinates,
)
from .graph_topology import extended_graph_diagnostics
from .laplacian_upcr import symmetric_normalized_laplacian
from .pooled_graph_roughness import (
    GraphRoughnessMoment,
    direction_cosine,
    fit_pooled_roughness_calibration,
    pool_graph_roughness_moments,
)


EPS = 1e-12
INTRINSIC_LAMBDA = 0.03
INTRINSIC_TRUST = 0.5
INTRINSIC_RULE_VERSION = "hard-validity-then-stability-v1-2026-08-23"


@dataclass(frozen=True)
class GeometrySpec:
    """One prespecified graph construction."""

    geometry_id: str
    representation: str
    metric: str
    topology: str
    k: int
    phase_a: bool
    rationale: str


# Priority is deliberate and is also the deterministic deduplication tie rule.
# The first four entries are exactly the Phase-A capacity class.
GEOMETRIES = (
    GeometrySpec(
        "residual_union_k7", "residual", "euclidean", "union", 7, True,
        "canonical duplicate-safe residual union-kNN anchor",
    ),
    GeometrySpec(
        "residual_union_k5", "residual", "euclidean", "union", 5, True,
        "smaller residual neighbourhood scale",
    ),
    GeometrySpec(
        "residual_union_k15", "residual", "euclidean", "union", 15, True,
        "larger residual neighbourhood scale",
    ),
    GeometrySpec(
        "residual_adaptive_k7", "residual", "euclidean", "adaptive", 7, True,
        "reviewed density-adaptive residual topology at mean k=7",
    ),
    GeometrySpec(
        "residual_mutual_k7", "residual", "euclidean", "mutual", 7, False,
        "mutual-neighbour topology control",
    ),
    GeometrySpec(
        "contribution_union_k7", "contribution", "euclidean", "union", 7, False,
        "unresidualized standardized family-contribution control",
    ),
    GeometrySpec(
        "dufs_union_k7", "dufs", "euclidean", "union", 7, False,
        "historical target-free DUFS-coordinate control",
    ),
    GeometrySpec(
        "residual_cosine_union_k7", "residual", "cosine", "union", 7, False,
        "angular residual metric without expanding topology or scale",
    ),
    GeometrySpec(
        "residual_shrinkage_mahalanobis_union_k7",
        "residual", "shrinkage_mahalanobis", "union", 7, False,
        "label-free Ledoit-Wolf residual metric at canonical capacity",
    ),
)
CONTROL_ONLY_GEOMETRY_IDS = ("dufs_union_k7",)


def geometry_index() -> dict[str, GeometrySpec]:
    return {spec.geometry_id: spec for spec in GEOMETRIES}


def phase_a_geometry_ids() -> tuple[str, ...]:
    return tuple(spec.geometry_id for spec in GEOMETRIES if spec.phase_a)


def selector_geometry_ids() -> tuple[str, ...]:
    """Compact Phase-B bank, excluding registered coordinate controls."""
    return tuple(
        spec.geometry_id for spec in GEOMETRIES
        if spec.geometry_id not in CONTROL_ONLY_GEOMETRY_IDS
    )


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def validate_physically_label_free_members(
    members: Iterable[str], expected_cells: Sequence[str]
) -> None:
    """Reject a bundle that even contains a target-like member.

    The allowed list is intentionally narrower than a substring blacklist:
    graph fitting only needs the raw view matrix, feature names, and legacy
    orientation signs for each registered cell.
    """

    members = set(map(str, members))
    allowed = {
        f"{cell}__{suffix}"
        for cell in expected_cells
        for suffix in ("V", "pool", "hand_signs")
    }
    if members != allowed:
        forbidden = sorted(members - allowed)
        missing = sorted(allowed - members)
        raise RuntimeError(
            "label-free bundle member registry mismatch: "
            f"forbidden_or_extra={forbidden}, missing={missing}"
        )


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms <= EPS] = 1.0
    return values / norms


def _shrinkage_whiten(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values, axis=0, keepdims=True)
    covariance = np.asarray(
        LedoitWolf(assume_centered=True).fit(centred).covariance_, dtype=float
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-10)
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    return centred @ inverse_root


def geometry_base_coordinates(
    spec: GeometrySpec,
    F: np.ndarray,
    state: FamilyResidualState,
    *,
    dufs_gates: np.ndarray | None,
) -> np.ndarray:
    """Return the representation coordinates before metric transformation."""

    if spec.representation == "residual":
        coordinates = np.asarray(state.residuals, dtype=float)
    elif spec.representation == "contribution":
        coordinates = np.asarray(state.standardized_contributions, dtype=float)
    elif spec.representation == "dufs":
        if dufs_gates is None:
            raise ValueError("DUFS geometry requires prefit label-free gates")
        gates = np.asarray(dufs_gates, dtype=float)
        if gates.shape != (F.shape[0],):
            raise ValueError("DUFS gates do not match the feature matrix")
        coordinates = np.asarray(F.T * gates[None, :], dtype=float)
    else:
        raise ValueError(f"unknown representation: {spec.representation}")

    if coordinates.ndim != 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"invalid base coordinates for {spec.geometry_id}")
    return np.asarray(coordinates, dtype=float)


def transform_geometry_metric(
    spec: GeometrySpec, coordinates: np.ndarray
) -> np.ndarray:
    """Fit/apply the target-free metric transform on one sample population."""
    coordinates = np.asarray(coordinates, dtype=float)
    if spec.metric == "euclidean":
        transformed = coordinates
    elif spec.metric == "cosine":
        transformed = _row_normalize(coordinates)
    elif spec.metric == "shrinkage_mahalanobis":
        transformed = _shrinkage_whiten(coordinates)
    else:
        raise ValueError(f"unknown metric: {spec.metric}")
    if transformed.ndim != 2 or not np.isfinite(transformed).all():
        raise ValueError(f"invalid coordinates for {spec.geometry_id}")
    return np.asarray(transformed, dtype=float)


def geometry_coordinates(
    spec: GeometrySpec,
    F: np.ndarray,
    state: FamilyResidualState,
    *,
    dufs_gates: np.ndarray | None,
) -> np.ndarray:
    """Return deterministic sample coordinates in the advertised metric."""
    base = geometry_base_coordinates(
        spec, F, state, dufs_gates=dufs_gates
    )
    return transform_geometry_metric(spec, base)


def graph_from_transformed_coordinates(
    spec: GeometrySpec,
    coordinates: np.ndarray,
    *,
    tie_keys: np.ndarray | None = None,
) -> csr_matrix:
    coordinates = np.asarray(coordinates, dtype=float)
    if tie_keys is None:
        tie_keys = np.arange(len(coordinates), dtype=float)
    return graphs_from_coordinates(
        coordinates,
        (spec.k,),
        topology=spec.topology,
        tie_keys=np.asarray(tie_keys, dtype=float),
    )[spec.k]


def build_geometry_graph(
    spec: GeometrySpec,
    F: np.ndarray,
    state: FamilyResidualState,
    *,
    dufs_gates: np.ndarray | None,
    tie_keys: np.ndarray | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    coordinates = geometry_coordinates(spec, F, state, dufs_gates=dufs_gates)
    graph = graph_from_transformed_coordinates(
        spec, coordinates, tie_keys=tie_keys
    )
    return graph, coordinates


def _edge_keys(graph: csr_matrix) -> np.ndarray:
    graph = csr_matrix(graph)
    rows, cols = graph.nonzero()
    keep = rows < cols
    n = graph.shape[0]
    return np.sort(rows[keep].astype(np.int64) * np.int64(n) + cols[keep])


def edge_jaccard(left: csr_matrix, right: csr_matrix) -> float:
    if left.shape != right.shape:
        raise ValueError("graphs have different shapes")
    a, b = _edge_keys(left), _edge_keys(right)
    union = np.union1d(a, b)
    if not len(union):
        return 1.0
    return float(len(np.intersect1d(a, b, assume_unique=True)) / len(union))


def operator_cosine(left: csr_matrix, right: csr_matrix) -> float:
    if left.shape != right.shape:
        raise ValueError("graphs have different shapes")
    A = symmetric_normalized_laplacian(left).tocsr()
    B = symmetric_normalized_laplacian(right).tocsr()
    numerator = float(A.multiply(B).sum())
    denominator = float(np.sqrt(A.multiply(A).sum() * B.multiply(B).sum()))
    return float(numerator / denominator) if denominator > EPS else float("nan")


def graph_energy(graph: csr_matrix, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values)
    scale = float(np.std(centred))
    if scale <= EPS:
        return 0.0
    z = centred / scale
    L = symmetric_normalized_laplacian(graph)
    return float(z @ (L @ z) / len(z))


def perturbation_diagnostics(
    spec: GeometrySpec,
    base_coordinates: np.ndarray,
    graph: csr_matrix,
    *,
    cell: str,
) -> dict:
    """Graph stability under three deterministic, target-free perturbations."""

    base_coordinates = np.asarray(base_coordinates, dtype=float)
    coordinates = transform_geometry_metric(spec, base_coordinates)
    n, p = base_coordinates.shape
    rng = stable_rng(INTRINSIC_RULE_VERSION, cell, spec.geometry_id)
    scale = np.std(base_coordinates, axis=0, ddof=0)
    scale[scale <= EPS] = 1.0
    jitter_base = (
        base_coordinates
        + rng.normal(size=base_coordinates.shape) * scale[None, :] * 1e-3
    )
    jitter_graph = graph_from_transformed_coordinates(
        spec, transform_geometry_metric(spec, jitter_base)
    )

    keep_n = max(spec.k + 2, int(np.floor(0.80 * n)))
    keep = np.sort(rng.choice(n, size=min(n, keep_n), replace=False))
    subset_graph = graph_from_transformed_coordinates(
        spec,
        transform_geometry_metric(spec, base_coordinates[keep]),
        tie_keys=keep.astype(float),
    )
    induced = graph[keep][:, keep].tocsr()

    loo_values = []
    if p > 1:
        loo_indices = list(range(p))
        for coordinate in loo_indices:
            reduced = np.delete(base_coordinates, coordinate, axis=1)
            loo_graph = graph_from_transformed_coordinates(
                spec, transform_geometry_metric(spec, reduced)
            )
            loo_values.append(edge_jaccard(graph, loo_graph))
    else:
        loo_indices = []
        loo_values = [1.0]
    return {
        "jitter_edge_jaccard": edge_jaccard(graph, jitter_graph),
        "subsample_induced_edge_jaccard": edge_jaccard(induced, subset_graph),
        "coordinate_loo_edge_jaccard_median": float(np.median(loo_values)),
        "coordinate_loo_indices": loo_indices,
        "minimum_stability": float(min(
            edge_jaccard(graph, jitter_graph),
            edge_jaccard(induced, subset_graph),
            min(loo_values),
        )),
    }


def aggregate_geometry_similarity(
    graphs_by_cell: dict[str, dict[str, csr_matrix]],
    geometry_ids: Sequence[str],
) -> dict:
    output = {}
    cells = tuple(sorted(graphs_by_cell))
    for left_index, left in enumerate(geometry_ids):
        for right in geometry_ids[left_index:]:
            edge = [
                edge_jaccard(graphs_by_cell[cell][left], graphs_by_cell[cell][right])
                for cell in cells
            ]
            operator = [
                operator_cosine(graphs_by_cell[cell][left], graphs_by_cell[cell][right])
                for cell in cells
            ]
            output[f"{left}__vs__{right}"] = {
                "edge_jaccard_mean": float(np.mean(edge)),
                "edge_jaccard_min": float(np.min(edge)),
                "operator_cosine_mean": float(np.mean(operator)),
                "operator_cosine_min": float(np.min(operator)),
            }
    return output


def deduplicate_geometries(
    similarity: dict,
    geometry_ids: Sequence[str],
    *,
    edge_threshold: float = 0.995,
    operator_threshold: float = 0.999,
) -> tuple[tuple[str, ...], dict[str, str]]:
    representatives: list[str] = []
    duplicate_of: dict[str, str] = {}
    for candidate in geometry_ids:
        match = None
        for representative in representatives:
            key = f"{representative}__vs__{candidate}"
            if key not in similarity:
                key = f"{candidate}__vs__{representative}"
            values = similarity[key]
            if (
                values["edge_jaccard_mean"] >= edge_threshold
                and values["operator_cosine_mean"] >= operator_threshold
            ):
                match = representative
                break
        if match is None:
            representatives.append(candidate)
        else:
            duplicate_of[candidate] = match
    return tuple(representatives), duplicate_of


def _moment_dispersion(
    moments: Sequence[GraphRoughnessMoment], groups: Sequence[str]
) -> float:
    unique = tuple(sorted(set(groups)))
    vectors = []
    for group in unique:
        selected = [m for m, g in zip(moments, groups) if g == group]
        A = np.mean([m.A for m in selected], axis=0)
        c = np.mean([m.c for m in selected], axis=0)
        vectors.append(np.r_[A.ravel(), c])
    values = np.asarray(vectors, dtype=float)
    centre = np.mean(values, axis=0)
    denominator = max(float(np.linalg.norm(centre)), EPS)
    return float(np.mean(np.linalg.norm(values - centre[None, :], axis=1)) / denominator)


def intrinsic_geometry_summary(
    cells: Sequence[dict],
    geometry_id: str,
    *,
    excluded_groups: Sequence[str] = (),
) -> dict:
    """Summarize one geometry using no outcomes."""

    excluded = set(map(str, excluded_groups))
    selected = [cell for cell in cells if cell["group"] not in excluded]
    moments = [cell["moments"][geometry_id] for cell in selected]
    groups = [cell["group"] for cell in selected]
    calibration = fit_pooled_roughness_calibration(
        moments, groups, INTRINSIC_LAMBDA, pooling="equal_group"
    )
    unique_groups = tuple(sorted(set(groups)))
    cosines = []
    for held in unique_groups:
        subset = [
            (moment, group)
            for moment, group in zip(moments, groups) if group != held
        ]
        leave_one = fit_pooled_roughness_calibration(
            [item[0] for item in subset],
            [item[1] for item in subset],
            INTRINSIC_LAMBDA,
            pooling="equal_group",
        )
        cosines.append(direction_cosine(calibration.direction, leave_one.direction))
    health = [cell["graph_diagnostics"][geometry_id] for cell in selected]
    perturb = [cell["perturbation_diagnostics"][geometry_id] for cell in selected]
    length_ratio = [
        cell["length_energy_ratio"][geometry_id] for cell in selected
        if cell["length_energy_ratio"][geometry_id] is not None
        and np.isfinite(cell["length_energy_ratio"][geometry_id])
    ]
    finite_cosines = [value for value in cosines if np.isfinite(value)]
    valid = bool(
        all(value["all_edge_weights_finite"] for value in health)
        and max(value["isolated_fraction"] for value in health) == 0.0
        # V3 made the earlier every-cell 95% connectivity rule diagnostic
        # after it rejected every reviewed graph.  Keep a coarse collapse
        # guard here and report the full connectedness distribution.
        and min(value["largest_component_fraction"] for value in health) >= 0.50
        and bool(length_ratio)
        and min(length_ratio) >= 0.10
        and float(np.median(length_ratio)) >= 0.50
        and finite_cosines
    )
    return {
        "geometry_id": geometry_id,
        "excluded_groups": sorted(excluded),
        "valid": valid,
        "health": {
            "maximum_isolated_fraction": float(max(v["isolated_fraction"] for v in health)),
            "minimum_largest_component_fraction": float(min(v["largest_component_fraction"] for v in health)),
            "median_degree_p90": float(np.median([v["binary_degree_p90"] for v in health])),
        },
        "minimum_perturbation_stability": float(min(v["minimum_stability"] for v in perturb)),
        "median_perturbation_stability": float(np.median([v["minimum_stability"] for v in perturb])),
        "minimum_direction_cosine": float(min(finite_cosines)),
        "mean_direction_cosine": float(np.mean(finite_cosines)),
        "moment_dispersion": _moment_dispersion(moments, groups),
        "predicted_roughness_decrease": float(-np.dot(calibration.c, calibration.direction)),
        "length_energy_ratio_available_cells": len(length_ratio),
        "length_energy_ratio_missing_cells": len(selected) - len(length_ratio),
        "length_energy_ratio_min": float(min(length_ratio)),
        "length_energy_ratio_median": float(np.median(length_ratio)),
        "direction": calibration.direction.tolist(),
    }


def choose_intrinsic_geometry(summaries: Sequence[dict]) -> tuple[str, dict]:
    """Frozen hard filter followed by a deterministic lexicographic rule."""

    valid = [summary for summary in summaries if summary["valid"]]
    if not valid:
        raise RuntimeError("no geometry passed the frozen intrinsic hard-validity filter")
    pool = valid
    selected = max(
        pool,
        key=lambda value: (
            value["minimum_perturbation_stability"],
            value["minimum_direction_cosine"],
            -value["moment_dispersion"],
            value["predicted_roughness_decrease"],
            -next(
                index for index, spec in enumerate(GEOMETRIES)
                if spec.geometry_id == value["geometry_id"]
            ),
        ),
    )
    return selected["geometry_id"], {
        "rule_version": INTRINSIC_RULE_VERSION,
        "hard_validity": {
            "all_weights_finite": True,
            "maximum_isolated_fraction": 0.0,
            "minimum_largest_component_fraction": 0.50,
            "minimum_length_energy_ratio": 0.10,
            "median_length_energy_ratio": 0.50,
        },
        "lexicographic_order": [
            "max_minimum_perturbation_stability",
            "max_minimum_direction_cosine",
            "min_moment_dispersion",
            "max_predicted_roughness_decrease",
            "frozen_geometry_priority",
        ],
        "valid_count": len(valid),
        "fallback_to_invalid_pool": False,
        "selected_geometry": selected["geometry_id"],
        "selected_summary": selected,
    }


def graph_health(graph: csr_matrix) -> dict:
    return extended_graph_diagnostics(graph)


__all__ = [
    "GEOMETRIES",
    "CONTROL_ONLY_GEOMETRY_IDS",
    "GeometrySpec",
    "INTRINSIC_LAMBDA",
    "INTRINSIC_RULE_VERSION",
    "INTRINSIC_TRUST",
    "aggregate_geometry_similarity",
    "build_geometry_graph",
    "choose_intrinsic_geometry",
    "deduplicate_geometries",
    "edge_jaccard",
    "geometry_coordinates",
    "geometry_base_coordinates",
    "geometry_index",
    "graph_energy",
    "graph_from_transformed_coordinates",
    "graph_health",
    "intrinsic_geometry_summary",
    "operator_cosine",
    "perturbation_diagnostics",
    "phase_a_geometry_ids",
    "selector_geometry_ids",
    "stable_rng",
    "transform_geometry_metric",
    "validate_physically_label_free_members",
]
