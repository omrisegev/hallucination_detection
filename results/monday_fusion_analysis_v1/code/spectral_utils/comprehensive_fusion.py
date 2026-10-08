"""Label-blind fusion variants for the comprehensive Local/Online cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix

from .adapted_dufs import adapted_dufs_soft_gates
from .laplacian_upcr import build_graph_from_features, laplacian_iu_path
from .local_online_comprehensive import (
    FIT_POSITIONS,
    IU_FIT,
    PreparedTrace,
    operator_matrix,
)
from .multitask_trajectory import equal_positions
from .upcr import upcr_fit
from .upcr_clustered import upcr_hierarchical_fit


GRAPH_K = 7
GRAPH_LAMBDA = 0.1
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
COMPAT_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}


def _temporal_graph(n_traces: int, rows_per_trace: int):
    rows, columns, values = [], [], []
    for trace in range(int(n_traces)):
        start = trace * int(rows_per_trace)
        left = np.arange(start, start + int(rows_per_trace) - 1)
        right = left + 1
        rows.extend(left.tolist()); columns.extend(right.tolist())
        values.extend([1.0] * len(left))
        rows.extend(right.tolist()); columns.extend(left.tolist())
        values.extend([1.0] * len(left))
    n = int(n_traces) * int(rows_per_trace)
    return coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()


def _orient(weights: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    weights = np.asarray(weights, dtype=float).copy()
    score = weights @ F
    anchor = np.mean(F, axis=0)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        weights = -weights
    return weights, correlation


@dataclass(frozen=True)
class FrozenFusionPanel:
    representation: str
    operators: tuple[str, ...]
    keep: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]

    def standardized(self, prepared: PreparedTrace) -> np.ndarray:
        raw = operator_matrix(
            prepared.representations[self.representation], self.operators
        )
        selected = raw[:, self.keep]
        clean = np.where(np.isfinite(selected), selected, self.median[None, :])
        return (clean - self.mean[None, :]) / self.std[None, :]

    def curve(self, prepared: PreparedTrace, variant: str) -> np.ndarray:
        return self.standardized(prepared) @ self.weights[variant]


def fit_fusion_panel(
    rows: Sequence[PreparedTrace],
    *,
    representation: str,
    operators: Sequence[str],
    include_temporal: bool,
    positions_per_trace: int = FIT_POSITIONS,
) -> FrozenFusionPanel:
    sampled = []
    for item in rows:
        raw = operator_matrix(item.representations[representation], operators)
        sampled.append(raw[equal_positions(len(raw), positions_per_trace)])
    raw = np.vstack(sampled)
    finite = np.isfinite(raw)
    finite_any = finite.any(axis=0)
    median_all = np.zeros(raw.shape[1], dtype=float)
    median_all[finite_any] = np.nanmedian(raw[:, finite_any], axis=0)
    clean_all = np.where(finite, raw, median_all[None, :])
    spread = np.std(clean_all, axis=0)
    keep = finite_any & np.isfinite(spread) & (spread > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError("fusion panel has fewer than three usable coordinates")
    median = median_all[keep]
    clean = np.where(np.isfinite(raw[:, keep]), raw[:, keep], median[None, :])
    mean = clean.mean(axis=0)
    std = clean.std(axis=0)
    V = (clean - mean[None, :]) / std[None, :]
    F = V.T

    ordinary = upcr_fit(F, **IU_FIT)
    uniform_graph = build_graph_from_features(F, k=GRAPH_K)
    uniform_path = laplacian_iu_path(
        F, (0.0, GRAPH_LAMBDA), graph=uniform_graph
    )
    if not np.array_equal(ordinary.w, uniform_path[0.0].w):
        raise AssertionError("uniform lambda=0 is not bit-identical to ordinary IU")
    gates, gate_diagnostics = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    dufs_graph = build_graph_from_features(F, gates=gates, k=GRAPH_K)
    dufs_path = laplacian_iu_path(
        F, (0.0, GRAPH_LAMBDA), graph=dufs_graph
    )
    if not np.array_equal(ordinary.w, dufs_path[0.0].w):
        raise AssertionError("DUFS lambda=0 is not bit-identical to ordinary IU")

    raw_weights = {
        "equal": np.ones(F.shape[0], dtype=float) / F.shape[0],
        "ordinary": ordinary.w,
        "upcr_compat": upcr_fit(F, **COMPAT_FIT).w,
        "uniform_laplacian": uniform_path[GRAPH_LAMBDA].w,
        "dufs_laplacian": dufs_path[GRAPH_LAMBDA].w,
    }
    failures = {}
    try:
        hierarchical, hierarchy_info = upcr_hierarchical_fit(F, **IU_FIT)
        raw_weights["hierarchical"] = hierarchical.w
    except (ValueError, np.linalg.LinAlgError) as error:
        hierarchy_info = None
        failures["hierarchical"] = str(error)
    if include_temporal:
        temporal = _temporal_graph(len(rows), positions_per_trace)
        temporal_path = laplacian_iu_path(
            F, (0.0, GRAPH_LAMBDA), graph=temporal
        )
        if not np.array_equal(ordinary.w, temporal_path[0.0].w):
            raise AssertionError("temporal lambda=0 is not bit-identical to ordinary IU")
        raw_weights["temporal_laplacian"] = temporal_path[GRAPH_LAMBDA].w

    weights, orientations = {}, {}
    for name, vector in raw_weights.items():
        weights[name], orientations[name] = _orient(vector, F)
    return FrozenFusionPanel(
        representation=representation,
        operators=tuple(operators),
        keep=keep,
        median=median,
        mean=mean,
        std=std,
        weights=weights,
        diagnostics={
            "labels_seen_during_fit": False,
            "representation": representation,
            "operators": list(operators),
            "input_coordinates": int(raw.shape[1]),
            "retained_coordinates": int(keep.sum()),
            "n_fit_rows": int(len(raw)),
            "graph_k": GRAPH_K,
            "graph_lambda": GRAPH_LAMBDA,
            "dufs_seeds": list(DUFS_SEEDS),
            "dufs_epochs": DUFS_EPOCHS,
            "dufs_effective_features": gate_diagnostics.get(
                "effective_feature_count"
            ),
            "hierarchy": hierarchy_info,
            "unavailable": failures,
            "orientations": orientations,
            "lambda_zero_exact": True,
        },
    )


__all__ = [
    "COMPAT_FIT",
    "DUFS_EPOCHS",
    "DUFS_SEEDS",
    "FrozenFusionPanel",
    "GRAPH_K",
    "GRAPH_LAMBDA",
    "fit_fusion_panel",
]
