"""Depth-spread token-tail consensus for white-box hallucination detection.

The candidate was found in a retrospective screen and is intentionally marked
as such.  It adds three label-free components to the earlier four-component
depth consensus:

* maximum target-token NLL, with two views forced from every depth quartile;
* maximum top-1 surprisal, with the same depth-spread rule;
* maximum entropy excess over top-1 surprisal, using eight anchor-reliable
  module/layer views.

All view orientation and selection use only final-layer residual target NLL.
Correctness labels are not accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .paper_benchmark_suite import standardize
from .whitebox_depth_consensus import COMPONENTS as BASE_COMPONENTS
from .whitebox_depth_consensus import extract_depth_consensus
from .whitebox_layer_fusion import FeatureMatrix, LayerCell


VERSION = "whitebox-depth-tail-consensus-v1-2026-08-14"
TAIL_COMPONENTS = (
    "max_top1_surprisal_spread8",
    "max_entropy_excess_over_top1_top8",
    "max_target_nll_spread8",
)
COMPONENTS = BASE_COMPONENTS + TAIL_COMPONENTS
REGISTRY: Mapping[str, Mapping[str, Any]] = {
    **{
        name: {
            "source": "whitebox-depth-consensus-v1",
            "role": "previous retrospective component",
        }
        for name in BASE_COMPONENTS
    },
    "max_top1_surprisal_spread8": {
        "formula": "max_t[-log p_top1(module, layer, token)]",
        "aggregation": "two anchor-reliable views per depth quartile; |Spearman| weights",
        "depth_rule": "four equal contiguous bands; two views per band",
    },
    "max_entropy_excess_over_top1_top8": {
        "formula": "max_t[H(module, layer, token) + log p_top1(module, layer, token)]",
        "aggregation": "eight globally anchor-reliable views; |Spearman| weights",
        "depth_rule": "not forced; selected layers are reported",
    },
    "max_target_nll_spread8": {
        "formula": "max_t[-log p_target(module, layer, token)]",
        "aggregation": "two anchor-reliable views per depth quartile; |Spearman| weights",
        "depth_rule": "four equal contiguous bands; two views per band",
    },
}


def registry_hash() -> str:
    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("rank transform expects samples x features")
    ranked = np.column_stack([
        rankdata(values[:, index], method="average") / (len(values) + 1.0)
        for index in range(values.shape[1])
    ])
    scale = ranked.std(axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("rank transform received a degenerate feature")
    return (ranked - ranked.mean(axis=0)) / scale


def _oriented_views(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, keep, _means, _scales = standardize(np.asarray(values, dtype=float))
    X = _rank_z(X)
    anchor_rank = rankdata(np.asarray(anchor, dtype=float), method="average")
    anchor_rank /= len(anchor_rank) + 1.0
    correlations = np.asarray([
        np.corrcoef(X[:, index], anchor_rank)[0, 1]
        for index in range(X.shape[1])
    ])
    if not np.isfinite(correlations).all():
        raise ValueError("non-finite anchor reliability")
    X *= np.where(correlations < 0.0, -1.0, 1.0)[None, :]
    return X, np.abs(correlations), keep


def _weighted(X: np.ndarray, reliability: np.ndarray, selected: np.ndarray) -> np.ndarray:
    weights = np.maximum(reliability[selected], 1e-8)
    weights /= weights.sum()
    return X[:, selected] @ weights


def _aggregate(
    values: np.ndarray,
    anchor: np.ndarray,
    *,
    n_layers: int,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    X, reliability, keep = _oriented_views(values, anchor)
    original_layers = keep % int(n_layers)
    virtual_layer_aggregation = False
    if mode == "top8":
        selected = np.argsort(-reliability, kind="stable")[: min(8, X.shape[1])]
        bands: Sequence[Sequence[int]] | None = None
    elif mode in {"spread4", "spread8"}:
        bands = tuple(np.array_split(np.arange(n_layers), 4))
        per_band = 1 if mode == "spread4" else 2
        chosen = []
        for band in bands:
            eligible = np.flatnonzero(np.isin(original_layers, band))
            order = eligible[np.argsort(-reliability[eligible], kind="stable")]
            chosen.extend(order[:per_band].tolist())
        selected = np.asarray(chosen, dtype=int)
    elif mode in {"organic_all", "organic_top8"}:
        bands = None
        layer_columns = []
        layer_names = []
        for layer in range(n_layers):
            members = np.flatnonzero(original_layers == layer)
            if members.size:
                layer_columns.append(X[:, members].mean(axis=1))
                layer_names.append(layer)
        layer_matrix = _rank_z(np.column_stack(layer_columns))
        anchor_rank = rankdata(np.asarray(anchor, dtype=float), method="average")
        anchor_rank /= len(anchor_rank) + 1.0
        layer_correlations = np.asarray([
            np.corrcoef(layer_matrix[:, index], anchor_rank)[0, 1]
            for index in range(layer_matrix.shape[1])
        ])
        layer_matrix *= np.where(layer_correlations < 0.0, -1.0, 1.0)[None, :]
        X = layer_matrix
        reliability = np.abs(layer_correlations)
        count = len(layer_names) if mode == "organic_all" else min(8, len(layer_names))
        selected = np.argsort(-reliability, kind="stable")[:count]
        original_layers = np.asarray(layer_names, dtype=int)
        keep = np.arange(len(layer_names), dtype=int)
        virtual_layer_aggregation = True
    else:
        raise ValueError(f"unknown aggregation mode: {mode}")
    score = _weighted(X, reliability, selected)
    # This exactly matches the final component-wise empirical-CDF transform in
    # the discovery screen while remaining label-free.
    score = _rank_z(score[:, None])[:, 0]
    if np.corrcoef(score, anchor)[0, 1] < 0.0:
        score = -score
    diagnostics = {
        "labels_seen_during_fit": False,
        "mode": mode,
        "selected_original_indices": keep[selected].tolist(),
        "selected_layers": original_layers[selected].tolist(),
        "selected_modules": (keep[selected] // int(n_layers)).tolist(),
        "selected_anchor_reliability": reliability[selected].tolist(),
        "normalized_weights": (
            np.maximum(reliability[selected], 1e-8)
            / np.maximum(reliability[selected], 1e-8).sum()
        ).tolist(),
        "depth_bands": None if bands is None else [np.asarray(band).tolist() for band in bands],
        "virtual_layer_aggregation": virtual_layer_aggregation,
    }
    if virtual_layer_aggregation:
        diagnostics["selected_modules"] = None
        diagnostics["within_layer_aggregation"] = "equal mean of anchor-oriented module views"
    return score, diagnostics


def _tail_views(cell: LayerCell) -> dict[str, np.ndarray]:
    n = cell.n_samples
    width = 3 * cell.n_layers
    outputs = {
        "max_top1_surprisal_spread8": np.empty((n, width), dtype=np.float32),
        "max_entropy_excess_over_top1_top8": np.empty((n, width), dtype=np.float32),
        "max_target_nll_spread8": np.empty((n, width), dtype=np.float32),
    }
    for index, record in enumerate(cell.records):
        entropy = np.asarray(record["lens_H"], dtype=np.float32)
        top1_surprisal = -np.asarray(record["lens_logp_top1"], dtype=np.float32)
        target_nll = -np.asarray(record["lens_logp_tgt"], dtype=np.float32)
        outputs["max_top1_surprisal_spread8"][index] = np.max(top1_surprisal, axis=-1).reshape(-1)
        outputs["max_entropy_excess_over_top1_top8"][index] = np.max(
            entropy - top1_surprisal, axis=-1
        ).reshape(-1)
        outputs["max_target_nll_spread8"][index] = np.max(target_nll, axis=-1).reshape(-1)
    return outputs


def extract_depth_tail_consensus(cell: LayerCell) -> FeatureMatrix:
    """Build the frozen seven-component retrospective candidate matrix."""

    base = extract_depth_consensus(cell)
    views = _tail_views(cell)
    modes = {
        "max_top1_surprisal_spread8": "spread8",
        "max_entropy_excess_over_top1_top8": "top8",
        "max_target_nll_spread8": "spread8",
    }
    columns = [np.asarray(base.values, dtype=float)]
    diagnostics = {}
    for name in TAIL_COMPONENTS:
        score, diagnostic = _aggregate(
            views[name], base.risk_anchor, n_layers=cell.n_layers, mode=modes[name]
        )
        columns.append(score[:, None])
        diagnostics[name] = diagnostic
    matrix = np.column_stack(columns)
    if matrix.shape != (cell.n_samples, len(COMPONENTS)):
        raise AssertionError("unexpected tail-consensus shape")
    return FeatureMatrix(
        values=matrix,
        feature_names=COMPONENTS,
        risk_anchor=np.asarray(base.risk_anchor, dtype=float),
        groups=COMPONENTS,
        protocol_signature=hashlib.sha256(
            f"{base.protocol_signature}:{VERSION}:{registry_hash()}".encode("utf-8")
        ).hexdigest(),
        metadata={
            "version": VERSION,
            "registry_sha256": registry_hash(),
            "analysis_role": "retrospective_discovery_candidate",
            "labels_seen_during_fit": False,
            "base_metadata": base.metadata,
            "tail_component_fits": diagnostics,
            "component_fits": {
                **dict(base.metadata.get("component_fits", {})),
                **diagnostics,
            },
        },
    )


__all__ = [
    "VERSION",
    "BASE_COMPONENTS",
    "TAIL_COMPONENTS",
    "COMPONENTS",
    "REGISTRY",
    "registry_hash",
    "extract_depth_tail_consensus",
]
