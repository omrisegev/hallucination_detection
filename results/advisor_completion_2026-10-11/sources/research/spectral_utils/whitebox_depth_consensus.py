"""Depth-distributed white-box consensus features discovered retrospectively.

The contract deliberately exposes four *component scores*, each constructed
without correctness labels from signals that span several transformer layers:

``prediction_revision``
    Mean standardized residual-stream depth revision (entropy step, adverse
    target/top-1 changes, and KL-to-final), reusing the frozen v1 extractor.
``entropy_burst``
    Token-to-token residual entropy burst, fused over five layers selected by
    label-free correlation to final-layer target NLL.
``top1_sharpness``
    Residual entropy relative to top-1 surprisal, fused over three layers.
``max_token_entropy``
    Maximum token entropy at every module/layer readout, fused over the three
    views most reliable with respect to the same label-free anchor.

The four component scores are empirical-CDF transformed, anchor-oriented, and
passed unchanged to the repository's deployed U-PCR solver.  The registry was
chosen after inspecting the white-box benchmark outcomes and therefore remains
a retrospective discovery candidate until tested on untouched data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .paper_benchmark_suite import standardize
from .whitebox_depth_metrics import extract_prediction_revision
from .whitebox_depth_token_metrics import extract_resid_entropy_burst
from .whitebox_layer_fusion import FeatureMatrix, LayerCell, fit_controls


VERSION = "whitebox-depth-consensus-v1-2026-08-13"
COMPONENTS = (
    "prediction_revision",
    "entropy_burst",
    "top1_sharpness",
    "max_token_entropy",
)
REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "prediction_revision": {
        "source": "residual entropy/NLL/top1/KL depth revisions",
        "aggregation": "equal mean after per-layer standardization",
        "selection_k": None,
        "reliability_power": None,
    },
    "entropy_burst": {
        "source": "RMS token-to-token residual entropy difference per layer",
        "aggregation": "anchor-reliability weighted rank consensus",
        "selection_k": 5,
        "reliability_power": 1.0,
    },
    "top1_sharpness": {
        "source": "mean_t[entropy + log p(top1)] per residual layer",
        "aggregation": "anchor-reliability weighted rank consensus",
        "selection_k": 3,
        "reliability_power": 2.0,
    },
    "max_token_entropy": {
        "source": "max-token entropy at every attention/MLP/residual layer",
        "aggregation": "anchor-reliability weighted rank consensus",
        "selection_k": 3,
        "reliability_power": 2.0,
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
    scale = np.std(ranked, axis=0)
    if np.any(scale < 1e-12):
        raise ValueError("rank transform received a degenerate feature")
    return (ranked - np.mean(ranked, axis=0)) / scale


def anchor_rank_consensus(
    values: np.ndarray,
    anchor: np.ndarray,
    *,
    k: int,
    reliability_power: float,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select and fuse views using only rank correlation to a risk anchor."""

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
    X = X * np.where(correlations < 0.0, -1.0, 1.0)[None, :]
    reliability = np.abs(correlations)
    selected = np.argsort(-reliability, kind="stable")[: min(int(k), X.shape[1])]
    weights = np.maximum(reliability[selected], 1e-8) ** float(reliability_power)
    weights /= np.sum(weights)
    score = X[:, selected] @ weights
    names = tuple(feature_names or tuple(f"feature_{i}" for i in range(values.shape[1])))
    return score, {
        "labels_seen_during_fit": False,
        "kept_original_indices": keep.tolist(),
        "selected_after_standardization": selected.tolist(),
        "selected_original_indices": [int(keep[index]) for index in selected],
        "selected_feature_names": [names[int(keep[index])] for index in selected],
        "selected_anchor_correlations": correlations[selected].tolist(),
        "normalized_weights": weights.tolist(),
        "k": int(k),
        "reliability_power": float(reliability_power),
    }


def _top1_sharpness(cell: LayerCell) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = []
    names = []
    for layer in range(cell.n_layers):
        columns.append(np.asarray([
            np.mean(
                np.asarray(record["lens_H"], dtype=np.float32)[2, layer]
                + np.asarray(record["lens_logp_top1"], dtype=np.float32)[2, layer]
            )
            for record in cell.records
        ], dtype=float))
        names.append(f"resid.top1_sharpness.layer_{layer:02d}")
    return np.column_stack(columns), tuple(names)


def _max_token_entropy(cell: LayerCell) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = []
    names = []
    for layer in range(cell.n_layers):
        for module_index, module in enumerate(("attn", "mlp", "resid")):
            columns.append(np.asarray([
                np.max(np.asarray(record["lens_H"], dtype=np.float32)[module_index, layer])
                for record in cell.records
            ], dtype=float))
            names.append(f"{module}.max_token_entropy.layer_{layer:02d}")
    return np.column_stack(columns), tuple(names)


def extract_depth_consensus(cell: LayerCell) -> FeatureMatrix:
    """Return the frozen four-component, label-free U-PCR input matrix."""

    revision_matrix = extract_prediction_revision(cell)
    revision_controls, revision_fit = fit_controls(revision_matrix)
    revision = revision_controls["equal_mean"]

    burst_matrix = extract_resid_entropy_burst(cell)
    burst, burst_fit = anchor_rank_consensus(
        burst_matrix.values,
        burst_matrix.risk_anchor,
        k=5,
        reliability_power=1.0,
        feature_names=burst_matrix.feature_names,
    )

    sharp_values, sharp_names = _top1_sharpness(cell)
    sharp, sharp_fit = anchor_rank_consensus(
        sharp_values,
        revision_matrix.risk_anchor,
        k=3,
        reliability_power=2.0,
        feature_names=sharp_names,
    )

    max_values, max_names = _max_token_entropy(cell)
    max_entropy, max_fit = anchor_rank_consensus(
        max_values,
        revision_matrix.risk_anchor,
        k=3,
        reliability_power=2.0,
        feature_names=max_names,
    )

    raw_components = np.column_stack([revision, burst, sharp, max_entropy])
    components = _rank_z(raw_components)
    anchor = np.asarray(revision_matrix.risk_anchor, dtype=float)
    correlations = np.asarray([
        np.corrcoef(components[:, index], anchor)[0, 1]
        for index in range(components.shape[1])
    ])
    components *= np.where(correlations < 0.0, -1.0, 1.0)[None, :]
    metadata = {
        "version": VERSION,
        "registry_sha256": registry_hash(),
        "analysis_role": "retrospective_discovery_candidate",
        "labels_seen_during_fit": False,
        "component_anchor_correlations": dict(zip(COMPONENTS, correlations.tolist())),
        "component_fits": {
            "prediction_revision": revision_fit,
            "entropy_burst": burst_fit,
            "top1_sharpness": sharp_fit,
            "max_token_entropy": max_fit,
        },
    }
    return FeatureMatrix(
        values=components,
        feature_names=COMPONENTS,
        risk_anchor=anchor,
        groups=COMPONENTS,
        protocol_signature=hashlib.sha256(
            f"{cell.protocol_signature}:{VERSION}:{registry_hash()}".encode("utf-8")
        ).hexdigest(),
        metadata=metadata,
    )


__all__ = [
    "VERSION",
    "COMPONENTS",
    "REGISTRY",
    "registry_hash",
    "anchor_rank_consensus",
    "extract_depth_consensus",
]
