"""Layer-organic extension of the depth-tail white-box consensus.

Each transformer layer is one real structural group.  Its three internal
features are token-maximum target NLL, top-1 surprisal, and entropy excess over
top-1 surprisal, each averaged over the attention/MLP/residual readout
positions.  Deployed U-PCR is fitted inside each layer and again across the
layer experts; the resulting label-free score becomes one additional expert
for the outer consensus.

The registry is retrospective and must not be represented as preregistered or
independently confirmed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .whitebox_depth_tail_consensus import (
    COMPONENTS as TAIL_COMPONENTS,
    REGISTRY as TAIL_REGISTRY,
    _oriented_views,
    extract_depth_tail_consensus,
)
from .whitebox_layer_fusion import FeatureMatrix, LayerCell, fit_hierarchical


VERSION = "whitebox-depth-tail-organic-consensus-v1-2026-08-14"
ORGANIC_COMPONENT = "layer_organic_max_tail_consensus"
COMPONENTS = TAIL_COMPONENTS + (ORGANIC_COMPONENT,)
REGISTRY: Mapping[str, Mapping[str, Any]] = {
    **TAIL_REGISTRY,
    ORGANIC_COMPONENT: {
        "grouping": "one transformer layer per group",
        "within_layer_features": (
            "module-mean max-token target NLL; module-mean max-token top-1 surprisal; "
            "module-mean max-token entropy excess over top-1 surprisal"
        ),
        "within_layer_solver": "deployed U-PCR (registered few-view fallback applies)",
        "across_layer_solver": "deployed U-PCR",
        "orientation": "final-layer residual target NLL anchor only",
    },
}


def registry_hash() -> str:
    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _organic_matrix(cell: LayerCell, anchor: np.ndarray) -> FeatureMatrix:
    raw = np.empty((cell.n_samples, cell.n_layers * 3), dtype=np.float32)
    for index, record in enumerate(cell.records):
        entropy = np.asarray(record["lens_H"], dtype=np.float32)
        top1_surprisal = -np.asarray(record["lens_logp_top1"], dtype=np.float32)
        target_nll = -np.asarray(record["lens_logp_tgt"], dtype=np.float32)
        # module x layer x metric, then average the three module positions so
        # every layer contains exactly the requested organic metric triad.
        tail = np.stack(
            [
                np.max(target_nll, axis=-1),
                np.max(top1_surprisal, axis=-1),
                np.max(entropy - top1_surprisal, axis=-1),
            ],
            axis=-1,
        ).mean(axis=0)
        raw[index] = tail.reshape(-1)

    oriented, reliability, keep = _oriented_views(raw, anchor)
    metric_names = ("max_target_nll", "max_top1_surprisal", "max_entropy_excess_top1")
    all_names = tuple(
        f"layer_{layer:02d}.{metric}"
        for layer in range(cell.n_layers)
        for metric in metric_names
    )
    feature_names = tuple(all_names[int(index)] for index in keep)
    groups = tuple(f"layer_{int(index) // 3:02d}" for index in keep)
    if tuple(dict.fromkeys(groups)) != tuple(f"layer_{layer:02d}" for layer in range(cell.n_layers)):
        raise ValueError("a layer-organic group disappeared after mechanical column filtering")
    signature = hashlib.sha256(
        f"{cell.protocol_signature}:{VERSION}:organic:{registry_hash()}".encode("utf-8")
    ).hexdigest()
    return FeatureMatrix(
        values=oriented,
        feature_names=feature_names,
        risk_anchor=np.asarray(anchor, dtype=float),
        groups=groups,
        protocol_signature=signature,
        metadata={
            "contract": "layer-organic-max-tail-triad",
            "labels_seen_during_fit": False,
            "n_layers": cell.n_layers,
            "within_layer_metrics": list(metric_names),
            "kept_original_indices": keep.tolist(),
            "anchor_reliability": reliability.tolist(),
        },
    )


def extract_depth_tail_organic_consensus(cell: LayerCell) -> FeatureMatrix:
    """Return eight white-box experts, including the layer-organic score."""

    base = extract_depth_tail_consensus(cell)
    organic_matrix = _organic_matrix(cell, base.risk_anchor)
    organic_score, organic_diagnostics = fit_hierarchical(organic_matrix, "upcr")
    values = np.column_stack([base.values, organic_score])
    return FeatureMatrix(
        values=values,
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
            "organic_matrix_metadata": organic_matrix.metadata,
            "organic_fit": organic_diagnostics,
            "component_fits": {
                **dict(base.metadata.get("component_fits", {})),
                ORGANIC_COMPONENT: organic_diagnostics,
            },
        },
    )


__all__ = [
    "VERSION",
    "ORGANIC_COMPONENT",
    "COMPONENTS",
    "REGISTRY",
    "registry_hash",
    "extract_depth_tail_organic_consensus",
]
