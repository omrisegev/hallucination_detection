"""Pure inner-state extension of the distributed-depth consensus."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .whitebox_depth_distributed_consensus import (
    COMPONENTS as BASE_COMPONENTS,
    REGISTRY as BASE_REGISTRY,
    extract_depth_distributed_consensus,
)
from .whitebox_depth_tail_consensus import _aggregate
from .whitebox_layer_fusion import FeatureMatrix, LayerCell


VERSION = "whitebox-depth-distributed-pure-v1-2026-08-14"
KL_COMPONENT = "mean_kl_to_final_spread8"
COMPONENTS = BASE_COMPONENTS + (KL_COMPONENT,)
REGISTRY: Mapping[str, Mapping[str, Any]] = {
    **BASE_REGISTRY,
    KL_COMPONENT: {
        "formula": "mean_t[KL(logit_lens(module, layer) || final_distribution)]",
        "aggregation": "two anchor-reliable module/layer views per depth quartile",
        "depth_rule": "four equal contiguous bands; eight total views",
        "label_use": "none",
    },
}


def registry_hash() -> str:
    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_depth_distributed_pure(cell: LayerCell) -> FeatureMatrix:
    base = extract_depth_distributed_consensus(cell)
    values = np.empty((cell.n_samples, 3 * cell.n_layers), dtype=np.float32)
    for index, record in enumerate(cell.records):
        kl = np.asarray(record["lens_kl_final"], dtype=np.float32)
        values[index] = np.mean(kl, axis=-1).reshape(-1)
    score, diagnostic = _aggregate(
        values, base.risk_anchor, n_layers=cell.n_layers, mode="spread8"
    )
    matrix = np.column_stack([base.values, score])
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
            "kl_component_fit": diagnostic,
            "component_fits": {
                **dict(base.metadata.get("component_fits", {})),
                KL_COMPONENT: diagnostic,
            },
        },
    )


__all__ = [
    "VERSION",
    "KL_COMPONENT",
    "COMPONENTS",
    "REGISTRY",
    "registry_hash",
    "extract_depth_distributed_pure",
]
