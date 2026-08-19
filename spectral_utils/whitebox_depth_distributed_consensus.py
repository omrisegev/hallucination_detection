"""Final retrospective depth-distributed white-box candidate.

This module extends the layer-organic tail consensus with four complementary
depth summaries.  One is the registered lens-96 hierarchical DUFS score; the
other three are label-free rank-consensus summaries selected in the final
retrospective search:

* maximum target NLL aggregated through every organic layer;
* maximum target-vs-top1 gap forced across four depth bands;
* mean entropy excess over top1 aggregated over eight organic layers.

The output contains white-box experts only.  The experiment adapter appends
ordinary generation entropy as a separately named output-level control before
the outer U-PCR fit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .whitebox_depth_tail_consensus import _aggregate
from .whitebox_depth_tail_organic_consensus import (
    COMPONENTS as ORGANIC_COMPONENTS,
    REGISTRY as ORGANIC_REGISTRY,
    extract_depth_tail_organic_consensus,
)
from .whitebox_layer_fusion import (
    FeatureMatrix,
    LayerCell,
    extract_lens96,
    fit_hierarchical,
)


VERSION = "whitebox-depth-distributed-consensus-v1-2026-08-14"
ADDITIONAL_COMPONENTS = (
    "lens96_hierarchical_dufs",
    "max_target_nll_organic_all",
    "max_target_gap_spread4",
    "mean_entropy_excess_top1_organic_top8",
)
COMPONENTS = ORGANIC_COMPONENTS + ADDITIONAL_COMPONENTS
REGISTRY: Mapping[str, Mapping[str, Any]] = {
    **ORGANIC_REGISTRY,
    "lens96_hierarchical_dufs": {
        "source": "four token-mean lens metrics x three module positions x spaced eight layers",
        "grouping": "twelve module-by-metric groups",
        "solver": "DUFS-LIU-PCR within groups and across virtual experts",
        "settings": "seeds 11/23/37; epochs 80; k=7; lambda=0.1",
    },
    "max_target_nll_organic_all": {
        "formula": "max_t[-log p_target(module, layer, token)]",
        "aggregation": "equal module mean inside every layer, then all-layer anchor-reliability consensus",
    },
    "max_target_gap_spread4": {
        "formula": "max_t[log p_top1 - log p_target]",
        "aggregation": "one anchor-reliable module/layer view from every depth quartile",
    },
    "mean_entropy_excess_top1_organic_top8": {
        "formula": "mean_t[H + log p_top1]",
        "aggregation": "equal module mean inside layers, then eight anchor-reliable layer experts",
    },
}


def registry_hash() -> str:
    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _distributed_views(cell: LayerCell) -> dict[str, np.ndarray]:
    width = 3 * cell.n_layers
    outputs = {
        "max_target_nll_organic_all": np.empty((cell.n_samples, width), dtype=np.float32),
        "max_target_gap_spread4": np.empty((cell.n_samples, width), dtype=np.float32),
        "mean_entropy_excess_top1_organic_top8": np.empty(
            (cell.n_samples, width), dtype=np.float32
        ),
    }
    for index, record in enumerate(cell.records):
        entropy = np.asarray(record["lens_H"], dtype=np.float32)
        top1_surprisal = -np.asarray(record["lens_logp_top1"], dtype=np.float32)
        target_nll = -np.asarray(record["lens_logp_tgt"], dtype=np.float32)
        outputs["max_target_nll_organic_all"][index] = np.max(target_nll, axis=-1).reshape(-1)
        outputs["max_target_gap_spread4"][index] = np.max(
            target_nll - top1_surprisal, axis=-1
        ).reshape(-1)
        outputs["mean_entropy_excess_top1_organic_top8"][index] = np.mean(
            entropy - top1_surprisal, axis=-1
        ).reshape(-1)
    return outputs


def extract_depth_distributed_consensus(cell: LayerCell) -> FeatureMatrix:
    """Return the frozen twelve-component white-box matrix."""

    base = extract_depth_tail_organic_consensus(cell)
    lens = extract_lens96(cell)
    lens_score, lens_diagnostics = fit_hierarchical(lens, "dufs_liu_pcr")
    raw = _distributed_views(cell)
    modes = {
        "max_target_nll_organic_all": "organic_all",
        "max_target_gap_spread4": "spread4",
        "mean_entropy_excess_top1_organic_top8": "organic_top8",
    }
    columns = [np.asarray(base.values, dtype=float), lens_score[:, None]]
    diagnostics: dict[str, Any] = {"lens96_hierarchical_dufs": lens_diagnostics}
    for name in ADDITIONAL_COMPONENTS[1:]:
        score, diagnostic = _aggregate(
            raw[name], base.risk_anchor, n_layers=cell.n_layers, mode=modes[name]
        )
        columns.append(score[:, None])
        diagnostics[name] = diagnostic
    values = np.column_stack(columns)
    if values.shape != (cell.n_samples, len(COMPONENTS)):
        raise AssertionError("unexpected final distributed-consensus shape")
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
            "additional_component_fits": diagnostics,
            "component_fits": {
                **dict(base.metadata.get("component_fits", {})),
                **diagnostics,
            },
        },
    )


__all__ = [
    "VERSION",
    "ADDITIONAL_COMPONENTS",
    "COMPONENTS",
    "REGISTRY",
    "registry_hash",
    "extract_depth_distributed_consensus",
]
