"""Frozen depth-distributed white-box metrics for the layer-fusion search.

The extractors in this module are deliberately label-free.  They turn a
validated :class:`LayerCell` into layer-local experts which can be passed to
the existing U-PCR/IU-PCR/DUFS-LIU-PCR interfaces without ever exposing an
outcome.  All signs are fixed from the metric definitions: larger means more
hallucination risk.

This is a retrospective research screen.  Its formulas are frozen in
``DEPTH_METRIC_REGISTRY`` before evaluation labels are opened.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .whitebox_layer_fusion import FeatureMatrix, LayerCell, all_layers


EPS = 1e-12
MODULES = ("attn", "mlp", "resid")


DEPTH_METRIC_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "target_commitment": {
        "display": "Target-token commitment trajectory",
        "literature_basis": (
            "Jiang et al. (NAACL 2024): generated-token probability evolves "
            "differently across depth for correct and hallucinated facts"
        ),
        "formula": "x_l = mean_t[-log p_l(generated_token_t)] on residual stream",
        "layer_policy": "all layers",
        "risk_direction": "+target NLL",
        "readout": "mean over generated-token positions",
    },
    "module_conflict": {
        "display": "TriLens module-conflict trajectory",
        "literature_basis": (
            "TriLens: attention, MLP, and residual logit-lens entropy pathways "
            "carry complementary evidence and may disagree while settling"
        ),
        "formula": (
            "per layer, standard deviation across attention/MLP/residual token-mean "
            "entropy, target NLL, and top-1 surprisal"
        ),
        "layer_policy": "all layers",
        "risk_direction": "+cross-module disagreement",
        "readout": "mean over generated-token positions before module dispersion",
    },
    "prediction_revision": {
        "display": "Prediction-revision trajectory",
        "literature_basis": (
            "DoLa and Jiang et al.: factuality is reflected by prediction changes "
            "between premature and mature layers"
        ),
        "formula": (
            "per transition l-1->l: |delta entropy|, positive target-NLL regression, "
            "positive top1-surprisal regression, and KL(current||final)"
        ),
        "layer_policy": "all transitions",
        "risk_direction": "+revision/instability",
        "readout": "mean over generated-token positions before depth difference",
    },
    "ghost_geometry": {
        "display": "GHOST geometric trajectory proxy",
        "literature_basis": (
            "Mao et al. (ACL 2026): adjacent-layer angular turbulence and premature "
            "similarity to the final state distinguish confused and stubborn hallucinations"
        ),
        "formula": (
            "per selected layer l: 1-cos(h_l,h_{l+1}) and cos(h_l,h_final)"
        ),
        "layer_policy": "floor(0.1L) through ceil(0.9L), transitions ending before upper bound",
        "risk_direction": "+turbulence; +premature convergence",
        "readout": (
            "fixed-seed 256-D Gaussian projection of the mean generated-token "
            "residual state; response-level proxy, not paper-exact token states"
        ),
    },
}


def registry_hash() -> str:
    payload = json.dumps(DEPTH_METRIC_REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ghost_layer_interval(n_layers: int) -> tuple[int, ...]:
    """Frozen architecture-relative GHOST interval.

    The paper reports layers 3..29 for a 32-layer backbone.  ``floor(.1L)`` and
    ``ceil(.9L)`` reproduce that rule and scale it without silent interpolation.
    """

    if int(n_layers) < 4:
        raise ValueError("GHOST interval needs at least four layers")
    start = int(np.floor(0.1 * int(n_layers)))
    stop = int(np.ceil(0.9 * int(n_layers)))
    stop = min(stop, int(n_layers) - 1)
    if stop <= start:
        raise ValueError("empty GHOST interval")
    return tuple(range(start, stop + 1))


def _token_mean(cell: LayerCell, quantity: str, module: str) -> np.ndarray:
    module_index = cell.modules.index(module)
    rows = []
    for record in cell.records:
        values = np.asarray(record[quantity], dtype=float)[module_index]
        if values.shape[0] != cell.n_layers:
            raise ValueError(f"{quantity}/{module}: layer dimension mismatch")
        rows.append(np.mean(values, axis=1))
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (cell.n_samples, cell.n_layers) or not np.isfinite(matrix).all():
        raise ValueError(f"{quantity}/{module}: non-finite or misaligned token means")
    return matrix


def _feature_matrix(
    cell: LayerCell,
    values: np.ndarray,
    names: Sequence[str],
    groups: Sequence[str],
    *,
    contract: str,
    metadata: Mapping[str, Any],
) -> FeatureMatrix:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != cell.n_samples:
        raise ValueError("feature matrix is not aligned to the LayerCell")
    std = np.std(values, axis=0)
    keep = np.isfinite(std) & (std > 1e-8)
    if not np.any(keep):
        raise ValueError(f"{contract}: every feature is mechanically degenerate")
    dropped = [str(names[index]) for index in np.flatnonzero(~keep)]
    kept_values = values[:, keep]
    kept_names = tuple(str(names[index]) for index in np.flatnonzero(keep))
    kept_groups = tuple(str(groups[index]) for index in np.flatnonzero(keep))
    final_target_nll = -_token_mean(cell, "lens_logp_tgt", "resid")[:, -1]
    return FeatureMatrix(
        values=kept_values,
        feature_names=kept_names,
        risk_anchor=final_target_nll,
        groups=kept_groups,
        protocol_signature=hashlib.sha256(
            f"{cell.protocol_signature}|{contract}|{registry_hash()}".encode("utf-8")
        ).hexdigest(),
        metadata={
            **dict(metadata),
            "contract": contract,
            "registry_sha256": registry_hash(),
            "n_features_before_degenerate_drop": int(values.shape[1]),
            "dropped_degenerate_features": dropped,
            "orientation_anchor": "final-layer residual target-token NLL",
            "outcomes_seen_during_extraction": False,
        },
    )


def extract_target_commitment(cell: LayerCell) -> FeatureMatrix:
    nll = -_token_mean(cell, "lens_logp_tgt", "resid")
    layers = all_layers(cell.n_layers)
    return _feature_matrix(
        cell,
        nll,
        [f"target_nll.resid.layer_{layer:02d}" for layer in layers],
        [f"layer_{layer:02d}" for layer in layers],
        contract="target-commitment-L",
        metadata={**DEPTH_METRIC_REGISTRY["target_commitment"], "layers": list(layers)},
    )


def extract_module_conflict(cell: LayerCell) -> FeatureMatrix:
    quantities = (
        ("lens_H", "entropy"),
        ("lens_logp_tgt", "target_nll"),
        ("lens_logp_top1", "top1_surprisal"),
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []
    for quantity, display in quantities:
        module_values = np.stack(
            [_token_mean(cell, quantity, module) for module in MODULES], axis=2
        )
        # Sign does not affect dispersion.  ddof=0 is deterministic and remains
        # defined even if a future architecture exposes only one valid module.
        disagreement = np.std(module_values, axis=2, ddof=0)
        for layer in all_layers(cell.n_layers):
            columns.append(disagreement[:, layer])
            names.append(f"module_conflict.{display}.layer_{layer:02d}")
            groups.append(f"layer_{layer:02d}")
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract="module-conflict-3L",
        metadata={
            **DEPTH_METRIC_REGISTRY["module_conflict"],
            "layers": list(all_layers(cell.n_layers)),
            "within_layer_features": [name for _, name in quantities],
        },
    )


def extract_prediction_revision(cell: LayerCell) -> FeatureMatrix:
    entropy = _token_mean(cell, "lens_H", "resid")
    target_nll = -_token_mean(cell, "lens_logp_tgt", "resid")
    top1_surprisal = -_token_mean(cell, "lens_logp_top1", "resid")
    kl_final = _token_mean(cell, "lens_kl_final", "resid")
    features = (
        ("abs_entropy_step", np.abs(np.diff(entropy, axis=1))),
        ("target_nll_regression", np.maximum(np.diff(target_nll, axis=1), 0.0)),
        ("top1_surprisal_regression", np.maximum(np.diff(top1_surprisal, axis=1), 0.0)),
        ("kl_to_final", kl_final[:, 1:]),
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []
    for metric, matrix in features:
        for layer in range(1, cell.n_layers):
            columns.append(matrix[:, layer - 1])
            names.append(f"prediction_revision.{metric}.transition_{layer - 1:02d}_{layer:02d}")
            groups.append(f"layer_{layer:02d}")
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract="prediction-revision-4Lm1",
        metadata={
            **DEPTH_METRIC_REGISTRY["prediction_revision"],
            "layers": list(range(1, cell.n_layers)),
            "within_layer_features": [name for name, _ in features],
        },
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def extract_ghost_geometry(cell: LayerCell) -> FeatureMatrix:
    interval = ghost_layer_interval(cell.n_layers)
    transition_layers = interval[:-1]
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []
    for layer in transition_layers:
        turbulence = []
        stubbornness = []
        for record in cell.records:
            trajectory = np.asarray(record["hid_proj"], dtype=float)
            if trajectory.shape != (cell.n_layers, cell.projection_dim):
                raise ValueError("hid_proj shape does not match LayerCell provenance")
            turbulence.append(1.0 - _cosine(trajectory[layer], trajectory[layer + 1]))
            stubbornness.append(_cosine(trajectory[layer], trajectory[-1]))
        columns.extend((np.asarray(turbulence), np.asarray(stubbornness)))
        names.extend((
            f"ghost.turbulence.layer_{layer:02d}",
            f"ghost.stubbornness.layer_{layer:02d}",
        ))
        groups.extend((f"layer_{layer:02d}", f"layer_{layer:02d}"))
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract="ghost-mean-projection-2K",
        metadata={
            **DEPTH_METRIC_REGISTRY["ghost_geometry"],
            "selected_interval": list(interval),
            "transition_layers": list(transition_layers),
            "projection_dim": cell.projection_dim,
            "rotation_invariant": True,
            "paper_fidelity": (
                "metric equations and relative depth window preserved; cached readout is "
                "mean-token JL projection rather than each generated token's full state"
            ),
        },
    )


EXTRACTORS = {
    "target_commitment": extract_target_commitment,
    "module_conflict": extract_module_conflict,
    "prediction_revision": extract_prediction_revision,
    "ghost_geometry": extract_ghost_geometry,
}


__all__ = [
    "DEPTH_METRIC_REGISTRY",
    "EXTRACTORS",
    "extract_ghost_geometry",
    "extract_module_conflict",
    "extract_prediction_revision",
    "extract_target_commitment",
    "ghost_layer_interval",
    "registry_hash",
]
