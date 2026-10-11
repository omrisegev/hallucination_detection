"""Layer-local token-dynamic metrics for depth-fusion discovery iteration 2.

Iteration 1 showed that token means were not distributed risk sensors: most of
their macro signal appeared only in the final depth decile.  These extractors
retain response dynamics inside every layer while still returning one scalar
expert per layer.  Each column is oriented using only its unlabeled Pearson
correlation with the final-layer residual target-token NLL anchor.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np

from .whitebox_layer_fusion import FeatureMatrix, LayerCell, all_layers


EPS = 1e-12
MODULES = ("attn", "mlp", "resid")


TOKEN_METRIC_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "resid_entropy_burst": {
        "display": "Residual entropy burst trajectory",
        "literature_basis": "trace-level entropy instability and localized entropy peaks",
        "formula": "per layer: RMS_t delta H_resid(l,t)",
        "layer_policy": "all layers",
        "risk_direction": "anchor-aligned from +burst magnitude",
        "readout": "all generated-token positions; one scalar per layer",
    },
    "resid_entropy_dispersion": {
        "display": "Residual entropy dispersion trajectory",
        "literature_basis": "trace-level entropy variability during erroneous generation",
        "formula": "per layer: sample standard deviation_t H_resid(l,t)",
        "layer_policy": "all layers",
        "risk_direction": "anchor-aligned from +dispersion",
        "readout": "all generated-token positions; one scalar per layer",
    },
    "resid_target_nll_burst": {
        "display": "Target-surprisal burst trajectory",
        "literature_basis": "generated-token commitment dynamics across depth and time",
        "formula": "per layer: RMS_t delta[-log p_l(generated_token_t)]",
        "layer_policy": "all layers",
        "risk_direction": "anchor-aligned from +burst magnitude",
        "readout": "all generated-token positions; one scalar per layer",
    },
    "module_entropy_disagreement": {
        "display": "Tokenwise TriLens entropy disagreement",
        "literature_basis": "TriLens pathway complementarity at attention, MLP, and residual readouts",
        "formula": "per layer: mean_t std_module[H_attn,H_mlp,H_resid]",
        "layer_policy": "all layers",
        "risk_direction": "anchor-aligned from +pathway disagreement",
        "readout": "module dispersion before token pooling; one scalar per layer",
    },
    "module_target_nll_disagreement": {
        "display": "Tokenwise target-commitment disagreement",
        "literature_basis": "pathway-specific commitment and competing continuations",
        "formula": "per layer: mean_t std_module[-log p_attn,-log p_mlp,-log p_resid]",
        "layer_policy": "all layers",
        "risk_direction": "anchor-aligned from +pathway disagreement",
        "readout": "module dispersion before token pooling; one scalar per layer",
    },
    "resid_kl_tail": {
        "display": "Residual KL settling-tail trajectory",
        "literature_basis": "DoLa-style premature-to-final prediction contrast",
        "formula": "per layer: q90_t log1p KL(p_l || p_final)",
        "layer_policy": "all layers except mechanically degenerate final KL",
        "risk_direction": "anchor-aligned from +unsettled tail",
        "readout": "all generated-token positions; one scalar per layer",
    },
}


def registry_hash() -> str:
    payload = json.dumps(TOKEN_METRIC_REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_index(cell: LayerCell, module: str) -> int:
    return cell.modules.index(module)


def _risk_anchor(cell: LayerCell) -> np.ndarray:
    index = _module_index(cell, "resid")
    return np.asarray([
        -np.mean(np.asarray(record["lens_logp_tgt"], dtype=float)[index, -1])
        for record in cell.records
    ], dtype=float)


def _align_to_anchor(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, list[int]]:
    aligned = np.asarray(values, dtype=float).copy()
    flips = []
    for column in range(aligned.shape[1]):
        corr = float(np.corrcoef(aligned[:, column], anchor)[0, 1])
        if np.isfinite(corr) and corr < 0.0:
            aligned[:, column] *= -1.0
            flips.append(column)
    return aligned, flips


def _matrix(
    cell: LayerCell,
    contract: str,
    values: np.ndarray,
    *,
    layer_numbers: tuple[int, ...],
) -> FeatureMatrix:
    values = np.asarray(values, dtype=float)
    anchor = _risk_anchor(cell)
    values, flips = _align_to_anchor(values, anchor)
    keep = np.isfinite(values).all(axis=0) & (np.std(values, axis=0) > 1e-8)
    if not np.any(keep):
        raise ValueError(f"{contract}: all layer experts are degenerate")
    kept_layers = tuple(layer for layer, flag in zip(layer_numbers, keep) if flag)
    aligned = values[:, keep]
    return FeatureMatrix(
        values=aligned,
        feature_names=tuple(f"{contract}.layer_{layer:02d}" for layer in kept_layers),
        risk_anchor=anchor,
        groups=tuple(f"layer_{layer:02d}" for layer in kept_layers),
        protocol_signature=hashlib.sha256(
            f"{cell.protocol_signature}|token-dynamics|{contract}|{registry_hash()}".encode()
        ).hexdigest(),
        metadata={
            **dict(TOKEN_METRIC_REGISTRY[contract]),
            "contract": contract,
            "registry_sha256": registry_hash(),
            "layers_before_degenerate_drop": list(layer_numbers),
            "layers_after_degenerate_drop": list(kept_layers),
            "anchor_flipped_column_indices_before_drop": flips,
            "anchor_alignment": "unlabeled Pearson sign against final-layer residual target NLL",
            "orientation_anchor": "final-layer residual target-token NLL",
            "outcomes_seen_during_extraction": False,
        },
    )


def _residual_trace_metric(
    cell: LayerCell,
    quantity: str,
    reducer: Callable[[np.ndarray], float],
    *,
    sign: float = 1.0,
) -> np.ndarray:
    index = _module_index(cell, "resid")
    rows = []
    for record in cell.records:
        traces = sign * np.asarray(record[quantity], dtype=float)[index]
        rows.append([reducer(traces[layer]) for layer in all_layers(cell.n_layers)])
    return np.asarray(rows, dtype=float)


def _rms_difference(trace: np.ndarray) -> float:
    differences = np.diff(np.asarray(trace, dtype=float))
    return float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0


def _sample_std(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=float)
    return float(np.std(trace, ddof=1)) if len(trace) > 1 else 0.0


def extract_resid_entropy_burst(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    return _matrix(
        cell,
        "resid_entropy_burst",
        _residual_trace_metric(cell, "lens_H", _rms_difference),
        layer_numbers=layers,
    )


def extract_resid_entropy_dispersion(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    return _matrix(
        cell,
        "resid_entropy_dispersion",
        _residual_trace_metric(cell, "lens_H", _sample_std),
        layer_numbers=layers,
    )


def extract_resid_target_nll_burst(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    return _matrix(
        cell,
        "resid_target_nll_burst",
        _residual_trace_metric(cell, "lens_logp_tgt", _rms_difference, sign=-1.0),
        layer_numbers=layers,
    )


def _module_disagreement(cell: LayerCell, quantity: str, sign: float) -> np.ndarray:
    rows = []
    for record in cell.records:
        tensor = sign * np.asarray(record[quantity], dtype=float)
        if tensor.shape[0] != 3 or tensor.shape[1] != cell.n_layers:
            raise ValueError(f"{quantity}: expected module x layer x token tensor")
        rows.append(np.mean(np.std(tensor, axis=0, ddof=0), axis=1))
    return np.asarray(rows, dtype=float)


def extract_module_entropy_disagreement(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    return _matrix(
        cell,
        "module_entropy_disagreement",
        _module_disagreement(cell, "lens_H", 1.0),
        layer_numbers=layers,
    )


def extract_module_target_nll_disagreement(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    return _matrix(
        cell,
        "module_target_nll_disagreement",
        _module_disagreement(cell, "lens_logp_tgt", -1.0),
        layer_numbers=layers,
    )


def extract_resid_kl_tail(cell: LayerCell) -> FeatureMatrix:
    layers = all_layers(cell.n_layers)
    values = _residual_trace_metric(
        cell,
        "lens_kl_final",
        lambda trace: float(np.quantile(np.log1p(np.maximum(trace, 0.0)), 0.9)),
    )
    return _matrix(cell, "resid_kl_tail", values, layer_numbers=layers)


EXTRACTORS = {
    "resid_entropy_burst": extract_resid_entropy_burst,
    "resid_entropy_dispersion": extract_resid_entropy_dispersion,
    "resid_target_nll_burst": extract_resid_target_nll_burst,
    "module_entropy_disagreement": extract_module_entropy_disagreement,
    "module_target_nll_disagreement": extract_module_target_nll_disagreement,
    "resid_kl_tail": extract_resid_kl_tail,
}


__all__ = ["EXTRACTORS", "TOKEN_METRIC_REGISTRY", "registry_hash"]
