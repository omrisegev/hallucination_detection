"""CPU-only replay adapters for the registered v2 multitask baselines.

The original architecture runner imports optional DUFS/PyTorch modules even when only
ordinary IU heads are requested.  Unified Causal IU-PCR remains runnable with the base
NumPy/SciPy/scikit-learn dependency set: the DUFS dependency is imported lazily and only
by the explicit registered-DUFS replay below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .feature_utils import extract_all_features
from .dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from .multitask_trajectory import (
    ChannelReference,
    FrozenIUHead,
    fit_channel_reference,
    fit_iu_head,
    truncate_row,
)
from .online_convergence import truncate_telemetry
from .repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .upcr import upcr_fit


ORDINARY_FIT = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}

# Frozen by GL-LIU v1 and the mixed-v2 factorial follow-up.  These are constants,
# not a search surface for the fair-comparison package.
REGISTERED_DUFS_SEEDS = (11, 23, 37)
REGISTERED_DUFS_EPOCHS = 80
REGISTERED_DUFS_K = 7
REGISTERED_DUFS_LAMBDA = 0.1


def causal_trace_features(
    row: Mapping[str, Any], budget: int | None
) -> dict[str, float]:
    """Recompute the registered answer-level features from one observed prefix."""

    prefix = truncate_telemetry(row, budget)
    features = extract_all_features(
        prefix["token_entropies"],
        spilled_energies=prefix.get("token_spilled_energies"),
        allow_short=True,
    ) or {}
    if prefix.get("token_logsumexp") is not None:
        features.update(energy_features_from_logsumexp(prefix["token_logsumexp"]))
    if prefix.get("top_k_logprobs") is not None:
        features.update(logprob_features(prefix["top_k_logprobs"]))
        features.update(logprob_features_extended(prefix["top_k_logprobs"]))
    features["trace_length"] = float(len(prefix["token_entropies"]))
    return {
        str(name): float(value)
        for name, value in features.items()
        if value is not None and np.isscalar(value)
    }


@dataclass(frozen=True)
class RegisteredGlobal:
    names: tuple[str, ...]
    transformer: FixedMixedV2Transformer
    weights: np.ndarray
    diagnostics: Mapping[str, Any]

    def score(self, row: Mapping[str, Any], budget: int | None = None) -> float:
        values = causal_trace_features(row, budget)
        raw = np.asarray([[values.get(name, np.nan) for name in self.names]])
        confidence = self.transformer.transform(raw)
        return float(-(confidence @ self.weights)[0])


def _registered_global_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_trace_length: bool,
    exact_registered_training_arithmetic: bool = False,
) -> tuple[tuple[str, ...], FixedMixedV2Transformer, np.ndarray, dict[str, float]]:
    """Build the one frozen mixed-v2 matrix shared by ordinary and DUFS IU."""

    feature_rows = [causal_trace_features(row, None) for row in rows]
    names, columns, availability = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        if exclude_trace_length and name == "trace_length":
            continue
        values = np.asarray([item.get(name, np.nan) for item in feature_rows])
        finite = np.isfinite(values)
        availability[name] = float(np.mean(finite))
        if availability[name] < 0.70 or not finite.any():
            continue
        clean = np.where(finite, values, np.median(values[finite]))
        if np.std(clean) < 1e-8 or np.mean(clean == np.median(clean)) > 0.40:
            continue
        names.append(name)
        columns.append(clean)
    if len(names) < 3:
        raise ValueError("fewer than three usable registered Global features")
    raw = np.column_stack(columns)
    transformer = FixedMixedV2Transformer.fit(raw, names)
    if exact_registered_training_arithmetic:
        confidence, exact_names, _ = dufs_liu_mixed_v2_matrix(raw, names)
        if tuple(exact_names) != tuple(names):
            raise RuntimeError("registered mixed-v2 feature order drifted")
    else:
        # Preserve the already-materialized classic-IU replay arithmetic exactly.
        confidence = transformer.training_output
    return tuple(names), transformer, confidence, availability


def fit_registered_global(rows: Sequence[Mapping[str, Any]]) -> RegisteredGlobal:
    """Fold-refit the historical mixed-v2 Global IU-PCR, excluding length."""

    names, transformer, confidence, availability = _registered_global_matrix(
        rows,
        exclude_trace_length=True,
    )
    fitted = upcr_fit(confidence.T, **ORDINARY_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score, anchor = confidence @ weights, confidence.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        weights = -weights
    return RegisteredGlobal(names, transformer, weights, {
        "labels_seen_during_fit": False,
        "feature_names": list(names),
        "availability": availability,
        "orientation_correlation": correlation,
        "g2_hat": float(fitted.g2_hat),
        "trace_length_excluded": True,
    })


@dataclass(frozen=True)
class RegisteredDUFSGlobal:
    """Frozen mixed-v2 DUFS-LIU Global head.

    ``exclude_trace_length=False`` exists solely for reproducing the historical
    registered full-cell score hash.  Fair direct comparisons must use the default
    no-length form.
    """

    names: tuple[str, ...]
    transformer: FixedMixedV2Transformer
    weights: np.ndarray
    training_scores: np.ndarray
    diagnostics: Mapping[str, Any]

    def score(self, row: Mapping[str, Any], budget: int | None = None) -> float:
        values = causal_trace_features(row, budget)
        raw = np.asarray([[values.get(name, np.nan) for name in self.names]])
        confidence = self.transformer.transform(raw)
        return float(-(confidence @ self.weights)[0])


def fit_registered_dufs_global(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_trace_length: bool = True,
) -> RegisteredDUFSGlobal:
    """Fit exactly the registered mixed-v2 DUFS-LIU head on frozen rows.

    Labels are not accepted.  Seeds, epochs, graph cardinality and lambda are module
    constants copied from the registered GL-LIU protocol; callers cannot tune them.
    """

    # Lazy imports preserve the ordinary-IU module's lightweight dependency surface.
    from .adapted_dufs import adapted_dufs_soft_gates
    from .laplacian_upcr import build_graph_from_features, laplacian_iu_path

    names, transformer, confidence, availability = _registered_global_matrix(
        rows,
        exclude_trace_length=bool(exclude_trace_length),
        exact_registered_training_arithmetic=True,
    )
    features = confidence.T
    gates, gate_diagnostics = adapted_dufs_soft_gates(
        features,
        seeds=REGISTERED_DUFS_SEEDS,
        epochs=REGISTERED_DUFS_EPOCHS,
    )
    graph = build_graph_from_features(
        features,
        gates=gates,
        k=REGISTERED_DUFS_K,
    )
    fitted = laplacian_iu_path(
        features,
        (REGISTERED_DUFS_LAMBDA,),
        graph=graph,
    )[REGISTERED_DUFS_LAMBDA]
    weights = np.asarray(fitted.w, dtype=float)
    training_scores = np.asarray(-(weights @ features), dtype=float)
    return RegisteredDUFSGlobal(
        names,
        transformer,
        weights,
        training_scores,
        {
            "labels_seen_during_fit": False,
            "feature_names": list(names),
            "availability": availability,
            "trace_length_excluded": bool(exclude_trace_length),
            "dufs_seeds": list(REGISTERED_DUFS_SEEDS),
            "dufs_epochs": REGISTERED_DUFS_EPOCHS,
            "graph_k": REGISTERED_DUFS_K,
            "lambda": REGISTERED_DUFS_LAMBDA,
            "gate_effective_feature_count": float(
                gate_diagnostics["effective_feature_count"]
            ),
            "gate_raw_probabilities": np.asarray(
                gate_diagnostics["raw_probabilities"], dtype=float
            ).tolist(),
            "laplacian": dict(fitted.diagnostics),
        },
    )


@dataclass(frozen=True)
class HistoricalCellModels:
    reference: ChannelReference
    global_head: RegisteredGlobal
    local_head: FrozenIUHead
    online_head: FrozenIUHead


def fit_historical_cell_models(
    rows: Sequence[Mapping[str, Any]],
) -> HistoricalCellModels:
    reference = fit_channel_reference(rows)
    return HistoricalCellModels(
        reference=reference,
        global_head=fit_registered_global(rows),
        local_head=fit_iu_head(rows, reference, "l_level9"),
        online_head=fit_iu_head(rows, reference, "o_ewma_area_persist27"),
    )


def historical_global_score(
    models: HistoricalCellModels, row: Mapping[str, Any], budget: int | None = None
) -> float:
    return models.global_head.score(row, budget)


def historical_local_curve(
    models: HistoricalCellModels, row: Mapping[str, Any], budget: int | None = None
) -> np.ndarray:
    selected = row if budget is None else truncate_row(row, budget)
    return np.asarray(models.local_head.score_curve(selected, models.reference), dtype=float)


def historical_online_curve(
    models: HistoricalCellModels, row: Mapping[str, Any]
) -> np.ndarray:
    return np.asarray(models.online_head.score_curve(row, models.reference), dtype=float)


__all__ = [
    "HistoricalCellModels",
    "REGISTERED_DUFS_EPOCHS",
    "REGISTERED_DUFS_K",
    "REGISTERED_DUFS_LAMBDA",
    "REGISTERED_DUFS_SEEDS",
    "RegisteredDUFSGlobal",
    "RegisteredGlobal",
    "causal_trace_features",
    "fit_historical_cell_models",
    "fit_registered_dufs_global",
    "fit_registered_global",
    "historical_global_score",
    "historical_local_curve",
    "historical_online_curve",
]
