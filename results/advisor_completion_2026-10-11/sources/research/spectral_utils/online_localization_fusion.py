"""Causal online adapters for the frozen GL-LIU localization architecture.

The original GL-LIU system separates whole-trace error detection from token
localization.  This module preserves that separation while making every score
causal: all features are recomputed from the observed prefix, never sliced
from features computed on the completed trace.

All public scores are risk-oriented (larger means more likely final-answer
error).  Fits are label-free and use calibration telemetry only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix

from .adapted_dufs import adapted_dufs_soft_gates
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .feature_utils import extract_all_features
from .laplacian_upcr import build_graph_from_features, laplacian_iu_path
from .online_convergence import truncate_telemetry
from .repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .streaming_utils import anchor_orient
from .token_feature_views import CORE_TOKEN_VIEWS, TOKEN_TO_GLOBAL_FEATURES, token_feature_views
from .upcr import upcr_fit


GLOBAL_MIN_AVAILABILITY = 0.70
GLOBAL_DUFS_SEEDS = (11, 23, 37)
GLOBAL_DUFS_EPOCHS = 80
GLOBAL_K = 7
GLOBAL_LAMBDA = 0.1
LOCAL_MAX_FIT_TOKENS = 60_000
LOCAL_LAMBDA = 0.3
EPS = 1e-12

LOCAL_GLOBAL_NAMES = tuple(TOKEN_TO_GLOBAL_FEATURES[name][0] for name in CORE_TOKEN_VIEWS)
LOCAL_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}


def causal_trace_features(
    row: Mapping[str, Any], budget: int | None
) -> dict[str, float]:
    """Extract the original GL-LIU answer features from one observed prefix."""

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


def causal_local_core_matrix(
    row: Mapping[str, Any], budget: int | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the five exact historical localization views for one prefix."""

    prefix = truncate_telemetry(row, budget)
    views = token_feature_views(prefix)
    matrix = np.column_stack([views[name] for name in CORE_TOKEN_VIEWS])
    return np.asarray(matrix, dtype=float), LOCAL_GLOBAL_NAMES


def _raw_feature_matrix(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> np.ndarray:
    output = np.full((len(rows), len(names)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        features = causal_trace_features(row, None)
        for j, name in enumerate(names):
            output[i, j] = features.get(name, np.nan)
    return output


@dataclass
class FrozenGlobalGLIUModel:
    """Frozen mixed-v2 DUFS-LIU answer detector."""

    feature_names: tuple[str, ...]
    transformer: FixedMixedV2Transformer
    weights: np.ndarray
    diagnostics: dict[str, Any]

    def risk_from_features(self, features: Mapping[str, float]) -> float:
        raw = np.asarray(
            [[features.get(name, np.nan) for name in self.feature_names]],
            dtype=float,
        )
        confidence = self.transformer.transform(raw)
        return float(-(confidence @ self.weights)[0])

    def risk(self, row: Mapping[str, Any], budget: int | None) -> float:
        return self.risk_from_features(causal_trace_features(row, budget))


def fit_frozen_global_gl_liu(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_elapsed_length: bool,
    dufs_epochs: int = GLOBAL_DUFS_EPOCHS,
) -> FrozenGlobalGLIUModel:
    """Fit the GL-LIU global head on completed calibration traces only."""

    candidates = [
        name for name in CONFIDENCE_FEATURE_SIGNS_V1
        if include_elapsed_length or name != "trace_length"
    ]
    raw_all = _raw_feature_matrix(rows, candidates)
    names, columns, dropped, availability = [], [], {}, {}
    for j, name in enumerate(candidates):
        values = raw_all[:, j]
        finite = np.isfinite(values)
        availability[name] = float(finite.mean())
        if availability[name] < GLOBAL_MIN_AVAILABILITY or not finite.any():
            dropped[name] = f"availability={availability[name]:.4f}"
            continue
        median = float(np.median(values[finite]))
        clean = np.where(finite, values, median)
        if float(clean.std()) < 1e-8:
            dropped[name] = "constant"
            continue
        if float(np.mean(clean == np.median(clean))) > 0.40:
            dropped[name] = "saturated"
            continue
        names.append(name)
        columns.append(clean)
    if len(names) < 3:
        raise ValueError("fewer than three usable global GL-LIU features")
    raw = np.column_stack(columns)
    transformer = FixedMixedV2Transformer.fit(raw, names)
    F = np.asarray(transformer.training_output, dtype=float).T
    gates, gate_diagnostics = adapted_dufs_soft_gates(
        F, seeds=GLOBAL_DUFS_SEEDS, epochs=int(dufs_epochs)
    )
    graph = build_graph_from_features(F, gates=gates, k=GLOBAL_K)
    fitted = laplacian_iu_path(F, (GLOBAL_LAMBDA,), graph=graph)[GLOBAL_LAMBDA]
    diagnostics = {
        "head": "answer_dufs_liu_mixed",
        "include_elapsed_length": bool(include_elapsed_length),
        "elapsed_length_is_observed_prefix_length": bool(include_elapsed_length),
        "n_fit_traces": int(len(rows)),
        "feature_names": list(names),
        "n_features": int(len(names)),
        "availability": availability,
        "dropped": dropped,
        "dufs_epochs": int(dufs_epochs),
        "dufs_seeds": list(GLOBAL_DUFS_SEEDS),
        "dufs_effective_feature_count": gate_diagnostics.get(
            "effective_feature_count"
        ),
        "lambda": GLOBAL_LAMBDA,
        "k": GLOBAL_K,
        "labels_seen_during_fit": False,
        "laplacian": fitted.diagnostics,
    }
    return FrozenGlobalGLIUModel(
        tuple(names), transformer, np.asarray(fitted.w, dtype=float), diagnostics
    )


def _ordered_local_chunks(
    rows: Sequence[Mapping[str, Any]], max_fit_tokens: int
) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
    """Take deterministic contiguous trace blocks for the temporal graph."""

    indexed = sorted(
        enumerate(rows),
        key=lambda item: (
            str(item[1].get("_group", "")),
            str(item[1].get("_trace_id", item[0])),
        ),
    )
    chunks, lengths, names = [], [], None
    remaining = int(max_fit_tokens)
    for _, row in indexed:
        if remaining <= 0:
            break
        matrix, current_names = causal_local_core_matrix(row, None)
        take = min(len(matrix), remaining)
        if take <= 0:
            continue
        chunks.append(matrix[:take])
        lengths.append(int(take))
        remaining -= int(take)
        names = current_names
    if not chunks or names is None:
        raise ValueError("no local token rows available")
    return np.vstack(chunks), names, lengths


def _temporal_graph(lengths: Sequence[int]):
    rows, cols, values, start = [], [], [], 0
    for length in lengths:
        length = int(length)
        if length > 1:
            left = np.arange(start, start + length - 1)
            right = left + 1
            rows.extend(left.tolist())
            cols.extend(right.tolist())
            values.extend([1.0] * len(left))
            rows.extend(right.tolist())
            cols.extend(left.tolist())
            values.extend([1.0] * len(left))
        start += length
    return coo_matrix((values, (rows, cols)), shape=(start, start)).tocsr()


@dataclass
class FrozenLocalArm:
    names: tuple[str, ...]
    keep: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    derived: np.ndarray
    weights: np.ndarray
    flipped: bool

    def curve(self, raw_matrix: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_matrix, dtype=float)[:, self.keep]
        clean = np.where(np.isfinite(raw), raw, self.median[None, :])
        standardized = (clean - self.mean[None, :]) / self.std[None, :]
        F = (standardized * self.derived[None, :]).T
        score = np.asarray(self.weights @ F, dtype=float)
        return -score if self.flipped else score


@dataclass
class FrozenLocalGLIUModel:
    """Frozen selected temporal locator plus DUFS local-detection arm."""

    temporal: FrozenLocalArm
    dufs: FrozenLocalArm
    diagnostics: dict[str, Any]

    def curves(
        self, row: Mapping[str, Any], budget: int | None
    ) -> dict[str, np.ndarray]:
        raw, _ = causal_local_core_matrix(row, budget)
        return {
            "temporal": self.temporal.curve(raw),
            "dufs": self.dufs.curve(raw),
        }


def fit_frozen_local_gl_liu(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_fit_tokens: int = LOCAL_MAX_FIT_TOKENS,
    dufs_epochs: int = GLOBAL_DUFS_EPOCHS,
) -> FrozenLocalGLIUModel:
    """Fit the two frozen local GL-LIU arms without labels."""

    raw, names, lengths = _ordered_local_chunks(rows, max_fit_tokens)
    finite = np.isfinite(raw)
    keep = finite.any(axis=0)
    medians_all = np.full(raw.shape[1], np.nan)
    medians_all[keep] = np.nanmedian(raw[:, keep], axis=0)
    filled = np.where(finite, raw, medians_all[None, :])
    scale_all = np.full(raw.shape[1], np.nan)
    scale_all[keep] = np.std(filled[:, keep], axis=0)
    keep &= np.isfinite(scale_all) & (scale_all > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError("fewer than three usable local GL-LIU views")
    clean = raw[:, keep]
    median = np.nanmedian(clean, axis=0)
    clean = np.where(np.isfinite(clean), clean, median[None, :])
    mean = clean.mean(axis=0)
    std = clean.std(axis=0)
    V = (clean - mean[None, :]) / std[None, :]
    first = upcr_fit(V.T, **LOCAL_FIT)
    derived = np.sign(first.rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V * derived[None, :]).T
    anchor_index = [name for name, use in zip(names, keep) if use].index("epr")
    anchor = V[:, anchor_index]

    gates, gate_diagnostics = adapted_dufs_soft_gates(
        F, seeds=GLOBAL_DUFS_SEEDS, epochs=int(dufs_epochs)
    )
    dufs_graph = build_graph_from_features(F, gates=gates, k=GLOBAL_K)
    temporal_graph = _temporal_graph(lengths)
    dufs_fit = laplacian_iu_path(
        F, (LOCAL_LAMBDA,), graph=dufs_graph
    )[LOCAL_LAMBDA]
    temporal_fit = laplacian_iu_path(
        F, (LOCAL_LAMBDA,), graph=temporal_graph
    )[LOCAL_LAMBDA]

    def arm(fitted) -> FrozenLocalArm:
        _, flipped = anchor_orient(fitted.w @ F, anchor)
        return FrozenLocalArm(
            tuple(names), keep.copy(), median.copy(), mean.copy(), std.copy(),
            derived.copy(), np.asarray(fitted.w, dtype=float), bool(flipped),
        )

    kept_names = [name for name, use in zip(names, keep) if use]
    diagnostics = {
        "heads": {
            "temporal": "token_temporal_liu_l0p3",
            "dufs": "token_dufs_liu_l0p3",
        },
        "input_names": list(names),
        "kept_names": kept_names,
        "n_fit_tokens": int(len(raw)),
        "fit_trace_blocks": int(len(lengths)),
        "max_fit_tokens": int(max_fit_tokens),
        "dufs_epochs": int(dufs_epochs),
        "dufs_effective_feature_count": gate_diagnostics.get(
            "effective_feature_count"
        ),
        "lambda": LOCAL_LAMBDA,
        "labels_seen_during_fit": False,
        "temporal_laplacian": temporal_fit.diagnostics,
        "dufs_laplacian": dufs_fit.diagnostics,
    }
    return FrozenLocalGLIUModel(arm(temporal_fit), arm(dufs_fit), diagnostics)


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    count = max(1, int(np.ceil(float(fraction) * len(values))))
    return float(np.mean(np.partition(values, len(values) - count)[-count:]))


@dataclass
class FrozenOnlineGLIUEnsemble:
    global_no_length: FrozenGlobalGLIUModel
    global_elapsed_length: FrozenGlobalGLIUModel
    local: FrozenLocalGLIUModel
    centres: dict[str, float]
    scales: dict[str, float]
    diagnostics: dict[str, Any]

    def _z(self, name: str, value: float) -> float:
        return float((value - self.centres[name]) / self.scales[name])

    def component_scores(
        self, row: Mapping[str, Any], budget: int | None
    ) -> dict[str, float]:
        features = causal_trace_features(row, budget)
        curves = self.local.curves(row, budget)
        return {
            "global_gl_liu_no_length": self.global_no_length.risk_from_features(features),
            "global_gl_liu_elapsed_length": (
                self.global_elapsed_length.risk_from_features(features)
            ),
            "local_temporal_gl_liu_max": float(np.max(curves["temporal"])),
            "local_dufs_gl_liu_top5": _top_fraction_mean(curves["dufs"]),
            "cusum_max": float(features.get("cusum_max", np.nan)),
            "sw_var_peak": float(features.get("sw_var_peak", np.nan)),
        }

    def scores(
        self, row: Mapping[str, Any], budget: int | None
    ) -> dict[str, float]:
        scores = self.component_scores(row, budget)
        scores["fused_gl_liu"] = 0.5 * (
            self._z("global_gl_liu_no_length", scores["global_gl_liu_no_length"])
            + self._z("local_dufs_gl_liu_top5", scores["local_dufs_gl_liu_top5"])
        )
        scores["cusum_swvar_equal"] = 0.5 * (
            self._z("cusum_max", scores["cusum_max"])
            + self._z("sw_var_peak", scores["sw_var_peak"])
        )
        return scores


def fit_frozen_online_gl_liu(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_fit_tokens: int = LOCAL_MAX_FIT_TOKENS,
    dufs_epochs: int = GLOBAL_DUFS_EPOCHS,
) -> FrozenOnlineGLIUEnsemble:
    """Fit both GL-LIU heads and a fixed label-free equal-weight fusion."""

    global_no_length = fit_frozen_global_gl_liu(
        rows, include_elapsed_length=False, dufs_epochs=dufs_epochs
    )
    global_elapsed = fit_frozen_global_gl_liu(
        rows, include_elapsed_length=True, dufs_epochs=dufs_epochs
    )
    local = fit_frozen_local_gl_liu(
        rows, max_fit_tokens=max_fit_tokens, dufs_epochs=dufs_epochs
    )
    provisional = FrozenOnlineGLIUEnsemble(
        global_no_length, global_elapsed, local, {}, {}, {}
    )
    by_name: dict[str, list[float]] = {}
    for row in rows:
        for name, value in provisional.component_scores(row, None).items():
            if np.isfinite(value):
                by_name.setdefault(name, []).append(float(value))
    required = (
        "global_gl_liu_no_length",
        "local_dufs_gl_liu_top5",
        "cusum_max",
        "sw_var_peak",
    )
    centres, scales = {}, {}
    for name in required:
        values = np.asarray(by_name.get(name, []), dtype=float)
        if not len(values):
            raise ValueError(f"no finite calibration values for {name}")
        centres[name] = float(values.mean())
        scales[name] = max(float(values.std()), EPS)
    diagnostics = {
        "global_no_length": global_no_length.diagnostics,
        "global_elapsed_length": global_elapsed.diagnostics,
        "local": local.diagnostics,
        "fusion": {
            "components": [
                "global_gl_liu_no_length", "local_dufs_gl_liu_top5"
            ],
            "weights": [0.5, 0.5],
            "standardization_population": "completed calibration traces",
            "labels_seen_during_fit": False,
        },
        "centres": centres,
        "scales": scales,
    }
    return FrozenOnlineGLIUEnsemble(
        global_no_length, global_elapsed, local, centres, scales, diagnostics
    )


__all__ = [
    "FrozenGlobalGLIUModel",
    "FrozenLocalGLIUModel",
    "FrozenOnlineGLIUEnsemble",
    "causal_local_core_matrix",
    "causal_trace_features",
    "fit_frozen_global_gl_liu",
    "fit_frozen_local_gl_liu",
    "fit_frozen_online_gl_liu",
]
