"""Token-native features and ordinary IU heads for three reasoning outputs.

The public fitting path is target-blind.  It accepts telemetry dictionaries but
never reads correctness, first-error, or step annotations.  All primitive
channels and public scores are risk-oriented: larger means more evidence of an
error.  Step spans and targets belong in the evaluation harness.

This module implements the roster frozen in
``docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .upcr import upcr_fit


EPS = 1e-12
EWMA_ALPHA = 2.0 / 17.0
REFERENCE_POSITIONS = 32
FIT_POSITIONS = 32
CLIP_Z = 8.0

CHANNEL_NAMES = (
    "entropy",
    "spilled",
    "neg_logsumexp",
    "neg_top1",
    "neg_margin",
    "topk_entropy",
    "topk_varentropy",
    "topk_renyi2",
    "topk_tail_mass",
)

STATE_NAMES = (
    "level",
    "ewma",
    "onset",
    "positive_mean",
    "persistence",
    "running_max",
)

HEAD_FEATURES = {
    "g_mean9": tuple(f"{name}__mean" for name in CHANNEL_NAMES),
    "g_mean_q90_18": tuple(
        f"{name}__{reducer}"
        for name in CHANNEL_NAMES
        for reducer in ("mean", "q90")
    ),
    "g_mean_q90_max_27": tuple(
        f"{name}__{reducer}"
        for name in CHANNEL_NAMES
        for reducer in ("mean", "q90", "max")
    ),
    "l_level9": tuple(f"{name}__level" for name in CHANNEL_NAMES),
    "l_onset9": tuple(f"{name}__onset" for name in CHANNEL_NAMES),
    "l_level_onset18": tuple(
        f"{name}__{state}"
        for name in CHANNEL_NAMES
        for state in ("level", "onset")
    ),
    "o_level_ewma18": tuple(
        f"{name}__{state}"
        for name in CHANNEL_NAMES
        for state in ("level", "ewma")
    ),
    "o_level_ewma_onset27": tuple(
        f"{name}__{state}"
        for name in CHANNEL_NAMES
        for state in ("level", "ewma", "onset")
    ),
    "o_ewma_area_persist27": tuple(
        f"{name}__{state}"
        for name in CHANNEL_NAMES
        for state in ("ewma", "positive_mean", "persistence")
    ),
}

GLOBAL_HEADS = tuple(name for name in HEAD_FEATURES if name.startswith("g_"))
LOCAL_HEADS = tuple(name for name in HEAD_FEATURES if name.startswith("l_"))
ONLINE_HEADS = tuple(name for name in HEAD_FEATURES if name.startswith("o_"))

IU_FIT = {
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


def _aligned(values: Any, n: int) -> np.ndarray:
    """Return an aligned float vector, with genuine absence represented by NaN."""

    if values is None:
        return np.full(n, np.nan, dtype=float)
    output = np.asarray(values, dtype=float).reshape(-1)
    if not len(output):
        return np.full(n, np.nan, dtype=float)
    if len(output) >= n:
        return output[:n].copy()
    return np.concatenate([output, np.full(n - len(output), output[-1])])


def raw_risk_channels(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the nine frozen risk-oriented primitive token channels."""

    entropy = np.asarray(row.get("token_entropies", ()), dtype=float).reshape(-1)
    n = len(entropy)
    if n < 1:
        raise ValueError("token_entropies must contain at least one token")
    spilled = _aligned(row.get("token_spilled_energies"), n)
    logsumexp = _aligned(row.get("token_logsumexp"), n)

    logprob = np.full((n, 0), np.nan, dtype=float)
    topk = row.get("top_k_logprobs")
    if isinstance(topk, Mapping) and topk.get("logprobs") is not None:
        candidate = np.asarray(topk["logprobs"], dtype=float)
        if candidate.ndim == 2 and candidate.shape[0] > 0:
            if len(candidate) < n:
                pad = np.repeat(candidate[-1:, :], n - len(candidate), axis=0)
                candidate = np.vstack([candidate, pad])
            logprob = candidate[:n]

    if logprob.shape[1]:
        top1 = logprob[:, 0]
        top2 = logprob[:, 1] if logprob.shape[1] > 1 else logprob[:, 0]
        # Subtracting the row maximum makes exponentiation stable and cancels
        # exactly under the subsequent top-k renormalization.
        shifted = logprob - np.nanmax(logprob, axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= np.sum(probability, axis=1, keepdims=True) + EPS
        surprisal = -np.log(probability + EPS)
        topk_entropy = -np.sum(probability * np.log(probability + EPS), axis=1)
        mean_surprisal = np.sum(probability * surprisal, axis=1, keepdims=True)
        varentropy = np.sum(
            probability * (surprisal - mean_surprisal) ** 2, axis=1
        )
        renyi2 = -np.log(np.sum(probability ** 2, axis=1) + EPS)
        leading = min(5, logprob.shape[1])
        tail_mass = np.clip(
            1.0 - np.sum(probability[:, :leading], axis=1), 0.0, 1.0
        )
    else:
        top1 = np.full(n, np.nan)
        top2 = np.full(n, np.nan)
        topk_entropy = np.full(n, np.nan)
        varentropy = np.full(n, np.nan)
        renyi2 = np.full(n, np.nan)
        tail_mass = np.full(n, np.nan)

    output = {
        "entropy": entropy,
        "spilled": spilled,
        "neg_logsumexp": -logsumexp,
        "neg_top1": -top1,
        "neg_margin": -(top1 - top2),
        "topk_entropy": topk_entropy,
        "topk_varentropy": varentropy,
        "topk_renyi2": renyi2,
        "topk_tail_mass": tail_mass,
    }
    if tuple(output) != CHANNEL_NAMES:
        raise RuntimeError("primitive channel order changed")
    return output


def equal_positions(length: int, count: int) -> np.ndarray:
    """Deterministic quantile positions with exactly equal trace contribution."""

    length, count = int(length), int(count)
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    return np.linspace(0, length - 1, count, dtype=int)


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return centre, scale


@dataclass(frozen=True)
class ChannelReference:
    centres: np.ndarray
    scales: np.ndarray
    availability: np.ndarray
    n_traces: int
    positions_per_trace: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": list(CHANNEL_NAMES),
            "centres": self.centres.tolist(),
            "scales": self.scales.tolist(),
            "availability": self.availability.tolist(),
            "n_traces": int(self.n_traces),
            "positions_per_trace": int(self.positions_per_trace),
        }


def fit_channel_reference(
    rows: Sequence[Mapping[str, Any]], *, positions_per_trace: int = REFERENCE_POSITIONS
) -> ChannelReference:
    """Fit label-free primitive references with equal contribution per trace."""

    samples = {name: [] for name in CHANNEL_NAMES}
    availability = np.zeros(len(CHANNEL_NAMES), dtype=float)
    for row in rows:
        channels = raw_risk_channels(row)
        positions = equal_positions(len(channels[CHANNEL_NAMES[0]]), positions_per_trace)
        for j, name in enumerate(CHANNEL_NAMES):
            values = channels[name][positions]
            availability[j] += float(np.mean(np.isfinite(values)))
            samples[name].append(values)
    centres, scales = [], []
    for name in CHANNEL_NAMES:
        values = np.concatenate(samples[name]) if samples[name] else np.zeros(0)
        centre, scale = _robust_location_scale(values)
        centres.append(centre)
        scales.append(scale)
    divisor = max(len(rows), 1)
    return ChannelReference(
        np.asarray(centres, dtype=float),
        np.asarray(scales, dtype=float),
        availability / divisor,
        len(rows),
        int(positions_per_trace),
    )


def standardized_risk_channels(
    row: Mapping[str, Any], reference: ChannelReference
) -> dict[str, np.ndarray]:
    """Apply frozen references; missing observations map to reference level zero."""

    raw = raw_risk_channels(row)
    output = {}
    for j, name in enumerate(CHANNEL_NAMES):
        values = (raw[name] - reference.centres[j]) / reference.scales[j]
        values = np.where(np.isfinite(values), values, 0.0)
        output[name] = np.clip(values, -CLIP_Z, CLIP_Z)
    return output


def causal_states(
    row: Mapping[str, Any], reference: ChannelReference
) -> dict[str, np.ndarray]:
    """Build every frozen O(1)-update state as a complete causal replay."""

    channels = standardized_risk_channels(row, reference)
    output: dict[str, np.ndarray] = {}
    for name in CHANNEL_NAMES:
        level = channels[name]
        n = len(level)
        ewma = np.empty(n, dtype=float)
        onset = np.zeros(n, dtype=float)
        previous = 0.0
        for index, value in enumerate(level):
            if index:
                onset[index] = max(0.0, float(value) - previous)
            current = (1.0 - EWMA_ALPHA) * previous + EWMA_ALPHA * float(value)
            ewma[index] = current
            previous = current
        positive = np.maximum(level, 0.0)
        denominator = np.arange(1, n + 1, dtype=float)
        positive_mean = np.cumsum(positive) / denominator
        persistence = np.cumsum(level > 0.0) / denominator
        running_max = np.maximum.accumulate(level)
        values = {
            "level": level,
            "ewma": ewma,
            "onset": onset,
            "positive_mean": positive_mean,
            "persistence": persistence,
            "running_max": running_max,
        }
        for state in STATE_NAMES:
            output[f"{name}__{state}"] = np.asarray(values[state], dtype=float)
    return output


def global_feature_row(
    row: Mapping[str, Any], reference: ChannelReference, *, upto: int | None = None
) -> dict[str, float]:
    """Return all Global reducers on a completed trace or observed prefix."""

    channels = standardized_risk_channels(row, reference)
    output: dict[str, float] = {}
    for name in CHANNEL_NAMES:
        values = channels[name]
        if upto is not None:
            values = values[: max(1, min(int(upto), len(values)))]
        output[f"{name}__mean"] = float(np.mean(values))
        output[f"{name}__q90"] = float(np.quantile(values, 0.90))
        output[f"{name}__max"] = float(np.max(values))
    return output


def feature_matrix_for_head(
    rows: Sequence[Mapping[str, Any]],
    reference: ChannelReference,
    head_name: str,
    *,
    positions_per_trace: int = FIT_POSITIONS,
) -> np.ndarray:
    """Build the frozen candidate matrix with equal token-head trace weighting."""

    if head_name not in HEAD_FEATURES:
        raise KeyError(head_name)
    names = HEAD_FEATURES[head_name]
    matrices = []
    if head_name.startswith("g_"):
        for row in rows:
            values = global_feature_row(row, reference)
            matrices.append(np.asarray([values[name] for name in names], dtype=float)[None])
    else:
        for row in rows:
            states = causal_states(row, reference)
            n = len(states[names[0]])
            positions = equal_positions(n, positions_per_trace)
            matrices.append(np.column_stack([states[name][positions] for name in names]))
    if not matrices:
        raise ValueError("at least one telemetry row is required")
    return np.vstack(matrices)


def _standardize_fit(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    median = np.nanmedian(matrix, axis=0)
    clean = np.where(np.isfinite(matrix), matrix, median[None, :])
    spread = np.std(clean, axis=0)
    keep = np.isfinite(median) & np.isfinite(spread) & (spread > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError("fewer than three non-degenerate IU coordinates remain")
    selected = clean[:, keep]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    return (selected - mean) / std, keep, median[keep], mean, std


def _additive_residual(standardized: np.ndarray) -> float:
    """Relative off-diagonal residual for C_ij = b_i + b_j."""

    matrix = np.asarray(standardized, dtype=float)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    p = covariance.shape[0]
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    if not pairs:
        return float("nan")
    design = np.zeros((len(pairs), p), dtype=float)
    target = np.empty(len(pairs), dtype=float)
    for k, (i, j) in enumerate(pairs):
        design[k, i] = 1.0
        design[k, j] = 1.0
        target[k] = covariance[i, j]
    fit = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return float(np.linalg.norm(target - fit) / (np.linalg.norm(target) + EPS))


@dataclass(frozen=True)
class FrozenIUHead:
    name: str
    feature_names: tuple[str, ...]
    keep: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    diagnostics: Mapping[str, Any]

    @property
    def retained_names(self) -> tuple[str, ...]:
        return tuple(name for name, use in zip(self.feature_names, self.keep) if use)

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("score matrix does not match the frozen feature roster")
        selected = matrix[:, self.keep]
        clean = np.where(np.isfinite(selected), selected, self.median[None, :])
        standardized = (clean - self.mean[None, :]) / self.std[None, :]
        return standardized @ self.weights

    def score_global(
        self,
        row: Mapping[str, Any],
        reference: ChannelReference,
        *,
        upto: int | None = None,
    ) -> float:
        if not self.name.startswith("g_"):
            raise ValueError("score_global requires a Global head")
        values = global_feature_row(row, reference, upto=upto)
        matrix = np.asarray([values[name] for name in self.feature_names], dtype=float)
        return float(self.score_matrix(matrix)[0])

    def score_curve(
        self, row: Mapping[str, Any], reference: ChannelReference
    ) -> np.ndarray:
        if self.name.startswith("g_"):
            raise ValueError("score_curve requires a Local or Online head")
        states = causal_states(row, reference)
        matrix = np.column_stack([states[name] for name in self.feature_names])
        return np.asarray(self.score_matrix(matrix), dtype=float)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "keep": self.keep.astype(int).tolist(),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


def fit_iu_head(
    rows: Sequence[Mapping[str, Any]],
    reference: ChannelReference,
    head_name: str,
    *,
    positions_per_trace: int = FIT_POSITIONS,
    feature_order: Sequence[str] | None = None,
) -> FrozenIUHead:
    """Fit one frozen ordinary-IU candidate without accessing labels."""

    canonical = HEAD_FEATURES[head_name]
    order = tuple(feature_order) if feature_order is not None else canonical
    if sorted(order) != sorted(canonical) or len(order) != len(canonical):
        raise ValueError("feature_order must be a permutation of the candidate roster")
    base = feature_matrix_for_head(
        rows, reference, head_name, positions_per_trace=positions_per_trace
    )
    indices = [canonical.index(name) for name in order]
    matrix = base[:, indices]
    standardized, keep, median, mean, std = _standardize_fit(matrix)
    fitted = upcr_fit(standardized.T, **IU_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score = standardized @ weights
    anchor = standardized.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        weights = -weights
    kept = standardized.shape[1]
    if kept > 1:
        corr = np.corrcoef(standardized, rowvar=False)
        max_abs = float(np.max(np.abs(corr[np.triu_indices(kept, 1)])))
    else:
        max_abs = float("nan")
    diagnostics = {
        "labels_seen_during_fit": False,
        "n_fit_rows": int(len(matrix)),
        "input_features": int(len(order)),
        "retained_features": int(keep.sum()),
        "orientation_correlation_before_flip": correlation,
        "orientation_flipped": bool(np.isfinite(correlation) and correlation < 0),
        "max_abs_pair_correlation": max_abs,
        "additive_relative_residual": _additive_residual(standardized),
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual),
        "components": int(fitted.n_components_used),
        "scale_ratio": IU_FIT["scale_ratio"],
    }
    return FrozenIUHead(
        head_name,
        order,
        keep,
        median,
        mean,
        std,
        weights,
        diagnostics,
    )


def stable_partition(identity: str) -> str:
    """Return the scorer-shared 50/50 calibration/evaluation assignment."""

    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
    return "calibration" if int(digest[0], 16) < 8 else "evaluation"


def truncate_row(row: Mapping[str, Any], length: int) -> dict[str, Any]:
    """Causally truncate telemetry without copying labels into the feature path."""

    n = max(1, min(int(length), len(row["token_entropies"])))
    output: dict[str, Any] = {"token_entropies": np.asarray(row["token_entropies"])[:n]}
    for name in ("token_spilled_energies", "token_logsumexp"):
        if row.get(name) is not None:
            output[name] = np.asarray(row[name])[:n]
    topk = row.get("top_k_logprobs")
    if isinstance(topk, Mapping):
        output["top_k_logprobs"] = {
            name: np.asarray(values)[:n] for name, values in topk.items()
        }
    return output


__all__ = [
    "CHANNEL_NAMES",
    "STATE_NAMES",
    "HEAD_FEATURES",
    "GLOBAL_HEADS",
    "LOCAL_HEADS",
    "ONLINE_HEADS",
    "ChannelReference",
    "FrozenIUHead",
    "raw_risk_channels",
    "fit_channel_reference",
    "standardized_risk_channels",
    "causal_states",
    "global_feature_row",
    "feature_matrix_for_head",
    "fit_iu_head",
    "stable_partition",
    "truncate_row",
]
