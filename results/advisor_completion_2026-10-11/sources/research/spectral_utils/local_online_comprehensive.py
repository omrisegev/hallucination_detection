"""Feature builders for the frozen comprehensive Local/Online cycle.

The fitting paths in this module never inspect correctness or ProcessBench
labels.  Labels and step spans belong to the experiment harness.  Public
feature matrices are risk-oriented: larger means stronger error evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .multitask_trajectory import (
    CHANNEL_NAMES,
    ChannelReference,
    equal_positions,
    fit_channel_reference,
    standardized_risk_channels,
)
from .token_feature_views import (
    BROAD_TOKEN_VIEWS,
    TOKEN_TO_GLOBAL_FEATURES,
    token_feature_views,
)
from .upcr import upcr_fit


EPS = 1e-12
FIT_POSITIONS = 32
CLIP_Z = 8.0
FAST_ALPHA = 2.0 / 9.0
SLOW_ALPHA = 2.0 / 33.0

RAW7_NAMES = tuple(
    name for name in CHANNEL_NAMES if name not in {"spilled", "topk_entropy"}
)

BROAD_FAMILIES = {
    "entropy_level": (
        "entropy_series",
    ),
    "entropy_dynamics": (
        "entropy_sw_var_series",
        "entropy_cusum_abs_series",
        "entropy_rolling_tail_ratio",
    ),
    "structural": (
        "entropy_rolling_spectral_entropy",
        "entropy_rolling_low_band_power",
        "entropy_rolling_high_band_power",
        "entropy_rolling_hl_ratio",
        "entropy_rolling_dominant_freq",
        "entropy_rolling_spectral_centroid",
        "entropy_stft_high_series",
        "entropy_stft_frame_entropy",
        "entropy_pe_series",
        "entropy_rolling_rs_hurst",
    ),
    "sampled_energy": (
        "spilled_series",
        "spilled_sw_var_series",
        "spilled_cusum_abs_series",
        "spilled_rolling_min",
    ),
    "partition_energy": (
        "energy_series",
        "energy_rolling_min",
        "energy_sw_var_series",
        "energy_cusum_abs_series",
    ),
    "topk_distribution": (
        "top1_logprob_series",
        "logprob_margin_series",
        "topk_entropy_series",
        "topk_varentropy_series",
        "topk_renyi2_series",
        "topk_tail_mass_series",
    ),
}
FAMILY_NAMES = tuple(BROAD_FAMILIES)

LOCAL_OPERATOR_ROSTERS = {
    "level": ("level",),
    "innovation": ("innovation",),
    "shortlong": ("shortlong",),
    "level_innovation": ("level", "innovation"),
    "level_shortlong": ("level", "shortlong"),
    "level_innovation_shortlong": ("level", "innovation", "shortlong"),
}

ONLINE_OPERATOR_ROSTERS = {
    "level_slow": ("level", "slow"),
    "fast_slow": ("fast", "slow"),
    "slow_area_persistence": ("slow", "positive_mean", "persistence"),
    "shortlong_innovation_recovery": ("shortlong", "innovation", "recovery"),
    "level_fast_slow_area_persistence": (
        "level", "fast", "slow", "positive_mean", "persistence",
    ),
}

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


def _risk_sign_for_view(name: str) -> float:
    mapped = TOKEN_TO_GLOBAL_FEATURES[name]
    signs = {int(CONFIDENCE_FEATURE_SIGNS_V1[item]) for item in mapped}
    if len(signs) != 1:
        raise ValueError(f"inconsistent confidence signs for {name}: {mapped}")
    return float(-next(iter(signs)))


BROAD_RISK_SIGNS = np.asarray(
    [_risk_sign_for_view(name) for name in BROAD_TOKEN_VIEWS], dtype=float
)


@dataclass(frozen=True)
class BroadReference:
    centres: np.ndarray
    scales: np.ndarray
    availability: np.ndarray
    n_traces: int
    positions_per_trace: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "views": list(BROAD_TOKEN_VIEWS),
            "centres": self.centres.tolist(),
            "scales": self.scales.tolist(),
            "availability": self.availability.tolist(),
            "n_traces": int(self.n_traces),
            "positions_per_trace": int(self.positions_per_trace),
        }


def raw_broad_risk_matrix(row: Mapping[str, Any]) -> np.ndarray:
    """Return the historical broad-28 token curves in frozen risk orientation."""

    views = token_feature_views(dict(row))
    matrix = np.column_stack([views[name] for name in BROAD_TOKEN_VIEWS])
    return np.asarray(matrix, dtype=float) * BROAD_RISK_SIGNS[None, :]


def fit_broad_reference(
    rows: Sequence[Mapping[str, Any]], *, positions_per_trace: int = FIT_POSITIONS
) -> BroadReference:
    samples = [[] for _ in BROAD_TOKEN_VIEWS]
    availability = np.zeros(len(BROAD_TOKEN_VIEWS), dtype=float)
    for row in rows:
        matrix = raw_broad_risk_matrix(row)
        positions = equal_positions(len(matrix), positions_per_trace)
        selected = matrix[positions]
        for index in range(selected.shape[1]):
            values = selected[:, index]
            availability[index] += float(np.mean(np.isfinite(values)))
            samples[index].append(values)
    centres, scales = [], []
    for columns in samples:
        values = np.concatenate(columns) if columns else np.zeros(0)
        centre, scale = _robust_location_scale(values)
        centres.append(centre)
        scales.append(scale)
    return BroadReference(
        centres=np.asarray(centres, dtype=float),
        scales=np.asarray(scales, dtype=float),
        availability=availability / max(len(rows), 1),
        n_traces=len(rows),
        positions_per_trace=int(positions_per_trace),
    )


def standardized_broad_matrix(
    row: Mapping[str, Any], reference: BroadReference
) -> np.ndarray:
    matrix = (raw_broad_risk_matrix(row) - reference.centres) / reference.scales
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return np.clip(matrix, -CLIP_Z, CLIP_Z)


def family_matrix(broad: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(BROAD_TOKEN_VIEWS)}
    output = []
    for family in FAMILY_NAMES:
        columns = [index[name] for name in BROAD_FAMILIES[family]]
        output.append(np.mean(broad[:, columns], axis=1))
    return np.column_stack(output)


@dataclass(frozen=True)
class References:
    raw: ChannelReference
    broad: BroadReference

    def as_dict(self) -> dict[str, Any]:
        return {"raw": self.raw.as_dict(), "broad": self.broad.as_dict()}


def fit_references(rows: Sequence[Mapping[str, Any]]) -> References:
    return References(
        raw=fit_channel_reference(rows),
        broad=fit_broad_reference(rows),
    )


def representation_matrix(
    row: Mapping[str, Any], references: References, representation: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    if representation in {"raw9", "raw7"}:
        channels = standardized_risk_channels(row, references.raw)
        names = tuple(CHANNEL_NAMES) if representation == "raw9" else RAW7_NAMES
        return np.column_stack([channels[name] for name in names]), names
    broad = standardized_broad_matrix(row, references.broad)
    if representation == "broad28":
        return broad, tuple(BROAD_TOKEN_VIEWS)
    if representation == "family6":
        return family_matrix(broad), FAMILY_NAMES
    raise KeyError(representation)


def causal_operator_matrices(level: np.ndarray) -> dict[str, np.ndarray]:
    """Return the frozen causal state matrices for a T-by-D risk matrix."""

    level = np.asarray(level, dtype=float)
    if level.ndim != 2 or not len(level):
        raise ValueError("level must be a non-empty T-by-D matrix")
    fast = np.empty_like(level)
    slow = np.empty_like(level)
    innovation = np.zeros_like(level)
    previous_fast = np.zeros(level.shape[1], dtype=float)
    previous_slow = np.zeros(level.shape[1], dtype=float)
    for index, values in enumerate(level):
        if index:
            innovation[index] = np.maximum(0.0, values - previous_slow)
        previous_fast = (1.0 - FAST_ALPHA) * previous_fast + FAST_ALPHA * values
        previous_slow = (1.0 - SLOW_ALPHA) * previous_slow + SLOW_ALPHA * values
        fast[index] = previous_fast
        slow[index] = previous_slow
    denominator = np.arange(1, len(level) + 1, dtype=float)[:, None]
    positive_mean = np.cumsum(np.maximum(level, 0.0), axis=0) / denominator
    persistence = np.cumsum(level > 0.0, axis=0) / denominator
    running_max = np.maximum.accumulate(level, axis=0)
    return {
        "level": level,
        "fast": fast,
        "slow": slow,
        "innovation": innovation,
        "shortlong": fast - slow,
        "positive_mean": positive_mean,
        "persistence": persistence,
        "recovery": level - running_max,
    }


def operator_matrix(level: np.ndarray, operators: Sequence[str]) -> np.ndarray:
    states = causal_operator_matrices(level)
    return np.column_stack([states[name] for name in operators])


@dataclass(frozen=True)
class FrozenTrajectoryHead:
    name: str
    representation: str
    operators: tuple[str, ...]
    feature_names: tuple[str, ...]
    keep: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    flipped: bool
    diagnostics: Mapping[str, Any]

    def curve_from_level(self, level: np.ndarray) -> np.ndarray:
        raw = operator_matrix(level, self.operators)
        standardized = (raw[:, self.keep] - self.mean) / self.std
        score = standardized @ self.weights
        return -score if self.flipped else score

    def curve(self, row: Mapping[str, Any], references: References) -> np.ndarray:
        level, _ = representation_matrix(row, references, self.representation)
        return self.curve_from_level(level)


@dataclass(frozen=True)
class PreparedTrace:
    representations: Mapping[str, np.ndarray]
    names: Mapping[str, tuple[str, ...]]


def prepare_trace(row: Mapping[str, Any], references: References) -> PreparedTrace:
    raw9, raw9_names = representation_matrix(row, references, "raw9")
    broad, broad_names = representation_matrix(row, references, "broad28")
    raw_index = {name: index for index, name in enumerate(raw9_names)}
    raw7 = np.column_stack([raw9[:, raw_index[name]] for name in RAW7_NAMES])
    family = family_matrix(broad)
    return PreparedTrace(
        representations={
            "raw9": raw9,
            "raw7": raw7,
            "broad28": broad,
            "family6": family,
        },
        names={
            "raw9": raw9_names,
            "raw7": RAW7_NAMES,
            "broad28": broad_names,
            "family6": FAMILY_NAMES,
        },
    )


def fit_trajectory_head(
    rows: Sequence[Mapping[str, Any]],
    references: References,
    *,
    name: str,
    representation: str,
    operators: Sequence[str],
    positions_per_trace: int = FIT_POSITIONS,
) -> FrozenTrajectoryHead:
    prepared = [prepare_trace(row, references) for row in rows]
    return fit_trajectory_head_prepared(
        prepared,
        name=name,
        representation=representation,
        operators=operators,
        positions_per_trace=positions_per_trace,
    )


def fit_trajectory_head_prepared(
    rows: Sequence[PreparedTrace],
    *,
    name: str,
    representation: str,
    operators: Sequence[str],
    positions_per_trace: int = FIT_POSITIONS,
) -> FrozenTrajectoryHead:
    sampled = []
    base_names: tuple[str, ...] | None = None
    for row in rows:
        level = np.asarray(row.representations[representation], dtype=float)
        current_names = tuple(row.names[representation])
        if base_names is None:
            base_names = current_names
        elif current_names != base_names:
            raise RuntimeError("representation feature order changed")
        matrix = operator_matrix(level, operators)
        sampled.append(matrix[equal_positions(len(matrix), positions_per_trace)])
    if base_names is None:
        raise ValueError("at least one calibration row is required")
    raw = np.vstack(sampled)
    median = np.nanmedian(raw, axis=0)
    clean = np.where(np.isfinite(raw), raw, median)
    spread = np.std(clean, axis=0)
    keep = np.isfinite(median) & np.isfinite(spread) & (spread > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError(f"{name}: fewer than three non-degenerate coordinates")
    selected = clean[:, keep]
    mean = selected.mean(axis=0)
    std = selected.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    standardized = (selected - mean) / std
    fitted = upcr_fit(standardized.T, **IU_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score = standardized @ weights
    consensus = standardized.mean(axis=1)
    correlation = float(np.corrcoef(score, consensus)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    names = tuple(
        f"{base}__{operator}"
        for operator in operators
        for base in base_names
    )
    return FrozenTrajectoryHead(
        name=name,
        representation=representation,
        operators=tuple(operators),
        feature_names=names,
        keep=keep,
        mean=mean,
        std=std,
        weights=weights,
        flipped=flipped,
        diagnostics={
            "labels_seen_during_fit": False,
            "representation": representation,
            "operators": list(operators),
            "input_coordinates": int(raw.shape[1]),
            "retained_coordinates": int(keep.sum()),
            "n_fit_rows": int(len(raw)),
            "g2_hat": float(fitted.g2_hat),
            "orientation_correlation_before_flip": correlation,
            "orientation_flipped": flipped,
        },
    )


def local_candidate_roster() -> dict[str, tuple[str, tuple[str, ...]]]:
    roster: dict[str, tuple[str, tuple[str, ...]]] = {}
    for representation in ("raw9", "broad28", "family6"):
        for suffix, operators in LOCAL_OPERATOR_ROSTERS.items():
            roster[f"l_{representation}__{suffix}"] = (representation, operators)
    roster["l_raw7_opened_drop__level"] = ("raw7", ("level",))
    return roster


def online_candidate_roster() -> dict[str, tuple[str, tuple[str, ...]]]:
    roster: dict[str, tuple[str, tuple[str, ...]]] = {}
    for representation in ("raw9", "broad28", "family6"):
        for suffix, operators in ONLINE_OPERATOR_ROSTERS.items():
            roster[f"o_{representation}__{suffix}"] = (representation, operators)
    return roster


__all__ = [
    "BROAD_FAMILIES",
    "BROAD_RISK_SIGNS",
    "FAMILY_NAMES",
    "LOCAL_OPERATOR_ROSTERS",
    "ONLINE_OPERATOR_ROSTERS",
    "RAW7_NAMES",
    "BroadReference",
    "References",
    "FrozenTrajectoryHead",
    "PreparedTrace",
    "causal_operator_matrices",
    "family_matrix",
    "fit_broad_reference",
    "fit_references",
    "fit_trajectory_head",
    "fit_trajectory_head_prepared",
    "local_candidate_roster",
    "online_candidate_roster",
    "operator_matrix",
    "prepare_trace",
    "raw_broad_risk_matrix",
    "representation_matrix",
    "standardized_broad_matrix",
]
