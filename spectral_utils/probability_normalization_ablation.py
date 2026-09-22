"""Probability-support and preprocessing ablations for Renyi/varentropy views.

The module accepts no benchmark labels.  It preserves the completed Stage-3b
single-view definitions and exposes only deterministic score construction and
group-balanced moment fitting.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .direct_probability_fusion import logprob_matrix, step_top_mean
from .direct_probability_fusion_v2 import residual_tail_mass
from .renyi_position_fusion import feature_bank as frozen_four_view_bank


EPS = 1e-12
FEATURE4 = ("H0lim", "ve0", "ve0.75", "ve1")
LOCAL_METHODS = (
    "ve075_q15_raw",
    "ve075_q50_raw",
    "ve075_tailbucket15_raw",
    "ve1_q15_raw",
    "ve1_q50_raw",
    "ve1_tailbucket15_raw",
    "tail15_raw",
    "equal4_raw",
    "equal4_center_only",
    "equal4_scale_only",
    "equal4_answer_z",
)
EXTERNAL_METHODS = ("equal4_fold_global_z", "equal5_tail_fold_global_z")
METHODS = LOCAL_METHODS + EXTERNAL_METHODS
PRIMARY = (
    ("ve075_q50_raw", "ve075_q15_raw"),
    ("ve075_tailbucket15_raw", "ve075_q15_raw"),
    ("ve1_q50_raw", "ve1_q15_raw"),
    ("ve1_tailbucket15_raw", "ve1_q15_raw"),
    ("equal4_fold_global_z", "equal4_answer_z"),
    ("equal5_tail_fold_global_z", "equal4_fold_global_z"),
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    training_groups: tuple[str, ...]
    training_answers: int


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(left, right)[0, 1])


def orient_to_anchor(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float, bool]:
    values = np.asarray(values, dtype=np.float64)
    correlation = _correlation(values, anchor)
    flipped = bool(np.isfinite(correlation) and correlation < 0)
    return (-values if flipped else values), correlation, flipped


def conditional_head(logprobs, k: int) -> tuple[np.ndarray, np.ndarray]:
    lp = logprob_matrix({"logprobs": logprobs}, k=k)
    p = np.exp(lp)
    q = p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
    return p, q


def escort_varentropy_distribution(probability: np.ndarray, alpha: float) -> np.ndarray:
    """Escort varentropy on a row-normalized probability distribution."""
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 2 or not len(probability) or not np.isfinite(probability).all():
        raise ValueError("escort varentropy needs a finite nonempty matrix")
    if (probability < 0).any():
        raise ValueError("probability cannot be negative")
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=2e-10, rtol=0)
    if float(alpha) == 0.0:
        weights = np.full_like(probability, 1.0 / probability.shape[1])
    else:
        weights = probability ** float(alpha)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    surprisal = -np.log(probability + EPS)
    mean = np.sum(weights * surprisal, axis=1)
    return np.sum(weights * surprisal * surprisal, axis=1) - mean * mean


def escort_varentropy_raw_head(probability: np.ndarray, alpha: float) -> np.ndarray:
    """Proper escort variance using an unnormalized retained probability head.

    Scaling every row by a positive constant cannot change the normalized
    escort weights, and it only shifts every surprisal by one row-constant.
    The result must therefore match the conditional-head calculation.
    """
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 2 or not len(probability) or not np.isfinite(probability).all():
        raise ValueError("raw head must be a finite nonempty matrix")
    if (probability <= 0).any():
        raise ValueError("raw head probabilities must be positive")
    if float(alpha) == 0.0:
        weights = np.full_like(probability, 1.0 / probability.shape[1])
    else:
        weights = probability ** float(alpha)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    surprisal = -np.log(probability + EPS)
    mean = np.sum(weights * surprisal, axis=1)
    return np.sum(weights * surprisal * surprisal, axis=1) - mean * mean


def tail_bucket_distribution(logprobs, k: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Coarsen the full vocabulary to K saved tokens plus one residual bucket."""
    lp = logprob_matrix({"logprobs": logprobs}, k=k)
    p = np.exp(lp)
    tail = residual_tail_mass(lp)
    distribution = np.column_stack((p, tail))
    # Float32 saved log-probabilities can exceed mass one by <5e-7.  The tail
    # helper accepts that known capture tolerance; make the coarse distribution
    # exactly normalized after clipping the residual bucket.
    distribution /= np.maximum(distribution.sum(axis=1, keepdims=True), EPS)
    np.testing.assert_allclose(distribution.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    return distribution, tail


def _moments(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0), "second": np.mean(values * values, axis=0)}


def feature_bank(logprobs) -> dict:
    """Return frozen singles, preprocessing arms and fit sufficient statistics."""
    frozen = frozen_four_view_bank(logprobs)
    p15, q15 = conditional_head(logprobs, 15)
    p50, q50 = conditional_head(logprobs, 50)
    coarse15, tail15 = tail_bucket_distribution(logprobs, 15)

    ve075_q15 = escort_varentropy_distribution(q15, 0.75)
    ve075_q50 = escort_varentropy_distribution(q50, 0.75)
    ve075_coarse = escort_varentropy_distribution(coarse15, 0.75)
    ve1_q15 = escort_varentropy_distribution(q15, 1.0)
    ve1_q50 = escort_varentropy_distribution(q50, 1.0)
    ve1_coarse = escort_varentropy_distribution(coarse15, 1.0)

    invariant = {
        "ve075_p15_vs_q15": float(np.max(np.abs(escort_varentropy_raw_head(p15, 0.75) - ve075_q15))),
        "ve075_p50_vs_q50": float(np.max(np.abs(escort_varentropy_raw_head(p50, 0.75) - ve075_q50))),
        "ve1_p15_vs_q15": float(np.max(np.abs(escort_varentropy_raw_head(p15, 1.0) - ve1_q15))),
        "ve1_p50_vs_q50": float(np.max(np.abs(escort_varentropy_raw_head(p50, 1.0) - ve1_q50))),
    }

    anchor = np.asarray(frozen["anchor"], dtype=np.float64)
    singles = {
        "ve075_q15_raw": np.asarray(frozen["singles"]["view__ve0.75"], dtype=np.float64),
        "ve1_q15_raw": np.asarray(frozen["singles"]["view__ve1"], dtype=np.float64),
        "tail15_raw": np.asarray(tail15, dtype=np.float64),
    }
    orientation = {}
    for name, values in (
        ("ve075_q50_raw", ve075_q50),
        ("ve075_tailbucket15_raw", ve075_coarse),
        ("ve1_q50_raw", ve1_q50),
        ("ve1_tailbucket15_raw", ve1_coarse),
    ):
        singles[name], correlation, flipped = orient_to_anchor(values, anchor)
        orientation[name] = {"anchor_correlation": correlation, "flipped": flipped}

    # Reproduce the exact completed-fusion orientation before changing only the
    # centering/scaling operator.
    oriented4 = (
        np.asarray(frozen["raw"], dtype=np.float64)
        * np.asarray(frozen["single_signs"], dtype=np.float64)
        * np.asarray(frozen["fusion_signs"], dtype=np.float64)
    )
    scale4 = oriented4.std(axis=0)
    if np.any(scale4 <= 1e-10):
        raise ValueError("four-view answer has a near-constant column")
    answer_z = (oriented4 - oriented4.mean(axis=0)) / scale4
    np.testing.assert_allclose(answer_z, frozen["z"], atol=1e-12, rtol=1e-10)
    oriented5 = np.column_stack((oriented4, tail15))

    local = dict(singles)
    local.update(
        equal4_raw=oriented4.mean(axis=1),
        equal4_center_only=(oriented4 - oriented4.mean(axis=0)).mean(axis=1),
        equal4_scale_only=(oriented4 / scale4).mean(axis=1),
        equal4_answer_z=answer_z.mean(axis=1),
    )
    if set(local) != set(LOCAL_METHODS):
        raise AssertionError("local method roster drift")
    if not all(np.isfinite(values).all() for values in local.values()):
        raise ValueError("nonfinite local score")
    return {
        "local": local,
        "oriented4": oriented4,
        "oriented5": oriented5,
        "moments4": _moments(oriented4),
        "moments5": _moments(oriented5),
        "tail15": tail15,
        "invariance_error": invariant,
        "orientation": orientation,
    }


def fit_grouped_standardizer(
    statistics: Mapping[int, Mapping[str, np.ndarray]],
    group_ids: Mapping[int, str],
    selected: Sequence[int],
    moment_key: str,
) -> Standardizer:
    """Fit equal-group/equal-answer global moments without token-length weighting."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        grouped[str(group_ids[index])].append(int(index))
    if not grouped:
        raise ValueError("standardizer needs at least one training group")
    group_means, group_seconds = [], []
    for group in sorted(grouped):
        members = grouped[group]
        group_means.append(np.mean([statistics[i][moment_key + "_mean"] for i in members], axis=0))
        group_seconds.append(np.mean([statistics[i][moment_key + "_second"] for i in members], axis=0))
    mean = np.mean(group_means, axis=0)
    second = np.mean(group_seconds, axis=0)
    variance = np.maximum(second - mean * mean, 0.0)
    scale = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 1e-10):
        raise ValueError("global standardizer has an invalid column")
    return Standardizer(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        training_groups=tuple(sorted(grouped)),
        training_answers=int(sum(map(len, grouped.values()))),
    )


def global_equal_score(values: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(standardizer.mean):
        raise ValueError("values and standardizer width differ")
    score = ((values - standardizer.mean) / standardizer.scale).mean(axis=1)
    if not np.isfinite(score).all():
        raise ValueError("nonfinite globally standardized score")
    return score


def step_readout(token_score: np.ndarray, spans: np.ndarray) -> np.ndarray:
    spans = np.asarray(spans, dtype=np.int64)
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError("step spans must be [steps,2]")
    return step_top_mean(token_score, spans[:, 0], spans[:, 1], count=10)


__all__ = [
    "EPS",
    "EXTERNAL_METHODS",
    "FEATURE4",
    "LOCAL_METHODS",
    "METHODS",
    "PRIMARY",
    "Standardizer",
    "conditional_head",
    "escort_varentropy_distribution",
    "escort_varentropy_raw_head",
    "feature_bank",
    "fit_grouped_standardizer",
    "global_equal_score",
    "orient_to_anchor",
    "step_readout",
    "tail_bucket_distribution",
]
