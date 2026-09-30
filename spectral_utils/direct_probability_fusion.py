"""Label-free fusion over sorted next-token probabilities.

The observation axis can be tokens from one answer (localization) or answers
from one benchmark cell (complete-answer detection).  Columns are vocabulary
probability ranks, kept as direct probabilities rather than collapsed to
entropy or another scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .laplacian_upcr import IU_FIT_DEFAULTS
from .shrinkage_iu import ledoit_wolf_alpha, shrink, target_matrix
from .upcr import upcr_fit, upcr_fit_covariance


EPS = 1e-12
METHODS = ("equal", "iu", "joint_lw")


@dataclass
class DirectProbabilityFit:
    score: np.ndarray
    method: str
    weights: np.ndarray
    kept_ranks: np.ndarray
    orientation_flipped: bool
    alpha: float | None
    fallback: str | None
    diagnostics: dict[str, Any]


def logprob_matrix(value: Any, k: int = 15) -> np.ndarray:
    """Return finite descending ``[observations, k]`` saved log-probabilities."""

    if not isinstance(value, dict) or value.get("logprobs") is None:
        raise ValueError("top-k value must contain a logprobs matrix")
    matrix = np.asarray(value["logprobs"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < int(k):
        raise ValueError(f"need a nonempty [N,K] matrix with K>={k}, got {matrix.shape}")
    matrix = matrix[:, : int(k)]
    if not np.isfinite(matrix).all():
        raise ValueError("log-probability matrix contains nonfinite values")
    if matrix.shape[1] > 1 and not np.all(np.diff(matrix, axis=1) <= 1e-7):
        raise ValueError("log-probability ranks are not descending")
    return matrix


def direct_rank_risk(logprobs: np.ndarray) -> np.ndarray:
    """Risk-oriented direct probabilities, without top-K renormalization.

    A small rank-1 probability signals uncertainty, so rank 1 is oriented as
    ``1-p1``.  Larger probabilities in ranks 2..K signal a flatter competing
    head and already point toward risk.
    """

    probabilities = np.exp(np.asarray(logprobs, dtype=float))
    risk = probabilities.copy()
    risk[:, 0] = 1.0 - probabilities[:, 0]
    return risk


def topk_renormalized_entropy(logprobs: np.ndarray) -> np.ndarray:
    """Reference entropy on exactly the retained support."""

    probabilities = np.exp(np.asarray(logprobs, dtype=float))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), EPS)
    return -np.sum(probabilities * np.log(np.maximum(probabilities, EPS)), axis=1)


def topk_renormalized_varentropy(logprobs: np.ndarray) -> np.ndarray:
    """Per-observation varentropy on exactly the retained support."""

    probabilities = np.exp(np.asarray(logprobs, dtype=float))
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), EPS)
    surprisal = -np.log(np.maximum(probabilities, EPS))
    mean = np.sum(probabilities * surprisal, axis=1, keepdims=True)
    return np.sum(probabilities * (surprisal - mean) ** 2, axis=1)


def zscore_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    keep = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-10)
    if not keep.any():
        return np.zeros((len(matrix), 0), dtype=float), keep, mean, scale
    return (matrix[:, keep] - mean[keep]) / scale[keep], keep, mean, scale


def _orient(score: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, bool, float]:
    score = np.asarray(score, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if score.shape != anchor.shape:
        raise ValueError("score and anchor lengths differ")
    if np.std(score) <= EPS or np.std(anchor) <= EPS:
        return score, False, float("nan")
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        return -score, True, correlation
    return score, False, correlation


def fit_rank_fusion(
    rank_risk: np.ndarray,
    *,
    method: str,
    anchor: np.ndarray,
) -> DirectProbabilityFit:
    """Fit one label-free rank fusion and score the fitting observations."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    values = np.asarray(rank_risk, dtype=float)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 3:
        raise ValueError("rank fusion needs at least 3 observations and 3 ranks")
    if not np.isfinite(values).all():
        raise ValueError("rank risk matrix contains nonfinite values")
    Z, keep, _, _ = zscore_columns(values)
    if Z.shape[1] < 3:
        raise ValueError("fewer than three nonconstant probability ranks")

    alpha: float | None = None
    fallback: str | None = None
    diagnostics: dict[str, Any] = {}
    if method == "equal":
        weights = np.full(Z.shape[1], 1.0 / Z.shape[1])
    else:
        reference = upcr_fit(Z.T, **dict(IU_FIT_DEFAULTS))
        diagnostics.update(
            g2_hat=float(reference.g2_hat),
            iu_abstained=bool(reference.abstained),
            n_components=int(reference.n_components_used),
        )
        weights = np.asarray(reference.w, dtype=float)
        if method == "joint_lw":
            covariance = Z.T @ Z / len(Z)
            groups = np.arange(Z.shape[1], dtype=int)
            target = target_matrix(covariance, groups, "joint")
            alpha = ledoit_wolf_alpha(Z, covariance, target)
            shrunk = shrink(covariance, target, alpha)
            try:
                result = upcr_fit_covariance(shrunk, **dict(IU_FIT_DEFAULTS))
                candidate = np.asarray(result.w, dtype=float)
                if np.isfinite(candidate).all() and np.linalg.norm(candidate) > EPS:
                    weights = candidate
                    diagnostics["joint_abstained"] = bool(result.abstained)
                else:
                    fallback = "iu_nonfinite_or_zero_joint_weights"
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                fallback = f"iu_joint_fit_error:{type(exc).__name__}"
    score = Z @ weights
    score, flipped, correlation = _orient(score, np.asarray(anchor, dtype=float))
    diagnostics["pre_orientation_anchor_correlation"] = correlation
    full_weights = np.zeros(values.shape[1], dtype=float)
    full_weights[keep] = (-weights if flipped else weights)
    return DirectProbabilityFit(
        score=score,
        method=method,
        weights=full_weights,
        kept_ranks=np.flatnonzero(keep),
        orientation_flipped=flipped,
        alpha=alpha,
        fallback=fallback,
        diagnostics=diagnostics,
    )


def top_mean(values: np.ndarray, count: int = 10) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("top-mean needs a nonempty finite vector")
    n = min(int(count), len(values))
    return float(np.partition(values, len(values) - n)[-n:].mean())


def step_top_mean(
    token_scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    count: int = 10,
) -> np.ndarray:
    scores = np.asarray(token_scores, dtype=float)
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    if starts.shape != ends.shape or starts.ndim != 1:
        raise ValueError("step starts and ends must be aligned vectors")
    output = []
    for start, end in zip(starts, ends):
        if start < 0 or end <= start or end > len(scores):
            raise ValueError(f"invalid step span [{start}, {end}) for {len(scores)} tokens")
        output.append(top_mean(scores[start:end], count=count))
    return np.asarray(output, dtype=float)


def answer_rank_features(rank_risk: np.ndarray, count: int = 10) -> np.ndarray:
    """Aggregate each direct probability rank across one complete answer."""

    values = np.asarray(rank_risk, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("answer rank features need a nonempty [tokens,ranks] matrix")
    return np.asarray([top_mean(values[:, index], count=count) for index in range(values.shape[1])])


__all__ = [
    "DEFAULT_K",
    "DirectProbabilityFit",
    "METHODS",
    "answer_rank_features",
    "direct_rank_risk",
    "fit_rank_fusion",
    "logprob_matrix",
    "step_top_mean",
    "top_mean",
    "topk_renormalized_entropy",
    "topk_renormalized_varentropy",
    "zscore_columns",
]


DEFAULT_K = 15
