"""Module B — learned trajectory-axis reducers over step order statistics.

Protocol: docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md, Section 6B.

Raw token positions are not aligned units across steps, but order statistics
are: the sorted top-k token risks within a step ("largest", "2nd largest", ...)
are exchangeable views of the step's risk level.  This module extracts those
views and fits reducer weights over them.

Replay rule (pre-registered): a step/span with L < k tokens scores with the
weight vector truncated to its first L slots and renormalized to unit sum —
which is exactly how the incumbent equal-weight top-min(10, L) mean already
behaves, so the equal-weight vector reproduces the frozen incumbent bit-exactly
(the Module-B identity test).

B3 (the supervised LR competitor) accepts labels as an argument and is called
ONLY from the evaluator stage, per SUPERVISED_ORACLE_CORRECTION.md
(class_weight='balanced', fold-scoped fits, no cross_val_predict calibration).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fusion_utils import sml_fuse_signed


ORDERSTAT_K = 10
POSITION_BINS = 5
_EPS = 1e-12


def step_order_statistics(
    token_risk: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    k: int = ORDERSTAT_K,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_steps, k) descending order statistics + per-step available lengths.

    Slots beyond a short step's length are filled with that step's LAST
    available order statistic; scoring functions never read them (they weight
    only the first L slots), the fill only keeps the matrix rectangular for
    the label-free fit, which uses full-length steps exclusively.
    """
    risk = np.asarray(token_risk, dtype=np.float64)
    n_steps = len(starts)
    matrix = np.empty((n_steps, int(k)), dtype=np.float64)
    lengths = np.empty(n_steps, dtype=np.int64)
    for row, (lo, hi) in enumerate(zip(starts, ends)):
        values = risk[int(lo):int(hi)]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"step {row} has no finite token risk")
        top = np.sort(finite)[::-1][: int(k)]
        lengths[row] = len(top)
        matrix[row, : len(top)] = top
        if len(top) < int(k):
            matrix[row, len(top):] = top[-1]
    return matrix, lengths


def reduce_with_weights(
    matrix: np.ndarray,
    lengths: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    """Apply reducer weights with the renormalized-truncation replay rule."""
    w = np.asarray(weights, dtype=np.float64)
    values = np.asarray(matrix, dtype=np.float64)
    if w.shape != (values.shape[1],) or not np.isfinite(w).all():
        raise ValueError("reducer weights are malformed")
    output = np.empty(len(values), dtype=np.float64)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        active = w[: int(length)]
        total = float(active.sum())
        if abs(total) < _EPS:
            raise RuntimeError("truncated reducer weights sum to zero")
        output[rows] = values[np.ix_(rows, np.arange(int(length)))] @ (active / total)
    return output


def equal_topk_weights(k: int = ORDERSTAT_K) -> np.ndarray:
    """The incumbent: equal weights == the frozen top-min(k, L) mean (B0)."""
    return np.ones(int(k), dtype=np.float64) / float(k)


def max_weights(k: int = ORDERSTAT_K) -> np.ndarray:
    """The PRMBench incumbent: top-1 (span max)."""
    weights = np.zeros(int(k), dtype=np.float64)
    weights[0] = 1.0
    return weights


def blend_step_scores(
    matrix: np.ndarray, lengths: np.ndarray, alpha: float, *, k: int = ORDERSTAT_K
) -> np.ndarray:
    """B2a: alpha*max + (1-alpha)*top-k-mean; alpha=0 reproduces B0 exactly."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    top_mean = reduce_with_weights(matrix, lengths, equal_topk_weights(k))
    top_one = matrix[:, 0]
    return alpha * top_one + (1.0 - alpha) * top_mean


def _scale_orient(
    weights: np.ndarray, matrix: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    score = reduce_with_weights(matrix, lengths, weights)
    sd = float(score.std())
    if not np.isfinite(sd) or sd < 1e-8:
        raise RuntimeError("reducer score SD is degenerate")
    anchor = matrix.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0.0)
    output = -weights if flipped else weights.copy()
    return output, {
        "score_sd": sd,
        "anchor_correlation": correlation,
        "orientation_flipped": flipped,
    }


def fit_orderstat_weights(
    matrix: np.ndarray, lengths: np.ndarray, *, k: int = ORDERSTAT_K
) -> tuple[np.ndarray, dict[str, Any]]:
    """B1: label-free SML weights over the k order-statistic views.

    Fitted on FULL-length steps only (L >= k); the replay rule extends the
    learned vector to shorter steps at application time.
    """
    full = np.flatnonzero(np.asarray(lengths) >= int(k))
    if len(full) < 50:
        raise RuntimeError(f"only {len(full)} full-length steps; B1 needs >= 50")
    views = np.asarray(matrix, dtype=np.float64)[full]
    _, weights = sml_fuse_signed(*[views[:, index] for index in range(int(k))])
    weights = np.asarray(weights, dtype=np.float64)
    oriented, meta = _scale_orient(weights, views, np.full(len(full), int(k)))
    return oriented, {
        **meta,
        "n_fit_steps": int(len(full)),
        "weight_profile": oriented.tolist(),
        "labels_accessed": False,
    }


def step_position_bins(
    token_risk: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    bins: int = POSITION_BINS,
) -> np.ndarray:
    """(n_steps, bins) mean token risk per relative-position bin.

    Replay rule: a bin with no tokens (very short steps) takes the step's
    overall mean.
    """
    risk = np.asarray(token_risk, dtype=np.float64)
    output = np.empty((len(starts), int(bins)), dtype=np.float64)
    for row, (lo, hi) in enumerate(zip(starts, ends)):
        values = risk[int(lo):int(hi)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"step {row} has no finite token risk")
        edges = np.linspace(0, len(values), int(bins) + 1).astype(int)
        overall = float(values.mean())
        for bin_index in range(int(bins)):
            chunk = values[edges[bin_index]:edges[bin_index + 1]]
            output[row, bin_index] = float(chunk.mean()) if chunk.size else overall
    return output


def fit_position_bin_weights(bin_matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """B2b: label-free SML weights over the positional-bin views."""
    views = np.asarray(bin_matrix, dtype=np.float64)
    _, weights = sml_fuse_signed(*[views[:, index] for index in range(views.shape[1])])
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if abs(total) < _EPS:
        raise RuntimeError("positional-bin weights sum to zero")
    weights = weights / total
    score = views @ weights
    anchor = views.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0.0:
        weights = -weights
    return weights, {
        "anchor_correlation": correlation,
        "weight_profile": weights.tolist(),
        "labels_accessed": False,
    }


def fit_lr_orderstats(
    matrix: np.ndarray,
    lengths: np.ndarray,
    labels: Sequence[int],
    *,
    k: int = ORDERSTAT_K,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """B3 — SUPERVISED LR competitor.  Evaluator-stage only.

    Trains on full-length steps (L >= k) with balanced class weights per
    SUPERVISED_ORACLE_CORRECTION.md; the caller is responsible for restricting
    ``labels`` to its training fold and (on ProcessBench) for excluding
    post-first-error steps before calling.
    """
    from sklearn.linear_model import LogisticRegression

    target = np.asarray(labels, dtype=np.int64)
    values = np.asarray(matrix, dtype=np.float64)
    full = np.flatnonzero(np.asarray(lengths) >= int(k))
    if len(np.unique(target[full])) < 2:
        raise RuntimeError("LR training fold has a single class")
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=int(seed)
    )
    model.fit(values[full], target[full])
    return model, {
        "n_train_steps": int(len(full)),
        "coefficient_profile": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "supervised": True,
    }


def score_lr_orderstats(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Decision-function scores for every step (rectangular fill is harmless
    because the LR consumed full-length steps and short steps reuse their last
    order statistic, a monotone-consistent imputation disclosed in the
    protocol)."""
    return np.asarray(model.decision_function(np.asarray(matrix, dtype=np.float64)), dtype=np.float64)


__all__ = [
    "ORDERSTAT_K", "POSITION_BINS", "blend_step_scores", "equal_topk_weights",
    "fit_lr_orderstats", "fit_orderstat_weights", "fit_position_bin_weights",
    "max_weights", "reduce_with_weights", "score_lr_orderstats",
    "step_order_statistics", "step_position_bins",
]
