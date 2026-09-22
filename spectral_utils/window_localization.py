"""Window geometry and feature measurements for explicit per-answer fitting.

No labels, model fitting, cross-answer normalization or hidden pooled fallback.
Feature definitions come from the existing response-level extractors. Window
coordinates are half-open token intervals. Short-window placeholders are barred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .feature_utils import FEAT_NAMES, extract_all_features, compute_spilled_energy_features


ENERGY_NAMES = ("epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy")
DISTRIBUTION_NAMES = (
    "mean_top1_logprob", "logprob_margin", "mean_logprob_entropy",
    "varentropy", "renyi_entropy_2", "topk_tail_mass",
)
DISTRIBUTION_STREAMS = (
    "top1_logprob_series", "logprob_margin_series", "topk_entropy_series",
    "topk_varentropy_series", "topk_renyi2_series", "topk_tail_mass_series",
)
WINDOW_FEATURE_NAMES = tuple(FEAT_NAMES) + ENERGY_NAMES + DISTRIBUTION_NAMES
FEATURE_MIN_WIDTH = 32  # Existing STFT returns placeholder zeros below 32.
DEFAULT_WIDTHS = (32, 48, 64, 96, 128)


@dataclass(frozen=True)
class WindowPlan:
    token_count: int
    width: int
    stride: int
    starts: np.ndarray
    ends: np.ndarray
    fit_indices: np.ndarray

    @property
    def nonoverlapping_capacity(self) -> int:
        """Nonoverlapping capacity, NOT an estimate of independent samples."""
        return self.token_count // self.width


@dataclass(frozen=True)
class WindowMatrix:
    plan: WindowPlan
    values: np.ndarray
    feature_names: tuple[str, ...]
    active: np.ndarray
    inactive_reasons: tuple[str | None, ...]

    @property
    def fit_values(self) -> np.ndarray:
        return self.values[self.plan.fit_indices][:, self.active]


def make_window_plan(token_count: int, width: int, stride: int | None = None) -> WindowPlan:
    """Dense scoring support plus an explicitly nonoverlapping fit subset.

    Add full-width windows anchored at both ends; never pad a short remainder.
    Include the nonoverlapping fit grid even when stride does not divide width.
    The added final window can overlap but is not added to the fit subset.
    """
    token_count, width = int(token_count), int(width)
    stride = int(stride if stride is not None else width)
    if width < FEATURE_MIN_WIDTH:
        raise ValueError(f"width must be >= {FEATURE_MIN_WIDTH}; STFT placeholders are not data")
    if stride < 1 or stride > width:
        raise ValueError("stride must be in [1, width] to ensure complete token coverage")
    if token_count < width:
        raise ValueError("TRACE_TOO_SHORT_FOR_FEATURES")
    fit_starts = np.arange(0, token_count - width + 1, width, dtype=np.int64)
    starts = np.unique(np.concatenate((
        np.arange(0, token_count - width + 1, stride, dtype=np.int64),
        fit_starts, np.asarray([token_count - width], dtype=np.int64),
    )))
    fit_indices = np.searchsorted(starts, fit_starts)
    return WindowPlan(token_count, width, stride, starts, starts + width, fit_indices)


def feasible_widths(
    token_count: int, *, widths: Sequence[int] = DEFAULT_WIDTHS, min_fit_windows: int = 8,
) -> list[dict]:
    """Report geometry; do not mislabel a geometry filter as statistical optimality.

    The minimum of eight is a configurable engineering floor, not a theorem.
    Model-specific stability checks must follow before selecting a width.
    """
    if min_fit_windows < 2:
        raise ValueError("min_fit_windows must be at least two")
    rows = []
    for width in sorted(set(map(int, widths))):
        if width < 1:
            raise ValueError("widths must be positive")
        count = int(token_count) // width
        reasons = []
        if width < FEATURE_MIN_WIDTH:
            reasons.append("FEATURE_WINDOW_TOO_SHORT")
        if count < min_fit_windows:
            reasons.append("TOO_FEW_NONOVERLAPPING_WINDOWS")
        rows.append({
            "width": width, "fit_windows": count,
            "remainder_tokens": int(token_count) % width,
            "feasible_geometry": not reasons, "reasons": reasons,
            "sample_covariance_rank_cap": max(0, min(len(WINDOW_FEATURE_NAMES) - 1, count - 1)),
        })
    return rows


def build_window_matrix(
    raw: np.ndarray, stream_names: Sequence[str], plan: WindowPlan,
) -> WindowMatrix:
    """Recompute the P=30 global feature definitions inside every full window.

    Raw entropy, sampled-token energy and log-partition are sufficient to
    recompute their window features. The six distribution views are means of
    their exact saved per-token counterparts. Precomputed rolling features are
    deliberately not averaged as substitutes for the original definitions.
    """
    raw = np.asarray(raw, dtype=np.float64)
    names = tuple(map(str, stream_names))
    if raw.ndim != 2 or raw.shape != (plan.token_count, len(names)) or len(set(names)) != len(names):
        raise ValueError("raw telemetry shape/schema does not match this answer")
    if "entropy_series" not in names:
        raise ValueError("entropy_series is required")
    columns = {name: raw[:, i] for i, name in enumerate(names)}
    output = np.full((len(plan.starts), len(WINDOW_FEATURE_NAMES)), np.nan)
    for i, (lo, hi) in enumerate(zip(plan.starts, plan.ends)):
        entropy = columns["entropy_series"][lo:hi]
        spilled = columns.get("spilled_series")
        spilled = None if spilled is None else spilled[lo:hi]
        with np.errstate(all="ignore"):
            if not np.isfinite(entropy).all():
                continue
            features = dict(extract_all_features(entropy, spilled_energies=spilled) or {})
            energy = columns.get("energy_series")
            if energy is not None and np.isfinite(energy[lo:hi]).all():
                f = compute_spilled_energy_features(energy[lo:hi])
                features.update({
                    "epr_energy": f["epr_spilled"], "min_energy": f["min_spilled"],
                    "sw_var_peak_energy": f["sw_var_peak_spilled"],
                    "cusum_max_energy": f["cusum_max_spilled"],
                })
            for feature, stream in zip(DISTRIBUTION_NAMES, DISTRIBUTION_STREAMS):
                if stream in columns:
                    features[feature] = float(np.mean(columns[stream][lo:hi]))
        output[i] = [features.get(name, np.nan) for name in WINDOW_FEATURE_NAMES]
    fit = output[plan.fit_indices]
    active = np.ones(output.shape[1], dtype=bool)
    reasons: list[str | None] = [None] * output.shape[1]
    for j in range(output.shape[1]):
        if not np.isfinite(fit[:, j]).all():
            active[j], reasons[j] = False, "UNAVAILABLE_OR_NONFINITE_ON_FIT_WINDOWS"
        elif np.ptp(fit[:, j]) <= 1e-10 * max(1.0, float(np.max(np.abs(fit[:, j])))):
            active[j], reasons[j] = False, "CONSTANT_ON_FIT_WINDOWS"
    return WindowMatrix(plan, output, WINDOW_FEATURE_NAMES, active, tuple(reasons))


def matrix_diagnostics(matrix: WindowMatrix) -> dict:
    """Label-free availability/rank diagnostics, not an accuracy verdict."""
    x = matrix.fit_values
    n, p = x.shape
    result = {
        "fit_scope": "one_answer", "width": matrix.plan.width, "stride": matrix.plan.stride,
        "n_fit_windows": n, "n_scoring_windows": len(matrix.values),
        "nonoverlapping_capacity_not_effective_n": matrix.plan.nonoverlapping_capacity,
        "declared_p": len(matrix.feature_names), "active_p": p,
        "inactive": {name: reason for name, reason in zip(matrix.feature_names, matrix.inactive_reasons) if reason},
        "status": "DESCRIPTIVE_ONLY", "labels_accessed": False,
    }
    if n < 2 or p == 0:
        return {**result, "status": "INSUFFICIENT_VARIATION", "rank": 0, "participation_rank": 0.0}
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    # Large offsets and small variation can leave roundoff in the first
    # centering pass. Remove it after scaling; centered N rows have rank <= N-1.
    z -= z.mean(axis=0)
    singular = np.linalg.svd(z, compute_uv=False)
    eigen = singular * singular / n
    participation = float(eigen.sum() ** 2 / np.square(eigen).sum())
    result.update(rank=min(n - 1, int(np.linalg.matrix_rank(z))), participation_rank=participation,
                  rank_cap=min(p, n - 1), raw_n_over_p=float(n / p))
    return result


def windows_to_tokens(plan: WindowPlan, scores: Sequence[float]) -> np.ndarray:
    """Mean of covering window scores; every token must have measured support."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != plan.starts.shape or not np.isfinite(scores).all():
        raise ValueError("one finite score is required for every scoring window")
    total = np.zeros(plan.token_count + 1)
    count = np.zeros(plan.token_count + 1, dtype=np.int64)
    np.add.at(total, plan.starts, scores)
    np.add.at(total, plan.ends, -scores)
    np.add.at(count, plan.starts, 1)
    np.add.at(count, plan.ends, -1)
    total, count = np.cumsum(total)[:-1], np.cumsum(count)[:-1]
    if np.any(count == 0):
        raise ValueError("UNSCORED_TOKEN_GAP")
    return total / count


def tokens_to_official_steps(
    token_scores: Sequence[float], starts: Sequence[int], ends: Sequence[int],
) -> np.ndarray:
    """Overlap-mean map back to supplied official spans; no invented boundaries."""
    scores = np.asarray(token_scores, dtype=float)
    starts, ends = np.asarray(starts), np.asarray(ends)
    if scores.ndim != 1 or not np.isfinite(scores).all() or starts.shape != ends.shape:
        raise ValueError("malformed score/span inputs")
    if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("span coordinates must be integers")
    if np.any(starts < 0) or np.any(ends > len(scores)) or np.any(ends <= starts):
        raise ValueError("invalid official step span")
    cumulative = np.concatenate(([0.0], np.cumsum(scores)))
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)
