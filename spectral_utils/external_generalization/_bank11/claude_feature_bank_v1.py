"""Token-local feature bank for the Claude handoff experiment.

This module is deliberately small and explicit.  It implements the candidate
bank proposed in the ``סקור ענפים ומזג תוצאות`` handoff at the token grid:

* one teacher-forced answer is the unit of extraction;
* causal windows may be used to calculate a value at token ``t``;
* every feature is manually oriented to ``higher = more risk`` before fusion;
* there is no learned feature anchor and no ``digit`` channel;
* L-SML and equal-weight fusion receive the same standardized token matrix;
* LOCO-5 is an answer-level, fixed experimental gate and is kept outside the
  locator fusion.

The sign map is a protocol choice, not a label-fitted calibration.  The
``global_sign_gauge`` used after L-SML only resolves the mathematical global
sign ambiguity of the returned coefficient vector; it never looks at labels
or at an answer-level anchor feature.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.stats import rankdata

from .feature_utils import compute_cusum_residuals, compute_spectral_features
from .fusion_utils import lsml_continuous
from .temporal_models import bocpd_gaussian
from .token_feature_views import token_feature_views


WINDOW = 16
BOCPD_HAZARD_LAMBDA = 32.0
TOP_K_REQUIRED = 50

# All values returned by build_token_feature_matrix are multiplied by these
# signs.  After multiplication, a larger value always means "more suspect".
FEATURE_NAMES = (
    "q15_H1",
    "q15_VE1",
    "chosen_surprisal",
    "logprob_margin",
    "true_tail50",
    "energy_level",
    "energy_innovation",
    "top15_turnover",
    "top50_js",
    "dominant_freq16",
    "bocpd_p0",
)

FEATURE_RISK_SIGNS = {
    "q15_H1": +1,
    "q15_VE1": +1,
    "chosen_surprisal": +1,
    "logprob_margin": -1,
    "true_tail50": +1,
    "energy_level": -1,
    "energy_innovation": +1,
    "top15_turnover": +1,
    "top50_js": +1,
    # This is the predeclared exploratory direction from the project feature
    # contract.  It is not re-oriented from the benchmark labels.
    "dominant_freq16": -1,
    "bocpd_p0": +1,
}

# LOCO-5 is a historically label-selected subset.  Keeping it fixed here is
# useful for the requested experiment, but all results using it are
# development-only and cannot be called label-free validation.
LOCO5_NAMES = (
    "cusum_max",
    "logprob_margin",
    "min_energy",
    "spectral_entropy",
    "topk_tail_mass",
)
LOCO5_RISK_SIGNS = {
    "cusum_max": +1,
    "logprob_margin": -1,
    "min_energy": -1,
    "spectral_entropy": +1,
    "topk_tail_mass": +1,
}


def _series(row: dict, key: str, n: int | None = None) -> np.ndarray:
    value = row.get(key)
    if value is None:
        raise ValueError(f"row is missing required series {key!r}")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if n is not None and arr.shape != (n,):
        raise ValueError(f"{key} has shape {arr.shape}, expected {(n,)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{key} contains non-finite values")
    return arr


def _topk(row: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    payload = row.get("top_k_logprobs") or row.get("top_k_logprobs_raw")
    if not isinstance(payload, dict):
        raise ValueError("row has no saved top-k log-probability payload")
    logprobs = np.asarray(payload.get("logprobs"), dtype=float)
    ids = np.asarray(payload.get("ids"))
    if logprobs.ndim != 2 or logprobs.shape[0] != n or logprobs.shape[1] < TOP_K_REQUIRED:
        raise ValueError(
            "top-k logprobs must have shape [tokens, K] with K >= 50; "
            f"got {logprobs.shape}"
        )
    if ids.shape != logprobs.shape:
        raise ValueError(f"top-k ids shape {ids.shape} does not match {logprobs.shape}")
    if not np.isfinite(logprobs).all():
        raise ValueError("top-k logprobs contain non-finite values")
    return ids.astype(np.int64, copy=False), logprobs


def _q15_views(logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return H1 and VE1 on the renormalized top-15 distribution."""
    p = np.exp(logprobs[:, :15])
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    surprisal = -np.log(np.maximum(p, 1e-12))
    mean = np.sum(p * surprisal, axis=1, keepdims=True)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
    varentropy = np.sum(p * (surprisal - mean) ** 2, axis=1)
    return entropy, varentropy


def _prefix_innovation(values: np.ndarray) -> np.ndarray:
    """Causal deviation from the mean of the strictly earlier prefix."""
    out = np.zeros(len(values), dtype=float)
    if len(values) > 1:
        prefix_sum = np.cumsum(values[:-1], dtype=float)
        out[1:] = values[1:] - prefix_sum / np.arange(1, len(values), dtype=float)
    return out


def _adjacent_views(ids: np.ndarray, logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identity-aligned top-15 turnover and top-50 Jensen-Shannon distance."""
    n = len(ids)
    turnover = np.zeros(n, dtype=float)
    js = np.zeros(n, dtype=float)
    probabilities = np.exp(logprobs)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    for t in range(1, n):
        previous_ids = ids[t - 1]
        current_ids = ids[t]
        previous_top15 = set(previous_ids[:15].tolist())
        turnover[t] = 1.0 - len(previous_top15.intersection(current_ids[:15].tolist())) / 15.0

        previous = {int(token): float(probabilities[t - 1, j]) for j, token in enumerate(previous_ids)}
        current = {int(token): float(probabilities[t, j]) for j, token in enumerate(current_ids)}
        union = set(previous).union(current)
        p = np.fromiter((previous.get(token, 0.0) for token in union), dtype=float)
        q = np.fromiter((current.get(token, 0.0) for token in union), dtype=float)
        midpoint = (p + q) / 2.0
        p_mask = p > 0
        q_mask = q > 0
        js[t] = 0.5 * np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
        js[t] += 0.5 * np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    return turnover, js


def _bocpd_curve(entropy: np.ndarray) -> np.ndarray:
    """Map BOCPD's causal p0[t-1] to token t; first token is inactive/zero."""
    if len(entropy) == 0:
        return np.zeros(0, dtype=float)
    result = np.zeros(len(entropy), dtype=float)
    if len(entropy) > 1:
        fitted = bocpd_gaussian(entropy, hazard_lambda=BOCPD_HAZARD_LAMBDA)
        p0 = np.asarray(fitted["p0"], dtype=float).reshape(-1)
        if len(p0) == len(entropy):
            result[1:] = p0[:-1]
        elif len(p0) == len(entropy) - 1:
            result[1:] = p0
        else:
            raise ValueError(f"BOCPD returned unexpected shape {p0.shape}")
    return result


def build_token_feature_matrix(row: dict) -> np.ndarray:
    """Build the manually risk-oriented ``[tokens, 11]`` feature matrix.

    Raw definitions, all causal where a prefix/window is involved:

    ``q15_H1``/``q15_VE1`` are entropy and variance of surprisal over the
    renormalized saved top-15 distribution; ``chosen_surprisal`` is the saved
    ``-log p(sampled token)``; ``logprob_margin`` is top-1 minus top-2 logprob;
    ``true_tail50`` is one minus the probability mass of the saved top-50
    (not a Z-subtracted quantity); ``energy_level`` is raw full-vocabulary
    logsumexp; ``energy_innovation`` is its deviation from the earlier prefix
    mean; the next two are identity-aligned adjacent-view dynamics;
    ``dominant_freq16`` is the causal 16-token rolling spectral frequency;
    ``bocpd_p0`` is causal Gaussian BOCPD with hazard lambda 32.
    """
    entropy = _series(row, "token_entropies")
    n = len(entropy)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    spilled = _series(row, "token_spilled_energies", n)
    energy = _series(row, "token_logsumexp", n)
    ids, logprobs = _topk(row, n)
    h1, ve1 = _q15_views(logprobs)
    turnover, js = _adjacent_views(ids, logprobs)
    margin = logprobs[:, 0] - logprobs[:, 1]
    true_tail = np.clip(1.0 - np.exp(logprobs[:, :TOP_K_REQUIRED]).sum(axis=1), 0.0, 1.0)
    rolling = token_feature_views({"token_entropies": entropy})
    dominant = np.asarray(rolling["entropy_rolling_dominant_freq"], dtype=float)
    if dominant.shape != (n,):
        raise ValueError("dominant-frequency token view is misaligned")
    raw = np.column_stack((
        h1,
        ve1,
        spilled,
        margin,
        true_tail,
        energy,
        _prefix_innovation(energy),
        turnover,
        js,
        dominant,
        _bocpd_curve(entropy),
    ))
    oriented = raw * np.asarray([FEATURE_RISK_SIGNS[name] for name in FEATURE_NAMES])
    if not np.isfinite(oriented).all():
        raise ValueError("feature bank contains non-finite values")
    return oriented.astype(np.float32)


def build_loco5_answer_features(row: dict) -> dict[str, float]:
    """Build the five fixed answer-level LOCO-5 gate features.

    This is intentionally separate from the token locator matrix.  The gate
    sees the whole response, while the locator later aggregates token risks
    inside each ProcessBench step.
    """
    entropy = _series(row, "token_entropies")
    energy = _series(row, "token_logsumexp", len(entropy))
    _ids, logprobs = _topk(row, len(entropy))
    p = np.exp(logprobs)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    tail_mass = np.clip(1.0 - p[:, :5].sum(axis=1), 0.0, 1.0)
    spectral = compute_spectral_features(entropy)
    spectral_entropy = float(spectral["spectral_entropy"]) if spectral else 0.0
    return {
        "cusum_max": float(compute_cusum_residuals(entropy)["cusum_max"]),
        "logprob_margin": float(np.mean(logprobs[:, 0] - logprobs[:, 1])),
        "min_energy": float(np.min(energy)),
        "spectral_entropy": spectral_entropy,
        "topk_tail_mass": float(np.mean(tail_mass)),
    }


def _midrank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=float)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def loco5_gate(
    answer_features: Sequence[dict[str, float]],
    cells: Sequence[str],
    *,
    threshold: float = 0.33,
    pb_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return ``(composite, opened, diagnostics)`` for the fixed LOCO-5 gate.

    Each raw feature is manually risk-oriented and converted to a within-cell
    midrank.  The gate score is the equal average of the five midranks and is
    opened at the predeclared 0.33 quantile.  No labels enter this function.
    """
    cells = np.asarray(cells, dtype=str)
    n = len(answer_features)
    if cells.shape != (n,):
        raise ValueError("answer features and cells are not aligned")
    if pb_mask is None:
        pb_mask = np.char.startswith(cells, "pb_")
    pb_mask = np.asarray(pb_mask, dtype=bool)
    if pb_mask.shape != (n,):
        raise ValueError("pb_mask is not answer-aligned")
    raw = np.full((n, len(LOCO5_NAMES)), np.nan, dtype=float)
    for i, item in enumerate(answer_features):
        raw[i] = [float(item[name]) * LOCO5_RISK_SIGNS[name] for name in LOCO5_NAMES]
    composite = np.full(n, np.nan, dtype=float)
    for cell in np.unique(cells[pb_mask]):
        indexes = np.flatnonzero(pb_mask & (cells == cell))
        ranked = np.column_stack([_midrank01(raw[indexes, j]) for j in range(raw.shape[1])])
        composite[indexes] = ranked.mean(axis=1)
    opened = np.zeros(n, dtype=bool)
    opened[pb_mask] = composite[pb_mask] >= float(threshold)
    diagnostics = {
        "name": "LOCO-5",
        "members": list(LOCO5_NAMES),
        "threshold": float(threshold),
        "risk_signs": dict(LOCO5_RISK_SIGNS),
        "labels_used": False,
        "development_only": True,
        "rank_scope": "within_pb_cell",
        "opened": int(opened[pb_mask].sum()),
        "eligible": int(pb_mask.sum()),
    }
    return composite, opened, diagnostics


def deterministic_token_sample(matrices: Iterable[np.ndarray], cap: int = 60_000) -> np.ndarray:
    """Take a deterministic, answer-stratified-by-length token sample."""
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    arrays = [matrix for matrix in arrays if matrix.ndim == 2 and len(matrix)]
    if not arrays:
        raise ValueError("no token matrices available for fit")
    total = sum(len(matrix) for matrix in arrays)
    pieces = []
    for matrix in arrays:
        count = min(len(matrix), max(1, int(round(cap * len(matrix) / total))))
        indexes = np.linspace(0, len(matrix) - 1, count, dtype=int)
        pieces.append(matrix[indexes])
    sample = np.concatenate(pieces, axis=0)
    if len(sample) > cap:
        sample = sample[np.linspace(0, len(sample) - 1, cap, dtype=int)]
    return sample


def fit_token_standardizer(matrices: Iterable[np.ndarray], cap: int = 60_000) -> dict[str, np.ndarray | int]:
    """Fit donor-only pooled means/stds for the token matrix."""
    sample = deterministic_token_sample(matrices, cap=cap)
    if not np.isfinite(sample).all():
        raise ValueError("non-finite token in standardizer fit")
    mean = sample.mean(axis=0)
    std = sample.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return {"mean": mean, "std": std, "sample_count": int(len(sample))}


def _flatten_l_sml_weights(meta: dict, feature_count: int) -> np.ndarray:
    weights = np.zeros(feature_count, dtype=float)
    cross = np.asarray(meta["cross_weights"], dtype=float)
    for group_index, (indexes, within) in enumerate(meta["group_weights"]):
        weights[np.asarray(indexes, dtype=int)] += cross[group_index] * np.asarray(within, dtype=float)
    if not np.isfinite(weights).all() or np.all(np.abs(weights) < 1e-12):
        raise ValueError("L-SML returned a degenerate coefficient vector")
    # Deterministic gauge only.  This is not an answer/feature anchor and does
    # not use labels: manually risk-oriented columns should have nonnegative
    # aggregate contribution under the risk-score convention.
    total = float(weights.sum())
    if abs(total) > 1e-12:
        weights *= 1.0 if total > 0 else -1.0
    else:
        first = int(np.flatnonzero(np.abs(weights) > 1e-12)[0])
        if weights[first] < 0:
            weights *= -1.0
    return weights


def fit_l_sml_weights(
    matrices: Iterable[np.ndarray],
    standardizer: dict[str, np.ndarray | int],
    *,
    method: str = "residual",
) -> tuple[np.ndarray, dict]:
    """Fit token-level continuous L-SML without labels or an anchor feature."""
    sample = deterministic_token_sample(matrices, cap=int(standardizer["sample_count"]))
    mean = np.asarray(standardizer["mean"], dtype=float)
    std = np.asarray(standardizer["std"], dtype=float)
    standardized = (sample - mean) / std
    _fused, meta = lsml_continuous(*standardized.T, method=method)
    weights = _flatten_l_sml_weights(meta, standardized.shape[1])
    return weights, meta


def standardized_risk_matrix(matrix: np.ndarray, standardizer: dict[str, np.ndarray | int]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mean = np.asarray(standardizer["mean"], dtype=float)
    std = np.asarray(standardizer["std"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(mean):
        raise ValueError("token matrix does not match standardizer")
    result = (matrix - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("non-finite standardized token matrix")
    return result


def fuse_token_matrix(
    matrix: np.ndarray,
    standardizer: dict[str, np.ndarray | int],
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(l_sml_or_equal, equal)`` token risks for one answer."""
    standardized = standardized_risk_matrix(matrix, standardizer)
    equal = standardized.mean(axis=1)
    if weights is None:
        weights = np.ones(standardized.shape[1], dtype=float) / standardized.shape[1]
    l_sml = standardized @ np.asarray(weights, dtype=float)
    return l_sml, equal
