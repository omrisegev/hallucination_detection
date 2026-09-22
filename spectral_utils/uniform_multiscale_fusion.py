"""Benchmark-uniform q15/q50 Renyi-varentropy feature banks and weights.

Feature construction and global moment fitting are label-free.  The only
label-accepting API is :func:`fit_simplex`, which fits one constrained vector
across both benchmark panels from an already exclusion-safe answer roster.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .probability_normalization_ablation import step_readout
from .renyi_alpha_sweep import anchor_stream, escort_varentropy
from .renyi_position_fusion import feature_bank as frozen_q15_bank
from .renyi_view_fusion import EPS, head_distribution


BASE_NAMES = ("H0lim", "ve0", "ve0.75", "ve1")
FEATURE_NAMES = tuple(f"q{k}__{name}" for k in (15, 50) for name in BASE_NAMES)
SUPPORTS = {
    "q15": np.arange(0, 4, dtype=np.int64),
    "q50": np.arange(4, 8, dtype=np.int64),
    "ms8": np.arange(0, 8, dtype=np.int64),
}
WEIGHT_EPSILON = 0.02
RIDGE = 1e-3


@dataclass(frozen=True)
class GlobalScale:
    mean: np.ndarray
    scale: np.ndarray
    training_groups: tuple[str, ...]
    training_answers: int
    panels: tuple[str, ...]
    cells: tuple[str, ...]


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(left, right)[0, 1])


def _support_views(logprobs, k: int, anchor: np.ndarray) -> tuple[np.ndarray, dict]:
    q, _ = head_distribution(logprobs, k=k)
    raw = np.column_stack(
        (
            np.log(q + EPS).mean(axis=1),
            escort_varentropy(q, 0.0),
            escort_varentropy(q, 0.75),
            escort_varentropy(q, 1.0),
        )
    )
    correlations = np.asarray([_corr(raw[:, j], anchor) for j in range(4)])
    signs = np.where(np.isfinite(correlations) & (correlations < 0), -1.0, 1.0)
    oriented = raw * signs
    if not np.isfinite(oriented).all():
        raise ValueError(f"q{k} bank contains nonfinite values")
    return oriented, {
        "correlations": correlations,
        "signs": signs,
        "flipped": signs < 0,
    }


def feature_bank(logprobs) -> dict:
    """Return one oriented T-by-8 q15/q50 bank and identity diagnostics."""
    anchor = np.asarray(anchor_stream(logprobs), dtype=np.float64)
    q15, info15 = _support_views(logprobs, 15, anchor)
    q50, info50 = _support_views(logprobs, 50, anchor)
    frozen = frozen_q15_bank(logprobs)
    expected = (
        np.asarray(frozen["raw"], dtype=np.float64)
        * np.asarray(frozen["single_signs"], dtype=np.float64)
        * np.asarray(frozen["fusion_signs"], dtype=np.float64)
    )
    identity = float(np.max(np.abs(q15 - expected)))
    bank = np.column_stack((q15, q50))
    return {
        "bank": bank,
        "anchor": anchor,
        "orientation": {"q15": info15, "q50": info50},
        "q15_frozen_identity_max": identity,
        "mean": bank.mean(axis=0),
        "second": np.mean(bank * bank, axis=0),
    }


def step_bank(token_bank: np.ndarray, spans: np.ndarray) -> np.ndarray:
    token_bank = np.asarray(token_bank, dtype=np.float64)
    if token_bank.ndim != 2 or token_bank.shape[1] != len(FEATURE_NAMES):
        raise ValueError("token bank must be T-by-8")
    output = np.column_stack([step_readout(token_bank[:, j], spans) for j in range(8)])
    if not np.isfinite(output).all():
        raise ValueError("nonfinite step bank")
    return output


def _panel(cell: str) -> str:
    return "pb" if str(cell).startswith("pb_") else "prm"


def _hierarchy(records: Sequence[Mapping], selected: Sequence[int]):
    nested: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index in selected:
        row = records[int(index)]
        nested[_panel(row["cell"])][str(row["cell"])][str(row["group_id"])].append(int(index))
    if set(nested) != {"pb", "prm"}:
        raise ValueError("training roster must contain both benchmark panels")
    return nested


def fit_global_scale(
    statistics: Mapping[int, Mapping[str, np.ndarray]],
    records: Sequence[Mapping],
    selected: Sequence[int],
) -> GlobalScale:
    """Equal panel/cell/group/answer moments; no benchmark-specific scale."""
    nested = _hierarchy(records, selected)
    panel_means, panel_seconds = [], []
    groups = []
    cells = []
    for panel in sorted(nested):
        cell_means, cell_seconds = [], []
        for cell in sorted(nested[panel]):
            group_means, group_seconds = [], []
            cells.append(cell)
            for group in sorted(nested[panel][cell]):
                members = nested[panel][cell][group]
                groups.append(group)
                group_means.append(np.mean([statistics[i]["mean"] for i in members], axis=0))
                group_seconds.append(np.mean([statistics[i]["second"] for i in members], axis=0))
            cell_means.append(np.mean(group_means, axis=0))
            cell_seconds.append(np.mean(group_seconds, axis=0))
        panel_means.append(np.mean(cell_means, axis=0))
        panel_seconds.append(np.mean(cell_seconds, axis=0))
    mean = np.mean(panel_means, axis=0)
    second = np.mean(panel_seconds, axis=0)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    if mean.shape != (8,) or scale.shape != (8,) or np.any(scale <= 1e-10):
        raise ValueError("invalid global eight-view scale")
    return GlobalScale(
        mean=np.asarray(mean),
        scale=np.asarray(scale),
        training_groups=tuple(sorted(set(groups))),
        training_answers=len(selected),
        panels=tuple(sorted(nested)),
        cells=tuple(sorted(set(cells))),
    )


def hierarchical_step_weights(
    records: Sequence[Mapping],
    selected: Sequence[int],
    labels: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Equal panel/cell/group/answer mass followed by global class balance."""
    nested = _hierarchy(records, selected)
    output = {}
    for panel, panel_cells in nested.items():
        panel_mass = 1.0 / len(nested)
        for _, cell_groups in panel_cells.items():
            cell_mass = panel_mass / len(panel_cells)
            for _, members in cell_groups.items():
                group_mass = cell_mass / len(cell_groups)
                for index in members:
                    y = np.asarray(labels[index], dtype=np.int64)
                    known = y >= 0
                    if not known.any():
                        raise ValueError("answer has no known training steps")
                    weights = np.zeros(len(y), dtype=np.float64)
                    weights[known] = group_mass / len(members) / int(known.sum())
                    output[index] = weights
    totals = np.zeros(2, dtype=np.float64)
    for index in selected:
        y = np.asarray(labels[index], dtype=np.int64)
        for klass in (0, 1):
            totals[klass] += output[index][y == klass].sum()
    if np.any(totals <= 0):
        raise ValueError("training roster must contain both known classes")
    for index in selected:
        y = np.asarray(labels[index], dtype=np.int64)
        for klass in (0, 1):
            output[index][y == klass] *= 0.5 / totals[klass]
    np.testing.assert_allclose(sum(w.sum() for w in output.values()), 1.0, atol=1e-12, rtol=0)
    return output


def fit_simplex(
    step_features: Mapping[int, np.ndarray],
    labels: Mapping[int, np.ndarray],
    records: Sequence[Mapping],
    selected: Sequence[int],
    columns: np.ndarray,
    scale: np.ndarray,
    *,
    epsilon: float = WEIGHT_EPSILON,
    ridge: float = RIDGE,
) -> tuple[np.ndarray, dict]:
    """Fit one nonnegative sum-one vector on exclusion-safe step labels."""
    columns = np.asarray(columns, dtype=np.int64)
    scale = np.asarray(scale, dtype=np.float64)
    sample_weights = hierarchical_step_weights(records, selected, labels)
    matrices, targets, weights = [], [], []
    for index in selected:
        y = np.asarray(labels[index], dtype=np.int64)
        known = y >= 0
        matrices.append(np.asarray(step_features[index], dtype=np.float64)[known][:, columns] / scale[columns])
        targets.append(y[known].astype(np.float64))
        weights.append(sample_weights[index][known])
    x = np.vstack(matrices)
    y = np.concatenate(targets)
    sw = np.concatenate(weights)
    dimension = len(columns)

    def objective(theta):
        coefficient, intercept = theta[:dimension], float(theta[-1])
        score = x @ coefficient + intercept
        loss = np.sum(sw * (np.logaddexp(0.0, score) - y * score))
        residual = sw * (expit(score) - y)
        gradient = np.r_[x.T @ residual + ridge * coefficient, residual.sum()]
        return float(loss + 0.5 * ridge * coefficient @ coefficient), gradient

    initial = np.r_[np.full(dimension, 1.0 / dimension), 0.0]
    result = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * dimension + [(None, None)],
        constraints={"type": "eq", "fun": lambda value: np.sum(value[:dimension]) - 1.0,
                     "jac": lambda value: np.r_[np.ones(dimension), 0.0]},
        options={"maxiter": 300, "ftol": 1e-11, "disp": False},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("simplex fit failed: " + str(result.message))
    raw = np.clip(np.asarray(result.x[:dimension], dtype=np.float64), 0.0, 1.0)
    raw /= raw.sum()
    sparse = raw.copy()
    sparse[sparse < float(epsilon)] = 0.0
    sparse /= sparse.sum()
    initial_loss = objective(initial)[0]
    final_loss = objective(np.r_[sparse, result.x[-1]])[0]
    return sparse, {
        "status": "FIT",
        "success": True,
        "message": str(result.message),
        "iterations": int(result.nit),
        "initial_loss": float(initial_loss),
        "optimized_loss": float(result.fun),
        "sparse_loss": float(final_loss),
        "intercept": float(result.x[-1]),
        "raw_weights": raw,
        "weights": sparse,
        "epsilon": float(epsilon),
        "ridge": float(ridge),
        "training_answers": len(selected),
        "training_steps": int(len(y)),
        "positive_weight": float(sw[y == 1].sum()),
        "negative_weight": float(sw[y == 0].sum()),
    }


def score(step_features: np.ndarray, columns: np.ndarray, scale: np.ndarray, weights: np.ndarray, *, centered=False):
    matrix = np.asarray(step_features, dtype=np.float64)[:, np.asarray(columns, dtype=np.int64)]
    matrix = matrix / np.asarray(scale, dtype=np.float64)[np.asarray(columns, dtype=np.int64)]
    if centered:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    value = matrix @ np.asarray(weights, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite uniform fusion score")
    return value


def natural_weights(columns: np.ndarray, scale: np.ndarray) -> np.ndarray:
    value = np.asarray(scale, dtype=np.float64)[np.asarray(columns, dtype=np.int64)].copy()
    return value / value.sum()


__all__ = [
    "BASE_NAMES", "FEATURE_NAMES", "GlobalScale", "RIDGE", "SUPPORTS", "WEIGHT_EPSILON",
    "feature_bank", "fit_global_scale", "fit_simplex", "hierarchical_step_weights",
    "natural_weights", "score", "step_bank",
]
