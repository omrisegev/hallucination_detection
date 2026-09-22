"""Frozen-bank readouts and prefix innovations; these APIs accept no labels."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from .direct_probability_fusion import step_top_mean
from .renyi_locator_feature_bank import feature_matrix

FEATURES = ("H0lim", "VE0", "VE075", "VE1")
SUBSETS = {"mean__" + "_".join(FEATURES[j] for j in subset): subset
           for size in range(1, 5) for subset in combinations(range(4), size)}
BASELINE = "mean__H0lim_VE0_VE075_VE1"


def prefix_innovation(values):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or not np.isfinite(values).all():
        raise ValueError("expected a finite token vector or matrix")
    residual = np.zeros_like(values)
    available = np.arange(len(values)) > 0
    if len(values) > 1:
        denominator = np.arange(1, len(values))
        if values.ndim == 2:
            denominator = denominator[:, None]
        residual[1:] = values[1:] - np.cumsum(values[:-1], axis=0) / denominator
    return residual, available


def score_features(logprobs, entropy, spans):
    features = feature_matrix(logprobs, entropy)
    matrix = features["matrix"][:, :4]
    spans = np.asarray(spans, dtype=int)
    if spans.ndim != 2 or spans.shape[1] != 2 or np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > len(matrix)) or np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError("invalid step spans")
    read = lambda x: step_top_mean(x, spans[:, 0], spans[:, 1], 10)
    per_view = np.column_stack([read(matrix[:, j]) for j in range(4)])
    scores = {name: per_view[:, subset].mean(axis=1) for name, subset in SUBSETS.items()}
    scores["entropy15"] = read(entropy)
    scores["mean_token_before_top10"] = read(matrix.mean(axis=1))
    for j in (0, 2):
        innovation, available = prefix_innovation(matrix[:, j])
        scores["innovation__" + FEATURES[j]] = read(innovation)
        scores["append_innovation__" + FEATURES[j]] = (per_view.sum(axis=1) + read(innovation)) / 5
    lengths = spans[:, 1] - spans[:, 0]
    base = scores[BASELINE]
    near = np.flatnonzero(base >= base.max() - .25 * base.std())[0]
    early = min(int(np.argmax(per_view[:, 1])), int(np.argmax(per_view[:, 2])))
    peaks = {"first_step": 0, "longest_step": int(np.argmax(lengths)),
             "first_near_max_025": int(near), "earlier_VE0_VE075_peak": early}
    diagnostics = dict(signs=features["signs"][:4], feature_std=matrix.std(axis=0),
                       n_tokens=len(matrix), step_lengths=lengths,
                       innovation_history_available=available)
    # Top10 overlap concerns within-stream selection, not fusion peak selection.
    intersections, unions = np.zeros((4, 4)), np.zeros((4, 4))
    for start, end in spans:
        sets = [set(np.argsort(matrix[start:end, j], kind="stable")[-min(10, end-start):]) for j in range(4)]
        for a in range(4):
            for b in range(4):
                intersections[a, b] += len(sets[a] & sets[b])
                unions[a, b] += len(sets[a] | sets[b])
    diagnostics["top10_intersection"] = intersections
    diagnostics["top10_union"] = unions
    return scores, peaks, diagnostics, matrix
