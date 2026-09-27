"""Stage B2 of expectation_realization_v1 (level reduction).  Frozen protocol: results/expectation_realization_v1/
PROTOCOL_STAGE_B2.json.  Helpers: merged partition, class-conditional dependence, and vectorized within-answer AUC for
the label-permutation nulls (ranks are fixed under a label permutation, so AUC = (sum of positive ranks - n1(n1+1)/2)/(n1 n0))."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def merge_groups_containing(groups, members: list[str], targets: set[str]) -> np.ndarray:
    """Relabel a partition so that every group containing any of `targets` becomes one group (labels 0..G-1)."""
    g = np.asarray(groups, int).copy(); hit = sorted({int(g[j]) for j, c in enumerate(members) if c in targets})
    if len(hit) > 1:
        for h in hit[1:]:
            g[g == h] = hit[0]
    return np.unique(g, return_inverse=True)[1]


def class_conditional_corr(X: np.ndarray, y: np.ndarray) -> dict:
    """Correlation matrices of the columns of X on y == 0 and on y == 1, with off-diagonal summaries."""
    y = np.asarray(y, bool); out = {}
    for name, sel in (('clean', ~y), ('error', y)):
        C = np.corrcoef(np.asarray(X, float)[sel], rowvar=False); off = ~np.eye(len(C), dtype=bool)
        out[name] = {'matrix': C, 'max_abs_offdiag': float(np.nanmax(np.abs(C[off]))), 'mean_abs_offdiag': float(np.nanmean(np.abs(C[off])))}
    return out


def within_ranks(S: np.ndarray, off: np.ndarray, answers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average-tie ranks of every column within each listed answer; returns (ranks over the concatenated steps of
    `answers`, local offsets)."""
    blocks = [rankdata(S[off[i]:off[i + 1]], axis=0) for i in answers]
    loc = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    return np.vstack(blocks), loc


def auc_from_ranks(Rk: np.ndarray, loc: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-answer within-AUC for every column given fixed ranks and a label vector over the concatenated steps.
    Answers without both classes give NaN."""
    y = np.asarray(y, float); n = np.diff(loc); n1 = np.add.reduceat(y, loc[:-1]); n0 = n - n1
    s = np.add.reduceat(Rk * y[:, None], loc[:-1], axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (s - (n1 * (n1 + 1) / 2)[:, None]) / (n1 * n0)[:, None]


def shuffle_within(y: np.ndarray, loc: np.ndarray, rng) -> np.ndarray:
    out = np.empty_like(y)
    for a, b in zip(loc[:-1], loc[1:]):
        out[a:b] = y[a:b][rng.permutation(b - a)]
    return out


def swap_same_length(y: np.ndarray, loc: np.ndarray, rng) -> np.ndarray:
    """Each answer receives the label vector of another answer with the same number of steps (a random derangement
    within each length stratum; singleton strata keep their own labels)."""
    n = np.diff(loc); out = y.copy()
    for L in np.unique(n):
        idx = np.flatnonzero(n == L)
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        while np.any(perm == np.arange(len(idx))):                              # derangement: nobody keeps its own labels
            fix = np.flatnonzero(perm == np.arange(len(idx))); j = rng.integers(len(idx)); perm[[fix[0], j]] = perm[[j, fix[0]]]
        for a, b in zip(idx, idx[perm]):
            out[loc[a]:loc[a + 1]] = y[loc[b]:loc[b + 1]]
    return out
