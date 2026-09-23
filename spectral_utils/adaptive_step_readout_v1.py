"""Per-answer adaptive step readout, chosen label-free by a signal-to-noise criterion.

Derived from what the readout grid established rather than from a swept parameter:

  * evidence of an error sits in an ABSOLUTE number of tokens, not a fraction of the
    step -- every fixed-quantile readout lost heavily to every fixed-count one, so the
    ladder here is over counts;
  * a small K is dominated by sampling noise and a large K dilutes the signal towards
    the plain step mean, so the useful width is an interior optimum;
  * the readout only ever has to be comparable WITHIN an answer, because the locator
    takes an argmax inside the answer. Choosing a different K per answer therefore costs
    nothing, which is what makes per-answer adaptation admissible at all. (A gate that
    compares answers to each other would not have that freedom.)

The criterion is a variance decomposition, computed from the answer's own tokens with
no labels:

    total(K)  = variance of the step scores across the steps of this answer
    noise(K)  = sampling variance of the step statistic, from repeated half splits
    signal(K) = max(total - noise, 0)
    SNR(K)    = signal / noise

and K* = argmax SNR. Small K inflates the denominator; large K collapses the numerator
as every step converges on its own mean. The interior optimum is a property of the
decomposition rather than of where a grid happened to stop.

All of the top-K means for every K are obtained from one sort per block: sorting
descending and taking the running mean gives the whole ladder at once.
"""

from __future__ import annotations

import numpy as np

MAX_K = 64
N_SPLITS = 5
EPS = 1e-9


def ladder_for(max_len: int, cap: int = MAX_K) -> np.ndarray:
    """Integer K ladder, dense where the optimum lives and sparse out in the tail."""
    top = max(1, min(cap, int(max_len)))
    if top <= 24:
        return np.arange(1, top + 1)
    dense = np.arange(1, 25)
    sparse = np.unique(np.round(np.geomspace(25, top, 10)).astype(int))
    return np.unique(np.concatenate([dense, sparse]))


def _topk_means(block: np.ndarray, ladder: np.ndarray) -> np.ndarray:
    """Top-K mean of `block` for every K in `ladder`, K clipped to len(block)."""
    n = len(block)
    if n == 0:
        return np.zeros(len(ladder))
    order = np.sort(block)[::-1]
    running = np.cumsum(order) / np.arange(1, n + 1)
    return running[np.minimum(ladder, n) - 1]


def step_scores_over_ladder(series: np.ndarray, spans: np.ndarray,
                            ladder: np.ndarray) -> np.ndarray:
    """[steps, len(ladder)] step scores for one fused token series."""
    out = np.zeros((len(spans), len(ladder)))
    for s, (a, b) in enumerate(spans):
        out[s] = _topk_means(series[a:b], ladder)
    return out


def choose_k(series: np.ndarray, spans: np.ndarray, ladder: np.ndarray,
             rng: np.random.Generator, n_splits: int = N_SPLITS) -> tuple[int, np.ndarray]:
    """Return (index of K* in `ladder`, the SNR curve over the ladder)."""
    full = step_scores_over_ladder(series, spans, ladder)
    total = full.var(axis=0)

    noise = np.zeros(len(ladder))
    for _ in range(n_splits):
        a = np.zeros_like(full)
        b = np.zeros_like(full)
        for s, (lo, hi) in enumerate(spans):
            block = series[lo:hi]
            n = len(block)
            if n < 2:
                a[s] = b[s] = _topk_means(block, ladder)
                continue
            perm = rng.permutation(n)
            half = n // 2
            a[s] = _topk_means(block[perm[:half]], ladder)
            b[s] = _topk_means(block[perm[half:]], ladder)
        # var of a half-sample estimator is ~2x that of the full-sample one, and
        # var(a-b) = 2 * var(half), so var(full) ~ var(a-b)/4
        noise += (a - b).var(axis=0) / 4.0
    noise /= n_splits

    signal = np.maximum(total - noise, 0.0)
    snr = signal / (noise + EPS)
    return int(np.argmax(snr)), snr


def adaptive_step_scores(series: np.ndarray, spans: np.ndarray,
                         rng: np.random.Generator, cap: int = MAX_K,
                         n_splits: int = N_SPLITS) -> tuple[np.ndarray, int]:
    """Step scores at the answer's own K*, and that K*."""
    spans = np.asarray(spans, int)
    lengths = spans[:, 1] - spans[:, 0]
    ladder = ladder_for(int(lengths.max()) if len(lengths) else 1, cap)
    if len(spans) < 2:
        return step_scores_over_ladder(series, spans, ladder)[:, 0], int(ladder[0])
    index, _ = choose_k(series, spans, ladder, rng, n_splits)
    return step_scores_over_ladder(series, spans, ladder)[:, index], int(ladder[index])
