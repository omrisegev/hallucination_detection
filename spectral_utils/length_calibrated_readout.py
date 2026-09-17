"""Length-calibrated Top-k step readout (Step 420).

The bank's step readout is the mean of the k largest token values in the step. Its expectation grows
with the step's token count even when every token has the same distribution (order statistics), so
the readout carries a hidden step-length prior.

Calibration, per answer and per stream, label-free and using only the answer's own tokens:
for a step with m valid tokens, the null is the Top-k mean of a CONTIGUOUS window of m valid tokens
taken from the same answer. Windows keep the token autocorrelation of the answer, which an i.i.d.
resample would destroy. At most MAX_WINDOWS start positions, evenly spaced and deterministic.

    z = (observed Top-k mean - mean over windows) / max(sd over windows, floor)
    floor = FLOOR_FRACTION * sd(valid tokens of the answer) / sqrt(min(k, m))

A single-step answer, or a stream constant over the answer, gives z = 0. Invalid tokens (the first
token of a prefix-innovation stream) are excluded from both the step and the windows, exactly as in
digitfree_broad50.step_bank.
"""
from __future__ import annotations

import numpy as np

K = 10
MAX_WINDOWS = 64
FLOOR_FRACTION = 0.5


def _window_topk_means(values: np.ndarray, m: int, k: int, max_windows: int) -> np.ndarray:
    L = len(values)
    starts = np.unique(np.round(np.linspace(0, L - m, min(max_windows, L - m + 1))).astype(int))
    windows = values[starts[:, None] + np.arange(m)[None, :]]
    kk = min(k, m)
    return np.partition(windows, m - kk, axis=1)[:, -kk:].mean(axis=1)


def step_topk_and_calibrated(x, valid, spans, k=K, max_windows=MAX_WINDOWS, floor_fraction=FLOOR_FRACTION):
    """Return (topk [S], calibrated_z [S], valid_count [S]) for one token stream of one answer."""
    x = np.asarray(x, float); valid = np.asarray(valid, bool); spans = np.asarray(spans, int)
    if x.shape != valid.shape or x.ndim != 1:
        raise ValueError("stream and mask must be aligned 1-d arrays")
    seq = x[valid]
    # map each step to its valid-token count
    counts = np.array([int(valid[a:b].sum()) for a, b in spans])
    topk = np.full(len(spans), np.nan); z = np.zeros(len(spans))
    for i, (a, b) in enumerate(spans):
        v = x[a:b][valid[a:b]]
        if len(v):
            kk = min(k, len(v)); topk[i] = np.partition(v, len(v) - kk)[-kk:].mean()
    token_sd = float(seq.std()) if len(seq) else 0.0
    if len(seq) < 2 or token_sd <= 1e-12 or len(spans) < 2:
        return topk, z, counts
    null = {}
    for m in np.unique(counts[counts > 0]):
        means = _window_topk_means(seq, int(m), k, max_windows)
        null[int(m)] = (float(means.mean()), float(means.std()))
    for i, m in enumerate(counts):
        if m == 0:
            continue
        mu, sd = null[int(m)]
        floor = floor_fraction * token_sd / np.sqrt(min(k, int(m)))
        z[i] = (topk[i] - mu) / max(sd, floor)
    return topk, z, counts


def self_test(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(400):
        lengths = rng.integers(3, 120, size=rng.integers(4, 14))
        T = int(lengths.sum()); ends = np.cumsum(lengths); spans = np.column_stack([ends - lengths, ends])
        # AR(1) tokens, identical distribution in every step (no signal)
        e = rng.standard_normal(T); x = np.empty(T); x[0] = e[0]
        for t in range(1, T):
            x[t] = .6 * x[t - 1] + e[t]
        topk, z, counts = step_topk_and_calibrated(x, np.ones(T, bool), spans)
        rows.append(np.column_stack([np.log(counts), topk, z]))
    R = np.vstack(rows)
    corr_top = float(np.corrcoef(R[:, 0], R[:, 1])[0, 1]); corr_z = float(np.corrcoef(R[:, 0], R[:, 2])[0, 1])
    assert corr_top > .4, corr_top          # the plain readout carries length
    assert abs(corr_z) < .06, corr_z        # the calibrated readout does not
    assert abs(R[:, 2].mean()) < .1, R[:, 2].mean()
    # a planted local spike in one short step must still be detected
    hits = 0
    for _ in range(200):
        lengths = rng.integers(8, 120, size=8); T = int(lengths.sum()); ends = np.cumsum(lengths)
        spans = np.column_stack([ends - lengths, ends]); x = rng.standard_normal(T)
        target = int(np.argmin(lengths)); a, b = spans[target]; x[a:a + 3] += 4.0
        _, z, _ = step_topk_and_calibrated(x, np.ones(T, bool), spans)
        hits += int(np.argmax(z) == target)
    assert hits >= 170, hits
    # invalid first token is excluded exactly as in step_bank
    x = np.arange(20, dtype=float); valid = np.ones(20, bool); valid[0] = False
    topk, _, counts = step_topk_and_calibrated(x, valid, np.array([[0, 5], [5, 20]]))
    assert counts[0] == 4 and np.isclose(topk[0], np.mean([1, 2, 3, 4]))
    return {"corr_log_length_topk": corr_top, "corr_log_length_calibrated": corr_z,
            "calibrated_mean": float(R[:, 2].mean()), "planted_spike_in_shortest_step_found": hits / 200}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=1))
