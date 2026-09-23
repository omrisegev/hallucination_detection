"""The step-measurement stage, isolated so it can be replaced without touching fusion.

The localizer is three stages: a token feature bank, a fusion that collapses the bank to
one token series, and a **step measurement** that collapses the tokens inside a step to
one number. This module is only the third. Fusion enters as a finished 1-D series and is
never re-opened here.

That stage is currently a Top-10 mean, and the evidence says it is the expensive one:
the width alone moves gate-free SLA from 20.42 at K=1 to 37.01 at K=20, a 16.6 pp range,
against a 3-to-8 pp range for every decision rule ever tried on top of it.

Two things are wrong with a Top-K mean as a detector, and this module exists to test
both:

**It assumes white noise.** The fused token series has lag-1 autocorrelation **0.500**,
decaying to zero only past lag 80, so a flat K-window keeps about `(1+r)/(1-r) = 3x` the
variance it would on independent samples. The measured `n_eff` per step is 22-29 in every
cell while the raw token count per step ranges 55-93 -- the extra tokens in a long step
are very nearly redundant. The textbook response to a known signal in coloured noise is
to prewhiten first and match afterwards, and nothing in the pipeline prewhitens.

**It throws away time.** A Top-K mean takes the K largest values anywhere in the step,
unordered. If the error is a contiguous burst rather than K scattered spikes, the
statistic matched to it is the best contiguous window, not the best unordered set.

So the family here is a 2x2 -- `{raw, whitened} x {unordered Top-K, best contiguous
window}` -- with the incumbent as its `raw x unordered` corner. The whitener is fitted
**per model** on all of that model's steps, which is the scope Omri asked for and which
was separately measured to be free for the fusion stage (-0.22 short / -0.65 long, both
intervals covering zero).

The sharp prediction, worth writing down before any of it is run: if the K=10-to-20
optimum exists *because* the noise is correlated, then after whitening the best width
should move DOWN toward 1 and the peak should move UP. If whitening leaves the optimum
where it is, correlated noise is not what the width is buying and this line is done.

Every function is answer-local at application time; only the AR coefficients come from
outside the answer, and they carry no labels.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

__all__ = [
    "answer_standardize",
    "accumulate_autocorrelation",
    "yule_walker",
    "whiten",
    "topk_mean_steps",
    "best_window_steps",
]


def answer_standardize(values: np.ndarray) -> np.ndarray:
    """Centre and scale a series by its own moments; constant series become zeros.

    Free to apply: the final decision is an argmax *within* the answer, so any answer-level
    affine transform leaves the prediction unchanged. It is done here so the pooled AR fit
    sees one common scale.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        raise ValueError("empty series")
    std = x.std()
    return (x - x.mean()) / std if std > 1e-12 else np.zeros_like(x)


def accumulate_autocorrelation(series, order: int, min_length: int = 0) -> np.ndarray:
    """Mean within-answer autocorrelation at lags 0..order over an iterable of series.

    Pooled by averaging each answer's own autocorrelation rather than by concatenating
    the answers, so that a long answer does not dominate and no lag ever spans a boundary
    between two different answers.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    total = np.zeros(order + 1)
    count = 0
    for values in series:
        x = answer_standardize(values)
        if len(x) < max(min_length, 2 * order + 2):
            continue
        row = np.array([1.0] + [float(np.mean(x[:-lag] * x[lag:])) for lag in range(1, order + 1)])
        total += row
        count += 1
    if not count:
        raise ValueError("no answer was long enough to estimate the autocorrelation")
    return total / count


def yule_walker(autocorrelation: np.ndarray) -> dict:
    """AR coefficients and innovation scale from an autocorrelation sequence.

    Solves the Toeplitz system `R a = r` for the one-step linear predictor, and also
    returns the **ladder** of every lower-order predictor from the same sequence. The
    ladder is what lets `whiten` treat the first few tokens of an answer correctly: at
    position t < order there are only t past samples, and the right predictor there is
    the order-t one, not the order-`order` one applied to invented history.
    """
    r = np.asarray(autocorrelation, dtype=float).reshape(-1)
    if len(r) < 2 or not np.isclose(r[0], 1.0, atol=1e-6):
        raise ValueError("expected a normalized autocorrelation starting at 1.0")
    ladder = []
    for t in range(1, len(r)):
        a_t = solve_toeplitz((r[:t], r[:t]), r[1:t + 1])
        variance_t = float(r[0] - a_t @ r[1:t + 1])
        ladder.append((a_t, float(np.sqrt(max(variance_t, 1e-12)))))
    coefficients, residual_std = ladder[-1]
    return {"coefficients": coefficients,
            "residual_std": residual_std,
            "order": len(coefficients),
            "ladder": ladder,
            "autocorrelation": r}


def whiten(values: np.ndarray, model: dict) -> np.ndarray:
    """Causal one-step prediction residual of a series under a fitted AR model.

    ``e[t] = x[t] - sum_j a[j] x[t-1-j]``, scaled to unit innovation variance.

    The first ``order`` positions do not have a full history and must not be handled by
    inventing one. An earlier version padded with ``x[0]``; the audit of 2026-09-19
    measured what that costs and it is severe. Because ``x[0]`` is systematically the
    lowest-risk token of an answer (about -1.65 SD), constant padding crushed e[0] to
    0.42*x[0] and injected a **+1.06 offset into e[1] of every answer**. The whitened
    readouts then predicted step 0 for 22-24% of erroneous answers against a true base
    rate of 12.25%, depressing every whitened arm by 0.35 to 0.56 pp -- an artefact of the
    padding, dressed up as a property of whitening.

    The fix is the statistically correct one rather than a different guess: at position
    ``t < order`` use the **order-t** predictor from the same autocorrelation, which is
    the best linear predictor given the history that actually exists. At t = 0 there is no
    history at all, so the predictor is the unconditional mean -- zero on a standardized
    series -- with unit innovation variance, and ``e[0] = x[0]``. Step 0 therefore stays
    scoreable and stays undistorted, which matters because 12.25% of first errors are
    there.
    """
    x = answer_standardize(values)
    a = np.asarray(model["coefficients"], dtype=float)
    ladder = model.get("ladder")
    if ladder is None:
        raise ValueError("model has no predictor ladder; refit with yule_walker")
    order = len(a)
    out = np.empty_like(x)
    out[0] = x[0]                                   # no history: predictor is the mean
    for t in range(1, min(order, len(x))):
        a_t, residual_t = ladder[t - 1]
        out[t] = (x[t] - a_t @ x[t - 1::-1][:t]) / residual_t
    if len(x) > order:
        full = lfilter(np.concatenate([[1.0], -a]), [1.0], x)
        out[order:] = full[order:] / model["residual_std"]
    return out


def topk_mean_steps(values: np.ndarray, spans: np.ndarray, k: int) -> np.ndarray:
    """Incumbent statistic: mean of the k largest values in each step, unordered."""
    x = np.asarray(values, dtype=float)
    spans = np.asarray(spans, dtype=int)
    out = np.empty(len(spans))
    for i, (a, b) in enumerate(spans):
        segment = x[a:b]
        if not len(segment):
            out[i] = 0.0
            continue
        kk = min(int(k), len(segment))
        out[i] = np.partition(segment, len(segment) - kk)[-kk:].mean()
    return out


def best_window_steps(values: np.ndarray, spans: np.ndarray, width: int) -> np.ndarray:
    """Matched-boxcar statistic: the best CONTIGUOUS run of ``width`` tokens in each step.

    Windows never cross a step boundary, so no evidence leaks between steps and the
    comparison against ``topk_mean_steps`` isolates one thing: whether the informative
    tokens are adjacent. A step shorter than ``width`` contributes its own mean.
    """
    x = np.asarray(values, dtype=float)
    spans = np.asarray(spans, dtype=int)
    prefix = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(len(spans))
    for i, (a, b) in enumerate(spans):
        length = b - a
        if length <= 0:
            out[i] = 0.0
            continue
        w = min(int(width), length)
        sums = prefix[a + w:b + 1] - prefix[a:b - w + 1]
        out[i] = float(sums.max() / w)
    return out
