"""Within-step token-to-step readouts beyond top-k, plus a sequential first-crossing rule
and a label-free per-channel readout choice.

Motivation (Steps 426-427, HISTORY): inside an erroneous step the informative tokens are
scattered, not contiguous, and a contiguous window loses 7 pp to the unordered top-5 mean,
so the readout wants MORE aggregation, not a sharper peak.  ActMap (arXiv 2609.11498)
makes the same point from the supervised side: it replaces every hidden-state trajectory
by a small set of temporal statistics and matches a detector that sees the full tensor.
The reducers here are that family, applied to the eleven grey-box channels of the frozen
bank, at the step granularity ProcessBench scores.  Nothing here is labelled with an
author's name or claims to be their method (project rule: tailor, never transplant).

Three conventions, fixed on purpose:

* Inputs are the answer-standardised token matrix exactly as
  ``scripts/experiments/cvf_v2/core.py::profiles`` standardises it (median / IQR per
  channel within the answer).  ``robust_standardize_tokens`` reproduces that formula and
  the tests pin the two together, so the seven frozen readouts and the ten new ones read
  the same numbers.
* Index 0 must stay reachable.  ``step_jump`` compares step 0 against the answer median
  (zero after standardisation) instead of against nothing; ``page_first_crossing`` can
  fire on step 0.
* A rule that fails to fire falls back to a finite profile, never to a silent default.
  ``page_first_crossing`` returns the untouched statistic when the threshold is never
  crossed, so the downstream argmax is the statistic's own answer and the fallback rate is
  reported, not hidden.

The Page statistic itself is ``changepoint_step_readout_v1.page_cusum`` (Step 424); what
is new is the per-step maximum, the training-fold quantile threshold and the token-to-step
mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from .changepoint_step_readout_v1 import page_cusum

__all__ = [
    "EXT_READOUTS",
    "DEFAULTS",
    "robust_standardize_tokens",
    "step_topk",
    "step_mean",
    "step_std",
    "step_iqr",
    "step_frac_above",
    "step_slope",
    "step_jump",
    "step_quantile",
    "step_first_token",
    "step_boxcar_max",
    "step_page_wmax",
    "ext_profiles",
    "answer_max_by_offsets",
    "page_threshold",
    "page_first_crossing",
    "soft_mass",
    "argmax_earliest",
    "tied_argmax",
    "consensus_readout_choice",
]

EXT_READOUTS = ['top30', 'std', 'iqr', 'frac_above_z', 'slope', 'jump', 'q90',
                'first_token', 'boxcar8_max', 'page_wmax']
DEFAULTS = {'frac_z_threshold': 1.0, 'boxcar_width': 8, 'page_k': 0.5, 'quantile': 0.9}


# ------------------------------------------------------------------ input handling
def robust_standardize_tokens(tokens: np.ndarray) -> np.ndarray:
    """Median / (IQR/1.349) per column, the exact formula of ``cvf_v2.core.profiles``."""
    x = np.asarray(tokens, float)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("expected a non-empty finite token matrix")
    med = np.median(x, axis=0)
    lo, hi = np.percentile(x, [25, 75], axis=0)
    scale = (hi - lo) / 1.349
    scale = np.where(scale > 1e-8, scale, np.where(x.std(0) > 1e-8, x.std(0), 1.))
    return (x - med) / scale


def _check(x: np.ndarray, spans: np.ndarray):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    spans = np.asarray(spans, int)
    if spans.ndim != 2 or spans.shape[1] != 2 or not len(spans):
        raise ValueError("expected a [steps, 2] span array")
    if not np.isfinite(x).all():
        raise ValueError("token matrix must be finite")
    for a, b in spans:
        if not 0 <= a < b <= len(x):
            raise ValueError(f"invalid span {(a, b)} / {len(x)}")
    return x, spans


def _reduce(x, spans, fn):
    x, spans = _check(x, spans)
    out = np.empty((len(spans), x.shape[1]))
    for s, (a, b) in enumerate(spans):
        out[s] = fn(x[a:b])
    return out


# ------------------------------------------------------------------ reducers
def step_topk(x, spans, k: int) -> np.ndarray:
    """Mean of the ``k`` largest tokens in the step (``k`` clipped to the step length)."""
    def top(v):
        kk = min(int(k), len(v))
        return np.partition(v, len(v) - kk, axis=0)[-kk:].mean(0)
    return _reduce(x, spans, top)


def step_mean(x, spans) -> np.ndarray:
    return _reduce(x, spans, lambda v: v.mean(0))


def step_std(x, spans) -> np.ndarray:
    """Population standard deviation inside the step; a one-token step gives 0."""
    return _reduce(x, spans, lambda v: v.std(0))


def step_iqr(x, spans) -> np.ndarray:
    return _reduce(x, spans, lambda v: np.subtract(*np.percentile(v, [75, 25], axis=0)))


def step_frac_above(x, spans, threshold: float = DEFAULTS['frac_z_threshold']) -> np.ndarray:
    """Fraction of the step's tokens at or above ``threshold`` (answer-standardised units)."""
    return _reduce(x, spans, lambda v: (v >= float(threshold)).mean(0))


def step_slope(x, spans) -> np.ndarray:
    """Ordinary least-squares slope of the tokens against their in-step index; 0 if n < 2."""
    def slope(v):
        n = len(v)
        if n < 2:
            return np.zeros(v.shape[1])
        t = np.arange(n) - (n - 1) / 2.
        return t @ (v - v.mean(0)) / (t @ t)
    return _reduce(x, spans, slope)


def step_jump(x, spans) -> np.ndarray:
    """Step mean minus the previous step's mean; step 0 is compared with the answer median
    (zero after standardisation) so it remains reachable."""
    means = step_mean(x, spans)
    out = np.empty_like(means)
    out[0] = means[0]
    out[1:] = np.diff(means, axis=0)
    return out


def step_quantile(x, spans, q: float = DEFAULTS['quantile']) -> np.ndarray:
    return _reduce(x, spans, lambda v: np.quantile(v, float(q), axis=0))


def step_first_token(x, spans) -> np.ndarray:
    """The step's first token.  Snel and Oh (arXiv 2507.20836) report the first hallucinated
    token as the most detectable one; this is the step-level analogue."""
    return _reduce(x, spans, lambda v: v[0])


def step_boxcar_max(x, spans, width: int = DEFAULTS['boxcar_width']) -> np.ndarray:
    """Best contiguous window of ``width`` tokens inside the step (mean), windows never cross
    a step boundary; a shorter step contributes its own mean.  Same statistic as
    ``step_measure_v1.best_window_steps``, vectorised over channels."""
    x, spans = _check(x, spans)
    prefix = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    out = np.empty((len(spans), x.shape[1]))
    for s, (a, b) in enumerate(spans):
        w = min(int(width), b - a)
        sums = prefix[a + w:b + 1] - prefix[a:b - w + 1]
        out[s] = sums.max(0) / w
    return out


def step_page_wmax(x, spans, k: float = DEFAULTS['page_k']) -> np.ndarray:
    """Per-step maximum of the one-sided Page statistic W_t = max(0, W_{t-1} + z_t - k).

    ``z`` is the channel re-centred by the answer MEAN and scaled by its std (the module's
    inputs are median / IQR standardised, which leaves a right-skewed channel with a positive
    mean; a CUSUM run on such increments drifts upward with no change at all, and its
    threshold degenerates into an answer-length statistic).  With mean-centred increments the
    statistic has drift -k in the absence of a change, which is what Page's rule assumes.
    The step maximum reaches a threshold exactly when the first token in the step does, so a
    threshold on this profile is the token-level first crossing mapped to steps."""
    x, spans = _check(x, spans)
    stat = np.column_stack([page_cusum(x[:, j], k=k, standardize=True)['statistic']
                            for j in range(x.shape[1])])
    return _reduce(stat, spans, lambda v: v.max(0))


def ext_profiles(x, spans, cfg: dict | None = None) -> np.ndarray:
    """(steps, channels, len(EXT_READOUTS)) in ``EXT_READOUTS`` order."""
    c = {**DEFAULTS, **(cfg or {})}
    x, spans = _check(x, spans)
    cols = [step_topk(x, spans, 30), step_std(x, spans), step_iqr(x, spans),
            step_frac_above(x, spans, c['frac_z_threshold']), step_slope(x, spans),
            step_jump(x, spans), step_quantile(x, spans, c['quantile']), step_first_token(x, spans),
            step_boxcar_max(x, spans, c['boxcar_width']), step_page_wmax(x, spans, c['page_k'])]
    out = np.stack(cols, axis=2)
    if not np.isfinite(out).all():
        raise AssertionError("extended readouts must be finite")
    return out


# ------------------------------------------------------------------ first crossing
def answer_max_by_offsets(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-answer maximum of a (steps, ...) array given answer step offsets (len n+1)."""
    offsets = np.asarray(offsets, int)
    return np.maximum.reduceat(np.asarray(values, float), offsets[:-1], axis=0)


def page_threshold(per_answer_max: np.ndarray, q: float) -> np.ndarray:
    """Per-channel quantile ``q`` of the per-answer maximum statistic over the training
    answers.  Labels are not consulted (precedent: the mean-entropy q0.3 gate)."""
    if not 0. <= q <= 1.:
        raise ValueError("q must lie in [0, 1]")
    return np.quantile(np.asarray(per_answer_max, float), float(q), axis=0)


def page_first_crossing(wmax: np.ndarray, h: np.ndarray):
    """Onset-style profile: ``wmax`` up to and including the first step whose statistic
    reaches ``h`` (per channel), ``-inf`` after it; if no step crosses, the finite ``wmax``
    column is returned unchanged (argmax fallback).  Returns (profile, crossed[channels])."""
    w = np.asarray(wmax, float)
    if w.ndim == 1:
        w = w[:, None]
    h = np.broadcast_to(np.asarray(h, float), (w.shape[1],))
    out = w.copy()
    crossed = np.zeros(w.shape[1], bool)
    for j in range(w.shape[1]):
        hit = np.flatnonzero(w[:, j] >= h[j])
        if len(hit):
            crossed[j] = True
            out[hit[0] + 1:, j] = -np.inf
    return out, crossed


# ------------------------------------------------------------------ consensus choice
def soft_mass(profile: np.ndarray) -> np.ndarray:
    """Softmax over steps of each column's standardised finite entries; ``-inf`` entries get
    zero mass.  The per-column normalisation of ``cvf_v2.core.encode('soft')`` without the
    cumulative sum."""
    p = np.asarray(profile, float)
    if p.ndim == 1:
        p = p[:, None]
    S, m = p.shape
    if S == 1:
        return np.ones((1, m))
    z = np.full_like(p, -np.inf)
    for j in range(m):
        finite = np.isfinite(p[:, j])
        if not finite.any():
            raise ValueError("empty profile")
        v = p[finite, j]
        sd = v.std()
        z[finite, j] = (v - v.mean()) / (sd if sd > 1e-8 else 1.)
    return softmax(z, axis=0)


def argmax_earliest(profile: np.ndarray) -> np.ndarray:
    """Argmax over steps per column, earliest step on exact ties (``np.argmax``)."""
    p = np.asarray(profile, float)
    return np.argmax(p, axis=0)


def tied_argmax(profile: np.ndarray) -> np.ndarray:
    """Whether each column attains its maximum at more than one step (the step-0 voter
    pathology of discrete channels, review of cumulative-vote v2, section 2)."""
    p = np.asarray(profile, float)
    return (p == p.max(axis=0, keepdims=True)).sum(axis=0) > 1


def consensus_readout_choice(answers, base: int = 0, sweeps: int = 2, cells=None,
                             candidate_mask=None):
    """Label-free per-channel readout choice by agreement with the other channels.

    ``answers``: list of (S_i, C, K) profile arrays (training answers only).  For channel
    ``c`` and candidate readout ``r`` the score is the fraction of answers (with S_i > 1)
    whose ``argmax_s p[:, c, r]`` equals the argmax of the equal soft fusion of the OTHER
    channels at their current readouts.  Sweep 1 holds the others at ``base``; each later
    sweep holds them at the previous sweep's choice.  Exact ties resolve to the lowest
    readout index.  With ``cells`` (one label per answer) the agreement is the mean over
    cells of the within-cell mean, matching the cell weighting of the training matrix.
    ``candidate_mask`` (K,) excludes readouts (e.g. non-finite ones on PRMBench).

    Returns (choice[C], agreement[sweeps, C, K]).  This rule is transductive over the
    training answers and uses no error labels; it is not the FUSE triplet statistic, which
    the 2026-09-22 note closed as a selector.
    """
    usable = [(i, np.asarray(p, float)) for i, p in enumerate(answers) if len(p) > 1]
    if not usable:
        raise ValueError("no answer with more than one step")
    C, K = usable[0][1].shape[1:]
    mask = np.ones(K, bool) if candidate_mask is None else np.asarray(candidate_mask, bool)
    if not mask.any() or not mask[base]:
        raise ValueError("base readout must be a candidate")
    arg = np.stack([argmax_earliest(p.reshape(len(p), C * K)).reshape(C, K) for _, p in usable])
    cell_of = None if cells is None else np.asarray(cells)[[i for i, _ in usable]]
    cur = np.full(C, int(base))
    history = []
    for _ in range(int(sweeps)):
        hit = np.empty((len(usable), C, K))
        for n, (_, p) in enumerate(usable):
            z = soft_mass(p[:, np.arange(C), cur])          # (S, C)
            total = z.sum(1, keepdims=True)
            others = (total - z) if C > 1 else z
            target = np.argmax(others, axis=0)              # (C,)
            hit[n] = arg[n] == target[:, None]
        if cell_of is None:
            agreement = hit.mean(0)
        else:
            agreement = np.mean([hit[cell_of == c].mean(0) for c in sorted(set(cell_of))], axis=0)
        history.append(agreement)
        scored = np.where(mask[None, :], agreement, -np.inf)
        cur = np.argmax(scored, axis=1)
    return cur, np.stack(history)
