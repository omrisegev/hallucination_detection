"""CPU-only protocol repairs; no labels, model inference, or implicit mean fusion.

The residual bank preserves channels until a caller's explicitly chosen learner.
Normalization uses the complete current answer: this is offline answer-local,
not streaming-causal. Existing frozen experiment artifacts remain unchanged.
"""
from __future__ import annotations

import numpy as np


def residual_step_bank(residuals, spans, *, offset=0, top_k=5):
    """Top-k per step/channel, then standardize each channel across answer steps.

    Empty/invalid spans and nonfinite inputs fail loudly. Constant channels are
    zero, including the one-step case. No channel averaging takes place.
    Apply exactly this function to predictive AND zero-predictor residuals.
    """
    x = np.asarray(residuals, dtype=np.float64)
    s = np.asarray(spans)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("residuals must be a nonempty finite token-by-channel matrix")
    if s.ndim != 2 or s.shape[1] != 2 or not len(s):
        raise ValueError("spans must be a nonempty step-by-two matrix")
    if not np.issubdtype(s.dtype, np.integer) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("integer spans and positive integer top_k required")
    s = s - offset
    if (s[:, 0] < 0).any() or (s[:, 1] > len(x)).any() or (s[:, 1] <= s[:, 0]).any():
        raise ValueError("spans must cover nonempty ranges inside this answer")
    profile = np.stack([
        np.partition(x[a:b], b-a-min(top_k, b-a), axis=0)[-min(top_k, b-a):].mean(0)
        for a, b in s
    ])
    sd = profile.std(axis=0)
    active = sd > 1e-8
    normalized = np.zeros_like(profile)
    normalized[:, active] = (profile[:, active] - profile[:, active].mean(0)) / sd[active]
    return normalized, {"raw_step_profile": profile, "active_channels": active,
                        "step_channel_sd": sd, "fit_scope": "current_answer_offline"}


def add_native_controls(methods, native, *, arms=("iu", "shrink_iu", "lsml")):
    """Add a same-mask equal comparator for every native learned window row.

    Mutates only the passed new result dictionary, never score arrays or the
    original equal row. Return a paired native contrast list for the evaluator.
    """
    pairs = []
    for arm in arms:
        for readout in ("top10", "mean"):
            learned = f"window_{arm}_{readout}_native"
            equal = f"window_equal_{readout}"
            if learned not in methods:
                continue
            mask = np.asarray(native[arm], dtype=bool)
            if not np.array_equal(methods[learned]["valid"], mask):
                raise ValueError(f"native mask disagrees for {learned}")
            equal_valid = np.asarray(methods[equal].get("valid", np.ones_like(mask)), dtype=bool)
            if equal_valid.shape != mask.shape or (mask & ~equal_valid).any():
                raise ValueError("equal comparator lacks native rows; refuse unmatched contrast")
            control = f"{equal}_on_{arm}_native"
            methods[control] = dict(methods[equal], valid=mask.copy())
            pairs.append((learned, control, "learned_minus_equal_native_rows"))
    return pairs


def paired_population(groups, eligible_a, eligible_b, *, draws, valid_draws):
    """Metadata for an ALREADY matched contrast, never disguise unequal masks.

    Bootstrap draws are Monte Carlo repetitions, not the number of observations.
    Callers must form masks for the specific endpoint before invoking this helper.
    """
    g = np.asarray(groups)
    a, b = np.asarray(eligible_a, bool), np.asarray(eligible_b, bool)
    if g.ndim != 1 or g.shape != a.shape or a.shape != b.shape:
        raise ValueError("aligned one-dimensional groups and masks required")
    if not np.array_equal(a, b):
        raise ValueError("unmatched populations: recompute estimates and draws on common rows")
    if not 0 <= valid_draws <= draws:
        raise ValueError("invalid bootstrap draw counts")
    return {"paired_N": int(a.sum()), "paired_groups": int(len(np.unique(g[a]))),
            "B": int(draws), "valid_draws": int(valid_draws)}
