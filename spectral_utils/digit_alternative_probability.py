"""Decoding-independent, top-K-censored second-digit probability.

This experimental module is independent of the provided/generated token IDs.
Historical disagreement is retained only as a comparison helper.
"""
import numpy as np


def second_digit_probability(ids, logprobs, digit_ids):
    ids = np.asarray(ids)
    lp = np.asarray(logprobs, dtype=float)
    digits = np.asarray(digit_ids)
    if ids.ndim != 2 or ids.shape != lp.shape or ids.shape[1] < 2:
        raise ValueError('Expected matching T x K IDs/logprobs, K >= 2')
    if len(digits) != 10 or len(np.unique(digits)) != 10:
        raise ValueError('Expected ten distinct verified digit IDs')
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Token IDs must be integers')
    if not np.isfinite(lp).all() or np.any(lp > 2e-6):
        raise ValueError('Invalid log probabilities')
    if np.any(np.diff(lp, axis=1) > 2e-6):
        raise ValueError('Top-K must be sorted by descending probability')
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError('Duplicate token IDs in top-K')
    p = np.exp(lp)
    if np.any(p.sum(axis=1) > 1.0001):
        raise ValueError('Saved probability mass exceeds one')
    mask = np.isin(ids, digits)
    seen = mask.sum(axis=1)
    # Partition is independent of vocabulary order and handles exact probability ties.
    lower = np.partition(np.where(mask, p, 0.), -2, axis=1)[:, -2]
    upper = np.where(seen >= 2, lower, p[:, -1])
    return lower, upper, seen


def disagreement(provided, top1, digit_ids):
    provided, top1 = np.asarray(provided), np.asarray(top1)
    if provided.ndim != 1 or provided.shape != top1.shape:
        raise ValueError('Misaligned provided/top1 IDs')
    return (np.isin(provided, digit_ids) & np.isin(top1, digit_ids)
            & (provided != top1)).astype(float)


def prefix_innovation(values):
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x)
    if len(x) > 1:
        out[1:] = x[1:] - np.cumsum(x)[:-1] / np.arange(1, len(x))
    return out


def digit_innovation_step_max(event, provided, digit_ids, spans):
    """Historical opportunity mask; no-opportunity steps retain storage zero.

    The fusion bank stored zero for unavailable step evidence. Return availability
    explicitly so callers can disclose that zero is missing evidence, not safety.
    """
    values = prefix_innovation(event)
    active = np.isin(provided, digit_ids) & (np.arange(len(values)) > 0)
    out = np.zeros(len(spans)); available = np.zeros(len(spans), dtype=bool)
    for i, (a, b) in enumerate(spans):
        selected = values[a:b][active[a:b]]
        if len(selected):
            out[i] = selected.max(); available[i] = True
    return out, available


def step_top_mean(values, spans, count):
    x = np.asarray(values, dtype=float)
    spans = np.asarray(spans, dtype=int)
    if x.ndim != 1 or not np.isfinite(x).all() or count < 1:
        raise ValueError('Invalid stream/count')
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError('Invalid spans')
    if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > len(x)) or np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError('Empty/out-of-range spans')
    return np.array([np.partition(x[a:b], -min(count, b-a))[-min(count, b-a):].mean()
                     for a, b in spans])
