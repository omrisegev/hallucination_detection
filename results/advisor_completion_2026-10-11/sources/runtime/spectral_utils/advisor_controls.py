"""Auditable PRMBench decision and nuisance-control arithmetic.

Scores are fixed risk scores (higher means error); labels enter evaluation only.
These helpers never select score signs, banks, or thresholds with labels.
"""
from __future__ import annotations

import re
import numpy as np


def answer_z(score, offsets):
    out = np.zeros_like(score, dtype=float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        v = np.asarray(score[a:b], float)
        sd = v.std()
        if sd > 1e-12:
            out[a:b] = (v-v.mean())/sd
    return out


def global_z(values):
    x = np.asarray(values, float)
    sd = x.std(axis=0)
    return (x-x.mean(axis=0))/np.where(sd > 1e-12, sd, 1.)


def top_flags(score, offsets, counts):
    out = np.zeros(len(score), dtype=bool)
    for a, b, k in zip(offsets[:-1], offsets[1:], counts):
        if not 0 <= int(k) <= b-a:
            raise ValueError("Invalid per-answer quota")
        out[a+np.argsort(-score[a:b], kind="stable")[:int(k)]] = True
    return out


def f1(tp, fp, fn):
    den = 2*np.asarray(tp)+fp+fn
    return np.divide(2*np.asarray(tp, dtype=float), den,
                     out=np.zeros_like(den, dtype=float), where=den != 0)


def metrics(y, flag):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp, fn, tn = [int(x.sum()) for x in
                      (y&flag, ~y&flag, y&~flag, ~y&~flag)]
    fe, fv = float(f1(tp, fp, fn)), float(f1(tn, fp, fn))
    return dict(tp_error=tp, fp_error=fp, fn_error=fn, tn_valid=tn,
                f1_error=fe, f1_valid=fv, prm_score=(fe+fv)/2)


def pair_stats(score, y, strata=None):
    """Within-answer matched-pair AUC; exact ties receive half credit."""
    score, y = np.asarray(score), np.asarray(y, bool)
    if strata is None:
        strata = np.zeros(len(y), int)
    wins, pairs = 0., 0
    for h in np.unique(strata):
        selected = strata == h
        error, valid = score[selected&y], score[selected&~y]
        if len(error) and len(valid):
            diff = error[:, None]-valid[None, :]
            pairs += diff.size
            wins += np.count_nonzero(diff > 0)+.5*np.count_nonzero(diff == 0)
    return dict(auc=wins/pairs if pairs else np.nan, wins=float(wins), pairs=int(pairs))


def repetition_risk(steps):
    """Max Jaccard with any earlier step: casefold word 3-grams.

    A pair involving fewer than three words uses unigrams for both members.
    Empty-token pairs have zero overlap. First step is always zero.
    """
    tokens = [re.findall(r"\w+", str(s).casefold(), flags=re.UNICODE) for s in steps]
    result = np.zeros(len(tokens))
    for i, current in enumerate(tokens):
        for previous in tokens[:i]:
            order = 3 if min(len(current), len(previous)) >= 3 else 1
            a = {tuple(current[j:j+order]) for j in range(len(current)-order+1)}
            b = {tuple(previous[j:j+order]) for j in range(len(previous)-order+1)}
            union = a|b
            if union:
                result[i] = max(result[i], len(a&b)/len(union))
    return result


def confusion_by_group(y, flag, groups, selected, n_groups):
    return np.column_stack([np.bincount(groups[selected&mask], minlength=n_groups)
                            for mask in (y&flag, ~y&flag, y&~flag, ~y&~flag)])


def score_from_confusion(confusion, multiplicity):
    tp, fp, fn, tn = (multiplicity@confusion[:, j] for j in range(4))
    return (f1(tp, fp, fn)+f1(tn, fp, fn))/2


def bootstrap_answer_mean(values, answer_groups, selected, multiplicity):
    selected = np.asarray(selected, bool)&np.isfinite(values)
    ng = multiplicity.shape[1]
    sums = np.bincount(answer_groups[selected], weights=values[selected], minlength=ng)
    counts = np.bincount(answer_groups[selected], minlength=ng)
    den = multiplicity@counts
    return np.divide(multiplicity@sums, den,
                     out=np.full(len(multiplicity), np.nan), where=den > 0)
