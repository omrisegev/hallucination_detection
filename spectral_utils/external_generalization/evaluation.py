"""Evaluator-only metrics; positive class is a correct step."""
import numpy as np
from scipy.stats import rankdata


def confusion(correct, predictions, include=None):
    y, p = np.asarray(correct, bool), np.asarray(predictions)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("unaligned decisions")
    mask = np.ones(len(y), bool) if include is None else np.asarray(include, bool)
    if mask.shape != y.shape or not np.isin(p, [-2, 0, 1]).all():
        raise ValueError("invalid inclusion mask or decision")
    # Author Hard2Verify grading penalizes unparsed/missing steps as incorrect.
    # This happens ONLY here, after sealing; no gold-derived prediction is saved.
    p = np.where(p == -2, ~y, p).astype(bool)
    return np.array([np.sum(mask & y & p), np.sum(mask & ~y & p),
                     np.sum(mask & ~y & ~p), np.sum(mask & y & ~p)], dtype=np.int64)


def metric(counts, benchmark):
    tp, fp, tn, fn = np.moveaxis(np.asarray(counts, float), -1, 0)
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    if benchmark == "hard2verify":
        tpr, tnr = div(tp, tp + fn), div(tn, tn + fp)
        result = div(2 * tpr * tnr, tpr + tnr)
        return np.where((tp + fn > 0) & (tn + fp > 0), result, np.nan)
    if benchmark == "socratic":
        return .5 * (div(2 * tp, 2 * tp + fp + fn) + div(2 * tn, 2 * tn + fp + fn))
    raise ValueError("unknown benchmark")


def within_auc(correct, risk, include):
    y, s = ~np.asarray(correct, bool), np.asarray(risk, float)
    keep = np.asarray(include, bool)
    y, s = y[keep], s[keep]
    pos, neg = y.sum(), (~y).sum()
    if not pos or not neg:
        return None
    return float((rankdata(s)[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def paired_bootstrap(a, b, groups, benchmark, family_size, draws=100000, seed=20260924):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.shape != (len(groups), 4) or family_size < 1:
        raise ValueError("unmatched contrasts")
    unique, index = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two source groups required")
    ga, gb = np.zeros((len(unique), 4)), np.zeros((len(unique), 4))
    np.add.at(ga, index, a)
    np.add.at(gb, index, b)
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, draws, 256):
        ix = rng.integers(len(unique), size=(min(256, draws-start), len(unique)))
        diffs.extend(metric(ga[ix].sum(1), benchmark) - metric(gb[ix].sum(1), benchmark))
    d = np.asarray(diffs)
    d = d[np.isfinite(d)]
    alpha = .05 / family_size
    return {"delta": float(metric(a.sum(0), benchmark) - metric(b.sum(0), benchmark)),
            "ci_bonferroni": np.quantile(d, [alpha/2, 1-alpha/2]).tolist() if len(d) else None,
            "family_size": family_size, "paired_N": len(a), "paired_groups": len(unique),
            "draws": draws, "valid_draws": len(d), "seed": seed}
