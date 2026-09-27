"""Exact evaluator components and source-question bootstrap for frozen transfer."""
import numpy as np
from .external_generalization.evaluation import metric as historical_metric


def divide(a, b, missing=-1.0):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, missing, dtype=float), where=b != 0)


def components(counts, benchmark):
    """Official Socratic sentinel semantics are intentionally preserved."""
    tp, fp, tn, fn = np.moveaxis(np.asarray(counts, float), -1, 0)
    pc, rc = divide(tp, tp + fp), divide(tp, tp + fn)
    pe, re = divide(tn, tn + fn), divide(tn, tn + fp)
    fc, fe = divide(2 * pc * rc, pc + rc), divide(2 * pe * re, pe + re)
    primary = (fc + fe) / 2 if benchmark == "socratic" else historical_metric(counts, benchmark)
    return dict(metric=primary, precision_correct=pc, recall_correct=rc,
                precision_error=pe, recall_error=re, f1_correct=fc, f1_error=fe)


def primary(counts, benchmark):
    return components(counts, benchmark)["metric"]


def summary(counts, benchmark):
    counts = np.asarray(counts, np.int64)
    c = counts.sum(0) if counts.ndim == 2 else counts
    out = {k: float(v) if np.isfinite(v) else None for k, v in components(c, benchmark).items()}
    out["confusion"] = c.tolist()
    out["steps"] = int(c.sum())
    out["undefined_components"] = [k for k, v in out.items()
                                   if k not in ("confusion", "steps", "metric") and (v is None or v < 0 or v > 1)]
    return out


def bootstrap(counts, groups, benchmark, draws=100000, seed=20260924):
    names = list(counts)
    x = np.stack([counts[n] for n in names], axis=1)
    if x.shape != (len(groups), len(names), 4):
        raise ValueError("unmatched counts")
    unique, index = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("need at least two source groups")
    grouped = np.zeros((len(unique), len(names), 4))
    np.add.at(grouped, index, x)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(names)))
    sentinel_draws = np.zeros(len(names), dtype=int)
    for start in range(0, draws, 64):
        stop = min(start + 64, draws)
        ix = rng.integers(len(unique), size=(stop-start, len(unique)))
        comp = components(grouped[ix].sum(axis=1), benchmark)
        samples[start:stop] = comp["metric"]
        sentinel_draws += np.any(np.stack([((v < 0) | (v > 1) | ~np.isfinite(v))
                                           for k, v in comp.items() if k != "metric"]), axis=0).sum(0)
    return names, samples, dict(zip(names, sentinel_draws.tolist()))


def contrast(counts, names, samples, left, right, groups, benchmark, family=18, seed=20260924):
    delta = samples[:, names.index(left)] - samples[:, names.index(right)]
    finite = delta[np.isfinite(delta)]
    alpha = .05 / family
    return dict(left=left, right=right,
                delta=float(primary(np.asarray(counts[left]).sum(0), benchmark)
                            - primary(np.asarray(counts[right]).sum(0), benchmark)),
                ci_bonferroni=np.quantile(finite, [alpha/2, 1-alpha/2]).tolist() if len(finite) else None,
                family_size=family, paired_N=len(groups), paired_groups=len(set(groups)),
                draws=len(samples), valid_draws=len(finite), seed=seed)
