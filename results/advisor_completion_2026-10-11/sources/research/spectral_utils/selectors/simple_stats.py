"""
Simple-statistic feature-selection floor (Step 186, the pre-fusion FS stage).

These are deliberately dumb, label-free selectors — the concrete form of the
Rajabinasab-et-al-2026 guardrail (many unsupervised FS methods lose to
random). If a sophisticated method (classical spectral FS, GroupFS,
Concrete-AE) does not clearly beat these, that is a result worth seeing.

Four selectors, each emitting sizes {4,5,6}:

  * random       — k features drawn uniformly (seeded). THE floor.
  * mad          — rank by median-absolute-deviation (a shape statistic; on
                   z-scored views this is non-degenerate, unlike variance).
  * kurtosis     — rank by |excess kurtosis| (non-gaussianity — the
                   projection-pursuit intuition that signal ≠ gaussian noise).
  * decorrelation— greedily pick a maximally-decorrelated set (min pairwise
                   |Spearman|, using the cell's precomputed rho): pure
                   min-redundancy, no relevance term (which would need labels).

IMPORTANT — variance is intentionally absent: `UnlabeledCell.V` is z-scored
per column (unit variance), so a variance selector is degenerate here.
"""

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis

from . import register

SIZES = (4, 5, 6)


def _emit(prefix, order_best_first, p, diag=None):
    out = []
    for s in SIZES:
        if s > p:
            continue
        out.append({'variant': f'{prefix}_s{s}',
                    'cols': np.sort(order_best_first[:s]),
                    'diag': dict(diag or {}, rule=prefix, size=s)})
    return out


def _decorrelation_order(rho):
    """Greedy min-redundancy: seed with the globally least-correlated feature,
    then repeatedly add the feature with the smallest MAX |rho| to the set."""
    p = rho.shape[0]
    R = np.array(rho, dtype=float)
    np.fill_diagonal(R, 0.0)
    start = int(np.argmin(R.mean(axis=1)))
    chosen = [start]
    remaining = set(range(p)) - {start}
    while remaining:
        best, best_val = None, np.inf
        for j in sorted(remaining):
            v = max(R[j, c] for c in chosen)
            if v < best_val:
                best_val, best = v, j
        chosen.append(best)
        remaining.discard(best)
    return np.asarray(chosen, dtype=np.int64)


@register('simple_stats')
def simple_stats(cell, rng, cache=None):
    V = cell.V
    p = V.shape[1]
    out = []

    # random floor — a fresh draw per size so it is an honest random subset
    for s in SIZES:
        if s > p:
            continue
        out.append({'variant': f'random_s{s}',
                    'cols': np.sort(rng.choice(p, size=s, replace=False)),
                    'diag': {'rule': 'random', 'size': s}})

    mad = np.median(np.abs(V - np.median(V, axis=0)), axis=0)
    out += _emit('mad', np.argsort(mad)[::-1], p)

    kurt = np.abs(sp_kurtosis(V, axis=0, fisher=True, bias=False))
    out += _emit('kurtosis', np.argsort(kurt)[::-1], p)

    out += _emit('decorr', _decorrelation_order(cell.rho), p)
    return out


def smoke():
    import sys
    sys.path.insert(0, __file__.rsplit('spectral_utils', 1)[0] + 'scripts')
    from smoke_selectors import _tiny_ctx
    from ..selector_bench import UnlabeledCell

    ctx = _tiny_ctx(informative=5)
    cell = UnlabeledCell.from_context(ctx)

    # determinism under equal-seeded rng
    a = {d['variant']: list(d['cols'])
         for d in simple_stats(cell, np.random.default_rng([0, 3]))}
    b = {d['variant']: list(d['cols'])
         for d in simple_stats(cell, np.random.default_rng([0, 3]))}
    assert a == b, "simple_stats not deterministic under equal-seeded rng"

    # decorrelation must pick a genuinely lower-redundancy set than a size-
    # matched exhaustive-worst — sanity: its mean |rho| below the pool mean
    R = np.array(cell.rho); np.fill_diagonal(R, 0.0)
    cols = a['decorr_s5']
    sub_mean = R[np.ix_(cols, cols)][np.triu_indices(len(cols), 1)].mean()
    pool_mean = R[np.triu_indices(R.shape[0], 1)].mean()
    assert sub_mean <= pool_mean + 1e-9, (
        f"decorr subset mean|rho| {sub_mean:.3f} not <= pool {pool_mean:.3f}")

    # random draws differ across seeds (not a constant selector)
    c = {d['variant']: list(d['cols'])
         for d in simple_stats(cell, np.random.default_rng([0, 99]))}
    assert a['random_s5'] != c['random_s5'] or cell.p <= 5, "random not seed-varying"

    for d in simple_stats(cell, np.random.default_rng([0, 3])):
        col = np.asarray(d['cols'])
        assert len(col) >= 3 and len(set(col.tolist())) == len(col)
        assert col.min() >= 0 and col.max() < cell.p
    print(f"    [note] simple_stats smoke: random/mad/kurtosis/decorr; "
          f"decorr mean|rho| {sub_mean:.3f} vs pool {pool_mean:.3f}")
