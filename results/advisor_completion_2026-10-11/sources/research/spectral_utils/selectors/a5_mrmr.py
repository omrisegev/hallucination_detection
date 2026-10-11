"""
A5 — mRMR hybrid selector: anchor-relevance minus redundancy penalty (Step 189).

Salvage of A4's anchor-affinity finding (Step 187): the anchor-correlation-
ranked family (a4.anchor_s*) is the best learned selector on H16 (0.6593) but
is statistically indistinguishable from bare epr (0.6606, 25W/26L) because it
just picks epr's most-correlated clones — high relevance, zero diversity.
This adds the missing mRMR term (max-Relevance min-Redundancy, Peng-Long-Ding
2005 IEEE TPAMI 27(8)): at each greedy step, score candidate j by

    relevance(j)       = |Spearman(V[:,j], anchor)|        (label-free)
    redundancy(j | S)  = mean_{k in S} |Spearman(V[:,j], V[:,k])|
                          (from cell.rho — already cached, no recompute)
    score(j | S)        = relevance(j) - alpha * redundancy(j | S)

alpha=0 reduces exactly to A4's anchor-affinity ranking (kept as
a5.mrmr_a0.0_* for a direct before/after comparison in the SAME family, same
harness run, same seeds). Higher alpha trades relevance for diversity.
"""

import numpy as np

from . import register

ALPHAS = (0.0, 0.3, 0.5, 0.7)
FIXED_SIZES = (4, 5, 6)


def _relevance(V, anchor):
    """|Spearman(feature, anchor)| per column — label-free (anchor carries
    no labels), matches A4's own relevance definition exactly."""
    import scipy.stats
    out = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        r, _ = scipy.stats.spearmanr(V[:, j], anchor)
        out[j] = abs(r) if np.isfinite(r) else 0.0
    return out


def _mrmr_path(rel, rho, max_size):
    """Greedy mRMR forward selection for every alpha in ALPHAS at once (same
    relevance/redundancy inputs, alpha only changes the score combination —
    cheap to fan out). Returns {alpha: {size: (cols, score)}}."""
    p = len(rel)
    paths = {a: {} for a in ALPHAS}
    for alpha in ALPHAS:
        cols = [int(np.argmax(rel))]          # seed: single most-relevant feature
        paths[alpha][1] = (list(cols), float(rel[cols[0]]))
        while len(cols) < min(max_size, p):
            best_j, best_score = None, -np.inf
            for j in range(p):
                if j in cols:
                    continue
                redund = float(np.mean(rho[j, cols])) if cols else 0.0
                score = rel[j] - alpha * redund
                if score > best_score:
                    best_score, best_j = score, j
            cols.append(best_j)
            paths[alpha][len(cols)] = (sorted(cols), best_score)
    return paths


@register('a5_mrmr')
def a5_mrmr(cell, rng, cache=None):
    V, anchor, rho = cell.V, cell.anchor, cell.rho
    p = V.shape[1]
    if p < 3:
        return []
    max_size = min(8, p)
    rel = _relevance(V, anchor)
    paths = _mrmr_path(rel, rho, max_size)

    out = []
    for alpha in ALPHAS:
        path = paths[alpha]
        a_tag = f"{alpha:.1f}"
        for s in FIXED_SIZES:
            if s in path and s <= p and s >= 3:
                cols, score = path[s]
                out.append({
                    'variant': f'a5.mrmr_a{a_tag}_s{s}',
                    'cols': np.sort(cols),
                    'diag': {'alpha': alpha, 'mrmr_score': score,
                             'mean_relevance': float(np.mean(rel[cols]))},
                })
        # adaptive size: elbow on the mRMR score gain per added feature
        # (diminishing returns — stop where the marginal gain drops below
        # 20% of the first-step gain), capped to [3, max_size]
        sizes = sorted(k for k in path if k >= 3)
        if len(sizes) >= 2:
            first_gain = path[sizes[0]][1] - path[max(1, sizes[0] - 1)][1] \
                if sizes[0] - 1 in path else path[sizes[0]][1]
            best_size = sizes[0]
            for i in range(1, len(sizes)):
                gain = path[sizes[i]][1] - path[sizes[i - 1]][1]
                if first_gain > 1e-9 and gain < 0.2 * first_gain:
                    break
                best_size = sizes[i]
            cols, score = path[best_size]
            out.append({
                'variant': f'a5.mrmr_a{a_tag}_adapt',
                'cols': np.sort(cols),
                'diag': {'alpha': alpha, 'mrmr_score': score, 'size': best_size},
            })
    return out


def smoke():
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL

    rng_np = np.random.default_rng(20260719)
    n, p = 300, 10
    y = rng_np.standard_normal(n)          # signal direction 1
    y2 = rng_np.standard_normal(n)         # signal direction 2, INDEPENDENT of y
    # anchor correlates with BOTH signal directions (label-free consensus view)
    anchor = zscore(y + y2 + 0.2 * rng_np.standard_normal(n))

    cols = []
    # feature 0: strong anchor-correlate via y ("epr"-like signal)
    cols.append(zscore(y + 0.1 * rng_np.standard_normal(n)))
    # features 1-3: near-perfect CLONES of feature 0 (high relevance via y,
    # ~1.0 redundancy with each other) — an mRMR selector with alpha>0 must
    # eventually prefer a genuinely-different informative feature over a 4th
    # near-clone
    for _ in range(3):
        cols.append(zscore(cols[0] + 0.05 * rng_np.standard_normal(n)))
    # feature 4: a SECOND independent signal direction (via y2) — comparable
    # relevance to the clones (anchor sees both y and y2 equally) but ~0
    # redundancy with the y-based clone cluster
    cols.append(zscore(y2 + 0.1 * rng_np.standard_normal(n)))
    # 5 pure-noise features
    for _ in range(5):
        cols.append(zscore(rng_np.standard_normal(n)))

    V = np.column_stack(cols)
    pool = list(CANONICAL_POOL[:p])
    pool_bits = np.arange(p, dtype=np.uint8)
    rho = np.abs(np.corrcoef(V.T))

    cell = UnlabeledCell(domain='smoke', cell_key='mrmr', pool=pool,
                         pool_bits=pool_bits, V=V, anchor=anchor,
                         anchor_name=pool[0], rho=rho)

    sels1 = a5_mrmr(cell, np.random.default_rng(123))
    sels2 = a5_mrmr(cell, np.random.default_rng(123))
    assert len(sels1) == len(ALPHAS) * (len(FIXED_SIZES) + 1)
    for s1, s2 in zip(sels1, sels2):
        assert s1['variant'] == s2['variant'], "order mismatch"
        assert list(s1['cols']) == list(s2['cols']), f"{s1['variant']} nondeterministic"

    # alpha=0.0 (pure relevance) at size 4 must pick the clone cluster
    # (features 0-3 are the 4 highest-relevance columns by construction)
    a0_s4 = next(s for s in sels1 if s['variant'] == 'a5.mrmr_a0.0_s4')
    assert set(a0_s4['cols']) == {0, 1, 2, 3}, (
        f"alpha=0 should pick the pure-relevance clone cluster, got {a0_s4['cols']}")

    # alpha=0.7 (strong diversity) at size 4 must include feature 4 (the
    # second independent signal) instead of a 4th clone — the mRMR payoff
    a7_s4 = next(s for s in sels1 if s['variant'] == 'a5.mrmr_a0.7_s4')
    assert 4 in a7_s4['cols'], (
        f"alpha=0.7 should trade a clone for the independent 2nd signal "
        f"(feature 4), got {a7_s4['cols']}")
    assert set(a7_s4['cols']) != {0, 1, 2, 3}, (
        "alpha=0.7 collapsed to the same pure-relevance clone cluster as alpha=0")

    print(f"    [note] a5 mrmr smoke: {len(sels1)} variants; "
          f"alpha=0 picks clones {sorted(a0_s4['cols'])}, "
          f"alpha=0.7 diversifies to {sorted(a7_s4['cols'])}")
