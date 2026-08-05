"""
A9 — Determinantal Point Process MAP: keep the subset spanning the largest volume.

THE CONDITION. `argmax_S det(L_S)` — a DPP over FEATURES with kernel `L = C` (the
feature correlation matrix). `P(S) proportional to det(L_S)` is the volume spanned by the
selected features, so a set containing two collinear views has a near-singular submatrix
and vanishing probability. This is the only rule on the Step-224 menu whose object is a
DISTRIBUTION OVER SUBSETS rather than a per-feature score, which is what the Step-223
stability finding (the good set is not a stable object; many different subsets are good)
says the channel needs.

ATTRIBUTION — read this before citing it. `papers/One-Pass Algorithms for MAP Inference of
Nonsymmetric Determinantal Point Processes.pdf` (Reddy, Rossi, Song, Rao, Mai, Lipka, Wu,
Koh, Ahmed) is a STREAMING-ALGORITHMS paper: it studies one-pass MAP inference for
non-symmetric DPPs under sublinear memory, where data points arrive in arbitrary order.
Its contribution is the streaming/online algorithm, NOT the selection criterion, and our
setting (28 features, all in memory) has no streaming constraint at all. What we implement
here is the OFFLINE GREEDY MAP that the paper itself uses as its comparison point — the
standard log-det greedy, which long predates it. So:

    DO write "DPP MAP (greedy log-det)".
    DO NOT write "Reddy et al.'s method", and do not report a null here as a null on
    that paper. It is the CONDITION that is being tested, not their algorithm.

Our kernel is SYMMETRIC (`C = corr(V)`), so the non-symmetric machinery that is the
paper's subject does not even apply. A non-symmetric variant is deliberately not
attempted: NDPP kernels are learned from observed subsets, and we have none.

THE ALGORITHM. Greedy log-det maximisation, the standard approach: start empty, and
repeatedly add the feature `j` maximising `log det(C_{S+j})`, which by the Schur
complement equals the current `log det(C_S)` plus the log of `j`'s residual variance
after projecting out `S`. Equivalently: at each step add the feature MOST ORTHOGONAL to
everything already kept. That is exactly a pivoted Cholesky / greedy-volume step, and it
is computed here in O(p) per step by maintaining the residual variances incrementally
rather than by re-factorising.

    d_j <- 1 (diagonal of the correlation matrix)
    pick j* = argmax d_j ; append; then for every remaining j:
        c_j   <- (C[j, j*] - <prev cholesky rows>) / sqrt(d_j*)
        d_j   <- d_j - c_j^2                       (Schur complement update)

`d_j` is `j`'s variance unexplained by the selected set, and `sum_i log d_i` over the
chosen order is exactly `log det(C_S)`. Numerically this is the classic
pivoted-Cholesky-with-greedy-pivoting loop.

WHAT IT MEASURES, stated honestly in advance. This is a PURE DIVERSITY rule: it looks at
the feature covariance and nothing else. It has no notion of which views carry signal, so
it will happily fill the set with mutually-orthogonal noise. Step 223 already measured
that the good subsets are LESS internally correlated than random (cohesion -0.127 vs the
floor) — but also that the arm which matched that cohesion almost exactly finished LAST,
while the label-handed oracle ignored cohesion and won. So the pre-registered expectation
is that this arm identifies-but-does-not-convert, like every other anti-redundancy rule
in this channel. It is run anyway, and not pre-filtered, because that expectation is a
prediction and the point of the round is to test it.

VARIANTS
    dpp.k3 .. dpp.k8   greedy log-det stopped at a fixed size
    dpp                stop when the next feature's residual variance falls below
                       LOGDET_TOL — i.e. when the candidate is already almost fully
                       explained by the kept set, so adding it adds no volume. This is a
                       genuine data-driven stopping rule (no K), but it is OURS: greedy
                       log-det MAP is defined for a fixed cardinality.
    dpp.ridge          same, on `C + RIDGE*I`. A DPP kernel must be PSD; sample
                       correlation matrices from n ~ 100-600 rows are PSD but can be
                       near-singular, and the ridge is the standard regularisation. If
                       `dpp` and `dpp.ridge` disagree materially, the un-ridged run was
                       being driven by numerical noise in the tail eigenvalues.

Determinism: no randomness except the tie-break draw from the passed `rng`.
"""

import numpy as np

from . import register

K_GRID = (3, 4, 5, 6, 7, 8)
LOGDET_TOL = 0.05      # stop when the best remaining residual variance drops below this
RIDGE = 1e-2
MIN_SET = 3
_EPS = 1e-12

_EXPECTED = tuple(f"dpp.k{k}" for k in K_GRID) + ("dpp", "dpp.ridge")


def greedy_logdet(C, max_k, tol=None, tiebreak=None):
    """Greedy `argmax_S log det(C_S)` by pivoted Cholesky.

    Returns (cols in selection order, residual variance at each pick). `tol` stops early
    when the best available residual variance falls below it; `max_k` always caps."""
    C = np.asarray(C, dtype=float)
    p = C.shape[0]
    d = np.diag(C).astype(float).copy()          # residual variance, starts at the diagonal
    chol = np.zeros((int(max_k), p))             # rows of the partial Cholesky factor
    chosen, gains = [], []
    tb = np.zeros(p) if tiebreak is None else np.asarray(tiebreak, dtype=float)

    for step in range(int(max_k)):
        avail = [j for j in range(p) if j not in chosen]
        if not avail:
            break
        # argmax residual variance, ties broken by the caller's draw (never by index
        # order — that would silently prefer whatever the pool happens to list first)
        best = max(avail, key=lambda j: (d[j], tb[j]))
        if tol is not None and len(chosen) >= MIN_SET and d[best] < tol:
            break
        if d[best] <= _EPS:                      # everything left is fully explained
            break
        gains.append(float(d[best]))
        piv = np.sqrt(d[best])
        # Schur complement update: c_j = (C[j,best] - <previous rows>) / sqrt(d[best])
        col = (C[:, best] - chol[:step].T @ chol[:step, best]) / piv
        chol[step] = col
        d = np.maximum(d - col ** 2, 0.0)
        d[best] = -np.inf                        # never re-pick
        chosen.append(int(best))
    return chosen, gains


def _emit(variant, cols, diag):
    cols = np.array(sorted(int(c) for c in cols), dtype=np.int64)
    if len(cols) < MIN_SET:
        return {'variant': variant, 'cols': np.arange(diag['p'], dtype=np.int64),
                'fallback': True,
                'diag': {**diag, 'fallback_reason': f'selection < {MIN_SET}'}}
    return {'variant': variant, 'cols': cols, 'diag': diag}


@register('a9_dpp')
def a9_dpp(cell, rng, cache=None):
    p = cell.p
    try:
        V = np.asarray(cell.V, dtype=float)
        C = np.corrcoef(V.T)
        C = np.nan_to_num(np.atleast_2d(C), nan=0.0)
        np.fill_diagonal(C, 1.0)
        Cr = C + RIDGE * np.eye(p)
        tb = rng.random(p)

        out = []
        for k in K_GRID:
            if k >= p:
                out.append(_emit(f'dpp.k{k}', [], {'p': p, 'k': k,
                                                   'note': f'k={k} >= pool size {p}'}))
                continue
            cols, gains = greedy_logdet(C, k, tiebreak=tb)
            out.append(_emit(f'dpp.k{k}', cols, {
                'p': p, 'k': int(k), 'order': [int(c) for c in cols],
                'residual_variance': [round(g, 4) for g in gains],
                'logdet': round(float(np.sum(np.log(np.maximum(gains, _EPS)))), 4),
                'note': 'greedy log-det MAP; residual_variance[i] is the volume the '
                        'i-th pick added, i.e. its variance orthogonal to the rest'}))

        # -- dpp : data-driven stop (OUR rule; greedy MAP is defined at fixed k) -------
        cols, gains = greedy_logdet(C, p, tol=LOGDET_TOL, tiebreak=tb)
        out.append(_emit('dpp', cols, {
            'p': p, 'n_selected': len(cols), 'tol': LOGDET_TOL,
            'residual_variance': [round(g, 4) for g in gains],
            'note': 'stops when the best remaining view is already explained by the '
                    'kept set (residual variance < tol). SIZE RULE IS OURS.'}))

        # -- dpp.ridge : PSD-regularised kernel ---------------------------------------
        cols_r, gains_r = greedy_logdet(Cr, p, tol=LOGDET_TOL, tiebreak=tb)
        out.append(_emit('dpp.ridge', cols_r, {
            'p': p, 'n_selected': len(cols_r), 'ridge': RIDGE, 'tol': LOGDET_TOL,
            'jaccard_vs_dpp': round(
                len(set(cols) & set(cols_r)) / max(1, len(set(cols) | set(cols_r))), 3),
            'note': 'C + ridge*I. A large disagreement with `dpp` means the un-ridged '
                    'run was driven by numerical noise in the tail eigenvalues.'}))
        return out

    except Exception as e:
        return [{'variant': v, 'cols': np.arange(p, dtype=np.int64), 'fallback': True,
                 'diag': {'error': str(e)}} for v in _EXPECTED]


# ---------------------------------------------------------------------------
# smoke() — known-answer test on a planted redundancy world
# ---------------------------------------------------------------------------

def smoke():
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL

    # 3 independent latent directions, each duplicated 3x (9 correlated columns), plus 3
    # independent columns. A volume-maximising rule must take ONE member per duplicate
    # block — taking two collapses the determinant.
    rng = np.random.default_rng(20260805)
    n = 600
    cols, block_of = [], []
    for g in range(3):
        latent = rng.standard_normal(n)
        for _ in range(3):
            cols.append(zscore(latent + 0.05 * rng.standard_normal(n)))
            block_of.append(g)
    for g in range(3, 6):
        cols.append(zscore(rng.standard_normal(n)))
        block_of.append(g)
    V = np.column_stack(cols)
    p = V.shape[1]
    block_of = np.array(block_of)
    cell = UnlabeledCell(domain='smoke', cell_key='dpp', pool=list(CANONICAL_POOL[:p]),
                         pool_bits=np.arange(p, dtype=np.uint8), V=V,
                         anchor=zscore(V[:, 0]), anchor_name=CANONICAL_POOL[0],
                         rho=np.abs(np.corrcoef(V.T)))

    s1 = a9_dpp(cell, np.random.default_rng([0, 5]))
    s2 = a9_dpp(cell, np.random.default_rng([0, 5]))
    by1 = {s['variant']: s for s in s1}
    by2 = {s['variant']: s for s in s2}
    assert set(by1) == set(_EXPECTED), f"variant set changed: {sorted(by1)}"

    # (a) determinism
    for v in _EXPECTED:
        assert list(by1[v]['cols']) == list(by2[v]['cols']), f"{v}: not deterministic"

    # (b) at k=6 exactly one representative per distinct block — this is the defining
    # property of a determinant/volume criterion, and the whole reason the arm exists
    sel = [int(c) for c in by1['dpp.k6']['cols']]
    blocks = block_of[sel]
    assert len(set(blocks.tolist())) == 6, \
        f"(b) k=6 took two from one block: cols={sel} blocks={blocks.tolist()}"

    # (c) log det must be non-increasing in the marginal gain (submodularity of volume):
    # each successive pick can only add less orthogonal variance than the one before
    g = by1['dpp.k8']['diag']['residual_variance']
    assert all(g[i] >= g[i + 1] - 1e-9 for i in range(len(g) - 1)), \
        f"(c) residual variance not monotone non-increasing: {g}"

    # (d) the ridge must not change the answer materially on a well-conditioned world
    j = by1['dpp.ridge']['diag']['jaccard_vs_dpp']
    assert j >= 0.8, f"(d) ridge changed the selection too much (Jaccard {j})"

    print(f"    [note] a9 smoke: k6={sel} blocks={blocks.tolist()} "
          f"dpp_size={by1['dpp']['diag']['n_selected']} ridge_jaccard={j}")
