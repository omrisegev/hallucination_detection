"""
A1 — Nadler/Kluger-lineage residual-guided subset selection (Step 186+).

Label-free objectives, all computable from the cell's unlabeled feature
matrix alone:

  * raw Eq-14 residual  — L-SML's own two-rank-one structural-fit residual
    (Jaffe-Fetaya-Nadler-Jiang-Kluger 2016), as stored per subset by the
    Step-153 exhaustive sweep.
  * relative residual   — raw residual / Σ_{i≠j} C_ij² over the subset's
    Pearson correlations. The smoke-gate three-world experiment (2026-07-17)
    showed the RAW residual is structure-agnostic (pure-noise subsets have
    tiny R_off, hence tiny raw residual); dividing by the off-diagonal energy
    turns it into "fraction of dependence structure the grouped rank-one
    model fails to explain" — the structure-seeking form.
  * U-PCR projection residual — upcr_proj_residual with n_components=1
    (auto mode absorbs block dependence; see fusion_utils docstring).

Variant families emitted per cell:
  exhaustive (need the label-free npz cache; h16 pool only)
    a1.minres_exh_s{4,5,6}, a1.minres_exh_adapt     (raw residual)
    a1.relres_exh_s{4,5,6}, a1.relres_exh_adapt     (relative residual)
  greedy (always available; the deployment path)
    a1.relres_greedy, a1.upcrres_greedy
  structural-model router (memo §2.4)
    a1.router@good5, a1.router@minres
  K-rule swaps (rank_tests; clustering-swap fold-in)
    a1.good5+K_ah, a1.good5+K_kn, a1.minres+K_ah

IMPORTANT: none of these objectives has a-priori synthetic support as an
AUROC surrogate (smoke-gate documented properties + the Step-153 ρ-filter
refutation). Whether low residual predicts high AUROC is decided by
scripts/selector_admissibility.py on real cells BEFORE these variants'
bench numbers are interpreted as anything but an empirical answer.
"""

import numpy as np

from ..fusion_utils import detect_dependent_groups, upcr_proj_residual
from ..rank_tests import ahn_horenstein_K, cov_eigvals, kritchman_nadler_K
from ..subset_sweep import GOOD_5
from . import register

ADAPT_SIZES = (3, 4, 5, 6, 7, 8)     # pre-registered adaptive-size range
FIXED_SIZES = (4, 5, 6)
MAX_GREEDY_SIZE = 8
N_SEED_TRIPLES = 200                 # greedy seeding sample


# ---------------------------------------------------------------------------
# label-free objective helpers
# ---------------------------------------------------------------------------

def _pearson(V):
    C = np.corrcoef(V.T)
    return np.nan_to_num(np.atleast_2d(C), nan=0.0)


def _offdiag_energy(C2, cols):
    sub = C2[np.ix_(cols, cols)]
    return float(sub.sum() - np.trace(sub))


def _eq14_residual(V, cols):
    """Raw Eq-14 residual at the residual-grid best K for V[:, cols]."""
    _, _, resid, _ = detect_dependent_groups([V[:, j] for j in cols])
    return float(resid)


def _lsml_rel_residual(V, cols=None):
    cols = np.arange(V.shape[1]) if cols is None else np.asarray(cols)
    resid = _eq14_residual(V, cols)
    C2 = _pearson(V) ** 2
    energy = _offdiag_energy(C2, cols)
    return resid / max(energy, 1e-12)


def _upcr_k1_residual(V, cols):
    res, _, _ = upcr_proj_residual(V[:, cols].T, auto_components=False,
                                   n_components=1)
    return float(res)


def _mask_bits_matrix(masks, pool_bits):
    """(n_masks, p) 0/1 matrix: entry [k, j] = pool column j in mask k."""
    shifts = np.asarray(pool_bits, dtype=np.uint64)
    return ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)


def _cols_from_mask(mask, pool_bits):
    bit_to_col = {int(b): j for j, b in enumerate(pool_bits)}
    cols = [bit_to_col[b] for b in range(int(mask).bit_length())
            if (int(mask) >> b) & 1]
    return np.asarray(sorted(cols), dtype=np.int64)


# ---------------------------------------------------------------------------
# greedy forward search (deployment path — no cache needed)
# ---------------------------------------------------------------------------

def _greedy_min(V, objective, rng, max_size=MAX_GREEDY_SIZE):
    """Greedy forward selection minimizing `objective(V, cols)`; seeded at
    the best of N_SEED_TRIPLES sampled triples; returns (cols, per-size path)."""
    p = V.shape[1]
    n_tri = min(N_SEED_TRIPLES, p * (p - 1) * (p - 2) // 6)
    seen = set()
    best_tri, best_val = None, np.inf
    while len(seen) < n_tri:
        tri = tuple(sorted(rng.choice(p, size=3, replace=False).tolist()))
        if tri in seen:
            continue
        seen.add(tri)
        v = objective(V, np.asarray(tri))
        if v < best_val:
            best_val, best_tri = v, tri
    cols = list(best_tri)
    path = {3: (list(cols), best_val)}
    while len(cols) < min(max_size, p):
        cand_best, cand_val = None, np.inf
        for j in range(p):
            if j in cols:
                continue
            v = objective(V, np.asarray(sorted(cols + [j])))
            if v < cand_val:
                cand_val, cand_best = v, j
        if cand_best is None:
            break
        cols.append(cand_best)
        path[len(cols)] = (sorted(cols), cand_val)
    best_size = min(path, key=lambda s: path[s][1])
    return np.asarray(path[best_size][0], dtype=np.int64), {
        int(s): {'cols': v[0], 'objective': float(v[1])} for s, v in path.items()}


# ---------------------------------------------------------------------------
# the registered selector family
# ---------------------------------------------------------------------------

@register('a1_residual')
def a1_residual(cell, rng, cache=None):
    V = cell.V
    p = V.shape[1]
    out = []
    C2 = _pearson(V) ** 2

    # -- exhaustive variants from the label-free npz cache ------------------
    if cache is not None:
        masks = cache['mask']
        sizes = cache['size'].astype(int)
        resid = cache['residual'].astype(float)
        finite = np.isfinite(resid)
        B = _mask_bits_matrix(masks, cache['pool_bits'])
        energy = ((B @ C2) * B).sum(axis=1) - B.sum(axis=1)  # Σ C² off-diag (diag C²=1)
        rel = resid / np.maximum(energy, 1e-12)

        def emit_exh(prefix, values):
            for s in FIXED_SIZES:
                m = finite & (sizes == s)
                if not m.any():
                    continue
                k = np.flatnonzero(m)[np.argmin(values[m])]
                out.append({'variant': f'{prefix}_s{s}',
                            'cols': _cols_from_mask(masks[k], cache['pool_bits']),
                            'diag': {'objective': float(values[k])}})
            m = finite & np.isin(sizes, ADAPT_SIZES)
            if m.any():
                if prefix == 'a1.minres_exh':      # raw: normalize per pair count
                    vals = values[m] / (sizes[m] * (sizes[m] - 1))
                else:                              # relative: dimensionless
                    vals = values[m]
                k = np.flatnonzero(m)[np.argmin(vals)]
                out.append({'variant': f'{prefix}_adapt',
                            'cols': _cols_from_mask(masks[k], cache['pool_bits']),
                            'diag': {'objective': float(values[k]),
                                     'size': int(sizes[k])}})

        emit_exh('a1.minres_exh', resid)
        emit_exh('a1.relres_exh', rel)

    # -- greedy variants (deployment path, always available) ----------------
    rel_obj = lambda V_, c: _eq14_residual(V_, c) / max(_offdiag_energy(C2, c), 1e-12)
    greedy_cols, rel_path = _greedy_min(V, rel_obj, rng)
    out.append({'variant': 'a1.relres_greedy', 'cols': greedy_cols,
                'diag': {'path': rel_path}})

    upcr_cols, upcr_path = _greedy_min(V, lambda V_, c: _upcr_k1_residual(V_, c), rng)
    out.append({'variant': 'a1.upcrres_greedy', 'cols': upcr_cols,
                'diag': {'path': upcr_path}})

    # -- structural-model router (full-pool residual comparison) ------------
    lsml_rel_full = _lsml_rel_residual(V)
    upcr_rel_full = _upcr_k1_residual(V, np.arange(p))
    route = 'lsml' if lsml_rel_full <= upcr_rel_full else 'upcr'
    router_diag = {'lsml_rel_residual': lsml_rel_full,
                   'upcr_k1_residual': upcr_rel_full, 'route': route}

    g5_cols = np.asarray([cell.pool.index(f) for f in GOOD_5 if f in cell.pool],
                         dtype=np.int64)
    g5_fallback = len(g5_cols) < 3
    if g5_fallback:
        g5_cols = np.arange(min(5, p))
    out.append({'variant': 'a1.router@good5', 'cols': g5_cols, 'fusion': route,
                'fallback': g5_fallback, 'diag': router_diag})
    out.append({'variant': 'a1.router@minres', 'cols': greedy_cols,
                'fusion': route, 'diag': router_diag})

    # -- K-rule swaps on fixed subsets (rank_tests) --------------------------
    def k_rule(cols, fn, **kw):
        lam = cov_eigvals(V[:, cols])
        k_hat = fn(lam, max_K=max(len(cols) - 1, 1), **kw)
        return max(int(k_hat), 1)      # K̂ ∈ {0,1} → 1 = single group = flat SML

    out.append({'variant': 'a1.good5+K_ah', 'cols': g5_cols,
                'K': k_rule(g5_cols, ahn_horenstein_K, n=cell.n),
                'fallback': g5_fallback,
                'diag': {'rule': 'ahn_horenstein'}})
    out.append({'variant': 'a1.good5+K_kn', 'cols': g5_cols,
                'K': k_rule(g5_cols, kritchman_nadler_K, n=cell.n),
                'fallback': g5_fallback,
                'diag': {'rule': 'kritchman_nadler', 'alpha': 0.01}})
    out.append({'variant': 'a1.minres+K_ah', 'cols': greedy_cols,
                'K': k_rule(greedy_cols, ahn_horenstein_K, n=cell.n),
                'diag': {'rule': 'ahn_horenstein'}})
    return out


# ---------------------------------------------------------------------------
# smoke() — mechanics + determinism (auto-discovered by smoke_selectors.py).
# Selection QUALITY is deliberately not asserted here: whether any residual
# objective tracks AUROC is exactly what the admissibility analysis decides
# on real cells (see module docstring).
# ---------------------------------------------------------------------------

def smoke():
    import sys
    sys.path.insert(0, __file__.rsplit('spectral_utils', 1)[0] + 'scripts')
    from smoke_selectors import _tiny_ctx
    from ..selector_bench import UnlabeledCell, selector_cache_from_table

    ctx = _tiny_ctx()
    cell = UnlabeledCell.from_context(ctx)

    sels1 = a1_residual(cell, np.random.default_rng([0, 123]), cache=None)
    sels2 = a1_residual(cell, np.random.default_rng([0, 123]), cache=None)
    names = [s['variant'] for s in sels1]
    assert names == [s['variant'] for s in sels2], "variant order not deterministic"
    for a, b in zip(sels1, sels2):
        assert list(a['cols']) == list(b['cols']), (
            f"{a['variant']}: cols not deterministic under fixed rng")

    expect_nocache = {'a1.relres_greedy', 'a1.upcrres_greedy', 'a1.router@good5',
                      'a1.router@minres', 'a1.good5+K_ah', 'a1.good5+K_kn',
                      'a1.minres+K_ah'}
    assert set(names) == expect_nocache, f"variant set changed: {sorted(names)}"
    for s in sels1:
        cols = np.asarray(s['cols'])
        assert len(cols) >= 3 and len(set(cols.tolist())) == len(cols)
        assert cols.min() >= 0 and cols.max() < cell.p
        if 'K' in s:
            assert 1 <= s['K'] <= len(cols)
        if 'fusion' in s:
            assert s['fusion'] in ('lsml', 'upcr')

    # cache path: fabricate a tiny label-free cache of all size-3..4 subsets
    import itertools
    from ..subset_sweep import mask_to_canonical
    masks, szs, resids = [], [], []
    rng = np.random.default_rng(9)
    for size in (3, 4):
        for comb in itertools.combinations(range(cell.p), size):
            local = 0
            for j in comb:
                local |= 1 << j
            masks.append(mask_to_canonical(local, ctx.pool_bits))
            szs.append(size)
            resids.append(float(rng.random()))
    cache = {'mask': np.asarray(masks, dtype=np.uint64),
             'size': np.asarray(szs, dtype=np.uint8),
             'residual': np.asarray(resids, dtype=np.float32),
             'K': np.full(len(masks), 2, dtype=np.uint8),
             'rho_mean': np.zeros(len(masks), dtype=np.float16),
             'rho_max': np.zeros(len(masks), dtype=np.float16),
             'rho_hi': np.zeros(len(masks), dtype=np.uint8),
             'pool_bits': ctx.pool_bits}
    sels3 = a1_residual(cell, np.random.default_rng([0, 123]), cache=cache)
    got = {s['variant'] for s in sels3}
    assert 'a1.minres_exh_s4' in got and 'a1.relres_exh_adapt' in got, got
    # exhaustive argmin must reproduce the planted minimum raw residual
    s4 = next(s for s in sels3 if s['variant'] == 'a1.minres_exh_s4')
    m4 = np.asarray(szs) == 4
    assert abs(s4['diag']['objective'] - float(np.min(np.asarray(resids)[m4]))) < 1e-7
    print(f"    [note] a1 smoke: {len(sels3)} variants with cache, "
          f"{len(sels1)} without")
