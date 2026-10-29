"""Per-count follow-up to check F (Omri, 2026-10-29): judge each format count ALONE, not the average of the 10.

What matters (Omri): is the count above chance, and are its ERRORS conditionally independent of our telemetry groups
and the PRM? 2-3 good counts are enough. Position removed with the Step 481 procedure (checked numerically against probe.py's own
residual_cell), then length removed from all counts except the two length counts, label-free orientation (sign of the correlation with the 16-channel average), all 16 cells.

Per count and cell:
  (a) within-answer AUC with a 1000-draw source-group bootstrap CI;
  (b) conditional |corr| given the label (error steps / correct steps, separately) of the position-cleaned answer-z count
      with each TELE16 group score (mean of its channels; groups_from_cov on the 16 cleaned channels, as in check F),
      with the PRM, and with every other count;
  (c) error-level dependence = the SML condition: GAP votes (+-1); conditional covariance of the count's vote with each
      group vote / the PRM vote inside each class (= covariance of their errors); Lemma-4 positivity of the pooled vote
      covariance products over the triplets {count, group A, group B};
  (d) whether groups_from_cov makes the count its own group when it alone is added to the 16 channels + PRM
      (continuous and votes).
Predeclared pass rule, per cell: AUC CI lower bound > 0.5, AND max conditional |corr| (both classes, vs every group and
the PRM) below the reference level = max conditional |corr| among the existing groups and the PRM themselves, AND max
conditional vote covariance below the same kind of reference on votes. Labels only split classes and score.

    python per_count.py      # -> P1_PER_COUNT.csv, P2_COUNT_PAIRS.csv, P3_PRMBENCH_TYPES.csv, P4_FUSION_ROWS.csv
"""
import itertools, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import format_features as FF  # noqa: E402

E, A, L = FF.E, FF.A, FF.L
S8 = L.S8
B = 1000


# ------------------------------------------------------------------ Step 481 position cleaning (shared, checked)
def step481_clean(Z, off, pool, const=None):
    """Step 481 (results/prm_view_selection_v1/position_probe/probe.py, residual_cell): answer-z columns, pooled least
    squares on a cubic in relative position (j+.5)/n over the pool rows, subtract, then answer-z again.
    const (steps x columns, optional): entries whose column is constant inside the answer. They are left out of the fit
    and stay 0; otherwise subtracting the cubic turns an all-zero answer into a pure position score (found in the first
    per-count run: rare counts such as '<>' showed AUC .38/.65 that was position only). With const all False this is
    exactly Step 481."""
    pos = FF.rel_pos(off); Bm = np.column_stack([np.ones_like(pos), pos, pos ** 2, pos ** 3])
    if const is None: const = np.zeros(Z.shape, bool)
    R = np.empty_like(Z)
    for k in range(Z.shape[1]):
        m = pool & ~const[:, k]
        beta = np.linalg.lstsq(Bm[m], Z[m, k], rcond=None)[0]
        R[:, k] = np.where(const[:, k], 0., Z[:, k] - Bm @ beta)
    return np.column_stack([E.answer_z(R[:, k], off) for k in range(R.shape[1])])


def const_in_answer(X, off):
    n = np.diff(off)
    sd = np.array([X[a:b].std(0) for a, b in zip(off[:-1], off[1:])])
    return np.repeat(sd < 1e-12, n, axis=0)


def parity_with_step481():
    """Run probe.py's own residual_cell (extracted unchanged from its source) on PRMBench / Qwen-7B and require the
    same cleaned channels from step481_clean."""
    import ast
    src = (HERE.parents[1] / 'prm_view_selection_v1/position_probe/probe.py')
    tree = ast.parse(src.read_text(encoding='utf8'))
    fn = [nd for nd in tree.body if isinstance(nd, ast.FunctionDef) and nd.name == 'residual_cell'][0]
    sys.path.insert(0, str(HERE.parents[1] / 'prm_view_selection_v1/stage1'))
    import stage1 as S
    ns = {'np': np, 'S': S, 'E': S.E}
    exec(compile(ast.Module([fn], []), str(src), 'exec'), ns)
    o = ns['residual_cell']('prmbench', 'qwen_prm_7b'); base = S.CellData('prmbench', 'qwen_prm_7b')
    X = base.Z * base.sign[None, :]
    mine = step481_clean(X[:, 1:], base.c.off, base.pool) * o.sign_res[None, 1:]
    err = float(np.max(np.abs(mine - o.Z[:, 1:])))
    assert err < 1e-8, f'position cleaning differs from Step 481 probe: {err}'
    return err


def cond_r(x, y, m):
    if x[m].std() < 1e-12 or y[m].std() < 1e-12: return float('nan')
    return abs(float(np.corrcoef(x[m], y[m])[0, 1]))


def cond_cov(x, y, m):
    return abs(float(np.cov(x[m], y[m])[0, 1]))


def auc_ci(c, s, keep, rng):
    wa = E.per_answer_auc(c.err, s, c.off, keep); ok = np.isfinite(wa)
    g = c.groups[ok]; v = wa[ok]; ug, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, v); cnts = np.bincount(inv)
    draws = rng.integers(0, len(ug), (B, len(ug)))
    bs = sums[draws].sum(1) / cnts[draws].sum(1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def run_cell(bench, prm):
    rng = np.random.default_rng(20261029)
    c, V, prm_risk = FF.load_cell(bench, prm)
    off = c.off; keep = FF.eval_keep(bench, c); y = c.err.astype(bool)
    n = np.diff(off); lens = np.repeat(n, n); pool = np.asarray(c.keep, bool) & (lens > 1)      # Step 481 pool
    Rw = FF.raw_counts(bench, c)                                                      # log1p counts
    Z0 = np.column_stack([E.answer_z(Rw[:, k], off) for k in range(Rw.shape[1])])     # before any cleaning
    cst = const_in_answer(Rw, off)
    Z1 = step481_clean(Z0, off, pool, cst)                                                 # position removed (Step 481)
    Z2 = Z1.copy()                                                                    # + length removed (not for the 2 length counts)
    for k in range(2, Z1.shape[1]):
        mk = pool & ~cst[:, k]
        beta = np.linalg.lstsq(np.column_stack([np.ones(mk.sum()), Z1[mk, 0]]), Z1[mk, k], rcond=None)[0]
        Z2[:, k] = E.answer_z(np.where(cst[:, k], 0., Z1[:, k] - beta[0] - beta[1] * Z1[:, 0]), off)
    Tm = np.column_stack([prm_risk, V])
    Mt = step481_clean(np.column_stack([E.answer_z(prm_risk, off), V]), off, pool, const_in_answer(Tm, off))     # PRM + 16 channels, same cleaning
    zp, Zc = Mt[:, 0], Mt[:, 1:]
    m16 = Zc.mean(1)
    sign = np.sign(np.nan_to_num([np.corrcoef(Z2[:, k], m16)[0, 1] if Z2[:, k].std() > 1e-12 else 0. for k in range(Z2.shape[1])]))
    sign[sign == 0] = 1                                                               # label-free orientation
    C = Z2 * sign; Zf = C
    M = np.column_stack([zp, Zc, Zf])
    G = np.where(FF.gap_votes(M, off) > 0, 1., -1.)
    gp, Gc, Gf = G[:, 0], G[:, 1:17], G[:, 17:]
    K16, g16 = A.groups_from_cov(np.cov(Zc[pool].T))
    zs = [Zc[:, g16 == g].mean(1) for g in range(K16)] + [zp]                     # partners: groups, then the PRM
    vs = [np.where(Gc[:, g16 == g].mean(1) >= 0, 1., -1.) for g in range(K16)] + [gp]   # group vote = majority (ties -> 1)
    names = [f'group{g}' for g in range(K16)] + ['PRM']
    classes = (('err', keep & y), ('ok', keep & ~y))
    # reference: the dependence the existing partners have between themselves
    ref_z = max(cond_r(zs[i], zs[j], m) for i, j in itertools.combinations(range(len(zs)), 2) for _, m in classes)
    ref_v = max(cond_cov(vs[i], vs[j], m) for i, j in itertools.combinations(range(len(vs)), 2) for _, m in classes)
    Pv = np.cov(np.column_stack(vs)[keep].T)                                     # pooled vote covariance of the partners
    rows, pairs, types = [], [], []
    for k, nm in enumerate(FF.COUNTS):
        if C[:, k].std() < 1e-12: continue
        auc, lo, hi = auc_ci(c, C[:, k], keep, rng)
        r = dict(benchmark=bench, prm=prm, count=nm, sign=int(sign[k]), auc_before_cleaning=FF.wauc(c, Z0[:, k] * sign[k], keep),
                 auc_after_position=FF.wauc(c, Z1[:, k] * sign[k], keep), auc=auc, auc_lo=lo, auc_hi=hi, K16=int(K16),
                 ref_z=ref_z, ref_vote_cov=ref_v)
        zr = {(p, cl): cond_r(Zf[:, k], zs[i], m) for i, p in enumerate(names) for cl, m in classes}
        vc = {(p, cl): cond_cov(Gf[:, k], vs[i], m) for i, p in enumerate(names) for cl, m in classes}
        r['max_z'] = max(zr.values()); r['max_z_partner'] = max(zr, key=zr.get)[0]
        r['max_z_groups'] = max(v for (p, _), v in zr.items() if p != 'PRM'); r['max_z_prm'] = max(v for (p, _), v in zr.items() if p == 'PRM')
        r['max_vote_cov'] = max(vc.values()); r['max_vote_cov_partner'] = max(vc, key=vc.get)[0]
        r['max_vote_cov_prm'] = max(v for (p, _), v in vc.items() if p == 'PRM')
        cv = np.array([np.cov(Gf[keep, k], v[keep])[0, 1] for v in vs])
        trip = [cv[i] * cv[j] * Pv[i, j] for i, j in itertools.combinations(range(len(vs)), 2)]
        r['lemma4_positive'] = f'{sum(t > 0 for t in trip)}/{len(trip)}'
        for kind, X in (('z', np.column_stack([zp, Zc, Zf[:, k]])), ('vote', np.column_stack([gp, Gc, Gf[:, k]]))):
            _, gx = A.groups_from_cov(np.cov(X[pool].T)); r[f'own_cluster_{kind}'] = bool((gx == gx[-1]).sum() == 1)
        r['passes'] = bool(lo > .5 and r['max_z'] < ref_z and r['max_vote_cov'] < ref_v)
        rows.append(r)
        for j in range(k + 1, len(FF.COUNTS)):
            if C[:, j].std() < 1e-12: continue
            pairs.append(dict(benchmark=bench, prm=prm, a=nm, b=FF.COUNTS[j],
                              z=max(cond_r(Zf[:, k], Zf[:, j], m) for _, m in classes),
                              vote_cov=max(cond_cov(Gf[:, k], Gf[:, j], m) for _, m in classes)))
        if bench == 'prmbench':
            cls_step = np.repeat(c.cls, n)
            for cl in sorted(set(c.cls) - {'correct'}):
                types.append(dict(prm=prm, count=nm, perturbation=cl, auc=FF.wauc(c, C[:, k], keep & (cls_step == cl))))
    # fusion rows: TELE16 L-SML with and without the counts that pass in this cell (on the cell's answer-z channels)
    fus = []
    good = [r['count'] for r in rows if r['passes']]
    base, _ = S8.fit(V, anchor=V.mean(1))
    fus.append(dict(benchmark=bench, prm=prm, arm='TELE16', counts='', within_auc=FF.wauc(c, base, keep)))
    if good:
        Xg = np.column_stack([V] + [C[:, FF.COUNTS.index(g)] for g in good])
        s, d = S8.fit(Xg, anchor=V.mean(1))
        fus.append(dict(benchmark=bench, prm=prm, arm='TELE16+passing counts', counts=' '.join(good),
                        within_auc=FF.wauc(c, s, keep), K=int(d['K'])))
    print(bench, prm, 'pass:', good, flush=True)
    return rows, pairs, types, fus


def main():
    print('Step 481 parity max abs diff', parity_with_step481(), flush=True)
    R, P, T, F = [], [], [], []
    for bench in FF.BENCHES:
        for prm in FF.PRMS:
            r, p, t, f = run_cell(bench, prm); R += r; P += p; T += t; F += f
    E.write_csv(HERE / 'P1_PER_COUNT.csv', R); E.write_csv(HERE / 'P2_COUNT_PAIRS.csv', P)
    E.write_csv(HERE / 'P3_PRMBENCH_TYPES.csv', T); E.write_csv(HERE / 'P4_FUSION_ROWS.csv', F)


if __name__ == '__main__':
    main()
