"""Task 1b: nonparametric length control (trace-length decile strata), per-fold AUROC, epr+length control, PRMBench ms check, within-answer rank precision."""
from load import *
from scipy.stats import rankdata

DETS = ['D1_upcr_full', 'D2_lsml_cont_good5', 'D3_lsml_full', 'D4_equal_full', 'D5_epr']
A = {d: DEC['A_' + d].astype(float) for d in DETS}
err = target >= 0; tl = X[:, names.index('trace_length')]; ltl = np.log(tl); epr = X[:, names.index('epr')]
cellix = {c: np.flatnonzero(cells == c) for c in PBc}


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(s, y, st):
    num = den = 0.0
    for v in np.unique(st):
        m = st == v; yy = y[m]
        if 0 < yy.sum() < m.sum(): p = yy.sum() * (m.sum() - yy.sum()); num += auc(s[m], yy) * p; den += p
    return num / den


# label-free epr + length control: equal average of within-cell z(epr) and z(log length)
def z(v): return (v - v.mean()) / v.std()
ctrl = np.zeros(n)
for c, ix in cellix.items(): ctrl[ix] = z(epr[ix]) + z(ltl[ix])
A['CTRL_epr_plus_loglen'] = ctrl
rng = np.random.default_rng(3)
rows = []
for d, s in A.items():
    r = {'detector': d}
    r['macro8'] = np.mean([auc(s[ix], err[ix]) for ix in cellix.values()])
    r['within_TLdecile'] = np.mean([strat_auc(s[ix], err[ix], pd.qcut(tl[ix], 10, labels=False, duplicates='drop')) for ix in cellix.values()])
    r['within_TL20'] = np.mean([strat_auc(s[ix], err[ix], pd.qcut(tl[ix], 20, labels=False, duplicates='drop')) for ix in cellix.values()])
    r['per_fold_mean'] = np.mean([np.mean([auc(s[ix][fold[ix] == k], err[ix][fold[ix] == k]) for k in range(5)]) for ix in cellix.values()])
    rows.append(r)
T = pd.DataFrame(rows).set_index('detector'); print(T.round(4).to_string())
# permutation null within cell x trace-length decile
print('\nnull: permute labels within cell x trace-length decile (100)')
st = {c: pd.qcut(tl[ix], 10, labels=False, duplicates='drop') for c, ix in cellix.items()}
res = {d: [] for d in A}
for p in range(100):
    yp = {}
    for c, ix in cellix.items():
        y = err[ix].copy()
        for v in np.unique(st[c]):
            m = np.flatnonzero(st[c] == v); y[m] = y[rng.permutation(m)]
        yp[c] = y
    for d in A: res[d].append(np.mean([auc(A[d][ix], yp[c]) for c, ix in cellix.items()]))
for d in A:
    v = np.array(res[d]); print(f'  {d:22s} null mean {v.mean():.4f} [{np.quantile(v,.025):.4f},{np.quantile(v,.975):.4f}]  obs {T.loc[d,"macro8"]:.4f}  obs-nullmean {T.loc[d,"macro8"]-v.mean():+.4f}')
print('\nD1-D2 within TL decile: %+.4f ; D1-D5: %+.4f ; D4-D1: %+.4f ; D1-CTRL: raw %+.4f' % (
    T.loc['D1_upcr_full', 'within_TLdecile'] - T.loc['D2_lsml_cont_good5', 'within_TLdecile'], T.loc['D1_upcr_full', 'within_TLdecile'] - T.loc['D5_epr', 'within_TLdecile'],
    T.loc['D4_equal_full', 'within_TLdecile'] - T.loc['D1_upcr_full', 'within_TLdecile'], T.loc['D1_upcr_full', 'macro8'] - T.loc['CTRL_epr_plus_loglen', 'macro8']))

# PRMBench fair comparison: erroneous vs multi_solutions (exact populations), and a length check
pos = err_nc; neg = ms
for d in ['D1_upcr_full', 'D2_lsml_cont_good5', 'D4_equal_full', 'D5_epr']:
    s = DEC['A_' + d].astype(float); m = pos | neg
    print(f'PRMBench err vs multi_solutions {d}: AUROC {auc(s[m], pos[m]):.4f}   within TL decile {strat_auc(s[m], pos[m], pd.qcut(tl[m], 10, labels=False)):.4f}')
m = pos | neg
print('  trace_length alone:', round(auc(tl[m], pos[m]), 4), ' steps alone:', round(auc(ns[m].astype(float), pos[m]), 4),
      ' median TL err/ms:', np.median(tl[pos]), np.median(tl[neg]), ' median steps err/ms:', np.median(ns[pos]), np.median(ns[neg]))

# within-answer rank precision on PRMBench non-control erroneous answers: P(error | rank r by S)
rk = np.empty(S_, int)
for i in range(n):
    a, b = off[i], off[i + 1]; rk[a:b] = np.argsort(np.argsort(-S[a:b], kind='stable'), kind='stable')
mstep = err_nc[aid]
print('\nPRMBench erroneous non-control: P(step is error | within-answer rank of S)')
print(pd.Series(labels[mstep]).groupby(np.minimum(rk[mstep], 10)).mean().round(3).to_string())
relr = np.empty(S_)
for i in range(n): a, b = off[i], off[i + 1]; relr[a:b] = rk[a:b] / max(b - a, 1)
print('by relative rank decile:'); print(pd.Series(labels[mstep]).groupby(np.floor(relr[mstep] * 10).astype(int)).mean().round(3).to_string())
