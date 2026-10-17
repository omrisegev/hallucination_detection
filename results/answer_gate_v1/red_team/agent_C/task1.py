"""Task 1: null + length control for answer-level PB AUROC."""
from load import *
from scipy.stats import rankdata, spearmanr

DETS = ['D1_upcr_full', 'D2_lsml_cont_good5', 'D3_lsml_full', 'D4_equal_full', 'D5_epr', 'D6_length', 'D6b_length_anchored']
A = {d: DEC['A_' + d].astype(float) for d in DETS}


def auc(score, y):
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_auc(score, y, strata):
    """pairs compared only within strata; weighted by number of pairs"""
    num = 0.0; den = 0.0
    for s in np.unique(strata):
        m = strata == s; yy = y[m]
        if yy.sum() == 0 or yy.sum() == m.sum(): continue
        p = yy.sum() * (m.sum() - yy.sum()); num += auc(score[m], yy) * p; den += p
    return num / den, den


rng = np.random.default_rng(7)
err = target >= 0
tl = X[:, names.index('trace_length')]; ltl = np.log(tl); lns = np.log(ns)
out = {}
rows = []
cellidx = {c: np.flatnonzero(cells == c) for c in PBc}


def exact_bins(c_ix):
    """exact step count, counts with fewer than 10 answers merged upward into the top bin"""
    k = ns[c_ix].copy(); vc = pd.Series(k).value_counts()
    small = sorted(vc[vc < 10].index); big = [v for v in sorted(vc.index) if v not in small]
    cap = None
    # merge tail: find the smallest count from which all larger are small
    srt = sorted(vc.index)
    for v in srt:
        if all(vc[u] < 10 for u in srt if u >= v): cap = v; break
    if cap is not None: k = np.minimum(k, cap)
    k = np.maximum(k, min(srt))
    return k


def quint_bins(c_ix):
    q = np.quantile(ns[c_ix], [.2, .4, .6, .8]); return np.searchsorted(q, ns[c_ix], 'right')


for d in DETS + ['trace_length_raw', 'step_count']:
    per = {}; per_res = {}; per_res2 = {}; per_sq = {}; per_sx = {}
    for c, ix in cellidx.items():
        y = err[ix]
        if d == 'trace_length_raw': s = tl[ix]
        elif d == 'step_count': s = ns[ix].astype(float) + 1e-9 * rng.random(len(ix)) * 0  # ties handled by rankdata
        else: s = A[d][ix]
        per[c] = auc(s, y)
        # residualize on log trace length within cell (all answers, label-free)
        Z1 = np.column_stack([np.ones(len(ix)), ltl[ix]]); b = np.linalg.lstsq(Z1, s, rcond=None)[0]; per_res[c] = auc(s - Z1 @ b, y)
        Z2 = np.column_stack([np.ones(len(ix)), ltl[ix], lns[ix]]); b2 = np.linalg.lstsq(Z2, s, rcond=None)[0]; per_res2[c] = auc(s - Z2 @ b2, y)
        per_sx[c] = strat_auc(s, y, exact_bins(ix))[0]
        per_sq[c] = strat_auc(s, y, quint_bins(ix))[0]
    rows.append({'detector': d, 'macro8': np.mean(list(per.values())), 'resid_logTL': np.mean(list(per_res.values())),
                 'resid_logTL_logSteps': np.mean(list(per_res2.values())), 'within_exact_stepcount': np.mean(list(per_sx.values())),
                 'within_stepcount_quintile': np.mean(list(per_sq.values())), 'min_cell': min(per.values()), 'per_cell': per, 'per_cell_resid': per_res})
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(T.drop(columns=['per_cell', 'per_cell_resid']).round(4).to_string(index=False))
print('\nper-cell raw AUROC')
print(pd.DataFrame({r['detector']: r['per_cell'] for r in rows}).round(3).to_string())
print('\nper-cell AUROC after residualizing log(trace_length)')
print(pd.DataFrame({r['detector']: r['per_cell_resid'] for r in rows}).round(3).to_string())

# permutation nulls, 100 each
NP = 100
for scheme in ('within_cell', 'within_cell_x_exact_steps', 'within_cell_x_step_quintile'):
    res = {d: [] for d in DETS}
    strata = {}
    for c, ix in cellidx.items():
        strata[c] = np.zeros(len(ix), int) if scheme == 'within_cell' else (exact_bins(ix) if 'exact' in scheme else quint_bins(ix))
    for p in range(NP):
        yp = {}
        for c, ix in cellidx.items():
            y = err[ix].copy(); st = strata[c]
            for s_ in np.unique(st):
                m = np.flatnonzero(st == s_); y[m] = y[rng.permutation(m)]
            yp[c] = y
        for d in DETS:
            res[d].append(np.mean([auc(A[d][ix], yp[c]) for c, ix in cellidx.items()]))
    print(f'\nNULL {scheme}: mean [2.5%, 97.5%] max  vs observed')
    for d in DETS:
        v = np.array(res[d]); obs = T.set_index('detector').loc[d, 'macro8']
        print(f'  {d:24s} {v.mean():.4f} [{np.quantile(v,.025):.4f}, {np.quantile(v,.975):.4f}] max {v.max():.4f}  obs {obs:.4f}  p={(1+(v>=obs).sum())/(NP+1):.3f}')

# D1 - D2 and D4 - D3 after length residualization (point estimates)
Tt = T.set_index('detector')
for a, b in (('D1_upcr_full', 'D2_lsml_cont_good5'), ('D4_equal_full', 'D3_lsml_full'), ('D1_upcr_full', 'D5_epr'), ('D4_equal_full', 'D1_upcr_full')):
    print(f'{a} - {b}: raw {Tt.loc[a,"macro8"]-Tt.loc[b,"macro8"]:+.4f}  resid logTL {Tt.loc[a,"resid_logTL"]-Tt.loc[b,"resid_logTL"]:+.4f}  within exact steps {Tt.loc[a,"within_exact_stepcount"]-Tt.loc[b,"within_exact_stepcount"]:+.4f}')

# correlations of detectors with log trace length per cell (Spearman), and q4 vs q8 duplication
print('\nSpearman(detector, trace_length) per cell')
print(pd.DataFrame({d: {c: spearmanr(A[d][ix], tl[ix])[0] for c, ix in cellidx.items()} for d in DETS}).round(3).to_string())
print('\nq4 vs q8 same answers? trace_length equal:', {c[:-3]: bool(np.array_equal(np.sort(tl[cells == c[:-3] + '_q4']), np.sort(tl[cells == c[:-3] + '_q8']))) for c in PBc if c.endswith('q4')})
print('ids equal q4/q8:', {c[:-3]: bool(np.array_equal(np.sort(ans.id[cells == c[:-3] + '_q4'].str.replace('q4', '')), np.sort(ans.id[cells == c[:-3] + '_q8'].str.replace('q8', '')))) for c in PBc if c.endswith('q4')})
print(ans.id[cells == 'pb_gsm8k_q4'].head(3).tolist(), ans.id[cells == 'pb_gsm8k_q8'].head(3).tolist())
