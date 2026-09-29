"""Group bootstrap (PB source groups, 1000 draws) of length-controlled AUROC differences."""
from load import *
from scipy.stats import rankdata
err = target >= 0; tl = X[:, names.index('trace_length')]; ltl = np.log(tl)
DETS = ['D1_upcr_full', 'D2_lsml_cont_good5', 'D4_equal_full', 'D5_epr', 'D3_lsml_full']
A = {d: DEC['A_' + d].astype(float) for d in DETS}
cellix = {c: np.flatnonzero(cells == c) for c in PBc}
dec = np.zeros(n, int); res = {d: np.zeros(n) for d in DETS}
for c, ix in cellix.items():
    dec[ix] = pd.qcut(tl[ix], 10, labels=False)
    Z = np.column_stack([np.ones(len(ix)), ltl[ix]])
    for d in DETS: res[d][ix] = A[d][ix] - Z @ np.linalg.lstsq(Z, A[d][ix], rcond=None)[0]


def wauc(s, y, w):
    """weighted AUROC with integer multiplicity weights w (bootstrap), ties half"""
    o = np.argsort(s, kind='stable'); s, y, w = s[o], y[o], w[o]
    # handle ties by grouping unique s
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * y, minlength=len(u)); wn = np.bincount(inv, weights=w * (~y), minlength=len(u))
    cn = np.cumsum(wn) - wn
    num = (wp * (cn + 0.5 * wn)).sum(); P = wp.sum(); N = wn.sum(); return num / (P * N), P * N


def strat(s, y, st, w):
    num = den = 0.0
    for v in np.unique(st):
        m = st == v
        a, p = wauc(s[m], y[m], w[m])
        if p > 0 and np.isfinite(a): num += a * p; den += p
    return num / den


ug, gi = np.unique(groups[pb], return_inverse=True); gmap = np.full(n, -1); gmap[np.flatnonzero(pb)] = gi
rng = np.random.default_rng(5); B = 1000
pairs = [('D1_upcr_full', 'D2_lsml_cont_good5'), ('D1_upcr_full', 'D5_epr'), ('D4_equal_full', 'D3_lsml_full'), ('D4_equal_full', 'D5_epr')]
out = {p: {'raw': [], 'decile': [], 'resid': []} for p in pairs}
for b in range(B + 1):
    wg = np.ones(len(ug)) if b == 0 else rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug))).astype(float)
    val = {}
    for d in DETS:
        r_ = []; s_ = []; q_ = []
        for c, ix in cellix.items():
            w = wg[gmap[ix]]; y = err[ix]
            r_.append(wauc(A[d][ix], y, w)[0]); s_.append(strat(A[d][ix], y, dec[ix], w)); q_.append(wauc(res[d][ix], y, w)[0])
        val[d] = (np.mean(r_), np.mean(s_), np.mean(q_))
    for p in pairs:
        for j, k in enumerate(('raw', 'decile', 'resid')): out[p][k].append(val[p[0]][j] - val[p[1]][j])
for p in pairs:
    print(p[0], '-', p[1])
    for k in ('raw', 'decile', 'resid'):
        v = np.array(out[p][k]); print(f'   {k:7s} point {v[0]:+.4f}  95% CI [{np.quantile(v[1:],.025):+.4f}, {np.quantile(v[1:],.975):+.4f}]')
