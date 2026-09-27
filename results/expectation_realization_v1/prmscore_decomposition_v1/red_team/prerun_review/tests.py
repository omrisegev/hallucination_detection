import importlib.util, sys, numpy as np, pandas as pd, warnings
from scipy import sparse
from scipy.stats import rankdata
W = r'C:/Users/omris/TAU/hallucination_detection/.worktrees/ssl-pseudolabel-residual-v1'
spec = importlib.util.spec_from_file_location('dec', W + '/scripts/experiments/er_prmscore_decomposition.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)   # import only; main() not called
from spectral_utils.prmbench import _prf, prmbench_evaluate, eval_on_hallucination_step
rng = np.random.default_rng(0)

# ---- T1 count_metrics vs official _prf on random confusions
dev = 0.0
for _ in range(5000):
    c = rng.integers(0, 50, 4); c[rng.random(4) < .1] = 0
    m = D.count_metrics(c); o = _prf(*c)
    for mk, ok in [('f1_correct','f1'),('f1_error','negative_f1'),('recall_correct','recall'),('recall_error','negative_recall'),('precision_correct','precision'),('precision_error','negative_precision')]:
        if np.isfinite(m[mk]): dev = max(dev, abs(m[mk]-o[ok]))
print('T1 count_metrics vs _prf max |diff| where ours finite:', dev)
# degenerate: all-correct class (multi_solutions): tn=fp=0
for c in ([10,0,0,3],[10,0,0,0],[0,0,0,5]):
    m = D.count_metrics(c); o = _prf(*c)
    print('   degenerate', c, 'ours f1/f1e/prm', m['f1_correct'], m['f1_error'], m['prmscore'], '| official f1/neg_f1', o['f1'], o['negative_f1'])

# ---- T2 cross-answer AUC formula vs brute force (ties, answers with only-error and only-correct steps)
def script_cross(z_ans, y_ans):
    z = np.concatenate(z_ans); y = np.concatenate(y_ans)
    rk = rankdata(z); P_ = int(y.sum()); N_ = len(y) - P_
    U_pool = rk[y].sum() - P_*(P_+1)/2; U_w = 0.; pw = 0.
    for zz, yy in zip(z_ans, y_ans):
        n1 = int(yy.sum()); n0 = len(yy)-n1
        if n1 == 0 or n0 == 0: continue          # script loops over eligible only
        U_w += rankdata(zz)[yy].sum() - n1*(n1+1)/2; pw += n1*n0
    return U_pool/(P_*N_), (U_pool-U_w)/(P_*N_-pw)
def brute(z_ans, y_ans):
    num = den = numa = dena = 0.
    for a,(za,ya) in enumerate(zip(z_ans,y_ans)):
        for b,(zb,yb) in enumerate(zip(z_ans,y_ans)):
            for i in np.flatnonzero(ya):
                for j in np.flatnonzero(~yb):
                    s = 1.0 if za[i] > zb[j] else 0.5 if za[i] == zb[j] else 0.
                    numa += s; dena += 1
                    if a != b: num += s; den += 1
    return numa/dena, num/den
md = 0.
for t in range(300):
    k = rng.integers(2, 7); za=[]; ya=[]
    for a in range(k):
        L = rng.integers(1, 7); za.append(np.round(rng.normal(size=L), 1)); yy = rng.random(L) < .35
        if a == 0: yy[:] = True            # only-error answer
        if a == 1: yy[:] = False           # only-correct answer
        ya.append(yy)
    if sum(y.sum() for y in ya)==0 or sum((~y).sum() for y in ya)==0: continue
    s = script_cross(za, ya); b = brute(za, ya); md = max(md, abs(s[0]-b[0]), abs(s[1]-b[1]))
print('T2 pooled+cross-answer AUC vs brute force (ties, only-error/only-correct answers) max |diff|:', md)

# ---- T3 group-weighted pooled AUC (script lines 346-353) vs replication with integer weights
md = 0.
for t in range(200):
    nst = rng.integers(20, 200); G = rng.integers(3, 12)
    z = np.round(rng.normal(size=nst), 1); y = rng.random(nst) < .3; gs = rng.integers(0, G, nst)
    if y.all() or (~y).all(): continue
    Wb = rng.multinomial(G, np.full(G, 1/G), size=5).astype(float).T   # (G, draws)
    u, inv = np.unique(z, return_inverse=True)
    Mp = sparse.csr_matrix((np.ones(int(y.sum())), (inv[y], gs[y])), shape=(len(u), G))
    Mn = sparse.csr_matrix((np.ones(int((~y).sum())), (inv[~y], gs[~y])), shape=(len(u), G))
    Pw = np.asarray(Mp @ Wb); Nw = np.asarray(Mn @ Wb); below = np.cumsum(Nw, 0) - Nw
    dr = (Pw*(below + .5*Nw)).sum(0) / (Pw.sum(0)*Nw.sum(0))
    for d in range(Wb.shape[1]):
        rep = np.repeat(np.arange(nst), Wb[gs, d].astype(int)); zz = z[rep]; yy = y[rep]
        if yy.all() or (~yy).all(): continue
        n1 = yy.sum(); n0 = len(yy)-n1; ref = (rankdata(zz)[yy].sum()-n1*(n1+1)/2)/(n1*n0)
        md = max(md, abs(ref - dr[d]))
print('T3 weighted pooled AUC vs replicated-data AUC max |diff|:', md)

# ---- T4 bootstrap with all-ones weights reproduces the point; ratio of sums
cnt = rng.integers(0, 30, (50, 3, 4)).astype(float)
e1 = D.count_metrics(np.einsum('bg,gkc->bkc', np.ones((1,50)), cnt))['prmscore'][0]; e0 = D.count_metrics(cnt.sum(0))['prmscore']
print('T4 unit-weight draw == point:', np.allclose(e1, e0, equal_nan=True))

# ---- T5 script q80_rule/valid_from vs runner cal_tau/zt on a toy (fold rule mirror)
n = 60; ns = rng.integers(1, 9, n); off = np.r_[0, np.cumsum(ns)]; S = off[-1]; fold = rng.integers(0, 5, n); prm = rng.random(n) < .8
s = rng.normal(size=S)
def runner_tau(cal): return float(np.quantile(np.concatenate([D.zt(s[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
z = np.full(S, np.nan)
for i in np.flatnonzero(prm): z[off[i]:off[i+1]] = D.zt(s[off[i]:off[i+1]])
step_fold = np.repeat(fold, ns); prm_step = np.repeat(prm, ns)
tau_s = {k: float(np.quantile(z[prm_step & (step_fold == (k+1) % 5)], .8)) for k in range(5)}
print('T5 fold rule max |tau diff|:', max(abs(tau_s[k] - runner_tau((k+1) % 5)) for k in range(5)))
v = np.zeros(S, bool)
for k in range(5): sel = prm_step & (step_fold == k); v[sel] = z[sel] < tau_s[k]
vr = np.concatenate([D.zt(s[off[i]:off[i+1]]) < tau_s[fold[i]] for i in np.flatnonzero(prm)])
print('T5 valid flags identical to runner rule:', np.array_equal(v[prm_step], vr))

# ---- T6 frozen METRICS filter used at line 204 (read-only on the frozen CSV)
M = pd.read_csv(W + '/results/expectation_realization_v1/run_20260927_stage_b/METRICS.csv')
Mf = M[(M.benchmark == 'prm') & (M.stratum == 'all')].set_index('metric').estimate
print('T6 rows under metric=within_auc after the line-204 filter:', int((Mf.index == 'within_auc').sum()))
try: float(Mf['within_auc']); print('   float() OK')
except Exception as ex: print('   float() raises:', type(ex).__name__, str(ex)[:80])

# ---- T7 official by-class negative_f1 for an all-correct class, and the nanmax gate behaviour
meta = [{'idx': 'multi_solutions_a', 'error_steps': [], 'classification': 'multi_solutions'},
        {'idx': 'confidence_b', 'error_steps': [2], 'classification': 'confidence'}]
res = prmbench_evaluate([{'idx': 'multi_solutions_a', 'labels': [1, 0, 1]}, {'idx': 'confidence_b', 'labels': [1, 0]}], meta)
print('T7 official MS negative_f1 with one flagged step:', res['by_classification']['negative_f1']['multi_solutions'],
      '(!= -1 -> enters cls_dev as |NaN - 0|)')
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); print('   np.nanmax([nan, nan]) =', np.nanmax([np.nan, np.nan]), '; nan > 1e-12 ->', np.nan > 1e-12)
