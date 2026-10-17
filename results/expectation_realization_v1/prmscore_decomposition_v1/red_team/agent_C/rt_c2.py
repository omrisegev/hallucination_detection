"""Red-team C part 2: AUC decomposition identity, group-weighted draw algorithm, q80 structural bounds, residual bootstrap."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import rankdata
sys.argv = ['x']
exec(open(Path(__file__).with_name('rt_c.py')).read().split("# ---------------- Null A")[0])   # reuse setup (population, scores, valid, conf)
out2 = {}
ARMS = ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'step_index']

# ---------------- 3a: pooled / within / cross AUC by an independent path
NCI = np.flatnonzero(nonc)
elig = np.array([lab[off[i]:off[i + 1]].any() and (~lab[off[i]:off[i + 1]]).any() for i in range(n)]) & nonc
def pooled_U_searchsorted(z, y):
    neg = np.sort(z[~y]); pos = z[y]
    lo = np.searchsorted(neg, pos, 'left'); hi = np.searchsorted(neg, pos, 'right')
    return float((lo + 0.5 * (hi - lo)).sum())
def within_U_brute(z):
    U = 0.0; pairs = 0
    for i in np.flatnonzero(elig):
        a, b = off[i:i + 2]; zz = z[a:b]; yy = lab[a:b]; zp, zn = zz[yy], zz[~yy]
        d = zp[:, None] - zn[None, :]; U += (d > 0).sum() + 0.5 * (d == 0).sum(); pairs += d.size
    return U, pairs
res = {}
for m in ARMS:
    z = scale[m][nc_step]; y = lab[nc_step]; P_, N_ = int(y.sum()), int((~y).sum())
    Up = pooled_U_searchsorted(z, y); Up_rank = rankdata(z)[y].sum() - P_ * (P_ + 1) / 2
    Uw, pw = within_U_brute(scale[m])
    pooled = Up / (P_ * N_); within = Uw / pw; cross = (Up - Uw) / (P_ * N_ - pw); w = pw / (P_ * N_)
    res[m] = {'P': P_, 'N': N_, 'within_pairs': int(pw), 'w_share_within': w, 'pooled': pooled, 'within_pair_weighted': within, 'cross': cross,
              'pooled_minus_cross': pooled - cross, 'identity_residual': pooled - (w * within + (1 - w) * cross),
              'U_rankdata_minus_U_searchsorted': float(Up_rank - Up)}
out2['auc_decomposition'] = res
# brute-force cross AUC on a random subset of 400 non-control answers (explicit cross pairs) vs the formula on that subset
rng = np.random.default_rng(11); sub = np.sort(rng.choice(NCI, 400, replace=False)); m = 'S_equal'
stp = np.concatenate([np.arange(off[i], off[i + 1]) for i in sub]); z = scale[m][stp]; y = lab[stp]; a_ = st_ans[stp]
D = z[y][:, None] - z[~y][None, :]; same = a_[y][:, None] == a_[~y][None, :]
Ucross_brute = ((D > 0) + 0.5 * (D == 0))[~same].sum(); cross_brute = Ucross_brute / (~same).sum()
Ptot = ((D > 0) + 0.5 * (D == 0)).sum(); Uw_sub = ((D > 0) + 0.5 * (D == 0))[same].sum()
cross_formula = (Ptot - Uw_sub) / (D.size - same.sum())
out2['cross_brute_subset'] = {'answers': 400, 'cross_brute': float(cross_brute), 'cross_formula': float(cross_formula), 'diff': float(cross_brute - cross_formula)}

# ---------------- 3b: script's group-weighted pooled AUC algorithm vs replication
grp = ans.source_group.to_numpy(); gmap = np.full(n, -1); gu, gi = np.unique(grp[prm], return_inverse=True); gmap[np.flatnonzero(prm)] = gi; G = len(gu)
st_group = gmap[st_ans]
def script_draw(z, y, gs, Wb):   # verbatim logic of lines 364-371
    u, inv = np.unique(z, return_inverse=True)
    Mp = sparse.csr_matrix((np.ones(int(y.sum())), (inv[y], gs[y])), shape=(len(u), G))
    Mn = sparse.csr_matrix((np.ones(int((~y).sum())), (inv[~y], gs[~y])), shape=(len(u), G))
    Pw = np.asarray(Mp @ Wb); Nw = np.asarray(Mn @ Wb); below = np.cumsum(Nw, 0) - Nw
    return (Pw * (below + 0.5 * Nw)).sum(0) / (Pw.sum(0) * Nw.sum(0))
m = 'S_equal'; z = scale[m][nc_step]; y = lab[nc_step]; gs = st_group[nc_step]
ones = script_draw(z, y, gs, np.ones((G, 1)))[0]
Wint = rng.multinomial(G, np.full(G, 1 / G), size=1).astype(float).T
wd = script_draw(z, y, gs, Wint)[0]
rep = np.repeat(np.arange(len(z)), Wint[gs, 0].astype(int)); zr, yr = z[rep], y[rep]
Pr = yr.sum(); Nr = len(yr) - Pr; auc_rep = (rankdata(zr)[yr].sum() - Pr * (Pr + 1) / 2) / (Pr * Nr)
out2['group_weighted_draw_check'] = {'W_ones': float(ones), 'pooled_direct': res[m]['pooled'], 'diff_ones': float(ones - res[m]['pooled']),
                                     'W_multinomial': float(wd), 'replicated_rankdata': float(auc_rep), 'diff_multinomial': float(wd - auc_rep)}
# the step's group ids: does gmap cover every non-control step? (gs >= 0)
out2['group_ids_nonnegative_on_nc'] = bool((gs >= 0).all())

# ---------------- 3c: q80 structural bounds and clean-answer flag shares
ms = nonc & (cls == 'multi_solutions')
has_err = np.array([lab[off[i]:off[i + 1]].any() for i in range(n)])
inert = nonc & ~has_err & ~ms
def flags_per_answer(v): return np.bincount(st_ans[prm_step & ~v], minlength=n)
q = {}
for m in ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'step_index']:
    t = np.array([tau[m][k] for k in range(5)]); tbar = float(t.mean()); fl = flags_per_answer(valid[m])
    zmax = np.array([np.nanmax(scale[m][off[i]:off[i + 1]]) if prm[i] else np.nan for i in range(n)])
    rec = {'tau_range': [float(t.min()), float(t.max())], 'n_forced_ge1_flag_upto': float(1 + 1 / t.max() ** 2) if t.max() > 0 else None,
           'max_flag_share_bound_1_over_1_plus_tau2': float(1 / (1 + t.min() ** 2))}
    for nm, msk in (('controls', control), ('multi_solutions', ms), ('inert', inert), ('erroneous_nc', nonc & has_err)):
        rec[nm] = {'answers': int(msk.sum()), 'share_ge1_flag': float((fl[msk] > 0).mean()), 'mean_flag_share': float((fl[msk] / ns[msk]).mean()),
                   'share_answers_with_n_le_forced': float((ns[msk] <= 1 + 1 / t.max() ** 2).mean()) if t.max() > 0 else None,
                   'share_n1': float((ns[msk] == 1).mean()), 'share_n2': float((ns[msk] == 2).mean())}
    # flag-share bound check (empirical max share must be <= 1/(1+tau^2) up to floor)
    shares = fl[prm] / ns[prm]; tau_ans = t[fold[prm]]
    rec['max_empirical_flag_share'] = float(shares.max()); rec['any_answer_violates_share_bound'] = bool((fl[prm] > np.floor(ns[prm] / (1 + tau_ans ** 2)) + 1e-9).any())
    rec['any_answer_max_z_above_samuelson'] = bool((zmax[prm] > np.sqrt(ns[prm] - 1) + 1e-9).any())
    q[m] = rec
# iid baseline: probability that an answer of length n has >=1 step with z >= tau, for Gaussian / exponential / uniform iid draws
lens_c = ns[control]; rs = np.random.default_rng(5); base = {}
for dist in ('gauss', 'expo', 'unif'):
    for tv in (0.85, 0.28):
        hits = []
        for L in lens_c:
            if L == 1: hits.append(0.0); continue
            x = {'gauss': rs.standard_normal((400, L)), 'expo': rs.exponential(size=(400, L)), 'unif': rs.random((400, L))}[dist]
            zz = (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-8)
            hits.append(float((zz.max(1) >= tv).mean()))
        base[f'{dist}_tau{tv}'] = float(np.mean(hits))
out2['iid_baseline_share_ge1_flag_controls_len_distribution'] = base
out2['control_len_quantiles'] = np.quantile(lens_c, [0, .1, .25, .5, .75, .9, 1]).tolist()
out2['q80_structure'] = q
# per-n table for S_equal on controls
tab = []
t = np.array([tau['S_equal'][k] for k in range(5)]); flS = flags_per_answer(valid['S_equal'])
for L in sorted(set(ns[control])):
    msk = control & (ns == L)
    tab.append({'n': int(L), 'answers': int(msk.sum()), 'share_ge1': float((flS[msk] > 0).mean()), 'mean_flags': float(flS[msk].mean()),
                'max_flags_bound': int(np.floor(L / (1 + t.max() ** 2))) if L > 1 else 0})
out2['S_equal_controls_by_n'] = tab

# ---------------- residual (answer-specific) component: obs diff minus whole-answer-swap null, paired group bootstrap
exec('#' + open(Path(__file__).with_name('rt_c.py')).read().split("# ---------------- Null B")[1].split("nullB = {k")[0].replace('def swap_labels', 'def swap_labels'))
NC = np.flatnonzero(nonc); gnc = gmap[NC]
def per_ans_conf(v, y):
    g = ~y; o = np.zeros((n, 4))
    for c, msk in enumerate([(v & g), (v & ~g), (~v & ~g), (~v & g)]): o[:, c] = np.bincount(st_ans[msk & nc_step], minlength=n)
    return o[NC]
def agg(c): A = np.zeros((G, 4)); np.add.at(A, gnc, c); return A
def ps(C):
    tp, fp, tn, fn = np.moveaxis(C, -1, 0); p, r = tp / (tp + fp), tp / (tp + fn); p2, r2 = tn / (tn + fn), tn / (tn + fp)
    return 0.5 * (2 * p * r / (p + r) + 2 * p2 * r2 / (p2 + r2))
arms3 = ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80']
Aobs = {m: agg(per_ans_conf(valid[m], lab)) for m in arms3}
swaps = [swap_labels(s)[0] for s in range(20)]
Asw = {m: np.stack([agg(per_ans_conf(valid[m], yp)) for yp in swaps]) for m in arms3}   # (20, G, 4)
# within-answer permutation null per group as well
permsA = []
for seed in range(20):
    r_ = np.random.default_rng(1000 + seed); order = np.lexsort((r_.random(S), st_ans)); permsA.append(lab[order])
Apa = {m: np.stack([agg(per_ans_conf(valid[m], yp)) for yp in permsA]) for m in arms3}
Wb = np.random.default_rng(77).multinomial(G, np.full(G, 1 / G), size=2000).astype(float)
def comp(a, b, Anull):
    obs_d = ps(Wb @ Aobs[a]) - ps(Wb @ Aobs[b])
    null_d = np.mean([ps(Wb @ Anull[a][k]) - ps(Wb @ Anull[b][k]) for k in range(20)], 0)
    r = obs_d - null_d
    pt = (ps(Aobs[a].sum(0)) - ps(Aobs[b].sum(0))) - np.mean([ps(Anull[a][k].sum(0)) - ps(Anull[b][k].sum(0)) for k in range(20)])
    return {'point': float(pt), 'ci95': [float(np.quantile(r, .025)), float(np.quantile(r, .975))], 'sd': float(r.std())}
out2['residual_vs_whole_answer_swap'] = {f'{a}-{b}': comp(a, b, Asw) for a, b in [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80')]}
out2['residual_vs_within_answer_perm'] = {f'{a}-{b}': comp(a, b, Apa) for a, b in [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80')]}
out2['per_method_alignment_excess_over_swap'] = {m: float(ps(Aobs[m].sum(0)) - np.mean([ps(Asw[m][k].sum(0)) for k in range(20)])) for m in arms3}
out2['per_method_alignment_excess_over_perm'] = {m: float(ps(Aobs[m].sum(0)) - np.mean([ps(Apa[m][k].sum(0)) for k in range(20)])) for m in arms3}
json.dump(out2, open(OUTD / 'rt_c_part2.json', 'w'), indent=1, default=float)
print(json.dumps(out2, indent=1, default=float))
