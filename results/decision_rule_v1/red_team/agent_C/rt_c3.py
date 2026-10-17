"""Red-team C, part 3: length-only increments ON TOP of R0 counts, class-level explanations, rate-matched versions of partial rules."""
import json
import numpy as np, pandas as pd
exec(open(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/rt_dr_C/rt_c_setup.py').read())
c0 = cnt['R0_frozen']; c2 = cnt['R2_allocate']; d = c2 - c0
out = {}
def res(c, name):
    c = np.clip(c, 0, ns); f = by_count(c); p = parts(f)
    out[name] = dict(prm=p['prm'], minus_R0=p['prm'] - P['R0_frozen']['prm'], minus_R2=p['prm'] - P['R2_allocate']['prm'], nc_rate=p['rate'], f1c=p['f1c'], f1e=p['f1e'], controls_share_flag=ctrl_share(f))
    return f
def cell_mean_increment(pool, keyarr):
    inc = np.zeros(n)
    for i in np.flatnonzero(prm):
        m = pool & (fold != fold[i]) & (keyarr == keyarr[i])
        inc[i] = d[m].mean() if m.any() else d[pool & (fold != fold[i])].mean()
    return inc
# (a5/a6) R0 counts + length-conditional mean increment learned on other folds (label-free with all PRMB; a6 uses control identity)
res(c0 + np.floor(cell_mean_increment(prm, ns) + 0.5).astype(int), 'a5_R0_plus_len_increment_allprm')
res(c0 + np.floor(cell_mean_increment(nc, ns) + 0.5).astype(int), 'a6_R0_plus_len_increment_nc')
# stochastic-rounding version to avoid rounding every small mean to zero
rng = np.random.default_rng(777004); inc = cell_mean_increment(prm, ns); v = []
for _ in range(50):
    fl = np.floor(inc); c = c0 + (fl + (rng.random(n) < inc - fl)).astype(int); v.append(parts(by_count(np.clip(c, 0, ns)))['prm'])
out['a5s_R0_plus_len_increment_stochastic_round'] = dict(prm_mean=float(np.mean(v)), minus_R0=float(np.mean(v) - P['R0_frozen']['prm']), minus_R2=float(np.mean(v) - P['R2_allocate']['prm']))
# class-level explanation (ORACLE: class is PRMBench metadata, not label-free)
cl_codes = pd.factorize(cls)[0]
res(c0 + np.floor(cell_mean_increment(nc, cl_codes) + 0.5).astype(int), 'k1_R0_plus_class_increment_ORACLE')
res(c0 + np.floor(cell_mean_increment(nc, cl_codes * 1000 + ns) + 0.5).astype(int), 'k2_R0_plus_class_len_increment_ORACLE')
# shuffle R2 increments within (fold, len, R0 count, class): keeps class-level allocation, destroys within-class answer specificity
vals = []
for _ in range(50):
    c = c2.copy(); ii = np.flatnonzero(nc); df = pd.DataFrame({'i': ii, 'f': fold[ii], 'L': ns[ii], 'c0': c0[ii], 'k': cl_codes[ii]})
    for _, g in df.groupby(['f', 'L', 'c0', 'k']).i:
        idx = g.to_numpy(); c[idx] = c2[idx][rng.permutation(len(idx))]
    vals.append(parts(by_count(c))['prm'])
vals = np.array(vals)
out['b5_shuffle_R2_within_fold_len_c0_class'] = dict(prm_mean=vals.mean(), prm_sd=vals.std(ddof=1), minus_R0=vals.mean() - P['R0_frozen']['prm'], R2_minus_mean=P['R2_allocate']['prm'] - vals.mean(),
                                                    z=(P['R2_allocate']['prm'] - vals.mean()) / vals.std(ddof=1))
# per-class contribution: apply R2 counts only in one class (others keep R0)
contrib = {}
for k in sorted(set(cls[nc])):
    c = c0.copy(); m = nc & (cls == k); c[m] = c2[m]; contrib[k] = parts(by_count(c))['prm'] - P['R0_frozen']['prm']
out['per_class_contribution_R2_minus_R0'] = contrib; out['sum_of_class_contributions'] = float(sum(contrib.values()))
# min(R0,R2) (only decreases) vs R0 at the same non-control rate
fmin = res(np.minimum(c0, c2), 'min_R0_R2_only_decreases')
target = out['min_R0_R2_only_decreases']['nc_rate']
def r0_at(q):
    f = np.zeros(S_, bool)
    for k in range(5):
        c = (k + 1) % 5; calm = prms & (fold[aid] == c); evm = fold[aid] == k; f[evm] = zS[evm] >= float(np.quantile(zS[calm], q))
    return f
zS = np.empty(S_)
for i in range(n):
    s = S[off[i]:off[i + 1]]; zS[off[i]:off[i + 1]] = (s - s.mean()) / max(s.std(), 1e-8)
lo, hi = 0.7, 0.95
for _ in range(50):
    mid = (lo + hi) / 2
    if parts(r0_at(mid))['rate'] > target: lo = mid
    else: hi = mid
pm = parts(r0_at((lo + hi) / 2)); out['R0_rate_matched_to_min'] = dict(q=(lo + hi) / 2, prm=pm['prm'], nc_rate=pm['rate'], min_minus_it=out['min_R0_R2_only_decreases']['prm'] - pm['prm'])
# best R0 over q (label-chosen oracle) for reference
best = max(((q, parts(r0_at(q))['prm']) for q in np.round(np.arange(0.80, 0.901, 0.0025), 4)), key=lambda t: t[1])
out['R0_best_q_ORACLE'] = dict(q=best[0], prm=best[1], R2_minus_it=P['R2_allocate']['prm'] - best[1])
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(OUT / 'rt_c3_results.json', 'w'), indent=1, default=float)
