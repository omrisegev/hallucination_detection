"""Red-team C follow-up: sharper shuffles (conditioning on R0 count), sibling structure of the swap null, singleton bias, analytic null-A check, per-class view."""
import json
import numpy as np, pandas as pd
exec(open(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/rt_dr_C/rt_c_setup.py').read())

rng = np.random.default_rng(777003)
c0 = cnt['R0_frozen']; c2 = cnt['R2_allocate']
def shuffle(base, pool, keys, rng):
    c = base.copy(); idx_all = np.flatnonzero(pool)
    df = pd.DataFrame({'i': idx_all, **{k: v[idx_all] for k, v in keys.items()}})
    moved = 0
    for _, g in df.groupby(list(keys)).i:
        idx = g.to_numpy(); c[idx] = base[idx][rng.permutation(len(idx))]
    return c
out = {}
tests = {
    'b0_shuffle_R0counts_fold_len_nc': (c0, nc, {'fold': fold, 'len': ns}),
    'b2_shuffle_R2counts_fold_len_nc': (c2, nc, {'fold': fold, 'len': ns}),
    'b3_shuffle_R2counts_fold_len_R0count_nc': (c2, nc, {'fold': fold, 'len': ns, 'c0': c0}),
    'b3p_shuffle_R2counts_fold_len_R0count_allprm': (c2, prm, {'fold': fold, 'len': ns, 'c0': c0}),
}
for name, (base, pool, keys) in tests.items():
    vals, cs, mad = [], [], []
    for d in range(50):
        c = shuffle(base, pool, keys, rng); f = by_count(c); vals.append(parts(f)['prm']); cs.append(ctrl_share(f)); mad.append(np.abs(c - base)[pool].mean())
    vals = np.array(vals); ref = P['R2_allocate']['prm'] if base is c2 else P['R0_frozen']['prm']
    out[name] = dict(prm_mean=vals.mean(), prm_sd=vals.std(ddof=1), prm_max=vals.max(), ref=ref, ref_minus_mean=ref - vals.mean(), z=(ref - vals.mean()) / vals.std(ddof=1),
                     frac_ge_ref=float((vals >= ref).mean()), minus_R0=vals.mean() - P['R0_frozen']['prm'], mean_abs_count_change=float(np.mean(mad)), controls_share_flag=float(np.mean(cs)))
# how much of the R2 count variation is left inside (fold,len,R0count) cells
df = pd.DataFrame({'c2': c2[nc], 'c0': c0[nc], 'L': ns[nc], 'f': fold[nc], 'd': (c2 - c0)[nc]})
out['var_R2count_nc'] = float(df.c2.var()); out['var_R2count_within_fold_len'] = float((df.c2 - df.groupby(['f', 'L']).c2.transform('mean')).var())
out['var_R2count_within_fold_len_c0'] = float((df.c2 - df.groupby(['f', 'L', 'c0']).c2.transform('mean')).var())
out['frac_nc_in_singleton_fold_len_c0_cells'] = float((df.groupby(['f', 'L', 'c0']).c2.transform('size') < 2).mean())
# sibling structure: share of swap partners (fold,len cells) that share the source group
grp = ans.source_group.to_numpy(); ncA = np.flatnonzero(nc); key = pd.DataFrame({'i': ncA, 'len': ns[ncA], 'fold': fold[ncA]})
same = []; tot = 0
for _, g in key.groupby(['len', 'fold']).i:
    idx = g.to_numpy()
    if len(idx) < 2: continue
    perm = rng.permutation(len(idx)); src = idx[np.roll(perm, 1)]; dst = idx[perm]
    same += list(grp[src] == grp[dst])
out['swap_partner_same_source_group_share'] = float(np.mean(same))
out['noncontrol_answers_per_source_group_mean'] = float(pd.Series(grp[nc]).value_counts().mean())
# singleton cells in the swap null keep their own labels: their share of the observed R2-R0 gain
key['size'] = key.groupby(['len', 'fold']).i.transform('size'); single = key[key['size'] < 2].i.to_numpy()
mix = F['R0_frozen'].copy(); m = np.isin(aid, single); mix[m] = F['R2_allocate'][m]
out['observed_gain_from_36_singleton_answers'] = parts(mix)['prm'] - P['R0_frozen']['prm']
# analytic expectation of null A (plug-in expected confusion under within-answer permutation)
e = np.bincount(aid, weights=lab, minlength=n)
def expected_prm(cn):
    L = ns[nc]; k = cn[nc].astype(float); ee = e[nc]
    TN = (k * ee / L).sum(); FN = (k * (L - ee) / L).sum(); FP = ee.sum() - TN; TP = (L - ee).sum() - FN
    return 0.5 * (2 * TP / (2 * TP + FP + FN) + 2 * TN / (2 * TN + FN + FP))
out['analytic_nullA_R2_minus_R0'] = expected_prm(c2) - expected_prm(c0); out['analytic_nullA_R1_minus_R0'] = expected_prm(cnt['R1_global']) - expected_prm(c0)
# per class: count change, error share, contribution
cl = pd.DataFrame({'cls': cls[nc], 'dc': (c2 - c0)[nc], 'L': ns[nc], 'e': e[nc]})
cl['es'] = cl.e / cl.L; cl['dc_per_step'] = cl.dc / cl.L
out['per_class'] = cl.groupby('cls').agg(n=('dc', 'size'), mean_dc=('dc', 'mean'), mean_dc_per_step=('dc_per_step', 'mean'), err_share=('es', 'mean'), mean_len=('L', 'mean')).round(4).to_dict('index')
# controls: count change
out['controls_mean_count_change'] = float((c2 - c0)[control].mean()); out['controls_mean_len'] = float(ns[control].mean())
# PRMScore when only the NON-control count changes are applied, split by direction
for nm, sel in [('only_increases', (c2 > c0)), ('only_decreases', (c2 < c0))]:
    cc = c0.copy(); cc[sel] = c2[sel]; out[f'apply_{nm}'] = dict(answers=int((sel & nc).sum()), prm_minus_R0=parts(by_count(cc))['prm'] - P['R0_frozen']['prm'])
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(OUT / 'rt_c2_results.json', 'w'), indent=1, default=float)
