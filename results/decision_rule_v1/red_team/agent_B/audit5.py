exec(open('load.py').read())
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception', 'multi_solutions']
wk = np.load('work.npz'); lab_meta = wk['lab_meta']; Gre = wk['Gre']
o = json.load(open('audit_out.json'))
print('max |official - independent| total:', max(abs(v['official'] - v['indep']) for v in o['C'].values()),
      ' per class:', max(abs(v['per_class'][c]['official'] - v['per_class'][c]['indep']) for v in o['C'].values() for c in CLASSES))
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
meta = {m['idx']: m for m in metaraw.values()}
aid = np.repeat(np.arange(n), ns)
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); noncontrol = prm & ~control
def prm_score(f, mask):
    sm = mask[aid]; v = ~f[sm]; g = ~lab_meta[sm]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))
base = prm_score(D['R0_frozen'], noncontrol); full = prm_score(D['R2_allocate'], noncontrol)
print('R2 - R0 pooled %.5f' % (full - base))
for c in CLASSES:
    f = D['R0_frozen'].copy(); m = (cls == c)[aid]; f[m] = D['R2_allocate'][m]
    g = D['R2_allocate'].copy(); g[m] = D['R0_frozen'][m]
    print('  %-22s R2 only in class: %+.5f | R2 everywhere except class: %+.5f' % (c, prm_score(f, noncontrol) - base, prm_score(g, noncontrol) - base))
two = np.isin(cls, ['circular', 'missing_condition'])[aid]
f = D['R0_frozen'].copy(); f[two] = D['R2_allocate'][two]; print('R2 only in circular+missing_condition: %+.5f' % (prm_score(f, noncontrol) - base))
g = D['R2_allocate'].copy(); g[two] = D['R0_frozen'][two]; print('R2 everywhere except circular+missing_condition: %+.5f' % (prm_score(g, noncontrol) - base))
# ms vs control in same source group: mean G
Gm = np.bincount(aid, weights=Gre, minlength=n) / ns
ctrl_by_g = {}
for i in np.flatnonzero(control): ctrl_by_g.setdefault(groups[i], []).append(Gm[i])
d = [Gm[i] - np.mean(ctrl_by_g[groups[i]]) for i in np.flatnonzero(prm & (cls == 'multi_solutions')) if groups[i] in ctrl_by_g]
print('multi_solutions minus same-group control mean answer-G: mean %.3f, share ms>control %.3f, n %d' % (np.mean(d), np.mean(np.array(d) > 0), len(d)))
print('mean answer-G: controls %.3f, multi_solutions %.3f, erroneous noncontrol %.3f' % (Gm[control].mean(), Gm[prm & (cls == 'multi_solutions')].mean(), Gm[noncontrol & (cls != 'multi_solutions')].mean()))
