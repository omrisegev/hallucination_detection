exec(open('load.py').read())
from collections import Counter
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception', 'multi_solutions']
wk = np.load('work.npz'); lab_meta = wk['lab_meta']; Seq = wk['Seq']
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
cells = ans.cell.to_numpy(); target = ans.target.to_numpy()
meta = {m['idx']: m for m in metaraw.values()}
aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); noncontrol = prm & ~control
has_err = np.array([lab_meta[off[i]:off[i + 1]].any() for i in range(n)]) & prm
ms = noncontrol & (cls == 'multi_solutions')
nf = {r: np.bincount(aid, weights=D[r].astype(float), minlength=n).astype(int) for r in RULES}
DSE = json.loads((W / 'DS_ESTIMATES.json').read_text(encoding='utf8')); chans = DSE['channels']
out = {}
# ---------- 4 DS truth on PRMBench-only fit steps
print('== DS estimates vs truth (fold-wise); truth recomputed from rebuilt marks')
rows = []
for k in range(5):
    fk = DSE['folds'][str(k)]; mk = wk[f'marks{k}']; fitm = ~np.isin(step_fold, [k, (k + 1) % 5])
    fp_ = fitm & prm_step; fpb = fitm & ~prm_step; y = lab_meta
    # replicate stored (mixed) truth using NPZ labels (PB all True)
    ymix = labels[fitm]
    psi_mix = mk[fitm][ymix].mean(0); eta_mix = (~mk[fitm][~ymix]).mean(0)
    psi_prm = mk[fp_][y[fp_]].mean(0); eta_prm = (~mk[fp_][~y[fp_]]).mean(0)
    rows.append(dict(fold=k, ds_prev=fk['ds_prevalence'], stored_true_prev_mixed=fk['true_prevalence_fit'], stored_true_prev_prm=fk['true_prevalence_fit_prm'],
                     pb_share_of_fit_steps=float(fpb.sum() / fitm.sum()), posterior_mean_fit_prm=fk['posterior_mean_fit_prm'],
                     posterior_mean_fit_pb=None,
                     psi_mix_rebuilt_maxdiff=float(np.max(np.abs(psi_mix - np.array(fk['true_psi'])))), eta_mix_rebuilt_maxdiff=float(np.max(np.abs(eta_mix - np.array(fk['true_eta'])))),
                     ds_psi=fk['ds_psi'], ds_eta=fk['ds_eta'], true_psi_mixed=fk['true_psi'], true_eta_mixed=fk['true_eta'], true_psi_prm=psi_prm.tolist(), true_eta_prm=eta_prm.tolist(),
                     mark_rate_pb_fit=mk[fpb].mean(0).tolist(), mark_rate_prm_fit=mk[fp_].mean(0).tolist()))
out['4'] = rows
post = D['posterior'].astype(float)
print('posterior mean on eval-fold PRMB steps %.4f, PB steps %.4f; PRMB true error share %.4f' % (post[prm_step].mean(), post[~prm_step].mean(), lab_meta[prm_step].mean()))
print('posterior mean on PRMB error steps %.3f vs PRMB correct steps %.3f' % (post[prm_step & lab_meta].mean(), post[prm_step & ~lab_meta].mean()))
f0 = rows[0]
print('fold 0: ds_prev %.3f | mixed truth prev %.3f | PRMB truth prev %.3f | PB share of fit steps %.3f' % (f0['ds_prev'], f0['stored_true_prev_mixed'], f0['stored_true_prev_prm'], f0['pb_share_of_fit_steps']))
print('rebuild check of stored mixed truth: psi maxdiff %.2e eta maxdiff %.2e' % (max(r['psi_mix_rebuilt_maxdiff'] for r in rows), max(r['eta_mix_rebuilt_maxdiff'] for r in rows)))
hdr = 'channel'.ljust(18) + ' ds_psi  psi_mix psi_PRMB |  ds_eta eta_mix eta_PRMB | mark_PB mark_PRMB'
print(hdr)
A = lambda key: np.mean([r[key] for r in rows], axis=0)
for j, c in enumerate(chans):
    print(c.ljust(18), '%.3f   %.3f   %.3f   |  %.3f   %.3f   %.3f   |  %.3f   %.3f' % (A('ds_psi')[j], A('true_psi_mixed')[j], A('true_psi_prm')[j], A('ds_eta')[j], A('true_eta_mixed')[j], A('true_eta_prm')[j], A('mark_rate_pb_fit')[j], A('mark_rate_prm_fit')[j]))
# ranges over folds and level channels (first 9 are the DERIVATIVE_CHANNELS level ones)
lvl = list(range(9))
for key in ['ds_psi', 'true_psi_mixed', 'true_psi_prm', 'ds_eta', 'true_eta_mixed', 'true_eta_prm']:
    v = np.array([r[key] for r in rows])
    print(key, 'level-channel range over folds: %.3f-%.3f' % (v[:, lvl].min(), v[:, lvl].max()), '| all 11: %.3f-%.3f' % (v.min(), v.max()))
# ---------- 5 ProcessBench per cell
print('== ProcessBench per cell')
PBc = sorted(set(cells[pb])); prows = []
for r in RULES:
    pr = np.full(n, -2)
    for i in np.flatnonzero(pb):
        fi = np.flatnonzero(D[r][off[i]:off[i + 1]]); pr[i] = int(fi[0]) if len(fi) else -1
    for c in PBc:
        m = cells == c; err = m & (target >= 0); cor = m & (target < 0)
        ae = (pr[err] == target[err]).mean(); ac = (pr[cor] == -1).mean()
        f1 = 0.0 if ae == 0 and ac == 0 else 2 * ae * ac / (ae + ac)
        prows.append(dict(rule=r, cell=c, n=int(m.sum()), n_err=int(err.sum()), n_cor=int(cor.sum()), acc_err=ae, acc_cor=ac, f1=f1,
                          err_pred_none=float((pr[err] == -1).mean()), err_pred_early=float((pr[err] >= 0).astype(bool)[pr[err] < target[err]].sum() / err.sum()),
                          err_pred_late=float(((pr[err] > target[err])).mean())))
PT = pd.DataFrame(prows)
piv = PT.pivot(index='cell', columns='rule', values='f1')[RULES]
print(PT[PT.rule == 'R0_frozen'][['cell', 'n', 'n_err', 'n_cor']].to_string(index=False))
print('F1'); print(piv.to_string(float_format=lambda x: f'{x:.3f}')); print('macro', piv.mean().round(4).to_dict())
print('acc_err'); print(PT.pivot(index='cell', columns='rule', values='acc_err')[RULES].to_string(float_format=lambda x: f'{x:.3f}'))
print('acc_cor'); print(PT.pivot(index='cell', columns='rule', values='acc_cor')[RULES].to_string(float_format=lambda x: f'{x:.3f}'))
mac = PT.groupby('rule')[['acc_err', 'acc_cor', 'err_pred_none', 'err_pred_early', 'err_pred_late']].mean().loc[RULES]
print(mac.round(4).to_string())
out['5'] = prows
# decompose F1 gain: counterfactual F1 using R1 acc_cor with R0 acc_err, and vice versa
dec = []
for c in PBc:
    a = PT[(PT.cell == c)].set_index('rule')
    def F(ae, ac): return 0.0 if ae == 0 and ac == 0 else 2 * ae * ac / (ae + ac)
    dec.append(dict(cell=c, R0=a.loc['R0_frozen', 'f1'], R2=a.loc['R2_allocate', 'f1'], R2_only_correct_side=F(a.loc['R0_frozen', 'acc_err'], a.loc['R2_allocate', 'acc_cor']),
                    R2_only_error_side=F(a.loc['R2_allocate', 'acc_err'], a.loc['R0_frozen', 'acc_cor'])))
Dd = pd.DataFrame(dec); print(Dd.round(4).to_string(index=False)); print('macro', Dd.mean(numeric_only=True).round(4).to_dict())
# ---------- extra: class-level allocation diagnostic (uses class metadata; not a candidate)
def prm_score(f, mask):
    sm = mask[aid]; v = ~f[sm]; g = ~lab_meta[sm]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))
def top_by(score, counts):
    f = np.zeros(S_, bool)
    for i in np.flatnonzero(prm):
        k = int(min(counts[i], ns[i]))
        if k > 0:
            a = off[i]; o_ = np.argsort(-score[a:off[i + 1]], kind='stable'); f[a + o_[:k]] = True
    return f
rate_cls = {c: nf['R1_global'][prm & (cls == c)].sum() / ns[prm & (cls == c)].sum() for c in CLASSES + ['correct']}
cnt_cls = np.zeros(n)
for i in np.flatnonzero(prm): cnt_cls[i] = np.floor(ns[i] * rate_cls[cls[i]] + 0.5)
glob = D['R1_global'][noncontrol[aid]].mean(); cnt_uni = np.zeros(n)
for i in np.flatnonzero(prm): cnt_uni[i] = np.floor(ns[i] * glob + 0.5)
out['alloc_diag'] = dict(class_rate_alloc=prm_score(top_by(Seq, cnt_cls), noncontrol), uniform_rate_alloc=prm_score(top_by(Seq, cnt_uni), noncontrol),
                         R0=prm_score(D['R0_frozen'], noncontrol), R2=prm_score(D['R2_allocate'], noncontrol), class_rates={k: float(v) for k, v in rate_cls.items()})
print('allocation diagnostic', out['alloc_diag'])
# ms length-matched vs controls
for name, mk in [('multi_solutions', ms), ('erroneous', has_err & noncontrol)]:
    rws = []
    for L in sorted(set(ns[mk])):
        a = mk & (ns == L); b = control & (ns == L)
        if a.any() and b.any(): rws.append((a.sum(), (nf['R1_global'][a] > 0).mean(), (nf['R1_global'][b] > 0).mean()))
    rw = np.array(rws, float); w = rw[:, 0] / rw[:, 0].sum()
    print(f'{name} R1 >=1 flag {(nf["R1_global"][mk] > 0).mean():.3f}; controls reweighted to {name} lengths {(w * rw[:, 2]).sum():.3f}; {name} covered {int(rw[:, 0].sum())}/{int(mk.sum())}')
dflt = lambda x: x.item() if isinstance(x, np.generic) else str(x)
json.dump(out, open('audit3_out.json', 'w'), indent=1, default=dflt)
