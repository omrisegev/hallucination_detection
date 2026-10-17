exec(open('load.py').read())
from collections import Counter
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception', 'multi_solutions']
wk = np.load('work.npz'); lab_meta = wk['lab_meta']; zS = wk['zS']; Seq = wk['Seq']; Gre = wk['Gre']
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
cells = ans.cell.to_numpy(); target = ans.target.to_numpy()
meta = {m['idx']: m for m in metaraw.values()}
aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); noncontrol = prm & ~control
has_err = np.array([lab_meta[off[i]:off[i + 1]].any() for i in range(n)]) & prm
ms = noncontrol & (cls == 'multi_solutions'); inert = noncontrol & ~has_err & ~ms
nf = {r: np.bincount(aid, weights=D[r].astype(float), minlength=n).astype(int) for r in RULES}
out = {}
# ---------- 3a controls with >=1 flag by length bin
bins = [(1, 5), (6, 8), (9, 11), (12, 15), (16, 20), (21, 400)]
def lb(x):
    for a, b in bins:
        if a <= x <= b: return f'{a}-{b}'
tab = []
for grp_name, mk in [('controls', control), ('erroneous_noncontrol', has_err & noncontrol), ('redundency', prm & (cls == 'redundency'))]:
    for a, b in bins + [(1, 400)]:
        m = mk & (ns >= a) & (ns <= b)
        row = {'group': grp_name, 'len': f'{a}-{b}', 'n': int(m.sum())}
        for r in RULES: row[r] = float((nf[r][m] > 0).mean()) if m.any() else np.nan
        tab.append(row)
T = pd.DataFrame(tab); out['3a'] = T.to_dict('records')
print(T.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
# where is the drop: of controls that lose all flags under R1 (R0 had >=1), distribution by length
lost = control & (nf['R0_frozen'] > 0) & (nf['R1_global'] == 0)
print('controls losing all flags R0->R1:', int(lost.sum()), 'of', int(control.sum()), 'by len bin', Counter(lb(x) for x in ns[lost]))
print('control length dist', Counter(lb(x) for x in ns[control]))
print('median len controls', np.median(ns[control]), 'erroneous noncontrol', np.median(ns[has_err & noncontrol]))
# paired: control vs its redundency sibling (same source idx)
src = {}
for i in np.flatnonzero(prm): src.setdefault(meta[ids[i]]['source_idx'], {})[cls[i]] = i
pairs = [(v['correct'], v['redundency']) for v in src.values() if 'correct' in v and 'redundency' in v]
pc = np.array([p[0] for p in pairs]); pr = np.array([p[1] for p in pairs])
out['3b_pairs'] = {}
for r in ['R0_frozen', 'R1_global']:
    cz = nf[r][pc] == 0; rz = nf[r][pr] == 0
    out['3b_pairs'][r] = dict(pairs=len(pairs), control_zero=int(cz.sum()), sibling_zero=int(rz.sum()), both_zero=int((cz & rz).sum()),
                              control_zero_sibling_flagged=int((cz & ~rz).sum()), control_flagged_sibling_zero=int((~cz & rz).sum()),
                              sibling_error_hit_when_control_zero=int(sum((D[r][off[j]:off[j + 1]] & lab_meta[off[j]:off[j + 1]]).any() for j in pr[cz])))
print('pairs', out['3b_pairs'])
# mean G per answer: controls vs sibling vs all erroneous
Gm = np.bincount(aid, weights=Gre, minlength=n) / ns
print('mean answer-G controls %.3f redundency sib %.3f erroneous-noncontrol %.3f; paired diff (ctrl - sib) mean %.3f, share ctrl<sib %.3f' % (
    Gm[control].mean(), Gm[pr].mean(), Gm[has_err & noncontrol].mean(), (Gm[pc] - Gm[pr]).mean(), (Gm[pc] < Gm[pr]).mean()))
# share of answers flagged at all, all PRMB answers vs controls: discrimination AUC of "answer max G" for control vs erroneous
from sklearn.metrics import roc_auc_score
Gmax = np.array([Gre[off[i]:off[i + 1]].max() for i in range(n)])
m = control | (has_err & noncontrol)
out['auc_answer_maxG_erroneous_vs_control'] = float(roc_auc_score(has_err[m], Gmax[m]))
out['auc_answer_meanG_erroneous_vs_control'] = float(roc_auc_score(has_err[m], Gm[m]))
out['auc_answer_len_erroneous_vs_control'] = float(roc_auc_score(has_err[m], ns[m]))
print('AUC maxG err vs ctrl', out['auc_answer_maxG_erroneous_vs_control'], 'meanG', out['auc_answer_meanG_erroneous_vs_control'], 'len', out['auc_answer_len_erroneous_vs_control'])
# length-matched: within each length, P(>=1 flag | control) vs P(>=1 flag | erroneous)
rows = []
for L in sorted(set(ns[control])):
    mc = control & (ns == L); me = has_err & noncontrol & (ns == L)
    if mc.sum() >= 1 and me.sum() >= 1:
        rows.append((L, int(mc.sum()), int(me.sum()), (nf['R1_global'][mc] > 0).mean(), (nf['R1_global'][me] > 0).mean()))
R = pd.DataFrame(rows, columns=['len', 'nc', 'ne', 'ctrl_flag', 'err_flag'])
w = R.nc / R.nc.sum()
out['length_matched_R1'] = dict(ctrl=float((w * R.ctrl_flag).sum()), err_reweighted_to_ctrl_lengths=float((w * R.err_flag).sum()), n_ctrl_covered=int(R.nc.sum()))
print('length-matched R1 >=1 flag: controls %.3f vs erroneous reweighted to control lengths %.3f (controls covered %d)' % (out['length_matched_R1']['ctrl'], out['length_matched_R1']['err_reweighted_to_ctrl_lengths'], R.nc.sum()))
# ---------- 3c zero-flag non-controls by class and erroneous vs clean
zrows = []
for r in RULES:
    z = noncontrol & (nf[r] == 0)
    row = {'rule': r, 'zero_flag_noncontrol': int(z.sum()), 'of': int(noncontrol.sum()), 'erroneous': int((z & has_err).sum()), 'clean_ms': int((z & ms).sum()), 'clean_inert': int((z & inert).sum()),
           'erroneous_error_steps_missed': int(lab_meta[(z & has_err)[aid]].sum())}
    for c in CLASSES: row[c] = int((z & (cls == c)).sum())
    zrows.append(row)
Zt = pd.DataFrame(zrows); out['3c'] = Zt.to_dict('records'); print(Zt.to_string(index=False))
# any-error hit on erroneous non-controls
for r in RULES:
    e = has_err & noncontrol
    hit = np.array([(D[r][off[i]:off[i + 1]] & lab_meta[off[i]:off[i + 1]]).any() for i in np.flatnonzero(e)])
    print(r, 'erroneous with an error flagged %.4f (%d/%d)' % (hit.mean(), hit.sum(), e.sum()), ' clean ms with flag %.3f inert with flag %.3f' % ((nf[r][ms] > 0).mean(), (nf[r][inert] > 0).mean()))
# ---------- 3d rate-matched R0 diagnostic (label-free threshold matched to R2 noncontrol flag rate, eval-fold distribution)
nc_step = noncontrol[aid]
target_rate = D['R2_allocate'][nc_step].mean()
t = np.quantile(zS[nc_step], 1 - target_rate)
R0m = zS >= t
def prm_score(f, mask):
    sm = mask[aid]; v = ~f[sm]; g = ~lab_meta[sm]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)), float((~v).mean())
out['3d_rate_matched_R0'] = dict(rate_target=float(target_rate), thr=float(t), prmscore=prm_score(R0m, noncontrol), R0=prm_score(D['R0_frozen'], noncontrol), R2=prm_score(D['R2_allocate'], noncontrol))
print('rate-matched R0', out['3d_rate_matched_R0'])
# R0 counts reallocated: same per-answer count as R0 but... ; count-shuffle control: R1 counts permuted among noncontrol answers of equal length and fold, placement by S
rng = np.random.default_rng(7); vals = []
key = pd.DataFrame({'i': np.flatnonzero(prm), 'L': ns[prm], 'f': fold[prm]})
for d in range(20):
    cnt = nf['R1_global'].copy()
    for _, g in key.groupby(['L', 'f']).i:
        idx = g.to_numpy(); cnt[idx] = nf['R1_global'][rng.permutation(idx)]
    fl = np.zeros(S_, bool)
    for i in np.flatnonzero(prm):
        k = int(min(cnt[i], ns[i]))
        if k > 0:
            a = off[i]; o_ = np.argsort(-Seq[a:off[i + 1]], kind='stable'); fl[a + o_[:k]] = True
    vals.append(prm_score(fl, noncontrol)[0])
out['3e_count_shuffle_within_len_fold'] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), min=float(np.min(vals)), max=float(np.max(vals)), draws=20)
print('R2 with R1 counts shuffled among same-length same-fold PRMB answers (placement by S):', out['3e_count_shuffle_within_len_fold'])
# swap-null singletons (noncontrol answers alone in their (len, fold) cell)
kb = pd.DataFrame({'i': np.flatnonzero(noncontrol), 'L': ns[noncontrol], 'f': fold[noncontrol]})
sz = kb.groupby(['L', 'f']).i.transform('size'); single = kb[sz < 2].i.to_numpy()
out['swap_singletons'] = dict(answers=int(len(single)), steps=int(ns[single].sum()), error_steps=int(lab_meta[np.isin(aid, single)].sum()), lengths=sorted(Counter(ns[single].tolist()).items()), classes=dict(Counter(cls[single])))
print('swap singletons', out['swap_singletons'])
dflt = lambda x: x.item() if isinstance(x, np.generic) else str(x)
json.dump(out, open('audit2_out.json', 'w'), indent=1, default=dflt)
