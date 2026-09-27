"""Red-team agent B: independent population coverage audit of the PRMScore decomposition claims.
Reads only raw inputs + frozen score arrays; never reads the decomposition outputs."""
import json, pickle, sys, hashlib
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
W = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ST = W / 'results/expectation_realization_v1'
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.prmbench import prmbench_evaluate, eval_on_hallucination_step, CATEGORY_OF

out = {}
def rep(k, v):
    out[k] = v; print(k, '=>', v if not isinstance(v, (dict, list)) or len(str(v)) < 1500 else str(v)[:1500] + ' ...')

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; lab = Z['labels'].astype(bool)
n = len(ans); ns = np.diff(off); S = int(off[-1])
rep('n_answers_all', n); rep('n_steps_all', S); rep('offsets_len_ok', len(off) == n + 1)
rep('cells', dict(Counter(ans.cell)))
prm = ~ans.cell.str.startswith('pb_').to_numpy()
rep('prm_answers', int(prm.sum()))
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
mp = Path(freeze['prm_metadata']['path'])
h = hashlib.sha256(mp.read_bytes()).hexdigest(); rep('meta_sha_matches_freeze', h == freeze['prm_metadata']['sha256'])
raw = pickle.load(open(mp, 'rb'))
rep('meta_container', (type(raw).__name__, len(raw)))
metas = list(raw.values()) if isinstance(raw, dict) else list(raw)
meta = {m['idx']: m for m in metas}
rep('meta_unique_idx', len(meta))
ids = ans.id.to_numpy(); P = np.flatnonzero(prm)
rep('prm_ids_unique', len(set(ids[P])) == len(P))
rep('prm_ids_all_in_meta', all(ids[i] in meta for i in P))
rep('meta_ids_all_in_population', set(meta) == set(ids[P]))
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
rep('classes_all_prm', dict(Counter(cls[P])))
control = prm & (cls == 'correct'); nc = prm & ~control
rep('controls', int(control.sum())); rep('noncontrol', int(nc.sum()))

# labels vs official annotations (1-based, out-of-range inert)
bad = [i for i in P if not np.array_equal(lab[off[i]:off[i+1]], np.isin(np.arange(ns[i]) + 1, meta[ids[i]]['error_steps']))]
rep('label_mismatch_answers', len(bad))
# steps vs metadata step count
stepmis = [i for i in P if 'steps' in meta[ids[i]] and len(meta[ids[i]]['steps']) != ns[i]]
rep('meta_steps_len_mismatch', len(stepmis))
has_err = np.array([prm[i] and lab[off[i]:off[i+1]].any() for i in range(n)])
has_ok = np.array([prm[i] and (~lab[off[i]:off[i+1]]).any() for i in range(n)])
elig = has_err & has_ok
rep('controls_with_error', int((control & has_err).sum()))
rep('controls_with_nonempty_error_steps', int(sum(len(meta[ids[i]]['error_steps']) > 0 for i in np.flatnonzero(control))))
rep('erroneous_noncontrol', int((nc & has_err).sum()))
ms = nc & (cls == 'multi_solutions')
rep('multi_solutions', int(ms.sum()))
rep('ms_nonempty_error_steps', int(sum(len(meta[ids[i]]['error_steps']) > 0 for i in np.flatnonzero(ms))))
inert = nc & ~has_err & ~ms
inert_ids = ids[inert].tolist()
rep('inert', int(inert.sum()))
rep('inert_detail', [(ids[i], cls[i], int(ns[i]), list(meta[ids[i]]['error_steps'])) for i in np.flatnonzero(inert)])
rep('inert_by_class', dict(Counter(cls[inert])))
rep('inert_all_indices_past_last', all(min(meta[ids[i]]['error_steps']) > ns[i] for i in np.flatnonzero(inert)))
partial = [i for i in P if len(meta[ids[i]]['error_steps']) and max(meta[ids[i]]['error_steps']) > ns[i] and has_err[i]]
rep('partially_out_of_range_answers_with_in_range_error', len(partial))
anyoob = [i for i in P if any((e < 1 or e > ns[i]) for e in meta[ids[i]]['error_steps'])]
rep('any_out_of_range_answers_total', len(anyoob))
rep('any_out_of_range_by_class', dict(Counter(cls[anyoob])))
rep('within_auc_eligible', int(elig.sum()))
allerr = np.flatnonzero(has_err & ~has_ok)
rep('erroneous_all_steps_error (no within-AUC)', [(ids[i], cls[i], int(ns[i])) for i in allerr])
rep('one_step_answers_prm', int((prm & (ns == 1)).sum()))
rep('one_step_answers_noncontrol', int((nc & (ns == 1)).sum()))
rep('one_step_by_class', dict(Counter(cls[prm & (ns == 1)])))

# first-error position
fe = {i: int(np.flatnonzero(lab[off[i]:off[i+1]])[0]) for i in np.flatnonzero(nc & has_err)}
def pb(i):
    f = fe[i]; rel = f / (ns[i] - 1) if ns[i] > 1 else 0.0
    return 'first_step' if f == 0 else 'early' if rel <= 1/3 else 'middle' if rel <= 2/3 else 'late'
pos = {i: pb(i) for i in fe}
rep('first_error_position_bins', dict(Counter(pos.values())))
fs = [i for i in fe if fe[i] == 0]
rep('first_step_by_class', dict(Counter(cls[fs])))
rep('first_step_eligible_for_within_auc', int(sum(elig[i] for i in fs)))
rep('first_step_n_steps_hist', dict(Counter(int(ns[i]) for i in fs)))

# classes & sizes
g_all = ans.source_group.to_numpy(); fold = ans.fold.to_numpy()
rep('prm_source_groups', len(set(g_all[P])))
rep('noncontrol_source_groups', len(set(g_all[nc])))
rep('control_only_groups', len(set(g_all[control]) - set(g_all[nc])))
rep('pb_prm_shared_groups', len(set(g_all[P]) & set(g_all[~prm])))
gf = pd.DataFrame({'g': g_all[P], 'f': fold[P]}).groupby('g').f.nunique()
rep('groups_crossing_folds_prm', int((gf > 1).sum()))
gf2 = pd.DataFrame({'g': g_all, 'f': fold}).groupby('g').f.nunique()
rep('groups_crossing_folds_all', int((gf2 > 1).sum()))
rep('folds_prm_answers', dict(Counter(fold[P].tolist())))
rep('folds_prm_steps', {k: int(ns[P][fold[P] == k].sum()) for k in range(5)})
rep('folds_noncontrol_answers', dict(Counter(fold[nc].tolist())))
rows = []
for c in sorted(set(cls[P])):
    sel = prm & (cls == c)
    rows.append({'class': c, 'answers': int(sel.sum()), 'groups': len(set(g_all[sel])), 'steps': int(ns[sel].sum()),
                 'error_steps': int(sum(lab[off[i]:off[i+1]].sum() for i in np.flatnonzero(sel))),
                 'erroneous': int((sel & has_err).sum()), 'eligible': int((sel & elig).sum()),
                 'inert': int((sel & inert).sum()), 'first_step_err': int(sum(1 for i in fs if cls[i] == c)),
                 'min_fold_answers': int(min(Counter(fold[sel]).values())), 'folds': len(set(fold[sel]))})
CL = pd.DataFrame(rows); print(CL.to_string(index=False)); out['class_table'] = rows
# answers per group in noncontrol: max share
gc = Counter(g_all[nc]); rep('noncontrol_answers_per_group_max', max(gc.values())); rep('noncontrol_answers_per_group_median', float(np.median(list(gc.values()))))

# ---------------- scores: finiteness, byte identity, thresholds
OURS = {'B13_equal': 'b', 'S_equal': 'b', 'G1_sml': 'b', 'B13_lsml': 'b', 'S_lsml': 'b', 'fam421': 'b', 'ct7': 'b', 'step_index': 'b', 'B_sml__merge': 'b2'}
F = {'b': np.load(ST / 'run_20260927_stage_b/STEP_SCORES.npz'), 'b2': np.load(ST / 'run_20260927_stage_b2/STEP_SCORES.npz')}
T = {'b': np.load(ST / 'run_20260927_stage_b_thr/STEP_SCORES.npz'), 'b2': np.load(ST / 'run_20260927_stage_b2_thr/STEP_SCORES.npz')}
TH = {'b': json.loads((ST / 'run_20260927_stage_b_thr/THRESHOLDS.json').read_text('utf8')), 'b2': json.loads((ST / 'run_20260927_stage_b2_thr/THRESHOLDS.json').read_text('utf8'))}
for st in F:
    fz, tz = F[st], T[st]
    rep(f'{st}_names_equal', sorted(fz.files) == sorted(tz.files))
    ne = [k for k in fz.files if not np.array_equal(fz[k], tz[k], equal_nan=True)]
    rep(f'{st}_arrays_not_equal', ne)
    rep(f'{st}_arrays_bitwise_equal_count', sum(fz[k].tobytes() == tz[k].tobytes() for k in fz.files))
    rep(f'{st}_n_arrays', len(fz.files))
    rep(f'{st}_offsets_equal_oof', np.array_equal(fz['offsets'], off))
    f1 = hashlib.sha256((ST / ('run_20260927_stage_' + st[1:] if False else '')).as_posix().encode()).hexdigest()
for st, d in (('b', 'run_20260927_stage_b'), ('b2', 'run_20260927_stage_b2')):
    a = hashlib.sha256((ST / d / 'STEP_SCORES.npz').read_bytes()).hexdigest(); b = hashlib.sha256((ST / (d + '_thr') / 'STEP_SCORES.npz').read_bytes()).hexdigest()
    rep(f'{st}_npz_file_sha_equal', a == b)
prm_step = np.repeat(prm, ns); step_fold = np.repeat(fold, ns); nc_step = np.repeat(nc, ns)
fin = {}
for m, st in OURS.items():
    s = F[st][m]
    th = TH[st].get(m)
    fin[m] = {'finite_prm_steps': int(np.isfinite(s[prm_step]).sum()), 'prm_steps': int(prm_step.sum()),
              'finite_all_steps': int(np.isfinite(s).sum()),
              'thr_folds': sorted(th.keys()) if th else None, 'thr_all_finite': bool(th and all(np.isfinite(float(v)) for v in th.values()))}
ch = np.load(ST / 'run_20260927_stage_b_thr/CHANNELS.npz'); names = [str(x) for x in ch['names']]
rd = ch['values'][:, names.index('realized_drv')].astype(float)
rep('channels_offsets_equal', np.array_equal(ch['offsets'], off))
fin['realized_drv'] = {'finite_prm_steps': int(np.isfinite(rd[prm_step]).sum()), 'prm_steps': int(prm_step.sum()), 'finite_all_steps': int(np.isfinite(rd).sum())}
reward = np.full(S, np.nan); badrw = []
for i in P:
    r = np.asarray(meta[ids[i]]['rewards'], float)
    if len(r) != ns[i]: badrw.append(ids[i]); continue
    reward[off[i]:off[i+1]] = r
rep('reward_len_mismatch', len(badrw))
fin['PRM_reward'] = {'finite_prm_steps': int(np.isfinite(reward[prm_step]).sum()), 'prm_steps': int(prm_step.sum()),
                     'min': float(np.nanmin(reward[prm_step])), 'max': float(np.nanmax(reward[prm_step]))}
rep('finiteness_and_thresholds', fin)

# ---------------- decisions under the frozen rule; recompute thresholds for fixed rows
def zt(x): return (x - x.mean()) / max(x.std(), 1e-8)
def answer_z(s):
    z = np.full(S, np.nan)
    for i in P: z[off[i]:off[i+1]] = zt(s[off[i]:off[i+1]])
    return z
def q80(z): return {k: float(np.quantile(z[prm_step & (step_fold == (k+1) % 5)], .8)) for k in range(5)}
def valid_from(z, tau):
    v = np.zeros(S, bool)
    for k in range(5):
        sel = prm_step & (step_fold == k); v[sel] = z[sel] < tau[k]
    return v
score, zs, tau, valid = {}, {}, {}, {}
for m, st in OURS.items():
    score[m] = F[st][m]; zs[m] = answer_z(score[m]); tau[m] = {int(k): float(v) for k, v in TH[st][m].items()}; valid[m] = valid_from(zs[m], tau[m])
thr_check = {}
for m in OURS:
    rq = q80(zs[m]); thr_check[m] = max(abs(rq[k] - tau[m][k]) for k in range(5))
rep('saved_thr_minus_OOF_q80_rule (0 expected only for fixed rows)', thr_check)
score['realized_drv'] = rd; zs['realized_drv'] = answer_z(rd); tau['realized_drv'] = q80(zs['realized_drv']); valid['realized_drv'] = valid_from(zs['realized_drv'], tau['realized_drv'])
risk = 1 - reward; score['PRM'] = risk
zs['PRM_z'] = answer_z(risk); tau['PRM_z_q80'] = q80(zs['PRM_z']); valid['PRM_z_q80'] = valid_from(zs['PRM_z'], tau['PRM_z_q80'])
tau['PRM_raw_q80'] = q80(risk); valid['PRM_raw_q80'] = valid_from(risk, tau['PRM_raw_q80'])
valid['PRM_native'] = prm_step & (reward >= 0.5)
rep('realized_drv_tau', tau['realized_drv']); rep('PRM_z_tau', tau['PRM_z_q80'])
# NaN in z scale?
rep('nan_in_answer_z', {m: int((~np.isfinite(zs[m][prm_step])).sum()) for m in zs})
# answers whose score is constant (z==0 everywhere => always valid)
const = {}
for m in list(OURS) + ['realized_drv', 'PRM']:
    s = score[m]; c = [i for i in P if np.ptp(s[off[i]:off[i+1]]) == 0]
    const[m] = {'prm': len(c), 'noncontrol': int(sum(nc[i] for i in c)), 'erroneous_multistep': int(sum(has_err[i] and ns[i] > 1 for i in c))}
rep('constant_score_answers', const)
# argmax ties (hit uses earliest argmax with 8 eps tolerance)
ties = {}
for m in list(OURS) + ['realized_drv', 'PRM']:
    s = score[m]; t = 0
    for i in np.flatnonzero(nc & has_err):
        v = s[off[i]:off[i+1]]
        if (v >= v.max() - 8 * np.finfo(float).eps).sum() > 1: t += 1
    ties[m] = t
rep('erroneous_answers_with_tied_argmax', ties)

# ---------------- official scorer: controls excluded, inert answers treated as all-correct
def official(v, keep=None):
    Pk = P if keep is None else np.array([i for i in P if keep[i]])
    return prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i+1]].astype(int).tolist()} for i in Pk], [meta[ids[i]] for i in Pk])
def counts(v, mask):
    tp = fp = tn = fn = 0
    for i in np.flatnonzero(mask):
        vv = v[off[i]:off[i+1]]; gg = ~lab[off[i]:off[i+1]]
        tp += int((vv & gg).sum()); fp += int((vv & ~gg).sum()); tn += int((~vv & ~gg).sum()); fn += int((~vv & gg).sum())
    return tp, fp, tn, fn
def prmscore(c):
    tp, fp, tn, fn = c
    p = tp/(tp+fp) if tp+fp else np.nan; r = tp/(tp+fn) if tp+fn else np.nan
    p2 = tn/(tn+fn) if tn+fn else np.nan; r2 = tn/(tn+fp) if tn+fp else np.nan
    f1 = 2*p*r/(p+r); f2 = 2*p2*r2/(p2+r2) if (p2+r2) else np.nan
    return 0.5*(f1+f2), f1, f2
ARMS = list(OURS) + ['realized_drv', 'PRM_native', 'PRM_raw_q80', 'PRM_z_q80']
offi = {}
for m in ARMS:
    res = official(valid[m]); tot = res['total']
    offi[m] = 0.5*(tot['f1'] + tot['negative_f1'])
    c_nc = counts(valid[m], nc); c_all = counts(valid[m], prm)
    offtot = (0,)  # placeholder
    print(m, 'official', round(offi[m], 6), 'counts-noncontrol', round(prmscore(c_nc)[0], 6), 'counts-incl-controls', round(prmscore(c_all)[0], 6))
    out.setdefault('prmscore_official', {})[m] = {'official': offi[m], 'noncontrol_counts': prmscore(c_nc)[0], 'if_controls_pooled': prmscore(c_all)[0],
                                                'n_scored': res['n_predictions_scored'], 'n_control_rows': res['n_correct_control_rows']}
rep('C1_S_equal_minus_realized_drv_official', offi['S_equal'] - offi['realized_drv'])
# inert: official per-row counts for inert answers have FP=TN=0
inert_rows = [eval_on_hallucination_step(meta[ids[i]]['error_steps'], valid['S_equal'][off[i]:off[i+1]].astype(int).tolist())['f1_matrix'] for i in np.flatnonzero(inert)]
rep('inert_rows_FP_TN_zero', all(r['FP'] == 0 and r['TN'] == 0 for r in inert_rows))

# frozen METRICS replay for our rows
Mb = pd.read_csv(ST / 'run_20260927_stage_b/METRICS.csv'); Mb2 = pd.read_csv(ST / 'run_20260927_stage_b2/METRICS.csv')
print(Mb.columns.tolist()); print(Mb.head())
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1*(n1+1)/2) / (n1*n0))
rp = {}
for m, st in OURS.items():
    M = (Mb if st == 'b' else Mb2); M = M[(M.method == m) & (M.benchmark == 'prm') & (M.stratum == 'all')].set_index('metric').estimate
    wa = np.mean([within_auc(lab[off[i]:off[i+1]], score[m][off[i]:off[i+1]]) for i in np.flatnonzero(elig)])
    rp[m] = {'prmscore_diff': abs(float(M['prmscore']) - offi[m]), 'within_auc_diff': abs(float(M['within_auc']) - wa)}
rep('frozen_metrics_replay', rp)

# ---------------- C4: pooled AUC pair shares
for m in ['S_equal', 'realized_drv', 'PRM_z_q80']:
    zz = (zs['PRM_z'] if m == 'PRM_z_q80' else zs[m])[nc_step]; y = lab[nc_step]
    P_ = int(y.sum()); N_ = len(y) - P_
    pw = sum(int(lab[off[i]:off[i+1]].sum()) * int((~lab[off[i]:off[i+1]]).sum()) for i in np.flatnonzero(nc & elig))
    rep(f'C4_{m}', {'error_steps_P': P_, 'correct_steps_N': N_, 'all_pairs': P_*N_, 'within_pairs': pw, 'cross_share': 1 - pw/(P_*N_)})
    break

# ---------------- C3 control false-alarm shares
for m in ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'PRM_raw_q80', 'PRM_native']:
    fl = np.array([(~valid[m][off[i]:off[i+1]]).sum() for i in np.flatnonzero(control)]); st_ = ns[control]
    rep(f'C3_{m}', {'controls': int(control.sum()), 'control_steps': int(st_.sum()), 'step_false_flag_rate': float(fl.sum()/st_.sum()), 'share_with_flag': float((fl > 0).mean())})
    msf = np.array([(~valid[m][off[i]:off[i+1]]).sum() for i in np.flatnonzero(ms)])
    inf_ = np.array([(~valid[m][off[i]:off[i+1]]).sum() for i in np.flatnonzero(inert)])
    rep(f'C3_{m}_ms_inert', {'ms_answers': int(ms.sum()), 'ms_share_flag': float((msf > 0).mean()), 'inert_answers': int(inert.sum()), 'inert_share_flag': float((inf_ > 0).mean()), 'inert_steps': int(ns[inert].sum())})
# fraction of steps flagged in the control set expected ~? q80 is fit on all PRM steps incl controls
rep('calibration_fold_steps_include_controls', {k: int((prm_step & (step_fold == (k+1) % 5) & np.repeat(control, ns)).sum()) for k in range(5)})

# ---------------- C6 hit
hit = {}
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8*np.finfo(float).eps)[0])
E = np.flatnonzero(nc & has_err)
for m in ['S_equal', 'realized_drv', 'B13_equal', 'PRM']:
    hit[m] = np.array([lab[off[i] + earliest_argmax(score[m][off[i]:off[i+1]])] for i in E])
rep('C6_n_erroneous', len(E))
rep('C6_hit_rates', {m: float(h.mean()) for m, h in hit.items()})
a, b = hit['S_equal'], hit['realized_drv']
rep('C6_S_equal_vs_realized_drv', {'both': int((a & b).sum()), 'only_S': int((a & ~b).sum()), 'only_rd': int((~a & b).sum()), 'neither': int((~a & ~b).sum())})
a, b = hit['S_equal'], hit['PRM']
rep('C6_S_equal_vs_PRM', {'both': int((a & b).sum()), 'only_S': int((a & ~b).sum()), 'only_PRM': int((~a & b).sum()), 'neither': int((~a & ~b).sum())})
# one-step erroneous answers hit trivially
rep('C6_one_step_erroneous (hit=1 for every arm)', int(sum(ns[i] == 1 for i in E)))
rep('C6_all_steps_error (hit=1 for every arm)', int(sum(not has_ok[i] for i in E)))

# ---------------- task 4: per-class PRMScore contrasts with inert answers included (official) vs excluded
CL9 = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception']
pairs = [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('B13_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80'), ('S_equal', 'PRM_raw_q80'), ('S_equal', 'PRM_native'), ('realized_drv', 'PRM_z_q80')]
t4 = []
for c in CL9 + ['soundness', 'sensitivity', 'total']:
    if c == 'soundness': memb = [x for x in CL9 if CATEGORY_OF[x] == 'soundness']
    elif c == 'sensitivity': memb = ['missing_condition', 'deception', 'multi_solutions']
    elif c == 'total': memb = CL9 + ['multi_solutions']
    else: memb = [c]
    base = nc & np.isin(cls, memb)
    if not (base & inert).any(): continue
    for a_, b_ in pairs:
        inc = prmscore(counts(valid[a_], base))[0] - prmscore(counts(valid[b_], base))[0]
        exc = prmscore(counts(valid[a_], base & ~inert))[0] - prmscore(counts(valid[b_], base & ~inert))[0]
        t4.append({'stratum': c, 'contrast': f'{a_}-{b_}', 'with_inert': round(inc, 5), 'without_inert': round(exc, 5), 'change': round(exc - inc, 5), 'sign_flip': bool(np.sign(inc) != np.sign(exc))})
T4 = pd.DataFrame(t4); print(T4.to_string(index=False)); out['task4'] = t4
# arm-level per-class PRMScore change
t4b = []
for c in ['confidence', 'counterfactual', 'deception', 'missing_condition']:
    base = nc & (cls == c)
    for m in ARMS:
        inc = prmscore(counts(valid[m], base)); exc = prmscore(counts(valid[m], base & ~inert))
        t4b.append({'class': c, 'arm': m, 'with': round(inc[0], 5), 'without': round(exc[0], 5), 'f1c_with': round(inc[1], 5), 'f1c_without': round(exc[1], 5)})
T4b = pd.DataFrame(t4b); print(T4b.to_string(index=False)); out['task4_arms'] = t4b
# ranking of arms per class changes?
rk = {}
for c in ['confidence', 'counterfactual', 'deception', 'missing_condition']:
    sub = T4b[T4b['class'] == c]
    rk[c] = {'order_with': sub.sort_values('with', ascending=False).arm.tolist(), 'order_without': sub.sort_values('without', ascending=False).arm.tolist()}
    rk[c]['same'] = rk[c]['order_with'] == rk[c]['order_without']
rep('task4_rank_orders_unchanged', {c: rk[c]['same'] for c in rk})

# ---------------- C1 with a quick group bootstrap (2000 draws) using noncontrol answers; groups = all PRM groups (as script) and noncontrol only
gP = g_all[P]; Gu, gi = np.unique(gP, return_inverse=True)
def pa_counts(v):
    C = np.zeros((len(P), 4))
    for j, i in enumerate(P):
        vv = v[off[i]:off[i+1]]; gg = ~lab[off[i]:off[i+1]]
        C[j] = [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    return C
Ca, Cb = pa_counts(valid['S_equal']), pa_counts(valid['realized_drv'])
ncP = nc[P]
Ga = np.zeros((len(Gu), 4)); Gb = np.zeros((len(Gu), 4)); np.add.at(Ga, gi[ncP], Ca[ncP]); np.add.at(Gb, gi[ncP], Cb[ncP])
def ps(C):
    tp, fp, tn, fn = C.T
    p = tp/(tp+fp); r = tp/(tp+fn); p2 = tn/(tn+fn); r2 = tn/(tn+fp)
    return 0.5*(2*p*r/(p+r) + 2*p2*r2/(p2+r2))
rng = np.random.default_rng(1)
Wt = rng.multinomial(len(Gu), np.full(len(Gu), 1/len(Gu)), size=2000).astype(float)
d = ps(Wt @ Ga) - ps(Wt @ Gb)
rep('C1_point_counts', float(ps(Ga.sum(0)[None])[0] - ps(Gb.sum(0)[None])[0]))
rep('C1_boot2000_all707groups', [float(np.quantile(d, .025)), float(np.quantile(d, .975))])
ncg = np.unique(gi[ncP]); Wt2 = np.zeros((2000, len(Gu)))
rng2 = np.random.default_rng(2)
Wt2[:, ncg] = rng2.multinomial(len(ncg), np.full(len(ncg), 1/len(ncg)), size=2000)
d2 = ps(Wt2 @ Ga) - ps(Wt2 @ Gb)
rep('C1_boot2000_noncontrol_groups_only', [float(np.quantile(d2, .025)), float(np.quantile(d2, .975))])
rep('n_groups_bootstrap_all_prm', len(Gu)); rep('n_groups_with_noncontrol', len(ncg))
# per-fold C1 sign
pf = {}
for k in range(5):
    mk = nc & (fold == k)
    pf[k] = {'answers': int(mk.sum()), 'delta': round(prmscore(counts(valid['S_equal'], mk))[0] - prmscore(counts(valid['realized_drv'], mk))[0], 5)}
rep('C1_per_fold', pf)
# per-class group counts for bootstrap (noncontrol)
rep('groups_per_class', {c: len(set(g_all[nc & (cls == c)])) for c in CL9 + ['multi_solutions']})

Path(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/redteam_B/audit_out.json').write_text(
    json.dumps(out, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf8')
print('DONE')
