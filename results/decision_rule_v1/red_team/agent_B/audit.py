exec(open('load.py').read())
from collections import Counter
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.prmbench import prmbench_evaluate
from spectral_utils.lsml_gate_locator_research import answer_standardize
RULES = ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']
CLASSES = ['redundency', 'circular', 'counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence', 'missing_condition', 'deception', 'multi_solutions']
out = {}
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); ids = ans.id.to_numpy(); groups = ans.source_group.to_numpy()
cells = ans.cell.to_numpy(); target = ans.target.to_numpy()
meta = {m['idx']: m for m in metaraw.values()}
aid = np.repeat(np.arange(n), ns); step_fold = fold[aid]; prm_step = prm[aid]
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); noncontrol = prm & ~control


def errsteps(m):
    e = m['error_steps']; e = json.loads(e) if isinstance(e, str) else e
    return list(e)


# ---------- B: independent labels from metadata (1-based error_steps), never from NPZ
lab_meta = np.zeros(S_, bool); nstep_mismatch = 0; oob_rows = 0
for i in np.flatnonzero(prm):
    m = meta[ids[i]]; e = errsteps(m)
    if m['n_steps'] != ns[i]: nstep_mismatch += 1
    if any(x < 1 or x > ns[i] for x in e): oob_rows += 1
    lab_meta[off[i]:off[i + 1]] = np.isin(np.arange(1, ns[i] + 1), e)
out['B_meta_vs_npz_label_mismatch_steps_prm'] = int((lab_meta[prm_step] != labels[prm_step]).sum())
out['B_nsteps_mismatch_answers'] = nstep_mismatch; out['B_prm_rows_with_oob_error_steps'] = oob_rows
out['B_meta_ids_equal_prm_ids'] = set(meta) == set(ids[prm])
has_err = np.array([lab_meta[off[i]:off[i + 1]].any() for i in range(n)]) & prm
ms = noncontrol & (cls == 'multi_solutions'); inert = noncontrol & ~has_err & ~ms
out['pop'] = dict(answers=n, prm=int(prm.sum()), pb=int(pb.sum()), controls=int(control.sum()), noncontrol=int(noncontrol.sum()),
                  erroneous=int(has_err.sum()), ms=int(ms.sum()), ms_with_errors=int((ms & has_err).sum()), inert=int(inert.sum()),
                  inert_classes=dict(Counter(cls[inert])), steps_total=S_, steps_prm=int(prm_step.sum()), steps_pb=int((~prm_step).sum()),
                  steps_noncontrol=int(noncontrol[aid].sum()), error_steps_noncontrol=int(lab_meta[noncontrol[aid]].sum()),
                  npz_pb_label_mean=float(labels[~prm_step].mean()))


def indep(flag, mask_ans):
    sm = mask_ans[aid]; v = ~flag[sm]; g = ~lab_meta[sm]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan; f1e = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), f1=f1, f1e=f1e, prmscore=(f1 + f1e) / 2, flag_rate=float((~v).mean()))


# ---------- C: official PRMScore per rule + independent recount
P = np.flatnonzero(prm)
res = {}
for r in RULES:
    f = D[r].astype(bool); v = ~f
    o = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
    ind = indep(f, noncontrol)
    pc = {}
    for c in CLASSES:
        ii = indep(f, noncontrol & (cls == c))
        pc[c] = dict(official=0.5 * (o['by_classification']['f1'][c] + o['by_classification']['negative_f1'][c]), indep=ii['prmscore'],
                     f1=o['by_classification']['f1'][c], nf1=o['by_classification']['negative_f1'][c], flag_rate=ii['flag_rate'])
    res[r] = dict(official=0.5 * (o['total']['f1'] + o['total']['negative_f1']), indep=ind['prmscore'], f1=o['total']['f1'], nf1=o['total']['negative_f1'],
                  flag_rate_noncontrol=ind['flag_rate'], counts=[ind['tp'], ind['fp'], ind['tn'], ind['fn']], per_class=pc, n_scored=o['n_predictions_scored'])
out['C'] = res
cu = {}
for c in CLASSES + ['correct']:
    mk = prm & (cls == c)
    cu[c] = dict(answers=int(mk.sum()), steps=int(ns[mk].sum()), error_steps=int(lab_meta[mk[aid]].sum()), erroneous_answers=int((has_err & mk).sum()),
                 source_groups=int(len(set(groups[mk]))), folds=sorted((int(a), int(b)) for a, b in Counter(fold[mk]).items()))
out['class_units'] = cu
# ---------- D coverage
gf = pd.DataFrame({'g': groups, 'f': fold}).groupby('g').f.nunique()
red_groups = set(groups[prm & (cls == 'redundency')])
out['D'] = dict(groups_total=int(len(set(groups))), groups_prm=int(len(set(groups[prm]))), groups_noncontrol=int(len(set(groups[noncontrol]))),
                groups_pb=int(len(set(groups[pb]))), groups_crossing_folds=int((gf > 1).sum()),
                groups_shared_pb_prm=int(len(set(groups[pb]) & set(groups[prm]))),
                controls_sharing_group_with_redundency=int(sum(1 for i in np.flatnonzero(control) if groups[i] in red_groups)),
                steps_per_fold={int(k): int((step_fold == k).sum()) for k in range(5)},
                answers_per_fold_prm={int(k): int((prm & (fold == k)).sum()) for k in range(5)})
# ---------- E decision integrity
post = D['posterior'].astype(float); G = D['G'].astype(float)
out['E_finite'] = dict(post_finite=int(np.isfinite(post).sum()), G_finite=int(np.isfinite(G).sum()), total=S_,
                       per_fold={int(k): [int(np.isfinite(post[step_fold == k]).sum()), int(np.isfinite(G[step_fold == k]).sum()), int((step_fold == k).sum())] for k in range(5)},
                       flag_dtypes={r: str(D[r].dtype) for r in RULES})
nf = {r: np.bincount(aid, weights=D[r].astype(float), minlength=n).astype(int) for r in RULES}
out['E_R2_counts_equal_R1'] = dict(answers_equal=int((nf['R1_global'] == nf['R2_allocate']).sum()), total=n)
Seq = np.load(SSL / 'results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz')['S_equal'].astype(float)


def top_by(score, counts):
    f = np.zeros(S_, bool)
    for i in range(n):
        k = int(min(counts[i], ns[i]))
        if k > 0:
            a = off[i]; o_ = np.argsort(-score[a:off[i + 1]], kind='stable'); f[a + o_[:k]] = True
    return f


R2re = top_by(Seq, nf['R1_global'])
out['E_R2_placement_equals_topS'] = dict(steps_equal=int((R2re == D['R2_allocate']).sum()), total=S_)
viol = 0; ties_at_boundary = 0
for i in range(n):
    a, b = off[i], off[i + 1]; f = D['R2_allocate'][a:b]
    if f.any() and (~f).any():
        if Seq[a:b][f].min() < Seq[a:b][~f].max(): viol += 1
        if Seq[a:b][f].min() == Seq[a:b][~f].max(): ties_at_boundary += 1
out['E_R2_monotone_violations'] = viol; out['E_R2_ties_at_boundary'] = ties_at_boundary
tauS = {int(k): float(v) for k, v in json.loads((SSL / 'results/expectation_realization_v1/run_20260927_stage_b_thr/THRESHOLDS.json').read_text(encoding='utf8'))['S_equal'].items()}
zS = np.empty(S_)
for i in range(n):
    s = Seq[off[i]:off[i + 1]]; zS[off[i]:off[i + 1]] = (s - s.mean()) / max(s.std(), 1e-8)
R0re = zS >= np.array([tauS[k] for k in step_fold])
out['E_R0_equal'] = int((R0re == D['R0_frozen']).sum())
out['E_tauS_vs_q80_calib'] = {k: [tauS[k], float(np.quantile(zS[prm_step & (step_fold == (k + 1) % 5)], .8))] for k in range(5)}
lv = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz')
level = lv['level'].astype(float); names11 = list(map(str, lv['channels'])); drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
CV = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
prof = np.load(CV / 'profiles.npy').astype(float); pnames = json.loads((CV / 'PROFILE_VALIDATION.json').read_text(encoding='utf8'))['channels']
raw = np.column_stack([level, prof[:, pnames.index('chosen_token_z_despiked')], drv]); names = names11 + ['realized_z', 'realized_drv']
surv = [j for j, c in enumerate(names) if c not in ['energy_innovation', 'top50_js']]; Xs = raw[:, surv]
out['E_rebuild_S_equal_maxabs'] = float(np.max(np.abs(answer_standardize(raw, off)[:, surv].mean(1) - Seq)))
R1re = np.zeros(S_, bool); Gre = np.full(S_, np.nan); marks = {}; tauG = {}
DSE = json.loads((W / 'DS_ESTIMATES.json').read_text(encoding='utf8'))
for k in range(5):
    c = (k + 1) % 5; fitm = ~np.isin(step_fold, [k, c]); calm = prm_step & (step_fold == c); evm = step_fold == k
    mu = Xs[fitm].mean(0); sd = np.maximum(Xs[fitm].std(0), 1e-12); Zg = (Xs - mu) / sd; Gk = Zg.mean(1); Gre[evm] = Gk[evm]
    tauG[k] = float(np.quantile(Gk[calm], .8)); R1re[evm] = Gk[evm] >= tauG[k]
    thr = np.quantile(Zg[fitm], .8, axis=0); marks[k] = (Zg >= thr)
out['E_R1_equal'] = int((R1re == D['R1_global']).sum()); out['E_G_maxabs_vs_saved_f32'] = float(np.nanmax(np.abs(Gre - G)))
out['E_tauG_vs_saved'] = {k: [tauG[k], DSE['folds'][str(k)]['tau_G']] for k in range(5)}
out['E_R1_from_saved_f32G'] = int(((G >= np.array([tauG[k] for k in step_fold])) == D['R1_global']).sum())
out['E_R3_eq_post_gt_half'] = int(((post > 0.5) == D['R3_ds_map']).sum())
cut = np.array([DSE['folds'][str(k)]['r4_cutoff'] for k in step_fold])
out['E_R4_eq_post_ge_cut'] = int(((post >= cut) == D['R4_ds_expected_f1']).sum())
c5 = np.floor(np.bincount(aid, weights=post, minlength=n) + 0.5)
out['E_R5_counts_eq_f32post'] = int((np.minimum(c5, ns) == nf['R5_ds_count']).sum())
out['E_R5_placement_eq_topS'] = int((top_by(Seq, nf['R5_ds_count']) == D['R5_ds_count']).sum())
out['E_mark_rate_fit_maxdiff'] = max(float(np.max(np.abs(marks[k][~np.isin(step_fold, [k, (k + 1) % 5])].mean(0) - np.array(DSE['folds'][str(k)]['mark_rate_fit'])))) for k in range(5))
np.savez_compressed('work.npz', lab_meta=lab_meta, zS=zS, Gre=Gre, Seq=Seq, **{f'marks{k}': marks[k] for k in range(5)})
dflt = lambda x: x.item() if isinstance(x, np.generic) else str(x)
json.dump(out, open('audit_out.json', 'w'), indent=1, default=dflt)
print(json.dumps(out, indent=1, default=dflt))
