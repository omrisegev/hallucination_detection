"""expectation_realization_v1: fuse classifiers of the model's EXPECTATION (bank11: level + change) with
classifiers of the REALIZATION (the written token against a forecast), PRMBench first.
Frozen protocol: results/expectation_realization_v1/PROTOCOL.json.

Fusion code is imported unchanged from depth-feature-fusion-v1 (L-SML, Joint, discovery) and from this
branch (tail recipe of Step 443, stage-A estimators).  Labels enter only the evaluation, stage-A truth
and control blocks.  Smoke overrides: ER_FOLDS=0 ER_DRAWS=2000.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, pickle, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, effective_rank, fit_fusion_weights  # noqa: E402
from spectral_utils.joint_lsml import discover_loao_consensus_groups  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from calfix_common import tail_marks  # noqa: E402
import tail_calib_common as TC  # noqa: E402
import er_stage_a as SA  # noqa: E402
TPFW = MAIN / '.worktrees/token-probability-fusion-v1'
_spec = importlib.util.spec_from_file_location('drv_v1', TPFW / 'spectral_utils/derivative_step_channel_v1.py'); DRV = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(DRV)

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927'
STAGE = ROOT / 'results/expectation_realization_v1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
P = json.loads((STAGE / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_FOLDS', 'ER_DRAWS'] if k in os.environ}
FOLDS = [int(x) for x in SMOKE['ER_FOLDS'].split(',')] if 'ER_FOLDS' in SMOKE else list(range(5))
DRAWS = int(SMOKE.get('ER_DRAWS', 100_000)); SEED = 20260927; FIT_SEED = 20260919
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
TPF = TPFW / 'results/token_probability_fusion_v1'
CT7P = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'token_matrices': TPF / 'TOKEN_MATRICES.npz', 'ct7_profiles': CT7P / 'profiles.npy', 'ct7_profile_validation': CT7P / 'PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'baseline_scores': DEPTH / 'results/step_level_bank_baseline_v1/STEP_SCORES.npz'}
T0 = time.perf_counter(); timing = {}
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE}; dump(OUT / 'RUN_STATUS.json', status)
checks = {}

# ------------------------------------------------------------------ population frame (as declared_joint_run.py)
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
classification = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
prm_steps = np.repeat(prm, ns); step_answer = np.repeat(np.arange(n), ns); step_fold = np.repeat(fold, ns)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])
has_error = np.array([prm[i] and labels[off[i]:off[i+1]].any() for i in range(n)])
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def mean_within(s, mask=None):
    sel = eligible if mask is None else eligible & mask
    return float(np.mean([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) for i in np.flatnonzero(sel)]))

# ------------------------------------------------------------------ channels
t = time.perf_counter()
lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(lv['channels'].astype(str)); assert level.shape == (S, 11)
drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
prof = np.load(INPUTS['ct7_profiles']).astype(float); pnames = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']; assert prof.shape == (S, 7)
checks['ct7_profile_mean_vs_oof_ct7'] = float(np.abs(prof.mean(1) - Zs['ct7']).max()); assert checks['ct7_profile_mean_vs_oof_ct7'] < 1e-12
rz = prof[:, pnames.index('chosen_token_z_despiked')]
names = names11 + ['realized_z', 'realized_drv']; raw = np.column_stack([level, rz, drv]); assert np.isfinite(raw).all()
values = answer_standardize(raw, off); assert np.isfinite(values).all()
CH = P['channels']; blocks = {'level': CH['view_A_expectation']['level'], 'change': CH['view_A_expectation']['change'], 'realization': CH['view_B_realization']['realization']}
lab = np.full(13, -1, int)
for gi, (bn, mem) in enumerate(blocks.items()):
    for c in mem: lab[names.index(c)] = gi
assert (lab >= 0).all() and np.bincount(lab).tolist() == [5, 5, 3]
A0 = names.index('q15_H1'); assert A0 == 0
w421 = np.array([{'H0lim': 1 / 12, 've0': 1 / 12, 've0.75': 1 / 12, 've1': 1 / 12, 'H0lim_prefix_innovation': 1 / 6, 'bocpd_residual': 1 / 6, 'chosen_token_z_despiked': 1 / 3}[c] for c in pnames])
fam421 = answer_standardize(prof, off) @ w421
timing['channels_s'] = time.perf_counter() - t
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in INPUTS.items()} | {'channels': names, 'declared_blocks': {b: m for b, m in blocks.items()}, 'block_labels': lab.tolist(),
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/expectation_realization_run.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/joint_lsml.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/derivative_step_channel_v1.py', TPFW),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
print(f'channels ready ({timing["channels_s"]:.0f}s): {names}; blocks {np.bincount(lab).tolist()}', flush=True)

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
T20, tail_degen = tail_marks(values, off, .2, tie_aware=True, centred=True)
votes, vote_info = SA.binary_votes(values, off, .2)
B13 = ['equal', 'block_equal', 'lsml', 'joint_declared', 'joint_own', 'tail']
scores = {f'B13_{a}': np.full(S, np.nan) for a in B13} | {m: np.full(S, np.nan) for m in ['B11_lsml', 'B11_equal', 'B11_lsml_replay4']}
scores['ct7'] = Zs['ct7'].astype(float); scores['fam421'] = fam421
fit_log = []; failures = []; stageA = {}
def wmeta(w): return {'weights': np.round(w, 6).tolist(), 'weight_ipr': float(1 / np.sum((np.abs(w) / np.abs(w).sum()) ** 2)), 'block_abs_share': [float(np.abs(w)[lab == g].sum() / np.abs(w).sum()) for g in range(3)] if len(w) == 13 else None}
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); cal_rows = rows_of(fold == cal); both = np.concatenate([ev_rows, cal_rows])
    # references on bank11
    X11 = values[:, :11]
    w, m = fit_fusion_weights(X11[fit_rows], FusionRecipe(name='B11_lsml', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED)
    scores['B11_lsml'][both] = X11[both] @ w; scores['B11_equal'][both] = X11[both].mean(1)
    fit_log.append({'fold': k, 'arm': 'B11_lsml', 'K': int(m['K']), 'groups': m['groups'], **wmeta(w), 'members': names11})
    w4, m4 = fit_fusion_weights(X11[rows_of(fold != k)], FusionRecipe(name='replay', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED)
    scores['B11_lsml_replay4'][ev_rows] = X11[ev_rows] @ w4
    # the 13-channel bank
    X = values; arms = {'equal': np.full(13, 1 / 13), 'block_equal': 1.0 / (3 * np.bincount(lab)[lab])}
    try:
        w, m = fit_fusion_weights(X[fit_rows], FusionRecipe(name='B13_lsml', members=tuple(names), mode='continuous', anchor=0), seed=FIT_SEED); arms['lsml'] = w
        fit_log.append({'fold': k, 'arm': 'B13_lsml', 'K': int(m['K']), 'groups': m['groups'], 'residual': m['residual'], 'anchor_flipped': m['anchor_flipped'], 'small_m_guarded': m['small_m_guarded'], **wmeta(w)})
    except Exception as e: failures.append({'fold': k, 'arm': 'B13_lsml', 'reason': repr(e)})
    for arm, grp in [('joint_declared', lab), ('joint_own', None)]:
        try:
            if grp is None:
                td = time.perf_counter()
                disc = discover_loao_consensus_groups(X[fit_rows], step_answer[fit_rows], k_range=(3, 4), seed=FIT_SEED, minimum_group_size=3)
                dmeta = {kk: disc.get(kk) for kk in ('status', 'K', 'group_sizes', 'median_ari', 'mean_ari', 'minimum_ari', 'exact_fraction', 'held_admissible_fraction')}
                dmeta['candidates'] = [{'K': c['K'], 'admissible': c['admissible'], 'group_sizes': c['group_sizes'], 'median_ari': c['median_ari'], 'rejection_reason': c['rejection_reason']} for c in disc['candidates']]
                dmeta['seconds'] = time.perf_counter() - td
                if disc['status'] != 'SELECTED':
                    failures.append({'fold': k, 'arm': 'B13_joint_own', 'reason': disc['status']}); fit_log.append({'fold': k, 'arm': 'B13_joint_own_discovery', **dmeta}); continue
                grp = np.asarray(disc['labels'], int)
            wj, mj = fit_fusion_weights(X[fit_rows], FusionRecipe(name='B13_' + arm, members=tuple(names), mode='joint', groups=tuple(int(v) for v in grp), anchor=0), seed=FIT_SEED)
            if not mj['converged']: failures.append({'fold': k, 'arm': 'B13_' + arm, 'reason': f'joint not converged, starts={mj["converged_starts"]}'})
            arms[arm] = wj
            fit_log.append({'fold': k, 'arm': 'B13_' + arm, 'groups': [int(v) for v in grp], 'group_members': {int(g): [names[j] for j in np.flatnonzero(grp == g)] for g in np.unique(grp)}, 'converged': bool(mj['converged']),
                            'converged_starts': int(mj['converged_starts']), 'relative_offdiag_misfit': float(mj['relative_offdiag_misfit']), 'anchor_flipped': mj['anchor_flipped'],
                            'cross_group_weights': [float(v) for v in mj['joint_weight_meta']['cross_group_weights']], **wmeta(wj), **({'discovery': dmeta} if arm == 'joint_own' else {})})
        except Exception as e: failures.append({'fold': k, 'arm': 'B13_' + arm, 'reason': repr(e)})
    try:
        ft = TC.lsml_fit_scaled(T20[fit_rows], A0, X[fit_rows], standardize=True, loading_scale='unit'); arms['tail'] = np.asarray(ft['weights'], float)
        fit_log.append({'fold': k, 'arm': 'B13_tail', 'K': ft['K'], 'groups': ft['groups'].tolist(), 'anchor_flipped': ft['anchor_flipped'], 'grouping_degenerate': ft['grouping_degenerate'], **wmeta(arms['tail'])})
    except Exception as e: failures.append({'fold': k, 'arm': 'B13_tail', 'reason': repr(e)})
    for arm, ww in arms.items(): scores[f'B13_{arm}'][both] = X[both] @ ww
    # stage A: label-free estimates on the PRMBench fit-fold steps vs the truth on the same steps
    pf = fit_rows[prm_steps[fit_rows]]; pe = ev_rows[prm_steps[ev_rows]]
    V = votes[pf]; tru = SA.truth(V, labels[pf]); tru_ev = SA.truth(votes[pe], labels[pe])
    res = {'rows': int(len(pf)), 'truth': tru, 'truth_eval_fold': tru_ev}
    for est_name, fn in [('DS', lambda: SA.em_estimate(V, 'ds')), ('HEM', lambda: SA.em_estimate(V, 'hem', groups=lab))]:
        try: res[est_name] = fn(); res[est_name]['bar'] = SA.bar(res[est_name], tru)
        except Exception as e: res[est_name] = {'error': repr(e)}; failures.append({'fold': k, 'arm': 'stageA_' + est_name, 'reason': repr(e)})
    b_hat = 2 * res['DS']['prevalence'] - 1 if 'prevalence' in res.get('DS', {}) else None
    res['SML'] = SA.sml_estimate(V, anchor=A0, b_hat=b_hat); res['SML']['bar'] = SA.bar(res['SML'], tru, prev_tol=None) if b_hat is not None else None
    stageA[k] = res
    print(f'fold {k}: fit {fitf} cal {cal}; ' + '; '.join(f'{r["arm"]} K={r.get("K", len(set(r.get("groups", []))))}' for r in fit_log if r['fold'] == k)
          + f"; stageA pass DS={res.get('DS', {}).get('bar', {}).get('passes')} HEM={res.get('HEM', {}).get('bar', {}).get('passes')} SML={res['SML']['bar']['passes'] if res['SML']['bar'] else None} ({time.perf_counter()-t:.0f}s)", flush=True)
timing['fit_s'] = time.perf_counter() - T0
base = np.load(INPUTS['baseline_scores']); ev_all = rows_of(np.isin(fold, FOLDS))
checks['B11_replay4_max_abs_diff'] = float(np.nanmax(np.abs(scores['B11_lsml_replay4'][ev_all] - base['continuous'][ev_all])))
checks['ct7_within_auc'] = mean_within(scores['ct7']); checks['fam421_within_auc'] = mean_within(scores['fam421'])
checks['replays_pass'] = bool(checks['B11_replay4_max_abs_diff'] <= 1e-9 and abs(checks['ct7_within_auc'] - 0.7723966352864217) <= 1e-9 and abs(checks['fam421_within_auc'] - 0.780120) <= 5e-4)
print('checks:', checks, flush=True)
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.generic) else str(v)) + '\n')
METHODS = [m for m in scores if m != 'B11_lsml_replay4' and np.isfinite(scores[m][ev_all]).all()]
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in scores})

# ------------------------------------------------------------------ stage A summary
rowsA = []; summA = {}
for k, res in stageA.items():
    for j, c in enumerate(names):
        r = {'fold': k, 'channel': c, 'block': list(blocks)[lab[j]], 'psi_true': res['truth']['psi'][j], 'eta_true': res['truth']['eta'][j], 'pi_true': res['truth']['pi'][j], 'pi_true_eval_fold': res['truth_eval_fold']['pi'][j]}
        for e in ('DS', 'HEM'):
            if 'psi' in res[e]: r.update({f'psi_{e}': res[e]['psi'][j], f'eta_{e}': res[e]['eta'][j], f'pi_{e}': res[e]['pi'][j]})
        if 'pi' in res['SML']: r['pi_SML'] = res['SML']['pi'][j]
        r['t_SML'] = res['SML']['t'][j]; rowsA.append(r)
pd.DataFrame(rowsA).to_csv(OUT / 'STAGE_A_CHANNELS.csv', index=False)
for e in ('SML', 'DS', 'HEM'):
    bars = [stageA[k][e]['bar'] for k in stageA if isinstance(stageA[k].get(e), dict) and stageA[k][e].get('bar')]
    summA[e] = {'folds_evaluated': len(bars), 'passes_all_folds': bool(len(bars) == len(stageA) and all(b['passes'] for b in bars)), 'per_fold': bars}
for e in ('DS', 'HEM'):
    per_block = {}
    for g, bn in enumerate(blocks):
        d = [r for r in rowsA if r['block'] == bn and f'psi_{e}' in r]
        per_block[bn] = {'mae_psi': float(np.mean([abs(r[f'psi_{e}'] - r['psi_true']) for r in d])), 'mae_eta': float(np.mean([abs(r[f'eta_{e}'] - r['eta_true']) for r in d]))} if d else None
    summA[e]['error_by_block'] = per_block
summA['prevalence_true'] = {k: stageA[k]['truth']['prevalence'] for k in stageA}; summA['vote_info'] = vote_info
summA['estimator_status'] = {k: {e: {kk: stageA[k][e].get(kk) for kk in ('converged', 'selected_start', 'boundary_emissions', 'orientation', 'prevalence', 'error')} for e in ('DS', 'HEM')} for k in stageA}
dump(OUT / 'STAGE_A.json', summA)

# ------------------------------------------------------------------ controls
t = time.perf_counter(); ctl = {}
TM = np.load(INPUTS['token_matrices']); toff = TM['token_offsets']; spans = TM['step_spans']; tok = TM['tokens'][:, list(TM['channels'].astype(str)).index('chosen_surprisal')].astype(float)
assert spans.shape[0] == S and np.all(spans[off[:-1], 0] == 0) and np.all(spans[off[1:] - 1, 1] == np.diff(toff))
rep = np.empty(S); shuf = np.empty(S); rng = np.random.default_rng(SEED)
for i in range(n):
    x = tok[toff[i]:toff[i+1], None]; sp = spans[off[i]:off[i+1]]
    rep[off[i]:off[i+1]] = DRV.derivative_step_readout(x, sp)[:, 0]
    shuf[off[i]:off[i+1]] = DRV.derivative_step_readout(x[rng.permutation(len(x))], sp)[:, 0]
ctl['derivative_replay_max_abs_diff'] = float(np.abs(rep - drv).max()); checks['derivative_replay_pass'] = bool(ctl['derivative_replay_max_abs_diff'] <= 1e-9)
ctl['single_channel_within_auc'] = {c: mean_within(values[:, j]) for j, c in enumerate(names)}
ctl['token_shuffle_derivative_within_auc'] = mean_within(answer_standardize(shuf[:, None], off)[:, 0])
loglen = np.log(np.maximum(spans[:, 1] - spans[:, 0], 1)).astype(float); lenc = {}
for c in ['chosen_surprisal', 'realized_z', 'realized_drv']:
    j = names.index(c); rhos = []; resid = values[:, j].copy()
    for i in np.flatnonzero(prm):
        a, b = off[i:i+2]
        if b - a >= 3 and loglen[a:b].std() > 0 and values[a:b, j].std() > 0: rhos.append(spearmanr(values[a:b, j], loglen[a:b]).statistic)
        if b - a >= 2 and loglen[a:b].std() > 0:
            L = np.column_stack([np.ones(b - a), loglen[a:b]]); resid[a:b] = values[a:b, j] - L @ np.linalg.lstsq(L, values[a:b, j], rcond=None)[0]
    lenc[c] = {'median_within_answer_spearman_log_length': float(np.nanmedian(rhos)), 'answers': len(rhos), 'within_auc_length_residualized': mean_within(resid)}
ctl['length'] = lenc; ctl['tail_mark_degeneracy'] = tail_degen; ctl['checks'] = checks
dump(OUT / 'CONTROLS.json', ctl); timing['controls_s'] = time.perf_counter() - t
print('controls:', json.dumps({k: v for k, v in ctl.items() if k != 'checks'}, default=float)[:1500], flush=True)

# ------------------------------------------------------------------ evaluation (labels enter here only)
t = time.perf_counter()
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
auc = {}; conf = {}; metrics = []; official = {}; cal_thr = {}; pbhit = {}
Gpr, ginv = np.unique(groups[prm], return_inverse=True); prm_pos = np.flatnonzero(prm)
pbe = np.flatnonzero(pb & (target >= 0) & np.isin(fold, FOLDS)); PBC = sorted(set(cells[pb])); Gpb, gpb = np.unique(groups[pbe], return_inverse=True)
for m in METHODS:
    s = scores[m]
    auc[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] and fold[i] in FOLDS else np.nan for i in range(n)])
    v = np.zeros(S, bool); thr = {}
    for k in FOLDS:
        cal = (k + 1) % 5; cs = np.concatenate([zt(s[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))])
        tau = float(np.quantile(cs, .8)) if np.isfinite(cs).all() else np.nan; thr[k] = tau
        for i in np.flatnonzero(prm & (fold == k)):
            a, b = off[i:i+2]; v[a:b] = zt(s[a:b]) < tau
    cal_thr[m] = thr
    ev_prm = [i for i in prm_pos if fold[i] in FOLDS]
    official[m] = prmbench_evaluate([{'idx': ids[i], 'labels': v[off[i]:off[i+1]].astype(int).tolist()} for i in ev_prm], [meta[ids[i]] for i in ev_prm])['total'] if all(np.isfinite(list(thr.values()))) else None
    good = ~labels; c = np.zeros((len(Gpr), 4))
    for gi, i in zip(ginv, prm_pos):
        if not noncontrol[i] or fold[i] not in FOLDS: continue
        a, b = off[i:i+2]; vv = v[a:b]; gg = good[a:b]
        c[gi] += [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    conf[m] = c
    el = np.isfinite(auc[m]); ps = float(.5 * (official[m]['f1'] + official[m]['negative_f1'])) if official[m] else np.nan
    hit = [labels[off[i] + earliest_argmax(s[off[i]:off[i+1]])] for i in np.flatnonzero(has_error & np.isin(fold, FOLDS))]
    metrics += [{'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'N': int(el.sum()), 'estimate': float(np.nanmean(auc[m]))},
                {'method': m, 'benchmark': 'prm', 'metric': 'prmscore', 'stratum': 'all', 'N': len(ev_prm), 'estimate': ps, 'from_counts': float(prmscore_from_counts(*c.sum(0)))},
                {'method': m, 'benchmark': 'prm', 'metric': 'any_error_hit', 'stratum': 'all', 'N': len(hit), 'estimate': float(np.mean(hit))}]
    for cl in sorted(set(classification[prm]) - {''}):
        sel = (classification == cl) & el; metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'class={cl}', 'N': int(sel.sum()), 'estimate': float(np.nanmean(auc[m][sel])) if sel.any() else np.nan})
    h = np.array([earliest_argmax(s[off[i]:off[i+1]]) == target[i] for i in pbe], float); pbhit[m] = h
    percell = {cc: float(h[cells[pbe] == cc].mean()) for cc in PBC if (cells[pbe] == cc).any()}
    metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': 'macro8', 'N': len(pbe), 'estimate': float(np.mean(list(percell.values())))})
    for cc, vv in percell.items(): metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': cc, 'N': int((cells[pbe] == cc).sum()), 'estimate': vv})
pd.DataFrame(metrics).to_csv(OUT / 'METRICS.csv', index=False); timing['eval_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ paired source-group bootstrap
t = time.perf_counter()
sums = {m: np.zeros(len(Gpr)) for m in METHODS}; cnts = np.zeros(len(Gpr))
for gi, i in zip(ginv, prm_pos):
    if eligible[i] and fold[i] in FOLDS:
        cnts[gi] += 1
        for m in METHODS: sums[m][gi] += auc[m][i]
cell_of = np.array([PBC.index(cc) for cc in cells[pbe]]); hsum = {m: np.zeros((len(Gpb), len(PBC))) for m in METHODS}; hcnt = np.zeros((len(Gpb), len(PBC)))
np.add.at(hcnt, (gpb, cell_of), 1)
for m in METHODS: np.add.at(hsum[m], (gpb, cell_of), pbhit[m])
rng = np.random.default_rng(SEED); est = {m: {'auc': np.empty(DRAWS), 'ps': np.empty(DRAWS)} for m in METHODS}; pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos); W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); den = W @ cnts
    for m in METHODS:
        est[m]['auc'][pos:pos+nb] = (W @ sums[m]) / den
        cc = W @ conf[m]; est[m]['ps'][pos:pos+nb] = prmscore_from_counts(cc[:, 0], cc[:, 1], cc[:, 2], cc[:, 3])
    pos += nb
rng2 = np.random.default_rng(SEED + 1); pbest = {m: np.empty(DRAWS) for m in METHODS}; pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos); W = rng2.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float); dc = W @ hcnt
    for m in METHODS: pbest[m][pos:pos+nb] = np.nanmean(np.divide(W @ hsum[m], dc, out=np.full(dc.shape, np.nan), where=dc > 0), axis=1)
    pos += nb
point = {m: {'auc': float(np.nanmean(auc[m])), 'ps': float(prmscore_from_counts(*conf[m].sum(0))), 'sla': float(np.mean([pbhit[m][cell_of == c].mean() for c in range(len(PBC)) if (cell_of == c).any()]))} for m in METHODS}
prim = [(a, b) for a, b, _ in P['contrasts']['primary']]
sec = [(f'B13_{a}', r) for a in B13 for r in ['B11_lsml', 'B11_equal', 'ct7', 'fam421']] + [('B13_equal', 'B11_equal'), ('B13_block_equal', 'B13_equal')]
K = 10; rows = []; pbrows = []; deltas = {}
for a, b in prim + [x for x in sec if x not in prim]:
    if a not in METHODS or b not in METHODS: continue
    primary = (a, b) in prim
    for ep, key in [('prm_within_auc', 'auc'), ('prmscore', 'ps')]:
        d = est[a][key] - est[b][key]; deltas[f'{a}__minus__{b}__{ep}'] = d.astype(np.float32)
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'delta': point[a][key] - point[b][key], 'ci95_lo': float(np.quantile(d, .025)), 'ci95_hi': float(np.quantile(d, .975)),
                     'ci_adj_lo': float(np.quantile(d, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(d, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': DRAWS, 'paired_groups': len(Gpr)})
    d = pbest[a] - pbest[b]; deltas[f'{a}__minus__{b}__pb_sla'] = d.astype(np.float32)
    pbrows.append({'contrast_id': f'{a} - {b}', 'primary_pair': primary, 'endpoint': 'pb_sla_macro8', 'delta': point[a]['sla'] - point[b]['sla'], 'ci95_lo': float(np.nanquantile(d, .025)), 'ci95_hi': float(np.nanquantile(d, .975)), 'B': DRAWS, 'paired_groups': len(Gpb)})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); pd.DataFrame(pbrows).to_csv(OUT / 'PB_CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
all_checks = checks['replays_pass'] and checks.get('derivative_replay_pass', False)
status.update({'status': 'COMPLETE' if not failures and all_checks else 'INCOMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': len(failures), 'checks': checks, 'methods': METHODS,
               'stage_A_passes_all_folds': {e: summA[e]['passes_all_folds'] for e in ('SML', 'DS', 'HEM')}})
dump(OUT / 'RUN_STATUS.json', status)
M = pd.DataFrame(metrics); print(M[M.stratum.isin(['all', 'macro8'])].pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.primary][['contrast_id', 'endpoint', 'delta', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string())
print(json.dumps({e: {'passes_all_folds': summA[e]['passes_all_folds'], 'spearman': [round(b['spearman_pi'], 3) for b in summA[e]['per_fold']]} for e in ('SML', 'DS', 'HEM')}))
print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
