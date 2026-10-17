"""expectation_realization_v1: fuse classifiers of the model's EXPECTATION (bank11: level + change) with
classifiers of the REALIZATION (the written token against a forecast), PRMBench first.
Frozen protocol: results/expectation_realization_v1/PROTOCOL.json (with amendment A1).

Fusion code is imported unchanged from depth-feature-fusion-v1 (L-SML, Joint, discovery) and from this
branch (tail recipe of Step 443, stage-A estimators).  Labels enter only the evaluation, stage-A truth
and control blocks.  Smoke overrides: ER_FOLDS=0 ER_DRAWS=2000.

Role separation (amendment A1): every fold-k model writes ONLY its evaluation rows, and the PRMScore
threshold of fold k is computed inside the fold loop from the SAME fold-k model's calibration-fold
scores.  A failed or non-converged fit leaves its evaluation rows unscored (no fallback); an arm is
evaluated on the folds where it was fitted, and paired contrasts use the folds both arms cover.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, pickle, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, fit_fusion_weights  # noqa: E402
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
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'baseline_scores': DEPTH / 'results/step_level_bank_baseline_v1/STEP_SCORES.npz',
          'joined_records': TPFW / 'results/localization_full_benchmark_v3/evaluation/JOINED.json'}
T0 = time.perf_counter(); timing = {}
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE}; dump(OUT / 'RUN_STATUS.json', status)
checks = {}
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks, 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
    raise SystemExit('HARD STOP: ' + reason)

# ------------------------------------------------------------------ population frame (as declared_joint_run.py)
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
classification = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
prm_steps = np.repeat(prm, ns); step_answer = np.repeat(np.arange(n), ns)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])
has_error = np.array([prm[i] and labels[off[i]:off[i+1]].any() for i in range(n)])
recs = json.loads(INPUTS['joined_records'].read_text(encoding='utf8'))['records']
checks['joined_uid_order_equals_oof'] = [r['uid'] for r in recs] == ans.uid.astype(str).tolist()
checks['joined_step_counts_equal_oof'] = bool(np.array_equal(np.array([int(r['steps']) for r in recs]), ns))
if not (checks['joined_uid_order_equals_oof'] and checks['joined_step_counts_equal_oof']): hard_stop('answer order')
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def mean_within(s, mask=None):
    sel = eligible if mask is None else eligible & mask
    return float(np.mean([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) for i in np.flatnonzero(sel)]))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def q80_tau(s_full):
    """0.8 quantile of the answer-z PRMBench step scores of the answers selected by the caller."""
    return float(np.quantile(np.concatenate(s_full), .8))

# ------------------------------------------------------------------ channels
t = time.perf_counter()
lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(map(str, lv['channels'])); assert level.shape == (S, 11)
drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
prof = np.load(INPUTS['ct7_profiles']).astype(float); pnames = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']; assert prof.shape == (S, 7)
checks['ct7_profile_mean_vs_oof_ct7'] = float(np.abs(prof.mean(1) - Zs['ct7']).max())
if checks['ct7_profile_mean_vs_oof_ct7'] >= 1e-12: hard_stop('ct7 profile order')
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
# row order of the level bank and of the derivative, proven against the token matrix (amendment A1)
TM = np.load(INPUTS['token_matrices']); toff = TM['token_offsets']; spans = TM['step_spans']; tok_all = TM['tokens']
if not (list(map(str, TM['channels'])) == names11 and spans.shape[0] == S and np.all(spans[off[:-1], 0] == 0) and np.all(spans[off[1:] - 1, 1] == np.diff(toff))): hard_stop('token matrix layout')
lev_rep = np.empty((S, 11)); drv_rep = np.empty(S); shuf = np.empty(S); rng_sh = np.random.default_rng(SEED)
cs_col = names11.index('chosen_surprisal')
for i in range(n):
    x = tok_all[toff[i]:toff[i+1]].astype(float); sp = spans[off[i]:off[i+1]]
    lev_rep[off[i]:off[i+1]] = DRV.level_step_readout(x, sp)
    c1 = x[:, [cs_col]]; drv_rep[off[i]:off[i+1]] = DRV.derivative_step_readout(c1, sp)[:, 0]
    shuf[off[i]:off[i+1]] = DRV.derivative_step_readout(c1[rng_sh.permutation(len(c1))], sp)[:, 0]
checks['level_bank_vs_token_matrix_max_abs_diff'] = float(np.abs(lev_rep - level).max())
checks['derivative_vs_token_matrix_max_abs_diff'] = float(np.abs(drv_rep - drv).max())
if checks['level_bank_vs_token_matrix_max_abs_diff'] > 1e-9 or checks['derivative_vs_token_matrix_max_abs_diff'] > 1e-9: hard_stop('level bank or derivative does not reproduce from the token matrix')
del lev_rep, tok_all
timing['channels_s'] = time.perf_counter() - t
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in INPUTS.items()} | {'channels': names, 'declared_blocks': blocks, 'block_labels': lab.tolist(),
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/expectation_realization_run.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/er_digit_share.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT),
                  ('scripts/experiments/calfix_common.py', ROOT), ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/joint_lsml.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH),
                  ('spectral_utils/prmbench.py', DEPTH), ('spectral_utils/derivative_step_channel_v1.py', TPFW), ('spectral_utils/chosen_token_calibration.py', MAIN / '.worktrees/lsml-ct7-levers-run'),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL.json')})
print(f'channels ready ({timing["channels_s"]:.0f}s): {names}; blocks {np.bincount(lab).tolist()}; checks {checks}', flush=True)

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
T20, tail_degen = tail_marks(values, off, .2, tie_aware=True, centred=True)
votes, vote_info = SA.binary_votes(values, off, .2)
B13 = ['equal', 'block_equal', 'lsml', 'joint_declared', 'joint_own', 'tail']
FITTED = [f'B13_{a}' for a in B13] + ['B11_lsml', 'B11_equal']
ALL = FITTED + ['ct7', 'fam421']; REF = {'ct7': Zs['ct7'].astype(float), 'fam421': fam421}
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}; replay4 = np.full(S, np.nan)
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}
fit_log = []; failures = []; stageA = {}
def wmeta(w): return {'weights': np.round(w, 6).tolist(), 'weight_ipr': float(1 / np.sum((np.abs(w) / np.abs(w).sum()) ** 2)), 'block_abs_share': [float(np.abs(w)[lab == g].sum() / np.abs(w).sum()) for g in range(3)] if len(w) == 13 else None}
def cal_tau(X_or_s, w, cal):
    """PRMScore threshold of the fold-k model: 0.8 quantile of its answer-z scores on the calibration fold's PRMBench answers."""
    segs = []
    for i in np.flatnonzero(prm & (fold == cal)):
        s = X_or_s[off[i]:off[i+1]] @ w if w is not None else X_or_s[off[i]:off[i+1]]; segs.append(zt(s))
    return q80_tau(segs)
def put(m, k, ev_rows, s_ev, t):
    if written[m][ev_rows].any(): raise AssertionError(f'{m}: evaluation rows of fold {k} already written')
    scores[m][ev_rows] = s_ev; written[m][ev_rows] += 1; tau[m][k] = t; fitted_folds[m].append(k)
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k)
    # references on bank11
    X11 = values[:, :11]
    w, m = fit_fusion_weights(X11[fit_rows], FusionRecipe(name='B11_lsml', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED)
    put('B11_lsml', k, ev_rows, X11[ev_rows] @ w, cal_tau(X11, w, cal)); w11e = np.full(11, 1 / 11); put('B11_equal', k, ev_rows, X11[ev_rows] @ w11e, cal_tau(X11, w11e, cal))
    fit_log.append({'fold': k, 'arm': 'B11_lsml', 'K': int(m['K']), 'groups': m['groups'], **wmeta(w), 'members': names11})
    w4, m4 = fit_fusion_weights(X11[rows_of(fold != k)], FusionRecipe(name='replay', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED)
    replay4[ev_rows] = X11[ev_rows] @ w4
    for ref, arr in REF.items(): put(ref, k, ev_rows, arr[ev_rows], cal_tau(arr, None, cal))
    # the 13-channel bank
    X = values; arms = {'equal': np.full(13, 1 / 13), 'block_equal': 1.0 / (3 * np.bincount(lab)[lab])}
    try:
        w, m = fit_fusion_weights(X[fit_rows], FusionRecipe(name='B13_lsml', members=tuple(names), mode='continuous', anchor=0), seed=FIT_SEED); arms['lsml'] = w
        fit_log.append({'fold': k, 'arm': 'B13_lsml', 'K': int(m['K']), 'groups': m['groups'], 'residual': m['residual'], 'anchor_flipped': m['anchor_flipped'], 'small_m_guarded': m['small_m_guarded'], **wmeta(w)})
    except Exception as e: failures.append({'fold': k, 'arm': 'B13_lsml', 'reason': repr(e)})
    for arm, grp in [('joint_declared', lab), ('joint_own', None)]:
        dmeta = None
        try:
            if grp is None:
                td = time.perf_counter()
                disc = discover_loao_consensus_groups(X[fit_rows], step_answer[fit_rows], k_range=(3, 4), minimum_group_size=3)   # other arguments default (protocol)
                dmeta = {kk: disc.get(kk) for kk in ('status', 'K', 'group_sizes', 'median_ari', 'mean_ari', 'minimum_ari', 'exact_fraction', 'held_admissible_fraction')}
                dmeta['candidates'] = [{'K': c['K'], 'admissible': c['admissible'], 'group_sizes': c['group_sizes'], 'median_ari': c['median_ari'], 'rejection_reason': c['rejection_reason']} for c in disc['candidates']]
                dmeta['seconds'] = time.perf_counter() - td
                if disc['status'] != 'SELECTED':
                    failures.append({'fold': k, 'arm': 'B13_joint_own', 'reason': disc['status']}); fit_log.append({'fold': k, 'arm': 'B13_joint_own', 'discovery': dmeta, 'scored': False}); continue
                grp = np.asarray(disc['labels'], int)
            wj, mj = fit_fusion_weights(X[fit_rows], FusionRecipe(name='B13_' + arm, members=tuple(names), mode='joint', groups=tuple(int(v) for v in grp), anchor=0), seed=FIT_SEED)
            rec = {'fold': k, 'arm': 'B13_' + arm, 'groups': [int(v) for v in grp], 'group_members': {int(g): [names[j] for j in np.flatnonzero(grp == g)] for g in np.unique(grp)}, 'converged': bool(mj['converged']),
                   'converged_starts': int(mj['converged_starts']), 'relative_offdiag_misfit': float(mj['relative_offdiag_misfit']), 'anchor_flipped': mj['anchor_flipped'],
                   'cross_group_weights': [float(v) for v in mj['joint_weight_meta']['cross_group_weights']], **wmeta(wj), **({'discovery': dmeta} if dmeta else {})}
            if not mj['converged']:                                        # amendment A1: a non-converged Joint fit is a failed fit, not scored
                failures.append({'fold': k, 'arm': 'B13_' + arm, 'reason': f'joint not converged, starts={mj["converged_starts"]}'}); fit_log.append(rec | {'scored': False}); continue
            arms[arm] = wj; fit_log.append(rec | {'scored': True})
        except Exception as e: failures.append({'fold': k, 'arm': 'B13_' + arm, 'reason': repr(e)})
    try:
        ft = TC.lsml_fit_scaled(T20[fit_rows], A0, X[fit_rows], standardize=True, loading_scale='unit'); arms['tail'] = np.asarray(ft['weights'], float)
        fit_log.append({'fold': k, 'arm': 'B13_tail', 'K': ft['K'], 'groups': ft['groups'].tolist(), 'anchor_flipped': ft['anchor_flipped'], 'grouping_degenerate': ft['grouping_degenerate'], **wmeta(arms['tail'])})
    except Exception as e: failures.append({'fold': k, 'arm': 'B13_tail', 'reason': repr(e)})
    for arm, ww in arms.items(): put(f'B13_{arm}', k, ev_rows, X[ev_rows] @ ww, cal_tau(X, ww, cal))
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
    print(f'fold {k}: fit {fitf} cal {cal}; ' + '; '.join(f'{r["arm"]} K={r.get("K", len(set(r.get("groups", []))))}{"" if r.get("scored", True) else " (NOT SCORED)"}' for r in fit_log if r['fold'] == k)
          + f"; stageA pass DS={res.get('DS', {}).get('bar', {}).get('passes')} HEM={res.get('HEM', {}).get('bar', {}).get('passes')} SML={res['SML']['bar']['passes'] if res['SML']['bar'] else None} ({time.perf_counter()-t:.0f}s)", flush=True)
timing['fit_s'] = time.perf_counter() - T0
base = np.load(INPUTS['baseline_scores']); ev_all = rows_of(np.isin(fold, FOLDS))
checks['B11_replay4_max_abs_diff'] = float(np.nanmax(np.abs(replay4[ev_all] - base['continuous'][ev_all])))
checks['ct7_within_auc'] = mean_within(REF['ct7']); checks['fam421_within_auc'] = mean_within(fam421)
for m in ALL: checks[f'written_once_{m}'] = bool(written[m].max() <= 1)
checks['replays_pass'] = bool(checks['B11_replay4_max_abs_diff'] <= 1e-9 and abs(checks['ct7_within_auc'] - 0.7723966352864217) <= 1e-9 and abs(checks['fam421_within_auc'] - 0.780120) <= 5e-4 and all(checks[f'written_once_{m}'] for m in ALL))
print('checks:', checks, flush=True)
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.generic) else str(v)) + '\n')
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, B11_lsml_replay4=replay4, **{m: scores[m] for m in ALL})
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
if not checks['replays_pass']: hard_stop('replay or write-once check failed')

# ------------------------------------------------------------------ stage A summary
rowsA = []; summA = {}
for k, res in stageA.items():
    for j, c in enumerate(names):
        r = {'fold': k, 'channel': c, 'block': list(blocks)[lab[j]], 'psi_true': res['truth']['psi'][j], 'eta_true': res['truth']['eta'][j], 'pi_true': res['truth']['pi'][j],
             'psi_true_eval_fold': res['truth_eval_fold']['psi'][j], 'eta_true_eval_fold': res['truth_eval_fold']['eta'][j], 'pi_true_eval_fold': res['truth_eval_fold']['pi'][j]}
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
    for bn in blocks:
        d = [r for r in rowsA if r['block'] == bn and f'psi_{e}' in r]
        per_block[bn] = {'mae_psi': float(np.mean([abs(r[f'psi_{e}'] - r['psi_true']) for r in d])), 'mae_eta': float(np.mean([abs(r[f'eta_{e}'] - r['eta_true']) for r in d]))} if d else None
    summA[e]['error_by_block'] = per_block
summA['prevalence_true'] = {k: stageA[k]['truth']['prevalence'] for k in stageA}; summA['prevalence_true_eval_fold'] = {k: stageA[k]['truth_eval_fold']['prevalence'] for k in stageA}; summA['vote_info'] = vote_info
summA['estimator_status'] = {k: {e: {kk: stageA[k][e].get(kk) for kk in ('converged', 'selected_start', 'boundary_emissions', 'orientation', 'prevalence', 'error')} for e in ('DS', 'HEM')} for k in stageA}
dump(OUT / 'STAGE_A.json', summA)

# ------------------------------------------------------------------ controls
t = time.perf_counter(); ctl = {}
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
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
PBC = sorted(set(cells[pb])); cell_idx = np.array([PBC.index(c) if c in PBC else -1 for c in cells])
cov = {m: np.isin(fold, fitted_folds[m]) for m in ALL}
aucA = {}; confA = {}; hitA = {}; metrics = []
for m in ALL:
    s = scores[m]; cv = cov[m]
    aucA[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] and cv[i] else np.nan for i in range(n)])
    confA[m] = np.zeros((n, 4)); valid_flags = {}
    for i in np.flatnonzero(prm & cv):
        a, b = off[i:i+2]; vv = zt(s[a:b]) < tau[m][fold[i]]; gg = ~labels[a:b]; valid_flags[i] = vv
        if noncontrol[i]: confA[m][i] = [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
    hitA[m] = np.array([float(earliest_argmax(s[off[i]:off[i+1]]) == target[i]) if pb[i] and target[i] >= 0 and cv[i] else np.nan for i in range(n)])
    folds_txt = ','.join(map(str, sorted(fitted_folds[m])))
    if not fitted_folds[m]:
        metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'folds': '', 'N': 0, 'estimate': np.nan, 'note': 'NOT_ESTIMABLE'}); continue
    ev_prm = [i for i in np.flatnonzero(prm & cv)]
    off_total = prmbench_evaluate([{'idx': ids[i], 'labels': valid_flags[i].astype(int).tolist()} for i in ev_prm], [meta[ids[i]] for i in ev_prm])['total']
    hit = [labels[off[i] + earliest_argmax(scores[m][off[i]:off[i+1]])] for i in np.flatnonzero(has_error & cv)]
    metrics += [{'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': 'all', 'folds': folds_txt, 'N': int(np.isfinite(aucA[m]).sum()), 'estimate': float(np.nanmean(aucA[m]))},
                {'method': m, 'benchmark': 'prm', 'metric': 'prmscore', 'stratum': 'all', 'folds': folds_txt, 'N': len(ev_prm), 'estimate': float(.5 * (off_total['f1'] + off_total['negative_f1'])), 'from_counts': float(prmscore_from_counts(*confA[m].sum(0)))},
                {'method': m, 'benchmark': 'prm', 'metric': 'any_error_hit', 'stratum': 'all', 'folds': folds_txt, 'N': len(hit), 'estimate': float(np.mean(hit))}]
    for cl in sorted(set(classification[prm]) - {''}):
        sel = (classification == cl) & np.isfinite(aucA[m]); metrics.append({'method': m, 'benchmark': 'prm', 'metric': 'within_auc', 'stratum': f'class={cl}', 'folds': folds_txt, 'N': int(sel.sum()), 'estimate': float(np.nanmean(aucA[m][sel])) if sel.any() else np.nan})
    h = hitA[m]; percell = {c: float(np.nanmean(h[cell_idx == ci])) for ci, c in enumerate(PBC) if np.isfinite(h[cell_idx == ci]).any()}
    metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': 'macro8', 'folds': folds_txt, 'N': int(np.isfinite(h).sum()), 'estimate': float(np.mean(list(percell.values())))})
    for c, v in percell.items(): metrics.append({'method': m, 'benchmark': 'pb', 'metric': 'sla', 'stratum': c, 'folds': folds_txt, 'N': int(np.isfinite(h[cells == c]).sum()), 'estimate': v})
pd.DataFrame(metrics).to_csv(OUT / 'METRICS.csv', index=False); timing['eval_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ paired source-group bootstrap on the folds BOTH arms cover
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
prim = [(a, b) for a, b, _ in P['contrasts']['primary']]
sec = [(f'B13_{a}', r) for a in B13 for r in ['B11_lsml', 'B11_equal', 'ct7', 'fam421']] + [('B13_equal', 'B11_equal'), ('B13_block_equal', 'B13_equal')]
PAIRS = list(dict.fromkeys(prim + sec))
prep = {}
for a, b in PAIRS:
    F = cov[a] & cov[b]; folds_c = sorted(set(fitted_folds[a]) & set(fitted_folds[b]))
    if not F.any(): prep[(a, b)] = None; continue
    e = F & eligible; nc = F & noncontrol; pe = F & pb & (target >= 0)
    d = {'folds': folds_c}
    for m in (a, b):
        d[m] = {'auc': np.bincount(gpr[e], weights=aucA[m][e], minlength=len(Gpr)), 'conf': np.stack([np.bincount(gpr[nc], weights=confA[m][nc, q], minlength=len(Gpr)) for q in range(4)], 1)}
        hs = np.zeros((len(Gpb), len(PBC))); np.add.at(hs, (gpbx[pe], cell_idx[pe]), hitA[m][pe]); d[m]['hit'] = hs
    d['cnt'] = np.bincount(gpr[e], minlength=len(Gpr)).astype(float); hc = np.zeros((len(Gpb), len(PBC))); np.add.at(hc, (gpbx[pe], cell_idx[pe]), 1); d['hcnt'] = hc
    prep[(a, b)] = d
K = 10; dl = {p: {'auc': np.empty(DRAWS), 'ps': np.empty(DRAWS), 'sla': np.empty(DRAWS)} for p in PAIRS if prep[p]}
rng = np.random.default_rng(SEED); rng2 = np.random.default_rng(SEED + 1); pos = 0
while pos < DRAWS:
    nb = min(5000, DRAWS - pos)
    W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); W2 = rng2.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
    for p, d in prep.items():
        if d is None: continue
        a, b = p; den = W @ d['cnt']; hden = W2 @ d['hcnt']
        dl[p]['auc'][pos:pos+nb] = (W @ d[a]['auc'] - W @ d[b]['auc']) / den
        ca, cb = W @ d[a]['conf'], W @ d[b]['conf']; dl[p]['ps'][pos:pos+nb] = prmscore_from_counts(*ca.T) - prmscore_from_counts(*cb.T)
        sa = np.nanmean(np.divide(W2 @ d[a]['hit'], hden, out=np.full(hden.shape, np.nan), where=hden > 0), 1); sb = np.nanmean(np.divide(W2 @ d[b]['hit'], hden, out=np.full(hden.shape, np.nan), where=hden > 0), 1)
        dl[p]['sla'][pos:pos+nb] = sa - sb
    pos += nb
rows = []; pbrows = []; deltas = {}
for p in PAIRS:
    a, b = p; primary = p in prim; d = prep[p]
    if d is None:
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': 'all', 'note': 'NOT_ESTIMABLE (no common fitted fold)'}); continue
    pt_auc = (d[a]['auc'].sum() - d[b]['auc'].sum()) / d['cnt'].sum(); pt_ps = float(prmscore_from_counts(*d[a]['conf'].sum(0)) - prmscore_from_counts(*d[b]['conf'].sum(0)))
    hc = d['hcnt'].sum(0); pt_sla = float(np.mean((d[a]['hit'].sum(0) - d[b]['hit'].sum(0))[hc > 0] / hc[hc > 0]))
    for ep, key, pt in [('prm_within_auc', 'auc', pt_auc), ('prmscore', 'ps', pt_ps)]:
        x = dl[p][key]; deltas[f'{a}__minus__{b}__{ep}'] = x.astype(np.float32)
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'folds': ','.join(map(str, d['folds'])), 'delta': float(pt), 'ci95_lo': float(np.quantile(x, .025)), 'ci95_hi': float(np.quantile(x, .975)),
                     'ci_adj_lo': float(np.quantile(x, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(x, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': DRAWS, 'paired_groups': len(Gpr)})
    x = dl[p]['sla']; deltas[f'{a}__minus__{b}__pb_sla'] = x.astype(np.float32)
    pbrows.append({'contrast_id': f'{a} - {b}', 'primary_pair': primary, 'endpoint': 'pb_sla_macro8', 'folds': ','.join(map(str, d['folds'])), 'delta': pt_sla, 'ci95_lo': float(np.nanquantile(x, .025)), 'ci95_hi': float(np.nanquantile(x, .975)), 'B': DRAWS, 'paired_groups': len(Gpb)})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); pd.DataFrame(pbrows).to_csv(OUT / 'PB_CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILED_FITS', 'finished': datetime.now().isoformat(timespec='seconds'), 'fits': len(fit_log), 'failures': failures, 'checks': checks,
               'fitted_folds': fitted_folds, 'stage_A_passes_all_folds': {e: summA[e]['passes_all_folds'] for e in ('SML', 'DS', 'HEM')}})
dump(OUT / 'RUN_STATUS.json', status)
M = pd.DataFrame(metrics); print(M[M.stratum.isin(['all', 'macro8'])].pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.primary == True][['contrast_id', 'endpoint', 'folds', 'delta', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string())
print(json.dumps({e: {'passes_all_folds': summA[e]['passes_all_folds'], 'spearman': [round(b['spearman_pi'], 3) for b in summA[e]['per_fold']]} for e in ('SML', 'DS', 'HEM')}))
print(json.dumps(status, indent=1, default=str)); print(json.dumps(timing, indent=1))
