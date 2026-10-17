"""expectation_realization_v1 stage B: a decision rule built from the binary classifier properties, with the
groups discovered from the data instead of declared.  Frozen protocol: results/expectation_realization_v1/
PROTOCOL_STAGE_B.json.  Chain (label-free, PRMBench fit-fold steps): random-tie top-20% marks -> Dawid-Skene
filter (keep pi_hat > 0.5) -> L-SML partition of the survivors on their marks -> continuous group scores and
binary group representatives -> Dawid-Skene on the representatives -> maximum-likelihood vote weights applied
to the continuous group scores (G_sml).  Amendment B1 (after the fold-0 smoke): the same rule on the partition of
the stage-A tail recipe (tie-aware marks, all fit-fold steps), arms G1_*.  Controls: G_equal, C_sml, S_equal, S_lsml, G_oracle (label-using
diagnostic).  References are refitted and must equal the stage-A scores.  Smoke: ER_FOLDS=0 ER_DRAWS=2000.

Role separation (stage-A amendment A1): every fold-k model writes ONLY its evaluation rows; the PRMScore
threshold of fold k comes from the SAME fold-k model's calibration-fold scores; a failed fit leaves its rows
unscored (no fallback); paired contrasts use the folds both arms cover.
"""
from pathlib import Path
import hashlib, json, os, pickle, shutil, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
DEPTH = MAIN / '.worktrees/depth-feature-fusion-v1'; sys.path.insert(0, str(DEPTH))
from spectral_utils.lsml_gate_locator_research import FusionRecipe, answer_standardize, fit_fusion_weights  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC  # noqa: E402
import er_stage_a as SA  # noqa: E402
import er_stage_b as SB  # noqa: E402
TPFW = MAIN / '.worktrees/token-probability-fusion-v1'

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927_stage_b'
STAGE = ROOT / 'results/expectation_realization_v1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
A_RUN = STAGE / 'run_20260927'
P = json.loads((STAGE / 'PROTOCOL_STAGE_B.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_FOLDS', 'ER_DRAWS'] if k in os.environ}
FOLDS = [int(x) for x in SMOKE['ER_FOLDS'].split(',')] if 'ER_FOLDS' in SMOKE else list(range(5))
DRAWS = int(SMOKE.get('ER_DRAWS', 100_000)); SEED = 20260927; FIT_SEED = 20260919
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
TPF = TPFW / 'results/token_probability_fusion_v1'
CT7P = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'ct7_profiles': CT7P / 'profiles.npy', 'ct7_profile_validation': CT7P / 'PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'joined_records': TPFW / 'results/localization_full_benchmark_v3/evaluation/JOINED.json'}
T0 = time.perf_counter(); timing = {}
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE}; dump(OUT / 'RUN_STATUS.json', status)
checks = {}
def _crash(et, ev, tb):
    import traceback; traceback.print_exception(et, ev, tb)
    if status.get('status') == 'RUNNING':
        status.update({'status': 'CRASHED', 'reason': repr(ev), 'checks': checks, 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
sys.excepthook = _crash
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks, 'finished': datetime.now().isoformat(timespec='seconds')}); dump(OUT / 'RUN_STATUS.json', status)
    raise SystemExit('HARD STOP: ' + reason)

# ------------------------------------------------------------------ inputs must be the stage-A inputs
manA = json.loads((A_RUN / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
hashes = {k: sha(v) for k, v in INPUTS.items()}
checks['input_hashes_equal_stage_A'] = all(hashes[k] == manA[k]['sha256'] for k in INPUTS)
if not checks['input_hashes_equal_stage_A']: hard_stop('input hashes differ from stage A')

# ------------------------------------------------------------------ population frame (as stage A)
ans = pd.read_csv(INPUTS['oof_answers'], encoding='utf-8-sig'); Zs = np.load(INPUTS['oof_step_scores'])
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta_path = Path(freeze['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
classification = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
prm_steps = np.repeat(prm, ns); step_answer = np.repeat(np.arange(n), ns); step_pos = (np.arange(S) - off[step_answer]).astype(float)
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

# ------------------------------------------------------------------ channels (as stage A)
t = time.perf_counter()
lv = np.load(INPUTS['level_bank']); level = lv['level'].astype(float); names11 = list(map(str, lv['channels'])); assert level.shape == (S, 11)
drv = lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)
prof = np.load(INPUTS['ct7_profiles']).astype(float); pnames = json.loads(INPUTS['ct7_profile_validation'].read_text(encoding='utf8'))['channels']; assert prof.shape == (S, 7)
checks['ct7_profile_mean_vs_oof_ct7'] = float(np.abs(prof.mean(1) - Zs['ct7']).max())
if checks['ct7_profile_mean_vs_oof_ct7'] >= 1e-12: hard_stop('ct7 profile order')
rz = prof[:, pnames.index('chosen_token_z_despiked')]
names = names11 + ['realized_z', 'realized_drv']; raw = np.column_stack([level, rz, drv]); assert np.isfinite(raw).all()
values = answer_standardize(raw, off); assert np.isfinite(values).all()
if names != manA['channels']: hard_stop('channel order differs from stage A')
lab = np.asarray(manA['block_labels'], int); assert np.bincount(lab).tolist() == [5, 5, 3]
A0 = names.index('q15_H1'); assert A0 == 0
w421 = np.array([{'H0lim': 1 / 12, 've0': 1 / 12, 've0.75': 1 / 12, 've1': 1 / 12, 'H0lim_prefix_innovation': 1 / 6, 'bocpd_residual': 1 / 6, 'chosen_token_z_despiked': 1 / 3}[c] for c in pnames])
fam421 = answer_standardize(prof, off) @ w421
KEYS = {'main': (np.random.default_rng(20260928).random((S, 13)), np.random.default_rng(20260929).random((S, 13)))}
for tag, sd in [('tie2', 20260930), ('tie3', 20260931)]:
    rg = np.random.default_rng(sd); KEYS[tag] = (rg.random((S, 13)), rg.random((S, 13)))
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))
VOTES = {tag: SB.random_tie_marks(values, off, .2, KEYS[tag][0]) for tag in ('main', 'tie2', 'tie3')}
checks['marks_exact_count'] = all(marks_ok(v) for v in VOTES.values())
if not checks['marks_exact_count']: hard_stop('mark counts')
timing['channels_s'] = time.perf_counter() - t
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': hashes[k]} for k, v in INPUTS.items()} | {'channels': names, 'declared_blocks_reference_only': lab.tolist(),
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/er_stage_b_run.py', ROOT), ('scripts/experiments/er_stage_b.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/prmbench.py', DEPTH),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL_STAGE_B.json')})
print(f'channels ready ({timing["channels_s"]:.0f}s); checks {checks}', flush=True)

# ------------------------------------------------------------------ the label-free chain
from scipy.stats import spearmanr  # noqa: E402
from calfix_common import tail_marks  # noqa: E402
# partition recipes: 'G' = frozen (random-tie binary marks, PRMBench fit-fold steps); 'G1' = amendment B1 (tie-aware centred marks, all fit-fold steps)
GP = ('G', 'G1')
STAGEB = ['G_sml', 'G_equal', 'C_sml', 'S_equal', 'S_lsml', 'G_oracle', 'G1_sml', 'G1_equal', 'G1_oracle']
def ds_meta(est): return {kk: est.get(kk) for kk in ('converged', 'selected_start', 'boundary_emissions', 'orientation', 'prevalence')}
def group_rule(V, surv, gs, key_g, pf, p, arms, d):
    """Steps 4-6 for one partition gs (labels 0..G-1) of the survivors; arm names prefixed p."""
    G = int(gs.max()) + 1
    d[f'{p}_partition'] = gs.tolist(); d[f'{p}_members'] = [[names[surv[j]] for j in np.flatnonzero(gs == g)] for g in range(G)]
    Z = SB.group_scores(V[:, surv], off, gs, answer_standardize)
    arms[f'{p}_equal'] = Z @ np.full(G, 1 / G)
    gv = SB.random_tie_marks(Z, off, .2, key_g[:, :G])
    if not marks_ok(gv): hard_stop('group mark counts')
    truG = SA.truth(gv[pf], labels[pf]); wO = SB.mle_weights(truG['psi'], truG['eta']); d[f'{p}_truth'] = truG; d[f'{p}_w_oracle'] = wO
    if wO.sum() > 0: arms[f'{p}_oracle'] = Z @ (wO / wO.sum())                  # label-using diagnostic
    if G < 3: d[f'{p}_sml_failure'] = f'{G} groups'; return
    try: estG = SA.em_estimate(gv[pf], 'ds')
    except Exception as e: d[f'{p}_sml_failure'] = 'DS on groups: ' + repr(e); return
    wG = SB.mle_weights(estG['psi'], estG['eta'])
    d[f'{p}_est'] = estG; d[f'{p}_ds'] = ds_meta(estG); d[f'{p}_w_est'] = wG; d[f'{p}_bar'] = SB.group_bar(estG, truG)
    d[f'{p}_weight_spearman'] = float(spearmanr(wG, wO).statistic)
    if wG.sum() > 0: arms[f'{p}_sml'] = Z @ (wG / wG.sum())
    else: d[f'{p}_sml_failure'] = 'all group weights 0'
def chain(V, votes, Tta, key_g, pf, fit_rows, tag, k):
    """Label-free chain; returns full-row scores of the stage-B arms and diagnostics.  Labels enter only the
    truth rows (diagnostic) and the *_oracle arms (label-using diagnostics, fit-fold labels only)."""
    arms = {}; d = {'fold': k, 'bank': tag}
    tru13 = SA.truth(votes[pf], labels[pf]); d['truth13'] = tru13; d['truly_anti'] = [names[j] for j in range(13) if tru13['pi'][j] <= 0.5]
    try: est13 = SA.em_estimate(votes[pf], 'ds')
    except Exception as e: d['failure'] = 'DS on 13 channels: ' + repr(e); return arms, d
    w13 = SB.mle_weights(est13['psi'], est13['eta']); d['est13'] = est13; d['ds13'] = ds_meta(est13); d['w_channel'] = w13
    surv = np.flatnonzero(est13['pi'] > 0.5); d['survivors'] = [names[j] for j in surv]; d['dropped'] = [names[j] for j in range(13) if j not in surv]
    if len(surv) < 3: d['failure'] = f'{len(surv)} survivors'; return arms, d
    anchor_s = int(np.flatnonzero(surv == A0)[0]) if A0 in surv else int(np.argmax(est13['pi'][surv])); d['anchor'] = names[surv[anchor_s]]
    w_s = np.zeros(13); w_s[surv] = 1 / len(surv); arms['S_equal'] = V @ w_s
    if w13.sum() > 0: arms['C_sml'] = V @ (w13 / w13.sum())
    try:
        wl, ml = fit_fusion_weights(V[fit_rows][:, surv], FusionRecipe(name='S_lsml', members=tuple(names[j] for j in surv), mode='continuous', anchor=anchor_s), seed=FIT_SEED)
        arms['S_lsml'] = V[:, surv] @ wl; d['S_lsml'] = {'weights': wl, 'K': int(ml['K']), 'groups': ml['groups'], 'anchor_flipped': ml['anchor_flipped'], 'small_m_guarded': ml['small_m_guarded']}
    except Exception as e: d['S_lsml_failure'] = repr(e)
    Tc = SB.answer_center((votes > 0).astype(float), off)
    for p, T, rr in (('G', Tc, pf), ('G1', Tta, fit_rows)):
        try: d[f'{p}_partition13'] = [int(v) for v in TC.lsml_fit_scaled(T[rr], A0, V[rr], standardize=True, loading_scale='unit')['groups']]
        except Exception as e: d[f'{p}_partition13'] = None; d[f'{p}_partition13_error'] = repr(e)
        try: ps = TC.lsml_fit_scaled(T[rr][:, surv], anchor_s, V[rr][:, surv], standardize=True, loading_scale='unit')
        except Exception as e: d[f'{p}_partition_failure'] = repr(e); continue
        gs = np.unique(np.asarray(ps['groups'], int), return_inverse=True)[1]; p13 = d[f'{p}_partition13']
        d[f'{p}_partition_equals_13_restricted'] = None if p13 is None else SB.canonical(np.asarray(p13)[surv]) == SB.canonical(gs)
        group_rule(V, surv, gs, key_g, pf, p, arms, d)
    return arms, d
def reason(a, d):
    p = a.split('_')[0]
    return d.get('failure') or (d.get('S_lsml_failure') if a == 'S_lsml' else None) or d.get(f'{p}_partition_failure') or (d.get(f'{p}_sml_failure') if a.endswith('_sml') else None) or 'not produced (all weights 0)'
CHAIN_KEYS = ['survivors', 'dropped', 'truly_anti', 'anchor', 'ds13', 'w_channel', 'S_lsml', 'S_lsml_failure', 'failure'] + [f'{p}_{s}' for p in GP for s in (
    'partition', 'members', 'partition13', 'partition13_error', 'partition_equals_13_restricted', 'partition_failure', 'ds', 'w_est', 'w_oracle', 'weight_spearman', 'bar', 'sml_failure')]

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
REFS = ['B13_equal', 'B13_block_equal', 'B13_lsml', 'B11_lsml', 'B11_equal', 'ct7', 'fam421', 'step_index']
POS = [a + '_pos' for a in STAGEB] + ['B13_equal_pos']; TIE = [a + s for s in ('_tie2', '_tie3') for a in STAGEB]
ALL = REFS + STAGEB + POS + TIE
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}; fit_log = []; failures = []; diag = []
A_SC = np.load(A_RUN / 'STEP_SCORES.npz'); replay = {}
TTA = tail_marks(values, off, .2, tie_aware=True, centred=True)[0]
def cal_tau(s_full, cal):
    """PRMScore threshold of the fold-k model: 0.8 quantile of its answer-z scores on the calibration fold's PRMBench answers."""
    return float(np.quantile(np.concatenate([zt(s_full[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
def put(m, k, ev_rows, s_full, cal):
    if written[m][ev_rows].any(): raise AssertionError(f'{m}: evaluation rows of fold {k} already written')
    scores[m][ev_rows] = s_full[ev_rows]; written[m][ev_rows] += 1; tau[m][k] = cal_tau(s_full, cal); fitted_folds[m].append(k)
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); pf = fit_rows[prm_steps[fit_rows]]
    # references, refitted exactly as stage A
    X11 = values[:, :11]
    w, m = fit_fusion_weights(X11[fit_rows], FusionRecipe(name='B11_lsml', members=tuple(names11), mode='continuous', anchor=0), seed=FIT_SEED); ref = {'B11_lsml': X11 @ w}
    ref['B11_equal'] = X11 @ np.full(11, 1 / 11); ref['B13_equal'] = values @ np.full(13, 1 / 13); ref['B13_block_equal'] = values @ (1.0 / (3 * np.bincount(lab)[lab]))
    w, m = fit_fusion_weights(values[fit_rows], FusionRecipe(name='B13_lsml', members=tuple(names), mode='continuous', anchor=0), seed=FIT_SEED); ref['B13_lsml'] = values @ w
    fit_log.append({'fold': k, 'arm': 'B13_lsml', 'K': int(m['K']), 'groups': m['groups'], 'weights': np.round(w, 6).tolist()})
    ref['ct7'] = Zs['ct7'].astype(float); ref['fam421'] = fam421; ref['step_index'] = step_pos
    for r, s in ref.items():
        if r in A_SC.files: replay[f'{r}_fold{k}'] = float(np.max(np.abs(s[ev_rows] - A_SC[r][ev_rows])))
        put(r, k, ev_rows, s, cal)
    # stage B on the original bank, on the position-adjusted bank and with two more tie keys
    runs = [('main', values, VOTES['main'], TTA, KEYS['main'][1], '')]
    Vpos = answer_standardize(values - SB.position_profile(values, off, pf), off); vpos = SB.random_tie_marks(Vpos, off, .2, KEYS['main'][0])
    if not marks_ok(vpos): hard_stop('position-bank mark counts')
    runs.append(('pos', Vpos, vpos, tail_marks(Vpos, off, .2, tie_aware=True, centred=True)[0], KEYS['main'][1], '_pos'))
    put('B13_equal_pos', k, ev_rows, Vpos @ np.full(13, 1 / 13), cal)
    runs += [(tg, values, VOTES[tg], TTA, KEYS[tg][1], '_' + tg) for tg in ('tie2', 'tie3')]
    for tag, V, votes, Tta, key_g, sfx in runs:
        arms, d = chain(V, votes, Tta, key_g, pf, fit_rows, tag, k); diag.append(d)
        for a in STAGEB:
            if a in arms: put(a + sfx, k, ev_rows, arms[a], cal)
            else: failures.append({'fold': k, 'arm': a + sfx, 'reason': reason(a, d)})
        fit_log.append({'fold': k, 'arm': 'stage_B_chain' + sfx, **{kk: d.get(kk) for kk in CHAIN_KEYS}})
    dm = [x for x in diag if x['fold'] == k and x['bank'] == 'main'][0]
    msg = f"fold {k}: fit {fitf} cal {cal}; dropped {dm.get('dropped')}; truly anti {dm.get('truly_anti')}; DS13 {dm.get('ds13')}"
    for p in GP:
        pe = round(dm[f'{p}_est']['prevalence'], 3) if f'{p}_est' in dm else None; pt = round(dm[f'{p}_truth']['prevalence'], 3) if f'{p}_truth' in dm else None
        msg += (f"\n   {p}: groups {dm.get(f'{p}_members')}; w_est {np.round(dm.get(f'{p}_w_est', []), 3).tolist()}; w_oracle {np.round(dm.get(f'{p}_w_oracle', []), 3).tolist()}; "
                f"prev est/true {pe}/{pt}; bar {(dm.get(f'{p}_bar') or {}).get('passes')}")
    print(msg + f' ({time.perf_counter()-t:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0
checks['replay_max_abs_diff'] = max(replay.values()); checks['replay_per_ref'] = replay
for m in ALL: checks[f'written_once_{m}'] = bool(written[m].max() <= 1)
checks['replays_pass'] = bool(checks['replay_max_abs_diff'] <= 1e-9 and all(checks[f'written_once_{m}'] for m in ALL))
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)) + '\n')
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in ALL})
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
print('checks:', {k: v for k, v in checks.items() if not k.startswith('written_once') and k != 'replay_per_ref'}, flush=True)
if not checks['replays_pass']: hard_stop('reference replay or write-once check failed')

# ------------------------------------------------------------------ stage-B diagnostics
frows = []; grows = []; parts = {}
for d in diag:
    k, bank = d['fold'], d['bank']
    for j, c in enumerate(names):
        r = {'fold': k, 'bank': bank, 'channel': c, 'psi_true': d['truth13']['psi'][j], 'eta_true': d['truth13']['eta'][j], 'pi_true': d['truth13']['pi'][j]}
        if 'est13' in d: r.update({'psi_hat': d['est13']['psi'][j], 'eta_hat': d['est13']['eta'][j], 'pi_hat': d['est13']['pi'][j], 'kept': c in d.get('survivors', []), 'w_channel': d['w_channel'][j]})
        frows.append(r)
    for p in GP:
        for g, mem in enumerate(d.get(f'{p}_members') or []):
            r = {'fold': k, 'bank': bank, 'recipe': p, 'group': g, 'members': ' + '.join(mem)}
            if f'{p}_truth' in d: r.update({'psi_true': d[f'{p}_truth']['psi'][g], 'eta_true': d[f'{p}_truth']['eta'][g], 'pi_true': d[f'{p}_truth']['pi'][g], 'w_oracle': d[f'{p}_w_oracle'][g]})
            if f'{p}_est' in d: r.update({'psi_hat': d[f'{p}_est']['psi'][g], 'eta_hat': d[f'{p}_est']['eta'][g], 'pi_hat': d[f'{p}_est']['pi'][g], 'w_est': d[f'{p}_w_est'][g]})
            grows.append(r)
    parts[f'{bank}_fold{k}'] = {kk: d.get(kk) for kk in CHAIN_KEYS if kk not in ('w_channel', 'S_lsml')} | {
        'prevalence_true': d['truth13']['prevalence'], 'prevalence_hat_channels': d['est13']['prevalence'] if 'est13' in d else None,
        **{f'{p}_prevalence_hat_groups': d[f'{p}_est']['prevalence'] if f'{p}_est' in d else None for p in GP}, 'channel_bar': SA.bar(d['est13'], d['truth13']) if 'est13' in d else None}
summ = {}
for bank in ('main', 'pos', 'tie2', 'tie3'):
    ds = [d for d in diag if d['bank'] == bank]
    sb = {'survivor_sets_identical': len({tuple(d.get('survivors') or []) for d in ds}) == 1,
          'channel_prev_error': [d['est13']['prevalence'] - d['truth13']['prevalence'] if 'est13' in d else None for d in ds]}
    common = sorted(set.intersection(*[set(d.get('survivors') or []) for d in ds])) if ds else []; sb['common_survivors'] = common
    for p in GP:
        labs = [np.asarray(d[f'{p}_partition'])[[d['survivors'].index(c) for c in common]] for d in ds if d.get(f'{p}_partition') is not None] if len(common) >= 2 else []
        sb[p] = {'partitions_identical_on_common': len({SB.canonical(l) for l in labs}) == 1 if labs else None,
                 'min_pairwise_ari_on_common': min([SB.ari(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]], default=None),
                 'n_groups': [len(d.get(f'{p}_members') or []) for d in ds], 'bar_passes': [(d.get(f'{p}_bar') or {}).get('passes') for d in ds],
                 'prev_error': [d[f'{p}_est']['prevalence'] - d[f'{p}_truth']['prevalence'] if f'{p}_est' in d else None for d in ds],
                 'ds_converged': [(d.get(f'{p}_ds') or {}).get('converged') for d in ds]}
    summ[bank] = sb
parts['summary'] = summ
pd.DataFrame(frows).to_csv(OUT / 'STAGE_B_FILTER.csv', index=False); pd.DataFrame(grows).to_csv(OUT / 'STAGE_B_GROUPS.csv', index=False); dump(OUT / 'STAGE_B_PARTITIONS.json', parts)
print('stage B summary:', json.dumps(summ, default=float)[:2500], flush=True)

# ------------------------------------------------------------------ evaluation (labels enter here only) - as stage A
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
M = pd.DataFrame(metrics); M.to_csv(OUT / 'METRICS.csv', index=False); timing['eval_s'] = time.perf_counter() - t
if FOLDS == list(range(5)):                                                   # stage-A metrics must reproduce on the full run
    MA = pd.read_csv(A_RUN / 'METRICS.csv'); mism = {}
    for r in ['B13_equal', 'B13_block_equal', 'B13_lsml', 'B11_lsml', 'B11_equal', 'ct7', 'fam421']:
        for met in ('within_auc', 'prmscore'):
            a = MA[(MA.method == r) & (MA.metric == met) & (MA.stratum == 'all')].estimate.item(); b = M[(M.method == r) & (M.metric == met) & (M.stratum == 'all')].estimate.item(); mism[f'{r}_{met}'] = abs(a - b)
    checks['stage_A_metrics_max_abs_diff'] = max(mism.values()); checks['stage_A_metrics_diffs'] = mism
    if checks['stage_A_metrics_max_abs_diff'] > 1e-9: hard_stop('stage-A reference metrics not reproduced')

# ------------------------------------------------------------------ paired source-group bootstrap on the folds BOTH arms cover - as stage A
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
prim = [(a, b) for a, b, _ in P['contrasts']['primary']]
amend = [(a, b) for a, b, _ in P['amendments'][0]['contrasts']]; FAM = {p: ('primary', 10) for p in prim} | {p: ('amendment_B1', 6) for p in amend}
sec = [(a, r) for a in STAGEB for r in ['B13_equal', 'B13_block_equal', 'B13_lsml', 'B11_lsml', 'ct7', 'fam421', 'step_index']] + [('S_equal', 'B13_equal'), ('G_oracle', 'G_equal'), ('G_oracle', 'G_sml'),
       ('G_sml_pos', 'B13_equal_pos'), ('G_sml_pos', 'G_equal_pos'), ('S_lsml_pos', 'S_equal_pos'), ('G1_oracle', 'G1_equal'), ('G1_sml_pos', 'B13_equal_pos'), ('G1_sml_pos', 'G1_equal_pos')]
PAIRS = list(dict.fromkeys(prim + amend + sec))
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
dl = {p: {'auc': np.empty(DRAWS), 'ps': np.empty(DRAWS), 'sla': np.empty(DRAWS)} for p in PAIRS if prep[p]}
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
    a, b = p; primary = p in prim; fam, K = FAM.get(p, ('secondary', None)); d = prep[p]
    if d is None:
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'family': fam, 'endpoint': 'all', 'note': 'NOT_ESTIMABLE (no common fitted fold)'}); continue
    pt_auc = (d[a]['auc'].sum() - d[b]['auc'].sum()) / d['cnt'].sum(); pt_ps = float(prmscore_from_counts(*d[a]['conf'].sum(0)) - prmscore_from_counts(*d[b]['conf'].sum(0)))
    hc = d['hcnt'].sum(0); pt_sla = float(np.mean((d[a]['hit'].sum(0) - d[b]['hit'].sum(0))[hc > 0] / hc[hc > 0]))
    for ep, key, pt in [('prm_within_auc', 'auc', pt_auc), ('prmscore', 'ps', pt_ps)]:
        x = dl[p][key]; deltas[f'{a}__minus__{b}__{ep}'] = x.astype(np.float32)
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'family': fam, 'endpoint': ep, 'folds': ','.join(map(str, d['folds'])), 'delta': float(pt), 'ci95_lo': float(np.quantile(x, .025)), 'ci95_hi': float(np.quantile(x, .975)),
                     'ci_adj_lo': float(np.quantile(x, .05 / K / 2)) if K else None, 'ci_adj_hi': float(np.quantile(x, 1 - .05 / K / 2)) if K else None, 'family_K': K, 'B': DRAWS, 'paired_groups': len(Gpr)})
    x = dl[p]['sla']; deltas[f'{a}__minus__{b}__pb_sla'] = x.astype(np.float32)
    pbrows.append({'contrast_id': f'{a} - {b}', 'primary_pair': primary, 'endpoint': 'pb_sla_macro8', 'folds': ','.join(map(str, d['folds'])), 'delta': pt_sla, 'ci95_lo': float(np.nanquantile(x, .025)), 'ci95_hi': float(np.nanquantile(x, .975)), 'B': DRAWS, 'paired_groups': len(Gpb)})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); pd.DataFrame(pbrows).to_csv(OUT / 'PB_CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t; timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILED_FITS', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': failures,
               'checks': {k: v for k, v in checks.items() if not k.startswith('written_once')}, 'fitted_folds': fitted_folds})
dump(OUT / 'RUN_STATUS.json', status)
print(M[M.stratum.isin(['all', 'macro8'])].pivot(index='method', columns='metric', values='estimate').round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.family != 'secondary'][['family', 'contrast_id', 'endpoint', 'folds', 'delta', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string())
print(json.dumps(timing, indent=1)); print('status', status['status'], 'failures', len(failures))
