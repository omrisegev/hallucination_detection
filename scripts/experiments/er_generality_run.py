"""er_generality_v1: does the frozen label-free chain of expectation_realization_v1 stages B/B2 (Dawid-Skene filter on
top-20% marks, then plain averaging, binary-mark or continuous grouping, or L-SML) transfer to larger pre-existing banks
(B20 of Step 438, B32 of Step 439, the 52-channel step pool) with no parameter changed?  Frozen protocol:
results/er_generality_v1/PROTOCOL.json.  The population frame, B13 channels, keys and evaluation block are those of the
reviewed er_stage_b_run.py.  Smoke: ER_FOLDS=0 ER_DRAWS=2000 ER_NULL_PERMS=5.

Role separation (stage-A amendment A1): every fold-k model writes ONLY its evaluation rows; the PRMScore threshold of fold k
comes from the SAME fold-k model's calibration-fold scores; failed fits stay unscored (no fallback).
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

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927'
STAGE = ROOT / 'results/expectation_realization_v1'; GEN = ROOT / 'results/er_generality_v1'; OUT = GEN / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
A_RUN = STAGE / 'run_20260927'
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_FOLDS', 'ER_DRAWS', 'ER_NULL_PERMS'] if k in os.environ}
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
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))
VOTES = {'main': SB.random_tie_marks(values, off, .2, KEYS['main'][0])}
checks['marks_exact_count'] = all(marks_ok(v) for v in VOTES.values())
if not checks['marks_exact_count']: hard_stop('mark counts')
timing['channels_s'] = time.perf_counter() - t
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': hashes[k]} for k, v in INPUTS.items()} | {'channels': names, 'declared_blocks_reference_only': lab.tolist(),
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/er_generality_run.py', ROOT), ('scripts/experiments/er_stage_b2.py', ROOT), ('scripts/experiments/er_stage_b_run.py', ROOT), ('scripts/experiments/er_stage_b.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/prmbench.py', DEPTH),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(GEN / 'PROTOCOL.json')})
print(f'channels ready ({timing["channels_s"]:.0f}s); checks {checks}', flush=True)

# ------------------------------------------------------------------ the larger banks (pre-existing step pool of Step 439)
import er_stage_b2 as C2  # noqa: E402
from calfix_common import tail_marks  # noqa: E402
SCR = Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad')
IND_RUN = ROOT / 'results/indbank_lsml_prmbench_v1'
POOL_IN = {'pool_z': SCR / 'pool_z.npy', 'pool_names': SCR / 'pool_names.json', 'pool_structure': IND_RUN / 'POOL_STRUCTURE.csv'}
manI = json.loads((IND_RUN / 'run_20260927_calfix/INPUT_MANIFEST.json').read_text(encoding='utf8'))
for kk, pth in POOL_IN.items(): hashes[kk] = sha(pth)
checks['pool_hashes_equal_step439'] = all(hashes[kk] == manI[kk]['sha256'] for kk in POOL_IN)
if not checks['pool_hashes_equal_step439']: hard_stop('pool inputs differ from the Step 439 manifest')
dump(OUT / 'POOL_MANIFEST.json', {kk: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': hashes[kk]} for kk, v in POOL_IN.items()})
POOL = np.load(POOL_IN['pool_z']); PN = json.loads(POOL_IN['pool_names'].read_text(encoding='utf8')); assert POOL.shape == (S, 52) and PN[:11] == names11
PS = pd.read_csv(POOL_IN['pool_structure']).set_index('channel')
IND = [c for c in PN[11:] if (not PS.loc[c, 'in_bank11']) and abs(PS.loc[c, 'r_level_marginal']) < 0.35 and PS.loc[c, 'max_r_with_bank11'] < 0.8 and PS.loc[c, 'sd_after_z'] > 0.5]
PI = json.loads((IND_RUN / 'PROTOCOL.json').read_text(encoding='utf8'))
if IND != PI['selected_IND_21']: hard_stop('IND selection differs from Step 439')
lf_sign = np.array([float(np.sign(PS.loc[c, 'r_level_marginal'])) or 1.0 for c in IND])            # label-free orientation of Step 439
LIVE = [j for j in range(POOL.shape[1]) if POOL[:, j].std() > 1e-12]; checks['pool_dead_channels'] = [PN[j] for j in range(POOL.shape[1]) if j not in LIVE]   # hygiene: zero-variance channels
ADD20 = ['ct7_chosen_std_excess', 'ct7_bocpd_residual', 'ct7_H0lim_prefix_innovation', 'ct7_ve0', 'H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk']
BANKS = {'B13': (values, names),
         'B20': (answer_standardize(POOL[:, [PN.index(c) for c in names11 + ADD20]], off), names11 + ADD20),
         'B32': (answer_standardize(np.column_stack([POOL[:, :11], POOL[:, [PN.index(c) for c in IND]] * lf_sign]), off), names11 + ['lf__' + c for c in IND]),
         'B51': (answer_standardize(POOL[:, LIVE], off), [PN[j] for j in LIVE])}
REPLAY_CALFIX = {'B20__ALL_equal': ('bank20_lsml_prmbench_v1', 'B20_equal'), 'B20__ALL_lsml': ('bank20_lsml_prmbench_v1', 'B20_lsml'),
                 'B32__ALL_equal': ('indbank_lsml_prmbench_v1', 'B11_IND_lf_equal'), 'B32__ALL_lsml': ('indbank_lsml_prmbench_v1', 'B11_IND_lf_lsml')}
B_SC_PATH = MAIN / '.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz'   # gitignored; lives in the stage-B worktree
hashes['stage_b_step_scores'] = sha(B_SC_PATH); B_SC = np.load(B_SC_PATH)
MARKS = {}; TTA = {}
t = time.perf_counter()
for bk, (V, nm) in BANKS.items():
    assert np.isfinite(V).all() and nm[0] == 'q15_H1'
    MARKS[bk] = SB.random_tie_marks(V, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
    if not marks_ok(MARKS[bk]): hard_stop(f'mark counts {bk}')
    TTA[bk] = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
checks['B13_marks_equal_stage_B'] = bool(np.array_equal(MARKS['B13'], VOTES['main']))
if not checks['B13_marks_equal_stage_B']: hard_stop('B13 marks differ from stage B')
timing['banks_s'] = time.perf_counter() - t
print('banks ready:', {bk: len(nm) for bk, (V, nm) in BANKS.items()}, f'({timing["banks_s"]:.0f}s)', flush=True)

def simple_filter(V, pf):
    """Label-free control: keep channel j iff corr(channel j, mean of the other channels) >= 0 on the PRMBench fit-fold steps."""
    X = V[pf]; m = X.shape[1]; other = (X.sum(1)[:, None] - X) / (m - 1)
    r = np.array([np.corrcoef(X[:, j], other[:, j])[0, 1] for j in range(m)]); return np.flatnonzero(~(r < 0)), r
def chain(bk, V, nm, votes, Tta, pf, fit_rows, k, pos=False):
    """Frozen stage-B chain on one bank; returns full-row scores and diagnostics (labels only in truth diagnostics)."""
    out = {}; d = {'fold': k, 'bank': bk, 'position_adjusted': pos}; m = V.shape[1]
    out['ALL_equal'] = V @ np.full(m, 1 / m)
    keep_sf, r_sf = simple_filter(V, pf); d['sf_dropped'] = [nm[j] for j in range(m) if j not in keep_sf]; d['sf_r'] = r_sf
    if len(keep_sf) >= 3: out['SF_equal'] = V[:, keep_sf] @ np.full(len(keep_sf), 1 / len(keep_sf))
    tru = SA.truth(votes[pf], labels[pf]); d['truth'] = tru
    try: est = SA.em_estimate(votes[pf], 'ds')
    except Exception as e: d['failure'] = 'DS: ' + repr(e); return out, d
    d['est'] = est; surv = np.flatnonzero(est['pi'] > 0.5); d['survivors'] = [nm[j] for j in surv]; d['dropped'] = [nm[j] for j in range(m) if j not in surv]
    d['truly_anti'] = [nm[j] for j in range(m) if tru['pi'][j] <= 0.5]; d['dropped_good'] = [nm[j] for j in range(m) if j not in surv and tru['pi'][j] >= 0.52]
    d['dropped_flip_informative'] = [nm[j] for j in range(m) if j not in surv and tru['pi'][j] < 0.45]
    if len(surv) < 3: d['failure'] = f'{len(surv)} survivors'; return out, d
    out['DSF_equal'] = V[:, surv] @ np.full(len(surv), 1 / len(surv))
    if pos: return out, d
    if A0 not in surv: hard_stop(f'{bk} fold {k}: anchor q15_H1 filtered out')
    anchor = int(np.flatnonzero(surv == A0)[0]); sn = tuple(nm[j] for j in surv)
    try:
        w, ml = fit_fusion_weights(V[fit_rows][:, surv], FusionRecipe(name='DSF_lsml', members=sn, mode='continuous', anchor=anchor), seed=FIT_SEED)
        out['DSF_lsml'] = V[:, surv] @ w; gc = np.unique(np.asarray(ml['groups'], int), return_inverse=True)[1]
        d['cont_partition'] = [[sn[j] for j in np.flatnonzero(gc == g)] for g in range(gc.max() + 1)]; d['DSF_lsml_weights'] = w
        out['DSF_Gcont'] = SB.group_scores(V[:, surv], off, gc, answer_standardize).mean(1)
    except Exception as e: d['DSF_lsml_failure'] = repr(e)
    try:
        ps = TC.lsml_fit_scaled(Tta[fit_rows][:, surv], anchor, V[fit_rows][:, surv], standardize=True, loading_scale='unit')
        gb = np.unique(np.asarray(ps['groups'], int), return_inverse=True)[1]; d['bin_partition'] = [[sn[j] for j in np.flatnonzero(gb == g)] for g in range(gb.max() + 1)]
        out['DSF_Gbin'] = SB.group_scores(V[:, surv], off, gb, answer_standardize).mean(1)
    except Exception as e: d['DSF_Gbin_failure'] = repr(e)
    return out, d

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
ARMS = ['ALL_equal', 'DSF_equal', 'SF_equal', 'DSF_Gbin', 'DSF_Gcont', 'ALL_lsml', 'DSF_lsml']; POSARMS = ['ALL_equal', 'DSF_equal', 'SF_equal']
REFS = ['ct7', 'fam421', 'step_index']
ALL = REFS + [f'{bk}__{a}' for bk in BANKS for a in ARMS] + [f'{bk}__{a}_pos' for bk in BANKS for a in POSARMS]
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}; fit_log = []; failures = []; diag = []; replay = {}
def cal_tau(s_full, cal):
    """PRMScore threshold of the fold-k model: 0.8 quantile of its answer-z scores on the calibration fold's PRMBench answers."""
    return float(np.quantile(np.concatenate([zt(s_full[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
def put(m, k, ev_rows, s_full, cal):
    if written[m][ev_rows].any(): raise AssertionError(f'{m}: evaluation rows of fold {k} already written')
    scores[m][ev_rows] = s_full[ev_rows]; written[m][ev_rows] += 1; tau[m][k] = cal_tau(s_full, cal); fitted_folds[m].append(k)
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); pf = fit_rows[prm_steps[fit_rows]]
    for r, s in {'ct7': Zs['ct7'].astype(float), 'fam421': fam421, 'step_index': step_pos}.items(): put(r, k, ev_rows, s, cal)
    msg = f'fold {k}: fit {fitf} cal {cal}'
    for bk, (V, nm) in BANKS.items():
        out, d = chain(bk, V, nm, MARKS[bk], TTA[bk], pf, fit_rows, k)
        try:
            w, ml = fit_fusion_weights(V[fit_rows], FusionRecipe(name='ALL_lsml', members=tuple(nm), mode='continuous', anchor=0), seed=FIT_SEED)
            out['ALL_lsml'] = V @ w; d['ALL_lsml_K'] = int(ml['K'])
        except Exception as e: d['ALL_lsml_failure'] = repr(e)
        Vp = answer_standardize(V - SB.position_profile(V, off, pf), off); vp = SB.random_tie_marks(Vp, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
        if not marks_ok(vp): hard_stop(f'position-bank mark counts {bk}')
        outp, dp = chain(bk, Vp, nm, vp, None, pf, fit_rows, k, pos=True)
        diag += [d, dp]
        for a in ARMS:
            if a in out: put(f'{bk}__{a}', k, ev_rows, out[a], cal)
            else: failures.append({'fold': k, 'arm': f'{bk}__{a}', 'reason': d.get('failure') or d.get(f'{a}_failure') or d.get('DSF_lsml_failure') or 'not produced'})
        for a in POSARMS:
            if a in outp: put(f'{bk}__{a}_pos', k, ev_rows, outp[a], cal)
            else: failures.append({'fold': k, 'arm': f'{bk}__{a}_pos', 'reason': dp.get('failure') or 'not produced'})
        if bk == 'B13':
            for mine, theirs in [('DSF_equal', 'S_equal'), ('DSF_lsml', 'S_lsml')]:
                if mine not in out: hard_stop(f'B13 {mine} missing in fold {k}')
                replay[f'B13__{mine}_fold{k}'] = float(np.max(np.abs(out[mine][ev_rows] - B_SC[theirs][ev_rows])))
        fit_log.append({kk: d.get(kk) for kk in ('fold', 'bank', 'survivors', 'dropped', 'truly_anti', 'dropped_good', 'dropped_flip_informative', 'sf_dropped', 'bin_partition', 'cont_partition', 'ALL_lsml_K', 'failure', 'DSF_lsml_failure', 'DSF_Gbin_failure', 'ALL_lsml_failure')}
                       | {'prevalence_hat': d['est']['prevalence'] if 'est' in d else None, 'prevalence_true': d['truth']['prevalence'], 'ds_converged': d['est']['converged'] if 'est' in d else None,
                          'pos_dropped': dp.get('dropped'), 'pos_sf_dropped': dp.get('sf_dropped')})
        msg += (f"\n   {bk}: {len(nm)} ch; DS dropped {d.get('dropped')}; truly anti {d.get('truly_anti')}; dropped good {d.get('dropped_good')}; SF dropped {d.get('sf_dropped')}; "
                f"bin groups {len(d.get('bin_partition') or [])}, cont groups {len(d.get('cont_partition') or [])}")
    print(msg + f' ({time.perf_counter()-t:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0
checks['B13_replay_max_abs_diff'] = max(replay.values()) if replay else None; checks['replay_per_fold'] = replay
for m in ALL: checks[f'written_once_{m}'] = bool(written[m].max() <= 1)
checks['replays_pass'] = bool((not replay or checks['B13_replay_max_abs_diff'] <= 1e-9) and all(checks[f'written_once_{m}'] for m in ALL))
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for r in fit_log: f.write(json.dumps(r, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)) + '\n')
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in ALL}); dump(OUT / 'THRESHOLDS.json', tau)
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
print('checks:', {k: v for k, v in checks.items() if not k.startswith('written_once') and k != 'replay_per_fold'}, flush=True)
if not checks['replays_pass']: hard_stop('B13 replay of stage B or write-once check failed')

# ------------------------------------------------------------------ channel diagnostics (labels for diagnosis only)
crow = []
single = {}
for bk, (V, nm) in BANKS.items():
    single[bk] = [mean_within(V[:, j]) for j in range(V.shape[1])]
for d in diag:
    V, nm = BANKS[d['bank']]
    for j, c in enumerate(nm):
        r = {'bank': d['bank'], 'fold': d['fold'], 'position_adjusted': d['position_adjusted'], 'channel': c, 'pi_true': d['truth']['pi'][j], 'psi_true': d['truth']['psi'][j], 'eta_true': d['truth']['eta'][j],
             'sf_r': d['sf_r'][j], 'sf_kept': c not in d['sf_dropped'], 'single_within_auc_all_answers': single[d['bank']][j]}
        if 'est' in d: r.update({'pi_hat': d['est']['pi'][j], 'psi_hat': d['est']['psi'][j], 'eta_hat': d['est']['eta'][j], 'ds_kept': c in d['survivors']})
        crow.append(r)
pd.DataFrame(crow).to_csv(OUT / 'CHANNELS.csv', index=False)
parts = {}
for bk in BANKS:
    ds = [d for d in diag if d['bank'] == bk and not d['position_adjusted']]
    parts[bk] = {'survivor_sets_identical': len({tuple(d.get('survivors') or []) for d in ds}) == 1,
                 'bin_partition_identical': len({frozenset(frozenset(g) for g in (d.get('bin_partition') or [])) for d in ds}) == 1,
                 'cont_partition_identical': len({frozenset(frozenset(g) for g in (d.get('cont_partition') or [])) for d in ds}) == 1,
                 'bin_equals_cont': [frozenset(frozenset(g) for g in (d.get('bin_partition') or [])) == frozenset(frozenset(g) for g in (d.get('cont_partition') or [])) for d in ds],
                 'n_bin_groups': [len(d.get('bin_partition') or []) for d in ds], 'n_cont_groups': [len(d.get('cont_partition') or []) for d in ds],
                 'dropped': [d.get('dropped') for d in ds], 'dropped_good': [d.get('dropped_good') for d in ds], 'truly_anti': [d.get('truly_anti') for d in ds],
                 'prevalence_hat': [d['est']['prevalence'] if 'est' in d else None for d in ds], 'prevalence_true': [d['truth']['prevalence'] for d in ds],
                 'fold0_bin_partition': ds[0].get('bin_partition') if ds else None, 'fold0_cont_partition': ds[0].get('cont_partition') if ds else None}
dump(OUT / 'PARTITIONS.json', parts)

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
if FOLDS == list(range(5)):                                                   # the larger banks must reproduce their earlier runs (definition check)
    mism = {}
    for mine, (stage, theirs) in REPLAY_CALFIX.items():
        MC = pd.read_csv(ROOT / f'results/{stage}/run_20260927_calfix/METRICS.csv')
        x = MC[(MC.method == theirs) & (MC.metric == 'within_auc') & (MC.stratum == 'all')].estimate.item(); y = M[(M.method == mine) & (M.metric == 'within_auc') & (M.stratum == 'all')].estimate.item()
        mism[mine] = abs(x - y)
    checks['calfix_replay_within_auc_diffs'] = mism; checks['calfix_replay_max'] = max(mism.values())
    if checks['calfix_replay_max'] > 1e-4: hard_stop('B20/B32 do not reproduce the earlier bank runs (definition mismatch)')

# ------------------------------------------------------------------ paired source-group bootstrap (as stage A/B)
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
CT = [(a, b) for a, b, _ in P['contrasts']['primary']]
prim = [(f'{bk}__{a}', f'{bk}__{b}') for bk in P['contrasts']['primary_banks'] for a, b in CT]
sec = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS if bk not in P['contrasts']['primary_banks'] for a, b in CT]
sec += [(f'{bk}__{a}', ref) for bk in BANKS for a in ARMS for ref in ('ct7', 'fam421')]
sec += [(f'{bk}__DSF_equal_pos', f'{bk}__ALL_equal_pos') for bk in BANKS] + [(f'{bk}__DSF_equal_pos', f'{bk}__SF_equal_pos') for bk in BANKS]
PAIRS = list(dict.fromkeys(prim + sec)); K = len(prim) * 2
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
dl = {p: {'auc': np.empty(DRAWS, np.float32), 'ps': np.empty(DRAWS, np.float32), 'sla': np.empty(DRAWS, np.float32)} for p in PAIRS if prep[p]}
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
    if d is None: rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'note': 'NOT_ESTIMABLE (no common fitted fold)'}); continue
    pt_auc = (d[a]['auc'].sum() - d[b]['auc'].sum()) / d['cnt'].sum(); pt_ps = float(prmscore_from_counts(*d[a]['conf'].sum(0)) - prmscore_from_counts(*d[b]['conf'].sum(0)))
    hc = d['hcnt'].sum(0); pt_sla = float(np.mean((d[a]['hit'].sum(0) - d[b]['hit'].sum(0))[hc > 0] / hc[hc > 0]))
    for ep, key, ptv in [('prm_within_auc', 'auc', pt_auc), ('prmscore', 'ps', pt_ps)]:
        x = dl[p][key]; deltas[f'{a}__minus__{b}__{ep}'] = x
        rows.append({'contrast_id': f'{a} - {b}', 'primary': primary, 'endpoint': ep, 'folds': ','.join(map(str, d['folds'])), 'delta': float(ptv), 'ci95_lo': float(np.quantile(x, .025)), 'ci95_hi': float(np.quantile(x, .975)),
                     'ci_adj_lo': float(np.quantile(x, .05 / K / 2)) if primary else None, 'ci_adj_hi': float(np.quantile(x, 1 - .05 / K / 2)) if primary else None, 'family_K': K if primary else None, 'B': DRAWS, 'paired_groups': len(Gpr)})
    x = dl[p]['sla']; deltas[f'{a}__minus__{b}__pb_sla'] = x
    pbrows.append({'contrast_id': f'{a} - {b}', 'primary_pair': primary, 'endpoint': 'pb_sla_macro8', 'folds': ','.join(map(str, d['folds'])), 'delta': pt_sla, 'ci95_lo': float(np.nanquantile(x, .025)), 'ci95_hi': float(np.nanquantile(x, .975)), 'B': DRAWS})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); pd.DataFrame(pbrows).to_csv(OUT / 'PB_CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ nulls for the primary contrasts (PRMBench within-AUC)
t = time.perf_counter(); nulls = {}; NPERM = int(os.environ.get('ER_NULL_PERMS', 200))
for a, b in prim:
    ans_idx = np.flatnonzero([eligible[i] and np.isfinite(scores[a][off[i]:off[i+1]]).all() and np.isfinite(scores[b][off[i]:off[i+1]]).all() for i in range(n)])
    cid = f'{a} - {b}'
    if not len(ans_idx): nulls[cid] = {'note': 'NOT_ESTIMABLE'}; continue
    Rk, loc = C2.within_ranks(np.column_stack([scores[a], scores[b]]), off, ans_idx); yobs = np.concatenate([labels[off[i]:off[i+1]] for i in ans_idx]).astype(float)
    def stat(y): A = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0); return float(A[0] - A[1])
    obs = stat(yobs); nulls[cid] = {'answers': int(len(ans_idx)), 'observed_on_null_set': obs}
    for nm_, fn, sd in [('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)]:
        rg = np.random.default_rng(sd); x = np.array([stat(fn(yobs, loc, rg)) for _ in range(NPERM)])
        nulls[cid][nm_] = {'mean': float(x.mean()), 'sd': float(x.std()), 'p01': float(np.quantile(x, .01)), 'p99': float(np.quantile(x, .99)),
                           'share_ge_observed': float(np.mean(x >= obs)), 'share_le_observed': float(np.mean(x <= obs)), 'permutations': NPERM}
dump(OUT / 'NULLS.json', nulls); timing['nulls_s'] = time.perf_counter() - t
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILED_FITS', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': failures,
               'checks': {k: v for k, v in checks.items() if not k.startswith('written_once')}, 'fitted_folds': fitted_folds})
dump(OUT / 'RUN_STATUS.json', status)
Mx = M[M.stratum.isin(['all', 'macro8'])].pivot(index='method', columns='metric', values='estimate')
print(Mx[['within_auc', 'prmscore', 'sla']].round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.primary == True][['contrast_id', 'endpoint', 'delta', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string())
print(json.dumps(nulls, indent=1)[:3000]); print(json.dumps(timing, indent=1)); print('status', status['status'], 'failures', len(failures))
