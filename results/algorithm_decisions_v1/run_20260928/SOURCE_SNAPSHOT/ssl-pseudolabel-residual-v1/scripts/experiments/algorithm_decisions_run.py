"""algorithm_decisions_v1: decide the open components of the label-free algorithm on 8 banks (B13/B16, B20/B23, B32/B35,
B51/B54 = without/with the three digit features): position handling (P0 raw, P1 learn without position, P2 full),
within-group weights (EQ, SML, HEM) x between-group weights (EQ, SML, DSM, HEM), after the decided stages (DS filter pi_hat > 0.5,
binary-mark partition + absorption merge).  Frozen protocol: results/algorithm_decisions_v1/PROTOCOL.json.
Population frame, banks, marks, folds and evaluation are those of the reviewed lsml_merge_step_run.py.
Smoke: ER_FOLDS=0 ER_DRAWS=2000 ER_NULL_PERMS=5.

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
from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata, trim_mean, spearmanr  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC  # noqa: E402
import er_stage_a as SA  # noqa: E402
import er_stage_b as SB  # noqa: E402
import er_stage_b2 as C2  # noqa: E402
import lsml_merge_step as MS  # noqa: E402
import ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402
TPFW = MAIN / '.worktrees/token-probability-fusion-v1'

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260928'
STAGE = ROOT / 'results/expectation_realization_v1'; GEN = ROOT / 'results/algorithm_decisions_v1'; OUT = GEN / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
A_RUN = STAGE / 'run_20260927'
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_FOLDS', 'ER_DRAWS', 'ER_NULL_PERMS'] if k in os.environ}
PHASE = os.environ.get('ER_PHASE', 'all'); assert PHASE in ('all', 'fit', 'assemble'), PHASE          # fit: one process per bank subset; assemble: the rest
ACTIVE = os.environ['ER_BANKS'].split(',') if 'ER_BANKS' in os.environ else None
FOLDS = [int(x) for x in SMOKE['ER_FOLDS'].split(',')] if 'ER_FOLDS' in SMOKE else list(range(5))
DRAWS = int(SMOKE.get('ER_DRAWS', 50_000)); SEED = 20260927; FIT_SEED = 20260919; NPERM = int(SMOKE.get('ER_NULL_PERMS', 200))
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
TPF = TPFW / 'results/token_probability_fusion_v1'
CT7P = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
INPUTS = {'level_bank': TPF / 'DERIVATIVE_CHANNELS.npz', 'ct7_profiles': CT7P / 'profiles.npy', 'ct7_profile_validation': CT7P / 'PROFILE_VALIDATION.json',
          'oof_answers': R / 'OOF_ANSWERS.csv', 'oof_step_scores': R / 'OOF_STEP_SCORES.npz', 'joined_records': TPFW / 'results/localization_full_benchmark_v3/evaluation/JOINED.json'}
T0 = time.perf_counter(); timing = {}
def jd(v): return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=jd), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
PARTS = OUT / 'parts'; PARTS.mkdir(exist_ok=True)
STATUS_FILE = OUT / 'RUN_STATUS.json' if PHASE != 'fit' else PARTS / f"status_{'_'.join(ACTIVE)}.json"
if STATUS_FILE.exists() and not SMOKE: raise SystemExit(f'{STATUS_FILE} already exists; pass a new run id')
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'run_id': RUN_ID, 'smoke_overrides': SMOKE, 'phase': PHASE, 'banks': ACTIVE}; dump(STATUS_FILE, status)
checks = {}
def _crash(et, ev, tb):
    import traceback; traceback.print_exception(et, ev, tb)
    if status.get('status') == 'RUNNING':
        status.update({'status': 'CRASHED', 'reason': repr(ev), 'checks': checks, 'finished': datetime.now().isoformat(timespec='seconds')}); dump(STATUS_FILE, status)
sys.excepthook = _crash
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks, 'finished': datetime.now().isoformat(timespec='seconds')}); dump(STATUS_FILE, status)
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
A0 = names.index('q15_H1'); assert A0 == 0
w421 = np.array([{'H0lim': 1 / 12, 've0': 1 / 12, 've0.75': 1 / 12, 've1': 1 / 12, 'H0lim_prefix_innovation': 1 / 6, 'bocpd_residual': 1 / 6, 'chosen_token_z_despiked': 1 / 3}[c] for c in pnames])
fam421 = answer_standardize(prof, off) @ w421
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))
timing['channels_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ banks: the four of lsml_merge_step_v1 and their digit extensions
SCR = Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad')
IND_RUN = ROOT / 'results/indbank_lsml_prmbench_v1'
LMS_SC = ROOT / 'results/lsml_merge_step_v1/run_20260928/STEP_SCORES.npz'                      # gitignored, local (sha recorded)
B2_SC = STAGE / 'run_20260927_stage_b2/STEP_SCORES.npz'
EXTRA = {'pool_z': SCR / 'pool_z.npy', 'pool_names': SCR / 'pool_names.json', 'pool_structure': IND_RUN / 'POOL_STRUCTURE.csv',
         'digit_features': ROOT / 'results/digit_family_extension_v1/FEATURES.npz', 'lsml_merge_step_scores': LMS_SC, 'stage_b2_step_scores': B2_SC}
manI = json.loads((IND_RUN / 'run_20260927_calfix/INPUT_MANIFEST.json').read_text(encoding='utf8'))
for kk, pth in EXTRA.items(): hashes[kk] = sha(pth)
checks['pool_hashes_equal_step439'] = all(hashes[kk] == manI[kk]['sha256'] for kk in ('pool_z', 'pool_names', 'pool_structure'))
if not checks['pool_hashes_equal_step439']: hard_stop('pool inputs differ from the Step 439 manifest')
POOL = np.load(EXTRA['pool_z']); PN = json.loads(EXTRA['pool_names'].read_text(encoding='utf8')); assert POOL.shape == (S, 52) and PN[:11] == names11
PS = pd.read_csv(EXTRA['pool_structure']).set_index('channel')
IND = [c for c in PN[11:] if (not PS.loc[c, 'in_bank11']) and abs(PS.loc[c, 'r_level_marginal']) < 0.35 and PS.loc[c, 'max_r_with_bank11'] < 0.8 and PS.loc[c, 'sd_after_z'] > 0.5]
if IND != json.loads((IND_RUN / 'PROTOCOL.json').read_text(encoding='utf8'))['selected_IND_21']: hard_stop('IND selection differs from Step 439')
lf_sign = np.array([float(np.sign(PS.loc[c, 'r_level_marginal'])) or 1.0 for c in IND])
LIVE = [j for j in range(POOL.shape[1]) if POOL[:, j].std() > 1e-12]; checks['pool_dead_channels'] = [PN[j] for j in range(POOL.shape[1]) if j not in LIVE]
ADD20 = ['ct7_chosen_std_excess', 'ct7_bocpd_residual', 'ct7_H0lim_prefix_innovation', 'ct7_ve0', 'H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk']
DNAMES = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
DF = np.load(EXTRA['digit_features']); assert np.array_equal(DF['offsets'], off) and DF['values'].shape == (S, 3)
def digit_std(x, active):
    """= spectral_utils.digit_feature_family.answer_standardize (masked per-answer z-score; inactive or constant -> 0)."""
    x = np.asarray(x, float); out = np.zeros(x.shape)
    for a, b in zip(off[:-1], off[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out
D3 = digit_std(DF['values'], DF['active'].astype(bool)); assert np.isfinite(D3).all()
V20 = answer_standardize(POOL[:, [PN.index(c) for c in names11 + ADD20]], off); n20 = names11 + ADD20
V32 = answer_standardize(np.column_stack([POOL[:, :11], POOL[:, [PN.index(c) for c in IND]] * lf_sign]), off); n32 = names11 + ['lf__' + c for c in IND]
V51 = answer_standardize(POOL[:, LIVE], off); n51 = [PN[j] for j in LIVE]
BANKS = {'B13': (values, names), 'B16': (np.column_stack([values, D3]), names + DNAMES), 'B20': (V20, n20), 'B23': (np.column_stack([V20, D3]), n20 + DNAMES),
         'B32': (V32, n32), 'B35': (np.column_stack([V32, D3]), n32 + DNAMES), 'B51': (V51, n51), 'B54': (np.column_stack([V51, D3]), n51 + DNAMES)}
DIGIT_PAIRS = [('B16', 'B13'), ('B23', 'B20'), ('B35', 'B32'), ('B54', 'B51')]
for bk, (V, nm) in BANKS.items():
    if not (np.isfinite(V).all() and nm[0] == 'q15_H1' and len(set(nm)) == len(nm)): hard_stop(f'bank {bk} malformed')
    if (V.std(0) <= 1e-12).any(): hard_stop(f'bank {bk} has a zero-variance channel')
if PHASE != 'fit': dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': hashes[k]} for k, v in (INPUTS | EXTRA).items()} | {'banks': {bk: nm for bk, (V, nm) in BANKS.items()},
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in ([] if PHASE == 'fit' else [('scripts/experiments/algorithm_decisions_run.py', ROOT), ('scripts/experiments/ds_group_weights.py', ROOT), ('scripts/experiments/lsml_merge_step.py', ROOT),
                  ('scripts/experiments/lsml_merge_step_run.py', ROOT), ('scripts/experiments/er_stage_b2.py', ROOT), ('scripts/experiments/er_stage_b.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT),
                  ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/prmbench.py', DEPTH),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]):
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
if PHASE != 'fit': dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(GEN / 'PROTOCOL.json')})
LMS = np.load(LMS_SC); B2 = np.load(B2_SC)
MARKS = {}; TTA = {}
t = time.perf_counter()
FIT_BANKS = [] if PHASE == 'assemble' else (ACTIVE or list(BANKS))
assert all(b in BANKS for b in FIT_BANKS), FIT_BANKS
if PHASE == 'fit':                                                                            # memory: keep only this process's banks
    BANKS = {b: BANKS[b] for b in FIT_BANKS}; del POOL, V20, V32, V51, D3; import gc; gc.collect()
for bk, (V, nm) in [(b, BANKS[b]) for b in FIT_BANKS]:
    MARKS[bk] = SB.random_tie_marks(V, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
    if not marks_ok(MARKS[bk]): hard_stop(f'mark counts {bk}')
    TTA[bk] = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
KEYG = np.random.default_rng(20260929).random((S, 13))                                        # stage-B2 group-mark key
timing['banks_s'] = time.perf_counter() - t
print('banks ready:', {bk: len(nm) for bk, (V, nm) in BANKS.items()}, f'({timing["banks_s"]:.0f}s); checks {checks}', flush=True)

# ------------------------------------------------------------------ learning the structure (label-free) and scoring
LEVEL = {'q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level'}                  # names for reporting only
WITHIN = ['EQ', 'SML', 'HEM']; BETWEEN = ['EQ', 'SML', 'DSM', 'HEM']; FUSION = [f'{w}_{b}' for w in WITHIN for b in BETWEEN]
def members(g, sn): return [[sn[j] for j in np.flatnonzero(g == h)] for h in range(int(g.max()) + 1)]
def learn(bk, L, nm, votes, T_all, pf, fit_rows, k, tag):
    """Everything label-free that is estimated from bank L (raw or position-adjusted).  Labels enter only d['truth_*']."""
    d = {'bank': bk, 'fold': k, 'learn': tag}; st = {'d': d}
    d['truth_channel'] = SA.truth(votes[pf], labels[pf])
    try: est = SA.em_estimate(votes[pf], 'ds')
    except Exception as e: d['failure'] = 'DS filter: ' + repr(e); return st
    surv = np.flatnonzero(est['pi'] > 0.5); d['est_channel'] = est; d['survivors'] = [nm[j] for j in surv]; st['surv'] = surv
    if len(surv) < 3: d['failure'] = f'{len(surv)} survivors'; return st
    sn = [nm[j] for j in surv]; X = L[:, surv]; Xf = X[fit_rows]
    anchor = int(np.flatnonzero(surv == A0)[0]) if A0 in surv else None; d['anchor_survives'] = anchor is not None
    part_anchor = anchor if anchor is not None else int(np.argmax(est['pi'][surv]))                 # declared fallback (stage-B2 rule); counted in ACTIVITY
    d['partition_anchor_fallback'] = anchor is None; tl = time.perf_counter()
    T = T_all[:, surv] if T_all is not None else tail_marks(X, off, .2, tie_aware=True, centred=True)[0]
    try:
        gb = MS.canon(TC.lsml_fit_scaled(T[fit_rows], part_anchor, Xf, standardize=True, loading_scale='unit')['groups'])
        gbm, seq = MS.absorb_merge(np.corrcoef(T[fit_rows], rowvar=False), gb)
    except Exception as e: d['failure'] = 'partition/merge: ' + repr(e); return st
    G = int(gbm.max()) + 1; st['g'] = gbm; st['gb'] = gb
    d['part_bin'] = members(gb, sn); d['part_binM'] = members(gbm, sn); d['merged'] = not np.array_equal(gb, gbm); d['merge_log'] = seq; d['K'] = G
    d['level_groups_before'] = len({int(gb[j]) for j, c in enumerate(sn) if c in LEVEL}); d['level_groups_after'] = len({int(gbm[j]) for j, c in enumerate(sn) if c in LEVEL})
    within = {'EQ': None}
    try:
        _, mt = lsml_continuous(*[Xf[:, j] for j in range(Xf.shape[1])], groups=gbm, compute_score_matrix=False, small_m_guard=True)
        within['SML'] = [np.asarray(w, float) for _, w in mt['group_weights']]; d['sml_within_guard'] = mt['small_m_guarded']; d['sml_within_flags'] = mt['small_m_flags']
    except Exception as e: d['SML_within_failure'] = repr(e)
    d['t_partition_sml_s'] = time.perf_counter() - tl; tl = time.perf_counter()
    try:
        hem = GW.hem_fit(votes[pf][:, surv], gbm); within['HEM'] = hem['within']; st['hem'] = hem; d['t_hem_s'] = time.perf_counter() - tl
        d['hem'] = {kk: hem[kk] for kk in ('psi', 'eta', 'pi', 'source', 'latent_flipped', 'sizes', 'prevalence', 'converged', 'boundary_emissions')} | {'within_equal_groups': [h for h, w in enumerate(hem['within']) if w is None]}
    except Exception as e: d['HEM_failure'] = repr(e); hem = None
    if anchor is not None:
        try:
            w_ch, ml = fit_fusion_weights(Xf, FusionRecipe(name='lsml', members=tuple(sn), mode='continuous', anchor=anchor, groups=tuple(int(v) for v in gbm)), seed=FIT_SEED)
            st['w_lsml'] = w_ch; d['lsml_fit'] = {'K': int(ml['K']), 'small_m_guarded': ml['small_m_guarded'], 'small_m_flags': ml['small_m_flags'], 'anchor_flipped': ml['anchor_flipped']}
        except Exception as e: d['SML_SML_failure'] = repr(e)
    st['within'] = within; st['between'] = {}; d['between'] = {}; d['group_truth'] = {}
    for W, wl in within.items():
        try: ZL = GW.group_matrix(X, off, gbm, wl, answer_standardize)
        except Exception as e: d[f'{W}_group_failure'] = repr(e); continue
        bw = {'EQ': np.ones(G)}; dd = {}
        if W != 'EQ':                                                                          # label-free: does the weighted group score keep the direction of the group mean?
            ZE = GW.group_matrix(X, off, gbm, None, answer_standardize)
            dd['spearman_vs_group_mean_fit_rows'] = [float(spearmanr(ZL[fit_rows][:, h], ZE[fit_rows][:, h]).statistic) for h in range(G)]
        if W != 'SML':
            if anchor is None: dd['SML_failure'] = 'anchor filtered out'
            else:
                _, sw = sml_fuse_signed(*[ZL[fit_rows][:, h] for h in range(G)], small_m_guard=True); sw = np.asarray(sw, float)
                rho = float(spearmanr(ZL[fit_rows] @ sw, Xf[:, anchor]).statistic)
                if not np.isfinite(rho): dd['SML_failure'] = 'undefined anchor orientation'
                else: bw['SML'] = -sw if rho < 0 else sw; dd['SML_guard'] = G == 3; dd['SML_anchor_flipped'] = rho < 0
        if G >= 3:
            if G > KEYG.shape[1]: hard_stop(f'{bk}: {G} groups exceed the group-mark key')
            gv = SB.random_tie_marks(ZL, off, .2, KEYG[:, :G])
            if not marks_ok(gv): hard_stop('group mark counts')
            truG = SA.truth(gv[pf], labels[pf]); d['group_truth'][W] = truG
            try:
                estG = SA.em_estimate(gv[pf], 'ds'); wd = SB.mle_weights(estG['psi'], estG['eta'])
                dd['DSM_est'] = {kk: estG[kk] for kk in ('psi', 'eta', 'pi', 'prevalence', 'converged')}
                if wd.sum() > 0: bw['DSM'] = wd
                else: dd['DSM_failure'] = 'all group weights 0'
            except Exception as e: dd['DSM_failure'] = repr(e)
            if W == 'EQ': st['oracle_w'] = SB.mle_weights(truG['psi'], truG['eta'])
        else: dd['DSM_failure'] = f'{G} groups'
        if hem is not None:
            wh = GW.mle_group_weights(hem['psi'], hem['eta'])
            if wh.sum() > 0: bw['HEM'] = wh
            else: dd['HEM_between_failure'] = 'all group weights 0'
        st['between'][W] = bw; d['between'][W] = {b: v for b, v in bw.items()} | dd
    return st
def score(st, Sb, with_oracle=False):
    """Scores on bank Sb (raw or position-adjusted) with a structure learned by `learn`."""
    out = {}
    if 'surv' not in st or len(st['surv']) < 3: return out
    X = Sb[:, st['surv']]; out['BASE'] = X @ np.full(X.shape[1], 1 / X.shape[1])
    if 'g' not in st: return out
    for W, bw in st['between'].items():
        Z = GW.group_matrix(X, off, st['g'], st['within'][W], answer_standardize)
        for B, w in bw.items(): out[f'{W}_{B}'] = GW.weighted_group_score(Z, w, signed=(B == 'SML'))
        if W == 'EQ' and with_oracle and 'oracle_w' in st and st['oracle_w'].sum() > 0: out['ORACLE'] = GW.weighted_group_score(Z, st['oracle_w'])
    if 'w_lsml' in st: out['SML_SML'] = X @ st['w_lsml']
    return out

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, learning on the other three
ARMS = ['ALL_equal', 'P0__ORACLE'] + [f'{p}__BASE' for p in ('P0', 'P1', 'P2')] + [f'{p}__{f}' for p in ('P0', 'P1', 'P2') for f in FUSION]
REFS = ['ct7', 'fam421', 'step_index']
ALL = REFS + [f'{bk}__{a}' for bk in BANKS for a in ARMS]
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}; failures = []; activity = []; replay = {}; replay_b2 = {}; logo = []; est_store = {}
def cal_tau(s_full, cal):
    return float(np.quantile(np.concatenate([zt(s_full[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
def put(m, k, ev_rows, s_full, cal):
    if written[m][ev_rows].any(): raise AssertionError(f'{m}: evaluation rows of fold {k} already written')
    if not np.isfinite(s_full).all(): failures.append({'fold': k, 'arm': m, 'reason': 'non-finite scores'}); return
    scores[m][ev_rows] = s_full[ev_rows]; written[m][ev_rows] += 1; tau[m][k] = cal_tau(s_full, cal); fitted_folds[m].append(k)
REPLAY_LMS = {'P0__BASE': 'DSF_equal', 'P0__EQ_EQ': 'DSF_GmB', 'P0__SML_SML': 'DSF_lsml_mB', 'P2__BASE': 'DSF_equal_pos', 'P2__SML_SML': 'DSF_lsml_mB_pos'}
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); pf = fit_rows[prm_steps[fit_rows]]
    if PHASE != 'fit':
        for r, s in {'ct7': Zs['ct7'].astype(float), 'fam421': fam421, 'step_index': step_pos}.items(): put(r, k, ev_rows, s, cal)
    msg = f'fold {k}: fit {fitf} cal {cal}'
    for bk, (V, nm) in [(b, BANKS[b]) for b in FIT_BANKS]:
        tb = time.perf_counter()
        put(f'{bk}__ALL_equal', k, ev_rows, V @ np.full(V.shape[1], 1 / V.shape[1]), cal)
        s0 = learn(bk, V, nm, MARKS[bk], TTA[bk], pf, fit_rows, k, 'raw')
        Vp = answer_standardize(V - SB.position_profile(V, off, pf), off); vp = SB.random_tie_marks(Vp, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
        if not marks_ok(vp): hard_stop(f'position-bank mark counts {bk}')
        s1 = learn(bk, Vp, nm, vp, None, pf, fit_rows, k, 'position_adjusted')
        outs = {'P0': score(s0, V, with_oracle=True), 'P1': score(s1, V), 'P2': score(s1, Vp)}
        for p, o in outs.items():
            for a in ['BASE'] + FUSION + (['ORACLE'] if p == 'P0' else []):
                m = f'{bk}__{p}__{a}'
                if a in o: put(m, k, ev_rows, o[a], cal)
                else: failures.append({'fold': k, 'arm': m, 'reason': {kk: v for kk, v in (s0 if p == 'P0' else s1)['d'].items() if 'failure' in kk} or 'not produced'})
        activity += [s0['d'], s1['d']]; est_store[(bk, k)] = s0['d']
        # ---- replays (definition checks)
        if bk in ('B13', 'B16', 'B20', 'B32', 'B51'):
            for mine, theirs in REPLAY_LMS.items():
                p, a = mine.split('__')
                if a not in outs[p]: hard_stop(f'{bk} {mine} missing in fold {k} (replay arm)')
                replay[f'{bk}__{mine}_fold{k}'] = float(np.max(np.abs(outs[p][a][ev_rows] - LMS[f'{bk}__{theirs}'][ev_rows])))
        if bk == 'B13':
            if 'g' not in s0: hard_stop(f'B13 fold {k}: no partition (stage-B2 replay impossible)')
            sn = s0['d']['survivors']; gbm = s0['g']; gb = s0['gb']
            manual = C2.merge_groups_containing(gb, sn, LEVEL); same = bool(np.array_equal(manual, gbm))
            replay_b2[f'fold{k}'] = {'auto_equals_manual_merge': same}
            if same:
                for mine, theirs in (('EQ_DSM', 'B_sml__merge'), ('ORACLE', 'B_oracle__merge')):
                    if mine not in outs['P0']: hard_stop(f'B13 fold {k}: {mine} missing (stage-B2 replay)')
                    replay_b2[f'fold{k}'][f'{mine}_vs_{theirs}'] = float(np.max(np.abs(outs['P0'][mine][ev_rows] - B2[theirs][ev_rows])))
            if any(not (np.isfinite(v) and v <= 1e-9) for v in replay_b2[f'fold{k}'].values() if not isinstance(v, bool)): hard_stop(f'B13 fold {k}: stage-B2 replay failed {replay_b2[f"fold{k}"]}')
        # ---- leave-one-group-out Dawid-Skene (diagnosis; P0 structure)
        if 'g' in s0:
            est0 = s0['d']['est_channel']; tr0 = s0['d']['truth_channel']; surv0 = s0['surv']; g0 = s0['g']
            for h in range(int(g0.max()) + 1):
                rm = surv0[g0 == h]; keep = np.array([j for j in range(V.shape[1]) if j not in set(rm)])
                rec = {'bank': bk, 'fold': k, 'group': h, 'members': [nm[j] for j in rm], 'size': int(len(rm)), 'prevalence_true': tr0['prevalence'], 'prevalence_before': est0['prevalence']}
                try:
                    e2 = SA.em_estimate(MARKS[bk][pf][:, keep], 'ds')
                    rec |= {'prevalence_after': e2['prevalence'], 'mean_abs_dpi_hat_others': float(np.mean(np.abs(e2['pi'] - est0['pi'][keep]))),
                            'mae_pi_before': float(np.mean(np.abs(est0['pi'][keep] - tr0['pi'][keep]))), 'mae_pi_after': float(np.mean(np.abs(e2['pi'] - tr0['pi'][keep]))),
                            'decisions_changed': int(np.sum((e2['pi'] > .5) != (est0['pi'][keep] > .5)))}
                except Exception as e: rec['failure'] = repr(e)
                logo.append(rec)
        d0 = s0['d']; d1 = s1['d']
        print(f"   fold {k} {bk}: surv {len(d0.get('survivors', []))}/{len(nm)} K {d0.get('K')} merged {d0.get('merged')} | pos surv {len(d1.get('survivors', []))} K {d1.get('K')} merged {d1.get('merged')}"
              f" | prev DS {d0.get('est_channel', {}).get('prevalence', float('nan')):.3f} hem {d0.get('hem', {}).get('prevalence', float('nan')):.3f} true {d0['truth_channel']['prevalence']:.3f}"
              f" | hem {d0.get('t_hem_s', 0):.0f}+{d1.get('t_hem_s', 0):.0f}s ({time.perf_counter()-tb:.0f}s)", flush=True)
    print(msg + f' done ({time.perf_counter()-t:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0
PART_KEYS = ('scores', 'written', 'tau', 'fitted_folds', 'failures', 'activity', 'replay', 'replay_b2', 'logo', 'est_store')
if PHASE == 'fit':
    part = {'banks': FIT_BANKS, 'folds': FOLDS, 'fit_s': timing['fit_s'], 'arms': [m for m in ALL if m.split('__')[0] in FIT_BANKS],
            'scores': {m: scores[m] for m in ALL if m.split('__')[0] in FIT_BANKS}, 'written': {m: written[m] for m in ALL if m.split('__')[0] in FIT_BANKS},
            'tau': {m: tau[m] for m in ALL if m.split('__')[0] in FIT_BANKS}, 'fitted_folds': {m: fitted_folds[m] for m in ALL if m.split('__')[0] in FIT_BANKS},
            'failures': failures, 'activity': activity, 'replay': replay, 'replay_b2': replay_b2, 'logo': logo, 'est_store': est_store}
    with open(PARTS / f"part_{'_'.join(FIT_BANKS)}.pkl", 'wb') as fh: pickle.dump(part, fh, protocol=4)
    status.update({'status': 'FIT_DONE', 'finished': datetime.now().isoformat(timespec='seconds'), 'fit_s': timing['fit_s'], 'failures': failures}); dump(STATUS_FILE, status)
    print('fit phase done', FIT_BANKS, f"{timing['fit_s']:.0f}s"); raise SystemExit(0)
if PHASE == 'assemble':
    got = set(); timing['fit_s_parts'] = {}
    for fpath in sorted(PARTS.glob('part_*.pkl')):
        with open(fpath, 'rb') as fh: part = pickle.load(fh)
        if part['folds'] != FOLDS: hard_stop(f'{fpath.name}: folds {part["folds"]} differ from {FOLDS}')
        if got & set(part['banks']): hard_stop(f'bank fitted twice: {got & set(part["banks"])}')
        got |= set(part['banks']); timing['fit_s_parts'][fpath.name] = part['fit_s']; checks[f'part_sha256_{fpath.name}'] = sha(fpath)
        for m in part['arms']: scores[m] = part['scores'][m]; written[m] = part['written'][m]; tau[m] = part['tau'][m]; fitted_folds[m] = part['fitted_folds'][m]
        failures += part['failures']; activity += part['activity']; replay |= part['replay']; replay_b2 |= part['replay_b2']; logo += part['logo']; est_store |= part['est_store']
    if got != set(BANKS): hard_stop(f'missing bank parts: {set(BANKS) - got}')
    timing['fit_s'] = max(timing['fit_s_parts'].values()) + timing['fit_s']
checks['replay_lsml_merge_step_max'] = max(replay.values()) if replay else None; checks['replay_per_fold'] = replay; checks['replay_stage_b2'] = replay_b2
for m in ALL: checks[f'written_once_{m}'] = bool(written[m].max() <= 1)
checks['replays_pass'] = bool(all(np.isfinite(v) and v <= 1e-9 for v in replay.values()) and all(checks[f'written_once_{m}'] for m in ALL))
with open(OUT / 'ACTIVITY.jsonl', 'w', encoding='utf8') as f:
    for d in activity: f.write(json.dumps({kk: v for kk, v in d.items() if kk not in ('truth_channel', 'est_channel', 'group_truth')}, default=jd) + '\n')
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in ALL}); dump(OUT / 'THRESHOLDS.json', tau)
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
pd.DataFrame(logo).to_csv(OUT / 'LOGO.csv', index=False)
print('checks:', {k: v for k, v in checks.items() if not k.startswith('written_once') and k != 'replay_per_fold'}, flush=True)
if not checks['replays_pass']: hard_stop('replay of lsml_merge_step_v1 or write-once check failed')
checks['reference_full_coverage'] = {bk: sorted(fitted_folds[f'{bk}__P0__BASE']) == sorted(FOLDS) for bk in BANKS}
if not all(checks['reference_full_coverage'].values()): hard_stop('P0__BASE (the selection reference) lacks a fold on some bank')

# ------------------------------------------------------------------ estimates, groups, digit shift (labels for diagnosis only)
erow = []; grow = []; shift = []
for d in activity:
    bk = d['bank']; nm = BANKS[bk][1]
    if 'est_channel' not in d: continue
    e, tr = d['est_channel'], d['truth_channel']
    for j, c in enumerate(nm):
        erow.append({'bank': bk, 'fold': d['fold'], 'learn': d['learn'], 'channel': c, 'pi_hat': e['pi'][j], 'pi_true': tr['pi'][j], 'psi_hat': e['psi'][j], 'psi_true': tr['psi'][j],
                     'eta_hat': e['eta'][j], 'eta_true': tr['eta'][j], 'kept': c in d.get('survivors', []), 'prevalence_hat': e['prevalence'], 'prevalence_true': tr['prevalence']})
    for W, gt in d.get('group_truth', {}).items():
        dsm = d['between'].get(W, {}).get('DSM_est'); hm = d.get('hem')
        for h in range(len(gt['pi'])):
            r = {'bank': bk, 'fold': d['fold'], 'learn': d['learn'], 'within': W, 'group': h, 'members': ' + '.join(d['part_binM'][h]), 'size': len(d['part_binM'][h]),
                 'pi_true_group_mark': gt['pi'][h], 'psi_true_group_mark': gt['psi'][h], 'eta_true_group_mark': gt['eta'][h], 'prevalence_true': gt['prevalence']}
            if dsm: r |= {'DSM_pi': dsm['pi'][h], 'DSM_psi': dsm['psi'][h], 'DSM_eta': dsm['eta'][h], 'DSM_prevalence': dsm['prevalence']}
            if hm and W == 'EQ': r |= {'HEM_pi': hm['pi'][h], 'HEM_psi': hm['psi'][h], 'HEM_eta': hm['eta'][h], 'HEM_source': hm['source'][h], 'HEM_prevalence': hm['prevalence']}
            for B in BETWEEN:
                w = d['between'].get(W, {}).get(B)
                if isinstance(w, np.ndarray) and len(w) == len(gt['pi']): r[f'w_{B}'] = float(w[h] / np.abs(w).sum())
            grow.append(r)
for (bw, bo) in DIGIT_PAIRS:
    for k in FOLDS:
        if (bw, k) not in est_store or (bo, k) not in est_store or 'est_channel' not in est_store[(bw, k)] or 'est_channel' not in est_store[(bo, k)]: continue
        a, b = est_store[(bo, k)], est_store[(bw, k)]; nmo = BANKS[bo][1]
        pa = np.asarray(a['est_channel']['pi']); pb_ = np.asarray(b['est_channel']['pi'])[:len(nmo)]
        shift.append({'with_digits': bw, 'without': bo, 'fold': k, 'prevalence_without': a['est_channel']['prevalence'], 'prevalence_with': b['est_channel']['prevalence'],
                      'prevalence_true': a['truth_channel']['prevalence'], 'mean_dpi_hat_shared': float((pb_ - pa).mean()), 'max_abs_dpi_hat_shared': float(np.abs(pb_ - pa).max()),
                      'shared_decisions_identical': bool(np.array_equal(pb_ > .5, pa > .5)),
                      'spearman_pi_without': float(spearmanr(pa, a['truth_channel']['pi']).statistic), 'spearman_pi_with_shared': float(spearmanr(pb_, np.asarray(b['truth_channel']['pi'])[:len(nmo)]).statistic),
                      'digit_pi_hat': list(np.asarray(b['est_channel']['pi'])[len(nmo):]), 'digit_pi_true': list(np.asarray(b['truth_channel']['pi'])[len(nmo):])})
pd.DataFrame(erow).to_csv(OUT / 'ESTIMATES.csv', index=False); pd.DataFrame(grow).to_csv(OUT / 'GROUPS.csv', index=False); pd.DataFrame(shift).to_csv(OUT / 'SHIFT.csv', index=False)

# ------------------------------------------------------------------ evaluation (labels enter here only) - as stage A
t = time.perf_counter()
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
PBC = sorted(set(cells[pb])); cell_idx = np.array([PBC.index(c) if c in PBC else -1 for c in cells])
cov = {m: np.isin(fold, fitted_folds[m]) for m in ALL}; full = {m: sorted(fitted_folds[m]) == sorted(FOLDS) for m in ALL}
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
WA = M[(M.metric == 'within_auc') & (M.stratum == 'all')].set_index('method').estimate

# ------------------------------------------------------------------ paired source-group bootstrap: per-arm draws on the common full-coverage population
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
FULLCOV = np.isin(fold, FOLDS); e_all = FULLCOV & eligible; nc_all = FULLCOV & noncontrol; pe_all = FULLCOV & pb & (target >= 0)
cntG = np.bincount(gpr[e_all], minlength=len(Gpr)).astype(float); hc = np.zeros((len(Gpb), len(PBC))); np.add.at(hc, (gpbx[pe_all], cell_idx[pe_all]), 1)
FULLARMS = [m for m in ALL if full[m]]
stats = {}
for m in FULLARMS:
    hs = np.zeros((len(Gpb), len(PBC))); np.add.at(hs, (gpbx[pe_all], cell_idx[pe_all]), hitA[m][pe_all])
    stats[m] = {'auc': np.bincount(gpr[e_all], weights=aucA[m][e_all], minlength=len(Gpr)), 'conf': np.stack([np.bincount(gpr[nc_all], weights=confA[m][nc_all, q], minlength=len(Gpr)) for q in range(4)], 1), 'hit': hs}
dr = {m: {'auc': np.empty(DRAWS, np.float32), 'ps': np.empty(DRAWS, np.float32), 'sla': np.empty(DRAWS, np.float32)} for m in FULLARMS}
rng = np.random.default_rng(SEED); rng2 = np.random.default_rng(SEED + 1); pos = 0
while pos < DRAWS:
    nb = min(2500, DRAWS - pos)
    W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); W2 = rng2.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
    den = W @ cntG; hden = W2 @ hc
    for m in FULLARMS:
        st_ = stats[m]; dr[m]['auc'][pos:pos+nb] = (W @ st_['auc']) / den
        dr[m]['ps'][pos:pos+nb] = prmscore_from_counts(*(W @ st_['conf']).T)
        dr[m]['sla'][pos:pos+nb] = np.nanmean(np.divide(W2 @ st_['hit'], hden, out=np.full(hden.shape, np.nan), where=hden > 0), 1)
    pos += nb
def point(m, ep):
    if ep == 'auc': return stats[m]['auc'].sum() / cntG.sum()
    if ep == 'ps': return float(prmscore_from_counts(*stats[m]['conf'].sum(0)))
    hcs = hc.sum(0); return float(np.mean(stats[m]['hit'].sum(0)[hcs > 0] / hcs[hcs > 0]))
def contrast(a, b):
    if a not in dr or b not in dr: return {'contrast_id': f'{a} - {b}', 'note': 'NOT_ESTIMABLE (an arm lacks full coverage)'}
    r = {'contrast_id': f'{a} - {b}'}
    for ep, nm_ in (('auc', 'prm_within_auc'), ('ps', 'prmscore'), ('sla', 'pb_sla_macro8')):
        x = dr[a][ep].astype(float) - dr[b][ep].astype(float)
        r |= {f'{nm_}_delta': point(a, ep) - point(b, ep), f'{nm_}_lo': float(np.nanquantile(x, .025)), f'{nm_}_hi': float(np.nanquantile(x, .975))}
    return r
PAIRS = []
for bk in BANKS:
    base = f'{bk}__P0__BASE'
    PAIRS += [(f'{bk}__{a}', base) for a in ARMS if a != 'P0__BASE']
    for p in ('P1', 'P2'):
        PAIRS += [(f'{bk}__{p}__{f}', f'{bk}__{p}__BASE') for f in FUSION] + [(f'{bk}__{p}__{f}', f'{bk}__P0__{f}') for f in FUSION + ['BASE']]
    PAIRS += [(base, 'ct7'), (base, 'fam421')]
for bw, bo in DIGIT_PAIRS: PAIRS += [(f'{bw}__{a}', f'{bo}__{a}') for a in ARMS]
PAIRS = list(dict.fromkeys(PAIRS)); CON = {f'{a} - {b}': contrast(a, b) for a, b in PAIRS}
timing['bootstrap_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ the frozen selection rule
SEL = P['selection_rule_frozen']; BK = list(BANKS); VARIANTS = [f'{p}__{f}' for p in ('P0', 'P1', 'P2') for f in FUSION] + ['P1__BASE', 'P2__BASE']
table = []
for v in VARIANTS:
    rs = [CON.get(f'{bk}__{v} - {bk}__P0__BASE', {}) for bk in BK]; fullv = all(f'{bk}__{v}' in dr for bk in BK)
    losses = [bk for bk, r in zip(BK, rs) if fullv and r['prm_within_auc_hi'] < 0]; wins = [bk for bk, r in zip(BK, rs) if fullv and r['prm_within_auc_lo'] > 0]
    table.append({'variant': v, 'full_coverage': fullv, 'mean_within_auc_8banks': float(np.mean([WA[f'{bk}__{v}'] for bk in BK])) if fullv else None,
                  'mean_delta_vs_P0_BASE': float(np.mean([r['prm_within_auc_delta'] for r in rs])) if fullv else None, 'wins': wins, 'losses': losses,
                  'eligible': fullv and not losses, 'qualifies': fullv and not losses and len(wins) >= 5})
TB = pd.DataFrame(table).sort_values('mean_within_auc_8banks', ascending=False)
qual = TB[TB.qualifies]; elig = TB[TB.eligible]
candidate = qual.iloc[0].variant if len(qual) else 'P0__BASE'
best_eligible = elig.iloc[0].variant if len(elig) else None; best_overall = TB.iloc[0].variant
selection = {'candidate': candidate, 'candidate_is_baseline': candidate == 'P0__BASE', 'best_eligible': best_eligible, 'best_overall': best_overall,
             'P0__BASE_mean_within_auc_8banks': float(np.mean([WA[f'{bk}__P0__BASE'] for bk in BK])), 'table': TB.to_dict('records')}
# advisor table: one factor changed at a time around a centre cell, every alternative with its paired interval vs the centre
# (centre = the candidate; if the candidate is P0__BASE, the best eligible variant, else the best overall, so the stage table exists)
centre = candidate if candidate != 'P0__BASE' else (best_eligible or best_overall)
cp, cf = centre.split('__')
alts = [(f'position={p}', f'{p}__{cf}') for p in ('P0', 'P1', 'P2')]
if cf != 'BASE':
    cw, cb = cf.split('_')
    alts += [(f'within={w}', f'{cp}__{w}_{cb}') for w in WITHIN] + [(f'between={b}', f'{cp}__{cw}_{b}') for b in BETWEEN]
alts += [('plain_average_P0', 'P0__BASE')]
abl = {}
for bk in BK:
    row = {}
    for lab_, arm in alts:
        a, c = f'{bk}__{arm}', f'{bk}__{centre}'; r = {'arm': arm, 'within_auc': WA.get(a), 'full_coverage': a in dr}
        if a != c and a in dr and c in dr:
            cc = contrast(a, c); CON[f'{a} - {c}'] = cc; r |= {'delta_vs_centre': cc['prm_within_auc_delta'], 'lo': cc['prm_within_auc_lo'], 'hi': cc['prm_within_auc_hi']}
        row[lab_] = r
    for bw, bo in DIGIT_PAIRS:                                                                  # digits at the centre cell
        if bk == bw and f'{bw}__{centre}' in dr and f'{bo}__{centre}' in dr:
            cc = contrast(f'{bw}__{centre}', f'{bo}__{centre}'); CON[cc['contrast_id']] = cc
            row['digits_added'] = {'vs_bank': bo, 'delta': cc['prm_within_auc_delta'], 'lo': cc['prm_within_auc_lo'], 'hi': cc['prm_within_auc_hi']}
    abl[bk] = row
selection['one_factor_at_a_time'] = {'centre': centre, 'centre_is_candidate': centre == candidate, 'rows': abl}
dump(OUT / 'SELECTION.json', selection)
for bk in BK:
    for ref in ('ct7', 'fam421'):
        if f'{bk}__{candidate}' in dr: CON[f'{bk}__{candidate} - {ref}'] = contrast(f'{bk}__{candidate}', ref)
pd.DataFrame(list(CON.values())).to_csv(OUT / 'CONTRASTS.csv', index=False)
print('selection:', {k: v for k, v in selection.items() if k not in ('table', 'one_factor_at_a_time')}, flush=True)

# ------------------------------------------------------------------ nulls and concentration
t = time.perf_counter(); nulls = {}; conc = {}
NULLPAIRS = [(f'{bk}__P1__BASE', f'{bk}__P0__BASE') for bk in BK] + [(f'{bk}__P2__BASE', f'{bk}__P0__BASE') for bk in BK]
NULLPAIRS += [(f'{bk}__P0__SML_SML', f'{bk}__P0__BASE') for bk in BK] + [(f'{bk}__P1__SML_SML', f'{bk}__P0__BASE') for bk in BK]
NULLPAIRS += [(f'{bw}__P0__BASE', f'{bo}__P0__BASE') for bw, bo in DIGIT_PAIRS]
for v in {candidate, best_eligible} - {None, 'P0__BASE'}: NULLPAIRS += [(f'{bk}__{v}', f'{bk}__P0__BASE') for bk in BK]
for a, b in dict.fromkeys(NULLPAIRS):
    cid = f'{a} - {b}'
    if a not in dr or b not in dr: nulls[cid] = conc[cid] = {'note': 'NOT_ESTIMABLE'}; continue
    ans_idx = np.flatnonzero(e_all)
    dx = aucA[a][ans_idx] - aucA[b][ans_idx]; sgn = float(np.sign(dx.mean())) or 1.0; top_idx = np.argsort(-sgn * dx, kind='stable')[:int(np.ceil(.01 * len(dx)))]
    conc[cid] = {'answers': int(len(dx)), 'mean_delta': float(dx.mean()), 'tail_direction': 'gain' if sgn > 0 else 'loss', 'share_from_top1pct': float(dx[top_idx].sum() / dx.sum()) if dx.sum() != 0 else None,
                 'mean_without_top1pct': float(np.delete(dx, top_idx).mean()), 'trimmed5_mean': float(trim_mean(dx, .05)), 'answers_changed': int((np.abs(dx) > 1e-12).sum()),
                 'per_fold': {int(f): float(dx[fold[ans_idx] == f].mean()) for f in sorted(set(fold[ans_idx]))}}
    Rk, loc = C2.within_ranks(np.column_stack([scores[a], scores[b]]), off, ans_idx); yobs = np.concatenate([labels[off[i]:off[i+1]] for i in ans_idx]).astype(float)
    def stat(y): A = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0); return float(A[0] - A[1])
    obs = stat(yobs); nulls[cid] = {'answers': int(len(ans_idx)), 'observed_on_null_set': obs}
    for nm_, fn, sd in [('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)]:
        rg = np.random.default_rng(sd); x = np.array([stat(fn(yobs, loc, rg)) for _ in range(NPERM)])
        nulls[cid][nm_] = {'mean': float(x.mean()), 'sd': float(x.std()), 'p01': float(np.quantile(x, .01)), 'p99': float(np.quantile(x, .99)),
                           'share_ge_observed': float(np.mean(x >= obs)), 'share_le_observed': float(np.mean(x <= obs)), 'permutations': NPERM}
dump(OUT / 'NULLS.json', nulls); dump(OUT / 'CONCENTRATION.json', conc); timing['nulls_s'] = time.perf_counter() - t
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILED_FITS', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': failures,
               'checks': {k: v for k, v in checks.items() if not k.startswith('written_once')}, 'fitted_folds': fitted_folds, 'candidate': candidate})
dump(OUT / 'RUN_STATUS.json', status)
print(TB[['variant', 'mean_within_auc_8banks', 'mean_delta_vs_P0_BASE', 'wins', 'losses', 'eligible', 'qualifies']].head(15).round(4).to_string())
print(json.dumps(timing, indent=1)); print('status', status['status'], 'failures', len(failures), 'candidate', candidate)
