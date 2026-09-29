"""per_dataset_fit_v1: the method fitted PER MODEL PER DATASET, transductively and without labels (Omri 2026-09-29).
Each of the 9 cells (PRMBench; 8 ProcessBench cells) is fitted on all of its own answers and scores the same answers: lf__ signs,
DS filter, partition, weights, position prior, slopes, PRMScore threshold.  ProcessBench is also read with the first-error
readout P(first error at t) = q_t prod_{s<t} (1 - q_s) of the same per-cell latent model.  Secondary: cross-fitted slopes.
Frozen protocol: results/per_dataset_fit_v1/PROTOCOL.json.  Population and banks as position_prior_run.py.
Smoke: ER_CELLS=pb_gsm8k_q4,pb_math_q4 ER_DRAWS=2000 ER_NULL_PERMS=5."""
from pathlib import Path
import hashlib, json, os, pickle, subprocess, sys, time
from datetime import datetime
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from scipy.stats import rankdata, trim_mean, spearmanr  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC, er_stage_a as SA, er_stage_b as SB, er_stage_b2 as C2, lsml_merge_step as MS, ds_group_weights as GW, position_prior_ds as PP  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

GEN = ROOT / 'results/per_dataset_fit_v1'; OUT = GEN / (sys.argv[1] if len(sys.argv) > 1 else 'run_20260929'); OUT.mkdir(parents=True, exist_ok=True)
AD = ROOT / 'results/algorithm_decisions_v1/run_20260928'
P = json.loads((GEN / 'PROTOCOL.json').read_text(encoding='utf8'))
SMOKE = {k: os.environ[k] for k in ['ER_CELLS', 'ER_DRAWS', 'ER_NULL_PERMS'] if k in os.environ}
DRAWS = int(SMOKE.get('ER_DRAWS', 50_000)); NPERM = int(SMOKE.get('ER_NULL_PERMS', 200)); SEED = 20260927
T0 = time.perf_counter(); timing = {}
def jd(v): return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)
def dump(p, v): Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=jd), encoding='utf8')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
if P.get('status') != 'FROZEN': raise SystemExit('protocol not frozen')
if (OUT / 'RUN_STATUS.json').exists() and not SMOKE: raise SystemExit(f'{OUT} already holds a run')
status = {'status': 'RUNNING', 'started': datetime.now().isoformat(timespec='seconds'), 'smoke_overrides': SMOKE}; checks = {}; dump(OUT / 'RUN_STATUS.json', status)
def hard_stop(reason):
    status.update({'status': 'STOPPED', 'reason': reason, 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status); raise SystemExit('HARD STOP: ' + reason)
def _crash(et, ev, tb):
    import traceback; traceback.print_exception(et, ev, tb)
    if status.get('status') == 'RUNNING': status.update({'status': 'CRASHED', 'reason': repr(ev), 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status)
sys.excepthook = _crash
dump(OUT / 'CODE_MANIFEST.json', {'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(GEN / 'PROTOCOL.json'),
     'helpers': {h: sha(ROOT / 'scripts/experiments' / h) for h in ('er_stage_a.py', 'er_stage_b.py', 'er_stage_b2.py', 'ds_group_weights.py', 'lsml_merge_step.py', 'tail_calib_common.py', 'calfix_common.py', 'position_prior_ds.py')},
     'reference_step_scores_sha256': {d: sha(ROOT / d / 'STEP_SCORES.npz') for d in ('results/algorithm_decisions_v1/run_20260928', 'results/position_channel_v1/run_20260929', 'results/position_prior_v1/run_20260929')},
     'depth_git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=MAIN / '.worktrees/depth-feature-fusion-v1', capture_output=True, text=True).stdout.strip(),
     'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()})

# ------------------------------------------------------------------ population (as algorithm_decisions_run.py)
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz')
off = Zs['offsets']; labels = Zs['labels'].astype(bool); n = len(ans); ns = np.diff(off); S = int(off[-1])
pb = ans.cell.str.startswith('pb_').to_numpy(); prm = ~pb; fold = ans.fold.to_numpy(); cells = ans.cell.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy(); target = ans.target.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig')); meta = {m['idx']: m for m in pickle.load(open(Path(freeze['prm_metadata']['path']), 'rb')).values()}
noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
prm_steps = np.repeat(prm, ns); step_answer = np.repeat(np.arange(n), ns); step_pos = (np.arange(S) - off[step_answer]).astype(float)
eligible = np.array([prm[i] and labels[off[i]:off[i+1]].any() and (~labels[off[i]:off[i+1]]).any() for i in range(n)])
has_error = np.array([prm[i] and labels[off[i]:off[i+1]].any() for i in range(n)])
def rows_of(mask): return np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(mask)])
def within_auc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
def answer_z(s):
    out = np.empty(len(s))
    for a, b in zip(off[:-1], off[1:]): out[a:b] = zt(s[a:b])
    return out
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))

# ------------------------------------------------------------------ banks (as partition_switch_run.py bank(); replays P0__BASE)
MI = json.loads((AD / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
paths = {k: Path(v['path']) for k, v in MI.items() if isinstance(v, dict) and 'path' in v}
for k in ('level_bank', 'ct7_profiles', 'ct7_profile_validation', 'pool_z', 'pool_names', 'digit_features', 'oof_answers', 'oof_step_scores', 'pool_structure'):
    if sha(paths[k]) != MI[k]['sha256']: hard_stop(f'input {k} differs from the algorithm_decisions_v1 manifest')
lv = np.load(paths['level_bank']); names11 = list(map(str, lv['channels']))
prof = np.load(paths['ct7_profiles']).astype(float); pn = json.loads(paths['ct7_profile_validation'].read_text(encoding='utf8'))['channels']
values = answer_standardize(np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)]), off)
DF = np.load(paths['digit_features']); act = DF['active'].astype(bool); DFV = DF['values']; D3 = np.zeros((S, 3))
for a, b in zip(off[:-1], off[1:]):
    for j in range(3):
        v = act[a:b, j]; y = DFV[a:b, j][v]
        if len(y) and y.std() > 1e-12: D3[a:b, j][v] = (y - y.mean()) / y.std()
POOL = np.load(paths['pool_z']); PN = json.loads(paths['pool_names'].read_text(encoding='utf8')); PS = pd.read_csv(paths['pool_structure']).set_index('channel')
BN = MI['banks']; DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
def bank(bk):
    cols = []
    for c in BN[bk]:
        if c in names11 + ['realized_z', 'realized_drv'] and bk in ('B13', 'B16'): cols.append(values[:, (names11 + ['realized_z', 'realized_drv']).index(c)])
        elif c in DIG: cols.append(D3[:, DIG.index(c)])
        else: cols.append(None)
    need = [i for i, c in enumerate(cols) if c is None]
    if need:
        Vp = answer_standardize(np.column_stack([POOL[:, PN.index(BN[bk][i][4:] if BN[bk][i].startswith('lf__') else BN[bk][i])] for i in need]), off)
        for q, i in enumerate(need):
            if BN[bk][i].startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[BN[bk][i][4:], 'r_level_marginal'])) or 1.0
            cols[i] = Vp[:, q]
    return np.column_stack(cols)
BANKS = ['B16', 'B23', 'B35', 'B54']
_SC = np.load(AD / 'STEP_SCORES.npz'); REF = {f'{bk}__{a}': _SC[f'{bk}__P0__{a}'] for bk in BANKS for a in ('BASE', 'EQ_DSM')}; REF['ct7'] = _SC['ct7']; REF['fam421'] = _SC['fam421']; del _SC
VB = {bk: bank(bk) for bk in BANKS}; MARKS = {}; TTA = {}; KEYB = {}
for bk, V in VB.items():
    KEYB[bk] = np.random.default_rng(20260928).random((S, V.shape[1])); MARKS[bk] = SB.random_tie_marks(V, off, .2, KEYB[bk])
    if not marks_ok(MARKS[bk]): hard_stop(f'mark counts {bk}')
    TTA[bk] = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
POSZ = answer_z(step_pos); KEYP = np.random.default_rng(20260931).random((S, 1)); POSM = SB.random_tie_marks(POSZ[:, None], off, .2, KEYP)
KEYG = np.random.default_rng(20260929).random((S, 13))
if not marks_ok(POSM): hard_stop('POS mark counts')
timing['setup_s'] = time.perf_counter() - T0; print(f'setup {timing["setup_s"]:.0f}s', flush=True)

# ------------------------------------------------------------------ per-cell machinery
CELLS_ALL = sorted(set(cells)); CELLS = SMOKE['ER_CELLS'].split(',') if 'ER_CELLS' in SMOKE else CELLS_ALL
PRMC = 'prmbench_qwen3_8b'; step_cell = np.repeat(cells, ns)
LEVEL = POOL[:, [PN.index(c) for c in ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level']]].mean(1)
BINS = {1: np.zeros(S, np.int64), 10: PP.position_bins(off, 10)}; NB = {1: 1, 10: 10}
LFCOL = {bk: [(i, c[4:]) for i, c in enumerate(BN[bk]) if c.startswith('lf__')] for bk in BANKS}
for bk in BANKS:   # column separability of the mark functions (recomputing one column alone reproduces it)
    j = LFCOL[bk][0][0] if LFCOL[bk] else 1
    if not (np.array_equal(SB.random_tie_marks(VB[bk][:, [j]], off, .2, KEYB[bk][:, [j]]), MARKS[bk][:, [j]])
            and np.array_equal(tail_marks(VB[bk][:, [j]], off, .2, tie_aware=True, centred=True)[0], TTA[bk][:, [j]])): hard_stop(f'mark functions not column-separable ({bk} column {j})')
def cell_bank(bk, rc, ce):
    """Per-cell label-free signs of the lf__ channels (sign of corr with the level mean on the cell's steps)."""
    flips = []; rs = {}
    for i, c in LFCOL[bk]:
        j = PN.index(c); r = float(np.corrcoef(POOL[rc, j], LEVEL[rc])[0, 1]); rs[c] = r
        if ce == PRMC and abs(r - float(PS.loc[c, 'r_level_marginal'])) > 1e-6: hard_stop(f'{c}: PRMBench-cell correlation {r} differs from POOL_STRUCTURE')
        if (float(np.sign(r)) or 1.0) != (float(np.sign(PS.loc[c, 'r_level_marginal'])) or 1.0): flips.append(i)
    if not flips: return VB[bk], MARKS[bk], TTA[bk], [], rs
    Vc = VB[bk].copy(); Vc[:, flips] *= -1
    Mc = MARKS[bk].copy(); Mc[:, flips] = SB.random_tie_marks(Vc[:, flips], off, .2, KEYB[bk][:, flips])
    if not marks_ok(Mc): hard_stop(f'mark counts after sign flips {bk} {ce}')
    Tc = TTA[bk].copy(); Tc[:, flips] = tail_marks(Vc[:, flips], off, .2, tie_aware=True, centred=True)[0]
    return Vc, Mc, Tc, [BN[bk][i] for i in flips], rs
def logit_pi(f): return np.log(f['pi_bins']) - np.log1p(-f['pi_bins'])
def one_bin_start(marks_fit, init, rows, tag):
    f = PP.fit_pds(marks_fit, BINS[1][rows], 1, init['psi'], init['eta'], init['prevalence'])
    dev = float(max(np.abs(f['psi'] - init['psi']).max(), np.abs(f['eta'] - init['eta']).max(), abs(f['pi_bins'][0] - init['prevalence'])))
    rel = float((f['loglik'] - f['loglik_start']) / abs(f['loglik_start']))
    if not (f['converged'] and f['oriented'] and rel <= 1e-6 and dev <= 1e-2):
        hard_stop(f'{tag}: one-bin fit is not the constant DS optimum (rel {rel}, move {dev}, oriented {f["oriented"]})')
    return {'psi': f['psi'], 'eta': f['eta'], 'prevalence': float(f['pi_bins'][0])}, {'max_param_move': dev, 'rel_loglik_rise': rel}, f
def prior_fit(marks_fit, init, rows, tag):
    f = PP.fit_pds(marks_fit, BINS[10][rows], 10, init['psi'], init['eta'], init['prevalence'])
    if not (f['converged'] and f['oriented']): hard_stop(f'{tag}: 10-bin fit converged {f["converged"]} oriented {f["oriented"]}')
    if f['bin_rows'].min() < 50: hard_stop(f'{tag}: a bin has {f["bin_rows"].min()} rows')
    return f
def slope(S_full, q, rows, tag):
    a, parts = PP.latent_slope(S_full[rows], q)
    if not a > 0: hard_stop(f'{tag}: slope {a}')
    return a, parts['mu1'], parts['mu0'], parts
def fe_score(L):
    """log P(first error at t) = log q_t + sum_{s<t} log(1 - q_s), q = sigmoid(L), within each answer."""
    lq = -np.logaddexp(0.0, -L); l1q = -np.logaddexp(0.0, L)
    prev = np.concatenate([[0.0], np.cumsum(l1q)[:-1]])
    return lq + prev - np.repeat(prev[off[:-1]], ns)
def fit_arms(V, MK, TT, dsr, ptr, tag):
    """All arms from one fit: DS/EM/prior/slopes on rows dsr, partition on rows ptr (the per-cell contract: dsr = ptr = the cell)."""
    out, d = {}, {}
    est = SA.em_estimate(MK[dsr], 'ds'); surv = np.flatnonzero(est['pi'] > 0.5)
    if len(surv) < 6: hard_stop(f'{tag}: only {len(surv)} survivors')
    anchor_ch = 0 if 0 in surv else int(surv[np.argmax(est['pi'][surv])]); d['anchor_fallback'] = bool(anchor_ch != 0)
    X = V[:, surv]; out['BASE'] = X @ np.full(len(surv), 1 / len(surv)); d['n_survivors'] = int(len(surv))
    e2 = SA.em_estimate(np.column_stack([MK, POSM])[dsr], 'ds'); s2 = np.flatnonzero(e2['pi'] > 0.5)
    out['BASE_POS'] = np.column_stack([V, POSZ])[:, s2] @ np.full(len(s2), 1 / len(s2)); d['pos_kept'] = bool(V.shape[1] in s2)
    ms = MK[:, surv][dsr]
    es, d['one_bin_base'], f1 = one_bin_start(ms, SA.em_estimate(ms, 'ds'), dsr, f'{tag} BASE')
    a0, m1, m0, _ = slope(out['BASE'], f1['q'], dsr, f'{tag} BASE one-bin')
    out['FE_BASE0'] = fe_score(logit_pi(f1)[0] + a0 * (out['BASE'] - (m1 + m0) / 2))
    f10 = prior_fit(ms, es, dsr, f'{tag} BASE'); lp = logit_pi(f10)[BINS[10]]
    a, m1, m0, parts = slope(out['BASE'], f10['q'], dsr, f'{tag} BASE')
    out['BASE_PRIOR'] = out['BASE'] + lp / a
    out['FE_BASE_PRIOR'] = fe_score(lp + a * (out['BASE'] - (m1 + m0) / 2)); out['FE_PRIOR_ALONE'] = fe_score(lp)
    d['base'] = {'pi10': f10['pi_bins'].tolist(), 'a': a, 'a_one_bin': a0, 'latent_prevalence': float(f10['q'].mean()), 'two_dLL': 2 * (f10['loglik'] - f10['loglik_start'])}
    # cross-fitted slopes: the latent class of one half of the survivors gives the slope of the other half's average
    HA, HB = surv[0::2], surv[1::2]
    if min(len(HA), len(HB)) < 3: hard_stop(f'{tag}: a cross-fit half has < 3 channels')
    SAv = V[:, HA].mean(1); SBv = V[:, HB].mean(1); qh = {}
    for nm, H in (('A', HA), ('B', HB)):
        mh = MK[:, H][dsr]; eh, _, _ = one_bin_start(mh, SA.em_estimate(mh, 'ds'), dsr, f'{tag} half {nm}'); qh[nm] = prior_fit(mh, eh, dsr, f'{tag} half {nm}')['q']
    aB, mB1, mB0, _ = slope(SBv, qh['A'], dsr, f'{tag} CF slope B'); aA, mA1, mA0, _ = slope(SAv, qh['B'], dsr, f'{tag} CF slope A')
    content_cf = aA * (SAv - (mA1 + mA0) / 2) + aB * (SBv - (mB1 + mB0) / 2)
    out['BASE_PRIOR_CF'] = content_cf + lp; out['FE_BASE_PRIOR_CF'] = fe_score(content_cf + lp)
    bc = out['BASE'][dsr] - out['BASE'][dsr].mean(); cc = content_cf[dsr] - content_cf[dsr].mean()
    d['cf'] = {'a_A': aA, 'a_B': aB, 'n_A': int(len(HA)), 'n_B': int(len(HB)), 'a_cf_equivalent_on_BASE': float(cc @ bc / (bc @ bc)), 'a_model': a}
    # grouped method
    T = TT[:, surv]; anchor = int(np.flatnonzero(surv == anchor_ch)[0])
    gb = MS.canon(TC.lsml_fit_scaled(T[ptr], anchor, X[ptr], standardize=True, loading_scale='unit')['groups'])
    gbm, _ = MS.absorb_merge(np.corrcoef(T[ptr], rowvar=False), gb); G = int(gbm.max()) + 1
    if G > KEYG.shape[1]: hard_stop(f'{tag}: {G} groups exceed the key')
    Z = GW.group_matrix(X, off, gbm, None, answer_standardize); gv = SB.random_tie_marks(Z, off, .2, KEYG[:, :G])
    if not marks_ok(gv): hard_stop(f'{tag}: group mark counts')
    eg = SA.em_estimate(gv[dsr], 'ds'); w = SB.mle_weights(eg['psi'], eg['eta'])
    if w.sum() <= 0: hard_stop(f'{tag}: GRP weights all 0')
    out['GRP'] = GW.weighted_group_score(Z, w)
    ex = SA.em_estimate(np.column_stack([gv, POSM])[dsr], 'ds'); wx = SB.mle_weights(ex['psi'], ex['eta'])
    out['GRP_POS'] = GW.weighted_group_score(np.column_stack([Z, POSZ]), wx)
    eg1, d['one_bin_grp'], g1 = one_bin_start(gv[dsr], eg, dsr, f'{tag} GRP')
    ag0, m1, m0, _ = slope(out['GRP'], g1['q'], dsr, f'{tag} GRP one-bin')
    out['FE_GRP0'] = fe_score(logit_pi(g1)[0] + ag0 * (out['GRP'] - (m1 + m0) / 2))
    fg10 = prior_fit(gv[dsr], eg1, dsr, f'{tag} GRP'); lpg = logit_pi(fg10)[BINS[10]]
    ag, m1, m0, _ = slope(out['GRP'], fg10['q'], dsr, f'{tag} GRP')
    out['GRP_PRIOR'] = out['GRP'] + lpg / ag; out['FE_GRP_PRIOR'] = fe_score(lpg + ag * (out['GRP'] - (m1 + m0) / 2))
    d['grp'] = {'K': G, 'pi10': fg10['pi_bins'].tolist(), 'a': ag, 'a_one_bin': ag0, 'weights': w.tolist()}
    return out, d
ARMS = ['BASE', 'BASE_POS', 'BASE_PRIOR', 'BASE_PRIOR_CF', 'GRP', 'GRP_POS', 'GRP_PRIOR', 'FE_BASE0', 'FE_BASE_PRIOR', 'FE_BASE_PRIOR_CF', 'FE_GRP0', 'FE_GRP_PRIOR', 'FE_PRIOR_ALONE']
REFS = {'REF_BASE': (AD, 'P0__BASE'), 'REF_GRP': (AD, 'P0__EQ_DSM'), 'REF_BASE_POS': (ROOT / 'results/position_channel_v1/run_20260929', 'BASE_POS'),
        'REF_GRP_POS': (ROOT / 'results/position_channel_v1/run_20260929', 'GRP_POS'), 'REF_BASE_PRIOR': (ROOT / 'results/position_prior_v1/run_20260929', 'BASE_PRIOR'),
        'REF_GRP_PRIOR': (ROOT / 'results/position_prior_v1/run_20260929', 'GRP_PRIOR')}
_ref = {}
for d_ in {v[0] for v in REFS.values()}:
    z = np.load(d_ / 'STEP_SCORES.npz'); _ref[d_] = {k: z[k] for k in z.files}; del z
REFSC = {f'{bk}__{r}': _ref[d_][f'{bk}__{key}'] for bk in BANKS for r, (d_, key) in REFS.items()}
REFSC['ct7'] = _ref[AD]['ct7'].astype(float); REFSC['fam421'] = _ref[AD]['fam421']; REFSC['POS_ALONE'] = POSZ; del _ref

# ------------------------------------------------------------------ code-path replay (fold 0, pooled rows and signs)
t = time.perf_counter(); replay = {}
fit0 = rows_of(np.isin(fold, [2, 3, 4])); ev0 = rows_of(fold == 0); pf0 = fit0[prm_steps[fit0]]
for bk in BANKS:
    o, _ = fit_arms(VB[bk], MARKS[bk], TTA[bk], pf0, fit0, f'replay {bk}')
    for arm, ref in (('BASE', 'REF_BASE'), ('GRP', 'REF_GRP'), ('BASE_POS', 'REF_BASE_POS'), ('GRP_POS', 'REF_GRP_POS'), ('BASE_PRIOR', 'REF_BASE_PRIOR'), ('GRP_PRIOR', 'REF_GRP_PRIOR')):
        replay[f'{bk}__{arm}'] = float(np.max(np.abs(o[arm][ev0] - REFSC[f'{bk}__{ref}'][ev0])))
        if not replay[f'{bk}__{arm}'] <= 1e-9: hard_stop(f'code-path replay {bk} {arm}: {replay[f"{bk}__{arm}"]}')
checks['code_path_replay_max'] = max(replay.values()); dump(OUT / 'REPLAY.json', replay)
timing['replay_s'] = time.perf_counter() - t; print(f'code-path replay max {checks["code_path_replay_max"]:.1e} ({timing["replay_s"]:.0f}s)', flush=True)

# ------------------------------------------------------------------ per-cell fits (transductive, label-free)
ALL = [f'{bk}__{a}' for bk in BANKS for a in ARMS + list(REFS)] + ['ct7', 'fam421', 'POS_ALONE']
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}; diag = []
def put(m, rows, s_full):
    if written[m][rows].any(): raise AssertionError(f'{m} written twice')
    if not np.isfinite(s_full[rows]).all(): hard_stop(f'non-finite scores {m}')
    scores[m][rows] = s_full[rows]; written[m][rows] += 1
for ce in CELLS:
    t = time.perf_counter(); rc = np.flatnonzero(step_cell == ce)
    for bk in BANKS:
        Vc, Mc, Tc, flips, rs = cell_bank(bk, rc, ce)
        o, d = fit_arms(Vc, Mc, Tc, rc, rc, f'{ce} {bk}')
        for a in ARMS: put(f'{bk}__{a}', rc, o[a])
        for r in REFS: put(f'{bk}__{r}', rc, REFSC[f'{bk}__{r}'])
        d |= {'cell': ce, 'bank': bk, 'sign_flips': flips, 'lf_level_corr': rs}; diag.append(d)
        print(f"  {ce} {bk}: surv {d['n_survivors']} flips {len(flips)} K {d['grp']['K']} | pi10 {d['base']['pi10'][0]:.2f}->{d['base']['pi10'][-1]:.2f} "
              f"a {d['base']['a']:.2f} a_cf_eq {d['cf']['a_cf_equivalent_on_BASE']:.2f} | GRP a {d['grp']['a']:.2f}", flush=True)
    for m in ('ct7', 'fam421', 'POS_ALONE'): put(m, rc, REFSC[m])
    print(f'{ce} done ({time.perf_counter()-t:.0f}s)', flush=True)
checks['written_once'] = all(written[m].max() <= 1 for m in ALL)
dump(OUT / 'FIT_DIAGNOSTICS.json', diag); np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **scores)
timing['fit_s'] = time.perf_counter() - T0

# ------------------------------------------------------------------ evaluation (transductive PRMScore threshold, the same rule for every arm)
def prmscore_from_counts(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p2 = ratio(tn, tn + fn); r2 = ratio(tn, tn + fp); nf = ratio(2 * p2 * r2, p2 + r2); return (f + nf) / 2
def earliest_argmax(v): return int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])
cov = np.isin(cells, CELLS); PBC = sorted(c for c in set(cells[pb]) if c in CELLS); cell_idx = np.array([PBC.index(c) if c in PBC else -1 for c in cells])
prm_cov = prm & cov; tau = {}
aucA, confA, hitA, metrics = {}, {}, {}, []
for m in ALL:
    s = scores[m]
    aucA[m] = np.array([within_auc(labels[off[i]:off[i+1]], s[off[i]:off[i+1]]) if eligible[i] and cov[i] else np.nan for i in range(n)])
    confA[m] = np.zeros((n, 4)); flags = {}
    if prm_cov.any():
        tau[m] = float(np.quantile(np.concatenate([zt(s[off[i]:off[i+1]]) for i in np.flatnonzero(prm_cov)]), .8))
        for i in np.flatnonzero(prm_cov):
            a, b = off[i:i+2]; vv = zt(s[a:b]) < tau[m]; gg = ~labels[a:b]; flags[i] = vv
            if noncontrol[i]: confA[m][i] = [(vv & gg).sum(), (vv & ~gg).sum(), (~vv & ~gg).sum(), (~vv & gg).sum()]
        tot = prmbench_evaluate([{'idx': ids[i], 'labels': flags[i].astype(int).tolist()} for i in np.flatnonzero(prm_cov)], [meta[ids[i]] for i in np.flatnonzero(prm_cov)])['total']
        ps = float(.5 * (tot['f1'] + tot['negative_f1']))
    else:
        ps = float('nan')
    hitA[m] = np.array([float(earliest_argmax(s[off[i]:off[i+1]]) == target[i]) if pb[i] and target[i] >= 0 and cov[i] else np.nan for i in range(n)])
    percell = [np.nanmean(hitA[m][cell_idx == ci]) for ci in range(len(PBC))]
    metrics.append({'method': m, 'within_auc': float(np.nanmean(aucA[m])) if np.isfinite(aucA[m]).any() else float('nan'), 'prmscore': ps,
                    'pb_sla_macro8': float(np.mean(percell)) if PBC else float('nan'), 'pb_sla_pooled': float(np.nanmean(hitA[m][pb & cov & (target >= 0)])) if PBC else float('nan'),
                    **{f'pb_{c}': v for c, v in zip(PBC, percell)}})
M = pd.DataFrame(metrics); M.to_csv(OUT / 'METRICS.csv', index=False); dump(OUT / 'THRESHOLDS.json', tau); timing['eval_s'] = time.perf_counter() - T0 - timing['fit_s']

# ------------------------------------------------------------------ paired source-group bootstrap
Gpr, gi = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = gi
Gpb, gj = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gj
e_all = cov & eligible; nc_all = cov & noncontrol; pe_all = cov & pb & (target >= 0)
cntG = np.bincount(gpr[e_all], minlength=len(Gpr)).astype(float); hc = np.zeros((len(Gpb), max(len(PBC), 1))); np.add.at(hc, (gpbx[pe_all], cell_idx[pe_all]), 1)
st = {}
for m in ALL:
    hs = np.zeros(hc.shape); np.add.at(hs, (gpbx[pe_all], cell_idx[pe_all]), hitA[m][pe_all])
    st[m] = {'auc': np.bincount(gpr[e_all], weights=aucA[m][e_all], minlength=len(Gpr)), 'conf': np.stack([np.bincount(gpr[nc_all], weights=confA[m][nc_all, q], minlength=len(Gpr)) for q in range(4)], 1), 'hit': hs}
dr = {m: {e: np.empty(DRAWS, np.float32) for e in ('auc', 'ps', 'sla')} for m in ALL}
rng = np.random.default_rng(SEED); rng2 = np.random.default_rng(SEED + 1); pos = 0
with np.errstate(invalid='ignore', divide='ignore'):
    while pos < DRAWS:
        nb = min(2500, DRAWS - pos); W = rng.multinomial(len(Gpr), np.full(len(Gpr), 1 / len(Gpr)), size=nb).astype(float); W2 = rng2.multinomial(len(Gpb), np.full(len(Gpb), 1 / len(Gpb)), size=nb).astype(float)
        den = W @ cntG; hden = W2 @ hc
        for m in ALL:
            dr[m]['auc'][pos:pos+nb] = (W @ st[m]['auc']) / den; dr[m]['ps'][pos:pos+nb] = prmscore_from_counts(*(W @ st[m]['conf']).T)
            dr[m]['sla'][pos:pos+nb] = np.nanmean(np.divide(W2 @ st[m]['hit'], hden, out=np.full(hden.shape, np.nan), where=hden > 0), 1)
        pos += nb
Mx = M.set_index('method')
PRIM_A = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in (('BASE_POS', 'REF_BASE_POS'), ('GRP_PRIOR', 'REF_GRP_PRIOR'))]
PRIM_B = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in (('FE_BASE_PRIOR', 'BASE'), ('FE_GRP_PRIOR', 'GRP'))]
SEC = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in (
    ('FE_BASE0', 'BASE'), ('FE_GRP0', 'GRP'), ('FE_BASE_PRIOR', 'FE_BASE0'), ('FE_GRP_PRIOR', 'FE_GRP0'), ('FE_BASE_PRIOR', 'BASE_PRIOR'), ('FE_GRP_PRIOR', 'GRP_PRIOR'),
    ('FE_BASE_PRIOR', 'REF_BASE'), ('FE_GRP_PRIOR', 'REF_GRP'), ('FE_PRIOR_ALONE', 'BASE'), ('BASE', 'REF_BASE'), ('GRP', 'REF_GRP'),
    ('BASE_PRIOR_CF', 'BASE_PRIOR'), ('FE_BASE_PRIOR_CF', 'FE_BASE_PRIOR'), ('BASE_PRIOR', 'REF_BASE_PRIOR'), ('GRP_POS', 'REF_GRP_POS'))] + [(f'{bk}__FE_GRP_PRIOR', 'ct7') for bk in BANKS]
K = 8; rows = []
for a, b in PRIM_A + PRIM_B + SEC:
    fam = 'A' if (a, b) in PRIM_A else 'B' if (a, b) in PRIM_B else ''; r = {'contrast_id': f'{a} - {b}', 'primary': fam}
    for e, col in (('auc', 'within_auc'), ('ps', 'prmscore'), ('sla', 'pb_sla_macro8')):
        x = dr[a][e].astype(float) - dr[b][e].astype(float); r[f'{col}_delta'] = float(Mx.loc[a, col] - Mx.loc[b, col])
        r[f'{col}_lo95'], r[f'{col}_hi95'] = (np.nanquantile(x, [.025, .975]).tolist() if np.isfinite(x).any() else [np.nan, np.nan])
        if (fam == 'A' and e == 'auc') or (fam == 'B' and e == 'sla'): r[f'{col}_lo_bonf'], r[f'{col}_hi_bonf'] = np.nanquantile(x, [.025 / K, 1 - .025 / K]).tolist()
    r['pb_sla_pooled_delta'] = float(Mx.loc[a, 'pb_sla_pooled'] - Mx.loc[b, 'pb_sla_pooled']); rows.append(r)
CON = pd.DataFrame(rows); CON.to_csv(OUT / 'CONTRASTS.csv', index=False); timing['bootstrap_s'] = time.perf_counter() - T0 - timing['fit_s'] - timing['eval_s']

# ------------------------------------------------------------------ nulls and concentration for primary A; decision
nulls = {}; conc = {}; E = np.flatnonzero(e_all)
if len(E):
    for a, b in PRIM_A:
        cid = f'{a} - {b}'; dx = aucA[a][E] - aucA[b][E]; sg = float(np.sign(dx.mean())) or 1.0; top = np.argsort(-sg * dx, kind='stable')[:int(np.ceil(.01 * len(dx)))]
        conc[cid] = {'mean_delta': float(dx.mean()), 'share_from_top1pct': float(dx[top].sum() / dx.sum()) if dx.sum() != 0 else None, 'trimmed5_mean': float(trim_mean(dx, .05))}
        Rk, loc = C2.within_ranks(np.column_stack([scores[a], scores[b]]), off, E); yE = np.concatenate([labels[off[i]:off[i+1]] for i in E]).astype(float)
        def stat(y): A_ = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0); return float(A_[0] - A_[1])
        obs = stat(yE); nulls[cid] = {'observed': obs}
        for nm_, fn, sd in (('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)):
            rg = np.random.default_rng(sd); x = np.array([stat(fn(yE, loc, rg)) for _ in range(NPERM)])
            nulls[cid][nm_] = {'mean': float(x.mean()), 'sd': float(x.std()), 'share_ge_observed': float(np.mean(x >= obs))}
dump(OUT / 'NULLS.json', nulls); dump(OUT / 'CONCENTRATION.json', conc)
Cx = CON.set_index('contrast_id'); decision = {}
if PBC:
    for fam, x in (('plain average', 'BASE'), ('grouped', 'GRP')):
        ok = {bk: bool(Cx.loc[f'{bk}__FE_{x}_PRIOR - {bk}__{x}', 'pb_sla_macro8_lo_bonf'] > 0) for bk in BANKS}
        decision[f'first_error_readout_{fam}'] = {'bonferroni_above_content_argmax': ok, 'adopt': all(ok.values()),
                                                   'delta_macro8': {bk: float(Cx.loc[f'{bk}__FE_{x}_PRIOR - {bk}__{x}', 'pb_sla_macro8_delta']) for bk in BANKS}}
    okf = {bk: bool(Cx.loc[f'{bk}__FE_BASE_PRIOR_CF - {bk}__FE_BASE_PRIOR', 'pb_sla_macro8_lo95'] > 0) for bk in BANKS}
    decision['cross_fit_in_first_error_readout'] = {'lo95_above_0': okf, 'adopt': all(okf.values())}
if prm_cov.any():
    okc = {bk: bool(Cx.loc[f'{bk}__BASE_PRIOR_CF - {bk}__BASE_PRIOR', 'within_auc_lo95'] > 0) for bk in BANKS}
    decision['cross_fit_in_plain_average_prior'] = {'lo95_above_0': okc, 'adopt': all(okc.values())}
    decision['per_dataset_vs_pooled_prmbench_within_auc'] = {c: {'delta': float(Cx.loc[f'{a} - {b}', 'within_auc_delta']), 'lo_bonf': float(Cx.loc[f'{a} - {b}', 'within_auc_lo_bonf']), 'hi_bonf': float(Cx.loc[f'{a} - {b}', 'within_auc_hi_bonf'])} for a, b in PRIM_A for c in [f'{a} - {b}']}
dump(OUT / 'DECISION.json', decision)
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE', 'finished': datetime.now().isoformat(timespec='seconds'), 'checks': checks}); dump(OUT / 'RUN_STATUS.json', status)
print(M[['method', 'within_auc', 'prmscore', 'pb_sla_macro8', 'pb_sla_pooled']].round(4).to_string(index=False))
print(CON[['contrast_id', 'primary', 'within_auc_delta', 'within_auc_lo95', 'pb_sla_macro8_delta', 'pb_sla_macro8_lo95', 'pb_sla_macro8_hi95']].round(4).to_string(index=False))
print(json.dumps(decision, indent=1)); print(json.dumps(timing))
