"""expectation_realization_v1 stage B2 (level reduction): count the level family once, by removing two of the five level
channels (all ten pairs, none selected) or by merging the two level sub-groups (control), and measure whether the label-free
Dawid-Skene estimates, the maximum-likelihood group weights and continuous L-SML improve.  Frozen protocol:
results/expectation_realization_v1/PROTOCOL_STAGE_B2.json.  Population frame, filter, marks, keys and evaluation are those of
er_stage_b_run.py (reviewed); base-configuration arms must reproduce the stage-B arms.  Smoke: ER_FOLDS=0 ER_DRAWS=2000 ER_NULL_PERMS=5.

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

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260927_stage_b2'
STAGE = ROOT / 'results/expectation_realization_v1'; OUT = STAGE / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
A_RUN = STAGE / 'run_20260927'
P = json.loads((STAGE / 'PROTOCOL_STAGE_B2.json').read_text(encoding='utf8'))
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
for rel, base in [('scripts/experiments/er_stage_b2_run.py', ROOT), ('scripts/experiments/er_stage_b2.py', ROOT), ('scripts/experiments/er_stage_b_run.py', ROOT), ('scripts/experiments/er_stage_b.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/prmbench.py', DEPTH),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(STAGE / 'PROTOCOL_STAGE_B2.json')})
print(f'channels ready ({timing["channels_s"]:.0f}s); checks {checks}', flush=True)

# ------------------------------------------------------------------ stage B2: level-reduction configurations
import itertools  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from calfix_common import tail_marks  # noqa: E402
import er_stage_b2 as C2  # noqa: E402
B_RUN = STAGE / 'run_20260927_stage_b'
LEVEL = ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level']; SHORT = {'q15_H1': 'H1', 'q15_VE1': 'VE1', 'logprob_margin': 'LM', 'true_tail50': 'TT', 'energy_level': 'EL'}
RM = {f'rm_{SHORT[a]}_{SHORT[b]}': (a, b) for a, b in itertools.combinations(LEVEL, 2)}
CONFIGS = ['base', 'merge'] + list(RM)
ARMS = ['B_sml', 'B_equal', 'B_oracle', 'E_equal', 'L_own', 'L_grp']
def arms_of(c): return [a for a in ARMS if not (c == 'merge' and a in ('E_equal', 'L_own'))]
def ds_meta(est): return {kk: est.get(kk) for kk in ('converged', 'selected_start', 'boundary_emissions', 'orientation', 'prevalence')}
def level_share(w, chans): w = np.abs(np.asarray(w, float)); return float(w[[j for j, c in enumerate(chans) if c in LEVEL]].sum() / w.sum())
def config_chain(V, votes13, Tta, key_g, pf, fit_rows, bank, k):
    """Stage-B filter, then every configuration.  Returns {arm__config: full-row score} and one diagnostic dict per
    configuration.  Labels enter only the truth/dependence diagnostics and B_oracle (fit-fold labels only)."""
    out = {}; diags = []
    try: est13 = SA.em_estimate(votes13[pf], 'ds')
    except Exception as e: return out, [{'fold': k, 'bank': bank, 'config': c, 'failure': 'DS on 13 channels: ' + repr(e)} for c in CONFIGS]
    surv = [names[j] for j in np.flatnonzero(est13['pi'] > 0.5)]; pi13 = dict(zip(names, est13['pi'])); base_part = None
    for c in CONFIGS:
        d = {'fold': k, 'bank': bank, 'config': c, 'survivors13': surv, 'ds13': ds_meta(est13)}
        chans = [x for x in surv if c in ('base', 'merge') or x not in RM[c]]; d['channels'] = chans; diags.append(d)
        if len(chans) < 3: d['failure'] = f'{len(chans)} channels'; continue
        idx = [names.index(x) for x in chans]
        anchor = chans.index('q15_H1') if 'q15_H1' in chans else int(np.argmax([pi13[x] for x in chans])); d['anchor'] = chans[anchor]
        try:
            if c == 'merge':
                if base_part is None: raise ValueError('base partition missing')
                gs = C2.merge_groups_containing(base_part, chans, set(LEVEL))
            else:
                ps = TC.lsml_fit_scaled(Tta[fit_rows][:, idx], anchor, V[fit_rows][:, idx], standardize=True, loading_scale='unit')
                gs = np.unique(np.asarray(ps['groups'], int), return_inverse=True)[1]
                if c == 'base': base_part = gs
        except Exception as e: d['failure'] = 'partition: ' + repr(e); continue
        G = int(gs.max()) + 1; d['partition'] = gs.tolist(); d['members'] = [[chans[j] for j in np.flatnonzero(gs == g)] for g in range(G)]
        Z = SB.group_scores(V[:, idx], off, gs, answer_standardize)
        out[f'B_equal__{c}'] = Z @ np.full(G, 1 / G)
        if c != 'merge':
            out[f'E_equal__{c}'] = V[:, idx] @ np.full(len(idx), 1 / len(idx))
            try:
                wl, ml = fit_fusion_weights(V[fit_rows][:, idx], FusionRecipe(name='L_own', members=tuple(chans), mode='continuous', anchor=anchor), seed=FIT_SEED)
                out[f'L_own__{c}'] = V[:, idx] @ wl
                d['L_own'] = {'weights': wl, 'K': int(ml['K']), 'groups': ml['groups'], 'level_share': level_share(wl, chans), 'anchor_flipped': ml['anchor_flipped']}
            except Exception as e: d['L_own_failure'] = repr(e)
        try:
            wg, mg = fit_fusion_weights(V[fit_rows][:, idx], FusionRecipe(name='L_grp', members=tuple(chans), mode='continuous', groups=tuple(int(v) for v in gs), anchor=anchor), seed=FIT_SEED)
            out[f'L_grp__{c}'] = V[:, idx] @ wg
            d['L_grp'] = {'weights': wg, 'K': int(mg['K']), 'groups': mg['groups'], 'level_share': level_share(wg, chans), 'anchor_flipped': mg['anchor_flipped']}
        except Exception as e: d['L_grp_failure'] = repr(e)
        gv = SB.random_tie_marks(Z, off, .2, key_g[:, :G])
        if not marks_ok(gv): hard_stop('group mark counts')
        truG = SA.truth(gv[pf], labels[pf]); wO = SB.mle_weights(truG['psi'], truG['eta']); d['truth'] = truG; d['w_oracle'] = wO
        if wO.sum() > 0: out[f'B_oracle__{c}'] = Z @ (wO / wO.sum())
        if G >= 2:
            d['dep_marks'] = C2.class_conditional_corr(gv[pf].astype(float), labels[pf]); d['dep_scores'] = C2.class_conditional_corr(Z[pf], labels[pf])
        if G < 3: d['B_sml_failure'] = f'{G} groups'; continue
        try: estG = SA.em_estimate(gv[pf], 'ds')
        except Exception as e: d['B_sml_failure'] = 'DS on groups: ' + repr(e); continue
        wG = SB.mle_weights(estG['psi'], estG['eta']); d['est'] = estG; d['ds'] = ds_meta(estG); d['w_est'] = wG; d['bar'] = SB.group_bar(estG, truG)
        if wG.sum() > 0: out[f'B_sml__{c}'] = Z @ (wG / wG.sum())
        else: d['B_sml_failure'] = 'all group weights 0'
    return out, diags
def reason(a, d):
    return d.get('failure') or d.get(f'{a}_failure') or ('B_sml_failure' in d and a == 'B_sml' and d['B_sml_failure']) or 'not produced (all weights 0)'

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
REFS = ['B13_equal', 'B13_lsml', 'ct7', 'fam421', 'step_index', 'B13_equal_pos']
CFG_ARMS = [f'{a}__{c}{s}' for s in ('', '_pos') for c in CONFIGS for a in arms_of(c)]
ALL = REFS + CFG_ARMS
REPLAY_B = {'B_sml__base': 'G1_sml', 'B_equal__base': 'G1_equal', 'B_oracle__base': 'G1_oracle', 'E_equal__base': 'S_equal', 'L_own__base': 'S_lsml'}
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}; fit_log = []; failures = []; diag = []
B_SC = np.load(B_RUN / 'STEP_SCORES.npz'); replay = {}
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
    ref = {'B13_equal': values @ np.full(13, 1 / 13)}
    w, m = fit_fusion_weights(values[fit_rows], FusionRecipe(name='B13_lsml', members=tuple(names), mode='continuous', anchor=0), seed=FIT_SEED); ref['B13_lsml'] = values @ w
    ref['ct7'] = Zs['ct7'].astype(float); ref['fam421'] = fam421; ref['step_index'] = step_pos
    Vpos = answer_standardize(values - SB.position_profile(values, off, pf), off); vpos = SB.random_tie_marks(Vpos, off, .2, KEYS['main'][0])
    if not marks_ok(vpos): hard_stop('position-bank mark counts')
    ref['B13_equal_pos'] = Vpos @ np.full(13, 1 / 13)
    for r, s in ref.items():
        if r in B_SC.files: replay[f'{r}_fold{k}'] = float(np.max(np.abs(s[ev_rows] - B_SC[r][ev_rows])))
        put(r, k, ev_rows, s, cal)
    for bank, V, votes, Tta, sfx in [('main', values, VOTES['main'], TTA, ''), ('pos', Vpos, vpos, tail_marks(Vpos, off, .2, tie_aware=True, centred=True)[0], '_pos')]:
        out, ds = config_chain(V, votes, Tta, KEYS['main'][1], pf, fit_rows, bank, k); diag += ds
        for d in ds:
            c = d['config']
            for a in arms_of(c):
                nm = f'{a}__{c}'
                if nm in out: put(nm + sfx, k, ev_rows, out[nm], cal)
                else: failures.append({'fold': k, 'arm': nm + sfx, 'reason': reason(a, d)})
            fit_log.append({kk: d.get(kk) for kk in ('fold', 'bank', 'config', 'survivors13', 'ds13', 'channels', 'anchor', 'partition', 'members', 'ds', 'w_est', 'w_oracle', 'bar', 'L_own', 'L_grp', 'failure', 'B_sml_failure', 'L_own_failure', 'L_grp_failure')})
        for a, b in REPLAY_B.items():
            if a in out: replay[f'{a}{sfx}_fold{k}'] = float(np.max(np.abs(out[a][ev_rows] - B_SC[b + sfx][ev_rows])))
    msg = f'fold {k}: fit {fitf} cal {cal}'
    for d in [x for x in diag if x['fold'] == k and x['bank'] == 'main']:
        pe = round(d['est']['prevalence'], 3) if 'est' in d else None; pt = round(d['truth']['prevalence'], 3) if 'truth' in d else None
        dm = d.get('dep_marks', {}).get('clean', {}).get('max_abs_offdiag')
        msg += (f"\n   {d['config']:10s} groups {d.get('members')}; prev est/true {pe}/{pt}; w_est {np.round(d.get('w_est', []), 2).tolist()} w_oracle {np.round(d.get('w_oracle', []), 2).tolist()}; "
                f"clean max|r| {None if dm is None else round(dm, 3)}; L_own level share {None if 'L_own' not in d else round(d['L_own']['level_share'], 3)}")
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
if not checks['replays_pass']: hard_stop('replay (stage-B arms or references) or write-once check failed')

# ------------------------------------------------------------------ diagnostics: estimates per family, residual dependence, partitions
erows = []; deprows = []; parts = {}
for d in diag:
    key = {'fold': d['fold'], 'bank': d['bank'], 'config': d['config']}
    for g, mem in enumerate(d.get('members') or []):
        r = key | {'group': g, 'members': ' + '.join(mem), 'is_level': any(x in LEVEL for x in mem)}
        if 'truth' in d: r.update({'psi_true': d['truth']['psi'][g], 'eta_true': d['truth']['eta'][g], 'pi_true': d['truth']['pi'][g], 'prevalence_true': d['truth']['prevalence'], 'w_oracle': d['w_oracle'][g]})
        if 'est' in d:
            r.update({'psi_hat': d['est']['psi'][g], 'eta_hat': d['est']['eta'][g], 'pi_hat': d['est']['pi'][g], 'prevalence_hat': d['est']['prevalence'], 'w_est': d['w_est'][g],
                      'psi_err': d['est']['psi'][g] - d['truth']['psi'][g], 'eta_err': d['est']['eta'][g] - d['truth']['eta'][g], 'prevalence_err': d['est']['prevalence'] - d['truth']['prevalence'],
                      'w_ratio_est_over_oracle': d['w_est'][g] / d['w_oracle'][g] if d['w_oracle'][g] > 0 else np.nan})
        erows.append(r)
    for src in ('dep_marks', 'dep_scores'):
        if src in d:
            deprows.append(key | {'object': 'marks' if src == 'dep_marks' else 'continuous_scores', 'groups': len(d['members']),
                                  **{f'{cl}_{s}': d[src][cl][s] for cl in ('clean', 'error') for s in ('max_abs_offdiag', 'mean_abs_offdiag')},
                                  'clean_matrix': json.dumps(np.round(d[src]['clean']['matrix'], 4).tolist()), 'error_matrix': json.dumps(np.round(d[src]['error']['matrix'], 4).tolist())})
    parts[f"{d['bank']}_{d['config']}_fold{d['fold']}"] = {kk: d.get(kk) for kk in ('channels', 'members', 'anchor', 'bar', 'failure', 'B_sml_failure')} | {
        'L_own_level_share': d['L_own']['level_share'] if 'L_own' in d else None, 'L_grp_level_share': d['L_grp']['level_share'] if 'L_grp' in d else None, 'L_own_K': d['L_own']['K'] if 'L_own' in d else None}
summ = {}
for bank in ('main', 'pos'):
    for c in CONFIGS:
        ds = [d for d in diag if d['bank'] == bank and d['config'] == c]
        summ[f'{bank}_{c}'] = {'partition_identical_5_folds': len({json.dumps(d.get('members')) for d in ds}) == 1, 'n_groups': [len(d.get('members') or []) for d in ds],
                               'prevalence_err': [d['est']['prevalence'] - d['truth']['prevalence'] if 'est' in d else None for d in ds],
                               'clean_max_abs_r_marks': [d['dep_marks']['clean']['max_abs_offdiag'] if 'dep_marks' in d else None for d in ds],
                               'bar_passes': [(d.get('bar') or {}).get('passes') for d in ds]}
parts['summary'] = summ
pd.DataFrame(erows).to_csv(OUT / 'ESTIMATES.csv', index=False); pd.DataFrame(deprows).to_csv(OUT / 'DEPENDENCE.csv', index=False); dump(OUT / 'PARTITIONS.json', parts)

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
if FOLDS == list(range(5)):                                                   # stage-B metrics must reproduce on the full run
    MB = pd.read_csv(B_RUN / 'METRICS.csv'); mism = {}
    for mine, theirs in [('B13_equal', 'B13_equal'), ('B13_lsml', 'B13_lsml'), ('ct7', 'ct7'), ('fam421', 'fam421'), ('B13_equal_pos', 'B13_equal_pos')] + [(a, b) for a, b in REPLAY_B.items()] + [(a + '_pos', b + '_pos') for a, b in REPLAY_B.items()]:
        for met in ('within_auc', 'prmscore'):
            x = MB[(MB.method == theirs) & (MB.metric == met) & (MB.stratum == 'all')].estimate.item(); y = M[(M.method == mine) & (M.metric == met) & (M.stratum == 'all')].estimate.item(); mism[f'{mine}_{met}'] = abs(x - y)
    checks['stage_B_metrics_max_abs_diff'] = max(mism.values()); checks['stage_B_metrics_diffs'] = mism
    if checks['stage_B_metrics_max_abs_diff'] > 1e-9: hard_stop('stage-B reference metrics not reproduced')

# ------------------------------------------------------------------ paired source-group bootstrap (as stage A/B) + medians over the ten removal configurations
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
RMS = list(RM)
PRIMARY = {'C1_Bsml_vs_B13equal': [(f'B_sml__{r}', 'B13_equal') for r in RMS], 'C2_Bsml_vs_Bequal': [(f'B_sml__{r}', f'B_equal__{r}') for r in RMS],
           'C3_Lown_vs_Eequal': [(f'L_own__{r}', f'E_equal__{r}') for r in RMS], 'C4_Lown_vs_Slsml': [(f'L_own__{r}', 'L_own__base') for r in RMS],
           'C5_merge_Lgrp_vs_Slsml': [('L_grp__merge', 'L_own__base')]}
POSMED = {k + '_pos': [(a + '_pos', b + '_pos') for a, b in v] for k, v in PRIMARY.items()}
sec = []
for r in RMS:
    sec += [(f'E_equal__{r}', 'E_equal__base'), (f'L_grp__{r}', f'E_equal__{r}'), (f'B_oracle__{r}', f'B_equal__{r}'), (f'E_equal__{r}', 'B13_equal')]
    sec += [(f'{a}__{r}', ref) for a in ('B_sml', 'L_own') for ref in ('ct7', 'fam421')]
sec += [('B_sml__merge', 'B_sml__base'), ('B_sml__merge', 'B13_equal'), ('B_sml__merge', 'B_equal__merge'), ('L_grp__merge', 'E_equal__base'), ('L_grp__base', 'E_equal__base'),
        ('B_sml__base', 'B13_equal'), ('L_own__base', 'E_equal__base'), ('E_equal__base', 'B13_equal')]
PAIRS = list(dict.fromkeys([p for v in PRIMARY.values() for p in v] + [p for v in POSMED.values() for p in v] + sec))
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
pt = {}
for p in PAIRS:
    a, b = p; d = prep[p]
    if d is None: continue
    hc = d['hcnt'].sum(0)
    pt[p] = {'auc': float((d[a]['auc'].sum() - d[b]['auc'].sum()) / d['cnt'].sum()), 'ps': float(prmscore_from_counts(*d[a]['conf'].sum(0)) - prmscore_from_counts(*d[b]['conf'].sum(0))),
             'sla': float(np.mean((d[a]['hit'].sum(0) - d[b]['hit'].sum(0))[hc > 0] / hc[hc > 0]))}
rows = []; pbrows = []; deltas = {}
EPN = [('prm_within_auc', 'auc'), ('prmscore', 'ps')]
for fam, groups_of in [('primary', PRIMARY), ('position_bank', POSMED)]:
    for cid, plist in groups_of.items():
        ok = [p for p in plist if prep.get(p)]
        if not ok: rows.append({'family': fam, 'contrast_id': cid, 'note': 'NOT_ESTIMABLE'}); continue
        for ep, key in EPN + [('pb_sla_macro8', 'sla')]:
            X = np.stack([dl[p][key] for p in ok]); med = np.median(X, 0) if len(ok) > 1 else X[0]; pts = [pt[p][key] for p in ok]
            r = {'family': fam, 'contrast_id': cid, 'endpoint': ep, 'configs': len(ok), 'delta': float(np.median(pts)), 'min_config': float(min(pts)), 'max_config': float(max(pts)),
                 'configs_positive': int(sum(v > 0 for v in pts)), 'ci95_lo': float(np.nanquantile(med, .025)), 'ci95_hi': float(np.nanquantile(med, .975)), 'B': DRAWS}
            if fam == 'primary' and ep != 'pb_sla_macro8':
                r.update({'family_K': 10, 'ci_adj_lo': float(np.quantile(med, .05 / 10 / 2)), 'ci_adj_hi': float(np.quantile(med, 1 - .05 / 10 / 2))})
            deltas[f'{cid}__{ep}'] = med.astype(np.float32); (pbrows if ep == 'pb_sla_macro8' else rows).append(r)
for p in PAIRS:
    a, b = p
    if not prep.get(p): rows.append({'family': 'pair', 'contrast_id': f'{a} - {b}', 'note': 'NOT_ESTIMABLE'}); continue
    for ep, key in EPN:
        x = dl[p][key]; rows.append({'family': 'pair', 'contrast_id': f'{a} - {b}', 'endpoint': ep, 'folds': ','.join(map(str, prep[p]['folds'])), 'delta': pt[p][key], 'ci95_lo': float(np.quantile(x, .025)), 'ci95_hi': float(np.quantile(x, .975)), 'B': DRAWS})
    x = dl[p]['sla']; pbrows.append({'family': 'pair', 'contrast_id': f'{a} - {b}', 'endpoint': 'pb_sla_macro8', 'delta': pt[p]['sla'], 'ci95_lo': float(np.nanquantile(x, .025)), 'ci95_hi': float(np.nanquantile(x, .975)), 'B': DRAWS})
pd.DataFrame(rows).to_csv(OUT / 'CONTRASTS.csv', index=False); pd.DataFrame(pbrows).to_csv(OUT / 'PB_CONTRASTS.csv', index=False)
np.savez_compressed(OUT / 'BOOTSTRAP_DELTAS.npz', **deltas, seed=SEED, draws=DRAWS)
timing['bootstrap_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ nulls for the five primary contrasts (PRMBench within-AUC, main bank)
t = time.perf_counter()
NULL_ARMS = sorted({m for v in PRIMARY.values() for p in v for m in p})
ok_ans = np.array([eligible[i] and all(np.isfinite(scores[m][off[i]:off[i+1]]).all() for m in NULL_ARMS) for i in range(n)])
ans_idx = np.flatnonzero(ok_ans); Smat = np.column_stack([scores[m] for m in NULL_ARMS])
Rk, loc = C2.within_ranks(Smat, off, ans_idx); yobs = np.concatenate([labels[off[i]:off[i+1]] for i in ans_idx]).astype(float)
col = {m: j for j, m in enumerate(NULL_ARMS)}
def stats_of(y):
    A = np.nanmean(C2.auc_from_ranks(Rk, loc, y), 0)
    return {cid: float(np.median([A[col[a]] - A[col[b]] for a, b in plist])) for cid, plist in PRIMARY.items()}
obs = stats_of(yobs); nulls = {'answers': int(len(ans_idx)), 'observed_on_null_set': obs}
for nm, fn, sd in [('within_answer_shuffle', C2.shuffle_within, 11), ('whole_answer_same_length_swap', C2.swap_same_length, 12)]:
    rg = np.random.default_rng(sd); draws = [stats_of(fn(yobs, loc, rg)) for _ in range(int(os.environ.get('ER_NULL_PERMS', 200)))]
    nulls[nm] = {cid: {'mean': float(np.mean([x[cid] for x in draws])), 'sd': float(np.std([x[cid] for x in draws])), 'p01': float(np.quantile([x[cid] for x in draws], .01)),
                       'p99': float(np.quantile([x[cid] for x in draws], .99)), 'share_ge_observed': float(np.mean([x[cid] >= obs[cid] for x in draws])),
                       'share_le_observed': float(np.mean([x[cid] <= obs[cid] for x in draws]))} for cid in PRIMARY}
dump(OUT / 'NULLS.json', nulls); timing['nulls_s'] = time.perf_counter() - t
timing['total_s'] = time.perf_counter() - T0; dump(OUT / 'TIMING.json', timing)
status.update({'status': 'COMPLETE' if not failures else 'COMPLETE_WITH_FAILED_FITS', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': failures,
               'checks': {k: v for k, v in checks.items() if not k.startswith('written_once')}, 'fitted_folds': fitted_folds})
dump(OUT / 'RUN_STATUS.json', status)
Mx = M[M.stratum.isin(['all', 'macro8']) & ~M.method.str.endswith('_pos')].pivot(index='method', columns='metric', values='estimate')
print(Mx[['within_auc', 'prmscore', 'sla']].sort_values('within_auc', ascending=False).round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.family.isin(['primary', 'position_bank'])][['family', 'contrast_id', 'endpoint', 'delta', 'min_config', 'max_config', 'configs_positive', 'ci_adj_lo', 'ci_adj_hi', 'ci95_lo', 'ci95_hi']].round(4).to_string())
print(json.dumps({k: v for k, v in nulls.items() if k != 'answers'}, indent=1)[:3000]); print(json.dumps(timing, indent=1)); print('status', status['status'], 'failures', len(failures))
