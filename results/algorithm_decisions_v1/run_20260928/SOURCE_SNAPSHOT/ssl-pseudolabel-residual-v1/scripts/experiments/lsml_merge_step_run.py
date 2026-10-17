"""lsml_merge_step_v1: L-SML unchanged, plus ONE label-free step between its grouping and its weights (merge groups by the
absorption ratio, lsml_merge_step.absorb_merge), on five banks (B13, B16 = B13 + the three digit features, B20, B32, B51);
and Omri's band rule for the Dawid-Skene estimates (drop 0.45 <= pi_hat <= 0.55, flip pi_hat < 0.45).
Frozen protocol: results/lsml_merge_step_v1/PROTOCOL.json.  Population frame, banks, marks, DS filter, folds, evaluation,
bootstrap and nulls are those of the reviewed er_generality_run.py.  Smoke: ER_FOLDS=0 ER_DRAWS=2000 ER_NULL_PERMS=5.

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
from scipy.stats import rankdata, trim_mean  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC  # noqa: E402
import er_stage_a as SA  # noqa: E402
import er_stage_b as SB  # noqa: E402
import er_stage_b2 as C2  # noqa: E402
import lsml_merge_step as MS  # noqa: E402
from calfix_common import tail_marks  # noqa: E402
TPFW = MAIN / '.worktrees/token-probability-fusion-v1'

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else 'run_20260928'
STAGE = ROOT / 'results/expectation_realization_v1'; GEN = ROOT / 'results/lsml_merge_step_v1'; OUT = GEN / RUN_ID; OUT.mkdir(parents=True, exist_ok=True)
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
if (OUT / 'RUN_STATUS.json').exists() and not SMOKE: raise SystemExit(f'{OUT} already holds a run; pass a new run id')
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
A0 = names.index('q15_H1'); assert A0 == 0
w421 = np.array([{'H0lim': 1 / 12, 've0': 1 / 12, 've0.75': 1 / 12, 've1': 1 / 12, 'H0lim_prefix_innovation': 1 / 6, 'bocpd_residual': 1 / 6, 'chosen_token_z_despiked': 1 / 3}[c] for c in pnames])
fam421 = answer_standardize(prof, off) @ w421
kneed = np.maximum(1, np.ceil(.2 * ns)).astype(int)
def marks_ok(v): return bool(np.all(np.add.reduceat((v > 0).astype(np.int64), off[:-1], axis=0) == kneed[:, None]))
timing['channels_s'] = time.perf_counter() - t

# ------------------------------------------------------------------ banks: B13, B16 (+ digit features), and the pre-existing pool banks of er_generality_v1
SCR = Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad')
IND_RUN = ROOT / 'results/indbank_lsml_prmbench_v1'
EG_SC_PATH = ROOT / 'results/er_generality_v1/run_20260927/STEP_SCORES.npz'                 # gitignored, local copy (sha recorded)
B2_SC_PATH = STAGE / 'run_20260927_stage_b2/STEP_SCORES.npz'
EXTRA = {'pool_z': SCR / 'pool_z.npy', 'pool_names': SCR / 'pool_names.json', 'pool_structure': IND_RUN / 'POOL_STRUCTURE.csv',
         'digit_features': ROOT / 'results/digit_family_extension_v1/FEATURES.npz', 'er_generality_step_scores': EG_SC_PATH, 'stage_b2_step_scores': B2_SC_PATH}
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
DNAMES = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']                      # spectral_utils/digit_feature_family.NAMES
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
BANKS = {'B13': (values, names), 'B16': (np.column_stack([values, D3]), names + DNAMES),
         'B20': (answer_standardize(POOL[:, [PN.index(c) for c in names11 + ADD20]], off), names11 + ADD20),
         'B32': (answer_standardize(np.column_stack([POOL[:, :11], POOL[:, [PN.index(c) for c in IND]] * lf_sign]), off), names11 + ['lf__' + c for c in IND]),
         'B51': (answer_standardize(POOL[:, LIVE], off), [PN[j] for j in LIVE])}
for bk, (V, nm) in BANKS.items():
    if not (np.isfinite(V).all() and nm[0] == 'q15_H1' and len(set(nm)) == len(nm)): hard_stop(f'bank {bk} malformed')
    if (V.std(0) <= 1e-12).any(): hard_stop(f'bank {bk} has a zero-variance channel')
dump(OUT / 'INPUT_MANIFEST.json', {k: {'path': str(v), 'bytes': v.stat().st_size, 'sha256': hashes[k]} for k, v in (INPUTS | EXTRA).items()} | {'banks': {bk: nm for bk, (V, nm) in BANKS.items()},
        'population': {'answers': n, 'prm': int(prm.sum()), 'eligible': int(eligible.sum()), 'noncontrol': int(noncontrol.sum()), 'steps': S, 'pb_erroneous': int((pb & (target >= 0)).sum())}})
snap = OUT / 'SOURCE_SNAPSHOT'; code = {}
for rel, base in [('scripts/experiments/lsml_merge_step_run.py', ROOT), ('scripts/experiments/lsml_merge_step.py', ROOT), ('scripts/experiments/er_generality_run.py', ROOT), ('scripts/experiments/er_stage_b2.py', ROOT),
                  ('scripts/experiments/er_stage_b.py', ROOT), ('scripts/experiments/er_stage_a.py', ROOT), ('scripts/experiments/tail_calib_common.py', ROOT), ('scripts/experiments/calfix_common.py', ROOT),
                  ('spectral_utils/lsml_gate_locator_research.py', DEPTH), ('spectral_utils/fusion_utils.py', DEPTH), ('spectral_utils/prmbench.py', DEPTH),
                  ('scripts/experiments/cvf_v2/em.py', MAIN / '.worktrees/cumulative-vote-fusion-v2'), ('scripts/experiments/cvf_v2/core.py', MAIN / '.worktrees/cumulative-vote-fusion-v2')]:
    dst = snap / base.name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / rel, dst); code[f'{base.name}/{rel}'] = sha(base / rel)
git = lambda cwd, *a: subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
dump(OUT / 'CODE_MANIFEST.json', {'files': code, 'git_head_ssl_worktree': git(ROOT, 'rev-parse', 'HEAD'), 'git_head_depth_worktree': git(DEPTH, 'rev-parse', 'HEAD'), 'protocol_sha256': sha(GEN / 'PROTOCOL.json')})
EG = np.load(EG_SC_PATH); B2 = np.load(B2_SC_PATH)
MARKS = {}; TTA = {}; TTAN = {}
t = time.perf_counter()
for bk, (V, nm) in BANKS.items():
    MARKS[bk] = SB.random_tie_marks(V, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
    if not marks_ok(MARKS[bk]): hard_stop(f'mark counts {bk}')
    TTA[bk] = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]; TTAN[bk] = tail_marks(-V, off, .2, tie_aware=True, centred=True)[0]
timing['banks_s'] = time.perf_counter() - t
print('banks ready:', {bk: len(nm) for bk, (V, nm) in BANKS.items()}, f'({timing["banks_s"]:.0f}s); checks {checks}', flush=True)

# ------------------------------------------------------------------ the chain: L-SML unchanged + the merge step; band rule
LEVEL = {'q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level'}          # names for reporting only
def lsml_w(Xf, sn, anchor, g=None):
    return fit_fusion_weights(Xf, FusionRecipe(name='lsml', members=tuple(sn), mode='continuous', anchor=anchor, groups=None if g is None else tuple(int(v) for v in g)), seed=FIT_SEED)
def members(g, sn): return [[sn[j] for j in np.flatnonzero(g == h)] for h in range(int(g.max()) + 1)]
def level_groups(g, sn): return len({int(g[j]) for j, c in enumerate(sn) if c.lstrip('-') in LEVEL})
def part_rec(g, sn, seq=None): return {'labels': g.tolist(), 'groups': members(g, sn), 'K': int(g.max()) + 1, 'level_groups': level_groups(g, sn)} | ({'merge_log': seq} if seq is not None else {})
def family(pre, X, T, sn, anchor, fit_rows, want, d, out):
    """Arms on the oriented channel matrix X (S x p; its tie-aware marks T or None).  want: subset of
    {'equal','lsml','lsml_mC','GmC','lsml_B','lsml_mB','Gbin','GmB'}.  Every L-SML fit records its K and the stages
    that L-SML's small-m guard replaced by equal SD weights (exactly 3 units), so a K=3 cross stage is visible."""
    p = X.shape[1]; Xf = X[fit_rows]
    if 'equal' in want: out[pre + 'equal'] = X @ np.full(p, 1 / p)
    def fit_arm(arm, g=None):
        try:
            w, ml = lsml_w(Xf, sn, anchor, g); out[pre + arm] = X @ w; d[pre + 'w_' + arm] = dict(zip(sn, w))
            d[pre + 'fit_' + arm] = {'K': int(ml['K']), 'small_m_guarded': ml['small_m_guarded'], 'small_m_flags': ml['small_m_flags'], 'anchor_flipped': ml['anchor_flipped']}
            return ml
        except Exception as e:
            d[pre + arm + '_failure'] = repr(e); return None
    def gscore(arm, g):
        try: out[pre + arm] = SB.group_scores(X, off, g, answer_standardize).mean(1)
        except Exception as e: d[pre + arm + '_failure'] = repr(e)
    if want & {'lsml', 'lsml_mC', 'GmC'}:
        ml = fit_arm('lsml'); gc = None if ml is None else MS.canon(ml['groups'])
        if gc is not None: d[pre + 'part_cont'] = part_rec(gc, sn)
        if gc is not None and want & {'lsml_mC', 'GmC'}:
            try: gm, seq = MS.absorb_merge(np.corrcoef(Xf, rowvar=False), gc)
            except Exception as e: d[pre + 'mergeC_failure'] = repr(e); gm = None
            if gm is not None:
                d[pre + 'part_contM'] = part_rec(gm, sn, seq); d[pre + 'merged_cont'] = not np.array_equal(gm, gc)
                if 'lsml_mC' in want and fit_arm('lsml_mC', gm) is not None and np.array_equal(gm, gc):
                    d[pre + 'unmerged_identity_diff'] = float(np.max(np.abs(out[pre + 'lsml_mC'] - out[pre + 'lsml'])))
                if 'GmC' in want: gscore('GmC', gm)
    if want & {'lsml_B', 'lsml_mB', 'Gbin', 'GmB'}:
        try:
            gb = MS.canon(TC.lsml_fit_scaled(T[fit_rows], anchor, Xf, standardize=True, loading_scale='unit')['groups']); d[pre + 'part_bin'] = part_rec(gb, sn)
        except Exception as e:
            d[pre + 'bin_failure'] = repr(e); gb = None
        if gb is not None:
            if 'Gbin' in want: gscore('Gbin', gb)
            if 'lsml_B' in want: fit_arm('lsml_B', gb)
            if want & {'lsml_mB', 'GmB'}:
                try: gbm, seqb = MS.absorb_merge(np.corrcoef(T[fit_rows], rowvar=False), gb)
                except Exception as e: d[pre + 'mergeB_failure'] = repr(e); gbm = None
                if gbm is not None:
                    d[pre + 'part_binM'] = part_rec(gbm, sn, seqb); d[pre + 'merged_bin'] = not np.array_equal(gbm, gb)
                    if 'lsml_mB' in want: fit_arm('lsml_mB', gbm)
                    if 'GmB' in want: gscore('GmB', gbm)
FULL = {'equal', 'lsml', 'lsml_mC', 'GmC', 'lsml_B', 'lsml_mB', 'Gbin', 'GmB'}
def chain(bk, V, nm, votes, pf, fit_rows, k, pos=False):
    out = {}; d = {'fold': k, 'bank': bk, 'position_adjusted': pos}; m = V.shape[1]
    if not pos:
        out['ALL_equal'] = V @ np.full(m, 1 / m)
        try: w, ml = lsml_w(V[fit_rows], nm, 0); out['ALL_lsml'] = V @ w; d['w_ALL_lsml'] = dict(zip(nm, w)); d['ALL_lsml_K'] = int(ml['K']); d['fit_ALL_lsml'] = {'K': int(ml['K']), 'small_m_guarded': ml['small_m_guarded']}
        except Exception as e: d['ALL_lsml_failure'] = repr(e)
    tru = SA.truth(votes[pf], labels[pf]); d['truth'] = tru
    try: est = SA.em_estimate(votes[pf], 'ds')
    except Exception as e: d['failure'] = 'DS: ' + repr(e); return out, d
    d['est'] = est; surv = np.flatnonzero(est['pi'] > 0.5); keep, flip, drop = MS.band_select(est['pi'])
    d['survivors'] = [nm[j] for j in surv]; d['band'] = {'keep': [nm[j] for j in keep], 'flip': [nm[j] for j in flip], 'drop': [nm[j] for j in drop]}
    if len(surv) >= 3:
        if A0 not in surv:
            if not pos: hard_stop(f'{bk} fold {k}: anchor q15_H1 filtered out')
            d['DSF_failure'] = 'anchor filtered out (position-adjusted bank)'; out['DSF_equal'] = V[:, surv].mean(1)
        else:
            anchor = int(np.flatnonzero(surv == A0)[0]); sn = [nm[j] for j in surv]
            T = TTA[bk][:, surv] if not pos else tail_marks(V[:, surv], off, .2, tie_aware=True, centred=True)[0]
            family('DSF_', V[:, surv], T, sn, anchor, fit_rows, FULL if not pos else {'equal', 'lsml', 'lsml_mC', 'lsml_mB'}, d, out)
            if not pos:                                                                   # diagnosis only: class-conditional dependence between / within groups
                cc = C2.class_conditional_corr(V[pf][:, surv], labels[pf]); cc['pooled'] = {'matrix': np.corrcoef(V[pf][:, surv], rowvar=False)}; d['dependence'] = {}
                for src in ('part_cont', 'part_contM', 'part_bin', 'part_binM'):
                    if 'DSF_' + src in d:
                        g = np.asarray(d['DSF_' + src]['labels']); d['dependence'][src] = {cls: MS.dependence_split(cc[cls]['matrix'], g) for cls in ('clean', 'error', 'pooled')}
    else: d['DSF_failure'] = f'{len(surv)} survivors'
    if not pos and len(keep) >= 3: out['BD_equal'] = V[:, keep].mean(1)
    bcols = np.sort(np.concatenate([keep, flip])); sg = np.where(np.isin(bcols, flip), -1.0, 1.0)
    if len(bcols) >= 3:
        Xb = V[:, bcols] * sg; bn = [('-' if s < 0 else '') + nm[j] for j, s in zip(bcols, sg)]
        if pos: out['BF_equal'] = Xb.mean(1)
        elif A0 in keep:
            Tb = np.where(sg > 0, TTA[bk][:, bcols], TTAN[bk][:, bcols])
            family('BF_', Xb, Tb, bn, int(np.flatnonzero(bcols == A0)[0]), fit_rows, {'equal', 'lsml', 'lsml_mB'}, d, out)
        else: out['BF_equal'] = Xb.mean(1); d['BF_failure'] = 'anchor not kept by the band rule'
    else: d['BF_failure'] = f'{len(bcols)} channels after the band rule'
    return out, d

# ------------------------------------------------------------------ fits: eval fold k, cal (k+1)%5, fit = the other three
ARMS = ['ALL_equal', 'ALL_lsml', 'DSF_equal', 'DSF_lsml', 'DSF_lsml_mC', 'DSF_lsml_B', 'DSF_lsml_mB', 'DSF_Gbin', 'DSF_GmB', 'DSF_GmC', 'BD_equal', 'BF_equal', 'BF_lsml', 'BF_lsml_mB']
POSARMS = ['DSF_equal', 'DSF_lsml', 'DSF_lsml_mC', 'DSF_lsml_mB', 'BF_equal']
REFS = ['ct7', 'fam421', 'step_index']
ALL = REFS + [f'{bk}__{a}' for bk in BANKS for a in ARMS] + [f'{bk}__{a}_pos' for bk in BANKS for a in POSARMS]
scores = {m: np.full(S, np.nan) for m in ALL}; written = {m: np.zeros(S, np.int8) for m in ALL}
tau = {m: {} for m in ALL}; fitted_folds = {m: [] for m in ALL}; failures = []; diag = []; replay = {}; replay_b2 = {}
def cal_tau(s_full, cal):
    """PRMScore threshold of the fold-k model: 0.8 quantile of its answer-z scores on the calibration fold's PRMBench answers."""
    return float(np.quantile(np.concatenate([zt(s_full[off[i]:off[i+1]]) for i in np.flatnonzero(prm & (fold == cal))]), .8))
def put(m, k, ev_rows, s_full, cal):
    if written[m][ev_rows].any(): raise AssertionError(f'{m}: evaluation rows of fold {k} already written')
    if not np.isfinite(s_full).all(): failures.append({'fold': k, 'arm': m, 'reason': 'non-finite scores'}); return
    scores[m][ev_rows] = s_full[ev_rows]; written[m][ev_rows] += 1; tau[m][k] = cal_tau(s_full, cal); fitted_folds[m].append(k)
EG_ARMS = ['ALL_equal', 'ALL_lsml', 'DSF_equal', 'DSF_lsml', 'DSF_Gbin']
for k in FOLDS:
    t = time.perf_counter(); cal = (k + 1) % 5; fitf = [f for f in range(5) if f not in (k, cal)]
    fit_rows = rows_of(np.isin(fold, fitf)); ev_rows = rows_of(fold == k); pf = fit_rows[prm_steps[fit_rows]]
    for r, s in {'ct7': Zs['ct7'].astype(float), 'fam421': fam421, 'step_index': step_pos}.items(): put(r, k, ev_rows, s, cal)
    msg = f'fold {k}: fit {fitf} cal {cal}'
    for bk, (V, nm) in BANKS.items():
        out, d = chain(bk, V, nm, MARKS[bk], pf, fit_rows, k)
        Vp = answer_standardize(V - SB.position_profile(V, off, pf), off); vp = SB.random_tie_marks(Vp, off, .2, np.random.default_rng(20260928).random((S, V.shape[1])))
        if not marks_ok(vp): hard_stop(f'position-bank mark counts {bk}')
        outp, dp = chain(bk, Vp, nm, vp, pf, fit_rows, k, pos=True)
        diag += [d, dp]
        for a in ARMS:
            if a in out: put(f'{bk}__{a}', k, ev_rows, out[a], cal)
            else: failures.append({'fold': k, 'arm': f'{bk}__{a}', 'reason': {kk: v for kk, v in d.items() if kk.endswith('failure')} or 'not produced'})
        for a in POSARMS:
            if a in outp: put(f'{bk}__{a}_pos', k, ev_rows, outp[a], cal)
            else: failures.append({'fold': k, 'arm': f'{bk}__{a}_pos', 'reason': {kk: v for kk, v in dp.items() if kk.endswith('failure')} or 'not produced'})
        # ---- replays (definition checks)
        if bk != 'B16':
            for a in EG_ARMS:
                if a not in out: hard_stop(f'{bk} {a} missing in fold {k} (replay arm)')
                replay[f'{bk}__{a}_fold{k}'] = float(np.max(np.abs(out[a][ev_rows] - EG[f'{bk}__{a}'][ev_rows])))
            if 'DSF_equal' not in outp: hard_stop(f'{bk} fold {k}: position-adjusted DSF_equal missing (replay arm)')
            replay[f'{bk}__DSF_equal_pos_fold{k}'] = float(np.max(np.abs(outp['DSF_equal'][ev_rows] - EG[f'{bk}__DSF_equal_pos'][ev_rows])))
        for pre, dd, o in (('DSF_', d, out), ('DSF_', dp, outp), ('BF_', d, out)):
            if pre + 'unmerged_identity_diff' in dd: replay[f'{bk}__{pre}unmerged_identity_{"pos" if dd is dp else "raw"}_fold{k}'] = dd[pre + 'unmerged_identity_diff']
        if bk == 'B13':
            if not all(x in d for x in ('DSF_part_bin', 'DSF_part_binM')) or not all(x in out for x in ('DSF_lsml_B', 'DSF_lsml_mB', 'DSF_GmB')): hard_stop(f'B13 fold {k}: stage-B2 replay inputs missing')
            sn = d['survivors']; gb = np.asarray(d['DSF_part_bin']['labels']); gbm = np.asarray(d['DSF_part_binM']['labels'])
            manual = C2.merge_groups_containing(gb, sn, LEVEL); same = bool(np.array_equal(manual, gbm))
            replay_b2[f'fold{k}'] = {'auto_equals_manual_merge': same, 'DSF_lsml_B_vs_L_grp__base': float(np.max(np.abs(out['DSF_lsml_B'][ev_rows] - B2['L_grp__base'][ev_rows])))}
            if same:
                replay_b2[f'fold{k}'] |= {'DSF_lsml_mB_vs_L_grp__merge': float(np.max(np.abs(out['DSF_lsml_mB'][ev_rows] - B2['L_grp__merge'][ev_rows]))),
                                         'DSF_GmB_vs_B_equal__merge': float(np.max(np.abs(out['DSF_GmB'][ev_rows] - B2['B_equal__merge'][ev_rows])))}
            if any(not (np.isfinite(v) and v <= 1e-9) for v in replay_b2[f'fold{k}'].values() if not isinstance(v, bool)): hard_stop(f'B13 fold {k}: stage-B2 replay failed {replay_b2[f"fold{k}"]}')
        msg += (f"\n   {bk}: {len(nm)} ch; DS drop {[c for c in nm if c not in d.get('survivors', [])]}; band flip {d['band']['flip']} drop {d['band']['drop']}; "
                f"cont K {d.get('DSF_part_cont', {}).get('K')}->{d.get('DSF_part_contM', {}).get('K')}; bin K {d.get('DSF_part_bin', {}).get('K')}->{d.get('DSF_part_binM', {}).get('K')}; "
                f"level groups cont {d.get('DSF_part_cont', {}).get('level_groups')}->{d.get('DSF_part_contM', {}).get('level_groups')}, bin {d.get('DSF_part_bin', {}).get('level_groups')}->{d.get('DSF_part_binM', {}).get('level_groups')}")
    print(msg + f' ({time.perf_counter()-t:.0f}s)', flush=True)
timing['fit_s'] = time.perf_counter() - T0
checks['replay_er_generality_max'] = max(v for kk, v in replay.items() if 'unmerged_identity' not in kk)
ident = [v for kk, v in replay.items() if 'unmerged_identity' in kk]; checks['unmerged_identity_max'] = max(ident) if ident else None
checks['replay_per_fold'] = replay; checks['replay_stage_b2'] = replay_b2
for m in ALL: checks[f'written_once_{m}'] = bool(written[m].max() <= 1)
checks['replays_pass'] = bool(all(np.isfinite(v) and v <= (1e-8 if 'unmerged_identity' in kk else 1e-9) for kk, v in replay.items()) and all(checks[f'written_once_{m}'] for m in ALL))
def jd(v): return v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)
with open(OUT / 'FIT_MANIFEST.jsonl', 'w', encoding='utf8') as f:
    for d in diag: f.write(json.dumps({kk: v for kk, v in d.items() if kk not in ('truth', 'est')} | {'pi_hat': d['est']['pi'] if 'est' in d else None, 'prevalence_hat': d['est']['prevalence'] if 'est' in d else None,
                                       'ds_converged': d['est']['converged'] if 'est' in d else None, 'pi_true': d['truth']['pi'], 'prevalence_true': d['truth']['prevalence']}, default=jd) + '\n')
np.savez_compressed(OUT / 'STEP_SCORES.npz', offsets=off, **{m: scores[m] for m in ALL}); dump(OUT / 'THRESHOLDS.json', tau)
pd.DataFrame(failures or [{'fold': '', 'arm': '', 'reason': 'none'}]).to_csv(OUT / 'FAILURES.csv', index=False)
print('checks:', {k: v for k, v in checks.items() if not k.startswith('written_once') and k != 'replay_per_fold'}, flush=True)
if not checks['replays_pass']: hard_stop('er_generality replay, unmerged identity or write-once check failed')

# ------------------------------------------------------------------ channel, partition and dependence diagnostics
crow = []; single = {bk: [mean_within(V[:, j]) for j in range(V.shape[1])] for bk, (V, nm) in BANKS.items()}
for d in diag:
    V, nm = BANKS[d['bank']]
    for j, c in enumerate(nm):
        r = {'bank': d['bank'], 'fold': d['fold'], 'position_adjusted': d['position_adjusted'], 'channel': c, 'pi_true': d['truth']['pi'][j], 'single_within_auc_all_answers': single[d['bank']][j],
             'w_ALL_lsml': d.get('w_ALL_lsml', {}).get(c)}
        if 'est' in d:
            r |= {'pi_hat': d['est']['pi'][j], 'ds_kept': c in d['survivors'], 'band': 'keep' if c in d['band']['keep'] else 'flip' if c in d['band']['flip'] else 'drop'}
            for wk in ('w_lsml', 'w_lsml_mC', 'w_lsml_B', 'w_lsml_mB'): r[f'DSF_{wk}'] = d.get(f'DSF_{wk}', {}).get(c)
            for wk in ('w_lsml', 'w_lsml_mB'): r[f'BF_{wk}_on_oriented_column'] = d.get(f'BF_{wk}', {}).get(c, d.get(f'BF_{wk}', {}).get('-' + c))
        crow.append(r)
pd.DataFrame(crow).to_csv(OUT / 'CHANNELS.csv', index=False)
drow = []
for d in diag:
    for src, dd in (d.get('dependence') or {}).items():
        for cls, sp in dd.items():
            drow.append({'bank': d['bank'], 'fold': d['fold'], 'partition': src, 'class': cls, 'K': d['DSF_' + src]['K'],
                         **{f'{w}_{q}': sp[w][q] for w in ('between', 'within') for q in ('pairs', 'mean_abs', 'max_abs')}})
pd.DataFrame(drow).to_csv(OUT / 'DEPENDENCE.csv', index=False)
parts = {}
for bk in BANKS:
    for pos in (False, True):
        ds = [d for d in diag if d['bank'] == bk and d['position_adjusted'] == pos]; key = bk + ('_pos' if pos else '')
        fz = lambda d, s: frozenset(frozenset(g) for g in d.get(s, {}).get('groups', []))
        parts[key] = {src: {'folds_merged': int(sum(bool(d.get(f'DSF_merged_{src}')) for d in ds)), 'K_before': [d.get(f'DSF_part_{src}', {}).get('K') for d in ds],
                            'K_after': [d.get(f'DSF_part_{src}M', {}).get('K') for d in ds], 'level_groups_before': [d.get(f'DSF_part_{src}', {}).get('level_groups') for d in ds],
                            'level_groups_after': [d.get(f'DSF_part_{src}M', {}).get('level_groups') for d in ds],
                            'identical_across_folds_before': len({fz(d, f'DSF_part_{src}') for d in ds}) == 1, 'identical_across_folds_after': len({fz(d, f'DSF_part_{src}M') for d in ds}) == 1,
                            'per_fold': [{'fold': d['fold'], 'before': d.get(f'DSF_part_{src}', {}).get('groups'), 'after': d.get(f'DSF_part_{src}M', {}).get('groups'),
                                          'merge_log': d.get(f'DSF_part_{src}M', {}).get('merge_log')} for d in ds]} for src in ('cont', 'bin')}
        parts[key]['band'] = [d.get('band') for d in ds]; parts[key]['survivors'] = [d.get('survivors') for d in ds]
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

# ------------------------------------------------------------------ paired source-group bootstrap (as stage A/B)
t = time.perf_counter()
Gpr, ginv_all = np.unique(groups[prm], return_inverse=True); gpr = np.full(n, -1); gpr[prm] = ginv_all
Gpb, gpb_all = np.unique(groups[pb], return_inverse=True); gpbx = np.full(n, -1); gpbx[pb] = gpb_all
CT = [(a, b) for a, b, _ in P['contrasts']['primary']]
prim = [(f'{bk}__{a}', f'{bk}__{b}') for bk in P['contrasts']['primary_banks'] for a, b in CT]
SEC_CT = [('DSF_lsml_mC', 'DSF_equal'), ('DSF_lsml_B', 'DSF_lsml'), ('DSF_lsml_mB', 'DSF_lsml_B'), ('DSF_GmB', 'DSF_Gbin'), ('DSF_GmB', 'DSF_equal'), ('DSF_GmC', 'DSF_equal'),
          ('BD_equal', 'DSF_equal'), ('BF_equal', 'BD_equal'), ('BF_lsml', 'DSF_lsml'), ('BF_lsml_mB', 'DSF_lsml_mB'), ('BF_lsml_mB', 'BF_equal'), ('ALL_lsml', 'ALL_equal')]
sec = [(f'{bk}__{a}', f'{bk}__{b}') for bk in BANKS for a, b in SEC_CT] + [(f'{bk}__{a}_pos', f'{bk}__{b}_pos') for bk in BANKS for a, b in CT]
sec += [(f'{bk}__{a}', ref) for bk in BANKS for a in ('DSF_equal', 'DSF_lsml_mB', 'DSF_lsml_mC', 'BF_equal', 'BF_lsml_mB') for ref in ('ct7', 'fam421')]
PAIRS = list(dict.fromkeys(prim + sec)); K = len(prim) * 2
prep = {}
for a, b in PAIRS:
    F = cov[a] & cov[b]; folds_c = sorted(set(fitted_folds[a]) & set(fitted_folds[b]))
    if not F.any(): prep[(a, b)] = None; continue
    e = F & eligible & np.isfinite(aucA[a]) & np.isfinite(aucA[b]); nc = F & noncontrol; pe = F & pb & (target >= 0)
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

# ------------------------------------------------------------------ nulls and concentration for the primary contrasts (PRMBench within-AUC)
t = time.perf_counter(); nulls = {}; conc = {}; NPERM = int(os.environ.get('ER_NULL_PERMS', 200))
for a, b in prim:
    cid = f'{a} - {b}'
    ans_idx = np.flatnonzero([eligible[i] and np.isfinite(scores[a][off[i]:off[i+1]]).all() and np.isfinite(scores[b][off[i]:off[i+1]]).all() for i in range(n)])
    if not len(ans_idx): nulls[cid] = conc[cid] = {'note': 'NOT_ESTIMABLE'}; continue
    dx = aucA[a][ans_idx] - aucA[b][ans_idx]; sgn = float(np.sign(dx.mean())) or 1.0; top_idx = np.argsort(-sgn * dx, kind='stable')[:int(np.ceil(.01 * len(dx)))]; top = dx[top_idx]
    conc[cid] = {'answers': int(len(dx)), 'mean_delta': float(dx.mean()), 'share_from_top1pct': float(top.sum() / dx.sum()) if dx.sum() != 0 else None,
                 'mean_without_top1pct': float(np.delete(dx, top_idx).mean()), 'tail_direction': 'gain' if sgn > 0 else 'loss', 'trimmed5_mean': float(trim_mean(dx, .05)), 'answers_changed': int((np.abs(dx) > 1e-12).sum()),
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
               'checks': {k: v for k, v in checks.items() if not k.startswith('written_once')}, 'fitted_folds': fitted_folds})
dump(OUT / 'RUN_STATUS.json', status)
Mx = M[M.stratum.isin(['all', 'macro8'])].pivot(index='method', columns='metric', values='estimate')
print(Mx[['within_auc', 'prmscore', 'sla']].round(4).to_string())
print(pd.DataFrame(rows)[lambda d: d.primary == True][['contrast_id', 'endpoint', 'delta', 'ci_adj_lo', 'ci_adj_hi']].round(4).to_string())
print(json.dumps(nulls, indent=1)[:3000]); print(json.dumps(timing, indent=1)); print('status', status['status'], 'failures', len(failures))
