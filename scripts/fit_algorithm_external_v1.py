"""algorithm_external_v1 source fit (label-free; never opens external labels or scores).
Learns, on the 13,769 source answers (folds 0-3), the DS filter, the merged binary partition and the DS-estimate group weights
for the four digit banks (B16, B23, B35, B54); thresholds = 0.8 quantile of the answer-z scores on fold-4 steps; the stopping
rule's (statistic, tau*) from the partition_switch_v1 statistics of all 40 development bank-folds.  Protocol:
results/algorithm_external_v1/PROTOCOL.json.  Output: results/algorithm_external_v1/BUNDLE.json (+ a source fold-4 sanity panel)."""
from pathlib import Path
import hashlib, json, subprocess, sys
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.lsml_gate_locator_research import answer_standardize  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import tail_calib_common as TC, er_stage_a as SA, er_stage_b as SB, lsml_merge_step as MS, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

OUT = ROOT / 'results/algorithm_external_v1'; P = json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf8'))
AD = ROOT / 'results/algorithm_decisions_v1/run_20260928'; FE = ROOT / 'results/external_banks_v4'
BANKS = ['B16', 'B23', 'B35', 'B54']; DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def stop(msg): raise SystemExit('HARD STOP: ' + msg)
if P.get('status') != 'FROZEN': stop('protocol not frozen')
if (OUT / 'BUNDLE.json').exists(): stop('bundle exists; refusing to overwrite')

# ------------------------------------------------------------------ source frame and features
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); off = np.load(R / 'OOF_STEP_SCORES.npz')['offsets']; ns = np.diff(off); S = int(off[-1])
prm = (~ans.cell.str.startswith('pb_')).to_numpy(); fold = ans.fold.to_numpy(); prm_steps = np.repeat(prm, ns); sfold = np.repeat(fold, ns)
F = np.load(FE / 'source/FEATURES.npz'); names = [str(x) for x in F['names']]; X = F['values']; DA = F['digit_active']
if not (np.array_equal(F['offsets'], off) and list(map(str, F['uids'])) == ans.uid.astype(str).tolist()): stop('source feature order differs from the OOF frame')
BN = json.loads((AD / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks']
PS = pd.read_csv(ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')

def masked_std(x, offsets, active):
    out = np.zeros(x.shape)
    for a, b in zip(offsets[:-1], offsets[1:]):
        for j in range(x.shape[1]):
            v = active[a:b, j]; y = x[a:b, j][v]
            if len(y) and y.std() > 1e-12: out[a:b, j][v] = (y - y.mean()) / y.std()
    return out
def build_bank(bk, Xr, nm, dact, offsets):
    """Answer-standardized bank from raw step features (non-digit: answer z; digits: masked answer z; lf__ sign)."""
    cols = BN[bk]; plain = [c for c in cols if c not in DIG]
    src = [c[4:] if c.startswith('lf__') else c for c in plain]
    Vp = answer_standardize(Xr[:, [nm.index(c) for c in src]], offsets)
    for q, c in enumerate(plain):
        if c.startswith('lf__'): Vp[:, q] *= float(np.sign(PS.loc[c[4:], 'r_level_marginal'])) or 1.0
    Vd = masked_std(Xr[:, [nm.index(c) for c in DIG]], offsets, dact)
    out = np.zeros((len(Xr), len(cols))); pi = 0; di = 0
    for j, c in enumerate(cols):
        if c in DIG: out[:, j] = Vd[:, DIG.index(c)]
        else: out[:, j] = Vp[:, plain.index(c)]
    return out

# ------------------------------------------------------------------ definition check vs the development banks (as partition_switch_run.py)
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location('ps_bank_src', ROOT / 'scripts/experiments/partition_switch_run.py')
src_text = (ROOT / 'scripts/experiments/partition_switch_run.py').read_text(encoding='utf8')
bank_def = src_text[src_text.index('MI = json.loads'):src_text.index("_SCZ = np.load")]            # the reviewed bank() construction only
ns_ = {'np': np, 'pd': pd, 'json': json, 'Path': Path, 'answer_standardize': answer_standardize, 'AD': AD, 'off': off, 'S': S, 'sha': sha, 'ROOT': ROOT,
       'hard_stop': stop, 'checks': {}}
exec(compile(bank_def, 'partition_switch_run.bank', 'exec'), ns_)
checks = {}
V = {}
for bk in BANKS:
    V[bk] = build_bank(bk, X, names, DA, off)
    ref = ns_['bank'](bk); checks[f'bank_definition_max_diff_{bk}'] = float(np.max(np.abs(V[bk] - ref)))
    if not checks[f'bank_definition_max_diff_{bk}'] <= 1e-9: stop(f'{bk} differs from the development bank by {checks[f"bank_definition_max_diff_{bk}"]}')
print('banks match development:', {k: f'{v:.1e}' for k, v in checks.items()}, flush=True)

# ------------------------------------------------------------------ the frozen fit (folds 0-3) and fold-4 thresholds
fit_rows = np.flatnonzero(sfold < 4); pf = fit_rows[prm_steps[fit_rows]]; cal_rows = np.flatnonzero(sfold == 4)
KEYG = np.random.default_rng(20260929).random((S, 13))
def answer_z(s, offsets):
    out = np.empty(len(s))
    for a, b in zip(offsets[:-1], offsets[1:]): v = s[a:b]; out[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return out
bundle = {'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'features_manifest_sha256': sha(FE / 'source/MANIFEST.json'), 'fit_folds': [0, 1, 2, 3], 'calibration_fold': 4,
          'population': 'PB+PRMB 13,769 source answers; DS filter and group DS on PRMBench steps of folds 0-3; partition on all steps of folds 0-3; no labels',
          'checks': checks, 'banks': {}}
sanity = {}
for bk in BANKS:
    Vb = V[bk]; nm = BN[bk]; m = Vb.shape[1]
    marks = SB.random_tie_marks(Vb, off, .2, np.random.default_rng(20260928).random((S, m)))
    est = SA.em_estimate(marks[pf], 'ds'); surv = np.flatnonzero(est['pi'] > 0.5); sn = [nm[j] for j in surv]
    if 0 not in surv: stop(f'{bk}: anchor q15_H1 filtered out')
    Xs = Vb[:, surv]; T = tail_marks(Xs, off, .2, tie_aware=True, centred=True)[0]
    gb = MS.canon(TC.lsml_fit_scaled(T[fit_rows], int(np.flatnonzero(surv == 0)[0]), Xs[fit_rows], standardize=True, loading_scale='unit')['groups'])
    gbm, seq = MS.absorb_merge(np.corrcoef(T[fit_rows], rowvar=False), gb); G = int(gbm.max()) + 1
    Z = GW.group_matrix(Xs, off, gbm, None, answer_standardize); gv = SB.random_tie_marks(Z, off, .2, KEYG[:, :G])
    estG = SA.em_estimate(gv[pf], 'ds'); w = SB.mle_weights(estG['psi'], estG['eta'])
    if w.sum() <= 0: stop(f'{bk}: all group weights 0')
    base = answer_z(Xs.mean(1), off); grp = answer_z(Z @ (w / w.sum()), off)
    thr = {'BASE': float(np.quantile(base[cal_rows], .8, method='linear')), 'GRP': float(np.quantile(grp[cal_rows], .8, method='linear'))}
    offd = ~np.eye(len(gbm), dtype=bool); diff = gbm[:, None] != gbm[None, :]; Rm = np.corrcoef(Xs[fit_rows], rowvar=False)
    bundle['banks'][bk] = {'channels': nm, 'survivors': sn, 'dropped': [c for c in nm if c not in sn], 'pi_hat': dict(zip(nm, map(float, est['pi']))), 'prevalence_hat': float(est['prevalence']),
                           'partition_before_merge': [[sn[j] for j in np.flatnonzero(gb == h)] for h in range(int(gb.max()) + 1)],
                           'partition': [[sn[j] for j in np.flatnonzero(gbm == h)] for h in range(G)], 'merge_log': seq, 'group_labels': gbm.tolist(),
                           'group_weights': (w / w.sum()).tolist(), 'group_psi_eta': {'psi': list(map(float, estG['psi'])), 'eta': list(map(float, estG['eta']))},
                           'thresholds': thr, 'source_S1_fit_rows': float(np.abs(Rm[offd & diff]).mean()),
                           'source_S2': float(np.bincount(gbm).max() / len(gbm))}
    print(bk, 'survivors', len(sn), '/', m, 'groups', G, 'weights', np.round(w / w.sum(), 3).tolist(), 'thr', {k: round(v, 3) for k, v in thr.items()}, flush=True)

# ------------------------------------------------------------------ the stopping rule (diagnostic): objective over all 40 development bank-folds
ST = pd.read_csv(ROOT / 'results/partition_switch_v1/run_20260928/STATS.csv'); STATN = ['S1_between_dependence', 'S2_largest_group_share', 'S3_rank_one_misfit']
best = None
for j, sn_ in enumerate(STATN):
    for tau in [-np.inf] + sorted(ST[sn_].unique()) + [np.inf]:
        obj = float(np.mean(ST['gain_EQ_DSM'] * (ST[sn_] <= tau)))
        key = (round(obj, 12), -tau if np.isfinite(tau) else (np.inf if tau < 0 else -np.inf), -j)
        if best is None or key > best[0]: best = (key, sn_, tau, obj)
bundle['stopping_rule'] = {'statistic': best[1], 'tau': best[2], 'objective_all_40_cells': best[3], 'source': 'results/partition_switch_v1/run_20260928/STATS.csv',
                           'note': 'diagnostic only (Step 458: a lookup for the 32-channel family)'}
bundle['code'] = {'fit_script_sha256': sha(Path(__file__)), 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}
(OUT / 'BUNDLE.json').write_text(json.dumps(bundle, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
print('stopping rule', bundle['stopping_rule']); print('BUNDLE written')
