"""Data for REPORT_HE.html of lsml_merge_step_v1 (post hoc, diagnosis and presentation; no new method is fitted).
(1) per-answer within-AUC of every arm and a paired source-group bootstrap of every head-to-head pair inside each bank
    (95%, 4,000 draws, not multiplicity-corrected - exploratory);
(2) per-cell metrics (from METRICS.csv of this run and of er_generality_v1 for its two extra arms);
(3) the L-SML assumption: for the fold-0 partitions of FIT_MANIFEST (fit folds 2, 3, 4, PRMBench steps), class-conditional
    correlations of the DS survivors between and within groups, on continuous values and on top-20% marks, against 2,000 random
    partitions with the same group sizes, and on the position-adjusted bank.
Output: report_data.json next to this file."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1')); sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from spectral_utils.lsml_gate_locator_research import answer_standardize
import er_stage_b as SB, er_stage_b2 as C2
from calfix_common import tail_marks
HERE = Path(__file__).resolve().parent; RUN = HERE / 'run_20260928'; EGR = ROOT / 'results/er_generality_v1/run_20260927'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'; TPF = MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1'
CT7P = MAIN / '.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1'
SCR = Path(r'C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad')
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Zs = np.load(R / 'OOF_STEP_SCORES.npz'); off = Zs['offsets']; lab = Zs['labels'].astype(bool)
n = len(ans); ns = np.diff(off); S = int(off[-1]); prm = (~ans.cell.str.startswith('pb_')).to_numpy(); fold = ans.fold.to_numpy(); grp = ans.source_group.to_numpy()
prm_steps = np.repeat(prm, ns)
elig = np.array([prm[i] and lab[off[i]:off[i+1]].any() and (~lab[off[i]:off[i+1]]).any() for i in range(n)]); E = np.flatnonzero(elig)
out = {}

# ---------------- (1) head-to-head
SC = np.load(RUN / 'STEP_SCORES.npz'); EG = np.load(EGR / 'STEP_SCORES.npz')
ARMS = ['ALL_equal', 'ALL_lsml', 'DSF_equal', 'DSF_lsml', 'DSF_lsml_mC', 'DSF_lsml_B', 'DSF_lsml_mB', 'DSF_Gbin', 'DSF_GmB', 'DSF_GmC', 'BD_equal', 'BF_equal', 'BF_lsml', 'BF_lsml_mB']
BANKS = ['B13', 'B16', 'B20', 'B32', 'B51']
def arms_of(bk):
    cols = {a: SC[f'{bk}__{a}'] for a in ARMS}
    if bk != 'B16':
        for a in ('SF_equal', 'DSF_Gcont'): cols[a] = EG[f'{bk}__{a}']
    cols['ct7'] = SC['ct7']; cols['fam421'] = SC['fam421']
    return cols
Gs, gi = np.unique(grp[E], return_inverse=True); cnt = np.bincount(gi, minlength=len(Gs)).astype(float)
rng = np.random.default_rng(20260928); W = rng.multinomial(len(Gs), np.full(len(Gs), 1 / len(Gs)), size=4000).astype(float); den = W @ cnt
yE = np.concatenate([lab[off[i]:off[i+1]] for i in E]).astype(float)
h2h = {}
for bk in BANKS:
    cols = arms_of(bk); names = list(cols)
    Rk, loc = C2.within_ranks(np.column_stack([cols[a] for a in names]), off, E)
    A = C2.auc_from_ranks(Rk, loc, yE)                                        # answers x methods
    assert np.isfinite(A).all()
    Gm = np.stack([np.bincount(gi, weights=A[:, j], minlength=len(Gs)) for j in range(len(names))], 1)
    Bm = (W @ Gm) / den[:, None]; mean = A.mean(0)
    D = mean[:, None] - mean[None, :]; lo = np.full(D.shape, np.nan); hi = np.full(D.shape, np.nan)
    for a in range(len(names)):
        dd = Bm[:, [a]] - Bm; lo[a] = np.quantile(dd, .025, axis=0); hi[a] = np.quantile(dd, .975, axis=0)
    h2h[bk] = {'methods': names, 'mean': mean.tolist(), 'delta': D.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist()}
    print('h2h', bk, len(names), flush=True)
out['h2h'] = h2h

# ---------------- (2) metrics per cell / class
M = pd.read_csv(RUN / 'METRICS.csv'); MG = pd.read_csv(EGR / 'METRICS.csv')
M = pd.concat([M, MG[MG.method.str.contains('__SF_equal$|__DSF_Gcont$')]], ignore_index=True)
out['metrics'] = M[['method', 'benchmark', 'metric', 'stratum', 'N', 'estimate']].to_dict('records')
out['contrasts'] = pd.read_csv(RUN / 'CONTRASTS.csv').to_dict('records')
out['pb_contrasts'] = pd.read_csv(RUN / 'PB_CONTRASTS.csv').to_dict('records')
out['nulls'] = json.loads((RUN / 'NULLS.json').read_text(encoding='utf8')); out['concentration'] = json.loads((RUN / 'CONCENTRATION.json').read_text(encoding='utf8'))
out['posthoc_digits'] = json.loads((HERE / 'POSTHOC_B16_vs_B13.json').read_text(encoding='utf8'))
out['partitions'] = json.loads((RUN / 'PARTITIONS.json').read_text(encoding='utf8'))

# ---------------- (3) the L-SML assumption (fold 0)
lv = np.load(TPF / 'DERIVATIVE_CHANNELS.npz'); names11 = list(map(str, lv['channels']))
prof = np.load(CT7P / 'profiles.npy').astype(float); pn = json.loads((CT7P / 'PROFILE_VALIDATION.json').read_text(encoding='utf8'))['channels']
raw13 = np.column_stack([lv['level'].astype(float), prof[:, pn.index('chosen_token_z_despiked')], lv['derivative'][:, names11.index('chosen_surprisal')].astype(float)])
V13 = answer_standardize(raw13, off); names13 = names11 + ['realized_z', 'realized_drv']
DF = np.load(ROOT / 'results/digit_family_extension_v1/FEATURES.npz'); act = DF['active'].astype(bool)
D3 = np.zeros((S, 3))
for a, b in zip(off[:-1], off[1:]):
    for j in range(3):
        v = act[a:b, j]; y = DF['values'][a:b, j][v]
        if len(y) and y.std() > 1e-12: D3[a:b, j][v] = (y - y.mean()) / y.std()
POOL = np.load(SCR / 'pool_z.npy'); PN = json.load(open(SCR / 'pool_names.json')); PS = pd.read_csv(ROOT / 'results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv').set_index('channel')
IND = json.loads((ROOT / 'results/indbank_lsml_prmbench_v1/PROTOCOL.json').read_text(encoding='utf8'))['selected_IND_21']
lf = np.array([float(np.sign(PS.loc[c, 'r_level_marginal'])) or 1.0 for c in IND]); LIVE = [j for j in range(52) if POOL[:, j].std() > 1e-12]
ADD20 = ['ct7_chosen_std_excess', 'ct7_bocpd_residual', 'ct7_H0lim_prefix_innovation', 'ct7_ve0', 'H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk']
BK = {'B13': (V13, names13), 'B16': (np.column_stack([V13, D3]), names13 + ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']),
      'B20': (answer_standardize(POOL[:, [PN.index(c) for c in names11 + ADD20]], off), names11 + ADD20),
      'B32': (answer_standardize(np.column_stack([POOL[:, :11], POOL[:, [PN.index(c) for c in IND]] * lf]), off), names11 + ['lf__' + c for c in IND]),
      'B51': (answer_standardize(POOL[:, LIVE], off), [PN[j] for j in LIVE])}
FM = [json.loads(l) for l in open(RUN / 'FIT_MANIFEST.jsonl', encoding='utf8')]
k = 0; fit_rows = np.concatenate([np.arange(off[i], off[i+1]) for i in np.flatnonzero(np.isin(fold, [2, 3, 4]))]); pf = fit_rows[prm_steps[fit_rows]]; y = lab[pf]
def cc(X):
    return {'clean': np.corrcoef(X[~y], rowvar=False), 'error': np.corrcoef(X[y], rowvar=False), 'pooled': np.corrcoef(X, rowvar=False)}
def split(C, g):
    off_ = ~np.eye(len(g), dtype=bool); same = g[:, None] == g[None, :]
    b = np.abs(C[off_ & ~same]); w = np.abs(C[off_ & same])
    return {'between_mean': float(b.mean()) if len(b) else None, 'between_max': float(b.max()) if len(b) else None, 'between_share_gt_0.1': float((b > .1).mean()) if len(b) else None,
            'between_share_gt_0.2': float((b > .2).mean()) if len(b) else None, 'within_mean': float(w.mean()) if len(w) else None, 'between_pairs': int(len(b) // 2)}
assum = {}
for bk, (V, nm) in BK.items():
    d = next(x for x in FM if x['bank'] == bk and x['fold'] == k and not x['position_adjusted'])
    surv = [nm.index(c) for c in d['survivors']]; sn = d['survivors']; X = V[:, surv]
    Vp = answer_standardize(V - SB.position_profile(V, off, pf), off)[:, surv]
    T = tail_marks(X, off, .2, tie_aware=True, centred=False)[0]
    mats = {'values': cc(X[pf]), 'marks': cc(T[pf]), 'values_position_removed': cc(Vp[pf])}
    parts = {src: np.asarray(d['DSF_' + src]['labels']) for src in ('part_cont', 'part_contM', 'part_bin', 'part_binM') if 'DSF_' + src in d}
    res = {'channels': sn, 'partitions': {s: {'labels': g.tolist(), 'groups': d['DSF_' + s]['groups']} for s, g in parts.items()}, 'stats': {}, 'random': {}, 'top_pairs': {}}
    rr = np.random.default_rng(1)
    for mk, mm in mats.items():
        for s, g in parts.items():
            res['stats'][f'{mk}|{s}'] = {cls: split(mm[cls], g) for cls in ('clean', 'error', 'pooled')}
            null = []
            for _ in range(2000):
                gp = rr.permutation(g); null.append(np.mean([split(mm[c], gp)['between_mean'] for c in ('clean', 'error')]))
            null = np.array(null); obs = np.mean([res['stats'][f'{mk}|{s}'][c]['between_mean'] for c in ('clean', 'error')])
            res['random'][f'{mk}|{s}'] = {'observed': float(obs), 'random_mean': float(null.mean()), 'random_p05': float(np.quantile(null, .05)), 'share_random_below': float((null <= obs).mean())}
        g = parts['part_binM']; Cm = (np.abs(mats[mk]['clean']) + np.abs(mats[mk]['error'])) / 2; pairs = []
        for a in range(len(sn)):
            for b in range(a + 1, len(sn)):
                if g[a] != g[b]: pairs.append((float(Cm[a, b]), sn[a], sn[b], float(mats[mk]['clean'][a, b]), float(mats[mk]['error'][a, b])))
        res['top_pairs'][mk] = sorted(pairs, reverse=True)[:8]
    if bk in ('B13', 'B16', 'B20'):
        res['matrices'] = {mk: {cls: np.round(mats[mk][cls], 3).tolist() for cls in ('clean', 'error')} for mk in mats}
    assum[bk] = res; print('assumption', bk, flush=True)
out['assumption'] = assum
out['fold0_steps'] = {'prm_fit_steps': int(len(pf)), 'error_steps': int(y.sum())}
(HERE / 'report_data.json').write_text(json.dumps(out, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
print('done')
