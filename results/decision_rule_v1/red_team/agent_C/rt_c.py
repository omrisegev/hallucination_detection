"""Red-team C: nulls, length-only / shuffled-count allocation, rate-vs-allocation decomposition for decision_rule_v1.
Reads only DECISIONS.npz of the run plus raw inputs. Writes nothing tracked."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
W = MAIN / '.worktrees/decision-rule-v1'
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1'
OUT = Path(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/rt_dr_C')

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz')
off = Z['offsets']; lab = Z['labels'].astype(bool); n = len(ans); ns = np.diff(off); S_ = int(off[-1])
aid = np.repeat(np.arange(n), ns); fold = ans.fold.to_numpy(); ids = ans.id.to_numpy()
prm = ~ans.cell.str.startswith('pb_').to_numpy()
meta_path = Path(json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['prm_metadata']['path'])
meta = {m['idx']: m for m in pickle.load(open(meta_path, 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); nc = prm & ~control
print('prm', prm.sum(), 'controls', control.sum(), 'noncontrol', nc.sum())
D = np.load(W / 'results/decision_rule_v1/run_20260929/DECISIONS.npz')
assert (D['offsets'] == off).all()
F = {r: D[r].astype(bool) for r in ['R0_frozen', 'R1_global', 'R2_allocate', 'R3_ds_map', 'R4_ds_expected_f1', 'R5_ds_count']}
S = np.load(SSL / 'run_20260927_stage_b/STEP_SCORES.npz')['S_equal'].astype(float)
tauS = {int(k): float(v) for k, v in json.loads((SSL / 'run_20260927_stage_b_thr/THRESHOLDS.json').read_text(encoding='utf8'))['S_equal'].items()}
ncs = nc[aid]; cts = control[aid]; prms = prm[aid]


def parts(flag, y=lab, mask=ncs):
    v = ~flag[mask]; g = ~y[mask]
    tp = (v & g).sum(); fp = (v & ~g).sum(); tn = (~v & ~g).sum(); fn = (~v & g).sum()
    f1 = 2 * tp / (2 * tp + fp + fn); f1e = 2 * tn / (2 * tn + fn + fp)
    return dict(prm=(f1 + f1e) / 2, f1c=f1, f1e=f1e, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), rate=float((~v).mean()))


def prmscore(flag, y=lab):
    return parts(flag, y)['prm']


def ctrl_share(flag):
    c = np.bincount(aid, weights=flag, minlength=n); return float((c[control] > 0).mean())


# --- within-answer rank of S (stable, earlier step first on ties), and count-based flags
rank = np.empty(S_, int)
for i in range(n):
    a, b = off[i], off[i + 1]; o = np.argsort(-S[a:b], kind='stable'); rank[a + o] = np.arange(b - a)
def by_count(cnt): return rank < np.asarray(cnt)[aid]
cnt = {r: np.bincount(aid, weights=F[r], minlength=n).astype(int) for r in F}

rep = {}
# sanity: R2 == top-by-S with R1 counts; R0 == top-by-S with R0 counts?
rep['R2_equals_topS_R1counts'] = bool((by_count(cnt['R1_global']) == F['R2_allocate']).all())
rep['R0_equals_topS_R0counts_steps_diff'] = int((by_count(cnt['R0_frozen']) != F['R0_frozen']).sum())
same = cnt['R2_allocate'] == cnt['R0_frozen']
rep['answers_with_equal_counts_R0_R2'] = int(same.sum())
rep['R2_nested_in_or_contains_R0_violations'] = int(sum(1 for i in range(n) if not (
    (F['R2_allocate'][off[i]:off[i+1]] <= F['R0_frozen'][off[i]:off[i+1]]).all() or (F['R2_allocate'][off[i]:off[i+1]] >= F['R0_frozen'][off[i]:off[i+1]]).all())))

# --- R0 recomputation from S and thresholds (label-free check); also q grid
def zt(s): return (s - s.mean()) / max(s.std(), 1e-8)
zS = np.empty(S_)
for i in range(n): zS[off[i]:off[i + 1]] = zt(S[off[i]:off[i + 1]])
def r0_at(q=None, taus=None):
    f = np.zeros(S_, bool); t_used = {}
    for k in range(5):
        c = (k + 1) % 5; calm = prms & (fold[aid] == c); evm = fold[aid] == k
        t = taus[k] if taus is not None else float(np.quantile(zS[calm], q)); t_used[k] = t; f[evm] = zS[evm] >= t
    return f, t_used
f80, t80 = r0_at(0.8)
rep['tauS_frozen_vs_recomputed_q80_maxabs'] = max(abs(t80[k] - tauS[k]) for k in range(5))
rep['R0_rebuilt_from_frozen_tau_steps_diff'] = int((r0_at(taus=tauS)[0] != F['R0_frozen']).sum())
rep['R0_rebuilt_q80_steps_diff'] = int((f80 != F['R0_frozen']).sum())

P = {r: parts(F[r]) for r in F}
rep['prmscore'] = {r: P[r]['prm'] for r in F}
rep['R0_vs_protocol_value'] = P['R0_frozen']['prm'] - 0.6565188557935739
rep['delta_R1_R0'] = P['R1_global']['prm'] - P['R0_frozen']['prm']; rep['delta_R2_R0'] = P['R2_allocate']['prm'] - P['R0_frozen']['prm']
rep['parts'] = {r: P[r] for r in ['R0_frozen', 'R1_global', 'R2_allocate']}
rep['controls_share_flag'] = {r: ctrl_share(F[r]) for r in F}
rep['flag_rate_prm_all_steps'] = {r: float(F[r][prms].mean()) for r in F}
rep['flag_rate_controls_steps'] = {r: float(F[r][cts].mean()) for r in F}
print(json.dumps(rep, indent=1, default=float))

# ===================================================================== Task 1: label nulls (50 perms each, own seeds)
rng = np.random.default_rng(777001); ncA = np.flatnonzero(nc)
key = pd.DataFrame({'i': ncA, 'len': ns[ncA], 'fold': fold[ncA]})
cells_B = [g.to_numpy() for _, g in key.groupby(['len', 'fold']).i]
single = np.concatenate([c for c in cells_B if len(c) < 2]) if any(len(c) < 2 for c in cells_B) else np.array([], int)
obs = {r: P[r]['prm'] - P['R0_frozen']['prm'] for r in ['R1_global', 'R2_allocate']}
NA = {r: [] for r in obs}; NB = {r: [] for r in obs}
for p in range(50):
    o = np.lexsort((rng.random(S_), aid)); yA = lab[o]
    assert (aid[o] == aid).all()
    yB = lab.copy()
    for idx in cells_B:
        if len(idx) < 2: continue
        # uniform random derangement-free cyclic reassignment (each answer receives another's labels)
        perm = rng.permutation(len(idx)); src = idx[np.roll(perm, 1)]; dst = idx[perm]
        for d_, s_ in zip(dst, src): yB[off[d_]:off[d_ + 1]] = lab[off[s_]:off[s_ + 1]]
    bA = prmscore(F['R0_frozen'], yA); bB = prmscore(F['R0_frozen'], yB)
    for r in obs:
        NA[r].append(prmscore(F[r], yA) - bA); NB[r].append(prmscore(F[r], yB) - bB)
T1 = {}
for r in obs:
    a = np.array(NA[r]); b = np.array(NB[r])
    T1[r] = dict(observed=obs[r], within_mean=a.mean(), within_sd=a.std(ddof=1), resid_within=obs[r] - a.mean(), z_within=(obs[r] - a.mean()) / a.std(ddof=1),
                 swap_mean=b.mean(), swap_sd=b.std(ddof=1), resid_swap=obs[r] - b.mean(), z_swap=(obs[r] - b.mean()) / b.std(ddof=1),
                 within_minus_swap=a.mean() - b.mean(), frac_swap_ge_obs=float((b >= obs[r]).mean()), frac_within_ge_obs=float((a >= obs[r]).mean()))
T1['unswapped_singleton_answers'] = int(len(single)); T1['unswapped_steps'] = int(ns[single].sum()) if len(single) else 0
print('TASK1', json.dumps(T1, indent=1, default=float))

# ===================================================================== Task 2: length-only allocation and shuffled answer level
c2 = cnt['R2_allocate']; tot_prm = int(c2[prm].sum()); tot_nc = int(c2[nc].sum())
def prop_counts(r, mask_ans):
    c = np.floor(r * ns + 0.5).astype(int); c[~mask_ans] = 0; return np.minimum(c, ns)
def solve_r(target, mask_ans, measure_mask):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if prop_counts(mid, mask_ans)[measure_mask].sum() < target: lo = mid
        else: hi = mid
    best = min([lo, hi], key=lambda r: abs(prop_counts(r, mask_ans)[measure_mask].sum() - target)); return best
T2 = {}
# (a1) proportional to length, r matched to R2's total flags over ALL PRMBench answers (label-free footing)
r1 = solve_r(tot_prm, prm, prm); ca1 = prop_counts(r1, prm)
# (a2) proportional to length, r matched to R2's flags over the NON-CONTROL answers (rate-matched in the scored population)
r2 = solve_r(tot_nc, prm, nc); ca2 = prop_counts(r2, prm)
# (a3) mean R2 count among PRMBench answers of the same length in the OTHER folds (all PRMB, controls included: label-free)
# (a4) same, but mean over NON-CONTROL answers of other folds (uses control identity; rate approx matched)
def cellmean_counts(pool_mask):
    c = np.zeros(n, int); miss = 0
    for i in np.flatnonzero(prm):
        m = pool_mask & (fold != fold[i]) & (ns == ns[i])
        if m.any(): c[i] = int(np.floor(c2[m].mean() + 0.5))
        else:
            miss += 1; c[i] = int(np.floor(c2[pool_mask & (fold != fold[i])].sum() / ns[pool_mask & (fold != fold[i])].sum() * ns[i] + 0.5))
    return np.minimum(c, ns), miss
ca3, miss3 = cellmean_counts(prm); ca4, miss4 = cellmean_counts(nc)
ALT = {'a1_prop_len_match_prm_total': ca1, 'a2_prop_len_match_nc_total': ca2, 'a3_len_cellmean_otherfolds_allprm': ca3, 'a4_len_cellmean_otherfolds_nconly': ca4}
for k_, c in ALT.items():
    f = by_count(c); p = parts(f)
    T2[k_] = dict(prm=p['prm'], minus_R2=p['prm'] - P['R2_allocate']['prm'], minus_R0=p['prm'] - P['R0_frozen']['prm'], f1c=p['f1c'], f1e=p['f1e'],
                  nc_rate=p['rate'], nc_flags=int(c[nc].sum()), prm_flags=int(c[prm].sum()), controls_share_flag=ctrl_share(f), controls_step_rate=float(f[cts].mean()))
T2['r_a1'] = r1; T2['r_a2'] = r2; T2['a3_fallback'] = miss3; T2['a4_fallback'] = miss4
T2['R2_flags_prm'] = tot_prm; T2['R2_flags_nc'] = tot_nc
# (b) shuffle R2 counts among answers within fold x length cell: b1 among ALL PRMBench answers; b2 among NON-CONTROL answers only
def shuffle_counts(pool_mask, rng):
    c = c2.copy(); k_ = pd.DataFrame({'i': np.flatnonzero(pool_mask), 'len': ns[pool_mask], 'fold': fold[pool_mask]})
    for _, g in k_.groupby(['len', 'fold']).i:
        idx = g.to_numpy(); c[idx] = c2[idx][rng.permutation(len(idx))]
    return c
rngb = np.random.default_rng(777002)
for name, pool in [('b1_shuffle_allprm', prm), ('b2_shuffle_nconly', nc)]:
    vals, cs, nr, f1cs, f1es = [], [], [], [], []
    for d in range(50):
        f = by_count(shuffle_counts(pool, rngb)); p = parts(f)
        vals.append(p['prm']); cs.append(ctrl_share(f)); nr.append(p['rate']); f1cs.append(p['f1c']); f1es.append(p['f1e'])
    vals = np.array(vals)
    T2[name] = dict(prm_mean=vals.mean(), prm_sd=vals.std(ddof=1), prm_max=vals.max(), R2_minus_mean=P['R2_allocate']['prm'] - vals.mean(),
                    z_R2=(P['R2_allocate']['prm'] - vals.mean()) / vals.std(ddof=1), frac_ge_R2=float((vals >= P['R2_allocate']['prm']).mean()),
                    minus_R0=vals.mean() - P['R0_frozen']['prm'], f1c_mean=np.mean(f1cs), f1e_mean=np.mean(f1es), nc_rate_mean=np.mean(nr), controls_share_flag_mean=np.mean(cs))
print('TASK2', json.dumps(T2, indent=1, default=float))

# ===================================================================== Task 3: component math and rate-vs-allocation
T3 = {}
for r in ['R1_global', 'R2_allocate']:
    add = F[r] & ~F['R0_frozen'] & ncs; rem = ~F[r] & F['R0_frozen'] & ncs
    T3[r] = dict(added_steps=int(add.sum()), added_error=int((add & lab).sum()), added_err_rate=float(lab[add].mean()),
                 removed_steps=int(rem.sum()), removed_error=int((rem & lab).sum()), removed_err_rate=float(lab[rem].mean()),
                 dTP=P[r]['TP'] - P['R0_frozen']['TP'], dFP=P[r]['FP'] - P['R0_frozen']['FP'], dTN=P[r]['TN'] - P['R0_frozen']['TN'], dFN=P[r]['FN'] - P['R0_frozen']['FN'],
                 dF1c=P[r]['f1c'] - P['R0_frozen']['f1c'], dF1e=P[r]['f1e'] - P['R0_frozen']['f1e'], d_rate=P[r]['rate'] - P['R0_frozen']['rate'])
# break-even error probability of one marginal flagged step, at R0's confusion
p0 = P['R0_frozen']; TP, FP, TN, FN = p0['TP'], p0['FP'], p0['TN'], p0['FN']; Dc = 2 * TP + FP + FN; De = 2 * TN + FN + FP
# dF1c(flag step with error prob p) = [2TP - 2(1-p)Dc]/Dc^2 ; dF1e = [2p De - 2TN]/De^2
pstar = (2 * TN / De**2 + 2 * Dc / Dc**2 - 2 * TP / Dc**2) / (2 * De / De**2 + 2 * Dc / Dc**2)
T3['breakeven'] = dict(p_star_prmscore=pstar, p_star_f1e_only=TN / De, p_star_f1c_only=1 - TP / Dc, base_error_prevalence_nc=float(lab[ncs].mean()))
# marginal error rate by within-answer rank among non-control steps just below / above R0 cut
# R0 at quantiles q (fold-calibrated identically) -> noncontrol flag rate and PRMScore
curve = []
for q in np.round(np.arange(0.70, 0.851, 0.005), 3):
    f, _ = r0_at(q); p = parts(f); curve.append(dict(q=float(q), nc_rate=p['rate'], prm=p['prm'], f1c=p['f1c'], f1e=p['f1e'], ctrl=ctrl_share(f)))
T3['R0_q_curve'] = curve
# q matched to R2's non-control flag rate
target = P['R2_allocate']['rate']; lo, hi = 0.6, 0.9
for _ in range(50):
    mid = (lo + hi) / 2; f, _ = r0_at(mid)
    if parts(f)['rate'] > target: lo = mid
    else: hi = mid
qm = (lo + hi) / 2; fm, tm = r0_at(qm); pm = parts(fm)
T3['R0_rate_matched'] = dict(q=qm, taus=tm, nc_rate=pm['rate'], prm=pm['prm'], minus_R0=pm['prm'] - P['R0_frozen']['prm'], R2_minus_it=P['R2_allocate']['prm'] - pm['prm'],
                             R1_minus_it=P['R1_global']['prm'] - pm['prm'], share_of_R2_gain=(pm['prm'] - P['R0_frozen']['prm']) / (P['R2_allocate']['prm'] - P['R0_frozen']['prm']),
                             f1c=pm['f1c'], f1e=pm['f1e'], controls_share_flag=ctrl_share(fm))
# oracle-free rate-matched version: same global PRMBench flag count as R2 (label-free): q chosen so the PRMB (all) flag rate matches
target2 = float(F['R2_allocate'][prms].mean()); lo, hi = 0.6, 0.9
for _ in range(50):
    mid = (lo + hi) / 2; f, _ = r0_at(mid)
    if f[prms].mean() > target2: lo = mid
    else: hi = mid
q2 = (lo + hi) / 2; f2, _ = r0_at(q2); p2 = parts(f2)
T3['R0_prmtotal_matched'] = dict(q=q2, prm_rate=float(f2[prms].mean()), nc_rate=p2['rate'], prm=p2['prm'], minus_R0=p2['prm'] - P['R0_frozen']['prm'])
# count-level decomposition: R2 flags restricted to non-controls vs a count vector that equals R0 per answer but scaled: proportional-to-R0 allocation at R2's nc total
c0 = cnt['R0_frozen']
def scale_counts(base, target_total, mask):
    lo, hi = 0.5, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2; c = np.minimum(np.floor(base * mid + 0.5).astype(int), ns)
        if c[mask].sum() < target_total: lo = mid
        else: hi = mid
    return np.minimum(np.floor(base * hi + 0.5).astype(int), ns), hi
cs0, sf = scale_counts(c0, tot_nc, nc); fs0 = by_count(cs0); ps0 = parts(fs0)
T3['R0_counts_scaled_to_R2_nc_total'] = dict(scale=sf, nc_rate=ps0['rate'], prm=ps0['prm'], minus_R0=ps0['prm'] - P['R0_frozen']['prm'], R2_minus_it=P['R2_allocate']['prm'] - ps0['prm'])
# correlation of R2-R0 count change with error share (answer level, non-control)
dc = (c2 - c0)[nc]; es = (np.bincount(aid, weights=lab, minlength=n) / ns)[nc]; lens = ns[nc]
T3['corr_countchange_errshare'] = float(np.corrcoef(dc, es)[0, 1]); T3['corr_countchange_len'] = float(np.corrcoef(dc, lens)[0, 1]); T3['corr_errshare_len'] = float(np.corrcoef(es, lens)[0, 1])
# partial: residualize count change and error share on length dummies
dfp = pd.DataFrame({'dc': dc, 'es': es, 'L': lens})
dfp['dc_r'] = dfp.dc - dfp.groupby('L').dc.transform('mean'); dfp['es_r'] = dfp.es - dfp.groupby('L').es.transform('mean')
T3['corr_countchange_errshare_within_length'] = float(np.corrcoef(dfp.dc_r, dfp.es_r)[0, 1])
print('TASK3', json.dumps(T3, indent=1, default=float))
json.dump(dict(rep=rep, T1=T1, T2=T2, T3=T3), open(OUT / 'rt_c_results.json', 'w'), indent=1, default=float)
