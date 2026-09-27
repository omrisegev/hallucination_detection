"""Red-team agent C: null + math check for the PRMScore decomposition (independent path; reads only raw artifacts)."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:/Users/omris/TAU/hallucination_detection')
W = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
OUTD = Path(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/redteam_C')
FZ = W / 'results/expectation_realization_v1/run_20260927_stage_b'
TH = W / 'results/expectation_realization_v1/run_20260927_stage_b_thr'
t0 = time.time()
out = {}

ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; lab = Z['labels'].astype(bool)
n = len(ans); ns = np.diff(off); S = int(off[-1]); st_ans = np.repeat(np.arange(n), ns)
pos_in = np.arange(S) - np.repeat(off[:-1], ns)
prm = ~ans.cell.astype(str).str.startswith('pb_').to_numpy(); fold = ans.fold.to_numpy(); ids = ans.id.to_numpy()
freeze = json.loads((R / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
meta = {m['idx']: m for m in pickle.load(open(freeze['prm_metadata']['path'], 'rb')).values()}
cls = np.array([meta[ids[i]]['classification'] if prm[i] else '' for i in range(n)])
control = prm & (cls == 'correct'); nonc = prm & ~control
# independent label check vs official error_steps (1-indexed)
lab_off = np.zeros(S, bool)
for i in np.flatnonzero(prm):
    es = [e - 1 for e in meta[ids[i]]['error_steps'] if 1 <= e <= ns[i]]
    lab_off[off[i] + np.array(es, int)] = True
assert np.array_equal(lab_off[np.repeat(prm, ns)], lab[np.repeat(prm, ns)]), 'labels differ from official error_steps'
out['population'] = {'prm': int(prm.sum()), 'control': int(control.sum()), 'noncontrol': int(nonc.sum()),
                     'nc_steps': int(ns[nonc].sum()), 'nc_error_steps': int(lab[np.repeat(nonc, ns)].sum())}

fz = np.load(FZ / 'STEP_SCORES.npz'); assert np.array_equal(fz['offsets'], off)
ch = np.load(TH / 'CHANNELS.npz'); assert np.array_equal(ch['offsets'], off)
names = [str(x) for x in ch['names']]
thr = json.loads((TH / 'THRESHOLDS.json').read_text(encoding='utf8'))
reward = np.full(S, np.nan)
for i in np.flatnonzero(prm): reward[off[i]:off[i + 1]] = np.asarray(meta[ids[i]]['rewards'], float)
score = {'S_equal': fz['S_equal'].astype(float), 'B13_equal': fz['B13_equal'].astype(float),
         'realized_drv': ch['values'][:, names.index('realized_drv')].astype(float), 'PRM_z_q80': 1 - reward,
         'step_index': pos_in.astype(float)}
prm_step = np.repeat(prm, ns); nc_step = np.repeat(nonc, ns); sfold = np.repeat(fold, ns)

def answer_z(s):
    """Vectorized per-answer z (population std, floor 1e-8) on PRMBench answers; NaN elsewhere."""
    s = np.where(prm_step, s, 0.0)
    cnt = ns.astype(float); mu = np.bincount(st_ans, s, n) / np.maximum(cnt, 1)
    d = s - mu[st_ans]; var = np.bincount(st_ans, d * d, n) / np.maximum(cnt, 1)
    sd = np.maximum(np.sqrt(var), 1e-8)
    z = d / sd[st_ans]; z[~prm_step] = np.nan; return z

def answer_z_loop(s):
    z = np.full(S, np.nan)
    for i in np.flatnonzero(prm):
        x = s[off[i]:off[i + 1]]; z[off[i]:off[i + 1]] = (x - x.mean()) / max(x.std(), 1e-8)
    return z

def q80(z): return {k: float(np.quantile(z[prm_step & (sfold == (k + 1) % 5)], .8)) for k in range(5)}
def valid_of(z, tau):
    tv = np.array([tau[k] for k in range(5)])[sfold]; return prm_step & (z < tv)

scale, tau, valid = {}, {}, {}
for m, s in score.items():
    scale[m] = answer_z(s)
# vectorized z vs the script's loop form (exactness of the decision scale)
zdev = {m: float(np.nanmax(np.abs(scale[m] - answer_z_loop(score[m])))) for m in ('S_equal', 'realized_drv')}
out['answer_z_vectorized_vs_loop_maxabs'] = zdev
for m in score:
    if m in ('S_equal', 'B13_equal'):
        tau[m] = {int(k): float(v) for k, v in thr[m].items()}
        out.setdefault('saved_tau_vs_q80_rule_on_OOF', {})[m] = {k: tau[m][k] - q80(scale[m])[k] for k in range(5)}
    else:
        tau[m] = q80(scale[m])
    valid[m] = valid_of(scale[m], tau[m])
# when my z differs from loop z by float rounding, flags could flip: count flips
flips = {m: int((valid_of(answer_z_loop(score[m]), tau[m]) != valid[m]).sum()) for m in ('S_equal', 'B13_equal', 'realized_drv')}
out['flag_flips_vectorized_vs_loop'] = flips
out['tau'] = tau

def conf(v, y, mask=nc_step):
    v = v[mask]; g = ~y[mask]
    return np.array([(v & g).sum(), (v & ~g).sum(), (~v & ~g).sum(), (~v & g).sum()], float)
def prmscore(c):
    tp, fp, tn, fn = c
    p, r = tp / (tp + fp), tp / (tp + fn); f1 = 2 * p * r / (p + r)
    p2, r2 = tn / (tn + fn), tn / (tn + fp); f1e = 2 * p2 * r2 / (p2 + r2)
    return 0.5 * (f1 + f1e), f1, f1e

ARMS = ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'step_index']
obs = {m: prmscore(conf(valid[m], lab)) for m in ARMS}
out['observed'] = {m: {'prmscore': obs[m][0], 'f1_correct': obs[m][1], 'f1_error': obs[m][2],
                       'flag_rate_nc': float((~valid[m][nc_step]).mean())} for m in ARMS}
# official scorer cross-check
sys.path.insert(0, str(MAIN / '.worktrees/depth-feature-fusion-v1'))
from spectral_utils.prmbench import prmbench_evaluate
P = np.flatnonzero(prm)
for m in ARMS:
    res = prmbench_evaluate([{'idx': ids[i], 'labels': valid[m][off[i]:off[i + 1]].astype(int).tolist()} for i in P], [meta[ids[i]] for i in P])
    out['observed'][m]['official_prmscore'] = 0.5 * (res['total']['f1'] + res['total']['negative_f1'])
DIFFS = [('S_equal', 'B13_equal'), ('S_equal', 'realized_drv'), ('S_equal', 'PRM_z_q80')]
obs_d = {f'{a}-{b}': obs[a][0] - obs[b][0] for a, b in DIFFS}
out['observed_diffs'] = obs_d

# quick paired source-group bootstrap of the observed diffs (independent of the script)
grp = ans.source_group.to_numpy(); gu, gi = np.unique(grp[nonc], return_inverse=True); G = len(gu)
NC = np.flatnonzero(nonc)
def per_answer_conf(v, y):
    gstep = ~y; a = st_ans
    out_ = np.zeros((n, 4))
    for c, msk in enumerate([(v & gstep), (v & ~gstep), (~v & ~gstep), (~v & gstep)]):
        out_[:, c] = np.bincount(a[msk & nc_step], minlength=n)
    return out_[NC]
agc = {m: np.zeros((G, 4)) for m in ARMS}
for m in ARMS: np.add.at(agc[m], gi, per_answer_conf(valid[m], lab))
rng = np.random.default_rng(7); Wb = rng.multinomial(G, np.full(G, 1 / G), size=4000).astype(float)
def ps_vec(C):
    tp, fp, tn, fn = C.T; p, r = tp / (tp + fp), tp / (tp + fn); p2, r2 = tn / (tn + fn), tn / (tn + fp)
    return 0.5 * (2 * p * r / (p + r) + 2 * p2 * r2 / (p2 + r2))
bs = {m: ps_vec(Wb @ agc[m]) for m in ARMS}
out['boot_check'] = {f'{a}-{b}': [float(np.quantile(bs[a] - bs[b], .025)), float(np.quantile(bs[a] - bs[b], .975)), float(np.std(bs[a] - bs[b]))] for a, b in DIFFS}

# ---------------- Null A: within-answer label permutation (20 seeds)
nullA = {k: [] for k in obs_d}; nullA_lvl = {m: [] for m in ARMS}
for seed in range(20):
    r_ = np.random.default_rng(1000 + seed); key = r_.random(S)
    order = np.lexsort((key, st_ans)); yp = lab[order]
    assert np.array_equal(np.bincount(st_ans, yp, n), np.bincount(st_ans, lab, n))
    sc = {m: prmscore(conf(valid[m], yp))[0] for m in ARMS}
    for m in ARMS: nullA_lvl[m].append(sc[m])
    for a, b in DIFFS: nullA[f'{a}-{b}'].append(sc[a] - sc[b])

# ---------------- Null B: whole-answer label swap within (fold, n_steps), non-control answers only; cyclic derangement
def swap_labels(seed, by_fold=True):
    r_ = np.random.default_rng(2000 + seed); yp = lab.copy(); fixed = 0
    key = pd.DataFrame({'i': NC, 'f': fold[NC] if by_fold else 0, 'n': ns[NC]})
    for _, g in key.groupby(['f', 'n']):
        mem = g.i.to_numpy()
        if len(mem) < 2: fixed += len(mem); continue
        perm = r_.permutation(mem); src = np.roll(perm, -1)
        for dst, sr in zip(perm, src): yp[off[dst]:off[dst + 1]] = lab[off[sr]:off[sr + 1]]
    return yp, fixed
nullB = {k: [] for k in obs_d}; nullB_lvl = {m: [] for m in ARMS}; fixedB = None
for seed in range(20):
    yp, fixedB = swap_labels(seed)
    assert yp[nc_step].sum() == lab[nc_step].sum()
    sc = {m: prmscore(conf(valid[m], yp))[0] for m in ARMS}
    for m in ARMS: nullB_lvl[m].append(sc[m])
    for a, b in DIFFS: nullB[f'{a}-{b}'].append(sc[a] - sc[b])
out['nullB_singleton_answers_keep_own_labels'] = fixedB

def summ(obsv, arr):
    arr = np.asarray(arr); mu, sd = float(arr.mean()), float(arr.std(ddof=1))
    return {'observed': obsv, 'null_mean': mu, 'null_sd': sd, 'z': (obsv - mu) / sd if sd > 0 else None,
            'null_min': float(arr.min()), 'null_max': float(arr.max())}
out['nullA_diffs'] = {k: summ(obs_d[k], v) for k, v in nullA.items()}
out['nullB_diffs'] = {k: summ(obs_d[k], v) for k, v in nullB.items()}
out['nullA_levels'] = {m: summ(obs[m][0], v) for m, v in nullA_lvl.items()}
out['nullB_levels'] = {m: summ(obs[m][0], v) for m, v in nullB_lvl.items()}

# ---------------- Task 2: realized_drv permuted within answer (label-free), own q80 rule
def within_auc_mean(s, y):
    vals = []
    for i in np.flatnonzero(nonc):
        a, b = off[i:i + 2]; yy = y[a:b]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 and n0: vals.append((rankdata(s[a:b])[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(vals))
def hit_rate(s, y):
    h = []
    for i in np.flatnonzero(nonc):
        a, b = off[i:i + 2]
        if y[a:b].any(): v = s[a:b]; h.append(bool(y[a + int(np.flatnonzero(v >= v.max() - 8 * np.finfo(float).eps)[0])]))
    return float(np.mean(h))
rd = score['realized_drv']; perm_res = []
for seed in range(20):
    r_ = np.random.default_rng(3000 + seed); order = np.lexsort((r_.random(S), st_ans)); sp = rd[order]
    zp = answer_z(sp); tp_ = q80(zp); vp = valid_of(zp, tp_)
    flags_same = bool(np.array_equal(np.bincount(st_ans[~vp & prm_step], minlength=n), np.bincount(st_ans[~valid['realized_drv'] & prm_step], minlength=n)))
    rec = {'seed': seed, 'prmscore': prmscore(conf(vp, lab))[0], 'tau_maxabs_change': max(abs(tp_[k] - tau['realized_drv'][k]) for k in range(5)),
           'per_answer_flag_counts_identical': flags_same}
    if seed < 3: rec['within_auc'] = within_auc_mean(sp, lab); rec['any_error_hit'] = hit_rate(sp, lab)
    perm_res.append(rec)
out['permuted_feature'] = {'observed_realized_drv': {'prmscore': obs['realized_drv'][0], 'within_auc': within_auc_mean(rd, lab), 'any_error_hit': hit_rate(rd, lab)},
                           'perm_prmscore_mean': float(np.mean([r['prmscore'] for r in perm_res])), 'perm_prmscore_sd': float(np.std([r['prmscore'] for r in perm_res], ddof=1)),
                           'perm_within_auc_first3': [r['within_auc'] for r in perm_res[:3]], 'perm_hit_first3': [r['any_error_hit'] for r in perm_res[:3]],
                           'tau_maxabs_change_max': max(r['tau_maxabs_change'] for r in perm_res),
                           'per_answer_flag_counts_identical_all': all(r['per_answer_flag_counts_identical'] for r in perm_res)}
# reference baselines: all-valid, step_index
allv = prm_step.copy(); out['baselines'] = {'all_steps_valid_prmscore_parts': [float(x) if np.isfinite(x) else None for x in prmscore(conf(allv, lab))],
                                            'step_index_prmscore': obs['step_index'][0], 'step_index_within_auc': within_auc_mean(score['step_index'], lab)}
# random flags at 20% per answer-independent rate (no answer structure)
r_ = np.random.default_rng(9); rv = prm_step & (r_.random(S) >= 0.2); out['baselines']['iid_random_20pct_flags_prmscore'] = prmscore(conf(rv, lab))[0]

json.dump(out, open(OUTD / 'rt_c_part1.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)); print('elapsed', time.time() - t0)
