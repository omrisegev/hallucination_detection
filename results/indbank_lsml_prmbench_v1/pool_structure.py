"""Label-free independence screen of every materialized step channel against the level family."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
A = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; y = Z['labels'].astype(bool); n = len(A); ns = np.diff(off)
prm = (~A.cell.str.startswith('pb_')).to_numpy(); prm_steps = np.repeat(prm, ns)
U = np.load('union_top10_profiles.npy')                      # 11 bank + 7 ct7 (Top10, raw)
H = np.load('hist29_top10_profiles.npy'); HN = json.load(open('hist29_names.json'))
lv = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz'); names11 = list(lv['channels'].astype(str))
prof = np.load(R / 'profiles_full.npy', mmap_mode='r'); shape = np.ascontiguousarray(prof[:, 0, [14, 11, 12, 10]]).astype(float)
ED = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/EVIDENCE_DROP.npz'); tk = np.load(MAIN / '.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz'); toff = tk['token_offsets']; spans = tk['step_spans']; rk = ED['risk_token'].astype(float)
ev = np.empty(int(off[-1]))
for i in range(n):
    r = rk[toff[i]:toff[i+1]]
    for s in range(off[i], off[i+1]):
        a, b = spans[s]; v = r[a:b]; k = min(10, len(v)); ev[s] = np.partition(v, len(v)-k)[-k:].mean()
X = np.column_stack([lv['level'].astype(float), U[:, 11:], shape, ev, H])
names = names11 + ['ct7_' + c for c in ['H0lim', 've0', 've0.75', 've1', 'H0lim_prefix_innovation', 'bocpd_residual', 'chosen_std_excess']] + ['H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z', 'evidence_drop_risk'] + ['hist_' + h for h in HN]
# fill NaN with answer column mean, answer-z
for i in range(n):
    a, b = off[i:i+2]; blk = X[a:b]
    if np.isnan(blk).any():
        mu = np.nanmean(blk, 0); mu = np.where(np.isfinite(mu), mu, 0.); idx = np.where(np.isnan(blk)); blk[idx] = mu[idx[1]]
    mu = blk.mean(0); sd = blk.std(0); X[a:b] = np.divide(blk - mu, sd, out=np.zeros_like(blk), where=sd > 1e-12)
Xp = X[prm_steps]; yp = y[prm_steps]
level = Xp[:, [names.index(c) for c in ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level']]].mean(1)
Xc = Xp.copy()
for c in (False, True): Xc[yp == c] -= Xc[yp == c].mean(0)
lc = level.copy(); lc[~yp] -= lc[~yp].mean(); lc[yp] -= lc[yp].mean()
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
elig = np.array([prm[i] and y[off[i]:off[i+1]].any() and (~y[off[i]:off[i+1]]).any() for i in range(n)])
def wauc(j):
    v = []
    for i in np.flatnonzero(elig):
        a, b = off[i:i+2]; yy = y[a:b]; s = X[a:b, j]; n1 = yy.sum(); n0 = len(yy) - n1
        v.append((rankdata(s)[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(v))
rows = []
B11 = Xp[:, :11]
for j, nm in enumerate(names):
    m = corr(Xp[:, j], level); mc = corr(Xc[:, j], lc); a = wauc(j)
    dup = max(abs(corr(Xp[:, j], B11[:, q])) for q in range(11)) if j >= 11 else np.nan
    rows.append({'channel': nm, 'in_bank11': j < 11, 'r_level_marginal': m, 'r_level_conditional': mc, 'max_r_with_bank11': dup, 'sd_after_z': float(Xp[:, j].std()), 'within_auc_single': a, 'auc_oriented': max(a, 1 - a)})
T = pd.DataFrame(rows); T.to_csv('pool_structure.csv', index=False)
pd.set_option('display.width', 220); print(T.round(3).to_string(index=False))
np.save('pool_z.npy', X); json.dump(names, open('pool_names.json', 'w'))
