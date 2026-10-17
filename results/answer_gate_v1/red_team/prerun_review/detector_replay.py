"""Label-free replay of answer_gate_run.py's detector block (lines 148-188) to inspect zA extremes.
Reads only ANSWER_FEATURES.npz (X, cells, folds). No labels, no metrics."""
import sys
import numpy as np
ROOT = r'C:/Users/omris/TAU/hallucination_detection/.worktrees/decision-rule-v1'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/scripts/experiments')
import answer_gate_run as agr  # noqa: E402  (module import only; main() not executed)
from spectral_utils.streaming_utils import anchor_orient  # noqa: E402

F = np.load(ROOT + '/results/answer_gate_v1/ANSWER_FEATURES.npz'); X = F['X'].astype(float); names = [str(x) for x in F['names']]
cells = F['cells']; fold = F['folds']; n = len(X)
DET = agr.DETECTORS; zA = {d: {k: np.full(n, np.nan) for k in range(5)} for d in DET}
e_ix = names.index('epr'); l_ix = names.index('trace_length'); g5 = [names.index(f) for f in agr.GOOD_5]
sig5 = np.array([agr.FEATURE_SIGNS[f] for f in agr.GOOD_5], float)
wlog = []
for cell in sorted(set(cells)):
    cm = cells == cell
    for k in range(5):
        c = (k + 1) % 5; fa = cm & ~np.isin(fold, [k, c]); use = cm & np.isin(fold, [k, c])
        med = np.nanmedian(X[fa], 0); Xall = np.where(np.isfinite(X), X, med)
        mu = Xall[fa].mean(0); sd = Xall[fa].std(0); keep = sd > 1e-12
        Z = np.zeros_like(Xall); Z[:, keep] = (Xall[:, keep] - mu[keep]) / sd[keep]
        anchor_fit = Z[fa, e_ix]; vals = {}
        Ff = Z[fa][:, keep].T
        probe = agr.upcr_fit(Ff, **agr.FIT); pol = np.sign(probe.rho_hat_full); pol[pol == 0] = 1.0
        res = agr.upcr_fit(Ff * pol[:, None], **agr.FIT); w = np.zeros(len(names)); w[np.flatnonzero(keep)] = res.w * pol
        vals['D1_upcr_full'] = w
        W5, fused5, m5 = agr.lsml_linear(Z[fa][:, g5] * sig5); w2 = np.zeros(len(names)); w2[g5] = W5 * sig5; vals['D2_lsml_cont_good5'] = w2
        kk = np.flatnonzero(keep); Wf, fusedf, mf = agr.lsml_linear(Z[fa][:, kk] * pol); w3 = np.zeros(len(names)); w3[kk] = Wf * pol
        vals['D3_lsml_full'] = w3
        w4 = np.zeros(len(names)); w4[kk] = pol / len(kk); vals['D4_equal_full'] = w4
        w5 = np.zeros(len(names)); w5[e_ix] = 1.0; vals['D5_epr'] = w5
        w6 = np.zeros(len(names)); w6[l_ix] = 1.0; vals['D6_length'] = w6
        ms_ix = names.index('min_spilled')
        wlog.append((cell, k, bool(keep[ms_ix]), {d: float(vals[d][ms_ix]) for d in ('D1_upcr_full', 'D3_lsml_full', 'D4_equal_full')},
                     bool(res.used_simple_average), bool(res.g2_at_ceiling), bool(m5['degenerate']), bool(mf['degenerate'])))
        for d, wd in vals.items():
            sf = Z[fa] @ wd; _, flipped = anchor_orient(sf, anchor_fit); sgn = -1.0 if flipped else 1.0
            sf = sgn * sf; m_, s_ = sf.mean(), max(sf.std(), 1e-12)
            zA[d][k][use] = (sgn * (Z[use] @ wd) - m_) / s_

print('cell fold keep_min_spilled  w_min_spilled(D1,D3,D4)  upcr_simple_avg  g2_ceiling  lsml5_degenerate  lsml27_degenerate')
for r in wlog:
    print(r)
print('\n|zA| extremes per detector (over eval+cal answers, all 5 fold models):')
for d in DET:
    allz = np.concatenate([zA[d][k][np.isfinite(zA[d][k])] for k in range(5)])
    print(f'{d:22s} max|zA| {np.abs(allz).max():9.2f}  n>6 {int((np.abs(allz) > 6).sum()):4d}  n>10 {int((np.abs(allz) > 10).sum()):4d}  n>50 {int((np.abs(allz) > 50).sum()):3d}  of {len(allz)}')
np.savez_compressed(r'C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/41041b9e-b636-4238-bb1a-111fd1f803fe/scratchpad/review_answer_gate/zA_replay.npz',
                    **{f'{d}_{k}': zA[d][k] for d in DET for k in range(5)})
