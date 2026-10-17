"""Channel atlas for PRM-dominant small banks (label-USING diagnostic, 2026-10-13, Omri's request).

For each of the 56 bank channels (+ PRM risk) on a given PRMBench FEATURES.npz:
  - within-answer AUC on the answer-standardized channel (digit channels: active-position standardization),
    and the sign that makes it >= .5;
  - conditional dependence given the true label: Spearman correlation of the oriented answer-z channel with the
    PRM risk, computed separately inside error steps and inside valid steps (non-control answers);
  - per-classification within-AUC (error-type profile).
Also the full channel x channel conditional correlation matrix (mean of the two class-conditional Spearmans).
Labels are used here: anything selected from this table is a label-selected subset.

    python channel_atlas.py <FEATURES.npz> <tag>
"""
import json, pickle, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import fuse as FU


def main():
    feat, tag = sys.argv[1], sys.argv[2]
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(FU.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    F = dict(np.load(feat)); names = [str(x) for x in F['names']]
    prm = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    ids = A.id.astype(str).to_numpy(); ans = np.flatnonzero(A.cell.to_numpy() == FU.CELL)
    risk = np.zeros(int(OFF[-1]))
    for i in ans:
        p = np.clip(1 - np.asarray(prm[ids[i]]['rewards'], float), 1e-6, 1 - 1e-6); risk[OFF[i]:OFF[i + 1]] = np.log(p / (1 - p))
    F['values'] = np.column_stack([F['values'], risk]); F['names'] = np.r_[F['names'].astype(str), ['prm_risk']]
    allc = names + ['prm_risk']
    FU.ext.BN = dict(FU.ext.BN, ALL=allc)
    pop = FU.ext.Population(F, ['ALL']); bank = pop.cell_bank('ALL', ans); V = bank['V']; o = bank['off']
    y = np.concatenate([LAB[OFF[i]:OFF[i + 1]] for i in ans])
    cls = np.repeat([prm[ids[i]]['classification'] for i in ans], np.diff(o))
    aid = np.repeat(np.arange(len(ans)), np.diff(o))
    def wauc(s, mask=None):
        out = []
        for j in range(len(ans)):
            if mask is not None and not mask[j]: continue
            yy = y[o[j]:o[j + 1]]; n1 = yy.sum(); n0 = len(yy) - n1
            if n1 and n0: out.append((rankdata(s[o[j]:o[j + 1]])[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        return float(np.mean(out))
    acls = np.array([prm[ids[i]]['classification'] for i in ans])
    rows = []; S = np.zeros_like(V)
    for k, c in enumerate(allc):
        a = wauc(V[:, k]); sg = 1. if a >= .5 else -1.; S[:, k] = sg * V[:, k]
        r = {'channel': c, 'within_auc_oriented': max(a, 1 - a), 'sign': sg}
        for cl in sorted(set(acls) - {'correct'}):
            r['auc__' + cl] = wauc(S[:, k], acls == cl)
        rows.append(r)
    nonc = cls != 'correct'
    R = {}
    for lab_v, nm in ((True, 'err'), (False, 'val')):
        m = nonc & (y == lab_v); R[nm] = np.corrcoef(np.apply_along_axis(rankdata, 0, S[m]).T)
    Cc = .5 * (R['err'] + R['val']); kp = allc.index('prm_risk')
    df = pd.DataFrame(rows); df['cond_rho_prm_err'] = R['err'][:, kp]; df['cond_rho_prm_val'] = R['val'][:, kp]
    df['cond_rho_prm'] = Cc[:, kp]
    df = df.sort_values('within_auc_oriented', ascending=False)
    df.to_csv(FU.OUT / f'CHANNEL_ATLAS_{tag}.csv', index=False)
    np.savez_compressed(FU.OUT / f'CHANNEL_CONDCORR_{tag}.npz', names=np.array(allc), cond_err=R['err'], cond_val=R['val'],
                        S=S.astype(np.float32), y=y, offsets=o, idx=ids[ans])
    pd.set_option('display.width', 250)
    print(df[['channel', 'within_auc_oriented', 'sign', 'cond_rho_prm_err', 'cond_rho_prm_val']].round(3).head(30).to_string())


if __name__ == '__main__':
    main()
