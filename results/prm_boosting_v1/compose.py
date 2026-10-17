"""PRM-dominant compositions (label-free), after Omri's 2026-10-12 request that the PRM step score not be
swallowed by the bank. Views are answer-z step scores (higher = riskier):
  PRM      = prm_risk (logit(1 - reward))
  BB       = N16 MEAN_MOM_ga fitted on the PRM-backbone telemetry bank (tag prm_backbone)
  Q3       = N16 MEAN_MOM_ga fitted on the Qwen3-8B think-fix mask1 bank (tag qwen3_8b_mask1)
Compositions:
  EQ2(PRM,BB), EQ2(PRM,Q3)   equal halves
  EQ3(PRM,BB,Q3)             equal thirds
  SML3(PRM,BB,Q3)            rank-1 weights from the three views' pooled covariance (Parisi et al. 2014 SML:
                             loading v_i^2 = C_ij C_ik / C_jk, weight w_i = v_i / (C_ii - v_i^2)); no labels
Same evaluation and bootstrap definitions as fuse.py / bootstrap.py.
"""
import pickle, sys
import numpy as np
import pandas as pd
import fuse as FU

DRAWS, SEED = 2000, 20261013


def main():
    S = {t: np.load(FU.OUT / f'SCORES_{t}.npz', allow_pickle=True) for t in ('prm_backbone', 'qwen3_8b_mask1', 'instruct_ref')
         if (FU.OUT / f'SCORES_{t}.npz').exists()}
    base = S['qwen3_8b_mask1']; o = base['offsets']; ids = base['idx'].astype(str)
    for t, z in S.items():
        if not (np.array_equal(z['offsets'], o) and np.array_equal(z['idx'].astype(str), ids)): raise SystemExit('order ' + t)
    az = lambda v: FU.ext.answer_standardize(np.asarray(v, float)[:, None], o)[:, 0]
    V = {'PRM': az(base['prm_risk']), 'Q3': az(base['N16__MEAN_MOM_ga'])}
    if 'prm_backbone' in S: V['BB'] = az(S['prm_backbone']['N16__MEAN_MOM_ga'])
    if 'instruct_ref' in S: V['INS'] = az(S['instruct_ref']['N16__MEAN_MOM_ga'])
    comps, diag = {}, {}
    for k in V: comps[k] = V[k]
    for a, b in (('PRM', 'BB'), ('PRM', 'Q3'), ('PRM', 'INS')):
        if b in V: comps[f'EQ2({a},{b})'] = .5 * (V[a] + V[b])
    for trio in (('PRM', 'BB', 'Q3'), ('PRM', 'INS', 'Q3')):
        if not all(t in V for t in trio): continue
        M = np.column_stack([V[t] for t in trio]); C = np.cov(M.T)
        comps[f'EQ3{trio}'] = M.mean(1)
        v = np.array([np.sqrt(max(C[i, j] * C[i, k] / C[j, k], 0)) for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1))])
        w = v / np.maximum(np.diag(C) - v ** 2, 1e-6); w = w / w.sum()
        comps[f'SML3{trio}'] = M @ w; diag[f'SML3{trio}'] = {'cov': C.round(4).tolist(), 'loadings': v.round(4).tolist(), 'weights': w.round(4).tolist()}
    A = pd.read_csv(FU.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig').set_index('id')
    with np.load(FU.SE / 'OOF_STEP_SCORES.npz') as q: OFF = q['offsets'].astype(np.int64); LAB = q['labels'].astype(bool)
    pos = {i: k for k, i in enumerate(A.index.astype(str))}
    lab = np.concatenate([LAB[OFF[pos[i]]:OFF[pos[i] + 1]] for i in ids])
    prm = {r['idx']: r for r in pickle.load(open(FU.PRM_PKL, 'rb')).values()}
    meta = [{'idx': i, 'classification': prm[i]['classification'], 'error_steps': prm[i]['error_steps']} for i in ids]
    nonc = np.array([m['classification'] != 'correct' for m in meta])
    err = np.zeros(o[-1], bool)
    for j, i in enumerate(ids):
        for e in prm[i]['error_steps']:
            if 1 <= e <= o[j + 1] - o[j]: err[o[j] + e - 1] = True
    groups = A.loc[ids, 'source_group'].to_numpy(); ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    W = np.vstack([np.ones(len(ug)), rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=DRAWS)])
    native = np.concatenate([np.asarray(prm[i]['labels']) == 0 for i in ids])
    rows, boots = [], {}
    for name, s in [('PRM_native0.5', V['PRM'])] + list(comps.items()):
        f = native if name == 'PRM_native0.5' else (lambda zs: zs > np.quantile(zs, .8))(az(s))
        au = np.array([FU.auc(lab[o[j]:o[j + 1]], s[o[j]:o[j + 1]]) for j in range(len(ids))])
        preds = [{'idx': i, 'labels': (~f[o[j]:o[j + 1]]).astype(int).tolist()} for j, i in enumerate(ids)]
        ev = FU.prmbench_evaluate(preds, meta); t = ev['total']; bc = ev['by_classification']
        r = {'arm': name, 'rule': 'NATIVE_0.5' if name == 'PRM_native0.5' else 'ANSWER_Z_Q80', 'within_auc': float(np.nanmean(au)),
             'prmscore': 100 * (t['f1'] + t['negative_f1']) / 2}
        for c in sorted({m['classification'] for m in meta} - {'correct'}):
            r['prmscore__' + c] = 100 * (bc['f1'][c] + bc['negative_f1'][c]) / 2
        rows.append(r)
        c4 = np.zeros((len(ids), 4))
        for j in np.flatnonzero(nonc):
            y, ff = err[o[j]:o[j + 1]], f[o[j]:o[j + 1]]
            c4[j] = [(~y & ~ff).sum(), (y & ~ff).sum(), (y & ff).sum(), (~y & ff).sum()]
        G = np.zeros((len(ug), 6)); np.add.at(G, inv, np.column_stack([np.where(np.isfinite(au), au, 0), np.isfinite(au), c4]))
        Sg = W @ G; tp, fp, tn, fn = Sg[:, 2:].T
        boots[name] = (Sg[:, 0] / Sg[:, 1], 100 * (tp / (tp + .5 * (fp + fn)) + tn / (tn + .5 * (fp + fn))) / 2)
        assert abs(boots[name][1][0] - r['prmscore']) < 1e-6, name
    df = pd.DataFrame(rows)
    ref = {'vs_PRM_native': 'PRM_native0.5', 'vs_PRM_Q80': 'PRM', 'vs_BB': 'BB', 'vs_Q3': 'Q3'}
    for col, rn in ref.items():
        if rn not in boots: continue
        for m, nm in ((0, 'auc'), (1, 'prmscore')):
            vals = []
            for a in df.arm:
                d = boots[a][m] - boots[rn][m]
                vals.append(f'{d[0]:+.4f} [{np.quantile(d[1:], .025):+.4f}, {np.quantile(d[1:], .975):+.4f}]' if a != rn else '')
            df[f'{nm}_{col}'] = vals
    df.to_csv(FU.OUT / 'COMPOSITIONS.csv', index=False)
    import json; json.dump(diag, open(FU.OUT / 'COMPOSITIONS_SML_DIAGNOSTICS.json', 'w'), indent=1)
    pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 40)
    print(df[['arm', 'within_auc', 'prmscore', 'auc_vs_PRM_native', 'prmscore_vs_PRM_native', 'prmscore_vs_PRM_Q80']].round(4).to_string())
    print(json.dumps(diag, indent=1))


if __name__ == '__main__':
    main()
