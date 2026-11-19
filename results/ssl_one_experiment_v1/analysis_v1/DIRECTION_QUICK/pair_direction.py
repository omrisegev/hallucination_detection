"""Omri's idea (2026-11-19): the paired difference (synthetic answer - its own parent) in the 17 answer features as an
'error direction'. Dev only, CPU. Directions are fitted on one gatefold of question groups and scored on the other.
Compared with the existing label-free answer score (L-SML) and a supervised LR (labels, cross-fit) as a ceiling."""
import json, sys
from pathlib import Path
import numpy as np

CODE, W, GOLD = sys.argv[1:4]; W = Path(W)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L, ssl_stages as S, run_analysis as RA, mix_v34 as MX
from sklearn.linear_model import LogisticRegression

split = json.load(open(W / 'inputs' / 'SPLIT.json')); labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
gold = {g['uid']: g for g in json.load(open(GOLD))}
one = np.ones(16); OUT = {}
for bench in ('processbench', 'prmbench'):
    C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev'); rawC = D.readout(Cd)
    ev0, s0, f0, X0 = S.step_eval(Cd, rawC)
    AM = S.AnswerModel(Cd, rawC, s0); FC = RA.feats(AM, Cd, rawC, one)[0]; AM.fit_fusion(FC); sC = AM.score(FC)
    mu, sd = FC.mean(0), FC.std(0); sd[sd == 0] = 1; Z = (FC - mu) / sd
    y = Cd.y_ans; f1 = RA.halves(Cd.group, 'gatefold'); pt = float(y.mean())
    res = dict(lsml_auroc=round(L.auroc(y, sC), 4))
    # supervised ceiling: LR on Z, cross-fit by gatefold
    pLR = np.zeros(Cd.n)
    for fold in (True, False):
        m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Z[f1 == fold], y[f1 == fold]); pLR[f1 != fold] = m.decision_function(Z[f1 != fold])
    res['supervised_LR_auroc'] = round(L.auroc(y, pLR), 4)
    pv = {}
    for V in ('V3', 'V4'):
        P = D.load_pool(W, bench, V, split); pv[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
    pv['V3+V4'] = MX.concat([pv['V3'], pv['V4']], 'V3+V4')
    for V, Pd in pv.items():
        ZV = (RA.feats(AM, Pd, D.readout(Pd), one)[0] - mu) / sd; pj = RA.parent_index(Pd, Cd); dZ = ZV - Z[pj]
        fpos = np.isin(Pd.group, Cd.group[f1])
        out = {}
        for name in ('mean_diff', 'whitened_mean_diff'):
            sc = np.zeros(Cd.n); dirs = []
            for fold in (True, False):
                dm = dZ[fpos == fold].mean(0)
                if name == 'whitened_mean_diff':
                    Sg = np.cov(Z[f1 == fold].T) + 1e-3 * np.eye(Z.shape[1]); dm = np.linalg.solve(Sg, dm)
                dirs.append(dm / np.linalg.norm(dm)); sc[f1 != fold] = Z[f1 != fold] @ dirs[-1]
            # threshold from the same pairs: BBE with the synthetic answers projected on the direction (cross-fit)
            sP = np.zeros(Pd.n); pis = []
            for k, fold in enumerate((True, False)):
                sP_o = ZV[fpos != fold] @ dirs[k]; pi_f, _ = L.bbe(sP_o, sc[f1 != fold]); pis.append(pi_f)
            ph = float(np.mean(pis)); op = RA.gate_open(sc, ph)
            out[name] = dict(auroc=round(L.auroc(y, sc), 4), cos_between_folds=round(float(dirs[0] @ dirs[1]), 3),
                             cos_with_LR=round(float(np.corrcoef(sc, pLR)[0, 1]), 3), pi_hat=round(ph, 3), pi_true=round(pt, 3),
                             wrong_opened=round(float(op[y].mean()), 3), clean_closed=round(float((~op)[~y].mean()), 3))
        out['mean_shift_size_sd'] = round(float(np.linalg.norm(dZ.mean(0))), 3)
        res[V] = out
        print(bench, V, json.dumps(out), flush=True)
    print(bench, 'reference', json.dumps({k: v for k, v in res.items() if 'auroc' in k}), flush=True)
    OUT[bench] = res
Path(sys.argv[4]).write_text(json.dumps(OUT, indent=1))
