"""Omri's follow-up (2026-11-19): one 'error direction' per injected error type (synthetic answer - own parent, mean of
the 17 z-scored answer features), each direction -> one answer score (projection), then fuse the scores with SML.
Checks the independence assumption (conditional correlation of the per-type scores within wrong and within clean).
Dev only, CPU; directions fitted on one gatefold, scored on the other."""
import json, sys
from pathlib import Path
import numpy as np

CODE, W, GOLD = sys.argv[1:4]; W = Path(W)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L, ssl_stages as S, run_analysis as RA

split = json.load(open(W / 'inputs' / 'SPLIT.json')); labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
gold = {g['uid']: g for g in json.load(open(GOLD))}
one = np.ones(16); OUT = {}


def sml_rank1(Zs):
    """classic SML: weights = leading eigenvector of the covariance of the standardized scores (sign: positive sum)."""
    Cv = np.cov(Zs.T); w, v = np.linalg.eigh(Cv); v1 = v[:, -1]; v1 = v1 if v1.sum() >= 0 else -v1
    return Zs @ v1, v1


for bench in ('processbench', 'prmbench'):
    C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev'); rawC = D.readout(Cd)
    ev0, s0, f0, X0 = S.step_eval(Cd, rawC)
    AM = S.AnswerModel(Cd, rawC, s0); FC = RA.feats(AM, Cd, rawC, one)[0]; AM.fit_fusion(FC); sC = AM.score(FC)
    mu, sd = FC.mean(0), FC.std(0); sd[sd == 0] = 1; Z = (FC - mu) / sd
    y = Cd.y_ans; f1 = RA.halves(Cd.group, 'gatefold')
    types = [V for V in ('V1', 'V2', 'V3', 'V4', 'V5') if (W / 'steps' / f'STEPS_{bench}_{V}.npz').exists()]
    scores, dirs = {}, {}
    for V in types:
        P = D.load_pool(W, bench, V, split); Pd, nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
        ZV = (RA.feats(AM, Pd, D.readout(Pd), one)[0] - mu) / sd; dZ = ZV - Z[RA.parent_index(Pd, Cd)]
        fpos = np.isin(Pd.group, Cd.group[f1]); sc = np.zeros(Cd.n); dd = []
        for fold in (True, False):
            dm = dZ[fpos == fold].mean(0); dm /= np.linalg.norm(dm); dd.append(dm); sc[f1 != fold] = Z[f1 != fold] @ dm
        scores[V], dirs[V] = sc, (dd[0] + dd[1]) / 2
    K = len(types); Sm = np.column_stack([scores[V] for V in types]); Zs = (Sm - Sm.mean(0)) / Sm.std(0)
    cos = {f'{a}-{b}': round(float(dirs[a] @ dirs[b] / np.linalg.norm(dirs[a]) / np.linalg.norm(dirs[b])), 3)
           for i, a in enumerate(types) for b in types[i + 1:]}
    def offd(M): return M[np.triu_indices(K, 1)]
    cw = np.corrcoef(Zs[y].T); cc = np.corrcoef(Zs[~y].T); ca = np.corrcoef(Zs.T)
    s_sml, w_sml = sml_rank1(Zs); s_ls, W_ls, meta = L.lsml_linear(Zs)
    res = dict(types=types, auroc_each={V: round(L.auroc(y, scores[V]), 4) for V in types},
               direction_cosine=cos,
               corr_all_mean=round(float(offd(ca).mean()), 3),
               cond_corr_wrong_mean=round(float(offd(cw).mean()), 3), cond_corr_clean_mean=round(float(offd(cc).mean()), 3),
               cond_corr_wrong=np.round(cw, 3).tolist(), cond_corr_clean=np.round(cc, 3).tolist(),
               sml_weights=dict(zip(types, np.round(w_sml, 3).tolist())), sml_auroc=round(L.auroc(y, s_sml), 4),
               lsml_auroc=round(L.auroc(y, s_ls), 4), lsml_groups=meta['groups'],
               equal_mean_auroc=round(L.auroc(y, Zs.mean(1)), 4),
               reference_answer_lsml_17feat=round(L.auroc(y, sC), 4))
    OUT[bench] = res
    print(bench, json.dumps({k: v for k, v in res.items() if k not in ('cond_corr_wrong', 'cond_corr_clean')}), flush=True)
Path(sys.argv[4]).write_text(json.dumps(OUT, indent=1))
