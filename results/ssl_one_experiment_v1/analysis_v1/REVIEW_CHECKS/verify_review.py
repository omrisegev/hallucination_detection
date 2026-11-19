"""Check the main quantitative claims of the external review (2026-11-19). Dev only, CPU, read-only.
  1 pool copy vs real wrong answers through the unchanged answer-level C2ST (RA.c2st_answers); BBE pi-hat with the pool as positives
  2 answer AUROC inside length strata: L-SML answer score vs the synthetic-pair direction (V4, V3+V4), cross-fit as DIRECTION_QUICK
  3 relative position of the injected step k vs the real first error
  4 soft top-M weights at M = 1
"""
import json, sys
from pathlib import Path
import numpy as np

CODE, W, GOLD, OUT = sys.argv[1:5]; W = Path(W)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L, ssl_stages as S, run_analysis as RA, mix_v34 as MX

split = json.load(open(W / 'inputs' / 'SPLIT.json')); labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
gold = {g['uid']: g for g in json.load(open(GOLD))}
one = np.ones(16); R = {}
w = L.soft_topn_weights(1.0, 10); R['soft_top_M1_weights_first5'] = np.round(w[:5] / w.sum(), 3).tolist()

def strat_auc(y, s, q):
    out = []
    for k in range(q.max() + 1):
        m = q == k
        out.append(round(L.auroc(y[m], s[m]), 4) if 0 < y[m].sum() < m.sum() else None)
    # length-adjusted pooled AUROC: rank inside stratum (percentile), then pooled
    r = np.zeros(len(s))
    for k in range(q.max() + 1):
        m = q == k; r[m] = (np.argsort(np.argsort(s[m])) + .5) / m.sum()
    return out, round(L.auroc(y, r), 4)

for bench in ('processbench', 'prmbench'):
    C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev'); rawC = D.readout(Cd)
    ev0, s0, f0, X0 = S.step_eval(Cd, rawC)
    AM = S.AnswerModel(Cd, rawC, s0); FC = RA.feats(AM, Cd, rawC, one)[0]; AM.fit_fusion(FC); sC = AM.score(FC)
    y = Cd.y_ans; nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8]); q = np.searchsorted(edges, nC)
    f1 = RA.halves(Cd.group, 'gatefold'); B = R[bench] = {}
    # 1 pool copy
    B['pool_copy_answer_c2st'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in
                                  RA.c2st_answers(FC, y, Cd.group, nC, FC, Cd.group, nC, edges).items()}
    B['bbe_pool_as_positives'] = round(L.bbe(sC[f1], sC[~f1])[0], 4)
    B['answer_score_auroc_by_n_steps_quintile'], B['answer_score_auroc_length_adjusted'] = strat_auc(y, sC, q)
    B['answer_score_auroc'] = round(L.auroc(y, sC), 4)
    # 2 pair direction inside strata
    mu, sd = FC.mean(0), FC.std(0); sd[sd == 0] = 1; Z = (FC - mu) / sd
    pv = {}
    for V in ('V3', 'V4'):
        P = D.load_pool(W, bench, V, split); pv[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
    pv['V3+V4'] = MX.concat([pv['V3'], pv['V4']], 'V3+V4')
    for V in ('V4', 'V3+V4'):
        Pd = pv[V]; ZV = (RA.feats(AM, Pd, D.readout(Pd), one)[0] - mu) / sd; dZ = ZV - Z[RA.parent_index(Pd, Cd)]
        fpos = np.isin(Pd.group, Cd.group[f1]); sc = np.zeros(Cd.n)
        for fold in (True, False):
            dm = dZ[fpos == fold].mean(0); dm /= np.linalg.norm(dm); sc[f1 != fold] = Z[f1 != fold] @ dm
        st, adj = strat_auc(y, sc, q)
        B[f'direction_{V}'] = dict(auroc=round(L.auroc(y, sc), 4), by_n_steps_quintile=st, length_adjusted=adj,
                                   corr_with_n_steps=round(float(np.corrcoef(sc, np.log(nC))[0, 1]), 3))
    B['lsml_corr_with_n_steps'] = round(float(np.corrcoef(sC, np.log(nC))[0, 1]), 3)
    # 3 relative position of k vs first real error
    first_rel = []
    for j, (a, b) in enumerate(zip(Cd.off[:-1], Cd.off[1:])):
        e = np.where(Cd.wrong_step[a:b])[0]
        if len(e): first_rel.append(e[0] / max(b - a - 1, 1))
    V3 = pv['V3']; krel = V3.k / np.maximum(np.diff(V3.off) - 1, 1)
    B['rel_position_median'] = dict(real_first_error=round(float(np.median(first_rel)), 3), V3_k=round(float(np.median(krel)), 3),
                                    V3_k_is_last_step_share=round(float(np.mean(V3.k == np.diff(V3.off) - 1)), 3))
    print(bench, json.dumps(B), flush=True)
R['note'] = 'dev only; gold labels used only for these checks'
Path(OUT).write_text(json.dumps(R, indent=1)); print(json.dumps({k: v for k, v in R.items() if k not in ('processbench', 'prmbench')}))
