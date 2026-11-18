"""Dev-only check: can a synthetic variant set the answer-gate threshold? pi-hat by BBE with that variant as the known-wrong set.
CPU, seconds. Prepared 2026-11-18 for the decision card after REALISM_V34.md (option "Check pi-hat with V3 on PB dev").

The same computation as stage G of run_analysis.py on dev, with M = 1 (stage A did not run): answer features of the dev C
pool (16 oriented ECDF maxima + Simes), answer L-SML score; 2-fold cross-fit by question group ('gatefold' halves);
in each fold BBE(scores of the variant answers of the other fold, scores of the C answers of the other fold); pi-hat = mean.
Also: the PU posterior P(wrong) per dev answer and its AUROC, and the share of answers the gate would open.
Gold labels: only for the true pi and the AUROC (reports). This is a change of the pre-registered plan (V3 is not named there).

    python pihat_dev.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --bench processbench --variant V3 --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np

import run_analysis as RA
import ssl_data as D
import ssl_lib as L
import ssl_stages as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bench', default='processbench'); ap.add_argument('--variant', default='V3')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W = Path(a.W); split = json.load(open(W / 'inputs' / 'SPLIT.json')); gold = {g['uid']: g for g in json.load(open(a.gold))}
    lp = W / 'inputs' / 'prmbench_C_labels.json'; labels = json.load(open(lp)) if lp.exists() else None
    C = D.load_pool(W, a.bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
    P = D.load_pool(W, a.bench, a.variant, split); Pd, nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
    rawC = D.readout(Cd); ev, sF, *_ = S.step_eval(Cd, rawC)
    AM = S.AnswerModel(Cd, rawC, sF); one = np.ones(16)
    FC = RA.feats(AM, Cd, rawC, one)[0]; FP = RA.feats(AM, Pd, D.readout(Pd), one)[0]
    AM.fit_fusion(FC); sC = AM.score(FC); sP = AM.score(FP)
    f1 = RA.halves(Cd.group, 'gatefold'); fpos = np.isin(Pd.group, Cd.group[f1])
    folds, P_dev = [], np.zeros(Cd.n)
    for fold in (True, False):
        o_pos, o_C = fpos != fold, f1 != fold
        pi_f, info = L.bbe(sP[o_pos], sC[o_C]); folds.append(dict(pi_hat=pi_f, n_pos=int(o_pos.sum()), n_C=int(o_C.sum()), **info))
        P_dev[o_C] = L.pu_posterior(sP[o_pos], sC[o_C], pi_f, sC[o_C])
    pi_hat = float(np.mean([f['pi_hat'] for f in folds])); pi_true = float(Cd.y_ans.mean())
    open_ = RA.gate_open(sC, pi_hat)
    R = dict(bench=a.bench, variant=a.variant, side='dev only', n_C=Cd.n, n_variant=Pd.n, dropped=nd,
             pi_hat=pi_hat, pi_true=pi_true, abs_err=abs(pi_hat - pi_true), pi_check_pass_10pts=bool(abs(pi_hat - pi_true) <= .10),
             folds=folds, answer_auroc_wrong_vs_clean=L.auroc(Cd.y_ans, sC), variant_vs_C_score_auroc=L.auroc(
                 np.r_[np.ones(Pd.n), np.zeros(Cd.n)].astype(bool), np.r_[sP, sC]),
             P_mean=float(P_dev.mean()), P_auroc=L.auroc(Cd.y_ans, P_dev),
             gate_open_share=float(open_.mean()), gate_open_true_wrong_share=float(Cd.y_ans[open_].mean()) if open_.any() else None,
             gate_closed_true_clean_share=float((~Cd.y_ans[~open_]).mean()) if (~open_).any() else None,
             seconds=round(time.time() - t0, 1))
    json.dump(RA.rnd(R), open(out / f'PIHAT_{a.bench}_{a.variant}.json', 'w'), indent=1)
    print(json.dumps(RA.rnd(R), indent=1))


if __name__ == '__main__':
    main()
