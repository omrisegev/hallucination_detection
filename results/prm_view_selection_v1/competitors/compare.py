"""Exploratory: the best stage-1 subset (q15_VE1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0),
fused by the stage-1 method, under the competitor protocols already implemented in
results/prm_boost_collection_v1/collection_competitors.py (PRISM rates under GAP, ARES CV threshold macro-F1, SLIDER
redundancy class). PRM and CHAIN_N16 recomputed by the same code for parity."""
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stage1')); sys.argv = sys.argv[:1]
import stage1 as S
sys.path.insert(0, str(HERE.parents[1] / 'prm_boost_collection_v1'))
import collection_competitors as CC
from sklearn.linear_model import LogisticRegression
E = S.E
NAMES = ['q15_H1','q15_VE1','chosen_surprisal','logprob_margin','true_tail50','energy_level','energy_innovation','top15_turnover',
         'top50_js','dominant_freq16','bocpd_p0','realized_z','realized_drv','digit_alternative','digit_spread','digit_alternative_innovation']
SUB = [NAMES.index(x) for x in ['q15_VE1','chosen_surprisal','top15_turnover','dominant_freq16','bocpd_p0']]
rows = []
for prm in S.PRMS:
    cell = S.CellData('prmbench', prm); c = cell.c
    o = cell.evaluate(SUB, want_detail=True); c.arms['SUB5'] = o['fused']
    ref = CC.best_scale(c)
    ug = np.unique(c.groups[c.cls != 'correct']); rng = np.random.default_rng(20261021); fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
    step_fold = np.repeat([fold_of.get(g, -1) for g in c.groups], np.diff(c.off))
    R = np.flatnonzero(c.cls == 'redundency'); sl = [slice(c.off[j], c.off[j + 1]) for j in R]
    for label, arm, rule in [('PRM native', 'PRM_native', 'native'), ('PRM same rule (GAP)', ref, 'GAP'), ('CHAIN_N16 (GAP)', 'CHAIN_N16', 'GAP'), ('SUB5 (GAP)', 'SUB5', 'GAP')]:
        m = E.measure(c, arm, rule, fast=True)[0]
        r = dict(prm=prm, row=label, PRMScore=m['official'], TPR=m['TPR'], FPR=m['FPR'], within_auc=m['within_auc'])
        # ARES: global threshold by 5-fold CV on labels (same code path as COMP_ARES)
        v = CC.thr_score(c, arm if arm != ref else 'PRM_native'); oof = np.zeros(len(v), bool)
        for k in range(5):
            trm = c.keep & (step_fold != k); tem = c.keep & (step_fold == k)
            ts = np.quantile(v[trm], np.linspace(0.01, 0.99, 99))
            best = max(ts, key=lambda t: E.official_from_counts(CC._counts(c, v >= t, trm), 'prmbench'))
            oof[tem] = v[tem] >= best
        r['ARES_macroF1'] = float(E.official_from_counts(CC._counts(c, oof, c.keep), 'prmbench'))
        # SLIDER: redundancy class, plain accuracy with a class-balanced LR threshold (their metric) and balanced accuracy
        s = c.arms[arm]; keep = [np.arange(x.start, x.stop) for x in sl]; acc, bal = [], []
        for sd in range(10):
            perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
            xh = np.concatenate([s[keep[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([c.err[keep[k]] for k in perm[:h]])
            xt = np.concatenate([s[keep[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([c.err[keep[k]] for k in perm[h:]])
            p = LogisticRegression(class_weight='balanced').fit(xh, yh).predict(xt).astype(bool)
            acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
        r['SLIDER_acc'], r['SLIDER_bal_acc'] = float(np.mean(acc)), float(np.mean(bal))
        rows.append(r); print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
E.write_csv(HERE / 'COMPARE.csv', rows)
