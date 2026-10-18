"""SLIDER (arXiv 2610.00571) step-level redundancy detection on PRMBench's redundancy class.

What the paper states (Sec. 5 + App. C/D.2): redundancy class only, "around 750 problems"; positive class =
steps annotated as redundant (PRMBench error_steps of the redundency class), negative = the other steps; "for
each method, we calibrate the decision threshold on a held-out set by training a linear classifier and use the
resulting threshold for step-level ground-truth redundancy detection on the test set"; metric = accuracy
(Fig. 7: Step-RRI 61.91%, +13.53 over InfoGain -> 48.38%, +10.31 over Cosine -> 51.60%).
NOT stated: split sizes or seeds, whether step 1 (no preceding steps) is scored, whether accuracy is balanced,
whether the linear classifier is class-weighted, and whether problems are the redundency rows or their
numerical variants. Step-RRI is estimated on 50 GPT-4o-mini numerical variants per problem.

Because those choices are unknown, we report a small FIXED grid of readings (no choice made on our results):
  split: 10 random trace-level 50/50 held-out/test splits (seeds 0..9);
  threshold: 1-D logistic regression on the held-out steps, unweighted and class_weight='balanced';
  metric: plain accuracy and balanced accuracy on the test steps; scope: all steps, and steps >= 2.
Plus the majority-class accuracy (always 'not redundant') and our deployed label-free Q80 rule (no threshold
calibration). The redundancy rows are scored with each arm's full-population score (fitted label-free).
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import HERE, ARMS, load_population, get_scores, flags_for, auc

SEEDS = range(10)


def main():
    P = load_population(); S = get_scores(P)
    off, y, cls = P['off'], P['y'], P['cls']
    R = np.flatnonzero(cls == 'redundency')
    sl = [slice(off[j], off[j + 1]) for j in R]
    out = {'n_problems': len(R), 'n_steps': int(sum(s.stop - s.start for s in sl)),
           'n_redundant_steps': int(sum(y[s].sum() for s in sl)),
           'n_problems_with_redundant_step': int(sum(y[s].any() for s in sl)), 'arms': {}}
    out['redundant_rate'] = out['n_redundant_steps'] / out['n_steps']
    for a in ARMS:
        res = {}
        flags = flags_for(a, S[a], off)
        for scope in ('all_steps', 'steps_ge2'):
            keep = [np.arange(s.start, s.stop)[(slice(None) if scope == 'all_steps' else slice(1, None))] for s in sl]
            yy = np.concatenate([y[k] for k in keep]); ff = np.concatenate([flags[k] for k in keep])
            res[f'{scope}__label_free_Q80__acc'] = 100 * float((ff == yy).mean())
            res[f'{scope}__label_free_Q80__bal_acc'] = 50 * (ff[yy].mean() + (~ff[~yy]).mean())
            res[f'{scope}__majority_acc'] = 100 * float((~yy).mean())
            res[f'{scope}__pooled_auroc'] = auc(yy, np.concatenate([S[a][k] for k in keep]))
            for cw in ('unweighted', 'balanced'):
                acc, bal = [], []
                for sd in SEEDS:
                    perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
                    hold, test = perm[:h], perm[h:]
                    xh = np.concatenate([S[a][keep[k]] for k in hold])[:, None]; yh = np.concatenate([y[keep[k]] for k in hold])
                    xt = np.concatenate([S[a][keep[k]] for k in test])[:, None]; yt = np.concatenate([y[keep[k]] for k in test])
                    m = LogisticRegression(class_weight=None if cw == 'unweighted' else 'balanced').fit(xh, yh)
                    p = m.predict(xt).astype(bool)
                    acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
                for nm, v in (('acc', acc), ('bal_acc', bal)):
                    v = np.array(v)
                    res[f'{scope}__LR_{cw}__{nm}'] = float(v.mean())
                    res[f'{scope}__LR_{cw}__{nm}_ci95'] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        out['arms'][a] = res
    json.dump(out, open(HERE / 'SLIDER_RESULTS.json', 'w'), indent=1)
    print({k: v for k, v in out.items() if k != 'arms'})
    for a, r in out['arms'].items():
        print(a, {k: round(v, 2) for k, v in r.items() if 'ci95' not in k and k.startswith('all_steps')})


if __name__ == '__main__':
    main()
