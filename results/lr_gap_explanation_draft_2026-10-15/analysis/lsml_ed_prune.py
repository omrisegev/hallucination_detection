"""Omri's proposal (2026-10-17): prune redundant channels inside L-SML clusters, then L-SML again. Label-free.

Per outer fold, training fold only:
1. Canonical L-SML (spectral_utils.advisor_lr.fit_labelfree 'LSML_CONT') on all channels -> clusters.
2. Inside each cluster, greedy: effective dimension ED = participation ratio of the members' correlation
   matrix, (sum lambda)^2 / sum lambda^2. Try removing each member; take the removal with the highest ED;
   accept it only if ED does not decrease. Repeat until no removal is accepted or one member is left.
3. Canonical L-SML on all kept channels (it re-clusters and weights within and between clusters).
Held-out within-answer AUC vs the saved LSML_FULL out-of-fold row.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path('/Users/osegev/Desktop/hallucination_detection')
sys.path.insert(0, str(ROOT))
from spectral_utils.advisor_lr import fit_labelfree

BASE = ROOT / 'results/advisor_completion_2026-10-11'
OUT = Path(__file__).resolve().parent

def within(score, off, y):
    a = []
    for b, e in zip(off[:-1], off[1:]):
        ss, yy = score[b:e], y[b:e]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 and n0: a.append((rankdata(ss)[yy.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(a))

def ed(C):
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())

def prune_cluster(C, members):
    keep = list(members)
    while len(keep) > 1:
        sub = lambda ks: C[np.ix_(ks, ks)]
        cur = ed(sub(keep))
        trials = [(ed(sub([k for k in keep if k != r])), r) for r in keep]
        best, r = max(trials)
        if best < cur: break
        keep.remove(r)
    return keep

p = np.load(BASE / 'inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
saved = {(r['bank'], r['arm']): float(r['within_answer_auc']) for r in csv.DictReader((BASE / 'lr/METRICS.csv').open())}
res = {}
for bank in sys.argv[1:] or ['N56']:
    z = np.load(BASE / f'inputs/bank_V_{bank}.npz'); X = z['V'].astype(float); names = z['columns'].astype(str).tolist()
    score = np.full(len(y), np.nan); folds = []
    for o in range(5):
        tr, te = sf != o, sf == o
        groups = np.asarray(fit_labelfree(X[tr], 'LSML_CONT').diagnostic['group'])
        C = np.corrcoef(X[tr].T)
        kept = sorted(k for g in np.unique(groups) for k in prune_cluster(C, np.flatnonzero(groups == g).tolist()))
        model = fit_labelfree(X[tr][:, kept], 'LSML_CONT')
        score[te] = model.predict(X[te][:, kept])
        tem = np.repeat(p['fold'].astype(int) == o, np.diff(off))
        o_off = np.r_[0, np.cumsum(np.diff(off)[p['fold'].astype(int) == o])]
        full = fit_labelfree(X[tr], 'LSML_CONT').predict(X[te])
        fa, fb = within(score[te], o_off, y[te]), within(full, o_off, y[te])
        print(f'  {bank} fold {o}: pruned {fa:.4f}  full refit {fb:.4f}  K {model.diagnostic["K"]}', flush=True)
        folds.append({'auc_pruned': fa, 'auc_full_refit': fb, 'K_before': int(len(np.unique(groups))), 'K_after': int(model.diagnostic['K']),
                      'kept': len(kept), 'removed': [names[j] for j in range(len(names)) if j not in kept]})
    res[bank] = {'channels': len(names), 'kept_per_fold': [f['kept'] for f in folds],
                 'LSML_ED_PRUNE': round(within(score, off, y), 4),
                 'LSML_FULL_saved': round(saved[(bank, 'LSML_FULL')], 4),
                 'LR_FULL_saved': round(saved[(bank, 'LR_FULL')], 4), 'folds': folds}
    print(bank, {k: v for k, v in res[bank].items() if k != 'folds'}, flush=True)
    print('  removed in fold 0:', folds[0]['removed'], flush=True)
(OUT / f'lsml_ed_prune_{"_".join(res)}.json').write_text(json.dumps(res, indent=1))
