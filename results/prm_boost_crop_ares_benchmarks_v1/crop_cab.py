"""CROP (arXiv 2605.30085) Table 6 protocol on CROP's own four extra datasets, for our label-free arms (local).

Datasets: crv_arith, crv_gsm8k, math_shepherd, prm800k, built by make_inputs.py with CROP's own selection rules
(counts match CROP Table 4 exactly). Protocol is copied from results/prmbench_paper_metric_parity_v1/crop_metrics.py
(split_traces, crop): 10 seeds 2806..2815, trace-level 60/20/20 stratified by has-error, alpha = 0.05, the largest
lambda whose calibration risk bound (1 + #contaminated) / (n_cal + 1) <= alpha, prefix = steps before the first
score > lambda. Metrics on the test split: pooled step AUROC, mean kept prefix (M/T) and prefix risk.

Grids: CROP uses 201 train-split quantiles on its CRV sets and a fixed {0, .01, ..., 1} grid on its external sets.
We run every arm on the q201 grid (label-free), and Direct PRM (1 - reward) also on the fixed grid as a bridge.
Our arms need no train split (fitted label-free per answer / per benchmark, transductive); CROP's threshold uses the
calibration labels for every arm, which is CROP's access, not ours.

    python crop_cab.py   -> eval/CROP_TABLE6_CAB.csv (mean and 95% normal interval over the 10 splits)
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'prmbench_paper_metric_parity_v1'))
sys.path.insert(0, str(HERE))
import evaluate_cab as E  # noqa: E402

SEEDS = list(range(2806, 2816)); ALPHA = .05
BENCHES = ['crv_arith', 'crv_gsm8k', 'math_shepherd', 'prm800k']
ARMS = {'prm': ['PRMPROB', 'TELE_N16', 'MEAN_N16', 'CHAIN_N16', 'CHAIN_N13', 'HALF_N16'], 'sa': ['TELE_N16', 'MEAN_N16']}


def split_traces(has_err, seed):                     # verbatim from crop_metrics.py
    rng = np.random.default_rng(seed)
    tr, ca, te = [], [], []
    for key in (False, True):
        idx = np.flatnonzero(has_err == key).tolist(); rng.shuffle(idx)
        n = len(idx); n_tr = int(round(.6 * n)); n_ca = int(round(.2 * n))
        tr += idx[:n_tr]; ca += idx[n_tr:n_tr + n_ca]; te += idx[n_tr + n_ca:]
    for part in (tr, ca, te): rng.shuffle(part)
    return np.array(tr), np.array(ca), np.array(te)


def prefix_len(s, lam):
    hit = np.flatnonzero(s > lam)
    return int(hit[0]) if len(hit) else len(s)


def crop(scores, labels, cal, test, grid):           # verbatim from crop_metrics.py (kept_stepweighted dropped)
    def first_err(y): f = np.flatnonzero(y); return f[0] if len(f) else None
    losses = np.zeros((len(grid), len(cal)))
    for k, i in enumerate(cal):
        fe = first_err(labels[i])
        if fe is None: continue
        losses[:, k] = grid >= np.max(scores[i][:fe + 1])
    risk = (1 + losses.sum(1)) / (len(cal) + 1)
    ok = np.flatnonzero(risk <= ALPHA)
    lam = grid[ok[np.argmax(grid[ok])]] if len(ok) else grid.min()
    M = np.array([prefix_len(scores[i], lam) for i in test]); T = np.array([len(scores[i]) for i in test])
    cont = np.array([bool(m > 0 and labels[i][:m].any()) for m, i in zip(M, test)])
    return dict(kept=100 * float(np.mean(M / np.maximum(T, 1))), risk=100 * float(cont.mean()), feasible=bool(len(ok)))


def main():
    rows = []
    for bench in BENCHES:
        for mode in ('prm', 'sa'):
            for prm in E.PRMS:
                c = E.load(mode, prm, bench)
                labels = [c.err[a:b] for a, b in zip(c.off[:-1], c.off[1:])]
                has_err = np.array([y.any() for y in labels])
                for arm in ARMS[mode]:
                    s = c.arms[arm]; per = [s[a:b] for a, b in zip(c.off[:-1], c.off[1:])]
                    grids = ['q201'] + (['fixed101'] if arm == 'PRMPROB' else [])
                    for gname in grids:
                        res = []
                        for seed in SEEDS:
                            tr, ca, te = split_traces(has_err, seed)
                            if gname == 'fixed101':
                                grid = np.round(np.linspace(0, 1, 101), 2)
                            else:
                                grid = np.unique(np.quantile(np.concatenate([per[i] for i in tr]), np.linspace(0, 1, 201)))
                            r = crop(per, labels, ca, te, grid)
                            yt = np.concatenate([labels[i] for i in te]); st = np.concatenate([per[i] for i in te])
                            r['auroc'] = float(roc_auc_score(yt, st)); res.append(r)
                        out = dict(benchmark=bench, mode=mode, prm=prm, arm=arm, grid=gname, n_traces=len(per),
                                   access='CROP calibration labels on top of: ' + E.EC.access_of(arm))
                        for k in ('auroc', 'kept', 'risk'):
                            v = np.array([r[k] for r in res])
                            out[k] = float(v.mean()); out[k + '_ci'] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
                        out['feasible_splits'] = int(sum(r['feasible'] for r in res))
                        rows.append(out)
                print(bench, mode, prm, flush=True)
    E.OUT.mkdir(exist_ok=True)
    with open(E.OUT / 'CROP_TABLE6_CAB.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('rows', len(rows))


if __name__ == '__main__':
    main()
