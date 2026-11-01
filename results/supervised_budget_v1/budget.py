"""Stage 2 of the supervised/calibrated plan, as frozen in PROTOCOL.json: label-budget curve of in-domain supervised
weighting (LR17, ANCH) against label-free AUTO and the PRM, plus the stage-1 Socratic re-check with problem groups.

    python budget.py      # -> BUDGET.csv, SUMMARY.json, RECHECK.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'supervised_anchor_v1'))
import anchor as S1  # noqa: E402  (Cell, fit, parity gate, stage-8 helpers)
K, S8, E = S1.K, S1.S8, S1.E

PROT = json.loads((HERE / 'PROTOCOL.json').read_text())
BUDGETS = [5, 10, 20, 50, 100, 200, 500, 1000, 'all']
REPS, SEED = 20, 20261101
INPUTS = {'LR17': 'Z16', 'ANCH': 'T16'}


def problem_groups(cell):
    if cell.bench == 'socratic':
        return np.unique([u.split(':')[-1] for u in cell.c.ids], return_inverse=True)[1]
    return np.unique(np.asarray(cell.c.groups), return_inverse=True)[1]


def lr(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)).fit(X, y)


def point(cell, s, ans_mask):
    return S8.point(cell.c, S8.stats(cell.c, s, ans_mask))


def run_cell(cell, prm, rows):
    c = cell.c; ga = problem_groups(cell); ng = ga.max() + 1
    labelable = c.keep[c.off[:-1]].astype(bool) if cell.bench == 'prmbench' else np.ones(len(c.ids), bool)
    step_ans = cell.ans_of_step
    for rep in range(REPS):
        rng = np.random.default_rng(SEED + rep)
        test_g = rng.permutation(ng)[: ng // 2]
        test = np.isin(ga, test_g); pool = np.flatnonzero(~test & labelable)
        order = rng.permutation(pool)
        base = {a: point(cell, s, test) for a, s in (('PRM', cell.zp), ('AUTO_N16', cell.auto))}
        for k in BUDGETS:
            n = len(order) if k == 'all' else k
            if n > len(order):
                continue
            ans = order[:n]; tr = np.isin(step_ans, ans) & cell.train
            r = dict(bench=cell.bench, prm=prm, rep=rep, k=k, n_answers=n, n_steps=int(tr.sum()),
                     **{f'{a}_{m}': base[a][m] for a in base for m in ('gap_prmscore', 'within_auc')})
            if len(np.unique(cell.y[tr])) < 2:
                r['fail'] = 'one class'; rows.append(r); continue
            for a, key in INPUTS.items():
                p = point(cell, lr(cell.X[key][tr], cell.y[tr]).decision_function(cell.X[key]), test)
                r.update({f'{a}_gap_prmscore': p['gap_prmscore'], f'{a}_within_auc': p['within_auc']})
            rows.append(r)
        print(cell.bench, prm, 'rep', rep, 'all-pool LR17 - AUTO',
              round(rows[-1].get('LR17_gap_prmscore', np.nan) - rows[-1]['AUTO_N16_gap_prmscore'], 2), flush=True)


def recheck(cell, prm):
    """Stage-1 Socratic in-domain cross-fit, with problem groups instead of per-answer groups."""
    g = problem_groups(cell)[cell.ans_of_step]; out = {}
    _, ans_eval = K.eval_rows(cell.c, cell.bench)
    for a, key in INPUTS.items():
        X, y = cell.X[key], cell.y; oof = np.zeros(len(y))
        for itr, ite in GroupKFold(5).split(X, y, g):
            m, _ = S1.fit(X[itr], y[itr], g[itr]); oof[ite] = m.decision_function(X[ite])
        out[a] = point(cell, oof, ans_eval)
    out['PRM'] = point(cell, cell.zp, ans_eval); out['AUTO_N16'] = point(cell, cell.auto, ans_eval)
    print('recheck', prm, {a: round(v['gap_prmscore'], 2) for a, v in out.items()}, flush=True)
    return out


def summarize(rows):
    S = []
    for bench in PROT['benchmarks']:
        for prm in PROT['prms']:
            for k in BUDGETS:
                R = [r for r in rows if r['bench'] == bench and r['prm'] == prm and r['k'] == k and 'fail' not in r]
                if not R:
                    continue
                d = dict(bench=bench, prm=prm, k=k, reps=len(R), fails=sum(1 for r in rows if r['bench'] == bench and r['prm'] == prm and r['k'] == k and 'fail' in r))
                for a in INPUTS:
                    for b in ('AUTO_N16', 'PRM'):
                        for m in ('gap_prmscore', 'within_auc'):
                            x = np.array([r[f'{a}_{m}'] - r[f'{b}_{m}'] for r in R])
                            d[f'{a}-{b}:{m}'] = dict(mean=float(x.mean()), p025=float(np.percentile(x, 2.5)),
                                                     p975=float(np.percentile(x, 97.5)), pos=int((x > 0).sum()))
                S.append(d)
    kstar = {}
    for bench in PROT['benchmarks']:
        for prm in PROT['prms']:
            for a in INPUTS:
                for b in ('AUTO_N16', 'PRM'):
                    D = [d for d in S if d['bench'] == bench and d['prm'] == prm]
                    ok = [x[f'{a}-{b}:gap_prmscore']['mean'] > 0 and x[f'{a}-{b}:gap_prmscore']['pos'] >= 16 for x in D]
                    ks = None
                    for i in range(len(D)):
                        if all(ok[i:]):
                            ks = D[i]['k']; break
                    kstar[f'{bench}|{prm}|{a} vs {b}'] = ks
    return S, kstar


def main():
    rows, rc = [], {}
    for prm in PROT['prms']:
        for bench in PROT['benchmarks']:
            cell = S1.Cell(bench, prm)
            _, ans_eval = K.eval_rows(cell.c, bench); p = point(cell, cell.auto, ans_eval)
            S1.gate(f'{prm} {bench} AUTO gap_prmscore', p['gap_prmscore'], S1.S8_REF[(bench, prm)]['gap_prmscore'])
            run_cell(cell, prm, rows)
            if bench == 'socratic':
                rc[prm] = recheck(cell, prm)
            del cell
    with open(HERE / 'BUDGET.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    S, kstar = summarize(rows)
    (HERE / 'SUMMARY.json').write_text(json.dumps(dict(kstar=kstar, summary=S, parity=S1.PAR), indent=1))
    (HERE / 'RECHECK.json').write_text(json.dumps(rc, indent=1))
    print(json.dumps(kstar, indent=1))


if __name__ == '__main__':
    t = time.time(); main(); print(f'{time.time() - t:.0f}s')
