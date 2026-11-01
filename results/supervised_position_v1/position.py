"""Position control for stage 2, as frozen in PROTOCOL.json: LR on [PRM, relative position] and LR17 + position, on the
same splits and label draws as supervised_budget_v1/budget.py.

    python position.py      # -> POSITION.csv, SUMMARY.json, run.log
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'supervised_budget_v1'))
import budget as B  # noqa: E402  (stage 2, unchanged)
S1, K = B.S1, B.K

PROT = json.loads((HERE / 'PROTOCOL.json').read_text())
ARMS = ('LR17', 'POS2', 'LR18')
CONTRASTS = [('LR17', 'POS2'), ('LR18', 'LR17'), ('POS2', 'PRM'), ('POS2', 'AUTO_N16'), ('LR17', 'AUTO_N16'), ('LR17', 'PRM')]
STAGE2 = {(r['bench'], r['prm'], int(r['rep']), r['k']): float(r['LR17_gap_prmscore'])
          for r in csv.DictReader(open(HERE.parent / 'supervised_budget_v1/BUDGET.csv')) if r.get('LR17_gap_prmscore')}


def run_cell(cell, prm, rows):
    c = cell.c; ga = B.problem_groups(cell); ng = ga.max() + 1
    pos = K.relative_position(c.off)
    X = {'LR17': cell.X['Z16'], 'POS2': np.column_stack([cell.zp, pos]), 'LR18': np.column_stack([cell.X['Z16'], pos])}
    labelable = c.keep[c.off[:-1]].astype(bool) if cell.bench == 'prmbench' else np.ones(len(c.ids), bool)
    for rep in range(B.REPS):
        rng = np.random.default_rng(B.SEED + rep)                 # identical draws to budget.run_cell
        test_g = rng.permutation(ng)[: ng // 2]
        test = np.isin(ga, test_g); pool = np.flatnonzero(~test & labelable)
        order = rng.permutation(pool)
        base = {a: B.point(cell, s, test) for a, s in (('PRM', cell.zp), ('AUTO_N16', cell.auto), ('POSONLY', pos))}
        for k in B.BUDGETS:
            n = len(order) if k == 'all' else k
            if n > len(order):
                continue
            tr = np.isin(cell.ans_of_step, order[:n]) & cell.train
            r = dict(bench=cell.bench, prm=prm, rep=rep, k=k, n_answers=n,
                     **{f'{a}_{m}': base[a][m] for a in base for m in ('gap_prmscore', 'within_auc')})
            if len(np.unique(cell.y[tr])) < 2:
                r['fail'] = 'one class'; rows.append(r); continue
            for a in ARMS:
                p = B.point(cell, B.lr(X[a][tr], cell.y[tr]).decision_function(X[a]), test)
                r.update({f'{a}_gap_prmscore': p['gap_prmscore'], f'{a}_within_auc': p['within_auc']})
            S1.gate(f'{cell.bench} {prm} rep{rep} k{k} LR17 = stage 2', r['LR17_gap_prmscore'], STAGE2[(cell.bench, prm, rep, str(k))])
            rows.append(r)
        print(cell.bench, prm, 'rep', rep, 'all: LR17-POS2', round(rows[-1]['LR17_gap_prmscore'] - rows[-1]['POS2_gap_prmscore'], 2),
              'LR18-LR17', round(rows[-1]['LR18_gap_prmscore'] - rows[-1]['LR17_gap_prmscore'], 2), flush=True)


def summarize(rows):
    S, kstar = [], {}
    for bench in PROT['benchmarks']:
        for prm in PROT['prms']:
            D = []
            for k in B.BUDGETS:
                R = [r for r in rows if r['bench'] == bench and r['prm'] == prm and r['k'] == k and 'fail' not in r]
                if not R:
                    continue
                d = dict(bench=bench, prm=prm, k=k, reps=len(R),
                         means={f'{a}_{m}': float(np.mean([r[f'{a}_{m}'] for r in R]))
                                for a in ARMS + ('PRM', 'AUTO_N16', 'POSONLY') for m in ('gap_prmscore', 'within_auc')})
                for a, b in CONTRASTS:
                    for m in ('gap_prmscore', 'within_auc'):
                        x = np.array([r[f'{a}_{m}'] - r[f'{b}_{m}'] for r in R])
                        d[f'{a}-{b}:{m}'] = dict(mean=float(x.mean()), p025=float(np.percentile(x, 2.5)),
                                                 p975=float(np.percentile(x, 97.5)), pos=int((x > 0).sum()))
                S.append(d); D.append(d)
            for a, b in CONTRASTS:
                ok = [x[f'{a}-{b}:gap_prmscore']['mean'] > 0 and x[f'{a}-{b}:gap_prmscore']['pos'] >= 16 for x in D]
                kstar[f'{bench}|{prm}|{a} vs {b}'] = next((D[i]['k'] for i in range(len(D)) if all(ok[i:])), None)
    return S, kstar


def main():
    rows = []
    for prm in PROT['prms']:
        for bench in PROT['benchmarks']:
            cell = S1.Cell(bench, prm); run_cell(cell, prm, rows); del cell
    with open(HERE / 'POSITION.csv', 'w', newline='') as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    S, kstar = summarize(rows)
    (HERE / 'SUMMARY.json').write_text(json.dumps(dict(kstar=kstar, summary=S,
                                                       parity=dict(n=len(S1.PAR), failed=[p for p in S1.PAR if not p['ok']])), indent=1))
    print(json.dumps(kstar, indent=1))


if __name__ == '__main__':
    t = time.time(); main(); print(f'{time.time() - t:.0f}s')
