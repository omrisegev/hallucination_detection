"""Stage 1 of the supervised/calibrated plan, as frozen in PROTOCOL.json: label-free L-SML telemetry direction,
supervised PRM/telemetry mixing ratio (ANCH), against AUTO, the PRM and a full supervised LR. CPU only, stored scores.

    python anchor.py      # -> RESULTS.json, ROWS.csv, PARITY.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'prm_boost_math_rigor_v1'))
import common as K  # noqa: E402
import stage8_auto_forced_k as S8  # noqa: E402
E = K.E

PROT = json.loads((HERE / 'PROTOCOL.json').read_text())
PRMS = PROT['prms']
CGRID = [0.01, 0.1, 1.0, 10.0]
S8_REF = {(r['bench'], r['prm']): r['A']['auto'] for r in
          json.loads((HERE.parent / 'prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())}
INPUTS = {'ANCH_T16': 'T16', 'LR17': 'Z16'}
CONTRASTS = [('ANCH_T16', 'AUTO_N16'), ('ANCH_T16', 'PRM'), ('ANCH_T16', 'LR17'),
             ('LR17', 'AUTO_N16'), ('LR17', 'PRM')]
PAR = []


def gate(name, got, ref, tol=1e-6):
    d = abs(float(got) - float(ref)); PAR.append(dict(check=name, got=float(got), ref=float(ref), ok=d <= tol))
    print(('PASS ' if d <= tol else 'FAIL ') + name, round(float(got), 6), round(float(ref), 6), flush=True)
    if d > tol:
        raise SystemExit('parity failed: ' + name)


class Cell:
    def __init__(self, bench, prm):
        c = self.c = E.load(bench, prm); self.bench = bench
        self.zp = E.answer_z(c.arms[K.prm_arm(c)], c.off)
        Z = E.channels(c, 'N16')
        self.X = {'T16': np.column_stack([self.zp, E.answer_z(c.arms['TELE_N16'], c.off)]),
                  'Z16': np.column_stack([self.zp] + [E.answer_z(Z[:, j], c.off) for j in range(Z.shape[1])])}
        self.y = c.err.astype(int)
        ans = np.arange(len(c.ids)); self.ans_of_step = np.repeat(ans, np.diff(c.off))
        self.g = np.asarray(c.groups)[self.ans_of_step]
        self.train = c.keep.astype(bool) if bench == 'prmbench' else np.ones(len(self.y), bool)
        self.auto, _ = S8.auto(Z, self.zp)


def fit(X, y, g):
    best = None
    for C in CGRID:
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000)).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
        if best is None or np.mean(aucs) > best[1]:
            best = (C, float(np.mean(aucs)))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], class_weight='balanced', max_iter=3000)).fit(X, y)
    return m, best[0]


def share(m):
    w = np.abs(m[-1].coef_[0]); return float(w[0] / w.sum())


def evaluate(cell, scores, regime, src, prm, out):
    c = cell.c; _, ans_eval = K.eval_rows(c, cell.bench)
    W, inv, _ = E.boot_weights(c); Wa = W[:, inv]
    arms = {'PRM': cell.zp, 'AUTO_N16': cell.auto, 'TELE_N16': c.arms['TELE_N16'], **scores}
    st = {a: S8.stats(c, s, ans_eval) for a, s in arms.items()}
    pts = {a: S8.point(c, v) for a, v in st.items()}
    con = {f'{a} - {b}': S8.boot(c, st[a], st[b], Wa) for a, b in CONTRASTS}
    out.append(dict(regime=regime, train=src, bench=cell.bench, prm=prm, points=pts, contrasts=con))
    print(regime, src, '->', cell.bench, prm, {a: round(p['gap_prmscore'], 2) for a, p in pts.items()}, flush=True)


def main():
    res, shares = [], []
    for prm in PRMS:
        cells = {b: Cell(b, prm) for b in K.BENCHES}
        for b, cell in cells.items():                     # parity first: AUTO must reproduce stage8
            _, ans_eval = K.eval_rows(cell.c, b); p = S8.point(cell.c, S8.stats(cell.c, cell.auto, ans_eval))
            gate(f'{prm} {b} AUTO gap_prmscore', p['gap_prmscore'], S8_REF[(b, prm)]['gap_prmscore'])
            gate(f'{prm} {b} AUTO within_auc', p['within_auc'], S8_REF[(b, prm)]['within_auc'])
        full = {}
        for src in ('prmbench', 'socratic'):              # IN: grouped cross-fit; also the full-data model for XFER
            cell = cells[src]; oof = {a: np.zeros(len(cell.y)) for a in INPUTS}
            for a, k in INPUTS.items():
                X, y, g, tr = cell.X[k], cell.y, cell.g, cell.train
                for fold, (itr, ite) in enumerate(GroupKFold(5).split(X, y, g)):
                    rows = itr[tr[itr]]
                    m, C = fit(X[rows], y[rows], g[rows]); oof[a][ite] = m.decision_function(X[ite])
                    shares.append(dict(regime='IN', train=src, prm=prm, arm=a, fold=fold, C=C, prm_share=share(m)))
                m, C = fit(X[tr], y[tr], g[tr]); full[(src, a)] = m
                shares.append(dict(regime='XFER', train=src, prm=prm, arm=a, fold='all', C=C, prm_share=share(m),
                                   coef=[float(v) for v in m[-1].coef_[0][:2]]))
            evaluate(cell, oof, 'IN', src, prm, res)
        for src, dst in (('prmbench', 'socratic'), ('prmbench', 'hard2verify'), ('socratic', 'prmbench'), ('socratic', 'hard2verify')):
            cell = cells[dst]
            evaluate(cell, {a: full[(src, a)].decision_function(cell.X[k]) for a, k in INPUTS.items()}, 'XFER', src, prm, res)
        (HERE / 'RESULTS.json').write_text(json.dumps(dict(results=res, shares=shares), indent=1))
    rows = []
    for r in res:
        for a, p in r['points'].items():
            rows.append(dict(regime=r['regime'], train=r['train'], bench=r['bench'], prm=r['prm'], arm=a,
                             main=p['gap_prmscore'], within_auc=p['within_auc']))
    with open(HERE / 'ROWS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    (HERE / 'PARITY.json').write_text(json.dumps(dict(n=len(PAR), failed=[p for p in PAR if not p['ok']], checks=PAR), indent=1))
    print('done', len(res), 'evaluations,', len(PAR), 'parity checks')


if __name__ == '__main__':
    t = time.time(); main(); print(f'{time.time() - t:.0f}s')
