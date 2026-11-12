"""DIAGNOSTIC ONLY (labels). Follow-up of oracle_flip.py: L-SML and GSP-T re-orient every column by corr with the
fused score, which undoes a manual flip. Here the oracle signs are FIXED: GSP-T skips its sign step (sign = +1 after
the flip) and L-SML is replaced by its groups + signed SML with the oracle-flipped columns (weights may still be
negative; reported). Output ORACLE_FLIP_FIXED.csv."""
import sys, runpy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
import signed_weights as SW  # noqa: E402
T, E, N = AB.T, AB.E, AB.N


def gsp_fixed(c, X):
    real = np.corrcoef
    def corr_pos(a, b=None):                       # make the sign step return +1 for every column
        m = real(a, b); m = np.array(m, float); m[0, 1] = m[1, 0] = abs(m[0, 1]); return m
    SW.np.corrcoef = corr_pos
    try:
        return SW.gsp_v(c, X, False, False)['rank']
    finally:
        SW.np.corrcoef = real


def evaluate(c, X, cell, rows, pos=None):
    fe = np.array([R0.fe_auc(c, X[:, j], pos) for j in range(16)]); flip = fe < .5
    Xf = X * np.where(flip, -1., 1.)
    def sla(pr):
        if pos is not None: pr = np.array([pos[j][p] if p >= 0 else -1 for j, p in enumerate(pr)])
        return T.pb_metrics(c, pr)['macro SLA']
    for name, r in (('GSP-T, oracle signs fixed', gsp_fixed(c, Xf)),):
        am = np.array([int(np.argmax(r[a:b])) for a, b in zip(c.off[:-1], c.off[1:])])
        rows.append(dict(cell=cell, rank=name, first_error_AUC=R0.fe_auc(c, r, pos), SLA_argmax=sla(am),
                         SLA_GAP=sla(T.pb_predict(c, AB.gap(c, r)))))
        print(cell, name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)


rows = []
g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
evaluate(g['c'], g['X'], 'Qwen3-8B', rows, g['pos'])
for prm in AB.PRMS:
    c, X, ab = AB.load_cell('processbench', prm); evaluate(c, X[:, 2:], prm, rows); del c
E.write_csv(HERE / 'ORACLE_FLIP_FIXED.csv', rows)
