"""DIAGNOSTIC ONLY (uses labels; not a method). How much do the channels that are inverted on the first error cost?
Per cell: channels with first-error AUC < .5 (first error vs earlier clean steps, labels used) are multiplied by -1,
then L-SML, equal mean and GSP-T are refitted. ProcessBench: Qwen3-8B (Mind the Gap scorer) and the 4 base models.
Metrics: first-error AUC of the rank, macro SLA of argmax and of GAP (first flagged step). Output ORACLE_FLIP.csv."""
import sys, runpy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
T, E, Rn, N = AB.T, AB.E, AB.Rn, AB.N


def evaluate(c, X, cell, rows, pos=None):
    fe = np.array([R0.fe_auc(c, X[:, j], pos) for j in range(16)])
    flip = fe < .5
    def sla(pr):
        if pos is not None: pr = np.array([pos[j][p] if p >= 0 else -1 for j, p in enumerate(pr)])
        return T.pb_metrics(c, pr)['macro SLA']
    for tag, Xv in (('original', X), ('inverted channels flipped (oracle)', X * np.where(flip, -1., 1.))):
        s, _, _ = N.lsml(Xv, Xv.mean(1))
        try: gsp = Rn.c4u_t(c, Xv)['rank']
        except Exception as e: gsp = None; print('GSP fail', e)
        for name, r in (('L-SML', s), ('equal mean', Xv.mean(1)), ('GSP-T', gsp)):
            if r is None: continue
            am = np.array([int(np.argmax(r[a:b])) for a, b in zip(c.off[:-1], c.off[1:])])
            rows.append(dict(cell=cell, columns=tag, flipped=' '.join(np.array(T.N16)[flip]), rank=name,
                             first_error_AUC=R0.fe_auc(c, r, pos), SLA_argmax=sla(am), SLA_GAP=sla(T.pb_predict(c, AB.gap(c, r)))))
            print(cell, tag[:8], name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)


rows = []
g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
evaluate(g['c'], g['X'], 'Qwen3-8B', rows, g['pos'])
for prm in AB.PRMS:
    c, X, ab = AB.load_cell('processbench', prm)
    evaluate(c, X[:, 2:], prm, rows); del c
E.write_csv(HERE / 'ORACLE_FLIP.csv', rows)
