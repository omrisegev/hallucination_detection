"""DIAGNOSTIC (labels). Direction of each TELE16 channel (R1 column, as GSP-T sees it) per benchmark x base model:
within-answer AUC (PRMBench/Socratic/H2V; error steps vs clean steps in the same answer), first-error AUC (first error
vs earlier clean steps; all 4 benchmarks) and the sign GSP-T assigns (corr with the L-SML score s). AUC < .5 = the
channel is inverted relative to errors although GSP gives it its sign. Also Qwen3-8B ProcessBench. Output CHANNEL_DIRECTION.csv."""
import sys, runpy
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
import r0 as R0  # noqa: E402
T, E, N = AB.T, AB.E, AB.N


def fe_err(c, x):
    v = []; err = c.err.astype(bool)
    for a, b in zip(c.off[:-1], c.off[1:]):
        f = np.flatnonzero(err[a:b])
        if not len(f) or f[0] == 0: continue
        k = f[0]; v.append(np.mean((x[a + k] > x[a:a + k]) + .5 * (x[a + k] == x[a:a + k])))
    return float(np.mean(v))


rows = []
for bench in AB.BENCHES:
    for prm in AB.PRMS:
        c, X, ab = AB.load_cell(bench, prm); Xt = X[:, 2:]
        s, _, _ = N.lsml(Xt, Xt.mean(1))
        for j, n in enumerate(T.N16):
            x = Xt[:, j]; sg = 1 if np.corrcoef(x, s)[0, 1] >= 0 else -1
            r = dict(bench=bench, model=prm, channel=n, gsp_sign=sg)
            if bench == 'processbench':
                r['first_error_AUC'] = R0.fe_auc(c, x)
            else:
                keep = None if bench == 'prmbench' else c.keep
                r['within_AUC'] = float(np.nanmean(E.per_answer_auc(c.err, x, c.off, keep)))
                r['first_error_AUC'] = fe_err(c, x)
            rows.append(r)
        print(bench, prm, flush=True); del c
g = runpy.run_path(str(HERE / 'qwen3_8b_pb.py'))
c, X, pos = g['c'], g['X'], g['pos']; s, _, _ = N.lsml(X, X.mean(1))
for j, n in enumerate(T.N16):
    rows.append(dict(bench='processbench', model='qwen3_8b_mtg', channel=n, gsp_sign=1 if np.corrcoef(X[:, j], s)[0, 1] >= 0 else -1,
                     first_error_AUC=R0.fe_auc(c, X[:, j], pos)))
E.write_csv(HERE / 'CHANNEL_DIRECTION.csv', rows)
