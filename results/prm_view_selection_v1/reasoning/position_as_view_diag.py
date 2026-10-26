"""Diagnostic for Omri's question (2026-10-26): after removing position from the channels, can position be added back
as a separate SML/Dawid-Skene voter?

Under the conditional-independence (CI) model every pair of views must satisfy (continuous analogue of Jaffe 2016,
Lemma 1)  corr(a, b) = corr(a, y) * corr(b, y)  and  corr(a, b | y) = 0.
Derivation (see the research note): residualizing channel x = u*y + beta*p + e on position p removes u*E[y|p] as well,
so x_res = u*(y - m(p)) + e with m(p) = E_lin[y | p]. Hence
    corr(x_res, p)      ~ 0                      by construction (marginal),
    corr(x_res, p | y)  = -u * Cov(m(p), p | y) / ... < 0     when errors are later (m increasing) and u > 0.
So position and a residualized channel cannot both be CI voters: their marginal covariance is 0 although both are
informative, i.e. a negative within-class dependence exactly cancels the shared signal.

Labels are used here as a diagnostic only. No fusion, no PRMScore.

    python position_as_view_diag.py   # -> POSITION_AS_VIEW_DIAG.csv
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval8 as V  # noqa: E402
S1 = V.S1


def centered_pos(off):
    n = np.diff(off); p = V.rel_pos(off)
    return p - np.repeat(np.add.reduceat(p, off[:-1]) / n, n)


def main():
    SR = json.loads((HERE / 'RULE_POS.json').read_text())['S_R']
    rows = []
    for b, p in V.CELLS:
        for rep, path in (('RAW', V.VS / 'stage1' / 'cache' / f'{V.key(b, p)}.npz'), ('POS', V.POSC / f's1_{V.key(b, p)}.npz')):
            o = S1.CellData.from_light(path); off = np.load(path)['off']
            m = o.pool & (o._yv_m | o._nyv_m); y = o._nyv_m[m].astype(float); pc = centered_pos(off)[m]
            ok, er = y == 0, y == 1
            r_py = np.corrcoef(pc, y)[0, 1]
            for view in ['PRM'] + SR:
                k = 0 if view == 'PRM' else V.NAMES.index(view) + 1
                x = o.Z[m, k]
                r_xy = np.corrcoef(x, y)[0, 1]
                rows.append(dict(cell=V.key(b, p), rep=rep, view=view, corr_pos_y=float(r_py), corr_x_y=float(r_xy),
                                 corr_x_pos=float(np.corrcoef(x, pc)[0, 1]),
                                 ci_prediction=float(r_py * r_xy),
                                 corr_x_pos_given_valid=float(np.corrcoef(x[ok], pc[ok])[0, 1]),
                                 corr_x_pos_given_error=float(np.corrcoef(x[er], pc[er])[0, 1])))
        print('done', b, p, flush=True)
    V.E.write_csv(HERE / 'POSITION_AS_VIEW_DIAG.csv', rows)


if __name__ == '__main__':
    main()
