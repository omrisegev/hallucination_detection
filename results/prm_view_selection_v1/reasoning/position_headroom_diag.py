"""POST-HOC oracle diagnostic (labels used; not pre-declared): how much can position add on top of a score, at the best
possible weight? score_c = answer_z(base) + c * answer_z(relative position), c on a grid; PRMScore under GAP.
Bases: the PRM alone, and the fused SUB5 / S_R scores in the position-neutral representation (saved in scores/).
This measures headroom only; choosing c with labels is not a method.

    python position_headroom_diag.py   # -> POSITION_HEADROOM.csv
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval8 as V  # noqa: E402
E = V.E
GRID = np.round(np.r_[0, np.linspace(.05, 1.5, 30)], 3)
BASES = {'PRM': 'PRM', 'SUB5|POS|+PRM': 'SUB5__POS__PPRM', 'S_R|POS|+PRM': 'S_R__POS__PPRM', 'SUB5|POS|alone': 'SUB5__POS__alone'}


def main():
    rows = []
    for b, p in V.CELLS:
        c = E.load(b, p); D = np.load(V.SCORES / f'{V.key(b, p)}.npz')
        zp = E.answer_z(V.rel_pos(c.off), c.off)
        for name, k in BASES.items():
            zb = E.answer_z(D[k], c.off)
            for g in GRID:
                m = V.metrics(c, zb + g * zp, b)
                rows.append(dict(cell=V.key(b, p), base=name, c=float(g), PRMScore=m['PRMScore'], within_auc=m['within_auc']))
        print('done', b, p, flush=True); del c
    E.write_csv(HERE / 'POSITION_HEADROOM.csv', rows)


if __name__ == '__main__':
    main()
