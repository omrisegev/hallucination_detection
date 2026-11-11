"""Add TELE16 standalone (stored TELE_N16 = L-SML of the 16 standalone channels, GAP flags, no PRM) to BRACHA_PANEL.csv.
PRMBench: the standalone cell; Socratic / Hard2Verify: the standalone CAB cells of bracha_lr_transfer_v1 (same answers;
on Socratic the standalone cell keeps the standalone model's nonempty steps, 2 steps differ from the PRM cell).
Parity: every value reproduces bracha_lr_transfer_v1/RESULTS.csv (TELE_N16, standalone) to 1e-6."""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as Rn  # noqa: E402
from bracha_panel import metric, rates  # noqa: E402  (bracha_panel's loop is guarded below)
E, A, P = Rn.E, Rn.C.A, Rn.N.PRMS
REF = {('prmbench', p): v for p, v in zip(P, [68.26, 67.80, 67.63, 68.20])}
REF.update({('socratic', p): v for p, v in zip(P, [63.29, 65.31, 63.81, 62.45])})
REF.update({('hard2verify', p): v for p, v in zip(P, [47.22, 43.47, 39.33, 40.24])})
rows = [r for r in csv.DictReader(open(HERE / 'BRACHA_PANEL.csv')) if r['arm'] != 'TELE16 standalone (no PRM)']
for p in P:
    for b in ('prmbench', 'socratic', 'hard2verify'):
        if b == 'prmbench':
            c = E.load('prmbench', f'{p}__standalone')
        else:
            key = f'cab|sa|{b}'; A.REG[key] = dict(mode='sa', bench=b, common=A.common_uids(b)); c = E.load(key, p)
        f = c.flags('TELE_N16', 'GAP'); v = metric(c, f)
        assert abs(round(v, 2) - REF[(b, p)]) < .006, (b, p, v)
        rows.append(dict(benchmark=b, prm=p, arm='TELE16 standalone (no PRM)', value=v, **(rates(c, f) if b == 'prmbench' else {})))
E.write_csv(HERE / 'BRACHA_PANEL.csv', rows)
print('ok', len(rows))
