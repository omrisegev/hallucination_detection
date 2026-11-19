"""Clean-side PU: a known-clean sample estimates 1-pi as its share in the pool (BBE on the reversed score)."""
import numpy as np, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from ssl_lib import bbe
import sim_bbe as S
pi, mu, lvl, nA = .646, 2.2, .4, 850
res = {}
for rep in range(30):
    U = S.make(nA, pi, mu, lvl); ref = np.concatenate([s for s, _ in U]); sU = S.score(U, ref)
    for nc in (150, 600):
        Cl = [a for a in S.make(int(nc / (1 - pi) * 1.3), pi, mu, lvl) if not a[1]][:nc]
        res.setdefault(f'clean anchor n={nc}: pi-hat', []).append(1 - bbe(-S.score(Cl, ref), -sU)[0])
    Wp = [a for a in S.make(int(600 / pi * 1.3), pi, mu, lvl) if a[1]][:600]
    res.setdefault('wrong anchor n=600: pi-hat', []).append(bbe(S.score(Wp, ref), sU)[0])
for k, v in res.items():
    v = np.array(v); print(f'{k:30s} mean {v.mean():.3f} sd {v.std():.3f}')
