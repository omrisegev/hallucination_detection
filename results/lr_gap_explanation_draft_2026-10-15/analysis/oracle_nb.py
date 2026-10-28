"""Ceiling for any per-channel (independence-assuming) weighting, 2026-10-20.
ORACLE_VOTER: w = mu (true training-fold class-mean difference, standardized) = perfect SML loadings as weights.
ORACLE_NB:    w_i = mu_i / within-class variance_i (best rule if channels were conditionally independent).
ORACLE_LDA:   w = (S+0.1I)^-1 mu (conditions channels on each other). Same folds/population as ladder.py."""
import sys
import numpy as np
from numpy.linalg import solve
from decomp import within, ROOT
p = np.load(f'{ROOT}/inputs/population.npz'); off = p['offsets'].astype(int); y = p['y'].astype(int)
sf = np.repeat(p['fold'].astype(int), np.diff(off))
for bank in sys.argv[1:]:
    X = np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m = X.shape[1]
    sc = {k: np.full(len(y), np.nan) for k in ['ORACLE_VOTER', 'ORACLE_NB', 'ORACLE_LDA']}
    for o in range(5):
        tr, te = sf != o, sf == o
        mu0, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Ztr, Zte = (X[tr] - mu0) / sd, (X[te] - mu0) / sd
        a, b = Ztr[y[tr] == 1], Ztr[y[tr] == 0]; mu = a.mean(0) - b.mean(0)
        wv = (len(a) * a.var(0) + len(b) * b.var(0)) / len(Ztr)
        sc['ORACLE_VOTER'][te] = Zte @ mu
        sc['ORACLE_NB'][te] = Zte @ (mu / wv)
        sc['ORACLE_LDA'][te] = Zte @ solve(np.cov(Ztr.T) + 0.1 * np.eye(m), mu)
    print(bank, {k: round(within(v, off, y), 4) for k, v in sc.items()}, flush=True)
