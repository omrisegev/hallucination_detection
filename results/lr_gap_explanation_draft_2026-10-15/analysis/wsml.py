"""Whitened SML: w = (S + lam I)^-1 mu_hat. mu_hat = rank-1 loadings fitted to CROSS-group covariance entries only."""
import json, sys
import numpy as np
from numpy.linalg import eigh, solve
from decomp import within, ROOT

def cross_rank1(S, G, iters=200):
    mask = G[:, None] != G[None, :]
    A = np.where(mask, S, 0.)
    v = eigh(A)[1][:, -1]; lam = 1.
    mu = v.copy()
    for _ in range(iters):  # impute within-block entries with the rank-1 fit (alternating)
        M = np.outer(mu, mu)
        A = np.where(mask, S, M)
        ev, V = eigh(A); mu = V[:, -1] * np.sqrt(max(ev[-1], 1e-12))
    return mu

if __name__=='__main__':
  p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
  sf=np.repeat(p['fold'].astype(int),np.diff(off))
  for bank in sys.argv[1:]:
      X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); sc={}
      for o in range(5):
          L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
          G=np.asarray(L['diagnostic']['group']); wl=np.asarray(L['weight']); tr,te=sf!=o,sf==o
          mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1; Ztr,Zte=(X[tr]-mu)/sd,(X[te]-mu)/sd
          S=np.cov(Ztr.T); m=S.shape[0]; ls=Ztr@(wl*sd)
          oracle=Ztr[y[tr]==1].mean(0)-Ztr[y[tr]==0].mean(0)
          mh=cross_rank1(S,G)
          if np.corrcoef(Ztr@mh, ls)[0,1]<0: mh=-mh
          for lam in (0.01,0.1,1.0):
              for name,mv in (('ORACLE_LDA',oracle),('WSML',mh)):
                  w=solve(S+lam*np.eye(m),mv)
                  sc.setdefault(f'{name}_lam{lam}',np.full(len(y),np.nan))[te]=Zte@w
          sc.setdefault('UNWHITENED_mu_hat',np.full(len(y),np.nan))[te]=Zte@mh
      print(bank,{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
  