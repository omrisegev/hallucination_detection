import json, sys
import numpy as np
from numpy.linalg import eigh, solve
from decomp import ROOT
from wsml import cross_rank1
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
    L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_0/MODELS.json'))['selector_models']['LSML_CONT']
    G=np.asarray(L['diagnostic']['group']); tr=sf!=0
    Z=(X[tr]-X[tr].mean(0))/X[tr].std(0); S=np.cov(Z.T)
    mu=Z[y[tr]==1].mean(0)-Z[y[tr]==0].mean(0); mh=cross_rank1(S,G)
    if cos(mh,mu)<0: mh=-mh
    pi=y[tr].mean(); Sw=S-pi*(1-pi)*np.outer(mu,mu)
    mask=G[:,None]!=G[None,:]
    # how much of the cross-group covariance is explained by the label rank-1 term?
    lab=pi*(1-pi)*np.outer(mu,mu)
    frac=np.sum((lab*mask)**2)/np.sum((S*mask)**2)
    ev,V=eigh(S)
    lo=V[:,:len(ev)//2]  # low-variance half
    print(bank,'cos(mu_hat,mu)=%.3f'%cos(mh,mu),
          'cos in low-var half=%.3f'%cos(lo.T@mh,lo.T@mu),
          'cross-group cov energy from label term=%.3f'%frac,
          'cos(Sw^-1 mu_hat, Sw^-1 mu)=%.3f'%cos(solve(S+0.1*np.eye(len(mu)),mh),solve(S+0.1*np.eye(len(mu)),mu)))
