import json, sys
import numpy as np
from numpy.linalg import solve
from decomp import within, ROOT
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off)); aid=np.repeat(np.arange(len(off)-1),np.diff(off))
def top1(s, rows):
    lab=np.zeros(len(s),bool)
    o=off; 
    for b,e in zip(o[:-1],o[1:]):
        if rows[b]: lab[b+np.argmax(s[b:e])]=True
    return lab
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); sc={}
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); wl=np.asarray(L['weight']); tr,te=sf!=o,sf==o
        mu,sd=X.mean(0),X.std(0); Z=(X-mu)/sd; S=np.cov(Z[tr].T); m=S.shape[0]
        s=X@wl
        # (a) global pseudo-label = top L-SML step per answer
        pl=top1(s,tr)&tr; mh=Z[pl].mean(0)-Z[tr&~pl].mean(0)
        for lam in (0.1,1.0):
            sc.setdefault(f'PSEUDO_LDA_lam{lam}',np.full(len(y),np.nan))[te]=Z[te]@solve(S+lam*np.eye(m),mh)
        # (b) cross-fitted: loadings of group g from pseudo-labels made WITHOUT group g
        mx=np.zeros(m)
        for g in np.unique(G):
            i=G==g; pg=top1(s-X[:,i]@wl[i],tr)&tr
            mx[i]=Z[pg][:,i].mean(0)-Z[tr&~pg][:,i].mean(0)
        for lam in (0.1,1.0):
            sc.setdefault(f'XFIT_PSEUDO_LDA_lam{lam}',np.full(len(y),np.nan))[te]=Z[te]@solve(S+lam*np.eye(m),mx)
    print(bank,{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
