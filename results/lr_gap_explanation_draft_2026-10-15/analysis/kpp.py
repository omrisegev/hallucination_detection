"""Whitened kurtosis projection pursuit (one-unit FastICA), init = L-SML direction. Label-free; fit on train fold."""
import json, sys
import numpy as np
from decomp import ROOT, within
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off)); aid=np.repeat(np.arange(len(off)-1),np.diff(off))
def acenter(X):
    cnt=np.bincount(aid); M=np.vstack([np.bincount(aid,X[:,j])/cnt for j in range(X.shape[1])]).T
    return X-M[aid]
ITS=(0,1,2,3,5,10,30)
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); Xc=acenter(X); sc={}
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        wl=np.asarray(L['weight']); tr,te=sf!=o,sf==o
        for lam in (0.01,0.1):
            S=np.cov(Xc[tr].T); ev,V=np.linalg.eigh(S); ev=ev+lam*ev.mean()
            Wh=V/np.sqrt(ev)            # whitening: z = Xc @ Wh
            Ztr=Xc[tr]@Wh
            u=np.linalg.solve(Wh.T@np.linalg.inv(Wh@Wh.T) if False else np.eye(len(ev)),(V*np.sqrt(ev)).T@wl)  # L-SML dir in whitened coords
            u/=np.linalg.norm(u)
            for it in range(max(ITS)+1):
                if it in ITS:
                    w=Wh@u
                    if np.corrcoef(Xc[tr]@w,Xc[tr]@wl)[0,1]<0: w=-w
                    sc.setdefault(f'KPP_lam{lam}_it{it}',np.full(len(y),np.nan))[te]=X[te]@w
                s=Ztr@u; u=(Ztr*(s**3)[:,None]).mean(0)-3*u; u/=np.linalg.norm(u)
    print(bank,{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
