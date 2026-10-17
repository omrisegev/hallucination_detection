"""Cluster-contrast kurtosis refinement: w = w_lsml + C a, C = within-group non-leading eigvecs (standardized coords).
a found by gradient ascent on within-answer kurtosis K = E[s^4]/E[s^2]^2 (label-free)."""
import json, sys
import numpy as np
from decomp import ROOT, within
from kpp import acenter, off, y, sf
def kurt_grad(Z, w):
    s=Z@w; m2=(s**2).mean(); m4=(s**4).mean()
    return m4/m2**2, (4*(Z*(s**3)[:,None]).mean(0)/m2**2 - 4*m4*(Z*s[:,None]).mean(0)/m2**3)
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); Xc=acenter(X); sc={}; ks={}
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); tr,te=sf!=o,sf==o
        sd=Xc[tr].std(0); sd[sd==0]=1; Z=Xc[tr]/sd
        w0=np.asarray(L['weight'])*sd; w0/=np.sqrt(np.var(Z@w0))
        cols=[]
        for g in np.unique(G):
            i=np.flatnonzero(G==g)
            if len(i)<2: continue
            ev,V=np.linalg.eigh(np.cov(Z[:,i].T))
            for j in range(len(i)-1):  # all but leading
                c=np.zeros(len(sd)); c[i]=V[:,j]/np.sqrt(max(ev[j],1e-3)); cols.append(c)  # unit-variance contrast directions
        C=np.array(cols).T; a=np.zeros(C.shape[1]); ZC=Z@C
        steps=[0,5,10,20,40]; lr_=0.05
        for it in range(max(steps)+1):
            w=w0+C@a
            if it in steps:
                sc.setdefault(f'it{it}',np.full(len(y),np.nan))[te]=(Xc[te]/sd)@w
                ks.setdefault(f'it{it}',[]).append(round(kurt_grad(Z,w)[0],3))
            k,gw=kurt_grad(Z,w); ga=C.T@gw; a+=lr_*ga/ (np.linalg.norm(ga)+1e-12)
    print(bank,'nC=',C.shape[1],{k:round(within(v,off,y),4) for k,v in sc.items()},'kurt',{k:v[0] for k,v in ks.items()},flush=True)
