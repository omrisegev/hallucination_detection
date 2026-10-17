import json, sys
import numpy as np
from decomp import ROOT, within
from kpp import acenter, off, y, sf
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); Xc=acenter(X); sc={}
    cosl=[]
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        wl=np.asarray(L['weight']); tr,te=sf!=o,sf==o
        S=np.cov(Xc[tr].T); ev,V=np.linalg.eigh(S); ev=ev+0.01*ev.mean(); Wh=V/np.sqrt(ev); Ztr=Xc[tr]@Wh
        u=(V*np.sqrt(ev)).T@wl; u/=np.linalg.norm(u)
        s=Ztr@u; g=(Ztr*(s**3)[:,None]).mean(0); g-= (g@u)*u; g/=np.linalg.norm(g)
        # does the kurtosis-gradient agree with the oracle LDA improvement direction (diagnostic only)?
        mu=Xc[tr][y[tr]==1].mean(0)-Xc[tr][y[tr]==0].mean(0); uo=(Wh.T@mu); uo/=np.linalg.norm(uo); d=uo-(uo@u)*u
        cosl.append(float(g@d/np.linalg.norm(d)))
        for t in (0.05,0.1,0.2,0.3,0.5):
            w=Wh@((u+t*g)/np.linalg.norm(u+t*g))
            sc.setdefault(f't{t}',np.full(len(y),np.nan))[te]=X[te]@w
    print(bank,'cos(kurt-grad, oracle-improvement)=',np.round(cosl,3),{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
