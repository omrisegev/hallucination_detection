"""Explain the gap: voter (w ∝ loadings) vs conditioned (w = Σ^-1 μ), with TRUE μ from the training fold."""
import json, sys
import numpy as np
from numpy.linalg import solve, eigh
from decomp import within, ROOT, lr
from wsml import cross_rank1
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
rows=[]
for bank in sys.argv[2:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m=X.shape[1]
    keys=['ALL_EQ','LSML','LF_VOTER_muhat','ORACLE_VOTER','ORACLE_BLOCK_LDA','ORACLE_LDA','LR']
    sc={k:np.full(len(y),np.nan) for k in keys}; pr=[]
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); tr,te=sf!=o,sf==o
        mu0,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1; Ztr,Zte=(X[tr]-mu0)/sd,(X[te]-mu0)/sd
        S=np.cov(Ztr.T); mu=Ztr[y[tr]==1].mean(0)-Ztr[y[tr]==0].mean(0)
        sc['ALL_EQ'][te]=Zte@np.sign(mu)   # oriented equal weights (oracle signs)
        sc['LSML'][te]=X[te]@np.asarray(L['weight'])
        mh=cross_rank1(S,G); mh=mh if mh@mu>0 else -mh
        sc['LF_VOTER_muhat'][te]=Zte@mh
        sc['ORACLE_VOTER'][te]=Zte@mu
        B=np.where(G[:,None]==G[None,:],S,0.)
        sc['ORACLE_BLOCK_LDA'][te]=Zte@solve(B+0.1*np.eye(m),mu)
        sc['ORACLE_LDA'][te]=Zte@solve(S+0.1*np.eye(m),mu)
        sc['LR'][te]=lr(X[tr],y[tr],X[te])[0]
        ev=np.clip(eigh(S)[0],0,None); pr.append(ev.sum()**2/(ev**2).sum())
    r={'bank':bank,'m':m,'eff_dim':round(float(np.mean(pr)),1),**{k:round(within(v,off,y),4) for k,v in sc.items()}}
    rows.append(r); print(r,flush=True)
json.dump(rows,open(sys.argv[1],'w'),indent=1)
