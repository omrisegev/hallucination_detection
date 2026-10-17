import json, sys
import numpy as np
from numpy.linalg import eigh, solve
from decomp import within, ROOT, group_pcs
from xg import sml_w
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); sc={}
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); tr,te=sf!=o,sf==o
        mu,sd=X[tr].mean(0),X[tr].std(0); sd[sd==0]=1; Ztr,Zte=(X[tr]-mu)/sd,(X[te]-mu)/sd
        # oracle instrument: label (within-group Fisher toward y), groups combined by equal z-sum
        W=[]
        for g in np.unique(G):
            i=G==g; Zg=Ztr[:,i]; yc=y[tr]-y[tr].mean()
            w=np.zeros(X.shape[1]); w[i]=solve(np.atleast_2d(np.cov(Zg.T))+0.1*np.eye(i.sum()),Zg.T@yc/len(yc)); W.append(w)
        W=np.array(W); xt=Ztr@W.T; m_,s_=xt.mean(0),xt.std(0)
        sc.setdefault('ORACLE_GROUPFISHER_eqsum',np.full(len(y),np.nan))[te]=((Zte@W.T-m_)/s_).sum(1)
        # label-free: top-r group PCs as SML views
        for r in (1,2,3):
            B,mu2=group_pcs(X[tr],G,r); P=np.column_stack([b[2] for b in B])
            Vt=(X[tr]-mu2)@P; m2,s2=Vt.mean(0),Vt.std(0); Vt=(Vt-m2)/s2
            v=sml_w(Vt)
            # orient global sign to agree with L-SML score
            if np.corrcoef(Vt@v, X[tr]@np.asarray(L['weight']))[0,1]<0: v=-v
            sc.setdefault(f'LSML_PC{r}_views',np.full(len(y),np.nan))[te]=(((X[te]-mu2)@P-m2)/s2)@v
    print(bank,{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
