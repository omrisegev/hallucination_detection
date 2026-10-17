import json, sys
import numpy as np
from numpy.linalg import eigh, solve
sys.path.insert(0,'results/lr_gap_explanation_draft_2026-10-15/analysis')
from decomp import within, ROOT
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
for bank in ['N16','N25','N43','N56']:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m=X.shape[1]
    K=[1,2,3,5,8,12,m]; K=sorted(set(k for k in K if k<=m))
    sc={k:np.full(len(y),np.nan) for k in K}; neg_iu=[]; neg_lr=[]; neg_or2=[]
    for o in range(5):
        d=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))
        wi=np.asarray(d['selector_models']['IU_PCR']['weight_std']); neg_iu.append(int((wi<0).sum()))
        wl=np.asarray(d['arms']['LR_FULL']['model']['weight_std']); neg_lr.append(int((wl<0).sum()))
        tr,te=sf!=o,sf==o; mu0,sd=X[tr].mean(0),X[tr].std(0); Ztr,Zte=(X[tr]-mu0)/sd,(X[te]-mu0)/sd
        S=np.cov(Ztr.T); mu=Ztr[y[tr]==1].mean(0)-Ztr[y[tr]==0].mean(0)
        ev,V=eigh(S); ev,V=ev[::-1],V[:,::-1]
        for k in K:
            w=V[:,:k]@((V[:,:k].T@mu)/ev[:k]); sc[k][te]=Zte@w
            if k==2: neg_or2.append(int((w<0).sum()))
    print(bank,'m',m,'| negative weights: IU-PCR',neg_iu,' LR',neg_lr,' oracle-PCR(k=2)',neg_or2)
    print('   oracle PCR AUC by k:',{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
