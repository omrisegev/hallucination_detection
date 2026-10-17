"""Does partialling out label-free nuisance covariates (step position, step length) raise the label's share of
cross-cluster covariance, and make the label-free mu usable in the weak directions?"""
import json, sys
import numpy as np
from numpy.linalg import eigh, solve
sys.path.insert(0,'results/lr_gap_explanation_draft_2026-10-15/analysis')
from decomp import within, ROOT
from wsml import cross_rank1
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off)); n=np.diff(off); aid=np.repeat(np.arange(len(n)),n)
pos=np.arange(len(y))-off[aid]; rel=pos/np.maximum(n[aid]-1,1); tl=np.log1p(p['token_lengths'].astype(float))
cnt=np.bincount(aid); tl=tl-(np.bincount(aid,tl)/cnt)[aid]
Q=np.column_stack([rel,rel**2,(pos==0),(pos==n[aid]-1),tl,tl**2]).astype(float)
Q=Q-(np.vstack([np.bincount(aid,Q[:,j])/cnt for j in range(Q.shape[1])]).T)[aid]  # within-answer centred
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
    for mode in ('raw','partialled'):
        st=[]; sc={'VOTER_muhat':np.full(len(y),np.nan),'COND_muhat':np.full(len(y),np.nan),'ORACLE_COND':np.full(len(y),np.nan)}
        for o in range(5):
            L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
            G=np.asarray(L['diagnostic']['group']); tr,te=sf!=o,sf==o
            Xm=X.copy()
            if mode=='partialled':
                B=solve(Q[tr].T@Q[tr]+1e-6*np.eye(Q.shape[1]),Q[tr].T@X[tr]); Xm=X-Q@B
            mu0,sd=Xm[tr].mean(0),Xm[tr].std(0); Ztr,Zte=(Xm[tr]-mu0)/sd,(Xm[te]-mu0)/sd
            S=np.cov(Ztr.T); mu=Ztr[y[tr]==1].mean(0)-Ztr[y[tr]==0].mean(0)
            mh=cross_rank1(S,G); mh=mh if mh@mu>0 else -mh
            pi=y[tr].mean(); mask=G[:,None]!=G[None,:]; lab=pi*(1-pi)*np.outer(mu,mu)
            ev,V=eigh(S); lo=V[:,:len(ev)//2]
            st.append([np.sum((lab*mask)**2)/np.sum((S*mask)**2), cos(lo.T@mh,lo.T@mu)])
            sc['VOTER_muhat'][te]=Zte@mh; sc['COND_muhat'][te]=Zte@solve(S+1.0*np.eye(len(mu)),mh)
            sc['ORACLE_COND'][te]=Zte@solve(S+0.1*np.eye(len(mu)),mu)
        s=np.mean(st,0)
        print(bank,mode,'label share of cross-cluster cov=%.3f'%s[0],'cos weak dirs=%.3f'%s[1],
              {k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
