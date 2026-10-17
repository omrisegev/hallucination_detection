"""Third-order cross-cluster moment as a label-free loading: t_i = E[z_i * (s_-g - mean)^2], s_-g = L-SML score w/o i's cluster.
For binary sparse y and symmetric nuisance independent of y, t ∝ mu (nuisance terms vanish at 3rd order)."""
import json, sys
import numpy as np
from numpy.linalg import eigh, solve
sys.path.insert(0,'results/lr_gap_explanation_draft_2026-10-15/analysis')
from decomp import within, ROOT
from wsml import cross_rank1
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
cos=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('pi =',round(y.mean(),3))
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float); m=X.shape[1]
    sc={k:np.full(len(y),np.nan) for k in ['VOTER_t','COND_t_lam1','COND_t_lam0.1','COND_mu2_lam1']}; st=[]
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); wl=np.asarray(L['weight']); tr,te=sf!=o,sf==o
        mu0,sd=X[tr].mean(0),X[tr].std(0); Ztr,Zte=(X[tr]-mu0)/sd,(X[te]-mu0)/sd
        S=np.cov(Ztr.T); mu=Ztr[y[tr]==1].mean(0)-Ztr[y[tr]==0].mean(0)
        ws=wl*sd; t=np.zeros(m)
        for g in np.unique(G):
            i=G==g; s=Ztr[:,~i]@ws[~i]; s=(s-s.mean())/s.std()
            t[i]=(Ztr[:,i]*((s**2-1))[:,None]).mean(0)
        if t@(Ztr.T@(Ztr@ws))<0: t=-t   # orient with L-SML (label-free)
        mh=cross_rank1(S,G); mh=mh if mh@mu>0 else -mh
        ev,V=eigh(S); hi,lo=V[:,m//2:],V[:,:m//2]
        st.append([cos(t,mu),cos(hi.T@t,hi.T@mu),cos(lo.T@t,lo.T@mu),cos(lo.T@mh,lo.T@mu)])
        sc['VOTER_t'][te]=Zte@t
        sc['COND_t_lam1'][te]=Zte@solve(S+np.eye(m),t); sc['COND_t_lam0.1'][te]=Zte@solve(S+.1*np.eye(m),t)
        sc['COND_mu2_lam1'][te]=Zte@solve(S+np.eye(m),mh)
    s=np.round(np.mean(st,0),3)
    print(bank,'cos(t,mu) all=%s strong=%s weak=%s | 2nd-order weak=%s'%tuple(s),{k:round(within(v,off,y),4) for k,v in sc.items()},flush=True)
