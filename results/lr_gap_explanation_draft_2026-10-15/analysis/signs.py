"""Per-cluster sign recovery: oracle w_g = (S_g+.1I)^-1 mu_g vs label-free (S_g+.1I)^-1 mu_hat_g.
mu_hat from 2nd-order Cov(z_i, s_-g) and from 3rd-order E[z_i (s_-g^2-1)]."""
import json, sys
import numpy as np
from numpy.linalg import solve
sys.path.insert(0,'results/lr_gap_explanation_draft_2026-10-15/analysis')
from decomp import ROOT
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
    R={'2nd':[0,0,0,0],'3rd':[0,0,0,0]}  # [neg found, neg total, all agree, all total]
    for o in range(5):
        L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_{o}/MODELS.json'))['selector_models']['LSML_CONT']
        G=np.asarray(L['diagnostic']['group']); wl=np.asarray(L['weight']); tr=sf!=o
        sd=X[tr].std(0); Z=(X[tr]-X[tr].mean(0))/sd; ws=wl*sd
        for g in np.unique(G):
            i=np.flatnonzero(G==g)
            if len(i)<2: continue
            Sg=np.cov(Z[:,i].T)+.1*np.eye(len(i))
            mu=Z[y[tr]==1][:,i].mean(0)-Z[y[tr]==0][:,i].mean(0); wo=solve(Sg,mu)
            s=Z[:,np.setdiff1d(np.arange(Z.shape[1]),i)]@ws[np.setdiff1d(np.arange(Z.shape[1]),i)]; s=(s-s.mean())/s.std()
            for name,mh in (('2nd',Z[:,i].T@s/len(s)),('3rd',(Z[:,i]*(s**2-1)[:,None]).mean(0))):
                if mh@mu<0: mh=-mh  # give label-free its best global sign per cluster
                wh=solve(Sg,mh); neg=wo<0
                R[name][0]+=int((neg&(wh<0)).sum()); R[name][1]+=int(neg.sum())
                R[name][2]+=int((np.sign(wh)==np.sign(wo)).sum()); R[name][3]+=len(i)
    for k,v in R.items():
        print(bank,k,'oracle-negative features found: %d/%d (%.0f%%) | sign agreement all features: %.0f%%'%(v[0],v[1],100*v[0]/max(v[1],1),100*v[2]/v[3]))
