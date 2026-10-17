import json, sys
import numpy as np
from numpy.linalg import solve
from decomp import ROOT, within
p=np.load(f'{ROOT}/inputs/population.npz'); off=p['offsets'].astype(int); y=p['y'].astype(int)
sf=np.repeat(p['fold'].astype(int),np.diff(off))
lens=np.diff(off); aid=np.repeat(np.arange(len(lens)),lens)
def spik(s):
    # within-answer z, then mean excess kurtosis and mean max-z over answers with >=5 steps
    cnt=np.bincount(aid); m=np.bincount(aid,s)/cnt; d=s-m[aid]
    v=np.bincount(aid,d**2)/cnt; z=d/np.sqrt(np.maximum(v,1e-12))[aid]
    k=np.bincount(aid,z**4)/cnt-3; mx=np.maximum.reduceat(z,off[:-1])
    ok=cnt>=5; return round(float(k[ok].mean()),3), round(float(mx[ok].mean()),3)
rng=np.random.default_rng(0)
for bank in sys.argv[1:]:
    X=np.load(f'{ROOT}/inputs/bank_V_{bank}.npz')['V'].astype(float)
    L=json.load(open(f'{ROOT}/lr/checkpoints/{bank}/outer_0/MODELS.json'))['selector_models']['LSML_CONT']
    Z=(X-X.mean(0))/X.std(0); S=np.cov(Z.T); m=S.shape[0]
    mu=Z[y==1].mean(0)-Z[y==0].mean(0)
    cands={'LSML':Z@(np.asarray(L['weight'])*X.std(0)),'ORACLE_LDA':Z@solve(S+0.1*np.eye(m),mu),'EQ':Z.sum(1)}
    out={k:spik(v)+(round(within(v,off,y),3),) for k,v in cands.items()}
    # random whitened directions
    ev,V=np.linalg.eigh(S); W=V/np.sqrt(np.maximum(ev,1e-6))
    r=[spik(Z@(W@rng.standard_normal(m))) for _ in range(20)]
    out['RANDOM_WHITENED(kurt,maxz) mean']=tuple(np.round(np.mean(r,0),3))
    print(bank,'(kurt, maxz, AUC):',out,flush=True)
