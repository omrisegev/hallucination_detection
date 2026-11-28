import numpy as np, sys
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from load import pb, prmb
def az(x,off):
    o=x.copy()
    for a,b in zip(off[:-1],off[1:]):
        s=x[a:b].std(); o[a:b]=(x[a:b]-x[a:b].mean())/s if s>1e-12 else 0
    return o
D={'PB':pb(),'PRMB':prmb()}
FE=['chosen_surprisal','realized_z','realized_drv','q15_H1','logprob_margin','ct7_chosen_std_excess']
for k,d in D.items():
    keep=d['keep']; off=d['off']; n=np.diff(off)
    print(k,'answers',len(n),'steps',len(keep),'kept',keep.sum(),'err rate kept',d['err'][keep].mean().round(3),'mean steps',n.mean().round(1))
    for f in FE:
        x=d['V'][:,d['names'].index(f)]
        print(f'  {f:24s} pooled raw {roc_auc_score(d["err"][keep],x[keep]):.3f}  pooled answer-z {roc_auc_score(d["err"][keep],az(x,off)[keep]):.3f}')
