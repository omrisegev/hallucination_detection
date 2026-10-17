"""Provided-token disagreement and answer-local fusion; no correctness labels."""
import hashlib
import numpy as np
from .predictor_subset_fusion import fast_iu,top10
from .step_evidence_fusion import standardize

METHODS=('digit_standalone','digit025','digit1','presence','rate','permuted',
         'tcn_digit_sum','tcn_digit_matched','bank5_equal','bank5_iu','bank6_equal','bank6_iu')


def digit_streams(provided,top1,digit_ids,uid):
    g=np.asarray(provided);p=np.asarray(top1)
    if g.ndim!=1 or p.shape!=g.shape:raise ValueError('Misaligned token IDs')
    digits=np.asarray(sorted(digit_ids),int)
    if len(digits)!=10:raise ValueError('Expected ten verified ASCII digit IDs')
    opportunity=np.isin(g,digits)
    d=opportunity&np.isin(p,digits)&(g!=p)
    seed=int.from_bytes(hashlib.sha256(('digit-null/'+uid).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);null=np.zeros(len(g),float)
    null[opportunity]=rng.permutation(d[opportunity].astype(float))
    return d.astype(float),opportunity.astype(float),null


def summarize(d,p,null,spans):
    values=top10(np.column_stack((d,p,null)),spans)
    count=np.array([d[a:b].sum() for a,b in spans])
    opp=np.array([p[a:b].sum() for a,b in spans])
    rate=np.divide(count,opp,out=np.zeros_like(count),where=opp>0)
    return np.column_stack((values,rate)),count,opp


def add_correction(base,aux,amplitude=.25):
    b=np.asarray(base,float)
    return b+amplitude*b.std()*standardize(aux)[0]


def direct_bank(bank,digit,spans,base):
    x=np.asarray(bank,float)
    if x.shape!=(len(digit),5):raise ValueError('Expected current five-stream bank')
    z,sd=standardize(np.column_stack((x,digit)))
    summaries=top10(z,spans);out=[];diagnostics=[]
    for m in (5,6):
        live=np.flatnonzero(sd[:m]>1e-12);C=z[:,live].T@z[:,live]/len(z)
        equal=np.zeros(m);equal[live]=1/max(len(live),1)
        reason=None;w=equal.copy()
        if len(live)<3:reason='fewer_than_three_live_views'
        elif np.linalg.eigvalsh(C)[-2]<=1e-10:reason='rank_below_two'
        else:
            ww,_=fast_iu(C)
            if ww@C@np.ones(len(live))<0:ww=-ww
            w[live]=ww
        signals=summaries[:,:m]@np.column_stack((equal,w))
        normalized,_=standardize(signals)
        result=base.mean()+base.std()*normalized
        out.extend([result[:,0],result[:,1]])
        eig=np.linalg.eigvalsh(C);pr=float(eig.sum()**2/max(np.square(eig).sum(),1e-12))
        diagnostics.append(dict(native=reason is None,reason=reason,live=live.tolist(),
            weights=w.tolist(),participation_ratio=pr,covariance=C.tolist()))
    return np.column_stack(out),diagnostics


def score_auxiliary(base,tcn,aux,bank_scores):
    b=np.asarray(base,float);a=np.asarray(aux,float)
    dz=standardize(a[:,0])[0];tz=standardize(tcn-b)[0]
    v=[a[:,0],add_correction(b,a[:,0]),add_correction(b,a[:,0],1.),
       add_correction(b,a[:,1]),add_correction(b,a[:,3]),add_correction(b,a[:,2]),
       tcn+.25*b.std()*dz,add_correction(b,tz+dz)]
    return np.column_stack(v+[bank_scores[:,j] for j in range(4)])
