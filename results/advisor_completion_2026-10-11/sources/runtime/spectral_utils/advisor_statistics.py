"""Within-answer metrics and source-group diagnostic summaries, without sign repair."""
import numpy as np
from scipy.stats import rankdata
from scipy.special import expit

def answer_auc(y,score,offsets,bins=None):
    X=np.asarray(score,float);one=X.ndim==1
    if one:X=X[:,None]
    out=np.full((len(offsets)-1,X.shape[1]),np.nan)
    pairs=np.zeros(len(out),int)
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        blocks=[np.arange(a,b)] if bins is None else [np.arange(a,b)[bins[a:b]==c] for c in np.unique(bins[a:b])]
        numerator=np.zeros(X.shape[1]);den=0
        for rows in blocks:
            yy=y[rows].astype(bool);p=int(yy.sum());v=len(yy)-p
            if not p or not v:continue
            r=rankdata(X[rows],axis=0);numerator+=r[yy].sum(0)-p*(p+1)/2;den+=p*v
        if den:out[i]=numerator/den;pairs[i]=den
    return (out[:,0] if one else out),pairs

def source_bootstrap(values,groups,mask,weights):
    x=np.asarray(values,float)
    if x.ndim==1:x=x[:,None]
    take=mask[:,None]&np.isfinite(x)
    inv=np.unique(groups,return_inverse=True)[1];G=weights.shape[1]
    sums=np.zeros((G,x.shape[1]));counts=sums.copy()
    np.add.at(sums,inv,np.where(take,x,0));np.add.at(counts,inv,take)
    den=weights@counts
    return np.divide(weights@sums,den,out=np.full_like(den,np.nan),where=den>0)

def posterior_from_mom(votes,estimate):
    pos=np.asarray(votes)>0;psi=np.asarray(estimate['psi']);eta=np.asarray(estimate['eta']);pi=estimate['prevalence']
    if not 0<pi<1 or not np.all((psi>0)&(psi<1)&(eta>0)&(eta<1)):raise ValueError('Invalid identifiable MoM parameters')
    odds=np.log(pi)-np.log1p(-pi)+np.sum(np.where(pos,np.log(psi)-np.log1p(-eta),np.log1p(-psi)-np.log(eta)),axis=1)
    return expit(odds)
