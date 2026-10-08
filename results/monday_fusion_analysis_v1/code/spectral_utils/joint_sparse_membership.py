"""Joint sparse loading membership with exact-alias model coordinates.

The provisional fit minimizes half the upper-triangle squared covariance error
plus lambda*(||v||_1+||u||_1), subject to v_i^2+u_i^2<=C_ii. A feature with both
loadings zero belongs only to the diagonal-noise component. Final native fitting
and external evaluation are deliberately separate from this label-free solver.
"""
import hashlib
import numpy as np
from .joint_lsml import _initial_loadings, coassignment_from_labels


def exact_aliases(values):
    x=np.asarray(values,float);groups=[];buckets={}
    for i in range(x.shape[1]):
        column=np.ascontiguousarray(x[:,i]).copy();column[column==0]=0.
        digest=hashlib.sha256(column.tobytes()).digest()
        match=next((g for g in buckets.get(digest,[]) if np.array_equal(x[:,groups[g][0]],column)),None)
        if match is None:
            match=len(groups);groups.append([i]);buckets.setdefault(digest,[]).append(match)
        else:groups[match].append(i)
    return groups


def alias_coordinates(values,groups):
    x=np.asarray(values,float);columns=[]
    for ids in groups:
        block=x[:,ids]
        columns.append(block[:,0] if np.all(block==block[:,:1]) else block.mean(axis=1))
    return np.column_stack(columns)


def initial_loadings(cov,labels,*,start,seed):
    mask=coassignment_from_labels(labels);np.fill_diagonal(mask,0.)
    v,u=_initial_loadings(cov,mask,start=start,seed=seed,anchor_index=0)
    radius=np.sqrt(np.maximum(np.diag(cov),0));length=np.sqrt(v*v+u*u)
    factor=np.minimum(1.,radius/np.maximum(length,1e-15))
    return v*factor,u*factor,mask


def sparse_objective(cov,v,u,mask,penalty):
    left,right=np.triu_indices(len(v),1)
    residual=(cov-np.outer(v,v)-mask*np.outer(u,u))[left,right]
    return float(.5*(residual@residual)+penalty*(np.abs(v).sum()+np.abs(u).sum()))


def sparse_start(cov,labels,*,penalty,start,seed,max_sweeps=3000):
    cov=np.asarray(cov,float);v,u,mask=initial_loadings(cov,labels,start=start,seed=seed)
    diag=np.maximum(np.diag(cov),0);loss=sparse_objective(cov,v,u,mask,penalty)
    trace=[loss];stable=0;converged=False
    for sweep in range(max_sweeps):
        oldv=v.copy();oldu=u.copy()
        for vector,other,is_local in ((v,u,False),(u,v,True)):
            for i in range(len(v)):
                coefficient=vector.copy()
                if is_local:coefficient*=mask[i]
                coefficient[i]=0.
                residual=cov[i]-other[i]*other*(1. if is_local else mask[i])
                numerator=float(coefficient@residual);denominator=float(coefficient@coefficient)
                candidate=np.sign(numerator)*max(abs(numerator)-penalty,0)/denominator if denominator>1e-15 else 0.
                bound=np.sqrt(max(diag[i]-other[i]**2,0.));vector[i]=np.clip(candidate,-bound,bound)
        newloss=sparse_objective(cov,v,u,mask,penalty)
        if newloss>loss+1e-10*max(1.,abs(loss)):raise ValueError('SPARSE_OBJECTIVE_INCREASE')
        change=max(float(np.max(np.abs(v-oldv))),float(np.max(np.abs(u-oldu))))
        stable=stable+1 if abs(loss-newloss)<=1e-8*max(1.,abs(loss)) and change<=1e-7 else 0
        trace.append(newloss);loss=newloss
        if stable>=5:converged=True;break
    return dict(v=v,u=u,objective=loss,converged=converged,sweeps=sweep+1,
        support=np.flatnonzero((v!=0)|(u!=0)),objective_trace=trace)


def sparse_membership(cov,labels,*,penalty,seed,starts=5):
    fits=[sparse_start(cov,labels,penalty=penalty,start=s,seed=seed) for s in range(starts)]
    converged=[r for r in fits if r['converged']]
    if not converged:raise ValueError('NO_CONVERGED_SPARSE_START')
    # Conservative inclusion across all converged starts, not a chosen favorable
    # support from a local optimum. Final checked Joint still needs its own pass.
    active=np.unique(np.concatenate([r['support'] for r in converged]))
    return active,dict(penalty=penalty,converged_starts=len(converged),starts=fits,
        support_union=active,supports_agree=all(np.array_equal(converged[0]['support'],r['support']) for r in converged))


def source_grams(values,offsets,source_ids,folds):
    x=np.asarray(values,float);keys,inverse=np.unique(source_ids,return_inverse=True)
    gram=np.zeros((len(keys),x.shape[1],x.shape[1]));sizes=np.zeros(len(keys),int);sourcefold=np.full(len(keys),-1,int)
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        g=inverse[i]
        if sourcefold[g]>=0 and sourcefold[g]!=folds[i]:raise ValueError('SOURCE_FOLD_LEAKAGE')
        block=x[a:b];gram[g]+=block.T@block;sizes[g]+=b-a;sourcefold[g]=folds[i]
    return gram,sizes,sourcefold


def calibrate_penalty(grams,nrows,cov,labels,*,seed,draws=31):
    """95th percentile null score scale under independent per-source signs.

    Signs preserve each source's per-channel temporal structure. This Monte
    Carlo calibration is a declared scale heuristic, not a formal FWER result.
    """
    v,u,mask=initial_loadings(cov,labels,start=0,seed=seed)
    rng=np.random.default_rng(seed);maxima=[]
    for _ in range(draws):
        signs=rng.choice(np.array([-1.,1.]),size=grams.shape[:2])
        null=np.einsum('gi,gij,gj->ij',signs,grams,signs,optimize=False)/max(nrows-1,1)
        np.fill_diagonal(null,0.)
        maxima.append(float(max(np.max(np.abs(null@v)),np.max(np.abs((null*mask)@u)))))
    penalty=max(float(np.quantile(maxima,.95)),1e-8)
    return penalty,dict(seed=seed,draws=draws,quantile=.95,null_maxima=maxima,penalty=penalty,
        calibration='independent feature signs within each source block')
