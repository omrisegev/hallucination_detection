"""Source-excluded position/length moments; fitting has no correctness input."""
from collections import Counter
import numpy as np
from .context_training import METADATA_KEYS

BINS=16
CENTERS=(np.arange(BINS)+.5)/BINS
VAR_FLOOR=1e-16
METHODS=('profile_only','constant_profile','mean_detrended','location_scale_detrended')

def length_class(n):return int(np.searchsorted([256,512,1024],n,side='left'))

def answer_moments(innovation):
    x=np.asarray(innovation,float);n=len(x)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('finite vector required')
    if n<=1:return np.zeros((BINS,3))
    b=np.minimum(((np.arange(1,n)+.5)/n*BINS).astype(int),BINS-1)
    return np.column_stack([np.bincount(b,weights=w,minlength=BINS) for w in
        (np.ones(n-1),x[1:],x[1:]**2)])/(n-1)

def _profile(moment,backup=None):
    w,s,ss=moment.T;total=w.sum()
    if total<=0:raise ValueError('profile has no eligible history')
    mu0=float(s.sum()/total);sigma0=float(np.sqrt(max(ss.sum()/total-mu0**2,VAR_FLOOR)))
    good=w>0;mu=np.full(BINS,mu0);variance=np.full(BINS,sigma0**2)
    mu[good]=s[good]/w[good];variance[good]=np.maximum(ss[good]/w[good]-mu[good]**2,0.)
    missing=np.flatnonzero(~good)
    if len(missing):
        if backup is None:raise ValueError('cell profile lacks a position bin')
        mu[missing]=np.asarray(backup['mean'])[missing]
        variance[missing]=np.asarray(backup['std'])[missing]**2
    flat=variance<=VAR_FLOOR;variance[flat]=sigma0**2
    return dict(mean=mu.tolist(),std=np.sqrt(variance).tolist(),global_mean=mu0,global_std=sigma0,
                empty_bins=missing.tolist(),flat_bins=np.flatnonzero(flat).tolist())

def fit_profiles(metadata,moments,excluded,min_groups=20):
    if any(set(m)!=METADATA_KEYS for m in metadata):raise ValueError('undeclared metadata/labels')
    moments=np.asarray(moments,float)
    if moments.shape!=(len(metadata),BINS,3) or not np.isfinite(moments).all():raise ValueError('invalid moments')
    eligible=[i for i,m in enumerate(metadata) if m['fold'] not in excluded and m['tokens']>1]
    train_groups={metadata[i]['group_id'] for i in eligible}
    held_groups={m['group_id'] for m in metadata if m['fold'] in excluded}
    if train_groups&held_groups:raise ValueError('source group overlap')
    def aggregate(ids):
        counts=Counter(metadata[i]['group_id'] for i in ids)
        if not counts:raise ValueError('no training groups')
        return sum((moments[i]/counts[metadata[i]['group_id']] for i in ids))/len(counts),len(counts)
    profiles={};fallbacks=[]
    for cell in sorted({m['cell'] for m in metadata}):
        ids=[i for i in eligible if metadata[i]['cell']==cell]
        pooled,ng=aggregate(ids);cell_profile=_profile(pooled)
        for length in range(4):
            subset=[i for i in ids if length_class(metadata[i]['tokens'])==length]
            groups={metadata[i]['group_id'] for i in subset};key=cell+'|'+str(length)
            if len(groups)<min_groups:
                profiles[key]=dict(cell_profile,fit_scope='cell_fallback',stratum_groups=len(groups),fit_groups=ng)
                fallbacks.append(key)
            else:
                mm,ns=aggregate(subset)
                profiles[key]=dict(_profile(mm,cell_profile),fit_scope='cell_length',stratum_groups=ns,fit_groups=ns)
    return dict(excluded_folds=list(excluded),training_groups=sorted(train_groups),profiles=profiles,fallbacks=fallbacks)

def transform(innovation,profile):
    x=np.asarray(innovation,float);p=(np.arange(len(x))+.5)/max(len(x),1)
    mu=np.interp(p,CENTERS,profile['mean']);sd=np.interp(p,CENTERS,profile['std'])
    mu0=profile['global_mean'];sd0=profile['global_std']
    if not np.isfinite(x).all() or np.any(sd<=0):raise ValueError('invalid innovation/profile')
    out=dict(profile_only=mu,constant_profile=np.full(len(x),mu0),mean_detrended=x-mu+mu0,
             location_scale_detrended=mu0+(x-mu)*sd0/sd)
    for a in out.values():
        if len(a):a[0]=0.
        if not np.isfinite(a).all():raise FloatingPointError('nonfinite profile score')
    return out
