"""Positive one-factor temporal fusion; unlabelled fits accept covariances only."""
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
BINS=16

def seed_for(uid,step):
    return int.from_bytes(hashlib.sha256(f'{uid}:time-shuffle:{step}'.encode()).digest()[:8],'little')

def regions(values,bins=BINS):
    v=np.asarray(values,dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('invalid token signal')
    edges=np.linspace(0,len(v),bins+1);whole=np.floor(edges).astype(int)
    integral=np.r_[0.,np.cumsum(v)][whole]+(edges-whole)*v[np.minimum(whole,len(v)-1)]
    return np.diff(integral)/(len(v)/bins)

def first_token_region_weight(length,bins=BINS):
    edges=np.linspace(0,length,bins+1)
    return np.maximum(0.,np.minimum(edges[1:],1.)-edges[:-1])/(length/bins)

def top_mean(v):
    n=min(10,len(v));return float(np.partition(v,len(v)-n)[-n:].mean())

def make_profiles(logits,spans,uid):
    ell=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(ell).all():raise ValueError('nonfinite reconstructed logits')
    raw=[];shuffled=[];top=[];contiguous=[];drop=[];first=[];first_weight=[];lengths=[]
    for j,(start,end) in enumerate(spans):
        v=ell[int(start):int(end)];raw.append(regions(v))
        shuffled.append(regions(np.random.default_rng(seed_for(uid,j)).permutation(v)))
        top.append(top_mean(v));n=min(10,len(v));cum=np.r_[0.,np.cumsum(v)]
        contiguous.append(float(np.max((cum[n:]-cum[:-n])/n)))
        drop.append(top_mean(v[1:]) if len(v)>1 else top[-1])
        first.append(v[0]);first_weight.append(first_token_region_weight(len(v)));lengths.append(len(v))
    raw=np.asarray(raw);shuffled=np.asarray(shuffled);scale=float(ell.std());mean=float(ell.mean())
    z=(raw-mean)/scale if scale>0 else np.zeros_like(raw)
    zs=(shuffled-mean)/scale if scale>0 else np.zeros_like(raw)
    def cov(x):
        x=x-x.mean(axis=0)
        return x.T@x/(len(x)-1) if len(x)>1 else np.zeros((BINS,BINS))
    return dict(raw=raw,shuffled=shuffled,normalized=z,covariance=cov(z),shuffled_covariance=cov(zs),
                top=np.asarray(top),contiguous=np.asarray(contiguous),drop_top=np.asarray(drop),
                first=np.asarray(first),first_weight=np.asarray(first_weight),lengths=np.asarray(lengths),
                token_mean=np.asarray(mean),token_scale=np.asarray(scale))

def objective(theta,covariance):
    p=len(covariance);a=theta[:p];logd=theta[p:];invd=np.exp(-logd);v=a*invd;den=1+a@v
    inverse=np.diag(invd)-np.outer(v,v)/den
    value=.5*(logd.sum()+np.log(den)+np.sum(inverse*covariance.T))
    g=.5*(inverse-inverse@covariance@inverse)
    cv=covariance@v
    diagonal=.5*(1-a*v/den-(np.diag(covariance)*invd-2*v*cv/den+a*v*(v@cv)/den**2))
    return float(value),np.r_[2*g@a,diagonal]

def fit_covariance(covariance):
    c=np.asarray(covariance,dtype=np.float64)
    if c.shape!=(BINS,BINS) or not np.isfinite(c).all():raise ValueError('invalid covariance')
    c=(c+c.T)/2
    if np.linalg.eigvalsh(c)[0]<-1e-8*max(1.,float(np.trace(c))):raise ValueError('non-PSD covariance')
    average=float(np.trace(c)/BINS);uniform=np.full(BINS,1/BINS)
    if average<=0:return uniform,dict(reason='ZERO_COVARIANCE',converged=True,starts=[])
    floor=max(1e-8,1e-3*average);ev,vec=np.linalg.eigh(c)
    starts=[np.full(BINS,.5*np.sqrt(average)),np.sqrt(max(ev[-1]-ev[:-1].mean(),.01*average))*np.abs(vec[:,-1])]
    fits=[];details=[]
    for a in starts:
        d=np.maximum(np.diag(c)-a*a,floor)
        r=minimize(objective,np.r_[a,np.log(d)],args=(c,),jac=True,method='L-BFGS-B',
                   bounds=[(0.,None)]*BINS+[(np.log(floor),None)]*BINS,
                   options=dict(maxiter=1000,ftol=1e-10,gtol=1e-6))
        value,g=objective(r.x,c)
        if not np.isfinite(value) or not np.isfinite(r.x).all():raise FloatingPointError('nonfinite factor fit')
        fits.append(r);details.append(dict(objective=value,converged=bool(r.success),iterations=int(r.nit),
                                          message=str(r.message),gradient_max=float(np.max(np.abs(g)))))
    selected=int(np.argmin([d['objective'] for d in details]));r=fits[selected]
    a=r.x[:BINS];d=np.exp(r.x[BINS:]);w=a/d;reason='FACTOR'
    if w.sum()<=1e-12:w=uniform;reason='NO_COMMON_FACTOR'
    else:w/=w.sum()
    sigma=np.outer(a,a)+np.diag(d)
    return w,dict(reason=reason,converged=bool(r.success),selected_start=selected,starts=details,
                  loadings=a.tolist(),noise=d.tolist(),noise_floor=floor,
                  residual_relative=float(np.linalg.norm(c-sigma)/max(np.linalg.norm(c),1e-30)),
                  condition=float(np.linalg.cond(sigma)))

def hierarchical(covariance,shared,steps):
    alpha=(steps-1)/(steps-1+BINS)
    return alpha*covariance+(1-alpha)*shared,alpha

def drop_first(raw_score,weights,profile):
    mass=profile['first_weight']@weights;single=profile['lengths']==1;den=1-mass
    valid=single|(den>1e-12);score=np.full(len(mass),np.nan);score[single]=raw_score[single]
    use=~single&valid;score[use]=(raw_score[use]-mass[use]*profile['first'][use])/den[use]
    return score,dict(single_token_retained=int(single.sum()),zero_remaining_mass=int((~valid).sum()))

def supervised_objective(theta,x,y,sample_weight):
    w,b=theta[:-1],theta[-1];ell=x@w+b
    value=np.sum(sample_weight*(np.logaddexp(0.,ell)-y*ell))+.005*(w@w)
    residual=sample_weight*(expit(ell)-y)
    return float(value),np.r_[x.T@residual+.01*w,residual.sum()]

def fit_supervised(x,y,sample_weight):
    if not np.isfinite(x).all() or set(np.unique(y))!={0,1}:raise ValueError('invalid data or absent class')
    sample_weight=sample_weight.copy()
    for value in (0,1):sample_weight[y==value]*=.5/sample_weight[y==value].sum()
    r=minimize(supervised_objective,np.r_[np.full(BINS,.01),0.],args=(x,y,sample_weight),jac=True,
               method='L-BFGS-B',bounds=[(0.,None)]*BINS+[(None,None)],
               options=dict(maxiter=1000,ftol=1e-10,gtol=1e-6))
    if not np.isfinite(r.fun) or not np.isfinite(r.x).all():raise FloatingPointError('nonfinite supervised fit')
    w=r.x[:-1];reason='SUPERVISED'
    if w.sum()<=1e-12:w=np.full(BINS,1/BINS);reason='ZERO_SUPERVISED_COEFFICIENTS'
    else:w=w/w.sum()
    return w,dict(reason=reason,converged=bool(r.success),iterations=int(r.nit),objective=float(r.fun),
                  message=str(r.message),coefficients=r.x[:-1].tolist(),intercept=float(r.x[-1]))
