"""Label-free, answer-local fusion of context and step-shape evidence."""
import hashlib
import numpy as np
from scipy.stats import rankdata
from .predictor_subset_fusion import fast_iu

METHODS = ('context','end','sustained','equal','iu','equal_context_end',
           'equal_context_sustained','equal_shuffled','iu_shuffled')


def standardize(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite evidence')
    sd = x.std(axis=0)
    return (x-x.mean(axis=0))/np.where(sd>1e-12,sd,1.), sd


def shape_views(streams, spans, uid):
    """Natural-unit readouts; labels are neither accepted nor accessed."""
    x = np.asarray(streams,float)
    if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all():
        raise ValueError('Expected five finite streams')
    spans=np.asarray(spans,int)
    if spans.ndim!=2 or spans.shape[1]!=2 or np.any(spans[:,0]<0) or np.any(spans[:,1]>len(x)) or np.any(spans[:,1]<=spans[:,0]):
        raise ValueError('Invalid spans')
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('step-evidence/'+uid).encode()).digest()[:8],'little'))
    base=[];real=[];shuffled=[]
    for a,b in spans:
        v=x[a:b];n=len(v);k=min(10,n)
        base.append(np.partition(v,n-k,axis=0)[-k:].mean())
        for out,y in ((real,v),(shuffled,v[rng.permutation(n)])):
            c=np.vstack((np.zeros((1,5)),np.cumsum(y,axis=0)))
            sustained=((c[k:]-c[:-k])/k).max(axis=0).mean()
            out.append([y[-min(4,n):].mean(),sustained])
    return np.array(base),np.array(real),np.array(shuffled)


def weights(z):
    c=z.T@z/len(z)
    reason=None
    if len(z)<3:reason='fewer_than_three_steps'
    elif np.any(z.std(0)<=1e-12):reason='constant_view'
    elif np.linalg.eigvalsh(c)[-2]<=1e-10:reason='rank_below_two'
    if reason:
        return np.ones(3)/3,dict(native=False,reason=reason,weights=[1/3]*3,correlation=c.tolist())
    w,d=fast_iu(c)
    flipped=bool(w@c@np.ones(3)<0)
    if flipped:w=-w
    ranks=rankdata(z,axis=0)
    rz,_=standardize(ranks);spearman=rz.T@rz/len(z)
    d.update(native=True,reason=None,weights=w.tolist(),flipped=flipped,
             correlation=c.tolist(),spearman=spearman.tolist(),
             max_abs_spearman=float(np.abs(spearman[np.triu_indices(3,1)]).max()))
    return w,d


def score_evidence(base,context_score,real_shape,shuffled_shape,amplitude=.25):
    b=np.asarray(base,float);t=np.asarray(context_score,float)
    if b.ndim!=1 or len(b)==0 or t.shape!=b.shape or np.asarray(real_shape).shape!=(len(b),2) or np.asarray(shuffled_shape).shape!=(len(b),2):
        raise ValueError('Misaligned evidence')
    # t-b is a positive scalar multiple of the frozen standardized TCN auxiliary.
    real=np.column_stack((t-b,real_shape));shuffled=np.column_stack((t-b,shuffled_shape))
    z,sd=standardize(real);zs,_=standardize(shuffled)
    w,d=weights(z);ws,ds=weights(zs)
    aux=np.column_stack((z,z.mean(1),z@w,z[:,:2].mean(1),z[:,[0,2]].mean(1),
                         zs.mean(1),zs@ws))
    az,asd=standardize(aux)
    scores=b[:,None]+amplitude*b.std()*az
    return scores,dict(real=d,shuffled=ds,view_sd=sd.tolist(),constant_auxiliary=(asd<=1e-12).tolist())
