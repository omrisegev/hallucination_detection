"""Two fixed RBM tests: remove distribution m3; fit raw degree-three rank powers."""
import time
import numpy as np
from scipy.special import expit
from .moment_rbm_fusion import representation as moments,fit_rbm
from .surprisal_power_fusion import representation as powers
from .direct_probability_fusion import zscore_columns,_orient

METHODS=('mom6_rbm','mom6_shared_rbm','mom5_rbm','power48_rbm','power48_initial')
RETAIN=(0,1,3,4,5)


def orient_posterior(q,anchor):
    oriented,flipped,corr=_orient(q,anchor)
    if flipped:oriented=1+oriented
    return oriented,dict(orientation=-1 if flipped else 1,
        anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None,
        score_sd=float(np.std(oriented)),collapsed=bool(np.std(oriented)<1e-3))


def banks(logprobs,chosen):
    X=moments(logprobs,chosen)
    return {'mom6':X,'mom5':X[:,RETAIN],'power48':powers(logprobs,chosen,3)}


def fit_all(logprobs,chosen,entropy):
    raw=banks(logprobs,chosen);norm={k:zscore_columns(x) for k,x in raw.items()}
    entropy=np.asarray(entropy,float)
    if entropy.shape!=(len(logprobs),) or not np.isfinite(entropy).all():raise ValueError('entropy alignment')
    anchor5=norm['mom5'][0].mean(axis=1)
    fits,failures,seconds={},{},{};cached={}
    for m in METHODS:
        started=time.perf_counter()
        bank='mom6' if m.startswith('mom6') else 'mom5' if m.startswith('mom5') else 'power48'
        X=raw[bank];Z,keep,mean,scale=norm[bank]
        try:
            if len(X)<3 or Z.shape[1]<3:raise ValueError('need three rows and varying columns')
            if m=='power48_initial':
                p=Z.shape[1];state=dict(a=np.zeros(p),w=np.full(p,2./p),b=np.asarray(0.))
                q=expit(Z@state['w']);diag={'trained':False}
            else:
                if bank not in cached:cached[bank]=fit_rbm(Z)
                q,state,extra=cached[bank];diag=dict(extra)
            anchor=Z.mean(axis=1) if m=='mom6_rbm' else entropy if bank=='power48' else anchor5
            score,orient=orient_posterior(q,anchor);diag.update(orient)
            diag.update(columns=np.flatnonzero(keep).tolist(),active_columns=int(keep.sum()),
                normalization_mean=mean.tolist(),normalization_scale=scale.tolist(),
                anchor='mean6' if m=='mom6_rbm' else 'saved token entropy' if bank=='power48' else 'mean5 without m3')
            if not np.isfinite(score).all():raise ValueError('nonfinite score')
            weights=np.zeros(X.shape[1]);weights[keep]=diag['orientation']*state['w']
            fits[m]=dict(score=score,state=state,weights=weights,diagnostics=diag)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:failures[m]=f'{type(e).__name__}: {e}'
        seconds[m]=time.perf_counter()-started
    return fits,failures,seconds
