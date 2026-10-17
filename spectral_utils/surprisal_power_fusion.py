"""Answer-local IU-PCR over raw surprisal powers, including the chosen token."""
import time
import numpy as np
from .direct_probability_fusion import logprob_matrix, zscore_columns, _orient
from .direct_probability_fusion_v2 import selected_surprisal, TAIL_TOLERANCE
from .laplacian_upcr import IU_FIT_DEFAULTS
from .upcr import upcr_fit

METHODS=tuple(f'd{d}__{m}' for d in (1,2,3) for m in ('equal','iu'))


def representation(logprobs, chosen, degree):
    if degree not in (1,2,3):raise ValueError('degree must be 1,2,3')
    lp=logprob_matrix({'logprobs':logprobs},k=15)
    if (lp>TAIL_TOLERANCE).any():raise ValueError('log probability exceeds zero')
    S=np.column_stack((np.maximum(-lp,0.),selected_surprisal(chosen,len(lp))))
    X=np.column_stack([S**n for n in range(1,degree+1)])
    if not np.isfinite(X).all():raise ValueError('nonfinite power input')
    return X


def fit_all(logprobs,chosen,entropy):
    entropy=np.asarray(entropy,float)
    if entropy.shape!=(len(logprobs),) or not np.isfinite(entropy).all():raise ValueError('entropy alignment')
    fits,failures,seconds={},{},{}
    for degree in (1,2,3):
        X=representation(logprobs,chosen,degree)
        Z,keep,mean,scale=zscore_columns(X)
        for solver in ('equal','iu'):
            name=f'd{degree}__{solver}';start=time.perf_counter()
            try:
                if len(X)<3 or Z.shape[1]<3:raise ValueError('fewer than three tokens or varying inputs')
                if solver=='equal':w=np.full(Z.shape[1],1/Z.shape[1])
                else:
                    fit=upcr_fit(Z.T,**dict(IU_FIT_DEFAULTS))
                    if fit.abstained or fit.used_simple_average:raise ValueError('unexpected IU fallback/abstention')
                    w=fit.w
                score=Z@w;flipped=False;corr=None
                if solver=='iu':
                    score,flipped,corr=_orient(score,entropy)
                    w=w*(-1 if flipped else 1)
                weights=np.zeros(X.shape[1]);weights[keep]=w
                effective=np.zeros_like(weights);effective[keep]=w/scale[keep]
                intercept=-float(mean@effective)
                np.testing.assert_allclose(X@effective+intercept,score,atol=1e-8,rtol=1e-8)
                if not np.isfinite(score).all() or not np.isfinite(weights).all():raise ValueError('nonfinite fusion')
                fits[name]=dict(score=score,weights=weights,effective=effective,intercept=intercept,
                    diagnostics=dict(active_columns=int(keep.sum()),orientation_flipped=bool(flipped),
                        anchor_correlation=float(corr) if corr is not None and np.isfinite(corr) else None))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError,AssertionError) as e:
                failures[name]=f'{type(e).__name__}: {e}'
            seconds[name]=time.perf_counter()-start
    return fits,failures,seconds
