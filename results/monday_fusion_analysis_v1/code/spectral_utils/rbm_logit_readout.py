"""Read saved RBM parameters; no fitting or inverse of rounded posteriors."""
import numpy as np
from scipy.special import expit
from .direct_probability_fusion import step_top_mean
from .rbm_data_diagnostics import first_near_max


def saved_scores(z, w, b, orientation, spans):
    if orientation not in (-1,1):raise ValueError('orientation must be saved +/-1')
    ell=float(b)+np.asarray(z,float)@np.asarray(w,float)
    if not np.isfinite(ell).all():raise ValueError('nonfinite logit; no clipping/fallback')
    q=expit(ell)
    posterior=q if orientation==1 else 1-q
    logit=orientation*ell
    np.testing.assert_allclose(expit(logit),posterior,atol=1e-15,rtol=1e-14)
    post_step=step_top_mean(posterior,spans[:,0],spans[:,1],count=10)
    logit_step=step_top_mean(logit,spans[:,0],spans[:,1],count=10)
    # Count losses of token order through finite-precision sigmoid, separately
    # from aggregation and the near-max selection rule.
    order=np.argsort(logit,kind='stable');dl=np.diff(logit[order]);dq=np.diff(posterior[order])
    if (dq < -1e-15).any():raise ValueError('non-monotonic oriented sigmoid')
    collapsed=int(((dl>0)&(dq==0)).sum())
    return dict(posterior=post_step,posterior_near=first_near_max(post_step),
                logit=logit_step,logit_near=first_near_max(logit_step)),dict(
                    collapsed_token_pairs=collapsed,posterior_zero_tokens=int((posterior==0).sum()),
                    posterior_one_tokens=int((posterior==1).sum()),
                    logit_min=float(logit.min()),logit_max=float(logit.max()))
