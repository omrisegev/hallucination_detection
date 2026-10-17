"""Active-set numerical follow-up for the unchanged nonnegative kernel QP.

Not used by the frozen Step412 fits. Acceptance checks the original KKT
criterion and objective; no ridge, positive mass floor or tolerance relaxation.
"""
import numpy as np


def polish_kernel_mass(kernel,initial,*,tolerance=1e-8,max_iterations=1000):
    k=np.asarray(kernel,float);x=np.maximum(np.asarray(initial,float),0.).copy()
    if k.shape!=(len(x),len(x)) or not np.isfinite(k).all() or np.any(np.diag(k)<=0):
        raise ValueError('POSITIVE_DIAGONAL_FINITE_KERNEL_REQUIRED')
    active=x>0
    def objective(v):return float(.5*v@k@v-v.sum())
    trace=[objective(x)]
    for iteration in range(max_iterations):
        gradient=k@x-1;projected=np.where(x>0,gradient,np.minimum(gradient,0))
        residual=float(np.max(np.abs(projected)))
        if residual<=tolerance:
            return x,dict(status='PASS',iterations=iteration,projected_kkt=residual,
                objective_trace=trace,tolerance=tolerance)
        if not active.any():active[int(np.argmin(gradient))]=True
        ids=np.flatnonzero(active);candidate=np.zeros_like(x)
        candidate[ids]=np.linalg.lstsq(k[np.ix_(ids,ids)],np.ones(len(ids)),rcond=1e-12)[0]
        negative=active&(candidate<0)
        if negative.any():
            direction=candidate-x
            blockers=np.flatnonzero(negative)
            ratios=-x[blockers]/direction[blockers]
            blocker=int(blockers[np.argmin(ratios)]);step=float(np.min(ratios))
            x=np.maximum(x+step*direction,0.);x[blocker]=0.;active[blocker]=False
        else:
            x=candidate
            gradient=k@x-1;inactive=np.flatnonzero(~active)
            if len(inactive) and gradient[inactive].min() < -tolerance:
                active[int(inactive[np.argmin(gradient[inactive])])]=True
        value=objective(x)
        if value>trace[-1]+1e-10*max(1.,abs(trace[-1])):raise ValueError('QP_POLISH_OBJECTIVE_INCREASE')
        trace.append(value)
    raise ValueError('QP_POLISH_DID_NOT_REACH_KKT')
