"""Shared token-logit/Top10 readout and step-label training derivatives.

This module does not choose the supervision scope or split contract. It never
replaces Top10 of fused token scores with fusion of step-averaged features.
"""
import numpy as np
from scipy.special import expit,logsumexp


def top10_value_gradient(x,spans,w,b):
    """Step logits and their derivatives w.r.t. [w,b].

    Ties select earlier tokens for a deterministic subgradient. Shared boundary
    tokens remain in each original step span; no annotation boundaries change.
    """
    x=np.asarray(x,float);spans=np.asarray(spans,int);w=np.asarray(w,float)
    if x.ndim!=2 or w.shape!=(x.shape[1],) or spans.ndim!=2 or spans.shape[1]!=2:
        raise ValueError('incompatible input shapes')
    if not all(np.isfinite(v).all() for v in (x,w,np.asarray(b))):
        raise ValueError('nonfinite model or features')
    token=x@w+b;value=[];gradient=[]
    for start,end in spans:
        if start<0 or end<=start or end>len(x):raise ValueError('invalid step span')
        selected=np.argsort(-token[start:end],kind='stable')[:min(10,end-start)]+start
        value.append(token[selected].mean())
        gradient.append(np.r_[x[selected].mean(axis=0),1.])
    return np.asarray(value),np.asarray(gradient).reshape(len(spans),len(w)+1)


def step_supervised_objective(theta,x,spans,*,first_error=None,labels=None,weights=None,ridge=0.):
    """Supervise only aggregated steps, not individual token labels.

    PB uses one categorical first-error target, without declaring later steps
    incorrect. PRMB uses explicitly known binary step labels and supplied
    training-only weights. The caller owns splitting and class balancing.
    """
    theta=np.asarray(theta,float)
    if (first_error is None)==(labels is None):raise ValueError('choose exactly one step-label task')
    s,jac=top10_value_gradient(x,spans,theta[:-1],theta[-1])
    if first_error is not None:
        t=int(first_error)
        if not 0<=t<len(s):raise ValueError('no valid first error for the location loss')
        normalizer=logsumexp(s);residual=np.exp(s-normalizer);residual[t]-=1
        loss=normalizer-s[t]
    else:
        y=np.asarray(labels);weight=np.asarray(weights,float)
        if y.shape!=s.shape or weight.shape!=s.shape or np.any(weight<0) or not np.isfinite(weight).all():
            raise ValueError('invalid step-label weights')
        known=(y==0)|(y==1)
        if not np.any(known) or np.any(weight[~known]!=0):raise ValueError('unknown labels must have zero weight')
        loss=np.sum(weight[known]*(np.logaddexp(0,s[known])-y[known]*s[known]))
        residual=np.zeros_like(s);residual[known]=weight[known]*(expit(s[known])-y[known])
    penalty=np.r_[theta[:-1],0.]
    return float(loss+.5*ridge*(penalty@penalty)),jac.T@residual+ridge*penalty
