"""Answer-local Shrinkage IU with borrowed covariance and coefficient GraphTV.

The marginal signal estimate rho and its two-dimensional subspace stay fixed
within the answer. Conditional solves are quadratic surrogates, not estimates
of conditional Cov(X,Y). No fitting function accepts target labels.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import eigh
from .upcr import upcr_fit_covariance
from .laplacian_upcr import IU_FIT_DEFAULTS
from .direct_probability_temporal import lw_alpha_memory_bounded
from .direct_probability_fusion import _orient, step_top_mean
from .answer_position_fusion import position_overlap
from .conditional_iu_graph import (build_token_graph, permute_token_graph, graph_edges,
                                   weighted_neighborhood_moments, solve_network_lasso)

BINS=16
ALPHA=.25
ETA=.1
EPS=1e-12
FAMILIES={
    'position':('baseline','pooled','position','position_scale_only','position_shuffled'),
    'graph_local':('baseline','pooled','sliding_window','graph_local','graph_local_shuffled'),
    'graph_tv':('baseline','position','position_scale_only','graph_tv','graph_tv_shuffled'),
}
PRIMARY={'position':('position','position_scale_only'),
         'graph_local':('graph_local','sliding_window'),
         'graph_tv':('graph_tv','position')}
LABELS={
    'top10':'Original answer-local RBM12',
    'baseline':'Answer-local Shrinkage IU',
    'pooled':'Shrinkage IU + pooled covariance prior',
    'position':'Shrinkage IU + answer-position covariance prior',
    'position_scale_only':'Position prior: baseline direction, scale change only',
    'position_shuffled':'Position prior with shuffled position assignments',
    'sliding_window':'Sliding-window IU + pooled covariance prior',
    'graph_local':'Local token-graph IU + pooled covariance prior',
    'graph_local_shuffled':'Local token-graph IU with shuffled graph',
    'graph_tv':'Position IU + graph penalty on feature weights',
    'graph_tv_shuffled':'Position IU + penalty on shuffled graph',
}


def regional_covariances(x, uid):
    """Center WITHIN each answer-region before averaging across answers.

Fractional intervals cover even short answers. Fractional coverage is not an
independent-sample count. Pooled control averages these same covariances.
"""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=12 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid cached twelve-feature answer')
    result={}
    for name,shuffled in [('real',False),('shuffle',True)]:
        weights=position_overlap(len(x),uid,shuffled)*BINS/len(x)
        means=weights.T@x
        second=np.einsum('tj,tk,tl->jkl',weights,x,x,optimize=True)
        covariance=second-np.einsum('jp,jq->jpq',means,means)
        result[name]=.5*(covariance+covariance.transpose(0,2,1))
    return result


def prepare_answer(x, uid, methods):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=12 or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('baseline needs finite twelve-column input and at least three tokens')
    active=x.std(axis=0)>1e-10
    if active.sum()<3:raise ValueError('fewer than three varying features')
    z=x[:,active]
    if not np.allclose(z.mean(0),0.,atol=1e-9,rtol=0):raise ValueError('cache is not answer-centered')
    if not np.allclose(z.std(0),1.,atol=1e-9,rtol=0):raise ValueError('cache is not answer-standardized')
    covariance=z.T@z/len(z)
    diagonal=np.diag(np.diag(covariance))
    beta=lw_alpha_memory_bounded(z,covariance,diagonal)
    cbase=(1-beta)*covariance+beta*diagonal
    native=upcr_fit_covariance(cbase,**IU_FIT_DEFAULTS)
    w=np.asarray(native.w)
    if not np.isfinite(w).all() or np.linalg.norm(w)==0:raise ValueError('invalid baseline IU fit')
    ev,u=eigh(cbase,subset_by_index=[len(w)-2,len(w)-1]);u=u[:,::-1]
    r=u.T@native.rho_hat
    g=u.T@cbase@u+EPS*np.eye(2)
    replay=u@np.linalg.solve(g,r)
    np.testing.assert_allclose(replay,w,atol=1e-10,rtol=1e-9)
    baseline_score,flipped,corr=_orient(z@w,x[:,1])
    sign=-1. if flipped else 1.
    answer=dict(x=x,z=z,active=active,p=z.shape[1],uid=uid,cbase=cbase,u=u,r=r,
                w=sign*w,sign=sign,gbase=g,score=baseline_score,beta=beta,
                anchor_correlation=float(corr) if np.isfinite(corr) else None,
                graph={},neighborhood={},graph_errors={})
    graph_needed=any(m.startswith('graph_') for m in methods)
    if graph_needed:
        try:
            adjacency=build_token_graph(z,mode='affinity',window=16,bandwidth_neighbor=8,bandwidth_floor=1e-8)
            shuffled,permutation=permute_token_graph(adjacency,uid)
            answer['graph'].update(real=adjacency,shuffle=shuffled)
        except Exception as exc:
            # A graph failure must not erase a successfully fitted IU baseline.
            for name in methods:
                if name.startswith('graph_'):answer['graph_errors'][name]=f'{type(exc).__name__}: {exc}'
    for name,key in [('sliding_window','uniform'),('graph_local','real'),('graph_local_shuffled','shuffle')]:
        if name not in methods:continue
        if name in answer['graph_errors']:continue
        try:
            graph=(build_token_graph(z,mode='uniform',window=16) if key=='uniform' else answer['graph'][key])
            answer['neighborhood'][name]=weighted_neighborhood_moments(z,graph)
        except Exception as exc:answer['graph_errors'][name]=f'{type(exc).__name__}: {exc}'
    return answer


def position_quadratic(answer, prior, alpha, shuffled=False, pooled=False):
    active=answer['active'];c=np.asarray(prior['shuffle' if shuffled else 'real'])
    c=c[:,active][:,:,active]
    u=answer['u'];projected=np.einsum('pi,jpq,qk->jik',u,c,u,optimize=True)
    overlap=position_overlap(len(answer['z']),answer['uid'],shuffled)
    projected=(np.broadcast_to(projected.mean(0),(len(overlap),2,2)) if pooled else
               np.einsum('tj,jab->tab',overlap,projected))
    g=(1-alpha)*(answer['gbase']-EPS*np.eye(2))+alpha*projected+EPS*np.eye(2)
    return .5*(g+g.transpose(0,2,1))


def direct_theta(g,r):
    if not np.isfinite(g).all() or np.linalg.eigvalsh(g).min()<=0:
        raise ValueError('conditional quadratic is not positive definite')
    rhs=np.broadcast_to(r,(len(g),2))
    return np.linalg.solve(g,rhs[...,None])[...,0]


def token_scores(answer, prior, method, *, alpha=ALPHA, eta=ETA):
    if not np.isfinite(alpha) or not 0<=alpha<1:raise ValueError('alpha must lie in [0,1)')
    if not np.isfinite(eta) or eta<0:raise ValueError('eta must be nonnegative')
    z,w0=answer['z'],answer['w'];n=len(z)
    if method=='baseline' or alpha==0:
        return answer['score'].copy(),dict(beta=answer['beta'],alpha=0.,baseline_exact=True,
                                           gain_mean=1.,direction_change_mean=0.),np.broadcast_to(w0,(n,len(w0)))
    if method in answer['graph_errors']:
        raise ValueError('graph preparation failed: '+answer['graph_errors'][method])
    info=dict(beta=answer['beta'],alpha=float(alpha),signal_estimate='fixed marginal rho',
              anchor_correlation=answer['anchor_correlation'])
    if method in ('pooled','position','position_scale_only','position_shuffled','graph_tv','graph_tv_shuffled'):
        g=position_quadratic(answer,prior,alpha,shuffled=method=='position_shuffled',pooled=method=='pooled')
    elif method in ('sliding_window','graph_local','graph_local_shuffled'):
        moments=answer['neighborhood'][method];neff=moments['effective_weight_count']
        gamma=answer['p']/(answer['p']+neff-1.)
        active=answer['active'];pool=prior['real'].mean(0)[active][:,active]
        target=(1-gamma[:,None,None])*moments['covariance']+gamma[:,None,None]*pool
        mixed=(1-alpha)*answer['cbase']+alpha*target
        g=np.einsum('pi,tpq,qk->tik',answer['u'],mixed,answer['u'],optimize=True)+EPS*np.eye(2)
        info.update(effective_weight_count_mean=float(neff.mean()),
                    effective_weight_count_min=float(neff.min()),local_prior_fraction_mean=float(gamma.mean()))
    else:raise ValueError('unknown method: '+method)
    theta=direct_theta(g,answer['r'])
    if method.startswith('graph_tv'):
        edges=graph_edges(answer['graph']['shuffle' if method.endswith('shuffled') else 'real'])
        i,j,a=edges
        a=a*(n/(2*a.sum())) if a.sum()>0 else a
        fitted=solve_network_lasso(g,np.broadcast_to(answer['r'],(n,2)),i,j,a,eta)
        theta=fitted['theta']
        info.update({k:v for k,v in fitted.items() if k!='theta'})
        info.update(eta=float(eta),edge_count=len(a),edge_weight_sum=float(a.sum()))
    weights=answer['sign']*theta@answer['u'].T
    gain=weights@w0/(w0@w0)
    orthogonal=weights-gain[:,None]*w0
    info.update(gain_mean=float(gain.mean()),gain_min=float(gain.min()),gain_max=float(gain.max()),
                direction_change_mean=float(np.linalg.norm(orthogonal,axis=1).mean()/np.linalg.norm(w0)),
                projected_trace_ratio_mean=float(np.trace(g,axis1=1,axis2=2).mean()/np.trace(answer['gbase'])),
                quadratic_condition_max=float(np.linalg.cond(g).max()))
    if method=='position_scale_only':weights=gain[:,None]*w0
    score=np.einsum('tp,tp->t',z,weights)
    if not np.isfinite(score).all():raise ValueError('nonfinite scores; no fallback')
    return score,info,weights


def score_answer(answer,prior,spans,methods):
    scores={};health={};maps={}
    for method in methods:
        started=time.perf_counter()
        try:
            token,info,weights=token_scores(answer,prior,method)
            scores[method]=step_top_mean(token,spans[:,0],spans[:,1],10)
            # Keep compact maps, not copies of all token-by-feature weights.
            region=position_overlap(len(token),answer['uid'])*BINS/len(token)
            full=np.zeros((BINS,12));full[:,answer['active']]=region.T@weights
            maps[method]=full
            health[method]=dict(info,status=info.get('status','OK'),seconds=time.perf_counter()-started)
        except Exception as exc:
            scores[method]=np.full(len(spans),np.nan)
            health[method]=dict(status='FAILED',reason=str(exc),seconds=time.perf_counter()-started)
    return scores,health,maps
