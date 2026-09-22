"""Numerical contract tests for covariance-conditioned IU (no benchmark labels)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils import conditional_iu_fusion as m
from spectral_utils.conditional_iu_graph import graph_edges,solve_network_lasso


def run():
    rng=np.random.default_rng(20260913)
    latent=rng.normal(size=(81,1))
    x=latent*np.linspace(.2,1.2,12)+rng.normal(size=(81,12))*.6
    x=(x-x.mean(0))/x.std(0)
    methods=tuple(dict.fromkeys(n for family in m.FAMILIES.values() for n in family))
    a=m.prepare_answer(x,'fixture-answer',methods)
    prior=m.regional_covariances(x,'different-training-answer')
    checks=[]
    assert a['u'].shape==(12,2)
    np.testing.assert_allclose(a['u'].T@a['u'],np.eye(2),atol=1e-14)
    checks.append('canonical two-component IU replay and orthonormal projection')
    for name in methods:
        s,_,w=m.token_scores(a,prior,name,alpha=0)
        np.testing.assert_array_equal(s,a['score'])
        np.testing.assert_array_equal(w,np.broadcast_to(a['w'],w.shape))
    checks.append('alpha-zero weights and scores array-exact in every arm')
    g=m.position_quadratic(a,prior,0)
    theta=m.direct_theta(g,a['r'])
    i,j,edge=graph_edges(a['graph']['real'])
    edge*=len(x)/(2*edge.sum())
    tv=solve_network_lasso(g,np.broadcast_to(a['r'],(len(x),2)),i,j,edge,m.ETA)
    np.testing.assert_allclose(tv['theta'],theta,atol=1e-10,rtol=1e-9)
    assert tv['converged']
    checks.append('alpha-zero constant optimum also through general GraphTV solver')
    for name in ('position','pooled','graph_local','sliding_window'):
        s,_,_=m.token_scores(a,prior,name,alpha=1e-8)
        np.testing.assert_allclose(s,a['score'],atol=1e-6,rtol=1e-6)
    checks.append('positive-alpha continuity towards original IU')
    s,_,w=m.token_scores(a,prior,'position')
    s0,_,w0=m.token_scores(a,prior,'graph_tv',eta=0)
    np.testing.assert_array_equal(s,s0);np.testing.assert_array_equal(w,w0)
    checks.append('eta-zero GraphTV equals unpenalized position solve')
    scalar=dict(real=np.stack([(1+j/16)*a['cbase'] for j in range(16)]),shuffle=prior['shuffle'])
    s,_,_=m.token_scores(a,scalar,'position')
    scale,info,_=m.token_scores(a,scalar,'position_scale_only')
    np.testing.assert_allclose(s,scale,atol=1e-10,rtol=1e-9)
    assert info['direction_change_mean']<1e-9
    checks.append('scalar covariance change detected as score scaling rather than direction learning')
    for shuffled in (False,True):
        weights=m.position_overlap(len(x),'different-training-answer',shuffled)*16/len(x)
        stats=prior['shuffle' if shuffled else 'real']
        for k in range(16):
            mean=weights[:,k]@x;z=x-mean
            expected=z.T@(z*weights[:,k,None])
            np.testing.assert_allclose(stats[k],expected,atol=2e-14)
        assert np.linalg.eigvalsh(stats).min()>-1e-12
    # The pooled control uses exactly the regional covariance, not total covariance.
    np.testing.assert_allclose(prior['real'].mean(0),sum(prior['real'])/16,atol=1e-15)
    checks.append('within-region centering and matched pooled covariance, true/shuffled positions')
    for n in (1,2,3,15,17):
        raw=rng.normal(size=(n,12));stat=m.regional_covariances(raw,'short')
        shifted=m.regional_covariances(raw+3.,'short')
        for key in stat:np.testing.assert_allclose(stat[key],shifted[key],atol=1e-12)
    checks.append('fractional short-answer covariances and translation invariance')
    spans=np.array([[0,1],[1,8],[8,41],[41,81]])
    scores,health,maps=m.score_answer(a,prior,spans,methods)
    assert all(np.isfinite(v).all() for v in scores.values()),health
    assert all(v.shape==(16,12) for v in maps.values())
    manual=np.array([np.sort(a['score'][lo:hi])[-10:].mean() for lo,hi in spans])
    np.testing.assert_allclose(scores['baseline'],manual,atol=1e-14)
    checks.append('Top10 preserves short steps, maps retain original twelve feature slots')
    for bad in (np.zeros((8,12)),np.full((8,12),np.nan),np.ones((1,12))):
        try:m.prepare_answer(bad,'invalid',methods)
        except ValueError:pass
        else:raise AssertionError('invalid baseline silently accepted')
    checks.append('short, constant and missing baseline inputs fail explicitly')
    from unittest.mock import patch
    with patch.object(m,'build_token_graph',side_effect=FloatingPointError('injected graph failure')):
        broken=m.prepare_answer(x,'fixture-answer',methods)
    values,health,_=m.score_answer(broken,prior,spans,methods)
    np.testing.assert_array_equal(values['baseline'],scores['baseline'])
    assert np.isfinite(values['position']).all()
    assert np.isnan(values['graph_local']).all() and health['graph_local']['status']=='FAILED'
    exact,_,_=m.token_scores(broken,prior,'graph_local',alpha=0)
    np.testing.assert_array_equal(exact,a['score'])
    checks.append('graph failure stays in its own arm and preserves the native IU baseline')
    return dict(status='PASS',checks=checks,count=len(checks))


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
