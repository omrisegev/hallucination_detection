"""Mathematical/identity tests; no development labels used to tune the model."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp, expit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils import answer_position_fusion as m
from spectral_utils.moment_rbm_fusion import rbm_objective
from spectral_utils.upcr import upcr_fit
from scripts.test_two_axis_factor_fusion import run as factor_tests


def run():
    inherited=factor_tests();rng=np.random.default_rng(20260913)
    for n in (1,2,7,16,19,55,513):
        x=rng.normal(size=(n,4));o=m.position_overlap(n,'test')
        os=m.position_overlap(n,'test',True)
        np.testing.assert_allclose(o.sum(1),1.,atol=1e-13)
        np.testing.assert_allclose(o.sum(0),n/16,atol=1e-13)
        np.testing.assert_allclose(os.sum(0),o.sum(0),atol=1e-13)
        np.testing.assert_array_equal(os,m.position_overlap(n,'test',True))
        s=m.statistics(x,'test')
        for name in ('real','shuffle'):
            np.testing.assert_allclose(s[name].mean(0),x.T@x/n,atol=1e-12)
            np.testing.assert_allclose(s[name+'_mean'].mean(0),x.mean(0),atol=1e-12)
        coef=np.tile([.2,-.3,.1,.5],(16,1));a=dict(coefficients=coef,intercept=np.full(16,.7))
        tok=m.token_scores(x,a,'test')
        np.testing.assert_allclose(tok,x@coef[0]+.7,atol=1e-12)
        np.testing.assert_allclose(tok,m.token_scores(x,a,'test',True),atol=1e-12)
        # Step boundaries can change readout, NEVER the underlying token score.
        cuts=sorted(set((0,n//2,n)));spans=list(zip(cuts[:-1],cuts[1:]))
        got=m.step_scores(x,spans,a,'test')
        expected=[np.sort(tok[lo:hi])[-min(10,hi-lo):].mean() for lo,hi in spans]
        np.testing.assert_allclose(got,expected,atol=1e-12)
    # Same token features at start and end obtain distinct POSITION coefficients.
    x=np.ones((32,4));a=dict(coefficients=np.repeat(np.arange(16)[:,None],4,axis=1).astype(float),intercept=np.zeros(16))
    tok=m.token_scores(x,a,'global');assert tok[0]!=tok[16]
    shuffled=m.token_scores(x,a,'global',True)
    assert not np.array_equal(tok,shuffled)
    # Coefficients, not feature content, move in the shuffled arm.
    np.testing.assert_allclose(shuffled,np.sum(x*(m.position_overlap(32,'global',True)@a['coefficients']),1))

    p=4;x=rng.normal(size=(47,p));a=rng.normal(size=p)*.2;w=rng.normal(size=p)*.2;b=.1
    blocks=[(x,np.full(len(x),1/(16*len(x)))) for _ in range(16)]
    obj=m.RBMObjective(blocks,1,True,ridge=0)
    theta=np.r_[a,1.,w,b];value,g=obj(theta)
    old_value,old_g=rbm_objective(np.r_[a,w,b],x)
    np.testing.assert_allclose(value,old_value,atol=1e-12)
    np.testing.assert_allclose(g[:p],old_g[:p],atol=1e-12)
    np.testing.assert_allclose(g[p+1:-1],old_g[p:2*p],atol=1e-12)
    np.testing.assert_allclose(g[-1],old_g[-1],atol=1e-12)
    for rank in (1,2):
        xs=[rng.normal(size=(11+j,p)) for j in range(16)]
        blocks=[(v,np.full(len(v),1/(16*len(v)))) for v in xs]
        obj=m.RBMObjective(blocks,rank)
        u=rng.normal(size=(16,rank))*.3;v=rng.normal(size=(p,rank))*.3
        theta=np.r_[a,u.ravel(),v.ravel(),b]
        value,g=obj(theta)
        fd=approx_derivative(lambda t:obj(t)[0],theta).ravel()
        np.testing.assert_allclose(g,fd,atol=2e-7,rtol=2e-5)
        # Independent exact mixture density: N(a,I) and N(a+w_j,I).
        weights=u@v.T;direct=0.
        for j,(xx,mass) in enumerate(blocks):
            logprior=b+a@weights[j]+.5*weights[j]@weights[j]
            l0=-.5*np.sum((xx-a)**2,1)-np.logaddexp(0.,logprior)
            l1=-.5*np.sum((xx-a-weights[j])**2,1)-np.logaddexp(0.,-logprior)
            direct-=mass@logsumexp(np.stack((l0,l1)),axis=0)
        direct+=.5*m.RIDGE*np.mean(weights*weights)
        np.testing.assert_allclose(value,direct,atol=1e-12)
    # Actual fits exercise both deterministic starts and raw-logit scoring.
    blocks=[(x,np.full(len(x),1/(16*len(x)))) for _ in range(16)]
    ar,health=m.fit_rbm(blocks,1,True,maxiter=30)
    assert len(health['starts'])==2 and np.isfinite(ar['coefficients']).all()
    np.testing.assert_allclose(m.token_scores(x,ar,'f'),x@ar['coefficients'][0]+ar['intercept'][0],atol=1e-12)
    # Genuine IU seam versus canonical feature-by-token API, including scaling.
    xx=rng.normal(size=(400,6));xx=(xx-xx.mean(0))/xx.std(0)
    stat=m.statistics(xx,'iu');ai,hi=m.fit_iu(stat['real'],stat['real_mean'],stationary=True)
    ref=upcr_fit(xx.T,**m.IU_FIT_DEFAULTS)
    c=xx.T@xx/len(xx);sgn=-1 if ref.w@c[:,1]<0 else 1
    np.testing.assert_allclose(ai['coefficients'][0],sgn*ref.w,atol=1e-10,rtol=1e-10)
    # True means are removed when obtaining covariance; shifts cannot be called correlations.
    means=np.tile(np.arange(4,dtype=float),(16,1))
    seconds=np.repeat(np.eye(4)[None],16,axis=0)+means[:,:,None]*means[:,None,:]
    np.testing.assert_allclose(m.centered_moments(seconds,means),np.repeat(np.eye(4)[None],16,axis=0))
    bad=dict(coefficients=np.full((16,4),np.nan),intercept=np.zeros(16))
    assert np.isnan(m.step_scores(x,[(0,len(x))],bad,'bad')).all()
    try:m.statistics(np.full((3,4),np.nan),'bad')
    except ValueError:pass
    else:raise AssertionError('missing features accepted')
    # Alter an excluded answer's features and labels; actual training blocks
    # and a fitted IU map must stay identical. Not merely a fit-signature check.
    from scripts.run_answer_position_fusion import rbm_blocks
    xs=rng.normal(size=(30,6));cache=dict(ids=np.array([0,1,2]),token_offsets=np.array([0,10,20,30]),x=xs.copy())
    records=[dict(uid=str(i),label=i%2) for i in range(3)]
    weights={0:.5,1:.5}
    before=rbm_blocks(cache,[0,1],weights,records,False)
    cache['x'][20:]=1000.;records[2]['label']=1-records[2]['label']
    after=rbm_blocks(cache,[0,1],weights,records,False)
    for (xb,wb),(xa,wa) in zip(before,after):
        np.testing.assert_array_equal(xb,xa);np.testing.assert_array_equal(wb,wa)
    assert np.isclose(sum(w.sum() for _,w in before),1.)
    def moments(blocks):
        return (np.array([(x.T*w)@x*16 for x,w in blocks]),
                np.array([w@x*16 for x,w in blocks]))
    ib,_=m.fit_iu(*moments(before),groups=2)
    ia,_=m.fit_iu(*moments(after),groups=2)
    np.testing.assert_array_equal(ib['coefficients'],ia['coefficients'])
    mean=rng.normal(size=(16,6))
    for groups in (None,2,500):
        ctrl=m.position_mean_control(ia,mean,groups)
        expected_mean=mean if groups is None else (groups*mean+16*mean.mean(0))/(groups+16)
        np.testing.assert_array_equal(ctrl['coefficients'],ia['coefficients'])
        np.testing.assert_allclose(ctrl['intercept'],-np.sum(ia['coefficients']*expected_mean,axis=1),atol=1e-12)
    return dict(status='PASS',whole_answer_clock=True,short_answers=True,
                step_segmentation_independent=True,shuffle_preserves_token_identity=True,
                rbm_exact_historical_objective=True,rbm_gradients_ranks=[1,2],
                rbm_independent_mixture_density=True,canonical_iu_replay=True,
                factor_tests=inherited,held_answer_feature_label_firewall=True,benchmark_inference=False)


if __name__=='__main__':print(run())
