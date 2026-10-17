"""Deterministic correctness fixtures, independent of benchmark labels."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils import rbm_hierarchical_time as m

def run():
    rng=np.random.default_rng(913)
    for n in (1,2,7,10,15,16,17,51,1500):
        v=rng.normal(size=n);edge=np.linspace(0,n,17)
        expected=[sum(max(0,min(t+1,edge[j+1])-max(t,edge[j]))*v[t] for t in range(n))/(n/16) for j in range(16)]
        np.testing.assert_allclose(m.regions(v),expected,atol=1e-12)
        np.testing.assert_allclose(m.regions(v).mean(),v.mean(),atol=1e-13)
        p=m.make_profiles(v,[(0,n)],'test');u=rng.uniform(size=16);u/=u.sum()
        score,d=m.drop_first(p['raw']@u,u,p)
        tw=np.array([sum(max(0,min(t+1,edge[j+1])-max(t,edge[j]))*u[j] for j in range(16))/(n/16) for t in range(n)])
        expected=v[0] if n==1 else tw[1:]@v[1:]/tw[1:].sum()
        np.testing.assert_allclose(score,[expected],atol=1e-12)
    for bad in ([],[np.nan],[np.inf]):
        try:m.regions(bad)
        except ValueError:pass
        else:raise AssertionError('bad input accepted')
    p=m.make_profiles(np.ones(4),[(0,1),(1,4)],'constant')
    u,d=m.fit_covariance(p['covariance']);assert d['reason']=='ZERO_COVARIANCE'
    np.testing.assert_allclose(u,1/16);assert np.argmax(p['top'])==0
    a=np.linspace(.3,1.1,16);noise=np.linspace(.4,1.,16);c=np.outer(a,a)+np.diag(noise)
    theta=np.r_[rng.uniform(.2,.8,16),np.log(rng.uniform(.3,1.,16))]
    numerical=approx_derivative(lambda t:m.objective(t,c)[0],theta,method='3-point')
    np.testing.assert_allclose(m.objective(theta,c)[1],numerical,atol=2e-7,rtol=2e-6)
    u,d=m.fit_covariance(c);oracle=a/noise;oracle/=oracle.sum()
    np.testing.assert_allclose(u,oracle,atol=2e-4)
    assert d['converged'] and abs(u.sum()-1)<1e-12 and np.all(u>=0)
    x=rng.normal(size=(100,16));y=rng.integers(0,2,100);sw=np.ones(100)/100
    theta=np.r_[rng.uniform(.01,.1,16),.2]
    numerical=approx_derivative(lambda t:m.supervised_objective(t,x,y,sw)[0],theta)
    np.testing.assert_allclose(m.supervised_objective(theta,x,y,sw)[1],numerical,atol=1e-7)
    from scipy.special import expit
    raw=np.array([-1000.,-2.,0.,2.,1000.])
    for sign in (-1,1):
        np.testing.assert_allclose(expit(sign*raw),expit(raw) if sign==1 else 1-expit(raw),atol=1e-15)
    return dict(status='PASS',fixtures='overlap, short/constant/missing, boundary, gradients, factor recovery, orientation')

if __name__=='__main__':print(run())
