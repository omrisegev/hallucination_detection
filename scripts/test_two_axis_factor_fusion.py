import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral_utils import two_axis_factor_fusion as m

def run():
    rng=np.random.default_rng(613)
    for rank in (1,2):
        j,p=16,4;u=rng.normal(size=(j,rank))*.4;v=rng.normal(size=(p,rank))
        z=rng.normal(size=(j,p,p));c=z@z.transpose(0,2,1)+np.eye(p)
        th=np.r_[u.ravel(),v.ravel(),np.zeros(p)]
        val,g=m.objective(th,c,rank)
        fd=approx_derivative(lambda t:m.objective(t,c,rank)[0],th).ravel()
        np.testing.assert_allclose(g,fd,rtol=2e-5,atol=2e-6)
        w=u@v.T;s=w[:,:,None]*w[:,None,:]+np.eye(p)
        direct=.5*np.mean(np.linalg.slogdet(s)[1]+np.einsum('jik,jki->j',np.linalg.inv(s),c))+.5*m.RIDGE*np.mean(w*w)
        np.testing.assert_allclose(val,direct,atol=1e-12)
    for n in (1,2,7,16,19,55):
        o=m.overlap(n);np.testing.assert_allclose(o.sum(1),1.,atol=1e-14)
        np.testing.assert_allclose(o.sum(0),n/16,atol=1e-14)
        x=rng.normal(size=(n,4));coef=np.tile([.2,.3,-.2,.5],(16,1))
        got=m.step_scores(x,[(0,n)],coef,'test')[0]
        s=x@coef[0];nt=min(n,10)
        np.testing.assert_allclose(got,np.sort(s)[-nt:].mean(),atol=1e-14)
        real,shuf=m.sufficient_statistics(x,[(0,n)],'test')
        np.testing.assert_allclose(real.mean(0),x.T@x/n,atol=1e-13)
        np.testing.assert_allclose(shuf.mean(0),real.mean(0),atol=1e-13)
    u=np.c_[np.ones(16),np.linspace(-.9,.9,16)]
    v=np.array([[.8,.5],[1.,0.],[.5,-.9],[.7,.7]])
    w=u@v.T;c=w[:,:,None]*w[:,None,:]+np.diag([.2,.3,.4,.3])
    a1,h1=m.fit(c,1);a2,h2=m.fit(c,2)
    assert h2['objective'] < h1['objective']-.01,(h1,h2)
    assert h2['residual_relative']<.03,h2
    assert np.linalg.matrix_rank(a1['coefficients'],tol=1e-9)==1
    assert np.linalg.matrix_rank(a2['coefficients'],tol=1e-9)==2
    az,hz=m.fit(np.zeros((16,4,4)),1)
    assert np.isfinite(az['coefficients']).all()
    try:m.sufficient_statistics(np.array([[np.nan]*4]),[(0,1)],'bad')
    except ValueError:pass
    else:raise AssertionError('nonfinite input accepted')
    return dict(status='PASS',gradient_ranks=[1,2],direct_gaussian=True,short_steps=True,
                second_moment_identity=True,stationary_token_score=True,synthetic_rank2_misfit=h2['residual_relative'])

if __name__=='__main__':print(run())
