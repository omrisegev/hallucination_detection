from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import check_grad
from scipy.special import logsumexp
from spectral_utils.rbm_first_error_objective import FirstErrorObjective


def main():
    rng=np.random.default_rng(350);x=rng.normal(size=(64,12));base=rng.normal(size=64)
    spans=np.array([[0,2],[2,20],[19,30],[30,32],[32,50],[50,64]])
    off=np.array([0,3,4,6]);target=np.array([1,-1,0]);delta=rng.normal(size=13)*.1
    f=FirstErrorObjective(x,base,spans,off,target)
    assert check_grad(lambda d:f(d)[0],lambda d:f(d)[1],delta)<1e-5
    token=base+x@delta[:-1]+delta[-1]
    steps=np.array([np.sort(token[a:b])[-min(10,b-a):].mean() for a,b in spans])
    exact=np.mean([logsumexp(steps[off[i]:off[i+1]])-steps[off[i]+t] for i,t in enumerate(target) if t>=0])+.005*(delta@delta)
    np.testing.assert_allclose(f(delta)[0],exact,atol=1e-12)
    shifted=delta.copy();shifted[-1]+=20
    np.testing.assert_allclose(f(shifted)[0]-.005*(shifted@shifted),f(delta)[0]-.005*(delta@delta),atol=1e-12)
    assert abs(f(delta)[1][-1]-.01*delta[-1])<1e-15
    # Clean-answer token changes cannot enter the location objective.
    other=base.copy();other[30:32]+=100
    g=FirstErrorObjective(x,other,spans,off,target);np.testing.assert_allclose(g(delta)[0],f(delta)[0],atol=1e-12)
    one=FirstErrorObjective(x,base,np.array([[0,64]]),np.array([0,1]),np.array([0]));zero=np.zeros(13)
    np.testing.assert_allclose(one(zero)[0],0,atol=1e-14);np.testing.assert_allclose(one(zero)[1],0,atol=1e-14)
    try:FirstErrorObjective(x,base,spans,off,np.full(3,-1))
    except ValueError:pass
    else:raise AssertionError('accepted no-error-only location training')
    print('PASS: direct first-error loss, gradient, clean exclusion, shared/short spans, one-step answer and shift invariance')


if __name__=='__main__':main()
