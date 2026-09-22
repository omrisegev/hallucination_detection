"""Preflight mechanics only: no benchmark fitting or outcome selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import check_grad
from spectral_utils.rbm_matched_top10 import top10_value_gradient,step_supervised_objective
from spectral_utils.direct_probability_fusion import step_top_mean


def main():
    rng=np.random.default_rng(20260911);x=rng.normal(size=(57,12));theta=rng.normal(size=13)
    spans=np.array([[0,3],[3,25],[24,57]])
    for sign in (1,-1):
        t=sign*theta;v,j=top10_value_gradient(x,spans,t[:-1],t[-1])
        np.testing.assert_allclose(v,step_top_mean(x@t[:-1]+t[-1],spans[:,0],spans[:,1],10),atol=1e-14)
        pb=lambda p:step_supervised_objective(p,x,spans,first_error=1,ridge=.01)
        prm=lambda p:step_supervised_objective(p,x,spans,labels=np.array([0,1,-1]),weights=np.array([.5,.5,0.]),ridge=.01)
        for f in (pb,prm):
            assert check_grad(lambda p:f(p)[0],lambda p:f(p)[1],t)<1e-5
    v,j=top10_value_gradient(x,spans,np.zeros(12),0.)
    np.testing.assert_array_equal(v,np.zeros(3));np.testing.assert_array_equal(j[1,:12],x[3:13].mean(axis=0))
    value,grad=step_supervised_objective(theta,x,np.array([[0,57]]),first_error=0)
    np.testing.assert_allclose(value,0,atol=1e-14);np.testing.assert_allclose(grad,0,atol=1e-14)
    v,g=step_supervised_objective(theta*1000,x,spans,first_error=1);assert np.isfinite(v) and np.isfinite(g).all()
    try:top10_value_gradient(x,[[0,0]],theta[:-1],theta[-1])
    except ValueError:pass
    else:raise AssertionError('empty step accepted')
    print('PASS: original Top10 replay, both orientations, step-label gradients, short/overlapping/one-step spans, ties and extreme scores')


if __name__=='__main__':main()
