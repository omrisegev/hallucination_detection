import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import check_grad
from scipy.special import logsumexp
from spectral_utils.supervised_position_diagnostic import *


def main():
    rng=np.random.default_rng(1701)
    x=rng.normal(size=(9,3));off=np.array([0,1,4,9]);target=np.array([0,2,1]);t=rng.normal(size=3)
    f=lambda p:listwise_loss(p,x,off,target)
    assert check_grad(lambda p:f(p)[0],lambda p:f(p)[1],t)<1e-6
    exact=np.mean([logsumexp(x[a:b]@t)-(x[a:b]@t)[y] for a,b,y in zip(off[:-1],off[1:],target)])+.005*(t@t)
    np.testing.assert_allclose(f(t)[0],exact,atol=1e-12)
    y=np.array([0,1,0,1,0,1,0,0,1]);answer=np.repeat(np.arange(3),3)
    weights=balanced_answer_weights(y,answer)
    assert abs(weights[y==0].sum()-.5)<1e-12 and abs(weights[y==1].sum()-.5)<1e-12
    t=rng.normal(size=4);g=lambda p:binary_loss(p,x,y,weights)
    assert check_grad(lambda p:g(p)[0],lambda p:g(p)[1],t)<1e-6
    c=np.array([-1]*5+[1]*4);d=design(x,c,'conditional')
    np.testing.assert_array_equal(d[:,:4],design(x,c,'prior'))
    w=rng.normal(size=3);beta=.7;delta=rng.normal(size=3)
    np.testing.assert_allclose(d@np.r_[w,beta,delta],x@w+beta*c+c*(x@delta))
    assert np.isfinite(listwise_loss(np.full(3,1000.),x,off,target)[0])
    theta,info=optimize(f,3);assert info['loss']<=info['initial_loss']
    theta,info=optimize(g,4);assert info['loss']<=info['initial_loss']
    print('PASS: finite-difference gradients, direct listwise loss, one-step answer, balancing, nesting, extreme logits, convergence')


if __name__=='__main__':main()
