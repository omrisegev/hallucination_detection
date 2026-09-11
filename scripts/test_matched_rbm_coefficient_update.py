from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import check_grad
from scipy.special import logsumexp
from spectral_utils.matched_rbm_coefficient_update import MatchedObjective,Top10
from spectral_utils.rbm_matched_top10 import top10_value_gradient


def main():
    rng=np.random.default_rng(241);p=12;n=60;x=rng.normal(size=(n,p));off=np.array([0,25,60]);owner=np.repeat([0,1],np.diff(off))
    a=rng.normal(size=(2,p));w=rng.normal(size=(2,p));b=rng.normal(size=2);ori=np.array([1.,-1.]);active=np.ones((2,p))
    active[1,-1]=0;x[25:,-1]=0;a[1,-1]=w[1,-1]=0
    base=ori[owner]*(b[owner]+np.sum(x*w[owner],axis=1));spans=np.array([[0,3],[3,25],[25,50],[49,60]])
    y=np.array([0,1,0,-1]);step_owner=np.array([0,0,1,1]);o=MatchedObjective(x,off,a,w,b,ori,active,base,spans,y,step_owner)
    delta=rng.normal(size=p+1)*.1
    for fun in (o.unsupervised,o.supervised):
        assert check_grad(lambda d:fun(d)[0],lambda d:fun(d)[1],delta)<1e-5
    token=base+x@delta[:-1]+delta[-1];s,g=o.top.evaluate(token,True)
    exact=np.array([np.sort(token[start:end])[-min(10,end-start):].mean() for start,end in spans])
    np.testing.assert_allclose(s,exact,atol=1e-14)
    # Direct two-component Gaussian mixture replay, including the inactive column.
    density=[]
    for i in range(2):
        xx=x[off[i]:off[i+1]][:,active[i].astype(bool)];aa=a[i,active[i].astype(bool)]
        ww=(w[i]+ori[i]*active[i]*delta[:-1])[active[i].astype(bool)];bb=b[i]+ori[i]*delta[-1]
        logit=bb+aa@ww+.5*(ww@ww);logpi=-np.logaddexp(0,-logit);logpi0=-np.logaddexp(0,logit)
        ll=logsumexp(np.c_[logpi0-.5*np.sum((xx-aa)**2,axis=1),logpi-.5*np.sum((xx-aa-ww)**2,axis=1)],axis=1)
        density.append(-ll.mean())
    np.testing.assert_allclose(o.unsupervised(delta)[0],np.mean(density)+.005*(delta@delta),atol=1e-12)
    z=np.zeros(n);s,g=o.top.evaluate(z,True);np.testing.assert_array_equal(s,np.zeros(4))
    np.testing.assert_array_equal(g[1,:-1],x[3:13].mean(axis=0))
    assert o.label_mass[-1]==0 and abs(o.label_mass[y==0].sum()-.5)<1e-12
    print('PASS: both gradients, exact Gaussian mixture, inactive columns, orientations, Top10, ties, unknown step labels')


if __name__=='__main__':main()
