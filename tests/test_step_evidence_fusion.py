import numpy as np
import unittest
from spectral_utils.step_evidence_fusion import shape_views,score_evidence,standardize,weights
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS


def test_shapes_against_scalar_windows_and_short_steps():
    x=np.random.default_rng(2).normal(size=(51,5));spans=np.array([[0,1],[1,4],[4,19],[19,51]])
    b,r,s=shape_views(x,spans,'a')
    for j,(a,c) in enumerate(spans):
        v=x[a:c];k=min(10,len(v))
        np.testing.assert_allclose(b[j],np.mean([sorted(v[:,f])[-k:] for f in range(5)]))
        np.testing.assert_allclose(r[j,0],v[-min(4,len(v)):].mean())
        expected=np.mean([max(v[t:t+k,f].mean() for t in range(len(v)-k+1)) for f in range(5)])
        np.testing.assert_allclose(r[j,1],expected)
    np.testing.assert_array_equal(s,shape_views(x,spans,'a')[2])
    assert not np.array_equal(s,shape_views(x,spans,'b')[2])


def test_identity_context_and_zero_amplitude():
    rng=np.random.default_rng(3);b=rng.normal(size=25);a=rng.normal(size=25)
    t=b+.25*b.std()*standardize(a)[0];sh=rng.normal(size=(25,2))
    result,_=score_evidence(b,t,sh,sh)
    np.testing.assert_allclose(result[:,0],t,atol=1e-13)
    off,_=score_evidence(b,t,sh,sh,amplitude=0)
    np.testing.assert_array_equal(off,np.broadcast_to(b[:,None],off.shape))


def test_canonical_weights():
    rng=np.random.default_rng(4)
    for n in (3,7,80):
        z,_=standardize(rng.normal(size=(n,3)))
        w,d=weights(z);c=z.T@z/n
        ref=upcr_fit_covariance(c,**IU_FIT_DEFAULTS).w
        if ref@c@np.ones(3)<0:ref=-ref
        assert d['native']
        np.testing.assert_allclose(w,ref,atol=2e-7,rtol=2e-7)


def test_explicit_fallback_and_nonfinite_failure():
    for n in (1,2):
        b=np.arange(n,dtype=float);shape=np.column_stack((b,b))
        out,d=score_evidence(b,b,shape,shape)
        assert not d['real']['native']
        np.testing.assert_array_equal(out[:,3],out[:,4])
    with unittest.TestCase().assertRaises(ValueError):standardize(np.array([1.,np.nan]))


def test_feature_agreement_can_create_new_peak():
    # No single-view maximum at the error candidate; fusion can create one.
    x=np.array([[10.,0.],[0.,10.],[6.,6.]])
    assert not np.any(x.argmax(0)==2)
    assert x.mean(1).argmax()==2


class EvidenceTests(unittest.TestCase):
    pass


for _name,_fn in list(globals().items()):
    if _name.startswith('test_'):
        setattr(EvidenceTests,_name,lambda self,fn=_fn:fn())
del _name,_fn


if __name__=='__main__':
    unittest.main()
