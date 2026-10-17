import numpy as np
from spectral_utils.joint_group_reliability import group_reliability_weights


def test_weak_group_is_not_promoted_by_standardization():
    labels = np.repeat(np.arange(3),3)
    v = np.r_[np.full(6,.5),np.full(3,1e-6)]
    c = np.outer(v,v)+np.eye(9)
    w,m = group_reliability_weights(c,v,labels)
    assert np.abs(w[6:]).sum()/np.abs(w).sum() < 2e-6
    assert m['groups'][2]['modeled_correlation'] < 2e-6
    for g in range(3):
        ids=np.flatnonzero(labels==g)
        np.testing.assert_allclose(w[ids]/v[ids],np.full(3,m['groups'][g]['multiplier']))


def test_group_replication_preserves_scores_for_fixed_model():
    # All measurements in a group duplicated, including their residual
    # covariance. This is a readout invariance, not a refit/group-discovery claim.
    v=np.array([.6,.4,.3,.2,.5,.4]); labels=np.repeat(np.arange(3),2)
    c=np.outer(v,v)+np.diag(np.linspace(.3,.8,6))
    w,_=group_reliability_weights(c,v,labels)
    index=np.r_[np.arange(6),[0,1]]
    wa,_=group_reliability_weights(c[np.ix_(index,index)],v[index],labels[index])
    effective=wa[:6].copy(); effective[:2]+=wa[6:]
    np.testing.assert_allclose(effective,w,atol=1e-12)
