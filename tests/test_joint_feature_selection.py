import numpy as np
from spectral_utils.joint_feature_selection import information


def test_schur_deletion_identity():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,8));cov=np.cov(x,rowvar=False);a=rng.normal(size=(8,3))
    info,loss=information(cov,a)
    for i in range(8):
        keep=np.arange(8)!=i;after,_=information(cov[np.ix_(keep,keep)],a[keep])
        np.testing.assert_allclose(info-after,loss[i],atol=1e-12)


def test_duplicate_has_negligible_conditional_information():
    cov=np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,1.]])
    a=np.array([[1.,0.],[1.,0.],[0.,1.]])
    info,loss=information(cov,a)
    assert np.sum(loss[0]/info)<1e-3
    assert np.sum(loss[2]/info)>.99
    after,_=information(cov[np.ix_([0,2],[0,2])],a[[0,2]])
    np.testing.assert_allclose(after,info,atol=1e-4)


def test_independent_irrelevant_feature_has_zero_contribution():
    cov=np.eye(4);a=np.array([[1.,0.],[0.,1.],[.5,.5],[0.,0.]])
    _,loss=information(cov,a)
    np.testing.assert_array_equal(loss[3],0)
    assert np.all(loss[:3].sum(axis=1)>0)


def test_joint_duplicate_removal_restores_score():
    from spectral_utils.joint_feature_selection import pruning_path
    rng=np.random.default_rng(321);n=3000;groups=np.repeat(np.arange(3),4)
    x=.55*rng.normal(size=(n,1))+.35*rng.normal(size=(n,3))[:,groups]+.25*rng.normal(size=(n,12))
    x=(x-x.mean(0))/x.std(0)
    original=pruning_path(x,groups,anchor_index=0,seed=123,maximum_removals=0)
    augmented=np.column_stack((x,x[:,0]))
    fit=pruning_path(augmented,np.r_[groups,0],anchor_index=0,seed=123,maximum_removals=1)
    assert fit['deletion_audit'][0]['removed'] in (0,12)
    a=x@original['path'][0]['weights'];b=augmented@fit['path'][-1]['weights']
    assert np.corrcoef(a,b)[0,1]>.9999


def test_early_stop_preserves_selected_model():
    from spectral_utils.joint_feature_selection import pruning_path
    rng=np.random.default_rng(9);groups=np.repeat(range(3),4)
    x=.5*rng.normal(size=(500,1))+.4*rng.normal(size=(500,3))[:,groups]+.3*rng.normal(size=(500,12))
    full=pruning_path(x,groups,anchor_index=0,seed=3,maximum_removals=4,retention=.99999)
    early=pruning_path(x,groups,anchor_index=0,seed=3,maximum_removals=4,retention=.99999,stop_after_automatic=True)
    np.testing.assert_array_equal(full['automatic']['active'],early['automatic']['active'])
    np.testing.assert_array_equal(full['automatic']['weights'],early['automatic']['weights'])
