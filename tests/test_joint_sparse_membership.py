import numpy as np
from spectral_utils.joint_sparse_membership import exact_aliases,alias_coordinates,sparse_membership,sparse_start


def test_alias_coordinates_are_replication_invariant():
    x=np.random.default_rng(9).normal(size=(80,12));more=np.column_stack((x,x[:,:4]))
    groups=exact_aliases(more)
    assert len(groups)==12 and groups[:4]==[[0,12],[1,13],[2,14],[3,15]]
    np.testing.assert_array_equal(alias_coordinates(more,groups),x)
    np.testing.assert_array_equal(more[:,:12],x)


def test_sparse_joint_can_make_independent_rows_noise_only():
    labels=np.repeat(np.arange(4),4);v=np.r_[np.full(12,.5),np.zeros(4)]
    u=np.r_[np.full(12,.3),np.zeros(4)]
    c=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)+np.eye(16)*.6
    active,audit=sparse_membership(c,labels,penalty=.03,seed=21,starts=5)
    np.testing.assert_array_equal(active,np.arange(12))
    assert audit['converged_starts']>=4
    for fit in audit['starts']:
        assert np.all(np.diff(fit['objective_trace'])<=1e-9)
        assert np.all(fit['v']**2+fit['u']**2<=np.diag(c)+1e-12)


def test_extreme_penalty_selects_diagonal_only_model():
    c=np.full((12,12),.3);np.fill_diagonal(c,1.)
    fit=sparse_start(c,np.repeat(np.arange(3),4),penalty=100.,start=0,seed=1)
    assert fit['converged'] and len(fit['support'])==0
