import numpy as np
from spectral_utils.joint_staged_membership import staged_group_support


def test_zero_rows_are_removed_before_global_zero_groups():
    labels=np.array([0,0,1,1,2,2]);local=np.array([0,1])
    audit={'starts':[{'converged':True,'v':np.zeros(6)}]}
    kept,detail=staged_group_support(local,labels,audit)
    np.testing.assert_array_equal(kept,[0,1]);assert detail['stage']=='zero_rows'
    # Stable all-local support is not permanently protected.
    kept,detail=staged_group_support(np.arange(2),np.array([0,0]),
        {'starts':[{'converged':True,'v':np.zeros(2)}]})
    assert len(kept)==0 and detail['stage']=='global_groups'


def test_stable_support_excludes_only_global_zero_groups():
    labels=np.array([0,0,1,1,2,2]);audit={'starts':[{'converged':True,'v':[.3,0,.2,0,0,0]}]}
    kept,detail=staged_group_support(np.arange(6),labels,audit)
    np.testing.assert_array_equal(kept,[0,1,2,3]);assert detail['stage']=='global_groups'
    assert detail['nuisance_groups']==[2]


def test_ordering_is_permutation_equivariant():
    labels=np.array([0,0,1,1,2,2]);v=np.array([.3,0,.2,0,0,0]);perm=np.array([5,2,0,4,3,1])
    for local in (np.arange(6),np.array([0,1,2,3])):
        kept,_=staged_group_support(local,labels,{'starts':[{'converged':True,'v':v}]})
        loc=np.flatnonzero(np.isin(perm,local))
        other,_=staged_group_support(loc,labels[perm],{'starts':[{'converged':True,'v':v[perm]}]})
        np.testing.assert_array_equal(np.sort(perm[other]),kept)
