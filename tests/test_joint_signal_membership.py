import numpy as np
from spectral_utils.joint_signal_membership import signal_group_support


def test_local_only_rows_survive_in_signal_group():
    labels=np.array([0,0,1,1,2,2])
    audit={'starts':[{'converged':True,'v':[1,0,0,0,0,0],
                       'u':[0,1,1,1,0,0]}]}
    selected,detail=signal_group_support(np.array([0,1,2,3]),labels,audit)
    np.testing.assert_array_equal(selected,[0,1])
    assert detail['nuisance_groups']==[1,2]


def test_any_converged_signal_preserves_group_and_ignores_failed_start():
    labels=np.array([7,7,4,4])
    audit={'starts':[{'converged':True,'v':[1,0,0,0]},
                     {'converged':True,'v':[1,0,.1,0]},
                     {'converged':False,'v':[0,0,1,1]}]}
    selected,detail=signal_group_support(np.arange(4),labels,audit)
    np.testing.assert_array_equal(selected,np.arange(4))
    assert detail['nuisance_groups']==[]
    audit['starts'][1]['v']=[1,0,0,0]
    selected,_=signal_group_support(np.arange(4),labels,audit)
    np.testing.assert_array_equal(selected,[0,1])


def test_support_is_permutation_equivariant_and_no_signal_is_empty():
    labels=np.array([0,0,1,1,2,2]);v=np.array([.2,0,0,0,.3,0])
    audit={'starts':[{'converged':True,'v':v}]}
    selected,_=signal_group_support(np.arange(6),labels,audit)
    perm=np.array([4,2,1,5,0,3])
    other,_=signal_group_support(np.arange(6),labels[perm],
        {'starts':[{'converged':True,'v':v[perm]}]})
    np.testing.assert_array_equal(np.sort(perm[other]),selected)
    empty,_=signal_group_support(np.arange(6),labels,
        {'starts':[{'converged':True,'v':np.zeros(6)}]})
    assert len(empty)==0
