import numpy as np
from unittest.mock import patch
from spectral_utils.joint_regroup_refinement import refine_with_regrouping


def fixture():
    n=160;labels=np.repeat(np.arange(4),3);rng=np.random.default_rng(409170)
    a=rng.normal(size=(n,17));a-=a.mean(axis=0)
    q=np.linalg.qr(a)[0]*np.sqrt(n-1)
    x=.35*q[:,0,None]+.5*q[:,1+labels]+np.sqrt(1-.35**2-.5**2)*q[:,5:]
    return x,labels,np.arange(n)%4


def test_initial_factor_information_is_not_reset_after_regrouping():
    x,labels,folds=fixture()
    fit=refine_with_regrouping(x,labels,folds,seed=409370,group_seed=409270)
    selection=fit['selection'];path=selection['path'];assert len(path)>=2
    cov=np.cov(x,rowvar=False);a=np.zeros((12,5));a[:,0]=.35
    for group in range(4):a[labels==group,1+group]=.5
    regularized=(1-1e-4)*cov+1e-4*np.diag(np.diag(cov))
    initial=np.sum(a*np.linalg.solve(regularized,a),axis=0)
    np.testing.assert_allclose(selection['initial_factor_information'],initial,atol=1e-6)
    for state in path:
        ids=np.asarray(state['active']);sub=regularized[np.ix_(ids,ids)]
        expected=np.sum(a[ids]*np.linalg.solve(sub,a[ids]),axis=0)/initial
        np.testing.assert_allclose(state['retention'],expected,atol=1e-6)
    index=selection['automatic_index'];assert path[index]['minimum_retention']>=.95
    if index+1<len(path):assert path[index+1]['minimum_retention']<.95


def test_inadmissible_regrouping_rejects_deletion_without_old_label_fallback():
    x,labels,folds=fixture()
    with patch('spectral_utils.joint_regroup_refinement.discover_sourcefold_groups',
               return_value={'status':'NO_ADMISSIBLE_PARTITION','candidates':[]}) as discover:
        fit=refine_with_regrouping(x,labels,folds,seed=409370,group_seed=409270)
    np.testing.assert_array_equal(fit['active'],np.arange(12));assert discover.call_count==3
    stopped=fit['selection']['deletion_audit'][-1]
    assert stopped['stopped']=='NO_VALID_DELETION_AMONG_THREE_BEST'
    assert all(r['reason']=='NO_ADMISSIBLE_REGROUPING' for r in stopped['rejected_candidates'])
