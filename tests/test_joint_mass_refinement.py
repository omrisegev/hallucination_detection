import numpy as np
from unittest.mock import patch
from spectral_utils.joint_mass_refinement import refine_mass


def fixture(weak=False):
    n=160;labels=np.repeat(np.arange(4),3);rng=np.random.default_rng(410170)
    a=rng.normal(size=(n,17));a-=a.mean(axis=0);q=np.linalg.qr(a)[0]*np.sqrt(n-1)
    v=np.tile([.35,.35,.02] if weak else [.35]*3,4)
    u=np.tile([.5,.5,.03] if weak else [.5]*3,4)
    x=q[:,0,None]*v+q[:,1+labels]*u+q[:,5:]*np.sqrt(1-v*v-u*u)
    return x,labels,np.arange(n)%4,v,u


def test_budget_rejection_precedes_group_discovery():
    x,labels,folds,_,_=fixture()
    with patch('spectral_utils.joint_mass_refinement.discover_mass_groups',
               side_effect=AssertionError('Budget violation must not reach discovery')) as discover:
        fit=refine_mass(x,labels,folds,seed=410370,group_seed=410270)
    assert discover.call_count==0;np.testing.assert_array_equal(fit['active'],np.arange(12))
    rejected=fit['selection']['deletion_audit'][-1]['rejected_candidates'];assert len(rejected)==3
    assert all(r['reason']=='INFORMATION_BUDGET' and r['discovery'] is None for r in rejected)


def test_accepted_path_always_preserves_fixed_initial_information():
    x,labels,folds,v,u=fixture(weak=True)
    # Known latent partition isolates the budget mechanism; this is not a
    # real-data grouping-quality test.
    def partition(sub,folds,seed):
        ids=[int(np.argmin(np.linalg.norm(x-column[:,None],axis=0))) for column in sub.T]
        return dict(status='SELECTED',labels=labels[ids],candidates=[])
    with patch('spectral_utils.joint_mass_refinement.discover_mass_groups',side_effect=partition):
        fit=refine_mass(x,labels,folds,seed=410370,group_seed=410270)
    s=fit['selection'];assert len(s['path'])>1 and s['automatic_index']==len(s['path'])-1
    cov=np.cov(x,rowvar=False);reg=(1-1e-4)*cov+1e-4*np.diag(np.diag(cov))
    a=np.zeros((12,5));a[:,0]=v
    for g in range(4):a[labels==g,g+1]=u[labels==g]
    initial=np.sum(a*np.linalg.solve(reg,a),axis=0)
    np.testing.assert_allclose(s['initial_factor_information'],initial,atol=1e-6)
    for state in s['path']:
        ids=np.asarray(state['active']);expected=np.sum(a[ids]*np.linalg.solve(reg[np.ix_(ids,ids)],a[ids]),axis=0)/initial
        np.testing.assert_allclose(state['retention'],expected,atol=1e-6)
        assert min(state['retention'])>=.95


def test_information_feasible_proposal_still_requires_valid_groups():
    x,labels,folds,_,_=fixture(weak=True)
    with patch('spectral_utils.joint_mass_refinement.discover_mass_groups',
               return_value={'status':'NO_ADMISSIBLE_PARTITION','candidates':[]}) as discover:
        fit=refine_mass(x,labels,folds,seed=410370,group_seed=410270)
    assert discover.call_count==3;np.testing.assert_array_equal(fit['active'],np.arange(12))
    assert all(r['reason']=='NO_ADMISSIBLE_REGROUPING' for r in fit['selection']['deletion_audit'][-1]['rejected_candidates'])
