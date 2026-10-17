import numpy as np
from spectral_utils.digitfree_broad50 import adjacent_views, token_bank, step_bank, NAMES
from spectral_utils.joint_block_balanced import block_pair_weights, fit_weighted_start
from spectral_utils.joint_lsml import _fit_one_start, coassignment_from_labels


def test_identity_aligned_js_and_turnover():
    ids=np.tile(np.arange(50),(3,1)); lp=np.full((3,50),-np.log(50))
    ids[1]=ids[1,::-1]; ids[2]+=50
    turnover,js=adjacent_views(ids,lp)
    np.testing.assert_allclose(js,[0,0,1],atol=1e-14)
    np.testing.assert_allclose(turnover,[0,1,1])


def test_rank_and_masked_readout():
    p=np.linspace(50,1,50); p=p/p.sum()*.9
    lp=np.tile(np.log(p),(3,1)); ids=np.tile(np.arange(50),(3,1))
    x,valid,_=token_bank(lp,ids,np.array([0,2,100]),np.array([-lp[0,0],-lp[0,2],10]))
    assert x.shape==(3,50) and len(NAMES)==50
    np.testing.assert_allclose(x[:,NAMES.index('censored_rank50')],[0,2,50])
    np.testing.assert_allclose(x[:,NAMES.index('mass_above')],[0,p[:2].sum(),.9])
    assert valid[0,:35].all() and not valid[0,35:].any()
    out,mask,_=step_bank(lp,ids,np.zeros(3,int),-lp[:,0],[[0,1],[1,3]])
    assert not mask[0,35:].any() and np.isnan(out[0,35:]).all()
    np.testing.assert_allclose(out[1,35:],0,atol=1e-12)


def test_block_total_weights():
    labels=np.repeat([0,1,2],[2,3,5]); w=block_pair_weights(labels)
    left,right=np.triu_indices(len(labels),1)
    totals=[]
    for a in range(3):
        for b in range(a,3):
            mask=(labels[left]==a)&(labels[right]==b)
            totals.append(w[left[mask],right[mask]].sum())
    np.testing.assert_allclose(totals,totals[0])
    assert np.allclose(w,w.T) and not np.diag(w).any()


def test_uniform_weights_replay_legacy_optimizer():
    rng=np.random.default_rng(17); labels=np.repeat(np.arange(3),3)
    v=rng.uniform(.3,.7,9); u=rng.uniform(.1,.4,9)
    mask=coassignment_from_labels(labels); np.fill_diagonal(mask,0)
    observed=np.outer(v,v)+mask*np.outer(u,u)+np.eye(9)
    old=_fit_one_start(observed,mask,start=0,seed=5,anchor_index=0,max_sweeps=5000,
        relative_tolerance=1e-10,consecutive_stable_sweeps=5,monotonicity_tolerance=1e-12)
    new=fit_weighted_start(observed,mask,np.ones((9,9))-np.eye(9),start=0,seed=5,anchor_index=0)
    np.testing.assert_allclose(new['fitted'],old.fitted_offdiag,atol=1e-12)
    assert new['converged']==old.converged
