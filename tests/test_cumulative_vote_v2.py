import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments'))
from cvf_v2.core import encode, profiles, pava, location, fit_spectral, FusionModel, training_matrix
from cvf_v2.em import expectation, fit_em, compress
from cvf_v2.readout import earliest_mode

def test_grid_equal_mass_peak_ties_and_single_step():
    p=np.array([[0.,2.],[3.,1.],[1.,0.]])
    assert encode(p,'hard','pb').shape==encode(p,'soft','pb').shape==(2,2)
    m=FusionModel('equal');mass,fail=location(m,p,'hard')
    np.testing.assert_allclose(mass,[.5,.5,0]);assert not fail
    assert mass.argmax()==0
    assert location(m,p[:1],'soft')[0][0]==1
    np.testing.assert_allclose(location(m,np.zeros((3,2)),'soft')[0],np.ones(3)/3)
    # No epsilon perturbation can invert a unique maximum.
    assert location(m,np.array([[1.],[1.00001]]),'soft')[0].argmax()==1
    assert earliest_mode(location(m,np.zeros((3,11)),'soft')[0])==0
    assert earliest_mode([.5,.5+1e-16])==0
    assert earliest_mode([.5,.5+1e-8])==1

def test_onset_negative_range_and_constant():
    p=profiles(np.repeat(np.array([[-5.],[-4.],[-3.]]),5,axis=0),[(0,5),(5,10),(10,15)])
    assert np.argmax(p[:,0,6])==2
    constant=profiles(np.ones((10,1)),[(0,5),(5,10)])
    assert constant[0,0,6]==0 and np.isneginf(constant[1,0,6])

def test_prm_multiple_errors_and_median_tie():
    p=np.array([[1.,0.],[3.,0.],[2.,0.],[4.,0.]])
    v=encode(p,'hard','prm')
    np.testing.assert_array_equal(v[:,0],[-1,1,-1,1]);assert np.all(v[:,1]==-1)

def test_training_weights_subsets_answers_thresholds():
    p=[np.zeros((3,2)),np.zeros((5,2)),np.zeros((2,2)),np.zeros((1,2))]
    x,w=training_matrix(p,range(4),np.array(['a','a','b','b']),'hard','pb')
    np.testing.assert_allclose([sum(w[:2]),sum(w[2:6]),w[6]],[.25,.25,.5])

def test_pava_projection_and_invalid_fallback():
    np.testing.assert_allclose(pava([0,2,1,3]),[0,1.5,1.5,3])
    m=FusionModel('spectral',mean=np.zeros(2),scale=np.ones(2),weights=np.zeros(2),status='invalid_endpoint_range')
    mass,failed=location(m,np.array([[1.,0.],[0.,1.]]),'hard')
    assert failed;np.testing.assert_allclose(mass,[.5,.5])

def test_saved_scaling_and_binary_hierarchy():
    rng=np.random.default_rng(7);x=rng.normal(size=(200,11));w=rng.uniform(.1,1,200)
    m=fit_spectral(x,w,'continuous_lsml')
    z=(x-m.mean)/m.scale
    expected=np.column_stack([z[:,idx]@v for idx,v in m.inner])@m.cross*m.orientation
    np.testing.assert_allclose(m.predict(x),expected)
    b=2*(x>0)-1;m=fit_spectral(b,w,'binary_lsml')
    expected=np.column_stack([np.where(b[:,idx]@v>=0,1.,-1.) for idx,v in m.inner])@m.cross*m.orientation
    np.testing.assert_allclose(m.predict(b),expected)

def test_hierarchical_likelihood_matches_bruteforce():
    b=np.array(list(itertools.product([0.,1.],repeat=3)))
    groups=np.array([0,0,1]);t=np.array([[.2,.8],[.3,.9]]);e=np.array([[.1,.8],[.2,.9],[.1,.7]]);prior=.4
    lp,q,_=expectation(b,prior,e,groups,t)
    for i,row in enumerate(b):
        ys=[]
        for y in [0,1]:
            total=0.
            for a in itertools.product([0,1],repeat=2):
                p=np.prod([t[g,y] if a[g] else 1-t[g,y] for g in range(2)])
                p*=np.prod([e[j,a[groups[j]]] if row[j] else 1-e[j,a[groups[j]]] for j in range(3)])
                total+=p
            ys.append(total*(prior if y else 1-prior))
        np.testing.assert_allclose(np.exp(lp[i]),sum(ys));np.testing.assert_allclose(q[i],ys[1]/sum(ys))

@pytest.mark.parametrize('kind',['ds','hem'])
def test_em_monotonic_deterministic_and_compression(kind):
    rng=np.random.default_rng(4);truth=rng.choice([-1,1],200)
    x=np.column_stack([truth*rng.choice([-1,1],200,p=[.2,.8]) for _ in range(5)])
    x[:,1]=x[:,0] # dependent duplicates; no claim that consensus is reliability
    w=rng.uniform(.1,1,200);s=fit_spectral(x,w,'spectral');g=np.array([0,0,1,1,1])
    a=fit_em(x,w,kind,s,g,max_iter=150);b=fit_em(x,w,kind,s,g,max_iter=150)
    np.testing.assert_allclose(a.predict(x),b.predict(x),rtol=0,atol=0)
    for start in a.diagnostics['starts']: assert min(np.diff(start['log_likelihood']))>=-1e-9
    u,ww=compress(x,w)
    full=expectation((x>0).astype(float),a.prior,a.emissions,a.groups,a.transition)[0]
    unique=expectation(u,a.prior,a.emissions,a.groups,a.transition)[0]
    np.testing.assert_allclose(np.average(full,weights=w),ww@unique)

def test_fast_threshold_objective_matches_official_evaluator():
    from cvf_v2.scoring import grid_prmscore,prm
    meta=[{'idx':'a','error_steps':[2,4],'classification':'deception'},
          {'idx':'b','error_steps':[1],'classification':'circular'},
          {'idx':'c','error_steps':[],'classification':'correct'}]
    labels=np.array([0,1,0,1,1,0,0,0,0]);eligible=np.array([1]*7+[0]*2,bool)
    rows=np.vstack([np.ones(9,bool),np.zeros(9,bool),np.random.default_rng(3).random((20,9))>.5])
    fast=grid_prmscore(rows,labels,eligible)
    for i,row in enumerate(rows):
        pred=[{'idx':'a','labels':row[:4].astype(int).tolist()}, {'idx':'b','labels':row[4:7].astype(int).tolist()}, {'idx':'c','labels':row[7:].astype(int).tolist()}]
        t=prm.prmbench_evaluate(pred,meta)['total']
        np.testing.assert_allclose(fast[i],(t['f1']+t['negative_f1'])/2)
