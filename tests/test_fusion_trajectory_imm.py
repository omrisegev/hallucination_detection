import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import logsumexp
from spectral_utils.fused_trajectory_readouts import imm_filter
from spectral_utils import fusion_trajectory_imm as t


def vector_imm(y,R,q):
    """Direct multivariate measurement equations; no scalar reduction."""
    y=np.asarray(y);p=y.shape[1];H=np.ones(p);means=np.zeros(2);variances=np.ones(2);prob=np.ones(2)/2
    transition=np.array([[.95,.05],[.05,.95]]);levels=[];var=[];modes=[]
    for observation in y:
        prior_modes=prob@transition;mix=prob[:,None]*transition/prior_modes
        mu=means@mix;pv=np.array([sum(mix[i,j]*(variances[i]+(means[i]-mu[j])**2) for i in range(2))+q[j] for j in range(2)])
        logs=[]
        for j in range(2):
            S=pv[j]*np.outer(H,H)+R;innovation=observation-H*mu[j];sgn,det=np.linalg.slogdet(S);assert sgn>0
            gain=pv[j]*np.linalg.solve(S,H)
            means[j]=mu[j]+gain@innovation
            variances[j]=(1-gain@H)**2*pv[j]+gain@R@gain
            logs.append(-.5*(p*np.log(2*np.pi)+det+innovation@np.linalg.solve(S,innovation)))
        mass=np.log(prior_modes)+logs;prob=np.exp(mass-logsumexp(mass));level=prob@means
        levels.append(level);var.append(prob@(variances+(means-level)**2));modes.append(prob.copy())
    return dict(level=np.array(levels),variance=np.array(var),mode_probability=np.array(modes))


class TrajectoryIMMTests(unittest.TestCase):
    def standardized(self,n=48):
        x=np.random.default_rng(14).normal(size=(n,2));x[:,1]+=.6*x[:,0];return (x-x.mean(0))/x.std(0)

    def test_vector_scalar_equivalence(self):
        y=self.standardized();fi=np.arange(len(y));gls,_,m=t.observation_model(y,fi)
        R=m['regularized_noise']/m['gls_scale']['sd']**2
        v=(y[:,m['retained']]-m['gls_scale']['mean'])/m['gls_scale']['sd'];r=m['effective_variance']
        expected=vector_imm(v,R,[.01*r,r]);actual=imm_filter(gls,r,[.01*r,r])
        for k in expected:np.testing.assert_allclose(actual[k],expected[k],atol=1e-11,rtol=1e-11)

    def test_duplicate_is_one_observation_and_opposite_fails(self):
        y=self.standardized()[:,0];fi=np.arange(len(y))
        g,_,a=t.observation_model(y[:,None],fi);g2,_,b=t.observation_model(np.column_stack((y,y)),fi)
        np.testing.assert_array_equal(g,g2);self.assertEqual(b['retained'],[0]);self.assertEqual(a['effective_variance'],b['effective_variance'])
        with self.assertRaisesRegex(ValueError,'CONFLICTING'):t.observation_model(np.column_stack((y,-y)),fi)

    def test_noise_covariance_is_positive_and_weights_normalized(self):
        y=self.standardized();y[:,1]=y[:,0]+1e-4*np.arange(len(y));y=(y-y.mean(0))/y.std(0)
        _,_,m=t.observation_model(y,np.arange(len(y)))
        self.assertGreater(np.linalg.eigvalsh(m['regularized_noise']).min(),0)
        self.assertLessEqual(m['condition'],100+1e-8);self.assertAlmostEqual(float(sum(m['weights'])),1)

    def test_tail_is_not_an_additional_measurement(self):
        np.testing.assert_array_equal(t.hold_curve(np.array([1.,2.,3.]),np.arange(3),4),[1,2,3,3])
        with self.assertRaises(ValueError):t.hold_curve(np.array([1.,2.]),np.array([0,2]),3)
        with self.assertRaises(ValueError):t.hold_curve(np.array([1.,2.]),np.arange(2),4)

    def test_permutation_reassigned_to_original_positions(self):
        y=self.standardized()[:,0];fi=np.arange(len(y));level,state,info=t.chronology(y,.3,fi,'fixture',True)
        expected=imm_filter(y[info['permutation']],.3,[.003,.3])
        np.testing.assert_array_equal(state['level'][info['permutation']],expected['level'])
        self.assertAlmostEqual(float(level.mean()),0);self.assertAlmostEqual(float(level.std()),1)

    def fixture(self):
        count=67;plan=t.moment_plan(count,8);fi=plan.fit_indices;source={};meta={'methods':{}}
        rng=np.random.default_rng(77)
        for name in t.SOURCES.values():
            x=rng.normal(size=len(plan.starts));x=(x-x[fi].mean())/x[fi].std()
            source[name+'__window']=x;meta['methods'][name]={'valid':True}
        original={'routing':{'routes':{'dual':'moment_joint'}},'methods':{'moment__iu':{'valid':True,'decision_valid':True,'prediction':-1}}}
        return count,np.array([0,32]),np.array([32,67]),source,meta,original,'fixture'

    def test_source_failure_is_not_silently_dropped(self):
        args=self.fixture();args[4]['methods'][t.SOURCES['joint_graph']]['valid']=False
        with patch('spectral_utils.fusion_token_gap.mixture_readout',return_value={'prediction':0}):_,methods,_=t.score_trajectories(*args)
        self.assertFalse(methods['traj_iu_joint_graph__mean']['valid']);self.assertTrue(methods['traj_iu__hold']['valid'])

    def test_imm_failure_does_not_invalidate_static_controls(self):
        with patch.object(t,'chronology',side_effect=ValueError('IMM_FAILED')),patch('spectral_utils.fusion_token_gap.mixture_readout',return_value={'prediction':0}):
            _,methods,_=t.score_trajectories(*self.fixture())
        self.assertTrue(methods['traj_iu_joint_graph__gls']['valid']);self.assertFalse(methods['traj_iu_joint_graph__imm']['valid'])


if __name__=='__main__':unittest.main()
