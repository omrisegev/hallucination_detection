"""Scientific checks against analytic filters and exact latent enumerations."""
from pathlib import Path
import itertools
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fused_trajectory_readouts import (CORES, bocpd_filter, imm_filter,
                                                     hold_steps, score_readouts)
from spectral_utils.latent_state_localizer import forward_backward


class TemporalContracts(unittest.TestCase):
    def test_imm_identical_models_equal_ordinary_kalman(self):
        x = np.array([0., .5, 2., -.5, 1.])
        result = imm_filter(x, .4, [.07, .07], [[.9,.1],[.2,.8]])
        mean, var, expected = 0., 1., []
        for value in x:
            prior = var + .07
            gain = prior/(prior+.4)
            mean += gain*(value-mean)
            var = (1-gain)*prior
            expected.append(mean)
        np.testing.assert_allclose(result['level'], expected, atol=1e-12)
        self.assertTrue(np.all(result['variance'] > 0))
        np.testing.assert_allclose(result['mode_probability'].sum(axis=1), 1.)

    def test_imm_mixing_includes_between_mean_variance(self):
        # Explicit two-step interaction independently assembled from first posterior.
        transition = np.array([[.8,.2],[.3,.7]])
        r, q = .4, np.array([.01,1.])
        prior_var = 1+q
        gain = prior_var/(prior_var+r)
        means, variances = 2*gain, (1-gain)*prior_var
        likelihood = np.exp(-.5*(np.log(2*np.pi*(prior_var+r))+4/(prior_var+r)))
        modes = (np.array([.5,.5])@transition)*likelihood
        modes /= modes.sum()
        cbar = modes @ transition
        mixed, mixed_var = [], []
        for j in range(2):
            weights = modes*transition[:,j]/cbar[j]
            m = sum(weights[i]*means[i] for i in range(2))
            p = sum(weights[i]*(variances[i]+(means[i]-m)**2) for i in range(2))
            mixed.append(m); mixed_var.append(p)
        mixed, mixed_var = np.array(mixed), np.array(mixed_var)
        prior = mixed_var+q
        gain = prior/(prior+r)
        new_mean = mixed+gain*(-1-mixed)
        likelihood = np.exp(-.5*(np.log(2*np.pi*(prior+r))+(-1-mixed)**2/(prior+r)))
        weights = cbar*likelihood; weights/=weights.sum()
        actual = imm_filter([2.,-1.], r, q, transition)
        self.assertAlmostEqual(actual['level'][1], float(weights@new_mean), places=12)

    def test_bocpd_matches_every_short_partition(self):
        x = np.array([-.2,.1,2.,2.2,-.5])
        hazard, r, prior = .17, .7, 1.3
        result = bocpd_filter(x, hazard, r, prior_variance=prior)
        for length in range(2, len(x)+1):
            logmass, reset_flags, final_means = [], [], []
            for flags in itertools.product([0,1], repeat=length-1):
                boundaries = [0]+[i+1 for i,f in enumerate(flags) if f]+[length]
                lp = sum(np.log(hazard if f else 1-hazard) for f in flags)
                for a,b in zip(boundaries[:-1],boundaries[1:]):
                    block=x[a:b]
                    cov = r*np.eye(len(block))+prior*np.ones((len(block),len(block)))
                    lp += -.5*(len(block)*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+block@np.linalg.solve(cov,block))
                tail=x[boundaries[-2]:length]
                final_means.append(float(tail.sum()/(r/prior+len(tail))))
                logmass.append(lp); reset_flags.append(flags[-1])
            weights=np.exp(logmass-logsumexp(logmass))
            self.assertAlmostEqual(result['reset_probability'][length-1],float(weights@reset_flags),places=12)
            self.assertAlmostEqual(result['level'][length-1],float(weights@final_means),places=12)
        self.assertGreater(np.ptp(result['reset_probability'][1:]), .01)

    def test_temporal_filter_prefix_invariance_given_fixed_parameters(self):
        x=[.2,-.4,2.,1.,5.]
        for function,args in [(bocpd_filter,()),(imm_filter,(.5,[.01,.5]))]:
            np.testing.assert_allclose(function(x,*args)['level'][:3],function(x[:3],*args)['level'])

    def test_hmm_posteriors_equal_state_path_enumeration(self):
        values=np.array([-.8,.1,1.1]); means=np.array([-1.,1.]); variance=.7
        transition=np.array([[.8,.2],[.1,.9]]); start=np.array([.4,.6])
        gamma,xi,ll=forward_backward(values,means,variance,transition,start)
        paths=list(itertools.product([0,1],repeat=3)); weights=[]
        for path in paths:
            logw=np.log(start[path[0]])
            for i,state in enumerate(path):
                logw+=-.5*(np.log(2*np.pi*variance)+(values[i]-means[state])**2/variance)
                if i: logw+=np.log(transition[path[i-1],state])
            weights.append(logw)
        self.assertAlmostEqual(ll,float(logsumexp(weights)),places=12)
        weights=np.exp(weights-logsumexp(weights))
        expected=np.zeros_like(gamma); expected_xi=np.zeros_like(xi)
        for path,w in zip(paths,weights):
            for i,state in enumerate(path):
                expected[i,state]+=w
                if i: expected_xi[i-1,path[i-1],state]+=w
        np.testing.assert_allclose(gamma,expected,atol=1e-12)
        np.testing.assert_allclose(xi,expected_xi,atol=1e-12)

    def test_hold_mapping_covers_tail_and_boundary_spans(self):
        np.testing.assert_array_equal(hold_steps([1.,3.],8,19,[0,7,16],[7,12,19]),[1.,3.,3.])
        with self.assertRaises(ValueError): hold_steps([1.,3.],8,19,[0],[20])

    def test_readout_cannot_change_parent_clean_gate_or_credit_failed_fit(self):
        arrays={'step_starts':np.array([0,32]),'step_ends':np.array([32,64]),
                'moments27_local8__fit_indices':np.arange(8)}
        methods={}
        for core in CORES:
            arrays[core+'__window']=np.arange(8,dtype=float)
            arrays[core+'__step']=np.array([3.,7.])
            methods[core]={'valid':True,'readout_valid':True,'readout':{'prediction':-1}}
        curves={'hold_peak':{'valid':True,'risk':np.arange(8.),'onset':np.arange(8.),'detail':{}},
                'hmm_entry':{'valid':False,'detail':{},'reason':'UNSTABLE'}}
        with patch('spectral_utils.fused_trajectory_readouts.temporal_curves',return_value=curves):
            _,details=score_readouts(arrays,{'tokens':64,'report':{'methods':methods}})
        for core in CORES:
            for name in ('parent_first','parent_peak','hold_peak'):
                self.assertEqual(details[core+'@@'+name]['prediction'],-1)
            self.assertFalse(details[core+'@@hmm_entry']['valid'])
            self.assertNotIn('prediction',details[core+'@@hmm_entry'])


if __name__=='__main__': unittest.main()
