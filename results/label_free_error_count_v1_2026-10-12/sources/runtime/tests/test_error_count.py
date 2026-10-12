"""Synthetic estimator checks; these fixtures contain no project data."""
import inspect
from itertools import product
import json
import unittest
from unittest.mock import patch

import numpy as np

from spectral_utils import error_count as ec


def independent_votes(n=30000, seed=71):
    rng = np.random.default_rng(seed)
    psi = np.array([.88, .83, .79, .75])
    eta = np.array([.93, .91, .89, .87])
    prevalence = .23
    truth = rng.random(n) < prevalence
    probabilities = np.where(truth[:, None], psi, 1 - eta)
    votes = np.where(rng.random((n, len(psi))) < probabilities, 1, -1)
    return votes, psi, eta, prevalence


def emission_model():
    return dict(status='PASS', promotable=True, n_vote_columns=3, columns=[0, 1, 2],
                psi=[.91, .85, .80], eta=[.95, .91, .86], prevalence=.2)


def answer_votes(clean=.35, theta=.27, answers=1100, n=8, seed=82):
    rng = np.random.default_rng(seed)
    model = emission_model()
    structurally_clean = rng.random(answers) < clean
    truth = rng.random((answers, n)) < theta
    truth[structurally_clean] = False
    chance = np.where(truth.reshape(-1, 1), np.array(model['psi']), 1 - np.array(model['eta']))
    votes = np.where(rng.random(chance.shape) < chance, 1, -1)
    return votes, np.arange(0, answers * n + 1, n), model


class CountDistributionTests(unittest.TestCase):
    def test_brute_force_and_decision_losses(self):
        probabilities = np.array([.17, .38, .71, .92])
        expected = np.zeros(len(probabilities) + 1)
        for bits in product((0, 1), repeat=len(probabilities)):
            probability = np.prod(np.where(np.array(bits), probabilities, 1 - probabilities))
            expected[sum(bits)] += probability
        result = ec.count_distribution(probabilities, np.array([0, 4]))
        np.testing.assert_allclose(result['pmf'], expected, rtol=1e-13, atol=1e-15)
        support = np.arange(5)
        absolute = [np.sum(expected * np.abs(support - k)) for k in support]
        squared = [np.sum(expected * (support - k) ** 2) for k in support]
        self.assertEqual(result['median_lower'][0], int(np.argmin(absolute)))
        self.assertAlmostEqual(squared[result['mean_half_up'][0]], min(squared))

    def test_lower_median_and_half_up_are_distinct_tie_rules(self):
        result = ec.count_distribution(np.array([.5]), np.array([0, 1]))
        self.assertEqual(result['median_lower'], [0])
        self.assertEqual(result['mean_half_up'], [1])

    def test_multiple_answers_empty_and_deterministic(self):
        result = ec.count_distribution(np.array([0., 1., .5]), np.array([0, 0, 2, 3]))
        self.assertEqual(result['pmf_offsets'], [0, 1, 4, 6])
        self.assertEqual(result['mean_half_up'], [0, 1, 1])
        self.assertEqual(result['median_lower'], [0, 1, 0])
        json.dumps(result, allow_nan=False)

    def test_invalid_probabilities_and_offsets_are_not_clipped(self):
        with self.assertRaises(ValueError):
            ec.count_distribution(np.array([-.001]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            ec.count_distribution(np.array([.2]), np.array([0, 2]))
        with self.assertRaises(ValueError):
            ec.count_distribution(np.array([.2]), np.array([0., 1.]))


class MomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.votes, cls.psi, cls.eta, cls.prevalence = independent_votes()

    def test_planted_all_group_recovery(self):
        model = ec.fit_mom(self.votes)
        self.assertEqual(model['status'], 'PASS', model)
        self.assertTrue(model['rank_one_completion']['converged'])
        self.assertLessEqual(model['rank_one_completion']['iterations'], 2000)
        np.testing.assert_allclose(model['psi'], self.psi, atol=.025)
        np.testing.assert_allclose(model['eta'], self.eta, atol=.015)
        self.assertAlmostEqual(model['prevalence'], self.prevalence, delta=.015)
        self.assertEqual(model['out_of_range'], 0)
        json.dumps(model, allow_nan=False)

    def test_constant_and_dependent_boundary_fail_explicitly(self):
        constant = -np.ones((100, 4))
        self.assertEqual(ec.fit_mom(constant)['status'], 'FAILED')
        copied = np.repeat(np.where(np.arange(400) < 100, 1, -1)[:, None], 4, axis=1)
        strict = ec.fit_mom(copied)
        self.assertEqual(strict['status'], 'FAILED')
        legacy = ec.fit_mom(copied, legacy_clip=True)
        self.assertEqual(legacy['status'], 'PASS', legacy)
        self.assertEqual(legacy['out_of_range'], 4)
        np.testing.assert_allclose(legacy['psi_raw'], 1., atol=1e-12)
        with self.assertRaises(ValueError):
            ec.posterior(copied, strict)

    def test_nonconvergence_is_not_a_usable_estimate(self):
        with patch.object(ec, 'MOM_MAX_ITER', 1):
            model = ec.fit_mom(self.votes)
        self.assertEqual(model['status'], 'FAILED')
        self.assertEqual(model['failure_reason'], 'RANK_ONE_NONCONVERGENCE')

    def test_fixed_triple_reweighting_matches_explicit_replication(self):
        votes = self.votes[:300]
        weight = np.tile([1, 2, 3], 100)
        weighted = ec.fit_fixed_triple(votes, [0, 1, 2], weights=weight)
        expanded = ec.fit_fixed_triple(np.repeat(votes, weight, axis=0), [0, 1, 2])
        self.assertEqual(weighted['status'], expanded['status'])
        for key in ('prevalence', 'psi_raw', 'eta_raw'):
            np.testing.assert_allclose(weighted[key], expanded[key], atol=1e-12)

    def test_archived_posterior_accepted_but_invalid_parameters_rejected(self):
        archived = dict(psi=self.psi, eta=self.eta, prevalence=self.prevalence)
        p = ec.posterior(self.votes[:50], archived)
        self.assertTrue(((p > 0) & (p < 1)).all())
        archived['psi'] = np.zeros(4)
        with self.assertRaises(ValueError):
            ec.posterior(self.votes[:50], archived)

    def test_latent_label_swap_complements_posterior_without_changing_votes(self):
        original = dict(psi=self.psi, eta=self.eta, prevalence=self.prevalence)
        swapped = dict(psi=1 - self.eta, eta=1 - self.psi, prevalence=1 - self.prevalence)
        probability = ec.posterior(self.votes[:100], original)
        complement = ec.posterior(self.votes[:100], swapped)
        np.testing.assert_allclose(complement, 1 - probability, atol=1e-14)

    def test_no_label_parameter_in_fitting_interfaces(self):
        for helper in (ec.fit_mom, ec.fit_fixed_triple, ec.fit_stable_triple, ec.fit_zero_mixture):
            parameters = inspect.signature(helper).parameters
            self.assertNotIn('y', parameters)
            self.assertNotIn('labels', parameters)


class StableTripleTests(unittest.TestCase):
    def test_source_block_bootstrap_and_seeded_selection(self):
        votes, _, _, _ = independent_votes(n=22000)
        counts = np.arange(150, 230)
        groups = np.repeat(np.arange(len(counts)), counts)
        votes = votes[:len(groups)]
        members = [[5], [2, 3], [8], [0, 1]]
        first = ec.fit_stable_triple(votes, groups, members)
        second = ec.fit_stable_triple(votes, groups, members)
        self.assertEqual(first['status'], 'PASS', first)
        self.assertEqual(first, second)
        metadata = first['bootstrap']
        multiplicity = np.random.default_rng(20261012).multinomial(80, np.full(80, 1 / 80), size=200)
        np.testing.assert_array_equal(metadata['first_draw_multiplicity'], multiplicity[0])
        np.testing.assert_array_equal(metadata['resampled_step_counts'], multiplicity @ counts)
        self.assertGreater(np.std(metadata['resampled_step_counts']), 0)
        eligible = [row for row in first['triple_table'] if row['eligible']]
        self.assertTrue(all(row['bootstrap_valid'] >= 180 for row in eligible))
        p = ec.posterior(votes, first)
        selected = ec.fit_fixed_triple(votes, first['columns'])
        np.testing.assert_allclose(p, ec.posterior(votes, selected))
        json.dumps(first, allow_nan=False)

    def test_insufficient_sources_and_no_eligible_triple(self):
        votes, _, _, _ = independent_votes(n=100)
        one = ec.fit_stable_triple(votes, np.zeros(100), [[0], [1], [2], [3]])
        self.assertEqual(one['failure_reason'], 'INSUFFICIENT_SOURCE_GROUPS')
        constant = ec.fit_stable_triple(-np.ones((100, 3)), np.repeat(np.arange(10), 10), [[0], [1], [2]])
        self.assertEqual(constant['failure_reason'], 'NO_ELIGIBLE_TRIPLE')
        self.assertFalse(constant['triple_table'][0]['eligible'])
        json.dumps(constant, allow_nan=False)

    def test_canonical_membership_tie_is_independent_of_column_order(self):
        votes, _, _, _ = independent_votes(n=8000)
        groups = np.repeat(np.arange(80), 100)
        members = [[5], [2, 3], [8], [0, 1]]
        order = np.array([3, 1, 0, 2])
        first = ec.fit_stable_triple(votes, groups, members)
        permuted = ec.fit_stable_triple(votes[:, order], groups, [members[j] for j in order])
        self.assertEqual(first['status'], 'PASS')
        self.assertEqual(permuted['status'], 'PASS')
        first_members = {tuple(members[j]) for j in first['columns']}
        other_members = {tuple(members[order[j]]) for j in permuted['columns']}
        self.assertEqual(first_members, other_members)


class ZeroMixtureTests(unittest.TestCase):
    def test_planted_answer_mixture_recovery_and_count_distribution(self):
        votes, offsets, triple = answer_votes()
        model = ec.fit_zero_mixture(votes, offsets, triple)
        self.assertEqual(model['status'], 'PASS', model)
        self.assertAlmostEqual(model['rho'], .35, delta=.07)
        self.assertAlmostEqual(model['theta'], .27, delta=.05)
        self.assertTrue(model['emissions_frozen'])
        self.assertEqual(len(model['starts']), 6)
        self.assertTrue(model['hessian']['healthy'])
        result = ec.predict_zero_mixture(votes, offsets, model)
        q = np.asarray(result['structural_clean_p'])
        active = np.asarray(result['active_p'])
        marginal = np.asarray(result['marginal_p'])
        np.testing.assert_allclose(marginal, active * np.repeat(1 - q, np.diff(offsets)))
        first_pmf = np.asarray(result['pmf'])[result['pmf_offsets'][0]:result['pmf_offsets'][1]]
        expected_p0 = q[0] + (1 - q[0]) * np.prod(1 - active[:8])
        self.assertAlmostEqual(first_pmf[0], expected_p0)
        self.assertGreater(first_pmf[0], q[0])
        np.testing.assert_allclose(result['mean'], np.add.reduceat(marginal, offsets[:-1]))
        json.dumps(model, allow_nan=False)
        json.dumps(result, allow_nan=False)

    def test_ordinary_boundary_and_four_argument_prediction(self):
        votes, offsets, triple = answer_votes(clean=0., answers=600, seed=107)
        zero = ec.fit_zero_mixture(votes, offsets, triple)
        boundary = zero['boundary_model']
        self.assertEqual(boundary['status'], 'PASS', boundary)
        prediction = ec.predict_zero_mixture(votes, offsets, triple, boundary)
        self.assertEqual(prediction['structural_clean_p'], [0.] * 600)
        standard = ec.count_distribution(np.array(prediction['active_p']), offsets)
        for key in ('pmf', 'mean', 'mean_half_up', 'median_lower'):
            np.testing.assert_allclose(prediction[key], standard[key])
        changed = dict(boundary, psi=[.6, .6, .6])
        with self.assertRaises(ValueError):
            ec.predict_zero_mixture(votes, offsets, triple, changed)

    def test_dependency_and_single_step_failures(self):
        votes, offsets, triple = answer_votes(answers=10)
        failed = ec.fit_zero_mixture(votes, offsets, dict(status='FAILED', failure_reason='NO_ELIGIBLE_TRIPLE'))
        self.assertEqual(failed['failure_reason'], 'FAILED_DEPENDENCY')
        single = ec.fit_zero_mixture(votes, np.arange(len(votes) + 1), triple)
        self.assertEqual(single['failure_reason'], 'SINGLE_STEP_ANSWER_NONIDENTIFICATION')

    def test_all_clean_boundary_is_explicit_failure(self):
        votes = -np.ones((160, 3))
        model = ec.fit_zero_mixture(votes, np.arange(0, 161, 8), emission_model())
        self.assertEqual(model['status'], 'FAILED')
        self.assertEqual(model['failure_reason'], 'DEGENERATE_ALL_CLEAN')

    def test_competing_optima_and_bad_hessian_are_not_promoted(self):
        votes, offsets, triple = answer_votes(answers=50)
        with patch.object(ec, '_zero_objective', return_value=lambda parameters: (0., np.zeros(2))):
            ambiguous = ec.fit_zero_mixture(votes, offsets, triple)
        self.assertEqual(ambiguous['status'], 'FAILED')
        self.assertEqual(ambiguous['failure_reason'], 'COMPETING_NEAR_OPTIMA')
        votes, offsets, triple = answer_votes(answers=700)
        with patch.object(ec, '_zero_hessian', return_value=dict(available=True, healthy=False, condition=1e10)):
            weak = ec.fit_zero_mixture(votes, offsets, triple)
        self.assertEqual(weak['status'], 'FAILED')
        self.assertEqual(weak['failure_reason'], 'WEAK_IDENTIFICATION')


if __name__ == '__main__':
    unittest.main()
