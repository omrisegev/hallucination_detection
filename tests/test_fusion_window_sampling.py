"""Sampling axis, fit-scope and replay checks for a scientific intervention."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '1'
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils import fusion_window_sampling as sample
from spectral_utils.answer_localization_v2 import prepare_local


class SamplingContract(unittest.TestCase):
    def test_budget_and_selection_uniqueness(self):
        for n in (8, 32, 33, 63, 64, 65, 256):
            m = sample.budget(n)
            self.assertTrue(1 <= m <= n)
            if n > 32: self.assertTrue(32 <= m < n)
            indices = np.rint(np.linspace(0, n-1, m)).astype(int)
            self.assertEqual(len(set(indices)), m)
        np.testing.assert_array_equal(sample.top_indices(np.ones(40), 32), np.arange(32))

    def test_transposed_axis_really_gates_windows(self):
        x = np.random.default_rng(82).normal(size=(41, 9))
        def fake_dufs(features, **kwargs):
            self.assertEqual(features.shape, (41, 9))
            np.testing.assert_allclose(features.mean(axis=1), 0, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1)
            return np.ones(41), {'raw_probabilities': np.linspace(0, 1, 41)}
        with patch.object(sample, 'adapted_dufs_soft_gates', fake_dufs):
            p, _ = sample.transposed_gates(x)
        self.assertEqual(len(p), 41)

    def test_window_graph_permutation_equivariance_and_representatives(self):
        rng = np.random.default_rng(159)
        x = rng.normal(size=(49, 8)); permutation = rng.permutation(49)
        indices, details = sample.diffusion_indices(x, 32)
        perm_indices, _ = sample.diffusion_indices(x[permutation], 32)
        self.assertEqual(details['nodes'], 49)
        np.testing.assert_array_equal(indices, np.sort(permutation[perm_indices]))
        np.testing.assert_array_equal(indices, sample.diffusion_indices(x, 32)[0])
        self.assertEqual(len(set(indices)), 32)

    def test_fusion_fit_ignores_unselected_values_once_selection_is_fixed(self):
        rng = np.random.default_rng(25)
        x = rng.normal(size=(70, 9)); selected = np.arange(0, 64, 2)
        changed = x.copy(); mask = np.ones(70, bool); mask[selected] = False
        changed[mask] = rng.normal(200, 50, size=(mask.sum(), 9))
        names = ['entropy_series__level']+[str(i) for i in range(8)]
        a, _, da = prepare_local(x, names, selected)
        b, _, db = prepare_local(changed, names, selected)
        np.testing.assert_array_equal(a[selected], b[selected])
        for field in ('mean', 'sd', 'feature_signs'): np.testing.assert_array_equal(da[field], db[field])

    def test_bootstrap_keeps_two_token_blocks_inside_original_window(self):
        raw = np.column_stack((np.arange(83), np.arange(83)*100))
        changed = sample.perturb_windows(raw, 'test', 0)
        np.testing.assert_array_equal(changed, sample.perturb_windows(raw, 'test', 0))
        np.testing.assert_array_equal(changed[80:], raw[80:])
        for lo in range(0, 80, 8):
            self.assertTrue(np.all((changed[lo:lo+8, 0] >= lo) & (changed[lo:lo+8, 0] < lo+8)))
            np.testing.assert_array_equal(np.diff(changed[lo:lo+8, 0].reshape(4, 2), axis=1), np.ones((4, 1)))
            np.testing.assert_array_equal(changed[lo:lo+8, 1], 100*changed[lo:lo+8, 0])

    def test_short_answer_is_exact_parent_peak_without_refitting(self):
        parent = ROOT/'results/answer_localization_representation_pilot_v1'
        record = next(r for r in json.loads((parent/'PREPARED.json').read_text())['selected'] if r['tokens'] < 256)
        uid = record['uid']; meta = json.loads((parent/'scores'/f'{uid}.json').read_text())
        with np.load(parent/'inputs'/f'{uid}.npz') as data: raw = data['raw']
        with np.load(parent/'scores'/f'{uid}.npz') as z, patch.object(sample, 'fit_local', side_effect=AssertionError('unexpected fit')):
            output, methods, detail = sample.score_sampling(z, meta, raw, 'test')
            for core in sample.CORES:
                source = meta['report']['methods'][core]
                valid = bool(source.get('valid') and source.get('readout_valid'))
                for selector in sample.SELECTORS:
                    arm = core+'@@'+selector
                    self.assertEqual(methods[arm]['valid'], valid)
                    if valid:
                        np.testing.assert_array_equal(output[arm+'__risk'], z[core+'__step'])
                        prediction = int(np.argmax(z[core+'__step'])) if source['readout']['prediction'] != -1 else -1
                        self.assertEqual(methods[arm]['prediction'], prediction)
            self.assertFalse(detail['sampling_eligible'])


if __name__ == '__main__': unittest.main()
