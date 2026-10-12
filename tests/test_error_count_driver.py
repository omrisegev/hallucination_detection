"""Synthetic checks for split/rank and immutable-resume driver boundaries."""
from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('error_count_driver_under_test',
                                             _ROOT/'scripts/run_label_free_error_count_v1.py')
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class DriverBoundaryTests(unittest.TestCase):
    def test_stable_rank_is_complete_within_each_answer(self):
        score = np.array([2., 2., -4., 1., 0., 1., 1.])
        off = np.array([0, 3, 7])
        rank = driver.rank_per_answer(score, off)
        np.testing.assert_array_equal(rank, [0, 1, 2, 3, 5, 6, 4])
        for a, b in zip(off[:-1], off[1:]):
            np.testing.assert_array_equal(np.sort(rank[a:b]), np.arange(a, b))

    def test_count_flags_preserves_zero_one_and_full_counts(self):
        flags = np.array([False, False, True, True, True, True])
        np.testing.assert_array_equal(driver.count_flags(flags, [0, 2, 3, 6]), [0, 1, 3])

    def test_missing_rank_is_explicit_minus_one_and_partial_nonfinite_rejects(self):
        score = np.array([np.nan, np.nan, 2., 1.])
        np.testing.assert_array_equal(driver.rank_per_answer(score, [0, 2, 4]), [-1, -1, 2, 3])
        with self.assertRaises((ValueError, RuntimeError)):
            driver.rank_per_answer(np.array([np.nan, 1.]), [0, 2])

    def test_cut_answers_preserves_explicit_order_and_whole_blocks(self):
        data = {'offsets': np.array([0, 2, 5, 6])}
        rows, off = driver.cut_answers(data, np.array([2, 0]))
        np.testing.assert_array_equal(rows, [5, 0, 1])
        np.testing.assert_array_equal(off, [0, 1, 3])

    def test_cut_answers_rejects_duplicate_negative_empty_and_invalid_indices(self):
        data = {'offsets': np.array([0, 2, 5, 6])}
        for answers in [np.array([0, 0]), np.array([-1]), np.array([3]),
                        np.array([], dtype=int), np.array([.5]), np.array([True])]:
            with self.subTest(answers=answers.tolist()):
                with self.assertRaises((ValueError, IndexError)):
                    driver.cut_answers(data, answers)

    def test_cut_answers_rejects_malformed_offsets(self):
        for off in [[1, 3, 4], [0, 0, 4], [0, 3, 2], [0., 2., 4.]]:
            with self.subTest(offsets=off):
                with self.assertRaises(ValueError):
                    driver.cut_answers({'offsets': np.array(off)}, np.array([0]))

    def _identity_fixture(self, root):
        out = root/'out'
        out.mkdir()
        (out/'PROTOCOL.json').write_text('{"fixture": true}\n')
        (out/'AMENDMENTS.json').write_text('[]\n')
        (root/'input.dat').write_text('unchanged donor input\n')
        code_paths = ['scripts/run_label_free_error_count_v1.py', 'spectral_utils/error_count.py',
                      'spectral_utils/error_count_inputs.py', 'spectral_utils/error_count_evaluation.py',
                      'scripts/evaluate_label_free_error_count_v1.py']
        for name in code_paths:
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic identity fixture\n')
        freeze = {'protocol_sha256': driver.sha(out/'PROTOCOL.json'),
                  'inputs': {'input.dat': {'sha256': driver.sha(root/'input.dat'), 'size': (root/'input.dat').stat().st_size}},
                  'python': sys.version, 'executable': sys.executable,
                  'versions': {key: importlib.metadata.version(key)
                               for key in ['numpy', 'scipy', 'scikit-learn', 'matplotlib']}}
        (out/'INPUT_FREEZE.json').write_text(json.dumps(freeze))
        return out, freeze

    def test_resume_rejects_changed_input_protocol_and_code_without_resealing(self):
        for changed in ['input.dat', 'out/PROTOCOL.json', 'spectral_utils/error_count_inputs.py']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out, _ = self._identity_fixture(root)
                with patch.object(driver, 'ROOT', root), patch.object(driver, 'OUT', out):
                    driver.validate_identity()
                    seal = (out/'CODE_FREEZE.json').read_bytes()
                    (root/changed).write_text('changed fixture\n')
                    with self.assertRaises((RuntimeError, AssertionError)):
                        driver.validate_identity()
                    self.assertEqual(seal, (out/'CODE_FREEZE.json').read_bytes())

    def test_resume_rejects_current_runtime_identity_drift(self):
        for field in ['python', 'executable', 'versions']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out, freeze = self._identity_fixture(root)
                if field == 'versions':
                    freeze['versions']['numpy'] = 'unavailable-fixture-version'
                else:
                    freeze[field] += '-different-runtime'
                (out/'INPUT_FREEZE.json').write_text(json.dumps(freeze))
                with patch.object(driver, 'ROOT', root), patch.object(driver, 'OUT', out):
                    with self.assertRaises((RuntimeError, AssertionError)):
                        driver.validate_identity()

    def _checkpoint_fixture(self, root):
        out, frozen = root/'out', root/'frozen'
        (out/'checkpoints').mkdir(parents=True)
        (out/'predictions').mkdir()
        (frozen/'inputs').mkdir(parents=True)
        (out/'INPUT_FREEZE.json').write_text('{"synthetic": true}\n')
        (out/'PROTOCOL.json').write_text('{"synthetic": true}\n')
        (out/'AMENDMENTS.json').write_text('[]\n')
        data = {'idx': np.array(['answer'+str(i) for i in range(5)]), 'offsets': np.arange(0, 11, 2),
                'group': np.array(['source'+str(i) for i in range(5)]), 'fold': np.arange(5),
                'entropy': np.arange(5, dtype=float), 'digit_active': np.ones((10, 3), dtype=bool)}
        names = driver.ARMS + ['Q80', 'R2_COUNT', 'ENTROPY_GATE_Q80', 'LEGACY_MOM', 'D_PRIOR_MEAN', 'D_PRIOR_MEDIAN']
        code = {'synthetic_code.py': 'fixture-code-hash'}
        for bank in ['N16', 'N56']:
            np.savez(frozen/f'inputs/bank_V_{bank}.npz', raw=np.arange(30).reshape(10, 3),
                     V=np.arange(30).reshape(10, 3), columns=np.array(['x', 'y', 'z']))
            for fold in range(5):
                folder = out/'checkpoints'/f'{bank}_fold{fold}'
                folder.mkdir()
                local = {'idx': data['idx'][fold:fold+1], 'offsets': np.array([0, 2]),
                         'IU_PCR_score': np.array([0., 1.]), 'LSML_CONT_score': np.array([1., 1.])}
                for name in names:
                    local['count__'+name] = np.zeros(1, dtype=np.int64)
                    local['posterior__'+name] = np.full(2, np.nan)
                    for method in ['IU_PCR', 'LSML_CONT']:
                        local[f'count__{method}__{name}'] = np.zeros(1, dtype=np.int64)
                np.savez(folder/'predictions.npz', **local)
                (folder/'models.json').write_text('{"synthetic": true}\n')
                ident = {'bank': bank, 'fold': fold, 'input_freeze': driver.sha(out/'INPUT_FREEZE.json'),
                         'code': code, 'protocol': driver.sha(out/'PROTOCOL.json'),
                         'amendments': driver.sha(out/'AMENDMENTS.json')}
                receipt = {'identity': ident, 'predictions_sha256': driver.sha(folder/'predictions.npz'),
                           'models_sha256': driver.sha(folder/'models.json')}
                (folder/'COMPLETE.json').write_text(json.dumps(receipt))
        return out, frozen, data, code

    def _run_resume(self, out, frozen, data, code):
        with patch.object(driver, 'OUT', out), patch.object(driver, 'FROZEN', frozen), \
                patch.object(driver, 'validate_identity', return_value=code), \
                patch.object(driver, 'load_unlabeled', return_value=data), \
                patch.object(driver, 'archived_estimators', return_value=(None, None)), \
                patch('spectral_utils.advisor_lr.fit_labelfree', side_effect=AssertionError('Resume must not fit')), \
                contextlib.redirect_stdout(io.StringIO()):
            driver.predict()

    def test_complete_resume_reuses_exact_shared_rankings_without_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, frozen, data, code = self._checkpoint_fixture(Path(tmp))
            self._run_resume(out, frozen, data, code)
            with np.load(out/'predictions/N16.npz') as z:
                np.testing.assert_array_equal(z['idx'], data['idx'])
                np.testing.assert_array_equal(z['rank_IU_PCR'], [1, 0, 3, 2, 5, 4, 7, 6, 9, 8])
                np.testing.assert_array_equal(z['rank_LSML_CONT'], np.arange(10))
                for name in driver.ARMS:
                    np.testing.assert_array_equal(z[f'count__IU_PCR__{name}'], z[f'count__LSML_CONT__{name}'])

    def test_complete_resume_rejects_receipt_or_payload_mutations(self):
        for changed in ['receipt', 'predictions', 'models']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                out, frozen, data, code = self._checkpoint_fixture(Path(tmp))
                folder = out/'checkpoints/N16_fold0'
                if changed == 'receipt':
                    receipt = json.loads((folder/'COMPLETE.json').read_text())
                    receipt['identity']['fold'] = 9
                    (folder/'COMPLETE.json').write_text(json.dumps(receipt))
                else:
                    path = folder/('predictions.npz' if changed == 'predictions' else 'models.json')
                    with path.open('ab') as f:
                        f.write(b'changed')
                with self.assertRaisesRegex(RuntimeError, 'Checkpoint identity mismatch'):
                    self._run_resume(out, frozen, data, code)

    def _replace_checkpoint(self, folder, update, models=None):
        with np.load(folder/'predictions.npz') as archive:
            contents = {key: archive[key].copy() for key in archive.files}
        contents.update(update)
        np.savez(folder/'predictions.npz', **contents)
        if models is not None:
            (folder/'models.json').write_text(json.dumps(models))
        receipt = json.loads((folder/'COMPLETE.json').read_text())
        receipt['predictions_sha256'] = driver.sha(folder/'predictions.npz')
        receipt['models_sha256'] = driver.sha(folder/'models.json')
        (folder/'COMPLETE.json').write_text(json.dumps(receipt))

    def test_resume_rejects_semantically_wrong_heldout_identity_even_with_matching_hash(self):
        for update in [{'idx': np.array(['different-answer'])}, {'offsets': np.array([0, 1, 2])}]:
            with self.subTest(key=list(update)), tempfile.TemporaryDirectory() as tmp:
                out, frozen, data, code = self._checkpoint_fixture(Path(tmp))
                self._replace_checkpoint(out/'checkpoints/N16_fold0', update)
                with self.assertRaises((RuntimeError, AssertionError, ValueError)):
                    self._run_resume(out, frozen, data, code)

    def test_resume_keeps_failed_arm_counts_missing_in_both_rankings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, frozen, data, code = self._checkpoint_fixture(Path(tmp))
            arm = 'A_MOM_MEAN'
            update = {f'count__{arm}': np.array([-1], dtype=np.int64),
                      f'count__IU_PCR__{arm}': np.array([-1], dtype=np.int64),
                      f'count__LSML_CONT__{arm}': np.array([-1], dtype=np.int64)}
            self._replace_checkpoint(out/'checkpoints/N16_fold0', update,
                                     models={'status': {arm: {'status': 'FAILED', 'reason': 'synthetic degeneracy'}}})
            self._run_resume(out, frozen, data, code)
            with np.load(out/'predictions/N16.npz') as archive:
                for key in update:
                    np.testing.assert_array_equal(archive[key], [-1, 0, 0, 0, 0])

    def test_holdout_prediction_failure_keeps_all_counts_missing_and_continues_folds(self):
        from spectral_utils.advisor_lr import AffineFusion

        for mode in ['raises', 'nonfinite']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                out, frozen, data, code = self._checkpoint_fixture(Path(tmp))
                for ready in (out/'checkpoints').glob('*/COMPLETE.json'):
                    ready.unlink()  # Exercise fresh synthetic fits rather than resume.
                fitted = []

                class InvalidHoldout(AffineFusion):
                    def predict(self, X):
                        if mode == 'raises':
                            raise RuntimeError('injected holdout prediction failure')
                        return np.full(len(X), np.nan)

                def fake_fit(X, method):
                    fitted.append((method, len(X)))
                    return InvalidHoldout(np.ones(X.shape[1]), 0., np.zeros(X.shape[1]),
                                          np.ones(X.shape[1]), {'synthetic_fit_valid': True})

                forbidden = AssertionError('Fuser failure must not reach count-model fitting')
                with patch.object(driver, 'OUT', out), patch.object(driver, 'FROZEN', frozen), \
                        patch.object(driver, 'validate_identity', return_value=code), \
                        patch.object(driver, 'load_unlabeled', return_value=data), \
                        patch.object(driver, 'archived_estimators', return_value=(None, None)), \
                        patch('spectral_utils.advisor_lr.fit_labelfree', side_effect=fake_fit), \
                        patch('spectral_utils.error_count_inputs.fit_vote_representation', side_effect=forbidden), \
                        patch('spectral_utils.error_count.fit_mom', side_effect=forbidden), \
                        patch('spectral_utils.error_count.fit_stable_triple', side_effect=forbidden), \
                        contextlib.redirect_stdout(io.StringIO()):
                    driver.predict()
                self.assertEqual(len(fitted), 20)  # Two fusers x all five folds x both banks.
                self.assertTrue((out/'checkpoints/N56_fold4/COMPLETE.json').exists())
                for bank in ['N16', 'N56']:
                    with np.load(out/f'predictions/{bank}.npz') as archive:
                        for method in ['IU_PCR', 'LSML_CONT']:
                            np.testing.assert_array_equal(archive['rank_'+method], -1)
                            self.assertTrue(np.isnan(archive[method+'_score']).all())
                        for key in archive.files:
                            if key.startswith('count__'):
                                np.testing.assert_array_equal(archive[key], -1)
                    for fold in range(5):
                        models = json.loads((out/f'checkpoints/{bank}_fold{fold}/models.json').read_text())
                        for method in ['IU_PCR', 'LSML_CONT']:
                            self.assertEqual(models[method]['status'], 'FAILED')
                        for arm, status in models['status'].items():
                            self.assertEqual(status['status'], 'FAILED')
                            self.assertEqual(status['reason'], 'FAILED_CANONICAL_FUSER_DEPENDENCY')

    def test_unlabeled_loader_never_accesses_label_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp)
            (frozen/'inputs').mkdir()
            (frozen/'controls').mkdir()
            idx = np.array(['synthetic'+str(i) for i in range(6969)])
            group = np.array(['group'+str(i % 707) for i in range(6969)])
            fold = np.array([(i % 707) % 5 for i in range(6969)])
            lengths = np.full(6969, 13)
            lengths[:3606] += 1
            off = np.r_[0, lengths.cumsum()]
            self.assertEqual(off[-1], 94203)
            np.savez(frozen/'inputs/population.npz', idx=idx, offsets=off, group=group,
                     fold=fold, feature_rows=np.arange(94203))
            rows = [{'row_id': str(uid), 'canonical_group_id': str(g)} for uid, g in zip(idx, group)]
            (frozen/'inputs/CANONICAL_GROUPS.json').write_text(json.dumps({'rows': rows}))
            (frozen/'inputs/FOLDS_V2.json').write_text(json.dumps({'prmbench': {'outer': {'group'+str(i): i % 5 for i in range(707)}}}))
            np.savez(frozen/'controls/NUISANCE_INPUTS.npz', idx=idx, answer_entropy=np.zeros(6969))
            np.savez(frozen/'inputs/features_thinkfix_mask1.npz', digit_active=np.ones((94203, 3), dtype=bool))
            real_load, accessed = np.load, []

            class LabelGuard:
                def __init__(self, archive):
                    self.archive = archive

                def __getitem__(self, key):
                    if key in ['y', 'label', 'labels', 'classification', 'true_count']:
                        raise AssertionError('Unlabeled loader accessed '+key)
                    accessed.append(key)
                    return self.archive[key]

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.archive.close()

                def __del__(self):
                    self.archive.close()

            def guarded_load(path, **kwargs):
                return LabelGuard(real_load(path, **kwargs))

            with patch.object(driver, 'FROZEN', frozen), patch.object(driver.np, 'load', side_effect=guarded_load):
                loaded = driver.load_unlabeled()
            self.assertEqual(set(loaded), {'idx', 'offsets', 'group', 'fold', 'feature_rows', 'entropy', 'digit_active'})
            self.assertNotIn('y', accessed)
            self.assertNotIn('classification', accessed)


if __name__ == '__main__':
    unittest.main()
