"""Scheduling, write exclusion and exact failure contract tests; no research fits."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('spectral_utils'); pkg.__path__ = [str(CODE / 'spectral_utils')]
sys.modules['spectral_utils'] = pkg
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'scripts/experiments'))
from spectral_utils import monday_lr_helper as helper


class HelperTests(unittest.TestCase):
    def test_duplicate_helper_ownership_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'owner.json'
            with helper.ownership(path):
                with self.assertRaisesRegex(RuntimeError, 'ownership file exists'):
                    with helper.ownership(path):
                        self.fail('Duplicate writer acquired ownership')
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())

    def test_priority_and_population(self):
        banks = ['B13', 'B54']; cells = ['pb_a', 'pb_b', 'prmbench_qwen3_8b']
        jobs = helper.runner_order(cells, banks)
        ordered = helper.helper_order(jobs, banks)
        self.assertEqual(len(jobs), 120)
        self.assertEqual(set(jobs), set(ordered))
        self.assertTrue(all(j[0].startswith('prm') and j[2] == 'original' for j in ordered[:8]))
        self.assertTrue(all(j[2] == 'original' for j in ordered[:24]))
        self.assertEqual(ordered[24], list(reversed(jobs))[0])

    def test_cursor_utf16_and_completed_distance(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); status = p / 'status.json'; log = p / 'run.log'; lr = p / 'lr'; lr.mkdir()
            jobs = helper.runner_order(['pb_a', 'prm_a'], ['B13'])
            status.write_text(json.dumps({'stage': 'position_decision'}))
            self.assertEqual(helper.reach_seconds(jobs, 19, lr, status, log), 19 * 12)
            status.write_text(json.dumps({'stage': 'lr'}))
            log.write_text('LR checkpoint ' + helper.stem_of(jobs[2]) + ' COMPLETE\n', encoding='utf16')
            (lr / (helper.stem_of(jobs[3]) + '.complete.json')).write_text('{}')
            self.assertEqual(helper.runner_cursor(jobs, lr, status, log), 4)
            self.assertEqual(helper.reach_seconds(jobs, 4, lr, status, log), 0)
            self.assertEqual(helper.reach_seconds(jobs, 22, lr, status, log), 16 * 12 + 2 * 240)

    def test_publish_refuses_collision_and_rechecks(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); staging = p / 'pending'; lr = p / 'lr'; staging.mkdir(); lr.mkdir()
            for s in (*helper.SUFFIXES, '.complete.json'):
                (staging / ('job' + s)).write_text(s)
            with patch.object(helper.os, 'link', wraps=helper.os.link) as link:
                checks = iter([True, False])
                self.assertFalse(helper.exclusive_publish(staging, lr, 'job', lambda: next(checks)))
                self.assertEqual(link.call_count, 0)
                self.assertTrue(helper.exclusive_publish(staging, lr, 'job', lambda: True))
                self.assertTrue(str(link.call_args_list[-1].args[1]).endswith('.complete.json'))
                self.assertFalse(helper.exclusive_publish(staging, lr, 'job', lambda: True))
                self.assertEqual(link.call_count, 4)

    def test_failure_rows_match_runner_schema(self):
        import numpy as np
        state = {'bank': {'answers': np.arange(7)}}
        with patch.object(helper, 'lr_rows', side_effect=ValueError('expected failure')):
            rows, folds, diag, status, elapsed = helper.compute(None, state, ('pb_a', 'B13', 'original', 'survivors', True))
        self.assertEqual(status, 'ARM_FAILURE'); self.assertEqual(len(rows), 8)
        self.assertEqual(folds, []); self.assertEqual(diag, {'failure': 'expected failure'})
        self.assertTrue(all(r['coverage'] == 0 and r['failures'] == 7 and r['n_total'] == 7 for r in rows))
        self.assertEqual({r['score_scope'] for r in rows}, {'OOF', 'IN_SAMPLE_DIAGNOSTIC'})
        self.assertNotIn('runtime_s', helper.deterministic_payload((rows, folds, diag, status, elapsed))['rows'][0])


if __name__ == '__main__':
    unittest.main()
