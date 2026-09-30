"""Safety tests for long-running completion and publication of reviewed evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


runner=module('completion_test','complete_research_consolidation_v1.py')
builder=module('builder_test','build_research_consolidation_v1.py')


class CompletionTests(unittest.TestCase):
    def test_shrinkage_family_and_metric_rows_render_in_existing_report(self):
        ledger = json.loads((ROOT/'results/research_consolidation_v1/LEDGER.json').read_text(encoding='utf-8'))
        ledger['families'] = builder.families(ledger['obligations'])
        page = builder.render(ledger)
        self.assertIn('full__joint__alw / shared entropy-q0.3', page)
        self.assertIn('fusion_shrinkage_iu_codex_review_v1/REPORT.md', page)

    def test_shrinkage_metrics_use_common_gate_without_borrowed_fold_auc(self):
        rows, sources = builder.collect()
        row = next(r for r in rows if r['study']=='full_shrinkage_review'
                   and r['method']=='full__joint__alw / shared entropy-q0.3')
        self.assertAlmostEqual(row['pb_all'], 0.31771774510247647)
        self.assertAlmostEqual(row['prm_within_auc'], 0.7100062612847627)
        self.assertIsNone(row['prm_fold_mean_auc'])
        self.assertEqual((row['pb_population'], row['pb_valid']), (6800, 6796))
        self.assertEqual(row['evidence_status'], 'FROZEN_SCORE_METRICS_REVIEWED')
        self.assertIn(row['metric_source'], sources)

    def test_failed_run_never_restarts(self):
        with patch.object(runner,'verified_complete',return_value=False), \
             patch.object(runner,'find_driver',return_value=None), \
             patch.object(runner,'load',return_value={'phase':'FAILED'}), \
             patch.object(runner.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'Refusing automatic restart'):
                runner.finish_job(runner.JOBS[0])
            launch.assert_not_called()

    def test_complete_run_not_repeated(self):
        with patch.object(runner,'verified_complete',return_value=True), \
             patch.object(runner,'find_driver') as discovery:
            runner.finish_job(runner.JOBS[0]); discovery.assert_not_called()

    def test_normal_invocation_cap_resumes_identical_driver(self):
        current={'phase':'SCORING'}
        launches=[]
        def launch(args,**kwargs):
            launches.append(args)
            return Mock(pid=90000+len(launches),wait=Mock(return_value=0))
        def wait(proc,job):
            current['phase']='CHECKPOINTED_INVOCATION_CAP' if len(launches)==1 else job['complete']
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runner,'OUT',Path(folder)), \
             patch.object(runner,'verified_complete',side_effect=lambda job:current['phase']==job['complete']), \
             patch.object(runner,'find_driver',return_value=None), \
             patch.object(runner,'load',side_effect=lambda path:dict(current)), \
             patch.object(runner,'state'), \
             patch.object(runner.shutil,'disk_usage',return_value=Mock(free=8*1024**3)), \
             patch.object(runner.subprocess,'Popen',side_effect=launch), \
             patch.object(runner.psutil,'Process'), \
             patch.object(runner,'wait_for',side_effect=wait):
            runner.finish_job(runner.JOBS[1])
        self.assertEqual(len(launches),2)
        self.assertEqual(launches[0],launches[1])
        self.assertEqual(launches[0][-2:],['--phase','run'])

    def test_final_report_requires_all_reviews(self):
        with patch.object(builder,'obligations',return_value={
                'historical_joint':{'status':'REVIEWED_PASS'},
                'full_sampling':{'status':'UNFINISHED'},
                'fixed_gate':{'status':'REVIEWED_PASS'}}), \
             patch.object(builder,'collect') as collect, \
             patch.object(builder,'save') as save:
            with self.assertRaisesRegex(RuntimeError,'Cannot finalize'):
                builder.main(False)
            collect.assert_not_called();save.assert_not_called()

    def test_transfer_groups_are_disjoint_in_actual_frozen_release(self):
        records=json.loads((ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
        easy={r['group_id'] for r in records if r['cell'].startswith(('pb_gsm8k_','pb_math_'))}
        hard={r['group_id'] for r in records if r['cell'].startswith(('pb_olympiadbench_','pb_omnimath_'))}
        self.assertEqual((len(easy),len(hard)),(1330,1512))
        self.assertFalse(easy & hard)

    def test_future_sampling_and_historical_contrasts_keep_undefined_visible(self):
        sampling={'paired':[{'left':'sample_risk_top__iu','right':'sample_full__iu','scope':'all',
                   'delta':{'within':.01,'pb_all':-.02},
                   'intervals':{'within':{'ci95':[.001,.02]},'pb_all':{'ci95':[-.03,.001]}},
                   'prm_common_answers':6000}]}
        historical={'contrasts':[{'candidate':'internal_joint_liu010','control':'internal_joint_modelinv_lam0',
                     'prm_common_answers':6000,'endpoints':{'prm_within_auc':{'difference':None,'low':None,'high':None}}}]}
        def load(path):
            return sampling if 'localization_full_sampling_v3' in str(path) else historical
        with patch.object(builder,'load',side_effect=load):
            rows=builder.paired_evidence(['results/localization_full_sampling_v3/evaluation/INTERVALS.json',
                                         'results/historical_joint_refit_v3/MECHANISM_CONTRASTS.json'])
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['ci95'],[.001,.02])
        self.assertEqual(rows[1]['difference'],-.02)
        self.assertIn('אפס',rows[1]['interpretation'])
        self.assertIsNone(rows[2]['ci95'])
        self.assertEqual(rows[2]['interpretation'],'לא מוגדר')

    def test_completion_docs_preserve_existing_crlf_and_are_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=b'# Existing guide\r\n\r\nUser content\r\n'
            for name in ('PROGRESS.md','Research_Directions.md','HISTORY.md'):
                (root/name).write_bytes(original)
            with patch.object(runner,'ROOT',root), \
                 patch.object(runner,'load',return_value={'status':'COMPLETE_REVIEWED_CONSOLIDATION'}):
                runner.document_completion()
                first={name:(root/name).read_bytes() for name in ('PROGRESS.md','Research_Directions.md','HISTORY.md')}
                runner.document_completion()
            for name,data in first.items():
                self.assertNotIn(b'\r\r\n',data)
                self.assertIn(b'User content\r\n',data)
                self.assertEqual(data,(root/name).read_bytes())


if __name__=='__main__':
    unittest.main(verbosity=2)
