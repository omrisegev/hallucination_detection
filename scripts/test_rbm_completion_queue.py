"""Scheduling tests: no duplicate writers and no stage before its dependencies."""
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_completion_queue import next_stage,ready_suites,SUITES,ExistingProcess

class QueueTests(unittest.TestCase):
    def test_stage_recovery(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            def put(name,**data):(p/name).write_text(json.dumps(data))
            self.assertEqual(next_stage(p),'smoke')
            put('SMOKE.json',status='PASS');self.assertEqual(next_stage(p),'smoke_review')
            put('SMOKE_REVIEW.json',status='PASS');self.assertEqual(next_stage(p),'full')
            put('RUN_STATE.json',status='SCORED_AWAITING_REVIEW');self.assertEqual(next_stage(p),'full_review')
            put('RUN_STATE.json',status='COMPLETE');put('RESULT_REVIEW.json',status='PASS')
            self.assertIsNone(next_stage(p))
    def test_dependencies_and_active_writer(self):
        stages={s:'full' for s in SUITES};stages['variance']=None
        self.assertEqual(ready_suites(stages,{'capacity'}, {'variance'}),['temporal'])
        stages['capacity']=None
        self.assertEqual(ready_suites(stages,{'temporal'},{'variance','capacity'}),['stability','depth'])
    def test_failed_review_cannot_unlock_dependents(self):
        stages={s:'full_review' for s in SUITES}
        self.assertNotIn('depth',ready_suites(stages,{},set()))
        self.assertNotIn('stability',ready_suites(stages,{},set()))
    @unittest.skipUnless(os.name=='nt','Windows handle test')
    def test_process_identity_and_exit(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.2)'])
        handle=ExistingProcess(child.pid)
        with self.assertRaises(ProcessLookupError):ExistingProcess(child.pid,handle.created+1)
        self.assertIn(handle.poll(),(None,0))
        self.assertEqual(child.wait(timeout=10),0)
        self.assertEqual(handle.poll(),0)
        handle.close()

if __name__=='__main__':unittest.main()
