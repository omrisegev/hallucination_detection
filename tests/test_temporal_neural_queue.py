import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_temporal_neural_queue import QueueLock,matrix_for_seeds,completed_job


class NeuralQueue(unittest.TestCase):
    def test_single_seed_complete_exclusion_matrix(self):
        selected=matrix_for_seeds([0]);self.assertEqual(len(selected),90)
        self.assertEqual([i for i,_ in selected[:3]],[3,4,5])
        self.assertTrue(all(len(j['excluded'])==1 for _,j in selected[:30]))
        self.assertTrue(all(len(j['excluded'])==2 for _,j in selected[30:]))
        self.assertEqual(len({i for i,_ in selected}),90)

    def test_queue_lock_exclusive_then_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            with QueueLock(path):
                with self.assertRaises(RuntimeError):
                    with QueueLock(path):pass
            with QueueLock(path):pass

    def test_completed_marker_requires_matching_data_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);scoring=dest/'scoring';scoring.mkdir()
            job=dict(method='fm',bank='innovation5',seed=0,excluded=(0,))
            self.assertFalse(completed_job(dest,job,'abc'))
            (scoring/'RUN_STATE.json').write_text(json.dumps(dict(status='SCORED',answers=2,expected=2)))
            (scoring/'MANIFEST.json').write_text(json.dumps(dict(smoke=False,model_manifest=dict(
                method='fm',bank='innovation5',seed=0,excluded_folds=[0],data_manifest_sha256='abc'))))
            with self.assertRaises(ValueError):completed_job(dest,job,'abc')
            (scoring/'STEP_SCORES.npz').write_bytes(b'fixture')
            self.assertTrue(completed_job(dest,job,'abc'))
            with self.assertRaises(ValueError):completed_job(dest,job,'different')


if __name__=='__main__':unittest.main()
