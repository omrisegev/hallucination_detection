import json
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.evaluate_temporal_context_models import assemble
from scripts.run_temporal_neural_job import key,jobs


class NestedCalibration(unittest.TestCase):
    def test_held_fold_never_enters_its_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata=[dict(fold=h,cell='prm',step_start=h,step_stop=h+1) for h in range(5)]
            for seed in (0,1):
                for excluded in [(h,) for h in range(5)]+list(combinations(range(5),2)):
                    path=root/key(dict(method='fm',bank='original4',seed=seed,excluded=excluded))/'scoring';path.mkdir(parents=True)
                    (path/'RUN_STATE.json').write_text(json.dumps(dict(status='SCORED',answers=len(excluded),expected=len(excluded))))
                    (path/'MANIFEST.json').write_text(json.dumps(dict(smoke=False)))
                    values=np.full(5,np.nan)
                    for h in excluded:values[h]=h+10*seed
                    np.savez(path/'STEP_SCORES.npz',signal=values)
            values,threshold=assemble(root,'fm','original4',(0,1),metadata,5)
            np.testing.assert_array_equal(values['signal'],np.arange(5)+5)
            for h in range(5):self.assertAlmostEqual(threshold['signal'][str(h)],np.quantile(np.delete(np.arange(5)+5,h),.8))
            path=root/key(dict(method='fm',bank='original4',seed=0,excluded=(0,1)))/'scoring'
            (path/'RUN_STATE.json').write_text(json.dumps(dict(status='SCORING',answers=1,expected=2)))
            with self.assertRaises(ValueError):assemble(root,'fm','original4',(0,1),metadata,5)

    def test_job_matrix_unique_and_bounded(self):
        matrix=jobs();self.assertEqual(len(matrix),270);self.assertEqual(len({key(j) for j in matrix}),270)


if __name__=='__main__':unittest.main()
