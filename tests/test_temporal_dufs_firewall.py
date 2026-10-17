import unittest
from unittest.mock import patch
import numpy as np
from scripts.run_temporal_dufs31 import fit_selection


class SelectorFirewall(unittest.TestCase):
    def test_held_features_do_not_change_fit(self):
        rng=np.random.default_rng(42);samples=rng.normal(size=(32,31));owners=np.repeat(np.arange(4),8)
        metadata={i:dict(cell='pb_x_q4',group_id=str(i),fold=i%2) for i in range(4)}
        # Capture the actual data entering the optimizer; the test doesn't depend
        # on DUFS learning a particular numeric solution.
        seen=[]
        def capture(matrix,**kwargs):
            seen.append(matrix.copy());return np.arange(31,dtype=float),{}
        with patch('scripts.run_temporal_dufs31.adapted_dufs_soft_gates',side_effect=capture):
            fit_selection(samples,owners,metadata,'pb_x_q4',(1,))
            changed=samples.copy();changed[np.isin(owners,[1,3])]=1e9
            fit_selection(changed,owners,metadata,'pb_x_q4',(1,))
        np.testing.assert_array_equal(seen[0],seen[1])

    def test_labels_and_overlapping_sources_rejected(self):
        samples=np.ones((12,31));owners=np.repeat(np.arange(3),4)
        metadata={i:dict(cell='pb_x_q4',group_id=str(i),fold=i) for i in range(3)}
        metadata[0]['label']=1
        with self.assertRaises(ValueError):fit_selection(samples,owners,metadata,'pb_x_q4',(1,))
        del metadata[0]['label'];metadata[0]['group_id']=metadata[1]['group_id']
        with self.assertRaises(ValueError):fit_selection(samples,owners,metadata,'pb_x_q4',(1,))


if __name__=='__main__':unittest.main()
