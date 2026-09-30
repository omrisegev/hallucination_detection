import unittest
import numpy as np
from spectral_utils.prm_label_contract import prm_error_flags


class PRMLabelContractTest(unittest.TestCase):
    def test_first_and_last_step(self):
        np.testing.assert_array_equal(prm_error_flags([1,4],4),[1,0,0,1])

    def test_no_index_base_guessing(self):
        np.testing.assert_array_equal(prm_error_flags([8],9),[0,0,0,0,0,0,0,1,0])

    def test_out_of_range_inert(self):
        np.testing.assert_array_equal(prm_error_flags([0,-1,2,5],4),[0,1,0,0])

    def test_empty_and_duplicate_annotations(self):
        np.testing.assert_array_equal(prm_error_flags([],3),[0,0,0])
        np.testing.assert_array_equal(prm_error_flags([2,2],3),[0,1,0])

    def test_invalid_annotation_rejected(self):
        for annotation in ([True],[1.5],['1']):
            with self.assertRaises(ValueError):prm_error_flags(annotation,3)


if __name__=='__main__':unittest.main()
