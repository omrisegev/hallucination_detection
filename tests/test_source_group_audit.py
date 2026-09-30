import importlib.util
import io
from pathlib import Path
import pickle
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_groups',ROOT/'scripts/audit_localization_source_groups.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class MetadataIdentity(unittest.TestCase):
    def test_metadata_matches_standard_pickle_across_frames_and_protocols(self):
        question='One shared question with Unicode: '+chr(945)+' and '+chr(8804)
        source={'a':{'idx':'confidence_prm_train_p1_7','source_idx':'confidence_prm_train_p1_7',
                     'question':question,'raw':np.arange(200000,dtype=np.float64),'blob':b'x'*200000},
                'b':{'idx':'circular_prm_train_p1_7','source_idx':'circular_prm_train_p1_7',
                     'question':question,'raw':np.ones((50,50),dtype=np.int32)}}
        for protocol in (4,5):
            encoded=pickle.dumps(source,protocol=protocol)
            standard=pickle.loads(encoded);metadata=a.MetadataUnpickler(io.BytesIO(encoded)).load()
            for key in source:
                for field in ('idx','source_idx','question'):
                    self.assertEqual(metadata[key][field],standard[key][field])
                self.assertIsInstance(metadata[key]['raw'],a.Discarded)
            self.assertEqual(metadata['a']['blob'],b'')

    def test_source_suffix_preserves_train_test_and_partition(self):
        self.assertEqual(a.source_seed('correct_redundency_prm_train_p1_7'),'prm_train_p1_7')
        self.assertNotEqual(a.source_seed('circular_prm_train_p1_7'),a.source_seed('circular_prm_test_p1_7'))
        self.assertNotEqual(a.source_seed('circular_prm_train_p1_7'),a.source_seed('circular_prm_train_p2_7'))
        with self.assertRaises(ValueError):a.source_seed('ambiguous_7')


if __name__=='__main__':unittest.main()
