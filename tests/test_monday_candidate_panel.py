"""Scientific edge cases raised by the independent candidate-panel review."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.monday_candidate_panel import CODE,canonical_peak,endpoints,setup
import spectral_utils
spectral_utils.__path__.append(str(CODE/'spectral_utils'))

def test_peak_matches_canonical_near_tie():
    eps=np.finfo(float).eps
    assert canonical_peak(np.array([1.,1.+eps]))==0
    assert canonical_peak(np.array([1.,1.+16*eps]))==1

def test_bootstrap_f1_missing_class_is_undefined():
    statistics=np.zeros((4,12))
    statistics[0,9]=1  # no erroneous-answer denominator
    statistics[1,7]=1  # no clean-answer denominator
    statistics[2,7]=statistics[2,9]=1  # both classes, zero successes
    statistics[3,6:10]=1  # both classes and all successes
    f1=endpoints(statistics)['pb_official_f1']
    assert np.isnan(f1[:2]).all()
    assert np.array_equal(f1[2:],[0.,1.])
