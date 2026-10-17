from pathlib import Path
import numpy as np
from spectral_utils.joint_mass_groups import feature_mass as original_mass
from spectral_utils.joint_polished_groups import feature_mass


def test_successful_optimizer_is_bitwise_unchanged():
    c=np.array([[1,.3,.2],[.3,1,.4],[.2,.4,1.]])
    old,old_audit=original_mass(c);new,audit=feature_mass(c)
    np.testing.assert_array_equal(old,new)
    np.testing.assert_array_equal(old_audit['raw_mass'],audit['raw_mass'])
    assert audit['polishing'] is None


def test_captured_failures_reach_original_criterion_without_new_objective():
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'results/joint_mass_membership_v1/MASS_SOLVER_FAILURE_INPUTS.npz') as z:
        for outer in (2,3):
            c=z[f'outer{outer}_covariance'];mass,audit=feature_mass(c)
            assert audit['initial_projected_gradient']>1e-6 and audit['polishing'] is not None
            raw=np.asarray(audit['raw_mass']);s=np.sqrt(np.diag(c));kernel=(c/s[:,None]/s[None,:])**2
            gradient=kernel@raw-1
            assert np.min(gradient)>=-1e-8 and np.max(np.abs(raw*gradient))<1e-8
            assert audit['projected_gradient']<=1e-8
            np.testing.assert_allclose(mass,raw/raw.sum(),atol=1e-12)
            assert audit['objective']<=audit['polishing']['objective_trace'][0]+1e-12
