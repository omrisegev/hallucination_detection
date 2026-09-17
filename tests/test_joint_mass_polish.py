import numpy as np
from spectral_utils.joint_mass_polish import polish_kernel_mass


def test_active_set_recovers_a_boundary_optimum():
    k=np.array([[1,.7,.7],[.7,1,.2],[.7,.2,1.]])
    x,audit=polish_kernel_mass(k,[.2,.8,.8])
    np.testing.assert_allclose(x,[0,5/6,5/6],atol=1e-10)
    assert audit['projected_kkt']<=1e-8 and np.all(np.diff(audit['objective_trace'])<=1e-10)


def test_singular_repeated_kernel_uses_aggregate_mass():
    x,audit=polish_kernel_mass(np.ones((2,2)),[.2,.3])
    np.testing.assert_allclose(x.sum(),1.,atol=1e-12)
    assert np.all(x>=0) and audit['projected_kkt']<=1e-8
