import numpy as np
from spectral_utils.joint_sparse_membership import exact_aliases,alias_coordinates
from spectral_utils.joint_sparse_refinement import refine_joint_membership


def test_refinement_is_identical_after_alias_expansion():
    rng=np.random.default_rng(321);n=3000;groups=np.repeat(np.arange(3),4)
    x=.55*rng.normal(size=(n,1))+.35*rng.normal(size=(n,3))[:,groups]+.25*rng.normal(size=(n,12))
    x=(x-x.mean(0))/x.std(0)
    duplicated=np.column_stack((x,x[:,:3]));coordinates=alias_coordinates(duplicated,exact_aliases(duplicated))
    a=refine_joint_membership(x,groups,seed=123)
    b=refine_joint_membership(coordinates,groups,seed=123)
    np.testing.assert_array_equal(a['active'],b['active'])
    np.testing.assert_array_equal(x@a['unoriented_weights'],coordinates@b['unoriented_weights'])
    assert a['selection']['automatic']['minimum_retention']>=.95
