import numpy as np

from spectral_utils.lsml_gate_locator_research import (
    FusionRecipe,
    answer_standardize,
    cell_midranks,
    fit_fusion_weights,
)


def test_answer_standardize_is_answer_local_and_constant_safe():
    values = np.asarray([[1., 4.], [3., 4.], [10., 2.], [12., 4.], [14., 6.]])
    got = answer_standardize(values, np.asarray([0, 2, 5]))
    np.testing.assert_allclose(got[:2].mean(0), 0., atol=1e-12)
    np.testing.assert_allclose(got[2:].mean(0), 0., atol=1e-12)
    np.testing.assert_array_equal(got[:2, 1], 0.)


def test_cell_midranks_do_not_cross_cells():
    values = np.asarray([[1.], [3.], [100.], [200.]])
    cells = np.asarray(["a", "a", "b", "b"])
    got = cell_midranks(values, cells, np.ones(4, bool))[:, 0]
    np.testing.assert_allclose(got, [0., 1., 0., 1.])


def test_equal_fusion_is_label_free_and_normalized():
    x = np.arange(15., dtype=float).reshape(5, 3)
    recipe = FusionRecipe("equal", ("a", "b", "c"), mode="equal")
    weight, meta = fit_fusion_weights(x, recipe, seed=1)
    np.testing.assert_allclose(weight, np.ones(3) / 3)
    assert meta["mode"] == "equal"
