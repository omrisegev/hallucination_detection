"""Perturbations must not change original inputs or depend on answer ordering."""
import numpy as np
from spectral_utils.joint_redundancy_stress import answer_noise, augment


def test_noise_is_identity_stable_and_standardized():
    original = answer_noise([0, 4, 7, 8], ['a', 'b', 'c'])
    reordered = answer_noise([0, 3, 7, 8], ['b', 'a', 'c'])
    np.testing.assert_array_equal(original[:4], reordered[3:7])
    np.testing.assert_array_equal(original[4:7], reordered[:3])
    for block in (original[:4], original[4:7]):
        np.testing.assert_allclose(block.mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(block.std(0), 1, atol=1e-14)
    np.testing.assert_array_equal(original[7], np.zeros(15))


def test_augmentation_preserves_original_bank():
    base = np.random.default_rng(73).normal(size=(8, 51))
    frozen = base.copy()
    for kind in ('duplicates', 'noise'):
        result = augment(base, [0, 4, 7, 8], ['a', 'b', 'c'], kind)
        assert result.shape == (8, 66)
        np.testing.assert_array_equal(result[:, :51], base)
        if kind == 'duplicates':
            np.testing.assert_array_equal(result[:, 51:], base[:, :15])
    np.testing.assert_array_equal(base, frozen)
