import numpy as np
import unittest

from spectral_utils.direct_probability_fusion_v2 import (
    augmented_feature_names,
    augmented_probability_risk,
    residual_tail_mass,
    selected_surprisal,
)


def _payload(n=6, saved_k=20):
    base = np.linspace(0.20, 0.08, n)[:, None]
    ratios = np.geomspace(1.0, 0.005, saved_k)[None, :]
    probabilities = base * ratios
    return {"logprobs": np.log(probabilities)}


def test_augmented_matrix_is_v1_ranks_plus_atp_and_raw_tail():
    payload = _payload()
    selected = np.linspace(0.2, 3.0, 6)
    matrix = augmented_probability_risk(payload, selected, k=15)
    probabilities = np.exp(payload["logprobs"][:, :15])
    assert matrix.shape == (6, 17)
    assert np.allclose(matrix[:, 0], 1.0 - probabilities[:, 0])
    assert np.allclose(matrix[:, 1:15], probabilities[:, 1:])
    assert np.allclose(matrix[:, 15], selected)
    assert np.allclose(matrix[:, 16], 1.0 - probabilities.sum(axis=1))
    assert augmented_feature_names(15)[-2:] == (
        "selected_token_surprisal",
        "residual_tail_mass",
    )


def test_tail_uses_raw_mass_without_topk_renormalization():
    probabilities = np.array([[0.5, 0.2, 0.1], [0.9, 0.05, 0.02]])
    observed = residual_tail_mass(np.log(probabilities))
    assert np.allclose(observed, [0.2, 0.03])


def test_invalid_selected_alignment_fails_closed():
    with unittest.TestCase().assertRaisesRegex(ValueError, "align"):
        selected_surprisal([0.1, 0.2], 3)


def test_probability_mass_above_one_fails_closed():
    with unittest.TestCase().assertRaisesRegex(ValueError, "exceeds one"):
        residual_tail_mass(np.log(np.array([[0.8, 0.3, 0.1]])))


if __name__ == "__main__":
    for test in (
        test_augmented_matrix_is_v1_ranks_plus_atp_and_raw_tail,
        test_tail_uses_raw_mass_without_topk_renormalization,
        test_invalid_selected_alignment_fails_closed,
        test_probability_mass_above_one_fails_closed,
    ):
        test()
    print("direct probability fusion v2 tests: PASS")
