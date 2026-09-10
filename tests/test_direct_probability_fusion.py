import numpy as np

from spectral_utils.direct_probability_fusion import (
    answer_rank_features,
    direct_rank_risk,
    fit_rank_fusion,
    logprob_matrix,
    step_top_mean,
)


def _planted(seed=7, n=160, k=15):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    probabilities = np.empty((n, k), dtype=float)
    probabilities[:, 0] = 0.65 - 0.10 * latent + rng.normal(0, 0.01, n)
    for rank in range(1, k):
        probabilities[:, rank] = (
            0.12 / rank + 0.008 * latent + rng.normal(0, 0.001, n)
        )
    probabilities = np.clip(probabilities, 1e-6, 0.99)
    probabilities.sort(axis=1)
    probabilities = probabilities[:, ::-1]
    return np.log(probabilities), latent


def test_direct_rank_matrix_keeps_raw_probability_mass():
    logprobs, _ = _planted(n=12)
    saved = {"logprobs": np.column_stack([logprobs, logprobs[:, -1:] - 1.0])}
    matrix = logprob_matrix(saved, k=15)
    risk = direct_rank_risk(matrix)
    assert matrix.shape == risk.shape == (12, 15)
    assert np.allclose(risk[:, 0], 1.0 - np.exp(matrix[:, 0]))
    assert np.allclose(risk[:, 1:], np.exp(matrix[:, 1:]))
    assert not np.allclose(np.exp(matrix).sum(axis=1), 1.0)


def test_all_registered_rank_fusions_are_finite_and_risk_oriented():
    logprobs, latent = _planted()
    risk = direct_rank_risk(logprobs)
    for method in ("equal", "iu", "joint_lw"):
        result = fit_rank_fusion(risk, method=method, anchor=latent)
        assert result.score.shape == latent.shape
        assert np.isfinite(result.score).all()
        assert np.corrcoef(result.score, latent)[0, 1] >= 0
        assert result.weights.shape == (risk.shape[1],)
    joint = fit_rank_fusion(risk, method="joint_lw", anchor=latent)
    assert joint.alpha is not None and 0.0 <= joint.alpha <= 1.0


def test_top10_readout_is_over_tokens_and_independent_of_rank_k():
    scores = np.arange(20, dtype=float)
    steps = step_top_mean(scores, np.array([0, 5]), np.array([5, 20]), count=10)
    assert np.allclose(steps, [2.0, 14.5])
    matrix = np.column_stack([scores, scores[::-1], np.ones(20)])
    features = answer_rank_features(matrix, count=10)
    assert np.allclose(features, [14.5, 14.5, 1.0])
