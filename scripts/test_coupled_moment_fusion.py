#!/usr/bin/env python3
"""Known-answer tests for the coupled-moment latent-factor premise."""

from __future__ import annotations

import os
import sys

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.coupled_moment_fusion import (  # noqa: E402
    all_distinct_third_moments,
    component_agreement,
    covariance_basis,
    fit_moment_factors,
    grouped_half_splits,
    permuted_cross_moment_values,
    select_rank_label_free,
    target_component,
    zscore_columns,
)
from scripts.coupled_moment_24cell_experiment import (  # noqa: E402
    LabelFreeBundleView,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402


def synthetic_latent(seed=5, n=4000, m=12, rank=3):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(m, rank))
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)
    sources = rng.exponential(size=(n, rank)) - 1.0
    values = sources @ loadings.T + 0.15 * rng.normal(size=(n, m))
    return values, loadings, sources


def test_rank_and_target_recovery():
    rng = np.random.default_rng(5)
    n, m, rank = 4000, 12, 3
    truth = np.zeros((m, rank), dtype=float)
    for component in range(rank):
        truth[4 * component:4 * component + 4, component] = rng.uniform(
            0.5, 1.5, 4
        )
    truth /= np.linalg.norm(truth, axis=0, keepdims=True)
    sources = rng.exponential(size=(n, rank)) - 1.0
    values = sources @ truth.T + 0.05 * rng.normal(size=(n, m))
    reliability = truth[:, 0] / values.std(axis=0)
    selection = select_rank_label_free(
        values,
        reliability,
        max_rank=3,
        split_seeds=(101, 211, 307, 401),
        cp_seeds=(11, 23, 37),
        max_nfev=250,
    )
    assert selection.selected_rank == 3, selection
    fitted = fit_moment_factors(
        values,
        selection.selected_rank,
        reliability,
        seeds=(11, 23, 37),
        max_nfev=250,
    )
    loading_cosine = abs(float(
        np.dot(fitted.loadings[:, fitted.target_index], reliability)
        / (
            np.linalg.norm(fitted.loadings[:, fitted.target_index])
            * np.linalg.norm(reliability)
        )
    ))
    score_correlation = abs(float(np.corrcoef(fitted.target_score, sources[:, 0])[0, 1]))
    assert loading_cosine > 0.90, loading_cosine
    assert score_correlation > 0.90, score_correlation


def test_independent_feature_skew_does_not_create_shared_factors():
    """The all-distinct mask must reject the reviewer's failure mode."""
    rng = np.random.default_rng(77)
    n, m = 4000, 10
    loading = rng.uniform(0.6, 1.4, m)
    loading /= np.linalg.norm(loading)
    shared = rng.exponential(size=n) - 1.0
    independent_skew = rng.exponential(size=(n, m)) - 1.0
    values = (
        shared[:, None] * loading[None, :]
        + 0.65 * independent_skew
        + 0.05 * rng.normal(size=(n, m))
    )
    selection = select_rank_label_free(
        values,
        loading / values.std(axis=0),
        max_rank=5,
        cp_seeds=(11, 23, 37),
        max_nfev=200,
    )
    assert selection.selected_rank == 1, selection


def test_fit_bundle_adapter_blocks_labels_structurally():
    class TrackingBundle:
        def __init__(self):
            self.arrays = {
                f"{cell}__{suffix}": np.asarray([1.0])
                for cell in INSCOPE
                for suffix in ("V", "pool", "hand_signs", "labels")
            }
            self.files = tuple(self.arrays)
            self.accessed = []

        def __getitem__(self, key):
            self.accessed.append(key)
            return self.arrays[key]

    raw = TrackingBundle()
    view = LabelFreeBundleView.from_npz(raw)
    assert not any("label" in key.lower() for key in raw.accessed)
    assert not any("label" in key.lower() for key in view.files)
    try:
        view[f"{INSCOPE[0]}__labels"]
    except KeyError:
        pass
    else:
        raise AssertionError("label access was not blocked")


def test_covariance_and_moments_share_frozen_coordinates():
    rng = np.random.default_rng(123)
    values = rng.normal(size=(800, 9))
    values[:400] *= np.linspace(0.25, 3.0, 9)[None, :]
    frozen = zscore_columns(values)
    train = frozen[:400]
    basis, eigenvalues = covariance_basis(
        train, dimension=6, standardize=False
    )
    centred = train - train.mean(axis=0, keepdims=True)
    manual_values, manual_basis = np.linalg.eigh(centred.T @ centred / len(train))
    manual_basis = manual_basis[:, -6:]
    manual_values = manual_values[-6:][::-1]
    manual_basis = manual_basis[:, ::-1]
    assert np.allclose(eigenvalues, manual_values, atol=1e-10)
    assert np.allclose(
        basis @ basis.T, manual_basis @ manual_basis.T, atol=1e-10
    )


def test_rank_one_is_exact_no_deflation():
    values, truth, _ = synthetic_latent(n=1500)
    standard = zscore_columns(values)
    fitted = fit_moment_factors(
        standard, 1, truth[:, 0], standardize=False, max_nfev=100
    )
    assert np.array_equal(fitted.deflated_values, standard)


def test_target_anchor_and_component_matching():
    rng = np.random.default_rng(8)
    loadings = np.linalg.qr(rng.normal(size=(8, 3)))[0]
    chosen, oriented, alignment, margin, similarities = target_component(
        loadings[:, [2, 0, 1]], -loadings[:, 0]
    )
    assert chosen == 1
    assert alignment > 1.0 - 1e-10
    assert margin > 0.9
    assert oriented[:, chosen] @ (-loadings[:, 0]) > 0
    assert similarities.shape == (3,)
    assert component_agreement(loadings, loadings[:, [2, 0, 1]]) > 1 - 1e-10


def test_group_splits_do_not_separate_repetitions():
    groups = np.repeat(np.arange(20), 10)
    for left, right in grouped_half_splits(len(groups), groups=groups):
        assert not (set(groups[left]) & set(groups[right]))


def test_permutation_destroys_cross_moments_but_preserves_marginals():
    values, _, _ = synthetic_latent(n=3000)
    permuted = permuted_cross_moment_values(values, 91)
    for column in range(values.shape[1]):
        assert np.array_equal(np.sort(values[:, column]), np.sort(permuted[:, column]))
    before = np.linalg.norm(all_distinct_third_moments(values))
    after = np.linalg.norm(all_distinct_third_moments(permuted))
    assert after < 0.35 * before, (before, after)


def test_unstable_symmetric_sources_fall_back():
    rng = np.random.default_rng(99)
    values = rng.normal(size=(500, 10))
    selection = select_rank_label_free(
        values,
        np.ones(10),
        split_seeds=(3, 5, 7, 9),
        max_nfev=60,
    )
    assert selection.selected_rank == 1


def main():
    tests = [
        test_rank_and_target_recovery,
        test_independent_feature_skew_does_not_create_shared_factors,
        test_fit_bundle_adapter_blocks_labels_structurally,
        test_covariance_and_moments_share_frozen_coordinates,
        test_rank_one_is_exact_no_deflation,
        test_target_anchor_and_component_matching,
        test_group_splits_do_not_separate_repetitions,
        test_permutation_destroys_cross_moments_but_preserves_marginals,
        test_unstable_symmetric_sources_fall_back,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} coupled-moment tests passed.")


if __name__ == "__main__":
    main()
