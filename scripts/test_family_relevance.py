#!/usr/bin/env python3
"""Known-answer tests for graph-coupled family relevance."""

import inspect
import os
import sys
import types

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.family_relevance import (  # noqa: E402
    family_prior_laplacian,
    fit_family_relevance_paths,
    generate_switching_family_world,
    local_family_evidence,
    smooth_family_evidence,
)
import scripts.family_relevance_fit as fit_script  # noqa: E402
import scripts.family_relevance_report as report_script  # noqa: E402


def main():
    F, names, labels, regime = generate_switching_family_world(seed=3, n=300)
    evidence = local_family_evidence(F, names)
    families = evidence["families"]
    L, adjacency = family_prior_laplacian(families)
    assert np.allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() >= -1e-10
    assert np.allclose(adjacency, adjacency.T)
    gates0 = smooth_family_evidence(
        evidence["raw_evidence"], evidence["observed_family"], L, beta=0.0
    )
    gates1 = smooth_family_evidence(
        evidence["raw_evidence"], evidence["observed_family"], L, beta=1.0
    )
    assert gates0.shape == gates1.shape == (F.shape[1], len(families))
    assert np.isfinite(gates1).all() and np.min(gates1) > 0
    assert np.allclose(np.mean(gates1, axis=1), 1.0)
    assert not np.allclose(gates0, gates1)

    scores, diagnostics = fit_family_relevance_paths(
        F, names, cell="known-answer", betas=(0.0, 1.0), blends=(0.5,)
    )
    assert scores["iu_pcr"].shape == (F.shape[1],)
    assert scores["family_experts"].shape == (len(families), F.shape[1])
    for prefix in (
        "manual_graph", "permuted_graph", "global_gate", "sample_permuted_gate"
    ):
        assert scores[f"{prefix}__beta_1__blend_0.5"].shape == (F.shape[1],)
    assert diagnostics["families"] == list(families)

    # Sample permutation equivariance, including the learned IU-PCR anchor.
    permutation = np.random.default_rng(11).permutation(F.shape[1])
    permuted_scores, _ = fit_family_relevance_paths(
        F[:, permutation], names, cell="known-answer-permuted",
        betas=(1.0,), blends=(0.5,),
    )
    assert np.allclose(
        scores["manual_graph__beta_1__blend_0.5"],
        permuted_scores["manual_graph__beta_1__blend_0.5"][np.argsort(permutation)],
        atol=1e-9,
    )

    assert "labels" not in inspect.signature(fit_family_relevance_paths).parameters
    assert "labels" not in inspect.signature(local_family_evidence).parameters
    assert labels.shape == regime.shape == (F.shape[1],)

    # The registered synthetic positive control must detect switching relevance,
    # while the correlated-nuisance world remains an explicit failure case.
    positive = roc_auc_score(
        labels, scores["manual_graph__beta_1__blend_0.5"]
    ) - roc_auc_score(labels, scores["iu_pcr"])
    bad_F, bad_names, bad_labels, _ = generate_switching_family_world(
        seed=3, n=300, correlated_nuisance=True
    )
    bad_scores, _ = fit_family_relevance_paths(
        bad_F, bad_names, cell="known-answer-bad", betas=(1.0,), blends=(0.5,)
    )
    failure = roc_auc_score(
        bad_labels, bad_scores["manual_graph__beta_1__blend_0.5"]
    ) - roc_auc_score(bad_labels, bad_scores["iu_pcr"])
    assert positive > 0
    assert failure < 0

    fit_source = inspect.getsource(fit_script)
    report_source = inspect.getsource(report_script)
    assert "__labels" not in fit_source
    assert report_source.index("definition = verify_freeze") < report_source.index(
        'data[f"{cell}__labels"]'
    )
    print("FAMILY RELEVANCE TEST PASS")


if __name__ == "__main__":
    main()
