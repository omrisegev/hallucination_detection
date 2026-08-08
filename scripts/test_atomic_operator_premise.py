#!/usr/bin/env python3
"""Known-answer and leakage-boundary tests for the Phase-0 audit."""

import inspect
import os
import sys
import types

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.atomic_operator_audit import (  # noqa: E402
    audit_cell,
    crossfit_alignment,
    duplicate_keep_mask,
    graph_operator,
    iu_state,
    weights_for_signature,
)
import scripts.atomic_operator_premise_fit as fit_script  # noqa: E402
from scripts.atomic_operator_premise_report import (  # noqa: E402
    FAMILY_NAMES,
    exact_sign_flip_p,
    family,
    safe_spearman,
    summarize,
    tie_aware_quartiles,
)
from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402


def main():
    rng = np.random.default_rng(17)
    n = 120
    latent = rng.standard_normal(n)
    raw = np.vstack([
        latent + 0.15 * rng.standard_normal(n),
        latent + 0.16 * rng.standard_normal(n),
        -0.6 * latent + 0.5 * rng.standard_normal(n),
        rng.standard_normal(n),
        0.3 * latent + rng.standard_normal(n),
    ])
    F = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    names = tuple(f"f{index}" for index in range(F.shape[0]))

    state = iu_state(F)
    quotient, signature, ambient, diagnostics = graph_operator(
        state, 0, graph_k=7
    )
    assert quotient.graph.shape == (quotient.n_unique, quotient.n_unique)
    assert quotient.laplacian.shape == quotient.graph.shape
    assert quotient.projection.shape == (n, quotient.n_unique)
    assert quotient.valid
    assert signature.shape == (2, 2)
    assert ambient.shape == (F.shape[0], F.shape[0])
    assert np.allclose(signature, signature.T)
    assert np.isclose(np.trace(signature), 1.0)
    assert np.linalg.eigvalsh(signature).min() >= -1e-9
    assert diagnostics["n_components"] >= 1
    assert np.array_equal(weights_for_signature(state, signature, 0.0), state.baseline_weights)
    assert not np.allclose(
        weights_for_signature(state, signature, 1.0), state.baseline_weights
    )

    keep, similarity, fallback = duplicate_keep_mask(F, 0, 0.90)
    assert not keep[0] and not keep[1]
    assert similarity[1] >= 0.90
    assert not fallback
    alignment = crossfit_alignment(
        F,
        0,
        quotient,
        duplicate_threshold=0.90,
        permutation_count=5,
        namespace="known-answer",
    )
    assert np.isfinite(alignment["alignment"])
    assert alignment["n_excluded"] >= 2

    # Atomic operators and scores must be equivariant to a sample-row
    # permutation even when the graph-defining feature has large tied groups.
    tied = F.copy()
    tied[0] = np.repeat(np.arange(12, dtype=float), 10)
    tied = (tied - tied.mean(axis=1, keepdims=True)) / np.maximum(
        tied.std(axis=1, keepdims=True), 1e-12
    )
    tied_state = iu_state(tied)
    tied_graph, tied_signature, _, tied_diag = graph_operator(
        tied_state, 0, graph_k=7
    )
    permutation = rng.permutation(n)
    permuted_state = iu_state(tied[:, permutation])
    permuted_graph, permuted_signature, _, permuted_diag = graph_operator(
        permuted_state, 0, graph_k=7
    )
    assert tied_diag["max_tie_size"] == 10
    assert permuted_diag["max_tie_size"] == 10
    assert np.allclose(tied_signature, permuted_signature, atol=1e-10)
    tied_score = weights_for_signature(tied_state, tied_signature, 1.0) @ tied
    permuted_score = (
        weights_for_signature(permuted_state, permuted_signature, 1.0)
        @ tied[:, permutation]
    )
    assert np.allclose(tied_score, permuted_score[np.argsort(permutation)], atol=1e-9)

    constant = tied.copy()
    constant[0] = 0.0
    constant_graph, _, _, constant_diag = graph_operator(iu_state(constant), 0, graph_k=7)
    assert not constant_graph.valid
    assert constant_diag["n_unique_values"] == 1
    keep_constant, similarity_constant, _ = duplicate_keep_mask(constant, 0, 0.90)
    assert np.isfinite(similarity_constant).all()
    assert not keep_constant[0]

    scores, audit = audit_cell(
        F,
        names,
        cell="synthetic",
        graph_ks=(5, 7),
        primary_graph_k=7,
        lambdas=(0.5, 1.0),
        primary_lambda=1.0,
        duplicate_threshold=0.90,
        duplicate_sensitivities=(0.85, 0.95),
        subsamples=4,
        sample_fraction=0.75,
        sample_cap=90,
        permutation_count=4,
        convergence_checkpoints=(2, 4),
    )
    assert scores["iu_pcr"].shape == (n,)
    assert scores["atomic__k_7__lambda_1"].shape == (len(names), n)
    assert scores["uniform_atomic__k_5__lambda_0.5"].shape == (n,)
    assert scores["ridge__lambda_1"].shape == (n,)
    assert len(audit["feature_records"]) == len(names)
    assert len(audit["convergence"]) == 2 * 2 * 2 * len(names)
    for record in audit["feature_records"]:
        assert np.isfinite(record["primary_proxy"])
        assert 0.0 <= record["operator_reproducibility"] <= 1.0
        assert 0.0 <= record["rank_change_reproducibility"] <= 1.0
        assert 0.0 <= record["relative_actuation"] <= 1.0

    # The fit side must not contain the bundle key used for correctness arrays,
    # and the numerical API must not accept a target argument.
    fit_source = inspect.getsource(fit_script)
    assert "__labels" not in fit_source
    assert "labels" not in inspect.signature(audit_cell).parameters

    assert np.isnan(safe_spearman(np.ones(5), np.arange(5)))
    bottom, top = tie_aware_quartiles(np.ones(8))
    assert not len(bottom) and not len(top)

    # Exhaustive eight-family sign flips have denominator 2**8. The second
    # vector is a registered-boundary known answer: exactly 12 assignments are
    # at least as large as the observed mean, so p=12/256 <= 0.05. Applying a
    # Monte Carlo +1 correction here would incorrectly move it above 0.05.
    assert np.isclose(exact_sign_flip_p(np.ones(8)), 1.0 / 256.0)
    signflip_boundary = np.asarray([-3, 7, 2, 0, 4, 0, 10, 5], dtype=float)
    assert np.isclose(exact_sign_flip_p(signflip_boundary), 12.0 / 256.0)
    assert exact_sign_flip_p(signflip_boundary) <= 0.05

    # A perfectly ordered proxy over candidates that all lose must fail the
    # absolute-headroom continuation gates.
    atomic_rows, control_rows, convergence_rows, primary_by_cell = [], [], [], {}
    for cell in INSCOPE:
        local = []
        for index in range(8):
            proxy = float(index)
            delta = float(index - 8)  # every candidate loses
            row = {
                "cell": cell,
                "domain": GROUP[cell],
                "family": family(cell),
                "feature": f"f{index}",
                "feature_index": index,
                "graph_k": 15,
                "lambda": 1.0,
                "valid_operator": True,
                "auroc": 0.75 + delta / 100.0,
                "auprc": 0.60,
                "delta_pp": delta,
                "path_proxy": proxy,
                "primary_proxy": proxy,
                "full_alignment": proxy,
                "bootstrap_alignment": proxy,
                "operator_reproducibility": proxy,
                "rank_change_reproducibility": proxy,
                "stability_actuation_proxy": proxy,
                "full_actuation": proxy,
                "anisotropy": proxy,
                "edge_mass_per_node": float((index * 3) % 7),
                "projected_effective_rank": float((index * 5) % 7),
                "duplicate_density": float((index * 2) % 7),
                "distance_from_ridge": float((index * 4) % 7),
                "operator_duplicate_density": 0.0,
                "full_duplicate_fallback": 0.0,
                "alignment_duplicate_0p9": proxy,
                "alignment_duplicate_0p99": proxy,
            }
            local.append(row)
            for graph_k in fit_script.GRAPH_KS:
                for lambda_ in fit_script.LAMBDAS:
                    atomic_rows.append({
                        **row,
                        "graph_k": graph_k,
                        "lambda": lambda_,
                    })
            for checkpoint in (1, 2):
                convergence_rows.append({
                    "cell": cell,
                    "family": family(cell),
                    "replicates": checkpoint,
                    "graph_k": 15,
                    "lambda": 1.0,
                    "feature": f"f{index}",
                    "primary_proxy": proxy,
                })
        primary_by_cell[cell] = {
            "rows": local,
            "spearman": 1.0,
            "top_bottom_pp": 4.0,
            "proxy_ridge_spearman": safe_spearman(
                [row["primary_proxy"] for row in local],
                [row["distance_from_ridge"] for row in local],
            ),
            "top_feature": "f7",
            "top_proxy_tie_count": 1,
            "top_feature_delta_pp": -1.0,
            "oracle_feature": "f7",
            "oracle_delta_pp": -1.0,
        }
        for method, delta in (
            ("iu_pcr", 0.0),
            ("top_proxy_atomic", -1.0),
            ("oracle_atomic", -1.0),
        ):
            control_rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": family(cell),
                "method": method,
                "auroc": 0.75 + delta / 100.0,
                "auprc": 0.60,
                "delta_pp": delta,
            })
    outputs = summarize(
        atomic_rows, control_rows, convergence_rows, primary_by_cell,
        randomization_count=40,
    )
    summary, gates = outputs[0], outputs[1]
    assert not summary["all_gates_passed"]
    absolute = next(
        row for row in gates
        if row["gate"] == "top-proxy atomic family-bootstrap AUROC lower > 0"
    )
    assert not absolute["passed"]
    print("ATOMIC OPERATOR PREMISE TEST PASS")


if __name__ == "__main__":
    main()
