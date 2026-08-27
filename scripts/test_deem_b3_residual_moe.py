#!/usr/bin/env python3
"""Mechanical tests for true-LOO residuals and B3-PGRD extensions."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    FAMILY_ORDER,
    FrozenB3Ensemble,
    GraphRoughnessMoment,
    ResidualCell,
    build_residual_cell,
    graph_roughness_moment,
    load_frozen_b3_ensemble,
    pooled_graph_roughness_direction,
    score_graph_roughness_direction,
)
from spectral_utils.graph_topology import self_safe_knn_graph  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem import canonical_sha256  # noqa: E402
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_target_free_bundle,
)


def _synthetic_ensemble(seed: int = 20260825) -> tuple[FrozenB3Ensemble, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    n = 80
    folds = np.arange(n, dtype=int) % 5
    latent = rng.normal(size=(n, 3))
    family = np.column_stack(
        [
            latent[:, 0] + 0.10 * rng.normal(size=n),
            latent[:, 1] + 0.10 * rng.normal(size=n),
            latent[:, 2] + 0.10 * rng.normal(size=n),
            latent[:, 0] - latent[:, 1] + 0.10 * rng.normal(size=n),
            latent[:, 1] + latent[:, 2] + 0.10 * rng.normal(size=n),
            latent[:, 0] - latent[:, 2] + 0.10 * rng.normal(size=n),
        ]
    )
    seed_family = np.stack(
        [family + 0.01 * rng.normal(size=family.shape) for _ in range(5)]
    )
    seed_logits = seed_family.sum(axis=2)
    seed_scores = expit(seed_logits)
    return (
        FrozenB3Ensemble(
            cell_id="synthetic",
            seeds=(0, 1, 2, 3, 4),
            score=seed_scores.mean(axis=0),
            seed_scores=seed_scores,
            seed_logits=seed_logits,
            seed_biases=np.zeros(5),
            seed_family_contributions=seed_family,
            present_mask=np.ones(len(FAMILY_ORDER), dtype=bool),
        ),
        folds,
    )


def _synthetic_cell(seed: int = 19, *, missing_last: bool = False) -> ResidualCell:
    rng = np.random.Generator(np.random.PCG64(seed))
    n = 96
    present = np.ones(len(FAMILY_ORDER), dtype=bool)
    if missing_last:
        present[-1] = False
    baseline_logit = rng.normal(size=n)
    baseline_mean = float(np.mean(baseline_logit))
    baseline_scale = float(np.std(baseline_logit))
    baseline_z = (baseline_logit - baseline_mean) / baseline_scale
    residuals = rng.normal(size=(n, len(FAMILY_ORDER)))
    loo = rng.normal(size=(n, len(FAMILY_ORDER)))
    residuals[:, ~present] = 0.0
    loo[:, ~present] = 0.0
    instability = 0.01 + 0.02 * rng.random(size=residuals.shape)
    instability[:, ~present] = 0.0
    return ResidualCell(
        cell_id="synthetic_missing" if missing_last else "synthetic_full",
        baseline_score=expit(baseline_logit),
        baseline_logit=baseline_logit,
        baseline_mean=baseline_mean,
        baseline_scale=baseline_scale,
        baseline_z=baseline_z,
        family_mean=rng.normal(size=residuals.shape),
        contribution_mean=np.zeros(len(FAMILY_ORDER)),
        contribution_scale=np.ones(len(FAMILY_ORDER)),
        baseline_loadings=np.zeros(len(FAMILY_ORDER)),
        residual_mean=np.zeros(len(FAMILY_ORDER)),
        residual_scale=np.ones(len(FAMILY_ORDER)),
        residuals=residuals,
        seed_instability=instability,
        loo_residuals=loo,
        loo_seed_instability=instability,
        loo_predictability=np.zeros(len(FAMILY_ORDER)),
        present_mask=present,
    )


def _test_true_loo() -> None:
    ensemble, folds = _synthetic_ensemble()
    original = build_residual_cell(ensemble, folds=folds)
    held = folds == 0
    shifted_family = np.asarray(ensemble.seed_family_contributions).copy()
    shifted_family[:, held, 0] += 4.0
    shifted_logits = shifted_family.sum(axis=2)
    shifted_scores = expit(shifted_logits)
    shifted = replace(
        ensemble,
        score=shifted_scores.mean(axis=0),
        seed_scores=shifted_scores,
        seed_logits=shifted_logits,
        seed_family_contributions=shifted_family,
    )
    changed = build_residual_cell(shifted, folds=folds)
    donor_target = ensemble.seed_family_contributions.mean(axis=0)[~held, 0]
    expected_delta = 4.0 / float(np.std(donor_target))
    observed_delta = (
        changed.loo_residuals[held, 0] - original.loo_residuals[held, 0]
    )
    # The fold-0 predictor and its target scale cannot consume fold-0 targets.
    assert np.allclose(observed_delta, expected_delta, atol=1e-10, rtol=1e-10)
    assert np.isfinite(original.loo_predictability).all()


def _test_pgrd_derivative_and_trace() -> None:
    cell = _synthetic_cell()
    row_ids = tuple(f"row-{index:04d}" for index in range(len(cell.baseline_z)))
    moment = graph_roughness_moment(cell, row_ids, residual_source="loo", k=7)
    present = cell.present_mask
    count = int(np.sum(present))
    assert abs(float(np.trace(moment.a0)) - count) <= 1e-10

    residuals = cell.loo_residuals[:, present]
    graph = self_safe_knn_graph(
        residuals,
        k=7,
        tie_keys=np.arange(len(residuals), dtype=float),
    )
    laplacian = symmetric_normalized_laplacian(graph)
    scale = float(moment.diagnostics["trace_scale"])

    def energy(delta: np.ndarray) -> float:
        value = cell.baseline_z + residuals @ delta
        return float(scale * value @ (laplacian @ value) / len(value))

    epsilon = 1e-6
    derivative = []
    for column in range(count):
        step = np.zeros(count)
        step[column] = epsilon
        derivative.append((energy(step) - energy(-step)) / (2.0 * epsilon))
    assert np.allclose(
        derivative,
        2.0 * moment.c0[present],
        atol=2e-6,
        rtol=2e-6,
    )


def _test_pooling_and_application() -> None:
    full = np.ones(len(FAMILY_ORDER), dtype=bool)
    missing = full.copy()
    missing[-1] = False
    first = GraphRoughnessMoment(
        cell_id="a1",
        residual_source="loo",
        a0=np.eye(len(FAMILY_ORDER)),
        c0=np.arange(1.0, len(FAMILY_ORDER) + 1.0),
        present_mask=full,
        trace_a0=2.0,
    )
    second = GraphRoughnessMoment(
        cell_id="a2",
        residual_source="loo",
        a0=2.0 * np.eye(len(FAMILY_ORDER)),
        c0=2.0 * np.arange(1.0, len(FAMILY_ORDER) + 1.0),
        present_mask=full,
        trace_a0=3.0,
    )
    third_c = 3.0 * np.arange(1.0, len(FAMILY_ORDER) + 1.0)
    third_c[-1] = 0.0
    third_a = 3.0 * np.eye(len(FAMILY_ORDER))
    third_a[-1, -1] = 0.0
    third = GraphRoughnessMoment(
        cell_id="b1",
        residual_source="loo",
        a0=third_a,
        c0=third_c,
        present_mask=missing,
        trace_a0=4.0,
    )
    direction, diagnostics = pooled_graph_roughness_direction(
        [first, second, third], ["family_a", "family_a", "family_b"]
    )
    expected = -0.5 * (0.5 * (first.c0 + second.c0) + third.c0)
    assert np.array_equal(direction, expected)
    assert diagnostics["n_pool_groups"] == 2
    # The absent structural coordinate contributes a literal zero in family_b.
    assert direction[-1] == -0.25 * (first.c0[-1] + second.c0[-1])

    cell = _synthetic_cell(missing_last=True)
    active = score_graph_roughness_direction(
        cell,
        np.arange(1.0, len(FAMILY_ORDER) + 1.0),
        residual_source="loo",
        trust_factor=1.0,
        gate_strength=0.25,
    )
    count = int(np.sum(cell.present_mask))
    assert abs(float(np.std(active.correction_z)) - 1.0 / count) <= 1e-12
    assert np.max(np.abs(active.gates.sum(axis=1) - count)) <= 1e-12
    assert np.min(active.gates[:, cell.present_mask]) >= 0.75 - 1e-12
    alias = score_graph_roughness_direction(
        cell,
        np.arange(1.0, len(FAMILY_ORDER) + 1.0),
        residual_source="loo",
        trust_factor=0.0,
        gate_strength=1.0,
    )
    assert np.array_equal(alias.score, cell.baseline_score)
    assert np.array_equal(alias.logit, cell.baseline_logit)
    assert np.count_nonzero(alias.correction_z) == 0


def _test_optional_frozen_binding() -> None:
    bundle_dir = ROOT / "local_cache/deem_b3_moe_v1/bundles"
    baseline_dir = ROOT / "local_cache/deem_b3_moe_v1/b3_frozen"
    cell_id = "lapeigvals_gsm8k_llama3b"
    bundle_path = bundle_dir / f"{cell_id}.npz"
    if not bundle_path.is_file():
        return
    bundle = load_target_free_bundle(bundle_path)
    ensemble = load_frozen_b3_ensemble(
        baseline_dir,
        cell_id,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_ordered_row_id_sha256=canonical_sha256(list(bundle.row_ids)),
    )
    assert ensemble.score.shape == (len(bundle.row_ids),)
    try:
        load_frozen_b3_ensemble(
            baseline_dir,
            cell_id,
            expected_bundle_sha256="0" * 64,
            expected_ordered_row_id_sha256=canonical_sha256(list(bundle.row_ids)),
        )
        raise AssertionError("wrong bundle binding must fail")
    except ValueError:
        pass


def _test_label_firewall() -> None:
    for relative in (
        "spectral_utils/deem_b3_residual_moe.py",
        "scripts/run_deem_b3_residual_moe_v1.py",
        "scripts/run_deem_b3_residual_pgrd_v1.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        imports = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any("label" in ast.unparse(node).lower() for node in imports)


def main() -> None:
    _test_true_loo()
    _test_pgrd_derivative_and_trace()
    _test_pooling_and_application()
    _test_optional_frozen_binding()
    _test_label_firewall()
    print("DEEM-B3 residual/PGRD mechanical tests: PASS")


if __name__ == "__main__":
    main()
