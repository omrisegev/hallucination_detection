#!/usr/bin/env python3
"""Mechanical tests for the target-free local-descent PGRD screen."""

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

from scripts.run_deem_b3_local_descent_pgrd_v1 import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPECTED_VARIANTS,
    load_config,
)
from spectral_utils.deem_b3_local_descent_pgrd import (  # noqa: E402
    LocalDescentCalibration,
    LocalDescentPGRDConfig,
    build_common_residual_graph,
    fit_leave_dataset_family_out_direction,
    score_local_descent_pgrd,
)
from spectral_utils.deem_b3_residual_moe import (  # noqa: E402
    FAMILY_ORDER,
    GraphRoughnessMoment,
    ResidualCell,
)


def _synthetic_cell(seed: int = 20260825) -> tuple[ResidualCell, tuple[str, ...]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    n_rows = 96
    family_count = len(FAMILY_ORDER)
    baseline_logit = rng.normal(size=n_rows)
    baseline_mean = float(np.mean(baseline_logit))
    baseline_scale = float(np.std(baseline_logit))
    baseline_z = (baseline_logit - baseline_mean) / baseline_scale
    latent = rng.normal(size=(n_rows, 3))
    loo = np.column_stack(
        [
            latent[:, 0] + 0.20 * rng.normal(size=n_rows),
            latent[:, 1] + 0.20 * rng.normal(size=n_rows),
            latent[:, 2] + 0.20 * rng.normal(size=n_rows),
            latent[:, 0] - latent[:, 1] + 0.20 * rng.normal(size=n_rows),
            latent[:, 1] - latent[:, 2] + 0.20 * rng.normal(size=n_rows),
            latent[:, 2] - latent[:, 0] + 0.20 * rng.normal(size=n_rows),
        ]
    )
    loo = (loo - np.mean(loo, axis=0)) / np.std(loo, axis=0)
    present = np.ones(family_count, dtype=bool)
    instability = 0.01 + 0.02 * rng.random(size=loo.shape)
    cell = ResidualCell(
        cell_id="synthetic_local_descent",
        baseline_score=expit(baseline_logit),
        baseline_logit=baseline_logit,
        baseline_mean=baseline_mean,
        baseline_scale=baseline_scale,
        baseline_z=baseline_z,
        family_mean=np.zeros_like(loo),
        contribution_mean=np.zeros(family_count),
        contribution_scale=np.ones(family_count),
        baseline_loadings=np.zeros(family_count),
        residual_mean=np.zeros(family_count),
        residual_scale=np.ones(family_count),
        residuals=loo.copy(),
        seed_instability=instability.copy(),
        loo_residuals=loo,
        loo_seed_instability=instability,
        loo_predictability=np.zeros(family_count),
        present_mask=present,
        diagnostics={"n_present_families": family_count},
    )
    row_ids = tuple(f"synthetic-row-{index:04d}" for index in range(n_rows))
    return cell, row_ids


def _calibration() -> LocalDescentCalibration:
    family_count = len(FAMILY_ORDER)
    direction = np.linspace(0.7, 1.4, family_count)
    stability = np.linspace(0.25, 1.0, family_count)
    return LocalDescentCalibration(
        target_dataset_family="held",
        donor_dataset_families=("donor_a", "donor_b"),
        direction=direction,
        stability=stability,
        donor_group_directions=np.vstack([direction, 1.5 * direction]),
        donor_group_presence=np.ones((2, family_count), dtype=bool),
    )


def _moment(cell_id: str, c0: np.ndarray, present: np.ndarray) -> GraphRoughnessMoment:
    family_count = len(FAMILY_ORDER)
    a0 = np.eye(family_count)
    a0[~present, :] = 0.0
    a0[:, ~present] = 0.0
    values = np.asarray(c0, dtype=np.float64).copy()
    values[~present] = 0.0
    return GraphRoughnessMoment(
        cell_id=cell_id,
        residual_source="loo",
        a0=a0,
        c0=values,
        present_mask=present.copy(),
        trace_a0=float(np.trace(a0)),
    )


def _must_raise(function, exception=ValueError) -> None:
    try:
        function()
        raise AssertionError("expected contract violation")
    except exception:
        pass


def _test_holdout_pooling_and_stability() -> None:
    family_count = len(FAMILY_ORDER)
    full = np.ones(family_count, dtype=bool)
    missing_last = full.copy()
    missing_last[-1] = False
    unit = np.arange(1.0, family_count + 1.0)
    moments = (
        _moment("a1", -unit, full),
        _moment("a2", -3.0 * unit, full),
        _moment("b1", -4.0 * unit, missing_last),
        _moment("held1", 1000.0 * unit, full),
    )
    groups = ("donor_a", "donor_a", "donor_b", "held")
    fitted = fit_leave_dataset_family_out_direction(
        moments, groups, target_dataset_family="held"
    )
    expected_a = 2.0 * unit
    expected_b = 4.0 * unit
    expected_b[-1] = 0.0
    assert np.array_equal(fitted.direction, 0.5 * (expected_a + expected_b))
    assert fitted.donor_dataset_families == ("donor_a", "donor_b")
    assert "held" not in fitted.donor_dataset_families
    assert np.array_equal(fitted.stability, np.ones(family_count))
    assert fitted.diagnostics["target_dataset_family_excluded"] is True


def _test_local_router_step_and_controls() -> None:
    cell, row_ids = _synthetic_cell()
    calibration = _calibration()
    graph = build_common_residual_graph(cell, row_ids, k=7)
    alias = score_local_descent_pgrd(
        cell,
        calibration,
        graph,
        config=LocalDescentPGRDConfig(gate_mode="alias"),
    )
    primary = score_local_descent_pgrd(
        cell,
        calibration,
        graph,
        config=LocalDescentPGRDConfig(gate_mode="local"),
    )
    static = score_local_descent_pgrd(
        cell,
        calibration,
        graph,
        config=LocalDescentPGRDConfig(gate_mode="static"),
    )
    permuted = score_local_descent_pgrd(
        cell,
        calibration,
        graph,
        config=LocalDescentPGRDConfig(gate_mode="row_permuted"),
    )

    assert np.array_equal(alias.score, cell.baseline_score)
    assert np.array_equal(alias.logit, cell.baseline_logit)
    assert np.count_nonzero(alias.correction_z) == 0
    expected_activation = primary.family_stability[None, :] * np.maximum(
        -primary.local_gradient[:, None] * primary.expert_terms, 0.0
    )
    assert np.array_equal(primary.activation, expected_activation)
    local_mass = np.sum(primary.local_gate_probabilities, axis=1)
    assert np.all((local_mass == 0.0) | np.isclose(local_mass, 1.0))
    assert np.all(
        primary.activation[primary.local_gate_probabilities > 0.0] > 0.0
    )
    assert 0.0 < primary.alpha <= primary.tau
    assert primary.roughness_after <= primary.roughness_before + 1e-10
    assert abs(float(np.std(primary.direction)) - 1.0) <= 1e-12

    expected_static = np.repeat(
        np.mean(primary.local_gate_probabilities, axis=0)[None, :],
        len(row_ids),
        axis=0,
    )
    assert np.array_equal(static.gate_probabilities, expected_static)
    assert np.allclose(
        np.mean(static.gate_probabilities, axis=0),
        np.mean(primary.local_gate_probabilities, axis=0),
        atol=1e-15,
        rtol=1e-15,
    )
    assert abs(
        float(np.mean(np.sum(static.gate_probabilities, axis=1)))
        - float(np.mean(local_mass))
    ) <= 1e-15
    assert not np.array_equal(
        permuted.row_permutation, np.arange(len(row_ids), dtype=np.int64)
    )
    assert np.array_equal(
        permuted.gate_probabilities,
        primary.local_gate_probabilities[permuted.row_permutation],
    )
    for result in (primary, static, permuted):
        assert 0.0 <= result.alpha <= result.tau
        assert result.roughness_after <= result.roughness_before + 1e-10
        assert np.isfinite(result.score).all()


def _test_calibration_and_graph_firewalls() -> None:
    cell, row_ids = _synthetic_cell(seed=911)
    calibration = _calibration()
    graph = build_common_residual_graph(cell, row_ids, k=7)
    base_call = lambda current: score_local_descent_pgrd(  # noqa: E731
        cell,
        current,
        graph,
        config=LocalDescentPGRDConfig(gate_mode="local"),
    )
    _must_raise(
        lambda: base_call(
            replace(calibration, direction=np.full(len(FAMILY_ORDER), np.nan))
        )
    )
    _must_raise(
        lambda: base_call(replace(calibration, direction=np.ones(3)))
    )
    _must_raise(
        lambda: base_call(
            replace(calibration, stability=np.full(len(FAMILY_ORDER), 1.01))
        )
    )
    swapped_ids = list(graph.row_ids)
    swapped_ids[0], swapped_ids[1] = swapped_ids[1], swapped_ids[0]
    _must_raise(
        lambda: score_local_descent_pgrd(
            cell,
            calibration,
            replace(graph, row_ids=tuple(swapped_ids)),
            config=LocalDescentPGRDConfig(gate_mode="local"),
        )
    )
    changed_laplacian = graph.laplacian.copy().tolil()
    changed_laplacian[0, 0] += 1e-6
    changed_laplacian = changed_laplacian.tocsr()
    _must_raise(
        lambda: score_local_descent_pgrd(
            cell,
            calibration,
            replace(graph, laplacian=changed_laplacian),
            config=LocalDescentPGRDConfig(gate_mode="local"),
        )
    )


def _test_config_and_label_firewall() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert tuple(row["id"] for row in config["variants"]) == EXPECTED_VARIANTS
    assert config["scientific_boundary"]["fit_is_label_free"] is True
    assert (
        config["scientific_boundary"][
            "target_cell_rows_enter_only_target_free_transductive_geometry"
        ]
        is True
    )
    for relative in (
        "spectral_utils/deem_b3_local_descent_pgrd.py",
        "scripts/run_deem_b3_local_descent_pgrd_v1.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        imports = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any("label" in ast.unparse(node).lower() for node in imports)


def main() -> None:
    _test_holdout_pooling_and_stability()
    _test_local_router_step_and_controls()
    _test_calibration_and_graph_firewalls()
    _test_config_and_label_firewall()
    print("DEEM-B3 local-descent PGRD mechanical tests: PASS")


if __name__ == "__main__":
    main()
