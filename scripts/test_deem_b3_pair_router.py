#!/usr/bin/env python3
"""Dataset-free mechanical tests for the pairwise antisymmetric B3 router."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_pair_router import (  # noqa: E402
    PairRouterConfig,
    _PairResidualRouterEnergy,
    fit_pair_residual_router,
    predict_pair_residual_router,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    _FamilyAdditiveEnergy,
)


NAMES = (
    "epr",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
    "trace_length",
)
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"


def _fixture(seed: int = 17):
    rng = np.random.Generator(np.random.PCG64(20260825))
    X = rng.normal(size=(48, len(NAMES)))
    base = _FamilyAdditiveEnergy(
        NAMES,
        ContinuousDeemConfig(family_width=4, init_sd=0.01),
        seed=seed,
    )
    return X, base, base.state_dict_numpy()


def _baseline_score(base: _FamilyAdditiveEnergy, X: np.ndarray, orientation: int) -> np.ndarray:
    with torch.no_grad():
        raw, _, _ = base.logit(torch.as_tensor(X, dtype=torch.float64))
    aligned = int(orientation) * raw.cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(aligned, -700.0, 700.0)))


def _calibrated_model(
    X: np.ndarray,
    baseline_state: dict[str, np.ndarray],
    *,
    rho: float,
    seed: int = 17,
) -> _PairResidualRouterEnergy:
    model = _PairResidualRouterEnergy(
        NAMES,
        PairRouterConfig(
            rho=rho,
            epochs=0,
            family_width=4,
            base_init_sd=0.01,
            posterior_sd_min=0.0,
        ),
        seed=seed,
    )
    model.load_baseline_state(baseline_state)
    model.calibrate_context(torch.as_tensor(X, dtype=torch.float64))
    return model


def _test_exact_identities_and_gate_geometry() -> None:
    X, base, baseline_state = _fixture()
    tensor = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        base_logit, base_atomic, _ = base.logit(tensor)

    # A zero-initialized router is the exact identity even when rho is active.
    zero = _calibrated_model(X, baseline_state, rho=0.73)
    with torch.no_grad():
        zero_logit, zero_values = zero.logit(tensor)
    assert torch.equal(zero_logit, base_logit)
    assert torch.equal(zero_values["routed_atomic"], base_atomic)
    assert torch.equal(zero_values["gates"], torch.ones_like(zero_values["gates"]))

    # rho=0 must be structurally nested at B3 for arbitrary finite router
    # parameters, not only at the initialization used by the optimizer.
    alias = _calibrated_model(X, baseline_state, rho=0.0)
    generator = torch.Generator(device="cpu").manual_seed(991)
    with torch.no_grad():
        alias.pair_weight.copy_(
            4.0
            * torch.randn(
                alias.pair_weight.shape,
                dtype=torch.float64,
                generator=generator,
            )
        )
        alias.pair_open_logit.copy_(
            3.0
            * torch.randn(
                alias.pair_open_logit.shape,
                dtype=torch.float64,
                generator=generator,
            )
        )
        alias_logit, alias_values = alias.logit(tensor)
    assert torch.equal(alias_logit, base_logit)
    assert torch.equal(alias_values["routed_atomic"], base_atomic)
    assert torch.equal(alias_values["gates"], torch.ones_like(alias_values["gates"]))

    active = _calibrated_model(X, baseline_state, rho=0.65)
    with torch.no_grad():
        active.pair_weight.copy_(
            2.0
            * torch.randn(
                active.pair_weight.shape,
                dtype=torch.float64,
                generator=generator,
            )
        )
        active.pair_open_logit.copy_(
            torch.linspace(-2.0, 2.0, len(active.pair_indices), dtype=torch.float64)
        )
        active_logit, values = active.logit(tensor)
    gates = values["gates"].cpu().numpy()
    count = len(active.family_order)
    assert np.max(np.abs(gates.sum(axis=1) - count)) <= 1e-12
    assert float(np.min(gates)) >= 1.0 - active.config.rho - 1e-12
    assert float(np.max(gates)) <= 1.0 + active.config.rho + 1e-12
    assert np.max(
        np.abs(values["family_probabilities"].cpu().numpy() - gates / count)
    ) <= 1e-12
    assert float(np.std(gates)) > 0.0
    reconstruction = (
        active.base.b.detach().item()
        + values["routed_atomic"].sum(dim=1).cpu().numpy()
        - active_logit.cpu().numpy()
    )
    assert np.max(np.abs(reconstruction)) <= 1e-12
    assert np.max(
        np.abs(
            values["routed_family"].cpu().numpy()
            - values["base_family"].cpu().numpy() * gates
        )
    ) <= 1e-12


def _test_self_free_jacobian_and_context_dependency() -> None:
    X, _, baseline_state = _fixture()
    model = _calibrated_model(X, baseline_state, rho=0.8)
    generator = torch.Generator(device="cpu").manual_seed(443)
    with torch.no_grad():
        model.pair_weight.copy_(
            torch.randn(
                model.pair_weight.shape,
                dtype=torch.float64,
                generator=generator,
            )
        )
        model.pair_open_logit.fill_(1.25)

    tensor = torch.as_tensor(X, dtype=torch.float64)
    with torch.no_grad():
        _, family = model.base.contributions(tensor)
        family_values = model._family_matrix(family)[:1].detach().clone()
    family_values.requires_grad_(True)

    def gate_function(values):
        return model.routing_from_family_values(values)["gates"]

    jacobian = torch.autograd.functional.jacobian(
        gate_function,
        family_values,
        create_graph=False,
        strict=False,
    )[0, :, 0, :]
    diagonal = torch.diagonal(jacobian)
    # Every gamma_g is assembled only from pairs incident to g, whose contexts
    # exclude g.  The zero derivative is therefore exact, not merely small.
    assert torch.equal(diagonal, torch.zeros_like(diagonal))
    off_diagonal = jacobian - torch.diag(diagonal)
    assert float(torch.max(torch.abs(off_diagonal))) > 1e-8

    # Stronger edge-local statement: each t_gh has exactly zero derivative
    # with respect to both endpoint family scores, but retains a nonzero
    # derivative with respect to at least one complement-family context.
    transfer_jacobian = torch.autograd.functional.jacobian(
        lambda values: model.routing_from_family_values(values)["pair_transfer"],
        family_values,
        create_graph=False,
        strict=False,
    )[0, :, 0, :]
    for pair_index, (left, right) in enumerate(model.pair_indices):
        assert transfer_jacobian[pair_index, left].item() == 0.0
        assert transfer_jacobian[pair_index, right].item() == 0.0
        complement = [
            index
            for index in range(model.family_count)
            if index not in {left, right}
        ]
        assert float(torch.max(torch.abs(transfer_jacobian[pair_index, complement]))) > 1e-10

    context = model.routing_from_family_values(family_values)["context_residual"]
    for pair_index, (left, right) in enumerate(model.pair_indices):
        assert torch.equal(
            context[:, pair_index, left],
            torch.zeros_like(context[:, pair_index, left]),
        )
        assert torch.equal(
            context[:, pair_index, right],
            torch.zeros_like(context[:, pair_index, right]),
        )
        perturbed = family_values.detach().clone()
        perturbed[:, left] += 17.0
        perturbed[:, right] -= 11.0
        changed = model.routing_from_family_values(perturbed)["context_residual"]
        assert torch.equal(changed[:, pair_index, :], context[:, pair_index, :])


def _test_fit_determinism_graph_contract_and_frozen_base() -> None:
    X, base, baseline_state = _fixture(seed=23)
    baseline_score = _baseline_score(base, X, 1)
    config = PairRouterConfig(
        rho=0.40,
        epochs=3,
        learning_rate=5e-3,
        mala_steps=1,
        replay_refresh=0.10,
        family_width=4,
        base_init_sd=0.01,
        open_warmup_epochs=1,
        posterior_sd_min=0.0,
    )
    first = fit_pair_residual_router(
        X,
        NAMES,
        baseline_state,
        baseline_score=baseline_score,
        baseline_orientation=1,
        seed=23,
        config=config,
    )
    second = fit_pair_residual_router(
        X,
        NAMES,
        baseline_state,
        baseline_score=baseline_score,
        baseline_orientation=1,
        seed=23,
        config=config,
    )
    for name in (
        "score",
        "logit",
        "contributions",
        "gates",
        "pair_transfers",
        "pair_context_residuals",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.objective_history == second.objective_history
    assert first.health["healthy"]
    assert first.diagnostics["contribution_reconstruction_max_abs"] <= 1e-8
    assert first.diagnostics["base_state_max_abs_change"] == 0.0
    assert np.max(
        np.abs(first.routed_family_contributions - first.base_family_contributions * first.gates)
    ) <= 1e-12
    assert np.max(
        np.abs(first.aligned_bias + first.contributions.sum(axis=1) - first.logit)
    ) <= 1e-8
    for name, value in baseline_state.items():
        assert np.array_equal(first.state[f"base::{name}"], value)
    replay = predict_pair_residual_router(first, X, baseline_score=baseline_score)
    for name in (
        "score",
        "logit",
        "contributions",
        "gates",
        "pair_transfers",
        "pair_context_residuals",
    ):
        assert np.array_equal(replay[name], getattr(first, name))
    assert replay["reconstruction_max_abs"] <= 1e-8

    graph_config = PairRouterConfig(
        epochs=0,
        graph_weight=0.25,
        family_width=4,
        base_init_sd=0.01,
        posterior_sd_min=0.0,
    )
    try:
        fit_pair_residual_router(
            X,
            NAMES,
            baseline_state,
            baseline_score=baseline_score,
            baseline_orientation=1,
            seed=23,
            config=graph_config,
            laplacian=None,
        )
    except ValueError as exc:
        assert "aligned fixed Laplacian" in str(exc)
    else:
        raise AssertionError("graph_weight>0 accepted a missing Laplacian")

    # A real symmetric graph must participate in backward without making the
    # fit non-deterministic or mutating the frozen base.
    n = len(X)
    adjacency = sparse.diags(
        [np.ones(n - 1), np.ones(n - 1)],
        offsets=[-1, 1],
        shape=(n, n),
        format="csr",
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    graph_laplacian = sparse.diags(degree, format="csr") - adjacency
    graph_fit = fit_pair_residual_router(
        X,
        NAMES,
        baseline_state,
        baseline_score=baseline_score,
        baseline_orientation=1,
        seed=23,
        config=PairRouterConfig(
            rho=0.25,
            epochs=2,
            learning_rate=5e-3,
            deem_weight=0.0,
            graph_weight=0.5,
            trust_weight=0.1,
            open_weight=0.0,
            l2_weight=0.0,
            open_warmup_epochs=0,
            family_width=4,
            base_init_sd=0.01,
            posterior_sd_min=0.0,
        ),
        laplacian=graph_laplacian,
    )
    assert graph_fit.health["healthy"]
    assert any(abs(row["graph_raw"]) > 0.0 for row in graph_fit.objective_history)
    assert any(row["grad_norm_before_clip"] > 0.0 for row in graph_fit.objective_history)
    for name, value in baseline_state.items():
        assert np.array_equal(graph_fit.state[f"base::{name}"], value)

    nonsymmetric = graph_laplacian.copy().tolil()
    nonsymmetric[0, 2] = 1.0
    nonsymmetric = nonsymmetric.tocsr()
    for invalid, message in (
        (nonsymmetric, "symmetric"),
        (
            sparse.csr_matrix(
                ([np.nan], ([0], [0])), shape=(len(X), len(X))
            ),
            "finite",
        ),
    ):
        try:
            fit_pair_residual_router(
                X,
                NAMES,
                baseline_state,
                baseline_score=baseline_score,
                baseline_orientation=1,
                seed=23,
                config=graph_config,
                laplacian=invalid,
            )
        except ValueError as exc:
            assert message in str(exc).lower()
        else:
            raise AssertionError(f"invalid Laplacian accepted: expected {message}")


def _test_artifact_exact_alias_and_negative_orientation() -> None:
    X, base, baseline_state = _fixture(seed=31)
    config = PairRouterConfig(
        rho=0.0,
        epochs=0,
        family_width=4,
        base_init_sd=0.01,
        posterior_sd_min=0.0,
    )
    for orientation in (1, -1):
        reconstructed = _baseline_score(base, X, orientation)
        # Frozen artifacts can legitimately differ from a reconstruction by a
        # final-bit rounding choice.  The alias must preserve those bytes.
        frozen = np.nextafter(reconstructed, np.ones_like(reconstructed))
        result = fit_pair_residual_router(
            X,
            NAMES,
            baseline_state,
            baseline_score=frozen,
            baseline_orientation=orientation,
            seed=31,
            config=config,
        )
        assert np.array_equal(result.score, frozen)
        assert result.orientation == orientation
        assert result.diagnostics["baseline_alias_max_abs"] == 0.0
        assert result.diagnostics["baseline_state_identity_max_abs"] <= 1e-12


def _test_label_firewall() -> None:
    assert LABEL_MODULE not in sys.modules
    source_path = ROOT / "spectral_utils/deem_b3_pair_router.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any("label" in ast.unparse(node).lower() for node in imports)
    assert LABEL_MODULE not in sys.modules


def main() -> None:
    assert LABEL_MODULE not in sys.modules
    _test_exact_identities_and_gate_geometry()
    _test_self_free_jacobian_and_context_dependency()
    _test_fit_determinism_graph_contract_and_frozen_base()
    _test_artifact_exact_alias_and_negative_orientation()
    _test_label_firewall()
    print("DEEM-B3 pair-router mechanical tests: PASS")


if __name__ == "__main__":
    main()
