#!/usr/bin/env python3
"""Dataset-free mechanical tests for the additive DEEM-B3 MoE extension."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_moe import (  # noqa: E402
    DeemMoEConfig,
    _FamilyRouterEnergy,
    fit_deem_b3_moe,
    predict_deem_b3_moe,
    sparsemax,
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


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(20260824))
    X = rng.normal(size=(72, len(NAMES)))
    baseline = _FamilyAdditiveEnergy(NAMES, ContinuousDeemConfig(), seed=3)
    baseline_state = baseline.state_dict_numpy()
    tensor = torch.as_tensor(X, dtype=torch.float64)

    # Every router is born at the B3 identity, and alpha=0 stays an exact
    # identity even if router parameters are later changed.
    with torch.no_grad():
        base_logit, base_atomic, _ = baseline.logit(tensor)
    for router in (
        "uniform",
        "static",
        "loo_scalar",
        "self_scalar",
        "dense_loo",
        "dense_full",
        "global_residual",
        "pairwise_residual",
        "multinomial_diagonal",
        "multinomial_offdiag",
        "multinomial_full",
    ):
        config = DeemMoEConfig(router=router, router_strength=0.75, epochs=0)
        model = _FamilyRouterEnergy(NAMES, config, seed=3)
        model.load_baseline_state(baseline_state)
        with torch.no_grad():
            logit, values = model.logit(tensor)
        assert torch.equal(logit, base_logit)
        assert torch.equal(values["routed_atomic"], base_atomic)
        assert torch.equal(values["gates"], torch.ones_like(values["gates"]))

        zero = _FamilyRouterEnergy(
            NAMES, replace(config, router_strength=0.0), seed=3
        )
        zero.load_baseline_state(baseline_state)
        with torch.no_grad():
            zero.router_bias.copy_(torch.linspace(-2.0, 2.0, len(zero.family_order)))
            zero.router_weight.fill_(3.0)
            zero.router_matrix.copy_(
                torch.arange(
                    len(zero.family_order) ** 2, dtype=torch.float64
                ).reshape(len(zero.family_order), len(zero.family_order))
            )
            zero.multinomial_matrix.fill_(0.25)
            zero.multinomial_bias.copy_(
                torch.linspace(-0.5, 0.5, 2, dtype=torch.float64)[None, :].repeat(
                    len(zero.family_order), 1
                )
            )
            zero_logit, zero_values = zero.logit(tensor)
        assert torch.equal(zero_logit, base_logit)
        assert torch.equal(zero_values["routed_atomic"], base_atomic)

    active = _FamilyRouterEnergy(
        NAMES,
        DeemMoEConfig(router="loo_scalar", router_strength=0.60, epochs=0),
        seed=3,
    )
    active.load_baseline_state(baseline_state)
    with torch.no_grad():
        active.router_weight.copy_(
            torch.linspace(-1.0, 1.0, len(active.family_order), dtype=torch.float64)
        )
        active.router_bias.copy_(
            torch.linspace(0.3, -0.3, len(active.family_order), dtype=torch.float64)
        )
        _, active_values = active.logit(tensor)
    gates = active_values["gates"].numpy()
    assert np.max(np.abs(gates.sum(axis=1) - len(active.family_order))) <= 1e-12
    assert np.std(gates) > 0.0
    reconstruction = (
        active.base.b.item()
        + active_values["routed_atomic"].sum(dim=1).numpy()
        - active.logit(tensor)[0].detach().numpy()
    )
    assert np.max(np.abs(reconstruction)) <= 1e-12

    # The DEEM-like multinomial layer is identity-initialized.  Sparsemax is
    # over the two class states inside each family, not over families.  A
    # non-zero off-diagonal tensor must induce bounded cross-family residuals
    # while preserving the exact atomic decomposition.
    multinomial = _FamilyRouterEnergy(
        NAMES,
        DeemMoEConfig(
            router="multinomial_offdiag",
            normalizer="sparsemax",
            router_strength=0.50,
            epochs=0,
            train_router_bias=False,
        ),
        seed=3,
    )
    multinomial.load_baseline_state(baseline_state)
    with torch.no_grad():
        initial_logit, initial_values = multinomial.logit(tensor)
        assert torch.equal(initial_logit, base_logit)
        assert torch.equal(initial_values["routed_atomic"], base_atomic)
        assert torch.allclose(
            initial_values["family_state_input"].sum(dim=2),
            torch.ones((len(X), len(multinomial.family_order)), dtype=torch.float64),
        )
        multinomial.multinomial_matrix.copy_(
            torch.linspace(
                -0.4,
                0.4,
                multinomial.multinomial_matrix.numel(),
                dtype=torch.float64,
            ).reshape_as(multinomial.multinomial_matrix)
        )
        mixed_logit, mixed_values = multinomial.logit(tensor)
    assert not torch.equal(mixed_logit, base_logit)
    assert float(mixed_values["family_state_delta"].abs().max()) <= 1.0 + 1e-12
    assert torch.allclose(
        mixed_values["family_state_output"].sum(dim=2),
        torch.ones((len(X), len(multinomial.family_order)), dtype=torch.float64),
    )
    assert torch.max(
        torch.abs(
            multinomial.base.b
            + mixed_values["routed_atomic"].sum(dim=1)
            - mixed_logit
        )
    ) <= 1e-12

    projected = sparsemax(
        torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    )
    assert torch.allclose(projected.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.equal(projected[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(projected[1], torch.full((3,), 1.0 / 3.0, dtype=torch.float64))

    # A short warm-started fit exercises persistent MALA, deterministic replay,
    # frozen-expert isolation, serialization, orientation, and reconstruction.
    fit_config = DeemMoEConfig(
        router="loo_scalar",
        router_strength=0.50,
        epochs=4,
        learning_rate=1e-2,
        posterior_sd_min=0.0,
        anchor_tolerance=1e-12,
    )
    first = fit_deem_b3_moe(X, NAMES, baseline_state, seed=3, config=fit_config)
    second = fit_deem_b3_moe(X, NAMES, baseline_state, seed=3, config=fit_config)
    assert np.array_equal(first.score, second.score)
    assert np.array_equal(first.gates, second.gates)
    assert first.health["healthy"]
    assert first.health["contribution_reconstruction_max_abs"] <= 1e-8
    assert first.diagnostics["gate_sum_max_abs_error"] <= 1e-10
    for name, value in baseline_state.items():
        assert np.array_equal(first.state[f"base::{name}"], value)
    predicted = predict_deem_b3_moe(first, X)
    assert np.array_equal(predicted["score"], first.score)
    assert np.array_equal(predicted["gates"], first.gates)
    assert predicted["reconstruction_max_abs"] <= 1e-8

    multinomial_fit = fit_deem_b3_moe(
        X,
        NAMES,
        baseline_state,
        seed=3,
        config=DeemMoEConfig(
            router="multinomial_offdiag",
            normalizer="sparsemax",
            router_strength=0.50,
            epochs=3,
            learning_rate=1e-2,
            train_router_bias=False,
            posterior_sd_min=0.0,
            anchor_tolerance=1e-12,
        ),
    )
    multinomial_predicted = predict_deem_b3_moe(multinomial_fit, X)
    assert np.array_equal(multinomial_predicted["score"], multinomial_fit.score)
    assert np.array_equal(
        multinomial_predicted["family_state_delta"],
        multinomial_fit.family_state_delta,
    )

    # Preserve the physical target firewall in the new fit module.
    source = (ROOT / "spectral_utils/deem_b3_moe.py").read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any("label" in ast.unparse(node).lower() for node in imports)
    print("DEEM-B3 MoE mechanical tests: PASS")


if __name__ == "__main__":
    main()
