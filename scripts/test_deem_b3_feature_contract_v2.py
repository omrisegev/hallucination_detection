#!/usr/bin/env python3
"""Mechanical tests for the Feature Contract V2 B3 baseline."""

from __future__ import annotations

import numpy as np

from spectral_utils.deem_b3_feature_contract_v2 import (
    BLOCK_ORDER,
    block_index_map,
    equal_block_anchor,
    fit_v2_b3,
    prepare_v2_risk,
)
from spectral_utils.residual_graph_deem import ContinuousDeemConfig


def main() -> None:
    names = (
        "entropy_common", "entropy_support_delta", "logprob_margin",
        "rpdi", "sw_var_peak", "cusum_max",
        "epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled",
        "epr_energy", "min_energy", "cusum_max_energy",
    )
    groups = block_index_map(names)
    assert tuple(groups) == BLOCK_ORDER
    rng = np.random.default_rng(7)
    X = rng.normal(size=(96, len(names)))
    X[:, 1] *= 0.03
    risk, transform = prepare_v2_risk(X, names)
    common = names.index("entropy_common")
    delta = names.index("entropy_support_delta")
    assert transform.scale[delta] == transform.scale[common]
    assert np.std(risk[:, delta]) < 0.1
    anchor = equal_block_anchor(risk, names)
    perturbed = risk.copy()
    perturbed[:, delta] += 1000.0
    assert np.array_equal(anchor, equal_block_anchor(perturbed, names))
    config = ContinuousDeemConfig(epochs=3, mala_steps=1)
    first = fit_v2_b3(risk, names, seed=0, config=config)
    second = fit_v2_b3(risk, names, seed=0, config=config)
    assert np.array_equal(first.score, second.score)
    assert first.health["healthy"]
    assert first.health["contribution_reconstruction_max_abs"] <= 1e-8
    assert np.max(np.abs(first.aligned_bias + first.contributions.sum(axis=1) - first.logit)) <= 1e-8
    print("deem_b3_feature_contract_v2 mechanical tests: PASS")


if __name__ == "__main__":
    main()
