#!/usr/bin/env python3
"""Mechanical tests for the target-free DEEM-B3 Crossed-Rook v1 arm."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_crossed_rook import (  # noqa: E402
    CORE_FEATURES,
    CrossedRookConfig,
    _CrossedRookEnergy,
    fit_deem_b3_crossed_rook,
    predict_deem_b3_crossed_rook,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    _FamilyAdditiveEnergy,
    equal_family_risk_anchor,
)


NAMES = CORE_FEATURES + (
    "mean_top1_logprob",
    "mean_logprob_entropy",
    "logprob_margin",
    "trace_length",
)


def baseline_orientation_and_score(model, tensor: torch.Tensor, X: np.ndarray):
    with torch.no_grad():
        ell, _, _ = model.logit(tensor)
    raw = ell.numpy()
    q = 1.0 / (1.0 + np.exp(-np.clip(raw, -700.0, 700.0)))
    anchor = equal_family_risk_anchor(X, NAMES)
    high = float(np.sum(q * anchor) / np.sum(q))
    low = float(np.sum((1.0 - q) * anchor) / np.sum(1.0 - q))
    orientation = 1 if high > low else -1
    if orientation < 0:
        q = 1.0 - q
    return orientation, q


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(20260825))
    X = rng.normal(size=(96, len(NAMES)))
    tensor = torch.as_tensor(X, dtype=torch.float64)
    baseline = _FamilyAdditiveEnergy(NAMES, ContinuousDeemConfig(), seed=7)
    baseline_state = baseline.state_dict_numpy()
    orientation, baseline_score = baseline_orientation_and_score(baseline, tensor, X)
    with torch.no_grad():
        baseline_logit, baseline_atomic, _ = baseline.logit(tensor)

    models = {}
    for mode in ("alias", "row_only", "column_only", "crossed", "nonrook_18"):
        config = CrossedRookConfig(
            mode=mode,
            strength=0.0 if mode == "alias" else 1.0,
            epochs=0,
            posterior_sd_min=0.0,
            anchor_tolerance=1e-12,
        )
        model = _CrossedRookEnergy(NAMES, config, seed=7)
        model.load_baseline_state(baseline_state)
        model.fit_coordinate_transform(tensor)
        models[mode] = model
        with torch.no_grad():
            logit, values = model.logit(tensor)
        assert torch.equal(logit, baseline_logit)
        assert torch.equal(values["contributions"], baseline_atomic)
        assert torch.count_nonzero(model.edge_weight) == 0
        assert all(not parameter.requires_grad for parameter in model.base.parameters()) is False
        parameters = model.parameters()
        assert all(not parameter.requires_grad for parameter in model.base.parameters())
        assert len(parameters) == (0 if mode == "alias" else 1)

    row = set(models["row_only"].edge_pairs)
    column = set(models["column_only"].edge_pairs)
    rook = set(models["crossed"].edge_pairs)
    nonrook = set(models["nonrook_18"].edge_pairs)
    all_pairs = {(left, right) for left in range(9) for right in range(left + 1, 9)}
    assert len(row) == len(column) == 9
    assert row.isdisjoint(column) and rook == row | column and len(rook) == 18
    assert len(nonrook) == 18 and rook.isdisjoint(nonrook)
    assert rook | nonrook == all_pairs
    for edge_set in (rook, nonrook):
        degrees = [sum(index in edge for edge in edge_set) for index in range(9)]
        assert degrees == [4] * 9

    # Every active coefficient has a finite, nonzero first-order path at the
    # exact B3 initialization.
    held_weights = torch.linspace(-1.0, 1.0, len(X), dtype=torch.float64)
    for mode in ("row_only", "column_only", "crossed", "nonrook_18"):
        model = models[mode]
        model.parameters()
        logit, _ = model.logit(tensor)
        objective = (logit * held_weights).sum()
        objective.backward()
        gradient = model.edge_weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == len(gradient)
        model.edge_weight.grad = None

    # Nonzero theta induces a bounded quadratic correction and preserves the
    # exact base+residual and atomic decompositions.
    crossed = models["crossed"]
    with torch.no_grad():
        crossed.edge_weight.copy_(
            torch.linspace(-0.4, 0.4, len(crossed.edge_pairs), dtype=torch.float64)
        )
        mixed_logit, mixed = crossed.logit(tensor)
    correction = mixed["correction"].numpy()
    assert np.std(correction) > 0.0
    assert np.max(np.abs(correction)) <= 0.5 + 1e-12
    assert torch.max(torch.abs(mixed_logit - baseline_logit - mixed["correction"])) <= 1e-12
    assert torch.max(
        torch.abs(crossed.base.b + mixed["contributions"].sum(dim=1) - mixed_logit)
    ) <= 1e-12

    alias = fit_deem_b3_crossed_rook(
        X,
        NAMES,
        baseline_state,
        baseline_orientation=orientation,
        baseline_score=baseline_score,
        seed=7,
        config=CrossedRookConfig(
            mode="alias",
            strength=0.0,
            epochs=0,
            posterior_sd_min=0.0,
            anchor_tolerance=1e-12,
        ),
    )
    assert np.array_equal(alias.score, baseline_score)
    assert alias.health["healthy"]
    assert alias.health["mala_acceptance_mean"] is None
    assert alias.diagnostics["saved_alias_max_abs"] <= 1e-12

    # Short deterministic target-free fit exercises MALA, Adam, clipping,
    # serialization/replay, fixed orientation and frozen-base isolation.
    fit_config = CrossedRookConfig(
        mode="crossed",
        epochs=4,
        learning_rate=2e-3,
        trust_weight=0.0,
        l2_weight=1e-4,
        posterior_sd_min=0.0,
        anchor_tolerance=1e-12,
    )
    first = fit_deem_b3_crossed_rook(
        X,
        NAMES,
        baseline_state,
        baseline_orientation=orientation,
        seed=7,
        config=fit_config,
    )
    second = fit_deem_b3_crossed_rook(
        X,
        NAMES,
        baseline_state,
        baseline_orientation=orientation,
        seed=7,
        config=fit_config,
    )
    assert np.array_equal(first.score, second.score)
    assert np.array_equal(first.edge_weights, second.edge_weights)
    assert np.std(first.correction) > 0.0
    assert first.orientation == orientation and first.health["healthy"]
    assert 0.0 < first.health["mala_acceptance_mean"] <= 1.0
    assert first.diagnostics["edge_weight_nonzero"] == 18
    assert first.diagnostics["gradient_norm_initial"] > 0.0
    for name, value in baseline_state.items():
        assert np.array_equal(first.state[f"base::{name}"], value)
    predicted = predict_deem_b3_crossed_rook(first, X)
    assert np.array_equal(predicted["score"], first.score)
    assert predicted["reconstruction_max_abs"] <= 1e-8
    assert predicted["residual_identity_max_abs"] <= 1e-8

    config = json.loads(
        (ROOT / "configs/deem_b3_crossed_rook_v1.json").read_text(encoding="utf-8")
    )
    assert [row["id"] for row in config["variants"]] == [
        "A0_B3_EXACT_ALIAS",
        "A1_ROW_9",
        "A2_COLUMN_9",
        "A3_CROSSED_ROOK_18",
        "A4_NONROOK_18_CONTROL",
    ]
    for row in config["variants"][1:]:
        arm = CrossedRookConfig(**row["config"])
        assert arm.optimizer == "adam" and arm.learning_rate == 2e-3
        assert arm.epochs == 100 and arm.trust_weight == 0.0
        assert arm.l2_weight == 1e-4 and arm.gradient_clip == 5.0
        assert arm.correction_cap == 0.5

    # The fit boundary may mention the scientific target firewall in prose,
    # but it must not import evaluators or any sidecar reader.
    for relative in (
        "spectral_utils/deem_b3_crossed_rook.py",
        "scripts/run_deem_b3_crossed_rook_v1.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(
            "evaluate" in name or "label_sidecar" in name for name in imported
        ), imported

    # Result JSON payloads must remain strict-JSON serializable (no NaN from
    # the zero-epoch alias).
    json.dumps(alias.health, allow_nan=False)
    json.dumps(alias.diagnostics, allow_nan=False)
    print("PASS: DEEM-B3 Crossed-Rook v1 mechanical tests")


if __name__ == "__main__":
    main()
