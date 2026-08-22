#!/usr/bin/env python3
"""Focused contract and mechanical tests for the graph-free B0-B3 pivot."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deem_vs_iupcr_24cell_v1 import (  # noqa: E402
    b0_score, expected_stems, load_experiment_config, source_hash,
)
from scripts.evaluate_deem_vs_iupcr_24cell_v1 import holm, whole_search_null  # noqa: E402
from spectral_utils.deem_adapter import hard_adapter020_config, repaired_soft_adapter020_config  # noqa: E402
from spectral_utils.residual_graph_deem import ContinuousDeemConfig, donor_risk_matrix, fit_continuous_deem  # noqa: E402
from spectral_utils.residual_graph_deem_data import load_registry  # noqa: E402


def main() -> None:
    config = load_experiment_config(ROOT / "configs/deem_vs_iupcr_24cell_v1.json")
    registry = load_registry(ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    assert [row["id"] for row in config["arms"]] == ["B0", "B1", "B2", "B3"]
    assert len(expected_stems(config)) == 20
    assert all(not stem.startswith("G") for stem in expected_stems(config))
    assert len(registry["cells"]) == 24 and sum(row["n_rows"] for row in registry["cells"]) == 48607
    assert sorted({row["n_features"] for row in registry["schemas"]}) == [19, 27, 28, 29, 30]
    hard, soft = hard_adapter020_config(), repaired_soft_adapter020_config()
    assert hard.learning_rate == 1e-3 and soft.learning_rate == 1e-4
    assert hard.epochs == soft.epochs == 100 and hard.sampler_steps == soft.sampler_steps == 5
    runner = (ROOT / "scripts/run_deem_vs_iupcr_24cell_v1.py").read_text(encoding="utf-8")
    assert "residual_graph_deem_labels" not in runner
    assert "sidecar" not in runner.lower()
    chain = (ROOT / "cluster/submit_deem_vs_iupcr_chain_v1.sh").read_text(encoding="utf-8")
    assert "phase0" not in chain and "afterok" in chain and "evaluate-999" not in chain
    manifest = json.loads((ROOT / "cluster/deem_vs_iupcr_24cell_v1_manifest.json").read_text(encoding="utf-8"))
    assert manifest["arms"] == ["B0", "B1", "B2", "B3"]
    protocol = (ROOT / "docs/experiments/DEEM_VS_IUPCR_24CELL_V1.md").read_text(encoding="utf-8")
    assert "CLOSE_RESIDUAL_GRAPH_EXTENSION_SPECIFICITY_FAILURE" in protocol
    assert "It does not\nfalsify B3" in protocol
    assert "Amendment A1" in protocol

    names = tuple(registry["schemas"][0]["feature_names"])
    rng = np.random.Generator(np.random.PCG64(20260821))
    raw = rng.normal(size=(96, len(names))); raw[:, 0] = 1.0
    X, _, transform = donor_risk_matrix(raw, raw, names)
    assert transform.constant_mask[0] and transform.scale[0] == 1.0
    b0, health = b0_score(X, names)
    assert health["healthy"] and np.isfinite(b0).all()
    small = replace(ContinuousDeemConfig(), epochs=3)
    b3a = fit_continuous_deem(X, names, seed=0, config=small)
    b3b = fit_continuous_deem(X, names, seed=0, config=small)
    assert np.array_equal(b3a.score, b3b.score)
    reconstruction = np.max(np.abs(b3a.aligned_bias + b3a.contributions.sum(axis=1) - b3a.logit))
    assert reconstruction <= 1e-8 and b3a.health["healthy"]
    targets = {"qa": np.array([0, 1] * 4), "math": np.array([1, 0] * 4)}
    bundles = {"qa": SimpleNamespace(task_type="QA", dataset_family="qa_family"),
               "math": SimpleNamespace(task_type="math", dataset_family="math_family")}
    scores = {}
    for cell, y in targets.items():
        scores[cell] = {"B0": y * .6 + np.arange(8) * .001,
                        "B1": y * .4 + np.arange(8) * .001,
                        "B2": y * .5 + np.arange(8) * .001,
                        "B3": y * .9 + np.arange(8) * .001}
    null_arrays = {
        name: {cell: np.column_stack([np.roll(y, shift) for shift in (1, 2, 3)])
               for cell, y in targets.items()}
        for name in ("exact", "crt", "family_group")
    }
    null = whole_search_null(targets, bundles, scores, null_arrays, B=3)
    assert all("p_by_statistic" in null[name] for name in null_arrays)
    assert holm({"a": .01, "b": .04, "c": .03}) == {"a": .03, "c": .06, "b": .06}
    assert len(source_hash()) == 64

    # Amendment A1: B2 health is recorded, not blocking; B0/B1/B3 stay gated.
    from scripts.run_deem_vs_iupcr_24cell_v1 import _fit_acceptable
    collapsed_b2 = {"status": "complete", "stem": "B2__seed0",
                    "health": {"healthy": False, "score_finite": True, "score_sd": 1e-6}}
    assert _fit_acceptable(collapsed_b2)
    assert not _fit_acceptable({**collapsed_b2,
                                "health": {"healthy": False, "score_finite": False}})
    assert not _fit_acceptable({"status": "failed", "stem": "B2__seed0", "health": {}})
    assert not _fit_acceptable({"status": "complete", "stem": "B1__seed0",
                                "health": {"healthy": False, "score_finite": True}})
    assert _fit_acceptable({"status": "complete", "stem": "B1__seed0",
                            "health": {"healthy": True, "score_finite": True}})
    assert _fit_acceptable({"status": "complete", "stem": "B3__seed0",
                            "health": {"healthy": True}})
    print("deem-vs-iupcr focused tests: PASS")


if __name__ == "__main__":
    main()
