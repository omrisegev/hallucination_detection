#!/usr/bin/env python3
"""Static gates for the frozen protocol, registry, CLI split, and run contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    ARM_SPECS,
    LAMBDA_GRID,
    SEEDS,
    ResidualGraphDeemError,
    canonical_sha256,
)
from spectral_utils.residual_graph_deem_data import load_registry  # noqa: E402
from spectral_utils.residual_graph_deem_labels import require_complete_score_freeze  # noqa: E402
from scripts.run_residual_graph_deem_24cell_v1 import (  # noqa: E402
    expected_stems,
    source_hash,
    write_cell_checkpoint,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  [PASS] {message}")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    output = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            output.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            output.add(node.module or "")
    return output


def main():
    registry = load_registry(ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    require([spec.arm_id for spec in ARM_SPECS] == [f"B{i}" for i in range(4)] + [f"G{i}" for i in range(6)],
            "exact B0-B3/G0-G5 arm roster")
    require(tuple(registry["solver"]["seeds"]) == SEEDS, "exact seeds 0..4")
    require(tuple(registry["graph"]["lambdas"]) == LAMBDA_GRID, "exact lambda grid")
    require(registry["graph"]["primary_k"] == 7 and registry["graph"]["sensitivity_k"] == [5, 10, 15],
            "headline/sensitivity k contract")
    require(registry["evaluation"]["whole_search_B"] == 199
            and registry["evaluation"]["promotion_B"] == 999, "B=199/999 promotion contract")
    require(registry["graph"]["claim_min_healthy_cells"] == 22, "22/24 graph-health claim threshold")
    require(len(registry["synthetic"]["worlds"]) == 10, "ten synthetic worlds frozen")
    phase0_fixture = {"nominated_lambdas": {"target": .1, "nuisance": .3, "family": .03}}
    required_stems = expected_stems(phase0_fixture)
    require(len(required_stems) == 255,
            "all core, lambda, control, k, and stable-inventory artifacts required per cell")
    records = [
        {
            "stem": stem, "status": "complete", "array_sha256": "a" * 64,
            "health": {"healthy": stem != sorted(required_stems)[0]},
        }
        for stem in sorted(required_stems)
    ]
    with tempfile.TemporaryDirectory() as temporary:
        write_cell_checkpoint(
            Path(temporary), "fixture_cell", phase0_fixture, "b" * 64, records
        )
        require(
            not (Path(temporary) / "fits/fixture_cell/CELL_COMPLETE.json").exists(),
            "unhealthy fit cannot produce a cell-complete checkpoint",
        )
        invalid_freeze = {
            "status": "complete", "debug": False, "cells": ["fixture_cell"],
            "missing_seeds": [], "incomplete_fits": [],
            "unhealthy_fits": [records[0]], "missing_artifacts": [],
            "artifacts": [],
        }
        invalid_freeze["content_sha256"] = canonical_sha256(invalid_freeze)
        freeze_path = Path(temporary) / "SCORE_FREEZE_MANIFEST.json"
        freeze_path.write_text(json.dumps(invalid_freeze), encoding="utf-8")
        try:
            require_complete_score_freeze(freeze_path, ["fixture_cell"])
        except ResidualGraphDeemError:
            rejected = True
        else:
            rejected = False
        require(rejected, "label-sidecar gate rejects unhealthy score freeze")
    require(all(cell["inventory_sha256"] == canonical_sha256({
        "feature_names": cell["feature_names"], "confidence_signs": cell["confidence_signs"]
    }) for cell in registry["cells"]), "every inventory hash recomputes")
    require(all(cell["source"]["admission_contract_sha256"] for cell in registry["cells"]),
            "every source has an admission hash")
    runner = ROOT / "scripts/run_residual_graph_deem_24cell_v1.py"
    modules = imported_modules(runner)
    require(not any("residual_graph_deem_labels" in module for module in modules),
            "Stage-A import graph excludes evaluation label module")
    evaluator = ROOT / "scripts/evaluate_residual_graph_deem_24cell_v1.py"
    require(any("residual_graph_deem_labels" in module for module in imported_modules(evaluator)),
            "only evaluator-side code imports label sidecars")
    runner_text = runner.read_text(encoding="utf-8")
    require("--sidecar" not in runner_text, "Stage-A CLI exposes no sidecar path")
    protocol = (ROOT / "docs/experiments/RESIDUAL_GRAPH_DEEM_24CELL_V1.md").read_text(encoding="utf-8")
    require("iu_pcr_full30" not in protocol and "deem30_" not in protocol and "stable26" not in protocol,
            "retired fixed-width arm names absent")
    require("stable_inventory_minus4" in protocol, "inventory sensitivity is explicitly named")
    require("cf01350f5bc141908e3f0563c1bc3037148fbad3a30c4eb05c63cd3c13a51e65" in protocol,
            "spilled raw-source byte hash appears in protocol")
    require(len(source_hash()) == 64, "run definition hashes every output-generating source")
    print("\nALL RESIDUAL-GRAPH DEEM PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
