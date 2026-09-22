#!/usr/bin/env python3
"""Corrected pre-label alias for the frozen H2/H3 PRMBench diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.token_local_fusion import prepare_localization_cell  # noqa: E402
from scripts.reasoning_localization import run_h3_llama_transfer as h3  # noqa: E402
from scripts.reasoning_localization import run_h3_prmbench_diagnostic as v1  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as p2c  # noqa: E402


EXPERIMENT = v1.EXPERIMENT
H0, H2, H3 = v1.H0, v1.H2, v1.H3
ARMS = v1.ARMS
PRM_CELL = v1.PRM_CELL
ROOT = p1.PROGRAM_ROOT / "phase_2/transfer/h3_prmbench_v2"
REGISTRY = (
    ROOT.parent / "H3_PRMBENCH_DIAGNOSTIC_EXECUTION_REGISTRY_AMENDMENT_V2.json"
)


def load_contract() -> dict[str, Any]:
    payload = json.loads(REGISTRY.read_text())
    required = {
        "schema": "reasoning-localization-h3-prmbench-diagnostic-execution-v2",
        "status": "FROZEN_BEFORE_RUN",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "arms": list(ARMS),
        "cell_id": PRM_CELL,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise v1.DiagnosticError(f"execution registry mismatch: {key}")
    return payload


def freeze_scores(contract: Mapping[str, Any]) -> dict[str, Any]:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    score_root = ROOT / "score_freeze"
    score_root.mkdir(parents=True)
    release = Path(contract["release_root"])
    qwen_alias = h3.qwen_alias(release)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    records = {str(row["cell_id"]): row for row in manifest["cells"]}
    source = records[PRM_CELL]
    input_path = input_root / source["artifact_path"]
    cell = load_prepared_localization_cell(input_path, source)

    preparation = prepare_localization_cell(cell)
    h0_token = p2c.fit_local_equal_family(preparation).token_risk
    h0_top5_local = p1.topk_step_mean(
        h0_token, cell.segment_starts, cell.segment_ends, k=5
    )
    h0_top5_control = p1.combine_with_common_detector(cell, h0_top5_local)
    h0_local, h0_combined, h2_local, h3_local, diagnostics = h3.score_cell(cell)
    h2_combined = p1.combine_with_common_detector(cell, h2_local)
    h3_combined = p1.combine_with_common_detector(cell, h3_local)

    phase1_path = (
        p1.PROGRAM_ROOT
        / "phase_1/r2_family6_top5_current/score_freeze/cells"
        / PRM_CELL
        / "scores.npz"
    )
    phase1 = load_npz_no_pickle(phase1_path)
    if tuple(phase1["row_ids"].astype(str)) != tuple(map(str, cell.row_ids)):
        raise v1.DiagnosticError("Phase-1 top-five control row alias failed")
    if not np.array_equal(phase1["segment_offsets"], cell.segment_offsets):
        raise v1.DiagnosticError("Phase-1 top-five control segment alias failed")
    phase1_alias = float(
        np.max(np.abs(h0_top5_control - phase1["combined_step_scores"]))
    )
    if phase1_alias > 1e-12:
        raise v1.DiagnosticError(
            f"Phase-1 top-five control score alias failed: {phase1_alias}"
        )

    arrays = {
        "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
        "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
        "segment_lengths": np.asarray(
            cell.segment_ends - cell.segment_starts, dtype="<i8"
        ),
        "h0_top5_control": np.asarray(h0_top5_control, dtype="<f8"),
        "h0_combined": np.asarray(h0_combined, dtype="<f8"),
        "h2_combined": np.asarray(h2_combined, dtype="<f8"),
        "h3_combined": np.asarray(h3_combined, dtype="<f8"),
    }
    target = score_root / "cells" / PRM_CELL
    target.mkdir(parents=True)
    score_sha = atomic_write_npz(target / "scores.npz", arrays)
    cell_manifest = {
        "schema": "reasoning-localization-h3-prmbench-cell-v2",
        "cell_id": PRM_CELL,
        "population_id": str(cell.population_id),
        "model_id": str(cell.model_id),
        "n_rows": len(cell.row_ids),
        "n_steps": len(h0_combined),
        "prepared_input": str(input_path),
        "prepared_input_sha256": sha256_file(input_path),
        "score_sha256": score_sha,
        "labels_seen": False,
        "phase1_h0_top5_control_alias_max_abs_error": phase1_alias,
        "phase1_h0_source": str(phase1_path.relative_to(REPO)),
        "phase1_h0_source_sha256": sha256_file(phase1_path),
        "qwen_processbench_alias": qwen_alias,
        "diagnostics": diagnostics,
    }
    atomic_write_json(target / "CELL_MANIFEST.json", cell_manifest)
    frozen = {
        "schema": "reasoning-localization-h3-prmbench-score-freeze-v2",
        "status": "FROZEN_BEFORE_LABEL_OPEN",
        "labels_seen": False,
        "cell_id": PRM_CELL,
        "score_sha256": score_sha,
        "cell_manifest_sha256": sha256_file(target / "CELL_MANIFEST.json"),
        "phase1_h0_score_alias_max_abs_error": phase1_alias,
        "qwen_processbench_alias": qwen_alias,
        "execution_registry_sha256": sha256_file(REGISTRY),
        "superseded_v1_failure_sha256": contract[
            "superseded_v1_failure_sha256"
        ],
    }
    atomic_write_json(ROOT / "SCORE_FREEZE_MANIFEST.json", frozen)
    return frozen


def main() -> None:
    contract = load_contract()
    freeze = freeze_scores(contract)
    (ROOT / "evaluation").mkdir()
    v1.ROOT = ROOT
    v1.REGISTRY = REGISTRY
    v1.evaluate_scores(freeze, contract)


if __name__ == "__main__":
    main()
