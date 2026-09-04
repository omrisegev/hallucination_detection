#!/usr/bin/env python3
"""Freeze all-eight ProcessBench scores for the Joint-or-flat coverage policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import scipy
import sklearn


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.joint_lsml_localization import run_existing_v1 as parent  # noqa: E402
from spectral_utils.joint_lsml_localization import JOINT_METHOD  # noqa: E402
from spectral_utils.joint_lsml_processbench_amendment import (  # noqa: E402
    COVERAGE_METHOD,
    COVERAGE_METHODS,
    fit_flat_fallback_and_controls,
    rename_candidate_method,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


EXPERIMENT_ID = "JOINT_LSML_PROCESSBENCH_AMENDMENT_V1"
PLAN = REPO / "configs/joint_lsml_processbench_amendment_v1.json"
PROTOCOL = REPO / "docs/experiments/JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md"
PRIOR_ORDER_AUDIT = REPO / "docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md"
RESULT_ROOT = parent.RESULT_ROOT / "processbench_amendment_v1"
REGISTRY = RESULT_ROOT / "EXECUTION_REGISTRY.json"
REGISTRATION_COMPLETE = RESULT_ROOT / "REGISTRATION_COMPLETE.json"
POLICY_LEDGER = RESULT_ROOT / "POLICY_LEDGER.json"
SCORE_ROOT = RESULT_ROOT / "score_freeze"
SCORE_MANIFEST = RESULT_ROOT / "SCORE_FREEZE_MANIFEST.json"
RUN_COMPLETE = RESULT_ROOT / "RUN_COMPLETE.json"
EXPECTED_BLOCKED = "processbench_math_qwen3_4b"


class AmendmentProtocolError(RuntimeError):
    pass


def _source_paths() -> tuple[str, ...]:
    return (
        "configs/joint_lsml_processbench_amendment_v1.json",
        "docs/experiments/JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md",
        "docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md",
        "scripts/joint_lsml_localization/run_processbench_amendment_v1.py",
        "scripts/joint_lsml_localization/evaluate_processbench_amendment_v1.py",
        "scripts/joint_lsml_localization/run_existing_v1.py",
        "spectral_utils/joint_lsml_processbench_amendment.py",
        "spectral_utils/joint_lsml_localization.py",
        "spectral_utils/joint_lsml.py",
        "spectral_utils/fusion_utils.py",
        "spectral_utils/upcr.py",
        "spectral_utils/fixed_application_pipelines.py",
        "spectral_utils/specrage_views.py",
        "spectral_utils/token_local_fusion.py",
        "spectral_utils/reconstruction_benchmark/io.py",
        "spectral_utils/reconstruction_benchmark/localization_contract.py",
        "tests/test_joint_lsml_processbench_amendment.py",
        "tests/test_joint_lsml_processbench_evaluation.py",
    )


def _source_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO / relative) for relative in _source_paths()}


def _parent_hashes() -> dict[str, str]:
    paths = {
        "execution_registry": parent.REGISTRY,
        "structural_ledger": parent.STRUCTURAL_LEDGER,
        "score_manifest": parent.SCORE_MANIFEST,
        "result_audit": parent.RESULT_ROOT / "INDEPENDENT_EVALUATION_RESULT_AUDIT.json",
        "final_complete": parent.RESULT_ROOT / "FINAL_COMPLETE.json",
        "sanitized_manifest": parent.SANITIZED_MANIFEST,
        "orientation": parent.ORIENTATION,
        "roster": parent.ROSTER,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _parent_weight_hashes(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    output = {}
    for cell_id in parent.PB_CELLS:
        cell = cells[cell_id]
        if cell_id == EXPECTED_BLOCKED:
            if cell.get("status") != "BLOCKED_NO_ADMISSIBLE_PARTITION":
                raise AmendmentProtocolError("expected blocked-cell status changed")
            continue
        if cell.get("status") != "FIT_COMPLETE" or not cell.get("structural_fit_pass"):
            raise AmendmentProtocolError(f"unexpected non-fitted parent cell: {cell_id}")
        output[cell_id] = payload_sha256(cell["weights"])
    if len(output) != 7:
        raise AmendmentProtocolError("parent fitted-weight roster is not seven cells")
    return output


def _json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if "payload_sha256" in payload and payload_sha256(body) != payload["payload_sha256"]:
        raise AmendmentProtocolError(f"noncanonical payload: {path}")
    return payload


def _verify_parent() -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    parent._verify_registration()
    parent.check()
    structural = _json_payload(parent.STRUCTURAL_LEDGER)
    cells = {row["cell_id"]: row for row in structural["cells"]}
    if set(parent.PB_CELLS) - set(cells):
        raise AmendmentProtocolError("parent structural ledger lacks a ProcessBench cell")
    blocked = [
        cell for cell in parent.PB_CELLS
        if cells[cell]["status"] != "FIT_COMPLETE" or not cells[cell].get("structural_fit_pass")
    ]
    if blocked != [EXPECTED_BLOCKED]:
        raise AmendmentProtocolError(f"unexpected parent blocked cells: {blocked}")
    if structural.get("processbench_panel_status") != "STRUCTURAL_NO_SCORE":
        raise AmendmentProtocolError("parent ProcessBench disposition changed")
    return structural, cells


def _firewall_audit() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("localization_postfreeze", "roc_auc", "average_precision", "evaluate_existing")
    offending = [name for name in imports if any(token in name for token in forbidden)]
    return {"status": "PASS" if not offending else "FAIL", "offending_imports": offending}


def register() -> None:
    if RESULT_ROOT.exists():
        raise AmendmentProtocolError("amendment result namespace already exists")
    _, cells = _verify_parent()
    plan = json.loads(PLAN.read_text())
    expected_plan = {
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT__POST_PRM_OPENED__PB_LABEL_FREE_AT_REGISTRATION",
        "candidate": COVERAGE_METHOD,
        "controls": list(COVERAGE_METHODS[1:]),
        "models": ["qwen3_4b", "qwen3_8b"],
        "subsets": ["gsm8k", "math", "olympiadbench", "omnimath"],
        "folds": 5,
        "fold_namespace": "joint-lsml-processbench-amendment-v1",
        "bootstrap_draws": 2000,
        "bootstrap_seed": 2026090408,
        "bootstrap_strata": "subset_x_frozen_fold",
        "detector": "max_token_risk",
        "locator": "argmax_fixed_top10_mean_step_risk",
        "fallback_cells": [EXPECTED_BLOCKED],
        "candidate_policies_seen": 2,
        "current_primary_contrasts": 3,
        "secondary_diagnostics": 1,
        "decision_state": {
            "HARM": "candidate_minus_iu_pcr_ci_high_below_zero",
            "DEVELOPMENT_SUPPORTED": "all_three_candidate_minus_control_ci_lows_above_or_equal_zero_and_candidate_minus_iu_pcr_point_positive",
            "INCONCLUSIVE": "otherwise",
        },
        "promotion_allowed": False,
        "generalization_claim_allowed": False,
    }
    exposure = plan.get("reporting", {}).get("cumulative_opened_exposure", {})
    observed_plan = {
        "scope": plan.get("scope"),
        "candidate": plan.get("candidate"),
        "controls": plan.get("controls"),
        "models": plan.get("processbench", {}).get("models"),
        "subsets": plan.get("processbench", {}).get("subsets"),
        "folds": plan.get("processbench", {}).get("folds"),
        "fold_namespace": plan.get("processbench", {}).get("fold_namespace"),
        "bootstrap_draws": plan.get("processbench", {}).get("bootstrap_draws"),
        "bootstrap_seed": plan.get("processbench", {}).get("bootstrap_seed"),
        "bootstrap_strata": plan.get("processbench", {}).get("bootstrap_strata"),
        "detector": plan.get("processbench", {}).get("detector"),
        "locator": plan.get("processbench", {}).get("locator"),
        "fallback_cells": plan.get("candidate_policy", {}).get("expected_fallback_cells"),
        "candidate_policies_seen": exposure.get("candidate_policies_seen"),
        "current_primary_contrasts": exposure.get("current_primary_contrasts"),
        "secondary_diagnostics": exposure.get("secondary_diagnostics"),
        "decision_state": plan.get("reporting", {}).get("decision_state"),
        "promotion_allowed": plan.get("promotion_allowed"),
        "generalization_claim_allowed": plan.get("generalization_claim_allowed"),
    }
    if observed_plan != expected_plan:
        raise AmendmentProtocolError(f"analysis plan contract mismatch: {observed_plan}")
    firewall = _firewall_audit()
    if firewall["status"] != "PASS":
        raise AmendmentProtocolError("target firewall failed")
    payload = {
        "schema": "joint-lsml-processbench-amendment-execution-registry-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "REGISTERED_POST_PRM_OPENED__PB_LABEL_FREE",
        "scope": plan["scope"],
        "candidate": COVERAGE_METHOD,
        "methods": list(COVERAGE_METHODS),
        "processbench_cells": list(parent.PB_CELLS),
        "expected_fallback_cells": [EXPECTED_BLOCKED],
        "parent_cell_statuses": {cell: cells[cell]["status"] for cell in parent.PB_CELLS},
        "analysis_plan_sha256": sha256_file(PLAN),
        "protocol_sha256": sha256_file(PROTOCOL),
        "prior_order_audit_sha256": sha256_file(PRIOR_ORDER_AUDIT),
        "source_hashes": _source_hashes(),
        "parent_hashes": _parent_hashes(),
        "parent_weight_hashes": _parent_weight_hashes(cells),
        "expected_blocked_status": "BLOCKED_NO_ADMISSIBLE_PARTITION",
        "expected_fallback_count": 1,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "processbench_labels_accessed": False,
        "prmbench_result_already_open": True,
        "firewall_audit": firewall,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    registry_hash = atomic_write_json(REGISTRY, payload)
    atomic_write_json(REGISTRATION_COMPLETE, {
        "status": "PASS",
        "registry_sha256": registry_hash,
        "processbench_labels_accessed": False,
    })


def _verify_registry() -> dict[str, Any]:
    if not REGISTRY.exists() or not REGISTRATION_COMPLETE.exists():
        raise AmendmentProtocolError("amendment is not registered")
    registry = _json_payload(REGISTRY)
    completion = json.loads(REGISTRATION_COMPLETE.read_text())
    if completion.get("registry_sha256") != sha256_file(REGISTRY):
        raise AmendmentProtocolError("registration completion mismatch")
    if registry.get("source_hashes") != _source_hashes():
        raise AmendmentProtocolError("registered source changed")
    if registry.get("parent_hashes") != _parent_hashes():
        raise AmendmentProtocolError("parent artifact changed")
    _, cells = _verify_parent()
    if registry.get("parent_weight_hashes") != _parent_weight_hashes(cells):
        raise AmendmentProtocolError("parent fitted weights changed")
    if registry.get("expected_blocked_status") != "BLOCKED_NO_ADMISSIBLE_PARTITION":
        raise AmendmentProtocolError("blocked-cell status contract changed")
    if registry.get("expected_fallback_count") != 1:
        raise AmendmentProtocolError("fallback-count contract changed")
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    if registry.get("runtime_versions") != runtime:
        raise AmendmentProtocolError("registered runtime changed")
    return registry


def _weights_from_parent(cell: Mapping[str, Any]) -> dict[str, np.ndarray]:
    weights = {name: np.asarray(value, dtype=np.float64) for name, value in cell["weights"].items()}
    if set(weights) != set(parent.METHODS):
        raise AmendmentProtocolError("parent weight roster drift")
    return weights


def score() -> None:
    registry = _verify_registry()
    if SCORE_ROOT.exists() or SCORE_MANIFEST.exists() or POLICY_LEDGER.exists():
        raise AmendmentProtocolError("amendment score namespace already exists")
    _, parent_cells = _verify_parent()
    sanitized = _json_payload(parent.SANITIZED_MANIFEST)
    by_cell = {row["cell_id"]: row for row in sanitized["cells"]}
    SCORE_ROOT.mkdir(parents=True, exist_ok=False)
    score_records = []
    policy_records = []
    for cell_id in parent.PB_CELLS:
        arrays = parent._load_sanitized(cell_id, by_cell)
        preparation = parent._preparation(arrays, json.loads(parent.REGISTRY.read_text()))
        if cell_id == EXPECTED_BLOCKED:
            fitted = fit_flat_fallback_and_controls(preparation)
            weights = fitted["weights"]
            component = fitted["candidate_component"]
            diagnostics = fitted["diagnostics"]
        else:
            weights = _weights_from_parent(parent_cells[cell_id])
            component = "frozen_parent_joint_lsml23_hierarchical_v1_1"
            diagnostics = {"parent_weights_reused": True}
        frozen, meta = parent._score_cell(cell_id, arrays, preparation, weights)
        renamed = rename_candidate_method(frozen)
        path = SCORE_ROOT / f"{cell_id}.npz"
        artifact_hash = atomic_write_npz(path, renamed)
        score_records.append({
            "cell_id": cell_id,
            "artifact_path": path.name,
            "artifact_sha256": artifact_hash,
            "members": sorted(renamed),
            "candidate_component": component,
            "n_rows": int(len(renamed["row_ids"])),
            **meta,
        })
        policy_records.append({
            "cell_id": cell_id,
            "candidate_component": component,
            "fallback": bool(cell_id == EXPECTED_BLOCKED),
            "preparation_payload_sha256": preparation.diagnostics["payload_sha256"],
            "weights": {
                (COVERAGE_METHOD if name == JOINT_METHOD else name): np.asarray(value).tolist()
                for name, value in weights.items()
            },
            "score_source_weight_keys": {COVERAGE_METHOD: JOINT_METHOD},
            "parent_weights_payload_sha256": (
                payload_sha256(parent_cells[cell_id]["weights"])
                if cell_id != EXPECTED_BLOCKED else None
            ),
            "diagnostics": diagnostics,
            "processbench_labels_accessed": False,
        })
        print(f"froze {cell_id}: {component}", flush=True)
    ledger = {
        "schema": "joint-lsml-processbench-coverage-policy-ledger-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "candidate": COVERAGE_METHOD,
        "cells": policy_records,
        "fallback_cells": [EXPECTED_BLOCKED],
        "processbench_labels_accessed": False,
    }
    ledger["payload_sha256"] = payload_sha256(ledger)
    atomic_write_json(POLICY_LEDGER, ledger)
    manifest = {
        "schema": "joint-lsml-processbench-amendment-score-freeze-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "SCORES_FROZEN_PENDING_INDEPENDENT_AUDIT",
        "scope": registry["scope"],
        "methods": list(COVERAGE_METHODS),
        "cells": score_records,
        "fallback_cells": [EXPECTED_BLOCKED],
        "registry_sha256": sha256_file(REGISTRY),
        "policy_ledger_sha256": sha256_file(POLICY_LEDGER),
        "parent_structural_ledger_sha256": sha256_file(parent.STRUCTURAL_LEDGER),
        "processbench_labels_accessed": False,
        "score_arrays_persisted": True,
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    manifest_hash = atomic_write_json(SCORE_MANIFEST, manifest)
    atomic_write_json(RUN_COMPLETE, {
        "status": "PASS_SCORE_FREEZE_PENDING_INDEPENDENT_AUDIT",
        "score_manifest_sha256": manifest_hash,
        "processbench_labels_accessed": False,
    })


def check() -> None:
    registry = _verify_registry()
    if not SCORE_MANIFEST.exists() or not POLICY_LEDGER.exists():
        print("PASS_REGISTERED")
        return
    manifest = _json_payload(SCORE_MANIFEST)
    ledger = _json_payload(POLICY_LEDGER)
    if manifest.get("registry_sha256") != sha256_file(REGISTRY):
        raise AmendmentProtocolError("score freeze registry mismatch")
    if manifest.get("policy_ledger_sha256") != sha256_file(POLICY_LEDGER):
        raise AmendmentProtocolError("score freeze policy ledger mismatch")
    if tuple(manifest.get("methods", ())) != COVERAGE_METHODS:
        raise AmendmentProtocolError("score freeze method roster mismatch")
    if [row["cell_id"] for row in manifest["cells"]] != list(parent.PB_CELLS):
        raise AmendmentProtocolError("score freeze is not exact all-eight ProcessBench")
    if manifest.get("fallback_cells") != [EXPECTED_BLOCKED] or ledger.get("fallback_cells") != [EXPECTED_BLOCKED]:
        raise AmendmentProtocolError("fallback roster mismatch")
    policy_by_cell = {row["cell_id"]: row for row in ledger["cells"]}
    sanitized = _json_payload(parent.SANITIZED_MANIFEST)
    sanitized_by_cell = {row["cell_id"]: row for row in sanitized["cells"]}
    parent_registry = json.loads(parent.REGISTRY.read_text())
    _, parent_cells = _verify_parent()
    for row in manifest["cells"]:
        cell_id = row["cell_id"]
        path = SCORE_ROOT / row["artifact_path"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise AmendmentProtocolError("score artifact changed")
        arrays = load_npz_no_pickle(path)
        if sorted(arrays) != sorted(row["members"]):
            raise AmendmentProtocolError("score member roster mismatch")
        if tuple(arrays["method_ids"].astype(str)) != COVERAGE_METHODS:
            raise AmendmentProtocolError("artifact method order mismatch")
        raw_arrays = parent._load_sanitized(cell_id, sanitized_by_cell)
        expected_rows = tuple(raw_arrays["row_ids"].astype(str))
        if tuple(arrays["row_ids"].astype(str)) != expected_rows or len(set(expected_rows)) != len(expected_rows):
            raise AmendmentProtocolError("score row IDs are incomplete, reordered, or duplicated")
        expected_shape = (len(expected_rows), len(COVERAGE_METHODS))
        if arrays["detector_scores"].shape != expected_shape or arrays["locators"].shape != expected_shape:
            raise AmendmentProtocolError("score matrix shape mismatch")
        if int(row.get("n_rows", -1)) != len(expected_rows):
            raise AmendmentProtocolError("manifest row count mismatch")
        if not np.isfinite(arrays["detector_scores"]).all():
            raise AmendmentProtocolError("non-finite detector score")
        step_counts = np.diff(raw_arrays["segment_offsets"])
        if np.any(arrays["locators"] < 0) or np.any(arrays["locators"] >= step_counts[:, None]):
            raise AmendmentProtocolError("invalid locator")
        policy = policy_by_cell[cell_id]
        if cell_id != EXPECTED_BLOCKED:
            expected_hash = payload_sha256(parent_cells[cell_id]["weights"])
            if policy.get("parent_weights_payload_sha256") != expected_hash:
                raise AmendmentProtocolError("parent weight hash reuse mismatch")
        elif policy.get("parent_weights_payload_sha256") is not None or not policy.get("fallback"):
            raise AmendmentProtocolError("blocked-cell fallback provenance mismatch")
        preparation = parent._preparation(raw_arrays, parent_registry)
        if policy.get("preparation_payload_sha256") != preparation.diagnostics["payload_sha256"]:
            raise AmendmentProtocolError("preparation replay hash mismatch")
        ledger_weights = {
            (JOINT_METHOD if name == COVERAGE_METHOD else name): np.asarray(value, dtype=np.float64)
            for name, value in policy["weights"].items()
        }
        replay, _ = parent._score_cell(cell_id, raw_arrays, preparation, ledger_weights)
        replay = rename_candidate_method(replay)
        for key in ("row_ids", "method_ids", "detector_scores", "locators"):
            if not np.array_equal(arrays[key], replay[key]):
                raise AmendmentProtocolError(f"score/reducer replay mismatch: {cell_id}/{key}")
    if registry.get("processbench_labels_accessed") is not False or manifest.get("processbench_labels_accessed") is not False:
        raise AmendmentProtocolError("pre-label artifact claims ProcessBench target access")
    print("PASS")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "register":
        register()
    elif command == "score":
        score()
    elif command == "check":
        check()
    else:
        raise SystemExit("usage: run_processbench_amendment_v1.py [register|score|check]")
