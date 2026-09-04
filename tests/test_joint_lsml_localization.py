from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from spectral_utils.fixed_application_pipelines import SHARED_GLOBAL_FEATURES, SHARED_TOKEN_VIEWS
from spectral_utils.joint_lsml import discover_loao_consensus_groups
from spectral_utils.joint_lsml_localization import METHODS, prepare_active23
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, atomic_write_npz, sha256_file
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256
from spectral_utils.specrage_views import FEATURE_TO_VIEW


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/joint_lsml_localization/run_existing_v1.py"
RETAINED = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28]
SIGNS = [-1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1]


def _runner_module():
    spec = importlib.util.spec_from_file_location("joint_existing_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evaluator_module():
    path = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1.py"
    spec = importlib.util.spec_from_file_location("joint_existing_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active23_preparation_uses_absolute_sign_before_zscore():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(120, 29))
    offsets = np.asarray([0, 40, 80, 120])
    preparation = prepare_active23(
        raw, offsets, ["a", "b", "c"], retained_indices=RETAINED,
        confidence_signs_29=SIGNS, stream_names_29=SHARED_TOKEN_VIEWS,
        raw_feature_names_29=SHARED_GLOBAL_FEATURES, fit_token_cap=120,
    )
    expected = raw[:, RETAINED] * np.asarray(SIGNS)[RETAINED][None, :]
    expected = (expected - expected.mean(axis=0)) / expected.std(axis=0)
    np.testing.assert_allclose(preparation.standardized_fit, expected, atol=1e-12, rtol=1e-12)
    assert preparation.feature_names == tuple(SHARED_TOKEN_VIEWS[index] for index in RETAINED)
    assert sum(preparation.diagnostics["family_counts"].values()) == 23
    assert len(preparation.diagnostics["family_counts"]) == 5


def test_active23_risk_is_negative_confidence():
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(90, 29))
    preparation = prepare_active23(
        raw, [0, 30, 60, 90], ["a", "b", "c"], retained_indices=RETAINED,
        confidence_signs_29=SIGNS, stream_names_29=SHARED_TOKEN_VIEWS,
        raw_feature_names_29=SHARED_GLOBAL_FEATURES, fit_token_cap=90,
    )
    weights = np.arange(1, 24, dtype=float)
    weights /= np.linalg.norm(weights)
    np.testing.assert_allclose(
        preparation.token_risk(weights), -(preparation.standardized_fit @ weights)
    )


def test_fixed_top10_is_not_top5_or_top_ten_percent():
    module = _runner_module()
    risk = np.arange(1.0, 21.0)
    observed = module._top10_step_scores(risk, np.asarray([0]), np.asarray([20]))
    assert observed[0] == np.mean(np.arange(11.0, 21.0))
    assert observed[0] != np.mean(np.arange(16.0, 21.0))


def test_runner_imports_no_label_evaluator_or_outcome_loader():
    tree = ast.parse(RUNNER.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("evaluator" in name for name in imported)
    assert not any("phase1_baseline" in name for name in imported)
    source = RUNNER.read_text()
    for forbidden in ("true_first_error", "step_labels", "error_steps", "final_answer_correct"):
        assert forbidden not in source


def test_retained_family_mapping_is_repository_derived():
    families = [FEATURE_TO_VIEW[SHARED_GLOBAL_FEATURES[index]] for index in RETAINED]
    assert len(set(families)) == 5
    assert len(families) == 23


def test_pairwise_cap_does_not_change_group_selection():
    rng = np.random.default_rng(101)
    owners = np.repeat(np.arange(12), 20)
    latent = rng.normal(size=(len(owners), 3))
    values = np.column_stack([
        latent[:, group] + 0.2 * rng.normal(size=len(owners))
        for group in range(3) for _ in range(3)
    ])
    exact = discover_loao_consensus_groups(
        values, owners, k_range=(3,), seed=33, pairwise_diagnostic_cap=10_000
    )
    capped = discover_loao_consensus_groups(
        values, owners, k_range=(3,), seed=33, pairwise_diagnostic_cap=10
    )
    assert exact["status"] == capped["status"]
    if exact["status"] == "SELECTED":
        assert exact["K"] == capped["K"]
        np.testing.assert_array_equal(exact["labels"], capped["labels"])
        assert exact["median_ari"] == capped["median_ari"]
        assert capped["candidates"][0]["pairwise_ari_sampling"] == "deterministic_uniform_pair_sample"


def test_panel_gate_is_all_eight_pb_and_separate_prm():
    module = _runner_module()
    records = {
        cell: {"status": "FIT_COMPLETE", "structural_fit_pass": True}
        for cell in (*module.PB_CELLS, module.PRM_CELL)
    }
    assert module._panel_gate_status(records) == (True, True)
    records[module.PB_CELLS[3]] = {"status": "BLOCKED_NO_ADMISSIBLE_PARTITION", "structural_fit_pass": False}
    assert module._panel_gate_status(records) == (False, True)
    records[module.PRM_CELL] = {"status": "BLOCKED_STRUCTURAL_FIT", "structural_fit_pass": False}
    assert module._panel_gate_status(records) == (False, False)


def test_pb_and_prm_reducers_cover_full_frozen_interfaces():
    module = _runner_module()

    class Prep:
        row_ids = ("a", "b")

    curves = {method: np.asarray([1., 2., 3., 4., 20., 19., 18., 17.]) + index for index, method in enumerate(METHODS)}
    pb = {
        "token_offsets": np.asarray([0, 4, 8]),
        "segment_offsets": np.asarray([0, 2, 4]),
        "segment_starts": np.asarray([0, 2, 4, 6]),
        "segment_ends": np.asarray([2, 4, 6, 8]),
    }
    with patch.object(module, "score_active23_arms", return_value=curves):
        frozen, _ = module._score_cell(module.PB_CELLS[0], pb, Prep(), {method: np.ones(23) for method in METHODS})
    np.testing.assert_allclose(frozen["detector_scores"][:, 0], [4., 20.])
    np.testing.assert_array_equal(frozen["locators"][:, 0], [1, 0])
    with patch.object(module, "score_active23_arms", return_value=curves):
        frozen, _ = module._score_cell(module.PRM_CELL, pb, Prep(), {method: np.ones(23) for method in METHODS})
    np.testing.assert_allclose(frozen["step_risk"][:, 0], [2., 4., 20., 18.])


def test_sanitized_loader_rejects_extra_member():
    module = _runner_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "cell.npz"
        arrays = {
            "raw": np.zeros((2, 29)), "token_offsets": np.asarray([0, 2]),
            "row_ids": np.asarray(["a"]), "segment_offsets": np.asarray([0, 1]),
            "segment_starts": np.asarray([0]), "segment_ends": np.asarray([2]),
            "forbidden": np.asarray([1]),
        }
        atomic_write_npz(path, arrays)
        record = {"artifact_path": path.name, "artifact_sha256": sha256_file(path)}
        with patch.object(module, "SANITIZED_ROOT", root):
            try:
                module._load_sanitized("cell", {"cell": record})
            except module.ProtocolError:
                pass
            else:
                raise AssertionError("unsafe sanitized member was accepted")


def test_contract_validation_rejects_semantic_tamper_even_with_rehashed_payload():
    module = _runner_module()
    orientation = json.loads(module.ORIENTATION.read_text())
    roster = json.loads(module.ROSTER.read_text())
    orientation["output_semantics"] = "HIGHER_IS_MORE_RISK"
    orientation["payload_sha256"] = payload_sha256({k: v for k, v in orientation.items() if k != "payload_sha256"})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        opath, rpath = root / "orientation.json", root / "roster.json"
        opath.write_text(json.dumps(orientation))
        roster["source_artifact_sha256"] = sha256_file(opath)
        roster["payload_sha256"] = payload_sha256({k: v for k, v in roster.items() if k != "payload_sha256"})
        rpath.write_text(json.dumps(roster))
        with patch.object(module, "ORIENTATION", opath), patch.object(module, "ROSTER", rpath):
            try:
                module._contracts()
            except module.ProtocolError:
                pass
            else:
                raise AssertionError("tampered orientation semantics were accepted")


def test_pb_pairing_requires_both_models_all_methods_and_shared_target():
    module = _evaluator_module()
    rows = []
    for model_index, model in enumerate(module.MODELS):
        for method in module.METHODS:
            rows.append({
                "source_key": "gsm8k::q1", "source_group_id": "q1", "subset": "gsm8k",
                "model_id": model, "method_id": method, "row_id": f"{model}-row",
                "first_error": 2, "stratify_label": 1, "fold": 3,
            })
    module._assert_pb_pairing(rows)
    try:
        module._assert_pb_pairing(rows[:-1])
    except module.EvaluationError:
        pass
    else:
        raise AssertionError("incomplete model/method payload was accepted")
    tampered = [dict(row) for row in rows]
    tampered[-1]["first_error"] = -1
    tampered[-1]["stratify_label"] = 0
    try:
        module._assert_pb_pairing(tampered)
    except module.EvaluationError:
        pass
    else:
        raise AssertionError("cross-model ProcessBench target drift was accepted")


def test_evaluator_freeze_chain_rejects_plan_tamper():
    module = _evaluator_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        score_root = root / "score_freeze"
        score_root.mkdir()
        plan_path = root / "plan.json"
        plan_path.write_text("{}\n")
        score_path = score_root / "cell.npz"
        atomic_write_npz(score_path, {
            "row_ids": np.asarray(["a"]), "method_ids": np.asarray(module.METHODS),
            "detector_scores": np.zeros((1, len(module.METHODS))),
            "locators": np.zeros((1, len(module.METHODS)), dtype=np.int64),
        })
        structural_path = root / "STRUCTURAL_LEDGER.json"
        structural = {"status": "COMPLETE", "labels_accessed": False}
        structural["payload_sha256"] = payload_sha256(structural)
        atomic_write_json(structural_path, structural)
        registry_path = root / "EXECUTION_REGISTRY.json"
        registry = {"source_hashes": {}, "analysis_plan_sha256": sha256_file(plan_path)}
        registry["payload_sha256"] = payload_sha256(registry)
        atomic_write_json(registry_path, registry)
        manifest_path = root / "SCORE_FREEZE_MANIFEST.json"
        manifest = {
            "registry_sha256": sha256_file(registry_path),
            "structural_ledger_sha256": sha256_file(structural_path),
            "labels_accessed": False,
            "cells": [{"cell_id": "cell", "artifact_path": score_path.name, "artifact_sha256": sha256_file(score_path)}],
        }
        manifest["payload_sha256"] = payload_sha256(manifest)
        atomic_write_json(manifest_path, manifest)
        audit_path = root / "INDEPENDENT_SCORE_FREEZE_AUDIT.json"
        audit = {"status": "PASS", "labels_accessed": False, "score_manifest_sha256": sha256_file(manifest_path)}
        audit["payload_sha256"] = payload_sha256(audit)
        atomic_write_json(audit_path, audit)
        with (
            patch.object(module, "RESULT_ROOT", root),
            patch.object(module, "REGISTRY", registry_path),
            patch.object(module, "STRUCTURAL_LEDGER", structural_path),
            patch.object(module, "SCORE_ROOT", score_root),
            patch.object(module, "SCORE_MANIFEST", manifest_path),
            patch.object(module, "AUDIT", audit_path),
            patch.object(module, "PLAN", plan_path),
        ):
            module._verified_score_manifest()
            plan_path.write_text('{"tampered":true}\n')
            try:
                module._verified_score_manifest()
            except module.EvaluationError:
                pass
            else:
                raise AssertionError("evaluation opened after plan tamper")
