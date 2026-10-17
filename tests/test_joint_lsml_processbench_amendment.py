from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from spectral_utils.joint_lsml import dispatch_alias
from spectral_utils.joint_lsml_localization import (
    EQUAL_FAMILY_METHOD,
    FIXED_FAMILY_METHOD,
    IU_METHOD,
    JOINT_METHOD,
    Active23Preparation,
)
from spectral_utils.joint_lsml_processbench_amendment import (
    COVERAGE_METHOD,
    COVERAGE_METHODS,
    fit_flat_fallback_and_controls,
    rename_candidate_method,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/joint_lsml_localization/run_processbench_amendment_v1.py"


def _preparation() -> Active23Preparation:
    rng = np.random.default_rng(2026090408)
    n, p = 800, 23
    common = rng.normal(size=(n, 1))
    values = 0.7 * common + 0.3 * rng.normal(size=(n, p))
    values -= values.mean(axis=0, keepdims=True)
    values /= values.std(axis=0, keepdims=True)
    families = (
        ("entropy_level",) * 5
        + ("entropy_dynamics",) * 5
        + ("sampled_token_energy",) * 5
        + ("partition_energy",) * 4
        + ("topk_distribution",) * 4
    )
    offsets = np.asarray([0, n], dtype=np.int64)
    return Active23Preparation(
        raw=values,
        token_offsets=offsets,
        row_ids=("synthetic",),
        retained_indices=np.arange(p, dtype=np.int64),
        signs=np.ones(p, dtype=np.int64),
        feature_names=tuple(f"f{index}" for index in range(p)),
        family_names=families,
        fit_indices=np.arange(n, dtype=np.int64),
        fit_row_indices=np.zeros(n, dtype=np.int64),
        medians=np.zeros(p),
        mean=np.zeros(p),
        std=np.ones(p),
        standardized_fit=values,
        diagnostics=MappingProxyType({"payload_sha256": "synthetic"}),
    )


def test_blocked_cell_policy_is_flat_alias_up_to_global_sign_and_all_controls_are_finite():
    preparation = _preparation()
    fitted = fit_flat_fallback_and_controls(preparation)
    _, alias_weight, meta = dispatch_alias(
        preparation.standardized_fit,
        np.zeros(23, dtype=np.int64),
        mode="flat_sml",
    )
    candidate = fitted["weights"][JOINT_METHOD]
    assert meta["bit_exact_alias"] is True
    assert np.array_equal(candidate, alias_weight) or np.array_equal(candidate, -alias_weight)
    assert set(fitted["weights"]) == {
        JOINT_METHOD, IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD,
    }
    assert all(np.isfinite(weight).all() and weight.shape == (23,) for weight in fitted["weights"].values())


def test_candidate_column_is_renamed_without_changing_scores():
    original = {
        "row_ids": np.asarray(["r"]),
        "method_ids": np.asarray([JOINT_METHOD, IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD]),
        "detector_scores": np.arange(4, dtype=float)[None, :],
        "locators": np.arange(4, dtype=np.int64)[None, :],
    }
    renamed = rename_candidate_method(original)
    assert tuple(renamed["method_ids"].astype(str)) == COVERAGE_METHODS
    assert renamed["method_ids"][0] == COVERAGE_METHOD
    assert np.array_equal(renamed["detector_scores"], original["detector_scores"])
    assert np.array_equal(renamed["locators"], original["locators"])


def test_runner_has_no_processbench_label_or_metric_import():
    tree = ast.parse(RUNNER.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("localization_postfreeze", "roc_auc", "average_precision", "evaluate_existing")
    assert not [name for name in imports if any(token in name for token in forbidden)]


def test_plan_discloses_second_opened_policy_and_single_fallback():
    plan = json.loads((REPO / "configs/joint_lsml_processbench_amendment_v1.json").read_text())
    assert plan["candidate"] == COVERAGE_METHOD
    assert plan["candidate_policy"]["expected_fallback_cells"] == ["processbench_math_qwen3_4b"]
    assert plan["reporting"]["cumulative_opened_exposure"]["candidate_policies_seen"] == 2
    assert plan["reporting"]["cumulative_opened_exposure"]["secondary_diagnostics"] == 1


def test_registration_contract_binds_exact_parent_weight_hashes():
    spec = importlib.util.spec_from_file_location("pb_amendment_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, cells = module._verify_parent()
    hashes = module._parent_weight_hashes(cells)
    assert len(hashes) == 7
    assert "processbench_math_qwen3_4b" not in hashes
