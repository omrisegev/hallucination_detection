from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EVALUATOR = REPO / "scripts/joint_lsml_localization/evaluate_processbench_amendment_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("pb_amendment_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _paired_rows(module):
    rows = []
    for model in module.MODELS:
        for method in module.COVERAGE_METHODS:
            rows.append({
                "source_key": "gsm8k::q1",
                "source_group_id": "q1",
                "subset": "gsm8k",
                "first_error": 2,
                "stratify_label": 1,
                "fold": 0,
                "model_id": model,
                "method_id": method,
                "row_id": f"{model}::opaque",
            })
    return rows


def test_pairing_accepts_exact_two_model_four_method_payload_and_rejects_target_drift():
    module = _module()
    rows = _paired_rows(module)
    module._assert_pairing(rows)
    rows[-1]["first_error"] = 3
    try:
        module._assert_pairing(rows)
    except module.ProcessBenchEvaluationError:
        pass
    else:
        raise AssertionError("q4/q8 target drift was accepted")


def test_fitted7_diagnostic_excludes_only_fallback_and_does_not_refit():
    module = _module()
    per_cell = {}
    for method_position, method in enumerate(module.COVERAGE_METHODS):
        for cell_position, cell in enumerate(module.freeze.parent.PB_CELLS):
            per_cell[f"{method}::{cell}"] = float(100 * method_position + cell_position)
    result = module._fitted7_diagnostic(per_cell)
    fitted = [cell for cell in module.freeze.parent.PB_CELLS if cell != module.FALLBACK_CELL]
    for method in module.COVERAGE_METHODS:
        expected = sum(per_cell[f"{method}::{cell}"] for cell in fitted) / 7.0
        assert result[method] == expected


def test_decision_rule_is_frozen_for_harm_support_and_inconclusive():
    module = _module()
    keys = [f"delta_candidate_vs::{control}" for control in module.COVERAGE_METHODS[1:]]
    harm = {key: {"point": -0.1, "ci_low": -0.2, "ci_high": -0.01} for key in keys}
    assert module._decision_state(harm) == "HARM"
    support = {key: {"point": 0.1, "ci_low": 0.01, "ci_high": 0.2} for key in keys}
    assert module._decision_state(support) == "DEVELOPMENT_SUPPORTED"
    inconclusive = {key: {"point": 0.01, "ci_low": -0.01, "ci_high": 0.02} for key in keys}
    assert module._decision_state(inconclusive) == "INCONCLUSIVE"


def test_evaluator_is_processbench_only_and_has_no_prm_loader():
    tree = ast.parse(EVALUATOR.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not [name for name in imports if "localization_postfreeze" in name or "prmbench" in name.lower()]
    assert "prmbench_steps.npz" not in EVALUATOR.read_text()
