from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from spectral_utils.fair_comparisons.evaluator import paired_grouped_bootstrap


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1_r2.py"


def _module():
    spec = importlib.util.spec_from_file_location("joint_eval_r2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _statistic(payloads, _fit):
    labels = np.concatenate([np.asarray(row["labels"]) for row in payloads])
    output = {}
    module = _module(); r1 = module._r1_module(); base = r1._base_module()
    for method in ("joint", "control"):
        scores = np.concatenate([np.asarray(row["scores"][method]) for row in payloads])
        metrics = base.detection_metrics(labels, scores)
        output[f"auroc::{method}"] = metrics["auroc"]
        output[f"auprc::{method}"] = metrics["error_auprc"]
        output[f"normalized_ap::{method}"] = metrics["prevalence_normalized_ap"]
    output["delta_auroc_joint_vs::control"] = output["auroc::joint"] - output["auroc::control"]
    return output


def test_fast_bootstrap_matches_generic_with_strata_and_ties():
    module = _module()
    groups = {
        "a": {"labels": [0, 1, 0], "scores": {"joint": [0.1, 0.8, 0.1], "control": [0.2, 0.6, 0.2]}},
        "b": {"labels": [1, 0], "scores": {"joint": [0.7, 0.3], "control": [0.5, 0.4]}},
        "c": {"labels": [0, 1], "scores": {"joint": [0.2, 0.9], "control": [0.1, 0.7]}},
        "d": {"labels": [1, 0, 0], "scores": {"joint": [0.8, 0.2, 0.2], "control": [0.6, 0.3, 0.3]}},
    }
    strata = {"a": "x", "b": "x", "c": "y", "d": "y"}
    generic = paired_grouped_bootstrap(groups, _statistic, strata=strata, n_boot=17, seed=91)
    fast = module.fast_paired_prm_bootstrap(
        groups, strata, methods=("joint", "control"), joint_method="joint",
        controls=("control",), draws=17, seed=91, chunk=5,
    )
    for key in generic["statistics"]:
        for field in ("point", "ci_low", "ci_high"):
            assert abs(generic["statistics"][key][field] - fast["statistics"][key][field]) < 1e-12


def test_single_class_family_serializes_undefined_metrics_as_null():
    module = _module(); r1 = module._r1_module(); base = r1._base_module()
    payload = {"labels": [0, 0], "scores": {method: [0.1, 0.2] for method in base.METHODS}}
    result = module._finite_family_metrics(base, [payload])
    assert result["metric_status"] == "SINGLE_CLASS_NO_POSITIVE"
    assert result[f"auroc::{base.JOINT_METHOD}"] is None
    assert result[f"auprc::{base.JOINT_METHOD}"] is None

