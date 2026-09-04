#!/usr/bin/env python3
"""Fast, finite-JSON R2 evaluator for the frozen Joint L-SML PRMBench scores."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, load_npz_no_pickle, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


RESULT_ROOT = REPO / "results/joint_lsml_existing_localization_v1"
R1_SCRIPT = REPO / "scripts/joint_lsml_localization/evaluate_existing_v1_r1.py"
SELF = Path(__file__).resolve()
TEST = REPO / "tests/test_joint_lsml_evaluation_amendment_r2.py"
PROTOCOL = REPO / "docs/experiments/JOINT_LSML_EXISTING_LOCALIZATION_EVALUATION_AMENDMENT_R2.md"
REGISTRY = RESULT_ROOT / "EVALUATION_AMENDMENT_R2_REGISTRY.json"
AUDIT = RESULT_ROOT / "INDEPENDENT_EVALUATION_AMENDMENT_R2_AUDIT.json"
EVALUATION_ROOT = RESULT_ROOT / "evaluation_r2"
SUMMARY = EVALUATION_ROOT / "EVALUATION_SUMMARY.json"
DEFAULT_RELEASE = Path(
    "/Users/osegev/Desktop/hallucination_detection/.worktrees/"
    "reconstruction-science-run-v1/results/reconstruction_benchmark_v1/"
    "releases/2026-08-24_localization_v1"
)


class EvaluationR2Error(RuntimeError):
    pass


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover
        raise EvaluationR2Error(f"cannot load {path.name}")
    spec.loader.exec_module(module)
    return module


def _r1_module():
    return _load(R1_SCRIPT, "joint_existing_evaluator_r1")


def _tie_structure(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    return order, starts


def fast_paired_prm_bootstrap(
    groups: Mapping[str, Mapping[str, Any]],
    strata: Mapping[str, str],
    *,
    methods: Sequence[str],
    joint_method: str,
    controls: Sequence[str],
    draws: int,
    seed: int,
    alpha: float = 0.05,
    chunk: int = 32,
) -> dict[str, Any]:
    group_ids = sorted(map(str, groups))
    normalized = {str(key): value for key, value in groups.items()}
    normalized_strata = {str(key): str(value) for key, value in strata.items()}
    if set(group_ids) != set(normalized_strata):
        raise EvaluationR2Error("bootstrap stratum roster mismatch")
    ids_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        ids_by_stratum[normalized_strata[group_id]].append(group_id)
    for key in ids_by_stratum:
        ids_by_stratum[key].sort()
    group_position = {group_id: index for index, group_id in enumerate(group_ids)}
    positions_by_stratum = {
        key: np.asarray([group_position[group_id] for group_id in values], dtype=np.int64)
        for key, values in sorted(ids_by_stratum.items())
    }

    labels_parts = []
    score_parts = {method: [] for method in methods}
    step_groups_parts = []
    for group_index, group_id in enumerate(group_ids):
        payload = normalized[group_id]
        labels = np.asarray(payload["labels"], dtype=np.int8)
        labels_parts.append(labels)
        step_groups_parts.append(np.full(len(labels), group_index, dtype=np.int64))
        for method in methods:
            values = np.asarray(payload["scores"][method], dtype=np.float64)
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise EvaluationR2Error("score/label shape or finiteness mismatch")
            score_parts[method].append(values)
    labels = np.concatenate(labels_parts)
    step_groups = np.concatenate(step_groups_parts)
    scores = {method: np.concatenate(parts) for method, parts in score_parts.items()}
    structures = {method: _tie_structure(values) for method, values in scores.items()}

    def one_metrics(values: np.ndarray) -> tuple[float, float, float]:
        order, starts = _tie_structure(values)
        y = labels[order].astype(np.float64)
        positives = np.add.reduceat(y, starts)
        negatives = np.add.reduceat(1.0 - y, starts)
        cp = np.cumsum(positives); cn = np.cumsum(negatives)
        tp = cp[-1]; tn = cn[-1]
        auc = np.sum(positives * (tn - cn + 0.5 * negatives)) / (tp * tn)
        ap = np.sum(positives * (cp / (cp + cn))) / tp
        prevalence = tp / (tp + tn)
        return float(auc), float(ap), float((ap - prevalence) / (1.0 - prevalence))

    point: dict[str, float] = {}
    for method in methods:
        auc, ap, nap = one_metrics(scores[method])
        point[f"auroc::{method}"] = auc
        point[f"auprc::{method}"] = ap
        point[f"normalized_ap::{method}"] = nap
    for control in controls:
        point[f"delta_auroc_joint_vs::{control}"] = point[f"auroc::{joint_method}"] - point[f"auroc::{control}"]

    samples = {key: np.empty(draws, dtype=np.float64) for key in point}
    rng = np.random.default_rng(int(seed))
    for offset in range(0, int(draws), int(chunk)):
        size = min(int(chunk), int(draws) - offset)
        counts = np.zeros((size, len(group_ids)), dtype=np.int64)
        # Preserve paired_grouped_bootstrap's draw-major, sorted-stratum RNG order.
        for row in range(size):
            for stratum in sorted(positions_by_stratum):
                positions = positions_by_stratum[stratum]
                picks = rng.integers(0, len(positions), size=len(positions))
                counts[row, positions] = np.bincount(picks, minlength=len(positions))
        for method in methods:
            order, starts = structures[method]
            weights = counts[:, step_groups[order]].astype(np.float64, copy=False)
            ordered_y = labels[order].astype(np.float64, copy=False)
            positives = np.add.reduceat(weights * ordered_y, starts, axis=1)
            negatives = np.add.reduceat(weights * (1.0 - ordered_y), starts, axis=1)
            cp = np.cumsum(positives, axis=1); cn = np.cumsum(negatives, axis=1)
            tp = cp[:, -1]; tn = cn[:, -1]
            auc = np.sum(positives * (tn[:, None] - cn + 0.5 * negatives), axis=1) / (tp * tn)
            precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0)
            ap = np.sum(positives * precision, axis=1) / tp
            prevalence = tp / (tp + tn)
            nap = (ap - prevalence) / (1.0 - prevalence)
            samples[f"auroc::{method}"][offset:offset + size] = auc
            samples[f"auprc::{method}"][offset:offset + size] = ap
            samples[f"normalized_ap::{method}"][offset:offset + size] = nap
        for control in controls:
            samples[f"delta_auroc_joint_vs::{control}"][offset:offset + size] = (
                samples[f"auroc::{joint_method}"][offset:offset + size]
                - samples[f"auroc::{control}"][offset:offset + size]
            )
    statistics = {}
    for key, values in samples.items():
        low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        statistics[key] = {
            "point": point[key], "ci_low": float(low), "ci_high": float(high),
            "n_valid": int(np.isfinite(values).sum()),
        }
    return {
        "n_groups": len(group_ids),
        "n_groups_by_stratum": {key: len(values) for key, values in sorted(ids_by_stratum.items())},
        "n_boot": int(draws), "seed": int(seed), "alpha": float(alpha),
        "resampling_unit": "source_question_within_family",
        "paired_payload": "all_methods_budgets_arms_and_copies",
        "recomputed_each_replicate": False,
        "implementation": "tie_block_sufficient_statistics_exact_rng_order_v1",
        "statistics": statistics,
    }


def _finite_family_metrics(base: Any, payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    raw = base._prm_statistic(payloads, None)
    output = {key: (float(value) if np.isfinite(value) else None) for key, value in raw.items()}
    if any(value is None for value in output.values()):
        output["metric_status"] = "SINGLE_CLASS_NO_POSITIVE"
    else:
        output["metric_status"] = "OK"
    return output


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/joint_lsml_localization/evaluate_existing_v1.py",
        "scripts/joint_lsml_localization/evaluate_existing_v1_r1.py",
        "scripts/joint_lsml_localization/evaluate_existing_v1_r2.py",
        "tests/test_joint_lsml_evaluation_amendment_r2.py",
        "docs/experiments/JOINT_LSML_EXISTING_LOCALIZATION_EVALUATION_AMENDMENT_R2.md",
        "scripts/reasoning_localization/run_h3_prmbench_diagnostic.py",
        "spectral_utils/fair_comparisons/evaluator.py",
    )
    return {path: sha256_file(REPO / path) for path in paths}


def _prior_hashes() -> dict[str, str]:
    names = (
        "EXECUTION_REGISTRY.json", "STRUCTURAL_LEDGER.json", "SCORE_FREEZE_MANIFEST.json",
        "INDEPENDENT_SCORE_FREEZE_AUDIT.json", "EVALUATION_AMENDMENT_R1_REGISTRY.json",
        "INDEPENDENT_EVALUATION_AMENDMENT_R1_AUDIT.json",
    )
    return {name: sha256_file(RESULT_ROOT / name) for name in names}


def _real_groups(release: Path):
    r1 = _r1_module(); base = r1._base_module()
    manifest, by_cell = base._verified_score_manifest()
    if manifest["processbench_panel_status"] != "STRUCTURAL_NO_SCORE":
        raise EvaluationR2Error("R2 is PRMBench-only")
    groups, strata = r1._prm_groups_r1(base, by_cell, release)
    return r1, base, groups, strata


def register(release: Path) -> None:
    if REGISTRY.exists() or EVALUATION_ROOT.exists():
        raise EvaluationR2Error("R2 registry/namespace already exists")
    r1, base, groups, strata = _real_groups(release)
    plan = json.loads(base.PLAN.read_text())
    probe_draws = 3
    generic = base.paired_grouped_bootstrap(
        groups, base._prm_statistic, strata=strata, n_boot=probe_draws,
        seed=int(plan["prmbench"]["bootstrap_seed"]), alpha=0.05,
    )
    fast = fast_paired_prm_bootstrap(
        groups, strata, methods=base.METHODS, joint_method=base.JOINT_METHOD,
        controls=(base.IU_METHOD, base.EQUAL_FAMILY_METHOD, base.FIXED_FAMILY_METHOD),
        draws=probe_draws, seed=int(plan["prmbench"]["bootstrap_seed"]), alpha=0.05,
    )
    errors = []
    for key in generic["statistics"]:
        for field in ("point", "ci_low", "ci_high"):
            errors.append(abs(float(generic["statistics"][key][field]) - float(fast["statistics"][key][field])))
    max_error = max(errors)
    if max_error > 1e-12:
        raise EvaluationR2Error(f"fast/generic probe mismatch: {max_error}")
    payload = {
        "schema": "joint-lsml-existing-evaluation-amendment-registry-r2",
        "status": "REGISTERED_AFTER_R1_POSTCOMPUTE_SERIALIZATION_FAILURE",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT_PRMBENCH_ONLY",
        "r1_failure": {
            "exception": "ValueError: Out of range float values are not JSON compliant",
            "cause": "multi_solutions has no positive step; family metrics are undefined",
            "evaluation_artifact_created": False,
            "registered_draws_completed_in_memory": 2000,
        },
        "method_scores_refit": False, "score_freeze_changed": False,
        "processbench_labels_allowed": False, "processbench_status": "STRUCTURAL_NO_SCORE",
        "bootstrap_contract": {
            "draws": int(plan["prmbench"]["bootstrap_draws"]),
            "seed": int(plan["prmbench"]["bootstrap_seed"]),
            "strata": "error_family", "paired": True,
            "generic_equivalence_probe_draws": probe_draws,
            "generic_equivalence_max_abs_error": max_error,
        },
        "single_class_serialization": "null_with_SINGLE_CLASS_NO_POSITIVE",
        "source_hashes": _source_hashes(), "prior_artifact_hashes": _prior_hashes(),
        "prmbench_label_sha256": r1.LABEL_SHA256,
        "independent_amendment_audit_required": True,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    atomic_write_json(REGISTRY, payload)


def _verified_registry(release: Path) -> dict[str, Any]:
    if not REGISTRY.exists() or not AUDIT.exists():
        raise EvaluationR2Error("registered and audited R2 amendment is required")
    registry = json.loads(REGISTRY.read_text())
    body = {key: value for key, value in registry.items() if key != "payload_sha256"}
    if payload_sha256(body) != registry.get("payload_sha256"):
        raise EvaluationR2Error("R2 registry payload mismatch")
    for path, expected in registry["source_hashes"].items():
        if sha256_file(REPO / path) != expected:
            raise EvaluationR2Error(f"R2 source changed: {path}")
    for name, expected in registry["prior_artifact_hashes"].items():
        if sha256_file(RESULT_ROOT / name) != expected:
            raise EvaluationR2Error(f"R2 prior artifact changed: {name}")
    label_path = release / "build_A/localization/evaluation/prmbench_steps.npz"
    if sha256_file(label_path) != registry["prmbench_label_sha256"]:
        raise EvaluationR2Error("R2 label source changed")
    audit = json.loads(AUDIT.read_text())
    audit_body = {key: value for key, value in audit.items() if key != "payload_sha256"}
    if (
        audit.get("status") != "PASS"
        or audit.get("amendment_registry_sha256") != sha256_file(REGISTRY)
        or payload_sha256(audit_body) != audit.get("payload_sha256")
    ):
        raise EvaluationR2Error("R2 independent audit absent or stale")
    return registry


def evaluate(release: Path) -> None:
    registry = _verified_registry(release)
    r1, base, groups, strata = _real_groups(release)
    plan = json.loads(base.PLAN.read_text())
    bootstrap = fast_paired_prm_bootstrap(
        groups, strata, methods=base.METHODS, joint_method=base.JOINT_METHOD,
        controls=(base.IU_METHOD, base.EQUAL_FAMILY_METHOD, base.FIXED_FAMILY_METHOD),
        draws=int(plan["prmbench"]["bootstrap_draws"]),
        seed=int(plan["prmbench"]["bootstrap_seed"]), alpha=0.05,
    )
    point = {key: value["point"] for key, value in bootstrap["statistics"].items()}
    per_family = {
        family: _finite_family_metrics(base, [groups[key] for key, value in strata.items() if value == family])
        for family in sorted(set(strata.values()))
    }
    prm = {
        "status": "COMPLETE", "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "point_metrics": point, "per_family": per_family, "paired_bootstrap": bootstrap,
        "n_source_groups": len(groups),
        "n_steps": sum(len(payload["labels"]) for payload in groups.values()),
    }
    prm["decision_state"] = base._panel_state(prm, panel="PRMBench")
    label_path = release / "build_A/localization/evaluation/prmbench_steps.npz"
    output = {
        "schema": "joint-lsml-existing-localization-evaluation-r2",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "promotion_allowed": False, "generalization_claim_allowed": False,
        "ProcessBench": {"status": "STRUCTURAL_NO_SCORE", "decision_state": "STRUCTURAL_NO_SCORE"},
        "PRMBench": prm,
        "score_manifest_sha256": sha256_file(base.SCORE_MANIFEST),
        "independent_score_audit_sha256": sha256_file(base.AUDIT),
        "evaluation_amendment_registry_sha256": sha256_file(REGISTRY),
        "evaluation_amendment_audit_sha256": sha256_file(AUDIT),
        "label_sources": {"PRMBench": {"path": str(label_path), "sha256": sha256_file(label_path)}},
        "fresh_generalization_recommended": False,
        "labels_accessed_only_after_score_freeze_audit": True,
        "r1_evaluation_artifact_created": False,
    }
    output["payload_sha256"] = payload_sha256(output)
    atomic_write_json(SUMMARY, output)


def check(release: Path) -> None:
    _verified_registry(release)
    if SUMMARY.exists():
        output = json.loads(SUMMARY.read_text())
        body = {key: value for key, value in output.items() if key != "payload_sha256"}
        if payload_sha256(body) != output.get("payload_sha256"):
            raise EvaluationR2Error("R2 summary payload mismatch")
        if output["ProcessBench"]["status"] != "STRUCTURAL_NO_SCORE":
            raise EvaluationR2Error("R2 unexpectedly evaluated ProcessBench")
        if output["PRMBench"]["status"] != "COMPLETE":
            raise EvaluationR2Error("R2 PRMBench incomplete")
    print("PASS")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    release = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_RELEASE
    if command == "register": register(release)
    elif command == "evaluate": evaluate(release)
    elif command == "check": check(release)
    else: raise SystemExit("usage: evaluate_existing_v1_r2.py [register|evaluate|check] [release]")

