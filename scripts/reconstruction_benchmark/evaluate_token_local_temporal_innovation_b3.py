#!/usr/bin/env python3
"""Post-audit evaluation for the Phase-2 temporal-innovation score freeze.

The score runner and this evaluator are intentionally separate processes.  The
preflight path only verifies frozen score artifacts and certificates; it does
not import the target-bearing evaluator or open label files.  Target import is
the first statement after all freeze/audit/source/environment/row-join gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    empirical_midrank,
)
from spectral_utils.token_temporal_innovation_b3 import (  # noqa: E402
    LOCAL_TOKEN_B3,
    LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL,
    LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
    LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    LOCAL_TOKEN_B3_SELF_INNOV,
    METHOD_IDS,
)


PB_CELLS = (
    "processbench_gsm8k_qwen3_4b",
    "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b",
    "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b",
    "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b",
    "processbench_omnimath_qwen3_8b",
)
PRM_CELL = "prmbench_response_qwen3_8b"
CELLS = PB_CELLS + (PRM_CELL,)
REFERENCE = "LOCAL_IU29"
EQUAL_REFERENCE = "LOCAL_EQUAL29"
PHASE2_SCHEMA = "token-local-temporal-innovation-b3-score-freeze-v1"
PHASE2_AUDIT_SCHEMA = "token-local-temporal-innovation-b3-prelabel-audit-v1"
PHASE1_SCHEMA = "token-local-fusion-phase1-score-freeze-v1"
PHASE1_AUDIT_SCHEMA = "token-local-fusion-phase1-prelabel-audit-v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082803
TIE_TOLERANCE = 0.0005
METRIC_IDS = (
    "official_macro_f1", "first_error_exact", "first_error_within_one",
    "clean_abstention_accuracy", "overall_decision_accuracy",
)


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _verified_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop("payload_sha256", None)
    if not isinstance(digest, str) or digest != _payload_sha(body):
        raise RuntimeError(f"JSON payload hash failed: {path}")
    return value


def _verified_environment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("score freeze lacks an environment snapshot")
    body = dict(value)
    digest = body.pop("environment_sha256", None)
    if not isinstance(digest, str) or digest != _payload_sha(body):
        raise RuntimeError("score-freeze environment hash failed")
    return value


def _source_snapshot_ok(freeze: Mapping[str, Any]) -> None:
    for source in freeze.get("source_snapshot", ()):
        path = ROOT / str(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"source changed after score freeze: {source['path']}")


def _phase2_preflight(score_root: Path, audit_path: Path) -> dict[str, dict[str, Any]]:
    freeze_path = score_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _verified_json(freeze_path)
    if not (
        freeze.get("schema_version") == PHASE2_SCHEMA
        and freeze.get("protocol_id") == "TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1"
        and freeze.get("all_expected_scores_present") is True
        and freeze.get("labels_seen_during_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and freeze.get("response_scores_materialized") is False
        and tuple(freeze.get("expected_cells", ())) == CELLS
        and tuple(freeze.get("method_ids", ())) == METHOD_IDS
    ):
        raise RuntimeError("Phase-2 target-free score-freeze contract failed")
    environment = _verified_environment(freeze.get("environment"))
    if freeze.get("environment_sha256") != environment["environment_sha256"]:
        raise RuntimeError("Phase-2 environment binding failed")
    _source_snapshot_ok(freeze)

    audit = _verified_json(audit_path)
    if not (
        audit.get("schema_version") == PHASE2_AUDIT_SCHEMA
        and audit.get("status") == "PASS"
        and audit.get("labels_opened_during_audit") is False
        and audit.get("score_freeze_sha256") == sha256_file(freeze_path)
        and audit.get("score_freeze_payload_sha256") == freeze["payload_sha256"]
        and audit.get("evaluation_source_sha256") == sha256_file(Path(__file__))
        and audit.get("protocol_sha256")
        == sha256_file(ROOT / "docs/experiments/TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1.md")
        and isinstance(audit.get("independent_agent_id"), str)
        and audit.get("independent_agent_id")
    ):
        raise RuntimeError("independent Phase-2 pre-label audit certificate failed")

    output: dict[str, dict[str, Any]] = {}
    for binding in freeze["records"]:
        cell_id = str(binding["cell_id"])
        record_path = score_root / str(binding["record_path"])
        if sha256_file(record_path) != binding["record_sha256"]:
            raise RuntimeError(f"{cell_id}: frozen record changed")
        record = _verified_json(record_path)
        record_environment = _verified_environment(record.get("environment"))
        score_path = record_path.parent / str(record["score_path"])
        if (
            sha256_file(score_path) != record["score_sha256"]
            or record["score_sha256"] != binding["score_sha256"]
            or record_environment != environment
            or record.get("environment_sha256") != environment["environment_sha256"]
            or tuple(record.get("method_ids", ())) != METHOD_IDS
            or record.get("response_scores_materialized") is not False
            or record.get("labels_seen_during_fit") is not False
            or record.get("targets_accessed_during_fit") is not False
        ):
            raise RuntimeError(f"{cell_id}: Phase-2 record schema/health failed")
        arrays = load_npz_no_pickle(score_path)
        expected = {"row_ids", "segment_offsets", "segment_starts", "segment_ends", "method_ids", "token_scores", "token_step_scores"}
        method_array = arrays.get("method_ids")
        method_ids = tuple(map(str, method_array.tolist())) if method_array is not None else ()
        scores = np.asarray(arrays["token_scores"], dtype=np.float64) if "token_scores" in arrays else np.empty((0, 0))
        steps = np.asarray(arrays["token_step_scores"], dtype=np.float64) if "token_step_scores" in arrays else np.empty((0, 0))
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64) if "segment_offsets" in arrays else np.empty(0, dtype=np.int64)
        if (
            set(arrays) != expected
            or method_ids != METHOD_IDS
            or scores.ndim != 2 or scores.shape[0] != len(METHOD_IDS)
            or steps.ndim != 2 or steps.shape[0] != len(METHOD_IDS)
            or steps.shape[1] != int(record["n_segments"])
            or scores.shape[1] != int(record["n_tokens"])
            or offsets.shape != (int(record["n_rows"]) + 1,)
            or offsets[-1] != int(record["n_segments"])
            or not np.isfinite(scores).all() or not np.isfinite(steps).all()
        ):
            raise RuntimeError(f"{cell_id}: frozen Phase-2 score schema failed")
        output[cell_id] = {
            "record": record, "arrays": arrays,
            "record_sha256": binding["record_sha256"],
        }
    if tuple(output) != CELLS:
        raise RuntimeError("Phase-2 score-freeze roster/order drifted")
    return output


def _phase1_preflight(score_root: Path, audit_path: Path) -> dict[str, dict[str, Any]]:
    """Verify the baseline freeze without importing target-bearing code."""

    freeze_path = score_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _verified_json(freeze_path)
    expected_methods = tuple(freeze.get("method_ids", ()))
    if not (
        freeze.get("schema_version") == PHASE1_SCHEMA
        and freeze.get("all_expected_scores_present") is True
        and freeze.get("labels_seen_during_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and tuple(freeze.get("expected_cells", ())) == CELLS
        and "LOCAL_IU29" in expected_methods
    ):
        raise RuntimeError("Phase-1 baseline score-freeze contract failed")
    environment = _verified_environment(freeze.get("environment"))
    if freeze.get("environment_sha256") != environment["environment_sha256"]:
        raise RuntimeError("Phase-1 baseline environment binding failed")
    _source_snapshot_ok(freeze)
    audit = _verified_json(audit_path)
    if not (
        audit.get("schema_version") == PHASE1_AUDIT_SCHEMA
        and audit.get("status") == "PASS"
        and audit.get("labels_opened_during_audit") is False
        and audit.get("score_freeze_sha256") == sha256_file(freeze_path)
        and audit.get("score_freeze_payload_sha256") == freeze["payload_sha256"]
        and audit.get("evaluation_source_sha256")
        == sha256_file(ROOT / "scripts/reconstruction_benchmark/evaluate_token_local_fusion_phase1.py")
        and audit.get("protocol_sha256")
        == sha256_file(ROOT / "docs/experiments/TOKEN_LOCAL_FUSION_OPTIMIZATION_V1.md")
        and isinstance(audit.get("independent_agent_id"), str)
        and audit.get("independent_agent_id")
    ):
        raise RuntimeError("Phase-1 baseline audit certificate failed")
    output: dict[str, dict[str, Any]] = {}
    for binding in freeze["records"]:
        cell_id = str(binding["cell_id"])
        record_path = score_root / str(binding["record_path"])
        if sha256_file(record_path) != binding["record_sha256"]:
            raise RuntimeError(f"{cell_id}: Phase-1 record changed")
        record = _verified_json(record_path)
        score_path = record_path.parent / str(record["score_path"])
        if sha256_file(score_path) != binding["score_sha256"]:
            raise RuntimeError(f"{cell_id}: Phase-1 score artifact changed")
        arrays = load_npz_no_pickle(score_path)
        required = {"row_ids", "segment_offsets", "method_ids", "token_step_scores", "primary_combined_scores", "equal_response_score"}
        if not required.issubset(arrays) or "LOCAL_IU29" not in tuple(map(str, arrays["method_ids"].tolist())):
            raise RuntimeError(f"{cell_id}: Phase-1 baseline arrays incomplete")
        output[cell_id] = {"record": record, "arrays": arrays}
    if tuple(output) != CELLS:
        raise RuntimeError("Phase-1 baseline roster/order drifted")
    return output


def _load_pb_labels(localization_release: Path) -> dict[str, dict[str, tuple[str, int]]]:
    evaluation_root = localization_release / "build_A/localization/evaluation"
    manifest = json.loads((evaluation_root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest["artifacts"]}
    path = evaluation_root / "localization_decisions.csv"
    if sha256_file(path) != expected.get("localization_decisions.csv"):
        raise RuntimeError("frozen ProcessBench label table hash failed")
    labels = {cell_id: {} for cell_id in PB_CELLS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell_id = str(row["cell_id"])
            if cell_id in labels and row["system_id"] == "deem_b3__loc_geomean_v1":
                row_id = str(row["row_id"])
                if row_id in labels[cell_id]:
                    raise RuntimeError(f"{cell_id}: duplicate ProcessBench label row")
                labels[cell_id][row_id] = (str(row["group_id"]), int(row["true_first_error"]))
    return labels


def _combined_step_scores(
    phase2: Mapping[str, Any], phase1: Mapping[str, Any], method_index: int,
) -> np.ndarray:
    """Join Phase-2 local step scores to the frozen equal-global response head."""

    p2 = np.asarray(phase2["arrays"]["token_step_scores"], dtype=np.float64)[method_index]
    p1_arrays = phase1["arrays"]
    p2_rows = tuple(map(str, phase2["arrays"]["row_ids"].tolist()))
    p1_rows = tuple(map(str, p1_arrays["row_ids"].tolist()))
    if p2_rows != p1_rows:
        raise RuntimeError("Phase-1/Phase-2 row roster differs")
    p2_offsets = np.asarray(phase2["arrays"]["segment_offsets"], dtype=np.int64)
    p1_offsets = np.asarray(p1_arrays["segment_offsets"], dtype=np.int64)
    if not np.array_equal(p2_offsets, p1_offsets):
        raise RuntimeError("Phase-1/Phase-2 segment roster differs")
    response_rank = empirical_midrank(np.asarray(p1_arrays["equal_response_score"], dtype=np.float64))
    expanded = np.repeat(response_rank, np.diff(p2_offsets))
    step_rank = empirical_midrank(p2)
    return np.sqrt(step_rank * expanded)


def _fit_rows(
    verified2: Mapping[str, Mapping[str, Any]],
    verified1: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, tuple[str, int]]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows_by_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell_id in PB_CELLS:
        p2 = verified2[cell_id]
        p1 = verified1[cell_id]
        p2_row_ids = tuple(map(str, p2["arrays"]["row_ids"].tolist()))
        if set(p2_row_ids) != set(labels[cell_id]):
            raise RuntimeError(f"{cell_id}: frozen score/label row join failed")
        offsets = np.asarray(p2["arrays"]["segment_offsets"], dtype=np.int64)
        record = p2["record"]
        for method_index, method in enumerate(METHOD_IDS):
            scores = _combined_step_scores(p2, p1, method_index)
            target = rows_by_method.setdefault((str(record["model_id"]), method), [])
            for row_index, row_id in enumerate(p2_row_ids):
                lo, hi = map(int, offsets[row_index:row_index + 2])
                group_id, first_error = labels[cell_id][row_id]
                target.append({
                    "row_id": row_id, "group_id": group_id,
                    "slice_id": str(record["slice_id"]), "first_error": int(first_error),
                    "step_scores": scores[lo:hi].tolist(), "cell_id": cell_id,
                    "model_id": str(record["model_id"]), "method_id": method,
                })
        # The incumbent is taken from the already frozen Phase-1 combined score,
        # not refit or recomputed from any target-bearing quantity.
        p1_methods = tuple(map(str, p1["arrays"]["method_ids"].tolist()))
        iu_index = p1_methods.index(REFERENCE)
        baseline = np.asarray(p1["arrays"]["primary_combined_scores"], dtype=np.float64)[iu_index]
        target = rows_by_method.setdefault((str(record["model_id"]), REFERENCE), [])
        for row_index, row_id in enumerate(p2_row_ids):
            lo, hi = map(int, offsets[row_index:row_index + 2])
            group_id, first_error = labels[cell_id][row_id]
            target.append({
                "row_id": row_id, "group_id": group_id,
                "slice_id": str(record["slice_id"]), "first_error": int(first_error),
                "step_scores": baseline[lo:hi].tolist(), "cell_id": cell_id,
                "model_id": str(record["model_id"]), "method_id": REFERENCE,
            })
    return rows_by_method


def _prmbench_rows(
    verified2: Mapping[str, Mapping[str, Any]],
    verified1: Mapping[str, Mapping[str, Any]],
    localization_release: Path,
    evaluation: Any,
) -> list[dict[str, Any]]:
    """Evaluate the secondary PRMBench step ranking after preflight only."""

    evaluation_root = localization_release / "build_A/localization/evaluation"
    path = evaluation_root / "prmbench_steps.npz"
    manifest_path = evaluation_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest["artifacts"]}
    if sha256_file(path) != expected.get("prmbench_steps.npz"):
        raise RuntimeError("frozen PRMBench label table hash failed")
    labels = load_npz_no_pickle(path)
    p2, p1 = verified2[PRM_CELL], verified1[PRM_CELL]
    row_ids = tuple(map(str, p2["arrays"]["row_ids"].tolist()))
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}
    offsets = np.asarray(p2["arrays"]["segment_offsets"], dtype=np.int64)
    response_ids = np.asarray(labels["response_row_ids"]).astype(str)
    step_counts = np.diff(np.asarray(labels["step_offsets"], dtype=np.int64))
    selected = {method: [] for method in (*METHOD_IDS, REFERENCE)}
    for response_id, count in zip(response_ids, step_counts):
        if response_id not in row_index:
            raise RuntimeError("PRMBench response row is absent from frozen score roster")
        index = row_index[response_id]
        lo, hi = map(int, offsets[index:index + 2])
        if hi - lo != int(count):
            raise RuntimeError("PRMBench response/step roster changed")
        for method_index, method in enumerate(METHOD_IDS):
            selected[method].append(_combined_step_scores(p2, p1, method_index)[lo:hi])
        p1_methods = tuple(map(str, p1["arrays"]["method_ids"].tolist()))
        iu_index = p1_methods.index(REFERENCE)
        baseline = np.asarray(p1["arrays"]["primary_combined_scores"], dtype=np.float64)[iu_index]
        selected[REFERENCE].append(baseline[lo:hi])
    labels_vector = np.asarray(labels["step_labels"], dtype=np.int8)
    return [
        {"method_id": method, **evaluation.prmbench_step_metrics(labels_vector, np.concatenate(selected[method]))}
        for method in selected
    ]


def _paired_bootstrap(
    decisions: Sequence[Mapping[str, Any]], methods: Sequence[str], reference: str,
    macro: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    subset_by_group: dict[str, str] = {}
    models = ("qwen3_4b", "qwen3_8b")
    subsets = ("gsm8k", "math", "olympiadbench", "omnimath")
    for row in decisions:
        group = str(row["group_id"])
        key = (str(row["model_id"]), str(row["method_id"]))
        if key in grouped.setdefault(group, {}):
            raise RuntimeError("duplicate paired ProcessBench row")
        grouped[group][key] = row
        previous = subset_by_group.setdefault(group, str(row["slice_id"]))
        if previous != str(row["slice_id"]):
            raise RuntimeError("bootstrap group crosses subsets")
    expected = {(model, method) for model in models for method in methods}
    if any(set(value) != expected for value in grouped.values()):
        raise RuntimeError("paired bootstrap group roster is incomplete")
    by_subset = {subset: sorted(group for group, value in subset_by_group.items() if value == subset) for subset in subsets}
    tensors = {}
    for subset in subsets:
        ids = by_subset[subset]
        tensor = np.zeros((len(ids), len(models), len(methods), 5), dtype=np.float64)
        for gi, group in enumerate(ids):
            for mi, model in enumerate(models):
                for method_i, method in enumerate(methods):
                    row = grouped[group][(model, method)]
                    label, prediction = int(row["true_first_error"]), int(row["prediction_step"])
                    error = label != -1
                    tensor[gi, mi, method_i] = (
                        float(error), float(not error), float(error and prediction == label),
                        float(error and prediction != -1 and abs(prediction - label) <= 1),
                        float((not error) and prediction == -1),
                    )
        tensors[subset] = tensor
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {method: np.empty(BOOTSTRAP_DRAWS, dtype=np.float64) for method in methods}
    stream = hashlib.sha256()
    for draw in range(BOOTSTRAP_DRAWS):
        panels = []
        for subset in subsets:
            tensor = tensors[subset]
            picks = rng.integers(0, len(tensor), size=len(tensor))
            stream.update(np.asarray(picks, dtype="<i8").tobytes(order="C"))
            counts = tensor[picks].sum(axis=0)
            errors, clean = counts[:, :, 0], counts[:, :, 1]
            exact = np.divide(counts[:, :, 2], errors, out=np.zeros_like(errors), where=errors > 0)
            within = np.divide(counts[:, :, 3], errors, out=np.zeros_like(errors), where=errors > 0)
            abstain = np.divide(counts[:, :, 4], clean, out=np.zeros_like(clean), where=clean > 0)
            f1 = np.divide(2 * exact * abstain, exact + abstain, out=np.zeros_like(exact), where=(exact + abstain) > 0)
            overall = (counts[:, :, 2] + counts[:, :, 4]) / (errors + clean)
            panels.append(np.stack((f1, exact, within, abstain, overall), axis=-1))
        observed = np.stack(panels).mean(axis=(0, 1))
        for method_i, method in enumerate(methods):
            samples[method][draw] = observed[method_i, 0]
    point = {method: float(macro[method]["official_macro_f1"]) for method in methods}
    comparisons = {}
    for method in methods:
        delta = samples[method] - samples[reference]
        value = {
            "delta_vs_local_iu29": point[method] - point[reference],
            "delta_vs_local_iu29_ci_low": float(np.percentile(delta, 2.5)),
            "delta_vs_local_iu29_ci_high": float(np.percentile(delta, 97.5)),
        }
        for other, label in (
            (LOCAL_TOKEN_B3, "b3"),
            (LOCAL_TOKEN_B3_SELF_INNOV, "self_innov"),
            (LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL, "nonrook"),
            (LOCAL_TOKEN_B3_ROOK_ALL_INNOV, "rook_all"),
        ):
            if method == other:
                continue
            paired = samples[method] - samples[other]
            value[f"delta_vs_{label}"] = point[method] - point[other]
            value[f"delta_vs_{label}_ci_low"] = float(np.percentile(paired, 2.5))
            value[f"delta_vs_{label}_ci_high"] = float(np.percentile(paired, 97.5))
        comparisons[method] = value
    return {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "source question paired across scorer models and methods", "draw_stream_sha256": stream.hexdigest(), "comparisons": comparisons}


def _final_selection(
    promotion: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered B3/rook/PSTG promotion and tie-break rules."""

    passed = {
        str(row["method_id"])
        for row in promotion if bool(row.get("promote", False))
    }
    if {
        LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
        LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    }.issubset(passed):
        delta = float(
            bootstrap["comparisons"][LOCAL_TOKEN_B3_ROOK_PSTG_INNOV][
                "delta_vs_rook_all"
            ]
        )
        if delta >= 0.001:
            selected = LOCAL_TOKEN_B3_ROOK_PSTG_INNOV
            reason = "PSTG passes all gates and exceeds all-rook by at least 0.001 F1"
        elif delta <= -0.001:
            selected = LOCAL_TOKEN_B3_ROOK_ALL_INNOV
            reason = "all-rook passes all gates and exceeds PSTG by more than 0.001 F1"
        else:
            selected = LOCAL_TOKEN_B3_ROOK_PSTG_INNOV
            reason = "PSTG and all-rook are within 0.001 F1; choose the sparser stable arm"
    elif LOCAL_TOKEN_B3_ROOK_PSTG_INNOV in passed:
        selected = LOCAL_TOKEN_B3_ROOK_PSTG_INNOV
        reason = "only PSTG among the cross-channel innovation candidates passes all gates"
    elif LOCAL_TOKEN_B3_ROOK_ALL_INNOV in passed:
        selected = LOCAL_TOKEN_B3_ROOK_ALL_INNOV
        reason = "only all-rook among the cross-channel innovation candidates passes all gates"
    elif LOCAL_TOKEN_B3 in passed:
        selected = LOCAL_TOKEN_B3
        reason = "B3 passes; no temporal-innovation arm passes its additional mechanism gates"
    else:
        selected = REFERENCE
        reason = "no promotable Phase-2 arm passes; retain LOCAL_IU29"
    innovation_positive = selected in {
        LOCAL_TOKEN_B3_ROOK_ALL_INNOV,
        LOCAL_TOKEN_B3_ROOK_PSTG_INNOV,
    }
    return {
        "schema_version": "token-local-temporal-innovation-b3-final-decision-v1",
        "selected_method": selected,
        "passed_promotion_methods": sorted(passed),
        "temporal_innovation_hypothesis": "SUPPORTED" if innovation_positive else "NOT_SUPPORTED",
        "selection_reason": reason,
        "fresh_confirmation_required": selected != REFERENCE,
        "fresh_confirmation_run": False,
    }


def evaluate(score_root: Path, audit_path: Path, phase1_root: Path, phase1_audit: Path, localization_release: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"evaluation output already exists: {output_root}")
    # No target-bearing module is imported before both score/audit preflights.
    verified2 = _phase2_preflight(score_root, audit_path)
    verified1 = _phase1_preflight(phase1_root, phase1_audit)

    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    pb_labels = _load_pb_labels(localization_release)
    rows_by_method = _fit_rows(verified2, verified1, pb_labels)
    by_model, decisions, by_cell = [], [], []
    for (model_id, method), rows in sorted(rows_by_method.items()):
        result = evaluation.crossfit_processbench_threshold(rows)
        aggregate = result["metrics"]["aggregate"]
        by_model.append({"model_id": model_id, "method_id": method, **aggregate})
        lookup = {str(row["row_id"]): row for row in result["decisions"]}
        for source in rows:
            decision = lookup[str(source["row_id"])]
            decisions.append({"method_id": method, "model_id": model_id, "cell_id": source["cell_id"], "slice_id": source["slice_id"], "row_id": source["row_id"], "group_id": source["group_id"], "true_first_error": int(source["first_error"]), "prediction_step": int(decision["prediction_step"]), "fold": int(decision["fold"])})
        for slice_id, metrics in result["metrics"]["per_subset"].items():
            by_cell.append({"model_id": model_id, "slice_id": slice_id, "cell_id": f"processbench_{slice_id}_{model_id}", "method_id": method, **{name: metrics[name] for name in METRIC_IDS}})
    methods = tuple(METHOD_IDS) + (REFERENCE,)
    macro = []
    for method in methods:
        values = [row for row in by_model if row["method_id"] == method]
        macro.append({"method_id": method, **{name: float(np.mean([row[name] for row in values])) for name in METRIC_IDS}})
    bootstrap = _paired_bootstrap(decisions, methods, REFERENCE, {row["method_id"]: row for row in macro})
    prm = _prmbench_rows(verified2, verified1, localization_release, evaluation)
    macro_by = {row["method_id"]: row for row in macro}
    prm_by = {row["method_id"]: row for row in prm}
    reference_cells = {row["cell_id"]: row for row in by_cell if row["method_id"] == REFERENCE}
    promotion = []
    for method in METHOD_IDS:
        comparison = bootstrap["comparisons"][method]
        deltas = [float(row["official_macro_f1"] - reference_cells[row["cell_id"]]["official_macro_f1"]) for row in by_cell if row["method_id"] == method]
        wins = sum(delta >= -TIE_TOLERANCE for delta in deltas)
        base = macro_by[method]
        ref = macro_by[REFERENCE]
        checks = {
            "f1_delta_at_least_0p005_vs_iu29": comparison["delta_vs_local_iu29"] >= 0.005,
            "paired_lower_bound_above_zero_vs_iu29": comparison["delta_vs_local_iu29_ci_low"] > 0,
            "six_of_eight_wins_or_ties": wins >= 6,
            "worst_cell_at_least_minus_0p02": min(deltas) >= -0.02,
            "exact_no_regression": base["first_error_exact"] - ref["first_error_exact"] >= -0.005,
            "within_one_no_regression": base["first_error_within_one"] - ref["first_error_within_one"] >= -0.005,
            "clean_abstention_no_regression": base["clean_abstention_accuracy"] - ref["clean_abstention_accuracy"] >= -0.01,
            "prmbench_auroc_guard": float(
                prm_by[method]["auroc"] - prm_by[REFERENCE]["auroc"]
            ) >= -0.002,
            "mechanical_and_firewall_checks": True,
        }
        if method != LOCAL_TOKEN_B3:
            checks.update({
                "innovation_delta_at_least_0p0025_vs_b3": comparison["delta_vs_b3"] >= 0.0025,
                "innovation_lower_bound_above_zero_vs_b3": comparison["delta_vs_b3_ci_low"] > 0,
            })
        if method in (LOCAL_TOKEN_B3_ROOK_ALL_INNOV, LOCAL_TOKEN_B3_ROOK_PSTG_INNOV):
            checks.update({
                "cross_channel_lower_bound_vs_self_only": comparison["delta_vs_self_innov_ci_low"] > 0,
                "cross_channel_lower_bound_vs_nonrook": comparison["delta_vs_nonrook_ci_low"] > 0,
            })
        promotable = method in (LOCAL_TOKEN_B3, LOCAL_TOKEN_B3_ROOK_ALL_INNOV, LOCAL_TOKEN_B3_ROOK_PSTG_INNOV)
        promotion.append({"method_id": method, "promote": bool(promotable and all(checks.values())), "f1_delta_vs_local_iu29": comparison["delta_vs_local_iu29"], "f1_delta_ci_low": comparison["delta_vs_local_iu29_ci_low"], "f1_delta_ci_high": comparison["delta_vs_local_iu29_ci_high"], "wins_or_ties": wins, "worst_cell_delta": min(deltas), "prmbench_auroc_delta": float(prm_by[method]["auroc"] - prm_by[REFERENCE]["auroc"]), "checks_json": json.dumps(checks, sort_keys=True, separators=(",", ":"))})

    final_decision = _final_selection(promotion, bootstrap)
    final_decision["payload_sha256"] = _payload_sha(final_decision)

    output_root.mkdir(parents=True, exist_ok=False)
    def write_csv(name: str, values: Sequence[Mapping[str, Any]]) -> None:
        values = list(values)
        with (output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write_csv("PROCESSBENCH_BY_MODEL.csv", by_model)
    write_csv("PROCESSBENCH_BY_CELL.csv", by_cell)
    write_csv("PROCESSBENCH_MACRO.csv", macro)
    write_csv("PROMOTION_DECISION.csv", promotion)
    write_csv("PRMBENCH_STEPS.csv", prm)
    atomic_write_json(output_root / "PROCESSBENCH_BOOTSTRAP.json", bootstrap)
    atomic_write_json(output_root / "FINAL_DECISION.json", final_decision)
    outputs = ("PROCESSBENCH_BY_MODEL.csv", "PROCESSBENCH_BY_CELL.csv", "PROCESSBENCH_MACRO.csv", "PROCESSBENCH_BOOTSTRAP.json", "PROMOTION_DECISION.csv", "PRMBENCH_STEPS.csv", "FINAL_DECISION.json")
    manifest = {"schema_version": "token-local-temporal-innovation-b3-evaluation-v1", "scores_preflighted_before_labels": True, "independent_audit_preflighted_before_labels": True, "phase2_score_freeze_sha256": sha256_file(score_root / "SCORE_FREEZE_MANIFEST.json"), "phase2_audit_certificate_sha256": sha256_file(audit_path), "phase1_score_freeze_sha256": sha256_file(phase1_root / "SCORE_FREEZE_MANIFEST.json"), "phase1_audit_certificate_sha256": sha256_file(phase1_audit), "promoted_methods": [row["method_id"] for row in promotion if row["promote"]], "selected_method": final_decision["selected_method"], "temporal_innovation_hypothesis": final_decision["temporal_innovation_hypothesis"], "retrospective_development_evidence": True, "fresh_confirmation": False, "outputs": {name: sha256_file(output_root / name) for name in outputs}}
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(output_root / "EVALUATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", **manifest}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-freeze", required=True)
    parser.add_argument("--audit-certificate", required=True)
    parser.add_argument("--phase1-score-freeze", required=True)
    parser.add_argument("--phase1-audit-certificate", required=True)
    parser.add_argument("--localization-release", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    evaluate(Path(args.score_freeze).resolve(), Path(args.audit_certificate).resolve(), Path(args.phase1_score_freeze).resolve(), Path(args.phase1_audit_certificate).resolve(), Path(args.localization_release).resolve(), Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
