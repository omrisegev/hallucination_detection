#!/usr/bin/env python3
"""Post-audit label import and evaluation for token-local Phase 1."""

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
from spectral_utils.token_local_fusion import (  # noqa: E402
    CONTROL_METHOD_IDS,
    LOCAL_EQUAL29,
    LOCAL_IU29,
    PRIMARY_METHOD_IDS,
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
ALL_LOCAL_METHODS = PRIMARY_METHOD_IDS + CONTROL_METHOD_IDS
IU_GLOBAL_CONTROL = "CONTROL_IU_GLOBAL__LOCAL_IU29"
ALL_EVALUATED_METHODS = ALL_LOCAL_METHODS + (IU_GLOBAL_CONTROL,)
REFERENCE = LOCAL_IU29
EQUAL_REFERENCE = LOCAL_EQUAL29
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082803
TIE_TOLERANCE = 0.0005
METRIC_IDS = (
    "official_macro_f1",
    "first_error_exact",
    "first_error_within_one",
    "clean_abstention_accuracy",
    "overall_decision_accuracy",
)
EXPECTED_SCORE_SCHEMA = "token-local-fusion-phase1-score-freeze-v1"
EXPECTED_AUDIT_SCHEMA = "token-local-fusion-phase1-prelabel-audit-v1"
HISTORICAL_PATH = ROOT / "results/gl_liu_factorial_v2/REPRODUCTION_CHECK.json"
PROTOCOL_PATH = ROOT / "docs/experiments/TOKEN_LOCAL_FUSION_OPTIMIZATION_V1.md"


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verified_json(path: Path, *, digest_field: str = "payload_sha256") -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop(digest_field, None)
    if digest != _payload_sha(body):
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


def _preflight(
    score_root: Path,
    audit_path: Path,
) -> dict[str, dict[str, Any]]:
    freeze_path = score_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _verified_json(freeze_path)
    if not (
        freeze.get("schema_version") == EXPECTED_SCORE_SCHEMA
        and freeze.get("all_expected_scores_present") is True
        and freeze.get("labels_seen_during_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and tuple(freeze.get("expected_cells", ())) == CELLS
        and tuple(freeze.get("method_ids", ())) == ALL_LOCAL_METHODS
    ):
        raise RuntimeError("target-free score-freeze contract failed")
    environment = _verified_environment(freeze.get("environment"))
    if freeze.get("environment_sha256") != environment["environment_sha256"]:
        raise RuntimeError("score-freeze environment binding failed")
    for source in freeze.get("source_snapshot", ()):
        path = ROOT / str(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"source changed after score freeze: {source['path']}")

    audit = _verified_json(audit_path)
    if not (
        audit.get("schema_version") == EXPECTED_AUDIT_SCHEMA
        and audit.get("status") == "PASS"
        and audit.get("labels_opened_during_audit") is False
        and audit.get("score_freeze_sha256") == sha256_file(freeze_path)
        and audit.get("score_freeze_payload_sha256") == freeze["payload_sha256"]
        and audit.get("evaluation_source_sha256") == sha256_file(Path(__file__))
        and audit.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
        and isinstance(audit.get("independent_agent_id"), str)
        and audit.get("independent_agent_id")
    ):
        raise RuntimeError("independent pre-label audit certificate failed")

    output: dict[str, dict[str, Any]] = {}
    for binding in freeze["records"]:
        cell_id = str(binding["cell_id"])
        record_path = score_root / str(binding["record_path"])
        if sha256_file(record_path) != binding["record_sha256"]:
            raise RuntimeError(f"{cell_id}: frozen cell record changed")
        record = _verified_json(record_path)
        record_environment = _verified_environment(record.get("environment"))
        score_path = record_path.parent / str(record["score_path"])
        if (
            sha256_file(score_path) != record["score_sha256"]
            or record["score_sha256"] != binding["score_sha256"]
        ):
            raise RuntimeError(f"{cell_id}: frozen score artifact changed")
        arrays = load_npz_no_pickle(score_path)
        expected_arrays = {
            "row_ids", "segment_offsets", "method_ids", "kept_stream_mask",
            "method_weights", "token_step_scores", "primary_combined_scores",
            "equal_response_score", "iu_response_score",
            "iu_global_iu_local_score",
        }
        method_ids = tuple(map(str, arrays["method_ids"].tolist()))
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        if (
            set(arrays) != expected_arrays
            or method_ids != ALL_LOCAL_METHODS
            or tuple(record["method_ids"]) != ALL_LOCAL_METHODS
            or record.get("labels_seen_during_fit") is not False
            or record.get("targets_accessed_during_fit") is not False
            or record_environment != environment
            or record.get("environment_sha256") != environment["environment_sha256"]
            or record["fit_diagnostics"].get("local_iu_incumbent_alias_within_1e_12") is not True
            or record["fit_diagnostics"].get("score_reconstruction_exact") is not True
            or offsets.shape != (int(record["n_rows"]) + 1,)
            or offsets[-1] != int(record["n_steps"])
            or np.asarray(arrays["primary_combined_scores"]).shape
            != (len(ALL_LOCAL_METHODS), int(record["n_steps"]))
        ):
            raise RuntimeError(f"{cell_id}: frozen score schema/health failed")
        output[cell_id] = {
            "record": record,
            "arrays": arrays,
            "record_sha256": binding["record_sha256"],
        }
    if tuple(output) != CELLS:
        raise RuntimeError("score-freeze cell order/roster drifted")
    return output


def _load_pb_labels(
    localization_release: Path,
) -> dict[str, dict[str, tuple[str, int]]]:
    evaluation_root = localization_release / "build_A/localization/evaluation"
    manifest = json.loads((evaluation_root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest["artifacts"]}
    decisions_path = evaluation_root / "localization_decisions.csv"
    if sha256_file(decisions_path) != expected.get("localization_decisions.csv"):
        raise RuntimeError("frozen ProcessBench label table hash failed")
    labels = {cell_id: {} for cell_id in PB_CELLS}
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell_id = str(row["cell_id"])
            if cell_id not in labels or row["system_id"] != "deem_b3__loc_geomean_v1":
                continue
            row_id = str(row["row_id"])
            if row_id in labels[cell_id]:
                raise RuntimeError(f"{cell_id}: duplicate ProcessBench label row")
            labels[cell_id][row_id] = (
                str(row["group_id"]), int(row["true_first_error"])
            )
    return labels


def _processbench(
    verified: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, tuple[str, int]]],
    evaluation: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fit_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell_id in PB_CELLS:
        item = verified[cell_id]
        record, arrays = item["record"], item["arrays"]
        row_ids = tuple(map(str, arrays["row_ids"].tolist()))
        if set(row_ids) != set(labels[cell_id]):
            raise RuntimeError(f"{cell_id}: frozen score/label row join failed")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        score_matrix = np.asarray(arrays["primary_combined_scores"], dtype=float)
        special = np.asarray(arrays["iu_global_iu_local_score"], dtype=float)
        for method_id in ALL_EVALUATED_METHODS:
            scores = (
                special if method_id == IU_GLOBAL_CONTROL
                else score_matrix[ALL_LOCAL_METHODS.index(method_id)]
            )
            target = fit_rows.setdefault((str(record["model_id"]), method_id), [])
            for row_index, row_id in enumerate(row_ids):
                lo, hi = map(int, offsets[row_index:row_index + 2])
                group_id, first_error = labels[cell_id][row_id]
                target.append({
                    "row_id": row_id,
                    "group_id": group_id,
                    "slice_id": str(record["slice_id"]),
                    "first_error": int(first_error),
                    "step_scores": scores[lo:hi].tolist(),
                    "cell_id": cell_id,
                    "model_id": str(record["model_id"]),
                    "method_id": method_id,
                })

    by_model = []
    decisions = []
    by_cell = []
    for (model_id, method_id), rows in sorted(fit_rows.items()):
        result = evaluation.crossfit_processbench_threshold(rows)
        aggregate = result["metrics"]["aggregate"]
        by_model.append({"model_id": model_id, "method_id": method_id, **aggregate})
        decision_lookup = {
            str(row["row_id"]): row for row in result["decisions"]
        }
        for source in rows:
            decision = decision_lookup[str(source["row_id"])]
            decisions.append({
                "method_id": method_id,
                "model_id": model_id,
                "cell_id": source["cell_id"],
                "slice_id": source["slice_id"],
                "row_id": source["row_id"],
                "group_id": source["group_id"],
                "true_first_error": int(source["first_error"]),
                "prediction_step": int(decision["prediction_step"]),
                "fold": int(decision["fold"]),
            })
        for slice_id, metrics in result["metrics"]["per_subset"].items():
            by_cell.append({
                "model_id": model_id,
                "slice_id": slice_id,
                "cell_id": f"processbench_{slice_id}_{model_id}",
                "method_id": method_id,
                **{name: metrics[name] for name in METRIC_IDS},
            })

    macro = []
    for method_id in ALL_EVALUATED_METHODS:
        rows = [row for row in by_model if row["method_id"] == method_id]
        if len(rows) != 2:
            raise RuntimeError(f"{method_id}: ProcessBench does not have two model panels")
        macro.append({
            "method_id": method_id,
            **{
                name: float(np.mean([row[name] for row in rows]))
                for name in METRIC_IDS
            },
        })
    bootstrap = _paired_pb_bootstrap(decisions, macro)
    return by_model, by_cell, macro, bootstrap


def _paired_pb_bootstrap(
    decisions: Sequence[Mapping[str, Any]],
    macro: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(decisions)
    models = ("qwen3_4b", "qwen3_8b")
    subsets = ("gsm8k", "math", "olympiadbench", "omnimath")
    methods = ALL_EVALUATED_METHODS
    grouped: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    group_subset = {}
    for row in rows:
        group_id = str(row["group_id"])
        key = (str(row["model_id"]), str(row["method_id"]))
        if key in grouped.setdefault(group_id, {}):
            raise RuntimeError("duplicate method/model row in ProcessBench bootstrap group")
        grouped[group_id][key] = row
        previous = group_subset.setdefault(group_id, str(row["slice_id"]))
        if previous != str(row["slice_id"]):
            raise RuntimeError("ProcessBench bootstrap group crosses subsets")
    by_subset = {
        subset: sorted(group_id for group_id, value in group_subset.items() if value == subset)
        for subset in subsets
    }
    expected_keys = {(model, method) for model in models for method in methods}
    if any(set(grouped[group_id]) != expected_keys for group_id in grouped):
        raise RuntimeError("ProcessBench paired bootstrap group roster is incomplete")

    # Tensor axes: subset, group-within-subset, model, method.  The five
    # sufficient indicators reproduce the frozen decisions without refitting.
    tensors = {}
    for subset in subsets:
        ids = by_subset[subset]
        values = np.zeros((len(ids), len(models), len(methods), 5), dtype=np.float64)
        for group_index, group_id in enumerate(ids):
            for model_index, model in enumerate(models):
                for method_index, method in enumerate(methods):
                    row = grouped[group_id][(model, method)]
                    label = int(row["true_first_error"])
                    prediction = int(row["prediction_step"])
                    error = label != -1
                    values[group_index, model_index, method_index] = (
                        float(error),
                        float(not error),
                        float(error and prediction == label),
                        float(error and prediction != -1 and abs(prediction - label) <= 1),
                        float((not error) and prediction == -1),
                    )
        tensors[subset] = values

    point = {str(row["method_id"]): row for row in macro}
    samples = {
        method: {name: np.empty(BOOTSTRAP_DRAWS, dtype=np.float64) for name in METRIC_IDS}
        for method in methods
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw_hasher = hashlib.sha256()
    for draw in range(BOOTSTRAP_DRAWS):
        per_subset = []
        for subset in subsets:
            values = tensors[subset]
            picks = rng.integers(0, len(values), size=len(values))
            draw_hasher.update(np.asarray(picks, dtype="<i8").tobytes(order="C"))
            counts = values[picks].sum(axis=0)  # model, method, five counters
            n_error = counts[:, :, 0]
            n_clean = counts[:, :, 1]
            exact = np.divide(
                counts[:, :, 2], n_error,
                out=np.full_like(n_error, np.nan), where=n_error > 0,
            )
            within = np.divide(
                counts[:, :, 3], n_error,
                out=np.full_like(n_error, np.nan), where=n_error > 0,
            )
            abstain = np.divide(
                counts[:, :, 4], n_clean,
                out=np.full_like(n_clean, np.nan), where=n_clean > 0,
            )
            denominator = exact + abstain
            f1 = np.divide(
                2.0 * exact * abstain, denominator,
                out=np.zeros_like(denominator), where=denominator > 0,
            )
            overall = (counts[:, :, 2] + counts[:, :, 4]) / (n_error + n_clean)
            per_subset.append(np.stack((f1, exact, within, abstain, overall), axis=-1))
        observed = np.stack(per_subset).mean(axis=(0, 1))  # method, metric
        if not np.isfinite(observed).all():
            raise RuntimeError("a ProcessBench bootstrap draw lost class support")
        for method_index, method in enumerate(methods):
            for metric_index, name in enumerate(METRIC_IDS):
                samples[method][name][draw] = observed[method_index, metric_index]

    comparisons = {}
    for method in methods:
        reference_delta = samples[method]["official_macro_f1"] - samples[REFERENCE]["official_macro_f1"]
        equal_delta = samples[method]["official_macro_f1"] - samples[EQUAL_REFERENCE]["official_macro_f1"]
        comparisons[method] = {
            "delta_vs_local_iu29": float(
                point[method]["official_macro_f1"] - point[REFERENCE]["official_macro_f1"]
            ),
            "delta_vs_local_iu29_ci_low": float(np.percentile(reference_delta, 2.5)),
            "delta_vs_local_iu29_ci_high": float(np.percentile(reference_delta, 97.5)),
            "delta_vs_equal_equal": float(
                point[method]["official_macro_f1"] - point[EQUAL_REFERENCE]["official_macro_f1"]
            ),
            "delta_vs_equal_equal_ci_low": float(np.percentile(equal_delta, 2.5)),
            "delta_vs_equal_equal_ci_high": float(np.percentile(equal_delta, 97.5)),
        }
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "source question paired across Qwen scorer models and methods",
        "strata": list(subsets),
        "draw_stream_sha256": draw_hasher.hexdigest(),
        "comparisons": comparisons,
    }


def _prmbench(
    verified: Mapping[str, Mapping[str, Any]],
    localization_release: Path,
    evaluation: Any,
) -> list[dict[str, Any]]:
    labels = load_npz_no_pickle(
        localization_release / "build_A/localization/evaluation/prmbench_steps.npz"
    )
    arrays = verified[PRM_CELL]["arrays"]
    row_ids = np.asarray(arrays["row_ids"]).astype(str)
    offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}
    selected: dict[str, list[np.ndarray]] = {method: [] for method in ALL_EVALUATED_METHODS}
    primary = np.asarray(arrays["primary_combined_scores"], dtype=np.float64)
    special = np.asarray(arrays["iu_global_iu_local_score"], dtype=np.float64)
    for response_id, n_steps in zip(
        labels["response_row_ids"].astype(str), np.diff(labels["step_offsets"])
    ):
        index = row_index[response_id]
        lo, hi = map(int, offsets[index:index + 2])
        if hi - lo != int(n_steps):
            raise RuntimeError("PRMBench response/step roster changed")
        for method in ALL_EVALUATED_METHODS:
            scores = special if method == IU_GLOBAL_CONTROL else primary[ALL_LOCAL_METHODS.index(method)]
            selected[method].append(scores[lo:hi])
    y = np.asarray(labels["step_labels"], dtype=np.int8)
    return [
        {"method_id": method, **evaluation.prmbench_step_metrics(
            y, np.concatenate(selected[method])
        )}
        for method in ALL_EVALUATED_METHODS
    ]


def _promotion_rows(
    macro: Sequence[Mapping[str, Any]],
    by_cell: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    prm: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    macro_by = {str(row["method_id"]): row for row in macro}
    prm_by = {str(row["method_id"]): row for row in prm}
    reference_cells = {
        str(row["cell_id"]): row for row in by_cell if row["method_id"] == REFERENCE
    }
    output = []
    for method in PRIMARY_METHOD_IDS:
        row = macro_by[method]
        ref = macro_by[REFERENCE]
        cell_deltas = [
            float(cell["official_macro_f1"] - reference_cells[str(cell["cell_id"])]["official_macro_f1"])
            for cell in by_cell if cell["method_id"] == method
        ]
        wins_or_ties = sum(delta >= -TIE_TOLERANCE for delta in cell_deltas)
        comparison = bootstrap["comparisons"][method]
        checks = {
            "f1_delta_at_least_0p005": comparison["delta_vs_local_iu29"] >= 0.005,
            "paired_lower_bound_above_zero": comparison["delta_vs_local_iu29_ci_low"] > 0.0,
            "six_of_eight_wins_or_ties": wins_or_ties >= 6,
            "worst_cell_at_least_minus_0p02": min(cell_deltas) >= -0.02,
            "exact_regression_at_most_0p005": row["first_error_exact"] - ref["first_error_exact"] >= -0.005,
            "within_one_regression_at_most_0p005": row["first_error_within_one"] - ref["first_error_within_one"] >= -0.005,
            "clean_abstention_regression_at_most_0p01": row["clean_abstention_accuracy"] - ref["clean_abstention_accuracy"] >= -0.01,
            "beats_equal_equal_by_0p03": comparison["delta_vs_equal_equal"] >= 0.03,
            "equal_equal_lower_bound_above_zero": comparison["delta_vs_equal_equal_ci_low"] > 0.0,
            "prmbench_auroc_guard": float(prm_by[method]["auroc"] - prm_by[REFERENCE]["auroc"]) >= -0.002,
            "mechanical_and_firewall_checks": True,
        }
        output.append({
            "method_id": method,
            "promote": bool(all(checks.values()) and method != REFERENCE),
            "f1_delta_vs_local_iu29": comparison["delta_vs_local_iu29"],
            "f1_delta_ci_low": comparison["delta_vs_local_iu29_ci_low"],
            "f1_delta_ci_high": comparison["delta_vs_local_iu29_ci_high"],
            "wins_or_ties": wins_or_ties,
            "losses": 8 - wins_or_ties,
            "worst_cell_delta": min(cell_deltas),
            "exact_delta": float(row["first_error_exact"] - ref["first_error_exact"]),
            "within_one_delta": float(row["first_error_within_one"] - ref["first_error_within_one"]),
            "clean_abstention_delta": float(row["clean_abstention_accuracy"] - ref["clean_abstention_accuracy"]),
            "delta_vs_equal_equal": comparison["delta_vs_equal_equal"],
            "delta_vs_equal_equal_ci_low": comparison["delta_vs_equal_equal_ci_low"],
            "prmbench_auroc_delta": float(prm_by[method]["auroc"] - prm_by[REFERENCE]["auroc"]),
            "checks_json": json.dumps(checks, sort_keys=True, separators=(",", ":")),
        })
    return output


def evaluate(
    score_root: Path,
    audit_path: Path,
    localization_release: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"evaluation output already exists: {output_root}")
    verified = _preflight(score_root, audit_path)

    # The target-bearing evaluator and tables are imported/opened only after
    # the score-freeze and independent-audit gates above have passed.
    evaluation = importlib.import_module(
        "spectral_utils.reconstruction_benchmark.localization_evaluation"
    )
    pb_labels = _load_pb_labels(localization_release)
    output_root.mkdir(parents=True, exist_ok=False)
    by_model, by_cell, macro, bootstrap = _processbench(
        verified, pb_labels, evaluation
    )
    prm = _prmbench(verified, localization_release, evaluation)
    promotion = _promotion_rows(macro, by_cell, bootstrap, prm)

    _write_csv(output_root / "PROCESSBENCH_BY_MODEL.csv", by_model)
    _write_csv(output_root / "PROCESSBENCH_BY_CELL.csv", by_cell)
    _write_csv(output_root / "PROCESSBENCH_MACRO.csv", macro)
    _write_csv(output_root / "PRMBENCH_STEPS.csv", prm)
    _write_csv(output_root / "PROMOTION_DECISION.csv", promotion)
    atomic_write_json(output_root / "PROCESSBENCH_BOOTSTRAP.json", bootstrap)

    historical = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
    if historical.get("all_frozen_hashes_equal") is not True:
        raise RuntimeError("historical GL-LIU/Mind-the-Gap reproduction check failed")
    headline = historical["headline"]
    primary_rows = {str(row["method_id"]): row for row in macro}
    prm_rows = {str(row["method_id"]): row for row in prm}
    lines = [
        "# Token-local fusion optimization v1 — Phase 1", "",
        "Retrospective development evidence. Scores were frozen and independently audited before label import.", "",
        "| Local method (equal-30 global unless noted) | PB macro F1 | exact | within one | clean abstain | PRM AUROC | PRM AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ALL_EVALUATED_METHODS:
        pb, pr = primary_rows[method], prm_rows[method]
        lines.append(
            f"| `{method}` | {pb['official_macro_f1']:.6f} | {pb['first_error_exact']:.6f} | "
            f"{pb['first_error_within_one']:.6f} | {pb['clean_abstention_accuracy']:.6f} | "
            f"{pr['auroc']:.6f} | {pr['auprc']:.6f} |"
        )
    lines += [
        "", "## Historical separate-protocol references", "",
        f"- Frozen GL-LIU v1: ProcessBench F1 `{float(headline['gl_liu_v1_reproduced_f1']):.6f}`.",
        f"- Mind-the-Gap common control: ProcessBench F1 `{float(headline['mindgap_control_f1']):.6f}`.",
        "- These use the historical 100 repeated calibration/evaluation splits and are not pooled with the new five-fold estimates.",
        "", "## Promotion decision", "",
    ]
    promoted = [row["method_id"] for row in promotion if row["promote"]]
    if promoted:
        lines.append("Passing retrospective challenger(s): " + ", ".join(f"`{value}`" for value in promoted) + ".")
    else:
        lines.append("No Phase-1 candidate passes every preregistered gate; retain `LOCAL_IU29`.")
    lines += [
        "", "PRMBench is a secondary transfer guard and cannot rescue a ProcessBench failure.",
        "No result here is fresh-population confirmation.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = (
        "PROCESSBENCH_BY_MODEL.csv", "PROCESSBENCH_BY_CELL.csv",
        "PROCESSBENCH_MACRO.csv", "PROCESSBENCH_BOOTSTRAP.json",
        "PRMBENCH_STEPS.csv", "PROMOTION_DECISION.csv", "REPORT.md",
    )
    manifest = {
        "schema_version": "token-local-fusion-phase1-evaluation-v1",
        "scores_preflighted_before_labels": True,
        "independent_audit_preflighted_before_labels": True,
        "score_freeze_sha256": sha256_file(score_root / "SCORE_FREEZE_MANIFEST.json"),
        "audit_certificate_sha256": sha256_file(audit_path),
        "historical_reference_sha256": sha256_file(HISTORICAL_PATH),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "promoted_methods": promoted,
        "retained_method": promoted[0] if len(promoted) == 1 else LOCAL_IU29,
        "retrospective_development_evidence": True,
        "fresh_confirmation": False,
        "outputs": {name: sha256_file(output_root / name) for name in outputs},
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(output_root / "EVALUATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", **manifest}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-freeze", required=True)
    parser.add_argument("--audit-certificate", required=True)
    parser.add_argument("--localization-release", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    evaluate(
        Path(args.score_freeze).resolve(),
        Path(args.audit_certificate).resolve(),
        Path(args.localization_release).resolve(),
        Path(args.out_dir).resolve(),
    )


if __name__ == "__main__":
    main()
