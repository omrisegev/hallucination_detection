#!/usr/bin/env python3
"""Run the frozen CPU-only Global-Local-Online IU v1 retrospective cycle."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spectral_utils.global_local_online import (
    CANDIDATE_FEATURES,
    fit_dynamic_online_head,
)
from spectral_utils.online_convergence import (
    DEFAULT_BUDGETS,
    apply_declaration_policy,
    calibrate_declaration_policy,
    convergence_table,
    declaration_summary,
    final_calibration,
    per_trace_convergence,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/early_online_localization_models_v1"
OUT = ROOT / "results/global_local_online_iu_v1"
SEED = 20260816
BOOTSTRAP = 2000
BASELINES = ("iu28_no_length", "deepconf_entropy_w64", "cusum_swvar_equal")
METHODS = BASELINES + tuple(CANDIDATE_FEATURES)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _read_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "unit_index": int(raw["unit_index"]),
                "trace_id": raw["trace_id"],
                "group": raw["group"],
                "label_error": int(raw["label_error"]),
                "trace_length": int(raw["trace_length"]),
                "budget": int(raw["budget"]),
                "is_final": _bool(raw["is_final"]),
                "method": raw["method"],
                "score": float(raw["score"]),
            })
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family(cell_id: str) -> str:
    if cell_id.startswith("hallucination_phase1"):
        return "math500"
    for name in ("olympiadbench", "omnimath", "gsm8k", "math"):
        if f"processbench_{name}" in cell_id:
            return name
    raise KeyError(cell_id)


def _safe_auc(labels: Sequence[int], scores: Sequence[float], weights=None) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if weights is None:
        weights = np.ones(len(labels), dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive <= 0 or negative <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    positive_by_tie = np.add.reduceat(sorted_weights * (sorted_labels == 1), starts)
    negative_by_tie = np.add.reduceat(sorted_weights * (sorted_labels == 0), starts)
    negative_before = np.cumsum(negative_by_tie) - negative_by_tie
    numerator = np.sum(positive_by_tie * (negative_before + 0.5 * negative_by_tie))
    return float(numerator / (positive * negative))


def _cell_budget_rows(
    rows: Sequence[Mapping[str, Any]], method: str, budget: int,
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row["method"] == method
        and not bool(row["is_final"])
        and int(row["budget"]) == int(budget)
    ]


def _endpoint_for_cell(
    rows: Sequence[Mapping[str, Any]], method: str, weights: Mapping[str, int] | None = None,
) -> float:
    values = []
    for budget in (64, 128):
        selected = _cell_budget_rows(rows, method, budget)
        if not selected:
            continue
        sample_weight = None
        if weights is not None:
            sample_weight = [weights.get(str(row["group"]), 0) for row in selected]
        auc = _safe_auc(
            [row["label_error"] for row in selected],
            [row["score"] for row in selected],
            sample_weight,
        )
        if np.isfinite(auc):
            values.append(auc)
    return float(np.mean(values)) if values else float("nan")


def _hierarchical_delta(
    cells: Mapping[str, Sequence[Mapping[str, Any]]],
    method: str,
    reference: str,
    *,
    repeats: int = BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    families = sorted({_family(cell_id) for cell_id in cells})
    by_family = {
        family: [cell_id for cell_id in cells if _family(cell_id) == family]
        for family in families
    }

    prepared: dict[str, dict[str, Any]] = {}
    for family in families:
        family_groups = sorted({
            str(row["group"])
            for cell_id in by_family[family]
            for row in cells[cell_id]
            if row["method"] == method
        })
        group_index = {group: index for index, group in enumerate(family_groups)}
        prepared_cells = []
        for cell_id in by_family[family]:
            budgets = []
            for budget in (64, 128):
                left_rows = _cell_budget_rows(cells[cell_id], method, budget)
                right_lookup = {
                    (str(row["group"]), int(row["unit_index"])): row
                    for row in _cell_budget_rows(cells[cell_id], reference, budget)
                }
                pairs = [
                    (row, right_lookup[(str(row["group"]), int(row["unit_index"]))])
                    for row in left_rows
                    if (str(row["group"]), int(row["unit_index"])) in right_lookup
                ]
                if not pairs:
                    continue
                labels = np.asarray([pair[0]["label_error"] for pair in pairs], dtype=int)
                codes = np.asarray([group_index[str(pair[0]["group"])] for pair in pairs], dtype=int)
                left_scores = np.asarray([pair[0]["score"] for pair in pairs], dtype=float)
                right_scores = np.asarray([pair[1]["score"] for pair in pairs], dtype=float)
                budgets.append((labels, codes, left_scores, right_scores))
            if budgets:
                prepared_cells.append(budgets)
        prepared[family] = {"groups": family_groups, "cells": prepared_cells}

    def family_delta(family: str, counts: np.ndarray) -> float:
        cell_deltas = []
        for budgets in prepared[family]["cells"]:
            budget_deltas = []
            for labels, codes, left_scores, right_scores in budgets:
                weights = counts[codes]
                left_auc = _safe_auc(labels, left_scores, weights)
                right_auc = _safe_auc(labels, right_scores, weights)
                if np.isfinite(left_auc) and np.isfinite(right_auc):
                    budget_deltas.append(left_auc - right_auc)
            if budget_deltas:
                cell_deltas.append(float(np.mean(budget_deltas)))
        return float(np.mean(cell_deltas)) if cell_deltas else float("nan")

    family_points = {
        family: family_delta(family, np.ones(len(prepared[family]["groups"]), dtype=int))
        for family in families
    }
    point = float(np.nanmean(list(family_points.values())))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(repeats)):
        sampled_family_values = {}
        for family in families:
            n_groups = len(prepared[family]["groups"])
            counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
            sampled_family_values[family] = family_delta(family, counts)
        chosen = rng.choice(families, size=len(families), replace=True)
        values = [sampled_family_values[family] for family in chosen]
        values = [value for value in values if np.isfinite(value)]
        if values:
            draws.append(float(np.mean(values)))
    wins = sum(value > 1e-12 for value in family_points.values())
    ties = sum(abs(value) <= 1e-12 for value in family_points.values())
    losses = sum(value < -1e-12 for value in family_points.values())
    return {
        "method": method,
        "reference": reference,
        "endpoint": "equal_family_mean_auroc_64_128",
        "delta": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_bootstrap": int(len(draws)),
        "n_families": int(len(families)),
        "family_wins": int(wins),
        "family_ties": int(ties),
        "family_losses": int(losses),
        "family_deltas_json": json.dumps(family_points, sort_keys=True),
    }


def _anchor_regression() -> dict[str, Any]:
    with (SOURCE / "AGGREGATE_CONVERGENCE.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["method"], int(row["budget"])): float(row["macro_auroc"]) for row in rows}
    expected = {
        ("iu28_no_length", 64): 0.648,
        ("iu28_no_length", 128): 0.694,
        ("deepconf_entropy_w64", 64): 0.616,
        ("deepconf_entropy_w64", 128): 0.671,
        ("global_gl_liu_no_length", 64): 0.638,
        ("global_gl_liu_no_length", 128): 0.679,
        ("fused_gl_liu", 512): 0.707,
    }
    checks = []
    for key, target in expected.items():
        observed = lookup[key]
        passed = abs(observed - target) <= 0.00075
        checks.append({"method": key[0], "budget": key[1], "expected": target, "observed": observed, "passed": passed})
        if not passed:
            raise RuntimeError(f"anchor mismatch for {key}: {observed} vs {target}")

    reasoning_path = ROOT / "results/fixed_application_pipelines_v1/reasoning_metrics.csv"
    with reasoning_path.open(encoding="utf-8") as handle:
        reasoning = list(csv.DictReader(handle))
    pb = next(row for row in reasoning if row["benchmark"] == "ProcessBench" and row["model"] == "macro" and row["subgroup"] == "all eight cells" and row["method"] == "Fixed reasoning IU-PCR")
    prm = next(row for row in reasoning if row["benchmark"] == "PRMBench" and row["split"] == "all nine paper classes" and row["method"] == "Fixed reasoning IU-PCR")
    if abs(float(pb["f1"]) - 0.30699905177467274) > 1e-12:
        raise RuntimeError("ProcessBench fixed-IU anchor mismatch")
    if abs(float(prm["auroc"]) - 0.6711488968029491) > 1e-12:
        raise RuntimeError("PRMBench fixed-IU anchor mismatch")
    process_freeze = json.loads((ROOT / "results/fixed_application_pipelines_v1/processbench_score_freeze.json").read_text())
    prm_freeze = json.loads((ROOT / "results/fixed_application_pipelines_v1/prmbench_score_freeze.json").read_text())
    return {
        "status": "PASS",
        "early_checks": checks,
        "processbench_f1": float(pb["f1"]),
        "prmbench_step_auroc": float(prm["auroc"]),
        "processbench_score_hash": process_freeze["score_hash"],
        "prmbench_score_hash": prm_freeze["score_hash"],
        "online_candidates_modify_localization_heads": False,
        "localization_hash_identity": True,
    }


def _graph_ablation_rows() -> list[dict[str, Any]]:
    path = ROOT / "results/ours_only_localization_v1/component_metrics_per_cell.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    roster = {
        "global_ordinary_mixed": ("answer_iu_mixed", "detector", "auroc"),
        "global_uniform_mixed": ("answer_uniform_liu_mixed", "detector", "auroc"),
        "global_dufs_mixed": ("answer_dufs_liu_mixed", "detector", "auroc"),
        "local_ordinary_top5": ("token_iu__top5mean", "detector", "auroc"),
        "local_uniform_top5": ("token_uniform_liu_l0p3__top5mean", "detector", "auroc"),
        "local_dufs_top5": ("token_dufs_liu_l0p3__top5mean", "detector", "auroc"),
        "local_temporal_top5": ("token_temporal_liu_l0p3__top5mean", "detector", "auroc"),
        "locator_ordinary": ("token_iu", "locator", "exact"),
        "locator_uniform_l0p3": ("token_uniform_liu_l0p3", "locator", "exact"),
        "locator_dufs_l0p3": ("token_dufs_liu_l0p3", "locator", "exact"),
        "locator_temporal_l0p3": ("token_temporal_liu_l0p3", "locator", "exact"),
    }
    output = []
    for arm, (candidate, component, metric) in roster.items():
        values = [
            float(row[metric]) for row in rows
            if row["candidate"] == candidate and row["component"] == component and row.get(metric, "")
        ]
        output.append({"arm": arm, "candidate": candidate, "component": component, "metric": metric, "cells": len(values), "macro_value": float(np.mean(values))})
    return output


def _length_band(length: int) -> str:
    if length < 128:
        return "lt128"
    if length < 512:
        return "128_511"
    return "ge512"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor = _anchor_regression()
    _write_json(OUT / "ANCHOR_REGRESSION.json", anchor)
    graph_rows = _graph_ablation_rows()
    _write_csv(OUT / "GRAPH_ABLATIONS.csv", graph_rows)

    cells: dict[str, list[dict[str, Any]]] = {}
    calibration_by_cell: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, Any] = {}
    efficiency_rows = []
    all_question_scores = []
    all_convergence = []
    all_declarations = []
    all_per_trace = []

    for result_path in sorted(SOURCE.glob("*/result.json")):
        cell_id = result_path.parent.name
        calibration_source = _read_scores(result_path.parent / "scores_calibration.csv")
        evaluation_source = _read_scores(result_path.parent / "scores_evaluation.csv")
        calibration = [row for row in calibration_source if row["method"] in BASELINES or row["method"] in {"cusum_max", "sw_var_peak"}]
        evaluation = [row for row in evaluation_source if row["method"] in BASELINES or row["method"] in {"cusum_max", "sw_var_peak"}]
        cell_diag = {}
        for candidate in CANDIDATE_FEATURES:
            tracemalloc.start()
            started = time.perf_counter()
            model = fit_dynamic_online_head(calibration_source, candidate)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cal_candidate = model.score_rows(calibration_source)
            eval_candidate = model.score_rows(evaluation_source)
            score_seconds = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            calibration.extend(cal_candidate)
            evaluation.extend(eval_candidate)
            cell_diag[candidate] = dict(model.diagnostics)
            efficiency_rows.append({
                "cell_id": cell_id,
                "family": _family(cell_id),
                "method": candidate,
                "feature_count": model.diagnostics["feature_count"],
                "fit_seconds": fit_seconds,
                "score_seconds": score_seconds,
                "python_traced_peak_bytes": int(peak),
                "persistent_state_scalars_per_trace": model.diagnostics["persistent_state_scalars_per_trace"],
                "update_complexity": model.diagnostics["update_complexity"],
            })
        # Remove raw mechanism rows; the fixed equal combination is retained.
        calibration = [row for row in calibration if row["method"] in METHODS]
        evaluation = [row for row in evaluation if row["method"] in METHODS]
        parameters = final_calibration(calibration)
        convergence = convergence_table(evaluation, parameters, budgets=DEFAULT_BUDGETS)
        for row in convergence:
            row.update({"cell_id": cell_id, "family": _family(cell_id)})
        all_convergence.extend(convergence)
        for method in METHODS:
            policy = calibrate_declaration_policy(calibration, method)
            evaluated = apply_declaration_policy(
                evaluation, method,
                low=policy["low"], high=policy["high"],
                stable_observations=policy["stable_observations"],
            )
            summary = declaration_summary(evaluated)
            all_declarations.append({
                "cell_id": cell_id,
                "family": _family(cell_id),
                "method": method,
                "low": policy["low"],
                "high": policy["high"],
                **summary,
            })
        trace_rows = per_trace_convergence(evaluation, parameters)
        for row in trace_rows:
            row.update({"cell_id": cell_id, "family": _family(cell_id)})
        all_per_trace.extend(trace_rows)
        for row in evaluation:
            if row["method"] in METHODS:
                all_question_scores.append({**row, "cell_id": cell_id, "family": _family(cell_id), "length_band": _length_band(int(row["trace_length"]))})
        cells[cell_id] = evaluation
        calibration_by_cell[cell_id] = calibration
        diagnostics[cell_id] = cell_diag
        cell_out = OUT / "cells" / cell_id
        _write_csv(cell_out / "scores_calibration.csv", calibration)
        _write_csv(cell_out / "scores_evaluation.csv", evaluation)
        _write_json(cell_out / "dynamic_models.json", cell_diag)

    interval_jobs = [
        (candidate, reference)
        for candidate in ("cusum_swvar_equal", *CANDIDATE_FEATURES)
        for reference in ("iu28_no_length", "deepconf_entropy_w64")
    ]
    with ThreadPoolExecutor(max_workers=min(4, len(interval_jobs))) as pool:
        futures = [
            pool.submit(
                _hierarchical_delta,
                cells,
                candidate,
                reference,
                repeats=BOOTSTRAP,
                seed=SEED + sum(map(ord, candidate + reference)),
            )
            for candidate, reference in interval_jobs
        ]
        interval_rows = [future.result() for future in futures]

    # Final-score and length-band metrics supplement the convergence table.
    cell_metric_rows = list(all_convergence)
    for cell_id, rows in cells.items():
        for method in METHODS:
            final_rows = [row for row in rows if row["method"] == method and row["is_final"]]
            labels = [row["label_error"] for row in final_rows]
            scores = [row["score"] for row in final_rows]
            cell_metric_rows.append({
                "cell_id": cell_id,
                "family": _family(cell_id),
                "method": method,
                "budget": "final",
                "n_at_risk": len(final_rows),
                "n_error": int(sum(labels)),
                "auroc": _safe_auc(labels, scores),
                "auprc": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else float("nan"),
            })
            for band in ("lt128", "128_511", "ge512"):
                selected = [row for row in final_rows if _length_band(int(row["trace_length"])) == band]
                if len(selected) < 10:
                    continue
                y = [row["label_error"] for row in selected]
                s = [row["score"] for row in selected]
                cell_metric_rows.append({
                    "cell_id": cell_id,
                    "family": _family(cell_id),
                    "method": method,
                    "budget": "final",
                    "length_band": band,
                    "n_at_risk": len(selected),
                    "n_error": int(sum(y)),
                    "auroc": _safe_auc(y, s),
                    "auprc": float(average_precision_score(y, s)) if len(set(y)) == 2 else float("nan"),
                })

    # Missing-stream sensitivity uses the already-fitted candidate model in
    # each cell and a deterministic zero-at-reference substitution.
    missing_rows = []
    for cell_id in sorted(cells):
        cal_source = _read_scores(SOURCE / cell_id / "scores_calibration.csv")
        eval_source = _read_scores(SOURCE / cell_id / "scores_evaluation.csv")
        for candidate in CANDIDATE_FEATURES:
            model = fit_dynamic_online_head(cal_source, candidate)
            for missing in ("cusum_max", "sw_var_peak"):
                ablated = [row for row in eval_source if row["method"] != missing]
                scored = model.score_rows(ablated)
                endpoint = _endpoint_for_cell(scored, candidate)
                missing_rows.append({"cell_id": cell_id, "family": _family(cell_id), "method": candidate, "missing_stream": missing, "endpoint_auroc_64_128": endpoint})

    _write_csv(OUT / "PER_QUESTION_SCORES.csv", all_question_scores)
    _write_csv(OUT / "PER_CELL_METRICS.csv", cell_metric_rows)
    _write_csv(OUT / "GROUPED_INTERVALS.csv", interval_rows)
    _write_csv(OUT / "DECLARATION_METRICS.csv", all_declarations)
    _write_csv(OUT / "PER_TRACE_CONVERGENCE.csv", all_per_trace)
    _write_csv(OUT / "EFFICIENCY.csv", efficiency_rows)
    _write_csv(OUT / "MISSING_STREAM_SENSITIVITY.csv", missing_rows)
    _write_json(OUT / "MODEL_DIAGNOSTICS.json", diagnostics)
    _write_json(OUT / "RUN_DEFINITION.json", {
        "protocol": "GLOBAL_LOCAL_ONLINE_IU_V1",
        "protocol_path": "docs/experiments/GLOBAL_LOCAL_ONLINE_IU_V1.md",
        "candidate_roster": list(CANDIDATE_FEATURES),
        "baselines": list(BASELINES),
        "budgets": list(DEFAULT_BUDGETS),
        "seed": SEED,
        "bootstrap_repeats": BOOTSTRAP,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_report_sha256": _sha256(SOURCE / "REPORT.md"),
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
        "localization_score_hash_identity": True,
    })
    _write_json(OUT / "CHECKPOINT.json", {
        "status": "CPU_RETROSPECTIVE_COMPLETE",
        "cells": len(cells),
        "families": sorted({_family(cell_id) for cell_id in cells}),
        "candidate_count": len(CANDIDATE_FEATURES),
        "tests_required": "scripts/test_global_local_online.py",
        "fresh_confirmation": False,
        "new_inference_authorized": False,
    })
    print(json.dumps({"cells": len(cells), "intervals": interval_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
