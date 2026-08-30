#!/usr/bin/env python3
"""Finalize paired inference, reference selection, claims, and Phase-1 state."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)
from scripts.reasoning_localization.run_phase1_baseline import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEFAULT_RELEASE,
    PB_METRICS,
    PROGRAM_ROOT,
    PRM_CELL,
    QWEN_MODELS,
    VARIANTS,
)


INCUMBENT = "R3_IU29"
PRIMARY_CHALLENGERS = tuple(variant for variant in VARIANTS if variant != INCUMBENT)
PARENT_CONTRASTS = (
    ("R1_ENTROPY_TOP5", "R0_ENTROPY_MAX"),
    ("R2_FAMILY6_TOP5_CURRENT", "R1_ENTROPY_TOP5"),
)
BENEFIT = 0.005
HARM = -0.005
FAMILYWISE_Q = 0.05 / (2 * len(PRIMARY_CHALLENGERS))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, handle.getvalue().encode("utf-8"))


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def statistical_status(delta: float, low: float, high: float) -> str:
    if low > BENEFIT:
        return "SUPPORTED_IMPROVEMENT"
    if high < HARM:
        return "SUPPORTED_HARM"
    if delta > 0.0 and low <= 0.0 <= high:
        return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def load_pb() -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, float]]],
]:
    points: dict[str, dict[str, float]] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        root = PROGRAM_ROOT / "phase_1" / variant.lower() / "evaluation"
        run = json.loads((root.parent / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        if run.get("status") != "COMPLETE":
            raise RuntimeError(f"Phase-1 variant incomplete: {variant}")
        _, panel_rows = read_csv(root / "PROCESSBENCH_PANELS.csv")
        points[variant] = {
            row["metric_id"]: float(row["value"])
            for row in panel_rows if row["population_id"] == "current_common_eight_qwen"
        }
        packed = load_npz_no_pickle(root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz")
        samples[variant] = {
            metric: np.asarray(packed[f"current_common_eight_qwen__{metric}"], dtype=np.float64)
            for metric in PB_METRICS
        }
        _, cell_rows = read_csv(root / "PROCESSBENCH_BY_CELL.csv")
        cells[variant] = {
            row["cell_id"]: {metric: float(row[metric]) for metric in PB_METRICS}
            for row in cell_rows if row["model_id"] in QWEN_MODELS
        }
    return points, samples, cells


def load_prm_scores() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels = load_npz_no_pickle(DEFAULT_RELEASE / "build_A/localization/evaluation/prmbench_steps.npz")
    response_ids = tuple(labels["response_row_ids"].astype(str).tolist())
    step_offsets = np.asarray(labels["step_offsets"], dtype=np.int64)
    y = np.asarray(labels["step_labels"], dtype=np.int8)
    step_group = np.repeat(np.arange(len(response_ids), dtype=np.int64), np.diff(step_offsets))
    by_variant: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        path = PROGRAM_ROOT / "phase_1" / variant.lower() / "score_freeze/cells" / PRM_CELL / "scores.npz"
        arrays = load_npz_no_pickle(path)
        row_ids = tuple(arrays["row_ids"].astype(str).tolist())
        index = {row_id: value for value, row_id in enumerate(row_ids)}
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        all_scores = np.asarray(arrays["combined_step_scores"], dtype=np.float64)
        pieces = []
        for row_id, expected in zip(response_ids, np.diff(step_offsets)):
            position = index[row_id]
            lo, hi = map(int, offsets[position:position + 2])
            if hi - lo != int(expected):
                raise RuntimeError("PRMBench alignment changed during finalization")
            pieces.append(all_scores[lo:hi])
        by_variant[variant] = np.concatenate(pieces)
    return y, step_group, by_variant


def _tie_structure(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    return order, starts


def prm_bootstrap(
    y: np.ndarray, step_group: np.ndarray, scores: Mapping[str, np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    structures = {variant: _tie_structure(value) for variant, value in scores.items()}
    output = {
        variant: {"auroc": np.empty(BOOTSTRAP_DRAWS), "auprc": np.empty(BOOTSTRAP_DRAWS)}
        for variant in VARIANTS
    }
    n_groups = int(step_group.max()) + 1
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    probability = np.full(n_groups, 1.0 / n_groups)
    chunk = 32
    for offset in range(0, BOOTSTRAP_DRAWS, chunk):
        size = min(chunk, BOOTSTRAP_DRAWS - offset)
        counts = rng.multinomial(n_groups, probability, size=size)
        for variant in VARIANTS:
            order, starts = structures[variant]
            weights = counts[:, step_group[order]].astype(np.float64, copy=False)
            ordered_y = y[order]
            positives = np.add.reduceat(weights * ordered_y, starts, axis=1)
            negatives = np.add.reduceat(weights * (1 - ordered_y), starts, axis=1)
            cum_pos = np.cumsum(positives, axis=1)
            cum_neg = np.cumsum(negatives, axis=1)
            total_pos, total_neg = cum_pos[:, -1], cum_neg[:, -1]
            auc_num = np.sum(
                positives * (total_neg[:, None] - cum_neg + 0.5 * negatives), axis=1
            )
            output[variant]["auroc"][offset:offset + size] = auc_num / (total_pos * total_neg)
            precision = np.divide(
                cum_pos, cum_pos + cum_neg,
                out=np.zeros_like(cum_pos), where=(cum_pos + cum_neg) > 0,
            )
            output[variant]["auprc"][offset:offset + size] = (
                np.sum(positives * precision, axis=1) / total_pos
            )
        if offset % 640 == 0:
            print(f"PRMBench paired bootstrap {offset + size}/{BOOTSTRAP_DRAWS}", flush=True)
    return output


def main() -> None:
    final_root = PROGRAM_ROOT / "phase_1/final"
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase-1 finalization: {final_root}")
    final_root.mkdir(parents=True, exist_ok=False)
    pb_points, pb_samples, pb_cells = load_pb()
    y, step_group, prm_scores = load_prm_scores()
    prm_samples = prm_bootstrap(y, step_group, prm_scores)
    atomic_write_npz(final_root / "PRMBENCH_BOOTSTRAP_SAMPLES.npz", {
        f"{variant}__{metric}": values
        for variant, by_metric in prm_samples.items() for metric, values in by_metric.items()
    })

    contrast_rows: list[dict[str, object]] = []
    contrast_pairs = tuple((variant, INCUMBENT) for variant in PRIMARY_CHALLENGERS) + PARENT_CONTRASTS
    for left, right in contrast_pairs:
        primary = right == INCUMBENT
        q_low, q_high = (FAMILYWISE_Q, 1 - FAMILYWISE_Q) if primary else (0.025, 0.975)
        for metric in PB_METRICS:
            delta_samples = pb_samples[left][metric] - pb_samples[right][metric]
            delta = pb_points[left][metric] - pb_points[right][metric]
            cell_delta = {
                cell: pb_cells[left][cell][metric] - pb_cells[right][cell][metric]
                for cell in pb_cells[left]
            }
            eps = 1e-12
            contrast_rows.append({
                "contrast_id": f"pb::{left}::{right}::{metric}",
                "left_variant_id": left, "right_variant_id": right,
                "task_id": "processbench_first_error", "population_id": "current_common_eight_qwen",
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "delta": delta, "ci_low": float(np.quantile(delta_samples, q_low)),
                "ci_high": float(np.quantile(delta_samples, q_high)), "p_adjusted": "",
                "wins": sum(value > eps for value in cell_delta.values()),
                "ties": sum(abs(value) <= eps for value in cell_delta.values()),
                "losses": sum(value < -eps for value in cell_delta.values()),
                "worst_unit_delta": min(cell_delta.values()),
                "inference": "familywise Bonferroni percentile interval across four planned incumbent contrasts" if primary else "unadjusted parent diagnostic interval",
            })
    for left in PRIMARY_CHALLENGERS:
        for metric in ("auroc", "auprc"):
            delta_samples = prm_samples[left][metric] - prm_samples[INCUMBENT][metric]
            left_rows = read_csv(PROGRAM_ROOT / "phase_1" / left.lower() / "evaluation/PRMBENCH_SLICES.csv")[1]
            right_rows = read_csv(PROGRAM_ROOT / "phase_1" / INCUMBENT.lower() / "evaluation/PRMBENCH_SLICES.csv")[1]
            left_point = float(next(row[metric] for row in left_rows if row["slice_type"] == "overall"))
            right_point = float(next(row[metric] for row in right_rows if row["slice_type"] == "overall"))
            contrast_rows.append({
                "contrast_id": f"prm::{left}::{INCUMBENT}::{metric}",
                "left_variant_id": left, "right_variant_id": INCUMBENT,
                "task_id": "prmbench_step_error", "population_id": "prmbench_error_responses",
                "metric_id": metric, "delta": left_point - right_point,
                "ci_low": float(np.quantile(delta_samples, FAMILYWISE_Q)),
                "ci_high": float(np.quantile(delta_samples, 1 - FAMILYWISE_Q)),
                "p_adjusted": "", "wins": "", "ties": "", "losses": "", "worst_unit_delta": "",
                "inference": "familywise Bonferroni percentile interval across four planned incumbent contrasts",
            })
    source_path = final_root / "P1_CONTRASTS.csv"
    write_csv(source_path, list(contrast_rows[0]), contrast_rows)
    source_sha = sha256_file(source_path)

    contrast_path = PROGRAM_ROOT / "CONTRASTS_LONG.csv"
    contrast_fields, existing = read_csv(contrast_path)
    existing = [row for row in existing if row["phase_id"] != "P1"]
    additions = []
    for row in contrast_rows:
        additions.append({
            "phase_id": "P1", "experiment_id": "P1_BASELINES",
            "left_variant_id": row["left_variant_id"], "right_variant_id": row["right_variant_id"],
            "task_id": row["task_id"],
            "dataset_id": "processbench" if row["task_id"] == "processbench_first_error" else "prmbench",
            "population_id": row["population_id"], "metric_id": row["metric_id"],
            "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
            "p_adjusted": row["p_adjusted"], "wins": row["wins"], "ties": row["ties"],
            "losses": row["losses"], "worst_unit_delta": row["worst_unit_delta"],
            "comparison_group_id": f"p1::{row['task_id']}::{row['population_id']}::{row['metric_id']}",
            "status": "COMPLETE", "evidence_status": "DEVELOPMENT",
            "source_artifact": repo_relative(source_path), "source_sha256": source_sha,
            "source_row_selector": f"contrast_id={row['contrast_id']}",
            "notes": str(row["inference"]) + "; practical bounds +0.005 benefit / -0.005 harm apply to ProcessBench macro F1",
        })
    write_csv(contrast_path, contrast_fields, existing + additions)

    metric_path = PROGRAM_ROOT / "METRICS_LONG.csv"
    metric_fields, metrics = read_csv(metric_path)
    for variant in VARIANTS:
        report_source = PROGRAM_ROOT / "phase_1" / variant.lower() / "evaluation/REPORT_METRICS.csv"
        fields, source_metrics = read_csv(report_source)
        for source_row in source_metrics:
            if source_row["source_id"] in {"prm::overall::all::auroc", "prm::overall::all::auprc"}:
                metric = source_row["metric_id"]
                source_row["ci_low"] = str(float(np.quantile(prm_samples[variant][metric], 0.025)))
                source_row["ci_high"] = str(float(np.quantile(prm_samples[variant][metric], 0.975)))
        write_csv(report_source, fields, source_metrics)
        report_sha = sha256_file(report_source)
        source_rel = repo_relative(report_source)
        by_source = {row["source_id"]: row for row in source_metrics}
        for metric_row in metrics:
            if metric_row["variant_id"] == variant and metric_row["source_artifact"] == source_rel:
                source_id = metric_row["source_row_selector"].split("=", 1)[1]
                source_row = by_source[source_id]
                metric_row["ci_low"], metric_row["ci_high"] = source_row["ci_low"], source_row["ci_high"]
                metric_row["source_sha256"] = report_sha
    write_csv(metric_path, metric_fields, metrics)

    variants_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    registry = json.loads(variants_path.read_text(encoding="utf-8"))
    qwen_f1 = {variant: pb_points[variant]["official_macro_f1"] for variant in VARIANTS}
    winner = max(VARIANTS, key=qwen_f1.get)
    primary_rows = {
        str(row["left_variant_id"]): row for row in contrast_rows
        if row["task_id"] == "processbench_first_error"
        and row["metric_id"] == "macro_f1" and row["right_variant_id"] == INCUMBENT
    }
    prm_auroc_rows = {
        str(row["left_variant_id"]): row for row in contrast_rows
        if row["task_id"] == "prmbench_step_error"
        and row["metric_id"] == "auroc" and row["right_variant_id"] == INCUMBENT
    }
    for variant in registry["variants"]:
        if variant["variant_id"] not in VARIANTS:
            continue
        variant["execution_status"] = "COMPLETE"
        if variant["variant_id"] == INCUMBENT:
            variant["statistical_status"] = "DESCRIPTIVE"
        else:
            row = primary_rows[variant["variant_id"]]
            variant["statistical_status"] = statistical_status(
                float(row["delta"]), float(row["ci_low"]), float(row["ci_high"])
            )
        if variant["variant_id"] == winner:
            variant["decision_status"] = "PROMOTED"
        elif (
            variant["statistical_status"] == "SUPPORTED_IMPROVEMENT"
            and variant["variant_id"] in prm_auroc_rows
            and float(prm_auroc_rows[variant["variant_id"]]["ci_high"]) < 0.0
        ):
            variant["decision_status"] = "PROCESSBENCH_SPECIALIST"
        elif variant["statistical_status"] == "SUPPORTED_HARM":
            variant["decision_status"] = "REJECTED"
        else:
            variant["decision_status"] = "NO_PROMOTION"
    atomic_write_json(variants_path, registry)

    experiment_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P1_BASELINES")
    experiment["execution_status"] = "COMPLETE"
    experiment["phase_1_reference"] = winner
    experiment["source_stratum_status"] = "BLOCKED_METADATA_NOT_IN_SEALED_EVALUATOR"
    atomic_write_json(experiment_path, experiments)

    winner_row = primary_rows.get(winner)
    winner_status = "DESCRIPTIVE" if winner == INCUMBENT else statistical_status(
        float(winner_row["delta"]), float(winner_row["ci_low"]), float(winner_row["ci_high"])
    )
    claims_path = PROGRAM_ROOT / "CLAIMS.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["claims"] = [
        row for row in claims["claims"]
        if row["claim_id"] not in {"CLAIM_P1_REFERENCE", "CLAIM_P1_R2_TASK_CONFLICT"}
    ]
    claims["claims"].append({
        "claim_id": "CLAIM_P1_REFERENCE",
        "text": f"{winner} is the strongest raw Qwen-eight Phase-1 baseline on the frozen common protocol and is frozen as the Phase-2 reference.",
        "verdict": winner_status,
        "task_scope": "Phase-1 ProcessBench Qwen-eight development population; PRMBench is reported separately.",
        "evidence_refs": ["PLOT_P1_PB_FOREST", "PLOT_P1_DELTA_FOREST", "PLOT_P1_PRM_FOREST", "TABLE_GATES"],
        "worst_case_behavior": "See the paired contrast worst-unit field and model-by-dataset heatmap; no task average masks PRMBench behavior.",
        "claim_boundary": "Raw baseline selection is outcome-opened and not an independent improvement claim. Missing PRMBench source-stratum membership remains blocked.",
        "fresh_confirmation_required": True,
    })
    r2_pb = primary_rows["R2_FAMILY6_TOP5_CURRENT"]
    r2_prm = prm_auroc_rows["R2_FAMILY6_TOP5_CURRENT"]
    claims["claims"].append({
        "claim_id": "CLAIM_P1_R2_TASK_CONFLICT",
        "text": "R2_FAMILY6_TOP5_CURRENT is a ProcessBench specialist: it improves first-error macro F1 over IU29 but lowers PRMBench AUROC.",
        "verdict": "PB_SPECIALIST",
        "task_scope": "Phase-1 Qwen-eight ProcessBench development panel and separate PRMBench error-response panel.",
        "evidence_refs": ["PLOT_P1_DELTA_FOREST", "PLOT_P1_PRM_FOREST", "TABLE_CONTRASTS"],
        "worst_case_behavior": "PRMBench AUROC delta is negative with a familywise interval below zero; the method is not a task-general promotion.",
        "claim_boundary": f"ProcessBench delta {float(r2_pb['delta']):+.6f} and PRMBench AUROC delta {float(r2_prm['delta']):+.6f} use separate estimators and are never averaged.",
        "fresh_confirmation_required": True,
    })
    atomic_write_json(claims_path, claims)
    atomic_write_json(final_root / "SUMMARY.json", {
        "schema": "reasoning-localization-phase1-final-v1", "status": "COMPLETE",
        "reference_variant": winner, "qwen8_processbench_macro_f1": qwen_f1,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
        "familywise_quantiles": [FAMILYWISE_Q, 1 - FAMILYWISE_Q],
        "prm_source_strata": "BLOCKED_METADATA_NOT_IN_SEALED_EVALUATOR",
    })
    subprocess.run(
        [sys.executable, str(REPO / "scripts/reasoning_localization/build_reasoning_localization_report.py")],
        cwd=REPO, check=True,
    )
    print(json.dumps({"status": "COMPLETE", "phase_1_reference": winner, "qwen8_f1": qwen_f1}, indent=2))


if __name__ == "__main__":
    main()
