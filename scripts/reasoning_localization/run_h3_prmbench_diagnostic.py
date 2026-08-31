#!/usr/bin/env python3
"""Freeze and evaluate the H0/H2/H3 ladder on PRMBench every-step ranking."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from scripts.reasoning_localization import run_h3_llama_transfer as h3  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402


EXPERIMENT = "P2_H3_PRMBENCH_DIAGNOSTIC"
H0 = "P2F_H0_FAMILY6_TOP10_PRM"
H2 = "P2F_H2_CLEAN_C7_PRM"
H3 = "P2F_H3_EQUAL_C8_RERANK_PRM"
ARMS = (H0, H2, H3)
ROOT = p1.PROGRAM_ROOT / "phase_2/transfer/h3_prmbench"
REGISTRY = ROOT.parent / "H3_PRMBENCH_DIAGNOSTIC_EXECUTION_REGISTRY.json"
PRM_CELL = p1.PRM_CELL
PRIMARY_FAMILY = 3
BENEFIT_BOUND = 0.003
HARM_BOUND = -0.005


class DiagnosticError(RuntimeError):
    pass


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise DiagnosticError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def load_contract() -> dict[str, Any]:
    payload = json.loads(REGISTRY.read_text())
    required = {
        "schema": "reasoning-localization-h3-prmbench-diagnostic-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "arms": list(ARMS),
        "cell_id": PRM_CELL,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise DiagnosticError(f"execution registry mismatch: {key}")
    return payload


def freeze_scores(contract: Mapping[str, Any]) -> dict[str, Any]:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    score_root = ROOT / "score_freeze"
    score_root.mkdir(parents=True)
    release = Path(contract["release_root"])
    qwen_alias = h3.qwen_alias(release)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    records = {str(row["cell_id"]): row for row in manifest["cells"]}
    source = records[PRM_CELL]
    input_path = input_root / source["artifact_path"]
    cell = load_prepared_localization_cell(input_path, source)
    h0_local, h0_combined, h2_local, h3_local, diagnostics = h3.score_cell(cell)
    h2_combined = p1.combine_with_common_detector(cell, h2_local)
    h3_combined = p1.combine_with_common_detector(cell, h3_local)

    phase1_path = (
        p1.PROGRAM_ROOT
        / "phase_1/r2_family6_top5_current/score_freeze/cells"
        / PRM_CELL
        / "scores.npz"
    )
    phase1 = load_npz_no_pickle(phase1_path)
    if tuple(phase1["row_ids"].astype(str)) != tuple(map(str, cell.row_ids)):
        raise DiagnosticError("Phase-1 H0 PRMBench row alias failed")
    if not np.array_equal(phase1["segment_offsets"], cell.segment_offsets):
        raise DiagnosticError("Phase-1 H0 PRMBench segment alias failed")
    phase1_alias = float(
        np.max(np.abs(h0_combined - phase1["combined_step_scores"]))
    )
    if phase1_alias > 1e-12:
        raise DiagnosticError(f"Phase-1 H0 PRMBench score alias failed: {phase1_alias}")

    arrays = {
        "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
        "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
        "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
        "h0_combined": np.asarray(h0_combined, dtype="<f8"),
        "h2_combined": np.asarray(h2_combined, dtype="<f8"),
        "h3_combined": np.asarray(h3_combined, dtype="<f8"),
    }
    target = score_root / "cells" / PRM_CELL
    target.mkdir(parents=True)
    score_sha = atomic_write_npz(target / "scores.npz", arrays)
    cell_manifest = {
        "schema": "reasoning-localization-h3-prmbench-cell-v1",
        "cell_id": PRM_CELL,
        "population_id": str(cell.population_id),
        "model_id": str(cell.model_id),
        "n_rows": len(cell.row_ids),
        "n_steps": len(h0_combined),
        "prepared_input": str(input_path),
        "prepared_input_sha256": sha256_file(input_path),
        "score_sha256": score_sha,
        "labels_seen": False,
        "phase1_h0_score_alias_max_abs_error": phase1_alias,
        "phase1_h0_source": str(phase1_path.relative_to(REPO)),
        "phase1_h0_source_sha256": sha256_file(phase1_path),
        "qwen_processbench_alias": qwen_alias,
        "diagnostics": diagnostics,
    }
    atomic_write_json(target / "CELL_MANIFEST.json", cell_manifest)
    frozen = {
        "schema": "reasoning-localization-h3-prmbench-score-freeze-v1",
        "status": "FROZEN_BEFORE_LABEL_OPEN",
        "labels_seen": False,
        "cell_id": PRM_CELL,
        "score_sha256": score_sha,
        "cell_manifest_sha256": sha256_file(target / "CELL_MANIFEST.json"),
        "phase1_h0_score_alias_max_abs_error": phase1_alias,
        "qwen_processbench_alias": qwen_alias,
        "execution_registry_sha256": sha256_file(REGISTRY),
    }
    atomic_write_json(ROOT / "SCORE_FREEZE_MANIFEST.json", frozen)
    return frozen


def _tie_structure(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    return order, starts


def paired_bootstrap(
    labels: np.ndarray,
    step_groups: np.ndarray,
    scores: Mapping[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    structures = {arm: _tie_structure(values) for arm, values in scores.items()}
    output = {
        arm: {"auroc": np.empty(draws), "auprc": np.empty(draws)} for arm in ARMS
    }
    n_groups = int(step_groups.max()) + 1
    rng = np.random.default_rng(seed)
    probability = np.full(n_groups, 1.0 / n_groups)
    chunk = 32
    for offset in range(0, draws, chunk):
        size = min(chunk, draws - offset)
        counts = rng.multinomial(n_groups, probability, size=size)
        for arm in ARMS:
            order, starts = structures[arm]
            weights = counts[:, step_groups[order]].astype(np.float64, copy=False)
            ordered_y = labels[order]
            positives = np.add.reduceat(weights * ordered_y, starts, axis=1)
            negatives = np.add.reduceat(weights * (1 - ordered_y), starts, axis=1)
            cumulative_positive = np.cumsum(positives, axis=1)
            cumulative_negative = np.cumsum(negatives, axis=1)
            total_positive = cumulative_positive[:, -1]
            total_negative = cumulative_negative[:, -1]
            auc_numerator = np.sum(
                positives
                * (total_negative[:, None] - cumulative_negative + 0.5 * negatives),
                axis=1,
            )
            output[arm]["auroc"][offset : offset + size] = (
                auc_numerator / (total_positive * total_negative)
            )
            precision = np.divide(
                cumulative_positive,
                cumulative_positive + cumulative_negative,
                out=np.zeros_like(cumulative_positive),
                where=(cumulative_positive + cumulative_negative) > 0,
            )
            output[arm]["auprc"][offset : offset + size] = (
                np.sum(positives * precision, axis=1) / total_positive
            )
        if offset % 640 == 0:
            print(f"PRMBench paired bootstrap {offset + size}/{draws}", flush=True)
    return output


def evaluate_scores(freeze: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    score_path = ROOT / "score_freeze/cells" / PRM_CELL / "scores.npz"
    if sha256_file(score_path) != freeze["score_sha256"]:
        raise DiagnosticError("frozen score hash mismatch")
    arrays = load_npz_no_pickle(score_path)
    release = Path(contract["release_root"])
    label_path = release / "build_A/localization/evaluation/prmbench_steps.npz"
    label_arrays = load_npz_no_pickle(label_path)
    response_ids = tuple(label_arrays["response_row_ids"].astype(str))
    score_ids = tuple(arrays["row_ids"].astype(str))
    score_index = {row_id: index for index, row_id in enumerate(score_ids)}
    if len(score_index) != len(score_ids) or not set(response_ids) <= set(score_ids):
        raise DiagnosticError("PRMBench row alignment failed")
    score_offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
    label_offsets = np.asarray(label_arrays["step_offsets"], dtype=np.int64)
    response_group_ids = label_arrays["group_ids"].astype(str)
    families = label_arrays["error_families"].astype(str)
    labels = np.asarray(label_arrays["step_labels"], dtype=np.int8)
    pieces: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    key_by_arm = {H0: "h0_combined", H2: "h2_combined", H3: "h3_combined"}
    for response_index, row_id in enumerate(response_ids):
        index = score_index[row_id]
        lo, hi = map(int, score_offsets[index : index + 2])
        expected = int(label_offsets[response_index + 1] - label_offsets[response_index])
        if hi - lo != expected:
            raise DiagnosticError("PRMBench score/label step count mismatch")
        for arm in ARMS:
            pieces[arm].append(np.asarray(arrays[key_by_arm[arm]][lo:hi], dtype=np.float64))
    scores = {arm: np.concatenate(parts) for arm, parts in pieces.items()}
    if any(values.shape != labels.shape or not np.isfinite(values).all() for values in scores.values()):
        raise DiagnosticError("missing or non-finite PRMBench scores")

    evaluation = importlib.import_module(
        "spectral_utils.reconstruction_benchmark.localization_evaluation"
    )
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    cursor = 0
    for response_index, (row_id, group_id, family) in enumerate(
        zip(response_ids, response_group_ids, families)
    ):
        lo, hi = map(int, label_offsets[response_index : response_index + 2])
        for step_index in range(hi - lo):
            for arm in ARMS:
                rows_by_arm[arm].append(
                    {
                        "arm_id": arm,
                        "group_id": str(group_id),
                        "response_row_id": row_id,
                        "error_family": str(family),
                        "step_index": step_index,
                        "step_label": int(labels[cursor]),
                        "step_score": float(scores[arm][cursor]),
                    }
                )
            cursor += 1
    if cursor != len(labels):
        raise DiagnosticError("PRMBench row construction mismatch")

    panels, family_rows = {}, []
    for arm in ARMS:
        panel = evaluation.prmbench_panel_metrics(rows_by_arm[arm])
        panels[arm] = panel
        for family, metrics in panel["per_family"].items():
            family_rows.append({"arm_id": arm, "error_family": family, **metrics})

    unique_groups = sorted(set(map(str, response_group_ids)))
    group_index = {group_id: index for index, group_id in enumerate(unique_groups)}
    response_step_groups = np.concatenate(
        [
            np.full(
                int(label_offsets[index + 1] - label_offsets[index]),
                group_index[str(group_id)],
                dtype=np.int64,
            )
            for index, group_id in enumerate(response_group_ids)
        ]
    )
    samples = paired_bootstrap(
        labels,
        response_step_groups,
        scores,
        draws=int(contract["bootstrap"]["draws"]),
        seed=int(contract["bootstrap"]["seed"]),
    )
    atomic_write_npz(
        ROOT / "evaluation/BOOTSTRAP_SAMPLES.npz",
        {
            f"{arm}__{metric}": values
            for arm in ARMS
            for metric, values in samples[arm].items()
        },
    )

    panel_rows = []
    for arm in ARMS:
        for metric in ("auroc", "auprc"):
            point = panels[arm]["overall"][metric]
            panel_rows.append(
                {
                    "arm_id": arm,
                    "metric_id": metric,
                    "value": point,
                    "ci_low": float(np.quantile(samples[arm][metric], 0.025)),
                    "ci_high": float(np.quantile(samples[arm][metric], 0.975)),
                    "n_steps": len(labels),
                    "n_positive": int(labels.sum()),
                    "n_negative": int(len(labels) - labels.sum()),
                    "n_responses": len(response_ids),
                    "n_groups": len(unique_groups),
                }
            )

    comparisons = ((H2, H0), (H3, H0), (H3, H2))
    family_lookup = {
        (row["arm_id"], row["error_family"]): row for row in family_rows
    }
    contrasts = []
    for left, right in comparisons:
        for metric in ("auroc", "auprc"):
            delta_samples = samples[left][metric] - samples[right][metric]
            q = 0.025 / PRIMARY_FAMILY
            family_deltas = []
            for family in evaluation.PRMBENCH_ERROR_FAMILIES:
                left_value = family_lookup[(left, family)][metric]
                right_value = family_lookup[(right, family)][metric]
                if left_value is not None and right_value is not None:
                    family_deltas.append((family, float(left_value) - float(right_value)))
            left_point = float(panels[left]["overall"][metric])
            right_point = float(panels[right]["overall"][metric])
            contrasts.append(
                {
                    "contrast_id": f"{left}__vs__{right}__{metric}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "delta": left_point - right_point,
                    "ci_low": float(np.quantile(delta_samples, q)),
                    "ci_high": float(np.quantile(delta_samples, 1 - q)),
                    "wins": sum(value > 1e-12 for _family, value in family_deltas),
                    "ties": sum(abs(value) <= 1e-12 for _family, value in family_deltas),
                    "losses": sum(value < -1e-12 for _family, value in family_deltas),
                    "worst_family": min(family_deltas, key=lambda item: item[1])[0],
                    "worst_family_delta": min(value for _family, value in family_deltas),
                    "interval": "Bonferroni simultaneous across three frozen contrasts for this metric",
                }
            )

    evaluation_root = ROOT / "evaluation"
    write_csv(evaluation_root / "PANELS.csv", panel_rows)
    write_csv(evaluation_root / "BY_FAMILY.csv", family_rows)
    write_csv(evaluation_root / "CONTRASTS.csv", contrasts)
    primary = next(
        row
        for row in contrasts
        if row["left"] == H3 and row["right"] == H0 and row["metric"] == "auroc"
    )
    summary = {
        "schema": "reasoning-localization-h3-prmbench-diagnostic-result-v1",
        "status": "COMPLETE",
        "evidence_status": "DEVELOPMENT",
        "phase4_promotion": False,
        "labels_opened_only_after_score_freeze_in_this_run": True,
        "historically_opened_population": True,
        "phase1_h0_score_alias_max_abs_error": freeze[
            "phase1_h0_score_alias_max_abs_error"
        ],
        "qwen_processbench_alias": freeze["qwen_processbench_alias"],
        "primary_contrast": primary,
        "panels": {
            arm: {
                metric: panels[arm]["overall"][metric]
                for metric in ("auroc", "auprc")
            }
            for arm in ARMS
        },
        "n_evaluable_responses": len(response_ids),
        "n_evaluable_steps": len(labels),
        "n_source_groups": len(unique_groups),
        "all_nine_families_visible": all(
            panels[arm]["all_nine_families_visible"] for arm in ARMS
        ),
        "source_strata_available": False,
        "bootstrap_draws": int(contract["bootstrap"]["draws"]),
        "bootstrap_seed": int(contract["bootstrap"]["seed"]),
    }
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    contract = load_contract()
    freeze = freeze_scores(contract)
    (ROOT / "evaluation").mkdir()
    evaluate_scores(freeze, contract)


if __name__ == "__main__":
    main()
