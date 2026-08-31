#!/usr/bin/env python3
"""Freeze and evaluate exact H3-equal on the four Llama scorer cells."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import defaultdict
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
from spectral_utils.token_local_fusion import prepare_localization_cell  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as p2c  # noqa: E402

EXPERIMENT = "P2_H3_LLAMA_TRANSFER"
H0 = "P2E_H0_FAMILY6_TOP10_LLAMA4"
H2 = "P2E_H2_CLEAN_C7_LLAMA4"
H3 = "P2E_H3_EQUAL_C8_RERANK_LLAMA4"
ARMS = (H0, H2, H3)
MODEL = "llama31_8b"
CELLS = tuple(f"processbench_{family}_{MODEL}" for family in p1.FAMILIES)
ROOT = p1.PROGRAM_ROOT / "phase_2" / "transfer" / "h3_llama4"
REGISTRY = ROOT.parent / "H3_LLAMA_TRANSFER_EXECUTION_REGISTRY.json"
SOURCE_H3 = p1.PROGRAM_ROOT / "phase_2" / "diagnostic" / "h3_reliability_fusion_v1"
PRIMARY_FAMILY = 3


class TransferError(RuntimeError):
    pass


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise TransferError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1.0)
    return result


def score_cell(
    cell: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    prep = prepare_localization_cell(cell)
    h0_token = p2c.fit_local_equal_family(prep).token_risk
    entropy = atomic.primitive_risks(cell)["entropy"]
    onset = atomic.response_map(entropy, cell.token_offsets, atomic.edis_onset)
    inserted = p2c._standardized_risk(onset, prep.fit_indices)
    old_dynamics = p2c._family_risk(prep, "entropy_dynamics")
    n_dynamics = sum(family == "entropy_dynamics" for family in prep.kept_family_names)
    h2_dynamics = (n_dynamics * old_dynamics + inserted) / (n_dynamics + 1)
    h2_token = np.mean(
        [
            p2c._family_risk(prep, "entropy_level"),
            h2_dynamics,
            p2c._family_risk(prep, "partition_energy", "energy_series"),
            p2c._family_risk(prep, "topk_distribution"),
        ],
        axis=0,
    )
    _iu_parent, c8_token, c8_diag = atomic.fit_self_innovation(cell)
    h0 = p1.topk_step_mean(h0_token, cell.segment_starts, cell.segment_ends, k=10)
    h2 = p1.topk_step_mean(h2_token, cell.segment_starts, cell.segment_ends, k=10)
    c8 = p1.topk_step_mean(c8_token, cell.segment_starts, cell.segment_ends, k=10)
    h3 = np.empty_like(h2)
    for lo, hi in zip(cell.segment_offsets[:-1], cell.segment_offsets[1:]):
        h3[int(lo):int(hi)] = 0.5 * (
            rank01(h2[int(lo):int(hi)]) + rank01(c8[int(lo):int(hi)])
        )
    h0_combined = p1.combine_with_common_detector(cell, h0)
    return h0, h0_combined, h2, h3, {
        "removed_family": "sampled_token_energy",
        "removed_view": "energy_series",
        "c7_placement": "additional equal member inside entropy_dynamics",
        "n_original_entropy_dynamics_members": n_dynamics,
        "h3_fusion": "0.5 within-response H2 rank + 0.5 within-response C8 rank",
        "c8": dict(c8_diag),
    }


def load_contract() -> dict[str, Any]:
    payload = json.loads(REGISTRY.read_text())
    required = {
        "schema": "reasoning-localization-h3-llama-transfer-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "arms": list(ARMS),
        "cells": list(CELLS),
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise TransferError(f"execution registry mismatch: {key}")
    return payload


def qwen_alias(release: Path) -> dict[str, float]:
    input_root = release / "build_A" / "localization" / "inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    errors = {
        "h0_combined_max_abs_error": 0.0,
        "h2_max_abs_error": 0.0,
        "h3_max_abs_error": 0.0,
    }
    source_cells = SOURCE_H3 / "score_freeze" / "cells"
    for cell_id in tuple(f"processbench_{family}_{model}" for model in p1.QWEN_MODELS for family in p1.FAMILIES):
        record = by_cell[cell_id]
        cell = load_prepared_localization_cell(input_root / record["artifact_path"], record)
        _h0_local, h0_combined, h2, h3, _diag = score_cell(cell)
        frozen = load_npz_no_pickle(source_cells / cell_id / "scores.npz")
        errors["h0_combined_max_abs_error"] = max(
            errors["h0_combined_max_abs_error"],
            float(np.max(np.abs(h0_combined - frozen["h0_combined"]))),
        )
        errors["h2_max_abs_error"] = max(errors["h2_max_abs_error"], float(np.max(np.abs(h2 - frozen["h2_local"]))))
        errors["h3_max_abs_error"] = max(errors["h3_max_abs_error"], float(np.max(np.abs(h3 - frozen["h3_equal_local"]))))
    if max(errors.values()) > 1e-12:
        raise TransferError(f"Qwen H2/H3 alias failed: {errors}")
    return errors


def freeze_scores(contract: Mapping[str, Any]) -> dict[str, Any]:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    score_root = ROOT / "score_freeze"
    score_root.mkdir(parents=True)
    release = Path(contract["release_root"])
    alias = qwen_alias(release)
    input_root = release / "build_A" / "localization" / "inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records = []
    for cell_id in CELLS:
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        h0, h0_combined, h2, h3, diagnostics = score_cell(cell)
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "h0_local": np.asarray(h0, dtype="<f8"),
            "h0_combined": np.asarray(h0_combined, dtype="<f8"),
            "h2_local": np.asarray(h2, dtype="<f8"),
            "h3_equal_local": np.asarray(h3, dtype="<f8"),
        }
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {
            "schema": "reasoning-localization-h3-llama-transfer-cell-v1",
            "cell_id": cell_id, "model_id": MODEL, "slice_id": cell.slice_id,
            "population_id": cell.population_id, "n_rows": len(cell.row_ids),
            "n_steps": len(h0), "score_sha256": score_sha,
            "prepared_input": str(input_path), "prepared_input_sha256": sha256_file(input_path),
            "labels_seen": False, "diagnostics": diagnostics,
        }
        atomic_write_json(target / "CELL_MANIFEST.json", record)
        records.append({"cell_id": cell_id, "score_sha256": score_sha, "cell_manifest_sha256": sha256_file(target / "CELL_MANIFEST.json")})
    result = {
        "schema": "reasoning-localization-h3-llama-transfer-score-freeze-v1",
        "status": "FROZEN_BEFORE_LABEL_OPEN", "cells": records,
        "qwen_alias": alias, "labels_seen": False,
        "execution_registry_sha256": sha256_file(REGISTRY),
    }
    atomic_write_json(ROOT / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def rows_from_scores(verified: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    rows = []
    for cell_id in CELLS:
        arrays = verified[cell_id]
        offsets, lengths = arrays["segment_offsets"], arrays["segment_lengths"]
        family = cell_id.split("_")[1]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, offsets[index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            rows.append({"row_id": row_id, "group_id": group_id, "slice_id": family, "cell_id": cell_id,
                         "model_id": MODEL, "first_error": first_error,
                         "step_scores": arrays[key][lo:hi].tolist(), "step_lengths": lengths[lo:hi].tolist()})
    return rows


def h0_evaluation(rows: list[dict[str, Any]], evaluation: Any) -> dict[str, Any]:
    result = evaluation.crossfit_processbench_threshold(rows)
    by_input = {row["row_id"]: row for row in rows}
    decisions = []
    for row in result["decisions"]:
        source = by_input[row["row_id"]]
        pred, target = int(row["prediction_step"]), int(source["first_error"])
        decisions.append({"arm_id": H0, "model_id": MODEL, "cell_id": source["cell_id"], "slice_id": source["slice_id"],
                          "row_id": row["row_id"], "group_id": source["group_id"], "fold": int(row["fold"]),
                          "true_first_error": target, "prediction_step": pred})
    return assemble(H0, decisions, evaluation)


def assemble(arm: str, decisions: list[dict[str, Any]], evaluation: Any) -> dict[str, Any]:
    by_cell = []
    for family in p1.FAMILIES:
        selected = [row for row in decisions if row["slice_id"] == family]
        metrics = evaluation.processbench_trace_metrics(
            [row["true_first_error"] for row in selected], [row["prediction_step"] for row in selected]
        )
        by_cell.append({"arm_id": arm, "model_id": MODEL, "slice_id": family,
                        "cell_id": f"processbench_{family}_{MODEL}",
                        **{metric: metrics[metric] for metric in p1.PB_METRICS},
                        "n_examples": metrics["n_examples"], "n_error": metrics["n_error"], "n_clean": metrics["n_clean"]})
    samples = p1._bootstrap_pb_panel(decisions, (MODEL,))
    panels = [{"arm_id": arm, "population_id": "current_llama4_scorer_transfer", "metric_id": metric,
               "value": float(np.mean([row[metric] for row in by_cell])),
               "ci_low": float(np.quantile(samples[metric], .025)), "ci_high": float(np.quantile(samples[metric], .975)),
               "n_rows": 3400, "n_groups": 3400} for metric in p1.PB_METRICS]
    return {"decisions": decisions, "by_cell": by_cell, "samples": samples, "panels": panels}


def rerank(arm: str, h0: Mapping[str, Any], rows: list[dict[str, Any]], evaluation: Any) -> dict[str, Any]:
    score_by_row = {row["row_id"]: row for row in rows}
    decisions = []
    for parent in h0["decisions"]:
        scores = score_by_row[parent["row_id"]]["step_scores"]
        prediction = -1 if int(parent["prediction_step"]) == -1 else int(np.argmax(scores))
        decisions.append({**parent, "arm_id": arm, "prediction_step": prediction})
    return assemble(arm, decisions, evaluation)


def evaluate_scores(freeze: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    verified = {}
    for item in freeze["cells"]:
        root = ROOT / "score_freeze" / "cells" / item["cell_id"]
        if sha256_file(root / "scores.npz") != item["score_sha256"]:
            raise TransferError("score hash mismatch")
        verified[item["cell_id"]] = load_npz_no_pickle(root / "scores.npz")
    labels = p1._load_pb_labels(Path(contract["release_root"]))
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0_rows = rows_from_scores(verified, labels, "h0_combined")
    h2_rows = rows_from_scores(verified, labels, "h2_local")
    h3_rows = rows_from_scores(verified, labels, "h3_equal_local")
    results = {H0: h0_evaluation(h0_rows, evaluation)}
    results[H2] = rerank(H2, results[H0], h2_rows, evaluation)
    results[H3] = rerank(H3, results[H0], h3_rows, evaluation)
    parent_abstain = {row["row_id"]: int(row["prediction_step"]) == -1 for row in results[H0]["decisions"]}
    mismatches = {arm: sum((int(row["prediction_step"]) == -1) != parent_abstain[row["row_id"]] for row in results[arm]["decisions"]) for arm in (H2, H3)}
    if any(mismatches.values()):
        raise TransferError(f"abstention alias failed: {mismatches}")
    comparisons = ((H2, H0), (H3, H0), (H3, H2))
    contrasts = []
    for left, right in comparisons:
        lp = {row["metric_id"]: row for row in results[left]["panels"]}
        rp = {row["metric_id"]: row for row in results[right]["panels"]}
        rc = {row["cell_id"]: row for row in results[right]["by_cell"]}
        for metric in p1.PB_METRICS:
            draws = results[left]["samples"][metric] - results[right]["samples"][metric]
            q = .025 / PRIMARY_FAMILY if metric == "official_macro_f1" else .025
            cell_delta = {row["cell_id"]: float(row[metric]) - float(rc[row["cell_id"]][metric]) for row in results[left]["by_cell"]}
            contrasts.append({"contrast_id": f"{left}__vs__{right}__{metric}", "left": left, "right": right,
                              "metric": metric, "delta": float(lp[metric]["value"] - rp[metric]["value"]),
                              "ci_low": float(np.quantile(draws, q)), "ci_high": float(np.quantile(draws, 1-q)),
                              "wins": sum(v>1e-12 for v in cell_delta.values()), "ties": sum(abs(v)<=1e-12 for v in cell_delta.values()),
                              "losses": sum(v<-1e-12 for v in cell_delta.values()), "worst_cell_delta": min(cell_delta.values()),
                              "interval": "Bonferroni simultaneous across three primary contrasts" if metric == "official_macro_f1" else "unadjusted paired diagnostic"})
    evaluation_root = ROOT / "evaluation"; evaluation_root.mkdir()
    write_csv(evaluation_root / "PANELS.csv", [row for arm in ARMS for row in results[arm]["panels"]])
    write_csv(evaluation_root / "BY_CELL.csv", [row for arm in ARMS for row in results[arm]["by_cell"]])
    write_csv(evaluation_root / "CONTRASTS.csv", contrasts)
    write_csv(evaluation_root / "DECISIONS.csv", [row for arm in ARMS for row in results[arm]["decisions"]])
    primary = next(row for row in contrasts if row["left"] == H3 and row["right"] == H0 and row["metric"] == "official_macro_f1")
    summary = {"schema": "reasoning-localization-h3-llama-transfer-result-v1", "status": "COMPLETE",
               "evidence_status": "TRANSFER", "fresh_confirmation": False, "abstention_mismatches": mismatches,
               "qwen_alias": freeze["qwen_alias"], "primary_contrast": primary,
               "panels": {arm: {row["metric_id"]: row["value"] for row in results[arm]["panels"]} for arm in ARMS},
               "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED}
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    contract = load_contract()
    freeze = freeze_scores(contract)
    evaluate_scores(freeze, contract)


if __name__ == "__main__":
    main()
