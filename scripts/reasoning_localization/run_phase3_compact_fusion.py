#!/usr/bin/env python3
"""Freeze then evaluate ordinary outer IU-PCR over the four H2 families."""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell, validate_fit_manifest  # noqa: E402
from spectral_utils.token_local_fusion import IU_CONFIG, fit_local_equal_family, prepare_localization_cell  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa: E402
from scripts.reasoning_localization import run_phase2_conditional as conditional  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization.register_phase3_compact_fusion import CANDIDATE, EXPERIMENT, FAMILY_SIZE, PARENT  # noqa: E402

ROOT = p1.PROGRAM_ROOT / "phase_3/compact_outer_iu"
OUTPUT = ROOT / f"{CANDIDATE.lower()}_v2"
REGISTRY = ROOT / "P3B_H2_OUTER_IU_EXECUTION_REGISTRY_AMENDMENT_V2.json"
SOURCE_H2 = p1.PROGRAM_ROOT / "phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"


class Phase3Error(RuntimeError):
    pass


def _h2_family_matrix(cell: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prep = prepare_localization_cell(cell)
    level = conditional._family_risk(prep, "entropy_level")
    old = conditional._family_risk(prep, "entropy_dynamics")
    n_old = sum(f == "entropy_dynamics" for f in prep.kept_family_names)
    entropy = atomic.primitive_risks(cell)["entropy"]
    onset = atomic.response_map(entropy, cell.token_offsets, atomic.edis_onset)
    c7 = conditional._standardized_risk(onset, prep.fit_indices)
    dynamics = (n_old * old + c7) / (n_old + 1)
    partition = conditional._family_risk(prep, "partition_energy", "energy_series")
    topk = conditional._family_risk(prep, "topk_distribution")
    risks = np.column_stack([level, dynamics, partition, topk])
    parent = np.mean(risks, axis=1)
    return risks, parent, {"n_original_dynamics": n_old, "family_names": [
        "entropy_level", "entropy_dynamics_plus_C7", "partition_energy_without_energy_series", "topk_distribution"
    ], "preparation_sha256": prep.diagnostics["preparation_sha256"]}


def _outer_iu(risks: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    fit = np.asarray(risks, dtype=np.float64)[np.asarray(fit_indices, dtype=np.int64)]
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 1e-8):
        raise Phase3Error("degenerate H2 family score")
    confidence_fit = -(fit - mean[None, :]) / std[None, :]
    fitted = upcr_fit(confidence_fit.T, **dict(IU_CONFIG))
    weights = np.asarray(fitted.w, dtype=np.float64).copy()
    anchor = confidence_fit.mean(axis=1)
    score = confidence_fit @ weights
    corr = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(corr) and corr < 0)
    if flipped:
        weights *= -1.0
    confidence_all = -(np.asarray(risks, dtype=np.float64) - mean[None, :]) / std[None, :]
    candidate = -(confidence_all @ weights)
    if not np.isfinite(candidate).all():
        raise Phase3Error("non-finite outer-IU curve")
    return candidate, {"weights": weights.tolist(), "confidence_anchor_correlation": corr,
        "orientation_flipped": flipped, "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual), "n_fit_tokens": int(len(fit))}


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text())
    required = {"schema": "reasoning-localization-p3-compact-fusion-execution-v2",
        "status": "FROZEN_BEFORE_RUN", "experiment_id": EXPERIMENT, "variant_id": CANDIDATE,
        "parent_variant_id": PARENT, "runner_sha256": sha256_file(Path(__file__).resolve())}
    for key, value in required.items():
        if row.get(key) != value:
            raise Phase3Error(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise Phase3Error("release mismatch")
    return row


def freeze(release: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    score_root = OUTPUT / "score_freeze"
    score_root.mkdir(parents=True)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records, alias = [], 0.0
    for pos, cell_id in enumerate(p2r.PB_CELLS, 1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep = prepare_localization_cell(cell)
        risks, parent_token, diagnostics = _h2_family_matrix(cell)
        candidate_token, iu_diag = _outer_iu(risks, prep.fit_indices)
        h0_token = np.asarray(fit_local_equal_family(prep).token_risk, dtype=np.float64)
        parent_local = p1.topk_step_mean(parent_token, cell.segment_starts, cell.segment_ends, k=10)
        candidate_local = p1.topk_step_mean(candidate_token, cell.segment_starts, cell.segment_ends, k=10)
        h0_local = p1.topk_step_mean(h0_token, cell.segment_starts, cell.segment_ends, k=10)
        frozen_h2 = load_npz_no_pickle(SOURCE_H2 / cell_id / "scores.npz")["h2_local"]
        alias = max(alias, float(np.max(np.abs(parent_local - frozen_h2))))
        arrays = {"row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends-cell.segment_starts, dtype="<i8"),
            "h0_combined": p1.combine_with_common_detector(cell, h0_local),
            "parent_local": parent_local, "candidate_local": candidate_local}
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {"schema": "reasoning-localization-p3-compact-cell-v1", "experiment_id": EXPERIMENT,
            "variant_id": CANDIDATE, "cell_id": cell_id, "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id), "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids), "n_steps": len(candidate_local), "score_sha256": score_sha,
            "prepared_input": str(input_path), "prepared_input_sha256": sha256_file(input_path),
            "labels_seen_during_fit": False, "targets_accessed_during_fit": False,
            "diagnostics": {**diagnostics, "outer_iu": iu_diag}}
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json",
            "record_sha256": sha256_file(target/"RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze {CANDIDATE}: {cell_id} ({pos}/8)", flush=True)
    if alias > 1e-12:
        raise Phase3Error(f"H2 parent alias failed: {alias}")
    result = {"schema": "reasoning-localization-p3-compact-score-freeze-v1", "status": "COMPLETE",
        "experiment_id": EXPERIMENT, "variant_id": CANDIDATE, "parent_alias_max_abs_error": alias,
        "cells": list(p2r.PB_CELLS), "records": records, "execution_registry_sha256": sha256_file(REGISTRY),
        "runner_sha256": sha256_file(Path(__file__).resolve()), "labels_seen_during_fit": False,
        "detector_contract": "H0 threshold and abstention decision copied exactly; P3 arms rerank H0 non-abstentions only"}
    result["payload_sha256"] = c1.payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for item in manifest["records"]:
        rp = OUTPUT / "score_freeze" / item["record_path"]
        if sha256_file(rp) != item["record_sha256"]:
            raise Phase3Error("record hash mismatch")
        rec = json.loads(rp.read_text()); sp = rp.parent / "scores.npz"
        if sha256_file(sp) != item["score_sha256"]:
            raise Phase3Error("score hash mismatch")
        result[item["cell_id"]] = {"record": rec, "arrays": load_npz_no_pickle(sp)}
    return result


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    result = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        rec, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        offsets, lengths = arrays["segment_offsets"], arrays["segment_lengths"]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, offsets[index:index+2]); group_id, first_error = labels[cell_id][row_id]
            result[rec["model_id"]].append({"row_id": row_id, "group_id": group_id, "slice_id": rec["slice_id"],
                "cell_id": cell_id, "model_id": rec["model_id"], "first_error": first_error,
                "step_scores": arrays[key][lo:hi].tolist(), "step_lengths": lengths[lo:hi].tolist()})
    return result


def _rerank(arm: str, h0: Mapping[str, Any], rows: Mapping[str, list[dict[str, Any]]], evaluation: Any) -> dict[str, Any]:
    scores = {(row["cell_id"], row["row_id"]): row for model_rows in rows.values() for row in model_rows}
    decisions = []
    for parent in h0["decisions"]:
        source = scores[(parent["cell_id"], parent["row_id"])]
        prediction = -1 if int(parent["prediction_step"]) == -1 else int(np.argmax(source["step_scores"]))
        decisions.append({**parent, "arm_id": arm, "prediction_step": prediction})
    by_cell = []
    for model in p1.QWEN_MODELS:
        for family in p1.FAMILIES:
            selected = [row for row in decisions if row["model_id"] == model and row["slice_id"] == family]
            metrics = evaluation.processbench_trace_metrics(
                [row["true_first_error"] for row in selected], [row["prediction_step"] for row in selected])
            by_cell.append({"arm_id": arm, "model_id": model, "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{metric: metrics[metric] for metric in p1.PB_METRICS}, "n_examples": metrics["n_examples"],
                "n_error": metrics["n_error"], "n_clean": metrics["n_clean"]})
    samples = p1._bootstrap_pb_panel(decisions, p1.QWEN_MODELS)
    panels = [{"arm_id": arm, "population_id": "current_common_eight_qwen", "metric_id": metric,
        "value": float(np.mean([row[metric] for row in by_cell])),
        "ci_low": float(np.quantile(samples[metric], .025)), "ci_high": float(np.quantile(samples[metric], .975)),
        "n_rows": 6800, "n_groups": 3400} for metric in p1.PB_METRICS]
    return {"decisions": decisions, "by_cell": by_cell, "samples": samples, "panels": panels}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)  # Explicitly after the score freeze and hash verification.
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0 = c1.evaluate_arm("P3_H0_REFERENCE", _rows(verified, labels, "h0_combined"), evaluator)
    arms = {"P3_H0_REFERENCE": h0}
    arms[PARENT] = _rerank(PARENT, h0, _rows(verified, labels, "parent_local"), evaluator)
    arms[CANDIDATE] = _rerank(CANDIDATE, h0, _rows(verified, labels, "candidate_local"), evaluator)
    h0_abstain = {(row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1 for row in h0["decisions"]}
    mismatches = {arm: sum((int(row["prediction_step"]) == -1) != h0_abstain[(row["cell_id"], row["row_id"])]
        for row in arms[arm]["decisions"]) for arm in (PARENT, CANDIDATE)}
    if any(mismatches.values()):
        raise Phase3Error(f"H0 abstention alias failed: {mismatches}")
    contrasts = []
    for left, right in ((CANDIDATE, PARENT), (CANDIDATE, "P3_H0_REFERENCE"), (PARENT, "P3_H0_REFERENCE")):
        lp = {row["metric_id"]: row for row in arms[left]["panels"]}
        rp = {row["metric_id"]: row for row in arms[right]["panels"]}
        right_cells = {row["cell_id"]: row for row in arms[right]["by_cell"]}
        for metric in p1.PB_METRICS:
            draws = np.asarray(arms[left]["samples"][metric]) - np.asarray(arms[right]["samples"][metric])
            q = .025/FAMILY_SIZE if metric == "official_macro_f1" else .025
            cells = {row["cell_id"]: float(row[metric])-float(right_cells[row["cell_id"]][metric]) for row in arms[left]["by_cell"]}
            contrasts.append({"contrast_id": f"pb::{left}::{right}::{metric}", "left_variant_id": left,
                "right_variant_id": right, "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "delta": float(lp[metric]["value"]-rp[metric]["value"]), "ci_low": float(np.quantile(draws,q)),
                "ci_high": float(np.quantile(draws,1-q)), "wins": sum(v>1e-12 for v in cells.values()),
                "ties": sum(abs(v)<=1e-12 for v in cells.values()), "losses": sum(v<-1e-12 for v in cells.values()),
                "worst_unit_delta": min(cells.values()), "worst_unit_id": min(cells,key=cells.get),
                "multiplicity_family_size": FAMILY_SIZE if metric == "official_macro_f1" else 1})
    eval_root = OUTPUT / "evaluation"; eval_root.mkdir()
    _write_csv(eval_root/"PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    _write_csv(eval_root/"PROCESSBENCH_PANELS.csv", [row for arm in arms.values() for row in arm["panels"]])
    _write_csv(eval_root/"PAIRWISE_CONTRASTS.csv", contrasts)
    primary = next(row for row in contrasts if row["left_variant_id"]==CANDIDATE and row["right_variant_id"]==PARENT and row["metric_id"]=="macro_f1")
    vs_h0 = next(row for row in contrasts if row["left_variant_id"]==CANDIDATE and row["right_variant_id"]=="P3_H0_REFERENCE" and row["metric_id"]=="macro_f1")
    summary = {"schema": "reasoning-localization-p3-compact-evaluation-v1", "status": "COMPLETE",
        "experiment_id": EXPERIMENT, "variant_id": CANDIDATE, "primary_contrast": primary,
        "candidate_vs_h0": vs_h0, "abstention_mismatches": mismatches,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED}
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(eval_root/"SUMMARY.json", summary)
    return summary


def main() -> None:
    started = time.perf_counter(); release = p1.DEFAULT_RELEASE.resolve(); registry = _load_registry(release)
    frozen = freeze(release, registry); summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT/"RUN_COMPLETE.json", {"schema": "reasoning-localization-p3-compact-run-v1",
        "status": "COMPLETE", "experiment_id": EXPERIMENT, "variant_id": CANDIDATE,
        "elapsed_seconds": time.perf_counter()-started, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
