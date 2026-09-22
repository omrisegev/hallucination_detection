#!/usr/bin/env python3
"""Freeze and evaluate the CIW response + frozen token-IU localization adapter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (
    empirical_midrank,
    load_localization_score_bundle,
)


METHOD = "ciw_deem_response_plus_token_iu29"
PB_CELLS = (
    "processbench_gsm8k_qwen3_4b", "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b", "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b", "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b", "processbench_omnimath_qwen3_8b",
    "processbench_gsm8k_llama31_8b", "processbench_math_llama31_8b",
    "processbench_olympiadbench_llama31_8b", "processbench_omnimath_llama31_8b",
)
PRM_CELL = "prmbench_response_qwen3_8b"
CELLS = PB_CELLS + (PRM_CELL,)
REFERENCES = ("deem_b3", "iu_pcr", "dufs_liu")


def _payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_ciw_cell(ciw_root: Path, cell_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record_path = ciw_root / "cells" / cell_id / "RECORD.json"
    record = json.loads(record_path.read_text())
    body = dict(record)
    digest = body.pop("payload_sha256", None)
    if digest != _payload_sha(body):
        raise RuntimeError(f"{cell_id}: CIW record hash failed")
    score_path = record_path.parent / str(record["score_path"])
    if sha256_file(score_path) != record["score_sha256"]:
        raise RuntimeError(f"{cell_id}: CIW score hash failed")
    return record, load_npz_no_pickle(score_path)


def fit(ciw_root: Path, localization_release: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for cell_id in CELLS:
        ciw_record, ciw = _load_ciw_cell(ciw_root, cell_id)
        loc_record_path = localization_release / "build_A/localization/fit/cells" / cell_id / "RECORD.json"
        loc_record, loc = load_localization_score_bundle(loc_record_path)
        row_ids = np.asarray(loc["row_ids"]).astype(str)
        if not np.array_equal(row_ids, np.asarray(ciw["row_ids"]).astype(str)):
            raise RuntimeError(f"{cell_id}: CIW/localization response rows differ")
        offsets = np.asarray(loc["segment_offsets"], dtype=np.int64)
        response_rank = empirical_midrank(np.asarray(ciw["score"], dtype=np.float64))
        step_rank = empirical_midrank(np.asarray(loc["token_step_score"], dtype=np.float64))
        expanded = np.repeat(response_rank, np.diff(offsets))
        combined = np.sqrt(expanded * step_rank)
        if combined.shape != step_rank.shape or not np.isfinite(combined).all():
            raise RuntimeError(f"{cell_id}: invalid CIW localization scores")
        target = out / "cells" / cell_id
        target.mkdir(parents=True)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, {
            "row_ids": row_ids.astype("<U80"),
            "segment_offsets": offsets.astype("<i8"),
            "step_scores": combined.astype("<f8"),
            "ciw_response_score": np.asarray(ciw["score"], dtype="<f8"),
            "token_step_score": np.asarray(loc["token_step_score"], dtype="<f8"),
        })
        record = {
            "schema_version": "ciw-localization-cell-v1",
            "method_id": METHOD,
            "cell_id": cell_id,
            "model_id": ciw_record["model_id"],
            "slice_id": ciw_record["slice_id"],
            "n_rows": len(row_ids),
            "n_steps": len(combined),
            "formula": "sqrt(midrank(CIW response risk) * midrank(frozen token-IU29 step risk))",
            "response_refit": False,
            "token_refit": False,
            "labels_opened_during_fit": False,
            "targets_accessed_during_fit": False,
            "ciw_record_sha256": sha256_file(ciw_root / "cells" / cell_id / "RECORD.json"),
            "localization_record_sha256": sha256_file(loc_record_path),
            "score_path": "scores.npz",
            "score_sha256": score_sha,
        }
        record["payload_sha256"] = _payload_sha(record)
        record_path = target / "RECORD.json"
        atomic_write_json(record_path, record)
        records.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(out).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
    freeze = {
        "schema_version": "ciw-localization-score-freeze-v1",
        "method_id": METHOD,
        "n_cells": len(records),
        "expected_cells": list(CELLS),
        "all_expected_scores_present": len(records) == len(CELLS),
        "labels_opened_during_fit": False,
        "targets_accessed_during_fit": False,
        "ciw_external_freeze_sha256": sha256_file(ciw_root / "SCORE_FREEZE_MANIFEST.json"),
        "records": records,
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({"status": "PASS", "stage": "fit", "cells": len(records)}, indent=2))


def _preflight(fit_root: Path, localization_release: Path) -> dict[str, dict[str, Any]]:
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text())
    body = dict(freeze)
    digest = body.pop("payload_sha256", None)
    if digest != _payload_sha(body):
        raise RuntimeError("localization freeze payload hash failed")
    if not (
        freeze.get("all_expected_scores_present") is True
        and freeze.get("labels_opened_during_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and tuple(freeze.get("expected_cells", ())) == CELLS
    ):
        raise RuntimeError("localization freeze contract failed")
    output = {}
    for row in freeze["records"]:
        cell_id = str(row["cell_id"])
        record_path = fit_root / row["record_path"]
        if sha256_file(record_path) != row["record_sha256"]:
            raise RuntimeError(f"{cell_id}: record changed")
        record = json.loads(record_path.read_text())
        body = dict(record)
        digest = body.pop("payload_sha256", None)
        if digest != _payload_sha(body):
            raise RuntimeError(f"{cell_id}: record payload failed")
        score_path = record_path.parent / record["score_path"]
        if sha256_file(score_path) != record["score_sha256"]:
            raise RuntimeError(f"{cell_id}: score changed")
        arrays = load_npz_no_pickle(score_path)
        loc_path = localization_release / "build_A/localization/fit/cells" / cell_id / "RECORD.json"
        loc_record, loc = load_localization_score_bundle(loc_path)
        if not np.array_equal(arrays["row_ids"].astype(str), loc["row_ids"].astype(str)):
            raise RuntimeError(f"{cell_id}: reference rows changed")
        output[cell_id] = {"record": record, "arrays": arrays, "loc_record": loc_record, "loc": loc}
    if set(output) != set(CELLS):
        raise RuntimeError("localization cell roster is incomplete")
    return output


def evaluate(
    fit_root: Path, localization_release: Path, out: Path,
) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    verified = _preflight(fit_root, localization_release)

    # This import is deliberately after the complete target-free score preflight.
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    evaluation_root = localization_release / "build_A/localization/evaluation"
    evaluation_manifest = json.loads((evaluation_root / "MANIFEST.json").read_text())
    expected_hashes = {row["path"]: row["sha256"] for row in evaluation_manifest["artifacts"]}
    decisions_path = evaluation_root / "localization_decisions.csv"
    if sha256_file(decisions_path) != expected_hashes["localization_decisions.csv"]:
        raise RuntimeError("frozen localization decision/label table hash failed")
    decision_labels: dict[str, dict[str, tuple[str, int]]] = {cell: {} for cell in PB_CELLS}
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cell_id = str(row["cell_id"])
            if (
                cell_id in decision_labels
                and row["system_id"] == "deem_b3__loc_geomean_v1"
            ):
                row_id = str(row["row_id"])
                if row_id in decision_labels[cell_id]:
                    raise RuntimeError(f"{cell_id}: duplicate frozen label row")
                decision_labels[cell_id][row_id] = (
                    str(row["group_id"]), int(row["true_first_error"])
                )
    method_system = {
        METHOD: None,
        "deem_b3": "deem_b3__loc_geomean_v1",
        "iu_pcr": "iu_pcr__loc_geomean_v1",
        "dufs_liu": "dufs_liu__loc_geomean_v1",
    }
    pb_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell_id in PB_CELLS:
        item = verified[cell_id]
        record = item["record"]
        row_ids = tuple(item["arrays"]["row_ids"].astype(str))
        if set(row_ids) != set(decision_labels[cell_id]):
            raise RuntimeError(f"{cell_id}: frozen label row join failed")
        offsets = np.asarray(item["arrays"]["segment_offsets"], dtype=np.int64)
        loc = item["loc"]
        loc_ids = tuple(map(str, loc["system_ids"].tolist()))
        for method, system_id in method_system.items():
            if method == METHOD:
                all_steps = np.asarray(item["arrays"]["step_scores"], dtype=float)
            else:
                all_steps = np.asarray(loc["system_scores"], dtype=float)[loc_ids.index(system_id)]
            key = (str(record["model_id"]), method)
            target = pb_rows.setdefault(key, [])
            for index, row_id in enumerate(row_ids):
                lo, hi = map(int, offsets[index:index + 2])
                group_id, first_error = decision_labels[cell_id][row_id]
                target.append({
                    "row_id": row_id,
                    "group_id": group_id,
                    "slice_id": str(record["slice_id"]),
                    "first_error": first_error,
                    "step_scores": all_steps[lo:hi].tolist(),
                })

    pb_metrics = []
    for (model_id, method), rows in sorted(pb_rows.items()):
        result = evaluation.crossfit_processbench_threshold(rows)
        metrics = result["metrics"]["aggregate"]
        pb_metrics.append({"model_id": model_id, "method_id": method, **metrics})
    pb_aggregate = []
    for method in method_system:
        rows = [row for row in pb_metrics if row["method_id"] == method]
        pb_aggregate.append({
            "method_id": method,
            "n_models": len(rows),
            **{name: float(np.mean([row[name] for row in rows])) for name in (
                "official_macro_f1", "first_error_exact", "first_error_within_one",
                "clean_abstention_accuracy", "overall_decision_accuracy",
            )},
        })

    prm_labels = load_npz_no_pickle(
        localization_release / "build_A/localization/evaluation/prmbench_steps.npz"
    )
    prm_item = verified[PRM_CELL]
    full_rows = prm_item["arrays"]["row_ids"].astype(str)
    full_offsets = np.asarray(prm_item["arrays"]["segment_offsets"], dtype=np.int64)
    row_index = {row_id: index for index, row_id in enumerate(full_rows)}
    selected_scores = []
    for row_id, n_steps in zip(
        prm_labels["response_row_ids"].astype(str), np.diff(prm_labels["step_offsets"])
    ):
        index = row_index[row_id]
        lo, hi = map(int, full_offsets[index:index + 2])
        if hi - lo != int(n_steps):
            raise RuntimeError("PRMBench step roster differs from frozen CIW localization")
        selected_scores.append(np.asarray(prm_item["arrays"]["step_scores"], dtype=float)[lo:hi])
    ciw_prm = np.concatenate(selected_scores)
    labels = np.asarray(prm_labels["step_labels"], dtype=np.int8)
    prm_results = [{"method_id": METHOD, **evaluation.prmbench_step_metrics(labels, ciw_prm)}]
    existing_ids = tuple(map(str, prm_labels["system_ids"].tolist()))
    existing_scores = np.asarray(prm_labels["system_scores"], dtype=float)
    for method, system_id in method_system.items():
        if method == METHOD:
            continue
        prm_results.append({
            "method_id": method,
            **evaluation.prmbench_step_metrics(labels, existing_scores[existing_ids.index(system_id)]),
        })

    _write_csv(out / "PROCESSBENCH_BY_MODEL.csv", pb_metrics)
    _write_csv(out / "PROCESSBENCH_MACRO.csv", pb_aggregate)
    _write_csv(out / "PRMBENCH_STEPS.csv", prm_results)
    ciw_pb = next(row for row in pb_aggregate if row["method_id"] == METHOD)
    ciw_prm_row = next(row for row in prm_results if row["method_id"] == METHOD)
    lines = [
        "# CIW-DEEM localization transfer", "",
        "CIW response risk is combined with the already frozen token-IU29 step risk. No new token model is fitted.", "",
        "| Method | ProcessBench macro-F1 | First-error exact | Within one step | PRMBench step AUROC | PRMBench step AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in method_system:
        pb = next(row for row in pb_aggregate if row["method_id"] == method)
        prm = next(row for row in prm_results if row["method_id"] == method)
        lines.append(
            f"| {method} | {pb['official_macro_f1']:.6f} | {pb['first_error_exact']:.6f} | "
            f"{pb['first_error_within_one']:.6f} | {prm['auroc']:.6f} | {prm['auprc']:.6f} |"
        )
    lines += ["", "This is an application transfer result, not a new method-selection test."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "ciw-localization-evaluation-v1",
        "scores_preflighted_before_labels": True,
        "n_cells": len(verified),
        "ciw_processbench_macro_f1": ciw_pb["official_macro_f1"],
        "ciw_prmbench_step_auroc": ciw_prm_row["auroc"],
        "outputs": {name: sha256_file(out / name) for name in (
            "PROCESSBENCH_BY_MODEL.csv", "PROCESSBENCH_MACRO.csv", "PRMBENCH_STEPS.csv", "REPORT.md"
        )},
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(out / "EVALUATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "stage": "evaluate", **manifest}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--ciw-root", required=True)
    fit_p.add_argument("--localization-release", required=True)
    fit_p.add_argument("--out-dir", required=True)
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--fit-dir", required=True)
    eval_p.add_argument("--localization-release", required=True)
    eval_p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit(Path(args.ciw_root).resolve(), Path(args.localization_release).resolve(), Path(args.out_dir).resolve())
    else:
        evaluate(
            Path(args.fit_dir).resolve(), Path(args.localization_release).resolve(),
            Path(args.out_dir).resolve(),
        )


if __name__ == "__main__":
    main()
