#!/usr/bin/env python3
"""Evaluate frozen external CIW-DEEM scores beside reconstruction v1 methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_file,
)


def _payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-dir", required=True)
    parser.add_argument("--reference-release", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    fit_dir = Path(args.fit_dir).resolve()
    reference = Path(args.reference_release).resolve()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    freeze_path = fit_dir / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_sha(payload):
        raise RuntimeError("CIW score-freeze payload hash failed")
    if not (
        freeze.get("all_expected_scores_present") is True
        and freeze.get("labels_opened_by_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and freeze.get("n_cells") == len(freeze.get("cells", ()))
    ):
        raise RuntimeError("CIW score freeze is incomplete or not target-free")

    # Pass 1: verify every CIW artifact and every reference score before labels.
    verified: dict[str, dict[str, Any]] = {}
    for item in freeze["cells"]:
        cell_id = str(item["cell_id"])
        record_path = fit_dir / str(item["record_path"])
        if sha256_file(record_path) != item["record_sha256"]:
            raise RuntimeError(f"{cell_id}: CIW record hash failed")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        body = dict(record)
        digest = body.pop("payload_sha256", None)
        if digest != _payload_sha(body):
            raise RuntimeError(f"{cell_id}: CIW record payload failed")
        score_path = record_path.parent / record["score_path"]
        if sha256_file(score_path) != record["score_sha256"]:
            raise RuntimeError(f"{cell_id}: CIW score hash failed")
        ciw = load_npz_no_pickle(score_path)
        reference_scores: dict[str, np.ndarray] = {}
        for method in ("deem_b3", "iu_pcr", "dufs_liu"):
            base_dir = reference / "build_A" / "external_final_answer" / "fit" / "cells" / cell_id / method
            base_record = json.loads((base_dir / "RECORD.json").read_text(encoding="utf-8"))
            base_score = base_dir / str(base_record["score_path"])
            if sha256_file(base_score) != base_record["score_sha256"]:
                raise RuntimeError(f"{cell_id}/{method}: reference score hash failed")
            arrays = load_npz_no_pickle(base_score)
            if not np.array_equal(arrays["row_ids"].astype(str), ciw["row_ids"].astype(str)):
                raise RuntimeError(f"{cell_id}/{method}: row roster differs from CIW")
            reference_scores[method] = np.asarray(arrays["score"], dtype=float)
        verified[cell_id] = {
            "record": record,
            "row_ids": ciw["row_ids"].astype(str),
            "ciw": np.asarray(ciw["score"], dtype=float),
            "references": reference_scores,
        }

    # Pass 2: labels are opened only after the complete score preflight.
    per_cell: list[dict[str, Any]] = []
    methods = ("ciw_deem", "deem_b3", "iu_pcr", "dufs_liu")
    for cell_id, item in sorted(verified.items()):
        label_path = reference / "build_A" / "external_final_answer" / "evaluation" / "labels" / f"{cell_id}.npz"
        labels = load_npz_no_pickle(label_path)
        if not np.array_equal(labels["row_ids"].astype(str), item["row_ids"]):
            raise RuntimeError(f"{cell_id}: label row roster differs from frozen scores")
        y_key = "incorrect" if "incorrect" in labels else "labels"
        y = np.asarray(labels[y_key], dtype=np.int8)
        score_map = {"ciw_deem": item["ciw"], **item["references"]}
        metrics = {method: _metric(y, score_map[method]) for method in methods}
        record = item["record"]
        for method in methods:
            auroc, auprc = metrics[method]
            per_cell.append({
                "cell_id": cell_id,
                "population_id": record["population_id"],
                "comparison_group_id": record["comparison_group_id"],
                "dataset_id": record["dataset_id"],
                "model_id": record["model_id"],
                "slice_id": record["slice_id"],
                "method_id": method,
                "n": len(y),
                "n_incorrect": int(y.sum()),
                "auroc": auroc,
                "auprc": auprc,
            })

    aggregates: list[dict[str, Any]] = []
    groups = sorted({row["comparison_group_id"] for row in per_cell})
    for group in groups:
        for method in methods:
            rows = [row for row in per_cell if row["comparison_group_id"] == group and row["method_id"] == method]
            aggregates.append({
                "comparison_group_id": group,
                "method_id": method,
                "n_cells": len(rows),
                "cell_macro_auroc": float(np.mean([row["auroc"] for row in rows])),
                "cell_macro_auprc": float(np.mean([row["auprc"] for row in rows])),
            })
    contrasts: list[dict[str, Any]] = []
    for group in groups:
        ciw = next(row for row in aggregates if row["comparison_group_id"] == group and row["method_id"] == "ciw_deem")
        for baseline in ("deem_b3", "iu_pcr", "dufs_liu"):
            ref = next(row for row in aggregates if row["comparison_group_id"] == group and row["method_id"] == baseline)
            contrasts.append({
                "comparison_group_id": group,
                "contrast": f"ciw_deem-minus-{baseline}",
                "n_cells": ciw["n_cells"],
                "delta_cell_macro_auroc": ciw["cell_macro_auroc"] - ref["cell_macro_auroc"],
                "delta_cell_macro_auprc": ciw["cell_macro_auprc"] - ref["cell_macro_auprc"],
            })
    _write_csv(out / "PER_CELL.csv", per_cell)
    _write_csv(out / "AGGREGATES.csv", aggregates)
    _write_csv(out / "CONTRASTS.csv", contrasts)
    report_lines = [
        "# CIW-DEEM external completed-response benchmark",
        "",
        "All CIW scores were frozen and hash-checked before the evaluator opened the existing reconstruction labels.",
        "",
        "| Population group | CIW AUROC | CIW AUPRC | CIW - B3 AUROC | CIW - IU AUROC | CIW - DUFS-LIU AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        ciw = next(row for row in aggregates if row["comparison_group_id"] == group and row["method_id"] == "ciw_deem")
        values = {
            row["contrast"].split("minus-")[-1]: row["delta_cell_macro_auroc"]
            for row in contrasts if row["comparison_group_id"] == group
        }
        report_lines.append(
            f"| {group} | {ciw['cell_macro_auroc']:.6f} | {ciw['cell_macro_auprc']:.6f} | "
            f"{values['deem_b3']:+.6f} | {values['iu_pcr']:+.6f} | {values['dufs_liu']:+.6f} |"
        )
    report_lines += [
        "",
        "These are retrospective point estimates on already-open application populations. They document transfer; they are not a new confirmatory promotion test.",
    ]
    (out / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "ciw-deem-external-evaluation-v1",
        "fit_freeze_sha256": sha256_file(freeze_path),
        "reference_release": str(reference),
        "labels_opened_after_complete_preflight": True,
        "n_cells": len(verified),
        "outputs": {
            name: sha256_file(out / name)
            for name in ("PER_CELL.csv", "AGGREGATES.csv", "CONTRASTS.csv", "REPORT.md")
        },
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(out / "EVALUATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "cells": len(verified), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
