#!/usr/bin/env python3
"""Re-run only the frozen non-selection transfer stage for architecture v2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_global_local_online_architecture_v2 as run  # noqa: E402


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    with (run.OUT / "HEAD_SELECTION.json").open(encoding="utf-8") as handle:
        head_selection = json.load(handle)
    with (run.OUT / "ARCHITECTURE_SELECTION.json").open(encoding="utf-8") as handle:
        architecture_selection = json.load(handle)

    architecture_records = _read_csv(run.OUT / "ARCHITECTURE_DEV_PER_QUESTION.csv")
    architecture_metrics = _read_csv(run.OUT / "ARCHITECTURE_DEV_METRICS.csv")
    efficiency = [
        row for row in _read_csv(run.OUT / "EFFICIENCY.csv")
        if row.get("stage") != "transfer"
    ]
    frozen_configs = [
        {
            "architecture": row["base_architecture"],
            "weight": float(row["weight"]),
            "locator": row["locator"],
        }
        for row in architecture_selection["best_by_base"].values()
    ]

    transfer_records, transfer_metrics, transfer_freeze = [], [], {}
    for model_name in run.MODELS:
        for family in run.FAMILIES:
            if (model_name, family) in run.DEV_CELLS:
                continue
            path = run._cell_path(model_name, family)
            rows = run.load_rows(path)
            calibration, evaluation = run._split(rows)
            print(f"[transfer] {model_name}/{family}: {len(calibration)} cal, {len(evaluation)} eval", flush=True)
            models = run._fit_selected_cell(calibration, head_selection["selected"])
            cal_output = run._selected_outputs(calibration, models, head_selection["selected"])
            eval_output = run._selected_outputs(evaluation, models, head_selection["selected"])
            records, metrics, frozen = run._architecture_cell(
                model_name, family, calibration, evaluation,
                cal_output, eval_output, frozen_configs,
            )
            transfer_records.extend(records)
            transfer_metrics.extend(metrics)
            transfer_freeze[f"{model_name}/{family}"] = frozen
            efficiency.extend({
                "model": model_name, "family": family, "stage": "transfer", **row
            } for row in models.efficiency)

    all_records = architecture_records + transfer_records
    all_metrics = architecture_metrics + transfer_metrics
    run._write_csv(run.OUT / "ARCHITECTURE_PER_QUESTION.csv", all_records)
    run._write_csv(run.OUT / "ARCHITECTURE_METRICS.csv", all_metrics)
    run._write_csv(
        run.OUT / "ARCHITECTURE_AGGREGATE.csv",
        run._aggregate_architecture(all_metrics, dev_only=False),
    )
    run._write_csv(run.OUT / "EFFICIENCY.csv", efficiency)
    run._write_json(run.OUT / "TRANSFER_SCORE_FREEZE.json", transfer_freeze)
    manifest_path = run.OUT / "RUN_MANIFEST.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["transfer_resume_corrected"] = True
    manifest["transfer_score_freeze_sha256"] = run._sha256(run.OUT / "TRANSFER_SCORE_FREEZE.json")
    run._write_json(manifest_path, manifest)
    print(f"[done] corrected frozen transfer in {run.OUT}", flush=True)


if __name__ == "__main__":
    main()
