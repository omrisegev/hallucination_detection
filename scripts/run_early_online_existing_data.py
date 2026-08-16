#!/usr/bin/env python3
"""Run the frozen existing-cache early/online detection screen (CPU only).

The driver never performs model inference and never changes its input caches.
It writes one isolated result directory per source cache plus an inventory and
an aggregate checkpoint under ``results/early_online_existing_data_v1``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spectral_utils.online_convergence import (  # noqa: E402
    DEFAULT_BUDGETS,
    apply_declaration_policy,
    build_score_rows,
    calibrate_declaration_policy,
    convergence_table,
    declaration_summary,
    final_calibration,
    fit_frozen_prefix_iu,
    grouped_calibration_split,
    normalize_cache_records,
    per_trace_convergence,
)


DEFAULT_OUT = REPO / "results" / "early_online_existing_data_v1"
TEMP_PHASE15 = Path(
    "/private/tmp/hallucination_phase1_audit_20260816/"
    "math500_qwen7b_T1.0_run0.pkl"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def source_id(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    return "__".join(part for part in (parent, stem) if part).replace(" ", "_")


def _score_lookup(
    rows: Sequence[Mapping[str, Any]], method: str, budget: int
) -> dict[int, Mapping[str, Any]]:
    return {
        int(row["unit_index"]): row for row in rows
        if row["method"] == method and not bool(row["is_final"])
        and int(row["budget"]) == int(budget)
    }


def grouped_bootstrap_delta(
    score_rows: Sequence[Mapping[str, Any]],
    method: str,
    baseline: str,
    budget: int,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    left = _score_lookup(score_rows, method, budget)
    right = _score_lookup(score_rows, baseline, budget)
    units = sorted(set(left) & set(right))
    if not units:
        return {"method": method, "baseline": baseline, "budget": budget, "n": 0}
    groups = sorted({str(left[unit]["group"]) for unit in units})
    by_group = {
        group: [unit for unit in units if str(left[unit]["group"]) == group]
        for group in groups
    }

    def delta(sampled_units: Sequence[int]) -> float:
        labels = np.asarray([left[unit]["label_error"] for unit in sampled_units], int)
        if len(np.unique(labels)) < 2:
            return float("nan")
        a = np.asarray([left[unit]["score"] for unit in sampled_units], float)
        b = np.asarray([right[unit]["score"] for unit in sampled_units], float)
        return float(roc_auc_score(labels, a) - roc_auc_score(labels, b))

    observed = delta(units)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(repeats)):
        draw = rng.choice(groups, size=len(groups), replace=True)
        sampled = [unit for group in draw for unit in by_group[str(group)]]
        value = delta(sampled)
        if np.isfinite(value):
            boot.append(value)
    low, high = (
        np.quantile(boot, [0.025, 0.975]) if boot else (float("nan"), float("nan"))
    )
    return {
        "method": method,
        "baseline": baseline,
        "budget": int(budget),
        "n": int(len(units)),
        "n_groups": int(len(groups)),
        "delta_auroc": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_repeats_valid": int(len(boot)),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(result: Mapping[str, Any]) -> str:
    metadata = result["metadata"]
    convergence = result["convergence"]
    declarations = result["declarations"]
    deltas = result["paired_deltas"]
    methods = ("iu28_no_length", "iu29_elapsed_length", "deepconf_entropy_w64")
    rows = []
    for budget in metadata["budgets"]:
        for method in methods:
            item = next((row for row in convergence
                         if row["method"] == method and row["budget"] == budget), None)
            if item:
                rows.append(
                    f"| {budget} | {method} | {item['n_at_risk']} | "
                    f"{_fmt(item['auroc'])} | {_fmt(item['spearman_vs_final'])} | "
                    f"{_fmt(item['final_decision_agreement'])} | "
                    f"{_fmt(item['above_chance_auc_recovery'])} |"
                )
    declaration_rows = []
    for method, item in declarations.items():
        summary = item["evaluation_summary"]
        declaration_rows.append(
            f"| {method} | {_fmt(summary['coverage'])} | "
            f"{_fmt(summary['ever_wrong_rate_all'])} | "
            f"{_fmt(summary['false_alarm_rate_all'])} | "
            f"{_fmt(summary['false_clearance_rate_all'])} | "
            f"{_fmt(summary['mean_decision_budget'], 1)} | "
            f"{_fmt(summary['mean_potential_tokens_remaining'], 1)} |"
        )
    delta_rows = [
        f"| {row['budget']} | {row['method']} − {row['baseline']} | "
        f"{_fmt(row.get('delta_auroc'))} | "
        f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] | {row['n']} |"
        for row in deltas
    ]
    dropped28 = ", ".join(
        result["models"]["iu28_no_length"]["dropped_feature_names"]
    ) or "none"
    return f"""# Existing-cache early/online detection screen

Source: `{metadata['source_path']}`

Source SHA-256: `{metadata['source_sha256']}`
Run mode: CPU-only retrospective; no inference or raw-data mutation.

## Data and split

- Usable traces: **{metadata['n_records']}**; error rate: **{metadata['error_rate']:.3f}**.
- Calibration/evaluation: **{metadata['n_calibration']} / {metadata['n_evaluation']}**,
  group-disjoint (split seed {metadata['split_seed']}).
- Absolute monitoring budgets: {', '.join(map(str, metadata['budgets']))}.
- The 28-stream fit dropped unavailable/degenerate telemetry: **{dropped28}**.

## Convergence to the completed-trace score

| budget | method | at risk | AUROC | Spearman vs final | decision agreement | AUROC recovery |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Two-sided early declaration (held-out)

Thresholds were selected on the calibration half under the registered 10%
ever-wrong-over-horizon constraint and require two consecutive observations.

| method | coverage | ever wrong | false alarm | false clearance | mean budget | potential tokens left |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(declaration_rows)}

Potential tokens left are not realized savings: no forced-closure generation
was performed.

## Paired access-matched comparison

| budget | contrast | delta AUROC | grouped 95% interval | n |
|---:|---|---:|---:|---:|
{chr(10).join(delta_rows)}

## Claim boundary

This run is a one-cell retrospective pilot. It can establish whether the
pipeline and convergence question are measurable on the existing cache; it
cannot establish cross-model/dataset generalization or exact equivalence to
DeepConf, REFRAIN, Streaming Hallucination Detection, LEASH, or Online
Auditing. Exact native-paper reconstruction remains gated on multi-cell
evidence under the frozen protocol.
"""


def run_cache(
    path: Path,
    out: Path,
    *,
    budgets: Sequence[int],
    rows_per_trace: int,
    bootstrap: int,
    seed: int,
    max_records: int | None,
    filter_field: str | None = None,
    filter_value: str | None = None,
    cell_id: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    records = normalize_cache_records(cache, min_tokens=min(budgets))
    if filter_field is not None:
        records = [
            row for row in records
            if str(row.get(filter_field)) == str(filter_value)
        ]
    if max_records is not None:
        records = records[:int(max_records)]
    if len(records) < 30:
        raise ValueError(f"only {len(records)} usable traces")
    labels = np.asarray([int(not bool(row["label"])) for row in records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 10:
        raise ValueError(f"too few examples in one class: {counts.tolist()}")

    calibration_indexes, evaluation_indexes, split_seed = grouped_calibration_split(
        records, seed=seed
    )
    calibration_records = [records[index] for index in calibration_indexes]
    models = {
        "iu28_no_length": fit_frozen_prefix_iu(
            calibration_records,
            include_elapsed_length=False,
            rows_per_trace=rows_per_trace,
        ),
        "iu29_elapsed_length": fit_frozen_prefix_iu(
            calibration_records,
            include_elapsed_length=True,
            rows_per_trace=rows_per_trace,
        ),
    }
    calibration_scores = build_score_rows(
        records, calibration_indexes, models, budgets=budgets
    )
    evaluation_scores = build_score_rows(
        records, evaluation_indexes, models, budgets=budgets
    )
    final_parameters = final_calibration(calibration_scores)
    convergence = convergence_table(
        evaluation_scores, final_parameters, budgets=budgets
    )

    declarations = {}
    methods = sorted({row["method"] for row in evaluation_scores})
    for method in methods:
        policy = calibrate_declaration_policy(calibration_scores, method)
        evaluated = apply_declaration_policy(
            evaluation_scores,
            method,
            low=policy["low"],
            high=policy["high"],
            stable_observations=policy["stable_observations"],
        )
        declarations[method] = {
            "policy": policy,
            "evaluation_summary": declaration_summary(evaluated),
            "evaluation_rows": evaluated,
        }

    paired_deltas = []
    for budget in budgets:
        paired_deltas.append(grouped_bootstrap_delta(
            evaluation_scores,
            "iu28_no_length",
            "deepconf_entropy_w64",
            int(budget),
            repeats=bootstrap,
            seed=seed + int(budget),
        ))

    result = {
        "metadata": {
            "protocol": "EARLY_ONLINE_EXISTING_DATA_V1",
            "source_path": str(path.resolve()),
            "source_size_bytes": int(path.stat().st_size),
            "source_sha256": file_sha256(path),
            "n_records": int(len(records)),
            "n_calibration": int(len(calibration_indexes)),
            "n_evaluation": int(len(evaluation_indexes)),
            "error_rate": float(labels.mean()),
            "class_counts_correct_error": [int(counts[0]), int(counts[1])],
            "split_seed": int(split_seed),
            "budgets": [int(value) for value in budgets],
            "rows_per_trace": int(rows_per_trace),
            "bootstrap_repeats": int(bootstrap),
            "elapsed_seconds": float(time.time() - started),
            "new_inference": False,
            "gpu_hours": 0,
            "record_filter": (
                {"field": filter_field, "value": filter_value}
                if filter_field is not None else None
            ),
        },
        "models": {name: model.diagnostics for name, model in models.items()},
        "final_calibration": final_parameters,
        "convergence": convergence,
        "per_trace_convergence": per_trace_convergence(
            evaluation_scores, final_parameters
        ),
        "declarations": declarations,
        "paired_deltas": paired_deltas,
    }
    cell_out = out / (cell_id or source_id(path))
    write_json(cell_out / "result.json", result)
    write_csv(cell_out / "convergence.csv", convergence)
    write_csv(cell_out / "per_trace_convergence.csv", result["per_trace_convergence"])
    write_csv(cell_out / "paired_deltas.csv", paired_deltas)
    for split_name, score_rows in (
        ("calibration", calibration_scores), ("evaluation", evaluation_scores)
    ):
        write_csv(cell_out / f"scores_{split_name}.csv", score_rows)
    (cell_out / "REPORT.md").write_text(build_report(_jsonable(result)), encoding="utf-8")
    return result


def discover_default_caches() -> list[Path]:
    candidates = []
    if TEMP_PHASE15.exists():
        candidates.append(TEMP_PHASE15)
    local = REPO / "local_cache" / "math500_qwen7b_T1.0_run0.pkl"
    if local.exists():
        candidates.append(local)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--rows-per-trace", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--filter-field")
    parser.add_argument("--filter-value")
    parser.add_argument("--cell-id")
    args = parser.parse_args()

    caches = args.cache or discover_default_caches()
    args.out.mkdir(parents=True, exist_ok=True)
    inventory_path = args.out / "INVENTORY.json"
    if inventory_path.exists():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except Exception:
            inventory = []
    else:
        inventory = []
    completed = []
    for path in caches:
        path = path.expanduser()
        item = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            item.update({"status": "blocked", "reason": "missing"})
            inventory.append(item)
            continue
        item["size_bytes"] = int(path.stat().st_size)
        if is_lfs_pointer(path):
            item.update({"status": "blocked", "reason": "git_lfs_pointer"})
            inventory.append(item)
            continue
        try:
            print(f"[run] {path}", flush=True)
            result = run_cache(
                path,
                args.out,
                budgets=tuple(sorted(set(args.budgets))),
                rows_per_trace=args.rows_per_trace,
                bootstrap=args.bootstrap,
                seed=args.seed,
                max_records=args.max_records,
                filter_field=args.filter_field,
                filter_value=args.filter_value,
                cell_id=args.cell_id,
            )
        except Exception as exc:
            item.update({
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            inventory.append(item)
            print(f"[blocked] {path}: {item['reason']}", flush=True)
            continue
        item.update({
            "status": "completed",
            "source_id": args.cell_id or source_id(path),
            "n_records": result["metadata"]["n_records"],
        })
        # Repeated invocations extend one isolated campaign.  Replace an older
        # entry for the same logical cell rather than losing prior cells or
        # duplicating a rerun.
        logical_id = item["source_id"]
        inventory = [
            old for old in inventory if old.get("source_id") != logical_id
        ]
        inventory.append(item)
        completed.append(result)
        print(
            f"[done] {path}: {result['metadata']['n_records']} traces, "
            f"{result['metadata']['elapsed_seconds']:.1f}s",
            flush=True,
        )
    write_json(inventory_path, inventory)
    completed_total = sum(item.get("status") == "completed" for item in inventory)
    checkpoint = {
        "protocol": "EARLY_ONLINE_EXISTING_DATA_V1",
        "completed_cells": int(completed_total),
        "inventory": inventory,
        "cross_cell_confirmation": False,
        "reason": (
            "A single materialized cell is a retrospective pilot; cross-family "
            "confirmation requires the other existing Drive/LFS caches."
            if completed_total < 2 else
            "Multiple cells completed; inspect grouped paired intervals before promotion."
        ),
        "new_inference": False,
        "gpu_hours": 0,
    }
    write_json(args.out / "CHECKPOINT.json", checkpoint)
    print(f"Results: {args.out}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
