#!/usr/bin/env python3
"""Run causal GL-LIU global/local/fused scorers on existing caches (CPU only)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_early_online_existing_data import (  # noqa: E402
    file_sha256,
    grouped_bootstrap_delta,
    write_csv,
    write_json,
)
from spectral_utils.online_convergence import (  # noqa: E402
    DEFAULT_BUDGETS,
    apply_declaration_policy,
    calibrate_declaration_policy,
    convergence_table,
    declaration_summary,
    final_calibration,
    fit_frozen_prefix_iu,
    grouped_calibration_split,
    normalize_cache_records,
    per_trace_convergence,
    prefix_method_scores,
)
from spectral_utils.online_localization_fusion import (  # noqa: E402
    FrozenOnlineGLIUEnsemble,
    fit_frozen_online_gl_liu,
)


DEFAULT_OUT = REPO / "results" / "early_online_localization_models_v1"
V1_ROOT = REPO / "results" / "early_online_existing_data_v1"
FOCUS_METHODS = (
    "global_gl_liu_no_length",
    "global_gl_liu_elapsed_length",
    "local_temporal_gl_liu_max",
    "local_dufs_gl_liu_top5",
    "fused_gl_liu",
    "cusum_max",
    "sw_var_peak",
    "cusum_swvar_equal",
    "iu28_no_length",
    "deepconf_entropy_w64",
)


def method_scores(
    row: Mapping[str, Any],
    budget: int | None,
    iu_model,
    gl_liu: FrozenOnlineGLIUEnsemble,
) -> dict[str, float]:
    scores = prefix_method_scores(
        row, budget, {"iu28_no_length": iu_model}
    )
    scores.update(gl_liu.scores(row, budget))
    return scores


def build_score_rows(
    rows: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    iu_model,
    gl_liu: FrozenOnlineGLIUEnsemble,
    *,
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    output = []
    for index in indexes:
        row = rows[int(index)]
        length = int(row["_length"])
        common = {
            "unit_index": int(index),
            "trace_id": str(row["_trace_id"]),
            "group": str(row["_group"]),
            "label_error": int(not bool(row["label"])),
            "trace_length": length,
        }
        for budget in budgets:
            if length <= int(budget):
                continue
            scores = method_scores(row, int(budget), iu_model, gl_liu)
            output.extend({
                **common,
                "budget": int(budget),
                "is_final": False,
                "method": method,
                "score": float(score),
            } for method, score in scores.items())
        scores = method_scores(row, None, iu_model, gl_liu)
        output.extend({
            **common,
            "budget": length,
            "is_final": True,
            "method": method,
            "score": float(score),
        } for method, score in scores.items())
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(result: Mapping[str, Any]) -> str:
    metadata = result["metadata"]
    lookup = {
        (row["method"], row["budget"]): row for row in result["convergence"]
    }
    rows = []
    for budget in metadata["budgets"]:
        for method in FOCUS_METHODS:
            item = lookup.get((method, budget))
            if item is None:
                continue
            rows.append(
                f"| {budget} | {method} | {item['n_at_risk']} | "
                f"{_fmt(item['auroc'])} | {_fmt(item['spearman_vs_final'])} | "
                f"{_fmt(item['final_decision_agreement'])} |"
            )
    declarations = []
    for method in FOCUS_METHODS:
        if method not in result["declarations"]:
            continue
        item = result["declarations"][method]["evaluation_summary"]
        declarations.append(
            f"| {method} | {_fmt(item['coverage'])} | "
            f"{_fmt(item['ever_wrong_rate_all'])} | "
            f"{_fmt(item['selective_error_rate'])} | "
            f"{_fmt(item['mean_decision_budget'], 1)} |"
        )
    deltas = [
        f"| {row['budget']} | {row['method']} − {row['baseline']} | "
        f"{_fmt(row.get('delta_auroc'))} | "
        f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] |"
        for row in result["paired_deltas"]
    ]
    return f"""# Causal localization-model online screen

Source: `{metadata['source_path']}`
Run mode: CPU-only retrospective; no inference or raw-data mutation.

- Traces: **{metadata['n_records']}**; calibration/evaluation:
  **{metadata['n_calibration']} / {metadata['n_evaluation']}**, group-disjoint.
- Target: final-answer error, not the ProcessBench step label.
- Every prefix feature was recomputed after telemetry truncation.

## Fixed-budget performance and convergence

| budget | method | at risk | AUROC | Spearman vs final | decision agreement |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Held-out early declarations

| method | coverage | ever wrong | selective error | mean budget |
|---|---:|---:|---:|---:|
{chr(10).join(declarations)}

## Within-cell paired contrasts

| budget | contrast | delta AUROC | grouped 95% interval |
|---:|---|---:|---:|
{chr(10).join(deltas)}

The global and local fits are label-free. Labels were used only for final-score
thresholds, early-declaration calibration, and held-out evaluation.
"""


def run_cache(
    path: Path,
    out: Path,
    *,
    cell_id: str,
    budgets: Sequence[int],
    rows_per_trace: int,
    bootstrap: int,
    seed: int,
    filter_field: str | None,
    filter_value: str | None,
    max_records: int | None,
    dufs_epochs: int,
    max_fit_tokens: int,
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
    labels = np.asarray([int(not bool(row["label"])) for row in records], int)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 10:
        raise ValueError(f"too few examples in one class: {counts.tolist()}")

    calibration, evaluation, split_seed = grouped_calibration_split(
        records, seed=seed
    )
    calibration_records = [records[index] for index in calibration]
    iu_model = fit_frozen_prefix_iu(
        calibration_records,
        include_elapsed_length=False,
        rows_per_trace=rows_per_trace,
    )
    gl_liu = fit_frozen_online_gl_liu(
        calibration_records,
        max_fit_tokens=max_fit_tokens,
        dufs_epochs=dufs_epochs,
    )
    calibration_scores = build_score_rows(
        records, calibration, iu_model, gl_liu, budgets=budgets
    )
    evaluation_scores = build_score_rows(
        records, evaluation, iu_model, gl_liu, budgets=budgets
    )
    parameters = final_calibration(calibration_scores)
    convergence = convergence_table(
        evaluation_scores, parameters, budgets=budgets
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
    primary_methods = (
        "global_gl_liu_no_length",
        "global_gl_liu_elapsed_length",
        "local_temporal_gl_liu_max",
        "local_dufs_gl_liu_top5",
        "fused_gl_liu",
        "cusum_swvar_equal",
    )
    for budget in budgets:
        for method in primary_methods:
            for baseline in ("deepconf_entropy_w64", "iu28_no_length"):
                if method == baseline:
                    continue
                paired_deltas.append(grouped_bootstrap_delta(
                    evaluation_scores,
                    method,
                    baseline,
                    int(budget),
                    repeats=bootstrap,
                    seed=seed + int(budget) + sum(map(ord, method + baseline)),
                ))

    result = {
        "metadata": {
            "protocol": "EARLY_ONLINE_LOCALIZATION_MODELS_V1",
            "source_path": str(path.resolve()),
            "source_size_bytes": int(path.stat().st_size),
            "source_sha256": file_sha256(path),
            "cell_id": cell_id,
            "n_records": int(len(records)),
            "n_calibration": int(len(calibration)),
            "n_evaluation": int(len(evaluation)),
            "error_rate": float(labels.mean()),
            "class_counts_correct_error": [int(counts[0]), int(counts[1])],
            "split_seed": int(split_seed),
            "budgets": [int(value) for value in budgets],
            "rows_per_trace": int(rows_per_trace),
            "bootstrap_repeats": int(bootstrap),
            "dufs_epochs": int(dufs_epochs),
            "local_max_fit_tokens": int(max_fit_tokens),
            "record_filter": (
                {"field": filter_field, "value": filter_value}
                if filter_field is not None else None
            ),
            "elapsed_seconds": float(time.time() - started),
            "new_inference": False,
            "gpu_hours": 0,
        },
        "models": {
            "iu28_no_length": iu_model.diagnostics,
            "gl_liu": gl_liu.diagnostics,
        },
        "final_calibration": parameters,
        "convergence": convergence,
        "per_trace_convergence": per_trace_convergence(
            evaluation_scores, parameters
        ),
        "declarations": declarations,
        "paired_deltas": paired_deltas,
    }
    cell_out = out / cell_id
    write_json(cell_out / "result.json", result)
    write_csv(cell_out / "convergence.csv", convergence)
    write_csv(cell_out / "paired_deltas.csv", paired_deltas)
    write_csv(
        cell_out / "per_trace_convergence.csv", result["per_trace_convergence"]
    )
    write_csv(cell_out / "scores_calibration.csv", calibration_scores)
    write_csv(cell_out / "scores_evaluation.csv", evaluation_scores)
    (cell_out / "REPORT.md").write_text(build_report(result), encoding="utf-8")
    return result


def v1_configs(root: Path) -> list[dict[str, Any]]:
    configs = []
    for result_path in sorted(root.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = result["metadata"]
        record_filter = metadata.get("record_filter") or {}
        configs.append({
            "path": Path(metadata["source_path"]),
            "cell_id": result_path.parent.name,
            "filter_field": record_filter.get("field"),
            "filter_value": record_filter.get("value"),
        })
    return configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument("--filter-field")
    parser.add_argument("--filter-value")
    parser.add_argument("--from-v1", type=Path, default=V1_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--rows-per-trace", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dufs-epochs", type=int, default=80)
    parser.add_argument("--max-fit-tokens", type=int, default=60_000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.cache is not None:
        if not args.cell_id:
            raise SystemExit("--cell-id is required with --cache")
        configs = [{
            "path": args.cache,
            "cell_id": args.cell_id,
            "filter_field": args.filter_field,
            "filter_value": args.filter_value,
        }]
    else:
        configs = v1_configs(args.from_v1)
    if not configs:
        raise SystemExit("no source cells found")
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for config in configs:
        path = Path(config["path"])
        item = {
            "cell_id": config["cell_id"],
            "path": str(path),
            "filter_field": config["filter_field"],
            "filter_value": config["filter_value"],
            "exists": path.exists(),
        }
        result_path = args.out / config["cell_id"] / "result.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            item.update({
                "status": "completed",
                "n_records": result["metadata"]["n_records"],
                "reused_existing_result": True,
            })
            inventory.append(item)
            continue
        if not path.exists():
            item.update({"status": "blocked", "reason": "source missing"})
            inventory.append(item)
            continue
        print(f"running {config['cell_id']} ({path})", flush=True)
        try:
            result = run_cache(
                path,
                args.out,
                cell_id=config["cell_id"],
                budgets=args.budgets,
                rows_per_trace=args.rows_per_trace,
                bootstrap=args.bootstrap,
                seed=args.seed,
                filter_field=config["filter_field"],
                filter_value=config["filter_value"],
                max_records=args.max_records,
                dufs_epochs=args.dufs_epochs,
                max_fit_tokens=args.max_fit_tokens,
            )
            item.update({
                "status": "completed",
                "n_records": result["metadata"]["n_records"],
                "elapsed_seconds": result["metadata"]["elapsed_seconds"],
            })
        except Exception as exc:
            item.update({
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            print(f"blocked {config['cell_id']}: {item['reason']}", flush=True)
        inventory.append(item)
        write_json(args.out / "INVENTORY.json", inventory)
    write_json(args.out / "INVENTORY.json", inventory)
    completed = sum(item["status"] == "completed" for item in inventory)
    write_json(args.out / "CHECKPOINT.json", {
        "protocol": "EARLY_ONLINE_LOCALIZATION_MODELS_V1",
        "completed_cells": int(completed),
        "blocked_cells": int(len(inventory) - completed),
        "new_inference": False,
        "gpu_hours": 0,
        "inventory": inventory,
    })
    print(f"completed {completed}/{len(inventory)} cells", flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
