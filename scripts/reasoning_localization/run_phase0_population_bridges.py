#!/usr/bin/env python3
"""Import and audit the registered Phase-0 current-population bridges.

S5A changes the population from the historical shared-row audit to the frozen
eight-cell Qwen panel. S5B then adds the four frozen Llama-3.1 cells. Because
these panels contain different generated traces, neither state is presented as
a row-paired causal effect relative to S4. The runner instead verifies the
dual-build release, recomputes every cell from frozen decisions, and produces
grouped panel bootstrap intervals plus a composition diagnostic for S5B-S5A.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reasoning_localization import run_phase0_reducer_bridge as s1  # noqa: E402


STATE_ID = "P0_S5_POPULATION_BRIDGES"
S5A_VARIANT = "P0_S5A_IU29_STEP_MAX_LOCAL_FIVEFOLD_QWEN8"
S5B_VARIANT = "P0_S5B_IU29_STEP_MAX_LOCAL_FIVEFOLD_FULL12"
SYSTEM_ID = "token_iu29__step_only_null_v1"
QWEN_MODELS = ("qwen3_4b", "qwen3_8b")
FULL_MODELS = QWEN_MODELS + ("llama31_8b",)
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
METRICS = ("macro_f1", "exact_error", "clean_abstention", "within_one")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026082901
DEFAULT_OUTPUT = REPO / "results/reasoning_localization_03662_v1/phase_0/p0_s5_population_bridges"
DEFAULT_REGISTRY = REPO / "results/reasoning_localization_03662_v1/phase_0/P0_S5_EXECUTION_REGISTRY.json"


class BridgeError(RuntimeError):
    """Raised when population/provenance/access constraints fail closed."""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise BridgeError(f"missing {label}: {path}")
    observed = s1.sha256_file(path)
    if observed != expected:
        raise BridgeError(f"{label} SHA mismatch: expected={expected} observed={observed}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("state_id") != STATE_ID or registry.get("status") != "FROZEN_BEFORE_RUN":
        raise BridgeError("population registry is not the frozen S5 contract")
    require_hash(Path(__file__).resolve(), registry["runner_sha256"], "runner")
    if registry.get("bootstrap_draws") != BOOTSTRAP_DRAWS or registry.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise BridgeError("bootstrap contract differs from frozen S5")
    if registry.get("system_id") != SYSTEM_ID:
        raise BridgeError("S5 must import the frozen token-only IU29 adapter")
    expected = [
        {"variant_id": S5A_VARIANT, "changed_factor": "population", "models": list(QWEN_MODELS), "n_cells": 8},
        {"variant_id": S5B_VARIANT, "changed_factor": "population_panel_extension", "models": list(FULL_MODELS), "n_cells": 12},
    ]
    if registry.get("states") != expected:
        raise BridgeError("S5 state roster differs from the frozen two-step population bridge")
    return registry


def preflight(registry: Mapping[str, Any]) -> dict[str, Any]:
    audited = []
    for item in registry["frozen_sources"]:
        path = resolve_path(item["path"])
        require_hash(path, item["sha256"], item["role"])
        audited.append({**item, "bytes": path.stat().st_size})
    by_role = {row["role"]: resolve_path(row["path"]) for row in registry["frozen_sources"]}
    if by_role["build_a_decisions"].read_bytes() != by_role["build_b_decisions"].read_bytes():
        raise BridgeError("dual-build decision artifacts differ")
    if by_role["build_a_metrics"].read_bytes() != by_role["build_b_metrics"].read_bytes():
        raise BridgeError("dual-build metric artifacts differ")
    return {"sources": audited, "dual_build_decisions_identical": True, "dual_build_metrics_identical": True}


def selected_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row["dataset_id"] != "processbench" or row["system_id"] != SYSTEM_ID:
            continue
        if row["model_id"] not in FULL_MODELS or row["slice_id"] not in FAMILIES:
            continue
        if row["cell_id"].startswith("processbench_panel_"):
            continue
        if row["status"] != "OK":
            raise BridgeError(f"non-OK frozen row: {row['cell_id']} {row['row_id']}")
        if row["access_level"] != "saved_output_probability_telemetry_one_pass":
            raise BridgeError("access contract mismatch")
        if row["fidelity"] != "registered_label_free_adapter_null":
            raise BridgeError("token-only fidelity contract mismatch")
        selected.append({
            "model": row["model_id"], "family": row["slice_id"], "cell_id": row["cell_id"],
            "row_id": row["row_id"], "group_id": row["group_id"], "fold": int(row["fold"]),
            "prediction": int(row["prediction_step"]), "target": int(row["true_first_error"]),
        })
    expected = {m: {f: (400 if f == "gsm8k" else 1000) for f in FAMILIES} for m in FULL_MODELS}
    observed = {m: {f: sum(r["model"] == m and r["family"] == f for r in selected) for f in FAMILIES} for m in FULL_MODELS}
    if observed != expected:
        raise BridgeError(f"unexpected current population: {observed}")
    if len({(r["model"], r["family"], r["row_id"]) for r in selected}) != len(selected):
        raise BridgeError("duplicate row identity")
    return selected


def cell_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result = s1._processbench(
        np.asarray([r["prediction"] for r in rows], dtype=int),
        np.asarray([r["target"] for r in rows], dtype=int),
    )
    return {"macro_f1": result["f1"], "exact_error": result["exact_error"],
            "clean_abstention": result["clean_abstention"], "within_one": result["within_one"]}


def bootstrap_cell(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> dict[str, np.ndarray]:
    pred = np.asarray([r["prediction"] for r in rows], dtype=int)
    target = np.asarray([r["target"] for r in rows], dtype=int)
    n = len(rows)
    out = {metric: np.empty(BOOTSTRAP_DRAWS, dtype=float) for metric in METRICS}
    offset = 0
    chunk = 250
    while offset < BOOTSTRAP_DRAWS:
        size = min(chunk, BOOTSTRAP_DRAWS - offset)
        idx = rng.integers(0, n, size=(size, n))
        p, t = pred[idx], target[idx]
        error = t != -1
        clean = ~error
        exact = np.sum((p == t) & error, axis=1) / np.sum(error, axis=1)
        abstain = np.sum((p == -1) & clean, axis=1) / np.sum(clean, axis=1)
        within = np.sum((p != -1) & (np.abs(p - t) <= 1) & error, axis=1) / np.sum(error, axis=1)
        f1 = np.where(exact + abstain > 0, 2 * exact * abstain / (exact + abstain), 0.0)
        sl = slice(offset, offset + size)
        out["macro_f1"][sl], out["exact_error"][sl] = f1, exact
        out["clean_abstention"][sl], out["within_one"][sl] = abstain, within
        offset += size
    return out


def evaluate_panel(rows: Sequence[Mapping[str, Any]], models: Sequence[str], variant: str,
                   seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    cells, draws_by_cell = [], {}
    rng = np.random.default_rng(seed)
    for model in models:
        for family in FAMILIES:
            subset = [r for r in rows if r["model"] == model and r["family"] == family]
            values = cell_metric(subset)
            cells.append({"variant_id": variant, "model": model, "family": family,
                          "n": len(subset), **values})
            draws_by_cell[(model, family)] = bootstrap_cell(subset, rng)
    aggregates = []
    panel_draws = {}
    for metric in METRICS:
        values = [r[metric] for r in cells]
        draw = np.mean([draws_by_cell[(m, f)][metric] for m in models for f in FAMILIES], axis=0)
        panel_draws[metric] = draw
        aggregates.append({"variant_id": variant, "metric_id": metric, "value": float(np.mean(values)),
                           "ci_low": float(np.quantile(draw, 0.025)), "ci_high": float(np.quantile(draw, 0.975)),
                           "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "source_question within scorer cell"})
    return cells, aggregates, panel_draws


def composition_contrasts(cells_a: Sequence[Mapping[str, Any]], cells_b: Sequence[Mapping[str, Any]],
                          draws_a: Mapping[str, np.ndarray], draws_b: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for metric in METRICS:
        delta_draw = draws_b[metric] - draws_a[metric]
        family_deltas = []
        for family in FAMILIES:
            a = np.mean([r[metric] for r in cells_a if r["family"] == family])
            b = np.mean([r[metric] for r in cells_b if r["family"] == family])
            family_deltas.append(float(b - a))
        output.append({
            "left_variant_id": S5B_VARIANT, "right_variant_id": S5A_VARIANT,
            "metric_id": metric,
            "delta": float(np.mean([r[metric] for r in cells_b]) - np.mean([r[metric] for r in cells_a])),
            "ci_low": float(np.quantile(delta_draw, 0.025)), "ci_high": float(np.quantile(delta_draw, 0.975)),
            "wins": sum(x > 0 for x in family_deltas), "ties": sum(x == 0 for x in family_deltas),
            "losses": sum(x < 0 for x in family_deltas), "worst_unit_delta": min(family_deltas),
            "contrast_semantics": "panel-composition diagnostic; shared Qwen cells plus independent Llama traces, not a row-paired treatment effect",
        })
    return output


def run(registry_path: Path, output: Path) -> None:
    registry = load_registry(registry_path)
    provenance = preflight(registry)
    source = next(resolve_path(r["path"]) for r in registry["frozen_sources"] if r["role"] == "build_a_decisions")
    rows = selected_rows(read_csv(source))
    cells_a, agg_a, draws_a = evaluate_panel(rows, QWEN_MODELS, S5A_VARIANT, BOOTSTRAP_SEED)
    cells_b, agg_b, draws_b = evaluate_panel(rows, FULL_MODELS, S5B_VARIANT, BOOTSTRAP_SEED)
    contrasts = composition_contrasts(cells_a, cells_b, draws_a, draws_b)
    metrics = cells_a + cells_b + agg_a + agg_b
    population_a = {"variant_id": S5A_VARIANT, "models": list(QWEN_MODELS), "n_cells": 8,
                    "n_rows": sum(r["model"] in QWEN_MODELS for r in rows), "n_unique_generated_traces": 6800,
                    "row_pairable_with_s4": False, "population_semantics": "current frozen Qwen generated-trace panel"}
    population_b = {"variant_id": S5B_VARIANT, "models": list(FULL_MODELS), "n_cells": 12,
                    "n_rows": len(rows), "n_unique_generated_traces": 10200,
                    "row_pairable_with_s4": False, "population_semantics": "current frozen Qwen plus Llama-3.1 generated-trace panel"}
    verification = {
        "state_id": STATE_ID, "status": "COMPLETE", "execution_registry_sha256": s1.sha256_file(registry_path),
        "runner_sha256": s1.sha256_file(Path(__file__).resolve()), "system_id": SYSTEM_ID,
        "access_contract_valid": True, "supervision_contract_valid": True, "source_mutation": False,
        "new_model_inference": False, "score_or_threshold_refit": False, "dual_build": provenance,
        "s5a": {"population": population_a, "aggregate": agg_a},
        "s5b": {"population": population_b, "aggregate": agg_b},
        "composition_contrasts": contrasts,
        "s4_adjacent_contrast_available": False,
        "s4_adjacent_contrast_reason": "historical and current panels contain different scorer models and generated traces; raw cumulative displacement is descriptive only",
        "prediction_flips_available": False,
        "prediction_flips_reason": "no shared row identity across the population bridge",
    }
    gates = [
        {"variant_id": S5A_VARIANT, "gate_id": "P0_S5A_DUAL_BUILD_IDENTICAL", "metric_id": "dual_build_identical", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "unit": "boolean", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5A_VARIANT, "gate_id": "P0_S5A_EXACT_QWEN_PANEL", "metric_id": "population_cells", "observed": "8", "threshold": "8", "direction": "eq", "passed": "true", "unit": "cells", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5A_VARIANT, "gate_id": "P0_S5A_ACCESS_VALID", "metric_id": "access_contract_valid", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "unit": "boolean", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5A_VARIANT, "gate_id": "P0_S5A_NO_REFIT_OR_INFERENCE", "metric_id": "new_fit_or_inference", "observed": "false", "threshold": "false", "direction": "eq", "passed": "true", "unit": "boolean", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5B_VARIANT, "gate_id": "P0_S5B_EXACT_FULL_PANEL", "metric_id": "population_cells", "observed": "12", "threshold": "12", "direction": "eq", "passed": "true", "unit": "cells", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5B_VARIANT, "gate_id": "P0_S5B_ACCESS_VALID", "metric_id": "access_contract_valid", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "unit": "boolean", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5B_VARIANT, "gate_id": "P0_S5B_NONPAIRED_BOUNDARY_DECLARED", "metric_id": "nonpaired_population_boundary", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "unit": "boolean", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"variant_id": S5B_VARIANT, "gate_id": "P0_S5_BOOTSTRAP_DRAWS", "metric_id": "bootstrap_draws", "observed": str(BOOTSTRAP_DRAWS), "threshold": str(BOOTSTRAP_DRAWS), "direction": "eq", "passed": "true", "unit": "draws", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    output.mkdir(parents=True, exist_ok=True)
    s1.write_csv(output / "P0_S5_METRICS.csv", metrics)
    s1.write_csv(output / "P0_S5_CONTRASTS.csv", contrasts)
    s1.write_json(output / "P0_S5A_POPULATION.json", population_a)
    s1.write_json(output / "P0_S5B_POPULATION.json", population_b)
    s1.write_json(output / "P0_S5_VERIFICATION.json", verification)
    s1.write_csv(output / "P0_S5_GATES.csv", gates)
    files = ["P0_S5_METRICS.csv", "P0_S5_CONTRASTS.csv", "P0_S5A_POPULATION.json",
             "P0_S5B_POPULATION.json", "P0_S5_VERIFICATION.json", "P0_S5_GATES.csv"]
    manifest = {"schema": "reasoning_localization_phase0_s5_run_manifest_v1", "state_id": STATE_ID,
                "execution_registry_sha256": s1.sha256_file(registry_path),
                "runner_sha256": s1.sha256_file(Path(__file__).resolve()), "python": sys.version,
                "platform": platform.platform(), "outputs": [{"file": f, "sha256": s1.sha256_file(output / f),
                "bytes": (output / f).stat().st_size} for f in files]}
    s1.write_json(output / "RUN_MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.registry.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
