#!/usr/bin/env python3
"""Replay Phase-0 state S0 from the frozen Stage-4 checkpoints.

This is a CPU-only historical audit.  It reads the eight original checkpoint
payloads, reconstructs the Local per-question, cell, aggregate, and grouped
interval artifacts with the historical code path, and compares them with the
frozen Stage-4 outputs.  Source checkpoints and frozen results are never
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_local_online_comprehensive_stage4 import (  # noqa: E402
    BOOTSTRAP,
    FAMILIES,
    FINALIST,
    MODELS,
    PROTOCOL,
    PROTOCOL_SHA256,
    SEED,
    _aggregate,
    _grouped_interval,
)


STATE_ID = "P0_S0_HISTORICAL_REPLAY"
REFERENCE = "max_entropy__step_top5mean"
DEFAULT_OUTPUT = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "p0_s0_historical_replay"
DEFAULT_REGISTRY = REPO / "results" / "reasoning_localization_03662_v1" / "phase_0" / "P0_S0_EXECUTION_REGISTRY.json"
REQUIRED_FROZEN = (
    "STAGE_4_LOCAL_PER_QUESTION.csv",
    "STAGE_4_CELL_METRICS.csv",
    "STAGE_4_AGGREGATE.csv",
    "STAGE_4_INTERVALS.csv",
    "STAGE_4_DECISION.json",
)


class ReplayError(RuntimeError):
    """Raised when the frozen replay contract does not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ReplayError(f"refusing to write empty replay table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        normalized.append({
            str(key): str(value)
            for key, value in row.items()
            if value is not None and str(value) != ""
        })
    return normalized


def checkpoint_name(model: str, family: str) -> str:
    return f"{model}__{family}.pkl"


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ReplayError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ReplayError(f"{label} SHA mismatch: {path}\nexpected={expected}\nobserved={observed}")


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("state_id") != STATE_ID:
        raise ReplayError("execution registry state_id mismatch")
    if registry.get("status") != "FROZEN_BEFORE_RUN":
        raise ReplayError("execution registry must be FROZEN_BEFORE_RUN")
    require_hash(Path(__file__).resolve(), registry["runner_sha256"], "frozen runner")
    require_hash(PROTOCOL, registry["protocol_sha256"], "frozen protocol")
    if registry["protocol_sha256"] != PROTOCOL_SHA256:
        raise ReplayError("registry protocol hash differs from historical code gate")
    if registry.get("bootstrap_draws") != BOOTSTRAP or registry.get("seed") != SEED:
        raise ReplayError("registry does not match historical bootstrap/seed")
    return registry


def preflight_sources(
    registry: Mapping[str, Any], checkpoint_root: Path, frozen_results_root: Path
) -> dict[str, Any]:
    checkpoint_specs = {row["file"]: row for row in registry["checkpoints"]}
    expected_names = {checkpoint_name(model, family) for model in MODELS for family in FAMILIES}
    if set(checkpoint_specs) != expected_names:
        raise ReplayError("checkpoint roster differs from the exact eight Stage-4 cells")
    sources = []
    for name in sorted(expected_names):
        path = checkpoint_root / name
        require_hash(path, checkpoint_specs[name]["sha256"], f"checkpoint {name}")
        sources.append({"role": "stage4_checkpoint", "file": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    frozen_specs = {row["file"]: row for row in registry["frozen_results"]}
    if set(frozen_specs) != set(REQUIRED_FROZEN):
        raise ReplayError("frozen-result roster mismatch")
    for name in REQUIRED_FROZEN:
        path = frozen_results_root / name
        require_hash(path, frozen_specs[name]["sha256"], f"frozen result {name}")
        sources.append({"role": "frozen_stage4_result", "file": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"sources": sources, "sources_sha256": canonical_sha256(sources)}


def load_checkpoints(checkpoint_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    local_records: list[dict[str, Any]] = []
    local_metrics: list[dict[str, Any]] = []
    population_cells = []
    family_units: dict[str, tuple[str, ...]] = {}
    for model in MODELS:
        for family in FAMILIES:
            path = checkpoint_root / checkpoint_name(model, family)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            records = list(payload.get("local_records", []))
            metrics = list(payload.get("local_metrics", []))
            if not records or not metrics:
                raise ReplayError(f"empty Local payload in {path.name}")
            for row in records:
                if row.get("model") != model or row.get("family") != family or row.get("task") != "local":
                    raise ReplayError(f"cell identity mismatch in {path.name}")
            finalists = [row for row in records if row["candidate"] == FINALIST]
            units = tuple(sorted(row["unit"] for row in finalists))
            if len(units) != len(set(units)):
                raise ReplayError(f"duplicate finalist units in {path.name}")
            if family in family_units and family_units[family] != units:
                raise ReplayError(f"source-question identity mismatch across scorer copies for {family}")
            family_units.setdefault(family, units)
            population_cells.append({
                "model": model,
                "family": family,
                "n_scorer_rows": len(units),
                "unit_sha256": hashlib.sha256("\n".join(units).encode("utf-8")).hexdigest(),
                "checkpoint_file": path.name,
                "checkpoint_sha256": sha256_file(path),
            })
            local_records.extend(records)
            local_metrics.extend(metrics)
    group_keys = [f"{family}|{unit}" for family in FAMILIES for unit in family_units[family]]
    population = {
        "schema": "reasoning_localization_p0_population_v1",
        "population_id": "historical_stage4_eight_cell_audit",
        "models": list(MODELS),
        "families": list(FAMILIES),
        "cells": population_cells,
        "n_cells": len(population_cells),
        "n_scorer_rows": sum(row["n_scorer_rows"] for row in population_cells),
        "n_source_question_groups": len(group_keys),
        "source_question_group_sha256": hashlib.sha256("\n".join(group_keys).encode("utf-8")).hexdigest(),
        "calibration_role": "historical deterministic 40 percent per family",
        "evaluation_role": "historical deterministic 20 percent audit per family",
        "scorer_copies_grouped": True,
    }
    if population["n_cells"] != 8 or population["n_scorer_rows"] != 1270 or population["n_source_question_groups"] != 635:
        raise ReplayError(f"unexpected historical population counts: {population}")
    return local_records, local_metrics, population


def reconstruct(local_records: list[dict[str, Any]], local_metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = _aggregate(local_metrics, "local")
    tier_a = [row for row in aggregate if row["candidate"] != FINALIST and row["access_tier"] == "A"]
    reference = max(tier_a, key=lambda row: row["primary"])["candidate"]
    if reference != REFERENCE:
        raise ReplayError(f"unexpected strongest historical Tier-A reference: {reference}")
    intervals = []
    candidates = [row["candidate"] for row in tier_a] + [FINALIST]
    for candidate in sorted(set(candidates)):
        if candidate == reference:
            continue
        delta, low, high, wins, losses = _grouped_interval(local_records, "local", candidate, reference)
        intervals.append({
            "candidate": candidate,
            "task": "local",
            "reference": reference,
            "delta": delta,
            "ci_low": low,
            "ci_high": high,
            "family_wins": wins,
            "family_losses": losses,
        })
    return aggregate, intervals


def compare_frozen(
    output: Path,
    frozen_results_root: Path,
    local_records: list[dict[str, Any]],
    local_metrics: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    per_question_path = output / "P0_S0_LOCAL_PER_QUESTION.csv"
    cell_path = output / "P0_S0_LOCAL_CELL_METRICS.csv"
    aggregate_path = output / "P0_S0_LOCAL_AGGREGATE.csv"
    intervals_path = output / "P0_S0_LOCAL_INTERVALS.csv"
    write_csv(per_question_path, local_records)
    write_csv(cell_path, local_metrics)
    write_csv(aggregate_path, aggregate)
    write_csv(intervals_path, intervals)

    frozen_per_question = frozen_results_root / "STAGE_4_LOCAL_PER_QUESTION.csv"
    byte_exact_per_question = per_question_path.read_bytes() == frozen_per_question.read_bytes()
    source_cell = [row for row in read_csv(frozen_results_root / "STAGE_4_CELL_METRICS.csv") if row["task"] == "local"]
    source_aggregate = [row for row in read_csv(frozen_results_root / "STAGE_4_AGGREGATE.csv") if row["task"] == "local"]
    source_intervals = [row for row in read_csv(frozen_results_root / "STAGE_4_INTERVALS.csv") if row["task"] == "local"]
    cell_exact = normalized_rows(read_csv(cell_path)) == normalized_rows(source_cell)
    aggregate_exact = normalized_rows(read_csv(aggregate_path)) == normalized_rows(source_aggregate)
    intervals_exact = normalized_rows(read_csv(intervals_path)) == normalized_rows(source_intervals)
    checks = {
        "per_question_byte_exact": byte_exact_per_question,
        "cell_metrics_semantic_exact": cell_exact,
        "aggregate_semantic_exact": aggregate_exact,
        "intervals_semantic_exact": intervals_exact,
        "replay_per_question_sha256": sha256_file(per_question_path),
        "frozen_per_question_sha256": sha256_file(frozen_per_question),
    }
    if not all(checks[key] for key in (
        "per_question_byte_exact", "cell_metrics_semantic_exact",
        "aggregate_semantic_exact", "intervals_semantic_exact",
    )):
        raise ReplayError(f"historical replay is not checksum-equivalent: {checks}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--frozen-results-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry.resolve())
    source_manifest = preflight_sources(registry, args.checkpoint_root.resolve(), args.frozen_results_root.resolve())
    if args.preflight_only:
        print(json.dumps({"state_id": STATE_ID, "status": "PREFLIGHT_PASS", **source_manifest}, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ReplayError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    local_records, local_metrics, population = load_checkpoints(args.checkpoint_root.resolve())
    aggregate, intervals = reconstruct(local_records, local_metrics)
    checks = compare_frozen(output, args.frozen_results_root.resolve(), local_records, local_metrics, aggregate, intervals)
    write_json(output / "P0_S0_POPULATION.json", population)

    finalist = next(row for row in aggregate if row["candidate"] == FINALIST)
    entropy = next(row for row in aggregate if row["candidate"] == REFERENCE)
    finalist_interval = next(row for row in intervals if row["candidate"] == FINALIST)
    metrics_rows = [{
        "variant_id": "R2_HISTORICAL_FAMILY6_BRIDGE",
        "metric_id": "macro_f1",
        "value": finalist["primary"],
        "ci_low": "",
        "ci_high": "",
        "n_rows": population["n_scorer_rows"],
        "n_groups": population["n_source_question_groups"],
        "population_sha256": population["source_question_group_sha256"],
        "status": "COMPLETE",
        "evidence_status": "RETROSPECTIVE",
    }]
    gates_rows = [
        {"gate_id": "P0_S0_CHECKSUM_EQUIVALENT", "observed": "true", "threshold": "true", "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S0_EXACT_EIGHT_CELL_POPULATION", "observed": population["n_cells"], "threshold": 8, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
        {"gate_id": "P0_S0_FROZEN_PROTOCOL_SHA", "observed": registry["protocol_sha256"], "threshold": PROTOCOL_SHA256, "direction": "eq", "passed": "true", "status": "COMPLETE", "evidence_status": "RETROSPECTIVE"},
    ]
    write_csv(output / "P0_S0_METRICS.csv", metrics_rows)
    write_csv(output / "P0_S0_GATES.csv", gates_rows)
    verification = {
        "schema": "reasoning_localization_p0_s0_verification_v1",
        "state_id": STATE_ID,
        "status": "CHECKSUM_EQUIVALENT",
        "execution": "historical_checkpoint_replay_no_new_inference",
        "finalist": FINALIST,
        "reference": REFERENCE,
        "finalist_macro_f1": finalist["primary"],
        "reference_macro_f1": entropy["primary"],
        "delta": finalist_interval["delta"],
        "ci_low": finalist_interval["ci_low"],
        "ci_high": finalist_interval["ci_high"],
        "family_wins": finalist_interval["family_wins"],
        "family_losses": finalist_interval["family_losses"],
        "checks": checks,
        "population_sha256": population["source_question_group_sha256"],
        "source_manifest": source_manifest,
    }
    write_json(output / "P0_S0_VERIFICATION.json", verification)
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_manifest = {
        "schema": "reasoning_localization_p0_s0_run_manifest_v1",
        "state_id": STATE_ID,
        "status": "COMPLETE",
        "source_commit": git_head(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "execution_registry_sha256": sha256_file(args.registry.resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "new_inference": False,
        "gpu_hours": 0,
        "source_mutation": False,
        "outputs": [
            {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    write_json(output / "RUN_MANIFEST.json", run_manifest)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
