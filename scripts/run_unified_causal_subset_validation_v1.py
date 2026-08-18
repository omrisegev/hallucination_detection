#!/usr/bin/env python3
"""Freeze selected subset-search finalists and score a scorer-model panel.

The script consumes an already completed development run.  It does not rerun
candidate selection and it never refits references, signs, IU/DUFS weights,
coordinate priors, or operating thresholds on the validation scorer.  The
ProcessBench questions and labels have been opened historically, so this is a
robustness/generalization analysis rather than untouched confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.run_unified_causal_iu_v1 import (  # noqa: E402
    _write_csv,
    _write_json,
    _write_jsonl,
    preflight,
)
from scripts.run_unified_causal_subset_search_v1 import (  # noqa: E402
    _choose_groups,
    _run_validation,
)
from spectral_utils.unified_causal_iu import (  # noqa: E402
    BASE_NAMES,
    base_matrix,
    parse_feature_name,
)


RUN_SCHEMA = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _macro(payload: Mapping[str, Any]) -> Mapping[str, float]:
    return payload["metrics"]["macro"]


def _validation_table(validation, control):
    reference = _macro(validation[control])
    rows = []
    for name, payload in validation.items():
        macro = _macro(payload)
        rows.append({
            "candidate": name,
            **{task: float(macro[task]) for task in ("global", "localization", "early")},
            **{
                f"delta_{task}": float(macro[task] - reference[task])
                for task in ("global", "localization", "early")
            },
        })
    rows.sort(key=lambda row: (
        -min(
            row["delta_global"] / 0.010,
            row["delta_localization"] / 0.010,
            row["delta_early"] / 0.015,
        ),
        row["candidate"],
    ))
    return rows


def _report(source_run, run_definition, table, validation) -> str:
    lines = [
        "# Unified Causal subset scorer-model validation v1",
        "",
        "Frozen transfer from Qwen3-8B-scored development telemetry to the complete "
        "Llama-3.1-8B-scored ProcessBench panel. This is robustness, not untouched confirmation.",
        "",
        f"- Source development run: `{source_run}`",
        f"- Development groups used for the frozen fit: {run_definition['development_groups']}",
        f"- Validation rows: {run_definition['validation_rows']}",
        f"- Validation groups: {run_definition['validation_groups']}",
        f"- Control: `{run_definition['control']}`",
        "",
        "| candidate | Global | ΔG | Localization | ΔL | Early | ΔE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['candidate']} | {row['global']:.4f} | {row['delta_global']:+.4f} | "
            f"{row['localization']:.4f} | {row['delta_localization']:+.4f} | "
            f"{row['early']:.4f} | {row['delta_early']:+.4f} |"
        )
    lines.extend([
        "",
        "No validation label entered feature selection, sign estimation, reference "
        "fitting, IU/DUFS fitting, alpha/lambda selection, or threshold calibration. "
        "The labels are used only to score this report.",
        "",
        "The panel is nevertheless not untouched: these ProcessBench questions and "
        "the Llama scorer cache have appeared in earlier repository analyses.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--validation-models", default="llama31_8b")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_definition_path = args.source_run / "RUN_DEFINITION.json"
    source_configs_path = args.source_run / "VARIANT_CONFIGS.json"
    if not source_definition_path.exists() or not source_configs_path.exists():
        raise FileNotFoundError("source run is missing RUN_DEFINITION.json or VARIANT_CONFIGS.json")
    source_definition = json.loads(source_definition_path.read_text())
    configs = json.loads(source_configs_path.read_text())
    candidates = _parse_names(args.candidates)
    if args.control not in candidates:
        candidates = (args.control, *candidates)
    unknown = set(candidates) - set(configs)
    if unknown:
        raise ValueError(f"candidates absent from source run: {sorted(unknown)}")
    rosters = {
        str(name): tuple(roster)
        for name, roster in source_definition["rosters"].items()
    }
    used_bases = {
        parse_feature_name(feature)[0]
        for candidate in candidates
        for feature in rosters[str(configs[candidate]["roster"])]
    }
    reference_names = tuple(name for name in BASE_NAMES if name in used_bases)
    if not reference_names:
        raise ValueError("selected candidates contain no causal base streams")

    models = tuple(source_definition["models"])
    families = tuple(source_definition["families"])
    validation_models = _parse_names(args.validation_models)
    inventory, loaded = preflight(args.data_root, models, families)
    development_rows, selection = _choose_groups(
        loaded,
        families,
        source_definition.get("questions_per_family"),
        int(source_definition["seed"]),
    )
    development_raw = [base_matrix(row, reference_names) for row in development_rows]
    validation_inventory, validation_rows = preflight(
        args.data_root, validation_models, families
    )
    fit_args = SimpleNamespace(
        positions_per_trace=int(source_definition["positions_per_trace"]),
        dufs_graph_k=int(source_definition["dufs_graph_k"]),
        dufs_seeds=tuple(int(value) for value in source_definition["dufs_seeds"]),
        dufs_epochs=int(source_definition["dufs_epochs"]),
    )
    run_definition = {
        "run_schema": RUN_SCHEMA,
        "source_run": str(args.source_run.resolve()),
        "source_run_definition_sha256": _sha256(source_definition_path),
        "source_variant_configs_sha256": _sha256(source_configs_path),
        "data_root": str(args.data_root.resolve()),
        "development_models": models,
        "validation_models": validation_models,
        "families": families,
        "selection": selection,
        "development_rows": len(development_rows),
        "development_groups": len({row["_source_group"] for row in development_rows}),
        "validation_rows": len(validation_rows),
        "validation_groups": len({row["_source_group"] for row in validation_rows}),
        "candidates": candidates,
        "control": args.control,
        "dufs_epochs": fit_args.dufs_epochs,
        "dufs_seeds": fit_args.dufs_seeds,
        "reference_names": reference_names,
        "inventory": inventory,
        "validation_inventory": validation_inventory,
        "claim_boundary": "opened scorer-model robustness; not untouched confirmation",
    }
    run_hash = hashlib.sha256(
        json.dumps(run_definition, sort_keys=True).encode("utf-8")
    ).hexdigest()
    run_definition["run_hash"] = run_hash
    args.out.mkdir(parents=True, exist_ok=True)
    definition_path = args.out / "RUN_DEFINITION.json"
    if definition_path.exists() and not args.force:
        existing = json.loads(definition_path.read_text())
        if existing.get("run_hash") != run_hash:
            raise RuntimeError("output contains a different validation run; use --force or a new path")
        if (args.out / "VALIDATION.json").exists():
            print(f"already complete: {args.out}")
            return
    _write_json(definition_path, run_definition)
    started = time.perf_counter()
    validation, records = _run_validation(
        fit_args,
        development_rows,
        development_raw,
        validation_rows,
        rosters,
        configs,
        candidates,
        reference_names=reference_names,
    )
    table = _validation_table(validation, args.control)
    run_definition["elapsed_seconds"] = time.perf_counter() - started
    _write_json(args.out / "VALIDATION.json", validation)
    _write_jsonl(args.out / "VALIDATION_RECORDS.jsonl", records)
    _write_csv(args.out / "VALIDATION_SUMMARY.csv", table)
    _write_json(definition_path, run_definition)
    (args.out / "REPORT.md").write_text(
        _report(args.source_run, run_definition, table, validation)
    )
    print(f"report={args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
