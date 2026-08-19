#!/usr/bin/env python3
"""Access-matched Global comparison with the classic mixed-v2 IU-PCR head.

The historical contract contains final response length.  This replay removes
that one non-causal coordinate, then refits the remaining registered mixed-v2
head inside the exact repeated Qwen splits used by a completed subset run.  It
also fits one label-free head per Qwen family and transfers it, frozen, to the
matching complete Llama scorer panel.  Labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.run_unified_causal_iu_v1 import (  # noqa: E402
    _evaluate_live_baselines,
    _grouped_splits,
    _record_primary_metrics,
    _weighted_bootstrap_auc,
    _write_csv,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_comparisons,
    preflight,
)
from scripts.run_unified_causal_subset_search_v1 import _choose_groups  # noqa: E402
from spectral_utils.historical_multitask_baselines import (  # noqa: E402
    fit_registered_global,
)
from spectral_utils.unified_causal_evaluation import final_wrong, safe_auc  # noqa: E402


RUN_SCHEMA = 1
TASK = "global"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family_macro(records: Sequence[Mapping[str, Any]], score_key: str) -> float:
    values = []
    for family in sorted({str(row["family"]) for row in records}):
        subset = [row for row in records if str(row["family"]) == family]
        values.append(safe_auc(
            [int(row["wrong"]) for row in subset],
            [float(row[score_key]) for row in subset],
        ))
    finite = np.asarray(values, dtype=float)
    return float(np.nanmean(finite))


def _development_replay(
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    base7_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeats = int(definition["repeats"])
    folds = int(definition["folds"])
    seed = int(definition["seed"])
    families = tuple(str(value) for value in definition["families"])
    output, summary = [], []
    for repeat in range(repeats):
        repeat_records = []
        fold_summary = []
        splits = _grouped_splits(rows, folds, seed + 1009 * repeat)
        for fold, (fit_indices, test_indices) in enumerate(splits):
            fit_rows = [rows[index] for index in fit_indices]
            test_rows = [rows[index] for index in test_indices]
            fold_records = []
            for family in families:
                fit_cell = [row for row in fit_rows if str(row["family"]) == family]
                test_cell = [row for row in test_rows if str(row["family"]) == family]
                if not fit_cell or not test_cell:
                    raise RuntimeError(f"empty classic replay cell: repeat={repeat} fold={fold} {family}")
                model = fit_registered_global(fit_cell)
                for row in test_cell:
                    record = {
                        "candidate": "classic_mixed_v2_no_length",
                        "repeat": repeat,
                        "fold": fold,
                        "family": family,
                        "model": str(row["model"]),
                        "source_group": str(row["_source_group"]),
                        "unit": str(row["_unit"]),
                        "wrong": final_wrong(row),
                        "global_score": float(model.score(row)),
                        "retained_features": len(model.names),
                    }
                    output.append(record)
                    repeat_records.append(record)
                    fold_records.append(record)
            test_groups = {str(row["_source_group"]) for row in test_rows}
            base_fold = [
                row for row in base7_records
                if int(row["repeat"]) == repeat
                and str(row["source_group"]) in test_groups
            ]
            if {str(row["source_group"]) for row in base_fold} != test_groups:
                raise AssertionError(
                    f"development fold pairing mismatch in repeat {repeat}, fold {fold}"
                )
            fold_summary.append({
                "base7_full28": _family_macro(base_fold, "global_score"),
                "classic_mixed_v2_no_length": _family_macro(
                    fold_records, "global_score"
                ),
            })
        base_repeat = [
            row for row in base7_records
            if int(row["repeat"]) == repeat
        ]
        if {str(row["source_group"]) for row in base_repeat} != {
            str(row["source_group"]) for row in repeat_records
        }:
            raise AssertionError(f"development pairing mismatch in repeat {repeat}")
        summary.append({
            "repeat": repeat,
            "base7_full28": float(np.mean([
                row["base7_full28"] for row in fold_summary
            ])),
            "classic_mixed_v2_no_length": float(np.mean([
                row["classic_mixed_v2_no_length"] for row in fold_summary
            ])),
        })
    for row in summary:
        row["delta_base7_minus_classic"] = (
            row["base7_full28"] - row["classic_mixed_v2_no_length"]
        )
    return output, summary


def _bootstrap_transfer(
    records: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    point_base = _family_macro(records, "base7_full28")
    point_classic = _family_macro(records, "classic_mixed_v2_no_length")
    rng = np.random.default_rng(seed)
    repeats = 2000
    family_deltas = []
    for family in families:
        subset = [row for row in records if str(row["family"]) == family]
        groups = [str(row["source_group"]) for row in subset]
        if len(groups) != len(set(groups)):
            raise AssertionError(f"validation family has repeated groups: {family}")
        weights = rng.multinomial(
            len(subset), np.full(len(subset), 1.0 / len(subset)), size=repeats
        )
        labels = np.asarray([row["wrong"] for row in subset], dtype=int)
        base_auc = _weighted_bootstrap_auc(
            labels,
            np.asarray([row["base7_full28"] for row in subset], dtype=float),
            weights,
        )
        classic_auc = _weighted_bootstrap_auc(
            labels,
            np.asarray([
                row["classic_mixed_v2_no_length"] for row in subset
            ], dtype=float),
            weights,
        )
        family_deltas.append(base_auc - classic_auc)
    distribution = np.nanmean(np.vstack(family_deltas), axis=0)
    finite = distribution[np.isfinite(distribution)]
    return {
        "repeats": repeats,
        "unit": "source question, resampled within family",
        "groups": len({str(row["source_group"]) for row in records}),
        "base7_full28": point_base,
        "classic_mixed_v2_no_length": point_classic,
        "delta_base7_minus_classic": point_base - point_classic,
        "ci95": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "valid_replicates": len(finite),
    }


def _frozen_transfer(
    development_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    base7_records: Sequence[Mapping[str, Any]],
    families: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_lookup = {
        str(row["source_group"]): row
        for row in base7_records
    }
    if len(base_lookup) != len(base7_records):
        raise AssertionError("base7 validation has duplicate source groups")
    output = []
    feature_counts = {}
    for family in families:
        fit_cell = [row for row in development_rows if str(row["family"]) == family]
        test_cell = [row for row in validation_rows if str(row["family"]) == family]
        model = fit_registered_global(fit_cell)
        feature_counts[family] = len(model.names)
        for row in test_cell:
            group = str(row["_source_group"])
            if group not in base_lookup:
                raise AssertionError(f"missing base7 validation group: {group}")
            base = base_lookup[group]
            if int(base["wrong"]) != final_wrong(row):
                raise AssertionError(f"label mismatch for {group}")
            output.append({
                "family": family,
                "source_group": group,
                "unit": str(row["_unit"]),
                "wrong": final_wrong(row),
                "base7_full28": float(base["global_score"]),
                "classic_mixed_v2_no_length": float(model.score(row)),
            })
    if len(output) != len(validation_rows):
        raise AssertionError("classic transfer did not score every validation row")
    development_groups = {
        str(row["_source_group"]) for row in development_rows
    }
    disjoint = [
        row for row in output
        if str(row["source_group"]) not in development_groups
    ]
    if len(disjoint) != len(output) - len(development_groups):
        raise AssertionError("unexpected Qwen/Llama source-group overlap")
    return output, {
        "full_panel": _bootstrap_transfer(output, families, seed=20260818),
        "question_disjoint": _bootstrap_transfer(disjoint, families, seed=20260819),
        "retained_features_by_family": feature_counts,
        "excluded_seen_development_groups": len(development_groups),
    }


def _taskwise_incumbent_comparison(
    development_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    base7_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare Local/Early with their access-matched live incumbents."""

    development_model = str(development_rows[0]["model"])
    validation_alias = [
        {**row, "model": development_model}
        for row in validation_rows
    ]
    live = _evaluate_live_baselines(development_rows, validation_alias)
    unit_to_group = {
        str(row["_unit"]): str(row["_source_group"])
        for row in validation_rows
    }
    selected = [
        {**row, "candidate": "base7_full28", "outer_fold": 0}
        for row in base7_records
    ]
    records = list(selected)
    for name in ("max_entropy_top5", "iu28_without_length"):
        for row in live[name]["per_question"]:
            unit = str(row["unit"])
            if unit not in unit_to_group:
                raise AssertionError(f"missing validation source group for {unit}")
            records.append({
                **row,
                "candidate": name,
                "source_group": unit_to_group[unit],
                "outer_fold": 0,
            })
    groups = {
        candidate: {
            str(row["source_group"])
            for row in records if row["candidate"] == candidate
        }
        for candidate in ("base7_full28", "max_entropy_top5", "iu28_without_length")
    }
    if not (
        groups["base7_full28"]
        == groups["max_entropy_top5"]
        == groups["iu28_without_length"]
    ):
        raise AssertionError("taskwise incumbent records are not question-paired")
    full = grouped_bootstrap_comparisons(
        records,
        repeats=2000,
        seed=20260820,
        primary_candidate="base7_full28",
    )
    development_groups = {
        str(row["_source_group"]) for row in development_rows
    }
    disjoint_records = [
        row for row in records
        if str(row["source_group"]) not in development_groups
    ]
    disjoint = grouped_bootstrap_comparisons(
        disjoint_records,
        repeats=2000,
        seed=20260821,
        primary_candidate="base7_full28",
    )
    return {
        "full_panel": full,
        "question_disjoint": disjoint,
        "taskwise_incumbents": {
            "localization": "max_entropy_top5",
            "early": "max_entropy_top5",
        },
        "base7_metrics": _record_primary_metrics(selected),
        "live_metrics": {
            name: dict(live[name]["metrics"]["macro"])
            for name in ("max_entropy_top5", "iu28_without_length")
        },
        "excluded_seen_development_groups": len(development_groups),
    }


def _report(definition, development, transfer, taskwise) -> str:
    dev_base = float(np.mean([row["base7_full28"] for row in development]))
    dev_classic = float(np.mean([
        row["classic_mixed_v2_no_length"] for row in development
    ]))
    full = transfer["full_panel"]
    disjoint = transfer["question_disjoint"]
    local = taskwise["full_panel"]["comparisons"]["max_entropy_top5"]["localization"]
    early = taskwise["full_panel"]["comparisons"]["max_entropy_top5"]["early"]
    early_iu28 = taskwise["full_panel"]["comparisons"]["iu28_without_length"]["early"]
    local_points = taskwise["full_panel"]["point_by_method"]
    early_points = taskwise["full_panel"]["point_by_method"]
    return "\n".join([
        "# Unified Causal versus classic Global IU-PCR v1",
        "",
        "This is the access-matched comparison that was missing from the feature-subset cycle. "
        "The classic registered mixed-v2 contract is refit without final response length, so "
        "both methods are causal with respect to length. The primary comparison here is Global; "
        "companion live Local/Early incumbents are reported below.",
        "",
        "## Repeated Qwen3-8B development splits",
        "",
        f"- base7_full28: {dev_base:.4f}",
        f"- classic_mixed_v2_no_length: {dev_classic:.4f}",
        f"- delta: {dev_base - dev_classic:+.4f}",
        "",
        "Both methods use the exact 3 x 3 source-question-grouped split schedule. The classic "
        "head is refit separately inside each fit family and sees no labels.",
        "",
        "## Frozen Qwen-to-Llama scorer transfer",
        "",
        f"- base7_full28: {full['base7_full28']:.4f}",
        f"- classic_mixed_v2_no_length: {full['classic_mixed_v2_no_length']:.4f}",
        f"- delta: {full['delta_base7_minus_classic']:+.4f} "
        f"[95% CI {full['ci95'][0]:+.4f}, {full['ci95'][1]:+.4f}]",
        "",
        f"After excluding the {transfer['excluded_seen_development_groups']} source questions "
        "used to fit the Qwen heads, the question-disjoint panel has "
        f"{disjoint['groups']} questions. Its delta is "
        f"{disjoint['delta_base7_minus_classic']:+.4f} "
        f"[{disjoint['ci95'][0]:+.4f}, {disjoint['ci95'][1]:+.4f}].",
        "",
        "The classic head is fit on the same 32 selected Qwen questions per family and then "
        "frozen before all 3,400 Llama questions are scored. Validation labels affect only "
        "the metric and paired bootstrap.",
        "",
        "## Taskwise live incumbents on the same frozen transfer",
        "",
        "| task | base7_full28 | incumbent | incumbent score | delta [95% CI] |",
        "|---|---:|---|---:|---:|",
        f"| Global | {full['base7_full28']:.4f} | classic mixed-v2, no length | "
        f"{full['classic_mixed_v2_no_length']:.4f} | "
        f"{full['delta_base7_minus_classic']:+.4f} "
        f"[{full['ci95'][0]:+.4f}, {full['ci95'][1]:+.4f}] |",
        f"| Localization | {local_points['base7_full28']['localization']:.4f} | "
        f"max entropy + top-5 step | "
        f"{local_points['max_entropy_top5']['localization']:.4f} | "
        f"{local['delta']:+.4f} [{local['ci95'][0]:+.4f}, {local['ci95'][1]:+.4f}] |",
        f"| Early | {early_points['base7_full28']['early']:.4f} | max entropy | "
        f"{early_points['max_entropy_top5']['early']:.4f} | "
        f"{early['delta']:+.4f} [{early['ci95'][0]:+.4f}, {early['ci95'][1]:+.4f}] |",
        "",
        "Against the historical IU28-without-length Early control alone, base7 is "
        f"{early_iu28['delta']:+.4f} [{early_iu28['ci95'][0]:+.4f}, "
        f"{early_iu28['ci95'][1]:+.4f}]; max entropy is nevertheless the stronger "
        "Early incumbent in this exact transfer protocol.",
        "",
        "## Interpretation",
        "",
        "A compact unified causal head improves Localization relative to the live baseline, "
        "but it does not clear the frozen noninferiority margins against the strongest Global "
        "and Early incumbents. It must not be promoted as one replacement for all three heads. "
        "The exact historical 30-coordinate Global method also contains final length and "
        "therefore has strictly greater end-of-trace access.",
        "",
        f"Source subset run: `{definition['source_run']}`.",
        "Retrospective opened-data comparison; not untouched confirmation.",
        "",
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run", type=Path,
        default=ROOT / "results/unified_causal_subset_search_compact_v1",
    )
    parser.add_argument(
        "--validation-run", type=Path,
        default=ROOT / "results/unified_causal_subset_validation_compact_llama31_v1",
    )
    parser.add_argument(
        "--data-root", type=Path,
        help="cache root; defaults to the source run's recorded data_root",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results/unified_causal_subset_classic30_v1",
    )
    parser.add_argument("--candidate", default="base7_full28")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_definition_path = args.source_run / "RUN_DEFINITION.json"
    source_oof_path = args.source_run / "OOF_RECORDS.jsonl"
    validation_records_path = args.validation_run / "VALIDATION_RECORDS.jsonl"
    definition = json.loads(source_definition_path.read_text())
    if int(definition["run_schema"]) != 1:
        raise RuntimeError("unsupported source subset run schema")
    models = tuple(str(value) for value in definition["models"])
    families = tuple(str(value) for value in definition["families"])
    data_root = args.data_root or Path(str(definition["data_root"]))
    inventory, loaded = preflight(data_root, models, families)
    development_rows, selection = _choose_groups(
        loaded,
        families,
        int(definition["questions_per_family"]),
        int(definition["seed"]),
    )
    validation_models = ("llama31_8b",)
    validation_inventory, validation_rows = preflight(
        data_root, validation_models, families
    )
    base7_oof = [
        row for row in _read_jsonl(source_oof_path)
        if str(row["candidate"]) == args.candidate
    ]
    base7_validation = [
        row for row in _read_jsonl(validation_records_path)
        if str(row["candidate"]) == args.candidate
    ]
    classic_development, development_summary = _development_replay(
        definition, development_rows, base7_oof
    )
    transfer_records, transfer = _frozen_transfer(
        development_rows, validation_rows, base7_validation, families
    )
    taskwise = _taskwise_incumbent_comparison(
        development_rows, validation_rows, base7_validation
    )
    run_definition = {
        "run_schema": RUN_SCHEMA,
        "source_run": str(args.source_run.resolve()),
        "source_run_sha256": _sha256(source_definition_path),
        "source_validation_run": str(args.validation_run.resolve()),
        "source_validation_records_sha256": _sha256(validation_records_path),
        "candidate": args.candidate,
        "models": models,
        "validation_models": validation_models,
        "families": families,
        "selection": selection,
        "development_groups": len({row["_source_group"] for row in development_rows}),
        "validation_groups": len({row["_source_group"] for row in validation_rows}),
        "classic_contract": "registered mixed-v2 30-feature contract with final length excluded",
        "classic_labels_seen_during_fit": False,
        "claim_boundary": "retrospective opened-data comparison; not untouched confirmation",
        "inventory": inventory,
        "validation_inventory": validation_inventory,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "RUN_DEFINITION.json", run_definition)
    _write_jsonl(args.out / "DEVELOPMENT_CLASSIC_RECORDS.jsonl", classic_development)
    _write_csv(args.out / "DEVELOPMENT_COMPARISON.csv", development_summary)
    _write_jsonl(args.out / "LLAMA_GLOBAL_RECORDS.jsonl", transfer_records)
    _write_json(args.out / "LLAMA_GLOBAL_BOOTSTRAP.json", transfer)
    _write_json(args.out / "TASKWISE_INCUMBENT_BOOTSTRAP.json", taskwise)
    report_definition = {**run_definition, "source_run": args.source_run}
    (args.out / "REPORT.md").write_text(
        _report(report_definition, development_summary, transfer, taskwise)
    )
    print(f"report={args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
