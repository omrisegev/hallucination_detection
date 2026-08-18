#!/usr/bin/env python3
"""Build Fair Paper-Exact Comparison Package v1 from frozen local artifacts.

The builder is CPU-only.  It never invokes a model, a cluster scheduler, or a
Google Drive mutation.  Drive acquisition is a separate, explicit step; this
script consumes only the verified local cache and previously frozen repository
artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import pickle
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils.fair_comparisons import PACKAGE_REVISION  # noqa: E402
from spectral_utils.fair_comparisons.evaluator import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    auroc,
    calibrate_prefix_ever_warning_thresholds,
    paired_grouped_bootstrap,
    prefix_warning_metrics,
    summary_dict as evaluator_summary,
)
from spectral_utils.fair_comparisons.drive_snapshot import (  # noqa: E402
    L0_INVENTORY,
    L1_METADATA,
    M2_STATUS_METADATA,
    S1_METADATA,
    S2_COMPLETE_METADATA,
    build_drive_metadata_observation,
)
from spectral_utils.fair_comparisons.folds import (  # noqa: E402
    assign_group_folds,
    fold_assignment_sha256,
)
from spectral_utils.fair_comparisons.global_lane import (  # noqa: E402
    MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
    bootstrap_global_contrasts,
    evaluate_global_panel,
    replay_registered_dufs_no_length,
)
from spectral_utils.fair_comparisons.localization import (  # noqa: E402
    DEDICATED_METHOD_ID,
    GL_LIU_METHOD_ID,
    MAX_ENTROPY_METHOD_ID,
    MIND_GAP_METHOD_ID,
    UNIFIED28_METHOD_ID,
    bootstrap_localization_contrasts,
    crossfit_score_method,
    evaluate_fixed_prediction_method,
    load_unified28_prethreshold_rows,
    native_mind_gap_metrics,
    replay_same_access_methods,
)
from spectral_utils.fair_comparisons.package_io import (  # noqa: E402
    indexed_pickle_rows,
    tree_manifest,
    write_json,
    write_jsonl,
    write_long_csv,
)
from spectral_utils.fair_comparisons.processbench import (  # noqa: E402
    PROCESSBENCH_DATASET_REVISION,
    PROCESSBENCH_SUBSETS,
    ProcessBenchPopulation,
    adapt_eq6_shard_records,
    adapt_external_localization_records,
    adapt_unified_global_records,
    adapt_unified_validation_records,
    build_processbench_population,
    load_pickle_bundle,
    sha256_file,
)
from spectral_utils.fair_comparisons.prefix import (  # noqa: E402
    HISTORICAL_DEEPCONF_METHOD_ID as PREFIX_HISTORICAL_DEEPCONF_METHOD_ID,
    IU28_METHOD_ID as PREFIX_IU28_METHOD_ID,
    MAX_ENTROPY_METHOD_ID as PREFIX_MAX_ENTROPY_METHOD_ID,
    MEAN_ENTROPY_METHOD_ID as PREFIX_MEAN_ENTROPY_METHOD_ID,
    PREFIX_BUDGETS,
    SELECTED_STEP272_ARCHITECTURE,
    STEP272_METHOD_ID,
    assemble_historical_common_panel,
    audit_s2_cot_telemetry,
    build_entropy_prefix_records,
    build_warning_inputs,
    load_historical_prefix_scores,
    load_step272_prefix_records,
    load_unified28_prefix_records,
    replay_frozen_prefix_incumbents,
    summarize_prefix_metrics,
)
from spectral_utils.fair_comparisons.registry import (  # noqa: E402
    build_asset_registry,
    build_hash_manifest,
    build_method_registry,
    build_population_registry,
    canonical_sha256,
    canonicalize_comparison_records,
    make_asset_record,
    make_comparison_record,
    make_derived_asset_record,
    make_eligible_population,
    make_method_entry,
    make_population_entry,
    ordered_id_sha256,
    population_index,
    audit_comparison_records,
    require_clean_join,
    validate_asset_record,
    verify_hash_manifest,
    write_canonical_json,
)
from spectral_utils.fair_comparisons.reporting import write_reports  # noqa: E402
from spectral_utils.fair_comparisons.stopping import (  # noqa: E402
    S2_ARMS,
    S2_SETTING,
    build_s2_stopping_lane,
    canonical_s2_group_id,
    canonical_s2_id,
)
from spectral_utils.fair_comparisons.twentyfour import (  # noqa: E402
    BLOCKED_CELL as TWENTYFOUR_BLOCKED_CELL,
    POPULATION_ID as TWENTYFOUR_POPULATION_ID,
    partial_identity_audit as twentyfour_partial_identity_audit,
    static_preflight as twentyfour_static_preflight,
)


OUT_DEFAULT = ROOT / "results" / "fair_paper_exact_comparisons_v1"
CACHE_DEFAULT = ROOT / "local_cache" / "fair_paper_exact_comparisons_v1"
TWENTYFOUR_SOURCE_DEFAULT = Path("/private/tmp/unified_24cell_raw")
TWENTYFOUR_SOURCE_TOKEN = "${UNIFIED_24CELL_RAW_ROOT}"
UNIFIED_VALIDATION = (
    ROOT
    / "results"
    / "unified_causal_subset_validation_base7_dufs_llama31_v1"
    / "VALIDATION_RECORDS.jsonl"
)
UNIFIED_GLOBAL = (
    ROOT
    / "results"
    / "unified_causal_subset_classic30_v1"
    / "LLAMA_GLOBAL_RECORDS.jsonl"
)
CLASSIC_GLOBAL_FIT_IDS = (
    ROOT
    / "results"
    / "unified_causal_subset_classic30_v1"
    / "DEVELOPMENT_CLASSIC_RECORDS.jsonl"
)
CLASSIC_GLOBAL_RUN_DEFINITION = CLASSIC_GLOBAL_FIT_IDS.parent / "RUN_DEFINITION.json"
DUFS_ANCHOR_MANIFEST = (
    ROOT / "results" / "processbench_latent_state_v1" / "FREEZE_MANIFEST.json"
)
PROTOCOL = ROOT / "docs" / "experiments" / "FAIR_PAPER_EXACT_COMPARISONS_V1.md"
PREFIX_STEP272 = ROOT / "results" / "global_local_online_architecture_v2" / "ARCHITECTURE_PER_QUESTION.csv"
PREFIX_HISTORICAL = ROOT / "results" / "global_local_online_iu_v1" / "PER_QUESTION_SCORES.csv"
PRMBENCH_METRICS = ROOT / "results" / "fixed_application_pipelines_v1" / "reasoning_metrics.csv"
PRMBENCH_FREEZE = ROOT / "results" / "fixed_application_pipelines_v1" / "prmbench_score_freeze.json"
PRMBENCH_DIAGNOSTICS = ROOT / "results" / "fixed_application_pipelines_v1" / "reasoning_diagnostics.json"
PRMBENCH_TELEMETRY = ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen3_8b_telemetry_full" / "prmbench_telemetry.pkl"
PRMBENCH_PRM = ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen25math7b_full" / "prmbench_prm.pkl"
RUNTIME_CODE_PATHS = (
    "scripts",
    "spectral_utils",
    "docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md",
)

PB_GLOBAL_POPULATION = f"{PROCESSBENCH_DATASET_REVISION}::llama31_8b::global"
PB_LOCALIZATION_POPULATION = f"{PROCESSBENCH_DATASET_REVISION}::llama31_8b::localization"
PB_PREFIX_POPULATION = f"{PROCESSBENCH_DATASET_REVISION}::llama31_8b::prefix"

GLOBAL_VALIDATION_METHODS = {
    "base7_full28__dufs_l0p1": "unified28_dufs_l0p1",
    "base7_full28__dufs_l0p3": "unified28_dufs_l0p3",
    "base7_full28__dufs_l1": "unified28_dufs_l1",
    "base7_full28__dufs_l3": "unified28_dufs_l3",
    "base7_full28__rw_a0p5": "unified28_task_reweighted_a0p5_historical",
    "raw9_full36": "ordinary36_historical_control",
}

# Report-policy rosters are deliberately separate from the record/audit roster.
# The validation candidates below were part of the frozen Unified search history;
# retaining their rows is useful for auditability, but allowing them into the
# paper-exact direct table would silently reopen method/DUFS selection.
GLOBAL_DIRECT_METHOD_IDS = (
    "unified28",
    "classic_mixed_v2_no_length",
    MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
    "max_entropy_global",
)
GLOBAL_CONTEXT_METHOD_IDS = tuple(GLOBAL_VALIDATION_METHODS.values())
GLOBAL_ALL_METHOD_IDS = GLOBAL_DIRECT_METHOD_IDS + GLOBAL_CONTEXT_METHOD_IDS

EXTERNAL_LOCALIZATION = {
    "prm_qwen25_math_7b": (
        "dataset_cache/four_localization/pb_prm_qwen25math7b_full",
        "pb_prm_{subset}.pkl",
    ),
    "critic_qwen25_72b_single_greedy": (
        "dataset_cache/four_localization/pb_critic_qwen72b_full",
        "pb_critic_{subset}.pkl",
    ),
    "eq6_qwen3_8b_precontract": (
        "dataset_cache/four_localization/pb_uprm_baseline_qwen3_8b_full",
        "pb_uprm_base_{subset}.pkl",
    ),
}
L1_METHOD_ID = "uprm_eq6_qwen25_14b_control"


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _runtime_code_dirty() -> str:
    return _git("status", "--porcelain", "--", *RUNTIME_CODE_PATHS)


def _lane_population(population: ProcessBenchPopulation, population_id: str) -> ProcessBenchPopulation:
    return ProcessBenchPopulation(
        population_id=population_id,
        rows=population.rows,
        ordered_ids=population.ordered_ids,
        ordered_id_sha256=population.ordered_id_sha256,
        source_hashes=population.source_hashes,
    )


def _retarget_records(
    records: Sequence[Mapping[str, Any]],
    *,
    population_id: str,
) -> list[dict[str, Any]]:
    return [{**dict(row), "population_id": population_id} for row in records]


def _json_safe(
    value: Any,
    *,
    path_replacements: Sequence[tuple[str, str]] = (),
) -> Any:
    """Encode portable canonical output, with unavailable statistics as null.

    Audit helpers operate on resolved paths so they can fail closed before reading an
    artifact.  Resolved host paths are execution details, not scientific provenance;
    replace the two registered local roots before serializing the package so an
    independent output directory or checkout location cannot change result bytes.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, path_replacements=path_replacements)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path_replacements=path_replacements) for item in value
        ]
    if isinstance(value, str):
        portable = value
        for source, token in sorted(
            path_replacements, key=lambda item: len(item[0]), reverse=True
        ):
            source = source.rstrip("/")
            if not source:
                raise ValueError("portable path roots must not be filesystem root")
            portable = portable.replace(f"{source}/", f"{token}/")
            if portable == source:
                portable = token
        return portable.replace(str(ROOT), "${REPO_ROOT}")
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _portable_twentyfour_output(
    twentyfour: Mapping[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Remove the 24-cell staging location from serialized scientific output.

    The partial identity audit hashes its complete in-memory payload, including
    resolved source paths.  Recompute that one container hash after replacing the
    execution-only root so independent staging locations produce identical bytes
    and the emitted hash still authenticates the emitted payload.
    """

    resolved_root = source_root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise ValueError("--twentyfour-source-root cannot be filesystem root")
    normalized = _json_safe(
        twentyfour,
        path_replacements=((str(resolved_root), TWENTYFOUR_SOURCE_TOKEN),),
    )
    if not isinstance(normalized, dict):  # Defensive type narrowing for callers.
        raise TypeError("24-cell preflight must serialize to a JSON object")
    identity_audit = normalized.get("identity_audit")
    if isinstance(identity_audit, Mapping) and "audit_sha256" in identity_audit:
        portable_audit = dict(identity_audit)
        portable_audit.pop("audit_sha256", None)
        portable_audit["audit_sha256"] = canonical_sha256(portable_audit)
        normalized["identity_audit"] = portable_audit
    return normalized


def _evaluator_run_contract(n_boot: int) -> dict[str, Any]:
    """Bind the evaluator summary to this run's actual uncertainty workload."""

    contract = dict(evaluator_summary())
    publication_default = int(contract["bootstrap_replicates"])
    contract["publication_default_bootstrap_replicates"] = publication_default
    contract["bootstrap_replicates"] = int(n_boot)
    return contract


def _report_identity(
    *,
    testing_only: bool,
    n_boot: int,
    testing_deviations: Sequence[str],
) -> dict[str, Any]:
    """Return the visible report watermark and run-specific uncertainty wording."""

    base_title = "Fair Paper-Exact Comparison Package v1"
    base_summary = (
        "Unified-28 is frozen as the ordinary method of record. Direct tables use "
        "strict canonical joins; native context and incomplete/blocked assets are separated."
    )
    if testing_only:
        title = f"TEST-ONLY — {base_title}"
        summary = (
            "TEST-ONLY NON-PUBLICATION OUTPUT. "
            f"Intervals use {int(n_boot):,} bootstrap replicate(s) and must not be "
            f"reported as publication uncertainty. {base_summary}"
        )
        ci_status = (
            f"test-only intervals from {int(n_boot)} paired grouped bootstrap "
            "replicate(s); not publication intervals"
        )
    else:
        title = base_title
        summary = base_summary
        ci_status = (
            f"95% percentile intervals from {int(n_boot)} paired grouped bootstrap "
            "replicates with thresholds recomputed where permitted"
        )
    return {
        "title": title,
        "summary": summary,
        "testing_only": bool(testing_only),
        "publication_eligible": not bool(testing_only),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "confidence_interval_status": ci_status,
        "testing_deviations": list(testing_deviations),
    }


def _pb_paths() -> dict[str, Path]:
    return {
        subset: ROOT
        / "dataset_cache"
        / "repgrid"
        / "pb_llama31_8b"
        / f"processbench_{subset}.pkl"
        for subset in PROCESSBENCH_SUBSETS
    }


def _qwen_pb_paths() -> dict[str, Path]:
    return {
        subset: ROOT
        / "cache"
        / "localization"
        / "processbench"
        / "pb_qwen3_8b"
        / f"processbench_{subset}.pkl"
        for subset in PROCESSBENCH_SUBSETS
    }


def build_pb_population_and_folds() -> dict[str, Any]:
    paths = _pb_paths()
    telemetry, hashes = load_pickle_bundle(paths)
    built = build_processbench_population(telemetry, source_hashes=hashes)
    population = built.population
    if population.ordered_id_sha256 != (
        "7a7edc7a6e4a67ac16968c915900805620fc86314368105afe05a4d1ffd20e10"
    ):
        raise RuntimeError("official ProcessBench ordered-ID hash drifted")
    global_folds = assign_group_folds(
        [
            {
                "group_id": row.group_id,
                "family": row.subset,
                "stratify_label": row.wrong_label,
            }
            for row in population.rows.values()
        ]
    )
    localization_folds = assign_group_folds(
        [
            {
                "group_id": row.group_id,
                "family": row.subset,
                "stratify_label": int(row.localization_label != -1),
            }
            for row in population.rows.values()
        ]
    )
    return {
        "population": population,
        "telemetry": telemetry,
        "paths": paths,
        "hashes": hashes,
        "audit": built.audit.to_dict(),
        "global_folds": global_folds,
        "localization_folds": localization_folds,
        "global_fold_hash": fold_assignment_sha256(global_folds),
        "localization_fold_hash": fold_assignment_sha256(localization_folds),
    }


def _max_entropy_global_records(
    pb: Mapping[str, Any],
    *,
    population_id: str,
) -> list[dict[str, Any]]:
    population: ProcessBenchPopulation = pb["population"]
    output = []
    for row_id in population.ordered_ids:
        pop = population.rows[row_id]
        source = pb["telemetry"][pop.subset]
        matching = [
            row
            for row in (source.values() if isinstance(source, Mapping) else source)
            if str(row.get("id")) == pop.official_id
        ]
        if len(matching) != 1:
            raise ValueError(f"max-entropy explicit ID join failed: {row_id}")
        entropy = np.asarray(matching[0]["token_entropies"], dtype=float)
        if entropy.ndim != 1 or not len(entropy) or not np.all(np.isfinite(entropy)):
            raise ValueError(f"max-entropy telemetry invalid: {row_id}")
        output.append(
            make_comparison_record(
                lane="global",
                population_id=population_id,
                row_id=row_id,
                group_id=pop.group_id,
                cell_id=pop.cell_id,
                method_id="max_entropy_global",
                continuous_score=float(np.max(entropy)),
                discrete_prediction=None,
                label=int(pop.wrong_label),
                budget="final",
                fold=int(pb["global_folds"][row_id]),
                calibration_hash=None,
                source_artifact_hash=pb["hashes"][pop.subset],
                extra={
                    "family": pop.subset,
                    "stratify_label": int(pop.wrong_label),
                    "prediction_status": "not_applicable",
                },
            )
        )
    return output


def build_global_pb(pb: Mapping[str, Any], *, n_boot: int) -> dict[str, Any]:
    population = _lane_population(pb["population"], PB_GLOBAL_POPULATION)
    paired = adapt_unified_global_records(
        UNIFIED_GLOBAL,
        pb["population"],
        folds=pb["global_folds"],
    )
    controls = adapt_unified_validation_records(
        UNIFIED_VALIDATION,
        pb["population"],
        lane="global",
        candidate_methods=GLOBAL_VALIDATION_METHODS,
        folds=pb["global_folds"],
    )
    qwen_paths = _qwen_pb_paths()
    qwen_telemetry, qwen_hashes = load_pickle_bundle(qwen_paths)
    dufs_replay = replay_registered_dufs_no_length(
        population=pb["population"],
        population_id=population.population_id,
        qwen_telemetry=qwen_telemetry,
        llama_telemetry=pb["telemetry"],
        fit_ids_path=CLASSIC_GLOBAL_FIT_IDS,
        folds=pb["global_folds"],
        qwen_source_hashes=qwen_hashes,
        llama_source_hashes=pb["hashes"],
        anchor_manifest_path=DUFS_ANCHOR_MANIFEST,
        classic_run_definition_path=CLASSIC_GLOBAL_RUN_DEFINITION,
    )
    dufs_replay["qwen_paths"] = qwen_paths
    dufs_replay["qwen_source_hashes"] = qwen_hashes
    records = _retarget_records(paired.records, population_id=population.population_id)
    records.extend(_retarget_records(controls.records, population_id=population.population_id))
    records.extend(dufs_replay["records"])
    records.extend(_max_entropy_global_records(pb, population_id=population.population_id))
    method_ids = GLOBAL_ALL_METHOD_IDS
    metrics = evaluate_global_panel(
        records,
        ordered_ids=population.ordered_ids,
        method_ids=method_ids,
    )
    intervals = bootstrap_global_contrasts(
        records,
        ordered_ids=population.ordered_ids,
        method_ids=("unified28", "classic_mixed_v2_no_length"),
        contrasts=(
            ("unified28", "classic_mixed_v2_no_length"),
        ),
        n_boot=n_boot,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    intervals["unavailable_primary_contrasts"] = [
        {
            "left": "unified28",
            "right_role": "strongest same-access published competitor",
            "reason": "no published paper score joins all 3,400 exact trace IDs",
        }
    ]
    return {
        "population": population,
        "records": records,
        "metrics": metrics,
        "intervals": intervals,
        "adapter_audits": {
            "unified_and_incumbent": paired.audit.to_dict(),
            "registered_controls": controls.audit.to_dict(),
            "mixed_v2_dufs_no_length": {
                "schema": dufs_replay["schema"],
                "coverage": dufs_replay["coverage"],
                "ordered_id_sha256": dufs_replay["ordered_id_sha256"],
                "fit_id_sha256": dufs_replay["fit_ids"]["fit_id_sha256"],
                "anchor_audit": dufs_replay["anchor_audit"],
                "provenance_audit": dufs_replay["provenance_audit"],
                "labels_read_during_score_construction": dufs_replay[
                    "labels_read_during_score_construction"
                ],
            },
        },
        "dufs_replay": dufs_replay,
        "method_ids": method_ids,
        "direct_method_ids": GLOBAL_DIRECT_METHOD_IDS,
        "context_method_ids": GLOBAL_CONTEXT_METHOD_IDS,
    }


def _external_localization_records(
    pb: Mapping[str, Any],
    *,
    cache_root: Path,
) -> dict[str, Any]:
    population: ProcessBenchPopulation = pb["population"]
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    audits = {}
    source_paths: dict[str, list[Path]] = {}
    for method_id, (directory, template) in EXTERNAL_LOCALIZATION.items():
        paths = {
            subset: ROOT / directory / template.format(subset=subset)
            for subset in PROCESSBENCH_SUBSETS
        }
        rows, hashes = load_pickle_bundle(paths)
        adapted = adapt_external_localization_records(
            rows,
            population,
            method_id=method_id,
            source_artifact_hashes=hashes,
            folds=pb["localization_folds"],
        )
        retargeted = _retarget_records(
            adapted.records, population_id=PB_LOCALIZATION_POPULATION
        )
        records_by_method[method_id] = retargeted
        audits[method_id] = adapted.audit.to_dict()
        source_paths[method_id] = list(paths.values())

    l1_dir = cache_root / "l1_uprm_judge_full"
    indexed = indexed_pickle_rows(l1_dir)
    adapted = adapt_eq6_shard_records(
        indexed["rows"],
        population,
        method_id=L1_METHOD_ID,
        source_artifact_hash=indexed["package_hash"],
        folds=pb["localization_folds"],
    )
    records_by_method[L1_METHOD_ID] = _retarget_records(
        adapted.records, population_id=PB_LOCALIZATION_POPULATION
    )
    audits[L1_METHOD_ID] = adapted.audit.to_dict()
    return {
        "records_by_method": records_by_method,
        "audits": audits,
        "source_paths": source_paths,
        "l1": indexed,
        "l1_dir": l1_dir,
    }


def build_localization_pb(
    pb: Mapping[str, Any],
    *,
    cache_root: Path,
    n_boot: int,
) -> dict[str, Any]:
    population = _lane_population(pb["population"], PB_LOCALIZATION_POPULATION)
    unified = load_unified28_prethreshold_rows(
        UNIFIED_VALIDATION,
        pb["population"],
        folds=pb["localization_folds"],
    )
    replay = replay_same_access_methods(ROOT, pb["population"], folds=pb["localization_folds"])
    score_methods = {UNIFIED28_METHOD_ID: unified["rows"], **replay["methods"]}
    scored = {
        method_id: crossfit_score_method(
            rows,
            method_id=method_id,
            population_id=population.population_id,
        )
        for method_id, rows in score_methods.items()
    }
    external = _external_localization_records(pb, cache_root=cache_root)
    records = [
        row
        for method_id in score_methods
        for row in scored[method_id]["records"]
    ]
    for method_rows in external["records_by_method"].values():
        records.extend(method_rows)
    fixed_metrics = {
        method_id: evaluate_fixed_prediction_method(
            method_rows,
            expected_order=population.ordered_ids,
        )
        for method_id, method_rows in external["records_by_method"].items()
    }
    intervals = bootstrap_localization_contrasts(
        score_methods,
        contrasts=(
            (UNIFIED28_METHOD_ID, DEDICATED_METHOD_ID),
            (UNIFIED28_METHOD_ID, MIND_GAP_METHOD_ID),
        ),
        n_boot=n_boot,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    return {
        "population": population,
        "records": records,
        "score_methods": score_methods,
        "scored": scored,
        "fixed_metrics": fixed_metrics,
        "intervals": intervals,
        "native_mind_gap": native_mind_gap_metrics(score_methods[MIND_GAP_METHOD_ID]),
        "replay": replay,
        "external": external,
        "method_ids": tuple(score_methods) + tuple(external["records_by_method"]),
    }


def load_prmbench_native_context() -> dict[str, Any]:
    freeze = json.loads(PRMBENCH_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("score_hash") != "fbca592262868b52f8e5dd3ce93255e457caff8298a1d055ad3462e5bcbafe9f":
        raise ValueError("PRMBench frozen score hash drifted")
    if len(freeze.get("excluded_alignment_ids", [])) != 3:
        raise ValueError("PRMBench registered alignment exclusions drifted")
    rows = []
    with PRMBENCH_METRICS.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if source.get("benchmark") != "PRMBench":
                continue
            method = str(source["method"])
            method_id = (
                "fixed_reasoning_iu_prmbench"
                if method == "Fixed reasoning IU-PCR"
                else "prm_qwen25_math_7b"
                if method == "Qwen2.5-Math-PRM-7B (supervised ceiling)"
                else None
            )
            if method_id is None:
                raise ValueError(f"unregistered PRMBench method: {method}")
            rows.append(
                {
                    "method_id": method_id,
                    "method": method,
                    "subgroup": source.get("subgroup") or "all nine paper classes",
                    "split": source["split"],
                    "unit": source["unit"],
                    "n_steps": int(source["n"]),
                    "auroc": float(source["auroc"]),
                    "error_auprc": float(source["auprc"]),
                    "error_prevalence": float(source["positive_rate"]),
                }
            )
    expected_subgroups = {
        "all nine paper classes",
        "circular",
        "confidence",
        "counterfactual",
        "deception",
        "domain_inconsistency",
        "missing_condition",
        "redundency",
        "step_contradiction",
    }
    if (
        len(rows) != 10
        or {row["subgroup"] for row in rows} != expected_subgroups
        or sum(row["subgroup"] == "all nine paper classes" for row in rows) != 2
    ):
        raise ValueError("PRMBench native metric roster drifted")
    diagnostics = json.loads(PRMBENCH_DIAGNOSTICS.read_text(encoding="utf-8"))["prmbench"]
    if int(diagnostics["n_scored_steps"]) != 83280:
        raise ValueError("PRMBench scored-step count drifted")
    return {
        "schema": "prmbench_native_every_step_context_v1",
        "fidelity": "adapted-common-protocol",
        "headline_eligible": False,
        "population": "PRMBench Preview after exactly three registered alignment exclusions",
        "positive_class": "erroneous reasoning step",
        "score_hash": freeze["score_hash"],
        "excluded_alignment_ids": freeze["excluded_alignment_ids"],
        "source_artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                PRMBENCH_METRICS,
                PRMBENCH_FREEZE,
                PRMBENCH_DIAGNOSTICS,
                PRMBENCH_TELEMETRY,
                PRMBENCH_PRM,
            )
        },
        "rows": rows,
        "diagnostics": diagnostics,
        "separate_from_processbench_f1": True,
    }


def _crossfit_prefix_warnings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit trace-level ever-warning thresholds on four folds and score the fifth."""

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(dict(row))
    results: dict[str, Any] = {}
    decision_rows: list[dict[str, Any]] = []
    for method_id, method_rows in sorted(by_method.items()):
        if {int(row["fold"]) for row in method_rows} != set(range(5)):
            raise ValueError(f"Prefix warning folds must be exactly 0..4: {method_id}")
        first_warnings: dict[str, list[int | None]] = {"fpr_05": [], "fpr_10": []}
        labels: dict[str, list[int]] = {"fpr_05": [], "fpr_10": []}
        cells: dict[str, list[str]] = {"fpr_05": [], "fpr_10": []}
        families: dict[str, list[str]] = {"fpr_05": [], "fpr_10": []}
        ledgers: dict[str, list[dict[str, Any]]] = {"fpr_05": [], "fpr_10": []}
        for held_out in range(5):
            train = [row for row in method_rows if int(row["fold"]) != held_out]
            test = [row for row in method_rows if int(row["fold"]) == held_out]
            fitted = calibrate_prefix_ever_warning_thresholds(
                [row["label"] for row in train],
                [row["score_path"] for row in train],
                budgets=PREFIX_BUDGETS,
            )
            for target_key in ("fpr_05", "fpr_10"):
                threshold = float(fitted[target_key]["threshold"])
                scored = prefix_warning_metrics(
                    [row["label"] for row in test],
                    [row["score_path"] for row in test],
                    threshold=threshold,
                    budgets=PREFIX_BUDGETS,
                )
                first_warnings[target_key].extend(scored["first_warning_budgets"])
                labels[target_key].extend(int(row["label"]) for row in test)
                cells[target_key].extend(str(row["cell_id"]) for row in test)
                families[target_key].extend(str(row["family"]) for row in test)
                ledger = {
                    **fitted[target_key],
                    "held_out_fold": held_out,
                    "train_folds": [fold for fold in range(5) if fold != held_out],
                    "n_held_out_rows": len(test),
                }
                ledger["calibration_hash"] = canonical_sha256(ledger)
                ledgers[target_key].append(ledger)
                for row, first_warning in zip(test, scored["first_warning_budgets"]):
                    decision_rows.append(
                        {
                            "population_id": row["population_id"],
                            "row_id": row["row_id"],
                            "group_id": row["group_id"],
                            "cell_id": row["cell_id"],
                            "family": row["family"],
                            "method_id": method_id,
                            "target": target_key,
                            "fold": held_out,
                            "label": int(row["label"]),
                            "threshold": threshold,
                            "first_warning_budget": first_warning,
                            "ever_warning": first_warning is not None,
                            "calibration_hash": ledger["calibration_hash"],
                        }
                    )

        operating_points: dict[str, Any] = {}
        for target_key in ("fpr_05", "fpr_10"):
            label_array = np.asarray(labels[target_key], dtype=int)
            warning_array = np.asarray(
                [value is not None for value in first_warnings[target_key]], dtype=bool
            )
            wrong = label_array == 1
            correct = label_array == 0
            warned_wrong_budgets = [
                int(value)
                for value, label in zip(first_warnings[target_key], label_array)
                if value is not None and label == 1
            ]
            per_cell = []
            for cell_id in sorted(set(cells[target_key])):
                indices = [
                    index
                    for index, observed in enumerate(cells[target_key])
                    if observed == cell_id
                ]
                cell_labels = label_array[indices]
                cell_warnings = warning_array[indices]
                cell_wrong = cell_labels == 1
                cell_correct = cell_labels == 0
                per_cell.append(
                    {
                        "cell_id": cell_id,
                        "n": len(indices),
                        "wrong_trace_warning_coverage": float(np.mean(cell_warnings[cell_wrong]))
                        if np.any(cell_wrong)
                        else None,
                        "correct_trace_ever_warning_fpr": float(np.mean(cell_warnings[cell_correct]))
                        if np.any(cell_correct)
                        else None,
                    }
                )
            per_family = []
            for family in sorted(set(families[target_key])):
                indices = [
                    index
                    for index, observed in enumerate(families[target_key])
                    if observed == family
                ]
                family_labels = label_array[indices]
                family_warnings = warning_array[indices]
                family_first = [first_warnings[target_key][index] for index in indices]
                family_wrong = family_labels == 1
                family_correct = family_labels == 0
                family_warned_wrong = [
                    int(value)
                    for value, label in zip(family_first, family_labels)
                    if value is not None and label == 1
                ]
                per_family.append(
                    {
                        "family": family,
                        "n": len(indices),
                        "wrong_trace_warning_coverage": float(
                            np.mean(family_warnings[family_wrong])
                        )
                        if np.any(family_wrong)
                        else None,
                        "correct_trace_ever_warning_fpr": float(
                            np.mean(family_warnings[family_correct])
                        )
                        if np.any(family_correct)
                        else None,
                        "median_first_warning_budget_on_warned_wrong_traces": float(
                            np.median(family_warned_wrong)
                        )
                        if family_warned_wrong
                        else None,
                    }
                )

            def equal_family_metric(name: str) -> float | None:
                values = [
                    float(row[name])
                    for row in per_family
                    if row[name] is not None and np.isfinite(float(row[name]))
                ]
                return float(np.mean(values)) if values else None

            operating_points[target_key] = {
                "wrong_trace_warning_coverage": float(np.mean(warning_array[wrong]))
                if np.any(wrong)
                else None,
                "correct_trace_ever_warning_fpr": float(np.mean(warning_array[correct]))
                if np.any(correct)
                else None,
                "median_first_warning_budget_on_warned_wrong_traces": float(
                    np.median(warned_wrong_budgets)
                )
                if warned_wrong_budgets
                else None,
                "n": len(label_array),
                "per_cell": per_cell,
                "per_family": per_family,
                "equal_family": {
                    "wrong_trace_warning_coverage": equal_family_metric(
                        "wrong_trace_warning_coverage"
                    ),
                    "correct_trace_ever_warning_fpr": equal_family_metric(
                        "correct_trace_ever_warning_fpr"
                    ),
                    "median_first_warning_budget_on_warned_wrong_traces": equal_family_metric(
                        "median_first_warning_budget_on_warned_wrong_traces"
                    ),
                },
                "calibration_ledgers": ledgers[target_key],
                "calibration_hash": canonical_sha256(ledgers[target_key]),
                "aggregation": "concatenated_discrete_out_of_fold_trace_warnings",
            }
        results[method_id] = operating_points
    decision_rows.sort(
        key=lambda row: (
            row["population_id"],
            row["method_id"],
            row["target"],
            row["row_id"],
        )
    )
    return {"methods": results, "decisions": decision_rows}


def _bootstrap_prefix_warnings(
    rows: Sequence[Mapping[str, Any]], *, n_boot: int
) -> dict[str, Any]:
    """Paired warning intervals with threshold refits in every replicate."""

    selected_methods = ("unified28", STEP272_METHOD_ID)
    groups: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for row in rows:
        method_id = str(row["method_id"])
        if method_id not in selected_methods:
            continue
        group_id = str(row["group_id"])
        payload = groups.setdefault(
            group_id,
            {
                "label": int(row["label"]),
                "family": str(row["family"]),
                "cell_id": str(row["cell_id"]),
                "fold": int(row["fold"]),
                "score_paths": {},
            },
        )
        metadata = (
            int(row["label"]),
            str(row["family"]),
            str(row["cell_id"]),
            int(row["fold"]),
        )
        if metadata != (
            payload["label"],
            payload["family"],
            payload["cell_id"],
            payload["fold"],
        ):
            raise ValueError(f"Prefix warning bootstrap metadata conflict: {group_id}")
        if method_id in payload["score_paths"]:
            raise ValueError(
                f"duplicate Prefix warning bootstrap path: {group_id}/{method_id}"
            )
        payload["score_paths"][method_id] = dict(row["score_path"])
        strata[group_id] = payload["family"]
    incomplete = [
        group_id
        for group_id, payload in groups.items()
        if set(payload["score_paths"]) != set(selected_methods)
    ]
    if incomplete:
        raise ValueError(
            "Prefix warning paired payload lacks a required method: "
            f"{incomplete[:5]}"
        )

    def recompute(sample: list[dict[str, Any]]) -> Mapping[str, Any]:
        fitted_methods: dict[str, Any] = {}
        for method_id in selected_methods:
            target_outputs: dict[str, Any] = {}
            collected: dict[str, dict[str, list[Any]]] = {
                "fpr_05": defaultdict(list),
                "fpr_10": defaultdict(list),
            }
            for held_out in range(5):
                train = [row for row in sample if row["fold"] != held_out]
                test = [row for row in sample if row["fold"] == held_out]
                if not train or not test:
                    raise ValueError(
                        "Prefix warning bootstrap lost a required held-out fold"
                    )
                fitted = calibrate_prefix_ever_warning_thresholds(
                    [row["label"] for row in train],
                    [row["score_paths"][method_id] for row in train],
                    budgets=PREFIX_BUDGETS,
                )
                for target_key in ("fpr_05", "fpr_10"):
                    scored = prefix_warning_metrics(
                        [row["label"] for row in test],
                        [row["score_paths"][method_id] for row in test],
                        threshold=float(fitted[target_key]["threshold"]),
                        budgets=PREFIX_BUDGETS,
                    )
                    collected[target_key]["labels"].extend(
                        int(row["label"]) for row in test
                    )
                    collected[target_key]["families"].extend(
                        str(row["family"]) for row in test
                    )
                    collected[target_key]["first"].extend(
                        scored["first_warning_budgets"]
                    )
            for target_key in ("fpr_05", "fpr_10"):
                labels = np.asarray(collected[target_key]["labels"], dtype=int)
                families = collected[target_key]["families"]
                first = collected[target_key]["first"]
                warnings = np.asarray([value is not None for value in first], dtype=bool)
                per_family: list[dict[str, float]] = []
                for family in sorted(set(families)):
                    indices = [
                        index for index, value in enumerate(families) if value == family
                    ]
                    family_labels = labels[indices]
                    family_warnings = warnings[indices]
                    wrong = family_labels == 1
                    correct = family_labels == 0
                    warned_wrong = [
                        int(first[index])
                        for index in indices
                        if first[index] is not None and labels[index] == 1
                    ]
                    per_family.append(
                        {
                            "coverage": float(np.mean(family_warnings[wrong]))
                            if np.any(wrong)
                            else float("nan"),
                            "fpr": float(np.mean(family_warnings[correct]))
                            if np.any(correct)
                            else float("nan"),
                            "first": float(np.median(warned_wrong))
                            if warned_wrong
                            else float("nan"),
                        }
                    )
                target_outputs[target_key] = {
                    name: float(
                        np.mean(
                            [
                                family[name]
                                for family in per_family
                                if np.isfinite(family[name])
                            ]
                        )
                    )
                    for name in ("coverage", "fpr", "first")
                }
            fitted_methods[method_id] = target_outputs
        return fitted_methods

    def statistic(
        _sample: list[dict[str, Any]], fitted: Mapping[str, Any]
    ) -> Mapping[str, float]:
        output: dict[str, float] = {}
        for method_id in selected_methods:
            for target_key in ("fpr_05", "fpr_10"):
                for metric in ("coverage", "fpr", "first"):
                    output[f"{method_id}__{metric}_{target_key}"] = float(
                        fitted[method_id][target_key][metric]
                    )
        for target_key in ("fpr_05", "fpr_10"):
            for metric in ("coverage", "fpr", "first"):
                output[
                    f"delta_{metric}_{target_key}__unified28__minus__{STEP272_METHOD_ID}"
                ] = float(fitted["unified28"][target_key][metric]) - float(
                    fitted[STEP272_METHOD_ID][target_key][metric]
                )
        return output

    return {
        "schema": "fair_prefix_warning_paired_intervals_v1",
        "predeclared_contrasts": [
            {"left": "unified28", "right": STEP272_METHOD_ID}
        ],
        **paired_grouped_bootstrap(
            groups,
            statistic,
            strata=strata,
            recompute=recompute,
            n_boot=n_boot,
            seed=DEFAULT_BOOTSTRAP_SEED,
        ),
    }


def _bootstrap_prefix_primary(
    records: Sequence[Mapping[str, Any]], *, n_boot: int
) -> dict[str, Any]:
    """Paired question bootstrap for the preregistered 64/128 AUROC contrast."""

    selected_methods = ("unified28", STEP272_METHOD_ID)
    by_group: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for row in records:
        if row["method_id"] not in selected_methods or row["budget"] not in (64, 128):
            continue
        group_id = str(row["group_id"])
        payload = by_group.setdefault(
            group_id,
            {
                "label": int(row["label"]),
                "family": str(row["family"]),
                "cell_id": str(row["cell_id"]),
                "scores": defaultdict(dict),
            },
        )
        if (
            payload["label"] != int(row["label"])
            or payload["cell_id"] != str(row["cell_id"])
        ):
            raise ValueError(f"Prefix bootstrap metadata conflict: {group_id}")
        payload["scores"][str(row["method_id"])][int(row["budget"])] = float(
            row["continuous_score"]
        )
        strata[group_id] = str(row["family"])

    def statistic(sample: list[dict[str, Any]], _fit: Any) -> Mapping[str, float]:
        cell_ids = sorted({row["cell_id"] for row in sample})
        method_values: dict[str, float] = {}
        for method_id in selected_methods:
            cell_budget_values = []
            for cell_id in cell_ids:
                cell_rows = [row for row in sample if row["cell_id"] == cell_id]
                for budget in (64, 128):
                    eligible = [
                        row
                        for row in cell_rows
                        if budget in row["scores"].get(method_id, {})
                    ]
                    cell_budget_values.append(
                        auroc(
                            [row["label"] for row in eligible],
                            [row["scores"][method_id][budget] for row in eligible],
                        )
                    )
            finite = [value for value in cell_budget_values if np.isfinite(value)]
            method_values[method_id] = float(np.mean(finite)) if finite else float("nan")
        return {
            "unified28__primary_mean_auroc_64_128": method_values["unified28"],
            f"{STEP272_METHOD_ID}__primary_mean_auroc_64_128": method_values[
                STEP272_METHOD_ID
            ],
            f"delta__unified28__minus__{STEP272_METHOD_ID}": (
                method_values["unified28"] - method_values[STEP272_METHOD_ID]
            ),
        }

    return {
        "schema": "fair_prefix_paired_intervals_v1",
        "predeclared_contrasts": [
            {"left": "unified28", "right": STEP272_METHOD_ID}
        ],
        **paired_grouped_bootstrap(
            by_group,
            statistic,
            strata=strata,
            n_boot=n_boot,
            seed=DEFAULT_BOOTSTRAP_SEED,
        ),
    }


def build_prefix_pb(pb: Mapping[str, Any], *, n_boot: int) -> dict[str, Any]:
    unified = load_unified28_prefix_records(
        UNIFIED_VALIDATION,
        pb["population"],
        folds=pb["global_folds"],
        require_registered_telemetry_provenance=True,
    )
    step272 = load_step272_prefix_records(
        PREFIX_STEP272,
        pb["population"],
        folds=pb["global_folds"],
        require_registered_telemetry_provenance=True,
    )
    entropy = build_entropy_prefix_records(
        pb["telemetry"],
        pb["population"],
        source_artifact_hashes=pb["hashes"],
        folds=pb["global_folds"],
    )
    incumbent_replay = replay_frozen_prefix_incumbents(
        pb["telemetry"],
        pb["population"],
        historical_results_root=ROOT / "results" / "early_online_localization_models_v1",
        source_artifact_hashes=pb["hashes"],
        folds=pb["global_folds"],
    )
    historical = load_historical_prefix_scores(
        PREFIX_HISTORICAL,
        population=pb["population"],
    )
    panel = assemble_historical_common_panel(
        [
            *unified["records"],
            *step272["records"],
            *entropy["records"],
            *incumbent_replay["records"],
            *historical["records"],
        ]
    )
    metrics = summarize_prefix_metrics(panel["records"])
    warning_inputs = build_warning_inputs(panel["records"])
    warning_result = _crossfit_prefix_warnings(warning_inputs["rows"])
    intervals = _bootstrap_prefix_primary(panel["records"], n_boot=n_boot)
    intervals["warning_operating_points"] = _bootstrap_prefix_warnings(
        warning_inputs["rows"], n_boot=n_boot
    )
    intervals["unavailable_primary_contrasts"] = [
        {
            "left": "unified28",
            "right_role": "strongest same-access published competitor",
            "reason": (
                "the retained DeepConf value is an explicitly named historical proxy, "
                "not an exact published score"
            ),
        }
    ]
    context_metrics = summarize_prefix_metrics(historical["records"])
    return {
        "records": list(panel["records"]),
        "populations": panel["populations"],
        "metrics": metrics,
        "warning_inputs": warning_inputs,
        "warnings": warning_result["methods"],
        "warning_decisions": warning_result["decisions"],
        "intervals": intervals,
        "coverage": panel["coverage"],
        "context_records": list(historical["records"]),
        "context_metrics": context_metrics,
        "audits": {
            "unified28": unified["audit"],
            "step272": step272["audit"],
            "entropy": entropy["audit"],
            "incumbent_replay": incumbent_replay["audit"],
            "historical": historical["audit"],
            "join": panel["audit"],
        },
        "method_ids": sorted({str(row["method_id"]) for row in panel["records"]}),
    }


def audit_s2_prefix_cache(cache_root: Path) -> dict[str, Any]:
    """Verify raw COT telemetry while keeping the frozen-model join fail-closed."""

    audits = []
    for run_dir in sorted(cache_root.glob("s2_leash_*")):
        manifest_path = run_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = str(manifest.get("dataset_source", ""))
        dataset = "aqua" if "aqua" in source else "gsm8k" if "gsm8k" in source else None
        if dataset is None:
            raise ValueError(f"unrecognized S2 dataset source: {source}")
        indexed = indexed_pickle_rows(run_dir)
        audit = audit_s2_cot_telemetry(
            indexed["rows"],
            dataset_revision=str(manifest["dataset_revision"]),
            dataset=dataset,
            model=str(manifest["model_id"]),
            ordered_question_ids=manifest["dataset_example_ids"],
        )
        audit["run_id"] = manifest["run_id"]
        audit["indexed_package_sha256"] = indexed["package_hash"]
        audits.append(audit)
    if len(audits) != 6:
        raise ValueError(f"expected six S2 COT telemetry audits, found {len(audits)}")
    prefix_eligible = all(bool(audit["prefix_scoring_eligible"]) for audit in audits)
    global_eligible = all(bool(audit["global_scoring_eligible"]) for audit in audits)
    return {
        "schema": "s2_frozen_transfer_suite_gate_v1",
        "runs": audits,
        "all_raw_telemetry_gates_passed": all(
            bool(audit["raw_telemetry_gate_passed"]) for audit in audits
        ),
        "all_frozen_model_join_gates_passed": all(
            bool(audit["frozen_model_join_gate_passed"]) for audit in audits
        ),
        "all_prefix_scoring_gates_passed": prefix_eligible,
        "all_global_scoring_gates_passed": global_eligible,
        "scores_materialized": False,
        "prefix_scores_materialized": False,
        "global_scores_materialized": False,
        "prefix_headline_eligible": prefix_eligible,
        "global_headline_eligible": global_eligible,
        "headline_eligible": prefix_eligible or global_eligible,
        "reason": (
            "the acquired COT rows retain raw/pre-warper streams, while the frozen "
            "methods require legacy post-warper entropy, sampled-token energy, and "
            "top-k streams; raw substitution would define an adapted new method"
        ),
    }


def _population_entry_from_pb(
    population: ProcessBenchPopulation,
    *,
    lane: str,
) -> dict[str, Any]:
    labels = {
        "global": {
            "field": "wrong",
            "positive": 1,
            "definition": "not final_answer_correct; not the ProcessBench process-error target",
        },
        "localization": {
            "field": "first_error",
            "clean": -1,
            "definition": "official earliest erroneous reasoning-step index",
        },
        "prefix": {
            "field": "wrong",
            "positive": 1,
            "definition": "not final_answer_correct at a strictly unfinished prefix",
        },
    }
    eligibility = (
        {"rule": "all official rows", "coverage_required": 1.0}
        if lane != "prefix"
        else {"rule": "final_length > budget", "budgets": [16, 32, 64, 128, 256, 512]}
    )
    return make_population_entry(
        population_id=population.population_id,
        lane=lane,
        dataset_revision=PROCESSBENCH_DATASET_REVISION,
        ordered_ids=population.ordered_ids,
        group_ids=[population.rows[row_id].group_id for row_id in population.ordered_ids],
        cell_ids=[population.rows[row_id].cell_id for row_id in population.ordered_ids],
        families=[population.rows[row_id].subset for row_id in population.ordered_ids],
        label_definition=labels[lane],
        eligibility_rules=eligibility,
        extra={"source_ordered_id_sha256": population.ordered_id_sha256},
    )


def _stopping_population_entries(stopping: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for audit in sorted(
        stopping["run_audits"],
        key=lambda row: (
            str(row["dataset_revision"]),
            str(row["dataset"]),
            str(row["model"]),
        ),
    ):
        revision = str(audit["dataset_revision"])
        dataset = str(audit["dataset"])
        model = str(audit["model"])
        cell_id = f"s2::{dataset}::{model}"
        population_id = f"s2_stopping::{revision}::{dataset}::{model}"
        question_ids = [str(value) for value in audit["registered_question_ids"]]
        paired_group_ids = [
            canonical_s2_group_id(revision, dataset, question_id)
            for question_id in question_ids
        ]
        if paired_group_ids != list(audit["registered_group_ids"]):
            raise ValueError(f"stopping manifest group roster drift: {population_id}")
        if ordered_id_sha256(paired_group_ids) != audit["paired_group_order_sha256"]:
            raise ValueError(f"stopping paired-group hash drift: {population_id}")
        ordered_ids = [
            canonical_s2_id(revision, dataset, question_id, model, arm)
            for question_id in question_ids
            for arm in S2_ARMS
        ]
        group_ids = [
            canonical_s2_group_id(revision, dataset, question_id)
            for question_id in question_ids
            for _arm in S2_ARMS
        ]
        method_populations = {
            f"{arm}|{S2_SETTING}": make_eligible_population(
                [
                    canonical_s2_id(revision, dataset, question_id, model, arm)
                    for question_id in question_ids
                ],
                rule=f"registered_manifest_questions_for_{arm}_{S2_SETTING}",
            )
            for arm in S2_ARMS
        }
        entries.append(
            make_population_entry(
                population_id=population_id,
                lane="stopping",
                dataset_revision=revision,
                ordered_ids=ordered_ids,
                group_ids=group_ids,
                cell_ids=[cell_id] * len(ordered_ids),
                families=[dataset] * len(ordered_ids),
                label_definition={
                    "field": "gold_answer",
                    "correct": "rescored prediction equals canonical gold answer",
                    "unparsed": "present and wrong",
                },
                eligibility_rules={
                    "rule": "complete S2 cell with all cot/leash/nocot arms on identical questions",
                    "realized_tokens": "reasoning + generated closure",
                },
                extra={
                    "eligible_populations": method_populations,
                    "eligible_group_populations": {
                        "paired_source_questions": make_eligible_population(
                            paired_group_ids,
                            rule=(
                                "manifest_ordered_source_questions_shared_by_all_arms_"
                                "and_model_copies"
                            ),
                        )
                    },
                    "paired_group_order_sha256": audit["paired_group_order_sha256"],
                },
            )
        )
    return entries


def _prefix_population_entries(prefix: Mapping[str, Any]) -> list[dict[str, Any]]:
    records_by_population: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prefix["records"]:
        records_by_population[str(row["population_id"])].append(row)
    entries = []
    for source in prefix["populations"]:
        population_id = str(source["population_id"])
        rows = records_by_population[population_id]
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            by_id.setdefault(str(row["row_id"]), row)
        ordered_ids = list(source["ordered_row_ids"])
        if set(by_id) != set(ordered_ids):
            raise ValueError(f"Prefix population materialization drift: {population_id}")
        entries.append(
            make_population_entry(
                population_id=population_id,
                lane="prefix",
                dataset_revision=PROCESSBENCH_DATASET_REVISION,
                ordered_ids=ordered_ids,
                group_ids=[str(by_id[row_id]["group_id"]) for row_id in ordered_ids],
                cell_ids=[str(by_id[row_id]["cell_id"]) for row_id in ordered_ids],
                families=[str(by_id[row_id]["family"]) for row_id in ordered_ids],
                label_definition={
                    "field": "wrong",
                    "positive": 1,
                    "definition": "not final_answer_correct on an unfinished registered trace",
                },
                eligibility_rules={
                    "rule": "final_length > budget",
                    "budgets": list(PREFIX_BUDGETS),
                    "direct_population": (
                        "intersection of Unified-28 and Step-272 on identical registered telemetry"
                    ),
                },
                extra={
                    "source_ordered_id_sha256": source["ordered_id_sha256"],
                    "required_methods": source["required_methods"],
                    "included_methods": source["included_methods"],
                    "final_lengths": source["final_lengths"],
                    "eligible_populations": source["eligible_populations"],
                    "outcome_filtering": False,
                },
            )
        )
    return entries


def _remote_asset(
    *, artifact_id: str, uri: str, size_bytes: int, sha256: str
) -> dict[str, Any]:
    """Bind one remotely observed file to its Drive-reported byte identity."""

    return validate_asset_record(
        {
            "schema": "asset_record_v1",
            "artifact_kind": "remote-file",
            "artifact_id": artifact_id,
            "uri": uri,
            "size_bytes": int(size_bytes),
            "sha256": sha256,
        }
    )


def _derived_asset(
    *,
    artifact_id: str,
    uri: str,
    projection: Mapping[str, Any],
    artifact_kind: str = "derived-ledger",
    source_fingerprint_aliases: Sequence[str] = (),
    source_fingerprint_preimages: Sequence[Any] = (),
) -> dict[str, Any]:
    """Materialize the exact canonical bytes represented by a derived ledger row."""

    return make_derived_asset_record(
        _json_safe(dict(projection)),
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        uri=uri,
        source_fingerprint_aliases=source_fingerprint_aliases,
        source_fingerprint_preimages=source_fingerprint_preimages,
    )


def _verified_local_asset(
    path: Path,
    *,
    expected_sha256: str,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    asset = _asset(path, artifact_id=artifact_id)
    if asset["sha256"] != expected_sha256:
        raise ValueError(
            f"local artifact hash drift for {asset['artifact_id']}: "
            f"expected {expected_sha256}, observed {asset['sha256']}"
        )
    return asset


def _access(
    input_type: str,
    supervision: str,
    passes: float | None,
    traces: float | None,
    *,
    pass_scope: str = "additional scorer passes beyond the registered source trace",
) -> dict[str, Any]:
    return {
        "input_type": input_type,
        "supervision": supervision,
        "model_passes_per_question": passes,
        "traces_per_question": traces,
        "model_passes_scope": pass_scope,
        "registered_source_traces_per_question": traces,
    }


def _method(
    *,
    method_id: str,
    display_name: str,
    fidelity: str,
    artifacts: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
    run_commit: str,
    evaluator_hash: str,
    training: str,
    checkpoint: str,
    deviations: Sequence[str],
    prompt: str = "not-applicable",
    decoding: str = "deterministic-offline-score",
    prompt_sha256: str | None = None,
    decoding_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    method_extra = {
        "package_build_commit": run_commit,
        "artifact_generation_commit": (
            "not-recorded-in-pre-contract-source; content hashes and frozen run definitions "
            "are authoritative"
        ),
    }
    if extra:
        method_extra.update(dict(extra))
    return make_method_entry(
        method_id=method_id,
        display_name=display_name,
        fidelity=fidelity,
        source_artifacts=artifacts,
        access=access,
        training_label_use=training,
        checkpoint_revision=checkpoint,
        prompt_sha256=prompt_sha256 or canonical_sha256({"prompt": prompt}),
        decoding_sha256=decoding_sha256 or canonical_sha256({"decoding": decoding}),
        evaluator_sha256=evaluator_hash,
        run_commit=run_commit,
        deviations=list(deviations),
        extra=method_extra,
    )


def _asset(path: Path, artifact_id: str | None = None) -> dict[str, Any]:
    return make_asset_record(path, artifact_id=artifact_id, root=ROOT)


def build_registries(
    *,
    pb: Mapping[str, Any],
    global_pb: Mapping[str, Any],
    localization: Mapping[str, Any],
    prefix: Mapping[str, Any],
    stopping: Mapping[str, Any],
    run_commit: str,
    unified_tree: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # This is a content hash of the complete comparison-evaluation closure, not just
    # the primitive metric module.  Lane aggregation, calibration, resampling,
    # parsing, strict joins, and builder-level statistics can all change a result and
    # therefore all belong under the evaluator identity.
    evaluator_paths = sorted(
        (ROOT / "spectral_utils" / "fair_comparisons").glob("*.py")
    ) + [
        ROOT / "spectral_utils" / "paper_exact" / "evaluator.py",
        ROOT / "spectral_utils" / "unified_causal_evaluation.py",
        Path(__file__).resolve(),
    ]
    evaluator_paths = sorted(set(evaluator_paths), key=lambda path: str(path.relative_to(ROOT)))
    evaluator_components = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for path in evaluator_paths
    ]
    evaluator_hash = canonical_sha256(evaluator_components)
    validation_asset = _asset(UNIFIED_VALIDATION)
    global_asset = _asset(UNIFIED_GLOBAL)
    prefix_step_asset = _asset(PREFIX_STEP272)
    prefix_historical_asset = _asset(PREFIX_HISTORICAL)
    prefix_replay_assets: dict[str, list[dict[str, Any]]] = {}
    prefix_fit_audits = prefix["audits"]["incumbent_replay"]["fit_audits"]
    prefix_replay_revision = prefix["audits"]["incumbent_replay"]["revision"]
    for method_id in (PREFIX_IU28_METHOD_ID, PREFIX_HISTORICAL_DEEPCONF_METHOD_ID):
        method_assets = []
        for family, fit_audit in sorted(prefix_fit_audits.items()):
            if method_id == PREFIX_IU28_METHOD_ID:
                preimage = {
                    "revision": prefix_replay_revision,
                    "method": method_id,
                    "parameter_sha256": fit_audit["parameter_sha256"],
                    "dependency_hashes": fit_audit["dependency_hashes"],
                    "artifact_hashes": fit_audit["artifact_hashes"],
                    "anchors": fit_audit["anchor_audits"],
                }
            else:
                preimage = {
                    "revision": prefix_replay_revision,
                    "method": method_id,
                    "window": 64,
                    "orientation": "risk_is_negative_lowest_group_confidence",
                    "streaming_utils_sha256": fit_audit["dependency_hashes"][
                        "streaming_utils.py"
                    ],
                    "artifact_hashes": fit_audit["artifact_hashes"],
                    "anchors": fit_audit["anchor_audits"],
                }
            expected_fingerprint = canonical_sha256(preimage)
            observed_fingerprints = {
                str(row["source_artifact_hash"])
                for row in prefix["records"]
                if row["method_id"] == method_id and row.get("family") == family
            }
            if observed_fingerprints != {expected_fingerprint}:
                raise ValueError(
                    f"{method_id}/{family}: replay fingerprint lacks its canonical preimage"
                )
            method_assets.append(
                _derived_asset(
                    artifact_id=f"prefix-frozen-replay/{method_id}/{family}",
                    uri=f"derived:prefix-frozen-replay/{method_id}/{family}",
                    projection={
                        "schema": "prefix_frozen_replay_fingerprint_v1",
                        "method_id": method_id,
                        "family": family,
                    },
                    source_fingerprint_aliases=[expected_fingerprint],
                    source_fingerprint_preimages=[preimage],
                )
            )
        if len(method_assets) != 4:
            raise ValueError(f"expected four family replay fingerprints: {method_id}")
        prefix_replay_assets[method_id] = method_assets
    pb_assets = [_asset(path) for path in pb["paths"].values()]
    dufs_fit_ledgers_by_family = {
        str(row["family"]): row for row in global_pb["dufs_replay"]["fit_ledgers"]
    }
    if len(dufs_fit_ledgers_by_family) != 4:
        raise ValueError("expected four mixed-v2 DUFS family replay fingerprints")
    dufs_replay_assets = []
    for family, ledger in sorted(dufs_fit_ledgers_by_family.items()):
        preimage = {
            key: value
            for key, value in ledger.items()
            if key
            not in {
                "fit_id_sha256",
                "model_state",
                "source_fingerprint",
                "model_diagnostics",
                "labels_read_during_score_construction",
            }
        }
        expected_fingerprint = canonical_sha256(preimage)
        if expected_fingerprint != ledger["source_fingerprint"]:
            raise ValueError(f"{family}: DUFS replay fingerprint preimage drift")
        dufs_replay_assets.append(
            _derived_asset(
                artifact_id=(
                    f"global-frozen-replay/{MIXED_V2_DUFS_NO_LENGTH_METHOD_ID}/{family}"
                ),
                uri=(
                    f"derived:global-frozen-replay/"
                    f"{MIXED_V2_DUFS_NO_LENGTH_METHOD_ID}/{family}"
                ),
                projection={
                    "schema": "global_mixed_v2_dufs_replay_fingerprint_v1",
                    "method_id": MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
                    "family": family,
                },
                source_fingerprint_aliases=[expected_fingerprint],
                source_fingerprint_preimages=[preimage],
            )
        )
    dufs_qwen_assets = [
        _verified_local_asset(
            path,
            expected_sha256=global_pb["dufs_replay"]["qwen_source_hashes"][family],
        )
        for family, path in global_pb["dufs_replay"]["qwen_paths"].items()
    ]
    dufs_code_assets = [
        _asset(path)
        for path in (
            ROOT / "spectral_utils" / "historical_multitask_baselines.py",
            ROOT / "spectral_utils" / "dufs_liu_feature_contract.py",
            ROOT / "spectral_utils" / "feature_contract.py",
            ROOT / "spectral_utils" / "feature_utils.py",
            ROOT / "spectral_utils" / "adapted_dufs.py",
            ROOT / "spectral_utils" / "laplacian_upcr.py",
            ROOT / "spectral_utils" / "online_convergence.py",
            ROOT / "spectral_utils" / "repeated_measurement_reliability.py",
            ROOT / "spectral_utils" / "repgrid_scoring.py",
            ROOT / "spectral_utils" / "upcr.py",
            ROOT / "scripts" / "gl_liu_v1" / "run.py",
            ROOT / "scripts" / "processbench_latent_state_v1" / "run.py",
            ROOT / "scripts" / "evaluate_unified_causal_classic30_v1.py",
        )
    ]
    dufs_fit_ids_asset = _asset(CLASSIC_GLOBAL_FIT_IDS)
    dufs_classic_definition_asset = _asset(CLASSIC_GLOBAL_RUN_DEFINITION)
    dufs_anchor_asset = _asset(DUFS_ANCHOR_MANIFEST)
    dufs_method_assets = [
        dufs_fit_ids_asset,
        dufs_classic_definition_asset,
        dufs_anchor_asset,
        *dufs_qwen_assets,
        *pb_assets,
        *dufs_code_assets,
        *dufs_replay_assets,
    ]
    external_assets = {
        method_id: [_asset(path) for path in localization["external"]["source_paths"][method_id]]
        for method_id in EXTERNAL_LOCALIZATION
    }
    prmbench_assets = [
        _asset(path)
        for path in (
            PRMBENCH_METRICS,
            PRMBENCH_FREEZE,
            PRMBENCH_DIAGNOSTICS,
            PRMBENCH_TELEMETRY,
            PRMBENCH_PRM,
        )
    ]
    external_assets["prm_qwen25_math_7b"] = [
        *external_assets["prm_qwen25_math_7b"],
        _asset(PRMBENCH_PRM),
    ]
    l1 = localization["external"]["l1"]
    l1_asset = _derived_asset(
        artifact_id="l1_uprm_judge_full/indexed-shard-package",
        uri="gdrive:hallucination_detection/cluster_results/paper_exact/l1_uprm_judge_full",
        artifact_kind="composite-ledger",
        projection={
            "schema": "indexed_pickle_package_projection_v1",
            "package_hash": l1["package_hash"],
            "provenance": l1["provenance"],
            "members": l1["shard_assets"],
        },
        source_fingerprint_aliases=[l1["package_hash"]],
        source_fingerprint_preimages=[l1["provenance"]],
    )
    drive_observation = build_drive_metadata_observation()
    drive_prefix = str(drive_observation["remote_prefix"]).rstrip("/")

    def drive_member_asset(member: Mapping[str, Any]) -> dict[str, Any]:
        path = str(member["path"])
        return _remote_asset(
            artifact_id=f"drive-metadata/{path}",
            uri=f"{drive_prefix}/{path}",
            size_bytes=int(member["size_bytes"]),
            sha256=str(member["sha256"]),
        )

    l0_asset = drive_member_asset(L0_INVENTORY)
    l1_remote_assets = [drive_member_asset(member) for member in L1_METADATA]
    s1_remote_assets = [drive_member_asset(member) for member in S1_METADATA]
    s2_remote_assets = [
        drive_member_asset(member) for member in S2_COMPLETE_METADATA
    ]
    m2_remote_assets = [drive_member_asset(member) for member in M2_STATUS_METADATA]
    partial_status_payloads = {
        "refrain_s1": {
            "finished": 512,
            "expected": 1000,
            "failed": 0,
            "summary_present": False,
        },
        "deepconf_m2": {
            "finished": 12370,
            "expected": 122880,
            "failed": 0,
            "formal_checkpoint_stale": True,
            "raw_logit_audit_n": 0,
        },
        "leash_mistral": {
            "aqua_finished": 0,
            "aqua_expected": 762,
            "gsm8k_finished": 0,
            "gsm8k_expected": 900,
            "failure": "tokenizer has no chat template",
        },
    }
    partial_status_assets = {
        method_id: _derived_asset(
            artifact_id=f"drive-snapshot/{method_id}/2026-08-18",
            uri=f"derived:read-only-drive-snapshot/{method_id}/2026-08-18",
            projection={
                "schema": "partial_acquisition_status_projection_v1",
                "method_id": method_id,
                "status": payload,
                "metadata_members": (
                    list(S1_METADATA)
                    if method_id == "refrain_s1"
                    else list(M2_STATUS_METADATA)
                    if method_id == "deepconf_m2"
                    else []
                ),
            },
        )
        for method_id, payload in partial_status_payloads.items()
    }
    cache_root = localization["external"]["l1_dir"].parent

    def reconciled_local_metadata_asset(
        member: Mapping[str, Any], *, expected_run_root: str
    ) -> dict[str, Any]:
        remote_path = PurePosixPath(str(member["path"]))
        if (
            len(remote_path.parts) < 3
            or remote_path.parts[0] != "paper_exact"
            or remote_path.parts[1] != expected_run_root
        ):
            raise ValueError(f"unexpected frozen Drive metadata path: {remote_path}")
        local_path = cache_root.joinpath(*remote_path.parts[1:])
        local_asset = _verified_local_asset(
            local_path,
            expected_sha256=str(member["sha256"]),
        )
        if int(local_asset["size_bytes"]) != int(member["size_bytes"]):
            raise ValueError(
                f"local/Drive size drift for {remote_path}: "
                f"expected {member['size_bytes']}, observed {local_asset['size_bytes']}"
            )
        return local_asset

    l1_run_id = "l1_uprm_judge_full"
    l1_metadata_assets = [
        reconciled_local_metadata_asset(member, expected_run_root=l1_run_id)
        for member in L1_METADATA
    ]
    s2_run_ids = sorted(
        {PurePosixPath(str(member["path"])).parts[1] for member in S2_COMPLETE_METADATA}
    )
    if len(s2_run_ids) != 6:
        raise ValueError("frozen S2 Drive metadata must contain exactly six complete runs")
    s2_metadata_assets = []
    for member in S2_COMPLETE_METADATA:
        run_id = PurePosixPath(str(member["path"])).parts[1]
        s2_metadata_assets.append(
            reconciled_local_metadata_asset(member, expected_run_root=run_id)
        )
    if len(s2_metadata_assets) != 30:
        raise ValueError("expected 30 locally reconciled S2 metadata objects")

    s2_shard_paths = sorted(
        path
        for run_id in s2_run_ids
        for path in (cache_root / run_id / "shards").glob("*.pkl")
    )
    if not s2_shard_paths:
        raise FileNotFoundError("no shards found for the six frozen complete S2 runs")
    s2_assets = [_asset(path) for path in s2_shard_paths]
    unified_method_assets = [
        validation_asset,
        global_asset,
        _asset(UNIFIED_VALIDATION.parent / "RUN_DEFINITION.json"),
        _asset(UNIFIED_GLOBAL.parent / "RUN_DEFINITION.json"),
        _asset(ROOT / "docs" / "experiments" / "UNIFIED_CAUSAL_IU_V1.md"),
        _asset(ROOT / "spectral_utils" / "unified_causal_iu.py"),
        _asset(ROOT / "spectral_utils" / "unified_causal_evaluation.py"),
        _asset(ROOT / "scripts" / "run_unified_causal_iu_v1.py"),
        _asset(ROOT / "scripts" / "evaluate_unified_causal_classic30_v1.py"),
    ]
    if unified_tree is not None:
        unified_method_assets.append(
            _derived_asset(
                artifact_id="unified-causal-iu-v1/source-worktree-manifest",
                uri="derived:unified-causal-iu-v1/source-worktree-manifest",
                artifact_kind="composite-ledger",
                projection={
                    "schema": "unified_source_worktree_projection_v1",
                    "manifest": unified_tree,
                },
            )
        )
    localization_replay_asset = _derived_asset(
        artifact_id="localization-frozen-replay/fit-ledgers",
        uri="derived:localization-frozen-replay/fit-ledgers",
        projection={
            "schema": "localization_frozen_replay_fit_ledgers_v1",
            "lane_revision": localization["replay"]["lane_revision"],
            "source_artifact_hash": localization["replay"]["source_artifact_hash"],
            "source_paths": localization["replay"]["source_paths"],
            "fit_ledgers": localization["replay"]["fit_ledgers"],
        },
    )
    localization_protocol_assets = [
        localization_replay_asset,
        _asset(ROOT / "spectral_utils" / "fair_comparisons" / "localization.py"),
        _asset(ROOT / "spectral_utils" / "local_online_comprehensive.py"),
        _asset(ROOT / "scripts" / "run_global_local_online_architecture_v2.py"),
        _asset(ROOT / "scripts" / "run_local_online_comprehensive_stage1.py"),
        _asset(ROOT / "scripts" / "gl_liu_v1" / "localization" / "evidence_drop.py"),
        _asset(
            ROOT
            / "scripts"
            / "gl_liu_v1"
            / "localization"
            / "localization_metrics.py"
        ),
        _asset(ROOT / "results" / "local_online_comprehensive_v1" / "RUN_MANIFEST.json"),
        _asset(
            ROOT
            / "results"
            / "local_online_comprehensive_v1"
            / "STAGE_1_LOCAL_SELECTION.json"
        ),
        *pb_assets,
    ]
    prefix_step_protocol_assets = [
        prefix_step_asset,
        _asset(ROOT / "scripts" / "run_global_local_online_architecture_v2.py"),
        _asset(
            ROOT / "results" / "global_local_online_architecture_v2" / "RUN_MANIFEST.json"
        ),
        _asset(
            ROOT
            / "results"
            / "global_local_online_architecture_v2"
            / "ARCHITECTURE_SCORE_FREEZE.json"
        ),
        _asset(ROOT / "results" / "global_local_online_architecture_v2" / "DECISION.json"),
    ]
    prefix_replay_physical_assets: list[dict[str, Any]] = []
    for fit_audit in prefix["audits"]["incumbent_replay"]["fit_audits"].values():
        fit_source = Path(str(fit_audit["fit_source"]))
        prefix_replay_physical_assets.append(
            _verified_local_asset(
                fit_source,
                expected_sha256=str(fit_audit["fit_source_sha256"]),
            )
        )
        cell_root = (
            ROOT
            / "results"
            / "early_online_localization_models_v1"
            / str(fit_audit["cell_id"])
        )
        for key, filename in (
            ("result", "result.json"),
            ("calibration_scores", "scores_calibration.csv"),
            ("evaluation_scores", "scores_evaluation.csv"),
        ):
            prefix_replay_physical_assets.append(
                _verified_local_asset(
                    cell_root / filename,
                    expected_sha256=str(fit_audit["artifact_hashes"][key]),
                )
            )
    prefix_replay_code_assets = [
        _asset(ROOT / "spectral_utils" / "fair_comparisons" / "prefix.py"),
        _asset(ROOT / "spectral_utils" / "online_convergence.py"),
        _asset(ROOT / "spectral_utils" / "fixed_application_pipelines.py"),
        _asset(ROOT / "spectral_utils" / "repeated_measurement_reliability.py"),
        _asset(ROOT / "spectral_utils" / "upcr.py"),
        _asset(ROOT / "spectral_utils" / "streaming_utils.py"),
    ]
    twentyfour_paths = [
        ROOT / "results" / "dependency_fusion_raw" / "cells.npz",
        ROOT / "results" / "frozen_24cell_benchmark" / "SCORE_FREEZE_MANIFEST.json",
        ROOT / "results" / "frozen_24cell_benchmark" / "RUN_DEFINITION.json",
        ROOT / "results" / "frozen_24cell_benchmark" / "FIT_COMPLETE.json",
    ]
    twentyfour_assets = [_asset(path) for path in twentyfour_paths]
    assets = [
        _asset(PROTOCOL),
        *[_asset(path) for path in evaluator_paths],
        validation_asset,
        global_asset,
        *unified_method_assets,
        prefix_step_asset,
        prefix_historical_asset,
        *prefix_step_protocol_assets,
        *prefix_replay_physical_assets,
        *prefix_replay_code_assets,
        *[
            asset
            for method_assets in prefix_replay_assets.values()
            for asset in method_assets
        ],
        *pb_assets,
        *dufs_method_assets,
        *[asset for values in external_assets.values() for asset in values],
        *localization_protocol_assets,
        *prmbench_assets,
        l1_asset,
        l0_asset,
        *l1_remote_assets,
        *s1_remote_assets,
        *s2_remote_assets,
        *m2_remote_assets,
        *partial_status_assets.values(),
        *l1_metadata_assets,
        *s2_assets,
        *s2_metadata_assets,
        *twentyfour_assets,
    ]
    # Artifact IDs are paths and therefore duplicates across method use are expected;
    # the package asset registry contains one physical/virtual artifact each.
    unique_assets = {asset["artifact_id"]: asset for asset in assets}

    methods = []
    project_access = _access("single generated trace token telemetry", "frozen project development", 0, 1)
    methods.append(
        _method(
            method_id="unified28",
            display_name="Unified-28 (ordinary method of record)",
            fidelity="adapted-common-protocol",
            artifacts=unified_method_assets,
            access=project_access,
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training="frozen supervised-developed roster/signs; IU fit itself label-free on registered Qwen IDs",
            checkpoint="unified-causal-iu-v1/base7_full28/ordinary",
            deviations=["project method rather than a published competitor reproduction"],
            extra={
                "artifact_generation_commit": (
                    "uncommitted Unified source worktree based on d3ca3a4; exact worktree "
                    "and imported file hashes are package-bound"
                ),
                "frozen_roster": "seven base streams x {level,ewma16,positive_area,persistence}",
                "iu_fit": "ordinary two-component L2",
                "accumulator": "identity",
                "dufs": False,
            },
        )
    )
    methods.append(
        _method(
            method_id="fixed_reasoning_iu_prmbench",
            display_name="Fixed reasoning IU-PCR (PRMBench native)",
            fidelity="adapted-common-protocol",
            artifacts=prmbench_assets,
            access=_access("teacher-forced every-step token telemetry", "none", 0, 1),
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training="label-free frozen token model; three alignment defects excluded before scoring",
            checkpoint="fixed-application-pipelines-v1/prmbench",
            deviations=["native PRMBench step task; never pooled with ProcessBench F1"],
        )
    )
    for method_id, display, artifacts, fidelity, checkpoint, deviations in (
        (
            STEP272_METHOD_ID,
            "Dedicated Prefix incumbent: Step-272 Global/Local 0.50/0.50 peak",
            prefix_step_protocol_assets,
            "adapted-common-protocol",
            f"global-local-online-v2/{SELECTED_STEP272_ARCHITECTURE}",
            ["frozen selected Online architecture; no fair-package refit"],
        ),
        (
            PREFIX_MEAN_ENTROPY_METHOD_ID,
            "Causal mean token entropy",
            pb_assets,
            "adapted-common-protocol",
            "transparent-causal-mean-entropy-v1",
            ["transparent same-telemetry baseline"],
        ),
        (
            PREFIX_MAX_ENTROPY_METHOD_ID,
            "Causal maximum token entropy",
            pb_assets,
            "adapted-common-protocol",
            "transparent-causal-max-entropy-v1",
            ["transparent same-telemetry baseline"],
        ),
        (
            PREFIX_IU28_METHOD_ID,
            "IU-28 no-length (exact frozen CPU replay)",
            [
                prefix_historical_asset,
                *prefix_replay_assets[PREFIX_IU28_METHOD_ID],
                *prefix_replay_physical_assets,
                *prefix_replay_code_assets,
            ],
            "adapted-common-protocol",
            "global-local-online-iu-v1/iu28_no_length",
            ["exact original-fit replay on registered Llama telemetry"],
        ),
        (
            PREFIX_HISTORICAL_DEEPCONF_METHOD_ID,
            "Historical DeepConf entropy-w64 proxy",
            [
                prefix_historical_asset,
                *prefix_replay_assets[PREFIX_HISTORICAL_DEEPCONF_METHOD_ID],
                *prefix_replay_physical_assets,
                *prefix_replay_code_assets,
            ],
            "adapted-common-protocol",
            "global-local-online-iu-v1/deepconf_entropy_w64",
            [
                "exact historical proxy replay; not the equality-verified paper-exact DeepConf scalar"
            ],
        ),
    ):
        methods.append(
            _method(
                method_id=method_id,
                display_name=display,
                fidelity=fidelity,
                artifacts=artifacts,
                access=_access("single generated trace token telemetry", "none", 0, 1),
                run_commit=run_commit,
                evaluator_hash=evaluator_hash,
                training="frozen historical construction; no evaluation-label fitting",
                checkpoint=checkpoint,
                deviations=deviations,
                extra={
                    "artifact_generation_commit": (
                        run_commit
                        if method_id
                        in {PREFIX_IU28_METHOD_ID, PREFIX_HISTORICAL_DEEPCONF_METHOD_ID}
                        else "historical pre-contract source; bound by selected run artifacts"
                    )
                },
            )
        )
    methods.append(
        _method(
            method_id="classic_mixed_v2_no_length",
            display_name="Dedicated Global incumbent: mixed-v2 IU-PCR (no length)",
            fidelity="adapted-common-protocol",
            artifacts=[
                global_asset,
                dufs_fit_ids_asset,
                dufs_classic_definition_asset,
                _asset(ROOT / "scripts" / "evaluate_unified_causal_classic30_v1.py"),
            ],
            access=project_access,
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training="label-free IU fit on the original registered Qwen fit IDs",
            checkpoint="classic-mixed-v2-no-length/frozen",
            deviations=["project incumbent rather than a published paper method"],
            extra={
                "artifact_generation_commit": (
                    "uncommitted Unified source worktree based on d3ca3a4; imported file "
                    "hashes and run definition are package-bound"
                )
            },
        )
    )
    dufs_fit_ledgers = global_pb["dufs_replay"]["fit_ledgers"]
    if len(dufs_fit_ledgers) != 4:
        raise ValueError("registered mixed-v2 DUFS replay must have four family fits")
    dufs_protocols = {
        (
            tuple(row["dufs_seeds"]),
            int(row["dufs_epochs"]),
            int(row["graph_k"]),
            float(row["lambda"]),
        )
        for row in dufs_fit_ledgers
    }
    if len(dufs_protocols) != 1:
        raise ValueError("registered mixed-v2 DUFS protocol differs across families")
    dufs_seeds, dufs_epochs, dufs_k, dufs_lambda = next(iter(dufs_protocols))
    methods.append(
        _method(
            method_id=MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
            display_name="Registered mixed-v2 DUFS-LIU (lambda=0.1, no length)",
            fidelity="adapted-common-protocol",
            artifacts=dufs_method_assets,
            access=project_access,
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training=(
                "label-free fit on the exact same 32 registered Qwen3-8B IDs per "
                "family as classic_mixed_v2_no_length; evaluation labels are read "
                "only after all 3,400 Llama scores freeze"
            ),
            checkpoint="mixed-v2-dufs-liu-l0p1-no-length/frozen-replay",
            deviations=[
                "trace length removed to match the dedicated Global incumbent's access",
                "the registered length-enabled Qwen3-8B/GSM8K pre-label score anchor is reproduced before transfer",
                "secondary frozen project control; not a published paper method",
            ],
            extra={
                "fit_id_sha256": global_pb["dufs_replay"]["fit_ids"][
                    "fit_id_sha256"
                ],
                "dufs_seeds": list(dufs_seeds),
                "dufs_epochs": dufs_epochs,
                "graph_k": dufs_k,
                "lambda": dufs_lambda,
                "registered_anchor_sha256": global_pb["dufs_replay"][
                    "anchor_audit"
                ]["observed_score_sha256"],
                "provenance_audit_sha256": global_pb["dufs_replay"][
                    "provenance_audit"
                ]["audit_sha256"],
                "family_model_state_sha256": {
                    row["family"]: row["model_state_sha256"]
                    for row in dufs_fit_ledgers
                },
                "labels_read_during_score_construction": False,
                "outcome_selection_performed": False,
                "artifact_generation_commit": run_commit,
            },
        )
    )
    for method_id in GLOBAL_VALIDATION_METHODS.values():
        methods.append(
            _method(
                method_id=method_id,
                display_name=method_id.replace("_", " "),
                fidelity="adapted-common-protocol",
                artifacts=[
                    validation_asset,
                    _asset(UNIFIED_VALIDATION.parent / "RUN_DEFINITION.json"),
                    _asset(ROOT / "spectral_utils" / "unified_causal_iu.py"),
                ],
                access=project_access,
                run_commit=run_commit,
                evaluator_hash=evaluator_hash,
                training="frozen historical registered control; no evaluation-label refit",
                checkpoint=f"unified-validation/{method_id}",
                deviations=["historical control; not the ordinary Unified-28 method of record"],
                extra={
                    "artifact_generation_commit": (
                        "uncommitted Unified source worktree based on d3ca3a4; imported "
                        "validation files are package-bound"
                    )
                },
            )
        )
    methods.append(
        _method(
            method_id="max_entropy_global",
            display_name="Maximum token entropy",
            fidelity="adapted-common-protocol",
            artifacts=pb_assets,
            access=_access("single trace token entropy", "none", 0, 1),
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training="none; only fair-comparison operating thresholds are cross-fit",
            checkpoint="transparent-max-entropy-v1",
            deviations=["transparent common-protocol baseline"],
        )
    )

    for method_id, display in (
        (DEDICATED_METHOD_ID, "Dedicated Local incumbent: family6 + level + step_top5mean"),
        (MAX_ENTROPY_METHOD_ID, "Maximum entropy + top-five step mean"),
        (GL_LIU_METHOD_ID, "GL-LIU frozen replay"),
        (MIND_GAP_METHOD_ID, "Mind-the-Gap common replay"),
    ):
        deviations = [
            "common ProcessBench calibration/evaluator; frozen score construction from original fit IDs"
        ]
        methods.append(
            _method(
                method_id=method_id,
                display_name=display,
                fidelity="adapted-common-protocol",
                artifacts=localization_protocol_assets,
                access=_access("single trace token telemetry", "none for score fit; threshold labels cross-fit", 0, 1),
                run_commit=run_commit,
                evaluator_hash=evaluator_hash,
                training="score parameters fit without labels on original registered calibration IDs; threshold only is cross-fit",
                checkpoint=f"local-online-comprehensive-v1/{method_id}",
                deviations=deviations,
                extra={"artifact_generation_commit": run_commit},
            )
        )
    for method_id, display, passes, supervision, deviations in (
        (
            "prm_qwen25_math_7b",
            "Qwen2.5-Math-PRM-7B",
            1,
            "released process-reward checkpoint",
            ["pre-contract run manifest; checkpoint hash audit pending"],
        ),
        (
            "critic_qwen25_72b_single_greedy",
            "Qwen2.5-72B critic (single greedy)",
            1,
            "critic language model",
            ["single greedy pass, not the paper's eight-sample majority protocol"],
        ),
        (
            "eq6_qwen3_8b_precontract",
            "Eq.6 Qwen3-8B reconstruction (pre-contract)",
            1,
            "judge language model",
            ["project Eq.6 reconstruction; must not be called uPRM"],
        ),
    ):
        methods.append(
            _method(
                method_id=method_id,
                display_name=display,
                fidelity="adapted-common-protocol",
                artifacts=external_assets[method_id],
                access=_access("full reasoning trace text", supervision, passes, 1),
                run_commit=run_commit,
                evaluator_hash=evaluator_hash,
                training="fixed output predictions; no fair-comparison fitting",
                checkpoint=f"existing-artifact/{method_id}",
                deviations=deviations,
                prompt="frozen in existing artifact; pre-contract provenance",
                decoding="fixed existing predictions",
            )
        )
    methods.append(
        _method(
            method_id=L1_METHOD_ID,
            display_name="uPRM Eq.6 Qwen2.5-14B control",
            fidelity="paper-specified-partial",
            artifacts=[l1_asset, *l1_metadata_assets, *l1_remote_assets],
            access=_access("full reasoning trace text", "Qwen2.5-14B judge", 1, 1),
            run_commit=run_commit,
            evaluator_hash=evaluator_hash,
            training="fixed acquired outputs; no fair-comparison fitting",
            checkpoint="paper-exact/l1_uprm_judge_full",
            deviations=["published prompt/code unavailable; project Eq.6 reconstruction"],
            prompt="frozen L1 Eq.6 reconstruction prompt",
            decoding="single frozen judge output",
            prompt_sha256=canonical_sha256(
                [
                    {"path": asset["artifact_id"], "sha256": asset["sha256"]}
                    for asset in (*l1_metadata_assets, *l1_remote_assets)
                ]
            ),
            decoding_sha256=canonical_sha256(
                [
                    {"path": asset["artifact_id"], "sha256": asset["sha256"]}
                    for asset in (*l1_metadata_assets, *l1_remote_assets)
                ]
            ),
        )
    )
    for arm, display in (
        ("cot|central", "LEASH study: CoT central arm"),
        ("leash|central", "LEASH study: LEASH central arm"),
        ("nocot|central", "LEASH study: No-CoT central arm"),
    ):
        methods.append(
            _method(
                method_id=arm,
                display_name=display,
                fidelity="paper-specified-partial",
                artifacts=[*s2_assets, *s2_metadata_assets, *s2_remote_assets],
                access=_access(
                    "generated answer trace",
                    "task model generation",
                    1,
                    1,
                    pass_scope="method generation passes including the reported answer trace",
                ),
                run_commit=run_commit,
                evaluator_hash=evaluator_hash,
                training="fixed acquisition arms; no evaluation fitting",
                checkpoint=f"paper-exact/s2/{arm}",
                deviations=["paper leaves key prompts, hyperparameters, and seed unspecified"],
                prompt=f"frozen S2 {arm} prompt",
                decoding="frozen S2 acquisition decoding",
                prompt_sha256=canonical_sha256(
                    [
                        {"path": asset["artifact_id"], "sha256": asset["sha256"]}
                        for asset in (*s2_metadata_assets, *s2_remote_assets)
                    ]
                ),
                decoding_sha256=canonical_sha256(
                    [
                        {"path": asset["artifact_id"], "sha256": asset["sha256"]}
                        for asset in (*s2_metadata_assets, *s2_remote_assets)
                    ]
                ),
            )
        )
    for method_id, display, access, deviations in (
        (
            "refrain_s1",
            "REFRAIN S1 (incomplete acquisition)",
            _access(
                "single generated trace",
                "adaptive REFRAIN policy",
                1,
                1,
                pass_scope="method generation passes including the reported answer trace",
            ),
            ["512/1000 acquired; no final summary; official repository non-runnable"],
        ),
        (
            "deepconf_m2",
            "DeepConf M2 (partial multi-trace acquisition)",
            _access(
                "multi-trace answer samples",
                "official pinned DeepConf code",
                4096,
                4096,
                pass_scope="method generation passes in the requested multi-trace frontier",
            ),
            ["12,370/122,880 traces; stale formal checkpoint; raw-logit audit n=0"],
        ),
        (
            "leash_mistral",
            "LEASH Mistral failed cells",
            _access(
                "single generated trace",
                "LEASH policy",
                1,
                1,
                pass_scope="method generation passes including the reported answer trace",
            ),
            ["both cells failed before inference because tokenizer had no chat template"],
        ),
    ):
        methods.append(
            _method(
                method_id=method_id,
                display_name=display,
                fidelity="paper-specified-partial",
                artifacts=(
                    [*s1_remote_assets, partial_status_assets[method_id]]
                    if method_id == "refrain_s1"
                    else [*m2_remote_assets, partial_status_assets[method_id]]
                    if method_id == "deepconf_m2"
                    else [partial_status_assets[method_id]]
                ),
                access=access,
                training="fixed acquisition protocol; incomplete or failed",
                checkpoint=f"{method_id}/read-only-partial-snapshot-2026-08-18",
                prompt=f"{method_id} acquisition prompt status bound by snapshot",
                decoding=f"{method_id} acquisition decoding status bound by snapshot",
                evaluator_hash=evaluator_hash,
                run_commit=run_commit,
                deviations=deviations,
            )
        )
    methods.append(
        make_method_entry(
            method_id="unified28_stopping",
            display_name="Unified-28 stopping policy (ineligible)",
            fidelity="published-context-only",
            source_artifacts=[validation_asset],
            access=_access("potential single-trace policy", "not frozen", None, None),
            training_label_use="no stopping threshold or closure branch is frozen",
            checkpoint_revision=None,
            prompt_sha256=None,
            decoding_sha256=None,
            evaluator_sha256=evaluator_hash,
            run_commit=run_commit,
            deviations=["no real forced-closure outputs; potential savings are not realized savings"],
            extra={
                "package_build_commit": run_commit,
                "artifact_generation_commit": "not-applicable; no stopping artifact exists",
            },
        )
    )
    # Visible method-registry rows for competitors that cannot yield a v1 score.
    for method_id, display in (
        ("uprm_full_trained", "Full trained uPRM"),
        ("streaming_probe", "Streaming hallucination probe"),
    ):
        methods.append(
            make_method_entry(
                method_id=method_id,
                display_name=display,
                fidelity="blocked-assets",
                source_artifacts=[],
                access={
                    "input_type": "unavailable official assets",
                    "supervision": "unavailable",
                    "model_passes_per_question": None,
                    "traces_per_question": None,
                    "model_passes_scope": "unavailable",
                    "registered_source_traces_per_question": None,
                },
                training_label_use="not runnable; no estimate substituted",
                checkpoint_revision=None,
                prompt_sha256=None,
                decoding_sha256=None,
                evaluator_sha256=evaluator_hash,
                run_commit=None,
                deviations=["required official assets are unavailable"],
                extra={
                    "package_build_commit": run_commit,
                    "artifact_generation_commit": "not-applicable; official assets unavailable",
                },
            )
        )

    population_entries = [
        _population_entry_from_pb(global_pb["population"], lane="global"),
        _population_entry_from_pb(localization["population"], lane="localization"),
        *_prefix_population_entries(prefix),
        *_stopping_population_entries(stopping),
    ]
    return {
        "assets": build_asset_registry(unique_assets.values()),
        "populations": build_population_registry(population_entries),
        "methods": build_method_registry(methods),
        "evaluator_hash": evaluator_hash,
        "evaluator_components": evaluator_components,
    }


def _global_join_expectations(global_pb: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind direct Global coverage separately from frozen historical controls."""

    direct = tuple(global_pb.get("direct_method_ids", ()))
    context = tuple(global_pb.get("context_method_ids", ()))
    all_methods = tuple(global_pb.get("method_ids", ()))
    if direct != GLOBAL_DIRECT_METHOD_IDS:
        raise ValueError(f"Global direct roster drifted: {direct!r}")
    if context != GLOBAL_CONTEXT_METHOD_IDS:
        raise ValueError(f"Global context roster drifted: {context!r}")
    if all_methods != direct + context:
        raise ValueError("Global record/audit roster is not direct + context")
    return [
        {
            "population_id": PB_GLOBAL_POPULATION,
            "method_id": method_id,
            "budget": "final",
            "headline": headline,
            "table_id": (
                "global-pb-llama3400"
                if headline
                else "global-pb-frozen-search-context"
            ),
        }
        for headline, roster in ((True, direct), (False, context))
        for method_id in roster
    ]


def build_join_audit(
    *,
    records: Sequence[Mapping[str, Any]],
    registries: Mapping[str, Any],
    global_pb: Mapping[str, Any],
    localization: Mapping[str, Any],
    prefix: Mapping[str, Any],
    stopping: Mapping[str, Any],
) -> dict[str, Any]:
    expectations = _global_join_expectations(global_pb)
    for method_id in localization["method_ids"]:
        expectations.append(
            {
                "population_id": PB_LOCALIZATION_POPULATION,
                "method_id": method_id,
                "budget": "final",
                "headline": True,
            }
        )
    # Prefix and stopping expectations come only from the frozen population
    # registry.  Observed method rows are the object being audited and therefore
    # cannot define their own expected coverage.
    registered_populations = population_index(registries["populations"])
    for population in sorted(
        (
            row
            for row in registered_populations.values()
            if row["lane"] == "prefix"
        ),
        key=lambda row: row["population_id"],
    ):
        population_id = str(population["population_id"])
        eligibility = population["eligible_populations"]
        for method_id in population["included_methods"]:
            expectations.append(
                {
                    "population_id": population_id,
                    "method_id": method_id,
                    "budget": "final",
                    "eligible_row_ids": population["ordered_ids"],
                    "eligible_ordered_id_sha256": population["ordered_id_sha256"],
                    "headline": True,
                    "table_id": "prefix-registered-common",
                }
            )
            for budget in PREFIX_BUDGETS:
                eligible = eligibility[f"budget_{budget}"]
                expectations.append(
                    {
                        "population_id": population_id,
                        "method_id": method_id,
                        "budget": budget,
                        "eligible_row_ids": eligible["ordered_ids"],
                        "eligible_ordered_id_sha256": eligible[
                            "ordered_id_sha256"
                        ],
                        "headline": True,
                        "table_id": "prefix-registered-common",
                    }
                )
    for population in sorted(
        (
            row
            for row in registered_populations.values()
            if row["lane"] == "stopping"
        ),
        key=lambda row: row["population_id"],
    ):
        population_id = str(population["population_id"])
        for method_id, eligible in population["eligible_populations"].items():
            expectations.append(
                {
                    "population_id": population_id,
                    "method_id": method_id,
                    "budget": "registered-realized-tokens",
                    "match_any_budget": True,
                    "eligible_row_ids": eligible["ordered_ids"],
                    "eligible_ordered_id_sha256": eligible["ordered_id_sha256"],
                    "headline": True,
                    "table_id": "stopping-s2-six-cells",
                }
            )
    audit = audit_comparison_records(
        records,
        registries["populations"],
        registries["methods"],
        expectations=expectations,
    )
    require_clean_join(audit, headline_only=True)
    return audit


def _method_lookup(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["method_id"]: row for row in registry["methods"]}


def _report_row(
    method: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    headline_eligible: bool = True,
) -> dict[str, Any]:
    return {
        "method_id": method["method_id"],
        "method": method["display_name"],
        "fidelity": method["fidelity"],
        "access": method["access"],
        "headline_eligible": headline_eligible,
        **dict(values),
    }


def _partition_global_report_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the frozen direct roster and the non-claim historical roster.

    Fail closed if a scored Global method is omitted from either roster.  This
    prevents a future registered search control from entering the direct table
    merely because it was appended to the score record stream.
    """

    by_method: dict[str, dict[str, Any]] = {}
    for source in rows:
        method_id = str(source.get("method_id"))
        if method_id in by_method:
            raise ValueError(f"duplicate Global report row: {method_id}")
        by_method[method_id] = dict(source)
    expected = set(GLOBAL_ALL_METHOD_IDS)
    observed = set(by_method)
    if observed != expected:
        raise ValueError(
            "Global report roster mismatch: "
            f"missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}"
        )
    direct = []
    for method_id in GLOBAL_DIRECT_METHOD_IDS:
        row = dict(by_method[method_id])
        row["headline_eligible"] = True
        row["comparison_scope"] = "direct"
        direct.append(row)
    context = []
    for method_id in GLOBAL_CONTEXT_METHOD_IDS:
        row = dict(by_method[method_id])
        row["headline_eligible"] = False
        row["comparison_scope"] = "frozen-search-context-only"
        row["status"] = (
            "historical registered search/control row; excluded from direct and "
            "primary claims"
        )
        context.append(row)
    return direct, context


def _interval_text(intervals: Mapping[str, Any], key: str) -> str | None:
    value = intervals.get("statistics", {}).get(key)
    if not isinstance(value, Mapping):
        return None
    point, low, high = value.get("point"), value.get("ci_low"), value.get("ci_high")
    if any(item is None or not np.isfinite(float(item)) for item in (point, low, high)):
        return None
    return f"{float(point):+.4f} [{float(low):+.4f}, {float(high):+.4f}]"


def _stopping_interval_text(
    intervals: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    model: str,
    arm: str,
    metric: str,
) -> str | None:
    matches = [
        row
        for row in intervals
        if row.get("dataset") == dataset
        and row.get("model") == model
        and row.get("arm") == arm
        and row.get("reference_arm") == "cot"
        and row.get("metric") == metric
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"non-unique stopping interval: {dataset}/{model}/{arm}/{metric}"
        )
    point, low, high = (
        matches[0].get("point"),
        matches[0].get("lo"),
        matches[0].get("hi"),
    )
    if any(item is None or not np.isfinite(float(item)) for item in (point, low, high)):
        return None
    return f"{float(point):+.4f} [{float(low):+.4f}, {float(high):+.4f}]"


def _prefix_eligible_population_set(
    prefix: Mapping[str, Any], eligibility_key: str
) -> dict[str, Any]:
    """Bind an aggregate Prefix claim to its exact per-cell eligible ID vectors."""

    populations = []
    for source in sorted(prefix["populations"], key=lambda row: row["population_id"]):
        try:
            eligible = source["eligible_populations"][eligibility_key]
        except KeyError as exc:
            raise ValueError(
                f"Prefix population lacks registered eligibility {eligibility_key}: "
                f"{source.get('population_id')}"
            ) from exc
        populations.append(
            {
                "population_id": str(source["population_id"]),
                "n_rows": len(eligible["ordered_ids"]),
                "ordered_id_sha256": str(eligible["ordered_id_sha256"]),
            }
        )
    projection = {
        "schema": "prefix_eligible_population_set_v1",
        "eligibility_key": eligibility_key,
        "populations": populations,
    }
    return {
        **projection,
        "n_rows": sum(int(row["n_rows"]) for row in populations),
        "population_set_sha256": canonical_sha256(projection),
    }


def build_report_tables(
    *,
    global_pb: Mapping[str, Any],
    localization: Mapping[str, Any],
    prefix: Mapping[str, Any],
    stopping: Mapping[str, Any],
    twentyfour: Mapping[str, Any],
    method_registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    methods = _method_lookup(method_registry)
    global_population_hash = str(global_pb["metrics"]["ordered_id_sha256"])
    localization_population_hash = str(localization["population"].ordered_id_sha256)
    prefix_population_64 = _prefix_eligible_population_set(prefix, "budget_64")
    prefix_population_128 = _prefix_eligible_population_set(prefix, "budget_128")
    prefix_population_warning = _prefix_eligible_population_set(
        prefix, "complete_six_budget_warning"
    )
    global_primary_interval = _interval_text(
        global_pb["intervals"],
        "delta__unified28__minus__classic_mixed_v2_no_length",
    )
    localization_primary_interval = _interval_text(
        localization["intervals"],
        f"delta__unified28__minus__{DEDICATED_METHOD_ID}",
    )
    localization_published_interval = _interval_text(
        localization["intervals"],
        f"delta__unified28__minus__{MIND_GAP_METHOD_ID}",
    )
    prefix_primary_interval = _interval_text(
        prefix["intervals"],
        f"delta__unified28__minus__{STEP272_METHOD_ID}",
    )
    prefix_warning_interval = _interval_text(
        prefix["intervals"]["warning_operating_points"],
        f"delta_coverage_fpr_05__unified28__minus__{STEP272_METHOD_ID}",
    )
    global_rows = []
    for row in global_pb["metrics"]["methods"]:
        point5 = row["operating_points"]["fpr_05"]
        point10 = row["operating_points"]["fpr_10"]
        equal5 = point5["equal_family"]
        equal10 = point10["equal_family"]
        global_rows.append(
            _report_row(
                methods[row["method_id"]],
                {
                    "n": row["n"],
                    "AUROC": row["equal_family_auroc"],
                    "error AUPRC": row["equal_family_error_auprc"],
                    "TPR@5%": equal5["error_tpr"],
                    "precision@5%": equal5["error_precision"],
                    "observed FPR@5%": equal5["observed_fpr"],
                    "TPR@10%": equal10["error_tpr"],
                    "precision@10%": equal10["error_precision"],
                    "observed FPR@10%": equal10["observed_fpr"],
                    "population hash": global_population_hash,
                    "evaluator hash": methods[row["method_id"]]["evaluator_sha256"],
                    "paired Δ vs incumbent (95% CI)": (
                        global_primary_interval if row["method_id"] == "unified28" else None
                    ),
                },
            )
        )
    global_rows, global_context_rows = _partition_global_report_rows(global_rows)

    localization_rows = []
    for method_id, result in localization["scored"].items():
        metric = result["metrics"]
        localization_rows.append(
            _report_row(
                methods[method_id],
                {
                    "n": metric["n"],
                    "macro F1": metric["equal_subset_macro_f1"],
                    "error exact accuracy": metric["equal_subset_error_accuracy"],
                    "clean accuracy": metric["equal_subset_clean_accuracy"],
                    "within one": metric["within_one_error_accuracy"],
                    "parser coverage": 1.0,
                    "population hash": localization_population_hash,
                    "evaluator hash": methods[method_id]["evaluator_sha256"],
                    "paired Δ vs incumbent (95% CI)": (
                        localization_primary_interval if method_id == "unified28" else None
                    ),
                    "paired Δ vs Mind-the-Gap (95% CI)": (
                        localization_published_interval
                        if method_id == "unified28"
                        else None
                    ),
                },
            )
        )
    for method_id, result in localization["fixed_metrics"].items():
        metric = result["metrics"]
        localization_rows.append(
            _report_row(
                methods[method_id],
                {
                    "n": metric["n"],
                    "macro F1": metric["equal_subset_macro_f1"],
                    "error exact accuracy": metric["equal_subset_error_accuracy"],
                    "clean accuracy": metric["equal_subset_clean_accuracy"],
                    "within one": metric["within_one_error_accuracy"],
                    "parser coverage": result["parser_coverage"],
                    "population hash": localization_population_hash,
                    "evaluator hash": methods[method_id]["evaluator_sha256"],
                    "paired Δ vs incumbent (95% CI)": None,
                },
            )
        )
    ceiling_method_ids = {
        "prm_qwen25_math_7b",
        "critic_qwen25_72b_single_greedy",
    }
    localization_common_rows = [
        row for row in localization_rows if row["method_id"] not in ceiling_method_ids
    ]
    localization_ceiling_rows = [
        row
        for row in localization_rows
        if row["method_id"] in {"unified28", DEDICATED_METHOD_ID} | ceiling_method_ids
    ]

    prefix_rows = []
    summaries_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    budget_metrics_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for metric in prefix["metrics"]["per_cell_method"]:
        summaries_by_method[str(metric["method_id"])].append(metric)
    for metric in prefix["metrics"]["per_budget"]:
        if metric["budget"] in (64, 128):
            budget_metrics_by_method[str(metric["method_id"])].append(metric)
    prefix_display_order = [
        "unified28",
        STEP272_METHOD_ID,
        PREFIX_IU28_METHOD_ID,
        PREFIX_MEAN_ENTROPY_METHOD_ID,
        PREFIX_MAX_ENTROPY_METHOD_ID,
        PREFIX_HISTORICAL_DEEPCONF_METHOD_ID,
    ]
    prefix_display_order.extend(
        method_id
        for method_id in prefix["method_ids"]
        if method_id not in prefix_display_order
    )
    expected_prefix_populations = {
        str(row["population_id"]) for row in prefix["populations"]
    }
    for method_id in prefix_display_order:
        summaries = summaries_by_method[method_id]
        budget_metrics = budget_metrics_by_method[method_id]
        observed_populations = {str(row["population_id"]) for row in summaries}
        if observed_populations != expected_prefix_populations:
            raise ValueError(
                "a direct Prefix aggregate must use the same registered cells for every "
                f"method: {method_id} has {sorted(observed_populations)}, expected "
                f"{sorted(expected_prefix_populations)}"
            )
        warnings = prefix["warnings"].get(method_id, {})
        op5 = warnings.get("fpr_05", {})
        op10 = warnings.get("fpr_10", {})
        op5_equal = op5.get("equal_family", op5)
        op10_equal = op10.get("equal_family", op10)
        earliest = [
            row["earliest_budget_reaching_95pct_final_signal"]
            for row in summaries
            if row["earliest_budget_reaching_95pct_final_signal"] is not None
        ]
        prefix_rows.append(
            _report_row(
                methods[method_id],
                {
                    "n traces": sum(int(row["n_traces"]) for row in summaries),
                    "mean AUROC 64/128": float(
                        np.mean([row["primary_mean_auroc_64_128"] for row in summaries])
                    ),
                    "mean error AP 64/128": float(
                        np.mean([row["error_auprc"] for row in budget_metrics])
                    ),
                    "mean normalized AP 64/128": float(
                        np.mean([row["prevalence_normalized_ap"] for row in budget_metrics])
                    ),
                    "mean recovered signal 64/128": float(
                        np.mean(
                            [row["recovered_above_chance_signal"] for row in budget_metrics]
                        )
                    ),
                    "median earliest 95% budget": float(np.median(earliest))
                    if earliest
                    else None,
                    "warning coverage@5%": op5_equal.get(
                        "wrong_trace_warning_coverage"
                    ),
                    "ever FPR@5%": op5_equal.get("correct_trace_ever_warning_fpr"),
                    "first warning@5%": op5_equal.get(
                        "median_first_warning_budget_on_warned_wrong_traces"
                    ),
                    "warning coverage@10%": op10_equal.get(
                        "wrong_trace_warning_coverage"
                    ),
                    "ever FPR@10%": op10_equal.get(
                        "correct_trace_ever_warning_fpr"
                    ),
                    "first warning@10%": op10_equal.get(
                        "median_first_warning_budget_on_warned_wrong_traces"
                    ),
                    "eligible n@64": prefix_population_64["n_rows"],
                    "eligible ID-set hash@64": prefix_population_64[
                        "population_set_sha256"
                    ],
                    "eligible n@128": prefix_population_128["n_rows"],
                    "eligible ID-set hash@128": prefix_population_128[
                        "population_set_sha256"
                    ],
                    "warning eligible n": prefix_population_warning["n_rows"],
                    "complete-six-budget warning ID-set hash": prefix_population_warning[
                        "population_set_sha256"
                    ],
                    "evaluator hash": methods[method_id]["evaluator_sha256"],
                    "paired Δ vs incumbent (95% CI)": (
                        prefix_primary_interval if method_id == "unified28" else None
                    ),
                    "paired warning-coverage Δ@5% (95% CI)": (
                        prefix_warning_interval if method_id == "unified28" else None
                    ),
                },
            )
        )

    stopping_group_rosters: dict[str, dict[str, Any]] = {}
    for audit in stopping["run_audits"]:
        cell_id = f"s2::{audit['dataset']}::{audit['model']}"
        group_ids = [str(value) for value in audit["registered_group_ids"]]
        group_hash = ordered_id_sha256(group_ids)
        if group_hash != audit["paired_group_order_sha256"]:
            raise ValueError(f"stopping paired-group roster hash drift: {cell_id}")
        stopping_group_rosters[cell_id] = {
            "n_groups": len(group_ids),
            "paired_group_order_sha256": group_hash,
        }
    stopping_rows = []
    for metric in stopping["cell_metrics"]:
        method = methods[metric["method_id"]]
        pass_delta = _stopping_interval_text(
            stopping["paired_intervals"],
            dataset=str(metric["dataset"]),
            model=str(metric["model"]),
            arm=str(metric["arm"]),
            metric="pass_at_1_delta_vs_cot",
        )
        token_delta = _stopping_interval_text(
            stopping["paired_intervals"],
            dataset=str(metric["dataset"]),
            model=str(metric["model"]),
            arm=str(metric["arm"]),
            metric="mean_token_delta_vs_cot",
        )
        stopping_rows.append(
            _report_row(
                method,
                {
                    "cell": metric["cell_id"],
                    "n": metric["n_questions"],
                    "pass@1": metric["pass_at_1"],
                    "mean tokens": metric["mean_tokens_per_question"],
                    "total tokens": metric["total_tokens"],
                    "mean latency s": metric["mean_wall_s"],
                    "early stop rate": metric["early_stop_rate"],
                    "forced closure rate": metric["forced_closure_rate"],
                    "parser failure rate": metric["parser_failure_rate"],
                    "paired pass@1 Δ vs CoT (95% CI)": pass_delta,
                    "paired mean-token Δ vs CoT (95% CI)": token_delta,
                    "paired source-question groups": stopping_group_rosters[
                        str(metric["cell_id"])
                    ]["n_groups"],
                    "paired source-question group-order hash": stopping_group_rosters[
                        str(metric["cell_id"])
                    ]["paired_group_order_sha256"],
                    "evaluator hash": method["evaluator_sha256"],
                },
            )
        )

    direct = [
        {
            "table_id": "global-pb-llama3400",
            "title": "Global — Llama ProcessBench final-answer wrongness (3,400)",
            "lane": "global",
            "description": (
                "Error-positive metrics on the exact official IDs; equal-family aggregation. "
                "The registered no-length DUFS-LIU row is a frozen secondary control: it "
                "uses the incumbent's exact Qwen fit IDs and passed its pre-label anchor."
            ),
            "required_method_ids": list(GLOBAL_DIRECT_METHOD_IDS),
            "direct_claim_contract": {
                "eligible_population_hash_fields": ["population hash"],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": "unified28",
                        "right_method_id": "classic_mixed_v2_no_length",
                        "field": "paired Δ vs incumbent (95% CI)",
                    }
                ],
            },
            "rows": global_rows,
            "columns": [
                "method",
                "fidelity",
                "access",
                "n",
                "AUROC",
                "error AUPRC",
                "TPR@5%",
                "precision@5%",
                "observed FPR@5%",
                "TPR@10%",
                "precision@10%",
                "observed FPR@10%",
                "population hash",
                "evaluator hash",
                "paired Δ vs incumbent (95% CI)",
            ],
        },
        {
            "table_id": "localization-pb-3400",
            "title": "Localization — official ProcessBench first-error common protocol",
            "lane": "localization",
            "description": "Discrete out-of-fold predictions on all 3,400 rows; unparsed predictions count wrong.",
            "required_method_ids": ["unified28", DEDICATED_METHOD_ID],
            "direct_claim_contract": {
                "eligible_population_hash_fields": ["population hash"],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": "unified28",
                        "right_method_id": DEDICATED_METHOD_ID,
                        "field": "paired Δ vs incumbent (95% CI)",
                    }
                ],
            },
            "rows": localization_common_rows,
            "columns": [
                "method",
                "fidelity",
                "access",
                "n",
                "macro F1",
                "error exact accuracy",
                "clean accuracy",
                "within one",
                "parser coverage",
                "population hash",
                "evaluator hash",
                "paired Δ vs incumbent (95% CI)",
                "paired Δ vs Mind-the-Gap (95% CI)",
            ],
        },
        {
            "table_id": "localization-pb-high-access-ceilings",
            "title": "Localization — high-access PRM/critic ceilings",
            "lane": "localization",
            "description": (
                "Same 3,400 IDs and evaluator; high-access rows are visually separated "
                "and cross-tier differences are not labeled wins or losses."
            ),
            "required_method_ids": ["unified28", DEDICATED_METHOD_ID],
            "direct_claim_contract": {
                "eligible_population_hash_fields": ["population hash"],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": "unified28",
                        "right_method_id": DEDICATED_METHOD_ID,
                        "field": "paired Δ vs incumbent (95% CI)",
                    }
                ],
            },
            "rows": localization_ceiling_rows,
            "columns": [
                "method",
                "fidelity",
                "access",
                "n",
                "macro F1",
                "error exact accuracy",
                "clean accuracy",
                "within one",
                "parser coverage",
                "population hash",
                "evaluator hash",
            ],
        },
        {
            "table_id": "prefix-pb-registered-common",
            "title": "Prefix — registered Llama ProcessBench common telemetry",
            "lane": "prefix",
            "description": (
                "Four identical-row cells; strict length > budget. Historical IU/DeepConf "
                "rows from the different trace realization are excluded."
            ),
            "required_method_ids": [
                "unified28",
                STEP272_METHOD_ID,
                PREFIX_IU28_METHOD_ID,
            ],
            "direct_claim_contract": {
                "eligible_population_hash_fields": [
                    "eligible ID-set hash@64",
                    "eligible ID-set hash@128",
                    "complete-six-budget warning ID-set hash",
                ],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": "unified28",
                        "right_method_id": STEP272_METHOD_ID,
                        "field": "paired Δ vs incumbent (95% CI)",
                    },
                    {
                        "left_method_id": "unified28",
                        "right_method_id": STEP272_METHOD_ID,
                        "field": "paired warning-coverage Δ@5% (95% CI)",
                    },
                ],
            },
            "rows": prefix_rows,
            "columns": [
                "method",
                "fidelity",
                "access",
                "n traces",
                "mean AUROC 64/128",
                "mean error AP 64/128",
                "mean normalized AP 64/128",
                "mean recovered signal 64/128",
                "median earliest 95% budget",
                "warning coverage@5%",
                "ever FPR@5%",
                "first warning@5%",
                "warning coverage@10%",
                "ever FPR@10%",
                "first warning@10%",
                "eligible n@64",
                "eligible ID-set hash@64",
                "eligible n@128",
                "eligible ID-set hash@128",
                "warning eligible n",
                "complete-six-budget warning ID-set hash",
                "evaluator hash",
                "paired Δ vs incumbent (95% CI)",
                "paired warning-coverage Δ@5% (95% CI)",
            ],
        },
        {
            "table_id": "stopping-s2-six-cells",
            "title": "Stopping — six complete LEASH cells",
            "lane": "stopping",
            "description": (
                "Single-trace arms on identical questions; tokens are realized reasoning plus "
                "closure. AQuA is conservatively rescored because stored correctness was invalid; "
                "unparsed option outputs remain present and wrong."
            ),
            "required_method_ids": ["cot|central", "leash|central", "nocot|central"],
            "direct_claim_contract": {
                "group_by": ["cell"],
                "eligible_population_hash_fields": [
                    "paired source-question group-order hash"
                ],
                "evaluator_hash_field": "evaluator hash",
                "paired_intervals": [
                    {
                        "left_method_id": f"{arm}|central",
                        "right_method_id": "cot|central",
                        "field": field,
                    }
                    for arm in ("leash", "nocot")
                    for field in (
                        "paired pass@1 Δ vs CoT (95% CI)",
                        "paired mean-token Δ vs CoT (95% CI)",
                    )
                ],
            },
            "rows": stopping_rows,
            "columns": [
                "cell",
                "method",
                "fidelity",
                "access",
                "n",
                "pass@1",
                "mean tokens",
                "total tokens",
                "mean latency s",
                "early stop rate",
                "forced closure rate",
                "parser failure rate",
                "paired pass@1 Δ vs CoT (95% CI)",
                "paired mean-token Δ vs CoT (95% CI)",
                "paired source-question groups",
                "paired source-question group-order hash",
                "evaluator hash",
            ],
        },
    ]
    context_prefix_rows = []
    context_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for metric in prefix["context_metrics"]["per_cell_method"]:
        context_by_method[str(metric["method_id"])].append(metric)
    for method_id, metrics in sorted(context_by_method.items()):
        finite = [
            float(metric["primary_mean_auroc_64_128"])
            for metric in metrics
            if np.isfinite(metric["primary_mean_auroc_64_128"])
        ]
        context_prefix_rows.append(
            _report_row(
                methods[method_id],
                {
                    "method": (
                        "IU-28 no-length (historical pre-contract scores)"
                        if method_id == PREFIX_IU28_METHOD_ID
                        else methods[method_id]["display_name"]
                    ),
                    "cells": len(metrics),
                    "n traces": sum(int(metric["n_traces"]) for metric in metrics),
                    "mean AUROC 64/128": float(np.mean(finite)) if finite else None,
                    "status": "context only: pre-contract pb_qwen3_4b trace realization",
                },
                headline_eligible=False,
            )
        )
    prmbench_rows = [
        _report_row(
            methods[row["method_id"]],
            {
                "subgroup": row["subgroup"],
                "n steps": row["n_steps"],
                "AUROC": row["auroc"],
                "error AUPRC": row["error_auprc"],
                "error prevalence": row["error_prevalence"],
                "score hash": localization["prmbench_native"]["score_hash"],
            },
            headline_eligible=False,
        )
        for row in localization["prmbench_native"]["rows"]
    ]
    context = [
        {
            "table_id": "global-pb-frozen-search-context",
            "title": "Global — frozen Unified search/history controls",
            "lane": "global",
            "description": (
                "Same 3,400 IDs and evaluator, retained for auditability only. "
                "Unified DUFS lambda sweeps, the task-reweighted candidate, and "
                "ordinary-36 cannot enter direct or primary claims."
            ),
            "rows": global_context_rows,
            "columns": [
                "method",
                "fidelity",
                "access",
                "n",
                "AUROC",
                "error AUPRC",
                "TPR@5%",
                "precision@5%",
                "observed FPR@5%",
                "TPR@10%",
                "precision@10%",
                "observed FPR@10%",
                "population hash",
                "evaluator hash",
                "status",
            ],
        },
        {
            "table_id": "mind-gap-native-sla",
            "title": "Mind-the-Gap native erroneous-trace SLA",
            "lane": "localization",
            "description": "Native SLA is not mixed with official ProcessBench F1.",
            "rows": [
                _report_row(methods[MIND_GAP_METHOD_ID], localization["native_mind_gap"])
            ],
            "columns": ["method", "population", "n", "native_sla", "tolerance_one_sla", "fidelity", "access"],
        },
        {
            "table_id": "prmbench-native-every-step",
            "title": "Localization — PRMBench native every-step panel",
            "lane": "localization",
            "description": (
                "Teacher-forced step scoring after exactly three registered alignment "
                "exclusions; never mixed with ProcessBench first-error F1."
            ),
            "rows": prmbench_rows,
            "columns": [
                "method",
                "subgroup",
                "n steps",
                "AUROC",
                "error AUPRC",
                "error prevalence",
                "fidelity",
                "access",
                "score hash",
            ],
        },
        {
            "table_id": "prefix-historical-context",
            "title": "Prefix — historical IU/DeepConf context",
            "lane": "prefix",
            "description": "Eleven historical cells retained without direct claims; same question ID did not establish the same generated trace.",
            "rows": context_prefix_rows,
            "columns": ["method", "cells", "n traces", "mean AUROC 64/128", "status", "fidelity", "access"],
        },
    ]
    blocked_access = {
        "input_type": "unavailable/incomplete acquisition",
        "supervision": "not comparable",
        "model_passes_per_question": None,
        "traces_per_question": None,
    }
    partial = [
        {
            "table_id": "partial-blocked-coverage",
            "title": "Partial and blocked coverage",
            "description": "These rows are visible but excluded from headline aggregates.",
            "rows": [
                {
                    "method_id": "global_24cell",
                    "method": "24-cell Global direct panel",
                    "status": (
                        f"identity gate incomplete: {twentyfour['identity_aligned_cells']}/"
                        f"{twentyfour['headline_eligible_cells']} eligible cells aligned and "
                        f"{twentyfour['source_file_size_ready_cells']} have size-ready local "
                        f"sources; {TWENTYFOUR_BLOCKED_CELL} has no frozen identity ledger"
                    ),
                    "fidelity": "blocked-assets",
                    "access": blocked_access,
                },
                {
                    "method_id": "global_s2_cot",
                    "method": "S2 COT Global transfer panel",
                    "status": prefix["s2_audit"]["reason"],
                    "fidelity": "blocked-assets",
                    "access": blocked_access,
                },
                {
                    "method_id": "prefix_s2_cot",
                    "method": "S2 COT Prefix transfer panel",
                    "status": prefix["s2_audit"]["reason"],
                    "fidelity": "blocked-assets",
                    "access": blocked_access,
                },
                {
                    "method_id": "refrain_s1",
                    "method": "REFRAIN S1",
                    "status": "incomplete: 512/1000; no final summary",
                    "fidelity": methods["refrain_s1"]["fidelity"],
                    "access": methods["refrain_s1"]["access"],
                },
                {
                    "method_id": "deepconf_m2",
                    "method": "DeepConf M2",
                    "status": "partial acquisition: 12,370/122,880; stale checkpoint; raw-logit audit zero",
                    "fidelity": methods["deepconf_m2"]["fidelity"],
                    "access": methods["deepconf_m2"]["access"],
                },
                {
                    "method_id": "leash_mistral",
                    "method": "LEASH Mistral AQuA/GSM8K",
                    "status": "failed: tokenizer has no chat template (0/762 and 0/900)",
                    "fidelity": methods["leash_mistral"]["fidelity"],
                    "access": methods["leash_mistral"]["access"],
                },
                {
                    "method_id": "uprm_full_trained",
                    "method": "Full trained uPRM",
                    "status": "blocked-assets: no official code/checkpoint or exact-fidelity path",
                    "fidelity": "blocked-assets",
                    "access": methods["uprm_full_trained"]["access"],
                },
                {
                    "method_id": "streaming_probe",
                    "method": "Streaming probe",
                    "status": "blocked-assets: trajectories, labels, filters, splits, layer, probes, evaluator",
                    "fidelity": "blocked-assets",
                    "access": methods["streaming_probe"]["access"],
                },
                {
                    "method_id": "unified28_stopping",
                    "method": "Unified-28 stopping policy",
                    "status": "ineligible: no frozen threshold with real forced-closure outputs",
                    "fidelity": methods["unified28_stopping"]["fidelity"],
                    "access": methods["unified28_stopping"]["access"],
                },
            ],
            "columns": ["method", "status", "fidelity", "access"],
        }
    ]
    return direct, context, partial


def write_package(
    *,
    output: Path,
    pb: Mapping[str, Any],
    global_pb: Mapping[str, Any],
    localization: Mapping[str, Any],
    prefix: Mapping[str, Any],
    stopping: Mapping[str, Any],
    twentyfour: Mapping[str, Any],
    registries: Mapping[str, Any],
    join_audit: Mapping[str, Any],
    n_boot: int,
    run_commit: str,
    unified_tree: Mapping[str, Any] | None,
    runtime_paths_clean: bool,
    testing_only: bool,
    testing_deviations: Sequence[str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    evaluator_contract = _evaluator_run_contract(n_boot)
    report_identity = _report_identity(
        testing_only=testing_only,
        n_boot=n_boot,
        testing_deviations=testing_deviations,
    )
    write_json(output / "ASSET_REGISTRY.json", registries["assets"])
    write_json(output / "POPULATION_REGISTRY.json", registries["populations"])
    write_json(output / "METHOD_REGISTRY.json", registries["methods"])
    write_json(output / "JOIN_AUDIT.json", join_audit)
    write_json(output / "PROCESSBENCH_SOURCE_AUDIT.json", pb["audit"])
    write_json(
        output / "FOLD_LEDGER.json",
        {
            "schema": "fair_comparison_fold_ledger_v1",
            "global": pb["global_folds"],
            "global_sha256": pb["global_fold_hash"],
            "localization": pb["localization_folds"],
            "localization_sha256": pb["localization_fold_hash"],
        },
    )
    if unified_tree is not None:
        write_json(output / "UNIFIED_TEMP_WORKTREE_MANIFEST.json", unified_tree)

    lanes = {
        "global": {
            "records": global_pb["records"],
            "metrics": global_pb["metrics"],
            "intervals": global_pb["intervals"],
            "calibration": {
                row["method_id"]: row["operating_points"]
                for row in global_pb["metrics"]["methods"]
            },
            "audits": global_pb["adapter_audits"],
        },
        "localization": {
            "records": localization["records"],
            "metrics": {
                "score_methods": {
                    method_id: result["metrics"]
                    for method_id, result in localization["scored"].items()
                },
                "fixed_methods": localization["fixed_metrics"],
                "native_mind_gap": localization["native_mind_gap"],
            },
            "intervals": localization["intervals"],
            "calibration": {
                method_id: result["calibration_ledgers"]
                for method_id, result in localization["scored"].items()
            },
            "audits": {
                "replay_fit_ledgers": localization["replay"]["fit_ledgers"],
                "external": localization["external"]["audits"],
                "l1_provenance": localization["external"]["l1"]["provenance"],
            },
            "metric_rows": [
                {
                    "method_id": method_id,
                    "method_kind": "crossfit_score",
                    **result["metrics"],
                }
                for method_id, result in localization["scored"].items()
            ]
            + [
                {
                    "method_id": method_id,
                    "method_kind": "fixed_prediction",
                    "parser_coverage": result["parser_coverage"],
                    **result["metrics"],
                }
                for method_id, result in localization["fixed_metrics"].items()
            ],
        },
        "prefix": {
            "records": prefix["records"],
            "metrics": {
                **prefix["metrics"],
                "crossfit_ever_warning": prefix["warnings"],
            },
            "metric_rows": prefix["metrics"]["per_budget"],
            "intervals": prefix["intervals"],
            "calibration": prefix["warnings"],
            "audits": {
                **prefix["audits"],
                "warning_inputs": prefix["warning_inputs"]["audit"],
                "coverage": prefix["coverage"],
                "s2_cot_gate": prefix["s2_audit"],
            },
        },
        "stopping": {
            "records": stopping["records"],
            "metrics": {
                "cell_metrics": stopping["cell_metrics"],
                "accuracy_compute_frontier": stopping["accuracy_compute_frontier"],
            },
            "intervals": stopping["paired_intervals"],
            "audits": {
                "runs": stopping["run_audits"],
                "suite": stopping["suite_audit"],
                "pairing": stopping["pairing_audit"],
            },
            "metric_rows": stopping["cell_metrics"],
        },
    }
    lanes["global"]["metric_rows"] = global_pb["metrics"]["methods"]
    for lane, values in lanes.items():
        lane_dir = output / "lanes" / lane
        canonical = canonicalize_comparison_records(
            values["records"], registries["populations"]
        )
        write_jsonl(lane_dir / "PER_QUESTION.jsonl", canonical)
        write_long_csv(lane_dir / "PER_QUESTION_LONG.csv", canonical)
        write_json(lane_dir / "METRICS.json", _json_safe(values["metrics"]))
        write_long_csv(lane_dir / "METRICS_LONG.csv", _json_safe(values["metric_rows"]))
        write_json(lane_dir / "PAIRED_INTERVALS.json", _json_safe(values["intervals"]))
        write_json(
            lane_dir / "CALIBRATION_LEDGER.json",
            _json_safe(values.get("calibration", {})),
        )
        write_json(lane_dir / "AUDIT.json", _json_safe(values["audits"]))

    prefix_context = sorted(
        prefix["context_records"],
        key=lambda row: (
            str(row["cell_id"]),
            str(row["row_id"]),
            str(row["method_id"]),
            10**9 if row["budget"] == "final" else int(row["budget"]),
        ),
    )
    write_jsonl(output / "lanes" / "prefix" / "HISTORICAL_CONTEXT.jsonl", prefix_context)
    write_long_csv(
        output / "lanes" / "prefix" / "HISTORICAL_CONTEXT_LONG.csv",
        prefix_context,
    )
    write_json(
        output / "lanes" / "prefix" / "HISTORICAL_CONTEXT_METRICS.json",
        _json_safe(prefix["context_metrics"]),
    )
    write_json(
        output / "lanes" / "prefix" / "S2_COT_GATE.json",
        _json_safe(prefix["s2_audit"]),
    )
    write_jsonl(
        output / "lanes" / "prefix" / "WARNING_DECISIONS.jsonl",
        prefix["warning_decisions"],
    )
    write_long_csv(
        output / "lanes" / "prefix" / "WARNING_DECISIONS_LONG.csv",
        prefix["warning_decisions"],
    )
    write_json(
        output / "lanes" / "global" / "TWENTYFOUR_PREFLIGHT.json",
        _json_safe(twentyfour),
    )
    write_json(
        output / "lanes" / "global" / "TWENTYFOUR_IDENTITY_AUDIT.json",
        _json_safe(twentyfour["identity_audit"]),
    )
    write_json(
        output / "lanes" / "global" / "MIXED_V2_DUFS_REPLAY_AUDIT.json",
        _json_safe(
            {
                key: value
                for key, value in global_pb["dufs_replay"].items()
                if key not in {"records", "qwen_paths"}
            }
        ),
    )
    write_json(
        output / "lanes" / "localization" / "PRMBENCH_NATIVE.json",
        _json_safe(localization["prmbench_native"]),
    )
    write_long_csv(
        output / "lanes" / "localization" / "PRMBENCH_NATIVE.csv",
        _json_safe(localization["prmbench_native"]["rows"]),
    )
    partial_acquisition = {
        "schema": "stopping_partial_acquisition_appendices_v1",
        "refrain_s1": {
            "uri": "gdrive:hallucination_detection/cluster_results/paper_exact/s1_refrain_full/",
            "fidelity": "paper-specified-partial",
            "finished": 512,
            "expected": 1000,
            "failed": 0,
            "shards": 8,
            "bytes_total": 1718513774,
            "complete": False,
            "summary_present": False,
            "dataset_order_sha256": "f59cccea6ec23dc33ee15d110c013a2eac63fad91327b4d3ac4ba3445d874e53",
            "bandit_state_sha256": "0bd1611d69f15a20d7ba24923aa4567a839066ddad1f642392e1179462743117",
            "bandit_round": 41,
            "arm_pulls": {"0.6": 7, "0.65": 6, "0.7": 14, "0.75": 12, "0.8": 2},
            "uncertainty_rule_if_completed": (
                "condition on the realized ordered bandit policy; retain dataset order hash"
            ),
            "headline_eligible": False,
        },
        "deepconf_m2": {
            "uri": "gdrive:hallucination_detection/cluster_results/paper_exact/m2_deepconf_full/part_*/STATUS.json",
            "fidelity": "paper-specified-partial",
            "finished": 12370,
            "expected": 122880,
            "failed": 0,
            "shards": 196,
            "bytes_total": 17741405381,
            "complete_workers": 0,
            "formal_checkpoint_finished": 4608,
            "formal_checkpoint_stale": True,
            "raw_logit_audit_n": 0,
            "grouping_if_completed": "all samples of an AIME24 question remain one bootstrap group",
            "headline_eligible": False,
        },
    }
    write_json(
        output / "lanes" / "stopping" / "PARTIAL_ACQUISITION_APPENDICES.json",
        partial_acquisition,
    )

    missing_assets = {
        "schema": "fair_comparison_missing_assets_v1",
        "incomplete": {
            "refrain_s1": "512/1000, no SUMMARY.json, official repository non-runnable",
            "deepconf_m2": "12,370/122,880 live aggregate; stale formal checkpoint; raw-logit audit n=0",
            "leash_mistral": "both cells failed at tokenizer chat-template setup",
        },
        "blocked": {
            "global_24cell": {
                "missing_or_stale_sources": twentyfour["blocked"],
                "identity_failures": [
                    row
                    for row in twentyfour["identity_audit"].get("audits", [])
                    if row.get("status") != "identity-proven"
                ],
                "identity_proven_cells": twentyfour["identity_aligned_cells"],
                "identity_proven_rows": twentyfour["identity_aligned_rows"],
                "scored_cells": twentyfour["scored_cells"],
                "headline_eligible": False,
            },
            "global_s2_cot": {
                "reason": prefix["s2_audit"]["reason"],
                "missing_exact_inputs": [
                    "post-warper top-15-renormalized entropy",
                    "post-warper sampled-token negative log probability / spilled energy",
                    "post-warper top-50 log probabilities",
                ],
                "raw_substitution_allowed": False,
                "gate_artifact": "lanes/prefix/S2_COT_GATE.json",
            },
            "prefix_s2_cot": {
                "reason": prefix["s2_audit"]["reason"],
                "missing_exact_inputs": [
                    "post-warper top-15-renormalized entropy",
                    "post-warper sampled-token negative log probability / spilled energy",
                    "post-warper top-50 log probabilities",
                ],
                "additional_binding_gates": (
                    "serialized Step-272 parameters and exact source-fit-to-S2 target bindings"
                ),
                "raw_substitution_allowed": False,
                "gate_artifact": "lanes/prefix/S2_COT_GATE.json",
            },
            "prefix_other_historical_cells": (
                "seven pre-contract non-Llama cells remain context-only because Unified-28 "
                "and Step-272 were never registered on those exact trace realizations"
            ),
            "full_uprm": "official code/checkpoint unavailable",
            "streaming": [
                "BBH/MuSiQue generated trajectories",
                "Claude Sonnet 4.5 step/prefix labels",
                "logical-filter decisions",
                "split files",
                "representation layer specification",
                "trained probe checkpoints",
                "official evaluator/scoring",
            ],
        },
        "excluded_from_headline": [
            "global_24cell",
            "global_s2_cot",
            "prefix_s2_cot",
            "prefix_other_historical_cells",
            "refrain_s1",
            "deepconf_m2",
            "leash_mistral",
            "full_uprm",
            "streaming",
        ],
        "verified_recoverable_without_gpu_but_not_fetched": {
            "internalstates_gsm8k_qwen25_7b": {
                "uri": (
                    "gdrive:hallucination_detection/cluster_results/regen/"
                    "internalstates_gsm8k_qwen25_7b/raw_gsm8k_T0.8.pkl"
                ),
                "bytes": 146217844,
                "sha256": "7ff68214158c740a88baf3959dff6484b68f43f40e6f6096ca6163fb64b5f82c",
                "manifest_sha256": (
                    "0951afa2aa81c7a8f6c52f4ba6077eadc40261fca11ce231a5dee5c87a3ca4bf"
                ),
                "note": (
                    "exact frozen payload verified read-only on Drive; not fetched because "
                    "the approved minimal-movement wave authorized only L1 and six S2 cells"
                ),
            }
        },
    }
    write_json(output / "MISSING_ASSETS.json", missing_assets)
    write_json(
        output / "GPU_GATES.json",
        {
            "schema": "fair_comparison_gpu_gates_v1",
            "default": "no new GPU work",
            "gates": {
                "REFRAIN remainder": {
                    "default": "do not resume during the CPU wave",
                    "estimated_compute": "approximately 6-8 additional B200 GPU-hours",
                    "scientific_value": "one complete native REFRAIN comparison point",
                    "approval_requirements": [
                        "separate explicit approval",
                        "append-only index verification",
                        "ordered bandit-state hash verification",
                        "run-manifest drift audit",
                        "advisor confirms the native point is required",
                    ],
                },
                "LEASH Mistral": {
                    "default": "retain both failed cells visibly",
                    "estimated_compute": "approximately 2-3 B200 GPU-hours",
                    "scientific_value": "complete the four-model by two-dataset LEASH matrix",
                    "approval_requirements": [
                        "paper-grounded chat template preregistered as a new protocol",
                        "tiny parser and forced-closure pilot passes",
                        "separate explicit approval",
                    ],
                },
                "DeepConf M2": {
                    "default": "do not resume",
                    "estimated_compute": "roughly 1,000 remaining B200 GPU-hours",
                    "scientific_value": "only the native multi-trace AIME24 frontier",
                    "approval_requirements": [
                        "separate explicit approval",
                        "fresh consolidated checkpoint",
                        "raw-logit audit",
                        "verified append-only shards",
                        "CPU evidence that more questions materially change the frontier",
                    ],
                },
                "Unified stopping": {
                    "default": "out of comparison package v1",
                    "estimated_compute": "sub-1-GPU-hour pilot; full run requires a new estimate",
                    "scientific_value": "realized single-trace accuracy-compute frontier",
                    "approval_requirements": [
                        "freeze score and model hash",
                        "freeze calibration IDs and thresholds",
                        "freeze closure prompt and evaluator",
                        "offline feasibility and parser/closure checks",
                        "separate explicit approval",
                    ],
                },
                "PRM critic and Mind-the-Gap provenance": {
                    "default": "use existing artifacts with explicit pre-contract and access labels",
                    "estimated_compute": "not estimated; no rerun is authorized merely to improve provenance",
                    "scientific_value": "recover an essential exact-ID or checkpoint binding only if the existing row cannot be audited",
                    "approval_requirements": [
                        "CPU recovery of IDs, manifests, and checkpoint hashes is exhausted",
                        "the affected competitor row is essential to a stated comparison",
                        "a separate run-specific cost estimate is supplied",
                        "separate explicit approval",
                    ],
                },
                "full trained uPRM": {
                    "default": "blocked-assets",
                    "estimated_compute": "approximately 44 H200 GPU-hours",
                    "scientific_value": "exact full trained uPRM row only if an exact path appears",
                    "approval_requirements": [
                        "official code or checkpoint becomes available",
                        "new exact-fidelity protocol approval",
                    ],
                },
                "Streaming": {
                    "default": "blocked-assets; no compute",
                    "estimated_compute": "not estimable until all official assets exist",
                    "scientific_value": "official streaming-probe comparison without substitutions",
                    "approval_requirements": [
                        "all trajectories, labels, filters, splits, layer specification, probes, and evaluator available",
                        "no substitute dataset or labeller",
                        "separate explicit approval",
                    ],
                },
                "new confirmation": {
                    "default": "out of scope",
                    "estimated_compute": "not estimated",
                    "scientific_value": "none for this retrospective comparison cycle",
                    "approval_requirements": [
                        "comparison package, policies, and hashes frozen",
                        "separate preregistration and approval",
                    ],
                },
            },
        },
    )
    write_json(
        output / "RUN_DEFINITION.json",
        {
            "schema": "fair_paper_exact_comparisons_run_definition_v1",
            "package_revision": PACKAGE_REVISION,
            "protocol_path": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256_file(PROTOCOL),
            "run_commit": run_commit,
            "evaluator": evaluator_contract,
            "evaluator_sha256": registries["evaluator_hash"],
            "evaluator_components": registries["evaluator_components"],
            "bootstrap_replicates": n_boot,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "gpu_used": False,
            "cluster_jobs_launched": False,
            "drive_mutated": False,
            "method_search_performed": False,
            "dufs_search_performed": False,
            "runtime_paths_clean_at_build": bool(runtime_paths_clean),
            "testing_only": bool(testing_only),
            "testing_deviations": list(testing_deviations),
            "publication_eligible": not bool(testing_only),
            "rclone_operational_risk": "shared Google Drive client_id is scheduled for retirement during 2026",
        },
    )

    write_json(
        output / "DRIVE_SNAPSHOT.json",
        build_drive_metadata_observation(),
    )

    method_lookup = _method_lookup(registries["methods"])
    coverage_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in join_audit["coverage"]:
        coverage_by_method[str(row["method_id"])].append(row)
    coverage_rows = []
    for method_id, method in sorted(method_lookup.items()):
        coverage = coverage_by_method.get(method_id, [])
        coverage_rows.append(
            {
                "method_id": method_id,
                "display_name": method["display_name"],
                "fidelity": method["fidelity"],
                "access": method["access"],
                "headline_join_groups": len(coverage),
                "minimum_join_coverage": min(
                    (float(row["coverage"]) for row in coverage), default=None
                ),
                "headline_join_passed": bool(coverage) and all(
                    bool(row["passes"]) for row in coverage
                ),
                "populations": sorted({str(row["population_id"]) for row in coverage}),
            }
        )
    write_json(
        output / "COMPETITOR_COVERAGE_FIDELITY_ACCESS.json",
        {"schema": "competitor_coverage_fidelity_access_v1", "methods": coverage_rows},
    )
    write_long_csv(output / "COMPETITOR_COVERAGE_FIDELITY_ACCESS.csv", coverage_rows)

    direct, context, partial = build_report_tables(
        global_pb=global_pb,
        localization=localization,
        prefix=prefix,
        stopping=stopping,
        twentyfour=twentyfour,
        method_registry=registries["methods"],
    )
    write_reports(
        output,
        title=report_identity["title"],
        summary=report_identity["summary"],
        direct_tables=_json_safe(direct),
        native_context_tables=_json_safe(context),
        partial_blocked_tables=_json_safe(partial),
        provenance={
            "package_revision": PACKAGE_REVISION,
            "run_commit": run_commit,
            "evaluator_sha256": registries["evaluator_hash"],
            "processbench_ordered_id_sha256": pb["population"].ordered_id_sha256,
            "testing_only": report_identity["testing_only"],
            "publication_eligible": report_identity["publication_eligible"],
            "bootstrap_replicates": report_identity["bootstrap_replicates"],
            "bootstrap_seed": report_identity["bootstrap_seed"],
            "confidence_interval_status": report_identity[
                "confidence_interval_status"
            ],
            "testing_deviations": report_identity["testing_deviations"],
        },
    )
    hash_manifest = build_hash_manifest(output)
    write_canonical_json(output / "HASH_MANIFEST.json", hash_manifest)
    verification = verify_hash_manifest(output, hash_manifest)
    if not verification["ok"]:
        raise RuntimeError(f"final package hash verification failed: {verification['problems']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DEFAULT)
    parser.add_argument(
        "--twentyfour-source-root",
        type=Path,
        default=TWENTYFOUR_SOURCE_DEFAULT,
        help=(
            "read-only staging root containing one directory per frozen 24-cell "
            "source; serialized output records a logical root token"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--hash-unified-worktree",
        type=Path,
        default=Path("/private/tmp/hallucination-unified-causal-iu-v1"),
    )
    parser.add_argument("--skip-unified-worktree-hash", action="store_true")
    parser.add_argument(
        "--testing-only",
        action="store_true",
        help="permit explicitly stamped smoke-test deviations; output is not publication eligible",
    )
    parser.add_argument(
        "--allow-dirty-code",
        action="store_true",
        help="testing only: permit runtime implementation paths that differ from HEAD",
    )
    parser.add_argument(
        "--skip-twentyfour-identity-audit",
        action="store_true",
        help="smoke testing only: skip the multi-GB read-only 24-cell identity audit",
    )
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    testing_deviations = []
    if args.bootstrap != DEFAULT_BOOTSTRAP_REPLICATES:
        testing_deviations.append(
            f"bootstrap_replicates={args.bootstrap} (required={DEFAULT_BOOTSTRAP_REPLICATES})"
        )
    if args.skip_unified_worktree_hash:
        testing_deviations.append("unified_temporary_worktree_hash_skipped")
    if args.allow_dirty_code:
        testing_deviations.append("dirty_runtime_code_permitted")
    if args.skip_twentyfour_identity_audit:
        testing_deviations.append("twentyfour_identity_audit_skipped")
    if testing_deviations and not args.testing_only:
        raise ValueError(
            "publication builds require 2,000 bootstraps, clean runtime code, and both "
            "provenance audits; pass --testing-only for a stamped smoke build: "
            + "; ".join(testing_deviations)
        )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to mix a deterministic build with existing files: {output}"
        )
    runtime_dirty = _runtime_code_dirty()
    if runtime_dirty and not args.allow_dirty_code:
        raise RuntimeError(
            "runtime comparison code differs from HEAD; commit it before a publication build:\n"
            + runtime_dirty
        )
    run_commit = _git("rev-parse", "HEAD")
    twentyfour_source_root = args.twentyfour_source_root.resolve()
    if twentyfour_source_root == Path(twentyfour_source_root.anchor):
        raise ValueError("--twentyfour-source-root cannot be filesystem root")
    pb = build_pb_population_and_folds()
    print("[1/8] ProcessBench population/folds verified", flush=True)
    global_pb = build_global_pb(pb, n_boot=args.bootstrap)
    print("[2/8] Global ProcessBench panel scored", flush=True)
    localization = build_localization_pb(
        pb, cache_root=args.cache_root.resolve(), n_boot=args.bootstrap
    )
    localization["prmbench_native"] = load_prmbench_native_context()
    print("[3/8] ProcessBench Localization panel scored", flush=True)
    prefix = build_prefix_pb(pb, n_boot=args.bootstrap)
    prefix["s2_audit"] = audit_s2_prefix_cache(args.cache_root.resolve())
    print("[4/8] Causal Prefix direct/context panels and S2 gate scored", flush=True)
    twentyfour = twentyfour_static_preflight(
        ROOT,
        source_root=twentyfour_source_root,
        verify_score_hashes=True,
        verify_raw_hashes=False,
    )
    if args.skip_twentyfour_identity_audit:
        identity_audit = {
            "schema": "24cell_partial_identity_audit_v1",
            "skipped_for_smoke_test": True,
            "identity_proven_cells": 0,
            "identity_proven_rows": 0,
            "failed_cells": None,
            "scoring_performed": False,
            "audits": [],
            "all_ok": False,
        }
    else:
        identity_audit = twentyfour_partial_identity_audit(
            ROOT,
            source_root=twentyfour_source_root,
            cells=[row["cell_id"] for row in twentyfour["sources"]],
        )
    twentyfour["identity_audit"] = identity_audit
    twentyfour["identity_aligned_cells"] = int(
        identity_audit["identity_proven_cells"]
    )
    twentyfour["identity_aligned_rows"] = int(identity_audit["identity_proven_rows"])
    twentyfour = _portable_twentyfour_output(
        twentyfour,
        source_root=twentyfour_source_root,
    )
    print("[5/8] 24-cell source and identity preflights completed", flush=True)
    stopping = build_s2_stopping_lane(
        args.cache_root.resolve(),
        verify_hashes=True,
        n_boot=args.bootstrap,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    print("[6/8] S2 stopping panel verified/scored", flush=True)
    unified_tree = None
    if not args.skip_unified_worktree_hash:
        unified_tree = tree_manifest(args.hash_unified_worktree)
    registries = build_registries(
        pb=pb,
        global_pb=global_pb,
        localization=localization,
        prefix=prefix,
        stopping=stopping,
        run_commit=run_commit,
        unified_tree=unified_tree,
    )
    all_records = (
        global_pb["records"]
        + localization["records"]
        + prefix["records"]
        + stopping["records"]
    )
    join_audit = build_join_audit(
        records=all_records,
        registries=registries,
        global_pb=global_pb,
        localization=localization,
        prefix=prefix,
        stopping=stopping,
    )
    print("[7/8] Registries and strict join gate passed", flush=True)
    write_package(
        output=output,
        pb=pb,
        global_pb=global_pb,
        localization=localization,
        prefix=prefix,
        stopping=stopping,
        twentyfour=twentyfour,
        registries=registries,
        join_audit=join_audit,
        n_boot=args.bootstrap,
        run_commit=run_commit,
        unified_tree=unified_tree,
        runtime_paths_clean=not bool(runtime_dirty),
        testing_only=bool(args.testing_only),
        testing_deviations=testing_deviations,
    )
    print(f"[8/8] Package written and hash-verified: {output}", flush=True)


if __name__ == "__main__":
    main()
