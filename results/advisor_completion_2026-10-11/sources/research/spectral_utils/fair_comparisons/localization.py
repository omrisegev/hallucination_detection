"""ProcessBench first-error Localization replay and common-protocol scoring.

This module reconstructs only frozen, previously registered scorers.  The original
``local-online-v1`` calibration IDs fit the unsupervised score constructions; the
five fair-comparison folds fit decision thresholds only.  No ProcessBench outcome is
used to alter a feature roster, sign, fusion weight, or locator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .evaluator import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    crossfit_localization_threshold,
    localization_metrics,
    mind_the_gap_sla,
    paired_grouped_bootstrap,
)
from .folds import canonical_sha256
from .processbench import (
    PROCESSBENCH_SUBSETS,
    ProcessBenchPopulation,
    canonical_processbench_id,
    sha256_file,
)
from .registry import make_comparison_record


LOCALIZATION_LANE_REVISION = "fair_processbench_localization_v1.0.0"
UNIFIED28_METHOD_ID = "unified28"
DEDICATED_METHOD_ID = "family6_level_step_top5mean"
MAX_ENTROPY_METHOD_ID = "max_entropy_step_top5mean"
GL_LIU_METHOD_ID = "gl_liu_v1_replay"
MIND_GAP_METHOD_ID = "mind_the_gap_common_replay"

SAME_ACCESS_METHODS = (
    UNIFIED28_METHOD_ID,
    DEDICATED_METHOD_ID,
    MAX_ENTROPY_METHOD_ID,
    GL_LIU_METHOD_ID,
    MIND_GAP_METHOD_ID,
)


def _combined_artifact_hash(paths: Sequence[Path]) -> str:
    ledger = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda value: str(value))
    ]
    return canonical_sha256(ledger)


def _zfit(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), max(float(np.std(array)), 1e-12)


def _zapply(values: Sequence[float], fit: tuple[float, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - fit[0]) / fit[1]


def _load_replay_dependencies():
    """Import the exact frozen historical scorer implementations without running them."""

    from scripts.run_global_local_online_architecture_v2 import (
        _cell_path,
        _peak_locator,
        fit_registered_global,
        fit_registered_local,
        load_rows,
    )
    # Importing stage1 installs its pinned Mind-the-Gap source directory on sys.path;
    # its main routine is protected by the ordinary __main__ guard.
    from scripts.run_local_online_comprehensive_stage1 import (
        _stage_partition,
        _step_top5_locator,
    )
    from spectral_utils.local_online_comprehensive import (
        fit_references,
        fit_trajectory_head_prepared,
        prepare_trace,
    )
    from evidence_drop import EVIDENCE_FNS, evidence_drop_risk
    from localization_metrics import step_drop_scores

    return {
        "cell_path": _cell_path,
        "peak_locator": _peak_locator,
        "fit_registered_global": fit_registered_global,
        "fit_registered_local": fit_registered_local,
        "load_rows": load_rows,
        "stage_partition": _stage_partition,
        "step_top5_locator": _step_top5_locator,
        "fit_references": fit_references,
        "fit_trajectory_head_prepared": fit_trajectory_head_prepared,
        "prepare_trace": prepare_trace,
        "evidence_fns": EVIDENCE_FNS,
        "evidence_drop_risk": evidence_drop_risk,
        "step_drop_scores": step_drop_scores,
    }


def _mind_gap_score(row: Mapping[str, Any], dependencies: Mapping[str, Any]) -> tuple[float, int]:
    evidence = dependencies["evidence_fns"]["shannon"](row, 20)
    detector = dependencies["evidence_drop_risk"](evidence, M=5, ema_span=5)
    scores = dependencies["step_drop_scores"](
        evidence, row["step_token_spans"], ema_span=5
    )
    locator = int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1
    return float(detector), locator


def replay_same_access_methods(
    repo_root: str | Path,
    population: ProcessBenchPopulation,
    *,
    folds: Mapping[str, int],
) -> dict[str, Any]:
    """Replay the four frozen transparent/incumbent scorers on all official rows.

    The returned rows are *pre-threshold*: each contains one frozen detector score and
    one frozen locator.  :func:`crossfit_score_method` is the only function allowed to
    turn those values into clean/error decisions.
    """

    root = Path(repo_root).resolve()
    dependencies = _load_replay_dependencies()
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in SAME_ACCESS_METHODS if method != UNIFIED28_METHOD_ID
    }
    fit_ledgers = []
    source_paths = []

    for subset in PROCESSBENCH_SUBSETS:
        path = Path(dependencies["cell_path"]("llama31_8b", subset)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"ProcessBench source escaped repository root: {path}") from exc
        source_paths.append(path)
        rows = dependencies["load_rows"](path)
        by_id = {str(row["_unit"]): row for row in rows}
        expected_ids = [population.rows[row_id].official_id for row_id in population.ids_for_subset(subset)]
        if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
            raise ValueError(f"{subset}: replay source does not match the official population")
        ordered_rows = [by_id[official_id] for official_id in expected_ids]
        calibration = [
            row
            for row in ordered_rows
            if dependencies["stage_partition"](subset, row["_unit"]) == "calibration"
        ]
        if not calibration:
            raise ValueError(f"{subset}: frozen historical calibration split is empty")

        references = dependencies["fit_references"](calibration)
        prepared_cal = [dependencies["prepare_trace"](row, references) for row in calibration]
        prepared_all = [dependencies["prepare_trace"](row, references) for row in ordered_rows]
        family_head = dependencies["fit_trajectory_head_prepared"](
            prepared_cal,
            name="fair_v1_frozen_family6_level",
            representation="family6",
            operators=("level",),
        )
        family_curves = [
            family_head.curve_from_level(item.representations["family6"])
            for item in prepared_all
        ]

        global_model = dependencies["fit_registered_global"](calibration)
        local_model = dependencies["fit_registered_local"](calibration)
        local_curves = [local_model.curve(row) for row in ordered_rows]
        calibration_ids = {str(row["_unit"]) for row in calibration}
        cal_positions = [
            index for index, row in enumerate(ordered_rows) if str(row["_unit"]) in calibration_ids
        ]
        global_scores = np.asarray([global_model.score(row) for row in ordered_rows], dtype=float)
        local_scores = np.asarray([float(np.max(curve)) for curve in local_curves], dtype=float)
        global_fit = _zfit(global_scores[cal_positions])
        local_fit = _zfit(local_scores[cal_positions])
        gl_liu_scores = 0.75 * _zapply(global_scores, global_fit) + 0.25 * _zapply(
            local_scores, local_fit
        )

        fit_ledger = {
            "subset": subset,
            "source_path": str(path.relative_to(root)),
            "source_sha256": sha256_file(path),
            "fit_id_count": len(calibration),
            "fit_ids": sorted(calibration_ids),
            "fit_id_sha256": canonical_sha256(sorted(calibration_ids)),
            "family6_head": {
                "representation": "family6",
                "operators": ["level"],
                "locator": "step_top5mean",
                "diagnostics": family_head.diagnostics,
            },
            "gl_liu": {
                "global_weight": 0.75,
                "local_weight": 0.25,
                "locator": "peak",
                "global_diagnostics": global_model.diagnostics,
                "local_diagnostics": local_model.diagnostics,
                "global_zfit": list(global_fit),
                "local_zfit": list(local_fit),
            },
            "labels_used_in_score_fit": False,
        }
        fit_ledger["fit_hash"] = canonical_sha256(fit_ledger)
        fit_ledgers.append(fit_ledger)

        for index, row in enumerate(ordered_rows):
            row_id = canonical_processbench_id(subset, row["_unit"])
            pop = population.rows[row_id]
            base = {
                "row_id": row_id,
                "group_id": pop.group_id,
                "cell_id": pop.cell_id,
                "family": subset,
                "subset": subset,
                "first_error": int(pop.localization_label),
                "stratify_label": int(pop.localization_label != -1),
                "fold": int(folds[row_id]),
                "fit_hash": fit_ledger["fit_hash"],
                # A row depends only on its subset's frozen telemetry artifact.  The
                # method registry lists all four artifacts and the strict join accepts
                # this exact member hash (rather than an invented package digest).
                "source_artifact_hash": fit_ledger["source_sha256"],
            }
            curve = family_curves[index]
            rows_by_method[DEDICATED_METHOD_ID].append(
                {
                    **base,
                    "continuous_score": float(np.max(curve)),
                    "locator": int(dependencies["step_top5_locator"](curve, row)),
                }
            )
            entropy = np.asarray(row["token_entropies"], dtype=float)
            rows_by_method[MAX_ENTROPY_METHOD_ID].append(
                {
                    **base,
                    "continuous_score": float(np.max(entropy)),
                    "locator": int(dependencies["step_top5_locator"](entropy, row)),
                }
            )
            rows_by_method[GL_LIU_METHOD_ID].append(
                {
                    **base,
                    "continuous_score": float(gl_liu_scores[index]),
                    "locator": int(dependencies["peak_locator"](local_curves[index], row)),
                }
            )
            mind_score, mind_locator = _mind_gap_score(row, dependencies)
            rows_by_method[MIND_GAP_METHOD_ID].append(
                {
                    **base,
                    "continuous_score": mind_score,
                    "locator": mind_locator,
                }
            )

    source_hash = _combined_artifact_hash(source_paths)
    expected_order = list(population.ordered_ids)
    for method_id, rows in rows_by_method.items():
        if [row["row_id"] for row in rows] != expected_order:
            raise ValueError(f"{method_id}: replay order differs from official ProcessBench order")
        if len(rows) != len(expected_order):
            raise ValueError(f"{method_id}: incomplete ProcessBench replay")
    return {
        "schema": "fair_localization_replay_v1",
        "lane_revision": LOCALIZATION_LANE_REVISION,
        "source_artifact_hash": source_hash,
        "source_paths": [str(path.relative_to(root)) for path in source_paths],
        "fit_ledgers": fit_ledgers,
        "methods": rows_by_method,
    }


def load_unified28_prethreshold_rows(
    validation_jsonl: str | Path,
    population: ProcessBenchPopulation,
    *,
    folds: Mapping[str, int],
) -> dict[str, Any]:
    """Load the frozen U28 detector score and locator without reusing old decisions."""

    path = Path(validation_jsonl)
    source_hash = sha256_file(path)
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("candidate") != "base7_full28":
                continue
            row_id = canonical_processbench_id(row["family"], row["unit"])
            if row_id in by_id:
                raise ValueError(f"duplicate Unified-28 row at line {line_number}: {row_id}")
            pop = population.rows.get(row_id)
            if pop is None:
                raise ValueError(f"Unified-28 contains an unregistered row: {row_id}")
            if int(row["target_step"]) != pop.localization_label:
                raise ValueError(f"Unified-28 label conflict: {row_id}")
            by_id[row_id] = {
                "row_id": row_id,
                "group_id": pop.group_id,
                "cell_id": pop.cell_id,
                "family": pop.subset,
                "subset": pop.subset,
                "first_error": int(pop.localization_label),
                "stratify_label": int(pop.localization_label != -1),
                "fold": int(folds[row_id]),
                "continuous_score": float(row["localization_score"]),
                "locator": int(row["localization_step"]),
                "source_artifact_hash": source_hash,
            }
    missing = [row_id for row_id in population.ordered_ids if row_id not in by_id]
    if missing or len(by_id) != len(population.ordered_ids):
        raise ValueError(f"Unified-28 localization join incomplete: missing={missing[:5]}")
    return {
        "method_id": UNIFIED28_METHOD_ID,
        "source_artifact_hash": source_hash,
        "rows": [by_id[row_id] for row_id in population.ordered_ids],
    }


def crossfit_score_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    population_id: str,
) -> dict[str, Any]:
    """Cross-fit only a threshold around one frozen score+locator method."""

    threshold_rows = [
        {
            **dict(row),
            "step_scores": [float(row["continuous_score"])],
            "step_indices": [int(row["locator"])],
        }
        for row in rows
    ]
    fitted = crossfit_localization_threshold(
        threshold_rows,
        expected_subsets=PROCESSBENCH_SUBSETS,
    )
    ledger_by_fold = {
        int(ledger["held_out_fold"]): ledger for ledger in fitted["calibration_ledgers"]
    }
    records = []
    for row, prediction in zip(rows, fitted["predictions"]):
        ledger = ledger_by_fold[int(row["fold"])]
        records.append(
            make_comparison_record(
                lane="localization",
                population_id=population_id,
                row_id=str(row["row_id"]),
                group_id=str(row["group_id"]),
                cell_id=str(row["cell_id"]),
                method_id=method_id,
                continuous_score=float(row["continuous_score"]),
                discrete_prediction=int(prediction),
                label=int(row["first_error"]),
                budget="final",
                fold=int(row["fold"]),
                calibration_hash=str(ledger["calibration_hash"]),
                source_artifact_hash=str(row["source_artifact_hash"]),
                extra={
                    "family": str(row["family"]),
                    "stratify_label": int(row["stratify_label"]),
                    "locator": int(row["locator"]),
                    "prediction_status": "parsed",
                },
            )
        )
    return {
        "method_id": method_id,
        "records": records,
        "calibration_ledgers": fitted["calibration_ledgers"],
        "metrics": fitted["official_oof_metrics"],
    }


def evaluate_fixed_prediction_method(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_order: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a no-fit PRM/critic/judge prediction vector without dropping nulls."""

    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        row_id = str(record["row_id"])
        if row_id in by_id:
            raise ValueError(f"duplicate fixed prediction: {row_id}")
        by_id[row_id] = record
    if set(by_id) != set(expected_order):
        raise ValueError("fixed prediction method does not cover the exact ProcessBench IDs")
    rows = [by_id[row_id] for row_id in expected_order]
    predictions = [row.get("discrete_prediction") for row in rows]
    metrics = localization_metrics(
        [
            {"subset": row["family"], "first_error": int(row["label"])}
            for row in rows
        ],
        predictions,
        expected_subsets=PROCESSBENCH_SUBSETS,
    )
    return {
        "method_id": str(rows[0]["method_id"]),
        "metrics": metrics,
        "parser_coverage": sum(value is not None for value in predictions) / len(predictions),
    }


def native_mind_gap_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Erroneous-trace-only SLA context, kept separate from ProcessBench F1."""

    error_rows = [row for row in rows if int(row["first_error"]) != -1]
    predictions = [int(row["locator"]) for row in error_rows]
    labels = [int(row["first_error"]) for row in error_rows]
    return {
        "population": "erroneous_traces_only",
        "n": len(error_rows),
        "native_sla": mind_the_gap_sla(predictions, labels, tolerance=0),
        "tolerance_one_sla": mind_the_gap_sla(predictions, labels, tolerance=1),
    }


def bootstrap_localization_contrasts(
    methods: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    contrasts: Sequence[tuple[str, str]],
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired ProcessBench macro-F1 intervals with threshold refits per replicate."""

    method_ids = list(methods)
    indexes: dict[str, dict[str, Mapping[str, Any]]] = {}
    common_ids: set[str] | None = None
    for method_id, rows in methods.items():
        index = {str(row["row_id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"{method_id}: duplicate localization bootstrap row")
        indexes[method_id] = index
        common_ids = set(index) if common_ids is None else common_ids.intersection(index)
    if common_ids is None or any(set(index) != common_ids for index in indexes.values()):
        raise ValueError("localization bootstrap requires identical row IDs for every method")
    groups: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for row_id in sorted(common_ids):
        reference = indexes[method_ids[0]][row_id]
        payload = {
            "row_id": row_id,
            "family": str(reference["family"]),
            "fold": int(reference["fold"]),
            "first_error": int(reference["first_error"]),
            "methods": {},
        }
        for method_id in method_ids:
            row = indexes[method_id][row_id]
            if int(row["first_error"]) != payload["first_error"]:
                raise ValueError(f"localization label disagreement at {row_id}")
            if row.get("continuous_score") is None:
                payload["methods"][method_id] = {
                    "fixed_prediction": row.get("discrete_prediction")
                }
            else:
                payload["methods"][method_id] = {
                    "score": float(row["continuous_score"]),
                    "locator": int(row["locator"]),
                }
        groups[str(reference["group_id"])] = payload
        strata[str(reference["group_id"])] = payload["family"]

    continuous = {
        method_id
        for method_id in method_ids
        if all("score" in payload["methods"][method_id] for payload in groups.values())
    }

    def recompute(sample: list[dict[str, Any]]) -> dict[str, list[int | None]]:
        predictions: dict[str, list[int | None]] = {}
        for method_id in method_ids:
            if method_id not in continuous:
                predictions[method_id] = [
                    row["methods"][method_id]["fixed_prediction"] for row in sample
                ]
                continue
            fit_rows = [
                {
                    "subset": row["family"],
                    "first_error": row["first_error"],
                    "fold": row["fold"],
                    "step_scores": [row["methods"][method_id]["score"]],
                    "step_indices": [row["methods"][method_id]["locator"]],
                }
                for row in sample
            ]
            predictions[method_id] = crossfit_localization_threshold(
                fit_rows, expected_subsets=PROCESSBENCH_SUBSETS
            )["predictions"]
        return predictions

    def statistic(
        sample: list[dict[str, Any]],
        predictions: Mapping[str, Sequence[int | None]],
    ) -> Mapping[str, float]:
        metric_rows = [
            {"subset": row["family"], "first_error": row["first_error"]}
            for row in sample
        ]
        values = {
            method_id: float(
                localization_metrics(
                    metric_rows,
                    predictions[method_id],
                    expected_subsets=PROCESSBENCH_SUBSETS,
                )["equal_subset_macro_f1"]
            )
            for method_id in method_ids
        }
        output = {f"{method_id}__macro_f1": value for method_id, value in values.items()}
        for left, right in contrasts:
            output[f"delta__{left}__minus__{right}"] = values[left] - values[right]
        return output

    result = paired_grouped_bootstrap(
        groups,
        statistic,
        strata=strata,
        recompute=recompute,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "schema": "fair_localization_paired_intervals_v1",
        "lane_revision": LOCALIZATION_LANE_REVISION,
        "predeclared_contrasts": [
            {"left": left, "right": right} for left, right in contrasts
        ],
        **result,
    }


__all__ = [
    "DEDICATED_METHOD_ID",
    "GL_LIU_METHOD_ID",
    "LOCALIZATION_LANE_REVISION",
    "MAX_ENTROPY_METHOD_ID",
    "MIND_GAP_METHOD_ID",
    "SAME_ACCESS_METHODS",
    "UNIFIED28_METHOD_ID",
    "bootstrap_localization_contrasts",
    "crossfit_score_method",
    "evaluate_fixed_prediction_method",
    "load_unified28_prethreshold_rows",
    "native_mind_gap_metrics",
    "replay_same_access_methods",
]
