"""CPU-only evaluation for the Global final-answer detection lane.

Evaluation scores enter this module already frozen and joined by canonical row ID.
The one replay adapter below reconstructs a registered label-free DUFS head only
from its original fit IDs and constants; it exposes no selection surface.  The
only evaluation-fitted quantities are the preregistered 5%/10% operating
thresholds, learned on four folds of correct rows and applied once to the held-out
fold.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    auroc,
    average_precision,
    calibrate_correct_only_threshold,
    detection_metrics,
    operating_point,
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
from ..dufs_liu_feature_contract import CONTRACT_VERSION
from ..historical_multitask_baselines import (
    REGISTERED_DUFS_EPOCHS,
    REGISTERED_DUFS_K,
    REGISTERED_DUFS_LAMBDA,
    REGISTERED_DUFS_SEEDS,
    fit_registered_dufs_global,
)


GLOBAL_LANE_REVISION = "fair_global_lane_v1.0.0"
OPERATING_TARGETS = (0.05, 0.10)
MIXED_V2_DUFS_NO_LENGTH_METHOD_ID = "mixed_v2_dufs_liu_l0p1_no_length"
REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256 = (
    "c75d27be8492278ced261f93c3d809ca16c5e95103ef6288be3212ad2c659be5"
)
# ``scipy.sparse.linalg.eigsh`` may vary the last few bits of its diagnostic
# eigenvalue across otherwise byte-identical CPU replays.  This value is not read
# by scoring or included in the frozen model-state fingerprint.  Round only its
# audit representation to 12 decimal places, still far finer than the precision
# used in reports; all fitted arrays and scientific metrics retain their original
# float64 values.  The emitted diagnostics carry this serialization contract.
DUFS_REPLAY_CONNECTIVITY_DECIMALS = 12


def _float_array_fingerprint(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype="<f8")
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _canonical_dufs_replay_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return stable audit-only DUFS diagnostics without mutating model state."""

    serialized = dict(diagnostics)
    laplacian = serialized.get("laplacian")
    if (
        not isinstance(laplacian, Mapping)
        or "algebraic_connectivity" not in laplacian
    ):
        raise ValueError("DUFS replay diagnostics lack algebraic connectivity")
    connectivity = float(laplacian["algebraic_connectivity"])
    if not np.isfinite(connectivity):
        raise ValueError("DUFS replay algebraic connectivity is non-finite")
    canonical_connectivity = float(
        round(connectivity, DUFS_REPLAY_CONNECTIVITY_DECIMALS)
    )
    # Avoid a platform-dependent signed-zero spelling in canonical JSON.
    if canonical_connectivity == 0.0:
        canonical_connectivity = 0.0
    serialized["laplacian"] = {
        **dict(laplacian),
        "algebraic_connectivity": canonical_connectivity,
        "algebraic_connectivity_serialization": {
            "mode": "round_decimal_places",
            "decimal_places": DUFS_REPLAY_CONNECTIVITY_DECIMALS,
            "scope": "audit-only",
        },
    }
    return serialized


def verify_registered_dufs_provenance(
    *,
    anchor_manifest_path: str | Path,
    classic_run_definition_path: str | Path,
    qwen_source_hashes: Mapping[str, str],
    llama_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Fail closed unless the replay inputs match both frozen source ledgers."""

    anchor_path = Path(anchor_manifest_path)
    anchor_manifest = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor_cells = [
        cell
        for cell in anchor_manifest.get("cells", [])
        if cell.get("model") == "qwen3_8b" and cell.get("subset") == "gsm8k"
    ]
    if len(anchor_cells) != 1:
        raise ValueError("registered DUFS manifest must contain one Qwen3-8B/GSM8K cell")
    registered_anchor = anchor_cells[0].get("score_hashes", {}).get(
        "global_mixed_v2_dufs"
    )
    if registered_anchor != REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256:
        raise ValueError("registered DUFS manifest anchor hash drifted")
    if anchor_manifest.get("labels_or_step_spans_read") is not False:
        raise ValueError("registered DUFS manifest does not prove label-free score fitting")
    if anchor_manifest.get("global_detector") != "mixed-v2 DUFS-LIU, lambda=0.1, k=7":
        raise ValueError("registered DUFS manifest detector contract drifted")

    definition_path = Path(classic_run_definition_path)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("classic_contract") != (
        "registered mixed-v2 30-feature contract with final length excluded"
    ):
        raise ValueError("classic Global no-length contract drifted")
    if definition.get("classic_labels_seen_during_fit") is not False:
        raise ValueError("classic Global definition does not prove label-free IU fitting")

    def registered_hashes(key: str, model: str) -> dict[str, str]:
        rows = definition.get(key, [])
        output: dict[str, str] = {}
        for row in rows:
            if row.get("model") != model:
                continue
            family = str(row.get("family"))
            if family in output:
                raise ValueError(f"duplicate {model} source ledger for {family}")
            output[family] = str(row.get("sha256"))
        return output

    expected_qwen = registered_hashes("inventory", "qwen3_8b")
    expected_llama = registered_hashes("validation_inventory", "llama31_8b")
    observed_qwen = {str(key): str(value) for key, value in qwen_source_hashes.items()}
    observed_llama = {str(key): str(value) for key, value in llama_source_hashes.items()}
    if expected_qwen != observed_qwen:
        raise ValueError("Qwen replay telemetry hashes differ from classic Global definition")
    if expected_llama != observed_llama:
        raise ValueError("Llama replay telemetry hashes differ from classic Global definition")
    audit = {
        "schema": "registered_mixed_v2_dufs_provenance_audit_v1",
        # Content identity is portable; resolved host paths are intentionally absent
        # from the hashed projection and remain an execution detail in the builder.
        "anchor_manifest_name": anchor_path.name,
        "anchor_manifest_sha256": sha256_file(anchor_path),
        "classic_run_definition_name": definition_path.name,
        "classic_run_definition_sha256": sha256_file(definition_path),
        "registered_anchor_sha256": registered_anchor,
        "qwen_source_hashes": expected_qwen,
        "llama_source_hashes": expected_llama,
        "labels_or_step_spans_read_for_registered_anchor": False,
        "classic_labels_seen_during_fit": False,
        "passed": True,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def load_classic_global_fit_ids(path: str | Path) -> dict[str, Any]:
    """Recover only the exact Qwen fit identities used by the classic incumbent.

    The source artifact also contains labels and old scores.  This adapter deliberately
    never reads those fields: only method/family/model/unit/source-group identity and the
    repeated-split coordinates are admitted.
    """

    source = Path(path)
    by_family: dict[str, dict[str, set[tuple[int, int]]]] = {
        family: {} for family in PROCESSBENCH_SUBSETS
    }
    observed_rows = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            observed_rows += 1
            required = ("candidate", "family", "model", "unit", "source_group", "repeat", "fold")
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(
                    f"classic Global fit-ID row {line_number} misses {missing}"
                )
            if row["candidate"] != "classic_mixed_v2_no_length":
                raise ValueError(f"unexpected classic fit-ID candidate at line {line_number}")
            if row["model"] != "qwen3_8b":
                raise ValueError(f"unexpected classic fit-ID model at line {line_number}")
            family = str(row["family"])
            if family not in by_family:
                raise ValueError(f"unexpected classic fit-ID family {family!r}")
            unit = str(row["unit"])
            group = str(row["source_group"])
            if group != f"{family}::{unit}":
                raise ValueError(f"classic fit-ID group/unit disagreement at line {line_number}")
            split = (int(row["repeat"]), int(row["fold"]))
            if split in by_family[family].setdefault(unit, set()):
                raise ValueError(f"duplicate classic fit-ID split at line {line_number}")
            by_family[family][unit].add(split)

    if observed_rows != 384:
        raise ValueError(f"classic fit-ID artifact must contain 384 rows, got {observed_rows}")
    ordered: dict[str, list[str]] = {}
    family_ledgers = []
    for family in PROCESSBENCH_SUBSETS:
        units = sorted(by_family[family])
        if len(units) != 32:
            raise ValueError(f"{family}: expected 32 classic Qwen fit IDs, got {len(units)}")
        for unit, splits in by_family[family].items():
            if {repeat for repeat, _ in splits} != {0, 1, 2} or len(splits) != 3:
                raise ValueError(f"{family}/{unit}: incomplete repeated classic fit identity")
        ordered[family] = units
        family_ledgers.append({"family": family, "ordered_ids": units})
    return {
        "schema": "classic_global_fit_ids_v1",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "observed_rows": observed_rows,
        "fit_ids_by_family": ordered,
        "fit_id_sha256": canonical_sha256(family_ledgers),
        "labels_or_scores_read": False,
    }


def _ordered_telemetry_rows(
    source: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
    *,
    historical_numeric_order: bool,
) -> list[dict[str, Any]]:
    """Materialize telemetry using one explicitly registered historical ordering."""

    if isinstance(source, Mapping):
        keys = (
            sorted(source)
            if historical_numeric_order
            else sorted(source, key=lambda value: str(value))
        )
        candidates = [source[key] for key in keys]
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        candidates = list(source)
    else:
        raise TypeError("ProcessBench telemetry must be a mapping or row sequence")
    output, seen = [], set()
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise TypeError(f"telemetry row {index} is not a mapping")
        problems = row.get("align_diag", {}).get("problems")
        if problems:
            raise ValueError(f"telemetry row {index} has alignment problems: {problems}")
        official_id = row.get("id")
        if not isinstance(official_id, str) or not official_id:
            raise ValueError(f"telemetry row {index} lacks an official string ID")
        if official_id in seen:
            raise ValueError(f"duplicate telemetry official ID: {official_id}")
        seen.add(official_id)
        output.append(dict(row))
    return output


def audit_registered_dufs_anchor(
    qwen_gsm8k_source: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reproduce the pre-label registered Qwen3-8B/GSM8K DUFS score hash."""

    rows = _ordered_telemetry_rows(
        qwen_gsm8k_source,
        historical_numeric_order=True,
    )
    model = fit_registered_dufs_global(rows, exclude_trace_length=False)
    observed = _float_array_fingerprint(model.training_scores)["sha256"]
    result = {
        "schema": "registered_mixed_v2_dufs_anchor_audit_v1",
        "anchor": "qwen3_8b/gsm8k/full-cell/global_mixed_v2_dufs",
        "n_rows": len(rows),
        "trace_length_excluded": False,
        "expected_score_sha256": REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256,
        "observed_score_sha256": observed,
        "passed": observed == REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256,
        "labels_read": False,
    }
    if not result["passed"]:
        raise ValueError(
            "registered mixed-v2 DUFS anchor mismatch: "
            f"expected={REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256}, observed={observed}"
        )
    return result


def replay_registered_dufs_no_length(
    *,
    population: ProcessBenchPopulation,
    population_id: str,
    qwen_telemetry: Mapping[str, Mapping[Any, Any] | Sequence[Mapping[str, Any]]],
    llama_telemetry: Mapping[str, Mapping[Any, Any] | Sequence[Mapping[str, Any]]],
    fit_ids_path: str | Path,
    folds: Mapping[str, int],
    qwen_source_hashes: Mapping[str, str],
    llama_source_hashes: Mapping[str, str],
    anchor_manifest_path: str | Path,
    classic_run_definition_path: str | Path,
) -> dict[str, Any]:
    """Replay frozen mixed-v2 DUFS-LIU from exact Qwen IDs onto all Llama IDs."""

    provenance = verify_registered_dufs_provenance(
        anchor_manifest_path=anchor_manifest_path,
        classic_run_definition_path=classic_run_definition_path,
        qwen_source_hashes=qwen_source_hashes,
        llama_source_hashes=llama_source_hashes,
    )
    fit_ids = load_classic_global_fit_ids(fit_ids_path)
    anchor = audit_registered_dufs_anchor(qwen_telemetry["gsm8k"])
    frozen_scores: dict[str, tuple[float, str]] = {}
    fit_ledgers = []
    for family in PROCESSBENCH_SUBSETS:
        qwen_rows = _ordered_telemetry_rows(
            qwen_telemetry[family], historical_numeric_order=False
        )
        qwen_by_id = {str(row["id"]): row for row in qwen_rows}
        family_fit_ids = list(fit_ids["fit_ids_by_family"][family])
        missing_fit = [unit for unit in family_fit_ids if unit not in qwen_by_id]
        if missing_fit:
            raise ValueError(f"{family}: missing classic Qwen fit IDs {missing_fit[:5]}")
        fit_set = set(family_fit_ids)
        # Preserve the classic runner's lexical cache-key order, rather than sorting
        # the extracted identity ledger itself.
        fit_rows = [row for row in qwen_rows if str(row["id"]) in fit_set]
        if len(fit_rows) != 32 or {str(row["id"]) for row in fit_rows} != fit_set:
            raise ValueError(f"{family}: classic Qwen fit-row materialization drifted")
        model = fit_registered_dufs_global(fit_rows, exclude_trace_length=True)
        if "trace_length" in model.names or not model.diagnostics["trace_length_excluded"]:
            raise RuntimeError(f"{family}: no-length DUFS contract was not enforced")

        llama_rows = _ordered_telemetry_rows(
            llama_telemetry[family], historical_numeric_order=False
        )
        llama_by_id = {str(row["id"]): row for row in llama_rows}
        expected_family_ids = [
            population.rows[row_id].official_id
            for row_id in population.ids_for_subset(family)
        ]
        if set(llama_by_id) != set(expected_family_ids):
            raise ValueError(f"{family}: Llama DUFS replay IDs differ from population")
        transformer = model.transformer
        model_state = {
            "schema": "registered_mixed_v2_dufs_model_state_v1",
            "feature_names": list(model.names),
            "weights": _float_array_fingerprint(model.weights),
            "training_scores": _float_array_fingerprint(model.training_scores),
            "transformer": {
                "raw_median": _float_array_fingerprint(transformer.raw_median),
                "oriented_mean": _float_array_fingerprint(transformer.oriented_mean),
                "oriented_std": _float_array_fingerprint(transformer.oriented_std),
                "sorted_oriented": [
                    _float_array_fingerprint(values)
                    for values in transformer.sorted_oriented
                ],
                "mode_centres": _float_array_fingerprint(transformer.mode_centres),
                "output_mean": _float_array_fingerprint(transformer.output_mean),
                "output_std": _float_array_fingerprint(transformer.output_std),
                "training_output": _float_array_fingerprint(
                    transformer.training_output
                ),
            },
        }
        model_state_sha256 = canonical_sha256(model_state)
        fit_row_order = [str(row["id"]) for row in fit_rows]
        dependency_ledger = {
            "schema": "mixed_v2_dufs_no_length_family_dependency_v1",
            "family": family,
            "fit_ids": family_fit_ids,
            "fit_row_order": fit_row_order,
            "fit_row_order_sha256": canonical_sha256(fit_row_order),
            "fit_ids_source_sha256": fit_ids["source_sha256"],
            "qwen_source_sha256": qwen_source_hashes[family],
            "llama_source_sha256": llama_source_hashes[family],
            "feature_contract": CONTRACT_VERSION,
            "trace_length_excluded": True,
            "dufs_seeds": list(REGISTERED_DUFS_SEEDS),
            "dufs_epochs": REGISTERED_DUFS_EPOCHS,
            "graph_k": REGISTERED_DUFS_K,
            "lambda": REGISTERED_DUFS_LAMBDA,
            "registered_anchor_sha256": anchor["observed_score_sha256"],
            "provenance_audit_sha256": provenance["audit_sha256"],
            "model_state_sha256": model_state_sha256,
        }
        source_fingerprint = canonical_sha256(dependency_ledger)
        for official_id in expected_family_ids:
            row_id = canonical_processbench_id(family, official_id)
            frozen_scores[row_id] = (
                float(model.score(llama_by_id[official_id])),
                source_fingerprint,
            )
        fit_ledgers.append(
            {
                **dependency_ledger,
                "fit_id_sha256": canonical_sha256(family_fit_ids),
                "model_state": model_state,
                "source_fingerprint": source_fingerprint,
                "model_diagnostics": _canonical_dufs_replay_diagnostics(
                    model.diagnostics
                ),
                "labels_read_during_score_construction": False,
            }
        )

    if set(frozen_scores) != set(population.ordered_ids):
        raise ValueError("no-length mixed-v2 DUFS replay did not cover all population IDs")
    records = []
    for row_id in population.ordered_ids:
        pop = population.rows[row_id]
        score, source_fingerprint = frozen_scores[row_id]
        records.append(
            make_comparison_record(
                lane="global",
                population_id=population_id,
                row_id=row_id,
                group_id=pop.group_id,
                cell_id=pop.cell_id,
                method_id=MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
                continuous_score=score,
                discrete_prediction=None,
                label=int(pop.wrong_label),
                budget="final",
                fold=int(folds[row_id]),
                calibration_hash=None,
                source_artifact_hash=source_fingerprint,
                extra={
                    "family": pop.subset,
                    "stratify_label": int(pop.wrong_label),
                    "prediction_status": "not_applicable",
                },
            )
        )
    return {
        "schema": "registered_mixed_v2_dufs_no_length_replay_v1",
        "method_id": MIXED_V2_DUFS_NO_LENGTH_METHOD_ID,
        "records": records,
        "fit_ids": fit_ids,
        "fit_ledgers": fit_ledgers,
        "anchor_audit": anchor,
        "provenance_audit": provenance,
        "coverage": len(records) / len(population.ordered_ids),
        "ordered_id_sha256": canonical_sha256(list(population.ordered_ids)),
        "labels_read_during_score_construction": False,
    }


def _ordered_method_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    method_id: str,
    ordered_ids: Sequence[str],
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in records
        if row["method_id"] == method_id and row.get("budget") == "final"
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for row in selected:
        row_id = str(row["row_id"])
        if row_id in by_id:
            raise ValueError(f"duplicate Global record for {method_id}/{row_id}")
        by_id[row_id] = row
    expected = list(ordered_ids)
    missing = [row_id for row_id in expected if row_id not in by_id]
    extra = sorted(set(by_id).difference(expected))
    if missing or extra:
        raise ValueError(
            f"Global identical-row gate failed for {method_id}: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    output = [by_id[row_id] for row_id in expected]
    for row in output:
        if row.get("continuous_score") is None:
            raise ValueError(f"Global score is null for {method_id}/{row['row_id']}")
        if int(row["label"]) not in (0, 1):
            raise ValueError("Global label must use 1=final-answer wrong")
        if row.get("fold") not in range(5):
            raise ValueError("Global cross-fitting requires frozen folds 0..4")
    return output


def crossfit_operating_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[float] = OPERATING_TARGETS,
) -> dict[str, Any]:
    """Fit correct-only operating thresholds on four folds and score the fifth."""

    rows = [dict(row) for row in rows]
    observed_folds = {int(row["fold"]) for row in rows}
    if observed_folds != set(range(5)):
        raise ValueError(f"Global folds must be exactly 0..4, got {sorted(observed_folds)}")
    outputs: dict[str, Any] = {}
    for target in targets:
        key = f"fpr_{int(round(100.0 * float(target))):02d}"
        predictions = np.zeros(len(rows), dtype=bool)
        thresholds = np.full(len(rows), np.nan, dtype=float)
        ledger = []
        for held_out in range(5):
            train = [row for row in rows if int(row["fold"]) != held_out]
            test_indices = [index for index, row in enumerate(rows) if int(row["fold"]) == held_out]
            fitted = calibrate_correct_only_threshold(
                [row["label"] for row in train],
                [row["continuous_score"] for row in train],
                target_fpr=float(target),
            )
            threshold = float(fitted["threshold"])
            for index in test_indices:
                thresholds[index] = threshold
                predictions[index] = float(rows[index]["continuous_score"]) >= threshold
            fold_ledger = {
                **fitted,
                "held_out_fold": held_out,
                "train_folds": [fold for fold in range(5) if fold != held_out],
                "n_held_out_rows": len(test_indices),
            }
            fold_ledger["calibration_hash"] = canonical_sha256(fold_ledger)
            ledger.append(fold_ledger)
        if not np.all(np.isfinite(thresholds)):
            raise RuntimeError("a Global held-out row did not receive a threshold")
        labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        tn = int(np.sum(~predictions & (labels == 0)))
        per_cell = []
        for cell_id in sorted({str(row["cell_id"]) for row in rows}):
            indices = np.asarray(
                [index for index, row in enumerate(rows) if str(row["cell_id"]) == cell_id],
                dtype=int,
            )
            cell_predictions = predictions[indices]
            cell_labels = labels[indices]
            cell_tp = int(np.sum(cell_predictions & (cell_labels == 1)))
            cell_fp = int(np.sum(cell_predictions & (cell_labels == 0)))
            cell_fn = int(np.sum(~cell_predictions & (cell_labels == 1)))
            cell_tn = int(np.sum(~cell_predictions & (cell_labels == 0)))
            per_cell.append(
                {
                    "cell_id": cell_id,
                    "family": str(rows[int(indices[0])].get("family", cell_id)),
                    "n": len(indices),
                    "error_tpr": cell_tp / (cell_tp + cell_fn)
                    if cell_tp + cell_fn
                    else float("nan"),
                    "error_precision": cell_tp / (cell_tp + cell_fp)
                    if cell_tp + cell_fp
                    else float("nan"),
                    "observed_fpr": cell_fp / (cell_fp + cell_tn)
                    if cell_fp + cell_tn
                    else float("nan"),
                    "tp": cell_tp,
                    "fp": cell_fp,
                    "fn": cell_fn,
                    "tn": cell_tn,
                }
            )
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for cell in per_cell:
            by_family[str(cell["family"])].append(cell)
        per_family = [
            {
                "family": family,
                "error_tpr": _macro([row["error_tpr"] for row in family_rows]),
                "error_precision": _macro(
                    [row["error_precision"] for row in family_rows]
                ),
                "observed_fpr": _macro([row["observed_fpr"] for row in family_rows]),
            }
            for family, family_rows in sorted(by_family.items())
        ]
        outputs[key] = {
            "target_fpr": float(target),
            "error_tpr": tp / (tp + fn) if tp + fn else float("nan"),
            "error_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "observed_fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "calibration_ledgers": ledger,
            "calibration_hash": canonical_sha256(ledger),
            "aggregation": "concatenated_discrete_out_of_fold_predictions",
            "per_cell": per_cell,
            "equal_cell": {
                "error_tpr": _macro([row["error_tpr"] for row in per_cell]),
                "error_precision": _macro([row["error_precision"] for row in per_cell]),
                "observed_fpr": _macro([row["observed_fpr"] for row in per_cell]),
            },
            "per_family": per_family,
            "equal_family": {
                "error_tpr": _macro([row["error_tpr"] for row in per_family]),
                "error_precision": _macro(
                    [row["error_precision"] for row in per_family]
                ),
                "observed_fpr": _macro([row["observed_fpr"] for row in per_family]),
            },
        }
    return outputs


def _cell_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell_id"])].append(row)
    output = []
    for cell_id, cell_rows in sorted(by_cell.items()):
        metric = detection_metrics(
            [row["label"] for row in cell_rows],
            [row["continuous_score"] for row in cell_rows],
        )
        metric.update(
            {
                "cell_id": cell_id,
                "family": str(cell_rows[0].get("family", cell_id)),
            }
        )
        output.append(metric)
    return output


def _macro(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_global_panel(
    records: Iterable[Mapping[str, Any]],
    *,
    ordered_ids: Sequence[str],
    method_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a strict identical-row Global table with equal-cell/family macros."""

    records = list(records)
    results = []
    ordered_by_method: dict[str, list[dict[str, Any]]] = {}
    for method_id in method_ids:
        rows = _ordered_method_rows(records, method_id=method_id, ordered_ids=ordered_ids)
        ordered_by_method[method_id] = rows
        cells = _cell_metrics(rows)
        family_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            family_cells[cell["family"]].append(cell)
        families = []
        for family, values in sorted(family_cells.items()):
            families.append(
                {
                    "family": family,
                    "n_cells": len(values),
                    "auroc": _macro([row["auroc"] for row in values]),
                    "error_auprc": _macro([row["error_auprc"] for row in values]),
                }
            )
        results.append(
            {
                "method_id": method_id,
                "n": len(rows),
                "cell_metrics": cells,
                "family_metrics": families,
                "equal_cell_auroc": _macro([row["auroc"] for row in cells]),
                "equal_cell_error_auprc": _macro([row["error_auprc"] for row in cells]),
                "equal_family_auroc": _macro([row["auroc"] for row in families]),
                "equal_family_error_auprc": _macro(
                    [row["error_auprc"] for row in families]
                ),
                "operating_points": crossfit_operating_points(rows),
            }
        )
    labels_by_id: dict[str, int] = {}
    for method, rows in ordered_by_method.items():
        for row in rows:
            row_id = str(row["row_id"])
            label = int(row["label"])
            previous = labels_by_id.setdefault(row_id, label)
            if previous != label:
                raise ValueError(f"Global label conflict at {row_id} for {method}")
    return {
        "schema": "fair_global_panel_metrics_v1",
        "lane_revision": GLOBAL_LANE_REVISION,
        "positive_class": "final_answer_wrong",
        "ordered_id_sha256": canonical_sha256(list(ordered_ids)),
        "methods": results,
    }


def bootstrap_global_contrasts(
    records: Iterable[Mapping[str, Any]],
    *,
    ordered_ids: Sequence[str],
    method_ids: Sequence[str],
    contrasts: Sequence[tuple[str, str]],
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired family-grouped intervals for predeclared Global AUROC contrasts."""

    ordered = {
        method: _ordered_method_rows(records, method_id=method, ordered_ids=ordered_ids)
        for method in method_ids
    }
    groups: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for index, row_id in enumerate(ordered_ids):
        reference = ordered[method_ids[0]][index]
        family = str(reference.get("family", reference["cell_id"]))
        payload = {
            "row_id": row_id,
            "family": family,
            "cell_id": str(reference["cell_id"]),
            "fold": int(reference["fold"]),
            "label": int(reference["label"]),
            "scores": {
                method: float(ordered[method][index]["continuous_score"])
                for method in method_ids
            },
        }
        groups[str(reference["group_id"])] = payload
        strata[str(reference["group_id"])] = family

    def recompute(sample: list[dict[str, Any]]) -> Mapping[str, Any]:
        return {
            method: crossfit_operating_points(
                [
                    {
                        "row_id": row["row_id"],
                        "group_id": row["row_id"],
                        "family": row["family"],
                        "cell_id": row["cell_id"],
                        "fold": row["fold"],
                        "label": row["label"],
                        "continuous_score": row["scores"][method],
                    }
                    for row in sample
                ]
            )
            for method in method_ids
        }

    def statistic(
        sample: list[dict[str, Any]],
        fitted: Mapping[str, Any],
    ) -> Mapping[str, float]:
        families = sorted({row["family"] for row in sample})
        method_macro = {}
        for method in method_ids:
            values = []
            for family in families:
                family_rows = [row for row in sample if row["family"] == family]
                values.append(
                    auroc(
                        [row["label"] for row in family_rows],
                        [row["scores"][method] for row in family_rows],
                    )
                )
            method_macro[method] = _macro(values)
        output: dict[str, float] = {
            f"{method}__equal_family_auroc": value
            for method, value in method_macro.items()
        }
        for left, right in contrasts:
            output[f"delta__{left}__minus__{right}"] = method_macro[left] - method_macro[right]
            for target_key in ("fpr_05", "fpr_10"):
                left_op = fitted[left][target_key]["equal_family"]
                right_op = fitted[right][target_key]["equal_family"]
                output[
                    f"delta_tpr_{target_key}__{left}__minus__{right}"
                ] = float(left_op["error_tpr"]) - float(right_op["error_tpr"])
        for method in method_ids:
            for target_key in ("fpr_05", "fpr_10"):
                op = fitted[method][target_key]["equal_family"]
                output[f"{method}__tpr_{target_key}"] = float(op["error_tpr"])
                output[f"{method}__precision_{target_key}"] = float(
                    op["error_precision"]
                )
                output[f"{method}__observed_{target_key}"] = float(op["observed_fpr"])
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
        "schema": "fair_global_paired_intervals_v1",
        "lane_revision": GLOBAL_LANE_REVISION,
        "predeclared_contrasts": [
            {"left": left, "right": right} for left, right in contrasts
        ],
        **result,
    }


__all__ = [
    "GLOBAL_LANE_REVISION",
    "MIXED_V2_DUFS_NO_LENGTH_METHOD_ID",
    "OPERATING_TARGETS",
    "REGISTERED_QWEN8_GSM8K_DUFS_ANCHOR_SHA256",
    "audit_registered_dufs_anchor",
    "bootstrap_global_contrasts",
    "crossfit_operating_points",
    "evaluate_global_panel",
    "load_classic_global_fit_ids",
    "replay_registered_dufs_no_length",
    "verify_registered_dufs_provenance",
]
