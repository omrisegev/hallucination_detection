"""Strict CPU-only assembly for the Fair Comparison v1 prefix lane.

The prefix lane is intentionally a *join and evaluation* layer.  It does not fit a
new detector or choose a feature roster.  The loaders below consume the frozen
Unified-28, historical Online, and selected Step-272 score artifacts and retain only
identities carried by the artifacts themselves.  The one reconstruction path rebuilds
the already frozen ``iu28_no_length`` parameters from their original hashed fit cache;
it is enabled only after exact identity against every stored score anchor.

Two population boundaries are important:

* ProcessBench identities are always
  ``processbench@e8024636bcab::<subset>::<official_id>``.  A mapping key or row
  position is never accepted as an ID.
* The old 11-cell Online package and Step-272 used different pre-existing held-out
  splits.  A direct historical table is therefore built on the explicit intersection
  of their registered artifact availability.  Its ordered row IDs and construction
  rule are hashed and reported.  Cells without Unified-28 and the selected Step-272
  incumbent remain coverage-only; missing Unified scores are never fabricated.

All causal metrics enforce ``final_length > budget`` for the fixed grid
``16/32/64/128/256/512``.  Final scores are sidecars used only to measure recovered
above-chance signal on the same unfinished-trace cohort.  Warning calibration is kept
outside this module: :func:`build_warning_inputs` emits complete six-budget paths so
the common evaluator can cross-fit trace-level ever-warning thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .folds import canonical_sha256, ordered_id_sha256
from .processbench import (
    PROCESSBENCH_DATASET_REVISION,
    PROCESSBENCH_PREFIX_BUDGETS,
    canonical_processbench_id,
)
from .registry import (
    make_comparison_record,
    make_eligible_population,
    validate_comparison_record,
)


PREFIX_LANE_REVISION = "fair_prefix_lane_v1.0.0"
PREFIX_BUDGETS = tuple(PROCESSBENCH_PREFIX_BUDGETS)
SELECTED_STEP272_ARCHITECTURE = "a_two_global_local__w0.50__peak"
REGISTERED_LLAMA_PREFIX_TELEMETRY = (
    f"{PROCESSBENCH_DATASET_REVISION}::repgrid::llama31_8b"
)
HISTORICAL_PREFIX_TELEMETRY = "early_online_existing_data_v1::pb_qwen3_4b_rescore"

UNIFIED28_METHOD_ID = "unified28"
IU28_METHOD_ID = "iu28_no_length"
STEP272_METHOD_ID = "step272_two_head_global_local_w0p50_peak"
MEAN_ENTROPY_METHOD_ID = "mean_entropy"
MAX_ENTROPY_METHOD_ID = "max_entropy"
HISTORICAL_DEEPCONF_METHOD_ID = "historical_deepconf_entropy_w64_proxy"
CLASSIC_GLOBAL_METHOD_ID = "classic_mixed_v2_no_length"
FROZEN_PREFIX_REPLAY_REVISION = "frozen_prefix_incumbent_replay_v1.0.0"

DIRECT_REQUIRED_METHODS = (
    UNIFIED28_METHOD_ID,
    STEP272_METHOD_ID,
)

# These four stored fields are sufficient to derive all seven frozen Unified-28
# streams.  ``top_k_logprobs`` supplies neg-top1, top-k entropy, varentropy, Renyi-2,
# and tail mass; no derived score is accepted as a substitute for a missing stream.
UNIFIED28_REQUIRED_TELEMETRY_FIELDS = (
    "gen_token_ids",
    "token_entropies",
    "token_logsumexp",
    "top_k_logprobs",
)

# S2's paper driver records raw token channels under ``channels`` and the retained
# top-k matrix under ``raw_top_k_logprobs``.  These aliases are schema mappings only;
# they do not establish frozen-model compatibility or authorize scoring.
S2_UNIFIED28_TELEMETRY_ALIASES = {
    "gen_token_ids": ("gen_token_ids",),
    "token_entropies": ("token_entropies", "channels.raw_entropy"),
    "token_logsumexp": ("token_logsumexp", "channels.raw_logsumexp"),
    "top_k_logprobs": ("top_k_logprobs", "raw_top_k_logprobs"),
}

# The acquired S2 streams use raw-distribution names, while all four historical
# scorers below were frozen on the legacy generation contract.  In that contract
# entropy is top-15-renormalized *after* generation processors/warpers, spilled
# energy is the sampled token's post-warper negative log-probability, and retained
# top-k log-probabilities are also post-warper.  Raw and sampled values are not
# interchangeable, even though both are causal.
S2_FROZEN_INPUT_ALIASES = {
    "gen_token_ids": ("gen_token_ids",),
    "token_entropies": ("token_entropies", "channels.sampled_entropy"),
    "token_spilled_energies": (
        "token_spilled_energies",
        "channels.sampled_spilled_energy",
    ),
    "token_logsumexp": ("token_logsumexp", "channels.raw_logsumexp"),
    "top_k_logprobs": ("top_k_logprobs", "sampled_top_k_logprobs"),
}

S2_FROZEN_INPUT_SEMANTICS = {
    "gen_token_ids": "sampled_generation_token_ids",
    "token_entropies": "postwarper_top15_renormalized_entropy",
    "token_spilled_energies": "postwarper_sampled_token_negative_logprob",
    "token_logsumexp": "raw_full_vocabulary_logsumexp",
    "top_k_logprobs": "postwarper_top50_logprobs",
}

S2_PREFIX_METHOD_INPUTS = {
    UNIFIED28_METHOD_ID: (
        "gen_token_ids",
        "token_entropies",
        "token_logsumexp",
        "top_k_logprobs",
    ),
    IU28_METHOD_ID: tuple(S2_FROZEN_INPUT_ALIASES),
    STEP272_METHOD_ID: tuple(S2_FROZEN_INPUT_ALIASES),
    CLASSIC_GLOBAL_METHOD_ID: tuple(S2_FROZEN_INPUT_ALIASES),
}

_PB_CELL_RE = re.compile(
    r"^processbench_(gsm8k|math|olympiadbench|omnimath)__(.+)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PrefixIntegrityError(ValueError):
    """A prefix artifact, causal budget, or identical-row join is invalid."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PrefixIntegrityError(f"{name} must be a lowercase SHA-256")
    return value


def _registered_population_hashes(population: Any) -> dict[str, str] | None:
    raw = getattr(population, "source_hashes", None)
    if not raw:
        return None
    if not isinstance(raw, Mapping):
        raise PrefixIntegrityError("population source_hashes must be a mapping")
    missing = {"gsm8k", "math", "olympiadbench", "omnimath"}.difference(raw)
    if missing:
        raise PrefixIntegrityError(
            f"registered ProcessBench population hashes missing {sorted(missing)}"
        )
    return {
        subset: _require_hash(str(raw[subset]), name=f"population {subset} hash")
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath")
    }


def _telemetry_bundle_sha256(hashes: Mapping[str, str] | None) -> str | None:
    if hashes is None:
        return None
    return canonical_sha256(dict(sorted(hashes.items())))


def _verify_unified_telemetry_provenance(
    source: Path,
    population: Any,
    *,
    required: bool,
) -> dict[str, Any]:
    registered = _registered_population_hashes(population)
    manifest_path = source.parent / "RUN_DEFINITION.json"
    if registered is None or not manifest_path.exists():
        if required:
            raise PrefixIntegrityError(
                "Unified-28 registered telemetry provenance is unavailable"
            )
        return {
            "verified": False,
            "verification": "unavailable",
            "manifest": str(manifest_path),
            "telemetry_sha256": _telemetry_bundle_sha256(registered),
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixIntegrityError(f"invalid Unified-28 run definition {manifest_path}") from exc
    inventory = manifest.get("validation_inventory")
    if not isinstance(inventory, list):
        raise PrefixIntegrityError("Unified-28 run definition lacks validation_inventory")
    declared: dict[str, str] = {}
    for item in inventory:
        if not isinstance(item, Mapping) or str(item.get("model")) != "llama31_8b":
            continue
        family = str(item.get("family"))
        if family in registered:
            declared[family] = _require_hash(
                str(item.get("sha256")), name=f"Unified-28 {family} telemetry hash"
            )
    if declared != registered:
        raise PrefixIntegrityError(
            "Unified-28 validation telemetry hashes disagree with the registered population"
        )
    return {
        "verified": True,
        "verification": "run_definition_validation_inventory_sha256",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_hashes": declared,
        "telemetry_sha256": _telemetry_bundle_sha256(declared),
    }


def _verify_step272_telemetry_provenance(
    source: Path,
    population: Any,
    *,
    model: str,
    required: bool,
) -> dict[str, Any]:
    registered = _registered_population_hashes(population)
    inventory_path = source.parent / "CACHE_INVENTORY.csv"
    if registered is None or not inventory_path.exists():
        if required:
            raise PrefixIntegrityError("Step-272 registered telemetry provenance is unavailable")
        return {
            "verified": False,
            "verification": "unavailable",
            "inventory": str(inventory_path),
            "telemetry_sha256": _telemetry_bundle_sha256(registered),
        }
    selected: dict[str, Mapping[str, str]] = {}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model and row.get("family") in registered:
                selected[str(row["family"])] = row
    if set(selected) != set(registered):
        raise PrefixIntegrityError(
            f"Step-272 cache inventory lacks registered {model} cells"
        )
    observed: dict[str, str] = {}
    for family, row in selected.items():
        telemetry_path = Path(str(row["path"]))
        if not telemetry_path.exists():
            raise PrefixIntegrityError(
                f"Step-272 registered telemetry path is missing: {telemetry_path}"
            )
        expected_bytes = _strict_int(row.get("bytes"), name="cache bytes", minimum=1)
        if telemetry_path.stat().st_size != expected_bytes:
            raise PrefixIntegrityError(
                f"Step-272 telemetry size drift for {family}: "
                f"{telemetry_path.stat().st_size} != {expected_bytes}"
            )
        observed[family] = _sha256_file(telemetry_path)
    if observed != registered:
        raise PrefixIntegrityError(
            "Step-272 cache inventory payload hashes disagree with the registered population"
        )
    return {
        "verified": True,
        "verification": "cache_inventory_path_size_plus_payload_sha256",
        "inventory": str(inventory_path),
        "inventory_sha256": _sha256_file(inventory_path),
        "source_hashes": observed,
        "telemetry_sha256": _telemetry_bundle_sha256(observed),
    }


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise PrefixIntegrityError(f"{name} must be an integer, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise PrefixIntegrityError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PrefixIntegrityError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, str) and str(integer) != value.strip():
        raise PrefixIntegrityError(f"{name} must be a canonical integer, got {value!r}")
    if minimum is not None and integer < minimum:
        raise PrefixIntegrityError(f"{name} must be >= {minimum}, got {integer}")
    return integer


def _binary(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    integer = _strict_int(value, name=name)
    if integer not in (0, 1):
        raise PrefixIntegrityError(f"{name} must be binary with 1=error, got {value!r}")
    return integer


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise PrefixIntegrityError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PrefixIntegrityError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise PrefixIntegrityError(f"{name} must be finite")
    return number


def _strict_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in ("True", "False"):
        return value == "True"
    raise PrefixIntegrityError(f"{name} must be True or False, got {value!r}")


def _fold_for(
    row_id: str,
    group_id: str,
    folds: Mapping[str, int] | None,
) -> int | None:
    if folds is None:
        return None
    if group_id in folds:
        value = folds[group_id]
    elif row_id in folds:
        value = folds[row_id]
    else:
        raise PrefixIntegrityError(f"no frozen fold for source group {group_id!r}")
    fold = _strict_int(value, name="fold")
    if fold not in range(5):
        raise PrefixIntegrityError(f"fold must be in [0,4], got {fold}")
    return fold


def _budget_sort_key(value: int | str) -> tuple[int, int]:
    return (1, 0) if value == "final" else (0, int(value))


def _processbench_cell(family: str, model: str) -> str:
    # Match ProcessBenchPopulationRow.cell_id in the shared adapter.  The historical
    # CSV's ``processbench_<subset>__<model>`` spelling is a source locator, not a
    # canonical comparison cell identity.
    return f"{PROCESSBENCH_DATASET_REVISION}::{model}::{family}"


def _historical_identity(
    *,
    cell_id: str,
    family: str,
    trace_id: str,
    group: str,
    population: Any | None,
) -> tuple[str, str, str, int | None]:
    """Return row/group/model/registered-length from explicit historical fields."""

    if not trace_id or not group:
        raise PrefixIntegrityError("historical trace_id and group must be non-empty")
    match = _PB_CELL_RE.fullmatch(cell_id)
    if match:
        subset, model = match.groups()
        if subset != family:
            raise PrefixIntegrityError(
                f"historical cell/family disagreement: {cell_id!r} versus {family!r}"
            )
        row_id = canonical_processbench_id(subset, trace_id)
        group_id = row_id
        registered_length = None
        if population is not None:
            pop = population.rows.get(row_id)
            if pop is None:
                raise PrefixIntegrityError(f"historical ID not in ProcessBench population: {row_id}")
            # The trusted telemetry population is the Llama-3.1-8B realization.  It
            # establishes source-question IDs for the other historical model copies,
            # but its trace length/label must not be imposed on those distinct outputs.
            if str(pop.model) == model:
                registered_length = int(pop.trace_length)
        return row_id, group_id, model, registered_length

    # Historical non-ProcessBench rows retain an explicit trace identifier.  The
    # question-level ``group`` deliberately excludes model/cell so repeated scorer
    # copies remain one bootstrap unit.
    row_id = f"historical::{family}::{trace_id}"
    group_id = f"historical::{family}::{group}"
    return row_id, group_id, cell_id, None


def _record(
    *,
    population_id: str,
    row_id: str,
    group_id: str,
    cell_id: str,
    method_id: str,
    score: float,
    label: int,
    budget: int | str,
    fold: int | None,
    source_hash: str,
    family: str,
    model: str,
    final_length: int,
    source_question_id: str,
    source_kind: str,
    direct_eligible: bool = True,
    direct_ineligibility_reason: str | None = None,
    registered_length_match: bool | None = None,
    registered_label_match: bool | None = None,
    input_telemetry_revision: str | None = None,
    input_telemetry_sha256: str | None = None,
) -> dict[str, Any]:
    return make_comparison_record(
        lane="prefix",
        population_id=population_id,
        row_id=row_id,
        group_id=group_id,
        cell_id=cell_id,
        method_id=method_id,
        continuous_score=float(score),
        discrete_prediction=None,
        label=int(label),
        budget=budget,
        fold=fold,
        calibration_hash=None,
        source_artifact_hash=source_hash,
        extra={
            "family": family,
            "model": model,
            "final_length": int(final_length),
            "source_question_id": source_question_id,
            "source_kind": source_kind,
            "direct_eligible": bool(direct_eligible),
            "direct_ineligibility_reason": direct_ineligibility_reason,
            "registered_length_match": registered_length_match,
            "registered_label_match": registered_label_match,
            "input_telemetry_revision": input_telemetry_revision,
            "input_telemetry_sha256": input_telemetry_sha256,
            "prefix_lane_revision": PREFIX_LANE_REVISION,
        },
    )


def _assert_no_duplicate_records(records: Sequence[Mapping[str, Any]], *, source: str) -> None:
    seen: set[tuple[str, str, str, int | str]] = set()
    for record in records:
        key = (
            str(record["cell_id"]),
            str(record["row_id"]),
            str(record["method_id"]),
            record["budget"],
        )
        if key in seen:
            raise PrefixIntegrityError(f"duplicate {source} record {key!r}")
        seen.add(key)


def load_historical_prefix_scores(
    path: str | Path,
    *,
    population: Any | None = None,
    method_map: Mapping[str, str] | None = None,
    folds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Load the frozen 11-cell Online score CSV without positional identities.

    Actual final-score rows use their trace length in the legacy ``budget`` column;
    they are canonicalized to ``budget='final'``.  Prefix rows outside the six-budget
    grid are invalid, and a row at ``final_length == budget`` is rejected rather than
    silently treated as unfinished.
    """

    source = Path(path)
    source_hash = _sha256_file(source)
    method_map = method_map or {
        IU28_METHOD_ID: IU28_METHOD_ID,
        "deepconf_entropy_w64": HISTORICAL_DEEPCONF_METHOD_ID,
    }
    records: list[dict[str, Any]] = []
    source_rows = 0
    selected_rows = 0
    cells: set[str] = set()
    labels: dict[tuple[str, str], int] = {}
    registered_compatibility: dict[
        tuple[str, str], tuple[bool | None, bool | None]
    ] = {}

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "budget",
            "cell_id",
            "family",
            "group",
            "is_final",
            "label_error",
            "method",
            "score",
            "trace_id",
            "trace_length",
        }
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise PrefixIntegrityError(
                f"historical prefix CSV missing columns {sorted(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            source_rows += 1
            if row["method"] not in method_map:
                continue
            selected_rows += 1
            try:
                is_final = _strict_bool(row["is_final"], name="is_final")
                final_length = _strict_int(row["trace_length"], name="trace_length", minimum=1)
                raw_budget = _strict_int(row["budget"], name="budget", minimum=0)
                if is_final:
                    if raw_budget != final_length:
                        raise PrefixIntegrityError(
                            "legacy final row budget must equal trace_length"
                        )
                    budget: int | str = "final"
                else:
                    if raw_budget not in PREFIX_BUDGETS:
                        raise PrefixIntegrityError(
                            f"non-final row has unregistered budget {raw_budget}"
                        )
                    if final_length <= raw_budget:
                        raise PrefixIntegrityError(
                            f"strict causal gate failed: {final_length} <= {raw_budget}"
                        )
                    budget = raw_budget
                family = str(row["family"])
                source_cell_id = str(row["cell_id"])
                trace_id = str(row["trace_id"])
                group = str(row["group"])
                row_id, group_id, model, registered_length = _historical_identity(
                    cell_id=source_cell_id,
                    family=family,
                    trace_id=trace_id,
                    group=group,
                    population=population,
                )
                cell_id = (
                    _processbench_cell(family, model)
                    if row_id.startswith(f"{PROCESSBENCH_DATASET_REVISION}::")
                    else source_cell_id
                )
                label = _binary(row["label_error"], name="label_error")
                registered_length_match = (
                    final_length == registered_length
                    if registered_length is not None
                    else None
                )
                registered_label_match: bool | None = None
                if (
                    population is not None
                    and row_id.startswith(PROCESSBENCH_DATASET_REVISION)
                    and str(population.rows[row_id].model) == model
                ):
                    pop = population.rows[row_id]
                    registered_label_match = label == int(pop.wrong_label)
                compatibility_key = (cell_id, row_id)
                compatibility = (registered_length_match, registered_label_match)
                prior_compatibility = registered_compatibility.get(compatibility_key)
                if prior_compatibility is not None and prior_compatibility != compatibility:
                    raise PrefixIntegrityError(
                        f"registered compatibility disagreement for {compatibility_key!r}"
                    )
                registered_compatibility[compatibility_key] = compatibility
                label_key = (cell_id, row_id)
                if label_key in labels and labels[label_key] != label:
                    raise PrefixIntegrityError(f"method label disagreement for {label_key!r}")
                labels[label_key] = label
                records.append(
                    _record(
                        population_id="prefix_historical_source_v1",
                        row_id=row_id,
                        group_id=group_id,
                        cell_id=cell_id,
                        method_id=method_map[row["method"]],
                        score=_finite(row["score"], name="score"),
                        label=label,
                        budget=budget,
                        fold=_fold_for(row_id, group_id, folds),
                        source_hash=source_hash,
                        family=family,
                        model=model,
                        final_length=final_length,
                        source_question_id=trace_id,
                        source_kind="historical_online_scores",
                        direct_eligible=False,
                        direct_ineligibility_reason=(
                            "precontract early-online telemetry is not the registered "
                            "ProcessBench repgrid realization; CPU replay required"
                        ),
                        registered_length_match=registered_length_match,
                        registered_label_match=registered_label_match,
                        input_telemetry_revision=HISTORICAL_PREFIX_TELEMETRY,
                        input_telemetry_sha256=None,
                    )
                )
                cells.add(cell_id)
            except Exception as exc:  # noqa: BLE001 - add exact source location
                if isinstance(exc, PrefixIntegrityError):
                    raise PrefixIntegrityError(f"{source}:{line_number}: {exc}") from exc
                raise

    if not records:
        raise PrefixIntegrityError("historical prefix CSV produced zero selected records")
    _assert_no_duplicate_records(records, source="historical prefix")
    return {
        "records": tuple(
            sorted(
                records,
                key=lambda row: (
                    row["cell_id"],
                    row["row_id"],
                    row["method_id"],
                    _budget_sort_key(row["budget"]),
                ),
            )
        ),
        "audit": {
            "schema": "prefix_source_audit_v1",
            "source": str(source),
            "source_artifact_hash": source_hash,
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "emitted_records": len(records),
            "cells": sorted(cells),
            "method_ids": sorted(set(method_map.values())),
            "context_only": True,
            "direct_eligible": False,
            "direct_ineligibility_reason": (
                "precontract early-online telemetry is not the registered ProcessBench "
                "repgrid realization; CPU replay required"
            ),
            "registered_comparable_traces": sum(
                length_match is not None
                for length_match, _ in registered_compatibility.values()
            ),
            "registered_length_match_traces": sum(
                length_match is True
                for length_match, _ in registered_compatibility.values()
            ),
            "registered_label_match_traces": sum(
                label_match is True
                for _, label_match in registered_compatibility.values()
            ),
            "positional_fallback": False,
            "strict_length_gt_budget": True,
        },
    }


def load_unified28_prefix_records(
    path: str | Path,
    population: Any,
    *,
    candidate: str = "base7_full28",
    method_id: str = UNIFIED28_METHOD_ID,
    model: str = "llama31_8b",
    folds: Mapping[str, int] | None = None,
    strict_population_coverage: bool = True,
    require_registered_telemetry_provenance: bool = False,
) -> dict[str, Any]:
    """Load ordinary frozen Unified-28 prefix and final scores by official ID."""

    source = Path(path)
    source_hash = _sha256_file(source)
    telemetry_provenance = _verify_unified_telemetry_provenance(
        source,
        population,
        required=require_registered_telemetry_provenance,
    )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_rows = 0
    selected_rows = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PrefixIntegrityError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping) or row.get("candidate") != candidate:
                continue
            selected_rows += 1
            required = {
                "family",
                "unit",
                "source_group",
                "model",
                "wrong",
                "global_score",
                *(f"risk_at_{budget}" for budget in PREFIX_BUDGETS),
            }
            missing = required.difference(row)
            if missing:
                raise PrefixIntegrityError(
                    f"{source}:{line_number}: Unified-28 row missing {sorted(missing)}"
                )
            family = str(row["family"])
            unit = str(row["unit"])
            if row["source_group"] != f"{family}::{unit}":
                raise PrefixIntegrityError(
                    f"{source}:{line_number}: source_group is not family::unit"
                )
            if str(row["model"]) != model:
                raise PrefixIntegrityError(
                    f"{source}:{line_number}: model {row['model']!r} != {model!r}"
                )
            row_id = canonical_processbench_id(family, unit)
            if row_id in seen_ids:
                raise PrefixIntegrityError(f"duplicate Unified-28 source ID {row_id}")
            seen_ids.add(row_id)
            pop = population.rows.get(row_id)
            if pop is None:
                raise PrefixIntegrityError(f"Unified-28 ID not in ProcessBench population: {row_id}")
            label = _binary(row["wrong"], name="wrong")
            if label != int(pop.wrong_label):
                raise PrefixIntegrityError(
                    f"Unified-28 label disagreement for {row_id}: "
                    f"artifact={label}, population={pop.wrong_label}"
                )
            cell_id = _processbench_cell(family, model)
            fold = _fold_for(row_id, row_id, folds)
            for budget in PREFIX_BUDGETS:
                if int(pop.trace_length) <= budget:
                    continue
                records.append(
                    _record(
                        population_id="prefix_unified28_processbench_source_v1",
                        row_id=row_id,
                        group_id=row_id,
                        cell_id=cell_id,
                        method_id=method_id,
                        score=_finite(row[f"risk_at_{budget}"], name=f"risk_at_{budget}"),
                        label=label,
                        budget=budget,
                        fold=fold,
                        source_hash=source_hash,
                        family=family,
                        model=model,
                        final_length=int(pop.trace_length),
                        source_question_id=unit,
                        source_kind="unified28_validation_records",
                        input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                        input_telemetry_sha256=telemetry_provenance["telemetry_sha256"],
                    )
                )
            records.append(
                _record(
                    population_id="prefix_unified28_processbench_source_v1",
                    row_id=row_id,
                    group_id=row_id,
                    cell_id=cell_id,
                    method_id=method_id,
                    score=_finite(row["global_score"], name="global_score"),
                    label=label,
                    budget="final",
                    fold=fold,
                    source_hash=source_hash,
                    family=family,
                    model=model,
                    final_length=int(pop.trace_length),
                    source_question_id=unit,
                    source_kind="unified28_validation_records",
                    input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                    input_telemetry_sha256=telemetry_provenance["telemetry_sha256"],
                )
            )

    expected_ids = set(population.ordered_ids)
    coverage = len(seen_ids.intersection(expected_ids)) / len(expected_ids) if expected_ids else 0.0
    if strict_population_coverage and seen_ids != expected_ids:
        missing = sorted(expected_ids.difference(seen_ids))
        extra = sorted(seen_ids.difference(expected_ids))
        raise PrefixIntegrityError(
            f"Unified-28 population coverage failed: coverage={coverage:.6f}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    _assert_no_duplicate_records(records, source="Unified-28")
    return {
        "records": tuple(records),
        "audit": {
            "schema": "prefix_source_audit_v1",
            "source": str(source),
            "source_artifact_hash": source_hash,
            "source_rows": source_rows,
            "selected_candidate_rows": selected_rows,
            "population_rows": len(expected_ids),
            "population_coverage": coverage,
            "emitted_records": len(records),
            "candidate": candidate,
            "method_id": method_id,
            "registered_telemetry_provenance": telemetry_provenance,
            "positional_fallback": False,
            "strict_length_gt_budget": True,
        },
    }


def load_step272_prefix_records(
    path: str | Path,
    population: Any,
    *,
    architecture: str = SELECTED_STEP272_ARCHITECTURE,
    method_id: str = STEP272_METHOD_ID,
    model: str = "llama31_8b",
    folds: Mapping[str, int] | None = None,
    require_registered_telemetry_provenance: bool = False,
) -> dict[str, Any]:
    """Load the selected Step-272 0.50/0.50 Global/Local Online architecture.

    Online rows provide the six causal scores.  The architecture's frozen Global row
    provides the corresponding full-trace sidecar; Local rows are irrelevant here.
    """

    source = Path(path)
    source_hash = _sha256_file(source)
    telemetry_provenance = _verify_step272_telemetry_provenance(
        source,
        population,
        model=model,
        required=require_registered_telemetry_provenance,
    )
    records: list[dict[str, Any]] = []
    source_rows = 0
    selected_rows = 0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "architecture",
            "budget",
            "family",
            "model",
            "score",
            "target",
            "task",
            "unit",
        }
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise PrefixIntegrityError(
                f"Step-272 CSV missing columns {sorted(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            source_rows += 1
            if row["architecture"] != architecture or row["model"] != model:
                continue
            if row["task"] not in ("online", "global"):
                continue
            if row["task"] == "global" and row["budget"] != "final":
                raise PrefixIntegrityError(f"{source}:{line_number}: Global row is not final")
            if row["task"] == "online":
                budget: int | str = _strict_int(row["budget"], name="budget", minimum=0)
                if budget not in PREFIX_BUDGETS:
                    raise PrefixIntegrityError(
                        f"{source}:{line_number}: unregistered online budget {budget}"
                    )
            else:
                budget = "final"
            selected_rows += 1
            family = str(row["family"])
            unit = str(row["unit"])
            row_id = canonical_processbench_id(family, unit)
            pop = population.rows.get(row_id)
            if pop is None:
                raise PrefixIntegrityError(f"Step-272 ID not in ProcessBench population: {row_id}")
            label = _binary(row["target"], name="target")
            if label != int(pop.wrong_label):
                raise PrefixIntegrityError(
                    f"Step-272 label disagreement for {row_id}: "
                    f"artifact={label}, population={pop.wrong_label}"
                )
            if budget != "final" and int(pop.trace_length) <= int(budget):
                raise PrefixIntegrityError(
                    f"Step-272 emitted an ineligible prefix: {row_id}, "
                    f"length={pop.trace_length}, budget={budget}"
                )
            records.append(
                _record(
                    population_id="prefix_step272_processbench_source_v1",
                    row_id=row_id,
                    group_id=row_id,
                    cell_id=_processbench_cell(family, model),
                    method_id=method_id,
                    score=_finite(row["score"], name="score"),
                    label=label,
                    budget=budget,
                    fold=_fold_for(row_id, row_id, folds),
                    source_hash=source_hash,
                    family=family,
                    model=model,
                    final_length=int(pop.trace_length),
                    source_question_id=unit,
                    source_kind="step272_selected_architecture",
                    input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                    input_telemetry_sha256=telemetry_provenance["telemetry_sha256"],
                )
            )

    if not records:
        raise PrefixIntegrityError(
            f"Step-272 artifact contains no rows for {architecture}/{model}"
        )
    _assert_no_duplicate_records(records, source="Step-272")
    return {
        "records": tuple(records),
        "audit": {
            "schema": "prefix_source_audit_v1",
            "source": str(source),
            "source_artifact_hash": source_hash,
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "emitted_records": len(records),
            "architecture": architecture,
            "method_id": method_id,
            "model": model,
            "registered_telemetry_provenance": telemetry_provenance,
            "positional_fallback": False,
            "strict_length_gt_budget": True,
        },
    }


def build_entropy_prefix_records(
    rows_by_subset: Mapping[str, Mapping[Any, Any] | Sequence[Any]],
    population: Any,
    *,
    source_artifact_hashes: Mapping[str, str],
    model: str = "llama31_8b",
    include_row_ids: set[str] | None = None,
    folds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build causal mean/max entropy controls from explicit telemetry rows.

    The prefix statistic uses exactly ``token_entropies[:budget]``.  No completed-trace
    mean, suffix, future window, or padded value enters a prefix score.
    """

    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    expected_selected = set(include_row_ids or population.ordered_ids)
    normalized_source_hashes = {
        subset: _require_hash(
            source_artifact_hashes.get(subset, ""), name=f"{subset} source hash"
        )
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath")
    }
    registered_hashes = _registered_population_hashes(population)
    if registered_hashes is not None and normalized_source_hashes != registered_hashes:
        raise PrefixIntegrityError(
            "entropy telemetry hashes disagree with the registered population"
        )
    telemetry_sha256 = _telemetry_bundle_sha256(normalized_source_hashes)
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
        if subset not in rows_by_subset:
            raise PrefixIntegrityError(f"missing telemetry subset {subset}")
        source_hash = normalized_source_hashes[subset]
        source = rows_by_subset[subset]
        values = source.values() if isinstance(source, Mapping) else source
        for source_index, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise PrefixIntegrityError(f"{subset} telemetry row {source_index} is not a mapping")
            if "id" not in row:
                raise PrefixIntegrityError(
                    f"{subset} telemetry row {source_index} lacks id; positional fallback forbidden"
                )
            row_id = canonical_processbench_id(subset, str(row["id"]))
            if row_id not in expected_selected:
                continue
            if row_id in observed_ids:
                raise PrefixIntegrityError(f"duplicate entropy telemetry ID {row_id}")
            observed_ids.add(row_id)
            required = ("gen_token_ids", "token_entropies", "final_answer_correct")
            missing = [field for field in required if field not in row]
            if missing:
                raise PrefixIntegrityError(f"entropy telemetry {row_id} missing {missing}")
            token_ids = row["gen_token_ids"]
            entropies = row["token_entropies"]
            if (
                isinstance(token_ids, (str, bytes))
                or isinstance(entropies, (str, bytes))
                or not isinstance(token_ids, Sequence)
                or not isinstance(entropies, Sequence)
            ):
                raise PrefixIntegrityError(f"entropy telemetry {row_id} has non-sequence trace")
            if len(token_ids) == 0 or len(entropies) != len(token_ids):
                raise PrefixIntegrityError(
                    f"entropy telemetry length mismatch for {row_id}: "
                    f"tokens={len(token_ids)}, entropy={len(entropies)}"
                )
            pop = population.rows.get(row_id)
            if pop is None:
                raise PrefixIntegrityError(f"entropy ID not in ProcessBench population: {row_id}")
            if len(token_ids) != int(pop.trace_length):
                raise PrefixIntegrityError(f"entropy/population length disagreement for {row_id}")
            label = 1 - _binary(row["final_answer_correct"], name="final_answer_correct")
            if label != int(pop.wrong_label):
                raise PrefixIntegrityError(f"entropy/population label disagreement for {row_id}")
            numeric_entropy = [
                _finite(value, name=f"token_entropies[{index}]")
                for index, value in enumerate(entropies)
            ]
            fold = _fold_for(row_id, row_id, folds)
            cell_id = _processbench_cell(subset, model)
            for budget in PREFIX_BUDGETS:
                if len(numeric_entropy) <= budget:
                    continue
                prefix = numeric_entropy[:budget]
                for method_id, score in (
                    (MEAN_ENTROPY_METHOD_ID, sum(prefix) / len(prefix)),
                    (MAX_ENTROPY_METHOD_ID, max(prefix)),
                ):
                    records.append(
                        _record(
                            population_id="prefix_entropy_processbench_source_v1",
                            row_id=row_id,
                            group_id=row_id,
                            cell_id=cell_id,
                            method_id=method_id,
                            score=score,
                            label=label,
                            budget=budget,
                            fold=fold,
                            source_hash=source_hash,
                            family=subset,
                            model=model,
                            final_length=len(numeric_entropy),
                            source_question_id=str(row["id"]),
                            source_kind="processbench_entropy_telemetry",
                            input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                            input_telemetry_sha256=telemetry_sha256,
                        )
                    )
            for method_id, score in (
                (MEAN_ENTROPY_METHOD_ID, sum(numeric_entropy) / len(numeric_entropy)),
                (MAX_ENTROPY_METHOD_ID, max(numeric_entropy)),
            ):
                records.append(
                    _record(
                        population_id="prefix_entropy_processbench_source_v1",
                        row_id=row_id,
                        group_id=row_id,
                        cell_id=cell_id,
                        method_id=method_id,
                        score=score,
                        label=label,
                        budget="final",
                        fold=fold,
                        source_hash=source_hash,
                        family=subset,
                        model=model,
                        final_length=len(numeric_entropy),
                        source_question_id=str(row["id"]),
                        source_kind="processbench_entropy_telemetry",
                        input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                        input_telemetry_sha256=telemetry_sha256,
                    )
                )

    missing_ids = sorted(expected_selected.difference(observed_ids))
    if missing_ids:
        raise PrefixIntegrityError(
            f"entropy telemetry is missing {len(missing_ids)} selected IDs: {missing_ids[:5]}"
        )
    _assert_no_duplicate_records(records, source="entropy baselines")
    return {
        "records": tuple(records),
        "audit": {
            "schema": "prefix_source_audit_v1",
            "source": "processbench_entropy_telemetry",
            "source_artifact_hashes": dict(sorted(source_artifact_hashes.items())),
            "registered_telemetry_sha256": telemetry_sha256,
            "registered_telemetry_hash_match": (
                registered_hashes is None or normalized_source_hashes == registered_hashes
            ),
            "selected_ids": len(expected_selected),
            "observed_ids": len(observed_ids),
            "emitted_records": len(records),
            "methods": [MEAN_ENTROPY_METHOD_ID, MAX_ENTROPY_METHOD_ID],
            "prefix_definition": "token_entropies[:budget]",
            "future_access": False,
            "strict_length_gt_budget": True,
            "positional_fallback": False,
        },
    }


def _array_fingerprint(value: Any) -> dict[str, Any]:
    """Content-address one fitted numeric array without serializing NaN as JSON."""

    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _frozen_iu_parameter_hash(model: Any) -> str:
    transformer = model.transformer
    projection = {
        "feature_names": list(model.feature_names),
        "raw_keep": _array_fingerprint(model.raw_keep),
        "transformer": {
            "names": list(transformer.names),
            "raw_median": _array_fingerprint(transformer.raw_median),
            "oriented_mean": _array_fingerprint(transformer.oriented_mean),
            "oriented_std": _array_fingerprint(transformer.oriented_std),
            "sorted_oriented": [
                _array_fingerprint(values) for values in transformer.sorted_oriented
            ],
            "mode_centres": _array_fingerprint(transformer.mode_centres),
            "output_mean": _array_fingerprint(transformer.output_mean),
            "output_std": _array_fingerprint(transformer.output_std),
            "training_output": _array_fingerprint(transformer.training_output),
        },
        "transformed_keep": _array_fingerprint(model.transformed_keep),
        "transformed_mean": _array_fingerprint(model.transformed_mean),
        "transformed_std": _array_fingerprint(model.transformed_std),
        "weights": _array_fingerprint(model.weights),
        "diagnostics": model.diagnostics,
    }
    return canonical_sha256(projection)


def _historical_anchor_map(
    path: Path,
    *,
    methods: set[str],
) -> dict[tuple[str, int, bool, str], dict[str, Any]]:
    anchors: dict[tuple[str, int, bool, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "budget",
            "group",
            "is_final",
            "label_error",
            "method",
            "score",
            "trace_id",
            "trace_length",
            "unit_index",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise PrefixIntegrityError(
                f"historical replay anchor {path} missing {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            if row["method"] not in methods:
                continue
            is_final = _strict_bool(row["is_final"], name="is_final")
            key = (
                str(row["trace_id"]),
                _strict_int(row["budget"], name="budget", minimum=0),
                is_final,
                str(row["method"]),
            )
            if key in anchors:
                raise PrefixIntegrityError(f"{path}:{line_number}: duplicate anchor {key!r}")
            anchors[key] = {
                "trace_id": key[0],
                "budget": key[1],
                "is_final": key[2],
                "method": key[3],
                "score": _finite(row["score"], name="score"),
                "group": str(row["group"]),
                "label_error": _binary(row["label_error"], name="label_error"),
                "trace_length": _strict_int(
                    row["trace_length"], name="trace_length", minimum=1
                ),
                "unit_index": _strict_int(row["unit_index"], name="unit_index", minimum=0),
            }
    if not anchors:
        raise PrefixIntegrityError(f"historical replay anchor {path} has no required rows")
    return anchors


def _generated_anchor_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, bool, str], dict[str, Any]]:
    selected = {IU28_METHOD_ID, "deepconf_entropy_w64"}
    output: dict[tuple[str, int, bool, str], dict[str, Any]] = {}
    for row in rows:
        if row["method"] not in selected:
            continue
        key = (
            str(row["trace_id"]),
            _strict_int(row["budget"], name="budget", minimum=0),
            bool(row["is_final"]),
            str(row["method"]),
        )
        if key in output:
            raise PrefixIntegrityError(f"duplicate reconstructed anchor {key!r}")
        output[key] = {
            "trace_id": key[0],
            "budget": key[1],
            "is_final": key[2],
            "method": key[3],
            "score": _finite(row["score"], name="score"),
            "group": str(row["group"]),
            "label_error": _binary(row["label_error"], name="label_error"),
            "trace_length": _strict_int(
                row["trace_length"], name="trace_length", minimum=1
            ),
            "unit_index": _strict_int(row["unit_index"], name="unit_index", minimum=0),
        }
    return output


def _assert_exact_anchor_identity(
    expected: Mapping[tuple[str, int, bool, str], Mapping[str, Any]],
    observed: Mapping[tuple[str, int, bool, str], Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    if set(expected) != set(observed):
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise PrefixIntegrityError(
            f"{source} anchor key disagreement: missing={missing[:3]}, extra={extra[:3]}"
        )
    fields = ("group", "label_error", "trace_length", "unit_index", "score")
    for key in sorted(expected):
        for field in fields:
            if expected[key][field] != observed[key][field]:
                raise PrefixIntegrityError(
                    f"{source} anchor mismatch {key!r}.{field}: "
                    f"{expected[key][field]!r} != {observed[key][field]!r}"
                )
    ordered = [dict(expected[key]) for key in sorted(expected)]
    return {
        "n_anchor_rows": len(ordered),
        "exact_float_identity": True,
        "max_abs_score_difference": 0.0,
        "anchor_sha256": canonical_sha256(ordered),
    }


def _reconstruct_frozen_prefix_incumbent(
    family: str,
    historical_results_root: Path,
) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """Reconstruct one cell's exact label-free IU fit and prove stored anchors."""

    from .. import online_convergence as online

    cell_id = f"processbench_{family}__llama31_8b"
    cell_root = historical_results_root / cell_id
    result_path = cell_root / "result.json"
    calibration_path = cell_root / "scores_calibration.csv"
    evaluation_path = cell_root / "scores_evaluation.csv"
    required_paths = (result_path, calibration_path, evaluation_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise PrefixIntegrityError(
            f"frozen {cell_id} replay assets missing: {missing_paths}"
        )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixIntegrityError(f"invalid frozen replay result {result_path}") from exc
    metadata = result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PrefixIntegrityError(f"{result_path} lacks metadata")
    if metadata.get("protocol") != "EARLY_ONLINE_LOCALIZATION_MODELS_V1":
        raise PrefixIntegrityError(f"unexpected frozen replay protocol for {cell_id}")
    if metadata.get("cell_id") != cell_id:
        raise PrefixIntegrityError(f"frozen replay cell ID disagreement for {cell_id}")
    if tuple(metadata.get("budgets", ())) != PREFIX_BUDGETS:
        raise PrefixIntegrityError(f"frozen replay budgets disagree for {cell_id}")
    source_path = Path(str(metadata.get("source_path", "")))
    if not source_path.is_file():
        raise PrefixIntegrityError(f"frozen fit source is missing: {source_path}")
    source_hash = _sha256_file(source_path)
    if source_hash != metadata.get("source_sha256"):
        raise PrefixIntegrityError(f"frozen fit source hash drift for {cell_id}")
    if source_path.stat().st_size != _strict_int(
        metadata.get("source_size_bytes"), name="source_size_bytes", minimum=1
    ):
        raise PrefixIntegrityError(f"frozen fit source size drift for {cell_id}")
    record_filter = metadata.get("record_filter")
    if not isinstance(record_filter, Mapping) or not record_filter.get("field"):
        raise PrefixIntegrityError(f"frozen replay filter is absent for {cell_id}")
    with source_path.open("rb") as handle:
        source_cache = pickle.load(handle)
    records = online.normalize_cache_records(source_cache, min_tokens=min(PREFIX_BUDGETS))
    records = [
        row
        for row in records
        if str(row.get(record_filter["field"])) == str(record_filter.get("value"))
    ]
    if len(records) != _strict_int(metadata.get("n_records"), name="n_records", minimum=1):
        raise PrefixIntegrityError(f"frozen replay record-count drift for {cell_id}")
    split_seed = _strict_int(metadata.get("split_seed"), name="split_seed", minimum=0)
    calibration, evaluation, used_seed = online.grouped_calibration_split(
        records, seed=split_seed
    )
    if used_seed != split_seed:
        raise PrefixIntegrityError(f"frozen replay split seed drift for {cell_id}")
    if len(calibration) != _strict_int(
        metadata.get("n_calibration"), name="n_calibration", minimum=1
    ) or len(evaluation) != _strict_int(
        metadata.get("n_evaluation"), name="n_evaluation", minimum=1
    ):
        raise PrefixIntegrityError(f"frozen replay split-count drift for {cell_id}")
    rows_per_trace = _strict_int(
        metadata.get("rows_per_trace"), name="rows_per_trace", minimum=1
    )
    model = online.fit_frozen_prefix_iu(
        [records[index] for index in calibration],
        include_elapsed_length=False,
        rows_per_trace=rows_per_trace,
    )
    expected_diagnostics = result.get("models", {}).get(IU28_METHOD_ID)
    if expected_diagnostics != model.diagnostics:
        raise PrefixIntegrityError(f"frozen IU diagnostics drift for {cell_id}")

    anchor_audits: dict[str, Any] = {}
    for split_name, indexes, anchor_path in (
        ("calibration", calibration, calibration_path),
        ("evaluation", evaluation, evaluation_path),
    ):
        generated_rows = online.build_score_rows(
            records,
            indexes,
            {IU28_METHOD_ID: model},
            budgets=PREFIX_BUDGETS,
        )
        expected = _historical_anchor_map(
            anchor_path,
            methods={IU28_METHOD_ID, "deepconf_entropy_w64"},
        )
        observed = _generated_anchor_map(generated_rows)
        anchor_audits[split_name] = _assert_exact_anchor_identity(
            expected, observed, source=f"{cell_id}/{split_name}"
        )

    dependency_names = (
        "online_convergence.py",
        "fixed_application_pipelines.py",
        "repeated_measurement_reliability.py",
        "upcr.py",
        "streaming_utils.py",
    )
    spectral_root = Path(__file__).resolve().parents[1]
    dependency_hashes = {
        name: _sha256_file(spectral_root / name) for name in dependency_names
    }
    parameter_hash = _frozen_iu_parameter_hash(model)
    artifact_hashes = {
        "fit_source": source_hash,
        "result": _sha256_file(result_path),
        "calibration_scores": _sha256_file(calibration_path),
        "evaluation_scores": _sha256_file(evaluation_path),
    }
    audit = {
        "cell_id": cell_id,
        "family": family,
        "fit_source": str(source_path),
        "fit_source_sha256": source_hash,
        "fit_records": len(records),
        "calibration_records": len(calibration),
        "evaluation_records": len(evaluation),
        "split_seed": split_seed,
        "rows_per_trace": rows_per_trace,
        "parameter_sha256": parameter_hash,
        "diagnostics_exact_identity": True,
        "anchor_audits": anchor_audits,
        "dependency_hashes": dependency_hashes,
        "artifact_hashes": artifact_hashes,
        "labels_used_for_score_parameter_fit": False,
        "original_labels_used_for_split_validity": True,
        "registered_outcomes_used_for_fit": False,
    }
    method_source_hashes = {
        IU28_METHOD_ID: canonical_sha256(
            {
                "revision": FROZEN_PREFIX_REPLAY_REVISION,
                "method": IU28_METHOD_ID,
                "parameter_sha256": parameter_hash,
                "dependency_hashes": dependency_hashes,
                "artifact_hashes": artifact_hashes,
                "anchors": anchor_audits,
            }
        ),
        HISTORICAL_DEEPCONF_METHOD_ID: canonical_sha256(
            {
                "revision": FROZEN_PREFIX_REPLAY_REVISION,
                "method": HISTORICAL_DEEPCONF_METHOD_ID,
                "window": 64,
                "orientation": "risk_is_negative_lowest_group_confidence",
                "streaming_utils_sha256": dependency_hashes["streaming_utils.py"],
                "artifact_hashes": artifact_hashes,
                "anchors": anchor_audits,
            }
        ),
    }
    return model, audit, method_source_hashes


def replay_frozen_prefix_incumbents(
    rows_by_subset: Mapping[str, Mapping[Any, Any] | Sequence[Any]],
    population: Any,
    *,
    historical_results_root: str | Path,
    source_artifact_hashes: Mapping[str, str],
    folds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Exactly replay IU28 and the historical DeepConf proxy on registered telemetry.

    Replay is authorized only after reconstructing each original per-family IU fit
    from its hashed calibration cache and reproducing every stored calibration and
    evaluation anchor score with exact floating-point identity.  The historical model
    is never refit on registered outcomes, and DeepConf remains explicitly a proxy.
    """

    from ..online_convergence import prefix_method_scores

    results_root = Path(historical_results_root)
    if not results_root.is_dir():
        raise PrefixIntegrityError(f"historical replay root is missing: {results_root}")
    normalized_source_hashes = {
        subset: _require_hash(
            source_artifact_hashes.get(subset, ""), name=f"{subset} source hash"
        )
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath")
    }
    registered_hashes = _registered_population_hashes(population)
    if registered_hashes is None or registered_hashes != normalized_source_hashes:
        raise PrefixIntegrityError(
            "incumbent replay requires exact registered ProcessBench telemetry hashes"
        )
    telemetry_sha256 = _telemetry_bundle_sha256(registered_hashes)

    models: dict[str, Any] = {}
    fit_audits: dict[str, Any] = {}
    method_source_hashes: dict[str, dict[str, str]] = {}
    for family in ("gsm8k", "math", "olympiadbench", "omnimath"):
        model, audit, hashes = _reconstruct_frozen_prefix_incumbent(
            family, results_root
        )
        models[family] = model
        fit_audits[family] = audit
        method_source_hashes[family] = hashes

    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for family in ("gsm8k", "math", "olympiadbench", "omnimath"):
        if family not in rows_by_subset:
            raise PrefixIntegrityError(f"incumbent replay missing telemetry subset {family}")
        source = rows_by_subset[family]
        values = source.values() if isinstance(source, Mapping) else source
        for source_index, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise PrefixIntegrityError(
                    f"{family} replay telemetry row {source_index} is not a mapping"
                )
            if "id" not in row:
                raise PrefixIntegrityError(
                    f"{family} replay telemetry row {source_index} lacks id; "
                    "positional fallback forbidden"
                )
            row_id = canonical_processbench_id(family, str(row["id"]))
            if row_id in observed_ids:
                raise PrefixIntegrityError(f"duplicate incumbent replay ID {row_id}")
            observed_ids.add(row_id)
            pop = population.rows.get(row_id)
            if pop is None:
                raise PrefixIntegrityError(f"incumbent replay ID outside population: {row_id}")
            required = (
                "gen_token_ids",
                "token_entropies",
                "token_spilled_energies",
                "token_logsumexp",
                "top_k_logprobs",
                "final_answer_correct",
            )
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise PrefixIntegrityError(f"incumbent replay {row_id} missing {missing}")
            final_length = len(row["gen_token_ids"])
            if final_length != int(pop.trace_length):
                raise PrefixIntegrityError(f"incumbent replay length disagreement for {row_id}")
            for field in (
                "token_entropies",
                "token_spilled_energies",
                "token_logsumexp",
            ):
                if len(row[field]) != final_length:
                    raise PrefixIntegrityError(
                        f"incumbent replay {row_id} has unaligned {field}"
                    )
            top_k = row["top_k_logprobs"]
            if not isinstance(top_k, Mapping) or not top_k:
                raise PrefixIntegrityError(f"incumbent replay {row_id} has invalid top-k data")
            if any(len(values) != final_length for values in top_k.values()):
                raise PrefixIntegrityError(f"incumbent replay {row_id} has unaligned top-k data")
            label = 1 - _binary(
                row["final_answer_correct"], name="final_answer_correct"
            )
            if label != int(pop.wrong_label):
                raise PrefixIntegrityError(f"incumbent replay label disagreement for {row_id}")
            fold = _fold_for(row_id, row_id, folds)
            cell_id = _processbench_cell(family, "llama31_8b")
            model = models[family]
            for budget in (*PREFIX_BUDGETS, "final"):
                if budget != "final" and final_length <= int(budget):
                    continue
                raw_budget = None if budget == "final" else int(budget)
                scores = prefix_method_scores(
                    row,
                    raw_budget,
                    {IU28_METHOD_ID: model},
                )
                for method_id, source_method in (
                    (IU28_METHOD_ID, IU28_METHOD_ID),
                    (HISTORICAL_DEEPCONF_METHOD_ID, "deepconf_entropy_w64"),
                ):
                    records.append(
                        _record(
                            population_id="prefix_frozen_incumbent_replay_source_v1",
                            row_id=row_id,
                            group_id=row_id,
                            cell_id=cell_id,
                            method_id=method_id,
                            score=_finite(scores[source_method], name=source_method),
                            label=label,
                            budget=budget,
                            fold=fold,
                            source_hash=method_source_hashes[family][method_id],
                            family=family,
                            model="llama31_8b",
                            final_length=final_length,
                            source_question_id=str(row["id"]),
                            source_kind="exact_cpu_replay_original_frozen_fit",
                            direct_eligible=True,
                            input_telemetry_revision=REGISTERED_LLAMA_PREFIX_TELEMETRY,
                            input_telemetry_sha256=telemetry_sha256,
                        )
                    )

    expected_ids = set(population.ordered_ids)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids.difference(observed_ids))
        extra = sorted(observed_ids.difference(expected_ids))
        raise PrefixIntegrityError(
            f"incumbent replay population coverage failed: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    _assert_no_duplicate_records(records, source="frozen incumbent replay")
    return {
        "records": tuple(
            sorted(
                records,
                key=lambda row: (
                    row["cell_id"],
                    row["row_id"],
                    row["method_id"],
                    _budget_sort_key(row["budget"]),
                ),
            )
        ),
        "audit": {
            "schema": "prefix_frozen_incumbent_replay_audit_v1",
            "revision": FROZEN_PREFIX_REPLAY_REVISION,
            "historical_results_root": str(results_root),
            "registered_telemetry_sha256": telemetry_sha256,
            "registered_source_hashes": registered_hashes,
            "population_rows": len(expected_ids),
            "population_coverage": 1.0,
            "emitted_records": len(records),
            "methods": [IU28_METHOD_ID, HISTORICAL_DEEPCONF_METHOD_ID],
            "fit_audits": fit_audits,
            "all_anchor_scores_exact": True,
            "parameters_reconstructed_from_original_fit_ids": True,
            "registered_outcomes_used_for_fit": False,
            "deepconf_is_historical_proxy": True,
            "strict_length_gt_budget": True,
            "positional_fallback": False,
            "gpu_used": False,
        },
    }


def _expected_budgets(final_length: int) -> set[int | str]:
    return {"final", *(budget for budget in PREFIX_BUDGETS if final_length > budget)}


def assemble_historical_common_panel(
    records: Iterable[Mapping[str, Any]],
    *,
    required_methods: Sequence[str] = DIRECT_REQUIRED_METHODS,
    reference_method: str = STEP272_METHOD_ID,
    coverage_reference_method: str = IU28_METHOD_ID,
) -> dict[str, Any]:
    """Build direct tables only where all required frozen methods are complete.

    The dedicated Step-272 method defines the registered direct-trace universe.  A
    trace enters a direct cell only when every required method has its final score and
    every causally eligible registered budget on the same validated telemetry
    realization.  Context-only records (notably the pre-contract 11-cell IU/DeepConf
    artifact) remain in the coverage matrix but can never enter this intersection.
    Optional methods enter a direct table only at 100% registered coverage.
    """

    normalized = [validate_comparison_record(record) for record in records]
    if not normalized:
        raise PrefixIntegrityError("cannot assemble an empty prefix panel")
    required_methods = tuple(dict.fromkeys(str(method) for method in required_methods))
    if reference_method not in required_methods:
        raise PrefixIntegrityError(
            f"dedicated reference method {reference_method!r} must be required"
        )

    records_by_trace_method: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    direct_index: dict[tuple[str, str, str, int | str], dict[str, Any]] = {}
    direct_observed_budgets: dict[
        tuple[str, str, str], set[int | str]
    ] = defaultdict(set)
    trace_meta: dict[
        tuple[str, str],
        tuple[int, int, str, str, int | None, str | None, str | None],
    ] = {}
    methods = sorted({str(record["method_id"]) for record in normalized})
    for record in normalized:
        trio = (
            str(record["cell_id"]),
            str(record["row_id"]),
            str(record["method_id"]),
        )
        records_by_trace_method[trio].append(record)
        if record.get("direct_eligible", True) is False:
            continue
        key = (
            trio[0],
            trio[1],
            trio[2],
            record["budget"],
        )
        if key in direct_index:
            raise PrefixIntegrityError(f"duplicate direct-eligible prefix record {key!r}")
        direct_index[key] = record
        direct_observed_budgets[trio].add(key[3])
        meta_key = (str(record["cell_id"]), str(record["row_id"]))
        current = (
            _strict_int(record["final_length"], name="final_length", minimum=1),
            _binary(record["label"], name="label"),
            str(record["group_id"]),
            str(record.get("family", "")),
            record.get("fold"),
            record.get("input_telemetry_revision"),
            record.get("input_telemetry_sha256"),
        )
        previous = trace_meta.get(meta_key)
        if previous is not None and previous != current:
            raise PrefixIntegrityError(
                f"trace metadata disagreement for {meta_key!r}: {previous!r} vs {current!r}"
            )
        trace_meta[meta_key] = current

    # For coverage, prefer a registered direct replay if it exists; otherwise retain
    # the context realization.  This also permits a future IU replay to supersede the
    # pre-contract IU row without combining budgets from the two realizations.
    coverage_budgets: dict[tuple[str, str, str], set[int | str]] = {}
    coverage_lengths: dict[tuple[str, str, str], int] = {}
    for trio, values in records_by_trace_method.items():
        eligible = [row for row in values if row.get("direct_eligible", True) is not False]
        selected = eligible or values
        budget_rows: dict[int | str, Mapping[str, Any]] = {}
        lengths: set[int] = set()
        for row in selected:
            budget = row["budget"]
            if budget in budget_rows:
                raise PrefixIntegrityError(
                    f"duplicate records within one telemetry realization: {trio!r}@{budget!r}"
                )
            budget_rows[budget] = row
            lengths.add(_strict_int(row["final_length"], name="final_length", minimum=1))
        if len(lengths) != 1:
            raise PrefixIntegrityError(
                f"within-method trace-length disagreement for {trio!r}: {sorted(lengths)}"
            )
        coverage_budgets[trio] = set(budget_rows)
        coverage_lengths[trio] = next(iter(lengths))

    coverage_reference_traces: dict[str, set[str]] = defaultdict(set)
    direct_reference_traces: dict[str, set[str]] = defaultdict(set)
    all_cells = {str(record["cell_id"]) for record in normalized}
    for cell_id, row_id, method_id in coverage_budgets:
        if (
            method_id == coverage_reference_method
            and "final" in coverage_budgets[(cell_id, row_id, method_id)]
        ):
            coverage_reference_traces[cell_id].add(row_id)
    for cell_id, row_id, method_id, budget in direct_index:
        if method_id == reference_method and budget == "final":
            direct_reference_traces[cell_id].add(row_id)
    if not direct_reference_traces:
        raise PrefixIntegrityError(f"reference method {reference_method!r} has no final rows")

    direct_records: list[dict[str, Any]] = []
    populations: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for cell_id in sorted(all_cells):
        coverage_universe = coverage_reference_traces.get(cell_id, set())
        if not coverage_universe:
            coverage_universe = {
                row_id
                for candidate_cell, row_id, candidate_method in coverage_budgets
                if candidate_cell == cell_id
                and "final"
                in coverage_budgets[(candidate_cell, row_id, candidate_method)]
            }
        complete_by_method: dict[str, set[str]] = defaultdict(set)
        key_coverage_by_method: dict[str, float] = {}
        for method in methods:
            observed_keys = 0
            expected_keys = 0
            for row_id in sorted(coverage_universe):
                trio = (cell_id, row_id, method)
                reference_trio = (cell_id, row_id, coverage_reference_method)
                final_length = coverage_lengths.get(
                    trio, coverage_lengths.get(reference_trio, 1)
                )
                expected = _expected_budgets(final_length)
                observed = coverage_budgets.get(trio, set())
                expected_keys += len(expected)
                observed_keys += len(expected.intersection(observed))
                if expected.issubset(observed):
                    complete_by_method[method].add(row_id)
            key_coverage_by_method[method] = (
                observed_keys / expected_keys if expected_keys else 0.0
            )

        direct_universe = set(direct_reference_traces.get(cell_id, set()))
        direct_complete_by_method: dict[str, set[str]] = defaultdict(set)
        for method in methods:
            for row_id in sorted(direct_universe):
                final_length = trace_meta[(cell_id, row_id)][0]
                expected = _expected_budgets(final_length)
                if expected.issubset(
                    direct_observed_budgets.get((cell_id, row_id, method), set())
                ):
                    direct_complete_by_method[method].add(row_id)

        required_available = [
            direct_complete_by_method.get(method, set()) for method in required_methods
        ]
        direct_ids = set(direct_universe)
        for available in required_available:
            direct_ids.intersection_update(available)
        ordered_ids = sorted(direct_ids)
        order_hash = ordered_id_sha256(ordered_ids) if ordered_ids else None
        eligible_optional = [
            method
            for method in methods
            if method not in required_methods
            and direct_ids.issubset(direct_complete_by_method[method])
        ]
        included_methods = list(required_methods) + eligible_optional if direct_ids else []
        population_id = (
            f"prefix_historical_common_v1::{cell_id}::{order_hash[:16]}"
            if order_hash is not None
            else f"prefix_historical_common_v1::{cell_id}::ineligible"
        )

        for method in included_methods:
            for row_id in ordered_ids:
                final_length = trace_meta[(cell_id, row_id)][0]
                for budget in sorted(_expected_budgets(final_length), key=_budget_sort_key):
                    source_record = direct_index[(cell_id, row_id, method, budget)]
                    rewritten = dict(source_record)
                    rewritten["population_id"] = population_id
                    direct_records.append(validate_comparison_record(rewritten))

        coverage_rows.append(
            {
                "cell_id": cell_id,
                "reference_method": reference_method,
                "coverage_reference_method": coverage_reference_method,
                "reference_traces": len(coverage_universe),
                "direct_reference_traces": len(direct_universe),
                "required_methods": list(required_methods),
                "complete_traces_by_method": {
                    method: len(complete_by_method.get(method, set()))
                    for method in methods
                },
                "complete_trace_coverage_by_method": {
                    method: (
                        len(complete_by_method.get(method, set()))
                        / len(coverage_universe)
                        if coverage_universe
                        else 0.0
                    )
                    for method in methods
                },
                "key_coverage_by_method": key_coverage_by_method,
                "direct_complete_traces_by_method": {
                    method: len(direct_complete_by_method.get(method, set()))
                    for method in methods
                },
                "context_only_method_ids": [
                    method
                    for method in methods
                    if complete_by_method.get(method)
                    and not direct_complete_by_method.get(method)
                ],
                "pending_registered_cpu_replay": [
                    method
                    for method in (IU28_METHOD_ID, HISTORICAL_DEEPCONF_METHOD_ID)
                    if complete_by_method.get(method)
                    and not direct_complete_by_method.get(method)
                ],
                "direct_common_traces": len(ordered_ids),
                "direct_table_eligible": bool(ordered_ids),
                "direct_method_ids": included_methods,
                "ordered_id_sha256": order_hash,
                "missing_required_methods": [
                    method
                    for method in required_methods
                    if not direct_complete_by_method.get(method)
                ],
            }
        )
        if ordered_ids:
            final_lengths = [trace_meta[(cell_id, row_id)][0] for row_id in ordered_ids]
            eligible_populations = {
                f"budget_{budget}": make_eligible_population(
                    [
                        row_id
                        for row_id, final_length in zip(ordered_ids, final_lengths)
                        if final_length > budget
                    ],
                    rule=f"final_length_strictly_greater_than_{budget}",
                )
                for budget in PREFIX_BUDGETS
            }
            eligible_populations["complete_six_budget_warning"] = (
                make_eligible_population(
                    [
                        row_id
                        for row_id, final_length in zip(ordered_ids, final_lengths)
                        if final_length > max(PREFIX_BUDGETS)
                    ],
                    rule=(
                        "complete_scores_at_16_32_64_128_256_512_and_"
                        "final_length_strictly_greater_than_512"
                    ),
                )
            )
            populations.append(
                {
                    "schema": "prefix_direct_population_v1",
                    "population_id": population_id,
                    "cell_id": cell_id,
                    "ordered_row_ids": ordered_ids,
                    "ordered_id_sha256": order_hash,
                    "n_rows": len(ordered_ids),
                    "population_construction": (
                        "intersection_of_registered_same_telemetry_artifact_availability"
                    ),
                    "outcome_filtering": False,
                    "required_methods": list(required_methods),
                    "included_methods": included_methods,
                    "final_lengths": final_lengths,
                    "eligible_populations": eligible_populations,
                }
            )

    direct_records.sort(
        key=lambda row: (
            row["cell_id"],
            row["row_id"],
            row["method_id"],
            _budget_sort_key(row["budget"]),
        )
    )
    return {
        "records": tuple(direct_records),
        "populations": populations,
        "coverage": coverage_rows,
        "audit": {
            "schema": "prefix_join_audit_v1",
            "prefix_lane_revision": PREFIX_LANE_REVISION,
            "source_records": len(normalized),
            "context_only_source_records": sum(
                record.get("direct_eligible", True) is False for record in normalized
            ),
            "direct_records": len(direct_records),
            "reference_method": reference_method,
            "coverage_reference_method": coverage_reference_method,
            "required_methods": list(required_methods),
            "n_reference_cells": len(all_cells),
            "n_direct_reference_cells": len(direct_reference_traces),
            "n_direct_cells": sum(row["direct_table_eligible"] for row in coverage_rows),
            "population_construction": (
                "intersection_of_registered_same_telemetry_artifact_availability"
            ),
            "question_id_equality_implies_trace_equality": False,
            "outcome_filtering": False,
            "positional_fallback": False,
            "strict_length_gt_budget": True,
        },
    }


def summarize_prefix_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute per-cell error-positive metrics with same-cohort final anchors.

    This function delegates AUROC/AP definitions to the package's one frozen evaluator.
    It never pools heterogeneous cells or separately fitted folds.
    """

    # Lazy import keeps read-only schema/inventory tooling usable in a minimal Python
    # environment; all actual scoring still goes through the shared frozen evaluator.
    from . import evaluator as common_evaluator

    normalized = [validate_comparison_record(record) for record in records]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        groups[(record["population_id"], record["cell_id"], record["method_id"])].append(record)

    metric_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (population_id, cell_id, method_id), values in sorted(groups.items()):
        final_by_id = {
            row["row_id"]: row for row in values if row["budget"] == "final"
        }
        if not final_by_id:
            raise PrefixIntegrityError(f"{cell_id}/{method_id} has no final score sidecars")
        budget_auroc: dict[int, float] = {}
        recoveries: dict[int, float] = {}
        for budget in PREFIX_BUDGETS:
            selected = [row for row in values if row["budget"] == budget]
            if not selected:
                continue
            if any(int(row["final_length"]) <= budget for row in selected):
                raise PrefixIntegrityError(
                    f"metric input violates strict length gate for {cell_id}/{method_id}@{budget}"
                )
            row_ids = [row["row_id"] for row in selected]
            if len(row_ids) != len(set(row_ids)):
                raise PrefixIntegrityError(
                    f"duplicate metric row IDs for {cell_id}/{method_id}@{budget}"
                )
            missing_final = sorted(set(row_ids).difference(final_by_id))
            if missing_final:
                raise PrefixIntegrityError(
                    f"missing same-cohort final scores: {missing_final[:5]}"
                )
            labels = [row["label"] for row in selected]
            scores = [row["continuous_score"] for row in selected]
            final_scores = [final_by_id[row_id]["continuous_score"] for row_id in row_ids]
            metrics = common_evaluator.detection_metrics(labels, scores)
            final_metrics = common_evaluator.detection_metrics(labels, final_scores)
            recovered = common_evaluator.recovered_above_chance_signal(
                metrics["auroc"], final_metrics["auroc"]
            )
            metric_rows.append(
                {
                    "population_id": population_id,
                    "cell_id": cell_id,
                    "method_id": method_id,
                    "budget": budget,
                    **metrics,
                    "final_auroc_same_cohort": final_metrics["auroc"],
                    "recovered_above_chance_signal": recovered,
                    "ordered_id_sha256": ordered_id_sha256(sorted(row_ids)),
                    "eligibility": "final_length_strictly_greater_than_budget",
                }
            )
            budget_auroc[budget] = metrics["auroc"]
            recoveries[budget] = recovered

        primary = [budget_auroc.get(64), budget_auroc.get(128)]
        primary_mean = (
            float(sum(primary) / 2.0)
            if all(value is not None and math.isfinite(value) for value in primary)
            else float("nan")
        )
        earliest = next(
            (
                budget
                for budget in PREFIX_BUDGETS
                if budget in recoveries
                and math.isfinite(recoveries[budget])
                and recoveries[budget] >= 0.95
            ),
            None,
        )
        full_values = [final_by_id[row_id] for row_id in sorted(final_by_id)]
        final_metrics = common_evaluator.detection_metrics(
            [row["label"] for row in full_values],
            [row["continuous_score"] for row in full_values],
        )
        summaries.append(
            {
                "population_id": population_id,
                "cell_id": cell_id,
                "method_id": method_id,
                "n_traces": len(full_values),
                "primary_mean_auroc_64_128": primary_mean,
                "earliest_budget_reaching_95pct_final_signal": earliest,
                "final_auroc": final_metrics["auroc"],
                "final_error_auprc": final_metrics["error_auprc"],
                "budgets_present": sorted(budget_auroc),
            }
        )
    return {"per_budget": metric_rows, "per_cell_method": summaries}


def build_warning_inputs(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Emit complete six-budget paths for external foldwise warning calibration.

    Only traces with ``final_length > 512`` can have the registered complete horizon.
    Incomplete paths are reported and excluded; no score is carried forward or padded.
    """

    normalized = [validate_comparison_record(record) for record in records]
    paths: dict[tuple[str, str, str, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    final_lengths: dict[tuple[str, str, str, str], int] = {}
    labels: dict[tuple[str, str, str, str], int] = {}
    exemplars: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in normalized:
        if row["budget"] == "final":
            continue
        key = (row["population_id"], row["cell_id"], row["method_id"], row["row_id"])
        budget = int(row["budget"])
        if budget in paths[key]:
            raise PrefixIntegrityError(f"duplicate warning-path budget for {key!r}@{budget}")
        paths[key][budget] = row
        final_lengths[key] = int(row["final_length"])
        labels[key] = _binary(row["label"], name="label")
        exemplars[key] = row

    output: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    required = set(PREFIX_BUDGETS)
    for key in sorted(paths):
        present = set(paths[key])
        if final_lengths[key] <= max(PREFIX_BUDGETS) or present != required:
            incomplete.append(
                {
                    "population_id": key[0],
                    "cell_id": key[1],
                    "method_id": key[2],
                    "row_id": key[3],
                    "final_length": final_lengths[key],
                    "missing_budgets": sorted(required.difference(present)),
                }
            )
            continue
        example = exemplars[key]
        output.append(
            {
                "population_id": key[0],
                "cell_id": key[1],
                "method_id": key[2],
                "row_id": key[3],
                "group_id": example["group_id"],
                "family": example.get("family"),
                "label": labels[key],
                "fold": example.get("fold"),
                "final_length": final_lengths[key],
                "score_path": {
                    budget: paths[key][budget]["continuous_score"]
                    for budget in PREFIX_BUDGETS
                },
                "path_sha256": canonical_sha256(
                    [paths[key][budget]["continuous_score"] for budget in PREFIX_BUDGETS]
                ),
                "calibration_unit": "trace_level_ever_warning",
            }
        )
    return {
        "rows": output,
        "audit": {
            "schema": "prefix_warning_input_audit_v1",
            "budgets": list(PREFIX_BUDGETS),
            "complete_paths": len(output),
            "incomplete_or_short_paths": len(incomplete),
            "incomplete": incomplete,
            "eligibility": "final_length_strictly_greater_than_512",
            "thresholds_fitted": False,
            "required_calibration": "maximum_over_complete_six_budget_correct_trace_path",
        },
    }


def canonical_s2_prefix_id(
    dataset_revision: str,
    dataset: str,
    question_id: str,
    model: str,
    arm: str = "cot|central",
) -> str:
    """Return the registered S2 trace key without inventing a positional ID."""

    parts = (dataset_revision, dataset, question_id, model, arm)
    if any(not isinstance(part, str) or not part for part in parts):
        raise PrefixIntegrityError("S2 canonical ID components must be non-empty strings")
    if any("::" in part for part in parts):
        raise PrefixIntegrityError("S2 canonical ID components contain reserved delimiter")
    return "::".join(parts)


def _resolve_dotted_field(row: Mapping[str, Any], dotted: str) -> Any:
    value: Any = row
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _s2_stream_validation_error(value: Any, field: str, token_count: int) -> str | None:
    """Return a compact contract error for one S2 token stream, else ``None``."""

    import numpy as np

    if field == "gen_token_ids":
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
            return "not_a_token_sequence"
        if len(value) != token_count or token_count < 1:
            return "token_count_mismatch"
        return None
    if field == "top_k_logprobs":
        if not isinstance(value, Mapping):
            return "missing_top_k_mapping"
        ids, logprobs = value.get("ids"), value.get("logprobs")
        if ids is None or logprobs is None:
            return "top_k_ids_or_logprobs_missing"
        ids_array = np.asarray(ids)
        logprob_array = np.asarray(logprobs, dtype=float)
        if (
            ids_array.ndim != 2
            or logprob_array.ndim != 2
            or ids_array.shape != logprob_array.shape
            or ids_array.shape[0] != token_count
            or ids_array.shape[1] < 1
        ):
            return "top_k_shape_or_alignment_mismatch"
        if not np.isfinite(logprob_array).all():
            return "top_k_logprobs_nonfinite"
        return None
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return "not_a_numeric_stream"
    if values.ndim != 1 or len(values) != token_count:
        return "stream_shape_or_alignment_mismatch"
    if not np.isfinite(values).all():
        return "stream_nonfinite"
    return None


_S2_INFERRED_FROZEN_SEMANTICS = {
    "gen_token_ids": S2_FROZEN_INPUT_SEMANTICS["gen_token_ids"],
    "token_entropies": S2_FROZEN_INPUT_SEMANTICS["token_entropies"],
    "channels.sampled_entropy": S2_FROZEN_INPUT_SEMANTICS["token_entropies"],
    "token_spilled_energies": S2_FROZEN_INPUT_SEMANTICS[
        "token_spilled_energies"
    ],
    "channels.sampled_spilled_energy": S2_FROZEN_INPUT_SEMANTICS[
        "token_spilled_energies"
    ],
    "token_logsumexp": S2_FROZEN_INPUT_SEMANTICS["token_logsumexp"],
    "channels.raw_logsumexp": S2_FROZEN_INPUT_SEMANTICS["token_logsumexp"],
    "top_k_logprobs": S2_FROZEN_INPUT_SEMANTICS["top_k_logprobs"],
    "sampled_top_k_logprobs": S2_FROZEN_INPUT_SEMANTICS["top_k_logprobs"],
}

_S2_RAW_SUBSTITUTE_SEMANTICS = {
    "token_entropies": (
        ("channels.raw_entropy", "raw_full_vocabulary_entropy"),
    ),
    "token_spilled_energies": (
        ("channels.spilled_energy", "raw_sampled_token_negative_logprob"),
    ),
    "top_k_logprobs": (
        ("raw_top_k_logprobs", "raw_top50_logprobs"),
    ),
}


def audit_s2_cot_telemetry(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_revision: str,
    dataset: str,
    model: str,
    arm: str = "cot",
    setting: str = "central",
    dedicated_required_fields: Sequence[str] = (),
    method_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    observed_stream_semantics: Mapping[str, str] | None = None,
    ordered_question_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evidence-driven gate for S2 COT transfer; never substitute raw for legacy data.

    ``method_bindings`` must explicitly attest both ``anchor_verified`` and
    ``target_binding_verified`` for a method.  Even then the method stays ineligible
    unless every trace carries its exact frozen input semantics with finite, aligned
    values.  This makes the gate capable of passing for a future exact sidecar while
    failing the acquired raw-only S2 payload for concrete, reported reasons.
    """

    arm_key = f"{arm}|{setting}"
    method_bindings = method_bindings or {}
    observed_stream_semantics = {
        **_S2_INFERRED_FROZEN_SEMANTICS,
        **(observed_stream_semantics or {}),
    }
    seen: set[str] = set()
    question_by_row_id: dict[str, str] = {}
    missing_rows: list[dict[str, Any]] = []
    accepted_ids: list[str] = []
    budget_eligible = {budget: 0 for budget in PREFIX_BUDGETS}
    resolved_alias_counts: dict[str, dict[str, int]] = {
        field: defaultdict(int) for field in UNIFIED28_REQUIRED_TELEMETRY_FIELDS
    }
    exact_field_status: dict[str, dict[str, Any]] = {
        field: {
            "valid_rows": 0,
            "missing_rows": 0,
            "invalid_rows": 0,
            "semantic_mismatch_rows": 0,
            "resolved_alias_counts": defaultdict(int),
            "error_counts": defaultdict(int),
            "raw_substitute_counts": defaultdict(int),
        }
        for field in S2_FROZEN_INPUT_ALIASES
    }
    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PrefixIntegrityError(f"S2 row {source_index} is not a mapping")
        required_identity = ("question_id", "arm", "setting_label")
        missing_identity = [field for field in required_identity if field not in row]
        if missing_identity:
            raise PrefixIntegrityError(
                f"S2 row {source_index} lacks {missing_identity}; positional fallback forbidden"
            )
        if str(row["arm"]) != arm or str(row["setting_label"]) != setting:
            continue
        source_question_id = row["question_id"]
        if (
            source_question_id is None
            or isinstance(source_question_id, (bool, bytes, Mapping))
            or (
                isinstance(source_question_id, Sequence)
                and not isinstance(source_question_id, str)
            )
        ):
            raise PrefixIntegrityError(
                f"S2 row {source_index} has a non-scalar question_id"
            )
        question_id = str(source_question_id)
        if not question_id:
            raise PrefixIntegrityError(f"S2 row {source_index} has an empty question_id")
        row_id = canonical_s2_prefix_id(
            dataset_revision, dataset, question_id, model, arm_key
        )
        if row_id in seen:
            raise PrefixIntegrityError(f"duplicate S2 COT ID {row_id}")
        seen.add(row_id)
        question_by_row_id[row_id] = question_id
        token_ids = _resolve_dotted_field(row, "gen_token_ids")
        if (
            isinstance(token_ids, (str, bytes, Mapping))
            or not isinstance(token_ids, Sequence)
            or not token_ids
        ):
            token_count = 0
        else:
            token_count = len(token_ids)
        for budget in PREFIX_BUDGETS:
            if token_count > budget:
                budget_eligible[budget] += 1
        missing: list[str] = []
        for field in UNIFIED28_REQUIRED_TELEMETRY_FIELDS:
            aliases = S2_UNIFIED28_TELEMETRY_ALIASES[field]
            resolved = next(
                (alias for alias in aliases if _resolve_dotted_field(row, alias) is not None),
                None,
            )
            if resolved is None:
                missing.append(field)
            else:
                resolved_alias_counts[field][resolved] += 1
        for field, aliases in S2_FROZEN_INPUT_ALIASES.items():
            status = exact_field_status[field]
            resolved = next(
                (alias for alias in aliases if _resolve_dotted_field(row, alias) is not None),
                None,
            )
            if resolved is None:
                status["missing_rows"] += 1
                for substitute_alias, substitute_semantics in (
                    _S2_RAW_SUBSTITUTE_SEMANTICS.get(field, ())
                ):
                    if _resolve_dotted_field(row, substitute_alias) is not None:
                        status["raw_substitute_counts"][
                            f"{substitute_alias}::{substitute_semantics}"
                        ] += 1
                continue
            status["resolved_alias_counts"][resolved] += 1
            expected_semantics = S2_FROZEN_INPUT_SEMANTICS[field]
            actual_semantics = observed_stream_semantics.get(resolved)
            if actual_semantics != expected_semantics:
                status["semantic_mismatch_rows"] += 1
                status["error_counts"][
                    f"semantic_mismatch::{actual_semantics or 'undeclared'}"
                ] += 1
                continue
            error = _s2_stream_validation_error(
                _resolve_dotted_field(row, resolved), field, token_count
            )
            if error is not None:
                status["invalid_rows"] += 1
                status["error_counts"][error] += 1
            else:
                status["valid_rows"] += 1
        missing.extend(
            field
            for field in dedicated_required_fields
            if _resolve_dotted_field(row, field) is None
        )
        if missing:
            missing_rows.append({"row_id": row_id, "missing_fields": missing})
        else:
            accepted_ids.append(row_id)

    if ordered_question_ids is not None:
        declared = [str(value) for value in ordered_question_ids]
        if len(declared) != len(set(declared)):
            raise PrefixIntegrityError("S2 ordered question IDs contain duplicates")
        observed_questions = set(question_by_row_id.values())
        if set(declared) != observed_questions:
            missing = sorted(set(declared).difference(observed_questions))
            extra = sorted(observed_questions.difference(declared))
            raise PrefixIntegrityError(
                f"S2 ordered question IDs disagree: missing={missing[:5]}, extra={extra[:5]}"
            )
        id_by_question = {question: row_id for row_id, question in question_by_row_id.items()}
        ordered_ids = [id_by_question[question] for question in declared]
    else:
        ordered_ids = sorted(seen)
    accepted_set = set(accepted_ids)
    accepted_ids = [row_id for row_id in ordered_ids if row_id in accepted_set]

    exact_field_audit: dict[str, dict[str, Any]] = {}
    for field, status in exact_field_status.items():
        exact_field_audit[field] = {
            "expected_semantics": S2_FROZEN_INPUT_SEMANTICS[field],
            "candidate_aliases": list(S2_FROZEN_INPUT_ALIASES[field]),
            "valid_rows": status["valid_rows"],
            "missing_rows": status["missing_rows"],
            "invalid_rows": status["invalid_rows"],
            "semantic_mismatch_rows": status["semantic_mismatch_rows"],
            "complete": bool(seen) and status["valid_rows"] == len(seen),
            "resolved_alias_counts": dict(
                sorted(status["resolved_alias_counts"].items())
            ),
            "error_counts": dict(sorted(status["error_counts"].items())),
            "raw_substitute_counts": dict(
                sorted(status["raw_substitute_counts"].items())
            ),
        }

    method_gates: dict[str, dict[str, Any]] = {}
    for method_id, required_fields in S2_PREFIX_METHOD_INPUTS.items():
        binding = method_bindings.get(method_id, {})
        input_blockers = [
            {
                "field": field,
                "valid_rows": exact_field_audit[field]["valid_rows"],
                "missing_rows": exact_field_audit[field]["missing_rows"],
                "invalid_rows": exact_field_audit[field]["invalid_rows"],
                "semantic_mismatch_rows": exact_field_audit[field][
                    "semantic_mismatch_rows"
                ],
                "raw_substitute_counts": exact_field_audit[field][
                    "raw_substitute_counts"
                ],
            }
            for field in required_fields
            if not exact_field_audit[field]["complete"]
        ]
        anchor_verified = binding.get("anchor_verified") is True
        target_binding_verified = binding.get("target_binding_verified") is True
        blockers: list[Any] = list(input_blockers)
        if not anchor_verified:
            blockers.append(
                binding.get(
                    "anchor_blocker",
                    "exact frozen parameter/provenance anchor not supplied",
                )
            )
        if not target_binding_verified:
            blockers.append(
                binding.get(
                    "target_binding_blocker",
                    "exact frozen source-fit to S2 target binding not supplied",
                )
            )
        method_gates[method_id] = {
            "required_fields": list(required_fields),
            "input_contract_passed": not input_blockers,
            "anchor_verified": anchor_verified,
            "target_binding_verified": target_binding_verified,
            "eligible": bool(seen) and not blockers,
            "blockers": blockers,
            "binding": dict(binding),
        }

    prefix_required = (UNIFIED28_METHOD_ID, IU28_METHOD_ID, STEP272_METHOD_ID)
    global_required = (UNIFIED28_METHOD_ID, CLASSIC_GLOBAL_METHOD_ID)
    prefix_join = all(
        method_gates[method]["anchor_verified"]
        and method_gates[method]["target_binding_verified"]
        for method in prefix_required
    )
    global_join = all(
        method_gates[method]["anchor_verified"]
        and method_gates[method]["target_binding_verified"]
        for method in global_required
    )
    return {
        "schema": "s2_prefix_telemetry_audit_v1",
        "dataset_revision": dataset_revision,
        "dataset": dataset,
        "model": model,
        "arm": arm_key,
        "cot_rows": len(seen),
        "raw_telemetry_complete_rows": len(accepted_ids),
        "raw_telemetry_coverage": len(accepted_ids) / len(seen) if seen else 0.0,
        "ordered_id_sha256": ordered_id_sha256(ordered_ids) if ordered_ids else None,
        "ordered_group_id_sha256": (
            ordered_id_sha256(
                [
                    f"{dataset_revision}::{dataset}::{question_by_row_id[row_id]}"
                    for row_id in ordered_ids
                ]
            )
            if ordered_ids
            else None
        ),
        "ordered_ids_source": (
            "declared_manifest_question_order"
            if ordered_question_ids is not None
            else "canonical_lexical_order"
        ),
        "missing_rows": missing_rows,
        "unified28_required_fields": list(UNIFIED28_REQUIRED_TELEMETRY_FIELDS),
        "unified28_schema_aliases": {
            field: list(aliases)
            for field, aliases in S2_UNIFIED28_TELEMETRY_ALIASES.items()
        },
        "resolved_alias_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in resolved_alias_counts.items()
        },
        "dedicated_required_fields": list(dedicated_required_fields),
        "raw_telemetry_gate_passed": bool(seen) and not missing_rows,
        "strict_budget_eligible_rows": {
            str(budget): budget_eligible[budget] for budget in PREFIX_BUDGETS
        },
        "strict_length_gt_budget": True,
        "frozen_input_contract": exact_field_audit,
        "method_gates": method_gates,
        "frozen_model_join_gate_passed": prefix_join,
        "prefix_scoring_eligible": all(
            method_gates[method]["eligible"] for method in prefix_required
        ),
        "global_model_join_gate_passed": global_join,
        "global_scoring_eligible": all(
            method_gates[method]["eligible"] for method in global_required
        ),
        "next_gate": (
            "provide exact legacy post-warper telemetry plus verified frozen anchors "
            "and target bindings; raw-distribution substitution is forbidden"
        ),
        "scores_materialized": False,
        "outcomes_used_for_gate": False,
        "positional_fallback": False,
    }


__all__ = [
    "PREFIX_LANE_REVISION",
    "PREFIX_BUDGETS",
    "SELECTED_STEP272_ARCHITECTURE",
    "REGISTERED_LLAMA_PREFIX_TELEMETRY",
    "HISTORICAL_PREFIX_TELEMETRY",
    "FROZEN_PREFIX_REPLAY_REVISION",
    "UNIFIED28_METHOD_ID",
    "IU28_METHOD_ID",
    "STEP272_METHOD_ID",
    "MEAN_ENTROPY_METHOD_ID",
    "MAX_ENTROPY_METHOD_ID",
    "HISTORICAL_DEEPCONF_METHOD_ID",
    "CLASSIC_GLOBAL_METHOD_ID",
    "DIRECT_REQUIRED_METHODS",
    "UNIFIED28_REQUIRED_TELEMETRY_FIELDS",
    "S2_UNIFIED28_TELEMETRY_ALIASES",
    "S2_FROZEN_INPUT_ALIASES",
    "S2_FROZEN_INPUT_SEMANTICS",
    "S2_PREFIX_METHOD_INPUTS",
    "PrefixIntegrityError",
    "load_historical_prefix_scores",
    "load_unified28_prefix_records",
    "load_step272_prefix_records",
    "build_entropy_prefix_records",
    "replay_frozen_prefix_incumbents",
    "assemble_historical_common_panel",
    "summarize_prefix_metrics",
    "build_warning_inputs",
    "canonical_s2_prefix_id",
    "audit_s2_cot_telemetry",
]
