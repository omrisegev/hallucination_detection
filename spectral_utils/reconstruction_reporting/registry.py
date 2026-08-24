"""Research registries used by the reconstruction benchmark report.

The registry separates algorithm identity, a concrete version, a task adapter,
and their runnable system.  This prevents task-specific adapters from being
smuggled into a method name and lets the report explain every acronym before
showing results.  Dataset, population, cell, and slice IDs are likewise
method-independent so one query can compare compatible systems exactly.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from .schemas import (
    ACCESS_TIERS,
    RANKABLE_STATUSES,
    SchemaError,
    canonical_sha256,
    derive_cohort_id,
    derive_comparison_group_id,
)


REGISTRY_SCHEMA = "reconstruction_registry_v1"
DONOR_REGIMES = (
    "within_cell_fully_unsupervised",
    "donor_unsupervised",
    "donor_label_selection",
    "not_applicable",
)
METHOD_ROLES = (
    "primary",
    "secondary",
    "control",
    "supervised_ceiling",
    "label_informed_reference",
    "published_context",
    "historical_only",
)
METHOD_STAGES = (
    "canonical",
    "new_unrun_ablation",
    "retrospective",
    "validation_blocked",
    "historical",
)
MARKERS = ("circle", "square", "triangle", "diamond", "cross", "plus", "star", "hexagon")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{where} must be a mapping")
    return value


def _fields(value: Mapping[str, Any], names: Sequence[str], *, where: str) -> None:
    missing = [name for name in names if name not in value]
    if missing:
        raise SchemaError(f"{where} missing required fields: {missing}")


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be text")
    if not allow_empty and not value.strip():
        raise SchemaError(f"{field} must be non-empty text")
    return value.strip() if not allow_empty else value


def _id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SchemaError(f"{field} must be a non-empty, trimmed ID")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SchemaError(f"{field} must be a trimmed ID without control characters")
    return value


def _choice(value: Any, choices: Sequence[str], *, field: str) -> str:
    value = _id(value, field=field)
    if value not in choices:
        raise SchemaError(f"{field} must be one of {tuple(choices)!r}; got {value!r}")
    return value


def _string_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{field} must be a sequence of strings")
    result = [_text(item, field=f"{field}[]") for item in value]
    if not allow_empty and not result:
        raise SchemaError(f"{field} must not be empty")
    if len(result) != len(set(result)):
        raise SchemaError(f"{field} must not repeat values")
    return result


def _unique(records: Sequence[Mapping[str, Any]], field: str, *, where: str) -> None:
    values = [record[field] for record in records]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise SchemaError(f"{where} repeats {field}: {duplicates}")


def make_system_id(method_version_id: str, adapter_id: str) -> str:
    return f"{_id(method_version_id, field='method_version_id')}::{_id(adapter_id, field='adapter_id')}"


METHOD_REQUIRED_FIELDS = (
    "method_id",
    "display_name",
    "acronym_expansion",
    "family_id",
    "plain_summary",
    "input_operation_output",
    "formula",
    "formula_terms",
    "origin",
    "development_history",
    "inputs",
    "access_tier",
    "supervision",
    "donor_regime",
    "model_passes",
    "assumptions",
    "fallbacks",
    "limitations",
    "role",
    "research_stage",
    "references",
    "style",
)


def validate_method_card(card: Mapping[str, Any]) -> dict[str, Any]:
    card = _mapping(card, where="method card")
    _fields(card, METHOD_REQUIRED_FIELDS, where="method card")
    output = dict(card)
    for field in (
        "method_id",
        "display_name",
        "family_id",
        "plain_summary",
        "input_operation_output",
        "formula",
        "development_history",
        "inputs",
        "supervision",
    ):
        output[field] = _text(card[field], field=f"method.{field}")
    # An acronym may legitimately equal the display name (for a descriptive
    # baseline), but the field must still explain that fact explicitly.
    output["acronym_expansion"] = _text(card["acronym_expansion"], field="method.acronym_expansion")
    output["access_tier"] = _choice(card["access_tier"], ACCESS_TIERS, field="method.access_tier")
    output["donor_regime"] = _choice(card["donor_regime"], DONOR_REGIMES, field="method.donor_regime")
    output["role"] = _choice(card["role"], METHOD_ROLES, field="method.role")
    output["research_stage"] = _choice(card["research_stage"], METHOD_STAGES, field="method.research_stage")
    if type(card["model_passes"]) is bool or not isinstance(card["model_passes"], int) or card["model_passes"] < 0:
        raise SchemaError("method.model_passes must be a non-negative integer")
    terms = _mapping(card["formula_terms"], where="method.formula_terms")
    if not terms:
        raise SchemaError("method.formula_terms must explain at least one symbol")
    output["formula_terms"] = {
        _text(symbol, field="method.formula_terms symbol"): _text(description, field=f"method.formula_terms[{symbol!r}]")
        for symbol, description in sorted(terms.items(), key=lambda item: str(item[0]))
    }
    for field in ("assumptions", "fallbacks", "limitations"):
        output[field] = _string_list(card[field], field=f"method.{field}")
    origin = _mapping(card["origin"], where="method.origin")
    _fields(origin, ("kind", "title", "year", "relationship"), where="method.origin")
    if type(origin["year"]) is bool or not isinstance(origin["year"], int):
        raise SchemaError("method.origin.year must be an integer")
    output["origin"] = {
        "kind": _text(origin["kind"], field="method.origin.kind"),
        "title": _text(origin["title"], field="method.origin.title"),
        "year": origin["year"],
        "relationship": _text(origin["relationship"], field="method.origin.relationship"),
    }
    references = card["references"]
    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise SchemaError("method.references must be a sequence")
    normalized_references = []
    for index, reference in enumerate(references):
        reference = _mapping(reference, where=f"method.references[{index}]")
        _fields(reference, ("title", "citation", "url"), where=f"method.references[{index}]")
        normalized_references.append(
            {
                "title": _text(reference["title"], field="reference.title"),
                "citation": _text(reference["citation"], field="reference.citation"),
                "url": _text(reference["url"], field="reference.url", allow_empty=True),
            }
        )
    output["references"] = normalized_references
    style = _mapping(card["style"], where="method.style")
    _fields(style, ("color", "marker"), where="method.style")
    color = _text(style["color"], field="method.style.color")
    if _HEX_COLOR_RE.fullmatch(color) is None:
        raise SchemaError("method.style.color must be a six-digit hex color")
    output["style"] = {
        "color": color.lower(),
        "marker": _choice(style["marker"], MARKERS, field="method.style.marker"),
    }
    return output


def validate_method_version(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="method version")
    required = (
        "method_version_id",
        "method_id",
        "version_label",
        "definition_sha256",
        "formula",
        "fixed_parameters",
        "source_paths",
        "feature_contract_id",
        "research_stage",
    )
    _fields(record, required, where="method version")
    output = dict(record)
    for field in ("method_version_id", "method_id", "version_label", "definition_sha256", "formula", "feature_contract_id"):
        output[field] = _text(record[field], field=f"method_version.{field}")
    if not re.fullmatch(r"[0-9a-f]{64}", output["definition_sha256"]):
        raise SchemaError("method_version.definition_sha256 must be a lowercase SHA-256")
    if not isinstance(record["fixed_parameters"], Mapping):
        raise SchemaError("method_version.fixed_parameters must be a mapping")
    output["fixed_parameters"] = dict(record["fixed_parameters"])
    output["source_paths"] = _string_list(record["source_paths"], field="method_version.source_paths")
    output["research_stage"] = _choice(record["research_stage"], METHOD_STAGES, field="method_version.research_stage")
    return output


def validate_adapter(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="adapter")
    required = (
        "adapter_id",
        "display_name",
        "task_id",
        "plain_summary",
        "input_unit",
        "output_unit",
        "definition_sha256",
        "source_paths",
        "limitations",
    )
    _fields(record, required, where="adapter")
    output = dict(record)
    for field in required:
        if field == "source_paths":
            output[field] = _string_list(record[field], field="adapter.source_paths")
        elif field == "limitations":
            output[field] = _string_list(record[field], field="adapter.limitations")
        else:
            output[field] = _text(record[field], field=f"adapter.{field}")
    if not re.fullmatch(r"[0-9a-f]{64}", output["definition_sha256"]):
        raise SchemaError("adapter.definition_sha256 must be a lowercase SHA-256")
    return output


def validate_system(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="system")
    required = (
        "system_id",
        "method_version_id",
        "adapter_id",
        "access_contract_id",
        "display_name",
        "enabled",
    )
    _fields(record, required, where="system")
    output = dict(record)
    for field in (
        "system_id",
        "method_version_id",
        "adapter_id",
        "access_contract_id",
        "display_name",
    ):
        output[field] = _text(record[field], field=f"system.{field}")
    if type(record["enabled"]) is not bool:
        raise SchemaError("system.enabled must be boolean")
    expected = make_system_id(output["method_version_id"], output["adapter_id"])
    if output["system_id"] != expected:
        raise SchemaError(f"system_id must be exactly {expected!r}")
    return output


def validate_task(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="task")
    required = (
        "task_id",
        "display_name",
        "description",
        "prediction_unit",
        "primary_metric_id",
        "positive_class",
        "bootstrap_unit",
    )
    _fields(record, required, where="task")
    return {**record, **{field: _text(record[field], field=f"task.{field}") for field in required}}


def validate_dataset_card(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="dataset")
    required = (
        "dataset_id",
        "task_id",
        "display_name",
        "description",
        "prediction_unit",
        "label_definition",
        "positive_class",
        "inclusion_reason",
        "dataset_family",
        "revision",
        "limitations",
        "source",
    )
    _fields(record, required, where="dataset")
    output = dict(record)
    for field in required:
        if field == "limitations":
            output[field] = _string_list(record[field], field="dataset.limitations", allow_empty=True)
        elif field == "source":
            source = _mapping(record[field], where="dataset.source")
            _fields(source, ("title", "citation", "url"), where="dataset.source")
            output[field] = {
                "title": _text(source["title"], field="dataset.source.title"),
                "citation": _text(source["citation"], field="dataset.source.citation"),
                "url": _text(source["url"], field="dataset.source.url", allow_empty=True),
            }
        else:
            output[field] = _text(record[field], field=f"dataset.{field}")
    return output


def validate_access_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="access contract")
    required = (
        "access_contract_id",
        "access_tier",
        "input_type",
        "supervision",
        "model_passes_per_question",
        "traces_per_question",
        "donor_regime",
    )
    _fields(record, required, where="access contract")
    output = dict(record)
    for field in ("access_contract_id", "input_type", "supervision"):
        output[field] = _text(record[field], field=f"access_contract.{field}")
    output["access_tier"] = _choice(record["access_tier"], ACCESS_TIERS, field="access_contract.access_tier")
    output["donor_regime"] = _choice(record["donor_regime"], DONOR_REGIMES, field="access_contract.donor_regime")
    for field in ("model_passes_per_question", "traces_per_question"):
        if type(record[field]) is bool or not isinstance(record[field], int) or record[field] < 0:
            raise SchemaError(f"access_contract.{field} must be a non-negative integer")
    return output


def validate_contract(record: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    record = _mapping(record, where=f"{kind} contract")
    id_field = f"{kind}_id"
    required = (id_field, "display_name", "definition", "sha256")
    _fields(record, required, where=f"{kind} contract")
    output = dict(record)
    for field in required:
        output[field] = _text(record[field], field=f"{kind}.{field}")
    if not re.fullmatch(r"[0-9a-f]{64}", output["sha256"]):
        raise SchemaError(f"{kind}.sha256 must be a lowercase SHA-256")
    return output


def validate_population(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="population")
    required = (
        "population_id",
        "task_id",
        "dataset_id",
        "display_name",
        "population_sha256",
        "expected_n",
        "group_unit",
        "eligibility_rule",
    )
    _fields(record, required, where="population")
    output = dict(record)
    for field in required:
        if field == "expected_n":
            if type(record[field]) is bool or not isinstance(record[field], int) or record[field] < 0:
                raise SchemaError("population.expected_n must be a non-negative integer")
        else:
            output[field] = _text(record[field], field=f"population.{field}")
    if not re.fullmatch(r"[0-9a-f]{64}", output["population_sha256"]):
        raise SchemaError("population.population_sha256 must be a lowercase SHA-256")
    return output


def validate_cell(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="cell")
    required = (
        "cell_id",
        "population_id",
        "task_id",
        "dataset_id",
        "generation_model_id",
        "scorer_model_id",
        "split_id",
        "decoding_id",
        "dataset_family",
        "expected_n",
        "status",
    )
    _fields(record, required, where="cell")
    output = dict(record)
    for field in required:
        if field == "expected_n":
            if type(record[field]) is bool or not isinstance(record[field], int) or record[field] < 0:
                raise SchemaError("cell.expected_n must be a non-negative integer")
        else:
            output[field] = _text(record[field], field=f"cell.{field}")
    return output


def validate_slice(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="slice")
    required = (
        "slice_id",
        "population_id",
        "cell_id",
        "slice_dimension",
        "slice_value",
        "display_name",
        "expected_n",
    )
    _fields(record, required, where="slice")
    output = dict(record)
    for field in required:
        if field == "expected_n":
            if type(record[field]) is bool or not isinstance(record[field], int) or record[field] < 0:
                raise SchemaError("slice.expected_n must be a non-negative integer")
        else:
            output[field] = _text(record[field], field=f"slice.{field}")
    return output


def validate_aggregation(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(record, where="aggregation")
    required = (
        "aggregation_id",
        "display_name",
        "rule",
        "unit_field",
        "component_ids",
        "bootstrap_unit",
        "weighting",
    )
    _fields(record, required, where="aggregation")
    output = dict(record)
    for field in required:
        if field == "component_ids":
            output[field] = _string_list(record[field], field="aggregation.component_ids")
        else:
            output[field] = _text(record[field], field=f"aggregation.{field}")
    if output["rule"] not in ("equal_unit_mean", "pooled_rows", "native_metric", "context_only"):
        raise SchemaError("aggregation.rule is not recognized")
    if output["unit_field"] not in ("cell_id", "dataset_id", "row_id", "native"):
        raise SchemaError("aggregation.unit_field is not recognized")
    return output


REGISTRY_SECTIONS = (
    "tasks",
    "datasets",
    "methods",
    "method_versions",
    "adapters",
    "systems",
    "access_contracts",
    "feature_contracts",
    "evaluators",
    "populations",
    "cells",
    "slices",
    "aggregations",
)


def build_registry(
    *,
    release_id: str,
    tasks: Iterable[Mapping[str, Any]],
    datasets: Iterable[Mapping[str, Any]],
    methods: Iterable[Mapping[str, Any]],
    method_versions: Iterable[Mapping[str, Any]],
    adapters: Iterable[Mapping[str, Any]],
    systems: Iterable[Mapping[str, Any]],
    access_contracts: Iterable[Mapping[str, Any]],
    feature_contracts: Iterable[Mapping[str, Any]],
    evaluators: Iterable[Mapping[str, Any]],
    populations: Iterable[Mapping[str, Any]],
    cells: Iterable[Mapping[str, Any]],
    slices: Iterable[Mapping[str, Any]],
    aggregations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    registry = {
        "schema": REGISTRY_SCHEMA,
        "release_id": _id(release_id, field="release_id"),
        "tasks": list(tasks),
        "datasets": list(datasets),
        "methods": list(methods),
        "method_versions": list(method_versions),
        "adapters": list(adapters),
        "systems": list(systems),
        "access_contracts": list(access_contracts),
        "feature_contracts": list(feature_contracts),
        "evaluators": list(evaluators),
        "populations": list(populations),
        "cells": list(cells),
        "slices": list(slices),
        "aggregations": list(aggregations),
    }
    return validate_registry(registry)


def validate_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    registry = _mapping(registry, where="registry")
    _fields(registry, ("schema", "release_id", *REGISTRY_SECTIONS), where="registry")
    if registry["schema"] != REGISTRY_SCHEMA:
        raise SchemaError(f"registry.schema must be {REGISTRY_SCHEMA!r}")
    release_id = _id(registry["release_id"], field="registry.release_id")
    validators = {
        "tasks": validate_task,
        "datasets": validate_dataset_card,
        "methods": validate_method_card,
        "method_versions": validate_method_version,
        "adapters": validate_adapter,
        "systems": validate_system,
        "access_contracts": validate_access_contract,
        "feature_contracts": lambda value: validate_contract(value, kind="feature_contract"),
        "evaluators": lambda value: validate_contract(value, kind="evaluator"),
        "populations": validate_population,
        "cells": validate_cell,
        "slices": validate_slice,
        "aggregations": validate_aggregation,
    }
    id_fields = {
        "tasks": "task_id",
        "datasets": "dataset_id",
        "methods": "method_id",
        "method_versions": "method_version_id",
        "adapters": "adapter_id",
        "systems": "system_id",
        "access_contracts": "access_contract_id",
        "feature_contracts": "feature_contract_id",
        "evaluators": "evaluator_id",
        "populations": "population_id",
        "cells": "cell_id",
        "slices": "slice_id",
        "aggregations": "aggregation_id",
    }
    normalized: dict[str, Any] = {"schema": REGISTRY_SCHEMA, "release_id": release_id}
    for section in REGISTRY_SECTIONS:
        values = registry[section]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise SchemaError(f"registry.{section} must be a sequence")
        records = [validators[section](value) for value in values]
        _unique(records, id_fields[section], where=f"registry.{section}")
        normalized[section] = sorted(records, key=lambda record: record[id_fields[section]])

    task_ids = {record["task_id"] for record in normalized["tasks"]}
    dataset_ids = {record["dataset_id"] for record in normalized["datasets"]}
    method_ids = {record["method_id"] for record in normalized["methods"]}
    method_version_ids = {record["method_version_id"] for record in normalized["method_versions"]}
    adapter_ids = {record["adapter_id"] for record in normalized["adapters"]}
    access_contract_ids = {
        record["access_contract_id"] for record in normalized["access_contracts"]
    }
    population_ids = {record["population_id"] for record in normalized["populations"]}
    cell_ids = {record["cell_id"] for record in normalized["cells"]}
    feature_contract_ids = {record["feature_contract_id"] for record in normalized["feature_contracts"]}

    for dataset in normalized["datasets"]:
        if dataset["task_id"] not in task_ids:
            raise SchemaError(f"dataset {dataset['dataset_id']!r} references unknown task")
    for version in normalized["method_versions"]:
        if version["method_id"] not in method_ids:
            raise SchemaError(f"method version {version['method_version_id']!r} references unknown method")
        if version["feature_contract_id"] not in feature_contract_ids:
            raise SchemaError(f"method version {version['method_version_id']!r} references unknown feature contract")
    for adapter in normalized["adapters"]:
        if adapter["task_id"] not in task_ids:
            raise SchemaError(f"adapter {adapter['adapter_id']!r} references unknown task")
    for system in normalized["systems"]:
        if system["method_version_id"] not in method_version_ids:
            raise SchemaError(f"system {system['system_id']!r} references unknown method version")
        if system["adapter_id"] not in adapter_ids:
            raise SchemaError(f"system {system['system_id']!r} references unknown adapter")
        if system["access_contract_id"] not in access_contract_ids:
            raise SchemaError(f"system {system['system_id']!r} references unknown access contract")
        version = next(
            item for item in normalized["method_versions"]
            if item["method_version_id"] == system["method_version_id"]
        )
        method = next(
            item for item in normalized["methods"]
            if item["method_id"] == version["method_id"]
        )
        access = next(
            item for item in normalized["access_contracts"]
            if item["access_contract_id"] == system["access_contract_id"]
        )
        expected_access = {
            "access_tier": method["access_tier"],
            "donor_regime": method["donor_regime"],
            "supervision": method["supervision"],
            "model_passes_per_question": method["model_passes"],
        }
        observed_access = {
            field: access[field] for field in expected_access
        }
        if observed_access != expected_access:
            raise SchemaError(
                f"system {system['system_id']!r} access contract disagrees with "
                f"its method card: expected={expected_access!r}, observed={observed_access!r}"
            )
    for population in normalized["populations"]:
        if population["task_id"] not in task_ids or population["dataset_id"] not in dataset_ids:
            raise SchemaError(f"population {population['population_id']!r} references unknown task/dataset")
        dataset_task = next(item["task_id"] for item in normalized["datasets"] if item["dataset_id"] == population["dataset_id"])
        if dataset_task != population["task_id"]:
            raise SchemaError(f"population {population['population_id']!r} task disagrees with its dataset")
    for cell in normalized["cells"]:
        if cell["population_id"] not in population_ids:
            raise SchemaError(f"cell {cell['cell_id']!r} references unknown population")
        if cell["dataset_id"] not in dataset_ids or cell["task_id"] not in task_ids:
            raise SchemaError(f"cell {cell['cell_id']!r} references unknown task/dataset")
        population = next(item for item in normalized["populations"] if item["population_id"] == cell["population_id"])
        if (cell["task_id"], cell["dataset_id"]) != (population["task_id"], population["dataset_id"]):
            raise SchemaError(f"cell {cell['cell_id']!r} disagrees with its population")
    for slice_record in normalized["slices"]:
        if slice_record["cell_id"] not in cell_ids or slice_record["population_id"] not in population_ids:
            raise SchemaError(f"slice {slice_record['slice_id']!r} references unknown population/cell")
        cell = next(item for item in normalized["cells"] if item["cell_id"] == slice_record["cell_id"])
        if cell["population_id"] != slice_record["population_id"]:
            raise SchemaError(f"slice {slice_record['slice_id']!r} population disagrees with its cell")

    # A fixed color/marker represents one method family everywhere.  Duplicated
    # styles make dense plots ambiguous, so require unique pairs.
    style_pairs = [
        (method["style"]["color"], method["style"]["marker"])
        for method in normalized["methods"]
    ]
    if len(style_pairs) != len(set(style_pairs)):
        raise SchemaError("method style color/marker pairs must be unique")

    content = {key: value for key, value in normalized.items() if key != "registry_sha256"}
    digest = canonical_sha256(content)
    if "registry_sha256" in registry and registry["registry_sha256"] != digest:
        raise SchemaError("registry_sha256 does not match registry content")
    normalized["registry_sha256"] = digest
    return normalized


def registry_indexes(registry: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    registry = validate_registry(registry)
    id_fields = {
        "tasks": "task_id",
        "datasets": "dataset_id",
        "methods": "method_id",
        "method_versions": "method_version_id",
        "adapters": "adapter_id",
        "systems": "system_id",
        "access_contracts": "access_contract_id",
        "feature_contracts": "feature_contract_id",
        "evaluators": "evaluator_id",
        "populations": "population_id",
        "cells": "cell_id",
        "slices": "slice_id",
        "aggregations": "aggregation_id",
    }
    return {
        section: {record[id_fields[section]]: record for record in registry[section]}
        for section in REGISTRY_SECTIONS
    }


_RESULT_CONTEXT_FIELDS = (
    "release_id",
    "run_id",
    "lane_id",
    "task_id",
    "dataset_id",
    "population_id",
    "cell_id",
    "slice_id",
    "cohort_id",
    "comparison_group_id",
    "feature_contract_id",
    "access_contract_id",
    "evaluator_id",
    "evidence_grade",
)


def _result_context_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in _RESULT_CONTEXT_FIELDS)


def _validate_prediction_relationships(
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    predictions = rows_by_table.get("predictions", ())
    groups: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in predictions:
        key = _result_context_key(row) + (row["system_id"],)
        groups.setdefault(key, []).append(row)

    for key, rows in groups.items():
        expected_cohort = derive_cohort_id(rows)
        declared = {row["cohort_id"] for row in rows}
        if declared != {expected_cohort}:
            raise SchemaError(
                f"prediction cohort {key!r} is not derived from its full row/group identities; "
                f"expected {expected_cohort!r}, observed={sorted(declared)!r}"
            )
        score_hashes = {row["score_hash"] for row in rows}
        if len(score_hashes) != 1:
            raise SchemaError(f"prediction cohort {key!r} mixes score_hash values")

    # A row/group identity has one label inside an exact comparison group,
    # independent of which system produced its score.
    labels: dict[tuple[Any, ...], Any] = {}
    for row in predictions:
        key = _result_context_key(row) + (row["row_id"], row["group_id"])
        label = row.get("label")
        if key in labels and labels[key] != label:
            raise SchemaError(f"prediction identity {key!r} has system-dependent labels")
        labels[key] = label

    for metric in rows_by_table.get("metrics", ()):
        if metric["aggregation_level"] != "cell":
            continue
        if metric["status"] not in RANKABLE_STATUSES:
            continue
        key = _result_context_key(metric) + (metric["system_id"],)
        matching = groups.get(key)
        if matching is None:
            raise SchemaError(
                f"rankable cell metric {metric['aggregation_id']!r}/{metric['system_id']!r} "
                "has no exact prediction cohort"
            )
        evaluated = [
            row for row in matching
            if row["eligible"]
            and row["status"] in RANKABLE_STATUSES
            and row["continuous_score"] is not None
        ]
        if metric["cohort_id"] != derive_cohort_id(matching):
            raise SchemaError("cell metric cohort_id disagrees with its prediction identities")
        if metric["n_rows"] != len(evaluated):
            raise SchemaError(
                f"cell metric n_rows={metric['n_rows']} but exact prediction cohort has "
                f"{len(evaluated)} scored eligible rows"
            )
        n_groups = len({row["group_id"] for row in evaluated})
        if metric["n_groups"] != n_groups:
            raise SchemaError(
                f"cell metric n_groups={metric['n_groups']} but exact prediction cohort has {n_groups}"
            )
        labels_present = [row["label"] for row in evaluated]
        if labels_present and all(type(value) is bool for value in labels_present):
            positive = sum(bool(value) for value in labels_present)
            if (metric["n_positive"], metric["n_negative"]) != (
                positive,
                len(labels_present) - positive,
            ):
                raise SchemaError("cell metric class counts disagree with prediction labels")

    for coverage in rows_by_table.get("coverage", ()):
        key = _result_context_key(coverage) + (coverage["system_id"],)
        matching = groups.get(key)
        if matching is None:
            if coverage["status"] in RANKABLE_STATUSES or coverage["scored_n"]:
                raise SchemaError(
                    f"rankable/nonempty coverage row {key!r} has no exact prediction cohort"
                )
            continue
        eligible = [row for row in matching if row["eligible"]]
        scored = [
            row for row in eligible
            if row["status"] in RANKABLE_STATUSES and row["continuous_score"] is not None
        ]
        expected = {
            "eligible_n": len(eligible),
            "scored_n": len(scored),
            "fallback_n": sum(bool(row["fallback_used"]) for row in scored),
        }
        observed = {field: coverage[field] for field in expected}
        if observed != expected:
            raise SchemaError(
                f"coverage row {key!r} disagrees with prediction rows: "
                f"expected={expected!r}, observed={observed!r}"
            )
        if coverage["cohort_id"] != derive_cohort_id(matching):
            raise SchemaError("coverage cohort_id disagrees with prediction identities")


def _validate_contrast_relationships(
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    metric_index: MutableMapping[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for metric in rows_by_table.get("metrics", ()):
        key = (
            _result_context_key(metric),
            metric["aggregation_id"],
            metric["aggregation_level"],
            metric["metric_id"],
            metric["system_id"],
        )
        metric_index.setdefault(key, []).append(metric)

    for contrast in rows_by_table.get("contrasts", ()):
        if contrast["comparison_group_id"] != derive_comparison_group_id(contrast):
            raise SchemaError("contrast comparison_group_id is not content-addressed")
        if contrast["system_id"] != contrast["left_system_id"]:
            raise SchemaError("contrast common system_id must identify its left system")
        sides = []
        for side_name in ("left_system_id", "right_system_id"):
            key = (
                _result_context_key(contrast),
                contrast["aggregation_id"],
                contrast["aggregation_level"],
                contrast["metric_id"],
                contrast[side_name],
            )
            matches = metric_index.get(key, [])
            if len(matches) != 1:
                raise SchemaError(
                    f"contrast {side_name}={contrast[side_name]!r} requires exactly one "
                    f"registered metric row; found {len(matches)}"
                )
            sides.append(matches[0])
        left, right = sides
        if left["status"] not in RANKABLE_STATUSES or right["status"] not in RANKABLE_STATUSES:
            if contrast["status"] in RANKABLE_STATUSES:
                raise SchemaError("rankable contrast cannot use a non-rankable metric side")
        for field in (
            "cohort_id",
            "metric_unit",
            "positive_class",
            "better_direction",
            "feature_contract_id",
            "access_contract_id",
            "evaluator_id",
            "evidence_grade",
            "fidelity",
        ):
            if contrast[field] != left[field] or contrast[field] != right[field]:
                raise SchemaError(f"contrast {field} disagrees with its metric sides")
        if contrast["status"] in RANKABLE_STATUSES:
            expected_delta = float(left["value"]) - float(right["value"])
            if abs(float(contrast["delta"]) - expected_delta) > 1e-12:
                raise SchemaError(
                    f"contrast delta {contrast['delta']} is not left minus right "
                    f"({expected_delta})"
                )


def validate_result_references(
    registry: Mapping[str, Any],
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Require every result ID to resolve to the same immutable registry."""

    registry = validate_registry(registry)
    indexes = registry_indexes(registry)
    rows_by_table = {
        table: list(rows) for table, rows in rows_by_table.items()
    }
    valid = {
        "task_id": set(indexes["tasks"]),
        "dataset_id": set(indexes["datasets"]),
        "population_id": set(indexes["populations"]),
        "cell_id": set(indexes["cells"]),
        "slice_id": set(indexes["slices"]),
        "method_id": set(indexes["methods"]),
        "method_version_id": set(indexes["method_versions"]),
        "adapter_id": set(indexes["adapters"]),
        "system_id": set(indexes["systems"]),
        "feature_contract_id": set(indexes["feature_contracts"]),
        "access_contract_id": set(indexes["access_contracts"]),
        "evaluator_id": set(indexes["evaluators"]),
    }
    for table, rows in rows_by_table.items():
        for index, row in enumerate(rows):
            if row.get("release_id") != registry["release_id"]:
                raise SchemaError(
                    f"{table} row {index} release_id does not match registry release"
                )
            for field, allowed in valid.items():
                if field in row and row[field] not in allowed:
                    raise SchemaError(
                        f"{table} row {index} references unknown {field}={row[field]!r}"
                    )
            system = indexes["systems"].get(row.get("system_id"))
            if system is not None:
                if system["method_version_id"] != row.get("method_version_id"):
                    raise SchemaError(f"{table} row {index} system/method_version mismatch")
                if system["adapter_id"] != row.get("adapter_id"):
                    raise SchemaError(f"{table} row {index} system/adapter mismatch")
                version = indexes["method_versions"][system["method_version_id"]]
                if version["method_id"] != row.get("method_id"):
                    raise SchemaError(f"{table} row {index} system/method mismatch")
                if version["feature_contract_id"] != row.get("feature_contract_id"):
                    raise SchemaError(f"{table} row {index} system/feature-contract mismatch")
                if system["access_contract_id"] != row.get("access_contract_id"):
                    raise SchemaError(f"{table} row {index} system/access-contract mismatch")
                adapter = indexes["adapters"][system["adapter_id"]]
                if adapter["task_id"] != row.get("task_id"):
                    raise SchemaError(f"{table} row {index} adapter/task mismatch")
            if table == "contrasts":
                for side in ("left_system_id", "right_system_id"):
                    if row.get(side) not in indexes["systems"]:
                        raise SchemaError(
                            f"contrasts row {index} references unknown {side}={row.get(side)!r}"
                        )
            cell = indexes["cells"].get(row.get("cell_id"))
            if cell is not None:
                for field in ("task_id", "dataset_id", "population_id"):
                    if cell[field] != row.get(field):
                        raise SchemaError(f"{table} row {index} {field} disagrees with its cell")
                slice_record = indexes["slices"].get(row.get("slice_id"))
                if slice_record is not None and slice_record["cell_id"] != cell["cell_id"]:
                    raise SchemaError(f"{table} row {index} slice disagrees with its cell")
    _validate_prediction_relationships(rows_by_table)
    _validate_contrast_relationships(rows_by_table)


def expected_coverage_rows(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the explicit system × slice grid used by the coverage gate."""

    registry = validate_registry(registry)
    systems = [system for system in registry["systems"] if system["enabled"]]
    adapters = {adapter["adapter_id"]: adapter for adapter in registry["adapters"]}
    cells = {cell["cell_id"]: cell for cell in registry["cells"]}
    output = []
    for slice_record in registry["slices"]:
        cell = cells[slice_record["cell_id"]]
        for system in systems:
            if adapters[system["adapter_id"]]["task_id"] != cell["task_id"]:
                continue
            output.append(
                {
                    "release_id": registry["release_id"],
                    "population_id": cell["population_id"],
                    "cell_id": cell["cell_id"],
                    "slice_id": slice_record["slice_id"],
                    "system_id": system["system_id"],
                }
            )
    return sorted(
        output,
        key=lambda row: (
            row["population_id"],
            row["cell_id"],
            row["slice_id"],
            row["system_id"],
        ),
    )
