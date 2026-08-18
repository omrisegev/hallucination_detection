"""Frozen registries and join audits for fair paper comparisons.

This module is intentionally independent of model/scorer code.  It defines the
three machine-readable interfaces used by the comparison package and refuses
the quiet fallbacks that make an apples-to-apples table look complete when it
is not:

``population_registry_v1``
    An ordered population, its grouping metadata, label semantics, and
    eligibility rule.  Row position is never an identifier.

``method_registry_v1``
    A content-addressed method definition with source-artifact provenance,
    fidelity, and an orthogonal access declaration.

``comparison_record_v1``
    One method score/prediction on one canonical row (and, where applicable,
    one causal budget).

The helpers use canonical JSON (sorted keys, no NaN/Infinity) so every registry
and final file can be verified byte-for-byte.  There are deliberately no
timestamp fields in these interfaces: a rebuild from the same inputs must
produce the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


POPULATION_SCHEMA = "population_registry_v1"
METHOD_SCHEMA = "method_registry_v1"
COMPARISON_SCHEMA = "comparison_record_v1"
ASSET_SCHEMA = "asset_record_v1"
ASSET_REGISTRY_SCHEMA = "asset_registry_v1"
JOIN_AUDIT_SCHEMA = "comparison_join_audit_v1"
HASH_MANIFEST_SCHEMA = "fair_comparison_hash_manifest_v1"

LANES = ("global", "localization", "prefix", "stopping")

# Canonical vocabulary from PROGRESS.md Step 274 and the paper-exact acquisition
# contract.  Access is *not* encoded in this label; it lives in ACCESS_FIELDS.
FIDELITY_LABELS = (
    "official-exact",
    "paper-specified",
    "paper-specified-partial",
    "adapted-common-protocol",
    "published-context-only",
    "blocked-assets",
)

ACCESS_FIELDS = (
    "input_type",
    "supervision",
    "model_passes_per_question",
    "traces_per_question",
)

HASH_ALGORITHM = "sha256"
ORDER_HASH_ENCODING = "canonical-json-array-v1"
_HEX_DIGITS = frozenset("0123456789abcdef")


class RegistryError(ValueError):
    """A frozen registry or comparison record violates its contract."""


class JoinAuditError(RegistryError):
    """The strict identical-row join gate did not pass."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes.

    NaN and infinity are rejected rather than serialized as JavaScript tokens.
    Those values are neither valid JSON nor stable missing-value semantics; use
    ``None`` explicitly for an unavailable score or prediction.
    """

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"value is not canonical-JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1 << 20) -> str:
    """Stream a file SHA-256 without loading a large artifact into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def ordered_id_sha256(row_ids: Sequence[str]) -> str:
    """Hash an ordered ID vector using an unambiguous canonical JSON array.

    This intentionally differs from hashing an ID set.  It also avoids the
    delimiter ambiguity of joining IDs with newlines (an ID is allowed to
    contain any Unicode character except the empty string).
    """

    ids = _validate_ids(row_ids, field="ordered_ids")
    return canonical_sha256(ids)


def write_canonical_json(
    path: os.PathLike[str] | str,
    value: Any,
    *,
    trailing_newline: bool = True,
) -> None:
    """Atomically write canonical JSON, creating parent directories as needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + (b"\n" if trailing_newline else b"")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_ids(values: Sequence[Any], *, field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RegistryError(f"{field} must be a sequence of canonical string IDs")
    ids = list(values)
    if any(not isinstance(value, str) or value == "" for value in ids):
        raise RegistryError(f"{field} must contain only non-empty strings")
    duplicates = _duplicates(ids)
    if duplicates:
        raise RegistryError(f"{field} contains duplicate IDs: {duplicates[:5]!r}")
    return ids


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicate: list[Any] = []
    duplicate_seen: set[Any] = set()
    for value in values:
        if value in seen and value not in duplicate_seen:
            duplicate.append(value)
            duplicate_seen.add(value)
        seen.add(value)
    return duplicate


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], *, where: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise RegistryError(f"{where} missing required fields: {missing}")


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{field} must be a non-empty string")
    return value


def _aligned_strings(values: Any, n_rows: int, *, field: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RegistryError(f"{field} must be a sequence aligned with ordered_ids")
    result = list(values)
    if len(result) != n_rows:
        raise RegistryError(
            f"{field} has {len(result)} values but ordered_ids has {n_rows}"
        )
    if any(not isinstance(value, str) or value == "" for value in result):
        raise RegistryError(f"{field} must contain only non-empty strings")
    return result


def make_population_entry(
    *,
    population_id: str,
    lane: str,
    dataset_revision: str,
    ordered_ids: Sequence[str],
    group_ids: Sequence[str],
    cell_ids: Sequence[str],
    families: Sequence[str],
    label_definition: Mapping[str, Any] | str,
    eligibility_rules: Mapping[str, Any] | Sequence[Any] | str,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Construct one validated, content-addressed population entry."""

    ids = _validate_ids(ordered_ids, field="ordered_ids")
    entry: dict[str, Any] = {
        "population_id": population_id,
        "lane": lane,
        "dataset_revision": dataset_revision,
        "ordered_ids": ids,
        "group_ids": list(group_ids),
        "cell_ids": list(cell_ids),
        "families": list(families),
        "label_definition": label_definition,
        "eligibility_rules": eligibility_rules,
        "ordered_id_sha256": ordered_id_sha256(ids),
        "order_hash_encoding": ORDER_HASH_ENCODING,
    }
    if extra:
        overlap = set(entry).intersection(extra)
        if overlap:
            raise RegistryError(f"population extra fields overwrite frozen fields: {sorted(overlap)}")
        entry.update(dict(extra))
    return validate_population_entry(entry)


def make_eligible_population(
    ordered_ids: Sequence[str],
    *,
    rule: str,
) -> dict[str, Any]:
    """Content-address one exact ordered eligibility cohort.

    A lane may expose several non-interchangeable cohorts inside one registered
    source population (for example Prefix ``length > 64``, ``length > 128``, and
    complete-six-budget warning traces).  Keeping the ordered IDs alongside their
    hash prevents a report from attaching the full-population hash to a metric that
    was evaluated on only a strict subset.
    """

    ids = _validate_ids(ordered_ids, field="eligible ordered_ids")
    _nonempty_string(rule, field="eligible population rule")
    return {
        "schema": "eligible_population_v1",
        "rule": rule,
        "ordered_ids": ids,
        "ordered_id_sha256": ordered_id_sha256(ids),
        "order_hash_encoding": ORDER_HASH_ENCODING,
    }


def validate_eligible_population(
    descriptor: Mapping[str, Any],
    *,
    registered_ordered_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate an eligibility descriptor as an ordered registered subset."""

    if not isinstance(descriptor, Mapping):
        raise RegistryError("eligible population descriptor must be a mapping")
    _require_fields(
        descriptor,
        ("schema", "rule", "ordered_ids", "ordered_id_sha256", "order_hash_encoding"),
        where="eligible population descriptor",
    )
    normalized = dict(descriptor)
    if normalized["schema"] != "eligible_population_v1":
        raise RegistryError(
            f"unexpected eligible population schema {normalized['schema']!r}"
        )
    _nonempty_string(normalized["rule"], field="eligible population rule")
    eligible = _validate_ids(normalized["ordered_ids"], field="eligible ordered_ids")
    registered = _validate_ids(
        registered_ordered_ids, field="registered ordered_ids"
    )
    eligible_set = set(eligible)
    outside = [row_id for row_id in eligible if row_id not in set(registered)]
    if outside:
        raise RegistryError(
            f"eligible population contains IDs outside its registered universe: {outside[:5]}"
        )
    canonical_subset = [row_id for row_id in registered if row_id in eligible_set]
    if eligible != canonical_subset:
        raise RegistryError(
            "eligible population ordered_ids must preserve registered population order"
        )
    if normalized["order_hash_encoding"] != ORDER_HASH_ENCODING:
        raise RegistryError("eligible population uses an unsupported ordered-ID encoding")
    expected_hash = ordered_id_sha256(eligible)
    if normalized["ordered_id_sha256"] != expected_hash:
        raise RegistryError(
            "eligible population ordered_id_sha256 does not match ordered_ids"
        )
    normalized["ordered_ids"] = eligible
    canonical_json_bytes(normalized)
    return normalized


def validate_population_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one population entry without changing row order."""

    if not isinstance(entry, Mapping):
        raise RegistryError("population entry must be a mapping")
    required = (
        "population_id",
        "lane",
        "dataset_revision",
        "ordered_ids",
        "group_ids",
        "cell_ids",
        "families",
        "label_definition",
        "eligibility_rules",
        "ordered_id_sha256",
        "order_hash_encoding",
    )
    _require_fields(entry, required, where="population entry")
    normalized = dict(entry)
    _nonempty_string(normalized["population_id"], field="population_id")
    if normalized["lane"] not in LANES:
        raise RegistryError(f"lane {normalized['lane']!r} not in {LANES}")
    _nonempty_string(normalized["dataset_revision"], field="dataset_revision")
    ids = _validate_ids(normalized["ordered_ids"], field="ordered_ids")
    if not ids:
        raise RegistryError("a registered comparison population must contain at least one row")
    normalized["ordered_ids"] = ids
    normalized["group_ids"] = _aligned_strings(
        normalized["group_ids"], len(ids), field="group_ids"
    )
    normalized["cell_ids"] = _aligned_strings(
        normalized["cell_ids"], len(ids), field="cell_ids"
    )
    normalized["families"] = _aligned_strings(
        normalized["families"], len(ids), field="families"
    )
    group_families: dict[str, str] = {}
    for group_id, family in zip(normalized["group_ids"], normalized["families"]):
        previous = group_families.setdefault(group_id, family)
        if previous != family:
            raise RegistryError(
                f"group_id {group_id!r} spans conflicting families {previous!r} and {family!r}"
            )
    if normalized["label_definition"] in (None, "", {}, []):
        raise RegistryError("label_definition must explicitly define the positive class")
    if normalized["eligibility_rules"] in (None, "", {}, []):
        raise RegistryError("eligibility_rules must be explicit, even when all rows are eligible")
    if normalized["order_hash_encoding"] != ORDER_HASH_ENCODING:
        raise RegistryError(
            f"unsupported order_hash_encoding {normalized['order_hash_encoding']!r}"
        )
    expected_hash = ordered_id_sha256(ids)
    if normalized["ordered_id_sha256"] != expected_hash:
        raise RegistryError(
            "ordered_id_sha256 does not match ordered_ids: "
            f"{normalized['ordered_id_sha256']!r} != {expected_hash!r}"
        )
    for field, universe in (
        ("eligible_populations", ids),
        ("eligible_group_populations", list(dict.fromkeys(normalized["group_ids"]))),
    ):
        if field not in normalized:
            continue
        descriptors = normalized[field]
        if not isinstance(descriptors, Mapping) or not descriptors:
            raise RegistryError(f"{field} must be a non-empty mapping")
        normalized[field] = {
            str(name): validate_eligible_population(
                descriptor,
                registered_ordered_ids=universe,
            )
            for name, descriptor in sorted(descriptors.items())
        }
    # Force a canonical-JSON check here so bad extra metadata cannot enter a
    # registry only to fail much later during final manifest construction.
    canonical_json_bytes(normalized)
    return normalized


def build_population_registry(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    populations = [validate_population_entry(entry) for entry in entries]
    populations.sort(key=lambda entry: entry["population_id"])
    duplicate_ids = _duplicates(entry["population_id"] for entry in populations)
    if duplicate_ids:
        raise RegistryError(f"duplicate population_id values: {duplicate_ids}")
    payload = {
        "schema": POPULATION_SCHEMA,
        "hash_algorithm": HASH_ALGORITHM,
        "order_hash_encoding": ORDER_HASH_ENCODING,
        "populations": populations,
        "content_sha256": canonical_sha256(populations),
    }
    return payload


def validate_population_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise RegistryError("population registry must be a mapping")
    _require_fields(
        registry,
        ("schema", "hash_algorithm", "order_hash_encoding", "populations", "content_sha256"),
        where="population registry",
    )
    if registry["schema"] != POPULATION_SCHEMA:
        raise RegistryError(f"unexpected population registry schema {registry['schema']!r}")
    if registry["hash_algorithm"] != HASH_ALGORITHM:
        raise RegistryError("population registry must use SHA-256")
    if registry["order_hash_encoding"] != ORDER_HASH_ENCODING:
        raise RegistryError("population registry uses an unsupported ordered-ID encoding")
    rebuilt = build_population_registry(registry["populations"])
    if registry["content_sha256"] != rebuilt["content_sha256"]:
        raise RegistryError("population registry content_sha256 mismatch")
    return rebuilt


def population_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return population metadata with an ID-keyed row index."""

    validated = validate_population_registry(registry)
    result: dict[str, dict[str, Any]] = {}
    for population in validated["populations"]:
        rows = {
            row_id: {
                "row_id": row_id,
                "group_id": population["group_ids"][index],
                "cell_id": population["cell_ids"][index],
                "family": population["families"][index],
                "position": index,
            }
            for index, row_id in enumerate(population["ordered_ids"])
        }
        result[population["population_id"]] = {**population, "row_index": rows}
    return result


def make_asset_record(
    path: os.PathLike[str] | str,
    *,
    artifact_id: Optional[str] = None,
    root: os.PathLike[str] | str | None = None,
    uri: Optional[str] = None,
) -> dict[str, Any]:
    """Hash one local source artifact and return a portable provenance row."""

    source = Path(path)
    if source.is_symlink():
        raise RegistryError(f"source artifact must not be a symlink: {source}")
    if not source.is_file():
        raise RegistryError(f"source artifact is not a regular file: {source}")
    resolved = source.resolve()
    if root is None:
        logical_path = resolved.as_posix()
    else:
        base = Path(root).resolve()
        try:
            logical_path = resolved.relative_to(base).as_posix()
        except ValueError as exc:
            raise RegistryError(f"artifact {resolved} is outside declared root {base}") from exc
    record = {
        "schema": ASSET_SCHEMA,
        "artifact_kind": "file",
        "artifact_id": artifact_id or logical_path,
        "uri": uri or logical_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    return validate_asset_record(record)


def make_derived_asset_record(
    projection: Mapping[str, Any],
    *,
    artifact_kind: str,
    artifact_id: str,
    uri: str,
    source_fingerprint_aliases: Sequence[str] = (),
    source_fingerprint_preimages: Sequence[Any] = (),
) -> dict[str, Any]:
    """Hash a derived/composite ledger over its exact canonical JSON bytes."""

    if artifact_kind not in {"derived-ledger", "composite-ledger"}:
        raise RegistryError(
            "make_derived_asset_record requires derived-ledger or composite-ledger"
        )
    preimage_entries = [
        {
            "sha256": canonical_sha256(preimage),
            "preimage": preimage,
        }
        for preimage in source_fingerprint_preimages
    ]
    derived_aliases = [entry["sha256"] for entry in preimage_entries]
    aliases = list(source_fingerprint_aliases)
    if aliases and not preimage_entries:
        raise RegistryError(
            "source_fingerprint_aliases require canonical source_fingerprint_preimages"
        )
    if aliases and aliases != derived_aliases:
        raise RegistryError(
            "source_fingerprint_aliases do not hash their canonical preimages"
        )
    if preimage_entries:
        aliases = derived_aliases
        existing_preimages = projection.get("source_fingerprint_preimages")
        if existing_preimages is not None and existing_preimages != preimage_entries:
            raise RegistryError(
                "projection source_fingerprint_preimages disagree with declared preimages"
            )
        projection = {
            **dict(projection),
            "source_fingerprint_preimages": preimage_entries,
        }
    if aliases:
        existing = projection.get("source_fingerprint_aliases")
        if existing is not None and existing != aliases:
            raise RegistryError(
                "projection source_fingerprint_aliases disagree with declared aliases"
            )
        projection = {**dict(projection), "source_fingerprint_aliases": aliases}
    projection_bytes = canonical_json_bytes(projection)
    return validate_asset_record(
        {
            "schema": ASSET_SCHEMA,
            "artifact_kind": artifact_kind,
            "artifact_id": artifact_id,
            "uri": uri,
            "size_bytes": len(projection_bytes),
            "sha256": hashlib.sha256(projection_bytes).hexdigest(),
            "projection": dict(projection),
            **({"source_fingerprint_aliases": aliases} if aliases else {}),
        }
    )


def validate_asset_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RegistryError("asset record must be a mapping")
    _require_fields(
        record,
        ("schema", "artifact_id", "uri", "size_bytes", "sha256"),
        where="asset record",
    )
    normalized = dict(record)
    if normalized["schema"] != ASSET_SCHEMA:
        raise RegistryError(f"unexpected asset schema {normalized['schema']!r}")
    _nonempty_string(normalized["artifact_id"], field="artifact_id")
    _nonempty_string(normalized["uri"], field="asset uri")
    if not isinstance(normalized["size_bytes"], int) or normalized["size_bytes"] < 0:
        raise RegistryError("asset size_bytes must be a non-negative integer")
    if not is_sha256(normalized["sha256"]):
        raise RegistryError("asset sha256 must be a lowercase 64-character SHA-256")
    artifact_kind = normalized.get("artifact_kind")
    if artifact_kind is None and "projection" in normalized:
        raise RegistryError(
            "asset records with a projection must declare derived/composite artifact_kind"
        )
    if artifact_kind is not None:
        allowed_kinds = {"file", "remote-file", "derived-ledger", "composite-ledger"}
        if artifact_kind not in allowed_kinds:
            raise RegistryError(
                f"asset artifact_kind {artifact_kind!r} not in {sorted(allowed_kinds)}"
            )
        projection = normalized.get("projection")
        if artifact_kind in {"derived-ledger", "composite-ledger"}:
            if not isinstance(projection, Mapping):
                raise RegistryError(
                    f"{artifact_kind} assets require an explicit canonical projection"
                )
            projection_bytes = canonical_json_bytes(projection)
            expected_hash = hashlib.sha256(projection_bytes).hexdigest()
            if normalized["sha256"] != expected_hash:
                raise RegistryError(
                    f"{artifact_kind} sha256 does not match its canonical projection"
                )
            if normalized["size_bytes"] != len(projection_bytes):
                raise RegistryError(
                    f"{artifact_kind} size_bytes does not match canonical projection bytes"
                )
            aliases = normalized.get("source_fingerprint_aliases", [])
            if not isinstance(aliases, list) or any(not is_sha256(value) for value in aliases):
                raise RegistryError(
                    f"{artifact_kind} source_fingerprint_aliases must be a list of SHA-256s"
                )
            if len(set(aliases)) != len(aliases):
                raise RegistryError(
                    f"{artifact_kind} source_fingerprint_aliases must be unique"
                )
            if aliases != list(projection.get("source_fingerprint_aliases", [])):
                raise RegistryError(
                    f"{artifact_kind} source_fingerprint_aliases must be declared in projection"
                )
            preimages = projection.get("source_fingerprint_preimages", [])
            if not isinstance(preimages, list) or len(preimages) != len(aliases):
                raise RegistryError(
                    f"{artifact_kind} aliases require one canonical preimage each"
                )
            for alias, entry in zip(aliases, preimages):
                if not isinstance(entry, Mapping) or set(entry) != {"sha256", "preimage"}:
                    raise RegistryError(
                        f"{artifact_kind} source fingerprint preimage entry is malformed"
                    )
                if entry["sha256"] != alias or canonical_sha256(entry["preimage"]) != alias:
                    raise RegistryError(
                        f"{artifact_kind} source fingerprint alias does not hash its preimage"
                    )
            if not aliases and preimages:
                raise RegistryError(
                    f"{artifact_kind} source fingerprint preimages require aliases"
                )
        elif projection is not None:
            raise RegistryError(
                f"{artifact_kind} assets must not claim a derived projection"
            )
    canonical_json_bytes(normalized)
    return normalized


def build_asset_registry(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the package-level registry of available, content-hashed assets."""

    assets = [validate_asset_record(record) for record in records]
    assets.sort(key=lambda record: record["artifact_id"])
    duplicate_ids = _duplicates(record["artifact_id"] for record in assets)
    if duplicate_ids:
        raise RegistryError(f"duplicate asset artifact_id values: {duplicate_ids}")
    return {
        "schema": ASSET_REGISTRY_SCHEMA,
        "hash_algorithm": HASH_ALGORITHM,
        "assets": assets,
        "content_sha256": canonical_sha256(assets),
    }


def validate_asset_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise RegistryError("asset registry must be a mapping")
    _require_fields(
        registry,
        ("schema", "hash_algorithm", "assets", "content_sha256"),
        where="asset registry",
    )
    if registry["schema"] != ASSET_REGISTRY_SCHEMA:
        raise RegistryError(f"unexpected asset registry schema {registry['schema']!r}")
    if registry["hash_algorithm"] != HASH_ALGORITHM:
        raise RegistryError("asset registry must use SHA-256")
    rebuilt = build_asset_registry(registry["assets"])
    if registry["content_sha256"] != rebuilt["content_sha256"]:
        raise RegistryError("asset registry content_sha256 mismatch")
    return rebuilt


def source_artifacts_sha256(artifacts: Sequence[Mapping[str, Any]]) -> str:
    normalized = [validate_asset_record(artifact) for artifact in artifacts]
    normalized.sort(key=lambda artifact: artifact["artifact_id"])
    projection = [
        {
            "artifact_id": artifact["artifact_id"],
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
        }
        for artifact in normalized
    ]
    return canonical_sha256(projection)


_METHOD_REQUIRED_FIELDS = (
    "method_id",
    "display_name",
    "fidelity",
    "source_artifacts",
    "source_artifacts_sha256",
    "access",
    "training_label_use",
    "checkpoint_revision",
    "prompt_sha256",
    "decoding_sha256",
    "evaluator_sha256",
    "run_commit",
    "deviations",
    "method_hash",
)


def _validate_access(access: Any, *, fidelity: str) -> dict[str, Any]:
    if not isinstance(access, Mapping):
        raise RegistryError("method access must be a mapping")
    _require_fields(access, ACCESS_FIELDS, where="method access")
    normalized = dict(access)
    _nonempty_string(normalized["input_type"], field="access.input_type")
    _nonempty_string(normalized["supervision"], field="access.supervision")
    direct = fidelity not in ("published-context-only", "blocked-assets")
    for field in ("model_passes_per_question", "traces_per_question"):
        value = normalized[field]
        if value is None and not direct:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise RegistryError(f"access.{field} must be a finite non-negative number")
    canonical_json_bytes(normalized)
    return normalized


def _method_hash_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "method_hash"}


def method_definition_sha256(entry: Mapping[str, Any]) -> str:
    return canonical_sha256(_method_hash_projection(entry))


def make_method_entry(
    *,
    method_id: str,
    display_name: str,
    fidelity: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
    training_label_use: Mapping[str, Any] | str,
    checkpoint_revision: Optional[str],
    prompt_sha256: Optional[str],
    decoding_sha256: Optional[str],
    evaluator_sha256: Optional[str],
    run_commit: Optional[str],
    deviations: Sequence[Mapping[str, Any] | str],
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Construct a method entry and bind all declared provenance into method_hash."""

    artifacts = [validate_asset_record(artifact) for artifact in source_artifacts]
    artifacts.sort(key=lambda artifact: artifact["artifact_id"])
    entry: dict[str, Any] = {
        "method_id": method_id,
        "display_name": display_name,
        "fidelity": fidelity,
        "source_artifacts": artifacts,
        "source_artifacts_sha256": source_artifacts_sha256(artifacts),
        "access": dict(access),
        "training_label_use": training_label_use,
        "checkpoint_revision": checkpoint_revision,
        "prompt_sha256": prompt_sha256,
        "decoding_sha256": decoding_sha256,
        "evaluator_sha256": evaluator_sha256,
        "run_commit": run_commit,
        "deviations": list(deviations),
    }
    if extra:
        overlap = set(entry).intersection(extra)
        if overlap:
            raise RegistryError(f"method extra fields overwrite frozen fields: {sorted(overlap)}")
        entry.update(dict(extra))
    entry["method_hash"] = method_definition_sha256(entry)
    return validate_method_entry(entry)


def validate_method_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RegistryError("method entry must be a mapping")
    _require_fields(entry, _METHOD_REQUIRED_FIELDS, where="method entry")
    normalized = dict(entry)
    _nonempty_string(normalized["method_id"], field="method_id")
    _nonempty_string(normalized["display_name"], field="display_name")
    fidelity = normalized["fidelity"]
    if fidelity not in FIDELITY_LABELS:
        raise RegistryError(f"fidelity {fidelity!r} not in {FIDELITY_LABELS}")

    artifacts = [validate_asset_record(artifact) for artifact in normalized["source_artifacts"]]
    artifacts.sort(key=lambda artifact: artifact["artifact_id"])
    duplicate_assets = _duplicates(artifact["artifact_id"] for artifact in artifacts)
    if duplicate_assets:
        raise RegistryError(f"duplicate source artifact IDs: {duplicate_assets}")
    normalized["source_artifacts"] = artifacts
    actual_artifact_hash = source_artifacts_sha256(artifacts)
    if normalized["source_artifacts_sha256"] != actual_artifact_hash:
        raise RegistryError("source_artifacts_sha256 does not match source_artifacts")

    normalized["access"] = _validate_access(normalized["access"], fidelity=fidelity)
    if normalized["training_label_use"] in (None, "", {}, []):
        raise RegistryError("training_label_use must explicitly state label use")
    if not isinstance(normalized["deviations"], list):
        raise RegistryError("deviations must be a list")

    direct = fidelity not in ("published-context-only", "blocked-assets")
    if direct and not artifacts:
        raise RegistryError(f"{fidelity} method must have at least one hashed source artifact")
    if direct:
        _nonempty_string(normalized["checkpoint_revision"], field="checkpoint_revision")
        _nonempty_string(normalized["run_commit"], field="run_commit")
        for field in ("prompt_sha256", "decoding_sha256", "evaluator_sha256"):
            if not is_sha256(normalized[field]):
                raise RegistryError(f"direct method {field} must be a SHA-256")
    else:
        for field in ("prompt_sha256", "decoding_sha256", "evaluator_sha256"):
            if normalized[field] is not None and not is_sha256(normalized[field]):
                raise RegistryError(f"{field} must be null or a SHA-256")

    if fidelity in ("official-exact", "paper-specified") and normalized["deviations"]:
        raise RegistryError(f"{fidelity} is incompatible with declared deviations")
    if fidelity in ("paper-specified-partial", "adapted-common-protocol") and not normalized[
        "deviations"
    ]:
        raise RegistryError(f"{fidelity} requires at least one explicit deviation")

    expected_method_hash = method_definition_sha256(normalized)
    if normalized["method_hash"] != expected_method_hash:
        raise RegistryError("method_hash does not match the frozen method definition")
    canonical_json_bytes(normalized)
    return normalized


def build_method_registry(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    methods = [validate_method_entry(entry) for entry in entries]
    methods.sort(key=lambda entry: entry["method_id"])
    duplicate_ids = _duplicates(entry["method_id"] for entry in methods)
    if duplicate_ids:
        raise RegistryError(f"duplicate method_id values: {duplicate_ids}")
    return {
        "schema": METHOD_SCHEMA,
        "hash_algorithm": HASH_ALGORITHM,
        "methods": methods,
        "content_sha256": canonical_sha256(methods),
    }


def validate_method_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise RegistryError("method registry must be a mapping")
    _require_fields(
        registry,
        ("schema", "hash_algorithm", "methods", "content_sha256"),
        where="method registry",
    )
    if registry["schema"] != METHOD_SCHEMA:
        raise RegistryError(f"unexpected method registry schema {registry['schema']!r}")
    if registry["hash_algorithm"] != HASH_ALGORITHM:
        raise RegistryError("method registry must use SHA-256")
    rebuilt = build_method_registry(registry["methods"])
    if registry["content_sha256"] != rebuilt["content_sha256"]:
        raise RegistryError("method registry content_sha256 mismatch")
    return rebuilt


def method_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_method_registry(registry)
    return {method["method_id"]: method for method in validated["methods"]}


_COMPARISON_FIELDS = (
    "schema",
    "lane",
    "population_id",
    "row_id",
    "group_id",
    "cell_id",
    "method_id",
    "continuous_score",
    "discrete_prediction",
    "label",
    "budget",
    "fold",
    "calibration_hash",
    "source_artifact_hash",
)


def make_comparison_record(
    *,
    lane: str,
    population_id: str,
    row_id: str,
    group_id: str,
    cell_id: str,
    method_id: str,
    continuous_score: Optional[float],
    discrete_prediction: Any,
    label: Any,
    budget: Optional[int | str],
    fold: Optional[int],
    calibration_hash: Optional[str],
    source_artifact_hash: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "lane": lane,
        "population_id": population_id,
        "row_id": row_id,
        "group_id": group_id,
        "cell_id": cell_id,
        "method_id": method_id,
        "continuous_score": continuous_score,
        "discrete_prediction": discrete_prediction,
        "label": label,
        "budget": budget,
        "fold": fold,
        "calibration_hash": calibration_hash,
        "source_artifact_hash": source_artifact_hash,
    }
    if extra:
        overlap = set(record).intersection(extra)
        if overlap:
            raise RegistryError(f"comparison extra fields overwrite frozen fields: {sorted(overlap)}")
        record.update(dict(extra))
    return validate_comparison_record(record)


def validate_comparison_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RegistryError("comparison record must be a mapping")
    _require_fields(record, _COMPARISON_FIELDS, where="comparison record")
    normalized = dict(record)
    if normalized["schema"] != COMPARISON_SCHEMA:
        raise RegistryError(f"unexpected comparison record schema {normalized['schema']!r}")
    if normalized["lane"] not in LANES:
        raise RegistryError(f"comparison lane {normalized['lane']!r} not in {LANES}")
    for field in ("population_id", "row_id", "group_id", "cell_id", "method_id"):
        _nonempty_string(normalized[field], field=field)
    score = normalized["continuous_score"]
    if score is not None and (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise RegistryError("continuous_score must be a finite number or null")
    if normalized["discrete_prediction"] is not None and isinstance(
        normalized["discrete_prediction"], (dict, list, tuple, set)
    ):
        raise RegistryError("discrete_prediction must be a scalar or null")
    if normalized["label"] is None or isinstance(normalized["label"], (dict, list, tuple, set)):
        raise RegistryError("label must be a non-null scalar")
    budget = normalized["budget"]
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, str)):
            raise RegistryError("budget must be a non-negative integer, non-empty string, or null")
        if isinstance(budget, int) and budget < 0:
            raise RegistryError("integer budget must be non-negative")
        if isinstance(budget, str) and not budget:
            raise RegistryError("string budget must be non-empty")
    fold = normalized["fold"]
    if fold is not None and (
        isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5)
    ):
        raise RegistryError("fold must be null or one of 0, 1, 2, 3, 4")
    calibration_hash = normalized["calibration_hash"]
    if calibration_hash is not None and not is_sha256(calibration_hash):
        raise RegistryError("calibration_hash must be null or a lowercase SHA-256")
    if not is_sha256(normalized["source_artifact_hash"]):
        raise RegistryError("source_artifact_hash must be a lowercase 64-character SHA-256")
    canonical_json_bytes(normalized)
    return normalized


def canonicalize_comparison_records(
    records: Iterable[Mapping[str, Any]],
    population_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return validated long-form rows in a deterministic population order."""

    populations = population_index(population_registry)
    normalized = [validate_comparison_record(record) for record in records]

    def key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        population = populations.get(record["population_id"])
        row_position = (
            population["row_index"].get(record["row_id"], {}).get("position", 10**18)
            if population
            else 10**18
        )
        budget = canonical_json_bytes(record["budget"])
        return (
            record["lane"],
            record["population_id"],
            budget,
            row_position,
            record["row_id"],
            record["method_id"],
        )

    return sorted(normalized, key=key)


def _budget_key(value: Any) -> str:
    """Canonical scalar token, preserving the distinction between 16 and "16"."""

    return canonical_json_bytes(value).decode("utf-8")


def _audit_record_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["population_id"]),
        str(record["row_id"]),
        str(record["method_id"]),
        _budget_key(record["budget"]),
    )


def audit_comparison_records(
    records: Iterable[Mapping[str, Any]],
    population_registry: Mapping[str, Any],
    method_registry: Mapping[str, Any],
    *,
    expectations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Audit an ID join before any metric is computed.

    ``expectations`` is an optional list with ``population_id``, ``method_id``,
    and ``budget``.  Each item may supply ``eligible_row_ids`` (otherwise the
    whole registered population), ``eligible_ordered_id_sha256``, ``headline``
    (default true), ``require_complete`` (default true), and ``require_order``
    (default true).  ``match_any_budget`` is reserved for registered realized-
    compute rows whose budget is an outcome rather than an eligibility choice;
    it still requires exactly one row per eligible ID.
    Prefix budgets should pass their strict ``final_length > budget`` ID vector
    here; this makes the causal eligibility population itself hash-addressed.

    The function returns a complete report instead of failing at the first
    defect, making duplicate, label-conflict, missing-field, coverage, and
    ordered-hash failures actionable in one pass.  Call
    :func:`require_clean_join` before scoring a headline table.
    """

    populations = population_index(population_registry)
    methods = method_index(method_registry)
    raw_records = list(records)
    valid_records: list[dict[str, Any]] = []
    valid_raw_indices: list[int] = []
    missing_or_invalid: list[dict[str, Any]] = []

    for index, record in enumerate(raw_records):
        try:
            valid_records.append(validate_comparison_record(record))
            valid_raw_indices.append(index)
        except RegistryError as exc:
            missing_or_invalid.append({"record_index": index, "problem": str(exc)})

    unknown_populations: list[dict[str, Any]] = []
    unknown_methods: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    metadata_conflicts: list[dict[str, Any]] = []
    lane_conflicts: list[dict[str, Any]] = []
    artifact_hash_conflicts: list[dict[str, Any]] = []
    for index, record in enumerate(valid_records):
        raw_index = valid_raw_indices[index]
        population = populations.get(record["population_id"])
        if population is None:
            unknown_populations.append(
                {"record_index": raw_index, "population_id": record["population_id"]}
            )
            continue
        method = methods.get(record["method_id"])
        if method is None:
            unknown_methods.append(
                {"record_index": raw_index, "method_id": record["method_id"]}
            )
        else:
            allowed_hashes = {method["source_artifacts_sha256"]}
            allowed_hashes.update(
                artifact["sha256"] for artifact in method["source_artifacts"]
            )
            for artifact in method["source_artifacts"]:
                allowed_hashes.update(artifact.get("source_fingerprint_aliases", []))
            if record["source_artifact_hash"] not in allowed_hashes:
                artifact_hash_conflicts.append(
                    {
                        "record_index": raw_index,
                        "method_id": record["method_id"],
                        "observed": record["source_artifact_hash"],
                        "allowed": sorted(allowed_hashes),
                    }
                )
        row = population["row_index"].get(record["row_id"])
        if row is None:
            unknown_rows.append(
                {
                    "record_index": raw_index,
                    "population_id": record["population_id"],
                    "row_id": record["row_id"],
                }
            )
            continue
        if record["lane"] != population["lane"]:
            lane_conflicts.append(
                {
                    "record_index": raw_index,
                    "population_id": record["population_id"],
                    "expected": population["lane"],
                    "observed": record["lane"],
                }
            )
        for field in ("group_id", "cell_id"):
            if record[field] != row[field]:
                metadata_conflicts.append(
                    {
                        "record_index": raw_index,
                        "population_id": record["population_id"],
                        "row_id": record["row_id"],
                        "field": field,
                        "expected": row[field],
                        "observed": record[field],
                    }
                )

    by_key: dict[tuple[str, str, str, str], list[int]] = {}
    for index, record in enumerate(valid_records):
        by_key.setdefault(_audit_record_key(record), []).append(valid_raw_indices[index])
    duplicates = [
        {
            "population_id": key[0],
            "row_id": key[1],
            "method_id": key[2],
            "budget": json.loads(key[3]),
            "record_indices": indices,
        }
        for key, indices in sorted(by_key.items())
        if len(indices) > 1
    ]

    labels_by_row: dict[tuple[str, str], dict[str, list[int]]] = {}
    label_values: dict[str, Any] = {}
    for index, record in enumerate(valid_records):
        row_key = (record["population_id"], record["row_id"])
        label_key = canonical_json_bytes(record["label"]).decode("utf-8")
        labels_by_row.setdefault(row_key, {}).setdefault(label_key, []).append(
            valid_raw_indices[index]
        )
        label_values[label_key] = record["label"]
    label_conflicts = []
    for row_key, labels in sorted(labels_by_row.items()):
        if len(labels) > 1:
            label_conflicts.append(
                {
                    "population_id": row_key[0],
                    "row_id": row_key[1],
                    "labels": [
                        {"value": label_values[key], "record_indices": indices}
                        for key, indices in sorted(labels.items())
                    ],
                }
            )

    if expectations is None:
        inferred = {
            (record["population_id"], record["method_id"], record["budget"])
            for record in valid_records
            if record["population_id"] in populations and record["method_id"] in methods
        }
        expectation_rows: list[Mapping[str, Any]] = [
            {"population_id": population_id, "method_id": method_id, "budget": budget}
            for population_id, method_id, budget in sorted(
                inferred, key=lambda item: (item[0], item[1], _budget_key(item[2]))
            )
        ]
    else:
        expectation_rows = list(expectations)

    coverage: list[dict[str, Any]] = []
    expectation_problems: list[dict[str, Any]] = []
    expected_groups: set[tuple[str, str, Any]] = set()
    wildcard_budget_groups: set[tuple[str, str]] = set()
    for expectation_index, expectation in enumerate(expectation_rows):
        try:
            _require_fields(
                expectation,
                ("population_id", "method_id", "budget"),
                where=f"expectation {expectation_index}",
            )
            population_id = expectation["population_id"]
            method_id = expectation["method_id"]
            budget = expectation["budget"]
            if population_id not in populations:
                raise RegistryError(f"unknown population_id {population_id!r}")
            if method_id not in methods:
                raise RegistryError(f"unknown method_id {method_id!r}")
            if budget is not None:
                if isinstance(budget, bool) or not isinstance(budget, (int, str)):
                    raise RegistryError(
                        "budget must be a non-negative integer, non-empty string, or null"
                    )
                if isinstance(budget, int) and budget < 0:
                    raise RegistryError("integer budget must be non-negative")
                if isinstance(budget, str) and not budget:
                    raise RegistryError("string budget must be non-empty")
            match_any_budget = bool(expectation.get("match_any_budget", False))
            if match_any_budget:
                wildcard_budget_groups.add((population_id, method_id))
            else:
                expected_groups.add((population_id, method_id, budget))

            population = populations[population_id]
            if "eligible_row_ids" in expectation:
                eligible = _validate_ids(
                    expectation["eligible_row_ids"], field="eligible_row_ids"
                )
                registered = set(population["ordered_ids"])
                outside = [row_id for row_id in eligible if row_id not in registered]
                if outside:
                    raise RegistryError(
                        f"eligible_row_ids contains IDs outside {population_id}: {outside[:5]}"
                    )
                # Eligibility is an ordered subset of the canonical population,
                # not a second opportunity to choose an order.
                eligible_set = set(eligible)
                canonical_eligible = [
                    row_id for row_id in population["ordered_ids"] if row_id in eligible_set
                ]
                if eligible != canonical_eligible:
                    raise RegistryError(
                        "eligible_row_ids must preserve the registered population order"
                    )
            else:
                eligible = list(population["ordered_ids"])
            eligible_hash = ordered_id_sha256(eligible)
            declared_eligible_hash = expectation.get("eligible_ordered_id_sha256")
            if (
                declared_eligible_hash is not None
                and declared_eligible_hash != eligible_hash
            ):
                raise RegistryError(
                    "eligible_ordered_id_sha256 does not match eligible_row_ids"
                )

            observed = [
                record["row_id"]
                for record in valid_records
                if record["population_id"] == population_id
                and record["method_id"] == method_id
                and (match_any_budget or record["budget"] == budget)
            ]
            if match_any_budget:
                # Realized budgets are outcomes and canonical record sorting places
                # budget before row position.  Recover the only meaningful order for
                # this wildcard join: the independently registered population order.
                observed_counts = Counter(observed)
                registered_order = list(population["ordered_ids"])
                registered_set = set(registered_order)
                observed = [
                    row_id
                    for row_id in registered_order
                    for _copy in range(observed_counts.get(row_id, 0))
                ] + [
                    row_id for row_id in observed if row_id not in registered_set
                ]
            expected_set = set(eligible)
            observed_set = set(observed)
            missing = [row_id for row_id in eligible if row_id not in observed_set]
            extra = [row_id for row_id in observed if row_id not in expected_set]
            observed_unique = list(dict.fromkeys(observed))
            require_complete = bool(expectation.get("require_complete", True))
            require_order = bool(expectation.get("require_order", True))
            complete = not missing and not extra and len(observed) == len(eligible)
            order_matches = observed == eligible
            coverage.append(
                {
                    "table_id": expectation.get("table_id"),
                    "population_id": population_id,
                    "lane": population["lane"],
                    "method_id": method_id,
                    "budget": budget,
                    "match_any_budget": match_any_budget,
                    "observed_order_basis": (
                        "registered_population_order"
                        if match_any_budget
                        else "canonical_record_order"
                    ),
                    "headline": bool(expectation.get("headline", True)),
                    "require_complete": require_complete,
                    "require_order": require_order,
                    "n_expected": len(eligible),
                    "n_observed": len(observed),
                    "n_unique_observed": len(observed_unique),
                    "coverage": len(observed_set.intersection(expected_set)) / len(eligible)
                    if eligible
                    else 1.0,
                    "missing_row_ids": missing,
                    "extra_row_ids": extra,
                    "expected_ordered_id_sha256": eligible_hash,
                    "observed_ordered_id_sha256": ordered_id_sha256(observed_unique),
                    "complete": complete,
                    "order_matches": order_matches,
                    "passes": (complete or not require_complete)
                    and (order_matches or not require_order),
                }
            )
        except (RegistryError, TypeError) as exc:
            expectation_problems.append(
                {"expectation_index": expectation_index, "problem": str(exc)}
            )

    observed_groups = {
        (record["population_id"], record["method_id"], record["budget"])
        for record in valid_records
    }
    unexpected_groups = [
        {"population_id": key[0], "method_id": key[1], "budget": key[2]}
        for key in sorted(
            {
                key
                for key in observed_groups.difference(expected_groups)
                if (key[0], key[1]) not in wildcard_budget_groups
            },
            key=lambda item: (item[0], item[1], _budget_key(item[2])),
        )
    ] if expectations is not None else []

    structural_problems = any(
        (
            missing_or_invalid,
            unknown_populations,
            unknown_methods,
            unknown_rows,
            metadata_conflicts,
            lane_conflicts,
            artifact_hash_conflicts,
            duplicates,
            label_conflicts,
            expectation_problems,
            unexpected_groups,
        )
    )
    headline_coverage = [row for row in coverage if row["headline"]]
    headline_ok = not structural_problems and all(row["passes"] for row in headline_coverage)
    all_expected_ok = not structural_problems and all(row["passes"] for row in coverage)
    report = {
        "schema": JOIN_AUDIT_SCHEMA,
        "ok": all_expected_ok,
        "headline_ok": headline_ok,
        "n_input_records": len(raw_records),
        "n_valid_records": len(valid_records),
        "population_registry_sha256": validate_population_registry(population_registry)[
            "content_sha256"
        ],
        "method_registry_sha256": validate_method_registry(method_registry)["content_sha256"],
        "missing_or_invalid_records": missing_or_invalid,
        "unknown_populations": unknown_populations,
        "unknown_methods": unknown_methods,
        "unknown_rows": unknown_rows,
        "metadata_conflicts": metadata_conflicts,
        "lane_conflicts": lane_conflicts,
        "artifact_hash_conflicts": artifact_hash_conflicts,
        "duplicates": duplicates,
        "label_conflicts": label_conflicts,
        "expectation_problems": expectation_problems,
        "unexpected_groups": unexpected_groups,
        "coverage": coverage,
    }
    report["audit_sha256"] = canonical_sha256(report)
    return report


def require_clean_join(report: Mapping[str, Any], *, headline_only: bool = False) -> None:
    """Raise with a compact diagnosis unless the strict join gate passed."""

    field = "headline_ok" if headline_only else "ok"
    if report.get("schema") != JOIN_AUDIT_SCHEMA:
        raise JoinAuditError("not a comparison_join_audit_v1 report")
    if report.get(field) is True:
        return
    counts = {
        key: len(report.get(key, []))
        for key in (
            "missing_or_invalid_records",
            "unknown_populations",
            "unknown_methods",
            "unknown_rows",
            "metadata_conflicts",
            "lane_conflicts",
            "artifact_hash_conflicts",
            "duplicates",
            "label_conflicts",
            "expectation_problems",
            "unexpected_groups",
        )
        if report.get(key)
    }
    failed_coverage = sum(
        1
        for row in report.get("coverage", [])
        if (not headline_only or row.get("headline")) and not row.get("passes")
    )
    counts["failed_coverage"] = failed_coverage
    raise JoinAuditError(f"identical-row join gate failed: {counts}")


def _safe_relative_file(root: Path, candidate: Path) -> tuple[Path, str]:
    if candidate.is_symlink():
        raise RegistryError(f"symlink cannot enter hash manifest: {candidate}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RegistryError(f"manifest path {candidate} resolves outside root {root}") from exc
    if not resolved.is_file():
        raise RegistryError(f"manifest path is not a regular file: {candidate}")
    return resolved, relative.as_posix()


def build_hash_manifest(
    root: os.PathLike[str] | str,
    *,
    include: Optional[Sequence[os.PathLike[str] | str]] = None,
    exclude: Sequence[str] = ("HASH_MANIFEST.json",),
) -> dict[str, Any]:
    """Hash a result tree in lexical path order.

    With ``include=None`` every regular file under ``root`` is included except
    exact relative paths in ``exclude``.  Explicit include entries may be files
    or directories, but they must resolve below ``root``.  Symlinks are rejected
    so a manifest cannot silently bind content outside the package.
    """

    base = Path(root).resolve()
    if not base.is_dir():
        raise RegistryError(f"hash-manifest root is not a directory: {base}")
    excluded = set(exclude)
    candidates: list[Path] = []
    if include is None:
        symlinks = [path for path in base.rglob("*") if path.is_symlink()]
        if symlinks:
            raise RegistryError(f"symlink cannot enter hash manifest: {symlinks[0]}")
        candidates = [path for path in base.rglob("*") if path.is_file() and not path.is_symlink()]
        scope = "all-files-except-exclude"
    else:
        scope = "explicit-files"
        for item in include:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = base / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(base)
            except ValueError as exc:
                raise RegistryError(f"included path resolves outside root: {candidate}") from exc
            if resolved.is_symlink():
                raise RegistryError(f"symlink cannot enter hash manifest: {candidate}")
            if resolved.is_dir():
                candidates.extend(
                    path for path in resolved.rglob("*") if path.is_file() and not path.is_symlink()
                )
            elif resolved.is_file():
                candidates.append(resolved)
            else:
                raise RegistryError(f"included path does not exist: {candidate}")

    files_by_path: dict[str, Path] = {}
    for candidate in candidates:
        resolved, relative = _safe_relative_file(base, candidate)
        if relative in excluded:
            continue
        files_by_path[relative] = resolved
    file_rows = [
        {
            "path": relative,
            "size_bytes": files_by_path[relative].stat().st_size,
            "sha256": sha256_file(files_by_path[relative]),
        }
        for relative in sorted(files_by_path)
    ]
    return {
        "schema": HASH_MANIFEST_SCHEMA,
        "hash_algorithm": HASH_ALGORITHM,
        "scope": scope,
        "excluded_paths": sorted(excluded),
        "files": file_rows,
        "tree_sha256": canonical_sha256(file_rows),
    }


def verify_hash_manifest(
    root: os.PathLike[str] | str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify file hashes, sizes, tree hash, and (for all-file scope) extras."""

    base = Path(root).resolve()
    problems: list[dict[str, Any]] = []
    if manifest.get("schema") != HASH_MANIFEST_SCHEMA:
        problems.append({"problem": "schema", "observed": manifest.get("schema")})
    if manifest.get("hash_algorithm") != HASH_ALGORITHM:
        problems.append({"problem": "hash_algorithm", "observed": manifest.get("hash_algorithm")})
    rows = manifest.get("files")
    if not isinstance(rows, list):
        rows = []
        problems.append({"problem": "files must be a list"})
    paths = [row.get("path") for row in rows if isinstance(row, Mapping)]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        problems.append({"problem": "file paths are not unique lexical order"})
    if manifest.get("tree_sha256") != canonical_sha256(rows):
        problems.append({"problem": "tree_sha256 mismatch"})

    for row in rows:
        if not isinstance(row, Mapping):
            problems.append({"problem": "non-mapping file row"})
            continue
        try:
            _require_fields(row, ("path", "size_bytes", "sha256"), where="hash file row")
            candidate = base / row["path"]
            resolved, relative = _safe_relative_file(base, candidate)
            if relative != row["path"]:
                raise RegistryError("path is not canonical relative POSIX form")
            observed_size = resolved.stat().st_size
            observed_hash = sha256_file(resolved)
            if observed_size != row["size_bytes"]:
                problems.append(
                    {
                        "path": row["path"],
                        "problem": "size mismatch",
                        "expected": row["size_bytes"],
                        "observed": observed_size,
                    }
                )
            if observed_hash != row["sha256"]:
                problems.append(
                    {
                        "path": row["path"],
                        "problem": "sha256 mismatch",
                        "expected": row["sha256"],
                        "observed": observed_hash,
                    }
                )
        except (RegistryError, OSError, TypeError) as exc:
            problems.append({"path": row.get("path"), "problem": str(exc)})

    if manifest.get("scope") == "all-files-except-exclude" and base.is_dir():
        excluded = set(manifest.get("excluded_paths", []))
        actual = {
            path.resolve().relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file() and not path.is_symlink()
            and path.resolve().relative_to(base).as_posix() not in excluded
        }
        expected = set(paths)
        extra = sorted(actual.difference(expected))
        missing_from_scope = sorted(expected.difference(actual))
        if extra:
            problems.append({"problem": "unexpected files", "paths": extra})
        if missing_from_scope:
            problems.append({"problem": "manifest files absent from scope", "paths": missing_from_scope})

    return {
        "schema": "fair_comparison_hash_verification_v1",
        "ok": not problems,
        "n_files": len(rows),
        "tree_sha256": manifest.get("tree_sha256"),
        "problems": problems,
    }


__all__ = [
    "ACCESS_FIELDS",
    "ASSET_REGISTRY_SCHEMA",
    "ASSET_SCHEMA",
    "COMPARISON_SCHEMA",
    "FIDELITY_LABELS",
    "HASH_MANIFEST_SCHEMA",
    "JOIN_AUDIT_SCHEMA",
    "JoinAuditError",
    "LANES",
    "METHOD_SCHEMA",
    "ORDER_HASH_ENCODING",
    "POPULATION_SCHEMA",
    "RegistryError",
    "audit_comparison_records",
    "build_asset_registry",
    "build_hash_manifest",
    "build_method_registry",
    "build_population_registry",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonicalize_comparison_records",
    "is_sha256",
    "make_asset_record",
    "make_comparison_record",
    "make_derived_asset_record",
    "make_eligible_population",
    "make_method_entry",
    "make_population_entry",
    "method_definition_sha256",
    "method_index",
    "ordered_id_sha256",
    "population_index",
    "require_clean_join",
    "sha256_bytes",
    "sha256_file",
    "source_artifacts_sha256",
    "validate_asset_record",
    "validate_asset_registry",
    "validate_comparison_record",
    "validate_eligible_population",
    "validate_method_entry",
    "validate_method_registry",
    "validate_population_entry",
    "validate_population_registry",
    "verify_hash_manifest",
    "write_canonical_json",
]
