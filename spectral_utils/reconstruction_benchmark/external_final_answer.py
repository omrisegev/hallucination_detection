"""Strict external final-answer reconstruction boundary.

The external populations were collected by several historical pipelines.  This
module gives them one fail-closed interface without pretending that every saved
cache satisfies the same feature contract.  Fitting code can receive only a
target-free :class:`PreparedCell`; labels are opened by a separate function and
only after a complete score-freeze manifest has been verified.

The contract is intentionally stricter than older application scripts.  All 30
nominal views are audited: a present view must be finite on every registered
row, while a uniformly unavailable view may remain absent.  Partial
availability, convenience column selection, row selection, median imputation,
raw/pre-warper substitution, and a second orientation pass are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
import pickle
import re
import secrets
import stat
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import CONTRACT_VERSION, PreparedCell, prepared_matrix_sha256
from .external_fit_contract import (
    build_fit_row_identity_contract,
    fit_row_roster_sha256,
    load_prepared_external_cell as load_fit_prepared_external_cell,
    row_namespace_sha256 as fit_row_namespace_sha256,
)
from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from ..dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from ..fair_comparisons.stopping import grade_aqua_option
from ..repgrid_scoring import _candidate_features, logprob_features_extended
from ..specrage_views import FEATURE_TO_VIEW


SCHEMA_VERSION = "reconstruction-external-final-answer-registry-v2"
PREPARED_SCHEMA_VERSION = "reconstruction-external-target-free-input-v2"
AUDIT_SCHEMA_VERSION = "reconstruction-external-applicability-v2"
LABEL_SCHEMA_VERSION = "reconstruction-external-label-vector-v2"
SCORE_FREEZE_SCHEMA_VERSION = "reconstruction-external-score-freeze-v2"
ID_CONTRACT_VERSION = "reconstruction-external-keyed-hmac-id-v1"
IDENTITY_KEY_CONTRACT_VERSION = "reconstruction-external-identity-key-v1"
IDENTITY_KEY_BYTES = 32
RAW_ID_FINGERPRINT_VERSION = "reconstruction-external-raw-id-fingerprint-v1"
ID_DIGEST_ALGORITHM = "hmac-sha256-canonical-json-v1"
IDENTITY_KEY_ID_PREFIX = "xkidv1_"
OPAQUE_ROW_ID_PREFIX = "xridv2_"
OPAQUE_GROUP_ID_PREFIX = "xgidv2_"
RAW_ID_FINGERPRINT_PREFIX = "xrfpv1_"
_IDENTITY_KEY_ID_RE = re.compile(r"^xkidv1_[0-9a-f]{64}$")
_OPAQUE_ROW_ID_RE = re.compile(r"^xridv2_[0-9a-f]{64}$")
_OPAQUE_GROUP_ID_RE = re.compile(r"^xgidv2_[0-9a-f]{64}$")
_RAW_ID_FINGERPRINT_RE = re.compile(r"^xrfpv1_[0-9a-f]{64}$")
CANONICAL_FEATURE_NAMES = tuple(FEATURE_TO_VIEW)

LFS_MARKER = b"version https://git-lfs.github.com/spec/v1\n"
TARGET_LIKE_FRAGMENTS = (
    "label",
    "target",
    "correct",
    "answer_key",
    "gold",
    "auc",
    "auprc",
)


class ReadinessStatus(str, Enum):
    READY_FOR_TARGET_FREE_PREPARATION = "READY_FOR_TARGET_FREE_PREPARATION"
    ELIGIBLE = "ELIGIBLE"
    BLOCKED_ASSET = "BLOCKED_ASSET"
    SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
    SOURCE_INVENTORY_MISMATCH = "SOURCE_INVENTORY_MISMATCH"
    ROW_CONTRACT_MISMATCH = "ROW_CONTRACT_MISMATCH"
    LABEL_PROVENANCE_BLOCKED = "LABEL_PROVENANCE_BLOCKED"
    INCOMPATIBLE_FEATURE_CONTRACT = "INCOMPATIBLE_FEATURE_CONTRACT"
    PROTOCOL_GATE_FAILED = "PROTOCOL_GATE_FAILED"
    QUARANTINED = "QUARANTINED"


class ExternalContractError(RuntimeError):
    def __init__(self, status: ReadinessStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    sha256: str
    size_bytes: int | None = None
    role: str = "telemetry"


@dataclass(frozen=True)
class ExternalCellSpec:
    cell_id: str
    population_id: str
    dataset_id: str
    model_id: str
    slice_id: str
    domain: str
    comparison_group_id: str
    expected_rows: int
    adapter_id: str
    fit_policy: str
    panel_role: str
    source: Mapping[str, Any]
    expected_incorrect: int | None = None
    expected_correct: int | None = None
    expected_group_count: int | None = None
    excluded_raw_id_fingerprints: tuple[str, ...] = ()
    configured_status: str | None = None
    status_reason: str | None = None
    known_contract_risk: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalCellSpec":
        required = {
            "cell_id", "population_id", "dataset_id", "model_id", "slice_id",
            "domain", "comparison_group_id", "expected_rows", "adapter_id",
            "fit_policy", "panel_role",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"external cell is missing fields: {missing}")
        fit_policy = str(value["fit_policy"])
        source = value.get("source", {})
        if fit_policy == "run_if_compatible" and not source:
            raise ValueError(f"{value['cell_id']}: runnable cell lacks a source")
        return cls(
            cell_id=str(value["cell_id"]),
            population_id=str(value["population_id"]),
            dataset_id=str(value["dataset_id"]),
            model_id=str(value["model_id"]),
            slice_id=str(value["slice_id"]),
            domain=str(value["domain"]),
            comparison_group_id=str(value["comparison_group_id"]),
            expected_rows=int(value["expected_rows"]),
            adapter_id=str(value["adapter_id"]),
            fit_policy=fit_policy,
            panel_role=str(value["panel_role"]),
            source=dict(source),
            expected_incorrect=_optional_int(value.get("expected_incorrect")),
            expected_correct=_optional_int(value.get("expected_correct")),
            expected_group_count=_optional_int(value.get("expected_group_count")),
            excluded_raw_id_fingerprints=tuple(
                map(str, value.get("excluded_raw_id_fingerprints", ()))
            ),
            configured_status=(
                None if value.get("configured_status") is None
                else str(value["configured_status"])
            ),
            status_reason=(
                None if value.get("status_reason") is None
                else str(value["status_reason"])
            ),
            known_contract_risk=(
                None if value.get("known_contract_risk") is None
                else str(value["known_contract_risk"])
            ),
        )


@dataclass(frozen=True)
class ExternalRegistry:
    path: Path
    sha256: str
    population_registry_path: Path
    population_registry_sha256: str
    raw: Mapping[str, Any]
    cells: tuple[ExternalCellSpec, ...]

    @property
    def by_cell(self) -> dict[str, ExternalCellSpec]:
        return {cell.cell_id: cell for cell in self.cells}


@dataclass(frozen=True)
class OpaqueIdentityRoster:
    """Opaque, namespace-bound identities that are safe to expose to fitting."""

    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    contract_binding: Mapping[str, Any]
    row_namespace_sha256: str
    group_namespace_sha256: str


@dataclass(frozen=True)
class ResolvedSources:
    feature_files: tuple[SourceFile, ...]
    label_files: tuple[SourceFile, ...]
    provenance_files: tuple[SourceFile, ...]
    inventory_bindings: tuple[Mapping[str, Any], ...]

    @property
    def all_files(self) -> tuple[SourceFile, ...]:
        unique: dict[str, SourceFile] = {}
        for item in (*self.feature_files, *self.label_files, *self.provenance_files):
            unique[item.relative_path] = item
        return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class RawFeatureCell:
    spec: ExternalCellSpec
    raw_matrix: np.ndarray
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_files: tuple[SourceFile, ...]
    feature_names: tuple[str, ...] = CANONICAL_FEATURE_NAMES
    preprocessing_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.raw_matrix, dtype=float)
        names = tuple(map(str, self.feature_names))
        expected_names = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
        if names != expected_names or len(set(names)) != len(names):
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: present features are not an ordered subset of the nominal frozen 30",
            )
        if len(names) < 3:
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: fewer than three fully present features",
            )
        if matrix.shape != (self.spec.expected_rows, len(names)):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: raw matrix shape {matrix.shape} does not equal "
                f"({self.spec.expected_rows}, {len(names)})",
            )
        if self.preprocessing_steps:
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: source claims prior preprocessing; mixed-v2 must run once",
            )
        if not np.isfinite(matrix).all():
            locations = np.argwhere(~np.isfinite(matrix))[:10].tolist()
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{self.spec.cell_id}: present-feature matrix has non-finite values at {locations}",
            )
        rows = tuple(map(str, self.row_ids))
        groups = tuple(map(str, self.group_ids))
        if len(rows) != len(groups) or len(rows) != len(matrix):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: row/group/matrix lengths disagree",
            )
        if any(not value for value in rows) or len(set(rows)) != len(rows):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: row IDs are empty or duplicated",
            )
        if any(not value for value in groups):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: group IDs must be nonempty",
            )
        if (
            self.spec.expected_group_count is not None
            and len(set(groups)) != self.spec.expected_group_count
        ):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.spec.cell_id}: expected {self.spec.expected_group_count} groups, "
                f"found {len(set(groups))}",
            )
        matrix = np.array(matrix, dtype=np.float64, order="C", copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "raw_matrix", matrix)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "feature_names", names)


@dataclass(frozen=True)
class LabelVector:
    cell_id: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    incorrect: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        labels = np.asarray(self.incorrect, dtype=np.int8)
        if labels.shape != (len(self.row_ids),):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{self.cell_id}: label and row counts disagree",
            )
        if not np.isin(labels, (0, 1)).all():
            raise ExternalContractError(
                ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
                f"{self.cell_id}: labels are not binary incorrect indicators",
            )
        labels = np.array(labels, dtype=np.int8, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "incorrect", labels)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def raw_id_fingerprint(value: Any) -> str:
    """Return a one-way fingerprint for a raw source identifier.

    These fingerprints are used only to register source-row exclusions.  They
    deliberately use a domain that differs from both prepared row IDs and
    prepared group IDs, so a fingerprint can never be substituted for either.
    """

    raw = str(value)
    if not raw:
        raise ValueError("raw source identity must be nonempty")
    digest = _payload_hash({
        "contract_version": RAW_ID_FINGERPRINT_VERSION,
        "domain": "external_raw_row_identity_exclusion",
        "raw_identity": raw,
    })
    return RAW_ID_FINGERPRINT_PREFIX + digest


def _external_id_contract_template(registry: ExternalRegistry) -> dict[str, Any]:
    """Validate the registry's public keyed-identity template."""

    configured = registry.raw.get("identity_contract")
    if not isinstance(configured, Mapping):
        raise ValueError("external registry lacks its opaque identity contract")
    rules = configured.get("group_namespace_by_population")
    if not isinstance(rules, Mapping):
        raise ValueError("external identity contract lacks group namespace rules")
    normalized_rules = {str(key): str(value) for key, value in rules.items()}
    populations = {
        cell.population_id
        for cell in registry.cells
        if cell.fit_policy == "run_if_compatible"
    }
    if set(normalized_rules) != populations:
        raise ValueError(
            "external group namespace rules must cover the exact population roster"
        )
    expected = {
        "version": ID_CONTRACT_VERSION,
        "digest_algorithm": ID_DIGEST_ALGORITHM,
        "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
        "identity_key_bytes": IDENTITY_KEY_BYTES,
        "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
        "opaque_group_id_prefix": OPAQUE_GROUP_ID_PREFIX,
        "row_namespace_scope": "cell",
        "canonical_row_order": "lexicographic_opaque_row_id",
        "group_namespace_by_population": normalized_rules,
    }
    observed = {
        key: configured.get(key)
        for key in expected
        if key != "group_namespace_by_population"
    }
    observed["group_namespace_by_population"] = normalized_rules
    if observed != expected:
        raise ValueError("external opaque identity contract disagrees with executable v1")

    aggregates = registry.raw.get("population_aggregates", {})
    scope_for_link = {
        "none": "cell",
        "slice_id": "population_slice",
        "all": "population",
    }
    for population_id in sorted(populations):
        aggregate = aggregates.get(population_id, {})
        link_rule = str(aggregate.get("link_cells_by", "none"))
        expected_scope = scope_for_link.get(link_rule)
        if expected_scope is None:
            raise ValueError(
                f"{population_id}: unsupported aggregate linkage for opaque IDs"
            )
        if normalized_rules[population_id] != expected_scope:
            raise ValueError(
                f"{population_id}: opaque group namespace does not match "
                f"link_cells_by={link_rule!r}"
            )
    return expected


def identity_key_id(identity_key: bytes) -> str:
    """Return a public, domain-separated commitment to a sealed release key."""

    key = bytes(identity_key)
    if len(key) != IDENTITY_KEY_BYTES:
        raise ValueError(f"external identity key must be exactly {IDENTITY_KEY_BYTES} bytes")
    digest = hashlib.sha256(
        IDENTITY_KEY_CONTRACT_VERSION.encode("utf-8")
        + b"\0external_identity_key_id\0"
        + key
    ).hexdigest()
    return IDENTITY_KEY_ID_PREFIX + digest


def load_identity_key(path: str | Path, *, create: bool = False) -> bytes:
    """Load (or exclusively create) the controller-only release identity key.

    The raw 32-byte key must live outside every fit-mounted release tree.  Only
    preparation and post-freeze label joining call this function.  Fitting sees
    the public ``key_id`` commitment, never this path or its bytes.
    """

    target = Path(path)
    if create:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            key = secrets.token_bytes(IDENTITY_KEY_BYTES)
            try:
                written = os.write(descriptor, key)
                if written != len(key):  # pragma: no cover - defensive short write
                    raise OSError("short write while creating external identity key")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    try:
        metadata = target.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"sealed external identity key is absent: {target}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("external identity key must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PermissionError("external identity key must not be group/world accessible")
    key = target.read_bytes()
    if len(key) != IDENTITY_KEY_BYTES:
        raise RuntimeError(
            f"external identity key must contain exactly {IDENTITY_KEY_BYTES} raw bytes"
        )
    return key


def external_id_contract_binding(
    registry: ExternalRegistry,
    *,
    identity_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Return the registry and sealed-release-key-bound public ID contract."""

    if (identity_key is None) == (key_id is None):
        raise ValueError("provide exactly one of identity_key or key_id")
    resolved_key_id = identity_key_id(identity_key) if identity_key is not None else str(key_id)
    if _IDENTITY_KEY_ID_RE.fullmatch(resolved_key_id) is None:
        raise ValueError("external identity key_id is malformed")
    binding = _external_id_contract_template(registry)
    binding["key_id"] = resolved_key_id
    binding["contract_sha256"] = _payload_hash(binding)
    return binding


def validate_public_identity_contract(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a fit-safe public binding without opening the full registry."""

    value = dict(binding)
    rules = value.get("group_namespace_by_population")
    if not isinstance(rules, Mapping) or not rules:
        raise RuntimeError("fit-safe identity contract lacks group namespace rules")
    normalized_rules = {str(key): str(scope) for key, scope in rules.items()}
    if any(scope not in {"cell", "population_slice", "population"} for scope in normalized_rules.values()):
        raise RuntimeError("fit-safe identity contract has an invalid group namespace")
    expected = {
        "version": ID_CONTRACT_VERSION,
        "digest_algorithm": ID_DIGEST_ALGORITHM,
        "identity_key_contract_version": IDENTITY_KEY_CONTRACT_VERSION,
        "identity_key_bytes": IDENTITY_KEY_BYTES,
        "opaque_row_id_prefix": OPAQUE_ROW_ID_PREFIX,
        "opaque_group_id_prefix": OPAQUE_GROUP_ID_PREFIX,
        "row_namespace_scope": "cell",
        "canonical_row_order": "lexicographic_opaque_row_id",
        "group_namespace_by_population": normalized_rules,
        "key_id": value.get("key_id"),
    }
    if _IDENTITY_KEY_ID_RE.fullmatch(str(expected["key_id"])) is None:
        raise RuntimeError("fit-safe identity contract key_id is malformed")
    observed = {key: value.get(key) for key in expected}
    if observed != expected or set(value) != {*expected, "contract_sha256"}:
        raise RuntimeError("fit-safe opaque identity contract disagrees with executable v1")
    if value.get("contract_sha256") != _payload_hash(expected):
        raise RuntimeError("fit-safe opaque identity contract hash failed")
    return value


def _identity_namespace(
    binding: Mapping[str, Any],
    *,
    kind: str,
    cell_id: str,
    population_id: str,
    slice_id: str,
) -> dict[str, str]:
    binding = validate_public_identity_contract(binding)
    if kind == "row":
        return {
            "contract_version": ID_CONTRACT_VERSION,
            "identity_kind": "row",
            "scope": "cell",
            "cell_id": cell_id,
        }
    if kind != "group":
        raise ValueError(f"unknown external identity kind: {kind!r}")
    try:
        scope = str(binding["group_namespace_by_population"][population_id])
    except KeyError as error:
        raise RuntimeError(
            f"fit-safe identity contract omits population {population_id!r}"
        ) from error
    namespace = {
        "contract_version": ID_CONTRACT_VERSION,
        "identity_kind": "group",
        "scope": scope,
        "population_id": population_id,
    }
    if scope == "cell":
        namespace["cell_id"] = cell_id
    elif scope == "population_slice":
        namespace["slice_id"] = slice_id
    elif scope != "population":  # pragma: no cover - validated above
        raise AssertionError(f"unvalidated identity namespace scope: {scope}")
    return namespace


def keyed_opaque_external_id(
    *,
    identity_key: bytes,
    kind: str,
    namespace: Mapping[str, str],
    raw: Any,
) -> str:
    value = str(raw)
    if not value:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"external {kind} source identity must be nonempty",
        )
    prefix = OPAQUE_ROW_ID_PREFIX if kind == "row" else OPAQUE_GROUP_ID_PREFIX
    payload = {
        "contract_version": ID_CONTRACT_VERSION,
        "domain": f"external_prepared_{kind}_identity",
        "namespace": dict(namespace),
        "raw_identity": value,
    }
    digest = hmac.new(
        bytes(identity_key), canonical_json_bytes(payload), hashlib.sha256
    ).hexdigest()
    return prefix + digest


def assert_opaque_external_ids(
    row_ids: Sequence[str],
    group_ids: Sequence[str],
) -> None:
    rows = tuple(map(str, row_ids))
    groups = tuple(map(str, group_ids))
    if len(rows) != len(groups) or not rows:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            "opaque row/group rosters must be equal-length and nonempty",
        )
    if len(set(rows)) != len(rows) or any(_OPAQUE_ROW_ID_RE.fullmatch(x) is None for x in rows):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            "prepared row identities are not unique keyed opaque v2 identifiers",
        )
    if any(_OPAQUE_GROUP_ID_RE.fullmatch(x) is None for x in groups):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            "prepared group identities are not keyed opaque v2 identifiers",
        )


def apply_external_id_contract(
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    raw_row_ids: Sequence[str],
    raw_group_ids: Sequence[str],
    *,
    identity_key: bytes,
) -> OpaqueIdentityRoster:
    """HMAC every raw identity behind explicit row/group namespaces."""

    raw_rows = tuple(map(str, raw_row_ids))
    raw_groups = tuple(map(str, raw_group_ids))
    if len(raw_rows) != len(raw_groups) or len(set(raw_rows)) != len(raw_rows):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: raw identity roster is misaligned or duplicates rows",
        )
    contract_binding = external_id_contract_binding(
        registry, identity_key=identity_key
    )
    namespace_fields = {
        "cell_id": spec.cell_id,
        "population_id": spec.population_id,
        "slice_id": spec.slice_id,
    }
    row_namespace = _identity_namespace(
        contract_binding, kind="row", **namespace_fields
    )
    group_namespace = _identity_namespace(
        contract_binding, kind="group", **namespace_fields
    )
    rows = tuple(
        keyed_opaque_external_id(
            identity_key=identity_key, kind="row", namespace=row_namespace, raw=value
        )
        for value in raw_rows
    )
    groups = tuple(
        keyed_opaque_external_id(
            identity_key=identity_key, kind="group", namespace=group_namespace, raw=value
        )
        for value in raw_groups
    )
    assert_opaque_external_ids(rows, groups)
    return OpaqueIdentityRoster(
        row_ids=rows,
        group_ids=groups,
        contract_binding=contract_binding,
        row_namespace_sha256=_payload_hash(row_namespace),
        group_namespace_sha256=_payload_hash(group_namespace),
    )


def canonicalize_external_identity_order(
    matrix: np.ndarray,
    identity: OpaqueIdentityRoster,
) -> tuple[np.ndarray, OpaqueIdentityRoster]:
    """Order a fit cohort only by opaque row identity.

    Historical adapters often sort raw source keys, and some of those keys
    encode an error family.  Even after hashing the values, retaining that raw
    order would expose a label-correlated positional tie breaker to graph
    methods.  Sorting the rows by their opaque cryptographic IDs removes that
    channel while remaining deterministic and independent of source iteration
    order.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(identity.row_ids):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            "matrix and opaque identity roster disagree before canonical ordering",
        )
    order = _opaque_order_indices(identity.row_ids)
    rows = tuple(identity.row_ids[index] for index in order.tolist())
    groups = tuple(identity.group_ids[index] for index in order.tolist())
    assert_opaque_external_ids(rows, groups)
    if rows != tuple(sorted(rows)):
        raise AssertionError("opaque row canonicalization did not produce sorted IDs")
    ordered = np.array(values[order], dtype=np.float64, order="C", copy=True)
    ordered.setflags(write=False)
    return ordered, OpaqueIdentityRoster(
        row_ids=rows,
        group_ids=groups,
        contract_binding=identity.contract_binding,
        row_namespace_sha256=identity.row_namespace_sha256,
        group_namespace_sha256=identity.group_namespace_sha256,
    )


def _opaque_order_indices(row_ids: Sequence[str]) -> np.ndarray:
    rows = tuple(map(str, row_ids))
    return np.asarray(
        sorted(range(len(rows)), key=lambda index: rows[index]),
        dtype=np.int64,
    )


def load_external_registry(
    *,
    repo: str | Path,
    registry_path: str | Path,
    population_registry_path: str | Path,
) -> ExternalRegistry:
    root = Path(repo).resolve()
    registry_file = _resolve_under(root, registry_path)
    population_file = _resolve_under(root, population_registry_path)
    raw = _read_json(registry_file)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected external final-answer registry schema")
    if raw.get("feature_contract_id") != CONTRACT_VERSION:
        raise ValueError("external registry does not bind the frozen mixed-v2 contract")
    cells = tuple(ExternalCellSpec.from_mapping(item) for item in raw.get("cells", ()))
    if not cells or len({item.cell_id for item in cells}) != len(cells):
        raise ValueError("external cell registry is empty or contains duplicate IDs")
    for source_row, item in zip(raw.get("cells", ()), cells):
        if "excluded_row_ids" in source_row:
            raise ValueError(
                f"{item.cell_id}: raw exclusion IDs are forbidden in the v2 registry"
            )
        fingerprints = item.excluded_raw_id_fingerprints
        if len(set(fingerprints)) != len(fingerprints) or any(
            _RAW_ID_FINGERPRINT_RE.fullmatch(value) is None
            for value in fingerprints
        ):
            raise ValueError(
                f"{item.cell_id}: exclusion fingerprints are malformed or duplicated"
            )
    populations = _read_json(population_file)
    population_rows = {
        str(item["population_id"]): item
        for item in populations.get("populations", ())
    }
    known = set(population_rows)
    missing = sorted({item.population_id for item in cells} - known)
    if missing:
        raise ValueError("external cells reference unregistered populations: " + ", ".join(missing))
    aliases = {str(key): str(value) for key, value in raw.get("model_id_aliases", {}).items()}
    for item in cells:
        if item.fit_policy not in {"run_if_compatible", "forbidden"}:
            raise ValueError(f"{item.cell_id}: unknown fit_policy {item.fit_policy}")
        population = population_rows[item.population_id]
        declared_models = []
        if population.get("model") is not None:
            declared_models.append(str(population["model"]))
        declared_models.extend(map(str, population.get("scorer_models", ())))
        declared_models.extend(map(str, population.get("models", ())))
        if item.fit_policy == "run_if_compatible" and declared_models:
            unaliased = sorted(set(declared_models) - set(aliases))
            if unaliased:
                raise ValueError(
                    f"{item.population_id}: population model metadata lacks canonical aliases: "
                    + ", ".join(unaliased)
                )
            expected_model_ids = {aliases[value] for value in declared_models}
            if item.model_id not in expected_model_ids:
                raise ValueError(
                    f"{item.cell_id}: model_id {item.model_id!r} conflicts with population "
                    f"metadata {sorted(expected_model_ids)!r}"
                )
        if item.fit_policy == "forbidden":
            if item.configured_status not in {
                ReadinessStatus.PROTOCOL_GATE_FAILED.value,
                ReadinessStatus.QUARANTINED.value,
            }:
                raise ValueError(f"{item.cell_id}: forbidden cell lacks a terminal status")
        if item.fit_policy == "run_if_compatible" and item.source.get("kind") == "explicit":
            manifest_registry = raw.get("explicit_source_manifests", {})
            for source_file in item.source.get("files", ()):
                if source_file.get("role", "telemetry") != "telemetry":
                    continue
                manifest = str(Path(str(source_file["path"])).parent / "manifest.json")
                if manifest not in manifest_registry:
                    raise ValueError(
                        f"{item.cell_id}: explicit telemetry lacks a hash-frozen source manifest"
                    )
    expectations = raw.get("population_expectations", {})
    aggregates = raw.get("population_aggregates", {})
    if set(expectations) != set(aggregates):
        raise ValueError("population expectation and aggregate registries must have identical IDs")
    for population_id, expected in expectations.items():
        members = [item for item in cells if item.population_id == population_id]
        observed = {
            "rows": sum(item.expected_rows for item in members),
            "cells": len(members),
        }
        for key, value in observed.items():
            if int(expected.get(key, -1)) != value:
                raise ValueError(
                    f"{population_id}: external {key} total {value} conflicts with "
                    f"registered expectation {expected.get(key)!r}"
                )
        aggregate = aggregates[population_id]
        if aggregate.get("enabled") is True:
            if aggregate.get("weighting") not in {"single_cell", "equal_cell"}:
                raise ValueError(f"{population_id}: unsupported aggregate weighting")
            if aggregate.get("link_cells_by") not in {"none", "slice_id", "all"}:
                raise ValueError(f"{population_id}: unsupported cross-cell linkage rule")
    registry = ExternalRegistry(
        path=registry_file,
        sha256=sha256_file(registry_file),
        population_registry_path=population_file,
        population_registry_sha256=sha256_file(population_file),
        raw=raw,
        cells=cells,
    )
    _external_id_contract_template(registry)
    if raw.get("exclusion_fingerprint_contract") != {
        "version": RAW_ID_FINGERPRINT_VERSION,
        "digest_algorithm": "sha256-canonical-json-v1",
        "prefix": RAW_ID_FINGERPRINT_PREFIX,
        "controller_only": True,
    }:
        raise ValueError("external exclusion-fingerprint contract is stale")
    return registry


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes the repository root: {path}") from error
    return resolved


def _inventory(root: Path, entry: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    path = _resolve_under(root, str(entry["path"]))
    if not path.is_file():
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"source inventory is missing: {path}"
        )
    observed = sha256_file(path)
    if observed != str(entry["sha256"]):
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"source inventory hash mismatch for {path}: expected {entry['sha256']}, got {observed}",
        )
    return path, _read_json(path)


def _source_file(
    root: Path,
    relative: str,
    digest: str,
    *,
    size_bytes: int | None = None,
    role: str = "telemetry",
) -> SourceFile:
    return SourceFile(
        path=_resolve_under(root, relative),
        relative_path=str(relative),
        sha256=str(digest),
        size_bytes=_optional_int(size_bytes),
        role=role,
    )


def resolve_sources(
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    *,
    repo: str | Path,
) -> ResolvedSources:
    root = Path(repo).resolve()
    source = spec.source
    kind = str(source.get("kind", ""))
    feature: list[SourceFile] = []
    labels: list[SourceFile] = []
    provenance: list[SourceFile] = []
    bindings: list[Mapping[str, Any]] = []
    if kind == "explicit":
        for item in source.get("files", ()):
            role = str(item.get("role", "telemetry"))
            resolved = _source_file(
                root,
                str(item["path"]),
                str(item["sha256"]),
                size_bytes=item.get("size_bytes"),
                role=role,
            )
            if role == "telemetry":
                feature.append(resolved)
            elif role == "labels":
                labels.append(resolved)
            elif role in {"manifest", "inventory", "provenance"}:
                provenance.append(resolved)
            else:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: unsupported explicit source role {role!r}",
                )
        registered_manifests = registry.raw.get("explicit_source_manifests", {})
        for telemetry in feature:
            manifest_relative = str(Path(telemetry.relative_path).parent / "manifest.json")
            digest = registered_manifests.get(manifest_relative)
            if digest is not None:
                provenance.append(_source_file(
                    root, manifest_relative, str(digest), role="manifest"
                ))
    elif kind == "data_readiness":
        inv_entry = registry.raw["source_inventories"]["data_readiness"]
        inv_path, inventory = _inventory(root, inv_entry)
        candidates = [
            item for item in inventory.get("datasets", ())
            if str(item.get("dataset_id")) == str(source["dataset_id"])
        ]
        if len(candidates) != 1:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: data-readiness selector did not resolve exactly once",
            )
        row = candidates[0]
        hashes = {str(key): str(value) for key, value in row.get("file_hashes", {}).items()}
        suffix = str(source["path_suffix"])
        matches = [path for path in row.get("source_paths", ()) if str(path).endswith(suffix)]
        if len(matches) != 1 or matches[0] not in hashes:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: feature source suffix {suffix!r} is ambiguous or unhashed",
            )
        feature.append(_source_file(root, matches[0], hashes[matches[0]]))
        label_suffix = source.get("label_path_suffix")
        if label_suffix is not None:
            label_matches = [
                path for path in row.get("source_paths", ())
                if str(path).endswith(str(label_suffix))
            ]
            if len(label_matches) != 1 or label_matches[0] not in hashes:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: label source suffix is ambiguous or unhashed",
                )
            labels.append(_source_file(
                root, label_matches[0], hashes[label_matches[0]], role="labels"
            ))
        for relative in row.get("source_paths", ()):
            if (
                str(relative).endswith(("manifest.json", "_manifest.json"))
                and relative in hashes
            ):
                provenance.append(_source_file(
                    root, relative, hashes[relative], role="manifest"
                ))
        provenance.append(SourceFile(
            inv_path,
            inv_path.relative_to(root).as_posix(),
            str(inv_entry["sha256"]),
            inv_path.stat().st_size,
            "inventory",
        ))
        bindings.append({
            "inventory": inv_path.relative_to(root).as_posix(),
            "inventory_sha256": str(inv_entry["sha256"]),
            "selector": {"dataset_id": source["dataset_id"]},
        })
    elif kind == "aqua_external":
        inv_entry = registry.raw["source_inventories"]["aqua_external"]
        inv_path, inventory = _inventory(root, inv_entry)
        candidates = [
            item for item in inventory.get("cells", ())
            if str(item.get("cell")) == str(source["cell"])
        ]
        if len(candidates) != 1:
            raise ExternalContractError(
                ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                f"{spec.cell_id}: AQuA inventory selector did not resolve exactly once",
            )
        row = candidates[0]
        hashes = {str(key): str(value) for key, value in row.get("source_sha256", {}).items()}
        for relative in row.get("source_paths", ()):
            if relative not in hashes:
                raise ExternalContractError(
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                    f"{spec.cell_id}: AQuA source lacks an expected hash: {relative}",
                )
            item = _source_file(root, relative, hashes[relative])
            if str(relative).endswith(".pkl"):
                feature.append(item)
            else:
                provenance.append(SourceFile(
                    item.path, item.relative_path, item.sha256, item.size_bytes, "manifest"
                ))
        provenance.append(SourceFile(
            inv_path,
            inv_path.relative_to(root).as_posix(),
            str(inv_entry["sha256"]),
            inv_path.stat().st_size,
            "inventory",
        ))
        bindings.append({
            "inventory": inv_path.relative_to(root).as_posix(),
            "inventory_sha256": str(inv_entry["sha256"]),
            "selector": {"cell": source["cell"]},
        })
    else:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"{spec.cell_id}: unsupported source resolver {kind!r}",
        )
    if not feature:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
            f"{spec.cell_id}: no feature source resolved",
        )
    return ResolvedSources(tuple(feature), tuple(labels), tuple(provenance), tuple(bindings))


def _parse_lfs_pointer(path: Path) -> tuple[str, int] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        prefix = handle.read(256)
    if not prefix.startswith(LFS_MARKER):
        return None
    fields: dict[str, str] = {}
    for line in prefix.decode("utf-8", errors="strict").splitlines()[1:]:
        if " " in line:
            key, value = line.split(" ", 1)
            fields[key] = value
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:") or "size" not in fields:
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"malformed Git-LFS pointer: {path}"
        )
    return oid.removeprefix("sha256:"), int(fields["size"])


def verify_source_file(item: SourceFile) -> Mapping[str, Any]:
    if not item.path.is_file():
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET, f"missing source asset: {item.relative_path}"
        )
    pointer = _parse_lfs_pointer(item.path)
    if pointer is not None:
        oid, size = pointer
        if oid != item.sha256 or (item.size_bytes is not None and size != item.size_bytes):
            raise ExternalContractError(
                ReadinessStatus.SOURCE_HASH_MISMATCH,
                f"Git-LFS pointer metadata disagrees with the registry: {item.relative_path}",
            )
        raise ExternalContractError(
            ReadinessStatus.BLOCKED_ASSET,
            f"Git-LFS object is not materialized: {item.relative_path} ({oid}, {size} bytes)",
        )
    observed_size = item.path.stat().st_size
    if item.size_bytes is not None and observed_size != item.size_bytes:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_HASH_MISMATCH,
            f"source size mismatch for {item.relative_path}: expected {item.size_bytes}, got {observed_size}",
        )
    observed_hash = sha256_file(item.path)
    if observed_hash != item.sha256:
        raise ExternalContractError(
            ReadinessStatus.SOURCE_HASH_MISMATCH,
            f"source hash mismatch for {item.relative_path}: expected {item.sha256}, got {observed_hash}",
        )
    return {
        "path": item.relative_path,
        "sha256": observed_hash,
        "size_bytes": observed_size,
        "role": item.role,
        "materialized": True,
    }


def verify_sources(sources: ResolvedSources, *, include_labels: bool = False) -> list[Mapping[str, Any]]:
    selected = [*sources.feature_files, *sources.provenance_files]
    if include_labels:
        selected.extend(sources.label_files)
    return [verify_source_file(item) for item in selected]


def _stable_key(value: Any) -> tuple[int, int | str, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _telemetry_features(row: Mapping[str, Any], *, allow_short: bool) -> Mapping[str, float]:
    # Only telemetry members are copied.  Embedded answer/label fields never
    # enter this function's output or a fitting artifact.
    telemetry = {
        "token_entropies": row.get("token_entropies"),
        "token_spilled_energies": row.get("token_spilled_energies"),
        "token_logsumexp": row.get("token_logsumexp"),
        "top_k_logprobs": row.get("top_k_logprobs"),
    }
    output = dict(_candidate_features(telemetry, allow_short=allow_short))
    if telemetry["top_k_logprobs"] is not None:
        output.update(logprob_features_extended(telemetry["top_k_logprobs"]))
    return output


def _matrix_from_feature_rows(
    spec: ExternalCellSpec,
    rows: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if len(rows) != spec.expected_rows:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: expected {spec.expected_rows} rows, found {len(rows)}",
        )
    nominal = np.asarray([
        [row.get(name, np.nan) for name in CANONICAL_FEATURE_NAMES]
        for row in rows
    ], dtype=np.float64)
    if nominal.shape != (spec.expected_rows, len(CANONICAL_FEATURE_NAMES)):
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: cannot audit the nominal raw 30-feature roster",
        )
    finite_counts = np.sum(np.isfinite(nominal), axis=0)
    partial = {
        name: int(count)
        for name, count in zip(CANONICAL_FEATURE_NAMES, finite_counts)
        if 0 < int(count) < spec.expected_rows
    }
    if partial:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: partially available views are forbidden; "
            f"finite_rows_by_feature={partial}, expected_rows={spec.expected_rows}",
        )
    keep = finite_counts == spec.expected_rows
    names = tuple(
        name for name, present in zip(CANONICAL_FEATURE_NAMES, keep) if bool(present)
    )
    if len(names) < 3:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: only {len(names)} nominal views are fully present",
        )
    matrix = nominal[:, keep]
    if not np.isfinite(matrix).all():
        raise AssertionError("present-feature mask admitted a non-finite value")
    return matrix, names


def _processbench_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "ProcessBench needs one cache")
    cache = _load_pickle(sources.feature_files[0].path)
    if not isinstance(cache, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench cache is not a mapping")
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        key_fingerprint = raw_id_fingerprint(key)
        if not isinstance(row, Mapping):
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench row is not a mapping")
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: registered population contains alignment problems at "
                f"source fingerprint {key_fingerprint}: {problems}",
            )
        official_id = row.get("id")
        if not isinstance(official_id, str) or not official_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"ProcessBench source fingerprint {key_fingerprint} lacks its official string ID",
            )
        row_id = official_id
        feature_rows.append(_telemetry_features(row, allow_short=True))
        row_ids.append(row_id)
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(row_ids), sources.feature_files,
        feature_names=names,
    )


def _prmbench_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "PRMBench needs one cache")
    cache = _load_pickle(sources.feature_files[0].path)
    if not isinstance(cache, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "PRMBench cache is not a mapping")
    excluded = set(spec.excluded_raw_id_fingerprints)
    seen_excluded: set[str] = set()
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    groups: list[str] = []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        key_fingerprint = raw_id_fingerprint(key)
        row_id = str(row.get("idx", ""))
        if not row_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench source fingerprint {key_fingerprint} lacks idx",
            )
        fingerprint = raw_id_fingerprint(row_id)
        if fingerprint in excluded:
            seen_excluded.add(fingerprint)
            continue
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: unexpected non-preregistered alignment exclusion "
                f"{fingerprint}",
            )
        feature_rows.append(_telemetry_features(row, allow_short=True))
        row_ids.append(row_id)
        source_group = str(row.get("source_idx", ""))
        if not source_group:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench row fingerprint {fingerprint} lacks source_idx",
            )
        groups.append(source_group)
    if seen_excluded != excluded:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: preregistered exclusion fingerprints missing from source: "
            f"{sorted(excluded - seen_excluded)}",
        )
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(groups), sources.feature_files,
        feature_names=names,
    )


def _repgrid_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if len(sources.feature_files) != 1:
        raise ExternalContractError(ReadinessStatus.SOURCE_INVENTORY_MISMATCH, "repgrid cell needs one cache")
    data = _load_pickle(sources.feature_files[0].path)
    if not isinstance(data, Mapping):
        raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "repgrid cache is not a mapping")
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    groups: list[str] = []
    for key in sorted(data, key=_stable_key):
        entry = data[key]
        key_fingerprint = raw_id_fingerprint(key)
        candidates = entry.get("candidates") if isinstance(entry, Mapping) else None
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"{spec.cell_id}: source-group fingerprint {key_fingerprint} lacks a candidate sequence",
            )
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "candidate is not a mapping")
            feature_rows.append(_telemetry_features(candidate, allow_short=False))
            group = str(key)
            row_ids.append(f"{group}:candidate:{index:04d}")
            groups.append(group)
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(groups), sources.feature_files,
        feature_names=names,
    )


def _aqua_raw(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    records: list[Mapping[str, Any]] = []
    for item in sorted(sources.feature_files, key=lambda value: value.relative_path):
        shard = _load_pickle(item.path)
        if not isinstance(shard, Sequence) or isinstance(shard, (str, bytes)):
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "AQuA shard is not a row sequence")
        records.extend(shard)
    selected = [
        row for row in records
        if row.get("arm") == "cot" and row.get("setting_label") == "central"
    ]
    selected.sort(key=lambda row: _stable_key(str(row.get("question_id", "")).split(":")[-1]))
    if len(selected) != spec.expected_rows:
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: expected {spec.expected_rows} cot|central rows, found {len(selected)}",
        )
    feature_rows: list[Mapping[str, float]] = []
    row_ids: list[str] = []
    for row in selected:
        # The frozen detector was developed on canonical post-warper streams.
        # Historical AQuA shards often contain only raw/pre-warper streams and
        # a NaN sampled-entropy channel.  Reconstructing a surrogate from raw
        # top-k values would define a different method and is forbidden.
        channels = row.get("channels", {})
        entropy = channels.get("sampled_entropy")
        spilled = channels.get("spilled_energy")
        logsumexp = channels.get("logsumexp")
        topk = row.get("top_k_logprobs")
        if any(value is None for value in (entropy, spilled, logsumexp, topk)):
            raise ExternalContractError(
                ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
                f"{spec.cell_id}: canonical post-warper entropy/spilled/logsumexp/top-k streams are absent; raw substitution is forbidden",
            )
        candidate = {
            "token_entropies": entropy,
            "token_spilled_energies": spilled,
            "token_logsumexp": logsumexp,
            "top_k_logprobs": topk,
        }
        feature_rows.append(_telemetry_features(candidate, allow_short=False))
        row_ids.append(str(row.get("question_id", "")))
    matrix, names = _matrix_from_feature_rows(spec, feature_rows)
    return RawFeatureCell(
        spec, matrix, tuple(row_ids), tuple(row_ids), sources.feature_files,
        feature_names=names,
    )


RAW_ADAPTERS = {
    "processbench_teacher_forced_v1": _processbench_raw,
    "prmbench_teacher_forced_v1": _prmbench_raw,
    "repgrid_embedded_label_v1": _repgrid_raw,
    "hle_separate_judge_v1": _repgrid_raw,
    "aqua_paper_exact_cot_v1": _aqua_raw,
}


def load_raw_feature_cell(spec: ExternalCellSpec, sources: ResolvedSources) -> RawFeatureCell:
    if spec.fit_policy != "run_if_compatible":
        status = ReadinessStatus(str(spec.configured_status))
        raise ExternalContractError(status, spec.status_reason or f"{spec.cell_id} is not runnable")
    adapter = RAW_ADAPTERS.get(spec.adapter_id)
    if adapter is None:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: no target-free adapter for {spec.adapter_id}",
        )
    try:
        return adapter(spec, sources)
    except ExternalContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{spec.cell_id}: telemetry cannot reconstruct the frozen raw 30-feature contract: "
            f"{type(error).__name__}: {error}",
        ) from error


def apply_mixed_v2_once(raw: RawFeatureCell) -> tuple[np.ndarray, Mapping[str, Any]]:
    if raw.preprocessing_steps:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: preprocessing already present; double transform forbidden",
        )
    transformed, names, details = dufs_liu_mixed_v2_matrix(
        raw.raw_matrix, raw.feature_names
    )
    if tuple(names) != raw.feature_names or transformed.shape != raw.raw_matrix.shape:
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: mixed-v2 changed the frozen present-feature roster",
        )
    if not np.isfinite(transformed).all():
        raise ExternalContractError(
            ReadinessStatus.INCOMPATIBLE_FEATURE_CONTRACT,
            f"{raw.spec.cell_id}: mixed-v2 produced non-finite values",
        )
    return np.asarray(transformed, dtype=np.float64), details


def _row_roster_signature(
    row_ids: Sequence[str],
    *,
    contract_binding: Mapping[str, Any],
    row_namespace_sha256: str,
) -> str:
    rows = tuple(map(str, row_ids))
    if not rows or len(set(rows)) != len(rows) or any(
        _OPAQUE_ROW_ID_RE.fullmatch(value) is None for value in rows
    ):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            "opaque row roster is empty, duplicated, or malformed",
        )
    return _payload_hash({
        "schema_version": "reconstruction-external-fit-row-roster-v2",
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": contract_binding["contract_sha256"],
        "key_id": contract_binding["key_id"],
        "row_namespace_sha256": row_namespace_sha256,
        "row_ids": list(rows),
    })


def sealed_group_roster_commitment(identity: OpaqueIdentityRoster) -> str:
    """Commit to post-freeze resampling groups without exposing membership."""

    assert_opaque_external_ids(identity.row_ids, identity.group_ids)
    return _payload_hash({
        "schema_version": "reconstruction-external-sealed-group-roster-v1",
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": identity.contract_binding["contract_sha256"],
        "key_id": identity.contract_binding["key_id"],
        "row_namespace_sha256": identity.row_namespace_sha256,
        "group_namespace_sha256": identity.group_namespace_sha256,
        "row_ids": list(identity.row_ids),
        "group_ids": list(identity.group_ids),
    })


def prepare_external_cell(
    *,
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    repo: str | Path,
    output_path: str | Path,
    identity_key: bytes,
) -> Mapping[str, Any]:
    sources = resolve_sources(registry, spec, repo=repo)
    verified = verify_sources(sources, include_labels=False)
    raw = load_raw_feature_cell(spec, sources)
    identity = apply_external_id_contract(
        registry, spec, raw.row_ids, raw.group_ids, identity_key=identity_key
    )
    ordered_raw_matrix, identity = canonicalize_external_identity_order(
        raw.raw_matrix, identity
    )
    ordered_raw = RawFeatureCell(
        spec=raw.spec,
        raw_matrix=ordered_raw_matrix,
        row_ids=identity.row_ids,
        group_ids=identity.group_ids,
        source_files=raw.source_files,
        feature_names=raw.feature_names,
        preprocessing_steps=raw.preprocessing_steps,
    )
    matrix, transform_details = apply_mixed_v2_once(ordered_raw)
    matrix_hash = prepared_matrix_sha256(matrix, raw.feature_names, identity.row_ids)
    fit_identity_contract = build_fit_row_identity_contract(
        identity.contract_binding,
        identity_key=identity_key,
    )
    fit_row_namespace = fit_row_namespace_sha256(
        contract=fit_identity_contract,
        cell_id=spec.cell_id,
    )
    if fit_row_namespace != identity.row_namespace_sha256:
        raise RuntimeError("controller and fit row namespaces disagree")
    fit_row_roster = fit_row_roster_sha256(
        identity.row_ids,
        contract=fit_identity_contract,
        row_namespace_sha256_value=fit_row_namespace,
    )
    # PreparedCell validation is the executable gate shared by all 13 methods.
    PreparedCell(
        population_id=spec.population_id,
        cell_id=spec.cell_id,
        domain=spec.domain,
        matrix=matrix,
        feature_names=raw.feature_names,
        row_ids=identity.row_ids,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=matrix_hash,
    )
    target = Path(output_path)
    arrays = {
        "X_confidence": matrix.astype("<f8", copy=False),
        "feature_names": np.asarray(raw.feature_names, dtype="<U64"),
        "family_ids": np.asarray([FEATURE_TO_VIEW[name] for name in raw.feature_names], dtype="<U32"),
        "row_ids": np.asarray(identity.row_ids, dtype="<U80"),
        "row_index": np.arange(len(identity.row_ids), dtype="<i8"),
        "id_contract_version": np.asarray([ID_CONTRACT_VERSION], dtype="<U64"),
        "id_contract_sha256": np.asarray(
            [fit_identity_contract["contract_sha256"]], dtype="<U64"
        ),
        "row_namespace_sha256": np.asarray(
            [identity.row_namespace_sha256], dtype="<U64"
        ),
        "identity_key_id": np.asarray(
            [identity.contract_binding["key_id"]], dtype="<U80"
        ),
    }
    forbidden = [
        name for name in arrays
        if any(fragment in name.lower() for fragment in TARGET_LIKE_FRAGMENTS)
    ]
    if forbidden:
        raise RuntimeError(f"target-like members in prepared artifact: {forbidden}")
    artifact_sha = atomic_write_npz(target, arrays)
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": ReadinessStatus.ELIGIBLE.value,
        "cell_id": spec.cell_id,
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "model_id": spec.model_id,
        "slice_id": spec.slice_id,
        "domain": spec.domain,
        "comparison_group_id": spec.comparison_group_id,
        "panel_role": spec.panel_role,
        "adapter_id": spec.adapter_id,
        "n_rows": len(identity.row_ids),
        "n_features": len(raw.feature_names),
        "feature_names": list(raw.feature_names),
        "present_feature_roster_sha256": _payload_hash(list(raw.feature_names)),
        "nominal_feature_count": len(CANONICAL_FEATURE_NAMES),
        "nominal_feature_roster_sha256": _payload_hash(list(CANONICAL_FEATURE_NAMES)),
        "absent_feature_names": [
            name for name in CANONICAL_FEATURE_NAMES if name not in set(raw.feature_names)
        ],
        "feature_contract_id": CONTRACT_VERSION,
        "preprocessing_steps": [CONTRACT_VERSION],
        "mixed_v2_applied_count": 1,
        "matrix_semantics": "higher_is_confidence",
        "prepared_matrix_sha256": matrix_hash,
        "identity_contract": dict(identity.contract_binding),
        "fit_row_identity_contract": dict(fit_identity_contract),
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": identity.contract_binding["contract_sha256"],
        "fit_row_id_contract_sha256": fit_identity_contract["contract_sha256"],
        "identity_key_id": identity.contract_binding["key_id"],
        "row_namespace_sha256": identity.row_namespace_sha256,
        "group_namespace_sha256": identity.group_namespace_sha256,
        "row_roster_sha256": fit_row_roster,
        "sealed_group_roster_commitment_sha256": sealed_group_roster_commitment(identity),
        "group_count": len(set(identity.group_ids)),
        "artifact_path": target.name,
        "artifact_sha256": artifact_sha,
        "source_files": verified,
        "source_inventory_bindings": list(sources.inventory_bindings),
        "transform_details": transform_details,
        "labels_opened": False,
        "historical_scores_opened": False,
    }


_FIT_SAFE_PUBLIC_CELL_FIELDS = (
    "cell_id", "population_id", "dataset_id", "model_id", "slice_id", "domain",
    "comparison_group_id", "panel_role", "status", "prepared",
)
_FIT_SAFE_ELIGIBLE_FIELDS = (
    "schema_version", "n_rows", "n_features", "feature_names",
    "present_feature_roster_sha256", "nominal_feature_count",
    "nominal_feature_roster_sha256", "absent_feature_names",
    "feature_contract_id", "preprocessing_steps", "mixed_v2_applied_count",
    "matrix_semantics", "prepared_matrix_sha256",
    "id_contract_version", "identity_key_id",
    "row_namespace_sha256", "row_roster_sha256", "artifact_path",
    "artifact_sha256",
)


def fit_safe_external_cell_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only per-cell metadata allowed inside the fit mount."""

    safe = {
        key: record[key]
        for key in _FIT_SAFE_PUBLIC_CELL_FIELDS
        if key in record
    }
    if record.get("status") == ReadinessStatus.ELIGIBLE.value:
        controller_required = {
            *_FIT_SAFE_ELIGIBLE_FIELDS,
            "fit_row_identity_contract", "fit_row_id_contract_sha256",
        }
        missing = sorted(controller_required - set(record))
        if missing:
            raise RuntimeError(f"eligible external record lacks fit-safe fields: {missing}")
        safe.update({key: record[key] for key in _FIT_SAFE_ELIGIBLE_FIELDS})
        safe.update({
            "identity_contract": record["fit_row_identity_contract"],
            "id_contract_sha256": record["fit_row_id_contract_sha256"],
        })
    elif record.get("prepared") is not False:
        safe["prepared"] = False
    forbidden = {
        "expected_rows", "expected_correct", "expected_incorrect",
        "expected_group_count", "group_count", "group_ids",
        "group_namespace_sha256", "sealed_group_roster_commitment_sha256",
        "fit_row_identity_contract", "fit_row_id_contract_sha256",
        "source", "source_root", "source_files", "source_inventory_bindings",
        "reason", "excluded_row_ids", "excluded_raw_id_fingerprints",
        "transform_details", "verified_source_files", "label_rule",
    }
    leaked = sorted(forbidden & set(safe))
    if leaked:
        raise RuntimeError(f"controller-only metadata crossed fit boundary: {leaked}")
    return safe


def load_prepared_external_cell(
    *,
    artifact_path: str | Path,
    record: Mapping[str, Any],
    identity_contract: Mapping[str, Any],
) -> PreparedCell:
    path = Path(artifact_path)
    if sha256_file(path) != str(record["artifact_sha256"]):
        raise RuntimeError(f"prepared artifact hash mismatch: {path}")
    arrays = load_npz_no_pickle(path)
    allowed = {
        "X_confidence", "feature_names", "family_ids", "row_ids",
        "row_index", "id_contract_version", "id_contract_sha256",
        "row_namespace_sha256", "identity_key_id",
    }
    if set(arrays) != allowed:
        raise RuntimeError(f"unexpected prepared arrays: {sorted(set(arrays) ^ allowed)}")
    if any(
        any(fragment in name.lower() for fragment in TARGET_LIKE_FRAGMENTS)
        for name in arrays
    ):
        raise RuntimeError("target-like array crossed the fitting boundary")
    names = tuple(map(str, arrays["feature_names"].tolist()))
    expected_names = tuple(name for name in CANONICAL_FEATURE_NAMES if name in set(names))
    if names != expected_names or names != tuple(record.get("feature_names", ())):
        raise RuntimeError("prepared external present-feature roster/order drifted")
    if _payload_hash(list(names)) != record.get("present_feature_roster_sha256"):
        raise RuntimeError("prepared external present-feature roster hash drifted")
    family_ids = tuple(map(str, arrays["family_ids"].tolist()))
    expected_family_ids = tuple(FEATURE_TO_VIEW[name] for name in names)
    if family_ids != expected_family_ids:
        raise RuntimeError("prepared external family roster/order drifted")
    rows = tuple(map(str, arrays["row_ids"].tolist()))
    if not rows or len(set(rows)) != len(rows) or any(
        _OPAQUE_ROW_ID_RE.fullmatch(value) is None for value in rows
    ):
        raise RuntimeError("prepared external row IDs are not keyed opaque v2 IDs")
    if rows != tuple(sorted(rows)):
        raise RuntimeError("prepared external row IDs are not in canonical opaque order")
    if not np.array_equal(
        np.asarray(arrays["row_index"], dtype=np.int64),
        np.arange(len(rows), dtype=np.int64),
    ):
        raise RuntimeError("prepared external row_index is not canonical")
    expected_contract = validate_public_identity_contract(identity_contract)
    row_namespace = _identity_namespace(
        expected_contract,
        kind="row",
        cell_id=str(record.get("cell_id", "")),
        population_id=str(record.get("population_id", "")),
        slice_id=str(record.get("slice_id", "")),
    )
    row_namespace_sha256 = _payload_hash(row_namespace)
    identity_exact = {
        "identity_contract": expected_contract,
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": expected_contract["contract_sha256"],
        "identity_key_id": expected_contract["key_id"],
        "row_namespace_sha256": row_namespace_sha256,
    }
    for key, value in identity_exact.items():
        if record.get(key) != value:
            raise RuntimeError(f"prepared external identity binding drifted: {key}")
    artifact_identity = {}
    for key in (
        "id_contract_version", "id_contract_sha256",
        "row_namespace_sha256", "identity_key_id",
    ):
        scalar = np.asarray(arrays[key])
        if scalar.shape != (1,):
            raise RuntimeError(f"prepared artifact {key} is not a scalar binding")
        artifact_identity[key] = str(scalar.tolist()[0])
    for key in artifact_identity:
        if artifact_identity[key] != identity_exact[key]:
            raise RuntimeError(f"prepared artifact identity binding drifted: {key}")
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    observed = prepared_matrix_sha256(matrix, names, rows)
    if observed != record.get("prepared_matrix_sha256"):
        raise RuntimeError("prepared matrix/row hash mismatch")
    if _row_roster_signature(
        rows,
        contract_binding=expected_contract,
        row_namespace_sha256=row_namespace_sha256,
    ) != record.get("row_roster_sha256"):
        raise RuntimeError("prepared opaque row-roster signature mismatch")
    cell = PreparedCell(
        population_id=str(record["population_id"]),
        cell_id=str(record["cell_id"]),
        domain=str(record["domain"]),
        matrix=matrix,
        feature_names=names,
        row_ids=rows,
        feature_contract=CONTRACT_VERSION,
        preprocessing_steps=(CONTRACT_VERSION,),
        preprocessed=True,
        declared_matrix_sha256=observed,
    )
    return cell


def audit_external_registry(
    *,
    registry: ExternalRegistry,
    repo: str | Path,
    deep: bool = False,
    identity_key: bytes | None = None,
) -> Mapping[str, Any]:
    if deep and identity_key is None:
        raise ValueError("deep external audit requires the sealed release identity key")
    rows: list[Mapping[str, Any]] = []
    for spec in registry.cells:
        base = {
            "cell_id": spec.cell_id,
            "population_id": spec.population_id,
            "dataset_id": spec.dataset_id,
            "model_id": spec.model_id,
            "slice_id": spec.slice_id,
            "comparison_group_id": spec.comparison_group_id,
            "panel_role": spec.panel_role,
            "fit_policy": spec.fit_policy,
            "expected_rows": spec.expected_rows,
        }
        if spec.fit_policy == "forbidden":
            rows.append({
                **base,
                "status": str(spec.configured_status),
                "reason": spec.status_reason,
                "fit_allowed": False,
                "sources_verified": False,
                "label_assets_hash_verified_without_parsing": False,
                "feature_contract_verified": False,
            })
            continue
        try:
            sources = resolve_sources(registry, spec, repo=repo)
            # Hash label ledgers for readiness without parsing their contents.
            # The label vector and even its class counts remain unopened until
            # the score-freeze gate in ``load_labels_after_score_freeze``.
            files = verify_sources(sources, include_labels=True)
            result: dict[str, Any] = {
                **base,
                "status": ReadinessStatus.READY_FOR_TARGET_FREE_PREPARATION.value,
                "reason": None,
                "fit_allowed": True,
                "sources_verified": True,
                "label_assets_hash_verified_without_parsing": True,
                "feature_contract_verified": False,
                "verified_files": files,
                "inventory_bindings": list(sources.inventory_bindings),
            }
            if deep:
                raw = load_raw_feature_cell(spec, sources)
                identity = apply_external_id_contract(
                    registry,
                    spec,
                    raw.row_ids,
                    raw.group_ids,
                    identity_key=identity_key,
                )
                ordered_matrix, identity = canonicalize_external_identity_order(
                    raw.raw_matrix, identity
                )
                ordered_raw = RawFeatureCell(
                    spec=raw.spec,
                    raw_matrix=ordered_matrix,
                    row_ids=identity.row_ids,
                    group_ids=identity.group_ids,
                    source_files=raw.source_files,
                    feature_names=raw.feature_names,
                    preprocessing_steps=raw.preprocessing_steps,
                )
                matrix, _ = apply_mixed_v2_once(ordered_raw)
                result.update({
                    "status": ReadinessStatus.ELIGIBLE.value,
                    "feature_contract_verified": True,
                    "identity_contract_verified": True,
                    "id_contract_version": ID_CONTRACT_VERSION,
                    "id_contract_sha256": identity.contract_binding["contract_sha256"],
                    "row_namespace_sha256": identity.row_namespace_sha256,
                    "group_namespace_sha256": identity.group_namespace_sha256,
                    "n_rows": len(identity.row_ids),
                    "n_features": matrix.shape[1],
                    "row_roster_sha256": _row_roster_signature(
                        identity.row_ids,
                        contract_binding=identity.contract_binding,
                        row_namespace_sha256=identity.row_namespace_sha256,
                    ),
                    "sealed_group_roster_commitment_sha256": sealed_group_roster_commitment(identity),
                })
            rows.append(result)
        except ExternalContractError as error:
            rows.append({
                **base,
                "status": error.status.value,
                "reason": str(error),
                "fit_allowed": False,
                "sources_verified": error.status not in {
                    ReadinessStatus.BLOCKED_ASSET,
                    ReadinessStatus.SOURCE_HASH_MISMATCH,
                    ReadinessStatus.SOURCE_INVENTORY_MISMATCH,
                },
                "label_assets_hash_verified_without_parsing": False,
                "feature_contract_verified": False,
            })
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "deep_feature_audit": bool(deep),
        "registry_path": registry.path.as_posix(),
        "registry_sha256": registry.sha256,
        "population_registry_path": registry.population_registry_path.as_posix(),
        "population_registry_sha256": registry.population_registry_sha256,
        "source_root": str(Path(repo).resolve()),
        "feature_contract_id": CONTRACT_VERSION,
        "identity_contract": (
            None
            if identity_key is None
            else external_id_contract_binding(registry, identity_key=identity_key)
        ),
        "id_contract_version": ID_CONTRACT_VERSION,
        "row_dropping_or_imputation_forbidden": True,
        "partial_feature_availability_forbidden": True,
        "uniformly_unavailable_nominal_views_may_remain_absent": True,
        "historical_scores_opened": False,
        "labels_opened": False,
        "cells": rows,
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted({str(row["status"]) for row in rows})
        },
    }
    manifest["payload_sha256"] = _payload_hash(manifest)
    return manifest


def _label_rows_processbench(spec: ExternalCellSpec, sources: ResolvedSources):
    cache = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        alignment = row.get("align_diag", {})
        problems = alignment.get("problems")
        if problems or alignment.get("ok") is False:
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "ProcessBench label roster contains misaligned row")
        official_id = row.get("id")
        if not isinstance(official_id, str) or not official_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"ProcessBench label source row {key} lacks its official string ID",
            )
        row_id = official_id
        if "final_answer_correct" not in row:
            raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{row_id}: final_answer_correct is absent")
        rows.append(row_id)
        groups.append(row_id)
        labels.append(1 - int(bool(row["final_answer_correct"])))
    return rows, groups, labels, {"label_rule": "1 - bool(final_answer_correct)", "source": "ProcessBench fixed response"}


def _label_rows_prmbench(spec: ExternalCellSpec, sources: ResolvedSources):
    cache = _load_pickle(sources.feature_files[0].path)
    excluded = set(spec.excluded_raw_id_fingerprints)
    rows, groups, labels = [], [], []
    for key in sorted(cache, key=_stable_key):
        row = cache[key]
        row_id = str(row.get("idx", ""))
        if not row_id:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench label source row {key} lacks idx",
            )
        fingerprint = raw_id_fingerprint(row_id)
        if fingerprint in excluded:
            continue
        classification = str(row.get("classification", ""))
        if not classification:
            raise ExternalContractError(
                ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
                f"PRMBench row fingerprint {fingerprint}: classification is absent",
            )
        rows.append(row_id)
        source_group = str(row.get("source_idx", ""))
        if not source_group:
            raise ExternalContractError(
                ReadinessStatus.ROW_CONTRACT_MISMATCH,
                f"PRMBench label row fingerprint {fingerprint} lacks source_idx",
            )
        groups.append(source_group)
        labels.append(int(classification != "correct"))
    return rows, groups, labels, {
        "label_rule": "classification != correct",
        "excluded_raw_id_fingerprints": sorted(excluded),
        "raw_id_fingerprint_version": RAW_ID_FINGERPRINT_VERSION,
    }


def _label_rows_repgrid(spec: ExternalCellSpec, sources: ResolvedSources):
    data = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(data, key=_stable_key):
        candidates = data[key]["candidates"]
        for index, candidate in enumerate(candidates):
            if "label" not in candidate:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{key}/{index}: embedded label absent")
            group = str(key)
            rows.append(f"{group}:candidate:{index:04d}")
            groups.append(group)
            labels.append(1 - int(bool(candidate["label"])))
    return rows, groups, labels, {"label_rule": "1 - bool(candidate.label)", "source": "frozen embedded grader output"}


def _label_rows_hle(spec: ExternalCellSpec, sources: ResolvedSources):
    if len(sources.label_files) != 1:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "HLE needs one separate judge ledger")
    judged: dict[str, int] = {}
    with sources.label_files[0].path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = str(record.get("correct", "")).strip().lower()
            if value not in {"yes", "no"}:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"invalid HLE judge value {value!r}")
            key = str(record["row_key"])
            if key in judged:
                raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"duplicate HLE judge key {key}")
            judged[key] = int(value == "no")
    raw = _load_pickle(sources.feature_files[0].path)
    rows, groups, labels = [], [], []
    for key in sorted(raw, key=_stable_key):
        candidates = raw[key]["candidates"]
        if len(candidates) != 1:
            raise ExternalContractError(ReadinessStatus.ROW_CONTRACT_MISMATCH, "HLE must have one response per question")
        group = str(key)
        if group not in judged:
            raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"missing HLE judge key {group}")
        rows.append(f"{group}:candidate:0000")
        groups.append(group)
        labels.append(judged[group])
    if set(judged) != set(groups):
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "HLE judge/source rosters differ")
    return rows, groups, labels, {
        "label_rule": "Codex interim judge correct=yes -> incorrect=0; correct=no -> incorrect=1",
        "paper_faithful_grader": False,
        "judge_file": sources.label_files[0].relative_path,
    }


def _label_rows_aqua(spec: ExternalCellSpec, sources: ResolvedSources):
    records: list[Mapping[str, Any]] = []
    for item in sorted(sources.feature_files, key=lambda value: value.relative_path):
        records.extend(_load_pickle(item.path))
    selected = [row for row in records if row.get("arm") == "cot" and row.get("setting_label") == "central"]
    selected.sort(key=lambda row: _stable_key(str(row.get("question_id", "")).split(":")[-1]))
    rows, labels = [], []
    for row in selected:
        row_id = str(row.get("question_id", ""))
        graded = grade_aqua_option(row.get("answer_text"), str(row.get("gold_answer", "")))
        rows.append(row_id)
        labels.append(1 - int(bool(graded["correct"])))
    return rows, rows, labels, {"label_rule": "fair_aqua_option_parser_v1.0.0", "derived_after_score_freeze": True}


LABEL_ADAPTERS = {
    "processbench_teacher_forced_v1": _label_rows_processbench,
    "prmbench_teacher_forced_v1": _label_rows_prmbench,
    "repgrid_embedded_label_v1": _label_rows_repgrid,
    "hle_separate_judge_v1": _label_rows_hle,
    "aqua_paper_exact_cot_v1": _label_rows_aqua,
}


def assert_score_freeze(
    freeze: Mapping[str, Any],
    *,
    registry: ExternalRegistry,
    identity_key: bytes,
) -> None:
    expected_fit_identity = build_fit_row_identity_contract(
        external_id_contract_binding(registry, identity_key=identity_key),
        identity_key=identity_key,
    )
    if freeze.get("schema_version") != SCORE_FREEZE_SCHEMA_VERSION:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "external score freeze schema is missing")
    if freeze.get("all_expected_scores_present") is not True:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze is incomplete")
    if freeze.get("labels_opened_by_fit") is not False or freeze.get("runtime_labels_used") is not False:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "fit freeze does not prove label isolation")
    if freeze.get("external_registry_sha256") != registry.sha256:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze binds another external registry")
    if (
        freeze.get("id_contract_version") != ID_CONTRACT_VERSION
        or freeze.get("identity_contract") != expected_fit_identity
    ):
        raise ExternalContractError(
            ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
            "score freeze binds another opaque identity contract",
        )
    payload = dict(freeze)
    recorded = payload.pop("payload_sha256", None)
    if recorded != _payload_hash(payload):
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, "score freeze payload hash failed")


def load_labels_after_score_freeze(
    *,
    registry: ExternalRegistry,
    spec: ExternalCellSpec,
    repo: str | Path,
    score_freeze: Mapping[str, Any],
    expected_row_ids: Sequence[str],
    expected_group_roster_commitment_sha256: str,
    identity_key: bytes,
) -> LabelVector:
    assert_score_freeze(
        score_freeze, registry=registry, identity_key=identity_key
    )
    sources = resolve_sources(registry, spec, repo=repo)
    verified = verify_sources(sources, include_labels=True)
    adapter = LABEL_ADAPTERS.get(spec.adapter_id)
    if adapter is None:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"no label adapter for {spec.adapter_id}")
    try:
        rows, groups, labels, provenance = adapter(spec, sources)
    except ExternalContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ExternalContractError(
            ReadinessStatus.LABEL_PROVENANCE_BLOCKED,
            f"{spec.cell_id}: label adapter failed closed: {type(error).__name__}: {error}",
        ) from error
    raw_rows = tuple(map(str, rows))
    raw_groups = tuple(map(str, groups))
    identity = apply_external_id_contract(
        registry,
        spec,
        raw_rows,
        raw_groups,
        identity_key=identity_key,
    )
    raw_values = np.asarray(labels)
    if raw_values.shape != (len(identity.row_ids),):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: label and raw identity rosters disagree",
        )
    order = _opaque_order_indices(identity.row_ids)
    rows = tuple(identity.row_ids[index] for index in order.tolist())
    groups = tuple(identity.group_ids[index] for index in order.tolist())
    labels = raw_values[order]
    identity = OpaqueIdentityRoster(
        row_ids=rows,
        group_ids=groups,
        contract_binding=identity.contract_binding,
        row_namespace_sha256=identity.row_namespace_sha256,
        group_namespace_sha256=identity.group_namespace_sha256,
    )
    if rows != tuple(map(str, expected_row_ids)):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: post-freeze label roster/order differs from the frozen score cohort",
        )
    if sealed_group_roster_commitment(identity) != str(
        expected_group_roster_commitment_sha256
    ):
        raise ExternalContractError(
            ReadinessStatus.ROW_CONTRACT_MISMATCH,
            f"{spec.cell_id}: post-freeze group roster differs from sealed preparation",
        )
    values = np.asarray(labels, dtype=np.int8)
    incorrect = int(values.sum())
    correct = int(len(values) - incorrect)
    if spec.expected_incorrect is not None and incorrect != spec.expected_incorrect:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{spec.cell_id}: expected {spec.expected_incorrect} incorrect, got {incorrect}")
    if spec.expected_correct is not None and correct != spec.expected_correct:
        raise ExternalContractError(ReadinessStatus.LABEL_PROVENANCE_BLOCKED, f"{spec.cell_id}: expected {spec.expected_correct} correct, got {correct}")
    label_provenance = {
        **dict(provenance),
        "verified_source_files": verified,
        "row_label_sha256": _payload_hash({"row_ids": list(rows), "incorrect": values.tolist()}),
        "positive_class": "incorrect",
        "n_incorrect": incorrect,
        "n_correct": correct,
        "score_freeze_payload_sha256": score_freeze["payload_sha256"],
        "identity_contract": dict(identity.contract_binding),
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": identity.contract_binding["contract_sha256"],
        "identity_key_id": identity.contract_binding["key_id"],
        "row_namespace_sha256": identity.row_namespace_sha256,
        "group_namespace_sha256": identity.group_namespace_sha256,
        "row_roster_sha256": _row_roster_signature(
            rows,
            contract_binding=identity.contract_binding,
            row_namespace_sha256=identity.row_namespace_sha256,
        ),
        "sealed_group_roster_commitment_sha256": sealed_group_roster_commitment(identity),
    }
    return LabelVector(spec.cell_id, rows, groups, values, label_provenance)


def write_label_vector(path: str | Path, value: LabelVector) -> Mapping[str, Any]:
    target = Path(path)
    artifact_sha = atomic_write_npz(target, {
        "row_ids": np.asarray(value.row_ids, dtype="<U80"),
        "group_ids": np.asarray(value.group_ids, dtype="<U80"),
        "incorrect": np.asarray(value.incorrect, dtype="i1"),
        "id_contract_version": np.asarray(
            [value.provenance["id_contract_version"]], dtype="<U64"
        ),
        "id_contract_sha256": np.asarray(
            [value.provenance["id_contract_sha256"]], dtype="<U64"
        ),
        "identity_key_id": np.asarray(
            [value.provenance["identity_key_id"]], dtype="<U80"
        ),
        "row_namespace_sha256": np.asarray(
            [value.provenance["row_namespace_sha256"]], dtype="<U64"
        ),
        "group_namespace_sha256": np.asarray(
            [value.provenance["group_namespace_sha256"]], dtype="<U64"
        ),
    })
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "cell_id": value.cell_id,
        "n_rows": len(value.row_ids),
        "artifact_path": target.name,
        "artifact_sha256": artifact_sha,
        "identity_contract": dict(value.provenance["identity_contract"]),
        "id_contract_version": value.provenance["id_contract_version"],
        "id_contract_sha256": value.provenance["id_contract_sha256"],
        "identity_key_id": value.provenance["identity_key_id"],
        "row_namespace_sha256": value.provenance["row_namespace_sha256"],
        "group_namespace_sha256": value.provenance["group_namespace_sha256"],
        "row_roster_sha256": value.provenance["row_roster_sha256"],
        "sealed_group_roster_commitment_sha256": value.provenance[
            "sealed_group_roster_commitment_sha256"
        ],
        "provenance": dict(value.provenance),
    }


# Public callers and tests use the same fit-only loader that is copied into the
# restricted capsule.  The controller module retains preparation/evaluation
# code, but none of it is in that loader's import closure.
load_prepared_external_cell = load_fit_prepared_external_cell


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CANONICAL_FEATURE_NAMES",
    "ExternalCellSpec",
    "ExternalContractError",
    "ExternalRegistry",
    "ID_CONTRACT_VERSION",
    "ID_DIGEST_ALGORITHM",
    "IDENTITY_KEY_BYTES",
    "IDENTITY_KEY_CONTRACT_VERSION",
    "LABEL_SCHEMA_VERSION",
    "LabelVector",
    "OpaqueIdentityRoster",
    "OPAQUE_GROUP_ID_PREFIX",
    "OPAQUE_ROW_ID_PREFIX",
    "PREPARED_SCHEMA_VERSION",
    "RAW_ADAPTERS",
    "ReadinessStatus",
    "ResolvedSources",
    "SCORE_FREEZE_SCHEMA_VERSION",
    "RAW_ID_FINGERPRINT_PREFIX",
    "RAW_ID_FINGERPRINT_VERSION",
    "SourceFile",
    "apply_mixed_v2_once",
    "apply_external_id_contract",
    "assert_opaque_external_ids",
    "assert_score_freeze",
    "audit_external_registry",
    "canonicalize_external_identity_order",
    "external_id_contract_binding",
    "fit_safe_external_cell_record",
    "identity_key_id",
    "keyed_opaque_external_id",
    "load_external_registry",
    "load_identity_key",
    "load_labels_after_score_freeze",
    "load_prepared_external_cell",
    "load_raw_feature_cell",
    "prepare_external_cell",
    "raw_id_fingerprint",
    "resolve_sources",
    "sealed_group_roster_commitment",
    "validate_public_identity_contract",
    "verify_source_file",
    "verify_sources",
    "write_label_vector",
]
