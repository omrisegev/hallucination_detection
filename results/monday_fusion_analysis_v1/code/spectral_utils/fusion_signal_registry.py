"""Immutable, label-free registry and operators for Fusion Independence Atlas v1.

The module deliberately has no correctness-label interface.  It contains only
semantic registration, causal transformations of already-cached telemetry,
fixed readouts/decoders, and positive label-free fusion heads.  Evaluation and
nested roster selection live outside this module.

Masks are explicit throughout: a numerical zero in an inactive stream is a
storage value, not evidence.  All public transforms return ``(values, active)``
with finite zero fill where ``active`` is false.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import heapq
import inspect
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "fusion-independence-atlas-v1"
EPS = 1e-12

INSERTION_POINTS = (
    "background",
    "token_pre_readout",
    "step_post_readout",
    "decoder",
    "answer_gate",
)
RESOLUTIONS = ("token", "step", "decision", "answer")
ACCESS_SCOPES = (
    "answer_only",
    "source_excluded",
    "pair_excluded",
    "historical_artifact",
    "not_available",
)
ORIENTATIONS = (
    "high_is_risk",
    "low_is_risk",
    "anchor_oriented",
    "preserve",
    "not_applicable",
)
MASK_SEMANTICS = (
    "all_active",
    "first_token_inactive",
    "opportunity_only",
    "propagate",
    "artifact_defined",
    "unavailable",
)
STATUSES = ("ELIGIBLE", "REPORT_ONLY", "PENDING_EXPANSION")
POSITIVE_FUSION_HEADS = (
    "equal_rank",
    "family_equal",
    "nonnegative_shrunk_simplex",
)

_EXPECTED_RESOLUTION = {
    "background": "token",
    "token_pre_readout": "token",
    "step_post_readout": "step",
    "decoder": "decision",
    "answer_gate": "answer",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+\-]*$")


class RegistryValidationError(ValueError):
    """Raised when an Atlas semantic or compatibility contract is invalid."""


def _canonical_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            output[key] = _canonical_value(item)
        return output
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical payloads cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the stable UTF-8 JSON representation used by every spec hash."""

    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_sha256(value: Any) -> str:
    """Hash an exact array together with shape and dtype.

    Object arrays are rejected because their byte representation is not a
    portable content identity.
    """

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays do not have a canonical score hash")
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_json({
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
    }).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def callable_sha256(*objects: Any) -> str:
    """Hash implementation source without executing it."""

    pieces = [SCHEMA_VERSION]
    for value in objects:
        try:
            pieces.append(inspect.getsource(value))
        except (OSError, TypeError):
            pieces.append(f"{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', repr(value))}")
    return hashlib.sha256("\n\0\n".join(pieces).encode("utf-8")).hexdigest()


def _semantic_hash(identity: str) -> str:
    return canonical_sha256({"schema": SCHEMA_VERSION, "semantic_identity": identity})


def _validate_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise RegistryValidationError(f"{field_name} must be a nonempty stable identifier")


def _validate_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{field_name} must be nonempty")


def _validate_hash(value: str | None, field_name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise RegistryValidationError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _spec_payload(instance: Any) -> dict[str, Any]:
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """One immutable signal identity; never a score selected with labels."""

    name: str
    provenance_family: str
    insertion_point: str
    resolution: str
    target: str
    transform: str
    access_scope: str
    orientation: str
    mask_semantics: str
    status: str
    provenance: tuple[str, ...]
    implementation_hash: str
    semantic_hash: str
    score_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "signal name")
        _validate_name(self.provenance_family, "provenance family")
        _validate_name(self.target, "target")
        _validate_text(self.transform, "transform")
        if self.insertion_point not in INSERTION_POINTS:
            raise RegistryValidationError(f"unknown insertion point {self.insertion_point!r}")
        if self.resolution not in RESOLUTIONS:
            raise RegistryValidationError(f"unknown resolution {self.resolution!r}")
        if _EXPECTED_RESOLUTION[self.insertion_point] != self.resolution:
            raise RegistryValidationError(
                f"{self.insertion_point} requires {_EXPECTED_RESOLUTION[self.insertion_point]} resolution"
            )
        if self.access_scope not in ACCESS_SCOPES:
            raise RegistryValidationError(f"unknown access scope {self.access_scope!r}")
        if self.orientation not in ORIENTATIONS:
            raise RegistryValidationError(f"unknown orientation {self.orientation!r}")
        if self.mask_semantics not in MASK_SEMANTICS:
            raise RegistryValidationError(f"unknown mask semantics {self.mask_semantics!r}")
        if self.status not in STATUSES:
            raise RegistryValidationError(f"unknown status {self.status!r}")
        if not isinstance(self.provenance, tuple) or not self.provenance:
            raise RegistryValidationError("provenance must be a nonempty tuple")
        for item in self.provenance:
            _validate_text(item, "provenance entry")
        _validate_hash(self.implementation_hash, "implementation_hash")
        _validate_hash(self.semantic_hash, "semantic_hash")
        _validate_hash(self.score_hash, "score_hash", optional=True)
        if self.status == "PENDING_EXPANSION" and self.access_scope != "not_available":
            raise RegistryValidationError("pending expansion signals must use not_available access")

    @property
    def family(self) -> str:
        return self.provenance_family

    @property
    def signal_id(self) -> str:
        return self.name

    @property
    def spec_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def deduplication_hash(self) -> str:
        return self.score_hash or self.semantic_hash

    def to_dict(self) -> dict[str, Any]:
        return _spec_payload(self)

    def bind_score(self, values: Any) -> "SignalSpec":
        return replace(self, score_hash=array_sha256(values))


@dataclass(frozen=True, slots=True)
class ReadoutSpec:
    """One immutable label-free reduction or decoder contract."""

    name: str
    provenance_family: str
    insertion_point: str
    input_resolution: str
    output_resolution: str
    target: str
    transform: str
    access_scope: str
    orientation: str
    mask_semantics: str
    chronological_semantics: str
    short_trace_rule: str
    status: str
    provenance: tuple[str, ...]
    implementation_hash: str
    semantic_hash: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "readout name")
        _validate_name(self.provenance_family, "provenance family")
        _validate_name(self.target, "target")
        _validate_text(self.transform, "transform")
        _validate_text(self.chronological_semantics, "chronological semantics")
        _validate_text(self.short_trace_rule, "short trace rule")
        if self.insertion_point not in INSERTION_POINTS:
            raise RegistryValidationError(f"unknown insertion point {self.insertion_point!r}")
        if self.input_resolution not in RESOLUTIONS or self.output_resolution not in RESOLUTIONS:
            raise RegistryValidationError("invalid readout resolution")
        if _EXPECTED_RESOLUTION[self.insertion_point] != self.output_resolution:
            raise RegistryValidationError("readout insertion point and output resolution disagree")
        if self.access_scope not in ACCESS_SCOPES:
            raise RegistryValidationError(f"unknown access scope {self.access_scope!r}")
        if self.orientation not in ORIENTATIONS:
            raise RegistryValidationError(f"unknown orientation {self.orientation!r}")
        if self.mask_semantics not in MASK_SEMANTICS:
            raise RegistryValidationError(f"unknown mask semantics {self.mask_semantics!r}")
        if self.status not in STATUSES:
            raise RegistryValidationError(f"unknown status {self.status!r}")
        if not isinstance(self.provenance, tuple) or not self.provenance:
            raise RegistryValidationError("provenance must be a nonempty tuple")
        for item in self.provenance:
            _validate_text(item, "provenance entry")
        _validate_hash(self.implementation_hash, "implementation_hash")
        _validate_hash(self.semantic_hash, "semantic_hash")

    @property
    def family(self) -> str:
        return self.provenance_family

    @property
    def resolution(self) -> str:
        return self.output_resolution

    @property
    def readout_id(self) -> str:
        return self.name

    @property
    def spec_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _spec_payload(self)


@dataclass(frozen=True, slots=True)
class FusionSetSpec:
    """One positive fusion candidate at exactly one legal fusion point."""

    name: str
    members: tuple[str, ...]
    provenance_families: tuple[str, ...]
    insertion_point: str
    resolution: str
    target: str
    head: str
    access_scope: str
    orientation: str
    mask_semantics: str
    status: str
    provenance: tuple[str, ...]
    implementation_hash: str
    semantic_hash: str
    fitted_weight_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "fusion-set name")
        _validate_name(self.target, "target")
        if not isinstance(self.members, tuple) or not 1 <= len(self.members) <= 6:
            raise RegistryValidationError("fusion sets require one to six members")
        if len(set(self.members)) != len(self.members):
            raise RegistryValidationError("fusion members must be unique")
        for member in self.members:
            _validate_name(member, "fusion member")
        if not isinstance(self.provenance_families, tuple) or not self.provenance_families:
            raise RegistryValidationError("fusion sets require provenance families")
        for family in self.provenance_families:
            _validate_name(family, "provenance family")
        if self.insertion_point not in INSERTION_POINTS:
            raise RegistryValidationError(f"unknown insertion point {self.insertion_point!r}")
        if self.resolution not in RESOLUTIONS or _EXPECTED_RESOLUTION[self.insertion_point] != self.resolution:
            raise RegistryValidationError("fusion insertion point and resolution disagree")
        if self.head not in POSITIVE_FUSION_HEADS:
            raise RegistryValidationError(f"unknown positive fusion head {self.head!r}")
        if self.access_scope not in ACCESS_SCOPES:
            raise RegistryValidationError(f"unknown access scope {self.access_scope!r}")
        if self.orientation not in ORIENTATIONS:
            raise RegistryValidationError(f"unknown orientation {self.orientation!r}")
        if self.mask_semantics not in MASK_SEMANTICS:
            raise RegistryValidationError(f"unknown mask semantics {self.mask_semantics!r}")
        if self.status not in STATUSES:
            raise RegistryValidationError(f"unknown status {self.status!r}")
        if not isinstance(self.provenance, tuple) or not self.provenance:
            raise RegistryValidationError("provenance must be a nonempty tuple")
        for item in self.provenance:
            _validate_text(item, "provenance entry")
        _validate_hash(self.implementation_hash, "implementation_hash")
        _validate_hash(self.semantic_hash, "semantic_hash")
        _validate_hash(self.fitted_weight_hash, "fitted_weight_hash", optional=True)

    @property
    def fusion_id(self) -> str:
        return self.name

    @property
    def spec_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _spec_payload(self)

    def bind_weights(self, weights: Any) -> "FusionSetSpec":
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(self.members),) or not np.isfinite(values).all():
            raise RegistryValidationError("fitted weights must be finite and align with members")
        if (values < -EPS).any() or not np.isclose(values.sum(), 1.0, atol=1e-10):
            raise RegistryValidationError("positive fusion weights must be nonnegative and sum to one")
        return replace(self, fitted_weight_hash=array_sha256(values))


def _compatibility_key(spec: SignalSpec) -> tuple[str, str, str]:
    return spec.insertion_point, spec.resolution, spec.target


def validate_fusion_members(members: Sequence[SignalSpec]) -> tuple[SignalSpec, ...]:
    """Reject cross-point, cross-resolution, or cross-target flat fusion."""

    values = tuple(members)
    if not 1 <= len(values) <= 6:
        raise RegistryValidationError("fusion needs one to six signal specs")
    if len({value.name for value in values}) != len(values):
        raise RegistryValidationError("fusion members must be distinct")
    expected = _compatibility_key(values[0])
    for value in values:
        if value.status != "ELIGIBLE":
            raise RegistryValidationError(f"{value.name} is not eligible for fusion")
        if _compatibility_key(value) != expected:
            raise RegistryValidationError(
                "flat fusion cannot mix insertion points, resolutions, or targets"
            )
        allowed_orientation = (
            {"preserve"} if value.insertion_point == "background"
            else {"high_is_risk", "anchor_oriented"}
        )
        if value.orientation not in allowed_orientation:
            raise RegistryValidationError(f"{value.name} is not risk-oriented")
    return values


_ACCESS_PRECEDENCE = {
    "answer_only": 0,
    "historical_artifact": 1,
    "source_excluded": 2,
    "pair_excluded": 3,
    "not_available": 4,
}


def make_fusion_set_spec(
    name: str,
    members: Sequence[SignalSpec],
    *,
    head: str,
    provenance: Sequence[str] = ("docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md",),
) -> FusionSetSpec:
    values = validate_fusion_members(members)
    families = tuple(dict.fromkeys(value.provenance_family for value in values))
    access_scope = max(values, key=lambda value: _ACCESS_PRECEDENCE[value.access_scope]).access_scope
    semantic = {
        "members": [value.semantic_hash for value in values],
        "head": head,
        "point": values[0].insertion_point,
        "resolution": values[0].resolution,
        "target": values[0].target,
    }
    return FusionSetSpec(
        name=name,
        members=tuple(value.name for value in values),
        provenance_families=families,
        insertion_point=values[0].insertion_point,
        resolution=values[0].resolution,
        target=values[0].target,
        head=head,
        access_scope=access_scope,
        orientation="preserve" if values[0].insertion_point == "background" else "high_is_risk",
        mask_semantics="propagate",
        status="ELIGIBLE",
        provenance=tuple(provenance),
        implementation_hash=callable_sha256(
            equal_rank_fusion if head == "equal_rank" else
            family_equal_fusion if head == "family_equal" else
            nonnegative_shrunk_simplex_fusion
        ),
        semantic_hash=_semantic_hash(canonical_json(semantic)),
    )


def deduplicate_signal_specs(
    specs: Sequence[SignalSpec],
    *,
    include_registered_aliases: bool = True,
) -> tuple[tuple[SignalSpec, ...], dict[str, str]]:
    """Return deterministic canonical specs and ``alias -> canonical`` names.

    Bound score hashes take precedence.  Before materialization, deliberately
    shared semantic hashes encode only registered exact identities.  A shared
    identity across incompatible fusion contracts fails closed.
    """

    aliases: dict[str, str] = {}
    by_name: dict[str, SignalSpec] = {}
    by_identity: dict[tuple[str, str], list[tuple[int, SignalSpec]]] = {}
    for index, spec in enumerate(specs):
        previous = by_name.get(spec.name)
        if previous is not None:
            if previous != spec:
                raise RegistryValidationError(f"conflicting definitions for signal {spec.name}")
            continue
        by_name[spec.name] = spec
        if spec.score_hash is not None:
            key = ("score", spec.score_hash)
        elif include_registered_aliases:
            key = ("semantic", spec.semantic_hash)
        else:
            key = ("name", spec.name)
        by_identity.setdefault(key, []).append((index, spec))

    status_priority = {"ELIGIBLE": 0, "REPORT_ONLY": 1, "PENDING_EXPANSION": 2}
    canonical = []
    for group in by_identity.values():
        # Status first prevents an opaque REPORT_ONLY artifact from shadowing a
        # replayable signal.  Name breaks equal-status ties independently of
        # caller order, making canonical serialization reproducible.
        _, leader = min(
            group, key=lambda item: (status_priority[item[1].status], item[1].name),
        )
        for _, spec in group:
            if _compatibility_key(leader) != _compatibility_key(spec):
                raise RegistryValidationError(
                    f"hash-identical signals {leader.name} and {spec.name} have incompatible contracts"
                )
            if leader.orientation != spec.orientation or leader.mask_semantics != spec.mask_semantics:
                raise RegistryValidationError(
                    f"hash-identical signals {leader.name} and {spec.name} disagree on semantics"
                )
            if spec.name != leader.name:
                aliases[spec.name] = leader.name
        canonical.append(leader)
    return tuple(sorted(canonical, key=lambda spec: spec.name)), aliases


@dataclass(frozen=True, slots=True)
class FusionSignalRegistry:
    """Immutable canonical registry with resolvable exact aliases."""

    signals: tuple[SignalSpec, ...]
    readouts: tuple[ReadoutSpec, ...] = ()
    fusion_sets: tuple[FusionSetSpec, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RegistryValidationError("registry schema version mismatch")
        if not isinstance(self.signals, tuple) or not isinstance(self.readouts, tuple) \
                or not isinstance(self.fusion_sets, tuple) or not isinstance(self.aliases, tuple):
            raise RegistryValidationError("registry collections must be immutable tuples")
        signal_map = _unique_specs(self.signals, "signal")
        _unique_specs(self.readouts, "readout")
        _unique_specs(self.fusion_sets, "fusion set")
        alias_map: dict[str, str] = {}
        for pair in self.aliases:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise RegistryValidationError("aliases must be (alias, canonical) tuples")
            alias, canonical = pair
            _validate_name(alias, "alias")
            if alias in signal_map or alias in alias_map:
                raise RegistryValidationError(f"duplicate alias {alias}")
            if canonical not in signal_map:
                raise RegistryValidationError(f"alias target {canonical} is absent")
            alias_map[alias] = canonical
        for fusion in self.fusion_sets:
            resolved = []
            for name in fusion.members:
                canonical = alias_map.get(name, name)
                if canonical not in signal_map:
                    raise RegistryValidationError(f"fusion member {name} is absent")
                resolved.append(signal_map[canonical])
            compatible = validate_fusion_members(resolved)
            if (fusion.insertion_point, fusion.resolution, fusion.target) != _compatibility_key(compatible[0]):
                raise RegistryValidationError(f"fusion metadata drift for {fusion.name}")
            if tuple(dict.fromkeys(value.provenance_family for value in compatible)) \
                    != fusion.provenance_families:
                raise RegistryValidationError(f"fusion provenance drift for {fusion.name}")
            expected_access = max(
                compatible, key=lambda value: _ACCESS_PRECEDENCE[value.access_scope],
            ).access_scope
            if fusion.access_scope != expected_access:
                raise RegistryValidationError(f"fusion access-scope drift for {fusion.name}")

    @classmethod
    def build(
        cls,
        signals: Sequence[SignalSpec],
        readouts: Sequence[ReadoutSpec] = (),
        fusion_sets: Sequence[FusionSetSpec] = (),
        *,
        include_registered_aliases: bool = True,
    ) -> "FusionSignalRegistry":
        canonical, aliases = deduplicate_signal_specs(
            signals, include_registered_aliases=include_registered_aliases,
        )
        return cls(
            signals=canonical,
            readouts=tuple(readouts),
            fusion_sets=tuple(fusion_sets),
            aliases=tuple(sorted(aliases.items())),
        )

    @property
    def registry_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def alias_map(self) -> dict[str, str]:
        return dict(self.aliases)

    def signal(self, name: str) -> SignalSpec:
        canonical = self.alias_map.get(name, name)
        for spec in self.signals:
            if spec.name == canonical:
                return spec
        raise KeyError(name)

    def readout(self, name: str) -> ReadoutSpec:
        for spec in self.readouts:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signals": [spec.to_dict() for spec in sorted(self.signals, key=lambda item: item.name)],
            "readouts": [spec.to_dict() for spec in sorted(self.readouts, key=lambda item: item.name)],
            "fusion_sets": [spec.to_dict() for spec in sorted(self.fusion_sets, key=lambda item: item.name)],
            "aliases": {alias: canonical for alias, canonical in sorted(self.aliases)},
        }


def _unique_specs(specs: Sequence[Any], kind: str) -> dict[str, Any]:
    output = {}
    for spec in specs:
        if spec.name in output:
            raise RegistryValidationError(f"duplicate {kind} name {spec.name}")
        output[spec.name] = spec
    return output


# ---------------------------------------------------------------------------
# Causal transforms


def _matrix_and_mask(
    values: Any, active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("values must be a nonempty vector or matrix")
    if active_mask is None:
        mask = np.ones(array.shape, dtype=bool)
    else:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.ndim == 1 and len(mask) == len(array):
            mask = np.broadcast_to(mask[:, None], array.shape).copy()
        if mask.shape != array.shape:
            raise ValueError("active mask does not align with values")
    if not np.isfinite(array[mask]).all():
        raise ValueError("active values must be finite")
    clean = np.where(mask, array, 0.0)
    return clean, mask, was_vector


def _restore_shape(values: np.ndarray, mask: np.ndarray, was_vector: bool) -> tuple[np.ndarray, np.ndarray]:
    if was_vector:
        return values[:, 0], mask[:, 0]
    return values, mask


def prefix_mean_background(
    values: Any, active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of strictly earlier active observations; token zero is inactive."""

    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    prefix_sum = np.cumsum(array, axis=0) - array
    prefix_count = np.cumsum(observed, axis=0) - observed
    available = observed & (prefix_count > 0)
    background = np.divide(
        prefix_sum, prefix_count, out=np.zeros_like(array), where=prefix_count > 0,
    )
    background[~available] = 0.0
    return _restore_shape(background, available, was_vector)


def prefix_mean_innovation(
    values: Any, active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    background, available = prefix_mean_background(array, observed)
    if background.ndim == 1:
        background = background[:, None]
        available = available[:, None]
    innovation = np.where(available, array - background, 0.0)
    return _restore_shape(innovation, available, was_vector)


def trailing_mean_background(
    values: Any,
    window: int = 16,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over the preceding ``window`` token positions, excluding current."""

    if not isinstance(window, (int, np.integer)) or int(window) <= 0:
        raise ValueError("window must be a positive integer")
    window = int(window)
    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    sums = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)))
    counts = np.vstack((np.zeros((1, array.shape[1]), dtype=np.int64), np.cumsum(observed, axis=0)))
    output = np.zeros_like(array)
    available = np.zeros_like(observed)
    for index in range(len(array)):
        start = max(0, index - window)
        total = sums[index] - sums[start]
        count = counts[index] - counts[start]
        live = observed[index] & (count > 0)
        output[index] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        available[index] = live
    output[~available] = 0.0
    return _restore_shape(output, available, was_vector)


def trailing_mean_innovation(
    values: Any,
    window: int = 16,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    background, available = trailing_mean_background(array, window, observed)
    if background.ndim == 1:
        background = background[:, None]
        available = available[:, None]
    innovation = np.where(available, array - background, 0.0)
    return _restore_shape(innovation, available, was_vector)


def causal_prefix_topk_background(
    values: Any,
    count: int = 10,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k mean of strictly earlier active observations in O(T P log k)."""

    if not isinstance(count, (int, np.integer)) or int(count) <= 0:
        raise ValueError("count must be a positive integer")
    count = int(count)
    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    output = np.zeros_like(array)
    available = np.zeros_like(observed)
    heaps: list[list[float]] = [[] for _ in range(array.shape[1])]
    totals = np.zeros(array.shape[1], dtype=np.float64)
    for index in range(len(array)):
        for column, heap in enumerate(heaps):
            if observed[index, column] and heap:
                output[index, column] = totals[column] / len(heap)
                available[index, column] = True
            if not observed[index, column]:
                continue
            value = float(array[index, column])
            if len(heap) < count:
                heapq.heappush(heap, value)
                totals[column] += value
            elif value > heap[0]:
                totals[column] += value - heapq.heapreplace(heap, value)
    return _restore_shape(output, available, was_vector)


def causal_prefix_topk_innovation(
    values: Any,
    count: int = 10,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    array, observed, was_vector = _matrix_and_mask(values, active_mask)
    background, available = causal_prefix_topk_background(array, count, observed)
    if background.ndim == 1:
        background = background[:, None]
        available = available[:, None]
    innovation = np.where(available, array - background, 0.0)
    return _restore_shape(innovation, available, was_vector)


def tail15_causal_innovations(
    tail15: Any, active_mask: Any | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The two registered causal Tail15 innovation roles."""

    return {
        "prefix_mean": prefix_mean_innovation(tail15, active_mask),
        "prefix_top10": causal_prefix_topk_innovation(tail15, 10, active_mask),
    }


def digit_token_clock_innovation(
    disagreement: Any, opportunity: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Digit innovation against all prior token-clock observations.

    Only digit-opportunity tokens emit evidence.  Non-opportunity zeros are
    included in the token-clock background but remain inactive outputs.
    """

    event, chance = _validated_digit_inputs(disagreement, opportunity)
    background, history = prefix_mean_background(event)
    active = chance & history
    return np.where(active, event - background, 0.0), active


def digit_opportunity_clock_innovation(
    disagreement: Any, opportunity: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Digit innovation against strictly earlier digit opportunities only."""

    event, chance = _validated_digit_inputs(disagreement, opportunity)
    background, history = prefix_mean_background(event, chance)
    active = chance & history
    return np.where(active, event - background, 0.0), active


def digit_clock_innovations(
    disagreement: Any, opportunity: Any,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "token_clock": digit_token_clock_innovation(disagreement, opportunity),
        "opportunity_clock": digit_opportunity_clock_innovation(disagreement, opportunity),
    }


def _validated_digit_inputs(disagreement: Any, opportunity: Any) -> tuple[np.ndarray, np.ndarray]:
    event = np.asarray(disagreement)
    chance = np.asarray(opportunity, dtype=bool)
    if event.ndim != 1 or chance.shape != event.shape or not len(event):
        raise ValueError("digit event and opportunity must be aligned nonempty vectors")
    if not np.isfinite(event.astype(float)).all() or not np.isin(event, (0, 1, False, True)).all():
        raise ValueError("digit disagreement must be binary")
    event = event.astype(np.float64)
    if np.any((event > 0) & ~chance):
        raise ValueError("digit disagreement cannot occur without an opportunity")
    return event, chance


# ---------------------------------------------------------------------------
# Fixed readouts and decoders


READOUT_NAMES = (
    "top1", "top2", "top3", "top5", "top8", "top10",
    "mean", "median", "top25pct", "top50pct", "q75", "q90",
    "first4", "last4", "best_contiguous10",
)
READOUT_ALIASES = {
    "top25": "top25pct", "top25%": "top25pct", "top-25%": "top25pct",
    "top50": "top50pct", "top50%": "top50pct", "top-50%": "top50pct",
    "best-contiguous10": "best_contiguous10",
}
DECODER_NAMES = (
    "argmax", "first_near_max_025", "persistent_q90_3",
    "step_top5", "earlier_ve_peak",
)


def _validated_spans(spans: Any, token_count: int) -> np.ndarray:
    values = np.asarray(spans, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("spans must be a nonempty S-by-2 array")
    if np.any(values[:, 0] < 0) or np.any(values[:, 1] <= values[:, 0]) \
            or np.any(values[:, 1] > token_count):
        raise ValueError("invalid half-open token span")
    # Three frozen PRMBench answers contain adjacent logical steps that share
    # the same terminal token (the source expands an otherwise empty step to a
    # one-token half-open span).  A readout is defined independently on every
    # logical step, so overlap is legal; only a reversal of either boundary is
    # not.  This preserves the source's step cardinality and label alignment.
    if len(values) > 1 and (
        np.any(values[1:, 0] < values[:-1, 0])
        or np.any(values[1:, 1] < values[:-1, 1])
    ):
        raise ValueError("step spans must have monotone chronological boundaries")
    return values


def _top_mean(values: np.ndarray, count: int) -> float:
    keep = min(int(count), len(values))
    return float(np.partition(values, len(values) - keep)[-keep:].mean())


def _best_contiguous_mean(values: np.ndarray, active: np.ndarray, count: int) -> float:
    best = -np.inf
    start = 0
    while start < len(values):
        while start < len(values) and not active[start]:
            start += 1
        end = start
        while end < len(values) and active[end]:
            end += 1
        if end > start:
            run = values[start:end]
            width = min(count, len(run))
            if len(run) == width:
                candidate = float(run.mean())
            else:
                sums = np.cumsum(np.r_[0.0, run])
                candidate = float(np.max((sums[width:] - sums[:-width]) / width))
            best = max(best, candidate)
        start = end + 1
    if not np.isfinite(best):
        raise ValueError("best contiguous readout has no active values")
    return best


def _readout_value(values: np.ndarray, active: np.ndarray, name: str) -> float:
    canonical = READOUT_ALIASES.get(name, name)
    if canonical not in READOUT_NAMES:
        raise ValueError(f"unknown readout {name!r}")
    selected = values[active]
    if not len(selected):
        raise ValueError("readout has no active values")
    if canonical.startswith("top") and canonical[3:].isdigit():
        return _top_mean(selected, int(canonical[3:]))
    if canonical == "mean":
        return float(selected.mean())
    if canonical == "median":
        return float(np.median(selected))
    if canonical == "top25pct":
        return _top_mean(selected, max(1, int(math.ceil(0.25 * len(selected)))))
    if canonical == "top50pct":
        return _top_mean(selected, max(1, int(math.ceil(0.50 * len(selected)))))
    if canonical == "q75":
        return float(np.quantile(selected, 0.75))
    if canonical == "q90":
        return float(np.quantile(selected, 0.90))
    if canonical == "first4":
        return float(selected[:4].mean())
    if canonical == "last4":
        return float(selected[-4:].mean())
    if canonical == "best_contiguous10":
        return _best_contiguous_mean(values, active, 10)
    raise AssertionError(canonical)


def readout_steps(
    values: Any,
    spans: Any,
    readout: str,
    *,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one of the exact 15 readouts independently within each step."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("step readout requires a nonempty token vector")
    if active_mask is None:
        active = np.ones(len(array), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != array.shape:
            raise ValueError("readout mask does not align with token values")
    if not np.isfinite(array[active]).all():
        raise ValueError("active readout values must be finite")
    steps = _validated_spans(spans, len(array))
    output = np.zeros(len(steps), dtype=np.float64)
    available = np.zeros(len(steps), dtype=bool)
    for index, (start, end) in enumerate(steps):
        local_active = active[start:end]
        if not local_active.any():
            continue
        output[index] = _readout_value(array[start:end], local_active, readout)
        available[index] = True
    return output, available


def apply_readout(
    values: Any,
    spans: Any,
    readout: str,
    *,
    active_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Named public alias for :func:`readout_steps`."""

    return readout_steps(values, spans, readout, active_mask=active_mask)


def answer_top10_minus_mean(values: Any, active_mask: Any | None = None) -> float:
    """Answer-only Tail15 prominence; never a step-varying locator score."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("answer prominence requires a nonempty vector")
    active = np.ones(len(array), dtype=bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != array.shape or not active.any():
        raise ValueError("answer prominence requires at least one aligned active token")
    selected = array[active]
    if not np.isfinite(selected).all():
        raise ValueError("answer prominence values must be finite")
    return _top_mean(selected, 10) - float(selected.mean())


def decode_argmax(scores: Any, active_mask: Any | None = None) -> int:
    values, active = _decoder_vector(scores, active_mask)
    if not active.any():
        return -1
    indexes = np.flatnonzero(active)
    return int(indexes[np.argmax(values[indexes])])


def decode_first_near_max(
    scores: Any, fraction: float = 0.25, active_mask: Any | None = None,
) -> int:
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError("near-max fraction must be nonnegative")
    values, active = _decoder_vector(scores, active_mask)
    if not active.any():
        return -1
    indexes = np.flatnonzero(active)
    selected = values[indexes]
    threshold = selected.max() - float(fraction) * selected.std()
    return int(indexes[np.flatnonzero(selected >= threshold)[0]])


def decode_persistent_q90_3(scores: Any, active_mask: Any | None = None) -> int:
    values, active = _decoder_vector(scores, active_mask)
    if not active.any():
        return -1
    threshold = float(np.quantile(values[active], 0.90))
    hits = active & (values >= threshold)
    for index in range(2, len(hits)):
        if bool(hits[index - 2:index + 1].all()):
            return index - 2
    return decode_argmax(values, active)


def decode_step_top5(
    token_scores: Any, spans: Any, active_mask: Any | None = None,
) -> int:
    scores, active = readout_steps(token_scores, spans, "top5", active_mask=active_mask)
    return decode_argmax(scores, active)


def decode_earlier_ve_peak(
    ve0_step_scores: Any,
    ve075_step_scores: Any,
    ve0_active: Any | None = None,
    ve075_active: Any | None = None,
) -> int:
    left = decode_argmax(ve0_step_scores, ve0_active)
    right = decode_argmax(ve075_step_scores, ve075_active)
    if left < 0:
        return right
    if right < 0:
        return left
    return min(left, right)


def _decoder_vector(scores: Any, active_mask: Any | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("decoder requires a nonempty score vector")
    if active_mask is None:
        active = np.ones(len(values), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != values.shape:
            raise ValueError("decoder mask does not align with scores")
    if not np.isfinite(values[active]).all():
        raise ValueError("active decoder scores must be finite")
    return values, active


# ---------------------------------------------------------------------------
# Positive fusion heads


@dataclass(frozen=True, slots=True)
class FusionResult:
    score: np.ndarray
    active: np.ndarray
    weights: np.ndarray
    active_members: np.ndarray
    diagnostics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        score = np.asarray(self.score, dtype=np.float64).copy()
        active = np.asarray(self.active, dtype=bool).copy()
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        members = np.asarray(self.active_members, dtype=bool).copy()
        if score.ndim != 1 or active.shape != score.shape or weights.ndim != 1 \
                or members.shape != weights.shape:
            raise ValueError("invalid fusion result shapes")
        if not np.isfinite(score).all() or not np.isfinite(weights).all():
            raise ValueError("fusion result must be finite")
        for array in (score, active, weights, members):
            array.setflags(write=False)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active_members", members)

    @property
    def diagnostic_map(self) -> dict[str, Any]:
        return dict(self.diagnostics)

    def __iter__(self):
        yield self.score
        yield self.active


def _score_matrix_and_mask(
    scores: Any, active_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("fusion scores must be a nonempty observation-by-member matrix")
    if active_mask is None:
        if not np.isfinite(values).all():
            raise ValueError("nonfinite fusion scores require an explicit inactive mask")
        active = np.ones(values.shape, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != values.shape:
            raise ValueError("fusion mask does not align with scores")
        if not np.isfinite(values[active]).all():
            raise ValueError("active fusion scores must be finite")
    return np.where(active, values, 0.0), active


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks / len(values)


def _rank_matrix(
    scores: Any, active_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, active = _score_matrix_and_mask(scores, active_mask)
    ranks = np.zeros_like(values)
    live = np.zeros(values.shape[1], dtype=bool)
    for column in range(values.shape[1]):
        rows = np.flatnonzero(active[:, column])
        if len(rows) < 2 or np.ptp(values[rows, column]) <= EPS:
            active[:, column] = False
            continue
        ranks[rows, column] = _midranks(values[rows, column])
        live[column] = True
    return ranks, active, live


def _apply_weights(
    values: np.ndarray, active: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    available_weight = active @ weights
    live = available_weight > EPS
    output = np.zeros(len(values), dtype=np.float64)
    output[live] = np.sum(values[live] * active[live] * weights, axis=1) / available_weight[live]
    return output, live


def equal_rank_fusion(
    scores: Any, *, active_mask: Any | None = None,
) -> FusionResult:
    ranks, active, live = _rank_matrix(scores, active_mask)
    weights = np.zeros(ranks.shape[1], dtype=np.float64)
    if live.any():
        weights[live] = 1.0 / live.sum()
    fused, available = _apply_weights(ranks, active, weights)
    return FusionResult(
        fused, available, weights, live,
        (("head", "equal_rank"), ("inactive_members", int((~live).sum()))),
    )


def family_equal_fusion(
    scores: Any,
    families: Sequence[str],
    *,
    active_mask: Any | None = None,
) -> FusionResult:
    ranks, active, live = _rank_matrix(scores, active_mask)
    family = np.asarray(tuple(str(value) for value in families), dtype=object)
    if family.shape != (ranks.shape[1],) or any(not value for value in family):
        raise ValueError("families must align one-to-one with fusion members")
    active_families = tuple(dict.fromkeys(family[live].tolist()))
    weights = np.zeros(ranks.shape[1], dtype=np.float64)
    if active_families:
        for name in active_families:
            indexes = live & (family == name)
            weights[indexes] = 1.0 / (len(active_families) * indexes.sum())
    # Preserve equal family mass even when a row masks only part of a family.
    output = np.zeros(len(ranks), dtype=np.float64)
    available = np.zeros(len(ranks), dtype=bool)
    for row in range(len(ranks)):
        family_values = []
        for name in active_families:
            indexes = active[row] & live & (family == name)
            if indexes.any():
                family_values.append(float(ranks[row, indexes].mean()))
        if family_values:
            output[row] = float(np.mean(family_values))
            available[row] = True
    return FusionResult(
        output, available, weights, live,
        (("head", "family_equal"), ("active_families", len(active_families))),
    )


def _ledoit_wolf_diagonal(Z: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, float]:
    target = np.diag(np.diag(covariance))
    changed = ~np.eye(len(covariance), dtype=bool)
    denominator = float(np.square(covariance - target)[changed].sum())
    if denominator <= EPS or len(Z) < 3:
        alpha = 0.0
    else:
        z2 = Z * Z
        variance = np.maximum(
            (z2.T @ z2 - len(Z) * covariance * covariance)
            / (len(Z) * max(len(Z) - 1, 1)),
            0.0,
        )
        alpha = float(np.clip(variance[changed].sum() / denominator, 0.0, 1.0))
    return (1.0 - alpha) * covariance + alpha * target, alpha


def _minimum_variance_simplex(covariance: np.ndarray) -> np.ndarray:
    """Exact active-support enumeration; Atlas groups have at most six members."""

    p = len(covariance)
    if p == 1:
        return np.ones(1, dtype=np.float64)
    scale = max(float(np.trace(covariance)) / p, 1.0)
    matrix = 0.5 * (covariance + covariance.T) + 1e-10 * scale * np.eye(p)
    best_weight = None
    best_value = np.inf
    for bits in range(1, 1 << p):
        indexes = np.asarray([index for index in range(p) if bits & (1 << index)], dtype=int)
        submatrix = matrix[np.ix_(indexes, indexes)]
        ones = np.ones(len(indexes), dtype=np.float64)
        try:
            candidate = np.linalg.solve(submatrix, ones)
        except np.linalg.LinAlgError:
            candidate = np.linalg.pinv(submatrix, rcond=1e-12) @ ones
        denominator = float(candidate.sum())
        if not np.isfinite(candidate).all() or denominator <= EPS:
            continue
        candidate /= denominator
        if np.min(candidate) < -1e-9:
            continue
        candidate = np.maximum(candidate, 0.0)
        candidate /= candidate.sum()
        full = np.zeros(p, dtype=np.float64)
        full[indexes] = candidate
        objective = float(full @ matrix @ full)
        if objective < best_value - 1e-15:
            best_weight, best_value = full, objective
    if best_weight is None:
        return np.full(p, 1.0 / p)
    return best_weight


def nonnegative_shrunk_simplex_fusion(
    scores: Any, *, active_mask: Any | None = None,
) -> FusionResult:
    """Fit a label-free nonnegative simplex on shrunk rank covariance."""

    ranks, active, live = _rank_matrix(scores, active_mask)
    weights = np.zeros(ranks.shape[1], dtype=np.float64)
    live_indexes = np.flatnonzero(live)
    alpha = 0.0
    if len(live_indexes) == 1:
        weights[live_indexes[0]] = 1.0
    elif len(live_indexes) > 1:
        Z = np.zeros((len(ranks), len(live_indexes)), dtype=np.float64)
        for local, column in enumerate(live_indexes):
            rows = active[:, column]
            mean = float(ranks[rows, column].mean())
            scale = float(ranks[rows, column].std())
            Z[rows, local] = (ranks[rows, column] - mean) / max(scale, EPS)
        covariance = Z.T @ Z / len(Z)
        shrunk, alpha = _ledoit_wolf_diagonal(Z, covariance)
        weights[live_indexes] = _minimum_variance_simplex(shrunk)
    fused, available = _apply_weights(ranks, active, weights)
    return FusionResult(
        fused, available, weights, live,
        (
            ("head", "nonnegative_shrunk_simplex"),
            ("shrinkage_alpha", alpha),
            ("inactive_members", int((~live).sum())),
        ),
    )


# ---------------------------------------------------------------------------
# Built-in Atlas roster


Q15_PRIMITIVE_NAMES = (
    "q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1",
)
Q15_PREFIX_INNOVATION_NAMES = tuple(f"{name}.prefix_mean_innovation" for name in Q15_PRIMITIVE_NAMES)
BACKGROUND_KINDS = (
    "prefix_mean", "mean16", "no_reset", "bocpd_hazard_1_32",
    "source_excluded_ridge", "source_excluded_tcn",
)
BACKGROUND_SIGNAL_NAMES = tuple(
    f"background.{primitive.split('.', 1)[1]}.{kind}"
    for primitive in Q15_PRIMITIVE_NAMES
    for kind in BACKGROUND_KINDS
)
_RENYI_ALPHA_NAMES = (
    "a0.001", "a0.01", "a0.02", "a0.05", "a0.1", "a0.15", "a0.2",
    "a0.25", "a0.3", "a0.4", "a0.5", "a0.75", "H1", "a1.5",
    "a2", "a3", "a4", "a8", "Hinf",
)
_ESCORT_NAMES = (
    "ve0", "ve0.1", "ve0.25", "ve0.5", "ve0.75", "ve1",
    "ve1.5", "ve2", "ve3", "ve4", "ve8",
)
RENYI_ESCORT_NAMES = tuple(
    f"renyi_escort.{name}" for name in (("H0lim",) + _RENYI_ALPHA_NAMES + _ESCORT_NAMES)
)
Q50_H1_HINF_NAMES = ("q50.VE1", "q15.H1_native", "q15.Hinf")
STEP395_SIGNAL_NAMES = tuple(
    f"step395.{name}" for name in (
        "surprisal", "rank", "mass_above", "gap", "logtail15", "logtail50",
        "digit", "tail15", "tail50",
    )
)
DIRECT_PROBABILITY_NAMES = tuple(
    [f"direct_probability.rank_{index}_risk" for index in range(1, 16)]
    + ["direct_probability.selected_token_surprisal", "direct_probability.residual_tail_mass"]
)
DIGIT_SIGNAL_NAMES = (
    "digit.disagreement", "digit.opportunity", "digit.count", "digit.rate",
    "digit.presence", "digit.permuted_location_control",
)
DIGIT_INNOVATION_NAMES = (
    "digit.token_clock_innovation", "digit.opportunity_clock_innovation",
)
TAIL15_ROLE_NAMES = (
    "tail15.level", "tail15.prefix_mean_innovation",
    "tail15.prefix_top10_innovation", "tail15.answer_prominence",
)
PENDING_EXPANSION_NAMES = ("pending.fm", "pending.diflo", "pending.dot")


def _signal(
    name: str,
    family: str,
    semantic_identity: str,
    *,
    transform: str,
    status: str = "ELIGIBLE",
    access_scope: str = "answer_only",
    orientation: str = "high_is_risk",
    mask_semantics: str = "all_active",
    insertion_point: str = "token_pre_readout",
    resolution: str = "token",
    target: str = "localization_risk",
    provenance: Sequence[str] = ("docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md",),
) -> SignalSpec:
    return SignalSpec(
        name=name,
        provenance_family=family,
        insertion_point=insertion_point,
        resolution=resolution,
        target=target,
        transform=transform,
        access_scope=access_scope,
        orientation=orientation,
        mask_semantics=mask_semantics,
        status=status,
        provenance=tuple(provenance),
        implementation_hash=_semantic_hash(f"implementation:{transform}"),
        semantic_hash=_semantic_hash(semantic_identity),
    )


def report_only_signal_spec(
    name: str,
    provenance_family: str,
    *,
    transform: str,
    semantic_identity: str | None = None,
    provenance: Sequence[str],
    insertion_point: str = "step_post_readout",
    resolution: str = "step",
    target: str = "localization_risk",
    orientation: str = "high_is_risk",
    mask_semantics: str = "artifact_defined",
    access_scope: str = "historical_artifact",
    score_hash: str | None = None,
) -> SignalSpec:
    """Create a visible, ineligible historical artifact record."""

    spec = _signal(
        name, provenance_family, semantic_identity or f"report-only:{name}",
        transform=transform, status="REPORT_ONLY", access_scope=access_scope,
        orientation=orientation, mask_semantics=mask_semantics,
        insertion_point=insertion_point, resolution=resolution, target=target,
        provenance=provenance,
    )
    if score_hash is not None:
        _validate_hash(score_hash, "score_hash")
        spec = replace(spec, score_hash=score_hash)
    return spec


def pending_expansion_signal_spec(
    name: str,
    provenance_family: str,
    *,
    provenance: Sequence[str] = ("docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md",),
) -> SignalSpec:
    """Create a non-runnable expansion hook; this does not authorize a fit."""

    return _signal(
        name, provenance_family, f"pending:{name}", transform="artifact unavailable",
        status="PENDING_EXPANSION", access_scope="not_available",
        orientation="not_applicable", mask_semantics="unavailable",
        provenance=provenance,
    )


def builtin_signal_specs() -> tuple[SignalSpec, ...]:
    output: list[SignalSpec] = []
    primitive = {
        "q15.H0lim": ("renyi_entropy", "q15:H0lim", "high_is_risk"),
        "q15.VE0": ("escort_varentropy", "q15:VE0", "anchor_oriented"),
        "q15.VE0.75": ("escort_varentropy", "q15:VE0.75", "anchor_oriented"),
        "q15.VE1": ("escort_varentropy", "q15:VE1", "high_is_risk"),
    }
    for name, (family, identity, orientation) in primitive.items():
        output.append(_signal(
            name, family, identity, transform="frozen q15 primitive level",
            orientation=orientation,
        ))
        output.append(_signal(
            f"{name}.prefix_mean_innovation", family,
            f"{identity}:prefix-mean-innovation",
            transform="observed primitive minus strictly prior prefix mean",
            orientation=orientation, mask_semantics="first_token_inactive",
        ))

    background_contract = {
        "prefix_mean": ("prefix_mean", "strictly prior prefix mean", "answer_only"),
        "mean16": ("mean16", "strictly prior trailing mean over 16 token positions", "answer_only"),
        "no_reset": ("no_reset", "causal no-reset Bayesian predictive mean", "answer_only"),
        "bocpd_hazard_1_32": (
            "bocpd", "causal Gaussian BOCPD predictive mean with hazard 1/32", "answer_only",
        ),
        "source_excluded_ridge": (
            "ridge", "past-only Ridge prediction fitted outside held source groups", "source_excluded",
        ),
        "source_excluded_tcn": (
            "tcn", "past-only four-target TCN prediction fitted outside held source groups", "source_excluded",
        ),
    }
    for primitive in Q15_PRIMITIVE_NAMES:
        target = primitive.split(".", 1)[1]
        for kind in BACKGROUND_KINDS:
            family, transform, access = background_contract[kind]
            output.append(_signal(
                f"background.{target}.{kind}", family,
                f"background:{target}:{kind}", transform=transform,
                access_scope=access, orientation="preserve",
                mask_semantics="first_token_inactive",
                insertion_point="background", resolution="token", target=target,
            ))

    for full_name in RENYI_ESCORT_NAMES:
        local = full_name.split(".", 1)[1]
        if local == "H0lim":
            family, identity, orientation = "renyi_entropy", "q15:H0lim", "high_is_risk"
        elif local.startswith("ve"):
            alpha = local[2:]
            identity = {
                "0": "q15:VE0", "0.75": "q15:VE0.75", "1": "q15:VE1",
            }.get(alpha, f"q15:escort:{alpha}")
            family, orientation = "escort_varentropy", "anchor_oriented"
            if alpha == "1":
                # VE1 is the answer-local orientation anchor itself.
                orientation = "high_is_risk"
        else:
            identity = {"H1": "q15:H1", "Hinf": "q15:Hinf"}.get(local, f"q15:renyi:{local}")
            family, orientation = "renyi_entropy", "high_is_risk"
        output.append(_signal(
            full_name, family, identity,
            transform=f"q15 Renyi/escort stream {local}", orientation=orientation,
        ))

    output.extend((
        _signal(
            "q50.VE1", "escort_varentropy", "q50:VE1",
            transform="q50 escort varentropy alpha 1", orientation="anchor_oriented",
        ),
        _signal("q15.H1_native", "renyi_entropy", "q15:H1", transform="native top-15 Shannon entropy"),
        _signal("q15.Hinf", "renyi_entropy", "q15:Hinf", transform="q15 min entropy"),
    ))

    step395 = {
        "surprisal": ("direct_probability", "selected_token_surprisal"),
        "rank": ("provided_token_rank", "step395:provided-token-rank"),
        "mass_above": ("provided_token_rank", "step395:mass-above-provided"),
        "gap": ("direct_probability", "step395:top1-selected-gap"),
        "logtail15": ("tail_mass", "step395:logtail15"),
        "logtail50": ("tail_mass", "step395:logtail50"),
        "digit": ("digit", "digit:disagreement"),
        "tail15": ("tail_mass", "tail15:raw-missing-mass"),
        "tail50": ("tail_mass", "tail50:raw-missing-mass"),
    }
    for local, (family, identity) in step395.items():
        output.append(_signal(
            f"step395.{local}", family, identity,
            transform=f"corrected Step395 {local} token stream",
        ))

    for index in range(1, 16):
        output.append(_signal(
            f"direct_probability.rank_{index}_risk", "direct_probability",
            f"direct-probability:rank-{index}-risk", transform=f"raw probability rank {index} risk",
        ))
    output.extend((
        _signal(
            "direct_probability.selected_token_surprisal", "direct_probability",
            "selected_token_surprisal", transform="negative log probability of provided token",
        ),
        _signal(
            "direct_probability.residual_tail_mass", "tail_mass",
            "tail15:raw-missing-mass", transform="probability mass outside saved top-15",
        ),
    ))

    digit = {
        "disagreement": ("digit:disagreement", "digit disagreement event", "all_active", "ELIGIBLE"),
        "opportunity": ("digit:opportunity", "provided token is an ASCII digit", "all_active", "ELIGIBLE"),
        "count": ("digit:count", "per-step digit disagreement count", "propagate", "ELIGIBLE"),
        "rate": ("digit:rate", "disagreements divided by opportunities", "opportunity_only", "ELIGIBLE"),
        "presence": ("digit:presence", "any digit opportunity in step", "propagate", "ELIGIBLE"),
        "permuted_location_control": (
            "digit:permuted-location", "deterministic within-opportunity permutation control",
            "opportunity_only", "REPORT_ONLY",
        ),
    }
    for local, (identity, transform, mask, status) in digit.items():
        insertion, resolution = ("step_post_readout", "step") if local in {"count", "rate", "presence"} else ("token_pre_readout", "token")
        output.append(_signal(
            f"digit.{local}", "digit", identity, transform=transform,
            mask_semantics=mask, status=status,
            insertion_point=insertion, resolution=resolution,
        ))

    output.extend((
        _signal(
            "digit.token_clock_innovation", "digit", "digit:token-clock-innovation",
            transform="digit disagreement minus prior token-clock mean",
            mask_semantics="opportunity_only",
        ),
        _signal(
            "digit.opportunity_clock_innovation", "digit", "digit:opportunity-clock-innovation",
            transform="digit disagreement minus prior opportunity-clock mean",
            mask_semantics="opportunity_only",
        ),
        _signal(
            "tail15.level", "tail_mass", "tail15:raw-missing-mass",
            transform="correct residual mass outside top-15",
        ),
        _signal(
            "tail15.prefix_mean_innovation", "tail_mass", "tail15:prefix-mean-innovation",
            transform="Tail15 minus strictly prior prefix mean",
            mask_semantics="first_token_inactive",
        ),
        _signal(
            "tail15.prefix_top10_innovation", "tail_mass", "tail15:prefix-top10-innovation",
            transform="Tail15 minus strictly prior prefix Top10 mean",
            mask_semantics="first_token_inactive",
        ),
        _signal(
            "tail15.answer_prominence", "tail_mass", "tail15:answer-top10-minus-mean",
            transform="answer Top10 Tail15 mean minus answer Tail15 mean",
            insertion_point="answer_gate", resolution="answer",
        ),
    ))

    output.extend(pending_expansion_signal_spec(name, name.split(".", 1)[1]) for name in PENDING_EXPANSION_NAMES)
    if len(RENYI_ESCORT_NAMES) != 31 or len(DIRECT_PROBABILITY_NAMES) != 17 or len(STEP395_SIGNAL_NAMES) != 9:
        raise AssertionError("built-in roster cardinality drift")
    return tuple(output)


def builtin_readout_specs() -> tuple[ReadoutSpec, ...]:
    implementation = callable_sha256(readout_steps, _readout_value)
    specs = []
    for name in READOUT_NAMES:
        specs.append(ReadoutSpec(
            name=name,
            provenance_family="fixed_step_readout",
            insertion_point="step_post_readout",
            input_resolution="token",
            output_resolution="step",
            target="localization_risk",
            transform=f"fixed per-step {name} reduction",
            access_scope="answer_only",
            orientation="preserve",
            mask_semantics="propagate",
            chronological_semantics=(
                "order-sensitive within each step" if name in {"first4", "last4", "best_contiguous10"}
                else "order-invariant within each step"
            ),
            short_trace_rule="use all active tokens when fewer than the requested count",
            status="ELIGIBLE",
            provenance=("docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md",),
            implementation_hash=implementation,
            semantic_hash=_semantic_hash(f"readout:{name}"),
        ))
    decoder_impl = callable_sha256(
        decode_argmax, decode_first_near_max, decode_persistent_q90_3,
        decode_step_top5, decode_earlier_ve_peak,
    )
    decoder_rules = {
        "argmax": "first stable maximum",
        "first_near_max_025": "first score within 0.25 standard deviations of maximum",
        "persistent_q90_3": "first three-score q90 run, otherwise argmax",
        "step_top5": "argmax of per-step token Top5 means",
        "earlier_ve_peak": "earlier of VE0 and VE0.75 stable peaks",
    }
    for name in DECODER_NAMES:
        specs.append(ReadoutSpec(
            name=name,
            provenance_family="fixed_decoder",
            insertion_point="decoder",
            input_resolution="step" if name != "step_top5" else "token",
            output_resolution="decision",
            target="localization_risk",
            transform=decoder_rules[name],
            access_scope="answer_only",
            orientation="not_applicable",
            mask_semantics="propagate",
            chronological_semantics="stable earliest selection where the rule admits a tie",
            short_trace_rule="fall back to the stable argmax when persistence is unavailable",
            status="ELIGIBLE",
            provenance=("docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md",),
            implementation_hash=decoder_impl,
            semantic_hash=_semantic_hash(f"decoder:{name}"),
        ))
    return tuple(specs)


def build_builtin_registry() -> FusionSignalRegistry:
    return FusionSignalRegistry.build(builtin_signal_specs(), builtin_readout_specs())


BUILTIN_REGISTRY = build_builtin_registry()


__all__ = [
    "ACCESS_SCOPES", "BACKGROUND_KINDS", "BACKGROUND_SIGNAL_NAMES", "BUILTIN_REGISTRY",
    "DECODER_NAMES", "DIGIT_INNOVATION_NAMES",
    "DIGIT_SIGNAL_NAMES",
    "DIRECT_PROBABILITY_NAMES", "FusionResult", "FusionSetSpec", "FusionSignalRegistry",
    "INSERTION_POINTS", "MASK_SEMANTICS", "ORIENTATIONS", "PENDING_EXPANSION_NAMES",
    "POSITIVE_FUSION_HEADS", "Q15_PREFIX_INNOVATION_NAMES", "Q15_PRIMITIVE_NAMES",
    "Q50_H1_HINF_NAMES",
    "READOUT_ALIASES", "READOUT_NAMES", "RENYI_ESCORT_NAMES", "RESOLUTIONS",
    "RegistryValidationError", "SCHEMA_VERSION", "STATUSES", "STEP395_SIGNAL_NAMES",
    "TAIL15_ROLE_NAMES",
    "ReadoutSpec", "SignalSpec", "answer_top10_minus_mean", "apply_readout",
    "array_sha256", "build_builtin_registry", "builtin_readout_specs", "builtin_signal_specs",
    "callable_sha256", "canonical_json", "canonical_sha256", "causal_prefix_topk_background",
    "causal_prefix_topk_innovation", "decode_argmax", "decode_earlier_ve_peak",
    "decode_first_near_max", "decode_persistent_q90_3", "decode_step_top5",
    "deduplicate_signal_specs", "digit_clock_innovations",
    "digit_opportunity_clock_innovation", "digit_token_clock_innovation",
    "equal_rank_fusion", "family_equal_fusion", "make_fusion_set_spec",
    "nonnegative_shrunk_simplex_fusion", "pending_expansion_signal_spec",
    "prefix_mean_background", "prefix_mean_innovation", "readout_steps",
    "report_only_signal_spec", "tail15_causal_innovations", "trailing_mean_background",
    "trailing_mean_innovation", "validate_fusion_members",
]
