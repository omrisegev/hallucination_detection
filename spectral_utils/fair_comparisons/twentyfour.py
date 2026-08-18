"""Strict 24-cell Global adapter for Fair Paper-Exact Comparisons v1.

The historical 24-cell bundle contains scores and labels but no source-question
identifiers.  This module recovers identifiers only from the hash-frozen raw
telemetry sources used by Phase A5.  Because a source question can have multiple
candidate traces, its comparison-level ``source_question_id`` is the immutable raw
problem key plus ``::candidate<ordinal>``.  The ordinal is taken from the frozen raw
candidate array before admission; it is never inferred from a bundle position.
Canonical row IDs are therefore
``<cell_id>::<raw_problem_key>::candidate<ordinal>``.  The recovery is deliberately
two-stage:

1. apply the frozen A0 admission rule in canonical sorted-problem/candidate order;
2. match those rows to the bundle using label-free feature fingerprints.

Only after the resulting permutation is unique and complete may labels be opened.
This avoids the tempting but invalid positional fallback.  The one bundle cell with
no frozen A0 source specification, ``spilled_triviaqa_llama8b``, remains an explicit
blocked asset.

No fitting or method selection is performed here.  Unified-28 is deserialized from
its frozen JSON representation, while the three historical incumbents are read from
their immutable per-row score files.  Historical correctness-oriented incumbent
scores are negated so every output uses the fair-package convention 1 = error/risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from spectral_utils.a5_target_free_data import (
    CORE_FEATURES,
    FROZEN_A0_SOURCE_SPECS,
    TELEMETRY_KEYS,
    FrozenSourceSpec,
    _a0_admitted,
    _cropped_telemetry_only,
    _feature_row,
    _validated_telemetry,
    canonical_revision,
    normalize_question,
)
from spectral_utils.unified_causal_iu import (
    AccumulatorSpec,
    BaseReference,
    UnifiedCausalIU,
    all_feature_names,
)

from .evaluator import (
    calibrate_correct_only_threshold,
    detection_metrics,
    operating_point,
)
from .folds import canonical_sha256, ordered_id_sha256
from .registry import make_comparison_record, sha256_file


ADAPTER_REVISION = "fair_24cell_global_adapter_v1.0.0"
BUNDLE_RELATIVE_PATH = Path("results/dependency_fusion_raw/cells.npz")
BENCHMARK_RELATIVE_ROOT = Path("results/frozen_24cell_benchmark")
MODEL_RELATIVE_PATH = Path(
    "results/unified_causal_subset_validation_base7_dufs_llama31_v1/VALIDATION.json"
)
MODEL_RECORDS_RELATIVE_PATH = Path(
    "results/unified_causal_subset_validation_base7_dufs_llama31_v1/VALIDATION_RECORDS.jsonl"
)
MODEL_RUN_DEFINITION_RELATIVE_PATH = Path(
    "results/unified_causal_subset_validation_base7_dufs_llama31_v1/RUN_DEFINITION.json"
)

BUNDLE_SHA256 = "693a5b634f975ea32c7f840f3ab8366dd8ad638fe41cc76a60e24b1ac5a013e1"
MODEL_ARTIFACT_SHA256 = "49168791e0687a235793e3bc818c0d0f46875ce88202b7f1d66ff2a21485b2fa"
MODEL_RECORDS_SHA256 = "0c2917cbfe827cefa616eb8d161e0583674c5498e0b7a14cdebb10c7dda73dc8"
MODEL_RUN_DEFINITION_SHA256 = "deed54d9a8ed8652e8670c93e94c14847acee5a1d11cbd0a0e0fb4e69389d4ee"
SCORE_FREEZE_MANIFEST_SHA256 = "d72f2c801ebcdc776cc8c7f2d40a6bc6d1e19f1b0d13167e33f842a3e745cc29"
SCORE_RUN_DEFINITION_SHA256 = "8f8d5c08ddbee6a52401dbeeb1154e7fa8229db3c0004fe48d25dea606820b15"
SCORE_FIT_COMPLETE_SHA256 = "892fc47047dc645d4c96cb61d9e48ea1abd3fb415b1ad71cf80e29ea90fce0b9"
DATASET_REVISION = f"frozen-24cell@{BUNDLE_SHA256[:12]}"
POPULATION_ID = f"{DATASET_REVISION}::identity-proven-23cell"
BLOCKED_CELL = "spilled_triviaqa_llama8b"
EXPECTED_CELLS = 24
ELIGIBLE_CELLS = 23
EXPECTED_ROWS = 48_607
ELIGIBLE_ROWS = 48_351
IDENTITY_SIGNATURE_DECIMALS = 12
IDENTITY_ATOL = 1e-10

U28_METHOD_ID = "unified28"
IU_METHOD_ID = "mixed_v2_iu_pcr"
DEPLOYED_METHOD_ID = "deployed_upcr"
DUFS_METHOD_ID = "mixed_v2_dufs_liu_l0p1"
MAX_ENTROPY_METHOD_ID = "max_entropy"
DIRECT_METHOD_IDS = (
    U28_METHOD_ID,
    IU_METHOD_ID,
    DEPLOYED_METHOD_ID,
    DUFS_METHOD_ID,
    MAX_ENTROPY_METHOD_ID,
)
DUFS_SCORE_KEY = "dufs_liu__lambda_0p1"

U28_BASE_STREAMS = (
    "raw::entropy",
    "raw::neg_logsumexp",
    "raw::neg_top1",
    "raw::topk_entropy",
    "raw::topk_varentropy",
    "raw::topk_renyi2",
    "raw::topk_tail_mass",
)
U28_TRANSFORMS = ("level", "ewma16", "positive_area", "persistence")
U28_FEATURES = tuple(
    f"{base}::{transform}" for base in U28_BASE_STREAMS for transform in U28_TRANSFORMS
)


class TwentyFourError(ValueError):
    """A 24-cell provenance, identity, or row-alignment gate failed."""


@dataclass(frozen=True)
class TraceIdentity:
    """One admitted raw trace before any target is opened."""

    cell_id: str
    dataset_revision: str
    dataset_family: str
    item_group_id: str
    candidate_ordinal: int
    row_id: str
    group_id: str
    core_features: np.ndarray = field(repr=False, compare=False)
    telemetry: Mapping[str, Any] = field(repr=False, compare=False)
    source_candidate: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def source_question_id(self) -> str:
        """Trace-level source ID, including the frozen raw candidate ordinal."""

        return _source_question_id(self.item_group_id, self.candidate_ordinal)


@dataclass(frozen=True)
class IdentityAlignment:
    """Label-free, complete permutation from canonical raw order to bundle order."""

    cell_id: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    bundle_position_by_row: tuple[int, ...]
    ordered_id_sha256: str
    identity_feature_names: tuple[str, ...]
    identity_feature_sha256: str
    max_abs_feature_error: float
    signature_decimals: int
    identity_frozen: bool
    bundle_sha256: str = BUNDLE_SHA256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "24cell_identity_alignment_v1",
            "adapter_revision": ADAPTER_REVISION,
            "cell_id": self.cell_id,
            "n_rows": len(self.row_ids),
            "bundle_sha256": self.bundle_sha256,
            "ordered_id_sha256": self.ordered_id_sha256,
            "identity_feature_names": list(self.identity_feature_names),
            "identity_feature_sha256": self.identity_feature_sha256,
            "bundle_permutation_sha256": canonical_sha256(
                list(self.bundle_position_by_row)
            ),
            "max_abs_feature_error": self.max_abs_feature_error,
            "signature_decimals": self.signature_decimals,
            "labels_accessed": False,
            "identity_frozen": self.identity_frozen,
        }


@dataclass(frozen=True)
class LabelAlignment:
    """Error labels opened only after a successful :class:`IdentityAlignment`."""

    identity: IdentityAlignment
    correct_labels: tuple[int, ...]
    error_labels: tuple[int, ...]
    raw_label_sha256: str
    bundle_label_sha256: str
    label_alignment_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity.as_dict(),
            "schema": "24cell_label_alignment_v1",
            "raw_label_sha256": self.raw_label_sha256,
            "bundle_label_sha256": self.bundle_label_sha256,
            "positive_class": "error/risk",
            "labels_accessed": True,
            "label_alignment_ok": self.label_alignment_ok,
        }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TwentyFourError(f"expected JSON object: {path}")
    return value


def _source_spec(cell_id: str) -> FrozenSourceSpec:
    matches = [spec for spec in FROZEN_A0_SOURCE_SPECS if spec.environment_id == cell_id]
    if not matches:
        if cell_id == BLOCKED_CELL:
            raise TwentyFourError(
                f"blocked-assets: {BLOCKED_CELL} has no frozen A0 source identity"
            )
        raise TwentyFourError(f"unregistered 24-cell raw source: {cell_id}")
    if len(matches) != 1:
        raise TwentyFourError(f"duplicate frozen source specifications: {cell_id}")
    return matches[0]


def eligible_cell_ids() -> tuple[str, ...]:
    return tuple(sorted(spec.environment_id for spec in FROZEN_A0_SOURCE_SPECS))


def _source_paths(
    repo_root: str | Path,
    spec: FrozenSourceSpec,
    *,
    source_root: str | Path | None,
) -> tuple[Path, Path]:
    repo = Path(repo_root).resolve()
    canonical_parent = (repo / "dataset_cache" / "repgrid" / spec.environment_id).resolve()
    if source_root is None:
        raw = (repo / spec.raw_relative_path).resolve()
        expected_parent = canonical_parent
    else:
        expected_parent = (Path(source_root).resolve() / spec.environment_id).resolve()
        raw = expected_parent / Path(spec.raw_relative_path).name
    if raw.parent != expected_parent:
        raise TwentyFourError(f"raw source escaped registered cell directory: {raw}")
    # A materialized raw LFS object may be staged outside the repository while its
    # immutable manifest remains in-tree.  Prefer the in-tree manifest when it is the
    # registered byte stream; this also rejects stale Drive-side metadata without
    # weakening the manifest hash gate on the raw payload.
    candidates = (canonical_parent / "manifest.json", expected_parent / "manifest.json")
    manifest = next(
        (
            path for path in candidates
            if path.is_file() and sha256_file(path) == spec.manifest_sha256
        ),
        candidates[-1],
    )
    return raw, manifest


def verify_source_artifact(
    repo_root: str | Path,
    cell_id: str,
    *,
    source_root: str | Path | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Verify a frozen A0 raw source before unpickling it."""

    spec = _source_spec(cell_id)
    raw, manifest_path = _source_paths(repo_root, spec, source_root=source_root)
    if not raw.is_file() or not manifest_path.is_file():
        raise TwentyFourError(f"missing materialized raw source or manifest: {cell_id}")
    if raw.stat().st_size != spec.source_size:
        detail = " (Git-LFS pointer is not a materialized source)" if raw.stat().st_size < 1024 else ""
        raise TwentyFourError(
            f"source size mismatch for {cell_id}: {raw.stat().st_size} != {spec.source_size}{detail}"
        )
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != spec.manifest_sha256:
        raise TwentyFourError(f"source manifest SHA-256 mismatch: {cell_id}")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset") != spec.dataset or manifest.get("split") != spec.split:
        raise TwentyFourError(f"source dataset/split mismatch: {cell_id}")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 1 or cells[0].get("pkl") != raw.name:
        raise TwentyFourError(f"source manifest does not bind one registered pkl: {cell_id}")
    raw_hash = sha256_file(raw) if verify_sha256 else None
    if verify_sha256 and raw_hash != spec.source_sha256:
        raise TwentyFourError(f"raw source SHA-256 mismatch: {cell_id}")
    return {
        "cell_id": cell_id,
        "raw_path": str(raw),
        "source_size": raw.stat().st_size,
        "source_sha256": raw_hash or spec.source_sha256,
        "source_sha256_verified": bool(verify_sha256),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "expected_admitted_count": spec.expected_admitted_count,
        "admission_mode": spec.admission_mode,
    }


def _source_question_id(item_group_id: str, ordinal: int) -> str:
    """Return the non-positional trace identity within one 24-cell source.

    ``ordinal`` is the candidate's index in the hash-frozen raw source array.  It
    remains part of the source-question identity even when another candidate is
    rejected by A0 admission, so admitted-row order can never substitute for it.
    """

    return f"{item_group_id}::candidate{int(ordinal)}"


def _canonical_ids(cell_id: str, item_group_id: str, ordinal: int) -> tuple[str, str]:
    """Build canonical row/group IDs without consulting bundle row positions."""

    group_id = f"{cell_id}::{item_group_id}"
    row_id = f"{cell_id}::{_source_question_id(item_group_id, ordinal)}"
    return row_id, group_id


def admit_source_rows(
    source: Mapping[Any, Any],
    spec: FrozenSourceSpec,
) -> tuple[TraceIdentity, ...]:
    """Apply the exact frozen A0 admission rule in canonical source order.

    Candidate labels are not indexed.  ``source_candidate`` is retained as an opaque
    mapping for the later, explicitly gated label-alignment step.
    """

    if not isinstance(source, Mapping):
        raise TwentyFourError("raw source must be a mapping")
    identities: list[TraceIdentity] = []
    assigned_questions = 0
    for problem_key in sorted(source, key=lambda value: str(value)):
        raw_row = source[problem_key]
        if not isinstance(raw_row, Mapping):
            raise TwentyFourError(f"raw source row is not a mapping: {problem_key!r}")
        try:
            normalize_question(raw_row["question"])
        except (KeyError, TypeError, ValueError):
            continue
        assigned_questions += 1
        candidates = raw_row.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise TwentyFourError(f"source candidates are not a sequence: {problem_key!r}")
        for ordinal, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise TwentyFourError(f"candidate is not a mapping: {problem_key!r}/{ordinal}")
            telemetry = (
                _cropped_telemetry_only(candidate)
                if spec.admission_mode == "cropped_all_rows"
                else {name: candidate[name] for name in TELEMETRY_KEYS}
            )
            _validated_telemetry(telemetry)
            if not _a0_admitted(telemetry, spec.admission_mode):
                continue
            features, _ = _feature_row(
                telemetry,
                allow_short=spec.admission_mode == "cropped_all_rows",
            )
            item_group_id = str(problem_key)
            row_id, group_id = _canonical_ids(
                spec.environment_id, item_group_id, ordinal
            )
            identities.append(
                TraceIdentity(
                    cell_id=spec.environment_id,
                    dataset_revision=canonical_revision(spec.dataset, spec.split),
                    dataset_family=spec.dataset_family,
                    item_group_id=item_group_id,
                    candidate_ordinal=int(ordinal),
                    row_id=row_id,
                    group_id=group_id,
                    core_features=np.asarray(features, dtype=float),
                    telemetry=telemetry,
                    source_candidate=candidate,
                )
            )
    if assigned_questions / max(len(source), 1) < 0.999:
        raise TwentyFourError(f"inadequate source-question identity coverage: {spec.environment_id}")
    if len(identities) != spec.expected_admitted_count:
        raise TwentyFourError(
            f"admitted population mismatch for {spec.environment_id}: "
            f"{len(identities)} != {spec.expected_admitted_count}"
        )
    row_ids = [row.row_id for row in identities]
    if len(row_ids) != len(set(row_ids)):
        raise TwentyFourError(f"duplicate canonical trace IDs: {spec.environment_id}")
    return tuple(identities)


def load_admitted_cell(
    repo_root: str | Path,
    cell_id: str,
    *,
    source_root: str | Path | None = None,
    verify_sha256: bool = True,
) -> tuple[tuple[TraceIdentity, ...], dict[str, Any]]:
    """Verify, unpickle, and admit one cell at a time to bound memory use."""

    audit = verify_source_artifact(
        repo_root, cell_id, source_root=source_root, verify_sha256=verify_sha256
    )
    with Path(audit["raw_path"]).open("rb") as handle:
        source = pickle.load(handle)
    identities = admit_source_rows(source, _source_spec(cell_id))
    audit.update(
        {
            "admitted_count": len(identities),
            "ordered_id_sha256": ordered_id_sha256([row.row_id for row in identities]),
            "labels_accessed": False,
        }
    )
    return identities, audit


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise TwentyFourError("identity feature contains non-finite values")
    scale = float(values.std())
    return values - values.mean() if scale <= 1e-8 else (values - values.mean()) / scale


def _row_signature(values: np.ndarray, decimals: int) -> tuple[float, ...]:
    rounded = np.round(np.asarray(values, dtype=float), int(decimals))
    # Canonicalize signed zero so JSON/hash behavior is platform independent.
    rounded[rounded == 0.0] = 0.0
    return tuple(float(value) for value in rounded)


def freeze_identity_alignment(
    identities: Sequence[TraceIdentity],
    bundle: Mapping[str, Any],
    *,
    signature_decimals: int = IDENTITY_SIGNATURE_DECIMALS,
    atol: float = IDENTITY_ATOL,
) -> IdentityAlignment:
    """Freeze a unique raw-to-bundle permutation without indexing labels."""

    identities = tuple(identities)
    if not identities:
        raise TwentyFourError("cannot align an empty cell")
    cells = {row.cell_id for row in identities}
    if len(cells) != 1:
        raise TwentyFourError(f"identity rows span multiple cells: {sorted(cells)}")
    cell_id = next(iter(cells))

    # These are the only bundle arrays touched before the identity freeze.
    matrix = np.asarray(bundle[f"{cell_id}__V"], dtype=float)
    pool = tuple(str(value) for value in bundle[f"{cell_id}__pool"])
    signs = np.asarray(bundle[f"{cell_id}__hand_signs"], dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(identities), len(pool)):
        raise TwentyFourError(
            f"bundle shape does not match admitted population for {cell_id}: {matrix.shape}"
        )
    if signs.shape != (len(pool),) or not np.isin(signs, (-1.0, 1.0)).all():
        raise TwentyFourError(f"invalid frozen hand-sign vector: {cell_id}")

    common = tuple(name for name in CORE_FEATURES if name in pool)
    if len(common) < 8:
        raise TwentyFourError(
            f"too few independent identity coordinates for {cell_id}: {len(common)}"
        )
    core = np.vstack([row.core_features for row in identities])
    reconstructed = np.column_stack(
        [
            _zscore(
                core[:, CORE_FEATURES.index(name)] * signs[pool.index(name)]
            )
            for name in common
        ]
    )
    observed = matrix[:, [pool.index(name) for name in common]]
    if not np.isfinite(observed).all():
        raise TwentyFourError(f"bundle identity features are non-finite: {cell_id}")

    by_signature: dict[tuple[float, ...], list[int]] = {}
    for position, row in enumerate(observed):
        by_signature.setdefault(_row_signature(row, signature_decimals), []).append(position)
    permutation: list[int] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for identity, row in zip(identities, reconstructed):
        matches = by_signature.get(_row_signature(row, signature_decimals), [])
        if not matches:
            missing.append(identity.row_id)
        elif len(matches) != 1:
            ambiguous.append(identity.row_id)
        else:
            permutation.append(matches[0])
    if missing or ambiguous:
        raise TwentyFourError(
            f"label-free identity join failed for {cell_id}: "
            f"missing={len(missing)}, ambiguous={len(ambiguous)}"
        )
    if len(permutation) != len(set(permutation)) or set(permutation) != set(range(len(identities))):
        raise TwentyFourError(f"identity join is not a complete permutation: {cell_id}")

    aligned_observed = observed[np.asarray(permutation, dtype=int)]
    max_error = float(np.max(np.abs(reconstructed - aligned_observed)))
    if not np.allclose(reconstructed, aligned_observed, rtol=0.0, atol=float(atol)):
        raise TwentyFourError(
            f"identity feature mismatch after join for {cell_id}: max_abs={max_error:.3g}"
        )
    feature_hash = canonical_sha256(
        {
            "names": list(common),
            "values": np.round(reconstructed, signature_decimals).tolist(),
        }
    )
    return IdentityAlignment(
        cell_id=cell_id,
        row_ids=tuple(row.row_id for row in identities),
        group_ids=tuple(row.group_id for row in identities),
        bundle_position_by_row=tuple(int(value) for value in permutation),
        ordered_id_sha256=ordered_id_sha256([row.row_id for row in identities]),
        identity_feature_names=common,
        identity_feature_sha256=feature_hash,
        max_abs_feature_error=max_error,
        signature_decimals=int(signature_decimals),
        identity_frozen=True,
        bundle_sha256=BUNDLE_SHA256,
    )


def _binary_correct_label(candidate: Mapping[str, Any]) -> int:
    """Open only registered correctness aliases, requiring exact agreement."""

    observed: list[tuple[str, int]] = []
    for name in ("label", "correct", "is_correct"):
        if name not in candidate:
            continue
        value = candidate[name]
        if isinstance(value, (bool, np.bool_)):
            parsed = int(value)
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            parsed = int(value)
        else:
            raise TwentyFourError(f"candidate {name} is not a binary correctness label")
        observed.append((name, parsed))
    if not observed:
        raise TwentyFourError("candidate has no registered correctness label")
    if len({value for _, value in observed}) != 1:
        raise TwentyFourError(f"candidate correctness aliases disagree: {observed}")
    return observed[0][1]


def open_and_verify_labels(
    identities: Sequence[TraceIdentity],
    alignment: IdentityAlignment,
    bundle: Mapping[str, Any],
) -> LabelAlignment:
    """Open labels only after a complete identity freeze and verify row agreement."""

    identities = tuple(identities)
    if not alignment.identity_frozen:
        raise TwentyFourError("labels may not be opened before identity is frozen")
    if alignment.bundle_sha256 != BUNDLE_SHA256:
        raise TwentyFourError("identity alignment references another 24-cell bundle")
    if tuple(row.row_id for row in identities) != alignment.row_ids:
        raise TwentyFourError("identity rows changed after the label-free freeze")
    # This is intentionally the first access to the bundle's label array.
    stored = np.asarray(bundle[f"{alignment.cell_id}__labels"])
    if stored.shape != (len(identities),) or not np.isin(stored, (0, 1)).all():
        raise TwentyFourError(f"bundle correctness labels are invalid: {alignment.cell_id}")
    positions = np.asarray(alignment.bundle_position_by_row, dtype=int)
    bundle_correct = stored[positions].astype(np.int8)
    raw_correct = np.asarray(
        [_binary_correct_label(row.source_candidate) for row in identities], dtype=np.int8
    )
    if not np.array_equal(raw_correct, bundle_correct):
        disagreement = int(np.sum(raw_correct != bundle_correct))
        raise TwentyFourError(
            f"label disagreement after identity freeze for {alignment.cell_id}: {disagreement}"
        )
    return LabelAlignment(
        identity=alignment,
        correct_labels=tuple(int(value) for value in bundle_correct),
        error_labels=tuple(int(not value) for value in bundle_correct),
        raw_label_sha256=canonical_sha256(raw_correct.tolist()),
        bundle_label_sha256=canonical_sha256(bundle_correct.tolist()),
        label_alignment_ok=True,
    )


def unified28_from_dict(payload: Mapping[str, Any]) -> UnifiedCausalIU:
    """Rehydrate a frozen :class:`UnifiedCausalIU` without fitting anything."""

    reference = payload.get("reference")
    accumulator = payload.get("accumulator")
    if not isinstance(reference, Mapping) or not isinstance(accumulator, Mapping):
        raise TwentyFourError("Unified-28 model is missing reference/accumulator")
    model = UnifiedCausalIU(
        reference=BaseReference(
            names=tuple(str(value) for value in reference["names"]),
            centres=np.asarray(reference["centres"], dtype=float),
            scales=np.asarray(reference["scales"], dtype=float),
            availability=np.asarray(reference["availability"], dtype=float),
            positions_per_trace=int(reference["positions_per_trace"]),
            n_traces=int(reference["n_traces"]),
        ),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        feature_indices=np.asarray(payload["feature_indices"], dtype=int),
        feature_medians=np.asarray(payload["feature_medians"], dtype=float),
        feature_centres=np.asarray(payload["feature_centres"], dtype=float),
        feature_scales=np.asarray(payload["feature_scales"], dtype=float),
        feature_signs=np.asarray(payload["feature_signs"], dtype=float),
        weights=np.asarray(payload["weights"], dtype=float),
        evidence_centre=float(payload["evidence_centre"]),
        evidence_scale=float(payload["evidence_scale"]),
        accumulator=AccumulatorSpec(
            kind=str(accumulator["kind"]),
            span=accumulator.get("span"),
            drift=float(accumulator.get("drift", 0.0)),
        ),
        warning_threshold_5pct=float(payload.get("warning_threshold_5pct", math.inf)),
        warning_threshold_10pct=float(payload.get("warning_threshold_10pct", math.inf)),
        diagnostics=dict(payload.get("diagnostics", {})),
    )
    validate_unified28_model(model)
    return model


def validate_unified28_model(model: UnifiedCausalIU) -> None:
    """Enforce the frozen ordinary Unified-28 method-of-record."""

    if model.feature_names != U28_FEATURES:
        raise TwentyFourError("Unified-28 feature roster is not the frozen 7 x 4 roster")
    if len(model.feature_names) != 28 or not np.array_equal(
        model.feature_indices,
        np.asarray(
            [all_feature_names(model.reference.names).index(name) for name in U28_FEATURES],
            dtype=int,
        ),
    ):
        raise TwentyFourError("Unified-28 feature indices do not match the frozen roster")
    if model.accumulator != AccumulatorSpec("identity"):
        raise TwentyFourError("Unified-28 accumulator must be Identity")
    if not np.array_equal(model.feature_signs, np.ones(28, dtype=float)):
        raise TwentyFourError("Unified-28 frozen feature signs changed")
    if model.diagnostics.get("components") != 2 or model.diagnostics.get("loss") != "l2":
        raise TwentyFourError("Unified-28 must use ordinary two-component L2 IU-PCR")
    if bool(model.diagnostics.get("graph_or_laplacian")):
        raise TwentyFourError("Unified-28 ordinary control may not contain a graph")
    arrays = (
        model.reference.centres,
        model.reference.scales,
        model.feature_medians,
        model.feature_centres,
        model.feature_scales,
        model.weights,
    )
    if not all(np.isfinite(values).all() for values in arrays):
        raise TwentyFourError("Unified-28 model contains non-finite fitted parameters")
    if model.evidence_scale <= 0.0 or not math.isfinite(model.evidence_scale):
        raise TwentyFourError("Unified-28 evidence scale is invalid")


def unified28_parameter_sha256(model: UnifiedCausalIU) -> str:
    """Content hash the in-memory scorer so an anchor cannot bless another copy."""

    validate_unified28_model(model)
    return canonical_sha256(
        {
            "reference": {
                "names": list(model.reference.names),
                "centres": np.asarray(model.reference.centres, dtype=float).tolist(),
                "scales": np.asarray(model.reference.scales, dtype=float).tolist(),
                "availability": np.asarray(
                    model.reference.availability, dtype=float
                ).tolist(),
                "positions_per_trace": int(model.reference.positions_per_trace),
                "n_traces": int(model.reference.n_traces),
            },
            "feature_names": list(model.feature_names),
            "feature_indices": np.asarray(model.feature_indices, dtype=int).tolist(),
            "feature_medians": np.asarray(model.feature_medians, dtype=float).tolist(),
            "feature_centres": np.asarray(model.feature_centres, dtype=float).tolist(),
            "feature_scales": np.asarray(model.feature_scales, dtype=float).tolist(),
            "feature_signs": np.asarray(model.feature_signs, dtype=float).tolist(),
            "weights": np.asarray(model.weights, dtype=float).tolist(),
            "evidence_centre": float(model.evidence_centre),
            "evidence_scale": float(model.evidence_scale),
            "accumulator": {
                "kind": model.accumulator.kind,
                "span": model.accumulator.span,
                "drift": float(model.accumulator.drift),
            },
            # Infinity is the frozen marker for "no stopping policy".  Encode it
            # explicitly because canonical JSON intentionally rejects non-finite
            # JavaScript number tokens.
            "warning_threshold_5pct": (
                float(model.warning_threshold_5pct)
                if math.isfinite(model.warning_threshold_5pct)
                else "+infinity"
            ),
            "warning_threshold_10pct": (
                float(model.warning_threshold_10pct)
                if math.isfinite(model.warning_threshold_10pct)
                else "+infinity"
            ),
            "diagnostics": dict(model.diagnostics),
        }
    )


def load_unified28_model(
    repo_root: str | Path,
    *,
    path: str | Path | None = None,
) -> tuple[UnifiedCausalIU, str]:
    model_path = Path(path) if path is not None else Path(repo_root) / MODEL_RELATIVE_PATH
    artifact_hash = sha256_file(model_path)
    if artifact_hash != MODEL_ARTIFACT_SHA256:
        raise TwentyFourError("frozen Unified-28 VALIDATION.json SHA-256 mismatch")
    document = _read_json(model_path)
    candidate = document.get("base7_full28")
    if not isinstance(candidate, Mapping) or not isinstance(candidate.get("model"), Mapping):
        raise TwentyFourError("VALIDATION.json has no base7_full28.model")
    return unified28_from_dict(candidate["model"]), artifact_hash


def verify_processbench_anchor(
    repo_root: str | Path,
    model: UnifiedCausalIU,
    *,
    model_artifact_sha256: str = MODEL_ARTIFACT_SHA256,
    rows_per_subset: int = 1,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Reproduce registered Llama ProcessBench scores before 24-cell transfer."""

    if int(rows_per_subset) < 1:
        raise TwentyFourError("rows_per_subset must be positive")
    validate_unified28_model(model)
    repo = Path(repo_root)
    if model_artifact_sha256 != MODEL_ARTIFACT_SHA256:
        raise TwentyFourError("ProcessBench anchor received an unregistered model artifact")
    run_path = repo / MODEL_RUN_DEFINITION_RELATIVE_PATH
    records_path = repo / MODEL_RECORDS_RELATIVE_PATH
    if sha256_file(run_path) != MODEL_RUN_DEFINITION_SHA256:
        raise TwentyFourError("Unified-28 run-definition SHA-256 mismatch")
    if sha256_file(records_path) != MODEL_RECORDS_SHA256:
        raise TwentyFourError("Unified-28 validation-record SHA-256 mismatch")
    run = _read_json(run_path)
    inventory = run.get("validation_inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise TwentyFourError("Unified validation inventory is not the registered four subsets")
    source_by_family: dict[str, tuple[Path, str]] = {}
    for item in inventory:
        family = str(item["family"])
        path = repo / "dataset_cache" / "repgrid" / "pb_llama31_8b" / f"processbench_{family}.pkl"
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise TwentyFourError(f"ProcessBench anchor source hash mismatch: {family}")
        source_by_family[family] = (path, str(item["sha256"]))

    wanted: dict[str, list[dict[str, Any]]] = {family: [] for family in source_by_family}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            family = record.get("family")
            if (
                record.get("candidate") == "base7_full28"
                and family in wanted
                and len(wanted[family]) < int(rows_per_subset)
            ):
                wanted[family].append(record)
            if all(len(rows) == int(rows_per_subset) for rows in wanted.values()):
                break
    if not all(len(rows) == int(rows_per_subset) for rows in wanted.values()):
        raise TwentyFourError("Unified validation records do not contain all anchor rows")

    comparisons: list[dict[str, Any]] = []
    for family in sorted(wanted):
        with source_by_family[family][0].open("rb") as handle:
            source = pickle.load(handle)
        rows = source.values() if isinstance(source, Mapping) else source
        by_id = {str(row.get("id")): row for row in rows if isinstance(row, Mapping)}
        for frozen in wanted[family]:
            unit = str(frozen["unit"])
            if unit not in by_id:
                raise TwentyFourError(f"ProcessBench anchor ID missing: {family}/{unit}")
            observed = float(model.score_row(by_id[unit]).global_score)
            expected = float(frozen["global_score"])
            comparisons.append(
                {
                    "family": family,
                    "unit": unit,
                    "expected": expected,
                    "observed": observed,
                    "abs_error": abs(observed - expected),
                }
            )
    maximum = max(row["abs_error"] for row in comparisons)
    if maximum > float(atol):
        raise TwentyFourError(
            f"Unified-28 ProcessBench anchor failed: max_abs_error={maximum:.3g}"
        )
    return {
        "schema": "unified28_processbench_anchor_v1",
        "model_artifact_sha256": model_artifact_sha256,
        "model_parameter_sha256": unified28_parameter_sha256(model),
        "run_definition_sha256": MODEL_RUN_DEFINITION_SHA256,
        "validation_records_sha256": MODEL_RECORDS_SHA256,
        "rows": comparisons,
        "n_rows": len(comparisons),
        "max_abs_error": maximum,
        "anchor_sha256": canonical_sha256(comparisons),
        "ok": True,
    }


def incumbent_risk_scores(
    score_checkpoint: Mapping[str, Any],
    alignment: IdentityAlignment,
) -> dict[str, np.ndarray]:
    """Return historical incumbent scores in canonical row order and risk polarity."""

    required = ("sample_index", "iu_pcr", "deployed_upcr", DUFS_SCORE_KEY)
    missing = [name for name in required if name not in score_checkpoint]
    if missing:
        raise TwentyFourError(f"incumbent checkpoint missing arrays: {missing}")
    n = len(alignment.row_ids)
    sample_index = np.asarray(score_checkpoint["sample_index"], dtype=int)
    if not np.array_equal(sample_index, np.arange(n, dtype=int)):
        raise TwentyFourError(f"incumbent sample_index is not canonical bundle order: {alignment.cell_id}")
    positions = np.asarray(alignment.bundle_position_by_row, dtype=int)
    output = {}
    for method, key in (
        (IU_METHOD_ID, "iu_pcr"),
        (DEPLOYED_METHOD_ID, "deployed_upcr"),
        (DUFS_METHOD_ID, DUFS_SCORE_KEY),
    ):
        score = np.asarray(score_checkpoint[key], dtype=float)
        if score.shape != (n,) or not np.isfinite(score).all():
            raise TwentyFourError(f"invalid incumbent score array {key}: {alignment.cell_id}")
        # Historical checkpoints are confidence-oriented (labels used correctness=1).
        output[method] = -score[positions]
    return output


def _verified_score_path(repo_root: str | Path, cell_id: str) -> tuple[Path, str]:
    """Resolve one immutable incumbent checkpoint through its freeze manifest."""

    benchmark = Path(repo_root) / BENCHMARK_RELATIVE_ROOT
    freeze_path = benchmark / "SCORE_FREEZE_MANIFEST.json"
    if sha256_file(freeze_path) != SCORE_FREEZE_MANIFEST_SHA256:
        raise TwentyFourError("24-cell score-freeze manifest SHA-256 mismatch")
    freeze = _read_json(freeze_path)
    if freeze.get("bundle_sha256") != BUNDLE_SHA256:
        raise TwentyFourError("score-freeze manifest references another bundle")
    manifest = freeze.get("score_manifest")
    if not isinstance(manifest, list):
        raise TwentyFourError("score-freeze manifest has no score roster")
    matches = [row for row in manifest if str(row.get("cell")) == cell_id]
    if len(matches) != 1:
        raise TwentyFourError(f"score-freeze manifest does not uniquely bind {cell_id}")
    expected_relative = f"scores/{cell_id}.npz"
    if matches[0].get("score_file") != expected_relative:
        raise TwentyFourError(f"score checkpoint path changed: {cell_id}")
    score_path = benchmark / expected_relative
    score_hash = sha256_file(score_path)
    if score_hash != matches[0].get("score_sha256"):
        raise TwentyFourError(f"score checkpoint SHA-256 mismatch: {cell_id}")
    return score_path, score_hash


def _require_processbench_anchor(
    anchor: Mapping[str, Any],
    model: UnifiedCausalIU,
    model_artifact_sha256: str,
) -> None:
    """Fail closed unless this exact in-memory model passed the frozen anchor."""

    expected = {
        "schema": "unified28_processbench_anchor_v1",
        "ok": True,
        "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
        "model_parameter_sha256": unified28_parameter_sha256(model),
        "run_definition_sha256": MODEL_RUN_DEFINITION_SHA256,
        "validation_records_sha256": MODEL_RECORDS_SHA256,
    }
    if model_artifact_sha256 != MODEL_ARTIFACT_SHA256:
        raise TwentyFourError("24-cell replay received an unregistered Unified-28 artifact")
    mismatches = [key for key, value in expected.items() if anchor.get(key) != value]
    if mismatches:
        raise TwentyFourError(
            f"Unified-28 ProcessBench anchor is absent or for another model: {mismatches}"
        )
    rows = anchor.get("rows")
    if not isinstance(rows, list) or int(anchor.get("n_rows", 0)) < 4:
        raise TwentyFourError("Unified-28 ProcessBench anchor is incomplete")
    if anchor.get("anchor_sha256") != canonical_sha256(rows):
        raise TwentyFourError("Unified-28 ProcessBench anchor row hash mismatch")
    if {str(row.get("family")) for row in rows} != {
        "gsm8k",
        "math",
        "olympiadbench",
        "omnimath",
    }:
        raise TwentyFourError("Unified-28 ProcessBench anchor lacks a frozen subset")
    if any(
        abs(float(row.get("observed")) - float(row.get("expected")))
        != float(row.get("abs_error"))
        for row in rows
    ):
        raise TwentyFourError("Unified-28 ProcessBench anchor arithmetic mismatch")
    if float(anchor.get("max_abs_error", math.inf)) > 1e-12:
        raise TwentyFourError("Unified-28 ProcessBench anchor exceeds exact tolerance")


def unified28_replay_source_artifact_sha256(
    model_artifact_sha256: str,
    raw_source_sha256: str,
) -> str:
    """Bind a transferred Unified-28 score to both model and raw source bytes.

    ``comparison_record_v1.source_artifact_hash`` has one field, while a replayed
    score has two indispensable inputs.  This canonical composite prevents a row
    scored by the right model on the wrong raw source (or vice versa) from sharing
    provenance with the registered replay.
    """

    return canonical_sha256(
        {
            "schema": "unified28_24cell_replay_source_v1",
            "model_artifact_sha256": str(model_artifact_sha256),
            "raw_source_sha256": str(raw_source_sha256),
        }
    )


def materialize_cell_records(
    repo_root: str | Path,
    identities: Sequence[TraceIdentity],
    labels: LabelAlignment,
    model: UnifiedCausalIU,
    *,
    model_artifact_sha256: str,
    source_artifact_sha256: str,
    processbench_anchor: Mapping[str, Any],
    folds: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Score all frozen direct methods on exactly one identity-proven population."""

    identities = tuple(identities)
    if tuple(row.row_id for row in identities) != labels.identity.row_ids:
        raise TwentyFourError("materialization population differs from label alignment")
    if not labels.label_alignment_ok:
        raise TwentyFourError("materialization requires a passing label alignment")
    validate_unified28_model(model)
    _require_processbench_anchor(processbench_anchor, model, model_artifact_sha256)
    cell_id = labels.identity.cell_id
    if source_artifact_sha256 != _source_spec(cell_id).source_sha256:
        raise TwentyFourError(f"24-cell replay source SHA-256 is unregistered: {cell_id}")
    score_path, score_hash = _verified_score_path(repo_root, cell_id)
    with np.load(score_path, allow_pickle=False) as checkpoint:
        scores = incumbent_risk_scores(checkpoint, labels.identity)
    scores[U28_METHOD_ID] = np.asarray(
        [model.score_row(row.telemetry).global_score for row in identities], dtype=float
    )
    scores[MAX_ENTROPY_METHOD_ID] = np.asarray(
        [float(np.max(np.asarray(row.telemetry["token_entropies"], dtype=float))) for row in identities],
        dtype=float,
    )
    if any(values.shape != (len(identities),) or not np.isfinite(values).all() for values in scores.values()):
        raise TwentyFourError(f"non-finite or incomplete direct scores: {cell_id}")
    if set(scores) != set(DIRECT_METHOD_IDS):
        raise TwentyFourError(f"direct method roster changed: {sorted(scores)}")

    source_hashes = {
        U28_METHOD_ID: unified28_replay_source_artifact_sha256(
            model_artifact_sha256,
            source_artifact_sha256,
        ),
        MAX_ENTROPY_METHOD_ID: source_artifact_sha256,
        IU_METHOD_ID: score_hash,
        DEPLOYED_METHOD_ID: score_hash,
        DUFS_METHOD_ID: score_hash,
    }
    records: list[dict[str, Any]] = []
    for method_id in DIRECT_METHOD_IDS:
        for index, identity in enumerate(identities):
            fold = None
            if folds is not None:
                if identity.group_id not in folds and identity.row_id not in folds:
                    raise TwentyFourError(f"fold assignment missing: {identity.row_id}")
                fold = int(folds.get(identity.group_id, folds.get(identity.row_id)))
            records.append(
                make_comparison_record(
                    lane="global",
                    population_id=POPULATION_ID,
                    row_id=identity.row_id,
                    group_id=identity.group_id,
                    cell_id=identity.cell_id,
                    method_id=method_id,
                    continuous_score=float(scores[method_id][index]),
                    discrete_prediction=None,
                    label=int(labels.error_labels[index]),
                    budget="final",
                    fold=fold,
                    calibration_hash=None,
                    source_artifact_hash=source_hashes[method_id],
                    extra={
                        "family": identity.dataset_family,
                        "source_question_id": identity.source_question_id,
                        "stratify_label": int(labels.error_labels[index]),
                        "positive_class": "error/risk",
                        "bundle_position": int(labels.identity.bundle_position_by_row[index]),
                    },
                )
            )
    for method_id in DIRECT_METHOD_IDS:
        method_ids = [row["row_id"] for row in records if row["method_id"] == method_id]
        if tuple(method_ids) != labels.identity.row_ids:
            raise TwentyFourError(f"method does not cover identical ordered IDs: {method_id}")
    return tuple(records)


def _crossfit_operating_point(
    rows: Sequence[Mapping[str, Any]], target_fpr: float
) -> dict[str, Any]:
    folds = {row.get("fold") for row in rows}
    if folds != set(range(5)):
        raise TwentyFourError(f"fixed-FPR cross-fitting requires folds 0..4, got {folds}")
    totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    ledgers = []
    for held in range(5):
        train = [row for row in rows if row["fold"] != held]
        test = [row for row in rows if row["fold"] == held]
        fit = calibrate_correct_only_threshold(
            [row["label"] for row in train],
            [row["continuous_score"] for row in train],
            target_fpr=target_fpr,
        )
        observed = operating_point(
            [row["label"] for row in test],
            [row["continuous_score"] for row in test],
            fit["threshold"],
        )
        for name in totals:
            totals[name] += int(observed[name])
        ledger = {**fit, "held_out_fold": held, "n_held_out": len(test)}
        ledger["calibration_hash"] = canonical_sha256(ledger)
        ledgers.append(ledger)
    tp, fp, fn, tn = (totals[name] for name in ("tp", "fp", "fn", "tn"))
    return {
        "target_fpr": float(target_fpr),
        **totals,
        "error_tpr": tp / (tp + fn) if tp + fn else float("nan"),
        "error_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "observed_fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "calibration_ledgers": ledgers,
        "calibration_ledger_sha256": canonical_sha256(ledgers),
        "aggregation": "concatenated_discrete_out_of_fold_decisions",
    }


def per_cell_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute error-positive Global metrics using the frozen evaluator."""

    rows = list(records)
    if not rows:
        raise TwentyFourError("cannot evaluate zero comparison records")
    cells = {str(row["cell_id"]) for row in rows}
    if len(cells) != 1:
        raise TwentyFourError("per-cell metrics may not pool heterogeneous cells")
    expected_ids: tuple[str, ...] | None = None
    output: dict[str, dict[str, Any]] = {}
    for method_id in DIRECT_METHOD_IDS:
        selected = [row for row in rows if row["method_id"] == method_id]
        ids = tuple(str(row["row_id"]) for row in selected)
        if expected_ids is None:
            expected_ids = ids
        if ids != expected_ids or len(ids) != len(set(ids)):
            raise TwentyFourError(f"method row IDs differ in direct table: {method_id}")
        metrics = detection_metrics(
            [row["label"] for row in selected],
            [row["continuous_score"] for row in selected],
        )
        metrics["operating_fpr_05"] = _crossfit_operating_point(selected, 0.05)
        metrics["operating_fpr_10"] = _crossfit_operating_point(selected, 0.10)
        output[method_id] = metrics
    return output


def static_preflight(
    repo_root: str | Path,
    *,
    source_root: str | Path | None = None,
    verify_score_hashes: bool = True,
    verify_raw_hashes: bool = False,
) -> dict[str, Any]:
    """Check immutable manifests without unpickling, aligning, or scoring sources.

    A source counted as ``source_file_size_ready`` has passed path, manifest, and
    byte-size checks (and optionally its raw SHA-256).  Static preflight never opens
    the pickle or bundle arrays, so it must report zero identity-aligned and scored
    rows even when every source file is ready.
    """

    repo = Path(repo_root)
    bundle = repo / BUNDLE_RELATIVE_PATH
    benchmark = repo / BENCHMARK_RELATIVE_ROOT
    freeze_path = benchmark / "SCORE_FREEZE_MANIFEST.json"
    run_path = benchmark / "RUN_DEFINITION.json"
    fit_path = benchmark / "FIT_COMPLETE.json"
    if sha256_file(freeze_path) != SCORE_FREEZE_MANIFEST_SHA256:
        raise TwentyFourError("24-cell score-freeze manifest SHA-256 mismatch")
    if sha256_file(run_path) != SCORE_RUN_DEFINITION_SHA256:
        raise TwentyFourError("24-cell run-definition SHA-256 mismatch")
    if sha256_file(fit_path) != SCORE_FIT_COMPLETE_SHA256:
        raise TwentyFourError("24-cell fit-complete SHA-256 mismatch")
    model_path = repo / MODEL_RELATIVE_PATH
    model_run_path = repo / MODEL_RUN_DEFINITION_RELATIVE_PATH
    model_records_path = repo / MODEL_RECORDS_RELATIVE_PATH
    if sha256_file(model_path) != MODEL_ARTIFACT_SHA256:
        raise TwentyFourError("frozen Unified-28 model artifact SHA-256 mismatch")
    if sha256_file(model_run_path) != MODEL_RUN_DEFINITION_SHA256:
        raise TwentyFourError("frozen Unified-28 run-definition SHA-256 mismatch")
    if sha256_file(model_records_path) != MODEL_RECORDS_SHA256:
        raise TwentyFourError("frozen Unified-28 validation-record SHA-256 mismatch")
    freeze = _read_json(freeze_path)
    run = _read_json(run_path)
    if sha256_file(bundle) != BUNDLE_SHA256 or freeze.get("bundle_sha256") != BUNDLE_SHA256:
        raise TwentyFourError("24-cell bundle SHA-256 mismatch")
    if sha256_file(run_path) != freeze.get("run_definition_sha256"):
        raise TwentyFourError("24-cell run-definition SHA-256 mismatch")
    if sha256_file(fit_path) != freeze.get("fit_complete_sha256"):
        raise TwentyFourError("24-cell fit-complete SHA-256 mismatch")
    if run.get("run_fingerprint") != freeze.get("run_fingerprint"):
        raise TwentyFourError("24-cell run fingerprint mismatch")
    if freeze.get("score_files_verified_before_labels") is not True:
        raise TwentyFourError("24-cell score freeze did not preserve the label firewall")
    frozen_lambda = run.get("frozen_lambda")
    if not isinstance(frozen_lambda, Mapping) or frozen_lambda.get("dufs_liu") != 0.1:
        raise TwentyFourError("registered DUFS-LIU lambda is not exactly 0.1")
    manifest = freeze.get("score_manifest")
    if not isinstance(manifest, list) or len(manifest) != EXPECTED_CELLS:
        raise TwentyFourError("24-cell score manifest is incomplete")
    observed_cells = [str(row.get("cell")) for row in manifest]
    if len(observed_cells) != len(set(observed_cells)):
        raise TwentyFourError("24-cell score manifest contains duplicates")
    if set(observed_cells) != set(eligible_cell_ids()) | {BLOCKED_CELL}:
        raise TwentyFourError("24-cell score manifest roster differs from frozen sources")
    if verify_score_hashes:
        for row in manifest:
            score_path = benchmark / str(row["score_file"])
            if sha256_file(score_path) != row.get("score_sha256"):
                raise TwentyFourError(f"score checkpoint SHA-256 mismatch: {row['cell']}")

    sources = []
    source_blockers = []
    for cell_id in eligible_cell_ids():
        try:
            sources.append(
                verify_source_artifact(
                    repo,
                    cell_id,
                    source_root=source_root,
                    verify_sha256=verify_raw_hashes,
                )
            )
        except (OSError, TwentyFourError) as exc:
            spec = _source_spec(cell_id)
            source_blockers.append(
                {
                    "cell_id": cell_id,
                    "rows": spec.expected_admitted_count,
                    "fidelity": "blocked-assets",
                    "reason": str(exc),
                }
            )
    ready_rows = sum(int(item["expected_admitted_count"]) for item in sources)
    blocked = [
        {
            "cell_id": BLOCKED_CELL,
            "rows": EXPECTED_ROWS - ELIGIBLE_ROWS,
            "fidelity": "blocked-assets",
            "reason": "no hash-frozen A0 source specification; exact 256-row identity is unproven",
        },
        *source_blockers,
    ]
    return {
        "schema": "24cell_static_preflight_v1",
        "adapter_revision": ADAPTER_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "model_artifact_sha256": MODEL_ARTIFACT_SHA256,
        "model_run_definition_sha256": MODEL_RUN_DEFINITION_SHA256,
        "model_records_sha256": MODEL_RECORDS_SHA256,
        "score_freeze_manifest_sha256": SCORE_FREEZE_MANIFEST_SHA256,
        "score_hashes_verified": bool(verify_score_hashes),
        "raw_hashes_verified": bool(verify_raw_hashes),
        "headline_eligible_cells": ELIGIBLE_CELLS,
        "headline_eligible_rows": ELIGIBLE_ROWS,
        "bundle_cells": EXPECTED_CELLS,
        "bundle_rows": EXPECTED_ROWS,
        "source_file_size_ready_cells": len(sources),
        "source_file_size_ready_rows": ready_rows,
        "identity_aligned_cells": 0,
        "identity_aligned_rows": 0,
        "scored_cells": 0,
        "scored_rows": 0,
        "row_identity_contract": {
            "source_question_id": "<raw_problem_key>::candidate<candidate_ordinal>",
            "row_id": "<cell_id>::<source_question_id>",
            "group_id": "<cell_id>::<raw_problem_key>",
            "candidate_ordinal_source": "hash-frozen raw candidate array before A0 admission",
            "positional_fallback_allowed": False,
        },
        "blocked": blocked,
        "sources": sources,
        "ok": not source_blockers,
    }


def real_identity_preflight(
    repo_root: str | Path,
    *,
    source_root: str | Path,
    cells: Sequence[str],
    verify_raw_hashes: bool = True,
) -> dict[str, Any]:
    """Run the expensive per-cell identity gate, but never score Unified-28."""

    repo = Path(repo_root)
    requested = [str(cell_id) for cell_id in cells]
    if not requested or len(requested) != len(set(requested)):
        raise TwentyFourError("real identity preflight requires unique eligible cells")
    for cell_id in requested:
        _source_spec(cell_id)
    bundle_path = repo / BUNDLE_RELATIVE_PATH
    if sha256_file(bundle_path) != BUNDLE_SHA256:
        raise TwentyFourError("24-cell bundle SHA-256 mismatch")
    audits = []
    with np.load(bundle_path, allow_pickle=True) as bundle:
        for cell_id in requested:
            identities, source_audit = load_admitted_cell(
                repo,
                cell_id,
                source_root=source_root,
                verify_sha256=verify_raw_hashes,
            )
            identity = freeze_identity_alignment(identities, bundle)
            labels = open_and_verify_labels(identities, identity, bundle)
            audits.append({"source": source_audit, "alignment": labels.as_dict()})
    return {
        "schema": "24cell_real_identity_preflight_v1",
        "bundle_sha256": BUNDLE_SHA256,
        "cells": requested,
        "rows": sum(item["alignment"]["n_rows"] for item in audits),
        "audits": audits,
        "all_ok": True,
    }


def partial_identity_audit(
    repo_root: str | Path,
    *,
    source_root: str | Path,
    cells: Sequence[str],
) -> dict[str, Any]:
    """Audit any size-ready subset through identity and label agreement only.

    Cells are processed in sorted order so the same supplied set has one canonical
    audit.  Every raw payload must pass its registered SHA-256 before unpickling.
    A failure in one cell is recorded with its exact stage, exception type, and
    message; subsequent cells are still attempted.  This function deliberately has
    no model or score arguments and cannot materialize comparison records.
    """

    repo = Path(repo_root)
    requested = [str(cell_id) for cell_id in cells]
    if not requested or any(not cell_id for cell_id in requested):
        raise TwentyFourError("partial identity audit requires non-empty cell IDs")
    if len(requested) != len(set(requested)):
        raise TwentyFourError("partial identity audit requires unique cells")
    requested = sorted(requested)

    audits: list[dict[str, Any]] = []

    def failure(
        cell_id: str,
        stage: str,
        exc: BaseException,
        *,
        expected_rows: int | None = None,
    ) -> dict[str, Any]:
        return {
            "cell_id": cell_id,
            "status": "failed",
            "identity_proven": False,
            "expected_rows": expected_rows,
            "failure_stage": stage,
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }

    def finish() -> dict[str, Any]:
        passed = [row for row in audits if row["status"] == "identity-proven"]
        payload = {
            "schema": "24cell_partial_identity_audit_v1",
            "adapter_revision": ADAPTER_REVISION,
            "bundle_sha256": BUNDLE_SHA256,
            "requested_cells": requested,
            "requested_cells_sha256": canonical_sha256(requested),
            "raw_sha256_required": True,
            "identity_proven_cells": len(passed),
            "identity_proven_rows": sum(int(row["rows"]) for row in passed),
            "failed_cells": len(audits) - len(passed),
            "scoring_performed": False,
            "audits": audits,
            "all_ok": len(passed) == len(requested),
        }
        return {**payload, "audit_sha256": canonical_sha256(payload)}

    bundle_path = repo / BUNDLE_RELATIVE_PATH
    try:
        observed_bundle_hash = sha256_file(bundle_path)
        if observed_bundle_hash != BUNDLE_SHA256:
            raise TwentyFourError("24-cell bundle SHA-256 mismatch")
        bundle_context = np.load(bundle_path, allow_pickle=True)
    except Exception as exc:  # Per-cell records are required even for a global gate.
        audits.extend(
            failure(cell_id, "bundle_verification", exc) for cell_id in requested
        )
        return finish()

    with bundle_context as bundle:
        for cell_id in requested:
            expected_rows: int | None = None
            stage = "source_registration"
            try:
                spec = _source_spec(cell_id)
                expected_rows = int(spec.expected_admitted_count)
                stage = "raw_sha256_and_admission"
                identities, source_audit = load_admitted_cell(
                    repo,
                    cell_id,
                    source_root=source_root,
                    verify_sha256=True,
                )
                if source_audit.get("source_sha256_verified") is not True:
                    raise TwentyFourError(
                        f"raw source SHA-256 was not verified: {cell_id}"
                    )
                stage = "label_free_identity_alignment"
                identity = freeze_identity_alignment(identities, bundle)
                stage = "label_agreement"
                labels = open_and_verify_labels(identities, identity, bundle)
                if not identity.identity_frozen or not labels.label_alignment_ok:
                    raise TwentyFourError(
                        f"identity or label agreement was not proven: {cell_id}"
                    )
                audits.append(
                    {
                        "cell_id": cell_id,
                        "status": "identity-proven",
                        "identity_proven": True,
                        "rows": len(identity.row_ids),
                        "expected_rows": expected_rows,
                        "raw_sha256_verified": True,
                        "labels_accessed_after_identity_freeze": True,
                        "source": source_audit,
                        "alignment": labels.as_dict(),
                    }
                )
            except Exception as exc:  # Continue to produce a complete partial audit.
                audits.append(
                    failure(
                        cell_id,
                        stage,
                        exc,
                        expected_rows=expected_rows,
                    )
                )
    return finish()


__all__ = [
    "ADAPTER_REVISION",
    "BLOCKED_CELL",
    "BUNDLE_SHA256",
    "DATASET_REVISION",
    "DIRECT_METHOD_IDS",
    "DUFS_METHOD_ID",
    "ELIGIBLE_CELLS",
    "ELIGIBLE_ROWS",
    "EXPECTED_CELLS",
    "EXPECTED_ROWS",
    "IdentityAlignment",
    "LabelAlignment",
    "MAX_ENTROPY_METHOD_ID",
    "MODEL_ARTIFACT_SHA256",
    "POPULATION_ID",
    "TraceIdentity",
    "TwentyFourError",
    "U28_FEATURES",
    "U28_METHOD_ID",
    "admit_source_rows",
    "eligible_cell_ids",
    "freeze_identity_alignment",
    "incumbent_risk_scores",
    "load_admitted_cell",
    "load_unified28_model",
    "materialize_cell_records",
    "open_and_verify_labels",
    "per_cell_metrics",
    "partial_identity_audit",
    "real_identity_preflight",
    "static_preflight",
    "unified28_from_dict",
    "unified28_parameter_sha256",
    "unified28_replay_source_artifact_sha256",
    "validate_unified28_model",
    "verify_processbench_anchor",
    "verify_source_artifact",
]
