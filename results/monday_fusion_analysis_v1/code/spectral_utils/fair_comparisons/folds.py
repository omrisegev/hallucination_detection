"""Deterministic population hashes and group-isolated comparison folds.

The fair-comparison package treats the source question as the unit of assignment.
Method rows, scorer copies, budgets, and arms must therefore share a ``group_id`` and
receive one fold together.  Fold assignment is deliberately independent of input row
order: unique source questions are stratified by ``(family, stratify_label)``, sorted by
a SHA-256 digest of the source-question group ID, then assigned round-robin to five
folds.  ``stratify_label`` is the binary clean/error label in v1; the lane's raw label
(for example ProcessBench ``-1`` or first-error step) remains a separate field.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


FOLD_REVISION = "fair_comparison_folds_v1.0.0"
DEFAULT_N_FOLDS = 5


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used by package hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    """SHA-256 of UTF-8 text, as a lowercase hexadecimal string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of :func:`canonical_json`."""

    return sha256_text(canonical_json(value))


def ordered_id_sha256(row_ids: Sequence[str]) -> str:
    """Hash an ordered row-id vector without delimiter ambiguity.

    A canonical JSON array is used rather than joining with a character that could
    itself occur in an official identifier.  Duplicate identifiers are rejected: an
    ordered population hash is meaningful only for a one-row-per-ID population.
    """

    ids = [str(row_id) for row_id in row_ids]
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for row_id in ids:
            if row_id in seen and row_id not in duplicates:
                duplicates.append(row_id)
            seen.add(row_id)
        raise ValueError(f"duplicate ordered row ids: {duplicates[:5]}")
    return canonical_sha256(ids)


def _stratum_token(value: Any) -> str:
    """Canonicalize an opaque scalar stratum label for equality and sorting.

    The registered v1 caller uses integer 0/1 clean/error labels.  Treating the label
    opaquely keeps this helper usable for a future preregistered stratification without
    coupling it to a lane's raw target.  JSON scalar/tuple values are accepted; mutable
    mapping/list labels are rejected so a stratum cannot change after assignment.
    """

    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"stratify_label must be hashable, got {value!r}") from exc
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"stratify_label must have a deterministic JSON representation, got {value!r}"
        ) from exc


def assign_group_folds(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    group_key: str = "group_id",
    family_key: str = "family",
    stratum_key: str = "stratify_label",
    namespace: str = "",
) -> dict[str, int]:
    """Assign deterministic stratified folds to unique source-question groups.

    All rows sharing ``group_key`` must agree on family and ``stratum_key``.  In v1 the
    caller supplies binary ``stratify_label`` (0 clean, 1 error), while preserving the
    lane's raw label separately.  This catches accidental method/scorer-copy
    disagreement before it can leak a source question across calibration and
    evaluation.  Within each ``(family, stratify_label)`` stratum, groups are sorted by
    ``SHA256(group_id)`` (with ``group_id`` as the collision tie-breaker) and assigned
    folds ``0..n_folds-1`` round-robin.  ``namespace`` is available only for an
    explicitly registered alternate population; the v1 default is unsalted.
    """

    if int(n_folds) != n_folds or int(n_folds) < 2:
        raise ValueError(f"n_folds must be an integer >=2, got {n_folds!r}")
    n_folds = int(n_folds)

    group_meta: dict[str, tuple[str, str]] = {}
    n_rows = 0
    for row in rows:
        n_rows += 1
        if group_key not in row or family_key not in row or stratum_key not in row:
            missing = [k for k in (group_key, family_key, stratum_key) if k not in row]
            raise KeyError(f"fold row missing required fields: {missing}")
        group_id = str(row[group_key])
        if not group_id:
            raise ValueError("group_id must be non-empty")
        family = str(row[family_key])
        if not family:
            raise ValueError(f"family must be non-empty for group {group_id!r}")
        stratum = _stratum_token(row[stratum_key])
        meta = (family, stratum)
        previous = group_meta.get(group_id)
        if previous is not None and previous != meta:
            raise ValueError(
                f"group {group_id!r} has conflicting (family,stratify_label): "
                f"{previous!r} versus {meta!r}"
            )
        group_meta[group_id] = meta
    if n_rows == 0:
        raise ValueError("cannot assign folds to an empty population")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, meta in group_meta.items():
        strata[meta].append(group_id)

    assignments: dict[str, int] = {}
    for stratum in sorted(strata):
        group_ids = sorted(
            strata[stratum],
            key=lambda group_id: (
                sha256_text(f"{namespace}\0{group_id}" if namespace else group_id),
                group_id,
            ),
        )
        for index, group_id in enumerate(group_ids):
            assignments[group_id] = index % n_folds
    return assignments


def attach_folds(
    rows: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    group_key: str = "group_id",
    fold_key: str = "fold",
) -> list[dict[str, Any]]:
    """Copy rows and attach their already-frozen group fold."""

    result: list[dict[str, Any]] = []
    for row in rows:
        group_id = str(row[group_key])
        if group_id not in assignments:
            raise KeyError(f"no fold assignment for group {group_id!r}")
        item = dict(row)
        item[fold_key] = int(assignments[group_id])
        result.append(item)
    return result


def fold_assignment_sha256(assignments: Mapping[str, int]) -> str:
    """Hash the sorted group-to-fold ledger for registry/manifests."""

    ledger = [
        {"group_id": str(group_id), "fold": int(fold)}
        for group_id, fold in sorted(assignments.items(), key=lambda item: str(item[0]))
    ]
    return canonical_sha256(
        {"fold_revision": FOLD_REVISION, "assignments": ledger}
    )


def assert_group_fold_isolation(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_key: str = "group_id",
    fold_key: str = "fold",
) -> None:
    """Raise when copies of one source question have been split across folds."""

    observed: dict[str, int] = {}
    for row in rows:
        group_id = str(row[group_key])
        fold = int(row[fold_key])
        previous = observed.get(group_id)
        if previous is not None and previous != fold:
            raise ValueError(
                f"group {group_id!r} appears in folds {previous} and {fold}"
            )
        observed[group_id] = fold
