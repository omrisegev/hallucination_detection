"""Strict A/B certification for external final-answer evaluation outputs.

This verifier runs after both independent post-freeze evaluations.  It
revalidates the already-certified score boundary, the controller preparation
manifests, every evaluation artifact and source hash, and the complete
cell/method/metric/population rosters.  It also independently reopens the
pinned label sources post-freeze, reruns their registered adapters, and binds
those labels to the signed score rows before rederiving every point estimate.

The two CSV tables and every label NPZ must be byte-identical.  Evaluation
manifests are compared after normalizing only the explicitly enumerated
build-bound fields below; an unknown difference is never ignored.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .external_ab import (
    assert_external_ab_certificate,
    verify_current_source_snapshot,
)
from .external_evaluation import METRIC_IDS, binary_metric_values
from .external_fit_contract import (
    fit_row_roster_sha256,
    validate_fit_row_identity_contract,
)
from .external_final_answer import (
    ExternalCellSpec,
    ExternalRegistry,
    ID_CONTRACT_VERSION,
    LABEL_SCHEMA_VERSION,
    LabelVector,
    OpaqueIdentityRoster,
    assert_opaque_external_ids,
    load_identity_key,
    load_external_registry,
    load_labels_after_score_freeze,
    sealed_group_roster_commitment,
    validate_public_identity_contract,
)
from .io import (
    canonical_json_bytes,
    canonical_tree_manifest,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .methods import PRIMARY_METHOD_IDS


EVALUATION_SCHEMA_VERSION = "reconstruction-external-evaluation-v2"
EVALUATION_AB_SCHEMA_VERSION = (
    "reconstruction-external-evaluation-ab-certificate-v1"
)
DEFAULT_BOOTSTRAP_DRAWS = 20_000
SUCCESS_STATUSES = frozenset({"OK", "OK_FALLBACK"})

# This is deliberately duplicated from the evaluation controller.  Equality
# of this exact ordered roster is itself part of the verification contract.
EVALUATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/feature_contract.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/evaluate_external_final_answer.py",
    "spectral_utils/fair_comparisons/stopping.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/external_evaluation.py",
    "spectral_utils/reconstruction_benchmark/io.py",
)

VERIFICATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/methods.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "scripts/reconstruction_benchmark/verify_external_evaluation_ab.py",
    "spectral_utils/reconstruction_benchmark/external_evaluation_ab.py",
    "spectral_utils/reconstruction_benchmark/external_evaluation.py",
    "spectral_utils/reconstruction_benchmark/external_fit_contract.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "spectral_utils/reconstruction_benchmark/external_ab.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/methods.py",
)

METRIC_FIELDS = (
    "comparison_group_id", "panel_role", "population_id", "cell_id",
    "dataset_id", "model_id", "slice_id", "method_id", "metric_id",
    "value", "ci_low", "ci_high", "status", "n", "n_incorrect",
    "n_correct", "bootstrap_unit", "bootstrap_draws",
    "bootstrap_valid_draws", "cohort_id", "score_sha256", "label_sha256",
    "record_level", "aggregate_weighting", "aggregate_interpretation",
    "linked_resampling", "stratified_by_label", "n_cells", "n_groups",
)
CONTRAST_FIELDS = (
    "comparison_group_id", "panel_role", "population_id", "cell_id",
    "dataset_id", "model_id", "slice_id", "method_id",
    "reference_method_id", "metric_id", "delta", "ci_low", "ci_high",
    "probability_delta_le_zero", "higher_is_better", "bootstrap_unit",
    "bootstrap_draws", "bootstrap_valid_draws", "n", "n_groups",
    "cohort_id", "record_level", "aggregate_weighting",
    "aggregate_interpretation", "linked_resampling", "stratified_by_label",
    "n_cells", "status",
)

EVALUATION_MANIFEST_FIELDS = frozenset({
    "schema_version", "release_id", "build_id", "scientific_full",
    "ab_verification_status", "ab_certificate_path",
    "ab_certificate_sha256", "ab_certificate_file_sha256",
    "score_freeze_sha256", "score_freeze_payload_sha256",
    "external_registry_sha256", "identity_contract", "id_contract_version",
    "evaluation_source_snapshot", "evaluation_source_snapshot_sha256",
    "source_root", "labels_opened_only_after_score_freeze",
    "score_semantics", "positive_class", "metric_intervals",
    "bootstrap_draws", "n_metric_rows", "n_contrast_rows", "metrics_path",
    "metrics_sha256", "contrasts_path", "contrasts_sha256",
    "label_records", "population_checks", "applicability_statuses",
    "payload_sha256",
})

LABEL_RECORD_FIELDS = frozenset({
    "schema_version", "cell_id", "n_rows", "artifact_path",
    "artifact_sha256", "identity_contract", "id_contract_version",
    "id_contract_sha256", "identity_key_id", "row_namespace_sha256",
    "group_namespace_sha256", "row_roster_sha256",
    "sealed_group_roster_commitment_sha256", "provenance",
})
LABEL_ARRAY_FIELDS = frozenset({
    "row_ids", "group_ids", "incorrect", "id_contract_version",
    "id_contract_sha256", "identity_key_id", "row_namespace_sha256",
    "group_namespace_sha256",
})

MANIFEST_NORMALIZATION_CONTRACT = {
    "top_level_exact_paths": [
        "build_id",
        "score_freeze_sha256",
        "score_freeze_payload_sha256",
        "payload_sha256",
    ],
    "repeated_exact_path": (
        "label_records[*].provenance.score_freeze_payload_sha256"
    ),
    "all_other_fields": "byte-identical canonical JSON required",
}


@dataclass(frozen=True)
class _BuildContext:
    build_id: str
    root: Path
    input_manifest: Mapping[str, Any]
    preparation_manifest: Mapping[str, Any]
    freeze: Mapping[str, Any]
    prepared_by_cell: Mapping[str, Mapping[str, Any]]
    records_by_pair: Mapping[tuple[str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class _LabelState:
    cell_id: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    incorrect: np.ndarray
    artifact_sha256: str
    row_label_sha256: str
    cohort_id: str
    n_groups: int


def _payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _read_regular_file_nofollow(path: Path) -> bytes:
    """Read one existing regular file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError("external evaluation A/B certificate target is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_immutable_certificate(path: Path, payload: bytes) -> None:
    """Publish by atomic no-clobber link, accepting only identical reruns.

    A temporary regular file is fully written and fsynced before ``link(2)``
    exposes it at the certificate path.  The hard-link operation is atomic and
    fails when another process (or symlink) already owns the target, closing
    the check-then-``replace`` race inherent in a conventional atomic writer.
    """

    target = Path(os.path.abspath(os.fspath(path)))
    requested_parent = target.parent
    try:
        parent = requested_parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError("external evaluation A/B certificate parent is absent") from error
    if not parent.is_dir():
        raise RuntimeError(
            "external evaluation A/B certificate parent must be a directory"
        )
    # Resolve directory aliases (including platform paths such as /var ->
    # /private/var) but deliberately never resolve the final component.
    target = parent / target.name
    try:
        existing = _read_regular_file_nofollow(target)
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise FileExistsError(
            "external evaluation A/B certificate target is unsafe"
        ) from error
    if existing is not None:
        if existing != payload:
            raise FileExistsError("external evaluation A/B certificate target already differs")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            try:
                raced = _read_regular_file_nofollow(target)
            except OSError as error:
                raise FileExistsError(
                    "external evaluation A/B certificate target was claimed unsafely"
                ) from error
            if raced != payload:
                raise FileExistsError(
                    "external evaluation A/B certificate target already differs"
                )
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_payload(
    value: Mapping[str, Any], *, field: str, description: str,
) -> None:
    payload = dict(value)
    recorded = payload.pop(field, None)
    if recorded != _payload_sha256(payload):
        raise RuntimeError(f"{description} {field} failed")


def _read_hashed_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} is absent, non-regular, or a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    _verify_payload(value, field="payload_sha256", description=description)
    return value


def _safe_regular_file(root: Path, relative: str, *, description: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{description} has an unsafe path")
    target = root / relative
    resolved_root = root.resolve()
    resolved = target.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{description} escapes its evaluation root") from error
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{description} is absent, non-regular, or a symlink")
    return target


def _scalar_text(
    arrays: Mapping[str, np.ndarray], key: str, *, description: str,
) -> str:
    value = np.asarray(arrays[key])
    if value.shape != (1,):
        raise RuntimeError(f"{description}/{key} is not a one-element scalar")
    return str(value.tolist()[0])


def _row_roster_sha256(
    row_ids: Sequence[str], *, identity_contract: Mapping[str, Any],
    row_namespace_sha256: str,
) -> str:
    return _payload_sha256({
        "schema_version": "reconstruction-external-fit-row-roster-v2",
        "id_contract_version": ID_CONTRACT_VERSION,
        "id_contract_sha256": identity_contract["contract_sha256"],
        "key_id": identity_contract["key_id"],
        "row_namespace_sha256": row_namespace_sha256,
        "row_ids": list(map(str, row_ids)),
    })


def _label_sha256(row_ids: Sequence[str], labels: np.ndarray) -> str:
    return _payload_sha256({
        "row_ids": list(map(str, row_ids)),
        "incorrect": np.asarray(labels, dtype=np.int8).tolist(),
    })


def _source_snapshot(repo: Path, paths: Sequence[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "files": [
            {"path": relative, "sha256": sha256_file(repo / relative)}
            for relative in paths
        ]
    }
    snapshot["snapshot_sha256"] = _payload_sha256(snapshot)
    return snapshot


def _validate_input_context(
    *, release: Path, release_root: Path, release_id: str, build_id: str,
    registry: ExternalRegistry, score_certificate: Mapping[str, Any],
) -> _BuildContext:
    root = release / f"build_{build_id}" / "external_final_answer"
    input_path = root / "inputs/MANIFEST.json"
    freeze_path = root / "fit/SCORE_FREEZE_MANIFEST.json"
    preparation_path = (
        release_root.parent / "private_control" / release_id
        / "external_final_answer" / f"build_{build_id}"
        / "preparation_provenance/MANIFEST.json"
    )
    expected = score_certificate.get("builds", {}).get(build_id)
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"score certificate omits build {build_id}")
    for path, field, description in (
        (input_path, "input_manifest_sha256", "fit-safe input manifest"),
        (freeze_path, "score_freeze_sha256", "score-freeze manifest"),
        (
            preparation_path, "preparation_manifest_sha256",
            "controller preparation manifest",
        ),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"build {build_id}: {description} is absent or unsafe")
        if sha256_file(path) != expected.get(field):
            raise RuntimeError(f"build {build_id}: {description} hash differs from certificate")
    input_manifest = _read_hashed_json(
        input_path, description=f"build {build_id} fit-safe input manifest",
    )
    preparation_manifest = _read_hashed_json(
        preparation_path,
        description=f"build {build_id} controller preparation manifest",
    )
    freeze = _read_hashed_json(
        freeze_path, description=f"build {build_id} score freeze",
    )
    if input_manifest.get("payload_sha256") != expected.get(
        "input_manifest_payload_sha256"
    ):
        raise RuntimeError(f"build {build_id}: input manifest payload is uncertified")
    if preparation_manifest.get("payload_sha256") != expected.get(
        "preparation_manifest_payload_sha256"
    ):
        raise RuntimeError(f"build {build_id}: preparation payload is uncertified")
    if freeze.get("payload_sha256") != expected.get("score_freeze_payload_sha256"):
        raise RuntimeError(f"build {build_id}: score-freeze payload is uncertified")

    for name, value in (
        ("fit-safe input", input_manifest),
        ("controller preparation", preparation_manifest),
        ("score freeze", freeze),
    ):
        if value.get("release_id") != release_id or value.get("build_id") != build_id:
            raise RuntimeError(f"build {build_id}: {name} belongs to another run")
    if (
        input_manifest.get("schema_version")
        != "reconstruction-external-fit-safe-build-v1"
        or preparation_manifest.get("schema_version")
        != "reconstruction-external-target-free-build-v2"
        or freeze.get("schema_version")
        != "reconstruction-external-score-freeze-v2"
    ):
        raise RuntimeError(f"build {build_id}: preparation/fit schema drifted")
    if (
        input_manifest.get("scientific_full_build") is not True
        or preparation_manifest.get("scientific_full_build") is not True
        or freeze.get("scientific_full") is not True
        or freeze.get("all_expected_scores_present") is not True
    ):
        raise RuntimeError(f"build {build_id}: partial/debug inputs cannot be certified")
    prelabel_contract = {
        "input_target_data_opened": input_manifest.get("target_data_opened"),
        "input_historical_scores_opened": input_manifest.get("historical_scores_opened"),
        "preparation_labels_opened": preparation_manifest.get("labels_opened"),
        "preparation_historical_scores_opened": preparation_manifest.get(
            "historical_scores_opened"
        ),
    }
    if any(value is not False for value in prelabel_contract.values()):
        raise RuntimeError(f"build {build_id}: preparation/input target firewall drifted")
    if (
        input_manifest.get("mixed_v2_applied_exactly_once") is not True
        or preparation_manifest.get("mixed_v2_applied_exactly_once") is not True
    ):
        raise RuntimeError(f"build {build_id}: one-pass mixed-v2 contract drifted")
    if (
        input_manifest.get("external_registry_sha256") != registry.sha256
        or preparation_manifest.get("external_registry_sha256") != registry.sha256
        or freeze.get("external_registry_sha256") != registry.sha256
    ):
        raise RuntimeError(f"build {build_id}: external registry binding drifted")
    if (
        input_manifest.get("population_registry_sha256")
        != registry.population_registry_sha256
        or preparation_manifest.get("population_registry_sha256")
        != registry.population_registry_sha256
        or freeze.get("population_registry_sha256")
        != registry.population_registry_sha256
    ):
        raise RuntimeError(f"build {build_id}: population registry binding drifted")
    isolation = {
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "historical_scores_opened": False,
        "donors_used": False,
        "family_nrm_pgrd_regime": "A_within_cell_fully_unsupervised",
    }
    if any(freeze.get(key) != value for key, value in isolation.items()):
        raise RuntimeError(f"build {build_id}: score-freeze isolation contract drifted")

    registered_ids = [cell.cell_id for cell in registry.cells]
    runnable_count = sum(cell.fit_policy == "run_if_compatible" for cell in registry.cells)
    for name, value in (
        ("fit-safe input", input_manifest),
        ("controller preparation", preparation_manifest),
    ):
        if (
            value.get("applicability_complete") is not True
            or int(value.get("n_registered_cells", -1)) != len(registered_ids)
            or int(value.get("n_runnable_cells", -1)) != runnable_count
            or value.get("id_contract_version") != ID_CONTRACT_VERSION
        ):
            raise RuntimeError(f"build {build_id}: {name} completeness/identity drifted")
    fit_identity = validate_fit_row_identity_contract(
        score_certificate["identity_contract"]
    )
    full_identity = validate_public_identity_contract(
        preparation_manifest.get("identity_contract", {})
    )
    if (
        input_manifest.get("identity_contract") != fit_identity
        or preparation_manifest.get("fit_row_identity_contract") != fit_identity
        or preparation_manifest.get("identity_contract") != full_identity
        or freeze.get("identity_contract") != fit_identity
        or freeze.get("id_contract_version") != ID_CONTRACT_VERSION
        or full_identity.get("key_id") != fit_identity.get("key_id")
    ):
        raise RuntimeError(f"build {build_id}: fit-row identity contract drifted")
    prep_cells = preparation_manifest.get("cells")
    input_cells = input_manifest.get("cells")
    if (
        not isinstance(prep_cells, list)
        or not isinstance(input_cells, list)
        or [str(row.get("cell_id", "")) for row in prep_cells] != registered_ids
        or [str(row.get("cell_id", "")) for row in input_cells] != registered_ids
    ):
        raise RuntimeError(f"build {build_id}: application roster is incomplete or reordered")
    eligible = [
        str(row["cell_id"]) for row in prep_cells if row.get("status") == "ELIGIBLE"
    ]
    if eligible != list(map(str, score_certificate.get("cell_ids", ()))):
        raise RuntimeError(f"build {build_id}: eligible cell roster differs from certificate")
    input_statuses = [
        (str(row.get("cell_id", "")), str(row.get("status", "")))
        for row in input_cells
    ]
    prep_statuses = [
        (str(row.get("cell_id", "")), str(row.get("status", "")))
        for row in prep_cells
    ]
    if input_statuses != prep_statuses:
        raise RuntimeError(f"build {build_id}: public/private applicability statuses differ")
    freeze_statuses = [
        (str(row.get("cell_id", "")), str(row.get("status", "")))
        for row in freeze.get("applicability_statuses", ())
    ]
    if freeze_statuses != prep_statuses:
        raise RuntimeError(f"build {build_id}: score-freeze applicability roster differs")
    if (
        int(input_manifest.get("n_prepared_cells", -1)) != len(eligible)
        or int(preparation_manifest.get("n_prepared_cells", -1)) != len(eligible)
        or input_manifest.get("complete_eligible_roster") is not True
        or preparation_manifest.get("complete_eligible_roster") is not True
    ):
        raise RuntimeError(f"build {build_id}: prepared-cell completeness gate failed")

    if tuple(map(str, freeze.get("cell_ids", ()))) != tuple(eligible):
        raise RuntimeError(f"build {build_id}: score-freeze cell roster drifted")
    if tuple(map(str, freeze.get("method_ids", ()))) != PRIMARY_METHOD_IDS:
        raise RuntimeError(f"build {build_id}: score-freeze method roster is not the exact 13")
    records = freeze.get("records")
    expected_pairs = {
        (cell_id, method_id)
        for cell_id in eligible for method_id in PRIMARY_METHOD_IDS
    }
    if not isinstance(records, list):
        raise RuntimeError(f"build {build_id}: score-freeze records are absent")
    pairs = [
        (str(row.get("cell_id", "")), str(row.get("method_id", "")))
        for row in records
    ]
    if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
        raise RuntimeError(f"build {build_id}: score records are not the complete Cartesian roster")
    if int(freeze.get("n_records", -1)) != len(expected_pairs):
        raise RuntimeError(f"build {build_id}: score-freeze record count drifted")
    records_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for pair, record in zip(pairs, records):
        if record.get("status") not in SUCCESS_STATUSES:
            raise RuntimeError(f"build {build_id}: unsuccessful frozen record {pair}")
        score_path = _safe_regular_file(
            root / "fit", str(record.get("score_path", "")),
            description=f"build {build_id} score {pair}",
        )
        if sha256_file(score_path) != record.get("score_sha256"):
            raise RuntimeError(f"build {build_id}: frozen score hash failed for {pair}")
        records_by_pair[pair] = record

    prepared_by_cell = {str(row["cell_id"]): row for row in prep_cells}
    return _BuildContext(
        build_id=build_id,
        root=root,
        input_manifest=input_manifest,
        preparation_manifest=preparation_manifest,
        freeze=freeze,
        prepared_by_cell=prepared_by_cell,
        records_by_pair=records_by_pair,
    )


def _load_label_states(
    *, evaluation_root: Path, manifest: Mapping[str, Any],
    context: _BuildContext, registry: ExternalRegistry,
    score_certificate: Mapping[str, Any], identity_key: bytes,
    source_root: Path,
    label_loader: Callable[..., LabelVector] = load_labels_after_score_freeze,
) -> tuple[dict[str, _LabelState], dict[str, str], dict[str, bytes]]:
    label_records = manifest.get("label_records")
    cell_ids = list(map(str, score_certificate["cell_ids"]))
    if (
        not isinstance(label_records, list)
        or [str(row.get("cell_id", "")) for row in label_records] != cell_ids
    ):
        raise RuntimeError(f"build {context.build_id}: label artifact roster drifted")
    states: dict[str, _LabelState] = {}
    file_hashes: dict[str, str] = {}
    file_bytes: dict[str, bytes] = {}
    fit_identity = validate_fit_row_identity_contract(
        score_certificate["identity_contract"]
    )
    full_identity = validate_public_identity_contract(
        context.preparation_manifest.get("identity_contract", {})
    )
    if (
        context.preparation_manifest.get("fit_row_identity_contract") != fit_identity
        or full_identity.get("key_id") != fit_identity.get("key_id")
    ):
        raise RuntimeError(f"build {context.build_id}: full/fit identity join drifted")
    for record in label_records:
        cell_id = str(record.get("cell_id", ""))
        if set(record) != LABEL_RECORD_FIELDS:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label record fields drifted")
        if (
            record.get("schema_version") != LABEL_SCHEMA_VERSION
            or record.get("identity_contract") != full_identity
            or record.get("id_contract_version") != ID_CONTRACT_VERSION
            or record.get("id_contract_sha256") != full_identity["contract_sha256"]
            or record.get("identity_key_id") != full_identity["key_id"]
        ):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label identity binding drifted")
        expected_relative = f"labels/{cell_id}.npz"
        if record.get("artifact_path") != expected_relative:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label path drifted")
        path = _safe_regular_file(
            evaluation_root, expected_relative,
            description=f"build {context.build_id}/{cell_id} label artifact",
        )
        artifact_payload = path.read_bytes()
        file_sha = sha256_bytes(artifact_payload)
        if file_sha != record.get("artifact_sha256"):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label file hash failed")
        arrays = load_npz_no_pickle(path)
        if path.read_bytes() != artifact_payload:
            raise RuntimeError(
                f"build {context.build_id}/{cell_id}: label artifact changed while verifying"
            )
        if set(arrays) != LABEL_ARRAY_FIELDS:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label array roster drifted")
        scalar_bindings = {
            key: _scalar_text(
                arrays, key, description=f"build {context.build_id}/{cell_id}",
            )
            for key in (
                "id_contract_version", "id_contract_sha256", "identity_key_id",
                "row_namespace_sha256", "group_namespace_sha256",
            )
        }
        expected_scalars = {
            key: str(record[key]) for key in scalar_bindings
        }
        if scalar_bindings != expected_scalars:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label scalar binding failed")
        row_ids = tuple(map(str, np.asarray(arrays["row_ids"]).tolist()))
        group_ids = tuple(map(str, np.asarray(arrays["group_ids"]).tolist()))
        incorrect_raw = np.asarray(arrays["incorrect"])
        if (
            incorrect_raw.ndim != 1
            or not np.isin(incorrect_raw, (0, 1)).all()
        ):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: labels are not binary")
        incorrect = incorrect_raw.astype(np.int8, copy=False)
        spec = registry.by_cell[cell_id]
        if (
            len(row_ids) != spec.expected_rows
            or int(record.get("n_rows", -1)) != spec.expected_rows
            or len(group_ids) != spec.expected_rows
            or incorrect.shape != (spec.expected_rows,)
            or row_ids != tuple(sorted(row_ids))
            or len(set(row_ids)) != len(row_ids)
            or len(np.unique(incorrect)) != 2
        ):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label cohort is invalid")
        assert_opaque_external_ids(row_ids, group_ids)
        row_namespace = str(record["row_namespace_sha256"])
        group_namespace = str(record["group_namespace_sha256"])
        label_row_roster = _row_roster_sha256(
            row_ids, identity_contract=full_identity,
            row_namespace_sha256=row_namespace,
        )
        identity_roster = OpaqueIdentityRoster(
            row_ids=row_ids,
            group_ids=group_ids,
            contract_binding=full_identity,
            row_namespace_sha256=row_namespace,
            group_namespace_sha256=group_namespace,
        )
        cohort_id = sealed_group_roster_commitment(identity_roster)
        prepared = context.prepared_by_cell[cell_id]
        prepared_fit_roster = fit_row_roster_sha256(
            row_ids,
            contract=fit_identity,
            row_namespace_sha256_value=row_namespace,
        )
        if (
            record.get("row_roster_sha256") != label_row_roster
            or record.get("sealed_group_roster_commitment_sha256") != cohort_id
            or prepared.get("identity_contract") != full_identity
            or prepared.get("fit_row_identity_contract") != fit_identity
            or prepared.get("id_contract_sha256") != full_identity["contract_sha256"]
            or prepared.get("fit_row_id_contract_sha256") != fit_identity["contract_sha256"]
            or prepared.get("identity_key_id") != full_identity["key_id"]
            or prepared.get("row_roster_sha256") != prepared_fit_roster
            or prepared.get("sealed_group_roster_commitment_sha256") != cohort_id
            or prepared.get("row_namespace_sha256") != row_namespace
            or prepared.get("group_namespace_sha256") != group_namespace
        ):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label/preparation cohort binding failed")
        provenance = record.get("provenance")
        label_sha = _label_sha256(row_ids, incorrect)
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"build {context.build_id}/{cell_id}: label provenance is absent")
        required_provenance = {
            "row_label_sha256": label_sha,
            "positive_class": "incorrect",
            "n_incorrect": int(incorrect.sum()),
            "n_correct": int(len(incorrect) - incorrect.sum()),
            "score_freeze_payload_sha256": context.freeze["payload_sha256"],
            "identity_contract": full_identity,
            "id_contract_version": ID_CONTRACT_VERSION,
            "id_contract_sha256": full_identity["contract_sha256"],
            "identity_key_id": full_identity["key_id"],
            "row_namespace_sha256": row_namespace,
            "group_namespace_sha256": group_namespace,
            "row_roster_sha256": label_row_roster,
            "sealed_group_roster_commitment_sha256": cohort_id,
        }
        for key, expected in required_provenance.items():
            if provenance.get(key) != expected:
                raise RuntimeError(
                    f"build {context.build_id}/{cell_id}: label provenance {key} drifted"
                )
        # An A/B equality check alone cannot detect the same coordinated label
        # permutation in both builds.  Re-open the pinned raw label sources and
        # execute the registered adapter independently, joined to the signed
        # score cohort rather than to the persisted label artifact.
        anchor_record = context.records_by_pair[(cell_id, "iu_pcr")]
        anchor_arrays = load_npz_no_pickle(
            context.root / "fit" / str(anchor_record["score_path"])
        )
        expected_score_rows = tuple(map(str, anchor_arrays["row_ids"].tolist()))
        rederived = label_loader(
            registry=registry,
            spec=spec,
            repo=source_root,
            score_freeze=context.freeze,
            expected_row_ids=expected_score_rows,
            expected_group_roster_commitment_sha256=prepared[
                "sealed_group_roster_commitment_sha256"
            ],
            identity_key=identity_key,
        )
        if (
            rederived.cell_id != cell_id
            or tuple(map(str, rederived.row_ids)) != expected_score_rows
            or tuple(map(str, rederived.row_ids)) != row_ids
            or tuple(map(str, rederived.group_ids)) != group_ids
            or not np.array_equal(
                np.asarray(rederived.incorrect, dtype=np.int8), incorrect
            )
            or canonical_json_bytes(dict(rederived.provenance))
            != canonical_json_bytes(dict(provenance))
        ):
            raise RuntimeError(
                f"build {context.build_id}/{cell_id}: persisted labels differ "
                "from independent registry/source rederivation"
            )
        if spec.expected_incorrect is not None and int(incorrect.sum()) != spec.expected_incorrect:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: incorrect count drifted")
        if spec.expected_correct is not None and int(len(incorrect) - incorrect.sum()) != spec.expected_correct:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: correct count drifted")
        observed_groups = len(set(group_ids))
        if spec.expected_group_count is not None and observed_groups != spec.expected_group_count:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: source-group count drifted")
        if int(prepared.get("group_count", observed_groups)) != observed_groups:
            raise RuntimeError(f"build {context.build_id}/{cell_id}: preparation group count drifted")
        states[cell_id] = _LabelState(
            cell_id=cell_id,
            row_ids=row_ids,
            group_ids=group_ids,
            incorrect=incorrect,
            artifact_sha256=file_sha,
            row_label_sha256=label_sha,
            cohort_id=cohort_id,
            n_groups=observed_groups,
        )
        file_hashes[expected_relative] = file_sha
        file_bytes[expected_relative] = artifact_payload
    return states, file_hashes, file_bytes


def _read_csv_exact(
    path: Path, *, fields: Sequence[str], description: str,
) -> tuple[list[dict[str, str]], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} is absent, non-regular, or a symlink")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{description} is not UTF-8") from error
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise RuntimeError(f"{description} header roster/order drifted")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise RuntimeError(f"{description} is empty or contains extra columns")
    return rows, payload


def _require_text(row: Mapping[str, str], key: str, expected: str, *, where: str) -> None:
    if row.get(key) != expected:
        raise RuntimeError(f"{where}: {key} drifted")


def _int_field(row: Mapping[str, str], key: str, *, where: str) -> int:
    raw = row.get(key, "")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{where}: {key} is not an integer") from error
    if str(value) != raw:
        raise RuntimeError(f"{where}: {key} is not canonical integer text")
    return value


def _float_field(row: Mapping[str, str], key: str, *, where: str) -> float:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{where}: {key} is not numeric") from error
    if not math.isfinite(value):
        raise RuntimeError(f"{where}: {key} is non-finite")
    return value


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _comparison_group_id(
    *, level: str, cell_id: str | None, population_id: str | None,
    cohort_id: str, metric_id: str, aggregate: Mapping[str, Any] | None = None,
) -> str:
    if level == "cell":
        payload = {
            "cell_id": cell_id,
            "cohort_id": cohort_id,
            "metric_id": metric_id,
            "positive_class": "incorrect",
        }
        prefix = "external_final_answer::cell::"
    elif level == "population":
        if aggregate is None:
            raise AssertionError("population comparison group lacks aggregate rule")
        payload = {
            "population_id": population_id,
            "cohort_id": cohort_id,
            "metric_id": metric_id,
            "positive_class": "incorrect",
            "weighting": aggregate["weighting"],
            "interpretation": aggregate["interpretation"],
        }
        prefix = "external_final_answer::population::"
    else:  # pragma: no cover - internal contract
        raise AssertionError(level)
    return prefix + _payload_sha256(payload)[:24]


def _population_link_blocks(
    *, specs: Sequence[ExternalCellSpec], labels: Mapping[str, _LabelState],
    aggregate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool, bool]:
    cell_ids = tuple(sorted(spec.cell_id for spec in specs))
    link_rule = str(aggregate["link_cells_by"])
    if link_rule == "none":
        link_keys = {cell_id: f"__independent__:{cell_id}" for cell_id in cell_ids}
    elif link_rule == "slice_id":
        by_id = {spec.cell_id: spec for spec in specs}
        link_keys = {cell_id: by_id[cell_id].slice_id for cell_id in cell_ids}
    elif link_rule == "all":
        link_keys = {cell_id: "all_registered_cells" for cell_id in cell_ids}
    else:
        raise RuntimeError("population aggregate has an unknown linkage rule")
    cells_by_link: dict[str, list[str]] = {}
    for cell_id in cell_ids:
        cells_by_link.setdefault(link_keys[cell_id], []).append(cell_id)
    stratified = aggregate.get("bootstrap") == "source_group_stratified_by_label"
    blocks: list[dict[str, Any]] = []
    for link_key in sorted(cells_by_link):
        linked_cells = tuple(sorted(cells_by_link[link_key]))
        first = labels[linked_cells[0]]
        roster = tuple(sorted(set(first.group_ids)))
        members = {
            group: [index for index, value in enumerate(first.group_ids) if value == group]
            for group in roster
        }
        counts = {group: len(members[group]) for group in roster}
        group_labels: dict[str, int] | None = None
        if stratified:
            group_labels = {}
            for group in roster:
                values = np.unique(first.incorrect[np.asarray(members[group], dtype=int)])
                if len(values) != 1:
                    raise RuntimeError("stratified population contains a mixed-label source group")
                group_labels[group] = int(values[0])
            if set(group_labels.values()) != {0, 1}:
                raise RuntimeError("stratified population lacks both group-label strata")
        for cell_id in linked_cells[1:]:
            current = labels[cell_id]
            current_roster = tuple(sorted(set(current.group_ids)))
            current_counts = {
                group: sum(value == group for value in current.group_ids)
                for group in current_roster
            }
            if current_roster != roster or current_counts != counts:
                raise RuntimeError(f"linked population group roster differs in {link_key}")
            if stratified:
                assert group_labels is not None
                for group in roster:
                    indices = np.asarray([
                        index for index, value in enumerate(current.group_ids)
                        if value == group
                    ], dtype=int)
                    values = np.unique(current.incorrect[indices])
                    if len(values) != 1 or int(values[0]) != group_labels[group]:
                        raise RuntimeError(f"linked population group labels differ in {link_key}")
        block: dict[str, Any] = {
            "link_key": link_key,
            "cell_ids": list(linked_cells),
            "linked": len(linked_cells) > 1,
            "n_groups": len(roster),
            "rows_per_cell": sum(counts.values()),
            "group_roster_sha256": _payload_sha256(list(roster)),
            "group_member_counts_sha256": _payload_sha256([
                {"group_id": group, "member_count": counts[group]}
                for group in roster
            ]),
        }
        if stratified:
            assert group_labels is not None
            block["group_labels_sha256"] = _payload_sha256([
                {"group_id": group, "label": group_labels[group]}
                for group in roster
            ])
            block["groups_by_label"] = {
                str(label): sum(group_labels[group] == label for group in roster)
                for label in (0, 1)
            }
        blocks.append(block)
    return blocks, any(bool(block["linked"]) for block in blocks), stratified


def _expected_population_checks(
    *, registry: ExternalRegistry, context: _BuildContext,
    labels: Mapping[str, _LabelState],
) -> list[dict[str, Any]]:
    statuses = {
        str(row["cell_id"]): str(row["status"])
        for row in context.preparation_manifest["cells"]
    }
    checks: list[dict[str, Any]] = []
    for population_id, expected in registry.raw.get(
        "population_expectations", {}
    ).items():
        specs = [cell for cell in registry.cells if cell.population_id == population_id]
        cell_statuses = {cell.cell_id: statuses.get(cell.cell_id, "MISSING") for cell in specs}
        if not specs or not all(value == "ELIGIBLE" for value in cell_statuses.values()):
            checks.append({
                "population_id": population_id,
                "status": "NOT_AGGREGATED_INCOMPLETE_OR_INAPPLICABLE",
                "cell_statuses": cell_statuses,
            })
            continue
        arrays = [labels[cell.cell_id].incorrect for cell in specs]
        joined = np.concatenate(arrays)
        observed = {
            "rows": int(len(joined)),
            "incorrect": int(joined.sum()),
            "correct": int(len(joined) - joined.sum()),
            "cells": len(specs),
        }
        keys = ("rows", "incorrect", "correct", "cells")
        try:
            registered = {key: int(expected[key]) for key in keys}
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"population expectation contract is malformed for {population_id}"
            ) from error
        class_contract_complete = all(
            cell.expected_incorrect is not None
            and cell.expected_correct is not None
            for cell in specs
        )
        if not class_contract_complete:
            if observed != registered:
                raise RuntimeError(
                    f"population label totals failed for {population_id}: "
                    f"{observed} != {registered}"
                )
        else:
            for cell in specs:
                if int(cell.expected_incorrect) + int(cell.expected_correct) != int(
                    cell.expected_rows
                ):
                    raise RuntimeError(
                        "atomic expectation contract is internally inconsistent "
                        f"for {population_id}/{cell.cell_id}"
                    )
            atomic_expected = {
                "rows": sum(int(cell.expected_rows) for cell in specs),
                "incorrect": sum(int(cell.expected_incorrect) for cell in specs),
                "correct": sum(int(cell.expected_correct) for cell in specs),
                "cells": len(specs),
            }
            if observed != atomic_expected:
                raise RuntimeError(
                    f"population atomic label totals failed for {population_id}: "
                    f"{observed} != {atomic_expected}"
                )
            if registered != atomic_expected:
                if any(
                    registered[key] != atomic_expected[key]
                    for key in ("rows", "cells")
                ):
                    raise RuntimeError(
                        f"population registry structural totals failed for {population_id}: "
                        f"{registered} != {atomic_expected}"
                    )
                checks.append({
                    "population_id": population_id,
                    "status": "AGGREGATE_BLOCKED_REGISTRY_CLASS_TOTAL_MISMATCH",
                    "registered_summary": registered,
                    "atomic_expected": atomic_expected,
                    "observed": observed,
                })
                continue
        aggregate = registry.raw["population_aggregates"][population_id]
        if aggregate.get("enabled") is not True:
            checks.append({
                "population_id": population_id,
                "status": "AGGREGATE_DISABLED_BY_REGISTRY",
                "reason": aggregate.get("reason"),
                "observed": observed,
            })
            continue
        blocks, linked, stratified = _population_link_blocks(
            specs=specs, labels=labels, aggregate=aggregate,
        )
        bootstrap_unit = "linked_source_group" if linked else "source_group"
        if stratified:
            bootstrap_unit += "_stratified_by_label"
        seed_payload = (
            f"{registry.sha256}:{population_id}:"
            "population-grouped-paired-bootstrap-v1"
        ).encode("utf-8")
        seed = int(aggregate.get("seed", int(sha256_bytes(seed_payload)[:8], 16)))
        checks.append({
            "population_id": population_id,
            "status": "OK_AGGREGATED",
            "observed": observed,
            "weighting": aggregate["weighting"],
            "interpretation": aggregate["interpretation"],
            "bootstrap_unit": bootstrap_unit,
            "seed": seed,
            "link_blocks": blocks,
        })
    return checks


def _load_score_values(
    *, context: _BuildContext, labels: Mapping[str, _LabelState],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, float]]]]:
    scores: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, dict[str, float]]] = {}
    fit_identity = validate_fit_row_identity_contract(
        context.freeze.get("identity_contract", {})
    )
    score_fields = {
        "row_ids", "score", "id_contract_version", "id_contract_sha256",
        "identity_key_id", "row_namespace_sha256", "row_roster_sha256",
    }
    for cell_id, label in labels.items():
        scores[cell_id] = {}
        points[cell_id] = {}
        for method_id in PRIMARY_METHOD_IDS:
            record = context.records_by_pair[(cell_id, method_id)]
            path = context.root / "fit" / str(record["score_path"])
            arrays = load_npz_no_pickle(path)
            if set(arrays) != score_fields:
                raise RuntimeError(f"build {context.build_id}/{cell_id}/{method_id}: score array roster drifted")
            row_ids = tuple(map(str, np.asarray(arrays["row_ids"]).tolist()))
            score = np.asarray(arrays["score"], dtype=float)
            if row_ids != label.row_ids or score.shape != label.incorrect.shape or not np.isfinite(score).all():
                raise RuntimeError(f"build {context.build_id}/{cell_id}/{method_id}: score/label binding failed")
            prepared = context.prepared_by_cell[cell_id]
            scalar_bindings = {
                key: _scalar_text(
                    arrays, key,
                    description=f"build {context.build_id}/{cell_id}/{method_id}",
                )
                for key in (
                    "id_contract_version", "id_contract_sha256",
                    "identity_key_id", "row_namespace_sha256",
                    "row_roster_sha256",
                )
            }
            expected_bindings = {
                "id_contract_version": ID_CONTRACT_VERSION,
                "id_contract_sha256": fit_identity["contract_sha256"],
                "identity_key_id": fit_identity["key_id"],
                "row_namespace_sha256": str(prepared["row_namespace_sha256"]),
                "row_roster_sha256": str(prepared["row_roster_sha256"]),
            }
            if scalar_bindings != expected_bindings or any(
                record.get(key) != expected
                for key, expected in expected_bindings.items()
            ):
                raise RuntimeError(f"build {context.build_id}/{cell_id}/{method_id}: fit identity binding drifted")
            scores[cell_id][method_id] = score
            points[cell_id][method_id] = binary_metric_values(label.incorrect, score)
    return scores, points


def _validate_numeric_interval(
    row: Mapping[str, str], *, metric_id: str, contrast: bool, where: str,
) -> tuple[float, float, float]:
    center_key = "delta" if contrast else "value"
    center = _float_field(row, center_key, where=where)
    low = _float_field(row, "ci_low", where=where)
    high = _float_field(row, "ci_high", where=where)
    if low > high:
        raise RuntimeError(f"{where}: confidence interval is reversed")
    if not contrast:
        bound = 1000.0 if metric_id == "aurc_x1000" else 1.0
        if not (0.0 <= center <= bound and 0.0 <= low <= bound and 0.0 <= high <= bound):
            raise RuntimeError(f"{where}: metric or interval is outside its range")
    return center, low, high


def _validate_cell_rows(
    *, metric_rows: Sequence[Mapping[str, str]],
    contrast_rows: Sequence[Mapping[str, str]], registry: ExternalRegistry,
    context: _BuildContext, labels: Mapping[str, _LabelState],
    point_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    cell_ids = list(labels)
    methods = tuple(sorted(PRIMARY_METHOD_IDS))
    candidate_methods = tuple(method for method in methods if method != "iu_pcr")
    expected_metric_order = [
        (cell_id, method_id, metric_id)
        for cell_id in cell_ids for method_id in methods for metric_id in METRIC_IDS
    ]
    observed_metric_order = [
        (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"]))
        for row in metric_rows if row.get("record_level") == "cell"
    ]
    if observed_metric_order != expected_metric_order:
        raise RuntimeError("external cell metrics are not the exact ordered cell x method x metric roster")
    expected_contrast_order = [
        (cell_id, method_id, metric_id)
        for cell_id in cell_ids for method_id in candidate_methods for metric_id in METRIC_IDS
    ]
    observed_contrast_order = [
        (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"]))
        for row in contrast_rows if row.get("record_level") == "cell"
    ]
    if observed_contrast_order != expected_contrast_order:
        raise RuntimeError("external cell contrasts are not the exact ordered Cartesian roster")

    metric_by_key = {
        (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"])): row
        for row in metric_rows if row.get("record_level") == "cell"
    }
    contrast_by_key = {
        (str(row["cell_id"]), str(row["method_id"]), str(row["metric_id"])): row
        for row in contrast_rows if row.get("record_level") == "cell"
    }
    for cell_id in cell_ids:
        spec = registry.by_cell[cell_id]
        label = labels[cell_id]
        n_incorrect = int(label.incorrect.sum())
        stratified = (
            registry.raw["population_aggregates"][spec.population_id].get("bootstrap")
            == "source_group_stratified_by_label"
        )
        valid_draws: set[int] = set()
        for method_id in methods:
            method_record = context.records_by_pair[(cell_id, method_id)]
            for metric_id in METRIC_IDS:
                row = metric_by_key[(cell_id, method_id, metric_id)]
                where = f"cell metric {cell_id}/{method_id}/{metric_id}"
                expected_text = {
                    "comparison_group_id": _comparison_group_id(
                        level="cell", cell_id=cell_id, population_id=None,
                        cohort_id=label.cohort_id, metric_id=metric_id,
                    ),
                    "panel_role": spec.panel_role,
                    "population_id": spec.population_id,
                    "cell_id": cell_id,
                    "dataset_id": spec.dataset_id,
                    "model_id": spec.model_id,
                    "slice_id": spec.slice_id,
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "status": str(method_record["status"]),
                    "bootstrap_unit": "source_group",
                    "cohort_id": label.cohort_id,
                    "score_sha256": str(method_record["score_sha256"]),
                    "label_sha256": label.row_label_sha256,
                    "record_level": "cell",
                    "aggregate_weighting": "",
                    "aggregate_interpretation": "",
                    "linked_resampling": "",
                    "stratified_by_label": _bool_text(stratified),
                }
                for key, expected in expected_text.items():
                    _require_text(row, key, expected, where=where)
                expected_ints = {
                    "n": len(label.incorrect),
                    "n_incorrect": n_incorrect,
                    "n_correct": len(label.incorrect) - n_incorrect,
                    "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                    "n_cells": 1,
                    "n_groups": label.n_groups,
                }
                for key, expected in expected_ints.items():
                    if _int_field(row, key, where=where) != expected:
                        raise RuntimeError(f"{where}: {key} drifted")
                valid = _int_field(row, "bootstrap_valid_draws", where=where)
                if not (0 < valid <= DEFAULT_BOOTSTRAP_DRAWS):
                    raise RuntimeError(f"{where}: bootstrap valid-draw count is invalid")
                valid_draws.add(valid)
                value, _, _ = _validate_numeric_interval(
                    row, metric_id=metric_id, contrast=False, where=where,
                )
                if value != point_metrics[cell_id][method_id][metric_id]:
                    raise RuntimeError(f"{where}: point estimate differs from frozen score/label data")
        if len(valid_draws) != 1:
            raise RuntimeError(f"{cell_id}: paired bootstrap valid-draw counts diverged")

        for method_id in candidate_methods:
            candidate_status = context.records_by_pair[(cell_id, method_id)]["status"]
            reference_status = context.records_by_pair[(cell_id, "iu_pcr")]["status"]
            expected_status = (
                "OK_FALLBACK"
                if "OK_FALLBACK" in {candidate_status, reference_status}
                else "OK"
            )
            for metric_id in METRIC_IDS:
                row = contrast_by_key[(cell_id, method_id, metric_id)]
                where = f"cell contrast {cell_id}/{method_id}/{metric_id}"
                expected_text = {
                    "comparison_group_id": _comparison_group_id(
                        level="cell", cell_id=cell_id, population_id=None,
                        cohort_id=label.cohort_id, metric_id=metric_id,
                    ),
                    "panel_role": spec.panel_role,
                    "population_id": spec.population_id,
                    "cell_id": cell_id,
                    "dataset_id": spec.dataset_id,
                    "model_id": spec.model_id,
                    "slice_id": spec.slice_id,
                    "method_id": method_id,
                    "reference_method_id": "iu_pcr",
                    "metric_id": metric_id,
                    "higher_is_better": _bool_text(metric_id != "aurc_x1000"),
                    "bootstrap_unit": "source_group",
                    "cohort_id": label.cohort_id,
                    "record_level": "cell",
                    "aggregate_weighting": "",
                    "aggregate_interpretation": "",
                    "linked_resampling": "",
                    "stratified_by_label": _bool_text(stratified),
                    "status": expected_status,
                }
                for key, expected in expected_text.items():
                    _require_text(row, key, expected, where=where)
                expected_ints = {
                    "n": len(label.incorrect),
                    "n_groups": label.n_groups,
                    "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                    "n_cells": 1,
                }
                for key, expected in expected_ints.items():
                    if _int_field(row, key, where=where) != expected:
                        raise RuntimeError(f"{where}: {key} drifted")
                if _int_field(row, "bootstrap_valid_draws", where=where) not in valid_draws:
                    raise RuntimeError(f"{where}: contrast draw count is not paired")
                delta, _, _ = _validate_numeric_interval(
                    row, metric_id=metric_id, contrast=True, where=where,
                )
                expected_delta = (
                    point_metrics[cell_id][method_id][metric_id]
                    - point_metrics[cell_id]["iu_pcr"][metric_id]
                )
                if delta != expected_delta:
                    raise RuntimeError(f"{where}: point delta differs from metric table")
                probability = _float_field(
                    row, "probability_delta_le_zero", where=where,
                )
                if not 0.0 <= probability <= 1.0:
                    raise RuntimeError(f"{where}: bootstrap probability is out of range")


def _population_metadata(
    *, population_id: str, registry: ExternalRegistry,
    labels: Mapping[str, _LabelState], point_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    context: _BuildContext, check: Mapping[str, Any],
) -> dict[str, Any]:
    specs = [cell for cell in registry.cells if cell.population_id == population_id]
    cell_ids = [cell.cell_id for cell in specs]
    aggregate = registry.raw["population_aggregates"][population_id]
    datasets = sorted({cell.dataset_id for cell in specs})
    models = sorted({cell.model_id for cell in specs})
    cohort_id = _payload_sha256([
        {"cell_id": cell_id, "cohort_id": labels[cell_id].cohort_id}
        for cell_id in cell_ids
    ])
    label_sha = _payload_sha256([
        {"cell_id": cell_id, "label_sha256": labels[cell_id].row_label_sha256}
        for cell_id in cell_ids
    ])
    n_groups = sum(int(block["n_groups"]) for block in check["link_blocks"])
    statuses = {
        method_id: (
            "OK_FALLBACK"
            if any(
                context.records_by_pair[(cell_id, method_id)]["status"] == "OK_FALLBACK"
                for cell_id in cell_ids
            )
            else "OK"
        )
        for method_id in PRIMARY_METHOD_IDS
    }
    score_hashes = {
        method_id: _payload_sha256([
            {
                "cell_id": cell_id,
                "score_sha256": context.records_by_pair[(cell_id, method_id)]["score_sha256"],
            }
            for cell_id in cell_ids
        ])
        for method_id in PRIMARY_METHOD_IDS
    }
    # population_grouped_paired_bootstrap canonicalizes its input mapping to
    # lexicographic cell order before taking the equal-cell mean.  Preserve
    # that arithmetic order exactly; a different summation order can differ by
    # one floating-point ULP even though the roster is the same.
    point_cell_ids = tuple(sorted(cell_ids))
    points = {
        method_id: {
            metric_id: float(np.mean([
                point_metrics[cell_id][method_id][metric_id]
                for cell_id in point_cell_ids
            ]))
            for metric_id in METRIC_IDS
        }
        for method_id in PRIMARY_METHOD_IDS
    }
    return {
        "specs": specs,
        "cell_ids": cell_ids,
        "aggregate": aggregate,
        "dataset_id": datasets[0] if len(datasets) == 1 else "__multiple__",
        "model_id": models[0] if len(models) == 1 else "__multiple__",
        "panel_role": (
            specs[0].panel_role
            if len({cell.panel_role for cell in specs}) == 1 else "mixed"
        ),
        "cohort_id": cohort_id,
        "label_sha256": label_sha,
        "n_groups": n_groups,
        "statuses": statuses,
        "score_hashes": score_hashes,
        "points": points,
    }


def _validate_population_rows(
    *, metric_rows: Sequence[Mapping[str, str]],
    contrast_rows: Sequence[Mapping[str, str]], registry: ExternalRegistry,
    context: _BuildContext, labels: Mapping[str, _LabelState],
    point_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    population_checks: Sequence[Mapping[str, Any]],
) -> None:
    enabled = [
        str(check["population_id"])
        for check in population_checks if check.get("status") == "OK_AGGREGATED"
    ]
    methods = tuple(sorted(PRIMARY_METHOD_IDS))
    candidates = tuple(method for method in methods if method != "iu_pcr")
    expected_metric_order = [
        (population_id, method_id, metric_id)
        for population_id in enabled for method_id in methods for metric_id in METRIC_IDS
    ]
    observed_metric_order = [
        (str(row["population_id"]), str(row["method_id"]), str(row["metric_id"]))
        for row in metric_rows if row.get("record_level") == "population"
    ]
    if observed_metric_order != expected_metric_order:
        raise RuntimeError("population metrics are not the exact ordered population x method x metric roster")
    expected_contrast_order = [
        (population_id, method_id, metric_id)
        for population_id in enabled for method_id in candidates for metric_id in METRIC_IDS
    ]
    observed_contrast_order = [
        (str(row["population_id"]), str(row["method_id"]), str(row["metric_id"]))
        for row in contrast_rows if row.get("record_level") == "population"
    ]
    if observed_contrast_order != expected_contrast_order:
        raise RuntimeError("population contrasts are not the exact ordered Cartesian roster")
    metric_by_key = {
        (str(row["population_id"]), str(row["method_id"]), str(row["metric_id"])): row
        for row in metric_rows if row.get("record_level") == "population"
    }
    contrast_by_key = {
        (str(row["population_id"]), str(row["method_id"]), str(row["metric_id"])): row
        for row in contrast_rows if row.get("record_level") == "population"
    }
    checks = {str(check["population_id"]): check for check in population_checks}
    for population_id in enabled:
        check = checks[population_id]
        metadata = _population_metadata(
            population_id=population_id, registry=registry, labels=labels,
            point_metrics=point_metrics, context=context, check=check,
        )
        aggregate = metadata["aggregate"]
        observed = check["observed"]
        valid_draws: set[int] = set()
        linked = any(bool(block["linked"]) for block in check["link_blocks"])
        stratified = aggregate.get("bootstrap") == "source_group_stratified_by_label"
        for method_id in methods:
            for metric_id in METRIC_IDS:
                row = metric_by_key[(population_id, method_id, metric_id)]
                where = f"population metric {population_id}/{method_id}/{metric_id}"
                expected_text = {
                    "comparison_group_id": _comparison_group_id(
                        level="population", cell_id=None,
                        population_id=population_id,
                        cohort_id=metadata["cohort_id"], metric_id=metric_id,
                        aggregate=aggregate,
                    ),
                    "panel_role": metadata["panel_role"],
                    "population_id": population_id,
                    "cell_id": "__population__",
                    "dataset_id": metadata["dataset_id"],
                    "model_id": metadata["model_id"],
                    "slice_id": f"population::{aggregate['interpretation']}",
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "status": metadata["statuses"][method_id],
                    "bootstrap_unit": check["bootstrap_unit"],
                    "cohort_id": metadata["cohort_id"],
                    "score_sha256": metadata["score_hashes"][method_id],
                    "label_sha256": metadata["label_sha256"],
                    "record_level": "population",
                    "aggregate_weighting": str(aggregate["weighting"]),
                    "aggregate_interpretation": str(aggregate["interpretation"]),
                    "linked_resampling": _bool_text(linked),
                    "stratified_by_label": _bool_text(stratified),
                }
                for key, expected in expected_text.items():
                    _require_text(row, key, expected, where=where)
                expected_ints = {
                    "n": int(observed["rows"]),
                    "n_incorrect": int(observed["incorrect"]),
                    "n_correct": int(observed["correct"]),
                    "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                    "n_cells": int(observed["cells"]),
                    "n_groups": int(metadata["n_groups"]),
                }
                for key, expected in expected_ints.items():
                    if _int_field(row, key, where=where) != expected:
                        raise RuntimeError(f"{where}: {key} drifted")
                valid = _int_field(row, "bootstrap_valid_draws", where=where)
                if not (0 < valid <= DEFAULT_BOOTSTRAP_DRAWS):
                    raise RuntimeError(f"{where}: bootstrap valid-draw count is invalid")
                valid_draws.add(valid)
                value, _, _ = _validate_numeric_interval(
                    row, metric_id=metric_id, contrast=False, where=where,
                )
                if value != metadata["points"][method_id][metric_id]:
                    raise RuntimeError(f"{where}: equal-cell point estimate drifted")
        if len(valid_draws) != 1:
            raise RuntimeError(f"{population_id}: paired bootstrap valid-draw counts diverged")
        for method_id in candidates:
            expected_status = (
                "OK_FALLBACK"
                if "OK_FALLBACK" in {
                    metadata["statuses"][method_id], metadata["statuses"]["iu_pcr"]
                }
                else "OK"
            )
            for metric_id in METRIC_IDS:
                row = contrast_by_key[(population_id, method_id, metric_id)]
                where = f"population contrast {population_id}/{method_id}/{metric_id}"
                expected_text = {
                    "comparison_group_id": _comparison_group_id(
                        level="population", cell_id=None,
                        population_id=population_id,
                        cohort_id=metadata["cohort_id"], metric_id=metric_id,
                        aggregate=aggregate,
                    ),
                    "panel_role": metadata["panel_role"],
                    "population_id": population_id,
                    "cell_id": "__population__",
                    "dataset_id": metadata["dataset_id"],
                    "model_id": metadata["model_id"],
                    "slice_id": f"population::{aggregate['interpretation']}",
                    "method_id": method_id,
                    "reference_method_id": "iu_pcr",
                    "metric_id": metric_id,
                    "higher_is_better": _bool_text(metric_id != "aurc_x1000"),
                    "bootstrap_unit": check["bootstrap_unit"],
                    "cohort_id": metadata["cohort_id"],
                    "record_level": "population",
                    "aggregate_weighting": str(aggregate["weighting"]),
                    "aggregate_interpretation": str(aggregate["interpretation"]),
                    "linked_resampling": _bool_text(linked),
                    "stratified_by_label": _bool_text(stratified),
                    "status": expected_status,
                }
                for key, expected in expected_text.items():
                    _require_text(row, key, expected, where=where)
                expected_ints = {
                    "n": int(observed["rows"]),
                    "n_groups": int(metadata["n_groups"]),
                    "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
                    "n_cells": int(observed["cells"]),
                }
                for key, expected in expected_ints.items():
                    if _int_field(row, key, where=where) != expected:
                        raise RuntimeError(f"{where}: {key} drifted")
                if _int_field(row, "bootstrap_valid_draws", where=where) not in valid_draws:
                    raise RuntimeError(f"{where}: contrast draw count is not paired")
                delta, _, _ = _validate_numeric_interval(
                    row, metric_id=metric_id, contrast=True, where=where,
                )
                expected_delta = (
                    metadata["points"][method_id][metric_id]
                    - metadata["points"]["iu_pcr"][metric_id]
                )
                if delta != expected_delta:
                    raise RuntimeError(f"{where}: point delta differs from metric table")
                probability = _float_field(
                    row, "probability_delta_le_zero", where=where,
                )
                if not 0.0 <= probability <= 1.0:
                    raise RuntimeError(f"{where}: bootstrap probability is out of range")


def _normalize_manifest(
    manifest: Mapping[str, Any], *, context: _BuildContext,
) -> bytes:
    """Canonicalize only fields proven to be build-bound before comparison."""

    value = deepcopy(dict(manifest))
    if value.get("build_id") != context.build_id:
        raise RuntimeError("evaluation manifest build_id cannot be normalized")
    if value.get("score_freeze_sha256") != sha256_file(
        context.root / "fit/SCORE_FREEZE_MANIFEST.json"
    ):
        raise RuntimeError("evaluation manifest score-freeze file hash cannot be normalized")
    if value.get("score_freeze_payload_sha256") != context.freeze["payload_sha256"]:
        raise RuntimeError("evaluation manifest score-freeze payload cannot be normalized")
    value["build_id"] = "<BUILD>"
    value["score_freeze_sha256"] = "<BUILD_SCORE_FREEZE_FILE_SHA256>"
    value["score_freeze_payload_sha256"] = "<BUILD_SCORE_FREEZE_PAYLOAD_SHA256>"
    value["payload_sha256"] = "<BUILD_MANIFEST_PAYLOAD_SHA256>"
    records = value.get("label_records")
    if not isinstance(records, list):
        raise RuntimeError("evaluation manifest label records cannot be normalized")
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "score_freeze_payload_sha256"
        ) != context.freeze["payload_sha256"]:
            raise RuntimeError("label provenance freeze payload cannot be normalized")
        provenance["score_freeze_payload_sha256"] = (
            "<BUILD_SCORE_FREEZE_PAYLOAD_SHA256>"
        )
    return canonical_json_bytes(value)


def _verify_evaluation_build(
    *, release_id: str, repo: Path, certificate_path: Path,
    registry: ExternalRegistry, score_certificate: Mapping[str, Any],
    context: _BuildContext, identity_key: bytes,
    label_loader: Callable[..., LabelVector] = load_labels_after_score_freeze,
) -> dict[str, Any]:
    evaluation_root = context.root / "evaluation"
    if evaluation_root.is_symlink() or not evaluation_root.is_dir():
        raise RuntimeError(f"build {context.build_id}: complete evaluation directory is absent")
    manifest_path = evaluation_root / "MANIFEST.json"
    manifest = _read_hashed_json(
        manifest_path, description=f"build {context.build_id} evaluation manifest",
    )
    if set(manifest) != EVALUATION_MANIFEST_FIELDS:
        raise RuntimeError(f"build {context.build_id}: evaluation manifest field roster drifted")
    full_identity = validate_public_identity_contract(
        context.preparation_manifest.get("identity_contract", {})
    )
    required = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "release_id": release_id,
        "build_id": context.build_id,
        "scientific_full": True,
        "ab_verification_status": "PASS",
        "ab_certificate_path": str(certificate_path.resolve()),
        "ab_certificate_sha256": score_certificate["certificate_sha256"],
        "ab_certificate_file_sha256": sha256_file(certificate_path),
        "score_freeze_sha256": sha256_file(
            context.root / "fit/SCORE_FREEZE_MANIFEST.json"
        ),
        "score_freeze_payload_sha256": context.freeze["payload_sha256"],
        "external_registry_sha256": registry.sha256,
        "identity_contract": full_identity,
        "id_contract_version": ID_CONTRACT_VERSION,
        "labels_opened_only_after_score_freeze": True,
        "score_semantics": "higher_is_incorrect",
        "positive_class": "incorrect",
        "metric_intervals": (
            "registered per-cell and population grouped paired source-level bootstrap"
        ),
        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "metrics_path": "metrics_long.csv",
        "contrasts_path": "contrasts_long.csv",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"build {context.build_id}: evaluation manifest {key} drifted")
    snapshot = manifest.get("evaluation_source_snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != {"files", "snapshot_sha256"}:
        raise RuntimeError(f"build {context.build_id}: evaluation source snapshot is absent")
    source_rows = snapshot.get("files", ())
    if any(not isinstance(row, Mapping) or set(row) != {"path", "sha256"} for row in source_rows):
        raise RuntimeError(f"build {context.build_id}: evaluation source records drifted")
    if [str(row.get("path", "")) for row in source_rows] != list(
        EVALUATION_SOURCE_FILES
    ):
        raise RuntimeError(f"build {context.build_id}: evaluation source roster/order drifted")
    verify_current_source_snapshot(
        snapshot, repo=repo, required_paths=EVALUATION_SOURCE_FILES,
        name=f"external evaluation build {context.build_id}",
    )
    if manifest.get("evaluation_source_snapshot_sha256") != snapshot.get(
        "snapshot_sha256"
    ):
        raise RuntimeError(f"build {context.build_id}: evaluation source hash binding drifted")
    source_root_value = manifest.get("source_root")
    certified_source_root_value = context.preparation_manifest.get("source_root")
    if (
        not isinstance(source_root_value, str)
        or source_root_value != certified_source_root_value
    ):
        raise RuntimeError(
            f"build {context.build_id}: evaluation source root differs from "
            "certified preparation source root"
        )
    source_root = Path(source_root_value)
    if (
        not source_root.is_absolute()
        or source_root.is_symlink()
        or not source_root.is_dir()
    ):
        raise RuntimeError(f"build {context.build_id}: evaluation source root is absent or unsafe")

    expected_applicability = [
        {
            "cell_id": str(row["cell_id"]),
            "status": str(row["status"]),
            "reason": row.get("reason"),
        }
        for row in context.preparation_manifest["cells"]
    ]
    if manifest.get("applicability_statuses") != expected_applicability:
        raise RuntimeError(f"build {context.build_id}: evaluation applicability roster drifted")

    labels, label_hashes, label_bytes = _load_label_states(
        evaluation_root=evaluation_root, manifest=manifest, context=context,
        registry=registry, score_certificate=score_certificate,
        identity_key=identity_key, source_root=source_root,
        label_loader=label_loader,
    )
    expected_population_checks = _expected_population_checks(
        registry=registry, context=context, labels=labels,
    )
    if manifest.get("population_checks") != expected_population_checks:
        raise RuntimeError(f"build {context.build_id}: population roster/audit drifted")

    metrics_path = _safe_regular_file(
        evaluation_root, "metrics_long.csv",
        description=f"build {context.build_id} metrics",
    )
    contrasts_path = _safe_regular_file(
        evaluation_root, "contrasts_long.csv",
        description=f"build {context.build_id} contrasts",
    )
    if (
        sha256_file(metrics_path) != manifest.get("metrics_sha256")
        or sha256_file(contrasts_path) != manifest.get("contrasts_sha256")
    ):
        raise RuntimeError(f"build {context.build_id}: evaluation table file hash failed")
    metric_rows, metric_bytes = _read_csv_exact(
        metrics_path, fields=METRIC_FIELDS,
        description=f"build {context.build_id} metrics",
    )
    contrast_rows, contrast_bytes = _read_csv_exact(
        contrasts_path, fields=CONTRAST_FIELDS,
        description=f"build {context.build_id} contrasts",
    )
    if (
        len(metric_rows) != int(manifest.get("n_metric_rows", -1))
        or len(contrast_rows) != int(manifest.get("n_contrast_rows", -1))
    ):
        raise RuntimeError(f"build {context.build_id}: evaluation table count binding failed")
    if any(row.get("record_level") not in {"cell", "population"} for row in metric_rows):
        raise RuntimeError(f"build {context.build_id}: metrics contain an unknown record level")
    if any(row.get("record_level") not in {"cell", "population"} for row in contrast_rows):
        raise RuntimeError(f"build {context.build_id}: contrasts contain an unknown record level")
    _, point_metrics = _load_score_values(context=context, labels=labels)
    _validate_cell_rows(
        metric_rows=metric_rows, contrast_rows=contrast_rows,
        registry=registry, context=context, labels=labels,
        point_metrics=point_metrics,
    )
    _validate_population_rows(
        metric_rows=metric_rows, contrast_rows=contrast_rows,
        registry=registry, context=context, labels=labels,
        point_metrics=point_metrics,
        population_checks=expected_population_checks,
    )

    expected_files = {
        "MANIFEST.json", "metrics_long.csv", "contrasts_long.csv", *label_hashes,
    }
    observed_files = {
        path.relative_to(evaluation_root).as_posix()
        for path in evaluation_root.rglob("*") if path.is_file()
    }
    symlinks = [path for path in evaluation_root.rglob("*") if path.is_symlink()]
    if observed_files != expected_files or symlinks:
        raise RuntimeError(f"build {context.build_id}: evaluation tree has missing/unregistered artifacts")
    return {
        "manifest": manifest,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "normalized_manifest_bytes": _normalize_manifest(manifest, context=context),
        "metrics_bytes": metric_bytes,
        "contrasts_bytes": contrast_bytes,
        "metrics_sha256": sha256_bytes(metric_bytes),
        "contrasts_sha256": sha256_bytes(contrast_bytes),
        "label_hashes": label_hashes,
        "label_bytes": label_bytes,
        "tree": canonical_tree_manifest(evaluation_root),
        "n_metric_rows": len(metric_rows),
        "n_contrast_rows": len(contrast_rows),
        "population_checks": expected_population_checks,
    }


def _require_exact_ab_identity(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> None:
    """Reject any A/B difference outside the enumerated manifest bindings."""

    if left["metrics_bytes"] != right["metrics_bytes"]:
        raise RuntimeError("external Evaluation A/B metrics CSVs are not byte-identical")
    if left["contrasts_bytes"] != right["contrasts_bytes"]:
        raise RuntimeError("external Evaluation A/B contrasts CSVs are not byte-identical")
    if left["label_bytes"] != right["label_bytes"]:
        raise RuntimeError("external Evaluation A/B label artifacts are not byte-identical")
    if left["label_hashes"] != right["label_hashes"]:
        raise RuntimeError("external Evaluation A/B label artifact hashes differ")
    if left["normalized_manifest_bytes"] != right["normalized_manifest_bytes"]:
        raise RuntimeError(
            "external Evaluation A/B manifests differ outside the explicit build normalization contract"
        )
    if left["population_checks"] != right["population_checks"]:
        raise RuntimeError("external Evaluation A/B population rosters differ")


def verify_external_evaluation_ab(
    *, release_id: str, release_root: str | Path,
    registry_path: str | Path,
    population_registry_path: str | Path,
    repo: str | Path,
    score_certificate_path: str | Path | None = None,
    identity_key_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Issue one fail-closed PASS certificate for exact Evaluation A/B output."""

    repo_path = Path(repo).resolve()
    release_root_path = Path(release_root).resolve()
    release = release_root_path / release_id
    registry = load_external_registry(
        repo=repo_path, registry_path=registry_path,
        population_registry_path=population_registry_path,
    )
    certificate_path = (
        Path(score_certificate_path).resolve()
        if score_certificate_path is not None
        else release / "external_final_answer/AB_VERIFICATION.json"
    )
    score_certificate = assert_external_ab_certificate(
        certificate_path, release_id=release_id,
        release_root=release_root_path, selected_build="A",
        registry=registry, repo=repo_path,
    )
    controller_root = (
        release_root_path.parent / "private_control" / release_id
        / "external_final_answer"
    )
    identity_key = load_identity_key(
        identity_key_path or (controller_root / "external-id-v2.key")
    )
    if (
        score_certificate.get("status") != "PASS"
        or score_certificate.get("scientific_full") is not True
        or tuple(map(str, score_certificate.get("method_ids", ())))
        != PRIMARY_METHOD_IDS
    ):
        raise RuntimeError("external score A/B certificate is not a full exact-13 PASS")
    contexts = {
        build_id: _validate_input_context(
            release=release, release_root=release_root_path,
            release_id=release_id, build_id=build_id, registry=registry,
            score_certificate=score_certificate,
        )
        for build_id in ("A", "B")
    }
    prep_projection = lambda context: [  # noqa: E731 - compact immutable view
        {
            "cell_id": row["cell_id"], "status": row["status"],
            "reason": row.get("reason"),
        }
        for row in context.preparation_manifest["cells"]
    ]
    if prep_projection(contexts["A"]) != prep_projection(contexts["B"]):
        raise RuntimeError("external Evaluation A/B preparation/applicability rosters differ")

    audits = {
        build_id: _verify_evaluation_build(
            release_id=release_id, repo=repo_path,
            certificate_path=certificate_path, registry=registry,
            score_certificate=score_certificate, context=contexts[build_id],
            identity_key=identity_key,
        )
        for build_id in ("A", "B")
    }
    left, right = audits["A"], audits["B"]
    _require_exact_ab_identity(left, right)

    verification_snapshot = _source_snapshot(repo_path, VERIFICATION_SOURCE_FILES)
    full_identity = validate_public_identity_contract(
        contexts["A"].preparation_manifest["identity_contract"]
    )
    fit_identity = validate_fit_row_identity_contract(
        score_certificate["identity_contract"]
    )
    certificate: dict[str, Any] = {
        "schema_version": EVALUATION_AB_SCHEMA_VERSION,
        "release_id": release_id,
        "lane_id": "external_final_answer",
        "status": "PASS",
        "scientific_full": True,
        "score_ab_certificate_sha256": score_certificate["certificate_sha256"],
        "score_ab_certificate_file_sha256": sha256_file(certificate_path),
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "method_ids": list(PRIMARY_METHOD_IDS),
        "cell_ids": list(map(str, score_certificate["cell_ids"])),
        "metric_ids": list(METRIC_IDS),
        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "positive_class": "incorrect",
        "score_semantics": "higher_is_incorrect",
        "label_rederivation": {
            "status": "PASS",
            "registry_adapters_reexecuted": True,
            "pinned_source_hashes_revalidated": True,
            "joined_to_signed_score_row_rosters": True,
            "coordinated_ab_label_permutation_rejected": True,
            "identity_key_id": full_identity["key_id"],
        },
        "identity_join": {
            "full_postfreeze_contract_sha256": full_identity["contract_sha256"],
            "fit_row_contract_sha256": fit_identity["contract_sha256"],
            "shared_identity_key_id": full_identity["key_id"],
            "contracts_are_distinct": (
                full_identity["contract_sha256"] != fit_identity["contract_sha256"]
            ),
            "prepared_score_row_roster_contract": (
                "reconstruction-external-fit-row-roster-v3"
            ),
            "label_row_roster_contract": (
                "reconstruction-external-fit-row-roster-v2"
            ),
            "label_row_identity_scope": "full_postfreeze_contract",
            "join_enforced_by": [
                "exact_opaque_row_ids_and_order",
                "shared_row_namespace_sha256",
                "shared_identity_key_id",
                "sealed_full_group_roster_commitment",
            ],
        },
        "normalization_contract": MANIFEST_NORMALIZATION_CONTRACT,
        "normalized_manifest_sha256": sha256_bytes(
            left["normalized_manifest_bytes"]
        ),
        "byte_identity": {
            "metrics_long.csv": left["metrics_sha256"],
            "contrasts_long.csv": left["contrasts_sha256"],
            "label_artifacts": left["label_hashes"],
        },
        "n_metric_rows": left["n_metric_rows"],
        "n_contrast_rows": left["n_contrast_rows"],
        "population_checks": left["population_checks"],
        "verification_source_snapshot": verification_snapshot,
        "builds": {
            build_id: {
                "evaluation_manifest_file_sha256": audits[build_id][
                    "manifest_file_sha256"
                ],
                "evaluation_manifest_payload_sha256": audits[build_id][
                    "manifest_payload_sha256"
                ],
                "evaluation_tree_sha256": audits[build_id]["tree"]["tree_sha256"],
                "score_freeze_sha256": sha256_file(
                    contexts[build_id].root / "fit/SCORE_FREEZE_MANIFEST.json"
                ),
                "score_freeze_payload_sha256": contexts[build_id].freeze[
                    "payload_sha256"
                ],
            }
            for build_id in ("A", "B")
        },
    }
    certificate["certificate_sha256"] = _payload_sha256(certificate)
    target = (
        Path(os.path.abspath(os.fspath(output_path)))
        if output_path is not None
        else release / "external_final_answer/EVALUATION_AB_VERIFICATION.json"
    )
    payload = canonical_json_bytes(certificate) + b"\n"
    _write_immutable_certificate(target, payload)
    return certificate


__all__ = [
    "CONTRAST_FIELDS",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "EVALUATION_AB_SCHEMA_VERSION",
    "EVALUATION_MANIFEST_FIELDS",
    "EVALUATION_SCHEMA_VERSION",
    "EVALUATION_SOURCE_FILES",
    "MANIFEST_NORMALIZATION_CONTRACT",
    "METRIC_FIELDS",
    "VERIFICATION_SOURCE_FILES",
    "verify_external_evaluation_ab",
]
