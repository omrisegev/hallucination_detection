"""Strict downstream amendment for the frozen localization score release.

The score registry and score A/B certificate are immutable.  This module opens
the disclosed, label-bearing amendment only after that certificate has been
fully reverified, validates its exact PRMBench corpus audit, and creates an
in-memory evaluation configuration.  It never mutates score artifacts or the
score-bound registry on disk.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import sha256_file
from .localization_contract import payload_sha256


AMENDMENT_SCHEMA_VERSION = "reconstruction-localization-postfreeze-amendment-v1"
EXPECTED_AMENDMENT_FILE_SHA256 = (
    "507ca8fda499b224c53b4f95dc4f83a37c960b621a328e7d580a41d06a0558b1"
)
EXPECTED_AMENDMENT_PAYLOAD_SHA256 = (
    "6874cc8c89583760fcf8a8e7f1600a5322b91140bc140333bfd6bfd104cd84ca"
)
EXPECTED_RELEASE_ID = "2026-08-24_localization_v1"
EXPECTED_LOCALIZATION_REGISTRY_SHA256 = (
    "611b609abf328df649d66330e4a8aadd919f240be264d3f6d7e04d205bf08f19"
)
EXPECTED_SCORE_AB_CERTIFICATE_SHA256 = (
    "79a9301808d9c15fa4cd1d18d023557823061526f0092590e3c92d9084610e9a"
)
EXPECTED_SCORE_AB_CERTIFICATE_FILE_SHA256 = (
    "779a7f6e614b03197a7f9bb8a8d3da7c67f53f42a834821b49719aeaac5a2631"
)
EXPECTED_SCORE_VERIFIER_GIT_HEAD = "d96efdcf112fb46241f0a6594da00ef26cd079c3"
EXPECTED_TELEMETRY_SHA256 = (
    "b934afad0889ffacf0f4420f885ad52ddcf08f2124e506a21cd24f216bd170be"
)
EXPECTED_TELEMETRY_MANIFEST_SHA256 = (
    "c82955e81cf93939de950b0a567a829290e10957273d2cd0724aed2a6bf9a452"
)
EXPECTED_OOB_RECORDS_SHA256 = (
    "ed43a0eb9c64095e2214bf8d69f528a5c5ecda4e8a086139179a2f50b2af99a3"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT = (
    REPO_ROOT
    / "configs/reconstruction_benchmark_v1/localization_postfreeze_amendment_v1.json"
)

_TOP_LEVEL_FIELDS = {
    "schema_version", "amendment_id", "release_id", "stage", "reason",
    "original_localization_registry", "score_ab_certificate",
    "score_verifier_repo", "telemetry", "semantics", "oob_audit",
    "original_prmbench_counts", "effective_prmbench_counts", "disclosure",
    "payload_sha256",
}
_COUNT_FIELDS = {
    "expected_error_responses", "expected_steps", "expected_positive_steps",
    "expected_by_family",
}
_RECORD_FIELDS = {"idx", "family", "n_steps", "error_steps", "invalid"}
_PRMBENCH_FAMILIES = (
    "circular", "confidence", "counterfactual", "deception",
    "domain_inconsistency", "missing_condition", "multi_solutions",
    "redundency", "step_contradiction",
)
_EXPECTED_ORIGINAL_COUNTS = {
    "expected_error_responses": 6_208,
    "expected_steps": 83_280,
    "expected_positive_steps": 13_295,
    "expected_by_family": {
        "circular": {"responses": 758, "steps": 7_813, "positive_steps": 2_149},
        "confidence": {"responses": 756, "steps": 10_711, "positive_steps": 1_320},
        "counterfactual": {"responses": 757, "steps": 10_477, "positive_steps": 2_092},
        "deception": {"responses": 749, "steps": 10_053, "positive_steps": 1_728},
        "domain_inconsistency": {
            "responses": 757, "steps": 10_031, "positive_steps": 1_341,
        },
        "missing_condition": {
            "responses": 756, "steps": 9_613, "positive_steps": 1_925,
        },
        "multi_solutions": {"responses": 160, "steps": 2_241, "positive_steps": 0},
        "redundency": {"responses": 758, "steps": 11_613, "positive_steps": 1_546},
        "step_contradiction": {
            "responses": 757, "steps": 10_728, "positive_steps": 1_194,
        },
    },
}
_EXPECTED_EFFECTIVE_COUNTS = {
    "expected_error_responses": 6_208,
    "expected_steps": 83_280,
    "expected_positive_steps": 13_144,
    "expected_by_family": {
        "circular": {"responses": 758, "steps": 7_813, "positive_steps": 2_135},
        "confidence": {"responses": 756, "steps": 10_711, "positive_steps": 1_303},
        "counterfactual": {"responses": 757, "steps": 10_477, "positive_steps": 2_074},
        "deception": {"responses": 749, "steps": 10_053, "positive_steps": 1_698},
        "domain_inconsistency": {
            "responses": 757, "steps": 10_031, "positive_steps": 1_338,
        },
        "missing_condition": {
            "responses": 756, "steps": 9_613, "positive_steps": 1_869,
        },
        "multi_solutions": {"responses": 160, "steps": 2_241, "positive_steps": 0},
        "redundency": {"responses": 758, "steps": 11_613, "positive_steps": 1_537},
        "step_contradiction": {
            "responses": 757, "steps": 10_728, "positive_steps": 1_190,
        },
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _as_exact_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"localization amendment integer field is malformed: {field}")
    return value


def _validate_counts(value: Any, *, field: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"localization amendment {field} is malformed")
    _require(set(value) == _COUNT_FIELDS, f"localization amendment {field} fields drifted")
    output = {
        key: _as_exact_int(value[key], field=f"{field}.{key}")
        for key in ("expected_error_responses", "expected_steps", "expected_positive_steps")
    }
    families = value["expected_by_family"]
    _require(isinstance(families, Mapping), f"localization amendment {field} families malformed")
    _require(
        tuple(families) == _PRMBENCH_FAMILIES,
        f"localization amendment {field} family roster/order drifted",
    )
    output["expected_by_family"] = {}
    for family, row in families.items():
        _require(isinstance(family, str) and family, "localization amendment family is malformed")
        _require(
            isinstance(row, Mapping) and set(row) == {"responses", "steps", "positive_steps"},
            f"localization amendment family counts drifted: {family}",
        )
        output["expected_by_family"][family] = {
            key: _as_exact_int(row[key], field=f"{field}.{family}.{key}")
            for key in ("responses", "steps", "positive_steps")
        }
    _require(
        sum(row["responses"] for row in output["expected_by_family"].values())
        == output["expected_error_responses"],
        f"localization amendment {field} response total drifted",
    )
    _require(
        sum(row["steps"] for row in output["expected_by_family"].values())
        == output["expected_steps"],
        f"localization amendment {field} step total drifted",
    )
    _require(
        sum(row["positive_steps"] for row in output["expected_by_family"].values())
        == output["expected_positive_steps"],
        f"localization amendment {field} positive total drifted",
    )
    return output


def _validate_oob_records(value: Any) -> tuple[dict[str, Any], ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "localization amendment OOB records are malformed",
    )
    records: list[dict[str, Any]] = []
    for raw in value:
        _require(
            isinstance(raw, Mapping) and set(raw) == _RECORD_FIELDS,
            "localization amendment OOB record fields drifted",
        )
        idx = raw["idx"]
        family = raw["family"]
        n_steps = _as_exact_int(raw["n_steps"], field="oob_audit.records.n_steps")
        error_steps = raw["error_steps"]
        invalid = raw["invalid"]
        _require(isinstance(idx, str) and idx, "localization amendment OOB idx is malformed")
        _require(isinstance(family, str) and family, "localization amendment OOB family malformed")
        _require(n_steps > 0, "localization amendment OOB n_steps must be positive")
        for name, items in (("error_steps", error_steps), ("invalid", invalid)):
            _require(
                isinstance(items, Sequence) and not isinstance(items, (str, bytes)),
                f"localization amendment OOB {name} is malformed",
            )
            _require(
                all(type(item) is int for item in items),
                f"localization amendment OOB {name} must contain exact integers",
            )
        error_values = list(error_steps)
        invalid_values = list(invalid)
        _require(error_values, "localization amendment OOB error_steps cannot be empty")
        _require(
            len(error_values) == len(set(error_values)),
            "localization amendment OOB annotations contain duplicates",
        )
        _require(
            all(item >= 1 for item in error_values),
            "localization amendment contains zero/negative PRMBench indices",
        )
        _require(
            invalid_values == [item for item in error_values if item > n_steps]
            and bool(invalid_values),
            "localization amendment OOB invalid values are not exact inert suffix indices",
        )
        records.append({
            "idx": idx, "family": family, "n_steps": n_steps,
            "error_steps": error_values, "invalid": invalid_values,
        })
    _require(
        records == sorted(records, key=lambda row: row["idx"]),
        "localization amendment OOB records are not canonically ordered",
    )
    _require(
        len({row["idx"] for row in records}) == len(records),
        "localization amendment OOB records contain duplicate IDs",
    )
    return tuple(records)


def load_localization_postfreeze_amendment(
    path: str | Path,
    *,
    release_id: str,
    localization_registry_path: str | Path,
    score_ab_certificate_path: str | Path,
    score_ab_certificate: Mapping[str, Any],
    source_root: str | Path,
) -> dict[str, Any]:
    """Load and authenticate the exact release-specific post-freeze amendment."""

    amendment_path = Path(path).resolve()
    _require(
        sha256_file(amendment_path) == EXPECTED_AMENDMENT_FILE_SHA256,
        "localization amendment file hash failed",
    )
    value = json.loads(amendment_path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), "localization amendment root is malformed")
    _require(set(value) == _TOP_LEVEL_FIELDS, "localization amendment top-level fields drifted")
    payload = dict(value)
    recorded_payload = payload.pop("payload_sha256", None)
    _require(
        recorded_payload == EXPECTED_AMENDMENT_PAYLOAD_SHA256
        and recorded_payload == payload_sha256(payload),
        "localization amendment payload hash failed",
    )
    _require(
        value["schema_version"] == AMENDMENT_SCHEMA_VERSION
        and value["release_id"] == EXPECTED_RELEASE_ID
        and value["release_id"] == release_id
        and value["stage"] == "post_score_ab_freeze_evaluation_only",
        "localization amendment does not apply to this release/stage",
    )

    original = value["original_localization_registry"]
    _require(
        isinstance(original, Mapping)
        and set(original) == {"path", "sha256"}
        and original["path"] == "configs/reconstruction_benchmark_v1/localization.json"
        and original["sha256"] == EXPECTED_LOCALIZATION_REGISTRY_SHA256
        and original["sha256"] == sha256_file(localization_registry_path),
        "localization amendment original registry binding failed",
    )
    certificate_binding = value["score_ab_certificate"]
    _require(
        isinstance(certificate_binding, Mapping)
        and set(certificate_binding)
        == {"path", "schema_version", "certificate_sha256", "file_sha256"}
        and certificate_binding["path"] == "localization/AB_VERIFICATION.json"
        and Path(score_ab_certificate_path).resolve().as_posix().endswith(
            f"/{release_id}/{certificate_binding['path']}"
        )
        and certificate_binding["schema_version"]
        == score_ab_certificate.get("schema_version")
        and certificate_binding["certificate_sha256"]
        == EXPECTED_SCORE_AB_CERTIFICATE_SHA256
        and certificate_binding["certificate_sha256"]
        == score_ab_certificate.get("certificate_sha256")
        and certificate_binding["file_sha256"]
        == EXPECTED_SCORE_AB_CERTIFICATE_FILE_SHA256
        and certificate_binding["file_sha256"] == sha256_file(score_ab_certificate_path),
        "localization amendment score A/B certificate binding failed",
    )

    verifier = value["score_verifier_repo"]
    _require(
        isinstance(verifier, Mapping)
        and set(verifier) == {"required_git_head"}
        and verifier["required_git_head"] == EXPECTED_SCORE_VERIFIER_GIT_HEAD,
        "localization amendment score verifier binding is malformed",
    )
    telemetry = value["telemetry"]
    _require(
        isinstance(telemetry, Mapping)
        and set(telemetry)
        == {"path", "sha256", "manifest_path", "manifest_sha256"},
        "localization amendment telemetry binding is malformed",
    )
    source_root_path = Path(source_root).resolve()
    telemetry_path = (source_root_path / str(telemetry["path"])).resolve()
    manifest_path = (source_root_path / str(telemetry["manifest_path"])).resolve()
    for candidate in (telemetry_path, manifest_path):
        try:
            candidate.relative_to(source_root_path)
        except ValueError as error:
            raise RuntimeError("localization amendment telemetry path escaped source root") from error
    _require(
        telemetry["sha256"] == EXPECTED_TELEMETRY_SHA256
        and telemetry["manifest_sha256"] == EXPECTED_TELEMETRY_MANIFEST_SHA256
        and sha256_file(telemetry_path) == telemetry["sha256"]
        and sha256_file(manifest_path) == telemetry["manifest_sha256"],
        "localization amendment telemetry binding failed",
    )

    semantics = value["semantics"]
    _require(
        semantics == {
            "index_base": 1,
            "out_of_range_behavior": "inert_upstream_membership_only",
            "rows_dropped": 0,
            "indices_shifted": 0,
            "indices_clamped": 0,
            "indices_repaired": 0,
        },
        "localization amendment PRMBench semantics drifted",
    )
    audit = value["oob_audit"]
    _require(isinstance(audit, Mapping), "localization amendment OOB audit is malformed")
    records = _validate_oob_records(audit.get("records"))
    _require(
        isinstance(audit, Mapping)
        and set(audit)
        == {
            "row_count", "annotation_count", "all_annotation_count", "minimum_annotation",
            "zero_count", "negative_count", "duplicate_annotation_rows",
            "records_sha256", "records",
        }
        and _as_exact_int(audit["row_count"], field="oob_audit.row_count") == len(records)
        and _as_exact_int(audit["annotation_count"], field="oob_audit.annotation_count")
        == sum(len(row["invalid"]) for row in records)
        and _as_exact_int(audit["all_annotation_count"], field="oob_audit.all_annotation_count")
        == 13_295
        and _as_exact_int(audit["minimum_annotation"], field="oob_audit.minimum_annotation") == 1
        and _as_exact_int(audit["zero_count"], field="oob_audit.zero_count") == 0
        and _as_exact_int(audit["negative_count"], field="oob_audit.negative_count") == 0
        and _as_exact_int(
            audit["duplicate_annotation_rows"], field="oob_audit.duplicate_annotation_rows"
        ) == 0
        and audit["records_sha256"] == EXPECTED_OOB_RECORDS_SHA256
        and audit["records_sha256"] == payload_sha256(list(records)),
        "localization amendment exact OOB audit failed",
    )
    _require(
        len(records) == 100 and sum(len(row["invalid"]) for row in records) == 151,
        "localization amendment exact 100-row/151-index contract drifted",
    )

    original_counts = _validate_counts(
        value["original_prmbench_counts"], field="original_prmbench_counts"
    )
    effective_counts = _validate_counts(
        value["effective_prmbench_counts"], field="effective_prmbench_counts"
    )
    _require(
        original_counts == _EXPECTED_ORIGINAL_COUNTS
        and effective_counts == _EXPECTED_EFFECTIVE_COUNTS,
        "localization amendment PRMBench counts differ from the audited correction",
    )
    registry = json.loads(Path(localization_registry_path).read_text(encoding="utf-8"))
    registered = registry.get("prmbench", {})
    observed_original = {
        "expected_error_responses": registered.get("expected_error_responses"),
        "expected_steps": registered.get("expected_steps"),
        "expected_positive_steps": registered.get("expected_positive_steps"),
        "expected_by_family": registered.get("expected_by_family"),
    }
    _require(
        observed_original == original_counts,
        "localization amendment no longer matches the score-bound PRMBench counts",
    )
    _require(
        original_counts["expected_positive_steps"]
        - effective_counts["expected_positive_steps"]
        == audit["annotation_count"],
        "localization amendment positive-count delta does not equal inert annotations",
    )
    _require(
        original_counts["expected_error_responses"]
        == effective_counts["expected_error_responses"]
        and original_counts["expected_steps"] == effective_counts["expected_steps"],
        "localization amendment changed the response/step population",
    )

    output = dict(value)
    output["path"] = str(amendment_path)
    output["file_sha256"] = sha256_file(amendment_path)
    output["oob_audit"] = {**dict(audit), "records": list(records)}
    output["original_prmbench_counts"] = original_counts
    output["effective_prmbench_counts"] = effective_counts
    return output


def apply_localization_postfreeze_amendment(
    config: Mapping[str, Any], amendment: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an evaluation-only config with only the disclosed counts changed."""

    output = deepcopy(dict(config))
    prmbench = output.get("prmbench")
    _require(isinstance(prmbench, dict), "localization PRMBench registry is malformed")
    original = amendment["original_prmbench_counts"]
    effective = amendment["effective_prmbench_counts"]
    observed = {
        "expected_error_responses": prmbench.get("expected_error_responses"),
        "expected_steps": prmbench.get("expected_steps"),
        "expected_positive_steps": prmbench.get("expected_positive_steps"),
        "expected_by_family": prmbench.get("expected_by_family"),
    }
    _require(observed == original, "localization amendment application boundary drifted")
    prmbench["expected_error_responses"] = effective["expected_error_responses"]
    prmbench["expected_steps"] = effective["expected_steps"]
    prmbench["expected_positive_steps"] = effective["expected_positive_steps"]
    prmbench["expected_by_family"] = deepcopy(effective["expected_by_family"])
    return output


def validate_observed_prmbench_oob_audit(
    observed_records: Sequence[Mapping[str, Any]], amendment: Mapping[str, Any], *,
    all_annotation_count: int, minimum_annotation: int,
    zero_count: int, negative_count: int, duplicate_annotation_rows: int,
) -> dict[str, Any]:
    """Require the opened target rows to equal the disclosed amendment byte-semantically."""

    observed = [dict(row) for row in observed_records]
    observed.sort(key=lambda row: str(row.get("idx", "")))
    expected = list(amendment["oob_audit"]["records"])
    _require(observed == expected, "PRMBench inert OOB audit differs from amendment")
    digest = payload_sha256(observed)
    _require(
        digest == amendment["oob_audit"]["records_sha256"],
        "PRMBench inert OOB audit hash differs from amendment",
    )
    observed_summary = {
        "all_annotation_count": _as_exact_int(
            all_annotation_count, field="observed.all_annotation_count"
        ),
        "minimum_annotation": _as_exact_int(
            minimum_annotation, field="observed.minimum_annotation"
        ),
        "zero_count": _as_exact_int(zero_count, field="observed.zero_count"),
        "negative_count": _as_exact_int(
            negative_count, field="observed.negative_count"
        ),
        "duplicate_annotation_rows": _as_exact_int(
            duplicate_annotation_rows, field="observed.duplicate_annotation_rows"
        ),
    }
    _require(
        observed_summary == {
            key: amendment["oob_audit"][key] for key in observed_summary
        },
        "PRMBench annotation-domain audit differs from amendment",
    )
    return {
        "schema_version": "reconstruction-localization-prmbench-oob-audit-v1",
        "amendment_id": amendment["amendment_id"],
        "amendment_file_sha256": amendment["file_sha256"],
        "records_sha256": digest,
        "row_count": len(observed),
        "annotation_count": sum(len(row["invalid"]) for row in observed),
        **observed_summary,
        "behavior": "one_based_out_of_range_indices_inert_rows_retained",
        "rows_dropped": 0,
        "indices_shifted": 0,
        "indices_clamped": 0,
        "indices_repaired": 0,
    }


__all__ = [
    "AMENDMENT_SCHEMA_VERSION", "DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT",
    "EXPECTED_AMENDMENT_FILE_SHA256", "EXPECTED_AMENDMENT_PAYLOAD_SHA256",
    "EXPECTED_SCORE_VERIFIER_GIT_HEAD",
    "apply_localization_postfreeze_amendment",
    "load_localization_postfreeze_amendment",
    "validate_observed_prmbench_oob_audit",
]
