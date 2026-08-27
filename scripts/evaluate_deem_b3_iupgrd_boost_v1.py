#!/usr/bin/env python3
"""Strict evaluation boundary for the frozen one-stage B3/IU-PGRD boost.

The evaluator performs two complete target-free passes before importing the
label-sidecar module:

1. contract/hash pass over the config, registry, run definition, completion,
   exact artifact roster, source closure, environment, bundles, and frozen B3
   members;
2. numerical/mechanical pass over every state, held-family calibration, and
   score artifact in the run (including artifacts outside a requested screen).

Only then are labels joined by row ID and correctness-positive AUROC/AUPRC
computed.  This makes ``--cells screen`` safe to use with an all-24-cell run:
the evaluation subset is eight cells, but the frozen donor fit is preflighted
in full.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_iupgrd_boost import (  # noqa: E402
    deterministic_row_permutation,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    environment_fingerprint,
    family_index_map,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    BUNDLE_SCHEMA,
    load_registry,
    load_target_free_bundle,
    registry_cell,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/deem_b3_iupgrd_boost_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
TOLERANCE = 5e-4

CONFIG_SCHEMA = "deem_b3_iupgrd_boost_v1_config"
RUN_SCHEMA = "deem_b3_iupgrd_boost_v1_run_definition"
MANIFEST_SCHEMA = "deem_b3_iupgrd_boost_v1_fit_artifact_manifest"
COMPLETE_SCHEMA = "deem_b3_iupgrd_boost_v1_fit_complete"
STATE_SCHEMA = "deem_b3_iupgrd_boost_v1_state"
CALIBRATION_SCHEMA = "deem_b3_iupgrd_boost_v1_calibration"
SCORE_SCHEMA = "deem_b3_iupgrd_boost_v1_fit"

EXPECTED_VARIANTS = (
    "E0_B3_EXACT_ALIAS",
    "E1_B3_ORTH_IUPGRD_FULL",
    "E2_B3_ORTH_IUPGRD_HALF",
    "E3_B3_UNPROJECTED_IUPGRD_FULL",
    "E4_B3_ORTH_FAMILY_PERMUTED_FULL",
    "E5_B3_ORTH_ROW_PERMUTED_FULL",
)

EXPECTED_VARIANT_CONFIGS = {
    "E0_B3_EXACT_ALIAS": {
        "trust_factor": 0.0,
        "project_against_b3": True,
        "family_axis_permuted": False,
        "row_permuted": False,
    },
    "E1_B3_ORTH_IUPGRD_FULL": {
        "trust_factor": 1.0,
        "project_against_b3": True,
        "family_axis_permuted": False,
        "row_permuted": False,
    },
    "E2_B3_ORTH_IUPGRD_HALF": {
        "trust_factor": 0.5,
        "project_against_b3": True,
        "family_axis_permuted": False,
        "row_permuted": False,
    },
    "E3_B3_UNPROJECTED_IUPGRD_FULL": {
        "trust_factor": 1.0,
        "project_against_b3": False,
        "family_axis_permuted": False,
        "row_permuted": False,
    },
    "E4_B3_ORTH_FAMILY_PERMUTED_FULL": {
        "trust_factor": 1.0,
        "project_against_b3": True,
        "family_axis_permuted": True,
        "row_permuted": False,
    },
    "E5_B3_ORTH_ROW_PERMUTED_FULL": {
        "trust_factor": 1.0,
        "project_against_b3": True,
        "family_axis_permuted": False,
        "row_permuted": True,
    },
}

STATE_ARRAYS = {
    "row_id",
    "feature_names",
    "family_order",
    "global_family_order",
    "risk_transform_mean",
    "risk_transform_scale",
    "risk_transform_constant_mask",
    "baseline_score",
    "baseline_logit",
    "baseline_z",
    "baseline_logit_mean",
    "baseline_logit_scale",
    "iu_score",
    "iu_score_aligned",
    "iu_orientation",
    "iu_weights",
    "raw_family_contributions",
    "standardized_family_contributions",
    "iu_family_residuals",
    "moment_A",
    "moment_c",
    "moment_presence",
    "iu_transform_baseline_mean",
    "iu_transform_baseline_scale",
    "iu_transform_contribution_mean",
    "iu_transform_contribution_scale",
    "iu_transform_baseline_loadings",
    "iu_transform_residual_mean",
    "iu_transform_residual_scale",
    "graph_data",
    "graph_indices",
    "graph_indptr",
    "graph_shape",
    "laplacian_data",
    "laplacian_indices",
    "laplacian_indptr",
    "laplacian_shape",
}

CALIBRATION_ARRAYS = {
    "direction",
    "donor_cells",
    "donor_dataset_families",
    "donor_moment_A",
    "donor_moment_c",
    "donor_moment_presence",
}

SCORE_ARRAYS = {
    "row_id",
    "score",
    "baseline_score",
    "logit",
    "baseline_logit",
    "correction_z",
    "raw_correction",
    "projected_correction",
    "projection_coefficients",
    "calibration_direction",
    "applied_direction",
    "family_axis_permutation",
    "row_permutation",
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualGraphDeemError(f"JSON object required: {path}")
    return value


def _verify_content_hash(value: Mapping[str, Any], *, context: str) -> str:
    payload = dict(value)
    expected = payload.pop("content_sha256", None)
    actual = canonical_sha256(payload)
    if not isinstance(expected, str) or expected != actual:
        raise ResidualGraphDeemError(f"content hash mismatch: {context}")
    return expected


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_logit(score: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(values) - np.log1p(-values)


def _max_abs(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _require_close(
    actual: Any,
    expected: Any,
    *,
    context: str,
    atol: float = 1e-10,
) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=atol):
        error = _max_abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        raise ResidualGraphDeemError(f"{context} mismatch (max_abs={error})")


def _load_config(path: Path) -> tuple[dict, dict[str, dict]]:
    value = _read_json(path)
    if value.get("schema") != CONFIG_SCHEMA:
        raise ResidualGraphDeemError("IU-PGRD boost config schema mismatch")
    rows = value.get("variants")
    if not isinstance(rows, list):
        raise ResidualGraphDeemError("variant roster is not a list")
    identifiers = tuple(str(row.get("id", "")) for row in rows)
    if identifiers != EXPECTED_VARIANTS:
        raise ResidualGraphDeemError("frozen E0--E5 roster/order changed")
    lookup = {str(row["id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ResidualGraphDeemError("duplicate IU-PGRD variant ID")
    for identifier, expected in EXPECTED_VARIANT_CONFIGS.items():
        if lookup[identifier].get("config") != expected:
            raise ResidualGraphDeemError(f"frozen variant config changed: {identifier}")
        if not isinstance(lookup[identifier].get("role"), str):
            raise ResidualGraphDeemError(f"variant role missing: {identifier}")
    if value.get("baseline", {}).get("seeds") != [0, 1, 2, 3, 4]:
        raise ResidualGraphDeemError("mean-of-five B3 contract changed")
    if value.get("baseline", {}).get("score_ensemble") != "mean_of_seed_posteriors":
        raise ResidualGraphDeemError("B3 ensemble contract changed")
    graph = value.get("graph", {})
    if (
        int(graph.get("k", -1)) != 7
        or graph.get("laplacian") != "symmetric_normalized"
        or graph.get("direction") != "cross_only_negative_pooled_c"
    ):
        raise ResidualGraphDeemError("graph/direction contract changed")
    calibration = value.get("calibration", {})
    if (
        calibration.get("pooling")
        != "equal_within_dataset_family_then_equal_across_dataset_families"
        or calibration.get("heldout_unit") != "entire_dataset_family"
        or calibration.get("target_moment_used_for_direction") is not False
        or int(calibration.get("iterations", -1)) != 1
    ):
        raise ResidualGraphDeemError("calibration contract changed")
    boundary = value.get("scientific_boundary", {})
    required_boundary = (
        "fit_is_label_free",
        "baseline_is_frozen_mean_of_five_b3",
        "iu_is_coordinate_generator_not_baseline",
        "pooled_calibration_excludes_entire_target_dataset_family",
        "one_stage_only",
        "natural_24cell_targets_previously_opened",
        "screen_is_exploratory_not_confirmation",
        "confirmation_requires_new_unopened_dataset_families",
    )
    if any(boundary.get(name) is not True for name in required_boundary):
        raise ResidualGraphDeemError("scientific boundary is incomplete")
    permutation = [int(value_) for value_ in value.get("family_axis_permutation", [])]
    if sorted(permutation) != list(range(len(VIEW_ORDER))):
        raise ResidualGraphDeemError("family-axis permutation is not a bijection")
    screen = value.get("screen_cells")
    if not isinstance(screen, list) or len(screen) != 8 or len(set(screen)) != 8:
        raise ResidualGraphDeemError("screen must contain eight unique cells")
    rule = value.get("frozen_screen_rule", {})
    if (
        rule.get("mechanistic_primary") != "E1_B3_ORTH_IUPGRD_FULL"
        or rule.get("secondary_variants_are_mechanism_controls_not_substitute_primaries")
        is not True
    ):
        raise ResidualGraphDeemError("frozen screen primary/control contract changed")
    thresholds = rule.get("primary_survives_only_if", {})
    expected_thresholds = {
        "equal_family_auroc_delta_min",
        "descriptive_family_bootstrap_lower_min",
        "exact_family_signflip_one_sided_p_max",
        "wins_plus_ties_min_of_8",
        "worst_cell_delta_min",
    }
    if set(thresholds) != expected_thresholds or not all(
        np.isfinite(float(thresholds[name])) for name in expected_thresholds
    ):
        raise ResidualGraphDeemError("frozen screen threshold contract mismatch")
    return value, lookup


def _select(raw: str, run_cells: Sequence[str], screen: Sequence[str]) -> list[str]:
    available = [str(value) for value in run_cells]
    if raw == "all":
        selected = list(available)
    elif raw == "screen":
        selected = [str(value) for value in screen]
    else:
        selected = [value.strip() for value in raw.split(",") if value.strip()]
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(set(available))
    ):
        raise ValueError("evaluation cells must be a unique subset of the frozen run")
    return selected


def _validate_bundle(bundle: Any, registered: Mapping[str, Any], path: Path) -> dict:
    source = registered["source"]
    row_hash = canonical_sha256(list(bundle.row_ids))
    checks = {
        "cell": bundle.cell_id == str(registered["cell_id"]),
        "family": bundle.dataset_family == str(registered["dataset_family"]),
        "task": bundle.task_type == str(registered["task_type"]),
        "rows": len(bundle.row_ids) == int(registered["n_rows"]),
        "features": tuple(bundle.feature_names) == tuple(registered["feature_names"]),
        "signs": np.array_equal(
            bundle.confidence_signs,
            np.asarray(registered["confidence_signs"], dtype=np.int8),
        ),
        "inventory": bundle.inventory_sha256 == registered["inventory_sha256"],
        "source": bundle.source_sha256 == source["source_sha256"],
        "source_manifest": bundle.manifest_sha256 == source["manifest_sha256"],
        "admission": bundle.admission_sha256
        == source["admission_contract_sha256"],
        "row_unique": len(bundle.row_ids) == len(set(bundle.row_ids)),
        "group_shape": len(bundle.group_ids) == len(bundle.row_ids),
        "trace_shape": np.asarray(bundle.raw_trace_length).shape
        == (len(bundle.row_ids),),
    }
    sidecar_path = path.with_suffix(".manifest.json")
    sidecar = _read_json(sidecar_path)
    checks.update(
        {
            "sidecar_schema": sidecar.get("schema") == BUNDLE_SCHEMA,
            "sidecar_cell": sidecar.get("cell_id") == bundle.cell_id,
            "sidecar_rows": int(sidecar.get("n_rows", -1)) == len(bundle.row_ids),
            "sidecar_features": int(sidecar.get("n_features", -1))
            == len(bundle.feature_names),
            "sidecar_bundle": sidecar.get("bundle_sha256") == bundle.bundle_sha256,
            "sidecar_row_hash": sidecar.get("ordered_row_id_sha256") == row_hash,
            "sidecar_inventory": sidecar.get("inventory_sha256")
            == bundle.inventory_sha256,
            "sidecar_source": sidecar.get("source_sha256") == bundle.source_sha256,
            "sidecar_source_manifest": sidecar.get("manifest_sha256")
            == bundle.manifest_sha256,
            "sidecar_admission": sidecar.get("admission_sha256")
            == bundle.admission_sha256,
            "sidecar_no_labels": sidecar.get("labels_accessed") is False,
            "sidecar_no_pickle": sidecar.get("allow_pickle") is False,
        }
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ResidualGraphDeemError(
            f"bundle/registry binding failed for {bundle.cell_id}: {failed}"
        )
    return {
        "bundle_path": str(path.resolve()),
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_manifest_path": str(sidecar_path.resolve()),
        "bundle_manifest_sha256": sha256_file(sidecar_path),
        "ordered_row_id_sha256": row_hash,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "source_manifest_sha256": bundle.manifest_sha256,
        "admission_sha256": bundle.admission_sha256,
    }


def _baseline_member_paths(
    baseline_dir: Path, cell: str, seed: int
) -> tuple[Path, Path]:
    array = baseline_dir / "fits" / cell / f"B3__seed{int(seed)}.npz"
    return array, array.with_suffix(".json")


def _validate_baseline_member_hashes(
    *,
    baseline_dir: Path,
    artifact_map: Mapping[str, str],
    bundle: Any,
    seed: int,
) -> tuple[dict, dict]:
    array_path, metadata_path = _baseline_member_paths(
        baseline_dir, bundle.cell_id, seed
    )
    metadata = _read_json(metadata_path)
    _verify_content_hash(
        metadata, context=f"B3 metadata {bundle.cell_id}/seed{seed}"
    )
    array_sha = sha256_file(array_path)
    metadata_sha = sha256_file(metadata_path)
    relative_array = array_path.relative_to(baseline_dir).as_posix()
    relative_metadata = metadata_path.relative_to(baseline_dir).as_posix()
    expected = {
        "status": "complete",
        "arm_id": "B3",
        "cell_id": bundle.cell_id,
        "stem": f"B3__seed{int(seed)}",
        "seed": int(seed),
        "array_sha256": array_sha,
        "bundle_sha256": bundle.bundle_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "source_sha256": bundle.source_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
    }
    failed = [name for name, wanted in expected.items() if metadata.get(name) != wanted]
    if (
        failed
        or metadata.get("health", {}).get("healthy") is not True
        or int(metadata.get("orientation", 0)) not in {-1, 1}
        or artifact_map.get(relative_array) != array_sha
        or artifact_map.get(relative_metadata) != metadata_sha
    ):
        raise ResidualGraphDeemError(
            f"invalid frozen B3 binding: {bundle.cell_id}/seed{seed}; fields={failed}"
        )
    audit = {
        "seed": int(seed),
        "array_path": str(array_path.resolve()),
        "array_sha256": array_sha,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": metadata_sha,
    }
    return metadata, audit


def _expected_artifacts(
    run_dir: Path,
    variants: Sequence[str],
    cells: Sequence[str],
    families: Sequence[str],
) -> list[dict]:
    rows: list[dict] = []
    for family in sorted(str(value) for value in families):
        for suffix, kind in (
            ("json", "calibration_metadata"),
            ("npz", "calibration"),
        ):
            relative = f"calibrations/held_{family}.{suffix}"
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256_file(run_dir / relative),
                    "kind": kind,
                    "held_dataset_family": family,
                }
            )
    for variant in variants:
        for cell in cells:
            for suffix, kind in (("json", "score_metadata"), ("npz", "score")):
                relative = f"scores/{variant}/{cell}/{variant}.{suffix}"
                rows.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(run_dir / relative),
                        "kind": kind,
                        "variant_id": variant,
                        "cell_id": cell,
                    }
                )
    for cell in cells:
        for suffix, kind in (("json", "cell_state_metadata"), ("npz", "cell_state")):
            relative = f"states/{cell}.{suffix}"
            rows.append(
                {
                    "path": relative,
                    "sha256": sha256_file(run_dir / relative),
                    "kind": kind,
                    "cell_id": cell,
                }
            )
    return sorted(rows, key=lambda row: row["path"])


def _assert_exact_artifact_tree(run_dir: Path, expected: Sequence[Mapping[str, Any]]) -> None:
    expected_paths = {str(row["path"]) for row in expected}
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for root in ("calibrations", "scores", "states")
        for path in (run_dir / root).rglob("*")
        if path.is_file() and path.suffix in {".json", ".npz"}
    }
    if actual_paths != expected_paths:
        raise ResidualGraphDeemError(
            "fit artifact tree is not exact; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )


def _hash_contract_pass(
    *,
    config_path: Path,
    registry_path: Path,
    bundle_dir: Path,
    baseline_dir: Path,
    run_dir: Path,
    cells_raw: str,
) -> dict:
    """Pass 1: verify every contract and file hash without importing labels."""

    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before hash preflight")
    config, variant_lookup = _load_config(config_path)
    registry = load_registry(registry_path)
    registry_ids = [str(row["cell_id"]) for row in registry["cells"]]
    if len(registry_ids) != len(set(registry_ids)):
        raise ResidualGraphDeemError("registry cell IDs are duplicated")

    run_dir = run_dir.resolve()
    run_path = run_dir / "RUN_DEFINITION.json"
    manifest_path = run_dir / "FIT_ARTIFACT_MANIFEST.json"
    completion_path = run_dir / "FIT_COMPLETE.json"
    definition = _read_json(run_path)
    _verify_content_hash(definition, context="IU-PGRD run definition")
    definition_file_sha = sha256_file(run_path)
    manifest = _read_json(manifest_path)
    _verify_content_hash(manifest, context="IU-PGRD fit artifact manifest")
    completion = _read_json(completion_path)
    _verify_content_hash(completion, context="IU-PGRD fit completion")

    source_dependencies = definition.get("source_dependencies")
    environment = definition.get("environment")
    if (
        not isinstance(source_dependencies, dict)
        or source_dependencies != dict(sorted(source_dependencies.items()))
        or canonical_sha256(source_dependencies)
        != definition.get("source_dependency_sha256")
        or any("residual_graph_deem_labels" in path for path in source_dependencies)
    ):
        raise ResidualGraphDeemError("invalid target-free source dependency closure")
    source_failures = []
    for relative, expected_sha in source_dependencies.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            source_failures.append(relative)
    if source_failures:
        raise ResidualGraphDeemError(
            f"current source does not match frozen fit: {source_failures}"
        )
    if (
        not isinstance(environment, dict)
        or canonical_sha256(
            {key: value for key, value in environment.items() if key != "environment_sha256"}
        )
        != environment.get("environment_sha256")
        or environment != environment_fingerprint()
    ):
        raise ResidualGraphDeemError("frozen/current environment contract mismatch")

    run_cells = [str(value) for value in definition.get("cells", [])]
    run_variants = [str(value) for value in definition.get("variants", [])]
    run_seeds = [int(value) for value in definition.get("baseline_seeds", [])]
    if (
        not run_cells
        or len(run_cells) != len(set(run_cells))
        or not set(run_cells).issubset(set(registry_ids))
        or tuple(run_variants) != EXPECTED_VARIANTS
        or run_seeds != [0, 1, 2, 3, 4]
    ):
        raise ResidualGraphDeemError("invalid frozen run roster")
    selected_cells = _select(cells_raw, run_cells, config["screen_cells"])
    family_by_cell = {
        cell: str(registry_cell(registry, cell)["dataset_family"])
        for cell in run_cells
    }
    families = sorted(set(family_by_cell.values()))
    expected_donors = {
        held: [cell for cell in run_cells if family_by_cell[cell] != held]
        for held in families
    }
    if (
        definition.get("schema") != RUN_SCHEMA
        or definition.get("status") != "frozen_before_fit"
        or definition.get("experiment_id") != config.get("experiment_id")
        or Path(str(definition.get("config_path", ""))).resolve()
        != config_path.resolve()
        or definition.get("config_sha256") != sha256_file(config_path)
        or Path(str(definition.get("registry_path", ""))).resolve()
        != registry_path.resolve()
        or definition.get("registry_sha256") != sha256_file(registry_path)
        or Path(str(definition.get("bundle_dir", ""))).resolve()
        != bundle_dir.resolve()
        or Path(str(definition.get("baseline_dir", ""))).resolve()
        != baseline_dir.resolve()
        or definition.get("dataset_family_by_cell") != family_by_cell
        or definition.get("donor_cells_by_held_dataset_family") != expected_donors
        or run_variants != [row["id"] for row in config["variants"]]
        or run_seeds != config["baseline"]["seeds"]
        or definition.get("targets_accessed_during_fit") is not False
        or definition.get("labels_module_imported") is not False
    ):
        raise ResidualGraphDeemError("invalid IU-PGRD run definition binding")

    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("experiment_id") != config.get("experiment_id")
        or manifest.get("run_definition_path") != "RUN_DEFINITION.json"
        or manifest.get("run_definition_sha256") != definition_file_sha
        or manifest.get("targets_accessed_during_fit") is not False
        or manifest.get("labels_module_imported") is not False
        or not isinstance(manifest.get("artifacts"), list)
        or int(manifest.get("artifact_count", -1)) != len(manifest.get("artifacts", []))
    ):
        raise ResidualGraphDeemError("invalid IU-PGRD fit artifact manifest")
    expected_artifacts = _expected_artifacts(
        run_dir, run_variants, run_cells, families
    )
    if manifest["artifacts"] != expected_artifacts:
        raise ResidualGraphDeemError("fit manifest is not the exact canonical artifact roster")
    _assert_exact_artifact_tree(run_dir, expected_artifacts)

    if (
        completion.get("schema") != COMPLETE_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("experiment_id") != config.get("experiment_id")
        or completion.get("run_definition_sha256") != definition_file_sha
        or completion.get("fit_artifact_manifest_path")
        != "FIT_ARTIFACT_MANIFEST.json"
        or completion.get("fit_artifact_manifest_sha256")
        != sha256_file(manifest_path)
        or completion.get("cells") != run_cells
        or completion.get("variants") != run_variants
        or completion.get("targets_accessed_during_fit") is not False
        or completion.get("labels_module_imported") is not False
    ):
        raise ResidualGraphDeemError("invalid IU-PGRD fit completion")

    baseline_freeze_path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    baseline_freeze = _read_json(baseline_freeze_path)
    baseline_content_sha = _verify_content_hash(
        baseline_freeze, context="B3 score-freeze manifest"
    )
    baseline_artifacts = baseline_freeze.get("artifacts")
    if not isinstance(baseline_artifacts, list) or not baseline_artifacts:
        raise ResidualGraphDeemError("B3 score-freeze artifact inventory missing")
    artifact_map = {
        str(row.get("path")): str(row.get("sha256")) for row in baseline_artifacts
    }
    if len(artifact_map) != len(baseline_artifacts):
        raise ResidualGraphDeemError("duplicate B3 score-freeze artifact path")
    run_baseline_audit = definition.get("baseline_freeze_audit", {})
    if (
        baseline_freeze.get("schema") != "deem_vs_iupcr_score_freeze_v1"
        or baseline_freeze.get("status") != "complete"
        or baseline_freeze.get("debug") is not False
        or "B3" not in baseline_freeze.get("arms", [])
        or baseline_freeze.get("seeds") != [0, 1, 2, 3, 4]
        or run_baseline_audit.get("path") != str(baseline_freeze_path.resolve())
        or run_baseline_audit.get("file_sha256") != sha256_file(baseline_freeze_path)
        or run_baseline_audit.get("content_sha256") != baseline_content_sha
        or run_baseline_audit.get("run_definition_sha256")
        != baseline_freeze.get("run_definition_sha256")
    ):
        raise ResidualGraphDeemError("B3 score-freeze/run binding failed")

    bundles: dict[str, Any] = {}
    bundle_audits: dict[str, dict] = {}
    baseline_metadata: dict[tuple[str, int], dict] = {}
    baseline_member_audits: dict[str, list[dict]] = {}
    for cell in run_cells:
        bundle_path = bundle_dir / f"{cell}.npz"
        bundle = load_target_free_bundle(bundle_path)
        audit = _validate_bundle(bundle, registry_cell(registry, cell), bundle_path)
        bundles[cell] = bundle
        bundle_audits[cell] = audit
        if definition.get("bundle_audits", {}).get(cell) != audit:
            raise ResidualGraphDeemError(f"run/bundle audit mismatch: {cell}")
        member_audits = []
        for seed in run_seeds:
            metadata, member_audit = _validate_baseline_member_hashes(
                baseline_dir=baseline_dir,
                artifact_map=artifact_map,
                bundle=bundle,
                seed=seed,
            )
            baseline_metadata[(cell, seed)] = metadata
            member_audits.append(member_audit)
        if definition.get("baseline_member_audits", {}).get(cell) != member_audits:
            raise ResidualGraphDeemError(f"run/B3 member audit mismatch: {cell}")
        baseline_member_audits[cell] = member_audits

    state_metadata = {}
    calibration_metadata = {}
    score_metadata = {}
    for cell in run_cells:
        path = run_dir / "states" / f"{cell}.json"
        value = _read_json(path)
        _verify_content_hash(value, context=f"state metadata {cell}")
        state_metadata[cell] = value
    for family in families:
        path = run_dir / "calibrations" / f"held_{family}.json"
        value = _read_json(path)
        _verify_content_hash(value, context=f"calibration metadata {family}")
        calibration_metadata[family] = value
    for variant, cell in itertools.product(run_variants, run_cells):
        path = run_dir / "scores" / variant / cell / f"{variant}.json"
        value = _read_json(path)
        _verify_content_hash(value, context=f"score metadata {variant}/{cell}")
        score_metadata[(variant, cell)] = value

    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module entered import closure in hash pass")
    return {
        "config": config,
        "variant_lookup": variant_lookup,
        "registry": registry,
        "definition": definition,
        "definition_file_sha256": definition_file_sha,
        "manifest": manifest,
        "completion": completion,
        "run_cells": run_cells,
        "selected_cells": selected_cells,
        "run_variants": run_variants,
        "run_seeds": run_seeds,
        "family_by_cell": family_by_cell,
        "families": families,
        "bundles": bundles,
        "bundle_audits": bundle_audits,
        "baseline_metadata": baseline_metadata,
        "baseline_member_audits": baseline_member_audits,
        "state_metadata": state_metadata,
        "calibration_metadata": calibration_metadata,
        "score_metadata": score_metadata,
        "paths": {
            "config": config_path.resolve(),
            "registry": registry_path.resolve(),
            "bundle_dir": bundle_dir.resolve(),
            "baseline_dir": baseline_dir.resolve(),
            "run_dir": run_dir,
            "run": run_path,
            "manifest": manifest_path,
            "completion": completion_path,
            "baseline_freeze": baseline_freeze_path,
        },
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "registry_sha256": sha256_file(registry_path),
            "registry_content_sha256": registry["registry_content_sha256"],
            "run_definition_sha256": definition_file_sha,
            "run_definition_content_sha256": definition["content_sha256"],
            "run_source_dependency_sha256": definition["source_dependency_sha256"],
            "run_environment_sha256": definition["environment"]["environment_sha256"],
            "fit_manifest_sha256": sha256_file(manifest_path),
            "fit_manifest_content_sha256": manifest["content_sha256"],
            "fit_completion_sha256": sha256_file(completion_path),
            "fit_completion_content_sha256": completion["content_sha256"],
            "baseline_score_freeze_manifest_sha256": sha256_file(
                baseline_freeze_path
            ),
            "baseline_score_freeze_content_sha256": baseline_content_sha,
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
    }


def _csr_from_arrays(arrays: Mapping[str, np.ndarray], prefix: str) -> sparse.csr_matrix:
    shape_raw = np.asarray(arrays[f"{prefix}_shape"], dtype=np.int64)
    data = np.asarray(arrays[f"{prefix}_data"], dtype=np.float64)
    indices = np.asarray(arrays[f"{prefix}_indices"], dtype=np.int64)
    indptr = np.asarray(arrays[f"{prefix}_indptr"], dtype=np.int64)
    if (
        shape_raw.shape != (2,)
        or data.ndim != 1
        or indices.shape != data.shape
        or indptr.ndim != 1
        or not np.isfinite(data).all()
    ):
        raise ResidualGraphDeemError(f"invalid {prefix} CSR payload")
    try:
        matrix = sparse.csr_matrix(
            (data, indices, indptr), shape=tuple(int(value) for value in shape_raw)
        )
        matrix.check_format(full_check=True)
    except Exception as exc:
        raise ResidualGraphDeemError(f"invalid {prefix} CSR structure") from exc
    if not matrix.has_sorted_indices:
        raise ResidualGraphDeemError(f"{prefix} CSR indices are not canonical")
    return matrix


def _sparse_max_abs(matrix: sparse.spmatrix) -> float:
    value = matrix.tocsr()
    return float(np.max(np.abs(value.data))) if value.nnz else 0.0


def _load_npz_exact(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        actual = set(data.files)
        if actual != required:
            raise ResidualGraphDeemError(
                f"NPZ schema mismatch: {path}; "
                f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
            )
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ResidualGraphDeemError(f"object dtype in allow_pickle=False artifact: {path}")
    return arrays


def _load_baseline_scores(hash_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output = {}
    baseline_dir = hash_state["paths"]["baseline_dir"]
    for cell in hash_state["run_cells"]:
        bundle = hash_state["bundles"][cell]
        members = []
        for seed in hash_state["run_seeds"]:
            path, _ = _baseline_member_paths(baseline_dir, cell, seed)
            with np.load(path, allow_pickle=False) as data:
                required = {"score", "posterior", "logit", "feature_names"}
                if not required.issubset(set(data.files)):
                    raise ResidualGraphDeemError(f"B3 score payload incomplete: {cell}/{seed}")
                score = np.asarray(data["score"], dtype=np.float64)
                posterior = np.asarray(data["posterior"], dtype=np.float64)
                logit = np.asarray(data["logit"], dtype=np.float64)
                names = tuple(str(value) for value in data["feature_names"].tolist())
            n = len(bundle.row_ids)
            if (
                score.shape != (n,)
                or posterior.shape != (n, 2)
                or logit.shape != (n,)
                or names != tuple(bundle.feature_names)
                or not np.isfinite(score).all()
                or not np.isfinite(posterior).all()
                or not np.isfinite(logit).all()
                or np.any((score < 0.0) | (score > 1.0))
            ):
                raise ResidualGraphDeemError(f"invalid B3 score arrays: {cell}/{seed}")
            _require_close(score, expit(logit), context="B3 score/logit", atol=1e-12)
            _require_close(
                posterior,
                np.column_stack([1.0 - score, score]),
                context="B3 score/posterior",
                atol=1e-12,
            )
            members.append(score)
        output[cell] = np.mean(np.stack(members), axis=0)
    return output


def _validate_state(
    *,
    cell: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    hash_state: Mapping[str, Any],
    baseline_score: np.ndarray,
) -> dict:
    bundle = hash_state["bundles"][cell]
    run_dir = hash_state["paths"]["run_dir"]
    path = run_dir / "states" / f"{cell}.npz"
    n = len(bundle.row_ids)
    p = len(bundle.feature_names)
    groups = family_index_map(bundle.feature_names)
    families = tuple(groups)
    g = len(families)
    global_g = len(VIEW_ORDER)

    row_ids = tuple(str(value) for value in arrays["row_id"].tolist())
    feature_names = tuple(str(value) for value in arrays["feature_names"].tolist())
    family_order = tuple(str(value) for value in arrays["family_order"].tolist())
    global_order = tuple(str(value) for value in arrays["global_family_order"].tolist())
    if (
        row_ids != tuple(bundle.row_ids)
        or feature_names != tuple(bundle.feature_names)
        or family_order != families
        or global_order != tuple(VIEW_ORDER)
    ):
        raise ResidualGraphDeemError(f"state row/feature/family order mismatch: {cell}")

    shapes = {
        "risk_transform_mean": (p,),
        "risk_transform_scale": (p,),
        "risk_transform_constant_mask": (p,),
        "baseline_score": (n,),
        "baseline_logit": (n,),
        "baseline_z": (n,),
        "iu_score": (n,),
        "iu_score_aligned": (n,),
        "iu_weights": (p,),
        "raw_family_contributions": (n, g),
        "standardized_family_contributions": (n, g),
        "iu_family_residuals": (n, g),
        "moment_A": (global_g, global_g),
        "moment_c": (global_g,),
        "moment_presence": (global_g,),
        "iu_transform_contribution_mean": (g,),
        "iu_transform_contribution_scale": (g,),
        "iu_transform_baseline_loadings": (g,),
        "iu_transform_residual_mean": (g,),
        "iu_transform_residual_scale": (g,),
    }
    numeric = {}
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value.astype(np.float64)).all():
            raise ResidualGraphDeemError(f"invalid state array {name}: {cell}")
        numeric[name] = value.astype(np.float64)
    for name in (
        "baseline_logit_mean",
        "baseline_logit_scale",
        "iu_orientation",
        "iu_transform_baseline_mean",
        "iu_transform_baseline_scale",
    ):
        value = np.asarray(arrays[name])
        if value.shape != () or not np.isfinite(float(value.item())):
            raise ResidualGraphDeemError(f"invalid scalar state array {name}: {cell}")

    if not np.array_equal(numeric["baseline_score"], baseline_score):
        raise ResidualGraphDeemError(f"state is not bound to exact mean-of-five B3: {cell}")
    expected_logit = _safe_logit(baseline_score)
    _require_close(
        numeric["baseline_logit"],
        expected_logit,
        context=f"stored B3 logit {cell}",
        atol=2e-15,
    )
    logit_mean = float(arrays["baseline_logit_mean"].item())
    logit_scale = float(arrays["baseline_logit_scale"].item())
    if logit_scale <= 0.0:
        raise ResidualGraphDeemError(f"nonpositive B3 logit scale: {cell}")
    _require_close(
        numeric["baseline_z"],
        (expected_logit - logit_mean) / logit_scale,
        context=f"B3 standardization {cell}",
        atol=1e-12,
    )
    if abs(logit_mean - float(np.mean(expected_logit))) > 1e-14 or abs(
        logit_scale - float(np.std(expected_logit))
    ) > 1e-14:
        raise ResidualGraphDeemError(f"B3 mean/scale mismatch: {cell}")

    risk_mean = numeric["risk_transform_mean"]
    risk_scale = numeric["risk_transform_scale"]
    constant_mask = numeric["risk_transform_constant_mask"].astype(np.int8)
    raw_scale = np.std(bundle.X_raw, axis=0)
    expected_constant = (raw_scale < 1e-12).astype(np.int8)
    expected_scale = raw_scale.copy()
    expected_scale[expected_constant.astype(bool)] = 1.0
    _require_close(risk_mean, np.mean(bundle.X_raw, axis=0), context=f"risk mean {cell}")
    _require_close(risk_scale, expected_scale, context=f"risk scale {cell}")
    if not np.array_equal(constant_mask, expected_constant) or np.any(risk_scale <= 0.0):
        raise ResidualGraphDeemError(f"risk constant-mask/scale mismatch: {cell}")
    X_risk = -(
        (np.asarray(bundle.X_raw, dtype=np.float64) - risk_mean[None, :])
        / risk_scale[None, :]
    ) * np.asarray(bundle.confidence_signs, dtype=np.float64)[None, :]

    weights = numeric["iu_weights"]
    raw_contributions = numeric["raw_family_contributions"]
    expected_contributions = np.column_stack(
        [
            np.sum(X_risk[:, list(indices)] * weights[list(indices)], axis=1)
            for indices in groups.values()
        ]
    )
    _require_close(
        raw_contributions,
        expected_contributions,
        context=f"IU family contributions {cell}",
        atol=1e-10,
    )
    raw_iu = X_risk @ weights
    _require_close(
        np.sum(raw_contributions, axis=1),
        raw_iu,
        context=f"IU contribution reconstruction {cell}",
        atol=1e-10,
    )
    iu_mean = float(arrays["iu_transform_baseline_mean"].item())
    iu_scale = float(arrays["iu_transform_baseline_scale"].item())
    if iu_scale <= 0.0:
        raise ResidualGraphDeemError(f"nonpositive IU baseline scale: {cell}")
    iu_score = numeric["iu_score"]
    _require_close(
        iu_score,
        (raw_iu - iu_mean) / iu_scale,
        context=f"standardized IU score {cell}",
        atol=1e-10,
    )
    contribution_mean = numeric["iu_transform_contribution_mean"]
    contribution_scale = numeric["iu_transform_contribution_scale"]
    residual_scale = numeric["iu_transform_residual_scale"]
    if np.any(contribution_scale <= 0.0) or np.any(residual_scale <= 0.0):
        raise ResidualGraphDeemError(f"nonpositive IU transform scale: {cell}")
    standardized = (raw_contributions - contribution_mean[None, :]) / (
        contribution_scale[None, :]
    )
    _require_close(
        numeric["standardized_family_contributions"],
        standardized,
        context=f"standardized IU contributions {cell}",
        atol=1e-10,
    )
    residual_unoriented = (
        standardized
        - iu_score[:, None]
        * numeric["iu_transform_baseline_loadings"][None, :]
        - numeric["iu_transform_residual_mean"][None, :]
    ) / residual_scale[None, :]
    orientation = int(np.asarray(arrays["iu_orientation"]).item())
    if orientation not in {-1, 1}:
        raise ResidualGraphDeemError(f"invalid IU orientation: {cell}")
    iu_aligned = orientation * iu_score
    residuals = orientation * residual_unoriented
    _require_close(
        numeric["iu_score_aligned"], iu_aligned, context=f"aligned IU score {cell}"
    )
    _require_close(
        numeric["iu_family_residuals"],
        residuals,
        context=f"aligned IU residuals {cell}",
        atol=1e-10,
    )
    correlation = float(np.dot(numeric["baseline_z"], iu_score) / n)
    if abs(correlation) <= 0.05 or orientation != (1 if correlation > 0.0 else -1):
        raise ResidualGraphDeemError(f"IU/B3 orientation anchor failed: {cell}")
    if max(
        _max_abs(np.mean(residuals, axis=0)),
        _max_abs(np.std(residuals, axis=0) - 1.0),
        _max_abs(residuals.T @ iu_aligned / n),
    ) > 1e-8:
        raise ResidualGraphDeemError(f"IU residual invariants failed: {cell}")

    graph = _csr_from_arrays(arrays, "graph")
    laplacian = _csr_from_arrays(arrays, "laplacian")
    if graph.shape != (n, n) or laplacian.shape != (n, n):
        raise ResidualGraphDeemError(f"graph/state row alignment failed: {cell}")
    if (
        _sparse_max_abs(graph - graph.T) > 1e-12
        or _sparse_max_abs(laplacian - laplacian.T) > 1e-12
        or _max_abs(graph.diagonal()) > 1e-14
        or graph.nnz == 0
        or np.any(graph.data <= 0.0)
    ):
        raise ResidualGraphDeemError(f"graph symmetry/weight contract failed: {cell}")
    expected_laplacian = symmetric_normalized_laplacian(graph)
    if _sparse_max_abs(laplacian - expected_laplacian) > 1e-12:
        raise ResidualGraphDeemError(f"stored normalized Laplacian mismatch: {cell}")

    raw_A = np.asarray(residuals.T @ (laplacian @ residuals) / n, dtype=float)
    raw_A = 0.5 * (raw_A + raw_A.T)
    raw_c = np.asarray(residuals.T @ (laplacian @ iu_aligned) / n, dtype=float)
    raw_trace = float(np.trace(raw_A))
    if not np.isfinite(raw_trace) or raw_trace <= 1e-12:
        raise ResidualGraphDeemError(f"nonpositive roughness trace: {cell}")
    trace_scale = g / raw_trace
    indices = np.asarray([VIEW_ORDER.index(name) for name in families], dtype=int)
    expected_A = np.zeros((global_g, global_g), dtype=float)
    expected_c = np.zeros(global_g, dtype=float)
    expected_A[np.ix_(indices, indices)] = trace_scale * raw_A
    expected_c[indices] = trace_scale * raw_c
    expected_presence = np.zeros(global_g, dtype=np.int8)
    expected_presence[indices] = 1
    _require_close(numeric["moment_A"], expected_A, context=f"roughness A {cell}")
    _require_close(numeric["moment_c"], expected_c, context=f"roughness c {cell}")
    if not np.array_equal(
        numeric["moment_presence"].astype(np.int8), expected_presence
    ):
        raise ResidualGraphDeemError(f"roughness presence mask mismatch: {cell}")

    state_hashes = {
        "array_sha256": sha256_file(path),
        "metadata_sha256": sha256_file(path.with_suffix(".json")),
    }
    expected_metadata = {
        "schema": STATE_SCHEMA,
        "status": "complete",
        "cell_id": cell,
        "dataset_family": hash_state["family_by_cell"][cell],
        "run_definition_sha256": hash_state["definition_file_sha256"],
        "array_sha256": state_hashes["array_sha256"],
        "bundle_audit": hash_state["bundle_audits"][cell],
        "baseline_member_audits": hash_state["baseline_member_audits"][cell],
        "row_id_array_sha256": _ndarray_sha256(np.asarray(bundle.row_ids, dtype=str)),
        "uses_labels": False,
    }
    failed = [name for name, wanted in expected_metadata.items() if metadata.get(name) != wanted]
    diagnostics = metadata.get("diagnostics", {})
    moment_diagnostics = metadata.get("moment_diagnostics", {})
    if (
        failed
        or diagnostics.get("uses_labels") is not False
        or int(diagnostics.get("n_rows", -1)) != n
        or int(diagnostics.get("n_features", -1)) != p
        or int(diagnostics.get("n_families", -1)) != g
        or int(diagnostics.get("graph_k", -1)) != min(
            int(hash_state["config"]["graph"]["k"]), n - 1
        )
        or int(diagnostics.get("graph_nnz", -1)) != graph.nnz
        or int(diagnostics.get("iu_orientation", 0)) != orientation
        or int(moment_diagnostics.get("n_samples", -1)) != n
        or int(moment_diagnostics.get("n_families_present", -1)) != g
        or abs(float(moment_diagnostics.get("roughness_trace_raw", np.nan)) - raw_trace)
        > 1e-10
        or abs(float(moment_diagnostics.get("trace_scale", np.nan)) - trace_scale)
        > 1e-10
    ):
        raise ResidualGraphDeemError(f"state metadata/mechanics binding failed: {cell}; {failed}")
    return {
        "arrays": arrays,
        "state_hashes": state_hashes,
        "families": families,
        "graph_nnz": int(graph.nnz),
        "iu_b3_correlation": correlation,
        "roughness_trace_raw": raw_trace,
    }


def _validate_calibration(
    *,
    family: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    hash_state: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
) -> dict:
    run_dir = hash_state["paths"]["run_dir"]
    path = run_dir / "calibrations" / f"held_{family}.npz"
    donors = hash_state["definition"]["donor_cells_by_held_dataset_family"][family]
    donor_families = [hash_state["family_by_cell"][cell] for cell in donors]
    global_g = len(VIEW_ORDER)
    d = len(donors)
    direction = np.asarray(arrays["direction"], dtype=np.float64)
    stored_donors = [str(value) for value in arrays["donor_cells"].tolist()]
    stored_families = [
        str(value) for value in arrays["donor_dataset_families"].tolist()
    ]
    donor_A = np.asarray(arrays["donor_moment_A"], dtype=np.float64)
    donor_c = np.asarray(arrays["donor_moment_c"], dtype=np.float64)
    donor_presence = np.asarray(arrays["donor_moment_presence"], dtype=np.int8)
    if (
        direction.shape != (global_g,)
        or donor_A.shape != (d, global_g, global_g)
        or donor_c.shape != (d, global_g)
        or donor_presence.shape != (d, global_g)
        or not all(
            np.isfinite(value).all() for value in (direction, donor_A, donor_c)
        )
        or stored_donors != donors
        or stored_families != donor_families
        or any(value == family for value in donor_families)
    ):
        raise ResidualGraphDeemError(f"invalid held-family calibration arrays: {family}")
    expected_A = np.stack(
        [states[cell]["arrays"]["moment_A"] for cell in donors]
    )
    expected_c = np.stack(
        [states[cell]["arrays"]["moment_c"] for cell in donors]
    )
    expected_presence = np.stack(
        [states[cell]["arrays"]["moment_presence"] for cell in donors]
    ).astype(np.int8)
    if not (
        np.array_equal(donor_A, expected_A)
        and np.array_equal(donor_c, expected_c)
        and np.array_equal(donor_presence, expected_presence)
    ):
        raise ResidualGraphDeemError(f"calibration donor moments changed: {family}")
    groups = sorted(set(donor_families))
    pooled_A = np.mean(
        [
            np.mean(
                [donor_A[index] for index, value in enumerate(donor_families) if value == group],
                axis=0,
            )
            for group in groups
        ],
        axis=0,
    )
    pooled_c = np.mean(
        [
            np.mean(
                [donor_c[index] for index, value in enumerate(donor_families) if value == group],
                axis=0,
            )
            for group in groups
        ],
        axis=0,
    )
    pooled_A = 0.5 * (pooled_A + pooled_A.T)
    _require_close(direction, -pooled_c, context=f"pooled -c direction {family}")
    calibration_hashes = {
        "array_sha256": sha256_file(path),
        "metadata_sha256": sha256_file(path.with_suffix(".json")),
    }
    expected_state_hashes = {
        cell: states[cell]["state_hashes"] for cell in donors
    }
    diagnostics = metadata.get("diagnostics", {})
    if (
        metadata.get("schema") != CALIBRATION_SCHEMA
        or metadata.get("status") != "complete"
        or metadata.get("held_dataset_family") != family
        or metadata.get("donor_cells") != donors
        or metadata.get("donor_dataset_families") != donor_families
        or metadata.get("whole_held_dataset_family_excluded") is not True
        or metadata.get("run_definition_sha256")
        != hash_state["definition_file_sha256"]
        or metadata.get("array_sha256") != calibration_hashes["array_sha256"]
        or metadata.get("donor_state_hashes") != expected_state_hashes
        or metadata.get("uses_labels") is not False
        or diagnostics.get("uses_labels") is not False
        or diagnostics.get("cross_only") is not True
        or int(diagnostics.get("n_donor_cells", -1)) != d
        or int(diagnostics.get("n_donor_groups", -1)) != len(groups)
        or diagnostics.get("donor_groups") != groups
        or abs(float(diagnostics.get("pooled_A_trace", np.nan)) - float(np.trace(pooled_A)))
        > 1e-10
        or abs(float(diagnostics.get("pooled_c_norm", np.nan)) - float(np.linalg.norm(pooled_c)))
        > 1e-10
        or abs(float(diagnostics.get("direction_norm", np.nan)) - float(np.linalg.norm(direction)))
        > 1e-10
    ):
        raise ResidualGraphDeemError(f"calibration metadata/mechanics failed: {family}")
    return {
        "arrays": arrays,
        "calibration_hashes": calibration_hashes,
        "n_donor_cells": d,
        "n_donor_families": len(groups),
        "direction_norm": float(np.linalg.norm(direction)),
    }


def _validate_score(
    *,
    variant: str,
    cell: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    hash_state: Mapping[str, Any],
    state: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict:
    run_dir = hash_state["paths"]["run_dir"]
    path = run_dir / "scores" / variant / cell / f"{variant}.npz"
    bundle = hash_state["bundles"][cell]
    family = hash_state["family_by_cell"][cell]
    settings = hash_state["variant_lookup"][variant]["config"]
    n = len(bundle.row_ids)
    global_g = len(VIEW_ORDER)
    local_families = state["families"]
    g = len(local_families)

    row_ids = tuple(str(value) for value in arrays["row_id"].tolist())
    vectors = {}
    for name in (
        "score",
        "baseline_score",
        "logit",
        "baseline_logit",
        "correction_z",
        "raw_correction",
        "projected_correction",
    ):
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ResidualGraphDeemError(f"invalid score array {name}: {variant}/{cell}")
        vectors[name] = value
    coefficients = np.asarray(arrays["projection_coefficients"], dtype=np.float64)
    calibration_direction = np.asarray(arrays["calibration_direction"], dtype=np.float64)
    applied_direction = np.asarray(arrays["applied_direction"], dtype=np.float64)
    permutation = np.asarray(arrays["family_axis_permutation"], dtype=np.int64)
    row_permutation = np.asarray(arrays["row_permutation"], dtype=np.int64)
    if (
        row_ids != tuple(bundle.row_ids)
        or coefficients.shape != (2,)
        or calibration_direction.shape != (global_g,)
        or applied_direction.shape != (global_g,)
        or permutation.shape != (global_g,)
        or row_permutation.shape != (n,)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(calibration_direction).all()
        or not np.isfinite(applied_direction).all()
        or sorted(permutation.tolist()) != list(range(global_g))
        or sorted(row_permutation.tolist()) != list(range(n))
    ):
        raise ResidualGraphDeemError(f"score order/direction payload invalid: {variant}/{cell}")
    if not np.array_equal(
        vectors["baseline_score"], state["arrays"]["baseline_score"]
    ) or not np.array_equal(
        vectors["baseline_logit"], state["arrays"]["baseline_logit"]
    ):
        raise ResidualGraphDeemError(f"score lost exact B3 baseline: {variant}/{cell}")
    if not np.array_equal(
        calibration_direction, calibration["arrays"]["direction"]
    ):
        raise ResidualGraphDeemError(f"score calibration direction changed: {variant}/{cell}")
    expected_axis_permutation = np.asarray(
        hash_state["config"]["family_axis_permutation"], dtype=np.int64
    )
    if not np.array_equal(permutation, expected_axis_permutation):
        raise ResidualGraphDeemError(f"family-axis permutation changed: {variant}/{cell}")
    expected_applied = (
        calibration_direction[permutation]
        if settings["family_axis_permuted"]
        else calibration_direction
    )
    if not np.array_equal(applied_direction, expected_applied):
        raise ResidualGraphDeemError(f"applied direction mismatch: {variant}/{cell}")

    local_indices = np.asarray([VIEW_ORDER.index(name) for name in local_families])
    raw = np.asarray(
        state["arrays"]["iu_family_residuals"]
        @ applied_direction[local_indices],
        dtype=np.float64,
    )
    if settings["row_permuted"]:
        expected_row_permutation = deterministic_row_permutation(
            bundle.row_ids, salt=str(hash_state["config"]["row_permutation_salt"])
        )
        raw = raw[expected_row_permutation]
    else:
        expected_row_permutation = np.arange(n, dtype=np.int64)
    if not np.array_equal(row_permutation, expected_row_permutation):
        raise ResidualGraphDeemError(f"row permutation mismatch: {variant}/{cell}")
    _require_close(vectors["raw_correction"], raw, context=f"raw correction {variant}/{cell}")

    baseline_z = np.asarray(state["arrays"]["baseline_z"], dtype=np.float64)
    if settings["project_against_b3"]:
        design = np.column_stack([np.ones(n), baseline_z])
        expected_coefficients = np.linalg.lstsq(design, raw, rcond=None)[0]
        projected = raw - design @ expected_coefficients
    else:
        expected_coefficients = np.zeros(2, dtype=np.float64)
        projected = raw.copy()
    _require_close(
        coefficients,
        expected_coefficients,
        context=f"projection coefficients {variant}/{cell}",
        atol=1e-12,
    )
    _require_close(
        vectors["projected_correction"],
        projected,
        context=f"projected correction {variant}/{cell}",
        atol=1e-12,
    )

    trust = float(settings["trust_factor"])
    if trust == 0.0:
        if (
            not np.array_equal(vectors["correction_z"], np.zeros(n, dtype=np.float64))
            or not np.array_equal(vectors["logit"], vectors["baseline_logit"])
            or not np.array_equal(vectors["score"], vectors["baseline_score"])
        ):
            raise ResidualGraphDeemError(f"E0 is not an exact B3 alias: {cell}")
    else:
        projected_scale = float(np.std(projected))
        if not np.isfinite(projected_scale) or projected_scale <= 1e-12:
            raise ResidualGraphDeemError(f"constant projected correction: {variant}/{cell}")
        correction_z = trust / g * projected / projected_scale
        updated_logit = (
            vectors["baseline_logit"]
            + float(state["arrays"]["baseline_logit_scale"].item()) * correction_z
        )
        _require_close(
            vectors["correction_z"], correction_z, context=f"correction z {variant}/{cell}"
        )
        _require_close(vectors["logit"], updated_logit, context=f"updated logit {variant}/{cell}")
        _require_close(vectors["score"], expit(updated_logit), context=f"updated score {variant}/{cell}")
    if np.any((vectors["score"] < 0.0) | (vectors["score"] > 1.0)):
        raise ResidualGraphDeemError(f"score escaped probability bounds: {variant}/{cell}")

    expected_missing = [name for name in VIEW_ORDER if name not in local_families]
    expected_present_direction = [
        float(applied_direction[VIEW_ORDER.index(name)]) for name in local_families
    ]
    diagnostics = metadata.get("diagnostics", {})
    expected_metadata = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "variant_id": variant,
        "role": hash_state["variant_lookup"][variant]["role"],
        "cell_id": cell,
        "dataset_family": family,
        "run_definition_sha256": hash_state["definition_file_sha256"],
        "array_sha256": sha256_file(path),
        "state_hashes": state["state_hashes"],
        "calibration_hashes": calibration["calibration_hashes"],
        "bundle_audit": hash_state["bundle_audits"][cell],
        "baseline_member_audits": hash_state["baseline_member_audits"][cell],
        "donor_cells": hash_state["definition"][
            "donor_cells_by_held_dataset_family"
        ][family],
        "variant_config": settings,
        "present_families": list(local_families),
        "missing_global_families": expected_missing,
        "applied_direction_present": expected_present_direction,
        "family_axis_permutation": expected_axis_permutation.tolist(),
        "uses_labels": False,
    }
    failed = [name for name, wanted in expected_metadata.items() if metadata.get(name) != wanted]
    if (
        failed
        or diagnostics.get("uses_labels") is not False
        or float(diagnostics.get("trust_factor", np.nan)) != trust
        or bool(diagnostics.get("exact_b3_alias")) != (trust == 0.0)
        or (trust > 0.0 and diagnostics.get("project_against_b3")
            is not bool(settings["project_against_b3"]))
        or (trust > 0.0 and diagnostics.get("row_permuted")
            is not bool(settings["row_permuted"]))
    ):
        raise ResidualGraphDeemError(
            f"score metadata/mechanics binding failed: {variant}/{cell}; {failed}"
        )
    return {
        "score": vectors["score"],
        "correction_z_sd": float(np.std(vectors["correction_z"])),
        "correction_logit_sd": float(
            np.std(vectors["logit"] - vectors["baseline_logit"])
        ),
        "score_pearson_vs_b3": float(
            np.corrcoef(vectors["score"], vectors["baseline_score"])[0, 1]
        ),
    }


def _mechanical_artifact_pass(hash_state: Mapping[str, Any]) -> dict:
    """Pass 2: validate all numerical artifacts in the frozen run."""

    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before mechanical preflight")
    run_dir = hash_state["paths"]["run_dir"]
    baseline_scores = _load_baseline_scores(hash_state)
    states = {}
    for cell in hash_state["run_cells"]:
        path = run_dir / "states" / f"{cell}.npz"
        arrays = _load_npz_exact(path, STATE_ARRAYS)
        states[cell] = _validate_state(
            cell=cell,
            arrays=arrays,
            metadata=hash_state["state_metadata"][cell],
            hash_state=hash_state,
            baseline_score=baseline_scores[cell],
        )
    calibrations = {}
    for family in hash_state["families"]:
        path = run_dir / "calibrations" / f"held_{family}.npz"
        arrays = _load_npz_exact(path, CALIBRATION_ARRAYS)
        calibrations[family] = _validate_calibration(
            family=family,
            arrays=arrays,
            metadata=hash_state["calibration_metadata"][family],
            hash_state=hash_state,
            states=states,
        )
    score_results = {}
    diagnostics = []
    for variant, cell in itertools.product(
        hash_state["run_variants"], hash_state["run_cells"]
    ):
        path = run_dir / "scores" / variant / cell / f"{variant}.npz"
        arrays = _load_npz_exact(path, SCORE_ARRAYS)
        result = _validate_score(
            variant=variant,
            cell=cell,
            arrays=arrays,
            metadata=hash_state["score_metadata"][(variant, cell)],
            hash_state=hash_state,
            state=states[cell],
            calibration=calibrations[hash_state["family_by_cell"][cell]],
        )
        diagnostics.append(
            {
                "variant": variant,
                "cell_id": cell,
                "dataset_family": hash_state["family_by_cell"][cell],
                **{name: value for name, value in result.items() if name != "score"},
            }
        )
        if cell in hash_state["selected_cells"]:
            score_results[(variant, cell)] = result["score"]
    selected_expected = set(
        itertools.product(hash_state["run_variants"], hash_state["selected_cells"])
    )
    if set(score_results) != selected_expected:
        raise ResidualGraphDeemError("selected score cache incomplete after full-run preflight")
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module entered import closure in mechanical pass")
    return {
        "baseline_scores": {
            cell: baseline_scores[cell] for cell in hash_state["selected_cells"]
        },
        "scores": score_results,
        "diagnostics": diagnostics,
        "state_summary": {
            cell: {
                name: value
                for name, value in states[cell].items()
                if name not in {"arrays"}
            }
            for cell in hash_state["run_cells"]
        },
        "calibration_summary": {
            family: {
                name: value
                for name, value in calibrations[family].items()
                if name not in {"arrays"}
            }
            for family in hash_state["families"]
        },
    }


def _load_targets_after_preflight(hash_state: Mapping[str, Any], sidecar_dir: Path):
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before explicit label phase")
    from spectral_utils.residual_graph_deem_labels import (  # noqa: PLC0415
        SIDECAR_SCHEMA,
        join_labels_by_id,
        load_label_sidecar,
    )

    targets = {}
    audits = {}
    for cell in hash_state["selected_cells"]:
        bundle = hash_state["bundles"][cell]
        path = sidecar_dir / f"{cell}.npz"
        manifest_path = path.with_suffix(".manifest.json")
        manifest = _read_json(manifest_path)
        sidecar = load_label_sidecar(path)
        if (
            manifest.get("schema") != SIDECAR_SCHEMA
            or manifest.get("cell_id") != cell
            or int(manifest.get("n_rows", -1)) != len(bundle.row_ids)
            or manifest.get("sidecar_sha256") != sidecar.sidecar_sha256
            or manifest.get("unordered_row_id_sha256")
            != canonical_sha256(sorted(bundle.row_ids))
        ):
            raise ResidualGraphDeemError(f"invalid label-sidecar manifest: {cell}")
        target = join_labels_by_id(bundle, sidecar)
        if target.shape != (len(bundle.row_ids),) or len(np.unique(target)) != 2:
            raise ResidualGraphDeemError(f"invalid/single-class target: {cell}")
        targets[cell] = np.asarray(target, dtype=np.int8)
        audits[cell] = {
            "sidecar_sha256": sidecar.sidecar_sha256,
            "sidecar_manifest_sha256": sha256_file(manifest_path),
            "ordered_join_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        }
    return targets, audits


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if (
        y.shape != values.shape
        or len(np.unique(y)) != 2
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ResidualGraphDeemError("invalid target/score pair")
    return {
        "auroc": float(roc_auc_score(y, values)),
        "auprc": float(average_precision_score(y, values)),
    }


def _summary(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ResidualGraphDeemError(f"no metric rows for {method}/{metric}")
    grouped = defaultdict(list)
    for row in selected:
        grouped[str(row["dataset_family"])].append(float(row[metric]))
    family_means = {
        family: float(np.mean(values)) for family, values in sorted(grouped.items())
    }
    return {
        "method": method,
        "metric": metric,
        "n_cells": len(selected),
        "n_dataset_families": len(family_means),
        "cell_macro": float(np.mean([float(row[metric]) for row in selected])),
        "equal_dataset_family_macro": float(np.mean(list(family_means.values()))),
        "worst_cell": float(min(float(row[metric]) for row in selected)),
        "dataset_family_means": family_means,
    }


def _family_deltas(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    metric: str,
) -> tuple[dict[str, float], dict[str, float]]:
    lookup = {(str(row["cell_id"]), str(row["method"])): row for row in rows}
    cells = sorted({str(row["cell_id"]) for row in rows})
    grouped = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        try:
            base = lookup[(cell, reference)]
            cand = lookup[(cell, candidate)]
        except KeyError as exc:
            raise ResidualGraphDeemError(
                f"incomplete paired rows: {candidate} vs {reference}, {cell}"
            ) from exc
        delta = float(cand[metric]) - float(base[metric])
        grouped[str(base["dataset_family"])].append(delta)
        cell_delta[cell] = delta
    family_delta = {
        family: float(np.mean(values)) for family, values in sorted(grouped.items())
    }
    return family_delta, cell_delta


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    *,
    draws: int,
    seed: int,
) -> dict:
    if int(draws) < 100:
        raise ValueError("bootstrap draws must be at least 100")
    family_auroc, cell_delta = _family_deltas(
        rows, candidate, reference, "auroc"
    )
    family_auprc, _ = _family_deltas(rows, candidate, reference, "auprc")
    families = tuple(family_auroc)
    values = np.asarray([family_auroc[name] for name in families], dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws_array = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        selected = rng.integers(0, len(values), size=len(values))
        draws_array[draw] = float(np.mean(values[selected]))
    if len(values) > 20:
        raise ResidualGraphDeemError("exact family sign-flip unexpectedly large")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values[None, :], axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "equal_dataset_family_auroc_delta": observed,
        "equal_dataset_family_auprc_delta": float(np.mean(list(family_auprc.values()))),
        "descriptive_family_bootstrap_lower": float(np.quantile(draws_array, 0.025)),
        "descriptive_family_bootstrap_upper": float(np.quantile(draws_array, 0.975)),
        "exact_family_signflip_one_sided_p": float(
            np.mean(null >= observed - 1e-15)
        ),
        "exact_family_signflip_assignments": int(2 ** len(values)),
        "wins": int(sum(value > TOLERANCE for value in cell_delta.values())),
        "ties": int(sum(abs(value) <= TOLERANCE for value in cell_delta.values())),
        "losses": int(sum(value < -TOLERANCE for value in cell_delta.values())),
        "worst_cell_delta": float(min(cell_delta.values())),
        "dataset_family_delta": family_auroc,
        "cell_delta": cell_delta,
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_is_descriptive_not_a_null_test": True,
    }


def _screen_verdict(
    config: Mapping[str, Any],
    primary_comparison: Mapping[str, Any],
    *,
    cells: Sequence[str],
    dataset_families: Sequence[str],
) -> dict:
    primary = str(config["frozen_screen_rule"]["mechanistic_primary"])
    eligible = (
        list(cells) == [str(value) for value in config["screen_cells"]]
        and len(cells) == 8
        and len(set(dataset_families)) == 8
        and primary_comparison.get("candidate") == primary
        and primary_comparison.get("reference") == "B3"
    )
    if not eligible:
        return {
            "status": "not_applicable_sensitivity_analysis",
            "verdict": "NOT_APPLICABLE",
            "mechanistic_primary": primary,
            "reason": (
                "evaluation selection is not the ordered frozen eight-cell, "
                "eight-dataset-family screen"
            ),
            "secondary_variants_cannot_substitute_for_primary": True,
        }
    thresholds = config["frozen_screen_rule"]["primary_survives_only_if"]
    clauses = [
        {
            "name": "equal_family_auroc_delta_min",
            "observed": primary_comparison["equal_dataset_family_auroc_delta"],
            "threshold": float(thresholds["equal_family_auroc_delta_min"]),
            "passed": primary_comparison["equal_dataset_family_auroc_delta"]
            >= float(thresholds["equal_family_auroc_delta_min"]),
        },
        {
            "name": "descriptive_family_bootstrap_lower_min",
            "observed": primary_comparison["descriptive_family_bootstrap_lower"],
            "threshold": float(thresholds["descriptive_family_bootstrap_lower_min"]),
            "passed": primary_comparison["descriptive_family_bootstrap_lower"]
            >= float(thresholds["descriptive_family_bootstrap_lower_min"]),
        },
        {
            "name": "exact_family_signflip_one_sided_p_max",
            "observed": primary_comparison["exact_family_signflip_one_sided_p"],
            "threshold": float(thresholds["exact_family_signflip_one_sided_p_max"]),
            "passed": primary_comparison["exact_family_signflip_one_sided_p"]
            <= float(thresholds["exact_family_signflip_one_sided_p_max"]),
        },
        {
            "name": "wins_plus_ties_min_of_8",
            "observed": int(primary_comparison["wins"] + primary_comparison["ties"]),
            "threshold": int(thresholds["wins_plus_ties_min_of_8"]),
            "passed": int(primary_comparison["wins"] + primary_comparison["ties"])
            >= int(thresholds["wins_plus_ties_min_of_8"]),
        },
        {
            "name": "worst_cell_delta_min",
            "observed": primary_comparison["worst_cell_delta"],
            "threshold": float(thresholds["worst_cell_delta_min"]),
            "passed": primary_comparison["worst_cell_delta"]
            >= float(thresholds["worst_cell_delta_min"]),
        },
    ]
    return {
        "status": "official_frozen_eight_cell_screen",
        "verdict": "PASS" if all(clause["passed"] for clause in clauses) else "FAIL",
        "mechanistic_primary": primary,
        "n_cells": 8,
        "n_dataset_families": 8,
        "clauses": clauses,
        "secondary_variants_cannot_substitute_for_primary": True,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cells", default="screen")
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory must be empty: {args.out_dir}")

    # Pass A1: contracts and hashes for the complete run, still target-free.
    hash_state = _hash_contract_pass(
        config_path=args.config,
        registry_path=args.registry,
        bundle_dir=args.bundle_dir,
        baseline_dir=args.baseline_dir,
        run_dir=args.run_dir,
        cells_raw=args.cells,
    )
    # Pass A2: numerical mechanics for every run artifact, still target-free.
    mechanics = _mechanical_artifact_pass(hash_state)
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before both preflight passes")

    # Phase B: the only point at which label-reading code becomes importable.
    targets, sidecar_audit = _load_targets_after_preflight(
        hash_state, args.sidecar_dir
    )
    cells = hash_state["selected_cells"]
    variants = hash_state["run_variants"]
    per_cell = []
    for cell in cells:
        bundle = hash_state["bundles"][cell]
        target = targets[cell]
        per_cell.append(
            {
                "cell_id": cell,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "method": "B3",
                "n_b3_seeds_averaged": 5,
                **_metrics(target, mechanics["baseline_scores"][cell]),
            }
        )
        for variant in variants:
            per_cell.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": variant,
                    "n_b3_seeds_averaged": 5,
                    **_metrics(target, mechanics["scores"][(variant, cell)]),
                }
            )

    methods = ["B3", *variants]
    summaries = [
        _summary(per_cell, method, metric)
        for method in methods
        for metric in ("auroc", "auprc")
    ]
    cell_hash = canonical_sha256(cells)
    versus_b3 = []
    for variant in variants:
        seed = _stable_seed(
            hash_state["config"]["experiment_id"],
            variant,
            "B3",
            cell_hash,
            "descriptive_family_block_bootstrap_v1",
        )
        versus_b3.append(
            _comparison(
                per_cell,
                variant,
                "B3",
                draws=int(args.bootstrap_draws),
                seed=seed,
            )
        )
    primary = "E1_B3_ORTH_IUPGRD_FULL"
    mechanism_controls = []
    for reference in (
        "E2_B3_ORTH_IUPGRD_HALF",
        "E4_B3_ORTH_FAMILY_PERMUTED_FULL",
        "E5_B3_ORTH_ROW_PERMUTED_FULL",
    ):
        seed = _stable_seed(
            hash_state["config"]["experiment_id"],
            primary,
            reference,
            cell_hash,
            "descriptive_family_block_bootstrap_v1",
        )
        mechanism_controls.append(
            _comparison(
                per_cell,
                primary,
                reference,
                draws=int(args.bootstrap_draws),
                seed=seed,
            )
        )
    primary_vs_b3 = next(row for row in versus_b3 if row["candidate"] == primary)
    verdict = _screen_verdict(
        hash_state["config"],
        primary_vs_b3,
        cells=cells,
        dataset_families=[hash_state["family_by_cell"][cell] for cell in cells],
    )
    mechanism_summary = {
        "primary": primary,
        "controls": [row["reference"] for row in mechanism_controls],
        "comparisons": mechanism_controls,
        "beats_all_controls_on_equal_family_auroc_point_estimate": all(
            row["equal_dataset_family_auroc_delta"] > 0.0
            for row in mechanism_controls
        ),
        "interpretation": (
            "The contrasts are descriptive mechanism checks. E2 tests trust amplitude, "
            "E4 family semantics, and E5 sample alignment; none may substitute for the "
            "registered E1-vs-B3 screen gate."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    _write_csv(args.out_dir / "ARTIFACT_DIAGNOSTICS.csv", mechanics["diagnostics"])
    atomic_write_json(args.out_dir / "SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "COMPARISONS_VS_B3.json", versus_b3)
    atomic_write_json(args.out_dir / "MECHANISM_CONTROLS.json", mechanism_summary)
    atomic_write_json(args.out_dir / "SCREEN_VERDICT.json", verdict)
    report = {
        "schema": "deem_b3_iupgrd_boost_v1_evaluation",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory_not_confirmation",
        "natural_24cell_targets_previously_opened": True,
        "strict_two_pass_preflight_before_dynamic_label_import": True,
        "preflight_passes": [
            "complete_run_contract_and_hash",
            "complete_run_array_and_mechanics",
        ],
        "full_run_preflight_n_cells": len(hash_state["run_cells"]),
        "full_run_preflight_n_score_artifacts": len(hash_state["run_cells"])
        * len(hash_state["run_variants"]),
        "evaluated_cells": cells,
        "evaluated_n_cells": len(cells),
        "evaluated_dataset_families": sorted(
            {hash_state["family_by_cell"][cell] for cell in cells}
        ),
        "variants": variants,
        "baseline": {
            "method": "B3",
            "ensemble": "mean_of_seed_posteriors",
            "seeds": hash_state["run_seeds"],
        },
        "bootstrap_draws": int(args.bootstrap_draws),
        "hashes": hash_state["hashes"],
        "bundle_audit": {
            cell: hash_state["bundle_audits"][cell] for cell in cells
        },
        "sidecar_audit": sidecar_audit,
        "summaries": summaries,
        "comparisons_vs_b3": versus_b3,
        "mechanism_controls": mechanism_summary,
        "screen_verdict": verdict,
        "claim_boundary": (
            "All targets in the natural panel were previously opened. The frozen screen "
            "is an exploratory mechanism test; confirmation requires new unopened "
            "dataset families."
        ),
    }
    atomic_write_json(args.out_dir / "REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
