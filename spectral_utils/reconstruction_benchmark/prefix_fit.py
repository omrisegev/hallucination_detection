"""CPU recomputation of the frozen three-system causal-prefix roster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import run_global_local_online_architecture_v2 as step272
from spectral_utils.fair_comparisons.twentyfour import validate_unified28_model
from spectral_utils.multitask_trajectory import truncate_row
from spectral_utils.online_convergence import prefix_method_scores

from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_tree_manifest,
    load_npz_no_pickle,
    sha256_file,
)
from .prefix_contract import (
    AtomicPrefixDirectory,
    BUDGETS,
    METHOD_IDS,
    SCORE_FREEZE_SCHEMA,
    SUBSETS,
    PrefixContractError,
    add_payload_sha256,
    load_registry,
    payload_sha256,
    validate_observation_arrays,
    verify_payload,
)
from .prefix_preparation import (
    EXPECTED_SCORE_FILENAME,
    FIT_INPUT_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    load_fit_input,
    load_preparation_manifest,
)


SCORES_FILENAME = "SCORES.npz"
SCORE_MANIFEST_FILENAME = "SCORE_FREEZE_MANIFEST.json"
SCORE_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/prefix.json",
    "spectral_utils/reconstruction_benchmark/prefix_contract.py",
    "spectral_utils/reconstruction_benchmark/prefix_preparation.py",
    "spectral_utils/reconstruction_benchmark/prefix_fit.py",
    "spectral_utils/reconstruction_benchmark/prefix_ab.py",
    "scripts/reconstruction_benchmark/verify_prefix_preparation_ab.py",
    "scripts/reconstruction_benchmark/run_prefix_methods.py",
    "scripts/reconstruction_benchmark/verify_prefix_ab.py",
    "scripts/run_global_local_online_architecture_v2.py",
    "spectral_utils/fair_comparisons/prefix.py",
    "spectral_utils/fair_comparisons/twentyfour.py",
    "spectral_utils/multitask_trajectory.py",
    "spectral_utils/online_convergence.py",
)


def _rows_by_id(rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for family in SUBSETS:
        for row in rows_by_family[family]:
            row_id = str(row["row_id"])
            if row_id in output:
                raise PrefixContractError(f"duplicate fit-visible row ID: {row_id}")
            output[row_id] = row
    return output


def _unified_scores(
    rows: Mapping[str, Mapping[str, Any]],
    keys: Sequence[tuple[str, int]],
    model: Any,
) -> dict[tuple[str, int], float]:
    validate_unified28_model(model)
    output = {}
    for row_id, budget in keys:
        row = rows[row_id]
        if len(row["token_entropies"]) <= budget:
            raise PrefixContractError(f"Unified-28 received a completed/nonexistent prefix: {row_id}@{budget}")
        score = float(model.score_row(truncate_row(row, budget)).global_score)
        if not np.isfinite(score):
            raise PrefixContractError(f"Unified-28 emitted a non-finite score: {row_id}@{budget}")
        output[(row_id, budget)] = score
    return output


def _iu_scores(
    rows: Mapping[str, Mapping[str, Any]],
    keys: Sequence[tuple[str, int]],
    models: Mapping[str, Any],
) -> dict[tuple[str, int], float]:
    if tuple(models) != SUBSETS:
        raise PrefixContractError("IU28 family-model roster drifted")
    output = {}
    for row_id, budget in keys:
        row = rows[row_id]
        if len(row["token_entropies"]) <= budget:
            raise PrefixContractError(f"IU28 received a completed/nonexistent prefix: {row_id}@{budget}")
        score = float(
            prefix_method_scores(
                row,
                budget,
                {METHOD_IDS[1]: models[str(row["family"])]},
            )[METHOD_IDS[1]]
        )
        if not np.isfinite(score):
            raise PrefixContractError(f"IU28 emitted a non-finite score: {row_id}@{budget}")
        output[(row_id, budget)] = score
    return output


def _step272_scores(
    rows_by_family: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    selection: Mapping[str, str],
) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    expected_selection = {
        "global": "g_registered_mixed",
        "local": "l_level9",
        "online": "o_ewma_area_persist27",
    }
    if dict(selection) != expected_selection:
        raise PrefixContractError("Step272 selected-head contract drifted")
    output: dict[tuple[str, int], float] = {}
    audits: dict[str, Any] = {}
    for family in SUBSETS:
        rows = list(rows_by_family[family])
        calibration = [row for row in rows if row["partition"] == "calibration"]
        evaluation = [row for row in rows if row["partition"] == "evaluation"]
        if not calibration or not evaluation:
            raise PrefixContractError(f"Step272 {family} calibration/evaluation split is empty")
        models = step272._fit_selected_cell(calibration, selection)
        cal_output = step272._selected_outputs(calibration, models, selection)
        eval_output = step272._selected_outputs(evaluation, models, selection)
        cal_global_fit = step272._zfit(cal_output["global_final"])
        cal_local_max = np.asarray(
            [float(np.max(curve)) for curve in cal_output["local_curves"]], dtype=float
        )
        cal_local_fit = step272._zfit(cal_local_max)
        emitted = 0
        for budget in BUDGETS:
            values = (
                0.5 * step272._zapply(eval_output["global_prefix"][budget], cal_global_fit)
                + 0.5 * step272._zapply(eval_output["local_prefix"][budget], cal_local_fit)
            )
            for row, score in zip(evaluation, values, strict=True):
                if len(row["token_entropies"]) <= budget:
                    if np.isfinite(float(score)):
                        raise PrefixContractError(
                            f"Step272 exposed a score beyond the unfinished prefix: {row['row_id']}@{budget}"
                        )
                    continue
                value = float(score)
                if not np.isfinite(value):
                    raise PrefixContractError(f"Step272 emitted a non-finite score: {row['row_id']}@{budget}")
                key = (str(row["row_id"]), budget)
                if key in output:
                    raise PrefixContractError(f"duplicate Step272 score: {key}")
                output[key] = value
                emitted += 1
        audits[family] = {
            "calibration_rows": len(calibration),
            "evaluation_rows": len(evaluation),
            "emitted_prefix_scores": emitted,
            "global_fit": list(map(float, cal_global_fit)),
            "local_fit": list(map(float, cal_local_fit)),
            "labels_seen_during_fit": False,
            "future_tokens_used_for_scored_trace": False,
        }
    return output, audits


def recompute_prefix_scores(
    fit_input: Mapping[str, Any],
    expected: Mapping[str, np.ndarray],
    registry: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validate_observation_arrays(expected, registry=registry, include_scores=True)
    row_ids = np.asarray(expected["row_id"]).astype(str)
    budgets = np.asarray(expected["budget"], dtype=int)
    keys = list(zip(row_ids.tolist(), budgets.tolist(), strict=True))
    rows_by_family = fit_input["rows_by_family"]
    rows = _rows_by_id(rows_by_family)
    missing = [row_id for row_id in row_ids if row_id not in rows]
    if missing:
        raise PrefixContractError(f"prepared prefix inputs lack registered rows: {missing[:3]}")
    frozen = fit_input["frozen_models"]
    unified = _unified_scores(rows, keys, frozen["unified28"])
    iu = _iu_scores(rows, keys, frozen["iu28_no_length"])
    step_method = next(row for row in registry["method_roster"] if row["method_id"] == METHOD_IDS[2])
    step, step_audit = _step272_scores(
        rows_by_family, selection=step_method["frozen_heads"]
    )
    score_maps = {METHOD_IDS[0]: unified, METHOD_IDS[1]: iu, METHOD_IDS[2]: step}
    if any(set(values) != set(keys) for values in score_maps.values()):
        detail = {method: (len(values), len(set(keys).difference(values))) for method, values in score_maps.items()}
        raise PrefixContractError(f"recomputed prefix key coverage failed: {detail}")
    arrays: dict[str, np.ndarray] = {
        "row_id": np.asarray(expected["row_id"]),
        "family": np.asarray(expected["family"]),
        "budget": np.asarray(expected["budget"], dtype=np.int16),
    }
    anchor_audit: dict[str, Any] = {}
    atol = float(registry["score_anchor"]["absolute_tolerance"])
    for method_id in METHOD_IDS:
        values = np.asarray([score_maps[method_id][key] for key in keys], dtype=np.float64)
        expected_values = np.asarray(expected[method_id], dtype=np.float64)
        difference = np.abs(values - expected_values)
        maximum = float(np.max(difference))
        if maximum > atol:
            index = int(np.argmax(difference))
            raise PrefixContractError(
                f"{method_id} CPU recomputation missed its signed anchor: "
                f"max_abs={maximum:.17g} at {keys[index]}, tolerance={atol:.3g}"
            )
        arrays[method_id] = values
        anchor_audit[method_id] = {
            "execution_mode": next(
                row["execution_mode"]
                for row in registry["method_roster"]
                if row["method_id"] == method_id
            ),
            "observations": len(values),
            "max_abs_score_difference": maximum,
            "exact_float_identity": bool(np.array_equal(values, expected_values)),
            "absolute_tolerance": atol,
            "status": registry["score_anchor"]["required_status"],
            "historical_score_rebind": False,
        }
    validate_observation_arrays(arrays, registry=registry, include_scores=True)
    return arrays, {"anchors": anchor_audit, "step272_fit": step_audit}


def run_prefix_methods(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    build_id: str,
    source_root: str | Path,
    scientific_full: bool,
) -> dict[str, Any]:
    if build_id not in {"A", "B"}:
        raise PrefixContractError("prefix build must be A or B")
    repo_path = Path(repo).resolve()
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "prefix"
    build_root = lane_root / build_id
    preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
    preparation = load_preparation_manifest(preparation_path)
    if preparation.get("release_id") != release_id or preparation.get("build_id") != build_id:
        raise PrefixContractError("prefix preparation release/build binding failed")
    # Local import avoids the prefix_ab -> prefix_fit module cycle.  The score
    # executor must not treat a self-hashed prep certificate as a trust root.
    from .prefix_ab import authenticate_prefix_preparation_certificate

    prep_certificate = authenticate_prefix_preparation_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=scientific_full,
    )
    prep_builds = prep_certificate.get("builds")
    if (
        not isinstance(prep_builds, Mapping)
        or not isinstance(prep_builds.get(build_id), Mapping)
        or prep_builds[build_id].get("preparation_manifest_sha256")
        != sha256_file(preparation_path)
        or prep_builds[build_id].get("preparation_manifest_payload_sha256")
        != preparation["payload_sha256"]
        or prep_certificate.get("core_input_sha256")
        != preparation["fit_input"]["sha256"]
        or prep_certificate.get("expected_score_anchor_sha256")
        != preparation["expected_scores"]["sha256"]
        or prep_certificate.get("source_binding_sha256")
        != preparation["source_binding_sha256"]
        or prep_certificate.get("source_asset_roster_sha256")
        != preparation["source_binding"].get("asset_roster_sha256")
        or prep_certificate.get("private_label_sha256")
        != preparation["private_labels"]["sha256"]
        or (scientific_full and prep_certificate.get("scientific_full_required") is not True)
        or (scientific_full and preparation.get("scientific_full_build") is not True)
    ):
        raise PrefixContractError("prefix fit is underbound to its preparation A/B certificate")
    inputs = build_root / "inputs"
    fit_path = inputs / FIT_INPUT_FILENAME
    expected_path = inputs / EXPECTED_SCORE_FILENAME
    if (
        sha256_file(fit_path) != preparation["fit_input"]["sha256"]
        or sha256_file(expected_path) != preparation["expected_scores"]["sha256"]
    ):
        raise PrefixContractError("prefix prepared input hash failed")
    fit_input = load_fit_input(fit_path, registry=registry)
    expected = load_npz_no_pickle(expected_path)
    scores, audit = recompute_prefix_scores(fit_input, expected, registry)
    fit_root = build_root / "fit"
    snapshot = [
        {"path": relative, "sha256": sha256_file(repo_path / relative)}
        for relative in SCORE_SOURCE_FILES
    ]
    fit_stage = AtomicPrefixDirectory(fit_root)
    try:
        score_sha = atomic_write_npz(fit_stage.path / SCORES_FILENAME, scores)
        manifest = add_payload_sha256(
            {
                "schema_version": SCORE_FREEZE_SCHEMA,
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": bool(scientific_full),
                "lane_id": registry["lane_id"],
                "task_id": registry["task_id"],
                "preparation_manifest_sha256": sha256_file(preparation_path),
                "preparation_ab_certificate_sha256": sha256_file(
                    lane_root / "PREPARATION_AB_VERIFICATION.json"
                ),
                "preparation_ab_certificate_payload_sha256": prep_certificate["payload_sha256"],
                "fit_input_sha256": sha256_file(fit_path),
                "expected_score_anchor_sha256": sha256_file(expected_path),
                "score_artifact": {
                    "path": SCORES_FILENAME,
                    "sha256": score_sha,
                    "observations": int(len(scores["row_id"])),
                    "method_scores": int(len(scores["row_id"]) * len(METHOD_IDS)),
                },
                "recomputation_audit": audit,
                "fit_visible_targets": False,
                "future_tokens_used_for_scored_trace": False,
                "historical_scores_are_execution_substitute": False,
                "execution_status": registry["score_anchor"]["required_status"],
                "claim_boundary": registry["claim_boundary"],
                "source_snapshot": snapshot,
                "source_snapshot_sha256": payload_sha256(snapshot),
            }
        )
        atomic_write_json(fit_stage.path / SCORE_MANIFEST_FILENAME, manifest)
        atomic_write_json(
            fit_stage.path / "TREE_MANIFEST.json",
            canonical_tree_manifest(fit_stage.path),
        )
        fit_stage.commit()
        return manifest
    finally:
        fit_stage.cleanup()


def load_score_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload(value, name="prefix score freeze")
    if set(value) != {
        "schema_version",
        "release_id",
        "build_id",
        "scientific_full_build",
        "lane_id",
        "task_id",
        "preparation_manifest_sha256",
        "preparation_ab_certificate_sha256",
        "preparation_ab_certificate_payload_sha256",
        "fit_input_sha256",
        "expected_score_anchor_sha256",
        "score_artifact",
        "recomputation_audit",
        "fit_visible_targets",
        "future_tokens_used_for_scored_trace",
        "historical_scores_are_execution_substitute",
        "execution_status",
        "claim_boundary",
        "source_snapshot",
        "source_snapshot_sha256",
        "payload_sha256",
    }:
        raise PrefixContractError("prefix score-freeze field roster drifted")
    if value.get("schema_version") != SCORE_FREEZE_SCHEMA:
        raise PrefixContractError("unexpected prefix score-freeze schema")
    if (
        value.get("build_id") not in {"A", "B"}
        or type(value.get("scientific_full_build")) is not bool
        or value.get("fit_visible_targets") is not False
        or value.get("future_tokens_used_for_scored_trace") is not False
        or value.get("historical_scores_are_execution_substitute") is not False
        or value.get("execution_status") != "CPU_RECOMPUTED_AND_ANCHOR_VERIFIED"
        or value.get("source_snapshot_sha256")
        != payload_sha256(value.get("source_snapshot", []))
    ):
        raise PrefixContractError("prefix score-freeze execution/causality boundary failed")
    return value


__all__ = [
    "SCORE_SOURCE_FILES",
    "SCORES_FILENAME",
    "SCORE_MANIFEST_FILENAME",
    "load_score_manifest",
    "recompute_prefix_scores",
    "run_prefix_methods",
]
