"""Post-certificate grouped evaluation for causal prefix scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    canonical_tree_manifest,
    deterministic_npz_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from .prefix_contract import (
    AtomicPrefixDirectory,
    BUDGETS,
    EVALUATION_AB_SCHEMA,
    EVALUATION_SCHEMA,
    METHOD_IDS,
    PREPARATION_AB_SCHEMA,
    SCORE_AB_SCHEMA,
    SUBSETS,
    PrefixContractError,
    add_payload_sha256,
    finite_metric,
    load_registry,
    payload_sha256,
    validate_observation_arrays,
    verify_payload,
    write_json_noreplace,
)
from .prefix_ab import authenticate_prefix_score_certificate
from .prefix_fit import SCORES_FILENAME, SCORE_MANIFEST_FILENAME, load_score_manifest
from .prefix_preparation import (
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    load_preparation_manifest,
    load_private_labels,
)


LABELED_SCORES_FILENAME = "LABELED_SCORES.npz"
METRICS_FILENAME = "METRICS.json"
CONTRASTS_FILENAME = "CONTRASTS.json"
BOOTSTRAP_FILENAME = "BOOTSTRAP_DRAWS.npz"
EVALUATION_MANIFEST_FILENAME = "EVALUATION_MANIFEST.json"
EVALUATION_SOURCE_FILES = (
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/prefix_contract.py",
    "spectral_utils/reconstruction_benchmark/prefix_preparation.py",
    "spectral_utils/reconstruction_benchmark/prefix_fit.py",
    "spectral_utils/reconstruction_benchmark/prefix_ab.py",
    "spectral_utils/reconstruction_benchmark/prefix_evaluation.py",
    "scripts/reconstruction_benchmark/evaluate_prefix.py",
    "scripts/reconstruction_benchmark/verify_prefix_evaluation_ab.py",
)


def _evaluation_source_snapshot(
    *, repo: str | Path, registry_path: str | Path
) -> list[dict[str, str]]:
    repo_path = Path(repo).resolve()
    registry = Path(registry_path).resolve()
    try:
        registry_name = registry.relative_to(repo_path).as_posix()
    except ValueError:
        registry_name = str(registry)
    snapshot = [
        {
            "role": "registry",
            "path": registry_name,
            "sha256": sha256_file(registry),
        }
    ]
    for relative in EVALUATION_SOURCE_FILES:
        source = (repo_path / relative).resolve()
        try:
            source.relative_to(repo_path)
        except ValueError as error:
            raise PrefixContractError("prefix evaluator source escapes repo") from error
        snapshot.append(
            {"role": "evaluator", "path": relative, "sha256": sha256_file(source)}
        )
    return snapshot


def _load_certificate(path: Path, *, schema: str, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    verify_payload(value, name=name)
    if value.get("schema_version") != schema or value.get("status") != "PASS":
        raise PrefixContractError(f"{name} is invalid")
    return value


def _load_bound_evaluation_inputs(
    *,
    registry: Mapping[str, Any],
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    build_id: str,
    require_scientific_full: bool,
    authenticated_score_certificate: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
]:
    """Validate the complete public certificate chain before opening labels."""

    lane_root = Path(release_root) / release_id / "prefix"
    preparation_certificate_path = lane_root / "PREPARATION_AB_VERIFICATION.json"
    preparation_certificate = _load_certificate(
        preparation_certificate_path,
        schema=PREPARATION_AB_SCHEMA,
        name="prefix preparation A/B certificate",
    )
    score_certificate_path = lane_root / "SCORE_AB_VERIFICATION.json"
    score_certificate = _load_certificate(
        score_certificate_path,
        schema=SCORE_AB_SCHEMA,
        name="prefix score A/B certificate",
    )
    if (
        score_certificate != authenticated_score_certificate
        or preparation_certificate.get("release_id") != release_id
        or score_certificate.get("release_id") != release_id
        or preparation_certificate.get("independent_source_reconstruction") is not True
        or score_certificate.get("source_asset_roster_sha256")
        != preparation_certificate.get("source_asset_roster_sha256")
        or score_certificate.get("preparation_certificate_sha256")
        != sha256_file(preparation_certificate_path)
        or (
            require_scientific_full
            and (
                preparation_certificate.get("scientific_full_required") is not True
                or score_certificate.get("scientific_full_required") is not True
            )
        )
    ):
        raise PrefixContractError("prefix evaluation certificate chain is stale or incomplete")

    build_root = lane_root / build_id
    preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
    preparation = load_preparation_manifest(preparation_path)
    score_manifest_path = build_root / "fit" / SCORE_MANIFEST_FILENAME
    score_manifest = load_score_manifest(score_manifest_path)
    prep_binding = preparation_certificate.get("builds", {}).get(build_id, {})
    score_binding = score_certificate.get("builds", {}).get(build_id, {})
    if (
        preparation.get("release_id") != release_id
        or preparation.get("build_id") != build_id
        or preparation.get("lane_id") != registry["lane_id"]
        or preparation.get("task_id") != registry["task_id"]
        or preparation.get("population_id") != registry["population"]["population_id"]
        or score_manifest.get("release_id") != release_id
        or score_manifest.get("build_id") != build_id
        or score_manifest.get("lane_id") != registry["lane_id"]
        or score_manifest.get("task_id") != registry["task_id"]
        or prep_binding.get("preparation_manifest_sha256")
        != sha256_file(preparation_path)
        or prep_binding.get("preparation_manifest_payload_sha256")
        != preparation["payload_sha256"]
        or score_binding.get("score_manifest_sha256")
        != sha256_file(score_manifest_path)
        or score_binding.get("score_manifest_payload_sha256")
        != score_manifest["payload_sha256"]
        or score_binding.get("preparation_manifest_sha256")
        != sha256_file(preparation_path)
        or score_manifest.get("preparation_manifest_sha256")
        != sha256_file(preparation_path)
        or score_manifest.get("preparation_ab_certificate_sha256")
        != sha256_file(preparation_certificate_path)
        or score_manifest.get("preparation_ab_certificate_payload_sha256")
        != preparation_certificate["payload_sha256"]
        or score_manifest.get("fit_input_sha256")
        != preparation["fit_input"]["sha256"]
        or score_manifest.get("expected_score_anchor_sha256")
        != preparation["expected_scores"]["sha256"]
        or score_manifest.get("source_snapshot_sha256")
        != score_certificate.get("source_snapshot_sha256")
        or score_manifest.get("claim_boundary") != registry["claim_boundary"]
        or preparation.get("claim_boundary") != registry["claim_boundary"]
        or (
            require_scientific_full
            and (
                preparation.get("scientific_full_build") is not True
                or score_manifest.get("scientific_full_build") is not True
            )
        )
    ):
        raise PrefixContractError(
            f"prefix evaluation build {build_id} is underbound to its certificates"
        )

    score_path = build_root / "fit" / SCORES_FILENAME
    if (
        score_manifest.get("score_artifact", {}).get("path") != SCORES_FILENAME
        or sha256_file(score_path) != score_manifest["score_artifact"]["sha256"]
        or score_manifest["score_artifact"]["sha256"]
        != score_certificate.get("score_artifact_sha256")
        or int(score_certificate.get("observations", -1))
        != int(registry["population"]["expected_prefix_observations"])
        or int(score_certificate.get("method_scores", -1))
        != int(registry["population"]["expected_prefix_observations"])
        * len(METHOD_IDS)
        or score_certificate.get("execution_status")
        != registry["score_anchor"]["required_status"]
    ):
        raise PrefixContractError("prefix evaluation score artifact/certificate binding failed")
    scores = load_npz_no_pickle(score_path)
    validate_observation_arrays(scores, registry=registry, include_scores=True)

    # No private target path is touched before every public score/preparation
    # binding above has passed.
    private_path = (
        Path(private_root) / release_id / "prefix" / build_id / PRIVATE_LABEL_FILENAME
    )
    if (
        Path(preparation["private_labels"].get("path", "")).resolve()
        != private_path.resolve()
        or preparation["private_labels"]["sha256"]
        != preparation_certificate.get("private_label_sha256")
        or sha256_file(private_path) != preparation["private_labels"]["sha256"]
    ):
        raise PrefixContractError("prefix evaluation private-label binding failed")
    labels = load_private_labels(private_path, registry=registry)
    return (
        score_certificate_path,
        score_certificate,
        preparation_path,
        preparation,
        score_manifest_path,
        score_manifest,
        scores,
        private_path,
        labels,
    )


def _safe_metric(metric: str, labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y, s))
    if metric == "auprc":
        return float(average_precision_score(y, s))
    raise PrefixContractError(f"unknown prefix metric: {metric}")


def _percentile(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return None, None, 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _macro_metric(
    *,
    method_id: str,
    budget: int,
    metric: str,
    selected_ids: Mapping[str, Sequence[str]],
    labels: Mapping[str, int],
    scores: Mapping[tuple[str, int, str], float],
) -> tuple[float, dict[str, float | None]]:
    by_family: dict[str, float | None] = {}
    for family in SUBSETS:
        ids = [
            row_id
            for row_id in selected_ids[family]
            if (row_id, budget, method_id) in scores
        ]
        value = _safe_metric(
            metric,
            [labels[row_id] for row_id in ids],
            [scores[(row_id, budget, method_id)] for row_id in ids],
        )
        by_family[family] = finite_metric(value)
    finite = [value for value in by_family.values() if value is not None]
    return (
        float(np.mean(finite)) if len(finite) == len(SUBSETS) else float("nan")
    ), by_family


def _weighted_metric_draws(
    *,
    metric: str,
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate bootstrap duplicates as integer weights without row expansion."""

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.int64)
    if y.ndim != 1 or s.shape != y.shape or w.ndim != 2 or w.shape[1] != len(y):
        raise PrefixContractError("weighted prefix metric received incompatible arrays")
    draws = w.shape[0]
    positive = w @ y.astype(np.int64)
    negative = w.sum(axis=1) - positive
    valid = (positive > 0) & (negative > 0)
    output = np.full(draws, np.nan, dtype=np.float64)
    if not len(y):
        return output
    if metric == "auroc":
        order = np.argsort(s, kind="mergesort")
        sorted_scores, sorted_y, sorted_w = s[order], y[order], w[:, order]
        contribution = np.zeros(draws, dtype=np.float64)
        negative_below = np.zeros(draws, dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            group_w = sorted_w[:, start:stop]
            group_y = sorted_y[start:stop]
            group_positive = group_w @ group_y.astype(np.int64)
            group_negative = group_w.sum(axis=1) - group_positive
            contribution += group_positive * (negative_below + 0.5 * group_negative)
            negative_below += group_negative
            start = stop
        output[valid] = contribution[valid] / (positive[valid] * negative[valid])
        return output
    if metric == "auprc":
        order = np.argsort(-s, kind="mergesort")
        sorted_scores, sorted_y, sorted_w = s[order], y[order], w[:, order]
        cumulative_positive = np.zeros(draws, dtype=np.float64)
        cumulative_total = np.zeros(draws, dtype=np.float64)
        average_precision = np.zeros(draws, dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            group_w = sorted_w[:, start:stop]
            group_y = sorted_y[start:stop]
            group_positive = group_w @ group_y.astype(np.int64)
            group_total = group_w.sum(axis=1)
            cumulative_positive += group_positive
            cumulative_total += group_total
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros(draws, dtype=np.float64),
                where=cumulative_total > 0,
            )
            average_precision += np.divide(
                group_positive * precision,
                positive,
                out=np.zeros(draws, dtype=np.float64),
                where=positive > 0,
            )
            start = stop
        output[valid] = average_precision[valid]
        return output
    raise PrefixContractError(f"unknown prefix metric: {metric}")


def _macro_bootstrap_draws(
    *,
    method_id: str,
    budget: int,
    metric: str,
    ids_by_family: Mapping[str, Sequence[str]],
    bootstrap_counts: Mapping[str, np.ndarray],
    labels: Mapping[str, int],
    scores: Mapping[tuple[str, int, str], float],
) -> np.ndarray:
    per_family = []
    for family in SUBSETS:
        ids = ids_by_family[family]
        eligible_indexes = np.asarray(
            [
                index
                for index, row_id in enumerate(ids)
                if (row_id, budget, method_id) in scores
            ],
            dtype=int,
        )
        family_labels = np.asarray([labels[ids[index]] for index in eligible_indexes], dtype=np.int8)
        family_scores = np.asarray(
            [scores[(ids[index], budget, method_id)] for index in eligible_indexes],
            dtype=np.float64,
        )
        family_weights = np.asarray(bootstrap_counts[family][:, eligible_indexes], dtype=np.int64)
        per_family.append(
            _weighted_metric_draws(
                metric=metric,
                labels=family_labels,
                scores=family_scores,
                weights=family_weights,
            )
        )
    stacked = np.vstack(per_family)
    complete = np.all(np.isfinite(stacked), axis=0)
    output = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    output[complete] = np.mean(stacked[:, complete], axis=0)
    return output


def evaluate_prefix_arrays(
    *,
    score_arrays: Mapping[str, np.ndarray],
    label_bundle: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    validate_observation_arrays(score_arrays, registry=registry, include_scores=True)
    label_rows = label_bundle.get("rows")
    if not isinstance(label_rows, list):
        raise PrefixContractError("prefix private label bundle has no rows")
    labels = {str(row["row_id"]): int(row["label"]) for row in label_rows}
    families = {str(row["row_id"]): str(row["family"]) for row in label_rows}
    lengths = {str(row["row_id"]): int(row["final_length"]) for row in label_rows}
    if len(labels) != len(label_rows) or any(value not in (0, 1) for value in labels.values()):
        raise PrefixContractError("prefix private labels are duplicate or non-binary")
    row_ids = np.asarray(score_arrays["row_id"]).astype(str)
    row_families = np.asarray(score_arrays["family"]).astype(str)
    budgets = np.asarray(score_arrays["budget"], dtype=int)
    score_lookup: dict[tuple[str, int, str], float] = {}
    for index, (row_id, family, budget) in enumerate(
        zip(row_ids, row_families, budgets, strict=True)
    ):
        if row_id not in labels or families[row_id] != family:
            raise PrefixContractError(f"prefix score/label identity mismatch: {row_id}")
        if lengths[row_id] <= int(budget):
            raise PrefixContractError(f"prefix score violates strict unfinished rule: {row_id}@{budget}")
        for method_id in METHOD_IDS:
            score_lookup[(row_id, int(budget), method_id)] = float(score_arrays[method_id][index])
    if set(row_ids) != set(labels):
        raise PrefixContractError("prefix score/label trace union mismatch")
    ids_by_family = {
        family: sorted(row_id for row_id, value in families.items() if value == family)
        for family in SUBSETS
    }
    evaluation = registry["evaluation"]
    n_boot = int(evaluation["bootstrap"]["draws"])
    rng = np.random.default_rng(int(evaluation["bootstrap"]["seed"]))
    bootstrap_counts = {
        family: np.zeros((n_boot, len(ids_by_family[family])), dtype=np.int32)
        for family in SUBSETS
    }
    # Freeze RNG consumption in draw-major, registered-subset order.  The same
    # count matrices feed every method, budget, metric, and contrast.
    for draw_index in range(n_boot):
        for family in SUBSETS:
            ids = ids_by_family[family]
            indexes = rng.integers(0, len(ids), len(ids))
            bootstrap_counts[family][draw_index] = np.bincount(
                indexes, minlength=len(ids)
            )

    metric_draws: dict[tuple[str, int, str], np.ndarray] = {}
    metric_rows = []
    per_subset_rows = []
    for method_id in METHOD_IDS:
        for budget in BUDGETS:
            for metric in evaluation["metrics"]:
                point, by_family = _macro_metric(
                    method_id=method_id,
                    budget=budget,
                    metric=metric,
                    selected_ids=ids_by_family,
                    labels=labels,
                    scores=score_lookup,
                )
                draws = _macro_bootstrap_draws(
                    method_id=method_id,
                    budget=budget,
                    metric=metric,
                    ids_by_family=ids_by_family,
                    bootstrap_counts=bootstrap_counts,
                    labels=labels,
                    scores=score_lookup,
                )
                metric_draws[(method_id, budget, metric)] = draws
                low, high, valid = _percentile(draws)
                metric_rows.append(
                    {
                        "method_id": method_id,
                        "budget": budget,
                        "metric": metric,
                        "aggregation": "equal_subset_macro",
                        "point": finite_metric(point),
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_draws": n_boot,
                        "valid_bootstrap_draws": valid,
                        "families_required": len(SUBSETS),
                        "families_used": sum(value is not None for value in by_family.values()),
                        "families_included": [
                            family for family in SUBSETS if by_family[family] is not None
                        ],
                        "families_excluded": [
                            family for family in SUBSETS if by_family[family] is None
                        ],
                        "missing_subset_policy": evaluation["missing_subset_policy"],
                        "status": (
                            "OK"
                            if math.isfinite(point)
                            else "METRIC_UNDEFINED_MISSING_REGISTERED_SUBSET"
                        ),
                    }
                )
                for family in SUBSETS:
                    eligible = sum(
                        (row_id, budget, method_id) in score_lookup
                        for row_id in ids_by_family[family]
                    )
                    eligible_ids = [
                        row_id
                        for row_id in ids_by_family[family]
                        if (row_id, budget, method_id) in score_lookup
                    ]
                    n_positive = sum(labels[row_id] == 1 for row_id in eligible_ids)
                    per_subset_rows.append(
                        {
                            "method_id": method_id,
                            "budget": budget,
                            "metric": metric,
                            "family": family,
                            "point": by_family[family],
                            "n_traces": eligible,
                            "n_positive": n_positive,
                            "n_negative": eligible - n_positive,
                            "status": "OK" if by_family[family] is not None else "METRIC_UNDEFINED_SINGLE_CLASS",
                        }
                    )

    contrast_rows = []
    contrast_draws: dict[tuple[str, str, int, str], np.ndarray] = {}
    point_lookup = {
        (row["method_id"], int(row["budget"]), row["metric"]): row["point"]
        for row in metric_rows
    }
    for left, right in evaluation["contrasts"]:
        for budget in BUDGETS:
            for metric in evaluation["metrics"]:
                left_draws = metric_draws[(left, budget, metric)]
                right_draws = metric_draws[(right, budget, metric)]
                draws = left_draws - right_draws
                contrast_draws[(left, right, budget, metric)] = draws
                low, high, valid = _percentile(draws)
                left_point = point_lookup[(left, budget, metric)]
                right_point = point_lookup[(right, budget, metric)]
                point = (
                    float(left_point) - float(right_point)
                    if left_point is not None and right_point is not None
                    else None
                )
                contrast_rows.append(
                    {
                        "left_method_id": left,
                        "right_method_id": right,
                        "budget": budget,
                        "metric": metric,
                        "aggregation": "equal_subset_macro",
                        "point_delta": point,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_draws": n_boot,
                        "valid_bootstrap_draws": valid,
                        "paired": True,
                        "resampling_unit": "source_question_within_subset",
                        "missing_subset_policy": evaluation["missing_subset_policy"],
                        "status": (
                            "OK"
                            if point is not None
                            else "METRIC_UNDEFINED_MISSING_REGISTERED_SUBSET"
                        ),
                    }
                )
    draw_arrays: dict[str, np.ndarray] = {}
    for (method_id, budget, metric), values in metric_draws.items():
        draw_arrays[f"metric__{method_id}__b{budget}__{metric}"] = values
    for (left, right, budget, metric), values in contrast_draws.items():
        draw_arrays[f"delta__{left}__minus__{right}__b{budget}__{metric}"] = values
    for family in SUBSETS:
        draw_arrays[f"group_id__{family}"] = np.asarray(ids_by_family[family])
        draw_arrays[f"group_count__{family}"] = np.asarray(
            bootstrap_counts[family], dtype=np.int32
        )
    labeled = {
        "row_id": row_ids,
        "family": row_families,
        "budget": budgets.astype(np.int16),
        "label": np.asarray([labels[row_id] for row_id in row_ids], dtype=np.int8),
        **{method_id: np.asarray(score_arrays[method_id], dtype=np.float64) for method_id in METHOD_IDS},
    }
    return {
        "metrics": metric_rows,
        "per_subset": per_subset_rows,
        "contrasts": contrast_rows,
        "bootstrap_arrays": draw_arrays,
        "labeled_scores": labeled,
    }


def _validate_no_cross_budget_aggregate(result: Mapping[str, Any]) -> None:
    for table in ("metrics", "per_subset", "contrasts"):
        rows = result[table]
        if any(row.get("budget") not in BUDGETS for row in rows):
            raise PrefixContractError(f"prefix {table} contains a cross-budget or unknown slice")
        if any(row.get("aggregation") == "cross_budget_macro" for row in rows):
            raise PrefixContractError(f"prefix {table} contains a forbidden cross-budget macro")


def _evaluation_json_payloads(
    result: Mapping[str, Any], registry: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_payload = add_payload_sha256(
        {
            "schema_version": "reconstruction-prefix-metrics-v1",
            "task_id": registry["task_id"],
            "population_id": registry["population"]["population_id"],
            "aggregation": "equal-subset macro separately at each budget",
            "missing_subset_policy": registry["evaluation"]["missing_subset_policy"],
            "cross_budget_macro": False,
            "rows": result["metrics"],
            "per_subset_rows": result["per_subset"],
        }
    )
    contrast_payload = add_payload_sha256(
        {
            "schema_version": "reconstruction-prefix-contrasts-v1",
            "task_id": registry["task_id"],
            "population_id": registry["population"]["population_id"],
            "paired_bootstrap": registry["evaluation"]["bootstrap"],
            "missing_subset_policy": registry["evaluation"]["missing_subset_policy"],
            "rows": result["contrasts"],
        }
    )
    return metrics_payload, contrast_payload


def evaluate_prefix_build(
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
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "prefix"
    build_root = lane_root / build_id
    authenticated_score_certificate = authenticate_prefix_score_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=scientific_full,
    )
    (
        score_certificate_path,
        _score_certificate,
        _preparation_path,
        _preparation,
        score_manifest_path,
        _score_manifest,
        scores,
        private_path,
        labels,
    ) = _load_bound_evaluation_inputs(
        registry=registry,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        build_id=build_id,
        require_scientific_full=scientific_full,
        authenticated_score_certificate=authenticated_score_certificate,
    )
    result = evaluate_prefix_arrays(score_arrays=scores, label_bundle=labels, registry=registry)
    _validate_no_cross_budget_aggregate(result)
    evaluation_source_snapshot = _evaluation_source_snapshot(
        repo=repo, registry_path=registry_path
    )
    evaluation_root = build_root / "evaluation"
    evaluation_stage = AtomicPrefixDirectory(evaluation_root)
    try:
        labeled_sha = atomic_write_npz(
            evaluation_stage.path / LABELED_SCORES_FILENAME,
            result["labeled_scores"],
        )
        draws_sha = atomic_write_npz(
            evaluation_stage.path / BOOTSTRAP_FILENAME,
            result["bootstrap_arrays"],
        )
        metrics_payload, contrast_payload = _evaluation_json_payloads(result, registry)
        metrics_sha = atomic_write_json(
            evaluation_stage.path / METRICS_FILENAME, metrics_payload
        )
        contrasts_sha = atomic_write_json(
            evaluation_stage.path / CONTRASTS_FILENAME, contrast_payload
        )
        manifest = add_payload_sha256(
            {
                "schema_version": EVALUATION_SCHEMA,
                "release_id": release_id,
                "build_id": build_id,
                "scientific_full_build": bool(scientific_full),
                "lane_id": registry["lane_id"],
                "task_id": registry["task_id"],
                "score_ab_certificate_sha256": sha256_file(score_certificate_path),
                "score_ab_certificate_payload_sha256": _score_certificate[
                    "payload_sha256"
                ],
                "score_manifest_sha256": sha256_file(score_manifest_path),
                "score_manifest_payload_sha256": _score_manifest["payload_sha256"],
                "private_label_sha256": sha256_file(private_path),
                "evaluation_source_snapshot": evaluation_source_snapshot,
                "evaluation_source_snapshot_sha256": payload_sha256(
                    evaluation_source_snapshot
                ),
                "artifacts": {
                    LABELED_SCORES_FILENAME: labeled_sha,
                    METRICS_FILENAME: metrics_sha,
                    CONTRASTS_FILENAME: contrasts_sha,
                    BOOTSTRAP_FILENAME: draws_sha,
                },
                "metric_rows": len(result["metrics"]),
                "per_subset_rows": len(result["per_subset"]),
                "contrast_rows": len(result["contrasts"]),
                "bootstrap_draws": int(registry["evaluation"]["bootstrap"]["draws"]),
                "missing_subset_policy": registry["evaluation"]["missing_subset_policy"],
                "labels_opened_after_score_ab": True,
                "causal_early_scoring_only": True,
                "stopping_claim_allowed": False,
                "cross_budget_macro_allowed": False,
                "cross_task_macro_allowed": False,
                "claim_boundary": registry["claim_boundary"],
            }
        )
        atomic_write_json(
            evaluation_stage.path / EVALUATION_MANIFEST_FILENAME, manifest
        )
        atomic_write_json(
            evaluation_stage.path / "TREE_MANIFEST.json",
            canonical_tree_manifest(evaluation_stage.path),
        )
        evaluation_stage.commit()
        return manifest
    finally:
        evaluation_stage.cleanup()


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload(value, name="prefix evaluation manifest")
    if set(value) != {
        "schema_version",
        "release_id",
        "build_id",
        "scientific_full_build",
        "lane_id",
        "task_id",
        "score_ab_certificate_sha256",
        "score_ab_certificate_payload_sha256",
        "score_manifest_sha256",
        "score_manifest_payload_sha256",
        "private_label_sha256",
        "evaluation_source_snapshot",
        "evaluation_source_snapshot_sha256",
        "artifacts",
        "metric_rows",
        "per_subset_rows",
        "contrast_rows",
        "bootstrap_draws",
        "missing_subset_policy",
        "labels_opened_after_score_ab",
        "causal_early_scoring_only",
        "stopping_claim_allowed",
        "cross_budget_macro_allowed",
        "cross_task_macro_allowed",
        "claim_boundary",
        "payload_sha256",
    }:
        raise PrefixContractError("prefix evaluation manifest field roster drifted")
    if value.get("schema_version") != EVALUATION_SCHEMA:
        raise PrefixContractError("unexpected prefix evaluation schema")
    if (
        value.get("labels_opened_after_score_ab") is not True
        or value.get("build_id") not in {"A", "B"}
        or type(value.get("scientific_full_build")) is not bool
        or value.get("causal_early_scoring_only") is not True
        or value.get("stopping_claim_allowed") is not False
        or value.get("cross_budget_macro_allowed") is not False
        or value.get("cross_task_macro_allowed") is not False
        or value.get("evaluation_source_snapshot_sha256")
        != payload_sha256(value.get("evaluation_source_snapshot", []))
    ):
        raise PrefixContractError("prefix evaluation claim boundary failed")
    return value


def verify_prefix_evaluation_ab(
    *,
    repo: str | Path,
    registry_path: str | Path,
    release_root: str | Path,
    private_root: str | Path,
    release_id: str,
    source_root: str | Path,
    output_path: str | Path | None = None,
    require_scientific_full: bool,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    expected_source_snapshot = _evaluation_source_snapshot(
        repo=repo, registry_path=registry_path
    )
    expected_source_snapshot_sha256 = payload_sha256(expected_source_snapshot)
    lane_root = Path(release_root) / release_id / "prefix"
    score_certificate_path = lane_root / "SCORE_AB_VERIFICATION.json"
    score_certificate = authenticate_prefix_score_certificate(
        repo=repo,
        registry_path=registry_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        source_root=source_root,
        require_scientific_full=require_scientific_full,
    )
    manifests = {}
    for build_id in ("A", "B"):
        (
            bound_score_certificate_path,
            bound_score_certificate,
            _preparation_path,
            _preparation,
            score_manifest_path,
            score_manifest,
            scores,
            private_path,
            labels,
        ) = _load_bound_evaluation_inputs(
            registry=registry,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            build_id=build_id,
            require_scientific_full=require_scientific_full,
            authenticated_score_certificate=score_certificate,
        )
        if bound_score_certificate_path != score_certificate_path:
            raise PrefixContractError("prefix evaluation score certificate path drifted")
        result = evaluate_prefix_arrays(
            score_arrays=scores, label_bundle=labels, registry=registry
        )
        _validate_no_cross_budget_aggregate(result)
        metrics_expected, contrasts_expected = _evaluation_json_payloads(result, registry)
        expected_bytes = {
            LABELED_SCORES_FILENAME: deterministic_npz_bytes(result["labeled_scores"]),
            BOOTSTRAP_FILENAME: deterministic_npz_bytes(result["bootstrap_arrays"]),
            METRICS_FILENAME: canonical_json_bytes(metrics_expected) + b"\n",
            CONTRASTS_FILENAME: canonical_json_bytes(contrasts_expected) + b"\n",
        }
        root = lane_root / build_id / "evaluation"
        manifest_path = root / EVALUATION_MANIFEST_FILENAME
        manifest = load_evaluation_manifest(manifest_path)
        if manifest.get("release_id") != release_id or manifest.get("build_id") != build_id:
            raise PrefixContractError(f"prefix evaluation {build_id} release/build binding failed")
        if require_scientific_full and manifest.get("scientific_full_build") is not True:
            raise PrefixContractError(f"prefix evaluation {build_id} is not scientific-full")
        if (
            manifest.get("lane_id") != registry["lane_id"]
            or manifest.get("task_id") != registry["task_id"]
            or manifest.get("score_ab_certificate_sha256")
            != sha256_file(score_certificate_path)
            or manifest.get("score_ab_certificate_payload_sha256")
            != bound_score_certificate["payload_sha256"]
            or manifest.get("score_manifest_sha256")
            != sha256_file(score_manifest_path)
            or manifest.get("score_manifest_payload_sha256")
            != score_manifest["payload_sha256"]
            or manifest.get("private_label_sha256") != sha256_file(private_path)
            or manifest.get("evaluation_source_snapshot")
            != expected_source_snapshot
            or manifest.get("evaluation_source_snapshot_sha256")
            != expected_source_snapshot_sha256
            or manifest.get("missing_subset_policy")
            != registry["evaluation"]["missing_subset_policy"]
            or manifest.get("claim_boundary") != registry["claim_boundary"]
            or set(manifest.get("artifacts", {})) != set(expected_bytes)
            or int(manifest.get("metric_rows", -1)) != len(result["metrics"])
            or int(manifest.get("per_subset_rows", -1)) != len(result["per_subset"])
            or int(manifest.get("contrast_rows", -1)) != len(result["contrasts"])
            or int(manifest.get("bootstrap_draws", -1))
            != int(registry["evaluation"]["bootstrap"]["draws"])
        ):
            raise PrefixContractError(
                f"prefix evaluation {build_id} lost exact score/label/roster binding"
            )
        for filename, payload in expected_bytes.items():
            path = root / filename
            if (
                path.read_bytes() != payload
                or manifest["artifacts"].get(filename) != sha256_bytes(payload)
            ):
                raise PrefixContractError(
                    f"prefix evaluation {build_id} differs from independent rederivation: {filename}"
                )
        manifests[build_id] = manifest
    comparison = {
        filename: manifests["A"]["artifacts"][filename]
        == manifests["B"]["artifacts"][filename]
        for filename in (
            LABELED_SCORES_FILENAME,
            METRICS_FILENAME,
            CONTRASTS_FILENAME,
            BOOTSTRAP_FILENAME,
        )
    }
    comparison.update(
        {
            "metric_rows": manifests["A"]["metric_rows"] == manifests["B"]["metric_rows"],
            "per_subset_rows": manifests["A"]["per_subset_rows"]
            == manifests["B"]["per_subset_rows"],
            "contrast_rows": manifests["A"]["contrast_rows"] == manifests["B"]["contrast_rows"],
            "bootstrap_draws": manifests["A"]["bootstrap_draws"]
            == manifests["B"]["bootstrap_draws"]
            == int(registry["evaluation"]["bootstrap"]["draws"]),
            "private_label_sha256": manifests["A"]["private_label_sha256"]
            == manifests["B"]["private_label_sha256"],
            "evaluation_source_snapshot": manifests["A"][
                "evaluation_source_snapshot_sha256"
            ]
            == manifests["B"]["evaluation_source_snapshot_sha256"]
            == expected_source_snapshot_sha256,
        }
    )
    if not all(comparison.values()):
        raise PrefixContractError(
            f"prefix A/B evaluation verification failed: "
            f"{[name for name, ok in comparison.items() if not ok]}"
        )
    certificate = add_payload_sha256(
        {
            "schema_version": EVALUATION_AB_SCHEMA,
            "release_id": release_id,
            "status": "PASS",
            "scientific_full_required": bool(require_scientific_full),
            "score_ab_certificate_sha256": sha256_file(score_certificate_path),
            "score_ab_certificate_payload_sha256": score_certificate["payload_sha256"],
            "comparison": comparison,
            "artifacts": manifests["A"]["artifacts"],
            "metric_rows": manifests["A"]["metric_rows"],
            "per_subset_rows": manifests["A"]["per_subset_rows"],
            "contrast_rows": manifests["A"]["contrast_rows"],
            "bootstrap_draws": manifests["A"]["bootstrap_draws"],
            "evaluation_source_snapshot_sha256": expected_source_snapshot_sha256,
            "causal_early_scoring_only": True,
            "stopping_claim_allowed": False,
            "cross_budget_macro_allowed": False,
            "cross_task_macro_allowed": False,
            "missing_subset_policy": registry["evaluation"]["missing_subset_policy"],
            "independent_rederivation": True,
        }
    )
    output = Path(output_path) if output_path else lane_root / "EVALUATION_AB_VERIFICATION.json"
    write_json_noreplace(output, certificate)
    return certificate


__all__ = [
    "BOOTSTRAP_FILENAME",
    "CONTRASTS_FILENAME",
    "EVALUATION_MANIFEST_FILENAME",
    "EVALUATION_SOURCE_FILES",
    "LABELED_SCORES_FILENAME",
    "METRICS_FILENAME",
    "evaluate_prefix_arrays",
    "evaluate_prefix_build",
    "load_evaluation_manifest",
    "verify_prefix_evaluation_ab",
]
