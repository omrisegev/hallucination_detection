#!/usr/bin/env python3
"""Run supervised conditional metric discovery on outcome-opened Global cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cross_dataset_hallucination_manifold_v1 import (  # noqa: E402
    EXPECTED_COMMON,
    FAMILY_NAMES,
    family,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.graph_topology import (  # noqa: E402
    exact_length_permutations,
    extended_graph_diagnostics,
    propensity_crt_permutations,
    self_safe_knn_graph,
    smoothness_against_permutations,
)
from spectral_utils.laplacian_upcr import laplacian_iu_path  # noqa: E402
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402
from spectral_utils.supervised_manifold_discovery import (  # noqa: E402
    FIT_SEEDS,
    K_SENSITIVITY,
    SUPPORT_SIZES,
    TIE_SEEDS,
    conditional_residual_smoothness,
    deterministic_subsample,
    fit_balanced_logistic,
    fit_metric_ensemble,
    graph_is_healthy,
    median_pairwise_cosine,
    median_pairwise_jaccard,
    metric_matrix,
    select_label_free_graph,
    stable_seed,
    support_indices,
    target_blind_tie_keys,
)


VERSION = "supervised-conditional-manifold-discovery-v1-2026-08-20"
PROTOCOL = ROOT / "docs/experiments/SUPERVISED_CONDITIONAL_MANIFOLD_DISCOVERY_V1.md"
DEFAULT_OUT = ROOT / "results/supervised_conditional_manifold_discovery_v1"
DEFAULT_BUNDLE = ROOT / "results/dependency_fusion_raw/cells.npz"
PERMUTATIONS = 199
DEFAULT_NULL_RERUNS = 199
MAXT_MAX_ROWS = 384
BOOTSTRAPS = 5000
EXACT_MOVABLE_FRACTION_MIN = 0.20
EXACT_MOVABLE_ROWS_MIN = 20
EXACT_MIXED_STRATA_MIN = 5
CRT_OVERLAP_FRACTION_MIN = 0.20
CRT_BRIER_TOLERANCE = 0.01
CRT_CALIBRATION_MAE_MAX = 0.10
ELIGIBLE_ENVIRONMENT_FRACTION_MIN = 2 / 3
POSITIVE_ENVIRONMENT_FRACTION_MIN = 2 / 3
MEDIAN_CONDITIONAL_EFFECT_MIN = 0.02
HEALTHY_CELL_FRACTION_MIN = 0.90
WEIGHT_COSINE_MIN = 0.80
SUPPORT_JACCARD_MIN = 0.60
FEATURE_SELECTION_FREQUENCY_MIN = 0.70
DISTINCT_ADVANTAGE_MIN = 0.02
UTILITY_DELTA_MIN = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _json_number(value: float):
    value = float(value)
    return value if np.isfinite(value) else None


def load_discovery_cells(bundle_path: Path = DEFAULT_BUNDLE) -> tuple[list[dict], tuple[str, ...]]:
    """Load exactly the registered explicit-length Global discovery cells."""

    with np.load(bundle_path, allow_pickle=True) as bundle:
        loaded = []
        for identifier in sorted(key[:-3] for key in bundle.files if key.endswith("__V")):
            if identifier not in set(INSCOPE):
                continue
            stored = np.asarray(bundle[f"{identifier}__V"], dtype=float)
            pool = tuple(str(item) for item in bundle[f"{identifier}__pool"])
            signs = np.asarray(bundle[f"{identifier}__hand_signs"], dtype=float)
            matrix, stored_names = fixed_stable_from_bundle(stored, pool, signs)
            if "trace_length" not in stored_names:
                continue
            loaded.append({
                "cell": identifier,
                "matrix": matrix,
                "names": tuple(str(item) for item in stored_names),
                "length": stored[:, pool.index("trace_length")],
                "target": 1 - np.asarray(bundle[f"{identifier}__labels"], dtype=int),
            })
    common = None
    for cell in loaded:
        allowed = {name for name in cell["names"] if "length" not in name.lower()}
        common = allowed if common is None else common.intersection(allowed)
    names = tuple(sorted(common or ()))
    if names != tuple(EXPECTED_COMMON):
        raise RuntimeError(f"discovery feature contract changed: {names} != {EXPECTED_COMMON}")
    output = []
    for cell in loaded:
        lookup = {name: index for index, name in enumerate(cell["names"])}
        matrix = np.asarray(cell["matrix"][:, [lookup[name] for name in names]], dtype=float)
        target = np.asarray(cell["target"], dtype=int)
        length = np.asarray(cell["length"], dtype=float)
        if matrix.shape != (len(target), len(names)) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid registered matrix for {cell['cell']}")
        output.append({
            "cell": str(cell["cell"]),
            "family": family(str(cell["cell"])),
            "X": matrix,
            "y": target,
            "length": length,
        })
    output.sort(key=lambda value: value["cell"])
    observed_families = {cell["family"] for cell in output}
    if observed_families != set(FAMILY_NAMES):
        raise RuntimeError(f"dataset-family roster changed: {sorted(observed_families)}")
    if len(output) != 21:
        raise RuntimeError(f"expected 21 explicit-length discovery cells, got {len(output)}")
    return output, names


def candidate_definitions(
    weights: np.ndarray,
    names: tuple[str, ...],
) -> list[dict]:
    equal = np.full(len(names), 1.0 / len(names))
    output = [{
        "candidate": "equal_all",
        "role": "no_label_baseline",
        "weights": equal,
        "support": np.arange(len(names), dtype=int),
        "support_size": len(names),
    }]
    for size in SUPPORT_SIZES:
        support = support_indices(weights, names, size)
        token = "all" if size is None else str(int(size))
        output.append({
            "candidate": f"supervised_s{token}",
            "role": "supervised_metric",
            "weights": np.asarray(weights, dtype=float),
            "support": support,
            "support_size": len(support),
        })
    return output


def exact_eligible(diagnostics: dict) -> bool:
    return bool(
        diagnostics["movable_fraction"] >= EXACT_MOVABLE_FRACTION_MIN
        and diagnostics["movable_rows"] >= EXACT_MOVABLE_ROWS_MIN
        and diagnostics["mixed_strata"] >= EXACT_MIXED_STRATA_MIN
    )


def crt_eligible(diagnostics: dict) -> bool:
    return bool(
        diagnostics["overlap_fraction"] >= CRT_OVERLAP_FRACTION_MIN
        and diagnostics["brier"] <= diagnostics["constant_brier"] + CRT_BRIER_TOLERANCE
        and diagnostics["calibration_mae"] <= CRT_CALIBRATION_MAE_MAX
        and diagnostics["all_draws_binary"]
    )


def prepare_conditional_nulls(
    cells: list[dict],
    *,
    permutations: int,
    targets: dict[str, np.ndarray] | None = None,
    namespace: str = "observed",
) -> dict[str, dict]:
    output = {}
    for cell in cells:
        identifier = cell["cell"]
        target = np.asarray(targets[identifier] if targets is not None else cell["y"], dtype=int)
        exact, exact_diag = exact_length_permutations(
            target,
            cell["length"],
            permutations=int(permutations),
            seed=stable_seed(VERSION, namespace, identifier, "exact"),
        )
        crt, crt_diag = propensity_crt_permutations(
            target,
            cell["length"],
            permutations=int(permutations),
            seed=stable_seed(VERSION, namespace, identifier, "crt"),
        )
        output[identifier] = {
            "exact": exact,
            "crt": crt,
            "exact_diagnostics": exact_diag,
            "crt_diagnostics": crt_diag,
            "exact_eligible": exact_eligible(exact_diag),
            "crt_eligible": crt_eligible(crt_diag),
        }
    return output


def graph_conditional_metrics(
    graph,
    target: np.ndarray,
    nulls: dict,
) -> dict:
    exact = smoothness_against_permutations(graph, target, nulls["exact"])
    crt = smoothness_against_permutations(graph, target, nulls["crt"])
    return {
        "exact_eligible": bool(nulls["exact_eligible"]),
        "crt_eligible": bool(nulls["crt_eligible"]),
        "exact_effect": exact["effect"],
        "exact_z": exact["z"],
        "exact_p": exact["p_smoother"],
        "crt_effect": crt["effect"],
        "crt_z": crt["z"],
        "crt_p": crt["p_smoother"],
        "min_conditional_effect": (
            min(float(exact["effect"]), float(crt["effect"]))
            if nulls["exact_eligible"] and nulls["crt_eligible"]
            else float("nan")
        ),
    }


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(target)) != 2 or not np.isfinite(score).all():
        return float("nan")
    return float(roc_auc_score(target, score))


def _iu_utility(matrix: np.ndarray, graph, target: np.ndarray) -> dict:
    try:
        fits = laplacian_iu_path(matrix.T, (0.0, 0.1), graph=graph)
        iu = -(fits[0.0].w @ matrix.T)
        liu = -(fits[0.1].w @ matrix.T)
        iu_auc = _safe_auc(target, iu)
        liu_auc = _safe_auc(target, liu)
        return {
            "iu_auroc": iu_auc,
            "liu_auroc": liu_auc,
            "liu_delta_auroc": liu_auc - iu_auc,
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return {"iu_auroc": float("nan"), "liu_auroc": float("nan"), "liu_delta_auroc": float("nan")}


def _fixed_k_positive_count(
    samples: np.ndarray,
    target: np.ndarray,
    nulls: dict,
    *,
    tie_keys: np.ndarray,
) -> tuple[int, dict]:
    details = {}
    positive = 0
    for k in K_SENSITIVITY:
        graph = self_safe_knn_graph(samples, k=int(k), tie_keys=tie_keys)
        diagnostics = graph_conditional_metrics(graph, target, nulls)
        value = diagnostics["min_conditional_effect"]
        current_positive = bool(np.isfinite(value) and value > 0)
        positive += int(current_positive)
        details[str(k)] = {
            "positive": current_positive,
            "min_conditional_effect": _json_number(value),
            "healthy": graph_is_healthy(extended_graph_diagnostics(graph)),
        }
    return positive, details


def evaluate_outer_folds(
    cells: list[dict],
    names: tuple[str, ...],
    nulls: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    rows, weight_rows = [], []
    for fold_index, held_family in enumerate(FAMILY_NAMES, start=1):
        donors = [cell for cell in cells if cell["family"] != held_family]
        held = [cell for cell in cells if cell["family"] == held_family]
        mean_weights, seed_weights = fit_metric_ensemble(donors)
        candidates = candidate_definitions(mean_weights, names)
        print(f"outer {fold_index}/{len(FAMILY_NAMES)} hold={held_family}", flush=True)
        for seed, vector in seed_weights.items():
            for feature, weight in zip(names, vector):
                weight_rows.append({
                    "held_family": held_family,
                    "fit_seed": int(seed),
                    "feature": feature,
                    "weight": float(weight),
                    "kind": "seed",
                })
        for feature, weight in zip(names, mean_weights):
            weight_rows.append({
                "held_family": held_family,
                "fit_seed": "ensemble",
                "feature": feature,
                "weight": float(weight),
                "kind": "ensemble",
            })
        for candidate in candidates:
            model = fit_balanced_logistic(
                donors,
                weights=candidate["weights"],
                support=candidate["support"],
                seed=stable_seed(VERSION, "logistic", held_family, candidate["candidate"]),
            )
            selected_names = [names[index] for index in candidate["support"]]
            for cell in held:
                identifier = cell["cell"]
                target = np.asarray(cell["y"], dtype=int)
                samples = metric_matrix(cell["X"], candidate["weights"], candidate["support"])
                linear_score = model.decision_function(samples)
                logistic_auc = _safe_auc(target, linear_score)
                for tie_seed in TIE_SEEDS:
                    tie_keys = target_blind_tie_keys(
                        len(target), namespace=identifier, seed=int(tie_seed)
                    )
                    metric_graph, metric_health = select_label_free_graph(
                        samples, tie_keys=tie_keys
                    )
                    linear_graph, linear_health = select_label_free_graph(
                        linear_score[:, None], tie_keys=tie_keys
                    )
                    for graph_role, graph, health, graph_samples in (
                        ("metric_graph", metric_graph, metric_health, samples),
                        ("linear_score_graph", linear_graph, linear_health, linear_score[:, None]),
                    ):
                        base = {
                            "held_family": held_family,
                            "cell": identifier,
                            "candidate": candidate["candidate"],
                            "candidate_role": candidate["role"],
                            "support_size": int(candidate["support_size"]),
                            "selected_features": "|".join(selected_names),
                            "graph_role": graph_role,
                            "tie_seed": int(tie_seed),
                            "n": len(target),
                            "error_rate": float(np.mean(target)),
                            "graph_eligible": bool(health["eligible"]),
                            "selected_k": health.get("selected_k"),
                            "largest_component_fraction": health.get("largest_component_fraction"),
                            "isolated_fraction": health.get("isolated_fraction"),
                            "logistic_auroc": logistic_auc,
                        }
                        if graph is None:
                            rows.append({
                                **base,
                                "exact_eligible": bool(nulls[identifier]["exact_eligible"]),
                                "crt_eligible": bool(nulls[identifier]["crt_eligible"]),
                                "exact_effect": float("nan"),
                                "crt_effect": float("nan"),
                                "min_conditional_effect": float("nan"),
                                "fixed_k_positive_count": 0,
                                "iu_auroc": float("nan"),
                                "liu_auroc": float("nan"),
                                "liu_delta_auroc": float("nan"),
                            })
                            continue
                        conditional = graph_conditional_metrics(graph, target, nulls[identifier])
                        if graph_role == "metric_graph":
                            positive_count, sensitivity = _fixed_k_positive_count(
                                graph_samples,
                                target,
                                nulls[identifier],
                                tie_keys=tie_keys,
                            )
                            utility = _iu_utility(graph_samples, graph, target)
                        else:
                            positive_count, sensitivity = 0, {}
                            utility = {"iu_auroc": float("nan"), "liu_auroc": float("nan"), "liu_delta_auroc": float("nan")}
                        rows.append({
                            **base,
                            **conditional,
                            "fixed_k_positive_count": int(positive_count),
                            "fixed_k_sensitivity": json.dumps(sensitivity, sort_keys=True),
                            **utility,
                        })
    return rows, weight_rows


def summarize_outer_families(rows: list[dict]) -> list[dict]:
    """Aggregate held-cell results while keeping tie seeds non-independent."""

    output = []
    keys = sorted({
        (str(row["held_family"]), str(row["candidate"]), str(row["graph_role"]))
        for row in rows
    })
    for held_family, candidate, graph_role in keys:
        current = [
            row for row in rows
            if row["held_family"] == held_family
            and row["candidate"] == candidate
            and row["graph_role"] == graph_role
        ]
        cells = sorted({str(row["cell"]) for row in current})
        cell_summaries = []
        for identifier in cells:
            cell_rows = [row for row in current if row["cell"] == identifier]
            eligible = [
                row for row in cell_rows
                if row["graph_eligible"] and row["exact_eligible"] and row["crt_eligible"]
            ]
            utilities = [
                float(row["liu_delta_auroc"])
                for row in cell_rows
                if np.isfinite(row["liu_delta_auroc"])
            ]
            cell_summaries.append({
                "eligible": bool(len(eligible) == len(cell_rows)),
                "exact": float(np.mean([row["exact_effect"] for row in eligible])) if eligible else float("nan"),
                "crt": float(np.mean([row["crt_effect"] for row in eligible])) if eligible else float("nan"),
                "minimum": float(np.mean([row["min_conditional_effect"] for row in eligible])) if eligible else float("nan"),
                "utility": float(np.mean(utilities)) if utilities else float("nan"),
            })
        eligible_cells = [value for value in cell_summaries if value["eligible"]]

        def finite_mean(field: str):
            values = [value[field] for value in eligible_cells if np.isfinite(value[field])]
            return float(np.mean(values)) if values else None

        output.append({
            "held_family": held_family,
            "candidate": candidate,
            "candidate_role": current[0]["candidate_role"],
            "graph_role": graph_role,
            "cell_count": len(cells),
            "tie_seed_count": len({int(row["tie_seed"]) for row in current}),
            "eligible_cell_fraction": float(np.mean([
                value["eligible"] for value in cell_summaries
            ])),
            "mean_exact_effect": finite_mean("exact"),
            "mean_crt_effect": finite_mean("crt"),
            "mean_min_conditional_effect": finite_mean("minimum"),
            "mean_liu_delta_auroc": finite_mean("utility"),
        })
    return output


def _target_map(cells: list[dict], overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    return {
        cell["cell"]: np.asarray(
            overrides[cell["cell"]] if overrides is not None else cell["y"], dtype=int
        )
        for cell in cells
    }


def whole_search_statistics(
    cells: list[dict],
    names: tuple[str, ...],
    *,
    targets: dict[str, np.ndarray] | None = None,
) -> dict:
    """Rerun all donor fits/supports and return held-family maxT statistics."""

    target_map = _target_map(cells, targets)
    by_candidate_metric = {f"supervised_s{'all' if size is None else size}": {} for size in SUPPORT_SIZES}
    by_candidate_linear = {key: {} for key in by_candidate_metric}
    for held_family in FAMILY_NAMES:
        donors = [cell for cell in cells if cell["family"] != held_family]
        held = [cell for cell in cells if cell["family"] == held_family]
        mean_weights, _ = fit_metric_ensemble(donors, targets=target_map)
        candidates = [value for value in candidate_definitions(mean_weights, names) if value["role"] == "supervised_metric"]
        for candidate in candidates:
            model = fit_balanced_logistic(
                donors,
                weights=candidate["weights"],
                support=candidate["support"],
                targets=target_map,
                seed=stable_seed(VERSION, "maxT-logistic", held_family, candidate["candidate"]),
            )
            metric_values, linear_values = [], []
            for cell in held:
                identifier = cell["cell"]
                indexes = deterministic_subsample(
                    len(cell["y"]), namespace=identifier, max_rows=MAXT_MAX_ROWS
                )
                target = target_map[identifier][indexes]
                length = np.asarray(cell["length"])[indexes]
                if len(np.unique(target)) != 2:
                    metric_values.append(-1.0)
                    linear_values.append(-1.0)
                    continue
                samples = metric_matrix(
                    np.asarray(cell["X"])[indexes],
                    candidate["weights"],
                    candidate["support"],
                )
                score = model.decision_function(samples)
                tie_keys = target_blind_tie_keys(
                    len(indexes), namespace=f"maxT|{identifier}", seed=TIE_SEEDS[0]
                )
                metric_graph, _ = select_label_free_graph(samples, tie_keys=tie_keys)
                linear_graph, _ = select_label_free_graph(score[:, None], tie_keys=tie_keys)
                metric_values.append(
                    conditional_residual_smoothness(
                        metric_graph, target, length,
                        seed=stable_seed(VERSION, "maxT-residual", identifier),
                    ) if metric_graph is not None else -1.0
                )
                linear_values.append(
                    conditional_residual_smoothness(
                        linear_graph, target, length,
                        seed=stable_seed(VERSION, "maxT-linear-residual", identifier),
                    ) if linear_graph is not None else -1.0
                )
            by_candidate_metric[candidate["candidate"]][held_family] = float(np.mean(metric_values))
            by_candidate_linear[candidate["candidate"]][held_family] = float(np.mean(linear_values))
    metric = {
        candidate: float(np.mean([values[family_name] for family_name in FAMILY_NAMES]))
        for candidate, values in by_candidate_metric.items()
    }
    linear = {
        candidate: float(np.mean([values[family_name] for family_name in FAMILY_NAMES]))
        for candidate, values in by_candidate_linear.items()
    }
    advantage = {candidate: metric[candidate] - linear[candidate] for candidate in metric}
    return {
        "metric": metric,
        "linear": linear,
        "advantage": advantage,
        "per_family_metric": by_candidate_metric,
        "per_family_linear": by_candidate_linear,
    }


def _save_null_checkpoint(path: Path, values: dict[str, np.ndarray], reruns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, reruns=np.asarray([reruns]), **values)
    os.replace(temporary, path)


def whole_search_null(
    cells: list[dict],
    names: tuple[str, ...],
    observed: dict,
    null_worlds: dict[str, dict],
    *,
    reruns: int,
    checkpoint: Path,
) -> dict:
    keys = ("exact_metric", "exact_advantage", "crt_metric", "crt_advantage")
    arrays = {key: np.full(int(reruns), np.nan, dtype=float) for key in keys}
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if int(saved["reruns"][0]) == int(reruns):
                for key in keys:
                    arrays[key][:] = saved[key]
    for index in range(int(reruns)):
        if all(np.isfinite(arrays[key][index]) for key in keys):
            continue
        for null_name in ("exact", "crt"):
            targets = {
                cell["cell"]: null_worlds[cell["cell"]][null_name][:, index]
                for cell in cells
            }
            statistic = whole_search_statistics(cells, names, targets=targets)
            arrays[f"{null_name}_metric"][index] = max(statistic["metric"].values())
            arrays[f"{null_name}_advantage"][index] = max(statistic["advantage"].values())
        if (index + 1) % 5 == 0 or index + 1 == int(reruns):
            _save_null_checkpoint(checkpoint, arrays, int(reruns))
            print(f"whole-search null {index + 1}/{reruns}", flush=True)
    output = {
        "reruns": int(reruns),
        "maxT_subsample_max_rows": MAXT_MAX_ROWS,
        "observed": observed,
        "null_quantiles": {},
        "candidate_p_values": {},
    }
    for null_name in ("exact", "crt"):
        output["null_quantiles"][null_name] = {
            "metric_95": float(np.quantile(arrays[f"{null_name}_metric"], 0.95)),
            "advantage_95": float(np.quantile(arrays[f"{null_name}_advantage"], 0.95)),
        }
    for candidate in observed["metric"]:
        output["candidate_p_values"][candidate] = {}
        for null_name in ("exact", "crt"):
            metric_null = arrays[f"{null_name}_metric"]
            advantage_null = arrays[f"{null_name}_advantage"]
            output["candidate_p_values"][candidate][f"{null_name}_metric_p_maxT"] = float(
                (1 + np.sum(metric_null >= observed["metric"][candidate])) / (len(metric_null) + 1)
            )
            output["candidate_p_values"][candidate][f"{null_name}_advantage_p_maxT"] = float(
                (1 + np.sum(advantage_null >= observed["advantage"][candidate])) / (len(advantage_null) + 1)
            )
    return output


def _bootstrap(values: list[float], *, namespace: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed(VERSION, "bootstrap", namespace))
    indexes = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    draws = np.mean(array[indexes], axis=1)
    return float(np.mean(array)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_candidates(
    rows: list[dict],
    weight_rows: list[dict],
    names: tuple[str, ...],
    whole_null: dict,
    controls: dict,
) -> tuple[list[dict], dict]:
    candidates = sorted({row["candidate"] for row in rows})
    summaries = []
    weight_lookup = {}
    ensemble_rows = [row for row in weight_rows if row["kind"] == "ensemble"]
    for held_family in FAMILY_NAMES:
        vector = np.asarray([
            next(row["weight"] for row in ensemble_rows if row["held_family"] == held_family and row["feature"] == feature)
            for feature in names
        ], dtype=float)
        weight_lookup[held_family] = vector
    for candidate in candidates:
        metric = [row for row in rows if row["candidate"] == candidate and row["graph_role"] == "metric_graph"]
        linear = [row for row in rows if row["candidate"] == candidate and row["graph_role"] == "linear_score_graph"]
        candidate_role = metric[0]["candidate_role"]
        support_size = int(metric[0]["support_size"])
        fold_supports = [support_indices(weight_lookup[held], names, None if support_size == len(names) else support_size) for held in FAMILY_NAMES]
        cosine = median_pairwise_cosine(list(weight_lookup.values()))
        jaccard = median_pairwise_jaccard(fold_supports)
        frequency = np.zeros(len(names), dtype=float)
        for support in fold_supports:
            frequency[support] += 1
        frequency /= len(fold_supports)
        consensus_weights = np.mean(list(weight_lookup.values()), axis=0)
        consensus_support = support_indices(
            consensus_weights,
            names,
            None if support_size == len(names) else support_size,
        )
        central_frequency = float(np.min(frequency[consensus_support]))
        tie_results = []
        for tie_seed in TIE_SEEDS:
            current = [row for row in metric if row["tie_seed"] == tie_seed]
            current_linear = [row for row in linear if row["tie_seed"] == tie_seed]
            family_exact, family_crt, family_min, family_adv, family_utility = [], [], [], [], []
            eligible_families = 0
            for held_family in FAMILY_NAMES:
                family_rows = [row for row in current if row["held_family"] == held_family]
                linear_rows = [row for row in current_linear if row["held_family"] == held_family]
                eligible = [row for row in family_rows if row["graph_eligible"] and row["exact_eligible"] and row["crt_eligible"]]
                linear_eligible = [row for row in linear_rows if row["graph_eligible"] and row["exact_eligible"] and row["crt_eligible"]]
                family_eligible = bool(
                    family_rows
                    and len(eligible) / len(family_rows) >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
                )
                linear_family_eligible = bool(
                    linear_rows
                    and len(linear_eligible) / len(linear_rows) >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
                )
                if family_eligible:
                    eligible_families += 1
                    exact_value = float(np.mean([row["exact_effect"] for row in eligible]))
                    crt_value = float(np.mean([row["crt_effect"] for row in eligible]))
                    family_exact.append(exact_value)
                    family_crt.append(crt_value)
                    family_min.append(min(exact_value, crt_value))
                if family_eligible and linear_family_eligible:
                    linear_min = float(np.mean([row["min_conditional_effect"] for row in linear_eligible]))
                    family_adv.append(float(np.mean([row["min_conditional_effect"] for row in eligible])) - linear_min)
                utility_values = [row["liu_delta_auroc"] for row in family_rows if np.isfinite(row["liu_delta_auroc"])]
                if utility_values:
                    family_utility.append(float(np.mean(utility_values)))
            exact_mean, exact_low, exact_high = _bootstrap(family_exact, namespace=f"{candidate}|{tie_seed}|exact")
            crt_mean, crt_low, crt_high = _bootstrap(family_crt, namespace=f"{candidate}|{tie_seed}|crt")
            advantage_mean, advantage_low, advantage_high = _bootstrap(family_adv, namespace=f"{candidate}|{tie_seed}|adv")
            utility_mean, utility_low, utility_high = _bootstrap(family_utility, namespace=f"{candidate}|{tie_seed}|utility")
            positive_fraction = float(np.mean(np.asarray(family_min) > 0)) if family_min else 0.0
            health_fraction = float(np.mean([row["graph_eligible"] for row in current]))
            sensitivity_fraction = float(np.mean([row["fixed_k_positive_count"] >= 3 for row in current]))
            tie_pass = bool(
                eligible_families / len(FAMILY_NAMES) >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
                and positive_fraction >= POSITIVE_ENVIRONMENT_FRACTION_MIN
                and family_min and float(np.median(family_min)) >= MEDIAN_CONDITIONAL_EFFECT_MIN
                and exact_low > 0 and crt_low > 0
                and health_fraction >= HEALTHY_CELL_FRACTION_MIN
                and sensitivity_fraction >= HEALTHY_CELL_FRACTION_MIN
            )
            tie_results.append({
                "tie_seed": int(tie_seed),
                "eligible_environment_fraction": eligible_families / len(FAMILY_NAMES),
                "positive_environment_fraction": positive_fraction,
                "median_min_conditional_effect": float(np.median(family_min)) if family_min else None,
                "exact_mean": _json_number(exact_mean),
                "exact_ci_low": _json_number(exact_low),
                "exact_ci_high": _json_number(exact_high),
                "crt_mean": _json_number(crt_mean),
                "crt_ci_low": _json_number(crt_low),
                "crt_ci_high": _json_number(crt_high),
                "health_fraction": health_fraction,
                "fixed_k_sensitivity_fraction": sensitivity_fraction,
                "linear_advantage_mean": _json_number(advantage_mean),
                "linear_advantage_ci_low": _json_number(advantage_low),
                "linear_advantage_ci_high": _json_number(advantage_high),
                "liu_delta_mean": _json_number(utility_mean),
                "liu_delta_ci_low": _json_number(utility_low),
                "liu_delta_ci_high": _json_number(utility_high),
                "pass": tie_pass,
            })
        null_p = whole_null["candidate_p_values"].get(candidate, {})
        maxT_pass = bool(
            candidate_role != "supervised_metric"
            or (
                null_p.get("exact_metric_p_maxT", 1.0) <= 0.05
                and null_p.get("crt_metric_p_maxT", 1.0) <= 0.05
            )
        )
        stability_pass = bool(
            cosine >= WEIGHT_COSINE_MIN
            and jaccard >= SUPPORT_JACCARD_MIN
            and central_frequency >= FEATURE_SELECTION_FREQUENCY_MIN
        )
        geometry_pass = bool(
            candidate_role == "supervised_metric"
            and all(result["pass"] for result in tie_results)
            and maxT_pass
            and stability_pass
            and controls["all_controls_pass"]
        )
        distinct_pass = bool(
            geometry_pass
            and all(
                result["linear_advantage_mean"] is not None
                and result["linear_advantage_mean"] >= DISTINCT_ADVANTAGE_MIN
                and result["linear_advantage_ci_low"] is not None
                and result["linear_advantage_ci_low"] > 0
                for result in tie_results
            )
            and null_p.get("exact_advantage_p_maxT", 1.0) <= 0.05
            and null_p.get("crt_advantage_p_maxT", 1.0) <= 0.05
        )
        utility_pass = bool(all(
            result["liu_delta_mean"] is not None
            and result["liu_delta_mean"] >= UTILITY_DELTA_MIN
            and result["liu_delta_ci_low"] is not None
            and result["liu_delta_ci_low"] > 0
            for result in tie_results
        ))
        summaries.append({
            "candidate": candidate,
            "candidate_role": candidate_role,
            "support_size": support_size,
            "weight_cosine": cosine,
            "support_jaccard": jaccard,
            "minimum_selected_feature_frequency": central_frequency,
            "stability_pass": stability_pass,
            "maxT_pass": maxT_pass,
            "geometry_pass": geometry_pass,
            "distinct_manifold_pass": distinct_pass,
            "utility_pass": utility_pass,
            "tie_results": tie_results,
            **null_p,
        })
    passing_geometry = [row for row in summaries if row["geometry_pass"]]
    passing_distinct = [row for row in summaries if row["distinct_manifold_pass"]]
    if not controls["all_controls_pass"]:
        decision = "CONTROL_FAILURE_INVALIDATES_SUPERVISED_DISCOVERY"
    elif passing_distinct:
        decision = "INTERNAL_NONLINEAR_GEOMETRY_CANDIDATE_AWAITING_EXTERNAL_VALIDATION"
    elif passing_geometry:
        decision = "TRANSFERABLE_SUPERVISED_DIRECTION_ONLY"
    else:
        decision = "NO_STABLE_SUPERVISED_CONDITIONAL_GEOMETRY"
    return summaries, {
        "decision": decision,
        "geometry_candidates": [row["candidate"] for row in passing_geometry],
        "distinct_manifold_candidates": [row["candidate"] for row in passing_distinct],
        "utility_candidates": [row["candidate"] for row in summaries if row["utility_pass"]],
    }


def build_frozen_candidate(
    cells: list[dict],
    names: tuple[str, ...],
    summaries: list[dict],
    decision: dict,
    definition: dict,
) -> dict | None:
    eligible = (
        decision["distinct_manifold_candidates"]
        if decision["distinct_manifold_candidates"]
        else decision["geometry_candidates"]
    )
    if not eligible:
        return None
    lookup = {row["candidate"]: row for row in summaries}
    selected = sorted(
        eligible,
        key=lambda name: (
            -min(result["median_min_conditional_effect"] for result in lookup[name]["tie_results"]),
            lookup[name]["support_size"],
            name,
        ),
    )[0]
    mean_weights, seed_weights = fit_metric_ensemble(cells)
    size = lookup[selected]["support_size"]
    support = support_indices(mean_weights, names, None if size == len(names) else size)
    logistic = fit_balanced_logistic(
        cells,
        weights=mean_weights,
        support=support,
        seed=stable_seed(VERSION, "frozen-logistic"),
    )
    return {
        "version": VERSION,
        "status": "internal_discovery_candidate_awaiting_external_validation",
        "decision": decision["decision"],
        "candidate": selected,
        "feature_names": list(names),
        "weights": [float(value) for value in mean_weights],
        "support_indices": [int(value) for value in support],
        "support_features": [names[index] for index in support],
        "linear_comparator": {
            "type": "balanced_l2_logistic",
            "coefficient": [float(value) for value in logistic.coef_[0]],
            "intercept": float(logistic.intercept_[0]),
            "input_order": [names[index] for index in support],
        },
        "fit_seed_weights": {
            str(seed): [float(value) for value in vector]
            for seed, vector in seed_weights.items()
        },
        "graph_rule": {
            "type": "self_safe_local_scale_union_knn",
            "k_grid": [3, 5, 7, 10, 15, 25],
            "largest_component_min": 0.90,
            "isolated_fraction_max": 0.05,
            "tie_seeds": list(TIE_SEEDS),
        },
        "source_run_fingerprint": definition["run_fingerprint"],
        "claim_boundary": "supervised internal discovery; external validation unopened",
    }


def _control_eligibility(
    cells: list[dict],
    nulls: dict[str, dict],
) -> dict:
    by_family = []
    for held_family in FAMILY_NAMES:
        members = [cell for cell in cells if cell["family"] == held_family]
        exact_fraction = float(np.mean([
            nulls[cell["cell"]]["exact_eligible"] for cell in members
        ]))
        crt_fraction = float(np.mean([
            nulls[cell["cell"]]["crt_eligible"] for cell in members
        ]))
        by_family.append({
            "family": held_family,
            "exact_fraction": exact_fraction,
            "crt_fraction": crt_fraction,
            "pass": bool(
                exact_fraction >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
                and crt_fraction >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
            ),
        })
    return {
        "per_family": by_family,
        "passing_family_fraction": float(np.mean([row["pass"] for row in by_family])),
        "pass": bool(
            np.mean([row["pass"] for row in by_family])
            >= ELIGIBLE_ENVIRONMENT_FRACTION_MIN
        ),
    }


def control_checks(
    cells: list[dict],
    names: tuple[str, ...],
    whole_null: dict,
    *,
    permutations: int,
) -> dict:
    length_targets = {}
    permuted_cells = []
    for cell in cells:
        length = np.asarray(cell["length"], dtype=float)
        threshold = float(np.median(length))
        target = (length > threshold).astype(int)
        if len(np.unique(target)) < 2:
            order = np.argsort(length, kind="mergesort")
            target = np.zeros(len(length), dtype=int)
            target[order[len(order) // 2 :]] = 1
        length_targets[cell["cell"]] = target
        permutation = np.random.default_rng(
            stable_seed(VERSION, "row-permuted-control", cell["cell"])
        ).permutation(len(cell["y"]))
        permuted_cells.append({**cell, "X": np.asarray(cell["X"])[permutation]})
    length_stats = whole_search_statistics(cells, names, targets=length_targets)
    permuted_stats = whole_search_statistics(permuted_cells, names)
    length_nulls = prepare_conditional_nulls(
        cells,
        permutations=int(permutations),
        targets=length_targets,
        namespace="length-control",
    )
    observed_nulls = prepare_conditional_nulls(
        cells,
        permutations=int(permutations),
        namespace="permuted-representation-control",
    )
    length_eligibility = _control_eligibility(cells, length_nulls)
    permuted_eligibility = _control_eligibility(cells, observed_nulls)
    thresholds = whole_null["null_quantiles"]
    def promoted(statistics: dict, eligibility: dict) -> bool:
        metric = max(statistics["metric"].values())
        return bool(
            eligibility["pass"]
            and
            metric >= thresholds["exact"]["metric_95"]
            and metric >= thresholds["crt"]["metric_95"]
        )
    length_promoted = promoted(length_stats, length_eligibility)
    permuted_promoted = promoted(permuted_stats, permuted_eligibility)
    return {
        "length_only_world": {
            "max_metric_statistic": max(length_stats["metric"].values()),
            "max_advantage_statistic": max(length_stats["advantage"].values()),
            "conditional_eligibility": length_eligibility,
            "false_promotion": length_promoted,
            "pass": not length_promoted,
            "reason": "planted nuisance world must not beat both whole-search null thresholds",
        },
        "target_independent_row_permutation": {
            "max_metric_statistic": max(permuted_stats["metric"].values()),
            "max_advantage_statistic": max(permuted_stats["advantage"].values()),
            "conditional_eligibility": permuted_eligibility,
            "false_promotion": permuted_promoted,
            "pass": not permuted_promoted,
        },
        "all_controls_pass": bool(not length_promoted and not permuted_promoted),
    }


def make_figures(out: Path, summaries: list[dict], weight_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    supervised = [row for row in summaries if row["candidate_role"] == "supervised_metric"]
    labels = [row["candidate"] for row in supervised]
    metric = [min(result["median_min_conditional_effect"] for result in row["tie_results"]) for row in supervised]
    advantage = [min(result["linear_advantage_mean"] for result in row["tie_results"]) for row in supervised]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, np.asarray(metric) * 100)
    axes[0].axhline(MEDIAN_CONDITIONAL_EFFECT_MIN * 100, color="black", linewidth=1)
    axes[0].set_ylabel("Worst-tie conditional effect (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, np.asarray(advantage) * 100)
    axes[1].axhline(DISTINCT_ADVANTAGE_MIN * 100, color="black", linewidth=1)
    axes[1].set_ylabel("Advantage over linear-score graph (%)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "01_conditional_geometry_and_linear_advantage.png", dpi=180)
    plt.close(fig)

    ensemble = [row for row in weight_rows if row["kind"] == "ensemble"]
    features = sorted({row["feature"] for row in ensemble})
    means = [np.mean([row["weight"] for row in ensemble if row["feature"] == feature]) for feature in features]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(features)), np.asarray(means)[order])
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(np.asarray(features)[order])
    ax.set_xlabel("Mean donor-only diagonal metric weight")
    fig.tight_layout()
    fig.savefig(out / "02_discovered_feature_weights.png", dpi=180)
    plt.close(fig)


def build_report(out: Path, summaries: list[dict], decision: dict, frozen: dict | None) -> None:
    lines = [
        "# Supervised conditional manifold discovery v1",
        "",
        f"**Decision: `{decision['decision']}`**",
        "",
        "This is supervised internal discovery on outcome-opened Global cells. It is not external validation and does not change the prior DUFS audit decision.",
        "",
        "## Candidate gates",
        "",
        "| candidate | support | geometry | distinct vs linear | utility | exact/CRT maxT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["candidate_role"] != "supervised_metric":
            continue
        p_text = f"{row.get('exact_metric_p_maxT', float('nan')):.3f}/{row.get('crt_metric_p_maxT', float('nan')):.3f}"
        lines.append(
            f"| `{row['candidate']}` | {row['support_size']} | {row['geometry_pass']} | "
            f"{row['distinct_manifold_pass']} | {row['utility_pass']} | {p_text} |"
        )
    lines += ["", "## Interpretation", ""]
    if decision["decision"] == "INTERNAL_NONLINEAR_GEOMETRY_CANDIDATE_AWAITING_EXTERNAL_VALIDATION":
        lines.append("At least one supervised diagonal metric passes the internal conditional-geometry, stability, health, whole-search-null, and search-matched linear-comparator gates. It is only a candidate until the frozen representation is evaluated on new dataset/model cells.")
    elif decision["decision"] == "TRANSFERABLE_SUPERVISED_DIRECTION_ONLY":
        lines.append("A stable supervised representation transfers internally, but it does not establish local geometry beyond the search-matched linear direction.")
    else:
        lines.append("No supervised diagonal metric passes the frozen internal geometry gates. Increasing model capacity is not justified by this v1 search.")
    if frozen is not None:
        lines += ["", f"Frozen candidate: `{frozen['candidate']}`. External validation remains unopened."]
    lines += [
        "", "## Figures", "",
        "![Conditional geometry and linear advantage](01_conditional_geometry_and_linear_advantage.png)", "",
        "![Discovered feature weights](02_discovered_feature_weights.png)", "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells, names = load_discovery_cells(args.bundle)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/cross_dataset_hallucination_manifold_v1.py",
        ROOT / "scripts/inscope_cells.py",
        ROOT / "scripts/verify_supervised_conditional_manifold_discovery_v1.py",
        ROOT / "spectral_utils/supervised_manifold_discovery.py",
        ROOT / "spectral_utils/graph_topology.py",
        ROOT / "spectral_utils/laplacian_upcr.py",
        ROOT / "spectral_utils/specrage_views.py",
        ROOT / "spectral_utils/upcr.py",
        ROOT / "scripts/validate_supervised_conditional_manifold_v1.py",
        ROOT / "scripts/test_supervised_conditional_manifold_discovery_v1.py",
        ROOT / "configs/supervised_conditional_manifold_validation.example.json",
        PROTOCOL,
    ]
    definition = {
        "version": VERSION,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "bundle": str(args.bundle.relative_to(ROOT)),
        "bundle_sha256": _sha256(args.bundle),
        "cells": [cell["cell"] for cell in cells],
        "families": list(FAMILY_NAMES),
        "features": list(names),
        "support_sizes": [5, 10, 15, "all"],
        "fit_seeds": list(FIT_SEEDS),
        "tie_seeds": list(TIE_SEEDS),
        "conditional_permutations": int(args.permutations),
        "whole_search_null_reruns": int(args.null_reruns),
        "whole_search_subsample_max_rows": MAXT_MAX_ROWS,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "claim_boundary": "supervised internal discovery; external validation unopened",
    }
    definition["run_fingerprint"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(args.out_dir / "RUN_DEFINITION.json", definition)

    print("prepare observed conditional nulls", flush=True)
    observed_nulls = prepare_conditional_nulls(cells, permutations=args.permutations)
    rows, weight_rows = evaluate_outer_folds(cells, names, observed_nulls)
    _write_csv(args.out_dir / "OUTER_CELL_METRICS.csv", rows)
    _write_csv(args.out_dir / "OUTER_FAMILY_SUMMARY.csv", summarize_outer_families(rows))
    _write_csv(args.out_dir / "WEIGHT_STABILITY.csv", weight_rows)

    print("prepare whole-search conditional worlds", flush=True)
    null_worlds = prepare_conditional_nulls(
        cells, permutations=args.null_reruns, namespace="whole-search"
    )
    observed_statistic = whole_search_statistics(cells, names)
    whole_null = whole_search_null(
        cells,
        names,
        observed_statistic,
        null_worlds,
        reruns=args.null_reruns,
        checkpoint=args.out_dir / "checkpoints/whole_search_null.npz",
    )
    _write_json(args.out_dir / "WHOLE_SEARCH_NULL.json", whole_null)

    controls = control_checks(
        cells, names, whole_null, permutations=args.permutations
    )
    _write_json(args.out_dir / "CONTROLS.json", controls)
    summaries, decision = summarize_candidates(
        rows, weight_rows, names, whole_null, controls
    )
    flat_summaries = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if key != "tie_results"}
        flat["tie_results"] = json.dumps(row["tie_results"], sort_keys=True)
        flat_summaries.append(flat)
    _write_csv(args.out_dir / "CANDIDATE_SUMMARY.csv", flat_summaries)
    _write_json(args.out_dir / "DECISION.json", decision)
    frozen = build_frozen_candidate(cells, names, summaries, decision, definition)
    if frozen is not None:
        _write_json(args.out_dir / "FROZEN_CANDIDATE.json", frozen)
    make_figures(args.out_dir, summaries, weight_rows)
    build_report(args.out_dir, summaries, decision, frozen)
    print(f"decision={decision['decision']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--null-reruns", type=int, default=DEFAULT_NULL_RERUNS)
    args = parser.parse_args()
    if args.permutations < 19 or args.null_reruns < 19:
        raise ValueError("development runs require at least 19 conditional draws")
    run(args)


if __name__ == "__main__":
    main()
