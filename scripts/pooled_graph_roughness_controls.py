#!/usr/bin/env python3
"""Mechanism controls for frozen Pooled Graph-Roughness Direction V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    DUFS_EPOCHS,
    DUFS_SEEDS,
    family as dataset_family,
    load_contract,
    validate_bundle_without_labels,
)
from scripts.pooled_graph_roughness_fit import (  # noqa: E402
    DEFAULT_OUT as DEVELOPMENT_OUT,
    ELIGIBLE_CELLS,
    K,
    LAMBDAS,
    TRUST_FACTORS,
    VERSION as DEVELOPMENT_VERSION,
    candidate_key,
    candidates,
    canonical_hash,
    sha256_file,
    write_json,
)
from scripts.pooled_graph_roughness_report import (  # noqa: E402
    BOOTSTRAPS,
    BOOTSTRAP_SEED,
    choose_one_se,
    load_after_freeze,
    nested_rows,
    verify_fit,
)
from spectral_utils.family_residual_graph import (  # noqa: E402
    build_family_graphs,
    fit_family_residual_state,
    graphs_from_coordinates,
)
from spectral_utils.laplacian_upcr import dufs_soft_gates  # noqa: E402
from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    GraphRoughnessMoment,
    align_family_matrix,
    apply_pooled_roughness,
    fit_pooled_roughness_calibration,
    graph_roughness_moment,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


CONTROL_VERSION = "pooled-graph-roughness-controls-v1-2026-08-23"
DEFAULT_OUT = DEVELOPMENT_OUT / "controls"
CAPACITY_METHODS = (
    "dufs_graph",
    "contribution_graph",
    "equal_cell_pooling",
    "cross_only",
    "family_axis_permuted",
    "node_permuted_00",
)
N_NODE_PERMUTATIONS = 20


def stable_rng(*parts):
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def read_selection(development: Path):
    path = development / "FROZEN_SELECTION.json"
    selection = json.loads(path.read_text())
    payload = dict(selection)
    recorded = payload.pop("selection_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("development selection is not self-consistent")
    result_path = development / "RESULT.json"
    result = json.loads(result_path.read_text())
    if result["final_selection"]["selection_hash"] != recorded:
        raise RuntimeError("development result/selection mismatch")
    return selection, result, path, result_path


def prepare(args):
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    complete = verify_fit(args.development, args.bundle)
    selection, result, selection_path, result_path = read_selection(
        args.development
    )
    config = {
        "version": CONTROL_VERSION,
        "phase": "label_derived_hyperparameter_projection",
        "development_fit_manifest_hash": complete["manifest_hash"],
        "development_selection_hash": selection["selection_hash"],
        "outer_selected_configs": [{
            "held_group": row["held_group"],
            "lambda": row["lambda"],
            "trust_factor": row["trust_factor"],
            "candidate_key": row["candidate_key"],
        } for row in result["primary"]["outer_rows"]],
        "final_selected_config": selection["selected_config"],
        "receives_label_derived_hyperparameters": True,
        "contains_row_labels_or_scores": False,
        "source_hashes": {
            "bundle": sha256_file(args.bundle),
            "development_fit": sha256_file(
                args.development / "FIT_COMPLETE.json"
            ),
            "development_selection": sha256_file(selection_path),
            "development_result": sha256_file(result_path),
            "control_script": sha256_file(Path(__file__)),
        },
    }
    config["config_hash"] = canonical_hash(config)
    write_json(args.out / "FROZEN_CONTROL_CONFIG.json", config)
    print(json.dumps({
        "status": config["phase"],
        "outer_configs": len(config["outer_selected_configs"]),
        "config_hash": config["config_hash"],
    }, indent=2))


def read_control_config(args):
    path = args.out / "FROZEN_CONTROL_CONFIG.json"
    config = json.loads(path.read_text())
    payload = dict(config)
    recorded = payload.pop("config_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("control config is not self-consistent")
    complete = verify_fit(args.development, args.bundle)
    if config["development_fit_manifest_hash"] != complete["manifest_hash"]:
        raise RuntimeError("control config/development fit mismatch")
    current = {
        "bundle": sha256_file(args.bundle),
        "development_fit": sha256_file(
            args.development / "FIT_COMPLETE.json"
        ),
        "development_selection": sha256_file(
            args.development / "FROZEN_SELECTION.json"
        ),
        "development_result": sha256_file(
            args.development / "RESULT.json"
        ),
        "control_script": sha256_file(Path(__file__)),
    }
    if config["source_hashes"] != current:
        raise RuntimeError("control config source/input hash changed")
    if config.get("contains_row_labels_or_scores") is not False:
        raise RuntimeError("control config contains forbidden target material")
    return config, path


def transform_moment(moment, permutation):
    permutation = np.asarray(permutation, dtype=int)
    return GraphRoughnessMoment(
        A=moment.A[np.ix_(permutation, permutation)],
        c=moment.c[permutation],
        presence=moment.presence[permutation],
        families=moment.families,
        diagnostics={
            **moment.diagnostics,
            "family_axis_permutation": permutation.tolist(),
        },
    )


def load_control_cells(bundle: Path, development: Path):
    output = []
    with np.load(bundle, allow_pickle=True) as data:
        validate_bundle_without_labels(data)
        for index, cell in enumerate(ELIGIBLE_CELLS, start=1):
            print(f"[{index}/{len(ELIGIBLE_CELLS)}] controls {cell}", flush=True)
            F, names = load_contract(data, cell, "mixed_v2")
            state = fit_family_residual_state(F, names)
            families = tuple(state.contribution_space.families)
            with np.load(development / "states" / f"{cell}.npz") as stored:
                if tuple(stored["families"].astype(str)) != families:
                    raise RuntimeError(f"family registry drift: {cell}")
                if not np.allclose(stored["baseline"], state.baseline, atol=1e-12):
                    raise RuntimeError(f"baseline drift: {cell}")
                if not np.allclose(stored["residuals"], state.residuals, atol=1e-12):
                    raise RuntimeError(f"residual drift: {cell}")
                real_moment = GraphRoughnessMoment(
                    A=np.asarray(stored["moment_A"], dtype=float),
                    c=np.asarray(stored["moment_c"], dtype=float),
                    presence=np.asarray(stored["moment_presence"], dtype=bool),
                    families=tuple(VIEW_ORDER),
                )
            real_graph = graphs_from_coordinates(
                state.residuals, (K,), topology="union",
                tie_keys=np.arange(F.shape[1], dtype=float),
            )[K]
            contribution_graph = graphs_from_coordinates(
                state.standardized_contributions, (K,), topology="union",
                tie_keys=np.arange(F.shape[1], dtype=float),
            )[K]
            contribution_moment = graph_roughness_moment(
                state.baseline, state.residuals, families, contribution_graph
            )
            gates, _ = dufs_soft_gates(
                F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            dufs_graph = build_family_graphs(
                F, gates, state, eta=0.0, beta=0.5, ks=(K,),
                family_mode="residual", topology="union", scale_seed=1729,
            )[K].graph
            dufs_moment = graph_roughness_moment(
                state.baseline, state.residuals, families, dufs_graph
            )
            family_permutation = stable_rng(
                CONTROL_VERSION, "family-axis", cell
            ).permutation(len(real_moment.families))
            aligned_residuals, aligned_presence = align_family_matrix(
                state.residuals, families, global_families=VIEW_ORDER
            )
            permuted_presence = aligned_presence[family_permutation]
            permuted_residuals = aligned_residuals[:, family_permutation][
                :, permuted_presence
            ]
            permuted_families = tuple(
                np.asarray(VIEW_ORDER)[permuted_presence].tolist()
            )
            node_moments = []
            for replicate in range(N_NODE_PERMUTATIONS):
                permutation = stable_rng(
                    CONTROL_VERSION, "node", replicate, cell
                ).permutation(F.shape[1])
                permuted_graph = real_graph[permutation][:, permutation]
                node_moments.append(graph_roughness_moment(
                    state.baseline, state.residuals, families, permuted_graph
                ))
            output.append({
                "cell": cell,
                "group": dataset_family(cell),
                "baseline": np.asarray(state.baseline, dtype=float),
                "residuals": np.asarray(state.residuals, dtype=float),
                "families": families,
                "family_axis_scoring": {
                    "residuals": permuted_residuals,
                    "families": permuted_families,
                },
                "moments": {
                    "real": real_moment,
                    "dufs_graph": dufs_moment,
                    "contribution_graph": contribution_moment,
                    "family_axis_permuted": transform_moment(
                        real_moment, family_permutation
                    ),
                    **{
                        f"node_permuted_{replicate:02d}": moment
                        for replicate, moment in enumerate(node_moments)
                    },
                },
            })
    return output


def calibration(cells, method, excluded, lambda_):
    source = [cell for cell in cells if cell["group"] not in excluded]
    base_method = method
    pooling = "equal_group"
    cross_only = False
    if method == "equal_cell_pooling":
        base_method = "real"
        pooling = "equal_cell"
    elif method == "cross_only":
        base_method = "real"
        cross_only = True
    return fit_pooled_roughness_calibration(
        [cell["moments"][base_method] for cell in source],
        [cell["group"] for cell in source],
        lambda_, pooling=pooling, cross_only=cross_only,
    )


def cache_for_method(cells, method):
    groups = tuple(sorted({cell["group"] for cell in cells}))
    exclusions = {()}
    exclusions.update((group,) for group in groups)
    exclusions.update(
        tuple(sorted((left, right)))
        for index, left in enumerate(groups)
        for right in groups[index + 1:]
    )
    return {
        (excluded, lambda_): calibration(cells, method, excluded, lambda_)
        for excluded in sorted(exclusions, key=lambda value: (len(value), value))
        for lambda_ in LAMBDAS
    }


def score_bank(cells, method, out: Path):
    cache = cache_for_method(cells, method)
    groups = tuple(sorted({cell["group"] for cell in cells}))
    score_dir = out / "scores" / method
    score_dir.mkdir(parents=True)
    hashes = {}
    for cell in cells:
        target_group = cell["group"]
        if method == "family_axis_permuted":
            score_residuals = cell["family_axis_scoring"]["residuals"]
            score_families = cell["family_axis_scoring"]["families"]
        else:
            score_residuals = cell["residuals"]
            score_families = cell["families"]
        values = {"iu": cell["baseline"]}
        for lambda_, trust in candidates():
            key = candidate_key(lambda_, trust)
            for prefix, excluded in (
                ("full", ()), ("outer", (target_group,))
            ):
                values[f"{prefix}__{key}"] = apply_pooled_roughness(
                    cell["baseline"], score_residuals, score_families,
                    cache[(excluded, lambda_)], trust,
                ).score
            for outer_group in groups:
                if outer_group == target_group:
                    continue
                excluded = tuple(sorted((outer_group, target_group)))
                values[f"inner={outer_group}__{key}"] = apply_pooled_roughness(
                    cell["baseline"], score_residuals, score_families,
                    cache[(excluded, lambda_)], trust,
                ).score
        path = score_dir / f"{cell['cell']}.npz"
        np.savez_compressed(path, **values)
        hashes[cell["cell"]] = sha256_file(path)
    calibrations = {
        f"exclude={'+'.join(excluded) if excluded else 'none'}__lambda={lambda_:g}": {
            "direction": value.direction.tolist(),
            "A": value.A.tolist(),
            "c": value.c.tolist(),
            "diagnostics": value.diagnostics,
        }
        for (excluded, lambda_), value in cache.items()
    }
    calibration_path = out / "calibrations" / f"{method}.json"
    write_json(calibration_path, calibrations)
    return hashes, sha256_file(calibration_path)


def matched_node_permutations(cells, control_config, out: Path):
    primary_by_group = {
        row["held_group"]: row
        for row in control_config["outer_selected_configs"]
    }
    final = control_config["final_selected_config"]
    values_by_cell = {cell["cell"]: {} for cell in cells}
    calibration_payload = {}
    for replicate in range(N_NODE_PERMUTATIONS):
        method = f"node_permuted_{replicate:02d}"
        for held, row in primary_by_group.items():
            fitted = calibration(cells, method, (held,), float(row["lambda"]))
            calibration_payload[f"{method}__exclude={held}"] = {
                "direction": fitted.direction.tolist(),
                "lambda": row["lambda"], "trust_factor": row["trust_factor"],
            }
            for cell in cells:
                if cell["group"] != held:
                    continue
                values_by_cell[cell["cell"]][f"outer__r{replicate:02d}"] = (
                    apply_pooled_roughness(
                        cell["baseline"], cell["residuals"], cell["families"],
                        fitted, float(row["trust_factor"]),
                    ).score
                )
        full = calibration(cells, method, (), float(final["lambda"]))
        calibration_payload[f"{method}__exclude=none"] = {
            "direction": full.direction.tolist(),
            "lambda": final["lambda"], "trust_factor": final["trust_factor"],
        }
    directory = out / "scores" / "matched_node_permutations"
    directory.mkdir(parents=True)
    hashes = {}
    for cell in cells:
        path = directory / f"{cell['cell']}.npz"
        np.savez_compressed(path, iu=cell["baseline"], **values_by_cell[cell["cell"]])
        hashes[cell["cell"]] = sha256_file(path)
    calibration_path = out / "calibrations" / "matched_node_permutations.json"
    write_json(calibration_path, calibration_payload)
    return hashes, sha256_file(calibration_path)


def fit(args):
    if not args.out.exists():
        raise FileNotFoundError("run prepare before fit")
    if (args.out / "FIT_MANIFEST.json").exists():
        raise FileExistsError("control fit already exists")
    extra = set(args.out.iterdir()) - {args.out / "FROZEN_CONTROL_CONFIG.json"}
    if extra:
        raise RuntimeError(f"unexpected pre-fit control artifacts: {sorted(extra)}")
    (args.out / "scores").mkdir()
    (args.out / "calibrations").mkdir()
    development_complete = verify_fit(args.development, args.bundle)
    control_config, control_config_path = read_control_config(args)
    cells = load_control_cells(args.bundle, args.development)
    score_hashes, calibration_hashes = {}, {}
    for method in CAPACITY_METHODS:
        print(f"capacity bank: {method}", flush=True)
        score_hashes[method], calibration_hashes[method] = score_bank(
            cells, method, args.out
        )
    matched_hashes, matched_calibration_hash = matched_node_permutations(
        cells, control_config, args.out
    )
    manifest = {
        "version": CONTROL_VERSION,
        "phase": "post_selection_target_free_control_scores_frozen",
        "development_version": DEVELOPMENT_VERSION,
        "development_fit_manifest_hash": development_complete["manifest_hash"],
        "development_selection_hash": control_config[
            "development_selection_hash"
        ],
        "control_config_hash": control_config["config_hash"],
        "capacity_methods": list(CAPACITY_METHODS),
        "n_node_permutations": N_NODE_PERMUTATIONS,
        "raw_row_labels_accessed_by_fit": False,
        "receives_label_derived_hyperparameters": True,
        "target_fields_received_by_fit": [
            "development_selected_hyperparameters",
            "development_outer_selected_hyperparameters",
        ],
        "hashes": {
            "bundle": sha256_file(args.bundle),
            "development_fit": sha256_file(args.development / "FIT_COMPLETE.json"),
            "control_config": sha256_file(control_config_path),
            "control_script": sha256_file(Path(__file__)),
            "core_module": sha256_file(
                REPO / "spectral_utils" / "pooled_graph_roughness.py"
            ),
            "family_graph_module": sha256_file(
                REPO / "spectral_utils" / "family_residual_graph.py"
            ),
            "graph_topology_module": sha256_file(
                REPO / "spectral_utils" / "graph_topology.py"
            ),
            "laplacian_module": sha256_file(
                REPO / "spectral_utils" / "laplacian_upcr.py"
            ),
            "contribution_module": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
            "upcr_module": sha256_file(
                REPO / "spectral_utils" / "upcr.py"
            ),
            "family_registry_module": sha256_file(
                REPO / "spectral_utils" / "specrage_views.py"
            ),
            "contract_loader_script": sha256_file(
                REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py"
            ),
            "dufs_trainer_module": sha256_file(
                REPO / "spectral_utils" / "selectors" / "a2_groupfs.py"
            ),
            "score_files": score_hashes,
            "calibrations": calibration_hashes,
            "matched_node_score_files": matched_hashes,
            "matched_node_calibrations": matched_calibration_hash,
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(args.out / "FIT_MANIFEST.json", manifest)
    print(json.dumps({
        "status": manifest["phase"],
        "methods": len(CAPACITY_METHODS),
        "node_permutations": N_NODE_PERMUTATIONS,
        "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


def verify_controls(args):
    manifest = json.loads((args.out / "FIT_MANIFEST.json").read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("control manifest is not self-consistent")
    if manifest.get("version") != CONTROL_VERSION:
        raise RuntimeError("control manifest version changed")
    if manifest.get("raw_row_labels_accessed_by_fit") is not False:
        raise RuntimeError("control fit accessed raw row labels")
    if manifest.get("receives_label_derived_hyperparameters") is not True:
        raise RuntimeError("control fit provenance omits meta-selection")
    if manifest.get("target_fields_received_by_fit") != [
        "development_selected_hyperparameters",
        "development_outer_selected_hyperparameters",
    ]:
        raise RuntimeError("control fit target-field disclosure changed")
    complete = verify_fit(args.development, args.bundle)
    control_config, control_config_path = read_control_config(args)
    if manifest["development_fit_manifest_hash"] != complete["manifest_hash"]:
        raise RuntimeError("control/development fit mismatch")
    if manifest["development_selection_hash"] != control_config[
        "development_selection_hash"
    ]:
        raise RuntimeError("control/development selection mismatch")
    if manifest["control_config_hash"] != control_config["config_hash"]:
        raise RuntimeError("control manifest/config mismatch")
    hashes = manifest["hashes"]
    fixed = {
        "bundle": sha256_file(args.bundle),
        "development_fit": sha256_file(args.development / "FIT_COMPLETE.json"),
        "control_config": sha256_file(control_config_path),
        "control_script": sha256_file(Path(__file__)),
        "core_module": sha256_file(REPO / "spectral_utils" / "pooled_graph_roughness.py"),
        "family_graph_module": sha256_file(REPO / "spectral_utils" / "family_residual_graph.py"),
        "graph_topology_module": sha256_file(REPO / "spectral_utils" / "graph_topology.py"),
        "laplacian_module": sha256_file(REPO / "spectral_utils" / "laplacian_upcr.py"),
        "contribution_module": sha256_file(REPO / "spectral_utils" / "contribution_subspace.py"),
        "upcr_module": sha256_file(REPO / "spectral_utils" / "upcr.py"),
        "family_registry_module": sha256_file(REPO / "spectral_utils" / "specrage_views.py"),
        "contract_loader_script": sha256_file(REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py"),
        "dufs_trainer_module": sha256_file(REPO / "spectral_utils" / "selectors" / "a2_groupfs.py"),
    }
    for key, value in fixed.items():
        if hashes[key] != value:
            raise RuntimeError(f"control source/input hash changed: {key}")
    for method, mapping in hashes["score_files"].items():
        for cell, expected in mapping.items():
            if sha256_file(args.out / "scores" / method / f"{cell}.npz") != expected:
                raise RuntimeError(f"control score hash changed: {method}/{cell}")
        if sha256_file(args.out / "calibrations" / f"{method}.json") != hashes["calibrations"][method]:
            raise RuntimeError(f"control calibration hash changed: {method}")
    for cell, expected in hashes["matched_node_score_files"].items():
        path = args.out / "scores" / "matched_node_permutations" / f"{cell}.npz"
        if sha256_file(path) != expected:
            raise RuntimeError(f"matched node score hash changed: {cell}")
    path = args.out / "calibrations" / "matched_node_permutations.json"
    if sha256_file(path) != hashes["matched_node_calibrations"]:
        raise RuntimeError("matched node calibration hash changed")
    return manifest


def load_control_scores(out: Path, method: str):
    output = {}
    for cell in ELIGIBLE_CELLS:
        with np.load(out / "scores" / method / f"{cell}.npz") as stored:
            output[cell] = {key: stored[key].astype(float) for key in stored.files}
        expected = {"iu"}
        group = dataset_family(cell)
        other_groups = sorted({
            dataset_family(value) for value in ELIGIBLE_CELLS
            if dataset_family(value) != group
        })
        for lambda_, trust in candidates():
            key = candidate_key(lambda_, trust)
            expected.update((f"full__{key}", f"outer__{key}"))
            expected.update(
                f"inner={outer_group}__{key}"
                for outer_group in other_groups
            )
        if set(output[cell]) != expected:
            raise RuntimeError(f"control score registry changed: {method}/{cell}")
        if any(
            value.ndim != 1 or not np.isfinite(value).all()
            for value in output[cell].values()
        ):
            raise RuntimeError(f"invalid control score: {method}/{cell}")
    return output


def group_metric(scores, labels, group, score_name):
    from sklearn.metrics import roc_auc_score
    values = []
    for cell in ELIGIBLE_CELLS:
        if dataset_family(cell) != group:
            continue
        y = labels[cell]
        values.append(
            roc_auc_score(y, scores[cell][score_name])
            - roc_auc_score(y, scores[cell]["iu"])
        )
    return float(np.mean(values))


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    draws = values[indices].mean(axis=1)
    return [
        100 * float(np.quantile(draws, .025)),
        100 * float(np.quantile(draws, .975)),
    ]


def report(args):
    manifest = verify_controls(args)
    primary_scores, labels = load_after_freeze(args.development, args.bundle)
    _, primary_result, _, _ = read_selection(args.development)
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    primary_rows = primary_result["primary"]["outer_rows"]
    primary_by_group = {row["held_group"]: row for row in primary_rows}
    primary_vector = np.asarray([
        row["held_delta_auroc"] for row in primary_rows
    ], dtype=float)
    capacity, matched = {}, {}
    for method in CAPACITY_METHODS:
        scores = load_control_scores(args.out, method)
        own_rows = nested_rows(scores, labels, choose_one_se)
        own = np.asarray([row["held_delta_auroc"] for row in own_rows])
        control = []
        for group in groups:
            key = primary_by_group[group]["candidate_key"]
            control.append(group_metric(scores, labels, group, f"outer__{key}"))
        control = np.asarray(control)
        difference = primary_vector - control
        capacity[method] = {
            "own_nested_delta_pp": 100 * float(np.mean(own)),
            "own_nested_positive_groups": int(np.sum(own > 0)),
            "own_nested_rows": own_rows,
        }
        matched[method] = {
            "control_delta_vs_iu_pp": 100 * float(np.mean(control)),
            "real_minus_control_pp": 100 * float(np.mean(difference)),
            "real_minus_control_ci_pp": paired_interval(difference),
            "group_control_values_pp": {
                group: 100 * float(value)
                for group, value in zip(groups, control)
            },
        }

    node_vectors = []
    for replicate in range(N_NODE_PERMUTATIONS):
        values = []
        for group in groups:
            cell_values = []
            for cell in ELIGIBLE_CELLS:
                if dataset_family(cell) != group:
                    continue
                with np.load(
                    args.out / "scores" / "matched_node_permutations"
                    / f"{cell}.npz"
                ) as stored:
                    score = stored[f"outer__r{replicate:02d}"].astype(float)
                    iu = stored["iu"].astype(float)
                from sklearn.metrics import roc_auc_score
                cell_values.append(
                    roc_auc_score(labels[cell], score)
                    - roc_auc_score(labels[cell], iu)
                )
            values.append(float(np.mean(cell_values)))
        node_vectors.append(np.asarray(values))
    node_means = np.asarray([np.mean(value) for value in node_vectors])
    node_average = np.mean(node_vectors, axis=0)
    real_minus_node = primary_vector - node_average
    node_summary = {
        "n_permutations": N_NODE_PERMUTATIONS,
        "permutation_mean_delta_pp": 100 * float(np.mean(node_means)),
        "permutation_min_delta_pp": 100 * float(np.min(node_means)),
        "permutation_max_delta_pp": 100 * float(np.max(node_means)),
        "real_minus_mean_permutation_pp": 100 * float(np.mean(real_minus_node)),
        "real_minus_mean_permutation_ci_pp": paired_interval(real_minus_node),
        "randomization_p_greater_or_equal": float(
            (1 + np.sum(node_means >= np.mean(primary_vector)))
            / (1 + N_NODE_PERMUTATIONS)
        ),
        "replicate_delta_pp": (100 * node_means).tolist(),
    }
    dufs = matched["dufs_graph"]
    cross = matched["cross_only"]
    mechanism_gates = {
        "real_beats_mean_node_permutation": (
            node_summary["real_minus_mean_permutation_ci_pp"][0] > 0
        ),
        "node_randomization_p_at_most_0_05": (
            node_summary["randomization_p_greater_or_equal"] <= .05
        ),
        "real_beats_dufs_graph": dufs["real_minus_control_ci_pp"][0] > 0,
        "real_beats_cross_only": cross["real_minus_control_ci_pp"][0] > 0,
    }
    result = {
        "version": CONTROL_VERSION,
        "status": "PASS" if all(mechanism_gates.values()) else "FAIL",
        "development_fit_manifest_hash": manifest["development_fit_manifest_hash"],
        "primary_real_delta_pp": 100 * float(np.mean(primary_vector)),
        "capacity_matched_hpo": capacity,
        "primary_hyperparameter_matched": matched,
        "node_permutation_null": node_summary,
        "mechanism_gates": mechanism_gates,
        "claim_boundary": "post-selection retrospective mechanism controls",
    }
    write_json(args.out / "RESULT.json", result)
    lines = [
        "# Pooled Graph-Roughness V2 mechanism controls", "",
        f"**{result['status']}** for the complete registered graph-attribution gate.", "",
        f"Real residual graph: {result['primary_real_delta_pp']:+.3f}pp versus IU.", "",
        "| matched control | control vs IU (pp) | real − control (pp) | 95% CI (pp) |",
        "|---|---:|---:|---:|",
    ]
    for method, row in matched.items():
        lines.append(
            f"| `{method}` | {row['control_delta_vs_iu_pp']:+.3f} | "
            f"{row['real_minus_control_pp']:+.3f} | "
            f"[{row['real_minus_control_ci_pp'][0]:+.3f}, "
            f"{row['real_minus_control_ci_pp'][1]:+.3f}] |"
        )
    lines += [
        "", f"Twenty matched node permutations average "
        f"{node_summary['permutation_mean_delta_pp']:+.3f}pp; real minus their "
        f"mean is {node_summary['real_minus_mean_permutation_pp']:+.3f}pp "
        f"[{node_summary['real_minus_mean_permutation_ci_pp'][0]:+.3f}, "
        f"{node_summary['real_minus_mean_permutation_ci_pp'][1]:+.3f}]pp, "
        f"randomization p={node_summary['randomization_p_greater_or_equal']:.4f}.",
        "", "Controls were fitted and hashed without row-level targets after the "
        "primary hyperparameters were frozen. They are retrospective mechanism "
        "tests, not independent validation.", "",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "node": node_summary,
        "mechanism_gates": mechanism_gates,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "report"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "fit":
        fit(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
