#!/usr/bin/env python3
"""Factorial DUFS representation x graph audit with a length-conditional null."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.graph_topology import (  # noqa: E402
    adaptive_knn_graph,
    diffusion_edge_matched_graph,
    exact_length_permutations,
    extended_graph_diagnostics,
    holm_adjust,
    length_only_graph,
    matched_pair_permutations,
    mutual_knn_graph,
    propensity_crt_permutations,
    purity_against_permutations,
    radius_edge_matched_graph,
    sample_tie_diagnostics,
    self_safe_knn_graph,
    smoothness_against_permutations,
    unconditional_target_permutations,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    dufs_soft_gates,
    laplacian_iu_path,
    permute_graph,
    self_tuning_knn_graph,
)
from scripts.direct_dufs_graph_semantics_audit_v1 import (  # noqa: E402
    DUFS_EPOCHS,
    DUFS_SEEDS,
    K,
    LAMBDA,
    gate_from_raw,
    stable_seed,
)
from scripts.direct_dufs_length_drop_ablation_v1 import length_mask  # noqa: E402
from scripts.direct_dufs_length_residualization_v1 import (  # noqa: E402
    apply_residualizer,
    fit_residualizer,
    load_global_cells,
    load_processbench_model,
    load_ragtruth_split,
    median_abs_length_correlation,
)


VERSION = "direct-dufs-conditional-graph-topology-v1-reviewed-2026-08-20"
PROTOCOL = ROOT / "docs/experiments/DIRECT_DUFS_CONDITIONAL_GRAPH_TOPOLOGY_AUDIT_V1.md"
DEFAULT_OUT = ROOT / "results/direct_dufs_conditional_graph_topology_audit_v1"
REPRESENTATIONS = ("original", "drop_length", "train_residualized")
CANDIDATE_GRAPHS = (
    "union_knn_k7_self_safe",
    "radius_edge_matched_k7",
    "adaptive_knn_mean7_k3_25",
    "diffusion_edge_matched_base25_t2",
    "diffusion_edge_matched_base25_t4",
)
DECISION_GRAPHS = CANDIDATE_GRAPHS[:-1]
CONTROL_GRAPHS = (
    "deployed_union_knn_k7",
    "mutual_knn_k7",
    "length_only_knn_k7",
    "permuted_self_safe_union_knn_k7",
)
GRAPH_ORDER = CANDIDATE_GRAPHS + CONTROL_GRAPHS
TIE_SEEDS = (101, 211, 307)
PERMUTATIONS = 199
BOOTSTRAPS = 5000
EXACT_MOVABLE_FRACTION_MIN = 0.20
EXACT_MOVABLE_ROWS_MIN = 20
EXACT_MIXED_STRATA_MIN = 5
CRT_OVERLAP_FRACTION_MIN = 0.20
CRT_BRIER_TOLERANCE = 0.01
CRT_CALIBRATION_MAE_MAX = 0.10
PAIR_MOVABLE_FRACTION_MIN = 0.20
PAIR_DISCORDANT_MIN = 10
PAIR_P95_LOG_GAP_MAX = float(np.log(1.25))
PAIR_MAX_LOG_GAP_MAX = float(np.log(2.0))
ELIGIBLE_CELL_FRACTION_MIN = 2 / 3
HEALTHY_LARGEST_COMPONENT_MIN = 0.90
HEALTHY_ISOLATED_MAX = 0.05


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_roundtrip(value):
    """Give fresh and resumed checkpoints identical JSON scalar semantics."""

    return json.loads(json.dumps(_jsonable(value), sort_keys=True))


def _write_json(path: Path, value, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n"
    if not atomic:
        path.write_text(payload, encoding="utf-8")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _manifest(paths: list[Path]) -> list[dict]:
    rows = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": int(stat.st_size),
            "sha256": _sha256(path),
        })
    return rows


def _source_paths() -> list[Path]:
    return [
        PROTOCOL,
        Path(__file__).resolve(),
        ROOT / "scripts/test_direct_dufs_conditional_graph_topology_v1.py",
        ROOT / "scripts/plot_direct_dufs_conditional_graph_topology_v1.py",
        ROOT / "scripts/verify_direct_dufs_conditional_graph_topology_v1.py",
        ROOT / "spectral_utils/adapted_dufs.py",
        ROOT / "spectral_utils/dufs_liu_feature_contract.py",
        ROOT / "spectral_utils/feature_contract.py",
        ROOT / "spectral_utils/graph_topology.py",
        ROOT / "spectral_utils/laplacian_upcr.py",
        ROOT / "spectral_utils/ragtruth_evidence_contrast.py",
        ROOT / "spectral_utils/selectors/a2_groupfs.py",
        ROOT / "spectral_utils/specrage_views.py",
        ROOT / "spectral_utils/upcr.py",
        ROOT / "scripts/direct_dufs_graph_semantics_audit_v1.py",
        ROOT / "scripts/direct_dufs_length_drop_ablation_v1.py",
        ROOT / "scripts/direct_dufs_length_residualization_v1.py",
        ROOT / "scripts/gl_liu_v1/run.py",
    ]


def _input_paths() -> list[Path]:
    paths = [
        ROOT / "results/dependency_fusion_raw/cells.npz",
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/scores_dev.npz",
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/scores_test.npz",
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/diagnostics/fit_dev.json",
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/diagnostics/fit_test.json",
        ROOT / "local_cache/ragtruth_ec/dev/ragtruth_ec_train.pkl",
        ROOT / "local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl",
        ROOT / "local_cache/RAGTruth_official/dataset/response.jsonl",
    ]
    paths.extend(sorted((ROOT / "results/frozen_24cell_benchmark/diagnostics").glob("*.json")))
    paths.extend(sorted((ROOT / "results/frozen_24cell_benchmark/scores").glob("*.npz")))
    paths.extend(sorted((ROOT / "results/processbench_latent_state_v1/label_free_scores").glob("*.npz")))
    paths.extend(sorted(
        path
        for path in (ROOT / "local_cache/qwen25_15b_tokenizer").rglob("*")
        if path.is_file()
    ))
    for model in ("qwen3_4b", "qwen3_8b"):
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
            paths.append(
                ROOT
                / f"cache/localization/processbench/pb_{model}/processbench_{subset}.pkl"
            )
    return paths


def _run_definition() -> dict:
    source_manifest = _manifest(_source_paths())
    input_manifest = _manifest(_input_paths())
    body = {
        "version": VERSION,
        "protocol_sha256": _sha256(PROTOCOL),
        "sources": source_manifest,
        "inputs": input_manifest,
        "representations": REPRESENTATIONS,
        "candidate_graphs": CANDIDATE_GRAPHS,
        "decision_graphs": DECISION_GRAPHS,
        "control_graphs": CONTROL_GRAPHS,
        "tie_seeds": TIE_SEEDS,
        "k": K,
        "lambda": LAMBDA,
        "dufs_seeds": DUFS_SEEDS,
        "dufs_epochs": DUFS_EPOCHS,
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "conditional_nulls": {
            "exact_length": {
                "movable_fraction_min": EXACT_MOVABLE_FRACTION_MIN,
                "movable_rows_min": EXACT_MOVABLE_ROWS_MIN,
                "mixed_strata_min": EXACT_MIXED_STRATA_MIN,
            },
            "cross_fitted_propensity_crt": {
                "overlap_fraction_min": CRT_OVERLAP_FRACTION_MIN,
                "brier_tolerance": CRT_BRIER_TOLERANCE,
                "calibration_mae_max": CRT_CALIBRATION_MAE_MAX,
            },
            "adjacent_pair_sensitivity": {
                "movable_fraction_min": PAIR_MOVABLE_FRACTION_MIN,
                "discordant_pairs_min": PAIR_DISCORDANT_MIN,
                "p95_log_gap_max": PAIR_P95_LOG_GAP_MAX,
                "max_log_gap_max": PAIR_MAX_LOG_GAP_MAX,
            },
        },
        "eligible_cell_fraction_min": ELIGIBLE_CELL_FRACTION_MIN,
        "health": {
            "largest_component_min": HEALTHY_LARGEST_COMPONENT_MIN,
            "isolated_fraction_max": HEALTHY_ISOLATED_MAX,
        },
        "no_target_labels_in_fit": True,
        "no_cross_task_pooling": True,
    }
    fingerprint_payload = json.dumps(_jsonable(body), sort_keys=True).encode("utf-8")
    body["run_fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    return body


def _global_specs() -> list[dict]:
    cells = load_global_cells()
    frozen = ROOT / "results/frozen_24cell_benchmark"
    output = []
    for held in cells:
        diagnostic = json.loads(
            (frozen / "diagnostics" / f"{held['cell']}.json").read_text(encoding="utf-8")
        )
        score_file = np.load(
            frozen / "scores" / f"{held['cell']}.npz", allow_pickle=False
        )

        def fitter(features):
            return dufs_soft_gates(
                features, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )

        output.append({
            **held,
            "lane": "global24",
            "split": "leave_cell_out",
            "training_cells": [
                {
                    "cell": cell["cell"],
                    "matrix": cell["matrix"],
                    "names": cell["names"],
                    "length": cell["length"],
                }
                for cell in cells
                if cell["cell"] != held["cell"]
            ],
            "original_gates": gate_from_raw(diagnostic["dufs"]["raw_probabilities"]),
            "gate_fitter": fitter,
            "frozen_liu": -np.asarray(
                score_file["dufs_liu__lambda_0p1"], dtype=float
            ),
            "frozen_tolerance": 5e-4,
        })
    return output


def _processbench_specs() -> list[dict]:
    training = load_processbench_model("qwen3_4b")
    validation = load_processbench_model("qwen3_8b")
    frozen_root = ROOT / "results/processbench_latent_state_v1/label_free_scores"
    output = []
    for held in validation:

        def fitter(features):
            return adapted_dufs_soft_gates(
                features, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )

        gates, _ = fitter(np.asarray(held["matrix"], dtype=float).T)
        frozen = np.load(
            frozen_root / f"{held['cell']}.npz", allow_pickle=False
        )
        output.append({
            **held,
            "lane": "processbench",
            "split": "model_validation",
            "training_cells": [
                {
                    "cell": cell["cell"],
                    "matrix": cell["matrix"],
                    "names": cell["names"],
                    "length": cell["length"],
                }
                for cell in training
            ],
            "original_gates": gates,
            "gate_fitter": fitter,
            "frozen_liu": np.asarray(frozen["global_mixed_v2_dufs"], dtype=float),
            "frozen_tolerance": 1e-10,
        })
    return output


def _ragtruth_specs() -> list[dict]:
    training = load_ragtruth_split("dev")
    held = load_ragtruth_split("test")

    def fitter(features):
        return dufs_soft_gates(features, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)

    archive = np.load(
        ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1/scores_test.npz",
        allow_pickle=False,
    )
    return [{
        **held,
        "lane": "ragtruth",
        "split": "test_validation",
        "training_cells": [{
            "cell": training["cell"],
            "matrix": training["matrix"],
            "names": training["names"],
            "length": training["length"],
        }],
        "original_gates": held["original_gates"],
        "gate_fitter": fitter,
        "frozen_liu": np.asarray(
            archive["score__original30_full__dufs_liu"], dtype=float
        ),
        "frozen_tolerance": 1e-10,
    }]


def validation_specs() -> list[dict]:
    return _global_specs() + _processbench_specs() + _ragtruth_specs()


def build_representations(spec: dict) -> tuple[dict, dict]:
    matrix = np.asarray(spec["matrix"], dtype=float)
    names = tuple(str(item) for item in spec["names"])
    keep = length_mask(names)
    if np.all(keep):
        raise ValueError(f"{spec['cell']}: no explicit length coordinate")
    kept_names = tuple(name for name, selected in zip(names, keep) if selected)
    dropped_names = [name for name, selected in zip(names, keep) if not selected]
    no_length = np.asarray(matrix[:, keep], dtype=float)
    drop_gates, drop_gate_diag = spec["gate_fitter"](no_length.T)
    coefficients, residualizer_diag = fit_residualizer(
        spec["training_cells"], kept_names
    )
    residualized = apply_residualizer(
        no_length, kept_names, spec["length"], coefficients
    )
    residual_gates, residual_gate_diag = spec["gate_fitter"](residualized.T)
    representations = {
        "original": {
            "matrix": matrix,
            "names": names,
            "gates": np.asarray(spec["original_gates"], dtype=float),
        },
        "drop_length": {
            "matrix": no_length,
            "names": kept_names,
            "gates": np.asarray(drop_gates, dtype=float),
        },
        "train_residualized": {
            "matrix": residualized,
            "names": kept_names,
            "gates": np.asarray(residual_gates, dtype=float),
        },
    }
    diagnostics = {
        "lane": spec["lane"],
        "cell": spec["cell"],
        "dropped_features": dropped_names,
        "feature_counts": {
            name: int(value["matrix"].shape[1])
            for name, value in representations.items()
        },
        "median_abs_feature_length_spearman": {
            name: median_abs_length_correlation(value["matrix"], spec["length"])
            for name, value in representations.items()
        },
        "drop_gate": drop_gate_diag,
        "residual_gate": residual_gate_diag,
        "residualizer": residualizer_diag,
    }
    return representations, diagnostics


def build_graphs(
    samples: np.ndarray,
    length: np.ndarray,
    *,
    tie_keys: np.ndarray,
    seed: int,
) -> dict[str, tuple[object, dict]]:
    deployed_union = self_tuning_knn_graph(samples, k=K)
    union = self_safe_knn_graph(samples, k=K, tie_keys=tie_keys)
    edge_budget = int(union.nnz // 2)
    radius, radius_diag = radius_edge_matched_graph(
        samples, edge_count=edge_budget, scale_k=K, tie_keys=tie_keys
    )
    adaptive, adaptive_diag = adaptive_knn_graph(
        samples,
        mean_k=7,
        min_k=3,
        max_k=25,
        scale_k=K,
        rank_power=8.0,
        tie_keys=tie_keys,
    )
    diffusion_t2, diffusion_t2_diag = diffusion_edge_matched_graph(
        samples,
        edge_count=edge_budget,
        base_k=25,
        steps=2,
        row_keep=25,
        tie_keys=tie_keys,
    )
    diffusion_t4, diffusion_t4_diag = diffusion_edge_matched_graph(
        samples,
        edge_count=edge_budget,
        base_k=25,
        steps=4,
        row_keep=25,
        tie_keys=tie_keys,
    )
    permutation = np.random.default_rng(seed).permutation(len(samples))
    bandwidth = {"bandwidth_rule": "kth_positive_distinct_location"}
    return {
        "union_knn_k7_self_safe": (
            union, {"edge_budget": edge_budget, **bandwidth}
        ),
        "radius_edge_matched_k7": (radius, {**radius_diag, **bandwidth}),
        "adaptive_knn_mean7_k3_25": (
            adaptive, {**adaptive_diag, **bandwidth}
        ),
        "diffusion_edge_matched_base25_t2": (
            diffusion_t2, {**diffusion_t2_diag, **bandwidth}
        ),
        "diffusion_edge_matched_base25_t4": (
            diffusion_t4, {**diffusion_t4_diag, **bandwidth}
        ),
        "deployed_union_knn_k7": (
            deployed_union,
            {"historical_reproduction_only": True},
        ),
        "mutual_knn_k7": (
            mutual_knn_graph(samples, k=K, tie_keys=tie_keys),
            {"k": K, **bandwidth},
        ),
        "length_only_knn_k7": (
            length_only_graph(length, k=K, tie_keys=tie_keys),
            {"k": K, **bandwidth},
        ),
        "permuted_self_safe_union_knn_k7": (
            permute_graph(union, permutation),
            {
                "permutation_seed": int(seed),
                "edge_budget": edge_budget,
                **bandwidth,
            },
        ),
    }


def _prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _prepare_nulls(spec: dict) -> tuple[dict, dict]:
    target = np.asarray(spec["target"], dtype=int)
    length = np.asarray(spec["length"], dtype=float)
    base_seed = stable_seed(VERSION, spec["lane"], spec["cell"], "target_null")
    raw = unconditional_target_permutations(
        target, permutations=PERMUTATIONS, seed=base_seed
    )
    exact, exact_diag = exact_length_permutations(
        target, length, permutations=PERMUTATIONS, seed=base_seed + 1
    )
    crt, crt_diag = propensity_crt_permutations(
        target, length, permutations=PERMUTATIONS, seed=base_seed + 2
    )
    pair, pair_diag = matched_pair_permutations(
        target, length, permutations=PERMUTATIONS, seed=base_seed + 3
    )
    exact_eligible = bool(
        exact_diag["movable_fraction"] >= EXACT_MOVABLE_FRACTION_MIN
        and exact_diag["movable_rows"] >= EXACT_MOVABLE_ROWS_MIN
        and exact_diag["mixed_strata"] >= EXACT_MIXED_STRATA_MIN
    )
    crt_eligible = bool(
        crt_diag["overlap_fraction"] >= CRT_OVERLAP_FRACTION_MIN
        and crt_diag["brier"] <= crt_diag["constant_brier"] + CRT_BRIER_TOLERANCE
        and crt_diag["calibration_mae"] <= CRT_CALIBRATION_MAE_MAX
        and crt_diag["all_draws_binary"]
    )
    pair_eligible = bool(
        pair_diag["movable_fraction"] >= PAIR_MOVABLE_FRACTION_MIN
        and pair_diag["discordant_pairs"] >= PAIR_DISCORDANT_MIN
        and pair_diag["p95_log_length_gap"] <= PAIR_P95_LOG_GAP_MAX
        and pair_diag["max_log_length_gap"] <= PAIR_MAX_LOG_GAP_MAX
    )
    length_rng = np.random.default_rng(
        stable_seed(VERSION, spec["lane"], spec["cell"], "length_null")
    )
    length_perms = np.column_stack([
        length_rng.permutation(length) for _ in range(PERMUTATIONS)
    ])
    return {
        "raw": raw,
        "exact": exact,
        "crt": crt,
        "pair": pair,
        "length": length_perms,
    }, {
        "exact": {**exact_diag, "eligible": exact_eligible},
        "crt": {**crt_diag, "eligible": crt_eligible},
        "pair": {**pair_diag, "eligible": pair_eligible},
    }


def evaluate_representation(
    spec: dict,
    representation: str,
    current: dict,
    representation_diag: dict,
    nulls: dict,
    null_diagnostics: dict,
) -> list[dict]:
    target = np.asarray(spec["target"], dtype=int)
    length = np.asarray(spec["length"], dtype=float)
    matrix = np.asarray(current["matrix"], dtype=float)
    features = matrix.T
    samples = matrix * np.asarray(current["gates"], dtype=float)[None, :]
    duplicate_diagnostics = sample_tie_diagnostics(samples)
    rows: list[dict] = []
    for tie_seed in TIE_SEEDS:
        tie_rng = np.random.default_rng(stable_seed(
            VERSION,
            spec["lane"],
            spec["cell"],
            representation,
            str(tie_seed),
            "tie_keys",
        ))
        tie_keys = tie_rng.random(len(samples))
        if len(np.unique(tie_keys)) != len(tie_keys):
            raise RuntimeError("target-blind tie-key collision")
        graphs = build_graphs(
            samples,
            length,
            tie_keys=tie_keys,
            seed=stable_seed(
                VERSION,
                spec["lane"],
                spec["cell"],
                representation,
                str(tie_seed),
                "graph_control",
            ),
        )
        seed_rows: list[dict] = []
        for graph_name in GRAPH_ORDER:
            graph, construction = graphs[graph_name]
            target_results = {
                name: smoothness_against_permutations(graph, target, nulls[name])
                for name in ("raw", "exact", "crt", "pair")
            }
            purity_results = {
                name: purity_against_permutations(graph, target, nulls[name])
                for name in ("raw", "exact", "crt", "pair")
            }
            length_result = smoothness_against_permutations(
                graph, length, nulls["length"]
            )
            path = laplacian_iu_path(features, (0.0, LAMBDA), graph=graph)
            iu_score = -(path[0.0].w @ features)
            liu_score = -(path[LAMBDA].w @ features)
            iu_auc = float(roc_auc_score(target, iu_score))
            liu_auc = float(roc_auc_score(target, liu_score))
            health = extended_graph_diagnostics(graph)
            row = {
                "lane": spec["lane"],
                "cell": spec["cell"],
                "split": spec["split"],
                "representation": representation,
                "tie_seed": int(tie_seed),
                "graph": graph_name,
                "graph_role": (
                    "candidate" if graph_name in CANDIDATE_GRAPHS else "control"
                ),
                "n": int(len(target)),
                "positives": int(np.sum(target)),
                "exact_eligible": bool(null_diagnostics["exact"]["eligible"]),
                "crt_eligible": bool(null_diagnostics["crt"]["eligible"]),
                "pair_eligible": bool(null_diagnostics["pair"]["eligible"]),
                "conditional_eligible": bool(
                    null_diagnostics["exact"]["eligible"]
                    and null_diagnostics["crt"]["eligible"]
                ),
                "iu_auroc": iu_auc,
                "liu_auroc": liu_auc,
                "liu_delta_auroc": liu_auc - iu_auc,
                "median_abs_feature_length_spearman": (
                    representation_diag["median_abs_feature_length_spearman"][representation]
                ),
                **_prefixed("exact_null", null_diagnostics["exact"]),
                **_prefixed("crt_null", null_diagnostics["crt"]),
                **_prefixed("pair_null", null_diagnostics["pair"]),
                **_prefixed("raw_target", target_results["raw"]),
                **_prefixed("exact_target", target_results["exact"]),
                **_prefixed("crt_target", target_results["crt"]),
                **_prefixed("pair_target", target_results["pair"]),
                **_prefixed("length", length_result),
                **_prefixed("raw_purity", purity_results["raw"]),
                **_prefixed("exact_purity", purity_results["exact"]),
                **_prefixed("crt_purity", purity_results["crt"]),
                **_prefixed("pair_purity", purity_results["pair"]),
                **health,
                **_prefixed("sample_ties", duplicate_diagnostics),
                **_prefixed("construction", construction),
            }
            row["healthy_graph"] = bool(
                row["largest_component_fraction"] >= HEALTHY_LARGEST_COMPONENT_MIN
                and row["isolated_fraction"] <= HEALTHY_ISOLATED_MAX
            )
            if representation == "original" and graph_name == "deployed_union_knn_k7":
                error = float(np.max(np.abs(liu_score - spec["frozen_liu"])))
                correlation = float(np.corrcoef(liu_score, spec["frozen_liu"])[0, 1])
                row["frozen_liu_max_abs_error"] = error
                row["frozen_liu_correlation"] = correlation
                if error > spec["frozen_tolerance"] or correlation < 0.999999:
                    raise RuntimeError(
                        f"{spec['cell']}: frozen LIU reproduction failed "
                        f"(max={error}, corr={correlation})"
                    )
            seed_rows.append(row)
        candidate_indexes = [
            index for index, row in enumerate(seed_rows)
            if row["graph"] in CANDIDATE_GRAPHS
        ]
        for null_name in ("exact", "crt", "pair"):
            adjusted = holm_adjust(np.asarray([
                seed_rows[index][f"{null_name}_target_p_smoother"]
                for index in candidate_indexes
            ]))
            for index, value in zip(candidate_indexes, adjusted):
                seed_rows[index][f"{null_name}_target_p_holm"] = float(value)
            for row in seed_rows:
                row.setdefault(f"{null_name}_target_p_holm", float("nan"))
        rows.extend(seed_rows)
    return rows


def evaluate_cell(spec: dict) -> tuple[list[dict], dict]:
    nulls, null_diagnostics = _prepare_nulls(spec)
    representations, representation_diag = build_representations(spec)
    representation_diag["conditional_nulls"] = null_diagnostics
    rows: list[dict] = []
    for representation in REPRESENTATIONS:
        rows.extend(evaluate_representation(
            spec,
            representation,
            representations[representation],
            representation_diag,
            nulls,
            null_diagnostics,
        ))
    return rows, representation_diag


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = np.mean(values[draws], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_intervals(rows: list[dict]) -> list[dict]:
    output = []
    for lane in ("global24", "processbench", "ragtruth"):
        for representation in REPRESENTATIONS:
            for graph in GRAPH_ORDER:
                for tie_seed in TIE_SEEDS:
                    current = [
                        row for row in rows
                        if row["lane"] == lane
                        and row["representation"] == representation
                        and row["graph"] == graph
                        and row["tie_seed"] == tie_seed
                    ]
                    values = np.asarray([
                        row["liu_delta_auroc"] for row in current
                    ], dtype=float)
                    low, high = _bootstrap_interval(
                        values,
                        seed=stable_seed(
                            VERSION,
                            lane,
                            representation,
                            graph,
                            str(tie_seed),
                            "bootstrap",
                        ),
                    )
                    output.append({
                        "lane": lane,
                        "representation": representation,
                        "graph": graph,
                        "tie_seed": int(tie_seed),
                        "cells": len(values),
                        "mean_liu_delta_auroc": float(np.mean(values)),
                        "median_liu_delta_auroc": float(np.median(values)),
                        "ci_low": low,
                        "ci_high": high,
                        "positive_cells_fraction": float(np.mean(values > 0)),
                    })
    return output


def lane_summaries(rows: list[dict]) -> list[dict]:
    output = []
    for lane in ("global24", "processbench", "ragtruth"):
        for representation in REPRESENTATIONS:
            for graph in GRAPH_ORDER:
                for tie_seed in TIE_SEEDS:
                    current = [
                        row for row in rows
                        if row["lane"] == lane
                        and row["representation"] == representation
                        and row["graph"] == graph
                        and row["tie_seed"] == tie_seed
                    ]
                    exact = [row for row in current if row["exact_eligible"]]
                    crt = [row for row in current if row["crt_eligible"]]
                    pair = [row for row in current if row["pair_eligible"]]
                    deltas = np.asarray([
                        row["liu_delta_auroc"] for row in current
                    ], dtype=float)
                    exact_effects = np.asarray([
                        row["exact_target_effect"] for row in exact
                    ], dtype=float)
                    if len(exact_effects) >= 3:
                        rho = float(spearmanr(
                            exact_effects,
                            np.asarray([
                                row["liu_delta_auroc"] for row in exact
                            ], dtype=float),
                        ).statistic)
                    else:
                        rho = float("nan")

                    def conditional_fields(name: str, eligible: list[dict]) -> dict:
                        effects = np.asarray([
                            row[f"{name}_target_effect"] for row in eligible
                        ], dtype=float)
                        adjusted = np.asarray([
                            row[f"{name}_target_p_holm"] for row in eligible
                        ], dtype=float)
                        purities = np.asarray([
                            row[f"{name}_purity_excess"] for row in eligible
                        ], dtype=float)
                        return {
                            f"{name}_eligible_cells": int(len(eligible)),
                            f"{name}_eligible_fraction": float(len(eligible) / len(current)),
                            f"median_{name}_target_effect": (
                                float(np.median(effects)) if len(effects) else float("nan")
                            ),
                            f"fraction_{name}_effect_positive": (
                                float(np.mean(effects > 0)) if len(effects) else 0.0
                            ),
                            f"fraction_{name}_holm_p_le_0p05": (
                                float(np.mean(adjusted <= 0.05)) if len(adjusted) else 0.0
                            ),
                            f"median_{name}_purity_excess": (
                                float(np.median(purities)) if len(purities) else float("nan")
                            ),
                        }

                    output.append({
                        "lane": lane,
                        "representation": representation,
                        "graph": graph,
                        "tie_seed": int(tie_seed),
                        "graph_role": current[0]["graph_role"],
                        "cells": len(current),
                        "median_raw_target_effect": float(np.median([
                            row["raw_target_effect"] for row in current
                        ])),
                        **conditional_fields("exact", exact),
                        **conditional_fields("crt", crt),
                        **conditional_fields("pair", pair),
                        "median_length_effect": float(np.median([
                            row["length_effect"] for row in current
                        ])),
                        "healthy_graph_fraction": float(np.mean([
                            row["healthy_graph"] for row in current
                        ])),
                        "median_largest_component_fraction": float(np.median([
                            row["largest_component_fraction"] for row in current
                        ])),
                        "median_isolated_fraction": float(np.median([
                            row["isolated_fraction"] for row in current
                        ])),
                        "mean_iu_auroc": float(np.mean([
                            row["iu_auroc"] for row in current
                        ])),
                        "mean_liu_auroc": float(np.mean([
                            row["liu_auroc"] for row in current
                        ])),
                        "mean_liu_delta_auroc": float(np.mean(deltas)),
                        "spearman_exact_effect_vs_liu_delta": rho,
                    })
    return output


def _summary_lookup(
    summaries: list[dict],
    lane: str,
    representation: str,
    graph: str,
    tie_seed: int,
) -> dict:
    for row in summaries:
        if (
            row["lane"] == lane
            and row["representation"] == representation
            and row["graph"] == graph
            and row["tie_seed"] == tie_seed
        ):
            return row
    raise KeyError((lane, representation, graph, tie_seed))


def _interval_lookup(
    intervals: list[dict],
    lane: str,
    representation: str,
    graph: str,
    tie_seed: int,
) -> dict:
    for row in intervals:
        if (
            row["lane"] == lane
            and row["representation"] == representation
            and row["graph"] == graph
            and row["tie_seed"] == tie_seed
        ):
            return row
    raise KeyError((lane, representation, graph, tie_seed))


def decide(summaries: list[dict], intervals: list[dict], controls: dict) -> dict:
    topology = []
    for graph in DECISION_GRAPHS:
        lane_details = []
        for lane in ("global24", "processbench", "ragtruth"):
            representation_details = []
            for representation in ("original", "drop_length"):
                tie_details = []
                for tie_seed in TIE_SEEDS:
                    row = _summary_lookup(
                        summaries, lane, representation, graph, tie_seed
                    )

                    def null_gate(name: str) -> bool:
                        coverage = row[f"{name}_eligible_fraction"]
                        if lane == "ragtruth":
                            return bool(
                                coverage == 1.0
                                and row[f"fraction_{name}_effect_positive"] == 1.0
                                and row[f"fraction_{name}_holm_p_le_0p05"] == 1.0
                            )
                        return bool(
                            coverage >= ELIGIBLE_CELL_FRACTION_MIN
                            and row[f"fraction_{name}_effect_positive"] >= 2 / 3
                            and row[f"fraction_{name}_holm_p_le_0p05"] >= 0.5
                        )

                    exact_gate = null_gate("exact")
                    crt_gate = null_gate("crt")
                    healthy = bool(row["healthy_graph_fraction"] >= 0.90)
                    tie_details.append({
                        "tie_seed": int(tie_seed),
                        "exact_gate": exact_gate,
                        "crt_gate": crt_gate,
                        "health_gate": healthy,
                        "pass": bool(exact_gate and crt_gate and healthy),
                    })
                representation_details.append({
                    "representation": representation,
                    "tie_seeds": tie_details,
                    "pass": bool(all(item["pass"] for item in tie_details)),
                })
            lane_details.append({
                "lane": lane,
                "representations": representation_details,
                "conditional_geometry": bool(all(
                    item["pass"] for item in representation_details
                )),
            })
        utility_representations = []
        for representation in ("original", "drop_length"):
            tie_details = []
            for tie_seed in TIE_SEEDS:
                global_interval = _interval_lookup(
                    intervals, "global24", representation, graph, tie_seed
                )
                process_interval = _interval_lookup(
                    intervals, "processbench", representation, graph, tie_seed
                )
                rag_interval = _interval_lookup(
                    intervals, "ragtruth", representation, graph, tie_seed
                )
                tie_details.append({
                    "tie_seed": int(tie_seed),
                    "global_mean_positive": global_interval["mean_liu_delta_auroc"] > 0,
                    "global_ci_positive": global_interval["ci_low"] > 0,
                    "processbench_mean_positive": process_interval["mean_liu_delta_auroc"] > 0,
                    "ragtruth_nonnegative": rag_interval["mean_liu_delta_auroc"] >= 0,
                    "pass": bool(
                        global_interval["mean_liu_delta_auroc"] > 0
                        and global_interval["ci_low"] > 0
                        and process_interval["mean_liu_delta_auroc"] > 0
                        and rag_interval["mean_liu_delta_auroc"] >= 0
                    ),
                })
            utility_representations.append({
                "representation": representation,
                "tie_seeds": tie_details,
                "pass": bool(all(item["pass"] for item in tie_details)),
            })
        geometry_all = bool(all(item["conditional_geometry"] for item in lane_details))
        utility = bool(all(item["pass"] for item in utility_representations))
        topology.append({
            "graph": graph,
            "lanes": lane_details,
            "conditional_geometry_all_lanes": geometry_all,
            "utility_representations": utility_representations,
            "detector_utility": utility,
            "joint_pass": bool(geometry_all and utility),
        })
    if not controls.get("all_controls_pass", False):
        decision = "CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT"
    elif any(item["joint_pass"] for item in topology):
        decision = "ROBUST_LENGTH_CONDITIONAL_GEOMETRY_AND_UTILITY"
    elif any(item["conditional_geometry_all_lanes"] for item in topology):
        decision = "CONDITIONAL_GEOMETRY_WITHOUT_DETECTOR_UTILITY"
    else:
        decision = "NO_GRAPH_REVEALS_LENGTH_CONDITIONAL_TARGET_GEOMETRY"
    return {
        "decision": decision,
        "control_gate": bool(controls.get("all_controls_pass", False)),
        "topologies": topology,
    }


def control_checks(rows: list[dict]) -> dict:
    originals = [row for row in rows if row["representation"] == "original"]
    length_control = [
        row for row in originals if row["graph"] == "length_only_knn_k7"
    ]
    permuted = [
        row for row in originals
        if row["graph"] == "permuted_self_safe_union_knn_k7"
    ]
    union = [
        row for row in originals if row["graph"] == "union_knn_k7_self_safe"
    ]
    deployed = [
        row for row in originals if row["graph"] == "deployed_union_knn_k7"
    ]
    edge_budget_checks = []
    for lane in ("global24", "processbench", "ragtruth"):
        lane_rows = [row for row in rows if row["lane"] == lane]
        for representation in REPRESENTATIONS:
            for tie_seed in TIE_SEEDS:
                current = [
                    row for row in lane_rows
                    if row["representation"] == representation
                    and row["tie_seed"] == tie_seed
                ]
                by_cell = {}
                for row in current:
                    by_cell.setdefault(row["cell"], {})[row["graph"]] = row
                for cell, mapping in by_cell.items():
                    baseline_edges = mapping["union_knn_k7_self_safe"]["n_edges"]
                    for graph in (
                        "radius_edge_matched_k7",
                        "diffusion_edge_matched_base25_t2",
                        "diffusion_edge_matched_base25_t4",
                    ):
                        edge_budget_checks.append({
                            "lane": lane,
                            "cell": cell,
                            "representation": representation,
                            "tie_seed": int(tie_seed),
                            "graph": graph,
                            "expected_edges": baseline_edges,
                            "observed_edges": mapping[graph]["n_edges"],
                            "pass": mapping[graph]["n_edges"] == baseline_edges,
                        })
    exact_length_eligible = [row for row in length_control if row["exact_eligible"]]
    crt_length_eligible = [row for row in length_control if row["crt_eligible"]]
    eligibility_by_lane = []
    baseline_for_eligibility = [
        row for row in union if row["tie_seed"] == TIE_SEEDS[0]
    ]
    for lane in ("global24", "processbench", "ragtruth"):
        current = [row for row in baseline_for_eligibility if row["lane"] == lane]
        exact_fraction = float(np.mean([row["exact_eligible"] for row in current]))
        crt_fraction = float(np.mean([row["crt_eligible"] for row in current]))
        eligibility_by_lane.append({
            "lane": lane,
            "cells": len(current),
            "exact_eligible_fraction": exact_fraction,
            "crt_eligible_fraction": crt_fraction,
            "pass": bool(
                exact_fraction >= ELIGIBLE_CELL_FRACTION_MIN
                and crt_fraction >= ELIGIBLE_CELL_FRACTION_MIN
            ),
        })
    length_signal_fraction = float(np.mean([
        row["length_effect"] > 0 for row in length_control
    ]))
    median_length_signal = float(np.median([
        row["length_effect"] for row in length_control
    ]))
    exact_false_positive_fraction = float(np.mean([
        row["exact_target_p_smoother"] <= 0.05
        for row in exact_length_eligible
    ])) if exact_length_eligible else 1.0
    crt_false_positive_fraction = float(np.mean([
        row["crt_target_p_smoother"] <= 0.05
        for row in crt_length_eligible
    ])) if crt_length_eligible else 1.0
    permuted_false_positive_fraction = float(np.mean([
        row["raw_target_p_smoother"] <= 0.05 for row in permuted
    ]))
    false_positive_by_lane = []
    for lane in ("global24", "processbench", "ragtruth"):
        exact_lane = [
            row for row in exact_length_eligible if row["lane"] == lane
        ]
        crt_lane = [
            row for row in crt_length_eligible if row["lane"] == lane
        ]
        permuted_lane = [row for row in permuted if row["lane"] == lane]
        exact_rate = float(np.mean([
            row["exact_target_p_smoother"] <= 0.05 for row in exact_lane
        ])) if exact_lane else 1.0
        crt_rate = float(np.mean([
            row["crt_target_p_smoother"] <= 0.05 for row in crt_lane
        ])) if crt_lane else 1.0
        permuted_rate = float(np.mean([
            row["raw_target_p_smoother"] <= 0.05 for row in permuted_lane
        ])) if permuted_lane else 1.0
        threshold = 0.0 if lane == "ragtruth" else 0.15
        false_positive_by_lane.append({
            "lane": lane,
            "threshold": threshold,
            "length_exact_false_positive_fraction": exact_rate,
            "length_crt_false_positive_fraction": crt_rate,
            "permuted_raw_false_positive_fraction": permuted_rate,
            "pass": bool(
                exact_rate <= threshold
                and crt_rate <= threshold
                and permuted_rate <= threshold
            ),
        })
    all_edge_budgets_exact = bool(all(row["pass"] for row in edge_budget_checks))
    all_radius_boundaries_proven = bool(all(
        row.get("construction_candidate_boundary_proven", False)
        for row in rows if row["graph"] == "radius_edge_matched_k7"
    ))
    all_adaptive_means_exact = bool(all(
        abs(row.get("construction_directed_k_mean", 0.0) - 7.0) < 1e-12
        for row in rows if row["graph"] == "adaptive_knn_mean7_k3_25"
    ))
    all_frozen_scores_reproduced = bool(all(
        row.get("frozen_liu_max_abs_error", float("inf")) <= (
            5e-4 if row["lane"] == "global24" else 1e-10
        )
        and row.get("frozen_liu_correlation", -1.0) >= 0.999999
        for row in deployed
    ))
    length_positive_pass = bool(
        length_signal_fraction >= 0.90 and median_length_signal > 0.20
    )
    conditional_specificity_pass = bool(all(
        row["length_exact_false_positive_fraction"] <= row["threshold"]
        and row["length_crt_false_positive_fraction"] <= row["threshold"]
        for row in false_positive_by_lane
    ))
    permuted_negative_pass = bool(all(
        row["permuted_raw_false_positive_fraction"] <= row["threshold"]
        for row in false_positive_by_lane
    ))
    eligibility_pass = bool(all(row["pass"] for row in eligibility_by_lane))
    output = {
        "length_positive_control": {
            "cells": len(length_control),
            "length_effect_positive_fraction": length_signal_fraction,
            "median_length_effect": median_length_signal,
            "exact_target_false_positive_fraction": exact_false_positive_fraction,
            "crt_target_false_positive_fraction": crt_false_positive_fraction,
            "positive_pass": length_positive_pass,
            "conditional_specificity_pass": conditional_specificity_pass,
        },
        "permuted_negative_control": {
            "cells": len(permuted),
            "raw_false_positive_fraction": permuted_false_positive_fraction,
            "median_raw_target_effect": float(np.median([
                row["raw_target_effect"] for row in permuted
            ])),
            "median_union_minus_permuted_raw_effect": float(np.median([
                original["raw_target_effect"] - shuffled["raw_target_effect"]
                for original, shuffled in zip(
                    sorted(union, key=lambda row: (
                        row["lane"], row["cell"], row["tie_seed"]
                    )),
                    sorted(permuted, key=lambda row: (
                        row["lane"], row["cell"], row["tie_seed"]
                    )),
                )
            ])),
            "pass": permuted_negative_pass,
        },
        "conditional_eligibility_by_lane": eligibility_by_lane,
        "conditional_eligibility_pass": eligibility_pass,
        "false_positive_controls_by_lane": false_positive_by_lane,
        "edge_budget_checks": edge_budget_checks,
        "all_edge_budgets_exact": all_edge_budgets_exact,
        "all_radius_boundaries_proven": all_radius_boundaries_proven,
        "all_adaptive_means_exact": all_adaptive_means_exact,
        "all_frozen_scores_reproduced": all_frozen_scores_reproduced,
    }
    output["all_controls_pass"] = bool(
        length_positive_pass
        and conditional_specificity_pass
        and permuted_negative_pass
        and eligibility_pass
        and all_edge_budgets_exact
        and all_radius_boundaries_proven
        and all_adaptive_means_exact
        and all_frozen_scores_reproduced
    )
    return output


def build_report(
    out: Path,
    decision: dict,
    summaries: list[dict],
    intervals: list[dict],
    controls: dict,
) -> None:
    lines = [
        "# Direct DUFS conditional graph-topology audit v1",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        "This retrospective closure audit requires agreement between exact-length swaps and a cross-fitted flexible propensity CRT. Raw smoothness, coarse length bins, and a single tie resolution cannot establish a hallucination manifold.",
        "",
        "## Candidate decisions",
        "",
        "| Graph | Conditional geometry in all lanes | Detector utility | Joint pass |",
        "|---|---:|---:|---:|",
    ]
    for row in decision["topologies"]:
        lines.append(
            f"| {row['graph']} | {str(row['conditional_geometry_all_lanes'])} | "
            f"{str(row['detector_utility'])} | {str(row['joint_pass'])} |"
        )
    lines += [
        "",
        "## Original-representation validation summary (worst tie seed)",
        "",
        "| Lane | Graph | Raw effect | Exact effect | Exact sig. | CRT effect | CRT sig. | Healthy | LIU ΔAUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in ("global24", "processbench", "ragtruth"):
        for graph in GRAPH_ORDER:
            current = [
                _summary_lookup(summaries, lane, "original", graph, tie_seed)
                for tie_seed in TIE_SEEDS
            ]
            interval_rows = [
                _interval_lookup(intervals, lane, "original", graph, tie_seed)
                for tie_seed in TIE_SEEDS
            ]
            lines.append(
                f"| {lane} | {graph} | "
                f"{min(row['median_raw_target_effect'] for row in current):.1%} | "
                f"{min(row['median_exact_target_effect'] for row in current):.1%} | "
                f"{min(row['fraction_exact_holm_p_le_0p05'] for row in current):.0%} | "
                f"{min(row['median_crt_target_effect'] for row in current):.1%} | "
                f"{min(row['fraction_crt_holm_p_le_0p05'] for row in current):.0%} | "
                f"{min(row['healthy_graph_fraction'] for row in current):.0%} | "
                f"{min(row['mean_liu_delta_auroc'] for row in interval_rows):+.4f} |"
            )
    lines += [
        "",
        "## Controls",
        "",
        f"- Overall fail-closed control gate: {controls['all_controls_pass']}.",
        f"- Length-only graph: median length effect {controls['length_positive_control']['median_length_effect']:.1%}; exact/CRT false-positive fractions {controls['length_positive_control']['exact_target_false_positive_fraction']:.1%}/{controls['length_positive_control']['crt_target_false_positive_fraction']:.1%}.",
        f"- Permuted-union raw false-positive fraction: {controls['permuted_negative_control']['raw_false_positive_fraction']:.1%}.",
        f"- Exact edge budgets: {controls['all_edge_budgets_exact']}; radius boundary proof: {controls['all_radius_boundaries_proven']}; adaptive mean-k: {controls['all_adaptive_means_exact']}; deployed-score reproduction: {controls['all_frozen_scores_reproduced']}.",
        "",
        "## Interpretation",
        "",
    ]
    if decision["decision"] == "CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT":
        lines.append(
            "At least one predeclared positive, negative, eligibility, construction, or replay control failed. Geometry outcomes are therefore invalidated rather than interpreted."
        )
    elif decision["decision"] == "NO_GRAPH_REVEALS_LENGTH_CONDITIONAL_TARGET_GEOMETRY":
        lines.append(
            "None of the fixed graph semantics survives both registered length-conditional nulls, both required representations, all three tie resolutions, graph health, and all three lanes."
        )
    elif decision["decision"] == "CONDITIONAL_GEOMETRY_WITHOUT_DETECTOR_UTILITY":
        lines.append(
            "At least one fixed topology retains conditional target geometry, but the same topology does not produce robust LIU ranking utility. The geometry is descriptive and is not a supported detector mechanism."
        )
    else:
        lines.append(
            "One fixed topology passes both the conditional geometry and detector-utility gates in all lanes. Because the caches are historically opened, this is still a retrospective candidate requiring a separately frozen prospective validation."
        )
    lines += [
        "",
        "Global answer hallucination, ProcessBench process error, and RAGTruth response hallucination remain separate estimands. See the machine-readable tables for the drop-length and train-residualized robustness arms.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    definition = _run_definition()
    definition_path = out / "RUN_DEFINITION.json"
    if definition_path.exists():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != definition["run_fingerprint"]:
            raise RuntimeError("output directory belongs to a different frozen run")
    else:
        _write_json(definition_path, definition, atomic=True)

    all_rows: list[dict] = []
    representation_diagnostics = []
    checkpoint_root = out / "checkpoints"
    for spec in validation_specs():
        checkpoints = {
            representation: checkpoint_root / (
                f"{spec['lane']}__{spec['cell']}__{representation}.json"
            )
            for representation in REPRESENTATIONS
        }
        existing_payloads = {}
        for representation, checkpoint in checkpoints.items():
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("run_fingerprint") != definition["run_fingerprint"]:
                    raise RuntimeError(f"stale checkpoint: {checkpoint}")
                existing_payloads[representation] = payload
        missing = [
            representation for representation in REPRESENTATIONS
            if representation not in existing_payloads
        ]
        if missing:
            print(
                f"prepare {spec['lane']} {spec['cell']} missing={','.join(missing)}",
                flush=True,
            )
            nulls, null_diagnostics = _prepare_nulls(spec)
            representations, diagnostic = build_representations(spec)
            diagnostic["conditional_nulls"] = null_diagnostics
        else:
            diagnostic = existing_payloads[REPRESENTATIONS[0]][
                "representation_diagnostics"
            ]
        for representation in REPRESENTATIONS:
            if representation in existing_payloads:
                rows = existing_payloads[representation]["rows"]
                print(
                    f"resume {spec['lane']} {spec['cell']} {representation} "
                    f"({len(rows)} rows)",
                    flush=True,
                )
            else:
                print(
                    f"run {spec['lane']} {spec['cell']} {representation}",
                    flush=True,
                )
                rows = evaluate_representation(
                    spec,
                    representation,
                    representations[representation],
                    diagnostic,
                    nulls,
                    null_diagnostics,
                )
                rows = _canonical_roundtrip(rows)
                _write_json(checkpoints[representation], {
                    "run_fingerprint": definition["run_fingerprint"],
                    "rows": rows,
                    "representation_diagnostics": diagnostic,
                }, atomic=True)
            all_rows.extend(_canonical_roundtrip(rows))
        representation_diagnostics.append(_canonical_roundtrip(diagnostic))

    summaries = lane_summaries(all_rows)
    intervals = paired_intervals(all_rows)
    controls = control_checks(all_rows)
    decision = decide(summaries, intervals, controls)
    _write_csv(out / "CELL_GRAPH_METRICS.csv", all_rows)
    _write_csv(out / "LANE_GRAPH_SUMMARY.csv", summaries)
    _write_csv(out / "PAIRED_INTERVALS.csv", intervals)
    _write_json(out / "REPRESENTATION_DIAGNOSTICS.json", representation_diagnostics)
    _write_json(out / "CONTROL_CHECKS.json", controls)
    _write_json(out / "DECISION.json", {
        "version": VERSION,
        **decision,
        "no_cross_task_pooling": True,
        "retrospective": True,
    })
    build_report(out, decision, summaries, intervals, controls)
    print(json.dumps(_jsonable(decision), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
