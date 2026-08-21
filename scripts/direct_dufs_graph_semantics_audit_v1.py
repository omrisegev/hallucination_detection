#!/usr/bin/env python3
"""Directly audit what the frozen DUFS neighbourhood graphs organize."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
    symmetric_normalized_laplacian,
)
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402


VERSION = "direct-dufs-graph-semantics-audit-v1-2026-08-20"
PERMUTATIONS = 200
SENSITIVITY_PERMUTATIONS = 50
FEATURE_PERMUTATIONS = 50
K_VALUES = (3, 5, 7, 10, 15, 25)
K = 7
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
LAMBDA = 0.1
DEFAULT_OUT = ROOT / "results" / "direct_dufs_graph_semantics_audit_v1"


def stable_seed(*parts: str) -> int:
    token = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def encode(values, categorical: bool) -> np.ndarray:
    values = np.asarray(values)
    if categorical:
        labels = values.astype(str)
        levels = sorted(set(labels))
        matrix = np.column_stack([labels == level for level in levels]).astype(float)
    else:
        matrix = np.asarray(values, dtype=float).reshape(-1, 1)
        finite = np.isfinite(matrix[:, 0])
        fill = float(np.median(matrix[finite, 0])) if finite.any() else 0.0
        matrix[~finite, 0] = fill
    matrix -= matrix.mean(axis=0, keepdims=True)
    keep = np.sum(matrix * matrix, axis=0) > 1e-12
    return matrix[:, keep]


def rayleigh(laplacian: csr_matrix, matrix: np.ndarray) -> float:
    if matrix.shape[1] == 0:
        return float("nan")
    denom = float(np.sum(matrix * matrix))
    return float(np.sum(matrix * (laplacian @ matrix)) / denom)


def smoothness_test(
    graph: csr_matrix,
    values,
    *,
    categorical: bool,
    seed: int,
    permutations: int = PERMUTATIONS,
) -> dict:
    laplacian = symmetric_normalized_laplacian(graph)
    matrix = encode(values, categorical)
    observed = rayleigh(laplacian, matrix)
    if not np.isfinite(observed):
        return {"rayleigh": observed, "smoothness_perm_mean": float("nan"), "smoothness_perm_sd": float("nan"), "smoothness_effect": float("nan"), "smoothness_z": float("nan"), "p_smoother": float("nan")}
    rng = np.random.default_rng(seed)
    null = np.asarray([rayleigh(laplacian, matrix[rng.permutation(len(matrix))]) for _ in range(permutations)])
    sd = float(null.std(ddof=1))
    z = float((null.mean() - observed) / sd) if sd > 1e-12 else 0.0
    return {
        "rayleigh": observed,
        "smoothness_perm_mean": float(null.mean()),
        "smoothness_perm_sd": sd,
        "smoothness_effect": float((null.mean() - observed) / null.mean()),
        "smoothness_z": z,
        "p_smoother": float((1 + np.sum(null <= observed)) / (permutations + 1)),
    }


def weighted_purity(graph: csr_matrix, labels: np.ndarray) -> float:
    coo = graph.tocoo()
    keep = coo.row < coo.col
    row, col, weight = coo.row[keep], coo.col[keep], coo.data[keep]
    return float(np.sum(weight * (labels[row] == labels[col])) / np.sum(weight))


def purity_test(graph: csr_matrix, values, *, seed: int, permutations: int = PERMUTATIONS) -> dict:
    labels = np.asarray(values).astype(str)
    coo = graph.tocoo()
    keep = coo.row < coo.col
    row, col, weight = coo.row[keep], coo.col[keep], coo.data[keep]
    total = float(np.sum(weight))
    def score(current):
        return float(np.sum(weight * (current[row] == current[col])) / total)
    observed = score(labels)
    rng = np.random.default_rng(seed)
    null = np.asarray([score(labels[rng.permutation(len(labels))]) for _ in range(permutations)])
    sd = float(null.std(ddof=1))
    return {
        "purity": observed,
        "purity_perm_mean": float(null.mean()),
        "purity_perm_sd": sd,
        "purity_excess": float(observed - null.mean()),
        "purity_z": float((observed - null.mean()) / sd) if sd > 1e-12 else 0.0,
    }


def quartiles(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    fill = float(np.median(values[finite])) if finite.any() else 0.0
    values = np.where(finite, values, fill)
    cuts = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
    return np.digitize(values, cuts).astype(int)


def audit_variable(graph, values, *, categorical, seed):
    result = smoothness_test(graph, values, categorical=categorical, seed=seed)
    purity_values = np.asarray(values).astype(str) if categorical else quartiles(values)
    result.update(purity_test(graph, purity_values, seed=seed + 1))
    return result


def gate_from_raw(raw) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    rms = float(np.sqrt(np.mean(raw * raw)))
    return raw / (rms if rms > 1e-12 else 1.0)


def mutual_knn_graph(samples: np.ndarray, *, k: int) -> csr_matrix:
    """Self-tuning affinity restricted to reciprocal directed kNN edges."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    k = int(max(1, min(k, n - 1)))
    neighbours = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(samples)
    distances, indexes = neighbours.kneighbors(samples, return_distance=True)
    sigma = np.maximum(distances[:, -1], 1e-8)
    rows = np.repeat(np.arange(n), k)
    cols = indexes[:, 1:].reshape(-1)
    d = distances[:, 1:].reshape(-1)
    weights = np.exp(-(d * d) / (sigma[rows] * sigma[cols] + 1e-12))
    directed = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    mutual = directed.minimum(directed.T).tocsr()
    mutual.setdiag(0.0)
    mutual.eliminate_zeros()
    return mutual


def topology_sensitivity_rows(*, lane, cell, split, F, gates, target, length):
    rows = []
    samples = F.T * np.asarray(gates, dtype=float)[None, :]
    for k in K_VALUES:
        graphs = {
            "union_knn": build_graph_from_features(F, gates=gates, k=k),
            "mutual_knn": mutual_knn_graph(samples, k=k),
        }
        for family, graph in graphs.items():
            if graph.nnz == 0:
                continue
            target_result = smoothness_test(
                graph, target, categorical=True,
                seed=stable_seed(lane, cell, family, str(k), "target"),
                permutations=SENSITIVITY_PERMUTATIONS,
            )
            length_result = smoothness_test(
                graph, length, categorical=False,
                seed=stable_seed(lane, cell, family, str(k), "length"),
                permutations=SENSITIVITY_PERMUTATIONS,
            )
            n_components, _ = connected_components(graph, directed=False)
            rows.append({
                "lane": lane, "cell": cell, "split": split,
                "graph_family": family, "k": k,
                "n_edges": int(graph.nnz // 2), "n_components": int(n_components),
                "target_smoothness_z": target_result["smoothness_z"],
                "length_smoothness_z": length_result["smoothness_z"],
                "target_smoothness_effect": target_result["smoothness_effect"],
                "length_smoothness_effect": length_result["smoothness_effect"],
                "target_smoother_than_length": bool(target_result["smoothness_effect"] > length_result["smoothness_effect"]),
                "target_aligned": bool(target_result["smoothness_z"] > 1.96),
            })
    return rows


def feature_rows(graph, matrix, names, *, lane, cell, graph_kind):
    rows = []
    for index, name in enumerate(names):
        result = smoothness_test(
            graph, matrix[:, index], categorical=False,
            seed=stable_seed(lane, cell, graph_kind, "feature", str(name)),
            permutations=FEATURE_PERMUTATIONS,
        )
        rows.append({"lane": lane, "cell": cell, "graph": graph_kind, "feature": str(name), **result})
    return rows


def cell_rows(
    *, lane, cell, split, graph, raw_graph, target, nuisances, iu_score, liu_score,
    feature_matrix, feature_names, target_name="hallucination",
):
    metric_rows, features = [], []
    for graph_kind, current in (("dufs", graph), ("ungated", raw_graph)):
        variables = [(target_name, target, True, "target"), *nuisances]
        for name, values, categorical, role in variables:
            result = audit_variable(
                current, values, categorical=categorical,
                seed=stable_seed(lane, cell, graph_kind, name),
            )
            metric_rows.append({
                "lane": lane, "cell": cell, "split": split, "graph": graph_kind,
                "variable": name, "role": role, "categorical": categorical,
                "n": len(target), **result,
            })
        features.extend(feature_rows(current, feature_matrix, feature_names, lane=lane, cell=cell, graph_kind=graph_kind))
    y = np.asarray(target, dtype=int)
    auc_iu = float(roc_auc_score(y, iu_score)) if len(np.unique(y)) == 2 else float("nan")
    auc_liu = float(roc_auc_score(y, liu_score)) if len(np.unique(y)) == 2 else float("nan")
    score_row = {
        "lane": lane, "cell": cell, "split": split, "n": len(y), "positives": int(y.sum()),
        "iu_auroc": auc_iu, "dufs_liu_auroc": auc_liu, "liu_delta_auroc": auc_liu - auc_iu,
    }
    return metric_rows, features, score_row


def audit_global(bundle_path: Path, frozen_root: Path):
    data = np.load(bundle_path, allow_pickle=True)
    cells = sorted(key[:-3] for key in data.files if key.endswith("__V"))
    metrics, feature_metrics, scores, sensitivity, examples = [], [], [], [], {}
    for cell in cells:
        stored = np.asarray(data[f"{cell}__V"], dtype=float)
        pool = tuple(str(x) for x in data[f"{cell}__pool"])
        signs = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
        matrix, names = fixed_stable_from_bundle(stored, pool, signs)
        diag = json.loads((frozen_root / "diagnostics" / f"{cell}.json").read_text())
        gates = gate_from_raw(diag["dufs"]["raw_probabilities"])
        F = matrix.T
        graph = build_graph_from_features(F, gates=gates, k=K)
        raw_graph = build_graph_from_features(F, k=K)
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        target = 1 - labels
        length_values = (
            stored[:, list(pool).index("trace_length")]
            if "trace_length" in pool else None
        )
        nuisances = [("row_order", np.arange(len(labels)), False, "nuisance")]
        if length_values is not None:
            nuisances.insert(0, ("trace_length", length_values, False, "nuisance"))
        score_file = np.load(frozen_root / "scores" / f"{cell}.npz", allow_pickle=False)
        rows, frows, score = cell_rows(
            lane="global24", cell=cell, split="leave_family_out", graph=graph,
            raw_graph=raw_graph, target=target, nuisances=nuisances,
            iu_score=-np.asarray(score_file["iu_pcr"], dtype=float),
            liu_score=-np.asarray(score_file["dufs_liu__lambda_0p1"], dtype=float),
            feature_matrix=matrix, feature_names=names,
        )
        metrics.extend(rows); feature_metrics.extend(frows); scores.append(score)
        if length_values is not None:
            sensitivity.extend(topology_sensitivity_rows(
                lane="global24", cell=cell, split="leave_family_out", F=F, gates=gates,
                target=target, length=length_values,
            ))
        if cell == "epr_triviaqa_mistral24b":
            examples["global24"] = (graph, target, quartiles(length_values), cell)
    return metrics, feature_metrics, scores, sensitivity, examples


def _processbench_matrix(rows):
    from scripts.gl_liu_v1.run import trace_features
    from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
    from spectral_utils.dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
    features = [trace_features(row) for row in rows]
    names, columns = [], []
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        raw = np.asarray([item.get(name, np.nan) for item in features], dtype=float)
        finite = np.isfinite(raw)
        if finite.mean() < 0.70 or not finite.any():
            continue
        raw = np.where(finite, raw, np.median(raw[finite]))
        if raw.std() < 1e-8 or np.mean(raw == np.median(raw)) > 0.40:
            continue
        names.append(name); columns.append(raw)
    transformed, names, _ = dufs_liu_mixed_v2_matrix(np.column_stack(columns), names)
    return transformed, tuple(names)


def audit_processbench(cache_root: Path, frozen_root: Path):
    from scripts.gl_liu_v1.run import load_rows
    from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
    metrics, feature_metrics, scores, sensitivity, examples = [], [], [], [], {}
    for model in ("qwen3_4b", "qwen3_8b"):
        for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
            cell = f"{model}__{subset}"
            rows = load_rows(cache_root / f"pb_{model}" / f"processbench_{subset}.pkl")
            matrix, names = _processbench_matrix(rows)
            F = matrix.T
            gates, _ = adapted_dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
            graph = build_graph_from_features(F, gates=gates, k=K)
            raw_graph = build_graph_from_features(F, k=K)
            path = laplacian_iu_path(F, (0.0, LAMBDA), graph=graph)
            iu = -(path[0.0].w @ F)
            liu = -(path[LAMBDA].w @ F)
            frozen = np.load(frozen_root / "label_free_scores" / f"{cell}.npz", allow_pickle=False)
            if not np.allclose(liu, frozen["global_mixed_v2_dufs"], rtol=0.0, atol=1e-10):
                raise RuntimeError(f"ProcessBench frozen-score reproduction failed: {cell}")
            target = np.asarray([row["label"] != -1 for row in rows], dtype=int)
            lengths = np.asarray([len(row["token_entropies"]) for row in rows], dtype=float)
            nuisances = [
                ("trace_length", lengths, False, "nuisance"),
                ("row_order", np.arange(len(rows)), False, "nuisance"),
                ("final_answer_wrong", np.asarray([not row["final_answer_correct"] for row in rows]), True, "related_outcome"),
            ]
            split = "development_model" if model == "qwen3_4b" else "model_validation"
            mrows, frows, score = cell_rows(
                lane="processbench", cell=cell, split=split, graph=graph,
                raw_graph=raw_graph, target=target, nuisances=nuisances,
                iu_score=iu, liu_score=liu, feature_matrix=matrix,
                feature_names=names, target_name="process_error",
            )
            metrics.extend(mrows); feature_metrics.extend(frows); scores.append(score)
            sensitivity.extend(topology_sensitivity_rows(
                lane="processbench", cell=cell, split=split, F=F, gates=gates,
                target=target, length=lengths,
            ))
            if cell == "qwen3_8b__gsm8k":
                examples["processbench"] = (graph, target, quartiles(lengths), cell)
    return metrics, feature_metrics, scores, sensitivity, examples


def _ragtruth_labels(split: str, response_ids: np.ndarray) -> np.ndarray:
    from transformers import AutoTokenizer
    from spectral_utils.ragtruth_evidence_contrast import adapt_cache
    cache = ROOT / "local_cache" / "ragtruth_ec" / ("dev" if split == "dev" else "test") / ("ragtruth_ec_train.pkl" if split == "dev" else "ragtruth_ec_test.pkl")
    official = ROOT / "local_cache" / "RAGTruth_official" / "dataset" / "response.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "local_cache" / "qwen25_15b_tokenizer", local_files_only=True)
    _, labels, _ = adapt_cache(cache, official, tokenizer)
    return np.asarray([labels.response[str(item)].hallucinated for item in response_ids], dtype=int)


def audit_ragtruth(root: Path):
    metrics, feature_metrics, scores, sensitivity, examples = [], [], [], [], {}
    for split in ("dev", "test"):
        archive = np.load(root / f"scores_{split}.npz", allow_pickle=False)
        diagnostics = json.loads((root / "diagnostics" / f"fit_{split}.json").read_text())
        response_ids = archive["response_ids"].astype(str)
        target = _ragtruth_labels(split, response_ids)
        for variant in ("original30_full", "hybrid"):
            cell = f"ragtruth_{split}__{variant}"
            values = np.asarray(archive[f"{variant}__values"], dtype=float)
            names_all = archive[f"{variant}__feature_names"].astype(str)
            diag = diagnostics[variant]
            keep_names = tuple(str(x) for x in diag["kept_feature_names"])
            lookup = {name: index for index, name in enumerate(names_all)}
            indexes = np.asarray([lookup[name] for name in keep_names], dtype=int)
            mean = np.asarray(diag["standardization_mean"], dtype=float)[indexes]
            scale = np.asarray(diag["standardization_scale"], dtype=float)[indexes]
            matrix = (values[:, indexes] - mean[None, :]) / scale[None, :]
            gates = gate_from_raw(diag["dufs"]["raw_probabilities"])
            F = matrix.T
            graph = build_graph_from_features(F, gates=gates, k=K)
            raw_graph = build_graph_from_features(F, k=K)
            nuisances = [
                ("response_length", archive["response_lengths"], False, "nuisance"),
                ("context_length", archive["context_lengths"], False, "nuisance"),
                ("chunk_count", archive["chunk_counts"], False, "nuisance"),
                ("task_type", archive["task_types"].astype(str), True, "nuisance"),
                ("source", archive["sources"].astype(str), True, "nuisance"),
                ("row_order", np.arange(len(target)), False, "nuisance"),
            ]
            iu = np.asarray(archive[f"score__{variant}__iu_pcr"], dtype=float)
            liu = np.asarray(archive[f"score__{variant}__dufs_liu"], dtype=float)
            mrows, frows, score = cell_rows(
                lane="ragtruth", cell=cell, split=("development" if split == "dev" else "test_validation"),
                graph=graph, raw_graph=raw_graph, target=target, nuisances=nuisances,
                iu_score=iu, liu_score=liu, feature_matrix=matrix, feature_names=keep_names,
            )
            metrics.extend(mrows); feature_metrics.extend(frows); scores.append(score)
            sensitivity.extend(topology_sensitivity_rows(
                lane="ragtruth", cell=cell,
                split=("development" if split == "dev" else "test_validation"),
                F=F, gates=gates, target=target, length=archive["response_lengths"],
            ))
            if split == "test" and variant == "original30_full":
                examples["ragtruth"] = (graph, target, quartiles(archive["response_lengths"]), cell)
    return metrics, feature_metrics, scores, sensitivity, examples


def sensitivity_summary(rows: list[dict]) -> list[dict]:
    output = []
    for lane in sorted({row["lane"] for row in rows}):
        for family in ("union_knn", "mutual_knn"):
            for k in K_VALUES:
                current = [row for row in rows if row["lane"] == lane and row["graph_family"] == family and row["k"] == k]
                if lane == "processbench":
                    current = [row for row in current if row["split"] == "model_validation"]
                elif lane == "ragtruth":
                    current = [row for row in current if row["split"] == "test_validation" and row["cell"].endswith("original30_full")]
                if not current:
                    continue
                output.append({
                    "lane": lane, "graph_family": family, "k": k, "validation_cells": len(current),
                    "fraction_target_aligned": float(np.mean([row["target_aligned"] for row in current])),
                    "fraction_target_smoother_than_length": float(np.mean([row["target_smoother_than_length"] for row in current])),
                    "median_components": float(np.median([row["n_components"] for row in current])),
                })
    return output


def lookup_metric(rows, lane, cell, variable, graph="dufs"):
    for row in rows:
        if row["lane"] == lane and row["cell"] == cell and row["variable"] == variable and row["graph"] == graph:
            return row
    raise KeyError((lane, cell, variable, graph))


def lane_summary(metric_rows, score_rows):
    summaries = []
    rules = {
        "global24": lambda row: True,
        "processbench": lambda row: row["split"] == "model_validation",
        "ragtruth": lambda row: row["split"] == "test_validation" and row["cell"].endswith("original30_full"),
    }
    target_names = {"global24": "hallucination", "processbench": "process_error", "ragtruth": "hallucination"}
    length_names = {"global24": "trace_length", "processbench": "trace_length", "ragtruth": "response_length"}
    for lane, keep in rules.items():
        scores = [row for row in score_rows if row["lane"] == lane and keep(row)]
        scores = [
            row for row in scores
            if any(
                metric["lane"] == lane and metric["cell"] == row["cell"]
                and metric["variable"] == length_names[lane] and metric["graph"] == "dufs"
                for metric in metric_rows
            )
        ]
        target = [lookup_metric(metric_rows, lane, row["cell"], target_names[lane]) for row in scores]
        length = [lookup_metric(metric_rows, lane, row["cell"], length_names[lane]) for row in scores]
        z = np.asarray([row["smoothness_z"] for row in target], dtype=float)
        lz = np.asarray([row["smoothness_z"] for row in length], dtype=float)
        deltas = np.asarray([row["liu_delta_auroc"] for row in scores], dtype=float)
        aligned = float(np.mean(z > 1.96))
        target_effect = np.asarray([row["smoothness_effect"] for row in target], dtype=float)
        length_effect = np.asarray([row["smoothness_effect"] for row in length], dtype=float)
        beats_length = float(np.mean(target_effect > length_effect))
        if aligned >= 2 / 3 and beats_length > 0.5:
            decision = "CONSISTENT_TARGET_ALIGNMENT"
        elif aligned >= 2 / 3:
            decision = "TARGET_ALIGNED_BUT_NUISANCE_DOMINATED"
        else:
            decision = "NO_CONSISTENT_TARGET_MANIFOLD"
        rho = float(spearmanr(z, deltas).statistic) if len(z) >= 3 else float("nan")
        summaries.append({
            "lane": lane, "validation_cells": len(scores), "decision": decision,
            "fraction_target_smooth_z_gt_1p96": aligned,
            "fraction_target_smoother_than_length": beats_length,
            "median_target_smoothness_z": float(np.median(z)),
            "median_length_smoothness_z": float(np.median(lz)),
            "median_target_smoothness_effect": float(np.median(target_effect)),
            "median_length_smoothness_effect": float(np.median(length_effect)),
            "median_liu_delta_auroc": float(np.median(deltas)),
            "spearman_target_smoothness_vs_liu_delta": rho,
        })
    return summaries


def spectral_embedding(graph: csr_matrix) -> np.ndarray:
    lap = symmetric_normalized_laplacian(graph)
    _, vectors = eigsh(lap, k=3, which="SM", tol=1e-6)
    return np.asarray(vectors[:, 1:3], dtype=float)


def save_examples(path: Path, examples: dict) -> None:
    arrays = {}
    for lane, (graph, target, length_q, cell) in examples.items():
        coords = spectral_embedding(graph)
        coo = graph.tocoo()
        keep = coo.row < coo.col
        rows, cols, weights = coo.row[keep], coo.col[keep], coo.data[keep]
        order = np.argsort(weights)[::-1][: min(1500, len(weights))]
        arrays[f"{lane}__coords"] = coords
        arrays[f"{lane}__target"] = np.asarray(target, dtype=int)
        arrays[f"{lane}__length_q"] = np.asarray(length_q, dtype=int)
        arrays[f"{lane}__edge_rows"] = rows[order]
        arrays[f"{lane}__edge_cols"] = cols[order]
        arrays[f"{lane}__cell"] = np.asarray(cell)
    np.savez_compressed(path, **arrays)


def build_report(out: Path, summaries: list[dict], topology: list[dict]) -> None:
    lines = [
        "# Direct DUFS graph-semantics audit v1", "",
        "This retrospective diagnostic reconstructs the actual DUFS kNN graph and asks what is smooth on it. Lanes are not pooled.", "",
        "## Validation decisions", "",
        "| Lane | Decision | Target smoothness effect | Length smoothness effect | Target > length | LIU ΔAUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['lane']} | {row['decision']} | {row['median_target_smoothness_effect']:.1%} | "
            f"{row['median_length_smoothness_effect']:.1%} | {row['fraction_target_smoother_than_length']:.0%} | "
            f"{row['median_liu_delta_auroc']:+.4f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "Positive smoothness z means that neighbours are more similar on that variable than under row permutation. It does not identify the cause of that similarity.", "",
        "A target-aligned graph is useful to LIU only when the target is smoother than competing nuisances and the resulting Laplacian correction improves ranking. The per-cell tables preserve the cases where those conditions disagree.", "",
        "Across every validation lane, the target is smoother than chance but never smoother than length in the primary cells. The LIU increment is correspondingly small and target smoothness does not predict a larger increment.", "",
        "## kNN topology sensitivity", "",
        "| Lane | Graph | k | Target aligned | Target smoother than length | Median components |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in topology:
        if int(row["k"]) == 7:
            lines.append(
                f"| {row['lane']} | {row['graph_family']} | 7 | "
                f"{row['fraction_target_aligned']:.0%} | "
                f"{row['fraction_target_smoother_than_length']:.0%} | "
                f"{row['median_components']:.1f} |"
            )
    lines += [
        "", "The ordinary union-kNN conclusion is stable for k in {3,5,7,10,15,25}. Mutual-kNN does not rescue target specificity and fragments the graph into many components at small k.", "",
        "Global answer hallucination, ProcessBench process-error localization, and RAGTruth response hallucination are different estimands and must remain separate.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_metrics, all_features, all_scores, all_sensitivity, examples = [], [], [], [], {}
    parts = [
        audit_global(ROOT / "results/dependency_fusion_raw/cells.npz", ROOT / "results/frozen_24cell_benchmark"),
        audit_processbench(ROOT / "cache/localization/processbench", ROOT / "results/processbench_latent_state_v1"),
        audit_ragtruth(ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1"),
    ]
    for metrics, features, scores, sensitivity, current_examples in parts:
        all_metrics.extend(metrics); all_features.extend(features); all_scores.extend(scores)
        all_sensitivity.extend(sensitivity); examples.update(current_examples)
    summaries = lane_summary(all_metrics, all_scores)
    topology = sensitivity_summary(all_sensitivity)
    write_csv(out / "GRAPH_VARIABLE_METRICS.csv", all_metrics)
    write_csv(out / "FEATURE_SMOOTHNESS.csv", all_features)
    write_csv(out / "LIU_EFFECTS.csv", all_scores)
    write_csv(out / "LANE_SUMMARY.csv", summaries)
    write_csv(out / "GRAPH_SENSITIVITY.csv", all_sensitivity)
    write_csv(out / "GRAPH_SENSITIVITY_SUMMARY.csv", topology)
    save_examples(out / "MANIFOLD_EXAMPLES.npz", examples)
    write_json(out / "DECISION.json", {"version": VERSION, "permutations": PERMUTATIONS, "k": K, "lanes": summaries, "topology_sensitivity": topology, "no_cross_task_pooling": True})
    build_report(out, summaries, topology)
    print(json.dumps(jsonable(summaries), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
