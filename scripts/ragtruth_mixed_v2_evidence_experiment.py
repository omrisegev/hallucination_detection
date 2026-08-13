#!/usr/bin/env python3
"""Run the exploratory RAGTruth original-30 evidence-aware comparison.

`score` never passes labels to feature construction or fitting.  It writes and
hashes a score bundle.  `evaluate` verifies that hash and is the only command
that reads the isolated RAGTruth label object.  `report` renders only saved
CSV/JSON/NPZ artifacts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_fit,
    laplacian_iu_path,
    permute_graph,
)
from spectral_utils.ragtruth_evidence_contrast import (  # noqa: E402
    adapt_cache,
    build_feature_tables,
    sha256_file,
    standardize_features,
)
from spectral_utils.ragtruth_mixed_v2_evidence import (  # noqa: E402
    CONTRACT_VERSION,
    ORIGINAL_FEATURES,
    VariantMatrix,
    build_mixed_v2_evidence_tensor,
    build_variant_matrices,
    condition_matrices,
    feature_availability,
    flatten_loo,
    permute_evidence_blocks,
)


VERSION = "ragtruth-original30-evidence-aware-v1-exploratory-2026-08-10"
DEFAULT_OUT = REPO / "results" / "ragtruth_mixed_v2_evidence_aware_v1"
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
GRAPH_K = 7
LAMBDA = 0.1
LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
BOOTSTRAP_SEED = 20260810
GRAPH_PERMUTATION_SEED = 20260811
CONDITION_PERMUTATION_SEED = 20260812


def _jsonable(value: Any) -> Any:
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reference_scores(path: Path, tensor: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists():
        hash_path = path.parent / (path.stem + ".sha256")
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError("reference Evidence-Contrast score hash changed")
    data = np.load(path, allow_pickle=False)
    prefix = "full_response__"
    reference_ids = data[prefix + "sample_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(reference_ids)}
    if set(lookup) != set(tensor.response_ids):
        raise ValueError("reference EC scores and original-30 LOO cohort disagree")
    order = np.asarray([lookup[value] for value in tensor.response_ids], dtype=int)
    scores = {
        name: np.asarray(data[prefix + "score__" + name], dtype=float)[order]
        for name in ("gasp_top50", "ec_iu_pcr", "ec_dufs_liu")
    }
    return scores, {
        "path": str(path.resolve()),
        "sha256": actual,
        "table": "full_response",
        "methods": tuple(scores),
    }


def _fit_variant(
    variant: VariantMatrix,
    task_types: np.ndarray,
    *,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    standardized, keep, mean, scale = standardize_features(variant.values)
    names = tuple(name for name, selected in zip(variant.feature_names, keep) if selected)
    blocks = tuple(name for name, selected in zip(variant.block_names, keep) if selected)
    bases = tuple(name for name, selected in zip(variant.base_features, keep) if selected)
    F = standardized.T
    started = time.time()
    gates, gate_diag = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS, return_history=True
    )
    graph = build_graph_from_features(F, gates=gates, k=GRAPH_K)
    path = laplacian_iu_path(F, LAMBDAS, graph=graph)
    if not np.array_equal(path[0.0].w, path[0.0].baseline.w):
        raise RuntimeError(f"{variant.name}: lambda=0 is not exactly IU-PCR")
    methods = {
        f"{variant.name}__iu_pcr": -(path[0.0].w @ F),
        f"{variant.name}__dufs_liu": -(path[LAMBDA].w @ F),
    }

    graph_rng = np.random.default_rng(
        GRAPH_PERMUTATION_SEED + (0 if split == "dev" else 1000)
    )
    graph_control = laplacian_iu_fit(
        F,
        lambda_=LAMBDA,
        graph=permute_graph(graph, graph_rng.permutation(F.shape[1])),
    )
    methods[f"{variant.name}__graph_permuted"] = -(graph_control.w @ F)

    condition_diag: dict[str, Any] | None = None
    if any(variant.permutable):
        permuted_values = permute_evidence_blocks(
            variant,
            task_types,
            seed=CONDITION_PERMUTATION_SEED + (0 if split == "dev" else 1000),
        )
        # Block permutation preserves column marginals, so the original
        # standardization is deliberately reused.
        permuted_standardized = (
            permuted_values[:, keep] - mean[keep][None, :]
        ) / scale[keep][None, :]
        F_permuted = permuted_standardized.T
        condition_gates, condition_gate_diag = dufs_soft_gates(
            F_permuted,
            seeds=DUFS_SEEDS,
            epochs=DUFS_EPOCHS,
            return_history=True,
        )
        condition_graph = build_graph_from_features(
            F_permuted, gates=condition_gates, k=GRAPH_K
        )
        condition_fit = laplacian_iu_fit(
            F_permuted, lambda_=LAMBDA, graph=condition_graph
        )
        # Score the permuted feature rows in their permuted order. This is a
        # negative control, not a deployable score for the original response.
        methods[f"{variant.name}__condition_permuted"] = -(
            condition_fit.w @ F_permuted
        )
        condition_diag = {
            "dufs": condition_gate_diag,
            "laplacian": condition_fit.diagnostics,
        }

    history_rows: list[dict[str, Any]] = []
    history = np.asarray(gate_diag.pop("training_history"), dtype=float)
    for seed_index, seed in enumerate(DUFS_SEEDS):
        for epoch, loss in enumerate(history[seed_index], start=1):
            history_rows.append({
                "split": split,
                "variant": variant.name,
                "control": "observed",
                "seed": seed,
                "epoch": epoch,
                "loss": float(loss),
            })
    if condition_diag is not None:
        condition_history = np.asarray(
            condition_diag["dufs"].pop("training_history"), dtype=float
        )
        for seed_index, seed in enumerate(DUFS_SEEDS):
            for epoch, loss in enumerate(condition_history[seed_index], start=1):
                history_rows.append({
                    "split": split,
                    "variant": variant.name,
                    "control": "condition_permuted",
                    "seed": seed,
                    "epoch": epoch,
                    "loss": float(loss),
                })

    weight_rows = []
    raw_probabilities = np.asarray(gate_diag["raw_probabilities"], dtype=float)
    per_seed = np.asarray(gate_diag["per_seed_probabilities"], dtype=float)
    for index, (name, block, base) in enumerate(zip(names, blocks, bases)):
        weight_rows.append({
            "split": split,
            "variant": variant.name,
            "column": name,
            "block": block,
            "base_feature": base,
            "dufs_gate": float(raw_probabilities[index]),
            "dufs_gate_seed_std": float(per_seed[:, index].std()),
            "iu_weight": float(path[0.0].w[index]),
            "dufs_liu_weight": float(path[LAMBDA].w[index]),
        })

    diagnostics = {
        "contract": CONTRACT_VERSION,
        "input_columns": len(variant.feature_names),
        "kept_columns": len(names),
        "constant_columns": [
            name for name, selected in zip(variant.feature_names, keep) if not selected
        ],
        "kept_feature_names": names,
        "kept_blocks": blocks,
        "kept_base_features": bases,
        "standardization_mean": mean,
        "standardization_scale": scale,
        "dufs": gate_diag,
        "runtime_seconds": time.time() - started,
        "lambda_zero_exact": True,
        "lambda_path": {
            str(lambda_): {
                "weights": fit.w,
                "diagnostics": fit.diagnostics,
            }
            for lambda_, fit in path.items()
        },
        "graph_permuted": graph_control.diagnostics,
        "condition_permuted": condition_diag,
    }
    for name, score in methods.items():
        score = np.asarray(score, dtype=float)
        if score.shape != (len(variant.values),) or not np.isfinite(score).all():
            raise RuntimeError(f"invalid score: {name}")
        methods[name] = score
    return methods, diagnostics, history_rows, weight_rows


def _condition_gate_rows(tensor: Any, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = np.asarray(tensor.task_types).astype(str)
    for condition, matrix in condition_matrices(tensor).items():
        for task in ("ALL", *sorted(set(tasks))):
            mask = np.ones(len(tasks), dtype=bool) if task == "ALL" else tasks == task
            F = np.asarray(matrix[mask], dtype=float).T
            gates, diagnostics = dufs_soft_gates(
                F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            raw = np.asarray(diagnostics["raw_probabilities"], dtype=float)
            per_seed = np.asarray(diagnostics["per_seed_probabilities"], dtype=float)
            for index, feature in enumerate(tensor.feature_names):
                rows.append({
                    "split": split,
                    "condition": condition,
                    "task": task,
                    "feature": feature,
                    "gate": float(raw[index]),
                    "gate_seed_std": float(per_seed[:, index].std()),
                    "effective_feature_count": diagnostics["effective_feature_count"],
                    "mean_seed_std_all_features": diagnostics["mean_seed_std"],
                    "n": int(mask.sum()),
                })
    return rows


def score_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, isolated_labels, adapter_audit = adapt_cache(
        Path(args.cache), Path(args.official_responses), tokenizer
    )
    del isolated_labels
    availability = feature_availability(dataset)
    _write_csv(out / f"feature_availability_{args.split}.csv", availability)
    unavailable = [row for row in availability if not row["fully_available"]]
    if unavailable:
        raise RuntimeError(
            f"{len(unavailable)} feature/condition entries are incomplete; see audit CSV"
        )

    tensor = build_mixed_v2_evidence_tensor(dataset)
    ec_table = build_feature_tables(dataset)["full_response"]
    variants = build_variant_matrices(tensor, ec_table)
    references, reference_manifest = _reference_scores(
        Path(args.reference_scores), tensor
    )
    arrays: dict[str, np.ndarray] = {
        "response_ids": np.asarray(tensor.response_ids),
        "source_ids": np.asarray(tensor.source_ids),
        "task_types": np.asarray(tensor.task_types),
        "sources": np.asarray(tensor.sources),
        "response_lengths": tensor.response_lengths,
        "context_lengths": tensor.context_lengths,
        "chunk_counts": tensor.chunk_counts,
        "original_feature_names": np.asarray(tensor.feature_names),
        "raw_full": tensor.raw_full,
        "raw_noctx": tensor.raw_noctx,
        "mixed_full": tensor.mixed_full,
        "mixed_noctx": tensor.mixed_noctx,
    }
    raw_loo, loo_offsets, loo_indexes = flatten_loo(
        tensor.raw_loo, tensor.loo_indexes
    )
    mixed_loo, mixed_offsets, mixed_indexes = flatten_loo(
        tensor.mixed_loo, tensor.loo_indexes
    )
    if not np.array_equal(loo_offsets, mixed_offsets) or not np.array_equal(
        loo_indexes, mixed_indexes
    ):
        raise RuntimeError("raw and transformed LOO layouts disagree")
    arrays.update({
        "raw_loo": raw_loo,
        "mixed_loo": mixed_loo,
        "loo_offsets": loo_offsets,
        "loo_indexes": loo_indexes,
    })
    arrays.update({"score__" + name: score for name, score in references.items()})

    diagnostics: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    tasks = np.asarray(tensor.task_types)
    for name, variant in variants.items():
        print(
            f"[score] {args.split}/{name}: n={len(variant.values)} "
            f"columns={len(variant.feature_names)}",
            flush=True,
        )
        methods, fit_diag, history, weights = _fit_variant(
            variant, tasks, split=args.split
        )
        arrays.update({"score__" + method: score for method, score in methods.items()})
        arrays[name + "__values"] = variant.values
        arrays[name + "__feature_names"] = np.asarray(variant.feature_names)
        arrays[name + "__block_names"] = np.asarray(variant.block_names)
        arrays[name + "__base_features"] = np.asarray(variant.base_features)
        diagnostics[name] = fit_diag
        history_rows.extend(history)
        weight_rows.extend(weights)

    print(f"[gates] {args.split}: full/noctx/LOO condition profiles", flush=True)
    condition_gate_rows = _condition_gate_rows(tensor, args.split)
    score_path = out / f"scores_{args.split}.npz"
    np.savez_compressed(score_path, **arrays)
    digest = sha256_file(score_path)
    (out / f"scores_{args.split}.sha256").write_text(
        f"{digest}  {score_path.name}\n", encoding="utf-8"
    )
    _write_json(out / "diagnostics" / f"fit_{args.split}.json", diagnostics)
    _write_csv(out / f"training_history_{args.split}.csv", history_rows)
    _write_csv(out / f"fusion_weights_{args.split}.csv", weight_rows)
    _write_csv(out / f"condition_gates_{args.split}.csv", condition_gate_rows)
    _write_json(out / f"score_manifest_{args.split}.json", {
        "experiment": VERSION,
        "split": args.split,
        "exploratory": True,
        "reason": "RAGTruth labels were opened by the earlier EC experiment",
        "label_free_scoring": True,
        "labels_passed_to_fitting": False,
        "cache_sha256": dataset.cache_sha256,
        "adapter_audit": adapter_audit,
        "reference_scores": reference_manifest,
        "score_sha256": digest,
        "n_responses": len(tensor.response_ids),
        "n_original_features": len(tensor.feature_names),
        "all_original_features_available": True,
        "exact_full_mixed_v2_transform_error": tensor.exact_full_contract_error,
        "variants": {
            name: {
                "nominal_columns": len(variant.feature_names),
                "blocks": sorted(set(variant.block_names)),
            }
            for name, variant in variants.items()
        },
        "settings": {
            "dufs_seeds": DUFS_SEEDS,
            "dufs_epochs": DUFS_EPOCHS,
            "graph_k": GRAPH_K,
            "lambda": LAMBDA,
            "lambda_path": LAMBDAS,
        },
        "python": platform.python_version(),
    })
    print(f"[frozen exploratory scores] {score_path} sha256={digest}", flush=True)


def _bootstrap_indices(groups: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    return [
        np.concatenate([
            members[group] for group in rng.choice(unique, size=len(unique), replace=True)
        ])
        for _ in range(int(count))
    ]


def _metric_bundle(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    groups: np.ndarray,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    if len(np.unique(labels)) < 2:
        return {}, {}
    samples = _bootstrap_indices(groups, bootstrap, seed)
    result: dict[str, dict[str, float]] = {}
    distributions: dict[str, np.ndarray] = {}
    valid_samples = [index for index in samples if len(np.unique(labels[index])) == 2]
    for method, score in scores.items():
        auc = np.asarray([
            roc_auc_score(labels[index], score[index]) for index in valid_samples
        ])
        pr = np.asarray([
            average_precision_score(labels[index], score[index])
            for index in valid_samples
        ])
        result[method] = {
            "auroc": float(roc_auc_score(labels, score)),
            "auroc_ci_low": float(np.quantile(auc, 0.025)),
            "auroc_ci_high": float(np.quantile(auc, 0.975)),
            "auprc": float(average_precision_score(labels, score)),
            "auprc_ci_low": float(np.quantile(pr, 0.025)),
            "auprc_ci_high": float(np.quantile(pr, 0.975)),
        }
        distributions[method] = auc
    return result, distributions


def _residualize(score: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(score)), confounds])
    keep = np.std(design, axis=0) > 1e-12
    keep[0] = True
    fitted = design[:, keep] @ np.linalg.lstsq(
        design[:, keep], score, rcond=None
    )[0]
    return score - fitted


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _comparison_pairs(methods: set[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for variant in (
        "original30_full", "original30_noctx", "original30_loo", "hybrid"
    ):
        iu = f"{variant}__iu_pcr"
        dufs = f"{variant}__dufs_liu"
        if iu in methods and dufs in methods:
            pairs.append((dufs, iu, "DUFS contribution beyond IU-PCR"))
        for method in (iu, dufs):
            if method in methods and "gasp_top50" in methods:
                pairs.append((method, "gasp_top50", "comparison with GASP-top50"))
        if variant != "original30_full":
            for solver in ("iu_pcr", "dufs_liu"):
                candidate = f"{variant}__{solver}"
                reference = f"original30_full__{solver}"
                if candidate in methods and reference in methods:
                    pairs.append((candidate, reference, "evidence gain over full-only"))
        graph = f"{variant}__graph_permuted"
        condition = f"{variant}__condition_permuted"
        if dufs in methods and graph in methods:
            pairs.append((dufs, graph, "observed graph versus sample-permuted graph"))
        if dufs in methods and condition in methods:
            pairs.append((dufs, condition, "observed versus condition-block permutation"))
    for method in ("ec_iu_pcr", "ec_dufs_liu"):
        if method in methods and "gasp_top50" in methods:
            pairs.append((method, "gasp_top50", "existing EC reference"))
    return pairs


def evaluate_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    score_path = out / f"scores_{args.split}.npz"
    expected = (out / f"scores_{args.split}.sha256").read_text().split()[0]
    actual = sha256_file(score_path)
    if actual != expected:
        raise RuntimeError("score hash changed before evaluation")
    scores_npz = np.load(score_path, allow_pickle=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, labels, _ = adapt_cache(
        Path(args.cache), Path(args.official_responses), tokenizer
    )
    del dataset
    response_ids = scores_npz["response_ids"].astype(str)
    y = np.asarray([labels.response[value].hallucinated for value in response_ids])
    groups = scores_npz["source_ids"].astype(str)
    tasks = scores_npz["task_types"].astype(str)
    methods = {
        key[len("score__"):]: np.asarray(scores_npz[key], dtype=float)
        for key in scores_npz.files if key.startswith("score__")
    }

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    distribution_by_task: dict[str, dict[str, np.ndarray]] = {}
    metrics_by_task: dict[str, dict[str, dict[str, float]]] = {}
    for task_index, task in enumerate(("ALL", *sorted(set(tasks)))):
        mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
        metrics, distributions = _metric_bundle(
            y[mask],
            {name: score[mask] for name, score in methods.items()},
            groups[mask],
            int(args.bootstrap),
            BOOTSTRAP_SEED + task_index,
        )
        distribution_by_task[task] = distributions
        metrics_by_task[task] = metrics
        for method, values in metrics.items():
            summary_rows.append({
                "split": args.split,
                "task": task,
                "method": method,
                "n": int(mask.sum()),
                "positives": int(y[mask].sum()),
                "n_source_groups": int(len(np.unique(groups[mask]))),
                **values,
            })
        for challenger, reference, question in _comparison_pairs(set(metrics)):
            delta = distributions[challenger] - distributions[reference]
            paired_rows.append({
                "split": args.split,
                "task": task,
                "challenger": challenger,
                "reference": reference,
                "question": question,
                "delta_auroc": metrics[challenger]["auroc"] - metrics[reference]["auroc"],
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "probability_positive": float(np.mean(delta > 0.0)),
            })

    task_names = sorted(set(tasks))
    if len(task_names) > 1:
        macro_distributions = {
            method: np.mean(
                np.vstack([distribution_by_task[task][method] for task in task_names]),
                axis=0,
            )
            for method in methods
        }
        macro_metrics: dict[str, dict[str, float]] = {}
        for method in methods:
            auc_values = [metrics_by_task[task][method]["auroc"] for task in task_names]
            pr_values = [metrics_by_task[task][method]["auprc"] for task in task_names]
            distribution = macro_distributions[method]
            macro_metrics[method] = {
                "auroc": float(np.mean(auc_values)),
                "auroc_ci_low": float(np.quantile(distribution, 0.025)),
                "auroc_ci_high": float(np.quantile(distribution, 0.975)),
                "auprc": float(np.mean(pr_values)),
                "auprc_ci_low": float("nan"),
                "auprc_ci_high": float("nan"),
            }
            summary_rows.append({
                "split": args.split,
                "task": "MACRO_TASK",
                "method": method,
                "n": len(y),
                "positives": int(y.sum()),
                "n_source_groups": int(len(np.unique(groups))),
                **macro_metrics[method],
            })
        for challenger, reference, question in _comparison_pairs(set(methods)):
            delta = macro_distributions[challenger] - macro_distributions[reference]
            paired_rows.append({
                "split": args.split,
                "task": "MACRO_TASK",
                "challenger": challenger,
                "reference": reference,
                "question": question,
                "delta_auroc": (
                    macro_metrics[challenger]["auroc"] - macro_metrics[reference]["auroc"]
                ),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "probability_positive": float(np.mean(delta > 0.0)),
            })

        task_standardized = {}
        for method, score in methods.items():
            adjusted = np.empty_like(score)
            for task in task_names:
                mask = tasks == task
                scale = float(np.std(score[mask]))
                adjusted[mask] = (
                    score[mask] - float(np.mean(score[mask]))
                ) / (scale if scale >= 1e-12 else 1.0)
            task_standardized[method] = adjusted
        standardized_metrics, standardized_distributions = _metric_bundle(
            y,
            task_standardized,
            groups,
            int(args.bootstrap),
            BOOTSTRAP_SEED + 100,
        )
        for method, values in standardized_metrics.items():
            summary_rows.append({
                "split": args.split,
                "task": "TASK_STANDARDIZED_POOL",
                "method": method,
                "n": len(y),
                "positives": int(y.sum()),
                "n_source_groups": int(len(np.unique(groups))),
                **values,
            })
        for challenger, reference, question in _comparison_pairs(set(methods)):
            delta = (
                standardized_distributions[challenger]
                - standardized_distributions[reference]
            )
            paired_rows.append({
                "split": args.split,
                "task": "TASK_STANDARDIZED_POOL",
                "challenger": challenger,
                "reference": reference,
                "question": question,
                "delta_auroc": (
                    standardized_metrics[challenger]["auroc"]
                    - standardized_metrics[reference]["auroc"]
                ),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "probability_positive": float(np.mean(delta > 0.0)),
            })

    confounds = np.column_stack([
        scores_npz["response_lengths"],
        scores_npz["context_lengths"],
        scores_npz["chunk_counts"],
    ]).astype(float)
    confound_rows = []
    for task in ("ALL", *sorted(set(tasks))):
        mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
        for method, score in methods.items():
            residual = _residualize(score[mask], confounds[mask])
            correlations = [
                _safe_spearman(score[mask], confounds[mask, index])
                for index in range(3)
            ]
            confound_rows.append({
                "split": args.split,
                "task": task,
                "method": method,
                "rho_response_length": correlations[0],
                "rho_context_length": correlations[1],
                "rho_chunk_count": correlations[2],
                "raw_auroc": roc_auc_score(y[mask], score[mask]),
                "residualized_auroc": roc_auc_score(y[mask], residual),
            })

    feature_rows: list[dict[str, Any]] = []
    for variant in (
        "original30_full", "original30_noctx", "original30_loo", "hybrid"
    ):
        values = np.asarray(scores_npz[variant + "__values"], dtype=float)
        names = scores_npz[variant + "__feature_names"].astype(str)
        blocks = scores_npz[variant + "__block_names"].astype(str)
        bases = scores_npz[variant + "__base_features"].astype(str)
        for task in ("ALL", *sorted(set(tasks))):
            mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
            if len(np.unique(y[mask])) < 2:
                continue
            for index, name in enumerate(names):
                column = values[mask, index]
                if np.std(column) < 1e-12:
                    auc = ""
                else:
                    auc = float(roc_auc_score(y[mask], -column))
                feature_rows.append({
                    "split": args.split,
                    "variant": variant,
                    "task": task,
                    "column": name,
                    "block": blocks[index],
                    "base_feature": bases[index],
                    "n": int(mask.sum()),
                    "hallucination_auroc_from_negative_column": auc,
                })

    chunk_rows: list[dict[str, Any]] = []
    mixed_full = np.asarray(scores_npz["mixed_full"], dtype=float)
    mixed_loo = np.asarray(scores_npz["mixed_loo"], dtype=float)
    offsets = np.asarray(scores_npz["loo_offsets"], dtype=int)
    loo_indexes = np.asarray(scores_npz["loo_indexes"], dtype=int)
    for chunk_index in sorted(set(loo_indexes.tolist())):
        response_positions, row_positions = [], []
        for response_index in range(len(response_ids)):
            start, stop = offsets[response_index], offsets[response_index + 1]
            local = np.flatnonzero(loo_indexes[start:stop] == chunk_index)
            if len(local):
                response_positions.append(response_index)
                row_positions.append(start + int(local[0]))
        response_positions = np.asarray(response_positions, dtype=int)
        row_positions = np.asarray(row_positions, dtype=int)
        drops = mixed_full[response_positions] - mixed_loo[row_positions]
        for task in ("ALL", *sorted(set(tasks))):
            mask = (
                np.ones(len(response_positions), dtype=bool)
                if task == "ALL" else tasks[response_positions] == task
            )
            labels_here = y[response_positions][mask]
            if len(labels_here) < 20 or len(np.unique(labels_here)) < 2:
                continue
            for feature_index, feature in enumerate(ORIGINAL_FEATURES):
                column = drops[mask, feature_index]
                chunk_rows.append({
                    "split": args.split,
                    "chunk_index": chunk_index,
                    "task": task,
                    "feature": feature,
                    "n": int(mask.sum()),
                    "hallucination_auroc_from_negative_drop": (
                        float(roc_auc_score(labels_here, -column))
                        if np.std(column) >= 1e-12 else ""
                    ),
                    "mean_drop": float(np.mean(column)),
                    "std_drop": float(np.std(column)),
                })

    _write_csv(out / f"summary_{args.split}.csv", summary_rows)
    _write_csv(out / f"paired_{args.split}.csv", paired_rows)
    _write_csv(out / f"confounds_{args.split}.csv", confound_rows)
    _write_csv(out / f"feature_task_diagnostics_{args.split}.csv", feature_rows)
    _write_csv(out / f"chunk_feature_diagnostics_{args.split}.csv", chunk_rows)
    _write_json(out / f"evaluation_manifest_{args.split}.json", {
        "experiment": VERSION,
        "split": args.split,
        "score_sha256_verified": actual,
        "exploratory": True,
        "labels_opened_before_experiment": True,
        "labels_used_only_by_evaluate_command": True,
        "bootstrap_samples": int(args.bootstrap),
        "bootstrap_group": "source_id",
        "same_bootstrap_draws_for_every_method_within_each_task": True,
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    if args.split == "test":
        _write_json(out / "exploratory_conclusion.json", _conclusion(
            summary_rows, paired_rows
        ))
    print(f"[evaluated] {args.split}: verified {actual}", flush=True)


def reproduction_audit(args: argparse.Namespace) -> None:
    """Compare the new full-only DUFS score with the prior post-hoc artifact."""
    out = Path(args.out)
    current_path = out / f"scores_{args.split}.npz"
    expected = (out / f"scores_{args.split}.sha256").read_text().split()[0]
    if sha256_file(current_path) != expected:
        raise RuntimeError("current score hash changed")
    current = np.load(current_path, allow_pickle=False)
    old_path = Path(args.old_scores)
    old = np.load(old_path, allow_pickle=False)
    current_ids = current["response_ids"].astype(str)
    old_ids = old["full_response__response_ids"].astype(str)
    lookup = {value: index for index, value in enumerate(old_ids)}
    if set(current_ids) != set(old_ids):
        raise ValueError("old and current full-response cohorts disagree")
    old_score = np.asarray(
        old["full_response__score__intrinsic_mixed_v2_dufs_liu"], dtype=float
    )[[lookup[value] for value in current_ids]]
    current_score = np.asarray(
        current["score__original30_full__dufs_liu"], dtype=float
    )
    difference = current_score - old_score
    audit = {
        "split": args.split,
        "current_score_sha256": expected,
        "old_score_path": str(old_path.resolve()),
        "old_score_sha256": sha256_file(old_path),
        "n": len(current_ids),
        "max_absolute_score_difference": float(np.max(np.abs(difference))),
        "mean_absolute_score_difference": float(np.mean(np.abs(difference))),
        "pearson_correlation": float(np.corrcoef(current_score, old_score)[0, 1]),
        "rank_order_exact": bool(np.array_equal(
            np.argsort(current_score), np.argsort(old_score)
        )),
        "interpretation": (
            "The original30_full DUFS-LIU arm reproduces the previous mixed-v2 "
            "full-context score; evidence-aware arms add condition information on top."
        ),
    }
    _write_json(out / f"full_only_reproduction_{args.split}.json", audit)
    print(
        f"[reproduction] {args.split}: max_abs="
        f"{audit['max_absolute_score_difference']:.3g}",
        flush=True,
    )


def _row_lookup(
    rows: list[dict[str, Any]],
    *,
    task: str,
    challenger: str,
    reference: str,
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["task"] == task and row["challenger"] == challenger
        and row["reference"] == reference
    )


def _conclusion(
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = {
        row["method"]: row for row in summary_rows if row["task"] == "ALL"
    }
    evidence_pairs = [
        row for row in paired_rows
        if row["task"] == "ALL" and row["question"] == "evidence gain over full-only"
        and row["challenger"].startswith("original30_")
    ]
    dufs_pairs = [
        row for row in paired_rows
        if row["task"] == "ALL" and row["question"] == "DUFS contribution beyond IU-PCR"
    ]
    macro_evidence_pairs = [
        row for row in paired_rows
        if row["task"] == "MACRO_TASK"
        and row["question"] == "evidence gain over full-only"
        and row["challenger"].startswith("original30_")
    ]
    macro_dufs_pairs = [
        row for row in paired_rows
        if row["task"] == "MACRO_TASK"
        and row["question"] == "DUFS contribution beyond IU-PCR"
    ]
    best_evidence = max(evidence_pairs, key=lambda row: float(row["delta_auroc"]))
    best_dufs = max(dufs_pairs, key=lambda row: float(row["delta_auroc"]))
    best_macro_evidence = max(
        macro_evidence_pairs, key=lambda row: float(row["delta_auroc"])
    )
    best_macro_dufs = max(
        macro_dufs_pairs, key=lambda row: float(row["delta_auroc"])
    )
    full_best = max(
        overall["original30_full__iu_pcr"]["auroc"],
        overall["original30_full__dufs_liu"]["auroc"],
    )
    summary_lookup = {
        (row["task"], row["method"]): row for row in summary_rows
    }
    return {
        "status": "exploratory comparison; not blinded confirmation",
        "questions": {
            "original_30_useful_for_rag": {
                "best_full_only_auroc": full_best,
                "gasp_top50_auroc": overall["gasp_top50"]["auroc"],
                "full_only_dufs_qa_auroc": summary_lookup[
                    ("QA", "original30_full__dufs_liu")
                ]["auroc"],
                "full_only_dufs_data2txt_auroc": summary_lookup[
                    ("Data2txt", "original30_full__dufs_liu")
                ]["auroc"],
                "full_only_dufs_task_macro_auroc": summary_lookup[
                    ("MACRO_TASK", "original30_full__dufs_liu")
                ]["auroc"],
                "interpretation_rule": (
                    "Report ranking performance and comparison with GASP; do not treat "
                    "test-selected superiority as confirmation."
                ),
            },
            "evidence_perturbation_improves_original_30": {
                "largest_observed_paired_gain": best_evidence,
                "largest_task_macro_paired_gain": best_macro_evidence,
                "any_registered_variant_interval_excludes_zero": any(
                    float(row["ci_low"]) > 0.0 for row in evidence_pairs
                ),
            },
            "dufs_adds_beyond_iu_pcr": {
                "largest_observed_paired_gain": best_dufs,
                "largest_task_macro_paired_gain": best_macro_dufs,
                "any_variant_interval_excludes_zero": any(
                    float(row["ci_low"]) > 0.0 for row in dufs_pairs
                ),
            },
        },
        "confirmation_requirement": (
            "Freeze one hypothesis and test it on a new benchmark or scorer before "
            "making a final method claim."
        ),
    }


PRIMARY_METHODS = (
    "gasp_top50",
    "ec_iu_pcr",
    "ec_dufs_liu",
    "original30_full__iu_pcr",
    "original30_full__dufs_liu",
    "original30_noctx__iu_pcr",
    "original30_noctx__dufs_liu",
    "original30_loo__iu_pcr",
    "original30_loo__dufs_liu",
    "hybrid__iu_pcr",
    "hybrid__dufs_liu",
)


DISPLAY_NAMES = {
    "gasp_top50": "GASP-top50",
    "ec_iu_pcr": "EC-IU-PCR",
    "ec_dufs_liu": "EC-DUFS-LIU",
    "original30_full__iu_pcr": "Original-30 full / IU",
    "original30_full__dufs_liu": "Original-30 full / DUFS",
    "original30_noctx__iu_pcr": "Original-30 noctx / IU",
    "original30_noctx__dufs_liu": "Original-30 noctx / DUFS",
    "original30_loo__iu_pcr": "Original-30 LOO / IU",
    "original30_loo__dufs_liu": "Original-30 LOO / DUFS",
    "hybrid__iu_pcr": "Hybrid / IU",
    "hybrid__dufs_liu": "Hybrid / DUFS",
}


def _figure_uri(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _performance_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected = {
        row["method"]: row for row in rows
        if row["task"] == "ALL" and row["method"] in PRIMARY_METHODS
    }
    methods = [method for method in PRIMARY_METHODS if method in selected]
    positions = np.arange(len(methods))
    colors = [
        "#64748b", "#7c3aed", "#a78bfa",
        "#0284c7", "#38bdf8", "#047857", "#34d399",
        "#b45309", "#f59e0b", "#be123c", "#fb7185",
    ][:len(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    for axis, metric in zip(axes, ("auroc", "auprc")):
        values = np.asarray([float(selected[m][metric]) for m in methods])
        low = np.asarray([float(selected[m][metric + "_ci_low"]) for m in methods])
        high = np.asarray([float(selected[m][metric + "_ci_high"]) for m in methods])
        axis.bar(positions, values, color=colors)
        axis.errorbar(
            positions, values, yerr=[values - low, high - values],
            fmt="none", color="#111827", capsize=3,
        )
        axis.set_xticks(
            positions,
            [DISPLAY_NAMES[m].replace(" / ", "\n") for m in methods],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel(metric.upper())
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(max(0.0, low.min() - 0.04), min(1.0, high.max() + 0.05))
    fig.suptitle("Test LOO responses: source-grouped 95% intervals")
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _task_heatmap(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    tasks = ["QA", "Data2txt"]
    methods = [
        method for method in PRIMARY_METHODS
        if any(row["method"] == method for row in rows)
    ]
    lookup = {
        (row["task"], row["method"]): float(row["auroc"])
        for row in rows if row["task"] in tasks
    }
    matrix = np.asarray([
        [lookup.get((task, method), np.nan) for method in methods] for task in tasks
    ])
    fig, axis = plt.subplots(figsize=(14.5, 3.5))
    image = axis.imshow(matrix, vmin=0.35, vmax=0.85, cmap="RdYlBu", aspect="auto")
    for row_index in range(len(tasks)):
        for column_index in range(len(methods)):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(
        range(len(methods)),
        [DISPLAY_NAMES[m].replace(" / ", "\n") for m in methods],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(tasks)), tasks)
    axis.set_title("Test AUROC by task")
    fig.colorbar(image, ax=axis, fraction=0.025)
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _delta_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected_by_task = {
        task: [
            row for row in rows if row["task"] == task
            and row["question"] in {
                "evidence gain over full-only", "DUFS contribution beyond IU-PCR"
            }
        ]
        for task in ("ALL", "MACRO_TASK")
    }
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.5), sharey=True)
    for axis, task, title in zip(
        axes,
        ("ALL", "MACRO_TASK"),
        ("Pooled responses", "Equal weight for QA and Data-to-Text"),
    ):
        selected = selected_by_task[task]
        labels = []
        for row in selected:
            question = "evidence" if row["question"].startswith("evidence") else "DUFS"
            labels.append(
                f"{DISPLAY_NAMES.get(row['challenger'], row['challenger'])}\n({question})"
            )
        values = np.asarray([float(row["delta_auroc"]) for row in selected])
        low = np.asarray([float(row["ci_low"]) for row in selected])
        high = np.asarray([float(row["ci_high"]) for row in selected])
        positions = np.arange(len(selected))
        colors = ["#16a34a" if value > 0 else "#dc2626" for value in values]
        axis.barh(positions, values, color=colors, alpha=0.8)
        axis.errorbar(
            values, positions, xerr=[values - low, high - values],
            fmt="none", color="#111827", capsize=3,
        )
        axis.axvline(0.0, color="#111827", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.set_xlabel("Paired AUROC change")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    fig.suptitle("Does evidence help? Does DUFS help beyond IU-PCR?")
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _gate_heatmap(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    variants = ["original30_full", "original30_noctx", "original30_loo", "hybrid"]
    lookup: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["variant"] in variants and row["base_feature"] in ORIGINAL_FEATURES:
            lookup.setdefault((row["variant"], row["base_feature"]), []).append(
                float(row["dufs_gate"])
            )
    matrix = np.asarray([
        [np.mean(lookup.get((variant, feature), [np.nan])) for feature in ORIGINAL_FEATURES]
        for variant in variants
    ])
    fig, axis = plt.subplots(figsize=(17, 4.2))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(ORIGINAL_FEATURES)), ORIGINAL_FEATURES, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(variants)), [value.replace("original30_", "") for value in variants])
    axis.set_title("Mean DUFS gate by original feature (averaged across its evidence blocks)")
    fig.colorbar(image, ax=axis, fraction=0.02)
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _condition_gate_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["task"] == "ALL"]
    lookup = {
        (row["condition"], row["feature"]): float(row["gate"]) for row in selected
    }
    matrix = np.asarray([
        [lookup[(condition, feature)] - lookup[("full", feature)] for feature in ORIGINAL_FEATURES]
        for condition in ("noctx", "loo_mean")
    ])
    bound = max(float(np.max(np.abs(matrix))), 1e-3)
    fig, axis = plt.subplots(figsize=(17, 3.3))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(range(len(ORIGINAL_FEATURES)), ORIGINAL_FEATURES, rotation=55, ha="right", fontsize=8)
    axis.set_yticks((0, 1), ("noctx - full", "mean LOO - full"))
    axis.set_title("How condition-only DUFS gates change")
    fig.colorbar(image, ax=axis, fraction=0.02)
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _training_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["control"] == "observed"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    for axis, variant in zip(axes.ravel(), (
        "original30_full", "original30_noctx", "original30_loo", "hybrid"
    )):
        subset = [row for row in selected if row["variant"] == variant]
        for seed in sorted({row["seed"] for row in subset}):
            points = [row for row in subset if row["seed"] == seed]
            axis.plot(
                [int(row["epoch"]) for row in points],
                [float(row["loss"]) for row in points],
                label=f"seed {seed}",
            )
        axis.set_title(variant.replace("original30_", "Original-30 "))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("DUFS epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("parameter-free loss")
    fig.suptitle("DUFS convergence on the four frozen matrices")
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _confound_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["task"] == "ALL" and row["method"] in PRIMARY_METHODS
    ]
    methods = [method for method in PRIMARY_METHODS if any(row["method"] == method for row in selected)]
    lookup = {row["method"]: row for row in selected}
    columns = ("rho_response_length", "rho_context_length", "rho_chunk_count")
    matrix = np.asarray([[float(lookup[method][column]) for column in columns] for method in methods])
    fig, axis = plt.subplots(figsize=(7.8, 7.0))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(columns)):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(3), ("answer length", "context length", "chunk count"))
    axis.set_yticks(range(len(methods)), [DISPLAY_NAMES[m] for m in methods])
    axis.set_title("Spearman score correlations with nuisance variables")
    fig.colorbar(image, ax=axis, fraction=0.04)
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _feature_task_figure(rows: list[dict[str, str]], path: Path) -> str:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["variant"] == "original30_loo"
        and row["task"] in {"QA", "Data2txt"}
        and row["block"].startswith("loo_")
        and row["hallucination_auroc_from_negative_column"] not in ("", None)
    ]
    lookup: dict[tuple[str, str], list[float]] = {}
    for row in selected:
        lookup.setdefault((row["task"], row["base_feature"]), []).append(
            float(row["hallucination_auroc_from_negative_column"])
        )
    matrix = np.asarray([
        [max(lookup.get((task, feature), [np.nan]), key=lambda value: abs(value - 0.5))
         for feature in ORIGINAL_FEATURES]
        for task in ("QA", "Data2txt")
    ])
    fig, axis = plt.subplots(figsize=(17, 3.2))
    image = axis.imshow(matrix, cmap="RdYlBu", vmin=0.3, vmax=0.7, aspect="auto")
    axis.set_xticks(range(len(ORIGINAL_FEATURES)), ORIGINAL_FEATURES, rotation=55, ha="right", fontsize=8)
    axis.set_yticks((0, 1), ("QA", "Data-to-Text"))
    axis.set_title("Strongest univariate LOO summary per original feature and task (descriptive)")
    fig.colorbar(image, ax=axis, fraction=0.02)
    uri = _figure_uri(fig, path)
    plt.close(fig)
    return uri


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _method_table(rows: list[dict[str, str]]) -> str:
    lookup = {
        (row["task"], row["method"]): row for row in rows
        if row["method"] in PRIMARY_METHODS
    }
    values = []
    for method in PRIMARY_METHODS:
        if ("ALL", method) not in lookup:
            continue
        row = lookup[("ALL", method)]
        values.append((
            DISPLAY_NAMES[method],
            f"{float(row['auroc']):.4f}",
            f"{float(lookup[('MACRO_TASK', method)]['auroc']):.4f}",
            f"{float(lookup[('TASK_STANDARDIZED_POOL', method)]['auroc']):.4f}",
            f"{float(lookup[('QA', method)]['auroc']):.4f}",
            f"{float(lookup[('Data2txt', method)]['auroc']):.4f}",
            f"{float(row['auprc']):.4f}",
        ))
    return _table((
        "Method", "Pooled AUROC", "Task-macro AUROC", "Task-standardized pooled AUROC",
        "QA AUROC", "Data-to-Text AUROC", "Pooled AUPRC",
    ), values)


def _mechanism_table(rows: list[dict[str, str]]) -> str:
    selected = [
        row for row in rows if row["task"] == "ALL"
        and row["question"] in {
            "evidence gain over full-only", "DUFS contribution beyond IU-PCR",
            "observed versus condition-block permutation",
        }
    ]
    values = [(
        DISPLAY_NAMES.get(row["challenger"], row["challenger"]),
        DISPLAY_NAMES.get(row["reference"], row["reference"]),
        row["question"],
        f"{float(row['delta_auroc']):+.4f}",
        f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]",
    ) for row in selected]
    return _table(("Challenger", "Reference", "Question", "Delta", "95% interval"), values)


def build_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    figures = out / "figures"
    summary = _read_csv(out / "summary_test.csv")
    paired = _read_csv(out / "paired_test.csv")
    confounds = _read_csv(out / "confounds_test.csv")
    weights = _read_csv(out / "fusion_weights_test.csv")
    condition_gates = _read_csv(out / "condition_gates_test.csv")
    training = _read_csv(out / "training_history_test.csv")
    feature_tasks = _read_csv(out / "feature_task_diagnostics_test.csv")
    if not summary:
        raise RuntimeError("test evaluation artifacts are missing")
    conclusion = json.loads((out / "exploratory_conclusion.json").read_text())
    test_manifest = json.loads((out / "score_manifest_test.json").read_text())
    dev_manifest = json.loads((out / "score_manifest_dev.json").read_text())
    diagnostics = json.loads((out / "diagnostics" / "fit_test.json").read_text())

    image_uris = {
        "performance": _performance_figure(summary, figures / "performance.png"),
        "tasks": _task_heatmap(summary, figures / "task_heatmap.png"),
        "deltas": _delta_figure(paired, figures / "paired_deltas.png"),
        "gates": _gate_heatmap(weights, figures / "feature_gates.png"),
        "conditions": _condition_gate_figure(condition_gates, figures / "condition_gate_changes.png"),
        "training": _training_figure(training, figures / "dufs_convergence.png"),
        "confounds": _confound_figure(confounds, figures / "confounds.png"),
        "feature_tasks": _feature_task_figure(feature_tasks, figures / "feature_task_map.png"),
    }

    overall = {row["method"]: row for row in summary if row["task"] == "ALL"}
    macro = {row["method"]: row for row in summary if row["task"] == "MACRO_TASK"}
    best_method = max(
        (method for method in PRIMARY_METHODS if method in overall),
        key=lambda method: float(overall[method]["auroc"]),
    )
    best_macro_method = max(
        (method for method in PRIMARY_METHODS if method in macro),
        key=lambda method: float(macro[method]["auroc"]),
    )
    evidence = conclusion["questions"]["evidence_perturbation_improves_original_30"]
    dufs = conclusion["questions"]["dufs_adds_beyond_iu_pcr"]
    constant_rows = []
    for variant, diag in diagnostics.items():
        constant_rows.append((
            variant,
            diag["input_columns"],
            diag["kept_columns"],
            ", ".join(diag["constant_columns"]) or "none",
            f"{float(diag['dufs']['effective_feature_count']):.2f}",
            f"{float(diag['dufs']['mean_seed_std']):.4f}",
        ))
    contract_table = _table(
        ("Matrix", "Nominal", "Used", "Constant columns", "Effective features", "Mean gate seed SD"),
        constant_rows,
    )

    result_statement = (
        f"The highest pooled exploratory test AUROC is "
        f"{float(overall[best_method]['auroc']):.4f} for {DISPLAY_NAMES[best_method]}. "
        f"After giving QA and Data-to-Text equal weight, the highest task-macro AUROC is "
        f"{float(macro[best_macro_method]['auroc']):.4f} for "
        f"{DISPLAY_NAMES[best_macro_method]}. This difference is important because pooled "
        "AUROC can reward task identification rather than within-task grounding. No test "
        "result was used to revise a variant."
    )
    methods_html = """
<div class="flow">
  <div><b>Fixed answer</b><small>same tokens in every condition</small></div><span>→</span>
  <div><b>RAG traces</b><small>full · noctx · each LOO</small></div><span>→</span>
  <div><b>Original 30</b><small>same mixed-v2 extractors</small></div><span>→</span>
  <div><b>Evidence blocks</b><small>full · changes · LOO summaries</small></div><span>→</span>
  <div><b>Fusion</b><small>IU-PCR or DUFS-LIU</small></div><span>→</span>
  <div><b>Evaluation</b><small>labels only here</small></div>
</div>
"""
    css = """
body{font-family:Inter,system-ui,sans-serif;margin:0;background:#f8fafc;color:#172033;line-height:1.55}
main{max-width:1180px;margin:auto;padding:36px 24px 70px}h1{font-size:2.35rem;margin-bottom:4px}h2{margin-top:44px;border-bottom:1px solid #dbe4ef;padding-bottom:8px}h3{margin-top:28px}.hero{background:linear-gradient(135deg,#0f172a,#1d4ed8);color:white;padding:34px;border-radius:18px}.tag{display:inline-block;background:#fef3c7;color:#92400e;padding:5px 10px;border-radius:999px;font-weight:700}.card{background:white;border:1px solid #dbe4ef;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 16px #0f172a0b}.flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.flow div{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:10px 13px;min-width:120px}.flow small{display:block;color:#475569}.flow span{font-size:1.4rem;color:#64748b}img{max-width:100%;height:auto;background:white;border-radius:12px;border:1px solid #e2e8f0}table{border-collapse:collapse;width:100%;font-size:.9rem;background:white}th,td{padding:8px 10px;border:1px solid #dbe4ef;text-align:left}th{background:#eaf0f8;position:sticky;top:0}.scroll{overflow-x:auto}.good{border-left:5px solid #16a34a}.warn{border-left:5px solid #f59e0b}.bad{border-left:5px solid #dc2626}code{background:#e2e8f0;padding:2px 5px;border-radius:4px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:800px){.two{grid-template-columns:1fr}}
"""
    html_report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAGTruth original-30 evidence-aware experiment</title><style>{css}</style></head><body><main>
<section class="hero"><span class="tag">Exploratory · labels previously opened</span><h1>RAGTruth original-30 evidence-aware experiment</h1><p>This experiment keeps the same 30 mixed-v2 features and asks whether RAG evidence conditions help IU-PCR or DUFS-LIU use them better.</p></section>
<section class="card warn"><b>Claim boundary.</b> This is not a blinded confirmation. RAGTruth labels had already been opened in the earlier Evidence-Contrast experiment. Feature extraction and fitting remain label-free, but any selected hypothesis needs new data.</section>
<h2>Question and experiment map</h2><p>{html.escape(result_statement)}</p>{methods_html}
<h2>Terms and metrics</h2><div class="two"><div class="card"><b>AUROC</b><p>The probability that a randomly chosen hallucinated response receives a higher risk score than a non-hallucinated response. 0.5 is random ranking.</p></div><div class="card"><b>AUPRC</b><p>Precision-recall area. It emphasizes the positive hallucination class and depends on its frequency.</p></div><div class="card"><b>Paired interval</b><p>We resample complete source_id groups 1,000 times. Every method uses the same draws, so differences are paired.</p></div><div class="card"><b>Effective feature count</b><p>A gate-concentration measure. It is high when DUFS spreads weight over many columns and low when it concentrates.</p></div></div>
<h2>Data contract</h2><p>Development has {dev_manifest['n_responses']} LOO responses. Test has {test_manifest['n_responses']}. All 30 original features are finite in full, no-context and every observed LOO condition. The shared full-fitted transform reproduces original mixed-v2 with maximum error {float(test_manifest['exact_full_mixed_v2_transform_error']):.3g}. No missing feature or chunk was imputed.</p><div class="scroll">{contract_table}</div>
<h2>Performance</h2><p>The plot shows pooled response performance. The table also gives task-macro and task-standardized results, which reduce the effect of task composition.</p><img src="{image_uris['performance']}" alt="Performance with grouped intervals"><div class="scroll">{_method_table(summary)}</div>
<h2>Task transfer</h2><p>QA and Data-to-Text are shown separately. A pooled score can hide a task reversal.</p><img src="{image_uris['tasks']}" alt="Task AUROC heatmap">
<h2>Three scientific questions</h2><div class="card"><b>Do evidence conditions improve the original 30?</b><p>The largest pooled comparison is <code>{html.escape(str(evidence['largest_observed_paired_gain']['challenger']))}</code> versus its full-only reference: {float(evidence['largest_observed_paired_gain']['delta_auroc']):+.4f}, interval [{float(evidence['largest_observed_paired_gain']['ci_low']):+.4f}, {float(evidence['largest_observed_paired_gain']['ci_high']):+.4f}]. The largest task-macro change is {float(evidence['largest_task_macro_paired_gain']['delta_auroc']):+.4f} [{float(evidence['largest_task_macro_paired_gain']['ci_low']):+.4f}, {float(evidence['largest_task_macro_paired_gain']['ci_high']):+.4f}].</p></div><div class="card"><b>Does DUFS add beyond IU-PCR?</b><p>The largest pooled DUFS-minus-IU comparison is {float(dufs['largest_observed_paired_gain']['delta_auroc']):+.4f}, interval [{float(dufs['largest_observed_paired_gain']['ci_low']):+.4f}, {float(dufs['largest_observed_paired_gain']['ci_high']):+.4f}]. The largest task-macro DUFS change is {float(dufs['largest_task_macro_paired_gain']['delta_auroc']):+.4f} [{float(dufs['largest_task_macro_paired_gain']['ci_low']):+.4f}, {float(dufs['largest_task_macro_paired_gain']['ci_high']):+.4f}].</p></div><img src="{image_uris['deltas']}" alt="Paired AUROC changes"><div class="scroll">{_mechanism_table(paired)}</div>
<h2>What DUFS relies on</h2><p>The first heatmap averages gates for the same original feature across its full, no-context and LOO summary columns. The second holds the 30-column basis fixed and shows how gates change when DUFS sees only full, no-context or mean-LOO coordinates.</p><img src="{image_uris['gates']}" alt="Feature gate heatmap"><img src="{image_uris['conditions']}" alt="Condition gate changes">
<h2>Gate stability and convergence</h2><p>Three fixed seeds are used. Stable loss is necessary but does not prove that the learned graph predicts grounding.</p><img src="{image_uris['training']}" alt="DUFS training curves">
<h2>Feature and task interaction</h2><p>This is a descriptive, label-open diagnostic. It shows the strongest univariate LOO summary for each original feature in each task. It is not used to select columns.</p><img src="{image_uris['feature_tasks']}" alt="Feature task interaction">
<h2>Confounds</h2><p>These correlations test whether a score mainly tracks answer length, context length or chunk count. The CSV also contains AUROC after linear residualization.</p><img src="{image_uris['confounds']}" alt="Confound correlations">
<h2>Audit and reproducibility</h2><ul><li>Score hashes: dev <code>{dev_manifest['score_sha256']}</code>; test <code>{test_manifest['score_sha256']}</code>.</li><li>Labels are structurally absent from tensor construction and fitting.</li><li>All methods use the same response order and grouped bootstrap draws.</li><li>Condition permutations are performed within task and preserve block marginals.</li><li>Every LIU path checks exact equality to IU-PCR at lambda=0.</li><li>The exact omitted-chunk text is unavailable in the cache; the experiment uses condition indexes only.</li></ul>
<h2>Conclusion</h2><p>{html.escape(result_statement)}</p><p>The correct interpretation is exploratory. Freeze one resulting hypothesis and confirm it with a new benchmark or scorer before making a final claim.</p>
<h2>Files</h2><p><code>METHODS.md</code> defines the mathematics and paper provenance. <code>RUNBOOK.md</code> gives exact commands. CSV files contain every metric, paired difference, gate, weight, confound and feature diagnostic.</p>
</main></body></html>"""
    (out / "REPORT.html").write_text(html_report, encoding="utf-8")

    report_md = f"""# RAGTruth original-30 evidence-aware experiment

**Status:** exploratory comparison, not a blinded confirmation.

## Result

{result_statement}

The largest evidence-versus-full-only change is
**{float(evidence['largest_observed_paired_gain']['delta_auroc']):+.4f}** with a
source-grouped 95% interval
**[{float(evidence['largest_observed_paired_gain']['ci_low']):+.4f},
{float(evidence['largest_observed_paired_gain']['ci_high']):+.4f}]**.

With equal weight for QA and Data-to-Text, the largest pure Original-30
evidence gain is
**{float(evidence['largest_task_macro_paired_gain']['delta_auroc']):+.4f}** with
interval **[{float(evidence['largest_task_macro_paired_gain']['ci_low']):+.4f},
{float(evidence['largest_task_macro_paired_gain']['ci_high']):+.4f}]**.

The largest DUFS-minus-IU change is
**{float(dufs['largest_observed_paired_gain']['delta_auroc']):+.4f}** with interval
**[{float(dufs['largest_observed_paired_gain']['ci_low']):+.4f},
{float(dufs['largest_observed_paired_gain']['ci_high']):+.4f}]**.

The largest task-macro DUFS-minus-IU change is
**{float(dufs['largest_task_macro_paired_gain']['delta_auroc']):+.4f}** with
interval **[{float(dufs['largest_task_macro_paired_gain']['ci_low']):+.4f},
{float(dufs['largest_task_macro_paired_gain']['ci_high']):+.4f}]**.

All 30 original features were available in every full, no-context and observed
LOO condition. No missing feature or chunk was imputed. The report separates
QA and Data-to-Text and includes condition permutations, graph permutations,
gate stability, fusion weights, confounds and chunk/task diagnostics.

See [`REPORT.html`](REPORT.html) for the visual report and [`METHODS.md`](METHODS.md)
for the mathematical definition.

## Claim boundary

RAGTruth labels were opened before this comparison. No labels enter fitting,
but a final method claim requires confirmation on a new benchmark or scorer.
"""
    (out / "REPORT.md").write_text(report_md, encoding="utf-8")
    _write_json(out / "experiment_manifest.json", {
        "experiment": VERSION,
        "status": "complete exploratory comparison",
        "dev_score_sha256": dev_manifest["score_sha256"],
        "test_score_sha256": test_manifest["score_sha256"],
        "report": "REPORT.html",
        "methods": "METHODS.md",
        "runbook": "RUNBOOK.md",
        "figures": sorted(path.name for path in figures.glob("*.png")),
        "confirmation_required": True,
    })
    print(f"[report] {out / 'REPORT.html'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split", choices=("dev", "test"), required=True)
    common.add_argument("--cache", required=True)
    common.add_argument("--official-responses", required=True)
    common.add_argument("--tokenizer", required=True)
    common.add_argument("--out", default=str(DEFAULT_OUT))

    score = subparsers.add_parser("score", parents=[common])
    score.add_argument("--reference-scores", required=True)
    score.set_defaults(function=score_split)

    evaluate = subparsers.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.set_defaults(function=evaluate_split)

    reproduce = subparsers.add_parser("reproduction-audit")
    reproduce.add_argument("--split", choices=("dev", "test"), required=True)
    reproduce.add_argument("--old-scores", required=True)
    reproduce.add_argument("--out", default=str(DEFAULT_OUT))
    reproduce.set_defaults(function=reproduction_audit)

    report = subparsers.add_parser("report")
    report.add_argument("--out", default=str(DEFAULT_OUT))
    report.set_defaults(function=build_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
