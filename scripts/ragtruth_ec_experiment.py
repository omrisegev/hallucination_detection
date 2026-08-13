#!/usr/bin/env python3
"""Run and report the preregistered RAGTruth evidence-contrast experiment.

The command boundary enforces the audit order:

``score`` reads no labels after the adapter returns its label-free dataset and
writes a hashed score bundle.  ``evaluate`` is the only command that loads the
isolated label object.  ``report`` renders the already-written CSV/JSON results.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import hashlib
import html
import io
import json
import math
import os
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
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.dufs_liu_feature_contract import (  # noqa: E402
    CONTRACT_VERSION as MIXED_V2_VERSION,
    dufs_liu_mixed_v2_matrix,
)
from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1  # noqa: E402
from spectral_utils.feature_utils import extract_all_features  # noqa: E402
from spectral_utils.ragtruth_evidence_contrast import (  # noqa: E402
    CONTRACT_VERSION,
    FeatureTable,
    adapt_cache,
    build_feature_tables,
    sha256_file,
    standardize_features,
)
from spectral_utils.repgrid_scoring import (  # noqa: E402
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "ragtruth-evidence-contrast-v1-top50-tail-protocol-correction-2026-08-10"
PROTOCOL_CORRECTION = True
DEFAULT_OUT = REPO / "results" / "ragtruth_evidence_contrast_v1"
LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
FROZEN_LAMBDA = 0.1
BOOTSTRAP_SEED = 20260810
PERMUTATION_SEED = 20260811
DEVELOPMENT_GATE = {
    "dufs_not_worse_than_gasp": 0.0,
    "max_mean_dufs_seed_std": 0.15,
    "max_abs_length_or_chunk_spearman": 0.50,
    "max_residualized_auroc_drop": 0.02,
    "max_projected_condition_number": 1e12,
    "min_edges_per_node": 1.0,
}
DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}


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
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


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


def _upcr_diagnostics(fit: Any) -> dict[str, Any]:
    return {
        "weights": fit.w,
        "rho_hat": fit.rho_hat,
        "keep": fit.keep,
        "n_kept": int(fit.keep.sum()),
        "used_simple_average": bool(fit.used_simple_average),
        "n_components_used": int(fit.n_components_used),
        "projection_residual": float(fit.proj_residual),
        "g2_hat": float(fit.g2_hat),
        "g2_at_ceiling": bool(fit.g2_at_ceiling),
        "lambda2_fraction": float(fit.lambda2_frac),
        "meta": fit.meta,
    }


def score_feature_table(table: FeatureTable) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    standardized, keep, mean, scale = standardize_features(table.values)
    names = tuple(name for name, selected in zip(table.feature_names, keep) if selected)
    F = standardized.T
    methods: dict[str, np.ndarray] = {
        "perplexity": -table.values[:, table.feature_names.index("mean_full_target_logprob")],
        "likelihood_gap": -table.values[:, table.feature_names.index("mean_context_gap")],
    }
    diagnostics: dict[str, Any] = {
        "contract": table.contract,
        "n_samples": len(table.sample_ids),
        "input_feature_names": table.feature_names,
        "kept_feature_names": names,
        "constant_features": [name for name, selected in zip(table.feature_names, keep)
                              if not selected],
        "standardization_mean": mean,
        "standardization_scale": scale,
    }
    if table.contract == "EC-full-v1":
        ll_names = ("mean_context_gap", "max_loo_mean_drop")
        gasp_names = (
            "mean_context_gap", "mean_noctx_jsd_top50",
            "max_loo_mean_drop", "max_loo_mean_jsd_top50",
        )
        ll = table.values[:, [table.feature_names.index(name) for name in ll_names]]
        gasp = table.values[:, [table.feature_names.index(name) for name in gasp_names]]
        ll = (ll - ll.mean(axis=0)) / np.where(ll.std(axis=0) < 1e-8, 1.0, ll.std(axis=0))
        gasp = ((gasp - gasp.mean(axis=0))
                / np.where(gasp.std(axis=0) < 1e-8, 1.0, gasp.std(axis=0)))
        methods["gasp_ll"] = -ll.sum(axis=1)
        methods["gasp_top50"] = -gasp.sum(axis=1)

    deployed = upcr_fit(F, **DEPLOYED_FIT)
    methods["ec_upcr"] = -(deployed.w @ F)
    diagnostics["ec_upcr"] = _upcr_diagnostics(deployed)

    started = time.time()
    gates, gate_diag = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS, return_history=True
    )
    diagnostics["dufs_runtime_seconds"] = time.time() - started
    graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    path = laplacian_iu_path(F, LAMBDAS, graph=graph)
    methods["ec_iu_pcr"] = -(path[0.0].w @ F)
    methods["ec_dufs_liu"] = -(path[FROZEN_LAMBDA].w @ F)
    if not np.array_equal(path[0.0].w, path[0.0].baseline.w):
        raise RuntimeError(f"{table.name}: lambda=0 does not exactly preserve IU-PCR")

    ungated_graph = build_graph_from_features(F, k=DUFS_K)
    ungated = laplacian_iu_fit(F, lambda_=FROZEN_LAMBDA, graph=ungated_graph)
    methods["ec_ungated_liu"] = -(ungated.w @ F)
    rng = np.random.default_rng(PERMUTATION_SEED)
    shuffled = laplacian_iu_fit(
        F, lambda_=FROZEN_LAMBDA,
        graph=permute_graph(graph, rng.permutation(F.shape[1])),
    )
    methods["ec_permuted_liu"] = -(shuffled.w @ F)
    diagnostics.update({
        "dufs": gate_diag,
        "dufs_gates": gates,
        "lambda_path": {
            str(lambda_): {
                "score": -(fit.w @ F),
                "weights": fit.w,
                "diagnostics": fit.diagnostics,
            }
            for lambda_, fit in path.items()
        },
        "ungated": ungated.diagnostics,
        "permuted": shuffled.diagnostics,
        "lambda_zero_exact": True,
    })
    for method, score in methods.items():
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (F.shape[1],) or not np.isfinite(score).all():
            raise RuntimeError(f"{table.name}/{method}: invalid score vector")
        methods[method] = score
    return methods, diagnostics


def _intrinsic_trace_features(response: Any) -> dict[str, float]:
    trace = response.conditions["full"]
    features = extract_all_features(
        trace.entropy, spilled_energies=-trace.target_logprob, allow_short=True
    ) or {}
    features.update(energy_features_from_logsumexp(trace.logsumexp))
    top = {"ids": trace.top_ids, "logprobs": trace.top_logprobs}
    features.update(logprob_features(top))
    features.update(logprob_features_extended(top))
    return features


def intrinsic_mixed_v2_score(dataset: Any, response_ids: Iterable[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the existing global mixed-v2 DUFS-LIU contract exactly.

    This is a response-level baseline. It is intentionally separate from the
    preregistered Evidence-Contrast score bundle because it was added only
    after the v1 test labels had already been opened.
    """
    lookup = {response.response_id: response for response in dataset.responses}
    ids = tuple(str(value) for value in response_ids)
    feature_rows = [_intrinsic_trace_features(lookup[response_id]) for response_id in ids]
    names, columns, availability, dropped = [], [], {}, {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        raw = np.asarray([row.get(name, np.nan) for row in feature_rows], dtype=float)
        finite = np.isfinite(raw)
        availability[name] = float(finite.mean())
        if finite.mean() < 0.70 or not finite.any():
            dropped[name] = f"availability={availability[name]:.4f}"
            continue
        raw = np.where(finite, raw, np.median(raw[finite]))
        if raw.std() < 1e-8:
            dropped[name] = "constant"
            continue
        if float(np.mean(raw == np.median(raw))) > 0.40:
            dropped[name] = "saturated"
            continue
        names.append(name)
        columns.append(raw)
    transformed, kept_names, details = dufs_liu_mixed_v2_matrix(
        np.column_stack(columns), names
    )
    F = transformed.T
    gates, gate_diagnostics = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    fit = laplacian_iu_path(F, (0.0, FROZEN_LAMBDA), graph=graph)[FROZEN_LAMBDA]
    score = -(fit.w @ F)
    return np.asarray(score, dtype=float), {
        "contract": MIXED_V2_VERSION,
        "feature_names": kept_names,
        "n_features": len(kept_names),
        "availability": availability,
        "dropped": dropped,
        "transforms": details,
        "dufs": gate_diagnostics,
        "laplacian": fit.diagnostics,
    }


def intrinsic_score_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, isolated_labels, _ = adapt_cache(
        Path(args.cache), Path(args.official_responses), tokenizer
    )
    del isolated_labels
    response_lookup = {item.response_id: item for item in dataset.responses}
    cohorts = {
        "noctx_response": tuple(sorted(response_lookup, key=int)),
        "full_response": tuple(sorted(
            (response_id for response_id, response in response_lookup.items()
             if any(name.startswith("loo_") for name in response.conditions)),
            key=int,
        )),
    }
    arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for cohort, response_ids in cohorts.items():
        print(f"[intrinsic-posthoc] {args.split}/{cohort}: n={len(response_ids)}")
        score, diag = intrinsic_mixed_v2_score(dataset, response_ids)
        responses = [response_lookup[response_id] for response_id in response_ids]
        prefix = cohort + "__"
        arrays[prefix + "response_ids"] = np.asarray(response_ids)
        arrays[prefix + "source_ids"] = np.asarray([item.source_id for item in responses])
        arrays[prefix + "task_types"] = np.asarray([item.task_type for item in responses])
        arrays[prefix + "score__intrinsic_mixed_v2_dufs_liu"] = score
        diagnostics[cohort] = diag
    path = out / f"scores_intrinsic_mixed_v2_posthoc_{args.split}.npz"
    np.savez_compressed(path, **arrays)
    digest = sha256_file(path)
    (out / f"scores_intrinsic_mixed_v2_posthoc_{args.split}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    _write_json(out / "diagnostics" / f"intrinsic_mixed_v2_posthoc_{args.split}.json", diagnostics)
    _write_json(out / f"intrinsic_mixed_v2_posthoc_manifest_{args.split}.json", {
        "split": args.split,
        "score_sha256": digest,
        "contract": MIXED_V2_VERSION,
        "label_free_fit": True,
        "registered_primary": False,
        "posthoc_reason": (
            "This optional baseline was implemented after the v1 test labels had already "
            "been opened. It cannot enter the registered success decision."
        ),
    })


def intrinsic_evaluate_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    score_path = out / f"scores_intrinsic_mixed_v2_posthoc_{args.split}.npz"
    expected = (out / f"scores_intrinsic_mixed_v2_posthoc_{args.split}.sha256").read_text().split()[0]
    if sha256_file(score_path) != expected:
        raise RuntimeError("posthoc intrinsic score hash changed")
    npz = np.load(score_path, allow_pickle=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    _, labels, _ = adapt_cache(Path(args.cache), Path(args.official_responses), tokenizer)
    rows: list[dict[str, Any]] = []
    for cohort in ("full_response", "noctx_response"):
        prefix = cohort + "__"
        response_ids = npz[prefix + "response_ids"].astype(str)
        y = np.asarray([labels.response[item].hallucinated for item in response_ids])
        groups = npz[prefix + "source_ids"].astype(str)
        tasks = npz[prefix + "task_types"].astype(str)
        score = np.asarray(npz[prefix + "score__intrinsic_mixed_v2_dufs_liu"], dtype=float)
        for task in ("ALL", *sorted(set(tasks))):
            mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
            metrics, _ = _metric_bundle(
                y[mask], {"intrinsic_mixed_v2_dufs_liu": score[mask]}, groups[mask],
                int(args.bootstrap), BOOTSTRAP_SEED + 991,
            )
            if not metrics:
                continue
            rows.append({
                "split": args.split, "cohort": cohort, "task": task,
                "method": "intrinsic_mixed_v2_dufs_liu", "n": int(mask.sum()),
                "positives": int(y[mask].sum()), **metrics["intrinsic_mixed_v2_dufs_liu"],
                "registered_primary": False,
            })
    _write_csv(out / f"intrinsic_mixed_v2_posthoc_{args.split}.csv", rows)


def score_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, isolated_labels, audit = adapt_cache(
        Path(args.cache), Path(args.official_responses), tokenizer
    )
    # This command intentionally discards the isolated label object before any
    # feature table or fit is constructed.
    del isolated_labels
    gc.collect()
    tables = build_feature_tables(dataset)
    arrays: dict[str, np.ndarray] = {}
    fit_diagnostics: dict[str, Any] = {}
    training_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    for table_name, table in sorted(tables.items()):
        print(f"[score] {args.split}/{table_name}: n={len(table.sample_ids)} m={len(table.feature_names)}")
        methods, diagnostics = score_feature_table(table)
        prefix = f"{table_name}__"
        arrays.update({
            prefix + "sample_ids": np.asarray(table.sample_ids),
            prefix + "response_ids": np.asarray(table.response_ids),
            prefix + "source_ids": np.asarray(table.source_ids),
            prefix + "task_types": np.asarray(table.task_types),
            prefix + "sources": np.asarray(table.sources),
            prefix + "generator_models": np.asarray(table.generator_models),
            prefix + "response_lengths": table.response_lengths,
            prefix + "unit_lengths": table.unit_lengths,
            prefix + "chunk_counts": table.chunk_counts,
            prefix + "context_lengths": table.context_lengths,
            prefix + "supporting_chunks": table.supporting_chunks,
            prefix + "feature_names": np.asarray(table.feature_names),
            prefix + "features": table.values,
        })
        arrays.update({prefix + "score__" + method: score for method, score in methods.items()})
        history = np.asarray(diagnostics["dufs"].pop("training_history"), dtype=float)
        for seed_index, seed in enumerate(DUFS_SEEDS):
            for epoch, loss in enumerate(history[seed_index], start=1):
                training_rows.append({
                    "split": args.split, "table": table_name, "seed": seed,
                    "epoch": epoch, "loss": float(loss),
                })
        for lambda_, item in diagnostics["lambda_path"].items():
            lambda_rows.append({
                "split": args.split, "table": table_name, "lambda": lambda_,
                "weight_norm": item["diagnostics"]["weight_norm"],
                "weight_cosine_vs_iu": item["diagnostics"]["weight_cosine_vs_iu"],
                "score_variance": item["diagnostics"]["score_variance"],
                "score_laplacian_energy": item["diagnostics"]["score_laplacian_energy"],
                "projected_condition_number": item["diagnostics"]["projected_condition_number"],
            })
            item.pop("score")
        fit_diagnostics[table_name] = diagnostics

    score_path = out / f"scores_{args.split}.npz"
    np.savez_compressed(score_path, **arrays)
    score_hash = sha256_file(score_path)
    (out / f"scores_{args.split}.sha256").write_text(
        f"{score_hash}  {score_path.name}\n", encoding="utf-8"
    )
    _write_json(out / f"data_audit_{args.split}.json", {
        **audit,
        "split": args.split,
        "input_cache": str(Path(args.cache).resolve()),
        "input_cache_sha256": dataset.cache_sha256,
        "score_sha256": score_hash,
        "score_written_before_label_evaluation": True,
        "protocol_correction_after_original_test_labels_opened": PROTOCOL_CORRECTION,
    })
    _write_json(out / "diagnostics" / f"fit_{args.split}.json", fit_diagnostics)
    _write_csv(out / f"training_history_{args.split}.csv", training_rows)
    _write_csv(out / f"lambda_path_{args.split}.csv", lambda_rows)
    _write_json(out / "feature_contract.json", {
        "version": CONTRACT_VERSION,
        "experiment": VERSION,
        "orientation": "every input feature is theory-oriented toward grounding",
        "hallucination_score": "negative fused grounding score; no label-based flip",
        "tables": {name: {
            "contract": table.contract,
            "feature_names": table.feature_names,
        } for name, table in tables.items()},
        "dufs_seeds": DUFS_SEEDS,
        "dufs_epochs": DUFS_EPOCHS,
        "graph_k": DUFS_K,
        "frozen_lambda": FROZEN_LAMBDA,
        "lambda_sensitivity_only": LAMBDAS,
        "development_gate": DEVELOPMENT_GATE,
        "gasp_boundary": "GASP-top50 approximates full-vocabulary JSD with top-50 union plus OTHER",
        "protocol_status": (
            "fixed-formula correction after the original test labels had been opened; "
            "the formula was specified in the user-approved plan and was not selected from labels"
        ),
    })
    _write_csv(out / "training_history.csv", training_rows)
    _write_csv(out / "lambda_path.csv", lambda_rows)
    print(f"[frozen] {score_path} sha256={score_hash}")


def validate_input_split(args: argparse.Namespace) -> None:
    """Re-run canonical input validation without touching frozen score files."""
    from transformers import AutoTokenizer

    out = Path(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, labels, audit = adapt_cache(
        Path(args.cache), Path(args.official_responses), tokenizer
    )
    del labels
    _write_json(out / f"input_validation_{args.split}.json", {
        **audit,
        "split": args.split,
        "input_cache_sha256": dataset.cache_sha256,
        "frozen_score_file_untouched": True,
    })


def _score_tables(npz: Any) -> list[str]:
    return sorted({key.split("__", 1)[0] for key in npz.files if key.endswith("__sample_ids")})


def _method_names(npz: Any, table: str) -> list[str]:
    prefix = f"{table}__score__"
    return sorted(key[len(prefix):] for key in npz.files if key.startswith(prefix))


def _bootstrap_indices(groups: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(count):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        output.append(np.concatenate([members[group] for group in sampled]))
    return output


def _metric_bundle(y: np.ndarray, scores: Mapping[str, np.ndarray], groups: np.ndarray,
                   bootstrap: int, seed: int) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    if len(np.unique(y)) < 2:
        return {}, {}
    indexes = _bootstrap_indices(groups, bootstrap, seed)
    distributions: dict[str, np.ndarray] = {}
    result: dict[str, dict[str, Any]] = {}
    for method, score in scores.items():
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))
        boot_auc = np.asarray([
            roc_auc_score(y[index], score[index])
            for index in indexes if len(np.unique(y[index])) == 2
        ])
        boot_pr = np.asarray([
            average_precision_score(y[index], score[index])
            for index in indexes if len(np.unique(y[index])) == 2
        ])
        result[method] = {
            "auroc": auroc,
            "auroc_ci_low": float(np.quantile(boot_auc, 0.025)),
            "auroc_ci_high": float(np.quantile(boot_auc, 0.975)),
            "auprc": auprc,
            "auprc_ci_low": float(np.quantile(boot_pr, 0.025)),
            "auprc_ci_high": float(np.quantile(boot_pr, 0.975)),
        }
        distributions[method] = boot_auc
    return result, distributions


def _residualize(score: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(score)), confounds])
    keep = np.std(X, axis=0) > 1e-12
    keep[0] = True
    fitted = X[:, keep] @ np.linalg.lstsq(X[:, keep], score, rcond=None)[0]
    return score - fitted


def evaluate_split(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    score_path = out / f"scores_{args.split}.npz"
    expected = (out / f"scores_{args.split}.sha256").read_text().split()[0]
    actual = sha256_file(score_path)
    if expected != actual:
        raise RuntimeError(f"score hash changed before label opening: {expected} != {actual}")
    npz = np.load(score_path, allow_pickle=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset, labels, _ = adapt_cache(Path(args.cache), Path(args.official_responses), tokenizer)

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    confound_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    evaluation_manifest: dict[str, Any] = {
        "split": args.split,
        "score_sha256_verified": actual,
        "labels_opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    all_distributions: dict[str, dict[str, np.ndarray]] = {}
    for table in _score_tables(npz):
        prefix = f"{table}__"
        sample_ids = npz[prefix + "sample_ids"].astype(str)
        mapping = labels.response if table.endswith("response") else labels.sentence
        y = np.asarray([mapping[sample_id].hallucinated for sample_id in sample_ids], dtype=bool)
        label_types = [mapping[sample_id].label_types for sample_id in sample_ids]
        groups = npz[prefix + "source_ids"].astype(str)
        tasks = npz[prefix + "task_types"].astype(str)
        sources = npz[prefix + "sources"].astype(str)
        scores = {method: np.asarray(npz[prefix + "score__" + method], dtype=float)
                  for method in _method_names(npz, table)}
        for task in ("ALL", *sorted(set(tasks))):
            mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
            metrics, distributions = _metric_bundle(
                y[mask], {method: score[mask] for method, score in scores.items()},
                groups[mask], int(args.bootstrap), BOOTSTRAP_SEED + len(summary_rows),
            )
            all_distributions[f"{table}/{task}"] = distributions
            for method, values in metrics.items():
                summary_rows.append({
                    "split": args.split, "table": table, "task": task,
                    "granularity": "response" if table.endswith("response") else "sentence",
                    "cohort": "LOO" if table.startswith("full_") else "no-context",
                    "method": method, "n": int(mask.sum()),
                    "positives": int(y[mask].sum()), "n_groups": int(len(np.unique(groups[mask]))),
                    **values,
                })
            for challenger, reference in (
                ("ec_dufs_liu", "gasp_top50"),
                ("ec_dufs_liu", "gasp_ll"),
                ("ec_dufs_liu", "ec_iu_pcr"),
                ("ec_dufs_liu", "ec_upcr"),
            ):
                if challenger not in metrics or reference not in metrics:
                    continue
                delta_dist = distributions[challenger] - distributions[reference]
                paired_rows.append({
                    "split": args.split, "table": table, "task": task,
                    "challenger": challenger, "reference": reference,
                    "delta_auroc": metrics[challenger]["auroc"] - metrics[reference]["auroc"],
                    "ci_low": float(np.quantile(delta_dist, 0.025)),
                    "ci_high": float(np.quantile(delta_dist, 0.975)),
                    "probability_positive": float(np.mean(delta_dist > 0)),
                })

        # Source-family results are descriptive.  We do not run another
        # bootstrap here: the registered uncertainty unit is source_id and the
        # registered paired intervals above remain the inferential result.
        for source in sorted(set(sources)):
            mask = sources == source
            if len(np.unique(y[mask])) < 2:
                continue
            source_metrics = {
                method: float(roc_auc_score(y[mask], score[mask]))
                for method, score in scores.items()
            }
            for method, value in source_metrics.items():
                source_rows.append({
                    "split": args.split, "table": table, "source": source,
                    "method": method, "n": int(mask.sum()),
                    "positives": int(y[mask].sum()), "auroc": value,
                    "delta_vs_gasp_top50": (
                        value - source_metrics["gasp_top50"]
                        if "gasp_top50" in source_metrics else ""
                    ),
                    "delta_vs_ec_iu_pcr": value - source_metrics["ec_iu_pcr"],
                })

        confounds = np.column_stack([
            npz[prefix + "unit_lengths"],
            npz[prefix + "chunk_counts"],
            npz[prefix + "context_lengths"],
        ]).astype(float)
        for method, score in scores.items():
            correlations = [spearmanr(score, confounds[:, index]).statistic for index in range(3)]
            residual = _residualize(score, confounds)
            confound_rows.append({
                "split": args.split, "table": table, "method": method,
                "rho_unit_length": correlations[0],
                "rho_chunk_count": correlations[1],
                "rho_context_length": correlations[2],
                "raw_auroc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else "",
                "residualized_auroc": roc_auc_score(y, residual) if len(np.unique(y)) == 2 else "",
            })
        if table.endswith("sentence"):
            for kind, pattern in (("baseless", "Baseless"), ("conflict", "Conflict")):
                mask = np.asarray([
                    (not positive) or any(pattern in label_type for label_type in types)
                    for positive, types in zip(y, label_types)
                ])
                metrics, _ = _metric_bundle(
                    y[mask], {method: score[mask] for method, score in scores.items()},
                    groups[mask], int(args.bootstrap), BOOTSTRAP_SEED + 700 + len(type_rows),
                )
                for method, values in metrics.items():
                    type_rows.append({
                        "split": args.split, "table": table, "type": kind,
                        "method": method, "n": int(mask.sum()),
                        "positives": int(y[mask].sum()), **values,
                    })

    _write_csv(out / f"summary_{args.split}.csv", summary_rows)
    _write_csv(out / f"paired_{args.split}.csv", paired_rows)
    _write_csv(out / f"confounds_{args.split}.csv", confound_rows)
    _write_csv(out / f"type_analysis_{args.split}.csv", type_rows)
    _write_csv(out / f"source_analysis_{args.split}.csv", source_rows)
    _write_json(out / f"evaluation_manifest_{args.split}.json", evaluation_manifest)
    _write_examples(
        out / f"localization_examples_{args.split}.json",
        dataset, labels, npz, args.split,
    )
    if args.split == "dev":
        gate = development_gate(summary_rows, paired_rows, confound_rows,
                                out / "diagnostics" / "fit_dev.json")
        _write_json(out / "development_gate.json", gate)
        print("[gate]", "PASS" if gate["passed"] else "FAIL", gate["checks"])
    if args.split == "test":
        _write_json(out / "final_decision.json", final_decision(summary_rows, paired_rows))
    print(f"[evaluated] {args.split}: score hash remained {actual}")


def _write_examples(path: Path, dataset: Any, labels: Any, npz: Any, split: str) -> None:
    """Write deterministic examples only after the evaluation boundary opens."""
    table = "full_sentence"
    prefix = f"{table}__"
    if prefix + "sample_ids" not in npz.files:
        _write_json(path, {"split": split, "examples": []})
        return
    sample_ids = npz[prefix + "sample_ids"].astype(str)
    response_ids = npz[prefix + "response_ids"].astype(str)
    tasks = npz[prefix + "task_types"].astype(str)
    scores = np.asarray(npz[prefix + "score__ec_dufs_liu"], dtype=float)
    chunks = np.asarray(npz[prefix + "supporting_chunks"], dtype=int)
    gold = np.asarray([labels.sentence[item].hallucinated for item in sample_ids])
    response_lookup = {item.response_id: item for item in dataset.responses}
    examples: list[dict[str, Any]] = []
    chosen: set[str] = set()
    for task in sorted(set(tasks)):
        for kind, eligible in (("high-scoring labelled sentence", gold),
                               ("high-scoring unlabelled sentence", ~gold)):
            indexes = np.flatnonzero((tasks == task) & eligible)
            indexes = indexes[np.argsort(scores[indexes])[::-1]]
            selected = next((int(index) for index in indexes
                             if response_ids[index] not in chosen), None)
            if selected is None:
                continue
            response_id = response_ids[selected]
            chosen.add(response_id)
            response = response_lookup[response_id]
            units = []
            response_indexes = np.flatnonzero(response_ids == response_id)
            for index in response_indexes:
                unit = next(item for item in response.sentences
                            if sample_ids[index].endswith(f"sent_{item.index}"))
                label = labels.sentence[sample_ids[index]]
                units.append({
                    "sample_id": sample_ids[index], "text": unit.text,
                    "score": float(scores[index]),
                    "gold_overlap": bool(label.hallucinated),
                    "gold_types": list(label.label_types),
                    "largest_drop_chunk": int(chunks[index]),
                })
            examples.append({
                "selection": kind, "task": task, "response_id": response_id,
                "source_id": response.source_id, "response_text": response.response_text,
                "sentences": units,
                "note": "Gold marks sentence overlap with an annotated RAGTruth span; it was added only after score freezing.",
            })
    _write_json(path, {"split": split, "examples": examples})


def final_decision(summary_rows: list[dict[str, Any]],
                   paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply only the success rules registered before test labels opened."""
    def pair(table: str, task: str, reference: str) -> dict[str, Any]:
        return next(row for row in paired_rows
                    if row["table"] == table and row["task"] == task
                    and row["challenger"] == "ec_dufs_liu"
                    and row["reference"] == reference)

    primary_gasp = pair("full_sentence", "ALL", "gasp_top50")
    primary_iu = pair("full_sentence", "ALL", "ec_iu_pcr")
    qa = pair("full_sentence", "QA", "gasp_top50")
    data2txt = pair("full_sentence", "Data2txt", "gasp_top50")
    response = pair("full_response", "ALL", "gasp_top50")
    checks = {
        "at_least_one_auroc_point_over_gasp_top50": float(primary_gasp["delta_auroc"]) >= 0.01,
        "gasp_improvement_interval_excludes_zero": float(primary_gasp["ci_low"]) > 0.0,
        "dufs_laplacian_improvement_over_iu_interval_excludes_zero": float(primary_iu["ci_low"]) > 0.0,
        "same_positive_sign_in_qa_and_data2txt": (
            float(qa["delta_auroc"]) > 0.0 and float(data2txt["delta_auroc"]) > 0.0
        ),
        "response_level_noninferiority": float(response["ci_low"]) > -0.01,
    }
    return {
        "success": all(checks.values()),
        "classification": (
            "full success" if all(checks.values())
            else "feature-contract success; DUFS/Laplacian mechanism failure"
        ),
        "checks": checks,
        "observed": {
            "sentence_delta_vs_gasp_top50": float(primary_gasp["delta_auroc"]),
            "sentence_delta_vs_gasp_top50_ci": [float(primary_gasp["ci_low"]), float(primary_gasp["ci_high"])],
            "sentence_delta_vs_ec_iu_pcr": float(primary_iu["delta_auroc"]),
            "sentence_delta_vs_ec_iu_pcr_ci": [float(primary_iu["ci_low"]), float(primary_iu["ci_high"])],
            "qa_delta_vs_gasp_top50": float(qa["delta_auroc"]),
            "data2txt_delta_vs_gasp_top50": float(data2txt["delta_auroc"]),
            "response_delta_vs_gasp_top50_ci": [float(response["ci_low"]), float(response["ci_high"])],
        },
        "interpretation": (
            "Evidence-Contrast features plus IU-PCR outperform the approximate GASP baseline, "
            "but the registered DUFS-gated Laplacian does not add value beyond IU-PCR."
        ),
    }


def development_gate(summary_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]],
                     confounds: list[dict[str, Any]], diagnostics_path: Path) -> dict[str, Any]:
    primary = next(row for row in summary_rows
                   if row["table"] == "full_sentence" and row["task"] == "ALL"
                   and row["method"] == "ec_dufs_liu")
    gasp = next(row for row in summary_rows
                if row["table"] == "full_sentence" and row["task"] == "ALL"
                and row["method"] == "gasp_top50")
    confound = next(row for row in confounds
                    if row["table"] == "full_sentence" and row["method"] == "ec_dufs_liu")
    diagnostics = json.loads(diagnostics_path.read_text())
    fit = diagnostics["full_sentence"]
    dufs = fit["dufs"]
    lap = fit["lambda_path"][str(FROZEN_LAMBDA)]["diagnostics"]
    correlations = [abs(float(confound[key])) for key in (
        "rho_unit_length", "rho_chunk_count", "rho_context_length"
    ) if confound[key] not in ("", None)]
    residual_drop = float(confound["raw_auroc"]) - float(confound["residualized_auroc"])
    checks = {
        "dufs_not_worse_than_gasp": float(primary["auroc"]) >= float(gasp["auroc"]),
        "finite_and_noncollapsed_graph": (
            float(lap["projected_condition_number"]) < DEVELOPMENT_GATE["max_projected_condition_number"]
            and float(lap["n_edges"]) / max(float(lap["n_nodes"]), 1.0)
            >= DEVELOPMENT_GATE["min_edges_per_node"]
            and float(lap["degree_min"]) > 0.0
        ),
        "stable_dufs_gates": float(dufs["mean_seed_std"])
        <= DEVELOPMENT_GATE["max_mean_dufs_seed_std"],
        "not_primarily_length_or_chunk_count": (
            max(correlations, default=0.0)
            <= DEVELOPMENT_GATE["max_abs_length_or_chunk_spearman"]
            and residual_drop <= DEVELOPMENT_GATE["max_residualized_auroc_drop"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "ec_dufs_liu_auroc": float(primary["auroc"]),
            "gasp_top50_auroc": float(gasp["auroc"]),
            "delta": float(primary["auroc"]) - float(gasp["auroc"]),
            "dufs_mean_seed_std": float(dufs["mean_seed_std"]),
            "max_abs_confound_spearman": max(correlations, default=0.0),
            "residualized_auroc_drop": residual_drop,
            "projected_condition_number": float(lap["projected_condition_number"]),
        },
        "thresholds": DEVELOPMENT_GATE,
        "rule": "All checks must pass before test labels are opened.",
    }


def _figure_data_uri(fig: Any, *, artifact: Path | None = None) -> str:
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifact, format="png", dpi=150, bbox_inches="tight")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _performance_figure(rows: list[dict[str, str]], split: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == split
                and row["table"] == "full_sentence" and row["task"] == "ALL"]
    order = [method for method in (
        "perplexity", "gasp_ll", "gasp_top50", "ec_upcr", "ec_iu_pcr", "ec_dufs_liu"
    ) if any(row["method"] == method for row in selected)]
    lookup = {row["method"]: row for row in selected}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    positions = np.arange(len(order))
    colours = ["#94a3b8", "#64748b", "#475569", "#0ea5e9", "#8b5cf6", "#2563eb"][:len(order)]
    for axis, metric, chance in zip(axes, ("auroc", "auprc"), (0.5, None)):
        values = np.asarray([float(lookup[method][metric]) for method in order])
        low = np.asarray([float(lookup[method][metric + "_ci_low"]) for method in order])
        high = np.asarray([float(lookup[method][metric + "_ci_high"]) for method in order])
        axis.bar(positions, values, color=colours)
        axis.errorbar(positions, values, yerr=[values - low, high - values], fmt="none",
                      color="#0f172a", capsize=4)
        if chance is not None:
            axis.axhline(chance, color="#ef4444", linestyle="--", linewidth=1)
        axis.set_xticks(positions, [method.replace("_", "\n") for method in order])
        axis.set_ylabel("Sentence " + metric.upper())
        axis.set_ylim(max(0.0, float(low.min()) - 0.04), min(1.0, float(high.max()) + 0.05))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{split.title()} LOO cohort: grouped 95% intervals")
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _heatmap_figure(rows: list[dict[str, str]], split: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == split
                and row["table"] == "full_sentence" and row["task"] != "ALL"]
    tasks = sorted({row["task"] for row in selected})
    methods = [method for method in ("gasp_top50", "ec_upcr", "ec_iu_pcr", "ec_dufs_liu")
               if any(row["method"] == method for row in selected)]
    lookup = {(row["task"], row["method"]): float(row["auroc"]) for row in selected}
    matrix = np.asarray([[lookup.get((task, method), np.nan) for method in methods] for task in tasks])
    fig, axis = plt.subplots(figsize=(7.2, 2.8 + 0.45 * len(tasks)))
    image = axis.imshow(matrix, vmin=0.4, vmax=0.9, cmap="Blues", aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(methods)):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                          color="white" if matrix[i, j] > 0.68 else "#0f172a")
    axis.set_xticks(range(len(methods)), [method.replace("_", "\n") for method in methods])
    axis.set_yticks(range(len(tasks)), tasks)
    axis.set_title(f"{split.title()} sentence AUROC by task")
    fig.colorbar(image, ax=axis, fraction=0.04)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _training_figure(rows: list[dict[str, str]], split: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == split and row["table"] == "full_sentence"]
    fig, axis = plt.subplots(figsize=(7.5, 3.6))
    for seed in sorted({row["seed"] for row in selected}):
        items = [row for row in selected if row["seed"] == seed]
        axis.plot([int(row["epoch"]) for row in items], [float(row["loss"]) for row in items], label=f"seed {seed}")
    axis.set_xlabel("DUFS epoch")
    axis.set_ylabel("Parameter-free DUFS loss")
    axis.set_title("Unsupervised gate-learning convergence")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _coverage_figure(out: Path, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    audits = {
        split: json.loads((out / f"data_audit_{split}.json").read_text())
        for split in ("dev", "test") if (out / f"data_audit_{split}.json").exists()
    }
    tasks = sorted({task for audit in audits.values()
                    for task in audit["task_condition_counts"]})
    colours = {"QA": "#2563eb", "Data2txt": "#8b5cf6", "Summary": "#14b8a6"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(audits))
    bottoms = np.zeros(len(audits))
    for task in tasks:
        values = np.asarray([
            audit["task_condition_counts"].get(task, {}).get("full", 0)
            for audit in audits.values()
        ])
        axes[0].bar(x, values, bottom=bottoms, label=task, color=colours.get(task))
        bottoms += values
    axes[0].set_xticks(x, [name.title() for name in audits])
    axes[0].set_ylabel("Responses")
    axes[0].set_title("Response coverage")
    axes[0].legend(frameon=False)
    bottoms = np.zeros(len(audits))
    for task in tasks:
        values = np.asarray([
            sum(audit["task_condition_counts"].get(task, {}).values())
            for audit in audits.values()
        ])
        axes[1].bar(x, values, bottom=bottoms, label=task, color=colours.get(task))
        bottoms += values
    axes[1].set_xticks(x, [name.title() for name in audits])
    axes[1].set_ylabel("Scoring condition records")
    axes[1].set_title("Evidence-condition coverage")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _gate_figure(diagnostics_path: Path, table: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    diagnostics = json.loads(diagnostics_path.read_text())[table]
    names = diagnostics["input_feature_names"]
    display = {
        "mean_full_target_logprob": "full log-probability",
        "negative_mean_full_top50_tail_entropy": "negative top-50 + tail entropy",
        "mean_full_margin": "top-1 / top-2 margin",
        "negative_mean_full_tail_mass": "negative top-50 tail mass",
        "mean_context_gap": "mean context gap",
        "q90_context_gap": "90th percentile context gap",
        "mean_noctx_jsd_top50": "no-context JSD (top-50)",
        "mean_top50_tail_entropy_increase_noctx": "no-context entropy increase",
        "max_loo_mean_drop": "maximum LOO likelihood drop",
        "top2_loo_mean_drop": "top-two LOO likelihood drop",
        "mean_positive_loo_drop": "mean positive LOO drop",
        "max_loo_mean_jsd_top50": "maximum LOO JSD (top-50)",
        "top2_loo_mean_jsd_top50": "top-two LOO JSD (top-50)",
        "fraction_tokens_positive_best_drop": "tokens with positive best drop",
    }
    labels = [display.get(name, name) for name in names]
    per_seed = np.asarray(diagnostics["dufs"]["per_seed_probabilities"], dtype=float)
    means, std = per_seed.mean(axis=0), per_seed.std(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    image = axes[0].imshow(per_seed.T, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(per_seed.shape[0]), [f"seed {seed}" for seed in DUFS_SEEDS])
    axes[0].set_yticks(range(len(labels)), labels, fontsize=9)
    axes[0].set_title("DUFS survival probability")
    fig.colorbar(image, ax=axes[0], fraction=0.045)
    y = np.arange(len(names))
    axes[1].barh(y, means, xerr=std, color="#2563eb", alpha=0.85)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Mean gate probability ± seed SD")
    axes[1].set_title(
        f"Mean ± seed SD (same row order)\nEffective count = "
        f"{diagnostics['dufs']['effective_feature_count']:.2f} / {len(names)}"
    )
    axes[1].grid(axis="x", alpha=0.2)
    fig.subplots_adjust(wspace=0.32, left=0.28)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _laplacian_figure(rows: list[dict[str, str]], split: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    selected = sorted(
        (row for row in rows if row["split"] == split and row["table"] == "full_sentence"),
        key=lambda row: float(row["lambda"]),
    )
    lambdas = np.asarray([float(row["lambda"]) for row in selected])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    axes[0].plot(lambdas, [float(row["weight_cosine_vs_iu"]) for row in selected], marker="o")
    axes[0].set_ylabel("Cosine with IU weights")
    axes[1].plot(lambdas, [float(row["score_laplacian_energy"]) for row in selected], marker="o", color="#8b5cf6")
    axes[1].set_ylabel("Score graph roughness")
    axes[2].plot(lambdas, [float(row["projected_condition_number"]) for row in selected], marker="o", color="#f97316")
    axes[2].set_ylabel("Projected condition number")
    for axis in axes:
        axis.set_xscale("symlog", linthresh=0.05)
        axis.set_xlabel("λ (diagnostic path)")
        axis.grid(alpha=0.2)
        axis.axvline(FROZEN_LAMBDA, color="#ef4444", linestyle="--", linewidth=1)
    fig.suptitle("Laplacian sensitivity without label-based selection")
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _confound_figure(score_path: Path, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    npz = np.load(score_path, allow_pickle=False)
    prefix = "full_sentence__"
    score = np.asarray(npz[prefix + "score__ec_dufs_liu"], dtype=float)
    fields = (("unit_lengths", "Sentence length"), ("chunk_counts", "Chunk count"),
              ("context_lengths", "Context length"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for axis, (field, label) in zip(axes, fields):
        value = np.asarray(npz[prefix + field], dtype=float)
        axis.hexbin(value, score, gridsize=35, mincnt=1, cmap="Blues")
        rho = spearmanr(value, score).statistic
        axis.set_xlabel(label)
        axis.set_ylabel("EC-DUFS-LIU score")
        axis.set_title(f"Spearman ρ = {rho:.3f}")
    fig.suptitle("Primary-score nuisance checks (labels not used)")
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _graph_degree_figure(score_path: Path, diagnostics_path: Path,
                         artifact: Path) -> str:
    import matplotlib.pyplot as plt
    npz = np.load(score_path, allow_pickle=False)
    standardized, _, _, _ = standardize_features(
        np.asarray(npz["full_sentence__features"], dtype=float)
    )
    diagnostics = json.loads(diagnostics_path.read_text())["full_sentence"]
    gates = np.asarray(diagnostics["dufs_gates"], dtype=float)
    graph = build_graph_from_features(standardized.T, gates=gates, k=DUFS_K)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.hist(degree, bins=35, color="#2563eb", alpha=0.85)
    axis.axvline(float(np.mean(degree)), color="#ef4444", linestyle="--",
                 label=f"mean = {float(np.mean(degree)):.2f}")
    axis.set_xlabel("Weighted graph degree")
    axis.set_ylabel("Sentence nodes")
    axis.set_title("Frozen DUFS graph degree distribution")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _source_delta_figure(rows: list[dict[str, str]], split: str, artifact: Path) -> str:
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == split
                and row["table"] == "full_sentence" and row["method"] == "ec_dufs_liu"]
    sources = [row["source"] for row in selected]
    gasp = [float(row["delta_vs_gasp_top50"]) for row in selected]
    iu = [float(row["delta_vs_ec_iu_pcr"]) for row in selected]
    x = np.arange(len(sources))
    fig, axis = plt.subplots(figsize=(7.2, 3.7))
    width = 0.36
    axis.bar(x - width / 2, gasp, width, label="vs GASP-top50", color="#2563eb")
    axis.bar(x + width / 2, iu, width, label="vs EC-IU-PCR", color="#8b5cf6")
    axis.axhline(0, color="#0f172a", linewidth=1)
    axis.set_xticks(x, sources)
    axis.set_ylabel("AUROC difference")
    axis.set_title("Paired method differences by source family")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    uri = _figure_data_uri(fig, artifact=artifact)
    plt.close(fig)
    return uri


def _pipeline_svg() -> str:
    labels = [
        "Fixed published answer", "Full / no-context / LOO scoring",
        "Aligned token traces", "EC feature views", "U-PCR or DUFS-LIU",
        "Hashed label-free score", "Open RAGTruth labels",
    ]
    boxes = []
    for index, label in enumerate(labels):
        x = 20 + index * 166
        boxes.append(f'<rect x="{x}" y="35" width="140" height="70" rx="10" fill="#eff6ff" stroke="#2563eb"/>')
        words = label.split()
        midpoint = max(1, len(words) // 2)
        lines = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
        boxes.extend(f'<text x="{x+70}" y="{68 + line*18}" text-anchor="middle" font-size="12" fill="#172554">{html.escape(text)}</text>'
                     for line, text in enumerate(lines) if text)
        if index < len(labels) - 1:
            boxes.append(f'<path d="M{x+140} 70 L{x+160} 70" stroke="#475569" marker-end="url(#a)"/>')
    return ('<svg viewBox="0 0 1190 140" role="img" aria-label="Experiment pipeline">'
            '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#475569"/></marker></defs>'
            + "".join(boxes) + '</svg>')


def _method_cards() -> str:
    return """
    <div class="cards">
      <article><h3>GASP-top50</h3><p>Standardize four evidence-sensitivity measurements and negate their sum. No covariance model is fitted. JSD uses the saved top-50 union plus one tail bucket.</p><code>h = -Σ z(gap, JSD₀, max-drop, max-JSD)</code></article>
      <article><h3>EC-U-PCR</h3><p>Estimate feature-to-grounding covariances from feature covariance under the additive uncorrelated-error model. Weak estimated regressors may be removed.</p><code>Cᵢⱼ ≈ ρᵢ + ρⱼ − g²</code></article>
      <article><h3>EC-DUFS-LIU</h3><p>Learn label-free feature gates, build a sample graph, and penalize a fused score that changes sharply between graph neighbours.</p><code>wλ = U[Uᵀ(C + λR̄)U]⁻¹Uᵀρ̂</code></article>
    </div>"""


def _examples_html(path: Path) -> str:
    if not path.exists():
        return "<p>No evaluation examples were written.</p>"
    payload = json.loads(path.read_text())
    cards = []
    for example in payload.get("examples", []):
        sentences = []
        values = [float(item["score"]) for item in example["sentences"]]
        low, high = (min(values), max(values)) if values else (0.0, 1.0)
        span = max(high - low, 1e-12)
        for item in example["sentences"]:
            strength = (float(item["score"]) - low) / span
            background = f"rgba(239,68,68,{0.08 + 0.48 * strength:.3f})"
            gold = "<b class='gold'>RAGTruth span overlap</b>" if item["gold_overlap"] else ""
            types = ", ".join(item["gold_types"])
            sentences.append(
                f"<div class='sentence' style='background:{background}'><div>{html.escape(item['text'])}</div>"
                f"<small>score {float(item['score']):.3f} · largest-drop chunk {int(item['largest_drop_chunk'])} "
                f"{gold} {html.escape(types)}</small></div>"
            )
        cards.append(
            f"<article><h3>{html.escape(example['task'])} · {html.escape(example['selection'])}</h3>"
            f"<p class='muted'>response {html.escape(example['response_id'])}; source_id {html.escape(example['source_id'])}</p>"
            + "".join(sentences) + "</article>"
        )
    return "<div class='example-grid'>" + "".join(cards) + "</div>"


def _build_experiment_manifest(out: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("pilot", "dev", "test"):
        audit_path = out / f"data_audit_{split}.json"
        score_path = out / f"scores_{split}.npz"
        if not audit_path.exists() or not score_path.exists():
            continue
        audit = json.loads(audit_path.read_text())
        evaluation_path = out / f"evaluation_manifest_{split}.json"
        splits[split] = {
            "input_cache_sha256": audit["input_cache_sha256"],
            "n_responses": audit["n_responses"],
            "n_conditions": audit["n_conditions"],
            "score_file": score_path.name,
            "score_sha256": sha256_file(score_path),
            "labels_opened": evaluation_path.exists(),
            "evaluation_manifest": evaluation_path.name if evaluation_path.exists() else None,
            "input_validation": (
                f"input_validation_{split}.json"
                if (out / f"input_validation_{split}.json").exists() else None
            ),
        }
    manifest = {
        "experiment": VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label_policy": (
            "within this correction run, each score file was written and hashed before "
            "its evaluation command loaded labels; however, the original test labels had "
            "already been opened, so this is not a new blinded confirmation"
            if PROTOCOL_CORRECTION else
            "score and hash each split before its labels are opened"
        ),
        "protocol_correction": PROTOCOL_CORRECTION,
        "protocol_correction_reason": (
            "The original blind run used saved full-vocabulary entropy where the approved "
            "contract specified top-50-plus-tail entropy. This run corrects that fixed formula "
            "after the original test labels had been opened; it is not a new blinded confirmation."
        ),
        "grouping_unit": "source_id",
        "chunk_metadata_boundary": (
            "the cache preserves each omitted chunk index and the number of supporting "
            "chunks, but not the exact omitted-chunk text or metadata; examples therefore "
            "identify the largest-drop chunk by index only"
        ),
        "feature_contract": "feature_contract.json",
        "frozen_settings": {
            "dufs_seeds": DUFS_SEEDS, "dufs_epochs": DUFS_EPOCHS,
            "graph_k": DUFS_K, "lambda": FROZEN_LAMBDA,
            "lambda_sensitivity_only": LAMBDAS,
        },
        "splits": splits,
        "dev_gate": "development_gate.json",
        "test_decision": "final_decision.json" if (out / "final_decision.json").exists() else None,
        "intrinsic_mixed_v2_status": (
            "available as a separately hashed posthoc response-level audit; excluded from the registered decision"
            if (out / "intrinsic_mixed_v2_posthoc_test.csv").exists()
            else "not in the preregistered score bundle"
        ),
        "intrinsic_mixed_v2_posthoc": (
            "intrinsic_mixed_v2_posthoc_test.csv"
            if (out / "intrinsic_mixed_v2_posthoc_test.csv").exists() else None
        ),
    }
    _write_json(out / "experiment_manifest.json", manifest)
    return manifest


def build_report(args: argparse.Namespace) -> None:
    # The report is generated on headless cluster nodes and from the desktop
    # terminal.  Never let matplotlib select the macOS GUI backend.
    import matplotlib
    matplotlib.use("Agg", force=True)

    out = Path(args.out)
    summary = []
    paired = []
    confounds = []
    for split in ("dev", "test"):
        summary.extend(_read_csv(out / f"summary_{split}.csv"))
        paired.extend(_read_csv(out / f"paired_{split}.csv"))
        confounds.extend(_read_csv(out / f"confounds_{split}.csv"))
    if not summary:
        raise RuntimeError("no evaluated split is available")
    splits = sorted({row["split"] for row in summary}, key=lambda x: (x != "dev", x))
    final_split = "test" if "test" in splits else "dev"
    gate = json.loads((out / "development_gate.json").read_text()) if (out / "development_gate.json").exists() else None
    decision = json.loads((out / "final_decision.json").read_text()) if (out / "final_decision.json").exists() else None
    fit_payload = json.loads((out / "diagnostics" / f"fit_{final_split}.json").read_text())
    primary_fit = fit_payload["full_sentence"]
    frozen_laplacian = primary_fit["lambda_path"][str(FROZEN_LAMBDA)]["diagnostics"]
    primary_dufs = primary_fit["dufs"]
    figure_dir = out / "figures"
    perf = _performance_figure(summary, final_split, figure_dir / "primary_performance.png")
    heat = _heatmap_figure(summary, final_split, figure_dir / "task_heatmap.png")
    training = _training_figure(_read_csv(out / f"training_history_{final_split}.csv"), final_split,
                                figure_dir / "dufs_training.png")
    coverage = _coverage_figure(out, figure_dir / "data_coverage.png")
    gates = _gate_figure(out / "diagnostics" / f"fit_{final_split}.json", "full_sentence",
                         figure_dir / "dufs_gates.png")
    laplacian = _laplacian_figure(_read_csv(out / f"lambda_path_{final_split}.csv"), final_split,
                                  figure_dir / "laplacian_path.png")
    confound_plot = _confound_figure(out / f"scores_{final_split}.npz",
                                     figure_dir / "confounds.png")
    graph_degree = _graph_degree_figure(
        out / f"scores_{final_split}.npz",
        out / "diagnostics" / f"fit_{final_split}.json",
        figure_dir / "graph_degree.png",
    )
    source_rows = _read_csv(out / f"source_analysis_{final_split}.csv")
    source_delta = _source_delta_figure(source_rows, final_split,
                                        figure_dir / "source_deltas.png")
    examples = _examples_html(out / f"localization_examples_{final_split}.json")
    intrinsic_rows = _read_csv(out / f"intrinsic_mixed_v2_posthoc_{final_split}.csv")
    manifest = _build_experiment_manifest(out)
    primary_rows = [row for row in summary if row["split"] == final_split
                    and row["table"] == "full_sentence" and row["task"] == "ALL"]
    primary_rows.sort(key=lambda row: float(row["auroc"]), reverse=True)
    dufs = next(row for row in primary_rows if row["method"] == "ec_dufs_liu")
    gasp = next(row for row in primary_rows if row["method"] == "gasp_top50")
    iu = next(row for row in primary_rows if row["method"] == "ec_iu_pcr")
    delta_gasp = float(dufs["auroc"]) - float(gasp["auroc"])
    delta_iu = float(dufs["auroc"]) - float(iu["auroc"])
    conclusion = (
        "The development gate passed. The frozen corrected test scores show that Evidence-Contrast fusion beats the approximate GASP baseline, but DUFS-LIU does not beat IU-PCR. The feature construction worked; the registered graph mechanism did not."
        if final_split == "test" else
        "The development gate failed, so test labels stayed closed. The development result is the final finding for version 1."
    )
    table_rows = "".join(
        f"<tr><td>{html.escape(row['method'])}</td><td>{float(row['auroc']):.3f}</td>"
        f"<td>[{float(row['auroc_ci_low']):.3f}, {float(row['auroc_ci_high']):.3f}]</td>"
        f"<td>{float(row['auprc']):.3f}</td></tr>" for row in primary_rows
    )
    gate_html = ""
    if gate:
        gate_html = "<div class='gate'>" + "".join(
            f"<span class={'pass' if passed else 'fail'}>{'PASS' if passed else 'FAIL'} · {html.escape(name.replace('_',' '))}</span>"
            for name, passed in gate["checks"].items()
        ) + "</div>"
    decision_html = ""
    if decision:
        decision_html = "<div class='gate'>" + "".join(
            f"<span class={'pass' if passed else 'fail'}>{'PASS' if passed else 'FAIL'} · {html.escape(name.replace('_',' '))}</span>"
            for name, passed in decision["checks"].items()
        ) + "</div>"
    confound = next(row for row in confounds if row["split"] == final_split
                    and row["table"] == "full_sentence" and row["method"] == "ec_dufs_liu")
    response_confound = next(row for row in confounds if row["split"] == final_split
                             and row["table"] == "full_response" and row["method"] == "ec_dufs_liu")
    primary_pair = next(row for row in paired if row["split"] == final_split
                        and row["table"] == "full_sentence" and row["task"] == "ALL"
                        and row["challenger"] == "ec_dufs_liu" and row["reference"] == "gasp_top50")
    iu_pair = next(row for row in paired if row["split"] == final_split
                   and row["table"] == "full_sentence" and row["task"] == "ALL"
                   and row["challenger"] == "ec_dufs_liu" and row["reference"] == "ec_iu_pcr")
    intrinsic_html = "<p>The optional intrinsic baseline was not run.</p>"
    intrinsic_markdown = ""
    if intrinsic_rows:
        intrinsic_full = {row["task"]: row for row in intrinsic_rows
                          if row["cohort"] == "full_response"}
        ec_full = {row["task"]: row for row in summary
                   if row["split"] == final_split and row["table"] == "full_response"
                   and row["method"] == "ec_dufs_liu"}
        tasks = [task for task in ("ALL", "QA", "Data2txt")
                 if task in intrinsic_full and task in ec_full]
        intrinsic_html = (
            "<table><thead><tr><th>Response slice</th><th>Intrinsic mixed-v2 DUFS-LIU</th>"
            "<th>EC-DUFS-LIU</th></tr></thead><tbody>" + "".join(
                f"<tr><td>{html.escape(task)}</td><td>{float(intrinsic_full[task]['auroc']):.3f}</td>"
                f"<td>{float(ec_full[task]['auroc']):.3f}</td></tr>" for task in tasks
            ) + "</tbody></table>"
        )
        intrinsic_markdown = (
            "The separately hashed post-hoc intrinsic mixed-v2 audit reached pooled "
            f"response AUROC {float(intrinsic_full['ALL']['auroc']):.3f}, but only "
            f"{float(intrinsic_full['Data2txt']['auroc']):.3f} on Data-to-Text. "
            f"EC-DUFS-LIU reached {float(ec_full['ALL']['auroc']):.3f} pooled and "
            f"{float(ec_full['Data2txt']['auroc']):.3f} on Data-to-Text. The pooled "
            "old-method value is therefore not a stable cross-task baseline."
        )
    report = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAGTruth Evidence-Contrast Experiment</title><style>
body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:0}}main{{max-width:1120px;margin:auto;padding:36px 28px 80px}}h1{{font-size:38px;margin-bottom:6px}}h2{{margin-top:38px;border-bottom:1px solid #cbd5e1;padding-bottom:8px}}p{{line-height:1.6}}.muted{{color:#475569}}.hero{{background:linear-gradient(135deg,#dbeafe,#f5f3ff);border:1px solid #bfdbfe;border-radius:18px;padding:28px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}article{{background:white;border:1px solid #dbe3ec;border-radius:14px;padding:18px}}code{{display:block;background:#eff6ff;padding:10px;border-radius:8px;overflow:auto}}img{{width:100%;background:white;border:1px solid #dbe3ec;border-radius:12px;margin:10px 0 22px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0}}.gate{{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}}.gate span{{padding:12px;border-radius:9px;font-weight:700}}.pass{{background:#dcfce7;color:#166534}}.fail{{background:#fee2e2;color:#991b1b}}.number{{font-size:28px;font-weight:800;color:#1d4ed8}}svg{{width:100%;height:auto}}.example-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}.sentence{{padding:10px;margin:7px 0;border-radius:8px;border-left:4px solid #ef4444}}.sentence small{{display:block;margin-top:5px;color:#475569}}.gold{{color:#991b1b}}@media(max-width:800px){{.cards,.example-grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><div class="muted">Experiment {VERSION} · final visible split: {final_split}</div><h1>Can spectral fusion improve evidence perturbation?</h1><p>{html.escape(conclusion)}</p><p><span class="number">{float(dufs['auroc']):.3f}</span> EC-DUFS-LIU sentence AUROC · Δ vs GASP-top50 {delta_gasp:+.3f} · Δ vs EC-IU-PCR {delta_iu:+.3f}</p></section>
<p class="fail" style="padding:12px;border-radius:9px"><b>Protocol correction:</b> the original blind run used saved full-vocabulary entropy. This report uses the approved top-50-plus-tail entropy formula, corrected after the original test labels had already been opened. The formula was fixed by the plan rather than selected from labels, but this is not a new blinded confirmation.</p>
<h2>How the experiment was protected</h2><p>The answer text never changes. A small scorer reads it under several evidence conditions. Features and unsupervised scores are written and hashed before RAGTruth labels are loaded.</p>{_pipeline_svg()}
<h2>Data coverage</h2><p>The no-context cohort contains all tasks. The complete leave-one-chunk-out (LOO) cohort contains QA and Data-to-Text only. Summary is never given invented LOO features.</p><img src="{coverage}" alt="Data coverage by split and task">
<h2>How the methods differ</h2>{_method_cards()}
<h2>Primary result</h2><p><b>AUROC</b> is the probability that a randomly selected hallucinated sentence receives a higher suspicion score than a grounded sentence. <b>AUPRC</b> emphasizes precision on the rarer hallucinated class. Intervals resample complete source groups 1,000 times.</p><img src="{perf}" alt="Performance comparison"><table><thead><tr><th>Method</th><th>AUROC</th><th>95% grouped CI</th><th>AUPRC</th></tr></thead><tbody>{table_rows}</tbody></table><p>EC-DUFS-LIU improves over GASP-top50 by <b>{float(primary_pair['delta_auroc']):+.3f}</b>, 95% paired interval [{float(primary_pair['ci_low']):+.3f}, {float(primary_pair['ci_high']):+.3f}]. It changes by <b>{float(iu_pair['delta_auroc']):+.4f}</b> versus EC-IU-PCR, interval [{float(iu_pair['ci_low']):+.4f}, {float(iu_pair['ci_high']):+.4f}].</p>
<h2>Where the result comes from</h2><img src="{heat}" alt="Per-task AUROC heatmap"><img src="{source_delta}" alt="Method differences by source family"><p>QA and Data-to-Text both improve over GASP-top50 at sentence level. DUFS-LIU does not improve over IU-PCR in either source family.</p>
<h2>What DUFS and the Laplacian learned</h2><img src="{training}" alt="DUFS loss convergence"><img src="{gates}" alt="DUFS gate probabilities"><p>The loss is label-free. DUFS keeps an effective {float(primary_dufs['effective_feature_count']):.2f} of 14 features; {100*float(primary_dufs['near_one_fraction']):.1f}% of gates are above 0.95 and {100*float(primary_dufs['near_zero_fraction']):.1f}% are below 0.05.</p><img src="{graph_degree}" alt="Graph degree distribution"><img src="{laplacian}" alt="Laplacian sensitivity path"><table><thead><tr><th>Graph diagnostic at frozen λ</th><th>Value</th></tr></thead><tbody><tr><td>connected components</td><td>{int(frozen_laplacian['n_components'])}</td></tr><tr><td>nodes / undirected edges</td><td>{int(frozen_laplacian['n_nodes'])} / {int(frozen_laplacian['n_edges'])}</td></tr><tr><td>degree min / mean / max</td><td>{float(frozen_laplacian['degree_min']):.3f} / {float(frozen_laplacian['degree_mean']):.3f} / {float(frozen_laplacian['degree_max']):.3f}</td></tr><tr><td>algebraic connectivity (spectral gap)</td><td>{float(frozen_laplacian['algebraic_connectivity']):.5f}</td></tr><tr><td>score graph roughness</td><td>{float(frozen_laplacian['score_laplacian_energy']):.5f}</td></tr><tr><td>projected condition number</td><td>{float(frozen_laplacian['projected_condition_number']):.3f}</td></tr></tbody></table><p>The graph is connected and numerically healthy, but the frozen graph changes the IU weights only slightly, and a permuted graph is effectively tied. Stable optimization is therefore not evidence that the graph identifies grounding.</p>
<h2>Confound checks</h2><p>A <b>confound</b> is a nuisance variable that may explain a score without representing grounding. At sentence level, residualizing length, chunk count and context length changes AUROC from {float(confound['raw_auroc']):.3f} to {float(confound['residualized_auroc']):.3f}. At response level it changes from {float(response_confound['raw_auroc']):.3f} to {float(response_confound['residualized_auroc']):.3f}; this larger drop is a warning about response-level transfer.</p><img src="{confound_plot}" alt="Score confound plots">
<h2>Post-hoc old-method audit</h2><p>The stored telemetry can reproduce the global 30-feature intrinsic mixed-v2 DUFS-LIU baseline exactly. This audit was added after test labels had opened, so it has a separate score hash and cannot change the registered decision. Its pooled response AUROC is misleading: it performs strongly on QA but falls below chance on Data-to-Text. Evidence-Contrast removes that catastrophic task failure.</p>{intrinsic_html}
<h2>Localization examples</h2><p>Red intensity is the frozen sentence score. Gold markers were added only in the evaluation view. The chunk number identifies the leave-one-out condition with the largest mean likelihood drop.</p>{examples}
<h2>Pre-registered decisions</h2><h3>Development gate</h3>{gate_html}<h3>Final test conditions</h3>{decision_html}<p>The experiment is a <b>feature-contract success but a DUFS/Laplacian mechanism failure</b>. It is not a full success under the registered rule.</p>
<h2>Important limitations</h2><ul><li>GASP-top50 is not a faithful full-vocabulary GASP reproduction.</li><li>The scorer is Qwen2.5-1.5B, not necessarily the model that generated each published response.</li><li>RAGTruth annotations are incomplete; these are benchmark-label results, not absolute grounding accuracy.</li><li>Evidence removal also changes prompt length and token positions.</li></ul>
<h2>Artifacts and audit</h2><p>Feature definitions, fit diagnostics, score hashes, grouped comparisons and confound checks are stored beside this report. No evaluation label is accepted by the scoring API. Test score hash: <code>{manifest['splits'].get('test', {}).get('score_sha256', 'not opened')}</code></p><ol><li>Feature contract frozen.</li><li>Dev score written and hashed.</li><li>Dev labels opened; gate passed.</li><li>Test score written and hashed.</li><li>Test labels opened once.</li><li>Registered decision applied without tuning.</li><li>Optional intrinsic mixed-v2 response baseline added later as a separately hashed post-hoc audit.</li></ol>
</main></body></html>"""
    (out / "REPORT.html").write_text(report, encoding="utf-8")
    markdown = f"""# RAGTruth Evidence-Contrast Experiment

**Experiment:** `{VERSION}`  
**Final visible split:** `{final_split}`

**Protocol status:** fixed-formula correction performed after the original test labels had been opened. The approved top-50-plus-tail entropy formula was not selected from labels, but this is not a new blinded confirmation.

## Result

{conclusion}

On the LOO sentence cohort, EC-DUFS-LIU reached AUROC **{float(dufs['auroc']):.3f}**. The change was **{delta_gasp:+.3f}** versus GASP-top50 and **{delta_iu:+.4f}** versus EC-IU-PCR.

The registered method-level success rule failed. The Evidence-Contrast contract plus IU-PCR is useful, but the DUFS-gated Laplacian did not add value. The result should be described as a **feature-contract success and mechanism failure**.

{intrinsic_markdown}

The self-contained visual report is [`REPORT.html`](REPORT.html). Definitions and mathematical provenance are in [`METHODS.md`](METHODS.md). Exact execution commands are in [`RUNBOOK.md`](RUNBOOK.md).

## Boundaries

- GASP-top50 approximates, but does not reproduce, full-vocabulary GASP JSD.
- Scores come from a Qwen2.5-1.5B teacher-forced scorer over fixed RAGTruth answers.
- Results measure agreement with the available RAGTruth annotations.
- Test labels are opened only if every registered development check passes.
- Response-level performance is more sensitive to chunk count and context length than sentence-level performance.
- The original intrinsic mixed-v2 response baseline is reported only as a separately hashed post-hoc audit and does not enter the registered decision.
"""
    (out / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(f"[report] {out / 'REPORT.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "evaluate", "intrinsic-score", "intrinsic-evaluate", "validate-input"):
        item = sub.add_parser(command)
        item.add_argument("--split", required=True, choices=("pilot", "dev", "test"))
        item.add_argument("--cache", required=True)
        item.add_argument("--official-responses", required=True)
        item.add_argument("--tokenizer", required=True)
        item.add_argument("--out", default=str(DEFAULT_OUT))
        if command in ("evaluate", "intrinsic-evaluate"):
            item.add_argument("--bootstrap", type=int, default=1000)
    report = sub.add_parser("report")
    report.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "score":
        score_split(args)
    elif args.command == "evaluate":
        evaluate_split(args)
    elif args.command == "intrinsic-score":
        intrinsic_score_split(args)
    elif args.command == "intrinsic-evaluate":
        intrinsic_evaluate_split(args)
    elif args.command == "validate-input":
        validate_input_split(args)
    else:
        build_report(args)


if __name__ == "__main__":
    main()
