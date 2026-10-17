#!/usr/bin/env python3
"""Compare global/model/dataset/cell covariance scopes for token L-SML."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_claude_feature_bank_v1 import (  # noqa: E402
    evaluate,
    load_roster,
    load_source_rows,
    step_top10,
    write_json,
)
from spectral_utils.claude_feature_bank_v1 import (  # noqa: E402
    FEATURE_NAMES,
    build_loco5_answer_features,
    build_token_feature_matrix,
    deterministic_token_sample,
    fit_l_sml_weights,
    fit_token_standardizer,
    fuse_token_matrix,
    loco5_gate,
)


SCHEMA = "claude-feature-bank-pooling-scope-v1"
OUTPUT = ROOT / "results" / "claude_feature_bank_pooling_scope_v1"
GLOBAL = ROOT / "results" / "claude_feature_bank_token_lsml_v1"
SCOPES = ("global", "model", "dataset", "cell")


def model_key(cell: str) -> str:
    if cell.startswith("prmbench_") or cell.endswith("_q8"):
        return "qwen3_8b"
    if cell.endswith("_q4"):
        return "qwen3_4b"
    raise ValueError(f"unrecognized cell {cell!r}")


def dataset_key(cell: str) -> str:
    if cell.startswith("prmbench_"):
        return "prmbench"
    if not cell.startswith("pb_"):
        raise ValueError(f"unrecognized cell {cell!r}")
    body = cell[3:]
    return body.rsplit("_q", 1)[0]


def scope_keys(cells: np.ndarray, scope: str) -> np.ndarray:
    cells = np.asarray(cells, dtype=str)
    if scope == "global":
        return np.full(len(cells), "all", dtype="U16")
    if scope == "model":
        return np.asarray([model_key(cell) for cell in cells])
    if scope == "dataset":
        return np.asarray([dataset_key(cell) for cell in cells])
    if scope == "cell":
        return cells.copy()
    raise ValueError(f"unknown scope {scope!r}")


def covariance_diagnostics(
    matrices: Sequence[np.ndarray],
    indexes: np.ndarray,
    standardizer: Mapping[str, Any],
) -> dict[str, Any]:
    sample = deterministic_token_sample(
        (matrices[int(index)] for index in indexes),
        cap=int(standardizer["sample_count"]),
    )
    mean = np.asarray(standardizer["mean"], dtype=float)
    std = np.asarray(standardizer["std"], dtype=float)
    standardized = (sample - mean) / std
    covariance = np.cov(standardized.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    denominator = float(np.sum(eigenvalues ** 2))
    participation = (
        float(np.sum(eigenvalues) ** 2 / denominator) if denominator > 1e-12 else 0.0
    )
    condition = float(eigenvalues[-1] / max(eigenvalues[0], 1e-12))
    return {
        "covariance_participation_ratio": participation,
        "covariance_eigenvalues": eigenvalues,
        "covariance_condition_number": condition,
    }


def weight_participation(weights: np.ndarray) -> float:
    normalized = np.abs(np.asarray(weights, dtype=float))
    normalized /= normalized.sum()
    return float(1.0 / np.sum(normalized ** 2))


def fit_scoped_oof(
    scope: str,
    matrices: Sequence[np.ndarray],
    spans: Sequence[np.ndarray],
    cells: np.ndarray,
    folds: np.ndarray,
    *,
    token_cap: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    keys = scope_keys(cells, scope)
    offsets = np.concatenate(([0], np.cumsum([len(item) for item in spans], dtype=int)))
    l_sml_steps = np.full(int(offsets[-1]), np.nan, dtype=float)
    equal_steps = np.full_like(l_sml_steps, np.nan)
    fit_records: list[dict[str, Any]] = []
    for key in sorted(np.unique(keys)):
        local = keys == key
        for fold in np.unique(folds):
            train = np.flatnonzero(local & (folds != fold))
            test = np.flatnonzero(local & (folds == fold))
            if not len(train) or not len(test):
                raise ValueError(f"scope={scope} key={key} fold={fold} lacks train/test rows")
            standardizer = fit_token_standardizer(
                (matrices[int(index)] for index in train), cap=token_cap
            )
            weights, meta = fit_l_sml_weights(
                (matrices[int(index)] for index in train), standardizer
            )
            diagnostics = covariance_diagnostics(matrices, train, standardizer)
            for answer in test:
                l_sml, equal = fuse_token_matrix(
                    matrices[int(answer)], standardizer, weights=weights
                )
                start, stop = int(offsets[answer]), int(offsets[answer + 1])
                l_sml_steps[start:stop] = step_top10(l_sml, spans[int(answer)])
                equal_steps[start:stop] = step_top10(equal, spans[int(answer)])
            fit_records.append({
                "scope": scope,
                "scope_key": str(key),
                "fold": str(fold),
                "train_answers": int(len(train)),
                "test_answers": int(len(test)),
                "token_sample_count": int(standardizer["sample_count"]),
                "means": np.asarray(standardizer["mean"]),
                "stds": np.asarray(standardizer["std"]),
                "weights": weights,
                "weight_participation_ratio": weight_participation(weights),
                **diagnostics,
                "lsml_meta": {
                    "K": int(meta["K"]),
                    "groups": np.asarray(meta["c"]),
                    "residual": float(meta["residual"]),
                    "degenerate": bool(meta.get("degenerate", False)),
                },
            })
            print(
                f"[fit] scope={scope} key={key} fold={fold} "
                f"train={len(train)} test={len(test)} "
                f"cov_pr={diagnostics['covariance_participation_ratio']:.4f}",
                flush=True,
            )
    if not np.isfinite(l_sml_steps).all() or not np.isfinite(equal_steps).all():
        raise ValueError(f"scope {scope} left non-finite OOF scores")
    return {"l_sml": l_sml_steps, "equal": equal_steps}, fit_records


def global_reference(
    matrices: Sequence[np.ndarray],
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    result_path = GLOBAL / "RESULTS.json"
    scores_path = GLOBAL / "OOF_SCORES.npz"
    if not result_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("frozen global baseline is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("global feature contract changed")
    with np.load(scores_path, allow_pickle=False) as loaded:
        scores = {"l_sml": loaded["l_sml"].copy(), "equal": loaded["equal"].copy()}
        gate_open = loaded["gate_open"].copy()
    fits = []
    for old in result["fits"]:
        fold = int(old["fold"])
        train = np.flatnonzero(folds != fold)
        standardizer = {
            "mean": np.asarray(old["means"], dtype=float),
            "std": np.asarray(old["stds"], dtype=float),
            "sample_count": int(old["token_sample_count"]),
        }
        weights = np.asarray(old["weights"], dtype=float)
        fits.append({
            "scope": "global",
            "scope_key": "all",
            "fold": str(fold),
            "train_answers": int(old["train_answers"]),
            "test_answers": int(old["test_answers"]),
            "token_sample_count": int(old["token_sample_count"]),
            "means": standardizer["mean"],
            "stds": standardizer["std"],
            "weights": weights,
            "weight_participation_ratio": weight_participation(weights),
            **covariance_diagnostics(matrices, train, standardizer),
            "lsml_meta": old["lsml_meta"],
        })
    return scores, fits, {"result": result, "gate_open": gate_open}


def summarize(scope: str, fits: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    local = [fit for fit in fits if fit["scope"] == scope]
    covariance_pr = np.asarray([fit["covariance_participation_ratio"] for fit in local])
    weight_pr = np.asarray([fit["weight_participation_ratio"] for fit in local])
    k_values = [int(fit["lsml_meta"]["K"]) for fit in local]
    l_sml = next(row for row in metrics if row["name"] == f"{scope}_l_sml")
    equal = next(row for row in metrics if row["name"] == f"{scope}_equal")
    return {
        "scope": scope,
        "fits": len(local),
        "scope_keys": sorted({str(fit["scope_key"]) for fit in local}),
        "covariance_participation_ratio": {
            "mean": float(covariance_pr.mean()),
            "min": float(covariance_pr.min()),
            "max": float(covariance_pr.max()),
        },
        "weight_participation_ratio": {
            "mean": float(weight_pr.mean()),
            "min": float(weight_pr.min()),
            "max": float(weight_pr.max()),
        },
        "K_counts": {str(key): int(value) for key, value in sorted(Counter(k_values).items())},
        "degenerate_fits": int(sum(bool(fit["lsml_meta"]["degenerate"]) for fit in local)),
        "pb_macro_f1": float(l_sml["pb_macro_f1"]),
        "pb_equal_macro_f1": float(equal["pb_macro_f1"]),
        "pb_l_sml_minus_equal": float(l_sml["pb_macro_f1"] - equal["pb_macro_f1"]),
        "within_answer_auc": float(l_sml["within_answer_auc"]),
        "within_equal_auc": float(equal["within_answer_auc"]),
        "within_l_sml_minus_equal": float(l_sml["within_answer_auc"] - equal["within_answer_auc"]),
    }


def extract(records: Sequence[Mapping[str, Any]], rows: Sequence[dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, float]]]:
    matrices: list[np.ndarray] = []
    spans: list[np.ndarray] = []
    gate_features: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        matrix = build_token_feature_matrix(row)
        if len(matrix) != int(records[index]["tokens"]):
            raise ValueError(f"token mismatch at {records[index]['uid']}")
        step_spans = np.asarray(row["step_token_spans"], dtype=int)
        if step_spans.shape != (int(records[index]["steps"]), 2):
            raise ValueError(f"step span mismatch at {records[index]['uid']}")
        matrices.append(matrix)
        spans.append(step_spans)
        gate_features.append(build_loco5_answer_features(row))
        if (index + 1) % 500 == 0:
            print(f"[extract] {index + 1}/{len(rows)}", flush=True)
    return matrices, spans, gate_features


def run(*, token_cap: int, smoke_only: bool) -> Path:
    records, data = load_roster()
    cells = np.asarray(data["cells"], dtype=str)
    folds = np.asarray(data["folds"])
    scope_inventory = {
        scope: {
            str(key): int(np.sum(scope_keys(cells, scope) == key))
            for key in np.unique(scope_keys(cells, scope))
        }
        for scope in SCOPES
    }
    if smoke_only:
        path = OUTPUT / "SMOKE.json"
        write_json(path, {
            "schema": SCHEMA + "/smoke",
            "quality_result": False,
            "answers": len(records),
            "scope_inventory": scope_inventory,
            "global_baseline_exists": (GLOBAL / "RESULTS.json").is_file(),
        })
        return path

    rows = load_source_rows(records)
    matrices, spans, gate_features = extract(records, rows)
    gate_score, gate_open, gate_diag = loco5_gate(gate_features, cells)
    global_scores, global_fits, global_meta = global_reference(matrices, folds)
    if not np.array_equal(gate_open, global_meta["gate_open"]):
        raise ValueError("LOCO-5 gate drift versus frozen global experiment")

    scores_by_scope: dict[str, dict[str, np.ndarray]] = {"global": global_scores}
    fits = list(global_fits)
    for scope in ("model", "dataset", "cell"):
        scores, local_fits = fit_scoped_oof(
            scope, matrices, spans, cells, folds, token_cap=token_cap
        )
        scores_by_scope[scope] = scores
        fits.extend(local_fits)

    metrics = []
    for scope in SCOPES:
        metrics.append(evaluate(f"{scope}_l_sml", scores_by_scope[scope]["l_sml"], gate_open, data))
        metrics.append(evaluate(f"{scope}_equal", scores_by_scope[scope]["equal"], gate_open, data))
    old_metrics = {row["name"]: row for row in global_meta["result"]["metrics"]}
    for new_name, old_name in (("global_l_sml", "token_l_sml"), ("global_equal", "token_equal_mean")):
        new = next(row for row in metrics if row["name"] == new_name)
        old = old_metrics[old_name]
        for key in ("pb_macro_f1", "pb_q4_macro_f1", "pb_q8_macro_f1", "within_answer_auc"):
            np.testing.assert_allclose(new[key], old[key], atol=1e-14, rtol=0)

    result = {
        "schema": SCHEMA,
        "development_only": True,
        "quality_result": True,
        "feature_names": list(FEATURE_NAMES),
        "scopes": list(SCOPES),
        "scope_inventory": scope_inventory,
        "contract": {
            "source_folds": 5,
            "token_cap_per_fit": token_cap,
            "same_loco5_gate": True,
            "same_top10_step_locator": True,
            "labels_used_in_fit": False,
            "global_is_frozen_existing_experiment": True,
            "pr_definition": "trace(C)^2 / trace(C^2) on donor-standardized token covariance",
            "weight_participation_definition": "1 / sum((abs(w)/L1(w))^2)",
            "q8_model_scope_includes_prmbench": True,
        },
        "gate": gate_diag,
        "summary": [summarize(scope, fits, metrics) for scope in SCOPES],
        "metrics": metrics,
        "fits": fits,
    }
    write_json(OUTPUT / "RESULTS.json", result)
    arrays = {"gate_score": gate_score, "gate_open": gate_open}
    for scope, scores in scores_by_scope.items():
        arrays[f"{scope}__l_sml"] = scores["l_sml"]
        arrays[f"{scope}__equal"] = scores["equal"]
    np.savez_compressed(OUTPUT / "OOF_SCORES.npz", **arrays)
    return OUTPUT / "RESULTS.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--token-cap", type=int, default=60_000)
    args = parser.parse_args(argv)
    print(run(token_cap=args.token_cap, smoke_only=args.smoke))


if __name__ == "__main__":
    main()
