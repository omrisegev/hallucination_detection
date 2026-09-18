#!/usr/bin/env python3
"""Run the development-only token-level Claude feature-bank experiment.

The runner reads the frozen localization roster and the existing raw telemetry,
fits token standardization and L-SML weights source-fold-out-of-fold, and then
compares L-SML with a simple mean under the same fixed LOCO-5 gate.  It does
not fit a gate, use labels in the fusion fit, or create a second sampled
answer.  A ``--smoke`` run extracts only a few answers and never writes a
quality result.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import numpy as np
    from scipy.stats import rankdata
except ModuleNotFoundError as error:  # pragma: no cover - environment diagnostic
    if error.name == "numpy":
        raise SystemExit(
            "NumPy is required for the numerical run; no experiment result was written."
        ) from None
    raise

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / ".worktrees" / "fusion-independence-atlas-v1"
FROZEN = ATLAS / "results" / "localization_full_benchmark_v3"
OUTPUT = ROOT / "results" / "claude_feature_bank_token_lsml_v1"
SCHEMA = "claude-feature-bank-token-lsml-v1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.claude_feature_bank_v1 import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_RISK_SIGNS,
    LOCO5_NAMES,
    build_loco5_answer_features,
    build_token_feature_matrix,
    fit_l_sml_weights,
    fit_token_standardizer,
    fuse_token_matrix,
    loco5_gate,
)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pb_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    cells: np.ndarray,
) -> dict[str, Any]:
    """Minimal frozen ProcessBench macro-F1 adapter for this runner."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    valid = np.asarray(valid, dtype=bool)
    cells = np.asarray(cells, dtype=str)
    results = {}
    correct = valid & (prediction == target)
    for cell in sorted(set(cells)):
        if not cell.startswith("pb_"):
            continue
        mask = cells == cell
        clean = mask & (target == -1)
        erroneous = mask & (target >= 0)
        clean_count = int(clean.sum())
        error_count = int(erroneous.sum())
        clean_accuracy = float(correct[clean].sum() / clean_count) if clean_count else None
        error_accuracy = float(correct[erroneous].sum() / error_count) if error_count else None
        f1 = (
            None
            if clean_accuracy is None or error_accuracy is None
            else float(2 * clean_accuracy * error_accuracy / (clean_accuracy + error_accuracy))
            if clean_accuracy + error_accuracy
            else 0.0
        )
        results[cell] = {
            "answers": int(mask.sum()),
            "clean": clean_count,
            "erroneous": error_count,
            "clean_accuracy": clean_accuracy,
            "error_exact_accuracy": error_accuracy,
            "f1": f1,
            "valid_decisions": int(valid[mask].sum()),
        }
    macros = {}
    for panel in ("q4", "q8", "all"):
        values = [
            item["f1"] for cell, item in results.items()
            if panel == "all" or cell.endswith(panel)
        ]
        macros[panel] = float(np.mean(values)) if values and None not in values else None
    return {"cells": results, "macros": macros}


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen experiment input(s):\n" + "\n".join(missing))


def load_roster() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Load the exact answer/step roster used by the atlas evaluation."""
    joined_json = FROZEN / "evaluation" / "JOINED.json"
    joined_npz = FROZEN / "evaluation" / "JOINED.npz"
    evaluation_npz = ATLAS / "results" / "fusion_independence_atlas_v1" / "dependence" / "EVALUATION.npz"
    _require_files((joined_json, joined_npz, evaluation_npz))
    records = json.loads(joined_json.read_text(encoding="utf-8"))["records"]
    with np.load(joined_npz, allow_pickle=False) as loaded:
        joined = {name: loaded[name].copy() for name in loaded.files}
    with np.load(evaluation_npz, allow_pickle=False) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    required = {"offsets", "target", "labels", "cells", "groups", "folds"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"frozen evaluation roster lacks {missing}")
    if len(records) != len(data["offsets"]) - 1:
        raise ValueError("JOINED.json and JOINED.npz answer counts disagree")
    for name in ("offsets", "target", "labels"):
        if name not in joined or not np.array_equal(joined[name], data[name]):
            raise ValueError(f"JOINED and atlas evaluation disagree on {name}")
    return records, data


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_specs() -> list[tuple[str, Path, str, str | None]]:
    pb_dirs = {
        "q4": ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_4b",
        "q8": ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_8b",
    }
    specs = [
        (f"pb_{dataset}_{model}", directory / f"processbench_{dataset}.pkl", "pb", dataset)
        for model, directory in pb_dirs.items()
        for dataset in ("gsm8k", "math", "olympiadbench", "omnimath")
    ]
    prm = ROOT / "dataset_cache" / "four_localization" / "prmbench_qwen3_8b_telemetry_full" / "prmbench_telemetry.pkl"
    return specs + [("prmbench_qwen3_8b", prm, "prm", None)]


def source_row_map(payload: Any, *, kind: str, dataset: str | None = None) -> dict[str, dict]:
    values = payload.values() if isinstance(payload, dict) else payload
    output = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        key = f"{dataset}::{row.get('id')}" if kind == "pb" else str(row.get("idx"))
        if key in output:
            raise ValueError(f"duplicate source row id {key}")
        output[key] = row
    return output


def load_source_rows(records: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Resolve every roster row to the existing raw telemetry row."""
    rows: list[dict | None] = [None] * len(records)
    by_cell: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_cell.setdefault(str(record["cell"]), []).append(index)
    for cell, path, kind, dataset in source_specs():
        indexes = by_cell.get(cell, [])
        if not indexes:
            continue
        _require_files((path,))
        source = source_row_map(load_pickle(path), kind=kind, dataset=dataset)
        for index in indexes:
            row_id = str(records[index]["row_id"])
            if row_id not in source:
                raise KeyError(f"roster row {row_id!r} is missing from {path}")
            rows[index] = source[row_id]
        del source
        print(f"[load] {cell}: {len(indexes)} answers", flush=True)
    missing = [index for index, row in enumerate(rows) if row is None]
    if missing:
        raise ValueError(f"no raw telemetry was resolved for roster indexes {missing[:10]}")
    return [row for row in rows if row is not None]


def smoke(records: Sequence[Mapping[str, Any]], data: Mapping[str, np.ndarray], rows: Sequence[dict], count: int = 3) -> dict:
    indexes = sorted({0, len(rows) // 2, len(rows) - 1})[:count]
    checks = []
    for index in indexes:
        matrix = build_token_feature_matrix(rows[index])
        spans = np.asarray(rows[index]["step_token_spans"], dtype=int)
        if spans.ndim != 2 or spans.shape[1] != 2 or spans.shape[0] != int(records[index]["steps"]):
            raise ValueError(f"step span mismatch at answer {index}")
        gate_features = build_loco5_answer_features(rows[index])
        checks.append({
            "index": index,
            "uid": records[index]["uid"],
            "tokens": int(len(matrix)),
            "features": int(matrix.shape[1]),
            "steps": int(len(spans)),
            "loco5": gate_features,
        })
    return {
        "schema": SCHEMA + "/smoke",
        "quality_result": False,
        "checks": checks,
        "feature_names": list(FEATURE_NAMES),
        "loco5_names": list(LOCO5_NAMES),
    }


def step_top10(token_scores: np.ndarray, spans: np.ndarray) -> np.ndarray:
    result = np.full(len(spans), np.nan, dtype=float)
    for index, (start, stop) in enumerate(np.asarray(spans, dtype=int)):
        if not (0 <= start < stop <= len(token_scores)):
            raise ValueError(f"invalid step span {(start, stop)}")
        values = np.asarray(token_scores[start:stop], dtype=float)
        take = min(10, len(values))
        result[index] = np.partition(values, len(values) - take)[-take:].mean()
    return result


def fit_oof(
    matrices: Sequence[np.ndarray],
    spans: Sequence[np.ndarray],
    cells: np.ndarray,
    folds: np.ndarray,
    *,
    token_cap: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Fit standardization/L-SML on donor folds and score held-out answers."""
    l_sml_steps = np.full(sum(len(item) for item in spans), np.nan, dtype=float)
    equal_steps = np.full_like(l_sml_steps, np.nan)
    offsets = np.concatenate(([0], np.cumsum([len(item) for item in spans], dtype=int)))
    fit_records = []
    for fold in np.unique(folds):
        test_answers = np.flatnonzero(folds == fold)
        train_answers = np.flatnonzero(folds != fold)
        standardizer = fit_token_standardizer(
            (matrices[index] for index in train_answers), cap=token_cap
        )
        weights, meta = fit_l_sml_weights(
            (matrices[index] for index in train_answers), standardizer
        )
        for answer in test_answers:
            l_sml, equal = fuse_token_matrix(matrices[answer], standardizer, weights=weights)
            start, stop = int(offsets[answer]), int(offsets[answer + 1])
            l_sml_steps[start:stop] = step_top10(l_sml, spans[answer])
            equal_steps[start:stop] = step_top10(equal, spans[answer])
        fit_records.append({
            "fold": str(fold),
            "train_answers": int(len(train_answers)),
            "test_answers": int(len(test_answers)),
            "token_sample_count": int(standardizer["sample_count"]),
            "means": np.asarray(standardizer["mean"]),
            "stds": np.asarray(standardizer["std"]),
            "weights": weights,
            "effective_rank": float(1.0 / np.sum((weights / np.sum(np.abs(weights))) ** 2)),
            "lsml_meta": {
                "K": int(meta["K"]),
                "groups": np.asarray(meta["c"]),
                "residual": float(meta["residual"]),
                "degenerate": bool(meta.get("degenerate", False)),
            },
        })
        print(f"[fit] fold={fold} train={len(train_answers)} test={len(test_answers)}", flush=True)
    return {"l_sml": l_sml_steps, "equal": equal_steps}, fit_records


def evaluate(
    name: str,
    step_scores: np.ndarray,
    gate_open: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    offsets = np.asarray(data["offsets"], dtype=int)
    target = np.asarray(data["target"])
    labels = np.asarray(data["labels"])
    cells = np.asarray(data["cells"], dtype=str)
    peaks = np.empty(len(offsets) - 1, dtype=np.int32)
    within = np.full(len(peaks), np.nan, dtype=float)
    for answer in range(len(peaks)):
        local = step_scores[int(offsets[answer]):int(offsets[answer + 1])]
        if not np.isfinite(local).all():
            raise ValueError(f"non-finite locator scores at answer {answer}")
        peaks[answer] = int(np.argmax(local))
        if cells[answer].startswith("prmbench_"):
            step_labels = labels[int(offsets[answer]):int(offsets[answer + 1])]
            valid = step_labels >= 0
            if valid.any() and (step_labels[valid] == 1).any() and (step_labels[valid] == 0).any():
                within[answer] = auc(step_labels[valid] == 1, local[valid])
    pb = np.char.startswith(cells, "pb_")
    prediction = np.where(gate_open & np.isfinite(peaks), peaks, -1)
    pb_result = pb_metrics(target[pb], prediction[pb], np.ones(int(pb.sum()), dtype=bool), cells[pb])
    return {
        "name": name,
        "pb_macro_f1": pb_result["macros"]["all"],
        "pb_q4_macro_f1": pb_result["macros"]["q4"],
        "pb_q8_macro_f1": pb_result["macros"]["q8"],
        "pb_cells": pb_result["cells"],
        "within_answer_auc": float(np.nanmean(within)),
        "within_answer_n": int(np.isfinite(within).sum()),
        "gate_open_pb": int(gate_open[pb].sum()),
        "answers": int(len(peaks)),
    }


def run(*, token_cap: int, smoke_only: bool, smoke_count: int) -> Path:
    records, data = load_roster()
    rows = load_source_rows(records)
    if smoke_only:
        result = smoke(records, data, rows, count=smoke_count)
        path = OUTPUT / "SMOKE.json"
        write_json(path, result)
        return path

    matrices = []
    gate_features = []
    spans = []
    for index, row in enumerate(rows):
        matrix = build_token_feature_matrix(row)
        expected_tokens = int(records[index]["tokens"])
        if len(matrix) != expected_tokens:
            raise ValueError(f"token count mismatch at {records[index]['uid']}")
        matrices.append(matrix)
        gate_features.append(build_loco5_answer_features(row))
        step_spans = np.asarray(row["step_token_spans"], dtype=int)
        if step_spans.shape != (int(records[index]["steps"]), 2):
            raise ValueError(f"step span shape mismatch at {records[index]['uid']}")
        spans.append(step_spans)
        if (index + 1) % 500 == 0:
            print(f"[extract] {index + 1}/{len(rows)}", flush=True)

    cells = np.asarray(data["cells"], dtype=str)
    folds = np.asarray(data["folds"])
    gate_score, gate_open, gate_diag = loco5_gate(gate_features, cells)
    oof, fits = fit_oof(matrices, spans, cells, folds, token_cap=token_cap)
    metrics = [
        evaluate("token_l_sml", oof["l_sml"], gate_open, data),
        evaluate("token_equal_mean", oof["equal"], gate_open, data),
    ]
    result = {
        "schema": SCHEMA,
        "development_only": True,
        "quality_result": True,
        "feature_names": list(FEATURE_NAMES),
        "feature_risk_signs": {name: int(FEATURE_RISK_SIGNS[name]) for name in FEATURE_NAMES},
        "fusion": {
            "token_level": True,
            "one_teacher_forced_answer": True,
            "window": 16,
            "standardizer": "pooled_donor_fold_tokens",
            "token_cap": token_cap,
            "l_sml_method": "continuous_residual",
            "global_sign_gauge": "sum_of_coefficients_positive_then_first_nonzero",
            "labels_used_in_fusion_fit": False,
        },
        "gate": gate_diag,
        "fits": fits,
        "metrics": metrics,
        "source": {
            "root": str(ROOT),
            "frozen_roster": str(FROZEN),
            "answers": len(records),
            "steps": int(len(data["labels"])),
            "folds": [str(item) for item in np.unique(folds)],
        },
    }
    write_json(OUTPUT / "RESULTS.json", result)
    np.savez_compressed(
        OUTPUT / "OOF_SCORES.npz",
        l_sml=oof["l_sml"],
        equal=oof["equal"],
        gate_score=gate_score,
        gate_open=gate_open,
    )
    return OUTPUT / "RESULTS.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="extract a few answers; no quality result")
    parser.add_argument("--smoke-count", type=int, default=3)
    parser.add_argument("--token-cap", type=int, default=60_000)
    args = parser.parse_args(argv)
    path = run(token_cap=args.token_cap, smoke_only=args.smoke, smoke_count=args.smoke_count)
    print(path)


if __name__ == "__main__":
    main()
