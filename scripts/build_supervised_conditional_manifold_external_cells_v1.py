#!/usr/bin/env python3
"""Build frozen, label-independent feature matrices for external manifold audit.

The output matrices use the same per-cell orientation and unlabeled z-scoring
contract as the internal discovery bundle.  Labels are carried only after the
feature matrix has been constructed.  Repeated-generation caches are reduced
to the first stored candidate per source question, a target-blind rule that
restores independent rows for the registered external validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.feature_contract import confidence_oriented_matrix  # noqa: E402
from spectral_utils.fair_comparisons.stopping import grade_aqua_option  # noqa: E402
from spectral_utils.repgrid_scoring import (  # noqa: E402
    _candidate_features,
    logprob_features_extended,
)
from spectral_utils.specrage_views import zscore_columns  # noqa: E402


VERSION = "supervised-conditional-manifold-external-cells-v1-2026-08-20"
DEFAULT_CANDIDATE = (
    ROOT / "results/supervised_conditional_manifold_discovery_v1/FROZEN_CANDIDATE.json"
)
DEFAULT_HLE = ROOT / "dataset_cache/four_localization/hle_full/raw_hle_T0.0.pkl"
DEFAULT_HLE_JUDGE = ROOT / "local_cache/data_readiness/hle_codex_5p6_sol_xhigh.jsonl"
DEFAULT_COQA = (
    ROOT / "local_cache/external_manifold_validation_v1/coqa/raw_coqa_T0.5.pkl"
)
DEFAULT_AQUA_ROOT = ROOT / "local_cache/fair_paper_exact_comparisons_v1"
DEFAULT_OUT = ROOT / "local_cache/external_manifold_validation_v1/matrices"
DEFAULT_MANIFEST = (
    ROOT / "configs/supervised_conditional_manifold_external_validation_v1.json"
)

AQUA_RUNS = {
    "aqua_qwen25_7b": (
        "s2_leash_Qwen2.5-7B-Instruct_aqua",
        "qwen2.5-7b",
        False,
    ),
    "aqua_llama31_8b": (
        "s2_leash_Llama-3.1-8B-Instruct_aqua",
        "llama3.1-8b",
        False,
    ),
    "aqua_phi3mini": (
        "s2_leash_Phi-3-mini-128k-instruct_aqua",
        "phi3-mini",
        False,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_features(candidate: dict) -> dict[str, float]:
    # Harmonize the distribution summaries on retained raw top-50 probabilities.
    # Several sampling caches contain -inf padding in the post-warper top-k array,
    # while the raw array is finite and is available in every registered cell.
    topk = candidate.get("top_k_logprobs_raw", candidate.get("top_k_logprobs"))
    normalized = dict(candidate)
    normalized["top_k_logprobs"] = topk
    features = dict(_candidate_features(normalized))
    if topk is not None:
        features.update(logprob_features_extended(topk))
    return features


def make_matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    raw = np.asarray([[row.get(name, np.nan) for name in names] for row in rows], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise RuntimeError("external feature matrix has the wrong shape")
    if not np.isfinite(raw).all():
        bad = np.argwhere(~np.isfinite(raw))[:10].tolist()
        raise RuntimeError(f"external feature matrix contains non-finite values at {bad}")
    oriented, kept, _ = confidence_oriented_matrix(raw, names)
    if tuple(kept) != names:
        raise RuntimeError("external feature order changed during orientation")
    return zscore_columns(oriented)


def write_npz(
    path: Path,
    *,
    rows: list[dict],
    feature_names: tuple[str, ...],
    trace_length: np.ndarray,
    target: np.ndarray,
    row_id: np.ndarray,
) -> None:
    matrix = make_matrix(rows, feature_names)
    target = np.asarray(target, dtype=np.int8)
    trace_length = np.asarray(trace_length, dtype=float)
    row_id = np.asarray(row_id, dtype=str)
    if len(matrix) != len(target) or trace_length.shape != target.shape:
        raise RuntimeError("external rows, labels, and lengths disagree")
    if not np.isin(target, (0, 1)).all() or len(np.unique(target)) != 2:
        raise RuntimeError("external target must contain both classes")
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        X=matrix,
        feature_names=np.asarray(feature_names),
        trace_length=trace_length,
        hallucination_target=target,
        row_id=row_id,
    )
    os.replace(temporary, path)


def load_repgrid_first(
    path: Path, names: tuple[str, ...]
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    rows: list[dict] = []
    targets, lengths, identifiers = [], [], []
    for key in sorted(data):
        candidates = data[key]["candidates"]
        if not candidates:
            continue
        selected = candidates[0]
        features = candidate_features(selected)
        if not all(np.isfinite(features.get(name, np.nan)) for name in names):
            continue
        rows.append(features)
        targets.append(1 - int(bool(selected.get("label", False))))
        lengths.append(len(selected.get("token_entropies", ()) or ()))
        identifiers.append(str(key))
    return rows, np.asarray(targets), np.asarray(lengths), np.asarray(identifiers)


def load_hle(
    raw_path: Path, judge_path: Path
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    labels = {}
    with judge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = str(record["correct"]).strip().lower()
            if value not in {"yes", "no"}:
                raise RuntimeError(f"unrecognized HLE judge label: {value}")
            labels[int(record["row_key"])] = int(value == "no")
    with raw_path.open("rb") as handle:
        data = pickle.load(handle)
    if set(labels) != set(map(int, data)):
        raise RuntimeError("HLE judge roster does not match the raw cache")
    rows: list[dict] = []
    targets, lengths, identifiers = [], [], []
    for key in sorted(data):
        candidates = data[key]["candidates"]
        if len(candidates) != 1:
            raise RuntimeError("HLE external cell must contain one response per question")
        selected = candidates[0]
        rows.append(candidate_features(selected))
        targets.append(labels[int(key)])
        lengths.append(len(selected.get("token_entropies", ()) or ()))
        identifiers.append(str(key))
    return rows, np.asarray(targets), np.asarray(lengths), np.asarray(identifiers)


def load_aqua(run_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    records = []
    for shard in sorted((run_dir / "shards").glob("shard_*.pkl")):
        with shard.open("rb") as handle:
            records.extend(pickle.load(handle))
    records = [
        row for row in records
        if row.get("arm") == "cot" and row.get("setting_label") == "central"
    ]
    records.sort(key=lambda row: int(str(row["question_id"]).split(":")[-1]))
    identifiers = [str(row["question_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"duplicate AQuA question IDs in {run_dir}")
    rows: list[dict] = []
    targets, lengths = [], []
    for record in records:
        channels = record["channels"]
        topk = record["raw_top_k_logprobs"]
        logprob = np.asarray(topk["logprobs"], dtype=float)
        top15 = logprob[:, : min(15, logprob.shape[1])]
        probability = np.exp(top15 - np.max(top15, axis=1, keepdims=True))
        probability /= np.sum(probability, axis=1, keepdims=True)
        entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1)
        candidate = {
            # The paper-exact greedy trace retained raw top-50 log-probabilities.
            # Recompute the project's registered normalized top-15 entropy from
            # those values because the live sampled_entropy channel is NaN for
            # the greedy arm (0 * -inf in the historical telemetry path).
            "token_entropies": entropy,
            "token_spilled_energies": channels["spilled_energy"],
            "token_logsumexp": channels["raw_logsumexp"],
            "top_k_logprobs": topk,
            "top_k_logprobs_raw": topk,
        }
        rows.append(candidate_features(candidate))
        rescored = grade_aqua_option(record.get("answer_text"), record.get("gold_answer"))
        targets.append(1 - int(bool(rescored["correct"])))
        lengths.append(len(entropy))
    return rows, np.asarray(targets), np.asarray(lengths), np.asarray(identifiers)


def cell_spec(
    *, cell: str, dataset: str, model: str, model_new: bool, path: Path, n: int,
    positives: int, source_paths: list[Path], derivation: str,
) -> dict:
    return {
        "cell": cell,
        "lane": "global_final_answer_error",
        "dataset_family": dataset,
        "model_family": model,
        "dataset_new": True,
        "model_new": bool(model_new),
        "npz": str(path.relative_to(ROOT)),
        "independent_rows": True,
        "n": int(n),
        "n_hallucination": int(positives),
        "n_correct": int(n - positives),
        "source_paths": [str(value.relative_to(ROOT)) for value in source_paths],
        "source_sha256": {str(value.relative_to(ROOT)): sha256(value) for value in source_paths},
        "derivation": derivation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--hle", type=Path, default=DEFAULT_HLE)
    parser.add_argument("--hle-judge", type=Path, default=DEFAULT_HLE_JUDGE)
    parser.add_argument("--coqa", type=Path, default=DEFAULT_COQA)
    parser.add_argument("--aqua-root", type=Path, default=DEFAULT_AQUA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    feature_names = tuple(map(str, candidate["feature_names"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = []

    for cell, (run_name, model, model_new) in AQUA_RUNS.items():
        run_dir = args.aqua_root / run_name
        rows, target, length, row_id = load_aqua(run_dir)
        path = args.out_dir / f"{cell}.npz"
        write_npz(path, rows=rows, feature_names=feature_names, trace_length=length,
                  target=target, row_id=row_id)
        source_paths = [run_dir / "RUN_MANIFEST.json", *sorted((run_dir / "shards").glob("shard_*.pkl"))]
        specs.append(cell_spec(
            cell=cell, dataset="aqua", model=model, model_new=model_new, path=path,
            n=len(target), positives=int(target.sum()), source_paths=source_paths,
            derivation=(
                "paper-exact cot|central rows; one response per AQuA question; "
                "labels replayed with fair_aqua_option_parser_v1.0.0"
            ),
        ))

    rows, target, length, row_id = load_hle(args.hle, args.hle_judge)
    path = args.out_dir / "hle_qwen25_72b.npz"
    write_npz(path, rows=rows, feature_names=feature_names, trace_length=length,
              target=target, row_id=row_id)
    specs.append(cell_spec(
        cell="hle_qwen25_72b", dataset="hle", model="qwen2.5-72b", model_new=True,
        path=path, n=len(target), positives=int(target.sum()),
        source_paths=[args.hle, args.hle_judge],
        derivation="one response per HLE question; interim official-prompt judge labels",
    ))

    rows, target, length, row_id = load_repgrid_first(args.coqa, feature_names)
    path = args.out_dir / "coqa_llama7b_first.npz"
    write_npz(path, rows=rows, feature_names=feature_names, trace_length=length,
              target=target, row_id=row_id)
    specs.append(cell_spec(
        cell="coqa_llama7b_first", dataset="coqa", model="llama1-7b", model_new=True,
        path=path, n=len(target), positives=int(target.sum()), source_paths=[args.coqa],
        derivation=(
            "first stored generation per source question; rows lacking the frozen feature "
            "contract are dropped without labels"
        ),
    ))

    manifest = {
        "version": "supervised-conditional-manifold-external-validation-manifest-v2",
        "builder_version": VERSION,
        "validation_name": "retrospective_external_to_discovery_aqua_hle_coqa_v1",
        "claim_status": "retrospective_external_to_discovery_not_prospective_confirmation",
        "standardization_contract": "confidence_oriented_per_cell_unlabeled_zscore",
        "candidate_sha256": sha256(args.candidate),
        "minimum_independent_dataset_families": 3,
        "cells": specs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(f"wrote {len(specs)} cells across {len(set(row['dataset_family'] for row in specs))} families")


if __name__ == "__main__":
    main()
