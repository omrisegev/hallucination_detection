#!/usr/bin/env python3
"""Run registered CIW-DEEM on RAGTruth response-level original-30 features.

The source NPZ was produced by the earlier label-free RAGTruth feature pass and
contains the exact 30 response features required by CIW-DEEM.  This runner
fits and freezes every score before it opens the official response labels.
Sentence, token, span, and claim units are deliberately out of scope because
they do not have the registered CIW response input contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.ciw_deem import METHOD_ID, fit_ciw_deem  # noqa: E402
from spectral_utils.deem_b3_contract_ablation import prepare_arm  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
)
from spectral_utils.residual_graph_deem import ContinuousDeemConfig  # noqa: E402


SCHEMA = "ciw-deem-ragtruth-response-v1"
SEEDS = (0, 1, 2, 3, 4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit_split(source_path: Path, out: Path, split: str) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as source:
        raw = np.asarray(source["raw_full"], dtype=np.float64)
        names = tuple(source["original_feature_names"].astype(str).tolist())
        response_ids = source["response_ids"].astype(str)
        source_ids = source["source_ids"].astype(str)
        task_types = source["task_types"].astype(str)
        lengths = np.asarray(source["response_lengths"], dtype=np.float64)
        references = {
            name: np.asarray(source["score__" + name], dtype=np.float64)
            for name in (
                "gasp_top50",
                "original30_full__iu_pcr",
                "original30_full__dufs_liu",
            )
        }
    if raw.shape != (len(response_ids), len(names)) or len(names) != 30:
        raise RuntimeError(f"{split}: invalid original-30 source shape")
    prepared = prepare_arm(raw, names, "D1_TRANSFORM_ONLY")
    per_seed = []
    health = []
    reliability = None
    for seed in SEEDS:
        result, gate_map = fit_ciw_deem(
            prepared,
            source_ids,
            lengths,
            seed=seed,
            config=ContinuousDeemConfig(),
        )
        if not result.health.get("healthy", False):
            raise RuntimeError(f"{split}: unhealthy seed {seed}: {result.health}")
        per_seed.append(np.asarray(result.score, dtype=np.float64))
        health.append({"seed": seed, **dict(result.health)})
        current = np.asarray(gate_map.reliability, dtype=np.float64)
        if reliability is None:
            reliability = current
        elif not np.array_equal(reliability, current):
            raise RuntimeError(f"{split}: reliability changed across seeds")
    seed_scores = np.stack(per_seed, axis=1)
    score = seed_scores.mean(axis=1)
    if not np.isfinite(score).all():
        raise RuntimeError(f"{split}: non-finite CIW score")
    output_path = out / f"scores_{split}.npz"
    digest = atomic_write_npz(output_path, {
        "response_ids": response_ids,
        "source_ids": source_ids,
        "task_types": task_types,
        "score": score,
        "per_seed_score": seed_scores,
        "reliability": np.asarray(reliability, dtype=np.float64),
        "feature_names": np.asarray(prepared.feature_names, dtype="<U64"),
        **{"reference__" + key: value for key, value in references.items()},
    })
    return {
        "split": split,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "score_path": output_path.name,
        "score_sha256": digest,
        "n": len(score),
        "seeds": list(SEEDS),
        "health": health,
    }


def _official_labels(path: Path) -> dict[str, int]:
    output: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row["id"])] = int(bool(row.get("labels")))
    return output


def _metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _evaluate(out: Path, freeze: dict[str, Any], official: Path) -> None:
    # Full score/hash preflight before opening labels.
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for record in freeze["splits"]:
        path = out / record["score_path"]
        if sha256_file(path) != record["score_sha256"]:
            raise RuntimeError(f"{record['split']}: score freeze mismatch")
        with np.load(path, allow_pickle=False) as data:
            if any("label" in key.lower() or "target" in key.lower() for key in data.files):
                raise RuntimeError("label-like field found in frozen score artifact")
            loaded[record["split"]] = {key: np.asarray(data[key]) for key in data.files}

    labels = _official_labels(official)
    rows: list[dict[str, Any]] = []
    for split, data in loaded.items():
        ids = data["response_ids"].astype(str)
        y = np.asarray([labels[value] for value in ids], dtype=int)
        tasks = data["task_types"].astype(str)
        methods = {
            "CIW-DEEM": np.asarray(data["score"], dtype=np.float64),
            "GASP-top50": np.asarray(data["reference__gasp_top50"], dtype=np.float64),
            "Original-30 IU-PCR": np.asarray(
                data["reference__original30_full__iu_pcr"], dtype=np.float64
            ),
            "Original-30 DUFS-LIU": np.asarray(
                data["reference__original30_full__dufs_liu"], dtype=np.float64
            ),
        }
        for task in ("ALL", *sorted(set(tasks.tolist()))):
            mask = np.ones(len(y), dtype=bool) if task == "ALL" else tasks == task
            for method, score in methods.items():
                auroc, auprc = _metrics(y[mask], score[mask])
                rows.append({
                    "split": split,
                    "task": task,
                    "method": method,
                    "n": int(mask.sum()),
                    "positive_rate": float(y[mask].mean()),
                    "auroc": auroc,
                    "auprc": auprc,
                })
        task_names = sorted(set(tasks.tolist()))
        for method, score in methods.items():
            task_metrics = [_metrics(y[tasks == task], score[tasks == task]) for task in task_names]
            rows.append({
                "split": split,
                "task": "MACRO_TASK",
                "method": method,
                "n": len(y),
                "positive_rate": float(y.mean()),
                "auroc": float(np.mean([value[0] for value in task_metrics])),
                "auprc": float(np.mean([value[1] for value in task_metrics])),
            })
    _write_csv(out / "METRICS.csv", rows)
    test = {
        (row["task"], row["method"]): row
        for row in rows if row["split"] == "test"
    }
    lines = [
        "# CIW-DEEM on RAGTruth response-level original-30 features",
        "",
        "All CIW scores were fitted without labels and hash-frozen before the official response labels were opened.",
        "",
        "| Aggregation | Method | AUROC | AUPRC |",
        "|---|---|---:|---:|",
    ]
    for task in ("ALL", "MACRO_TASK", "Data2txt", "QA"):
        for method in ("CIW-DEEM", "Original-30 IU-PCR", "Original-30 DUFS-LIU", "GASP-top50"):
            row = test[(task, method)]
            lines.append(f"| {task} | {method} | {row['auroc']:.6f} | {row['auprc']:.6f} |")
    lines.extend([
        "",
        "This is a retrospective response-level RAG transfer result. It does not cover sentence, token, span, or claim units.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(out / "EVALUATION_MANIFEST.json", {
        "schema_version": SCHEMA,
        "method_id": METHOD_ID,
        "scores_preflighted_before_labels": True,
        "official_labels_path": str(official),
        "official_labels_sha256": sha256_file(official),
        "metrics_sha256": sha256_file(out / "METRICS.csv"),
        "report_sha256": sha256_file(out / "REPORT.md"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--official-responses", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    records = [
        _fit_split(Path(args.dev_source), out, "dev"),
        _fit_split(Path(args.test_source), out, "test"),
    ]
    freeze = {
        "schema_version": SCHEMA,
        "method_id": METHOD_ID,
        "labels_opened_during_fit": False,
        "targets_accessed_during_fit": False,
        "splits": records,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    _evaluate(out, freeze, Path(args.official_responses))


if __name__ == "__main__":
    main()
