#!/usr/bin/env python3
"""Answer-level component evaluation used by the frozen GL-LIU v1 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
LOC = Path(__file__).resolve().parent / "localization"
package = types.ModuleType("spectral_utils")
package.__path__ = [str(ROOT / "spectral_utils")]
sys.modules["spectral_utils"] = package
sys.path[:0] = [str(ROOT), str(LOC)]

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.feature_contract import (
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    SCHEMA_VERSION,
)
from spectral_utils.feature_utils import extract_all_features
from spectral_utils.laplacian_upcr import build_graph_from_features, laplacian_iu_path
from spectral_utils.repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)
from spectral_utils.upcr import upcr_fit
from evidence_drop import candidate_risks


MIN_AVAIL = 0.70
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
LAMBDA = 0.1
DEPLOYED_FIT = {
    "loss": "l2", "exclusion": True, "difficulty_gate": False,
    "simple_avg_fallback": True, "recompute_after_exclusion": True,
    "g2_projection_k": 1, "scale_ratio": 0.25,
}


def load_rows(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return [cache[key] for key in sorted(cache) if not cache[key]["align_diag"]["problems"]]


def trace_features(row):
    out = extract_all_features(
        row["token_entropies"], spilled_energies=row.get("token_spilled_energies"),
        allow_short=True,
    ) or {}
    if row.get("token_logsumexp") is not None:
        out.update(energy_features_from_logsumexp(row["token_logsumexp"]))
    if row.get("top_k_logprobs") is not None:
        out.update(logprob_features(row["top_k_logprobs"]))
        out.update(logprob_features_extended(row["top_k_logprobs"]))
    return out


def fit_scores(rows):
    """Fit label-free scores.  This function deliberately has no label argument."""
    features = [trace_features(row) for row in rows]
    candidates = [
        name for name in CONFIDENCE_FEATURE_SIGNS_V1
        if name not in FIXED_STABLE_EXCLUDED_V1
    ]
    availability = {
        name: float(np.mean([np.isfinite(item.get(name, np.nan)) for item in features]))
        for name in candidates
    }
    names, columns, dropped = [], [], {}
    for name in candidates:
        if availability[name] < MIN_AVAIL:
            dropped[name] = f"availability={availability[name]:.4f}"
            continue
        raw = np.asarray([item.get(name, np.nan) for item in features], dtype=float)
        finite = np.isfinite(raw)
        median = float(np.median(raw[finite]))
        raw = np.where(finite, raw, median)
        if raw.std() < 1e-8:
            dropped[name] = "constant"
            continue
        if float(np.mean(raw == np.median(raw))) > 0.40:
            dropped[name] = "saturated"
            continue
        oriented = raw * float(CONFIDENCE_FEATURE_SIGNS_V1[name])
        columns.append((oriented - oriented.mean()) / oriented.std())
        names.append(name)
    F = np.column_stack(columns).T

    gates, gate_diag = adapted_dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    dufs_graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    path = laplacian_iu_path(F, (0.0, LAMBDA), graph=dufs_graph)
    iu = path[0.0].baseline
    dufs = path[LAMBDA]
    uniform_graph = build_graph_from_features(F, k=DUFS_K)
    uniform = laplacian_iu_path(F, (LAMBDA,), graph=uniform_graph)[LAMBDA]
    deployed = upcr_fit(F, **DEPLOYED_FIT)

    # All scores below are oriented higher = more likely correct.
    scores = {
        "deployed_upcr_fixed_stable": deployed.w @ F,
        "iu_pcr_fixed_stable": iu.w @ F,
        "dufs_liu_fixed_stable_l0p1": dufs.w @ F,
        "uniform_liu_fixed_stable_l0p1": uniform.w @ F,
    }
    baseline_names = None
    for i, row in enumerate(rows):
        risks = candidate_risks(row)
        if baseline_names is None:
            baseline_names = sorted(risks)
            for name in baseline_names:
                scores[name] = np.empty(len(rows), dtype=float)
        for name in baseline_names:
            scores[name][i] = -float(risks[name])

    diagnostics = {
        "labels_seen_during_fit": False,
        "feature_schema": SCHEMA_VERSION,
        "feature_names": names,
        "n_features": len(names),
        "availability": availability,
        "dropped": dropped,
        "dufs_gate": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in gate_diag.items()
        },
        "dufs_liu": dufs.diagnostics,
        "uniform_liu": uniform.diagnostics,
    }
    return scores, diagnostics


def metrics(y, score, seed=0, n_boot=1000):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ix = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        boots[i] = roc_auc_score(y[ix], score[ix])
    lo, hi = np.quantile(boots, (0.025, 0.975))
    return {"auroc": auc, "auroc_lo": float(lo), "auroc_hi": float(hi), "auprc": ap}


def run(path, out_dir):
    rows = load_rows(path)
    scores, diagnostics = fit_scores(rows)
    hashes = {
        name: hashlib.sha256(np.asarray(score, dtype="<f8").tobytes()).hexdigest()
        for name, score in scores.items()
    }

    # Labels are opened only after every method score has been frozen and hashed.
    targets = {
        "final_answer_correct": np.asarray([row["final_answer_correct"] for row in rows]),
        "all_process_steps_correct": np.asarray([row["label"] == -1 for row in rows]),
    }
    subset = os.path.basename(path).removeprefix("processbench_").removesuffix(".pkl")
    records = []
    for target_name, target in targets.items():
        for method, score in scores.items():
            records.append({
                "subset": subset, "target": target_name, "method": method,
                "n": len(target), "positive_rate": float(target.mean()),
                **metrics(target, score),
            })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{subset}__answer_level.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with open(os.path.join(out_dir, f"{subset}__answer_level_diagnostics.json"), "w") as f:
        json.dump({
            "n_rows": len(rows), "score_hashes_before_evaluation": hashes,
            "labels_used_only_after_scores_frozen": True, "fit": diagnostics,
        }, f, indent=2, default=str)

    print("\n", subset)
    for target in targets:
        print(target)
        selected = [row for row in records if row["target"] == target]
        selected.sort(key=lambda row: row["auroc"], reverse=True)
        for row in selected:
            print(f"  {row['method']:36s} {row['auroc']:.4f} "
                  f"[{row['auroc_lo']:.4f}, {row['auroc_hi']:.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    for name in sorted(os.listdir(args.data_dir)):
        if name.startswith("processbench_") and name.endswith(".pkl"):
            run(os.path.join(args.data_dir, name), args.out_dir)


if __name__ == "__main__":
    main()
