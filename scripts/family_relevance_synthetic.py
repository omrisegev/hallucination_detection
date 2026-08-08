#!/usr/bin/env python3
"""Synthetic mechanism and failure-world study for graph-coupled family gates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.family_relevance import (  # noqa: E402
    fit_family_relevance_paths,
    generate_switching_family_world,
)


DEFAULT_OUT = os.path.join(REPO, "results", "family_relevance_synthetic")
BETAS = (0.0, 0.3, 1.0, 3.0)
BLENDS = (0.25, 0.5, 1.0)
SEEDS = tuple(range(20))


def write_csv(path, rows):
    fields = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for scenario, correlated in (("independent_noise", False), ("correlated_nuisance", True)):
        for seed in range(args.seeds):
            F, names, labels, _ = generate_switching_family_world(
                seed=seed, correlated_nuisance=correlated
            )
            scores, diagnostics = fit_family_relevance_paths(
                F, names, cell=f"synthetic:{scenario}:{seed}", betas=BETAS, blends=BLENDS
            )
            baseline = roc_auc_score(labels, scores["iu_pcr"])
            for method, score in scores.items():
                if method in {
                    "sample_index", "feature_names", "family_names", "family_experts",
                    "raw_family_evidence", "context_trace_length",
                    "context_family_disagreement", "context_iu_rank",
                } or np.asarray(score).ndim != 1:
                    continue
                auc = roc_auc_score(labels, score)
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "auroc": auc,
                    "delta_pp": 100.0 * (auc - baseline),
                })
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "method": "iu_pcr",
                "auroc": baseline,
                "delta_pp": 0.0,
            })
    write_csv(os.path.join(args.out_dir, "per_seed.csv"), rows)

    summary = []
    for scenario in ("independent_noise", "correlated_nuisance"):
        methods = sorted({row["method"] for row in rows if row["scenario"] == scenario})
        for method in methods:
            values = np.asarray([
                row["delta_pp"] for row in rows
                if row["scenario"] == scenario and row["method"] == method
            ], dtype=float)
            summary.append({
                "scenario": scenario,
                "method": method,
                "mean_delta_pp": float(np.mean(values)),
                "median_delta_pp": float(np.median(values)),
                "wins": int(np.sum(values > 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "worst_delta_pp": float(np.min(values)),
            })
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)

    candidates = [
        row for row in summary
        if row["scenario"] == "independent_noise"
        and row["method"].startswith("manual_graph__")
    ]
    selected = max(
        candidates,
        key=lambda row: (row["wins"], row["mean_delta_pp"], row["worst_delta_pp"]),
    )
    failure = next(
        row for row in summary
        if row["scenario"] == "correlated_nuisance"
        and row["method"] == selected["method"]
    )
    decision = {
        "selection_rule": "maximum wins, then mean delta, then worst delta in independent-noise world",
        "selected_method": selected["method"],
        "independent_noise": selected,
        "correlated_nuisance": failure,
        "mechanism_gate": bool(
            selected["wins"] >= int(np.ceil(0.8 * args.seeds))
            and selected["mean_delta_pp"] >= 0.5
        ),
        "failure_world_expected_to_be_hard": True,
    }
    with open(os.path.join(args.out_dir, "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
