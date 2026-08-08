#!/usr/bin/env python3
"""Test whether deployed U-PCR hard filtering makes DUFS-LIU more useful.

The ``fit`` command never reads correctness labels.  It builds a deployed-U-PCR
keep mask at several predeclared strictness levels, then refits ordinary IU-PCR
and DUFS-LIU from scratch on exactly the same surviving feature set.  Score
files are hashed before the separate ``report`` command opens labels.

This is a sensitivity experiment on the existing 24 development cells.  A
threshold that looks best here is retrospective and needs external validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.dufs_liu_feature_contract import (  # noqa: E402
    CONTRACT_VERSION as MIXED_CONTRACT_VERSION,
    dufs_liu_mixed_v2_from_bundle,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    IU_FIT_DEFAULTS,
    build_graph_from_features,
    dufs_soft_gates,
    graph_diagnostics,
    laplacian_iu_path,
)
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "hard-filter-dufs-liu-24cell-v1-2026-08-08"
DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_OUT = REPO / "results" / "hard_filter_dufs_liu_24cell"

CONTRACTS = ("fixed_stable_v1", "mixed_v2")
FILTERS = (
    ("full", None),
    ("rho_max_over_3", 3.0),
    ("rho_max_over_2p5", 2.5),
    ("rho_max_over_2", 2.0),
    ("rho_max_over_1p5", 1.5),
)
SOLVERS = ("iu_pcr", "dufs_liu")
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
LIU_LAMBDA = 0.1
MIN_FRAC = 0.05
FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)

FILTER_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "min_frac": MIN_FRAC,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_key(contract: str, filter_name: str, solver: str) -> str:
    return f"{contract}__{filter_name}__{solver}"


def filter_score_key(contract: str, filter_name: str) -> str:
    return f"{contract}__{filter_name}__filter_upcr"


def family(cell: str) -> str:
    return next((name for name in FAMILY_NAMES if name in cell), cell)


def bundle_cells(data) -> set[str]:
    suffix = "__V"
    return {key[:-len(suffix)] for key in data.files if key.endswith(suffix)}


def validate_bundle_without_labels(data) -> None:
    if bundle_cells(data) != set(INSCOPE):
        raise RuntimeError("bundle does not contain the registered 24-cell roster")
    for cell in INSCOPE:
        for suffix in ("V", "pool", "hand_signs"):
            if f"{cell}__{suffix}" not in data.files:
                raise RuntimeError(f"bundle is missing {cell}__{suffix}")


def load_contract(data, cell: str, contract: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load an oriented, standardized features-by-samples matrix without labels."""
    stored = np.asarray(data[f"{cell}__V"], dtype=float)
    names = tuple(str(value) for value in data[f"{cell}__pool"])
    legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
    if contract == "fixed_stable_v1":
        matrix, kept_names = fixed_stable_from_bundle(stored, names, legacy)
    elif contract == "mixed_v2":
        matrix, kept_names, _ = dufs_liu_mixed_v2_from_bundle(stored, names, legacy)
    else:
        raise ValueError(f"unknown feature contract: {contract}")
    F = np.asarray(matrix.T, dtype=float)
    if F.shape[0] != len(kept_names) or not np.isfinite(F).all():
        raise RuntimeError(f"invalid feature matrix for {cell}/{contract}")
    return F, tuple(kept_names)


def fit_one_subset(F: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Refit DUFS and both LIU solvers on one already-selected feature set."""
    gates, gate_diag = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    path = laplacian_iu_path(F, (0.0, LIU_LAMBDA), graph=graph)
    iu = path[0.0]
    dufs = path[LIU_LAMBDA]
    iu_score = np.asarray(iu.w @ F, dtype=np.float64)
    dufs_score = np.asarray(dufs.w @ F, dtype=np.float64)
    if not np.array_equal(iu_score, iu.baseline.w @ F):
        raise RuntimeError("lambda=0 did not exactly reproduce ordinary IU-PCR")
    if not np.isfinite(iu_score).all() or not np.isfinite(dufs_score).all():
        raise RuntimeError("non-finite score")
    diagnostics = {
        "dufs": {
            "raw_probabilities": np.asarray(
                gate_diag["raw_probabilities"], dtype=float
            ).tolist(),
            "mean_probability": float(gate_diag["mean_probability"]),
            "near_zero_fraction": float(gate_diag["near_zero_fraction"]),
            "near_one_fraction": float(gate_diag["near_one_fraction"]),
            "effective_feature_count": float(gate_diag["effective_feature_count"]),
            "mean_seed_std": float(gate_diag["mean_seed_std"]),
        },
        "graph": graph_diagnostics(graph),
        "iu": {
            "n_components_used": int(iu.baseline.n_components_used),
            "projection_residual": float(iu.baseline.proj_residual),
            "score_variance": float(np.var(iu_score)),
        },
        "dufs_liu": {
            "weight_cosine_vs_iu": float(
                dufs.diagnostics["weight_cosine_vs_iu"]
            ),
            "score_variance": float(np.var(dufs_score)),
            "score_laplacian_energy": float(
                dufs.diagnostics["score_laplacian_energy"]
            ),
        },
    }
    return {"iu_pcr": iu_score, "dufs_liu": dufs_score}, diagnostics


def fit_cell(data, cell: str) -> tuple[dict[str, np.ndarray], dict]:
    """Fit every registered arm. This function never accesses ``__labels``."""
    scores: dict[str, np.ndarray] = {}
    diagnostics = {"cell": cell, "domain": GROUP[cell], "contracts": {}}
    n_samples = None
    for contract in CONTRACTS:
        F, names = load_contract(data, cell, contract)
        if n_samples is None:
            n_samples = F.shape[1]
        elif n_samples != F.shape[1]:
            raise RuntimeError("contracts disagree on sample count")
        contract_diag = {
            "n_input_features": int(F.shape[0]),
            "input_features": list(names),
            "filters": {},
        }
        for filter_name, exclude_frac in FILTERS:
            started = time.time()
            if exclude_frac is None:
                keep = np.ones(F.shape[0], dtype=bool)
                filter_fit = None
            else:
                filter_fit = upcr_fit(
                    F, **FILTER_FIT, exclude_frac=float(exclude_frac)
                )
                keep = np.asarray(filter_fit.keep, dtype=bool)
                scores[filter_score_key(contract, filter_name)] = np.asarray(
                    filter_fit.w @ F, dtype=np.float64
                )
            if int(keep.sum()) < 3:
                raise RuntimeError(f"fewer than three features in {cell}/{contract}/{filter_name}")
            selected_F = F[keep]
            selected_names = [name for name, flag in zip(names, keep) if flag]
            fitted, subset_diag = fit_one_subset(selected_F)
            for solver, values in fitted.items():
                scores[score_key(contract, filter_name, solver)] = values
            filter_diag = {
                "exclude_frac": exclude_frac,
                "min_frac": MIN_FRAC if exclude_frac is not None else None,
                "n_kept": int(keep.sum()),
                "kept_fraction": float(np.mean(keep)),
                "kept_features": selected_names,
                "removed_features": [
                    name for name, flag in zip(names, keep) if not flag
                ],
                "runtime_seconds": float(time.time() - started),
                **subset_diag,
            }
            if filter_fit is not None:
                filter_diag["filter"] = {
                    "rho_hat_full": np.asarray(
                        filter_fit.rho_hat_full, dtype=float
                    ).tolist(),
                    "g2_fraction": float(filter_fit.g2_frac_of_var_y),
                    "n_components_used": int(filter_fit.n_components_used),
                    "used_simple_average": bool(filter_fit.used_simple_average),
                }
            contract_diag["filters"][filter_name] = filter_diag
        diagnostics["contracts"][contract] = contract_diag
    scores["sample_index"] = np.arange(int(n_samples), dtype=np.int64)
    return scores, diagnostics


def run_definition(bundle: Path) -> dict:
    payload = {
        "version": VERSION,
        "scientific_run": True,
        "bundle": os.path.relpath(bundle, REPO),
        "bundle_sha256": sha256_file(bundle),
        "cells": list(INSCOPE),
        "contracts": list(CONTRACTS),
        "mixed_contract_version": MIXED_CONTRACT_VERSION,
        "filters": [
            {"name": name, "exclude_frac": value, "min_frac": MIN_FRAC}
            for name, value in FILTERS
        ],
        "filter_fit": FILTER_FIT,
        "iu_fit": dict(IU_FIT_DEFAULTS),
        "dufs_seeds": list(DUFS_SEEDS),
        "dufs_epochs": DUFS_EPOCHS,
        "dufs_k": DUFS_K,
        "liu_lambda": LIU_LAMBDA,
        "labels_used_during_fit": False,
        "source_sha256": sha256_file(Path(__file__)),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def fit_command(args) -> None:
    bundle = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    scores_dir = out / "scores"
    diagnostics_dir = out / "diagnostics"
    scores_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    definition = run_definition(bundle)
    definition_path = out / "RUN_DEFINITION.json"
    if definition_path.exists():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        if existing != definition:
            raise RuntimeError("existing output has a different run definition")
    else:
        write_json(definition_path, definition)
    data = np.load(bundle, allow_pickle=True)
    validate_bundle_without_labels(data)
    for index, cell in enumerate(INSCOPE, start=1):
        score_path = scores_dir / f"{cell}.npz"
        diagnostic_path = diagnostics_dir / f"{cell}.json"
        if score_path.exists() and diagnostic_path.exists() and args.resume:
            print(f"[{index:02d}/24] {cell}: resume skip", flush=True)
            continue
        print(f"[{index:02d}/24] {cell}: fitting", flush=True)
        scores, diagnostics = fit_cell(data, cell)
        np.savez_compressed(score_path, **scores)
        write_json(diagnostic_path, diagnostics)
    manifest = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "score_files_verified_before_labels": True,
        "score_sha256": {
            cell: sha256_file(scores_dir / f"{cell}.npz") for cell in INSCOPE
        },
        "diagnostic_sha256": {
            cell: sha256_file(diagnostics_dir / f"{cell}.json")
            for cell in INSCOPE
        },
    }
    write_json(out / "SCORE_FREEZE_MANIFEST.json", manifest)
    write_json(out / "FIT_COMPLETE.json", {
        "version": VERSION,
        "cells_complete": len(INSCOPE),
        "labels_used": False,
        "manifest_sha256": sha256_file(out / "SCORE_FREEZE_MANIFEST.json"),
    })
    print(f"Fit complete: {out}")


def verify_freeze(out: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    definition = json.loads((out / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    if definition["version"] != VERSION or manifest["version"] != VERSION:
        raise RuntimeError("version mismatch")
    if manifest["run_fingerprint"] != definition["run_fingerprint"]:
        raise RuntimeError("run fingerprint mismatch")
    frozen = {}
    expected_score_keys = {
        score_key(contract, filter_name, solver)
        for contract in CONTRACTS
        for filter_name, _ in FILTERS
        for solver in SOLVERS
    }
    expected_filter_keys = {
        filter_score_key(contract, filter_name)
        for contract in CONTRACTS
        for filter_name, value in FILTERS if value is not None
    }
    for cell in INSCOPE:
        score_path = out / "scores" / f"{cell}.npz"
        if sha256_file(score_path) != manifest["score_sha256"][cell]:
            raise RuntimeError(f"score hash mismatch: {cell}")
        checkpoint = np.load(score_path, allow_pickle=False)
        if any("label" in key.lower() for key in checkpoint.files):
            raise RuntimeError(f"labels found in score checkpoint: {cell}")
        missing = (expected_score_keys | expected_filter_keys) - set(checkpoint.files)
        if missing:
            raise RuntimeError(f"missing scores in {cell}: {sorted(missing)}")
        frozen[cell] = {
            key: np.asarray(checkpoint[key]) for key in checkpoint.files
        }
    return definition, frozen


def safe_metric(metric, labels, values) -> float:
    try:
        return float(metric(labels, values))
    except ValueError:
        return float("nan")


def bootstrap_ci(values, namespace: str, count: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(count, len(values)))]
    return tuple(float(x) for x in np.quantile(sampled.mean(axis=1), (0.025, 0.975)))


def grouped_values(cells: list[str], values: np.ndarray) -> np.ndarray:
    groups = sorted({family(cell) for cell in cells})
    return np.asarray([
        np.mean([value for cell, value in zip(cells, values) if family(cell) == group])
        for group in groups
    ])


def evaluate(bundle: Path, frozen: dict) -> tuple[list[dict], list[dict]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    filter_rows = []
    data = np.load(bundle, allow_pickle=True)
    for cell in INSCOPE:
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        if not np.array_equal(frozen[cell]["sample_index"], np.arange(len(labels))):
            raise RuntimeError(f"sample order mismatch: {cell}")
        for contract in CONTRACTS:
            for filter_name, exclude_frac in FILTERS:
                for solver in SOLVERS:
                    key = score_key(contract, filter_name, solver)
                    values = np.asarray(frozen[cell][key], dtype=float)
                    rows.append({
                        "cell": cell,
                        "family": family(cell),
                        "domain": GROUP[cell],
                        "contract": contract,
                        "filter": filter_name,
                        "exclude_frac": exclude_frac,
                        "solver": solver,
                        "n": len(labels),
                        "positive_rate": float(labels.mean()),
                        "auroc": safe_metric(roc_auc_score, labels, values),
                        "auprc": safe_metric(average_precision_score, labels, values),
                    })
                if exclude_frac is not None:
                    values = np.asarray(
                        frozen[cell][filter_score_key(contract, filter_name)], dtype=float
                    )
                    filter_rows.append({
                        "cell": cell,
                        "family": family(cell),
                        "domain": GROUP[cell],
                        "contract": contract,
                        "filter": filter_name,
                        "exclude_frac": exclude_frac,
                        "solver": "filter_upcr",
                        "auroc": safe_metric(roc_auc_score, labels, values),
                        "auprc": safe_metric(average_precision_score, labels, values),
                    })
    return rows, filter_rows


def metric_lookup(rows: list[dict], metric: str) -> dict[tuple, float]:
    return {
        (row["cell"], row["contract"], row["filter"], row["solver"]):
        float(row[metric]) for row in rows
    }


def load_keep_diagnostics(out: Path) -> list[dict]:
    rows = []
    for cell in INSCOPE:
        payload = json.loads(
            (out / "diagnostics" / f"{cell}.json").read_text(encoding="utf-8")
        )
        for contract in CONTRACTS:
            input_count = payload["contracts"][contract]["n_input_features"]
            for filter_name, exclude_frac in FILTERS:
                item = payload["contracts"][contract]["filters"][filter_name]
                rows.append({
                    "cell": cell,
                    "family": family(cell),
                    "domain": GROUP[cell],
                    "contract": contract,
                    "filter": filter_name,
                    "exclude_frac": exclude_frac,
                    "n_input": input_count,
                    "n_kept": item["n_kept"],
                    "kept_fraction": item["kept_fraction"],
                    "dufs_effective_count": item["dufs"]["effective_feature_count"],
                    "weight_cosine_vs_iu": item["dufs_liu"]["weight_cosine_vs_iu"],
                })
    return rows


def summarize(rows: list[dict], keep_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    from scipy.stats import wilcoxon

    lookup = metric_lookup(rows, "auroc")
    pr_lookup = metric_lookup(rows, "auprc")
    keep_lookup = {
        (row["cell"], row["contract"], row["filter"]): row
        for row in keep_rows
    }
    summary = []
    comparisons = []
    for contract in CONTRACTS:
        full_increment = np.asarray([
            lookup[cell, contract, "full", "dufs_liu"]
            - lookup[cell, contract, "full", "iu_pcr"]
            for cell in INSCOPE
        ])
        for filter_name, exclude_frac in FILTERS:
            iu = np.asarray([
                lookup[cell, contract, filter_name, "iu_pcr"] for cell in INSCOPE
            ])
            dufs = np.asarray([
                lookup[cell, contract, filter_name, "dufs_liu"] for cell in INSCOPE
            ])
            increment = dufs - iu
            iu_vs_full = iu - np.asarray([
                lookup[cell, contract, "full", "iu_pcr"] for cell in INSCOPE
            ])
            dufs_vs_full = dufs - np.asarray([
                lookup[cell, contract, "full", "dufs_liu"] for cell in INSCOPE
            ])
            did = increment - full_increment
            kept = np.asarray([
                keep_lookup[cell, contract, filter_name]["n_kept"]
                for cell in INSCOPE
            ])
            for solver, values in (("iu_pcr", iu), ("dufs_liu", dufs)):
                pr_values = np.asarray([
                    pr_lookup[cell, contract, filter_name, solver] for cell in INSCOPE
                ])
                lo, hi = bootstrap_ci(values, f"summary-{contract}-{filter_name}-{solver}")
                summary.append({
                    "contract": contract,
                    "filter": filter_name,
                    "exclude_frac": exclude_frac,
                    "solver": solver,
                    "mean_auroc": float(values.mean()),
                    "mean_auprc": float(pr_values.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "qa_auroc": float(np.mean([
                        value for cell, value in zip(INSCOPE, values)
                        if GROUP[cell] == "QA"
                    ])),
                    "math_auroc": float(np.mean([
                        value for cell, value in zip(INSCOPE, values)
                        if GROUP[cell] == "math"
                    ])),
                    "family_macro_auroc": float(np.mean(grouped_values(list(INSCOPE), values))),
                    "mean_features_kept": float(kept.mean()),
                    "median_features_kept": float(np.median(kept)),
                    "min_features_kept": int(kept.min()),
                    "max_features_kept": int(kept.max()),
                })
            inc_lo, inc_hi = bootstrap_ci(
                increment, f"increment-{contract}-{filter_name}"
            )
            did_lo, did_hi = bootstrap_ci(did, f"did-{contract}-{filter_name}")
            try:
                inc_p = float(wilcoxon(increment, zero_method="pratt").pvalue)
            except ValueError:
                inc_p = 1.0
            try:
                did_p = float(wilcoxon(did, zero_method="pratt").pvalue)
            except ValueError:
                did_p = 1.0
            comparisons.append({
                "contract": contract,
                "filter": filter_name,
                "exclude_frac": exclude_frac,
                "mean_features_kept": float(kept.mean()),
                "mean_dufs_minus_iu_pp": float(100 * increment.mean()),
                "increment_ci_low_pp": float(100 * inc_lo),
                "increment_ci_high_pp": float(100 * inc_hi),
                "increment_wins": int(np.sum(increment > 1e-12)),
                "increment_losses": int(np.sum(increment < -1e-12)),
                "increment_wilcoxon_p": inc_p,
                "iu_change_vs_full_pp": float(100 * iu_vs_full.mean()),
                "dufs_change_vs_full_pp": float(100 * dufs_vs_full.mean()),
                "difference_in_difference_pp": float(100 * did.mean()),
                "did_ci_low_pp": float(100 * did_lo),
                "did_ci_high_pp": float(100 * did_hi),
                "did_wilcoxon_p": did_p,
                "did_wins": int(np.sum(did > 1e-12)),
                "did_losses": int(np.sum(did < -1e-12)),
                "worst_dufs_change_vs_full_pp": float(100 * dufs_vs_full.min()),
            })
    return summary, comparisons


def summarize_filter_upcr(filter_rows: list[dict]) -> list[dict]:
    output = []
    for contract in CONTRACTS:
        for filter_name, exclude_frac in FILTERS:
            if exclude_frac is None:
                continue
            selected = [
                row for row in filter_rows
                if row["contract"] == contract and row["filter"] == filter_name
            ]
            output.append({
                "contract": contract,
                "filter": filter_name,
                "exclude_frac": exclude_frac,
                "mean_auroc": float(np.mean([row["auroc"] for row in selected])),
                "mean_auprc": float(np.mean([row["auprc"] for row in selected])),
            })
    return output


def make_plots(out: Path, rows: list[dict], keep_rows: list[dict], comparisons: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hallucination_detection_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_fig = out / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    lookup = metric_lookup(rows, "auroc")
    labels = [name.replace("rho_max_over_", "1/").replace("p", ".") for name, _ in FILTERS]
    x = np.arange(len(FILTERS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, contract in zip(axes, CONTRACTS):
        for solver, color, marker in (
            ("iu_pcr", "#64748b", "o"), ("dufs_liu", "#2563eb", "s")
        ):
            means = []
            lows = []
            highs = []
            for filter_name, _ in FILTERS:
                values = np.asarray([
                    lookup[cell, contract, filter_name, solver] for cell in INSCOPE
                ])
                lo, hi = bootstrap_ci(values, f"plot-{contract}-{filter_name}-{solver}")
                means.append(values.mean())
                lows.append(values.mean() - lo)
                highs.append(hi - values.mean())
            axis.errorbar(x, means, yerr=[lows, highs], color=color, marker=marker,
                          capsize=3, linewidth=2, label=solver.replace("_", " ").upper())
        axis.set_title(contract.replace("_", " "))
        axis.set_xticks(x, labels, rotation=25)
        axis.set_xlabel("Hard-filter setting")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cell-macro AUROC (95% bootstrap CI)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_fig / "auroc_by_filter.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    for offset, contract, color in (
        (-width / 2, "fixed_stable_v1", "#0f766e"),
        (width / 2, "mixed_v2", "#7c3aed"),
    ):
        selected = [row for row in comparisons if row["contract"] == contract]
        values = [row["mean_dufs_minus_iu_pp"] for row in selected]
        lows = [value - row["increment_ci_low_pp"] for value, row in zip(values, selected)]
        highs = [row["increment_ci_high_pp"] - value for value, row in zip(values, selected)]
        axis.bar(x + offset, values, width, color=color, alpha=0.85,
                 label=contract.replace("_", " "))
        axis.errorbar(x + offset, values, yerr=[lows, highs], fmt="none",
                      ecolor="#111827", capsize=3, linewidth=1)
    axis.axhline(0, color="#111827", linewidth=1)
    axis.set_xticks(x, labels, rotation=25)
    axis.set_ylabel("DUFS-LIU minus IU-PCR (AUROC points)")
    axis.set_xlabel("Hard-filter setting")
    axis.set_title("Does pruning increase the value added by the Laplacian?")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_fig / "dufs_incremental_gain.png", dpi=180)
    plt.close(fig)

    keep_lookup = {
        (row["cell"], row["contract"], row["filter"]): row["n_kept"]
        for row in keep_rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for axis, contract in zip(axes, CONTRACTS):
        matrix = np.asarray([
            [keep_lookup[cell, contract, filter_name] for filter_name, _ in FILTERS]
            for cell in INSCOPE
        ])
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(contract.replace("_", " "))
        axis.set_xticks(x, labels, rotation=25)
        axis.set_yticks(np.arange(len(INSCOPE)), INSCOPE, fontsize=7)
        axis.set_xlabel("Hard-filter setting")
        fig.colorbar(image, ax=axis, label="Features kept")
    fig.tight_layout()
    fig.savefig(out_fig / "features_kept_heatmap.png", dpi=180)
    plt.close(fig)

    contract = "mixed_v2"
    full_increment = np.asarray([
        lookup[cell, contract, "full", "dufs_liu"]
        - lookup[cell, contract, "full", "iu_pcr"] for cell in INSCOPE
    ])
    matrix = []
    for cell_index, cell in enumerate(INSCOPE):
        row = []
        for filter_name, _ in FILTERS[1:]:
            increment = (
                lookup[cell, contract, filter_name, "dufs_liu"]
                - lookup[cell, contract, filter_name, "iu_pcr"]
            )
            row.append(100 * (increment - full_increment[cell_index]))
        matrix.append(row)
    matrix = np.asarray(matrix)
    limit = max(0.1, float(np.max(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(8.5, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_yticks(np.arange(len(INSCOPE)), INSCOPE, fontsize=7)
    axis.set_xticks(np.arange(len(FILTERS) - 1), labels[1:], rotation=25)
    axis.set_title("Mixed-v2: change in DUFS value added after pruning")
    axis.set_xlabel("Hard-filter setting")
    fig.colorbar(image, ax=axis, label="Difference-in-difference (AUROC points)")
    fig.tight_layout()
    fig.savefig(out_fig / "mixed_cell_difference_in_difference.png", dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict], contract: str) -> str:
    selected = [row for row in rows if row["contract"] == contract]
    lines = [
        "| filter | mean features | IU-PCR AUROC | DUFS-LIU AUROC | DUFS-IU (pp) | change in DUFS value added vs full (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(lines)


def report_text(
    summary: list[dict], comparisons: list[dict], filter_summary: list[dict]
) -> str:
    summary_lookup = {
        (row["contract"], row["filter"], row["solver"]): row for row in summary
    }
    comp_lookup = {
        (row["contract"], row["filter"]): row for row in comparisons
    }
    sections = []
    for contract in CONTRACTS:
        lines = [
            f"### {contract}", "",
            "| filter | mean features | IU-PCR AUROC / AUPRC | DUFS-LIU AUROC / AUPRC | DUFS-LIU - IU-PCR | change in DUFS value added vs full |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for filter_name, _ in FILTERS:
            iu = summary_lookup[contract, filter_name, "iu_pcr"]
            dufs = summary_lookup[contract, filter_name, "dufs_liu"]
            comp = comp_lookup[contract, filter_name]
            lines.append(
                f"| {filter_name} | {comp['mean_features_kept']:.1f} | "
                f"{iu['mean_auroc']:.6f} / {iu['mean_auprc']:.6f} | "
                f"{dufs['mean_auroc']:.6f} / {dufs['mean_auprc']:.6f} | "
                f"{comp['mean_dufs_minus_iu_pp']:+.3f} pp | "
                f"{comp['difference_in_difference_pp']:+.3f} pp "
                f"[{comp['did_ci_low_pp']:+.3f}, {comp['did_ci_high_pp']:+.3f}] |"
            )
        sections.append("\n".join(lines))

    primary = [row for row in comparisons if row["contract"] == "mixed_v2"]
    best_absolute = max(
        FILTERS, key=lambda item: summary_lookup["mixed_v2", item[0], "dufs_liu"]["mean_auroc"]
    )[0]
    best_increment = max(primary, key=lambda row: row["difference_in_difference_pp"])
    full = comp_lookup["mixed_v2", "full"]
    best_summary = summary_lookup["mixed_v2", best_absolute, "dufs_liu"]
    filter_lookup = {
        (row["contract"], row["filter"]): row for row in filter_summary
    }
    deployed_reference = filter_lookup["fixed_stable_v1", "rho_max_over_3"]
    supported = (
        best_increment["filter"] != "full"
        and best_increment["did_ci_low_pp"] > 0
    )
    verdict = (
        "The current 24 cells support the mechanism: hard filtering increases the "
        "incremental value added by DUFS-LIU."
        if supported else
        "The current 24 cells do not establish that hard filtering increases the "
        "incremental value added by DUFS-LIU."
    )
    return f"""# Hard-filtered IU-PCR and DUFS-LIU on 24 cells

**Status:** Retrospective sensitivity experiment on development data.  No setting
selected here is confirmed on external data.

## Question

Deployed U-PCR estimates each feature's covariance with the unknown correctness
target and removes weak features. Ordinary IU-PCR and DUFS-LIU normally keep the
full input pool. This experiment asks whether applying the deployed hard filter
first makes the DUFS graph more useful.

The key quantity is **DUFS-LIU minus IU-PCR on the same selected features**. It
isolates the Laplacian contribution. We also report a difference-in-difference:

`(filtered DUFS-LIU - filtered IU-PCR) - (full DUFS-LIU - full IU-PCR)`.

A positive value means pruning increased the value added by DUFS. It does not
necessarily mean that the final filtered method is better in absolute AUROC.

## Leakage boundary

The fit stage did not read labels. It estimated every hard-filter mask, trained
DUFS, created the graphs, and froze hashes for all score files first. Labels were
opened only by the report stage to compute AUROC and AUPRC.

The strictness grid changed only `exclude_frac`; `min_frac=0.05`, DUFS seeds,
80 epochs, `k=7`, and Laplacian `lambda=0.1` stayed fixed. Lower denominators are
stricter. The implementation always keeps at least three features.

## Metrics

**AUROC** is the probability that a random correct answer receives a higher score
than a random incorrect answer. **AUROC points** are percentage-point changes in
AUROC; for example, 0.002 AUROC equals 0.2 points. Confidence intervals bootstrap
the 24 cells and therefore describe uncertainty across the current cell roster.

## Results

{chr(10).join(sections)}

### Deployed-style U-PCR reference

The exact deployed reference uses `fixed_stable_v1` and `rho_max_over_3`. Its
cell-macro AUROC is {deployed_reference['mean_auroc']:.6f} and its AUPRC is
{deployed_reference['mean_auprc']:.6f}. The other filter thresholds are
sensitivity arms, not deployed methods.

![AUROC by filter](figures/auroc_by_filter.png)

![Incremental DUFS gain](figures/dufs_incremental_gain.png)

![Features kept](figures/features_kept_heatmap.png)

![Per-cell difference in difference](figures/mixed_cell_difference_in_difference.png)

## Conclusion

{verdict}

For the current mixed-v2 contract, unfiltered DUFS-LIU adds
{full['mean_dufs_minus_iu_pp']:+.3f} AUROC points over IU-PCR. The largest
retrospective increase in that incremental contribution occurs at
`{best_increment['filter']}`: {best_increment['difference_in_difference_pp']:+.3f}
points with a 95% cell-bootstrap interval of
[{best_increment['did_ci_low_pp']:+.3f}, {best_increment['did_ci_high_pp']:+.3f}].

The best absolute mixed-v2 DUFS-LIU row is `{best_absolute}` at
{best_summary['mean_auroc']:.6f} cell-macro AUROC. This is a descriptive result,
not a valid new hyperparameter choice, because all thresholds were compared on
the same 24 development cells.

## Interpretation rule

- If filtered DUFS-LIU improves but filtered IU-PCR improves by the same amount,
  pruning helped the base solver; it did not rescue the DUFS mechanism.
- If the difference-in-difference is positive with a stable lower bound, pruning
  made the graph penalty more useful.
- If aggressive settings collapse to three features, apparent gains must be
  treated as small-subset behavior rather than evidence for DUFS.
- Any candidate threshold must be frozen and tested on new dataset/model cells
  before it can replace the current method.
"""


def report_command(args) -> None:
    out = Path(args.out).resolve()
    definition, frozen = verify_freeze(out)
    bundle = (REPO / definition["bundle"]).resolve()
    if sha256_file(bundle) != definition["bundle_sha256"]:
        raise RuntimeError("input bundle changed after fitting")
    rows, filter_rows = evaluate(bundle, frozen)
    keep_rows = load_keep_diagnostics(out)
    summary, comparisons = summarize(rows, keep_rows)
    filter_summary = summarize_filter_upcr(filter_rows)
    write_csv(out / "per_cell_metrics.csv", rows + filter_rows)
    write_csv(out / "feature_counts.csv", keep_rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "filter_upcr_summary.csv", filter_summary)
    write_csv(out / "comparisons.csv", comparisons)
    make_plots(out, rows, keep_rows, comparisons)
    (out / "REPORT.md").write_text(
        report_text(summary, comparisons, filter_summary), encoding="utf-8"
    )
    write_json(out / "REPORT_COMPLETE.json", {
        "version": VERSION,
        "scores_verified_before_labels": True,
        "n_cells": len(INSCOPE),
        "report_sha256": sha256_file(out / "REPORT.md"),
    })
    print(f"Report complete: {out / 'REPORT.md'}")


def self_test() -> None:
    assert len(INSCOPE) == 24
    assert FILTERS[0] == ("full", None)
    assert [value for _, value in FILTERS[1:]] == [3.0, 2.5, 2.0, 1.5]
    assert score_key("mixed_v2", "rho_max_over_3", "dufs_liu") == (
        "mixed_v2__rho_max_over_3__dufs_liu"
    )
    assert filter_score_key("fixed_stable_v1", "rho_max_over_2") == (
        "fixed_stable_v1__rho_max_over_2__filter_upcr"
    )
    print("HARD-FILTER DUFS-LIU BENCHMARK SELF-TEST PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    fit.add_argument("--out", default=str(DEFAULT_OUT))
    fit.add_argument("--resume", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--out", default=str(DEFAULT_OUT))
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "fit":
        fit_command(args)
    elif args.command == "report":
        report_command(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
