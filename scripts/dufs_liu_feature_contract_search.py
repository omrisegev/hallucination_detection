#!/usr/bin/env python3
"""Search per-feature replacements for the four quarantined raw views.

The fit phase is label-free.  It evaluates every global contract in

    {drop, raw, squared, mode-centred} ** 4

with the current DUFS-LIU implementation (three DUFS seeds, 80 epochs, k=7,
lambda=0.1).  One score bank per cell is written before the report phase is
allowed to read labels.  The report then separates three quantities:

* the retrospective winner on all 24 cells (a development choice, not an
  unbiased performance estimate);
* leave-one-dataset-family-out selection, which estimates how well the contract
  choice transfers to a family that did not choose it; and
* fixed controls in which all four views receive the same treatment.

Missing views stay missing.  A transformed view replaces its raw parent and is
never added beside it, so deterministic duplicates cannot receive extra votes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import sys
import time
import types

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.inscope_cells import GROUP, INSCOPE                         # noqa: E402
from spectral_utils.feature_contract import (                            # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    LEGACY_FEATURE_SIGNS,
    SCHEMA_VERSION,
)
from spectral_utils.laplacian_upcr import (                               # noqa: E402
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_fit,
)


VERSION = "dufs-liu-feature-contract-search-v1-2026-08-07"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "dufs_liu_feature_contract_search")

FEATURES = (
    "pe_mean",
    "stft_spectral_entropy",
    "cusum_shift_idx",
    "rpdi",
)
CHOICES = ("drop", "raw", "squared", "mode")
CONTRACTS = tuple(itertools.product(CHOICES, repeat=len(FEATURES)))

DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
LAMBDA = 0.1
EPS = 1e-12

FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)


def family(cell: str) -> str:
    return next((name for name in FAMILY_NAMES if name in cell), cell)


def contract_id(index: int) -> str:
    return f"c{int(index):03d}"


def contract_dict(contract) -> dict[str, str]:
    return dict(zip(FEATURES, contract))


def contract_label(contract) -> str:
    return ";".join(f"{name}={choice}" for name, choice in contract_dict(contract).items())


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def zscore_columns(matrix):
    matrix = np.asarray(matrix, dtype=float)
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, keepdims=True)
    scale[scale < EPS] = 1.0
    return centred / scale


def percentile_rank(values):
    values = np.asarray(values, dtype=float)
    return (rankdata(values) - 0.5) / len(values)


def mode_percentile(values, grid_size=512, min_prominence=0.05):
    """Return the label-free KDE mode location on the percentile scale."""
    values = np.asarray(values, dtype=float)
    if len(values) < 50 or np.std(values) < EPS:
        return 0.5
    try:
        kde = gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), int(grid_size))
        density = kde(grid)
    except Exception:
        return 0.5
    if not np.isfinite(density).all() or density.max() <= 0:
        return 0.5
    peaks, properties = find_peaks(
        density, prominence=float(min_prominence) * float(density.max())
    )
    if len(peaks):
        peak = int(peaks[np.argmax(properties["prominences"])])
    else:
        peak = int(np.argmax(density))
    return float(np.mean(values < grid[peak]))


def reconstruct_raw(data, cell: str):
    names = tuple(str(name) for name in data[f"{cell}__pool"])
    legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
    expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
    if not np.array_equal(legacy, expected):
        raise RuntimeError(f"{cell}: stored legacy signs disagree with the registry")
    stored = np.asarray(data[f"{cell}__V"], dtype=float)
    return stored * legacy[None, :], names


def prepare_contract_columns(raw, names):
    """Precompute every label-free replacement once for one cell."""
    if set(FIXED_STABLE_EXCLUDED_V1) != set(FEATURES):
        raise RuntimeError("registered quarantine changed; update the search definition")
    unknown = sorted(set(names) - set(CONFIDENCE_FEATURE_SIGNS_V1))
    if unknown:
        raise KeyError("unregistered feature(s): " + ", ".join(unknown))
    signs = np.asarray([CONFIDENCE_FEATURE_SIGNS_V1[name] for name in names], dtype=float)
    oriented = zscore_columns(np.asarray(raw, dtype=float) * signs[None, :])
    options = {}
    centres = {}
    for index, name in enumerate(names):
        values = oriented[:, index]
        options[name, "raw"] = values
        if name in FEATURES:
            options[name, "squared"] = -(values ** 2)
            centre = mode_percentile(values)
            options[name, "mode"] = -np.abs(percentile_rank(values) - centre)
            centres[name] = centre
    return options, centres


def build_matrix(raw, names, contract, *, prepared=None):
    """Apply one mixed contract and return a confidence-oriented matrix."""
    decisions = contract_dict(contract)
    options, all_centres = (
        prepare_contract_columns(raw, names) if prepared is None else prepared
    )
    columns = []
    kept_names = []
    centres = {}
    for index, name in enumerate(names):
        choice = decisions.get(name, "raw")
        if choice == "drop":
            continue
        if choice == "raw":
            replacement = options[name, "raw"]
        elif choice == "squared":
            replacement = options[name, "squared"]
        elif choice == "mode":
            replacement = options[name, "mode"]
            centres[name] = all_centres[name]
        else:
            raise ValueError(f"unknown transform choice: {choice}")
        columns.append(replacement)
        kept_names.append(name)
    if len(columns) < 3:
        raise RuntimeError("contract left fewer than three features")
    return zscore_columns(np.column_stack(columns)), tuple(kept_names), centres


def _fit_one_cell(bundle: str, out_dir: str, cell: str, resume: bool) -> dict:
    """Fit every unique applicable contract for one cell without reading labels."""
    score_path = os.path.join(out_dir, "scores", f"{cell}.npz")
    diagnostic_path = os.path.join(out_dir, "diagnostics", f"{cell}.json")
    if resume and os.path.exists(score_path) and os.path.exists(diagnostic_path):
        with np.load(score_path, allow_pickle=False) as checkpoint:
            required = {"sample_index", "iu_pcr", "dufs_liu"}
            if required <= set(checkpoint.files):
                shape = np.asarray(checkpoint["dufs_liu"]).shape
                if shape[0] == len(CONTRACTS):
                    return {
                        "cell": cell,
                        "score_file": os.path.relpath(score_path, out_dir),
                        "score_sha256": sha256_file(score_path),
                        "diagnostic_file": os.path.relpath(diagnostic_path, out_dir),
                        "diagnostic_sha256": sha256_file(diagnostic_path),
                        "status": "reused",
                    }

    started = time.time()
    with np.load(bundle, allow_pickle=True) as data:
        raw, names = reconstruct_raw(data, cell)
    prepared = prepare_contract_columns(raw, names)
    n = raw.shape[0]
    iu_scores = np.empty((len(CONTRACTS), n), dtype=np.float64)
    liu_scores = np.empty((len(CONTRACTS), n), dtype=np.float64)
    cache = {}
    rows = []
    present = set(names) & set(FEATURES)
    for index, contract in enumerate(CONTRACTS):
        applicable = tuple(
            (name, choice) for name, choice in zip(FEATURES, contract) if name in present
        )
        if applicable in cache:
            iu, liu, diagnostics = cache[applicable]
            reused = True
        else:
            matrix, kept_names, centres = build_matrix(
                raw, names, contract, prepared=prepared
            )
            F = matrix.T
            gates, gate_diagnostics = dufs_soft_gates(
                F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
            fit = laplacian_iu_fit(F, lambda_=LAMBDA, graph=graph)
            iu = np.asarray(fit.baseline.w @ F, dtype=np.float64)
            liu = np.asarray(fit.w @ F, dtype=np.float64)
            diagnostics = {
                "n_features": int(F.shape[0]),
                "kept_names": list(kept_names),
                "mode_centres": centres,
                "dufs_effective_feature_count": float(
                    gate_diagnostics["effective_feature_count"]
                ),
                "dufs_mean_seed_std": float(gate_diagnostics["mean_seed_std"]),
                "weight_cosine_vs_iu": float(
                    fit.diagnostics["weight_cosine_vs_iu"]
                ),
                "n_components": int(fit.baseline.n_components_used),
                "n_edges": int(fit.diagnostics["n_edges"]),
                "score_laplacian_energy": float(
                    fit.diagnostics["score_laplacian_energy"]
                ),
            }
            cache[applicable] = (iu, liu, diagnostics)
            reused = False
        iu_scores[index] = iu
        liu_scores[index] = liu
        rows.append({
            "contract_id": contract_id(index),
            "applied_signature": list(applicable),
            "reused_within_cell": reused,
            **diagnostics,
        })
        if (index + 1) % 32 == 0:
            print(
                f"{cell}: {index + 1}/{len(CONTRACTS)} contracts "
                f"({len(cache)} unique)",
                flush=True,
            )
    if not np.isfinite(iu_scores).all() or not np.isfinite(liu_scores).all():
        raise RuntimeError(f"{cell}: non-finite score bank")
    np.savez_compressed(
        score_path,
        sample_index=np.arange(n, dtype=np.int64),
        iu_pcr=iu_scores,
        dufs_liu=liu_scores,
    )
    write_json(diagnostic_path, {
        "cell": cell,
        "domain": GROUP[cell],
        "family": family(cell),
        "n_samples": n,
        "present_quarantined_features": sorted(present),
        "missing_quarantined_features": sorted(set(FEATURES) - present),
        "n_global_contracts": len(CONTRACTS),
        "n_unique_applied_contracts": len(cache),
        "runtime_seconds": time.time() - started,
        "contracts": rows,
    })
    return {
        "cell": cell,
        "score_file": os.path.relpath(score_path, out_dir),
        "score_sha256": sha256_file(score_path),
        "diagnostic_file": os.path.relpath(diagnostic_path, out_dir),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "status": "fit",
    }


def _bundle_cells(data) -> set[str]:
    return {key[:-6] for key in data.files if key.endswith("__pool")}


def validate_bundle(bundle: str) -> None:
    with np.load(bundle, allow_pickle=True) as data:
        observed = _bundle_cells(data)
        if observed != set(INSCOPE):
            raise RuntimeError(
                f"bundle roster mismatch: missing={sorted(set(INSCOPE)-observed)}, "
                f"extra={sorted(observed-set(INSCOPE))}"
            )
        for cell in INSCOPE:
            for suffix in ("V", "pool", "hand_signs", "labels"):
                if f"{cell}__{suffix}" not in data.files:
                    raise RuntimeError(f"bundle missing {cell}__{suffix}")


def run_definition(bundle: str) -> dict:
    source_files = (
        "scripts/dufs_liu_feature_contract_search.py",
        "scripts/inscope_cells.py",
        "spectral_utils/feature_contract.py",
        "spectral_utils/laplacian_upcr.py",
        "spectral_utils/upcr.py",
        "spectral_utils/selectors/a2_groupfs.py",
    )
    payload = {
        "version": VERSION,
        "bundle": os.path.relpath(bundle, REPO),
        "bundle_sha256": sha256_file(bundle),
        "feature_orientation_schema": SCHEMA_VERSION,
        "features": list(FEATURES),
        "choices": list(CHOICES),
        "contracts": [contract_dict(contract) for contract in CONTRACTS],
        "transform_definitions": {
            "drop": "remove the raw view",
            "raw": "frozen confidence sign followed by z-score",
            "squared": "-z^2 followed by z-score",
            "mode": "-|percentile_rank-mode_percentile| followed by z-score",
        },
        "dufs_seeds": list(DUFS_SEEDS),
        "dufs_epochs": DUFS_EPOCHS,
        "dufs_k": DUFS_K,
        "liu_lambda": LAMBDA,
        "cells": list(INSCOPE),
        "source_sha256": {
            path: sha256_file(os.path.join(REPO, path)) for path in source_files
        },
        "labels_opened_by_fit": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def fit(args) -> None:
    validate_bundle(args.bundle)
    os.makedirs(os.path.join(args.out_dir, "scores"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "diagnostics"), exist_ok=True)
    definition = run_definition(args.bundle)
    definition_path = os.path.join(args.out_dir, "RUN_DEFINITION.json")
    if os.path.exists(definition_path):
        with open(definition_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("run_fingerprint") != definition["run_fingerprint"]:
            raise RuntimeError("output directory contains a different run definition")
    else:
        write_json(definition_path, definition)

    started = time.time()
    records = {}
    jobs = max(1, min(int(args.jobs), len(INSCOPE)))
    if jobs == 1:
        for index, cell in enumerate(INSCOPE, start=1):
            records[cell] = _fit_one_cell(args.bundle, args.out_dir, cell, args.resume)
            print(f"completed {index}/{len(INSCOPE)}: {cell}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _fit_one_cell, args.bundle, args.out_dir, cell, args.resume
                ): cell
                for cell in INSCOPE
            }
            for index, future in enumerate(as_completed(futures), start=1):
                cell = futures[future]
                records[cell] = future.result()
                print(f"completed {index}/{len(INSCOPE)}: {cell}", flush=True)
    manifest = [records[cell] for cell in INSCOPE]
    write_json(os.path.join(args.out_dir, "FIT_COMPLETE.json"), {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "n_cells": len(manifest),
        "n_contracts": len(CONTRACTS),
        "runtime_seconds": time.time() - started,
        "labels_opened_by_fit": False,
        "score_manifest": manifest,
    })


def verify_fit(bundle: str, out_dir: str):
    with open(os.path.join(out_dir, "RUN_DEFINITION.json"), encoding="utf-8") as handle:
        definition = json.load(handle)
    with open(os.path.join(out_dir, "FIT_COMPLETE.json"), encoding="utf-8") as handle:
        complete = json.load(handle)
    if definition["version"] != VERSION or complete["version"] != VERSION:
        raise RuntimeError("fit/report version mismatch")
    if definition["run_fingerprint"] != complete["run_fingerprint"]:
        raise RuntimeError("run fingerprints disagree")
    if sha256_file(bundle) != definition["bundle_sha256"]:
        raise RuntimeError("bundle changed after fitting")
    for relative, expected in definition.get("source_sha256", {}).items():
        path = os.path.join(REPO, relative)
        if not os.path.exists(path) or sha256_file(path) != expected:
            raise RuntimeError(
                f"registered source changed after fitting: {relative}; refit in a new directory"
            )
    if complete.get("labels_opened_by_fit") is not False:
        raise RuntimeError("fit did not attest to label isolation")
    if [row["cell"] for row in complete["score_manifest"]] != list(INSCOPE):
        raise RuntimeError("score manifest does not contain the canonical roster")
    scores = {}
    for row in complete["score_manifest"]:
        score_path = os.path.join(out_dir, row["score_file"])
        diagnostic_path = os.path.join(out_dir, row["diagnostic_file"])
        if sha256_file(score_path) != row["score_sha256"]:
            raise RuntimeError(f"score file changed: {score_path}")
        if sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"diagnostic file changed: {diagnostic_path}")
        with np.load(score_path, allow_pickle=False) as checkpoint:
            if any("label" in key.lower() for key in checkpoint.files):
                raise RuntimeError(f"label-like score array found: {score_path}")
            scores[row["cell"]] = {
                key: np.asarray(checkpoint[key]) for key in checkpoint.files
            }
    freeze = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "bundle_sha256": definition["bundle_sha256"],
        "score_files_verified_before_labels": True,
        "manifest": complete["score_manifest"],
    }
    freeze_path = os.path.join(out_dir, "SCORE_FREEZE_MANIFEST.json")
    if os.path.exists(freeze_path):
        with open(freeze_path, encoding="utf-8") as handle:
            if json.load(handle) != freeze:
                raise RuntimeError("immutable freeze manifest disagrees")
    else:
        with open(freeze_path, "x", encoding="utf-8") as handle:
            json.dump(freeze, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return definition, complete, scores


def bootstrap_ci(values, namespace: str, count=20000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(count), len(values)))
    estimates = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def choice_complexity(contract) -> int:
    # Tie-break only.  Dropping is simplest; raw adds a view; squared and KDE-mode
    # add progressively more transformation machinery.
    cost = {"drop": 0, "raw": 1, "squared": 2, "mode": 3}
    return sum(cost[value] for value in contract)


def _best_contract(indices, matrix) -> int:
    means = np.asarray(matrix[:, indices].mean(axis=1), dtype=float)
    order = sorted(
        range(len(CONTRACTS)),
        key=lambda index: (-means[index], choice_complexity(CONTRACTS[index]), index),
    )
    return int(order[0])


def evaluate(args) -> None:
    definition, complete, scores = verify_fit(args.bundle, args.out_dir)
    with np.load(args.bundle, allow_pickle=True) as data:
        labels = {cell: np.asarray(data[f"{cell}__labels"], dtype=int) for cell in INSCOPE}

    methods = ("iu_pcr", "dufs_liu")
    metric = {
        method: np.empty((len(CONTRACTS), len(INSCOPE)), dtype=float)
        for method in methods
    }
    auprc = {method: np.empty_like(metric[method]) for method in methods}
    rows = []
    for cell_index, cell in enumerate(INSCOPE):
        y = labels[cell]
        frozen = scores[cell]
        if not np.array_equal(frozen["sample_index"], np.arange(len(y))):
            raise RuntimeError(f"sample order mismatch: {cell}")
        for contract_index, contract in enumerate(CONTRACTS):
            for method in methods:
                values = np.asarray(frozen[method][contract_index], dtype=float)
                value = float(roc_auc_score(y, values))
                precision = float(average_precision_score(y, values))
                metric[method][contract_index, cell_index] = value
                auprc[method][contract_index, cell_index] = precision
                rows.append({
                    "cell": cell,
                    "domain": GROUP[cell],
                    "family": family(cell),
                    "method": method,
                    "contract_id": contract_id(contract_index),
                    **contract_dict(contract),
                    "auroc": value,
                    "auprc": precision,
                })

    stable_index = CONTRACTS.index(tuple("drop" for _ in FEATURES))
    raw_index = CONTRACTS.index(tuple("raw" for _ in FEATURES))
    squared_index = CONTRACTS.index(tuple("squared" for _ in FEATURES))
    mode_index = CONTRACTS.index(tuple("mode" for _ in FEATURES))
    controls = {
        "stable_drop_all": stable_index,
        "raw_all": raw_index,
        "squared_all": squared_index,
        "mode_all": mode_index,
    }

    summary_rows = []
    selection_rows = []
    families = sorted({family(cell) for cell in INSCOPE})
    family_by_cell = np.asarray([family(cell) for cell in INSCOPE])
    final_choices = {}
    for method in methods:
        values = metric[method]
        best = _best_contract(np.arange(len(INSCOPE)), values)
        final_choices[method] = best
        ranked = sorted(
            range(len(CONTRACTS)),
            key=lambda index: (
                -float(values[index].mean()), choice_complexity(CONTRACTS[index]), index
            ),
        )
        for rank, index in enumerate(ranked, start=1):
            summary_rows.append({
                "method": method,
                "rank": rank,
                "contract_id": contract_id(index),
                **contract_dict(CONTRACTS[index]),
                "cell_macro_auroc": float(values[index].mean()),
                "family_macro_auroc": float(np.mean([
                    values[index, family_by_cell == held].mean() for held in families
                ])),
                "qa_auroc": float(values[index, np.asarray([
                    GROUP[cell] == "QA" for cell in INSCOPE
                ])].mean()),
                "math_auroc": float(values[index, np.asarray([
                    GROUP[cell] == "math" for cell in INSCOPE
                ])].mean()),
                "cell_macro_auprc": float(auprc[method][index].mean()),
            })

        for held in families:
            train_indices = np.flatnonzero(family_by_cell != held)
            test_indices = np.flatnonzero(family_by_cell == held)
            chosen = _best_contract(train_indices, values)
            for cell_index in test_indices:
                selection_rows.append({
                    "method": method,
                    "heldout_family": held,
                    "chosen_contract_id": contract_id(chosen),
                    **contract_dict(CONTRACTS[chosen]),
                    "training_macro_auroc": float(values[chosen, train_indices].mean()),
                    "cell": INSCOPE[cell_index],
                    "domain": GROUP[INSCOPE[cell_index]],
                    "auroc": float(values[chosen, cell_index]),
                    "stable_auroc": float(values[stable_index, cell_index]),
                    "delta_vs_stable": float(
                        values[chosen, cell_index] - values[stable_index, cell_index]
                    ),
                })

    control_rows = []
    for method in methods:
        best = final_choices[method]
        for name, index in {**controls, "retrospective_best": best}.items():
            delta = metric[method][index] - metric[method][stable_index]
            lo, hi = bootstrap_ci(delta, f"{method}-{name}-vs-stable")
            try:
                pvalue = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            except ValueError:
                pvalue = float("nan")
            control_rows.append({
                "method": method,
                "contract": name,
                "contract_id": contract_id(index),
                **contract_dict(CONTRACTS[index]),
                "macro_auroc": float(metric[method][index].mean()),
                "qa_auroc": float(metric[method][index, [
                    GROUP[cell] == "QA" for cell in INSCOPE
                ]].mean()),
                "math_auroc": float(metric[method][index, [
                    GROUP[cell] == "math" for cell in INSCOPE
                ]].mean()),
                "delta_vs_stable_pp": float(100 * delta.mean()),
                "ci95_low_pp": float(100 * lo),
                "ci95_high_pp": float(100 * hi),
                "wins": int(np.sum(delta > 0)),
                "losses": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "p_wilcoxon": pvalue,
            })

    marginal_rows = []
    for method in methods:
        for feature_index, feature in enumerate(FEATURES):
            for choice in CHOICES:
                indices = [
                    index for index, contract in enumerate(CONTRACTS)
                    if contract[feature_index] == choice
                ]
                marginal_rows.append({
                    "method": method,
                    "feature": feature,
                    "choice": choice,
                    "mean_over_other_contracts": float(metric[method][indices].mean()),
                })

    stability_rows = []
    for method in methods:
        method_selection = [row for row in selection_rows if row["method"] == method]
        by_family = {}
        for row in method_selection:
            by_family.setdefault(row["heldout_family"], row)
        for feature in FEATURES:
            counts = Counter(row[feature] for row in by_family.values())
            stability_rows.append({
                "method": method,
                "feature": feature,
                **{f"n_{choice}": counts.get(choice, 0) for choice in CHOICES},
                "modal_choice": max(CHOICES, key=lambda choice: (counts.get(choice, 0), -CHOICES.index(choice))),
                "modal_fraction": max(counts.values()) / len(families),
            })

    write_csv(os.path.join(args.out_dir, "per_cell_metrics.csv"), rows)
    write_csv(os.path.join(args.out_dir, "contract_ranking.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, "fixed_contract_summary.csv"), control_rows)
    write_csv(os.path.join(args.out_dir, "lofo_selection.csv"), selection_rows)
    write_csv(os.path.join(args.out_dir, "choice_marginals.csv"), marginal_rows)
    write_csv(os.path.join(args.out_dir, "selection_stability.csv"), stability_rows)

    make_plots(args.out_dir, metric, final_choices, stable_index, selection_rows)
    report = render_report(
        complete, metric, final_choices, controls, control_rows,
        selection_rows, stability_rows, stable_index,
    )
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)


def make_plots(out_dir, metric, final_choices, stable_index, selection_rows):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figures = os.path.join(out_dir, "figures")
    os.makedirs(figures, exist_ok=True)
    for method in ("iu_pcr", "dufs_liu"):
        best = final_choices[method]
        delta = 100 * (metric[method][best] - metric[method][stable_index])
        order = np.argsort(delta)
        fig, axis = plt.subplots(figsize=(11, 5.5))
        colours = np.where(delta[order] >= 0, "#2a9d8f", "#e76f51")
        axis.bar(np.arange(len(order)), delta[order], color=colours)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(order)), [INSCOPE[i] for i in order], rotation=75, ha="right")
        axis.set_ylabel("AUROC change vs stable-only (percentage points)")
        axis.set_title(f"{method}: retrospective contract winner by cell")
        fig.tight_layout()
        fig.savefig(os.path.join(figures, f"{method}_best_cell_deltas.png"), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, method in zip(axes, ("iu_pcr", "dufs_liu")):
        selected = [row for row in selection_rows if row["method"] == method]
        families = sorted({row["heldout_family"] for row in selected})
        delta = [
            100 * np.mean([row["delta_vs_stable"] for row in selected
                           if row["heldout_family"] == held])
            for held in families
        ]
        axis.bar(families, delta, color=["#2a9d8f" if value >= 0 else "#e76f51" for value in delta])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=55)
        axis.set_title(method)
        axis.set_ylabel("Held-family AUROC change (pp)")
    fig.suptitle("Leave-one-family-out contract selection vs stable-only")
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "lofo_family_deltas.png"), dpi=180)
    plt.close(fig)


def render_report(complete, metric, final_choices, controls, control_rows,
                  selection_rows, stability_rows, stable_index):
    lines = [
        "# DUFS-LIU mixed feature-contract search",
        "",
        f"Version: `{VERSION}`. The fit took {complete['runtime_seconds']:.1f} seconds.",
        "",
        "## Question and protocol",
        "",
        "The four quarantined views were not assumed to need the same treatment. For each view, "
        "the search tested `drop`, confidence-oriented `raw`, `squared` (`-z²`), and "
        "label-free KDE `mode` (`-|rank-mode_rank|`). This gives 256 global contracts. A "
        "transformation replaces its parent; it is never added beside it.",
        "",
        "DUFS-LIU was kept at its current frozen settings: three gate seeds (11, 23, 37), "
        "80 epochs, graph k=7, and lambda=0.1. The fit phase wrote and hashed every score "
        "without reading labels. Labels were opened only after the complete score bank was frozen.",
        "",
        "The retrospective winner is useful for choosing the next candidate, but its score is "
        "optimistic because the same 24 cells chose it. Leave-one-dataset-family-out (LOFO) "
        "selection is the main check on whether the choice transfers.",
        "",
        "## Fixed controls and retrospective winners",
        "",
        "| method | contract | feature decisions | macro AUROC | change vs stable [95% CI] | W/L/T |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in control_rows:
        choices = ", ".join(f"{name}={row[name]}" for name in FEATURES)
        lines.append(
            f"| `{row['method']}` | `{row['contract']}` | {choices} | "
            f"{row['macro_auroc']:.6f} | {row['delta_vs_stable_pp']:+.3f}pp "
            f"[{row['ci95_low_pp']:+.3f}, {row['ci95_high_pp']:+.3f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )

    lines.extend([
        "",
        "![DUFS-LIU winner cell changes](figures/dufs_liu_best_cell_deltas.png)",
        "",
        "## Leave-one-family-out selection",
        "",
        "Each held-out dataset family is evaluated with the contract chosen only from the other "
        "seven families. This is not a new-data confirmation set—the 24 cells have influenced "
        "earlier research—but it prevents a cell or family from selecting its own transform.",
        "",
        "| method | LOFO macro AUROC | stable macro AUROC | change | W/L/T |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in ("iu_pcr", "dufs_liu"):
        selected = [row for row in selection_rows if row["method"] == method]
        delta = np.asarray([row["delta_vs_stable"] for row in selected], dtype=float)
        lines.append(
            f"| `{method}` | {np.mean([row['auroc'] for row in selected]):.6f} | "
            f"{metric[method][stable_index].mean():.6f} | {100*delta.mean():+.3f}pp | "
            f"{np.sum(delta>0)}/{np.sum(delta<0)}/{np.sum(delta==0)} |"
        )
    lines.extend([
        "",
        "![LOFO family changes](figures/lofo_family_deltas.png)",
        "",
        "### Selection stability across the eight held-family folds",
        "",
        "| method | feature | drop | raw | squared | mode | modal choice | stability |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in stability_rows:
        lines.append(
            f"| `{row['method']}` | `{row['feature']}` | {row['n_drop']} | {row['n_raw']} | "
            f"{row['n_squared']} | {row['n_mode']} | `{row['modal_choice']}` | "
            f"{row['modal_fraction']:.0%} |"
        )

    best = final_choices["dufs_liu"]
    decisions = contract_dict(CONTRACTS[best])
    lofo = [row for row in selection_rows if row["method"] == "dufs_liu"]
    lofo_delta = np.asarray([row["delta_vs_stable"] for row in lofo], dtype=float)
    lines.extend([
        "",
        "## Development decision",
        "",
        "The next DUFS-LIU feature-contract candidate selected on all available development "
        "cells is:",
        "",
        *[f"- `{name}`: `{decisions[name]}`" for name in FEATURES],
        "",
        f"Its retrospective cell-macro AUROC is {metric['dufs_liu'][best].mean():.6f}, "
        f"versus {metric['dufs_liu'][stable_index].mean():.6f} for stable-only. The LOFO "
        f"selection procedure changes the held-out cells by {100*lofo_delta.mean():+.3f}pp "
        "on average. The candidate may be frozen for the next external run only if that "
        "transfer result and the per-feature stability table do not reveal a collapse.",
        "",
        "No score in this report is prospective evidence for the selected candidate. A new "
        "dataset/model family is required for an unbiased confirmation.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/dufs_liu_feature_contract_search.py fit --jobs 4 --resume",
        "python scripts/dufs_liu_feature_contract_search.py report",
        "```",
        "",
    ])
    return "\n".join(lines)


def self_test() -> None:
    assert len(CONTRACTS) == 256
    assert set(FEATURES) == set(FIXED_STABLE_EXCLUDED_V1)
    names = ("epr", *FEATURES, "trace_length", "spectral_entropy")
    x = np.linspace(-2, 2, 101)
    raw = np.column_stack((x, x, np.sin(x), x ** 3, np.cos(x), -x, np.sin(2 * x)))
    for contract in CONTRACTS:
        matrix, kept, _ = build_matrix(raw, names, contract)
        assert matrix.shape[0] == len(raw)
        assert matrix.shape[1] == len(kept)
        assert np.isfinite(matrix).all()
    dropped, kept, _ = build_matrix(raw, names, ("drop",) * 4)
    assert dropped.shape[1] == 3 and set(kept).isdisjoint(FEATURES)
    print("DUFS-LIU FEATURE CONTRACT SEARCH SELF-TEST PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "report", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--bundle", default=DEFAULT_BUNDLE)
        child.add_argument("--out-dir", default=DEFAULT_OUT)
        child.add_argument("--jobs", type=int, default=1)
        child.add_argument("--resume", action="store_true")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "fit":
        fit(args)
    elif args.command == "report":
        evaluate(args)
    else:
        fit(args)
        evaluate(args)


if __name__ == "__main__":
    main()
