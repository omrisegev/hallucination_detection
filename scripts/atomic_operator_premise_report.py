#!/usr/bin/env python3
"""Verify the atomic-operator freeze, then evaluate the Phase-0 premise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import types

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "hallucination_detection_mpl")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.atomic_operator_premise_fit import (  # noqa: E402
    CONTINUATION_GATES,
    DEFAULT_BUNDLE,
    DEFAULT_OUT,
    GRAPH_KS,
    LAMBDAS,
    PRIMARY_GRAPH_K,
    PRIMARY_LAMBDA,
    VERSION,
    dependency_version,
    sha256_file,
)
from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.atomic_operator_audit import path_token  # noqa: E402


FAMILY_NAMES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)
NUISANCES = (
    "edge_mass_per_node",
    "projected_effective_rank",
    "duplicate_density",
    "distance_from_ridge",
)
PROXIES = (
    "primary_proxy",
    "full_alignment",
    "bootstrap_alignment",
    "operator_reproducibility",
    "rank_change_reproducibility",
    "stability_actuation_proxy",
    "full_actuation",
    "anisotropy",
)


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


def family(cell: str) -> str:
    return next((name for name in FAMILY_NAMES if name in cell), cell)


def safe_spearman(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def tie_aware_quartiles(proxy):
    proxy = np.asarray(proxy, dtype=float)
    low = float(np.quantile(proxy, 0.25))
    high = float(np.quantile(proxy, 0.75))
    bottom = np.where(proxy <= low + 1e-15)[0]
    top = np.where(proxy >= high - 1e-15)[0]
    if not len(bottom) or not len(top) or np.intersect1d(bottom, top).size:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    return bottom, top


def exact_sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return 1.0
    observed = float(np.mean(values))
    null = [
        float(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    # This is exhaustive over all 2**n assignments, not a Monte Carlo sample.
    # Therefore the exact randomization p-value is count / 2**n; the usual +1
    # finite-sample correction applies only to sampled null distributions.
    return float(np.sum(np.asarray(null) >= observed - 1e-15) / len(null))


def bootstrap_mean_ci(values, namespace: str, count: int = 20000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    draws = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def metric(labels, scores, function) -> float:
    try:
        return float(function(labels, scores))
    except ValueError:
        return float("nan")


def verify_freeze(out_dir: str, bundle: str):
    with open(os.path.join(out_dir, "RUN_DEFINITION.json"), encoding="utf-8") as handle:
        definition = json.load(handle)
    with open(os.path.join(out_dir, "FIT_COMPLETE.json"), encoding="utf-8") as handle:
        complete = json.load(handle)
    if not definition.get("scientific_run") or not complete.get("scientific_run"):
        raise RuntimeError("debug output cannot be evaluated")
    if definition.get("version") != VERSION or complete.get("version") != VERSION:
        raise RuntimeError("fit/report version mismatch")
    if definition.get("run_fingerprint") != complete.get("run_fingerprint"):
        raise RuntimeError("run fingerprints disagree")
    if tuple(definition.get("cells", ())) != tuple(INSCOPE) or complete.get("n_cells") != 24:
        raise RuntimeError("the fit does not contain the exact 24-cell roster")
    if sha256_file(bundle) != definition.get("bundle_sha256"):
        raise RuntimeError("input bundle changed after fitting")
    fit_bundle = os.path.join(REPO, definition["label_free_fit_bundle"])
    if (
        not os.path.exists(fit_bundle)
        or sha256_file(fit_bundle) != definition.get("label_free_fit_bundle_sha256")
    ):
        raise RuntimeError("label-free fit bundle changed after fitting")
    if platform.python_version() != definition.get("python"):
        raise RuntimeError(
            "report Python version differs: "
            f"fit={definition.get('python')}, report={platform.python_version()}"
        )
    if np.__version__ != definition.get("numpy"):
        raise RuntimeError(
            "report NumPy version differs: "
            f"fit={definition.get('numpy')}, report={np.__version__}"
        )
    for name, expected in definition.get("dependency_versions", {}).items():
        if dependency_version(name) != expected:
            raise RuntimeError(
                f"report dependency version differs for {name}: "
                f"fit={expected}, report={dependency_version(name)}"
            )
    for relative, expected in definition.get("source_sha256", {}).items():
        path = os.path.join(REPO, relative)
        if not os.path.exists(path) or sha256_file(path) != expected:
            raise RuntimeError(f"registered source changed after fitting: {relative}")
    manifest = complete.get("artifact_manifest", [])
    if [row.get("cell") for row in manifest] != list(INSCOPE):
        raise RuntimeError("artifact manifest roster/order mismatch")
    for row in manifest:
        for key, hash_key in (
            ("score_file", "score_sha256"),
            ("diagnostic_file", "diagnostic_sha256"),
        ):
            path = os.path.join(out_dir, row[key])
            if not os.path.exists(path) or sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"artifact hash mismatch: {row['cell']}/{key}")
    freeze = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "bundle_sha256": definition["bundle_sha256"],
        "source_sha256": definition["source_sha256"],
        "artifact_manifest": manifest,
    }
    freeze_path = os.path.join(out_dir, "SCORE_FREEZE_MANIFEST.json")
    if os.path.exists(freeze_path):
        with open(freeze_path, encoding="utf-8") as handle:
            if json.load(handle) != freeze:
                raise RuntimeError("immutable score-freeze manifest differs")
    else:
        write_json(freeze_path, freeze)
    return definition, complete


def partial_rank_residuals(rows: list[dict]):
    """Return nuisance-adjusted rank residuals and their cell membership."""
    transformed = []
    cells = sorted({row["cell"] for row in rows})
    for cell in cells:
        local = [row for row in rows if row["cell"] == cell]
        fields = ("primary_proxy", "delta_pp") + NUISANCES
        ranked = {
            field: rankdata([row[field] for row in local]) / len(local)
            for field in fields
        }
        for index, row in enumerate(local):
            transformed.append({
                "cell": cell,
                **{field: ranked[field][index] for field in fields},
            })
    y = np.asarray([row["delta_pp"] for row in transformed], dtype=float)
    x = np.asarray([row["primary_proxy"] for row in transformed], dtype=float)
    columns = [np.ones(len(transformed))]
    columns.extend(np.asarray([row[name] for row in transformed]) for name in NUISANCES)
    for cell in cells[1:]:
        columns.append(np.asarray([row["cell"] == cell for row in transformed], dtype=float))
    design = np.column_stack(columns)
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    fit_y = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    residual_y = y - fit_y
    memberships = np.asarray([cells.index(row["cell"]) for row in transformed], dtype=int)
    return residual_x, residual_y, fit_y, design, memberships


def partial_rank_association(rows: list[dict]) -> float:
    """Partial Spearman after within-cell ranking and nuisance adjustment."""
    residual_x, residual_y, _, _, _ = partial_rank_residuals(rows)
    if np.std(residual_x) <= 1e-12 or np.std(residual_y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def feature_identity_permutation_p(primary_by_cell, observed: float, count=5000) -> float:
    """Within-cell feature-identity null for the equal-family association."""
    if not np.isfinite(observed):
        return 1.0
    rng = np.random.default_rng(
        int(hashlib.sha256(b"atomic-feature-identity-null").hexdigest()[:8], 16)
    )
    cell_data = {}
    for cell in INSCOPE:
        rows = primary_by_cell[cell]["rows"]
        cell_data[cell] = (
            np.asarray([row["primary_proxy"] for row in rows]),
            np.asarray([row["delta_pp"] for row in rows]),
        )
    exceed = 0
    for _ in range(int(count)):
        cell_values = {}
        for cell, (proxy, delta) in cell_data.items():
            cell_values[cell] = safe_spearman(rng.permutation(proxy), delta)
        family_values = [
            np.nanmean([
                cell_values[cell] for cell in INSCOPE if family(cell) == name
            ])
            for name in FAMILY_NAMES
        ]
        statistic = float(np.nanmean(family_values))
        exceed += int(statistic >= observed - 1e-15)
    return float((exceed + 1) / (int(count) + 1))


def freedman_lane_p(rows: list[dict], observed: float, count=5000) -> float:
    """Within-cell residual permutation for the nuisance-adjusted statistic."""
    if not np.isfinite(observed):
        return 1.0
    residual_x, residual_y, fit_y, design, membership = partial_rank_residuals(rows)
    rng = np.random.default_rng(
        int(hashlib.sha256(b"atomic-freedman-lane-null").hexdigest()[:8], 16)
    )
    exceed = 0
    groups = np.unique(membership)
    for _ in range(int(count)):
        permuted = residual_y.copy()
        for group in groups:
            index = np.where(membership == group)[0]
            permuted[index] = rng.permutation(permuted[index])
        pseudo_y = fit_y + permuted
        pseudo_residual = pseudo_y - design @ np.linalg.lstsq(
            design, pseudo_y, rcond=None
        )[0]
        statistic = float(np.corrcoef(residual_x, pseudo_residual)[0, 1])
        exceed += int(statistic >= observed - 1e-15)
    return float((exceed + 1) / (int(count) + 1))


def evaluate(out_dir: str, bundle: str):
    data = np.load(bundle, allow_pickle=True)
    atomic_rows = []
    control_rows = []
    convergence_rows = []
    primary_by_cell = {}
    for cell in INSCOPE:
        score_path = os.path.join(out_dir, "scores", f"{cell}.npz")
        diagnostic_path = os.path.join(out_dir, "diagnostics", f"{cell}.json")
        with np.load(score_path, allow_pickle=False) as loaded:
            scores = {key: np.asarray(loaded[key]) for key in loaded.files}
        with open(diagnostic_path, encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        baseline = scores["iu_pcr"]
        baseline_auc = metric(labels, baseline, roc_auc_score)
        baseline_ap = metric(labels, baseline, average_precision_score)
        records = diagnostic["feature_records"]
        for record in diagnostic["convergence"]:
            convergence_rows.append({
                "cell": cell,
                "family": family(cell),
                **record,
            })
        for graph_k in GRAPH_KS:
            for lambda_ in LAMBDAS:
                key = f"atomic__k_{graph_k}__lambda_{lambda_:g}"
                token = path_token(graph_k, lambda_)
                for index, record in enumerate(records):
                    auc = metric(labels, scores[key][index], roc_auc_score)
                    atomic_rows.append({
                        "cell": cell,
                        "domain": GROUP[cell],
                        "family": family(cell),
                        "feature_index": index,
                        "feature": record["feature"],
                        "graph_k": graph_k,
                        "lambda": lambda_,
                        "valid_operator": bool(record[f"valid_operator__k_{graph_k}"]),
                        "auroc": auc,
                        "auprc": metric(
                            labels, scores[key][index], average_precision_score
                        ),
                        "delta_pp": 100.0 * (auc - baseline_auc),
                        "path_proxy": float(record[f"primary_proxy__{token}"]),
                        **{
                            name: float(record[name])
                            for name in set(PROXIES + NUISANCES + (
                                "operator_duplicate_density",
                                "full_duplicate_fallback",
                                "alignment_duplicate_0p9",
                                "alignment_duplicate_0p99",
                            ))
                        },
                    })
                uniform_key = f"uniform_atomic__k_{graph_k}__lambda_{lambda_:g}"
                uniform_auc = metric(labels, scores[uniform_key], roc_auc_score)
                control_rows.append({
                    "cell": cell,
                    "domain": GROUP[cell],
                    "family": family(cell),
                    "method": f"uniform_atomic_k{graph_k}_lambda{lambda_:g}",
                    "auroc": uniform_auc,
                    "auprc": metric(labels, scores[uniform_key], average_precision_score),
                    "delta_pp": 100.0 * (uniform_auc - baseline_auc),
                })
        for lambda_ in LAMBDAS:
            key = f"ridge__lambda_{lambda_:g}"
            auc = metric(labels, scores[key], roc_auc_score)
            control_rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": family(cell),
                "method": f"projected_ridge_lambda{lambda_:g}",
                "auroc": auc,
                "auprc": metric(labels, scores[key], average_precision_score),
                "delta_pp": 100.0 * (auc - baseline_auc),
            })
        control_rows.append({
            "cell": cell,
            "domain": GROUP[cell],
            "family": family(cell),
            "method": "iu_pcr",
            "auroc": baseline_auc,
            "auprc": baseline_ap,
            "delta_pp": 0.0,
        })
        primary = [
            row for row in atomic_rows
            if row["cell"] == cell
            and row["graph_k"] == PRIMARY_GRAPH_K
            and row["lambda"] == PRIMARY_LAMBDA
            and row["valid_operator"]
        ]
        if len(primary) < 4:
            raise RuntimeError(f"fewer than four valid atomic operators in {cell}")
        primary_by_cell[cell] = primary
        proxy = np.asarray([row["primary_proxy"] for row in primary])
        delta = np.asarray([row["delta_pp"] for row in primary])
        bottom, top = tie_aware_quartiles(proxy)
        maximum = float(np.max(proxy))
        tied_maximum = np.where(np.isclose(proxy, maximum, atol=1e-15, rtol=0.0))[0]
        chosen = min(tied_maximum, key=lambda index: primary[index]["feature"])
        oracle = int(np.argmax(delta))
        for label, index in (("top_proxy_atomic", chosen), ("oracle_atomic", oracle)):
            row = primary[index]
            control_rows.append({
                "cell": cell,
                "domain": GROUP[cell],
                "family": family(cell),
                "method": label,
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "delta_pp": row["delta_pp"],
                "feature": row["feature"],
            })
        primary_by_cell[cell] = {
            "rows": primary,
            "spearman": safe_spearman(proxy, delta),
            "top_bottom_pp": (
                float(np.mean(delta[top]) - np.mean(delta[bottom]))
                if len(top) and len(bottom) else float("nan")
            ),
            "proxy_ridge_spearman": safe_spearman(
                proxy, [row["distance_from_ridge"] for row in primary]
            ),
            "top_feature": primary[chosen]["feature"],
            "top_proxy_tie_count": int(len(tied_maximum)),
            "top_feature_delta_pp": float(delta[chosen]),
            "oracle_feature": primary[oracle]["feature"],
            "oracle_delta_pp": float(delta[oracle]),
        }
    return atomic_rows, control_rows, convergence_rows, primary_by_cell


def summarize(
    atomic_rows, control_rows, convergence_rows, primary_by_cell,
    randomization_count=5000,
):
    cell_rows = []
    for cell in INSCOPE:
        item = primary_by_cell[cell]
        cell_rows.append({
            "cell": cell,
            "domain": GROUP[cell],
            "family": family(cell),
            **{key: value for key, value in item.items() if key != "rows"},
        })
    family_rows = []
    for name in FAMILY_NAMES:
        local = [row for row in cell_rows if row["family"] == name]
        primary = [row for cell in local for row in primary_by_cell[cell["cell"]]["rows"]]
        family_rows.append({
            "family": name,
            "n_cells": len(local),
            "mean_cell_spearman": float(np.nanmean([row["spearman"] for row in local])),
            "mean_top_bottom_pp": float(np.nanmean([row["top_bottom_pp"] for row in local])),
            "partial_spearman": partial_rank_association(primary),
            "mean_top_feature_delta_pp": float(np.mean([row["top_feature_delta_pp"] for row in local])),
            "mean_oracle_delta_pp": float(np.mean([row["oracle_delta_pp"] for row in local])),
        })
    undefined_cell_associations = int(np.sum([
        not np.isfinite(row["spearman"]) for row in cell_rows
    ]))
    undefined_top_bottom = int(np.sum([
        not np.isfinite(row["top_bottom_pp"]) for row in cell_rows
    ]))
    association_values = np.asarray(
        [row["mean_cell_spearman"] for row in family_rows], dtype=float
    )
    partial_values = np.asarray(
        [row["partial_spearman"] for row in family_rows], dtype=float
    )
    association_for_inference = np.where(np.isfinite(association_values), association_values, 0.0)
    partial_for_inference = np.where(np.isfinite(partial_values), partial_values, 0.0)
    association_ci = bootstrap_mean_ci(
        association_for_inference, "atomic-primary-association"
    )
    partial_ci = bootstrap_mean_ci(partial_for_inference, "atomic-primary-partial")
    finite_cell = [row["spearman"] for row in cell_rows if np.isfinite(row["spearman"])]
    median_cell = float(np.median(finite_cell)) if finite_cell else 0.0
    positive_families = int(sum(row["mean_top_bottom_pp"] > 0 for row in family_rows))
    ridge_values = [
        abs(row["proxy_ridge_spearman"]) for row in cell_rows
        if np.isfinite(row["proxy_ridge_spearman"])
    ]
    median_abs_ridge = float(np.median(ridge_values)) if ridge_values else 1.0
    primary_rows = [
        row for cell in INSCOPE for row in primary_by_cell[cell]["rows"]
    ]
    global_partial = partial_rank_association(primary_rows)
    association_permutation_p = feature_identity_permutation_p(
        primary_by_cell,
        float(np.mean(association_for_inference)),
        count=randomization_count,
    )
    association_signflip_p = exact_sign_flip_p(association_for_inference)
    partial_permutation_p = freedman_lane_p(
        primary_rows, global_partial, count=randomization_count
    )
    partial_signflip_p = exact_sign_flip_p(partial_for_inference)

    def method_family_values(method):
        rows = [row for row in control_rows if row["method"] == method]
        return np.asarray([
            np.mean([row["delta_pp"] for row in rows if row["family"] == name])
            for name in FAMILY_NAMES
        ], dtype=float), rows

    top_family, top_rows = method_family_values("top_proxy_atomic")
    oracle_family, oracle_rows = method_family_values("oracle_atomic")
    top_ci = bootstrap_mean_ci(top_family, "top-proxy-absolute-headroom")
    oracle_ci = bootstrap_mean_ci(oracle_family, "oracle-atomic-headroom")
    top_wins = int(np.sum([row["delta_pp"] > 1e-12 for row in top_rows]))
    top_worst = float(np.min([row["delta_pp"] for row in top_rows]))
    top_signflip_p = exact_sign_flip_p(top_family)
    gates = [
        {
            "gate": "all primary cell associations and quartile contrasts are defined",
            "observed": undefined_cell_associations + undefined_top_bottom,
            "passed": bool(undefined_cell_associations == 0 and undefined_top_bottom == 0),
        },
        {
            "gate": "median within-cell Spearman > 0",
            "observed": median_cell,
            "passed": bool(median_cell > 0),
        },
        {
            "gate": "family-bootstrap association lower > 0",
            "observed": association_ci[0],
            "passed": bool(association_ci[0] > 0),
        },
        {
            "gate": "feature-identity permutation p <= 0.05",
            "observed": association_permutation_p,
            "passed": bool(association_permutation_p <= 0.05),
        },
        {
            "gate": "eight-family association sign-flip p <= 0.05",
            "observed": association_signflip_p,
            "passed": bool(association_signflip_p <= 0.05),
        },
        {
            "gate": "positive top-minus-bottom in at least 6 of 8 families",
            "observed": positive_families,
            "passed": bool(positive_families >= 6),
        },
        {
            "gate": "partial-association family-bootstrap lower > 0",
            "observed": partial_ci[0],
            "passed": bool(partial_ci[0] > 0),
        },
        {
            "gate": "partial Freedman-Lane p <= 0.05",
            "observed": partial_permutation_p,
            "passed": bool(partial_permutation_p <= 0.05),
        },
        {
            "gate": "eight-family partial sign-flip p <= 0.05",
            "observed": partial_signflip_p,
            "passed": bool(partial_signflip_p <= 0.05),
        },
        {
            "gate": "median abs(proxy, ridge-distance Spearman) < 0.8",
            "observed": median_abs_ridge,
            "passed": bool(median_abs_ridge < 0.8),
        },
        {
            "gate": "top-proxy atomic family-bootstrap AUROC lower > 0",
            "observed": top_ci[0],
            "passed": bool(top_ci[0] > 0),
        },
        {
            "gate": "top-proxy atomic family sign-flip p <= 0.05",
            "observed": top_signflip_p,
            "passed": bool(top_signflip_p <= 0.05),
        },
        {
            "gate": "top-proxy atomic improves at least 14 of 24 cells",
            "observed": top_wins,
            "passed": bool(top_wins >= 14),
        },
        {
            "gate": "top-proxy atomic worst loss no worse than -2pp",
            "observed": top_worst,
            "passed": bool(top_worst >= -2.0),
        },
        {
            "gate": "oracle atomic family-bootstrap AUROC lower > 0",
            "observed": oracle_ci[0],
            "passed": bool(oracle_ci[0] > 0),
        },
    ]
    executable_gate_names = [row["gate"] for row in gates]
    if (
        len(executable_gate_names) != len(set(executable_gate_names))
        or set(executable_gate_names) != set(CONTINUATION_GATES)
    ):
        raise RuntimeError(
            "executable continuation gates differ from the frozen gate registry"
        )

    proxy_rows = []
    for proxy in PROXIES:
        per_cell = []
        for cell in INSCOPE:
            local = [row for row in primary_rows if row["cell"] == cell]
            per_cell.append(safe_spearman(
                [row[proxy] for row in local], [row["delta_pp"] for row in local]
            ))
        family_values = [
            float(np.nanmean([value for value, cell in zip(per_cell, INSCOPE) if family(cell) == name]))
            for name in FAMILY_NAMES
        ]
        inference_values = np.where(np.isfinite(family_values), family_values, 0.0)
        lo, hi = bootstrap_mean_ci(inference_values, f"proxy-{proxy}")
        proxy_rows.append({
            "proxy": proxy,
            "median_cell_spearman": float(np.nanmedian(per_cell)),
            "family_mean_spearman": float(np.nanmean(family_values)),
            "family_ci_low": lo,
            "family_ci_high": hi,
        })

    sensitivity_rows = []
    for graph_k in GRAPH_KS:
        for lambda_ in LAMBDAS:
            local_rows = [
                row for row in atomic_rows
                if row["graph_k"] == graph_k and row["lambda"] == lambda_
            ]
            correlations, top_bottom = [], []
            for cell in INSCOPE:
                local = [
                    row for row in local_rows
                    if row["cell"] == cell and row["valid_operator"]
                ]
                proxy = np.asarray([row["path_proxy"] for row in local])
                delta = np.asarray([row["delta_pp"] for row in local])
                correlations.append(safe_spearman(proxy, delta))
                bottom, top = tie_aware_quartiles(proxy)
                top_bottom.append(
                    float(np.mean(delta[top]) - np.mean(delta[bottom]))
                    if len(top) and len(bottom) else float("nan")
                )
            sensitivity_rows.append({
                "graph_k": graph_k,
                "lambda": lambda_,
                "median_cell_spearman": float(np.nanmedian(correlations)),
                "mean_top_bottom_pp": float(np.nanmean(top_bottom)),
                "positive_cells": int(np.sum(np.asarray(top_bottom) > 0)),
            })

    duplicate_sensitivity_rows = []
    for threshold, field in (
        (0.90, "alignment_duplicate_0p9"),
        (0.95, "full_alignment"),
        (0.99, "alignment_duplicate_0p99"),
    ):
        per_cell = []
        for cell in INSCOPE:
            local = primary_by_cell[cell]["rows"]
            per_cell.append(safe_spearman(
                [row[field] for row in local],
                [row["delta_pp"] for row in local],
            ))
        family_values = np.asarray([
            np.nanmean([
                value for value, cell in zip(per_cell, INSCOPE)
                if family(cell) == name
            ])
            for name in FAMILY_NAMES
        ], dtype=float)
        inference_values = np.where(np.isfinite(family_values), family_values, 0.0)
        lo, hi = bootstrap_mean_ci(
            inference_values, f"duplicate-threshold-{threshold}"
        )
        duplicate_sensitivity_rows.append({
            "duplicate_threshold": threshold,
            "median_cell_spearman": float(np.nanmedian(per_cell)),
            "family_mean_spearman": float(np.mean(inference_values)),
            "family_ci_low": lo,
            "family_ci_high": hi,
            "scope": "full-alignment component only; not the complete proxy",
        })

    convergence_summary = []
    checkpoints = sorted({row["replicates"] for row in convergence_rows})
    for checkpoint in checkpoints:
        correlations = []
        for cell in INSCOPE:
            partial = [
                row for row in convergence_rows
                if row["cell"] == cell and row["replicates"] == checkpoint
                and row["graph_k"] == PRIMARY_GRAPH_K
                and row["lambda"] == PRIMARY_LAMBDA
            ]
            final = [
                row for row in convergence_rows
                if row["cell"] == cell and row["replicates"] == max(checkpoints)
                and row["graph_k"] == PRIMARY_GRAPH_K
                and row["lambda"] == PRIMARY_LAMBDA
            ]
            final_map = {row["feature"]: row["primary_proxy"] for row in final}
            correlations.append(safe_spearman(
                [row["primary_proxy"] for row in partial],
                [final_map[row["feature"]] for row in partial],
            ))
        convergence_summary.append({
            "replicates": checkpoint,
            "median_proxy_spearman_vs_final": float(np.nanmedian(correlations)),
            "min_proxy_spearman_vs_final": float(np.nanmin(correlations)),
        })

    control_summary = []
    methods = sorted({row["method"] for row in control_rows})
    for method in methods:
        local = [row for row in control_rows if row["method"] == method]
        control_summary.append({
            "method": method,
            "cell_macro_auroc": float(np.mean([row["auroc"] for row in local])),
            "mean_delta_pp": float(np.mean([row["delta_pp"] for row in local])),
            "wins": int(np.sum([row["delta_pp"] > 1e-12 for row in local])),
            "losses": int(np.sum([row["delta_pp"] < -1e-12 for row in local])),
            "worst_delta_pp": float(np.min([row["delta_pp"] for row in local])),
        })
    summary = {
        "median_cell_spearman": median_cell,
        "family_mean_spearman": float(np.mean(association_for_inference)),
        "family_association_ci": association_ci,
        "association_feature_permutation_p": association_permutation_p,
        "association_family_signflip_p": association_signflip_p,
        "family_mean_partial_spearman": float(np.mean(partial_for_inference)),
        "family_partial_ci": partial_ci,
        "global_partial_spearman": (
            float(global_partial) if np.isfinite(global_partial) else None
        ),
        "partial_freedman_lane_p": partial_permutation_p,
        "partial_family_signflip_p": partial_signflip_p,
        "positive_top_bottom_families": positive_families,
        "median_abs_proxy_ridge_distance_spearman": median_abs_ridge,
        "undefined_cell_associations": undefined_cell_associations,
        "undefined_top_bottom": undefined_top_bottom,
        "top_proxy_family_delta_pp": float(np.mean(top_family)),
        "top_proxy_family_ci": top_ci,
        "top_proxy_family_signflip_p": top_signflip_p,
        "top_proxy_wins": top_wins,
        "top_proxy_worst_delta_pp": top_worst,
        "oracle_family_delta_pp": float(np.mean(oracle_family)),
        "oracle_family_ci": oracle_ci,
        "invalid_primary_operators": int(np.sum([
            not row["valid_operator"] for row in atomic_rows
            if row["graph_k"] == PRIMARY_GRAPH_K
            and row["lambda"] == PRIMARY_LAMBDA
        ])),
        "all_gates_passed": bool(all(row["passed"] for row in gates)),
    }
    return (
        summary, gates, cell_rows, family_rows, proxy_rows,
        sensitivity_rows, duplicate_sensitivity_rows, convergence_summary,
        control_summary,
    )


def plots(out_dir, atomic_rows, cell_rows, family_rows, sensitivity_rows,
          convergence_summary, control_summary):
    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    primary = [
        row for row in atomic_rows
        if row["graph_k"] == PRIMARY_GRAPH_K and row["lambda"] == PRIMARY_LAMBDA
        and row["valid_operator"]
    ]
    x, y = [], []
    for cell in INSCOPE:
        local = [row for row in primary if row["cell"] == cell]
        x.extend(rankdata([row["primary_proxy"] for row in local]) / len(local))
        y.extend(rankdata([row["delta_pp"] for row in local]) / len(local))
    plt.figure(figsize=(6.5, 5))
    plt.scatter(x, y, s=12, alpha=0.35)
    coefficient = np.polyfit(x, y, 1)
    grid = np.linspace(0, 1, 100)
    plt.plot(grid, coefficient[0] * grid + coefficient[1], color="black")
    plt.xlabel("Primary proxy rank within cell")
    plt.ylabel("Atomic usefulness rank within cell")
    plt.title("Does the frozen label-free proxy predict usefulness?")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "proxy_vs_utility.png"), dpi=180)
    plt.close()

    ordered = sorted(cell_rows, key=lambda row: row["spearman"])
    plt.figure(figsize=(9, 6))
    colors = ["#2a9d8f" if row["spearman"] > 0 else "#e76f51" for row in ordered]
    plt.barh(range(len(ordered)), [row["spearman"] for row in ordered], color=colors)
    plt.yticks(range(len(ordered)), [row["cell"] for row in ordered], fontsize=7)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Spearman(proxy, atomic AUROC change)")
    plt.title("Primary association in each dataset/model cell")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "cell_associations.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    colors = ["#2a9d8f" if row["mean_top_bottom_pp"] > 0 else "#e76f51" for row in family_rows]
    plt.bar([row["family"] for row in family_rows], [row["mean_top_bottom_pp"] for row in family_rows], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Top-proxy minus bottom-proxy usefulness (pp)")
    plt.xticks(rotation=35, ha="right")
    plt.title("Transfer across eight dataset families")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "family_top_bottom.png"), dpi=180)
    plt.close()

    matrix = np.empty((len(GRAPH_KS), len(LAMBDAS)))
    for i, graph_k in enumerate(GRAPH_KS):
        for j, lambda_ in enumerate(LAMBDAS):
            row = next(
                item for item in sensitivity_rows
                if item["graph_k"] == graph_k and item["lambda"] == lambda_
            )
            matrix[i, j] = row["median_cell_spearman"]
    plt.figure(figsize=(6, 4))
    image = plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, label="Median cell Spearman")
    plt.xticks(range(len(LAMBDAS)), [str(value) for value in LAMBDAS])
    plt.yticks(range(len(GRAPH_KS)), [str(value) for value in GRAPH_KS])
    plt.xlabel("lambda")
    plt.ylabel("graph k")
    plt.title("Registered sensitivity path (diagnostic only)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "k_lambda_sensitivity.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(
        [row["replicates"] for row in convergence_summary],
        [row["median_proxy_spearman_vs_final"] for row in convergence_summary],
        marker="o",
        label="median cell",
    )
    plt.plot(
        [row["replicates"] for row in convergence_summary],
        [row["min_proxy_spearman_vs_final"] for row in convergence_summary],
        marker="o",
        label="worst cell",
    )
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Stability subsamples")
    plt.ylabel("Proxy rank agreement with 40-subsample result")
    plt.title("Label-free proxy convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "proxy_convergence.png"), dpi=180)
    plt.close()

    wanted = (
        "iu_pcr", "projected_ridge_lambda1", "uniform_atomic_k15_lambda1",
        "top_proxy_atomic", "oracle_atomic",
    )
    rows = [next(item for item in control_summary if item["method"] == name) for name in wanted]
    plt.figure(figsize=(7, 4.5))
    plt.bar([row["method"] for row in rows], [row["mean_delta_pp"] for row in rows])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Mean AUROC change versus IU-PCR (pp)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Practical and oracle atomic headroom")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "atomic_headroom.png"), dpi=180)
    plt.close()


def report_markdown(summary, gates, family_rows, proxy_rows, sensitivity_rows,
                    duplicate_sensitivity_rows, convergence_summary,
                    control_summary, complete):
    status = "CONTINUE TO PHASE 1" if summary["all_gates_passed"] else "STOP AOG FOR THIS PROXY"
    controls = {row["method"]: row for row in control_summary}
    lines = [
        "# Frozen atomic-operator premise audit",
        "",
        f"**Decision: {status}.**",
        "",
        "## Terms",
        "",
        "- **Atomic operator:** the Laplacian penalty created by one feature.",
        "- **Proxy:** the registered score computed without correctness labels.",
        "- **Usefulness:** AUROC change when that operator is added to IU-PCR.",
        "- **Spearman association:** whether higher proxy ranks correspond to higher usefulness ranks; 1 is perfect, 0 is no monotone relation, and -1 is reversed.",
        "- **pp:** AUROC percentage points; +0.5pp means AUROC increases by 0.005.",
        "",
        "## Primary result",
        "",
        f"The median within-cell proxy/usefulness association is **{summary['median_cell_spearman']:+.3f}**.",
        f"The equal-family mean is **{summary['family_mean_spearman']:+.3f}**, with bootstrap interval "
        f"[{summary['family_association_ci'][0]:+.3f}, {summary['family_association_ci'][1]:+.3f}].",
        f"The within-cell feature-identity permutation p-value is **{summary['association_feature_permutation_p']:.4f}** and the exact eight-family sign-flip p-value is **{summary['association_family_signflip_p']:.4f}**.",
        f"Top-proxy operators beat bottom-proxy operators in **{summary['positive_top_bottom_families']}/8 families**.",
        f"After nuisance adjustment, the family mean partial association is **{summary['family_mean_partial_spearman']:+.3f}**, interval "
        f"[{summary['family_partial_ci'][0]:+.3f}, {summary['family_partial_ci'][1]:+.3f}].",
        f"The selected top-proxy atom changes equal-family AUROC by **{summary['top_proxy_family_delta_pp']:+.3f}pp**, interval [{summary['top_proxy_family_ci'][0]:+.3f}, {summary['top_proxy_family_ci'][1]:+.3f}].",
        f"The label-only atomic oracle changes equal-family AUROC by **{summary['oracle_family_delta_pp']:+.3f}pp**, interval [{summary['oracle_family_ci'][0]:+.3f}, {summary['oracle_family_ci'][1]:+.3f}].",
        f"Order-invariant unique-value quotient graphs marked **{summary['invalid_primary_operators']}** cell-feature operators invalid because they had fewer than three unique values.",
        "",
        "## Continuation gates",
        "",
        "| gate | observed | pass |",
        "|---|---:|:---:|",
    ]
    for row in gates:
        value = row["observed"]
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        lines.append(f"| {row['gate']} | {rendered} | {'yes' if row['passed'] else 'no'} |")
    lines.extend([
        "",
        "## Practical headroom",
        "",
        "| method | cell-macro AUROC | change vs IU-PCR | wins/losses | worst |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in (
        "iu_pcr", "projected_ridge_lambda1", "uniform_atomic_k15_lambda1",
        "top_proxy_atomic", "oracle_atomic",
    ):
        row = controls[key]
        lines.append(
            f"| {key} | {row['cell_macro_auroc']:.4f} | {row['mean_delta_pp']:+.3f}pp | "
            f"{row['wins']}/{row['losses']} | {row['worst_delta_pp']:+.3f}pp |"
        )
    lines.extend([
        "",
        "The oracle is a label-only headroom diagnostic. It is not a usable method.",
        "",
        "## Family diagnosis",
        "",
        "| family | cells | proxy association | top-bottom usefulness | partial association |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['n_cells']} | {row['mean_cell_spearman']:+.3f} | "
            f"{row['mean_top_bottom_pp']:+.3f}pp | {row['partial_spearman']:+.3f} |"
        )
    lines.extend([
        "",
        "## What the proxy components say",
        "",
        "| label-free quantity | median cell association | family interval |",
        "|---|---:|---:|",
    ])
    for row in proxy_rows:
        lines.append(
            f"| {row['proxy']} | {row['median_cell_spearman']:+.3f} | "
            f"[{row['family_ci_low']:+.3f}, {row['family_ci_high']:+.3f}] |"
        )
    lines.extend([
        "",
        "### Duplicate-threshold diagnostic",
        "",
        "This table changes only the full cross-fitted alignment component. It is not a rerun of the complete proxy and cannot replace the registered 0.95 threshold.",
        "",
        "| threshold | median cell association | family interval |",
        "|---:|---:|---:|",
    ])
    for row in duplicate_sensitivity_rows:
        lines.append(
            f"| {row['duplicate_threshold']:.2f} | {row['median_cell_spearman']:+.3f} | "
            f"[{row['family_ci_low']:+.3f}, {row['family_ci_high']:+.3f}] |"
        )
    best_sensitivity = max(sensitivity_rows, key=lambda row: row["median_cell_spearman"])
    final_convergence = convergence_summary[-1]
    lines.extend([
        "",
        "## Parameter sensitivity",
        "",
        f"The largest descriptive median association on the frozen grid is {best_sensitivity['median_cell_spearman']:+.3f} at "
        f"k={best_sensitivity['graph_k']}, lambda={best_sensitivity['lambda']:g}. This is not a selected replacement for the primary setting.",
        f"At {final_convergence['replicates']} subsamples the convergence reference is, by definition, 1.0; earlier checkpoints are shown in `figures/proxy_convergence.png`.",
        "",
        "Parameters that can change the mechanism are graph neighbourhood size `k`, Laplacian strength `lambda`, duplicate threshold, and the stability sampling budget. They may be changed only in a newly registered run. If the primary association is absent across the sensitivity grid, more tuning cannot solve the missing target-identification problem.",
        "",
        "## Conclusion",
        "",
    ])
    if summary["all_gates_passed"]:
        lines.extend([
            "The registered label-free proxy carries transferable information about atomic operator usefulness on these development families. The next step may implement a global gate learner, while keeping labels outside fitting and reserving new families for confirmation.",
        ])
    else:
        lines.extend([
            "The registered premise did not pass. Do not build or tune AOG-IU-PCR from this proxy. The failure means that reproducibility, cross-fitted smoothness, and actuation do not jointly identify correctness-relevant atomic graphs well enough on the existing cells. Continue only if a new source of self-supervision supplies a different, theoretically justified target; do not rescue the result by selecting the best observed k or lambda.",
        ])
    lines.extend([
        "",
        "## Audit",
        "",
        f"The fit completed 24 cells in {complete['runtime_seconds'] / 60.0:.1f} minutes. Source and artifact hashes were verified before labels were read. The 24 cells are retrospective development data, not external confirmation.",
        "",
        "Figures: `figures/proxy_vs_utility.png`, `cell_associations.png`, `family_top_bottom.png`, `proxy_convergence.png`, `k_lambda_sensitivity.png`, and `atomic_headroom.png`.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    bundle = os.path.abspath(args.bundle)
    out_dir = os.path.abspath(args.out_dir)
    _, complete = verify_freeze(out_dir, bundle)
    atomic_rows, control_rows, convergence_rows, primary_by_cell = evaluate(out_dir, bundle)
    outputs = summarize(atomic_rows, control_rows, convergence_rows, primary_by_cell)
    (
        summary, gates, cell_rows, family_rows, proxy_rows,
        sensitivity_rows, duplicate_sensitivity_rows, convergence_summary,
        control_summary,
    ) = outputs
    write_csv(os.path.join(out_dir, "atomic_metrics.csv"), atomic_rows)
    write_csv(os.path.join(out_dir, "control_metrics.csv"), control_rows)
    write_csv(os.path.join(out_dir, "cell_associations.csv"), cell_rows)
    write_csv(os.path.join(out_dir, "family_summary.csv"), family_rows)
    write_csv(os.path.join(out_dir, "proxy_summary.csv"), proxy_rows)
    write_csv(os.path.join(out_dir, "sensitivity_summary.csv"), sensitivity_rows)
    write_csv(
        os.path.join(out_dir, "duplicate_threshold_summary.csv"),
        duplicate_sensitivity_rows,
    )
    write_csv(os.path.join(out_dir, "proxy_convergence.csv"), convergence_summary)
    write_csv(os.path.join(out_dir, "control_summary.csv"), control_summary)
    write_json(os.path.join(out_dir, "CONTINUATION_GATES.json"), {
        "summary": summary,
        "gates": gates,
    })
    plots(
        out_dir, atomic_rows, cell_rows, family_rows, sensitivity_rows,
        convergence_summary, control_summary,
    )
    report = report_markdown(
        summary, gates, family_rows, proxy_rows, sensitivity_rows,
        duplicate_sensitivity_rows, convergence_summary, control_summary,
        complete,
    )
    with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
