#!/usr/bin/env python3
"""Evaluate the frozen confidence-oriented feature contract without raw data.

The committed dependency-fusion bundle contains the old oriented matrix ``V``
and its exact historical hand-sign vector.  Their product reconstructs the raw
z-scored features, so this script can compare four orientation policies without
re-running feature extraction:

``signrho``
    The deployed two-pass per-cell orientation (the stored ``F`` matrix).
``fixed_all_v1`` / ``fixed_stable_v1``
    The frozen feature contract, with all views or with the four unstable raw
    views quarantined.  No per-cell polarity estimation is performed.
``lofo_*``
    Diagnostic only: learn one sign per feature on the other dataset families,
    then score the held-out family.  Target-family labels never choose a sign.

Labels enter only after each method score is frozen, except in the explicitly
named leave-one-family-out calibration diagnostic.  The frozen-v1 arms are
retrospective on these cells and require a genuinely new family for external
confirmation.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.dependency_fusion import sparse_upcr_fit          # noqa: E402
from spectral_utils.feature_contract import (                         # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
    SCHEMA_VERSION,
    confidence_oriented_matrix,
    consensus_anchor,
)
from spectral_utils.streaming_utils import anchor_orient              # noqa: E402
from spectral_utils.upcr import upcr_fit                               # noqa: E402


VERSION = "fixed-orientation-validation-v1"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "fixed_orientation_validation")

INCUMBENT_FIT = dict(
    loss="l2", exclusion=True, difficulty_gate=False,
    simple_avg_fallback=True, recompute_after_exclusion=True,
    g2_projection_k=1, scale_ratio=0.25,
)
SPARSE_FIT = dict(
    scale_ratio=0.25, rank=2, n_components=2,
    g2_projection_components=1, threshold_multiplier=1.0,
    max_iter=100, inner_completion_iter=40, decomposition_tol=1e-8,
    max_sparse_fraction=None, target_condition=100.0,
)

KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)
METHODS = ("upcr", "su_pcr", "sdsf")
ARMS = (
    "signrho",
    "fixed_all_v1",
    "fixed_stable_v1",
    "lofo_all_diagnostic",
    "lofo_stable_diagnostic",
)


def dataset_family(cell_key):
    return next((name for name in KNOWN_FAMILIES if name in cell_key), cell_key)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_bundle(path):
    data = np.load(path, allow_pickle=True)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    cells = {}
    for key in keys:
        pool = [str(name) for name in data[f"{key}__pool"]]
        legacy = np.asarray(data[f"{key}__hand_signs"], dtype=float)
        old_v = np.asarray(data[f"{key}__V"], dtype=float)
        raw = old_v * legacy
        stored_f = np.asarray(data[f"{key}__F"], dtype=float)
        polarity = np.asarray(data[f"{key}__rho_polarity"], dtype=float)
        reconstruction_error = float(np.max(np.abs(stored_f - (raw * polarity).T)))
        if reconstruction_error > 1e-10:
            raise RuntimeError(f"{key}: stored F reconstruction error {reconstruction_error:.3e}")
        missing = sorted(set(pool) - set(CONFIDENCE_FEATURE_SIGNS_V1))
        if missing:
            raise RuntimeError(f"{key}: unregistered raw features: {missing}")
        cells[key] = {
            "pool": pool,
            "raw": raw,
            "F": stored_f,
            "labels": np.asarray(data[f"{key}__labels"], dtype=int),
            "epr_anchor": np.asarray(data[f"{key}__anchor"], dtype=float),
            "family": dataset_family(key),
        }
    return cells


def family_feature_auc(cells):
    values = {}
    families = sorted({cell["family"] for cell in cells.values()})
    features = sorted({name for cell in cells.values() for name in cell["pool"]})
    for feature in features:
        for family in families:
            aucs = []
            for cell in cells.values():
                if cell["family"] != family or feature not in cell["pool"]:
                    continue
                j = cell["pool"].index(feature)
                aucs.append(roc_auc_score(cell["labels"], cell["raw"][:, j]))
            if aucs:
                values[feature, family] = float(np.mean(aucs))
    return values, families, features


def lofo_signs(heldout_family, family_auc, families, features):
    signs = {}
    for feature in features:
        values = [family_auc[feature, family] for family in families
                  if family != heldout_family and (feature, family) in family_auc]
        if not values:
            raise RuntimeError(f"no training-family direction for {feature}")
        signs[feature] = 1.0 if float(np.mean(values)) >= 0.5 else -1.0
    return signs


def orient_with_signs(raw, pool, signs, *, stable):
    keep = [i for i, name in enumerate(pool)
            if not stable or name not in FIXED_STABLE_EXCLUDED_V1]
    names = [pool[i] for i in keep]
    vector = np.asarray([signs[name] for name in names], dtype=float)
    return raw[:, keep] * vector, names


def tail_fraction(F, rho):
    covariance = (F @ F.T) / F.shape[1]
    values, vectors = np.linalg.eigh(covariance)
    top = vectors[:, np.argsort(values)[::-1][:2]]
    head = top @ (top.T @ rho)
    return float(np.linalg.norm(rho - head) / (np.linalg.norm(rho) + 1e-12))


def score_matrix(F, labels, anchor):
    upcr = upcr_fit(F, **INCUMBENT_FIT)
    sparse = sparse_upcr_fit(F, **SPARSE_FIT)
    weights = {
        "upcr": upcr.w,
        "su_pcr": sparse.w_pcr,
        "sdsf": sparse.w_structured,
    }
    aucs = {}
    for method, weight in weights.items():
        raw_score = np.asarray(weight @ F, dtype=float)
        score, _ = anchor_orient(raw_score, anchor)
        aucs[method] = float(roc_auc_score(labels, score))
    return aucs, sparse


def direction_audit(cells, family_auc, families, features):
    rows = []
    for feature in features:
        values = np.asarray([
            family_auc[feature, family] for family in families
            if (feature, family) in family_auc
        ])
        global_sign = 1 if float(values.mean()) >= 0.5 else -1
        strong = values[np.abs(values - 0.5) >= 0.01]
        agreement = (float(np.mean(np.sign(strong - 0.5) == global_sign))
                     if len(strong) else 0.0)
        lofo = []
        for heldout in families:
            train = np.asarray([
                family_auc[feature, family] for family in families
                if family != heldout and (feature, family) in family_auc
            ])
            if len(train):
                lofo.append(1 if float(train.mean()) >= 0.5 else -1)
        contract = int(CONFIDENCE_FEATURE_SIGNS_V1[feature])
        rows.append({
            "feature": feature,
            "contract_sign": contract,
            "equal_family_raw_auc": float(values.mean()),
            "global_empirical_sign": global_sign,
            "contract_matches_global": int(contract == global_sign),
            "strong_family_direction_agreement": agreement,
            "n_families": len(values),
            "n_strong_families": len(strong),
            "lofo_sign_stable": int(len(set(lofo)) == 1),
            "quarantined_in_stable_v1": int(feature in FIXED_STABLE_EXCLUDED_V1),
        })
    return sorted(rows, key=lambda row: row["equal_family_raw_auc"])


def summarize(rows, families):
    summary_rows = []
    for method in METHODS:
        for arm in ARMS:
            selected = [row for row in rows if row["method"] == method and row["arm"] == arm]
            values = np.asarray([row["auroc"] for row in selected], dtype=float)
            family_values = [
                np.mean([row["auroc"] for row in selected if row["family"] == family])
                for family in families
            ]
            summary_rows.append({
                "method": method,
                "arm": arm,
                "n_cells": len(values),
                "cell_macro": float(values.mean()),
                "family_macro": float(np.mean(family_values)),
                "mean_n_features": float(np.mean([row["n_features"] for row in selected])),
            })
    return summary_rows


def contrasts(rows, families):
    out = []
    for method in METHODS:
        baseline = {
            row["cell"]: row for row in rows
            if row["method"] == method and row["arm"] == "signrho"
        }
        for arm in ARMS[1:]:
            candidate = {
                row["cell"]: row for row in rows
                if row["method"] == method and row["arm"] == arm
            }
            deltas = np.asarray([
                candidate[cell]["auroc"] - baseline[cell]["auroc"]
                for cell in sorted(baseline)
            ])
            family_delta = np.asarray([
                np.mean([
                    candidate[cell]["auroc"] - baseline[cell]["auroc"]
                    for cell in baseline if baseline[cell]["family"] == family
                ]) for family in families
            ])
            p = (float(wilcoxon(family_delta).pvalue)
                 if np.any(np.abs(family_delta) > 0) else 1.0)
            out.append({
                "method": method,
                "reference": "signrho",
                "candidate": arm,
                "mean_delta_pp": float(deltas.mean() * 100),
                "family_macro_delta_pp": float(family_delta.mean() * 100),
                "wins": int(np.sum(deltas > 1e-12)),
                "losses": int(np.sum(deltas < -1e-12)),
                "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                "family_wilcoxon_p": p,
                "worst_cell_delta_pp": float(deltas.min() * 100),
            })
    return out


def render(summary):
    lines = [
        "# Fixed confidence-orientation validation",
        "",
        f"Feature schema: `{SCHEMA_VERSION}`.",
        "",
        "The fixed arms do not estimate per-cell feature polarity. The LOFO arms are a "
        "cross-family calibration diagnostic; the frozen-v1 arms are retrospective and "
        "must be confirmed on a new dataset/model family.",
        "",
        "## Method scores",
        "",
        "| method | arm | cell macro | equal-family macro | mean views |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["method_summary"]:
        lines.append(
            f"| `{row['method']}` | `{row['arm']}` | {row['cell_macro']:.4f} | "
            f"{row['family_macro']:.4f} | {row['mean_n_features']:.1f} |"
        )
    lines.extend([
        "",
        "## Contrasts against per-cell sign(rho)",
        "",
        "| method | candidate | cell delta | family delta | W/L/T | worst cell |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary["contrasts"]:
        lines.append(
            f"| `{row['method']}` | `{row['candidate']}` | "
            f"{row['mean_delta_pp']:+.2f}pp | {row['family_macro_delta_pp']:+.2f}pp | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | "
            f"{row['worst_cell_delta_pp']:+.2f}pp |"
        )
    lines.extend([
        "",
        "## Global-sign check",
        "",
        f"For fixed-schema scores, the consensus anchor and historical `epr` anchor "
        f"selected the same sign in **{summary['global_sign_agreement']['matches']}/"
        f"{summary['global_sign_agreement']['comparisons']}** method/cell/arm comparisons.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    started = time.time()
    cells = load_bundle(args.raw_bundle)
    family_auc, families, features = family_feature_auc(cells)
    audit = direction_audit(cells, family_auc, families, features)
    rows = []
    sign_matches = 0
    sign_comparisons = 0

    for cell_key, cell in cells.items():
        fixed_all, fixed_all_names, _ = confidence_oriented_matrix(
            cell["raw"], cell["pool"], stable=False,
        )
        fixed_stable, fixed_stable_names, _ = confidence_oriented_matrix(
            cell["raw"], cell["pool"], stable=True,
        )
        learned = lofo_signs(cell["family"], family_auc, families, features)
        lofo_all, lofo_all_names = orient_with_signs(
            cell["raw"], cell["pool"], learned, stable=False,
        )
        lofo_stable, lofo_stable_names = orient_with_signs(
            cell["raw"], cell["pool"], learned, stable=True,
        )
        matrices = {
            "signrho": (cell["F"].T, cell["pool"], cell["epr_anchor"]),
            "fixed_all_v1": (
                fixed_all, fixed_all_names, consensus_anchor(fixed_all),
            ),
            "fixed_stable_v1": (
                fixed_stable, fixed_stable_names, consensus_anchor(fixed_stable),
            ),
            "lofo_all_diagnostic": (
                lofo_all, lofo_all_names, consensus_anchor(lofo_all),
            ),
            "lofo_stable_diagnostic": (
                lofo_stable, lofo_stable_names, consensus_anchor(lofo_stable),
            ),
        }
        for arm, (matrix, names, anchor) in matrices.items():
            F = matrix.T
            aucs, sparse = score_matrix(F, cell["labels"], anchor)
            if arm != "signrho":
                for weight in (upcr_fit(F, **INCUMBENT_FIT).w,
                               sparse.w_pcr, sparse.w_structured):
                    raw_score = weight @ F
                    by_consensus, _ = anchor_orient(raw_score, anchor)
                    by_epr, _ = anchor_orient(raw_score, cell["epr_anchor"])
                    sign_matches += int(np.corrcoef(by_consensus, by_epr)[0, 1] > 0)
                    sign_comparisons += 1
            rho_tail = tail_fraction(F, sparse.rho_hat)
            for method, auc in aucs.items():
                rows.append({
                    "cell": cell_key,
                    "family": cell["family"],
                    "method": method,
                    "arm": arm,
                    "auroc": auc,
                    "n_features": len(names),
                    "rho_tail_fraction": rho_tail,
                })

    method_summary = summarize(rows, families)
    contrast_rows = contrasts(rows, families)
    summary = {
        "version": VERSION,
        "feature_schema": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "raw_bundle": os.path.abspath(args.raw_bundle),
        "n_cells": len(cells),
        "n_families": len(families),
        "families": families,
        "stable_excluded": sorted(FIXED_STABLE_EXCLUDED_V1),
        "global_sign_agreement": {
            "matches": sign_matches,
            "comparisons": sign_comparisons,
        },
        "method_summary": method_summary,
        "contrasts": contrast_rows,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "per_cell.csv"), rows)
    write_csv(os.path.join(args.out_dir, "direction_audit.csv"), audit)
    write_csv(os.path.join(args.out_dir, "method_summary.csv"), method_summary)
    write_csv(os.path.join(args.out_dir, "contrasts.csv"), contrast_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(render(summary))
    print(render(summary))
    print(f"\nOutputs: {args.out_dir}")


if __name__ == "__main__":
    main()
