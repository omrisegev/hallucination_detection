#!/usr/bin/env python3
"""Frozen scorer-family confirmation for CB-CS-IU on Llama ProcessBench.

Run ``fit`` first.  It freezes and hashes every score without accessing either
target key.  ``report`` verifies the complete freeze before opening the primary
ProcessBench error-present label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_iu_fit,
    leverage_balanced_contribution_score,
)
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)


VERSION = "cardinality-balanced-llama-processbench-confirmation-v1-2026-08-12"
DEFAULT_CACHE = REPO / "dataset_cache" / "repgrid" / "pb_llama31_8b"
DEFAULT_OUT = (
    REPO / "results" / "cardinality_balanced_llama_processbench_v1"
)
SPEC = REPO / "SPEC_CARDINALITY_BALANCED_LLAMA_PROCESSBENCH_CONFIRMATION_V1.md"
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
DUFS_LAMBDA = 0.1
BOOTSTRAP_DRAWS = 20000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aligned_items(path):
    """Materialize aligned rows without reading any target field value."""
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    return [
        (str(key), cache[key])
        for key in sorted(cache)
        if not cache[key]["align_diag"]["problems"]
    ]


def direction_score(baseline, residuals, direction):
    direction = np.asarray(direction, dtype=float)
    raw = residuals @ direction
    scale = float(np.std(raw))
    correction = (
        np.zeros(len(baseline), dtype=float)
        if scale <= 1e-12 or np.linalg.norm(direction) <= 1e-12
        else raw / (len(direction) * scale)
    )
    return baseline + correction


def source_paths():
    return {
        "script": Path(__file__),
        "candidate_spec": SPEC,
        "candidate_module": (
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "feature_contract": (
            REPO / "spectral_utils" / "dufs_liu_feature_contract.py"
        ),
        "feature_registry": REPO / "spectral_utils" / "specrage_views.py",
        "dufs_module": REPO / "spectral_utils" / "adapted_dufs.py",
        "laplacian_module": REPO / "spectral_utils" / "laplacian_upcr.py",
        "mixed_v2_dependency": (
            REPO / "scripts" / "leverage_balanced_processbench_transfer.py"
        ),
    }


def fit_scores(cache_root, out):
    cache_root = Path(cache_root)
    out = Path(out)
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, data_hashes, score_hashes = [], {}, {}

    for subset in SUBSETS:
        path = cache_root / f"processbench_{subset}.pkl"
        items = aligned_items(path)
        row_ids = [key for key, _ in items]
        rows = [row for _, row in items]
        F, names, availability, contract = mixed_v2_matrix(rows)

        fitted = cardinality_balanced_iu_fit(F, names)
        primary = fitted.balanced
        leverage = leverage_balanced_contribution_score(
            fitted.contribution_space, fitted.baseline.w
        )
        baseline = primary.baseline_score
        if not np.array_equal(baseline, leverage.baseline_score):
            raise RuntimeError("candidate variants do not share an IU baseline")

        _, residuals = primary.transform.apply(
            fitted.contribution_space.baseline_score,
            fitted.contribution_space.contributions,
        )
        uniform = direction_score(
            baseline, residuals, np.ones(len(fitted.contribution_space.families))
        )
        reverse = baseline - primary.correction

        gates, gate_diag = adapted_dufs_soft_gates(
            F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
        )
        graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
        dufs_path = laplacian_iu_path(
            F, (0.0, DUFS_LAMBDA), graph=graph
        )
        dufs_iu = dufs_path[0.0].baseline.w @ F
        dufs = dufs_path[DUFS_LAMBDA].w @ F
        z_dufs_iu = (dufs_iu - np.mean(dufs_iu)) / np.std(dufs_iu)
        iu_identity_error = float(np.max(np.abs(z_dufs_iu - baseline)))
        if iu_identity_error > 1e-10:
            raise RuntimeError(
                f"DUFS and CB baselines disagree in {subset}: "
                f"{iu_identity_error:.3e}"
            )

        score_path = score_dir / f"llama31_8b__{subset}.npz"
        np.savez_compressed(
            score_path,
            row_ids=np.asarray(row_ids),
            feature_names=np.asarray(names),
            family_names=np.asarray(fitted.contribution_space.families),
            iu_risk=-baseline,
            cardinality_risk=-primary.score,
            leverage_risk=-leverage.score,
            dufs_liu_risk=-dufs,
            uniform_risk=-uniform,
            reverse_cardinality_risk=-reverse,
            cardinality_delta=primary.delta,
            cardinality_effective_weights=primary.effective_weights,
            cardinality_intercept=np.asarray(primary.intercept),
            leverage_delta=leverage.delta,
            leverage_effective_weights=leverage.effective_weights,
            leverage_intercept=np.asarray(leverage.intercept),
        )
        data_hashes[subset] = sha256_file(path)
        score_hashes[subset] = sha256_file(score_path)
        diagnostics.append({
            "version": VERSION,
            "subset": subset,
            "n_rows": len(rows),
            "n_features": int(F.shape[0]),
            "n_families": len(fitted.contribution_space.families),
            "target_keys_accessed_during_fit": False,
            "contribution_reconstruction_error": (
                fitted.contribution_space.diagnostics["reconstruction_error"]
            ),
            "cardinality_weight_reconstruction_error": (
                primary.diagnostics["weight_reconstruction_error"]
            ),
            "leverage_weight_reconstruction_error": (
                leverage.diagnostics["weight_reconstruction_error"]
            ),
            "cardinality_orthogonality": primary.diagnostics[
                "baseline_correction_covariance"
            ],
            "cardinality_correction_scale": primary.diagnostics[
                "correction_scale"
            ],
            "expected_correction_scale": 1.0 / len(
                fitted.contribution_space.families
            ),
            "dufs_iu_identity_error": iu_identity_error,
            "dufs_effective_feature_count": gate_diag.get(
                "effective_feature_count"
            ),
            "availability": json.dumps(availability, sort_keys=True),
            "contract": json.dumps(contract, sort_keys=True, default=str),
        })
        print(f"llama31_8b__{subset}: scores frozen", flush=True)

    write_csv(out / "fit_diagnostics.csv", diagnostics)
    manifest = {
        "version": VERSION,
        "status": "scores_frozen_before_per_row_targets",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "subsets": list(SUBSETS),
        "cache_root": str(cache_root.relative_to(REPO)),
        "labels_read_during_fit": False,
        "primary": "cardinality_balanced_contribution_subspace_iu",
        "target": "reasoning_error_present = (row['label'] != -1)",
        "dufs": {
            "seeds": list(DUFS_SEEDS),
            "epochs": DUFS_EPOCHS,
            "k": DUFS_K,
            "lambda": DUFS_LAMBDA,
        },
        "data_sha256": data_hashes,
        "score_sha256": score_hashes,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths().items()
        },
        "upstream_manifest_sha256": sha256_file(cache_root / "manifest.json"),
    }
    write_json(out / "FIT_MANIFEST.json", manifest)
    print(out / "FIT_MANIFEST.json")


def bootstrap_contrast(rows, method, reference):
    delta = np.asarray([
        row[f"{method}_auroc"] - row[f"{reference}_auroc"]
        for row in rows
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{method}:{reference}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = delta[
        rng.integers(0, len(delta), size=(BOOTSTRAP_DRAWS, len(delta)))
    ].mean(axis=1)
    return {
        "version": VERSION,
        "method": method,
        "reference": reference,
        "n_cells": len(rows),
        "method_macro_auroc": float(np.mean([
            row[f"{method}_auroc"] for row in rows
        ])),
        "reference_macro_auroc": float(np.mean([
            row[f"{reference}_auroc"] for row in rows
        ])),
        "delta_pp": float(100 * np.mean(delta)),
        "ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "worst_delta_pp": float(100 * np.min(delta)),
    }


def verify_freeze(cache_root, out, manifest):
    if manifest["version"] != VERSION:
        raise RuntimeError("fit manifest version mismatch")
    for name, path in source_paths().items():
        actual = sha256_file(path)
        if actual != manifest["source_sha256"][name]:
            raise RuntimeError(f"source changed after fit: {name}")
    if sha256_file(cache_root / "manifest.json") != manifest[
        "upstream_manifest_sha256"
    ]:
        raise RuntimeError("upstream manifest changed after fit")
    for subset in SUBSETS:
        data_path = cache_root / f"processbench_{subset}.pkl"
        score_path = out / "scores" / f"llama31_8b__{subset}.npz"
        if sha256_file(data_path) != manifest["data_sha256"][subset]:
            raise RuntimeError(f"data changed after fit: {subset}")
        if sha256_file(score_path) != manifest["score_sha256"][subset]:
            raise RuntimeError(f"score changed after fit: {subset}")


def report_scores(cache_root, out):
    cache_root, out = Path(cache_root), Path(out)
    with (out / "FIT_MANIFEST.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_freeze(cache_root, out, manifest)

    rows = []
    methods = (
        "iu", "cardinality", "leverage", "dufs_liu", "uniform",
        "reverse_cardinality",
    )
    for subset in SUBSETS:
        items = aligned_items(cache_root / f"processbench_{subset}.pkl")
        row_ids = [key for key, _ in items]
        # This is the first per-row target access in the two-phase protocol.
        target = np.asarray([
            int(row["label"] != -1) for _, row in items
        ], dtype=int)
        with np.load(
            out / "scores" / f"llama31_8b__{subset}.npz",
            allow_pickle=False,
        ) as scores:
            if list(scores["row_ids"].astype(str)) != row_ids:
                raise RuntimeError(f"row alignment changed: {subset}")
            score_lookup = {
                "iu": scores["iu_risk"],
                "cardinality": scores["cardinality_risk"],
                "leverage": scores["leverage_risk"],
                "dufs_liu": scores["dufs_liu_risk"],
                "uniform": scores["uniform_risk"],
                "reverse_cardinality": scores["reverse_cardinality_risk"],
            }
            row = {
                "version": VERSION,
                "model": "llama31_8b",
                "subset": subset,
                "n": len(target),
                "n_positive": int(np.sum(target)),
            }
            for method in methods:
                row[f"{method}_auroc"] = float(roc_auc_score(
                    target, score_lookup[method]
                ))
            rows.append(row)
    write_csv(out / "cell_results.csv", rows)

    comparisons = (
        ("cardinality", "iu"),
        ("leverage", "iu"),
        ("dufs_liu", "iu"),
        ("cardinality", "leverage"),
        ("cardinality", "dufs_liu"),
        ("cardinality", "uniform"),
        ("cardinality", "reverse_cardinality"),
    )
    summary = [
        bootstrap_contrast(rows, method, reference)
        for method, reference in comparisons
    ]
    write_csv(out / "summary.csv", summary)
    lookup = {
        (row["method"], row["reference"]): row for row in summary
    }
    primary = lookup[("cardinality", "iu")]

    fit_diag = list(csv.DictReader(
        (out / "fit_diagnostics.csv").open(encoding="utf-8")
    ))
    invariant_max = {
        "contribution_reconstruction": max(abs(float(
            row["contribution_reconstruction_error"]
        )) for row in fit_diag),
        "cardinality_weight_reconstruction": max(abs(float(
            row["cardinality_weight_reconstruction_error"]
        )) for row in fit_diag),
        "orthogonality": max(abs(float(
            row["cardinality_orthogonality"]
        )) for row in fit_diag),
        "trust_scale": max(abs(
            float(row["cardinality_correction_scale"])
            - float(row["expected_correction_scale"])
        ) for row in fit_diag),
        "iu_identity": max(abs(float(
            row["dufs_iu_identity_error"]
        )) for row in fit_diag),
    }
    max_invariant = max(invariant_max.values())
    gates = [
        {
            "name": "positive mean subset delta",
            "passed": primary["delta_pp"] > 0,
            "value": primary["delta_pp"],
        },
        {
            "name": "positive paired subset interval",
            "passed": primary["ci_low_pp"] > 0,
            "value": primary["ci_low_pp"],
        },
        {
            "name": "at least three wins",
            "passed": primary["wins"] >= 3,
            "value": primary["wins"],
        },
        {
            "name": "tail safety",
            "passed": primary["worst_delta_pp"] >= -1.0,
            "value": primary["worst_delta_pp"],
        },
        {
            "name": "numerical invariants",
            "passed": max_invariant < 1e-10,
            "value": max_invariant,
        },
    ]
    result = {
        "version": VERSION,
        "status": "new_scorer_family_same_processbench_examples",
        "all_primary_gates_passed": bool(all(gate["passed"] for gate in gates)),
        "primary": primary,
        "gates": gates,
        "invariant_max": invariant_max,
        "claim_boundary": (
            "The telemetry/scorer family is new to candidate selection; the "
            "underlying ProcessBench examples and labels are not independent."
        ),
    }
    write_json(out / "RESULT.json", result)

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# Frozen CB-CS-IU transfer to Llama ProcessBench",
        "",
        "**Status:** new scorer-family confirmation on the same underlying "
        "ProcessBench examples used in the Qwen3 study.",
        "",
        f"CB-CS-IU changed cell-macro reasoning-error AUROC by "
        f"**{signed(primary['delta_pp'])}pp** versus ordinary IU "
        f"({primary['wins']}W/{primary['losses']}L; worst "
        f"{signed(primary['worst_delta_pp'])}pp). The paired four-subset "
        f"interval is [{signed(primary['ci_low_pp'])}, "
        f"{signed(primary['ci_high_pp'])}]pp.",
        "",
        "| contrast | delta | paired subset 95% interval | W/L | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']} - {row['reference']}` "
            f"| {signed(row['delta_pp'])}pp "
            f"| [{signed(row['ci_low_pp'])}, {signed(row['ci_high_pp'])}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {signed(row['worst_delta_pp'])}pp |"
        )
    lines.extend([
        "",
        "## Frozen gates",
        "",
    ])
    for gate in gates:
        lines.append(
            f"- **{'PASS' if gate['passed'] else 'FAIL'} — "
            f"{gate['name']}:** {gate['value']}"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Fit accessed neither per-row target key; data, scores, source, and "
        "upstream manifest hashes were verified before report-time label "
        "access. Aggregate class counts were visible in the upstream manifest "
        "before the run.",
        "",
        "This confirms transfer across telemetry/scorer model families. It is "
        "not independent-example confirmation because the reasoning chains and "
        "labels are shared with the earlier Qwen3 ProcessBench study.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "report"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.phase == "fit":
        fit_scores(args.cache_root, args.out)
    else:
        report_scores(args.cache_root, args.out)


if __name__ == "__main__":
    main()
