#!/usr/bin/env python3
"""Frozen ProcessBench transfer test for leverage-balanced IU-PCR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
    leverage_balanced_iu_fit,
)
from spectral_utils.dufs_liu_feature_contract import (  # noqa: E402
    dufs_liu_mixed_v2_matrix,
)
from spectral_utils.feature_contract import (  # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
)
from spectral_utils.feature_utils import extract_all_features  # noqa: E402
from spectral_utils.repgrid_scoring import (  # noqa: E402
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)


VERSION = "leverage-balanced-processbench-transfer-v1-2026-08-12"
DEFAULT_CACHE = REPO / "dataset_cache" / "repgrid"
DEFAULT_OUT = REPO / "results" / "leverage_balanced_processbench_transfer_v1"
INCUMBENT_OUT = REPO / "results" / "processbench_latent_state_v1"
MODELS = ("qwen3_4b", "qwen3_8b")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
DEVELOPMENT = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
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
        raise ValueError("cannot write empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_data_path(path):
    """Resolve a local Git-LFS pointer without modifying the worktree."""
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(200)
    marker = b"version https://git-lfs.github.com/spec/v1\n"
    if not prefix.startswith(marker):
        return path
    fields = {}
    for line in prefix.decode("utf-8").splitlines()[1:]:
        if " " in line:
            key, value = line.split(" ", 1)
            fields[key] = value
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:"):
        raise RuntimeError(f"invalid Git-LFS pointer: {path}")
    digest = oid.removeprefix("sha256:")
    resolved = REPO / ".git" / "lfs" / "objects" / digest[:2] / digest[2:4] / digest
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Git-LFS object is not present locally for {path}: {digest}"
        )
    expected_size = int(fields.get("size", -1))
    if expected_size < 0 or resolved.stat().st_size != expected_size:
        raise RuntimeError(f"Git-LFS object size mismatch: {path}")
    return resolved


def load_rows_without_targets(path):
    """Load telemetry rows without accessing either evaluation target key."""
    with resolve_data_path(path).open("rb") as handle:
        cache = pickle.load(handle)
    selected = [
        (str(key), cache[key]) for key in sorted(cache)
        if not cache[key]["align_diag"]["problems"]
    ]
    return [key for key, _ in selected], [row for _, row in selected]


def trace_features_without_targets(row):
    values = extract_all_features(
        row["token_entropies"],
        spilled_energies=row.get("token_spilled_energies"),
        allow_short=True,
    ) or {}
    if row.get("token_logsumexp") is not None:
        values.update(energy_features_from_logsumexp(row["token_logsumexp"]))
    if row.get("top_k_logprobs") is not None:
        values.update(logprob_features(row["top_k_logprobs"]))
        values.update(logprob_features_extended(row["top_k_logprobs"]))
    return values


def mixed_v2_matrix(rows):
    """Exact global mixed-v2 construction used by GL-LIU v1."""
    features = [trace_features_without_targets(row) for row in rows]
    names, columns, availability = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        raw = np.asarray([
            item.get(name, np.nan) for item in features
        ], dtype=float)
        finite = np.isfinite(raw)
        availability[name] = float(np.mean(finite))
        if finite.mean() < 0.70 or not finite.any():
            continue
        raw = np.where(finite, raw, np.median(raw[finite]))
        if raw.std() < 1e-8 or np.mean(raw == np.median(raw)) > 0.40:
            continue
        names.append(name)
        columns.append(raw)
    raw = np.column_stack(columns)
    transformed, transformed_names, details = dufs_liu_mixed_v2_matrix(
        raw, names
    )
    return transformed.T, tuple(transformed_names), availability, details


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


def fit_scores(cache_root, out):
    cache_root = Path(cache_root)
    out = Path(out)
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, cache_hashes, score_hashes = [], {}, {}
    for model in MODELS:
        for subset in SUBSETS:
            cell = f"{model}__{subset}"
            cache_path = (
                cache_root / f"pb_{model}" / f"processbench_{subset}.pkl"
            )
            row_ids, rows = load_rows_without_targets(cache_path)
            F, names, availability, contract = mixed_v2_matrix(rows)
            fitted = leverage_balanced_iu_fit(F, names)
            balanced = fitted.balanced
            baseline = balanced.baseline_score
            _, residuals = balanced.transform.apply(
                fitted.contribution_space.baseline_score,
                fitted.contribution_space.contributions,
            )
            family_count = len(fitted.contribution_space.families)
            uniform = direction_score(
                baseline, residuals, np.ones(family_count)
            )
            cardinality_balanced = cardinality_balanced_contribution_score(
                fitted.contribution_space, fitted.baseline.w
            )
            cardinality_score = cardinality_balanced.score
            reverse = baseline - balanced.correction

            score_path = score_dir / f"{cell}.npz"
            np.savez_compressed(
                score_path,
                row_ids=np.asarray(row_ids),
                iu_risk=-baseline,
                leverage_balanced_risk=-balanced.score,
                uniform_risk=-uniform,
                cardinality_risk=-cardinality_score,
                reverse_risk=-reverse,
                family_names=np.asarray(fitted.contribution_space.families),
                family_leverage=balanced.family_leverage,
                delta=balanced.delta,
                effective_weights=balanced.effective_weights,
                intercept=np.asarray(balanced.intercept),
                cardinality_effective_weights=(
                    cardinality_balanced.effective_weights
                ),
                cardinality_intercept=np.asarray(
                    cardinality_balanced.intercept
                ),
            )
            cache_hashes[cell] = sha256_file(resolve_data_path(cache_path))
            score_hashes[cell] = sha256_file(score_path)
            diagnostics.append({
                "version": VERSION,
                "cell": cell,
                "model": model,
                "subset": subset,
                "split": (
                    "development" if (model, subset) in DEVELOPMENT
                    else "confirmation"
                ),
                "n": len(rows),
                "n_features": F.shape[0],
                "n_families": family_count,
                "reconstruction_error": fitted.contribution_space.diagnostics[
                    "reconstruction_error"
                ],
                "weight_reconstruction_error": balanced.diagnostics[
                    "weight_reconstruction_error"
                ],
                "orthogonality": balanced.diagnostics[
                    "baseline_correction_covariance"
                ],
                "correction_scale": balanced.diagnostics["correction_scale"],
                "expected_correction_scale": 1.0 / family_count,
                "cardinality_weight_reconstruction_error": (
                    cardinality_balanced.diagnostics[
                        "weight_reconstruction_error"
                    ]
                ),
                "target_keys_accessed_during_fit": False,
            })
            print(f"{cell}: frozen", flush=True)

    write_csv(out / "fit_diagnostics.csv", diagnostics)
    sources = {
        "script": sha256_file(Path(__file__)),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "spec": sha256_file(
            REPO / "SPEC_LEVERAGE_BALANCED_PROCESSBENCH_TRANSFER_V1.md"
        ),
    }
    write_json(out / "FIT_MANIFEST.json", {
        "version": VERSION,
        "status": "frozen_label_free_external_task_transfer",
        "target_keys_accessed_during_fit": False,
        "cells": [f"{model}__{subset}" for model in MODELS for subset in SUBSETS],
        "development_cells": [
            f"{model}__{subset}" for model, subset in sorted(DEVELOPMENT)
        ],
        "cache_sha256": cache_hashes,
        "score_sha256": score_hashes,
        "source_sha256": sources,
        "formula": "unchanged leverage-balanced CS-IU v1",
    })


def grouped_bootstrap(rows, method, reference, namespace):
    subsets = sorted({row["subset"] for row in rows})
    group_delta = np.asarray([
        np.mean([
            row[f"{method}_auroc"] - row[f"{reference}_auroc"]
            for row in rows if row["subset"] == subset
        ])
        for subset in subsets
    ], dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = group_delta[
        rng.integers(
            0, len(group_delta), size=(BOOTSTRAP_DRAWS, len(group_delta))
        )
    ].mean(axis=1)
    return float(np.mean(group_delta)), float(np.quantile(
        draws, 0.025
    )), float(np.quantile(draws, 0.975))


def summarize(rows, slice_name, target, method):
    selected = [
        row for row in rows
        if row["slice"] == slice_name and row["target"] == target
    ]
    values = np.asarray([row[f"{method}_auroc"] for row in selected])
    iu = np.asarray([row["iu_auroc"] for row in selected])
    delta, low, high = grouped_bootstrap(
        selected, method, "iu", f"{VERSION}:{slice_name}:{target}:{method}"
    )
    return {
        "version": VERSION,
        "slice": slice_name,
        "target": target,
        "method": method,
        "n_cells": len(selected),
        "n_subsets": len({row["subset"] for row in selected}),
        "cell_macro_auroc": float(np.mean(values)),
        "cell_macro_iu_auroc": float(np.mean(iu)),
        "cell_macro_delta_pp": float(100 * np.mean(values - iu)),
        "equal_subset_delta_pp": float(100 * delta),
        "equal_subset_ci_low_pp": float(100 * low),
        "equal_subset_ci_high_pp": float(100 * high),
        "wins": int(np.sum(values > iu + 1e-12)),
        "losses": int(np.sum(values < iu - 1e-12)),
        "ties": int(np.sum(np.abs(values - iu) <= 1e-12)),
        "worst_delta_pp": float(100 * np.min(values - iu)),
    }


def render_report(summary, gates, invariant_max):
    lookup = {
        (row["slice"], row["target"], row["method"]): row
        for row in summary
    }
    primary = lookup["confirmation", "reasoning_error_present", "leverage_balanced"]
    dufs = lookup["confirmation", "reasoning_error_present", "dufs_liu"]
    lines = [
        "# Frozen leverage-balanced IU transfer to ProcessBench",
        "",
        "**Status:** external-task transfer; formula frozen before ProcessBench evaluation, but ProcessBench is not a historically untouched benchmark in this repository.",
        "",
        "## Primary confirmation result",
        "",
        (
            f"Across the six confirmation/model-transfer cells, leverage-balanced "
            f"IU changed reasoning-error AUROC by "
            f"**{primary['cell_macro_delta_pp']:+.3f}pp** versus ordinary IU "
            f"({primary['wins']}W/{primary['losses']}L; worst "
            f"{primary['worst_delta_pp']:+.3f}pp). The equal-subset interval is "
            f"[{primary['equal_subset_ci_low_pp']:+.3f}, "
            f"{primary['equal_subset_ci_high_pp']:+.3f}]pp."
        ),
        "",
        (
            f"Frozen DUFS-LIU changed the same baseline by "
            f"{dufs['cell_macro_delta_pp']:+.3f}pp."
        ),
        "",
        "## Primary-target table",
        "",
        "| slice | method | AUROC | delta vs IU | equal-subset 95% interval | W/L | worst |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for slice_name in ("confirmation", "development", "all"):
        for method in (
            "leverage_balanced", "dufs_liu", "uniform", "cardinality", "reverse"
        ):
            row = lookup[slice_name, "reasoning_error_present", method]
            lines.append(
                f"| {slice_name} | `{method}` | {row['cell_macro_auroc']:.4f} | "
                f"{row['cell_macro_delta_pp']:+.3f}pp | "
                f"[{row['equal_subset_ci_low_pp']:+.3f}, "
                f"{row['equal_subset_ci_high_pp']:+.3f}] | "
                f"{row['wins']}/{row['losses']} | {row['worst_delta_pp']:+.3f}pp |"
            )
    lines.extend(["", "## Transfer gates", ""])
    for gate in gates:
        mark = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- **{mark} — {gate['name']}:** {gate['detail']}")
    lines.extend([
        "",
        "## Boundary",
        "",
        (
            "Maximum effective-weight reconstruction / orthogonality / trust-scale "
            f"errors were {invariant_max['weight_reconstruction_error']:.3e} / "
            f"{invariant_max['orthogonality']:.3e} / "
            f"{invariant_max['correction_scale_error']:.3e}."
        ),
        "",
        "Final-answer incorrect is saved as a secondary target in `summary.csv`; it did not select the method or gates.",
    ])
    return "\n".join(lines) + "\n"


def report_scores(cache_root, out):
    cache_root = Path(cache_root)
    out = Path(out)
    manifest = json.loads((out / "FIT_MANIFEST.json").read_text())
    expected_sources = {
        "script": sha256_file(Path(__file__)),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "spec": sha256_file(
            REPO / "SPEC_LEVERAGE_BALANCED_PROCESSBENCH_TRANSFER_V1.md"
        ),
    }
    if manifest["version"] != VERSION or manifest["source_sha256"] != expected_sources:
        raise RuntimeError("fit manifest/source mismatch")
    incumbent_manifest_path = INCUMBENT_OUT / "FREEZE_MANIFEST.json"
    incumbent_manifest = json.loads(incumbent_manifest_path.read_text())
    incumbent_lookup = {
        f"{row['model']}__{row['subset']}": row
        for row in incumbent_manifest["cells"]
    }

    base_rows = []
    for model in MODELS:
        for subset in SUBSETS:
            cell = f"{model}__{subset}"
            cache_path = (
                cache_root / f"pb_{model}" / f"processbench_{subset}.pkl"
            )
            if sha256_file(resolve_data_path(cache_path)) != manifest[
                "cache_sha256"
            ][cell]:
                raise RuntimeError(f"cache changed after fit: {cell}")
            score_path = out / "scores" / f"{cell}.npz"
            if sha256_file(score_path) != manifest["score_sha256"][cell]:
                raise RuntimeError(f"score changed after fit: {cell}")
            incumbent_meta = incumbent_lookup[cell]
            incumbent_path = INCUMBENT_OUT / incumbent_meta["scores"]
            if sha256_file(incumbent_path) != incumbent_meta["scores_file_sha256"]:
                raise RuntimeError(f"incumbent score changed: {cell}")

            row_ids, telemetry_rows = load_rows_without_targets(cache_path)
            with np.load(score_path, allow_pickle=False) as scores, np.load(
                incumbent_path, allow_pickle=False
            ) as incumbent:
                if not np.array_equal(scores["row_ids"], np.asarray(row_ids)):
                    raise RuntimeError(f"local row alignment mismatch: {cell}")
                incumbent_row_ids = np.asarray([
                    f"{subset}-{row_id}" for row_id in row_ids
                ])
                if not np.array_equal(
                    incumbent["row_ids"], incumbent_row_ids
                ):
                    raise RuntimeError(f"incumbent row alignment mismatch: {cell}")
                targets = {
                    "reasoning_error_present": np.asarray([
                        row["label"] != -1 for row in telemetry_rows
                    ], dtype=int),
                    "final_answer_incorrect": np.asarray([
                        not bool(row["final_answer_correct"])
                        for row in telemetry_rows
                    ], dtype=int),
                }
                methods = {
                    "iu": scores["iu_risk"],
                    "leverage_balanced": scores["leverage_balanced_risk"],
                    "uniform": scores["uniform_risk"],
                    "cardinality": scores["cardinality_risk"],
                    "reverse": scores["reverse_risk"],
                    "dufs_liu": incumbent["global_mixed_v2_dufs"],
                }
                for target_name, labels in targets.items():
                    metrics = {
                        f"{name}_auroc": float(roc_auc_score(labels, values))
                        for name, values in methods.items()
                    }
                    base_rows.append({
                        "version": VERSION,
                        "cell": cell,
                        "model": model,
                        "subset": subset,
                        "split": (
                            "development" if (model, subset) in DEVELOPMENT
                            else "confirmation"
                        ),
                        "target": target_name,
                        "n": len(labels),
                        "n_positive": int(labels.sum()),
                        **metrics,
                    })

    expanded = []
    for row in base_rows:
        for slice_name in ("all", row["split"]):
            expanded.append({**row, "slice": slice_name})
    methods = (
        "leverage_balanced", "dufs_liu", "uniform", "cardinality", "reverse"
    )
    targets = ("reasoning_error_present", "final_answer_incorrect")
    summary = [
        summarize(expanded, slice_name, target, method)
        for slice_name in ("confirmation", "development", "all")
        for target in targets
        for method in methods
    ]
    lookup = {
        (row["slice"], row["target"], row["method"]): row
        for row in summary
    }
    primary = lookup["confirmation", "reasoning_error_present", "leverage_balanced"]
    dufs = lookup["confirmation", "reasoning_error_present", "dufs_liu"]
    invariant_rows = list(csv.DictReader(
        (out / "fit_diagnostics.csv").open(encoding="utf-8")
    ))
    invariant_max = {
        "weight_reconstruction_error": max(
            float(row["weight_reconstruction_error"])
            for row in invariant_rows
        ),
        "orthogonality": max(abs(float(row["orthogonality"])) for row in invariant_rows),
        "correction_scale_error": max(abs(
            float(row["correction_scale"])
            - float(row["expected_correction_scale"])
        ) for row in invariant_rows),
    }
    gates = [
        {
            "name": "confirmation cell-macro improvement",
            "passed": primary["cell_macro_delta_pp"] > 0,
            "detail": f"{primary['cell_macro_delta_pp']:+.3f}pp",
        },
        {
            "name": "positive equal-subset interval",
            "passed": primary["equal_subset_ci_low_pp"] > 0,
            "detail": (
                f"[{primary['equal_subset_ci_low_pp']:+.3f}, "
                f"{primary['equal_subset_ci_high_pp']:+.3f}]pp"
            ),
        },
        {
            "name": "confirmation wins",
            "passed": primary["wins"] >= 4,
            "detail": f"{primary['wins']}/{primary['n_cells']} wins",
        },
        {
            "name": "tail safety",
            "passed": primary["worst_delta_pp"] >= -1.0,
            "detail": f"worst={primary['worst_delta_pp']:+.3f}pp",
        },
        {
            "name": "beats frozen DUFS-LIU",
            "passed": primary["cell_macro_auroc"] > dufs["cell_macro_auroc"],
            "detail": (
                f"LB={primary['cell_macro_auroc']:.4f}; "
                f"DUFS={dufs['cell_macro_auroc']:.4f}"
            ),
        },
        {
            "name": "numerical invariants",
            "passed": max(invariant_max.values()) < 1e-10,
            "detail": f"max={max(invariant_max.values()):.2e}",
        },
    ]
    write_csv(out / "cell_results.csv", base_rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "GATES.json", {
        "version": VERSION,
        "all_passed": bool(all(gate["passed"] for gate in gates)),
        "gates": gates,
        "invariant_max": invariant_max,
        "incumbent_manifest": str(incumbent_manifest_path.relative_to(REPO)),
        "incumbent_manifest_sha256": sha256_file(incumbent_manifest_path),
    })
    (out / "REPORT.md").write_text(
        render_report(summary, gates, invariant_max), encoding="utf-8"
    )
    print(json.dumps({
        "primary": primary,
        "dufs": dufs,
        "all_gates_passed": all(gate["passed"] for gate in gates),
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "report"))
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path(os.environ.get("LB_PB_CACHE", str(DEFAULT_CACHE))),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(os.environ.get("LB_PB_OUT", str(DEFAULT_OUT))),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "fit":
        fit_scores(args.cache_root, args.out)
    else:
        report_scores(args.cache_root, args.out)


if __name__ == "__main__":
    main()
