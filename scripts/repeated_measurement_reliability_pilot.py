#!/usr/bin/env python3
"""Phase 0 validity pilot for repeated-measurement reliability U-PCR.

This script deliberately does not read outcome labels.  It asks a narrower
question first: does a synchronized moving-block bootstrap of one saved token
trace behave like a target-preserving repeated measurement?  Only a procedure
that passes the registered diagnostics is eligible for later AUROC evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectral_utils.dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from spectral_utils.feature_utils import extract_all_features
from spectral_utils.repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)
from spectral_utils.repeated_measurement_reliability import (
    FixedMixedV2Transformer,
    bootstrap_trace_row,
    circular_moving_block_indices,
    covariance_components,
    generalized_reliability,
    matrix_correlation,
    subspace_overlap,
)


MIN_AVAILABILITY = 0.95
MAX_SATURATION = 0.40
MIN_TRACE_LENGTH = 64
VALIDITY_THRESHOLDS = {
    "median_replicate_mean_bias_max": 0.35,
    "fraction_feature_bias_le_0p5_min": 0.70,
    "within_split_correlation_min": 0.80,
    "negative_signal_mass_max": 0.25,
    "top3_split_overlap_min": 0.70,
    "within_trace_ratio_min": 0.001,
    "within_trace_ratio_max": 0.80,
    "eligible_feature_count_min": 8,
}


def load_rows(path):
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    return [
        cache[key] for key in sorted(cache)
        if not cache[key]["align_diag"]["problems"]
    ]


def trace_features(row):
    output = extract_all_features(
        row["token_entropies"],
        spilled_energies=row.get("token_spilled_energies"),
        allow_short=True,
    ) or {}
    if row.get("token_logsumexp") is not None:
        output.update(energy_features_from_logsumexp(row["token_logsumexp"]))
    if row.get("top_k_logprobs") is not None:
        output.update(logprob_features(row["top_k_logprobs"]))
        output.update(logprob_features_extended(row["top_k_logprobs"]))
    return output


def feature_matrix(feature_rows, *, exclude_fixed_length=True):
    names, columns, exclusions = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        if name == "trace_length" and exclude_fixed_length:
            exclusions[name] = "bootstrap fixes length, so within variance is identically zero"
            continue
        values = np.asarray([row.get(name, np.nan) for row in feature_rows], dtype=float)
        finite = np.isfinite(values)
        if finite.mean() < MIN_AVAILABILITY:
            exclusions[name] = f"availability={finite.mean():.3f}"
            continue
        median = float(np.median(values[finite]))
        filled = np.where(finite, values, median)
        if filled.std() < 1e-8:
            exclusions[name] = "constant"
            continue
        if np.mean(filled == median) > MAX_SATURATION:
            exclusions[name] = "saturated"
            continue
        names.append(name)
        columns.append(values)
    return np.column_stack(columns), tuple(names), exclusions


def _negative_mass(eigenvalues):
    eigenvalues = np.asarray(eigenvalues)
    denominator = float(np.sum(np.abs(eigenvalues)))
    return float(np.sum(np.abs(eigenvalues[eigenvalues < 0])) / max(denominator, 1e-12))


def _validity(summary):
    t = VALIDITY_THRESHOLDS
    checks = {
        "replicate_mean_bias": summary["median_replicate_mean_bias"]
        <= t["median_replicate_mean_bias_max"],
        "feature_bias_coverage": summary["fraction_feature_bias_le_0p5"]
        >= t["fraction_feature_bias_le_0p5_min"],
        "within_split_stability": summary["within_split_correlation"]
        >= t["within_split_correlation_min"],
        "signal_psd_compatibility": summary["negative_signal_mass"]
        <= t["negative_signal_mass_max"],
        "subspace_split_stability": summary["top3_split_overlap"]
        >= t["top3_split_overlap_min"],
        "nontrivial_bounded_noise": (
            t["within_trace_ratio_min"] <= summary["within_trace_ratio"]
            <= t["within_trace_ratio_max"]
        ),
        "enough_eligible_features": summary["eligible_feature_count"]
        >= t["eligible_feature_count_min"],
    }
    return checks, bool(all(checks.values()))


def run_configuration(rows, *, fraction, repeats, seed):
    original_features = [trace_features(row) for row in rows]
    raw, names, exclusions = feature_matrix(original_features)
    transformer = FixedMixedV2Transformer.fit(raw, names)
    frozen, frozen_names, _ = dufs_liu_mixed_v2_matrix(
        np.where(np.isfinite(raw), raw, transformer.raw_median[None, :]), names
    )
    transform_error = float(np.max(np.abs(frozen - transformer.training_output)))
    if frozen_names != names or transform_error > 1e-10:
        raise RuntimeError(f"fixed mixed-v2 transformer mismatch: {transform_error:.3g}")

    rng = np.random.default_rng(seed)
    raw_replicates = np.full((len(rows), repeats, len(names)), np.nan)
    block_lengths = []
    for row_index, row in enumerate(rows):
        length = len(row["token_entropies"])
        block_length = int(np.clip(round(fraction * length), 8, 128))
        block_lengths.append(block_length)
        for repeat in range(repeats):
            indices = circular_moving_block_indices(length, block_length, rng)
            features = trace_features(bootstrap_trace_row(row, indices))
            raw_replicates[row_index, repeat] = [
                features.get(name, np.nan) for name in names
            ]
    transformed = transformer.transform(
        raw_replicates.reshape(-1, len(names))
    ).reshape(len(rows), repeats, len(names))
    original = transformer.training_output

    total, within, signal = covariance_components(original, transformed)
    generalized = generalized_reliability(signal, within)
    midpoint = repeats // 2
    _, within_left, signal_left = covariance_components(
        original, transformed[:, :midpoint]
    )
    _, within_right, signal_right = covariance_components(
        original, transformed[:, midpoint:]
    )
    generalized_left = generalized_reliability(signal_left, within_left)
    generalized_right = generalized_reliability(signal_right, within_right)

    replicate_mean_bias = np.sqrt(
        np.mean((transformed.mean(axis=1) - original) ** 2, axis=0)
    )
    reliability = np.diag(signal) / np.maximum(np.diag(total), 1e-12)
    # Development decision on GSM8K: a feature is procedure-compatible only
    # when its replicate mean stays within half an original-population SD and
    # the resampling variance does not exceed its total population variance.
    # This rule is frozen before the MATH confirmation cell is inspected.
    eligible = (replicate_mean_bias <= 0.5) & (
        np.diag(within) <= np.diag(total) + 1e-12
    )
    if eligible.sum() < 3:
        raise RuntimeError("fewer than three procedure-compatible features")
    eligible_ix = np.flatnonzero(eligible)
    restricted_total = total[np.ix_(eligible_ix, eligible_ix)]
    restricted_within = within[np.ix_(eligible_ix, eligible_ix)]
    restricted_signal = signal[np.ix_(eligible_ix, eligible_ix)]
    restricted_within_left = within_left[np.ix_(eligible_ix, eligible_ix)]
    restricted_within_right = within_right[np.ix_(eligible_ix, eligible_ix)]
    restricted_signal_left = signal_left[np.ix_(eligible_ix, eligible_ix)]
    restricted_signal_right = signal_right[np.ix_(eligible_ix, eligible_ix)]
    restricted_generalized = generalized_reliability(
        restricted_signal, restricted_within
    )
    restricted_left = generalized_reliability(
        restricted_signal_left, restricted_within_left
    )
    restricted_right = generalized_reliability(
        restricted_signal_right, restricted_within_right
    )
    signal_eigenvalues = np.linalg.eigvalsh(restricted_signal)
    within_trace_ratio = float(
        np.trace(restricted_within) / max(np.trace(restricted_total), 1e-12)
    )
    convergence = []
    for prefix in range(3, repeats + 1):
        _, estimate, _ = covariance_components(original, transformed[:, :prefix])
        convergence.append({
            "repeats": prefix,
            "correlation_to_full": matrix_correlation(estimate, within),
            "relative_frobenius_error": float(
                np.linalg.norm(estimate - within, ord="fro")
                / max(np.linalg.norm(within, ord="fro"), 1e-12)
            ),
        })
    summary = {
        "fraction": float(fraction),
        "n_rows": len(rows),
        "n_features": len(names),
        "eligible_feature_count": int(eligible.sum()),
        "repeats": repeats,
        "block_length_median": float(np.median(block_lengths)),
        "median_replicate_mean_bias": float(np.median(replicate_mean_bias[eligible])),
        "fraction_feature_bias_le_0p5": float(np.mean(replicate_mean_bias[eligible] <= 0.5)),
        "within_split_correlation": matrix_correlation(
            restricted_within_left, restricted_within_right
        ),
        "negative_signal_mass": _negative_mass(signal_eigenvalues),
        "top3_split_overlap": subspace_overlap(
            restricted_left["vectors"], restricted_right["vectors"], k=3
        ),
        "within_trace_ratio": within_trace_ratio,
        "generalized_ridge": restricted_generalized["ridge"],
        "generalized_noise_condition": restricted_generalized["noise_condition"],
        "n_generalized_eigenvalues_gt_1": int(
            np.sum(restricted_generalized["eigenvalues"] > 1.0)
        ),
        "fixed_transform_max_error": transform_error,
        "full_pool_diagnostics": {
            "median_replicate_mean_bias": float(np.median(replicate_mean_bias)),
            "fraction_feature_bias_le_0p5": float(np.mean(replicate_mean_bias <= 0.5)),
            "within_split_correlation": matrix_correlation(within_left, within_right),
            "negative_signal_mass": _negative_mass(np.linalg.eigvalsh(signal)),
            "top3_split_overlap": subspace_overlap(
                generalized_left["vectors"], generalized_right["vectors"], k=3
            ),
            "within_trace_ratio": float(
                np.trace(within) / max(np.trace(total), 1e-12)
            ),
        },
    }
    checks, passed = _validity(summary)
    summary["validity_checks"] = checks
    summary["validity_passed"] = passed
    feature_rows = [
        {
            "feature": name,
            "replicate_mean_bias": float(replicate_mean_bias[j]),
            "within_variance": float(within[j, j]),
            "total_variance": float(total[j, j]),
            "raw_reliability": float(reliability[j]),
            "procedure_compatible": bool(eligible[j]),
            "exclusion_reason": (
                "" if eligible[j] else
                "replicate_mean_bias>0.5" if replicate_mean_bias[j] > 0.5 else
                "within_variance>total_variance"
            ),
        }
        for j, name in enumerate(names)
    ]
    return {
        "summary": summary,
        "features": feature_rows,
        "feature_names": names,
        "eligible_feature_names": tuple(np.asarray(names)[eligible]),
        "eligible_mask": eligible,
        "exclusions": exclusions,
        "original_matrix": original,
        "total": total,
        "within": restricted_within,
        "signal": restricted_signal,
        "generalized_eigenvalues": restricted_generalized["eigenvalues"],
        "generalized_vectors": restricted_generalized["vectors"],
        "convergence": convergence,
    }


def plot_result(result, output_path, title):
    names = result["feature_names"]
    feature_rows = result["features"]
    bias = np.asarray([row["replicate_mean_bias"] for row in feature_rows])
    reliability = np.asarray([row["raw_reliability"] for row in feature_rows])
    order = np.argsort(bias)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].barh(np.arange(len(names)), bias[order])
    axes[0, 0].set_yticks(np.arange(len(names)), np.asarray(names)[order], fontsize=7)
    axes[0, 0].axvline(0.5, color="crimson", linestyle="--", label="registered limit")
    axes[0, 0].set_title("Replicate-mean bias (lower is better)")
    axes[0, 0].legend()
    axes[0, 1].barh(np.arange(len(names)), reliability[order])
    axes[0, 1].set_yticks(np.arange(len(names)), np.asarray(names)[order], fontsize=7)
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Raw reliability diag(S_signal) / diag(S_total)")
    image = axes[1, 0].imshow(result["within"], cmap="coolwarm", aspect="auto")
    axes[1, 0].set_title("Estimated within-procedure covariance")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
    values = np.maximum(result["generalized_eigenvalues"], 1e-10)
    axes[1, 1].semilogy(np.arange(1, len(values) + 1), values, marker="o")
    axes[1, 1].axhline(1.0, color="crimson", linestyle="--", label="signal/noise = 1")
    axes[1, 1].set_xlabel("Generalized direction")
    axes[1, 1].set_ylabel("Estimated signal/noise eigenvalue")
    axes[1, 1].set_title("Generalized reliability spectrum")
    axes[1, 1].legend()
    passed = result["summary"]["validity_passed"]
    figure.suptitle(f"{title} — validity {'PASS' if passed else 'FAIL'}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for path_text in args.paths:
        path = Path(path_text)
        rows = [
            row for row in load_rows(path)
            if len(row["token_entropies"]) >= MIN_TRACE_LENGTH
        ]
        if len(rows) > args.max_rows:
            selection_rng = np.random.default_rng(args.seed)
            selected = np.sort(selection_rng.choice(len(rows), args.max_rows, replace=False))
            rows = [rows[index] for index in selected]
        cell = f"{path.parent.name}__{path.stem.removeprefix('processbench_')}"
        for fraction in args.fractions:
            result = run_configuration(
                rows, fraction=fraction, repeats=args.repeats,
                seed=args.seed + int(round(fraction * 1000)),
            )
            result["summary"]["cell"] = cell
            all_summaries.append(result["summary"])
            tag = f"{cell}__block_{str(fraction).replace('.', 'p')}"
            with open(output_dir / f"{tag}__features.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=result["features"][0].keys())
                writer.writeheader()
                writer.writerows(result["features"])
            with open(output_dir / f"{tag}__details.json", "w") as handle:
                json.dump({
                    "summary": result["summary"],
                    "feature_names": result["feature_names"],
                    "eligible_feature_names": result["eligible_feature_names"],
                    "exclusions": result["exclusions"],
                    "generalized_eigenvalues": result["generalized_eigenvalues"].tolist(),
                    "convergence": result["convergence"],
                }, handle, indent=2)
            plot_result(result, output_dir / f"{tag}.png", tag)
            status = "PASS" if result["summary"]["validity_passed"] else "FAIL"
            print(
                f"{tag}: {status}; median bias="
                f"{result['summary']['median_replicate_mean_bias']:.3f}, "
                f"within split r={result['summary']['within_split_correlation']:.3f}, "
                f"negative mass={result['summary']['negative_signal_mass']:.3f}, "
                f"top-3 overlap={result['summary']['top3_split_overlap']:.3f}"
            )
    with open(output_dir / "summary.json", "w") as handle:
        json.dump({
            "phase": "0_replicate_validity",
            "labels_read": False,
            "replicate_interpretation": "moving-block-bootstrap procedure sensitivity",
            "thresholds": VALIDITY_THRESHOLDS,
            "configurations": all_summaries,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
