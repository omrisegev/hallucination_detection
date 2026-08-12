#!/usr/bin/env python3
"""Retrospective audit of label-free leverage-balanced contribution IU.

Run ``fit`` first.  It does not read any correctness-label key and freezes all
scores and diagnostics.  Run ``report`` separately to verify the freeze, open
labels, and evaluate the fixed scores.  The formula was discovered on these
development cells, so this is a mechanism audit rather than external
confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family,
    load_contract,
    validate_bundle_without_labels,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    cardinality_balanced_contribution_score,
    iu_family_contributions,
    leverage_balanced_contribution_score,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "leverage-balanced-cs-iu-v1-2026-08-12"
DEFAULT_OUT = REPO / "results" / "leverage_balanced_cs_iu_v1"
INCUMBENT_OUT = REPO / "results" / "hard_filter_dufs_liu_24cell"
N_PERMUTATIONS = 64
MIN_POSITIVES = 20
BOOTSTRAP_DRAWS = 20000
SUPERVISED_TEACHER_EQUAL_FAMILY_DELTA_PP = 0.7210027136680253


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction_score(baseline, residuals, direction):
    """Apply the same frozen 1/G trust rule to a mechanism control."""
    direction = np.asarray(direction, dtype=float)
    raw = residuals @ direction
    scale = float(np.std(raw))
    if scale <= 1e-12 or float(np.linalg.norm(direction)) <= 1e-12:
        correction = np.zeros(len(baseline), dtype=float)
    else:
        correction = raw / (len(direction) * scale)
    return baseline + correction, correction


def fit_scores(bundle, out):
    bundle = Path(bundle)
    out = Path(out)
    score_dir = out / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    score_hashes = {}
    with np.load(bundle, allow_pickle=True) as data:
        validate_bundle_without_labels(data)
        for index, cell in enumerate(INSCOPE, start=1):
            F, names = load_contract(data, cell, "mixed_v2")
            iu = upcr_fit(F, **IU_FIT_DEFAULTS)
            space = iu_family_contributions(F, names, iu.w)
            primary = leverage_balanced_contribution_score(space, iu.w)
            baseline = primary.baseline_score
            _, residuals = primary.transform.apply(
                space.baseline_score, space.contributions
            )
            family_count = len(space.families)

            uniform_score, uniform_correction = direction_score(
                baseline, residuals, np.ones(family_count)
            )
            cardinality_balanced = cardinality_balanced_contribution_score(
                space, iu.w
            )
            cardinality_score = cardinality_balanced.score
            cardinality_correction = cardinality_balanced.correction
            reverse_score = baseline - primary.correction

            # Preserve the leverage values but break their correspondence to
            # provenance families.  Seeds are cell-specific and deterministic.
            seed = int(hashlib.sha256(
                f"{VERSION}:{cell}".encode()
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            primary_direction = np.mean(np.log(np.maximum(
                primary.family_leverage,
                1e-12 * max(float(np.max(primary.family_leverage)), 1e-12),
            ))) - np.log(np.maximum(
                primary.family_leverage,
                1e-12 * max(float(np.max(primary.family_leverage)), 1e-12),
            ))
            permutation_scores = []
            for _ in range(N_PERMUTATIONS):
                permuted = rng.permutation(primary_direction)
                score, _ = direction_score(baseline, residuals, permuted)
                permutation_scores.append(score)
            permutation_scores = np.asarray(permutation_scores, dtype=float)

            score_path = score_dir / f"{cell}.npz"
            np.savez_compressed(
                score_path,
                iu=baseline,
                leverage_balanced=primary.score,
                uniform=uniform_score,
                cardinality=cardinality_score,
                reverse=reverse_score,
                permutations=permutation_scores,
                family_names=np.asarray(space.families),
                family_leverage=primary.family_leverage,
                delta=primary.delta,
                effective_weights=primary.effective_weights,
                intercept=np.asarray(primary.intercept),
                cardinality_effective_weights=(
                    cardinality_balanced.effective_weights
                ),
                cardinality_intercept=np.asarray(
                    cardinality_balanced.intercept
                ),
            )
            score_hashes[cell] = sha256_file(score_path)
            diagnostics.append({
                "version": VERSION,
                "cell": cell,
                "dataset_family": family(cell),
                "n_samples": F.shape[1],
                "n_features": F.shape[0],
                "n_families": family_count,
                "reconstruction_error": space.diagnostics[
                    "reconstruction_error"
                ],
                "correction_scale": primary.diagnostics["correction_scale"],
                "expected_correction_scale": 1.0 / family_count,
                "baseline_correction_covariance": primary.diagnostics[
                    "baseline_correction_covariance"
                ],
                "delta_norm": primary.diagnostics["delta_norm"],
                "zero_correction": primary.diagnostics["zero_correction"],
                "weight_reconstruction_error": primary.diagnostics[
                    "weight_reconstruction_error"
                ],
                "uniform_correction_scale": float(np.std(uniform_correction)),
                "cardinality_correction_scale": float(np.std(
                    cardinality_correction
                )),
                "cardinality_weight_reconstruction_error": (
                    cardinality_balanced.diagnostics[
                        "weight_reconstruction_error"
                    ]
                ),
            })
            print(f"[{index:02d}/{len(INSCOPE)}] {cell}: frozen", flush=True)

    write_csv(out / "fit_diagnostics.csv", diagnostics)
    sources = {
        "script": sha256_file(Path(__file__)),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "spec": sha256_file(REPO / "SPEC_LEVERAGE_BALANCED_CS_IU_V1.md"),
    }
    manifest = {
        "version": VERSION,
        "status": "retrospective_label_free_fit_formula_label_exposed",
        "labels_read_during_fit": False,
        "bundle": str(bundle.relative_to(REPO)),
        "bundle_sha256": sha256_file(bundle),
        "cells": list(INSCOPE),
        "contract": "dufs-liu-mixed-v2-development-2026-08-07",
        "formula": "centered_log_family_l1_leverage",
        "trust_scale": "1 / number_of_present_provenance_families",
        "n_permutations": N_PERMUTATIONS,
        "score_sha256": score_hashes,
        "source_sha256": sources,
    }
    write_json(out / "FIT_MANIFEST.json", manifest)
    print(f"fit manifest: {out / 'FIT_MANIFEST.json'}")


def equal_family_bootstrap(rows, method, reference="iu"):
    families = sorted({row["dataset_family"] for row in rows})
    family_delta = np.asarray([
        np.mean([
            row[f"{method}_auroc"] - row[f"{reference}_auroc"]
            for row in rows if row["dataset_family"] == name
        ])
        for name in families
    ], dtype=float)
    seed = int(hashlib.sha256(
        f"{VERSION}:{method}:{reference}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = family_delta[
        rng.integers(
            0,
            len(family_delta),
            size=(BOOTSTRAP_DRAWS, len(family_delta)),
        )
    ].mean(axis=1)
    return {
        "equal_family_delta_pp": float(100 * np.mean(family_delta)),
        "equal_family_ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "equal_family_ci_high_pp": float(100 * np.quantile(draws, 0.975)),
    }


def summarize_method(rows, method):
    values = np.asarray([row[f"{method}_auroc"] for row in rows])
    baseline = np.asarray([row["iu_auroc"] for row in rows])
    delta = values - baseline
    return {
        "version": VERSION,
        "method": method,
        "n_cells": len(rows),
        "n_dataset_families": len({row["dataset_family"] for row in rows}),
        "cell_macro_auroc": float(np.mean(values)),
        "cell_macro_iu_auroc": float(np.mean(baseline)),
        "cell_macro_delta_pp": float(100 * np.mean(delta)),
        **equal_family_bootstrap(rows, method),
        "wins": int(np.sum(delta > 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "worst_delta_pp": float(100 * np.min(delta)),
    }


def render_report(summary, contrasts, gates, exclusions, invariant_max):
    lookup = {row["method"]: row for row in summary}
    contrast_lookup = {row["reference"]: row for row in contrasts}
    primary = lookup["leverage_balanced"]
    lines = [
        "# Leverage-Balanced Contribution-Subspace IU v1",
        "",
        "**Status:** retrospective mechanism audit; label-free fit, but the formula was discovered after these development labels had been inspected.",
        "",
        "## Main result",
        "",
        (
            f"The frozen leverage-balanced score changed cell-macro AUROC by "
            f"**{primary['cell_macro_delta_pp']:+.3f}pp** and equal-family "
            f"AUROC by **{primary['equal_family_delta_pp']:+.3f}pp** "
            f"([{primary['equal_family_ci_low_pp']:+.3f}, "
            f"{primary['equal_family_ci_high_pp']:+.3f}]pp), with "
            f"{primary['wins']}W/{primary['losses']}L and worst-cell "
            f"{primary['worst_delta_pp']:+.3f}pp."
        ),
        "",
        "This is evidence that the fusion-internal leverage mechanism is worth an external-family confirmation. It is not prospective confirmation and must not be reported as one.",
        "",
        (
            "Against the independently frozen mixed-v2 DUFS-LIU incumbent on "
            "the same eligible cells, the descriptive equal-family contrast is "
            f"{contrast_lookup['dufs_liu']['equal_family_delta_pp']:+.3f}pp "
            f"([{contrast_lookup['dufs_liu']['equal_family_ci_low_pp']:+.3f}, "
            f"{contrast_lookup['dufs_liu']['equal_family_ci_high_pp']:+.3f}]pp)."
        ),
        "",
        "## Methods",
        "",
        "| method | cell AUROC | cell delta | equal-family delta [95%] | W/L | worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in (
        "leverage_balanced", "dufs_liu", "uniform", "cardinality",
        "reverse", "permuted_mean",
    ):
        row = lookup[method]
        lines.append(
            f"| `{method}` | {row['cell_macro_auroc']:.4f} | "
            f"{row['cell_macro_delta_pp']:+.3f}pp | "
            f"{row['equal_family_delta_pp']:+.3f} "
            f"[{row['equal_family_ci_low_pp']:+.3f}, "
            f"{row['equal_family_ci_high_pp']:+.3f}] | "
            f"{row['wins']}/{row['losses']} | "
            f"{row['worst_delta_pp']:+.3f}pp |"
        )
    lines.extend([
        "",
        "## Mechanism contrasts",
        "",
        "| contrast | equal-family delta [95%] |",
        "|---|---:|",
    ])
    for row in contrasts:
        lines.append(
            f"| `{row['primary']} - {row['reference']}` | "
            f"{row['equal_family_delta_pp']:+.3f} "
            f"[{row['equal_family_ci_low_pp']:+.3f}, "
            f"{row['equal_family_ci_high_pp']:+.3f}]pp |"
        )
    lines.extend([
        "",
        "## Continuation gates",
        "",
    ])
    for gate in gates:
        mark = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- **{mark} — {gate['name']}:** {gate['detail']}")
    lines.extend([
        "",
        "## Audit boundary",
        "",
        f"Cells excluded by the pre-existing positive-count rule: `{', '.join(exclusions)}`.",
        (
            "Maximum reconstruction error / correction-scale error / absolute "
            "baseline-correction covariance / effective-weight reconstruction "
            "error / frozen-IU AUROC mismatch: "
            f"{invariant_max['reconstruction_error']:.3e} / "
            f"{invariant_max['correction_scale_error']:.3e} / "
            f"{invariant_max['orthogonality']:.3e} / "
            f"{invariant_max['weight_reconstruction_error']:.3e} / "
            f"{invariant_max['incumbent_iu_auroc_mismatch']:.3e}."
        ),
        "",
        "The leverage-specific advantages over uniform and cardinality balancing have intervals that cross zero. The positive primary therefore supports contribution-family balancing more strongly than it uniquely identifies L1 leverage as the only mechanism.",
        "",
        "The next admissible claim requires an unchanged run on a new intrinsic-detection dataset or model family whose labels were not used during discovery.",
    ])
    return "\n".join(lines) + "\n"


def report_scores(bundle, out):
    bundle = Path(bundle)
    out = Path(out)
    manifest_path = out / "FIT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise RuntimeError("fit manifest version mismatch")
    if sha256_file(bundle) != manifest["bundle_sha256"]:
        raise RuntimeError("bundle changed after fit")
    expected_sources = {
        "script": sha256_file(Path(__file__)),
        "module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "spec": sha256_file(REPO / "SPEC_LEVERAGE_BALANCED_CS_IU_V1.md"),
    }
    if expected_sources != manifest["source_sha256"]:
        raise RuntimeError("source changed after score freeze; rerun fit")

    fit_diagnostics = list(csv.DictReader(
        (out / "fit_diagnostics.csv").open(encoding="utf-8")
    ))
    diag_lookup = {row["cell"]: row for row in fit_diagnostics}
    incumbent_manifest_path = INCUMBENT_OUT / "SCORE_FREEZE_MANIFEST.json"
    incumbent_manifest = json.loads(
        incumbent_manifest_path.read_text(encoding="utf-8")
    )
    rows = []
    exclusions = []
    with np.load(bundle, allow_pickle=True) as data:
        for cell in manifest["cells"]:
            score_path = out / "scores" / f"{cell}.npz"
            if sha256_file(score_path) != manifest["score_sha256"][cell]:
                raise RuntimeError(f"score hash mismatch: {cell}")
            labels = np.asarray(data[f"{cell}__labels"], dtype=int)
            if int(labels.sum()) < MIN_POSITIVES:
                exclusions.append(cell)
                continue
            with np.load(score_path, allow_pickle=False) as scores:
                incumbent_path = INCUMBENT_OUT / "scores" / f"{cell}.npz"
                if sha256_file(incumbent_path) != incumbent_manifest[
                    "score_sha256"
                ][cell]:
                    raise RuntimeError(f"incumbent score hash mismatch: {cell}")
                with np.load(incumbent_path, allow_pickle=False) as incumbent:
                    if not np.array_equal(
                        incumbent["sample_index"], np.arange(len(labels))
                    ):
                        raise RuntimeError(
                            f"incumbent sample alignment mismatch: {cell}"
                        )
                    incumbent_iu_auc = float(roc_auc_score(
                        labels, incumbent["mixed_v2__full__iu_pcr"]
                    ))
                    dufs_liu_auc = float(roc_auc_score(
                        labels, incumbent["mixed_v2__full__dufs_liu"]
                    ))
                permutation_auc = np.asarray([
                    roc_auc_score(labels, score)
                    for score in scores["permutations"]
                ], dtype=float)
                row = {
                    "version": VERSION,
                    "cell": cell,
                    "dataset_family": family(cell),
                    "n": len(labels),
                    "n_positive": int(labels.sum()),
                    "iu_auroc": float(roc_auc_score(labels, scores["iu"])),
                    "incumbent_iu_auroc": incumbent_iu_auc,
                    "dufs_liu_auroc": dufs_liu_auc,
                    "leverage_balanced_auroc": float(roc_auc_score(
                        labels, scores["leverage_balanced"]
                    )),
                    "uniform_auroc": float(roc_auc_score(
                        labels, scores["uniform"]
                    )),
                    "cardinality_auroc": float(roc_auc_score(
                        labels, scores["cardinality"]
                    )),
                    "reverse_auroc": float(roc_auc_score(
                        labels, scores["reverse"]
                    )),
                    "permuted_mean_auroc": float(np.mean(permutation_auc)),
                    "permuted_sd_auroc": float(np.std(permutation_auc)),
                }
                rows.append(row)

    methods = (
        "leverage_balanced", "dufs_liu", "uniform", "cardinality",
        "reverse", "permuted_mean",
    )
    summary = [summarize_method(rows, method) for method in methods]
    primary = next(
        row for row in summary if row["method"] == "leverage_balanced"
    )
    contrasts = []
    for reference in (
        "dufs_liu", "uniform", "cardinality", "reverse", "permuted_mean"
    ):
        contrast = equal_family_bootstrap(
            rows, "leverage_balanced", reference=reference
        )
        contrasts.append({
            "version": VERSION,
            "primary": "leverage_balanced",
            "reference": reference,
            **contrast,
        })
    contrast_lookup = {row["reference"]: row for row in contrasts}

    invariant_max = {
        "reconstruction_error": max(
            float(row["reconstruction_error"]) for row in fit_diagnostics
        ),
        "correction_scale_error": max(abs(
            float(row["correction_scale"])
            - float(row["expected_correction_scale"])
        ) for row in fit_diagnostics),
        "orthogonality": max(abs(
            float(row["baseline_correction_covariance"])
        ) for row in fit_diagnostics),
        "weight_reconstruction_error": max(
            float(row["weight_reconstruction_error"])
            for row in fit_diagnostics
        ),
        "incumbent_iu_auroc_mismatch": max(abs(
            row["iu_auroc"] - row["incumbent_iu_auroc"]
        ) for row in rows),
    }
    recovery = (
        primary["equal_family_delta_pp"]
        / SUPERVISED_TEACHER_EQUAL_FAMILY_DELTA_PP
    )
    gates = [
        {
            "name": "positive equal-family interval",
            "passed": primary["equal_family_ci_low_pp"] > 0,
            "detail": (
                f"low={primary['equal_family_ci_low_pp']:+.3f}pp"
            ),
        },
        {
            "name": "cell wins",
            "passed": primary["wins"] >= 16,
            "detail": f"{primary['wins']}/{primary['n_cells']} wins",
        },
        {
            "name": "tail safety",
            "passed": primary["worst_delta_pp"] >= -1.0,
            "detail": f"worst={primary['worst_delta_pp']:+.3f}pp",
        },
        {
            "name": "teacher recovery",
            "passed": recovery >= 0.30,
            "detail": f"{100 * recovery:.1f}% of frozen teacher gain",
        },
        {
            "name": "specificity beyond simple balancing",
            "passed": (
                contrast_lookup["uniform"]["equal_family_delta_pp"] > 0
                and contrast_lookup["cardinality"][
                    "equal_family_delta_pp"
                ] > 0
            ),
            "detail": (
                f"vs uniform {contrast_lookup['uniform']['equal_family_delta_pp']:+.3f}pp; "
                f"vs cardinality {contrast_lookup['cardinality']['equal_family_delta_pp']:+.3f}pp"
            ),
        },
        {
            "name": "orientation falsifier",
            "passed": contrast_lookup["reverse"]["equal_family_delta_pp"] > 0,
            "detail": (
                f"primary-reverse={contrast_lookup['reverse']['equal_family_delta_pp']:+.3f}pp"
            ),
        },
        {
            "name": "family correspondence",
            "passed": (
                contrast_lookup["permuted_mean"][
                    "equal_family_delta_pp"
                ] > 0
            ),
            "detail": (
                f"primary-permuted={contrast_lookup['permuted_mean']['equal_family_delta_pp']:+.3f}pp"
            ),
        },
        {
            "name": "numerical invariants",
            "passed": (
                invariant_max["correction_scale_error"] < 1e-10
                and invariant_max["orthogonality"] < 1e-10
                and invariant_max["weight_reconstruction_error"] < 1e-10
                and invariant_max["incumbent_iu_auroc_mismatch"] < 1e-12
            ),
            "detail": (
                f"scale error={invariant_max['correction_scale_error']:.2e}; "
                f"|cov|={invariant_max['orthogonality']:.2e}; "
                f"weight error={invariant_max['weight_reconstruction_error']:.2e}; "
                f"IU mismatch={invariant_max['incumbent_iu_auroc_mismatch']:.2e}"
            ),
        },
    ]

    write_csv(out / "cell_results.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "contrasts.csv", contrasts)
    write_json(out / "GATES.json", {
        "version": VERSION,
        "all_passed": bool(all(gate["passed"] for gate in gates)),
        "gates": gates,
        "supervised_teacher_equal_family_delta_pp": (
            SUPERVISED_TEACHER_EQUAL_FAMILY_DELTA_PP
        ),
        "teacher_recovery_fraction": float(recovery),
        "invariant_max": invariant_max,
        "excluded_cells": exclusions,
        "incumbent_score_manifest": str(
            incumbent_manifest_path.relative_to(REPO)
        ),
        "incumbent_score_manifest_sha256": sha256_file(
            incumbent_manifest_path
        ),
    })
    (out / "REPORT.md").write_text(
        render_report(
            summary, contrasts, gates, exclusions, invariant_max
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "primary": primary,
        "all_gates_passed": all(gate["passed"] for gate in gates),
        "teacher_recovery_fraction": recovery,
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "report"))
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(os.environ.get("LB_CS_IU_BUNDLE", str(DEFAULT_BUNDLE))),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("LB_CS_IU_OUT", str(DEFAULT_OUT))),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "fit":
        fit_scores(args.bundle, args.out)
    else:
        report_scores(args.bundle, args.out)


if __name__ == "__main__":
    main()
