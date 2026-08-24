#!/usr/bin/env python3
"""Build plots and a concise signed-data report for the graph-order ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)


FAMILIES = {
    "feature_smooth_residual_graph": ("Smooth X on residual graph, then IU", "#2563EB", "o"),
    "feature_smooth_raw_graph": ("Smooth X on raw-X graph, then IU", "#0891B2", "s"),
    "score_smooth_residual_graph": ("Smooth IU score on residual graph", "#7C3AED", "^") ,
    "residual_ridge_correction": ("IU + residual ridge correction", "#BE123C", "D"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _lambda(arm: str) -> float:
    return float(arm.rsplit("__lam_", 1)[1].replace("p", ".").replace("m", "-"))


def _macro_contrast(rows: list[dict[str, str]], arm: str, reference: str, metric: str) -> dict[str, str]:
    return next(
        row for row in rows
        if row["scope"] == "macro24"
        and row["arm_id"] == arm
        and row["reference_id"] == reference
        and row["metric"] == metric
    )


def _pp(row: dict[str, str], field: str) -> float:
    return 100.0 * float(row[field])


def _response_curve(contrasts: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharex=True)
    for axis, metric, title in zip(axes, ("auroc", "auprc"), ("AUROC", "AUPRC")):
        for family, (label, color, marker) in FAMILIES.items():
            rows = sorted(
                (
                    row for row in contrasts
                    if row["scope"] == "macro24"
                    and row["reference_id"] == "iu_pcr"
                    and row["metric"] == metric
                    and row["arm_id"].startswith(family + "__")
                ),
                key=lambda row: _lambda(row["arm_id"]),
            )
            x = np.asarray([_lambda(row["arm_id"]) for row in rows])
            y = np.asarray([_pp(row, "value") for row in rows])
            lower = np.asarray([_pp(row, "ci_lower") for row in rows])
            upper = np.asarray([_pp(row, "ci_upper") for row in rows])
            axis.errorbar(
                x,
                y,
                yerr=np.vstack([y - lower, upper - y]),
                label=label,
                color=color,
                marker=marker,
                linewidth=1.8,
                capsize=3,
            )
        axis.axhline(0.0, color="#111827", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("Frozen graph strength λ (log scale)")
        axis.set_ylabel(f"Δ {title} vs IU-PCR (percentage points)")
        axis.set_title(f"Equal-cell Macro-24 {title}")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Graph-order mechanism response curve\n"
        "Points: equal-cell Macro-24; bars: 95% paired source-group bootstrap CI (20,000 draws)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.91))
    fig.savefig(output / "macro_response_curve.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "macro_response_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def _cell_heatmap(contrasts: list[dict[str, str]], output: Path) -> None:
    families = ("feature_smooth_residual_graph", "residual_ridge_correction")
    rows = [
        row for row in contrasts
        if row["scope"] == "cell"
        and row["metric"] == "auroc"
        and row["reference_id"] == "iu_pcr"
        and any(row["arm_id"].startswith(family + "__") for family in families)
    ]
    cells = sorted({row["cell_id"] for row in rows})
    arms = sorted(
        {row["arm_id"] for row in rows},
        key=lambda arm: (families.index(arm.split("__", 1)[0]), _lambda(arm)),
    )
    lookup = {(row["cell_id"], row["arm_id"]): _pp(row, "value") for row in rows}
    matrix = np.asarray([[lookup[(cell, arm)] for arm in arms] for cell in cells])
    limit = max(0.05, float(np.quantile(np.abs(matrix), 0.98)))
    fig, axis = plt.subplots(figsize=(13.4, 8.4))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_yticks(np.arange(len(cells)), labels=cells, fontsize=7)
    axis.set_xticks(
        np.arange(len(arms)),
        labels=[
            ("Smooth X" if arm.startswith("feature_") else "Residual correction")
            + f"\nλ={_lambda(arm):g}"
            for arm in arms
        ],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axis.set_title("Per-cell ΔAUROC vs IU-PCR (pp); red is better, blue is worse")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.84)
    colorbar.set_label("ΔAUROC percentage points")
    fig.tight_layout()
    fig.savefig(output / "per_cell_delta_heatmap.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "per_cell_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    release = args.release.resolve()
    evaluation = release / "evaluation"
    manifest = json.loads((evaluation / "EVALUATION_MANIFEST.json").read_text(encoding="utf-8"))
    for name, key in (
        ("EVALUATION.json", "evaluation_sha256"),
        ("metrics_long.csv", "metrics_sha256"),
        ("contrasts_long.csv", "contrasts_sha256"),
    ):
        if sha256_file(evaluation / name) != manifest[key]:
            raise RuntimeError(f"signed evaluation artifact drifted: {name}")
    contrasts = _rows(evaluation / "contrasts_long.csv")
    _response_curve(contrasts, evaluation)
    _cell_heatmap(contrasts, evaluation)

    key_arms = (
        "residual_ridge_correction__lam_0p03",
        "residual_ridge_correction__lam_0p3",
        "feature_smooth_residual_graph__lam_0p03",
        "feature_smooth_raw_graph__lam_0p1",
        "score_smooth_residual_graph__lam_0p03",
    )
    lines = [
        "# IU graph-order ablation v1 — result",
        "",
        "**Status:** D0 retrospective mechanism evidence; not independent validation.",
        "",
        "The current roster's DEEM-B3 is graph-free. The exact residual/Laplacian",
        "objective proposed for this ablation is reported as residual ridge correction,",
        "not as DEEM-B3.",
        "",
        "## Macro-24 paired results",
        "",
        "All deltas are percentage points versus freshly recomputed IU-PCR; intervals",
        "use 20,000 paired source-group bootstrap draws.",
        "",
        "| arm | ΔAUROC [95% CI] | ΔAUPRC [95% CI] |",
        "|---|---:|---:|",
    ]
    for arm in key_arms:
        auc = _macro_contrast(contrasts, arm, "iu_pcr", "auroc")
        ap = _macro_contrast(contrasts, arm, "iu_pcr", "auprc")
        lines.append(
            f"| `{arm}` | {_pp(auc, 'value'):+.4f} "
            f"[{_pp(auc, 'ci_lower'):+.4f}, {_pp(auc, 'ci_upper'):+.4f}] | "
            f"{_pp(ap, 'value'):+.4f} "
            f"[{_pp(ap, 'ci_lower'):+.4f}, {_pp(ap, 'ci_upper'):+.4f}] |"
        )
    correction = _macro_contrast(
        contrasts, "residual_ridge_correction__lam_0p03", "signed_deem_b3", "auroc"
    )
    equal = _macro_contrast(
        contrasts, "residual_ridge_correction__lam_0p03", "equal_family_mean", "auroc"
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "At weak regularization every graph arm is statistically tied with IU-PCR.",
        "Increasing lambda makes the graph operation mechanically stronger but degrades",
        "the macro result. The exact residual-ridge arm at lambda=.03 remains "
        f"{_pp(correction, 'value'):+.3f}pp below signed DEEM-B3 "
        f"[{_pp(correction, 'ci_lower'):+.3f}, {_pp(correction, 'ci_upper'):+.3f}] "
        f"and {_pp(equal, 'value'):+.3f}pp below equal-family mean "
        f"[{_pp(equal, 'ci_lower'):+.3f}, {_pp(equal, 'ci_upper'):+.3f}].",
        "",
        "Therefore neither smoothing X before IU nor the exact constrained residual",
        "correction explains the DEEM-B3/equal-family advantage on frozen24. The result",
        "supports treating equal-family balancing as the simpler live explanation and",
        "does not support a graph-guided residual-correction claim on this panel.",
        "",
        "See `macro_response_curve.png` and `per_cell_delta_heatmap.png`. Every plotted",
        "number is sourced from `contrasts_long.csv`.",
    ])
    (evaluation / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_manifest = {
        "schema_version": "iu-graph-order-report-manifest-v1",
        "evaluation_manifest_sha256": sha256_file(evaluation / "EVALUATION_MANIFEST.json"),
        "report_sha256": sha256_file(evaluation / "REPORT.md"),
        "macro_response_curve_png_sha256": sha256_file(evaluation / "macro_response_curve.png"),
        "macro_response_curve_pdf_sha256": sha256_file(evaluation / "macro_response_curve.pdf"),
        "per_cell_delta_heatmap_png_sha256": sha256_file(evaluation / "per_cell_delta_heatmap.png"),
        "per_cell_delta_heatmap_pdf_sha256": sha256_file(evaluation / "per_cell_delta_heatmap.pdf"),
    }
    atomic_write_json(evaluation / "REPORT_MANIFEST.json", report_manifest)
    print(json.dumps(report_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
