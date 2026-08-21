#!/usr/bin/env python3
"""Plots for the train-fitted DUFS length-residualization diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/direct_dufs_length_residualization_v1"
LANES = ("global24", "processbench", "ragtruth")
LANE_NAMES = {
    "global24": "Global (21 held cells)",
    "processbench": "ProcessBench model validation",
    "ragtruth": "RAGTruth test",
}
CONDITIONS = (
    "original",
    "drop_length_refit_gates",
    "train_residualized_refit_gates",
)
CONDITION_NAMES = {
    "original": "Original",
    "drop_length_refit_gates": "Drop explicit\nlength",
    "train_residualized_refit_gates": "Train-fitted\nresidualization",
}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def geometry_plot():
    data = rows(OUT / "SUMMARY.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), sharey=True)
    for ax, lane in zip(axes, LANES):
        current = {row["condition"]: row for row in data if row["lane"] == lane}
        x = np.arange(3)
        width = 0.34
        target = [100 * float(current[c]["median_target_smoothness_effect"]) for c in CONDITIONS]
        length = [100 * float(current[c]["median_length_smoothness_effect"]) for c in CONDITIONS]
        ax.bar(x - width / 2, target, width, color="#4C78A8", label="Target")
        ax.bar(x + width / 2, length, width, color="#E45756", label="Held-out length")
        for pos, value in zip(x - width / 2, target):
            ax.text(pos, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
        for pos, value in zip(x + width / 2, length):
            ax.text(pos, value + 1.2, f"{value:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [CONDITION_NAMES[c] for c in CONDITIONS], fontsize=8)
        ax.set_title(LANE_NAMES[lane])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Laplacian energy reduction vs permutation (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Does training-only residualization remove length geometry?", y=1.02)
    save(fig, "01_residualized_smoothness.png")


def graph_plot():
    data = np.load(OUT / "MANIFOLD_EXAMPLES.npz", allow_pickle=False)
    fig, axes = plt.subplots(3, 6, figsize=(19, 10.8))
    columns = (
        ("original", "target", "coolwarm", "Original · target"),
        ("original", "length_q", "viridis", "Original · length"),
        ("drop_length_refit_gates", "target", "coolwarm", "Drop · target"),
        ("drop_length_refit_gates", "length_q", "viridis", "Drop · length"),
        ("train_residualized_refit_gates", "target", "coolwarm", "Residual · target"),
        ("train_residualized_refit_gates", "length_q", "viridis", "Residual · length"),
    )
    for row_index, lane in enumerate(LANES):
        for col_index, (condition, value_key, cmap, title) in enumerate(columns):
            ax = axes[row_index, col_index]
            prefix = f"{lane}__{condition}__"
            coords = data[prefix + "coords"]
            values = data[prefix + value_key]
            edge_rows = data[prefix + "edge_rows"]
            edge_cols = data[prefix + "edge_cols"]
            for a, b in zip(edge_rows, edge_cols):
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], color="0.72", alpha=0.08, lw=0.25)
            ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=6, alpha=0.82, linewidths=0)
            ax.set_title(f"{LANE_NAMES[lane]}\n{title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Same held-out samples: target color versus held-out length color", y=1.005)
    save(fig, "02_residualized_graph_examples.png")


def mechanism_plot():
    summary = rows(OUT / "SUMMARY.csv")
    scores = rows(OUT / "SCORE_METRICS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    x = np.arange(3)
    width = 0.23
    colors = ("#9D755D", "#BAB0AC", "#59A14F")
    for offset, condition, color in zip((-width, 0.0, width), CONDITIONS, colors):
        values = []
        for lane in LANES:
            row = next(item for item in summary if item["lane"] == lane and item["condition"] == condition)
            values.append(float(row["median_abs_feature_length_spearman"]))
        ax.bar(x + offset, values, width, color=color, label=CONDITION_NAMES[condition].replace("\n", " "))
    ax.set_xticks(x, [LANE_NAMES[lane] for lane in LANES], fontsize=8)
    ax.set_ylabel("Median |Spearman(feature, length)|")
    ax.set_title("Direct feature–length dependence")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1]
    width = 0.34
    iu_change, liu_change = [], []
    for lane in LANES:
        by_cell = {}
        for row in scores:
            if row["lane"] == lane:
                by_cell.setdefault(row["cell"], {})[row["condition"]] = row
        iu_delta, liu_delta = [], []
        for conditions in by_cell.values():
            base = conditions["drop_length_refit_gates"]
            residual = conditions["train_residualized_refit_gates"]
            iu_delta.append(float(residual["iu_auroc"]) - float(base["iu_auroc"]))
            liu_delta.append(float(residual["dufs_liu_auroc"]) - float(base["dufs_liu_auroc"]))
        iu_change.append(100 * float(np.median(iu_delta)))
        liu_change.append(100 * float(np.median(liu_delta)))
    ax.bar(x - width / 2, iu_change, width, color="#72B7B2", label="IU-PCR")
    ax.bar(x + width / 2, liu_change, width, color="#F58518", label="DUFS-LIU")
    for pos, value in zip(np.r_[x - width / 2, x + width / 2], np.r_[iu_change, liu_change]):
        ax.annotate(
            f"{value:+.2f}", (pos, value), xytext=(0, 5 if value >= 0 else -13),
            textcoords="offset points", ha="center", fontsize=8,
        )
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x, [LANE_NAMES[lane] for lane in LANES], fontsize=8)
    ax.set_ylabel("Median AUROC change vs drop-only (percentage points)")
    ax.set_title("Ranking effect of residualization")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    save(fig, "03_residualized_mechanism_and_performance.png")


def main():
    geometry_plot()
    graph_plot()
    mechanism_plot()
    print("wrote 3 figures")


if __name__ == "__main__":
    main()
