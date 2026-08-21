#!/usr/bin/env python3
"""Build the direct DUFS graph audit figures from frozen CSV/NPZ outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "direct_dufs_graph_semantics_audit_v1"
LANE_LABEL = {"global24": "Global (21 cells with length)", "processbench": "ProcessBench", "ragtruth": "RAGTruth"}
COLORS = {"global24": "#4C78A8", "processbench": "#F58518", "ragtruth": "#54A24B"}


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_plot():
    rows = read_csv("LANE_SUMMARY.csv")
    x = np.arange(len(rows))
    target = [100 * float(row["median_target_smoothness_effect"]) for row in rows]
    length = [100 * float(row["median_length_smoothness_effect"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.34
    ax.bar(x - width / 2, target, width, label="Hallucination / process-error", color="#4C78A8")
    ax.bar(x + width / 2, length, width, label="Length nuisance", color="#E45756")
    for pos, value in zip(x - width / 2, target):
        ax.text(pos, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for pos, value in zip(x + width / 2, length):
        ax.text(pos, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [LANE_LABEL[row["lane"]] for row in rows])
    ax.set_ylabel("Laplacian energy reduction vs permutation (%)")
    ax.set_title("The DUFS graph tracks hallucination — but tracks length much more strongly")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "01_target_vs_length_smoothness.png")


def sensitivity_plot():
    rows = read_csv("GRAPH_SENSITIVITY_SUMMARY.csv")
    lanes = ["global24", "processbench", "ragtruth"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharex=True, sharey=True)
    for ax, lane in zip(axes, lanes):
        current = [row for row in rows if row["lane"] == lane]
        for family, color, marker in (("union_knn", "#4C78A8", "o"), ("mutual_knn", "#B279A2", "s")):
            selected = sorted((row for row in current if row["graph_family"] == family), key=lambda row: int(row["k"]))
            k = [int(row["k"]) for row in selected]
            target = [float(row["fraction_target_aligned"]) for row in selected]
            wins = [float(row["fraction_target_smoother_than_length"]) for row in selected]
            label = "ordinary kNN" if family == "union_knn" else "mutual-kNN"
            ax.plot(k, target, color=color, marker=marker, lw=1.8, label=f"{label}: target aligned")
            ax.plot(k, wins, color=color, marker=marker, lw=1.8, ls="--", label=f"{label}: target > length")
        ax.set_title(LANE_LABEL[lane])
        ax.set_xlabel("Number of neighbours k")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of validation cells")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Conclusion is stable across k: target clusters, but length clusters more", y=1.16)
    save(fig, "02_knn_topology_sensitivity.png")


def graph_examples_plot():
    archive = np.load(OUT / "MANIFOLD_EXAMPLES.npz", allow_pickle=False)
    lanes = ["global24", "processbench", "ragtruth"]
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 13.8))
    for row_index, lane in enumerate(lanes):
        coords = archive[f"{lane}__coords"]
        target = archive[f"{lane}__target"]
        length_q = archive[f"{lane}__length_q"]
        edge_rows = archive[f"{lane}__edge_rows"]
        edge_cols = archive[f"{lane}__edge_cols"]
        cell = str(archive[f"{lane}__cell"])
        for column, values, cmap, label in (
            (0, target, "coolwarm", "Target: 0 clean, 1 hallucination/error"),
            (1, length_q, "viridis", "Length quartile: short → long"),
        ):
            ax = axes[row_index, column]
            for a, b in zip(edge_rows, edge_cols):
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], color="0.75", alpha=0.12, lw=0.35, zorder=1)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=9, alpha=0.78, linewidths=0, zorder=2)
            ax.set_title(f"{LANE_LABEL[lane]} · {label}\n{cell}", fontsize=10)
            ax.set_xlabel("Laplacian coordinate 1")
            ax.set_ylabel("Laplacian coordinate 2")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Actual DUFS neighbourhood graphs: the same geometry colored two ways", y=1.005)
    save(fig, "03_actual_dufs_graph_examples.png")


def liu_plot():
    metrics = read_csv("GRAPH_VARIABLE_METRICS.csv")
    scores = read_csv("LIU_EFFECTS.csv")
    target_name = {"global24": "hallucination", "processbench": "process_error", "ragtruth": "hallucination"}
    lookup = {
        (row["lane"], row["cell"]): float(row["smoothness_z"])
        for row in metrics
        if row["graph"] == "dufs" and row["variable"] == target_name.get(row["lane"])
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for lane in ("global24", "processbench", "ragtruth"):
        selected = [row for row in scores if row["lane"] == lane]
        x = [lookup[(lane, row["cell"])] for row in selected]
        y = [float(row["liu_delta_auroc"]) * 100 for row in selected]
        ax.scatter(x, y, s=38, alpha=0.8, color=COLORS[lane], label=LANE_LABEL[lane])
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xlabel("Target smoothness on DUFS graph (z)")
    ax.set_ylabel("DUFS-LIU − IU-PCR (AUROC percentage points)")
    ax.set_title("Smoother hallucination labels do not imply a larger LIU gain")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "04_smoothness_vs_liu_gain.png")


def main():
    summary_plot()
    sensitivity_plot()
    graph_examples_plot()
    liu_plot()
    print("wrote 4 figures")


if __name__ == "__main__":
    main()
