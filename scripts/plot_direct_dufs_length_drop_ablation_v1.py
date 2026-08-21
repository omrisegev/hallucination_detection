#!/usr/bin/env python3
"""Plots for the explicit-length-drop DUFS graph ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/direct_dufs_length_drop_ablation_v1"
BASE = ROOT / "results/direct_dufs_graph_semantics_audit_v1"
LANES = ("global24", "processbench", "ragtruth")
LANE_NAMES = {"global24": "Global (21 cells)", "processbench": "ProcessBench validation", "ragtruth": "RAGTruth test"}
CONDITIONS = ("original", "drop_length_fixed_gates", "drop_length_refit_gates")
CONDITION_NAMES = {"original": "Original", "drop_length_fixed_gates": "Drop length\nfixed gates", "drop_length_refit_gates": "Drop length\nrefit gates"}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_plot():
    data = rows(OUT / "SUMMARY.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for ax, lane in zip(axes, LANES):
        current = {row["condition"]: row for row in data if row["lane"] == lane}
        x = np.arange(3); width = 0.34
        target = [100 * float(current[c]["median_target_smoothness_effect"]) for c in CONDITIONS]
        length = [100 * float(current[c]["median_length_smoothness_effect"]) for c in CONDITIONS]
        ax.bar(x - width / 2, target, width, color="#4C78A8", label="Target")
        ax.bar(x + width / 2, length, width, color="#E45756", label="Held-out length")
        for pos, value in zip(x - width / 2, target):
            ax.text(pos, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
        for pos, value in zip(x + width / 2, length):
            ax.text(pos, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [CONDITION_NAMES[c] for c in CONDITIONS], fontsize=8)
        ax.set_title(LANE_NAMES[lane])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Laplacian energy reduction vs permutation (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Dropping explicit length weakens the length geometry, but does not remove it", y=1.02)
    save(fig, "01_length_drop_smoothness.png")


def examples_plot():
    data = np.load(OUT / "MANIFOLD_EXAMPLES.npz", allow_pickle=False)
    fig, axes = plt.subplots(3, 4, figsize=(15, 11.5))
    columns = (
        ("original", "target", "coolwarm", "Original · target"),
        ("original", "length_q", "viridis", "Original · length quartile"),
        ("drop_length_refit_gates", "target", "coolwarm", "No length · target"),
        ("drop_length_refit_gates", "length_q", "viridis", "No length · held-out length"),
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
                ax.plot(coords[[a, b], 0], coords[[a, b], 1], color="0.75", alpha=0.10, lw=0.3)
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=7, alpha=0.8, linewidths=0)
            ax.set_title(f"{LANE_NAMES[lane]}\n{title}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("The graph after deleting length: held-out length structure remains visible", y=1.005)
    save(fig, "02_length_drop_graph_examples.png")


def performance_plot():
    original = rows(BASE / "LIU_EFFECTS.csv")
    dropped = rows(OUT / "LIU_EFFECTS.csv")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(3); width = 0.34
    iu_change, liu_change = [], []
    for lane in LANES:
        base = {row["cell"]: row for row in original if row["lane"] == lane}
        drop = {row["cell"]: row for row in dropped if row["lane"] == lane}
        cells = sorted(set(base) & set(drop))
        iu_change.append(100 * float(np.median([float(drop[c]["iu_auroc"]) - float(base[c]["iu_auroc"]) for c in cells])))
        liu_change.append(100 * float(np.median([float(drop[c]["dufs_liu_auroc"]) - float(base[c]["dufs_liu_auroc"]) for c in cells])))
    ax.bar(x - width / 2, iu_change, width, color="#72B7B2", label="IU-PCR")
    ax.bar(x + width / 2, liu_change, width, color="#F58518", label="DUFS-LIU")
    for pos, value in zip(np.r_[x - width/2, x + width/2], np.r_[iu_change, liu_change]):
        ax.text(pos, value + (0.015 if value >= 0 else -0.025), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_ylim(min(min(iu_change), min(liu_change)) - 0.08, 0.04)
    ax.set_xticks(x, [LANE_NAMES[lane] for lane in LANES])
    ax.set_ylabel("Median AUROC change after dropping length (percentage points)")
    ax.set_title("Removing explicit length does not improve ranking")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "03_length_drop_performance.png")


def main():
    summary_plot(); examples_plot(); performance_plot()
    print("wrote 3 figures")


if __name__ == "__main__":
    main()
