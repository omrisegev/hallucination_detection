#!/usr/bin/env python3
"""Plot the completed conditional graph-topology audit tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/direct_dufs_conditional_graph_topology_audit_v1"
LANES = ("global24", "processbench", "ragtruth")
GRAPHS = (
    "union_knn_k7_self_safe",
    "radius_edge_matched_k7",
    "adaptive_knn_mean7_k3_25",
    "diffusion_edge_matched_base25_t2",
    "diffusion_edge_matched_base25_t4",
    "deployed_union_knn_k7",
    "mutual_knn_k7",
    "length_only_knn_k7",
    "permuted_self_safe_union_knn_k7",
)
SHORT = {
    "union_knn_k7_self_safe": "union self-safe",
    "radius_edge_matched_k7": "radius",
    "adaptive_knn_mean7_k3_25": "adaptive-k",
    "diffusion_edge_matched_base25_t2": "diffusion t=2",
    "diffusion_edge_matched_base25_t4": "diffusion t=4",
    "deployed_union_knn_k7": "deployed control",
    "mutual_knn_k7": "mutual-kNN",
    "length_only_knn_k7": "length control",
    "permuted_self_safe_union_knn_k7": "permuted control",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lookups(rows, lane, representation, graph):
    return [
        row for row in rows
        if row["lane"] == lane
        and row["representation"] == representation
        and row["graph"] == graph
    ]


def worst(rows, field):
    return float(np.nanmin([float(row[field]) for row in rows]))


def plot_raw_conditional(root: Path, summaries: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    x = np.arange(len(GRAPHS))
    for axis, lane in zip(axes, LANES):
        raw = np.asarray([
            worst(lookups(summaries, lane, "original", graph), "median_raw_target_effect")
            for graph in GRAPHS
        ])
        exact = np.asarray([
            worst(lookups(summaries, lane, "original", graph), "median_exact_target_effect")
            for graph in GRAPHS
        ])
        crt = np.asarray([
            worst(lookups(summaries, lane, "original", graph), "median_crt_target_effect")
            for graph in GRAPHS
        ])
        axis.bar(x - 0.25, 100 * raw, width=0.24, label="raw", color="#4c78a8")
        axis.bar(x, 100 * exact, width=0.24, label="exact length", color="#f58518")
        axis.bar(x + 0.25, 100 * crt, width=0.24, label="propensity CRT", color="#54a24b")
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(lane)
        axis.set_xticks(x, [SHORT[item] for item in GRAPHS], rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Median target smoothness effect (%)")
    axes[0].legend(frameon=False)
    figure.suptitle("Raw smoothness versus two independent length-conditional nulls", fontsize=14)
    figure.tight_layout()
    figure.savefig(root / "01_raw_vs_length_conditional.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_representation_factorial(root: Path, summaries: list[dict]) -> None:
    candidates = GRAPHS[:5]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), sharey=True)
    x = np.arange(len(candidates))
    colors = ("#4c78a8", "#f58518", "#54a24b")
    representations = ("original", "drop_length", "train_residualized")
    for axis, lane in zip(axes, LANES):
        for offset, representation, color in zip((-0.25, 0.0, 0.25), representations, colors):
            values = [
                100 * min(
                    worst(lookups(summaries, lane, representation, graph), "median_exact_target_effect"),
                    worst(lookups(summaries, lane, representation, graph), "median_crt_target_effect"),
                )
                for graph in candidates
            ]
            axis.bar(x + offset, values, width=0.24, color=color, label=representation)
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(lane)
        axis.set_xticks(x, [SHORT[item] for item in candidates], rotation=45, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Worst primary-null smoothness effect (%)")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Representation × graph factorial", fontsize=14)
    figure.tight_layout()
    figure.savefig(root / "02_representation_graph_factorial.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_geometry_utility(root: Path, summaries: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=False, sharey=False)
    representations = ("original", "drop_length")
    lane_colors = {"global24": "#4c78a8", "processbench": "#f58518", "ragtruth": "#54a24b"}
    markers = ("o", "s", "^", "D", "P")
    for axis, representation in zip(axes, representations):
        for graph, marker in zip(GRAPHS[:5], markers):
            for lane in LANES:
                current = lookups(summaries, lane, representation, graph)
                axis.scatter(
                    100 * min(
                        worst(current, "median_exact_target_effect"),
                        worst(current, "median_crt_target_effect"),
                    ),
                    100 * worst(current, "mean_liu_delta_auroc"),
                    color=lane_colors[lane], marker=marker, s=65, alpha=0.85,
                )
        axis.axhline(0, color="black", lw=0.8)
        axis.axvline(0, color="black", lw=0.8)
        axis.set_title(representation)
        axis.set_xlabel("Worst exact/CRT target smoothness (%)")
        axis.set_ylabel("Mean LIU − IU AUROC (points)")
        axis.grid(alpha=0.2)
    lane_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=lane, markersize=8)
        for lane, color in lane_colors.items()
    ]
    graph_handles = [
        plt.Line2D([0], [0], marker=marker, color="black", linestyle="none", label=SHORT[graph], markersize=7)
        for graph, marker in zip(GRAPHS[:5], markers)
    ]
    axes[1].legend(handles=lane_handles + graph_handles, frameon=False, fontsize=8, loc="best")
    figure.suptitle("Does conditional geometry translate into LIU utility?", fontsize=14)
    figure.tight_layout()
    figure.savefig(root / "03_geometry_vs_liu_utility.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    summaries = read_csv(root / "LANE_GRAPH_SUMMARY.csv")
    plot_raw_conditional(root, summaries)
    plot_representation_factorial(root, summaries)
    plot_geometry_utility(root, summaries)


if __name__ == "__main__":
    main()
