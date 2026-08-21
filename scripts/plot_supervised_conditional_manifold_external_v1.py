#!/usr/bin/env python3
"""Render compact static figures for the frozen external manifold audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/supervised_conditional_manifold_external_validation_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    with (args.results / "FAMILY_SUMMARY.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    decision = json.loads((args.results / "DECISION.json").read_text(encoding="utf-8"))
    families = [row["dataset_family"] for row in rows]
    x = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    width = .23
    axes[0].bar(x - width, [float(row["metric_effect"]) for row in rows], width,
                label="frozen metric", color="#2563eb")
    axes[0].bar(x, [float(row["residual_effect"]) for row in rows], width,
                label="after removing linear score", color="#7c3aed")
    axes[0].bar(x + width, [float(row["linear_advantage"]) for row in rows], width,
                label="metric minus linear", color="#059669")
    axes[0].axhline(.02, color="#dc2626", linestyle="--", linewidth=1.2, label="gate = 0.02")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set_xticks(x, families)
    axes[0].set_ylabel("conditional smoothness effect")
    axes[0].set_title("Does geometry survive length and the linear direction?")
    axes[0].legend(frameon=False, fontsize=8)

    utility = [float(row["liu_delta"]) for row in rows]
    colors = ["#059669" if value >= .005 else "#d97706" for value in utility]
    axes[1].bar(x, utility, color=colors, width=.55)
    axes[1].axhline(.005, color="#dc2626", linestyle="--", linewidth=1.2, label="utility gate = 0.005")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set_xticks(x, families)
    axes[1].set_ylabel("LIU - IU AUROC")
    axes[1].set_title("Does the graph improve detection utility?")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(decision["decision"].replace("_", " ").title(), fontsize=12)
    fig.savefig(args.results / "01_external_family_gates.png", dpi=180)
    plt.close(fig)

    gates = [
        ("coverage", decision["coverage_pass"]),
        ("conditional-null eligibility", decision["conditional_null_coverage_pass"]),
        ("conditional geometry", decision["geometry_pass"]),
        ("survives linear removal", decision["linear_residual_pass"]),
        ("beats linear graph", decision["distinct_vs_linear_pass"]),
        ("useful LIU gain", decision["utility_pass"]),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 2.4), constrained_layout=True)
    for index, (name, passed) in enumerate(gates):
        ax.scatter(index, 0, s=650, color="#059669" if passed else "#dc2626", zorder=3)
        ax.text(index, 0, "✓" if passed else "×", ha="center", va="center",
                color="white", fontsize=16, weight="bold")
        if index < len(gates) - 1:
            ax.plot([index + .12, index + .88], [0, 0], color="#9ca3af", linewidth=2)
    ax.set_xticks(range(len(gates)), [name.replace(" ", "\n") for name, _ in gates])
    ax.set_yticks([])
    ax.set_xlim(-.45, len(gates) - .55)
    ax.set_ylim(-.35, .35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Evidence gate ladder (green = passed; red = not passed)")
    fig.savefig(args.results / "02_external_gate_ladder.png", dpi=180, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
