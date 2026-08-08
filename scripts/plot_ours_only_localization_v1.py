#!/usr/bin/env python3
"""Create the frozen figures for the ours-only localization v1 report."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ours_only": "#1D5E8C",
    "mindgap_control": "#B23B31",
    "mindgap_detector_ours_locator": "#C68A2B",
}


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def detector_ranking(root, figures):
    rows = read_csv(os.path.join(root, "development_detector_ranking.csv"))[:10]
    rows = rows[::-1]
    labels = [row["candidate"].replace("answer_", "") for row in rows]
    values = [100 * float(row["auroc"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1D5E8C" if "answer_" in row["candidate"] else "#8A98A5" for row in rows]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Error-detection AUROC (%)")
    ax.set_title("Development: global trace fusion beats token-peak aggregation")
    ax.set_xlim(min(values) - 1.2, max(values) + 0.5)
    ax.grid(axis="x", alpha=0.2)
    save(fig, os.path.join(figures, "development_detector_ranking.png"))


def locator_ranking(root, figures):
    rows = read_csv(os.path.join(root, "development_locator_ranking.csv"))
    labels = [row["candidate"].replace("token_", "") for row in rows]
    exact = [100 * float(row["exact"]) for row in rows]
    tol1 = [100 * float(row["tol1"]) for row in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(y + 0.18, exact, height=0.34, label="Exact first-error step", color="#1D5E8C")
    ax.barh(y - 0.18, tol1, height=0.34, label="Within one step", color="#78A9C7")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Localization accuracy (%)")
    ax.set_title("Development: continuous token-stream locators")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, os.path.join(figures, "development_locator_ranking.png"))


def system_per_cell(root, figures):
    rows = read_csv(os.path.join(root, "final_systems_per_cell.csv"))
    cells = []
    table = {}
    for row in rows:
        cell = row["model"].replace("qwen3_", "") + "\n" + row["subset"]
        if cell not in cells:
            cells.append(cell)
        table[(cell, row["system"])] = 100 * float(row["f1"])
    systems = ["mindgap_control", "mindgap_detector_ours_locator", "ours_only"]
    labels = ["Mind the Gap", "Mind the Gap detector + our locator", "Ours only"]
    x = np.arange(len(cells)); width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, (system, label) in enumerate(zip(systems, labels)):
        values = [table[(cell, system)] for cell in cells]
        ax.bar(x + (index - 1) * width, values, width, label=label, color=COLORS[system])
    ax.set_xticks(x, cells)
    ax.set_ylabel("ProcessBench F1 (%)")
    ax.set_title("The ours-only system improves F1 in every evaluated cell")
    ax.legend(frameon=False, ncols=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    save(fig, os.path.join(figures, "final_f1_per_cell.png"))


def laplacian_transfer(root, figures):
    rows = read_csv(os.path.join(root, "component_metrics_per_cell.csv"))
    rows = [row for row in rows if row["component"] == "locator"]
    dev_cells = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
    families = {"uniform": [], "dufs": [], "temporal": []}
    for family in families:
        for lam in (0.03, 0.1, 0.3):
            candidate = f"token_{family}_liu_l{str(lam).replace('.', 'p')}"
            selected = [row for row in rows if row["candidate"] == candidate]
            dev = [100 * float(row["exact"]) for row in selected
                   if (row["model"], row["subset"]) in dev_cells]
            confirmation = [100 * float(row["exact"]) for row in selected
                            if (row["model"], row["subset"]) not in dev_cells]
            families[family].append((lam, np.mean(dev), np.mean(confirmation)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for family, values in families.items():
        x = [item[0] for item in values]
        axes[0].plot(x, [item[1] for item in values], marker="o", label=family)
        axes[1].plot(x, [item[2] for item in values], marker="o", label=family)
    axes[0].set_title("Development")
    axes[1].set_title("Confirmation/model transfer")
    for ax in axes:
        ax.set_xlabel("Laplacian strength λ")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Exact localization without threshold (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Temporal Laplacian gain is development-sensitive")
    save(fig, os.path.join(figures, "laplacian_lambda_transfer.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    root = os.path.abspath(args.result_dir)
    figures = os.path.join(root, "figures")
    os.makedirs(figures, exist_ok=True)
    detector_ranking(root, figures)
    locator_ranking(root, figures)
    system_per_cell(root, figures)
    laplacian_transfer(root, figures)


if __name__ == "__main__":
    main()
