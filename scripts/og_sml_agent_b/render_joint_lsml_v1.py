#!/usr/bin/env python3
"""Render signed, target-free Joint L-SML v1 artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "results/joint_lsml_v1_r2"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, root: Path, stem: str) -> list[str]:
    outputs = []
    for suffix in ("svg", "png"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _short_cell(cell_id: str) -> str:
    return (
        cell_id.replace("processbench_", "PB/")
        .replace("prmbench_response_", "PRM/")
        .replace("qwen3_", "Q")
    )


def _plot_ridge(task1: Mapping[str, Any], root: Path) -> list[str]:
    rows = list(task1["lanes"])
    values = np.asarray([row["minimum_pairwise_score_spearman"] for row in rows], dtype=float)
    colours = ["tab:blue" if row["lane"] == "v2_active28" else "tab:orange" for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(np.arange(len(rows)), values, color=colours, width=0.78)
    ax.axhline(0.99, color="black", linestyle="--", linewidth=1.1, label="frozen pass gate = 0.99")
    ax.set_ylim(min(0.94, float(values.min()) - 0.005), 1.001)
    ax.set_ylabel("Minimum pairwise donor-score Spearman correlation")
    ax.set_xlabel("Frozen donor cell / roster lane")
    ax.set_title("Ridge target sensitivity of frozen C-v2 donor fusion scores")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(
        [f"{_short_cell(row['cell_id'])}\n{row['lane']}" for row in rows],
        rotation=65, ha="right", fontsize=7,
    )
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.01, -0.04,
        "Source: hash-locked C-v2 sanitized donor inputs and structural ledger; conditions 1e2, 1e3, 1e4. No outcome targets.",
        fontsize=8,
    )
    return _save(fig, root, "task1_ridge_score_stability")


def _plot_orientation(orientation: Mapping[str, Any], root: Path) -> list[str]:
    cells = list(orientation["cells"])
    names = list(orientation["active_stream_names"])
    signs = np.asarray([row["signs"] for row in cells], dtype=float).T
    magnitude = np.asarray([row["absolute_loading"] for row in cells], dtype=float).T
    magnitude = magnitude / np.maximum(magnitude.max(axis=1, keepdims=True), 1e-12)
    rgba = plt.get_cmap("coolwarm")((signs + 1.0) / 2.0)
    rgba[..., 3] = 0.12 + 0.88 * magnitude
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_title("Absolute raw-stream sign estimates; opacity is within-stream |v|")
    ax.set_xlabel("Target-free donor cell")
    ax.set_ylabel("Raw token stream")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([_short_cell(row["cell_id"]) for row in cells], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for index, name in enumerate(names):
        if name == "entropy_series":
            ax.get_yticklabels()[index].set_fontweight("bold")
    fig.text(
        0.01, -0.015,
        "Source: nine sanitized donor cells. Blue=-1, red=+1; entropy_series is the semantic gauge anchor. No outcome targets.",
        fontsize=8,
    )
    return _save(fig, root, "orientation_sign_stability")


def _plot_joint(structural: Mapping[str, Any], root: Path) -> list[str]:
    rows = [row for row in structural["lanes"] if row.get("status") == "FIT_COMPLETE"]
    if not rows:
        return []
    x = np.arange(len(rows))
    joint = np.asarray([row["joint_fit"]["relative_offdiag_misfit"] for row in rows], dtype=float)
    hard = np.asarray([row["hard_lsml_relative_offdiag_misfit"] for row in rows], dtype=float)
    minimum_rho = np.asarray([row["minimum_weight_map_score_spearman"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})
    width = 0.38
    axes[0].bar(x - width / 2, hard, width, label="hard continuous L-SML model", color="0.68")
    axes[0].bar(x + width / 2, joint, width, label="joint disjoint factor model", color="tab:blue")
    axes[0].set_ylabel("Relative off-diagonal covariance misfit")
    axes[0].set_title("Structural fit on identical donor covariance and consensus partition")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, minimum_rho, color=["tab:blue" if row["lane"] == "v2_active28" else "tab:orange" for row in rows])
    axes[1].axhline(0.99, color="black", linestyle="--", linewidth=1.1, label="reference line = 0.99")
    axes[1].set_ylabel("Minimum pairwise score Spearman")
    axes[1].set_xlabel("Frozen donor cell / pruned roster lane")
    axes[1].set_title("Agreement among three Joint L-SML weight maps and existing continuous L-SML")
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{_short_cell(row['cell_id'])}\n{row['lane']}" for row in rows],
        rotation=65, ha="right", fontsize=7,
    )
    fig.text(
        0.01, -0.025,
        "Source: Joint L-SML v1 donor-only structural run. Correlations compare in-memory Xw rankings; scores are not persisted.",
        fontsize=8,
    )
    return _save(fig, root, "joint_lsml_structural_overview")


def _plot_grouping(structural: Mapping[str, Any], root: Path) -> list[str]:
    rows = [row for row in structural["lanes"] if row.get("status") == "FIT_COMPLETE"]
    if not rows:
        return []
    median_ari = np.asarray([row["grouping"]["median_ari"] for row in rows], dtype=float)
    minimum_ari = np.asarray([row["grouping"]["minimum_ari"] for row in rows], dtype=float)
    selected_k = np.asarray([row["grouping"]["K"] for row in rows], dtype=int)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(x, median_ari, marker="o", linewidth=1.5, label="median LOAO-to-consensus ARI")
    ax.plot(x, minimum_ari, marker="x", linewidth=1.2, label="minimum LOAO-to-consensus ARI")
    for index, (value, k) in enumerate(zip(median_ari, selected_k)):
        ax.text(index, value + 0.025, f"K={k}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Adjusted Rand index")
    ax.set_xlabel("Frozen donor cell / pruned roster lane")
    blocked = int(structural.get("blocked_lane_count", 0))
    ax.set_title(f"LOAO stability of selected consensus groupings ({blocked} lanes blocked)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{_short_cell(row['cell_id'])}\n{row['lane']}" for row in rows],
        rotation=65, ha="right", fontsize=7,
    )
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.01, -0.035,
        "Source: donor-answer LOAO partitions; K selected by ARI stability only, with K>=3 and every group size>=3.",
        fontsize=8,
    )
    return _save(fig, root, "loao_consensus_stability")


def render(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    task1 = _load(root / "TASK1_RIDGE_SCORE_STABILITY.json")
    orientation = _load(root / "ORIENTATION_CELL_LEDGER.json")
    structural = _load(root / "JOINT_STRUCTURAL_LEDGER.json")
    files = []
    files.extend(_plot_ridge(task1, root))
    files.extend(_plot_orientation(orientation, root))
    files.extend(_plot_joint(structural, root))
    files.extend(_plot_grouping(structural, root))
    return {"plot_files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(render(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
