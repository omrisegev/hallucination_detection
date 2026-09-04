#!/usr/bin/env python3
"""Render the frozen T0 result without recomputing any analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("results/og_sml_agent_b_v1/T0_REPORT.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/og_sml_agent_b_v1"))
    args = parser.parse_args()
    payload = json.loads(args.report.read_text())
    records = payload["records"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(0.85, 2.4, 0.27))

    ax0 = fig.add_subplot(grid[0])
    categories = ["Prior C-v2 gate PASS", "Prior C-v2 gate FAIL"]
    admissible = [
        payload["cross_tab"]["prior_pass_and_admissible"],
        payload["cross_tab"]["prior_fail_and_admissible"],
    ]
    inadmissible = [
        payload["cross_tab"]["prior_pass_and_inadmissible"],
        payload["cross_tab"]["prior_fail_and_inadmissible"],
    ]
    positions = np.arange(2)
    ax0.barh(positions, admissible, label="OG-SML admissible", color="#2f6f4e")
    ax0.barh(positions, inadmissible, left=admissible, label="OG-SML inadmissible", color="#a54f4f")
    for index, (yes, no) in enumerate(zip(admissible, inadmissible)):
        if yes:
            ax0.text(yes / 2, index, str(yes), ha="center", va="center", color="white", fontweight="bold")
        if no:
            ax0.text(yes + no / 2, index, str(no), ha="center", va="center", color="white", fontweight="bold")
    ax0.set_yticks(positions, categories)
    ax0.invert_yaxis()
    ax0.set_xlabel("Lane count (of 18 frozen C-v2 lane-cells)")
    ax0.set_title("T0 contingency: frozen C-v2 primary gate versus graph admissibility", loc="left", fontweight="bold")
    ax0.legend(loc="center right", frameon=False)
    ax0.spines[["top", "right"]].set_visible(False)

    ax1 = fig.add_subplot(grid[1])
    ordered = sorted(
        records,
        key=lambda record: (
            not record["previous_gates"]["primary_gate_pass"],
            -record["graph"]["j_selection"],
            record["cell_id"],
            record["lane"],
        ),
    )
    y = np.arange(len(ordered))
    labels = [
        f"{record['cell_id'].replace('processbench_', 'PB:').replace('prmbench_response_', 'PRM:')} · {record['lane']} · K={record['selected_k']}"
        for record in ordered
    ]
    for index, record in enumerate(ordered):
        gate_pass = record["previous_gates"]["primary_gate_pass"]
        graph_ok = record["graph"]["admissible"]
        colour = "#2f6f4e" if graph_ok else "#a54f4f"
        marker = "D" if gate_pass else "o"
        ax1.scatter(
            record["graph"]["j_selection"],
            index,
            marker=marker,
            s=75 if gate_pass else 52,
            color=colour,
            edgecolor="black" if gate_pass else "none",
            linewidth=0.8,
            zorder=3,
        )
        if not graph_ok and record["graph"]["j_raw"] > 0:
            ax1.plot(
                [0, record["graph"]["j_raw"]],
                [index, index],
                color="#999999",
                linewidth=0.8,
                linestyle=":",
                zorder=1,
            )
            ax1.scatter(record["graph"]["j_raw"], index, marker="|", s=65, color="#666666", zorder=2)
    ax1.set_yticks(y, labels, fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("J_selection (inadmissible structures are assigned 0; dotted endpoint = J_raw)")
    ax1.set_ylabel("Frozen C-v2 lane-cell and selected K")
    ax1.set_title("No separation in the predicted direction", loc="left", fontweight="bold")
    ax1.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#a54f4f", markeredgecolor="black", label="Prior gate PASS"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="none", label="Prior gate FAIL"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f6f4e", markeredgecolor="none", label="Graph admissible"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#a54f4f", markeredgecolor="none", label="Graph inadmissible"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
    )

    fig.suptitle("OG-SML Agent B T0 — preregistered explanation of C-v2 is falsified", fontsize=17, fontweight="bold")
    caption = fig.add_subplot(grid[2])
    caption.axis("off")
    caption.text(0.0, 0.82, "Observation", fontweight="bold", fontsize=9.5, transform=caption.transAxes)
    caption.text(0.105, 0.82, "0/3 prior passes are admissible, while 6/15 prior failures are admissible.", fontsize=9.5, transform=caption.transAxes)
    caption.text(0.0, 0.47, "Inference", fontweight="bold", fontsize=9.5, transform=caption.transAxes)
    caption.text(0.105, 0.47, "The proposed graph explanation does not explain C-v2's primary-gate outcomes on the structures actually fitted.", fontsize=9.5, transform=caption.transAxes)
    caption.text(0.0, 0.12, "Limitation", fontweight="bold", fontsize=9.5, transform=caption.transAxes)
    caption.text(0.105, 0.12, "Retrospective, 18 related donor lanes; no labels or localization outcome metrics were used.", fontsize=9.5, transform=caption.transAxes)

    png_path = args.output_dir / "T0_RESULT.png"
    svg_path = args.output_dir / "T0_RESULT.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    manifest_path = args.output_dir / "T0_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["renderer"] = {
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__)),
        "input_report_sha256": sha256_file(args.report),
    }
    manifest["artifacts"][png_path.name] = sha256_file(png_path)
    manifest["artifacts"][svg_path.name] = sha256_file(svg_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
