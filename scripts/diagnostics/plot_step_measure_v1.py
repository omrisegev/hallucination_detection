#!/usr/bin/env python
"""Figure for the step-measurement sweep. Reads the saved JSON only.

  left   SLA against width for all four corners of the 2x2. One reading carries the whole
         experiment: every curve rises with width, and both attempts to SHARPEN the
         statistic -- whitening in time, contiguity in position -- sit below the raw
         unordered mean they were meant to improve.
  right  the three pre-registered contrasts, each with its paired interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "token_probability_fusion_v1"
FIG = RES / "figures"
BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cc7"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8a85", "#fcfcfb"
RED = "#c0392b"

K_GRID = (1, 3, 5, 10, 20, 40, 80)
W_GRID = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = (
    ("raw, unordered Top-K  (the incumbent family)", "raw", "topk", K_GRID, BLUE, "o", "-"),
    ("whitened, unordered Top-K", "white_per_model", "topk", K_GRID, ORANGE, "^", "-"),
    ("raw, best contiguous window", "raw", "win", W_GRID, AQUA, "s", (0, (5, 2))),
    ("whitened, best contiguous window", "white_per_model", "win", W_GRID, PLUM, "D", (0, (5, 2))),
)


def style(ax, title, xlabel, ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


report = json.loads((RES / "STEP_MEASURE.json").read_text(encoding="utf8"))
point, anchor = report["point"], report["anchor_raw_topk10"]

fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor("white")

# ------------------------------------------------- panel 1: SLA against width, all corners
ax = axes[0]
for label, source, kind, grid, colour, marker, dash in FAMILIES:
    y = [100 * point[f"{source}__{kind}{w}"]["mean_sla"] for w in grid]
    ax.plot(grid, y, color=colour, lw=2, ls=dash, marker=marker, ms=6, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(set(K_GRID) | set(W_GRID)))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.axhline(anchor, color=RED, lw=1.4, ls=(0, (5, 3)), zorder=1)
ax.annotate(f"incumbent Top-10, {anchor:.2f}", (1.05, anchor), color=RED, fontsize=8.5,
            ha="left", va="bottom", xytext=(0, 4), textcoords="offset points")
peak = max(100 * point[f"raw__topk{k}"]["mean_sla"] for k in K_GRID)
# P1 predicted this optimum would move LEFT after whitening. It moved right, 20 -> 40.
ax.annotate("", xy=(40, 35.63), xytext=(20, peak),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                            connectionstyle="arc3,rad=-0.45"))
ax.annotate("P1 predicted this optimum would move LEFT.\nWhitening moved it right, and lowered it.",
            (40, 35.63), textcoords="offset points", xytext=(8, -34), fontsize=8.5, color=ORANGE)
ax.set_ylim(19.4, 39.2)
style(ax, "Every curve rises with width, and both sharpened statistics sit below the blunt one",
      "width: K tokens (unordered) or w tokens (contiguous)",
      "gate-free SLA, mean over the eight ProcessBench cells (%)")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower center")

# --------------------------------------------------- panel 2: the pre-registered contrasts
ax = axes[1]
LABELS = {
    "white_per_model__topk40 minus raw__topk20":
        "P1  whitening\nbest whitened − best raw",
    "white_pooled__win16 minus raw__topk20":
        "P2  contiguity\nbest window − best Top-K",
    "white_per_model__topk40 minus white_pooled__topk40":
        "P3  per-model scope\nper-model − pooled whitener",
}
rows = list(LABELS)
y = np.arange(len(rows))[::-1]
for yi, key in zip(y, rows):
    iv = report["key_contrasts"][key]
    lo, hi = iv["ci95_pp"]
    colour = RED if iv["excludes_zero"] and iv["point_pp"] < 0 else MUTED
    ax.plot([lo, hi], [yi, yi], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
    ax.scatter([iv["point_pp"]], [yi], s=95, color=colour, zorder=3,
               edgecolors="white", linewidths=0.9)
    ax.annotate(f"{iv['point_pp']:+.2f}", (iv["point_pp"], yi + 0.19), ha="center",
                fontsize=9, color=INK, fontweight="bold")
    note = ("FALSIFIED on direction — the optimum moved the wrong way"
            if key.startswith("white_per_model__topk40 minus raw")
            else "FALSIFIED" + ("" if iv["excludes_zero"] else " — no difference at all"))
    ax.annotate(note, (0.985, yi - 0.26), xycoords=("axes fraction", "data"),
                ha="right", fontsize=8.5, color=RED, fontweight="bold")
ax.axvline(0, color=INK, lw=1.3, zorder=1)
style(ax, "All three pre-registered predictions fail",
      "gate-free SLA difference (pp), 95% paired interval")
ax.set_yticks(y)
ax.set_yticklabels([LABELS[k] for k in rows], fontsize=8.5)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.grid(axis="y", linewidth=0)

fig.suptitle("Redesigning the step-measurement stage — fusion held fixed, whitener fitted per model, "
             "4,442 erroneous ProcessBench answers",
             fontsize=11, color=INK, x=0.006, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(FIG / f"step_measure.{ext}", dpi=190, facecolor="white")
print(f"written: {FIG / 'step_measure.svg'} and .png")
