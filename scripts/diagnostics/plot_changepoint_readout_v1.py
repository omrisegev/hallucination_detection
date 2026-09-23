#!/usr/bin/env python
"""Figure for the change-point readout grid. Reads the saved JSON only.

  left   every arm x rule against the published anchor. One horizontal line at 35.92 does
         the work: nothing in the grid reaches it.
  right  the same predictions re-referenced to EACH SERIES' OWN argmax, which is the
         question Omri actually asked. Here the answer is not flat -- a change-point rule
         beats argmax on the long token series and loses on the short step series.
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

ARMS = (("our level readout, per step\n(the published arm)", "level__step", 8),
        ("our level fusion, per token", "level__token", 713),
        ("their evidence drop, per step\n(mean of 5 worst)", "drop_m5__step", 8),
        ("their evidence drop, per step\n(single worst)", "drop_worst__step", 8),
        ("their evidence drop, per token", "drop__token", 713))
RULES = (("argmax", "argmax", INK, "o"),
         ("first crossing q=0.9", "first_q0.9", MUTED, "s"),
         ("CUSUM alarm", "cusum_alarm_k0.5_h5", ORANGE, "^"),
         ("CUSUM onset", "cusum_onset_k0.5_h5", BLUE, "v"),
         ("BOCPD rise", "bocpd_rise_hz32", AQUA, "D"),
         ("BOCPD reset", "bocpd_reset_hz32", PLUM, "P"))


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


grid = json.loads((RES / "CHANGEPOINT_READOUT.json").read_text(encoding="utf8"))
within = json.loads((RES / "CHANGEPOINT_WITHIN_ARM.json").read_text(encoding="utf8"))
point, anchor = grid["point"], grid["anchor_c1_mean_sla"]

fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
fig.patch.set_facecolor("white")
rows = np.arange(len(ARMS))[::-1]

# --------------------------------------------------- panel 1: absolute, against the anchor
ax = axes[0]
for label, rule, colour, marker in RULES:
    xs, ys = [], []
    for yi, (_, arm, _) in zip(rows, ARMS):
        name = f"{arm}@@{rule}"
        if name in point:
            xs.append(100 * point[name]["mean_sla"])
            ys.append(yi)
    ax.scatter(xs, ys, s=62, color=colour, marker=marker, zorder=3, label=label,
               edgecolors="white", linewidths=0.8)
for yi, (_, arm, _) in zip(rows, ARMS):
    values = [100 * point[f"{arm}@@{r}"]["mean_sla"] for _, r, _, _ in RULES
              if f"{arm}@@{r}" in point]
    ax.plot([min(values), max(values)], [yi, yi], color="#dedcd6", lw=2.5, zorder=1)
ax.axvline(anchor, color=RED, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax.annotate(f"published arm, {anchor:.2f}", (anchor, len(ARMS) - 0.42), color=RED,
            fontsize=9, ha="right", va="center", rotation=0,
            xytext=(-6, 0), textcoords="offset points")
chance = 100 * np.mean(list(grid["chance_sla"].values()))
ax.axvline(chance, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=2)
ax.annotate(f"chance {chance:.1f}", (chance, len(ARMS) - 0.42), color=MUTED,
            fontsize=8.5, ha="center", va="center")
style(ax, "Nothing in the grid reaches the published arm",
      "gate-free SLA, mean over the eight ProcessBench cells (%)")
ax.set_yticks(rows)
ax.set_yticklabels([label for label, _, _ in ARMS], fontsize=8.5)
ax.set_ylim(-1.1, len(ARMS) - 0.15)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, loc="lower left",
          bbox_to_anchor=(0.0, -0.005), columnspacing=1.1, handletextpad=0.25)

# ------------------------------------------- panel 2: re-referenced to each arm's own argmax
ax = axes[1]
for label, rule, colour, marker in RULES:
    if rule == "argmax":
        continue
    for yi, (_, arm, _) in zip(rows, ARMS):
        entry = within["arms"][arm]["rules"].get(rule)
        if entry is None:
            continue
        offset = 0.30 - 0.12 * [r for _, r, _, _ in RULES].index(rule)
        lo, hi = entry["ci95_pp"]
        ax.plot([lo, hi], [yi + offset] * 2, color=colour, lw=1.6, alpha=0.75, zorder=2)
        ax.scatter([entry["point_pp"]], [yi + offset], s=48, color=colour, marker=marker,
                   zorder=3, edgecolors="white", linewidths=0.7)
ax.axvline(0, color=INK, lw=1.3, zorder=1)
for yi, (_, arm, length) in zip(rows, ARMS):
    ax.axhspan(yi - 0.42, yi + 0.42, color="#f3f2ee" if length > 100 else "white",
               zorder=0, lw=0)
    ax.annotate(f"~{length} observations per answer", (0.995, yi - 0.36),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                va="center", ha="right")
style(ax, "Holding the series fixed: does a change-point rule beat argmax?",
      "gate-free SLA minus the SAME series' argmax (pp), 95% paired interval")
ax.set_yticks(rows)
ax.set_yticklabels([])
ax.set_ylim(-1.1, len(ARMS) - 0.15)
ax.annotate("on the ~8-step series, nothing beats argmax —\nthe error is an isolated impulse "
            "and there is nothing to accumulate",
            (0.01, 0.045), xycoords="axes fraction", fontsize=8.5, color=INK2, va="bottom")
ax.annotate("CUSUM wins here", (3.6, rows[1] + 0.06), fontsize=9, color=BLUE,
            fontweight="bold", ha="left")
ax.annotate("and here", (3.9, rows[4] + 0.06), fontsize=9, color=BLUE,
            fontweight="bold", ha="left")

fig.suptitle("Change-point readouts for the first reasoning error — 4,442 erroneous ProcessBench "
             "answers, source-group bootstrap, 10,000 draws",
             fontsize=11, color=INK, x=0.006, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(FIG / f"changepoint_readout.{ext}", dpi=190, facecolor="white")
print(f"written: {FIG / 'changepoint_readout.svg'} and .png")
