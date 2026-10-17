#!/usr/bin/env python
"""Figure for the impulse finding and its position control. Reads the saved JSON only.

Three panels, because the claim has three parts and they do not all survive:

  left    the profile around the true first error -- the impulse. It survives.
  middle  the SAME answers with the target ignored, against position. This is the panel
          the amendment did not have: the score collapses in the last two deciles of
          every answer whether or not an error is there.
  right   "after minus before" at the true error against the within-answer position
          control. Under the control the sign REVERSES: the true error's neighbourhood
          declines LESS than an arbitrary index in the same answer, not more.
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
ARM = "C1_token_l_sml"
SUBSETS = (("GSM8K", "gsm8k", BLUE), ("MATH", "math", PLUM),
           ("OlympiadBench", "olympiadbench", ORANGE), ("Omni-MATH", "omnimath", AQUA))
OFFSETS = list(range(-3, 4))


def style(ax, title, ylabel, xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


report = json.loads((RES / "IMPULSE_SHAPE_CONTROL.json").read_text(encoding="utf8"))
arm = report["arms"][ARM]

fig, axes = plt.subplots(1, 4, figsize=(19.6, 4.3))
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------- panel 1: the impulse
ax = axes[0]
for label, key, colour in SUBSETS:
    y = [arm[key]["profile_sd"][str(o)] for o in OFFSETS]
    ax.plot(OFFSETS, y, color=colour, lw=2, marker="o", ms=5.5, label=label)
ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.axvline(0, color="#d8d7d2", lw=1.2, zorder=0)
peak = np.mean([arm[k]["profile_sd"]["0"] for _, k, _ in SUBSETS])
ax.annotate(f"+{peak:.2f} SD at the error,\nneighbours at ~0", (0, peak),
            textcoords="offset points", xytext=(12, -6), fontsize=8.5, color=INK2)
style(ax, "The error step is an isolated impulse", "step score (answer SD)",
      "step offset from the true first error")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

# --------------------------------------------- panel 2: the confound the amendment lacked
ax = axes[1]
deciles = np.arange(10)
for label, key, colour in SUBSETS:
    ax.plot(deciles, arm[key]["positional_decile_sd"], color=colour, lw=2, marker="o", ms=5)
ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.axvspan(7.5, 9.5, color="#f2d9cf", alpha=0.55, zorder=0)
ax.annotate("every answer collapses here,\nwhether or not an error is in it", (7.3, -0.38),
            ha="right", fontsize=8.5, color=INK2)
style(ax, "The same answers with the target ignored", "step score (answer SD)",
      "decile of position within the answer")
ax.set_xticks(deciles)

# -------------------------------------------------- panel 3: the claim under its control
# A dumbbell, not paired bars: the quantity of interest is the DISTANCE between the true
# error and its own position control, and paired bars bury that in two long negative bars.
ax = axes[2]
y = np.arange(len(SUBSETS))[::-1]
true_gap = np.array([arm[k]["after_minus_before"]["at_true_error"] for _, k, _ in SUBSETS])
null_gap = np.array([arm[k]["after_minus_before"]["same_answer_any_index"] for _, k, _ in SUBSETS])
for yi, lo, hi in zip(y, null_gap, true_gap):
    ax.plot([lo, hi], [yi, yi], color="#c9c8c3", lw=3, solid_capstyle="round", zorder=1)
ax.scatter(null_gap, y, s=95, color=MUTED, zorder=3, label="any index, same answer (control)")
ax.scatter(true_gap, y, s=95, color=BLUE, zorder=3, label="at the true first error")
for yi, lo, hi in zip(y, null_gap, true_gap):
    ax.annotate(f"{hi - lo:+.2f}", ((lo + hi) / 2, yi + 0.17), ha="center", fontsize=9,
                color=INK, fontweight="bold")
ax.axvline(0, color=INK2, lw=1.0, zorder=0)
style(ax, 'The "after minus before" claim, controlled', "", "mean after − mean before (answer SD)")
ax.set_yticks(y)
ax.set_yticklabels([label for label, _, _ in SUBSETS], fontsize=9)
ax.set_ylim(-0.75, len(SUBSETS) + 0.35)
ax.grid(axis="y", linewidth=0)
ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left",
          bbox_to_anchor=(-0.02, -0.02), handletextpad=0.4)
ax.annotate("positive in every subset — the error's\nneighbourhood declines LESS than an\narbitrary index in the same answer",
            (0.99, 0.995), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.5, color=INK2)

# ------------------------------------- panel 4: does THEIR statistic have a different shape?
# The diagnostic the amendment asked for. If the evidence-drop series shows structure our
# level readout lacks, a different decision rule for it is justified; if it is also an
# isolated impulse, it is not.
ax = axes[3]
COMPARE = (("our level readout (C1)", "C1_token_l_sml", BLUE, "-"),
           ("their evidence drop, mean of 5 worst", "evidence_drop_mean_of_5_worst", ORANGE, "-"),
           ("their evidence drop, single worst", "evidence_drop_single_worst", ORANGE, (0, (4, 2))))
for label, key, colour, dash in COMPARE:
    if key not in report["arms"]:
        continue
    a = report["arms"][key]
    y = [np.mean([a[s]["profile_sd"][str(o)] for _, s, _ in SUBSETS]) for o in OFFSETS]
    ax.plot(OFFSETS, y, color=colour, lw=2, ls=dash, marker="o", ms=5, label=label)
ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
ax.axvline(0, color="#d8d7d2", lw=1.2, zorder=0)
style(ax, "Their statistic has the same shape as ours", "step score (answer SD), 4 subsets pooled",
      "step offset from the true first error")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

fig.suptitle("Shape of the step score around the true first error — published C1 token L-SML arm, "
             "4,442 erroneous ProcessBench answers",
             fontsize=11, color=INK, x=0.008, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.945))
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(FIG / f"impulse_shape_control.{ext}", dpi=190, facecolor="white")
print(f"written: {FIG / 'impulse_shape_control.svg'} and .png")
