#!/usr/bin/env python
"""Figure for the readout-calibration result. Reads the saved JSON only."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "token_probability_fusion_v1"
FIG = RES / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8a85", "#fcfcfb"
KS = ["K=1", "K=3", "K=5", "K=10", "K=20", "K=40", "K=80", "K=160", "K=all (step mean)"]
XL = ["1", "3", "5", "10", "20", "40", "80", "160", "all"]

def style(ax, title, ylabel, xlabel=""):
    ax.set_facecolor(SURFACE); ax.set_title(title, color=INK, fontsize=10, loc="left", pad=7)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)

c1 = json.loads((RES / "READOUT_CALIBRATION_C1.json").read_text(encoding="utf8"))
V = c1["variants"]; x = np.arange(len(KS))
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))

ax = axes[0]
for j, (label, key, colour) in enumerate((("GSM8K", "gsm8k", BLUE),
                                          ("OlympiadBench", "olympiadbench", ORANGE),
                                          ("Omni-MATH", "omnimath", AQUA))):
    y = [100 * V[k]["l_sml"]["per_subset"][key]["sla"] for k in KS]
    ax.plot(x, y, color=colour, lw=2, marker="o", ms=6, label=label)
    # Olympiad and Omni sit within 0.1 pp of each other at K=10; stagger or they overprint
    dy = (9, 9, -15)[j]
    ax.annotate(f"{y[3]:.1f}", (3, y[3]), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=8, color=colour)
ax.axvline(3, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(3.08, ax.get_ylim()[0] + 1, "K=10 today", fontsize=8, color=MUTED)
style(ax, "Localization accuracy against the readout width", "gate-free SLA %", "K (top-K mean inside a step)")
ax.set_xticks(x); ax.set_xticklabels(XL); ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

ax = axes[1]
y = [100 * V[k]["l_sml"]["reproducibility"] for k in KS]
ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6)
peak = int(np.argmax(y))
ax.axvline(peak, color=BLUE, lw=1.4, ls=(0, (3, 3)))
ax.text(peak + 0.1, min(y) + 2, f"criterion picks K={XL[peak]}", fontsize=8.5, color=BLUE)
best = max(range(len(KS)), key=lambda i: V[KS[i]]["l_sml"]["sla"])
ax.axvline(best, color=ORANGE, lw=1.4, ls=(0, (1, 2)))
ax.text(best - 0.12, min(y) + 8, f"accuracy optimum K={XL[best]}", fontsize=8.5,
        color=ORANGE, ha="right")
for i in (peak, best):
    ax.annotate(f"{y[i]:.1f}", (i, y[i]), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color=INK)
style(ax, "The label-free criterion has a real interior peak", "% of answers whose two halves agree",
      "K (top-K mean inside a step)")
ax.set_xticks(x); ax.set_xticklabels(XL)  # one series: the title names it, no legend box

fig.suptitle("A wider readout helps, most on the long chains — and a label-free rule finds it",
             color=INK, fontsize=11.5, x=0.005, ha="left", y=1.03)
fig.tight_layout(); fig.patch.set_facecolor(SURFACE)
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(FIG / f"readout_calibration.{ext}", dpi=125, bbox_inches="tight", facecolor=SURFACE)
print("figures/readout_calibration.svg")
