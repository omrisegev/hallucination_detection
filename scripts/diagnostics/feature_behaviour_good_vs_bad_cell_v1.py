#!/usr/bin/env python
"""How do the channels behave in a cell we do well on versus one we do badly on?

GSM8K (we score 47.8-49.8) against Omni-MATH and OlympiadBench (30.4-31.0), split by
erroneous versus error-free answers, for every channel we extract.

This is a **label-using descriptive diagnostic**, not a method: it looks at the true
first-error step. Nothing here selects a feature, a sign or a threshold.

Figures:
  1  per-channel trajectory by relative position, erroneous vs clean, short vs long cell
  2  the mechanism panel: error-step lift, the distractor margin, and channel degeneracy
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
FIG = RES / "figures"
OUT = RES / "FEATURE_BEHAVIOUR.json"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"
GOOD, BAD = "gsm8k", "omnimath"
SUBS = ("gsm8k", "math", "olympiadbench", "omnimath")
NBINS = 6


def style(ax, title, ylabel="", xlabel=""):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, format="svg", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(FIG / name.replace(".svg", ".png"), format="png", dpi=125,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}"


def load():
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    with np.load(RES / "STEP_VIEWS_12CH.npz", allow_pickle=False) as z:
        views = z["views"].astype(float)
        channels = [str(c) for c in z["channels"]]
    sub = np.array([c[3:-3] if c.startswith("pb_") else "prm" for c in cells])
    return offsets, target, sub, views, channels


def profile(offsets, target, sub, views, channel, subset, erroneous):
    """Mean standardized value in each relative-position bin."""
    want = (sub == subset) & ((target >= 0) if erroneous else (target == -1))
    acc = [[] for _ in range(NBINS)]
    for i in np.flatnonzero(want):
        a, b = offsets[i], offsets[i + 1]
        v = views[a:b, channel]
        if len(v) < 2:
            continue
        pos = np.linspace(0.0, 1.0, len(v))
        for value, p in zip(v, pos):
            acc[min(NBINS - 1, int(p * NBINS))].append(value)
    return np.array([np.mean(x) if x else np.nan for x in acc])


def main() -> None:
    offsets, target, sub, views, channels = load()
    report: dict = {"schema": "token-probability-fusion-v1-feature-behaviour",
                    "label_using_diagnostic": True, "good_cell": GOOD, "bad_cell": BAD}

    # ---------------- figure 1: per-channel trajectories -------------------
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 7.6), sharex=True)
    x = np.arange(NBINS) + 0.5
    for c, name in enumerate(channels):
        ax = axes[c // 4][c % 4]
        for subset, ls in ((GOOD, "-"), (BAD, (0, (4, 2)))):
            for err, colour in ((True, ORANGE), (False, BLUE)):
                y = profile(offsets, target, sub, views, c, subset, err)
                ax.plot(x, y, color=colour, linestyle=ls, linewidth=1.8)
        style(ax, name, "SD (answer-std)" if c % 4 == 0 else "")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks([0.5, NBINS / 2, NBINS - 0.5])
        ax.set_xticklabels(["start", "middle", "end"], fontsize=7.5)
    handles = [plt.Line2D([], [], color=ORANGE, lw=2, label="has an error"),
               plt.Line2D([], [], color=BLUE, lw=2, label="error-free"),
               plt.Line2D([], [], color=INK2, lw=2, ls="-", label=f"{GOOD} (we do well)"),
               plt.Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label=f"{BAD} (we do badly)")]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2,
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Each channel through the answer — colour is error status, dash is the cell",
                 color=INK, fontsize=12, x=0.005, ha="left", y=1.005)
    fig.tight_layout()
    f1 = save(fig, "feature_trajectories.svg")

    # ---------------- the numbers behind figure 2 ---------------------------
    lift, ties = {}, {}
    for c, name in enumerate(channels):
        lift[name], ties[name] = {}, {}
        for s in SUBS:
            idx = np.flatnonzero((sub == s) & (target >= 0))
            lift[name][s] = float(np.mean([views[offsets[i] + target[i], c] for i in idx]))
            ties[name][s] = float(np.mean([
                np.mean(np.isclose(views[offsets[i]:offsets[i + 1], c],
                                   views[offsets[i]:offsets[i + 1], c].min())) for i in idx]))
    fused = views[:, :11].mean(1)
    margin = {}
    for s in SUBS:
        idx = np.flatnonzero((sub == s) & (target >= 0))
        e, d = [], []
        for i in idx:
            v = fused[offsets[i]:offsets[i + 1]]
            e.append(v[target[i]])
            d.append(np.delete(v, target[i]).max())
        margin[s] = {"error_step": float(np.mean(e)), "max_distractor": float(np.mean(d)),
                     "margin": float(np.mean(np.array(e) - np.array(d))),
                     "steps": float(np.mean(np.diff(offsets)[idx]))}
    report.update({"error_step_lift": lift, "tie_fraction_at_min": ties, "distractor": margin})

    # ---------------- figure 2: the mechanism ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    order = sorted(channels, key=lambda n: -lift[n][GOOD])
    y = np.arange(len(order))
    ax = axes[0]
    ax.barh(y - 0.2, [lift[n][GOOD] for n in order], 0.38, color=BLUE, label=f"{GOOD} (good cell)")
    ax.barh(y + 0.2, [lift[n][BAD] for n in order], 0.38, color=ORANGE, label=f"{BAD} (bad cell)")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.5)
    ax.invert_yaxis()
    style(ax, "Lift at the true error step", "", "SD above the answer's own mean")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    ax = axes[1]
    xs = np.arange(len(SUBS))
    ax.plot(xs, [margin[s]["error_step"] for s in SUBS], color=BLUE, lw=2, marker="o",
            label="the true error step")
    ax.plot(xs, [margin[s]["max_distractor"] for s in SUBS], color=ORANGE, lw=2, marker="o",
            label="strongest wrong step")
    for i, s in enumerate(SUBS):
        # Midway between the two lines: below the lower one collides with the tick labels.
        mid = 0.5 * (margin[s]["error_step"] + margin[s]["max_distractor"])
        ax.annotate(f"{margin[s]['margin']:+.2f}", (i, mid), textcoords="offset points",
                    xytext=(0, -4), ha="center", fontsize=8, color=INK)
    style(ax, "Why argmax fails more as chains grow", "SD (answer-std)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s[:9]}\n{margin[s]['steps']:.1f} steps" for s in SUBS], fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")

    ax = axes[2]
    deg = sorted(channels, key=lambda n: -ties[n][BAD])[:6]
    y = np.arange(len(deg))
    ax.barh(y - 0.2, [100 * ties[n][GOOD] for n in deg], 0.38, color=BLUE, label=GOOD)
    ax.barh(y + 0.2, [100 * ties[n][BAD] for n in deg], 0.38, color=ORANGE, label=BAD)
    ax.set_yticks(y)
    ax.set_yticklabels(deg, fontsize=7.5)
    ax.invert_yaxis()
    style(ax, "Channels that go flat on long chains", "", "% of steps tied at the answer minimum")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    fig.suptitle("The channels do not weaken on long chains — the competition does",
                 color=INK, fontsize=12, x=0.005, ha="left", y=1.02)
    fig.tight_layout()
    f2 = save(fig, "feature_mechanism.svg")

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    print("figures:", f1, f2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
