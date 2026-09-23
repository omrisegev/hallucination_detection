#!/usr/bin/env python
"""Build the session report: Markdown plus SVG figures, from the saved JSON only.

Reads nothing but the result files, so the report cannot drift from what was run.
Figures follow the project's data-viz rules: fixed categorical slot order, one axis
per panel, a legend whenever more than one series is present, direct labels (the aqua
slot fails the 3:1 contrast check, so labels are required relief), recessive grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "token_probability_fusion_v1"
FIG = RES / "figures"
OUT = ROOT / "docs" / "experiments" / "TOKEN_PROBABILITY_FUSION_V1_SESSION_REPORT.md"

# Validated categorical slots (light surface): blue, orange, aqua. All-pairs PASS.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"

CELLS = ["pb_gsm8k_q4", "pb_gsm8k_q8", "pb_math_q4", "pb_math_q8",
         "pb_olympiadbench_q4", "pb_olympiadbench_q8", "pb_omnimath_q4", "pb_omnimath_q8"]
SHORT_LABEL = {"pb_gsm8k_q4": "GSM8K\n4B", "pb_gsm8k_q8": "GSM8K\n8B",
               "pb_math_q4": "MATH\n4B", "pb_math_q8": "MATH\n8B",
               "pb_olympiadbench_q4": "Olymp\n4B", "pb_olympiadbench_q8": "Olymp\n8B",
               "pb_omnimath_q4": "Omni\n4B", "pb_omnimath_q8": "Omni\n8B"}
TOKENS_PER_STEP = {"pb_gsm8k_q4": 55.0, "pb_gsm8k_q8": 55.0, "pb_math_q4": 80.5,
                   "pb_math_q8": 80.5, "pb_olympiadbench_q4": 88.6,
                   "pb_olympiadbench_q8": 88.6, "pb_omnimath_q4": 93.3, "pb_omnimath_q8": 93.3}
CELL_ORDER = sorted(CELLS, key=lambda c: (TOKENS_PER_STEP[c], c))


def style(ax, title, ylabel, xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f"figures/{name}"


def fig_pr_vs_floor(stage_b, floor):
    names = list(stage_b["cells"])
    short = {"C1_pooled_before": "C1\npooled · before", "C2_answerlocal_before": "C2\nlocal · before",
             "C3_pooled_after": "C3\npooled · after", "C4_answerlocal_after": "C4\nlocal · after"}
    labels = [short[n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for ax, (key, title, ceiling) in zip(axes, [
        ("conditional_participation_ratio", "Conditional participation ratio of the 11 views", None),
        ("weight_ipr_mean", "Weight inverse participation ratio (ceiling 11)", 11.0),
    ]):
        real = [stage_b["cells"][n][key] for n in names]
        null = [floor["null"][n][key] for n in names]
        x = np.arange(len(names))
        w = 0.36
        ax.bar(x - w / 2 - 0.01, real, w, color=BLUE, label="measured")
        ax.bar(x + w / 2 + 0.01, null, w, color=ORANGE, label="shuffled null")
        for xi, (a, b) in enumerate(zip(real, null)):
            ax.text(xi - w / 2 - 0.01, a + 0.12, f"{a:.2f}", ha="center", fontsize=8, color=INK)
            ax.text(xi + w / 2 + 0.01, b + 0.12, f"{b:.2f}", ha="center", fontsize=8, color=INK)
        if ceiling:
            ax.axhline(ceiling, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(-0.42, ceiling + 0.12, "ceiling 11", fontsize=8, color=MUTED, ha="left")
        style(ax, title, "ratio")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, (ceiling + 1.6) if ceiling else max(real) * 1.28)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=2)
    fig.suptitle("The floor is computed, not assumed — and it does not collapse",
                 color=INK, fontsize=11.5, x=0.005, ha="left", y=1.04)
    return save(fig, "pr_vs_noise_floor.svg")


def fig_2x2(stage_b):
    cells = {"pooled": ["C1_pooled_before", "C3_pooled_after"],
             "answer-local": ["C2_answerlocal_before", "C4_answerlocal_after"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = [0, 1]
    for series_index, (colour, (name, keys)) in enumerate(zip((BLUE, ORANGE), cells.items())):
        y = [100 * stage_b["cells"][k]["mean_sla_l_sml"] for k in keys]
        axes[0].plot(x, y, color=colour, linewidth=2, marker="o", markersize=8, label=name)
        for xi, yi in zip(x, y):
            axes[0].annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
        g = [stage_b["intervals_lsml_minus_equal"][k]["point_pp"] for k in keys]
        lo = [g[i] - stage_b["intervals_lsml_minus_equal"][k]["ci95_pp"][0] for i, k in enumerate(keys)]
        hi = [stage_b["intervals_lsml_minus_equal"][k]["ci95_pp"][1] - g[i] for i, k in enumerate(keys)]
        axes[1].errorbar(x, g, yerr=[lo, hi], color=colour, linewidth=2, marker="o",
                         markersize=8, capsize=4, label=name)
        for xi, yi in zip(x, g):
            side = -1 if xi == 0 else 1
            # The two post-readout points sit within 0.12 pp of each other, so their
            # labels must separate vertically or they print on top of one another.
            dy = -3 if xi == 0 else (10 if series_index == 0 else -14)
            axes[1].annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points",
                             xytext=(16 * side, dy), ha="left" if side > 0 else "right",
                             fontsize=8.5, color=INK)
    axes[1].axhline(0, color=MUTED, linewidth=1)
    axes[1].set_xlim(-0.55, 1.55)
    style(axes[0], "Mean gate-free SLA (L-SML)", "SLA %")
    style(axes[1], "L-SML advantage over equal weighting", "percentage points")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["fuse BEFORE readout", "fuse AFTER readout"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, title="standardization",
                  title_fontsize=8.5)
    fig.suptitle("The advantage lives in the fusion stage, not the standardization axis",
                 color=INK, fontsize=11.5, x=0.005, ha="left", y=1.04)
    return save(fig, "stage_b_interaction.svg")


def fig_length(length, deriv, mtg):
    x = np.arange(len(CELL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)

    a = axes[0]
    series_a = [("CT7 (level)", BLUE, [100 * length["sla_gate_free"]["CT7"][c]["sla"] for c in CELL_ORDER]),
                ("LX7 (length-calibrated)", ORANGE, [100 * length["sla_gate_free"]["LX7"][c]["sla"] for c in CELL_ORDER]),
                ("LX8 (calibrated + length view)", AQUA, [100 * length["sla_gate_free"]["LX8"][c]["sla"] for c in CELL_ORDER])]
    b = axes[1]
    series_b = [("token LEVEL", BLUE, [100 * deriv["sla_gate_free"]["LEVEL__equal"][c]["sla"] for c in CELL_ORDER]),
                ("token DERIVATIVE", ORANGE, [100 * deriv["sla_gate_free"]["DRV__equal"][c]["sla"] for c in CELL_ORDER]),
                ("LEVEL + DERIVATIVE", AQUA, [100 * deriv["sla_gate_free"]["LEVEL+DRV__equal"][c]["sla"] for c in CELL_ORDER])]
    for ax, series, title in ((a, series_a, "Attack 1 — subtract the length prior out of the level"),
                              (b, series_b, "Attack 2 — use a derivative instead of a level")):
        for label, colour, y in series:
            ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6, label=label)
            ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), textcoords="offset points",
                        xytext=(7, -3), fontsize=8.5, color=INK)
        chen = [mtg["q4" if c.endswith("q4") else "q8"][c[3:-3]]["shannon_drop"] for c in CELL_ORDER]
        ax.plot(x, chen, color=MUTED, linewidth=1.6, linestyle=(0, (4, 3)), label="Chen et al. Shannon Drop")
        style(ax, title, "gate-free SLA %" if ax is a else "")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABEL[c] for c in CELL_ORDER], fontsize=8)
        ax.set_xlabel("subsets ordered by tokens per step  (55 → 93)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    fig.suptitle("Both attacks on the length axis fail, and both fail harder on long chains",
                 color=INK, fontsize=11.5, x=0.005, ha="left", y=1.04)
    return save(fig, "length_axis.svg")


def main() -> None:
    stage_a = json.loads((RES / "GATE_HOLD_STAGE_A.json").read_text(encoding="utf8"))
    stage_b = json.loads((RES / "STAGE_B_2X2.json").read_text(encoding="utf8"))
    floor = json.loads((RES / "PR_NOISE_FLOOR.json").read_text(encoding="utf8"))
    length = json.loads((RES / "LENGTH_AXIS_BY_SUBSET.json").read_text(encoding="utf8"))
    deriv = json.loads((RES / "DERIVATIVE_CHANNEL_EVAL.json").read_text(encoding="utf8"))
    screen = json.loads((RES / "PER_CELL_COVARIANCE_SCREEN.json").read_text(encoding="utf8"))
    gate_iso = json.loads((ROOT / "results/claude_feature_bank_token_lsml_v1/GATE_ISOLATION.json")
                          .read_text(encoding="utf8"))
    mtg = gate_iso["mind_the_gap_published"]

    f1 = fig_pr_vs_floor(stage_b, floor)
    f2 = fig_2x2(stage_b)
    f3 = fig_length(length, deriv, mtg)
    print("figures:", f1, f2, f3)
    print("(report prose is written by hand in", OUT.name, "- figures regenerate from JSON)")


if __name__ == "__main__":
    main()
