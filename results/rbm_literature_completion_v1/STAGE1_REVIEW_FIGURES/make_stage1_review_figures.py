"""Independent visual review of the completed RBM literature-completion suites.

READ-ONLY over every existing file. Writes only PNG figures and CHECKS.json into
STAGE1_REVIEW_FIGURES/. Single process, no benchmark pickles, no sqlite, the 48 MB
stability FIT_HEALTH.json is streamed object by object.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = r"C:\Users\omris\TAU\hallucination_detection\.worktrees\rbm-literature-completion-v1\results\rbm_literature_completion_v1"
OUT = os.path.join(ROOT, "STAGE1_REVIEW_FIGURES")
SUITES = ["variance", "capacity", "temporal", "stability", "depth_amended"]
N_ANSWERS = 13769
CELLS = ["pb_gsm8k_q4", "pb_gsm8k_q8", "pb_math_q4", "pb_math_q8",
         "pb_olympiadbench_q4", "pb_olympiadbench_q8", "pb_omnimath_q4", "pb_omnimath_q8"]
DPI = 150
CHECKS = {"matched": [], "mismatched": [], "notes": []}


def check(name, observed, expected, tol=1e-6, note=""):
    ok = abs(float(observed) - float(expected)) <= tol
    rec = {"check": name, "observed": float(observed), "expected": float(expected), "tol": tol, "note": note}
    (CHECKS["matched"] if ok else CHECKS["mismatched"]).append(rec)
    return ok


def check_bool(name, ok, note=""):
    rec = {"check": name, "ok": bool(ok), "note": note}
    (CHECKS["matched"] if ok else CHECKS["mismatched"]).append(rec)
    return ok


plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": DPI})

# --------------------------------------------------------------------------------------
# Load the small tables
# --------------------------------------------------------------------------------------
fus = pd.read_csv(os.path.join(ROOT, "RBM_FUSION_COMPARISON.csv"))
stage_tbl = pd.read_csv(os.path.join(ROOT, "STAGE1_COMPARISON_TABLE.csv"))
metrics = {s: json.load(open(os.path.join(ROOT, s, "METRICS.json"), encoding="utf-8")) for s in SUITES}
comp = {s: pd.read_csv(os.path.join(ROOT, s, "COMPARISON.csv")) for s in SUITES}
cells = {s: pd.read_csv(os.path.join(ROOT, s, "PB_CELLS.csv")) for s in SUITES}
conv_json = json.load(open(os.path.join(ROOT, "capacity", "CAPACITY_CONVERGENCE.json"), encoding="utf-8"))
probe = json.load(open(os.path.join(ROOT, "capacity", "CAPACITY_MAXITER_PROBE.json"), encoding="utf-8"))
smoke_diag = json.load(open(os.path.join(ROOT, "depth", "SMOKE_DIAGNOSIS.json"), encoding="utf-8"))
smoke_orig = json.load(open(os.path.join(ROOT, "depth", "SMOKE.json"), encoding="utf-8"))
smoke_amend = json.load(open(os.path.join(ROOT, "depth_amended", "SMOKE.json"), encoding="utf-8"))
ledger = json.load(open(os.path.join(ROOT, "EXPERIMENT_LEDGER.json"), encoding="utf-8"))
var_loss = json.load(open(os.path.join(ROOT, "variance", "VARIANCE_LOSS_DECOMPOSITION.json"), encoding="utf-8"))

REF = {"entropy": (35.4444, 0.730111, 0.625426), "var15": (35.9610, 0.737786, 0.625781),
       "length": 33.6944, "random": 20.2750}

# ======================================================================================
# Figure 1: PB vs within scatter
# ======================================================================================
def family_of(row):
    m, exp, panel = row["method"], row["experiment"], row["panel"]
    if panel == "simple_control":
        return "simple control (longest / random step)"
    if panel == "other_answer_diagnostic":
        return "other-answer diagnostic (supervised / unsupervised correction)"
    if m in ("entropy__old", "var15__old", "var50__old"):
        return "simple reference (entropy, varentropy)"
    if exp == "literature-depth_amended":
        return "depth: stacked second layer on H4"
    if exp == "literature-temporal":
        return "temporal: token-chain Markov / shuffled / reset"
    if exp == "literature-variance":
        return "variance: two-state Gaussian shared / separate"
    if "exact4" in m or "cd4" in m:
        return "four hidden units (exact / CD-10 / best-of-3)"
    if exp == "rbm-position-fusion-v1" and m != "shared__max":
        return "position-conditioned weights"
    if exp == "dufs-moment-selection-v1" and m in ("dufs6__rbm", "correlation6__rbm"):
        return "column selection (DUFS / low-corr) + RBM"
    if "_iu" in m or "__equal" in m or "_equal" in m:
        return "IU-PCR / equal-weight fusion"
    if "initial" in m:
        return "RBM before training (fixed initial coefficients)"
    return "single-unit RBM, trained (moment banks, CD-10, best-of-3, shrinkage, diag.)"


FAM_STYLE = {
    "simple reference (entropy, varentropy)": ("black", "^", 90),
    "simple control (longest / random step)": ("dimgray", "X", 90),
    "other-answer diagnostic (supervised / unsupervised correction)": ("crimson", "*", 160),
    "single-unit RBM, trained (moment banks, CD-10, best-of-3, shrinkage, diag.)": ("tab:blue", "o", 40),
    "RBM before training (fixed initial coefficients)": ("tab:cyan", "s", 40),
    "IU-PCR / equal-weight fusion": ("tab:olive", "P", 55),
    "column selection (DUFS / low-corr) + RBM": ("tab:green", "D", 45),
    "position-conditioned weights": ("tab:brown", "v", 50),
    "variance: two-state Gaussian shared / separate": ("tab:purple", "h", 55),
    "four hidden units (exact / CD-10 / best-of-3)": ("tab:orange", "o", 45),
    "temporal: token-chain Markov / shuffled / reset": ("tab:pink", "d", 45),
    "depth: stacked second layer on H4": ("tab:red", "s", 45),
}

LABELS = {
    "entropy__old": "Token entropy", "var15__old": "Varentropy15", "var50__old": "Varentropy50",
    "rbm6__old": "RBM6 posterior", "rbm12__logit_old": "RBM12 logit", "rbm12__old": "RBM12 posterior",
    "b12_variance_shared_posterior": "Shared-var. bank12 post.",
    "b12_variance_separate_logit": "Separate-var. bank12 logit",
    "b6_variance_separate_logit": "Separate-var. bank6 logit",
    "correlation6__rbm": "Low-corr 6/12 RBM", "dufs6__rbm": "DUFS 6/12 RBM",
    "b6_exact4_posterior": "Exact H4 bank6 post.", "b12_exact4_logit": "Exact H4 bank12 logit",
    "b6_best_exact4_posterior": "Best-of-3 H4 bank6", "b12_best_exact4_logit": "Best-of-3 H4 bank12",
    "power48_rbm": "48-col RBM trained", "power48_initial": "48-col RBM initial",
    "b6_layer2_logit_exact_posterior": "Depth amend. exact L2, bank6 post.",
    "b12_layer2_logit_exact_logit": "Depth amend. exact L2, bank12 logit",
    "b6_layer2_logit_cd_posterior": "Depth amend. CD L2, bank6 post.",
    "b12_layer2_logit_cd_logit": "Depth amend. CD L2, bank12 logit",
    "b12_layer2_logit_cd_posterior": "Depth amend. CD L2, bank12 post.",
    "b6_layer2_exact_posterior": "Depth orig. exact L2, bank6 (cov .990)",
    "b12_layer2_exact_logit": "Depth orig. exact L2, bank12 (cov .974)",
    "supervised_update": "Supervised step-BCE (diagnostic)",
    "length__old": "Longest-step control", "random__old": "Random-step control",
    "var15_equal__old": "Var15 contrib. equal", "var15_iu__old": "Var15 contrib. IU-PCR",
    "d6__iu": "IU-PCR, 12-moment bank", "initial6__logit_old": "RBM6 initial, logit readout",
    "b6_cd1_logit": "CD-10 H1 bank6 logit", "b12_cd1_posterior": "CD-10 H1 bank12 post.",
    "initial12__old": "RBM12 initial post.", "b6_exact1_logit": "RBM6 logit",
    "position__max": "Position-conditioned RBM12",
}

OFFSETS = {"rbm12__logit_old": (4, -9), "initial12__old": (4, 6), "b12_cd1_posterior": (-78, 8),
           "var50__old": (4, 6), "position__max": (4, -9), "var15_equal__old": (-20, 8),
           "rbm12__old": (-60, 6), "power48_initial": (4, 5), "rbm6__old": (4, -9), "b6_exact1_logit": (4, -9)}
f1 = fus[fus.panel.isin(["answer_local", "simple_control", "other_answer_diagnostic"])].copy()
f1["family"] = f1.apply(family_of, axis=1)
f1["hollow"] = f1["valid_answers"] < N_ANSWERS

fig, axes = plt.subplots(1, 2, figsize=(16, 8.2))
for ax, zoom in zip(axes, [False, True]):
    for fam, (c, mk, sz) in FAM_STYLE.items():
        sub = f1[f1.family == fam]
        if sub.empty:
            continue
        full = sub[~sub.hollow]
        hol = sub[sub.hollow]
        ax.scatter(full.prm_within, full.pb_macro_percent, c=c, marker=mk, s=sz, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=fam, zorder=3)
        if not hol.empty:
            ax.scatter(hol.prm_within, hol.pb_macro_percent, facecolors="none", edgecolors=c, marker=mk,
                       s=sz + 20, linewidths=1.2, label=fam + " [coverage < 1]", zorder=3)
    ax.axhline(REF["length"], color="dimgray", ls="--", lw=1, zorder=1)
    ax.axhline(REF["random"], color="dimgray", ls=":", lw=1, zorder=1)
    ax.axhline(REF["entropy"][0], color="black", ls="-", lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(REF["entropy"][1], color="black", ls="-", lw=0.6, alpha=0.5, zorder=1)
    ax.set_xlabel("PRMBench within-answer AUC (n = 6,030 fitted answers; 5,914 / 6,022 for depth rows with failures)")
    ax.set_ylabel("ProcessBench macro F1 over 8 cells, % (full population; failures = missed decisions)")
    ax.grid(True, lw=0.4, alpha=0.5)
    if zoom:
        ax.set_xlim(0.727, 0.7505)
        ax.set_ylim(33.2, 37.8)
        ax.set_title("Zoom on the reference band (same points)")
        sel = f1[(f1.prm_within > 0.727) & (f1.pb_macro_percent > 33.2)]
    else:
        ax.set_xlim(0.485, 0.755)
        ax.set_ylim(17.5, 38.5)
        ax.set_title("110 answer-local rows + 2 simple controls + 3 other-answer diagnostics")
        ax.text(0.487, REF["length"] + 0.25, "longest-step control 33.69", fontsize=8, color="dimgray")
        ax.text(0.487, REF["random"] + 0.25, "random-step control 20.28", fontsize=8, color="dimgray")
        sel = f1[(f1.prm_within <= 0.727) | (f1.pb_macro_percent <= 33.2)]
    seen = set()
    for _, r in sel.iterrows():
        if r.method in LABELS and r.method not in seen:
            key = (round(r.pb_macro_percent, 4), round(r.prm_within, 5))
            if key in seen:
                continue
            seen.add(r.method); seen.add(key)
            off = OFFSETS.get(r.method, (4, 3))
            ax.annotate(LABELS[r.method], (r.prm_within, r.pb_macro_percent), fontsize=7,
                        xytext=off, textcoords="offset points", zorder=4)
axes[0].legend(loc="center left", bbox_to_anchor=(0.005, 0.47), framealpha=0.95, markerscale=0.9)
fig.suptitle("ProcessBench macro F1 versus PRMBench within-answer AUC, every completed RBM-related row "
             "(development evidence; 110 answer-local rows hold 91 distinct score sets because H1 references are re-listed in each suite)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_pb_vs_within_scatter.png"), dpi=DPI)
plt.close(fig)

# ======================================================================================
# Figure 2: forest plot of primary contrasts (parsed from STAGE1_CONTRASTS.md, cross-checked to METRICS)
# ======================================================================================
rows = []
with open(os.path.join(ROOT, "STAGE1_CONTRASTS.md"), encoding="utf-8") as fh:
    for line in fh:
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Suite"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        suite, contrast, pb_d, pb_ci, w_d, w_ci, gl = parts
        ci = lambda s: [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)]
        rows.append(dict(suite=suite, contrast=contrast, pb_delta=float(pb_d.replace("+", "")), pb_ci=ci(pb_ci),
                         w_delta=float(w_d.replace("+", "")), w_ci=ci(w_ci), gl=gl))
md_rows = pd.DataFrame(rows)

# cross-check to METRICS.json
n_ok = 0
for r in rows:
    s = r["suite"].split(" ")[0]
    m = metrics[s]
    a, b = [x.strip() for x in r["contrast"].split(" − ")]
    # find by display names in suite COMPARISON? contrasts are keyed by method ids; map via fus display_name
    name2m = dict(zip(fus.display_name.str.replace(r" \[coverage.*\]", "", regex=True), fus.method))
    key = f"{name2m[a]}_minus_{name2m[b]}"
    src = m["conditional_contrasts"] if "conditional" in r["suite"] else m["contrasts"]
    c = src[key]
    okp = check(f"STAGE1_CONTRASTS.md pb_delta {key} ({'cond' if 'conditional' in r['suite'] else 'full'})",
                r["pb_delta"], c["pb_delta"] * 100, tol=5e-4)
    okc = check(f"STAGE1_CONTRASTS.md pb_ci_low {key}", r["pb_ci"][0], c["pb_ci"][0] * 100, tol=5e-4)
    okc2 = check(f"STAGE1_CONTRASTS.md pb_ci_high {key}", r["pb_ci"][1], c["pb_ci"][1] * 100, tol=5e-4)
    okw = check(f"STAGE1_CONTRASTS.md within_delta {key}", r["w_delta"], c["prm_within_delta_common"], tol=5e-6)
    okw1 = check(f"STAGE1_CONTRASTS.md within_ci {key}", r["w_ci"][0], c["prm_within_ci"][0], tol=5e-6)
    okw2 = check(f"STAGE1_CONTRASTS.md within_ci_high {key}", r["w_ci"][1], c["prm_within_ci"][1], tol=5e-6)
    check_bool(f"STAGE1_CONTRASTS.md primary flag {key}", bool(c.get("primary")) and c["ci_level"] == 0.975)
    r["gained"], r["lost"] = c.get("gained"), c.get("lost")
    r["lost_early"], r["lost_late"], r["lost_failure"] = c.get("lost_early"), c.get("lost_late"), c.get("lost_failure")

def short(s):
    s = s.replace("Gaussian fusion, ", "Gauss. ").replace("state variance", "var.").replace("state variances", "var.")
    s = s.replace("RBM, ", "").replace(" hidden units", "H").replace(" hidden unit", "H").replace("exact training", "exact")
    s = s.replace("best density fit of 3 starts", "best-of-3 starts").replace("RBM with token sequence fusion across steps", "token-chain Markov")
    s = s.replace("RBM with shuffled token order, control", "shuffled-order control")
    s = s.replace("Stacked RBMs, 4 to 1H, exact second layer", "stacked 4->1, exact 2nd layer")
    s = s.replace(" features", "f")
    return s

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(rows))[::-1]
ylabels = []
for r in rows:
    a, b = [x.strip() for x in r["contrast"].split(" − ")]
    tag = r["suite"] if "conditional" not in r["suite"] else r["suite"].replace("depth_amended (conditional, common covered answers ", "depth cond. (")
    gl = f"  gained/lost {r['gained']}/{r['lost']}" if r["gained"] is not None else ""
    ylabels.append(f"[{tag}]{gl}\n{short(a)}\n  −  {short(b)}")
for ax, key, lab, scale in [(axes[0], "pb", "ProcessBench macro F1 delta, percentage points (97.5% paired source-group bootstrap)", 1),
                            (axes[1], "w", "PRMBench within-answer AUC delta (97.5% paired source-group bootstrap)", 1)]:
    for yi, r in zip(y, rows):
        d, lo, hi = r[f"{key}_delta"], r[f"{key}_ci"][0], r[f"{key}_ci"][1]
        col = "tab:red" if hi < 0 else ("tab:green" if lo > 0 else "tab:gray")
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ecolor=col, capsize=3, ms=5, lw=1.4)
        ax.text(hi, yi + 0.22, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if key == "w" else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                fontsize=6.8, va="bottom", ha="left", color=col)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(lab)
    ax.grid(True, axis="x", lw=0.4, alpha=0.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels(ylabels, fontsize=7)
axes[0].set_xlim(-19, 5)
axes[1].set_xlim(-0.165, 0.065)
fig.suptitle("Every pre-registered PRIMARY contrast of the five suites (rows from STAGE1_CONTRASTS.md; all values re-read from METRICS.json)\n"
             "red = interval entirely below zero, gray = interval includes zero; no interval lies entirely above zero", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_primary_contrasts_forest.png"), dpi=DPI)
plt.close(fig)

# ======================================================================================
# Figure 3: capacity convergence
# ======================================================================================
usecols = ["bank", "exact4_gradient_max", "exact4_gain_vs_exact1", "exact4_converged", "exact1_converged",
           "surviving_views", "saturated_units", "duplicate_units", "dead_units", "depth_would_fail", "is_pb",
           "pb_gained", "pb_lost", "cell"]
conv = pd.read_csv(os.path.join(ROOT, "capacity", "CAPACITY_CONVERGENCE.csv"), usecols=usecols)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = {6: "tab:blue", 12: "tab:orange"}
ax = axes[0, 0]
bins = np.logspace(-4, 0, 50)
for b in (6, 12):
    g = conv[conv.bank == b].exact4_gradient_max.clip(lower=1e-4)
    ax.hist(g, bins=bins, alpha=0.55, color=colors[b], label=f"bank{b} (median {g.median():.3f})")
ax.axvline(1e-6 * 100, color="gray", lw=0.8, ls=":")  # gtol is 1e-6, off-scale; mark left edge
ax.set_xscale("log")
ax.set_xlabel("exact-H4 final max |gradient| (L-BFGS-B, maxiter 100; gtol = 1e-6 is 2-5 orders of magnitude to the left)")
ax.set_ylabel("fits")
ax.set_title("(a) Every exact-H4 fit stopped at the iteration cap")
ax.legend()
ax = axes[0, 1]
for b in (6, 12):
    g = conv[conv.bank == b].exact4_gain_vs_exact1
    ax.hist(g, bins=60, alpha=0.55, color=colors[b], label=f"bank{b} (median {g.median():.3f}; min {g.min():.3f})")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("NLL gain of exact H4 over exact H1, nats per token (positive = H4 fits the density better)")
ax.set_ylabel("answers")
ax.set_title("(b) H4 fits the density better than H1 on every answer")
ax.legend()
ax = axes[1, 0]
qs = ["q1", "q2", "q3", "q4"]
w = 0.2
for i, b in enumerate((6, 12)):
    d = conv_json[f"bank{b}"]["pb_vs_exact1"]["by_gradient_quartile"]
    gained = [d[q]["gained"] for q in qs]
    lost = [d[q]["lost"] for q in qs]
    x = np.arange(4)
    ax.bar(x + (2 * i - 1.5) * w, gained, w, color=colors[b], alpha=0.9, label=f"bank{b} gained")
    ax.bar(x + (2 * i - 0.5) * w, lost, w, color=colors[b], alpha=0.4, hatch="//", label=f"bank{b} lost")
    for xi, (gg, ll, q) in enumerate(zip(gained, lost, qs)):
        ax.text(xi + (2 * i - 1.5) * w, gg + 3, str(gg), ha="center", fontsize=7)
        ax.text(xi + (2 * i - 0.5) * w, ll + 3, str(ll), ha="center", fontsize=7)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(["q1 (best optimized)", "q2", "q3", "q4 (least optimized)"])
ax.set_ylabel("PB exact successes changed, exact H4 vs exact H1")
ax.set_title("(c) Losses exceed gains in every gradient quartile, including the best-optimized one")
ax.legend(ncol=2)
ax = axes[1, 1]
for i, b in enumerate((6, 12)):
    h = conv_json[f"bank{b}"]["units"]["surviving_views_hist"]
    x = np.arange(5)
    vals = [h[str(k)] for k in range(5)]
    ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colors[b], label=f"bank{b}")
    for xi, v in zip(x, vals):
        if v:
            ax.text(xi + (i - 0.5) * 0.38, v + 100, str(v), ha="center", fontsize=7.5)
ax.set_xticks(np.arange(5))
ax.set_xlabel("varying hidden posterior views per exact-H4 fit (of 4); < 3 = depth second layer cannot fit")
ax.set_ylabel("answers")
ax.set_title("(d) Surviving views: 140 (bank6) / 353 (bank12) answers have only two")
ax.legend()
fig.suptitle("Capacity suite, exact-H4 optimization and representation diagnostics (CAPACITY_CONVERGENCE.csv/.json; descriptive, associations only)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_capacity_convergence.png"), dpi=DPI)
plt.close(fig)

# capacity checks against the account
for b in (6, 12):
    sub = conv[conv.bank == b]
    cj = conv_json[f"bank{b}"]
    check(f"capacity bank{b} exact4 nonconverged (csv)", int((~sub.exact4_converged).sum()), cj["exact4_nonconverged"], 0)
    check(f"capacity bank{b} exact1 nonconverged (csv)", int((~sub.exact1_converged).sum()), cj["exact1_nonconverged"], 0)
    check(f"capacity bank{b} median gradient", sub.exact4_gradient_max.median(), {6: 0.031, 12: 0.072}[b], 5e-4, "account section 2")
    check(f"capacity bank{b} median H4-H1 NLL gain", sub.exact4_gain_vs_exact1.median(), {6: 0.98, 12: 1.78}[b], 5e-3, "account section 2")
    check(f"capacity bank{b} two-view answers", int((sub.surviving_views == 2).sum()), {6: 140, 12: 353}[b], 0)
    check(f"capacity bank{b} depth_would_fail sum", int(sub.depth_would_fail.sum()), {6: 140, 12: 353}[b], 0)
    check(f"capacity bank{b} pct fits with 1 saturated unit", 100 * (sub.saturated_units == 1).mean(), {6: 9.9, 12: 16.6}[b], 0.05, "account: 9.9% / 16.6%")
    check(f"capacity bank{b} pct fits with 2 saturated units", 100 * (sub.saturated_units == 2).mean(), {6: 1.0, 12: 2.6}[b], 0.05, "account: 1.0% / 2.6%")
    check(f"capacity bank{b} pct fits with duplicate unit", 100 * (sub.duplicate_units > 0).mean(), 0.4, 0.05, "account: 0.4%")
    check(f"capacity bank{b} dead units", int(sub.dead_units.sum()), 0, 0)
    check(f"capacity bank{b} PB gained (csv)", int(sub.pb_gained.sum()), cj["pb_vs_exact1"]["gained"], 0)
    check(f"capacity bank{b} PB lost (csv)", int(sub.pb_lost.sum()), cj["pb_vs_exact1"]["lost"], 0)
    check(f"capacity bank{b} expected coverage", cj["depth_expected_coverage"], (N_ANSWERS - {6: 140, 12: 353}[b]) / N_ANSWERS, 1e-9)
    sat0 = cj["pb_vs_exact1"]["by_unit_condition"]["saturated_units"]["0"]
    CHECKS["notes"].append(f"bank{b}: losses with 0 saturated units {sat0['lost']}/{cj['pb_vs_exact1']['lost']} = {100*sat0['lost']/cj['pb_vs_exact1']['lost']:.1f}% (account: 'fits with no saturated unit account for most losses')")
check("capacity bank6 q4 gained", conv_json["bank6"]["pb_vs_exact1"]["by_gradient_quartile"]["q4"]["gained"], 114, 0)
check("capacity bank6 q4 lost", conv_json["bank6"]["pb_vs_exact1"]["by_gradient_quartile"]["q4"]["lost"], 333, 0)
check("capacity bank6 q1 gained", conv_json["bank6"]["pb_vs_exact1"]["by_gradient_quartile"]["q1"]["gained"], 24, 0)
check("capacity bank6 q1 lost", conv_json["bank6"]["pb_vs_exact1"]["by_gradient_quartile"]["q1"]["lost"], 48, 0)
# capacity primary contrasts match ledger nonconverged counts
led = {e["suite"]: e for e in ledger}
check("ledger capacity b6_exact4 nonconverged", led["capacity"]["fit_summary"]["b6_exact4"]["nonconverged"], 13768, 0)
check("ledger capacity b12_exact4 nonconverged", led["capacity"]["fit_summary"]["b12_exact4"]["nonconverged"], 13769, 0)
del conv

# ======================================================================================
# Figure 4: maxiter probe
# ======================================================================================
pr = pd.DataFrame([{"bank": r["bank"], "nll100": r["budget100"]["nll"], "nll_probe": r["budget_probe"]["nll"],
                    "conv": r["budget_probe"]["converged"], "iters": r["budget_probe"]["iterations"],
                    "sec": r["budget_probe"]["seconds"], "peak_logit": r["peaks_changed"]["logit"],
                    "peak_post": r["peaks_changed"]["posterior"], "dec": r["nll_decrease_vs_budget100"]} for r in probe["rows"]])
fig, ax = plt.subplots(figsize=(8.5, 7))
lim = [min(pr.nll100.min(), pr.nll_probe.min()) - 0.2, max(pr.nll100.max(), pr.nll_probe.max()) + 0.2]
ax.plot(lim, lim, color="black", lw=0.8, label="x = y")
for b in (6, 12):
    s = pr[pr.bank == b]
    chg = s[(s.peak_logit) | (s.peak_post)]
    same = s[~((s.peak_logit) | (s.peak_post))]
    ax.scatter(same.nll100, same.nll_probe, color=colors[b], s=45, label=f"bank{b}, peak unchanged (n={len(same)})", zorder=3)
    ax.scatter(chg.nll100, chg.nll_probe, facecolors="none", edgecolors=colors[b], marker="D", s=95, lw=1.6,
               label=f"bank{b}, top-10 peak moved in logit or posterior (n={len(chg)})", zorder=4)
    nc = s[~s.conv]
    ax.scatter(nc.nll100, nc.nll_probe, marker="x", color="black", s=40, zorder=5, label=f"bank{b}, still unconverged at 1000 (n={len(nc)})" if len(nc) else None)
ax.set_xlabel("exact-H4 NLL per token at the registered budget (maxiter 100)")
ax.set_ylabel("exact-H4 NLL per token after refit at maxiter 1000 (same seeded start)")
ax.set_title(f"maxiter probe on the 27 capacity smoke answers (54 exact-H4 fits)\n{int(pr.conv.sum())}/{len(pr)} converge at maxiter 1000, "
             f"median NLL decrease {pr.dec.median():.3f}, top-10 peak moved: logit {int(pr.peak_logit.sum())} / posterior {int(pr.peak_post.sum())} of {len(pr)}", fontsize=9.5)
ax.text(0.02, 0.97, "FEASIBILITY ONLY\nno benchmark inference, no candidate,\nnot a full-population refit (that is a new registered experiment)",
        transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="lightyellow", ec="goldenrod"))
ax.legend(loc="lower right", fontsize=7.5)
ax.grid(True, lw=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_capacity_probe.png"), dpi=DPI)
plt.close(fig)
check("probe n", len(pr), 54, 0)
check("probe converged", int(pr.conv.sum()), 48, 0)
check("probe median NLL decrease", pr.dec.median(), 0.47, 5e-3)
check("probe peaks changed logit", int(pr.peak_logit.sum()), 7, 0)
check("probe peaks changed posterior", int(pr.peak_post.sum()), 8, 0)
check("probe max seconds <= 1.4", pr.sec.max(), 1.4, 1.4)  # tol allows anything <=2.8; refine:
check_bool("probe max seconds <= 1.4 (strict)", pr.sec.max() <= 1.4, f"max seconds {pr.sec.max():.3f}")
check("probe median iterations (account: about 400)", pr[pr.conv].iters.median(), 400, 60)

# ======================================================================================
# Figure 5: stability starts (stream FIT_HEALTH.json)
# ======================================================================================
def stream_objects(path):
    """Yield the top-level objects of a JSON list one at a time (the 48 MB text is held once;
    only one decoded object is alive at a time)."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    dec = json.JSONDecoder()
    i = text.index("[") + 1
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        obj, i = dec.raw_decode(text, i)
        yield obj

MODELS = ["b6_best_exact1", "b6_best_exact4", "b12_best_exact1", "b12_best_exact4"]
acc = {m: {"spread": [], "improve": [], "unique_peaks": [], "chosen": [], "conv12": [], "n": 0} for m in MODELS}
n_entries = 0
for e in stream_objects(os.path.join(ROOT, "stability", "FIT_HEALTH.json")):
    n_entries += 1
    for m in MODELS:
        mm = e["models"][m]
        nll = [s["nll_final"] for s in mm["starts"]]
        acc[m]["spread"].append(max(nll) - min(nll))
        acc[m]["improve"].append(nll[0] - min(nll))
        acc[m]["unique_peaks"].append(mm["unique_peaks"])
        acc[m]["chosen"].append(mm["chosen_start"])
        acc[m]["conv12"].extend([bool(s.get("converged")) for s in mm["starts"][1:]])
        acc[m]["n"] += 1
for m in MODELS:
    for k in ("spread", "improve", "unique_peaks", "chosen", "conv12"):
        acc[m][k] = np.asarray(acc[m][k])
check("stability FIT_HEALTH entries", n_entries, N_ANSWERS, 0)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
bins = np.logspace(-9, 1, 60)
for m, c in zip(["b6_best_exact1", "b12_best_exact1"], ["tab:blue", "tab:orange"]):
    s = np.clip(acc[m]["spread"], 1e-9, None)
    ax.hist(s, bins=bins, alpha=0.55, color=c, label=f"{m.split('_')[0]} H1 (median {np.median(acc[m]['spread']):.2e}, p95 {np.percentile(acc[m]['spread'], 95):.2e})")
ax.set_xscale("log")
ax.set_xlabel("NLL spread across the 3 exact starts, max - min (nats/token; values < 1e-9 clipped)")
ax.set_ylabel("answers")
ax.set_title("(a) H1: the three starts reach the same NLL")
ax.legend()
ax = axes[0, 1]
bins = np.logspace(-5, 1, 60)
for m, c in zip(["b6_best_exact4", "b12_best_exact4"], ["tab:blue", "tab:orange"]):
    s = np.clip(acc[m]["spread"], 1e-5, None)
    ax.hist(s, bins=bins, alpha=0.55, color=c, label=f"{m.split('_')[0]} H4 (median {np.median(acc[m]['spread']):.3f}, p95 {np.percentile(acc[m]['spread'], 95):.3f})")
ax.set_xscale("log")
ax.set_xlabel("NLL spread across the 3 exact starts, max - min (nats/token)")
ax.set_ylabel("answers")
ax.set_title("(b) H4 at maxiter 100: starts end at different NLL")
ax.legend()
ax = axes[1, 0]
x = np.arange(3)
w = 0.2
for i, (m, c) in enumerate(zip(MODELS, ["tab:blue", "tab:blue", "tab:orange", "tab:orange"])):
    up = acc[m]["unique_peaks"]
    vals = [int((up == k).sum()) for k in (1, 2, 3)]
    alpha = 0.9 if "exact1" in m else 0.45
    ax.bar(x + (i - 1.5) * w, vals, w, color=c, alpha=alpha, hatch="" if "exact1" in m else "//", label=f"{m.replace('_best_', ' ')}")
    for xi, v in zip(x, vals):
        ax.text(xi + (i - 1.5) * w, v + 0.01 * N_ANSWERS, str(v), ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(["1 (all starts agree)", "2", "3 (all differ)"])
ax.set_xlabel("number of distinct top-10 peaks across the 3 starts")
ax.set_ylabel("answers")
ax.set_title("(c) Peak agreement across starts")
ax.legend()
ax = axes[1, 1]
bins = np.linspace(-0.05, 1.5, 60)
for m, c in zip(["b6_best_exact4", "b12_best_exact4"], ["tab:blue", "tab:orange"]):
    s = acc[m]["improve"]
    ax.hist(np.clip(s, -0.05, 1.5), bins=bins, alpha=0.55, color=c,
            label=f"{m.split('_')[0]} H4 (median {np.median(s):.3f}; chosen start 0/1/2: "
                  f"{(acc[m]['chosen']==0).sum()}/{(acc[m]['chosen']==1).sum()}/{(acc[m]['chosen']==2).sum()})")
ax.set_xlabel("NLL improvement of the selected (lowest-NLL) start over the capacity start, nats/token")
ax.set_ylabel("answers")
ax.set_title("(d) H4: how much lower the chosen start's NLL is than the capacity fit")
ax.legend(fontsize=7)
fig.suptitle("Stability suite: three exact starts per answer (streamed from stability/FIT_HEALTH.json, 13,769 entries)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_stability_starts.png"), dpi=DPI)
plt.close(fig)

# stability checks against the account section 4
check("stability H1 bank6 same peak count", int((acc["b6_best_exact1"]["unique_peaks"] == 1).sum()), 13768, 0)
check("stability H1 bank12 same peak count", int((acc["b12_best_exact1"]["unique_peaks"] == 1).sum()), 13705, 0)
check("stability H1 bank6 median spread == 0", float(np.median(acc["b6_best_exact1"]["spread"])), 0.0, 1e-9)
check("stability H1 bank12 median spread == 0", float(np.median(acc["b12_best_exact1"]["spread"])), 0.0, 1e-9)
check_bool("stability H1 p95 spread <= 0.01 both banks", max(np.percentile(acc["b6_best_exact1"]["spread"], 95), np.percentile(acc["b12_best_exact1"]["spread"], 95)) <= 0.01,
           f"p95 bank6 {np.percentile(acc['b6_best_exact1']['spread'], 95):.2e}, bank12 {np.percentile(acc['b12_best_exact1']['spread'], 95):.2e}")
check("stability H1 restart convergence % bank6", 100 * acc["b6_best_exact1"]["conv12"].mean(), 99.99, 0.006)
check("stability H1 restart convergence % bank12", 100 * acc["b12_best_exact1"]["conv12"].mean(), 100.0, 0.006)
check("stability H4 restart convergence % bank6", 100 * acc["b6_best_exact4"]["conv12"].mean(), 0.004, 0.001)
check("stability H4 restart convergence % bank12", 100 * acc["b12_best_exact4"]["conv12"].mean(), 0.0, 0.001)
check("stability H4 bank6 median spread", float(np.median(acc["b6_best_exact4"]["spread"])), 0.056, 5e-4)
check("stability H4 bank12 median spread", float(np.median(acc["b12_best_exact4"]["spread"])), 0.23, 5e-3)
check("stability H4 bank6 median improvement over capacity start", float(np.median(acc["b6_best_exact4"]["improve"])), 0.009, 5e-4)
check("stability H4 bank12 median improvement over capacity start", float(np.median(acc["b12_best_exact4"]["improve"])), 0.070, 5e-4)
check("stability H4 bank6 peak disagreement %", 100 * (acc["b6_best_exact4"]["unique_peaks"] > 1).mean(), 18, 0.5)
check("stability H4 bank12 peak disagreement %", 100 * (acc["b12_best_exact4"]["unique_peaks"] > 1).mean(), 33, 0.5)
c = metrics["stability"]["contrasts"]["b12_best_exact4_logit_minus_b12_exact4_logit"]
check("stability bank12 lost_early", c["lost_early"], 114, 0)
check("stability bank12 lost_late", c["lost_late"], 19, 0)
check("stability bank12 lost", c["lost"], 133, 0)
del acc

# ======================================================================================
# Figure 6: depth coverage and performance
# ======================================================================================
da = comp["depth_amended"].set_index("method")
cfg = [("b6_exact1_posterior", "exact H1\n(ref.)"), ("b6_exact4_posterior", "exact H4\n(ref.)"),
       ("b6_layer2_exact_posterior", "orig. exact L2\non posteriors"), ("b6_layer2_cd_posterior", "orig. CD-10 L2\non posteriors"),
       ("b6_layer2_logit_exact_posterior", "amend. exact L2\non logits"), ("b6_layer2_logit_cd_posterior", "amend. CD-10 L2\non logits"),
       ("b12_exact1_logit", "exact H1\n(ref.)"), ("b12_exact4_logit", "exact H4\n(ref.)"),
       ("b12_layer2_exact_logit", "orig. exact L2\non posteriors"), ("b12_layer2_cd_logit", "orig. CD-10 L2\non posteriors"),
       ("b12_layer2_logit_exact_logit", "amend. exact L2\non logits"), ("b12_layer2_logit_cd_logit", "amend. CD-10 L2\non logits")]
fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), gridspec_kw={"height_ratios": [1.3, 1]})
ax = axes[0]
x = np.arange(len(cfg))
x = x + (x >= 6) * 0.8
full = [100 * da.loc[m, "pb_all8"] for m, _ in cfg]
cov_pb = [100 * da.loc[m, "pb_all8_conditional"] for m, _ in cfg]
cover = [da.loc[m, "coverage"] for m, _ in cfg]
ax.bar(x - 0.2, full, 0.4, color="tab:blue", label="PB macro F1 %, full population (failures = missed decisions)")
ax.bar(x + 0.2, cov_pb, 0.4, color="tab:cyan", label="PB macro F1 %, covered answers only (conditional; not comparable without coverage)")
for xi, f, cpb in zip(x, full, cov_pb):
    ax.text(xi - 0.2, f + 0.4, f"{f:.2f}", ha="center", fontsize=7)
    ax.text(xi + 0.2, cpb + 0.4, f"{cpb:.2f}", ha="center", fontsize=7, color="teal")
ax.axhline(REF["entropy"][0], color="black", lw=0.8, ls="-", label="token entropy 35.44")
ax.axhline(REF["length"], color="dimgray", lw=0.8, ls="--", label="longest-step control 33.69")
ax.set_ylim(0, 46)
ax.set_ylabel("ProcessBench macro F1, %")
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in cfg], fontsize=7.5)
ax.text(2.5, 44.3, "bank6, posterior readout (retained)", ha="center", fontsize=9, weight="bold")
ax.text(9.3, 44.3, "bank12, logit readout (retained)", ha="center", fontsize=9, weight="bold")
ax2 = ax.twinx()
ax2.plot(x, cover, "kD", ms=6, label="coverage (valid answers / 13,769)")
for xi, cv, (m, _) in zip(x, cover, cfg):
    ax2.text(xi, cv + 0.004, f"cov {cv:.4f} ({int(da.loc[m,'valid_answers'])})", ha="center", va="bottom", fontsize=6.2)
ax2.set_ylim(0.78, 1.045)
ax2.set_ylabel("coverage")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7)
ax.set_title("Depth suite (amended run): stacked 4->1 second layer over the saved maxiter-100 exact-H4 first layer; original posterior-input variants carry 140 / 353 declared COLLAPSED_HIDDEN_VIEWS failures")
ax = axes[1]
within = [da.loc[m, "prm_within"] for m, _ in cfg]
wn = [int(da.loc[m, "prm_within_n"]) for m, _ in cfg]
pooled = [da.loc[m, "prm_pooled"] for m, _ in cfg]
ax.bar(x - 0.2, within, 0.4, color="tab:purple", label="PRMBench within-answer AUC (over covered answers)")
ax.bar(x + 0.2, pooled, 0.4, color="plum", label="PRMBench pooled AUC")
for xi, wv, n, pv in zip(x, within, wn, pooled):
    ax.text(xi - 0.2, wv + 0.004, f"{wv:.4f}\nn={n}", ha="center", fontsize=6.5)
    ax.text(xi + 0.2, pv + 0.004, f"{pv:.3f}", ha="center", fontsize=6.5, color="purple")
ax.axhline(REF["entropy"][1], color="black", lw=0.8, label="token entropy within 0.7301")
ax.set_ylim(0.45, 0.80)
ax.set_ylabel("AUC")
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in cfg], fontsize=7.5)
ax.legend(loc="lower left", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_depth_coverage_and_performance.png"), dpi=DPI)
plt.close(fig)

# depth checks
mf = metrics["depth_amended"]["failures"]
for k, exp in [("b6_layer2_exact", 140), ("b6_layer2_cd", 140), ("b12_layer2_exact", 353), ("b12_layer2_cd", 353)]:
    check(f"depth failures {k} count", mf[k]["count"], exp, 0)
    check(f"depth failures {k} named_collapse", mf[k]["named_collapse"], exp, 0)
    check_bool(f"depth failures {k} no other kinds", len(mf[k]["other"]) == 0)
    check(f"ledger depth_amended {k} failures", led["depth_amended"]["fit_summary"][k]["failures"], exp, 0)
for m, n in [("b6_layer2_exact_posterior", 13629), ("b12_layer2_exact_logit", 13416)]:
    check(f"depth coverage {m} = valid/13769", da.loc[m, "coverage"], n / N_ANSWERS, 1e-9)
    check(f"depth valid_answers {m}", da.loc[m, "valid_answers"], n, 0)
check("smoke_diagnosis expected failures bank6", smoke_diag["expected_full_population_failures"]["bank6"], 140, 0)
check("smoke_diagnosis expected failures bank12", smoke_diag["expected_full_population_failures"]["bank12"], 353, 0)
n_fail_records = sum(len(f) for f in smoke_orig["failures"])
n_fail_answers = sum(1 for f in smoke_orig["failures"] if f)
check("depth original smoke failed records", n_fail_records, 14, 0)
check("depth original smoke failed answers", n_fail_answers, 6, 0)
check("depth amended smoke exact replays", smoke_amend["original_smoke_replay"]["exact_replays"], 188, 0)
check("depth amended smoke replays == (108-14 records) x 2 readouts", (27 * 4 - n_fail_records) * 2, 188, 0,
      "account wording '188 exact replays of the 21 non-failing original records': 21 is the non-failing ANSWER count; non-failing RECORDS are 94")
check("depth amendment gain bank6 exact (logit-in posterior minus posterior-in posterior), pp", 100 * (da.loc["b6_layer2_logit_exact_posterior", "pb_all8"] - da.loc["b6_layer2_exact_posterior", "pb_all8"]), 12.4, 0.05)
check("depth amendment gain bank12 exact (logit-in logit minus posterior-in logit), pp", 100 * (da.loc["b12_layer2_logit_exact_logit", "pb_all8"] - da.loc["b12_layer2_exact_logit", "pb_all8"]), 10.7, 0.05)
# "the stacked model still sits below the single-unit first layer on every endpoint"
below = []
for m, ref in [("b6_layer2_logit_exact_posterior", "b6_exact1_posterior"), ("b6_layer2_logit_cd_posterior", "b6_exact1_posterior"),
               ("b12_layer2_logit_exact_logit", "b12_exact1_logit"), ("b12_layer2_logit_cd_logit", "b12_exact1_logit")]:
    for ep in ("pb_all8", "prm_within", "prm_pooled", "prmscore_q08"):
        d = da.loc[m, ep] - da.loc[ref, ep]
        below.append((m, ep, float(d)))
        check_bool(f"depth amendment below H1: {m} {ep} (delta {d:+.6f})", d < 0, f"{m} minus {ref} on {ep} = {d:+.6f}")
CHECKS["depth_amendment_vs_H1_deltas"] = below

# ======================================================================================
# Figure 7: PB cells heatmap
# ======================================================================================
heat_rows = [("Token entropy (ref.)", "capacity", "entropy__old"), ("Varentropy15 (ref.)", "capacity", "var15__old"),
             ("RBM6 posterior (= exact H1 bank6)", "capacity", "rbm6__old"), ("RBM12 logit (= exact H1 bank12)", "capacity", "rbm12__logit_old"),
             ("Shared-variance bank12 posterior", "variance", "b12_variance_shared_posterior"),
             ("Separate-variance bank12 logit", "variance", "b12_variance_separate_logit"),
             ("Exact H4 bank6 posterior", "capacity", "b6_exact4_posterior"), ("Exact H4 bank12 logit", "capacity", "b12_exact4_logit"),
             ("Best-of-3 H4 bank6 posterior", "stability", "b6_best_exact4_posterior"), ("Best-of-3 H4 bank12 logit", "stability", "b12_best_exact4_logit"),
             ("Token-chain Markov bank12 logit", "temporal", "b12_chain_full_logit"), ("Shuffled-order control bank12 logit", "temporal", "b12_chain_shuffled_logit"),
             ("Depth orig. exact L2 bank6 posterior [cov .990]", "depth_amended", "b6_layer2_exact_posterior"),
             ("Depth orig. exact L2 bank12 logit [cov .974]", "depth_amended", "b12_layer2_exact_logit"),
             ("Depth amend. exact L2 (logit-in) bank6 posterior", "depth_amended", "b6_layer2_logit_exact_posterior"),
             ("Depth amend. exact L2 (logit-in) bank12 logit", "depth_amended", "b12_layer2_logit_exact_logit"),
             ("Depth amend. CD-10 L2 (logit-in) bank6 posterior", "depth_amended", "b6_layer2_logit_cd_posterior"),
             ("Depth amend. CD-10 L2 (logit-in) bank12 logit", "depth_amended", "b12_layer2_logit_cd_logit"),
             ("Longest-step control (ref.)", None, None)]
H = []
absent = []
for name, suite, m in heat_rows:
    if suite is None:
        H.append([np.nan] * 8 + [REF["length"]])
        absent.append(name)
        continue
    p = cells[suite]
    sub = p[p.method == m].set_index("cell")
    vals = [100 * sub.loc[c, "f1"] for c in CELLS]
    H.append(vals + [np.mean(vals)])
H = np.array(H)
fig, ax = plt.subplots(figsize=(13, 9))
Hm = np.ma.masked_invalid(H)
im = ax.imshow(Hm[:, :8], cmap="viridis", aspect="auto", vmin=5, vmax=50)
ax.set_xticks(np.arange(9))
ax.set_xticklabels([c.replace("pb_", "") for c in CELLS] + ["macro\n(mean of 8)"], rotation=30, ha="right", fontsize=8)
ax.set_yticks(np.arange(len(heat_rows)))
ax.set_yticklabels([r[0] for r in heat_rows], fontsize=8)
for i in range(H.shape[0]):
    for j in range(9):
        v = H[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a\n(cells absent)" if j < 8 else "", ha="center", va="center", fontsize=6.5, color="gray")
            continue
        if j == 8:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, weight="bold", color="black")
        else:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v < 32 else "black")
ax.set_xlim(-0.5, 8.5)
ax.add_patch(plt.Rectangle((7.5, -0.5), 1, len(heat_rows), fill=True, color="whitesmoke", zorder=0))
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("PB cell F1, % (full population)")
ax.set_title("ProcessBench per-cell F1 (%) for selected configurations (values from each suite's PB_CELLS.csv; macro column recomputed as the mean of the 8 cells)\n"
             "longest-step control: per-cell values are not in these suites' PB_CELLS files, only its macro 33.69 is listed", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_pb_cells_heatmap.png"), dpi=DPI)
plt.close(fig)
CHECKS["heatmap_rows_without_cells"] = absent

# ======================================================================================
# Cross-file checks
# ======================================================================================
# 1) macro PB = mean of 8 cell F1 vs METRICS pb_all8, for every method of every suite
n_macro_ok = 0
n_macro = 0
for s in SUITES:
    p = cells[s]
    for m, g in p.groupby("method"):
        n_macro += 1
        mean_f1 = g.set_index("cell").loc[CELLS, "f1"].mean()
        ref = metrics[s]["metrics"][m]["pb_all8"]
        if abs(mean_f1 - ref) <= 1e-9:
            n_macro_ok += 1
        else:
            CHECKS["mismatched"].append({"check": f"macro PB mean-of-8 {s}/{m}", "observed": float(mean_f1), "expected": float(ref)})
CHECKS["matched"].append({"check": "macro PB = mean of 8 cell F1 (all suites, all methods)", "n_ok": n_macro_ok, "n": n_macro})
# COMPARISON.csv pb_all8 vs METRICS
n_c = n_ok = 0
for s in SUITES:
    for _, r in comp[s].iterrows():
        n_c += 1
        if abs(r.pb_all8 - metrics[s]["metrics"][r.method]["pb_all8"]) <= 1e-12 and abs(r.prm_within - metrics[s]["metrics"][r.method]["prm_within"]) <= 1e-12:
            n_ok += 1
CHECKS["matched" if n_ok == n_c else "mismatched"].append({"check": "COMPARISON.csv pb_all8/prm_within == METRICS.json", "n_ok": n_ok, "n": n_c})
# 2) STAGE1_COMPARISON_TABLE.csv rows vs RBM_FUSION_COMPARISON.csv (value match)
n_t = n_tok = 0
unmatched = []
for _, r in stage_tbl.iterrows():
    n_t += 1
    hit = fus[(np.abs(fus.pb_macro_percent - r.pb_all8_pct) < 1e-6) & (np.abs(fus.prm_within - r.prm_within) < 1e-6)]
    if len(hit):
        n_tok += 1
    else:
        unmatched.append(r.label)
CHECKS["matched" if n_tok == n_t else "mismatched"].append({"check": "STAGE1_COMPARISON_TABLE rows found in RBM_FUSION_COMPARISON (pb, within)", "n_ok": n_tok, "n": n_t, "unmatched": unmatched})
# coverage column of the table
for _, r in stage_tbl.iterrows():
    check(f"STAGE1 table coverage {r.label}", r.coverage, r.valid_answers / N_ANSWERS, 1e-9)
# 3) reference rows
fr = fus.set_index("method")
check("token entropy PB", fr.loc["entropy__old", "pb_macro_percent"], 35.4444, 5e-5)
check("token entropy within", fr.loc["entropy__old", "prm_within"], 0.730111, 5e-7)
check("token entropy PRMScore", fr.loc["entropy__old", "prmscore_q08"], 0.625426, 5e-7)
check("varentropy15 PB", fr.loc["var15__old", "pb_macro_percent"], 35.9610, 5e-5)
check("varentropy15 within", fr.loc["var15__old", "prm_within"], 0.737786, 5e-7)
check("longest-step PB", fr.loc["length__old", "pb_macro_percent"], 33.6944, 5e-5)
check("random-step PB", fr.loc["random__old", "pb_macro_percent"], 20.2750, 5e-5)
check("fusion csv rows", len(fus), 130, 0)
check("fusion csv answer_local rows", int((fus.panel == "answer_local").sum()), 110, 0)
uniq = fus[fus.panel == "answer_local"].round({"pb_macro_percent": 6, "prm_within": 6}).drop_duplicates(["pb_macro_percent", "prm_within"])
CHECKS["notes"].append(f"answer_local rows: 110; distinct (PB, within) score sets: {len(uniq)}")
# 4) account section 5 numeric statements
check("account: DUFS minus low-corr PB pp", fr.loc["dufs6__rbm", "pb_macro_percent"] - fr.loc["correlation6__rbm", "pb_macro_percent"], -0.87, 5e-3)
check("account: position-conditioned minus shared PB pp", fr.loc["position__max", "pb_macro_percent"] - fr.loc["shared__max", "pb_macro_percent"], -0.69, 5e-3)
check("account: supervised minus unsupervised update PB pp", fr.loc["supervised_update", "pb_macro_percent"] - fr.loc["unsupervised_update", "pb_macro_percent"], 0.93, 5e-3)
check("account: 48-col trained minus initial PB pp", fr.loc["power48_rbm", "pb_macro_percent"] - fr.loc["power48_initial", "pb_macro_percent"], -16.94, 5e-3)
check("account: var15 IU minus raw var15 within", fr.loc["var15_iu__old", "prm_within"] - fr.loc["var15__old", "prm_within"], 0.009, 5e-4)
check("account: var15 IU minus raw var15 PB pp", fr.loc["var15_iu__old", "pb_macro_percent"] - fr.loc["var15__old", "pb_macro_percent"], -0.6, 5e-2)
check("account: separate-variance bank12 logit PB", fr.loc["b12_variance_separate_logit", "pb_macro_percent"], 21.09, 5e-3)
check("account: shared-variance bank12 posterior PB", fr.loc["b12_variance_shared_posterior", "pb_macro_percent"], 36.81, 5e-3)
check("account: low-corr-6 PB", fr.loc["correlation6__rbm", "pb_macro_percent"], 36.99, 5e-3)
check("account: var15 equal within (0.7470)", fr.loc["var15_equal__old", "prm_within"], 0.7470, 5e-5)
# readout confound: "changes PB by up to 1.5 pp and within-AUC by 0.01"
pairs = []
for m in fus.method:
    if m.endswith("_posterior"):
        base = m[:-len("_posterior")]
        other = base + "_logit"
    elif m.endswith("__old") and not m.endswith("__logit_old"):
        base = m[:-len("__old")]
        other = base + "__logit_old"
    else:
        continue
    if other in fr.index:
        pairs.append((m, other, float(fr.loc[m, "pb_macro_percent"] - fr.loc[other, "pb_macro_percent"]),
                      float(fr.loc[m, "prm_within"] - fr.loc[other, "prm_within"])))
pairs_df = pd.DataFrame(pairs, columns=["posterior_row", "logit_row", "pb_pp_posterior_minus_logit", "within_posterior_minus_logit"]).drop_duplicates()
CHECKS["readout_pairs"] = pairs_df.to_dict("records")
big = pairs_df[pairs_df.pb_pp_posterior_minus_logit.abs() > 1.5]
check_bool("account: posterior vs logit readout changes PB by at most 1.5 pp (all same-weight pairs in the record)", len(big) == 0,
           f"{len(big)} of {len(pairs_df)} same-weight pairs exceed 1.5 pp; max |delta| = {pairs_df.pb_pp_posterior_minus_logit.abs().max():.2f} pp")
lr = fus[fus.experiment == "rbm-logit-readout-v1"]
lr_pairs = pairs_df[pairs_df.posterior_row.isin(lr.method) & pairs_df.logit_row.isin(lr.method)]
check_bool("account: readout confound <= 1.5 pp within rbm-logit-readout-v1 rows only", (lr_pairs.pb_pp_posterior_minus_logit.abs() <= 1.5).all(),
           f"max |delta| within rbm-logit-readout-v1 = {lr_pairs.pb_pp_posterior_minus_logit.abs().max():.2f} pp ({lr_pairs.loc[lr_pairs.pb_pp_posterior_minus_logit.abs().idxmax(), 'posterior_row']})")
# variance decomposition claim
check("variance: quadratic reverses linear in lost cases", var_loss["summary"]["lost"]["quadratic_reverses_linear"], 922, 0)
check("variance: lost n", var_loss["summary"]["lost"]["n"], 934, 0)
# temporal claims
c = metrics["temporal"]["contrasts"]
check("temporal bank12 within delta", c["b12_chain_full_logit_minus_b12_chain_shuffled_logit"]["prm_within_delta_common"], -0.0028, 5e-5)
check("temporal bank6 within delta", c["b6_chain_full_posterior_minus_b6_chain_shuffled_posterior"]["prm_within_delta_common"], -0.0036, 5e-5)
# H1 references identical across suites (account: "all 13 reference rows reproduce")
n_ref = n_ref_ok = 0
ALIAS = {"shrinkage__old": "rbm_shrinkage", "diagonal__old": "rbm_diagonal"}
for s in SUITES:
    cs = comp[s].set_index("method")
    for m in cs.index:
        if m.endswith("__old"):
            n_ref += 1
            fm = ALIAS.get(m, m)
            if abs(cs.loc[m, "pb_all8"] * 100 - fr.loc[fm, "pb_macro_percent"]) < 1e-9 and abs(cs.loc[m, "prm_within"] - fr.loc[fm, "prm_within"]) < 1e-9:
                n_ref_ok += 1
CHECKS["matched" if n_ref == n_ref_ok else "mismatched"].append({"check": "__old reference rows in suite COMPARISON.csv == RBM_FUSION_COMPARISON source values", "n_ok": n_ref_ok, "n": n_ref})
# exact H1 rows in suites equal RBM6/RBM12 rows (the H1 'capacity start' is the historical fit)
for a, b in [("b6_exact1_posterior", "rbm6__old"), ("b12_exact1_logit", "rbm12__logit_old"), ("b6_exact1_logit", "rbm6__logit_old"), ("b12_exact1_posterior", "rbm12__old")]:
    check(f"capacity {a} PB == {b}", comp["capacity"].set_index("method").loc[a, "pb_all8"], comp["capacity"].set_index("method").loc[b, "pb_all8"], 1e-12)
# depth account table values
check("account depth table: orig exact L2 bank6 within", da.loc["b6_layer2_exact_posterior", "prm_within"], 0.6215, 5e-5)
check("account depth table: orig exact L2 bank6 n", da.loc["b6_layer2_exact_posterior", "prm_within_n"], 6022, 0)
check("account depth table: orig exact L2 bank12 n", da.loc["b12_layer2_exact_logit", "prm_within_n"], 5914, 0)
check("account depth table: amend CD L2 bank6 within", da.loc["b6_layer2_logit_cd_posterior", "prm_within"], 0.7374, 5e-5)
check("account depth table: amend CD L2 bank12 (logit) PB", 100 * da.loc["b12_layer2_logit_cd_logit", "pb_all8"], 22.97, 5e-3)
CHECKS["notes"].append(f"depth bank12 CD-10 logit-input: logit readout {100*da.loc['b12_layer2_logit_cd_logit','pb_all8']:.2f} vs posterior readout {100*da.loc['b12_layer2_logit_cd_posterior','pb_all8']:.2f} PB; within {da.loc['b12_layer2_logit_cd_logit','prm_within']:.4f} vs {da.loc['b12_layer2_logit_cd_posterior','prm_within']:.4f}")
CHECKS["notes"].append(f"depth bank12 exact logit-input: logit readout {100*da.loc['b12_layer2_logit_exact_logit','pb_all8']:.2f} vs posterior readout {100*da.loc['b12_layer2_logit_exact_posterior','pb_all8']:.2f} PB")
CHECKS["summary"] = {"n_matched": len(CHECKS["matched"]), "n_mismatched": len(CHECKS["mismatched"])}
with open(os.path.join(OUT, "CHECKS.json"), "w", encoding="utf-8") as fh:
    json.dump(CHECKS, fh, indent=1, default=float)
print("matched", len(CHECKS["matched"]), "mismatched", len(CHECKS["mismatched"]))
for r in CHECKS["mismatched"]:
    print("MISMATCH", json.dumps(r, default=float))
for n in CHECKS["notes"]:
    print("NOTE", n)
print("readout pairs:\n", pairs_df.to_string())
print("depth vs H1 deltas:", json.dumps(below, indent=0))
