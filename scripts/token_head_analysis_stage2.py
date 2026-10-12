"""Stage 2 of token-head feature analysis v1: diagnostics + figures.

Reads the per-token arrays persisted by ``token_head_feature_analysis_v1.py
compute`` and produces:

  * correlation / eigen-structure / conditional-dependence diagnostics of the
    28 oriented token-head coordinates;
  * structural-fit comparison against the IU-PCR model (diagonal + rank-2),
    the SU-PCR sparse-error model (rank-2 + sparse), and block/family models;
  * descriptive weight-vector and per-token AUROC / localization comparisons
    (labels opened, diagnostic only — no method is selected or promoted);
  * all report figures (PNG) and ``stats.json``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats as sstats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.unified_causal_iu import IU_FIT, _robust_location_scale
from spectral_utils.upcr import upcr_fit
from spectral_utils.dependency_fusion import sparse_upcr_fit
from spectral_utils.multitask_trajectory import equal_positions
from spectral_utils.fair_comparisons.twentyfour import (
    U28_BASE_STREAMS,
    U28_TRANSFORMS,
    load_unified28_model,
)

OUT_DIR = os.path.join(REPO, "results", "token_head_feature_analysis_v1")
FIG_DIR = os.path.join(OUT_DIR, "figures")

STREAMS = [name.removeprefix("raw::") for name in U28_BASE_STREAMS]
TRANSFORMS = list(U28_TRANSFORMS)
LEVEL_COLS = [index * 4 for index in range(7)]

# --- dataviz reference palette (light mode) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CRITICAL = "#d03b3b"
RED = "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_MID = "#f0efec"

DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL_MID, RED])
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
})

RNG = np.random.default_rng(20260828)


def load_ds(name):
    data = np.load(os.path.join(OUT_DIR, f"tokens_{name}.npz"))
    with open(os.path.join(OUT_DIR, f"meta_{name}.json"), encoding="utf-8") as handle:
        metas = json.load(handle)
    return dict(data), metas


def trace_arrays(data, index):
    begin, end = int(data["offsets"][index]), int(data["offsets"][index + 1])
    view = slice(begin, end)
    return data["z"][view], data["evidence"][view], data["mask"][view], data["raw"][view]


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = sstats.rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def offdiag(matrix):
    output = matrix.copy()
    np.fill_diagonal(output, 0.0)
    return output


def sample_tokens(data, metas, per_trace=32):
    rows, ys, trace_ids, labels, wrongs = [], [], [], [], []
    for index, meta in enumerate(metas):
        z, _, mask, _ = trace_arrays(data, index)
        positions = equal_positions(len(z), per_trace)
        rows.append(z[positions])
        ys.append(mask[positions])
        trace_ids.append(np.full(len(positions), index))
        labels.append(np.full(len(positions), meta["label"]))
        wrongs.append(np.full(len(positions), meta["final_wrong"]))
    return (np.vstack(rows).astype(float), np.concatenate(ys).astype(int),
            np.concatenate(trace_ids), np.concatenate(labels), np.concatenate(wrongs))


def robust_standardize(matrix):
    centres, scales = [], []
    for column in matrix.T:
        centre, scale = _robust_location_scale(column)
        centres.append(centre)
        scales.append(scale)
    centres = np.asarray(centres)
    scales = np.asarray(scales)
    return (matrix - centres) / scales, centres, scales


def feature_labels():
    return [f"{stream}\n{transform}" for stream in STREAMS for transform in TRANSFORMS]


# ---------------------------------------------------------------- figures ---

def fig_weights(model):
    weights = np.asarray(model.weights).reshape(7, 4)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    image = ax.imshow(weights, cmap=SEQ_BLUE, vmin=0.0, vmax=weights.max())
    ax.set_xticks(range(4), TRANSFORMS)
    ax.set_yticks(range(7), STREAMS)
    ax.grid(False)
    for row in range(7):
        for col in range(4):
            value = weights[row, col]
            color = "#ffffff" if value > 0.55 * weights.max() else INK
            ax.text(col, row, f"{value:.4f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("Frozen Unified-28 IU-PCR weights (7 streams x 4 transforms)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="weight")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_weights.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _example_panel(ax_heat, ax_curve, z, evidence, meta, title):
    level = z[:, LEVEL_COLS].T  # 7 x T
    image = ax_heat.imshow(
        np.clip(level, -3, 3), cmap=DIVERGING, vmin=-3, vmax=3,
        aspect="auto", interpolation="nearest",
    )
    ax_heat.set_yticks(range(7), STREAMS, fontsize=7)
    ax_heat.set_xticks([])
    ax_heat.grid(False)
    ax_heat.set_title(title, fontsize=9)

    tokens = np.arange(len(evidence))
    ax_curve.plot(tokens, evidence, color=BLUE, linewidth=1.4,
                  label="IU-PCR evidence")
    ax_curve.axhline(0.0, color=BASELINE, linewidth=0.8)
    for span in meta.get("step_spans", []):
        ax_curve.axvline(span[0], color=GRID, linewidth=0.6)
    error_span = meta.get("error_span")
    if error_span:
        for axis in (ax_heat, ax_curve):
            axis.axvspan(error_span[0], error_span[1] - 1, color=CRITICAL,
                         alpha=0.14, linewidth=0)
        ax_curve.axvspan(error_span[0], error_span[1] - 1, color=CRITICAL,
                         alpha=0.0, label="first-error step")
    peak = int(np.argmax(evidence))
    ax_curve.plot([peak], [evidence[peak]], marker="o", markersize=5,
                  color=ORANGE, linestyle="none", label="argmax (localization)")
    ax_curve.set_xlim(0, len(evidence) - 1)
    ax_curve.set_xlabel("token index", fontsize=8)
    return image


def fig_examples(data, metas, picks, filename, suptitle):
    fig, axes = plt.subplots(
        2, len(picks), figsize=(4.6 * len(picks), 5.2),
        gridspec_kw={"height_ratios": [1.15, 1.0]}, squeeze=False,
    )
    image = None
    for col, index in enumerate(picks):
        z, evidence, _, _ = trace_arrays(data, index)
        meta = metas[index]
        label = meta.get("label", -1)
        status = "clean" if label < 0 else f"first error @ step {label}"
        title = f"{meta['id']}  ({status}, T={meta['n_tokens']})"
        image = _example_panel(axes[0][col], axes[1][col], z, evidence, meta, title)
    handles, labels_ = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.colorbar(image, ax=[axes[0][col] for col in range(len(picks))],
                 shrink=0.85, pad=0.01, label="oriented z (level)")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def aligned_matrix(curves, onsets, window=60):
    width = 2 * window + 1
    output = np.full((len(curves), width), np.nan)
    for index, (curve, onset) in enumerate(zip(curves, onsets)):
        for offset in range(-window, window + 1):
            position = onset + offset
            if 0 <= position < len(curve):
                output[index, offset + window] = curve[position]
    return output


def boot_mean_ci(matrix, n_boot=500):
    mean = np.nanmean(matrix, axis=0)
    draws = np.empty((n_boot, matrix.shape[1]))
    for b in range(n_boot):
        picks = RNG.integers(0, len(matrix), len(matrix))
        draws[b] = np.nanmean(matrix[picks], axis=0)
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return mean, low, high


def fig_event_aligned(datasets, window=60, filename="fig_event_aligned.png"):
    error_curves = {name: [] for name in STREAMS + ["evidence"]}
    error_onsets = []
    control_curves = {name: [] for name in STREAMS + ["evidence"]}
    control_onsets = []
    relative_onsets = []
    for data, metas in datasets:
        for index, meta in enumerate(metas):
            if meta["label"] >= 0 and meta.get("error_span"):
                relative_onsets.append(meta["error_span"][0] / max(meta["n_tokens"], 1))
    for data, metas in datasets:
        for index, meta in enumerate(metas):
            z, evidence, _, _ = trace_arrays(data, index)
            if meta["label"] >= 0 and meta.get("error_span"):
                onset = meta["error_span"][0]
                for s_index, stream in enumerate(STREAMS):
                    error_curves[stream].append(z[:, LEVEL_COLS[s_index]])
                error_curves["evidence"].append(evidence)
                error_onsets.append(onset)
            elif meta["label"] < 0:
                fraction = relative_onsets[RNG.integers(0, len(relative_onsets))]
                onset = int(round(fraction * meta["n_tokens"]))
                for s_index, stream in enumerate(STREAMS):
                    control_curves[stream].append(z[:, LEVEL_COLS[s_index]])
                control_curves["evidence"].append(evidence)
                control_onsets.append(onset)

    names = STREAMS + ["evidence"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), sharex=True)
    offsets = np.arange(-window, window + 1)
    for panel, name in enumerate(names):
        ax = axes[panel // 4][panel % 4]
        err = aligned_matrix(error_curves[name], error_onsets, window)
        ctl = aligned_matrix(control_curves[name], control_onsets, window)
        for matrix, color, label in ((ctl, MUTED, "clean traces (matched position)"),
                                     (err, BLUE, "error traces (aligned at onset)")):
            mean, low, high = boot_mean_ci(matrix)
            ax.fill_between(offsets, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(offsets, mean, color=color, linewidth=1.6, label=label)
        ax.axvline(0, color=CRITICAL, linewidth=0.9, alpha=0.7)
        ax.set_title(name, fontsize=9,
                     fontweight="bold" if name == "evidence" else "normal")
        if panel // 4 == 1:
            ax.set_xlabel("token offset from first-error onset")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Event-aligned means around the first-error onset "
                 "(oriented level z per stream; frozen IU-PCR evidence; 95% trace bootstrap)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_error_traces": len(error_onsets),
        "n_control_traces": len(control_onsets),
    }


def fig_positional(datasets, filename="fig_positional.png", n_bins=10):
    per_wrong = {0: {name: [] for name in STREAMS + ["evidence"]},
                 1: {name: [] for name in STREAMS + ["evidence"]}}
    for data, metas in datasets:
        for index, meta in enumerate(metas):
            z, evidence, _, _ = trace_arrays(data, index)
            n_tokens = len(evidence)
            bins = np.minimum((np.arange(n_tokens) * n_bins) // max(n_tokens, 1), n_bins - 1)
            wrong = int(meta["final_wrong"])
            for s_index, stream in enumerate(STREAMS):
                series = z[:, LEVEL_COLS[s_index]]
                means = [np.nanmean(series[bins == b]) for b in range(n_bins)]
                per_wrong[wrong][stream].append(means)
            per_wrong[wrong]["evidence"].append(
                [np.nanmean(evidence[bins == b]) for b in range(n_bins)])

    names = STREAMS + ["evidence"]
    centers = (np.arange(n_bins) + 0.5) / n_bins
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.4), sharex=True)
    for panel, name in enumerate(names):
        ax = axes[panel // 4][panel % 4]
        for wrong, color, label in ((0, MUTED, "final answer correct"),
                                    (1, BLUE, "final answer wrong")):
            matrix = np.asarray(per_wrong[wrong][name])
            mean, low, high = boot_mean_ci(matrix)
            ax.fill_between(centers, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(centers, mean, color=color, linewidth=1.6, label=label)
        ax.set_title(name, fontsize=9,
                     fontweight="bold" if name == "evidence" else "normal")
        if panel // 4 == 1:
            ax.set_xlabel("relative position in trace")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Positional profile of the seven stream levels and the frozen evidence "
                 "(decile means, equal trace weight, 95% trace bootstrap)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_correlation(spearman, filename="fig_corr.png"):
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    image = ax.imshow(spearman, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.grid(False)
    for boundary in range(4, 28, 4):
        ax.axhline(boundary - 0.5, color=INK, linewidth=0.7)
        ax.axvline(boundary - 0.5, color=INK, linewidth=0.7)
    ax.set_xticks(range(1, 28, 4),
                  [stream for stream in STREAMS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(1, 28, 4), [stream for stream in STREAMS], fontsize=8)
    ax.set_title("Spearman correlation of the 28 oriented token-head coordinates\n"
                 "(blocks of 4 = level, ewma16, positive_area, persistence per stream)")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_eigen(pearson, cond_stats, filename="fig_eigen.png"):
    eigenvalues = np.linalg.eigvalsh(pearson)[::-1]
    base = np.linalg.norm(offdiag(pearson))
    vals, vecs = np.linalg.eigh(pearson)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    residual_fraction = []
    for k in range(0, 6):
        approx = (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T if k else np.zeros_like(pearson)
        residual_fraction.append(float(np.linalg.norm(offdiag(pearson - approx)) / base))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    ax = axes[0]
    ax.bar(np.arange(1, 11), eigenvalues[:10], color=BLUE, width=0.62)
    ax.axvline(2.5, color=CRITICAL, linewidth=1.0, alpha=0.8)
    ax.text(2.65, eigenvalues[0] * 0.9, "IU-PCR uses 2 components",
            fontsize=8, color=CRITICAL)
    ax.set_title("Eigenvalue spectrum (top 10 of 28)")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")

    ax = axes[1]
    ax.plot(range(0, 6), residual_fraction, color=BLUE, linewidth=1.8, marker="o",
            markersize=5)
    ax.set_xticks(range(0, 6))
    ax.set_ylim(0, 1.02)
    ax.set_title("Off-diagonal residual after rank-k removal")
    ax.set_xlabel("rank k")
    ax.set_ylabel("residual fraction (Frobenius)")

    ax = axes[2]
    keys = ["marginal", "conditional y=0", "conditional y=1"]
    values = [cond_stats["marginal_mean_abs"], cond_stats["cond0_mean_abs"],
              cond_stats["cond1_mean_abs"]]
    bars = ax.bar(keys, values, color=[BLUE, MUTED, CRITICAL], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", fontsize=8)
    ax.set_title("Mean |off-diagonal corr| (error traces)\nmodel-check: conditional should be ~0")
    ax.set_ylabel("mean |rho|")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return eigenvalues, residual_fraction


def fig_weight_compare(weight_table, filename="fig_weight_compare.png"):
    names = list(weight_table)
    matrix = np.vstack([weight_table[name] / np.abs(weight_table[name]).sum()
                        for name in names])
    fig, ax = plt.subplots(figsize=(12.6, 3.4))
    limit = np.abs(matrix).max()
    image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xticks(range(1, 28, 4), STREAMS, fontsize=8)
    for boundary in range(4, 28, 4):
        ax.axvline(boundary - 0.5, color=INK, linewidth=0.6)
    ax.grid(False)
    ax.set_title("L1-normalized fusion weight vectors across methods "
                 "(columns: 7 streams x {level, ewma16, positive_area, persistence})")
    fig.colorbar(image, ax=ax, shrink=0.85, label="normalized weight")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_method_metrics(metric_table, filename="fig_token_auroc.png"):
    names = list(metric_table)
    pooled = [metric_table[name]["pooled_token_auroc"] for name in names]
    exact = [metric_table[name]["localization_exact"] for name in names]
    within = [metric_table[name]["localization_within1"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.9))
    positions = np.arange(len(names))
    ax = axes[0]
    bars = ax.barh(positions, pooled, color=BLUE, height=0.6)
    ax.set_yticks(positions, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.5, max(pooled) + 0.04)
    ax.axvline(0.5, color=BASELINE, linewidth=0.9)
    for bar, value in zip(bars, pooled):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=8)
    ax.set_title("Per-token AUROC, first-error tokens vs rest\n(error traces, sampled positions; descriptive)")

    ax = axes[1]
    width = 0.38
    ax.bar(positions - width / 2, exact, width=width, color=BLUE, label="exact step hit")
    ax.bar(positions + width / 2, within, width=width, color=AQUA, label="within-1 step")
    ax.set_ylim(0, 0.74)
    ax.set_xticks(positions, names, fontsize=7, rotation=20, ha="right")
    for x, value in zip(positions - width / 2, exact):
        ax.text(x, value + 0.008, f"{value:.3f}", ha="center", fontsize=7)
    for x, value in zip(positions + width / 2, within):
        ax.text(x, value + 0.008, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_title("Localization by argmax of the fused curve\n(error traces; descriptive)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_ragtruth(data, metas, filename="fig_ragtruth.png"):
    # pick two 'full' examples with spans and moderate length
    picks = []
    for index, meta in enumerate(metas):
        if meta["condition"] == "full" and meta["halluc_spans"] and 60 <= meta["n_tokens"] <= 320:
            picks.append(index)
        if len(picks) == 2:
            break
    if not picks:
        return {}

    fig = plt.figure(figsize=(12.8, 6.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.35, wspace=0.18)
    for col, index in enumerate(picks):
        z, evidence, mask, _ = trace_arrays(data, index)
        meta = metas[index]
        ax_heat = fig.add_subplot(grid[0, col])
        ax_curve = fig.add_subplot(grid[1, col])
        meta_like = {
            "step_spans": [],
            "error_span": None,
        }
        image = _example_panel(ax_heat, ax_curve, z, evidence, meta_like,
                               f"{meta['id']}  ({meta['task_type']}, "
                               f"{', '.join(meta['label_types'])}, T={meta['n_tokens']})")
        for span in meta["halluc_spans"]:
            for axis in (ax_heat, ax_curve):
                axis.axvspan(span[0], span[1] - 1, color=CRITICAL, alpha=0.16, linewidth=0)
    fig.suptitle("RAGTruth-EC pilot (condition=full): oriented stream levels and frozen "
                 "IU-PCR evidence, annotated hallucination spans shaded", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # descriptive pooled token AUROC inside 'full' condition + evidence contrast
    by_key = {meta["id"]: index for index, meta in enumerate(metas)}
    pooled_scores, pooled_contrast, pooled_labels = [], [], []
    for index, meta in enumerate(metas):
        if meta["condition"] != "full" or not meta["halluc_spans"]:
            continue
        z, evidence, mask, _ = trace_arrays(data, index)
        noctx_key = meta["id"].replace("::full", "::noctx")
        contrast = None
        if noctx_key in by_key:
            _, evidence_noctx, _, _ = trace_arrays(data, by_key[noctx_key])
            if len(evidence_noctx) == len(evidence):
                contrast = evidence - evidence_noctx
        pooled_scores.append(evidence)
        pooled_labels.append(mask)
        pooled_contrast.append(contrast if contrast is not None else np.full(len(evidence), np.nan))
    if not pooled_scores:
        return {}
    scores = np.concatenate(pooled_scores)
    labels = np.concatenate(pooled_labels)
    contrast = np.concatenate(pooled_contrast)
    valid = np.isfinite(contrast)
    return {
        "n_full_responses_with_spans": int(len(pooled_scores)),
        "n_tokens": int(len(labels)),
        "n_span_tokens": int(labels.sum()),
        "token_auroc_evidence_full": rank_auc(scores, labels),
        "token_auroc_contrast_full_minus_noctx": rank_auc(contrast[valid], labels[valid]),
    }


# ---------------------------------------------------------------- analysis ---

def run_analysis():
    os.makedirs(FIG_DIR, exist_ok=True)
    model, sha = load_unified28_model(REPO)
    stats = {"model_artifact_sha256": sha,
             "model_diagnostics": dict(model.diagnostics)}

    gsm = load_ds("gsm8k")
    math_ds = load_ds("math")
    datasets = [gsm, math_ds]

    fig_weights(model)

    # ---- example traces (gsm8k) ----
    metas = gsm[1]
    def first_match(predicate):
        for index, meta in enumerate(metas):
            if predicate(meta):
                return index
        return None
    picks = [
        first_match(lambda m: m["label"] < 0 and 150 <= m["n_tokens"] <= 400),
        first_match(lambda m: 0 <= m["label"] <= 1 and 120 <= m["n_tokens"] <= 450),
        first_match(lambda m: m["label"] >= 3 and 150 <= m["n_tokens"] <= 500),
    ]
    picks = [pick for pick in picks if pick is not None]
    fig_examples(gsm[0], metas, picks, "fig_example_traces.png",
                 "ProcessBench GSM8K (Llama-3.1-8B): stream levels and the frozen token-head evidence")

    # ---- event aligned + positional ----
    stats["event_aligned"] = fig_event_aligned(datasets)
    fig_positional(datasets)

    # ---- token sample for correlation work ----
    X_parts, y_parts, label_parts = [], [], []
    for data, ds_metas in datasets:
        X, y, _, labels, _ = sample_tokens(data, ds_metas)
        X_parts.append(X)
        y_parts.append(y)
        label_parts.append(labels)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    trace_label = np.concatenate(label_parts)
    stats["sample"] = {"n_tokens": int(len(X)), "n_error_span_tokens": int(y.sum()),
                       "n_traces": sum(len(m) for _, m in datasets)}

    spearman = sstats.spearmanr(X).statistic
    Xs, _, _ = robust_standardize(X)
    pearson = np.corrcoef(Xs.T)
    fig_correlation(spearman)

    # conditional structure inside error traces
    in_error_traces = trace_label >= 0
    Xe = Xs[in_error_traces]
    ye = y[in_error_traces]
    corr_marginal = np.corrcoef(Xe.T)
    corr_y0 = np.corrcoef(Xe[ye == 0].T)
    corr_y1 = np.corrcoef(Xe[ye == 1].T)
    cond_stats = {
        "marginal_mean_abs": float(np.abs(offdiag(corr_marginal)).mean()),
        "cond0_mean_abs": float(np.abs(offdiag(corr_y0)).mean()),
        "cond1_mean_abs": float(np.abs(offdiag(corr_y1)).mean()),
        "n_error_trace_tokens": int(len(ye)),
        "n_y1": int(ye.sum()),
    }
    stats["conditional"] = cond_stats

    eigenvalues, residual_fraction = fig_eigen(pearson, cond_stats)
    stats["eigen"] = {"eigenvalues": eigenvalues.tolist(),
                      "offdiag_residual_fraction_rank_k": residual_fraction}

    # block/family structure
    same_stream, same_transform, neither = [], [], []
    for a in range(28):
        for b in range(a + 1, 28):
            value = abs(spearman[a, b])
            if a // 4 == b // 4:
                same_stream.append(value)
            elif a % 4 == b % 4:
                same_transform.append(value)
            else:
                neither.append(value)
    stats["blocks"] = {
        "mean_abs_rho_same_stream": float(np.mean(same_stream)),
        "mean_abs_rho_same_transform": float(np.mean(same_transform)),
        "mean_abs_rho_neither": float(np.mean(neither)),
    }

    # ---- structural fits ----
    iu = upcr_fit(Xs.T, **IU_FIT)
    su = sparse_upcr_fit(Xs.T)
    vals, vecs = np.linalg.eigh(pearson)
    pc1 = vecs[:, -1]
    if pc1.mean() < 0:
        pc1 = -pc1

    stats["iu_pcr_refit"] = {
        "rho_hat_range": [float(np.min(iu.rho_hat)), float(np.max(iu.rho_hat))],
        "g2_hat": float(iu.g2_hat),
        "lambda2_frac": float(iu.lambda2_frac),
        "proj_residual": float(iu.proj_residual),
        "n_components_used": int(iu.n_components_used),
    }
    decomp = su.decomposition
    stats["su_pcr"] = {
        "sparse_fraction": float(decomp.sparse_fraction),
        "relative_residual": float(decomp.relative_residual),
        "converged": bool(decomp.converged),
        "theorem_support_ok": bool(decomp.theorem_support_ok),
        "projection_residual": float(su.projection_residual),
        "pcr_eigenvalues": np.asarray(su.pcr_eigenvalues, dtype=float).tolist(),
    }
    support = np.asarray(decomp.support, dtype=bool)
    pair_names = []
    for a in range(28):
        for b in range(a + 1, 28):
            if support[a, b]:
                pair_names.append(
                    f"{STREAMS[a // 4]}::{TRANSFORMS[a % 4]} ~ {STREAMS[b // 4]}::{TRANSFORMS[b % 4]}")
    stats["su_pcr"]["sparse_support_pairs"] = pair_names[:40]

    weight_table = {
        "IU-PCR (frozen Unified-28)": np.asarray(model.weights, dtype=float),
        "IU-PCR (refit, this token sample)": np.asarray(iu.w, dtype=float),
        "SU-PCR (sparse-error reproduction)": np.asarray(su.w_pcr, dtype=float),
        "SDSF (dependency-weighted ridge)": np.asarray(su.w_structured, dtype=float),
        "Simple average": np.ones(28) / 28.0,
        "First principal component": pc1,
    }
    fig_weight_compare(weight_table)

    names = list(weight_table)
    unit = {name: weight_table[name] / np.linalg.norm(weight_table[name]) for name in names}
    cosine = {a: {b: float(unit[a] @ unit[b]) for b in names} for a in names}
    stats["weight_cosine"] = cosine

    # ---- descriptive per-token AUROC + localization ----
    metric_table = {}
    sampled_scores_frozen = None
    for name, weights in weight_table.items():
        scores = Xs @ weights
        pooled = rank_auc(scores[in_error_traces], ye)
        exact_hits, within_hits, n_error = 0, 0, 0
        for data, ds_metas in datasets:
            for index, meta in enumerate(ds_metas):
                if meta["label"] < 0 or not meta.get("step_spans"):
                    continue
                z, evidence, _, _ = trace_arrays(data, index)
                curve = z @ weights if name != "IU-PCR (frozen Unified-28)" else evidence
                peak = int(np.argmax(curve))
                step = 0
                for s_index, span in enumerate(meta["step_spans"]):
                    if span[0] <= peak < span[1]:
                        step = s_index
                        break
                else:
                    step = len(meta["step_spans"]) - 1
                n_error += 1
                exact_hits += int(step == meta["label"])
                within_hits += int(abs(step - meta["label"]) <= 1)
        metric_table[name] = {
            "pooled_token_auroc": float(pooled),
            "localization_exact": exact_hits / max(n_error, 1),
            "localization_within1": within_hits / max(n_error, 1),
            "n_error_traces": n_error,
        }
    stats["method_metrics"] = metric_table
    fig_method_metrics(metric_table)

    # ---- per-trace AUROC of the frozen evidence ----
    per_trace = []
    for data, ds_metas in datasets:
        for index, meta in enumerate(ds_metas):
            if meta["label"] < 0:
                continue
            _, evidence, mask, _ = trace_arrays(data, index)
            value = rank_auc(evidence, mask)
            if np.isfinite(value):
                per_trace.append(value)
    stats["frozen_evidence_per_trace_auroc"] = {
        "n": len(per_trace),
        "mean": float(np.mean(per_trace)),
        "median": float(np.median(per_trace)),
        "q25": float(np.quantile(per_trace, 0.25)),
        "q75": float(np.quantile(per_trace, 0.75)),
    }

    # ---- RAGTruth pilot ----
    try:
        rt = load_ds("ragtruth_pilot")
        stats["ragtruth_pilot"] = fig_ragtruth(*rt)
    except FileNotFoundError:
        stats["ragtruth_pilot"] = {"missing": True}

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps({key: stats[key] for key in
                      ("sample", "conditional", "blocks", "iu_pcr_refit", "su_pcr")},
                     indent=2)[:2000])
    print("[done] figures in", FIG_DIR)


if __name__ == "__main__":
    run_analysis()
