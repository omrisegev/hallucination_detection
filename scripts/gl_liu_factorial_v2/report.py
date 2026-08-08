#!/usr/bin/env python3
"""Build plots and plain-English reports for the GL-LIU factorial run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GLOBAL_LABELS = {"global_iu": "Global IU-PCR", "global_dufs": "Global DUFS-LIU"}
LOCAL_LABELS = {
    "local_temporal_core": "Temporal LIU\ncore 5",
    "local_dufs_core": "DUFS-LIU\ncore 5",
    "local_dufs_broad": "DUFS-LIU\nbroad 28",
}


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def _float(row, key):
    return float(row[key])


def _macro_lookup(rows, group="all_8_cells"):
    return {
        (row["global"], row["local"]): row
        for row in rows if row["group"] == group
    }


def _system_lookup(rows):
    return {
        (row["model"], row["subset"], row["system"]): row
        for row in rows
    }


def _component_means(rows, split=None):
    selected = [row for row in rows if split is None or row["split"] == split]
    output = {}
    for candidate in sorted({row["candidate"] for row in selected}):
        candidates = [row for row in selected if row["candidate"] == candidate]
        for metric in ("auroc", "exact", "tol1"):
            values = [float(row[metric]) for row in candidates if row.get(metric, "")]
            if values:
                output[(candidate, metric)] = float(np.mean(values))
    return output


def plot_factorials(macros, out_dir):
    lookup = _macro_lookup(macros)
    matrices = [
        ("A. Graph construction, fixed five-view locator",
         ["local_temporal_core", "local_dufs_core"]),
        ("B. Feature pool, fixed DUFS-LIU locator",
         ["local_dufs_core", "local_dufs_broad"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for axis, (title, local_names) in zip(axes, matrices):
        values = np.asarray([
            [100.0 * _float(lookup[(global_name, local_name)], "f1")
             for local_name in local_names]
            for global_name in ("global_iu", "global_dufs")
        ])
        image = axis.imshow(values, cmap="Blues", vmin=27.5, vmax=32.5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(j, i, f"{values[i, j]:.2f}%", ha="center", va="center",
                          color="white" if values[i, j] > 30.5 else "black",
                          fontsize=12, fontweight="bold")
        axis.set_xticks(range(2), [LOCAL_LABELS[name] for name in local_names])
        axis.set_yticks(range(2), [GLOBAL_LABELS[name] for name in ("global_iu", "global_dufs")])
        axis.set_xlabel("Local head")
        axis.set_ylabel("Global head")
        axis.set_title(title)
    fig.colorbar(image, ax=axes, label="ProcessBench F1 (%)", shrink=0.8)
    fig.savefig(out_dir / "factorial_matrices.png", dpi=180)
    plt.close(fig)


def plot_cell_deltas(systems, out_dir):
    lookup = _system_lookup(systems)
    cells = [(model, subset) for model in ("qwen3_4b", "qwen3_8b")
             for subset in ("gsm8k", "math", "olympiadbench", "omnimath")]
    unified, broad = [], []
    for model, subset in cells:
        base = _float(lookup[(model, subset, "global_dufs__local_temporal_core")], "f1")
        core = _float(lookup[(model, subset, "global_dufs__local_dufs_core")], "f1")
        wide = _float(lookup[(model, subset, "global_dufs__local_dufs_broad")], "f1")
        unified.append(100.0 * (core - base))
        broad.append(100.0 * (wide - core))
    labels = [f"{model.replace('qwen3_', '').upper()}\n{subset}" for model, subset in cells]
    x = np.arange(len(cells))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12.2, 4.8), constrained_layout=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.bar(x - width / 2, unified, width, label="Unified DUFS local − temporal local")
    axis.bar(x + width / 2, broad, width, label="Broad 28 − core 5")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Change in ProcessBench F1 (percentage points)")
    axis.set_title("The unified graph is a small mixed gain; the broad pool usually hurts")
    axis.legend(frameon=False, ncol=2)
    fig.savefig(out_dir / "per_cell_deltas.png", dpi=180)
    plt.close(fig)


def plot_progression(macros, out_dir):
    lookup = _macro_lookup(macros)
    mindgap = next(row for row in macros
                   if row["group"] == "all_8_cells" and row["system"] == "mindgap_control")
    items = [
        ("Mind the Gap\ncontrol", 100.0 * _float(mindgap, "f1")),
        ("GL-LIU v1\nglobal DUFS + temporal", 100.0 * _float(
            lookup[("global_dufs", "local_temporal_core")], "f1")),
        ("Unified DUFS-LIU\ncore 5", 100.0 * _float(
            lookup[("global_dufs", "local_dufs_core")], "f1")),
        ("Unified DUFS-LIU\nbroad 28", 100.0 * _float(
            lookup[("global_dufs", "local_dufs_broad")], "f1")),
    ]
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    bars = axis.bar(np.arange(len(items)), [value for _, value in items])
    axis.set_xticks(np.arange(len(items)), [name for name, _ in items])
    axis.set_ylabel("ProcessBench F1 (%)")
    axis.set_ylim(0, 35)
    axis.set_title("Method progression under one shared evaluation protocol")
    for bar, (_, value) in zip(bars, items):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}%",
                  ha="center", va="bottom", fontweight="bold")
    fig.savefig(out_dir / "method_progression.png", dpi=180)
    plt.close(fig)


def plot_gates(diagnostics_dir, out_dir):
    per_cell = []
    effective = []
    ranks = []
    for path in sorted(diagnostics_dir.glob("*.json")):
        local = json.loads(path.read_text())["local"]
        ranked = local["local_dufs_broad"]["dufs_gate"]["ranked_features"]
        per_cell.append({item["feature"]: float(item["probability"]) for item in ranked})
        effective.append(float(local["local_dufs_broad"]["dufs_gate"]["effective_feature_count"]))
        ranks.append(float(local["rank_displacement"]["dufs_core_vs_dufs_broad"]))
    names = sorted(per_cell[0], key=lambda name: -np.median([row[name] for row in per_cell]))
    values = np.asarray([np.median([row[name] for row in per_cell]) for name in names])
    names, values = names[:14][::-1], values[:14][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.7, 1.0]})
    axes[0].barh(np.arange(len(names)), values)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].set_xlabel("Median DUFS survival probability across cells")
    axes[0].set_title("Broad-pool DUFS preferences")
    axes[0].set_xlim(0, 1.03)
    axes[1].scatter(effective, ranks, s=55)
    axes[1].set_xlabel("Effective number of gated features")
    axes[1].set_ylabel("Normalized within-trace rank displacement")
    axes[1].set_title("Geometry changes, but not usefully")
    for index, path in enumerate(sorted(diagnostics_dir.glob("*.json"))):
        axes[1].annotate(path.stem.replace("qwen3_", "").replace("__", "/"),
                         (effective[index], ranks[index]), fontsize=8, xytext=(3, 3),
                         textcoords="offset points")
    fig.savefig(out_dir / "broad_gate_diagnostics.png", dpi=180)
    plt.close(fig)


def build_report(results_dir):
    systems = read_csv(results_dir / "systems_per_cell.csv")
    components = read_csv(results_dir / "components_per_cell.csv")
    macros = read_csv(results_dir / "system_macros.csv")
    lookup = _macro_lookup(macros)
    nonselection = _macro_lookup(macros, "nonselection_6_cells")
    component_all = _component_means(components)
    component_nonselection = _component_means(components, "nonselection")

    temporal = 100.0 * _float(lookup[("global_dufs", "local_temporal_core")], "f1")
    unified = 100.0 * _float(lookup[("global_dufs", "local_dufs_core")], "f1")
    broad = 100.0 * _float(lookup[("global_dufs", "local_dufs_broad")], "f1")
    mindgap = 100.0 * _float(next(row for row in macros
        if row["group"] == "all_8_cells" and row["system"] == "mindgap_control"), "f1")
    unified_nonselection = 100.0 * _float(
        nonselection[("global_dufs", "local_dufs_core")], "f1")
    temporal_nonselection = 100.0 * _float(
        nonselection[("global_dufs", "local_temporal_core")], "f1")

    per_cell = _system_lookup(systems)
    cells = [(model, subset) for model in ("qwen3_4b", "qwen3_8b")
             for subset in ("gsm8k", "math", "olympiadbench", "omnimath")]
    unified_deltas = []
    broad_deltas = []
    for model, subset in cells:
        temp = _float(per_cell[(model, subset, "global_dufs__local_temporal_core")], "f1")
        core = _float(per_cell[(model, subset, "global_dufs__local_dufs_core")], "f1")
        wide = _float(per_cell[(model, subset, "global_dufs__local_dufs_broad")], "f1")
        unified_deltas.append(core - temp)
        broad_deltas.append(wide - core)

    report = f"""# GL-LIU factorial experiment: unified graph and broad token views

Date: 2026-08-08

Status: completed diagnostic. No candidate was selected or hidden after labels were opened.

## Short conclusion

Using DUFS-LIU in both heads is the cleanest current system, but the gain over
GL-LIU v1 is small. Its eight-cell ProcessBench F1 is **{unified:.2f}%**, versus
**{temporal:.2f}%** for the frozen temporal-locator version and **{mindgap:.2f}%**
for the reproduced Mind the Gap control. The unified system wins in
{sum(value > 0 for value in unified_deltas)} of 8 cells. On the six cells outside
component selection, its F1 is **{unified_nonselection:.2f}%**, versus
**{temporal_nonselection:.2f}%** for GL-LIU v1.

Expanding the local head from five native curves to 28 token-resolved curves
does **not** help. It lowers F1 to **{broad:.2f}%**, a change of
**{broad - unified:+.2f} percentage points**, and loses in
{sum(value < 0 for value in broad_deltas)} of 8 cells. This is a useful negative
result: more token telemetry is not automatically more localization information.

## Metrics used in this report

- **Global AUROC** measures whether the complete-trace score ranks erroneous
  traces above clean traces. It does not require a decision threshold.
- **Exact localization** is the fraction of erroneous traces whose highest-risk
  token maps to the annotated first erroneous step. The global detector is not
  involved in this component metric.
- **Within-one-step localization** also counts predictions one reasoning step
  before or after the annotation.
- **ProcessBench F1** is the harmonic mean of exact erroneous-step accuracy and
  clean-trace abstention accuracy. A system must both locate errors and avoid
  flagging clean traces.

All numbers use the same 100 repeated calibration/evaluation splits. The split
spread measures calibration sensitivity; it is not a confidence interval over
new datasets.

## What was crossed

The experiment used two separate 2x2 matrices so two scientific questions were
not mixed together.

### Matrix A: graph construction

The local feature pool was fixed to the same five native curves. The global
head was IU-PCR or DUFS-LIU, and the local graph was temporal or DUFS-gated.
This tests whether a single DUFS-LIU construction can be used in both heads.

| global head | temporal LIU, core 5 | DUFS-LIU, core 5 |
|---|---:|---:|
| IU-PCR | {100*_float(lookup[("global_iu", "local_temporal_core")], "f1"):.2f}% | {100*_float(lookup[("global_iu", "local_dufs_core")], "f1"):.2f}% |
| DUFS-LIU | {temporal:.2f}% | **{unified:.2f}%** |

### Matrix B: local feature pool

The local graph was fixed to DUFS-LIU. The local feature pool was the frozen
five-view core or the broad 28-view token contract.

| global head | DUFS-LIU, core 5 | DUFS-LIU, broad 28 |
|---|---:|---:|
| IU-PCR | {100*_float(lookup[("global_iu", "local_dufs_core")], "f1"):.2f}% | {100*_float(lookup[("global_iu", "local_dufs_broad")], "f1"):.2f}% |
| DUFS-LIU | **{unified:.2f}%** | {broad:.2f}% |

![Factorial matrices](factorial_matrices.png)

## Component results

The global result reproduces the previous finding. DUFS-LIU reaches
**{100*component_all[("global_dufs", "auroc")]:.2f}% AUROC**, versus
**{100*component_all[("global_iu", "auroc")]:.2f}%** for IU-PCR. DUFS-LIU is
better in every cell, but the average difference remains only about 0.22
percentage points.

For localization before the detector threshold:

| local head | exact, all 8 | within one step, all 8 | exact, six non-selection cells |
|---|---:|---:|---:|
| temporal LIU, core 5 | {100*component_all[("local_temporal_core", "exact")]:.2f}% | {100*component_all[("local_temporal_core", "tol1")]:.2f}% | {100*component_nonselection[("local_temporal_core", "exact")]:.2f}% |
| DUFS-LIU, core 5 | **{100*component_all[("local_dufs_core", "exact")]:.2f}%** | {100*component_all[("local_dufs_core", "tol1")]:.2f}% | **{100*component_nonselection[("local_dufs_core", "exact")]:.2f}%** |
| DUFS-LIU, broad 28 | {100*component_all[("local_dufs_broad", "exact")]:.2f}% | {100*component_all[("local_dufs_broad", "tol1")]:.2f}% | {100*component_nonselection[("local_dufs_broad", "exact")]:.2f}% |

The core DUFS locator is slightly better in exact localization overall and on
the six non-selection cells. It is not uniformly better: the end-to-end gain is
positive in five cells and negative in three. The result supports simplicity
and slightly better transfer, not a large new localization effect.

![Per-cell changes](per_cell_deltas.png)

## What the 28 local curves mean

The global schema has 30 registered feature names. Twenty-nine survived in the
frozen mixed global pool for these caches. They cannot all be copied directly
to tokens:

1. `trace_length` is constant inside one trace and cannot move a token argmax;
2. `cusum_max` and `cusum_shift_idx` are two reductions of the same absolute
   CUSUM curve, so the local curve is included once;
3. `min_spilled` was saturated globally, but its rolling-minimum curve varies
   locally and was retained.

This gives 28 unique curves: raw entropy, spilled energy, log-partition energy,
top-k distribution statistics, and rolling spectral, variance, CUSUM,
permutation-entropy, tail-ratio, Hurst-proxy, and minimum curves. The rolling
spectral window was fixed at 32 tokens and the local window at 16. These are
token-resolved proxies for global reductions, not mathematically identical
copies of the full-trace statistics.

## Why the broad pool failed

The failure is not a numerical collapse:

- all 28 curves survived in every cell;
- the broad feature effective rank is about 9, so the matrix is not constant;
- the DUFS effective feature count is about 12--14;
- the broad pool changes the within-trace score ranking substantially, with a
  normalized displacement of about 0.21--0.28.

DUFS is optimizing neighbourhood preservation, not first-error localization.
It gives high survival probability to entropy and several top-k distribution
curves. Those curves form a coherent token-state geometry, but the evaluation
shows that this geometry is less aligned with the first erroneous step than the
five native dynamics. This is the central diagnosis: **DUFS can preserve a
real, stable geometry that is irrelevant to our target.**

![Broad gate diagnostics](broad_gate_diagnostics.png)

## Reproduction checks

For all eight cells, the hashes of both global scores, the temporal-core token
curve, and the DUFS-core token curve exactly match the frozen GL-LIU v1
artifacts. Therefore the {temporal:.2f}% versus {unified:.2f}% comparison changes
only the declared local graph. The broad pool is the only new score constructor.

## Scientific conclusion

1. Keep global mixed DUFS-LIU. It remains the reliable component.
2. Use five-view local DUFS-LIU as the **simplest leading candidate** for the
   next external test. It gives one graph construction in both heads and a
   small transfer advantage.
3. Do not claim that local DUFS-LIU is confirmed. The gain over temporal LIU is
   only {unified - temporal:+.2f} points and is mixed by cell.
4. Reject the naive broad-28 local pool. Do not tune subsets or windows on these
   same labels to rescue it.
5. The next useful evidence is external: a new dataset family and preferably a
   new model/output family, with both temporal-core and DUFS-core frozen.

## Claim boundary

This remains calibrated unsupervised scoring. Correctness labels are not used
to fit scores, DUFS gates, graphs, or weights. Labels are used for the repeated
calibration-half threshold and final evaluation. The 4B and 8B cells reuse the
same benchmark examples, so there are four dataset families, not eight
independent datasets.
"""
    (results_dir / "REPORT.md").write_text(report)

    advisor = f"""# Advisor brief: from GL-LIU v1 to unified DUFS-LIU

## One-sentence result

We tested whether the same DUFS-LIU construction should be used for global
error detection and token localization. It improves ProcessBench F1 from
{temporal:.2f}% to {unified:.2f}%, while expanding the local feature pool from
5 to 28 curves reduces it to {broad:.2f}%.

## What we built

GL-LIU has two heads over one LLM generation:

1. a global head decides whether the complete reasoning trace contains an error;
2. a local head ranks tokens to find the first erroneous step.

Both heads use the same two-component Laplacian IU-PCR equation. GL-LIU v1 used
a DUFS-gated sample graph globally and a temporal-chain graph locally. The new
unified candidate uses a DUFS-gated sample graph in both heads.

## What the controlled experiment says

- Global DUFS-LIU again beats global IU-PCR in all eight cells, by about 0.22
  AUROC percentage points.
- Local DUFS-LIU with the five frozen curves is slightly better than temporal
  LIU outside the development cells.
- The unified system is +{unified-temporal:.2f} ProcessBench-F1 points over
  GL-LIU v1 and +{unified-mindgap:.2f} points over Mind the Gap.
- The unified improvement is mixed: five cell wins and three losses.
- The 28-curve locator loses {unified-broad:.2f} points against the five-curve
  locator and loses in seven of eight cells.

## Interpretation

The DUFS graph is useful as a small regularizer, but DUFS does not know what an
error is. With many token curves it learns a stable geometry of token
confidence and distribution shape that does not match first-error position.
The five native dynamics contain less information but a better target-aligned
inductive bias.

## Proposed discussion decision

Freeze two local candidates for external confirmation:

- primary simplicity candidate: local DUFS-LIU, five views;
- robustness control: temporal LIU, the same five views.

Do not optimize another token feature pool on the current ProcessBench labels.
The next decision should come from a new dataset/model family and additional
published localization baselines.

## Exact claim we can make now

On the existing ProcessBench outputs and shared repeated-calibration protocol,
using DUFS-LIU in both heads gives the best internal macro result, {unified:.2f}%
F1. The gain over frozen GL-LIU v1 is small and not uniform, so it is a leading
candidate rather than a confirmed replacement.
"""
    (results_dir / "ADVISOR_BRIEF.md").write_text(advisor)

    escaped = html.escape(report)
    html_report = f"""<!doctype html><html><head><meta charset="utf-8"><title>GL-LIU factorial report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 24px;line-height:1.5}}pre{{white-space:pre-wrap;font-family:system-ui,sans-serif}}img{{max-width:100%;margin:16px 0}}</style>
</head><body><pre>{escaped}</pre>
<h2>Figures</h2>
<img src="factorial_matrices.png"><img src="per_cell_deltas.png">
<img src="method_progression.png"><img src="broad_gate_diagnostics.png">
</body></html>"""
    (results_dir / "REPORT.html").write_text(html_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    plot_factorials(read_csv(results_dir / "system_macros.csv"), results_dir)
    systems = read_csv(results_dir / "systems_per_cell.csv")
    plot_cell_deltas(systems, results_dir)
    plot_progression(read_csv(results_dir / "system_macros.csv"), results_dir)
    plot_gates(results_dir / "diagnostics", results_dir)
    build_report(results_dir)
    print(results_dir / "REPORT.md")


if __name__ == "__main__":
    main()
