#!/usr/bin/env python3
"""Build the ProcessBench latent-state report from generated CSV/JSON files.

The report builder never fits a method and never changes scores.  It refuses to
run until the explicit ``evaluate`` command has created an evaluation manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NO_ERROR = -1
VERSION = "processbench-latent-state-v1-2026-08-11"
EXPECTED_MODELS = ("qwen3_4b", "qwen3_8b")
EXPECTED_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
EXPECTED_CELLS = {
    (model, subset) for model in EXPECTED_MODELS for subset in EXPECTED_SUBSETS
}
ROOT = Path(__file__).resolve().parents[2]

SYSTEM_LABELS = {
    "global_dufs__local_iu_core": "IU-PCR core",
    "global_dufs__local_temporal_core": "Temporal LIU core",
    "global_dufs__local_dufs_core": "DUFS-LIU core",
    "global_dufs__hmm_reversible": "IU-HMM reversible",
    "global_dufs__hmm_absorbing": "IU-HMM absorbing",
    "mindgap_control": "Mind the Gap",
}
LOCAL_LABELS = {
    "local_iu_core": "IU-PCR core",
    "local_temporal_core": "Temporal LIU core",
    "local_dufs_core": "DUFS-LIU core",
    "local_hmm_reversible_core_iu": "IU-HMM reversible",
    "local_hmm_absorbing_core_iu": "IU-HMM absorbing",
    "mindgap_locator": "Mind the Gap",
}
ORDER = tuple(SYSTEM_LABELS)
LOCAL_ORDER = tuple(LOCAL_LABELS)
COLORS = {
    "global_dufs__local_iu_core": "#64748b",
    "global_dufs__local_temporal_core": "#0ea5e9",
    "global_dufs__local_dufs_core": "#2563eb",
    "global_dufs__hmm_reversible": "#16a34a",
    "global_dufs__hmm_absorbing": "#f59e0b",
    "mindgap_control": "#dc2626",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(out_dir):
    out_dir = Path(out_dir)
    evaluation = out_dir / "evaluation"
    evaluation_manifest_path = evaluation / "EVALUATION_MANIFEST.json"
    if not evaluation_manifest_path.exists():
        raise FileNotFoundError(
            "evaluation manifest is missing; run the explicit evaluate command first"
        )
    evaluation_manifest = json.load(evaluation_manifest_path.open())
    if evaluation_manifest.get("version") != VERSION:
        raise RuntimeError("evaluation-manifest version mismatch")
    cells = {
        (item["model"], item["subset"])
        for item in evaluation_manifest.get("cells", [])
    }
    if cells != EXPECTED_CELLS or len(evaluation_manifest.get("cells", [])) != 8:
        raise RuntimeError("the final report requires exactly the registered eight cells")
    required_results = {
        "systems_per_cell.csv",
        "components_per_cell.csv",
        "split_metrics.csv",
        "localization_rows.csv",
        "error_aligned_entry.csv",
    }
    recorded_results = evaluation_manifest.get("files_sha256", {})
    if set(recorded_results) != required_results:
        raise RuntimeError("evaluation manifest has an incomplete result-file roster")
    for name, expected in recorded_results.items():
        path = evaluation / name
        if sha256_file(path) != expected:
            raise RuntimeError(f"evaluated result file changed: {name}")

    freeze_path = out_dir / "FREEZE_MANIFEST.json"
    if sha256_file(freeze_path) != evaluation_manifest["score_manifest_sha256"]:
        raise RuntimeError("score-freeze manifest changed after evaluation")
    freeze = json.load(freeze_path.open())
    if freeze.get("run_fingerprint") != evaluation_manifest.get("run_fingerprint"):
        raise RuntimeError("evaluation and score-freeze fingerprints disagree")
    run_definition_path = out_dir / "RUN_DEFINITION.json"
    if sha256_file(run_definition_path) != freeze["run_definition_sha256"]:
        raise RuntimeError("run definition changed after score freeze")
    definition = json.load(run_definition_path.open())
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after fitting: {relative}")
    frozen_cells = {
        (item["model"], item["subset"]): item for item in freeze["cells"]
    }
    if set(frozen_cells) != EXPECTED_CELLS or len(freeze["cells"]) != 8:
        raise RuntimeError("score freeze does not contain exactly eight cells")
    for item in frozen_cells.values():
        if sha256_file(out_dir / item["scores"]) != item["scores_file_sha256"]:
            raise RuntimeError("frozen score file changed before report")
        if sha256_file(out_dir / item["diagnostics"]) != item["diagnostics_file_sha256"]:
            raise RuntimeError("frozen diagnostic file changed before report")
    return evaluation_manifest


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def macro(rows, key, methods, split=None):
    output = {}
    for method in methods:
        selected = [row for row in rows if row.get("system", row.get("candidate")) == method]
        if split is not None:
            selected = [row for row in selected if row.get("split") == split]
        values = np.asarray([as_float(row, key) for row in selected], dtype=float)
        output[method] = float(np.nanmean(values)) if len(values) else float("nan")
    return output


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_cells(rows, out):
    cells = sorted({(row["model"], row["subset"]) for row in rows})
    methods = ORDER
    width = 0.84 / len(methods)
    x = np.arange(len(cells))
    fig, axis = plt.subplots(figsize=(15, 5.8))
    for index, method in enumerate(methods):
        lookup = {(row["model"], row["subset"]): as_float(row, "f1")
                  for row in rows if row["system"] == method}
        values = [100.0 * lookup.get(cell, np.nan) for cell in cells]
        axis.bar(x - 0.42 + width / 2 + index * width, values, width,
                 label=SYSTEM_LABELS[method], color=COLORS[method])
    axis.set_ylabel("ProcessBench F1 (%)")
    axis.set_xticks(x, [f"{model.replace('qwen3_', '')}\n{subset}" for model, subset in cells])
    axis.set_title("End-to-end first-error performance by model and dataset")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, out)


def plot_local_exact(rows, out):
    all_macro = macro(rows, "exact", LOCAL_ORDER)
    nonselection = macro(rows, "exact", LOCAL_ORDER, split="nonselection")
    methods = LOCAL_ORDER
    x = np.arange(len(methods))
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - 0.18, [100 * all_macro[m] for m in methods], 0.36,
             label="All eight cells", color="#2563eb")
    axis.bar(x + 0.18, [100 * nonselection[m] for m in methods], 0.36,
             label="Six non-selection cells", color="#93c5fd")
    axis.set_xticks(x, [LOCAL_LABELS[m] for m in methods], rotation=18, ha="right")
    axis.set_ylabel("Exact first-error localization (%)")
    axis.set_title("Localizers before the global detection threshold")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    save_figure(fig, out)


def plot_paired_delta(rows, out):
    baseline = {
        (row["model"], row["subset"]): as_float(row, "f1")
        for row in rows if row["system"] == "global_dufs__local_dufs_core"
    }
    primary = {
        (row["model"], row["subset"]): as_float(row, "f1")
        for row in rows if row["system"] == "global_dufs__hmm_reversible"
    }
    cells = sorted(set(baseline).intersection(primary))
    delta = [100 * (primary[cell] - baseline[cell]) for cell in cells]
    fig, axis = plt.subplots(figsize=(11, 5.3))
    color = ["#16a34a" if value >= 0 else "#dc2626" for value in delta]
    axis.barh(np.arange(len(cells)), delta, color=color)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(cells)), [f"{m.replace('qwen3_', '')} / {s}" for m, s in cells])
    axis.set_xlabel("F1 difference: reversible IU-HMM minus DUFS-LIU (points)")
    axis.set_title("Does the latent-state onset model improve the frozen localizer?")
    axis.grid(axis="x", alpha=0.2)
    save_figure(fig, out)


def plot_signed_errors(rows, out):
    methods = (
        "local_dufs_core",
        "local_hmm_reversible_core_iu",
        "local_hmm_absorbing_core_iu",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, method in zip(axes, methods):
        values = []
        for row in rows:
            if row["candidate"] != method:
                continue
            gold, predicted = int(row["gold_step"]), int(row["predicted_step"])
            if gold != NO_ERROR and predicted != NO_ERROR:
                values.append(predicted - gold)
        clipped = np.clip(values, -6, 6) if values else np.asarray([])
        axis.hist(clipped, bins=np.arange(-6.5, 7.5), color="#64748b", edgecolor="white")
        axis.axvline(0, color="#16a34a", linewidth=1.5)
        axis.set_title(LOCAL_LABELS[method])
        axis.set_xlabel("Predicted step − gold step\n(clipped to ±6)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Erroneous traces")
    fig.suptitle("Localization bias: early predictions are negative; late predictions are positive")
    save_figure(fig, out)


def plot_position_length(rows, out):
    methods = ("local_iu_core", "local_hmm_reversible_core_iu")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, method in zip(axes, methods):
        selected = [
            row for row in rows
            if row["candidate"] == method
        ]
        length = np.asarray([int(row["trace_tokens"]) for row in selected], dtype=float)
        position = np.asarray([int(row["predicted_token"]) for row in selected], dtype=float)
        relative = position / np.maximum(length - 1, 1)
        keep = np.linspace(0, len(selected) - 1, min(len(selected), 1800), dtype=int) if selected else []
        axis.scatter(length[keep], relative[keep], s=6, alpha=0.20, color="#2563eb")
        axis.set_xscale("log")
        axis.set_title(LOCAL_LABELS[method])
        axis.set_xlabel("Trace length in tokens (log scale)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted relative token position")
    fig.suptitle("Position shortcut diagnostic")
    save_figure(fig, out)


def diagnostic_rows(out_dir):
    rows = []
    for path in sorted((Path(out_dir) / "label_free_diagnostics").glob("*.json")):
        data = json.load(path.open())
        for kind in ("reversible", "absorbing"):
            item = data["hmm"][kind]
            selected = item.get("selected")
            applied = item.get("apply", {})
            rows.append({
                "cell": f"{data['model'].replace('qwen3_', '')}/{data['subset']}",
                "kind": kind,
                "fallback": bool(item["fallback"]),
                "separation": np.nan if selected is None else float(selected["separation"]),
                "p01": np.nan if selected is None else float(selected["transition"][0][1]),
                "p10": np.nan if selected is None else float(selected["transition"][1][0]),
                "occupancy_high": np.nan if selected is None else float(selected["occupancy"][1]),
                "mean_low": np.nan if selected is None else float(selected["means"][0]),
                "mean_high": np.nan if selected is None else float(selected["means"][1]),
                "variance": np.nan if selected is None else float(selected["variance"]),
                "seed_agreement": float(item["fit"].get("mean_pair_exact_argmax_agreement", np.nan)),
                "seed_displacement": float(item["fit"].get("mean_pair_normalized_argmax_displacement", np.nan)),
                "mean_peak": float(applied.get("mean_peak_entry_probability", np.nan)),
                "entry_entropy": float(applied.get("mean_normalized_entry_entropy", np.nan)),
                "no_entry_fraction": float(applied.get("fraction_without_entry_above_0p10", np.nan)),
                "mean_position": float(applied.get("mean_normalized_entry_position", np.nan)),
            })
    return rows


def plot_hmm_diagnostics(rows, out):
    cells = sorted({row["cell"] for row in rows})
    lookup = {(row["cell"], row["kind"]): row for row in rows}
    x = np.arange(len(cells))
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    for offset, kind, color in ((-0.18, "reversible", "#16a34a"), (0.18, "absorbing", "#f59e0b")):
        axes[0].bar(x + offset, [lookup[(cell, kind)]["separation"] for cell in cells],
                    0.36, label=kind, color=color)
        axes[1].bar(x + offset, [lookup[(cell, kind)]["occupancy_high"] for cell in cells],
                    0.36, label=kind, color=color)
        axes[2].plot(x + offset, [lookup[(cell, kind)]["p01"] for cell in cells],
                     marker="o", label=f"{kind} 0→1", color=color)
    axes[2].plot(x, [lookup[(cell, "reversible")]["p10"] for cell in cells],
                 marker="x", linestyle="--", color="#2563eb", label="reversible 1→0")
    for offset, kind, color in ((-0.18, "reversible", "#16a34a"), (0.18, "absorbing", "#f59e0b")):
        axes[3].bar(x + offset, [lookup[(cell, kind)]["seed_agreement"] for cell in cells],
                    0.36, label=kind, color=color)
    axes[0].axhline(0.25, color="black", linestyle="--", linewidth=1, label="guard")
    axes[0].set_ylabel("State mean separation (shared SD)")
    axes[0].set_title("Label-free state identifiability")
    axes[1].set_ylabel("High-risk state occupancy")
    axes[1].set_title("How much of the token population each state explains")
    axes[2].set_ylabel("Transition probability")
    axes[2].set_title("Learned entry and exit probabilities")
    axes[2].legend(frameon=False, ncol=3)
    axes[3].set_ylabel("Exact locator agreement across valid seeds")
    axes[3].set_ylim(0, 1.02)
    axes[3].set_xticks(x, cells, rotation=25, ha="right")
    axes[3].set_title("Optimization stability")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    save_figure(fig, out)


def plot_posterior_diagnostics(rows, out):
    cells = sorted({row["cell"] for row in rows})
    lookup = {(row["cell"], row["kind"]): row for row in rows}
    metrics = (
        ("mean_peak", "Mean peak entry probability"),
        ("entry_entropy", "Normalized entry-curve entropy"),
        ("no_entry_fraction", "Fraction with no entry above 0.10"),
        ("mean_position", "Mean predicted relative position"),
    )
    x = np.arange(len(cells))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for offset, kind, color in (
            (-0.18, "reversible", "#16a34a"),
            (0.18, "absorbing", "#f59e0b"),
        ):
            axis.bar(
                x + offset,
                [lookup[(cell, kind)][metric] for cell in cells],
                0.36,
                label=kind,
                color=color,
            )
        axis.set_title(title)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[-1]:
        axis.set_xticks(x, cells, rotation=30, ha="right")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Can the HMM identify one credible onset?")
    save_figure(fig, out)


def plot_error_aligned(rows, out):
    labels = {
        "local_hmm_reversible_core_iu": "reversible IU-HMM",
        "local_hmm_absorbing_core_iu": "absorbing IU-HMM",
    }
    colors = {
        "local_hmm_reversible_core_iu": "#16a34a",
        "local_hmm_absorbing_core_iu": "#f59e0b",
    }
    fig, axis = plt.subplots(figsize=(9, 5))
    for method in labels:
        selected = [row for row in rows if row["candidate"] == method]
        offsets = sorted({int(row["relative_token"]) for row in selected})
        means = []
        for offset in offsets:
            current = [row for row in selected if int(row["relative_token"]) == offset]
            count = sum(int(row["n"]) for row in current)
            total = sum(float(row["sum_entry_probability"]) for row in current)
            means.append(total / count if count else np.nan)
        axis.plot(offsets, means, label=labels[method], color=colors[method])
    axis.axvline(0, color="black", linestyle="--", linewidth=1,
                 label="first token of annotated error step")
    axis.set_xlabel("Tokens relative to the annotated first-error step")
    axis.set_ylabel("Mean HMM entry probability")
    axis.set_title("Error-aligned behavior (evaluation view; fallback cells excluded)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(fig, out)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def inline_markup(value):
    """Render the tiny inline subset used by this generated report."""

    escaped = html.escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def markdown_to_html(markdown):
    """Render this report's small Markdown subset, including real HTML tables."""

    lines = markdown.splitlines()
    sections = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# "):
            sections.append(f"<h1>{inline_markup(line[2:])}</h1>")
        elif line.startswith("## "):
            sections.append(f"<h2>{inline_markup(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line:
            alt = line[2:line.index("]")]
            src = line[line.index("(") + 1:line.rindex(")")]
            sections.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"></figure>'
            )
        elif line.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            index -= 1
            cells = [
                [inline_markup(value.strip()) for value in row.strip("|").split("|")]
                for row in table_lines
            ]
            if len(cells) >= 2:
                header_cells = "".join(f"<th>{value}</th>" for value in cells[0])
                body_rows = "".join(
                    "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
                    for row in cells[2:]
                )
                sections.append(
                    f'<div class="table-wrap"><table><thead><tr>{header_cells}</tr></thead>'
                    f"<tbody>{body_rows}</tbody></table></div>"
                )
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(f"<li>{inline_markup(lines[index][2:])}</li>")
                index += 1
            index -= 1
            sections.append("<ul>" + "".join(items) + "</ul>")
        elif line.strip():
            sections.append(f"<p>{inline_markup(line)}</p>")
        index += 1
    return "\n".join(sections)


def fmt(value, scale=100.0):
    return "—" if not np.isfinite(value) else f"{scale * value:.2f}"


def build_report(out_dir):
    out_dir = Path(out_dir)
    evaluation = out_dir / "evaluation"
    verify_artifacts(out_dir)
    systems = read_csv(evaluation / "systems_per_cell.csv")
    components = read_csv(evaluation / "components_per_cell.csv")
    split_rows = read_csv(evaluation / "split_metrics.csv")
    predictions = read_csv(evaluation / "localization_rows.csv")
    aligned = read_csv(evaluation / "error_aligned_entry.csv")
    expected_systems = {
        (model, subset, method)
        for model, subset in EXPECTED_CELLS for method in ORDER
    }
    observed_systems = {
        (row["model"], row["subset"], row["system"]) for row in systems
    }
    if observed_systems != expected_systems or len(systems) != len(expected_systems):
        raise RuntimeError("system table is not the exact eight-cell method grid")
    expected_components = {
        (model, subset, method)
        for model, subset in EXPECTED_CELLS for method in LOCAL_ORDER
    }
    observed_components = {
        (row["model"], row["subset"], row["candidate"]) for row in components
    }
    if observed_components != expected_components or len(components) != len(expected_components):
        raise RuntimeError("component table is not the exact eight-cell method grid")
    expected_splits = {
        (model, subset, method, index)
        for model, subset in EXPECTED_CELLS
        for method in ORDER for index in range(100)
    }
    observed_splits = {
        (row["model"], row["subset"], row["method"], int(row["split_index"]))
        for row in split_rows
    }
    if observed_splits != expected_splits or len(split_rows) != len(expected_splits):
        raise RuntimeError("split table is incomplete or contains duplicates")
    prediction_keys = [
        (row["model"], row["subset"], row["row_id"], row["candidate"])
        for row in predictions
    ]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise RuntimeError("localization rows contain duplicates")
    expected_aligned = {
        (model, subset, method, offset)
        for model, subset in EXPECTED_CELLS
        for method in (
            "local_hmm_reversible_core_iu",
            "local_hmm_absorbing_core_iu",
        )
        for offset in range(-50, 51)
    }
    observed_aligned = {
        (
            row["model"], row["subset"], row["candidate"],
            int(row["relative_token"]),
        )
        for row in aligned
    }
    if observed_aligned != expected_aligned or len(aligned) != len(expected_aligned):
        raise RuntimeError("error-aligned table is incomplete or duplicated")
    diagnostics = diagnostic_rows(out_dir)
    if len(diagnostics) != 16:
        raise RuntimeError("expected reversible and absorbing diagnostics for eight cells")
    figures = out_dir / "figures"
    plot_f1_cells(systems, figures / "end_to_end_f1_per_cell.png")
    plot_local_exact(components, figures / "local_exact_macro.png")
    plot_paired_delta(systems, figures / "paired_f1_delta.png")
    plot_signed_errors(predictions, figures / "signed_step_error.png")
    plot_position_length(predictions, figures / "position_length.png")
    plot_hmm_diagnostics(diagnostics, figures / "hmm_diagnostics.png")
    plot_posterior_diagnostics(diagnostics, figures / "posterior_diagnostics.png")
    plot_error_aligned(aligned, figures / "error_aligned_entry.png")

    all_f1 = macro(systems, "f1", ORDER)
    non_f1 = macro(systems, "f1", ORDER, split="nonselection")
    all_error_accuracy = macro(systems, "acc_erroneous", ORDER)
    all_clean_accuracy = macro(systems, "acc_correct", ORDER)
    all_system_tol1 = macro(systems, "sla_tol1", ORDER)
    all_exact = macro(components, "exact", LOCAL_ORDER)
    non_exact = macro(components, "exact", LOCAL_ORDER, split="nonselection")
    all_tol1 = macro(components, "tol1", LOCAL_ORDER)
    all_signed = macro(components, "mean_signed_step_error", LOCAL_ORDER)
    all_distance = macro(components, "mean_normalized_token_distance", LOCAL_ORDER)
    rows = [
        [
            SYSTEM_LABELS[method],
            fmt(all_f1[method]),
            fmt(all_error_accuracy[method]),
            fmt(all_clean_accuracy[method]),
            fmt(all_system_tol1[method]),
            fmt(non_f1[method]),
        ]
        for method in ORDER
    ]
    local_rows = [
        [
            LOCAL_LABELS[method],
            fmt(all_exact[method]),
            fmt(all_tol1[method]),
            fmt(all_signed[method], scale=1.0),
            fmt(all_distance[method]),
            fmt(non_exact[method]),
        ]
        for method in LOCAL_ORDER
    ]
    mechanism_rows = [
        [
            row["cell"],
            row["kind"],
            "yes" if row["fallback"] else "no",
            fmt(row["separation"], scale=1.0),
            fmt(row["occupancy_high"]),
            fmt(row["variance"], scale=1.0),
            fmt(row["p01"]),
            fmt(row["p10"]),
            fmt(row["mean_peak"]),
            fmt(row["entry_entropy"]),
            fmt(row["no_entry_fraction"]),
        ]
        for row in diagnostics
    ]

    delta_f1 = all_f1["global_dufs__hmm_reversible"] - all_f1["global_dufs__local_dufs_core"]
    delta_f1_non = non_f1["global_dufs__hmm_reversible"] - non_f1["global_dufs__local_dufs_core"]
    delta_f1_iu = all_f1["global_dufs__hmm_reversible"] - all_f1["global_dufs__local_iu_core"]
    delta_exact = all_exact["local_hmm_reversible_core_iu"] - all_exact["local_dufs_core"]
    delta_exact_non = non_exact["local_hmm_reversible_core_iu"] - non_exact["local_dufs_core"]
    delta_exact_iu = all_exact["local_hmm_reversible_core_iu"] - all_exact["local_iu_core"]
    cell_delta = []
    for row in systems:
        if row["system"] != "global_dufs__hmm_reversible":
            continue
        match = next(item for item in systems if item["model"] == row["model"]
                     and item["subset"] == row["subset"]
                     and item["system"] == "global_dufs__local_dufs_core")
        cell_delta.append(as_float(row, "f1") - as_float(match, "f1"))
    fallback_count = sum(row["fallback"] for row in diagnostics if row["kind"] == "reversible")

    split_lookup = {}
    for row in split_rows:
        key = (row["model"], row["subset"], int(row["split_index"]))
        split_lookup.setdefault(key, {})[row["method"]] = as_float(row, "f1")
    paired_by_split = {}
    for (_, _, split_index), values in split_lookup.items():
        if {"global_dufs__hmm_reversible", "global_dufs__local_dufs_core"}.issubset(values):
            # The protocol reuses each split index across cells.  Average the
            # paired cell deltas first so the displayed range is macro split
            # variability, rather than treating cell-by-split rows as iid.
            paired_by_split.setdefault(split_index, []).append(
                values["global_dufs__hmm_reversible"]
                - values["global_dufs__local_dufs_core"]
            )
    paired = [float(np.mean(values)) for values in paired_by_split.values()]
    split_interval = np.quantile(paired, (0.025, 0.975)) if paired else (np.nan, np.nan)

    gate = {
        "no reversible HMM fallback": fallback_count == 0,
        "all-cell exact localization improves": delta_exact > 0,
        "non-selection exact localization improves": delta_exact_non > 0,
        "all-cell ProcessBench F1 improves": delta_f1 > 0,
        "non-selection ProcessBench F1 improves": delta_f1_non > 0,
        "no cell loses more than one F1 point": min(cell_delta, default=-np.inf) >= -0.01,
    }
    verdict = "PROMISING" if all(gate.values()) else "NOT PROMOTED"
    gate_rows = [[name, "PASS" if passed else "FAIL"] for name, passed in gate.items()]

    md = f"""# IU-PCR latent-state localization on ProcessBench

## Result

**Decision: {verdict}.** The reversible IU-HMM changes all-cell ProcessBench
F1 by **{100 * delta_f1:+.2f} points** and exact localization by
**{100 * delta_exact:+.2f} points** relative to the frozen core-five
DUFS-LIU localizer. On the six cells not used by the earlier GL-LIU component
selection, the differences are **{100 * delta_f1_non:+.2f} F1 points** and
**{100 * delta_exact_non:+.2f} exact-localization points**.

The direct mechanism control is also negative: relative to the ordinary IU-PCR
sequence that initializes it, the HMM changes PB-F1 by
**{100 * delta_f1_iu:+.2f} points** and raw exact localization by
**{100 * delta_exact_iu:+.2f} points**. The six non-selection cells are held-out
method-selection cells from four dataset families and two scorer models; they
are not six independent datasets.

This comparison is exploratory: these ProcessBench labels were already opened
in earlier project experiments. Labels were not used to fit IU-PCR, either HMM,
the global detector, or any score. A separate process froze and hashed every
score before the evaluation command read labels or step spans.

## What the metrics mean

- **Exact localization** is the percentage of erroneous traces whose predicted
  token maps to the annotated first erroneous reasoning step.
- **Within one step** also accepts the adjacent step.
- **Clean accuracy** is the percentage of fully correct traces on which the
  global detector abstains.
- **ProcessBench F1** is the harmonic mean of exact localization on erroneous
  traces and clean accuracy. It therefore tests both detection and placement.

All headline tables are equal-cell macro averages. They do not pool every trace
as though the two scorer-model views of a dataset were independent samples.

## Methods

Every method except Mind the Gap uses the same frozen mixed-v2 DUFS-LIU global
detector. The local input is always the same five token curves: entropy,
entropy sliding variance, absolute entropy CUSUM, spilled-energy sliding
variance, and absolute spilled-energy CUSUM.

Ordinary two-component IU-PCR fuses these curves into one scalar token-risk
sequence. The primary HMM has two reversible latent states. Its output at token
`t` is the posterior probability of entering the higher-IU-risk state at `t`.
The absorbing HMM is a falsification control for the stronger assumption that
the risk state persists after the first error. Both HMMs share one emission
variance, use three deterministic starts, and select the valid start with the
largest label-free likelihood. Failed state-separation or collapse guards fall
back exactly to the IU-PCR argmax.

## End-to-end results

{markdown_table(["System", "All 8 PB-F1 (%)", "Erroneous exact (%)", "Clean accuracy (%)", "Within-one SLA (%)", "Non-selection 6 PB-F1 (%)"], rows)}

![End-to-end F1](figures/end_to_end_f1_per_cell.png)

## Localizer results

{markdown_table(["Localizer", "All 8 exact (%)", "Within one step (%)", "Mean signed step error", "Normalized token distance (%)", "Non-selection 6 exact (%)"], local_rows)}

![Local exact](figures/local_exact_macro.png)

![Paired F1 delta](figures/paired_f1_delta.png)

The 95% range across the existing repeated calibration splits is
[{100 * split_interval[0]:+.2f}, {100 * split_interval[1]:+.2f}] F1 points.
This is split variability, not an independent-data confidence interval.

## Mechanism diagnostics

![HMM diagnostics](figures/hmm_diagnostics.png)

![Posterior diagnostics](figures/posterior_diagnostics.png)

{markdown_table(["Cell", "HMM", "Fallback", "Separation (SD)", "High occupancy (%)", "Variance", "P(0 to 1) (%)", "P(1 to 0) (%)", "Mean peak (%)", "Entry entropy (%)", "No credible entry (%)"], mechanism_rows)}

![Error-aligned entry probability](figures/error_aligned_entry.png)

The error-aligned plot includes only cells that produced a true posterior entry
curve. A guarded fallback returns the IU-PCR risk curve and is never averaged
with probabilities. Its sharp mean peak at the annotated boundary is an
interesting mechanism observation, not a performance claim: every annotated
error begins at a reasoning-step boundary, so a matched non-error-boundary
control is still required to distinguish error onset from generic step syntax.

![Signed step error](figures/signed_step_error.png)

![Position-length diagnostic](figures/position_length.png)

## Pre-declared decision panel

{markdown_table(["Condition", "Result"], gate_rows)}

## Interpretation boundary

An HMM gain would show that explicit temporal state transitions improve the
same IU-PCR signal. It would not show that the hidden state is literally
correctness. A likelihood gain without localization gain means the HMM found a
stable temporal regime unrelated to the benchmark target. An absorbing-model
failure would specifically reject persistent post-error telemetry, not the
reversible onset model.
"""
    (out_dir / "REPORT.md").write_text(md)

    # The HTML embeds the generated content and local figures. Values come only
    # from CSV/JSON above; no result is typed manually.
    rendered = markdown_to_html(md)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>IU-PCR latent-state localization</title>
<style>body{font-family:Inter,Arial,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172033;line-height:1.55}h1{font-size:38px}h2{margin-top:38px;color:#174ea6}p,li{font-size:16px}img{width:100%;border:1px solid #dbe3ef;border-radius:12px;background:white}figure{margin:24px 0}.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse}th,td{padding:10px 12px;border-bottom:1px solid #dbe3ef;text-align:left}th{background:#eef4fb;color:#174ea6}</style></head><body>""" + rendered + "</body></html>"
    (out_dir / "REPORT.html").write_text(page)
    (out_dir / "REPORT_MANIFEST.json").write_text(json.dumps({
        "version": VERSION,
        "evaluation_manifest_sha256": sha256_file(
            evaluation / "EVALUATION_MANIFEST.json"
        ),
        "report_md_sha256": sha256_file(out_dir / "REPORT.md"),
        "report_html_sha256": sha256_file(out_dir / "REPORT.html"),
        "figures_sha256": {
            path.name: sha256_file(path) for path in sorted(figures.glob("*.png"))
        },
    }, indent=2, sort_keys=True) + "\n")
    print(out_dir / "REPORT.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    build_report(args.out_dir)


if __name__ == "__main__":
    main()
