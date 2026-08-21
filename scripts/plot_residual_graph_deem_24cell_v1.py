#!/usr/bin/env python3
"""Generate the twelve preregistered Residual-Graph DEEM figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import LAMBDA_GRID, SEEDS, symmetric_normalized_laplacian
from spectral_utils.residual_graph_deem_data import load_target_free_bundle
from spectral_utils.residual_graph_deem_labels import join_labels_by_id, load_label_sidecar


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def csr_from_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return sparse.csr_matrix(
            (data["graph_data"], data["graph_indices"], data["graph_indptr"]),
            shape=tuple(data["graph_shape"]),
        )


def score(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["score"], dtype=float)


def placeholder(out: Path, name: str, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(.5, .5, message, ha="center", va="center", wrap=True)
    save(fig, out, name)


def architecture(out: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 3.4))
    axis.axis("off")
    labels = [
        "risk-oriented\ncell inventory", "present-family\nadditive DEEM",
        "grouped 5-fold\nOOF contributions", "donor-only\nlength/logit residuals",
        "cross-view DUFS\nmetric", "frozen sparse\nkNN graph",
        "target smooth /\nnuisance DEEM",
    ]
    x = np.linspace(.06, .94, len(labels))
    for index, (position, label) in enumerate(zip(x, labels)):
        axis.text(position, .5, label, ha="center", va="center", fontsize=9,
                  bbox={"boxstyle": "round,pad=.45", "facecolor": "#e8f1fa", "edgecolor": "#315a7d"})
        if index:
            axis.annotate("", xy=(position-.055, .5), xytext=(x[index-1]+.055, .5),
                          arrowprops={"arrowstyle": "->", "color": "#315a7d"})
    axis.set_title("Residual-Graph DEEM v1 — target-free Stage A", fontsize=14)
    save(fig, out, "01_architecture.png")


def score_map(evaluation: Path, out: Path) -> None:
    selected = [row for row in rows(evaluation / "PER_CELL.csv") if row["method"] in {"B0", "B3", "G2", "G3", "G4"}]
    cells = []
    for row in selected:
        if row["cell_id"] not in cells:
            cells.append(row["cell_id"])
    methods = ("B0", "B3", "G2", "G3", "G4")
    matrix = np.asarray([[float(next(r["auroc"] for r in selected if r["cell_id"] == c and r["method"] == m)) for m in methods] for c in cells])
    fig, axis = plt.subplots(figsize=(8, 10))
    image = axis.imshow(matrix, aspect="auto", vmin=.35, vmax=.85, cmap="viridis")
    axis.set_xticks(range(len(methods)), methods)
    axis.set_yticks(range(len(cells)), cells, fontsize=7)
    axis.set_title("Per-cell AUROC (all 24 cells)")
    fig.colorbar(image, ax=axis, label="AUROC")
    save(fig, out, "02_per_cell_score_map.png")


def forest(evaluation: Path, out: Path) -> None:
    data = json.loads((evaluation / "BOOTSTRAP.json").read_text(encoding="utf-8"))
    keys = [key for key in ("B3_vs_G3", "B3_vs_G4", "B0_vs_B3") if key in data]
    fig, axis = plt.subplots(figsize=(8, max(3, 1.1 * len(keys))))
    for y, key in enumerate(keys):
        value = data[key]
        axis.errorbar(value["observed"], y,
                      xerr=[[value["observed"]-value["lower"]], [value["upper"]-value["observed"]]],
                      fmt="o", capsize=4)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_yticks(range(len(keys)), keys)
    axis.set_xlabel("equal-family paired AUROC change")
    axis.set_title("Family-blocked paired changes (95% interval)")
    save(fig, out, "03_paired_change_forest.png")


def atlas(run: Path, bundle_dir: Path, sidecar_dir: Path, evaluation: Path, out: Path) -> None:
    phase = json.loads((evaluation / "EVALUATION_COMPLETE.json").read_text(encoding="utf-8"))
    target_lambda = float(phase["nominated_lambdas"]["target"])
    token = str(target_lambda).replace(".", "p")
    cells = ("sciq_llama8b", "math500_dsmath7b", "seiclr_triviaqa_opt30b")
    fig, axes = plt.subplots(len(cells), 3, figsize=(12, 10))
    for row_index, cell in enumerate(cells):
        bundle = load_target_free_bundle(bundle_dir / f"{cell}.npz")
        y = join_labels_by_id(bundle, load_label_sidecar(sidecar_dir / f"{cell}.npz"))
        graph_path = run / "fits" / cell / f"G3__lambda{token}__seed0.npz"
        W = csr_from_npz(graph_path)
        L = symmetric_normalized_laplacian(W)
        try:
            _, vectors = eigsh(L, k=3, which="SM", tol=1e-4)
            xy = vectors[:, 1:3]
        except Exception:
            xy = np.column_stack([np.arange(len(y)), np.zeros(len(y))])
        b3 = score(run / "fits" / cell / "B3__seed0.npz")
        for axis, color, title in zip(
            axes[row_index], (y, np.log1p(bundle.raw_trace_length), b3),
            ("hallucination", "log length", "B3 posterior"),
        ):
            take = np.linspace(0, len(y)-1, min(len(y), 1800), dtype=int)
            scatter = axis.scatter(xy[take, 0], xy[take, 1], c=np.asarray(color)[take], s=4, cmap="viridis", alpha=.7)
            axis.set_title(f"{cell}\n{title}", fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
            fig.colorbar(scatter, ax=axis, fraction=.04)
    save(fig, out, "04_residual_graph_atlas.png")


def neighbor_panels(evaluation: Path, out: Path) -> None:
    data = rows(evaluation / "CONDITIONAL_GEOMETRY.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    roles = ("B0_LINEAR", "G1", "G2", "G3", "G4")
    for role in roles:
        current = [row for row in data if row["graph_role"] == role]
        if current:
            axes[0].plot([1, 2], [np.mean([float(r["target_effect_exact"]) for r in current]),
                                  np.mean([float(r["target_effect_crt"]) for r in current])], marker="o", label=role)
            axes[1].scatter(np.mean([float(r["length_rayleigh"]) for r in current]),
                            np.mean([float(r["target_rayleigh"]) for r in current]), label=role)
    axes[0].set_xticks([1, 2], ["exact length", "CRT"]); axes[0].set_ylabel("target smoothness effect")
    axes[1].set_xlabel("length Rayleigh"); axes[1].set_ylabel("target Rayleigh")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    axes[0].set_title("Conditional neighborhood composition")
    axes[1].set_title("Target versus length neighborhood structure")
    save(fig, out, "05_neighbor_composition.png")


def raw_residual(evaluation: Path, out: Path) -> None:
    health = rows(evaluation / "GRAPH_HEALTH.csv")
    roles = ("G0", "G1", "G2", "G3")
    component = [np.mean([float(r["largest_component_fraction"]) for r in health if r["arm_id"] == role]) for role in roles]
    isolated = [np.mean([float(r["isolated_fraction"]) for r in health if r["arm_id"] == role]) for role in roles]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(roles, component); axes[0].axhline(.9, color="red", linestyle="--"); axes[0].set_ylim(0, 1); axes[0].set_title("Largest component")
    axes[1].bar(roles, isolated); axes[1].axhline(.05, color="red", linestyle="--"); axes[1].set_title("Isolated fraction")
    save(fig, out, "06_raw_vs_residual.png")


def gate_heatmap(evaluation: Path, out: Path) -> None:
    data = rows(evaluation / "GATE_STABILITY.csv")
    features = sorted({r["feature_name"] for r in data})
    cells = sorted({r["cell_id"] for r in data})
    matrix = np.full((len(cells), len(features)), np.nan)
    for i, cell in enumerate(cells):
        for j, feature in enumerate(features):
            current = [float(r["gate_weight"]) for r in data if r["cell_id"] == cell and r["feature_name"] == feature]
            if current: matrix[i, j] = np.mean(current)
    fig, axis = plt.subplots(figsize=(14, 8))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(features)), features, rotation=90, fontsize=6)
    axis.set_yticks(range(len(cells)), cells, fontsize=6)
    axis.set_title("Residual DUFS gates (fold/seed mean)")
    fig.colorbar(image, ax=axis)
    save(fig, out, "07_dufs_gate_heatmap.png")


def actuation(evaluation: Path, out: Path) -> None:
    data = rows(evaluation / "PER_FIT.csv")
    g3 = [r for r in data if r.get("arm_id") == "G3" and r.get("is_headline") == "True"]
    g4 = [r for r in data if r.get("arm_id") == "G4" and r.get("is_headline") == "True"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([float(r["posterior_sd"]) for r in g3 if r.get("posterior_sd")], bins=20, alpha=.7, label="G3")
    axes[0].hist([float(r["posterior_sd"]) for r in g4 if r.get("posterior_sd")], bins=20, alpha=.7, label="G4")
    axes[0].axvline(.001, color="red", linestyle="--"); axes[0].legend(); axes[0].set_title("Posterior variance / collapse check")
    axes[1].scatter([float(r.get("nuisance_variance_min", "nan")) for r in g4],
                    [float(r.get("logit_nuisance_dependence", "nan")) for r in g4], s=10)
    axes[1].set_xlabel("minimum nuisance variance"); axes[1].set_ylabel("logit–nuisance dependence"); axes[1].set_title("Nuisance actuation")
    save(fig, out, "08_target_vs_nuisance_actuation.png")


def linear_graph(evaluation: Path, out: Path) -> None:
    data = rows(evaluation / "CONDITIONAL_GEOMETRY.csv")
    lookup = {(r["cell_id"], r["seed"], r["graph_role"]): r for r in data}
    points = []
    for (cell, seed, role), value in lookup.items():
        if role == "G3" and (cell, seed, "B0_LINEAR") in lookup:
            points.append((float(lookup[(cell, seed, "B0_LINEAR")]["min_conditional_effect"]), float(value["min_conditional_effect"])))
    fig, axis = plt.subplots(figsize=(6, 6))
    if points:
        array = np.asarray(points); axis.scatter(array[:, 0], array[:, 1], s=12, alpha=.7)
        limits = [float(np.nanmin(array)), float(np.nanmax(array))]; axis.plot(limits, limits, "k--")
    axis.set_xlabel("B0 1-D graph effect"); axis.set_ylabel("G3 residual graph effect"); axis.set_title("Linear versus residual graph")
    save(fig, out, "09_linear_vs_graph.png")


def controls(evaluation: Path, out: Path) -> None:
    value = json.loads((evaluation / "CONTROLS.json").read_text(encoding="utf-8"))
    names, deltas = [], []
    for name, summary in value.get("summary", {}).items():
        names.append(name); deltas.append(float(summary["equal_family_delta_vs_B3"]))
    fig, axis = plt.subplots(figsize=(10, 4))
    colors = ["#b2182b" if value.get("failures", {}).get(name) else "#4d9221" for name in names]
    axis.bar(names, deltas, color=colors); axis.axhline(0, color="black", linewidth=.8)
    axis.tick_params(axis="x", rotation=35); axis.set_ylabel("AUROC change vs B3"); axis.set_title("Control dashboard (red = failed lane)")
    save(fig, out, "10_control_dashboard.png")


def lambda_paths(phase0_dir: Path, out: Path) -> None:
    values = json.loads((phase0_dir / "PHASE0_RESULTS.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(9, 5))
    for mechanism in ("target", "nuisance", "family"):
        x, y = [], []
        for value in LAMBDA_GRID:
            selected = [float(r["delta"]) for r in values if r["mechanism"] == mechanism and float(r["lambda"]) == value]
            if selected: x.append(value); y.append(np.mean(selected))
        axis.plot(x, y, marker="o", label=mechanism)
    axis.axhline(0, color="black", linewidth=.8); axis.set_xscale("symlog", linthresh=.005)
    axis.set_xlabel("lambda"); axis.set_ylabel("synthetic mean ΔAUROC"); axis.set_title("Frozen Phase-0 lambda paths"); axis.legend()
    save(fig, out, "11_lambda_paths.png")


def stability(evaluation: Path, out: Path) -> None:
    value = json.loads((evaluation / "SEED_STABILITY.json").read_text(encoding="utf-8"))
    cells = sorted(value); methods = ("B1", "B2", "B3", "G3", "G4")
    fig, axis = plt.subplots(figsize=(13, 6))
    width = .15; x = np.arange(len(cells))
    for index, method in enumerate(methods):
        axis.bar(x + (index-2)*width, [float(value[c][method]["median_abs_spearman"]) for c in cells], width, label=method)
    axis.axhline(.9, color="red", linestyle="--"); axis.set_ylim(0, 1.02)
    axis.set_xticks(x, cells, rotation=90, fontsize=6); axis.set_ylabel("median |Spearman| across seeds"); axis.set_title("Seed stability across all cells"); axis.legend(ncol=5)
    save(fig, out, "12_seed_stability.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--phase0-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    architecture(out)
    score_map(args.evaluation_dir, out)
    forest(args.evaluation_dir, out)
    atlas(args.run_dir, args.bundle_dir, args.sidecar_dir, args.evaluation_dir, out)
    neighbor_panels(args.evaluation_dir, out)
    raw_residual(args.evaluation_dir, out)
    gate_heatmap(args.evaluation_dir, out)
    actuation(args.evaluation_dir, out)
    linear_graph(args.evaluation_dir, out)
    controls(args.evaluation_dir, out)
    lambda_paths(args.phase0_dir, out)
    stability(args.evaluation_dir, out)
    print(f"wrote 12 figures to {out}")


if __name__ == "__main__":
    main()
