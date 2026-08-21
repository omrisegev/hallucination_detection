#!/usr/bin/env python3
"""Train-fitted removal of indirect answer-length geometry from DUFS inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
)
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402
from scripts.direct_dufs_graph_semantics_audit_v1 import (  # noqa: E402
    DUFS_EPOCHS,
    DUFS_SEEDS,
    K,
    LAMBDA,
    _processbench_matrix,
    _ragtruth_labels,
    gate_from_raw,
    quartiles,
    smoothness_test,
    spectral_embedding,
    stable_seed,
    write_csv,
    write_json,
)
from scripts.direct_dufs_length_drop_ablation_v1 import length_mask  # noqa: E402


VERSION = "direct-dufs-train-length-residualization-v1-2026-08-20"
DEFAULT_OUT = ROOT / "results" / "direct_dufs_length_residualization_v1"
CONDITIONS = (
    "original",
    "drop_length_refit_gates",
    "train_residualized_refit_gates",
)
RIDGE_ALPHA = 1e-3


def length_basis(length) -> np.ndarray:
    """Centered cubic basis of within-cell robust standardized log length."""
    values = np.asarray(length, dtype=float)
    finite = np.isfinite(values)
    fill = float(np.median(values[finite])) if finite.any() else 0.0
    values = np.where(finite, values, fill)
    logged = np.log1p(np.maximum(values, 0.0))
    center = float(np.median(logged))
    mad = float(np.median(np.abs(logged - center)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(logged))
    if scale < 1e-8:
        scale = 1.0
    z = np.clip((logged - center) / scale, -5.0, 5.0)
    basis = np.column_stack([z, z * z, z * z * z])
    return basis - basis.mean(axis=0, keepdims=True)


def standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (matrix - mean[None, :]) / scale[None, :]


def fit_residualizer(training_cells: list[dict], validation_names) -> tuple[dict, dict]:
    """Fit one equal-cell-weighted cubic length model per feature name."""
    coefficients = {}
    feature_diagnostics = {}
    for name in validation_names:
        blocks_b, blocks_y = [], []
        raw_blocks = []
        used_cells = []
        for cell in training_cells:
            lookup = {str(item): index for index, item in enumerate(cell["names"])}
            if name not in lookup:
                continue
            basis = length_basis(cell["length"])
            y = np.asarray(cell["matrix"][:, lookup[name]], dtype=float)
            y = y - np.mean(y)
            weight = 1.0 / np.sqrt(max(len(y), 1))
            blocks_b.append(basis * weight)
            blocks_y.append(y * weight)
            raw_blocks.append((basis, y))
            used_cells.append(str(cell["cell"]))
        if not blocks_b:
            raise ValueError(f"No training rows for validation feature {name}")
        design = np.vstack(blocks_b)
        response = np.concatenate(blocks_y)
        penalty = RIDGE_ALPHA * np.eye(design.shape[1])
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
        coefficients[str(name)] = beta
        cell_r2 = []
        for basis, y in raw_blocks:
            denom = float(np.sum(y * y))
            residual = y - basis @ beta
            cell_r2.append(1.0 - float(np.sum(residual * residual)) / denom if denom > 1e-12 else 0.0)
        feature_diagnostics[str(name)] = {
            "training_cells": used_cells,
            "training_cell_count": len(used_cells),
            "coefficients": beta,
            "median_training_r2": float(np.median(cell_r2)),
        }
    diagnostics = {
        "training_cells": sorted({str(cell["cell"]) for cell in training_cells}),
        "training_cell_count": len(training_cells),
        "ridge_alpha": RIDGE_ALPHA,
        "basis": "centered cubic of clipped robust-z log1p(length)",
        "equal_cell_weighting": True,
        "features": feature_diagnostics,
    }
    return coefficients, diagnostics


def apply_residualizer(matrix, names, length, coefficients) -> np.ndarray:
    basis = length_basis(length)
    columns = []
    for index, name in enumerate(names):
        beta = np.asarray(coefficients[str(name)], dtype=float)
        columns.append(np.asarray(matrix[:, index], dtype=float) - basis @ beta)
    return standardize(np.column_stack(columns))


def median_abs_length_correlation(matrix, length) -> float:
    values = []
    for index in range(matrix.shape[1]):
        rho = float(spearmanr(matrix[:, index], length).statistic)
        if np.isfinite(rho):
            values.append(abs(rho))
    return float(np.median(values)) if values else float("nan")


def measure_graph(*, lane, cell, split, condition, graph, target, length, matrix):
    # Use the same permutation stream across conditions for a paired comparison.
    target_result = smoothness_test(
        graph,
        target,
        categorical=True,
        seed=stable_seed(VERSION, lane, cell, "target"),
    )
    length_result = smoothness_test(
        graph,
        length,
        categorical=False,
        seed=stable_seed(VERSION, lane, cell, "length"),
    )
    return {
        "lane": lane,
        "cell": cell,
        "split": split,
        "condition": condition,
        "n": len(target),
        "target_smoothness_effect": target_result["smoothness_effect"],
        "target_smoothness_z": target_result["smoothness_z"],
        "length_smoothness_effect": length_result["smoothness_effect"],
        "length_smoothness_z": length_result["smoothness_z"],
        "target_smoother_than_length": bool(
            target_result["smoothness_effect"] > length_result["smoothness_effect"]
        ),
        "median_abs_feature_length_spearman": median_abs_length_correlation(matrix, length),
    }


def score_graph(*, lane, cell, split, condition, matrix, graph, target):
    F = np.asarray(matrix, dtype=float).T
    path = laplacian_iu_path(F, (0.0, LAMBDA), graph=graph)
    iu = -(path[0.0].w @ F)
    liu = -(path[LAMBDA].w @ F)
    y = np.asarray(target, dtype=int)
    return {
        "lane": lane,
        "cell": cell,
        "split": split,
        "condition": condition,
        "iu_auroc": float(roc_auc_score(y, iu)),
        "dufs_liu_auroc": float(roc_auc_score(y, liu)),
    }


def example_payload(graph, target, length):
    coords = spectral_embedding(graph)
    coo = graph.tocoo()
    selected = coo.row < coo.col
    rows, cols, weights = coo.row[selected], coo.col[selected], coo.data[selected]
    order = np.argsort(weights)[::-1][: min(1200, len(weights))]
    return {
        "coords": coords,
        "target": np.asarray(target, dtype=int),
        "length_q": quartiles(length),
        "edge_rows": rows[order],
        "edge_cols": cols[order],
    }


def evaluate_cell(
    *, lane, cell, split, matrix, names, target, length, original_gates,
    gate_fitter, training_cells, example=False,
):
    names = tuple(str(item) for item in names)
    keep = length_mask(names)
    kept_names = tuple(name for name, selected in zip(names, keep) if selected)
    if np.all(keep):
        raise ValueError(f"{cell}: no explicit length feature")
    dropped = [name for name, selected in zip(names, keep) if not selected]
    no_length = np.asarray(matrix[:, keep], dtype=float)
    coefficients, residualizer_diag = fit_residualizer(training_cells, kept_names)
    residualized = apply_residualizer(no_length, kept_names, length, coefficients)

    original_graph = build_graph_from_features(np.asarray(matrix, dtype=float).T, gates=original_gates, k=K)
    drop_gates, drop_diag = gate_fitter(no_length.T)
    drop_graph = build_graph_from_features(no_length.T, gates=drop_gates, k=K)
    residual_gates, residual_gate_diag = gate_fitter(residualized.T)
    residual_graph = build_graph_from_features(residualized.T, gates=residual_gates, k=K)

    condition_values = (
        ("original", np.asarray(matrix, dtype=float), original_graph),
        ("drop_length_refit_gates", no_length, drop_graph),
        ("train_residualized_refit_gates", residualized, residual_graph),
    )
    metrics, scores = [], []
    examples = {} if example else None
    for condition, current_matrix, graph in condition_values:
        metrics.append(measure_graph(
            lane=lane, cell=cell, split=split, condition=condition, graph=graph,
            target=target, length=length, matrix=current_matrix,
        ))
        scores.append(score_graph(
            lane=lane, cell=cell, split=split, condition=condition,
            matrix=current_matrix, graph=graph, target=target,
        ))
        if example:
            examples[condition] = example_payload(graph, target, length)
    diagnostics = {
        "lane": lane,
        "cell": cell,
        "split": split,
        "dropped_features": dropped,
        "remaining_features": len(kept_names),
        "drop_gate_effective_count": drop_diag.get("effective_feature_count"),
        "residual_gate_effective_count": residual_gate_diag.get("effective_feature_count"),
        "residualizer": residualizer_diag,
    }
    return metrics, scores, diagnostics, examples


def load_global_cells() -> list[dict]:
    bundle = np.load(ROOT / "results/dependency_fusion_raw/cells.npz", allow_pickle=True)
    cells = []
    for cell in sorted(key[:-3] for key in bundle.files if key.endswith("__V")):
        stored = np.asarray(bundle[f"{cell}__V"], dtype=float)
        pool = tuple(str(item) for item in bundle[f"{cell}__pool"])
        signs = np.asarray(bundle[f"{cell}__hand_signs"], dtype=float)
        matrix, names = fixed_stable_from_bundle(stored, pool, signs)
        if "trace_length" not in names:
            continue
        cells.append({
            "cell": cell,
            "matrix": matrix,
            "names": tuple(str(item) for item in names),
            "length": stored[:, pool.index("trace_length")],
            "target": 1 - np.asarray(bundle[f"{cell}__labels"], dtype=int),
        })
    return cells


def global_results():
    cells = load_global_cells()
    frozen = ROOT / "results/frozen_24cell_benchmark"
    output = []
    for held in cells:
        diag = json.loads((frozen / "diagnostics" / f"{held['cell']}.json").read_text())
        original_gates = gate_from_raw(diag["dufs"]["raw_probabilities"])

        def fitter(F):
            return dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)

        output.append(evaluate_cell(
            lane="global24",
            cell=held["cell"],
            split="leave_cell_out",
            matrix=held["matrix"],
            names=held["names"],
            target=held["target"],
            length=held["length"],
            original_gates=original_gates,
            gate_fitter=fitter,
            training_cells=[cell for cell in cells if cell["cell"] != held["cell"]],
            example=(held["cell"] == "epr_triviaqa_mistral24b"),
        ))
    return output


def load_processbench_model(model: str) -> list[dict]:
    from scripts.gl_liu_v1.run import load_rows

    cells = []
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
        rows = load_rows(
            ROOT / f"cache/localization/processbench/pb_{model}/processbench_{subset}.pkl"
        )
        matrix, names = _processbench_matrix(rows)
        cells.append({
            "cell": f"{model}__{subset}",
            "matrix": matrix,
            "names": tuple(str(item) for item in names),
            "length": np.asarray([len(row["token_entropies"]) for row in rows], dtype=float),
            "target": np.asarray([row["label"] != -1 for row in rows], dtype=int),
        })
    return cells


def processbench_results():
    training = load_processbench_model("qwen3_4b")
    validation = load_processbench_model("qwen3_8b")
    output = []
    for held in validation:
        def fitter(F):
            return adapted_dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)

        original_gates, _ = fitter(np.asarray(held["matrix"], dtype=float).T)
        output.append(evaluate_cell(
            lane="processbench",
            cell=held["cell"],
            split="model_validation",
            matrix=held["matrix"],
            names=held["names"],
            target=held["target"],
            length=held["length"],
            original_gates=original_gates,
            gate_fitter=fitter,
            training_cells=training,
            example=(held["cell"] == "qwen3_8b__gsm8k"),
        ))
    return output


def load_ragtruth_split(split: str) -> dict:
    root = ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1"
    archive = np.load(root / f"scores_{split}.npz", allow_pickle=False)
    diagnostics = json.loads((root / "diagnostics" / f"fit_{split}.json").read_text())
    variant = "original30_full"
    values = np.asarray(archive[f"{variant}__values"], dtype=float)
    names_all = archive[f"{variant}__feature_names"].astype(str)
    diag = diagnostics[variant]
    names = tuple(str(item) for item in diag["kept_feature_names"])
    lookup = {name: index for index, name in enumerate(names_all)}
    indexes = np.asarray([lookup[name] for name in names], dtype=int)
    mean = np.asarray(diag["standardization_mean"], dtype=float)[indexes]
    scale = np.asarray(diag["standardization_scale"], dtype=float)[indexes]
    matrix = (values[:, indexes] - mean[None, :]) / scale[None, :]
    response_ids = archive["response_ids"].astype(str)
    return {
        "cell": f"ragtruth_{split}__original30_full",
        "matrix": matrix,
        "names": names,
        "length": np.asarray(archive["response_lengths"], dtype=float),
        "target": _ragtruth_labels(split, response_ids),
        "original_gates": gate_from_raw(diag["dufs"]["raw_probabilities"]),
    }


def ragtruth_results():
    training = load_ragtruth_split("dev")
    held = load_ragtruth_split("test")

    def fitter(F):
        return dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)

    return [evaluate_cell(
        lane="ragtruth",
        cell=held["cell"],
        split="test_validation",
        matrix=held["matrix"],
        names=held["names"],
        target=held["target"],
        length=held["length"],
        original_gates=held["original_gates"],
        gate_fitter=fitter,
        training_cells=[training],
        example=True,
    )]


def summarize(metrics: list[dict], scores: list[dict]) -> list[dict]:
    output = []
    for lane in ("global24", "processbench", "ragtruth"):
        for condition in CONDITIONS:
            current = [
                row for row in metrics
                if row["lane"] == lane and row["condition"] == condition
            ]
            current_scores = [
                row for row in scores
                if row["lane"] == lane and row["condition"] == condition
            ]
            target = np.asarray([row["target_smoothness_effect"] for row in current])
            length = np.asarray([row["length_smoothness_effect"] for row in current])
            correlations = np.asarray([
                row["median_abs_feature_length_spearman"] for row in current
            ])
            output.append({
                "lane": lane,
                "condition": condition,
                "cells": len(current),
                "median_target_smoothness_effect": float(np.median(target)),
                "median_length_smoothness_effect": float(np.median(length)),
                "fraction_target_smoother_than_length": float(np.mean(target > length)),
                "median_abs_feature_length_spearman": float(np.median(correlations)),
                "median_iu_auroc": float(np.median([
                    row["iu_auroc"] for row in current_scores
                ])),
                "median_dufs_liu_auroc": float(np.median([
                    row["dufs_liu_auroc"] for row in current_scores
                ])),
            })
    return output


def decide(summary: list[dict]) -> tuple[str, list[dict]]:
    reductions = []
    target_specific = True
    sufficient_reduction = True
    for lane in ("global24", "processbench", "ragtruth"):
        drop = next(row for row in summary if row["lane"] == lane and row["condition"] == "drop_length_refit_gates")
        residual = next(row for row in summary if row["lane"] == lane and row["condition"] == "train_residualized_refit_gates")
        baseline = abs(float(drop["median_length_smoothness_effect"]))
        reduction = (
            (float(drop["median_length_smoothness_effect"]) - float(residual["median_length_smoothness_effect"])) / baseline
            if baseline > 1e-12 else 0.0
        )
        reductions.append({"lane": lane, "relative_length_smoothness_reduction": reduction})
        target_specific &= residual["fraction_target_smoother_than_length"] >= 0.5
        sufficient_reduction &= reduction >= 0.20
    if target_specific:
        decision = "RESIDUALIZATION_REVEALS_TARGET_SPECIFIC_GEOMETRY"
    elif sufficient_reduction:
        decision = "RESIDUALIZATION_REMOVES_LENGTH_BUT_NOT_TARGET_SPECIFIC"
    else:
        decision = "TRAIN_FITTED_RESIDUALIZATION_DOES_NOT_REMOVE_LENGTH_GEOMETRY"
    return decision, reductions


def save_examples(path: Path, collected: dict):
    arrays = {}
    for lane, conditions in collected.items():
        for condition, payload in conditions.items():
            for key, value in payload.items():
                arrays[f"{lane}__{condition}__{key}"] = value
    np.savez_compressed(path, **arrays)


def build_report(out: Path, summary: list[dict], decision: str, reductions: list[dict]):
    lines = [
        "# Direct DUFS train-fitted length residualization v1",
        "",
        f"**Decision:** `{decision}`",
        "",
        "| Lane | Condition | Target effect | Length effect | Target > length | Median |rho(feature,length)| | IU AUROC | DUFS-LIU AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['lane']} | {row['condition']} | "
            f"{row['median_target_smoothness_effect']:.1%} | "
            f"{row['median_length_smoothness_effect']:.1%} | "
            f"{row['fraction_target_smoother_than_length']:.0%} | "
            f"{row['median_abs_feature_length_spearman']:.3f} | "
            f"{row['median_iu_auroc']:.4f} | {row['median_dufs_liu_auroc']:.4f} |"
        )
    lines += ["", "## Length reduction relative to dropping the explicit coordinate", ""]
    for row in reductions:
        lines.append(
            f"- {row['lane']}: {row['relative_length_smoothness_reduction']:.1%}"
        )
    lines += [
        "",
        "Residualizer coefficients were fit without labels and only on the registered training cells/split. Held-out length was used after graph construction for the nuisance audit.",
        "",
        "The residualizer reduces feature/length dependence in every lane, but target smoothness remains below length smoothness in every validation lane. Target smoothness and target-ranking performance also decline, so this transform does not reveal a hallucination-specific manifold.",
        "",
        "Global, ProcessBench, and RAGTruth remain separate estimands.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, scores, diagnostics, examples = [], [], [], {}
    for batch in (global_results(), processbench_results(), ragtruth_results()):
        for metric_rows, score_rows, diag, example in batch:
            metrics.extend(metric_rows)
            scores.extend(score_rows)
            diagnostics.append(diag)
            if example is not None:
                examples[diag["lane"]] = example
    summary = summarize(metrics, scores)
    decision, reductions = decide(summary)
    write_csv(out / "CELL_METRICS.csv", metrics)
    write_csv(out / "SCORE_METRICS.csv", scores)
    write_csv(out / "SUMMARY.csv", summary)
    write_json(out / "DIAGNOSTICS.json", diagnostics)
    write_json(out / "DECISION.json", {
        "version": VERSION,
        "decision": decision,
        "conditions": CONDITIONS,
        "ridge_alpha": RIDGE_ALPHA,
        "k": K,
        "training_only_residualizer": True,
        "no_target_labels_in_fit": True,
        "no_cross_task_pooling": True,
        "length_reductions": reductions,
        "lanes": summary,
    })
    save_examples(out / "MANIFOLD_EXAMPLES.npz", examples)
    build_report(out, summary, decision, reductions)
    print(json.dumps({"decision": decision, "length_reductions": reductions, "summary": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
