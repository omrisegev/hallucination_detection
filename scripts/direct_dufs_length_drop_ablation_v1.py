#!/usr/bin/env python3
"""Remove explicit length coordinates and reconstruct/refit DUFS graphs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
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


VERSION = "direct-dufs-explicit-length-drop-v1-2026-08-20"
DEFAULT_OUT = ROOT / "results" / "direct_dufs_length_drop_ablation_v1"
CONDITIONS = ("original", "drop_length_fixed_gates", "drop_length_refit_gates")


def length_mask(names, base_names=None):
    if base_names is None:
        base_names = names
    return np.asarray([
        "length" not in str(name).lower() and "length" not in str(base).lower()
        for name, base in zip(names, base_names)
    ], dtype=bool)


def measure_graph(*, lane, cell, split, condition, graph, target, length):
    target_result = smoothness_test(
        graph, target, categorical=True,
        seed=stable_seed(VERSION, lane, cell, condition, "target"),
    )
    length_result = smoothness_test(
        graph, length, categorical=False,
        seed=stable_seed(VERSION, lane, cell, condition, "length"),
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
    }


def fit_score_row(*, lane, cell, split, condition, F, graph, target):
    path = laplacian_iu_path(F, (0.0, LAMBDA), graph=graph)
    iu = -(path[0.0].w @ F)
    liu = -(path[LAMBDA].w @ F)
    y = np.asarray(target, dtype=int)
    auc_iu = float(roc_auc_score(y, iu))
    auc_liu = float(roc_auc_score(y, liu))
    return {
        "lane": lane, "cell": cell, "split": split, "condition": condition,
        "iu_auroc": auc_iu, "dufs_liu_auroc": auc_liu,
        "liu_delta_auroc": auc_liu - auc_iu,
    }, liu


def handle_cell(
    *, lane, cell, split, matrix, names, target, length, original_gates,
    gate_fitter, frozen_liu=None, example=False,
):
    names = tuple(str(x) for x in names)
    keep = length_mask(names)
    dropped = [name for name, selected in zip(names, keep) if not selected]
    if not dropped:
        raise ValueError(f"{cell}: no explicit length feature")
    F = np.asarray(matrix, dtype=float).T
    F_drop = np.asarray(matrix[:, keep], dtype=float).T
    original_graph = build_graph_from_features(F, gates=original_gates, k=K)
    fixed_graph = build_graph_from_features(F_drop, gates=np.asarray(original_gates)[keep], k=K)
    refit_gates, refit_diag = gate_fitter(F_drop)
    refit_graph = build_graph_from_features(F_drop, gates=refit_gates, k=K)

    metrics = [
        measure_graph(lane=lane, cell=cell, split=split, condition="original", graph=original_graph, target=target, length=length),
        measure_graph(lane=lane, cell=cell, split=split, condition="drop_length_fixed_gates", graph=fixed_graph, target=target, length=length),
        measure_graph(lane=lane, cell=cell, split=split, condition="drop_length_refit_gates", graph=refit_graph, target=target, length=length),
    ]
    score_row, refit_liu = fit_score_row(
        lane=lane, cell=cell, split=split, condition="drop_length_refit_gates",
        F=F_drop, graph=refit_graph, target=target,
    )
    scores = [score_row]
    frozen_reproduction_max_abs = None
    if frozen_liu is not None:
        reproduced = -(laplacian_iu_path(F, (LAMBDA,), graph=original_graph)[LAMBDA].w @ F)
        frozen_reproduction_max_abs = float(np.max(np.abs(frozen_liu - reproduced)))
        correlation = float(np.corrcoef(frozen_liu, reproduced)[0, 1])
        # Dense eigensolver differences across SciPy builds can move the final
        # projected solve at ~1e-4 while leaving the graph/ranking unchanged.
        if frozen_reproduction_max_abs > 5e-4 or correlation < 0.999999:
            raise RuntimeError(
                f"{cell}: original frozen LIU reproduction failed "
                f"(max={frozen_reproduction_max_abs}, corr={correlation})"
            )

    example_arrays = None
    if example:
        example_arrays = {}
        for condition, graph in (("original", original_graph), ("drop_length_refit_gates", refit_graph)):
            coords = spectral_embedding(graph)
            coo = graph.tocoo()
            selected = coo.row < coo.col
            rows, cols, weights = coo.row[selected], coo.col[selected], coo.data[selected]
            order = np.argsort(weights)[::-1][:min(1200, len(weights))]
            example_arrays[condition] = {
                "coords": coords,
                "target": np.asarray(target, dtype=int),
                "length_q": quartiles(length),
                "edge_rows": rows[order],
                "edge_cols": cols[order],
            }
    diagnostics = {
        "lane": lane, "cell": cell, "dropped_features": dropped,
        "remaining_features": int(np.sum(keep)),
        "refit_gate_effective_count": refit_diag.get("effective_feature_count"),
        "frozen_reproduction_max_abs": frozen_reproduction_max_abs,
    }
    return metrics, scores, diagnostics, example_arrays


def global_cells():
    bundle = np.load(ROOT / "results/dependency_fusion_raw/cells.npz", allow_pickle=True)
    frozen = ROOT / "results/frozen_24cell_benchmark"
    output = []
    for cell in sorted(key[:-3] for key in bundle.files if key.endswith("__V")):
        stored = np.asarray(bundle[f"{cell}__V"], dtype=float)
        pool = tuple(str(x) for x in bundle[f"{cell}__pool"])
        signs = np.asarray(bundle[f"{cell}__hand_signs"], dtype=float)
        matrix, names = fixed_stable_from_bundle(stored, pool, signs)
        if "trace_length" not in names:
            continue
        length = stored[:, pool.index("trace_length")]
        target = 1 - np.asarray(bundle[f"{cell}__labels"], dtype=int)
        diag = json.loads((frozen / "diagnostics" / f"{cell}.json").read_text())
        original_gates = gate_from_raw(diag["dufs"]["raw_probabilities"])
        scores = np.load(frozen / "scores" / f"{cell}.npz", allow_pickle=False)
        def fitter(F):
            return dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
        output.append(handle_cell(
            lane="global24", cell=cell, split="validation", matrix=matrix,
            names=names, target=target, length=length, original_gates=original_gates,
            gate_fitter=fitter, frozen_liu=-np.asarray(scores["dufs_liu__lambda_0p1"]),
            example=(cell == "epr_triviaqa_mistral24b"),
        ))
    return output


def processbench_cells():
    from scripts.gl_liu_v1.run import load_rows
    frozen_root = ROOT / "results/processbench_latent_state_v1/label_free_scores"
    output = []
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
        cell = f"qwen3_8b__{subset}"
        rows = load_rows(ROOT / f"cache/localization/processbench/pb_qwen3_8b/processbench_{subset}.pkl")
        matrix, names = _processbench_matrix(rows)
        target = np.asarray([row["label"] != -1 for row in rows], dtype=int)
        length = np.asarray([len(row["token_entropies"]) for row in rows], dtype=float)
        def fitter(F):
            return adapted_dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
        original_gates, _ = fitter(matrix.T)
        frozen = np.load(frozen_root / f"{cell}.npz", allow_pickle=False)
        output.append(handle_cell(
            lane="processbench", cell=cell, split="model_validation", matrix=matrix,
            names=names, target=target, length=length, original_gates=original_gates,
            gate_fitter=fitter, frozen_liu=np.asarray(frozen["global_mixed_v2_dufs"]),
            example=(subset == "gsm8k"),
        ))
    return output


def ragtruth_cell():
    root = ROOT / "results/ragtruth_mixed_v2_evidence_aware_v1"
    archive = np.load(root / "scores_test.npz", allow_pickle=False)
    diagnostics = json.loads((root / "diagnostics/fit_test.json").read_text())
    variant = "original30_full"
    values = np.asarray(archive[f"{variant}__values"], dtype=float)
    names_all = archive[f"{variant}__feature_names"].astype(str)
    diag = diagnostics[variant]
    names = tuple(str(x) for x in diag["kept_feature_names"])
    lookup = {name: index for index, name in enumerate(names_all)}
    indexes = np.asarray([lookup[name] for name in names], dtype=int)
    mean = np.asarray(diag["standardization_mean"], dtype=float)[indexes]
    scale = np.asarray(diag["standardization_scale"], dtype=float)[indexes]
    matrix = (values[:, indexes] - mean[None, :]) / scale[None, :]
    original_gates = gate_from_raw(diag["dufs"]["raw_probabilities"])
    response_ids = archive["response_ids"].astype(str)
    target = _ragtruth_labels("test", response_ids)
    length = np.asarray(archive["response_lengths"], dtype=float)
    def fitter(F):
        return dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    return [handle_cell(
        lane="ragtruth", cell="ragtruth_test__original30_full", split="test_validation",
        matrix=matrix, names=names, target=target, length=length,
        original_gates=original_gates, gate_fitter=fitter,
        frozen_liu=np.asarray(archive["score__original30_full__dufs_liu"]),
        example=True,
    )]


def summarize(rows):
    output = []
    for lane in ("global24", "processbench", "ragtruth"):
        for condition in CONDITIONS:
            current = [row for row in rows if row["lane"] == lane and row["condition"] == condition]
            target = np.asarray([row["target_smoothness_effect"] for row in current], dtype=float)
            length = np.asarray([row["length_smoothness_effect"] for row in current], dtype=float)
            output.append({
                "lane": lane, "condition": condition, "cells": len(current),
                "median_target_smoothness_effect": float(np.median(target)),
                "median_length_smoothness_effect": float(np.median(length)),
                "fraction_target_smoother_than_length": float(np.mean(target > length)),
            })
    return output


def save_examples(path, collected):
    arrays = {}
    for lane, value in collected.items():
        for condition, payload in value.items():
            for key, item in payload.items():
                arrays[f"{lane}__{condition}__{key}"] = item
    np.savez_compressed(path, **arrays)


def report(out, summary, decision):
    lines = [
        "# Direct DUFS explicit-length-drop ablation v1", "",
        f"**Decision:** `{decision}`", "",
        "| Lane | Condition | Target effect | Held-out length effect | Target > length |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['lane']} | {row['condition']} | {row['median_target_smoothness_effect']:.1%} | "
            f"{row['median_length_smoothness_effect']:.1%} | {row['fraction_target_smoother_than_length']:.0%} |"
        )
    lines += [
        "", "The held-out length variable is never used to construct a no-length graph. Residual length smoothness therefore measures indirect length information in the remaining features.", "",
        "Global, ProcessBench, and RAGTruth remain separate estimands.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, scores, diagnostics, examples = [], [], [], {}
    for batch in (global_cells(), processbench_cells(), ragtruth_cell()):
        for metric_rows, score_rows, diag, example in batch:
            metrics.extend(metric_rows); scores.extend(score_rows); diagnostics.append(diag)
            if example is not None:
                examples[diag["lane"]] = example
    summary = summarize(metrics)
    refit = [row for row in summary if row["condition"] == "drop_length_refit_gates"]
    decision = (
        "EXPLICIT_LENGTH_WAS_PRIMARY_CHANNEL"
        if all(row["fraction_target_smoother_than_length"] >= 0.5 for row in refit)
        else "EXPLICIT_LENGTH_NOT_SOLE_NUISANCE_CHANNEL"
    )
    write_csv(out / "CELL_METRICS.csv", metrics)
    write_csv(out / "LIU_EFFECTS.csv", scores)
    write_csv(out / "SUMMARY.csv", summary)
    write_json(out / "DIAGNOSTICS.json", diagnostics)
    write_json(out / "DECISION.json", {"version": VERSION, "decision": decision, "lanes": summary, "no_cross_task_pooling": True})
    save_examples(out / "MANIFOLD_EXAMPLES.npz", examples)
    report(out, summary, decision)
    print(json.dumps({"decision": decision, "summary": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
