#!/usr/bin/env python3
"""Retrospective screen for token-tail signals distributed through model depth.

This script is deliberately a discovery tool, not a promoted benchmark.  It
constructs every candidate without correctness labels, then opens labels only
to rank candidates and combinations.  The intended use is to find a compact
metric registry that can subsequently be frozen and rerun by the formal
prepare -> fit -> evaluate harness.

The central question is stricter than beating final-layer NLL: can deployed
U-PCR beat both the broad per-cell best-single-view oracle and the local
TriLens grouped-CV approximation on the 13 eligible cells?
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_experiment import (  # noqa: E402
    CELLS,
    DEFAULT_CACHE,
    ORIGINAL_LLAMA_CELLS,
    PRIMARY_CELLS,
    load_evaluation_labels,
    load_feature_matrix,
)
from spectral_utils.paper_benchmark_suite import DEPLOYED_FIT, standardize  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


RESULTS = REPO / "results" / "whitebox_depth_tail_screen_v1"
BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
CONSENSUS_RESULTS = REPO / "results" / "whitebox_depth_consensus_v1"

METRICS = (
    "entropy",
    "target_nll",
    "top1_surprisal",
    "target_gap",
    "target_excess_over_entropy",
    "entropy_excess_over_top1",
    "kl_to_final",
)
REDUCERS = ("mean", "q90", "max", "std", "cvar80")
MODES = (
    "flat_top3_p2",
    "flat_top5_p1",
    "flat_top8_p1",
    "spread4_p1",
    "spread8_p1",
    "organic_all_p1",
    "organic_top8_p1",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def metric_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def rank_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranked = np.column_stack([
        rankdata(values[:, index], method="average") / (len(values) + 1.0)
        for index in range(values.shape[1])
    ])
    scales = ranked.std(axis=0)
    if np.any(scales < 1e-12):
        raise ValueError("rank transform received a degenerate feature")
    return (ranked - ranked.mean(axis=0)) / scales


def oriented_rank_views(
    values: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, keep, _means, _scales = standardize(values)
    X = rank_z(X)
    anchor_rank = rankdata(np.asarray(anchor, dtype=float), method="average")
    anchor_rank /= len(anchor_rank) + 1.0
    correlations = np.asarray([
        np.corrcoef(X[:, index], anchor_rank)[0, 1]
        for index in range(X.shape[1])
    ])
    if not np.isfinite(correlations).all():
        raise ValueError("non-finite anchor correlations")
    X *= np.where(correlations < 0.0, -1.0, 1.0)[None, :]
    return X, np.abs(correlations), keep


def weighted_score(X: np.ndarray, reliability: np.ndarray, selected: np.ndarray, power: float) -> np.ndarray:
    weights = np.maximum(reliability[selected], 1e-8) ** float(power)
    weights /= weights.sum()
    return X[:, selected] @ weights


def component_scores(
    values: np.ndarray,
    anchor: np.ndarray,
    *,
    n_layers: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return flat, depth-spread, and layer-organic label-free summaries."""

    X, reliability, keep = oriented_rank_views(values, anchor)
    original_layers = keep % int(n_layers)
    outputs: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}

    for k, power, name in (
        (3, 2.0, "flat_top3_p2"),
        (5, 1.0, "flat_top5_p1"),
        (8, 1.0, "flat_top8_p1"),
    ):
        selected = np.argsort(-reliability, kind="stable")[: min(k, X.shape[1])]
        outputs[name] = weighted_score(X, reliability, selected, power)
        diagnostics[name] = {
            "selected_original_indices": keep[selected].tolist(),
            "selected_layers": original_layers[selected].tolist(),
            "selected_reliability": reliability[selected].tolist(),
        }

    layer_bands = np.array_split(np.arange(n_layers), 4)
    for per_band, name in ((1, "spread4_p1"), (2, "spread8_p1")):
        selected_parts = []
        for band in layer_bands:
            eligible = np.flatnonzero(np.isin(original_layers, band))
            order = eligible[np.argsort(-reliability[eligible], kind="stable")]
            selected_parts.extend(order[:per_band].tolist())
        selected = np.asarray(selected_parts, dtype=int)
        outputs[name] = weighted_score(X, reliability, selected, 1.0)
        diagnostics[name] = {
            "selected_original_indices": keep[selected].tolist(),
            "selected_layers": original_layers[selected].tolist(),
            "selected_reliability": reliability[selected].tolist(),
            "depth_bands": [band.tolist() for band in layer_bands],
        }

    layer_columns = []
    layer_names = []
    for layer in range(n_layers):
        selected = np.flatnonzero(original_layers == layer)
        if selected.size:
            layer_columns.append(X[:, selected].mean(axis=1))
            layer_names.append(layer)
    layer_matrix = rank_z(np.column_stack(layer_columns))
    anchor_rank = rankdata(np.asarray(anchor, dtype=float), method="average")
    anchor_rank /= len(anchor_rank) + 1.0
    layer_corr = np.asarray([
        np.corrcoef(layer_matrix[:, index], anchor_rank)[0, 1]
        for index in range(layer_matrix.shape[1])
    ])
    layer_matrix *= np.where(layer_corr < 0.0, -1.0, 1.0)[None, :]
    layer_rel = np.abs(layer_corr)
    for k, name in ((len(layer_names), "organic_all_p1"), (8, "organic_top8_p1")):
        selected = np.argsort(-layer_rel, kind="stable")[: min(k, len(layer_names))]
        outputs[name] = weighted_score(layer_matrix, layer_rel, selected, 1.0)
        diagnostics[name] = {
            "selected_layers": [int(layer_names[index]) for index in selected],
            "selected_reliability": layer_rel[selected].tolist(),
            "within_layer_groups": "equal mean of anchor-oriented attention/MLP/residual views",
        }

    return outputs, diagnostics


def reduced_value(array: np.ndarray, reducer: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if reducer == "mean":
        return np.mean(array, axis=-1)
    if reducer == "q90":
        return np.quantile(array, 0.9, axis=-1).astype(np.float32)
    if reducer == "max":
        return np.max(array, axis=-1)
    if reducer == "std":
        return np.std(array, axis=-1)
    if reducer == "cvar80":
        threshold = np.quantile(array, 0.8, axis=-1, keepdims=True)
        mask = array >= threshold
        return (np.sum(np.where(mask, array, 0.0), axis=-1) / np.maximum(mask.sum(axis=-1), 1)).astype(np.float32)
    raise KeyError(reducer)


def extract_reduced_views(records: Sequence[Mapping[str, Any]], n_layers: int) -> dict[str, np.ndarray]:
    n_samples = len(records)
    n_views = 3 * int(n_layers)
    matrices = {
        f"{metric}__{reducer}": np.empty((n_samples, n_views), dtype=np.float32)
        for metric in METRICS for reducer in REDUCERS
    }
    for row_index, record in enumerate(records):
        entropy = np.asarray(record["lens_H"], dtype=np.float32)
        target_nll = -np.asarray(record["lens_logp_tgt"], dtype=np.float32)
        top1_surprisal = -np.asarray(record["lens_logp_top1"], dtype=np.float32)
        quantities = {
            "entropy": entropy,
            "target_nll": target_nll,
            "top1_surprisal": top1_surprisal,
            "target_gap": target_nll - top1_surprisal,
            "target_excess_over_entropy": target_nll - entropy,
            "entropy_excess_over_top1": entropy - top1_surprisal,
            "kl_to_final": np.asarray(record["lens_kl_final"], dtype=np.float32),
        }
        for metric, array in quantities.items():
            for reducer in REDUCERS:
                matrices[f"{metric}__{reducer}"][row_index] = reduced_value(array, reducer).reshape(-1)
        if (row_index + 1) % 2000 == 0:
            print(f"  reduced {row_index + 1}/{n_samples}", flush=True)
    return matrices


def fit_upcr(values: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    X, keep, _means, _scales = standardize(values)
    fitted = upcr_fit(X.T, **DEPLOYED_FIT)
    score = fitted.w @ X.T
    if np.corrcoef(score, anchor)[0, 1] < 0.0:
        score = -score
    return score, {
        "weights": fitted.w.tolist(),
        "keep": fitted.keep.tolist(),
        "used_simple_average": bool(fitted.used_simple_average),
        "n_kept": int(fitted.keep.sum()),
        "kept_standardized_columns": keep.tolist(),
    }


def prepare(cache: Path, results: Path) -> None:
    prepared_dir = results / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    all_diagnostics = {}
    for cell, spec in CELLS.items():
        if cell not in PRIMARY_CELLS:
            continue
        print(f"[tail-prepare] {cell}", flush=True)
        with np.load(BASE_RESULTS / "prepared" / f"{cell}__rows.npz", allow_pickle=False) as rows:
            row_ids = tuple(str(value) for value in rows["row_ids"].tolist())
            problem_ids = np.asarray(rows["problem_ids"], dtype="U")
        sidecar = load_pickle(cache / spec["sidecar"])
        records = [sidecar[row_id] for row_id in row_ids]
        n_layers = int(np.asarray(records[0]["lens_H"]).shape[1])
        base = load_feature_matrix(CONSENSUS_RESULTS / "prepared" / f"{cell}__depth_consensus.npz")
        reduced = extract_reduced_views(records, n_layers)
        candidate_names = []
        candidate_columns = []
        diagnostics = {}
        for family, values in reduced.items():
            outputs, family_diag = component_scores(values, base.risk_anchor, n_layers=n_layers)
            for mode in MODES:
                candidate_names.append(f"{family}__{mode}")
                candidate_columns.append(outputs[mode])
            diagnostics[family] = family_diag
        candidates = rank_z(np.column_stack(candidate_columns))
        # Orient every component once more after its final rank transform.
        correlations = np.asarray([
            spearmanr(candidates[:, index], base.risk_anchor).statistic
            for index in range(candidates.shape[1])
        ])
        candidates *= np.where(correlations < 0.0, -1.0, 1.0)[None, :]
        np.savez_compressed(
            prepared_dir / f"{cell}.npz",
            row_ids=np.asarray(row_ids, dtype="U"),
            problem_ids=problem_ids,
            anchor=np.asarray(base.risk_anchor, dtype=np.float32),
            base_components=np.asarray(base.values, dtype=np.float32),
            base_names=np.asarray(base.feature_names, dtype="U"),
            candidates=np.asarray(candidates, dtype=np.float32),
            candidate_names=np.asarray(candidate_names, dtype="U"),
        )
        all_diagnostics[cell] = {
            "n_samples": len(row_ids),
            "n_layers": n_layers,
            "n_candidates": len(candidate_names),
            "families": diagnostics,
        }
        del sidecar, records, reduced, candidates
    (results / "component_diagnostics.json").write_text(
        json.dumps(_jsonable(all_diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate(cache: Path, results: Path) -> None:
    cell_data = {}
    candidate_names: tuple[str, ...] | None = None
    for cell, spec in CELLS.items():
        if cell not in PRIMARY_CELLS:
            continue
        print(f"[tail-labels] {cell}", flush=True)
        with np.load(results / "prepared" / f"{cell}.npz", allow_pickle=False) as bundle:
            names = tuple(str(value) for value in bundle["candidate_names"].tolist())
            if candidate_names is None:
                candidate_names = names
            elif candidate_names != names:
                raise RuntimeError("candidate registry differs across cells")
            row_ids = tuple(str(value) for value in bundle["row_ids"].tolist())
            raw = load_pickle(cache / spec["raw"])
            y = load_evaluation_labels(raw, row_ids)
            cell_data[cell] = {
                "y": y,
                "anchor": bundle["anchor"].copy(),
                "base": bundle["base_components"].copy(),
                "candidates": bundle["candidates"].copy(),
            }
            del raw
    assert candidate_names is not None

    discovery = tuple(cell for cell in ORIGINAL_LLAMA_CELLS if cell in PRIMARY_CELLS)
    confirmation = tuple(cell for cell in PRIMARY_CELLS if cell not in discovery)
    comparator_rows = list(csv.DictReader((CONSENSUS_RESULTS / "headline_summary.csv").open()))
    comparators = {row["method"]: float(row["macro_auroc"]) for row in comparator_rows}

    individual_rows = []
    addition_rows = []
    per_candidate_cell: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for index, name in enumerate(candidate_names):
        for cell in PRIMARY_CELLS:
            data = cell_data[cell]
            y = data["y"]
            candidate = data["candidates"][:, index]
            single_auc, single_ap = metric_pair(y, candidate)
            upcr, diag = fit_upcr(np.column_stack([data["base"], candidate]), data["anchor"])
            auc, ap = metric_pair(y, upcr)
            per_candidate_cell[index][cell] = {
                "single_auroc": single_auc,
                "single_auprc": single_ap,
                "upcr_auroc": auc,
                "upcr_auprc": ap,
                "upcr_n_kept": diag["n_kept"],
                "upcr_used_simple_average": diag["used_simple_average"],
            }
        values = per_candidate_cell[index]
        individual_rows.append({
            "candidate": name,
            "discovery_macro_auroc": np.mean([values[cell]["single_auroc"] for cell in discovery]),
            "confirmation_macro_auroc": np.mean([values[cell]["single_auroc"] for cell in confirmation]),
            "all_macro_auroc": np.mean([values[cell]["single_auroc"] for cell in PRIMARY_CELLS]),
            "all_macro_auprc": np.mean([values[cell]["single_auprc"] for cell in PRIMARY_CELLS]),
        })
        addition_rows.append({
            "candidate": name,
            "discovery_macro_auroc": np.mean([values[cell]["upcr_auroc"] for cell in discovery]),
            "confirmation_macro_auroc": np.mean([values[cell]["upcr_auroc"] for cell in confirmation]),
            "all_macro_auroc": np.mean([values[cell]["upcr_auroc"] for cell in PRIMARY_CELLS]),
            "all_macro_auprc": np.mean([values[cell]["upcr_auprc"] for cell in PRIMARY_CELLS]),
            "mean_n_kept": np.mean([values[cell]["upcr_n_kept"] for cell in PRIMARY_CELLS]),
            "fallback_cells": sum(values[cell]["upcr_used_simple_average"] for cell in PRIMARY_CELLS),
        })
        if (index + 1) % 35 == 0:
            print(f"  screened additions {index + 1}/{len(candidate_names)}", flush=True)

    individual_rows.sort(key=lambda row: row["discovery_macro_auroc"], reverse=True)
    addition_rows.sort(key=lambda row: row["discovery_macro_auroc"], reverse=True)
    name_to_index = {name: index for index, name in enumerate(candidate_names)}
    top_names = [row["candidate"] for row in addition_rows[:20]]

    combination_rows = []
    combination_scores: dict[str, dict[str, np.ndarray]] = {}
    specs: list[tuple[str, ...]] = [(name,) for name in top_names]
    specs.extend(itertools.combinations(top_names[:14], 2))
    specs.extend(itertools.combinations(top_names[:9], 3))
    # Also test tail-only subsets: these force the spectral estimator to work
    # without the previously discovered four components.
    specs.extend(itertools.combinations(top_names[:9], 5))
    seen = set()
    for spec_index, names in enumerate(specs):
        key = " + ".join(names)
        if key in seen:
            continue
        seen.add(key)
        per_cell = {}
        saved_scores = {}
        for cell in PRIMARY_CELLS:
            data = cell_data[cell]
            columns = data["candidates"][:, [name_to_index[name] for name in names]]
            tail_only = len(names) == 5
            matrix = columns if tail_only else np.column_stack([data["base"], columns])
            score, diag = fit_upcr(matrix, data["anchor"])
            auc, ap = metric_pair(data["y"], score)
            per_cell[cell] = (auc, ap, diag)
            saved_scores[cell] = score
        row = {
            "combination": key,
            "n_added": len(names),
            "tail_only": len(names) == 5,
            "discovery_macro_auroc": np.mean([per_cell[cell][0] for cell in discovery]),
            "confirmation_macro_auroc": np.mean([per_cell[cell][0] for cell in confirmation]),
            "all_macro_auroc": np.mean([per_cell[cell][0] for cell in PRIMARY_CELLS]),
            "all_macro_auprc": np.mean([per_cell[cell][1] for cell in PRIMARY_CELLS]),
            "mean_n_kept": np.mean([per_cell[cell][2]["n_kept"] for cell in PRIMARY_CELLS]),
            "fallback_cells": sum(per_cell[cell][2]["used_simple_average"] for cell in PRIMARY_CELLS),
        }
        combination_rows.append(row)
        combination_scores[key] = saved_scores
        if (spec_index + 1) % 50 == 0:
            print(f"  screened combinations {spec_index + 1}/{len(specs)}", flush=True)
    combination_rows.sort(key=lambda row: row["discovery_macro_auroc"], reverse=True)

    # The discovery-selected winner is fixed before inspecting the confirmation
    # column in the summary below.  The data have been seen in prior work, so this
    # is organizational discipline rather than a claim of independent evidence.
    winner = combination_rows[0]
    winner_scores = combination_scores[winner["combination"]]
    winner_names = tuple(winner["combination"].split(" + "))
    per_cell_winner = []
    winner_component_oracle_aucs = []
    for cell in PRIMARY_CELLS:
        data = cell_data[cell]
        y = data["y"]
        component_indices = [name_to_index[name] for name in winner_names]
        component_matrix = np.column_stack([data["base"], data["candidates"][:, component_indices]])
        component_oracle = float(max(roc_auc_score(y, component_matrix[:, j]) for j in range(component_matrix.shape[1])))
        auc, ap = metric_pair(y, winner_scores[cell])
        winner_component_oracle_aucs.append(component_oracle)
        per_cell_winner.append({
            "cell": cell,
            "winner_auroc": auc,
            "winner_auprc": ap,
            "winner_component_oracle_auroc": component_oracle,
        })

    results.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("individual_components.csv", individual_rows),
        ("single_additions.csv", addition_rows),
        ("combinations.csv", combination_rows),
        ("winner_per_cell.csv", per_cell_winner),
    ):
        with (results / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "analysis_role": "retrospective discovery screen; labels used only for global candidate ranking/evaluation",
        "discovery_cells": discovery,
        "confirmation_cells": confirmation,
        "winner": winner,
        "winner_names": winner_names,
        "winner_component_oracle_macro_auroc": float(np.mean(winner_component_oracle_aucs)),
        "broad_existing_best_single_macro_auroc": comparators["best_single_layer"],
        "trilens_grouped_probe_macro_auroc": comparators["trilens_supervised_lr"],
        "beats_broad_existing_best_single": bool(winner["all_macro_auroc"] > comparators["best_single_layer"]),
        "beats_trilens": bool(winner["all_macro_auroc"] > comparators["trilens_supervised_lr"]),
        "beats_own_component_oracle": bool(winner["all_macro_auroc"] > np.mean(winner_component_oracle_aucs)),
        "note": "Confirmation cells were already observed in earlier experiments and are not independent confirmation.",
    }
    (results / "screen_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        prepare(args.cache, args.results)
    if args.phase in ("evaluate", "all"):
        evaluate(args.cache, args.results)


if __name__ == "__main__":
    main()
