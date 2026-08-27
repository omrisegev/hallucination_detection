#!/usr/bin/env python3
"""Evaluation-only boundary for retrospective DEEM-B3 MoE variants."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deem_b3_moe_v1 import load_config, variant_lookup  # noqa: E402
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
)
from spectral_utils.residual_graph_deem_labels import (  # noqa: E402
    join_labels_by_id,
    load_label_sidecar,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, score) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if values.shape != target.shape or not np.isfinite(values).all() or len(np.unique(target)) != 2:
        raise ResidualGraphDeemError("invalid target/score pair")
    return {
        "auroc": float(roc_auc_score(target, values)),
        "auprc": float(average_precision_score(target, values)),
    }


def _fit_paths(run_dir: Path, variant: str, cell: str, seed: int) -> tuple[Path, Path]:
    stem = f"{variant}__seed{int(seed)}"
    directory = run_dir / "fits" / variant / cell
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def load_fit(run_dir: Path, variant: str, cell: str, seed: int) -> tuple[dict, dict]:
    array_path, metadata_path = _fit_paths(run_dir, variant, cell, seed)
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing MoE fit: {variant}/{cell}/seed{seed}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_content = metadata.get("content_sha256")
    unhashed = dict(metadata)
    unhashed.pop("content_sha256", None)
    if (
        metadata.get("status") != "complete"
        or not metadata.get("health", {}).get("healthy")
        or canonical_sha256(unhashed) != expected_content
        or sha256_file(array_path) != metadata.get("array_sha256")
        or metadata.get("targets_accessed_during_fit") is not False
    ):
        raise ResidualGraphDeemError(f"invalid MoE fit artifact: {variant}/{cell}/seed{seed}")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if not name.startswith("state__")}
    return arrays, metadata


def load_baseline_score(baseline_dir: Path, cell: str, seed: int) -> np.ndarray:
    array_path = baseline_dir / "fits" / cell / f"B3__seed{int(seed)}.npz"
    metadata_path = array_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(array_path) != metadata.get("array_sha256"):
        raise ResidualGraphDeemError(f"frozen B3 hash mismatch: {cell}/seed{seed}")
    with np.load(array_path, allow_pickle=False) as data:
        return np.asarray(data["score"], dtype=np.float64)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def posthoc_controls(arrays: dict, metadata: dict, *, variant: str, cell: str, seed: int):
    base = np.asarray(arrays["base_family_contributions"], dtype=np.float64)
    gates = np.asarray(arrays["gates"], dtype=np.float64)
    if base.shape != gates.shape:
        raise ResidualGraphDeemError("family contribution/gate shape mismatch")
    generator = np.random.Generator(np.random.PCG64(_stable_seed(variant, cell, str(seed))))
    permutation = generator.permutation(len(gates))
    permuted_logit = float(metadata["aligned_bias"]) + np.sum(base * gates[permutation], axis=1)
    permuted_score = 1.0 / (1.0 + np.exp(-np.clip(permuted_logit, -700.0, 700.0)))
    mapped_gates = np.roll(gates, shift=1, axis=1)
    mapped_logit = float(metadata["aligned_bias"]) + np.sum(base * mapped_gates, axis=1)
    mapped_score = 1.0 / (1.0 + np.exp(-np.clip(mapped_logit, -700.0, 700.0)))
    return permuted_score, mapped_score


def summarize(per_cell: list[dict], method: str, metric: str = "auroc") -> dict:
    selected = [row for row in per_cell if row["method"] == method]
    by_family = defaultdict(list)
    for row in selected:
        by_family[row["dataset_family"]].append(float(row[metric]))
    family_means = {family: float(np.mean(values)) for family, values in by_family.items()}
    qa = [row[metric] for row in selected if row["task_type"] == "QA"]
    math_rows = [row[metric] for row in selected if row["task_type"] == "math"]
    return {
        "method": method,
        "metric": metric,
        "n_cells": len(selected),
        "n_families": len(family_means),
        "cell_macro": float(np.mean([row[metric] for row in selected])),
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "qa_macro": float(np.mean(qa)) if qa else None,
        "math_macro": float(np.mean(math_rows)) if math_rows else None,
        "worst_cell": float(min(row[metric] for row in selected)),
        "family_means": family_means,
    }


def paired_family_bootstrap(
    per_cell: list[dict], candidate: str, *, draws: int, seed: int
) -> dict:
    lookup = {(row["cell_id"], row["method"]): row for row in per_cell}
    cells = sorted({row["cell_id"] for row in per_cell})
    paired = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        if (cell, candidate) not in lookup or (cell, "B3") not in lookup:
            continue
        family = lookup[(cell, candidate)]["dataset_family"]
        delta = float(
            lookup[(cell, candidate)]["auroc"] - lookup[(cell, "B3")]["auroc"]
        )
        paired[family].append(delta)
        cell_delta[cell] = delta
    families = sorted(paired)
    family_delta = {family: float(np.mean(paired[family])) for family in families}
    observed = float(np.mean(list(family_delta.values())))
    rng = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.choice(families, len(families), replace=True)
        values = []
        for family in selected:
            within = np.asarray(paired[family], dtype=np.float64)
            values.append(float(np.mean(within[rng.integers(0, len(within), len(within))])))
        distribution[draw] = float(np.mean(values))
    tolerance = 5e-4
    return {
        "candidate": candidate,
        "reference": "B3",
        "equal_family_auroc_delta": observed,
        "lower": float(np.quantile(distribution, 0.025)),
        "upper": float(np.quantile(distribution, 0.975)),
        "one_sided_p": float((1 + np.sum(distribution <= 0.0)) / (draws + 1)),
        "wins": int(sum(value > tolerance for value in cell_delta.values())),
        "ties": int(sum(abs(value) <= tolerance for value in cell_delta.values())),
        "losses": int(sum(value < -tolerance for value in cell_delta.values())),
        "worst_cell_delta": float(min(cell_delta.values())),
        "family_delta": family_delta,
        "cell_delta": cell_delta,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def leave_one_family_out_selection(
    per_cell: list[dict], variants: list[str], complexity_order: list[str]
) -> dict:
    lookup = {(row["cell_id"], row["method"]): row for row in per_cell}
    cells = sorted({row["cell_id"] for row in per_cell})
    family_cells = defaultdict(list)
    for cell in cells:
        family_cells[lookup[(cell, "B3")]["dataset_family"]].append(cell)
    if len(family_cells) < 2:
        return {
            "status": "insufficient_dataset_families",
            "n_families": len(family_cells),
            "folds": [],
            "equal_family_held_delta": None,
            "selection_frequency": {variant: 0 for variant in variants},
            "selection_uses_donor_family_labels": True,
        }
    rank = {variant: index for index, variant in enumerate(complexity_order)}
    folds = []
    for held_family in sorted(family_cells):
        donor_families = [family for family in family_cells if family != held_family]
        donor_score = {}
        for variant in variants:
            values = []
            for family in donor_families:
                values.append(
                    float(
                        np.mean(
                            [lookup[(cell, variant)]["auroc"] for cell in family_cells[family]]
                        )
                    )
                )
            donor_score[variant] = float(np.mean(values))
        best_value = max(donor_score.values())
        tied = [variant for variant, value in donor_score.items() if value >= best_value - 1e-12]
        selected = min(tied, key=lambda variant: rank[variant])
        held_candidate = float(
            np.mean([lookup[(cell, selected)]["auroc"] for cell in family_cells[held_family]])
        )
        held_baseline = float(
            np.mean([lookup[(cell, "B3")]["auroc"] for cell in family_cells[held_family]])
        )
        folds.append(
            {
                "held_family": held_family,
                "selected_variant": selected,
                "donor_equal_family_auroc": donor_score[selected],
                "held_candidate_auroc": held_candidate,
                "held_b3_auroc": held_baseline,
                "held_delta": held_candidate - held_baseline,
            }
        )
    return {
        "folds": folds,
        "equal_family_held_delta": float(np.mean([row["held_delta"] for row in folds])),
        "selection_frequency": {
            variant: int(sum(row["selected_variant"] == variant for row in folds))
            for variant in variants
        },
        "selection_uses_donor_family_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", required=True, help="comma-separated variant IDs")
    parser.add_argument("--cells", default="all", help="all, screen, or comma-separated IDs")
    parser.add_argument("--seeds", default="0", help="comma-separated integer seeds")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    config = load_config(args.config)
    registry = load_registry(args.registry)
    definitions = variant_lookup(config)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(definitions))
    if unknown or not variants:
        raise ValueError("unknown/empty variant selection: " + ", ".join(unknown))
    all_cells = [row["cell_id"] for row in registry["cells"]]
    if args.cells == "all":
        cells = all_cells
    elif args.cells == "screen":
        cells = list(config["screen_cells"])
    else:
        cells = [value.strip() for value in args.cells.split(",") if value.strip()]
    if not cells or not set(cells).issubset(set(all_cells)):
        raise ValueError("invalid cell selection")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("empty seed selection")

    per_fit, per_cell, gate_rows = [], [], []
    for cell in cells:
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        target = join_labels_by_id(bundle, sidecar)
        baseline_seed_scores = [load_baseline_score(args.baseline_dir, cell, seed) for seed in seeds]
        baseline = np.mean(baseline_seed_scores, axis=0)
        base_metrics = metrics(target, baseline)
        per_cell.append(
            {
                "cell_id": cell,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "method": "B3",
                **base_metrics,
            }
        )
        for seed, score in zip(seeds, baseline_seed_scores):
            per_fit.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": "B3",
                    "seed": seed,
                    **metrics(target, score),
                }
            )

        for variant in variants:
            seed_scores, permuted_scores, mapped_scores = [], [], []
            for seed, baseline_seed in zip(seeds, baseline_seed_scores):
                arrays, metadata = load_fit(args.run_dir, variant, cell, seed)
                if not np.array_equal(arrays["baseline_score"], baseline_seed):
                    raise ResidualGraphDeemError(
                        f"candidate does not bind exact baseline: {variant}/{cell}/seed{seed}"
                    )
                score = np.asarray(arrays["score"], dtype=np.float64)
                permuted, mapped = posthoc_controls(
                    arrays, metadata, variant=variant, cell=cell, seed=seed
                )
                seed_scores.append(score)
                permuted_scores.append(permuted)
                mapped_scores.append(mapped)
                per_fit.append(
                    {
                        "cell_id": cell,
                        "dataset_family": bundle.dataset_family,
                        "task_type": bundle.task_type,
                        "method": variant,
                        "seed": seed,
                        **metrics(target, score),
                    }
                )
                gate_rows.append(
                    {
                        "cell_id": cell,
                        "dataset_family": bundle.dataset_family,
                        "variant": variant,
                        "seed": seed,
                        **metadata["diagnostics"],
                    }
                )
            for method, matrix in (
                (variant, seed_scores),
                (f"{variant}::ROW_PERMUTED_GATE", permuted_scores),
                (f"{variant}::FAMILY_PERMUTED_GATE", mapped_scores),
            ):
                score = np.mean(matrix, axis=0)
                per_cell.append(
                    {
                        "cell_id": cell,
                        "dataset_family": bundle.dataset_family,
                        "task_type": bundle.task_type,
                        "method": method,
                        **metrics(target, score),
                    }
                )

    methods = ["B3"] + variants
    summaries = [summarize(per_cell, method, metric) for method in methods for metric in ("auroc", "auprc")]
    comparisons = [
        paired_family_bootstrap(
            per_cell,
            variant,
            draws=int(args.bootstrap_draws),
            seed=20260824 + index,
        )
        for index, variant in enumerate(variants)
    ]
    adjusted = holm({row["candidate"]: row["one_sided_p"] for row in comparisons})
    summary_lookup = {(row["method"], row["metric"]): row for row in summaries}
    for row in comparisons:
        variant = row["candidate"]
        candidate = summary_lookup[(variant, "auroc")]
        baseline = summary_lookup[("B3", "auroc")]
        row["holm_p"] = adjusted[variant]
        row["qa_delta"] = (
            candidate["qa_macro"] - baseline["qa_macro"]
            if candidate["qa_macro"] is not None and baseline["qa_macro"] is not None
            else None
        )
        row["math_delta"] = (
            candidate["math_macro"] - baseline["math_macro"]
            if candidate["math_macro"] is not None and baseline["math_macro"] is not None
            else None
        )
        row["point_estimate_improved"] = row["equal_family_auroc_delta"] > 0.0
        row["promotion_gate"] = bool(
            row["equal_family_auroc_delta"] >= 0.0025
            and row["lower"] > 0.0
            and row["holm_p"] <= 0.05
            and row["qa_delta"] is not None
            and row["math_delta"] is not None
            and row["qa_delta"] >= -0.005
            and row["math_delta"] >= -0.005
            and row["wins"] + row["ties"] >= 14
            and row["worst_cell_delta"] >= -0.02
        )

    lofo = leave_one_family_out_selection(per_cell, variants, variants)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "PER_FIT_METRICS.csv", per_fit)
    write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    write_csv(args.out_dir / "GATE_DIAGNOSTICS.csv", gate_rows)
    atomic_write_json(args.out_dir / "SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "COMPARISONS.json", comparisons)
    atomic_write_json(args.out_dir / "LOFO_SELECTION.json", lofo)
    report = {
        "schema": "deem_b3_moe_evaluation_v1",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory",
        "natural_24cell_targets_previously_opened": True,
        "variants": variants,
        "cells": cells,
        "seeds": seeds,
        "bootstrap_draws": int(args.bootstrap_draws),
        "comparisons": comparisons,
        "lofo_selection": lofo,
        "posthoc_gate_controls_are_diagnostic_not_valid_ebm_fits": True,
    }
    atomic_write_json(args.out_dir / "REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
