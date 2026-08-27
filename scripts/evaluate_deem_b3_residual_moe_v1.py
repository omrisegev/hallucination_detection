#!/usr/bin/env python3
"""Evaluation-only boundary for frozen B3 residual-family MoE scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


DEFAULT_CONFIG = ROOT / "configs/deem_b3_residual_moe_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
ALLOWED_CONFIG_SCHEMAS = {
    "deem_b3_residual_moe_v1_config",
    "deem_b3_residual_pgrd_v1_config",
}


def _load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") not in ALLOWED_CONFIG_SCHEMAS:
        raise ResidualGraphDeemError("unsupported residual experiment config")
    variants = value.get("variants", [])
    identifiers = [str(row.get("id")) for row in variants]
    if not variants or len(identifiers) != len(set(identifiers)):
        raise ResidualGraphDeemError("empty or duplicated residual experiment roster")
    if not value.get("screen_cells"):
        raise ResidualGraphDeemError("residual experiment has no frozen screen")
    boundary = value.get("scientific_boundary", {})
    if (
        boundary.get("fit_is_label_free") is not True
        or boundary.get("natural_24cell_targets_previously_opened") is not True
    ):
        raise ResidualGraphDeemError("residual experiment boundary is incomplete")
    return value


def _load_run_contract(
    run_dir: Path,
    config_path: Path,
    registry_path: Path,
) -> tuple[dict, dict[tuple[str, str], dict]]:
    definition_path = run_dir / "RUN_DEFINITION.json"
    manifest_path = run_dir / "SCORE_FREEZE_MANIFEST.json"
    if not definition_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("frozen run definition or score manifest is missing")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        definition.get("status") != "complete"
        or definition.get("targets_accessed_during_fit") is not False
        or definition.get("labels_module_imported") is not False
        or definition.get("config_sha256") != sha256_file(config_path)
        or definition.get("registry_sha256") != sha256_file(registry_path)
        or definition.get("manifest_sha256") != canonical_sha256(manifest)
        or int(definition.get("n_score_artifacts", -1)) != len(manifest)
    ):
        raise ResidualGraphDeemError("run definition is not a valid frozen fit")
    lookup: dict[tuple[str, str], dict] = {}
    for row in manifest:
        key = (str(row.get("variant_id")), str(row.get("cell_id")))
        if key in lookup:
            raise ResidualGraphDeemError(f"duplicated score manifest key: {key}")
        lookup[key] = row
    return definition, lookup


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    values = np.asarray(score, dtype=float)
    if y.shape != values.shape or len(np.unique(y)) != 2 or not np.isfinite(values).all():
        raise ResidualGraphDeemError("invalid target/score pair")
    return {
        "auroc": float(roc_auc_score(y, values)),
        "auprc": float(average_precision_score(y, values)),
    }


def _load_score(
    run_dir: Path,
    variant: str,
    cell: str,
    *,
    definition: dict,
    manifest_row: dict,
) -> tuple[dict, dict]:
    path = run_dir / "scores" / variant / cell / f"{variant}.npz"
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing residual-MoE score: {variant}/{cell}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    unhashed = dict(metadata)
    expected_content = unhashed.pop("content_sha256", None)
    if (
        metadata.get("status") != "complete"
        or metadata.get("targets_accessed_during_fit") is not False
        or metadata.get("labels_module_imported") is not False
        or not (
            metadata.get("calibration_excludes_entire_target_dataset_family") is True
            or metadata.get("within_cell_transductive_graph") is True
        )
        or metadata.get("variant_id") != variant
        or metadata.get("cell_id") != cell
        or metadata.get("source_sha256") != definition.get("source_sha256")
        or canonical_sha256(unhashed) != expected_content
        or sha256_file(path) != metadata.get("array_sha256")
        or metadata.get("array_sha256") != manifest_row.get("array_sha256")
        or sha256_file(metadata_path) != manifest_row.get("metadata_sha256")
        or manifest_row.get("array_path") != str(path.relative_to(run_dir))
        or manifest_row.get("metadata_path") != str(metadata_path.relative_to(run_dir))
    ):
        raise ResidualGraphDeemError(f"invalid score artifact: {variant}/{cell}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if not np.array_equal(arrays["score"], arrays["baseline_score"]) and variant.endswith(
        "EXACT_ALIAS"
    ):
        raise ResidualGraphDeemError(f"identity artifact changed B3: {cell}")
    return arrays, metadata


def _summary(rows: list[dict], method: str, metric: str) -> dict:
    selected = [row for row in rows if row["method"] == method]
    by_family = defaultdict(list)
    for row in selected:
        by_family[row["dataset_family"]].append(float(row[metric]))
    family_means = {
        family: float(np.mean(values)) for family, values in sorted(by_family.items())
    }
    return {
        "method": method,
        "metric": metric,
        "n_cells": len(selected),
        "n_families": len(family_means),
        "cell_macro": float(np.mean([row[metric] for row in selected])),
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "family_means": family_means,
    }


def _comparison(rows: list[dict], variant: str, *, draws: int, seed: int) -> dict:
    lookup = {(row["cell_id"], row["method"]): row for row in rows}
    cells = sorted({row["cell_id"] for row in rows})
    by_family = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        delta = float(lookup[(cell, variant)]["auroc"] - lookup[(cell, "B3")]["auroc"])
        family = str(lookup[(cell, "B3")]["dataset_family"])
        by_family[family].append(delta)
        cell_delta[cell] = delta
    family_delta = {
        family: float(np.mean(values)) for family, values in sorted(by_family.items())
    }
    observed = float(np.mean(list(family_delta.values())))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    families = list(family_delta)
    distribution = np.empty(int(draws), dtype=float)
    for index in range(int(draws)):
        sampled = rng.choice(families, size=len(families), replace=True)
        distribution[index] = float(np.mean([family_delta[item] for item in sampled]))

    # Exact synchronized sign-flip test over dataset-family mean deltas.  This
    # is a valid small-cluster randomization test under family-level symmetry;
    # unlike the historical evaluator, it does not bootstrap around the
    # observed estimate and call that tail area a null p-value.
    values = np.asarray(list(family_delta.values()), dtype=float)
    signs = np.asarray(
        [
            [1.0 if (mask >> bit) & 1 else -1.0 for bit in range(len(values))]
            for mask in range(1 << len(values))
        ],
        dtype=float,
    )
    null = np.mean(signs * values[None, :], axis=1)
    one_sided = float(np.mean(null >= observed - 1e-15))
    tolerance = 5e-4
    return {
        "candidate": variant,
        "reference": "B3",
        "equal_family_auroc_delta": observed,
        "descriptive_bootstrap_lower": float(np.quantile(distribution, 0.025)),
        "descriptive_bootstrap_upper": float(np.quantile(distribution, 0.975)),
        "exact_family_signflip_one_sided_p": one_sided,
        "wins": int(sum(value > tolerance for value in cell_delta.values())),
        "ties": int(sum(abs(value) <= tolerance for value in cell_delta.values())),
        "losses": int(sum(value < -tolerance for value in cell_delta.values())),
        "worst_cell_delta": float(min(cell_delta.values())),
        "family_delta": family_delta,
        "cell_delta": cell_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--cells", default="screen")
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()

    config = _load_config(args.config)
    registry = load_registry(args.registry)
    definition, manifest = _load_run_contract(
        args.run_dir, args.config, args.registry
    )
    all_variants = [str(row["id"]) for row in config["variants"]]
    variants = all_variants if args.variants == "all" else [
        item.strip() for item in args.variants.split(",") if item.strip()
    ]
    if not variants or not set(variants).issubset(set(all_variants)):
        raise ValueError("invalid variant selection")
    if not set(variants).issubset(set(definition.get("variants", []))):
        raise ResidualGraphDeemError("requested variants are absent from the frozen run")
    all_cells = [str(row["cell_id"]) for row in registry["cells"]]
    if args.cells == "screen":
        cells = list(config["screen_cells"])
    elif args.cells == "all":
        cells = all_cells
    else:
        cells = [item.strip() for item in args.cells.split(",") if item.strip()]
    if not cells or not set(cells).issubset(set(all_cells)):
        raise ValueError("invalid cell selection")

    rows = []
    gate_rows = []
    for cell in cells:
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        target = join_labels_by_id(bundle, sidecar)
        loaded = {}
        for variant in variants:
            manifest_row = manifest.get((variant, cell))
            if manifest_row is None:
                raise ResidualGraphDeemError(
                    f"score is absent from frozen manifest: {variant}/{cell}"
                )
            arrays, metadata = _load_score(
                args.run_dir,
                variant,
                cell,
                definition=definition,
                manifest_row=manifest_row,
            )
            if not np.array_equal(arrays["baseline_score"], loaded.get("baseline", arrays["baseline_score"])):
                raise ResidualGraphDeemError(f"variants bind different B3 arrays: {cell}")
            loaded["baseline"] = arrays["baseline_score"]
            loaded[variant] = arrays["score"]
            gates = np.asarray(arrays["gates"], dtype=float)
            present = np.any(gates != 0.0, axis=0)
            if not np.any(present):
                raise ResidualGraphDeemError(f"no present gate family: {variant}/{cell}")
            iteration_rows = metadata.get("iteration_diagnostics", [])
            calibration = (
                iteration_rows[-1].get("calibration", {}) if iteration_rows else {}
            )
            gate_rows.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "variant": variant,
                    "gate_mean_abs_deviation_from_one_present": float(
                        np.mean(np.abs(gates[:, present] - 1.0))
                    ),
                    "gate_min_present": float(np.min(gates[:, present])),
                    "gate_max_present": float(np.max(gates[:, present])),
                    "correction_sd": float(np.std(arrays["correction_z"])),
                    "selected_eigenvalue": float(
                        calibration.get("selected_eigenvalue", np.nan)
                    ),
                    "graph_trace_a0": float(calibration.get("trace_a0", np.nan)),
                    "direction_norm": float(
                        calibration.get("direction_norm", np.nan)
                    ),
                }
            )
        rows.append(
            {
                "cell_id": cell,
                "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type,
                "method": "B3",
                **_metrics(target, loaded["baseline"]),
            }
        )
        for variant in variants:
            rows.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": variant,
                    **_metrics(target, loaded[variant]),
                }
            )

    methods = ["B3"] + variants
    summaries = [
        _summary(rows, method, metric)
        for method in methods
        for metric in ("auroc", "auprc")
    ]
    comparisons = [
        _comparison(
            rows, variant, draws=args.bootstrap_draws, seed=20260825 + index
        )
        for index, variant in enumerate(variants)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", rows)
    _write_csv(args.out_dir / "GATE_DIAGNOSTICS.csv", gate_rows)
    atomic_write_json(args.out_dir / "SUMMARY.json", summaries)
    atomic_write_json(args.out_dir / "COMPARISONS.json", comparisons)
    report = {
        "schema": "deem_b3_residual_moe_evaluation_v1",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory",
        "natural_24cell_targets_previously_opened": True,
        "config_schema": config["schema"],
        "run_source_sha256": definition["source_sha256"],
        "multiplicity_status": "unadjusted_exploratory_screen",
        "cells": cells,
        "variants": variants,
        "comparisons": comparisons,
    }
    atomic_write_json(args.out_dir / "REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
