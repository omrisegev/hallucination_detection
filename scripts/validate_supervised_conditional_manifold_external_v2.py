#!/usr/bin/env python3
"""Conservative external-to-discovery validation of the frozen metric.

This validator is deliberately stricter than the historical v1 entry point:
it enforces exactly 999 conditional draws, requires independent-dataset
coverage, evaluates equal-weight and linear-residual controls, keeps tie seeds
as robustness dimensions, and refuses to overwrite a completed one-shot run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import bootstrap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.graph_topology import (  # noqa: E402
    exact_length_permutations,
    holm_adjust,
    propensity_crt_permutations,
)
from spectral_utils.laplacian_upcr import symmetric_normalized_laplacian  # noqa: E402
from spectral_utils.supervised_manifold_discovery import (  # noqa: E402
    TIE_SEEDS,
    metric_matrix,
    select_label_free_graph,
    stable_seed,
    target_blind_tie_keys,
)
from scripts.supervised_conditional_manifold_discovery_v1 import (  # noqa: E402
    DISTINCT_ADVANTAGE_MIN,
    MEDIAN_CONDITIONAL_EFFECT_MIN,
    _iu_utility,
    crt_eligible,
    exact_eligible,
)


VERSION = "supervised-conditional-manifold-external-validation-v2-2026-08-20"
PERMUTATIONS = 999
UTILITY_MIN = 0.005
DEFAULT_CANDIDATE = (
    ROOT / "results/supervised_conditional_manifold_discovery_v1/FROZEN_CANDIDATE.json"
)
DEFAULT_MANIFEST = (
    ROOT / "configs/supervised_conditional_manifold_external_validation_v1.json"
)
DEFAULT_OUT = ROOT / "results/supervised_conditional_manifold_external_validation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    manifest_relative = (manifest_path.parent / candidate).resolve()
    return manifest_relative if manifest_relative.exists() else (ROOT / candidate).resolve()


def load_inputs(manifest_path: Path, candidate_path: Path) -> tuple[dict, dict, list[dict]]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "internal_discovery_candidate_awaiting_external_validation":
        raise RuntimeError("candidate is not frozen for external validation")
    if manifest.get("version") != "supervised-conditional-manifold-external-validation-manifest-v2":
        raise RuntimeError("unknown external validation manifest version")
    if manifest.get("candidate_sha256") != sha256(candidate_path):
        raise RuntimeError("manifest was not sealed against this candidate")
    if manifest.get("standardization_contract") != "confidence_oriented_per_cell_unlabeled_zscore":
        raise RuntimeError("external standardization contract mismatch")
    expected = tuple(map(str, candidate["feature_names"]))
    cells = []
    for spec in manifest.get("cells", ()):
        if not spec.get("dataset_new", False):
            raise RuntimeError(f"{spec.get('cell')}: external v2 requires a new dataset family")
        if not spec.get("independent_rows", False):
            raise RuntimeError(f"{spec.get('cell')}: rows are not certified independent")
        path = resolve(spec["npz"], manifest_path)
        with np.load(path, allow_pickle=False) as bundle:
            matrix = np.asarray(bundle["X"], dtype=float)
            names = tuple(map(str, bundle["feature_names"]))
            length = np.asarray(bundle["trace_length"], dtype=float)
            target = np.asarray(bundle["hallucination_target"], dtype=int)
            row_id = np.asarray(bundle["row_id"]).astype(str)
        if names != expected:
            raise RuntimeError(f"{spec['cell']}: frozen feature order mismatch")
        if matrix.shape != (len(target), len(expected)) or length.shape != target.shape:
            raise RuntimeError(f"{spec['cell']}: matrix, target, or length shape mismatch")
        if len(row_id) != len(target) or len(set(row_id)) != len(row_id):
            raise RuntimeError(f"{spec['cell']}: row IDs are not unique")
        if not np.isfinite(matrix).all() or not np.isfinite(length).all():
            raise RuntimeError(f"{spec['cell']}: non-finite external inputs")
        if len(np.unique(target)) != 2:
            raise RuntimeError(f"{spec['cell']}: target is single class")
        cells.append({**spec, "path": str(path), "path_sha256": sha256(path),
                      "X": matrix, "length": length, "target": target})
    if not cells:
        raise RuntimeError("external manifest contains no cells")
    return candidate, manifest, cells


def residualize_against_score(matrix: np.ndarray, score: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    score = np.asarray(score, dtype=float)
    design = np.column_stack((np.ones(len(score)), score))
    coefficient = np.linalg.lstsq(design, matrix, rcond=None)[0]
    residual = matrix - design @ coefficient
    scale = residual.std(axis=0, keepdims=True)
    keep = scale[0] > 1e-10
    if not np.any(keep):
        raise RuntimeError("linear residualization removed every external feature")
    return residual[:, keep] / scale[:, keep]


def graph_result(graph, target: np.ndarray, permutations: np.ndarray) -> tuple[dict, np.ndarray]:
    laplacian = symmetric_normalized_laplacian(graph)
    observed = np.asarray(target, dtype=float)
    null = np.asarray(permutations, dtype=float)

    def rayleigh(values: np.ndarray) -> np.ndarray:
        values = values if values.ndim == 2 else values[:, None]
        numerator = np.sum(values * (laplacian @ values), axis=0)
        denominator = np.sum(values * values, axis=0)
        return numerator / np.maximum(denominator, 1e-12)

    observed_value = float(rayleigh(observed)[0])
    null_values = rayleigh(null)
    null_mean = float(np.mean(null_values))
    effect = float((null_mean - observed_value) / max(abs(null_mean), 1e-12))
    null_effect = (null_mean - null_values) / max(abs(null_mean), 1e-12)
    return {
        "rayleigh": observed_value,
        "null_mean": null_mean,
        "effect": effect,
        "p": float((1 + np.sum(null_values <= observed_value)) / (len(null_values) + 1)),
    }, null_effect


def evaluate_cell(cell: dict, candidate: dict) -> list[dict]:
    weights = np.asarray(candidate["weights"], dtype=float)
    support = np.asarray(candidate["support_indices"], dtype=int)
    metric = metric_matrix(cell["X"], weights, support)
    coefficient = np.asarray(candidate["linear_comparator"]["coefficient"], dtype=float)
    linear = metric @ coefficient + float(candidate["linear_comparator"]["intercept"])
    equal = np.asarray(cell["X"], dtype=float)
    residual = residualize_against_score(metric, linear)
    values = {
        "metric_graph": metric,
        "linear_score_graph": linear[:, None],
        "equal_weight_graph": equal,
        "linear_residual_graph": residual,
    }
    exact, exact_diag = exact_length_permutations(
        cell["target"], cell["length"], permutations=PERMUTATIONS,
        seed=stable_seed(VERSION, cell["cell"], "exact"),
    )
    crt, crt_diag = propensity_crt_permutations(
        cell["target"], cell["length"], permutations=PERMUTATIONS,
        seed=stable_seed(VERSION, cell["cell"], "crt"),
    )
    rows = []
    for tie_seed in TIE_SEEDS:
        tie_keys = target_blind_tie_keys(len(cell["target"]), namespace=cell["cell"], seed=tie_seed)
        current = {}
        for role, matrix in values.items():
            graph, health = select_label_free_graph(matrix, tie_keys=tie_keys)
            base = {
                "cell": cell["cell"], "dataset_family": cell["dataset_family"],
                "model_family": cell["model_family"], "dataset_new": bool(cell["dataset_new"]),
                "model_new": bool(cell["model_new"]), "graph_role": role,
                "tie_seed": int(tie_seed), "n": len(cell["target"]),
                "hallucination_rate": float(np.mean(cell["target"])),
                "graph_eligible": bool(health["eligible"]), "selected_k": health.get("selected_k"),
                "largest_component_fraction": float(health["largest_component_fraction"]),
                "isolated_fraction": float(health["isolated_fraction"]),
                "exact_eligible": bool(exact_eligible(exact_diag)),
                "crt_eligible": bool(crt_eligible(crt_diag)),
            }
            if graph is None:
                current[role] = {"base": base, "exact": None, "crt": None}
                continue
            exact_result, exact_null = graph_result(graph, cell["target"], exact)
            crt_result, crt_null = graph_result(graph, cell["target"], crt)
            current[role] = {
                "base": base, "exact": exact_result, "crt": crt_result,
                "exact_null": exact_null, "crt_null": crt_null,
                "utility": _iu_utility(matrix, graph, cell["target"])
                if role == "metric_graph" else None,
            }
        metric_row = current["metric_graph"]
        linear_row = current["linear_score_graph"]
        equal_row = current["equal_weight_graph"]
        primary_p = []
        if all(current[role]["exact"] is not None for role in current):
            adv_exact = metric_row["exact"]["effect"] - linear_row["exact"]["effect"]
            adv_crt = metric_row["crt"]["effect"] - linear_row["crt"]["effect"]
            adv_exact_null = metric_row["exact_null"] - linear_row["exact_null"]
            adv_crt_null = metric_row["crt_null"] - linear_row["crt_null"]
            advantage_p_exact = float((1 + np.sum(adv_exact_null >= adv_exact)) / (PERMUTATIONS + 1))
            advantage_p_crt = float((1 + np.sum(adv_crt_null >= adv_crt)) / (PERMUTATIONS + 1))
            primary_p = [
                metric_row["exact"]["p"], metric_row["crt"]["p"],
                current["linear_residual_graph"]["exact"]["p"],
                current["linear_residual_graph"]["crt"]["p"],
                advantage_p_exact, advantage_p_crt,
            ]
            adjusted = holm_adjust(np.asarray(primary_p, dtype=float))
        else:
            adv_exact = adv_crt = advantage_p_exact = advantage_p_crt = float("nan")
            adjusted = np.ones(6, dtype=float)
        for role, result in current.items():
            row = dict(result["base"])
            if result["exact"] is None:
                row.update({key: float("nan") for key in (
                    "exact_effect", "exact_p", "crt_effect", "crt_p",
                    "min_conditional_effect", "liu_delta_auroc",
                )})
            else:
                row.update({
                    "exact_effect": result["exact"]["effect"],
                    "exact_p": result["exact"]["p"],
                    "crt_effect": result["crt"]["effect"],
                    "crt_p": result["crt"]["p"],
                    "min_conditional_effect": min(result["exact"]["effect"], result["crt"]["effect"]),
                    "liu_delta_auroc": (
                        result["utility"]["liu_delta_auroc"] if result["utility"] else float("nan")
                    ),
                })
            if role == "metric_graph":
                row.update({
                    "exact_p_holm": float(adjusted[0]), "crt_p_holm": float(adjusted[1]),
                    "advantage_vs_linear_exact": adv_exact,
                    "advantage_vs_linear_crt": adv_crt,
                    "advantage_vs_linear_min": min(adv_exact, adv_crt),
                    "advantage_vs_linear_exact_p_holm": float(adjusted[4]),
                    "advantage_vs_linear_crt_p_holm": float(adjusted[5]),
                    "advantage_vs_equal_min": min(
                        result["exact"]["effect"] - equal_row["exact"]["effect"],
                        result["crt"]["effect"] - equal_row["crt"]["effect"],
                    ) if result["exact"] is not None else float("nan"),
                })
            elif role == "linear_residual_graph":
                row.update({"exact_p_holm": float(adjusted[2]), "crt_p_holm": float(adjusted[3])})
            rows.append(row)
    return rows


def interval(values: list[float], namespace: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    result = bootstrap(
        (array,), np.mean, n_resamples=5000, confidence_level=.95,
        method="percentile", random_state=stable_seed(VERSION, namespace),
    )
    return mean, float(result.confidence_interval.low), float(result.confidence_interval.high)


def decide(rows: list[dict], manifest: dict) -> tuple[dict, list[dict]]:
    metric = [row for row in rows if row["graph_role"] == "metric_graph"]
    residual = [row for row in rows if row["graph_role"] == "linear_residual_graph"]
    cells = sorted({row["cell"] for row in metric})
    per_cell = []
    for cell in cells:
        m = [row for row in metric if row["cell"] == cell]
        r = [row for row in residual if row["cell"] == cell]
        geometry = all(
            row["graph_eligible"] and row["exact_eligible"] and row["crt_eligible"]
            and row["exact_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["crt_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["exact_p_holm"] <= .05 and row["crt_p_holm"] <= .05
            for row in m
        )
        residual_pass = all(
            row["graph_eligible"] and row["exact_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["crt_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["exact_p_holm"] <= .05 and row["crt_p_holm"] <= .05
            for row in r
        )
        distinct = bool(geometry and all(
            row["advantage_vs_linear_min"] >= DISTINCT_ADVANTAGE_MIN
            and row["advantage_vs_linear_exact_p_holm"] <= .05
            and row["advantage_vs_linear_crt_p_holm"] <= .05
            for row in m
        ))
        per_cell.append({
            "cell": cell, "dataset_family": m[0]["dataset_family"],
            "model_family": m[0]["model_family"], "geometry_pass": geometry,
            "conditional_null_eligible": all(
                row["exact_eligible"] and row["crt_eligible"] for row in m
            ),
            "linear_residual_pass": residual_pass, "distinct_vs_linear_pass": distinct,
            "median_metric_effect": float(np.median([row["min_conditional_effect"] for row in m])),
            "median_residual_effect": float(np.median([row["min_conditional_effect"] for row in r])),
            "median_linear_advantage": float(np.median([row["advantage_vs_linear_min"] for row in m])),
            "median_equal_advantage": float(np.median([row["advantage_vs_equal_min"] for row in m])),
            "median_liu_delta": float(np.median([row["liu_delta_auroc"] for row in m])),
        })
    families = sorted({row["dataset_family"] for row in per_cell})
    family_rows = []
    for family in families:
        members = [row for row in per_cell if row["dataset_family"] == family]
        needed = int(np.ceil(2 * len(members) / 3))
        family_rows.append({
            "dataset_family": family, "n_cells": len(members),
            "conditional_null_eligible": (
                sum(row["conditional_null_eligible"] for row in members) >= needed
            ),
            "geometry_pass": sum(row["geometry_pass"] for row in members) >= needed,
            "linear_residual_pass": sum(row["linear_residual_pass"] for row in members) >= needed,
            "distinct_vs_linear_pass": sum(row["distinct_vs_linear_pass"] for row in members) >= needed,
            "metric_effect": float(np.mean([row["median_metric_effect"] for row in members])),
            "residual_effect": float(np.mean([row["median_residual_effect"] for row in members])),
            "linear_advantage": float(np.mean([row["median_linear_advantage"] for row in members])),
            "equal_advantage": float(np.mean([row["median_equal_advantage"] for row in members])),
            "liu_delta": float(np.mean([row["median_liu_delta"] for row in members])),
        })
    min_families = int(manifest.get("minimum_independent_dataset_families", 3))
    coverage = bool(
        len(families) >= min_families
        and any(row["dataset_new"] and row["model_new"] for row in metric)
    )
    required = int(np.ceil(2 * len(family_rows) / 3))
    geometry_pass = sum(row["geometry_pass"] for row in family_rows) >= required
    null_coverage_pass = (
        sum(row["conditional_null_eligible"] for row in family_rows) >= required
    )
    residual_pass = sum(row["linear_residual_pass"] for row in family_rows) >= required
    distinct_count_pass = sum(row["distinct_vs_linear_pass"] for row in family_rows) >= required
    advantage_mean, advantage_low, advantage_high = interval(
        [row["linear_advantage"] for row in family_rows], "linear-advantage"
    )
    utility_mean, utility_low, utility_high = interval(
        [row["liu_delta"] for row in family_rows], "utility"
    )
    distinct_pass = bool(
        geometry_pass and residual_pass and distinct_count_pass
        and advantage_mean >= DISTINCT_ADVANTAGE_MIN and advantage_low > 0
    )
    utility_pass = bool(utility_mean >= UTILITY_MIN and utility_low > 0)
    if not coverage:
        label = "INSUFFICIENT_EXTERNAL_COVERAGE"
    elif not null_coverage_pass:
        label = "CONDITIONAL_NULL_INELIGIBILITY_INVALIDATES_EXTERNAL_AUDIT"
    elif distinct_pass:
        label = "RETROSPECTIVE_EXTERNAL_DISTINCT_GEOMETRY_CANDIDATE"
    elif geometry_pass:
        label = "RETROSPECTIVE_EXTERNAL_SHARED_DIRECTION_ONLY"
    else:
        label = "RETROSPECTIVE_EXTERNAL_TRANSFER_FAILURE"
    return {
        "decision": label,
        "claim_boundary": manifest["claim_status"],
        "coverage_pass": coverage,
        "conditional_null_coverage_pass": null_coverage_pass,
        "geometry_pass": geometry_pass,
        "linear_residual_pass": residual_pass,
        "distinct_vs_linear_pass": distinct_pass,
        "utility_pass": utility_pass,
        "n_independent_dataset_families": len(families),
        "minimum_independent_dataset_families": min_families,
        "linear_advantage_equal_family_mean": advantage_mean,
        "linear_advantage_family_bootstrap_95": [advantage_low, advantage_high],
        "liu_delta_equal_family_mean": utility_mean,
        "liu_delta_family_bootstrap_95": [utility_low, utility_high],
        "per_cell": per_cell,
    }, family_rows


def run(args) -> None:
    if args.permutations != PERMUTATIONS:
        raise ValueError("external v2 requires exactly 999 conditional draws")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("one-shot output directory is non-empty; refusing overwrite")
    candidate, manifest, cells = load_inputs(args.manifest, args.candidate)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    definition = {
        "version": VERSION, "permutations": PERMUTATIONS,
        "candidate": str(args.candidate), "candidate_sha256": sha256(args.candidate),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "claim_status": manifest["claim_status"],
        "inputs": [{"cell": cell["cell"], "path": cell["path"], "sha256": cell["path_sha256"]}
                   for cell in cells],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "scripts/validate_supervised_conditional_manifold_external_v2.py",
                ROOT / "scripts/build_supervised_conditional_manifold_external_cells_v1.py",
                ROOT / "scripts/plot_supervised_conditional_manifold_external_v1.py",
                ROOT / "scripts/test_supervised_conditional_manifold_external_v2.py",
                ROOT / "scripts/verify_supervised_conditional_manifold_external_v1.py",
                ROOT / "docs/experiments/SUPERVISED_CONDITIONAL_MANIFOLD_EXTERNAL_VALIDATION_V1.md",
            )
        },
    }
    write_json(args.out_dir / "RUN_DEFINITION.json", definition)
    rows = []
    for index, cell in enumerate(cells, start=1):
        print(f"cell {index}/{len(cells)}: {cell['cell']}", flush=True)
        rows.extend(evaluate_cell(cell, candidate))
    write_csv(args.out_dir / "CELL_GRAPH_METRICS.csv", rows)
    decision, family_rows = decide(rows, manifest)
    write_csv(args.out_dir / "FAMILY_SUMMARY.csv", family_rows)
    write_json(args.out_dir / "DECISION.json", decision)
    lines = [
        "# Frozen supervised manifold external-to-discovery audit", "",
        f"**Decision: `{decision['decision']}`**", "",
        "This is a retrospective external-to-discovery audit, not prospective confirmation.", "",
        f"Coverage: {decision['n_independent_dataset_families']}/"
        f"{decision['minimum_independent_dataset_families']} dataset families. "
        f"Conditional-null coverage={decision['conditional_null_coverage_pass']}; "
        f"geometry={decision['geometry_pass']}; residual={decision['linear_residual_pass']}; "
        f"distinct-vs-linear={decision['distinct_vs_linear_pass']}; utility={decision['utility_pass']}.", "",
        "The four fixed graphs are the learned metric, its one-dimensional linear score, "
        "an equal-weight feature graph, and the metric after removing the linear score.", "",
    ]
    if not decision["conditional_null_coverage_pass"]:
        lines.extend([
            "The registered conditional tests were not eligible in enough independent "
            "dataset families. Descriptive effects below cannot be interpreted as a "
            "manifold pass or a transfer failure.", "",
        ])
    for row in family_rows:
        lines.append(
            f"- `{row['dataset_family']}`: null-eligible={row['conditional_null_eligible']}, "
            f"geometry={row['geometry_pass']}, "
            f"residual={row['linear_residual_pass']}, distinct={row['distinct_vs_linear_pass']}, "
            f"metric effect={row['metric_effect']:+.3f}, linear advantage={row['linear_advantage']:+.3f}, "
            f"LIU delta={row['liu_delta']:+.4f}."
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"decision={decision['decision']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
