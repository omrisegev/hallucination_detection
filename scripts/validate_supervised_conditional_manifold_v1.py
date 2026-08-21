#!/usr/bin/env python3
"""One-shot external validation of a frozen supervised metric candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.graph_topology import (  # noqa: E402
    exact_length_permutations,
    propensity_crt_permutations,
    smoothness_against_permutations,
)
from spectral_utils.supervised_manifold_discovery import (  # noqa: E402
    TIE_SEEDS,
    metric_matrix,
    select_label_free_graph,
    stable_seed,
    target_blind_tie_keys,
)
from scripts.supervised_conditional_manifold_discovery_v1 import (  # noqa: E402
    CRT_BRIER_TOLERANCE,
    CRT_CALIBRATION_MAE_MAX,
    CRT_OVERLAP_FRACTION_MIN,
    DISTINCT_ADVANTAGE_MIN,
    EXACT_MIXED_STRATA_MIN,
    EXACT_MOVABLE_FRACTION_MIN,
    EXACT_MOVABLE_ROWS_MIN,
    MEDIAN_CONDITIONAL_EFFECT_MIN,
    _iu_utility,
    _sha256,
    _write_json,
    crt_eligible,
    exact_eligible,
)


VERSION = "supervised-conditional-manifold-external-validation-v1-2026-08-20"
DEFAULT_CANDIDATE = ROOT / "results/supervised_conditional_manifold_discovery_v1/FROZEN_CANDIDATE.json"
DEFAULT_OUT = ROOT / "results/supervised_conditional_manifold_external_validation_v1"
PERMUTATIONS = 999


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _resolve(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    manifest_relative = (manifest_path.parent / value).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (ROOT / value).resolve()


def load_manifest_cells(manifest_path: Path, candidate: dict) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "supervised-conditional-manifold-validation-manifest-v1":
        raise RuntimeError("unknown external validation manifest version")
    if manifest.get("standardization_contract") != "registered_per_cell_unlabeled":
        raise RuntimeError("external matrix standardization contract differs from discovery")
    discovery_families = set(manifest.get("discovery_dataset_families", ()))
    expected_features = tuple(candidate["feature_names"])
    cells = []
    for spec in manifest.get("cells", ()):
        if not spec.get("independent_rows", False):
            raise RuntimeError(
                f"{spec.get('cell')}: grouped/repeated rows require a group-aware null; "
                "v1 validator accepts only manifest-certified independent rows"
            )
        if bool(spec.get("dataset_new")) and spec["dataset_family"] in discovery_families:
            raise RuntimeError(f"{spec['cell']}: dataset_new conflicts with discovery-family overlap")
        path = _resolve(spec["npz"], manifest_path)
        with np.load(path, allow_pickle=True) as bundle:
            matrix = np.asarray(bundle[spec.get("matrix_key", "X")], dtype=float)
            names = tuple(map(str, bundle[spec.get("feature_names_key", "feature_names")]))
            length = np.asarray(bundle[spec.get("length_key", "trace_length")], dtype=float)
            target = np.asarray(bundle[spec.get("target_key", "hallucination_target")], dtype=int)
        lookup = {name: index for index, name in enumerate(names)}
        missing = [name for name in expected_features if name not in lookup]
        if missing:
            raise RuntimeError(f"{spec['cell']}: missing frozen features {missing}")
        matrix = matrix[:, [lookup[name] for name in expected_features]]
        if matrix.shape != (len(target), len(expected_features)):
            raise RuntimeError(f"{spec['cell']}: matrix/target shape mismatch")
        if length.shape != target.shape or not np.isfinite(matrix).all():
            raise RuntimeError(f"{spec['cell']}: invalid length or feature values")
        if not np.isin(target, (0, 1)).all() or len(np.unique(target)) != 2:
            raise RuntimeError(f"{spec['cell']}: target must contain both binary classes")
        cells.append({
            **spec,
            "path": str(path),
            "path_sha256": _sha256(path),
            "X": matrix,
            "length": length,
            "target": target,
        })
    if not cells:
        raise RuntimeError("external validation manifest contains no cells")
    return manifest, cells


def evaluate_cell(cell: dict, candidate: dict, *, permutations: int) -> list[dict]:
    weights = np.asarray(candidate["weights"], dtype=float)
    support = np.asarray(candidate["support_indices"], dtype=int)
    samples = metric_matrix(cell["X"], weights, support)
    comparator = candidate["linear_comparator"]
    coefficient = np.asarray(comparator["coefficient"], dtype=float)
    if coefficient.shape != (samples.shape[1],):
        raise RuntimeError("frozen linear comparator dimension mismatch")
    linear_score = samples @ coefficient + float(comparator["intercept"])
    exact, exact_diag = exact_length_permutations(
        cell["target"], cell["length"], permutations=permutations,
        seed=stable_seed(VERSION, cell["cell"], "exact"),
    )
    crt, crt_diag = propensity_crt_permutations(
        cell["target"], cell["length"], permutations=permutations,
        seed=stable_seed(VERSION, cell["cell"], "crt"),
    )
    rows = []
    for tie_seed in TIE_SEEDS:
        tie_keys = target_blind_tie_keys(
            len(cell["target"]), namespace=cell["cell"], seed=int(tie_seed)
        )
        for graph_role, values in (
            ("metric_graph", samples),
            ("linear_score_graph", linear_score[:, None]),
        ):
            graph, health = select_label_free_graph(values, tie_keys=tie_keys)
            base = {
                "cell": cell["cell"],
                "lane": cell["lane"],
                "dataset_family": cell["dataset_family"],
                "model_family": cell["model_family"],
                "dataset_new": bool(cell["dataset_new"]),
                "model_new": bool(cell["model_new"]),
                "graph_role": graph_role,
                "tie_seed": int(tie_seed),
                "n": len(cell["target"]),
                "error_rate": float(np.mean(cell["target"])),
                "graph_eligible": bool(health["eligible"]),
                "selected_k": health.get("selected_k"),
                "exact_eligible": exact_eligible(exact_diag),
                "crt_eligible": crt_eligible(crt_diag),
            }
            if graph is None:
                rows.append({**base, "exact_effect": float("nan"), "exact_p": float("nan"), "crt_effect": float("nan"), "crt_p": float("nan"), "min_conditional_effect": float("nan")})
                continue
            exact_result = smoothness_against_permutations(graph, cell["target"], exact)
            crt_result = smoothness_against_permutations(graph, cell["target"], crt)
            utility = _iu_utility(values, graph, cell["target"]) if graph_role == "metric_graph" else {"iu_auroc": float("nan"), "liu_auroc": float("nan"), "liu_delta_auroc": float("nan")}
            rows.append({
                **base,
                "exact_effect": exact_result["effect"],
                "exact_p": exact_result["p_smoother"],
                "crt_effect": crt_result["effect"],
                "crt_p": crt_result["p_smoother"],
                "min_conditional_effect": min(exact_result["effect"], crt_result["effect"]),
                **utility,
            })
    return rows


def decide(rows: list[dict]) -> dict:
    cells = sorted({row["cell"] for row in rows})
    per_cell = []
    for cell in cells:
        metric = [row for row in rows if row["cell"] == cell and row["graph_role"] == "metric_graph"]
        linear = [row for row in rows if row["cell"] == cell and row["graph_role"] == "linear_score_graph"]
        metric_pass = all(
            row["graph_eligible"] and row["exact_eligible"] and row["crt_eligible"]
            and row["exact_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["crt_effect"] >= MEDIAN_CONDITIONAL_EFFECT_MIN
            and row["exact_p"] <= .05 and row["crt_p"] <= .05
            for row in metric
        )
        advantages = [
            metric_row["min_conditional_effect"] - linear_row["min_conditional_effect"]
            for metric_row, linear_row in zip(metric, linear)
        ]
        distinct = bool(metric_pass and all(value >= DISTINCT_ADVANTAGE_MIN for value in advantages))
        per_cell.append({
            "cell": cell,
            "metric_pass": metric_pass,
            "distinct_vs_linear": distinct,
            "minimum_advantage": float(np.min(advantages)),
            "dataset_new": bool(metric[0]["dataset_new"]),
            "model_new": bool(metric[0]["model_new"]),
        })
    if all(row["distinct_vs_linear"] for row in per_cell):
        decision = "EXTERNAL_VALIDATION_PASS_DISTINCT_GEOMETRY"
    elif all(row["metric_pass"] for row in per_cell):
        decision = "EXTERNAL_VALIDATION_SHARED_DIRECTION_ONLY"
    else:
        decision = "EXTERNAL_VALIDATION_FAIL"
    return {"decision": decision, "per_cell": per_cell}


def run(args) -> None:
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if candidate.get("status") != "internal_discovery_candidate_awaiting_external_validation":
        raise RuntimeError("candidate is not frozen for external validation")
    manifest, cells = load_manifest_cells(args.manifest, candidate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    definition = {
        "version": VERSION,
        "candidate": str(args.candidate),
        "candidate_sha256": _sha256(args.candidate),
        "candidate_source_run_fingerprint": candidate["source_run_fingerprint"],
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "validation_name": manifest["validation_name"],
        "permutations": int(args.permutations),
        "inputs": [
            {"cell": cell["cell"], "path": cell["path"], "sha256": cell["path_sha256"]}
            for cell in cells
        ],
        "one_shot_external_validation": True,
    }
    _write_json(args.out_dir / "RUN_DEFINITION.json", definition)
    rows = []
    for cell in cells:
        rows.extend(evaluate_cell(cell, candidate, permutations=args.permutations))
    _write_csv(args.out_dir / "CELL_METRICS.csv", rows)
    decision = decide(rows)
    _write_json(args.out_dir / "DECISION.json", decision)
    lines = [
        "# Supervised conditional manifold external validation v1", "",
        f"**Decision: `{decision['decision']}`**", "",
        "The candidate, graph rule, features, weights, comparator, and thresholds were frozen before this manifest's labels were evaluated.", "",
    ]
    for row in decision["per_cell"]:
        lines.append(f"- `{row['cell']}`: metric={row['metric_pass']}, distinct={row['distinct_vs_linear']}, min advantage={row['minimum_advantage']:+.3f}.")
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"decision={decision['decision']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    if args.permutations < 199:
        raise ValueError("external validation requires at least 199 conditional draws")
    run(args)


if __name__ == "__main__":
    main()
