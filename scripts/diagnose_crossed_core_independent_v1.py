#!/usr/bin/env python3
"""Independent label-free audit of the physical 3x3 source-by-operator core.

This file is deliberately separate from the main hierarchy analysis.  It reads
only target-free bundles and already-frozen B3 fit states.  It never imports or
opens a label sidecar.

The physical grid is

    source:   H15 entropy | sampled-token surprisal | raw log-partition
    operator: mean        | sliding variance        | CUSUM

For every cell and deterministic panel, exactly one response is selected per
source question and exactly 180 questions are retained.  Five balanced folds
are formed by stable hashes.  In each fold, cubic log-length residualization,
centering, scaling, and ridge fitting are learned on donor rows only.

The B3 analysis uses mean-of-five frozen per-feature contributions.  These are
called ``b3_atomic_core`` rather than physical measurements because B3's
nonlinear atomic contribution c_j depends on every input in j's historical
provenance family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "crossed_core_independent_v1"
N_PANELS = 20
N_GROUPS = 180
N_FOLDS = 5
RIDGE_ALPHA = 1.0
SEEDS = (0, 1, 2, 3, 4)
LOS_CELL = "losnet_hotpotqa_mistral7b"

SOURCES = ("entropy_h15", "sampled_surprisal", "raw_log_partition")
OPERATORS = ("mean", "sliding_variance", "cusum")
GRID = {
    ("entropy_h15", "mean"): "epr",
    ("entropy_h15", "sliding_variance"): "sw_var_peak",
    ("entropy_h15", "cusum"): "cusum_max",
    ("sampled_surprisal", "mean"): "epr_spilled",
    ("sampled_surprisal", "sliding_variance"): "sw_var_peak_spilled",
    ("sampled_surprisal", "cusum"): "cusum_max_spilled",
    ("raw_log_partition", "mean"): "epr_energy",
    ("raw_log_partition", "sliding_variance"): "sw_var_peak_energy",
    ("raw_log_partition", "cusum"): "cusum_max_energy",
}
FEATURES = tuple(GRID[(source, operator)] for source in SOURCES for operator in OPERATORS)
SOURCE_OF = tuple(source for source in SOURCES for _ in OPERATORS)
OPERATOR_OF = tuple(operator for _ in SOURCES for operator in OPERATORS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hex(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(value) for value in parts).encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "residual_graph_deem_24cell_v1_registry":
        raise ValueError("registry schema mismatch")
    if len(value.get("cells", [])) != 24:
        raise ValueError("expected the frozen 24-cell registry")
    return value


def load_bundle(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        forbidden = [name for name in data.files if any(token in name.lower() for token in ("label", "target", "correct"))]
        if forbidden:
            raise ValueError(f"target-like field in target-free bundle {path}: {forbidden}")
        if str(data["schema"].item()) != "residual_graph_deem_target_free_bundle_v1":
            raise ValueError(f"bundle schema mismatch: {path}")
        output = {
            "cell_id": str(data["cell_id"].item()),
            "X_raw": np.asarray(data["X_raw"], dtype=np.float64),
            "feature_names": tuple(str(value) for value in data["feature_names"].tolist()),
            "row_ids": tuple(str(value) for value in data["row_id"].tolist()),
            "group_ids": tuple(str(value) for value in data["group_id"].tolist()),
            "length": np.asarray(data["raw_trace_length"], dtype=np.float64),
            "dataset_family": str(data["dataset_family"].item()),
        }
    n = len(output["row_ids"])
    if len(output["group_ids"]) != n or len(output["length"]) != n or len(output["X_raw"]) != n:
        raise ValueError(f"bundle alignment mismatch: {path}")
    if len(set(output["row_ids"])) != n:
        raise ValueError(f"duplicate row IDs: {path}")
    return output


def load_b3_atomic(baseline_dir: Path, cell_id: str, expected_names: Sequence[str], n: int) -> tuple[np.ndarray, list[dict]]:
    values = []
    artifacts = []
    expected = tuple(expected_names)
    for seed in SEEDS:
        path = baseline_dir / "fits" / cell_id / f"B3__seed{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(value) for value in data["feature_names"].tolist())
            contribution = np.asarray(data["contributions"], dtype=np.float64)
        if contribution.shape != (n, len(names)) or set(names) != set(expected):
            raise ValueError(f"B3 atomic shape/inventory mismatch: {cell_id} seed {seed}")
        lookup = {name: index for index, name in enumerate(names)}
        values.append(contribution[:, [lookup[name] for name in expected]])
        artifacts.append({"path": path.as_posix(), "sha256": sha256_file(path)})
    return np.mean(np.stack(values, axis=0), axis=0), artifacts


def select_panel(
    cell_id: str,
    row_ids: Sequence[str],
    group_ids: Sequence[str],
    panel: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        by_group.setdefault(str(group), []).append(index)
    if len(by_group) < N_GROUPS:
        raise ValueError(f"{cell_id} has only {len(by_group)} source groups")
    selected_groups = sorted(
        by_group,
        key=lambda group: stable_hex(SCHEMA, "panel-group", panel, cell_id, group),
    )[:N_GROUPS]
    indices = []
    selected_rows = []
    for group in selected_groups:
        candidates = by_group[group]
        index = min(
            candidates,
            key=lambda item: stable_hex(SCHEMA, "panel-row", panel, cell_id, group, row_ids[item]),
        )
        indices.append(index)
        selected_rows.append(str(row_ids[index]))
    return np.asarray(indices, dtype=np.int64), tuple(selected_groups), tuple(selected_rows)


def balanced_folds(cell_id: str, panel: int, groups: Sequence[str]) -> np.ndarray:
    order = sorted(
        range(len(groups)),
        key=lambda index: stable_hex(SCHEMA, "fold", panel, cell_id, groups[index]),
    )
    folds = np.empty(len(groups), dtype=np.int64)
    for rank, index in enumerate(order):
        folds[index] = rank % N_FOLDS
    counts = np.bincount(folds, minlength=N_FOLDS)
    if counts.max() - counts.min() > 1:
        raise AssertionError("fold balancing failed")
    return folds


def donor_transform(values: np.ndarray, lengths: np.ndarray, train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_length = np.log1p(lengths)
    mean_l = float(log_length[train].mean())
    scale_l = max(float(log_length[train].std()), 1e-12)
    z_train = (log_length[train] - mean_l) / scale_l
    z_held = (log_length[held] - mean_l) / scale_l
    design_train = np.column_stack([np.ones(len(train)), z_train, z_train ** 2, z_train ** 3])
    design_held = np.column_stack([np.ones(len(held)), z_held, z_held ** 2, z_held ** 3])
    coefficient = np.linalg.lstsq(design_train, values[train], rcond=None)[0]
    residual_train = values[train] - design_train @ coefficient
    residual_held = values[held] - design_held @ coefficient
    mean = residual_train.mean(axis=0)
    scale = residual_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (residual_train - mean) / scale, (residual_held - mean) / scale


def ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_held: np.ndarray) -> np.ndarray:
    gram = X_train.T @ X_train
    coefficient = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(X_train.shape[1]),
        X_train.T @ y_train,
    )
    return X_held @ coefficient


def predictor_sets(target: int) -> dict[str, tuple[tuple[int, ...], ...]]:
    peers = tuple(index for index in range(len(FEATURES)) if index != target)
    same_source = tuple(index for index in peers if SOURCE_OF[index] == SOURCE_OF[target])
    same_operator = tuple(index for index in peers if OPERATOR_OF[index] == OPERATOR_OF[target])
    neither = tuple(index for index in peers if index not in set(same_source + same_operator))
    union = tuple(sorted(set(same_source + same_operator)))
    if not (len(same_source) == len(same_operator) == 2 and len(neither) == len(union) == 4):
        raise AssertionError("3x3 peer geometry mismatch")
    return {
        "source": (same_source,),
        "operator": (same_operator,),
        "union": (union,),
        "all": (peers,),
        "random2": tuple(itertools.combinations(peers, 2)),
        "random4": tuple(itertools.combinations(peers, 4)),
        "source_plus_random2": tuple(tuple(sorted(same_source + choice)) for choice in itertools.combinations(neither, 2)),
        "operator_plus_random2": tuple(tuple(sorted(same_operator + choice)) for choice in itertools.combinations(neither, 2)),
    }


def evaluate_panel(values: np.ndarray, lengths: np.ndarray, folds: np.ndarray) -> tuple[list[dict], dict[str, float]]:
    transformed = []
    oof = np.zeros_like(values, dtype=np.float64)
    for fold in range(N_FOLDS):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        donor, held_values = donor_transform(values, lengths, train, held)
        transformed.append((train, held, donor, held_values))
        oof[held] = held_values

    rows = []
    for target in range(len(FEATURES)):
        sets = predictor_sets(target)
        denominators = 0.0
        sse = {name: np.zeros(len(subsets), dtype=np.float64) for name, subsets in sets.items()}
        for _, _, donor, held_values in transformed:
            y_train = donor[:, target]
            y_held = held_values[:, target]
            denominators += float(y_held @ y_held)
            for name, subsets in sets.items():
                for subset_index, subset in enumerate(subsets):
                    columns = list(subset)
                    prediction = ridge_predict(donor[:, columns], y_train, held_values[:, columns])
                    error = y_held - prediction
                    sse[name][subset_index] += float(error @ error)
        if denominators <= 1e-12:
            raise ValueError(f"degenerate held target: {FEATURES[target]}")
        metrics = {
            name: float(np.mean(1.0 - errors / denominators))
            for name, errors in sse.items()
        }
        rows.append({
            "target": FEATURES[target],
            "source_axis": SOURCE_OF[target],
            "operator_axis": OPERATOR_OF[target],
            **{f"r2_{name}": value for name, value in metrics.items()},
            "delta_operator_given_source": metrics["union"] - metrics["source"],
            "delta_source_given_operator": metrics["union"] - metrics["operator"],
            "delta_operator_specific_vs_random_add": metrics["union"] - metrics["source_plus_random2"],
            "delta_source_specific_vs_random_add": metrics["union"] - metrics["operator_plus_random2"],
            "delta_union_vs_random4": metrics["union"] - metrics["random4"],
            "delta_source_vs_random2": metrics["source"] - metrics["random2"],
            "delta_operator_vs_random2": metrics["operator"] - metrics["random2"],
            "delta_all_vs_union": metrics["all"] - metrics["union"],
        })

    corr = np.corrcoef(oof, rowvar=False)
    categories: dict[str, list[float]] = {"same_source": [], "same_operator": [], "neither": []}
    for left, right in itertools.combinations(range(len(FEATURES)), 2):
        if SOURCE_OF[left] == SOURCE_OF[right]:
            category = "same_source"
        elif OPERATOR_OF[left] == OPERATOR_OF[right]:
            category = "same_operator"
        else:
            category = "neither"
        categories[category].append(abs(float(corr[left, right])))
    geometry = {f"mean_abs_corr_{name}": float(np.mean(value)) for name, value in categories.items()}
    return rows, geometry


def equal_family_values(rows: Sequence[Mapping[str, object]], key: str, *, representation: str, primary: bool) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["representation"] == representation and bool(row["primary_k50"]) == primary
    ]
    families = sorted({str(row["dataset_family"]) for row in selected})
    output = {}
    for family in families:
        family_rows = [row for row in selected if row["dataset_family"] == family]
        output[family] = float(np.mean([float(row[key]) for row in family_rows]))
    return output


def exact_signflip(values: Mapping[str, float]) -> dict[str, object]:
    families = tuple(sorted(values))
    observed_values = np.asarray([values[family] for family in families], dtype=np.float64)
    observed = float(observed_values.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(families)):
        null.append(float(np.mean(observed_values * np.asarray(signs))))
    null_values = np.asarray(null)
    p_greater = float(np.mean(null_values >= observed - 1e-15))
    p_two_sided = float(np.mean(np.abs(null_values) >= abs(observed) - 1e-15))
    lofo = {
        family: float(np.mean([value for name, value in values.items() if name != family]))
        for family in families
    }
    rng = np.random.default_rng(20260825)
    bootstrap = observed_values[rng.integers(0, len(families), size=(20_000, len(families)))].mean(axis=1)
    return {
        "n_dataset_families": len(families),
        "observed_equal_family_mean": observed,
        "exact_one_sided_p": p_greater,
        "exact_two_sided_p": p_two_sided,
        "family_values": {family: float(values[family]) for family in families},
        "bootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "lofo": lofo,
        "lofo_min": float(min(lofo.values())),
        "lofo_max": float(max(lofo.values())),
        "lofo_positive": int(sum(value > 0 for value in lofo.values())),
    }


def aggregate(rows: Sequence[Mapping[str, object]], geometry_rows: Sequence[Mapping[str, object]]) -> dict:
    metrics = (
        "r2_source", "r2_operator", "r2_union", "r2_all", "r2_random2", "r2_random4",
        "r2_source_plus_random2", "r2_operator_plus_random2",
        "delta_operator_given_source", "delta_source_given_operator",
        "delta_operator_specific_vs_random_add", "delta_source_specific_vs_random_add",
        "delta_union_vs_random4", "delta_source_vs_random2", "delta_operator_vs_random2",
        "delta_all_vs_union",
    )
    contrasts = tuple(metric for metric in metrics if metric.startswith("delta_"))
    output: dict[str, object] = {}
    for scope, primary in (("primary_k50_23_cells", True), ("losnet_k1000_sensitivity", False)):
        output[scope] = {}
        for representation in ("raw_physical_core", "b3_atomic_core"):
            record: dict[str, object] = {}
            for metric in metrics:
                family_values = equal_family_values(rows, metric, representation=representation, primary=primary)
                record[metric] = {
                    "equal_dataset_family_mean": float(np.mean(list(family_values.values()))),
                    "family_values": family_values,
                }
            geometry = {}
            for metric in ("mean_abs_corr_same_source", "mean_abs_corr_same_operator", "mean_abs_corr_neither"):
                family_values = equal_family_values(
                    geometry_rows, metric, representation=representation, primary=primary
                )
                geometry[metric] = {
                    "equal_dataset_family_mean": float(np.mean(list(family_values.values()))),
                    "family_values": family_values,
                }
            record["crossfit_geometry"] = geometry
            if primary:
                record["exact_family_signflip"] = {
                    contrast: exact_signflip(
                        equal_family_values(rows, contrast, representation=representation, primary=True)
                    )
                    for contrast in contrasts
                }
            else:
                record["inference_caveat"] = "one held sensitivity cell; no family-level p-value"
            output[scope][representation] = record
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--registry", type=Path, default=root_default / "configs" / "residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--bundle-dir", type=Path, default=root_default / "local_cache" / "deem_b3_moe_v1" / "bundles")
    parser.add_argument("--baseline-dir", type=Path, default=root_default / "local_cache" / "deem_b3_moe_v1" / "b3_frozen")
    parser.add_argument("--out-dir", type=Path, default=root_default / "local_cache" / "deem_b3_moe_v1" / "crossed_core_independent_v1")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    rows: list[dict] = []
    geometry_rows: list[dict] = []
    panel_rows: list[dict] = []
    input_artifacts: list[dict] = []
    b3_artifact_seen: set[str] = set()

    for cell_record in registry["cells"]:
        cell_id = str(cell_record["cell_id"])
        bundle_path = args.bundle_dir / f"{cell_id}.npz"
        bundle = load_bundle(bundle_path)
        names = tuple(bundle["feature_names"])
        lookup = {name: index for index, name in enumerate(names)}
        missing = [name for name in FEATURES if name not in lookup]
        if missing:
            raise ValueError(f"universal 3x3 core missing in {cell_id}: {missing}")
        raw = np.asarray(bundle["X_raw"], dtype=np.float64)[:, [lookup[name] for name in FEATURES]]
        b3_full, b3_artifacts = load_b3_atomic(
            args.baseline_dir, cell_id, names, len(raw)
        )
        b3 = b3_full[:, [lookup[name] for name in FEATURES]]
        input_artifacts.append({"path": bundle_path.as_posix(), "sha256": sha256_file(bundle_path)})
        for artifact in b3_artifacts:
            if artifact["path"] not in b3_artifact_seen:
                input_artifacts.append(artifact)
                b3_artifact_seen.add(artifact["path"])

        for panel in range(N_PANELS):
            indices, selected_groups, selected_rows = select_panel(
                cell_id, bundle["row_ids"], bundle["group_ids"], panel
            )
            folds = balanced_folds(cell_id, panel, selected_groups)
            panel_rows.append({
                "cell_id": cell_id,
                "dataset_family": bundle["dataset_family"],
                "panel": panel,
                "n_groups": len(selected_groups),
                "n_rows": len(indices),
                "selected_group_sha256": canonical_hash(list(selected_groups)),
                "selected_row_sha256": canonical_hash(list(selected_rows)),
                "fold_assignment_sha256": canonical_hash(folds.tolist()),
                "fold_counts": ";".join(str(value) for value in np.bincount(folds, minlength=N_FOLDS)),
            })
            for representation, values in (
                ("raw_physical_core", raw),
                ("b3_atomic_core", b3),
            ):
                target_rows, geometry = evaluate_panel(
                    values[indices], np.asarray(bundle["length"])[indices], folds
                )
                prefix = {
                    "cell_id": cell_id,
                    "dataset_family": bundle["dataset_family"],
                    "primary_k50": cell_id != LOS_CELL,
                    "representation": representation,
                    "panel": panel,
                }
                for target_row in target_rows:
                    rows.append({**prefix, **target_row})
                geometry_rows.append({**prefix, **geometry})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "PER_TARGET.csv", rows)
    write_csv(args.out_dir / "CROSSFIT_GEOMETRY.csv", geometry_rows)
    write_csv(args.out_dir / "PANEL_MANIFEST.csv", panel_rows)
    summary = {
        "schema": SCHEMA,
        "labels_accessed": False,
        "physical_grid": {
            "sources": list(SOURCES),
            "operators": list(OPERATORS),
            "feature_order": list(FEATURES),
        },
        "protocol": {
            "n_panels": N_PANELS,
            "n_groups_per_cell_panel": N_GROUPS,
            "one_row_per_source_question": True,
            "n_folds": N_FOLDS,
            "fold_assignment": "stable-hash balanced ranks",
            "length_adjustment": "donor-fold OLS on [1,z(log1pL),z^2,z^3]",
            "scaling": "donor-fold mean and standard deviation only",
            "regressor": f"ridge alpha={RIDGE_ALPHA}, no intercept after donor centering",
            "random_controls": "exact exhaustive cardinality-matched subsets",
            "aggregation": "targets -> cells -> dataset families -> equal family mean",
            "inference": "exact sign flips over the seven primary dataset-family means",
        },
        "scope": {
            "primary": "23 K=50 lineage cells, excluding LOS-Net",
            "sensitivity": "LOS-Net K=1000 acquisition lineage only",
            "note": "the physical 3x3 grid itself does not include saved-support entropy",
        },
        "representation_caveat": {
            "raw_physical_core": "direct source-by-operator measurements",
            "b3_atomic_core": "mean-of-five c_j outputs; each nonlinear c_j sees its full historical provenance family",
        },
        "results": aggregate(rows, geometry_rows),
    }
    write_json(args.out_dir / "SUMMARY.json", summary)

    output_paths = [
        args.out_dir / "PER_TARGET.csv",
        args.out_dir / "CROSSFIT_GEOMETRY.csv",
        args.out_dir / "PANEL_MANIFEST.csv",
        args.out_dir / "SUMMARY.json",
    ]
    script_path = Path(__file__).resolve()
    freeze = {
        "schema": SCHEMA,
        "labels_accessed": False,
        "script": script_path.as_posix(),
        "script_sha256": sha256_file(script_path),
        "registry": args.registry.resolve().as_posix(),
        "registry_sha256": sha256_file(args.registry),
        "input_artifacts": sorted(input_artifacts, key=lambda value: value["path"]),
        "output_artifacts": [
            {"path": path.resolve().as_posix(), "sha256": sha256_file(path)} for path in output_paths
        ],
        "config_sha256": canonical_hash(summary["protocol"]),
    }
    write_json(args.out_dir / "FREEZE.json", freeze)
    print(json.dumps({
        "status": "PASS_LABEL_FREE_INDEPENDENT_CROSSED_CORE",
        "out_dir": args.out_dir.resolve().as_posix(),
        "script_sha256": freeze["script_sha256"],
        "summary_sha256": sha256_file(args.out_dir / "SUMMARY.json"),
        "n_target_rows": len(rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
