#!/usr/bin/env python3
"""Retrospective/C-tier reliability audit for B3 residual corrections.

The official eight-cell screen is the only source of labels used to select the
primary gate.  The selected rule is serialized before the remaining sixteen
label sidecars are loaded.  The held panel is not a clean confirmation panel:
all natural targets were already open elsewhere in the project.  This script
also records supervised, per-target ceilings; those are diagnostics, not
deployable methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "local_cache/deem_b3_moe_v1/iupgrd_boost_all24_v1"
DEFAULT_CACHE = ROOT / "local_cache/deem_b3_moe_v1"
DEFAULT_CONFIG = ROOT / "configs/deem_b3_iupgrd_boost_v1.json"
DEFAULT_OUTPUT = ROOT / "local_cache/deem_b3_moe_v1/residual_reliability_ceiling_v1"


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    scale = float(np.std(x))
    return (x - float(np.mean(x))) / (scale if scale > 1e-12 else 1.0)


def _percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return (rankdata(x, method="average") - 0.5) / len(x)


def _load_labels(cache: Path, cell: str) -> np.ndarray:
    with np.load(cache / "label_sidecars" / f"{cell}.npz", allow_pickle=False) as data:
        return np.asarray(data["y_H"], dtype=np.int8)


def _load_diagnostics(cache: Path, run: Path, cell: str) -> dict:
    state_path = run / "states" / f"{cell}.npz"
    score_path = (
        run
        / "scores/E1_B3_ORTH_IUPGRD_FULL"
        / cell
        / "E1_B3_ORTH_IUPGRD_FULL.npz"
    )
    with np.load(state_path, allow_pickle=False) as state, np.load(
        score_path, allow_pickle=False
    ) as scored:
        baseline = np.asarray(state["baseline_z"], dtype=float)
        iu = _zscore(np.asarray(state["iu_score_aligned"], dtype=float))
        pgrd = np.asarray(scored["correction_z"], dtype=float)
        residuals = np.asarray(state["iu_family_residuals"], dtype=float)
        graph = csr_matrix(
            (
                state["graph_data"],
                state["graph_indices"],
                state["graph_indptr"],
            ),
            shape=tuple(int(v) for v in state["graph_shape"]),
        )
        family_order = tuple(str(v) for v in state["family_order"].tolist())
        global_order = tuple(str(v) for v in state["global_family_order"].tolist())

    seed_scores = []
    for seed in range(5):
        with np.load(
            cache / "b3_frozen/fits" / cell / f"B3__seed{seed}.npz",
            allow_pickle=False,
        ) as member:
            seed_scores.append(np.asarray(member["score"], dtype=float))
    seed_instability = np.std(np.stack(seed_scores), axis=0)
    family_novelty = np.sqrt(np.mean(np.square(residuals), axis=1))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    neighbor_mean = np.asarray(graph.dot(baseline)).ravel() / np.maximum(degree, 1e-12)
    # Rank the raw non-negative quantities.  Z-scoring first can collapse tiny
    # differences after subtracting a large mean and introduce artificial ties.
    graph_novelty = np.abs(baseline - neighbor_mean)

    metadata = json.loads((run / "states" / f"{cell}.json").read_text())
    diagnostics = {
        "cell": cell,
        "dataset_family": str(metadata["dataset_family"]),
        "b3": baseline,
        "iu": iu,
        "iu_residual": iu - baseline,
        "pgrd": pgrd,
        "b3_rank": _percentile(baseline),
        "disagreement_rank": _percentile(np.abs(iu - baseline)),
        "pgrd_magnitude_rank": _percentile(np.abs(pgrd)),
        "seed_instability_rank": _percentile(seed_instability),
        "graph_novelty_rank": _percentile(graph_novelty),
        "family_novelty_rank": _percentile(family_novelty),
        "residuals": residuals,
        "family_order": family_order,
        "global_family_order": global_order,
    }
    arrays = (
        value
        for key, value in diagnostics.items()
        if isinstance(value, np.ndarray)
        and key not in {"family_order", "global_family_order"}
    )
    if any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays):
        raise ValueError(f"non-finite diagnostic: {cell}")
    return diagnostics


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def _summarize(
    cells: Sequence[str],
    diagnostics: Mapping[str, dict],
    targets: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
) -> dict:
    rows = []
    for cell in cells:
        row = _metrics(targets[cell], scores[cell])
        row.update(cell=cell, dataset_family=diagnostics[cell]["dataset_family"])
        rows.append(row)
    families = sorted({row["dataset_family"] for row in rows})
    return {
        "equal_family_auroc": float(
            np.mean(
                [
                    np.mean([r["auroc"] for r in rows if r["dataset_family"] == family])
                    for family in families
                ]
            )
        ),
        "equal_family_auprc": float(
            np.mean(
                [
                    np.mean([r["auprc"] for r in rows if r["dataset_family"] == family])
                    for family in families
                ]
            )
        ),
        "cell_macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "cell_macro_auprc": float(np.mean([row["auprc"] for row in rows])),
        "rows": rows,
    }


def _gate(diagnostic: Mapping[str, np.ndarray], gate_id: str) -> np.ndarray:
    if gate_id == "all":
        return np.ones_like(diagnostic["b3"])
    if gate_id == "align_b3":
        return (np.sign(diagnostic["active_residual"]) == np.sign(diagnostic["b3"])).astype(float)
    if gate_id == "oppose_b3":
        return (np.sign(diagnostic["active_residual"]) != np.sign(diagnostic["b3"])).astype(float)
    feature, side, quantile_text = gate_id.rsplit("_", 2)
    quantile = float(quantile_text)
    values = np.asarray(diagnostic[feature], dtype=float)
    if side == "high":
        return (values >= quantile).astype(float)
    if side == "low":
        return (values <= 1.0 - quantile).astype(float)
    raise ValueError(gate_id)


def _candidate_score(diagnostic: dict, candidate: Mapping[str, object]) -> np.ndarray:
    source = str(candidate["residual"])
    residual = np.asarray(diagnostic[source], dtype=float)
    local = dict(diagnostic)
    local["active_residual"] = residual
    gate = _gate(local, str(candidate["gate"]))
    return np.asarray(diagnostic["b3"], dtype=float) + float(candidate["alpha"]) * residual * gate


def _select_gate(
    screen: Sequence[str],
    diagnostics: Mapping[str, dict],
    targets: Mapping[str, np.ndarray],
    *,
    expanded: bool,
) -> tuple[dict, list[dict]]:
    baseline = _summarize(
        screen, diagnostics, targets, {cell: diagnostics[cell]["b3"] for cell in screen}
    )
    gate_ids = ["all", "align_b3", "oppose_b3"]
    features = ["b3_rank", "disagreement_rank"]
    if expanded:
        features += [
            "pgrd_magnitude_rank",
            "seed_instability_rank",
            "graph_novelty_rank",
            "family_novelty_rank",
        ]
    for feature in features:
        for quantile in (0.5, 0.65, 0.75, 0.85):
            gate_ids.extend(
                [f"{feature}_high_{quantile}", f"{feature}_low_{quantile}"]
            )
    candidates = []
    for residual in ("iu_residual", "pgrd"):
        for alpha in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
            for gate_id in gate_ids:
                candidate = {"residual": residual, "alpha": alpha, "gate": gate_id}
                scores = {
                    cell: _candidate_score(diagnostics[cell], candidate) for cell in screen
                }
                summary = _summarize(screen, diagnostics, targets, scores)
                candidate.update(
                    screen_equal_family_auroc=summary["equal_family_auroc"],
                    screen_delta_auroc=(
                        summary["equal_family_auroc"] - baseline["equal_family_auroc"]
                    ),
                    screen_delta_auprc=(
                        summary["equal_family_auprc"] - baseline["equal_family_auprc"]
                    ),
                )
                candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            -float(row["screen_equal_family_auroc"]),
            str(row["residual"]),
            float(row["alpha"]),
            str(row["gate"]),
        )
    )
    return dict(candidates[0]), candidates


def _jackknife_reliability(run: Path, diagnostic: Mapping[str, object]) -> dict:
    family = str(diagnostic["dataset_family"])
    with np.load(run / "calibrations" / f"held_{family}.npz", allow_pickle=False) as cal:
        donor_c = np.asarray(cal["donor_moment_c"], dtype=float)
        donor_groups = np.asarray(cal["donor_dataset_families"], dtype=str)
    groups = sorted(set(donor_groups.tolist()))
    baseline = np.asarray(diagnostic["b3"], dtype=float)
    residuals = np.asarray(diagnostic["residuals"], dtype=float)
    global_order = tuple(diagnostic["global_family_order"])
    local_indices = [global_order.index(name) for name in diagnostic["family_order"]]
    design = np.column_stack([np.ones(len(baseline)), baseline])
    corrections = []
    for omitted in groups:
        group_means = [
            np.mean(donor_c[donor_groups == group], axis=0)
            for group in groups
            if group != omitted
        ]
        direction = -np.mean(group_means, axis=0)
        raw = residuals @ direction[local_indices]
        projected = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
        corrections.append(projected / np.std(projected) / len(local_indices))
    values = np.stack(corrections, axis=1)
    sign_consensus = np.abs(np.mean(np.sign(values), axis=1))
    relative_sd = np.std(values, axis=1) / (np.mean(np.abs(values), axis=1) + 1e-12)
    reliability = sign_consensus / (1.0 + relative_sd)
    return {
        "reliability": reliability,
        "sign_consensus": sign_consensus,
        "relative_sd": relative_sd,
        "mean_correction": np.mean(values, axis=1),
        "n_jackknives": len(groups),
    }


def _oracle_ceiling(
    cells: Sequence[str], diagnostics: Mapping[str, dict], targets: Mapping[str, np.ndarray]
) -> dict:
    rows = []
    for cell in cells:
        d = diagnostics[cell]
        y = targets[cell]
        baseline = np.asarray(d["b3"], dtype=float)
        base_auc = float(roc_auc_score(y, baseline))
        iu_best = max(
            float(roc_auc_score(y, baseline + alpha * d["iu_residual"]))
            for alpha in np.linspace(-1.0, 2.0, 121)
        )
        pgrd_best = max(
            float(roc_auc_score(y, baseline + alpha * d["pgrd"]))
            for alpha in np.linspace(-4.0, 4.0, 161)
        )
        joint_best = base_auc
        for alpha in np.linspace(-0.5, 1.5, 21):
            for beta in np.linspace(-2.0, 2.0, 21):
                joint_best = max(
                    joint_best,
                    float(
                        roc_auc_score(
                            y,
                            baseline + alpha * d["iu_residual"] + beta * d["pgrd"],
                        )
                    ),
                )
        rows.append(
            {
                "cell": cell,
                "dataset_family": d["dataset_family"],
                "iu_delta": iu_best - base_auc,
                "pgrd_delta": pgrd_best - base_auc,
                "joint_delta": joint_best - base_auc,
            }
        )
    families = sorted({row["dataset_family"] for row in rows})
    result = {"rows": rows, "scientific_role": "per_target_supervised_in_sample_ceiling"}
    for key in ("iu_delta", "pgrd_delta", "joint_delta"):
        result[f"equal_family_{key}"] = float(
            np.mean(
                [
                    np.mean([row[key] for row in rows if row["dataset_family"] == family])
                    for family in families
                ]
            )
        )
    return result


def _delta(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in (
            "equal_family_auroc",
            "equal_family_auprc",
            "cell_macro_auroc",
            "cell_macro_auprc",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    screen = tuple(sorted(str(cell) for cell in config["screen_cells"]))
    all_cells = tuple(sorted(path.stem for path in (args.run / "states").glob("*.npz")))
    held = tuple(cell for cell in all_cells if cell not in set(screen))
    if len(screen) != 8 or len(held) != 16:
        raise ValueError("expected the official 8/16 split")

    # Target-free construction happens for all cells before any labels are opened.
    diagnostics = {
        cell: _load_diagnostics(args.cache, args.run, cell) for cell in all_cells
    }

    # Selection phase: only the official screen sidecars are loaded.
    screen_targets = {cell: _load_labels(args.cache, cell) for cell in screen}
    primary, primary_menu = _select_gate(
        screen, diagnostics, screen_targets, expanded=False
    )
    frozen = {
        "schema": "deem_b3_residual_reliability_frozen_gate_v1",
        "tier": "retrospective_C",
        "selection_cells": list(screen),
        "held_cells": list(held),
        "gate": primary,
        "held_labels_loaded_before_freeze": False,
        "candidate_count": len(primary_menu),
    }
    (args.output / "FROZEN_GATE.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )

    # One held-panel scoring pass for the frozen primary.
    held_targets = {cell: _load_labels(args.cache, cell) for cell in held}
    held_baseline = _summarize(
        held,
        diagnostics,
        held_targets,
        {cell: diagnostics[cell]["b3"] for cell in held},
    )
    held_primary = _summarize(
        held,
        diagnostics,
        held_targets,
        {cell: _candidate_score(diagnostics[cell], primary) for cell in held},
    )

    # This broader menu was requested after the primary held labels had opened.
    # It is retained only as a post-hoc sensitivity and cannot replace primary.
    expanded, _ = _select_gate(screen, diagnostics, screen_targets, expanded=True)
    held_expanded = _summarize(
        held,
        diagnostics,
        held_targets,
        {cell: _candidate_score(diagnostics[cell], expanded) for cell in held},
    )

    jackknife = {
        cell: _jackknife_reliability(args.run, diagnostics[cell]) for cell in all_cells
    }
    jackknife_scores = {
        cell: diagnostics[cell]["b3"]
        + diagnostics[cell]["pgrd"] * jackknife[cell]["reliability"]
        for cell in held
    }
    held_jackknife = _summarize(
        held, diagnostics, held_targets, jackknife_scores
    )
    helpful = np.concatenate(
        [
            ((2 * held_targets[cell] - 1) * diagnostics[cell]["pgrd"] > 0).astype(int)
            for cell in held
        ]
    )
    reliability = np.concatenate([jackknife[cell]["reliability"] for cell in held])
    reliability_auc = float(roc_auc_score(helpful, reliability))

    oracle = _oracle_ceiling(held, diagnostics, held_targets)
    result = {
        "schema": "deem_b3_residual_reliability_ceiling_v1",
        "tier": "retrospective_C",
        "confirmation_status": "not_confirmation_targets_previously_opened",
        "primary": {
            "gate": primary,
            "held_baseline": {k: v for k, v in held_baseline.items() if k != "rows"},
            "held_candidate": {k: v for k, v in held_primary.items() if k != "rows"},
            "held_delta": _delta(held_primary, held_baseline),
            "passes_0p0025_equal_family_auroc": (
                _delta(held_primary, held_baseline)["equal_family_auroc"] >= 0.0025
            ),
        },
        "expanded_posthoc_sensitivity": {
            "gate": expanded,
            "held_delta": _delta(held_expanded, held_baseline),
            "selection_timeline": "expanded_after_primary_held_labels_opened",
        },
        "donor_jackknife_posthoc": {
            "definition": "abs_mean_sign_over_one_plus_relative_sd",
            "held_delta": _delta(held_jackknife, held_baseline),
            "pointwise_helpfulness_auc": reliability_auc,
            "selection_timeline": "requested_after_primary_held_labels_opened",
        },
        "supervised_oracle": oracle,
        "decision": "NO_TRANSFERABLE_RELIABILITY_SIGNAL_AT_REQUIRED_SCALE",
    }
    (args.output / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    with (args.output / "PER_CELL.csv").open("w", newline="") as handle:
        fields = [
            "cell",
            "dataset_family",
            "b3_auroc",
            "primary_auroc",
            "delta_auroc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        base_rows = {row["cell"]: row for row in held_baseline["rows"]}
        primary_rows = {row["cell"]: row for row in held_primary["rows"]}
        for cell in held:
            writer.writerow(
                {
                    "cell": cell,
                    "dataset_family": diagnostics[cell]["dataset_family"],
                    "b3_auroc": base_rows[cell]["auroc"],
                    "primary_auroc": primary_rows[cell]["auroc"],
                    "delta_auroc": primary_rows[cell]["auroc"] - base_rows[cell]["auroc"],
                }
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
