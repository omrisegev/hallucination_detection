"""Vectorized grouped AUROC/AUPRC bootstrap for the EDIS reconstruction lane.

Sampling a source question with replacement is exactly equivalent to assigning
an integer weight to every row from that question.  This module keeps the
registered grouped, paired estimand while evaluating those integer-weighted
replicates in NumPy chunks.  It intentionally implements only the two frozen
EDIS metrics; the generic evaluator's AURC work would be wasted here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_IDS = ("auroc", "auprc")
SCHEMA_VERSION = "reconstruction-edis-vectorized-grouped-bootstrap-v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_cell(
    *,
    labels: Sequence[int],
    scores_by_method: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
    reference_method: str,
    canonical_group_order: bool,
) -> dict[str, Any]:
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or not np.isin(raw_labels, (0, 1)).all():
        raise ValueError("labels must be a binary vector")
    y = raw_labels.astype(np.int8, copy=False)
    groups = np.asarray(group_ids, dtype=str)
    if groups.shape != y.shape or len(y) < 2 or len(np.unique(y)) != 2:
        raise ValueError("labels/groups must be equal-length and contain both classes")
    first_seen: list[str] = []
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        if not group:
            raise ValueError("group IDs must be nonempty")
        if group not in members:
            first_seen.append(group)
            members[group] = []
        members[group].append(index)
    roster = tuple(sorted(members) if canonical_group_order else first_seen)
    if len(roster) < 2:
        raise ValueError("grouped bootstrap requires at least two source groups")
    group_position = {group: index for index, group in enumerate(roster)}
    row_group_index = np.asarray(
        [group_position[group] for group in groups.tolist()], dtype=np.int32
    )
    group_pos = np.bincount(
        row_group_index, weights=y, minlength=len(roster)
    ).astype(np.int64)
    group_total = np.bincount(
        row_group_index, minlength=len(roster)
    ).astype(np.int64)
    methods = tuple(sorted(scores_by_method))
    if not methods or reference_method not in methods:
        raise ValueError("reference method is absent from scores_by_method")
    scores: dict[str, np.ndarray] = {}
    for method_id in methods:
        if not method_id:
            raise ValueError("method IDs must be nonempty")
        score = np.asarray(scores_by_method[method_id], dtype=np.float64)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError(f"{method_id}: invalid score vector")
        scores[method_id] = score
    return {
        "labels": y,
        "groups": groups,
        "roster": roster,
        "row_group_index": row_group_index,
        "group_pos": group_pos,
        "group_total": group_total,
        "members": {
            group: np.asarray(members[group], dtype=np.int64)
            for group in roster
        },
        "methods": methods,
        "scores": scores,
    }


def _draw_counts(
    *, draws: int, n_groups: int, rng: np.random.Generator
) -> np.ndarray:
    selected = rng.integers(0, n_groups, size=(draws, n_groups))
    counts = np.zeros((draws, n_groups), dtype=np.int16)
    rows = np.repeat(np.arange(draws, dtype=np.int64), n_groups)
    np.add.at(counts, (rows, selected.reshape(-1)), 1)
    return counts


def _weighted_draw_metrics(
    *,
    labels: np.ndarray,
    score: np.ndarray,
    row_group_index: np.ndarray,
    counts: np.ndarray,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AUROC/AUPRC for integer-weighted replicates.

    Exact score ties are collapsed before cumulative precision/ROC operations,
    matching scikit-learn's threshold semantics and making results invariant to
    row ordering within a tie.
    """

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_labels = labels[order].astype(np.float64, copy=False)
    sorted_groups = row_group_index[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    )
    n_draws = counts.shape[0]
    auroc = np.full(n_draws, np.nan, dtype=np.float64)
    auprc = np.full(n_draws, np.nan, dtype=np.float64)
    for begin in range(0, n_draws, chunk_size):
        end = min(begin + chunk_size, n_draws)
        row_weight = np.asarray(
            counts[begin:end, sorted_groups], dtype=np.float64
        )
        positive_rows = row_weight * sorted_labels[None, :]
        positive_inc = np.add.reduceat(positive_rows, starts, axis=1)
        total_inc = np.add.reduceat(row_weight, starts, axis=1)
        negative_inc = total_inc - positive_inc
        cumulative_positive = np.cumsum(positive_inc, axis=1)
        cumulative_total = np.cumsum(total_inc, axis=1)
        cumulative_negative = cumulative_total - cumulative_positive
        total_positive = cumulative_positive[:, -1]
        total_negative = cumulative_negative[:, -1]
        valid = (total_positive > 0.0) & (total_negative > 0.0)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0.0,
        )
        ap_numerator = np.sum(positive_inc * precision, axis=1)
        negatives_below_plus_half_tie = (
            total_negative[:, None]
            - cumulative_negative
            + 0.5 * negative_inc
        )
        auc_numerator = np.sum(
            positive_inc * negatives_below_plus_half_tie, axis=1
        )
        local_ap = np.divide(
            ap_numerator,
            total_positive,
            out=np.full_like(total_positive, np.nan),
            where=valid,
        )
        local_auc = np.divide(
            auc_numerator,
            total_positive * total_negative,
            out=np.full_like(total_positive, np.nan),
            where=valid,
        )
        auprc[begin:end] = local_ap
        auroc[begin:end] = local_auc
    return auroc, auprc


def _point_metrics(labels: np.ndarray, scores: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        method_id: {
            "auroc": float(roc_auc_score(labels, score)),
            "auprc": float(average_precision_score(labels, score)),
        }
        for method_id, score in scores.items()
    }


def _summaries(
    *,
    methods: Sequence[str],
    point: Mapping[str, Mapping[str, float]],
    draws_by_method: Mapping[str, Mapping[str, np.ndarray]],
    valid: np.ndarray,
    reference_method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid_draws = int(np.sum(valid))
    if valid_draws == 0:
        raise RuntimeError("every grouped bootstrap draw was single-class")
    metrics: dict[str, Any] = {}
    for method_id in methods:
        metrics[method_id] = {}
        for metric_id in METRIC_IDS:
            values = np.asarray(draws_by_method[method_id][metric_id][valid], dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError("valid EDIS bootstrap metric contains non-finite values")
            metrics[method_id][metric_id] = {
                "value": float(point[method_id][metric_id]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "valid_draws": valid_draws,
            }
    contrasts: dict[str, Any] = {}
    for method_id in methods:
        if method_id == reference_method:
            continue
        contrasts[method_id] = {}
        for metric_id in METRIC_IDS:
            delta_draws = (
                draws_by_method[method_id][metric_id][valid]
                - draws_by_method[reference_method][metric_id][valid]
            )
            contrasts[method_id][metric_id] = {
                "reference_method": reference_method,
                "delta": float(
                    point[method_id][metric_id]
                    - point[reference_method][metric_id]
                ),
                "ci_low": float(np.quantile(delta_draws, 0.025)),
                "ci_high": float(np.quantile(delta_draws, 0.975)),
                "probability_delta_le_zero": float(np.mean(delta_draws <= 0.0)),
                "valid_draws": valid_draws,
                "higher_is_better": True,
            }
    return metrics, contrasts


def grouped_paired_bootstrap_auroc_auprc(
    *,
    labels: Sequence[int],
    scores_by_method: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
    reference_method: str = "iu_pcr",
    draws: int = 20_000,
    seed: int = 20260824,
) -> dict[str, Any]:
    if int(draws) <= 0:
        raise ValueError("draws must be positive")
    state = _validate_cell(
        labels=labels,
        scores_by_method=scores_by_method,
        group_ids=group_ids,
        reference_method=reference_method,
        canonical_group_order=False,
    )
    rng = np.random.default_rng(int(seed))
    counts = _draw_counts(
        draws=int(draws), n_groups=len(state["roster"]), rng=rng
    )
    positive = counts @ state["group_pos"]
    total = counts @ state["group_total"]
    valid = (positive > 0) & ((total - positive) > 0)
    draw_metrics: dict[str, dict[str, np.ndarray]] = {}
    for method_id in state["methods"]:
        auroc, auprc = _weighted_draw_metrics(
            labels=state["labels"],
            score=state["scores"][method_id],
            row_group_index=state["row_group_index"],
            counts=counts,
        )
        draw_metrics[method_id] = {"auroc": auroc, "auprc": auprc}
    point = _point_metrics(state["labels"], state["scores"])
    metrics, contrasts = _summaries(
        methods=state["methods"], point=point,
        draws_by_method=draw_metrics, valid=valid,
        reference_method=reference_method,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_unit": "source_group",
        "paired": True,
        "stratified_by_group_label": False,
        "draws_requested": int(draws),
        "valid_draws": int(np.sum(valid)),
        "seed": int(seed),
        "n_rows": len(state["labels"]),
        "n_groups": len(state["roster"]),
        "reference_method": reference_method,
        "metrics": metrics,
        "contrasts": contrasts,
    }


def population_grouped_paired_bootstrap_auroc_auprc(
    *,
    cells: Mapping[str, Mapping[str, object]],
    link_keys: Mapping[str, str] | None = None,
    reference_method: str = "iu_pcr",
    draws: int = 20_000,
    seed: int = 20260824,
    weighting: str = "equal_cell",
) -> dict[str, Any]:
    if int(draws) <= 0 or not cells:
        raise ValueError("population bootstrap requires cells and positive draws")
    cell_ids = tuple(sorted(cells))
    if weighting not in {"equal_cell", "single_cell"}:
        raise ValueError("population weighting must be equal_cell or single_cell")
    if weighting == "single_cell" and len(cell_ids) != 1:
        raise ValueError("single_cell weighting requires exactly one cell")
    if link_keys is None:
        effective_links = {
            cell_id: f"__independent__:{cell_id}" for cell_id in cell_ids
        }
    else:
        if set(link_keys) != set(cell_ids):
            raise ValueError("link_keys must cover the exact cell roster")
        effective_links = {cell_id: str(link_keys[cell_id]) for cell_id in cell_ids}
        if any(not value for value in effective_links.values()):
            raise ValueError("link keys must be nonempty")
    states: dict[str, dict[str, Any]] = {}
    methods: tuple[str, ...] | None = None
    for cell_id in cell_ids:
        cell = cells[cell_id]
        missing = {"labels", "group_ids", "scores_by_method"} - set(cell)
        if missing:
            raise ValueError(f"{cell_id}: missing fields {sorted(missing)}")
        scores = cell["scores_by_method"]
        if not isinstance(scores, Mapping):
            raise ValueError(f"{cell_id}: scores_by_method must be a mapping")
        state = _validate_cell(
            labels=cell["labels"],  # type: ignore[arg-type]
            scores_by_method=scores,  # type: ignore[arg-type]
            group_ids=cell["group_ids"],  # type: ignore[arg-type]
            reference_method=reference_method,
            canonical_group_order=True,
        )
        if methods is None:
            methods = state["methods"]
        elif state["methods"] != methods:
            raise ValueError("population method rosters differ")
        states[cell_id] = state
    if methods is None:  # pragma: no cover
        raise AssertionError("population method roster was not initialized")

    block_cells: dict[str, list[str]] = {}
    for cell_id in cell_ids:
        block_cells.setdefault(effective_links[cell_id], []).append(cell_id)
    link_blocks: list[dict[str, Any]] = []
    for link_key in sorted(block_cells):
        linked = tuple(sorted(block_cells[link_key]))
        first = states[linked[0]]
        first_counts = {
            group: int(len(first["members"][group])) for group in first["roster"]
        }
        for cell_id in linked[1:]:
            state = states[cell_id]
            if state["roster"] != first["roster"]:
                raise ValueError(f"linked group roster mismatch for {link_key!r}")
            counts = {
                group: int(len(state["members"][group])) for group in state["roster"]
            }
            if counts != first_counts:
                raise ValueError(f"linked group member-count mismatch for {link_key!r}")
        link_blocks.append({
            "link_key": link_key,
            "cell_ids": list(linked),
            "linked": len(linked) > 1,
            "n_groups": len(first["roster"]),
            "rows_per_cell": int(sum(first_counts.values())),
            "group_roster_sha256": _canonical_sha256(list(first["roster"])),
            "group_member_counts_sha256": _canonical_sha256([
                {"group_id": group, "member_count": first_counts[group]}
                for group in first["roster"]
            ]),
        })

    rng = np.random.default_rng(int(seed))
    counts_by_block = {
        block["link_key"]: np.zeros(
            (int(draws), int(block["n_groups"])), dtype=np.int16
        )
        for block in link_blocks
    }
    # Draw-major order matches the generic registered bootstrap stream even
    # when a population has several linkage blocks.
    for draw_index in range(int(draws)):
        for block in link_blocks:
            target = counts_by_block[block["link_key"]][draw_index]
            selected = rng.integers(0, len(target), size=len(target))
            np.add.at(target, selected, 1)

    valid = np.ones(int(draws), dtype=bool)
    cell_point_metrics: dict[str, dict[str, dict[str, float]]] = {}
    draw_metrics = {
        method_id: {
            metric_id: np.zeros(int(draws), dtype=np.float64)
            for metric_id in METRIC_IDS
        }
        for method_id in methods
    }
    for cell_id in cell_ids:
        state = states[cell_id]
        counts = counts_by_block[effective_links[cell_id]]
        positive = counts @ state["group_pos"]
        total = counts @ state["group_total"]
        valid &= (positive > 0) & ((total - positive) > 0)
        points = _point_metrics(state["labels"], state["scores"])
        cell_point_metrics[cell_id] = points
        for method_id in methods:
            auroc, auprc = _weighted_draw_metrics(
                labels=state["labels"], score=state["scores"][method_id],
                row_group_index=state["row_group_index"], counts=counts,
            )
            draw_metrics[method_id]["auroc"] += auroc / len(cell_ids)
            draw_metrics[method_id]["auprc"] += auprc / len(cell_ids)
    point = {
        method_id: {
            metric_id: float(np.mean([
                cell_point_metrics[cell_id][method_id][metric_id]
                for cell_id in cell_ids
            ]))
            for metric_id in METRIC_IDS
        }
        for method_id in methods
    }
    metrics, contrasts = _summaries(
        methods=methods, point=point, draws_by_method=draw_metrics,
        valid=valid, reference_method=reference_method,
    )
    any_linked = any(bool(block["linked"]) for block in link_blocks)
    return {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_unit": "linked_source_group" if any_linked else "source_group",
        "point_estimate_unit": "cell",
        "weighting": weighting,
        "paired": True,
        "linked_resampling": any_linked,
        "stratified_by_group_label": False,
        "draws_requested": int(draws),
        "valid_draws": int(np.sum(valid)),
        "seed": int(seed),
        "n_cells": len(cell_ids),
        "cell_ids": list(cell_ids),
        "n_rows": int(sum(len(states[cell_id]["labels"]) for cell_id in cell_ids)),
        "n_group_instances": int(sum(len(states[cell_id]["roster"]) for cell_id in cell_ids)),
        "n_resampling_groups": int(sum(block["n_groups"] for block in link_blocks)),
        "reference_method": reference_method,
        "link_keys": {cell_id: effective_links[cell_id] for cell_id in cell_ids},
        "link_blocks": link_blocks,
        "cell_point_metrics": cell_point_metrics,
        "metrics": metrics,
        "contrasts": contrasts,
    }


__all__ = [
    "METRIC_IDS",
    "SCHEMA_VERSION",
    "grouped_paired_bootstrap_auroc_auprc",
    "population_grouped_paired_bootstrap_auroc_auprc",
]
