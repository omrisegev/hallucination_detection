"""Deterministic grouped evaluation for external final-answer cells."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_IDS = ("auroc", "auprc", "aurc_x1000")


def _binary_labels(labels: Sequence[int]) -> np.ndarray:
    """Validate before casting so fractional or missing labels cannot truncate."""

    value = np.asarray(labels)
    if value.ndim != 1:
        raise ValueError("labels must be a vector")
    if not np.isin(value, (0, 1)).all():
        raise ValueError("labels must be binary")
    return value.astype(np.int8, copy=False)


def aurc_x1000(labels: np.ndarray, score: np.ndarray) -> float:
    """Risk-coverage area when low-risk responses are retained first.

    Exact score ties are integrated over their expected random ordering.  This
    makes the result independent of input row order: if a tie block contains
    ``m`` rows and ``e`` errors, the expected number of errors after accepting
    ``j`` rows from that block is ``j * e / m``.
    """

    labels = _binary_labels(labels)
    score = np.asarray(score, dtype=float)
    if labels.ndim != 1 or score.shape != labels.shape or len(labels) == 0:
        raise ValueError("labels and score must be equal-length nonempty vectors")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_labels = labels[order]
    block_starts = np.flatnonzero(
        np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    )
    block_ends = np.r_[block_starts[1:], len(labels)]
    accepted_before = 0
    errors_before = 0.0
    risk_sum = 0.0
    for start, end in zip(block_starts.tolist(), block_ends.tolist()):
        block_size = int(end - start)
        block_errors = float(np.sum(sorted_labels[start:end]))
        within_block = np.arange(1, block_size + 1, dtype=float)
        expected_errors = errors_before + within_block * block_errors / block_size
        risk_sum += float(
            np.sum(expected_errors / (accepted_before + within_block))
        )
        accepted_before += block_size
        errors_before += block_errors
    return float(1000.0 * risk_sum / len(labels))


def _validate_scores(
    *,
    labels: Sequence[int],
    scores_by_method: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
    reference_method: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    """Validate and normalize one cell's paired evaluation arrays."""

    y = _binary_labels(labels)
    groups = np.asarray(group_ids, dtype=str)
    if y.ndim != 1 or groups.shape != y.shape or len(y) < 2:
        raise ValueError("labels and group_ids must be equal-length nontrivial vectors")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("grouped bootstrap requires both classes")
    if any(
        not isinstance(method_id, str) or not method_id
        for method_id in scores_by_method
    ):
        raise ValueError("method IDs must be nonempty strings")
    methods = tuple(sorted(scores_by_method))
    if not methods or reference_method not in methods:
        raise ValueError("reference method is absent from scores_by_method")
    scores: dict[str, np.ndarray] = {}
    for method_id in methods:
        value = np.asarray(scores_by_method[method_id], dtype=float)
        if value.shape != y.shape or not np.isfinite(value).all():
            raise ValueError(f"{method_id}: invalid score vector")
        scores[method_id] = value
    return y, groups, methods, scores


def _group_members(
    groups: np.ndarray,
    *,
    canonical_order: bool,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Return a validated source-group roster and row membership map."""

    member_lists: dict[str, list[int]] = {}
    first_seen: list[str] = []
    for row_index, group in enumerate(groups.tolist()):
        if not group:
            raise ValueError("group IDs must be nonempty")
        if group not in member_lists:
            first_seen.append(group)
            member_lists[group] = []
        member_lists[group].append(row_index)
    roster = tuple(sorted(member_lists) if canonical_order else first_seen)
    if len(roster) < 2:
        raise ValueError("grouped bootstrap requires at least two source groups")
    members = {
        group: np.asarray(member_lists[group], dtype=np.int64)
        for group in roster
    }
    return roster, members


def _pure_group_labels(
    labels: np.ndarray,
    roster: Sequence[str],
    members: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Return one binary label per group, refusing mixed-label groups."""

    result: dict[str, int] = {}
    for group in roster:
        values = np.unique(labels[members[group]])
        if len(values) != 1:
            raise ValueError(
                "label-stratified group bootstrap requires every source group "
                f"to be label-pure; {group!r} contains {values.tolist()}"
            )
        result[group] = int(values[0])
    if set(result.values()) != {0, 1}:
        raise ValueError("label-stratified group bootstrap requires both group labels")
    return result


def _sample_group_roster(
    *,
    roster: Sequence[str],
    rng: np.random.Generator,
    group_labels: Mapping[str, int] | None,
) -> tuple[str, ...]:
    """Sample a roster with replacement, optionally within label strata."""

    if group_labels is None:
        selected = rng.integers(0, len(roster), size=len(roster))
        return tuple(roster[int(position)] for position in selected)
    sampled: list[str] = []
    for label in (0, 1):
        stratum = tuple(group for group in roster if group_labels[group] == label)
        selected = rng.integers(0, len(stratum), size=len(stratum))
        sampled.extend(stratum[int(position)] for position in selected)
    return tuple(sampled)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def binary_metric_values(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    labels = _binary_labels(labels)
    score = np.asarray(score, dtype=float)
    if labels.ndim != 1 or score.shape != labels.shape:
        raise ValueError("labels and score must be equal-length vectors")
    if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
        raise ValueError("binary metrics require both classes")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
        "aurc_x1000": aurc_x1000(labels, score),
    }


def grouped_paired_bootstrap(
    *,
    labels: Sequence[int],
    scores_by_method: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
    reference_method: str = "iu_pcr",
    draws: int = 20_000,
    seed: int = 20260824,
    stratify_by_label: bool = False,
) -> dict:
    """Bootstrap source groups with one shared draw stream for every method.

    Every occurrence of a sampled group brings all of its rows.  Thus sibling
    generations, scorer copies, or PRMBench variants never split across a draw.
    Deltas are paired because all methods use the exact same resampled indices.
    """

    if int(draws) <= 0:
        raise ValueError("draws must be positive")
    y, groups, methods, scores = _validate_scores(
        labels=labels,
        scores_by_method=scores_by_method,
        group_ids=group_ids,
        reference_method=reference_method,
    )
    roster, members = _group_members(groups, canonical_order=False)
    group_labels = (
        _pure_group_labels(y, roster, members)
        if bool(stratify_by_label)
        else None
    )

    rng = np.random.default_rng(int(seed))
    bootstrap_values = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in methods
    }
    valid_draws = 0
    for _ in range(int(draws)):
        sampled_groups = _sample_group_roster(
            roster=roster,
            rng=rng,
            group_labels=group_labels,
        )
        indices = np.concatenate([members[group] for group in sampled_groups])
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) != 2:
            continue
        valid_draws += 1
        for method_id in methods:
            observed = binary_metric_values(sampled_y, scores[method_id][indices])
            for metric in METRIC_IDS:
                bootstrap_values[method_id][metric].append(observed[metric])
    if valid_draws == 0:
        raise RuntimeError("every grouped bootstrap draw was single-class")

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for method_id in methods:
        point = binary_metric_values(y, scores[method_id])
        metrics[method_id] = {}
        for metric in METRIC_IDS:
            values = np.asarray(bootstrap_values[method_id][metric], dtype=float)
            metrics[method_id][metric] = {
                "value": point[metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "valid_draws": int(len(values)),
            }

    contrasts: dict[str, dict[str, dict[str, float | int | str | bool]]] = {}
    for method_id in methods:
        if method_id == reference_method:
            continue
        contrasts[method_id] = {}
        for metric in METRIC_IDS:
            left = np.asarray(bootstrap_values[method_id][metric], dtype=float)
            right = np.asarray(bootstrap_values[reference_method][metric], dtype=float)
            if left.shape != right.shape:
                raise AssertionError("paired bootstrap draw arrays diverged")
            delta_draws = left - right
            delta = metrics[method_id][metric]["value"] - metrics[reference_method][metric]["value"]
            contrasts[method_id][metric] = {
                "reference_method": reference_method,
                "delta": float(delta),
                "ci_low": float(np.quantile(delta_draws, 0.025)),
                "ci_high": float(np.quantile(delta_draws, 0.975)),
                "probability_delta_le_zero": float(np.mean(delta_draws <= 0.0)),
                "valid_draws": int(len(delta_draws)),
                "higher_is_better": metric != "aurc_x1000",
            }
    return {
        "schema_version": "reconstruction-grouped-paired-bootstrap-v1",
        "bootstrap_unit": "source_group",
        "paired": True,
        "stratified_by_group_label": bool(stratify_by_label),
        "draws_requested": int(draws),
        "valid_draws": int(valid_draws),
        "seed": int(seed),
        "n_rows": int(len(y)),
        "n_groups": int(len(roster)),
        "reference_method": reference_method,
        "metrics": metrics,
        "contrasts": contrasts,
    }


def population_grouped_paired_bootstrap(
    *,
    cells: Mapping[str, Mapping[str, object]],
    link_keys: Mapping[str, str] | None = None,
    reference_method: str = "iu_pcr",
    draws: int = 20_000,
    seed: int = 20260824,
    weighting: str = "equal_cell",
    stratify_by_label: bool = False,
) -> dict:
    """Bootstrap a registered population while preserving its cell structure.

    ``cells`` maps each registered cell ID to three arrays/mappings named
    ``labels``, ``group_ids``, and ``scores_by_method``.  Point estimates and
    every bootstrap replicate are equal-weight averages of the cell metrics;
    rows are never pooled across cells.

    Cells assigned the same value in ``link_keys`` are repeated measurements
    of the same source groups (for example, the same AQuA questions scored by
    different models).  Such cells share one resampling draw.  Their canonical
    group-ID rosters and the row count of every group must match exactly, or
    evaluation stops.  If ``link_keys`` is omitted, every cell is resampled
    independently.  A supplied map must cover the exact cell roster.

    Label stratification operates at the source-group level.  It is intended
    for registered imbalanced panels such as HLE and therefore refuses a group
    containing both labels.  Linked cells must also agree on every group label.
    All methods share every accepted draw, so all reported contrasts are paired.
    """

    if int(draws) <= 0:
        raise ValueError("draws must be positive")
    if not cells:
        raise ValueError("population bootstrap requires at least one cell")
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cells):
        raise ValueError("cell IDs must be nonempty strings")
    cell_ids = tuple(sorted(cells))
    if weighting not in {"equal_cell", "single_cell"}:
        raise ValueError("population weighting must be equal_cell or single_cell")
    if weighting == "single_cell" and len(cell_ids) != 1:
        raise ValueError("single_cell weighting requires exactly one cell")

    if link_keys is None:
        effective_link_keys = {
            cell_id: f"__independent__:{cell_id}"
            for cell_id in cell_ids
        }
    else:
        if set(link_keys) != set(cell_ids):
            missing = sorted(set(cell_ids) - set(link_keys))
            extra = sorted(set(link_keys) - set(cell_ids))
            raise ValueError(
                "link_keys must cover the exact population cell roster; "
                f"missing={missing}, extra={extra}"
            )
        effective_link_keys = {}
        for cell_id in cell_ids:
            link_key = link_keys[cell_id]
            if not isinstance(link_key, str) or not link_key:
                raise ValueError(f"{cell_id}: link key must be a nonempty string")
            effective_link_keys[cell_id] = link_key

    state: dict[str, dict[str, object]] = {}
    method_roster: tuple[str, ...] | None = None
    for cell_id in cell_ids:
        cell = cells[cell_id]
        missing_fields = sorted(
            {"labels", "group_ids", "scores_by_method"} - set(cell)
        )
        if missing_fields:
            raise ValueError(f"{cell_id}: missing fields {missing_fields}")
        scores_value = cell["scores_by_method"]
        if not isinstance(scores_value, Mapping):
            raise ValueError(f"{cell_id}: scores_by_method must be a mapping")
        y, groups, methods, scores = _validate_scores(
            labels=cell["labels"],  # type: ignore[arg-type]
            scores_by_method=scores_value,  # type: ignore[arg-type]
            group_ids=cell["group_ids"],  # type: ignore[arg-type]
            reference_method=reference_method,
        )
        if method_roster is None:
            method_roster = methods
        elif methods != method_roster:
            raise ValueError(
                f"{cell_id}: method roster {methods} differs from {method_roster}"
            )
        roster, members = _group_members(groups, canonical_order=True)
        group_labels = (
            _pure_group_labels(y, roster, members)
            if bool(stratify_by_label)
            else None
        )
        state[cell_id] = {
            "labels": y,
            "scores": scores,
            "roster": roster,
            "members": members,
            "group_labels": group_labels,
        }
    if method_roster is None:  # pragma: no cover - guarded by nonempty cells
        raise AssertionError("method roster was not initialized")

    block_cells: dict[str, list[str]] = {}
    for cell_id in cell_ids:
        block_cells.setdefault(effective_link_keys[cell_id], []).append(cell_id)
    link_blocks: list[dict[str, object]] = []
    block_state: dict[str, dict[str, object]] = {}
    for link_key in sorted(block_cells):
        linked_cells = tuple(sorted(block_cells[link_key]))
        first_cell = linked_cells[0]
        first_roster = state[first_cell]["roster"]
        first_members = state[first_cell]["members"]
        if not isinstance(first_roster, tuple) or not isinstance(first_members, dict):
            raise AssertionError("internal source-group state is invalid")
        first_counts = {
            group: int(len(first_members[group]))
            for group in first_roster
        }
        first_group_labels = state[first_cell]["group_labels"]
        for cell_id in linked_cells[1:]:
            roster = state[cell_id]["roster"]
            members = state[cell_id]["members"]
            if roster != first_roster:
                raise ValueError(
                    f"linked group roster mismatch for {link_key!r}: "
                    f"{first_cell} versus {cell_id}"
                )
            if not isinstance(members, dict):
                raise AssertionError("internal source-group state is invalid")
            counts = {group: int(len(members[group])) for group in roster}
            if counts != first_counts:
                raise ValueError(
                    f"linked group member-count mismatch for {link_key!r}: "
                    f"{first_cell} versus {cell_id}"
                )
            if stratify_by_label and state[cell_id]["group_labels"] != first_group_labels:
                raise ValueError(
                    f"linked group-label mismatch for {link_key!r}: "
                    f"{first_cell} versus {cell_id}"
                )
        roster_payload = list(first_roster)
        count_payload = [
            {"group_id": group, "member_count": first_counts[group]}
            for group in first_roster
        ]
        audit = {
            "link_key": link_key,
            "cell_ids": list(linked_cells),
            "linked": len(linked_cells) > 1,
            "n_groups": len(first_roster),
            "rows_per_cell": int(sum(first_counts.values())),
            "group_roster_sha256": _canonical_sha256(roster_payload),
            "group_member_counts_sha256": _canonical_sha256(count_payload),
        }
        if stratify_by_label:
            if not isinstance(first_group_labels, dict):
                raise AssertionError("internal group-label state is invalid")
            label_payload = [
                {"group_id": group, "label": first_group_labels[group]}
                for group in first_roster
            ]
            audit["group_labels_sha256"] = _canonical_sha256(label_payload)
            audit["groups_by_label"] = {
                str(label): sum(
                    first_group_labels[group] == label for group in first_roster
                )
                for label in (0, 1)
            }
        link_blocks.append(audit)
        block_state[link_key] = {
            "roster": first_roster,
            "group_labels": first_group_labels,
        }

    cell_point_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for cell_id in cell_ids:
        y = state[cell_id]["labels"]
        scores = state[cell_id]["scores"]
        if not isinstance(y, np.ndarray) or not isinstance(scores, dict):
            raise AssertionError("internal cell state is invalid")
        cell_point_metrics[cell_id] = {
            method_id: binary_metric_values(y, scores[method_id])
            for method_id in method_roster
        }
    point_metrics = {
        method_id: {
            metric: float(np.mean([
                cell_point_metrics[cell_id][method_id][metric]
                for cell_id in cell_ids
            ]))
            for metric in METRIC_IDS
        }
        for method_id in method_roster
    }

    bootstrap_values = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in method_roster
    }
    rng = np.random.default_rng(int(seed))
    valid_draws = 0
    for _ in range(int(draws)):
        sampled_by_block: dict[str, tuple[str, ...]] = {}
        for link_key in sorted(block_state):
            roster = block_state[link_key]["roster"]
            group_labels = block_state[link_key]["group_labels"]
            if not isinstance(roster, tuple):
                raise AssertionError("internal link-block state is invalid")
            sampled_by_block[link_key] = _sample_group_roster(
                roster=roster,
                rng=rng,
                group_labels=group_labels if isinstance(group_labels, dict) else None,
            )

        sampled_indices: dict[str, np.ndarray] = {}
        draw_is_valid = True
        for cell_id in cell_ids:
            members = state[cell_id]["members"]
            y = state[cell_id]["labels"]
            if not isinstance(members, dict) or not isinstance(y, np.ndarray):
                raise AssertionError("internal cell state is invalid")
            sampled_groups = sampled_by_block[effective_link_keys[cell_id]]
            indices = np.concatenate([members[group] for group in sampled_groups])
            if len(np.unique(y[indices])) != 2:
                draw_is_valid = False
                break
            sampled_indices[cell_id] = indices
        if not draw_is_valid:
            continue

        valid_draws += 1
        for method_id in method_roster:
            cell_draw_metrics: list[dict[str, float]] = []
            for cell_id in cell_ids:
                y = state[cell_id]["labels"]
                scores = state[cell_id]["scores"]
                if not isinstance(y, np.ndarray) or not isinstance(scores, dict):
                    raise AssertionError("internal cell state is invalid")
                indices = sampled_indices[cell_id]
                cell_draw_metrics.append(
                    binary_metric_values(y[indices], scores[method_id][indices])
                )
            for metric in METRIC_IDS:
                bootstrap_values[method_id][metric].append(float(np.mean([
                    observed[metric] for observed in cell_draw_metrics
                ])))
    if valid_draws == 0:
        raise RuntimeError("every population bootstrap draw was invalid")

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for method_id in method_roster:
        metrics[method_id] = {}
        for metric in METRIC_IDS:
            values = np.asarray(bootstrap_values[method_id][metric], dtype=float)
            metrics[method_id][metric] = {
                "value": point_metrics[method_id][metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "valid_draws": int(len(values)),
            }

    contrasts: dict[str, dict[str, dict[str, float | int | str | bool]]] = {}
    for method_id in method_roster:
        if method_id == reference_method:
            continue
        contrasts[method_id] = {}
        for metric in METRIC_IDS:
            left = np.asarray(bootstrap_values[method_id][metric], dtype=float)
            right = np.asarray(bootstrap_values[reference_method][metric], dtype=float)
            if left.shape != right.shape:
                raise AssertionError("paired population bootstrap draw arrays diverged")
            delta_draws = left - right
            delta = (
                point_metrics[method_id][metric]
                - point_metrics[reference_method][metric]
            )
            contrasts[method_id][metric] = {
                "reference_method": reference_method,
                "delta": float(delta),
                "ci_low": float(np.quantile(delta_draws, 0.025)),
                "ci_high": float(np.quantile(delta_draws, 0.975)),
                "probability_delta_le_zero": float(np.mean(delta_draws <= 0.0)),
                "valid_draws": int(len(delta_draws)),
                "higher_is_better": metric != "aurc_x1000",
            }

    any_linked = any(bool(block["linked"]) for block in link_blocks)
    bootstrap_unit = "linked_source_group" if any_linked else "source_group"
    if stratify_by_label:
        bootstrap_unit += "_stratified_by_label"
    return {
        "schema_version": "reconstruction-population-grouped-paired-bootstrap-v1",
        "bootstrap_unit": bootstrap_unit,
        "point_estimate_unit": "cell",
        "weighting": weighting,
        "paired": True,
        "linked_resampling": any_linked,
        "stratified_by_group_label": bool(stratify_by_label),
        "draws_requested": int(draws),
        "valid_draws": int(valid_draws),
        "seed": int(seed),
        "n_cells": len(cell_ids),
        "cell_ids": list(cell_ids),
        "n_rows": int(sum(len(state[cell_id]["labels"]) for cell_id in cell_ids)),
        "n_group_instances": int(sum(
            len(state[cell_id]["roster"]) for cell_id in cell_ids
        )),
        "n_resampling_groups": int(sum(
            int(block["n_groups"]) for block in link_blocks
        )),
        "reference_method": reference_method,
        "link_keys": {
            cell_id: effective_link_keys[cell_id]
            for cell_id in cell_ids
        },
        "link_blocks": link_blocks,
        "cell_point_metrics": cell_point_metrics,
        "metrics": metrics,
        "contrasts": contrasts,
    }


__all__ = [
    "METRIC_IDS",
    "aurc_x1000",
    "binary_metric_values",
    "grouped_paired_bootstrap",
    "population_grouped_paired_bootstrap",
]
