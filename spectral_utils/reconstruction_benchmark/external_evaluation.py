"""Deterministic grouped evaluation for external final-answer cells."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_IDS = ("auroc", "auprc", "aurc_x1000")

# Keep the largest temporary draw-by-row matrix bounded.  In particular, HLE
# and PRMBench have thousands of mostly unique score values, so a dense
# group-by-score-block representation would be unacceptably large.  The fast
# path below is O(batch * rows) memory and processes methods one at a time.
_BOOTSTRAP_TARGET_ROW_WEIGHTS = 1_000_000
_BOOTSTRAP_MAX_BATCH_DRAWS = 4_096
_CONTRAST_NUMERICAL_ZERO_ATOL = {
    "auroc": 1e-14,
    "auprc": 1e-14,
    "aurc_x1000": 2e-10,
}


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


def _sample_group_counts(
    *,
    roster: Sequence[str],
    rng: np.random.Generator,
    group_labels: Mapping[str, int] | None,
) -> np.ndarray:
    """Sample the original group roster and return multiplicities.

    The calls to ``rng.integers`` deliberately match
    :func:`_sample_group_roster`: one call per ordinary draw, or label-0 then
    label-1 calls for a stratified draw.  Keeping this helper draw-at-a-time is
    important because population evaluation interleaves RNG calls across
    linked resampling blocks.
    """

    n_groups = len(roster)
    if group_labels is None:
        selected = rng.integers(0, n_groups, size=n_groups)
        return np.bincount(selected, minlength=n_groups).astype(np.int64, copy=False)

    counts = np.zeros(n_groups, dtype=np.int64)
    for label in (0, 1):
        positions = np.asarray(
            [position for position, group in enumerate(roster)
             if group_labels[group] == label],
            dtype=np.int64,
        )
        selected = rng.integers(0, len(positions), size=len(positions))
        counts += np.bincount(positions[selected], minlength=n_groups)
    return counts


def _row_group_positions(
    *,
    n_rows: int,
    roster: Sequence[str],
    members: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Map each row to its position in ``roster`` without a dense G-by-N map."""

    result = np.empty(int(n_rows), dtype=np.int64)
    assigned = np.zeros(int(n_rows), dtype=bool)
    for position, group in enumerate(roster):
        indices = members[group]
        result[indices] = int(position)
        assigned[indices] = True
    if not assigned.all():  # pragma: no cover - guarded by _group_members
        raise AssertionError("source-group membership does not cover every row")
    return result


def _metric_plan(
    *,
    labels: np.ndarray,
    score: np.ndarray,
    row_group_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Freeze a method's stable ascending score order and exact tie blocks."""

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    block_starts = np.flatnonzero(
        np.r_[True, sorted_score[1:] != sorted_score[:-1]]
    ).astype(np.int64, copy=False)
    return (
        row_group_positions[order],
        labels[order].astype(np.int64, copy=False),
        block_starts,
    )


def _bootstrap_batch_size(*, n_rows: int, n_groups: int, remaining: int) -> int:
    width = max(1, int(n_rows) + int(n_groups))
    bounded = max(1, _BOOTSTRAP_TARGET_ROW_WEIGHTS // width)
    return min(int(remaining), _BOOTSTRAP_MAX_BATCH_DRAWS, bounded)


def _weighted_binary_metric_batch(
    *,
    group_counts: np.ndarray,
    plan: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate duplicated-row bootstrap samples from integer row weights.

    A sampled group appearing ``k`` times is exactly equivalent to giving each
    of its rows integer weight ``k``.  AUROC and average precision are computed
    from fixed score-tie blocks.  AURC uses the same expected-random-order tie
    estimand as :func:`aurc_x1000`; the harmonic identity merely sums a whole
    duplicated tie block at once.
    """

    if group_counts.ndim != 2 or len(group_counts) == 0:
        raise ValueError("group_counts must be a nonempty draw-by-group matrix")
    ordered_group_positions, sorted_labels, block_starts = plan
    row_weights = group_counts[:, ordered_group_positions]
    block_rows = np.add.reduceat(row_weights, block_starts, axis=1)
    block_positive = np.add.reduceat(
        row_weights * sorted_labels[np.newaxis, :],
        block_starts,
        axis=1,
    )
    block_negative = block_rows - block_positive

    total_positive = np.sum(block_positive, axis=1, dtype=np.int64)
    total_negative = np.sum(block_negative, axis=1, dtype=np.int64)
    if np.any(total_positive <= 0) or np.any(total_negative <= 0):
        raise ValueError("weighted binary metrics require both classes in every draw")

    negatives_before = (
        np.cumsum(block_negative, axis=1, dtype=np.int64) - block_negative
    )
    auc_numerator = np.sum(
        block_positive
        * (negatives_before + 0.5 * block_negative),
        axis=1,
        dtype=float,
    )
    auroc = auc_numerator / (total_positive * total_negative)

    positive_at_or_above = np.cumsum(
        block_positive[:, ::-1], axis=1, dtype=np.int64
    )[:, ::-1]
    rows_at_or_above = np.cumsum(
        block_rows[:, ::-1], axis=1, dtype=np.int64
    )[:, ::-1]
    precision = np.divide(
        positive_at_or_above,
        rows_at_or_above,
        out=np.zeros_like(positive_at_or_above, dtype=float),
        where=rows_at_or_above > 0,
    )
    auprc = (
        np.sum(precision * block_positive, axis=1, dtype=float)
        / total_positive
    )

    accepted = np.cumsum(block_rows, axis=1, dtype=np.int64)
    accepted_before = accepted - block_rows
    errors = np.cumsum(block_positive, axis=1, dtype=np.int64)
    errors_before = errors - block_positive
    max_rows = int(np.max(accepted[:, -1]))
    harmonic = np.zeros(max_rows + 1, dtype=float)
    if max_rows:
        harmonic[1:] = np.cumsum(
            1.0 / np.arange(1, max_rows + 1, dtype=float)
        )
    harmonic_delta = harmonic[accepted] - harmonic[accepted_before]
    error_fraction = np.divide(
        block_positive,
        block_rows,
        out=np.zeros_like(block_positive, dtype=float),
        where=block_rows > 0,
    )
    risk_sum = np.sum(
        errors_before * harmonic_delta
        + error_fraction
        * (block_rows - accepted_before * harmonic_delta),
        axis=1,
        dtype=float,
    )
    aurc = 1000.0 * risk_sum / accepted[:, -1]
    return {
        "auroc": auroc.astype(float, copy=False),
        "auprc": auprc.astype(float, copy=False),
        "aurc_x1000": aurc.astype(float, copy=False),
    }


def _probability_delta_le_zero(delta_draws: np.ndarray, *, metric: str) -> float:
    """Return the weak-tail probability with mathematical ties counted once.

    The scalar sklearn path and the integer-weight fast path can differ by an
    IEEE-754 rounding unit even when a draw's two estimands are mathematically
    identical.  A weak inequality must count that equality as ``<= 0`` rather
    than let an implementation-specific ULP choose a side.  The frozen bounds
    are far below report precision and cover the larger ``x1000`` AURC scale.
    """

    tolerance = _CONTRAST_NUMERICAL_ZERO_ATOL[metric]
    return float(np.mean(delta_draws <= tolerance))


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
    For ``probability_delta_le_zero``, differences within the frozen numerical
    zero bounds (AUROC/AUPRC 1e-14; AURC x1000 2e-10) count as mathematical
    ties, so an IEEE-754 rounding unit cannot choose a tail.
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

    row_group_positions = _row_group_positions(
        n_rows=len(y), roster=roster, members=members,
    )
    group_positive = np.bincount(
        row_group_positions, weights=y, minlength=len(roster),
    ).astype(np.int64, copy=False)
    group_rows = np.bincount(
        row_group_positions, minlength=len(roster),
    ).astype(np.int64, copy=False)
    group_negative = group_rows - group_positive
    metric_plans = {
        method_id: _metric_plan(
            labels=y,
            score=scores[method_id],
            row_group_positions=row_group_positions,
        )
        for method_id in methods
    }

    rng = np.random.default_rng(int(seed))
    bootstrap_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in methods
    }
    valid_draws = 0
    generated_draws = 0
    while generated_draws < int(draws):
        batch_draws = _bootstrap_batch_size(
            n_rows=len(y),
            n_groups=len(roster),
            remaining=int(draws) - generated_draws,
        )
        group_counts = np.empty((batch_draws, len(roster)), dtype=np.int64)
        for draw_index in range(batch_draws):
            group_counts[draw_index] = _sample_group_counts(
                roster=roster,
                rng=rng,
                group_labels=group_labels,
            )
        generated_draws += batch_draws

        total_positive = group_counts @ group_positive
        total_negative = group_counts @ group_negative
        valid = (total_positive > 0) & (total_negative > 0)
        if not np.any(valid):
            continue
        valid_counts = group_counts[valid]
        valid_draws += int(np.sum(valid))
        for method_id in methods:
            observed = _weighted_binary_metric_batch(
                group_counts=valid_counts,
                plan=metric_plans[method_id],
            )
            for metric in METRIC_IDS:
                bootstrap_chunks[method_id][metric].append(observed[metric])
    if valid_draws == 0:
        raise RuntimeError("every grouped bootstrap draw was single-class")

    bootstrap_values = {
        method_id: {
            metric: np.concatenate(bootstrap_chunks[method_id][metric])
            for metric in METRIC_IDS
        }
        for method_id in methods
    }

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for method_id in methods:
        point = binary_metric_values(y, scores[method_id])
        metrics[method_id] = {}
        for metric in METRIC_IDS:
            values = bootstrap_values[method_id][metric]
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
            left = bootstrap_values[method_id][metric]
            right = bootstrap_values[reference_method][metric]
            if left.shape != right.shape:
                raise AssertionError("paired bootstrap draw arrays diverged")
            delta_draws = left - right
            delta = metrics[method_id][metric]["value"] - metrics[reference_method][metric]["value"]
            contrasts[method_id][metric] = {
                "reference_method": reference_method,
                "delta": float(delta),
                "ci_low": float(np.quantile(delta_draws, 0.025)),
                "ci_high": float(np.quantile(delta_draws, 0.975)),
                "probability_delta_le_zero": _probability_delta_le_zero(
                    delta_draws, metric=metric,
                ),
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
    For ``probability_delta_le_zero``, differences within the frozen numerical
    zero bounds (AUROC/AUPRC 1e-14; AURC x1000 2e-10) count as mathematical
    ties, so an IEEE-754 rounding unit cannot choose a tail.
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
        row_group_positions = _row_group_positions(
            n_rows=len(y), roster=roster, members=members,
        )
        group_positive = np.bincount(
            row_group_positions, weights=y, minlength=len(roster),
        ).astype(np.int64, copy=False)
        group_rows = np.bincount(
            row_group_positions, minlength=len(roster),
        ).astype(np.int64, copy=False)
        state[cell_id] = {
            "labels": y,
            "scores": scores,
            "roster": roster,
            "members": members,
            "group_labels": group_labels,
            "group_positive": group_positive,
            "group_negative": group_rows - group_positive,
            "metric_plans": {
                method_id: _metric_plan(
                    labels=y,
                    score=scores[method_id],
                    row_group_positions=row_group_positions,
                )
                for method_id in methods
            },
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

    bootstrap_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in method_roster
    }
    rng = np.random.default_rng(int(seed))
    valid_draws = 0
    generated_draws = 0
    block_keys = tuple(sorted(block_state))
    max_cell_rows = max(len(state[cell_id]["labels"]) for cell_id in cell_ids)
    total_block_groups = sum(
        len(block_state[link_key]["roster"]) for link_key in block_keys
    )
    while generated_draws < int(draws):
        batch_draws = _bootstrap_batch_size(
            n_rows=int(max_cell_rows),
            n_groups=int(total_block_groups),
            remaining=int(draws) - generated_draws,
        )
        counts_by_block: dict[str, np.ndarray] = {}
        for link_key in block_keys:
            roster = block_state[link_key]["roster"]
            if not isinstance(roster, tuple):
                raise AssertionError("internal link-block state is invalid")
            counts_by_block[link_key] = np.empty(
                (batch_draws, len(roster)), dtype=np.int64,
            )
        # Preserve the original RNG nesting exactly: draw, then sorted link
        # block, then label-0/label-1 strata when requested.
        for draw_index in range(batch_draws):
            for link_key in block_keys:
                roster = block_state[link_key]["roster"]
                group_labels = block_state[link_key]["group_labels"]
                if not isinstance(roster, tuple):
                    raise AssertionError("internal link-block state is invalid")
                counts_by_block[link_key][draw_index] = _sample_group_counts(
                    roster=roster,
                    rng=rng,
                    group_labels=(
                        group_labels if isinstance(group_labels, dict) else None
                    ),
                )
        generated_draws += batch_draws

        valid = np.ones(batch_draws, dtype=bool)
        for cell_id in cell_ids:
            group_positive = state[cell_id]["group_positive"]
            group_negative = state[cell_id]["group_negative"]
            if (
                not isinstance(group_positive, np.ndarray)
                or not isinstance(group_negative, np.ndarray)
            ):
                raise AssertionError("internal cell state is invalid")
            group_counts = counts_by_block[effective_link_keys[cell_id]]
            valid &= (
                (group_counts @ group_positive > 0)
                & (group_counts @ group_negative > 0)
            )
        if not np.any(valid):
            continue

        valid_draws += int(np.sum(valid))
        cell_metric_chunks = {
            method_id: {metric: [] for metric in METRIC_IDS}
            for method_id in method_roster
        }
        for cell_id in cell_ids:
            metric_plans = state[cell_id]["metric_plans"]
            if not isinstance(metric_plans, dict):
                raise AssertionError("internal cell metric plan is invalid")
            valid_counts = counts_by_block[effective_link_keys[cell_id]][valid]
            for method_id in method_roster:
                observed = _weighted_binary_metric_batch(
                    group_counts=valid_counts,
                    plan=metric_plans[method_id],
                )
                for metric in METRIC_IDS:
                    cell_metric_chunks[method_id][metric].append(observed[metric])
        for method_id in method_roster:
            for metric in METRIC_IDS:
                bootstrap_chunks[method_id][metric].append(
                    np.mean(
                        np.stack(cell_metric_chunks[method_id][metric], axis=0),
                        axis=0,
                    )
                )
    if valid_draws == 0:
        raise RuntimeError("every population bootstrap draw was invalid")

    bootstrap_values = {
        method_id: {
            metric: np.concatenate(bootstrap_chunks[method_id][metric])
            for metric in METRIC_IDS
        }
        for method_id in method_roster
    }

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for method_id in method_roster:
        metrics[method_id] = {}
        for metric in METRIC_IDS:
            values = bootstrap_values[method_id][metric]
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
            left = bootstrap_values[method_id][metric]
            right = bootstrap_values[reference_method][metric]
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
                "probability_delta_le_zero": _probability_delta_le_zero(
                    delta_draws, metric=metric,
                ),
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
