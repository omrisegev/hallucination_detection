#!/usr/bin/env python3
"""Adversarial equivalence tests for the external bootstrap fast path.

The helpers in this file intentionally retain the original slow algorithm:
materialize every sampled group's duplicated rows, then call the public scalar
metric implementation for every method and draw.  Production code must not
import these references.
"""

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark import external_evaluation
from spectral_utils.reconstruction_benchmark.external_evaluation import (
    METRIC_IDS,
    binary_metric_values,
    grouped_paired_bootstrap,
    population_grouped_paired_bootstrap,
)


FROZEN_EQUIVALENCE_ATOL = {
    "auroc": 2e-15,
    "auprc": 2e-15,
    "aurc_x1000": 2e-10,
}
FROZEN_NUMERICAL_ZERO_ATOL = {
    "auroc": 1e-14,
    "auprc": 1e-14,
    "aurc_x1000": 2e-10,
}


def _members(
    group_ids: Sequence[str], *, canonical_order: bool,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    member_lists: dict[str, list[int]] = {}
    first_seen: list[str] = []
    for row_index, group in enumerate(group_ids):
        if group not in member_lists:
            first_seen.append(group)
            member_lists[group] = []
        member_lists[group].append(row_index)
    roster = tuple(sorted(member_lists) if canonical_order else first_seen)
    return roster, {
        group: np.asarray(member_lists[group], dtype=np.int64)
        for group in roster
    }


def _group_labels(
    labels: np.ndarray,
    roster: Sequence[str],
    members: Mapping[str, np.ndarray],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for group in roster:
        values = np.unique(labels[members[group]])
        if len(values) != 1:
            raise ValueError("slow stratified reference requires pure groups")
        result[group] = int(values[0])
    return result


def _sample_roster(
    roster: Sequence[str],
    rng: np.random.Generator,
    group_labels: Mapping[str, int] | None,
) -> tuple[str, ...]:
    if group_labels is None:
        selected = rng.integers(0, len(roster), size=len(roster))
        return tuple(roster[int(position)] for position in selected)
    sampled: list[str] = []
    for label in (0, 1):
        stratum = tuple(group for group in roster if group_labels[group] == label)
        selected = rng.integers(0, len(stratum), size=len(stratum))
        sampled.extend(stratum[int(position)] for position in selected)
    return tuple(sampled)


def _slow_grouped_draws(
    *,
    labels: Sequence[int],
    scores_by_method: Mapping[str, Sequence[float]],
    group_ids: Sequence[str],
    draws: int,
    seed: int,
    stratify_by_label: bool,
) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    """Original grouped loop, kept only as a regression oracle."""

    y = np.asarray(labels, dtype=np.int8)
    scores = {
        method_id: np.asarray(value, dtype=float)
        for method_id, value in scores_by_method.items()
    }
    methods = tuple(sorted(scores))
    roster, members = _members(group_ids, canonical_order=False)
    labels_by_group = (
        _group_labels(y, roster, members) if stratify_by_label else None
    )
    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in methods
    }
    valid_draws = 0
    for _ in range(draws):
        sampled = _sample_roster(roster, rng, labels_by_group)
        indices = np.concatenate([members[group] for group in sampled])
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) != 2:
            continue
        valid_draws += 1
        for method_id in methods:
            observed = binary_metric_values(sampled_y, scores[method_id][indices])
            for metric in METRIC_IDS:
                values[method_id][metric].append(observed[metric])
    return valid_draws, {
        method_id: {
            metric: np.asarray(values[method_id][metric], dtype=float)
            for metric in METRIC_IDS
        }
        for method_id in methods
    }


def _slow_population_draws(
    *,
    cells: Mapping[str, Mapping[str, object]],
    link_keys: Mapping[str, str] | None,
    draws: int,
    seed: int,
    stratify_by_label: bool,
) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    """Original linked-population loop, kept only as a regression oracle."""

    cell_ids = tuple(sorted(cells))
    methods = tuple(sorted(cells[cell_ids[0]]["scores_by_method"]))  # type: ignore[arg-type]
    effective_links = (
        {cell_id: link_keys[cell_id] for cell_id in cell_ids}
        if link_keys is not None
        else {cell_id: f"__independent__:{cell_id}" for cell_id in cell_ids}
    )
    state: dict[str, dict[str, object]] = {}
    for cell_id in cell_ids:
        cell = cells[cell_id]
        y = np.asarray(cell["labels"], dtype=np.int8)
        roster, members = _members(
            cell["group_ids"], canonical_order=True,  # type: ignore[arg-type]
        )
        state[cell_id] = {
            "labels": y,
            "scores": {
                method_id: np.asarray(value, dtype=float)
                for method_id, value in cell["scores_by_method"].items()  # type: ignore[union-attr]
            },
            "roster": roster,
            "members": members,
            "group_labels": (
                _group_labels(y, roster, members) if stratify_by_label else None
            ),
        }

    block_state: dict[str, dict[str, object]] = {}
    for link_key in sorted(set(effective_links.values())):
        first_cell = next(
            cell_id for cell_id in cell_ids
            if effective_links[cell_id] == link_key
        )
        block_state[link_key] = {
            "roster": state[first_cell]["roster"],
            "group_labels": state[first_cell]["group_labels"],
        }

    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        method_id: {metric: [] for metric in METRIC_IDS}
        for method_id in methods
    }
    valid_draws = 0
    for _ in range(draws):
        sampled_by_block = {
            link_key: _sample_roster(
                block_state[link_key]["roster"],  # type: ignore[arg-type]
                rng,
                block_state[link_key]["group_labels"],  # type: ignore[arg-type]
            )
            for link_key in sorted(block_state)
        }
        sampled_indices: dict[str, np.ndarray] = {}
        for cell_id in cell_ids:
            members = state[cell_id]["members"]
            sampled_indices[cell_id] = np.concatenate([
                members[group]  # type: ignore[index]
                for group in sampled_by_block[effective_links[cell_id]]
            ])
            labels = state[cell_id]["labels"]
            if len(np.unique(labels[sampled_indices[cell_id]])) != 2:  # type: ignore[index]
                break
        else:
            valid_draws += 1
            for method_id in methods:
                cell_metrics = []
                for cell_id in cell_ids:
                    indices = sampled_indices[cell_id]
                    labels = state[cell_id]["labels"]
                    scores = state[cell_id]["scores"]
                    cell_metrics.append(binary_metric_values(
                        labels[indices],  # type: ignore[index]
                        scores[method_id][indices],  # type: ignore[index]
                    ))
                for metric in METRIC_IDS:
                    values[method_id][metric].append(float(np.mean([
                        observed[metric] for observed in cell_metrics
                    ])))
    return valid_draws, {
        method_id: {
            metric: np.asarray(values[method_id][metric], dtype=float)
            for metric in METRIC_IDS
        }
        for method_id in methods
    }


class ExternalBootstrapFastPathTests(unittest.TestCase):
    def assert_summary_matches_draws(
        self,
        *,
        fast: Mapping[str, object],
        draw_values: Mapping[str, Mapping[str, np.ndarray]],
        reference_method: str = "iu_pcr",
    ) -> None:
        metrics = fast["metrics"]
        contrasts = fast["contrasts"]
        for method_id, method_draws in draw_values.items():
            for metric, values in method_draws.items():
                fast_metric = metrics[method_id][metric]  # type: ignore[index]
                self.assertEqual(fast_metric["valid_draws"], len(values))
                self.assertAlmostEqual(
                    fast_metric["ci_low"],
                    float(np.quantile(values, 0.025)),
                    delta=FROZEN_EQUIVALENCE_ATOL[metric],
                )
                self.assertAlmostEqual(
                    fast_metric["ci_high"],
                    float(np.quantile(values, 0.975)),
                    delta=FROZEN_EQUIVALENCE_ATOL[metric],
                )
                if method_id == reference_method:
                    continue
                delta_draws = values - draw_values[reference_method][metric]
                fast_contrast = contrasts[method_id][metric]  # type: ignore[index]
                self.assertAlmostEqual(
                    fast_contrast["ci_low"],
                    float(np.quantile(delta_draws, 0.025)),
                    delta=FROZEN_EQUIVALENCE_ATOL[metric],
                )
                self.assertAlmostEqual(
                    fast_contrast["ci_high"],
                    float(np.quantile(delta_draws, 0.975)),
                    delta=FROZEN_EQUIVALENCE_ATOL[metric],
                )
                self.assertAlmostEqual(
                    fast_contrast["probability_delta_le_zero"],
                    float(np.mean(
                        delta_draws <= FROZEN_NUMERICAL_ZERO_ATOL[metric]
                    )),
                    delta=0.0,
                )

    @staticmethod
    def tied_repeated_fixture() -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
        labels = np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int8)
        groups = (
            "neg_first", "neg_first", "positive", "positive",
            "neg_mid", "neg_last", "neg_last", "neg_last",
        )
        scores = {
            "iu_pcr": np.asarray([0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
            "candidate": np.asarray([0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.9, 0.9]),
            "constant": np.zeros(8, dtype=float),
        }
        return labels, groups, scores

    def test_non_stratified_ties_repeated_groups_and_rejected_draws_match(self) -> None:
        labels, groups, scores = self.tied_repeated_fixture()
        kwargs = dict(
            labels=labels,
            scores_by_method=scores,
            group_ids=groups,
            draws=511,
            seed=329,
            stratify_by_label=False,
        )
        fast = grouped_paired_bootstrap(**kwargs)
        valid_draws, slow = _slow_grouped_draws(**kwargs)
        self.assertEqual(fast["valid_draws"], valid_draws)
        self.assertLess(valid_draws, kwargs["draws"])
        self.assert_summary_matches_draws(fast=fast, draw_values=slow)

    def test_count_sampler_matches_materialized_roster_and_rng_stream(self) -> None:
        labels, groups, _ = self.tied_repeated_fixture()
        roster, members = _members(groups, canonical_order=False)
        labels_by_group = _group_labels(labels, roster, members)
        for stratified in (False, True):
            group_labels = labels_by_group if stratified else None
            count_rng = np.random.default_rng(1987)
            roster_rng = np.random.default_rng(1987)
            for _ in range(101):
                counts = external_evaluation._sample_group_counts(
                    roster=roster,
                    rng=count_rng,
                    group_labels=group_labels,
                )
                sampled = _sample_roster(roster, roster_rng, group_labels)
                expected = np.asarray(
                    [sampled.count(group) for group in roster], dtype=np.int64,
                )
                np.testing.assert_array_equal(counts, expected)
            self.assertEqual(count_rng.integers(0, 2**31), roster_rng.integers(0, 2**31))

    def test_weighted_metrics_match_every_materialized_draw(self) -> None:
        labels, groups, scores = self.tied_repeated_fixture()
        roster, members = _members(groups, canonical_order=False)
        row_positions = external_evaluation._row_group_positions(
            n_rows=len(labels), roster=roster, members=members,
        )
        rng = np.random.default_rng(451)
        counts: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        while len(counts) < 97:
            sampled = _sample_roster(roster, rng, None)
            sampled_indices = np.concatenate([members[group] for group in sampled])
            if len(np.unique(labels[sampled_indices])) != 2:
                continue
            counts.append(np.asarray(
                [sampled.count(group) for group in roster], dtype=np.int64,
            ))
            indices.append(sampled_indices)
        count_matrix = np.stack(counts, axis=0)
        for method_id, score in scores.items():
            plan = external_evaluation._metric_plan(
                labels=labels,
                score=score,
                row_group_positions=row_positions,
            )
            fast = external_evaluation._weighted_binary_metric_batch(
                group_counts=count_matrix, plan=plan,
            )
            slow = [
                binary_metric_values(labels[index], score[index])
                for index in indices
            ]
            for metric in METRIC_IDS:
                np.testing.assert_allclose(
                    fast[metric],
                    np.asarray([value[metric] for value in slow]),
                    rtol=0.0,
                    atol=FROZEN_EQUIVALENCE_ATOL[metric],
                    err_msg=f"{method_id}/{metric}",
                )

    def test_batch_boundaries_do_not_change_rng_or_serialized_values(self) -> None:
        labels, groups, scores = self.tied_repeated_fixture()
        kwargs = dict(
            labels=labels,
            scores_by_method=scores,
            group_ids=groups,
            draws=73,
            seed=701,
        )
        with mock.patch.object(
            external_evaluation, "_BOOTSTRAP_MAX_BATCH_DRAWS", 1,
        ):
            one_draw_batches = grouped_paired_bootstrap(**kwargs)
        with mock.patch.object(
            external_evaluation, "_BOOTSTRAP_MAX_BATCH_DRAWS", 10_000,
        ), mock.patch.object(
            external_evaluation, "_BOOTSTRAP_TARGET_ROW_WEIGHTS", 10_000_000,
        ):
            one_large_batch = grouped_paired_bootstrap(**kwargs)
        self.assertEqual(one_draw_batches, one_large_batch)

    def test_stratified_ties_and_repeated_groups_match_every_draw(self) -> None:
        labels, groups, scores = self.tied_repeated_fixture()
        kwargs = dict(
            labels=labels,
            scores_by_method=scores,
            group_ids=groups,
            draws=257,
            seed=991,
            stratify_by_label=True,
        )
        fast = grouped_paired_bootstrap(**kwargs)
        valid_draws, slow = _slow_grouped_draws(**kwargs)
        self.assertEqual(valid_draws, kwargs["draws"])
        self.assertEqual(fast["valid_draws"], valid_draws)
        self.assert_summary_matches_draws(fast=fast, draw_values=slow)

    def test_linked_population_preserves_rng_order_and_equal_cell_estimand(self) -> None:
        labels, groups, scores = self.tied_repeated_fixture()
        cells = {
            "linked_a": {
                "labels": labels,
                "group_ids": groups,
                "scores_by_method": scores,
            },
            "linked_b": {
                "labels": labels,
                "group_ids": groups,
                "scores_by_method": {
                    method_id: np.roll(value, 1)
                    for method_id, value in scores.items()
                },
            },
            "independent": {
                "labels": [0, 1, 0, 0, 0],
                "group_ids": ["i0", "i1", "i2", "i3", "i4"],
                "scores_by_method": {
                    "iu_pcr": [0.1, 0.8, 0.3, 0.2, 0.6],
                    "candidate": [0.3, 0.7, 0.2, 0.1, 0.9],
                    "constant": [0.0] * 5,
                },
            },
        }
        link_keys = {
            "linked_a": "shared_questions",
            "linked_b": "shared_questions",
            "independent": "other_questions",
        }
        kwargs = dict(
            cells=cells,
            link_keys=link_keys,
            draws=313,
            seed=771,
            stratify_by_label=False,
        )
        fast = population_grouped_paired_bootstrap(**kwargs)
        valid_draws, slow = _slow_population_draws(**kwargs)
        self.assertEqual(fast["valid_draws"], valid_draws)
        self.assertLess(valid_draws, kwargs["draws"])
        self.assert_summary_matches_draws(fast=fast, draw_values=slow)

    def test_stratified_population_matches_slow_reference(self) -> None:
        cells = {
            "hle": {
                "labels": [0, 0, 0, 1, 1],
                "group_ids": ["n0", "n1", "n2", "p0", "p1"],
                "scores_by_method": {
                    "iu_pcr": [0.1, 0.4, 0.4, 0.4, 0.9],
                    "candidate": [0.2, 0.2, 0.7, 0.7, 0.7],
                },
            },
        }
        kwargs = dict(
            cells=cells,
            link_keys=None,
            draws=211,
            seed=4070942594,
            stratify_by_label=True,
        )
        fast = population_grouped_paired_bootstrap(
            **kwargs, weighting="single_cell",
        )
        valid_draws, slow = _slow_population_draws(**kwargs)
        self.assertEqual(valid_draws, kwargs["draws"])
        self.assertEqual(fast["valid_draws"], valid_draws)
        self.assert_summary_matches_draws(fast=fast, draw_values=slow)

    def test_equal_auroc_different_rankings_are_canonical_numerical_ties(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0, 0] * 2, dtype=np.int8)
        candidate = np.asarray([0.0, 3.0, 1.0, 2.0, 4.0, 5.0, 6.0] * 2)
        reference = np.asarray([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0] * 2)
        result = grouped_paired_bootstrap(
            labels=labels,
            scores_by_method={"iu_pcr": reference, "candidate": candidate},
            group_ids=["g0"] * 7 + ["g1"] * 7,
            draws=17,
            seed=1,
        )
        contrast = result["contrasts"]["candidate"]["auroc"]
        self.assertGreater(contrast["delta"], 0.0)
        self.assertLess(contrast["delta"], FROZEN_NUMERICAL_ZERO_ATOL["auroc"])
        self.assertEqual(contrast["probability_delta_le_zero"], 1.0)

    def test_identical_ranking_contrast_probability_is_exactly_one(self) -> None:
        labels = [0, 0, 1, 1, 0, 1]
        reference = np.asarray([0.0, 0.2, 0.4, 0.7, 0.1, 0.9])
        result = grouped_paired_bootstrap(
            labels=labels,
            scores_by_method={
                "iu_pcr": reference,
                "candidate": 10.0 + 3.0 * reference,
            },
            group_ids=[f"g{index}" for index in range(len(labels))],
            draws=101,
            seed=88,
        )
        for metric in METRIC_IDS:
            self.assertEqual(
                result["contrasts"]["candidate"][metric][
                    "probability_delta_le_zero"
                ],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
