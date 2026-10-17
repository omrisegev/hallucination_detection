import inspect
import itertools
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import spectral_utils.error_dependence as dependence_backend
from spectral_utils.error_dependence import (
    AtlasBackendError,
    ConditionedErrors,
    DependenceStatus,
    ErrorMatrix,
    ErrorTarget,
    ObservationMetadata,
    PairInterval,
    Resolution,
    apply_label_free_fusion_weights,
    bootstrap_pair_diagnostics,
    build_compatibility_graph,
    classify_pair,
    combine_error_matrices,
    cross_fit_nuisance,
    deterministic_group_folds,
    enumerate_cliques,
    final_pb_error_matrix,
    full_population_gate_bootstrap,
    fit_label_free_fusion_weights,
    gate_error_matrices_from_scores,
    gate_false_close_matrix,
    gate_false_open_matrix,
    group_diagnostics,
    pair_diagnostics,
    pb_raw_locator_miss_matrix,
    predictor_residual_matrix,
    prmb_pairwise_misorder_matrix,
    run_atlas_dependence,
    run_nested_fusion_search,
    simultaneous_intervals,
    source_group_bootstrap,
)

from spectral_utils.fusion_signal_registry import READOUT_NAMES


def naive_block_rank_columns(scores, active_mask, block_ids=None):
    """Slow reference for the exact answer/cell-local fusion rank contract."""
    scores = np.asarray(scores, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool).copy()
    blocks = (
        np.zeros(len(scores), dtype=np.int64)
        if block_ids is None
        else np.asarray(block_ids)
    )
    ranks = np.zeros_like(scores)
    live = np.zeros(scores.shape[1], dtype=bool)
    for column in range(scores.shape[1]):
        active[:, column] &= np.isfinite(scores[:, column])
        all_rows = np.flatnonzero(active[:, column])
        if not len(all_rows):
            continue
        for block in np.unique(blocks[all_rows]):
            rows = np.flatnonzero(active[:, column] & (blocks == block))
            if not len(rows):
                continue
            values = scores[rows, column]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            local_ranks = np.empty(len(rows), dtype=np.float64)
            start = 0
            while start < len(rows):
                stop = start + 1
                while stop < len(rows) and sorted_values[stop] == sorted_values[start]:
                    stop += 1
                # scipy.stats.rankdata(method="average"), followed by the
                # production half-rank normalization.
                local_ranks[order[start:stop]] = (start + stop + 1.0) / 2.0
                start = stop
            ranks[rows, column] = (local_ranks - 0.5) / len(rows)
            live[column] |= len(rows) > 1 and np.ptp(values) > 1e-12
    return ranks, active, live


def naive_materialized_clique_samples(matrix, cliques, conditioned, draws, seed):
    """Legacy all-cliques-in-RAM reference for streaming equivalence tests."""
    unique_groups, row_group = np.unique(matrix.metadata.groups, return_inverse=True)
    row_group = np.asarray(row_group, dtype=np.int64)
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros((draws, len(unique_groups)), dtype=np.int32)
    for draw in range(draws):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        multiplicities[draw] = np.bincount(sampled, minlength=len(unique_groups))
    name_to_index = {name: index for index, name in enumerate(matrix.method_names)}
    points = []
    sample_columns = []
    supports = []
    for members in cliques:
        indexes = np.asarray([name_to_index[name] for name in members], dtype=np.int64)
        active = np.all(matrix.active[:, indexes], axis=1)
        residuals = conditioned.residuals[np.ix_(active, indexes)]
        local_group = row_group[active]
        point = dependence_backend._weighted_group_point(
            residuals, np.ones(len(residuals)),
        )
        width = len(indexes)
        sufficient = np.zeros(
            (len(unique_groups), 1 + width + width * width), dtype=np.float64,
        )
        sufficient[:, 0] = np.bincount(local_group, minlength=len(unique_groups))
        for column in range(width):
            sufficient[:, 1 + column] = np.bincount(
                local_group,
                weights=residuals[:, column],
                minlength=len(unique_groups),
            )
        cursor = 1 + width
        for left in range(width):
            for right in range(width):
                sufficient[:, cursor + left * width + right] = np.bincount(
                    local_group,
                    weights=residuals[:, left] * residuals[:, right],
                    minlength=len(unique_groups),
                )
        totals = multiplicities @ sufficient
        n = totals[:, 0]
        sums = totals[:, 1:1 + width]
        cross = totals[:, 1 + width:].reshape(draws, width, width)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = sums / n[:, None]
            covariance = cross / n[:, None, None] - mean[:, :, None] * mean[:, None, :]
            scale = np.sqrt(
                np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0)
            )
            correlation = covariance / (scale[:, :, None] * scale[:, None, :])
        invalid = (
            (n <= 0)
            | np.any(scale <= 1e-12, axis=1)
            | ~np.isfinite(correlation).all(axis=(1, 2))
        )
        triangle = np.triu_indices(width, 1)
        maximum = np.max(np.abs(correlation[:, triangle[0], triangle[1]]), axis=1)
        safe = np.where(np.isfinite(correlation), correlation, 0.0)
        eigenvalues = np.maximum(
            np.linalg.eigvalsh((safe + np.swapaxes(safe, 1, 2)) / 2.0),
            0.0,
        )
        denominator = np.sum(eigenvalues * eigenvalues, axis=1)
        effective = np.divide(
            np.sum(eigenvalues, axis=1) ** 2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-15,
        )
        samples = np.column_stack((maximum, effective))
        samples[invalid] = np.nan
        points.extend(point)
        sample_columns.append(samples)
        supports.append((int(active.sum()), int(np.unique(row_group[active]).size)))
    return (
        np.asarray(points, dtype=np.float64),
        np.concatenate(sample_columns, axis=1),
        supports,
    )


def metadata(n_groups=30, per_group=20, *, with_folds=True):
    n = n_groups * per_group
    groups = np.repeat([f"g{index:03d}" for index in range(n_groups)], per_group)
    cells = np.where(np.arange(n) % 3 == 0, "pb_math_q4", "pb_math_q8")
    folds = np.repeat(np.arange(n_groups) % 5, per_group) if with_folds else None
    return ObservationMetadata(
        groups=groups,
        cells=cells,
        tokens=10 + np.arange(n) % 200,
        steps=1 + np.arange(n) % 12,
        folds=folds,
        first_error_position=(np.arange(n) % 11) / 10,
        digit_opportunities=np.arange(n) % 8,
    )


def binary_matrix(values, meta=None, names=None, scores=None, peaks=None, topk=None, active=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    meta = metadata(len(values) // 20, 20) if meta is None else meta
    names = tuple(f"m{index}" for index in range(values.shape[1])) if names is None else tuple(names)
    return ErrorMatrix(
        ErrorTarget.PB_FINAL_ERROR,
        Resolution.ANSWER,
        names,
        values,
        meta,
        scores=scores,
        peaks=peaks,
        topk_sets=topk,
        active=active,
    )


def fixed_conditioning(matrix, expected):
    expected = np.asarray(expected, dtype=float)
    return ConditionedErrors(
        expected=expected,
        residuals=np.where(matrix.active, matrix.values - expected, np.nan),
        folds=matrix.metadata.folds,
        design_columns=(),
        model_kinds=tuple(() for _ in matrix.method_names),
        train_groups_by_fold={},
        source_identity=matrix.identity,
        method_names=matrix.method_names,
        active=matrix.active.copy(),
    )


class ProductionScalePrimitiveTests(unittest.TestCase):
    def test_precomputed_block_ranks_are_exact_and_skip_reranking(self):
        scores = np.asarray((
            (0.2, 4.0, 8.0), (0.7, 1.0, 3.0), (0.4, 2.0, 5.0),
            (9.0, 7.0, 2.0), (8.0, 6.0, 1.0), (7.0, 5.0, 4.0),
        ))
        active = np.ones_like(scores, dtype=bool)
        active[1, 2] = False
        blocks = np.asarray((0, 0, 0, 1, 1, 1))
        ranks, _, _ = dependence_backend._rank_columns_for_fusion(
            scores, active, blocks,
        )
        expected_weights, _ = fit_label_free_fusion_weights(
            scores, active, ("a", "b", "c"), "nonnegative_shrunk_simplex",
            block_ids=blocks,
        )
        expected_output = apply_label_free_fusion_weights(
            scores, active, expected_weights, block_ids=blocks,
            families=("a", "b", "c"), head="nonnegative_shrunk_simplex",
        )
        with mock.patch.object(
            dependence_backend, "_rank_columns_for_fusion",
            side_effect=AssertionError("precomputed path reranked raw scores"),
        ):
            observed_weights, _ = fit_label_free_fusion_weights(
                ranks, active, ("a", "b", "c"), "nonnegative_shrunk_simplex",
                block_ids=blocks, precomputed_ranks=ranks,
            )
            observed_output = apply_label_free_fusion_weights(
                ranks, active, observed_weights, block_ids=blocks,
                families=("a", "b", "c"), head="nonnegative_shrunk_simplex",
                precomputed_ranks=ranks,
            )
        np.testing.assert_allclose(observed_weights, expected_weights, atol=1e-12, rtol=0)
        np.testing.assert_allclose(observed_output[0], expected_output[0], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(observed_output[1], expected_output[1])

    def test_grouped_rank_optimization_is_exact_for_noncontiguous_tied_inactive_rows(self):
        scores = np.asarray((
            (2.0, 4.0, np.nan, 7.0, 1.0),
            (9.0, 1.0, 2.0, 7.0, np.inf),
            (2.0, 3.0, 5.0, 7.0, 1.0),
            (1.0, 1.0, 4.0, 7.0, 8.0),
            (9.0, 2.0, 2.0, 7.0, 3.0),
            (1.0, 3.0, 4.0, 7.0, 8.0),
            (2.0, 2.0, 3.0, 7.0, 2.0),
            (9.0, 4.0, 3.0, 7.0, 3.0),
            (1.0, 5.0, 5.0, 7.0, 2.0),
            (2.0, 5.0, 8.0, 7.0, 4.0),
            (9.0, 6.0, 8.0, 7.0, 4.0),
            (1.0, 6.0, 9.0, 7.0, 9.0),
        ))
        # Deliberately interleave the blocks.  Production answer rows are
        # usually contiguous, but filtering and cell-level gates need this
        # general path to remain exact too.
        blocks = np.asarray(("z", "a", "z", "m", "a", "m", "z", "a", "m", "z", "a", "m"))
        active = np.ones_like(scores, dtype=bool)
        active[[0, 4, 7], [0, 1, 2]] = False
        active[:, 4] = False
        active[1, 4] = True  # active but nonfinite: must be deactivated

        expected = naive_block_rank_columns(scores, active, blocks)
        observed = dependence_backend._rank_columns_for_fusion(scores, active, blocks)
        np.testing.assert_array_equal(observed[0], expected[0])
        np.testing.assert_array_equal(observed[1], expected[1])
        np.testing.assert_array_equal(observed[2], expected[2])
        # The constant fourth column is not live even though it has support in
        # every block; the entirely inactive fifth column is also not live.
        np.testing.assert_array_equal(observed[2][-2:], (False, False))

    def test_grouped_rank_path_does_not_rescan_all_rows_per_block(self):
        block_count = 400
        repeats = 3
        blocks = np.tile(np.arange(block_count), repeats)
        scores = np.column_stack((
            np.arange(len(blocks), dtype=float) % 11,
            np.arange(len(blocks), dtype=float) % 7,
            np.arange(len(blocks), dtype=float) % 5,
        ))
        active = np.ones_like(scores, dtype=bool)
        original = np.flatnonzero
        scanned_sizes = []

        def tracked_flatnonzero(values):
            scanned_sizes.append(np.asarray(values).size)
            return original(values)

        with mock.patch.object(
            dependence_backend.np, "flatnonzero", side_effect=tracked_flatnonzero,
        ):
            dependence_backend._rank_columns_for_fusion(scores, active, blocks)
        # This is a structural runtime guard, not a wall-clock assertion.  The
        # former implementation made one *full-row* flatnonzero scan for every
        # (column, block).  Local scans inside an already grouped block remain
        # cheap and are allowed; only the boundary plan may scan the full axis.
        full_axis_scans = sum(size >= len(blocks) - 1 for size in scanned_sizes)
        self.assertLessEqual(full_axis_scans, 1)

    def test_metadata_fold_validation_is_vectorized_and_rejects_split_groups(self):
        group_count = 1_000
        repeats = 4
        permutation = np.random.default_rng(51).permutation(group_count * repeats)
        groups = np.repeat([f"g{index:04d}" for index in range(group_count)], repeats)[permutation]
        group_numbers = np.repeat(np.arange(group_count), repeats)[permutation]
        folds = group_numbers % 5
        cells = np.where(group_numbers % 2, "pb_a", "pb_b")
        original = np.unique
        with mock.patch.object(
            dependence_backend.np, "unique", side_effect=original,
        ) as calls:
            observed = ObservationMetadata(
                groups=groups,
                cells=cells,
                tokens=np.ones(len(groups)),
                steps=np.ones(len(groups)),
                folds=folds,
            )
        self.assertEqual(len(observed), len(groups))
        # Stable operation-count guard: validation may obtain group codes and
        # grouped extrema, but must not invoke unique once per source group.
        self.assertLessEqual(calls.call_count, 4)

        bad_folds = folds.copy()
        first_group_rows = np.flatnonzero(groups == groups[0])
        bad_folds[first_group_rows[-1]] = (bad_folds[first_group_rows[-1]] + 1) % 5
        with self.assertRaisesRegex(ValueError, "source group cannot occur in multiple folds"):
            ObservationMetadata(
                groups=groups,
                cells=cells,
                tokens=np.ones(len(groups)),
                steps=np.ones(len(groups)),
                folds=bad_folds,
            )

    def test_group_rows_preserves_sorted_names_and_original_row_order_without_group_scans(self):
        group_count = 500
        repeats = 5
        rng = np.random.default_rng(79)
        groups = np.repeat([f"g{index:04d}" for index in range(group_count)], repeats)
        groups = groups[rng.permutation(len(groups))]
        expected_names = np.asarray(sorted(set(groups)), dtype=str)
        expected_rows = tuple(np.flatnonzero(groups == group) for group in expected_names)

        original = np.flatnonzero
        with mock.patch.object(
            dependence_backend.np, "flatnonzero", side_effect=original,
        ) as calls:
            names, rows = dependence_backend._group_rows(groups)
        np.testing.assert_array_equal(names, expected_names)
        self.assertEqual(len(rows), len(expected_rows))
        for observed, expected in zip(rows, expected_rows):
            np.testing.assert_array_equal(observed, expected)
        # The result can be built by one stable grouping/sort pass; a full
        # equality scan per group is the production-scale failure mode.
        self.assertLessEqual(calls.call_count, 2)


class ErrorMatrixContractTests(unittest.TestCase):
    def test_prmb_label_one_is_the_positive_high_risk_orientation(self):
        labels = np.asarray((1, 0, 1, 0), dtype=np.int8)
        scores = np.asarray((4.0, 1.0, 3.0, 2.0))
        active = np.ones(4, dtype=bool)
        self.assertEqual(
            dependence_backend._risk_pair_auc(labels, scores, active),
            1.0,
        )

        pair_meta = ObservationMetadata(
            groups=np.asarray(("g0", "g0", "g0", "g0")),
            cells=np.asarray(("prmbench_fixture",) * 4),
            tokens=np.full(4, 4),
            steps=np.full(4, 4),
            folds=np.zeros(4, dtype=np.int64),
        )
        direct = prmb_pairwise_misorder_matrix(
            np.asarray((4.0, 4.0, 3.0, 3.0)),
            np.asarray((1.0, 2.0, 1.0, 2.0)),
            ("perfect",),
            pair_meta,
        )
        np.testing.assert_array_equal(direct.values, 0.0)

        inputs = SimpleNamespace(
            offsets=np.asarray((0, 4), dtype=np.int64),
            joined={"labels": labels},
            metadata=({
                "cell": "prmbench_fixture",
                "group_id": "g0",
                "tokens": 4,
                "fold": 0,
            },),
        )
        consolidated = SimpleNamespace(
            names=("perfect",),
            points=("step_post_readout",),
            step_scores=scores[:, None],
            step_active=np.ones((4, 1), dtype=bool),
        )
        integrated = dependence_backend._prmb_pair_matrix(
            inputs, consolidated, "step_post_readout",
        )
        self.assertIsNotNone(integrated)
        np.testing.assert_array_equal(integrated.values, 0.0)

    def test_builders_preserve_separate_targets_and_prmb_ties(self):
        meta = metadata(2, 3, with_folds=False)
        residual = predictor_residual_matrix(
            np.arange(6.0), np.column_stack((np.arange(6.0), np.arange(6.0) + 1)), ("a", "b"), meta
        )
        self.assertEqual(residual.target, ErrorTarget.PREDICTOR_RESIDUAL)
        np.testing.assert_array_equal(residual.values[:, 1], -1)

        pair = prmb_pairwise_misorder_matrix(
            np.array([[2, 1], [1, 2], [1, 2], [2, 1], [1, 1], [3, 0]], float),
            np.array([[1, 1], [2, 1], [1, 2], [1, 2], [1, 2], [3, 1]], float),
            ("a", "b"),
            meta,
        )
        self.assertEqual(pair.resolution, Resolution.PAIR)
        self.assertEqual(pair.values[0, 1], 0.5)
        self.assertEqual(pair.values[1, 0], 1.0)
        self.assertEqual(pair.values[2, 0], 0.5)

        with self.assertRaisesRegex(ValueError, "different error targets"):
            combine_error_matrices(residual, pair)
        with self.assertRaisesRegex(ValueError, "requires token"):
            ErrorMatrix(ErrorTarget.PREDICTOR_RESIDUAL, Resolution.ANSWER, ("x",), np.zeros((6, 1)), meta)

    def test_pb_and_gate_error_definitions(self):
        meta = metadata(2, 3, with_folds=False)
        peaks = np.array([[0, 1], [2, 1], [0, 0], [1, 1], [4, 3], [0, 0]])
        targets = np.array([0, 1, -1, 1, -1, 2])
        valid = np.ones_like(peaks, dtype=bool)
        valid[1, 0] = False
        raw = pb_raw_locator_miss_matrix(peaks, targets, valid, ("a", "b"), meta)
        self.assertEqual(raw.shape, (4, 2))
        np.testing.assert_array_equal(raw.values[0], [0, 1])
        np.testing.assert_array_equal(raw.values[1], [1, 0])

        opened = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 1], [0, 1]], dtype=bool)
        clean = targets == -1
        erroneous = targets >= 0
        false_open = gate_false_open_matrix(opened, clean, ("a", "b"), meta)
        false_close = gate_false_close_matrix(opened, erroneous, ("a", "b"), meta)
        np.testing.assert_array_equal(false_open.values, opened[clean].astype(float))
        np.testing.assert_array_equal(false_close.values, (~opened[erroneous]).astype(float))
        with self.assertRaisesRegex(ValueError, "different error targets"):
            combine_error_matrices(false_open, false_close)

        final = final_pb_error_matrix(peaks, targets, valid, ("a", "b"), meta)
        self.assertEqual(final.target, ErrorTarget.PB_FINAL_ERROR)
        self.assertEqual(final.values[1, 0], 1)

    def test_high_level_inactive_views_remain_defined_population_outcomes(self):
        meta = metadata(2, 3, with_folds=False)
        targets = np.array([0, 1, -1, 1, -1, 2])
        peaks = np.array([[0, 0], [-1, 1], [0, 0], [1, -1], [0, 0], [0, 2]])
        valid = peaks >= 0
        raw = pb_raw_locator_miss_matrix(
            peaks,
            targets,
            valid,
            ("a", "b"),
            meta,
            retain_invalid_as_failures=True,
        )
        self.assertTrue(raw.active.all())
        self.assertEqual(raw.values[1, 0], 1.0)

        opened = np.zeros_like(peaks, dtype=bool)
        active = np.ones_like(opened, dtype=bool)
        active[0, 1] = False
        active[1, 0] = False
        errors = gate_error_matrices_from_scores(
            np.arange(12, dtype=float).reshape(6, 2),
            ("a", "b"),
            {"a": ("a",), "b": ("b",)},
            targets >= 0,
            meta,
            detector_active=active,
            retain_inactive_as_closed=True,
        )
        self.assertTrue(errors.false_open.active.all())
        self.assertTrue(errors.false_close.active.all())

    def test_boolean_take_and_string_safe_homogeneous_combine(self):
        meta = metadata(2, 3, with_folds=False)
        left = binary_matrix(np.arange(6)[:, None] % 2, meta, ("left",))
        right = binary_matrix((np.arange(6)[:, None] + 1) % 2, meta, ("right",))
        combined = combine_error_matrices(left, right)
        self.assertEqual(combined.shape, (6, 2))
        mask = np.array([True, False, False, True, False, True])
        selected = combined.take(mask)
        np.testing.assert_array_equal(selected.values, combined.values[[0, 3, 5]])
        np.testing.assert_array_equal(selected.metadata.groups, meta.groups[[0, 3, 5]])

    def test_pairwise_error_support_is_separate_from_optional_evidence_masks(self):
        rng = np.random.default_rng(12)
        meta = metadata(25, 8)
        values = rng.binomial(1, 0.4, size=(len(meta), 2))
        active = np.ones_like(values, dtype=bool)
        active[:40, 1] = False
        peaks = np.zeros_like(values)
        peaks[40:55, 0] = -1
        topk = (
            tuple(None if index in range(55, 65) else {index, index + 1} for index in range(len(meta))),
            tuple({index, index + 2} for index in range(len(meta))),
        )
        matrix = binary_matrix(values, meta, ("a", "digit"), peaks=peaks, topk=topk, active=active)
        expected_active = active[:, 0] & active[:, 1]
        conditioned = cross_fit_nuisance(matrix)
        diagnostics = pair_diagnostics(matrix, "a", "digit", conditioned=conditioned)
        self.assertEqual(diagnostics.n, int(expected_active.sum()))
        self.assertEqual(diagnostics.source_groups, len(np.unique(meta.groups[expected_active])))
        self.assertTrue(np.isfinite(diagnostics.topk_jaccard))
        self.assertTrue(np.isfinite(diagnostics.peak_agreement))

        reordered = matrix.take(np.arange(len(meta))[::-1])
        with self.assertRaisesRegex(ValueError, "different matrix or row order"):
            pair_diagnostics(reordered, 0, 1, conditioned=conditioned)


class ConditioningAndPairTests(unittest.TestCase):
    def _synthetic(self, seed=8):
        rng = np.random.default_rng(seed)
        meta = metadata(50, 30)
        cell = (meta.cells == "pb_math_q4").astype(float)
        probability = 1 / (1 + np.exp(-(-1.0 + 1.2 * cell + 0.25 * np.log1p(meta.steps))))
        independent_a = rng.binomial(1, probability)
        independent_b = rng.binomial(1, probability)
        dependent = independent_a.copy()
        values = np.column_stack((independent_a, independent_b, dependent))
        scores = values + rng.normal(0, 0.05, size=values.shape)
        peaks = np.column_stack((np.arange(len(values)) % 4,) * 3)
        return binary_matrix(values, meta, ("a", "b", "copy_a"), scores=scores, peaks=peaks)

    def test_group_folds_are_deterministic_and_never_leak(self):
        meta = metadata(17, 7, with_folds=False)
        first = deterministic_group_folds(meta.groups, seed=19)
        second = deterministic_group_folds(meta.groups, seed=19)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 5)
        for group in np.unique(meta.groups):
            self.assertEqual(len(np.unique(first[meta.groups == group])), 1)

        rng = np.random.default_rng(4)
        matrix = binary_matrix(rng.binomial(1, 0.4, size=(len(meta), 2)), meta)
        conditioned = cross_fit_nuisance(matrix, seed=19)
        for fold, train_groups in conditioned.train_groups_by_fold.items():
            held_groups = set(meta.groups[conditioned.folds == fold])
            self.assertFalse(held_groups & set(train_groups))
        self.assertTrue(all("balanced_logistic" in kinds for kinds in conditioned.model_kinds))

    def test_conditional_pair_diagnostics_separate_independent_and_copy(self):
        matrix = self._synthetic()
        conditioned = cross_fit_nuisance(matrix)
        independent = pair_diagnostics(matrix, "a", "b", conditioned=conditioned)
        dependent = pair_diagnostics(matrix, "a", "copy_a", conditioned=conditioned)
        self.assertLess(abs(independent.conditional_phi), 0.12)
        self.assertLess(independent.conditional_odds_ratio, 1.7)
        self.assertGreater(dependent.conditional_phi, 0.95)
        self.assertGreater(dependent.conditional_odds_ratio, 20)
        self.assertTrue(dependent.exact_error_match)
        self.assertEqual(dependent.peak_agreement, 1.0)

    def test_pair_status_support_complementarity_and_redundancy(self):
        matrix = self._synthetic()
        conditioned = cross_fit_nuisance(matrix)
        independent = pair_diagnostics(matrix, "a", "b", conditioned=conditioned)
        dependent = pair_diagnostics(matrix, "a", "copy_a", conditioned=conditioned)
        self.assertEqual(
            classify_pair(independent, PairInterval(-0.1, 0.1, 0.8, 1.2)),
            DependenceStatus.INDEPENDENCE_COMPATIBLE,
        )
        bad = PairInterval(0.5, 0.9, 3.0, 30.0)
        self.assertEqual(classify_pair(dependent, bad), DependenceStatus.REDUNDANT)
        complementary = pair_diagnostics(matrix, "a", "b", conditioned=conditioned)
        self.assertEqual(
            classify_pair(complementary, bad, held_fold_unique_successes=20, oof_fusion_improved=True),
            DependenceStatus.DEPENDENT_COMPLEMENTARY,
        )
        too_small = replace_pair_support(independent, source_groups=5)
        self.assertEqual(classify_pair(too_small, PairInterval(-0.1, 0.1, 0.9, 1.1)), DependenceStatus.UNRESOLVED)

    def test_pair_bootstrap_is_deterministic_and_uses_common_draws(self):
        matrix = self._synthetic(seed=11)
        first = bootstrap_pair_diagnostics(matrix, draws=30, seed=71)
        second = bootstrap_pair_diagnostics(matrix, draws=30, seed=71)
        self.assertTrue(first.common_draws)
        self.assertEqual(first.intervals, second.intervals)
        self.assertEqual(first.statuses, second.statuses)
        self.assertEqual(first.finite_counts, second.finite_counts)
        self.assertEqual(first.fully_finite_draws, 30)

    def test_group_sufficient_statistic_bootstrap_matches_explicit_resampling(self):
        matrix = self._synthetic(seed=29)
        conditioned = cross_fit_nuisance(matrix)
        draws = 20
        seed = 113
        fast = bootstrap_pair_diagnostics(
            matrix, conditioned=conditioned, draws=draws, seed=seed,
        )
        pairs = [
            (left, right)
            for left in range(matrix.shape[1])
            for right in range(left + 1, matrix.shape[1])
        ]

        def vector(sample, local_conditioned):
            diagnostics = [
                pair_diagnostics(sample, left, right, conditioned=local_conditioned)
                for left, right in pairs
            ]
            phi = [row.conditional_phi for row in diagnostics]
            log_odds = [np.log(row.conditional_odds_ratio) for row in diagnostics]
            return np.asarray(phi + log_odds)

        point = vector(matrix, conditioned)
        unique = np.asarray(sorted(set(matrix.metadata.groups)))
        rows = tuple(np.flatnonzero(matrix.metadata.groups == group) for group in unique)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(draws):
            chosen = rng.integers(0, len(rows), size=len(rows))
            indices = np.concatenate([rows[index] for index in chosen])
            sample = matrix.take(indices)
            samples.append(vector(sample, conditioned.take(indices, sample)))
        expected = simultaneous_intervals(point, np.asarray(samples))
        for position, (left, right) in enumerate(pairs):
            interval = fast.intervals[(matrix.method_names[left], matrix.method_names[right])]
            np.testing.assert_allclose(
                (interval.phi_low, interval.phi_high), expected[position], atol=1e-12, rtol=0,
            )
            np.testing.assert_allclose(
                (np.log(interval.odds_ratio_low), np.log(interval.odds_ratio_high)),
                expected[len(pairs) + position], atol=1e-12, rtol=0,
            )

    def test_continuous_residual_screen_uses_correlation_not_binary_or(self):
        rng = np.random.default_rng(91)
        meta = metadata(30, 15)
        observed = rng.normal(size=len(meta))
        predictions = np.column_stack((observed - rng.normal(size=len(meta)), observed - rng.normal(size=len(meta))))
        matrix = predictor_residual_matrix(observed, predictions, ("mean16", "ridge"), meta)
        conditioned = cross_fit_nuisance(matrix)
        diagnostics = pair_diagnostics(matrix, 0, 1, conditioned=conditioned)
        self.assertTrue(np.isnan(diagnostics.conditional_odds_ratio))
        self.assertTrue(np.isnan(diagnostics.left_successes))
        self.assertEqual(
            classify_pair(diagnostics, PairInterval(-0.10, 0.12, np.nan, np.nan)),
            DependenceStatus.INDEPENDENCE_COMPATIBLE,
        )
        self.assertEqual(
            classify_pair(diagnostics, PairInterval(-0.10, 0.30, np.nan, np.nan)),
            DependenceStatus.UNRESOLVED,
        )


def replace_pair_support(value, **changes):
    data = dict(value.__dict__)
    data.update(changes)
    return type(value)(**data)


class BootstrapGroupAndGraphTests(unittest.TestCase):
    def test_source_group_bootstrap_callback_runs_inside_each_draw(self):
        rng = np.random.default_rng(3)
        meta = metadata(20, 5)
        matrix = binary_matrix(rng.binomial(1, 0.5, size=(len(meta), 2)), meta)
        calls = []

        def recalibrate(sample, draw):
            calls.append(draw)
            # Stands in for within-draw midrank/fusion/threshold recomputation.
            return np.zeros_like(sample.values)

        statistic = lambda sample: sample.values.mean(axis=0)
        first = source_group_bootstrap(matrix, statistic, draws=12, seed=5, within_draw=recalibrate)
        second = source_group_bootstrap(matrix, statistic, draws=12, seed=5)
        self.assertEqual(calls, list(range(12)))
        np.testing.assert_array_equal(first.samples, 0)
        self.assertFalse(np.array_equal(first.samples, second.samples))
        np.testing.assert_array_equal(
            source_group_bootstrap(matrix, statistic, draws=12, seed=5).samples,
            second.samples,
        )
        self.assertEqual(first.requested_draws, 12)
        self.assertEqual(first.finite_counts, (12, 12))
        self.assertEqual(first.fully_finite_draws, 12)

    def test_simultaneous_interval_rejects_too_many_nonfinite_draws(self):
        samples = np.ones((100, 1))
        samples[:6, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "94/100 finite"):
            simultaneous_intervals(np.array([1.0]), samples)

    def test_group_rank_screen_accepts_independent_and_rejects_duplicate(self):
        rng = np.random.default_rng(17)
        meta = metadata(40, 25)
        independent = rng.binomial(1, np.array([0.35, 0.45, 0.55]), size=(len(meta), 3))
        matrix = binary_matrix(independent, meta, ("a", "b", "c"))
        conditioned = fixed_conditioning(matrix, np.tile(independent.mean(axis=0), (len(meta), 1)))
        accepted = group_diagnostics(matrix, ("a", "b", "c"), conditioned=conditioned, draws=80, seed=4)
        self.assertEqual(accepted.status, DependenceStatus.INDEPENDENCE_COMPATIBLE)
        self.assertGreaterEqual(accepted.effective_rank_ci[0], 2.1)

        duplicate_values = np.column_stack((independent[:, 0], independent[:, 1], independent[:, 0]))
        duplicate = binary_matrix(duplicate_values, meta, ("a", "b", "copy"))
        duplicate_conditioned = fixed_conditioning(
            duplicate, np.tile(duplicate_values.mean(axis=0), (len(meta), 1))
        )
        rejected = group_diagnostics(duplicate, ("a", "b", "copy"), conditioned=duplicate_conditioned, draws=40, seed=4)
        self.assertEqual(rejected.status, DependenceStatus.UNRESOLVED)
        self.assertLess(rejected.effective_rank, 2.1)

    def test_graph_and_cliques_are_deterministic_and_capped_at_six(self):
        names = tuple("abcdefg")
        statuses = {
            (left, right): DependenceStatus.INDEPENDENCE_COMPATIBLE
            for position, left in enumerate(names)
            for right in names[position + 1 :]
        }
        graph = build_compatibility_graph(names, statuses)
        cliques = enumerate_cliques(graph)
        self.assertEqual(cliques, enumerate_cliques(graph))
        self.assertTrue(cliques)
        self.assertEqual(max(map(len, cliques)), 6)
        self.assertNotIn(tuple(names), cliques)
        with self.assertRaisesRegex(ValueError, "max <= 6"):
            enumerate_cliques(graph, max_size=7)
        with self.assertRaisesRegex(RuntimeError, "limit"):
            enumerate_cliques(graph, max_results=3)

    def test_complete_clique_enumeration_is_all_or_explicitly_fails(self):
        names = tuple("abcd")
        graph = {
            name: frozenset(other for other in names if other != name)
            for name in names
        }
        expected = {
            tuple(names[index] for index in indexes)
            for size in (2, 3, 4)
            for indexes in itertools.combinations(range(len(names)), size)
        }
        first = dependence_backend._enumerate_cliques_complete(
            graph, min_size=2, max_size=6, max_results=len(expected),
        )
        second = dependence_backend._enumerate_cliques_complete(
            graph, min_size=2, max_size=6, max_results=len(expected),
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), expected)
        self.assertEqual(len(first), len(expected))

        with self.assertRaisesRegex(
            AtlasBackendError, "would exceed non-truncating limit",
        ):
            dependence_backend._enumerate_cliques_complete(
                graph, min_size=2, max_size=6, max_results=len(expected) - 1,
            )

    def test_clique_preflight_reports_streaming_memory_without_truncation(self):
        clique_count = 123
        draws = 10_000
        group_count = 3_483
        first = dependence_backend.clique_diagnostics_preflight(
            clique_count, draws, group_count,
        )
        second = dependence_backend.clique_diagnostics_preflight(
            clique_count, draws, group_count,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["non_truncating"])
        self.assertEqual(first["cliques"], clique_count)
        self.assertEqual(first["draws"], draws)
        self.assertEqual(first["groups"], group_count)
        self.assertEqual(
            first["legacy_materialized_sample_bytes"],
            clique_count * draws * 2 * np.dtype(np.float64).itemsize,
        )
        self.assertEqual(
            first["bootstrap_multiplicity_bytes"],
            draws * group_count * np.dtype(np.int32).itemsize,
        )
        self.assertLess(
            first["streamed_peak_sample_bytes"],
            first["legacy_materialized_sample_bytes"],
        )
        # The preflight is persisted in JSON manifests, so numpy scalars or
        # other implementation-only objects are not permitted in the payload.
        json.dumps(first, sort_keys=True)

    def test_streamed_clique_diagnostics_match_materialized_reference_exactly(self):
        rng = np.random.default_rng(211)
        meta = metadata(30, 10)
        values = rng.binomial(1, (0.42, 0.48, 0.54, 0.60), size=(len(meta), 4))
        matrix = binary_matrix(values, meta, ("a", "b", "c", "d"))
        conditioned = fixed_conditioning(
            matrix,
            np.tile(values.mean(axis=0), (len(meta), 1)),
        )
        cliques = (
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("a", "b", "c"),
            ("b", "c", "d"),
        )
        draws = 40
        seed = 613
        points, samples, supports = naive_materialized_clique_samples(
            matrix, cliques, conditioned, draws, seed,
        )
        intervals = simultaneous_intervals(points, samples, confidence=0.95)

        original_concatenate = np.concatenate

        def reject_wide_sample_materialization(arrays, *args, **kwargs):
            materialized = tuple(arrays)
            axis = kwargs.get("axis", args[0] if args else 0)
            if (
                axis == 1
                and len(materialized) == len(cliques)
                and all(np.asarray(value).shape == (draws, 2) for value in materialized)
            ):
                raise AssertionError("all clique samples were materialized in RAM")
            return original_concatenate(materialized, *args, **kwargs)

        with mock.patch.object(
            dependence_backend.np,
            "concatenate",
            side_effect=reject_wide_sample_materialization,
        ):
            observed = dependence_backend._common_group_diagnostics(
                matrix,
                cliques,
                conditioned=conditioned,
                draws=draws,
                seed=seed,
            )
        self.assertEqual([row.members for row in observed], list(cliques))
        self.assertEqual(len(observed), len(cliques))
        for index, row in enumerate(observed):
            np.testing.assert_allclose(
                (
                    row.max_conditional_residual_correlation,
                    row.effective_rank,
                ),
                points[2 * index:2 * index + 2],
                atol=1e-12,
                rtol=0,
            )
            np.testing.assert_allclose(
                (row.max_correlation_ci, row.effective_rank_ci),
                intervals[2 * index:2 * index + 2],
                atol=1e-12,
                rtol=0,
            )
            self.assertEqual((row.n, row.source_groups), supports[index])
            self.assertEqual(
                row.finite_draws,
                tuple(
                    int(np.isfinite(samples[:, 2 * index + offset]).sum())
                    for offset in range(2)
                ),
            )

    def test_full_population_gate_bootstrap_reranks_before_error_split(self):
        meta = metadata(20, 4)
        # One detector increases inside each source group; the second reverses
        # it.  Digit is inactive in every fourth answer.
        base = np.tile(np.arange(4, dtype=float), 20)
        scores = np.column_stack((base, 3.0 - base))
        detector_active = np.ones_like(scores, dtype=bool)
        detector_active[::4, 1] = False
        erroneous = np.tile([False, False, True, True], 20)
        fusion_sets = {"tail": ("tail",), "tail_digit": ("tail", "digit")}
        point = gate_error_matrices_from_scores(
            scores,
            ("tail", "digit"),
            fusion_sets,
            erroneous,
            meta,
            detector_active=detector_active,
        )
        self.assertEqual(point.false_open.target, ErrorTarget.GATE_FALSE_OPEN)
        self.assertEqual(point.false_close.target, ErrorTarget.GATE_FALSE_CLOSE)
        self.assertEqual(point.false_open.shape[0], 40)
        self.assertEqual(point.false_close.shape[0], 40)
        self.assertTrue(point.active[0, 1])  # tail keeps the fusion active

        statistic = lambda errors: np.concatenate(
            (errors.false_open.values.mean(axis=0), errors.false_close.values.mean(axis=0))
        )
        first = full_population_gate_bootstrap(
            scores,
            ("tail", "digit"),
            fusion_sets,
            erroneous,
            meta,
            statistic,
            detector_active=detector_active,
            draws=15,
            seed=44,
        )
        second = full_population_gate_bootstrap(
            scores,
            ("tail", "digit"),
            fusion_sets,
            erroneous,
            meta,
            statistic,
            detector_active=detector_active,
            draws=15,
            seed=44,
        )
        np.testing.assert_array_equal(first.statistics.samples, second.statistics.samples)
        self.assertEqual(first.statistics.finite_counts, (15, 15, 15, 15))


def _write_fixture_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf8")


def _build_atlas_backend_fixture(root):
    """Build a small, fully frozen on-disk Atlas contract.

    The fixture deliberately includes clean and erroneous ProcessBench rows in
    every source fold, plus PRMBench rows.  It contains only the signal subset
    required by the high-level backend, while preserving the production packed
    extraction layout.
    """
    root = Path(root)
    bundle = root / "bundle"
    extract = root / "extract"
    predictors = root / "predictors"
    reconcile = root / "reconcile"
    registry_root = root / "registry"
    contract = root / "contract"
    for path in (bundle, extract / "answers", predictors, reconcile, registry_root, contract):
        path.mkdir(parents=True, exist_ok=True)

    tokens_per_answer = 6
    steps_per_answer = 3
    answers = []
    targets = []
    labels = []
    records = []
    global_spans = []
    primitive_rows = []
    gate_raw = []
    # Four PB answers and two PRMB answers in every one of five folds.
    for fold in range(5):
        for local in range(6):
            index = len(answers)
            is_pb = local < 4
            cell = "pb_fixture" if is_pb else "prmbench_fixture"
            target = -1 if is_pb and local in (0, 2) else (
                (fold + local) % steps_per_answer if is_pb else -1
            )
            token_offset = index * tokens_per_answer
            step_start = index * steps_per_answer
            step_stop = step_start + steps_per_answer
            uid = f"fixture-{fold}-{local}"
            answers.append({
                "uid": uid,
                "cell": cell,
                "group_id": f"group-{fold}-{local}",
                "tokens": tokens_per_answer,
                "offset": token_offset,
                "step_start": step_start,
                "step_stop": step_stop,
                "fold": fold,
            })
            records.append({"uid": uid, "row_id": index})
            targets.append(target)
            local_labels = np.ones(steps_per_answer, dtype=np.int8)
            if is_pb and target >= 0:
                local_labels[target] = 0
            elif not is_pb:
                local_labels[:] = (1, 0, 1)
            labels.extend(local_labels.tolist())
            local_spans = np.asarray(((0, 2), (2, 4), (4, 6)), dtype=np.int64)
            global_spans.extend((local_spans + token_offset).tolist())

            step_risk = np.asarray((0.15, 0.25, 0.35), dtype=np.float64)
            if is_pb and target >= 0:
                step_risk[target] += 3.0
            elif not is_pb:
                step_risk[1] += 2.0
            token_risk = np.repeat(step_risk, 2) + np.linspace(0.0, 0.025, tokens_per_answer)
            primitive_rows.extend(np.column_stack([
                token_risk * (1.0 + 0.07 * column) + 0.013 * column
                for column in range(4)
            ]).tolist())
            gate_raw.append(float((target >= 0) * 3.0 + 0.01 * index))

            arrays = {}
            signal_names = (
                "q15.H0lim",
                "q15.VE0",
                "q15.VE0.75",
                "q15.VE1",
                "q15.H1_native",
                "q15.Hinf",
                "q15.H0lim.prefix_mean_innovation",
                "direct_probability.rank_1_risk",
                "direct_probability.residual_tail_mass",
                "step395.tail50",
                "fixture.token_a",
                "fixture.token_b",
            )
            for signal_index, signal in enumerate(signal_names):
                if signal == "fixture.token_a":
                    values = np.tile((0.0, 2.0), steps_per_answer).astype(np.float32)
                elif signal == "fixture.token_b":
                    values = np.tile((2.0, 0.0), steps_per_answer).astype(np.float32)
                else:
                    values = (
                        token_risk * (1.0 + 0.025 * signal_index)
                        + 0.002 * signal_index * np.arange(tokens_per_answer)
                    ).astype(np.float32)
                active = np.ones(tokens_per_answer, dtype=bool)
                arrays[f"token__{signal}__values"] = values
                arrays[f"token__{signal}__active"] = active
                if signal.startswith("fixture.token_"):
                    # Same singleton readout/decoder result, but different raw
                    # within-step trajectories.  They are duplicates only
                    # after readout, not at the token-fusion insertion point.
                    special = step_risk ** 2 + np.asarray((0.031, 0.017, 0.043))
                    score = np.column_stack([
                        special * (1.0 + 0.004 * readout_index)
                        for readout_index in range(len(READOUT_NAMES))
                    ]).astype(np.float32)
                else:
                    score = np.column_stack([
                        step_risk * (1.0 + 0.004 * readout_index)
                        + 0.0001 * signal_index * np.arange(steps_per_answer)
                        for readout_index in range(len(READOUT_NAMES))
                    ]).astype(np.float32)
                available = np.ones_like(score, dtype=bool)
                decisions = np.empty((len(READOUT_NAMES), 3), dtype=np.int32)
                for readout_index in range(len(READOUT_NAMES)):
                    peak = int(np.argmax(score[:, readout_index]))
                    decisions[readout_index] = peak
                arrays[f"step__{signal}__readouts"] = score
                arrays[f"step__{signal}__readout_active"] = available
                arrays[f"decision__{signal}__readouts"] = decisions
                arrays[f"decision__{signal}__step_top5"] = np.asarray(
                    int(np.argmax(score[:, READOUT_NAMES.index("top5")])), dtype=np.int32,
                )

            disagreement = np.zeros(tokens_per_answer, dtype=np.float32)
            if is_pb and target >= 0:
                disagreement[2 * target:2 * target + 2] = 1.0
            elif not is_pb:
                disagreement[2:4] = 1.0
            opportunity = np.ones(tokens_per_answer, dtype=np.float32)
            arrays["token__digit.disagreement__values"] = disagreement
            arrays["token__digit.disagreement__active"] = np.ones(tokens_per_answer, dtype=bool)
            arrays["token__digit.opportunity__values"] = opportunity
            arrays["token__digit.opportunity__active"] = np.ones(tokens_per_answer, dtype=bool)
            digit_steps = disagreement.reshape(steps_per_answer, 2).mean(axis=1)
            digit_score = np.column_stack([
                digit_steps + 0.0002 * readout_index * np.arange(steps_per_answer)
                for readout_index in range(len(READOUT_NAMES))
            ]).astype(np.float32)
            digit_decisions = np.empty((len(READOUT_NAMES), 3), dtype=np.int32)
            for readout_index in range(len(READOUT_NAMES)):
                digit_decisions[readout_index] = int(np.argmax(digit_score[:, readout_index]))
            arrays["step__digit.disagreement__readouts"] = digit_score
            arrays["step__digit.disagreement__readout_active"] = np.ones_like(digit_score, dtype=bool)
            arrays["decision__digit.disagreement__readouts"] = digit_decisions
            arrays["decision__digit.disagreement__step_top5"] = np.asarray(
                int(np.argmax(digit_score[:, READOUT_NAMES.index("top5")])), dtype=np.int32,
            )

            digit_count = disagreement.reshape(steps_per_answer, 2).sum(axis=1)
            digit_rate = digit_count / 2.0
            digit_presence = np.ones(steps_per_answer, dtype=np.float32)
            for name, values in (
                ("digit.count", digit_count),
                ("digit.rate", digit_rate),
                ("digit.presence", digit_presence),
            ):
                arrays[f"step__{name}__native__values"] = np.asarray(values, dtype=np.float32)
                arrays[f"step__{name}__native__active"] = np.ones(steps_per_answer, dtype=bool)
            arrays["decision__earlier_ve_peak"] = np.asarray(int(np.argmax(step_risk)), dtype=np.int32)
            arrays["answer__tail15.answer_prominence"] = np.asarray(
                float((target >= 0) * 2.0 + 0.005 * index), dtype=np.float32,
            )
            np.savez_compressed(extract / "answers" / f"{index:05d}.npz", **arrays)

    primitive = np.asarray(primitive_rows, dtype=np.float32)
    np.save(bundle / "primitive_levels.npy", primitive)
    np.save(bundle / "step_spans.npy", np.asarray(global_spans, dtype=np.int64))
    _write_fixture_json(bundle / "METADATA.json", answers)

    offsets = np.arange(0, (len(answers) + 1) * steps_per_answer, steps_per_answer, dtype=np.int64)
    joined_json = contract / "JOINED.json"
    joined_npz = contract / "JOINED.npz"
    _write_fixture_json(joined_json, {"records": records})
    np.savez_compressed(
        joined_npz,
        offsets=offsets,
        labels=np.asarray(labels, dtype=np.int8),
        target=np.asarray(targets, dtype=np.int32),
    )

    registry_signals = []
    for signal in (
        "q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1", "q15.H1_native",
        "q15.Hinf", "q15.H0lim.prefix_mean_innovation",
        "direct_probability.rank_1_risk", "direct_probability.residual_tail_mass",
        "step395.tail50", "fixture.token_a", "fixture.token_b",
        "digit.disagreement", "digit.count", "digit.rate",
        "digit.presence", "tail15.answer_prominence",
    ):
        registry_signals.append({
            "name": signal,
            "status": "ELIGIBLE",
            "provenance_family": (
                "fixture_token_pair" if signal.startswith("fixture.token_") else "fixture_family"
            ),
            "access_scope": "answer_only",
        })
    registry_path = registry_root / "REGISTRY.json"
    _write_fixture_json(registry_path, {"signals": registry_signals, "aliases": {}})
    _write_fixture_json(extract / "MANIFEST.json", {
        "schema": "fixture/atomic-extraction-v2",
        "answers": len(answers),
        "label_free": True,
        "bundle_root": str(bundle),
        "joined_json": str(joined_json),
        "joined_npz": str(joined_npz),
        "registry_path": str(registry_path),
        "readout_names": list(READOUT_NAMES),
    })
    _write_fixture_json(extract / "STATUS.json", {"status": "COMPLETE"})
    np.save(extract / "completion.npy", np.ones(len(answers), dtype=bool))

    fixed = np.empty((len(primitive), 4, 4), dtype=np.float32)
    learned = np.empty((len(primitive), 2, 4), dtype=np.float32)
    for predictor in range(4):
        fixed[:, predictor] = primitive * (0.12 + 0.11 * predictor) + 0.01 * predictor
    for predictor in range(2):
        learned[:, predictor] = primitive * (0.61 + 0.11 * predictor) - 0.02 * predictor
    (predictors / "fixed").mkdir()
    (predictors / "learned_oof").mkdir()
    (predictors / "learned_inner").mkdir()
    np.save(predictors / "fixed/backgrounds.npy", fixed)
    np.save(predictors / "learned_oof/backgrounds.npy", learned)
    np.save(predictors / "learned_oof/active.npy", np.ones(len(primitive), dtype=bool))
    inner = np.zeros((len(primitive), 5, 2, 4), dtype=np.float32)
    inner_active = np.zeros((len(primitive), 5), dtype=bool)
    for row in answers:
        start = int(row["offset"])
        stop = start + int(row["tokens"])
        source_fold = int(row["fold"])
        for validation_fold in range(5):
            if validation_fold == source_fold:
                continue
            inner[start:stop, validation_fold] = learned[start:stop]
            inner_active[start + 1:stop, validation_fold] = True
    np.save(predictors / "learned_inner/backgrounds.npy", inner)
    np.save(predictors / "learned_inner/active.npy", inner_active)
    _write_fixture_json(predictors / "learned_inner/MANIFEST.json", {
        "labels_used": False,
        "shape": list(inner.shape),
        "axes": {
            "0": "token",
            "1": "inner_fold_f",
            "2": ["ridge", "tcn"],
            "3": ["H0lim", "VE0", "VE0.75", "VE1"],
        },
    })
    _write_fixture_json(predictors / "FREEZE.json", {
        "labels_used": False,
        "targets": ["H0lim", "VE0", "VE0.75", "VE1"],
    })
    _write_fixture_json(reconcile / "LEDGER.json", {
        "historical_unique_score_arrays": 178,
        "historical_unique_peak_vectors": 173,
        "available_archives": 1,
        "report_only_archives": 1,
        "scored_columns": 2,
        "available_score_columns": 2,
        "columns": [
            {
                "name": "historical::fixture::primary",
                "archive": "fixture", "artifact_status": "AVAILABLE_FROZEN",
                "score_sha256": "score-a", "peak_sha256": "peak-a",
                "canonical_score": "historical::fixture::primary",
                "exact_score_duplicate": False, "peak_verified": True,
                "included": True, "control_tagged": False,
            },
            {
                "name": "historical::fixture::alias",
                "archive": "fixture", "artifact_status": "AVAILABLE_FROZEN",
                "score_sha256": "score-a", "peak_sha256": "peak-a",
                "canonical_score": "historical::fixture::primary",
                "exact_score_duplicate": True, "peak_verified": True,
                "included": False, "control_tagged": True,
            },
        ],
    })
    (root / "baseline_replay").mkdir()
    np.savez_compressed(root / "baseline_replay/SCORES_FROZEN.npz", gate_raw=np.asarray(gate_raw))
    return {
        "extract": extract,
        "predictors": predictors,
        "reconcile": reconcile,
        "contract": contract,
        "pb_count": 20,
        "pb_error_count": 10,
    }


class AtlasFilesystemBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.fixture = _build_atlas_backend_fixture(cls._temporary.name)
        cls.dependence = Path(cls._temporary.name) / "dependence"
        cls.fusion = Path(cls._temporary.name) / "fusion"
        cls.dependence_summary = run_atlas_dependence(
            extract_root=cls.fixture["extract"],
            predictor_root=cls.fixture["predictors"],
            reconciliation_root=cls.fixture["reconcile"],
            contract_root=cls.fixture["contract"],
            output_root=cls.dependence,
            draws=3,
            seed=27,
        )
        cls.fusion_summary = run_nested_fusion_search(
            dependence_root=cls.dependence,
            extraction_root=cls.fixture["extract"],
            output_root=cls.fusion,
            folds=5,
            family_cap=2,
            maximum_size=6,
            roster_stability=4,
            heads=("singleton", "equal_rank", "family_equal", "nonnegative_shrunk_simplex", "iu"),
            iu_eligible_only=True,
            seed=27,
        )

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_dependence_backend_emits_typed_nonempty_artifacts(self):
        self.assertEqual(self.dependence_summary["status"], "COMPLETE")
        self.assertEqual(self.dependence_summary["unsupported_points"], {})
        for name in (
            "CANDIDATES.json", "CONSOLIDATED.npz", "EVALUATION.npz",
            "ERROR_LEDGER.json", "PAIRWISE.json", "COMPATIBILITY_GRAPH.json", "GROUPS.json",
        ):
            self.assertTrue((self.dependence / name).is_file(), name)
        candidates = json.loads((self.dependence / "CANDIDATES.json").read_text())
        self.assertTrue(candidates["eligible"])
        self.assertFalse(candidates["held_fold_labels_used_for_its_roster"])
        historical = candidates["historical_report_only"]
        self.assertEqual(len(historical), 2)
        self.assertTrue(all(row["status"] == "REPORT_ONLY" for row in historical))
        self.assertTrue(all(not row["promotion_eligible"] for row in historical))
        self.assertTrue(historical[1]["exact_score_duplicate"])
        screened_names = {row["name"] for row in candidates["screening"]}
        gate_names = {name for name in screened_names if name.startswith("answer_gate::")}
        self.assertEqual(len(gate_names), 11 * 12 + 3 + 1)
        self.assertTrue({
            "answer_gate::digit_count",
            "answer_gate::digit_rate",
            "answer_gate::digit_presence",
            "answer_gate::tail15.answer_prominence",
        }.issubset(gate_names))
        eligible_token_pair = {
            row["signal"] for row in candidates["eligible"]
            if row["point"] == "token_pre_readout" and row["family"] == "fixture_token_pair"
        }
        self.assertEqual(eligible_token_pair, {"fixture.token_a", "fixture.token_b"})
        ledger = json.loads((self.dependence / "ERROR_LEDGER.json").read_text())
        matrices = {row["name"]: row for row in ledger["matrices"]}
        self.assertTrue(any(name.startswith("background:") for name in matrices))
        self.assertIn("background:pb_raw_locator_miss", matrices)
        self.assertIn("background:prmb_pairwise_misorder", matrices)
        self.assertIn("token_pre_readout:pb_raw_locator_miss", matrices)
        self.assertIn("token_pre_readout:prmb_pairwise_misorder", matrices)
        self.assertIn("step_post_readout:pb_raw_locator_miss", matrices)
        self.assertIn("step_post_readout:prmb_pairwise_misorder", matrices)
        self.assertIn("answer_gate:gate_false_open", matrices)
        self.assertIn("answer_gate:gate_false_close", matrices)
        self.assertIn("answer_gate:pb_final", matrices)
        self.assertEqual(
            matrices["step_post_readout:pb_raw_locator_miss"]["observations"],
            self.fixture["pb_error_count"],
        )
        self.assertEqual(
            matrices["answer_gate:pb_final"]["observations"],
            self.fixture["pb_count"],
        )
        pairwise = json.loads((self.dependence / "PAIRWISE.json").read_text())
        graph = json.loads((self.dependence / "COMPATIBILITY_GRAPH.json").read_text())
        self.assertTrue(pairwise["rows"])
        self.assertTrue(pairwise["outer_training_rows"])
        self.assertTrue(graph["graphs"])
        self.assertEqual(set(graph["outer_training_graphs"]), {"0", "1", "2", "3", "4"})
        expected_points = {
            "background", "token_pre_readout", "step_post_readout", "decoder", "answer_gate",
        }
        for fold_graphs in graph["outer_training_graphs"].values():
            self.assertEqual(set(fold_graphs), expected_points)

    def test_nested_search_is_five_fold_and_keeps_canonical_pb_denominator(self):
        points = {
            "background", "token_pre_readout", "step_post_readout", "decoder", "answer_gate",
        }
        # All five independent insertion-point finalists are complete, while
        # the cross-point factorial correctly remains fail-closed until a
        # typed pipeline-composition rule exists.
        self.assertEqual(self.fusion_summary["status"], "PARTIAL_FAIL_CLOSED")
        self.assertEqual(set(self.fusion_summary["finalists"]), points)
        self.assertEqual(self.fusion_summary["factorial_results"], [])
        self.assertEqual(
            self.fusion_summary["factorial_status"],
            "UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED",
        )
        for name in (
            "ALL_GROUPS.json", "NESTED_SELECTION.json", "PARETO.json",
            "LEAVE_ONE_SIGNAL_OUT.json", "FINALISTS.json", "UNCERTAINTY.json",
            "DEPENDENT_COMPLEMENTARY.json", "COMPOSITION_CONTRACT.json",
            "FUSION_ORDER_SENSITIVITY.json",
        ):
            self.assertTrue((self.fusion / name).is_file(), name)
        rank_cache = self.fusion / "rank_cache"
        self.assertTrue((rank_cache / "TOKEN_RANKS.npy").is_file())
        self.assertTrue((rank_cache / "STEP_RANKS.npy").is_file())
        rank_manifest = json.loads((rank_cache / "MANIFEST.json").read_text())
        self.assertTrue(rank_manifest["label_free"])
        self.assertEqual(rank_manifest["dtype"], "float64")
        groups = json.loads((self.fusion / "ALL_GROUPS.json").read_text())
        nested = json.loads((self.fusion / "NESTED_SELECTION.json").read_text())
        pareto = json.loads((self.fusion / "PARETO.json").read_text())
        loso = json.loads((self.fusion / "LEAVE_ONE_SIGNAL_OUT.json").read_text())
        finalist_document = json.loads((self.fusion / "FINALISTS.json").read_text())
        uncertainty = json.loads((self.fusion / "UNCERTAINTY.json").read_text())
        complementary = json.loads(
            (self.fusion / "DEPENDENT_COMPLEMENTARY.json").read_text()
        )
        order_sensitivity = json.loads(
            (self.fusion / "FUSION_ORDER_SENSITIVITY.json").read_text()
        )
        composition = json.loads(
            (self.fusion / "COMPOSITION_CONTRACT.json").read_text()
        )
        finalists = finalist_document["finalists"]
        self.assertTrue(groups["rows"])
        self.assertTrue(groups["group_level_compatibility_required"])
        self.assertEqual(len(nested["folds"]), 5)
        self.assertTrue(nested["labels_never_passed_to_weight_fit"])
        self.assertTrue(pareto["rows"])
        self.assertTrue(nested["background_pair_excluded_inner_consumed"])
        self.assertEqual(
            loso["schema"],
            "fusion-independence-atlas-v1/leave-one-signal-out-v1",
        )
        self.assertIsInstance(loso["rows"], list)
        models = {row["name"]: row for row in groups["rows"]}
        outer_graphs = json.loads(
            (self.dependence / "COMPATIBILITY_GRAPH.json").read_text()
        )["outer_training_graphs"]
        for fold in nested["folds"]:
            self.assertFalse(fold["held_labels_used_for_weights"])
            self.assertEqual(set(fold["selections"]), points)
            outer = int(fold["outer"])
            for point, selection in fold["selections"].items():
                selected_model = models[selection["selected"]]
                self.assertIn(outer, selected_model["eligible_outer_folds"])
                self.assertTrue(
                    set(selection["members"]).issubset(
                        outer_graphs[str(outer)][point]["nodes"]
                    )
                )
                # All four held-fold PB rows remain in the denominator: two
                # clean and two erroneous answers.  Inactive rows may be wrong,
                # but may not silently disappear.
                cell_counts = selection["outer_result"]["pb_cells"].values()
                self.assertEqual(
                    sum(row["clean_total"] + row["error_total"] for row in cell_counts),
                    4,
                )
                self.assertEqual(sum(row["clean_total"] for row in cell_counts), 2)
                self.assertEqual(sum(row["error_total"] for row in cell_counts), 2)
        self.assertEqual(set(finalists), points)
        self.assertEqual(set(uncertainty["points"]), points)
        self.assertEqual(uncertainty["draws"], 3)
        self.assertTrue(uncertainty["paired_source_group_draws"])
        self.assertTrue(uncertainty["simultaneous_across_insertion_points_and_metrics"])
        self.assertTrue(uncertainty["gate_reranked_within_every_draw"])
        self.assertIsNone(uncertainty["interval_error"])
        self.assertTrue(complementary["rows"])
        self.assertTrue(all(row["diagnostic_only"] for row in complementary["rows"]))
        self.assertIn(order_sensitivity["status"], {"EVALUATED", "UNRESOLVED"})
        self.assertFalse(composition["derivable_uniquely_from_current_incumbent"])
        self.assertEqual(
            {row["id"] for row in composition["smallest_material_user_decisions"]},
            {"locator_composition_topology", "serial_replacement_port"},
        )
        self.assertEqual(
            self.fusion_summary["required_user_decisions"],
            composition["smallest_material_user_decisions"],
        )
        self.assertFalse(finalist_document["complete_five_point_factorial_ready"])
        for point in points:
            self.assertIn(point, finalists)
            self.assertEqual(finalists[point]["pb_total"], self.fixture["pb_count"])
        # The fixture's locator is exact on all erroneous PB rows and its gate
        # closes most clean rows.  More than the ten error-only successes proves
        # final PB uses the canonical clean=-1 decision, not error-only recall.
        self.assertGreater(finalists["step_post_readout"]["pb_correct"], self.fixture["pb_error_count"])

    def test_nested_search_fails_closed_without_each_per_outer_compatibility_artifact(self):
        for filename, key in (
            ("COMPATIBILITY_GRAPH.json", "outer_training_graphs"),
            ("GROUPS.json", "outer_training_rows"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                dependence = Path(root) / "dependence"
                shutil.copytree(self.dependence, dependence)
                artifact = dependence / filename
                payload = json.loads(artifact.read_text())
                payload.pop(key)
                artifact.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    AtlasBackendError,
                    "per-outer training-only compatibility graphs and group screens",
                ):
                    run_nested_fusion_search(
                        dependence_root=dependence,
                        extraction_root=self.fixture["extract"],
                        output_root=Path(root) / "fusion",
                        folds=5,
                        family_cap=2,
                        maximum_size=6,
                        roster_stability=4,
                        heads=(
                            "singleton", "equal_rank", "family_equal",
                            "nonnegative_shrunk_simplex", "iu",
                        ),
                        iu_eligible_only=True,
                        seed=27,
                    )

    def test_fusion_weight_fit_api_is_structurally_label_free(self):
        forbidden = {"label", "labels", "target", "targets", "correctness", "annotations", "y"}
        for function in (fit_label_free_fusion_weights, apply_label_free_fusion_weights):
            self.assertFalse(forbidden & set(inspect.signature(function).parameters))
        scores = np.asarray(((0.1, 0.8), (0.4, 0.2), (0.9, 0.5), (0.3, 0.7)))
        active = np.ones_like(scores, dtype=bool)
        labels = np.asarray((0, 1, 0, 1))
        first, _ = fit_label_free_fusion_weights(scores, active, ("a", "b"), "equal_rank")
        labels[:] = 1 - labels
        second, _ = fit_label_free_fusion_weights(scores, active, ("a", "b"), "equal_rank")
        np.testing.assert_array_equal(first, second)
        # Even the internal typed dispatcher receives only row geometry, not
        # the evaluation mapping that also contains benchmark annotations.
        nested_parameters = set(
            inspect.signature(dependence_backend._fit_nested_model).parameters
        )
        self.assertNotIn("evaluation", nested_parameters)
        self.assertFalse(forbidden & nested_parameters)

    def test_uncertainty_does_not_redefine_pb_when_a_draw_omits_a_cell(self):
        evaluation = {
            "target": np.asarray((-1, 0, -1, 0), dtype=np.int64),
            "cells": np.asarray(("pb_a", "pb_a", "pb_b", "pb_b")),
        }
        ledger = {
            "prediction": np.asarray((-1, 0, -1, 0), dtype=np.int64),
            "valid": np.ones(4, dtype=bool),
            "within": np.full(4, np.nan),
        }
        pb, _ = dependence_backend._ledger_metrics(
            ledger, np.asarray((0, 1)), evaluation,
            required_pb_cells=("pb_a", "pb_b"),
        )
        self.assertTrue(np.isnan(pb))

    def test_shrunk_simplex_fit_uses_the_same_block_local_ranks_as_application(self):
        scores = np.asarray((
            (0.0, 3.0, 1.0), (1.0, 1.0, 2.0), (2.0, 2.0, 0.0),
            (10.0, 13.0, 11.0), (11.0, 11.0, 12.0), (12.0, 12.0, 10.0),
        ))
        blocks = np.asarray((0, 0, 0, 1, 1, 1))
        shifted = scores.copy()
        shifted[blocks == 1] += np.asarray((1000.0, -1000.0, 500.0))
        active = np.ones_like(scores, dtype=bool)
        first, _ = fit_label_free_fusion_weights(
            scores, active, ("a", "b", "c"), "nonnegative_shrunk_simplex",
            block_ids=blocks,
        )
        second, _ = fit_label_free_fusion_weights(
            shifted, active, ("a", "b", "c"), "nonnegative_shrunk_simplex",
            block_ids=blocks,
        )
        # Per-block ordering is unchanged, so a block-ranked covariance fit is
        # exactly invariant to these cross-block offsets.
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)

    def test_cross_block_level_shift_does_not_make_a_constant_view_live(self):
        scores = np.asarray(((1.0,), (1.0,), (9.0,), (9.0,)))
        active = np.ones_like(scores, dtype=bool)
        weights, diagnostics = fit_label_free_fusion_weights(
            scores, active, ("constant",), "singleton",
            block_ids=np.asarray((0, 0, 1, 1)),
        )
        np.testing.assert_array_equal(weights, 0.0)
        self.assertEqual(diagnostics["status"], "NO_LIVE_MEMBERS")


if __name__ == "__main__":
    unittest.main()
