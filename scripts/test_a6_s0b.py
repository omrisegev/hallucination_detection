from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from spectral_utils.a6_s0b import (
    CONTINUOUS_COLUMNS,
    GroupMatchingRecord,
    OofBundle,
    RIDGES,
    ShortcutRow,
    binary_auc,
    bootstrap_group_multiplicities,
    brute_force_assignment,
    build_shortcut_rows,
    canonical_partition_memberships,
    control2_derangement,
    control3_matching,
    distance_caliper,
    fit_shortcut_logistic,
    freeze_vocabulary,
    group_matching_records,
    hungarian_exact,
    logistic_objective_gradient,
    marginal_prevalence_audit,
    materialize_control_schedule,
    matching_vectors,
    pythia_prompt_mean_nll,
    shortcut_gate_bootstrap,
    weighted_binary_auc,
)


ROOT = Path(__file__).resolve().parents[1]
S0A = ROOT / "results" / "automatic_group_free_phase_a6_s0a_v1"


def _load_frozen_quartets() -> list[dict]:
    payloads = []
    for path in sorted((S0A / "checkpoints" / "quartet").glob("*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(wrapper["payload"])
    return payloads


def _fake_nll_map(payloads: list[dict]) -> dict[str, float]:
    result = {}
    for payload in payloads:
        for evidence in payload["contextual_evidence"].values():
            for row in evidence:
                prompt_hash = row["prompt_sha256"]
                result[prompt_hash] = 1.0 + int(prompt_hash[:8], 16) / 2**32
    return result


def _synthetic_row(group: int, fold: int, target: int, value: float) -> ShortcutRow:
    return ShortcutRow(
        row_id=f"g{group}:{target}", population_id="qwen-source",
        group_id=f"g{group}", outer_fold=fold, scorer_id="qwen3-4b",
        rendering_family="canonical", prompt_world="A" if target == 0 else "B",
        response_world="A", prompt_sha256=f"p{group}:{target}",
        response_sha256="rA", target=target,
        continuous=tuple(value + index / 100 for index in range(len(CONTINUOUS_COLUMNS))),
        categorical=(
            "arithmetic", "value_leaf", "short", "canonical", "record_value",
            f"bank{group % 5}", f"template{group}", f"donor{group}",
        ),
    )


class TestA6S0b(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payloads = _load_frozen_quartets()
        if len(cls.payloads) != 1_800:
            raise unittest.SkipTest("canonical S0a quartet artifacts are unavailable")
        cls.rows = build_shortcut_rows(cls.payloads, _fake_nll_map(cls.payloads))
        cls.null_strata = json.loads((S0A / "NULL_STRATA.json").read_text(encoding="utf-8"))
        cls.inner_folds = json.loads((S0A / "INNER_FOLDS.json").read_text(encoding="utf-8"))

    def test_frozen_crossed_table_and_marginals(self) -> None:
        qwen = [row for row in self.rows if row.population_id == "qwen-source"]
        llama = [row for row in self.rows if row.population_id == "llama-audit"]
        self.assertEqual(len(qwen), 900 * 2 * 16)
        self.assertEqual(len(llama), 900 * 16)
        self.assertTrue(marginal_prevalence_audit(qwen)["pass"])
        self.assertTrue(marginal_prevalence_audit(llama)["pass"])
        for group_id in {row.group_id for row in qwen}:
            group_rows = [row for row in qwen if row.group_id == group_id]
            self.assertEqual(sum(row.target for row in group_rows), 16)

    def test_crossed_row_equations_use_own_response(self) -> None:
        group_id = self.payloads[0]["group"]["group_id"]
        rows = [
            row for row in self.rows
            if row.group_id == group_id and row.scorer_id == "qwen3-4b"
            and row.rendering_family == "canonical"
        ]
        self.assertEqual(len(rows), 4)
        by_world = {(row.prompt_world, row.response_world): row for row in rows}
        self.assertEqual(by_world[("A", "A")].target, 0)
        self.assertEqual(by_world[("A", "B")].target, 1)
        self.assertEqual(
            by_world[("A", "A")].prompt_sha256,
            by_world[("A", "B")].prompt_sha256,
        )
        values_a = by_world[("A", "A")].continuous_dict()
        values_b = by_world[("A", "B")].continuous_dict()
        self.assertEqual(values_a["prompt_levenshtein_distance"], values_b["prompt_levenshtein_distance"])
        group = self.payloads[0]["group"]
        self.assertEqual(values_a["response_char_length"], len(group["response_text_a"]))
        self.assertEqual(values_b["response_char_length"], len(group["response_text_b"]))
        self.assertEqual(by_world[("A", "A")].response_sha256, group["response_sha256_a"])
        self.assertEqual(by_world[("A", "B")].response_sha256, group["response_sha256_b"])

    def test_matching_vector_order_and_dimension(self) -> None:
        qwen = [row for row in self.rows if row.population_id == "qwen-source"]
        vocabulary = freeze_vocabulary(qwen)
        group_ids, vectors = matching_vectors(qwen, vocabulary)
        self.assertEqual(len(group_ids), 900)
        self.assertEqual(vectors.shape[0], 900)
        self.assertGreater(vectors.shape[1], 32 * len(CONTINUOUS_COLUMNS))
        self.assertTrue(np.isfinite(vectors).all())
        shuffled = list(reversed(qwen))
        shuffled_ids, shuffled_vectors = matching_vectors(shuffled, vocabulary)
        self.assertEqual(group_ids, shuffled_ids)
        np.testing.assert_array_equal(vectors, shuffled_vectors)

    def test_logistic_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(81415)
        design = rng.normal(size=(40, 5))
        labels = np.asarray([-1.0, 1.0] * 20)
        parameters = rng.normal(scale=0.2, size=6)
        value, gradient = logistic_objective_gradient(parameters, design, labels, 0.1)
        self.assertTrue(math.isfinite(value))
        epsilon = 1e-6
        numeric = []
        for index in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            value_plus = logistic_objective_gradient(plus, design, labels, 0.1)[0]
            value_minus = logistic_objective_gradient(minus, design, labels, 0.1)[0]
            numeric.append((value_plus - value_minus) / (2 * epsilon))
        np.testing.assert_allclose(gradient, numeric, rtol=2e-6, atol=2e-7)
        sparse_value, sparse_gradient = logistic_objective_gradient(
            parameters, sparse.csr_matrix(design), labels, 0.1,
        )
        self.assertEqual(value, sparse_value)
        np.testing.assert_allclose(gradient, sparse_gradient, rtol=2e-15, atol=2e-16)

    def test_logistic_fit_is_zero_initialized_and_usable(self) -> None:
        x = np.asarray([[0.0], [1.0], [0.2], [0.8]] * 20)
        y = [0, 1, 0, 1] * 20
        for ridge in RIDGES:
            fit = fit_shortcut_logistic(x, y, ridge)
            self.assertLessEqual(fit.gradient_inf, 1e-8)
            self.assertTrue(math.isfinite(fit.objective))

    def test_auc_ties(self) -> None:
        self.assertEqual(binary_auc([0, 1], [0.0, 1.0]), 1.0)
        self.assertEqual(binary_auc([0, 1], [1.0, 0.0]), 0.0)
        self.assertEqual(binary_auc([0, 1], [1.0, 1.0]), 0.5)
        self.assertEqual(weighted_binary_auc([0, 1], [0.0, 1.0], [3, 7]), 1.0)
        self.assertEqual(weighted_binary_auc([0, 1], [1.0, 1.0], [3, 7]), 0.5)

    def test_group_bootstrap_and_max_ridge_statistic(self) -> None:
        rows = []
        for fold in range(5):
            for group_offset in range(4):
                group = 4 * fold + group_offset
                rows.extend((
                    _synthetic_row(group, fold, 0, float(group)),
                    _synthetic_row(group, fold, 1, float(group) + 0.25),
                ))
        group_ids, multiplicities, seed = bootstrap_group_multiplicities(
            rows, "overall", n_draws=25,
        )
        self.assertEqual(len(group_ids), 20)
        self.assertEqual(multiplicities.shape, (25, 20))
        self.assertTrue(np.all(np.sum(multiplicities, axis=1) == 20))
        self.assertEqual(
            (group_ids, multiplicities.tolist(), seed),
            (lambda result: (result[0], result[1].tolist(), result[2]))(
                bootstrap_group_multiplicities(rows, "overall", n_draws=25)
            ),
        )
        scores = tuple(float(row.target) for row in rows)
        bundles = tuple(OofBundle(
            population_id="qwen-source", ridge=ridge, scores=scores,
            fold_auc=(1.0,) * 5, fits=(),
        ) for ridge in RIDGES)
        result = shortcut_gate_bootstrap(rows, bundles, "overall", n_draws=25)
        self.assertEqual(result.observed_max_macro_auc, 1.0)
        self.assertEqual(result.upper_95, 1.0)
        self.assertEqual(result.selected_ridge, 10.0)
        self.assertFalse(result.gate_pass)

    def test_control2_is_deterministic_and_fixed_point_free(self) -> None:
        group_ids = tuple(f"g{index}" for index in range(12))
        seed = (12345).to_bytes(8, "big")
        first = control2_derangement(group_ids, seed, "outer:0:held", "stratum")
        second = control2_derangement(group_ids, seed, "outer:0:held", "stratum")
        self.assertEqual(first, second)
        self.assertTrue(all(left != right for left, right in first))
        self.assertEqual({left for left, _ in first}, {right for _, right in first})

    def test_hungarian_matches_brute_force(self) -> None:
        rng = np.random.default_rng(1718)
        for n in range(2, 8):
            for _ in range(30):
                matrix = rng.integers(0, 10_000, size=(n, n)).tolist()
                self.assertEqual(hungarian_exact(matrix), brute_force_assignment(matrix))
        tied = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertEqual(hungarian_exact(tied), (0, 1, 2))
        with self.assertRaisesRegex(RuntimeError, "NO_PERFECT_MATCHING"):
            hungarian_exact([[None, 1], [None, 2]])

    def test_hungarian_sparse_regression_and_contract_scale_oracle(self) -> None:
        # Regression: a pending column set by an earlier tree row must stay
        # visible to the delta scan when the active row has no edge to it.
        reproduction = [[None, 8, 8], [2, None, None], [4, 5, None]]
        self.assertEqual(hungarian_exact(reproduction), (2, 0, 1))
        self.assertEqual(brute_force_assignment(reproduction), (2, 0, 1))
        # Contract 4.1: brute-force comparison on 1,000 development graphs of
        # sizes 2..8, including tied-primary and infeasible cases.  Support is
        # symmetric (undirected eligibility), values asymmetric, mirroring the
        # production caliper graphs.
        rng = np.random.default_rng(240815)
        densities = (0.35, 0.6, 0.85, 1.0)
        for graph_index in range(1_000):
            n = 2 + graph_index % 7
            density = densities[graph_index % len(densities)]
            high = 13 if graph_index % 5 == 0 else 10_000
            costs: list[list[int | None]] = [[None] * n for _ in range(n)]
            for row in range(n):
                for column in range(row + 1, n):
                    if rng.random() <= density:
                        costs[row][column] = int(rng.integers(0, high))
                        costs[column][row] = int(rng.integers(0, high))
            best_total = None
            argmins: list[tuple[int, ...]] = []
            for permutation in itertools.permutations(range(n)):
                selected = [costs[r][c] for r, c in enumerate(permutation)]
                if any(value is None for value in selected):
                    continue
                total = sum(selected)
                if best_total is None or total < best_total:
                    best_total, argmins = total, [permutation]
                elif total == best_total:
                    argmins.append(permutation)
            if best_total is None:
                with self.assertRaisesRegex(
                    RuntimeError, "NO_PERFECT_MATCHING", msg=f"graph {graph_index}",
                ):
                    hungarian_exact(costs)
                continue
            solved = hungarian_exact(costs)
            if len(argmins) == 1:
                self.assertEqual(solved, argmins[0], msg=f"graph {graph_index}")
            else:
                solved_total = sum(costs[r][c] for r, c in enumerate(solved))
                self.assertEqual(solved_total, best_total, msg=f"graph {graph_index}")

    def test_control3_component_decomposition_matches_global_optimum(self) -> None:
        # The composed integer cost makes the global optimum unique, so the
        # component-wise solve must reproduce the full-graph brute force.
        group_ids = ("a", "b", "c", "d", "e", "f", "g")
        half = {"a", "b", "c"}
        edges = {
            (left, right) for left in group_ids for right in group_ids
            if left != right and ((left in half) == (right in half))
        }
        seed = (2026).to_bytes(8, "big")
        partition = "outer:0:held"
        mapping = control3_matching(group_ids, edges, seed, partition)
        ordered = tuple(sorted(group_ids))
        index = {value: position for position, value in enumerate(ordered)}
        n = len(ordered)
        base = (n + 1) ** n
        costs = [[None] * n for _ in range(n)]
        for left, right in sorted(edges):
            row, column = index[left], index[right]
            payload = (
                seed + b"\0" + partition.encode("utf-8") + b"\0"
                + left.encode("utf-8") + b"\0" + right.encode("utf-8")
            )
            primary = int.from_bytes(hashlib.sha256(payload).digest(), "big")
            costs[row][column] = primary * base + column * (n + 1) ** (n - 1 - row)
        expected = brute_force_assignment(costs)
        expected_pairs = tuple(
            (ordered[row], ordered[column]) for row, column in enumerate(expected)
        )
        self.assertEqual(mapping, expected_pairs)

    def test_bootstrap_upper_endpoint_is_order_statistic_higher(self) -> None:
        # Contract 4: two-sided 95% upper endpoint = ascending one-based order
        # statistic ceil-at-method="higher", i.e. numpy quantile
        # method="higher".  The conventions differ exactly when 0.975*n is an
        # integer — including n_draws=40 here and the registered 20,000.
        self.assertEqual(math.ceil(0.975 * (40 - 1)), 39)
        self.assertNotEqual(math.ceil(0.975 * 40) - 1, 39)
        rows = []
        for fold in range(5):
            for offset in range(20):
                group = 20 * fold + offset
                rows.extend((
                    _synthetic_row(group, fold, 0, float(group)),
                    _synthetic_row(group, fold, 1, float(group) + 0.25),
                ))
        scores = tuple(
            float(row.target if int(row.group_id[1:]) % 4 else 1 - row.target)
            for row in rows
        )
        bundles = tuple(OofBundle(
            population_id="qwen-source", ridge=ridge, scores=scores,
            fold_auc=(0.75,) * 5, fits=(),
        ) for ridge in RIDGES)
        result = shortcut_gate_bootstrap(rows, bundles, "overall", n_draws=40)
        draws = np.asarray(result.bootstrap_max_macro_auc, dtype=np.float64)
        self.assertGreater(len(set(result.bootstrap_max_macro_auc)), 4)
        self.assertEqual(
            result.upper_95, float(np.quantile(draws, 0.975, method="higher")),
        )

    def test_control3_uses_every_row_and_column_once(self) -> None:
        group_ids = ("a", "b", "c", "d")
        edges = {(left, right) for left in group_ids for right in group_ids if left != right}
        mapping = control3_matching(
            group_ids, edges, (77).to_bytes(8, "big"), "outer:0:held",
        )
        self.assertEqual({left for left, _ in mapping}, set(group_ids))
        self.assertEqual({right for _, right in mapping}, set(group_ids))
        self.assertTrue(all(left != right for left, right in mapping))

    def test_frozen_partition_rosters(self) -> None:
        records = group_matching_records(self.payloads, self.null_strata)
        partitions = canonical_partition_memberships(records, self.inner_folds)
        self.assertEqual(len(records), 900)
        self.assertEqual(len(partitions), 60)
        by_id = dict(partitions)
        for outer in range(5):
            self.assertEqual(len(by_id[f"outer:{outer}:train"]), 720)
            self.assertEqual(len(by_id[f"outer:{outer}:held"]), 180)
            for inner in range(5):
                self.assertEqual(len(by_id[f"outer:{outer}:inner:{inner}:train"]), 576)
                self.assertEqual(len(by_id[f"outer:{outer}:inner:{inner}:validation"]), 144)

    def test_complete_control_schedule_is_deterministic(self) -> None:
        group_ids = tuple(f"g{index}" for index in range(8))
        records = tuple(GroupMatchingRecord(
            group_id=group_id, outer_fold=0,
            null_stratum_id="s0" if index < 4 else "s1",
            source_record_id=f"source{index}", donor_id=f"donor{index}",
            template_bank_id=f"bank{index}",
        ) for index, group_id in enumerate(group_ids))
        partitions = (("outer:0:held", group_ids),)
        edges = tuple(
            (left, right) for left in group_ids for right in group_ids
            if left != right and ((int(left[1:]) < 4) == (int(right[1:]) < 4))
        )
        for family in (2, 3):
            first = materialize_control_schedule(family, 3, partitions, records, edges)
            second = materialize_control_schedule(family, 3, partitions, records, edges)
            self.assertEqual(first, second)
            self.assertEqual(len(first.assignments[0][1]), 8)
            self.assertNotEqual(first.schedule_sha256, "0" * 64)

    def test_caliper_uses_one_based_ceiling_order_statistic(self) -> None:
        self.assertEqual(distance_caliper([4.0, 1.0, 3.0, 2.0]), 3.0)
        self.assertEqual(distance_caliper([5.0, 1.0, 4.0, 2.0, 3.0]), 4.0)

    def test_pythia_nll_known_answer(self) -> None:
        import torch

        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long)}

        class Output:
            def __init__(self):
                self.logits = torch.zeros((1, 3, 4), dtype=torch.float32)

        class Model:
            def eval(self):
                self.evaluated = True

            def __call__(self, **kwargs):
                return Output()

        tokenizer, model = Tokenizer(), Model()
        self.assertAlmostEqual(pythia_prompt_mean_nll(model, tokenizer, "hello"), math.log(4.0))
        self.assertTrue(model.evaluated)
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])
        self.assertFalse(tokenizer.kwargs["padding"])
        self.assertFalse(tokenizer.kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
