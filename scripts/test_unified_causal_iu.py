#!/usr/bin/env python3
"""Deterministic causal, leakage, and state tests for Unified Causal IU-PCR."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The repository package exports GPU helpers eagerly.  These CPU-only regression tests
# intentionally run in the base setup.py dependency set, without installing PyTorch.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.unified_causal_evaluation import (
    AtlasSamples,
    assert_group_split_isolation,
    build_atlas_samples,
    grouped_block_permutation,
    heldout_logloss_gain,
    processbench_metrics,
)
from spectral_utils.unified_causal_iu import (
    AccumulatorSpec,
    AccumulatorState,
    BASE_NAMES,
    CausalFeatureBankState,
    UnifiedCausalIU,
    all_feature_names,
    base_matrix,
    calibrate_warning_thresholds,
    causal_feature_matrix,
    fit_base_reference,
)
from scripts.run_unified_causal_iu_v1 import (
    _select_dufs_lambda,
    _step_top5_location,
    grouped_bootstrap_comparisons,
)


def _row(seed: int, n: int = 72, *, family: str = "gsm8k", model: str = "qwen") -> dict:
    rng = np.random.default_rng(seed)
    change = n // 2 if seed % 2 == 0 else n + 1
    latent = rng.normal(0.0, 0.10, n)
    latent[change:] += np.linspace(0.2, 1.2, max(0, n - change))
    entropy = np.clip(1.0 + latent + rng.normal(0.0, 0.04, n), 1e-4, None)
    spilled = np.clip(0.7 + 0.7 * latent + rng.normal(0.0, 0.05, n), 1e-4, None)
    logsumexp = 25.0 - 0.8 * latent + rng.normal(0.0, 0.08, n)
    topk = rng.normal(-3.0, 0.3, size=(n, 8))
    topk[:, 0] = -0.2 - 0.5 * latent + rng.normal(0.0, 0.03, n)
    topk[:, 1] = topk[:, 0] - np.abs(0.5 - 0.2 * latent + rng.normal(0.0, 0.03, n))
    topk = -np.sort(-topk, axis=1)
    wrong = seed % 2 == 0
    return {
        "id": f"{family}-{seed}",
        "_unit": f"{family}-{seed}",
        "family": family,
        "model": model,
        "token_entropies": entropy,
        "token_spilled_energies": spilled,
        "token_logsumexp": logsumexp,
        "top_k_logprobs": {
            "ids": np.tile(np.arange(topk.shape[1]), (n, 1)),
            "logprobs": topk,
        },
        "label": 1 if wrong else -1,
        "final_answer_correct": not wrong,
        "step_token_spans": [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)],
    }


def _small_roster() -> tuple[str, ...]:
    names = all_feature_names()
    wanted = (
        "raw::entropy::level",
        "raw::entropy::ewma8",
        "raw::entropy::fastminus4_16",
        "raw::entropy::var16",
        "raw::spilled::level",
        "raw::spilled::positive_area",
        "raw::neg_margin::level",
        "raw::topk_varentropy::ewma16",
        "broad::entropy_sw_var_series::level",
        "broad::entropy_cusum_abs_series::innovation64",
        "broad::topk_tail_mass_series::persistence",
        "broad::energy_series::page_hinkley_pos_d005",
    )
    assert set(wanted) <= set(names)
    return wanted


def test_suffix_invariance() -> None:
    rows = [_row(index) for index in range(10)]
    reference = fit_base_reference(rows[:6])
    row = rows[7]
    changed = deepcopy(row)
    cut = 31
    changed["token_entropies"][cut:] += 40.0
    changed["token_spilled_energies"][cut:] += 30.0
    changed["token_logsumexp"][cut:] -= 20.0
    changed["top_k_logprobs"]["logprobs"][cut:, 0] -= 4.0
    left = causal_feature_matrix(row, reference)
    right = causal_feature_matrix(changed, reference)
    np.testing.assert_array_equal(left[:cut], right[:cut])


def test_tokenwise_chunked_identity() -> None:
    rows = [_row(index) for index in range(12)]
    reference = fit_base_reference(rows[:8])
    matrix = reference.transform(base_matrix(rows[9], reference.names))
    tokenwise = CausalFeatureBankState(reference.names).update_many(matrix, chunk_size=1)
    for chunk in (2, 7, 16, 128):
        replay = CausalFeatureBankState(reference.names).update_many(matrix, chunk_size=chunk)
        np.testing.assert_array_equal(tokenwise, replay)

    model = UnifiedCausalIU.fit(rows[:8], feature_roster=_small_roster())
    one = model.score_row(rows[9], chunk_size=1)
    many = model.score_row(rows[9], chunk_size=13)
    assert one == many


def test_precomputed_causal_matrix_scoring_identity() -> None:
    rows = [_row(index, family="math") for index in range(14)]
    model = UnifiedCausalIU.fit(
        rows[:10],
        feature_roster=_small_roster(),
        feature_signs={
            name: -1.0 if index % 3 == 0 else 1.0
            for index, name in enumerate(_small_roster())
        },
        accumulator=AccumulatorSpec("leaky", 16, 0.25),
    ).with_thresholds(0.75, 0.25)
    row = rows[12]
    matrix = causal_feature_matrix(row, model.reference)
    live = model.score_row(row, chunk_size=1)

    evidence = model.evidence_from_feature_matrix(matrix, chunk_size=7)
    np.testing.assert_array_equal(
        evidence,
        np.asarray([update.evidence for update in live.trajectory], dtype=float),
    )
    assert model.score_causal_matrix(matrix, chunk_size=7) == live
    for chunk_size in (1, 2, 13, 128):
        np.testing.assert_array_equal(
            model.evidence_from_feature_matrix(matrix, chunk_size=chunk_size),
            evidence,
        )
        assert model.score_causal_matrix(matrix, chunk_size=chunk_size) == live

    feature_order = tuple(reversed(all_feature_names(model.reference.names)))
    permuted = causal_feature_matrix(
        row,
        model.reference,
        feature_order=feature_order,
    )
    np.testing.assert_array_equal(
        model.evidence_from_feature_matrix(
            permuted,
            feature_order=feature_order,
            chunk_size=5,
        ),
        evidence,
    )
    assert model.score_causal_matrix(
        permuted,
        feature_order=feature_order,
        chunk_size=5,
    ) == live
    assert np.asarray(live.global_score, dtype=np.float64).view(np.uint64) == np.asarray(
        model.score_causal_matrix(matrix).trajectory[-1].risk,
        dtype=np.float64,
    ).view(np.uint64)


def test_raw_only_compact_reference_identity() -> None:
    rows = [_row(index, family="omnimath") for index in range(12)]
    raw_names = tuple(name for name in BASE_NAMES if name.startswith("raw::"))
    raw_matrices = [base_matrix(row, raw_names) for row in rows[:8]]
    for row, raw in zip(rows[:8], raw_matrices):
        np.testing.assert_array_equal(raw, base_matrix(row)[:, : len(raw_names)])
    reference = fit_base_reference(
        rows[:8], names=raw_names, raw_base_matrices=raw_matrices
    )
    matrices = [
        causal_feature_matrix(row, reference, raw_base=raw)
        for row, raw in zip(rows[:8], raw_matrices)
    ]
    roster = tuple(
        name for name in all_feature_names(raw_names)
        if name.rsplit("::", 1)[-1] in {"level", "ewma16", "persistence"}
    )
    model = UnifiedCausalIU.fit(
        rows[:8],
        feature_roster=roster,
        reference=reference,
        feature_matrices=matrices,
    )
    row = rows[10]
    compact = causal_feature_matrix(row, reference)
    assert compact.shape[1] == len(raw_names) * 28
    assert model.score_causal_matrix(compact) == model.score_row(row)
    samples = build_atlas_samples(
        rows[:8], reference, target="early", feature_matrices=matrices
    )
    assert samples.baseline_names == ("entropy",)


def test_terminal_global_bit_identity() -> None:
    rows = [_row(index) for index in range(12)]
    model = UnifiedCausalIU.fit(
        rows[:8],
        feature_roster=_small_roster(),
        accumulator=AccumulatorSpec("leaky", 16, 0.25),
    )
    final = model.score_row(rows[10])
    assert np.asarray(final.global_score, dtype=np.float64).view(np.uint64) == np.asarray(
        final.trajectory[-1].risk, dtype=np.float64
    ).view(np.uint64)


def test_feature_order_invariance() -> None:
    rows = [_row(index, family="math") for index in range(16)]
    roster = _small_roster()
    signs = {name: -1.0 if index % 3 == 0 else 1.0 for index, name in enumerate(roster)}
    first = UnifiedCausalIU.fit(rows[:10], feature_roster=roster, feature_signs=signs)
    second = UnifiedCausalIU.fit(
        rows[:10], feature_roster=tuple(reversed(roster)), feature_signs=signs
    )
    for row in rows[10:]:
        left = np.asarray([item.risk for item in first.score_row(row).trajectory])
        right = np.asarray([item.risk for item in second.score_row(row).trajectory])
        np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-8)


def test_dufs_laplacian_path_contract() -> None:
    """Exercise the graph/IU path without pretending to test the optional optimizer."""

    rows = [_row(index, family="math") for index in range(14)]
    ordinary = UnifiedCausalIU.fit(rows[:10], feature_roster=_small_roster())
    module_name = "spectral_utils.adapted_dufs"
    previous = sys.modules.get(module_name)
    fake = types.ModuleType(module_name)

    def deterministic_gates(F, **_kwargs):
        gates = np.linspace(0.5, 1.5, F.shape[0])
        return gates, {"test_double": True, "effective_feature_count": float(F.shape[0])}

    fake.adapted_dufs_soft_gates = deterministic_gates
    sys.modules[module_name] = fake
    try:
        path = UnifiedCausalIU.fit_dufs_path(
            rows[:10],
            lambdas=(0.1, 0.3),
            feature_roster=_small_roster(),
            ordinary_model=ordinary,
            dufs_seeds=(1,),
            dufs_epochs=1,
        )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    assert set(path) == {0.1, 0.3}
    for lambda_, model in path.items():
        assert model.feature_names == ordinary.feature_names
        np.testing.assert_array_equal(model.feature_indices, ordinary.feature_indices)
        assert model.diagnostics["graph_lambda"] == lambda_
        assert model.diagnostics["same_roster_as_ordinary"] is True
        assert model.diagnostics["lambda_zero_exact"] is True
        assert np.isfinite(model.score_row(rows[12]).global_score)
    assert not np.array_equal(path[0.1].weights, ordinary.weights)


def test_actual_dufs_optimizer_smoke() -> None:
    rows = [_row(index, family="omnimath") for index in range(12)]
    ordinary = UnifiedCausalIU.fit(rows[:8], feature_roster=_small_roster())
    model = UnifiedCausalIU.fit_dufs_path(
        rows[:8],
        lambdas=(0.1,),
        feature_roster=_small_roster(),
        ordinary_model=ordinary,
        dufs_seeds=(11,),
        dufs_epochs=2,
    )[0.1]
    diagnostics = model.diagnostics["dufs_gate_diagnostics"]
    assert diagnostics["effective_feature_count"] > 0.0
    assert diagnostics["per_seed_probabilities"].shape == (1, len(model.feature_names))
    assert np.isfinite(model.score_row(rows[10]).global_score)


def test_dufs_lambda_selection_is_maximin() -> None:
    ordinary = {"global": 0.70, "localization": 0.35, "early": 0.61}
    selected, ledger = _select_dufs_lambda(ordinary, {
        0.1: {"global": 0.704, "localization": 0.354, "early": 0.612},
        0.3: {"global": 0.715, "localization": 0.349, "early": 0.620},
        1.0: {"global": 0.730, "localization": 0.320, "early": 0.640},
    })
    assert selected == 0.1
    assert next(row for row in ledger if row["lambda"] == 1.0)["survives_margins"] is False
    selected, _ = _select_dufs_lambda(ordinary, {
        0.1: {"global": 0.704, "localization": 0.349, "early": 0.612},
    })
    assert selected == 0.0


def test_no_length_or_future_feature() -> None:
    assert all("trace_length" not in name for name in all_feature_names())
    rows = [_row(index) for index in range(5)]
    try:
        fit_base_reference(rows, names=("raw::entropy", "trace_length", "raw::spilled"))
    except ValueError as error:
        assert "non-causal" in str(error)
    else:
        raise AssertionError("trace length entered the causal feature path")


def test_missing_channels_are_deterministic() -> None:
    rows = [_row(index) for index in range(12)]
    model = UnifiedCausalIU.fit(rows[:8], feature_roster=_small_roster())
    missing = deepcopy(rows[9])
    missing.pop("token_spilled_energies")
    missing.pop("token_logsumexp")
    missing.pop("top_k_logprobs")
    first = model.score_row(missing)
    second = model.score_row(missing)
    assert first == second
    assert all(np.isfinite(item.risk) for item in first.trajectory)


def test_synthetic_filters() -> None:
    identity = AccumulatorState(AccumulatorSpec("identity"))
    identity_values = [identity.update(value) for value in (0.0, 0.0, 4.0, 0.0)]
    assert int(np.argmax([value[1] for value in identity_values])) == 2

    leaky = AccumulatorState(AccumulatorSpec("leaky", 8, 0.0))
    drift = [leaky.update(value)[0] for value in (0.2, 0.3, 0.4, 0.5)]
    assert np.all(np.diff(drift) > 0.0)
    recovered = leaky.update(-3.0)[0]
    assert recovered < drift[-1]

    hazard = AccumulatorState(AccumulatorSpec("hazard"))
    persistent = [hazard.update(1.0)[0] for _ in range(8)]
    assert np.all(np.diff(persistent) >= 0.0)
    before = persistent[-1]
    assert hazard.update(-5.0)[0] == before


def test_warning_calibration_and_finalization() -> None:
    rows = [_row(index) for index in range(20)]
    model = UnifiedCausalIU.fit(rows[:12], feature_roster=_small_roster())
    calibrated = calibrate_warning_thresholds(
        model,
        rows[:12],
        is_clean=[not bool(index % 2 == 0) for index in range(12)],
    )
    assert calibrated.warning_threshold_5pct >= calibrated.warning_threshold_10pct
    final = calibrated.score_row(rows[15])
    assert final.first_alarm_token == final.first_alarm_token_10pct


def test_group_split_isolation() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    good = [(np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7]))]
    assert_group_split_isolation(good, groups)
    bad = [(np.asarray([0, 2, 4]), np.asarray([1, 3, 5]))]
    try:
        assert_group_split_isolation(bad, groups)
    except AssertionError:
        pass
    else:
        raise AssertionError("question leakage was not detected")


def test_grouped_label_permutation_removes_gain() -> None:
    rng = np.random.default_rng(41)
    n_groups, per_group = 100, 2
    group_labels = np.asarray([index % 2 for index in range(n_groups)], dtype=int)
    y = np.repeat(group_labels, per_group)
    groups = np.repeat([f"q{index}" for index in range(n_groups)], per_group)
    signal = y + rng.normal(0.0, 0.05, len(y))
    noise = rng.normal(size=(len(y), 2))
    context = np.column_stack([np.ones(len(y)), noise[:, 0]])
    samples = AtlasSamples(
        target="global",
        feature_names=("signal",),
        X=signal[:, None],
        y=y,
        groups=groups,
        families=np.repeat("gsm8k", len(y)),
        models=np.repeat("qwen", len(y)),
        budgets=np.repeat("final", len(y)),
        context=context,
        context_names=("constant", "noise"),
        baseline=noise[:, 1:],
        baseline_names=("noise",),
    )
    actual, _ = heldout_logloss_gain(samples, [0], n_splits=5, seed=4)
    permuted = grouped_block_permutation(y, groups, np.random.default_rng(8))
    null, _ = heldout_logloss_gain(samples, [0], n_splits=5, seed=4, labels=permuted)
    assert actual > 0.20
    assert abs(null) < 0.08


def test_information_atlas_conditions_on_iu28() -> None:
    rows = [_row(index) for index in range(4)]
    for row in rows:
        row["_source_group"] = row["_unit"]
    reference = fit_base_reference(rows)
    curves = [np.linspace(index, index + 1.0, len(row["token_entropies"])) for index, row in enumerate(rows)]
    samples = build_atlas_samples(rows, reference, target="early", iu28_curves=curves)
    assert "IU28" in samples.baseline_names
    iu_index = samples.baseline_names.index("IU28")
    assert np.unique(samples.baseline[:, iu_index]).size > 4


def test_localization_metric_contract() -> None:
    result = processbench_metrics([1, -1, 2, -1], [1, -1, 1, -1])
    assert result["exact"] == 0.5
    assert result["within_one"] == 1.0
    assert result["clean_abstention"] == 1.0
    np.testing.assert_allclose(result["f1"], 2.0 / 3.0)


def test_max_entropy_top5_step_locator() -> None:
    row = {"step_token_spans": [(0, 6), (6, 12)]}
    curve = np.asarray([10.0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3])
    token, step = _step_top5_location(curve, row)
    assert step == 1
    assert token == 6


def test_grouped_bootstrap_gate_contract() -> None:
    records = []
    methods = ("unified_causal_iu", "baseline")
    for fold in range(2):
        for family in ("gsm8k", "math"):
            for question in range(4):
                wrong = int(question % 2)
                group = f"{family}::fold{fold}-q{question}"
                for model in ("m1", "m2"):
                    for method in methods:
                        good = method == "unified_causal_iu"
                        score = (0.9 if wrong else 0.1) if good else (0.1 if wrong else 0.9)
                        records.append({
                            "outer_fold": fold,
                            "candidate": method,
                            "source_group": group,
                            "family": family,
                            "model": model,
                            "wrong": wrong,
                            "target_step": 0 if wrong else -1,
                            "prediction": (0 if wrong else -1) if good else (-1 if wrong else 0),
                            "global_score": score,
                            "risk_at_64": score,
                            "risk_at_128": score,
                        })
    result = grouped_bootstrap_comparisons(records, repeats=50, seed=11)
    for task in ("global", "localization", "early"):
        comparison = result["comparisons"]["baseline"][task]
        assert comparison["delta"] > 0.9
        assert comparison["ci95"][0] > (0.4 if task == "localization" else 0.9)


def main() -> None:
    tests = (
        test_suffix_invariance,
        test_tokenwise_chunked_identity,
        test_precomputed_causal_matrix_scoring_identity,
        test_raw_only_compact_reference_identity,
        test_terminal_global_bit_identity,
        test_feature_order_invariance,
        test_dufs_laplacian_path_contract,
        test_actual_dufs_optimizer_smoke,
        test_dufs_lambda_selection_is_maximin,
        test_no_length_or_future_feature,
        test_missing_channels_are_deterministic,
        test_synthetic_filters,
        test_warning_calibration_and_finalization,
        test_group_split_isolation,
        test_grouped_label_permutation_removes_gain,
        test_information_atlas_conditions_on_iu28,
        test_localization_metric_contract,
        test_max_entropy_top5_step_locator,
        test_grouped_bootstrap_gate_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    main()
