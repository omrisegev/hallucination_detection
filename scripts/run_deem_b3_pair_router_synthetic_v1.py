#!/usr/bin/env python3
"""Leakage-safe held-out synthetic audit for the frozen-B3 pair router.

The fit phase receives donor features only.  Synthetic held features, labels,
and planted specialist identities are generated only after B3, A0, A4, and A5
have been frozen.  A5's graph is a donor-only graph over cross-fitted genuine
leave-one-family-out B3-family residuals; no held row is inserted into it.

The default run uses twenty fixed replicates in each of four worlds.  The
runtime-scaled A4/A5 analogues preserve the production actuation and penalty
weights while reducing epochs and MALA steps for this mechanism screen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_pair_router import (  # noqa: E402
    PairRouterConfig,
    PairRouterResult,
    fit_pair_residual_router,
    predict_pair_residual_router,
)
from spectral_utils.graph_topology import self_safe_knn_graph  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    ContinuousDeemResult,
    atomic_write_json,
    fit_continuous_deem,
    predict_continuous_deem,
)


NAMES = (
    "epr",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
    "trace_length",
)
WORLDS = (
    "switching_specialists",
    "no_switch",
    "coherent_nuisance",
    "density_only",
)
DEFAULT_REPLICATE_SEEDS = tuple(range(20))
A0 = "A0_B3_EXACT_ALIAS"
A4 = "A4_PAIR_RESIDUAL_CD_A50"
A5 = "A5_PAIR_RESIDUAL_CD_GEO_A50"
VARIANTS = (A0, A4, A5)


def generate_world(
    world: str,
    *,
    seed: int,
    n: int,
    reveal_truth: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Generate one world, optionally returning truth for held evaluation."""

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    y = rng.integers(0, 2, size=int(n), dtype=np.int8)
    target = 2.0 * y - 1.0
    regime = rng.integers(0, 2, size=int(n), dtype=np.int8)
    regime_sign = 2.0 * regime - 1.0
    noise = rng.normal(size=(int(n), len(NAMES)))
    nuisance = rng.normal(size=int(n))

    if world == "switching_specialists":
        # Families zero and one alternate as specialists.  Persistent weak
        # target coordinates keep the EBM target identifiable, while the last
        # family exposes the target-free regime context to every routed pair.
        X = np.column_stack(
            [
                np.where(
                    regime == 0,
                    1.50 * target + 0.50 * noise[:, 0],
                    1.80 * noise[:, 0],
                ),
                np.where(
                    regime == 1,
                    1.50 * target + 0.50 * noise[:, 1],
                    1.80 * noise[:, 1],
                ),
                0.25 * target + noise[:, 2],
                0.25 * target + noise[:, 3],
                0.25 * target + noise[:, 4],
                0.80 * regime_sign + 0.55 * noise[:, 5],
            ]
        )
        active = regime
    elif world == "no_switch":
        # Exchangeable equally useful families: row-adaptive routing should
        # not improve systematically over frozen B3.
        X = 0.80 * target[:, None] + noise
        active = None
    elif world == "coherent_nuisance":
        # Two target-bearing families and four strongly coherent nuisance
        # families test whether density fit invents harmful specialization.
        X = np.column_stack(
            [
                0.90 * target + 0.75 * noise[:, 0],
                0.90 * target + 0.75 * noise[:, 1],
                1.80 * nuisance + 0.35 * noise[:, 2],
                1.75 * nuisance + 0.35 * noise[:, 3],
                1.70 * nuisance + 0.35 * noise[:, 4],
                1.60 * nuisance + 0.40 * noise[:, 5],
            ]
        )
        active = None
    elif world == "density_only":
        # A strong, label-independent bimodal density factor competes with a
        # deliberately weak target carried by the first two families.
        mode = rng.choice(np.array([-2.0, 2.0]), size=int(n))
        X = np.column_stack(
            [
                0.25 * target + noise[:, 0],
                0.25 * target + noise[:, 1],
                mode + 0.25 * noise[:, 2],
                -mode + 0.25 * noise[:, 3],
                0.75 * mode + 0.35 * noise[:, 4],
                -0.65 * mode + 0.40 * noise[:, 5],
            ]
        )
        active = None
    else:
        raise ValueError(f"unknown synthetic world: {world}")

    truth = None
    if reveal_truth:
        truth = {"label": y}
        if active is not None:
            truth["active_specialist"] = np.asarray(active, dtype=np.int8)
    return np.asarray(X, dtype=np.float64), truth


def donor_standardization(donor_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = donor_raw.mean(axis=0)
    scale = donor_raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (donor_raw - mean[None, :]) / scale[None, :], mean, scale


def apply_standardization(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean[None, :]) / scale[None, :]


def family_matrix(result: ContinuousDeemResult) -> tuple[np.ndarray, tuple[str, ...]]:
    order = tuple(result.family_indices)
    matrix = np.column_stack([result.family_contributions[name] for name in order])
    return np.asarray(matrix, dtype=np.float64), order


def true_loo_family_residuals(
    donor_family: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit each family from all other families on donor rows only.

    This mirrors the true family-LOO residual definition: each target family
    is absent from its predictors, and every reported residual is evaluated on
    a row excluded from the corresponding Ridge fit.  Residuals are divided by
    donor target SD, not residual SD, so predictable families stay near zero.
    """

    values = np.asarray(donor_family, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("donor family contributions must be a finite matrix")
    n, family_count = values.shape
    fold_count = int(folds)
    if fold_count < 2 or n <= fold_count + family_count:
        raise ValueError("not enough donor rows for cross-fitted family LOO")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    permutation = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int64)
    fold_id[permutation] = np.arange(n, dtype=np.int64) % fold_count
    residuals = np.zeros_like(values)

    for target in range(family_count):
        predictors = np.asarray(
            [column for column in range(family_count) if column != target],
            dtype=np.int64,
        )
        for fold in range(fold_count):
            held = np.flatnonzero(fold_id == fold)
            train = np.flatnonzero(fold_id != fold)
            donor_x = values[np.ix_(train, predictors)]
            held_x = values[np.ix_(held, predictors)]
            x_mean = donor_x.mean(axis=0)
            x_scale = donor_x.std(axis=0)
            x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
            donor_x = (donor_x - x_mean[None, :]) / x_scale[None, :]
            held_x = (held_x - x_mean[None, :]) / x_scale[None, :]
            donor_y = values[train, target]
            y_mean = float(np.mean(donor_y))
            y_scale = float(np.std(donor_y))
            if y_scale <= 1e-12:
                y_scale = 1.0
            standardized_y = (donor_y - y_mean) / y_scale
            held_y = (values[held, target] - y_mean) / y_scale
            estimator = Ridge(alpha=1.0, fit_intercept=True)
            estimator.fit(donor_x, standardized_y)
            residuals[held, target] = held_y - estimator.predict(held_x)
    if not np.isfinite(residuals).all():
        raise RuntimeError("non-finite donor LOO residual")
    return residuals, fold_id


def donor_only_graph(
    baseline: ContinuousDeemResult,
    *,
    seed: int,
    k: int,
):
    contributions, family_order = family_matrix(baseline)
    residuals, fold_id = true_loo_family_residuals(
        contributions,
        seed=int(seed),
    )
    graph = self_safe_knn_graph(
        residuals,
        k=int(k),
        tie_keys=np.arange(len(residuals), dtype=np.int64),
    )
    laplacian = symmetric_normalized_laplacian(graph)
    diagnostics = {
        "source": "donor_crossfit_leave_one_family_out_b3_contribution_residuals",
        "family_order": family_order,
        "n_donor_rows": int(len(residuals)),
        "n_held_rows": 0,
        "fold_count": int(len(np.unique(fold_id))),
        "k": int(k),
        "undirected_edges": int(graph.nnz // 2),
        "residual_mean_abs": float(np.mean(np.abs(residuals))),
        "residual_sd": float(np.std(residuals)),
    }
    return laplacian, diagnostics


def variant_configs(*, epochs: int, mala_steps: int) -> dict[str, PairRouterConfig]:
    common: dict[str, Any] = {
        "epochs": int(epochs),
        "learning_rate": 2e-3,
        "deem_weight": 1.0,
        "trust_weight": 0.01,
        "open_weight": 0.001,
        "l2_weight": 1e-4,
        "open_warmup_epochs": min(10, int(epochs)),
        "mala_steps": int(mala_steps),
    }
    return {
        A0: PairRouterConfig(
            rho=0.0,
            epochs=0,
            deem_weight=1.0,
            graph_weight=0.0,
            mala_steps=int(mala_steps),
        ),
        A4: PairRouterConfig(rho=0.5, graph_weight=0.0, **common),
        A5: PairRouterConfig(rho=0.5, graph_weight=0.1, **common),
    }


def fit_donor_only(
    donor: np.ndarray,
    *,
    baseline_seed: int,
    router_seed: int,
    graph_seed: int,
    baseline_epochs: int,
    router_epochs: int,
    mala_steps: int,
    graph_k: int,
) -> tuple[
    ContinuousDeemResult,
    dict[str, PairRouterResult],
    dict[str, Any],
    dict[str, PairRouterConfig],
]:
    """Fit all models through an API that cannot receive evaluation truth."""

    baseline = fit_continuous_deem(
        donor,
        NAMES,
        seed=int(baseline_seed),
        config=ContinuousDeemConfig(
            epochs=int(baseline_epochs),
            mala_steps=int(mala_steps),
            anchor_tolerance=1e-12,
            posterior_sd_min=1e-3,
        ),
    )
    if not baseline.health["healthy"]:
        raise RuntimeError("unhealthy donor B3 fit")
    laplacian, graph_diagnostics = donor_only_graph(
        baseline,
        seed=int(graph_seed),
        k=int(graph_k),
    )
    if tuple(laplacian.shape) != (len(donor), len(donor)):
        raise AssertionError("donor graph has an unexpected shape")

    configurations = variant_configs(epochs=router_epochs, mala_steps=mala_steps)
    fitted: dict[str, PairRouterResult] = {}
    for variant in VARIANTS:
        configuration = configurations[variant]
        fitted[variant] = fit_pair_residual_router(
            donor,
            NAMES,
            baseline.state,
            baseline_score=baseline.score,
            baseline_orientation=baseline.orientation,
            seed=int(router_seed),
            config=configuration,
            laplacian=laplacian if configuration.graph_weight > 0.0 else None,
        )
        if not fitted[variant].health["healthy"]:
            raise RuntimeError(f"unhealthy donor pair-router fit: {variant}")
    if not np.array_equal(fitted[A0].score, baseline.score):
        raise AssertionError("A0 donor score is not a byte-exact B3 alias")
    return baseline, fitted, graph_diagnostics, configurations


def sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0)))


def specialist_alignment(
    gates: np.ndarray,
    active: np.ndarray | None,
    family_order: tuple[str, ...],
) -> dict[str, float | None]:
    if active is None:
        return {"gate_alignment": None, "gate_selection_accuracy": None}
    first = family_order.index("entropy_level")
    second = family_order.index("entropy_dynamics")
    selected = np.where(active == 0, gates[:, first], gates[:, second])
    rejected = np.where(active == 0, gates[:, second], gates[:, first])
    chosen = (gates[:, second] > gates[:, first]).astype(np.int8)
    return {
        "gate_alignment": float(np.mean(selected - rejected)),
        "gate_selection_accuracy": float(np.mean(chosen == active)),
    }


def evaluate_score(
    *,
    world: str,
    replicate: int,
    variant: str,
    control: str,
    score: np.ndarray,
    gates: np.ndarray | None,
    y_held: np.ndarray,
    active_held: np.ndarray | None,
    family_order: tuple[str, ...] | None,
    baseline_auc: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    auc = float(roc_auc_score(y_held, score))
    alignment = (
        specialist_alignment(gates, active_held, family_order)
        if gates is not None and family_order is not None
        else {"gate_alignment": None, "gate_selection_accuracy": None}
    )
    return {
        "world": world,
        "replicate": int(replicate),
        "variant": variant,
        "control": control,
        "held_auroc": auc,
        "delta_vs_frozen_b3": auc - float(baseline_auc),
        **alignment,
        "gate_mean_abs_deviation_from_one": (
            float(np.mean(np.abs(gates - 1.0))) if gates is not None else None
        ),
        "fit_runtime_seconds": float(runtime_seconds),
    }


def evaluate_frozen_models(
    *,
    world: str,
    replicate: int,
    held: np.ndarray,
    truth: Mapping[str, np.ndarray],
    baseline: ContinuousDeemResult,
    fitted: Mapping[str, PairRouterResult],
    row_permutation_seed: int,
) -> tuple[list[dict[str, Any]], float]:
    """Evaluate only after every donor-only fit has returned and frozen."""

    y_held = np.asarray(truth["label"], dtype=np.int8)
    active_held = truth.get("active_specialist")
    baseline_prediction = predict_continuous_deem(baseline, held)
    baseline_score = np.asarray(baseline_prediction["score"], dtype=np.float64)
    baseline_auc = float(roc_auc_score(y_held, baseline_score))
    rows = [
        evaluate_score(
            world=world,
            replicate=replicate,
            variant="B3_FROZEN",
            control="BASELINE",
            score=baseline_score,
            gates=None,
            y_held=y_held,
            active_held=active_held,
            family_order=None,
            baseline_auc=baseline_auc,
            runtime_seconds=float(baseline.health["runtime_seconds"]),
        )
    ]

    permutation_rng = np.random.Generator(
        np.random.PCG64(int(row_permutation_seed))
    )
    permutation = permutation_rng.permutation(len(held))
    if len(held) > 1 and np.array_equal(permutation, np.arange(len(held))):
        permutation = np.roll(permutation, 1)
    alias_error = 0.0
    for variant in VARIANTS:
        result = fitted[variant]
        prediction = predict_pair_residual_router(
            result,
            held,
            baseline_score=baseline_score,
        )
        gates = np.asarray(prediction["gates"], dtype=np.float64)
        if np.max(np.abs(gates.sum(axis=1) - gates.shape[1])) > 1e-10:
            raise AssertionError(f"held fixed-sum gate invariant failed: {variant}")
        active_score = np.asarray(prediction["score"], dtype=np.float64)
        if variant == A0:
            alias_error = float(np.max(np.abs(active_score - baseline_score)))
            if not np.array_equal(active_score, baseline_score):
                raise AssertionError("A0 held score is not a byte-exact B3 alias")
        rows.append(
            evaluate_score(
                world=world,
                replicate=replicate,
                variant=variant,
                control="ACTIVE",
                score=active_score,
                gates=gates,
                y_held=y_held,
                active_held=active_held,
                family_order=result.family_order,
                baseline_auc=baseline_auc,
                runtime_seconds=float(result.health["runtime_seconds"]),
            )
        )
        if variant == A0:
            continue

        base_family = np.asarray(
            prediction["base_family_contributions"], dtype=np.float64
        )
        donor_mean_gate = np.asarray(result.gates, dtype=np.float64).mean(axis=0)
        static_gates = np.broadcast_to(donor_mean_gate, gates.shape).copy()
        static_logit = result.aligned_bias + np.sum(
            base_family * static_gates,
            axis=1,
        )
        rows.append(
            evaluate_score(
                world=world,
                replicate=replicate,
                variant=variant,
                control="STATIC_DONOR_MEAN_GATE",
                score=sigmoid(static_logit),
                gates=static_gates,
                y_held=y_held,
                active_held=active_held,
                family_order=result.family_order,
                baseline_auc=baseline_auc,
                runtime_seconds=float(result.health["runtime_seconds"]),
            )
        )
        permuted_gates = gates[permutation]
        permuted_logit = result.aligned_bias + np.sum(
            base_family * permuted_gates,
            axis=1,
        )
        rows.append(
            evaluate_score(
                world=world,
                replicate=replicate,
                variant=variant,
                control="HELD_ROW_PERMUTED_GATE",
                score=sigmoid(permuted_logit),
                gates=permuted_gates,
                y_held=y_held,
                active_held=active_held,
                family_order=result.family_order,
                baseline_auc=baseline_auc,
                runtime_seconds=float(result.health["runtime_seconds"]),
            )
        )
    return rows, alias_error


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["world"], row["variant"], row["control"])].append(row)
    output = []
    for (world, variant, control), selected in sorted(grouped.items()):
        deltas = np.asarray(
            [row["delta_vs_frozen_b3"] for row in selected],
            dtype=np.float64,
        )
        aucs = np.asarray([row["held_auroc"] for row in selected], dtype=np.float64)
        alignments = [
            row["gate_alignment"]
            for row in selected
            if row["gate_alignment"] is not None
        ]
        accuracies = [
            row["gate_selection_accuracy"]
            for row in selected
            if row["gate_selection_accuracy"] is not None
        ]
        output.append(
            {
                "world": world,
                "variant": variant,
                "control": control,
                "n": int(len(selected)),
                "mean_held_auroc": float(np.mean(aucs)),
                "mean_delta_vs_frozen_b3": float(np.mean(deltas)),
                "median_delta_vs_frozen_b3": float(np.median(deltas)),
                "sd_delta_vs_frozen_b3": float(np.std(deltas)),
                "wins": int(np.sum(deltas > 0.0)),
                "ties": int(np.sum(deltas == 0.0)),
                "losses": int(np.sum(deltas < 0.0)),
                "worst_delta_vs_frozen_b3": float(np.min(deltas)),
                "mean_gate_alignment": (
                    float(np.mean(alignments)) if alignments else None
                ),
                "mean_gate_selection_accuracy": (
                    float(np.mean(accuracies)) if accuracies else None
                ),
            }
        )
    return output


def mechanism_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["world"], row["replicate"], row["variant"], row["control"]): row
        for row in rows
    }
    output = []
    for world in WORLDS:
        for variant in (A4, A5):
            active_minus_static = []
            active_minus_permuted = []
            alignment_minus_permuted = []
            for replicate in sorted({row["replicate"] for row in rows}):
                prefix = (world, replicate, variant)
                active = lookup.get((*prefix, "ACTIVE"))
                static = lookup.get((*prefix, "STATIC_DONOR_MEAN_GATE"))
                permuted = lookup.get((*prefix, "HELD_ROW_PERMUTED_GATE"))
                if active is None or static is None or permuted is None:
                    continue
                active_minus_static.append(active["held_auroc"] - static["held_auroc"])
                active_minus_permuted.append(
                    active["held_auroc"] - permuted["held_auroc"]
                )
                if active["gate_alignment"] is not None:
                    alignment_minus_permuted.append(
                        active["gate_alignment"] - permuted["gate_alignment"]
                    )
            if active_minus_static:
                output.append(
                    {
                        "world": world,
                        "variant": variant,
                        "n": int(len(active_minus_static)),
                        "mean_active_minus_static_auroc": float(
                            np.mean(active_minus_static)
                        ),
                        "mean_active_minus_row_permuted_auroc": float(
                            np.mean(active_minus_permuted)
                        ),
                        "mean_active_minus_row_permuted_gate_alignment": (
                            float(np.mean(alignment_minus_permuted))
                            if alignment_minus_permuted
                            else None
                        ),
                    }
                )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", default=",".join(WORLDS))
    parser.add_argument("--replicates", type=int, default=len(DEFAULT_REPLICATE_SEEDS))
    parser.add_argument("--n-donor", type=int, default=192)
    parser.add_argument("--n-held", type=int, default=512)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--mala-steps", type=int, default=1)
    parser.add_argument("--graph-k", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worlds = tuple(value.strip() for value in args.worlds.split(",") if value.strip())
    if not worlds or not set(worlds).issubset(WORLDS):
        raise ValueError(f"worlds must be selected from {WORLDS}")
    if args.smoke:
        replicate_seeds = DEFAULT_REPLICATE_SEEDS[:1]
        n_donor = min(int(args.n_donor), 96)
        n_held = min(int(args.n_held), 192)
        baseline_epochs = min(int(args.baseline_epochs), 3)
        router_epochs = min(int(args.router_epochs), 3)
    else:
        if int(args.replicates) < len(DEFAULT_REPLICATE_SEEDS):
            raise ValueError("the non-smoke audit requires at least 20 fixed replicates")
        replicate_seeds = tuple(range(int(args.replicates)))
        n_donor = int(args.n_donor)
        n_held = int(args.n_held)
        baseline_epochs = int(args.baseline_epochs)
        router_epochs = int(args.router_epochs)
    if n_donor < 32 or n_held < 32:
        raise ValueError("donor and held splits must each contain at least 32 rows")
    if baseline_epochs < 1 or router_epochs < 1 or int(args.mala_steps) < 1:
        raise ValueError("epoch counts and MALA steps must be positive")

    rows: list[dict[str, Any]] = []
    graph_audits: list[dict[str, Any]] = []
    alias_errors = []
    executed_configs: dict[str, Any] | None = None
    world_indices = {name: index for index, name in enumerate(WORLDS)}
    for replicate in replicate_seeds:
        for world in worlds:
            world_index = world_indices[world]
            donor_seed = 50_000 + 211 * int(replicate) + world_index
            held_seed = 80_000 + 223 * int(replicate) + world_index
            baseline_seed = 10_000 + int(replicate)
            router_seed = 20_000 + int(replicate)
            graph_seed = 30_000 + int(replicate)
            permutation_seed = 90_000 + 307 * int(replicate) + world_index

            # The donor generator withholds its latent truth, and every fit API
            # below accepts only this feature matrix.
            donor_raw, donor_truth = generate_world(
                world,
                seed=donor_seed,
                n=n_donor,
                reveal_truth=False,
            )
            if donor_truth is not None:
                raise AssertionError("donor truth firewall failed")
            donor, donor_mean, donor_scale = donor_standardization(donor_raw)
            baseline, fitted, graph_diagnostics, configurations = fit_donor_only(
                donor,
                baseline_seed=baseline_seed,
                router_seed=router_seed,
                graph_seed=graph_seed,
                baseline_epochs=baseline_epochs,
                router_epochs=router_epochs,
                mala_steps=int(args.mala_steps),
                graph_k=int(args.graph_k),
            )
            executed_configs = {
                "baseline": asdict(
                    ContinuousDeemConfig(
                        epochs=baseline_epochs,
                        mala_steps=int(args.mala_steps),
                        anchor_tolerance=1e-12,
                        posterior_sd_min=1e-3,
                    )
                ),
                "variants": {
                    name: asdict(configuration)
                    for name, configuration in configurations.items()
                },
            }
            graph_audits.append(
                {
                    "world": world,
                    "replicate": int(replicate),
                    "donor_seed": int(donor_seed),
                    "held_seed": int(held_seed),
                    **graph_diagnostics,
                }
            )

            # Held features and truth come into existence only after all
            # donor-only fits (including A5's graph fit) have completed.
            held_raw, held_truth = generate_world(
                world,
                seed=held_seed,
                n=n_held,
                reveal_truth=True,
            )
            if held_truth is None:
                raise AssertionError("held truth was not generated")
            held = apply_standardization(held_raw, donor_mean, donor_scale)
            selected_rows, alias_error = evaluate_frozen_models(
                world=world,
                replicate=int(replicate),
                held=held,
                truth=held_truth,
                baseline=baseline,
                fitted=fitted,
                row_permutation_seed=permutation_seed,
            )
            rows.extend(selected_rows)
            alias_errors.append(alias_error)
            print(
                f"replicate={replicate} world={world} "
                f"b3_auc={selected_rows[0]['held_auroc']:.4f}",
                flush=True,
            )

    summary = summarize(rows)
    mechanisms = mechanism_summary(rows)
    output = {
        "schema": "deem_b3_pair_router_synthetic_v1",
        "status": "complete",
        "mode": "smoke" if args.smoke else "fixed_20plus_replicate_audit",
        "fit_contract": {
            "label_free": True,
            "donor_truth_returned": False,
            "held_generated_after_all_models_frozen": True,
            "held_rows_used_in_fit": 0,
            "held_rows_used_in_graph": 0,
            "graph_source": "donor_crossfit_true_family_loo_residuals",
            "baseline_frozen_during_router_fit": True,
        },
        "replicate_seeds": list(replicate_seeds),
        "worlds": list(worlds),
        "n_donor": int(n_donor),
        "n_held": int(n_held),
        "variants": list(VARIANTS),
        "runtime_scaling": {
            "production_router_epochs": 100,
            "executed_router_epochs": int(router_epochs),
            "production_mala_steps": 5,
            "executed_mala_steps": int(args.mala_steps),
            "actuation_and_penalty_weights_match_A0_A4_A5": True,
        },
        "executed_configs": executed_configs,
        "a0_exact_alias_max_abs": float(max(alias_errors, default=0.0)),
        "summary": summary,
        "mechanism_summary": mechanisms,
        "graph_audits": graph_audits,
        "rows": rows,
    }
    atomic_write_json(args.out, output)
    print(
        json.dumps(
            {
                "mode": output["mode"],
                "a0_exact_alias_max_abs": output["a0_exact_alias_max_abs"],
                "summary": summary,
                "mechanism_summary": mechanisms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
