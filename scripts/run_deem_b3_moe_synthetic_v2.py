#!/usr/bin/env python3
"""Held-out synthetic audit for DEEM-like cross-family B3 extensions.

The latent-label fit remains fully unsupervised.  Natural/synthetic labels and
the planted specialist identity are used only on an independently generated
test split.  The switching world retains a persistent target signal so the
single binary EBM state is not forced to choose between an unrelated density
mode and the target factor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deem_b3_moe_v1 import load_config, variant_lookup  # noqa: E402
from spectral_utils.deem_b3_moe import (  # noqa: E402
    DeemMoEConfig,
    fit_deem_b3_moe,
    predict_deem_b3_moe,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
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
    "class_specialists",
    "exchangeable_null",
    "coherent_nuisance",
    "density_only",
)


def generate_world(world: str, *, seed: int, n: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    y = rng.integers(0, 2, size=n, dtype=np.int8)
    target = 2.0 * y - 1.0
    regime = rng.integers(0, 2, size=n, dtype=np.int8)
    regime_sign = 2.0 * regime - 1.0
    noise = rng.normal(size=(n, len(NAMES)))
    nuisance = rng.normal(size=n)

    if world == "switching_specialists":
        # Two conditional specialists, three persistent target-bearing
        # generalists, and one deliberately weak regime coordinate.
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
    elif world == "class_specialists":
        # Mirrors the paper's class-specific oracle experiment more closely:
        # the first family is an expert for class 0, the second for class 1.
        X = np.column_stack(
            [
                np.where(y == 0, -1.50 + 0.50 * noise[:, 0], 1.80 * noise[:, 0]),
                np.where(y == 1, +1.50 + 0.50 * noise[:, 1], 1.80 * noise[:, 1]),
                0.25 * target + noise[:, 2],
                0.25 * target + noise[:, 3],
                0.25 * target + noise[:, 4],
                0.25 * target + noise[:, 5],
            ]
        )
        active = y
    elif world == "exchangeable_null":
        X = 0.80 * target[:, None] + noise
        active = np.zeros(n, dtype=np.int8)
    elif world == "coherent_nuisance":
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
        active = np.zeros(n, dtype=np.int8)
    elif world == "density_only":
        mode = rng.choice(np.array([-2.0, 2.0]), size=n)
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
        active = np.zeros(n, dtype=np.int8)
    else:
        raise ValueError(f"unknown synthetic world {world}")
    return np.asarray(X, dtype=np.float64), y, active


def train_standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def specialist_impact(prediction: dict, active: np.ndarray, family_order):
    first = family_order.index("entropy_level")
    second = family_order.index("entropy_dynamics")
    if np.max(np.abs(prediction["family_state_delta"])) > 0.0:
        evidence = np.abs(prediction["family_state_delta"])
    else:
        evidence = np.abs(prediction["gates"] - 1.0)
    selected = np.where(active == 0, evidence[:, first], evidence[:, second])
    rejected = np.where(active == 0, evidence[:, second], evidence[:, first])
    return float(np.mean(selected - rejected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--worlds", default=",".join(WORLDS))
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    definitions = variant_lookup(config)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    worlds = [value.strip() for value in args.worlds.split(",") if value.strip()]
    if not variants or not set(variants).issubset(definitions):
        raise ValueError("invalid synthetic variant selection")
    if not worlds or not set(worlds).issubset(WORLDS):
        raise ValueError("invalid synthetic world selection")
    for variant in variants:
        if definitions[variant]["config"].get("train_experts", False):
            raise ValueError("v2 isolates frozen B3 experts; joint variants need B3_CONTINUE")

    rows = []
    for replicate in range(int(args.replicates)):
        for world_index, world in enumerate(worlds):
            train_raw, _, _ = generate_world(
                world, seed=50_000 + 211 * replicate + world_index, n=int(args.n_train)
            )
            test_raw, y_test, active_test = generate_world(
                world, seed=80_000 + 223 * replicate + world_index, n=int(args.n_test)
            )
            train, test = train_standardize(train_raw, test_raw)
            baseline = fit_continuous_deem(
                train,
                NAMES,
                seed=replicate,
                config=ContinuousDeemConfig(
                    epochs=int(args.baseline_epochs),
                    anchor_tolerance=1e-12,
                    posterior_sd_min=1e-3,
                ),
            )
            if not baseline.health["healthy"]:
                raise RuntimeError(f"unhealthy synthetic B3: {replicate}/{world}")
            baseline_prediction = predict_continuous_deem(baseline, test)
            baseline_auc = float(roc_auc_score(y_test, baseline_prediction["score"]))
            rows.append(
                {
                    "world": world,
                    "replicate": replicate,
                    "method": "B3",
                    "test_auroc": baseline_auc,
                    "delta_vs_b3": 0.0,
                    "specialist_impact": None,
                }
            )
            for variant in variants:
                variant_config = DeemMoEConfig(**definitions[variant]["config"])
                result = fit_deem_b3_moe(
                    train,
                    NAMES,
                    baseline.state,
                    seed=replicate,
                    config=variant_config,
                )
                if not result.health["healthy"]:
                    raise RuntimeError(f"unhealthy synthetic fit: {variant}/{replicate}/{world}")
                prediction = predict_deem_b3_moe(result, test)
                auc = float(roc_auc_score(y_test, prediction["score"]))
                impact = None
                if world in {"switching_specialists", "class_specialists"}:
                    impact = specialist_impact(prediction, active_test, result.family_order)
                rows.append(
                    {
                        "world": world,
                        "replicate": replicate,
                        "method": variant,
                        "test_auroc": auc,
                        "delta_vs_b3": auc - baseline_auc,
                        "specialist_impact": impact,
                        "healthy": True,
                        "reconstruction_max_abs": prediction["reconstruction_max_abs"],
                        "mala_acceptance": result.health["mala_acceptance_mean"],
                        "state_delta_mean_abs": float(
                            np.mean(np.abs(prediction["family_state_delta"]))
                        ),
                    }
                )
            print(f"replicate={replicate} world={world}", flush=True)

    by_key = defaultdict(list)
    for row in rows:
        if row["method"] != "B3":
            by_key[(row["world"], row["method"])].append(row)
    summary = []
    for (world, method), selected in sorted(by_key.items()):
        deltas = np.asarray([row["delta_vs_b3"] for row in selected], dtype=float)
        impacts = [row["specialist_impact"] for row in selected if row["specialist_impact"] is not None]
        summary.append(
            {
                "world": world,
                "method": method,
                "mean_delta_vs_b3": float(np.mean(deltas)),
                "median_delta_vs_b3": float(np.median(deltas)),
                "wins": int(np.sum(deltas > 0.0)),
                "losses": int(np.sum(deltas < 0.0)),
                "worst_delta": float(np.min(deltas)),
                "mean_specialist_impact": float(np.mean(impacts)) if impacts else None,
            }
        )

    diagonal = [
        variant
        for variant in variants
        if definitions[variant]["config"].get("router") == "multinomial_diagonal"
    ]
    lookup = {(row["world"], row["replicate"], row["method"]): row for row in rows}
    decisions = []
    required_wins = max(1, int(np.ceil(0.8 * int(args.replicates))))
    for variant in variants:
        by_world = {
            row["world"]: row for row in summary if row["method"] == variant
        }
        switching = by_world.get("switching_specialists")
        class_specialists = by_world.get("class_specialists")
        exchangeable = by_world.get("exchangeable_null")
        coherent = by_world.get("coherent_nuisance")
        density = by_world.get("density_only")
        diagonal_delta = None
        if diagonal and variant not in diagonal and switching is not None:
            differences = []
            for replicate in range(int(args.replicates)):
                candidate = lookup[("switching_specialists", replicate, variant)]["test_auroc"]
                control = lookup[("switching_specialists", replicate, diagonal[0])]["test_auroc"]
                differences.append(candidate - control)
            diagonal_delta = float(np.mean(differences))
        checks = {
            "switching_delta_at_least_0p005": bool(
                switching and switching["mean_delta_vs_b3"] >= 0.005
            ),
            "switching_wins_at_least_80pct": bool(
                switching and switching["wins"] >= required_wins
            ),
            "switching_specialist_impact_positive": bool(
                switching and switching["mean_specialist_impact"] > 0.0
            ),
            "class_specialist_delta_nonnegative": bool(
                class_specialists and class_specialists["mean_delta_vs_b3"] >= 0.0
            ),
            "exchangeable_abs_delta_at_most_0p0005": bool(
                exchangeable and abs(exchangeable["mean_delta_vs_b3"]) <= 0.0005
            ),
            "coherent_nuisance_safe": bool(
                coherent
                and coherent["mean_delta_vs_b3"] >= -0.0025
                and coherent["worst_delta"] >= -0.02
            ),
            "density_only_safe": bool(
                density
                and density["mean_delta_vs_b3"] >= -0.0025
                and density["worst_delta"] >= -0.02
            ),
            "beats_diagonal_on_switching": bool(
                diagonal_delta is None or diagonal_delta >= 0.0025
            ),
        }
        decisions.append(
            {
                "method": variant,
                "mean_switching_delta_vs_diagonal": diagonal_delta,
                "checks": checks,
                "pass": bool(all(checks.values())),
            }
        )

    output = {
        "schema": "deem_b3_moe_synthetic_v2",
        "status": "complete",
        "split": "independent_train_test",
        "replicates": int(args.replicates),
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "variants": variants,
        "worlds": worlds,
        "summary": summary,
        "decisions": decisions,
        "rows": rows,
    }
    atomic_write_json(args.out, output)
    print(json.dumps({"summary": summary, "decisions": decisions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
