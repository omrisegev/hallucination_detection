#!/usr/bin/env python3
"""Synthetic switching-specialist and coherent-nuisance audit for DEEM-B3 MoE."""

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
from spectral_utils.deem_b3_moe import DeemMoEConfig, fit_deem_b3_moe  # noqa: E402
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    atomic_write_json,
    fit_continuous_deem,
)


NAMES = (
    "epr",
    "spectral_entropy",
    "epr_spilled",
    "epr_energy",
    "mean_top1_logprob",
    "trace_length",
)
WORLDS = ("switching_specialists", "no_switch", "coherent_nuisance", "density_only")


def zscore(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return centered / scale


def make_world(world: str, *, seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    y = rng.integers(0, 2, size=n, dtype=np.int8)
    target = 2.0 * y - 1.0
    regime = rng.integers(0, 2, size=n, dtype=np.int8)
    regime_sign = 2.0 * regime - 1.0
    nuisance = rng.normal(size=n)
    noise = rng.normal(size=(n, len(NAMES)))

    if world == "switching_specialists":
        X = np.column_stack(
            [
                np.where(regime == 0, 1.7 * target + 0.55 * noise[:, 0],
                         1.45 * nuisance + 0.55 * noise[:, 0]),
                np.where(regime == 1, 1.7 * target + 0.55 * noise[:, 1],
                         1.45 * nuisance + 0.55 * noise[:, 1]),
                1.25 * regime_sign + 0.55 * noise[:, 2],
                -1.10 * regime_sign + 0.55 * noise[:, 3],
                0.30 * target + 0.90 * noise[:, 4],
                0.85 * regime_sign + 0.70 * noise[:, 5],
            ]
        )
    elif world == "no_switch":
        X = np.column_stack(
            [
                1.20 * target + 0.75 * noise[:, 0],
                1.20 * target + 0.75 * noise[:, 1],
                0.90 * target + 0.90 * noise[:, 2],
                0.90 * target + 0.90 * noise[:, 3],
                0.70 * target + noise[:, 4],
                0.60 * target + noise[:, 5],
            ]
        )
    elif world == "coherent_nuisance":
        X = np.column_stack(
            [
                0.85 * target + 0.80 * noise[:, 0],
                0.85 * target + 0.80 * noise[:, 1],
                1.80 * nuisance + 0.35 * noise[:, 2],
                1.75 * nuisance + 0.35 * noise[:, 3],
                1.70 * nuisance + 0.35 * noise[:, 4],
                1.60 * nuisance + 0.40 * noise[:, 5],
            ]
        )
    elif world == "density_only":
        # Strong, structured density modes are statistically unrelated to y.
        mode = rng.choice(np.array([-2.0, 2.0]), size=n)
        X = np.column_stack(
            [
                0.20 * target + noise[:, 0],
                0.20 * target + noise[:, 1],
                mode + 0.25 * noise[:, 2],
                -mode + 0.25 * noise[:, 3],
                0.75 * mode + 0.35 * noise[:, 4],
                -0.65 * mode + 0.40 * noise[:, 5],
            ]
        )
    else:
        raise ValueError(f"unknown synthetic world {world}")
    return zscore(np.asarray(X, dtype=np.float64)), y, regime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deem_b3_moe_v1.json")
    parser.add_argument("--variants", required=True, help="comma-separated variant IDs")
    parser.add_argument("--worlds", default=",".join(WORLDS))
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--n", type=int, default=800)
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

    rows = []
    for replicate in range(int(args.replicates)):
        for world_index, world in enumerate(worlds):
            seed = 30_000 + 101 * replicate + world_index
            X, y, regime = make_world(world, seed=seed, n=int(args.n))
            baseline = fit_continuous_deem(
                X,
                NAMES,
                seed=replicate,
                config=ContinuousDeemConfig(
                    epochs=int(args.baseline_epochs),
                    anchor_tolerance=1e-12,
                    posterior_sd_min=0.0,
                ),
            )
            baseline_auc = float(roc_auc_score(y, baseline.score))
            rows.append(
                {
                    "world": world,
                    "replicate": replicate,
                    "method": "B3",
                    "auroc": baseline_auc,
                    "delta_vs_b3": 0.0,
                    "gate_active_minus_inactive": None,
                    "gate_mean_abs_deviation_from_one": 0.0,
                    "mala_acceptance": baseline.health["mala_acceptance_mean"],
                }
            )
            for variant in variants:
                variant_config = DeemMoEConfig(**definitions[variant]["config"])
                result = fit_deem_b3_moe(
                    X,
                    NAMES,
                    baseline.state,
                    seed=replicate,
                    config=variant_config,
                )
                auc = float(roc_auc_score(y, result.score))
                gate_alignment = None
                if world == "switching_specialists":
                    entropy_index = result.family_order.index("entropy_level")
                    dynamics_index = result.family_order.index("entropy_dynamics")
                    active = np.where(
                        regime == 0,
                        result.gates[:, entropy_index],
                        result.gates[:, dynamics_index],
                    )
                    inactive = np.where(
                        regime == 0,
                        result.gates[:, dynamics_index],
                        result.gates[:, entropy_index],
                    )
                    gate_alignment = float(np.mean(active - inactive))
                rows.append(
                    {
                        "world": world,
                        "replicate": replicate,
                        "method": variant,
                        "auroc": auc,
                        "delta_vs_b3": auc - baseline_auc,
                        "gate_active_minus_inactive": gate_alignment,
                        "gate_mean_abs_deviation_from_one": result.diagnostics[
                            "gate_mean_abs_deviation_from_one"
                        ],
                        "mala_acceptance": result.health["mala_acceptance_mean"],
                        "reconstruction_max_abs": result.health[
                            "contribution_reconstruction_max_abs"
                        ],
                    }
                )
            print(f"replicate={replicate} world={world}", flush=True)

    summary = []
    by_key = defaultdict(list)
    for row in rows:
        if row["method"] != "B3":
            by_key[(row["world"], row["method"])].append(row)
    for (world, method), selected in sorted(by_key.items()):
        deltas = np.asarray([row["delta_vs_b3"] for row in selected], dtype=float)
        alignments = [
            row["gate_active_minus_inactive"]
            for row in selected
            if row["gate_active_minus_inactive"] is not None
        ]
        summary.append(
            {
                "world": world,
                "method": method,
                "mean_delta_vs_b3": float(np.mean(deltas)),
                "median_delta_vs_b3": float(np.median(deltas)),
                "wins": int(np.sum(deltas > 0.0)),
                "losses": int(np.sum(deltas < 0.0)),
                "worst_delta": float(np.min(deltas)),
                "mean_active_minus_inactive_gate": (
                    float(np.mean(alignments)) if alignments else None
                ),
                "mean_gate_deviation": float(
                    np.mean([row["gate_mean_abs_deviation_from_one"] for row in selected])
                ),
            }
        )
    decision = {
        "schema": "deem_b3_moe_synthetic_v1",
        "status": "complete",
        "replicates": int(args.replicates),
        "n": int(args.n),
        "variants": variants,
        "worlds": worlds,
        "summary": summary,
        "rows": rows,
        "gates": {
            "switching_mean_delta": 0.005,
            "switching_min_wins": max(1, int(np.ceil(0.8 * args.replicates))),
            "switching_gate_alignment_positive": True,
            "no_switch_abs_mean_delta_max": 0.005,
            "coherent_nuisance_mean_delta_min": -0.005,
            "coherent_nuisance_worst_delta_min": -0.02,
        },
    }
    atomic_write_json(args.out, decision)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
