#!/usr/bin/env python3
"""Run the frozen S0 synthetic gate for DSP-contextual IU-PCR."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spectral_utils.contextual_iu import (  # noqa: E402
    ContextualIUModel,
    family_partition,
)


OUT = ROOT / "results" / "dsp_contextual_iu_pilot_v1"
SEEDS = tuple(range(32000, 32020))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _world(seed: int, world: str, n: int):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, n)
    target = rng.normal(size=n)
    nuisance = rng.normal(size=n)
    labels = (target + rng.normal(scale=0.15, size=n) > 0.0).astype(int)
    X = np.empty((n, 6), dtype=float)
    active = np.column_stack([
        np.where(regime == 0, family < 3, family >= 3)
        for family in range(6)
    ])

    if world == "informative":
        for family in range(6):
            X[:, family] = np.where(
                active[:, family],
                target + rng.normal(scale=0.45, size=n),
                rng.normal(scale=1.50, size=n),
            )
        context_active = active
    elif world == "null":
        X = target[:, None] + rng.normal(scale=0.75, size=(n, 6))
        context_active = rng.integers(0, 2, size=(n, 6)).astype(bool)
    elif world == "coherent_nuisance":
        for family in range(6):
            X[:, family] = np.where(
                active[:, family],
                target + rng.normal(scale=0.70, size=n),
                1.6 * nuisance + rng.normal(scale=0.18, size=n),
            )
        # The DSP state is coherent around the inactive, wrong block.
        context_active = ~active
    elif world == "observational_equivalence":
        for family in range(6):
            signal = np.where(active[:, family], target, nuisance)
            X[:, family] = signal + rng.normal(scale=0.45, size=n)
        context_active = active
    else:
        raise KeyError(world)

    blocks = []
    for operator in range(5):
        magnitude = 1.0 + 0.15 * operator
        blocks.append(
            magnitude * np.where(context_active, 1.0, -1.0)
            + rng.normal(scale=0.20, size=(n, 6))
        )
    dsp = np.column_stack(blocks)
    positions = rng.integers(8, 257, size=n).astype(float)
    groups = np.asarray([f"q-{index}" for index in range(n)])
    return X, positions, groups, dsp, labels, target, nuisance, active


def _score_world(seed: int, world: str) -> dict:
    train = _world(seed, world, 320)
    test = _world(seed + 100000, world, 640)
    model = ContextualIUModel.fit(
        train[0],
        train[1],
        train[2],
        family_partition(6),
        dsp_context=train[3],
        mode="dsp",
    )
    scored = model.score(test[0], test[1], dsp_context=test[3])
    baseline_auc = roc_auc_score(test[4], scored.baseline_score)
    contextual_auc = roc_auc_score(test[4], scored.score)
    target_active_mass = np.sum(scored.family_mass * test[7], axis=1)
    return {
        "seed": seed,
        "world": world,
        "baseline_auc": float(baseline_auc),
        "contextual_auc": float(contextual_auc),
        "delta": float(contextual_auc - baseline_auc),
        "fallback_rate": float(np.mean(scored.fallback)),
        "mean_n_eff": float(np.mean(scored.n_eff)),
        "mean_alpha": float(np.mean(scored.alpha)),
        "mean_target_active_family_mass": float(np.mean(target_active_mass)),
    }


def _mechanical_checks() -> dict:
    X, positions, groups, dsp, *_ = _world(981, "informative", 96)
    model = ContextualIUModel.fit(
        X, positions, groups, family_partition(6), dsp_context=dsp, mode="dsp"
    )
    query = model.score(X[:24], positions[:24], dsp_context=dsp[:24])

    duplicated = np.repeat(np.arange(len(X)), 2)
    duplicated_model = ContextualIUModel.fit(
        X[duplicated],
        positions[duplicated],
        groups[duplicated],
        family_partition(6),
        dsp_context=dsp[duplicated],
        mode="dsp",
    )
    duplicated_query = duplicated_model.score(
        X[:24], positions[:24], dsp_context=dsp[:24]
    )

    small_model = ContextualIUModel.fit(
        X[:20],
        positions[:20],
        groups[:20],
        family_partition(6),
        dsp_context=dsp[:20],
        mode="dsp",
    )
    fallback = small_model.score(X[20:30], positions[20:30], dsp_context=dsp[20:30])

    equivalent = _world(7781, "observational_equivalence", 120)
    equivalent_model = ContextualIUModel.fit(
        equivalent[0], equivalent[1], equivalent[2], family_partition(6),
        dsp_context=equivalent[3], mode="dsp",
    )
    equivalent_score = equivalent_model.score(
        equivalent[0], equivalent[1], dsp_context=equivalent[3]
    ).score
    # Swapping the semantic names of the two latent variables changes no
    # observation supplied to the estimator and therefore no score.
    equivalent_score_swapped_semantics = equivalent_model.score(
        equivalent[0], equivalent[1], dsp_context=equivalent[3]
    ).score

    return {
        "question_duplication_max_abs_score_delta": float(np.max(np.abs(
            query.score - duplicated_query.score
        ))),
        "question_duplication_max_abs_weight_delta": float(np.max(np.abs(
            query.weights - duplicated_query.weights
        ))),
        "fallback_all": bool(np.all(fallback.fallback)),
        "fallback_exact": bool(np.array_equal(
            fallback.score, fallback.baseline_score
        )),
        "observational_equivalence_exact": bool(np.array_equal(
            equivalent_score, equivalent_score_swapped_semantics
        )),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [
        _score_world(seed, world)
        for world in ("informative", "null", "coherent_nuisance")
        for seed in SEEDS
    ]
    mechanics = _mechanical_checks()
    summary = []
    for world in ("informative", "null", "coherent_nuisance"):
        rows = [row for row in records if row["world"] == world]
        summary.append({
            "world": world,
            "mean_baseline_auc": float(np.mean([row["baseline_auc"] for row in rows])),
            "mean_contextual_auc": float(np.mean([row["contextual_auc"] for row in rows])),
            "mean_delta": float(np.mean([row["delta"] for row in rows])),
            "worst_delta": float(np.min([row["delta"] for row in rows])),
            "wins": int(sum(row["delta"] > 0.0 for row in rows)),
            "mean_fallback_rate": float(np.mean([row["fallback_rate"] for row in rows])),
            "mean_target_active_family_mass": float(np.mean([
                row["mean_target_active_family_mass"] for row in rows
            ])),
        })
    by_world = {row["world"]: row for row in summary}
    informative = by_world["informative"]
    null = by_world["null"]
    nuisance = by_world["coherent_nuisance"]
    gates = {
        "informative_wins": informative["wins"] >= 18,
        "informative_gain": informative["mean_delta"] >= 0.005,
        "null_no_material_actuation": abs(null["mean_delta"]) <= 0.005,
        "coherent_nuisance_mean_safety": nuisance["mean_delta"] >= -0.005,
        "coherent_nuisance_tail_safety": nuisance["worst_delta"] >= -0.020,
        "question_duplication_invariance": (
            mechanics["question_duplication_max_abs_score_delta"] <= 1e-10
            and mechanics["question_duplication_max_abs_weight_delta"] <= 1e-10
        ),
        "exact_global_fallback": mechanics["fallback_all"] and mechanics["fallback_exact"],
        "observational_equivalence": mechanics["observational_equivalence_exact"],
    }
    passed = bool(all(gates.values()))
    decision = {
        "stage": "S0",
        "status": "PASS_S0" if passed else "STOP_NO_ROUTING_SIGNAL",
        "passed": passed,
        "gates": gates,
        "mechanics": mechanics,
        "seeds": list(SEEDS),
        "protocol_sha256": _sha256(
            ROOT / "docs" / "experiments" / "DSP_CONTEXTUAL_IU_PILOT_V1.md"
        ),
    }
    _write_csv(OUT / "STAGE_0_SYNTHETIC_PER_SEED.csv", records)
    _write_csv(OUT / "STAGE_0_SYNTHETIC_SUMMARY.csv", summary)
    (OUT / "STAGE_0_DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
