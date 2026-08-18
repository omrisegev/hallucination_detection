#!/usr/bin/env python3
"""Focused contract tests for the structured causal subset search."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.unified_causal_evaluation import AtlasSamples  # noqa: E402
from spectral_utils.unified_causal_iu import BASE_NAMES, all_feature_names  # noqa: E402
from spectral_utils.unified_causal_subset_search import (  # noqa: E402
    TRANSFORM_FAMILIES,
    base_mask_rosters,
    blended_multipliers,
    rank_against_control,
    structured_rosters,
    supervised_relevance,
)
from scripts.run_unified_causal_subset_search_v1 import _aggregate_payloads  # noqa: E402


def test_stage_a_counts_and_schema() -> None:
    rosters = structured_rosters("a")
    assert "family6_level" not in rosters
    assert {name: len(roster) for name, roster in rosters.items()} == {
        "core5_level": 5,
        "raw9_level": 9,
        "raw9_fastslow": 27,
        "raw9_sustained": 27,
        "all37_level": 37,
        "all37_multiscale_sustained": 407,
        "all37_window_moments": 444,
        "all37_change": 185,
        "all37_no_bocpd": 962,
        "all37_full": 1036,
    }
    canonical = set(all_feature_names())
    for roster in rosters.values():
        assert len(roster) == len(set(roster))
        assert set(roster) <= canonical
        assert all("trace_length" not in name for name in roster)


def test_stage_b_crosses_one_frozen_transform_family() -> None:
    rosters = base_mask_rosters(TRANSFORM_FAMILIES["sustained"])
    assert {name: len(roster) for name, roster in rosters.items()} == {
        "core5": 15,
        "raw9": 27,
        "joint15": 45,
        "joint18": 54,
        "broad28": 84,
        "all37": 111,
    }
    assert all(
        name.rsplit("::", 1)[-1] in set(TRANSFORM_FAMILIES["sustained"])
        for roster in rosters.values()
        for name in roster
    )


def test_stage_c_targeted_unions() -> None:
    rosters = structured_rosters("c")
    assert {name: len(roster) for name, roster in rosters.items()} == {
        "raw9_level": 9,
        "joint18_level": 18,
        "broad28_level": 28,
        "raw9_fastslow": 27,
        "raw9_level_fastslow": 36,
        "joint18_level_plus_raw9_fastslow": 45,
        "broad28_level_plus_raw9_fastslow": 55,
        "raw9_level_sustained": 36,
        "joint18_level_plus_raw9_sustained": 45,
        "broad28_level_plus_raw9_sustained": 55,
    }
    assert all(len(roster) == len(set(roster)) for roster in rosters.values())


def test_stage_d_winner_attribution() -> None:
    rosters = structured_rosters("d")
    assert len(rosters) == 18
    assert len(rosters["raw9_level"]) == 9
    assert len(rosters["raw9_level_ewma16"]) == 18
    assert len(rosters["raw9_level_ewma16_area"]) == 27
    assert len(rosters["raw9_sustained"]) == 27
    assert len(rosters["raw9_level_sustained"]) == 36
    drops = {name: roster for name, roster in rosters.items() if name.startswith("winner_drop_")}
    assert len(drops) == 9
    assert {len(roster) for roster in drops.values()} == {32}


def test_stage_e_compact_finalists() -> None:
    rosters = structured_rosters("e")
    assert {name: len(roster) for name, roster in rosters.items()} == {
        "raw9_full36": 36,
        "drop_margin_full32": 32,
        "drop_spilled_full32": 32,
        "base7_full28": 28,
        "raw9_no_area27": 27,
        "drop_margin_no_area24": 24,
        "drop_spilled_no_area24": 24,
        "base7_no_area21": 21,
        "base6_no_entropy18": 18,
        "base6_no_logsumexp18": 18,
        "base6_no_top1_18": 18,
        "base5_no_entropy_top1_15": 15,
    }


def _samples(target: str, names: tuple[str, ...]) -> AtlasSamples:
    # f0 has a strong class shift in both families, f1 is constant, and f2 is
    # weaker.  The construction also exercises equal target/family averaging.
    y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    family = np.asarray(["a"] * 4 + ["b"] * 4)
    X = np.column_stack([
        np.asarray([0, 0.1, 3, 3.2, 0.2, 0.1, 2.8, 3.0]),
        np.ones(8),
        np.asarray([0, 0.2, 0.6, 0.7, 0.1, 0.0, 0.5, 0.8]),
    ])
    return AtlasSamples(
        target=target,
        feature_names=names,
        X=X,
        y=y,
        groups=np.asarray([f"g{i}" for i in range(8)]),
        families=family,
        models=np.asarray(["m"] * 8),
        budgets=np.asarray(["final"] * 8),
        context=np.empty((8, 0)),
        context_names=(),
        baseline=np.empty((8, 0)),
        baseline_names=(),
    )


def test_supervised_relevance_and_blend() -> None:
    names = tuple(all_feature_names()[:3])
    relevance = supervised_relevance(
        [_samples(target, names) for target in ("global", "localization", "early")],
        names,
    )
    assert relevance.shape == (3,)
    assert relevance[0] > relevance[1]
    assert relevance[2] > relevance[1]
    assert np.array_equal(blended_multipliers(relevance, 0.0), np.ones(3))
    assert np.allclose(blended_multipliers(relevance, 1.0), relevance)


def test_pareto_maximin_ranking() -> None:
    records = [
        {"candidate": "full", "global": 0.60, "localization": 0.30, "early": 0.60, "n_features": 1036},
        {"candidate": "small", "global": 0.61, "localization": 0.31, "early": 0.61, "n_features": 9},
        {"candidate": "tradeoff", "global": 0.63, "localization": 0.291, "early": 0.62, "n_features": 27},
        {"candidate": "bad", "global": 0.55, "localization": 0.20, "early": 0.50, "n_features": 3},
    ]
    ranked = rank_against_control(records, "full")
    assert ranked[0]["candidate"] == "small"
    lookup = {row["candidate"]: row for row in ranked}
    assert lookup["small"]["pareto"]
    assert lookup["small"]["survives_noninferiority"]
    assert not lookup["bad"]["survives_noninferiority"]
    assert lookup["tradeoff"]["survives_noninferiority"]
    assert np.isclose(lookup["small"]["max_oracle_regret"], 2.0)
    assert len(BASE_NAMES) == 37


def test_outer_metrics_are_averaged_within_fold() -> None:
    """Different fold score scales must never be pooled into one AUROC."""

    configs = {
        name: {
            "candidate": name,
            "roster": name,
            "fusion": "ordinary",
            "lambda": 0.0,
            "reweight_alpha": 0.0,
            "input_features": 3,
            "retained_features": 3,
        }
        for name in ("control", "candidate")
    }

    def records(fold: int, candidate: str):
        return [
            {
                "unit": f"{fold}-{label}",
                "source_group": f"f::{fold}-{label}",
                "family": "f",
                "model": "m",
                "wrong": label,
                "target_step": -1 if label == 0 else 0,
                "global_score": float(label),
                "prediction": -1 if label == 0 else 0,
                "risk_at_64": float(label),
                "risk_at_128": float(label),
            }
            for label in (0, 1)
        ]

    payloads = []
    for fold, global_value in ((0, 0.9), (1, 0.5)):
        payloads.append({
            "repeat": 0,
            "fold": fold,
            "variant_configs": configs,
            "results": {
                name: {
                    "metrics": {"macro": {
                        "global": global_value,
                        "localization": 0.4 + 0.2 * fold,
                        "early": 0.8 - 0.2 * fold,
                    }},
                    "records": records(fold, name),
                }
                for name in configs
            },
        })
    aggregate = _aggregate_payloads(payloads, "control", ("f",))
    row = next(
        item for item in aggregate["repeat_metrics"]
        if item["candidate"] == "candidate"
    )
    assert np.isclose(row["global"], 0.7)
    assert np.isclose(row["localization"], 0.5)
    assert np.isclose(row["early"], 0.7)


def main() -> None:
    tests = [
        test_stage_a_counts_and_schema,
        test_stage_b_crosses_one_frozen_transform_family,
        test_stage_c_targeted_unions,
        test_stage_d_winner_attribution,
        test_stage_e_compact_finalists,
        test_supervised_relevance_and_blend,
        test_pareto_maximin_ranking,
        test_outer_metrics_are_averaged_within_fold,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all ({len(tests)} tests)")


if __name__ == "__main__":
    main()
