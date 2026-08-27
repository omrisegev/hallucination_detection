#!/usr/bin/env python3
"""Post-hoc selector audit for the frozen Family-residual graph V3 bank."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.family_residual_graph_liu_fit import DEFAULT_OUT
from scripts.family_residual_graph_liu_report import (
    family_deltas,
    graph_health_registry,
    load_metrics,
)
from scripts.hard_filter_dufs_liu_benchmark import DEFAULT_BUNDLE, family


def main():
    configs = json.loads((DEFAULT_OUT / "CONFIG_INDEX.json").read_text())
    healthy, _ = graph_health_registry(DEFAULT_OUT)
    rows, auc, _, all_candidates = load_metrics(
        DEFAULT_OUT, DEFAULT_BUNDLE, configs, healthy
    )
    candidates = tuple(
        key for key in all_candidates if configs[key]["topology"] == "union"
    )
    cells = [row["cell"] for row in rows]
    families = sorted({family(cell) for cell in cells})
    deltas = family_deltas(cells, candidates, auc)

    def stats(key, groups):
        values = np.asarray([deltas[key][group] for group in groups])
        return {
            "mean": float(np.mean(values)),
            "se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "worst": float(np.min(values)),
        }

    rules = {
        "max_mean": lambda row: row["mean"],
        "max_mean_minus_half_se": lambda row: row["mean"] - 0.5 * row["se"],
        "max_mean_minus_se": lambda row: row["mean"] - row["se"],
        "max_worst": lambda row: row["worst"],
    }
    output = {"families": families, "rules": {}, "top_global": []}
    for name, objective in rules.items():
        held_values = []
        selections = []
        for held in families:
            training = [group for group in families if group != held]
            selected = max(
                candidates,
                key=lambda key: (objective(stats(key, training)), key),
            )
            held_values.append(deltas[selected][held])
            selections.append(selected)
        final = max(
            candidates,
            key=lambda key: (objective(stats(key, families)), key),
        )
        output["rules"][name] = {
            "nested_delta_pp": 100 * float(np.mean(held_values)),
            "held_values_pp": [100 * float(value) for value in held_values],
            "positive_families": int(np.sum(np.asarray(held_values) > 0)),
            "unique_selections": len(set(selections)),
            "selections": selections,
            "final_key": final,
            "final_config": configs[final],
            "final_mean_pp": 100 * stats(final, families)["mean"],
            "final_se_pp": 100 * stats(final, families)["se"],
            "final_worst_pp": 100 * stats(final, families)["worst"],
        }
    ranked = sorted(
        candidates,
        key=lambda key: (stats(key, families)["mean"], key),
        reverse=True,
    )
    for key in ranked[:25]:
        row = stats(key, families)
        output["top_global"].append({
            "key": key,
            "config": configs[key],
            "mean_pp": 100 * row["mean"],
            "se_pp": 100 * row["se"],
            "worst_pp": 100 * row["worst"],
        })
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
