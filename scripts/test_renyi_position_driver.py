"""Synthetic exclusion, serialization and calibration tests for the v1 driver."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_renyi_position_temporal_fusion as driver
from spectral_utils import renyi_position_fusion as model


def _statistics(seed, uid):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(96 + seed % 7, len(model.FEATURE_NAMES)))
    z = (z - z.mean(0)) / z.std(0)
    return model.regional_statistics(z, uid)


def run():
    checks = []

    metadata = {}
    statistics = {}
    records = []
    fold = {}
    index = 0
    for cell in ("pb_demo_q4", "prmbench_qwen3_8b"):
        for held in range(5):
            for repeat in range(2):
                uid = f"{cell}-{held}-{repeat}"
                group = f"{cell}-group-{held}-{repeat}"
                metadata[index] = dict(uid=uid, cell=cell, group_id=group, fold=held)
                statistics[index] = _statistics(index + 1, uid)
                records.append(dict(uid=uid, cell=cell, group_id=group, tokens=20 + index, steps=2))
                fold[index] = held
                index += 1

    weights = driver.group_weights(metadata, [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(sum(weights.values()), 1.0, atol=1e-15)
    assert all(value == 1 / 6 for value in weights.values())
    checks.append("equal source-group then answer weights")

    fitted, info = driver.fit_prior(statistics, metadata, "pb_demo_q4", (0,))
    assert not set(info["training_groups"]).intersection(info["excluded_groups"])
    assert info["training_answers"] == 8 and info["fit_api_accepts_no_labels"]
    checks.append("held-fold and held-group exclusion")

    flat = driver.flatten_model(fitted)
    replay = driver.unflatten_model(flat)
    assert set(replay) == set(fitted)
    for name, value in driver.flatten_model(replay).items():
        np.testing.assert_array_equal(value, flat[name])
    checks.append("external model serialization round trip")

    selected = list(range(len(records)))
    assert len(driver.excluded_sets(metadata, selected, "pb_demo_q4")) == 5
    assert len(driver.excluded_sets(metadata, selected, "prmbench_qwen3_8b")) == 15
    checks.append("outer and nested exclusion-key roster")

    poisoned = [{**row, "labels": [1, 0], "target": 1} for row in records]
    rebuilt = driver.training_metadata(poisoned, fold)
    assert rebuilt == metadata
    assert {"label", "labels", "target"}.isdisjoint(inspect.signature(driver.fit_prior).parameters)
    checks.append("fit API and stripped metadata label firewall")

    calibration_records = [
        dict(uid=f"u{held}", cell="prmbench_qwen3_8b", group_id=f"g{held}")
        for held in range(5)
    ]
    calibration_fold = {held: held for held in range(5)}
    joined = {"offsets": np.arange(0, 11, 2)}
    scores = {
        "external_iu_position": np.zeros(10),
        "view__ve1": np.arange(10, dtype=float),
    }
    predictions = {}
    for answer in range(5):
        row = {}
        for held in range(5):
            if held != answer:
                row["external_iu_position__inner_for_" + str(held)] = np.full(2, 10 * held + answer)
        predictions[answer] = row
    thresholds, coverage = driver.build_calibration(
        predictions, scores, calibration_records, joined, calibration_fold
    )
    expected = np.quantile(np.concatenate([np.full(2, answer) for answer in range(1, 5)]), 0.8)
    np.testing.assert_allclose(thresholds["external_iu_position"]["0"], expected)
    assert len(coverage) == 10 and all(row["answers"] == 4 for row in coverage)
    checks.append("nested PRMScore calibration uses pair-excluded predictions")

    selection_records = []
    for cell_number in range(9):
        for answer in range(20):
            selection_records.append(
                dict(
                    uid=f"cell{cell_number}-{answer}",
                    cell=f"cell{cell_number}",
                    tokens=answer + 5,
                )
            )
    smoke = driver.smoke_selection(selection_records)
    assert len(smoke) == 27 and len(set(smoke)) == 27
    checks.append("deterministic three-length-per-cell smoke roster")

    primary = set(model.PRIMARY)
    pairs = driver.contrast_pairs()
    assert primary.issubset(pairs)
    np.testing.assert_allclose(driver.PRIMARY_CI, 1.0 - 0.05 / 3.0)
    checks.append("frozen 98.333-percent primary interval and control roster")

    return dict(status="PASS", checks=checks, count=len(checks))


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
