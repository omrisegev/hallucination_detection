#!/usr/bin/env python3
"""Acceptance tests for the paper-aligned report suite."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_aligned_benchmark_suite import (
    BAD_PRM_IDS, build_detection_rows, build_registry, render_protocol,
    score_refchecker_claims,
)
from spectral_utils.paper_benchmark_suite import (
    ProtocolSignature, assert_protocol_match, binary_metrics,
    fit_spectral_scores, forbid_cross_task_macro, read_csv, score_hash, write_csv,
)


def test_protocol_guard() -> None:
    left = ProtocolSignature("D", "M", "test", "answer", "auroc", "g")
    assert_protocol_match(left, left)
    try:
        assert_protocol_match(left, ProtocolSignature("D", "M", "test", "sentence", "auroc", "g"))
    except ValueError as error:
        assert "prediction_unit" in str(error)
    else:
        raise AssertionError("mismatched prediction units were accepted")


def test_no_cross_task_macro() -> None:
    forbid_cross_task_macro([{"scope": "protocol", "value": 1}])
    try:
        forbid_cross_task_macro([{"scope": "suite", "value": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("suite macro was accepted")
    try:
        forbid_cross_task_macro([{
            "protocol_id": "not-processbench", "subgroup": "four-subset macro",
        }])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-protocol four-subset macro was accepted")


def test_lambda_zero_and_direction() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=90)
    X = np.column_stack([
        latent + rng.normal(scale=0.2, size=90),
        0.8 * latent + rng.normal(scale=0.3, size=90),
        0.6 * latent + rng.normal(scale=0.4, size=90),
        0.4 * latent + rng.normal(scale=0.5, size=90),
        rng.normal(size=90),
    ])
    scores, diagnostics = fit_spectral_scores(
        X, feature_names=[f"f{i}" for i in range(5)], risk_anchor=X[:, 0],
        dufs_seeds=(3,), dufs_epochs=2, k=5, lambda_=0.1,
    )
    assert diagnostics["lambda_zero_exact"] is True
    assert set(scores) == {"deployed_upcr", "iu_pcr", "dufs_liu_pcr"}
    assert all(np.corrcoef(value, X[:, 0])[0, 1] >= -1e-12 for value in scores.values())


def test_row_order_invariance_and_hash() -> None:
    y = np.asarray([0, 1, 0, 1, 1, 0])
    s = np.asarray([0.1, 0.8, 0.2, 0.9, 0.7, 0.3])
    permutation = np.asarray([5, 2, 4, 0, 3, 1])
    assert binary_metrics(y, s) == binary_metrics(y[permutation], s[permutation])
    ids = ["a", "b", "c"]
    mapping = {"m": np.asarray([1.0, 2.0, 3.0])}
    assert score_hash(mapping, ids) == score_hash(mapping, ids)


def test_generated_registry_and_no_incomplete_critic() -> None:
    rows, competitors = build_detection_rows()
    registry = build_registry(rows, competitors)
    assert any(row["protocol_id"] == "internal-transfer" for row in registry)
    assert len([row for row in rows if row["protocol_id"] == "internal-transfer" and row["role"] == "ours"]) == 18
    assert len(BAD_PRM_IDS) == 3


def test_detection_rows_are_deduplicated_and_losses_visible() -> None:
    rows, _ = build_detection_rows()
    published = [row for row in rows if str(row.get("role", "")).startswith("published")]
    identities = [(row["protocol_id"], row.get("cell"), row.get("method_key"), row.get("value")) for row in published]
    assert len(identities) == len(set(identities))
    inside = [row for row in rows if row.get("cell") == "inside_coqa_llama7b"]
    assert any(row.get("role") == "ours_reference" for row in inside)
    assert any(str(row.get("role", "")).startswith("published") for row in inside)


def test_refchecker_full_key_and_no_setting_pool() -> None:
    rows, diagnostics = score_refchecker_claims(Path("."))
    assert diagnostics["n_claims"] == 10_733
    assert diagnostics["n_grouped_claims"] == 10_733
    assert diagnostics["settings_pooled"] is False
    assert {row["subgroup"] for row in rows} == {"zero_context", "noisy_context", "accurate_context"}


def test_report_value_comes_from_result_row() -> None:
    rows, competitors = build_detection_rows()
    registry = build_registry(rows, competitors)
    entry = next(item for item in registry if item["protocol_id"] == "internal-transfer")
    protocol_rows = [row for row in rows if row["protocol_id"] == "internal-transfer"]
    sentinel = 0.712345
    protocol_rows[0]["value"] = sentinel
    page = render_protocol(entry, protocol_rows, registry)
    assert f"{sentinel:.4f}" in page
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "rows.csv"
        write_csv(path, protocol_rows)
        assert f"{sentinel}" in path.read_text()


def test_generated_artifact_role_and_control_invariants() -> None:
    score_path = ROOT / "results/paper_aligned_benchmark_suite_2026_08_11/benchmark_scores.csv"
    if not score_path.exists():
        return
    rows = read_csv(score_path)
    critic = [row for row in rows if row.get("method_key") == "qwen72b_critic"]
    assert len(critic) == 5
    assert all(row.get("role") == "published_ceiling" for row in rows if row.get("method_key") == "qwen_prm")
    assert all(row.get("role") == "protocol_reproduction" for row in rows if row.get("method_key") == "gasp")
    refchecker = [row for row in rows if row.get("protocol_id") == "localization-refchecker-knowhalbench-claim"]
    assert {row["subgroup"] for row in refchecker} == {"zero_context", "noisy_context", "accurate_context"}
    external = [row for row in rows if row.get("subgroup") == "external-family: four-subset macro"]
    assert {row["method_key"] for row in external} == {"gl_liu_v1", "max_entropy"}


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"paper_aligned_benchmark_suite: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
