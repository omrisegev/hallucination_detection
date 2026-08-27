#!/usr/bin/env python3
"""Mechanical, target-free tests for the B3-orthogonal IU-PGRD boost."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_iupgrd_boost import (  # noqa: E402
    _fit_iu_residual_state,
    deterministic_row_permutation,
    fit_b3_iupgrd_cell,
    permute_family_direction,
    pooled_cross_only_direction,
    score_b3_iupgrd_boost,
)
from spectral_utils.residual_graph_deem import canonical_sha256, sha256_file  # noqa: E402
from spectral_utils.specrage_views import FEATURE_TO_VIEW, VIEW_ORDER  # noqa: E402


def _synthetic_inputs(seed: int = 1729):
    rng = np.random.default_rng(seed)
    n = 96
    names = tuple(
        next(name for name, family in FEATURE_TO_VIEW.items() if family == target)
        for target in VIEW_ORDER
    )
    shared = rng.normal(size=n)
    X = np.column_stack(
        [
            (0.75 - 0.08 * index) * shared
            + (0.45 + 0.05 * index) * rng.normal(size=n)
            for index in range(len(names))
        ]
    )
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rows = tuple(f"synthetic::{index:04d}" for index in range(n))
    return X, names, rows


def _cell_with_orientation(sign: int, cell_id: str):
    X, names, rows = _synthetic_inputs()
    iu = _fit_iu_residual_state(X, names)
    score = expit(float(sign) * np.asarray(iu.baseline, dtype=float))
    # Exercise the exact-array path with a value that cannot be reproduced by
    # an unnecessary logit/expit round trip.
    score = score.copy()
    score[0] = np.nextafter(score[0], 1.0 if score[0] < 0.5 else 0.0)
    cell = fit_b3_iupgrd_cell(cell_id, X, names, score, rows, k=7)
    return cell, rows


def test_core_mechanics() -> None:
    positive, rows = _cell_with_orientation(+1, "positive")
    negative, _ = _cell_with_orientation(-1, "negative")
    assert positive.iu_orientation == 1
    assert negative.iu_orientation == -1
    for cell in (positive, negative):
        R = cell.residuals
        assert np.max(np.abs(R.mean(axis=0))) < 1e-10
        assert np.max(np.abs(R.std(axis=0) - 1.0)) < 1e-10
        assert np.max(np.abs(R.T @ cell.iu_score_aligned / len(R))) < 1e-10
        assert abs(np.trace(cell.moment.A) - len(cell.families)) < 1e-10

    direction, pool = pooled_cross_only_direction(
        [positive, negative], ["family_a", "family_b"]
    )
    assert pool["n_donor_cells"] == 2 and pool["n_donor_groups"] == 2
    assert np.all(np.isfinite(direction)) and np.linalg.norm(direction) > 0.0

    alias = score_b3_iupgrd_boost(
        positive,
        direction,
        trust_factor=0.0,
        project_against_b3=True,
        row_ids=rows,
    )
    assert np.array_equal(alias.score, positive.baseline_score)
    assert alias.score[0] == positive.baseline_score[0]

    full = score_b3_iupgrd_boost(
        positive,
        direction,
        trust_factor=1.0,
        project_against_b3=True,
        row_ids=rows,
    )
    assert abs(np.mean(full.projected_correction)) < 1e-10
    assert abs(
        np.dot(positive.baseline_z, full.projected_correction) / len(rows)
    ) < 1e-10
    assert abs(np.std(full.correction_z) - 1.0 / len(VIEW_ORDER)) < 1e-12

    permutation = [2, 5, 1, 4, 0, 3]
    permuted_direction = permute_family_direction(direction, permutation)
    assert np.array_equal(permuted_direction, direction[permutation])
    row_a = deterministic_row_permutation(rows, salt="mechanical-null")
    row_b = deterministic_row_permutation(rows, salt="mechanical-null")
    assert np.array_equal(row_a, row_b)
    assert not np.array_equal(row_a, np.arange(len(rows)))
    null_a = score_b3_iupgrd_boost(
        positive,
        direction,
        trust_factor=1.0,
        project_against_b3=True,
        row_ids=rows,
        row_permutation_salt="mechanical-null",
    )
    null_b = score_b3_iupgrd_boost(
        positive,
        direction,
        trust_factor=1.0,
        project_against_b3=True,
        row_ids=rows,
        row_permutation_salt="mechanical-null",
    )
    assert np.array_equal(null_a.score, null_b.score)
    assert not np.array_equal(null_a.score, full.score)

    # Explicit whole-family exclusion rule used by the runner.
    families = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}
    for held in sorted(set(families.values())):
        donors = [cell for cell, family in families.items() if family != held]
        assert donors and all(families[cell] != held for cell in donors)


def _verify_content_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.pop("content_sha256")
    assert canonical_sha256(value) == expected
    return value


def test_bound_runner_smoke() -> None:
    bundle_dir = ROOT / "local_cache/deem_b3_moe_v1/bundles"
    baseline_dir = ROOT / "local_cache/deem_b3_moe_v1/b3_frozen"
    required = (
        bundle_dir / "lapeigvals_gsm8k_llama3b.npz",
        bundle_dir / "math500_r1distill8b.npz",
        baseline_dir / "SCORE_FREEZE_MANIFEST.json",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("bound two-cell smoke inputs are missing")
    with tempfile.TemporaryDirectory(prefix="deem_b3_iupgrd_test_") as temporary:
        out = Path(temporary) / "fit"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_deem_b3_iupgrd_boost_v1.py"),
            "--bundle-dir",
            str(bundle_dir),
            "--baseline-dir",
            str(baseline_dir),
            "--out-dir",
            str(out),
            "--cells",
            "lapeigvals_gsm8k_llama3b,math500_r1distill8b",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout)["status"] == "complete"
        run = _verify_content_json(out / "RUN_DEFINITION.json")
        manifest = _verify_content_json(out / "FIT_ARTIFACT_MANIFEST.json")
        complete = _verify_content_json(out / "FIT_COMPLETE.json")
        assert not run["targets_accessed_during_fit"]
        assert not run["labels_module_imported"]
        assert all("residual_graph_deem_labels" not in path for path in run["source_dependencies"])
        assert sha256_file(out / complete["fit_artifact_manifest_path"]) == (
            complete["fit_artifact_manifest_sha256"]
        )
        for artifact in manifest["artifacts"]:
            path = out / artifact["path"]
            assert path.is_file() and sha256_file(path) == artifact["sha256"]
        for held_family, donors in run["donor_cells_by_held_dataset_family"].items():
            assert donors and all(
                run["dataset_family_by_cell"][cell] != held_family for cell in donors
            )
        for cell in run["cells"]:
            path = out / "scores/E0_B3_EXACT_ALIAS" / cell / "E0_B3_EXACT_ALIAS.npz"
            with np.load(path, allow_pickle=False) as arrays:
                assert np.array_equal(arrays["score"], arrays["baseline_score"])


def main() -> None:
    test_core_mechanics()
    test_bound_runner_smoke()
    print("deem_b3_iupgrd_boost mechanical tests: PASS")


if __name__ == "__main__":
    main()
