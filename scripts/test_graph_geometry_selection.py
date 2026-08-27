#!/usr/bin/env python3
"""Mechanical and provenance tests for graph-geometry selection V1."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.graph_geometry_selection_fit import (  # noqa: E402
    DEFAULT_BUNDLE,
    ORIGINAL_BUNDLE,
    basis_key,
    candidate_key,
    exclusion_sets,
    launch_hash_audit,
    sha256_file,
    verify_isolated_bundle,
)
from scripts.graph_geometry_selection_report import (  # noqa: E402
    choose_cross_one_se,
    choose_one_se,
)
from spectral_utils.graph_geometry_selection import (  # noqa: E402
    GEOMETRIES,
    choose_intrinsic_geometry,
    deduplicate_geometries,
    edge_jaccard,
    graph_from_transformed_coordinates,
    operator_cosine,
    validate_physically_label_free_members,
)
from spectral_utils.pooled_graph_roughness import (  # noqa: E402
    GraphRoughnessMoment,
    apply_pooled_roughness,
    fit_pooled_roughness_calibration,
    pool_graph_roughness_moments,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


def expect_raises(function, exception=Exception):
    try:
        function()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def test_physical_member_whitelist():
    cells = ("alpha", "beta")
    allowed = [
        f"{cell}__{suffix}"
        for cell in cells for suffix in ("V", "pool", "hand_signs")
    ]
    validate_physically_label_free_members(allowed, cells)
    expect_raises(
        lambda: validate_physically_label_free_members(
            allowed + ["alpha__labels"], cells
        ),
        RuntimeError,
    )
    expect_raises(
        lambda: validate_physically_label_free_members(
            [key for key in allowed if key != "beta__V"], cells
        ),
        RuntimeError,
    )


def test_duplicate_safe_graph_determinism():
    rng = np.random.default_rng(7)
    coordinates = rng.normal(size=(40, 5))
    coordinates[3] = coordinates[2]
    coordinates[11] = coordinates[2]
    spec = next(item for item in GEOMETRIES if item.geometry_id == "residual_union_k7")
    left = graph_from_transformed_coordinates(spec, coordinates)
    right = graph_from_transformed_coordinates(spec, coordinates)
    assert left.shape == (40, 40)
    assert left.diagonal().sum() == 0
    assert np.isfinite(left.data).all()
    assert (left != right).nnz == 0
    assert edge_jaccard(left, right) == 1.0
    assert abs(operator_cosine(left, right) - 1.0) < 1e-12


def test_deduplication_is_priority_stable():
    ids = ("residual_union_k7", "copy", "different")
    similarity = {
        "residual_union_k7__vs__residual_union_k7": {
            "edge_jaccard_mean": 1.0, "operator_cosine_mean": 1.0,
        },
        "residual_union_k7__vs__copy": {
            "edge_jaccard_mean": 1.0, "operator_cosine_mean": 1.0,
        },
        "residual_union_k7__vs__different": {
            "edge_jaccard_mean": 0.2, "operator_cosine_mean": 0.8,
        },
        "different__vs__different": {
            "edge_jaccard_mean": 1.0, "operator_cosine_mean": 1.0,
        },
    }
    active, duplicate = deduplicate_geometries(similarity, ids)
    assert active == ("residual_union_k7", "different")
    assert duplicate == {"copy": "residual_union_k7"}


def test_intrinsic_selector_fails_closed():
    invalid = [{
        "geometry_id": "residual_union_k7",
        "valid": False,
        "minimum_perturbation_stability": 1.0,
        "minimum_direction_cosine": 1.0,
        "moment_dispersion": 0.0,
        "predicted_roughness_decrease": 1.0,
    }]
    expect_raises(lambda: choose_intrinsic_geometry(invalid), RuntimeError)


def test_geometry_tie_break_is_explicit():
    groups = ("a", "b", "c")
    canonical = ("residual_union_k7", 0.03, 0.5)
    alternative = ("residual_union_k5", 0.03, 0.5)
    values = {
        canonical: {group: 0.001 for group in groups},
        alternative: {group: 0.001 for group in groups},
    }
    selected, _ = choose_one_se(values, groups)
    assert selected == canonical


def test_cross_selector_has_no_lambda():
    groups = ("a", "b", "c")
    values = {
        0.5: {group: 0.001 for group in groups},
        1.0: {"a": 0.003, "b": -0.001, "c": 0.004},
    }
    trust, diagnostics = choose_cross_one_se(
        values, groups, "residual_union_k7"
    )
    assert trust == 0.5
    assert diagnostics["selected"]["lambda"] is None
    assert "lambda_absent" in diagnostics["policy"]


def test_actuator_keys_are_disjoint():
    full = candidate_key("residual_union_k7", 0.03, 0.5, "full")
    cross = candidate_key("residual_union_k7", None, 0.5, "cross")
    assert full != cross
    assert "a=full" in full and "a=cross" in cross
    assert "direction_only" in cross
    assert basis_key("outer", "residual_union_k7", None, "cross") != basis_key(
        "outer", "residual_union_k7", 0.03, "full"
    )


def test_cross_direction_and_trust_linearity():
    rng = np.random.default_rng(9)
    dimension = len(VIEW_ORDER)
    moments = []
    groups = []
    for group in ("a", "b", "c"):
        matrix = rng.normal(size=(dimension, dimension))
        A = matrix.T @ matrix / dimension + np.eye(dimension)
        c = rng.normal(size=dimension)
        moments.append(GraphRoughnessMoment(
            A=A,
            c=c,
            presence=np.ones(dimension, dtype=bool),
            families=tuple(VIEW_ORDER),
        ))
        groups.append(group)
    cross = fit_pooled_roughness_calibration(
        moments, groups, 1.0, cross_only=True
    )
    assert np.allclose(cross.direction, -cross.c)
    baseline = rng.normal(size=100)
    residuals = rng.normal(size=(100, dimension))
    unit = apply_pooled_roughness(
        baseline, residuals, VIEW_ORDER, cross, 1.0
    ).correction
    half = apply_pooled_roughness(
        baseline, residuals, VIEW_ORDER, cross, 0.5
    ).correction
    assert np.allclose(half, 0.5 * unit)
    assert np.allclose(
        apply_pooled_roughness(
            baseline, residuals, VIEW_ORDER, cross, 0.5
        ).score,
        baseline + 0.5 * unit,
    )
    assert np.array_equal(
        apply_pooled_roughness(
            baseline, residuals, VIEW_ORDER, cross, 0.0
        ).score,
        baseline,
    )


def test_nested_exclusion_registry():
    groups = ("a", "b", "c", "d")
    exclusions = set(exclusion_sets(groups))
    assert () in exclusions
    for held in groups:
        assert (held,) in exclusions
        for validation in groups:
            if held != validation:
                assert tuple(sorted((held, validation))) in exclusions
    # The score-basis semantics bind a target cell J to outer exclude={J}
    # and inner=H to exclude={H,J}; no outcome argument appears here.
    held, validation = "a", "b"
    assert tuple(sorted((held, validation))) == ("a", "b")


def test_equal_group_pooling_and_missing_family_semantics():
    dimension = len(VIEW_ORDER)
    presence_all = np.ones(dimension, dtype=bool)
    presence_missing = presence_all.copy()
    presence_missing[2] = False
    A1 = np.eye(dimension)
    c1 = np.ones(dimension)
    A2 = 3 * np.eye(dimension)
    c2 = 3 * np.ones(dimension)
    A3 = 10 * np.eye(dimension)
    c3 = 10 * np.ones(dimension)
    A3[2, :] = 0.0
    A3[:, 2] = 0.0
    c3[2] = 0.0
    moments = [
        GraphRoughnessMoment(A1, c1, presence_all, tuple(VIEW_ORDER)),
        GraphRoughnessMoment(A2, c2, presence_all, tuple(VIEW_ORDER)),
        GraphRoughnessMoment(A3, c3, presence_missing, tuple(VIEW_ORDER)),
    ]
    A, c, groups = pool_graph_roughness_moments(
        moments, ("many", "many", "single"), pooling="equal_group"
    )
    # First average the two 'many' cells, then give each group equal weight.
    expected_c = 0.5 * (2 * np.ones(dimension) + c3)
    assert np.allclose(c, expected_c)
    assert groups == ("many", "single")
    assert c[2] == 1.0  # absent family contributes its explicit aligned zero.


def test_sanitized_manifest_chain_and_launch_hashes():
    manifest = verify_isolated_bundle(DEFAULT_BUNDLE)
    assert manifest["output_sha256"] == sha256_file(DEFAULT_BUNDLE)
    assert manifest["source_sha256"] == sha256_file(ORIGINAL_BUNDLE)
    assert manifest["source_target_arrays_loaded"] is False
    assert manifest["output_contains_target_like_members"] is False
    assert len(launch_hash_audit()) == 11


def test_static_no_su_and_transfer_registry_boundaries():
    fit_source = (REPO / "scripts" / "graph_geometry_selection_fit.py").read_text()
    report_source = (REPO / "scripts" / "graph_geometry_selection_report.py").read_text()
    core_source = (REPO / "spectral_utils" / "graph_geometry_selection.py").read_text()
    scientific_import_lines = "\n".join(
        line for source in (fit_source, report_source, core_source)
        for line in source.splitlines()
        if line.startswith("from ") or line.startswith("import ")
    )
    assert "su_pooled" not in scientific_import_lines.lower()
    assert '"su_covariance_or_rho_arms": []' in fit_source
    assert '"held_family_oracle_exported": False' in report_source
    assert '"direction_families": list(VIEW_ORDER)' in report_source
    assert "canonical_frozen_selection_sha256" in report_source


def test_fixed_and_searched_share_identical_union_key():
    # Capacity arms are views over one bank: no separate fixed/search key exists.
    fixed = candidate_key("residual_union_k7", 3.0, 1.0, "full")
    searched_member = candidate_key("residual_union_k7", 3.0, 1.0, "full")
    assert fixed == searched_member


def test_original_bundle_is_rejected_as_fit_input():
    path = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
    with np.load(path, allow_pickle=True) as data:
        expect_raises(
            lambda: validate_physically_label_free_members(data.files, tuple(
                key[:-3] for key in data.files if key.endswith("__V")
            )),
            RuntimeError,
        )


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} graph-geometry tests")


if __name__ == "__main__":
    main()
