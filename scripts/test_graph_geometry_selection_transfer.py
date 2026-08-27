#!/usr/bin/env python3
"""Mechanical guards for the graph-geometry external transfer pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.graph_geometry_selection_transfer import (  # noqa: E402
    BASE_METHODS,
    CANONICAL_DEVELOPMENT_SELECTION_SHA256,
    DEVELOPMENT_VERSION,
    EXPECTED_ISOLATED_KEYS,
    TRANSFER_METHODS,
    array_hash,
    assert_identity_equal,
    canonical_hash,
    score_frozen_directions,
    sha256_file,
    source_hashes,
    validate_isolated_file,
    validate_selection_payload,
    verify_development_selection_chain,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


def expect_failure(function, *args) -> None:
    try:
        function(*args)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def selection_payload(
    *,
    fit_manifest_hash: str = "fit",
    label_free_sha256: str = "label-free",
    canonical_selection_hash: str = "canonical-selection",
) -> dict:
    entries = {}
    for method_index, name in enumerate(BASE_METHODS, start=1):
        direction = np.arange(1, len(VIEW_ORDER) + 1, dtype=float)
        direction *= method_index
        entries[name] = {
            "selector_type": name,
            "actuator": "full",
            "geometry_id": "residual_union_k7" if name == "canonical" else f"geometry_{name}",
            "lambda": 0.03,
            "trust_factor": 0.5,
            "direction_families": list(VIEW_ORDER),
            "direction": direction.tolist(),
            "calibration_key": f"full_{name}",
            "selection_diagnostics": None,
        }
        entries[f"{name}_cross"] = {
            "selector_type": f"{name}_cross",
            "actuator": "cross",
            "geometry_id": entries[name]["geometry_id"],
            "lambda": None,
            "trust_factor": 0.5,
            "direction_families": list(VIEW_ORDER),
            "direction": (-direction).tolist(),
            "calibration_key": f"cross_{name}",
            "selection_diagnostics": None,
        }
    payload = {
        "version": DEVELOPMENT_VERSION,
        "fit_manifest_hash": fit_manifest_hash,
        "fit_label_free_selection_sha256": label_free_sha256,
        "canonical_frozen_selection_sha256": (
            CANONICAL_DEVELOPMENT_SELECTION_SHA256
        ),
        "canonical_frozen_selection_hash": canonical_selection_hash,
        "development_outcomes_opened_for_supervised_entries": True,
        "held_family_oracle_exported": False,
        "retrospective_transfer_only": True,
        "entries": entries,
    }
    payload["selection_hash"] = canonical_hash(payload)
    return payload


def test_selection_guards() -> None:
    valid = selection_payload()
    assert validate_selection_payload(valid) == valid
    assert set(valid["entries"]) == set(TRANSFER_METHODS)

    corrupted = json.loads(json.dumps(valid))
    corrupted["entries"]["canonical"]["trust_factor"] = 1.0
    expect_failure(validate_selection_payload, corrupted)

    oracle = json.loads(json.dumps(valid))
    oracle["entries"]["held_family_oracle"] = oracle["entries"]["canonical"]
    oracle["selection_hash"] = canonical_hash({
        key: value for key, value in oracle.items() if key != "selection_hash"
    })
    expect_failure(validate_selection_payload, oracle)

    wrong_registry = json.loads(json.dumps(valid))
    wrong_registry["entries"]["label_free"]["direction_families"][0] = "wrong"
    wrong_registry["selection_hash"] = canonical_hash({
        key: value for key, value in wrong_registry.items()
        if key != "selection_hash"
    })
    expect_failure(validate_selection_payload, wrong_registry)

    canonical_unpinned = json.loads(json.dumps(valid))
    canonical_unpinned["canonical_frozen_selection_sha256"] = "changed"
    canonical_unpinned["selection_hash"] = canonical_hash({
        key: value for key, value in canonical_unpinned.items()
        if key != "selection_hash"
    })
    expect_failure(validate_selection_payload, canonical_unpinned)


def _write_self_hashed(path: Path, payload: dict, hash_field: str) -> dict:
    output = dict(payload)
    output[hash_field] = canonical_hash(output)
    path.write_text(json.dumps(output), encoding="utf-8")
    return output


def test_development_selection_chain_tamper_guards() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fit_complete = _write_self_hashed(
            root / "FIT_COMPLETE.json",
            {
                "version": DEVELOPMENT_VERSION,
                "label_free_selection_sha256": "pending",
            },
            "manifest_hash",
        )
        label_free = _write_self_hashed(
            root / "FROZEN_LABELFREE_SELECTION.json",
            {"version": DEVELOPMENT_VERSION, "contexts": {}},
            "selection_hash",
        )
        label_free_path = root / "FROZEN_LABELFREE_SELECTION.json"
        label_free_sha = sha256_file(label_free_path)
        fit_complete = _write_self_hashed(
            root / "FIT_COMPLETE.json",
            {
                "version": DEVELOPMENT_VERSION,
                "label_free_selection_sha256": label_free_sha,
            },
            "manifest_hash",
        )

        # The production verifier pins this file to the canonical launch SHA.
        # In the focused unit test, use the real canonical artifact while all
        # mutable development-chain fixtures stay inside the temporary root.
        canonical_path = (
            REPO / "results" / "pooled_graph_roughness_direction_v2"
            / "FROZEN_SELECTION.json"
        )
        canonical = json.loads(canonical_path.read_text())
        selection = selection_payload(
            fit_manifest_hash=fit_complete["manifest_hash"],
            label_free_sha256=label_free_sha,
            canonical_selection_hash=canonical["selection_hash"],
        )
        verify_development_selection_chain(
            validate_selection_payload(selection),
            development_root=root,
            canonical_selection_path=canonical_path,
        )

        wrong_fit = selection_payload(
            fit_manifest_hash="wrong",
            label_free_sha256=label_free_sha,
            canonical_selection_hash=canonical["selection_hash"],
        )
        expect_failure(
            lambda value: verify_development_selection_chain(
                validate_selection_payload(value),
                development_root=root,
                canonical_selection_path=canonical_path,
            ),
            wrong_fit,
        )

        wrong_label_free = selection_payload(
            fit_manifest_hash=fit_complete["manifest_hash"],
            label_free_sha256="wrong",
            canonical_selection_hash=canonical["selection_hash"],
        )
        expect_failure(
            lambda value: verify_development_selection_chain(
                validate_selection_payload(value),
                development_root=root,
                canonical_selection_path=canonical_path,
            ),
            wrong_label_free,
        )

        wrong_canonical = selection_payload(
            fit_manifest_hash=fit_complete["manifest_hash"],
            label_free_sha256=label_free_sha,
            canonical_selection_hash="wrong",
        )
        expect_failure(
            lambda value: verify_development_selection_chain(
                validate_selection_payload(value),
                development_root=root,
                canonical_selection_path=canonical_path,
            ),
            wrong_canonical,
        )

        label_free["contexts"] = {"tampered": {}}
        label_free_path.write_text(json.dumps(label_free), encoding="utf-8")
        expect_failure(
            lambda value: verify_development_selection_chain(
                validate_selection_payload(value),
                development_root=root,
                canonical_selection_path=canonical_path,
            ),
            selection,
        )

    assert "transfer_test" in source_hashes()


def test_score_formula() -> None:
    payload = selection_payload()
    n = 11
    residuals = np.column_stack((
        np.linspace(-2, 2, n),
        np.cos(np.linspace(0, 3, n)),
        np.sin(np.linspace(0, 2, n)),
        np.linspace(1, -1, n) ** 2,
        np.arange(n) % 3,
        np.linspace(-0.5, 0.8, n),
    ))
    baseline = np.linspace(-1, 1, n)
    state = SimpleNamespace(
        baseline=baseline,
        residuals=residuals,
        contribution_space=SimpleNamespace(families=tuple(VIEW_ORDER)),
    )
    scores = score_frozen_directions(state, payload["entries"])
    assert set(scores) == {"iu", *TRANSFER_METHODS}
    assert np.array_equal(scores["iu"], baseline)
    for method in TRANSFER_METHODS:
        correction = scores[method] - baseline
        assert abs(np.std(correction) - 0.5 / len(VIEW_ORDER)) < 1e-12
        assert len(array_hash(scores[method])) == 64


def test_physical_schema_and_identity_guard() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        n = 7
        F = np.arange(21, dtype=float).reshape(3, n)
        path = root / "cell.npz"
        np.savez_compressed(
            path,
            F=F,
            feature_names=np.asarray(("epr", "rpdi", "trace_length")),
            row_ids=np.asarray([f"row-{index}" for index in range(n)]),
        )
        entry = {
            "panel": "process_semgrad",
            "cell": "cell",
            "n": n,
            "feature_names": ["epr", "rpdi", "trace_length"],
        }
        arrays = validate_isolated_file(path, entry)
        assert set(arrays) == set(EXPECTED_ISOLATED_KEYS["process_semgrad"])
        assert_identity_equal(arrays, arrays, ("row_ids",))

        contaminated = root / "contaminated.npz"
        np.savez_compressed(
            contaminated,
            F=F,
            feature_names=np.asarray(("epr", "rpdi", "trace_length")),
            row_ids=np.asarray([f"row-{index}" for index in range(n)]),
            labels=np.zeros(n, dtype=int),
        )
        expect_failure(validate_isolated_file, contaminated, entry)

        shifted = dict(arrays)
        shifted["row_ids"] = np.roll(shifted["row_ids"], 1)
        expect_failure(assert_identity_equal, arrays, shifted, ("row_ids",))


def main() -> None:
    test_selection_guards()
    test_development_selection_chain_tamper_guards()
    test_score_formula()
    test_physical_schema_and_identity_guard()
    print("graph geometry selection transfer tests: PASS")


if __name__ == "__main__":
    main()
