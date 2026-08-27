#!/usr/bin/env python3
"""Frozen retrospective transfer for Graph Geometry Selection Research V1.

The pipeline is intentionally split into three processes:

``isolate``
    Read the historically opened raw caches through telemetry/identifier
    whitelists and write physically target-free feature archives.
``fit``
    Consume only those target-free archives plus the already frozen
    ``FROZEN_TRANSFER_SELECTIONS.json`` and freeze every score array.
``report``
    Verify all isolation, selection, score, provenance, and canonical
    reproduction hashes before opening any target outcome.

All outputs are retrospective stress tests.  They are not new confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    process_items,
)
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from scripts.neutral_residual_mode_hle_confirmation import (  # noqa: E402
    DEFAULT_JUDGE_MANIFEST,
    DEFAULT_LABELS,
    read_jsonl,
    telemetry_payload as hle_telemetry,
)
from scripts.neutral_residual_mode_prmbench_confirmation import (  # noqa: E402
    ordered_eligible_rows,
    telemetry_payload as prm_telemetry,
)
from scripts.pooled_graph_roughness_external import (  # noqa: E402
    DEFAULT_HLE_NRM,
    DEFAULT_HLE_RAW,
    DEFAULT_PRM_NRM,
    DEFAULT_PRM_RAW,
    process_semgrad_inputs,
)
from spectral_utils.family_residual_graph import (  # noqa: E402
    fit_family_residual_state,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


VERSION = "graph-geometry-selection-transfer-v1-2026-08-23"
DEVELOPMENT_VERSION = "graph-geometry-selection-research-v1-2026-08-23"
DEFAULT_SELECTION = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "development_fit" / "FROZEN_TRANSFER_SELECTIONS.json"
)
DEVELOPMENT_ROOT = DEFAULT_SELECTION.parent
DEVELOPMENT_FIT_COMPLETE = DEVELOPMENT_ROOT / "FIT_COMPLETE.json"
DEVELOPMENT_LABEL_FREE_SELECTION = (
    DEVELOPMENT_ROOT / "FROZEN_LABELFREE_SELECTION.json"
)
CANONICAL_DEVELOPMENT_SELECTION = (
    REPO / "results" / "pooled_graph_roughness_direction_v2"
    / "FROZEN_SELECTION.json"
)
CANONICAL_DEVELOPMENT_SELECTION_SHA256 = (
    "ff0b6e824d0140b7e5fbdab0d10f97b7a32ff80217d6b740915436c5ce8d1aa3"
)
DEFAULT_OUT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "external_transfer"
)
LEGACY_ROOT = REPO / "results" / "pooled_graph_roughness_external_v1"
PROCESS_NRM = (
    REPO / "results" / "neutral_residual_mode_cs_iu_v1"
    / "cell_results.csv"
)

BASE_METHODS = (
    "canonical",
    "label_free",
    "supervised_one_se",
    "supervised_max_mean",
)
TRANSFER_METHODS = BASE_METHODS + tuple(
    f"{name}_cross" for name in BASE_METHODS
)
SCORE_METHODS = ("iu",) + TRANSFER_METHODS
PANELS = ("process_semgrad", "prmbench", "hle")
DOMAIN_ORDER = (
    "processbench_llama",
    "processbench_qwen",
    "semgrad",
    "prmbench",
    "hle",
)
EXPECTED_ISOLATED_KEYS = {
    "process_semgrad": frozenset(("F", "feature_names", "row_ids")),
    "prmbench": frozenset((
        "F", "feature_names", "row_keys", "row_ids", "source_ids"
    )),
    "hle": frozenset(("F", "feature_names", "row_keys")),
}
FORBIDDEN_TARGET_FIELDS = frozenset((
    "label", "labels", "bem_correct", "classification", "correct",
    "correctness", "target", "targets", "y",
))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260823
FROZEN_REFERENCE_HASHES = {
    "legacy_process_manifest": "052153800667e26f413d78987be92cba9318fb99d65046df7f0cee6a99dd4c43",
    "legacy_prmbench_manifest": "b4d76c558b9fcc1b472fc6ce519d96982baeb4e71609099bba500dd1cd4aa9d8",
    "legacy_hle_manifest": "a179f0c665bd5abe82478c0263d134d86ab25c4a8b931edbce34db8024cc9a71",
    "legacy_directions": "789bc6cbc8616baf5bbb01d0e161bca5f3cdeade48f4d0bf1d884e7c8339e0e8",
    "process_semgrad_nrm_cells": "0c4f836ac2ae046c228b943c3afcf26a6abe20f84dc138d4ada9dc6a3d96c278",
    "prmbench_nrm_scores": "8945d52bf6781abad03f35f8ba104c72bbbf194b3511ccb4e6481c5fe97d0227",
    "hle_nrm_scores": "e74afa0d46002cad493fbcd691e2076f3d8daf941992213e66c4f8cad39e687d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def array_hash(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def source_hashes() -> dict[str, str]:
    paths = {
        "transfer_script": Path(__file__),
        "legacy_external_script": (
            REPO / "scripts" / "pooled_graph_roughness_external.py"
        ),
        "mixed_v2_builder": (
            REPO / "scripts" / "leverage_balanced_processbench_transfer.py"
        ),
        "process_semgrad_loader": (
            REPO / "scripts" / "harp_global_contribution_teacher.py"
        ),
        "prmbench_loader": (
            REPO / "scripts" / "neutral_residual_mode_prmbench_confirmation.py"
        ),
        "hle_loader": (
            REPO / "scripts" / "neutral_residual_mode_hle_confirmation.py"
        ),
        "family_graph_module": (
            REPO / "spectral_utils" / "family_residual_graph.py"
        ),
        "family_registry": REPO / "spectral_utils" / "specrage_views.py",
        "contribution_module": (
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "upcr_module": REPO / "spectral_utils" / "upcr.py",
        "dufs_contract": (
            REPO / "spectral_utils" / "dufs_liu_feature_contract.py"
        ),
        "feature_contract": REPO / "spectral_utils" / "feature_contract.py",
        "feature_utils": REPO / "spectral_utils" / "feature_utils.py",
        "repgrid_scoring": REPO / "spectral_utils" / "repgrid_scoring.py",
        "transfer_test": (
            REPO / "scripts" / "test_graph_geometry_selection_transfer.py"
        ),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_selection_payload(selection: dict) -> dict:
    payload = dict(selection)
    recorded = payload.pop("selection_hash", None)
    if recorded is None or canonical_hash(payload) != recorded:
        raise RuntimeError("transfer selection is not self-consistent")
    if selection.get("version") != DEVELOPMENT_VERSION:
        raise RuntimeError("unexpected development selection version")
    if selection.get("held_family_oracle_exported") is not False:
        raise RuntimeError("held-family oracle must not enter transfer")
    if selection.get("retrospective_transfer_only") is not True:
        raise RuntimeError("transfer scope is not frozen as retrospective")
    if selection.get("canonical_frozen_selection_sha256") != (
        CANONICAL_DEVELOPMENT_SELECTION_SHA256
    ):
        raise RuntimeError("canonical development selection SHA changed")
    if not isinstance(selection.get("canonical_frozen_selection_hash"), str):
        raise RuntimeError("canonical development selection hash is absent")
    entries = selection.get("entries", {})
    if set(entries) != set(TRANSFER_METHODS):
        raise RuntimeError("transfer method registry changed")
    for name in BASE_METHODS:
        full = entries[name]
        cross = entries[f"{name}_cross"]
        if full.get("actuator") != "full" or cross.get("actuator") != "cross":
            raise RuntimeError(f"actuator mismatch: {name}")
        if tuple(full.get("direction_families", ())) != tuple(VIEW_ORDER):
            raise RuntimeError(f"full family registry mismatch: {name}")
        if tuple(cross.get("direction_families", ())) != tuple(VIEW_ORDER):
            raise RuntimeError(f"cross family registry mismatch: {name}")
        for suffix, row in (("full", full), ("cross", cross)):
            direction = np.asarray(row.get("direction"), dtype=float)
            if direction.shape != (len(VIEW_ORDER),) or not np.isfinite(
                direction
            ).all():
                raise RuntimeError(f"invalid {suffix} direction: {name}")
            trust = float(row.get("trust_factor", np.nan))
            if not np.isfinite(trust) or trust <= 0:
                raise RuntimeError(f"invalid {suffix} trust: {name}")
        if full.get("lambda") is None or float(full["lambda"]) <= 0:
            raise RuntimeError(f"invalid full lambda: {name}")
        if cross.get("lambda") is not None:
            raise RuntimeError(f"cross actuator unexpectedly has lambda: {name}")
        if full.get("geometry_id") != cross.get("geometry_id"):
            raise RuntimeError(f"matched-cross geometry mismatch: {name}")
        if float(full["trust_factor"]) != float(cross["trust_factor"]):
            raise RuntimeError(f"matched-cross trust mismatch: {name}")
    canonical = entries["canonical"]
    if not (
        canonical["geometry_id"] == "residual_union_k7"
        and float(canonical["lambda"]) == 0.03
        and float(canonical["trust_factor"]) == 0.5
    ):
        raise RuntimeError("canonical transfer anchor changed")
    return selection


def _verify_self_hash(path: Path, hash_field: str, description: str) -> dict:
    payload = json.loads(Path(path).read_text())
    unhashed = dict(payload)
    recorded = unhashed.pop(hash_field, None)
    if recorded is None or canonical_hash(unhashed) != recorded:
        raise RuntimeError(f"{description} is not self-consistent")
    return payload


def verify_development_selection_chain(
    selection: dict,
    *,
    development_root: Path = DEVELOPMENT_ROOT,
    canonical_selection_path: Path = CANONICAL_DEVELOPMENT_SELECTION,
) -> dict:
    """Bind an outcome-facing transfer selection to its frozen fit lineage."""

    development_root = Path(development_root)
    fit_complete_path = development_root / "FIT_COMPLETE.json"
    label_free_path = development_root / "FROZEN_LABELFREE_SELECTION.json"

    fit_complete = _verify_self_hash(
        fit_complete_path, "manifest_hash", "development fit manifest"
    )
    if selection.get("fit_manifest_hash") != fit_complete["manifest_hash"]:
        raise RuntimeError("transfer selection/development fit manifest mismatch")

    label_free = _verify_self_hash(
        label_free_path, "selection_hash", "development label-free selection"
    )
    label_free_sha = sha256_file(label_free_path)
    if fit_complete.get("label_free_selection_sha256") != label_free_sha:
        raise RuntimeError("development fit/label-free selection SHA mismatch")
    if selection.get("fit_label_free_selection_sha256") != label_free_sha:
        raise RuntimeError("transfer selection/label-free selection SHA mismatch")

    canonical_selection_path = Path(canonical_selection_path)
    canonical_sha = sha256_file(canonical_selection_path)
    if canonical_sha != CANONICAL_DEVELOPMENT_SELECTION_SHA256:
        raise RuntimeError("canonical development selection artifact changed")
    canonical = _verify_self_hash(
        canonical_selection_path,
        "selection_hash",
        "canonical development selection",
    )
    if selection.get("canonical_frozen_selection_sha256") != canonical_sha:
        raise RuntimeError("transfer/canonical selection SHA mismatch")
    if selection.get("canonical_frozen_selection_hash") != canonical["selection_hash"]:
        raise RuntimeError("transfer/canonical selection hash mismatch")

    return selection


def load_selection(path: Path) -> dict:
    selection = validate_selection_payload(json.loads(Path(path).read_text()))
    return verify_development_selection_chain(selection)


def isolated_path(out: Path, cell: str) -> Path:
    return Path(out) / "isolated" / "cells" / f"{cell}.npz"


def score_path(out: Path, cell: str) -> Path:
    return Path(out) / "scores" / f"{cell}.npz"


def _write_isolated_cell(
    out: Path,
    *,
    panel: str,
    domain: str,
    group: str,
    cell: str,
    source: Path,
    telemetry: list[dict],
    identity: dict[str, np.ndarray],
) -> dict:
    F, names, availability, contract = mixed_v2_matrix(telemetry)
    n = len(telemetry)
    if F.shape[1] != n or len(names) != F.shape[0] or not np.isfinite(F).all():
        raise RuntimeError(f"invalid isolated feature matrix: {cell}")
    for key, values in identity.items():
        if np.asarray(values).shape != (n,):
            raise RuntimeError(f"invalid isolated identity: {cell}/{key}")
    path = isolated_path(out, cell)
    np.savez_compressed(
        path,
        F=np.asarray(F, dtype=float),
        feature_names=np.asarray(names),
        **identity,
    )
    source = Path(source).resolve()
    return {
        "panel": panel,
        "domain": domain,
        "group": group,
        "cell": cell,
        "n": n,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "isolated_path": str(path.resolve()),
        "isolated_sha256": sha256_file(path),
        "allowed_fields": sorted(EXPECTED_ISOLATED_KEYS[panel]),
        "feature_names": list(names),
        "availability": availability,
        "feature_contract": contract,
    }


def isolate(args) -> None:
    manifest_path = args.out / "isolated" / "ISOLATION_MANIFEST.json"
    cell_dir = args.out / "isolated" / "cells"
    if manifest_path.exists() or cell_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite isolated transfer inputs: {cell_dir}"
        )
    cell_dir.mkdir(parents=True)
    roster = []

    process_rows = process_semgrad_inputs()
    process_count = len(process_rows)
    for index, row in enumerate(process_rows, start=1):
        print(
            f"[isolate {index}/{process_count + 2}] {row['cell']}",
            flush=True,
        )
        roster.append(_write_isolated_cell(
            args.out,
            panel="process_semgrad",
            domain=row["domain"],
            group=row["group"],
            cell=row["cell"],
            source=row["path"],
            telemetry=row["telemetry"],
            identity={"row_ids": np.asarray(row["row_ids"]).astype(str)},
        ))
        # The loader materializes every panel row at once.  Release each large
        # telemetry payload immediately after its target-free archive exists.
        row["telemetry"] = None
    del process_rows

    prm_cache = load_pickle(args.prm_raw)
    selected = ordered_eligible_rows(prm_cache)
    prm_telemetry_rows = [prm_telemetry(row[3]) for row in selected]
    print(f"[isolate {process_count + 1}/{process_count + 2}] prmbench", flush=True)
    roster.append(_write_isolated_cell(
        args.out,
        panel="prmbench",
        domain="prmbench",
        group="prmbench",
        cell="prmbench",
        source=args.prm_raw,
        telemetry=prm_telemetry_rows,
        identity={
            "row_keys": np.asarray([row[0] for row in selected], dtype=int),
            "row_ids": np.asarray([row[1] for row in selected]).astype(str),
            "source_ids": np.asarray([row[2] for row in selected]).astype(str),
        },
    ))
    del prm_cache, selected, prm_telemetry_rows

    hle_cache = load_pickle(args.hle_raw)
    row_keys = sorted(hle_cache, key=lambda value: int(value))
    hle_rows = []
    for key in row_keys:
        candidates = hle_cache[key].get("candidates")
        if not candidates:
            raise RuntimeError(f"missing HLE candidate: {key}")
        hle_rows.append(hle_telemetry(candidates[0]))
    print(f"[isolate {process_count + 2}/{process_count + 2}] hle", flush=True)
    roster.append(_write_isolated_cell(
        args.out,
        panel="hle",
        domain="hle",
        group="hle",
        cell="hle",
        source=args.hle_raw,
        telemetry=hle_rows,
        identity={
            "row_keys": np.asarray([int(key) for key in row_keys], dtype=int),
        },
    ))

    payload = {
        "version": VERSION,
        "phase": "physically_target_free_feature_isolation",
        "scope": "retrospective_known_outcome_stress_test",
        "roster": roster,
        "source_hashes": source_hashes(),
        "outcome_fields_indexed_by_isolation": [],
        "isolation_whitelist": {
            "telemetry": [
                "token_entropies", "token_spilled_energies",
                "token_logsumexp", "top_k_logprobs",
            ],
            "identifiers": ["row_key", "idx", "source_idx", "cache_key"],
            "non_target_eligibility": ["align_diag.problems", "candidate_presence"],
        },
        "forbidden_target_fields": sorted(FORBIDDEN_TARGET_FIELDS),
        "target_fields_physically_present_in_score_fit_inputs": False,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    write_json(manifest_path, payload)
    print(json.dumps({
        "phase": payload["phase"],
        "cells": len(roster),
        "samples": sum(row["n"] for row in roster),
        "manifest_hash": payload["manifest_hash"],
    }, indent=2))


def validate_isolated_file(path: Path, entry: dict) -> dict[str, np.ndarray]:
    panel = entry["panel"]
    if panel not in EXPECTED_ISOLATED_KEYS:
        raise RuntimeError(f"unknown isolated panel: {panel}")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(EXPECTED_ISOLATED_KEYS[panel]):
            raise RuntimeError(f"target-free schema changed: {entry['cell']}")
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    F = np.asarray(arrays["F"], dtype=float)
    names = arrays["feature_names"].astype(str)
    if F.ndim != 2 or F.shape[0] != len(names) or F.shape[1] != entry["n"]:
        raise RuntimeError(f"isolated feature shape changed: {entry['cell']}")
    if not np.isfinite(F).all() or names.tolist() != entry["feature_names"]:
        raise RuntimeError(f"isolated feature content changed: {entry['cell']}")
    normalized_names = {
        str(name).strip().lower() for name in names
    }
    if normalized_names & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError(f"target-like feature entered isolation: {entry['cell']}")
    for name in EXPECTED_ISOLATED_KEYS[panel] - {"F", "feature_names"}:
        if arrays[name].shape != (entry["n"],):
            raise RuntimeError(f"isolated identity shape changed: {entry['cell']}/{name}")
    return arrays


def verify_isolation(out: Path) -> dict:
    path = out / "isolated" / "ISOLATION_MANIFEST.json"
    manifest = json.loads(path.read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash", None)
    if recorded is None or canonical_hash(payload) != recorded:
        raise RuntimeError("isolation manifest is not self-consistent")
    if manifest.get("version") != VERSION:
        raise RuntimeError("isolation version changed")
    if manifest.get("phase") != "physically_target_free_feature_isolation":
        raise RuntimeError("isolation phase changed")
    if manifest.get("outcome_fields_indexed_by_isolation") != []:
        raise RuntimeError("isolation indexed an outcome field")
    if manifest.get("target_fields_physically_present_in_score_fit_inputs") is not False:
        raise RuntimeError("score-fit inputs are not physically target-free")
    cells = [row["cell"] for row in manifest["roster"]]
    if len(cells) != 16 or len(cells) != len(set(cells)):
        raise RuntimeError("external transfer roster changed")
    if set(row["panel"] for row in manifest["roster"]) != set(PANELS):
        raise RuntimeError("external panel roster changed")
    for entry in manifest["roster"]:
        path = isolated_path(out, entry["cell"])
        if str(path.resolve()) != entry["isolated_path"]:
            raise RuntimeError(f"isolated path changed: {entry['cell']}")
        if sha256_file(path) != entry["isolated_sha256"]:
            raise RuntimeError(f"isolated file hash changed: {entry['cell']}")
        validate_isolated_file(path, entry)
    return manifest


def score_frozen_directions(state, entries: dict) -> dict[str, np.ndarray]:
    families = tuple(state.contribution_space.families)
    if not families or any(name not in VIEW_ORDER for name in families):
        raise RuntimeError("target IU state has an unknown family registry")
    scores = {"iu": np.asarray(state.baseline, dtype=float)}
    for name in TRANSFER_METHODS:
        row = entries[name]
        registry = tuple(row["direction_families"])
        indices = np.asarray([registry.index(family) for family in families], dtype=int)
        direction = np.asarray(row["direction"], dtype=float)[indices]
        raw = np.asarray(state.residuals, dtype=float) @ direction
        scale = float(np.std(raw, ddof=0))
        if scale <= 1e-12:
            correction = np.zeros_like(scores["iu"])
        else:
            correction = (
                float(row["trust_factor"]) / len(families)
            ) * raw / scale
        scores[name] = np.asarray(scores["iu"] + correction, dtype=float)
        if not np.isfinite(scores[name]).all():
            raise RuntimeError(f"non-finite transfer score: {name}")
    return scores


def _identity_names(panel: str) -> tuple[str, ...]:
    return tuple(sorted(
        EXPECTED_ISOLATED_KEYS[panel] - {"F", "feature_names"}
    ))


def assert_identity_equal(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], names
) -> None:
    for name in names:
        left = np.asarray(expected[name])
        right = np.asarray(actual[name])
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            equal = np.array_equal(left.astype(str), right.astype(str))
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise RuntimeError(f"row identity mismatch: {name}")


def legacy_panel_entry(panel: str) -> tuple[Path, dict]:
    root = LEGACY_ROOT / panel
    manifest_path = root / "FIT_MANIFEST.json"
    expected_key = {
        "process_semgrad": "legacy_process_manifest",
        "prmbench": "legacy_prmbench_manifest",
        "hle": "legacy_hle_manifest",
    }[panel]
    if sha256_file(manifest_path) != FROZEN_REFERENCE_HASHES[expected_key]:
        raise RuntimeError(f"frozen legacy manifest changed: {panel}")
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash", None)
    if recorded is None or canonical_hash(payload) != recorded:
        raise RuntimeError(f"legacy external manifest is inconsistent: {panel}")
    if manifest.get("panel") != panel:
        raise RuntimeError(f"legacy external panel mismatch: {panel}")
    if manifest.get("uses_target_labels_for_scoring") is not False:
        raise RuntimeError(f"legacy score bank used labels: {panel}")
    return root, manifest


def verify_legacy_directions(selection: dict) -> dict[str, str]:
    hashes = {}
    for panel in PANELS:
        path = LEGACY_ROOT / panel / "FROZEN_DIRECTIONS.json"
        if sha256_file(path) != FROZEN_REFERENCE_HASHES["legacy_directions"]:
            raise RuntimeError(f"frozen legacy directions changed: {panel}")
        registry = json.loads(path.read_text())
        mappings = {
            "canonical": "primary_one_se",
            "canonical_cross": "cross_only",
        }
        for new_name, legacy_name in mappings.items():
            new = selection["entries"][new_name]
            old = registry[legacy_name]
            if not np.array_equal(
                np.asarray(new["direction"], dtype=float),
                np.asarray(old["direction"], dtype=float),
            ) or float(new["trust_factor"]) != float(old["trust_factor"]):
                raise RuntimeError(
                    f"canonical frozen direction drift: {panel}/{new_name}"
                )
        hashes[panel] = sha256_file(path)
    return hashes


def canonical_reproduction_check(
    out: Path,
    isolation: dict,
    scores_by_cell: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    panels = {panel: legacy_panel_entry(panel) for panel in PANELS}
    rows = []
    for entry in isolation["roster"]:
        panel = entry["panel"]
        legacy_root, legacy_manifest = panels[panel]
        old_path = legacy_root / "scores" / f"{entry['cell']}.npz"
        expected_hash = legacy_manifest["score_hashes"][entry["cell"]]
        if sha256_file(old_path) != expected_hash:
            raise RuntimeError(f"legacy score hash changed: {entry['cell']}")
        isolated = validate_isolated_file(
            isolated_path(out, entry["cell"]), entry
        )
        with np.load(old_path, allow_pickle=False) as stored:
            old = {name: np.asarray(stored[name]) for name in stored.files}
        assert_identity_equal(
            isolated, old, _identity_names(panel)
        )
        new_scores = scores_by_cell[entry["cell"]]
        comparisons = {
            "iu": "iu",
            "canonical": "primary_one_se",
            "canonical_cross": "cross_only",
        }
        for new_name, old_name in comparisons.items():
            if not np.array_equal(new_scores[new_name], old[old_name]):
                maximum = float(np.max(np.abs(
                    new_scores[new_name] - np.asarray(old[old_name], dtype=float)
                )))
                raise RuntimeError(
                    f"canonical score reproduction failed: "
                    f"{entry['cell']}/{new_name}, max_abs={maximum}"
                )
        rows.append({
            "panel": panel,
            "cell": entry["cell"],
            "n": entry["n"],
            "legacy_score_sha256": expected_hash,
            "iu_exact": True,
            "canonical_full_exact": True,
            "canonical_cross_exact": True,
        })
    return rows


def fit(args) -> None:
    if (args.out / "FIT_MANIFEST.json").exists() or (args.out / "scores").exists():
        raise FileExistsError(f"refusing to overwrite transfer fit: {args.out}")
    isolation = verify_isolation(args.out)
    selection = load_selection(args.selection)
    legacy_direction_hashes = verify_legacy_directions(selection)
    (args.out / "scores").mkdir(parents=True)
    frozen_selection = args.out / "FROZEN_TRANSFER_SELECTIONS.json"
    frozen_selection.write_bytes(Path(args.selection).read_bytes())

    score_hashes, score_file_hashes, scores_by_cell = {}, {}, {}
    for index, entry in enumerate(isolation["roster"], start=1):
        cell = entry["cell"]
        print(f"[fit {index}/{len(isolation['roster'])}] {cell}", flush=True)
        arrays = validate_isolated_file(isolated_path(args.out, cell), entry)
        state = fit_family_residual_state(
            np.asarray(arrays["F"], dtype=float),
            tuple(arrays["feature_names"].astype(str)),
        )
        scores = score_frozen_directions(state, selection["entries"])
        if set(scores) != set(SCORE_METHODS):
            raise RuntimeError(f"score registry changed: {cell}")
        scores_by_cell[cell] = scores
        identity = {
            name: arrays[name] for name in _identity_names(entry["panel"])
        }
        output_path = score_path(args.out, cell)
        np.savez_compressed(
            output_path,
            feature_names=arrays["feature_names"],
            families=np.asarray(state.contribution_space.families),
            **identity,
            **scores,
        )
        score_file_hashes[cell] = sha256_file(output_path)
        score_hashes[cell] = {
            name: array_hash(scores[name]) for name in SCORE_METHODS
        }

    reproduction = canonical_reproduction_check(
        args.out, isolation, scores_by_cell
    )
    write_json(args.out / "SCORE_HASHES.json", score_hashes)
    write_json(args.out / "CANONICAL_REPRODUCTION.json", {
        "all_exact": True,
        "legacy_method_mapping": {
            "iu": "iu",
            "canonical": "primary_one_se",
            "canonical_cross": "cross_only",
        },
        "rows": reproduction,
    })
    manifest = {
        "version": VERSION,
        "phase": "target_free_scores_frozen",
        "scope": "retrospective_known_outcome_stress_test",
        "selection_hash": selection["selection_hash"],
        "selection_sha256": sha256_file(frozen_selection),
        "selection_source_path": str(Path(args.selection).resolve()),
        "isolation_manifest_hash": isolation["manifest_hash"],
        "isolation_manifest_sha256": sha256_file(
            args.out / "isolated" / "ISOLATION_MANIFEST.json"
        ),
        "score_file_hashes": score_file_hashes,
        "score_hashes_sha256": sha256_file(args.out / "SCORE_HASHES.json"),
        "canonical_reproduction_sha256": sha256_file(
            args.out / "CANONICAL_REPRODUCTION.json"
        ),
        "legacy_direction_hashes": legacy_direction_hashes,
        "source_hashes": source_hashes(),
        "methods": list(SCORE_METHODS),
        "labels_accessed_by_fit": False,
        "target_fields_received_by_fit": [],
        "target_fields_physically_present_in_fit_input": False,
        "raw_target_caches_accessed_by_fit": False,
        "canonical_scores_exactly_reproduced": True,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(args.out / "FIT_MANIFEST.json", manifest)
    print(json.dumps({
        "phase": manifest["phase"],
        "cells": len(scores_by_cell),
        "methods": len(SCORE_METHODS),
        "canonical_exact": True,
        "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


def _score_expected_keys(panel: str) -> set[str]:
    return set(SCORE_METHODS) | set(_identity_names(panel)) | {
        "feature_names", "families",
    }


def verify_fit(args) -> tuple[dict, dict, dict, dict]:
    isolation = verify_isolation(args.out)
    selection = load_selection(args.selection)
    manifest = json.loads((args.out / "FIT_MANIFEST.json").read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash", None)
    if recorded is None or canonical_hash(payload) != recorded:
        raise RuntimeError("transfer fit manifest is not self-consistent")
    if manifest.get("version") != VERSION or manifest.get("phase") != "target_free_scores_frozen":
        raise RuntimeError("transfer fit version/phase changed")
    if manifest.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("transfer fit accessed labels")
    if manifest.get("target_fields_physically_present_in_fit_input") is not False:
        raise RuntimeError("transfer fit input was not physically target-free")
    if manifest.get("raw_target_caches_accessed_by_fit") is not False:
        raise RuntimeError("transfer fit touched a raw target cache")
    if manifest.get("methods") != list(SCORE_METHODS):
        raise RuntimeError("frozen transfer score registry changed")
    if manifest["selection_hash"] != selection["selection_hash"]:
        raise RuntimeError("fit/selection hash mismatch")
    frozen_selection = args.out / "FROZEN_TRANSFER_SELECTIONS.json"
    if sha256_file(frozen_selection) != manifest["selection_sha256"]:
        raise RuntimeError("frozen transfer selection file changed")
    if json.loads(frozen_selection.read_text()) != selection:
        raise RuntimeError("development/frozen transfer selections disagree")
    if sha256_file(args.selection) != manifest["selection_sha256"]:
        raise RuntimeError("development transfer selection changed after fit")
    isolation_path = args.out / "isolated" / "ISOLATION_MANIFEST.json"
    if sha256_file(isolation_path) != manifest["isolation_manifest_sha256"]:
        raise RuntimeError("fit isolation file hash changed")
    if isolation["manifest_hash"] != manifest["isolation_manifest_hash"]:
        raise RuntimeError("fit isolation manifest changed")
    if source_hashes() != manifest["source_hashes"]:
        raise RuntimeError("transfer source code changed after score freeze")
    if verify_legacy_directions(selection) != manifest["legacy_direction_hashes"]:
        raise RuntimeError("legacy canonical direction source changed")
    score_hashes_path = args.out / "SCORE_HASHES.json"
    if sha256_file(score_hashes_path) != manifest["score_hashes_sha256"]:
        raise RuntimeError("transfer score-hash registry changed")
    score_hashes = json.loads(score_hashes_path.read_text())
    scores_by_cell = {}
    for entry in isolation["roster"]:
        cell = entry["cell"]
        path = score_path(args.out, cell)
        if sha256_file(path) != manifest["score_file_hashes"][cell]:
            raise RuntimeError(f"frozen transfer score file changed: {cell}")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != _score_expected_keys(entry["panel"]):
                raise RuntimeError(f"frozen transfer score schema changed: {cell}")
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        isolated = validate_isolated_file(isolated_path(args.out, cell), entry)
        assert_identity_equal(isolated, arrays, _identity_names(entry["panel"]))
        if not np.array_equal(
            isolated["feature_names"].astype(str),
            arrays["feature_names"].astype(str),
        ):
            raise RuntimeError(f"frozen transfer feature registry changed: {cell}")
        methods = {}
        for method in SCORE_METHODS:
            values = np.asarray(arrays[method], dtype=float)
            if values.shape != (entry["n"],) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid frozen score: {cell}/{method}")
            if array_hash(values) != score_hashes[cell][method]:
                raise RuntimeError(f"frozen score array hash changed: {cell}/{method}")
            methods[method] = values
        scores_by_cell[cell] = methods
    reproduction_path = args.out / "CANONICAL_REPRODUCTION.json"
    if sha256_file(reproduction_path) != manifest["canonical_reproduction_sha256"]:
        raise RuntimeError("canonical reproduction artifact changed")
    reproduction = json.loads(reproduction_path.read_text())
    if reproduction.get("all_exact") is not True:
        raise RuntimeError("canonical reproduction is not exact")
    replay = canonical_reproduction_check(args.out, isolation, scores_by_cell)
    if replay != reproduction["rows"]:
        raise RuntimeError("canonical reproduction replay changed")
    return manifest, isolation, selection, scores_by_cell


def _verify_raw_source(entry: dict) -> Path:
    path = Path(entry["source_path"])
    if sha256_file(path) != entry["source_sha256"]:
        raise RuntimeError(f"raw target source changed: {entry['cell']}")
    return path


def load_labels_after_verification(
    args, isolation: dict
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    labels, label_hashes = {}, {}
    for entry in isolation["roster"]:
        cell = entry["cell"]
        source = _verify_raw_source(entry)
        panel = entry["panel"]
        isolated = validate_isolated_file(isolated_path(args.out, cell), entry)
        if panel == "process_semgrad" and cell.startswith("semgrad__"):
            cache = load_pickle(source)
            ids, values = [], []
            for key in sorted(cache):
                candidates = cache[key].get("candidates")
                if not candidates:
                    continue
                ids.append(str(key))
                values.append(int(candidates[0]["bem_correct"]))
            actual = {"row_ids": np.asarray(ids)}
            y = np.asarray(values, dtype=int)
        elif panel == "process_semgrad":
            items = process_items(source)
            actual = {"row_ids": np.asarray([key for key, _ in items])}
            y = np.asarray([
                int(row["label"] == -1) for _, row in items
            ], dtype=int)
        elif panel == "prmbench":
            cache = load_pickle(source)
            selected = ordered_eligible_rows(cache)
            actual = {
                "row_keys": np.asarray([row[0] for row in selected], dtype=int),
                "row_ids": np.asarray([row[1] for row in selected]),
                "source_ids": np.asarray([row[2] for row in selected]),
            }
            y = np.asarray([
                str(row[3]["classification"]) == "correct" for row in selected
            ], dtype=int)
        elif panel == "hle":
            if sha256_file(args.hle_labels) != json.loads(
                args.hle_judge_manifest.read_text()
            )["hashes"]["output_judgments_sha256"]:
                raise RuntimeError("HLE judge manifest does not authenticate labels")
            judged = read_jsonl(args.hle_labels)
            actual = {
                "row_keys": np.asarray([
                    int(row["row_key"]) for row in judged
                ], dtype=int),
            }
            y = np.asarray([
                row["correct"] == "yes" for row in judged
            ], dtype=int)
            label_hashes[str(args.hle_labels.resolve())] = sha256_file(args.hle_labels)
            label_hashes[str(args.hle_judge_manifest.resolve())] = sha256_file(
                args.hle_judge_manifest
            )
        else:
            raise RuntimeError(f"unknown label panel: {panel}")
        assert_identity_equal(
            isolated, actual, _identity_names(panel)
        )
        if y.shape != (entry["n"],) or not np.all(np.isin(y, (0, 1))):
            raise RuntimeError(f"invalid target outcome: {cell}")
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"one-class target outcome: {cell}")
        labels[cell] = y
        label_hashes[str(source.resolve())] = entry["source_sha256"]
    return labels, label_hashes


def process_nrm_reference() -> dict[str, dict]:
    rows = {}
    with PROCESS_NRM.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] == "source23_transfer":
                rows[row["cell"]] = row
    return rows


def verify_comparison_references(args) -> dict[str, str]:
    hashes = {
        "process_semgrad_nrm_cells": sha256_file(PROCESS_NRM),
        "prmbench_nrm_scores": sha256_file(args.prm_nrm),
        "hle_nrm_scores": sha256_file(args.hle_nrm),
    }
    for name, observed in hashes.items():
        if observed != FROZEN_REFERENCE_HASHES[name]:
            raise RuntimeError(f"frozen Family-NRM reference changed: {name}")
    return hashes


def _load_nrm_scores(args, entry, score_arrays) -> np.ndarray | None:
    if entry["panel"] == "process_semgrad":
        return None
    path = args.prm_nrm if entry["panel"] == "prmbench" else args.hle_nrm
    isolated = validate_isolated_file(
        isolated_path(args.out, entry["cell"]), entry
    )
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    assert_identity_equal(
        isolated, arrays, _identity_names(entry["panel"])
    )
    with np.load(
        score_path(args.out, entry["cell"]), allow_pickle=False
    ) as stored:
        score_features = stored["feature_names"].astype(str)
        score_families = stored["families"].astype(str)
    if not np.array_equal(
        arrays["feature_names"].astype(str),
        score_features,
    ):
        raise RuntimeError(f"Family-NRM feature mismatch: {entry['cell']}")
    if not np.array_equal(
        arrays["families"].astype(str), score_families,
    ):
        raise RuntimeError(f"Family-NRM family mismatch: {entry['cell']}")
    if not np.allclose(
        arrays["iu_correctness_score"], score_arrays["iu"], atol=1e-12, rtol=0,
    ):
        raise RuntimeError(f"Family-NRM IU baseline mismatch: {entry['cell']}")
    return np.asarray(arrays["nrm_correctness_score"], dtype=float)


def metric_rows(
    args, isolation: dict, selection: dict,
    scores_by_cell: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    references = process_nrm_reference()
    rows, row_nrm_scores = [], {}
    for entry in isolation["roster"]:
        cell = entry["cell"]
        y = labels[cell]
        scores = scores_by_cell[cell]
        base_auc = float(roc_auc_score(y, scores["iu"]))
        nrm_score = _load_nrm_scores(args, entry, scores)
        if nrm_score is None:
            reference = references[cell]
            if (
                int(reference["n"]) != len(y)
                or int(reference["n_correct"]) != int(np.sum(y))
                or reference["domain"] != entry["domain"]
                or reference["group"] != entry["group"]
                or abs(float(reference["iu_auroc"]) - base_auc) > 1e-12
            ):
                raise RuntimeError(f"Family-NRM reference drift: {cell}")
            nrm_delta = float(reference["nrm_delta_pp"])
        else:
            row_nrm_scores[cell] = nrm_score
            nrm_delta = 100 * (
                float(roc_auc_score(y, nrm_score)) - base_auc
            )
        for method in SCORE_METHODS:
            auc = float(roc_auc_score(y, scores[method]))
            auprc = float(average_precision_score(y, scores[method]))
            method_row = None if method == "iu" else selection["entries"][method]
            rows.append({
                "domain": entry["domain"],
                "group": entry["group"],
                "cell": cell,
                "n": len(y),
                "n_correct": int(np.sum(y)),
                "method": method,
                "selector_type": "baseline" if method_row is None else method_row["selector_type"],
                "actuator": "none" if method_row is None else method_row["actuator"],
                "geometry_id": "none" if method_row is None else method_row["geometry_id"],
                "lambda": "" if method_row is None or method_row["lambda"] is None else method_row["lambda"],
                "trust_factor": "" if method_row is None else method_row["trust_factor"],
                "auroc": auc,
                "auprc": auprc,
                "delta_vs_iu_pp": 100 * (auc - base_auc),
                "family_nrm_delta_pp": nrm_delta,
            })
    return rows, row_nrm_scores


def _interval(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _group_bootstrap(values, seed_offset: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True)
    return _interval(np.mean(sampled, axis=1))


def _row_bootstrap(
    y, scores, *, cluster=None, seed_offset=0
) -> dict[str, list[float]]:
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    methods = tuple(name for name in scores if name != "iu")
    draws = {name: np.empty(BOOTSTRAP_DRAWS, dtype=float) for name in methods}
    if cluster is None:
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        for draw in range(BOOTSTRAP_DRAWS):
            indices = np.concatenate((
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ))
            baseline = roc_auc_score(y[indices], scores["iu"][indices])
            for name in methods:
                draws[name][draw] = 100 * (
                    roc_auc_score(y[indices], scores[name][indices]) - baseline
                )
    else:
        unique, inverse = np.unique(np.asarray(cluster).astype(str), return_inverse=True)
        for draw in range(BOOTSTRAP_DRAWS):
            counts = rng.multinomial(
                len(unique), np.full(len(unique), 1 / len(unique))
            )
            weights = counts[inverse]
            baseline = roc_auc_score(y, scores["iu"], sample_weight=weights)
            for name in methods:
                draws[name][draw] = 100 * (
                    roc_auc_score(y, scores[name], sample_weight=weights) - baseline
                )
    return {name: _interval(values) for name, values in draws.items()}


def summarize_domains(
    args, rows: list[dict], isolation: dict,
    scores_by_cell: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray], row_nrm_scores: dict[str, np.ndarray],
) -> dict:
    summaries = {}
    for domain_index, domain in enumerate(DOMAIN_ORDER):
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        methods = {}
        for method in SCORE_METHODS:
            group_values = np.asarray([
                np.mean([
                    row["delta_vs_iu_pp"] for row in selected
                    if row["group"] == group and row["method"] == method
                ]) for group in groups
            ], dtype=float)
            methods[method] = {
                "equal_group_delta_pp": float(np.mean(group_values)),
                "ci_pp": _group_bootstrap(
                    group_values, 100 * domain_index + SCORE_METHODS.index(method)
                ),
                "positive_groups": int(np.sum(group_values > 0)),
                "worst_group_pp": float(np.min(group_values)),
                "group_values_pp": dict(zip(groups, map(float, group_values))),
            }
        nrm_values = np.asarray([
            np.mean([
                row["family_nrm_delta_pp"] for row in selected
                if row["group"] == group and row["method"] == "iu"
            ]) for group in groups
        ], dtype=float)
        methods["family_nrm"] = {
            "equal_group_delta_pp": float(np.mean(nrm_values)),
            "ci_pp": _group_bootstrap(nrm_values, 100 * domain_index + 91),
            "positive_groups": int(np.sum(nrm_values > 0)),
            "worst_group_pp": float(np.min(nrm_values)),
            "group_values_pp": dict(zip(groups, map(float, nrm_values))),
        }
        summaries[domain] = {"groups": groups, "methods": methods}

    roster = {row["cell"]: row for row in isolation["roster"]}
    for domain, cell, clustered in (
        ("prmbench", "prmbench", True),
        ("hle", "hle", False),
    ):
        entry = roster[cell]
        isolated = validate_isolated_file(isolated_path(args.out, cell), entry)
        bootstrap_scores = {
            **scores_by_cell[cell],
            "family_nrm": row_nrm_scores[cell],
        }
        intervals = _row_bootstrap(
            labels[cell], bootstrap_scores,
            cluster=isolated.get("source_ids") if clustered else None,
            seed_offset=700 if clustered else 800,
        )
        for method, interval in intervals.items():
            summaries[domain]["methods"][method]["ci_pp"] = interval
    return summaries


def make_transfer_plot(out: Path, summaries: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "PB Llama", "PB Qwen", "SemGrad", "PRMBench", "HLE"
    ]
    x = np.arange(len(DOMAIN_ORDER))
    colors = {
        "canonical": "#3366cc",
        "label_free": "#109618",
        "supervised_one_se": "#ff9900",
        "supervised_max_mean": "#dc3912",
        "family_nrm": "#666666",
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    for axis, suffix, title in (
        (axes[0], "", "Frozen full actuator"),
        (axes[1], "_cross", "Matched cross-only actuator"),
    ):
        for base in BASE_METHODS:
            method = f"{base}{suffix}"
            values = [
                summaries[domain]["methods"][method]["equal_group_delta_pp"]
                for domain in DOMAIN_ORDER
            ]
            axis.plot(
                x, values, marker="o", linewidth=1.8,
                color=colors[base], label=base.replace("_", " "),
            )
        nrm = [
            summaries[domain]["methods"]["family_nrm"]["equal_group_delta_pp"]
            for domain in DOMAIN_ORDER
        ]
        axis.plot(
            x, nrm, marker="s", linestyle="--", color=colors["family_nrm"],
            label="Family-NRM",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("AUROC delta vs IU (pp)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(ncol=3, fontsize=8, loc="best")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Historically opened external domain")
    fig.suptitle(
        "Graph geometry selection: frozen retrospective transfer",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "plot_07_frozen_transfer.png", dpi=180)
    fig.savefig(out / "plot_07_frozen_transfer.pdf")
    plt.close(fig)


def report(args) -> None:
    manifest, isolation, selection, scores_by_cell = verify_fit(args)
    nrm_hashes = verify_comparison_references(args)
    # No function above this line indexes any target outcome field.
    labels, label_hashes = load_labels_after_verification(args, isolation)
    rows, row_nrm_scores = metric_rows(
        args, isolation, selection, scores_by_cell, labels
    )
    write_csv(args.out / "cell_metrics.csv", rows)
    summaries = summarize_domains(
        args, rows, isolation, scores_by_cell, labels, row_nrm_scores
    )
    make_transfer_plot(args.out, summaries)
    result = {
        "version": VERSION,
        "status": "RETROSPECTIVE_STRESS_TEST_COMPLETE",
        "scope": "historically_opened_external_data_not_confirmation",
        "fit_manifest_hash": manifest["manifest_hash"],
        "transfer_selection_hash": selection["selection_hash"],
        "canonical_scores_exactly_reproduced": True,
        "scores_verified_before_outcome_access": True,
        "target_score_fit_was_physically_outcome_free": True,
        "label_source_hashes": label_hashes,
        "family_nrm_source_hashes": nrm_hashes,
        "summaries": summaries,
    }
    write_json(args.out / "RESULT.json", result)
    lines = [
        "# Graph Geometry Selection Research V1 — frozen external transfer",
        "",
        "**Retrospective stress test on historically opened outcomes; not independent confirmation.**",
        "",
        "Every telemetry-derived score and row identifier was physically isolated, frozen, and hash-verified before this report opened an outcome field. The fixed canonical full and matched-cross scores reproduce the prior frozen external arrays exactly in every cell.",
        "",
        "| domain | canonical | label-free | supervised one-SE | supervised max-mean | Family-NRM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        methods = summaries[domain]["methods"]
        lines.append(
            f"| `{domain}` | "
            f"{methods['canonical']['equal_group_delta_pp']:+.3f} | "
            f"{methods['label_free']['equal_group_delta_pp']:+.3f} | "
            f"{methods['supervised_one_se']['equal_group_delta_pp']:+.3f} | "
            f"{methods['supervised_max_mean']['equal_group_delta_pp']:+.3f} | "
            f"{methods['family_nrm']['equal_group_delta_pp']:+.3f} |"
        )
    lines += [
        "",
        "Matched-cross scores were frozen for every selector as a separate actuator control; no outcome-facing selector chose between full and cross.",
        "",
        "The score fit was physically target-free, but the preceding isolation process necessarily unpickled historically opened ProcessBench, SemGrad, and PRMBench caches. It accessed only the registered telemetry, identifier, and non-target eligibility fields.",
        "",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "canonical_exact": True,
        "selection_hash": selection["selection_hash"],
        "domains": {
            domain: summaries[domain]["methods"]["label_free"]["equal_group_delta_pp"]
            for domain in DOMAIN_ORDER
        },
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("isolate", "fit", "report"))
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prm-raw", type=Path, default=DEFAULT_PRM_RAW)
    parser.add_argument("--hle-raw", type=Path, default=DEFAULT_HLE_RAW)
    parser.add_argument("--prm-nrm", type=Path, default=DEFAULT_PRM_NRM)
    parser.add_argument("--hle-nrm", type=Path, default=DEFAULT_HLE_NRM)
    parser.add_argument("--hle-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--hle-judge-manifest", type=Path, default=DEFAULT_JUDGE_MANIFEST
    )
    args = parser.parse_args()
    if args.phase == "isolate":
        isolate(args)
    elif args.phase == "fit":
        fit(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
