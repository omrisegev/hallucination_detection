#!/usr/bin/env python3
"""Freeze the P3D member roster and inference contract before labels open."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as run  # noqa: E402


def main() -> None:
    if run.REGISTRY.exists():
        raise FileExistsError(run.REGISTRY)
    release = p1.DEFAULT_RELEASE.resolve()
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    names = None
    families = None
    cells = []
    for cell_id in p2r.PB_CELLS:
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        _, _, current_names, current_families = run._member_matrix(cell)
        if names is None:
            names, families = current_names, current_families
        if current_names != names or current_families != families:
            raise RuntimeError(f"member roster differs in {cell_id}")
        cells.append({
            "cell_id": cell_id,
            "prepared_input": str(input_path),
            "prepared_input_sha256": sha256_file(input_path),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
        })
    assert names is not None and families is not None
    registry = {
        "schema": "reasoning-localization-p3d-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": run.EXPERIMENT_ID,
        "variant_order": list(run.VARIANT_IDS),
        "release_root": str(release),
        "cells": cells,
        "member_names": list(names),
        "member_families": list(families),
        "n_member_views": len(names),
        "fit_contract": {
            "n_grouped_folds": run.N_FOLDS,
            "fold_assignment": "sha256(row_id) modulo 5 from token_local_fusion._row_folds",
            "held_responses": "projection-only",
            "fit_token_subset": "incumbent frozen uniform token cap, excluding held fold",
            "standardization": "donor-fold median imputation, mean and population std",
            "orientation": "donor-only correlation with equal-view confidence anchor",
            "labels_seen": False,
        },
        "fullpool_config": run.FULLPOOL_CONFIG,
        "deployed_config": run.DEPLOYED_CONFIG,
        "no_prune_alias_config": run.NO_PRUNE_ALIAS_CONFIG,
        "random_mask_seeds": list(run.RANDOM_MASK_SEEDS),
        "random_mask_rule": "for each cell/fold/seed, sha256-derived RNG chooses n_keep views uniformly without replacement",
        "random_mask_aggregation": "arithmetic mean of 20 per-mask metrics and paired bootstrap draws; no best-mask selection",
        "primary_contrasts": [list(pair) for pair in run.PRIMARY_CONTRASTS],
        "multiplicity_family_size": run.MULTIPLICITY_FAMILY_SIZE,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "promotion_bounds": {
            "practical_benefit": run.PRACTICAL_BENEFIT,
            "practical_harm": run.PRACTICAL_HARM,
            "exact_error_floor": run.EXACT_FLOOR,
            "clean_abstention_floor": run.CLEAN_FLOOR,
            "worst_cell_floor": run.WORST_CELL_FLOOR,
            "minimum_cell_mean_pairwise_mask_jaccard": run.MASK_MEAN_JACCARD_FLOOR,
        },
        "runner_sha256": sha256_file(Path(run.__file__).resolve()),
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_DEPLOYED_UPCR_PRUNE_REFIT_V1.md",
        "labels_opened": False,
    }
    run.ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run.REGISTRY, registry)
    print(json.dumps({
        "status": registry["status"],
        "registry": str(run.REGISTRY),
        "n_member_views": len(names),
        "member_names": list(names),
        "runner_sha256": registry["runner_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
