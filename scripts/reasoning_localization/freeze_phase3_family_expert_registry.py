#!/usr/bin/env python3
"""Freeze the P3E roster and inference contract before labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell, validate_fit_manifest  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402
from scripts.reasoning_localization import run_phase3_family_expert_attribution as run  # noqa: E402


def main() -> None:
    if run.REGISTRY.exists():
        raise FileExistsError(run.REGISTRY)
    release = p1.DEFAULT_RELEASE.resolve()
    root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(root / "MANIFEST.json", input_root=root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    names = families = None
    cells = []
    for cell_id in p2r.PB_CELLS:
        source = by_cell[cell_id]
        path = root / source["artifact_path"]
        cell = load_prepared_localization_cell(path, source)
        _, _, current_names, current_families = p3d._member_matrix(cell)
        names = current_names if names is None else names
        families = current_families if families is None else families
        if current_names != names or current_families != families:
            raise RuntimeError("member roster drift")
        cells.append({"cell_id": cell_id, "input_sha256": sha256_file(path), "n_rows": len(cell.row_ids)})
    registry = {"schema": "reasoning-localization-p3e-execution-v1", "status": "FROZEN_BEFORE_RUN", "experiment_id": run.EXPERIMENT, "variant_order": list(run.VARIANTS), "release_root": str(release), "cells": cells, "member_names": list(names), "member_families": list(families), "family_counts": {family: list(families).count(family) for family in sorted(set(families))}, "fit_contract": "five grouped donor folds; held responses projection-only; donor imputation/scaling/IU/orientation", "iu_config": dict(run.IU_CONFIG), "primary_contrasts": [list(pair) for pair in run.PRIMARY], "multiplicity_family_size": run.FAMILY_SIZE, "practical_benefit": run.BENEFIT, "practical_harm": run.HARM, "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED, "runner_sha256": sha256_file(Path(run.__file__).resolve()), "labels_opened": False}
    run.ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run.REGISTRY, registry)
    print(json.dumps({"status": registry["status"], "family_counts": registry["family_counts"], "runner_sha256": registry["runner_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
