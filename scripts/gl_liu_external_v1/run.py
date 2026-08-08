#!/usr/bin/env python3
"""External scorer-family confirmation for GL-LIU v1 vs unified core-five DUFS-LIU.

Frozen v1 (`docs/methods/gl_liu_v1.md`, `results/ours_only_localization_v1/selection.json`)
and the unified core-five candidate (`results/gl_liu_factorial_v2/`) were both selected on
Qwen3-4B GSM8K/MATH and confirmed on six Qwen3 cells. This runner tests a genuinely new
scorer family (currently: Llama-3.1-8B-Instruct) with ZERO selection performed on the new
family's own labels — the control/candidate NAMES below are copied from the frozen
artifacts, never recomputed. `docs/research_notes/external_data_collection_plan_2026.md`'s
mandatory competitor gate is `cluster/manifests/pb_llama31_8b_external_v1.json`.

Everything this script needs (`answer_detectors`, `token_scores`, `mindgap_control`,
`token_locators`, `load_rows`) is reused UNCHANGED from `scripts/gl_liu_v1/run.py` — the
same functions that produced the frozen v1 numbers — plus the six transparent locator
baselines in `token_baselines.py`, which neither v1 nor the factorial study ever ran.

Label boundary: every detector/locator score is fit and hashed (`--freeze`) before
`row["label"]` is read, exactly as in `gl_liu_v1/run.py` and `gl_liu_factorial_v2/run.py`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.gl_liu_v1.run import (  # noqa: E402
    DEV, answer_detectors, load_rows, mindgap_control, token_locators, token_scores,
)
from scripts.gl_liu_v1.two_stage_localization import evaluate_two_stage  # noqa: E402

from token_baselines import all_locators as baseline_locators  # noqa: E402

FROZEN_V1_SELECTION = ROOT / "results" / "ours_only_localization_v1" / "selection.json"
CANDIDATE_DETECTOR = "answer_dufs_liu_mixed"
CANDIDATE_LOCATOR = "token_dufs_liu_l0p3"


def _score_hash(values) -> str:
    packed = np.concatenate([np.asarray(row, dtype=float) for row in values]).astype("<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def load_frozen_selection() -> dict:
    with open(FROZEN_V1_SELECTION) as handle:
        selection = json.load(handle)
    for pair in selection["development_cells"]:
        assert tuple(pair) in DEV, (
            f"frozen selection.json's development_cells {pair} is not in gl_liu_v1.run.DEV "
            f"{DEV} -- the frozen artifact and the imported module have diverged"
        )
    return selection


def evaluate_cell(model_tag: str, subset: str, path: str, selection: dict) -> dict:
    rows = load_rows(path)
    labels = np.asarray([row["label"] for row in rows], dtype=int)

    detectors, answer_diag = answer_detectors(rows)
    curves, token_diag = token_scores(rows)
    locators = token_locators(curves, rows)
    locators.update(baseline_locators(rows))
    paper_detector, paper_locator = mindgap_control(rows)

    systems = {
        "gl_liu_v1_frozen": (
            detectors[selection["selected_detector"]],
            locators[selection["selected_locator"]],
        ),
        "unified_core_five_dufs": (
            detectors[CANDIDATE_DETECTOR],
            locators[CANDIDATE_LOCATOR],
        ),
        "mindgap_control": (paper_detector, paper_locator),
    }
    for name in [n for n in locators if n.startswith("baseline_")]:
        systems[f"candidate_detector__{name}"] = (detectors[CANDIDATE_DETECTOR], locators[name])

    score_hashes = {
        "detectors": {
            selection["selected_detector"]: _score_hash([detectors[selection["selected_detector"]]]),
            CANDIDATE_DETECTOR: _score_hash([detectors[CANDIDATE_DETECTOR]]),
        },
        "locators": {name: hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()
                      for name, value in locators.items()},
        "mindgap_detector": _score_hash([paper_detector]),
        "mindgap_locator": hashlib.sha256(
            np.asarray(paper_locator, dtype="<i8").tobytes()).hexdigest(),
    }

    per_system = []
    for name, (risk, locator) in systems.items():
        metrics = evaluate_two_stage(risk, locator, labels)
        per_system.append({"model": model_tag, "subset": subset, "system": name, **metrics})

    return {
        "model": model_tag, "subset": subset, "n_rows": len(rows),
        "per_system": per_system,
        "diag": {
            "answer": answer_diag, "token": token_diag,
            "score_hashes_before_labels": score_hashes,
            "labels_seen_during_fit": False,
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True,
                        help="path to a cells_*.json manifest (see cells_llama31_8b.json)")
    parser.add_argument("--cache-root", default=None,
                        help="overrides the manifest's cache_root_default")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.cells) as handle:
        cells_def = json.load(handle)
    cache_root = args.cache_root or cells_def["cache_root_default"]
    model_tag = cells_def["model_tag"]

    selection = load_frozen_selection()
    print(f"FROZEN v1 selection: detector={selection['selected_detector']!r} "
          f"locator={selection['selected_locator']!r}", flush=True)
    print(f"CANDIDATE (unified core-five): detector={CANDIDATE_DETECTOR!r} "
          f"locator={CANDIDATE_LOCATOR!r}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    all_per_system, all_hashes = [], {}
    for subset in cells_def["subsets"]:
        rel = cells_def["cell_path_pattern"].format(model_tag=model_tag, subset=subset)
        path = os.path.join(cache_root, rel)
        print(f"FIT {model_tag} {subset}  <-  {path}", flush=True)
        cell = evaluate_cell(model_tag, subset, path, selection)
        all_per_system.extend(cell["per_system"])
        all_hashes[f"{model_tag}__{subset}"] = cell["diag"]["score_hashes_before_labels"]
        with open(os.path.join(args.out_dir, f"{model_tag}__{subset}__diagnostics.json"), "w") as handle:
            json.dump(cell["diag"], handle, indent=2, default=str)

    write_csv(os.path.join(args.out_dir, "external_systems_per_cell.csv"), all_per_system)

    macro = {}
    for row in all_per_system:
        macro.setdefault(row["system"], []).append(row["f1"])
    macro_summary = [{"system": name, "n_cells": len(vals), "macro_f1": float(np.mean(vals))}
                     for name, vals in macro.items()]
    macro_summary.sort(key=lambda r: r["macro_f1"], reverse=True)
    write_csv(os.path.join(args.out_dir, "external_macro_f1.csv"), macro_summary)

    with open(os.path.join(args.out_dir, "FREEZE_MANIFEST.json"), "w") as handle:
        json.dump({
            "model_tag": model_tag, "cells_manifest": args.cells,
            "frozen_v1_selection": selection,
            "candidate_detector": CANDIDATE_DETECTOR, "candidate_locator": CANDIDATE_LOCATOR,
            "score_hashes_before_labels": all_hashes,
            "note": "Hashes recorded above were computed before any row['label'] access in "
                    "this run. This file documents that the freeze happened; it is not "
                    "itself a security artifact.",
        }, handle, indent=2, default=str)

    print("\nMacro F1 by system:", flush=True)
    for row in macro_summary:
        print(f"  {row['system']:36s} {100*row['macro_f1']:6.2f}  (n={row['n_cells']})", flush=True)
    print(f"\n{os.path.join(args.out_dir, 'external_systems_per_cell.csv')}")


if __name__ == "__main__":
    main()
