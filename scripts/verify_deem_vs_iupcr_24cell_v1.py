#!/usr/bin/env python3
"""Verify resume-evaluation and fresh-Stage-A rebuild evidence for B0-B3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402

FILES = ("PER_FIT_METRICS.csv", "PER_CELL_METRICS.csv", "SEED_STABILITY.csv",
         "PAIRWISE_COMPARISONS.csv", "FAMILY_SUMMARY.json", "BOOTSTRAP.json",
         "WHOLE_SEARCH_NULL.json", "DECISION.json")


def semantic_npz_manifest(root: Path) -> dict:
    import numpy as np
    output = {}
    for path in sorted((root / "fits").glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            output[path.relative_to(root).as_posix()] = canonical_sha256(
                {name: data[name] for name in sorted(data.files) if not name.startswith("state__")}
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--fresh-run-dir", type=Path, required=True)
    parser.add_argument("--original-evaluation-dir", type=Path, required=True)
    parser.add_argument("--resume-evaluation-dir", type=Path, required=True)
    parser.add_argument("--fresh-evaluation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    resume_match = {name: sha256_file(args.original_evaluation_dir / name) == sha256_file(args.resume_evaluation_dir / name)
                    for name in FILES}
    fresh_match = {name: sha256_file(args.original_evaluation_dir / name) == sha256_file(args.fresh_evaluation_dir / name)
                   for name in FILES}
    stage_match = semantic_npz_manifest(args.original_run_dir) == semantic_npz_manifest(args.fresh_run_dir)
    passed = all(resume_match.values()) and all(fresh_match.values()) and stage_match
    value = {"schema": "deem_vs_iupcr_rebuild_verification_v1",
             "status": "pass" if passed else "REBUILD_VERIFICATION_FAILURE",
             "resume_evaluation_byte_match": resume_match,
             "fresh_evaluation_byte_match": fresh_match,
             "fresh_stage_a_semantic_match": stage_match}
    value["content_sha256"] = canonical_sha256(value)
    atomic_write_json(args.out, value)
    if not passed:
        raise SystemExit("REBUILD_VERIFICATION_FAILURE")


if __name__ == "__main__":
    main()
