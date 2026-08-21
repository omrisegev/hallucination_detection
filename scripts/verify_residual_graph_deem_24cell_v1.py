#!/usr/bin/env python3
"""Verify resume and genuinely fresh deterministic rebuilds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file


COMPACT = (
    "PER_FIT.csv", "PER_CELL.csv", "FAMILY_SUMMARY.csv",
    "CONTRIBUTION_RECONSTRUCTION.csv", "RESIDUAL_DIAGNOSTICS.csv",
    "NUISANCE_DIAGNOSTICS.csv", "SENSITIVITY.csv", "LAMBDA_SENSITIVITY.csv",
    "GRAPH_HEALTH.csv", "GATE_STABILITY.csv", "CONDITIONAL_GEOMETRY.csv",
    "CONTROLS.json", "WHOLE_SEARCH_NULL.json", "PAIRWISE_COMPARISONS.csv",
    "BOOTSTRAP.json", "SEED_STABILITY.json", "DECISION.json",
    "EVALUATION_COMPLETE.json",
)


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"rebuild destination must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True)
    if completed.returncode:
        raise RuntimeError(f"rebuild command failed ({completed.returncode}): {' '.join(command)}")


def evaluation_command(args, run_dir: Path, out_dir: Path) -> list[str]:
    command = [
        args.python, str(ROOT / "scripts/evaluate_residual_graph_deem_24cell_v1.py"),
        "--run-dir", str(run_dir), "--bundle-dir", str(args.bundle_dir),
        "--sidecar-dir", str(args.sidecar_dir), "--phase0-complete", str(args.phase0_complete),
        "--registry", str(args.registry), "--out-dir", str(out_dir), "--B", str(args.B),
    ]
    if args.B == 999:
        command.extend(["--promotion-decision", str(args.promotion_decision)])
    return command


def semantic_npz_hash(path: Path) -> str:
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return canonical_sha256({key: data[key] for key in sorted(data.files)})


def compare_evaluations(reference: Path, candidate: Path) -> dict:
    def normalized(path: Path, name: str) -> bytes:
        if name == "PER_FIT.csv":
            with path.open(newline="", encoding="utf-8") as handle:
                content = list(csv.DictReader(handle))
            for row in content:
                row.pop("runtime_seconds", None)
            return canonical_sha256(content).encode("ascii")
        if name == "EVALUATION_COMPLETE.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            value.pop("score_freeze_sha256", None)
            return canonical_sha256(value).encode("ascii")
        return path.read_bytes()

    rows = {}
    all_match = True
    for name in COMPACT:
        left, right = reference / name, candidate / name
        present = left.is_file() and right.is_file()
        match = bool(present and normalized(left, name) == normalized(right, name))
        rows[name] = {
            "present": present,
            "reference_sha256": sha256_file(left) if left.is_file() else None,
            "candidate_sha256": sha256_file(right) if right.is_file() else None,
            "match": match,
        }
        all_match &= match
    return {"all_match": all_match, "files": rows}


def compare_score_arrays(reference_run: Path, fresh_run: Path) -> dict:
    reference = sorted((reference_run / "fits").glob("*/*.npz"))
    rows = {}
    all_match = True
    for left in reference:
        relative = left.relative_to(reference_run)
        right = fresh_run / relative
        match = right.is_file() and semantic_npz_hash(left) == semantic_npz_hash(right)
        rows[str(relative)] = {
            "reference_semantic_sha256": semantic_npz_hash(left),
            "fresh_semantic_sha256": semantic_npz_hash(right) if right.is_file() else None,
            "match": bool(match),
        }
        all_match &= bool(match)
    extra = sorted(str(path.relative_to(fresh_run)) for path in (fresh_run / "fits").glob("*/*.npz") if not (reference_run / path.relative_to(fresh_run)).is_file())
    all_match &= not extra
    return {"all_match": all_match, "artifacts": rows, "extra": extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--original-evaluation-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--phase0-complete", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--resume-evaluation-dir", type=Path, required=True)
    parser.add_argument("--fresh-run-dir", type=Path, required=True)
    parser.add_argument("--fresh-evaluation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--adapter-device", default="auto")
    parser.add_argument("--B", type=int, choices=(199, 999), default=199)
    parser.add_argument("--promotion-decision", type=Path)
    parser.add_argument("--use-existing", action="store_true",
                        help="verify already-created rebuild directories without rerunning")
    args = parser.parse_args()
    original_run = args.original_run_dir.resolve()
    original_eval = args.original_evaluation_dir.resolve()
    resume_eval = args.resume_evaluation_dir.resolve()
    fresh_run = args.fresh_run_dir.resolve()
    fresh_eval = args.fresh_evaluation_dir.resolve()
    try:
        original_decision = json.loads(
            (original_eval / "DECISION.json").read_text(encoding="utf-8")
        ).get("primary_decision")
        if not args.use_existing:
            require_empty(resume_eval); require_empty(fresh_run); require_empty(fresh_eval)
            # Resume path: re-evaluate only after the existing checkpoint/freeze is
            # revalidated by the Stage-A runner's immutable-resume route.
            run([
                args.python, str(ROOT / "scripts/run_residual_graph_deem_24cell_v1.py"),
                "stage-a", "--out-dir", str(original_run), "--bundle-dir", str(args.bundle_dir),
                "--phase0-complete", str(args.phase0_complete), "--registry", str(args.registry),
                "--python", args.python, "--adapter-device", args.adapter_device,
            ])
            run(evaluation_command(args, original_run, resume_eval))
            # Fresh path recomputes every target-free fit into a new directory.
            run([
                args.python, str(ROOT / "scripts/run_residual_graph_deem_24cell_v1.py"),
                "stage-a", "--out-dir", str(fresh_run), "--bundle-dir", str(args.bundle_dir),
                "--phase0-complete", str(args.phase0_complete), "--registry", str(args.registry),
                "--python", args.python, "--adapter-device", args.adapter_device,
            ])
            run(evaluation_command(args, fresh_run, fresh_eval))
        resume = compare_evaluations(original_eval, resume_eval)
        fresh = compare_evaluations(original_eval, fresh_eval)
        scores = compare_score_arrays(original_run, fresh_run)
        passed = bool(resume["all_match"] and fresh["all_match"] and scores["all_match"])
        value = {
            "schema": "residual_graph_deem_rebuild_verification_v1",
            "status": "pass" if passed else "REBUILD_VERIFICATION_FAILURE",
            "primary_decision": (
                original_decision if passed else "REBUILD_VERIFICATION_FAILURE"
            ),
            "resume": resume,
            "fresh": fresh,
            "fresh_score_semantic_hashes": scores,
            "original_score_freeze_sha256": sha256_file(original_run / "SCORE_FREEZE_MANIFEST.json"),
            "fresh_score_freeze_sha256": sha256_file(fresh_run / "SCORE_FREEZE_MANIFEST.json"),
        }
    except Exception as exc:
        value = {
            "schema": "residual_graph_deem_rebuild_verification_v1",
            "status": "REBUILD_VERIFICATION_FAILURE",
            "primary_decision": "REBUILD_VERIFICATION_FAILURE",
            "error_type": type(exc).__name__, "error": str(exc),
        }
    value["content_sha256"] = canonical_sha256(value)
    atomic_write_json(args.out, value)
    if value["status"] != "pass":
        raise SystemExit("REBUILD_VERIFICATION_FAILURE")


if __name__ == "__main__":
    main()
