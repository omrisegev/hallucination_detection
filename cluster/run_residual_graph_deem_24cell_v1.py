#!/usr/bin/env python3
"""AIRCC orchestration for the frozen Residual-Graph DEEM v1 pipeline.

Large raw sources and artifacts move only between AIRCC shared storage and the
registered Drive prefix.  Every stage first verifies the immutable remote run
identity, so resume cannot silently attach to another run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file
from spectral_utils.residual_graph_deem_data import load_registry
from scripts.run_residual_graph_deem_24cell_v1 import source_hash


SHARED = Path("/shared/cycle2_tau_averbuch_prj/omrisegev1")
DEFAULT_RUN = SHARED / "results/residual_graph_deem_24cell_v1"
DEFAULT_REMOTE = "gdrive:hallucination_detection/cluster_results/residual_graph_deem_24cell_v1"
DEFAULT_RCLONE = SHARED / "bin/rclone"
MANIFEST = ROOT / "cluster/residual_graph_deem_24cell_v1_manifest.json"
REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
PROTOCOL = ROOT / "docs/experiments/RESIDUAL_GRAPH_DEEM_24CELL_V1.md"


def command(args, *, check=True, capture=False):
    completed = subprocess.run(
        [str(value) for value in args], check=False, text=True,
        capture_output=capture,
    )
    if check and completed.returncode:
        if capture:
            sys.stderr.write(completed.stdout + completed.stderr)
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(map(str, args))}")
    return completed


def sync_stamp() -> dict:
    path = ROOT / "SYNC_COMMIT.json"
    if not path.is_file():
        raise RuntimeError("SYNC_COMMIT.json is required on AIRCC")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dirty"):
        raise RuntimeError("full experiment refuses a dirty synchronized tree")
    return value


def run_identity() -> dict:
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = load_registry(REGISTRY)
    if registry["registry_content_sha256"] != frozen["registry_content_sha256"]:
        raise RuntimeError("AIRCC manifest/registry mismatch")
    stamp = sync_stamp()
    value = {
        "schema": "residual_graph_deem_remote_identity_v1",
        "experiment_id": frozen["experiment_id"],
        "git": {"commit": stamp["commit"], "dirty": stamp["dirty"]},
        "required_base": frozen["required_base"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "registry_file_sha256": sha256_file(REGISTRY),
        "registry_content_sha256": registry["registry_content_sha256"],
        "code_sha256": source_hash(),
        "aircc_manifest_sha256": sha256_file(MANIFEST),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def rclone(args, *extra, check=True, capture=False):
    return command([args.rclone, *extra], check=check, capture=capture)


def guard_remote(args) -> dict:
    identity = run_identity()
    local = args.run_root / "RUN_IDENTITY.json"
    if local.is_file():
        existing = json.loads(local.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("local run identity mismatch")
    else:
        atomic_write_json(local, identity)
    remote_path = f"{args.remote_root}/RUN_IDENTITY.json"
    observed = rclone(args, "cat", remote_path, check=False, capture=True)
    if observed.returncode == 0:
        if json.loads(observed.stdout) != identity:
            raise RuntimeError("Drive prefix belongs to a different run")
    else:
        rclone(args, "copyto", str(local), remote_path, "--immutable")
    return identity


def upload(args, local: Path, remote_suffix: str) -> None:
    if not local.exists():
        raise FileNotFoundError(local)
    destination = f"{args.remote_root}/{remote_suffix}".rstrip("/")
    if local.is_dir():
        rclone(args, "copy", str(local), destination, "--checksum", "--transfers", "4", "--checkers", "8")
    else:
        rclone(args, "copyto", str(local), destination, "--checksum")


def download_tree(args, remote_suffix: str, local: Path) -> None:
    if local.exists() and any(local.iterdir()):
        return
    local.mkdir(parents=True, exist_ok=True)
    rclone(args, "copy", f"{args.remote_root}/{remote_suffix}", str(local), "--checksum")


def phase0(args) -> int:
    guard_remote(args)
    out = args.run_root / "phase0"
    completed = command([
        args.python, ROOT / "scripts/run_residual_graph_deem_24cell_v1.py",
        "phase0", "--registry", REGISTRY, "--out-dir", out,
    ], check=False)
    upload(args, out, "phase0")
    return completed.returncode


def bundles(args) -> int:
    guard_remote(args)
    registry = load_registry(REGISTRY)
    raw_root = args.run_root / "raw_sources"
    data_root = args.run_root / "data"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for cell in registry["cells"]:
        cell_id = cell["cell_id"]
        source = manifest["source_overrides"].get(
            cell_id, manifest["source_default"].format(cell_id=cell_id)
        )
        local = raw_root / cell_id
        local.mkdir(parents=True, exist_ok=True)
        rclone(args, "copy", source, str(local), "--immutable", "--checksum")
        command([
            args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py",
            "bundles", "--registry", REGISTRY, "--repo-root", ROOT,
            "--source-root", raw_root, "--cells", cell_id, "--out-dir", data_root,
        ])
        upload(args, data_root / "bundles" / f"{cell_id}.npz", f"data/bundles/{cell_id}.npz")
        upload(args, data_root / "bundles" / f"{cell_id}.manifest.json", f"data/bundles/{cell_id}.manifest.json")
    command([
        args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py",
        "finalize-bundles", "--registry", REGISTRY,
        "--bundle-dir", data_root / "bundles",
        "--out", data_root / "TARGET_FREE_BUNDLES.json",
    ])
    upload(args, data_root / "TARGET_FREE_BUNDLES.json", "data/TARGET_FREE_BUNDLES.json")
    return 0


def ensure_stage_inputs(args) -> None:
    phase = args.run_root / "phase0"
    bundles_dir = args.run_root / "data/bundles"
    if not (phase / "PHASE0_COMPLETE.json").is_file():
        download_tree(args, "phase0", phase)
    if not (args.run_root / "data/TARGET_FREE_BUNDLES.json").is_file():
        download_tree(args, "data", args.run_root / "data")
    if not bundles_dir.is_dir():
        raise RuntimeError("target-free bundles unavailable")


def stage_a(args, *, fresh=False) -> int:
    guard_remote(args); ensure_stage_inputs(args)
    out = args.run_root / ("rebuild/fresh_stage_a" if fresh else "stage_a")
    completed = command([
        args.python, ROOT / "scripts/run_residual_graph_deem_24cell_v1.py",
        "stage-a", "--registry", REGISTRY, "--out-dir", out,
        "--bundle-dir", args.run_root / "data/bundles",
        "--phase0-complete", args.run_root / "phase0/PHASE0_COMPLETE.json",
        "--python", args.python, "--adapter-device", "cuda",
    ], check=False)
    upload(args, out, "rebuild/fresh_stage_a" if fresh else "stage_a")
    return completed.returncode


def evaluate(args, *, B: int, fresh=False, resume=False) -> int:
    guard_remote(args); ensure_stage_inputs(args)
    run_dir = args.run_root / ("rebuild/fresh_stage_a" if fresh else "stage_a")
    if not (run_dir / "SCORE_FREEZE_MANIFEST.json").is_file():
        download_tree(args, "rebuild/fresh_stage_a" if fresh else "stage_a", run_dir)
    sidecars = args.run_root / "evaluation/label_sidecars"
    if not sidecars.is_dir() or not (sidecars / "LABEL_SIDECARS.json").is_file():
        if fresh:
            download_tree(args, "evaluation/label_sidecars", sidecars)
        else:
            command([
                args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py",
                "sidecars", "--registry", REGISTRY, "--repo-root", ROOT,
                "--source-root", args.run_root / "raw_sources",
                "--bundle-dir", args.run_root / "data/bundles",
                "--score-freeze-manifest", run_dir / "SCORE_FREEZE_MANIFEST.json",
                "--out-dir", sidecars,
            ])
            upload(args, sidecars, "evaluation/label_sidecars")
    if resume:
        out = args.run_root / "rebuild/resume_evaluation"
    elif fresh:
        out = args.run_root / "rebuild/fresh_evaluation"
    else:
        out = args.run_root / f"evaluation/B{B}"
    command_args = [
        args.python, ROOT / "scripts/evaluate_residual_graph_deem_24cell_v1.py",
        "--run-dir", run_dir, "--bundle-dir", args.run_root / "data/bundles",
        "--sidecar-dir", sidecars, "--phase0-complete", args.run_root / "phase0/PHASE0_COMPLETE.json",
        "--registry", REGISTRY, "--out-dir", out, "--B", str(B),
    ]
    if B == 999:
        command_args.extend(["--promotion-decision", args.run_root / "evaluation/B199/DECISION.json"])
    completed = command(command_args, check=False)
    upload(args, out, str(out.relative_to(args.run_root)))
    return completed.returncode


def report(args, *, B: int) -> int:
    guard_remote(args)
    evaluation = args.run_root / f"evaluation/B{B}"
    out = args.run_root / f"report/B{B}"
    command([
        args.python, ROOT / "scripts/plot_residual_graph_deem_24cell_v1.py",
        "--run-dir", args.run_root / "stage_a", "--evaluation-dir", evaluation,
        "--bundle-dir", args.run_root / "data/bundles",
        "--sidecar-dir", args.run_root / "evaluation/label_sidecars",
        "--phase0-dir", args.run_root / "phase0", "--out-dir", out,
    ])
    command([
        args.python, ROOT / "scripts/report_residual_graph_deem_24cell_v1.py",
        "--run-dir", args.run_root / "stage_a", "--evaluation-dir", evaluation,
        "--bundle-dir", args.run_root / "data/bundles",
        "--sidecar-dir", args.run_root / "evaluation/label_sidecars",
        "--phase0-complete", args.run_root / "phase0/PHASE0_COMPLETE.json",
        "--out-dir", out,
    ])
    upload(args, out, f"report/B{B}")
    return 0


def finalize_rebuild(args, *, B: int) -> int:
    guard_remote(args)
    original = args.run_root / f"evaluation/B{B}"
    resume = args.run_root / "rebuild/resume_evaluation"
    fresh = args.run_root / "rebuild/fresh_evaluation"
    output = args.run_root / "rebuild/REBUILD_VERIFICATION.json"
    completed = command([
        args.python, ROOT / "scripts/verify_residual_graph_deem_24cell_v1.py",
        "--original-run-dir", args.run_root / "stage_a",
        "--original-evaluation-dir", original,
        "--bundle-dir", args.run_root / "data/bundles",
        "--sidecar-dir", args.run_root / "evaluation/label_sidecars",
        "--phase0-complete", args.run_root / "phase0/PHASE0_COMPLETE.json",
        "--registry", REGISTRY,
        "--resume-evaluation-dir", resume,
        "--fresh-run-dir", args.run_root / "rebuild/fresh_stage_a",
        "--fresh-evaluation-dir", fresh,
        "--out", output, "--B", str(B), "--use-existing",
    ], check=False)
    upload(args, output, "rebuild/REBUILD_VERIFICATION.json")
    # The final report must reflect rebuild failure as the primary decision.
    report_path = args.run_root / f"report/B{B}"
    command([
        args.python, ROOT / "scripts/report_residual_graph_deem_24cell_v1.py",
        "--run-dir", args.run_root / "stage_a", "--evaluation-dir", original,
        "--bundle-dir", args.run_root / "data/bundles",
        "--sidecar-dir", args.run_root / "evaluation/label_sidecars",
        "--phase0-complete", args.run_root / "phase0/PHASE0_COMPLETE.json",
        "--out-dir", report_path,
    ])
    upload(args, report_path, f"report/B{B}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "phase0", "bundles", "stage-a", "evaluate-199", "evaluate-999",
        "report-199", "report-999", "resume-evaluate", "fresh-stage-a",
        "fresh-evaluate", "finalize-rebuild", "resume-evaluate-999",
        "fresh-evaluate-999", "finalize-rebuild-999",
    ))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE)
    parser.add_argument("--rclone", type=Path, default=DEFAULT_RCLONE)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve(); args.run_root.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "phase0": lambda: phase0(args), "bundles": lambda: bundles(args),
        "stage-a": lambda: stage_a(args), "evaluate-199": lambda: evaluate(args, B=199),
        "evaluate-999": lambda: evaluate(args, B=999), "report-199": lambda: report(args, B=199),
        "report-999": lambda: report(args, B=999),
        "resume-evaluate": lambda: evaluate(args, B=199, resume=True),
        "resume-evaluate-999": lambda: evaluate(args, B=999, resume=True),
        "fresh-stage-a": lambda: stage_a(args, fresh=True),
        "fresh-evaluate": lambda: evaluate(args, B=199, fresh=True),
        "fresh-evaluate-999": lambda: evaluate(args, B=999, fresh=True),
        "finalize-rebuild": lambda: finalize_rebuild(args, B=199),
        "finalize-rebuild-999": lambda: finalize_rebuild(args, B=999),
    }
    raise SystemExit(dispatch[args.stage]())


if __name__ == "__main__":
    main()
