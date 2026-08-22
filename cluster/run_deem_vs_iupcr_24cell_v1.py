#!/usr/bin/env python3
"""AIRCC/Drive orchestration for the frozen graph-free DEEM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402
from spectral_utils.residual_graph_deem_data import load_registry  # noqa: E402
from scripts.run_deem_vs_iupcr_24cell_v1 import source_hash  # noqa: E402

SHARED = Path("/shared/cycle2_tau_averbuch_prj/omrisegev1")
DEFAULT_RUN = SHARED / "results/deem_vs_iupcr_24cell_v1"
DEFAULT_REMOTE = "gdrive:hallucination_detection/cluster_results/deem_vs_iupcr_24cell_v1"
DEFAULT_RCLONE = SHARED / "bin/rclone"
MANIFEST = ROOT / "cluster/deem_vs_iupcr_24cell_v1_manifest.json"
CONFIG = ROOT / "configs/deem_vs_iupcr_24cell_v1.json"
REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
PROTOCOL = ROOT / "docs/experiments/DEEM_VS_IUPCR_24CELL_V1.md"


def command(values, *, check=True, capture=False):
    completed = subprocess.run([str(value) for value in values], check=False, text=True,
                               capture_output=capture)
    if check and completed.returncode:
        if capture:
            sys.stderr.write(completed.stdout + completed.stderr)
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(map(str, values))}")
    return completed


def rclone(args, *values, check=True, capture=False):
    return command([args.rclone, *values], check=check, capture=capture)


def run_identity() -> dict:
    stamp_path = ROOT / "SYNC_COMMIT.json"
    if not stamp_path.is_file():
        raise RuntimeError("SYNC_COMMIT.json is required on AIRCC")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    if stamp.get("dirty"):
        raise RuntimeError("full experiment refuses a dirty synchronized tree")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = load_registry(REGISTRY)
    if registry["registry_content_sha256"] != manifest["registry_content_sha256"]:
        raise RuntimeError("manifest/registry mismatch")
    value = {"schema": "deem_vs_iupcr_remote_identity_v1",
             "experiment_id": manifest["experiment_id"], "git": stamp,
             "required_base": manifest["required_base"],
             "protocol_sha256": sha256_file(PROTOCOL), "config_sha256": sha256_file(CONFIG),
             "registry_file_sha256": sha256_file(REGISTRY),
             "registry_content_sha256": registry["registry_content_sha256"],
             "code_sha256": source_hash(), "aircc_manifest_sha256": sha256_file(MANIFEST)}
    value["content_sha256"] = canonical_sha256(value)
    return value


def guard_remote(args) -> None:
    identity = run_identity()
    local = args.run_root / "RUN_IDENTITY.json"
    if local.is_file() and json.loads(local.read_text(encoding="utf-8")) != identity:
        raise RuntimeError("local run identity mismatch")
    if not local.is_file():
        atomic_write_json(local, identity)
    destination = f"{args.remote_root}/RUN_IDENTITY.json"
    observed = rclone(args, "cat", destination, check=False, capture=True)
    if observed.returncode == 0:
        if json.loads(observed.stdout) != identity:
            raise RuntimeError("Drive prefix belongs to a different run")
    else:
        rclone(args, "copyto", str(local), destination, "--immutable")


def upload(args, local: Path, suffix: str) -> None:
    if not local.exists():
        raise FileNotFoundError(local)
    destination = f"{args.remote_root}/{suffix}".rstrip("/")
    if local.is_dir():
        rclone(args, "copy", str(local), destination, "--checksum", "--transfers", "4", "--checkers", "8")
    else:
        rclone(args, "copyto", str(local), destination, "--checksum")


def download(args, suffix: str, local: Path) -> None:
    if local.is_dir() and any(local.iterdir()):
        return
    local.mkdir(parents=True, exist_ok=True)
    rclone(args, "copy", f"{args.remote_root}/{suffix}", str(local), "--checksum")


def preflight(args) -> int:
    guard_remote(args)
    out = args.run_root / "preflight"
    completed = command([args.python, ROOT / "scripts/run_deem_vs_iupcr_24cell_v1.py",
                         "preflight", "--config", CONFIG, "--registry", REGISTRY,
                         "--out-dir", out, "--python", args.python, "--adapter-device", "cuda"], check=False)
    upload(args, out, "preflight")
    return completed.returncode


def bundles(args) -> int:
    guard_remote(args)
    registry = load_registry(REGISTRY)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_root, data_root = args.run_root / "raw_sources", args.run_root / "data"
    for cell in registry["cells"]:
        cell_id = cell["cell_id"]
        source = manifest["source_overrides"].get(cell_id, manifest["source_default"].format(cell_id=cell_id))
        local = raw_root / cell_id; local.mkdir(parents=True, exist_ok=True)
        rclone(args, "copy", source, str(local), "--immutable", "--checksum")
        command([args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py", "bundles",
                 "--registry", REGISTRY, "--repo-root", ROOT, "--source-root", raw_root,
                 "--cells", cell_id, "--out-dir", data_root])
        for suffix in (".npz", ".manifest.json"):
            upload(args, data_root / "bundles" / f"{cell_id}{suffix}", f"data/bundles/{cell_id}{suffix}")
    command([args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py", "finalize-bundles",
             "--registry", REGISTRY, "--bundle-dir", data_root / "bundles",
             "--out", data_root / "TARGET_FREE_BUNDLES.json"])
    upload(args, data_root / "TARGET_FREE_BUNDLES.json", "data/TARGET_FREE_BUNDLES.json")
    return 0


def ensure_inputs(args) -> None:
    if not (args.run_root / "preflight/PREFLIGHT_COMPLETE.json").is_file():
        download(args, "preflight", args.run_root / "preflight")
    if not (args.run_root / "data/TARGET_FREE_BUNDLES.json").is_file():
        download(args, "data", args.run_root / "data")


def stage_a(args, *, fresh=False) -> int:
    guard_remote(args); ensure_inputs(args)
    suffix = "rebuild/fresh_stage_a" if fresh else "stage_a"
    out = args.run_root / suffix
    completed = command([args.python, ROOT / "scripts/run_deem_vs_iupcr_24cell_v1.py", "stage-a",
                         "--config", CONFIG, "--registry", REGISTRY, "--out-dir", out,
                         "--bundle-dir", args.run_root / "data/bundles",
                         "--preflight-complete", args.run_root / "preflight/PREFLIGHT_COMPLETE.json",
                         "--python", args.python, "--adapter-device", "cuda"], check=False)
    upload(args, out, suffix)
    return completed.returncode


def ensure_sidecars(args, run_dir: Path) -> Path:
    sidecars = args.run_root / "evaluation/label_sidecars"
    if (sidecars / "LABEL_SIDECARS.json").is_file():
        return sidecars
    raw_root = args.run_root / "raw_sources"
    if not raw_root.is_dir():
        raise RuntimeError("raw sources unavailable for the post-freeze sidecar boundary")
    command([args.python, ROOT / "scripts/build_residual_graph_deem_data_v1.py", "sidecars",
             "--registry", REGISTRY, "--repo-root", ROOT, "--source-root", raw_root,
             "--bundle-dir", args.run_root / "data/bundles",
             "--score-freeze-manifest", run_dir / "SCORE_FREEZE_MANIFEST.json", "--out-dir", sidecars])
    upload(args, sidecars, "evaluation/label_sidecars")
    return sidecars


def evaluate(args, *, B: int, fresh=False, resume=False) -> int:
    guard_remote(args); ensure_inputs(args)
    run_suffix = "rebuild/fresh_stage_a" if fresh else "stage_a"
    run_dir = args.run_root / run_suffix
    if not (run_dir / "SCORE_FREEZE_MANIFEST.json").is_file():
        download(args, run_suffix, run_dir)
    sidecars = ensure_sidecars(args, run_dir) if not fresh else args.run_root / "evaluation/label_sidecars"
    if not (sidecars / "LABEL_SIDECARS.json").is_file():
        download(args, "evaluation/label_sidecars", sidecars)
    suffix = "rebuild/resume_evaluation" if resume else "rebuild/fresh_evaluation" if fresh else f"evaluation/B{B}"
    out = args.run_root / suffix
    values = [args.python, ROOT / "scripts/evaluate_deem_vs_iupcr_24cell_v1.py",
              "--run-dir", run_dir, "--bundle-dir", args.run_root / "data/bundles",
              "--sidecar-dir", sidecars, "--registry", REGISTRY, "--config", CONFIG,
              "--out-dir", out, "--B", str(B)]
    if B == 999:
        values.extend(["--promotion-decision", args.run_root / "evaluation/B199/DECISION.json"])
    completed = command(values, check=False); upload(args, out, suffix)
    return completed.returncode


def report(args, B: int) -> int:
    guard_remote(args)
    evaluation, out = args.run_root / f"evaluation/B{B}", args.run_root / f"report/B{B}"
    command([args.python, ROOT / "scripts/report_deem_vs_iupcr_24cell_v1.py",
             "--evaluation-dir", evaluation, "--out-dir", out])
    upload(args, out, f"report/B{B}"); return 0


def finalize(args, B: int) -> int:
    guard_remote(args)
    out = args.run_root / "rebuild/REBUILD_VERIFICATION.json"
    completed = command([args.python, ROOT / "scripts/verify_deem_vs_iupcr_24cell_v1.py",
                         "--original-run-dir", args.run_root / "stage_a",
                         "--fresh-run-dir", args.run_root / "rebuild/fresh_stage_a",
                         "--original-evaluation-dir", args.run_root / f"evaluation/B{B}",
                         "--resume-evaluation-dir", args.run_root / "rebuild/resume_evaluation",
                         "--fresh-evaluation-dir", args.run_root / "rebuild/fresh_evaluation",
                         "--out", out], check=False)
    upload(args, out, "rebuild/REBUILD_VERIFICATION.json"); return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "bundles", "stage-a", "evaluate-199", "report-199",
                                          "resume-evaluate", "fresh-stage-a", "fresh-evaluate", "finalize-rebuild",
                                          "evaluate-999", "report-999", "resume-evaluate-999",
                                          "fresh-evaluate-999", "finalize-rebuild-999"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE)
    parser.add_argument("--rclone", type=Path, default=DEFAULT_RCLONE)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(); args.run_root = args.run_root.resolve(); args.run_root.mkdir(parents=True, exist_ok=True)
    dispatch = {"preflight": lambda: preflight(args), "bundles": lambda: bundles(args),
                "stage-a": lambda: stage_a(args), "evaluate-199": lambda: evaluate(args, B=199),
                "report-199": lambda: report(args, 199), "resume-evaluate": lambda: evaluate(args, B=199, resume=True),
                "fresh-stage-a": lambda: stage_a(args, fresh=True), "fresh-evaluate": lambda: evaluate(args, B=199, fresh=True),
                "finalize-rebuild": lambda: finalize(args, 199), "evaluate-999": lambda: evaluate(args, B=999),
                "report-999": lambda: report(args, 999), "resume-evaluate-999": lambda: evaluate(args, B=999, resume=True),
                "fresh-evaluate-999": lambda: evaluate(args, B=999, fresh=True),
                "finalize-rebuild-999": lambda: finalize(args, 999)}
    raise SystemExit(dispatch[args.stage]())


if __name__ == "__main__":
    main()
