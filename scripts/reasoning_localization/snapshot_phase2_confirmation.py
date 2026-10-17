#!/usr/bin/env python3
"""Create an immutable post-Phase-2 Llama-confirmation report snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_bytes, atomic_write_json, sha256_file  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT  # noqa: E402


def main() -> None:
    target=PROGRAM_ROOT/"snapshots/amendment_phase2_llama_confirmation"
    if target.exists(): raise FileExistsError(f"refusing to overwrite immutable snapshot: {target}")
    experiment=next(row for row in json.loads((PROGRAM_ROOT/"EXPERIMENT_REGISTRY.json").read_text())["experiments"] if row["experiment_id"]=="P2_CONFIRMATION_LLAMA4")
    if experiment["execution_status"]!="COMPLETE" or experiment["verdict"]!="NO_TRANSFER_PROMOTION": raise RuntimeError("confirmation is not terminal")
    target.mkdir(parents=True,exist_ok=False)
    for name in ("REPORT.html","REPORT_MANIFEST.json"):
        source=PROGRAM_ROOT/name; atomic_write_bytes(target/name,source.read_bytes())
        if sha256_file(target/name)!=sha256_file(source): raise RuntimeError(f"snapshot mismatch: {name}")
    manifest={"schema":"reasoning-localization-immutable-snapshot-v1","snapshot_id":"amendment_phase2_llama_confirmation",
              "experiment_id":"P2_CONFIRMATION_LLAMA4","status":"IMMUTABLE","files":[{"path":name,"sha256":sha256_file(target/name)} for name in ("REPORT.html","REPORT_MANIFEST.json")]}
    atomic_write_json(target/"SNAPSHOT_MANIFEST.json",manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
