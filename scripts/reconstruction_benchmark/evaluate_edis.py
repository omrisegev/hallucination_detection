#!/usr/bin/env python3
"""Open EDIS labels and source-question groups only after exact A/B PASS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.edis_evaluation import evaluate  # noqa: E402
from spectral_utils.reconstruction_benchmark.edis_identity import load_edis_identity_controller  # noqa: E402


DEFAULT_RELEASE_ROOT = REPO / "results/reconstruction_benchmark_v1/releases"
DEFAULT_PRIVATE_CONTROL = REPO / "results/reconstruction_benchmark_v1/private_control"
DEFAULT_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/edis_target_free.json"
DEFAULT_POSTFREEZE = REPO / "configs/reconstruction_benchmark_v1/edis_postfreeze.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--private-control-root", type=Path, default=DEFAULT_PRIVATE_CONTROL)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--postfreeze-registry", type=Path, default=DEFAULT_POSTFREEZE)
    parser.add_argument("--ab-certificate", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    if args.bootstrap_draws != 20_000:
        raise RuntimeError("scientific EDIS evaluation requires exactly 20,000 draws")
    identity = load_edis_identity_controller(
        private_control_root=args.private_control_root,
        release_id=args.release_id,
        create=False,
        release_root=args.release_root,
        repo=REPO,
    )
    manifest = evaluate(
        release_id=args.release_id,
        build_id=args.build,
        release_root=args.release_root,
        private_control_root=args.private_control_root,
        source_root=args.source_root,
        preparation_registry_path=args.registry,
        postfreeze_registry_path=args.postfreeze_registry,
        identity=identity,
        repo=REPO,
        certificate_path=args.ab_certificate,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "build_id": args.build,
        "evidence_status": manifest["evidence_status"],
        "bootstrap_draws": manifest["bootstrap_draws"],
        "artifacts": manifest["artifacts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
