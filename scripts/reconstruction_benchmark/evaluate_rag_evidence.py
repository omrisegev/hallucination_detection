#!/usr/bin/env python3
"""Open labels only after RAG A/B score certification and emit tidy tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.rag_evidence_evaluation import evaluate_rag_evidence_build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--score-verifier-repo", required=True, type=Path)
    parser.add_argument("--release-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/releases")
    parser.add_argument("--private-root", type=Path, default=REPO / "results/reconstruction_benchmark_v1/private_control")
    parser.add_argument("--registry", type=Path, default=REPO / "configs/reconstruction_benchmark_v1/rag_evidence.json")
    args = parser.parse_args()
    manifest = evaluate_rag_evidence_build(
        repo=REPO, score_verifier_repo=args.score_verifier_repo,
        registry_path=args.registry, source_root=args.source_root,
        release_root=args.release_root, private_root=args.private_root,
        release_id=args.release_id, build_id=args.build,
    )
    print(manifest["payload_sha256"])


if __name__ == "__main__":
    main()
