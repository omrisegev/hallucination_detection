#!/usr/bin/env python3
"""Verify byte-exact post-label localization evaluation A/B outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.localization_ab import (  # noqa: E402
    DEFAULT_EXTERNAL_REGISTRY,
    DEFAULT_LOCALIZATION_REGISTRY,
    DEFAULT_POPULATION_REGISTRY,
    DEFAULT_SOURCE_ROOT,
)
from spectral_utils.reconstruction_benchmark.localization_postfreeze import (  # noqa: E402
    verify_localization_evaluation_ab,
)
from spectral_utils.reconstruction_benchmark.localization_postfreeze_amendment import (  # noqa: E402
    DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--release-root", type=Path,
        default=REPO / "results/reconstruction_benchmark_v1/releases",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identity-key", type=Path)
    parser.add_argument("--score-ab-certificate", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_LOCALIZATION_REGISTRY)
    parser.add_argument("--external-registry", type=Path, default=DEFAULT_EXTERNAL_REGISTRY)
    parser.add_argument("--populations", type=Path, default=DEFAULT_POPULATION_REGISTRY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--score-verifier-repo", type=Path, required=True,
        help="Clean checkout at the exact score-frozen commit used to reverify score A/B.",
    )
    parser.add_argument(
        "--postfreeze-amendment", type=Path,
        default=DEFAULT_LOCALIZATION_POSTFREEZE_AMENDMENT,
    )
    args = parser.parse_args()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("scientific localization evaluation A/B verification requires a clean worktree")
    certificate = verify_localization_evaluation_ab(
        release_id=args.release_id,
        release_root=args.release_root,
        output_path=args.output,
        identity_key_path=args.identity_key,
        localization_ab_certificate_path=args.score_ab_certificate,
        localization_registry_path=args.registry,
        external_registry_path=args.external_registry,
        population_registry_path=args.populations,
        source_root=args.source_root,
        score_verifier_repo=args.score_verifier_repo,
        evaluation_repo=REPO,
        localization_postfreeze_amendment_path=args.postfreeze_amendment,
    )
    print(json.dumps({
        "release_id": args.release_id,
        "status": certificate["status"],
        "certificate_sha256": certificate["certificate_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
