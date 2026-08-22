#!/usr/bin/env python3
"""Run one isolated, frozen packaged-DEEM 0.2.0 B1/B2 control fit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_adapter import (  # noqa: E402
    fit_deem_score,
    hard_adapter020_config,
    repaired_soft_adapter020_config,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    environment_fingerprint,
    jsonable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("hard", "soft"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        X = np.asarray(data["X_risk"], dtype=np.float64)
        names = tuple(str(value) for value in data["feature_names"].tolist())
        context = {key: str(data[key].item()) for key in data.files if key != "X_risk" and key != "feature_names"}
    config = (hard_adapter020_config(device=args.device) if args.mode == "hard"
              else repaired_soft_adapter020_config(device=args.device))
    record = {
        "schema": "deem_vs_iupcr_adapter020_fit_v1",
        "status": "failed",
        "cell_id": context["cell_id"],
        "arm_id": "B1" if args.mode == "hard" else "B2",
        "stem": args.output.name,
        "seed": int(args.seed),
        "mode": args.mode,
        "config": jsonable(config),
        "config_sha256": canonical_sha256(config),
        "environment": environment_fingerprint(),
        "determinism": {"torch_deterministic_algorithms": True},
        **context,
    }
    try:
        result = fit_deem_score(X, seed=args.seed, config=config, feature_names=names)
        array_path = args.output.with_suffix(".npz")
        array_hash = atomic_save_npz(
            array_path,
            score=np.asarray(result.score, dtype=np.float64),
            posterior=np.asarray(result.aligned_probabilities, dtype=np.float64),
            feature_names=np.asarray(names, dtype=str),
        )
        score_finite = bool(np.isfinite(result.score).all())
        healthy = bool(score_finite and np.std(result.score) >= 1e-3)
        record.update({
            "status": "complete",
            "array_path": str(array_path.resolve()),
            "array_sha256": array_hash,
            "package_version": result.package_version,
            "class_map": result.class_map,
            "package_class_map": result.package_class_map,
            "alignment": result.alignment,
            "history": result.history,
            "health": {
                "healthy": healthy,
                "score_finite": score_finite,
                "score_sd": float(np.std(result.score)),
                "score_n_unique": int(len(np.unique(result.score))),
            },
        })
    except Exception as exc:
        record.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "history": jsonable(getattr(exc, "deem_history", {})),
        })
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(args.output.with_suffix(".json"), record)
    # Amendment A1: a complete fit with finite scores is a valid measurement
    # even when the score has collapsed (healthy=False).  Health stays recorded
    # in the JSON; whether to block on it is the caller's per-arm policy, not
    # the worker's.  A failed fit or a non-finite score still exits 2.
    if record["status"] != "complete" or not record.get("health", {}).get("score_finite"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
