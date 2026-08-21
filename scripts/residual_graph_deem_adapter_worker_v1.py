#!/usr/bin/env python3
"""One isolated packaged-DEEM 0.2.0 fit for B1/B2."""

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
        X = np.asarray(data["X_risk"], dtype=float)
        cell_id = str(data["cell_id"].item())
        names = tuple(str(value) for value in data["feature_names"].tolist())
        input_sha256 = str(data["input_sha256"].item())
        provenance = {
            key: str(data[key].item())
            for key in ("bundle_sha256", "source_sha256", "inventory_sha256", "code_sha256")
        }
    config = (
        hard_adapter020_config(device=args.device)
        if args.mode == "hard"
        else repaired_soft_adapter020_config(device=args.device)
    )
    metadata = {
        "schema": "residual_graph_deem_adapter020_fit_v1",
        "mode": args.mode,
        "cell_id": cell_id,
        "arm_id": "B1" if args.mode == "hard" else "B2",
        "stem": args.output.name,
        "seed": int(args.seed),
        "input_sha256": input_sha256,
        "config_sha256": canonical_sha256(config),
        "status": "failed",
        **provenance,
        "environment": environment_fingerprint(),
        "determinism": {"torch_deterministic_algorithms": True},
    }
    try:
        result = fit_deem_score(
            X, seed=args.seed, config=config, feature_names=names, verbose=False
        )
        array_path = args.output.with_suffix(".npz")
        array_sha256 = atomic_save_npz(
            array_path,
            score=result.score,
            posterior=result.aligned_probabilities,
            feature_names=np.asarray(names, dtype=str),
        )
        metadata.update(
            {
                "status": "complete",
                "array_path": str(array_path),
                "array_sha256": array_sha256,
                "package_version": result.package_version,
                "class_map": result.class_map,
                "package_class_map": result.package_class_map,
                "alignment": result.alignment,
                "history": result.history,
                "score_sd": float(np.std(result.score)),
                "score_n_unique": int(len(np.unique(result.score))),
                "healthy": bool(np.isfinite(result.score).all() and np.std(result.score) >= 1e-3),
                "health": {
                    "healthy": bool(
                        np.isfinite(result.score).all() and np.std(result.score) >= 1e-3
                    ),
                    "score_sd": float(np.std(result.score)),
                    "score_n_unique": int(len(np.unique(result.score))),
                },
            }
        )
    except Exception as exc:
        metadata.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "history": jsonable(getattr(exc, "deem_history", {})),
            }
        )
    metadata["content_sha256"] = canonical_sha256(metadata)
    atomic_write_json(args.output.with_suffix(".json"), metadata)
    if metadata["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
