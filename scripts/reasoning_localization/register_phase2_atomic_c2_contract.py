#!/usr/bin/env python3
"""Register the exact preregistered C2 adaptive-SWVar sensitivity contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT  # noqa: E402


def main() -> None:
    variants_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variants_path.read_text(encoding="utf-8"))
    by_id = {row["variant_id"]: row for row in payload["variants"]}
    if by_id["C1_ENT_SW16"]["execution_status"] != "HARD_FAIL":
        raise RuntimeError("C1 terminal result is not registered")
    c2 = by_id["C2_ENT_SWADAPT"]
    if c2["execution_status"] != "PLANNED" or "c2_contract" in c2:
        raise RuntimeError("C2 is not unopened")
    c2.update({
        "parent_variant_ids": ["P2A_TOPK10_REFERENCE"],
        "detector": "equal_feature_mean response detector with per-arm grouped five-fold threshold",
        "fusion": "equal mean of within-cell entropy-step and adaptive-SWVar-step empirical midranks",
        "step_reducer": "apply frozen top-ten separately to entropy and causal adaptive SWVar; then equal step-rank fusion",
        "supervision": "all scores and fusion frozen label-free; labels enter only grouped held-fold threshold calibration and evaluation",
        "causal_validity": "adaptive SWVar uses observed prefix length only and resets per response; localization readout waits for the completed step",
        "limitations": "Preregistered sensitivity arm after C1; it tests window mismatch only and cannot be combined post hoc with fixed SWVar16.",
        "c2_contract": {
            "entropy_input": "negative mixed-v2 entropy confidence coordinate",
            "adaptive_window": "w_t = clip(floor(0.10 * (t+1)), 3, 32), recomputed from observed response-prefix length",
            "variance": "population variance ddof=0 over the last min(w_t,t+1) observed tokens",
            "reset": "every response",
            "channel_reducer": "mean of largest min(10, step_length) values",
            "fusion": "0.5 * empirical_midrank(entropy_step) + 0.5 * empirical_midrank(adaptive_swvar_step), within cell",
            "response_combination": "geometric mean with equal_feature_mean response empirical midrank",
            "threshold": "separate deterministic grouped five-fold cross-fit per arm after score freeze",
            "comparators": ["P2A_TOPK10_REFERENCE", "R1_ENTROPY_TOP5"],
            "suffix_invariance": "exact deterministic prefix replay audit",
        },
    })
    atomic_write_json(variants_path, payload)

    experiments_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P2_ATOMIC")
    if experiment.get("opened_variants") != ["C1_ENT_SW16"]:
        raise RuntimeError("C2 must open immediately after C1")
    experiment["c2_contract"] = c2["c2_contract"]
    experiment["next_variant"] = "C2_ENT_SWADAPT"
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status": "REGISTERED_BEFORE_C2", "candidate": "C2_ENT_SWADAPT"}, indent=2))


if __name__ == "__main__":
    main()
