#!/usr/bin/env python3
"""Register exact C3--C8 contracts before opening any remaining result."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization.run_phase1_baseline import PROGRAM_ROOT  # noqa: E402


CONTRACTS = {
    "C3_ENT_CCUSUM": {
        "parent_variant_ids": ["C1_ENT_SW16"],
        "fusion": "equal mean of within-cell step midranks for entropy, SWVar16, and absolute reset CUSUM",
        "step_reducer": "frozen top-ten applied separately to all three channels before equal step-rank fusion",
        "supervision": "all transforms and fusion are label-free; labels enter only after score freeze",
        "causal_validity": "two-sided CUSUM uses observed prefix only, resets per response, and uses the frozen mixed-v2 zero center",
        "limitations": "User-authorized diagnostic amendment after C1 hard failure; this run cannot promote or reopen the SWVar branch.",
        "exact_contract": {
            "inputs": ["negative mixed-v2 entropy confidence", "causal SWVar16 of that entropy risk"],
            "cusum": "g+_t=max(0,g+_{t-1}+z_t-kappa); g-_t=max(0,g-_{t-1}-z_t-kappa); output=max(g+,g-)",
            "center": "mixed-v2 standardized zero", "kappa": 0.0, "reset": "every response",
            "channel_reducer": "top-10 mean", "fusion": "equal empirical-midrank mean",
            "eligibility": "diagnostic only because exact parent C1 hard-failed",
        },
    },
    "C4_ENT_SAMPLED": {
        "parent_variant_ids": ["P2A_TOPK10_REFERENCE"],
        "fusion": "equal mean of within-cell entropy-step and sampled-surprisal-step empirical midranks",
        "step_reducer": "frozen top-ten applied separately to entropy and sampled-token surprisal",
        "supervision": "target-free score; labels only after complete score freeze",
        "causal_validity": "primitive token sources are prefix-valid; localization readout waits for the completed step",
        "exact_contract": {"channels": ["entropy_series", "spilled_series"], "orientation": "negative confidence is risk",
                           "channel_reducer": "top-10 mean", "fusion": "equal empirical-midrank mean"},
    },
    "C5_ENT_ENERGY": {
        "parent_variant_ids": ["P2A_TOPK10_REFERENCE"],
        "fusion": "equal mean of within-cell entropy-step and partition-energy-step empirical midranks",
        "step_reducer": "frozen top-ten applied separately to entropy and partition energy",
        "supervision": "target-free score; labels only after complete score freeze",
        "causal_validity": "primitive token sources are prefix-valid; localization readout waits for the completed step",
        "exact_contract": {"channels": ["entropy_series", "energy_series"], "orientation": "negative confidence is risk",
                           "channel_reducer": "top-10 mean", "fusion": "equal empirical-midrank mean"},
    },
    "C6_DSP12": {
        "parent_variant_ids": ["C4_ENT_SAMPLED", "C5_ENT_ENERGY"],
        "fusion": "equal mean of twelve within-cell step midranks; no learned or label-selected weights",
        "step_reducer": "frozen top-ten applied separately to each of the twelve source-transform channels",
        "supervision": "target-free fixed grammar; labels only after complete score freeze",
        "causal_validity": "all four operators use the observed prefix and reset per response",
        "exact_contract": {
            "sources": ["entropy_series", "spilled_series", "energy_series"],
            "operators": ["level", "EWMA span16 alpha=2/17", "running positive-area mean above standardized zero", "running persistence fraction above standardized zero"],
            "n_channels": 12, "channel_reducer": "top-10 mean", "fusion": "equal empirical-midrank mean",
            "forbidden": ["operator search", "source search", "learned weights", "task-label tuning"],
            "eligibility": "conditional on the preregistered atomic-parent premise; otherwise diagnostic only",
        },
    },
    "C7_EDIS_ONSET": {
        "parent_variant_ids": ["P2A_TOPK10_REFERENCE"],
        "fusion": "single maximum-of-burst-and-rebound-onset curve",
        "step_reducer": "frozen top-ten over the standardized EDIS-onset curve",
        "supervision": "fixed morphology; labels only after score freeze",
        "causal_validity": "running minimum and one-step differences use observed prefix only and reset per response",
        "limitations": "Exploratory standardized repository adaptation. Raw entropy in nats is absent from the sealed input, so this is not paper-exact EDIS and cannot auto-promote.",
        "exact_contract": {"input": "negative mixed-v2 entropy confidence", "burst": "max(delta z - 1.36, 0)",
                           "rebound": "positive onset increment of max(z-running_min(z)-1.33,0)",
                           "curve": "max(burst,rebound_onset)", "fidelity": "standardized adaptation, not raw-nat paper replay"},
    },
    "C8_SELF_INNOV": {
        "parent_variant_ids": ["R3_IU29"],
        "fusion": "ordinary two-component IU-PCR over the original 29 confidence streams plus one 29-stream self-residual block",
        "step_reducer": "frozen top-ten on both the IU29 top-ten parent and augmented-IU candidate",
        "supervision": "target-free per-cell fit on the incumbent deterministic token cap; labels only after score freeze",
        "causal_validity": "fixed fitted map uses only intercept, log token position, and one-step self lag; resets at response boundaries",
        "limitations": "Diagnostic only. Residual magnitude is a fixed unpredictability hypothesis; prior ProcessBench evidence was uncertain and PRMBench worsened.",
        "exact_contract": {
            "base": "incumbent mixed-v2 IU29 preparation and two-component IU configuration",
            "predictors": ["intercept", "log1p(token position)", "one-step same-stream lag"],
            "ridge": 1.0, "fit_weighting": "equal total weight per response", "first_token_residual": 0.0,
            "innovation_confidence": "negative absolute donor-RMS-standardized residual",
            "augmented_dimension": 58, "orientation": "correlation with frozen IU29 fit-token confidence",
            "parent_audit": "original-only reconstruction must alias frozen R3 step-max; C8 comparison parent uses the common top-ten reducer",
        },
    },
}


def main() -> None:
    variants_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(variants_path.read_text(encoding="utf-8"))
    by_id = {row["variant_id"]: row for row in payload["variants"]}
    if by_id["C1_ENT_SW16"]["execution_status"] != "HARD_FAIL" or by_id["C2_ENT_SWADAPT"]["execution_status"] != "HARD_FAIL":
        raise RuntimeError("C1/C2 terminal evidence is not registered")
    for variant_id, contract in CONTRACTS.items():
        row = by_id[variant_id]
        if "exact_contract" in row:
            raise RuntimeError(f"{variant_id} exact contract already registered")
        if variant_id == "C3_ENT_CCUSUM":
            if row["execution_status"] != "NOT_RUN_BY_GATE":
                raise RuntimeError("C3 parent-gate state drifted")
            row.update({"execution_status":"PLANNED", "decision_status":"NO_PROMOTION",
                        "statistical_status":"NOT_EVALUATED", "rankable":False,
                        "amendment":"User-authorized diagnostic continuation; parent hard failure remains binding."})
        elif row["execution_status"] != "PLANNED":
            raise RuntimeError(f"{variant_id} is not unopened")
        row.update(contract)
    atomic_write_json(variants_path, payload)

    experiments_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = next(row for row in experiments["experiments"] if row["experiment_id"] == "P2_ATOMIC")
    experiment.update({
        "remaining_batch_contract_status":"REGISTERED_BEFORE_RESULTS",
        "remaining_batch_order":list(CONTRACTS),
        "remaining_batch_contracts":{key:value["exact_contract"] for key,value in CONTRACTS.items()},
        "primary_comparison_family_size_at_completion":16,
        "c3_amendment":"diagnostic continuation only; C1 hard failure remains binding",
        "next_variant":"C3_ENT_CCUSUM",
    })
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status":"REGISTERED_BEFORE_C3_C8","variants":list(CONTRACTS)}, indent=2))


if __name__ == "__main__":
    main()
