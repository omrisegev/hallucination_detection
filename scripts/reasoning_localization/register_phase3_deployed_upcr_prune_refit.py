#!/usr/bin/env python3
"""Register the planned Phase-3 deployed U-PCR prune/refit branch.

This is a reporting/roster amendment only.  It deliberately does not create a
FROZEN_BEFORE_RUN execution registry and does not evaluate any score.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402

PROGRAM_ROOT = REPO / "results/reasoning_localization_03662_v1"
METHOD_ID = "deployed_upcr_prune_refit"
EXPERIMENT_ID = "P3_DEPLOYED_UPCR_PRUNE_REFIT"
VARIANT_IDS = (
    "P3D0_H2_VIEW_FULLPOOL_IU",
    "P3D1_H2_VIEW_DEPLOYED_UPCR",
    "P3D2_H2_VIEW_MASK_EQUAL_CONTROL",
    "P3D3_H2_VIEW_RANDOM_MASK_CONTROL",
)


def _upsert_unique(rows: list[dict], key: str, row: dict) -> None:
    matches = [index for index, current in enumerate(rows) if current.get(key) == row[key]]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate {key}={row[key]}")
    if matches:
        rows[matches[0]] = row
    else:
        rows.append(row)


def main() -> None:
    methods_path = PROGRAM_ROOT / "METHOD_REGISTRY.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    method = {
        "method_id": METHOD_ID,
        "display_name": "Deployed U-PCR weak-view prune/refit",
        "problem": "Test whether U-PCR's own target-free reliability estimate can remove weak compact token views before a second spectral fit.",
        "plain_summary": "Fits U-PCR once, freezes a weak-expert mask from estimated response covariance, then refits U-PCR on the survivors.",
        "input_operation_output": "compact H2 member views -> full-pool U-PCR rho_hat -> frozen weak-view mask -> survivor U-PCR refit -> token risk",
        "novelty": "Adds the repository's deployed two-pass exclusion policy, which was absent from the completed full-pool ordinary-IU Phase-3 arms.",
        "assumptions": [
            "The additive covariance model estimates relative view reliability well enough to support exclusion.",
            "A donor-stable weak-view mask removes noise without discarding complementary localized evidence.",
            "Refitting after exclusion is materially different from merely zeroing full-pool weights.",
        ],
        "limitations": [
            "Planned development study; no localization result yet.",
            "The exact deployed policy is dimension-ineligible on four outer H2 family scores because its fewer-than-five fallback is a simple mean.",
            "The member-view test changes the compression level and therefore requires a same-matrix full-pool IU parent and mask controls.",
        ],
        "references": [
            "docs/methods/deployed_upcr.md",
            "spectral_utils/upcr.py:347-416",
            "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_DEPLOYED_UPCR_PRUNE_REFIT_V1.md",
        ],
    }
    _upsert_unique(methods["methods"], "method_id", method)
    atomic_write_json(methods_path, methods)

    variants_path = PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    source = next(
        row for row in variants["variants"]
        if row["variant_id"] == "P3A_H2_EQUAL_OUTER_REFERENCE"
    )
    shared = deepcopy(source)
    shared.update({
        "phase": "P3",
        "method_id": METHOD_ID,
        "execution_status": "PLANNED",
        "decision_status": "PENDING",
        "evidence_status": "DEVELOPMENT",
        "statistical_status": "NOT_EVALUATED",
        "detector": "frozen H0 response detector; rerank non-abstentions only",
        "step_reducer": "mean of largest min(10, |I_s|) token risks",
        "supervision": "donor/calibration-only fit and mask; score freeze before label import",
        "signals": [
            "entropy level",
            "entropy dynamics plus C7 member views",
            "partition energy member views without energy_series",
            "top-k distribution member views",
        ],
        "transforms": ["method-native weak-view exclusion", "survivor refit"],
        "prior_evidence": "Deployed U-PCR historically used weak-expert exclusion and recomputation; completed P3B/P3C tested ordinary full-pool IU instead.",
        "causal_validity": "completed-trace localization; a Phase-5 transfer would require prefix-safe member views and donor statistics",
        "access_tier": "gray_box_single_pass",
    })

    rows = []
    definitions = [
        (
            VARIANT_IDS[0], "P3D0 compact-view full-pool IU control", 173,
            ["P3A_H2_EQUAL_OUTER_REFERENCE"], "matched_fullpool_control", False,
            "ordinary two-component IU-PCR on the exact compact H2 member-view matrix; exclusion disabled",
            "Creates the exact same-matrix parent needed to isolate the deployed exclusion/refit policy.",
            "Full-pool view fusion may be less stable than provenance-balanced H2 even before pruning.",
        ),
        (
            VARIANT_IDS[1], "P3D1 compact-view deployed U-PCR", 174,
            [VARIANT_IDS[0]], "deployed_upcr_candidate", True,
            "exact deployed U-PCR: full-pool rho_hat, frozen weak-view exclusion, recomputation on survivors, automatic one/two-PC rule",
            "Tests the previously omitted deployed two-pass U-PCR mechanism on a dimension-eligible compact localization pool.",
            "Estimated rho is unstable or complementary weak views are removed; fallback may erase any distinction from an equal mean.",
        ),
        (
            VARIANT_IDS[2], "P3D2 survivor-mask equal control", 175,
            [VARIANT_IDS[1]], "selection_only_control", False,
            "reuse P3D1's donor-frozen survivor mask; equal-weight retained standardized views; no U-PCR refit weights",
            "Separates the value of view selection from the value of survivor U-PCR reweighting.",
            "Any apparent benefit may be generic dimension reduction rather than rho-informed selection.",
        ),
        (
            VARIANT_IDS[3], "P3D3 cardinality-matched random-mask control", 176,
            [VARIANT_IDS[1]], "random_mask_control", False,
            "predeclared cardinality-matched random masks with frozen seeds and aggregation",
            "Tests whether the rho-derived survivor set beats generic pruning at the same dimension.",
            "A rho mask that does not beat random masks lacks a supported selection mechanism.",
        ),
    ]
    for variant_id, name, order, parents, role, rankable, fusion, novelty, failure in definitions:
        row = deepcopy(shared)
        row.update({
            "variant_id": variant_id,
            "display_name": name,
            "display_order": order,
            "parent_variant_ids": parents,
            "role": role,
            "rankable": rankable,
            "fusion": fusion,
            "novelty": novelty,
            "failure_hypothesis": failure,
            "limitations": "Development-open population; requires frozen member roster, same-matrix parent alias, mask stability, and random/cardinality controls before scores open.",
        })
        rows.append(row)
        _upsert_unique(variants["variants"], "variant_id", row)
    atomic_write_json(variants_path, variants)

    experiments_path = PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "display_name": "Compact deployed U-PCR weak-view exclusion and refit",
        "phase": "P3",
        "execution_status": "PLANNED",
        "question": "Does exact deployed U-PCR weak-view exclusion plus survivor refit improve compact H2 member-view fusion over a matched full-pool ordinary-IU parent?",
        "prerequisite": "Freeze exact compact H2 member names and a dimension-eligible same-matrix full-pool parent; four-family outer U-PCR is NOT_APPLICABLE_BY_DIMENSION.",
        "population_ids": ["current_common_eight_qwen"],
        "task_ids": ["processbench_first_error"],
        "variant_order": list(VARIANT_IDS),
        "registered_comparators": [
            "P3A_H2_EQUAL_OUTER_REFERENCE",
            "P3D0_H2_VIEW_FULLPOOL_IU",
            "P2C_F6_TOP10_REFERENCE",
        ],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired whole-source-question grouped draws; multiplicity family must be amended and frozen before execution",
        "promotion_gates": [
            "P3D0 member roster and no-exclusion alias frozen before labels",
            "P3D1 beats P3D0 and the strongest compact system parent with the registered practical-benefit interval",
            "H0 abstention mismatch equals zero",
            "exact-error, clean-abstention, and worst-cell bounds pass",
            "rho mask is stable across five grouped donor folds",
            "rho mask beats cardinality-matched random masks and selection-only equal control",
            "all kept identities, rho estimates, fallback events, component counts, and projection diagnostics are reported",
        ],
        "method_native_adapter_roster": {
            "ordinary_IU": "additive-model rho_hat then fixed two-PC refit",
            "SU_PCR": "sparse-error-corrected reliability only after premise gate",
            "STG_SU": "fold-stable support plus cardinality-matched random support",
            "DUFS_LIU": "donor-fold-stable DUFS gates then LIU refit",
            "L_SML_B3": "eligible only after method-native reliability and no-prune alias are frozen",
            "tensor_query": "donor-only loading/stability criterion; U-PCR rho cannot be borrowed without a separate model contract",
        },
        "adapter_rule": "At most one full-pool/prune-refit pair per independently surviving method; no failed method is reopened automatically.",
        "next_variant": VARIANT_IDS[0],
        "report_sections": ["p3_parent_fusion", "p3_complexity"],
        "protocol": "docs/experiments/REASONING_LOCALIZATION_03662_PHASE3_DEPLOYED_UPCR_PRUNE_REFIT_V1.md",
    }
    _upsert_unique(experiments["experiments"], "experiment_id", experiment)
    atomic_write_json(experiments_path, experiments)

    print(json.dumps({
        "status": "PLANNED_ROSTER_REGISTERED_NO_EXECUTION",
        "experiment_id": EXPERIMENT_ID,
        "variants": list(VARIANT_IDS),
    }, indent=2))


if __name__ == "__main__":
    main()
