#!/usr/bin/env python3
"""Register the bounded C7/C8 Llama scorer-family transfer before results."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_confirmation as runner  # noqa: E402


def derived(source: dict, *, variant_id: str, name: str, order: int, parent: str,
            role: str, novelty: str, limitations: str) -> dict:
    row = deepcopy(source)
    row.update({
        "variant_id": variant_id, "display_name": name, "display_order": order,
        "parent_variant_ids": [parent], "role": role, "phase": "P2C",
        "execution_status": "PLANNED", "decision_status": "PENDING",
        "evidence_status": "TRANSFER", "statistical_status": "NOT_EVALUATED",
        "rankable": False, "novelty": novelty, "limitations": limitations,
        "task_ids": ["processbench_first_error"],
    })
    return row


def main() -> None:
    path = p1.PROGRAM_ROOT / "VARIANT_REGISTRY.json"
    payload = json.loads(path.read_text())
    rows = payload["variants"]
    by_id = {row["variant_id"]: row for row in rows}
    ids = (runner.TOP10, runner.TOP5, runner.FAMILY6, runner.IU29_PARENT, *runner.VARIANTS)
    if any(variant_id in by_id for variant_id in ids):
        raise RuntimeError("Llama confirmation roster is already registered")
    if by_id["C7_EDIS_ONSET"]["statistical_status"] != "PROMISING_UNCONFIRMED" or by_id["C8_SELF_INNOV"]["statistical_status"] != "PROMISING_UNCONFIRMED":
        raise RuntimeError("C7/C8 Qwen evidence is not in the required uncertain-positive state")

    top10 = derived(by_id["P2A_TOPK10_REFERENCE"], variant_id=runner.TOP10,
                    name="Llama transfer entropy / top-ten", order=120,
                    parent="P2A_TOPK10_REFERENCE", role="transfer_reference",
                    novelty="Exact top-ten entropy reference on the frozen four-cell Llama scorer panel.",
                    limitations="Transfer comparator only; Llama labels were opened in Phase 1.")
    top5 = derived(by_id["R1_ENTROPY_TOP5"], variant_id=runner.TOP5,
                   name="Llama transfer entropy / top-five", order=121,
                   parent="R1_ENTROPY_TOP5", role="transfer_reference",
                   novelty="Checksum-audited Phase-1 top-five score on the same Llama rows.",
                   limitations="Required retained reference; not fresh confirmation.")
    family6 = derived(by_id["R2_FAMILY6_TOP5_CURRENT"], variant_id=runner.FAMILY6,
                      name="Llama transfer family6 / top-five", order=122,
                      parent="R2_FAMILY6_TOP5_CURRENT", role="mechanism_comparator",
                      novelty="Frozen family6 comparator for error-overlap and score-correlation diagnostics.",
                      limitations="Mechanism comparator only; oracle unions are inaccessible descriptive ceilings.")
    iu29 = derived(by_id["C8_IU29_TOP10_PARENT"], variant_id=runner.IU29_PARENT,
                   name="Llama transfer IU29 / top-ten", order=123,
                   parent="C8_IU29_TOP10_PARENT", role="derived_parent_control",
                   novelty="Matched-reducer IU29 parent for isolating C8 innovation on Llama.",
                   limitations="C8 mechanism control, not a promotion reference.")
    c7 = derived(by_id["C7_EDIS_ONSET"], variant_id=runner.VARIANTS[0],
                 name="C7 EDIS onset / Llama transfer", order=124,
                 parent="C7_EDIS_ONSET", role="transfer_candidate",
                 novelty="Exact frozen C7 morphology transferred without retuning to a new scorer family.",
                 limitations="Same source questions were evaluated in Phase 1; evidence grade is TRANSFER, not FRESH_CONFIRMATION.")
    c8 = derived(by_id["C8_SELF_INNOV"], variant_id=runner.VARIANTS[1],
                 name="C8 self innovation / Llama transfer", order=125,
                 parent="C8_SELF_INNOV", role="transfer_candidate",
                 novelty="Exact frozen C8 residual map transferred without retuning to a new scorer family.",
                 limitations="Same source questions were evaluated in Phase 1; evidence grade is TRANSFER, not FRESH_CONFIRMATION.")
    rows.extend([top10, top5, family6, iu29, c7, c8])
    atomic_write_json(path, payload)

    experiments_path = p1.PROGRAM_ROOT / "EXPERIMENT_REGISTRY.json"
    experiments = json.loads(experiments_path.read_text())
    if any(row["experiment_id"] == "P2_CONFIRMATION_LLAMA4" for row in experiments["experiments"]):
        raise RuntimeError("P2 confirmation experiment already registered")
    experiments["experiments"].append({
        "experiment_id": "P2_CONFIRMATION_LLAMA4", "display_name": "C7/C8 Llama scorer-family transfer",
        "phase": "P2C", "execution_status": "RUNNING",
        "question": "Do the exact uncertain-positive C7/C8 candidates transfer to the four Llama scorer cells, and are their errors complementary to family6?",
        "prerequisite": "Completed C1--C8 Qwen screen with C7/C8 labeled PROMISING_UNCONFIRMED and both execution registries frozen before either Llama result.",
        "population_ids": ["current_llama4_scorer_transfer"],
        "task_ids": ["processbench_first_error"],
        "variant_order": list(runner.VARIANTS),
        "registered_comparators": [runner.TOP10, runner.TOP5, runner.FAMILY6, runner.IU29_PARENT],
        "primary_metrics": ["paired_delta_macro_f1"],
        "bootstrap": "20,000 paired source-question grouped draws; Bonferroni simultaneous interval across four C7/C8 by required-reference contrasts",
        "promotion_gates": [
            "delta and simultaneous CI lower exceed +0.005 versus both top-ten and top-five",
            "at least three of four cells nonnegative versus each required reference",
            "worst-cell delta >= -0.020 and no hard delta below -0.030",
            "exact-error and clean-abstention deltas each >= -0.010",
            "all provenance, score-alias, label-firewall, and suffix-invariance checks pass",
        ],
        "evidence_boundary": "scorer-family transfer only; source questions and labels were previously opened in Phase 1",
        "fusion_boundary": "family6 complementarity is descriptive; no router or fusion is fit or selected in this experiment",
        "report_sections": ["p2c_llama_forest", "p2c_complementarity", "p2c_cell_heatmap"],
        "next_variant": runner.VARIANTS[0],
    })
    atomic_write_json(experiments_path, experiments)
    print(json.dumps({"status":"REGISTERED_BEFORE_RESULTS", "variants":ids}, indent=2))


if __name__ == "__main__":
    main()
