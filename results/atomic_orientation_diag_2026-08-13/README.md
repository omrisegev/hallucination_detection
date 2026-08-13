# Atomic orientation diagnosis — 2026-08-13

Companion artifacts for
[`docs/research_notes/atomic_orientation_reply_2026-08-13.md`](../../docs/research_notes/atomic_orientation_reply_2026-08-13.md),
the reply to the atomic NRM grouping audit (`686c4ef`).

## Contents

| file | what it is |
|---|---|
| `atomic_orientation_diag.py` | Main diagnostic. Reproduces the frozen atomic calibration through `spectral_utils.atomic_neutral_residual` itself (all 17 eigenvalues match the structural audit to 4 decimals; the frozen direction's transfer deltas reproduce exactly: PB Qwen −1.305pp, PB Llama −1.106pp), then measures: target eigenmass per calibration mode, anchor/direction alignments, three label-free b-coupled orientation estimators, LOFO + transfer scoring under the frozen 1/√17 machinery with supervised references, and the premise checks for the accuracy-proxy and joint-diagonalization directions. |
| `atomic_orientation_diag.log` | Full run log (readable end-to-end; all headline numbers quoted in the memo appear here). |
| `RESULT.json` | Machine-readable results of the main diagnostic. |
| `trust_sweep_addendum.py` | Matched-trust control: supervised / γ̂3 / frozen directions at trust 1/p, 1/6, 1/√17, and a self-consistency scale, on originals-LOFO and ProcessBench. |
| `trust_sweep.log`, `TRUST_SWEEP.json` | Its log and results. |
| `family_coherence_addendum.py` | Family-level per-cell target-direction coherence and the family-level b-coupled sign-bit margin (the §5-item-1 numbers). |
| `refined_partition_nrm_v0.py` | Refined-partition NRM v0 (memo §10): provenance families split by pooled-γ̂3 sign (G=10), mode selected/signed by the γ̂3 witness. Includes the family-NRM control through the same code, which reproduces the published +0.277/+0.557/+1.580 exactly. Candidate is negative everywhere. |
| `refined_v0.log`, `refined_v0_RESULT.json` | Its log and results. |
| `labelfree_partition_selection_test.py` | Partition-search closure (memo §11): reproduces Codex's 50 seeded random partitions (fidelity: #36 re-scores at the published +0.514 exactly), computes five label-free selection criteria, correlates with the published labeled quality. All criteria ≈ uninformative; best label-free pick scores +0.52 vs provenance +0.93. |
| `partition_selection.log`, `partition_selection_RESULT.json` | Its log and results. |

## How to rerun

The scripts import project code from a checkout of `master` at `686c4ef` or
later (they were run against an exported copy; point `MW` at the repo root)
and read data from this repo: `results/dependency_fusion_raw/cells.npz` and
`dataset_cache/repgrid/pb_{qwen3_4b,qwen3_8b,llama31_8b}/processbench_*.pkl`.
The SemGrad regraded cache (`local_cache/semgrad_bem_regraded/`) was absent on
the machine that ran these, so SemGrad rows are missing from all tables.

Labels are read only for reference directions (per-cell Fisher class-mean
differences) and AUROC readouts. No candidate estimator receives a label.
