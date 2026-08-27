# Advisor update packet — August 2026

Suggested attachment order:

1. 01_basic_fusion_methods.html
2. 02_graphs_and_nuisance.html
3. 03_whitebox_depth.html
4. 04_localization_and_early.html

The short email is ../Advisor_Update_Aug21_2026.md.

Each brief stands alone. It explains the task, gives the smallest useful method equation, shows performance visually,
states the evidence boundary, and ends with questions for the advisors. CLAIM_LEDGER.md is an internal accuracy check
and is not intended as an attachment.

The order follows the previous advisor emails:

- Brief 1: the U-PCR/DUFS fusion line from Bracha's July 30 summary.
- Brief 2: clustering, dependence and nuisance, including the latest Family-NRM graph test.
- Brief 3: Amir's feature-pool question and Bracha's internal-feature access suggestion.
- Brief 4: certified ProcessBench localization, prefix detection, paper-specified-partial LEASH callback stopping and seven separate retrospective RAG evidence panels.

Conformal calibration remains unstarted and is left as a meeting decision rather than presented as a result.

The packet also preserves a benchmark distinction that is still a plan rather than a result: new within-cell
Family-NRM/PGRD variants use no donors or labels; donor-unsupervised and donor-label-selected variants appear only as
separate controls.

The packet is pinned to the certified unified reporting release with nine authenticated source bindings; this
provenance bridge does not create a pooled cross-task result.

Rebuild:

    python3 scripts/build_advisor_update_aug21_2026.py

Verify without changing files:

    python3 scripts/build_advisor_update_aug21_2026.py --check
