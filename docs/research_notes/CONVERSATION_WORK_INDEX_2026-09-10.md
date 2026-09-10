# Research work created during the September 5-10 conversation

This index records where the main documents and experiment artifacts were created before the
direct-probability experiment was isolated. The original working checkout was
`C:\Users\omris\TAU\hallucination_detection`. The dedicated experiment worktree is
`C:\Users\omris\TAU\hallucination_detection\.worktrees\direct-probability-fusion-v1` on branch
`codex/direct-probability-fusion-v1`.

## Main review documents

| Document | Purpose | Status at snapshot |
|---|---|---|
| `docs/reviews/project_publication_review_2026-09-05.html` | Project history, publication gaps, cleanup and next steps | Working review |
| `docs/reviews/joint_lsml_visual_guide_2026-09-06.html` | Undergraduate-level visual guide to L-SML, Joint L-SML and graph variants | Completed guide |
| `docs/reviews/research_consolidation_2026-09-08.html` | Hebrew consolidation of experiments, decisions and open work | Draft; full-sampling run remained incomplete |
| `docs/research_notes/joint_lsml_research_story_he_2026-09-08.md` | Hebrew research narrative and method lineage | Working note |
| `docs/research_notes/los_direct_probability_fusion_review_2026-09-10.md` | LOS-Net representation review and direct-probability proposal | Updated for the cached-data contract |

## Experiment contracts

| Document | Purpose | Status at snapshot |
|---|---|---|
| `docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md` | Current 13,769-row ProcessBench/PRMBench contract | Frozen development contract |
| `docs/experiments/RESEARCH_CONSOLIDATION_20260908.md` | Plan to finish old runs and consolidate evidence | Approved; sampling obligation incomplete |
| `docs/experiments/DIRECT_PROBABILITY_FUSION_V1.md` | Gray-box probability-rank fusion experiment | Completed on both full tracks; not promoted |
| `docs/experiments/DIRECT_PROBABILITY_FUSION_V1_SOURCE_SNAPSHOT.json` | SHA-256 record of files copied from the latest working checkout | Reproducibility record |

## The Codex run that reached 3,038 rows

The run was `results/localization_full_sampling_v3/`, driven by
`scripts/run_full_sampling_v3.py` and supervised by
`scripts/complete_research_consolidation_v1.py`. The progress note captured 3,038 rows; the last
checkpoint reached 3,547 of 13,769. No matching process was alive when this snapshot was made.
Its `RUN_STATE.json` is therefore historical state, not evidence that a run is active.

The related reflection artifacts are under `results/research_consolidation_v1/`. Their status is
`IN_PROGRESS_NOT_FINAL` / `DRAFT_CHECKS_PASS_NOT_FINAL`, because the full-sampling run did not
finish. Partial sampling scores are excluded from scientific comparisons.

## Claude results relevant to the new experiment

The committed Joint L-SML v2 worktree is `C:\Users\omris\TAU\hd_jlsml_v2_wt`, branch
`claude/joint-lsml-optimization-v2`, commit `ae420bcb`. Its main report is
`results/joint_lsml_optimization_v2/REPORT.md`. Later Claude work was written in the main working
checkout rather than that worktree:

- `results/fusion_fixed_gate_v1/`: mean raw entropy, q=0.3 gate.
- `results/fusion_shrinkage_iu_v1/`: Joint-inspired Ledoit-Wolf covariance shrinkage inside IU.
- `results/token_level_readout_v1/`: token entropy/varentropy, top-10 step readout and PRMScore.
- `results/token_bocpd_v1/`: negative BOCPD result; retained as context, not a v1 candidate.

The new experiment reuses the first three as frozen infrastructure or baselines. It does not add
BOCPD, graph variants, lambda search, new window sizes or white-box signals.

## Direct Probability Fusion v1 result

The completed report is
`results/direct_probability_fusion_v1/REPORT.html`. Machine-readable results are
`LOCALIZATION.json` and `HISTORICAL_24.json`; exact score arrays and separate
`RUN_MANIFEST_LOCALIZATION.json` / `RUN_MANIFEST_HISTORICAL.json` files are in
the same directory. The direct IU and Joint arms were not promoted. Token
entropy remains the localization anchor and Historical IU-PCR remains the
24-cell anchor.

The v1 matrix used sorted probability ranks only. It did not include a distinct
sampled-token probability or explicit residual-tail coordinate. Existing raw
caches contain both ingredients for one future frozen augmentation, if that
bounded follow-up is approved.
