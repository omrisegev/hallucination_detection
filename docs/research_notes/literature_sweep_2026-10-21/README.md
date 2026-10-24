# Literature sweep for PRM boosting (2026-10-21)

Omri asked: which data lets us compare with the competitors that matter, what Socratic is worth if neither we nor
the competitors do well on it, and whether recent ICML / NeurIPS / ICLR / ACL / EMNLP papers add anything.
Three search agents (ML venues, NLP venues, benchmarks) and one independent fact-checker. Supplements
`../competitor_registry_2026-10-21/REGISTRY.md`. No experiment was run.

Files: `ML_VENUES.md`, `NLP_VENUES.md`, `BENCHMARKS.md` (agent reports), `VERIFY.md` (10 key claims checked
against primary sources: 7 confirmed, 2 partly, 1 corrected). Numbers below are from the papers' own tables as
read by the agents; re-check against the PDFs before they enter a paper.

## Findings
1. **Shared benchmarks.** PRMBench is used by 4 of our 6 direct competitors (PRISM, CROP, ARES, SLIDER), and
   ProcessBench by 3 (PRISM, CROP, VerifySteer). ProcessBench is also the most common step benchmark at the top venues (6 of 24 NLP
   papers, 6 of 25 ML papers). Socratic-PRMBench: 0 papers outside its own. Hard2Verify: VerifySteer only.
2. **No label-free PRM + telemetry fusion found** at the six venues (VERIFY 3-7).
   - Closest telemetry-only localizer: *Mind the Gap* (ICML 2026, PMLR v306). Entropy drop over top-20 tokens,
     one teacher-forced pass, no PRM. ProcessBench step-localization accuracy, not F1.
   - Closest fusion cousin: *BR2-PRM* (PRMBench-V paper, ACL 2026 Main). Dawid-Skene-style reliability over a frozen
     judge's per-error-type scores, fitted on labelled calibration data, trace-level selection.
   - Label-free aggregation with dependencies at ICML 2026: *CARE* (sparse + low-rank precision, tensor or SVD
     variants) and Ising-model aggregation. Neither is tested on step-level reasoning data. Candidate alternative fusion rules.
   - *AutoMeco* (EMNLP 2025 Main): label-free step telemetry, but it uses Qwen2.5-Math-PRM-7B as its ground truth.
   - *R-PRM* (EMNLP 2025 Main): PRMScore 66.8 (DPO variant) vs 65.5 for Qwen2.5-Math-PRM-7B. A trained-PRM row for the leaderboard.
3. **Contamination.** PRMBench takes correct solutions from both PRM800K splits, and PURE-PRM-7B was trained on PRM800K train
   (VERIFY 1-2). PURE is therefore in-sample on PRMBench's source solutions. Math-Shepherd PRM trained on GSM8K/MATH
   train; the sources of Qwen-PRM and RLHFlow are not stated on their cards.
4. **Venues.**
   - Hard2Verify is ACL 2026 (Anthology 2026.acl-long.1031).
   - CRV is ICLR 2026 Oral, with public step-labelled traces (CC-BY-NC).
   - CROP's EMNLP 2026 acceptance is stated on arXiv only, with no official program page found.
   - SLIDER is an ICLR 2026 workshop paper.
   - PRISM and VerifySteer list no venue.

## Candidate datasets for matched comparisons
| dataset | competitors on it | what it needs | caveat |
|---|---|---|---|
| PRMBench | PRISM, CROP, ARES, SLIDER | done | PURE in-sample |
| ProcessBench | PRISM, CROP, VerifySteer, Mind the Gap | done (job 289297); L-SML loses on strong PRMs | out of thesis focus (Omri); first-error labels only |
| CRV Arithmetic / GSM8K | CROP | PRM telemetry inference on public traces | non-commercial license; very few errors |
| Math-Shepherd, PRM800K (CROP's splits) | CROP | PRM telemetry inference | in-sample for Math-Shepherd PRM / PURE |
| Hard2Verify | VerifySteer | done (177 answers) | small; no-error answers need an abstaining rule |
