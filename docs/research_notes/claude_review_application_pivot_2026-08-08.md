# Review Request: Close the Fusion Cycle and Pivot to Applications

**Date:** 2026-08-08

**Status:** draft decision package for independent review

**Requested reviewer:** Claude
**Code or result change in this step:** none; this document summarizes existing artifacts

## 1. Review question

Please review whether the evidence supports this project decision:

> Freeze DUFS-LIU mixed-v2 as the common fusion core. Stop creating small
> variants for the current single-pass static feature pool. Focus next on two
> applications: hallucination localization and hallucination in RAG citations.

The conclusion is deliberately narrow. We do **not** claim that U-PCR can never
be improved. We claim that the tested unsupervised refinements did not provide
a robust target-aligned improvement with the current measurements and
evaluation protocol.

## 2. Naming and current method

**DUFS-LIU** means **DUFS-gated Laplacian-regularized IU-PCR**.

- IU-PCR provides the unsupervised covariance-based regression fusion.
- A sample graph and its Laplacian regularize the U-PCR solution.
- A DUFS-inspired differentiable gate controls feature participation in the
  graph/fusion objective.
- The forward feature contract is **mixed-v2**:
  `pe_mean=squared`, `stft_spectral_entropy=mode`,
  `cusum_shift_idx=raw`, and `rpdi=raw`.

Historical stable-only experiments are still valid records. They must not be
silently relabeled as mixed-v2 results.

## 3. What was tested in this development cycle

| Step | Main question | Result used in the decision |
|---|---|---|
| 225 | Can the repository support a reproducible new fusion study? | Contracts, runners, reports, and audits were made reproducible. |
| 226 | Do sparse/dependent error models improve U-PCR? | SU-PCR/SDSF/DEEM comparisons did not establish a robust deployable gain. |
| 227 | Can view fusion or SpecRaGE-style geometry improve IU-PCR? | Geometry was learnable, but the label-free objective did not reliably identify target-useful geometry. |
| 228 | Are atomic features or micro-views better than semantic families? | Atomic and micro-view proxies did not predict downstream utility; oracle headroom remained small. |
| 229 | Is family relevance local to examples? | Conditional expertise exists with labels, but the deployable unsupervised gate did not recover it. |
| 230 | Does repeated cross-view diffusion reveal a common manifold? | Convergence was strong; AUROC improvement was effectively zero. |
| 231 | Can a small mixed transformation contract improve the baseline? | mixed-v2 gave a small retrospective gain and became the forward implementation contract, with fragility noted. |
| 232 | Can the core be used for first-error localization? | Yes. GL-LIU v1 gave the clearest application-level gain. |
| 233 | Should the same DUFS-LIU implementation be used in both localization heads? | Core-five local DUFS-LIU was slightly better descriptively; broad-28 was worse. |
| 234 | Can one trace create useful repeated measurements? | Replicate covariance was stable, but reliability filtering did not improve target ranking. |
| Closing sensitivity check | Should deployed U-PCR hard-filter features before IU-PCR/DUFS-LIU? | No. Full-pool mixed-v2 remained best, and filtering reduced both final AUROC and DUFS's incremental value. |

The common mechanism-level lesson is important: stable covariance, graph,
family, or replicate structure is not automatically structure that identifies
answer correctness. Several methods optimized agreement or stability without
knowing which stable component was target-relevant.

## 4. Evidence for freezing the core

On the frozen 24-cell global benchmark:

- deployed U-PCR: 0.7735 macro AUROC;
- IU-PCR: 0.7741;
- DUFS-LIU: 0.7741;
- atomic conditional-alignment variant: +0.023 percentage points;
- micro-view conditional-alignment variant: -0.363 percentage points.

Other mechanism tests support the same conclusion:

- label-only family expertise oracle: +2.833 points, but deployable GCFR:
  -0.135 points;
- repeated cross-view diffusion: +0.004 points;
- repeated-measurement Wiener DUFS-LIU: +0.0006 on GSM8K and +0.0013 on
  MATH, with paired intervals containing zero;
- direct generalized latent coordinates removed the off-diagonal covariance
  that U-PCR requires and caused failure;
- full-pool mixed-v2 DUFS-LIU: 0.776562, compared with 0.774249 after the
  deployed `rho_max/3` hard filter and 0.764153 after the strictest filter;
- DUFS minus matched IU-PCR: +0.048 AUROC points without filtering and -0.025
  points after the deployed filter.

The hard-filter mechanism is also informative. The median Spearman agreement
between estimated rho and the full-pool DUFS gate is 0.794. Hard deletion
therefore mostly duplicates the soft DUFS suppression, but also removes the
feature from covariance estimation and final fusion. All previous IU-PCR,
DUFS-LIU, and deployed-U-PCR scores reproduced exactly in 24/24 cells.

These numbers justify stopping the current search pattern. They do not justify
a universal negative statement about spectral fusion.

## 5. Application 1: hallucination localization

### Frozen result

GL-LIU v1 uses:

1. global mixed-v2 DUFS-LIU to estimate whether the answer contains an error;
2. temporal LIU to estimate the first erroneous step;
3. a calibrated decision rule that combines detection and localization.

Under the shared ProcessBench protocol:

| Method | ProcessBench F1 |
|---|---:|
| Mind the Gap | 25.71% |
| Frozen GL-LIU v1 | 31.36% |
| Unified global/local core-five DUFS-LIU | 31.72% |
| Unified broad-28 local DUFS-LIU | 29.03% |

On the six cells not used for component selection, frozen v1 scores 30.76% and
unified core-five scores 31.41%. Local exact accuracy is 26.41% for temporal
LIU and 26.70% for core-five local DUFS-LIU.

### Claim boundary

- The eight cells contain four independent dataset families reused across two
  model sizes. They are not eight independent datasets.
- The +0.37-point unified-core advantage is descriptive and does not replace
  frozen v1 without a new external test.
- Labels are absent from score fitting, but they are used for component
  selection, split-local threshold calibration, and final evaluation.
- Therefore the accurate phrase is **calibrated unsupervised scoring**, not a
  fully label-free end-to-end decision system.

### Next localization test

Use a new dataset/model family. Keep both systems frozen:

- control: global mixed-v2 DUFS-LIU plus temporal LIU;
- candidate: global mixed-v2 DUFS-LIU plus core-five local DUFS-LIU.

Report global AUROC, exact first-error accuracy, tolerance-one accuracy,
clean-trace accuracy, ProcessBench F1, per-cell results, paired uncertainty,
and score/threshold failures. Do not tune local views, graph parameters, or
threshold rules on the current eight cells again.

## 6. Application 2: hallucination in RAG citations

The proposed application is **evidence-contrast U-PCR/DUFS-LIU**.

For one already generated answer, construct several score traces:

1. score the answer with all retrieved evidence;
2. score it without retrieved evidence;
3. score it repeatedly while leaving out one chunk or citation source;
4. optionally contrast supported and contradicted evidence when the benchmark
   provides this structure.

The answer remains fixed. The intervention changes only the evidence context.
Evidence sensitivity is the desired signal here, not nuisance covariance to
remove. The dependent intervention traces can be fused with DUFS-LIU for:

- global unsupported-answer detection;
- citation-level attribution;
- token or span localization of unsupported content.

This proposal is a new measurement contract, not another small fusion variant.
It can succeed even if the core remains unchanged because it creates more
identifiable target information.

### Required preregistration

Before implementation, freeze:

- a primary benchmark with response- and span-level grounding labels;
- grouping by source or question, not random answer rows;
- the exact label boundary for fitting, calibration, and final evaluation;
- external baselines and their code/model revisions;
- global and local metrics;
- failure tests for short answers, weak retrieval, citation sparsity, and
  evidence-order sensitivity;
- a confirmation dataset not used for method or threshold selection.

The old Phase-10 RAG cache is exploratory only. It prompts from the first 15 of
a median 223 documents, contains gold evidence in those 15 documents for only
25 of 240 answers, and has explicit citations in only 19.2% of answers. Its
substring fallback label is not semantic support. Its reported V4 AUROC of
0.756 must not be presented as validation of citation grounding.

## 7. What is paused

Do not open another core variant based only on:

- a new graph kernel;
- a new semantic family partition;
- agreement or stability as an unsupervised proxy;
- another value of latent rank without an identifiable model check;
- another transformation chosen after reading the same 24-cell labels;
- another hard-filter threshold or gate fitted on the same 24 cells.

Reopen core research only if there is a new identifiable signal, a valid
nuisance intervention, materially different features, or an external failure
that clearly diagnoses a missing mechanism.

## 8. Source artifacts to verify

1. `HISTORY.md`, Steps 225--235.
2. `PROGRESS.md`, Steps 234--235.
3. `Research_Directions.md`, current decision and application priorities.
4. `docs/research_notes/dufs_liu_leading_direction.md`.
5. `docs/research_notes/frozen_24cell_view_fusion_conclusion.md`.
6. `docs/research_notes/family_relevance_diagnostic_conclusion.md`.
7. `docs/research_notes/repeated_cross_view_diffusion_conclusion.md`.
8. `results/repeated_measurement_reliability/REPORT.md`.
9. `docs/research_notes/localization_research_handoff_2026-08-08.md`.
10. `results/ours_only_localization_v1/REPORT.md`.
11. `docs/experiments/GL_LIU_FACTORIAL_V2.md`.
12. `docs/research_notes/evidence_contrast_upcr_rag_direction.md`.
13. `results/hard_filter_dufs_liu_24cell/REPORT.md`.
14. `results/hard_filter_dufs_liu_24cell/MECHANISM_ANALYSIS.md`.

## 9. Questions for the reviewer

Please answer explicitly:

1. Is the bounded saturation claim supported by the listed evidence?
2. Are any numerical or causal claims stronger than the artifacts support?
3. Does the hard-filter evidence justify closing threshold and gating variants,
   and is the proposed mechanism interpretation stated cautiously enough?
4. Should frozen GL-LIU v1 remain the formal localization system while the
   unified core-five system is tested as the simpler external candidate?
5. Is evidence-contrast fusion a meaningful application-specific extension,
   rather than a disguised repeat of the failed stability objectives?
6. Which benchmark, baseline, or failure mode is missing from the RAG-citation
   preregistration?
7. Are there contradictions between this decision, `HISTORY.md`,
   `PROGRESS.md`, and `Research_Directions.md`?

## 10. Statements that must not appear in the final research claim

- “DUFS-LIU is proved optimal.”
- “The Laplacian always improves U-PCR.”
- “The unified localization system is confirmed better than frozen v1.”
- “The localization evidence contains eight independent datasets.”
- “The complete pipeline is label-free.”
- “The old RAG cache proves citation-grounding performance.”
- “All possible U-PCR improvements have been exhausted.”

The intended final statement is narrower: **within the current single-pass
static feature setting, additional unsupervised structural refinements did not
produce a reliable core gain; application-specific measurements are now the
more evidence-based research direction.**
