# Reasoning Localization 0.3662 — Frozen H2/H3 PRMBench Diagnostic V1

Status: `FROZEN_BEFORE_RUN`; development-only cross-task diagnostic, not Phase
4 and not promotion evidence.

## Why this branch exists

No locally available ProcessBench source questions are fresh. Re-running the
same 3,400 questions cannot confirm H2 or H3, while the program's second
primary task is PRMBench every-step error ranking. This bounded diagnostic asks
whether the already frozen H0→H2→H3 scoring ladder has compatible behavior on
PRMBench. It does not waive the Phase-3 or Phase-4 prerequisites.

ProcessBench and PRMBench use different evaluator contracts. ProcessBench
predicts one first-error step or clean abstention per response. PRMBench ranks
all annotated steps inside error responses and reports AUROC/AUPRC. No F1,
abstention threshold, or cross-task aggregate is defined here.

## Frozen arms

- `P2F_H0_FAMILY6_TOP10_PRM`: exact five-family H0 token curve, top-ten step
  reducer, and the unchanged Phase-1 common response detector.
- `P2F_H2_CLEAN_C7_PRM`: remove sampled-token energy, remove partition
  `energy_series`, insert frozen C7 inside entropy dynamics, then apply the
  same reducer and response detector.
- `P2F_H3_EQUAL_C8_RERANK_PRM`: equal within-response rank fusion of H2 and
  frozen C8 step scores, then the same response detector.

The response detector is identical across arms. It may alter ranking between
responses, but it is constant within one response and cannot select a step.
H3 has no ProcessBench abstention role in this evaluator.

Before PRMBench labels open, H0 must reproduce the Phase-1 R2 PRMBench frozen
score at maximum absolute error `<=1e-12`; the implementation must also replay
the imported eight-Qwen H0/H2/H3 ProcessBench scores at the same tolerance.

## Evaluation

- Population: sealed `prmbench_response_qwen3_8b` input; 6,208 evaluable error
  responses and 83,280 annotated steps expected.
- Positive class: annotated error step.
- Primary metric: overall AUROC. AUPRC is required secondary.
- Required slices: all nine error families; `multi_solutions` remains visible
  as single-class rather than being converted to zero.
- Source strata: remain explicitly blocked because the sealed evaluator lacks
  `prm_train`/`prm_test` membership.
- Bootstrap: 20,000 paired whole-`source_idx` grouped draws, shared across all
  arms; Bonferroni simultaneous intervals across H2−H0, H3−H0 and H3−H2.
- Required robustness: W/T/L and worst delta across the eight evaluable error
  families, complete finite coverage, exact row/step alignment, and score
  hashes.

Practical diagnostic bounds are `+0.003` AUROC for supported benefit and
`-0.005` for material harm. A positive point estimate whose simultaneous
interval crosses zero is `PROMISING_UNCONFIRMED`, not rejected. Regardless of
the result, opened labels and outcome-selected ancestry forbid promotion.

No PRMBench result may be averaged with ProcessBench. A PRMBench gain with
uncertain or negative ProcessBench behavior is a task interaction, not a
universal winner.

