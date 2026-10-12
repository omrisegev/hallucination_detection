# Direct probability-rank fusion v1 (gray-box)

**Frozen before outcomes:** 2026-09-10  
**Scope:** cached final-layer output probabilities only; no hidden states, attention, or white-box capture.

## Question

Does label-free fusion of the saved sorted next-token probabilities preserve useful information
that is lost when the same distribution is collapsed to token entropy?

## Fixed representation

- Use the first **K=15** sorted log-probabilities saved at every token.
- Convert them to direct probabilities with `exp(log p)`. Do **not** renormalize the retained
  probabilities before fusion; their missing mass is part of the gray-box signal.
- Orient rank 1 as risk `1-p1`; orient ranks 2 through 15 as risk `p_rank`.
- K=15 is fixed because the existing `token_entropies` were computed from top-15 probabilities.
  The cache contains at least top-50, but v1 does not search K.
- The **top-10 readout** is a different operation: it averages the ten highest token risks inside
  a reasoning step or complete answer. It does not limit the number of vocabulary ranks.

## Fusion arms

1. `entropy`: existing token entropy and top-10 readout.
2. `rank_equal`: equal weight over standardized direct rank coordinates.
3. `rank_iu`: answer-local (localization) or cell-local (complete answer) IU-PCR over the 15 ranks.
4. `rank_joint_lw`: the same IU solver after the existing Joint rank-one covariance target and
   label-free Ledoit-Wolf shrinkage rule. This is the primary direct-probability candidate.

No graph, lambda, K, readout, gate, feature, or detector sweep is permitted in v1.

## Track A: one-answer localization

For each answer independently, tokens are observations and probability ranks are fusion views:

`T tokens x 15 probability ranks -> answer-only fusion -> token risk -> top-10 mean per step`.

Use the complete matched ProcessBench and PRMBench population. ProcessBench uses the already
frozen mean-entropy `q=0.3` fold gate for every arm. Report all-eight, Q4, Q8 and every cell,
raw exact location, within-one, clean accuracy, coverage, fallbacks, PRMBench within-answer AUC,
pooled AUC and PRMScore. The stored Step334 token-entropy result must reproduce before accepting
the new result.

## Track B: complete-answer hallucination detection

For each answer, aggregate each of the 15 risk-oriented rank sequences with the same top-10 token
mean. In each historical cell, answers are observations and ranks are fusion views:

`N answers x 15 probability-rank summaries -> cell-local fusion -> answer risk`.

Evaluate all canonical 24 cells and report candidate-level AUROC, QA/math/cell macro, coverage,
weights and comparisons with the historical `upcr.rho` row on the exact roster. Labels are opened
only after scores are produced.

## DEEM boundary

DEEM 0.2.0 accepts soft probabilities shaped as samples x latent classes x base learners. The
saved LM distribution is samples x vocabulary ranks: its columns are token alternatives, not
probabilities that independent classifiers assign to hallucination. Feeding them directly to DEEM
therefore requires a declared pseudo-classifier adapter. The current project adapter converts
continuous columns to empirical rank probabilities, which would remove the raw probability geometry
being tested here. DEEM is retained as a second-stage nonlinear fusion diagnostic if v1 establishes
signal in the direct ranks; it is not silently substituted into the v1 primary result.

## Decision rule

The representation is promising only if `rank_joint_lw` or `rank_iu` improves over entropy on the
predeclared localization comparisons without a material PRMBench regression, or adds a consistent
complete-answer gain across the 24-cell panel. A gain on pooled AUC alone is insufficient. Results
remain development evidence until frozen confirmation on untouched data.
