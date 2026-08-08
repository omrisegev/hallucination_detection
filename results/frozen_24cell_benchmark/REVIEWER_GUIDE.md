# Independent review guide for the frozen 24-cell benchmark

## Purpose

Review the experiment without accepting the generated conclusion. The reviewer
should inspect the registered configuration, recompute the tables from frozen
scores, and look for leakage, unfair comparisons, silent failures, and claims
that exceed the evidence.

## Order of review

1. Read `RUN_DEFINITION.json`. Confirm that it lists exactly 24 cells, 9 QA and
   15 math, and that `scientific_run` is true.
2. Read `FIT_COMPLETE.json` and `SCORE_FREEZE_MANIFEST.json`. Recompute SHA-256
   for every score file. Confirm that no score checkpoint contains labels.
3. Inspect the method documents under `docs/methods/`. For every equation, mark
   whether it comes from a paper or is a project extension.
4. Re-run `python3 scripts/frozen_24cell_report.py`. The CSV files and figures
   should reproduce without fitting a model again.
5. Check that no method receives a per-cell sign flip, a different feature pool,
   or more favorable rows. Confirm that all LIU arms equal IU-PCR exactly at
   lambda zero.
6. Check AUROC and AUPRC independently from the raw scores and bundle labels.
7. Inspect the per-cell deltas, lower tail, view weights, graph diagnostics,
   rank displacement, convergence, and runtime. A mean-only review is incomplete.
8. State explicitly that the 24 cells are retrospective development data. An
   independent reviewer reduces analysis bias; it does not create an unseen
   confirmation set.

## Questions the review must answer

- Does CA-SpecRaGE beat deployed U-PCR, IU-PCR, and DUFS-LIU at its predeclared
  synthetic-transfer setting (`lambda=10`), or only somewhere on the sensitivity
  path?
- If performance changes, did sample-specific alpha matter relative to the
  global, uniform, and permuted controls?
- Did the learned graph materially change IU-PCR ranks, or are methods tied
  because LIU is almost inactive?
- Are gains concentrated in one domain, dataset family, or class prevalence?
- Did any graph collapse, optimizer fail, seed disagree, or projected system
  become ill-conditioned?
- Do LOCO micro-views improve over both manual provenance families and
  duplicate-balanced atomic views at the frozen setting?
- Are the selected micro partitions stable enough to interpret, and are any
  gains preserved when the Y interface replaces the alpha interface?

## Claim boundary

Do not call a result externally validated, unbiased confirmation, or proof of
generalization. The feature contract and method development previously used
information from these cells. A positive result can justify a new-data test.
