# Direct probability-rank fusion v1 (gray-box)

**Frozen before outcomes:** 2026-09-10  
**Scope:** cached final-layer output probabilities only; no hidden states, attention, or white-box capture.

## Isolated source

- Worktree: `.worktrees/direct-probability-fusion-v1`
- Branch: `codex/direct-probability-fusion-v1`
- Joint/localization base: `claude/joint-lsml-optimization-v2` at `ae420bcb`
- Token/local-fusion line merged: `codex/token-local-fusion-optimization-v1` at `b214217a`
- Later uncommitted research files were copied from the main working checkout and hashed in
  `DIRECT_PROBABILITY_FUSION_V1_SOURCE_SNAPSHOT.json`.
- Large caches and frozen benchmark artifacts are read-only inputs from the explicit
  `--source-root`. All new code, checkpoints, metrics and reports stay inside the worktree.

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

## Fusion methods

1. **Token Entropy**: existing token entropy and top-10 readout.
2. **Direct Probability Fusion - Equal Weights**: equal weight over standardized direct rank
   coordinates.
3. **Direct Probability Fusion - IU-PCR**: answer-local (localization) or cell-local (complete
   answer) IU-PCR over the 15 ranks. This is the primary candidate because it is the simplest
   direct transfer of the established fusion core.
4. **Direct Probability Fusion - Joint Shrinkage**: the same IU solver after the existing Joint
   rank-one covariance target and label-free Ledoit-Wolf shrinkage rule. This tests whether
   Claude's latest small Joint-inspired gain also transfers to probability ranks.

Internal code IDs may appear in machine-readable files, but plots and tables must use the names
above.

No graph, lambda, K, readout, gate, feature, or detector sweep is permitted in v1.

## Track A: one-answer localization

For each answer independently, tokens are observations and probability ranks are fusion views:

`T tokens x 15 probability ranks -> answer-only fusion -> token risk -> top-10 mean per step`.

Use the complete matched v3 population: 6,800 ProcessBench model-answer rows in eight cells and
6,969 PRMBench answers, for 13,769 rows in total. Preserve the existing row IDs, corrected step
annotations, canonical source groups, folds and exclusions. ProcessBench uses the already frozen
mean-entropy `q=0.3` fold gate for every arm. Report all-eight, Q4, Q8 and every cell,
raw exact location, within-one, clean accuracy, coverage, fallbacks, PRMBench within-answer AUC,
pooled AUC and PRMScore. The stored Step334 token-entropy result must reproduce before accepting
the new result.

Run the existing paper-form **Mind the Gap locator** on the same eight ProcessBench cells and give
it the same frozen entropy gate. Label it **Mind the Gap Locator - Common Gate**. This is a common
ProcessBench-contract comparator; it is not the paper's native erroneous-trace-only SLA. Keep the
older 25.71% headline outside the matched table unless it reproduces under this v3 contract.

Show the accepted token-varentropy and token-feature IU results from Step334 as frozen references.
Show the supervised Qwen2.5-Math-PRM-7B PRMScore only in a separate reference row because it has
different supervision and compute access.

## Track B: complete-answer hallucination detection

First reproduce the exact complete-case candidate mask and order that built
`dependency_fusion_raw/cells.npz`, and verify its row count and labels. For each retained answer,
aggregate each of the 15 risk-oriented rank sequences with the same top-10 token mean. In each
historical cell, answers are observations and ranks are fusion views:

`N answers x 15 probability-rank summaries -> cell-local fusion -> answer risk`.

Evaluate all canonical 24 cells and report candidate-level AUROC, QA/math/cell macro, coverage,
weights and comparisons with the frozen historical IU-PCR rows selected by the exact
`mixed_v2 / full / iu_pcr` contract on the same 24-cell roster. The similarly named
`upcr.rho` column in `benchmark_standing.csv` is U-PCR with a sign heuristic and is not
used as the IU-PCR reference.
The primary visual comparison is **Direct Probability Fusion - IU-PCR** versus **Historical
IU-PCR**, one row per cell. Replay the exact `mixed_v2 / full / iu_pcr` score from the frozen
historical feature bundle and require its AUROC to equal the published frozen value. Do not refit
slower historical methods; differences near 0.001 are less important here than a simple,
reproducible comparison. Labels are opened only after scores are produced. The primary 10,000-draw
97.5% interval uses paired resampling of canonical problem groups inside each cell and paired
resampling of the 24 cells for the macro; every cell also reports this paired interval.
The legacy bundle uses `1=correct`; Track B changes both the target and the replayed historical
score sign so every visible output follows `1=hallucination, high score=risk` without changing
the historical AUROC.

## Output contract

Create a self-contained English HTML report with little prose, clear legends, and no development
code names in the visible plots. It must contain:

1. ProcessBench Q4, Q8 and all-eight bars.
2. ProcessBench per-cell results and error-location diagnostics.
3. PRMBench within-answer AUC, pooled AUC and PRMScore.
4. A 24-cell one-to-one chart and table versus Historical IU-PCR.
5. QA9, math15 and all24 macro results, coverage, fallbacks and runtime.

The JSON and CSV files remain the numerical source of truth. The HTML is a compact presentation of
those files.

Before scoring, compare hashes for every raw cache against `DATA_AUDIT.json`, and compare the
roster, annotations, folds, fixed gate, historical bundle and frozen reference tables against the
committed input-freeze file. Any mismatch stops the run.

## DEEM boundary

DEEM 0.2.0 accepts soft probabilities shaped as samples x latent classes x base learners. The
saved LM distribution is samples x vocabulary ranks: its columns are token alternatives, not
probabilities that independent classifiers assign to hallucination. Feeding them directly to DEEM
therefore requires a declared pseudo-classifier adapter. The current project adapter converts
continuous columns to empirical rank probabilities, which would remove the raw probability geometry
being tested here. DEEM is retained as a second-stage nonlinear fusion diagnostic if v1 establishes
signal in the direct ranks; it is not silently substituted into the v1 primary result.

## Decision rule

The representation is promising only if **Direct Probability Fusion - IU-PCR** improves over Token
Entropy on the predeclared localization comparisons without a material PRMBench regression, or
adds a consistent complete-answer gain over Historical IU-PCR across the 24-cell panel. Joint
Shrinkage is supporting evidence and may replace the primary only after a separate frozen
confirmation. A gain on pooled AUC alone is insufficient. Results remain development evidence
until frozen confirmation on untouched data.
