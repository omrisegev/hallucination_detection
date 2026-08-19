# Unified Causal IU-PCR v1 — final research report

**Date:** 2026-08-18

**Scope:** one causal IU-PCR trajectory for Global detection, first-error
Localization and Early detection

**Verdict:** retain Unified-28 as the unified method of record; do not replace
the task-specific heads

## Executive result

The experiment established that one causal IU-PCR trajectory can technically
serve all three tasks, but not yet at the accuracy of the best dedicated
methods. A compact 28-coordinate roster transfers better than the complete
1,036-coordinate DSP bank. Its Localization improvement is meaningful and
statistically supported, while its Global and Early losses exceed the frozen
non-inferiority margins.

| frozen Llama transfer | Unified-28 | matched incumbent | delta |
|---|---:|---:|---:|
| Global AUROC | 0.6629 | 0.6870 | -0.0241 |
| Localization macro-F1 | 0.2880 | 0.2419 | +0.0461 |
| Early AUROC | 0.5587 | 0.5777 | -0.0189 |

The Localization delta has a positive grouped interval in the surviving
summary. Global and Early are not merely uncertain ties: both breach their
predeclared regression margins.

## What was learned

### 1. One trajectory is feasible

The causal construction is coherent: the same frozen IU-PCR evidence is
available at each prefix, its local update defines a token locator, and its
terminal value defines the Global score. No task requires final trace length
or a future-looking window.

### 2. More features are not automatically better

The 1,036-coordinate bank mixed levels, multiscale smoothers, moments,
innovations, persistence and change detectors across raw and broad token
sources. Its dimensionality created redundancy and unstable task tradeoffs.
The transferable compromise was only 28 coordinates: seven raw streams with
`level`, `ewma16`, `positive_area` and `persistence`.

### 3. Localization and Early are related but not identical

Both consume the token trajectory, but their losses reward different behavior.
Localization needs a sharp and correctly placed positive update. Early
detection benefits from stable accumulated separation while controlling the
maximum false-warning probability over the whole prefix horizon. A smoother
that improves early separation can blur the peak; a sharp locator can create
too many early false alarms.

### 4. The terminal stream does not recover the best Global head

Although `R_T` is a valid Global score, removing completion-only information
and forcing one causal representation sacrifices information used by the
dedicated Global detector. This is the empirical reason the conceptual
convergence of the tasks did not become performance equivalence.

### 5. DUFS did not rescue transfer

DUFS-LIU and learned task weights improved selected Qwen development cells,
but their gains did not survive frozen Llama transfer. Increasing lambda is
therefore not supported as the next default action.

## Corrected development result

The originally displayed Qwen values 0.7012/0.3278/0.5435 were produced by
pooling scores across separately fitted folds. They are withdrawn. Metrics
must be computed within fold and then aggregated with equal family weight.

The corrected Qwen development values are:

| Global AUROC | Localization F1 | Early AUROC |
|---:|---:|---:|
| 0.6914 | 0.3040 | 0.5301 |

The correction did not change the selected roster or the frozen Llama
decision.

## Relationship to the dedicated incumbents

- **Global:** the matched classic mixed-v2 head without length reaches 0.6870
  on the frozen Llama panel; Unified-28 loses 0.0241.
- **Localization:** max-entropy plus top-five reaches 0.2419; Unified-28 gains
  0.0461.
- **Early:** max entropy reaches 0.5777; Unified-28 loses 0.0189. Unified-28
  does beat the matched IU28 row by 0.0213, but max entropy is the stronger
  incumbent on this exact panel.

The task-specific references must be reported independently. A single mixed
leaderboard across AUROC, ProcessBench F1 and token savings would be invalid.

## Relationship to the separate 24-cell Global experiment

Ordinary 36-feature IU-PCR reaches 0.7591 macro AUROC and the frozen mixed-v2
DUFS-LIU reference reaches 0.7766, a delta of -0.0175 with 95% CI
[-0.0477,+0.0140]. Math is tied; QA drives the regression. This experiment has
a different population and contract and is not a replacement Global score for
Unified-28.

## Decision and next use

The frozen decision is:

```text
DO_NOT_PROMOTE_UNIFIED_CAUSAL_V1_REGRESSES_GLOBAL_AND_EARLY_INCUMBENTS
```

Use Unified-28 when the research question explicitly requires one causal,
single-pass method shared by all three tasks. Continue to use the dedicated
heads as the accuracy incumbents. The next cycle should compare both against
external methods on identical rows, with separate lanes for:

1. final-answer Global detection;
2. ProcessBench first-error Localization;
3. causal prefix detection;
4. stopping/adaptive compute.

That comparison program is specified in
[`HANDOFF_fair_paper_exact_comparisons_2026-08-18.md`](../../HANDOFF_fair_paper_exact_comparisons_2026-08-18.md).

## Evidence and artifact ledger

| item | repository status |
|---|---|
| final conclusion and corrected metrics | committed in `PROGRESS.md`, `HISTORY.md` and `Research_Directions.md` |
| frozen comparison handoff | committed |
| protocol and subset-search narrative | committed with this report |
| source implementation and focused tests | committed under `spectral_utils/unified_causal_*` and `scripts/` |
| final per-row records and decision/bootstrap bundle | committed under `results/unified_causal_subset_*` |
| fair identical-row replay | accepted Step-279 package under `results/fair_paper_exact_comparisons_v1/` |
| original exploratory fold checkpoints | not all retained; no claim depends on rebuilding search selection from them |
| local/Drive and source provenance | hash-bound by the fair package registries, source audit, and temporary-worktree manifest |
| historical console logs | not retained; focused tests and deterministic package artifacts are retained |

This ledger is intentional. The selected method and publication comparisons
are executable and independently rebuild-verified; the ledger prevents a
future reader from confusing that final closure with complete retention of
every exploratory development checkpoint.
