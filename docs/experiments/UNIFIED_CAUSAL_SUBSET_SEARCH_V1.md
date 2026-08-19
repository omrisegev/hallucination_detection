# Unified Causal IU-PCR subset search v1 — completed search record

**Status:** completed, retrospective supervised development

**Parent protocol:** [`UNIFIED_CAUSAL_IU_V1.md`](UNIFIED_CAUSAL_IU_V1.md)

**Winner:** `base7_full28`, reported as **Unified-28**

## Purpose

The unified causal pilot exposed 1,036 source-transform coordinates. The
subset cycle asked whether a structured, task-balanced roster could preserve
the Localization gain while avoiding the Global and Early degradation caused
by the full bank, and whether DUFS-LIU or learned task weights improved the
selected ordinary-IU roster.

This cycle was authorized to use the existing labels for development. Its
selection result is therefore a discovery result, not untouched validation.

## Search principles

The search followed these principles:

1. Select a single roster shared by Global, Localization and Early.
2. Fit ordinary IU-PCR first with the trajectory accumulator fixed, isolating
   the effect of the feature roster.
3. Compare candidates by a three-task vector rather than a single pooled
   metric.
4. Reject candidates breaching the frozen regression margins.
5. Prefer the maximin/Pareto compromise; use smaller state as a tie-breaker.
6. Only after roster selection, test DUFS lambda and task reweighting on the
   same roster.
7. Select on grouped Qwen development and freeze before Llama transfer.

The source question was the split and bootstrap unit. Scorer copies were kept
together and families were equally weighted.

## Structured candidate families

The search represented the large combinatorial space using structured
families rather than attempting all subsets of 1,036 coordinates:

- raw-nine and raw-seven causal channels;
- core and broad source groups;
- level-only controls;
- multiscale EWMA and fast/slow groups;
- sustained evidence (`ewma`, positive area and persistence);
- moving-window moment groups;
- change-detection groups including innovation, CUSUM, Page-Hinkley and
  BOCPD;
- full-bank and no-BOCPD controls;
- source-family additions and removals around surviving transform families.

The historical six-family representation was kept conceptually separate: it
is a compression by provenance means, not literally a subset of the 1,036
coordinates. Likewise the classic mixed-v2 29/30-feature head is an external
baseline, not a slice of the causal bank.

## Selection objective

For a roster `S`, the primary development vector was:

```text
d(S) = [Global AUROC, ProcessBench macro-F1,
        mean Early AUROC at budgets 64 and 128]
```

Candidate deltas were normalized by the frozen margins
`[0.010, 0.010, 0.015]`. Selection favored the largest worst normalized delta
while retaining the Pareto frontier over Global, Localization, Early and
state size. Exact feature overlap across repeats was not required because
strongly correlated features may be exchangeable; direction, group inclusion
and score stability were the meaningful stability quantities.

## DUFS-LIU and weighting follow-up

DUFS was intentionally not multiplied across the broad subset search. It was
tested after the roster narrowed, using shared graph/gate computation and a
lambda path extending through 3. Learned task weights were also evaluated as a
post-roster alternative.

The interpretation rule was strict:

- a development-only gain was insufficient;
- a variant had to survive the frozen scorer transfer;
- the accumulator and roster could not change across lambda values.

Neither DUFS-LIU nor learned task weighting survived Qwen-to-Llama transfer.
Ordinary IU-PCR remained the selected fusion mechanism.

## Aggregation correction

An early report incorrectly pooled out-of-fold scores before computing the
metrics. Because scores from separately fitted folds do not share a common
scale, that aggregation was invalid.

Withdrawn Qwen values:

| Global | Localization | Early |
|---:|---:|---:|
| 0.7012 | 0.3278 | 0.5435 |

Corrected foldwise, equal-family Qwen development values:

| Global | Localization | Early |
|---:|---:|---:|
| 0.6914 | 0.3040 | 0.5301 |

Only the corrected values may be quoted. The bug changed aggregation and
reported values; it did not change the selected `base7_full28` roster.

## Selected roster

`base7_full28` contains seven causal base streams crossed with four transforms:

| base stream | level | ewma16 | positive_area | persistence |
|---|:---:|:---:|:---:|:---:|
| entropy | ✓ | ✓ | ✓ | ✓ |
| negative log-sum-exp | ✓ | ✓ | ✓ | ✓ |
| negative top-1 | ✓ | ✓ | ✓ | ✓ |
| negative top1-top2 margin | ✓ | ✓ | ✓ | ✓ |
| top-k varentropy | ✓ | ✓ | ✓ | ✓ |
| top-k Renyi-2 | ✓ | ✓ | ✓ | ✓ |
| top-k tail mass | ✓ | ✓ | ✓ | ✓ |

This gives 28 coordinates. The full bank was not the right operating point;
feature abundance did not translate into robust three-task performance.

## Frozen transfer result

| Llama transfer task | Unified-28 | matched task incumbent | delta |
|---|---:|---:|---:|
| Global AUROC | 0.6629 | 0.6870 | -0.0241 |
| Localization macro-F1 | 0.2880 | 0.2419 | +0.0461 |
| Early AUROC | 0.5587 | 0.5777 | -0.0189 |

Localization improved significantly. Global and Early breached their frozen
non-inferiority margins. The final decision is therefore:

```text
DO_NOT_PROMOTE_UNIFIED_CAUSAL_V1_REGRESSES_GLOBAL_AND_EARLY_INCUMBENTS
```

Unified-28 remains the method of record when one shared causal method is
required, but it does not replace the dedicated task heads.

## Separate Global replay

The 24-cell Global replay is context, not a score for Unified-28:

| method | macro Global AUROC |
|---|---:|
| ordinary 36-feature IU-PCR | 0.7591 |
| frozen mixed-v2 DUFS-LIU | 0.7766 |
| delta | -0.0175, 95% CI [-0.0477, +0.0140] |

Math was effectively tied (0.7869 versus 0.7862); QA drove the loss (0.7128
versus 0.7604). DUFS lambda 0.3 and 1.0 scored 0.7575 and 0.7552, and learned
reweighting was worse. These results must not be pooled with the Unified-28
ProcessBench transfer table.

## Final interpretation

- A bounded supervised search was justified because the causal DSP bank was
  materially different from the old static 30-view pool.
- The search found a compact roster with a real Localization advantage.
- The evidence does not support one method replacing all dedicated heads.
- More aggressive DUFS lambda or generic feature expansion is not supported by
  the transfer evidence.
- The next scientific priority is fair, lane-specific comparison against
  published competitors using identical rows and evaluators.

## Artifact note

The original runner, fold checkpoints, decision JSON and bootstrap bundle were
created in a temporary worktree but were not committed before that worktree
disappeared. This document records the frozen scientific outcome and the known
aggregation correction. It does not pretend that the original machine-readable
bundle remains in the repository.
