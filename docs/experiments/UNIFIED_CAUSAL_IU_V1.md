# Unified Causal IU-PCR v1 — frozen protocol and implementation contract

**Status:** completed retrospective supervised-development cycle

**Decision date:** 2026-08-18

**Method of record:** `Unified-28`, ordinary IU-PCR

**Promotion decision:** not promoted over the task-specific heads

## Research question

Can one causal, stateful score run over the complete token stream and support
all three hallucination tasks without changing its feature weights?

1. **Global detection:** use the final trajectory value to predict final-answer
   wrongness.
2. **First-error localization:** use the token with the strongest positive
   contribution to the trajectory update.
3. **Early detection:** use the prefix trajectory and its first calibrated
   threshold crossing.

The intended identity is that the Global score is not a separate head:
`global_score == R_T`, where `R_t` is the same causal risk process exposed at
every prefix.

This cycle deliberately allowed labels for feature and hyperparameter
development. The resulting method is therefore accurately described as a
**supervised-developed, IU-PCR-fused, causal streaming method**. It is not a
fully label-free confirmation result.

## Stateful method contract

The conceptual API is:

```text
fit(development_traces)
    freeze reference statistics, feature roster, signs and IU-PCR weights

update(token_telemetry)
    return instantaneous evidence e_t, causal risk R_t and warning status

finalize()
    return R_T, localization token and first alarm token
```

The full trajectory may be persisted for evaluation and audit, while online
execution needs only constant state per retained causal filter.

## Input contract and causality

The candidate bank began with nine risk-oriented token streams:

1. entropy;
2. sampled-token spilled energy;
3. negative log-sum-exp;
4. negative top-1 log-probability;
5. negative top1-top2 margin;
6. top-k entropy;
7. top-k varentropy;
8. top-k Renyi-2 entropy;
9. top-k tail mass outside the leading five saved entries.

The broader source pool also included the 28 existing token views grouped as
entropy level, entropy dynamics, structural dynamics, sampled energy,
partition energy and top-k distribution views.

All transforms are prefix-only. Final trace length, future windows and any
statistic requiring suffix tokens are forbidden. The original candidate
design included:

- level;
- EWMA with horizons 4, 8, 16, 32 and 64;
- fast-minus-slow contrasts 4/16, 8/32 and 16/64;
- causal moving mean, variance and MAD at 8, 16, 32 and 64;
- innovation against a slow EWMA;
- normalized positive area and persistence;
- one-sided CUSUM and Page-Hinkley;
- BOCPD with hazards 1/50 and 1/100.

Crossing the 37 source coordinates with the causal transforms produced a
1,036-coordinate DSP bank. This full bank was a search space, not the final
method.

## Information-atlas development rule

Feature usefulness was investigated for three targets:

- final-answer wrongness at completion and at fixed prefix budgets;
- whether a token/step is the first annotated error;
- incremental information beyond entropy, `sw_var` and IU28.

Raw mutual information was descriptive. The primary intended criterion was
held-out nonlinear log-loss gain conditioned on budget, dataset and scorer
model, with grouping and resampling at source-question level. Family-level
joint gains were included because univariate MI can miss synergy.

The cycle did not treat a nominal per-feature FDR pass as the sole inclusion
gate. With 200 permutations over roughly 3,108 feature-target tests, the
minimum attainable p-value is too coarse for a strict sparse BH screen.
Repeated grouped held-out gain, task balance and transfer were therefore the
substantive selection evidence.

## IU-PCR fusion contract

For each fit partition:

- reference statistics and robust scaling are fit on that partition only;
- traces receive equal weight rather than weight proportional to length;
- 32 equal positions per trace are used for fitting;
- feature signs and order are frozen after development;
- IU-PCR uses two components, L2 fitting and no graph/Laplacian term;
- the same weights and signs are applied at every token and at completion.

Ordinary IU-PCR was searched first. DUFS-LIU and task reweighting were applied
only as second-stage alternatives so that feature selection and fusion effects
were not conflated.

## Trajectory outputs

Let `e_t` denote the instantaneous IU-PCR evidence. Candidate risk paths were:

1. identity, `R_t = e_t`;
2. leaky accumulation with recovery, horizons 8/16/32/64 and drift
   0/0.25/0.5;
3. irreversible cumulative hazard as a control.

The frozen task interpretations were:

- **Global:** `R_T`;
- **Localization:** the token with maximum positive contribution to the
  update of `R_t`;
- **Early:** `R_t` at absolute token budgets and the first calibrated crossing.

First crossing and a three-token persistent crossing were localization/alarm
ablations. False-warning thresholds were calibrated on the maximum monitored
trajectory, not independently at each token.

## Data and evaluation

The primary shared population was the ProcessBench intersection carrying:

- token telemetry;
- final-answer correctness;
- first-error or clean annotation;
- the four ProcessBench families;
- Qwen development and frozen Llama scorer transfer.

The split and uncertainty unit was the source question. All scorer copies of a
question stayed together. Every label-aware operation—including references,
signs, roster choice, IU fit, thresholds, accumulator and DUFS lambda—belonged
inside the applicable grouped training partition.

Primary metrics:

- **Global:** AUROC, AUPRC and fixed-FPR operating points;
- **Localization:** ProcessBench macro-F1, exact, within-one and clean
  abstention;
- **Early:** AUROC/AUPRC at 16/32/64/128/256/512, first-alarm time and
  ever-warning FPR;
- **Efficiency:** update time and state size.

Frozen non-inferiority margins were 0.010 Global AUROC, 0.010 Localization F1
and 0.015 Early AUROC. A universal finalist had to improve at least one task by
0.010 without breaching either other task's margin.

## Required audits

The experiment contract required:

- suffix invariance;
- tokenwise/chunked endpoint identity;
- no final-length or future-window leakage;
- `R_T` identical to the reported Global score;
- source-question grouping and split isolation;
- label permutation removing information gain;
- feature-order invariance;
- deterministic missing-channel handling;
- synthetic spike, drift, persistent-error and recovery cases.

The completed cycle reports 39 focused and regression tests passing. The
historical machine-readable test outputs and source implementation were not
included in the surviving commit; this is an artifact-retention gap, not a
claim that those files remain reproducible from the current tree.

## Frozen outcome

The full 1,036-coordinate bank was rejected. The strongest transferable
single-method candidate was ordinary **Unified-28**:

- seven raw streams: entropy, negative log-sum-exp, negative top-1,
  negative margin, top-k varentropy, top-k Renyi-2 and top-k tail mass;
- four causal transforms per stream: `level`, `ewma16`, `positive_area` and
  `persistence`.

Spilled energy and top-k entropy were therefore absent from the frozen
seven-stream roster. This is a supervised-developed roster and must not be
described as independently confirmed.

See the complete decision report in
[`docs/reports/UNIFIED_CAUSAL_IU_V1_REPORT.md`](../reports/UNIFIED_CAUSAL_IU_V1_REPORT.md)
and the subset-search record in
[`UNIFIED_CAUSAL_SUBSET_SEARCH_V1.md`](UNIFIED_CAUSAL_SUBSET_SEARCH_V1.md).

## Reproducibility status

The scientific contract, corrected results and decision are committed. The
original temporary worktree containing the implementation, fold checkpoints
and detailed machine-readable result bundle was not committed and is no
longer present. Consequently:

- the numerical conclusions are preserved in the official handoff documents;
- the present repository does not contain a byte-for-byte replay package for
  this cycle;
- a future replay must reimplement this frozen contract and must not silently
  retune the roster or quote the withdrawn pooled metrics.
