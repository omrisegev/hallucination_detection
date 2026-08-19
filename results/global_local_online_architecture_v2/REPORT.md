# Token-native Global-Local-Online architecture v2

## Decision

**Retain a two-head ordinary-IU architecture: the historical mixed-v2 Global
head plus the new raw token-level Local head. Use an equal 0.50/0.50
calibration-standardized Global/Local detector and the peak locator. Derive the
Online score causally from the prefix Global score and running maximum Local
evidence. Do not retain a third independent Online head, DUFS, or a Laplacian.**

This is a retrospective development decision over existing caches, not fresh
confirmation. It is an unsupervised scorer with calibrated decision policies.
No inference, GPU work, or Drive mutation occurred.

## What the broader search changed

Step 270's narrow conclusion is preserved: its three aggregate-of-aggregate
coarse-grid mechanisms stay closed. The v2 search starts from raw token
telemetry and changes all three previously frozen axes.

- Global independently selects `g_registered_mixed` at **0.7895** development AUROC. The best raw mean/tail replacement reaches 0.7560; its delta is -0.0335 with 95% CI [-0.0600,-0.0065]. The full-trace spectral transformations still matter for Global.
- Local selects `l_level9` at **0.3484** ProcessBench F1. Onset-only (0.2464) and level+onset (0.2685) are worse. The registered core-five replay is 0.3330 and statistically tied, but outside the frozen 0.005 simplicity window.
- Online independently selects `o_ewma_area_persist27` at **0.6596** 64/128 AUROC versus 0.6370 for registered IU28, a +0.0225 point difference whose paired interval still crosses zero. Instantaneous/onset variants lose; sustained EWMA, positive area, and persistence are the useful dynamic mechanism.
- The harness then makes the independent Online head unnecessary: the two-head Global+Local derivation is within every development margin of three heads, so the cheaper architecture wins. The old 0.75/0.25 blend is not selected; 0.50/0.50 with the peak locator is the frozen development choice.

## Twelve-cell architecture result

The table equal-weights the twelve scorer-model/family cells. Scorer copies are
repeated measurements; grouped intervals resample each source question once and
carry all scorer copies together before equal family weighting.

| label | global | local | online |
|---|---|---|---|
| one shared | 0.6892 | 0.2397 | 0.6009 |
| two Global+Local | 0.7164 | 0.3136 | 0.6075 |
| three independent | 0.7164 | 0.3136 | 0.6009 |

| comparison | task | delta | 95% CI | family W/L |
|---|---|---|---|---|
| two Global+Local − one shared | global | +0.0271 | [+0.0085, +0.0449] | 4/0 |
| two Global+Local − one shared | local | +0.0740 | [+0.0458, +0.1013] | 4/0 |
| two Global+Local − one shared | online | +0.0067 | [-0.0121, +0.0260] | 3/1 |
| three independent − two Global+Local | global | +0.0000 | [+0.0000, +0.0000] | 0/0 |
| three independent − two Global+Local | local | +0.0000 | [+0.0000, +0.0000] | 0/0 |
| three independent − two Global+Local | online | -0.0067 | [-0.0248, +0.0126] | 1/3 |

Relative to one shared head, two heads improve Global by
**+0.0271** [+0.0085,+0.0449]
and Local by **+0.0740** [+0.0458,+0.1013].
The Online delta is +0.0067
[-0.0121,+0.0260].
Adding the third independent Online head changes only Online and is
-0.0067
[-0.0248,+0.0126]
versus two heads. It does not earn its 27 features and 36 state scalars.

![Architecture summary](ARCHITECTURE_SUMMARY.png)

## Fusion decision

All graph rows use the exact selected matrix, preprocessing, two-dimensional IU
subspace, and reducer. `lambda=0` is bit-identical in every path.

| method | Global | Local | Online |
|---|---|---|---|
| global_ordinary__local_ordinary | 0.7893 | 0.3613 | 0.6571 |
| global_dufs__local_ordinary | 0.7908 | 0.3626 | 0.6576 |
| global_ordinary__local_uniform | 0.7893 | 0.3673 | 0.6581 |
| global_ordinary__local_dufs | 0.7893 | 0.3673 | 0.6581 |
| global_ordinary__local_temporal | 0.7893 | 0.3673 | 0.6578 |

DUFS changes Global by only +0.0014 AUROC.
The best Local graph increment is about
+0.0059 F1.
The paired intervals for these increments cross zero, while DUFS fit costs up to
24.9×
the uniform path in the measured development cells. Ordinary IU-PCR is the
supported fusion choice; neither DUFS nor temporal/uniform Laplacians are needed.

## Causal warning and length failure tests

Declaration thresholds are calibrated on the maximum score over the entire
absolute-budget horizon. They control trace-level ever-warning, not a per-time
FPR.

| target FPR | observed false warning | wrong warning coverage | mean first budget | potential remaining tokens |
|---|---|---|---|---|
| 5% | 3.6% | 14.7% | 238.2 | 428.2 |
| 10% | 8.1% | 25.0% | 239.7 | 413.9 |

The policy is useful but modest: it catches roughly one quarter of wrong traces
at the 10% target. Remaining-token numbers are potential exposure only, not
realized savings, because no forced-closure branches were generated.

| budget | length Spearman | raw AUROC | residual AUROC | short | medium | long |
|---|---|---|---|---|---|---|
| 64 | 0.1007 | 0.5947 | 0.5801 | 0.5823 | 0.5920 | 0.5764 |
| 128 | 0.1749 | 0.6204 | 0.5980 | 0.6183 | 0.6117 | 0.5716 |

Length correlation is small to moderate. Calibration-only isotonic
residualization lowers AUROC but leaves above-chance signal, and every length
band remains informative at 64/128 tokens. The effect is not merely final-length
prediction, although long-trace discrimination is the weakest band.

The Phase-15 MATH-500 T=1.0 cache, which lacks log-sum-exp, is an Online-only
transfer: AUROC is **0.5142** at 64 and **0.5555** at 128 tokens,
with final AUROC 0.8368. This is a clear failure of robust
early transfer, not evidence for a new-model win.

## Feature requirements and cost

The Global missing-family audit identifies top-k log-probabilities as the main
increment: removing them costs about 0.0266 development AUROC. The selected
Online sustained head is insensitive to a single primitive (largest drop about
0.004). For Local, dropping spilled or top-k entropy improves development F1;
because those are outcome-opened diagnostics, no post-hoc pruned head is
promoted. They define the next frozen subset roster.

Median per-cell fit time is 70.1s and
median complete calibration+evaluation three-output scoring time is
47.2s on local CPU. The long-family
maxima are 89.4s fit and
56.8s score. Profiling shows the
historical mixed-v2 Global prefix recomputation, not the O(1) token-native Local
state, is the bottleneck. Therefore the selected system is the statistical
winner in this roster but not yet a fully optimized streaming implementation.

## Claim boundary and next gate

- All twelve ProcessBench cells and Phase-15 are historically opened. The
  non-selection application protects this run's mechanics but is not fresh
  confirmation.
- Global/Online final-answer wrongness and Local trace-error presence are
  distinct labels; no metric substitutes one for the other.
- PRMBench remains a separate Local-only anchor (existing frozen step AUROC
  0.6711); v2 did not refit a PRMBench-compatible Global/Online task.
- The historical Local core replay contains a completed-trace CUSUM curve and
  is not called suffix-invariant. Every new v2 recurrence and every deployable
  Online score passes suffix/chunk replay tests.
- The 0.50/0.50 blend, Local subset pruning, and the weak Phase-15 early transfer
  require fresh confirmation before a deployment or paper-level claim.

No GPU run is justified merely to add DUFS or the third head. A narrowly scoped
fresh-data run may be justified for the frozen two-head ordinary-IU architecture
only after preregistering a cheaper causal implementation of the mixed-v2
Global prefix and the Local drop-one subset candidates. That future run needs
explicit approval.
