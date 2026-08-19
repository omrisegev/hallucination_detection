# Token-native Global-Local-Online architecture search — v2

Status: frozen before v2 candidate scoring, 2026-08-16.

This is a CPU-only retrospective study over already materialized telemetry. It
does not authorize new inference, a cluster/GPU job, a large download, a Google
Drive mutation, or any change to the frozen A6/PTNI program.

## Why this protocol exists

Step 270 answered a narrow question: three temporal summaries of the already
aggregated `cusum_max` and `sw_var_peak` signals, observed on a coarse monitor
grid, did not improve a frozen IU28 Online head. The Global head, Local head,
0.75/0.25 decision blend, IU settings, feature allocation, and number of heads
were not optimized. Therefore Step 270 does **not** establish that IU28 or the
frozen three-head stack is optimal, and it does not close token-native dynamic
features or architecture search.

This protocol makes three outputs co-primary and evaluates every architecture
on all three:

1. **Global:** detect whether the completed answer is wrong.
2. **Local:** detect whether a reasoning error exists and locate its first step.
3. **Online:** predict the eventual Global target from a causal unfinished
   prefix.

The target for Global/Online is `not final_answer_correct`. The Local target is
the ProcessBench first-error label (`label`, where `-1` means a certified clean
trace). They are deliberately different targets. In particular, final-answer
wrongness and the presence of a labeled trace error disagree often enough on
OlympiadBench and OmniMath that one may not substitute for the other.

All results are retrospective development evidence. Every ProcessBench family
and scorer-model copy has been opened in earlier project work. A held-back row
or family below protects the mechanics of this run, but it is not fresh
confirmation.

## Data, roles, and independence

The joint factorial uses the twelve complete ProcessBench telemetry cells:

- scorer `qwen3_4b`: GSM8K, MATH, OlympiadBench, OmniMath;
- scorer `qwen3_8b`: the same four source populations; and
- scorer `llama31_8b`: the same four source populations.

Each row has token entropy, spilled energy, log-sum-exp, saved top-k
log-probabilities, token IDs, step-token spans, a first-error label, and final
answer correctness. The three scorer copies contain the same 3,400 source
questions and answers with different scorer tokenization. They are repeated
measurements, not independent samples.

- **Roster-selection cells:** qwen3_4b/GSM8K and qwen3_4b/MATH only.
- **Non-selection cells:** qwen3_4b/OlympiadBench and OmniMath, plus all eight
  qwen3_8b and llama31_8b cells.
- **Online-only transfer:** the local Phase-15 MATH-500 Qwen-7B T=1.0 cache,
  after architecture identity is frozen. It lacks log-sum-exp, so the declared
  missing-channel path is used.
- **Local-only transfer:** PRMBench may be reported from the existing frozen
  artifact, but is not averaged with the ProcessBench factorial because it has
  no final-answer target under the same protocol.

Within every ProcessBench family, SHA-256 of the saved question ID assigns the
same question to `calibration` or `evaluation` for all three scorer models. The
first hexadecimal bit gives an approximately 50/50 split. Per-cell reference
statistics and IU weights use calibration telemetry without labels. Calibration
labels choose decision thresholds only. Candidate selection reads evaluation
labels only in the two roster-selection cells. Candidate identities and all
hyperparameters are then frozen before non-selection metrics are computed.

Bootstrap resampling is by source question within family. A resampled question
carries all available scorer-model copies together. Dataset families receive
equal weight; model copies inside a family do not multiply family weight. The
seed is `20260816`, with 2,000 grouped draws for the final intervals.

## Raw token channels and orientation

The new candidates begin from token telemetry, not from global summaries. The
following nine primitive **risk-oriented** channels are frozen:

| name | saved quantity and transform | why larger means more risk |
|---|---|---|
| `entropy` | saved token entropy | frozen `epr` confidence sign is -1 |
| `spilled` | saved spilled energy | frozen `epr_spilled` confidence sign is -1 |
| `neg_logsumexp` | negative saved log-sum-exp | frozen `epr_energy` confidence sign is +1 |
| `neg_top1` | negative top-1 log probability | frozen confidence sign is +1 |
| `neg_margin` | negative top1-minus-top2 margin | frozen confidence sign is +1 |
| `topk_entropy` | entropy of renormalized saved top-k support | confidence sign is -1 |
| `topk_varentropy` | surprisal variance on that support | confidence sign is -1 |
| `topk_renyi2` | order-2 Renyi entropy on that support | confidence sign is -1 |
| `topk_tail_mass` | mass outside the leading five saved entries | confidence sign is -1 |

No sign is estimated from target labels. The top-k quantities are explicitly
restricted-support telemetry and are not called exact full-vocabulary values.

For each channel, the calibration reference is the median and robust standard
deviation `(q75-q25)/1.349`, with ordinary standard deviation and then one as
deterministic fallbacks. Each calibration trace contributes the same 32
quantile-spaced token positions. Standardized values are clipped to `[-8, 8]`
to limit a single numerical tail without reversing order. A missing channel is
replaced by its fitted reference, hence zero after standardization, and its
availability is reported. A feature is removed only when it is non-finite or
constant on the calibration matrix; every fit must retain at least three views.

## Token-native operators and requirement audit

For standardized primitive risk `z_t`, the causal state contains:

- `level_t = z_t`;
- `ewma_t = (1-a) ewma_(t-1) + a z_t`, with `a=2/17` (a frozen 16-token
  effective span);
- `onset_t = max(0, z_t - ewma_(t-1))`;
- `positive_mean_t = mean_{i<=t} max(0,z_i)`;
- `persistence_t = mean_{i<=t} 1[z_i>0]`;
- `running_max_t = max_{i<=t} z_i`.

These are true per-token recurrences. They do not use the completed-trace mean,
the final trace length, a future window, or a completed feature curve. Prefix
states must be bit-identical after arbitrary suffix replacement and after
tokenwise versus chunked replay.

The algorithmic requirements are not overstated:

- primitive orientation, level, EWMA, positive mean, persistence, running max,
  and the final readouts below are non-decreasing in their corresponding input
  evidence when all other coordinates are fixed;
- onset is a nonnegative event magnitude but is not coordinate-wise monotone in
  the previous history, because a larger previous EWMA can reduce the current
  surprise;
- positive mean and persistence need not be monotone **over time**, although
  they are monotone in positive evidence at fixed `t`;
- IU-PCR itself requires at least three non-degenerate views and an approximately
  additive off-diagonal covariance model, not temporal monotonicity. We audit
  pairwise `|rho|`, additive residual, weight sign, component concentration,
  and score agreement with the risk-coordinate consensus. Event coordinates
  whose fitted effective sign opposes the frozen risk consensus are reported;
  they are never target-flipped.

Sequential declarations use the running maximum of the Online risk score, so a
warning cannot be withdrawn. This policy-level monotonicity is separate from
the behavior of any individual feature.

## Frozen head-specific candidate roster

All new arms use ordinary IU-PCR with two components, L2 solve, no feature
exclusion, no difficulty fallback, `scale_ratio=0.25`, and label-free alignment
to the mean risk-coordinate consensus. No candidate may be added, deleted,
renamed, sign-flipped, or tuned after its score is read.

### Global head

The completed trace is reduced separately for each primitive channel.

| id | coordinates | hypothesis |
|---|---|---|
| `g_mean9` | mean level | distributed uncertainty predicts final error |
| `g_mean_q90_18` | mean and 90th percentile level | a robust tail adds burst information |
| `g_mean_q90_max_27` | mean, 90th percentile, and maximum level | an extreme adds value beyond the robust tail |
| `g_registered_mixed` | frozen historical mixed-v2 ordinary IU artifact | registered full-trace reference |

### Local head

Every candidate returns a token-risk curve before step aggregation.

| id | coordinates per token | hypothesis |
|---|---|---|
| `l_level9` | primitive levels | the erroneous step has an abnormal local state |
| `l_onset9` | nonnegative onsets | the first error is best marked by a change event |
| `l_level_onset18` | levels and onsets | state and change are complementary |
| `l_registered_core5` | frozen historical core-five token head | registered localization reference |

### Online head

Every candidate returns a causal prefix-risk trajectory at every token.

| id | coordinates per token | hypothesis |
|---|---|---|
| `o_level_ewma18` | level and EWMA | instantaneous and smoothed uncertainty suffice |
| `o_level_ewma_onset27` | level, EWMA, and onset | a change event adds early warning |
| `o_ewma_area_persist27` | EWMA, positive mean, and persistence | sustained abnormality is more stable than a spike |
| `o_iu28_registered` | frozen historical IU28-no-length score where an exact saved score exists | registered Online reference |

The registered arms are regression references. If an exact per-question
registered score is unavailable for one of the newly discovered scorer cells,
the report says so rather than silently reconstructing or imputing it.

## Head selection

Each task is selected independently on the two roster-selection cells:

- Global primary: equal-cell completed-answer AUROC; AUPRC is co-reported.
- Local primary: equal-cell ProcessBench F1; exact erroneous-step accuracy,
  within-one-step SLA, and clean abstention accuracy are co-reported.
- Online primary: equal-cell mean AUROC at 64 and 128 absolute tokens among
  unfinished traces; AUPRC and eligible counts are co-reported.

For the Local head-only screen, the sequence detector is that curve's maximum,
the locator is the step containing its maximum, and a calibration-label F1
threshold decides error versus clean. The persistent-onset readout and
Global/Local detector blends enter only in the subsequent architecture cross.

For a task, identify the highest primary point estimate. Select the lowest-cost
candidate no more than `0.005` below it when the paired 95% grouped interval
against the numerical best includes zero. Otherwise select the numerical best.
This one-standard-error-like rule is frozen before outcomes and resolves ties
toward fewer features, fewer persistent states, then ordinary registered code.
The selected head is a development choice, not a confirmation result.

## Frozen one/two/three-head architecture roster

After the three head identities are selected, compare:

| id | Global output | Local curve | Online output |
|---|---|---|---|
| `a_one_shared` | final selected Online score | selected Online curve | selected Online score |
| `a_two_global_local` | selected Global score | selected Local curve | causal prefix Global score blended with running-max Local evidence |
| `a_three_independent` | selected Global score | selected Local curve | selected Online score |

For `a_two_global_local`, the Global feature reducer is replayed on each observed
prefix; it may not slice a completed feature matrix. A streaming quantile data
structure may be used, but the result must equal direct prefix recomputation.

The Local sequence detector uses standardized Global risk and the standardized
maximum Local risk available to that architecture. The preregistered Global
weight is in `{0, 0.25, 0.50, 0.75, 1}`. The same weight is used by the
two-head Online blend at every prefix. Calibration statistics standardize the
two inputs, and calibration labels choose the detector threshold. The weight is
selected on the two roster-selection evaluation cells, then frozen. This tests
the historical 0.75/0.25 rule rather than assuming it.

Two locator readouts are crossed with each architecture:

1. `peak`: step containing the maximum token risk;
2. `persistent_onset`: first token whose calibration-standardized risk exceeds
   the label-free calibration 90th percentile for three consecutive tokens,
   falling back to `peak` if no event occurs.

Architecture selection is lexicographic because the outputs are co-primary and
must not be averaged into an arbitrary scalar:

1. discard any architecture more than `0.010` Global AUROC, `0.010`
   ProcessBench F1, or `0.015` Online 64/128 AUROC below the best development
   architecture for that panel;
2. among survivors, prefer an architecture with a paired interval wholly above
   zero on any panel versus the simplest survivor and no negative margin breach;
3. otherwise retain the fewest heads, then the fewest features/state scalars.

## Low-priority fusion-mechanism decision

Only after feature and architecture identities are frozen, run same-matrix
controls on their exact calibration matrices:

- ordinary IU-PCR (`lambda=0`);
- uniform average;
- DUFS-gated Laplacian IU with `k=7`, frozen repository seeds, and
  `lambda=0.1`; and
- temporal-chain Laplacian for the Local token matrix only, also
  `lambda=0.1`.

No graph hyperparameter is searched. The ordinary arm in every graph runner
must be bit-identical at `lambda=0`. A graph component is retained only if its
paired 95% interval versus ordinary is wholly positive on its target task,
there is no co-primary architecture margin breach, and the gain is not
dominated by measured time/memory. Otherwise the decision is ordinary IU-PCR.
This stage decides whether the Laplacian/DUFS machinery is needed; it is not
allowed to rescue a failed feature or architecture arm.

## Evaluation and stopping policy

Global reports AUROC/AUPRC and fixed-FPR operating points. Local reports
ProcessBench F1, exact erroneous-step accuracy, within-one-step SLA, clean
abstention, and first-error offset. Online reports AUROC/AUPRC at
16/32/64/128/256/512 tokens among traces unfinished at that budget, convergence
to its own final score, and results by trace-length band.

Online declaration thresholds are calibrated on the **maximum over the entire
preregistered monitoring horizon** of each calibration trace, not at a single
time point. This controls trace-level ever-warning on correct answers at target
rates 5% and 10%. Report warning coverage on wrong answers, false warnings on
correct answers, first-warning budget, selective error, and potential remaining
tokens. Potential remaining tokens are not called realized savings: realized
accuracy/compute savings require new forced-closure inference, which is outside
this CPU-only protocol.

Length leakage is tested by reporting score/length Spearman correlation,
within-length-band AUROC, and residualized-score AUROC where a calibration-only
isotonic length model is subtracted. Elapsed length is never a feature in the
primary arms.

## Efficiency and sensitivity

Measure primitive extraction, head fitting, full token scoring, prefix scoring,
and complete three-output evaluation separately. Record wall time, Python peak
allocation, resident-memory change where available, feature count, fit-row
count, and persistent scalar state per trace. Upstream generation/logit capture
is already paid for and is reported as an access requirement, not hidden in the
head benchmark.

For every selected head run primitive-family drop-one tests, missing-channel
tests, and pairwise/redundancy diagnostics. A coordinate group is retained only
if its grouped effect is larger than uncertainty or necessary for the chosen
simpler mechanism.

## Required tests before outcomes

1. raw-channel formulas against the existing global log-probability helpers;
2. arbitrary suffix replacement invariance at multiple token positions;
3. tokenwise versus chunked replay identity;
4. direct prefix recomputation identity for Global prefix scores;
5. feature-order invariance;
6. fit identity after labels are permuted, replaced, or removed;
7. repeated-run and hash identity;
8. deterministic missing-channel and constant-feature handling;
9. frozen risk-orientation and operator monotonicity counterexamples/checks;
10. shared-ID split equality across scorer models;
11. exact `lambda=0` equality on every graph path;
12. reproduction or explicit unavailability of each registered anchor.

Candidate outcomes may be read only after tests 1--10 pass. Graph outcomes may
be read only after test 11 passes.

## Claim and next gate

The strongest possible conclusion here is selection of a promising
retrospective architecture for three tasks under existing telemetry. It remains
an **unsupervised scorer with calibrated decision policies**. It cannot establish
fresh generalization, an exact paper reproduction, or realized compute savings.

New inference or a GPU/cluster job is justified only if the frozen architecture
shows a positive grouped improvement on at least one co-primary panel, stays
within all other margins, survives a dataset-family and scorer-model transfer
audit, and yields a credible causal warning policy. Any such run requires a new
explicit approval with exact model, dataset, decoding, storage, and GPU-hour
estimates.
