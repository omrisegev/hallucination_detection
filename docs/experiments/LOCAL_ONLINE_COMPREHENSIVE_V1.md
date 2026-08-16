# Comprehensive token-native Local and Online research — v1

Date frozen: 2026-08-16

Status: frozen before any new candidate score is computed.

## Objective

Run a broad, staged optimization of two separate co-primary outputs over the
available reasoning telemetry:

1. **Local** — decide whether a trace contains a reasoning error and identify
   the first erroneous ProcessBench step;
2. **Online** — predict final-answer wrongness from a genuinely unfinished
   prefix and support a non-withdrawable early warning.

This protocol explicitly supersedes treating the Step-272 nine-channel screen
as comprehensive. Step 272 is an incumbent and regression anchor. The present
cycle widens the local feature pool, temporal operators, locators, online state
families, and joint architecture. Global mixed-v2 remains a fixed input where a
joint detector needs it; Global feature selection is out of scope.

All source populations have been opened in previous project work. Therefore
this is retrospective method development, not fresh confirmation or a new SOTA
claim. No new inference, GPU/cluster work, large download, Google Drive
mutation, raw-data mutation, staging, commit, or push is authorized here.

## Data boundary

The main evidence is the twelve complete ProcessBench telemetry cells:

- scorer telemetry: Qwen3-4B, Qwen3-8B, and Llama-3.1-8B;
- families: GSM8K, MATH, OlympiadBench, and OmniMath;
- 3,400 fixed ProcessBench traces per scorer, with the same source identities
  repeated across scorer models;
- available token streams: entropy, sampled-token spilled energy,
  log-sum-exp, top-k log-probabilities, generated-token IDs, token-to-step
  spans, first-error labels, and final-answer correctness.

The full-size ProcessBench competitor artifacts are read-only inputs:

- Qwen2.5-Math-PRM-7B predictions;
- Qwen2.5-72B critic-protocol predictions;
- the Qwen3-8B no-training judge control, which must never be called uPRM.

Scorer copies are repeated measurements, not independent questions. Every
interval and win/loss summary resamples a source question once and carries all
available scorer copies together. Dataset families receive equal weight.

### Frozen stage roles

- **S0 mechanics/calibration:** label-blind score fitting uses the deterministic
  calibration subset within each cell.
- **S1 Local selection:** Qwen3-4B GSM8K and MATH development rows only.
- **S2 Online selection:** the same two development cells, with the Local S1
  identity frozen before Online outcomes are interpreted.
- **S3 architecture selection:** Qwen3-4B OlympiadBench and OmniMath only.
- **S4 retrospective audit:** all Qwen3-8B and Llama-3.1-8B cells. These reuse
  the same source questions and are not called independent-data confirmation.

Within every family, a SHA-256 hash of the canonical source identity assigns
questions to `calibration` (40%), `development` (20%), `architecture` (20%),
or `audit` (20%). The assignment is shared across scorer models. If a cache
identity cannot be reconciled exactly, the run closes rather than falling back
to row position.

Score fitting, feature orientation, missing-channel handling, and fusion are
label-blind. Labels may be used only for the declared stage metric, candidate
selection, and detector/declaration threshold calibration. Threshold rows and
evaluation rows are group-disjoint. Ranking metrics are never computed on
concatenated probabilities from separately fitted supervised folds.

## Competitor reporting contract

Every experimental stage emits one scoreboard before a candidate is promoted.
The scoreboard contains the incumbent, all candidates evaluated at that stage,
and the relevant competitors on exactly the same evaluation rows. It reports
absolute performance, paired delta, grouped 95% interval when row-level scores
exist, family wins/losses, access class, and fidelity.

### Tier A — direct, same telemetry or same fixed trace

Local:

- maximum token entropy;
- Mind the Gap evidence-drop reproduction;
- frozen GL-LIU v1;
- Step-272 `l_level9` two-head incumbent.

Online:

- mean entropy;
- maximum entropy;
- DeepConf lowest-group-confidence with frozen windows 32 and 64;
- registered IU28 without elapsed/final length where exactly reconstructible;
- Step-272 two-head 0.50 Global-prefix/Local-running-max incumbent.

Only Tier-A rows support a same-access improvement claim.

### Tier B — same ProcessBench rows, different compute/access

- Qwen2.5-72B critic-protocol reproduction;
- Qwen2.5-Math-PRM-7B supervised ceiling;
- Qwen3-8B no-training judge control.

These are shown on every Local stage where the row join is complete. They do
not support a same-cost delta. The judge control is not uPRM.

### Tier C — cross-protocol context only

- uPRM, arXiv:2605.10158, which trains a model and has no public code linked by
  the paper as of the freeze date;
- Streaming Hallucination Detection in Long Chain-of-Thought Reasoning,
  arXiv:2601.02170, a supervised hidden-state probe on a different annotated
  dataset;
- published ProcessBench QwQ/o1/GPT-4o critic results when quoted from the
  paper rather than reproduced on this project's rows.

Tier-C numbers are never subtracted from a local metric and never labelled a
head-to-head result.

Primary sources:

- https://arxiv.org/abs/2605.10158
- https://arxiv.org/abs/2601.02170
- https://arxiv.org/abs/2412.06559
- https://arxiv.org/abs/2508.15260

## Feature source contract

### Primitive raw-nine representation

The Step-272 risk-oriented token channels are retained:

1. entropy;
2. sampled-token spilled energy;
3. negative log-sum-exp;
4. negative top-1 log-probability;
5. negative top1-top2 margin;
6. top-k entropy;
7. top-k varentropy;
8. top-k Renyi-2 entropy;
9. top-k tail mass outside the leading five saved entries.

`raw7_opened_drop` is preregistered as raw nine without spilled energy and
top-k entropy. The identity is motivated by Step-272 opened drop-one results;
it is a new candidate, not confirmation evidence.

### Broad token-28 representation

Use the exact `BROAD_TOKEN_VIEWS` contract in
`spectral_utils/token_feature_views.py`: entropy level; rolling entropy
spectral, STFT, tail-ratio, variance, permutation-entropy, Hurst, and CUSUM
curves; sampled-token energy level/variance/CUSUM/minimum; log-partition
level/variance/CUSUM/minimum; and six top-k distribution curves.

The prior broad-28 DUFS arm is a failure anchor, not a reason to omit this
representation. The present experiment separately tests ordinary IU,
task-specific operators, and provenance-balanced family compression.

### Provenance-balanced family-six representation

The 28 oriented and calibration-standardized views are averaged inside six
frozen provenance families before cross-family IU:

1. `entropy_level`: entropy level;
2. `entropy_dynamics`: entropy variance, CUSUM, and rolling tail ratio;
3. `structural`: rolling spectral/STFT/permutation-entropy/Hurst views;
4. `sampled_energy`: spilled level, variance, CUSUM, and rolling minimum;
5. `partition_energy`: log-partition level, variance, CUSUM, and rolling
   minimum;
6. `topk_distribution`: top-1, margin, entropy, varentropy, Renyi-2, and tail
   mass.

This makes one family contribute one view regardless of how many correlated
coordinates it contains. It is deterministic and target-blind.

## Causal operator contract

For every standardized risk coordinate `z_t`, compute:

- `level_t = z_t`;
- `fast_t`, EWMA with span 8;
- `slow_t`, EWMA with span 32;
- `innovation_t = max(0, z_t - slow_(t-1))`;
- `shortlong_t = fast_t - slow_t`;
- `positive_mean_t = mean_{i<=t} max(0,z_i)`;
- `persistence_t = mean_{i<=t} 1[z_i>0]`;
- `recovery_t = z_t - max_{i<=t} z_i`.

All states are O(1) update per coordinate. Prefix outputs must be identical
under arbitrary suffix replacement and tokenwise versus chunk replay.

Level, EWMAs, positive mean, persistence, and running maximum are monotone in
the corresponding evidence at fixed time. Innovation, short-long contrast,
and recovery are event coordinates and are not claimed to be monotone in
history. IU-PCR does not require temporal monotonicity; it requires at least
three non-degenerate views and an approximately additive covariance model.

## S1 — Local candidate roster

Every candidate returns a token-risk curve. Ordinary two-component IU-PCR is
the default scorer during feature screening. The historical core-five and
Step-272 raw-nine heads are exact regression anchors.

For each representation `R` in `{raw9, broad28, family6}`, evaluate:

- `R__level`;
- `R__innovation`;
- `R__shortlong`;
- `R__level_innovation`;
- `R__level_shortlong`;
- `R__level_innovation_shortlong`.

Also evaluate:

- `raw7_opened_drop__level`;
- historical `core5`;
- the prior broad-28 DUFS result as a non-refitted failure anchor.

Each curve crosses three frozen locators:

1. `peak`: the step containing the maximum token score;
2. `persistent_q90_3`: first token exceeding the label-blind calibration 90th
   percentile for three consecutive tokens, otherwise peak;
3. `step_top5mean`: the step with the largest mean of its five largest token
   risks.

The S1 detector is Local-only: calibration labels select one scalar threshold
on maximum Local risk. S3 later tests Global/Local detector mixtures.

S1 primary is ProcessBench F1. Co-reported metrics are error-trace exact
first-step accuracy, within-one-step accuracy, clean abstention accuracy,
detector AUROC/AUPRC, mean absolute step error, and late-location rate.

Select the simplest candidate within 0.005 F1 of the numerical best when its
paired interval versus the best includes zero. Otherwise select the numerical
best. A candidate may not be promoted if it loses more than 0.010 F1 to the
Step-272 incumbent on either development family.

## S2 — Online candidate roster

For each representation `R` in `{raw9, broad28, family6}`, evaluate:

- `R__level_slow`;
- `R__fast_slow`;
- `R__slow_area_persistence`;
- `R__shortlong_innovation_recovery`;
- `R__level_fast_slow_area_persistence`.

Also evaluate registered IU28 and the Step-272 sustained raw-nine head. Every
head is fitted without labels using equal trace contribution and scored by
direct causal replay at absolute budgets 16, 32, 64, 128, 256, and 512.

S2 primary is the equal-family mean of AUROC at 64 and 128 tokens among traces
strictly longer than the budget. Co-reported metrics include per-budget AUROC
and AUPRC; 32/64 and 64/128 early slopes; prefix-to-final Spearman; length
correlation; and trace-level declaration behavior at 5% and 10% calibration
false-warning targets.

Select the simplest candidate within 0.005 AUROC of the numerical best when
its grouped paired interval versus the best includes zero. Otherwise select
the numerical best. A candidate may not be promoted if it is more than 0.015
below the best of DeepConf-w32, DeepConf-w64, IU28, or the Step-272 incumbent
on either development family.

## S3 — Joint Local/Online architecture and fusion

With S1 and S2 identities frozen, compare:

1. Local-selected head used for both Local and Online;
2. independent Local and Online heads;
3. fixed mixed-v2 Global-prefix plus Local-derived Online;
4. fixed mixed-v2 Global-prefix plus independent Online;
5. a three-signal Global-prefix/Local-derived/independent-Online blend.

Two-signal weights are `{0, 0.25, 0.50, 0.75, 1}`. Three-signal weights are
the simplex points with coordinates in `{0, 0.25, 0.50, 0.75, 1}` summing to
one. Inputs are standardized only from calibration rows. Labels choose only
the detector/declaration threshold and the declared S3 architecture.

After the exact feature matrix is frozen, compare:

- equal standardized average;
- ordinary IU-PCR;
- U-PCR compatibility path;
- uniform Laplacian IU;
- DUFS-gated Laplacian IU;
- temporal Laplacian IU for Local only;
- provenance-hierarchical IU where applicable.

Every graph path must reproduce ordinary IU exactly at `lambda=0`. A graph or
alternative fusion is retained only if its grouped 95% interval versus
ordinary is wholly positive on its target task, no other co-primary metric
breaches its margin, and the measured cost is not more than 5x for a gain below
0.010.

Architecture selection is lexicographic: remain within 0.010 Local F1 and
0.015 Online AUROC of the panel best; then prefer a wholly positive paired
improvement against the simplest survivor; otherwise choose fewer heads,
fewer coordinates, and less state.

## S4 — transfer, failure tests, and efficiency

The frozen finalist is audited on Qwen3-8B and Llama-3.1-8B scorer telemetry.
Required outputs:

- per-family and equal-family Local/Online scoreboards against Tier-A
  competitors;
- Local same-row context beside the complete critic and PRM outputs;
- question-grouped paired intervals and family wins/losses;
- error-position quartiles and short/medium/long trace strata;
- correct-final-answer-but-erroneous-process and wrong-final-answer-but-clean-
  process strata kept separate;
- missing primitive/family ablations;
- prefix suffix/chunk identity;
- feature-order and repeated-run identity;
- label removal/permutation fit identity;
- length-residualized Online AUROC;
- false-warning, detection coverage, first-warning budget, and potential
  remaining tokens without claiming realized savings;
- fit time, scoring time, peak memory where measurable, feature count, and
  persistent state count.

## Stage reporting and claim language

The canonical result directory is
`results/local_online_comprehensive_v1/`. It must contain:

- `STAGE_0_BASELINES.md` and `.csv`;
- `STAGE_1_LOCAL.md` and machine-readable candidate/competitor tables;
- `STAGE_2_ONLINE.md` and machine-readable candidate/competitor tables;
- `STAGE_3_ARCHITECTURE.md` and machine-readable candidate/competitor tables;
- `STAGE_4_TRANSFER.md` and machine-readable candidate/competitor tables;
- frozen selection JSON after each stage;
- per-question score files sufficient for paired intervals;
- `REPORT.md`, self-contained `REPORT.html`, `DECISION.json`, `AUDIT.json`,
  and `RUN_MANIFEST.json`.

Each stage report begins with one of four verdicts:

- `IMPROVES_DIRECT_COMPETITOR`;
- `PARITY_WITH_DIRECT_COMPETITOR`;
- `REGRESSES_DIRECT_COMPETITOR`;
- `MECHANICS_ONLY_NO_PERFORMANCE_CLAIM`.

“Leading”, “best”, and “SOTA” may be used only with an explicit access tier,
task, dataset, metric, and uncertainty statement. Tier-B or Tier-C ceilings
cannot be used to claim a same-access loss or win. Failure to beat the
supervised/critic ceiling remains visible rather than being hidden.

## Stop and next-authorization rules

- If no S1 candidate improves or ties the best Tier-A Local competitor within
  the frozen margin, keep the incumbent and record the failed mechanisms.
- If no S2 candidate improves or ties the best direct Online competitor within
  the frozen margin, keep the direct competitor as the empirical bar; do not
  rescue the result with a later-budget or final-trace score.
- If S4 does not show a stable direct-competitor improvement, do not request a
  fresh GPU run merely because one development cell is positive.
- New model inference, exact reproduction of the supervised streaming probe,
  or training a uPRM-like model requires a new protocol, cost estimate, and
  explicit user authorization.
