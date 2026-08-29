# Reasoning Localization 0.3662 Anchor v1

Status: PHASE 0 IN PROGRESS. P0-S0 checksum-equivalent historical replay and P0-S1 one-factor reducer bridge are complete; P0-S2 detector bridge is not started. Both completed states used the frozen historical population and performed no new model inference or method promotion.

Program ID: REASONING_LOCALIZATION_03662_ANCHOR_V1

Design date: 2026-08-29

Isolated branch: codex/reasoning-localization-03662-v1

Base commit: 59359001e5398a6df1783239326cdb1b2215bb07

## 1. Program goal and ordered tasks

The primary goal is a compact, evidence-backed signal family that is strong at reasoning-error localization under two distinct evaluator contracts:

1. ProcessBench first-error localization, directly compared with the repository adaptation of Mind the Gap on exactly the same rows, access tier, step spans, calibration policy, and evaluator.
2. PRMBench every-step error localization, using the ProcessBench-frozen scorer without PRMBench retuning.
3. Lower priority: causal early detection under prefix-only access.

ProcessBench and PRMBench are mandatory separate scorecards. They must not be averaged. A ProcessBench gain with a PRMBench regression is a ProcessBench specialist, not a task-general localizer. Likewise, a PRMBench-only gain is a PRMBench specialist.

The historical Local F1 of approximately 0.3662 is the Phase 0 anchor. It is neither promoted as a current result nor dismissed as a curiosity. The first scientific question is why that regime produced an absolute score near 0.3662, while its raw-entropy reference produced approximately 0.3614 and the method-specific delta was only approximately +0.0048 with an interval crossing zero.

## 2. Evidence and state boundary

This design distinguishes four evidence states:

- Committed repository evidence at or before the base commit.
- Historical frozen artifacts whose exact protocol snapshot was later recovered.
- Working-tree-only evidence observed in the pre-existing token-local-fusion worktree. These artifacts are cited by content hash but are not imported or treated as committed here.
- New hypotheses in this document. These are not results.

The source worktree at local_cache/worktrees/token_local_fusion_optimization_v1 contained pre-existing modified and untracked files during this audit. It was read only. Steps 294-297 observed there are therefore reserved; this program is registered as Step 298 to avoid a future HISTORY collision.

## 3. Phase 0: exact reconstruction of the 0.3662 regime

### 3.1 What produced 0.3662

The historical Stage 4 finalist was a hybrid system, not a single local entropy curve:

- Local representation: family6, formed by averaging the broad28 token curves into six provenance families.
- Local transform: level.
- Step reducer: step_top5mean, implemented as the mean of the five largest token scores in a step.
- Answer-level detector: RegisteredGlobal, a mixed-v2 complete-answer feature system with ordinary IU, rather than the maximum of the family6 local curve.
- Calibration: a fixed 40 percent calibration partition; reporting used the fixed 20 percent audit partition.
- Population: the Stage 4 audit evaluated eight Qwen3-8B and Llama-3.1-8B cells, not the current full twelve-cell panel.

Exact evidence:

- docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md:28-71 defines the 12-cell source population, shared scorer copies, and 40/20/20/20 deterministic roles.
- docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md:131-176 defines raw9, broad28, and family6.
- docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md:178-239 defines the causal operators, representations, and Local reducers.
- scripts/run_local_online_comprehensive_stage1.py:149-163 implements top-five step pooling.
- scripts/run_global_local_online_architecture_v2.py:360-405 implements the RegisteredGlobal complete-answer representation and ordinary IU.
- scripts/run_local_online_comprehensive_stage4.py:178-231 combines family6 level/top5 localization with the separate RegisteredGlobal answer detector.
- scripts/run_local_online_comprehensive_stage4.py:698-740 constructs the raw-entropy reference on the same Stage 4 rows.

### 3.2 Exact historical scores and statistical interpretation

The frozen Stage 4 aggregate records:

- Finalist Local F1: 0.3662328341717007.
- max_entropy__step_top5mean Local F1: 0.3614213583669282.
- Raw delta: +0.0048114758047725.
- Grouped bootstrap interval for the delta: [-0.0263871, +0.03750365].
- Source-family directions: three wins and one loss.
- Online result: finalist 0.5882 versus entropy reference 0.6104.

Exact evidence:

- results/local_online_comprehensive_v1/STAGE_4_AGGREGATE.csv.
- results/local_online_comprehensive_v1/STAGE_4_INTERVALS.csv.
- results/local_online_comprehensive_v1/REPORT.md:5-16 and 34-40.
- HISTORY.md:13158-13206 records the Stage 4 selection, result, and negative promotion verdict.

Therefore 0.3662 is the raw best absolute score in that frozen regime, while the only supported method comparison is that family6 plus the hybrid detector was statistically indistinguishable from the much simpler entropy reference. Most of the absolute score may come from the shared evaluator regime rather than from the finalist machinery.

### 3.3 Reproducibility caveat

The original protocol hash later failed to match the live document. The exact frozen snapshot was recovered from commit 2c2f5a9, and Stage 1 was reproduced byte-for-byte except for floating-point score and threshold differences that did not change predictions. Phase 0 must run from the frozen snapshot or an explicitly checksum-equivalent port, never from a plausible current reconstruction.

Evidence: HISTORY.md:13654-13735.

### 3.4 Executed state P0-S0: checksum-equivalent historical replay

P0-S0 was executed on 2026-08-29 from the eight original Stage-4 checkpoint
payloads. Before opening the replay, the runner, recovered protocol, every
checkpoint, and every frozen comparison artifact were registered by exact
SHA256 in
`results/reasoning_localization_03662_v1/phase_0/P0_S0_EXECUTION_REGISTRY.json`.
The replay imported the historical aggregation and grouped-bootstrap functions
and wrote only into the isolated program result tree. It performed no model
inference and did not mutate the historical result directory.

Acceptance results:

- `STAGE_4_LOCAL_PER_QUESTION.csv` reproduced byte-for-byte, SHA256
  `161c598f5e4d6fbddc52f66da55b8cc485a0c20111346b7fb5dd1f87e5ea0c77`.
- Cell metrics, aggregate metrics, and intervals reproduced semantically
  exactly.
- The finalist remained `0.3662328341717007`; entropy/top5 remained
  `0.3614213583669282`; their frozen delta remained
  `+0.004811475804772508` with interval
  `[-0.02638710838275541,+0.037503652325203835]` and 3/0/1 family W/T/L.
- The exact population is eight scorer cells and 1,270 scorer-row
  observations, corresponding to 635 source-question bootstrap groups. The
  earlier reporting-context count of 3,400 rows/four groups was corrected.
- The source-question population hash is
  `d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05`.

Canonical outputs are under
`results/reasoning_localization_03662_v1/phase_0/p0_s0_historical_replay/`.
`P0_S0_VERIFICATION.json` records the exact checks and `RUN_MANIFEST.json`
binds the execution registry, runner, protocol, runtime, sources, and outputs.
The living report registers R2 as `COMPLETE / NO_PROMOTION / RETROSPECTIVE`.
This closes only S0 reproducibility. It does not identify which factor explains
the high score and does not authorize S1 implicitly.

### 3.5 Executed state P0-S1: one-factor reducer bridge

P0-S1 was frozen before execution in
`results/reasoning_localization_03662_v1/phase_0/P0_S1_EXECUTION_REGISTRY.json`.
It retained the exact S0 population, 40/20 roles, family6 level
representation, fitted local head, RegisteredGlobal detector, calibration
policy, and access tier. The only opened factor was the step reducer:
`step_top5mean` became token argmax mapped to step. Because historical
checkpoints do not store token curves, the CPU runner refit the historical
heads from the exact source artifacts and hard-failed unless every S0 unit,
target, locator, prediction, score, threshold, metric, and fitted diagnostic
reconstructed within `1e-12`.

All reconstruction and source-stability gates passed. On the unchanged eight
cells, 1,270 scorer rows, and 635 source-question groups, step max reached
macro F1 `0.33007771561392063`, versus the S0/top-five value
`0.3662328341717007`. The paired 20,000-draw source-question bootstrap delta
was `-0.03615511855778009`, 95% interval
`[-0.06666876234842496,-0.006984802202460412]`, with 0/0/4 family W/T/L and
worst scorer-cell delta `-0.09208456432490503`.

The mechanism components do not support a generic detection failure. Clean
abstention increased by `+0.05180805805805805`, while exact-error localization
fell by `-0.04886162153449389` and within-one localization fell by
`-0.05475208646751201`; all three paired intervals exclude zero in their
respective directions. The deterministic flip audit records 59 exact-to-
nonexact changes versus 26 error-to-exact changes, along with 978 unchanged
predictions. Thus the historical top-five reducer explains a statistically
supported 3.62-point portion of the 0.3662 regime on this retrospective
population. This does not establish its superiority under the modern split or
detector.

Canonical outputs are under
`results/reasoning_localization_03662_v1/phase_0/p0_s1_reducer_bridge/`.
`P0_S1_VERIFICATION.json` and `RUN_MANIFEST.json` bind the frozen inputs,
single-factor contract, runtime, outputs, paired inference, and flip audit.
P0-S1 is `COMPLETE / NO_PROMOTION / RETROSPECTIVE`; Phase 0 remains open and no
phase snapshot is created yet.

### 3.6 Preregistered bridge, one factor at a time

The 0.3662 regime is not directly comparable with the modern approximately 0.307 ProcessBench score. Phase 0 freezes an intersection of shared rows and changes one factor at a time in this order:

1. Historical replay: historical eight-cell audit, fixed historical roles, RegisteredGlobal detector, family6 level, step_top5mean.
2. Reducer bridge: change only step_top5mean to step max.
3. Detector bridge: change only RegisteredGlobal to the modern registered answer detector, then to a purely local maximum detector.
4. Representation bridge: change only family6 to raw entropy, then IU29.
5. Split bridge: change only fixed calibration/audit roles to the current five-fold source-grouped threshold cross-fit.
6. Population bridge: expand the shared-row eight-cell result to the current eight-Qwen development panel and then the full twelve-cell transfer panel.

For every bridge edge after S0, report the paired delta and grouped interval. Do not use an omnibus rerun to claim which factor explains the drop. The intended Phase 0 deliverable is a waterfall of attributable score changes plus residual interaction, not a new leaderboard winner.

## 4. Evaluator contracts are not interchangeable

### 4.1 ProcessBench contract

The current frozen ProcessBench reconstruction uses:

- Three scorer models by four ProcessBench subsets, twelve cells total.
- Five deterministic, label-stratified, source-question-grouped folds.
- Threshold selection on the four-subset joint validation portion.
- A trace-level first-error decision with a clean-answer abstention sentinel.
- The official macro F1 defined as the harmonic mean of exact first-error accuracy on erroneous traces and correct abstention on clean traces.
- Paired, source-grouped bootstrap inference.

Evidence:

- configs/reconstruction_benchmark_v1/localization.json:30-72.
- spectral_utils/reconstruction_benchmark/localization_evaluation.py:211-420.
- scripts/run_global_local_online_architecture_v2.py:260-310.

### 4.2 PRMBench contract

The current frozen PRMBench reconstruction instead uses:

- One Qwen3-8B scorer.
- Error responses only; synthetic-correct examples are excluded.
- All nine error families, with multi_solutions forming a single-class panel.
- Every-step binary ranking, reported primarily by AUROC and AUPRC.
- No first-error threshold, no clean abstention decision, and no ProcessBench-style official F1.
- Paired source-response bootstrap.

The official-port adaptation keeps 151 one-based annotation memberships that are out of bounds as inert rather than dropping rows. The effective positive membership count is 13,144 rather than the originally expected 13,295.

Evidence:

- configs/reconstruction_benchmark_v1/localization.json:74-103.
- configs/reconstruction_benchmark_v1/localization_postfreeze_amendment_v1.json:2-24 and 1490-1569.
- spectral_utils/prmbench.py:1-60.
- spectral_utils/reconstruction_benchmark/localization_postfreeze.py:571-739.
- spectral_utils/reconstruction_benchmark/localization_evaluation.py:467-526.

### 4.3 Consequence for this program

The same frozen token scorer may be tested on both tasks, but scores, populations, thresholds, and decision semantics are not comparable. The program will report two panels and two verdicts. No combined mean, weighted mean, or normalized aggregate is an acceptance statistic.

## 5. Leakage and split register

The following risks are preregistered:

1. Historical outcome opening. The Stage 4 finalist and its entropy ablation were selected after prior development. They are retrospective evidence only.
2. Reused ProcessBench labels. Current ProcessBench rows have already informed many repository decisions. Five-fold threshold cross-fitting prevents within-fold threshold leakage, but it does not convert feature search into fresh confirmation.
3. Shared scorer copies. Multiple scorer-model copies of the same source question must remain in the same fold and bootstrap group.
4. PRMBench source strata. The frozen artifact contains identifiers from prm_train and prm_test. Overall PRMBench is a transfer panel, not a fresh held-out confirmation set. Results must also be shown separately by original source stratum.
5. PRMBench annotation amendment. Out-of-bounds memberships are inert; no row or step may be silently deleted.
6. Reconstruction fidelity. Seventeen of the twenty-eight token streams are exact identities and eleven are best-effort reconstructions in the token-local line. Exact and reconstructed coordinates must be labeled separately.
7. Noncausal CUSUM. The current token CUSUM view subtracts a full-trace mean. It is permitted only in the completed-trace diagnostic lane. Early detection must use a frozen donor-centered or running causal reference.
8. Mind the Gap paper metrics. The paper's published SLA-style evidence-drop numbers are not a direct ProcessBench first-error result and do not include the repository clean-abstention contract. Only a same-row repository adaptation is a direct comparison.
9. Model and feature selection. Labels may tune thresholds and compute metrics only. Orientation, scaling, unsupervised fitting, and feature construction must remain label-blind unless a separately named supervised ceiling is run.
10. Fresh confirmation. Because both target benchmarks are already open, any final universal-method claim requires a future untouched model, dataset, or preregistered release panel.

## 6. Evidence-ranked signal inventory

### 6.1 Proven useful or robust enough for a primary arm

#### Entropy level and dynamics

Include. The Stage 4 raw-entropy top5mean reference reached 0.3614, and removing entropy dynamics or primitive entropy from the historical finalist cost approximately 0.0127 and 0.0125 Local F1 respectively. This is the strongest anchor evidence.

Evidence: results/local_online_comprehensive_v1/REPORT.md:5-16 and 69-80.

#### family6 level plus step_top5mean

Include as an exact historical bridge, not automatically as the modern incumbent. It defines the 0.3662 localizer and keeps provenance families separate before averaging.

Evidence: docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md:131-176; scripts/run_local_online_comprehensive_stage4.py:178-231.

#### Sliding-window variance

Include one compact causal form. Early repository studies repeatedly ranked sw_var near the top, found it robust across models, and measured a positive marginal contribution in an exhaustive honest-LOCO analysis. In early detection, removing sw_var cost approximately 0.026 to 0.046 AUROC.

Freeze the primary form as trailing variance with window 16. Run the historical adaptive 10 percent window clipped to 3 through 32 as a one-factor sensitivity arm, not simultaneously in the same feature vector.

Evidence:

- spectral_utils/feature_utils.py:118-164.
- HISTORY.md:1959-2024 and 3300-3325.
- HISTORY.md:4924-4934.
- PROGRESS.md:1050-1054.

#### Compact DSP transform grammar

Include a reduced form. The useful DSP result was not the 1,036-feature bank; it was seven sources by four simple transforms: level, ewma16, positive_area, and persistence. That Unified28 representation improved the matched Llama Local result by approximately +0.0461 while regressing Global and Early, making it relevant but task-specific.

Evidence: PROGRESS.md:680-715.

### 6.2 Promising, but weak or task-specific evidence

#### CUSUM and change timing

Include as a targeted ablation, not a core coordinate by default. cusum_max and sw_var had top average rank in early completed-answer experiments, and shift timing appeared in strong subsets. But CUSUM was neutral or slightly harmful in later early-prefix ablations. Freeze two separate variants:

- Offline centered CUSUM for the completed-trace diagnostic lane only.
- Causal donor-centered absolute CUSUM for the primary transferable lane; the donor center is fitted label-blind on calibration donors and never uses future tokens.

Evidence: spectral_utils/feature_utils.py:256-274; HISTORY.md:3300-3325 and 4924-4934; PROGRESS.md:1050-1054 and 1102-1109.

#### Sampled-token surprisal and partition energy

Include as atomic source ablations and in the compact DSP arm. They are already part of the historical entropy-energy family and recent token-local source roster, but no current artifact establishes either as an independent localization winner. They survive because they are small, access-compatible uncertainty coordinates with a clear entropy complement, not because a fusion result has proven them.

Evidence: docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md:131-176; PROGRESS.md:680-715.

#### Mind the Gap evidence-drop curve

Include as the required direct reference and as a single atomic curve. The transferable construction is the EMA-smoothed local evidence drop or worst local drop, evaluated through the repository ProcessBench contract. Do not import the paper's SLA calibration, teacher-forced assumptions, or its reported table as if they were ProcessBench F1.

Evidence: papers/digests/mind-the-gap-catching-hallucinations-via-evidence-drop.md:20-28, 63-72, 95-116, and 120-160.

#### EDIS burst and rebound morphology

Keep one exploratory local burst/rebound curve. EDIS achieved a strong historical response-level score but was highly correlated with L-SML, had a formula bug repaired locally, and failed to replicate reliably. Its scalar score is excluded from the compact primary roster; only its interpretable onset morphology may be tested after the entropy arms.

Evidence: papers/digests/edis-paper.md:15-47; results/repgrid/edis_scores.csv.

#### Self-basis token innovation

Keep only as a diagnostic if the compact entropy/dynamics arm leaves complementary residual error. The recent self-basis arm had ProcessBench delta approximately +0.0026 with an interval crossing zero and PRMBench delta approximately -0.0043. This is not promotion evidence.

Working-tree evidence, not imported: local_cache/worktrees/token_local_fusion_optimization_v1/results/token_local_linear_innovation_v1/RESULTS_REPORT.md:14-45, SHA256 696fff7186fd433b14c2f7e812b41adf0f20faa9349e01ed4b229a570dfc6d7c.

#### B3 / confidence-information weighting

Keep as an unfinished conditional arm. Response-level B3 had a small positive result, but localization-specific CIW results did not improve both ProcessBench and PRMBench, and the token-local B3 score artifact was not yet frozen at the audit boundary.

Evidence: HISTORY.md:14504-14527; results/ciw_cross_scale_localization_v1/REPORT.md:1-21.

### 6.3 Negative or excluded from the primary roster

#### Rook, non-rook, and cross-basis innovation

Exclude. Rook improved target-free reconstruction MSE but harmed ProcessBench and PRMBench localization; non-rook gains were small and uncertain. Better reconstruction is not sufficient localization evidence.

Working-tree evidence, not imported: local_cache/worktrees/token_local_fusion_optimization_v1/results/token_local_linear_innovation_v1/RESULTS_REPORT.md:14-45 and 68-75, SHA256 696fff7186fd433b14c2f7e812b41adf0f20faa9349e01ed4b229a570dfc6d7c.

#### SU-PCR, STG-SU-PCR, DUFS-token selection, and global/local IU fusion

Do not use as Phase 1 foundations. On the recent token-local panel, IU29 scored 0.307052 ProcessBench and 0.593236 PRMBench; SU-PCR, STG-SU-PCR, and DUFS did not produce a supported improvement, while IU-global/IU29 reached only 0.308570 ProcessBench.

Working-tree evidence, not imported: local_cache/worktrees/token_local_fusion_optimization_v1/results/token_local_fusion_optimization_v1/phase1_evaluation/REPORT.md:5-21, SHA256 87adc473078703d45f68c6fd470fd8b91cfd4f3bca17e8f3b851a8de8c8460f4.

#### Transform hierarchy and clustering

Exclude from the primary fusion stage. Equal28 was only marginally above flat IU with an interval touching zero, while transform-hierarchy fusion was significantly worse.

Working-tree evidence, not imported: local_cache/worktrees/token_local_fusion_optimization_v1/results/unified28_transform_clustering_v1/evaluation/REPORT.md:5-25, SHA256 ee68d2f580c4e59bc8ae98efcd983c14ddb7c1e5da5e485fbeed12eb1ba8553c.

#### Broad feature selection and residual soups

Exclude. Across 21 conditions and 111 variants, DUFS, CAE, Laplacian/SPEC, and residual variants all failed to beat the deployed baseline; anti-redundancy methods were often harmful, and the selected target set was unstable. Earlier exhaustive subset search also showed substantial selection bias.

Evidence: PROGRESS.md:2343-2387; HISTORY.md:4924-4934.

#### Full 1,036-feature DSP bank

Exclude. Only the compact seven-by-four DSP representation is supported; the broad bank is a feature-soup risk and did not justify its search burden.

Evidence: PROGRESS.md:680-715.

## 7. Frozen compact candidate roster

Phase 1 begins with transparent references, all on the same current common population:

- R0: raw entropy, step max, modern registered answer detector.
- R1: raw entropy, step_top5mean, modern registered answer detector.
- R2: exact historical family6 level plus step_top5mean with RegisteredGlobal, used only for the Phase 0 bridge.
- R3: modern IU29 incumbent.
- R4: same-access Mind the Gap EMA-drop adaptation.

Phase 2 atomic candidates are deliberately small:

- C1 ENT-SW16: entropy level plus trailing sw_var16.
- C2 ENT-SWADAPT: entropy level plus the historical adaptive sw_var, a sensitivity arm against C1.
- C3 ENT-CCUSUM: entropy level, trailing sw_var16, and donor-centered causal absolute CUSUM.
- C4 ENT-SAMPLED: entropy level plus sampled-token surprisal.
- C5 ENT-ENERGY: entropy level plus partition energy.
- C6 DSP12: three sources — entropy, sampled-token surprisal, partition energy — by four transforms — level, ewma16, positive_area, persistence. Maximum dimension: 12.
- C7 EDIS-ONSET: the single entropy burst/rebound onset curve, exploratory.
- C8 SELF-INNOV: the smallest self-basis token innovation residual, diagnostic only.

No candidate may silently add length, correctness labels, outcome tokens, external verifiers, or a new source family. Trace length is always reported as a nuisance/control variable, never counted as localization evidence.

## 8. Staged execution plan

### Phase 0 — historical reconstruction and causal audit

Deliverables:

- Exact checksum-verified replay of Stage 4.
- The six-edge bridge in Section 3.4.
- Per-edge paired deltas, intervals, prediction flips, and waterfall attribution.
- A statement of how much of 0.3662 is explained by top5 pooling, detector choice, split, population, and residual interactions.

Gate: no Phase 0 result promotes a new method. Phase 0 only decides which historical components enter Phase 1.

### Phase 1 — compact baselines on a fair common population

Use the eight Qwen cells as the retrospective development panel and all twelve cells as a scorer-family transfer panel. Recompute R0-R4 on exact common rows, identical step spans, identical folds, identical threshold code, and identical bootstrap groups.

Deliverables:

- ProcessBench overall, per-cell, per-model, exact, within-one, and clean-abstention metrics.
- PRMBench overall, per-error-family, and original source-stratum AUROC/AUPRC for the score-only references.
- Runtime, peak memory, and missing-score audit.

Gate: a reference must be executable, checksum-frozen, and common-population complete before it can anchor Phase 2.

### Phase 2 — targeted entropy-centered ablations

Before C1-C8, execute the registered Phase-2 reducer branch below on the
frozen Phase-1 reference curve. Freeze its selected aggregation rule, then run
C1-C8 in the listed order with that same reducer. Each atomic arm changes only
one source, transform, or compact block relative to its parent. C7 remains an
explicit onset-morphology diagnostic rather than silently redefining the
common reducer. Stop a branch after a hard failure rather than expanding it.

#### Phase 2R — step scoring / reducer study

This branch separates three objects that prior work sometimes called one
"locator":

```text
frozen fused token-risk curve r[1:T]
    -> optional temporal transform T
    -> within-step aggregation A over known token span I_s
    -> step score R_s and prediction argmax_s R_s
```

Raw feature construction and family fusion are upstream and remain frozen.
The answer detector, detector score, decision threshold, split, population,
step spans, orientation, standardization, and bootstrap groups also remain
fixed. Therefore changing `A` cannot be rescued by rethresholding.

Historical `step_top5mean` was one of only three Stage-1 locator candidates,
beside `peak` and `persistent_q90_3`; it was selected using historical
development evidence. P0-S1 establishes only that `step_max` is materially
worse than `step_top5mean` in the exact retrospective Stage-4 regime. It does
not establish five as an optimal tail size or beat top-2, top-3, top-8,
top-10, a length-normalized top fraction, a quantile, mean, or median on the
current common population.

##### Stage A — identity curve, aggregation only

For a step span `I_s` of length `n_s`, the registered ladder is:

| variant | definition of `R_s` |
|---|---|
| `P2R_A_MAX_K1` | `max_{t in I_s} r_t` |
| `P2R_A_MEAN_ALL` | `n_s^-1 sum_{t in I_s} r_t` |
| `P2R_A_TOPK2` | mean of the largest `min(2,n_s)` values |
| `P2R_A_TOPK3` | mean of the largest `min(3,n_s)` values |
| `P2R_A_TOPK5_REFERENCE` | mean of the largest `min(5,n_s)` values; direct reference |
| `P2R_A_TOPK8` | mean of the largest `min(8,n_s)` values |
| `P2R_A_TOPK10` | mean of the largest `min(10,n_s)` values |
| `P2R_A_TOPQ25` | mean of the largest `max(1,ceil(0.25 n_s))` values |
| `P2R_A_TOPQ50` | mean of the largest `max(1,ceil(0.50 n_s))` values |
| `P2R_A_QUANTILE75` | empirical upper quantile `Q_0.75({r_t:t in I_s})` |
| `P2R_A_QUANTILE90` | empirical upper quantile `Q_0.90({r_t:t in I_s})` |
| `P2R_A_MEDIAN` | empirical median; robust control expected to dilute localized peaks |

Run one registered row at a time and rebuild the living report before opening
the next. Every row is compared directly with `P2R_A_TOPK5_REFERENCE` and its
best simpler parent on identical questions. Fixed-`k` reducers are explicitly
length-sensitive; top-fraction alternatives are included to test whether
normalizing the tail size transfers better across step spans.

##### Stage B — survivor-only temporal transform before fixed aggregation

Freeze one Stage-A aggregator before Stage B. Then at most four transform
templates may be instantiated, one at a time, using that same aggregator:

1. `P2R_B_POS_CUSUM_TEMPLATE`: standardized risk `z_t`, reset recursion
   `c_t=max(0,c_{t-1}+z_t-kappa)` with reference `kappa`, reset semantics, and
   warm-up frozen before labels.
2. `P2R_B_SWVAR_TEMPLATE`: one fixed trailing window
   `v_t=Var(z[max(1,t-w+1):t])`; the chosen Stage-A aggregator acts on `v_t`.
   This is the curve-level analogue of historical `sw_var_peak`, not a second
   post-hoc peak choice.
3. `P2R_B_HIGHPASS_TEMPLATE`: one causal residual, default
   `h_t=z_t-EMA_alpha(z)_t`, with `alpha`, initialization, and risk orientation
   frozen before labels.
4. `P2R_B_DSP_CAUSAL_TEMPLATE`: one fixed trailing DSP energy/band-pass curve
   only if its implementation passes the registered causal and
   suffix-invariance contract. Otherwise it remains retrospective-local only
   and is ineligible for Phase 5.

These are templates, not four automatic runs. The exact derived variant names
must bind the selected Stage-A parent and the frozen transform parameters.
CUSUM requires the C3 premise gate, SW variance the C1/C2 premise gate,
residual/high-pass the C8 complementarity premise, and DSP the C6 causal
premise. No arbitrary filter, transform, and aggregator cross is permitted.

##### Evaluation and promotion

Primary endpoint is ProcessBench first-error macro F1 on the same common rows,
source-question groups, step spans, and frozen threshold as the reference.
Use 20,000 paired whole-question bootstrap draws. Across every opened Stage-A
and Stage-B primary contrast, report simultaneous intervals or Holm-adjusted
one-sided inference. Raw best remains separately labelled `selection-opened`.

Required secondary outputs are exact-error localization, within-one,
clean-trace abstention, family W/T/L, worst scorer cell and family, exact
prediction flips, and step-length strata. Define short/medium/long cut points
from calibration-only 1/3 and 2/3 quantiles of the annotated first-error-step
token length, freeze them, and apply them to evaluation errors. Clean traces
remain in the clean-abstention panel rather than receiving a fictitious target
step length. Also report selected-step-length distributions as descriptive
bias diagnostics.

A reducer promotes only with a multiplicity-aware paired F1 interval lower
bound above zero versus `step_top5mean`, no material exact-error regression,
and no worst-cell breach; it must also beat its best simpler parent under the
registered parent gate. ProcessBench and PRMBench remain separate. The frozen
ProcessBench reducer transfers later to PRMBench without PRMBench tuning or a
shared aggregate. Phase 5 accepts only prefix-safe transforms: future-token
pooling, full-trace CUSUM, and future-dependent DSP are forbidden.

Primary ProcessBench promotion rule relative to the strongest same-access Phase 1 reference:

- Mean Local F1 delta at least +0.005.
- Paired, source-grouped 20,000-bootstrap interval lower bound greater than zero.
- At least six of eight Qwen cells nonnegative.
- Worst Qwen cell delta no worse than -0.020.
- Neither exact-error nor clean-abstention component regresses by more than 0.010.

Because several Phase 2 arms are screened, report simultaneous paired-bootstrap intervals or Holm-adjusted one-sided tests across all attempted promotion contrasts. The raw best point estimate is always labeled selection-opened until this correction is applied.

Hard failures:

- Any source-copy leakage or fold inconsistency.
- Missing-score population change.
- Any noncausal transform in the causal lane.
- ProcessBench worst-cell delta below -0.030.
- PRMBench overall point delta below -0.010 in an interim score-only check.

### Phase 3 — principled fusion and selection

Only Phase 2 survivors may enter fusion. Freeze at most three evidence blocks and at most twelve scalar coordinates unless an explicit dimension amendment is approved.

Fusion order:

1. Equal provenance-family average.
2. Ordinary label-blind IU covariance fusion.
3. One conditional mechanism only if its premise audit passes: self-basis innovation for residual complementarity, or B3 weighting for reliability heterogeneity.

SU-PCR, STG-SU-PCR, DUFS, graph hierarchy, and transform clustering are negative controls, not default escalation paths. A method may be rerun only when the new survivor set establishes the exact premise the older experiment lacked.

Phase 3 uses the same ProcessBench promotion gate as Phase 2 and must also beat the best atomic parent by +0.003 with a paired interval lower bound above zero. This prevents a complex fusion from being promoted merely for matching a compact source.

#### Registered survivor-only hierarchical family-expert branch

`P3_HIER_FAMILY_EXPERTS` is a bounded design branch, not an executed result
and not permission for an unconstrained fusion search. It may be instantiated
only after Phase 2 has identified the exact surviving families and transforms.
The branch replaces the fixed equal mean *inside multi-view provenance
families* with one family expert. The `entropy_level` family is the singleton
`{entropy_series}` and therefore passes through unchanged; applying U-PCR,
IU-PCR, or SU-PCR inside that singleton is undefined and forbidden.

The frozen broad-view family inventory is:

| family | member count | inner-expert eligibility |
|---|---:|---|
| `entropy_level` | 1 | pass through only |
| `entropy_dynamics` | 3 | U-PCR or IU-PCR; SU-PCR only after a family-specific sparse-error premise and identifiability gate |
| `structural` | 10 | U-PCR or IU-PCR; SU-PCR only after the same premise gate |
| `sampled_energy` | 4 | U-PCR or IU-PCR; SU-PCR only after the same premise gate |
| `partition_energy` | 4 | U-PCR or IU-PCR; SU-PCR only after the same premise gate |
| `topk_distribution` | 6 | U-PCR or IU-PCR; SU-PCR only after the same premise gate |

Before any fit, the executable Phase-3 registry must replace this eligibility
table with an exact allowed inner-flavour roster for each surviving family.
The family-member names must be a subset of the frozen
`BROAD_FAMILIES` mapping in `spectral_utils/local_online_comprehensive.py`;
no new view may be introduced through this branch. Non-surviving families are
absent rather than zero-filled. Each multi-view expert produces exactly one
family score, and the surviving family scores are combined by one outer rule
selected from ordinary U-PCR or ordinary IU-PCR and then frozen.

All view standardization, inner experts, and the outer fusion are fitted on
calibration rows only. A family flavour or the outer rule may be chosen by a
predeclared label-free stability criterion or a nested calibration split.
Audit/test ProcessBench F1 and all PRMBench labels are forbidden for this
choice. The selection rule, tie break, folds, seeds, and every rejected option
must be written to the execution registry before scores are opened.

This paragraph is the explicit dimension amendment for this branch only: it
may consume the surviving subset of the frozen 28 raw views internally and
expose at most six family scores to the outer fusion. It does not loosen the
three-block/twelve-coordinate cap for any other Phase-3 arm.

The frozen hierarchy must be compared separately with (a) the matched
equal-within-family reference using the same outer rule, (b) the current
equal-family-mean plus U-PCR reference, and (c) its strongest atomic parent.
It must pass the existing paired bootstrap, worst-cell, and parent-improvement
gates. Phase 4 reports PRMBench transfer separately: ProcessBench improvement
with unacceptable PRMBench degradation is a `PROCESSBENCH_SPECIALIST`, never
an averaged cross-task win. Phase 5 may receive only prefix-safe family
members; full-trace CUSUM and STFT-style views cannot transfer unchanged.

This branch is intentionally narrower than the rejected transform hierarchy:
the grouping is the frozen provenance family contract, singleton handling is
explicit, selection is nested/calibration-only, and atomic survival is a hard
prerequisite. Those differences justify a bounded rerun, but provide no
positive evidence until the registered comparisons are executed.

### Phase 4 — PRMBench localization transfer

Freeze the winning ProcessBench scorer, orientation, normalization, and fusion weights before reading Phase 4 outputs. Do not tune on PRMBench labels.

Report:

- Overall AUROC and AUPRC.
- Paired deltas and 20,000-response-group bootstrap intervals.
- Every error family separately; multi_solutions remains descriptive because it is single class.
- prm_train and prm_test source strata separately.
- Worst evaluable family and worst source-stratum delta.

Cross-task verdicts:

- UNIVERSAL-CANDIDATE: passes ProcessBench promotion; PRMBench AUROC point delta at least -0.002 and interval lower bound at least -0.005; no evaluable family below -0.020.
- PROCESSBENCH-SPECIALIST: passes ProcessBench but fails a PRMBench guard.
- PRMBENCH-SPECIALIST: PRMBench interval lower bound is above zero but ProcessBench promotion fails.
- NO-PROMOTION: neither task supports improvement.

A PRMBench improvement claim requires its own paired interval lower bound above zero. Noninferiority is not an improvement claim.

### Phase 5 — lower-priority early-detection transfer

Only causal versions of a frozen Phase 4 candidate may enter. Future-token top5 pooling and full-trace-mean CUSUM are forbidden.

Primary prefix budgets: 64 and 128 tokens. Report outcome AUROC/AUPRC, declaration false-warning rate, declaration accuracy, and tokens observed. Include suffix-invariance tests: changing an unseen suffix must not alter any prefix score.

This phase evaluates early warning, not valid stopping, unless the evaluator explicitly forces answer closure after a stop. A complete stopping claim requires the separate risk-delay contract described in the Online Auditing digest and is outside the primary goal.

## 9. Mind the Gap direct-comparison contract

The direct reference must satisfy all of the following:

- Same ProcessBench response IDs and scorer copies as the candidate.
- Same token-to-step spans and missing-step rules.
- Same access tier; no teacher-forced or external-verifier advantage.
- Same five grouped folds and threshold objective.
- Same official exact-error/clean-abstention F1 code.
- Same paired bootstrap draws.

Report the paper's published setting separately as literature context. Never subtract the paper table from the repository score or call that a direct delta.

## 10. Paper scan and transferable ideas

The repository paper index and digests were scanned across reasoning localization, fusion, feature selection, temporal/change-point methods, spectral methods, and hallucination detection.

Ideas retained:

- Mind the Gap: one local EMA evidence-drop curve and worst-drop reference, subject to the direct-comparison contract.
- EDIS: burst/rebound morphology only, after the entropy baseline.
- UUC: positive area and persistence as compact duration transforms; its response-level integrated score is not first-error evidence.
- HALT and streaming hallucination work: treat uncertainty as a temporal process and keep supervised sequence models as high-access ceilings, not same-access baselines.
- Online Auditing: risk-delay and transition-cost framing for the tertiary early lane only.
- Spectral and graph papers: use only when a target-free premise audit establishes nontrivial geometry beyond length or scale.

Ideas not retained as primary candidates:

- Supervised hidden-state probes, teacher-forced top-k GRUs, attention-only systems, or externally verified multi-sample selection. Their access and supervision differ from the primary gray-box task.
- Generic Laplacian/SPEC/graph selection without a passed target-free premise audit.
- Large feature banks or unconstrained subset search.

Relevant repository evidence:

- papers/digests/mind-the-gap-catching-hallucinations-via-evidence-drop.md.
- papers/digests/edis-paper.md.
- papers/digests/uncertainty-under-the-curve-a-sequence-level-entropy-area-me.md.
- papers/digests/halt-hallucination-assessment-via-log-probs-as-time-series.md.
- papers/digests/streaming-hallucination-detection-in-long-chain-of-thought-r.md.
- papers/digests/online-auditing-of-information-flow.md.
- PROGRESS.md:972-975 and 2343-2387.

## 11. arXiv:2603.09906 transfer audit

Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs studies closed-book factual QA, not ProcessBench or PRMBench localization. It finds two mechanisms: a content-independent compute buffer and factual priming through self-generated intermediate facts. Clean intermediate factual traces correlate with better final answers, and multi-sample selection using fact extraction plus an external verifier can improve expected answer accuracy.

Transferable to this program:

- Stratify candidate errors by whether a trace contains factual claims, and use within-question paired analyses so question difficulty is not mistaken for a localization signal.
- Treat trace length or compute budget as a nuisance variable; the paper's length effect is non-monotonic and does not make length a correctness feature.
- Optionally run an external-verifier semantic ceiling in a separately named high-access lane.

Not transferable to the primary compact localizer:

- The paper uses closed-book factual QA, multiple sampled traces per question, fact extraction, and external search-enabled verification.
- It does not evaluate first-error localization, ProcessBench, or PRMBench.
- Its factual-correctness labels cannot be used to choose or orient label-free entropy features.
- It gives no evidence that a factual-priming scalar should be added to a single-pass gray-box feature vector.

Repository digest: papers/digests/2603-09906.md.

Official source: https://arxiv.org/abs/2603.09906 and https://arxiv.org/pdf/2603.09906.

## 12. Required artifacts and acceptance checklist

Before Phase 1 execution, create and checksum:

- Executable candidate roster and dependency graph.
- Exact population registry and common-row hashes for both tasks.
- Fold/source-group manifest.
- Access and supervision declaration for every candidate.
- Feature identity/reconstruction manifest.
- Mind the Gap direct-reference implementation audit.
- Frozen metrics and simultaneous-inference plan.
- Environment and source commit manifest.

Every phase report must separate:

- Raw best score.
- Registered comparison and paired uncertainty.
- Per-cell or per-family worst behavior.
- Development, transfer, and genuinely fresh confirmation evidence.
- Committed artifacts from working-tree-only artifacts.
- Task-general candidates from task specialists.

No scorer implementation or scientific experiment follows from this design until the executable scorer roster is frozen and explicitly approved. Reporting infrastructure may be implemented first. No commit or push is authorized.

## 13. Implemented reporting contract

The Reporting Phase is implemented at
`results/reasoning_localization_03662_v1/`. It freezes the presentation and
validation layer before any Phase 0 scorer is run.

### 13.1 Authoritative interfaces

- `METHOD_REGISTRY.json`: fifteen method-family explanations with repository
  evidence.
- `VARIANT_REGISTRY.json`: all R0-R4 and C1-C8 definitions, twelve planned
  Stage-A reducer rows, four non-rankable Stage-B transform templates, the
  matched equal-family reference and survivor-only hierarchical family-expert
  templates, fifteen historical context rows, deterministic derived names,
  and independent execution, decision, and evidence states.
- `EXPERIMENT_REGISTRY.json`: Reporting, P0-P5, and the registered Phase-2
  reducer-branch questions, populations, comparators, grouped bootstrap
  definitions, prerequisites, and promotion gates.
- `METRICS_LONG.csv`, `CONTRASTS_LONG.csv`, and `GATES_LONG.csv`: the only
  numeric/result inputs to the report.
- `CLAIMS.json` and `EXAMPLES.json`: machine-validated claim links and the
  seed-`2026082901` deterministic trace-case contract.
- `PLOT_MANIFEST.json`: twenty-one figure contracts, including source table,
  selection, comparison group, bootstrap definition, and selection rule.
- `REPORT_MANIFEST.json`: source commit, every report input and frozen
  historical source SHA, resolved figure source hashes, embedded-data hash,
  and final HTML hash.

Historical Stage-4 rows are bound back to
`STAGE_4_AGGREGATE.csv` and `STAGE_4_INTERVALS.csv` by exact SHA and row
selector. A copied value that differs from its source fails the build.

### 13.2 Renderer and update workflow

Run:

```bash
python3 scripts/reasoning_localization/build_reasoning_localization_report.py
python3 scripts/reasoning_localization/build_reasoning_localization_report.py --check
python3 scripts/reasoning_localization/build_reasoning_localization_report.py --snapshot phase_0
```

The renderer in `spectral_utils/reasoning_localization_reporting.py` produces
one English, self-contained `REPORT.html` with inline CSS, JavaScript, SVG,
print rules, filters, sorting, accordions, and CSV reconstruction from embedded
JSON. It implements forest, contrast-forest, waterfall, heatmap, gate-matrix,
scatter, line, and lineage renderers. A plot with no eligible registered rows
is `PLANNED`; it receives neither a point nor a zero.

After every attempted variant, update the registries/long tables and rebuild
the live report before discussion. Do not run the next variant until the
current result and gate rows are reviewed. At each approved phase boundary,
create `snapshots/phase_N/`. An existing snapshot is verified byte-for-byte
and can never be overwritten; a changed reporting-only release must use an
`amendment_<slug>` snapshot.

### 13.3 Fail-closed rules

The build fails when:

- a context row is rankable;
- a planned/blocked row carries a numeric result;
- a comparison group mixes task, population, or metric regimes;
- a source artifact, copied value, or SHA is inconsistent;
- a claim names a missing plot, table, contrast, or manifest;
- a parent is missing or the lineage contains a cycle;
- a completed deterministic-example artifact omits a required category; or
- a generated report or immutable snapshot differs from the registered build.

The report never computes a ProcessBench/PRMBench aggregate. ProcessBench,
PRMBench, and Early columns remain separate in the master table, and
historical context has its own gray visual lane.

### 13.4 Reporting Phase acceptance

The design-state report contains 28 method cards, 18 figure contracts, three
currently rendered SVGs, all seven experiment contracts, the claim ledger,
the deterministic-case placeholders, and full provenance appendices. Sixteen
reporting tests pass, including synthetic coverage of every registered chart
kind and tamper detection for immutable snapshots. Two fresh builds are
byte-identical. Browser QA passed at 1440px and 390px; the mobile document has
no page-level horizontal overflow, while wide tables and SVGs retain local
scroll containers. No browser console warnings or errors were observed.

At the time of this immutable acceptance snapshot, P0-P5 were `PLANNED` and no
scientific variant carried a numeric result. The living report has since added
the source-bound P0-S0 and P0-S1 retrospective audit states without altering
the immutable Reporting snapshots or making a method promotion. It then
registered the unexecuted `P3_HIER_FAMILY_EXPERTS` design branch and its
matched equal-family reference. The current living amendment adds the
unexecuted Phase-2 reducer branch: the roster now contains 47 cards, eight
experiment contracts, and 21 plot contracts, including a reducer paired-delta
forest and step-length heatmap. None of these design additions carries a new
metric or promotion.

The final design-state release is frozen at
`snapshots/amendment_reporting_manifest_plot_contract/`; earlier Reporting
and mobile/chart-renderer snapshots remain immutable provenance rather than
being overwritten.
