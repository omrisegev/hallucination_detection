# Reasoning Localization 0.3662 Anchor v1

Status: PHASE 0 COMPLETE. P0-S0 through P0-S4, the bounded P0-S2I interaction control, and both registered P0-S5 population states are complete. S0-S4 preserve common historical rows for adjacent attribution; S5A/S5B are explicitly nonpaired retrospective population transfers. No state performs new model inference or method promotion.

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

### 3.6 Frozen next state P0-S2A: modern answer-detector bridge

P0-S2A is frozen in
`results/reasoning_localization_03662_v1/phase_0/P0_S2A_EXECUTION_REGISTRY.json`
before its result is opened. Its parent is the completed P0-S1 state. The
population, historical 40-percent calibration and 20-percent audit roles,
family6 level representation, local-head fit, step-max locator, within-cell
ProcessBench-F1 threshold objective, scorer-copy grouping, and 20,000-draw
paired source-question bootstrap are unchanged.

The single opened factor is the complete-answer detector. RegisteredGlobal
mixed-v2 ordinary IU is replaced by the modern registered
`answer_dufs_liu_mixed` head: mixed-v2 confidence features, elapsed length
excluded, a label-free DUFS-gated graph with `k=7`, Laplacian-IU
`lambda=0.1`, seeds 11/23/37, and 80 epochs. Unlike the original GL-LIU study's
transductive score fit, this bridge fits the detector on the historical
calibration rows only so the split factor does not change. P0-S2A is an audit
edge and cannot promote a method.

The runner must reproduce every P0-S1 unit, target, locator, prediction,
detector score, threshold, and metric within `1e-12` before the new detector is
evaluated. A changed population hash, a label entering detector fitting, or
opening the purely local detector in the same run is a hard failure. The
purely local detector remains a separate unopened P0-S2B state.

P0-S2A completed without a hard failure. All 1,270 scorer rows and 635 paired
source-question groups retained the S1 population hash
`d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05`;
every S1 target, locator, prediction, score, threshold, and metric reproduced
within `1e-12` before the detector changed. The modern detector fit 28 features
in Qwen cells and 29 in Llama cells, used no labels during fitting, ran no new
model inference, and left S2B unopened.

The candidate macro F1 was `0.32859546976358334`, versus
`0.33007771561392063` for S1. The paired delta was
`-0.001482245850337259`, 95-percent interval
`[-0.00871467835648367, 0.00525866350827324]`, family W/T/L `1/1/2`, and
worst scorer-cell delta `-0.013490784822629609`. Exact-error changed by
`+0.0032013396375098514` with an interval crossing zero. Within-one improved
by `+0.012837148076509777`, interval
`[0.005292749484365179, 0.02127993059645836]`, but clean abstention fell by
`-0.04244661327994661`, interval
`[-0.06541774191302986, -0.02171650406760252]`.

Decision: `COMPLETE / NO_PROMOTION / RETROSPECTIVE`, with statistical status
`INCONCLUSIVE`. The calibration-only modern DUFS-LIU detector is not a
supported macro-F1 improvement on this bridge edge, but the interval crossing
zero is not a rejection and not evidence of equality. It does not explain the
high historical 0.3662 score. It changes
the operating tradeoff toward more localized error declarations at the cost
of clean abstention. This result does not evaluate the purely local detector
or the modern five-fold population.

### 3.7 Executed state P0-S2I: bounded reducer-by-detector interaction control

The S2A discussion exposed a valid attribution limitation: the DUFS-LIU
detector edge had been measured only after changing the reducer to step max.
Therefore the detector effect under the historical top-five reducer and the
reducer effect under DUFS-LIU remained unmeasured. The original P0-S2B state
does not answer this question because it changes to a purely local detector
while retaining step max.

Before P0-S2B, the program inserted one bounded audit state rather than an
unregistered factorial search. P0-S2I was frozen in
`results/reasoning_localization_03662_v1/phase_0/P0_S2I_EXECUTION_REGISTRY.json`.
Its parent is P0-S2A, and the sole changed factor is
`step_max_token_argmax -> step_top5mean`. The historical population, 40/20
roles, family6 level local fit, calibration-only mixed-v2 DUFS-LIU detector,
threshold objective, scorer-copy grouping, access tier, and 20,000 common
paired source-question bootstrap draws are unchanged. The runner first
reconstructs every S2A unit, score, locator, prediction, threshold, and metric
within `1e-12`. It also reproduces the frozen S1-S0 and S2A-S1 bootstrap edges
on the common four-state draws. No new model inference or GPU execution occurs,
and P0-S2B remains unopened.

The four audited cells are:

| Detector | `step_top5mean` | `step_max` |
| --- | ---: | ---: |
| RegisteredGlobal | S0: `0.3662328342` | S1: `0.3300777156` |
| calibration-only DUFS-LIU | S2I: `0.3632846791` | S2A: `0.3285954698` |

Three estimands must remain separate:

- Adjacent pooling effect under DUFS-LIU: S2I minus S2A is
  `+0.0346892093`, 95-percent interval
  `[0.0060214426, 0.0650215022]`, family W/T/L `4/0/0`, with worst
  scorer-cell delta `-0.0168441264`. Exact-error improves by `+0.0436373934`
  and within-one by `+0.0421858210`; both intervals exclude zero. Clean
  abstention changes by `-0.0131589923` with an interval crossing zero and a
  worst scorer cell of `-0.2222222222`.
- Same-reducer detector effect under top-five: S2I minus S0 is
  `-0.0029481551`, interval `[-0.0085742309, 0.0021306150]`, family W/T/L
  `1/0/3`. Thus the raw S2I score is close to 0.3662, but neither superiority,
  equivalence, nor noninferiority is established.
- Cumulative S2A displacement from the anchor: S2A minus S0 is
  `-0.0376373644`, interval `[-0.0677649616, -0.0089256541]`. This is the
  total pooling-plus-detector displacement and is never labeled a detector
  effect.

The preregistered macro-F1 difference-in-differences,
`(S2A-S1) - (S2I-S0)`, is `+0.0014659092`, interval
`[-0.0068185789, 0.0094863622]`. The audit therefore does not support a
macro-F1 reducer-by-detector interaction large enough to explain the observed
drop. This is not proof of exact additivity: clean-abstention and within-one
interaction contrasts are nonzero in opposite directions. The claim-safe
conclusion is narrower: historical top-five pooling recovers most of the high
absolute score in both tested detector regimes, while neither detector
contrast supports an improvement.

P0-S2I is `COMPLETE / NO_PROMOTION / RETROSPECTIVE`; its adjacent pooling edge
is `SUPPORTED_IMPROVEMENT` for Phase-0 directional attribution, while the
same-reducer detector edge and interaction residual are `INCONCLUSIVE`.
Canonical artifacts are
under
`results/reasoning_localization_03662_v1/phase_0/p0_s2i_interaction_control/`.
The live report shows the mainline waterfall separately from a factorial
contrast forest, interaction-residual forest, and deterministic S2A-to-S2I
prediction-flip audit.

### 3.8 Executed state P0-S2B: purely-local detector bridge

P0-S2B returns to the frozen mainline parent P0-S2A rather than treating the
S2I interaction branch as a sequential state. It retains the exact historical
1,270 scorer rows and 635 source-question groups, fixed 40/20 roles, family6
level representation and local-head fit, step-max locator, threshold objective,
access tier, and 20,000 paired source-question bootstrap draws. The sole changed
factor is the answer detector: calibration-only mixed-v2 DUFS-LIU is replaced
by the maximum of the same fitted family6 local-risk curve used by the locator.

The registry was frozen before result access at
`results/reasoning_localization_03662_v1/phase_0/P0_S2B_EXECUTION_REGISTRY.json`.
The first execution attempt stopped before writing any scientific artifact
because the reconstruction wrapper passed S2A artifacts into the inner S1
reconstruction slot. The runner was corrected to enforce the complete
S1-to-S2A-to-S2B chain, rehashed, and preflighted again. The successful run
reconstructs every S2A unit, target, locator, prediction, detector score,
threshold and metric within `1e-12`; all candidate locators are identical to
S2A, so only detector score and the resulting thresholded decision can change.

P0-S2B reaches macro F1 `0.3065027012935364`, compared with
`0.32859546976358334` for S2A. The adjacent detector-only delta is
`-0.0220927684700469`, 95-percent interval
`[-0.044614445807279905, -0.00039874276031798663]`, family W/T/L `0/0/4`,
and worst scorer-cell delta `-0.08180147058823528`. Exact-error localization
falls by `-0.020620433053943694`, interval
`[-0.04024650299367711, -0.0017378509755708066]`. Clean abstention changes
by `-0.025961378044711415` and within-one by `-0.024147727272727286`; both
secondary intervals cross zero.

The raw cumulative displacement from the S0 anchor is
`0.3065027012935364 - 0.3662328341717007 = -0.0597301328781643`. This value
is displayed as the overall bridge trajectory, not labeled as a detector
effect: reducer and multiple detector changes lie between S0 and S2B. The
claim-safe adjacent conclusion is that, under step max in this retrospective
regime, the separate DUFS-LIU complete-answer detector is better than using
the local curve maximum for both detection and localization.

P0-S2B is `COMPLETE / NO_PROMOTION / RETROSPECTIVE`, with statistical status
`SUPPORTED_HARM` on its adjacent detector edge. No model inference, GPU
work, label-seeing curve fit, source mutation, representation change, split
change, or population change occurred. The next bridge remains unopened.

### 3.9 Executed state P0-S3A: raw-entropy representation bridge

P0-S3A uses P0-S2B as its direct mainline parent. It retains the exact
historical 1,270 scorer rows and 635 source-question groups, fixed 40/20 roles,
purely-local detector construction, step-max token argmax mapped to the known
step span, threshold objective and tie break, access tier, and 20,000 paired
source-question bootstrap draws. The sole changed factor is the local-risk
representation: the fitted family6 level curve is replaced by the raw
`token_entropies` curve. The same raw curve supplies both the maximum answer
detector and the step-max locator.

The registry was frozen before result access at
`results/reasoning_localization_03662_v1/phase_0/P0_S3A_EXECUTION_REGISTRY.json`.
The successful run reconstructs every S2B unit, target, locator, prediction,
score, threshold and metric within `1e-12`. All raw curves are finite; their
construction sees no label; the population SHA remains
`d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05`.
Changing representation changes the token locator on 559 of 1,270 scorer rows,
as expected, while no other registered factor changes.

P0-S3A reaches macro F1 `0.3110940034934562`, compared with
`0.3065027012935364` for S2B. The adjacent representation-only delta is
`+0.004591302199919791`, 95-percent interval
`[-0.030495488290424935, 0.03921200441794944]`, family W/T/L `2/0/2`, and
worst scorer-cell delta `-0.04639371496084849`. Exact-error has an unsupported
point gain of `+0.011837121212121212`; clean abstention has an unsupported
point loss of `-0.020799966633299936`. Within-one rises by
`+0.03510436814958093`, but its interval
`[-0.0012564890012784482, 0.07076080810426001]` still crosses zero.

The raw cumulative displacement from S0 is
`0.3110940034934562 - 0.3662328341717007 = -0.0551388306782445`. It is a
waterfall position, not a representation effect: the earlier reducer and two
detector changes remain between S0 and S3A. This state also does not reconstruct
the historical approximately 0.3614 entropy baseline, whose original
top-five/RegisteredGlobal contract is different.

P0-S3A is `COMPLETE / NO_PROMOTION / RETROSPECTIVE`, with statistical status
`PROMISING_UNCONFIRMED`. Raw entropy is simpler and descriptively slightly
better than family6 on this specific weak parent, but the paired evidence does
not support superiority. The interval crossing zero is not rejection and the
branch remains eligible for an independently preregistered confirmation. No model inference, GPU
work, label-seeing representation fit, source mutation, split change, or
population change occurred.

### 3.10 Executed state P0-S3B: IU29 representation bridge

P0-S3B uses P0-S3A as its direct mainline parent. It retains the exact
historical 1,270 scorer rows and 635 source-question groups, fixed 40/20 roles,
purely-local detector construction, step-max token argmax mapped to the known
step span, threshold objective and tie break, access tier, and 20,000 paired
source-question bootstrap draws. The sole changed factor is the local-risk
representation: raw token entropy becomes the registered `LOCAL_IU29` curve.
All 29 `SHARED_TOKEN_VIEWS` streams are transformed under the frozen mixed-v2
contract, standardized and oriented on calibration rows only, and fused by the
two-component IU-PCR rule with `scale_ratio=0.25`. The same fitted IU29 curve
supplies both the maximum answer detector and the step-max locator.

The registry was frozen before result access at
`results/reasoning_localization_03662_v1/phase_0/P0_S3B_EXECUTION_REGISTRY.json`.
The successful run reconstructs every S3A unit, target, locator, prediction,
score, threshold and metric within `1e-12`. All eight cells retain all 29 finite
streams; method fitting uses calibration rows only and sees no labels; a second
fit/score reconstruction agrees within `1e-12`; and the population SHA remains
`d12d651cad9bec326686c2c83070644d22ca058ed57e942f683452050e757a05`.
Changing representation moves the token locator on 501 of 1,270 scorer rows.

P0-S3B reaches macro F1 `0.2996587594711835`, compared with
`0.3110940034934562` for S3A. The adjacent representation-only delta is
`-0.011435244022272775`, 95-percent interval
`[-0.04461391138148266, 0.022121445421450597]`, family W/T/L `2/0/2`, and
worst scorer-cell delta `-0.20003988831272437`. Exact-error changes by
`-0.0019091625474604267`; clean abstention by `-0.012247664330997658`; and
within-one by `+0.017067169926212475`. Every secondary paired interval also
crosses zero. The severe worst-cell point loss is therefore a robustness warning
that requires stratified follow-up, not a supported aggregate harm verdict.

The raw cumulative displacement from S0 is
`0.2996587594711835 - 0.3662328341717007 = -0.0665740747005172`. It is a
waterfall position, not the IU29 representation effect: the earlier reducer,
detector and representation changes remain between S0 and S3B. This historical
40/20 purely-local step-max result is also not the recent approximately 0.307
IU29 result, which uses a different population, split and detector contract.

P0-S3B is `COMPLETE / NO_PROMOTION / RETROSPECTIVE`, with statistical status
`INCONCLUSIVE`. Its negative point estimate and zero-crossing interval do not
support an improvement, a material-harm claim, rejection, or equality. No model
inference, GPU work, label-seeing representation fit, source mutation, split
change, or population change occurred.

### 3.11 Executed P0-S4 and P0-S5: split and population bridges

P0-S4 changes only the threshold split on the unchanged 1,270 historical
scorer rows: historical fixed 40/20 roles become deterministic five-fold
source-question cross-fit. The IU29 score, locator, local detector, step-max
reducer and population hash are unchanged. Macro F1 is `0.29401957271717755`;
the paired S4-minus-S3B delta is `-0.005639186754005907`, 95-percent interval
`[-0.02022305591272321, 0.009502877546169077]`, family W/T/L `1/0/3`, and
worst-cell delta `-0.06735524934618886`. This is `INCONCLUSIVE`, not rejection.
Clean abstention separately falls by `-0.04343093093093093`, with interval
`[-0.07775322652501392, -0.009152817979158545]`; that supported component harm
is an operating-point warning and does not overwrite the aggregate verdict.

P0-S5 imports the dual-build-identical frozen `token_iu29__step_only_null_v1`
adapter without new fitting or inference. S5A scores `0.2931182814184147`,
95-percent grouped panel interval `[0.278524366896694, 0.30625044155576914]`,
on the current eight-Qwen panel. S5B scores `0.2943961703375378`, interval
`[0.2822583006109942, 0.30491459952679506]`, after adding four Llama-3.1
cells. The S5B-minus-S5A panel-composition delta is `+0.0012778889191231158`,
interval `[-0.006708916091292374, 0.00920811521404516]`, family W/T/L `2/0/2`.
Because the added scorer panel contains independently generated traces, this is
an `INCONCLUSIVE` composition diagnostic, not a row-paired treatment effect.
Likewise, the small raw S4-to-S5A difference must not be interpreted as a
population effect: the historical and current panels lack a shared row identity.

The final raw waterfall positions relative to the `0.3662328341717007` anchor
are `-0.07221326145452315` for S4, `-0.0731145527532860` for S5A and
`-0.0718366638341629` for S5B. Only adjacent S0-S4 common-row edges support
factor attribution; the S5 positions are descriptive population states.

### 3.12 Registered bridge order and final accounting

The 0.3662 regime is not directly comparable with the modern approximately 0.307 ProcessBench score. Phase 0 freezes an intersection of shared rows and changes one factor at a time in this order:

1. Historical replay: historical eight-cell audit, fixed historical roles, RegisteredGlobal detector, family6 level, step_top5mean.
2. Reducer bridge: change only step_top5mean to step max.
3. Detector bridge: change only RegisteredGlobal to the modern registered answer detector under step max.
4. Bounded interaction control: before the purely-local detector, hold DUFS-LIU fixed and change only step max back to top-five so detector and reducer effects can be measured in both 2x2 strata. This is a branch control, not a waterfall edge.
5. Purely-local detector bridge: return to the frozen mainline parent and change only the answer detector; completed as P0-S2B, with the local curve maximum worse than S2A by 0.02209 macro F1 and a paired interval below zero.
6. Representation bridge: change only family6 to raw entropy, then IU29. P0-S3A has an unsupported +0.00459 raw-entropy point gain; P0-S3B then changes only raw entropy to IU29 and has an inconclusive -0.01144 point delta. Both sub-bridges are complete.
7. Split bridge: completed as P0-S4; changing only fixed calibration/audit roles to five-fold cross-fit is aggregate-inconclusive but produces a supported clean-abstention loss.
8. Population bridge: completed as P0-S5A/S5B using the frozen current eight-Qwen and full twelve-cell panels; these are explicitly nonpaired descriptive transfers.

For every common-row bridge edge after S0, report the paired delta and grouped interval. Population states without shared row identity receive grouped state intervals but no fabricated paired delta or flips. Phase 0 is an audit waterfall with explicit residual interaction and population boundaries, not a new leaderboard winner.

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
- R2: current-protocol family6 level plus step_top5mean with the same modern
  answer detector used by the other Phase-1 references. The exact historical
  family6/RegisteredGlobal row remains `R2_HISTORICAL_FAMILY6_BRIDGE` in Phase
  0 and is never ranked with this current-population R2.
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

The Phase-1 score contract is frozen before labels open: every local step score
is empirical-midrank normalized, combined with the same `equal_feature_mean`
response score by a geometric mean, and evaluated by the same model-specific
five-fold threshold code. R3 is the current same-access incumbent comparator;
the strongest raw Qwen-eight point estimate that passes all execution and
population gates becomes the Phase-2 reference. Baseline selection is reported
as selection-opened and is not itself an improvement claim. Directional
contrasts use the preregistered aggregate macro-F1 practical bounds of `+0.005`
for benefit and `-0.005` for harm, with 20,000 paired source-question draws.
Intervals crossing zero use the registered uncertainty vocabulary and are never
generic rejections.

Deliverables:

- ProcessBench overall, per-cell, per-model, exact, within-one, and clean-abstention metrics.
- PRMBench overall, per-error-family, and original source-stratum AUROC/AUPRC for the score-only references.
- Runtime, peak memory, and missing-score audit.

The sealed Phase-1 PRMBench evaluator exposes source-group identifiers and
error families but not the original `prm_train`/`prm_test` membership. Phase 1
therefore reports those source-stratum rows as
`BLOCKED_METADATA_NOT_IN_SEALED_EVALUATOR` instead of guessing membership. This
does not alter the overall or error-family estimators, but the metadata must be
repaired and frozen before Phase 4 can claim source-stratum transfer.

Gate: a reference must be executable, checksum-frozen, and common-population complete before it can anchor Phase 2.

### Phase 2 — targeted entropy-centered ablations

#### C3--C8 completion amendment (2026-08-30, registered before results)

The user requested completion of the entire frozen atomic roster. This permits
execution of scientifically diagnostic arms after a parent gate has failed; it
does not waive that gate or make the arm promotion-eligible. In particular,
C3 is a diagnostic continuation after the C1 hard failure, C7 remains an
exploratory repository adaptation, and C8 remains diagnostic. C6 may promote
only if its atomic source-parent premise is satisfied before C6 execution.

All remaining exact contracts are frozen together before C3 labels open:

- C3 uses a two-sided absolute reset CUSUM around the mixed-v2 standardized
  zero, with `kappa=0`, plus the unchanged entropy and SWVar16 channels. Each
  channel is reduced by top-ten and their step midranks receive equal weight.
- C4 and C5 add exactly one level source to entropy: sampled-token surprisal
  (`spilled_series`) or partition energy (`energy_series`), respectively.
- C6 contains exactly twelve coordinates: entropy, sampled surprisal, and
  partition energy crossed with level, causal EWMA16 (`alpha=2/17`), running
  positive-area mean above zero, and running persistence above zero. Every
  source-transform channel has equal weight; there is no operator search,
  learned weighting, or label-selected subset.
- C7 converts EDIS morphology into a causal local onset curve: burst excess
  and positive rebound-onset increment, then their maximum. The sealed input
  exposes standardized affine entropy rather than raw entropy in nats, so its
  fixed `1.36/1.33` thresholds are standardized-unit thresholds here. This is
  explicitly not a paper-exact EDIS reproduction and cannot auto-promote.
- C8 fits, for every retained IU29 stream, a response-weighted ridge-1 model
  with intercept, `log1p(token_position)`, and one-step self lag. The added
  confidence block is negative absolute donor-RMS-standardized residual. An
  ordinary two-component IU-PCR is then fit over the original 29 streams plus
  this one 29-stream residual block. The original-only reconstruction must
  alias frozen R3 under step-max; the matched C8 parent comparison uses the
  common top-ten reducer.

The completed C1--C8 primary ProcessBench family contains sixteen contrasts:
every candidate versus atomic top-ten and retained top-five. Bonferroni
simultaneous percentile intervals use that family size for macro F1. Exact
parent contrasts required for mechanism interpretation are reported
separately as paired diagnostics and cannot substitute for either primary
comparator. Hard technical, provenance, suffix-invariance, leakage, and
worst-cell failures remain binding. An inconclusive interval remains
inconclusive rather than becoming a generic rejection.

#### Post-Phase-2 scorer-family transfer confirmation

Because C7 and C8 ended the Qwen-eight screen with positive point estimates
but uncertainty spanning zero, they receive one bounded scorer-family transfer
on the four frozen Llama-3.1 cells.  The exact contract, promotion bounds, and
family6 complementarity audit are preregistered in
`REASONING_LOCALIZATION_03662_LLAMA_CONFIRMATION_V1.md`.  This is transfer
evidence, not independent fresh confirmation: the Llama traces differ from the
Qwen scorer panel, but their source questions and labels were already opened in
Phase 1.  No family6 fusion, router, or algorithmic extension may be selected
from these labels.  Such a branch can be registered only for a transfer
survivor and must begin with the bounded equal-family parent defined there.

Before C1-C8, execute the registered Phase-2 reducer branch below on the
frozen Phase-1 reference curve. Freeze its selected aggregation rule, then run
C1-C8 in the listed order with that same reducer. Each atomic arm changes only
one source, transform, or compact block relative to its parent. C7 remains an
explicit onset-morphology diagnostic rather than silently redefining the
common reducer. Stop a branch after a hard failure rather than expanding it.

Before C1 opens, `P2A_TOPK10_REFERENCE` must reproduce the frozen Stage-A
top-ten local and combined scores exactly, then obtain its own deterministic
grouped five-fold threshold under the atomic calibration contract. This is a
calibration reference, not a second reducer-selection result. Every atomic arm
fits only its held-fold threshold after complete label-free score freeze and is
paired against both `P2A_TOPK10_REFERENCE` and `R1_ENTROPY_TOP5`.

For `C1_ENT_SW16`, the entropy input is the negative mixed-v2 entropy
confidence coordinate, which is affine-equivalent to raw entropy risk. The
causal SWVar curve resets for every response and at token `t` is the population
variance (`ddof=0`) of tokens `max(0,t-15)..t`; warm-up uses only the available
prefix and a one-token window has variance zero. Entropy and SWVar are each
reduced by the frozen top-ten rule. The candidate step score is the equal mean
of their within-cell empirical step midranks, before the unchanged geometric
combination with the response detector. A deterministic suffix-invariance
replay audit is a hard gate. The SWVar premise opens Phase 2R-B only if C1 (or
the preregistered C2 sensitivity arm) survives the full ProcessBench gate.

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

`P2R_A_TOPK5_REFERENCE` executes first as an exact alias audit of
`R1_ENTROPY_TOP5`. It reconstructs and freezes the fold-specific R1 threshold
ledgers that Phase 1 used but did not export as a standalone artifact. The
reference must reproduce every local score, combined score, fold, prediction,
panel metric, and bootstrap sample within `1e-12`. This reconstruction is the
only threshold fit in Phase 2R; every later reducer is scored first and then
evaluated with these immutable reference thresholds.

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
Use 20,000 paired whole-question bootstrap draws. The original eleven
preregistered Stage-A contrasts form one closed Bonferroni family. Any arm
added after those outcomes opened belongs to a separately labelled post-hoc
family, is descriptive on this ProcessBench population, and cannot change the
confirmatory status of the original family. Raw best remains separately
labelled `selection-opened`.

Required secondary outputs are exact-error localization, within-one,
clean-trace abstention, family W/T/L, worst scorer cell and family, exact
prediction flips, and step-length strata. Define short/medium/long cut points
from calibration-only 1/3 and 2/3 quantiles of the annotated first-error-step
token length, freeze them, and apply them to evaluation errors. Clean traces
remain in the clean-abstention panel rather than receiving a fictitious target
step length. Also report selected-step-length distributions as descriptive
bias diagnostics.

Length cut points are model- and held-fold-specific, use NumPy's linear
quantile definition, and are fit only from erroneous calibration questions.
An error row is `short` when its true-error step length is at most the first
cut point, `medium` when it is at most the second, and `long` otherwise. A
descriptive stratum macro F1 combines that stratum's exact-error rate with the
unchanged full-cell clean-abstention rate; it is never used for promotion.

A reducer promotes only with a multiplicity-aware paired F1 interval lower
bound above the preregistered `+0.005` practical-benefit delta versus
`step_top5mean`, no material exact-error regression,
and no worst-cell breach; it must also beat its best simpler parent under the
registered parent gate. ProcessBench and PRMBench remain separate. The frozen
ProcessBench reducer transfers later to PRMBench without PRMBench tuning or a
shared aggregate. Phase 5 accepts only prefix-safe transforms: future-token
pooling, full-trace CUSUM, and future-dependent DSP are forbidden.

Primary ProcessBench promotion rule relative to the strongest same-access Phase 1 reference:

- Mean Local F1 delta at least +0.005.
- Paired, source-grouped 20,000-bootstrap interval lower bound greater than the preregistered +0.005 practical-benefit delta.
- At least six of eight Qwen cells nonnegative.
- Worst Qwen cell delta no worse than -0.020.
- Neither exact-error nor clean-abstention component regresses by more than 0.010.

Because several Phase 2 arms are screened, report simultaneous paired-bootstrap intervals or Holm-adjusted one-sided tests across all attempted promotion contrasts. The raw best point estimate is always labeled selection-opened until this correction is applied.

Statistical status is distinct from execution, method decision, and evidence
grade. Every completed contrast shows the raw score, point delta, interval,
practical benefit/harm bounds, and multiplicity status. The frozen vocabulary
is:

- `SUPPORTED_IMPROVEMENT`: the multiplicity-valid lower interval bound exceeds
  the preregistered minimal practical benefit.
- `SUPPORTED_HARM`: the upper interval bound is below the preregistered
  practical-harm delta. A hard failure, leakage, or gate violation is recorded
  separately as `HARD_FAILURE` and also stops the branch.
- `NONINFERIOR_ONLY` or `COMPATIBLE_WITH_PARITY`: only when the corresponding
  noninferiority/equivalence margin was preregistered and passes.
- `PROMISING_UNCONFIRMED`: positive point estimate with an interval crossing
  zero; visible for independent confirmation, never rankable or promotable.
- `INCONCLUSIVE`: the interval admits relevant benefit and harm; no directional
  verdict and no rejection.
- `DESCRIPTIVE`: retrospective, context-only, or explicitly post-hoc evidence
  that requires fresh confirmation before promotion.

An interval crossing zero is never called equality, generic failure, or
rejection. An inconclusive or promising branch may continue for a
preregistered scientific diagnostic or fresh confirmation. Supported material
harm and hard failure stop it. For future Phase 2 ProcessBench contrasts, the
preregistered aggregate macro-F1 bounds are `+0.005` benefit and `-0.005` harm;
the multiplicity adjustment remains mandatory. Phase 0 uses the already frozen
zero directional boundary only for retrospective one-factor attribution and
has no practical promotion margin.

Hard failures:

- Any source-copy leakage or fold inconsistency.
- Missing-score population change.
- Any noncausal transform in the causal lane.
- ProcessBench worst-cell delta below -0.030.
- PRMBench overall point delta below -0.010 in an interim score-only check.

##### Executed Stage-A checkpoint: `P2R_A_MAX_K1`

The required max/k=1 control was frozen and executed after the top-five
reference. It changes only the within-step reducer. The entropy token-risk
curve, empirical-rank combination with the answer detector, eight-Qwen rows,
step spans, source-group folds, ten model-by-held-fold thresholds, length
cutpoints, and 20,000-draw bootstrap stream are identical to the reference.
There is no candidate-specific threshold fit.

The frozen-threshold score is `0.2953486126694177`, with marginal interval
`[0.277955026133502, 0.31128939938482164]`. Its paired delta from
`P2R_A_TOPK5_REFERENCE` is `-0.049878127211192225`. Its interval when it was
the sole opened contrast was `[-0.063891277326654,
-0.035862667968481685]`; after the mean arm opened, the two-contrast
Bonferroni simultaneous interval became `[-0.06615091863351291,
-0.03388117032208545]`. After top-k=2 opened, the three-contrast interval
became `[-0.06736414011880687, -0.03269903465484411]`. All three intervals
support the same harm verdict. All eight scorer
cells and all four task families decline. The worst cell is ProcessBench GSM8K/Qwen-3 8B at
`-0.08682706890611058`, and the worst family is GSM8K at
`-0.07217595815927216`. The exact-error component falls by
`-0.05419328178405336`, interval
`[-0.07044855897949436, -0.038169638241617065]`; clean abstention changes by
`-0.00610838377851286`, interval
`[-0.015591295279493328, 0.00312137003672684]`, so that component remains
directionally inconclusive rather than being called zero. There are 2,280
exact row-level prediction flips.

The primary interval is wholly below the preregistered practical-harm bound
`-0.005`, and the worst-cell delta breaches the hard bound `-0.030`. Thus the
result is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`; the directional
harm claim is independently supported by the paired interval. This is a
scientific robustness failure, not an execution or provenance malfunction:
the score freeze, label firewall, threshold hash, fold alias, population, and
bootstrap gates all pass. The max arm is closed, while the separately
registered mean and other Stage-A arms remain eligible after this checkpoint
is discussed.

The value differs slightly from Phase-1 `R0_ENTROPY_MAX` (`0.2957374`) because
R0 fitted its own cross-fold decision thresholds. Phase 2R deliberately holds
the top-five thresholds fixed, so `-0.0498781` is the reducer-plus-induced-score
effect at one common operating point. Neither number is directly compared to
the historical `0.3662` as a one-factor effect; that value remains the Phase-0
anchor under a different historical regime.

##### Executed Stage-A checkpoint: `P2R_A_MEAN_ALL`

The full-step mean was frozen and executed as the next independent Stage-A
arm. It replaces only the top-five tail mean with the arithmetic mean of every
token risk in each step. It uses the same signed candidate runner, score
population, spans, folds, detector construction, ten top-five thresholds,
length cutpoints, and 20,000 bootstrap draws as the max control and reference.
No label or candidate score selected a new operating point.

Its frozen-threshold ProcessBench macro F1 is `0.2731562609201911`, marginal
interval `[0.2542193527657983, 0.29067711294773024]`. The paired delta versus
the top-five reference is `-0.07207047896041885`; after top-k=2 opened, the
simultaneous interval across the three reducer contrasts is
`[-0.09364073578340677, -0.05103358047601641]`. Cell W/T/L is `0/0/8`,
family W/T/L is `0/0/4`, the worst cell is GSM8K/Qwen-3 8B at
`-0.09884542817835773`, and the worst family is Math at
`-0.0869622178885095`. Exact-error drops `-0.07514858953719705`, interval
`[-0.09384727489580041, -0.05636562172427741]`, and within-one drops
`-0.05010252938293458`, interval
`[-0.07057214573277983, -0.03001802090982172]`. Clean abstention has a small
positive point delta `+0.0006039548990113275`, interval
`[-0.010528173321108353, 0.011722121859102305]`; it is inconclusive and does
not offset the supported localization loss. There are 2,838 exact prediction
flips.

The length panel is descriptive but diagnostic: mean pooling scores
short/medium/long strata `0.28075/0.26048/0.28690`, compared with
`0.24493/0.34457/0.44825` for top-five. Thus the full mean does not merely
shift one common operating threshold; it strongly dilutes the medium and long
localized signal while its short-step slice is descriptively higher. The
candidate selects substantially shorter steps on average (`80.82` tokens
versus `123.89` for the reference). These opened-slice observations may
motivate the already-preregistered length-normalized top-fraction controls,
but they cannot select a new reducer or promote the failed mean arm.

The primary interval is wholly below the practical-harm boundary and the
worst-cell value breaches the hard bound. Status is therefore `HARD_FAIL /
REJECTED / DEVELOPMENT / HARD_FAILURE`, with a separate supported-harm claim.
All technical/provenance gates pass. The mean arm is closed; `P2R_A_TOPK2`
remains the next unopened registered row.

##### Executed Stage-A checkpoint: `P2R_A_TOPK2`

The top-two tail mean was frozen and executed next. It changes only the number
of retained high-risk tokens from five to two. Entropy token scores, step
spans, empirical-rank answer-detector combination, all 6,800 scorer rows,
source-group folds, ten top-five thresholds, length cutpoints, and the
20,000-draw stream remain identical. This directly tests whether two locally
extreme tokens carry enough evidence; it is not a new feature or detector
experiment.

Its frozen-threshold ProcessBench macro F1 is `0.3206520589072424`, marginal
interval `[0.30264194740120864, 0.33737661546824227]`. The delta versus
top-five is `-0.024574680973367513`; after top-three opened, the four-contrast
simultaneous interval is `[-0.03825894245936002, -0.011261919293666715]`.
All eight scorer cells and
all four task families decline. The worst cell is GSM8K/Qwen-3 8B at
`-0.062465615815852915`, and the worst family is GSM8K at
`-0.054034541952463655`. Exact-error localization falls by
`-0.031913723833081886`, interval
`[-0.043930622865086734, -0.02004342983147544]`. Within-one changes by
`-0.00787476931165676`, interval
`[-0.019917749064226824, 0.004337713300272927]`, and remains inconclusive.
Clean abstention improves `+0.007619551589805429`, interval
`[0.0008681924064067067, 0.014649758533087375]`. Thus top-two makes a real
operating-behavior trade: it abstains more successfully on clean traces but
loses substantially more exact error locations, yielding supported net harm.
There are 1,373 exact prediction flips.

The length panel is consistent with evidence accumulating across several
tokens. Top-two short/medium/long F1 is
`0.24322/0.31736/0.41186`, compared with
`0.24493/0.34457/0.44825` for top-five. It nearly matches the short slice but
falls increasingly behind on medium and long steps. Selected steps average
`118.24` tokens, closer to top-five's `123.89` than max's `111.54`; this is a
descriptive diagnostic, not a post-hoc selection rule.

The primary interval is wholly below the practical-harm boundary, and the
worst-cell delta breaches the hard bound. Status is `HARD_FAIL / REJECTED /
DEVELOPMENT / HARD_FAILURE`, with a separate supported-harm claim. All
technical/provenance gates pass. The top-two arm is closed;
at that checkpoint `P2R_A_TOPK3` was the next unopened row and was subsequently
opened in the checkpoint below.

##### Executed Stage-A checkpoint: `P2R_A_TOPK3`

The top-three tail mean was frozen and executed next. It changes only the
number of retained high-risk tokens from five to three. The token-risk curve,
step spans, detector, all 6,800 rows / 3,400 source groups, folds, ten top-five
thresholds, length cutpoints, and 20,000 paired draws remain unchanged. No
candidate threshold was fit.

Its frozen-threshold ProcessBench macro F1 is `0.33338386192086655`, marginal
interval `[0.31516152241228323, 0.3502586859228561]`. Delta versus top-five is
`-0.01184287795974337`; the four-contrast simultaneous interval is
`[-0.021739354150665868, -0.002250461455371104]`. Cell W/T/L is `1/0/7`,
family W/T/L is `0/0/4`, worst cell is GSM8K/Qwen-3 8B at
`-0.025059218319582355`, and worst family is GSM8K at
`-0.018637637998141332`. The worst-cell hard bound passes, though the stricter
promotion robustness gate does not. There are 821 exact prediction flips.

Exact-error localization changes `-0.01603603732295439`, interval
`[-0.024847856774751365, -0.0074578810504534795]`. Clean abstention improves
`+0.006893048868810148`, interval
`[0.001701863811718038, 0.012402992925695206]`. Within-one changes
`-0.004559520637761771`, interval
`[-0.013327265911688879, 0.004172042864985972]`, and is inconclusive. Thus
top-three reduces clean false alarms but loses exact first-error locations.

Short/medium/long descriptive F1 is `0.23978/0.33697/0.43092`, versus
`0.24493/0.34457/0.44825` for top-five. The gap grows with step length, but is
smaller than for top-two. Selected steps average `121.06` tokens versus
`123.89` for the reference. This remains a diagnostic, not a tuning rule.

The simultaneous interval is below zero but crosses the preregistered
`-0.005` practical-harm boundary. It therefore supports neither improvement
nor practical harm at the planned confidence level. Status is `COMPLETE /
NO_PROMOTION / DEVELOPMENT / INCONCLUSIVE`; the arm is not rejected and did
not hard-fail. At that checkpoint `P2R_A_TOPK8` was the next unopened row.

##### Executed Stage-A checkpoint: `P2R_A_TOPK8`

The top-eight tail mean was frozen and executed, changing only the tail width
from five to eight. Token-risk scores, detector, all 6,800 rows / 3,400 source
groups, spans, folds, ten reference thresholds, length cutpoints, and 20,000
paired draws remain fixed. No candidate-specific threshold was fit.

Macro F1 is `0.3544258115519002`, marginal interval
`[0.3361090230798102, 0.37143626059360485]`. Delta versus top-five is
`+0.009199071671290304`; the five-contrast simultaneous interval is
`[+0.0006298228035607323, +0.018131629210238576]`. Cell W/T/L is `7/0/1`,
family W/T/L is `3/0/1`, worst cell is Math/Qwen-3 8B at
`-0.0030603574140507517`, and worst family is Math at
`-0.0010530788600555396`. There are 669 prediction flips.

Exact-error improves `+0.013202789698382678`, interval
`[+0.0053996581480495595, +0.02137803009481023]`; within-one is
`+0.005709488048500133`, interval
`[-0.0030949671200639697, +0.014537955752043601]`; clean abstention is
`-0.002705749459947504`, interval
`[-0.008130195613053412, +0.0025370475328909577]`. The primary gain is
therefore associated with more exact localizations without supported
clean-trace harm.

Short/medium/long descriptive F1 is `0.24749/0.34906/0.46003`, versus
`0.24493/0.34457/0.44825` for top-five. The advantage grows with step length,
consistent with broader evidence accumulation. Selected steps average `125.84`
tokens versus `123.89` for the reference. These slices are not a tuning rule.

The simultaneous interval is above zero but its lower bound does not exceed
the preregistered `+0.005` minimal practical benefit. Status is `COMPLETE /
NO_PROMOTION / DEVELOPMENT / INCONCLUSIVE`: a positive directional difference
is supported, but the registered practical-improvement claim is not. Top-eight
is retained for fresh confirmation, not promoted. At that checkpoint
`P2R_A_TOPK10` was next; after it opened, top-eight's six-contrast interval
became `[+0.0004300141124132161, +0.018332325483982944]` without changing the
verdict.

##### Executed Stage-A checkpoint: `P2R_A_TOPK10`

Top-ten was frozen only after the top-eight checkpoint completed. It changes
only the retained tail width from five to ten. Token scores, detector, all
6,800 rows / 3,400 source groups, spans, folds, ten reference thresholds,
length cutpoints, and 20,000 paired draws remain fixed. No rethresholding
occurred.

Macro F1 is `0.3581627690347784`, marginal interval
`[0.33988094512220596, 0.3750167586992808]`. Delta versus top-five is
`+0.012936029154168471`; the six-contrast simultaneous interval is
`[+0.0022677722876254937, +0.02434841872258895]`. Cell W/T/L is `6/0/2`,
family W/T/L is `3/0/1`, worst cell is Math/Qwen-3 4B at
`-0.0017168298967323303`, and worst family is Math at
`-0.0005954226189631429`. There are 911 prediction flips.

Exact-error improves `+0.01627408953265691`, interval
`[+0.006296861210834302, +0.026509252025967427]`; within-one is
`+0.008432886717600285`, interval
`[-0.0017202310091515903, +0.01855400873202672]`; clean abstention is
`-0.0036403551859929273`, interval
`[-0.009369707404457644, +0.0018859412300232426]`. The primary gain is again
driven by exact localization without supported clean-trace harm.

Short/medium/long descriptive F1 is `0.24426/0.35821/0.46407`, versus
`0.24493/0.34457/0.44825` for top-five. Top-ten is flat on short steps and
improves medium/long steps more than top-eight. Selected steps average `127.11`
tokens versus `123.89` for the reference. These slices remain diagnostic.

The simultaneous interval is above zero but its lower bound is below the
preregistered `+0.005` minimal practical benefit. Status is `COMPLETE /
NO_PROMOTION / DEVELOPMENT / INCONCLUSIVE`: top-ten is the raw best and has a
supported directional gain, but not a supported practical-improvement claim.
It requires fresh confirmation. At that checkpoint `P2R_A_TOPQ25` was next.

##### Executed Stage-A checkpoint: `P2R_A_TOPQ25`

The first length-normalized control was frozen and executed next. It averages
the largest `ceil(0.25 |I_s|)` token risks in each step. Token scores, detector,
all 6,800 rows / 3,400 groups, spans, folds, ten top-five thresholds, length
cutpoints, and 20,000 paired draws remain fixed. No rethresholding occurred.

Macro F1 is `0.2777731912009317`, marginal interval
`[0.2585103116168444, 0.29542584859839494]`. Delta versus top-five is
`-0.06745354867967823`; the seven-contrast simultaneous interval is
`[-0.09007822431644956, -0.04577551185266464]`. Cell W/T/L is `0/0/8`,
family W/T/L is `0/0/4`, worst cell is GSM8K/Qwen-3 4B at
`-0.09295958736931953`, and worst family is GSM8K at
`-0.08779840176697665`. There are 2,491 prediction flips.

Exact-error falls `-0.07187892208940688`, interval
`[-0.08946128262211377, -0.05463661100053455]`; within-one falls
`-0.04755285982675317`, interval
`[-0.06661523955324929, -0.02874396079829202]`. Clean abstention is
`+0.0037060040587443277`, interval
`[-0.0070678798121362124, +0.014408526915645758]`, and is inconclusive.

Short/medium/long descriptive F1 is `0.26861/0.28183/0.30004`, versus
`0.24493/0.34457/0.44825` for top-five. The short slice rises descriptively,
but medium and long localization collapse. Selected steps average `84.90`
tokens versus `123.89` for the reference. A fixed fraction therefore does not
solve fixed-K length sensitivity: in long spans it admits many weak tokens,
dilutes localized evidence, and changes the winning step.

Status is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`: the primary
interval is wholly below the practical-harm bound and the worst cell breaches
`-0.030`. Top-quarter is closed. At that checkpoint `P2R_A_TOPQ50` was next.

##### Executed Stage-A checkpoint: `P2R_A_TOPQ50`

The second and final fixed-fraction tail control averages the largest
`ceil(0.50 |I_s|)` token risks. Token scores, detector, all 6,800 rows / 3,400
groups, spans, folds, ten top-five thresholds, length cutpoints, and 20,000
paired draws remain fixed. No rethresholding occurred.

Macro F1 is `0.2711074781371915`, marginal interval
`[0.25193681934575884, 0.2888037215609885]`. Delta versus top-five is
`-0.07411926174341843`; the eight-contrast simultaneous interval is
`[-0.09859538496309983, -0.05016154717906007]`. Cell W/T/L is `0/0/8`,
family W/T/L is `0/0/4`, worst cell is GSM8K/Qwen-3 8B at
`-0.10197748134215262`, and worst family is GSM8K at
`-0.08595502954631112`. There are 2,774 prediction flips.

Exact-error falls `-0.0770579072312283`, interval
`[-0.09552616066238696, -0.058309766022541565]`; within-one falls
`-0.05125547279234055`, interval
`[-0.0713467215784813, -0.03161236466282118]`. Clean abstention is
`+0.0019202361277120827`, interval
`[-0.009116598209305327, +0.012934710519692819]`, and is inconclusive.

Short/medium/long descriptive F1 is `0.27291/0.26229/0.28800`, versus
`0.24493/0.34457/0.44825` for top-five. The descriptive short-slice rise is
overwhelmed by medium/long collapse. Selected steps average `81.77` tokens
versus `123.89`; increasing the retained fraction worsens the same dilution
mechanism observed for top-quarter.

Status is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`: supported harm
and a worst-cell hard-bound breach. The registered fixed-fraction subbranch is
closed as negative evidence. At that checkpoint `P2R_A_QUANTILE75` was next.

##### Executed Stage-A checkpoint: `P2R_A_QUANTILE75`

The single 0.75 empirical quantile of each step's token risks was frozen and
executed next. It changes only the reducer; token scores, detector, all 6,800
rows / 3,400 groups, spans, folds, ten top-five thresholds, length cutpoints,
and 20,000 paired draws remain fixed. No rethresholding occurred.

Macro F1 is `0.27027639253640545`, marginal interval
`[0.25093578455446003, 0.28766160540409935]`. Delta versus top-five is
`-0.07495034734420447`; the nine-contrast simultaneous interval is
`[-0.10125692313621162, -0.05012743474485039]`. Cell W/T/L is `0/0/8`,
family W/T/L is `0/0/4`, worst cell is GSM8K/Qwen-3 4B at
`-0.10491563750863309`, and worst family is GSM8K at
`-0.10344655942539285`. There are 2,973 prediction flips.

Exact-error falls `-0.07870161358370925`, interval
`[-0.09829177070014146, -0.05920390278066903]`; within-one falls
`-0.05278938555676693`, interval
`[-0.07328878238878991, -0.03250331328441689]`. Clean abstention is
`-0.004382330346056729`, interval
`[-0.016114744801413866, +0.007268971913130232]`, and is inconclusive.

Short/medium/long descriptive F1 is `0.27654/0.25830/0.28096`, versus
`0.24493/0.34457/0.44825` for top-five. The descriptive short-slice gain is
overwhelmed by medium/long collapse; selected steps average `81.85` tokens
versus `123.89`. A single interior quantile does not aggregate upper-tail mass.

Status is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`: supported harm
and a worst-cell breach. At that checkpoint `P2R_A_QUANTILE90` was next.

##### Executed Stage-A checkpoint: `P2R_A_QUANTILE90`

The single empirical 0.90 quantile was frozen only after the 0.75 checkpoint.
It changes only the reducer; token scores, detector, all 6,800 rows / 3,400
groups, spans, folds, ten reference thresholds, length cutpoints, and 20,000
paired draws remain fixed. No rethresholding occurred.

Macro F1 is `0.2759396801681213`, marginal interval
`[0.25736058961003183, 0.29305102694647384]`. Delta versus top-five is
`-0.06928705971248861`; the ten-contrast simultaneous interval is
`[-0.09216456414588237, -0.04777617901623373]`. Cell W/T/L is `0/0/8`,
family W/T/L is `0/0/4`, worst cell is Math/Qwen-3 8B at
`-0.09236129048604769`, and worst family is Math at
`-0.08937895201763082`. There are 2,592 prediction flips.

Exact-error falls `-0.0723397035761372`, interval
`[-0.08935278310157803, -0.05552275069793505]`; within-one falls
`-0.04801650657731693`, interval
`[-0.06607455089523387, -0.03026228011384097]`. Clean abstention is
`-0.0028146494553622503`, interval
`[-0.013496437841156636, +0.007869803068842942]`, and is inconclusive.

Short/medium/long descriptive F1 is `0.27244/0.27838/0.29552`, versus
`0.24493/0.34457/0.44825` for top-five. Moving from quantile 0.75 to 0.90
recovers some medium/long performance but remains far below top-five. Selected
steps average `85.44` tokens versus `123.89`. One high quantile still cannot
replace aggregation across several upper-tail observations.

Status is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`: supported harm
and a worst-cell breach. Both registered single-quantile controls are closed as
negative evidence.

##### Executed Stage-A checkpoint: `P2R_A_MEDIAN`

The final originally preregistered Stage-A control replaced top-five mean with
the empirical median while keeping the same token scores, detector, 6,800
rows / 3,400 groups, spans, folds, ten reference thresholds, length cutpoints,
and 20,000 paired draws. No rethresholding occurred.

Macro F1 is `0.26898093491578023`, marginal interval
`[0.25002349332047513, 0.28651028486264585]`. Delta versus top-five is
`-0.07624580496482969`; the closed eleven-contrast simultaneous interval is
`[-0.1038968516280433, -0.04856378253714272]`. Cell W/T/L is `0/0/8`, family
W/T/L is `0/0/4`, worst cell is GSM8K/Qwen-3 8B at
`-0.12185761302296211`, and worst family is GSM8K at
`-0.08874923642377605`. There are 3,226 prediction flips.

Status is `HARD_FAIL / REJECTED / DEVELOPMENT / HARD_FAILURE`. The median
confirms that the useful localization evidence is not a full-step level shift;
it is concentrated in an upper tail that must be aggregated without diluting
it across the span.

##### Post-hoc amendment: upper-tail fraction means

After the original Stage-A family closed, the user requested two bounded
mechanism diagnostics that are mathematically different from the failed single
quantiles:

- `P2R_A_TOPQ10_EXPLORATORY` averages the largest
  `max(1, ceil(0.10 |I_s|))` token risks.
- `P2R_A_TOPQ05_EXPLORATORY` averages the largest
  `max(1, ceil(0.05 |I_s|))` token risks.

They run sequentially with the same frozen scores, thresholds, rows, folds,
spans, and bootstrap groups. Because they were proposed after ProcessBench
reducer outcomes opened, they form a separate two-contrast descriptive
Bonferroni family. Neither is promotion-eligible on this population regardless
of point estimate or interval; fresh confirmation is mandatory. They diagnose
whether the successful fixed top-eight/top-ten behavior reflects an average
over a small length-normalized upper tail rather than a single order statistic.

`P2R_A_TOPQ10_EXPLORATORY` was frozen and executed first. Macro F1 is
`0.2755016413673608`, marginal interval
`[0.2566306287704972, 0.29285481534177266]`; delta versus top-five is
`-0.06972509851324915`, with descriptive interval
`[-0.0845742047637569, -0.055774484288490314]`. All eight cells and all four
families lose, and the worst cell falls `-0.09207400382778269`. Exact-error
falls `-0.07577709277209377`, while clean abstention rises
`+0.011813875130009865`.

Short/medium/long F1 is `0.24557/0.28712/0.31298`, versus
`0.24493/0.34457/0.44825` for top-five. The candidate nearly preserves the
short stratum but progressively dilutes medium and long steps because its tail
cardinality grows with span length. Status is
`HARD_FAIL / NO_PROMOTION / DEVELOPMENT / DESCRIPTIVE`; the statistical label
remains descriptive by amendment contract even though the observed loss is
directionally clear. `P2R_A_TOPQ05_EXPLORATORY` remains next.

`P2R_A_TOPQ05_EXPLORATORY` was then frozen and executed. It averages the
largest `max(1, ceil(0.05 |I_s|))` risks: across the 51,394 scorer-step records
the effective cardinality has median 4, mean 4.74, interquartile range 3--6,
and 90th percentile 8. Macro F1 is `0.2739996862119836`, marginal interval
`[0.25515347267752125, 0.2914824578690824]`; delta versus top-five is
`-0.0712270536686263`. With both post-hoc arms open, its separate descriptive
Bonferroni interval is `[-0.08814277154091034, -0.05540668829243298]`.

All eight cells and all four families lose; worst cell and family deltas are
`-0.0878825819163705` and `-0.08787707306359674`. Exact-error falls
`-0.0775673013101481`, within-one falls `-0.04785848796899245`, while clean
abstention rises `+0.011956791634416208`. Short/medium/long F1 is
`0.24250/0.26911/0.31972`. Status is
`HARD_FAIL / NO_PROMOTION / DEVELOPMENT / DESCRIPTIVE`.

The completed aggregation study therefore favors fixed cardinality, not an
interior order statistic or a length-proportional tail. `P2R_A_TOPK10` is the
raw best at F1 `0.3581627690347784`, delta `+0.012936029154168471`, with the
closed eleven-contrast interval
`[+0.0014438677363772625, +0.02522463975302794]`. It passes the point-benefit,
cell-count, worst-cell, exact-error, and clean-abstention gates, but its lower
bound does not exceed the preregistered `+0.005` practical-benefit threshold.
Thus it is a selection-opened development parent, not a promoted improvement;
fresh confirmation remains mandatory and future arms must compare against
both top-five and this strongest atomic reducer parent.

### Phase 2C — conditional contribution ablation before fusion

Before any Phase-2R-B transform or Phase-3 survivor-only fusion opens, run the
bounded family6 conditional-contribution amendment specified in
`docs/experiments/REASONING_LOCALIZATION_03662_CONDITIONAL_ABLATION_V1.md`.
The executable correction established that current R2 averages five
non-structural local families; `structural` is retained with zero local weight.
The frozen roster therefore contains one exact five-family/top-ten parent,
five true family leave-one-outs, a structural insertion control, four targeted
within-family view leave-one-outs, one exact C1-SWVar formulation swap, C7 in
`entropy_dynamics`, and C8 as a separate outer expert.

This branch corrects an important interpretation boundary: rejection of an
equal-weight atomic formulation does not prove that its underlying signal has
no conditional value in a multi-family fusion. Later eligibility therefore
has two bounded routes: an atomic survivor or a multiplicity-supported
conditional contribution. Conditional-only eligibility preserves only the
exact family placement supported by the ablation and does not authorize an
unbounded feature or weight search.

The amendment was requested after some atomic outputs existed. Its full roster
is consequently structural and development-only, not retroactively
preregistered from unseen outcomes. It must be frozen in executable registries
before its own first score opens, and any universal conclusion requires fresh
confirmation.

The removal substage is complete. Entropy level and top-k distribution have
supported positive aggregate conditional contributions of `+0.024646`
(`[+0.007506,+0.042031]`) and `+0.022669`
(`[+0.004708,+0.040473]`), respectively. Both expose material
exact-error-versus-clean-abstention tradeoffs and fail the complete promotion
gate. Partition energy is promising but unconfirmed; entropy dynamics,
sampled energy, SWVar16, CUSUM, sampled level and partition level remain
inconclusive. No interval crossing zero is interpreted as rejection. The next
unopened state is the structural insertion control.

Phase 2C is now complete. Structural insertion and the exact C1-SWVar swap do
not improve the parent. C7 gives a small uncertain gain. C8 as an equal
donor-rank outer expert is the raw best at F1 `0.364997`, delta `+0.010735`,
with simultaneous interval `[-0.002481,+0.024255]`; exact-error rises
`+0.014429` while clean abstention falls `-0.022491`. It is
`PROMISING_UNCONFIRMED`, not rejected, but it fails the inferential and clean
promotion gates. The closed verdict is `NO_FULL_CONDITIONAL_PROMOTION`.

#### Post-Phase-2C H2/H3 role-separation diagnostic

The subsequently requested combination is complete under a separate,
outcome-selected development contract. H0 preserves its exact clean/error
decision. H2 removes sampled-token energy and the partition-level
`energy_series` view and inserts C7 inside entropy dynamics. H3 equal adds C8
step ranks only on H0 non-abstentions. H3 equal reaches F1 `0.366653`, delta
`+0.012392` with four-contrast simultaneous interval
`[+0.001769,+0.022807]`; exact error and within-one improve, while clean
abstention is identical by construction.

The donor-reliability alternative does not improve equal fusion and learns
weights close to 0.5. H3 equal is the frozen priority for fresh-question
confirmation, but it is not a Phase-3 survivor: its lower interval bound does
not exceed the existing `+0.003` practical-benefit threshold and its roster was
selected after the component outcomes opened. The raw `0.366653` is not a
direct comparison with historical `0.3662` because their populations, splits,
and detector contracts differ. Full method and evidence boundary:
`REASONING_LOCALIZATION_03662_H3_RELIABILITY_V1.md`.

#### Frozen Llama scorer-family transfer

The sealed local inventory contains no new ProcessBench questions: Qwen3-4B,
Qwen3-8B and Llama-3.1-8B cover the exact same 3,400 source groups. The first
follow-up therefore tested scorer-family transfer only. It reconstructed the
frozen Qwen H0/H2/H3 scores exactly before importing Llama labels, then reused
H0's clean/error decisions for both candidate rerankers.

On the four Llama cells, H0 reaches F1 `0.348909`, H2 reaches `0.355583`, and
H3 reaches `0.353281`. The frozen simultaneous deltas are H2−H0
`+0.006674 [-0.007091,+0.020943]`, H3−H0
`+0.004372 [-0.009677,+0.018452]`, and H3−H2
`-0.002303 [-0.011662,+0.007001]`. Both candidates are
`PROMISING_UNCONFIRMED`, not rejected. H2 is raw-best, while H3's secondary
within-one gain does not establish primary incremental value over H2.

Consequently, a future fresh-question confirmation must retain the full
H0→H2→H3 ladder. H3 must beat both the original H0 reference and its H2 parent
before any Phase-3 fusion eligibility. Frozen contract:
`REASONING_LOCALIZATION_03662_H3_LLAMA_TRANSFER_V1.md`; result note:
`REASONING_LOCALIZATION_03662_H3_LLAMA_TRANSFER_RESULTS_V1.md`.

#### Frozen PRMBench H2/H3 mechanism diagnostic

Because no local fresh ProcessBench source questions exist, the next bounded
study transferred the already frozen H0→H2→H3 scores to PRMBench every-step
ranking without PRMBench tuning. Its first executable attempt hard-failed
before labels because it incorrectly compared top-ten H0 to the Phase-1
top-five R2 artifact. Amendment V2 introduced a non-rankable top-five control,
which reproduced R2 exactly; all Qwen H0/H2/H3 source scores also reproduced
exactly before label import.

On 83,280 steps in 6,208 paired source groups, H0/H2/H3 AUROC is respectively
`0.592057`, `0.597871`, and `0.619469`. H3−H0 is
`+0.027412 [+0.023675,+0.031091]`; H3−H2 is
`+0.021598 [+0.017653,+0.025457]`; H2−H0 is
`+0.005814 [+0.004710,+0.006973]`. The intervals are
Bonferroni-simultaneous within the three frozen AUROC contrasts. H3 also
improves AUPRC against both parents and improves AUROC in all eight evaluable
families. The ninth family, `multi_solutions`, is single-class and remains
undefined rather than zero-filled.

This is a supported `PRMBENCH_SPECIALIST` result, not a universal winner or a
Phase-4 promotion. ProcessBench first-error value remains unconfirmed; the
PRMBench labels were historically opened, the H2/H3 ancestry was
outcome-selected, and source-stratum membership is unavailable. Detailed
result note:
`REASONING_LOCALIZATION_03662_H3_PRMBENCH_DIAGNOSTIC_RESULTS_V2.md`.

#### Matched historical-regime H3 head-to-head

The direct bridge requested before further fusion is complete. The executable
first reproduced the exact Stage-4 entropy and finalist scores at
`0.3614213584` and `0.3662328342`, then froze H0/H2/H3 locators before opening
the same 1,270-row, 635-group historical audit. H0/H2/H3 reach F1
`0.374099`, `0.374793`, and `0.372663` respectively. H3 minus historical is
`+0.006431 [-0.026891,+0.039473]`; H2 minus historical is
`+0.008560 [-0.024610,+0.040869]`. H3 is numerically above 0.3662, but the
improvement claim is unresolved rather than supported or rejected.

The 2x2 mechanism cross finds no supported isolated H3-localizer or
interaction effect. Historical-detector plus H3-localizer changes F1 by
`-0.003070 [-0.028453,+0.020618]` relative to the historical localizer;
H0-detector plus historical-localizer changes it by
`+0.013119 [-0.005697,+0.034240]`; interaction is
`-0.003619 [-0.012159,+0.004363]`. The favorable end-to-end point estimate is
therefore descriptively associated with the current detector's improved clean
abstention, not with a demonstrated H3-localizer gain.

H3 retains its separate supported PRMBench advantage but does not satisfy the
preregistered dual-task promotion condition. H2 remains the ProcessBench
raw-best candidate and H3 the PRMBench-enhanced candidate; both require an
independent confirmation roster. Frozen contract:
`REASONING_LOCALIZATION_03662_H3_HISTORICAL_HEADTOHEAD_V1.md`; result note:
`REASONING_LOCALIZATION_03662_H3_HISTORICAL_HEADTOHEAD_RESULTS_V1.md`.

### Phase 3 — principled fusion and selection

Only Phase 2 survivors may enter fusion. Freeze at most three evidence blocks and at most twelve scalar coordinates unless an explicit dimension amendment is approved.

Fusion order:

1. Equal provenance-family average.
2. Ordinary label-blind IU covariance fusion.
3. Exact deployed U-PCR weak-expert exclusion plus survivor refit on an
   eligible compact member-view pool, with a matched full-pool IU parent.
4. One conditional mechanism only if its premise audit passes: self-basis innovation for residual complementarity, or B3 weighting for reliability heterogeneity.

The third rung is not an alias of ordinary IU-PCR.  Deployed U-PCR estimates
fit-side `rho_hat`, removes views below frozen relative-reliability thresholds,
and recomputes its spectral weights on the survivors.  It is dimension-
ineligible on the four outer H2 family scores because the exact project policy
falls back to a simple average below five experts.  The registered P3D test
therefore uses the compact H2 member-view pool, a same-matrix full-pool IU
control, a mask-only equal control, and a cardinality-matched random-mask
control.  Full details are in
`REASONING_LOCALIZATION_03662_PHASE3_DEPLOYED_UPCR_PRUNE_REFIT_V1.md`.

The completed P3D ladder does not promote.  On the common eight-Qwen panel,
P3D0 full-pool IU scores `0.354240`, P3D1 deployed U-PCR `0.356740`, the
rho-mask equal control `0.353551`, and the predeclared mean over twenty random
masks `0.354007`, versus `0.364090` for equal-family H2.  P3D1 improves its
same-matrix P3D0 parent by `+0.002499 [-0.008683,+0.013781]`, which is
`PROMISING_UNCONFIRMED`; against H2 it is
`-0.007350 [-0.017885,+0.003263]`.  Five-fold masks are highly stable (minimum
cell mean Jaccard `0.9571`) and no fallback occurs, but the equal-mask and
random-mask controls do not support a useful pruning mechanism.  Thus deployed
U-PCR remains a bounded idea for independently surviving methods, not the next
default fusion parent and not a PRMBench-transfer candidate from this run.

SU-PCR, DUFS, graph hierarchy, and transform clustering are not default
escalation paths. The corrected final-answer `STG_SU_STABLE` result provides a
narrow fold-stable sparse-support premise, but not supported superiority or
localization evidence. It is registered only as the survivor-gated
feature/temporal graph branch in
`REASONING_LOCALIZATION_03662_STG_GRAPH_TRANSFER_V1.md`, with exact-parent and
random/permutation controls. A method may be rerun only when the new survivor
set establishes the exact premise the older experiment lacked.

The subsequent development-only STEP-CUT audit failed the within-answer graph
premise: the full graph lost to chain-only, step-permuted, and random-edge
controls, and entropy-plus-graph fusion caused CI-supported Hit@1 harm on both
Qwen and Llama panels. Consequently `P3G_T1_TEMPORAL_GRAPH` and any combined
feature-by-time graph are `NOT_RUN_BY_GATE`. This does not close the separate
STG feature/family-support arm, and it does not authorize the diagnostically
interesting chain-only score without a new frozen premise protocol.

Phase 3 uses the same ProcessBench promotion gate as Phase 2 and must also beat the best atomic parent with a multiplicity-valid paired interval lower bound above the preregistered +0.003 parent-benefit delta. This prevents a complex fusion from being promoted merely for matching a compact source.

#### Historical unexecuted trajectory-feature tensor registration

`P3_TRAJECTORY_TENSOR` is a bounded later-stage branch. It does not alter or
delay Phase 1 and cannot open until Phase 2 has frozen both a compact atomic
signal roster and the reducer-study parent. Its retrospective motivation is a
task conflict, not a positive method claim: the CIW cross-scale correction
raised PRMBench step AUROC by about `+0.0013` in several already-opened
response-head diagnostics while ProcessBench macro F1 fell by roughly
`0.0006` to `0.0009`. This does **not** establish that changing operation order
helps PRMBench. Exact evidence is
`docs/experiments/CIW_CROSS_SCALE_LOCALIZATION_V1.md:34-64`,
`results/ciw_cross_scale_localization_v1/REPORT.md:5-18`, and
`PROGRESS.md:435-455`; the evidence grade is retrospective.

The scientific question is whether applying one frozen temporal encoder before
feature fusion, followed by a compact response-by-time-by-feature projection,
can improve ProcessBench first-error localization or PRMBench every-step
ranking. The task estimands remain separate and are never averaged.

The fit object is

```text
X[i,t,f]
  i = response/source group
  t = token index or one preregistered fixed time bin, with an explicit mask
  f = one Phase-2-surviving primitive token feature
```

Variable-length traces are right-padded only inside batches; padded values are
masked out of every mean, covariance, factor, projection, and score. The
execution registry must freeze token versus bin coordinates, bin boundaries,
padding value, mask semantics, minimum observed length, and empty-bin rule.
All scorer copies of a source question remain in one fold. Fit-side
standardization, temporal templates, DSP parameters, cross-response
statistics, factor/weight selection, orientation, and sign tie-breaks use
donor/calibration responses only. Held responses are projection-only.
ProcessBench and PRMBench labels are evaluation-only and cannot select a
transform, rank, weight, reducer, or orientation. A future early-detection
subarm must be prefix-only: full-trace CUSUM, future-token pooling, total-answer
relative time, and any statistic of an unseen suffix are forbidden.

The compact ordered ladder is:

1. `P3T_T0_FROZEN_PARENT`: exact alias of the surviving token-fusion curve and
   frozen Phase-2 reducer. It must be byte-identical to its parent.
2. `P3T_T1_DSP_FIRST`: one fixed DSP-first trajectory transform before the
   unchanged feature fusion. No filter bank or label-chosen frequency is
   allowed.
3. `P3T_T2_CAUSAL_TEMPORAL`: one predeclared causal time-series alternative,
   instantiated only if its matching atomic temporal premise survives. Its
   parameters are derived from prelabel donor data.
4. `P3T_T3_TWO_AXIS_LOWRANK`: one donor-trained structured low-rank projection
   over feature and temporal coordinates. It is additive/projection-only and
   has an exact zero-strength alias of T0. It may not infer a factor and then
   hard-deflate it: CM-LFF third-moment nuisance deflation was a demonstrated
   negative control, not a reusable premise (`HISTORY.md:11918-11939` and
   `results/coupled_moment_kfactor_24cell_v1/INDEPENDENT_REVIEW.md:44-70`).

Arbitrary DSP-by-filter-by-fusion crosses are forbidden. T1 and T2 each beat
their exact parent before T3 may use them. Every candidate must beat both its
exact parent and the frozen Phase-1 reference on ProcessBench with the existing
paired grouped-bootstrap, worst-cell, exact-error, clean-abstention, and
multiplicity gates. Only a ProcessBench survivor transfers to PRMBench. A
ProcessBench gain with PRMBench harm is `PROCESSBENCH_SPECIALIST`; the reverse
is `PRMBENCH_SPECIALIST`; neither is hidden by an aggregate.

The frozen causal chain is:

```text
primitive token streams
  -> within-trace DSP / temporal encoder
  -> donor-only cross-response feature x time projection
  -> token-risk curve
  -> frozen Phase-2 step reducer
  -> ProcessBench detector/localizer

the same frozen token/step score -> PRMBench's separate ranking evaluator
```

Negative controls are mandatory: T0 exact alias; time-coordinate permutation
within donor masks; feature-coordinate permutation before the structured
projection; zero-strength T3; and the forbidden CM-LFF-style hard-deflation
arm recorded as `NOT_RUN_BY_GATE`, not silently omitted. The tensor branch
inherits the reducer-study result as its parent and cannot choose top-k or a
temporal transform after either task's labels are opened.

#### ASTGI-inspired task-query amendment — Q1 complete, later rungs closed

The trajectory branch now contains a bounded `P3_ASTGI_QUERY_HEADS` subladder,
defined fully in
`docs/experiments/REASONING_LOCALIZATION_03662_ASTGI_QUERY_HEADS_V1.md`.
It is an ASTGI-inspired adaptation, not a paper-exact reproduction, and the
paper citation/component-fidelity map remains a prerequisite for any
paper-derived claim.

The amendment freezes three distinct repository parents: H0/D0 for response
detection, H2/O0 for ProcessBench onset, and H3/S0 for PRMBench state-error
ranking. Q1 adds query-conditioned point pooling without a graph; Q2 may add
one donor-learned coordinate system; Q3 may add one causal neighborhood only
after Q2; Q4 may add exactly one propagation layer only after Q3. Each rung
must beat its exact parent and registered null/topology controls. The failed
STEP-CUT conductance graph remains closed and is not a parent or premise.

No cross-task score is defined. A dual-head architecture advances only when
the onset head separately passes its H2 gate, the state head separately passes
its H3 gate, and H0 abstention is unchanged. Hierarchical family fusion, DSP,
and tensor rank are held fixed throughout Q1--Q4 and may enter later only by a
separate one-factor contrast.

#### Historical unexecuted hierarchical family-expert registration

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

The one-family-at-a-time P3E attribution is now complete. A matched donor-
cross-fitted equal H2 parent scores `0.364284`. Replacing only dynamics+C7 by
ordinary IU reaches `0.366876`, delta
`+0.002592 [-0.001839,+0.007194]`, 6/0/2; replacing only top-k reaches
`0.365603`, delta `+0.001319 [-0.001433,+0.004355]`. Both are
`PROMISING_UNCONFIRMED`, not promoted. Partition-only IU is
`-0.004908 [-0.013100,+0.003050]`, and the all-family closure is
`-0.004708 [-0.013846,+0.004211]`. This establishes family heterogeneity as a
useful development observation: a later method-specific variant may target
dynamics first and top-k only as a control, while partition remains equal.

That bounded DUFS follow-up is complete. Dynamics-local DUFS-LIU is only
`+0.000168 [-0.001466,+0.001818]` above its exact IU parent. The requested
context-conditioned arm lets all 24 compact H2 member views define the donor
graph while only dynamics receives output weights; it is
`-0.000006 [-0.001577,+0.001586]` versus the IU parent, does not beat the
family-local graph, and differs from the within-response context-permutation
control by `+0.000020 [-0.000986,+0.001021]`. The single permitted top-k
local-DUFS control is `-0.000065 [-0.001657,+0.001623]` versus top-k IU. These
are inconclusive/tied rather than supported harms, but neither DUFS geometry
nor outside-family context earns promotion. Ordinary IU remains the compact
family-expert default.

#### Registered method-native prune/refit wrapper

The two-pass deployed-U-PCR mechanism may also be tested as a bounded wrapper
around later fusion survivors.  It is not a shared universal definition of
"feature importance": ordinary IU uses additive-model `rho_hat`; SU uses its
sparse-error-corrected estimate; STG uses fold-stable support; DUFS-LIU uses
donor-stable gates; and L-SML/B3 or tensor/query methods are eligible only if
their own label-free reliability quantity and exact no-pruning alias are
frozen first.

Every method receives at most one full-pool/prune-refit pair and must include
the exact unpruned parent, no-pruning alias, cardinality-matched random mask,
and mask-only equal control when defined.  Mask thresholds and member counts
cannot be selected from ProcessBench or PRMBench outcomes.  A method that did
not survive in full-pool form is not reopened automatically by this wrapper.

#### Phase-3 development freeze — Step 346

Phase 3 is closed on the current opened population with verdict
`PHASE3_DEVELOPMENT_CLOSED__NO_PROMOTION`.  Steps 340--345 remain frozen
development evidence; no score, bootstrap draw, or raw result was regenerated
during the audit.  All Phase-3 experiments are `COMPLETE` with no next
variant.  The unexecuted hierarchy/two-block, tensor, residual feature/temporal
STG, and Q2--Q4 templates are `NOT_RUN_BY_GATE / NO_PROMOTION /
NOT_EVALUATED`: they were never evaluated and are not scientific failures.
Phase-4 PRMBench transfer and Phase-5 early detection remain blocked by the
absence of a promoted ProcessBench survivor; already completed P4H diagnostics
are unchanged.

The numerical claim boundary is immutable.  The canonical fair 3,400-row
ProcessBench record is dedicated `family6 + level + step_top5mean`, F1
`0.326141`.  The historical Local/Online finalist `0.3662328342` is a rejected
historical-regime audit anchor; the two values have unmatched protocols and
must not receive a direct delta or rank.  H2 `0.364090` is opened-development
evidence.  P3E0 `0.364284` is a matched donor-cross-fitted equal control, not
an exact H2 alias.  Dynamics-IU `0.366876` remains
`PROMISING_UNCONFIRMED`.  Dynamics DUFS `0.367045` and STG-SU `0.366762` do
not establish their proposed mechanisms.

ASTGI-Q1 scores `0.354584` versus H2 `0.364090`, with primary macro-F1 delta
`-0.009507 [-0.019618,+0.000516]`; that interval yields
`INCONCLUSIVE__NO_PROMOTION`.  Its separately reported exact-error delta is a
degradation, `-0.010644 [-0.020123,-0.001210]`, and must not be erased by the
primary verdict.  Later Phase-3 confidence families were frozen adaptively
after earlier outcomes on the same population.  They are retrospective
development diagnostics, not one joint confirmatory family, and cannot support
promotion across the adaptive ladder.  Any reopening requires a separate
protocol and branch on independently fresh questions or population.

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
