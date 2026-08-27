# Research evidence ledger

This ledger distinguishes results that transferred to the real hallucination
features from mechanisms demonstrated only in synthetic data.  A method is
marked **retained** only when its supported component remains part of the next
experiment; synthetic success alone is never recorded as a real-data win.

## Multi-population benchmark inventory (Step 289; design only)

No result is added by this step. It records where existing and proposed
methods may be compared without mixing prediction units or access contracts.

| design claim | evidence | status | consequence |
|---|---|---|---|
| all collected data can enter one overall leaderboard | response AUROC, first-error F1, prefix AUROC, stopping pass@1/tokens, and RAG unit metrics estimate different targets | **rejected** | use one method registry and separate task leaderboards |
| current external panels establish new confirmation | their labels were opened during earlier method or application work | **false** | label them retrospective transfer/stress; reserve a future sealed population |
| every response method can be copied directly to localization, prefix, RAG, or stopping | each lane needs a token/step, causal-prefix, evidence-condition, or live-policy adapter | **false** | record `ADAPTER_NEEDED` or `INELIGIBLE` rather than a blank or loss |
| all 24-cell rows are independent | seven cells contain ten generations per source question and the consolidated matrix lacks parent IDs | **false for row uncertainty** | recover raw IDs and group by question before per-cell intervals |
| the inventory covers the acquired data | 34 registered population panels include core, transfer, applications, calibration, white-box, repeated-generation, and negative-scope data | **supported as a soft roster** | pre-run review may reclassify but should not silently drop a panel |

Protocol: `docs/experiments/MULTI_POPULATION_METHOD_BENCHMARK_V1.md`.

## Repeated-measurement reliability U-PCR (Step 234)

The experiment used synchronized moving-block bootstraps of one saved token
trace to estimate within-procedure covariance. It required no additional LLM
pass. Scores were frozen and hashed before evaluation labels were read.

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| the full feature pool behaves like repeated measurements under block bootstrap | full pool within/total trace ratio about 1.00 and negative signal eigenmass about 50% on GSM8K | **rejected** | do not treat all feature resampling variance as model noise |
| a restricted procedure-compatible pool yields stable covariance separation | 17/28 GSM8K and 18/28 MATH views; split covariance r=0.993/0.999; negative mass 4.12%/0.76% | **supported for the bootstrap procedure** | retain the Phase-0 validity diagnostics |
| generalized latent coordinates can be passed directly to U-PCR | off-diagonal covariance fraction falls from about 0.89 to 0.03/0.07; MATH AUROC falls below chance | **rejected mathematically and empirically** | U-PCR must keep correlated feature axes |
| hard reliable-subspace projection improves DUFS-LIU | 0.7617 vs 0.7673 GSM8K; 0.7108 vs 0.7188 MATH | **rejected on development cells** | do not promote hard projection |
| Wiener reliability filtering improves DUFS-LIU | +0.0006 and +0.0013 AUROC; both paired intervals include zero; score correlations about 0.98 | **unsupported / tie** | keep ordinary DUFS-LIU mixed-v2 |
| bootstrap stability identifies hallucination-relevant directions | stable covariance and subspace did not yield a meaningful AUROC gain | **rejected for this replicate process** | require a replicate intervention with a clearer target-preserving nuisance |

Full report: `results/repeated_measurement_reliability/REPORT.md`.

## GL-LIU v1 on ProcessBench (current leading end-to-end result, Step 232)

GL-LIU v1 combines a global mixed-contract DUFS-LIU error detector with a
continuous-token temporal-LIU locator. The score-construction path does not
receive correctness labels or reasoning-step boundaries. Development labels
select the components, calibration labels set the decision threshold, and
evaluation labels are opened after the scores freeze.

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| our spectral system can replace Mind the Gap end to end | ProcessBench F1 31.36% vs 25.71% over eight cells; GL-LIU wins 8/8 cells | **supported under the shared calibration protocol** | GL-LIU v1 is the leading ProcessBench method |
| the result remains outside the two component-selection cells | 30.76% vs 24.74% F1 over six non-selection/model-transfer cells | **supported, with dependence caveat** | carry the frozen system to external confirmation |
| the eight cells are eight independent datasets | 4B and 8B reuse the same ProcessBench examples | **false** | report four dataset families; only OlympiadBench and OmniMath are new families |
| the global DUFS Laplacian improves mixed IU-PCR | about +0.22 AUROC percentage points on average, 8 wins in 8 cells | **small but consistent real-data support** | retain mixed DUFS-LIU as the global detector |
| the mixed contract is the main source of the detector gain | development mixed DUFS-LIU 0.7812 vs stable DUFS-LIU 0.7800 | **unsupported as the main mechanism** | retain stable DUFS-LIU as a contract control |
| global detection should use an aggregate of local token risk | token maximum/top-5% detector candidates are around 0.72 AUROC vs 0.7812 for full-trace DUFS-LIU on development | **rejected** | keep global and local inference as separate stages |
| native moving-window scores localize errors without step-wise feature construction | GL-LIU locator improves exact SLA from 17.84% to 21.79% vs Mind the Gap in the full system | **supported for the feature source** | retain continuous token-grid entropy and spilled-energy curves |
| the temporal Laplacian is a universal localization improvement | selected exact 30.22% on development, but about 25.14% vs 25.78% for DUFS feature-graph IU on six non-selection cells | **fragile / not confirmed** | freeze it as v1, but pre-register ordinary IU and DUFS feature-graph IU controls |
| GL-LIU is fully label-free | labels select components and calibrate the threshold | **false** | describe it as calibrated unsupervised scoring |
| the reported Mind the Gap number reproduces its original decision policy | both systems use the same split-local F1 threshold, not the paper's Neyman-Pearson operating point | **false** | claim a fair common-protocol comparison, not exact policy reproduction |

Canonical definition: `docs/methods/gl_liu_v1.md`. Frozen report:
`results/ours_only_localization_v1/REPORT.md`.

## DUFS-LIU mixed feature contract (development candidate, Step 231)

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| the current DUFS-LIU benchmark already used transformed non-monotone views | runner and frozen definition say `fixed_stable_v1` | **false** | the four views were removed in the historical run |
| one common transform is appropriate for all four views | mixed winner uses squared, mode, raw, raw | **rejected as a restriction** | freeze per-feature operations |
| the selected mixed contract improves stable-only on development cells | +0.242pp, 17W/7L, worst -0.279pp | **retrospective development signal** | carry the fixed mapping to external confirmation |
| contract selection transfers across held dataset families | LOFO +0.123pp | **fragile / unsupported as a headline** | without one MATH-500/Qwen cell the mean is about +0.022pp |
| `rpdi` and STFT operations are stable | raw 8/8 and mode 7/8 held-family folds | **supported for freezing** | keep these decisions unchanged in the next run |
| `pe_mean` and CUSUM operations are stable | modal choice only 4/8 for each | **unsupported** | treat exact choices as candidate settings, not established facts |
| the mixed contract makes the Laplacian useful | DUFS-LIU minus IU-PCR is +0.048pp under mixed-v2 | **control-level hint** | still too small for a method claim |

The historical stable-only score remains the headline. Mixed-v2 is a frozen
candidate for an unseen-family confirmation, not a new confirmed baseline. See
`docs/research_notes/dufs_liu_mixed_feature_contract_conclusion.md`.

## Repeated cross-view alternating diffusion (current, Step 230)

The frozen RCV-AD-IU-PCR experiment tested whether a correctness-relevant
sample manifold survives repeated complementary partitions of the existing
static feature pool.

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| alternating diffusion is implemented and can recover a shared latent view | known-answer common-manifold graph CKA 0.143 vs 0.016 after node permutation | **mechanism only** | implementation is capable of the intended operation |
| dependency-blocked repeated partitions converge | median graph CKA 0.536; partition-score and T8/T16 Spearman about 1.000 | **supported on real features** | repeated estimation is stable enough; more partitions are not needed |
| converged graph stability predicts usefulness | graph CKA vs AUROC delta Spearman -0.240, p=0.259 | **unsupported** | do not select cells or parameters from convergence |
| dependency-blocked AD improves IU-PCR | +0.004pp, 10W/14L, family interval [-0.052,+0.029] | **control-level tie** | do not promote RCV-AD-IU-PCR |
| dependency blocking prevents a useful signal from being hidden by duplicate leakage | primary is -0.013pp vs atomic random and -0.015pp vs family blocked | **unsupported** | do not tune the correlation threshold |
| alternating composition is better than direct averaging | +0.037pp primary minus direct average | **narrow mechanism support only** | composition is active but not enough to beat IU-PCR |
| graph disconnection explains the tie | k=11 repairs most disconnection but remains +0.005pp | **rejected diagnosis** | do not rescue by a broader k sweep |
| stronger Laplacian influence reveals hidden benefit | dependency path -0.061pp at lambda 1 and -0.127pp at lambda 3 | **rejected** | larger correction amplifies target-neutral geometry |

Static repartitioning is closed as the leading route. Conditional family
specialization from Step 229 remains the supported problem, but the next view
must add information outside the existing feature matrix. See
`docs/research_notes/repeated_cross_view_diffusion_conclusion.md`.

## Graph-coupled family relevance (Step 229)

The frozen GCFR-U-PCR diagnostic tested whether within-family agreement plus a
small prior graph can identify sample-local family relevance. It separates a
supported scientific premise from a rejected implementation of that premise.

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| family expertise changes across samples or regimes | IU-PCR-rank context oracle +2.833pp equal-family headroom; permutation p=0.0020, Holm p=0.0060 | **supported as a label-only diagnosis** | retain conditional specialization as the problem to solve |
| trace length organizes family relevance | +0.802pp headroom, Holm p=0.585 | **unsupported** | do not route families by length |
| raw family disagreement organizes family relevance | +0.990pp headroom, Holm p=0.232 | **unsupported** | do not use disagreement as the router target |
| within-family agreement identifies useful local experts | primary GCFR -0.135pp vs IU-PCR, 8W/16L | **rejected on real cells** | do not learn a more flexible version of this gate |
| semantic family adjacency improves the local gate | primary -0.243pp vs beta=0; every beta>0 mean was negative | **rejected on real cells** | related measurement lineage is not a reliability graph |
| the failure is graph collapse or inactive gates | gate variation and 1.4--4.6% mean absolute rank changes were observed | **rejected diagnosis** | failure is target alignment, not execution |
| local gate without cross-family smoothing improves IU-PCR | best descriptive beta=0 result +0.108pp; family interval [-0.002,+0.428] | **weak post-evaluation hint only** | retain as a future control; do not promote or tune on these cells |
| the mechanism works when inactive members disagree | synthetic +0.773pp, 20W/0L | **mechanism only** | confirms implementation and narrow assumption |
| the mechanism handles coherent nuisance | synthetic -9.272pp, 0W/20L | **rejected synthetically** | coherent wrong agreement is the explicit falsification case |

The current junction is not another family-gating optimizer. IU-PCR rank may
define frozen regimes, but an independent interventional self-supervised
observation must identify which family is reliable inside a regime. See
`docs/research_notes/family_relevance_diagnostic_conclusion.md`.

## Atomic-operator Phase 0 (Step 228)

The registered v2 audit froze all label-free outputs before evaluation and was
independently reviewed. It blocks AOG Phase 1 for the tested proxy.

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| the frozen label-free atomic proxy predicts useful IU-PCR regularizers | median within-cell Spearman -0.312; family interval [-0.319,+0.249]; permutation p=0.690 | **rejected on real cells** | do not optimize global gates from stability/agreement/actuation |
| top-proxy selection is safe enough to develop | -0.838pp cell-macro, 7W/17L, worst -3.658pp | **rejected on real cells** | AOG Phase 1 is blocked |
| useful atomic actuation exists | label-only in-sample oracle +0.447pp cell-macro, 23W/0L | **oracle evidence only** | retain atomic operators as diagnostics; do not claim a transferable selector |
| a registered `k` or `lambda` rescues the proxy | all nine settings have negative association and negative top-bottom contrast | **rejected** | no wider graph-strength search on this objective |
| more stability sampling fixes the result | proxy rank agreement with the 40-subsample result is 0.990 after four subsamples | **rejected diagnosis** | the proxy converges to the wrong target |
| uniform atomic regularization improves IU-PCR | best registered cell-macro change is about +0.04pp | **control-level tie** | keep as a shrinkage control, not a new method |

The current problem is label-free target identifiability. The next premise must
use an independent interventional self-supervised signal rather than another
functional of the same static feature covariance. See
`docs/research_notes/atomic_operator_premise_audit_conclusion.md`.

## Evidence entering hierarchical-active v2

| component or claim | strongest evidence | status | consequence |
|---|---|---|---|
| confidence-oriented U-PCR is the incumbent | fixed-stable real macro 0.7735 over 24 cells; fixed orientation stays within -0.06pp of per-cell `sign(rho)` | **retained** | every new head is anchored on U-PCR and must beat it directly |
| fixed feature orientation removes the polarity failure mode | consensus and historical EPR anchors agreed in 288/288 fixed-schema comparisons | **retained** | no correctness-derived per-cell polarity is allowed |
| low-dimensional PCR protects the useful head | adding the inverse tail cost 3.28pp; full inverse SDSF cost 3.32pp | **retained** | corrections stay two-dimensional at the target |
| bootstrap SDSF stabilizes SDSF | +1.80pp over current SDSF, 23W/1L | **mechanism only** | useful stabilization, but not a replacement for U-PCR |
| bootstrap SDSF improves the incumbent | -2.91pp versus SU-PCR, 2W/22L | **rejected on real cells** | do not reopen full-inverse SDSF |
| raw pair-product covariance solves dependency-aware rho | every sealed v5 solver worsened full-rho error; GLS damaged retained coordinates | **rejected synthetically** | do not try more raw GLS estimators for the same equations |
| per-cell six-direction trusted-label correction | -0.36pp versus U-PCR at 20 labels, CI [-0.64, -0.05] | **rejected on real cells** | reduce target correction dimension and share information across cells |
| per-cell two-direction trusted-label correction | safer than six directions; -0.15pp at 20 labels and approximately tied by 80 labels | **retained control** | use as the local labelled baseline, not yet as an improvement |
| U-PCR pseudo-label self-training | at 20 labels, pseudo+gold lost to anchored-6 in all 24 cells by 3.97pp | **rejected on real cells** | no pseudo-label arm in v2 |
| dependency correction can repair a biased U-PCR head | anchored-6 gained +0.79pp on sparse pairs and +30.71pp on a correlated weak block | **mechanism only** | preserve a shared-correction synthetic positive control |
| anti-redundancy feature selection | DPP -8.08pp (0W/24L); decorrelation -5.98pp against matched random subsets | **rejected on real cells** | do not treat diversity as an automatic benefit |
| a label-derived feature subset has real headroom | label-handed oracle +2.25pp; half-split oracle transfers 84% of the gain | **oracle evidence** | labels can help, but the target is non-unique and unstable |
| fixed choices transfer across cells/families | numerous LOCO/LOFO feature and shape choices were flat or negative | **unsupported** | v2 excludes the entire target family and treats transfer as a falsifiable mechanism |

## Current cycle

The registered protocol is `SPEC_HIERARCHICAL_ACTIVE_SPECTRAL_V2.md`.  The
confirmatory decision is **STOP_AND_REVISE**.

| component or claim | v2 evidence | status | consequence |
|---|---|---|---|
| a same-domain LOFO correction transfers | pooled-only -1.44pp vs U-PCR, CI [-2.01, -0.92], 1W/23L | **rejected on real cells** | stop cross-family linear correction for this feature bundle |
| broad pooling fixes the hierarchy | pooled-all -0.96pp, 3W/21L; less harmful but uses more donor labels | **rejected on real cells** | QA/math membership is not a useful correction hierarchy |
| active acquisition improves a local two-score head at 20 labels | active-local minus uniform-local -0.13pp, CI [-0.30, +0.04] | **unsupported** | do not claim generic active-learning gain |
| active acquisition can reject a bad transferred prior | hybrid-active minus hybrid-uniform +0.78pp, CI [+0.53, +1.08], 23W/1L | **retained safety mechanism** | informative labels should be used to test/shrink external priors, not assumed to improve U-PCR |
| hierarchical-active combined method improves U-PCR | -0.19pp, CI [-0.41, +0.02], 8W/16L | **rejected on real cells** | do not promote the v2 candidate |
| shared correction is learnable when truly shared | +47.63pp in the sealed shared-correction meta-world | **mechanism only** | harness is capable of detecting the intended effect; real transfer failure is informative |
| a selector over U-PCR and v2 can yield a major gain | perfect cell-level switch ceiling +0.12pp | **closed as headline** | safety gating cannot reach the contribution bar on these candidates |

## Pending aligned v2 benchmark: method family versus selection regime

This is a registered design distinction, not a new result. Family-NRM and PGRD
will each be compared under three regimes: A, target-cell only with no donors
or labels; B, donor-unsupervised; and C, donor-label selection. Only A enters
the main unsupervised 24-cell leaderboard. Target-label oracles remain separate
diagnostics.

| component or claim | current evidence | status | consequence |
|---|---|---|---|
| Family-NRM or PGRD inherently needs donor data | both corrections can be constructed from target-cell family residuals alone | **false as a method definition** | run new within-cell A variants in the main leaderboard |
| existing Family-NRM result measures regime A or clean B | direction was calibrated across 23 source cells; donor eligibility used a label-derived minimum-positive rule | **false; legacy C under the strict axis** | do not relabel the old result as within-cell or donor-unsupervised |
| existing PGRD result measures regime A | graph moments were donor-pooled and donor labels selected strength/policy | **false; historical C** | keep as a secondary supervised-selection result |
| Step-286 intrinsic selector is clean end-to-end B | geometry was label-free but inherited donor-label-selected lambda/trust | **false; legacy C** | rerun cross-only with structurally fixed strength for clean B |
| DEEM-B3 is currently comparable to the old graph table | DEEM used the full present inventory and registered equal-family primary; old graph scores use `fixed_stable_v1` and cell-macro | **not yet comparable** | rerun all main methods under one feature and macro contract |
| Residual-Graph DEEM falsifies graph-free DEEM | graph arms stopped at a synthetic specificity gate; B3 later completed natural-data evaluation | **false** | keep the two experiments separate |

The pending roster and protocol are
`configs/global_24cell_method_benchmark_v2_registry.csv` and
`docs/experiments/GLOBAL_24CELL_METHOD_BENCHMARK_V2.md`.

## Frozen 24-cell view-fusion cycle

The frozen benchmark supersedes the ten-cell CA-SpecRaGE pilot. It used all 24
available cells and kept labels sealed until the feature contracts,
hyperparameters, score directions, and output scores were frozen. The current
decision is a negative result: none of the view-fusion methods is promoted.

| component or claim | full 24-cell evidence | status | consequence |
|---|---|---|---|
| manual semantic families are necessary fusion units | balanced atomic CA is +0.215pp versus manual CA, although the family-bootstrap interval crosses zero | **rejected as a default** | semantic provenance remains a reporting factor, not a required view definition |
| balanced atomic CA improves IU-PCR | +0.023pp, 11 wins / 1 tie / 12 losses; paired interval crosses zero | **unsupported / tie** | retain as a diagnostic control, not a new method |
| sample-specific view reliability helps | sample alpha loses slightly to global and permuted alpha for manual, atomic, and micro views | **rejected on real cells** | stop local alpha development on this feature bundle |
| fusion-aware micro-views are reproducible | bootstrap ARI 0.84–0.94; all cells select three groups | **retained diagnostic** | stability can verify reproducibility, not usefulness |
| fusion-aware micro-views improve IU-PCR | -0.363pp, 5 wins / 19 losses, worst cell -2.855pp | **rejected on real cells** | do not replace semantic families with learned micro-views |
| a hidden Laplacian strength rescues the result | atomic best grid point +0.056pp and global-alpha +0.072pp; neither clears uncertainty or promotion gates | **unsupported** | do not launch a broader lambda sweep |
| the negative result is graph collapse or solver failure | headline graphs are connected and finite; 135 null values are only undefined algebraic connectivity for secondary Y graphs | **rejected diagnosis** | interpret the failure as target identifiability, not execution failure |
| DUFS-LIU improves IU-PCR | +0.008pp macro | **control-level tie** | keep it as a baseline and design source |
| synthetic CA-SpecRaGE transfer predicts real benefit | full real benchmark does not reproduce the synthetic gain | **rejected as transfer evidence** | synthetic success remains mechanism evidence only |

Stable confidence-oriented U-PCR/IU-PCR remains the incumbent. The next
question is narrower than another fusion model: can a pre-frozen, label-free
atomic-operator diagnostic predict which feature-induced Laplacians help the
IU-PCR head across held-out families? The registered Phase-0 premise audit is
in `docs/research_notes/atomic_operator_gating_plan.md`. If the premise fails,
the graph-regularization line closes before another learner is built. The full
post-run interpretation is in
`docs/research_notes/frozen_24cell_view_fusion_conclusion.md`.
