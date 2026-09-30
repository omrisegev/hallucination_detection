# Non-graph localization: prior experiment audit

Private review artifact, 2026-09-06. Bounded read-only audit of actual reports, frozen protocols and implementation references. No evaluation ran. No unopened Joint L-SML v2 outcomes were read. The active Module-B descriptions below come only from the protocol and its R1 amendment.

Scope: current checkout reports; `origin/codex/reasoning-localization-03662-v1` (Phase-3 closure); and `C:/Users/omris/TAU/hd_jlsml_v2_wt/docs/experiments/`. “No located test” means the reviewed report/protocol/code inventory did not establish a completed test. It is not a claim that every historical notebook or private conversation has been searched.

## Already tested: do not present these as new ideas

| Approach | Evidence and outcome | Exact source |
|---|---|---|
| Two-state temporal HMM / inferred error onset | Step 246: reversible IU-HMM PB F1 **30.03%**, ordinary IU core **31.67%**, DUFS core **31.72%**; absorbing HMM **12.64%**. Reversible local exact **25.20%** vs IU **26.62%**. The HMM outputs posterior entry into the high-risk state, not simply state occupancy. No promotion. | `results/processbench_latent_state_v1/REPORT.md`; `spectral_utils/latent_state_localizer.py` |
| First persistent crossing instead of peak/argmax | Step 273 explicitly crossed Local families/operators with peak, first calibration-q90 run of three tokens, and step-top-five mean. Same development family6+level: persistent-q90 **0.3223**, peak **0.3057**, top-five **0.3517** F1. All uncertain against direct Step-272 **0.3503**; first crossing is not an untested suggestion. | `results/local_online_comprehensive_v1/STAGE_1_LOCAL.md`; `REPORT.md` |
| Generic innovation / fast-slow / onset / CUSUM / BOCPD bank | Steps 270–274 cover level, innovation, short-long contrast, persistence, positive area, causal moments, one-sided CUSUM, Page-Hinkley, BOCPD hazards 1/50 and 1/100. Step-272 Local level9 **0.3484**, onset-only **0.2464**, level+onset **0.2685**. Unified full **1,036-coordinate** DSP bank rejected in favor of 28 coordinates. Generic “use change points” repeats an existing direction. | `results/global_local_online_architecture_v2/REPORT.md`; `docs/experiments/UNIFIED_CAUSAL_IU_V1.md`; `docs/reports/UNIFIED_CAUSAL_IU_V1_REPORT.md` |
| Leaky evidence accumulation / cumulative hazard / first and persistent alarm crossing | Unified causal protocol explicitly included identity, leaky recovery horizons 8/16/32/64 with three drifts, irreversible cumulative hazard, first crossing and three-token persistent crossing. Completed cycle retained Unified-28 and did not promote one universal head: Llama localization **0.2880** vs **0.2419**, but Global/Early regressed. Historical exploratory checkpoints are incompletely retained, so a numerical result for every hazard arm cannot be reconstructed from the final report alone. The family has been explored; a particular new supervised hazard loss is a separate question. | `docs/experiments/UNIFIED_CAUSAL_IU_V1.md`, sections Trajectory outputs/Frozen outcome; `spectral_utils/unified_causal_iu.py` (AccumulatorSpec / cumulative_hazard); `docs/reports/UNIFIED_CAUSAL_IU_V1_REPORT.md` |
| Step-first versus trajectory-first feature fusion | Step 253: PRMBench **0.6136 step-first vs 0.6711 trajectory-first**; latter is the fixed baseline. Current v2 explicitly drops this as a new search axis and retains one PRMB-only diagnostic row. | `results/fixed_application_pipelines_v1/REPORT.md`; v2 `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md`, prior-art table |
| Whole-answer-relative centering / local innovation | Step 293 predicts each of 29 token coordinates from whole-answer mean and frozen CIW answer risk, then uses bounded OOF-R2-gated standardized innovations. CIW-token-IU PB **0.308301**, PRM AUROC **0.582489**; older CIW adapter PB **0.309136**, PRM **0.581138**. Approximately +0.0013 PRM across response heads but worse PB. Preserving original within-answer mean/scale removes both loss and gain. Not promoted. | `results/ciw_cross_scale_localization_v1/REPORT.md`; `docs/experiments/CIW_CROSS_SCALE_LOCALIZATION_V1.md`; PROGRESS Step 293 |
| Context/self-basis innovation + simple rank fusion | C8 self-innovation was tested alone, conditionally in family6, and as H2/H3 role-separated reranking. H3 on Qwen **0.366653**, H0 difference **+0.012392 [+0.001769,+0.022807]**, but outcome-selected ancestry and practical gate prevent promotion. Llama H3 **0.353281** vs H2 **0.355583**; PRMB H3 **0.619469** vs H2 **0.597871**, supported specialist result on historically opened labels. Generic “use context-dependent surprise” must distinguish itself from this existing C8 method. | newer ref: `docs/experiments/REASONING_LOCALIZATION_03662_H3_RELIABILITY_V1.md`, `...H3_LLAMA_TRANSFER_RESULTS_V1.md`, `...H3_PRMBENCH_DIAGNOSTIC_RESULTS_V2.md`; `...ANCHOR_V1.md` |
| Reliability weighting of H2/C8 | Donor rank stability under 12 within-step moving-block perturbations gave near-equal C8 weights **0.4813–0.5000**, median **0.4968**, no advantage over 50/50. | newer ref: `docs/experiments/REASONING_LOCALIZATION_03662_H3_RELIABILITY_V1.md` |
| Family-level IU / pruning / STG-SU / graph-free query pooling | Phase 3 closed with no promotion. H2 **0.364090**, matched equal control **0.364284**, dynamics-IU **0.366876**, increment **+0.002592 [-0.001839,+0.007194]**. Deployed prune/refit **0.356740** vs unpruned IU **0.354240**, unconfirmed. STG-SU **0.366762** does not establish its mechanism. Q1 point-query pooling, explicitly no graph, **0.354584**; exact-error delta **-0.010644 [-0.020123,-0.001210]**. | newer ref PROGRESS Steps 340–346; `results/reasoning_localization_03662_v1/phase_3/`; `docs/experiments/REASONING_LOCALIZATION_03662_ASTGI_QUERY_HEADS_V1.md` |
| Length residualization and strata | Step 272 calibration-only isotonic residualization: AUROC@64 **0.5947 → 0.5801**, @128 **0.6204 → 0.5980**. Step 273 also retained short/medium/long and first-error-position strata. These are early-detection diagnostics, not a scored step-length-conditioned tail-calibration localizer. | `results/global_local_online_architecture_v2/REPORT.md`; `results/local_online_comprehensive_v1/REPORT.md`, `STAGE_4_STRATA.csv` |

## Active Module B: already in the running study, outcomes not inspected

The v2 registered module already covers:

- B0 frozen top-10 mean on PB / official-span maximum on PRMB.
- B1 learned label-free weights over ten sorted within-step token-risk order statistics.
- B2a max/mean blend, inner-CV alpha grid {0,.25,.5,.75,1}.
- B2b five relative-position bins and learned weights.
- B3 **supervised balanced logistic regression over the same ten order statistics**. PB first-error steps are positive, pre-error steps negative, post-error steps excluded; PRMB uses official every-step labels.
- A within-answer-centered covariance fit as an **unscored weight-profile diagnostic**.
- R1 amendment: three feature-axis substrates (IU, internal continuous L-SML, internal Joint) by three order-statistic fusers (SML, IU, Joint), donor-fitted slot standardization and score SD=1. One nested-selected primary comparison replaces the earlier B1 primary; remaining rows are descriptive.

Therefore “learn the reducer,” “try top-k weights,” “learn where inside the step,” “compare max with mean,” and “try a supervised step LR” are not new opportunities. Source: v2 worktree `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md` §6B and `JOINT_LSML_OPTIMIZATION_V2_AMENDMENT_R1.md` §3. The study's outcome status was deliberately not inferred from result files.

## Proposed but not run, or only a diagnostic request

- **Matched non-error step-boundary control:** HMM report explicitly says the sharp error-boundary posterior peak still needs a matched non-error-boundary control to separate error onset from generic step syntax. A completed follow-up test was not located in this audit.
- **ASTGI Q2 donor-learned causal coordinates:** masked feature reconstruction / next-observation / cross-family prediction was a planned non-graph rung. Q1 did not pass, so Q2–Q4 are explicitly **NOT_RUN_BY_GATE**, not negative results. Q3–Q4 are graph-related and outside this request. The document explicitly excludes end-to-end supervised ASTGI on the current labels.
- **Supervised streaming probe:** cited as a high-access comparator, but fair comparison report marks official trajectories/labels/splits/layer/probe/evaluator assets blocked. That is not evidence of a trained project-specific sequence localizer. `results/fair_paper_exact_comparisons_v1/REPORT.md`.

## Four concrete gaps, distinguished from tested cousins

These are opportunity descriptions for the parent to assess, not authorizations or recommendations to launch all four. No completed matching test was located after reviewing the report families, method contracts, and relevant implementation inventory above.

### 1. Token-role-conditioned nuisance calibration

Question: do risk spikes on numbers, operators, copied entities, punctuation and step-opening phrases mean different things? Estimate the expected telemetry distribution conditional on a small frozen token role and distance from a step boundary; score excess risk relative to that expectation. Begin by checking true error boundaries against matched correct boundaries of similar role/position.

**Distinct from tested work:** “structural” family/relative position, HMM onset, C8 self-innovation, and Module-B positional bins do not constitute explicit semantic/syntactic token-role conditioning with matched non-error boundaries. The HMM report itself documents this unresolved confound. A hard hand-picked mask would change the project's no-prior ideal; learned or fully disclosed role conditioning needs that boundary stated.

### 2. Step-length-conditioned tail significance

Question: does a maximum/top-ten score look extreme merely because the step offered more token opportunities? Calibrate a frozen reducer against a donor distribution conditional on step length and, if justified, token dependence or token roles; keep the response detector fixed to isolate locator changes.

**Distinct from tested work:** Module B learns weights across ranks; Step293 centers relative to whole answers; prior length work residualizes final-length effects in early AUROC. None is an established completed test of the conditional null distribution of a within-step extreme. This should not be sold as an untested generic “length control”; the novelty is the step-tail estimand and multiple-opportunity correction. Using clean labels to estimate a null would be supervised calibration; using unlabeled steps gives only a mixture-reference percentile, not a guaranteed clean null.

### 3. Controlled semantic step verification

Question: when the model is confidently wrong, can evidence beyond output uncertainty verify the step's actual relation to the question and prior steps? Examples: deterministic arithmetic/equation checking where a check is well-defined, or fixed-answer teacher-forced contrasts under a narrowly specified premise correction/deletion.

**Distinct from tested work:** RAG evidence contrast and Evidence-Drop already perturb supplied evidence; C7/EDIS and C8 use telemetry structure. They do not establish a completed natural-reasoning test that isolates a candidate mathematical claim against its required premises. Raw NLI or “add a verifier” would be underspecified and may duplicate existing PRM/critic comparisons. The new element must be a controlled step-validity target with matched benign perturbations. Model rescoring requires additional passes and changes the access/cost tier; deterministic verification introduces explicit task knowledge and incomplete coverage.

### 4. First-error-specific supervised sequence readout

Question: under an explicitly supervised branch, can a small discrete-time event model or within-answer listwise ranker predict **first error or no error**, instead of independently scoring every step? Pre-error steps form the at-risk set, clean traces are censored/no-event observations, and later steps must not silently become clean labels.

**Distinct from tested work:** Step246 HMM fitted unlabeled sequence likelihood; Unified cumulative hazard was a hand-defined accumulator; Module-B LR fits independent step labels/order statistics. A grouped, directly first-error-trained sequence/listwise objective with a no-error option was not located. This is not a claim that “hazard models were never tried.” It changes the training objective and supervision boundary, and should be rejected as a suggestion if the user requires strictly label-free methods.

## Interpretation limits

- The strongest reason to consider new signal/target formulations is the repeated difference between fitting structure and ranking actual errors, not a proof that every non-graph fusion is saturated.
- PB first-error F1, PRMB every-step ranking and causal final-error warnings must stay separate. A new supervised first-error model cannot use a PRMB AUROC win as proof of PB benefit.
- The Qwen/Llama PB telemetry repeats the same 3,400 questions. A new scorer model or another frozen fit on those questions is not fresh-question confirmation.
- Do not revive the proposed Q2 rung or add experiments to Module B merely because this review identifies a distinction. A distinct idea requires a new bounded contract and explicit scientific rationale.

## Clarification after the user's matrix-observation correction — 2026-09-06

**The earlier “already tested” labels were too broad if read as ruling out the same idea on a new step-observation matrix.** The observational unit is part of the method. A covariance estimated across whole answers, across token positions, and across reasoning steps answers three different questions. An old negative transfers directly only when the representation, observation unit, fit population and readout are sufficiently matched. The matrix change must be stated before deciding whether a proposed test is a repetition.

Three distinctions are separate:

1. What is one observation/column in the mathematical `p × N` matrix: answer, token or step?
2. Are observations pooled across donor answers, or is a separate model fitted inside each answer?
3. Does fitting use chronological order, or just the covariance of otherwise exchangeable observations?

The implementation often stores `N × p` and transposes it for `upcr_fit`; this storage convention does not change the observation unit.

### Verified historical matrix map

| Experiment | Features p and observations N | Pooling and chronological treatment | Source anchors |
|---|---|---|---|
| Historical completed-answer mixed-v2 Global | One scalar full-trace feature per response; nominal global pool around 30, retained count data-dependent. `F` columns are different **answers**. | One fit over a cell's answer population. Temporal shape enters already-aggregated features; covariance is across answers. | `scripts/gl_liu_v1/run.py:80` builds one `trace_features(row)` per answer; `:94` stacks response rows; `:96` transposes into `F`. |
| Step 246 IU-HMM | First-stage IU uses the **five token curves**, so `F` is approximately `5 × N_tokens`, not `5 × N_answers`. HMM then sees **one scalar IU risk sequence per answer**. | Token values are concatenated from selected response chunks up to the inherited 60,000-token cap for IU. HMM parameters are shared across answer sequences; forward-backward runs separately in chronological order inside each sequence. It does not model ten order statistics per step and does not fit a separate HMM to every answer. | `scripts/gl_liu_factorial_v2/run.py:127` builds chunks; `:144` concatenates token curves; `:161` fits transposed token matrix; `scripts/gl_liu_v1/run.py:59` defines cap; `scripts/processbench_latent_state_v1/run.py:316` prepares token input, `:329` restores individual IU risk sequences; `spectral_utils/latent_state_localizer.py:268` loops sequences in forward-backward; `:498` pools only normalization statistics. |
| Step 253 fixed trajectory-first reasoning | **29 token coordinates × pooled sampled tokens**. Whole-answer traces stay uninterrupted until risk-to-step aggregation. | Stable-ID-ordered token matrices are vertically stacked and sampled. Shared IU weights fit across donor tokens. Chronology affects token features, not IU covariance itself; official step mapping occurs after fusion. The older step-first comparator was explicitly a different observation construction. | `spectral_utils/fixed_application_pipelines.py:76` token feature matrix; `:106` token sampling; `:123` pooled `vstack`; `:143` IU confidence fit; `:196` shared token fit. `results/fixed_application_pipelines_v1/REPORT.md` records trajectory-first PRM 0.6711; current v2 prior table records step-first 0.6136. |
| Step 272 Local / Online raw-token architecture | Selected Local `l_level9`: **9 feature coordinates × sampled token positions**. Other local and online heads change feature/operator p. Global heads separately use one row per answer. | Local/online sample 32 equal token positions per calibration trace and pool them. Causal states use order within each trace; covariance fitting itself ignores ordering. Shared fit, not a per-answer local fit. | `spectral_utils/multitask_trajectory.py:26` fixes 32 positions; `:62` declares level9; `:320` branches Global answer rows vs Local/Online token rows; `:339` samples positions; `:345` pools; `:464` fits `standardized.T`. |
| Step 273 comprehensive Local | Raw9, broad28, family6 or core5 representation, crossed with causal operators. `N` is again **sampled tokens**, not answers or official steps. | Token matrices formed separately per trace; selected positions pooled across calibration answers; causal operators use within-trace order. Step top-five / peak / persistent locator is downstream. | `spectral_utils/local_online_comprehensive.py:233` representations; `:248` causal `T × D` operators; `:358` trajectory fit; `:375` operator matrix; `:376` token sampling; `:379` pooling; `:391` transposed IU fit. |
| Step 293 CIW cross-scale token innovation | **29 token streams × pooled sampled tokens** plus two response-level predictors for each coordinate (its answer mean and CIW answer risk). | Response ownership defines OOF folds; context is whole-answer. After coordinate correction, a shared token IU head fits pooled transformed tokens. It is not a covariance fit over step-level rows. No temporal transition likelihood is fitted. | `spectral_utils/ciw_cross_scale_localization.py:81` enforces tokens-by-29; `:94` token sample; `:112` per-answer coordinate means; `:119` owners; `:129` response-held folds; `:151` fits transformed token matrix; `:221` step maxima afterward. |
| Phase-3 compact H2 outer IU | **4 family-risk coordinates × sampled tokens**. | Family risks computed per token, then shared outer IU over fit tokens. Top-ten step aggregation is afterward. C7 is a within-response onset transform, not a change in covariance sample unit. | Newer worktree `scripts/reasoning_localization/run_phase3_compact_fusion.py:41` builds H2 family matrix; `:52` column-stacks four per-token families; `:59` fits outer IU; `:66` transposes token rows; `:113` top-ten reduction afterward. |
| Phase-3 member / family-expert ladders | **24 H2 member streams × sampled donor tokens**; family experts use 1/14/3/6 features (singleton passes through; IU fits 14/3/6). | Five folds grouped by response/source ownership; held rows are projection-only. Shared donor-token covariance per family; top-ten step reduction follows. Thus these negatives are already local-token evidence, but not the new step-order-statistic matrix evidence. | Newer worktree `scripts/reasoning_localization/run_phase3_deployed_upcr_prune_refit.py:106` member construction, `:137` token columns; `run_phase3_family_expert_attribution.py:115` token ownership, `:118` donor folds, `:123` donor token indices, `:129` family IU, `:150` step reduction. |

Consequences for interpreting prior results:

- The response-global graph, feature-selection and reconstruction negatives directly predate the **answer → step** observation change. They cannot reject a method fitted on step observations.
- HMM246, Local272/273, CIW293 and Phase3 were **already token-local**. It would be inaccurate to say those experiments all used one observation per answer. However, they also did **not** fit covariance across ten within-step order-statistic coordinates with steps as observations. Their negatives are relevant neighboring evidence, not direct tests of that new matrix.
- The HMM is the clearest genuinely chronological historical model: it preserves token sequence order inside every answer. Ordinary token or step covariance is not a temporal transition model merely because observations originated in a trajectory.
- In the current Module B, the code verified by the parent and independently checked here pools outer-training **steps from multiple answers**. “Steps are observations” is correct; “a fresh model is fitted within each answer” is not what that code implements. `scripts/joint_lsml_optimization_v2/run_v2.py:324` uses answer ownership to select donor steps, and `:328` passes all those selected steps to one reducer fit. Ten sorted token-risk ranks are coordinates; sorting discards the original positions of those tokens, while B2b is the separate positional-bin arm.

### Independently verified protocol/code mismatch: the promised within-answer covariance diagnostic

The v2 protocol `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md:235` promises a **within-answer-centered covariance** weight-profile diagnostic. The source currently implements:

```python
# scripts/joint_lsml_optimization_v2/run_v2.py:332–334
centered = matrix[train_steps] - matrix[train_steps].mean(axis=0, keepdims=True)
centered_weights, _ = fit_orderstat_weights(
    centered + matrix[train_steps].mean(), lengths[train_steps]
)
```

This subtracts one global mean per coordinate across **all** donor steps, then adds one scalar grand mean. The `step_rows` answer-owner vector is not used for centering. `spectral_utils/trajectory_reducer.py:137–141` selects full-length steps and calls SML; `spectral_utils/fusion_utils.py:378` computes `np.cov(X.T)`. Subtracting any constant per coordinate leaves that covariance unchanged mathematically, including when the full-length subset is selected afterward. Thus this diagnostic cannot remove between-answer covariance and does not implement the promised separation of within-answer and between-answer structure. This is a source-level finding; no outcome or score file was opened and no rerun or repair was performed.

The earlier audit's statement that within-answer centering was “covered as an unscored diagnostic” must therefore be narrowed: **it was promised by the protocol, but the inspected source does not perform it.** Global centering, CIW answer-context innovation, and actual answer-group centering are different operations. This distinction directly affects whether the recent step matrix has isolated comparisons among steps of the same answer or still contains strong between-answer variation.
