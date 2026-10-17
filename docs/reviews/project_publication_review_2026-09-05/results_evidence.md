# Publication review: result evidence memo

Read-only result extraction, 2026-09-05. No models, evaluations, rescoring, branch checkout, result edits, or remote transfers were run. This memo reports evidence and existing decisions; it does not choose new experiments or deletion targets.

## Source boundary

- Current checkout: `b214217a`, branch `codex/token-local-fusion-optimization-v1`; its PROGRESS starts with August 27 and is stale relative to the September work.
- Reconstruction implementation reference: `origin/codex/reconstruction-benchmark-v1` = `1780572a2db83958a8889aa0114c4996e9f10422`.
- Reconstruction committed scientific results: `origin/codex/reconstruction-science-results-v1` = `d9827a835bd2b43fc107d297834c8e190224b9bd`.
- Reasoning localization closure: `250e092e`, `origin/codex/reasoning-localization-03662-v1`.
- Joint L-SML v1 evaluation and failure diagnostic: `0ce48968`, `origin/codex/joint-lsml-localization-eval-v1`.
- Newer accessible worktree: `C:/Users/omris/TAU/hd_jlsml_v2_wt`, branch `claude/joint-lsml-optimization-v2`, HEAD observed `ff800082`. Its PROGRESS confirms September 4 closure of v1 and Phase 3. A historical status note is not a live running-job check.

## 1. Current global benchmark: simple family averaging is the key control

Certified reconstruction release `2026-08-24_frozen24_v1`, 13 methods on the same mixed-v2 input. Completed-answer error detection. Retrospective development, not untouched confirmation. Report totals: 48,607 rows / 16,195 source groups across 24 cells. These are reported aggregate group counts, not a verified count of globally distinct questions across all models/cells. Intervals use 20,000 source-group draws within cells, then equal-cell averaging; they quantify uncertainty within this fixed panel, not uncertainty across all possible datasets/models.

| Method | Cell-macro AUROC | Marginal 95% interval |
|---|---:|---:|
| DEEM-B3 | 0.7812467305 | [0.7720274349, 0.7900728928] |
| Equal-family mean | 0.7810131581 | [0.7718608711, 0.7898503339] |
| DUFS-LIU | 0.7765559611 | [0.7673111728, 0.7855732579] |
| IU-PCR | 0.7760865493 | [0.7668127152, 0.7851070084] |
| Family-NRM A | 0.7746413046 | [0.7654958172, 0.7835300957] |
| CA-SpecRaGE atomic | 0.7742308964 | [0.7647111160, 0.7834782517] |
| Deployed-style U-PCR | 0.7739543039 | [0.7643496636, 0.7831434799] |
| Equal-feature mean | 0.7739254952 | [0.7647880335, 0.7826831929] |
| SU-PCR | 0.7713811066 | [0.7619926174, 0.7804605795] |
| Continuous L-SML | 0.7709542079 | [0.7615009323, 0.7801510709] |

The decisive direct paired contrasts, candidate minus B3:

- Equal-family mean: **-0.0002335724 [-0.0009597310, +0.0003211225]**. No demonstrated advantage of B3 over this simple control; not evidence of equivalence.
- IU-PCR: **-0.0051601811 [-0.0077938899, -0.0024945218]**.
- DUFS-LIU: **-0.0046907694 [-0.0072835644, -0.0020946563]**.
- These winner-reference intervals have `multiplicity_adjustment=NONE`; they are post-ranking descriptive paired comparisons, not a corrected winner-selection procedure.

Sources: current `docs/meetings/advisor_update_aug21_2026/leaderboards/release_leaderboard.csv`; direct contrasts at `d9827a83:results/reconstruction_benchmark_v1/derived/winner_contrasts_v1_certified/frozen24/build_A/winner_reference_contrasts.csv`. The certified reporting bridge has nine authenticated source bindings but does not create a pooled cross-task estimand.

Do not confuse this table with `results/frozen_24cell_benchmark/REPORT.md`, the older August 7 stable-contract CA-SpecRaGE study: IU 0.7741, DUFS 0.7741, CA atomic 0.7743. Different release/contracts explain plausible-looking competing current tables.

## 2. CIW-DEEM: official challenger, no promotion

Current `results/ciw_deem_v1/{REPORT.md,RESULT.json}`:

- Cell-macro AUROC/AUPRC: **0.7820255514 / 0.7517170842**.
- Equal-dataset-family AUROC/AUPRC: **0.7492330051 / 0.7791317277**.
- Registered equal-family AUROC delta vs B3: **+0.0007316506**; exact eight-family one-sided sign-flip **p=0.13671875**; required improvement **+0.0025**, unmet.
- Status `OFFICIAL_CHALLENGER_NOT_PROMOTED`.
- The registered CIW table's B3 cell macro is **0.781815346014**, whereas the certified reconstruction B3 is **0.781246730461**. Do not subtract CIW from the reconstruction number. No harmonization/recomputation was performed in this review.
- Supervised group-OOF balanced LR on CIW input: 0.7827757141 cell / 0.7427084969 equal-family. Pre-CIW LR: 0.7834087246 / 0.7433574384. CIW did not improve linear separability.
- IU on CIW: 0.7739522562 / 0.7411060399. DUFS on CIW: 0.7743883890 / 0.7419007565. Pre-CIW D1 DUFS is higher: 0.7754416158 / 0.7428118625. Thus CIW is a small B3-specific challenger, not a universally useful representation.

## 3. External completed-answer transfer: heterogeneous

Reconstruction external release `2026-08-24_external_final_answer_v3_opaque`, at `d9827a83:results/reconstruction_benchmark_v1/releases/.../build_A/external_final_answer/evaluation/metrics_long.csv`:

| Population | B3 AUROC | Equal-family | IU-PCR | DUFS-LIU | Rows/groups |
|---|---:|---:|---:|---:|---:|
| ProcessBench Llama 4 cells | 0.702077 | 0.701895 | 0.694509 | 0.694872 | 3,400/3,400 |
| ProcessBench Qwen 8 cells | 0.704434 | 0.704096 | 0.696417 | 0.697139 | 6,800/3,400 |
| PRMBench response | 0.696677 | 0.694796 | 0.720917 | 0.722894 | 6,966/6,208 |
| HLE interim | 0.499909 | 0.502139 | 0.518316 | 0.508113 | 2,158/2,158 |

CIW compatible transfer covers 22 external completed-response cells. Current `results/ciw_deem_multi_application_v1/{REPORT.md,METRICS.csv,COVERAGE.csv,RUN_MANIFEST.json}`:

| Population | CIW | B3 | Difference |
|---|---:|---:|---:|
| Evidence-Drop 4 cells | 0.830674 | 0.831904 | -0.001230 |
| GPQA K10 stress 4 cells | 0.561741 | 0.560316 | +0.001424 |
| HLE interim | 0.505685 | 0.499909 | +0.005777 |
| PRMBench response | 0.688643 | 0.696677 | -0.008035 |
| ProcessBench response 12 cells | 0.701802 | 0.703648 | -0.001846 |

These compact CIW deltas are descriptive; paired intervals are not in the compact artifact read here. Their scientific interpretation cannot be upgraded from point estimates.

Coverage limits are substantive: EDIS lacks the partition-energy source needed by exact CIW's three-source by three-operator core. Sentence/token/span/claim RAG units, stopping policies, and white-box hidden-state inputs cannot silently inherit the completed-response algorithm. The stored large CIW score-freeze root is `local_cache/ciw_multi_application_v1`; it was not present in this current checkout.

## 4. Localization: keep distinct contracts separate

### Certified/fair and historical regimes

At `250e092e` and the newer worktree's Step 346:

- Canonical fair 3,400-row ProcessBench incumbent: **family6 + level + step_top5mean, F1 0.326141**.
- Historical Local/Online Stage-4 finalist: **0.3662328342**, a rejected historical-regime audit anchor, not a current matched incumbent for a 0.326141 comparison.
- Opened Qwen development H2: **0.364090**. It is also not directly rankable against those two different regimes.
- Matched historical bridge used 1,270 scorer rows / 635 source-question groups, historical 40% calibration and 20% audit roles. Historical finalist 0.366233; H0 0.374099; H2 0.374793; H3 0.372663. H2 minus historical **+0.008560 [-0.024610,+0.040869]**, unconfirmed. The favorable F1 points are mainly associated with changed clean-answer abstention, not a demonstrated better error locator.

Source: `250e092e:docs/experiments/REASONING_LOCALIZATION_03662_H3_HISTORICAL_HEADTOHEAD_RESULTS_V1.md`.

### H3 PRMBench specialist: supported within the frozen diagnostic

`250e092e:docs/experiments/REASONING_LOCALIZATION_03662_H3_PRMBENCH_DIAGNOSTIC_RESULTS_V2.md`; artifacts under `results/reasoning_localization_03662_v1/phase_2/transfer/h3_prmbench_v2/`.

- 83,280 annotated steps, 6,208 paired source groups; 13,144 error steps / 70,136 negative steps; 20,000 whole-source bootstrap draws; simultaneous Bonferroni intervals across three frozen contrasts separately per metric.
- H0 family6/top-ten AUROC/AUPRC **0.592057/0.209760**.
- H2 cleanup+C7 **0.597871/0.210778**.
- H3 equal+C8 **0.619469/0.225194**.
- H3 minus H0 AUROC **+0.027412 [+0.023675,+0.031091]**, AUPRC **+0.015434 [+0.011378,+0.019404]**.
- H3 minus H2 AUROC **+0.021598 [+0.017653,+0.025457]**, 8/0/0 evaluable error-family wins.
- Status `PRMBENCH_SPECIALIST / NO_PHASE4_PROMOTION`. Labels were historically opened; sealed evaluator lacks `prm_train`/`prm_test` source membership; outcome-selected ancestry still requires independent confirmation. `multi_solutions` is single-class and is undefined, not zero.
- Do not rank the 0.619469 directly against September active-23 0.672619: the input/normalization/reducer contracts differ.

### Scorer-family transfer and Phase-3 closure

Llama H0/H2/H3 macro F1 **0.348909/0.355583/0.353281**. H2-H0 **+0.006674 [-0.007091,+0.020943]**; H3-H0 **+0.004372 [-0.009677,+0.018452]**. Scorer-family transfer on reused questions, not fresh-question confirmation. H2 remains the stronger ProcessBench raw parent. Source: `250e092e:docs/experiments/REASONING_LOCALIZATION_03662_H3_LLAMA_TRANSFER_RESULTS_V1.md`.

Phase 3 final status: all 11 experiments COMPLETE, `PHASE3_DEVELOPMENT_CLOSED__NO_PROMOTION`; later adaptively opened confidence families are retrospective diagnostics, not one jointly confirmatory family. Unrun hierarchy/tensor/residual-STG/Q2-Q4 templates are `NOT_RUN_BY_GATE`, not falsified experiments.

- H2 0.364090; matched donor-cross-fitted equal parent P3E0 0.364284 (not an exact alias).
- Dynamics-IU 0.366876, delta vs equal parent **+0.002592 [-0.001839,+0.007194]**, `PROMISING_UNCONFIRMED`.
- Family-local DUFS 0.367045, increment vs dynamics-IU **+0.000168 [-0.001466,+0.001818]**. No supported graph mechanism.
- STG-SU 0.366762; no supported added mechanism.
- Q1 0.354584: primary macro-F1 inconclusive; exact-error delta **-0.010644 [-0.020123,-0.001210]**.
- Partition-only IU 0.359376; all-family IU 0.359577. More adaptive fusion is not automatically better than equal family compression.

Primary status source: `250e092e:PROGRESS.md`, Step 346; actual per-arm panels and contrasts under `results/reasoning_localization_03662_v1/phase_3/`.

## 5. September Joint L-SML v1: audited negative results

All following sources at `0ce48968:results/joint_lsml_existing_localization_v1/`.

Parent `REPORT.md`: active-23, shared absolute orientation/preprocessing/reducers across arms. Joint learns a residual partition and fits global/group factors. PRMBench K=3, sizes 13/7/3. Misfit **0.203177** vs hard L-SML **0.245183**, a 17.1% reduction. Numerical fit success did not imply better target ranking.

- Joint PRMBench AUROC **0.669063**, interval **[0.662967,0.674967]**, AUPRC **0.251757**.
- IU **0.671539**; Joint-IU **-0.002476 [-0.004103,-0.000908]**.
- Fixed-family continuous L-SML **0.672619**; Joint-fixed **-0.003556 [-0.004572,-0.002578]**.
- Equal-family **0.668774**; Joint-equal **+0.000289 [-0.001620,+0.002271]**.
- 6,208 error responses, 83,280 official steps, 2,000 paired source-group draws. Verdict **HARM**.
- Parent ProcessBench: 7/8 admissible; q4/MATH had no K with minimum three features per group. Frozen all-eight gate produced **STRUCTURAL_NO_SCORE** before labels. This was coverage failure, not a measured accuracy failure.

`processbench_amendment_v1/REPORT.md`: explicitly authorized post-result retrospective coverage policy uses seven Joint heads plus one flat-SML fallback. 3,400 source questions / 6,800 paired model rows, 2,000 paired source-question threshold-refit draws.

| PB arm | Macro F1 |
|---|---:|
| Joint-or-flat | 0.269290 |
| IU-PCR | 0.340378 |
| Fixed-family continuous L-SML | 0.342940 |
| Equal-family | 0.285986 |

- Joint-IU **-0.071087 [-0.084721,-0.054335]**.
- Joint-fixed **-0.073650 [-0.091279,-0.061706]**.
- Joint-equal **-0.016696 [-0.033615,-0.000629]**.
- Verdict **HARM**. Seven fitted-cell selection-conditioned means: Joint 0.298009, IU 0.341730, fixed 0.341834; therefore not just the fallback's failure. This seven-cell diagnostic reuses full-panel thresholds and is not its own complete-panel estimand.

`failure_diagnostic_v1/REPORT.md` identifies two supported explanations, not a causal proof:

1. Final hierarchical head lacks a common score scale, while the panel shares model-level thresholds. q4/MATH and pure-Joint q8/GSM8K account for **89.4%** of summed candidate-IU per-cell loss. Candidate-vs-fixed detector ranks remain highly aligned (Spearman **0.989/0.980**) and locator agreement **0.918/0.887**, but activation falls to **9.6%/14.5%**, versus fixed **67.0%/53.8%**.
2. Structural objective/head mismatch: the covariance model fits global v and group u, while deployed hierarchical weights use global v plus a second SML and omit direct use of fitted u. PRMBench's AUROC loss survives independently of threshold scale.

Signs, removed weak streams, preprocessing drift, reducer drift, convergence, and the threshold implementation were not supported as the main causes. Original freeze did not score grouping x map factorial, so INTERNAL partition and hierarchical map cannot be causally separated from this diagnostic. September v2 work is development; it cannot make these already-opened questions fresh.

Reducer warning: this active-23 PB uses detector **maximum token risk**, locator **fixed top-min(10, step length) mean**; PRMBench uses max token risk within official spans. It is not top-5 or top-10-percent.

## 6. Prefix prediction is not stopping

Certified prefix release at `d9827a83:results/reconstruction_benchmark_v1/releases/2026-08-24_prefix_v1/prefix/A/evaluation/{METRICS.json,CONTRASTS.json}`. Equal macro over GSM8K, MATH, OlympiadBench, OmniMath separately at each token budget; 2,000 paired source-question draws within subset. Outcome is final-answer error on saved prefixes.

| Budget | Unified-28 AUROC | IU28 no length | Step272 global/local |
|---|---:|---:|---:|
| 64 | 0.562949 | 0.587251 | 0.595518 |
| 256 | 0.611410 | 0.648290 | 0.657225 |

Step272-Unified at 64 **+0.032569 [+0.003478,+0.062525]**; at 256 **+0.045815 [+0.014703,+0.076511]**. But Step272-IU28 no length: 64 **+0.008267 [-0.007467,+0.023172]**, 256 **+0.008934 [-0.007835,+0.025013]**, unconfirmed. Only 256-token AUPRC separates from IU28-no-length: **+0.018492 [+0.002328,+0.035166]**. At 512 tokens registered macro AUROC is undefined because a subset is single-class. Per-budget eligible populations change, so the sequence is not a fixed-population longitudinal curve.

CIW early AUROC at 16/32/64/128/256 is **0.563896/0.562387/0.587503/0.611073/0.646165**. It is very close to IU28 and is not an across-budget improvement.

Actual stopping: certified LEASH has six READY callback cells and two Mistral protocol failures. Relative to CoT, all six lose pass@1 while saving tokens. Equal-model AQuA: **0.317585 to 0.217848**, delta **-0.099738 [-0.139108,-0.062959]**, token reduction **0.453000**. GSM8K: **0.625556 to 0.360000**, delta **-0.265556 [-0.305556,-0.223333]**, reduction **0.323523**. This is an accuracy/compute tradeoff, not a detector win or matched-accuracy improvement. Source: current advisor `CLAIM_LEDGER.md`, certified `2026-08-25_leash_v1` release.

## 7. RAG is promising only within clearly named units/populations

CIW on **1,800 Original-30 test responses**: AUROC/AUPRC **0.771222/0.635797**, versus Original-30 IU **0.760523/0.596613** and DUFS **0.762882/0.598308**. Equal QA/Data2txt task macro AUROC **0.630244**, versus DUFS **0.602162**. The large pooled/task-macro gap signals strong task composition effects. Current compact report gives point differences, not a paired interval supporting superiority.

The separate certified evidence-contrast RAGTruth release uses **2,700 responses / 450 source groups**, so its result cannot be subtracted from the 1,800-response CIW table:

- Answer AUROC **0.7273659490 [0.7051984006,0.7488793332]**, error prevalence **0.349259**.
- Sentence AUROC **0.6891729751 [0.6684661455,0.7104480643]**, 17,747 sentences / 450 groups, prevalence **0.087902**.
- Token AUROC **0.6586875328 [0.6341182484,0.6841876694]**, 430,202 tokens / 450 groups, prevalence **0.042210**.
- All use 20,000 source-group draws. Hundreds of thousands of tokens are not hundreds of thousands of independent examples.

Local GASP **0.6708** vs fixed **0.6597**, delta **+0.0111 [-0.0125,+0.0343]**, no demonstrated superiority. Lettuce F1 **0.7929** is a supervised ceiling. RefChecker accurate/noisy/zero settings remain separate, using fixed claims; claim extraction is outside the run. Sources: `d9827a83:results/reconstruction_benchmark_v1/releases/2026-08-25_rag_evidence_v1/rag_evidence/A/evaluation/metrics.csv`; advisor `CLAIM_LEDGER.md`.

## 8. Evidence-level conclusions for the parent review

- The project has a large, reproducible comparative evidence base, but most current method evidence is retrospective. Frozen scores prevent new label leakage; they do not undo past adaptive development.
- The most consequential simple baseline is **equal provenance-family averaging**, which nearly matches B3 and outperforms several sophisticated global fusion heads. Do not omit it.
- Dense PRMBench step ranking and ProcessBench first-error F1 are different tasks. H3's supported specialist result coexists with unconfirmed PB transfer. The new Joint v1 result is a separate matched negative.
- Better covariance/graph fit is repeatedly dissociated from better target ranking. The Joint failure gives both a concrete engineering issue (scale transfer) and an objective/readout issue.
- Current token-only and response-plus-token adapters must both remain visible: advisor ledger reports token-only IU exceeds response+token IU on PRMBench by **+7.23 AUROC percentage points / +4.62 AUPRC points**. Global response signal can dilute local evidence.
- Historical 0.3662, current fair 0.3261, Phase3 0.3641, active23 0.3429, and PRMBench 0.6195/0.6726 are not one leaderboard. Every headline needs its cohort, input contract, reducer, calibration regime, sample unit, and evidence status.
- CIW RAG response gains are an application signal, not evidence of localization or a universal CIW improvement. Task composition and paired uncertainty remain central to interpretation.
- Neither high reproducibility nor many tests constitute fresh scientific confirmation. The September running study must retain that boundary even if its code and score-scale repair work correctly.
