# Reconstruction Benchmark v1

**Protocol date:** 2026-08-24
**Status:** frozen protocol; no new scientific headline has been issued
**Main score direction:** higher means more likely to be incorrect

## 1. Purpose

This benchmark has two goals:

1. Rebuild the corrected 24-cell static-fusion study from one frozen feature matrix and run the same 13 methods under one contract.
2. Put the other research tracks into one honest reporting package: response detection, reasoning localization, early detection, multi-sample inference, stopping, RAG, and white-box measurements.

The first goal has executable fitting and evaluation code. The second goal now
also has a strict A/B runner for compatible external final-answer populations,
plus the audited population/comparator inventory and reporting framework. It
does **not** yet have one executable adapter for every localization, prefix,
stopping, RAG, multi-sample, or white-box lane.

This document freezes the human-readable protocol. Exact IDs, hashes, formulas, and fixed parameters remain machine-readable in:

- `configs/reconstruction_benchmark_v1/feature_contract.json`
- `configs/reconstruction_benchmark_v1/frozen24_cells.json`
- `configs/reconstruction_benchmark_v1/methods.json`
- `configs/reconstruction_benchmark_v1/populations.json`
- `configs/reconstruction_benchmark_v1/comparators.json`
- `configs/reconstruction_benchmark_v1/external_final_answer.json`

If this document and an executable contract disagree, stop the release. Do not choose the version that gives the better result.

## 2. Claims boundary

### 2.1 The frozen 24 cells are development data

The corrected 24-cell population has 48,607 responses: 26,667 incorrect and 21,940 correct, across 9 QA cells and 15 mathematics cells.

The fit is label-free at run time, but the cells were inspected while the feature transforms and earlier methods were developed. This is therefore **D0 reused-development evidence**, not untouched validation. A rebuilt score can show that the code and claim are reproducible. It cannot turn D0 evidence into prospective confirmation.

### 2.2 One pass means one saved generation

For the static-fusion roster, each row is one already-generated answer or reasoning trace. The methods use saved output-probability telemetry. They do not generate another answer and do not read correctness while fitting.

Other panels use different access contracts and must stay separate. For example, the fixed reasoning and RAG pipelines use teacher-forced scoring passes; white-box methods read internal layers; EDIS and DeepConf use hundreds of sampled generations. These are not one-pass gray-box comparisons.

### 2.3 No number is a result merely because it exists

A row may be rankable only when its population, rows, labels, unit, metric, access, and method version are registered. Historical or published numbers with an incomplete match remain context only.

## 3. Two kinds of reconstruction

### 3.1 Matrix-level reconstruction

This is the executable frozen-24 study. It starts from:

`results/dependency_fusion_raw/cells.npz`

The source is a saved consolidated matrix. The reconstruction:

- checks its frozen SHA-256;
- reads the saved measurement matrix, feature pool, and registered signs;
- applies mixed-v2 preprocessing exactly once;
- fits the 13 methods;
- freezes scores before opening labels;
- evaluates only after source-group identity has been verified.

This level does **not** rerun an LLM and does not prove raw-generation identity for every historical row. Its row identity is the stable consolidated-matrix order. A separate fair-comparison ledger has strict raw-row identity for only six cells.

### 3.2 Model regeneration or raw-feature rebuilding

This is a different operation. It reopens prompts, models, tokenizers, generation settings, graders, and feature extraction. It is needed only when the scientific question requires fresh generations or when a required raw feature cannot be recovered.

A regenerated population must receive a new population and release ID. Stochastic generations are not expected to be byte-identical to the old cache. They must never be silently merged with matrix-level reconstruction.

Drive materialization is not regeneration. Copying an already-frozen EDIS or DeepConf cache from its registered read-only Drive path preserves the old generations; rerunning the model does not.

## 4. Frozen mixed-v2 feature contract

Let $X\in\mathbb{R}^{n\times p}$ be one cell's saved measurement matrix. A column is a measurement of confidence, such as entropy-, energy-, top-$K$-, or trace-based evidence. The nominal pool has 30 columns, but a cell may have fewer.

The contract ID is `dufs-liu-mixed-v2-development-2026-08-07`.

### 4.1 Apply the transform once

Every present column receives its registered confidence sign and population standardization. Then:

- ordinary measurements remain standardized confidence measurements;
- `pe_mean` is mapped to the negative square of its standardized value;
- STFT spectral entropy is mapped to the negative absolute percentile distance from a label-free KDE-mode percentile;
- CUSUM shift index and RPDI keep their standardized signed values.

The result is a confidence-oriented matrix. Some spectral solvers still identify their *whole fused score* only up to one global sign. A single frozen label-free rule resolves that sign: correlate the fused confidence score with the equal-family mean of the prepared confidence coordinates, require $|r|>10^{-6}$, and flip the whole score when the correlation is negative. This never changes an individual feature. The aligned confidence score is then negated once to risk, so a larger final score means “more likely incorrect.”

The contract pins both the mixed-v2 transform wrapper and the underlying orientation registry in `spectral_utils/feature_contract.py` by SHA-256. Every prepared manifest repeats those hashes, so a sign-map change cannot reuse an older matrix under the same name.

The following are forbidden:

- orienting a feature a second time;
- adding `sign(rho)` as a headline arm after mixed-v2 has already oriented the columns;
- dropping rows or features because of labels;
- loading labels during transformation or fitting;
- replacing a missing feature with an imputed value;
- shrinking every cell to an easier common feature subset;
- silently changing the feature contract.

### 4.2 Target-free prepared files

Each prepared cell archive contains exactly:

`X_confidence`, `feature_names`, `family_ids`, `row_ids`, and `row_index`.

It must contain no target-like field. The matrix must be finite, have at least three rows and three features, follow the canonical feature order, be centered, and have population standard deviation one unless a column is constant zero.

The prepared `row_id` binds the consolidated matrix row. It is not, by itself, proof of the original prompt or generation identity.

## 5. The 13 static-fusion methods

All 13 methods receive the same prepared matrix. Let $x_j$ be one confidence-oriented feature, $C$ the feature covariance, $z(\cdot)$ population standardization, $L$ a normalized graph Laplacian, and $s$ the final incorrectness score.

| ID | Plain definition | Frozen mathematical form | Main assumption or limit |
|---|---|---|---|
| `equal_feature_mean` | Give every available feature equal weight. | $s=-\lvert J\rvert^{-1}\sum_{j\in J}x_j$ | A large feature family gets more total weight. |
| `equal_family_mean` | Average inside each of six manual provenance families, then average the families. | $s=-G^{-1}\sum_g \lvert J_g\rvert^{-1}\sum_{j\in J_g}x_j$ | The six-family map is a manual prior. |
| `continuous_lsml` | Cluster dependent measurements, fuse within each cluster, then fuse the cluster scores. | $z_g=F_gw_g,\ s=Za$ | L-SML theory was written for binary classifiers; this is a continuous adaptation. |
| `dufs_pf_lsml` | Use parameter-free DUFS gates to select smooth features, then run Continuous L-SML. | $\hat J=\{j:\bar\mu_j>0\},\ s=L\text{-}SML(X_{\hat J})$ | Graph smoothness need not mean relevance to correctness. |
| `dufs_stability_lsml` | Choose the DUFS penalty whose selected features agree most across five seeds, then run L-SML. | $\lambda^*=\arg\max_\lambda\operatorname{mean}J(\hat J_{\lambda,a},\hat J_{\lambda,b})$ | A nuisance can be stable too. |
| `upcr` | Estimate relation to one hidden target from covariance, remove weak features, refit, and fuse the survivors. | $C_{ij}\approx\rho_i+\rho_j-g^2$, followed by projected covariance weighting | Conditional feature errors are assumed pairwise uncorrelated. The registered deployed-style exclusion path is used. |
| `iu_pcr` | Use the same U-PCR estimate but keep the full pool and two covariance components. | $w=U_2(U_2^TCU_2)^{-1}U_2^T\hat\rho,\ s=-Xw$ | This is the matched non-graph control for extensions. |
| `dufs_liu` | Build a nearest-neighbour graph from DUFS-gated features and penalize IU-PCR weights that are rough on it. | $R=X^TLX/n,\ w_\lambda=U[U^T(C+\lambda\bar R)U]^{-1}U^T\hat\rho$ | A healthy graph may follow length or another nuisance. |
| `su_pcr` | Split covariance into a low-rank shared part and sparse dependent errors before PCR. | $C=L_{low}+S_{sparse}$ | The dependence correction is useful only if violations are sparse. This is the 2022 follow-up in the U-PCR line. |
| `ca_specrage_atomic` | Give near-duplicate families equal prior mass, estimate which single-feature neighbourhoods agree, fuse their graphs, then regularize IU-PCR. | $W=\sum_j\alpha_jW_j$, followed by LIU | Agreement may reward repeated nuisance. Uniform, prior, global, and permuted controls are required. This is a new atomic adaptation, not the old LOCO-micro arm. |
| `deem_b3` | Fit a nonlinear energy model to the continuous feature vector and use its latent posterior as risk. | $E(x)=\tfrac12\lVert x-a\rVert^2-\operatorname{softplus}(\ell_\theta(x))$ | This is our continuous additive adapter inspired by the 2026 DEEM paper, not a reproduction of its hard multinomial/iRBM model. |
| `family_nrm_a` | Remove the shared IU-PCR direction from six family contributions, choose a neutral residual covariance mode, and add a small correction. | $r_g=z(h_g)-\operatorname{Proj}_{z(b)}z(h_g)$; choose the eigenmode nearest eigenvalue 1 | New, unrun, donor-free ablation. The family map and nearest-to-one rule are priors. |
| `pgrd_a` | Build a graph in residual family space and move IU-PCR in the residual direction that most lowers graph roughness. | $c=R^TLb/n,\ d=-c,\ s=-[z(b)+G^{-1}Rd/sd(Rd)]$ | New, unrun, donor-free ablation. Lower roughness alone does not identify correctness. |

`IU-PCR` is the reference method for paired static-fusion contrasts.

### 5.1 Fallback rule

A registered fallback is allowed only when the method definition names it. It must be reported as `OK_FALLBACK` with a non-empty reason. Graph, sparse-decomposition, and DEEM arms must fail closed where their registry says that no IU-PCR substitution is allowed.

### 5.2 A, B, and C variants

The letters describe where the rule comes from. They are most important for Family-NRM and PGRD.

- **A — within-cell, fully unsupervised:** fit only on the target cell, with no donor cells and no labels. `family_nrm_a` and `pgrd_a` are the two new A variants in the primary roster.
- **B — donor-unsupervised:** learn a rule from unlabeled donor cells while holding out the target family. `family_nrm_b` and `pgrd_b` are secondary and still need implementation.
- **C — donor-label selection:** use donor labels to select the rule. This is a supervised model-selection ceiling, never part of the label-free leaderboard. `family_nrm_c` needs implementation and `pgrd_c` is a secondary rerun.

The confirmed PRMBench method `NRM-CS-IU` is a historical frozen cross-dataset method. It is **not** the same algorithm or evidence claim as the new within-cell `Family-NRM-A`. Their names must not be collapsed in a table.

### 5.3 Secondary and historical arms

GroupFS-L-SML and DEEM-B1/B2 are secondary reruns. GOOD-5, GOOD-6, and LOCO-5 are label-informed references only.

The old `sign(rho)` arm, the 111-selector sweep, clustered U-PCR, SDSF, and residual-graph DEEM remain historical. They may be discussed as mechanism evidence, but they are not silently added to the 13-method primary freeze.

## 6. Frozen-24 scientific workflow

### Stage 1: prepare two independent target-free builds

Run mixed-v2 independently as build A and build B. Each build writes one manifest and 24 cell archives. `PREPARED_AB_VERIFICATION.json` must say that every cell is byte-identical and semantically identical across builds.

### Stage 2: fit before labels

Fit the exact ordered 13-method roster on both builds. The fitting command has no label argument and does not import the evaluator.

Before the first method starts, each build writes an immutable `FIT_SOURCE_SNAPSHOT.json` that binds the clean Git commit, executable/config hashes, input manifest, requested cells, and requested methods. A resume must reproduce that record exactly. A build without this pre-fit record cannot be resumed, and a build that already has `SCORE_FREEZE_MANIFEST.json` cannot be run again. The source snapshot is checked again at the end; any mid-run code change blocks the score freeze.

A full build has $24\times13=312$ cell-method records. It receives `SCORE_FREEZE_MANIFEST.json` only if every record has a score and status `OK` or `OK_FALLBACK`. A partial cell or method run is a debug run and writes `DEBUG_RUN.json`. A failed full run writes `FIT_INCOMPLETE.json`; it is not a scientific freeze.

### Stage 3: verify A against B

Rehash all inputs, records, scores, and artifacts. `SCORE_AB_VERIFICATION.json` is issued only when all 312 A/B records are byte-identical and use the same source snapshot.

### Stage 4: verify groups before opening labels

Every cell needs a separate audited sidecar that maps each matrix row to an independent source group. Allowed units are:

- `source_question_id`
- `source_prompt_id`
- `source_item_id`
- `problem_id`

The sidecar must be `VERIFIED`, say that labels were not used, preserve exact row order, contain at least two non-empty groups, bind the source and matrix hashes, and pass source-hash, row-count, row-order, and group-semantics checks. Reusing row IDs as independent bootstrap groups is forbidden.

### Stage 5: open labels and evaluate

Only after stages 1–4 pass may the evaluator open `<cell>__labels`. It converts `y_correct` to `y_error=1-y_correct` and computes:

- AUROC as weighted Mann-Whitney probability with half credit for ties;
- AUPRC as weighted non-interpolated average precision, following the scikit-learn convention.

The primary point estimate is the equal-cell mean AUROC across all 24 cells. AUPRC and per-cell results are secondary.

### Stage 6: package the result

The scientific evaluation writes an immutable evaluation directory. A separate reporting build converts registered producer tables into tidy files, a DuckDB database, static plot data, and one self-contained HTML report.

## 7. Bootstrap and grouping rules

### 7.1 Frozen 24-cell headline

The canonical evaluator uses 20,000 draws and base seed `20260824`. For each cell it:

1. samples verified source groups with replacement;
2. keeps all sibling rows from a sampled group together;
3. uses one PCG64 draw stream shared by all 13 methods and all paired contrasts;
4. computes each cell metric;
5. uses the same draw index across cells, then takes an equal-cell mean.

Inference is conditional on the 24 frozen cells. Cells, datasets, and model families are not resampled.

A single-class cell receives `METRIC_UNDEFINED_SINGLE_CLASS`, a null metric, and no valid interval. One bad component invalidates that aggregate draw.

### 7.2 Other panels

Do not force the frozen-24 bootstrap onto another task. Keep the population's registered rule:

- ProcessBench Llama response, localization, and prefix: 2,000 paired source-question draws within subset.
- ProcessBench fixed Qwen localization: 100 paired 50/50 calibration/evaluation splits; this is not a bootstrap.
- PRMBench response Family-NRM comparison: 5,000 paired source-group draws over 6,208 registered groups.
- SemGrad and HLE: 20,000 question-level draws under their registered stratification.
- LEASH: 2,000 paired dataset-question draws.
- RAGTruth response and sentence, and the GASP cohort: 1,000 source-ID draws.
- RAGTruth token results currently have no interval. Never treat 430,202 tokens as independent rows.
- White-box versus gray-box exact intersection: 20,000 problem-group-within-cell draws, then equal-cell aggregation.
- A null draw count means no interval is frozen. Do not invent one during reconstruction.

## 8. Population panels and readiness

`READY` means that the frozen evaluation ledger is auditable. It does not always mean that raw features are present in this worktree. The exact raw-cache boundary is stated below.

### 8.1 Static fusion, reasoning, and external response detection

| Population | Exact unit/count | Registered use | Readiness |
|---|---|---|---|
| `frozen24_response_v1` | 48,607 responses; 24 cells; 26,667 incorrect, 21,940 correct | D0 equal-cell static-fusion reconstruction | `RETROSPECTIVE_ONLY`; matrix present, source-group sidecars still required for the new strict evaluator |
| `processbench_llama31_response_v1` | 3,400 responses; 1,700 final-answer errors, 1,700 correct | Response AUROC/AUPRC | `READY` ledger; raw cache absent here |
| `processbench_llama31_first_error_v1` | 3,400 responses; 2,221 with an official error step, 1,179 clean | First-error localization | `READY` ledger; raw cache absent here |
| `processbench_qwen3_fixed_first_error_v1` | 3,400 responses scored by two Qwen models; 6,800 scorer-response rows | Fixed teacher-forced localization | `READY_WITH_LIMITATIONS`; repeated splits, not an independent CI; raw files are absent or LFS pointers |
| `processbench_llama31_prefix_v1` | 1,717 trace union; 9,277 prefix observations; 967 traces complete at all six budgets | Causal early scoring, not stopping | `READY` ledger; raw cache absent here |
| `prmbench_qwen3_response_v1` | 6,969 raw, 3 excluded, 6,966 evaluated; 6,208 error and 758 control responses | Response detection and confirmed `NRM-CS-IU` comparison | `READY` ledger; raw cache absent here |
| `prmbench_qwen3_native_steps_v1` | 83,280 evaluated steps from 6,966 responses | Native step detection | `READY` ledger; interval draw count not frozen |
| `prmbench_qwen3_error_classes_v1` | 6,208 error responses, nine classes; 83,280 evaluated steps | Error-class slices | `READY_WITH_LIMITATIONS`; strata have different sizes |
| `prmbench_multi_solutions_steps_quarantine_v1` | 2,241 valid steps and 0 erroneous steps | Single-class diagnostic only | `PROTOCOL_GATE_FAILED`; no standalone binary AUROC/AUPRC |
| `semgrad_bem_v1` | 1,817 responses: SciQ 1,000 and TruthfulQA 817 | External response check with BEM labels | `READY` ledger; raw cache absent here; background evidence only |
| `hle_qwen72b_interim_v1` | 2,158 responses; 2,090 incorrect, 68 correct | External transfer check | `READY_WITH_LIMITATIONS`; uses an interim local judge, not the official grader |
| `evidence_drop_primary_v1` | 5,638 reasoning traces; 825 incorrect, 4,813 correct | Selective reasoning/evidence-drop study | `READY_WITH_LIMITATIONS`; raw cache absent and MATH has a truncation limitation |
| `evidence_drop_gsm8k_qwen3_8b_pilot_v1` | 30 traces; 1 incorrect | Pilot mechanics only | `PROTOCOL_GATE_FAILED` |
| `aqua_cot_central_three_models_v1` | 762 responses: 254 questions on each of three models; 520 incorrect, 242 correct | Retrospective external/mechanism audit and stopping context | `RETROSPECTIVE_ONLY`; no response bootstrap frozen |

The ProcessBench response and localization populations use the same 3,400 traces but different labels. Final-answer wrongness is exactly 1,700/1,700. Official first-error localization is 2,221/1,179. They must never be substituted for each other.

### 8.2 Negative-stress panels

| Population | Exact unit/count | Readiness and allowed claim |
|---|---|---|
| `gpqa_k10_negative_stress_v1` | 7,920 traces; 198 questions × 10 samples × 4 models; 5,113 incorrect, 2,807 correct | `READY_WITH_LIMITATIONS`; negative stress only, with no frozen grouped interval |
| `lciteeval_negative_stress_v1` | 4,400 coarse response rows; 20 cells; 4,019 ungrounded, 381 grounded | `PROTOCOL_GATE_FAILED`; 19/20 cells fail balance and the stored label is not canonical citation-level LCiteEval |

### 8.3 EDIS, AIME, DeepConf, and other multi-sample panels

| Population | Exact acquisition | Readiness and boundary |
|---|---|---|
| `edis_aime24_full_v1` | 5,760 traces = 30 questions × 64 samples × 3 temperatures (0.2, 0.6, 1.0); 60 correct, 5,700 incorrect | Complete on Drive, but `PROTOCOL_GATE_FAILED`: every stored balance gate fails. Problem IDs are only integers 0–29 and dataset revision is not frozen. Descriptive/mechanics use only. |
| `edis_amc23_full_v1` | 3,840 traces = 40 questions × 32 samples × 3 temperatures; 447 correct, 3,393 incorrect | Complete on Drive, but `PROTOCOL_GATE_FAILED`: every stored balance gate fails. Problem IDs are only integers 0–39. |
| `edis_pilot3_v1` | 2,160 traces: GSM8K, MATH-500, and AMC23 each have 30 questions × 8 samples × 3 temperatures = 720 | `PROTOCOL_GATE_FAILED` for all three grids; local raw files are absent or LFS pointers. GSM8K also used a known unboxed-answer grading deviation. |
| `edis_aime24_legacy_demo_v0` | 720 traces; exact class audit unavailable | `QUARANTINED`; superseded by the 5,760-row full grid |
| `deepconf_aime24_k512_v1` | 15,360 traces = 30 AIME-2024 questions × 512; 0 generation failures; 24 shards; Qwen3-8B, T=0.6, top-p=0.95, top-k=20, max-new-tokens=32,768 | `READY_AFTER_DRIVE_MATERIALIZATION`; 361 objects, 20,189,077,984 bytes. Raw acquisition is complete, but correctness counts and offline evaluation are not frozen. Supports budgets only through 512, not the paper's 4,096. |
| `deepconf_aime24_k4096_partial_v1` | 12,370 of intended 122,880 traces; only four questions have rows and only two reach 4,096 | `INCOMPLETE`; 297 objects, 17,744,439,979 bytes. Acquisition appendix only. Never merge with K=512. |
| `math500_phase15_temperature_runs_v1` | 1,800 traces over 200 questions: five temperatures in run 0 plus four extra T=1 runs | `RETROSPECTIVE_ONLY`; exact extra-run label totals and original interval are not frozen |
| `math500_phase15_same_temperature_k5_v1` | 1,000 traces = 200 questions × 5 T=1 samples | `RETROSPECTIVE_ONLY`; raw cache absent here; per-trace correctness counts not frozen |
| `math500_phase15_multi_temperature_k5_v1` | 1,000 traces = 200 questions × five temperatures | `RETROSPECTIVE_ONLY`; raw cache absent here |

The EDIS full grids and DeepConf K=512 do not require new model generation for an offline reconstruction. They require read-only Drive materialization, a frozen grader/evaluator, and a new output manifest. Their current readiness does not permit a headline performance claim.

### 8.4 Stopping

| Population | Exact unit/count | Readiness and allowed claim |
|---|---|---|
| `refrain_math500_qwen3_8b_v1` | 500 questions × vanilla/REFRAIN = 1,000 traces; vanilla 453 correct, REFRAIN 413 correct | `READY`; local study is complete, but the older fair package is stale and must be rebound before a paired release claim |
| `leash_six_complete_cells_v1` | 4,986 traces: 3 models × (AQuA 254 + GSM8K 300 questions) × 3 arms | `READY`; local paper-specified-partial study, with 2,000 paired question draws |
| `leash_mistral_failed_cells_v1` | 1,662 attempted traces and 0 usable | `PROTOCOL_GATE_FAILED`; explicit failed coverage row only |

### 8.5 RAG

| Population | Exact unit/count | Readiness and boundary |
|---|---|---|
| `ragtruth_test_response_v1` | 2,700 responses; 450 source groups; 943 hallucinated, 1,757 clean; 16,200 evidence conditions | `READY_WITH_LIMITATIONS`; score ledgers present, raw cache absent; multi-pass evidence-contrast access |
| `ragtruth_test_sentence_v1` | 17,747 sentences; 1,560 hallucinated | `READY_WITH_LIMITATIONS`; 1,000 source-ID draws |
| `ragtruth_test_token_v1` | 430,202 scorer tokens; 18,159 hallucinated | `READY_WITH_LIMITATIONS`; no token CI is frozen |
| `gasp_ragtruth_balanced400_v1` | 400 responses, 228 source groups, 2,714 sentences, 287 hallucinated sentences | `READY_WITH_LIMITATIONS`; local protocol reproduction, not paper-exact sample identity |
| `refchecker_knowhalbench_claim_v1` | 10,733 claims; 7,176 entailment, 2,622 neutral, 935 contradiction | `READY_WITH_LIMITATIONS`; local three-way NLI and project binary unsupported-claim metrics are different targets |
| `lettucedetect_ragtruth_full_v1` | 2,700 RAGTruth responses with frozen span predictions | `READY`; supervised span method has a different unit, metric, and access level from gray-box AUROC |

### 8.6 White-box and quarantine

| Population | Exact unit/count | Readiness and boundary |
|---|---|---|
| `whitebox_graybox_exact_intersection_v1` | 31,440 exact common rows; 13 primary cells, 8 model families | `RETROSPECTIVE_ONLY`; matched comparison is post-hoc, not validation |
| `whitebox_graybox_coverage_v1` | 47,265 raw candidates; 47,238 evaluable; 42,238 white-box scorable; 31,467 gray-box complete; 31,440 common | `RETROSPECTIVE_ONLY`; all 28 exact source files exist outside this worktree and are rebuildable |
| `coqa_llama7b_protocol_rejected_v1` | 5,000 rows from 500 questions × 10 candidates; 4,338 incorrect, 662 correct | `QUARANTINED`; the base model received an invalid chat template. It is appendix/mechanism context only, never a benchmark cell. |

## 9. Materialization rules

As audited on 2026-08-24:

- the frozen 24-cell matrix is fully materialized in this worktree;
- many ProcessBench, PRMBench, SemGrad, HLE, RAG, negative-stress, and Phase-15 lanes have local score ledgers but their raw caches are absent or Git-LFS pointers;
- EDIS AIME full, EDIS AMC full, and DeepConf K=512 are complete on Drive only;
- the exact white-box source set is available outside this worktree under `/Users/osegev/Desktop/hallucination_detection_whitebox_layer_fusion/`; its prepared NPZ files are absent but can be rebuilt;
- readiness of a score ledger does not authorize a raw-feature rerun without restoring and verifying its source cache.

All Drive inspection is read-only until exact source path, size, manifest, and destination have been checked. Never merge incomplete and complete sampling lattices.

## 10. Comparator matching

A direct numerical subtraction is allowed only when all relevant axes are established:

1. dataset revision;
2. model;
3. exact row IDs or cohort identity;
4. generation trace or generation protocol;
5. grader and labels;
6. prediction unit;
7. metric and positive class;
8. access contract.

Score direction must also be normalized explicitly. AUROC can be direction-inverted, but AUPRC must be recomputed after changing the positive class; it cannot be copied from a correct-positive report into an error-positive table.

Each match axis is one of `EXACT`, `COMMON_REPLAY`, `ADAPTED`, `DIFFERENT`, `PARTIAL`, `UNKNOWN`, or `NOT_APPLICABLE`.

- `EXACT` supports a direct comparison on that axis.
- `COMMON_REPLAY` means that a comparator was recomputed on the same frozen rows under one local protocol. It is not automatically a paper-exact reproduction.
- `ADAPTED`, `DIFFERENT`, `PARTIAL`, or `UNKNOWN` must be visible and usually block direct subtraction.

Examples:

- The local Mind-the-Gap score is an adapted common-row replay, not the authors' released system.
- GASP is a local protocol reproduction without the paper's sample IDs.
- LettuceDetect and RefChecker may use related rows, but their unit, target, metric, or access differs from gray-box response detection.
- DeepConf K=512 is not matched to the published K=4096 result.
- Semantic Entropy's published numbers and the legacy TSV bundle remain context only because no verified TSV-to-cell mapping exists.
- A published comparator table must never be averaged into a synthetic “published 24-cell macro” when the rows are not identical.

The complete per-comparator decision is in `comparators.json`. Its “executable or scored” section means that an implementation or frozen ledger exists under the stated conditions; it does not mean every source cache is present in this checkout.

## 11. Graph assumption checks

A graph can be numerically stable and still encode a nuisance. Graph health is therefore not evidence of target alignment.

### 11.1 Before labels

For each graph-producing method, freeze:

- method version and parameters;
- prepared matrix hash;
- graph hash and graph variant;
- number of nodes and edges;
- finite/topology/solver health diagnostics;
- any registered stability measure;
- all graph-derived scores and artifacts.

`label_stage` must be `label_free` for these checks.

### 11.2 After score and graph freeze

Only then may post-freeze diagnostics ask whether the graph follows correctness or a nuisance. At minimum, graph claims must address:

- the paired method gain against the matched IU-PCR non-graph control;
- a real-graph comparison with every registered null or control;
- target alignment versus known nuisance alignment, including trace length where available;
- robustness across cells rather than a chosen favorable example.

For CA-SpecRaGE, the required controls are uniform, provenance-prior, global, and permuted graphs. For DUFS-LIU and PGRD-A, lower roughness by itself is not a pass. A graph contribution can be claimed only if the registered assumption diagnostic and the paired utility contrast both support it.

Graph diagnostics must record `value`, `null_value`, `effect`, optional `p_value`, permutation count, node/edge counts, hashes, and whether labels were still closed or opened after freeze. Any example visualization must be chosen by a preregistered label-free health rule, not by AUROC.

The graph-diagnostics producer and reporting bridge now enforce the registered
controls, the relation `effect = value - null_value`, the graph/matrix/source
hashes, label stage, and exact panel coverage. Raw permutation draws remain a
signed auxiliary; only their registered summaries enter the comparable table.

## 12. Status rules

Only `OK` and `OK_FALLBACK` are rankable.

Reporting may also use:

- `NOT_APPLICABLE`
- `NOT_RUN`
- `ADAPTER_MISSING`
- `BLOCKED_ASSET`
- `INPUT_INVALID`
- `FIT_FAILED`
- `SCORE_INCOMPLETE`
- `METRIC_UNDEFINED_SINGLE_CLASS`
- `EXCLUDED_BY_PROTOCOL`
- `QUARANTINED`
- `UNVERIFIED`
- `CONTEXT_ONLY`

Every expected system × cell × slice must have one coverage row. A missing result is a named status, never an omitted row and never zero.

`CONTEXT_ONLY` and `UNVERIFIED` may carry a descriptive number. Other non-rankable statuses carry no metric value. `OK_FALLBACK` must say exactly which frozen fallback ran and why.

Readiness values such as `READY_AFTER_DRIVE_MATERIALIZATION` and comparator execution labels such as `SCORED_REBIND_REQUIRED` are inventory metadata. They are not result statuses and require an explicit conversion when building the final reporting tables.

## 13. Release trees

There are currently two separate release builders.

### 13.1 Scientific frozen-24 tree

```text
<scientific-release>/
  PREPARED_AB_VERIFICATION.json
  SCORE_AB_VERIFICATION.json
  build_A/
    inputs/
      MANIFEST.json
      cells/<cell>.npz
    fit/
      FIT_SOURCE_SNAPSHOT.json
      SCORE_FREEZE_MANIFEST.json
      cells/<cell>/
        CELL_FIT_MANIFEST.json
        <method>/
          RECORD.json
          score.npz
          ARTIFACT_INDEX.json
          artifacts.npz                 # when the method has artifacts
  build_B/
    ...                                 # same structure as build_A
  evaluation/
    EVALUATION.json
    BOOTSTRAP_DRAWS.npz
    EVALUATION_MANIFEST.json
  group_sidecars/
    GROUP_SIDECARS.json
    cells/<cell>.npz
    evidence/<cell>.json
  graph_diagnostics/
    GRAPH_DIAGNOSTICS.json
    GRAPH_DIAGNOSTICS_MANIFEST.json
    PLOT_DATA.npz
    EXAMPLE_GRAPH_DATA.npz
  reporting_inputs/
    research_registry.json
    predictions.jsonl
    metrics_long.csv
    contrasts_long.csv
    coverage_long.csv
    graph_diagnostics_long.csv
```

The group manifest, group sidecars, and their identity evidence are mandatory
evaluation inputs and live inside this immutable scientific release. The
evaluator hash-binds them before opening labels.

### 13.2 Reporting tree

```text
<reporting-release>/
  REPORTING_MANIFEST.json
  01_registries/
    research_registry.json
  05_evaluation/
    predictions.parquet
    metrics_long.csv
    metrics_long.parquet
    contrasts_long.csv
    contrasts_long.parquet
    coverage_long.csv
    coverage_long.parquet
    benchmark.duckdb
  06_diagnostics/
    graph_diagnostics_long.csv
    graph_diagnostics_long.parquet
  07_reports/
    REPORT.html
    plot_manifest.json
    plot_data/<plot_id>.csv
```

The reporting build is immutable and atomic. The directory name must equal its `release_id`, and an existing release is never overwritten.

## 14. Acceptance gates

A release passes only if all gates relevant to its scope pass.

### Gate A — source and registry freeze

- Frozen source, manifest, transform, roster, method versions, and configs match their SHA-256 values.
- The 24 unique cells are in the registered order and domain.
- The D0 disclosure remains present.

### Gate B — one target-free preprocessing

- No label or target-like field enters fitting.
- Mixed-v2 runs once.
- Missing features remain missing.
- Prepared matrix semantics and score conversion are explicit.

### Gate C — independent preparation

- Builds A and B are byte- and semantic-identical for all 24 cells.

### Gate D — complete fitting

- The exact 13-method order, versions, and config hashes match the executable roster.
- All 312 records have a score and status `OK` or registered `OK_FALLBACK`.
- No silent method substitution occurs.
- Partial runs remain debug artifacts.

### Gate E — score freeze reproducibility

- All 312 score, record, and artifact hashes are identical across A and B.
- Both builds use the same source snapshot.

### Gate F — independent-group identity

- Every cell has a verified, label-free group sidecar with exact row binding and at least two real source groups.
- Row-IID bootstrap is rejected.

### Gate G — label opening and metrics

- Labels are opened only after gates A–F.
- Positive class is incorrect.
- AUROC/AUPRC definitions and one global score conversion are fixed.
- Single-class rows are explicit and non-rankable.

### Gate H — headline completeness

- The static headline uses all 24 cells, all 13 methods, and exactly 20,000 canonical grouped draws.
- The primary estimate is the equal-cell macro AUROC.
- Paired contrasts use IU-PCR as reference and the same bootstrap draws.
- If any component is incomplete or noncanonical, `headline_macro24_auroc` stays empty.

### Gate I — external-lane identity

- Each external result names its population, exact cohort, unit, label/grader, metric, access, evidence grade, and fidelity.
- Its own grouping and interval contract is preserved.
- Missing assets and failed protocol gates remain explicit.

### Gate J — comparator fairness

- Every direct subtraction passes the registered match axes.
- Common replay and adapted reproduction are named as such.
- Published context is not ranked with executable common-row systems.

### Gate K — graph mechanism

- Graph and matrix hashes are frozen before target checks.
- Required real/null/nuisance diagnostics exist.
- Any graph improvement is paired against IU-PCR on the same cohort.
- Stability or lower roughness alone is not presented as target identification.

### Gate L — reporting package

- One validated `reconstruction_registry_v1` and all five producer tables exist: predictions, metrics, contrasts, coverage, and graph diagnostics.
- Every expected coverage row exists.
- Comparison groups share one exact estimand.
- Registered equal-unit means and plot source hashes reproduce.
- A second clean reporting build is byte-identical for every canonical
  artifact. DuckDB's physical container bytes are excluded because the engine
  may choose a different file layout; its registered source hashes, view SQL,
  schemas, constraints, row counts, and bidirectional `EXCEPT ALL` results must
  instead be logically identical.

## 15. Current implementation gaps

These are blockers, not optional cleanup:

1. **Population ID:** resolved before launch. Scientific and reporting contracts both use `frozen24_response_v1`.
2. **Matrix semantics:** resolved before launch. Every executable contract now uses `higher_is_confidence`; the final score is converted once to `higher_is_incorrect`.
3. **Bootstrap identity:** resolved for the 24-cell release. Seventeen single-generation cells use a verified singleton source partition. For the seven repeated-generation cells, the builder verifies raw file size/hash/manifest, reproduces the historical candidate order, matches all 17 common core columns into `cells.npz`, and then proves exact equality with the prepared `mixed-v2` matrix. The evaluator never resamples matrix rows as if they were independent.
4. **Release layout:** the scientific release owns `results/reconstruction_benchmark_v1/releases/<release_id>/`. The immutable reporting subrelease is built at `<scientific release>/reporting/<release_id>/`; it may read only signed evaluation artifacts and cannot overwrite the scientific tree.
5. **Reporting bridge:** resolved for the frozen-24 release. The bridge verifies the signed evaluation and fit ledger, converts them into one strict `reconstruction_registry_v1` plus the five long tables, and refuses publication without the signed graph package. External lanes still need their own validated bridge rows after their adapters run.
6. **Status vocabularies:** resolved for frozen-24. The bridge has a frozen, fail-closed mapping and keeps blocked or incomplete scientific states non-rankable.
7. **Null counts:** resolved in the frozen-24 bridge. Unknown inventory counts become explicit blocked/context coverage; no integer is guessed.
8. **Graph diagnostics:** resolved before launch. A signed v2 producer verifies the A/B release, emits the required L-SML, DUFS-LIU, CA-SpecRaGE, PGRD-A, Family-NRM-A, and SU-PCR panels and controls, and binds deterministic plot data and label-free example selection. The reporting bridge independently re-verifies the package.
9. **Clean-worktree gate:** resolved. Scientific fitting refuses a dirty worktree and records the exact Git/source snapshot in both freezes.
10. **Valid-draw gate:** resolved before labels. Every interval requests exactly 20,000 grouped draws and reports its valid count. A cell, aggregate, or contrast is rankable only when at least 95% of those draws are defined; otherwise its status is `BOOTSTRAP_INSUFFICIENT_VALID_DRAWS`, and Macro-24 is blocked.

## 16. Current run state

The earlier `2026-08-24_rebuild1` directory is a non-publishable preparation rehearsal: its cohort identity predates the strengthened source binding and is intentionally rejected by the current validator.

The next valid action is a new immutable release from the committed clean
snapshot: prepare both input builds, build and verify the 24 source-group
sidecars against those exact prepared matrices, finish the two 312-record fits,
issue the independent A/B score certificate, and only then open labels. After
evaluation, build the signed graph package, reporting inputs, and two identical
reporting releases.
