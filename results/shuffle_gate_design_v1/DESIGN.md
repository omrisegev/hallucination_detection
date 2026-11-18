# Shuffled-step answers as a known-error sample for the answer-gate threshold: experiment design (2026-11-16)

Design only. No GPU job runs before Omri approves it. Numbers in this document come from stored results (cited per row) or from a count on the repository files.

## 1. The question, from zero

| Item | Content |
|---|---|
| Pipeline | Base model reads question + answer (teacher forcing, one forward pass, no generation) -> 16 telemetry channels per step (TELE16) -> L-SML -> step score -> GAP flags the top steps of each answer. |
| Open problem | The answer gate. Ranking of answers works (Simes answer AUROC on ProcessBench .75-.79, cross_model_gate_check_v1). The threshold (how many answers to close) needs the clean share pi0 of the pool. Every past gate used the pool itself as reference, so it cannot see pi0 (REVIEW.md of ssl_gate_threshold_review_2026-11-15, section 2). |
| Omri's idea | Shuffle the steps of real answers. A shuffled answer is wrong with a known label. Run the base model on it, get its telemetry, and use it as a "known-error" sample. |
| Why this can identify pi0 | Theory route R4 (REVIEW.md): one external sample with a known label + the unlabeled pool -> mixture proportion estimation (MPE). The pool score distribution is pi1 x (wrong) + pi0 x (clean). If the shuffled answers have the same score distribution as real wrong answers, the ratio pool / shuffled in the most suspicious tail gives pi1. |
| Condition that this experiment tests | Score distribution of shuffled answers = score distribution of real wrong answers (SCAR assumption). If shuffled answers are "too easy", they put too much mass in the suspicious tail, pi1 is under-estimated, and the gate closes too many answers, also wrong ones with subtle errors. |
| Why new forward passes are needed | Telemetry of a step depends on all text before it. Moving stored step vectors gives no new information. Simes uses only the set of step p-values, so permuting stored scores leaves Simes unchanged. |

## 2. Dependency on E0

| E0 result (thread "Research self-supervised gate threshold calibration", running now) | This experiment |
|---|---|
| GO (PB abs(pi-hat - pi) <= .10 in >= 3/4 models and PB F1 >= 30, with real labeled errors from other benchmarks) | Run it. E0 shows that a known-error sample from outside the pool can give pi on PB. This experiment replaces the labeled source by label-free shuffled answers. |
| NO-GO | Do not run the GPU step by default. Real labeled errors are the best possible known-error sample; if they do not transfer, synthetic errors are not likely to transfer. One exception for Omri to judge: if E0 fails only because of shift between benchmarks (different questions, different generators), shuffled answers of the same PB questions do not have that shift. |

## 3. Benchmark, model, sample

| Item | Choice | Reason |
|---|---|---|
| Benchmark | ProcessBench (3,400 answers, 1,179 clean, 2,221 wrong, 25,697 steps; counted from processbench_gold.json) | Natural errors from real generators. PRMBench errors are injected by an LLM, so a synthetic-error test there is circular. PB is also where a correct threshold is worth most (+34-40 F1, simes_answer_gate SWEEP). |
| Base model | Qwen2.5-Math-7B-Instruct, prompt style qwen_cot_chat (the stored "qwen_prm" standalone cell) | Cheapest model of the panel; Simes AUROC .785 on PB. Other 3 models only after GO. |
| Answers | All 3,400 answers, all used as parents (label-free: in deployment the labels of the pool are not known) | Cost is small (section 7). Labels are read only in the evaluation (section 5). |
| Excluded | Variant V1: 28 answers with 2 steps (no permutation of steps 2..n is possible) | - |
| Seed | 20261116 for all random draws | - |

## 4. Exact construction of the inputs (CPU, label-free)

All three runs use the same driver, model, prompt and batch size. Each run is one answers.json file in the layout of results/prm_boost_processbench_v1/inputs/answers.json (uid, question, steps, benchmark, source_id).

| Run | Rows | Construction | Known first error j* |
|---|---|---|---|
| C (control) | 3,400 | Original answers, unchanged | none |
| V1 within-answer shuffle | 3,372 | Keep step 1. Uniform random permutation of steps 2..n, redrawn until it is not the identity. Question unchanged. | First position where the step differs from the original |
| V2 cross-answer swap | 3,400 | Pick position k uniform in 2..n. Replace step k by one step of an answer to a DIFFERENT question, same subset (gsm8k / math / olympiadbench / omnimath). Donor step: relative position k/n within +-.2 of the donor answer, character length within +-20% of the replaced step (median step = 201 characters). All other steps unchanged. | k |

| Design decision | Reason |
|---|---|
| Step 1 stays in place in V1 | Step 1 usually restates the problem. Real first errors are mostly later (PB median first error = step 3; 12% at step 1). A wrong step 1 makes every answer wrong from the start = "too easy". |
| V2 replaces exactly one step | Real PB errors are one local wrong step, then a cascade. One foreign step followed by the original continuation has the same shape. Full mixing of all steps would make nonsense answers ("too easy"). |
| Length match in V2, no length change in V1 | Length is a known confound of answer scores (length-only AUROC .588 on PB). V1 keeps the same steps, so token count is the same. |
| No labels used | The construction does not read gold labels. A shuffle of an already-wrong answer stays wrong, so "all shuffled answers = wrong" is true without labels. |

Uid scheme: `<orig uid>__v1` and `<orig uid>__v2`, plus a sidecar JSON per row with parent uid, permutation or (k, donor uid, donor step), and j*.

## 5. Processing and readouts

### 5.1 Telemetry (GPU, then CPU)

| Step | Code | Note |
|---|---|---|
| Forward pass | cluster/run_prmbench_prm_telemetry.py --mode instruct --model Qwen/Qwen2.5-Math-7B-Instruct --prompt-style qwen_cot_chat --answers <file> --max-batch 1 --out <dir> | Existing driver; --answers path already supports external answers.json (used for Socratic and Hard2Verify). Output external_telemetry.pkl + manifest.json. --max-batch 1 removes padding effects, so equal prefixes must give equal telemetry. |
| Bank (56 channels per step) | results/prm_boost_external_v1/build_bank_external.py (mask1 rule) -> FEATURES.npz | Needs the frozen AIRCC code dirs code_thinkfix_bank_553de79 and code_thinkfix_rescore_v1. They must be restored from the AIRCC backup on Drive (aircc_workspace_final) or rebuilt from git. To check before the run. The script reads REWARDS from the pkl; for instruct pkls the standalone cells used a variant (to locate in the hd_collection_data full_sa_* build). |
| TELE16 raw16 | same channel selection as results/cross_model_gate_check_v1/extract.py | - |

### 5.2 Score with a frozen reference (CPU)

| Quantity | Definition |
|---|---|
| Reference pool | Steps of run C only (25,697 steps). |
| Channel transform | For each of 16 channels: ECDF on the reference pool -> rank -> N(0,1). Applied unchanged (interpolated) to V1 and V2 steps. |
| Step score s | L-SML on the 16 transformed channels, weights fitted on the reference pool only (anchor = channel mean, as in cross_model_gate_check_v1). |
| Step p-value | Share of reference-pool steps with score >= s. |
| T_simes | -log10( min_i n p_(i) / i ) over the n steps of one answer. High = suspicious. |

The shuffled answers never enter the fit. This keeps the reference identical to the deployed pipeline.

### 5.3 Readouts (labels used only here)

| ID | Readout | What it answers |
|---|---|---|
| R1 | Length-stratified AUROC of T_simes: shuffled (V1 or V2) vs real wrong answers of C (5 bins of n_steps, as in cross_model_gate_check_v1). A two-sample test: .5 = same distribution. | Do shuffled answers look like real wrong answers? |
| R2 | AUROC shuffled vs real clean, next to AUROC real wrong vs real clean. | Is the synthetic error "too easy" (higher) or not an error at all (lower)? |
| R3 | pi1-hat by BBE (Garg et al. 2021): pi1-hat = min over upper-tail thresholds t of P_pool(T >= t) / P_shuffled(T >= t), with its upper confidence bound correction. Pool = C. | Prevalence estimate without labels. True pi1 = .653. |
| R4 | Gate: close the (1 - pi1-hat) share of answers with the lowest T_simes. Then the unchanged pipeline TELE16 L-SML R0 + GAP. PB average F1 and SLA, 95% bootstrap CI (2,000 draws, answers within subset). | Payoff. Context: no gate 0, Simes alpha .7 33.6-39.6, oracle-threshold sweep best in the same range. |
| Plots | fig1 T_simes histograms: clean / real wrong / V1 / V2. fig2 R3 + R4 per variant. Separate PNGs. | - |

## 6. Go / no-go (fixed before the run)

| Gate | Rule | If it fails |
|---|---|---|
| P0 parity | (a) Run C raw16 vs stored standalone qwen_prm PB bank: Spearman >= .99 per channel; no-gate pipeline on C gives SLA 38.47 +-1. (b) In V1 and V2, steps before j* have telemetry equal to C (max abs diff <= 1e-4). | Stop. Fix the pipeline. No result reported. |
| G1 realism | For one variant: R1 AUROC <= .65 AND abs(R2 shuffled-vs-clean - R2 wrong-vs-clean) <= .05. | That variant is not a valid known-error sample. |
| G2 prevalence | abs(pi1-hat - .653) <= .10 | No calibration from this variant. |
| G3 payoff | PB average F1 >= 30 | Gate not useful even if pi1-hat is close. |
| Verdict | GO if at least one variant passes P0, G1, G2 and G3. | NO-GO: the synthetic route stops; remaining choices are an explicit budget gate or an open gate. |

GO -> next: same test on the other 3 base models (7B-base, Llama-8B, 72B-Inst), then PRMBench as a secondary (circular) check.

## 7. Compute

### 7.1 Cost

| Item | Value | Source |
|---|---|---|
| Rows | 3,400 + 3,372 + 3,400 = 10,172 | Section 3 |
| Rate, 7B class, unbatched, AIRCC GPU | 0.5-0.8 s/row | PRM_STANDALONE_TELEMETRY_2026-10-22.md (prm_boost_processbench_v1 timing) |
| GPU time, A100-class | 1.4-2.3 GPU-h + about 0.5 h model download and setup | Estimate |
| GPU time, T4-class | about 3-4x longer (5-9 h); T4 has no bf16 | Estimate, not measured |
| Disk | Model 15 GB + 3 pkls (top-50 logprobs per token) about 3-6 GB | Estimate |
| Max input | 4,096-token context; PB longest answer 7,947 characters; no overflow in the stored PB runs | prm_boost_processbench_v1 README |
| CPU after GPU | Bank build + analysis, about 1-2 h | Estimate |

Timing first: 12 rows, check s/row and manifest status, then the full run.

### 7.2 Where to run (AIRCC closed 2026-11-15)

| Option | GPU | Cost | Pros | Cons |
|---|---|---|---|---|
| A. Google Colab Pro / Pro+ | A100 40 GB or L4 24 GB | Subscription about $10-50 per month with compute units; this run uses a small part | Data and backups are already on Drive (mount). bf16 support. Fast to start. | Session time limits; must checkpoint (driver already resumes from --out). |
| B. Rented cloud GPU (RunPod, Lambda, Vast.ai) | A100 80 GB / H100 | about $1.5-3 per GPU-h -> about $5-10 total | Full control, SSH, no time limits. | Account and payment setup; copy data in and out. |
| C. Omri's PC | Only if it has an NVIDIA GPU with >= 16 GB (bf16: RTX 30xx/40xx) | Free | Local data (hd_collection_data). | GPU unknown to me. C: has about 3.5 GB free -> model cache (HF_HOME) and outputs must go to another drive. |
| D. Kaggle notebooks | 2 x T4 16 GB, about 30 h per week free | Free | No payment. | 7B bf16 does not fit one T4; fp16 only, split over 2 GPUs; slower. Run C on the same hardware absorbs the numeric difference, but P0(a) Spearman may drop. |
| E. Other TAU cluster | - | - | - | Needs an account; I have no information. |
| This cloud container | none (4 CPU, 15 GB RAM) | - | - | 7B in fp32 needs about 30 GB RAM. Not possible. |

Recommendation: A (Colab with A100 or L4), because the data is already on Drive and the run is about 2-3 GPU-h. B if Colab limits are a problem. C if the PC has a suitable GPU.

## 8. Risks

| Risk | Effect | Control |
|---|---|---|
| Shuffle of independent steps is still a valid answer (V1) | Part of the "known-error" sample is clean -> pi1-hat biased down | R2; also report j* distribution. Not removable without labels. |
| Synthetic errors too easy | Shuffled scores sit far out in the tail; BBE ratio wrong | G1 catches it. |
| BBE needs a tail where clean answers are rare | PB Simes AUROC .78 -> tail not pure -> pi1-hat biased up | Report BBE curve over t; G2 is the test. |
| Frozen AIRCC code dirs missing | Bank build not reproducible | Check Drive backup before any GPU spend; P0(a). |
| One model, one benchmark | GO can be chance | Next stage = 3 more models. |

## Sources

- REVIEW.md, ssl_gate_threshold_review_2026-11-15 (routes R1-R4, E0).
- simes_answer_gate_2026-11-15 README and SWEEP.csv.
- cross_model_gate_check_v1 README (Simes definition, AUROC, length stratification).
- Garg et al. 2021, Mixture Proportion Estimation and PU Learning: A Modern Approach (BBE).
