# Competitor registry for the "PRM boosting" paper (2026-10-21)

**Purpose.** Omri asked: mark the competitors the paper compares against, read their papers, and record
exactly how they measured (datasets, splits, metrics), so we can see where we stand. This is stage 1:
positioning. No code, no inference, no tracked file was changed.

**Verification legend**
- **V**: read today (2026-10-21) from the arXiv abstract page or the arXiv HTML full text. Numbers quoted
  from the paper's own table.
- **P**: verified earlier inside the project (`results/prmbench_paper_metric_parity_v1/README.md`, 2026-10-18),
  including an exact reproduction of the paper's baseline row.
- **A**: abstract only. The body was not read, so a missing benchmark may still be in the paper.
- **U**: unverified (secondary source, or the page did not show it).

The WebFetch tool returns a model summary of the page, not the raw PDF. Every number below that will go into
the paper must be re-checked against the PDF before submission.

---

## 0. What we report, so the comparisons are defined

| Our metric | Exact definition | Where implemented |
|---|---|---|
| PRMBench PRMScore | Step-level confusion matrix pooled over all non-control answers. **Positive class = VALID (correct) step.** PRMScore = 0.5·F1 + 0.5·F1_neg (= two-class macro-F1 on pooled steps). The 758 synthetic all-correct "correct" rows are scored but **excluded from the pooled totals**. Categories (NR, NCL, ES, SC, DC, CI, PS, DR, MS) are the same formula per class. | `spectral_utils/prmbench.py::prmbench_evaluate` (port of the official `prmtest_classified/task.py::evaluate_function`) |
| Caveat | The official evaluator can route NR/NCL (redundancy, circular) through a separate redundancy head. We have no such head, so these two classes go through the validity fallback. This is an adaptation, and it is flagged in the evaluator's return value. | same |
| Socratic-PRMBench PRMScore | Same formula, w1 = w2 = 0.5 (the Socratic paper states the weights). Positive = correct step. | `spectral_utils/external_generalization/evaluation.py::metric` |
| Hard2Verify Balanced F1 | 2·TPR·TNR/(TPR+TNR), TPR on correct steps, steps pooled. Unparsed/missing steps count as wrong, following the authors' grading. Only the step-level task is implemented; ErrorID and response-level are not. | same |
| within-answer AUC | AUC of step risk vs error label, computed inside each answer and averaged. No competitor reports it. | same |
| Our decision rule | Label-free ANSWER_Z_Q80: flags the top 20% of answer-standardized step scores. The PRM-alone row uses the PRM's native rule (reward ≤ 0.5 → error). | `results/prm_boosting_controls_v1/` |

PRMBench counts. The paper reports 6,216 problems and 83,456 step labels. We score 6,211 non-control answers /
83,371 steps, plus 758 controls, for 6,969 rows. The Hugging Face `hitsmy/PRMBench_Preview` shows 6.22k rows (V).

---

## 1. Tier 1: PRIMARY head-to-head competitors

These methods improve or re-read PRM step judgments, with no training or with light training. They report on
PRMBench, so a table-to-table or CPU-recomputed comparison is possible.

### 1.1 PRISM: The Hidden Bias of PRMs (arXiv 2606.09078v1, 8 Jun 2026; no venue listed) **V, P**
- **Method.** Re-trains Qwen2.5-Math-PRM-7B with a pairwise step-contrastive loss −log σ(r⁺ − r⁻), plus
  temporal-lookahead hard negatives and a curriculum.
- **Access.** Supervised. It uses PRM800K human step labels: about 26k explicit pos/neg pairs, augmented to
  about 220k pairs. Ablation sizes are 12k and 132k pairs. "No new human labels." Trained on 4× RTX A6000;
  GPU-hours are not reported.
- **Benchmarks.** PRMBench (Table 2: FPR, FNR, Precision, TPR, TNR, PRMScore; Table 7: 9 categories).
  ProcessBench (Table 6: TPR/TNR/F1 per subset). Downstream: MATH-500 (100-problem subsample), AIME-24 and
  LiveCodeBench for BoN and guided decoding. GRPO with Qwen2.5-1.5B on six math sets.
- **Metric definitions.** FPR = P(predict correct | step wrong). PRMScore is cited to PRMBench and not
  redefined. Macro-F1 is not defined.
- **Headline (PRMBench, Table 2).**
  - Baseline Qwen-PRM-7B: FPR 69.34, TPR 95.36, TNR 30.66, PRMScore 65.50.
  - PRISM: FPR 47.13, FNR 12.69, Precision 91.93, TPR 87.30, TNR 52.86, PRMScore 68.00.
  - Table 7, best curricula (132k C3/C4): overall 68.0. NR 52.3/53.5, NCL 63.7/66.0, MS 99.3/98.7.
- **Internal inconsistencies.**
  - The abstract says "22% fewer false positives", but Table 2 shows a 32.22% reduction.
  - The baseline FNR 4.27 does not equal 100 − TPR 95.36 (P: reproduced only when the controls are pooled in).
  - The ProcessBench Omni-MATH and OlympiadBench cells appear transposed (P).
- **Code / checkpoints.** The paper links github.com/Aakriti05/PRISM. **Today the repository is empty** (V).
  No checkpoint has been released, so PRISM cannot be re-run on Socratic or Hard2Verify.
- **For us.**
  - Same metric: YES for PRMScore, the 9 categories, FPR, TPR and TNR.
  - Parity: our PRM_native reproduces their baseline (65.49 / FPR 69.42 / TPR 95.40) (P).
  - Remaining confound: decision rule. Moving the untouched PRM's global threshold to PRISM's TPR = 87.30
    gives FPR 47.70 and PRMScore 68.26 (P). That point uses labels.
  - Comparison status: DONE (table-to-table plus a matched-TPR reading).

### 1.2 CROP: Conformal Certification of Reasoning Trace Prefixes (arXiv 2605.30085v2, 10 Oct 2026; "Accepted at EMNLP 2026 (Oral, Main)") **V, P**
- **Method.** A verifier-agnostic split-conformal calibration of a step-risk threshold. It returns the longest
  prefix below the threshold. Guarantee: marginal P(returned prefix contains an annotated error) ≤ α, under
  exchangeability.
- **Access.**
  - Risk proxies: Random; Token/format LR; Trace-feature LR (Llama-3.1-8B-Instruct likelihood, confidence,
    entropy and trajectory features); Direct PRM (frozen Qwen2.5-Math-PRM-7B, risk = 1 − P(correct)); and
    Trace+PRM LR.
  - The LRs are trained on a 60% step-labelled split (ℓ2, C = 1, balanced class weights).
  - **Every proxy's threshold is calibrated with labels on a 20% calibration split.**
- **Benchmarks (Table 4/6).**

  | dataset | traces | steps | step error rate |
  |---|---|---|---|
  | CRV Arithmetic | 1,500 | 10,316 | — |
  | CRV GSM8K | 1,319 | 8,995 | — |
  | ProcessBench | 3,400 | 25,697 | 6.6% |
  | Math-Shepherd | 10,000 | 37,514 | 47.6% |
  | **PRMBench** | **5,000** | **67,445** | **15.9%** |
  | PRM800K | 8,000 | 70,698 | 4.2% |

  - Splits are 60/20/20 at trace level, stratified by has-error.
  - 20 seeds (2806–2825) for the main CRV sets; **10 seeds (2806–2815) for the additional sets, PRMBench
    included**.
  - The fixed grid is 101 points on [0, 1] (P, from their code).
- **Metrics.**
  - Step AUROC, **pooled over all test steps**.
  - Prefix kept = mean over test traces of M/T.
  - Prefix risk = share of test traces whose kept prefix contains an error.
  - Boundary deviation. Repair accuracy (CRV only).
  - Primary α = 0.05; 0.025 and 0.10 are also used.
- **Headline (PRMBench, Table 6, α = 0.05).**

  | proxy | step AUROC | kept % | risk % |
  |---|---|---|---|
  | Random | .4994 | 8.7 | 4.7 |
  | Token/format | .7751 | 16.2 | 5.0 |
  | Trace features | .7812 | 15.9 | 4.6 |
  | Direct PRM | .8010 | 4.8 | 0.6 |
  | **Trace+PRM** | **.8336** | **21.8** | 5.0 |

- **Code.** github.com/matthewyccheung/crop (V link; contents checked in P).
- **For us.**
  - Same metric: NO. Pooled AUROC (ours is within-answer) and kept/risk at α.
  - Status: DONE by CPU on an APPROXIMATE subset (P). The first 5,000 non-control rows give 67,465 steps.
    Their split assignment is re-implemented.
  - Our label-free fused arms keep 23.0–25.1% at risk ≤ 5.0%, vs 21.8% for supervised Trace+PRM.
  - The Direct-PRM 4.8% is largely a grid artifact: the same scores on a quantile grid keep 17.2%. Our tables
    must show this bridge row.
  - Caveat: our fusion was fitted label-free on all 6,969 answers, test traces included (transductive).
  - Missing:
    - their Token/format and Trace-feature proxies (these need step text and Llama-3.1-8B features);
    - the other five datasets (these need new inference);
    - Socratic: we could run CROP's calibration on our existing Socratic scores, by CPU.

### 1.3 SLIDER / Step-RRI (arXiv 2610.00571v1, 30 Sep 2026; "Accepted at ICLR 2026 Workshop on Logical Reasoning of LLMs") **V, P**
- **Method.** Partial Information Decomposition of information about the final answer across consecutive steps.
  - Encoder: all-MiniLM-L6-v2 embeddings, k-means with 10 clusters, the CVX PID estimator.
  - It needs the final answer A.
  - It generates up to 50 numerical variants per problem with GPT-4o-mini, so it makes **extra LLM calls**.
  - About 38 s per PRMBench sample (App. D.5).
- **Access.** The PID itself is training-free. The **decision threshold comes from a linear classifier
  calibrated on a held-out labelled set**.
- **Benchmark.** Only the PRMBench redundancy class ("around 750 problems"). Positive = steps annotated
  redundant.
- **Metric.** "Accuracy" 61.91% (Step-RRI) vs InfoGain (QwQ-32B top-10 entropy drop) and Cosine (MiniLM). The
  text gives gains of +13.53 and +10.31 points, which imply 48.38% and 51.60%. These come from a chart.
  - Not stated: split sizes, class balance, whether the accuracy is balanced, whether step 1 is scored.
- **Code.** No code or data release statement found (V, full HTML).
- **For us.**
  - Same metric: UNKNOWN, so the comparison is APPROXIMATE (P).
  - Under every reading, our fused arms reach 71–74% (balanced accuracy, label-free Q80: 72.8–73.3) vs 61.91.
  - The protocol is under-specified. Use this only as a category-specific supporting comparison, or ask the
    authors for their protocol.

### 1.4 ARES: Probabilistic Soundness Guarantees in LLM Reasoning Chains (arXiv 2507.12948v2, 28 Sep 2025; "EMNLP 2025 camera ready") **V**. New, not in the earlier project notes.
- **Method.** Autoregressive Reasoning Entailment Stability. Each step is scored by the probability that it is
  entailed by previously verified premises, with sampling over premise subsets. The number of samples is
  N = log(2m/δ)/(2ε²) (in practice 0.03–0.31× of that bound).
  - Entailment model: GPT-4o-mini or Qwen3-4B; a Qwen2.5-Math-PRM-7B backbone variant is in App. Table A7.
  - This means **many extra LLM calls per step**.
- **Access.** Scoring is training-free. The **error threshold is swept on a training split to maximize Macro-F1
  (5-fold CV)**, and the prompts were tuned by inspection.
- **Benchmarks.** PRMBench, DeltaBench, ClaimTrees (synthetic), CaptainCookRecipes. "We use a subset of
  examples for each experiment." **The PRMBench subset size is not given.**
- **Metric.** Macro-Recall, Macro-Precision and Macro-F1 (classes not stated). With step-level pooling this
  equals PRMScore's form, but the subset and the CV-tuned threshold differ.
- **Headline, Table 1 (PRMBench Macro-F1, GPT-4o-mini / Qwen3-4B).**

  | method | Macro-F1 |
  |---|---|
  | ARES | .640 / .636 |
  | Entail-Prev | .596 / .641 |
  | LLM-Judge | .643 / .675 |
  | ROSCOE-LI-Source | .570 / .495 |
  | ReCEval-Intra | .568 / .554 |

  - Table A7 (Qwen2.5-Math-PRM-7B backbone, PRMBench): **PRM alone F1 .749 > ARES .736**.
- **Code.** github.com/fallcat/ares.
- **For us.**
  - Same metric: APPROXIMATE. PRMScore-like macro-F1 on an unknown subset with a label-tuned threshold.
  - The numbers are not table-comparable: their PRM gets .749 vs the official 65.5, so the subset or threshold
    clearly differs.
  - What we can do by CPU: report our arms under an "ARES-style" rule (threshold chosen by 5-fold CV on
    PRMBench) as a labelled-threshold reading.
  - Value for the paper: ARES is the clearest published **training-free step scorer evaluated on PRMBench**, and
    its own appendix shows it does not beat the PRM there.

### 1.5 Optimal Aggregation of LLM and PRM Signals (arXiv 2510.13918v2, 23 Apr 2026; "Published as a conference paper at ICLR 2026") **V, P**
- **Method.** Theory for optimal weighted aggregation of responses using PRM score and answer frequency.
  - Weights are calibrated per LLM–PRM pair and are often negative.
  - Calibration uses **a small, one-time labeled dataset**: a MATH pool of 2k questions with 1k sampled per run,
    or a MATH500 pool of 200 with 100 sampled.
  - Unsupervised calibration is listed as future work.
- **Benchmarks.** MATH (N = 32 responses per question) and MATH500 (N = 112). Five LLMs and seven PRMs
  (including "Qwen-PRM-7B").
- **Metric.** Final-answer accuracy of BoN / weighted voting.
- **Headline.** Table 1, Qwen-PRM-7B on MATH, averaged over 5 LLMs:

  | method | accuracy |
  |---|---|
  | Logit WV | 63.3 |
  | BoN | 61.8 |
  | Vanilla WV | 60.8 |
  | MV | 57.1 |
  | oracle | 66.5 |

- **For us.**
  - Not table-comparable: it is a response-selection task, so every cell is a GAP (P).
  - Comparing would need new generation: N ≥ 32 samples per question with graded answers, plus PRM telemetry.
  - Keep it as the closest "fusion of PRM with another signal" paper at a top venue, and state the task
    difference.

### 1.6 VerifySteer: The Hidden Signal of Verifier Strictness (arXiv 2605.20745v1, 20 May 2026; no venue) **V**. Primary only for a Hard2Verify panel.
- **Method.** A gray-box method on the verifier's hidden states at the `\n\n` paragraph boundary.
  - It steers along mean-difference "strictness" and "leniency" directions.
  - A 3-layer MLP correctness probe routes samples.
  - No fine-tuning of the verifier.
- **Access.** **Labelled.** The steering vectors and the probe are built from about 11K ActPRM samples with
  known correctness (500 erroneous + 500 correct, ×16 rollouts, for the vectors). Hard2Verify is not used for
  fitting.
- **Benchmarks.** ProcessBench (Qwen3-1.7B / 8B) and Hard2Verify (Qwen3-8B-thinking, FARE-20B).
- **Metric.** ErrorID: predict the first wrong step. F1 = harmonic mean of TPR (accuracy on fully correct
  solutions) and TNR (accuracy on incorrect ones).
- **Headline, Hard2Verify ErrorID F1.**

  | method | Qwen3-8B-thinking | FARE-20B |
  |---|---|---|
  | basic prompt | 22.4 | 41.5 |
  | Cons@8 (self-consistency) | 31.2 | 42.7 |
  | VerifySteer-Bi | 33.4 | 51.1 |

  About 7× less compute than Cons@8.
- **Code.** github.com/YefanZhou/VerifySteer.
- **For us.**
  - This is the nearest published "gray-box signal improves a verifier" method on a PRMBench-like benchmark,
    but it uses labels.
  - Comparison needs Hard2Verify inference (not run) and our own ErrorID readout.

---

## 2. Tier 2: BASELINES we must include

### 2.1 The PRMs themselves (native rule), official PRMBench and Socratic numbers
PRMBench values are from PRMBench Table 3 (arXiv 2501.03124v5, ACL 2025 Main) **V**.

| PRM | PRMBench (paper) | ours, native | ours, boosted (CONT L-SML, Q80) | Socratic (paper) | ours, Socratic native |
|---|---|---|---|---|---|
| Qwen2.5-Math-PRM-7B | 65.5 | 65.49 | 69.48 (3 channels chosen with labels); CHAIN_N16 label-free 68.25 | 68.0 | 68.09 |
| Qwen2.5-Math-PRM-72B | 68.2 | 68.18 | 69.84 | not in paper | 72.0 |
| Pure-PRM-7B | 65.3 | 65.31 | 66.69 | not in paper | 63.7 |
| RLHFlow-PRM-Mistral-8B | 54.4 | 54.43 | 63.53 | 48.8 | 48.1 |
| MathShepherd-Mistral-7B | 47.0 | 47.02 | 63.68 | 64.4 | 64.4 |

- The "ours" columns come from `results/prm_boosting_generalization_v1/run_v1/eval_lsml4/README.md` and
  `results/prm_boost_external_v1/README.md`.
- Socratic boosted rows **lose to the native PRM for Qwen-7B, Qwen-72B and MathShepherd**.
- Parity: all 5 PRMBench native values match the paper to 0.05. Socratic Qwen-7B matches (68.09 vs 68.0).

### 2.2 Strong trained PRMs reported on PRMBench (overall PRMScore, each paper's own threshold, usually 0.5)
| model | PRMBench overall | source | supervision |
|---|---|---|---|
| Qwen2.5-Math-PRM-72B | 68.2 | PRMBench T3 **V** | LLM-judge + MC labels (Qwen) |
| PRISM (re-trained 7B) | 68.0 | PRISM T2 **V** | PRM800K pairs |
| PathFinder-PRM-7B (arXiv 2505.19706v1; no venue on arXiv; "EMNLP Findings 2025" per ACL Anthology search hit, **U**) | 67.7 | its T1 **V** | ~400K: PRM800K human + RLHFlow MC; error types from R1-Distill-32B |
| R-PRM-7B-DPO (arXiv 2503.21295v1) | 66.8 | its T2 **V** | PRM800K + Llama-3.3-70B critiques; about 10 sampled evaluations per step (inference-time scaling) |
| ActPRM-X / ActPRM (arXiv 2504.10559v1) | 66.7 / 65.5 | its T2 **V** | QwQ-32B as label annotator |
| Skywork-PRM-7B | 65.1 in PRMBench v5 T3; **36.2** in the R-PRM / PathFinder tables | **V**, both | the discrepancy is presumably a PRMBench version update, **U** |
| ThinkPRM-7B / GenPRM-7B | 64.3 / 50.5 | as quoted in PRISM T1 ("from Song et al.") **U**, original source not found | generative PRMs |
| ReasonEval-34B / 7B | 60.5 / 60.1 | PRMBench T3 **V** | |

### 2.3 LLM-as-critic baselines (prompted judges)
- **PRMBench T3 (V).**
  - Gemini-2.0-thinking-exp-1219: 68.8.
  - o1-mini: 68.8, on a 394-sample subset.
  - DeepSeek-R1: 67.8, on the subset.
  - GPT-4o: 66.8. Gemini-2.0-flash: 66.0. QwQ-Preview-32B: 63.6.
  - Human: 83.8, on 450 instances.
  - 2-shot prompting.
- **Socratic-PRMBench T3/T4 (arXiv 2505.23474v1; no venue listed) (V).**
  - o3-mini 75.7, QwQ-32B 73.8, Gemini-2.5-Pro 73.5, DeepSeek-R1 73.0, GPT-4o 70.8.
  - Best PRM: Qwen-PRM-7B 68.0.
- **Hard2Verify T2 (arXiv 2510.13744v1; ACL 2026 long, pp. 22502–22517, per the ACL Anthology search hit **U**).**
  Values are step-level Balanced F1.
  - GPT-5 85.83, Gemini-2.5-Pro 83.09, gpt-oss-120B 74.64.
  - PRMs: Qwen-PRM-7B 42.37, Qwen-PRM-72B 35.50, UniversalPRM-7B 60.27.

### 2.4 Majority vote / self-consistency / ensembles at step level
- **Published step-level self-consistency.** VerifySteer Cons@4/8 (ProcessBench, Hard2Verify, ErrorID F1).
  R-PRM aggregates about 10 sampled critiques per step.
- **No published step-level majority vote over several PRMs on PRMBench was found.** We must compute it
  ourselves by CPU, because 5 PRMs already exist on PRMBench. `results/prm_voters_prmbench_v1/` already did
  this: a no-go, below the best single PRM. This must be shown as a baseline row.

### 2.5 Label-free verifier ensembling (the conceptual neighbours of our L-SML fusion)
| paper | level | labels | benchmarks | step-level PRMBench result? |
|---|---|---|---|---|
| FUSE (arXiv 2604.18547v1; Candès group) | response | none | GPQA-Diamond, HLE, IMO Shortlist (project digest) | **No** |
| Weaver (arXiv 2506.18203v3; NeurIPS 2025) **V-A** | response (BoN selection) | weak supervision; "mostly" label-free | reasoning and math; 87.7% avg with Llama-3.3-70B generator | **No** |
| ScalePRM (arXiv 2512.03244v2, 31 Aug 2026; no venue) **V-A** | step verdicts aggregated into **training labels**, then a generative PRM is trained | no ground truth | ProcessBench 67.5 F1 (vs 66.4 reference-guided, 61.9 GPT-4o); RL 47.4% | **No** |
| uPRM (arXiv 2605.10158) (project digest) | step, trained with LoRA + RL | none | ProcessBench F1 58.3/52.6/42.7/39.8 | **No** |

For us: none of these is table-comparable on PRMBench. The honest comparison is a **labelled transplant**: apply
their aggregator (spectral / Dawid-Skene / accuracy-weighted / majority) to the same step-level inputs, our 5
PRMs plus telemetry, by CPU. Label each transplant as an adaptation (CLAUDE.md rule: tailor, never transplant
under the author's name).

### 2.6 Supervised readout ceiling and simple controls (ours, already computed)
- Telemetry13+PRM LR, the analog of CROP's Trace+PRM: pooled AUROC .8285, kept 24.8%.
- Position/length LR: .7488.
- Socratic LR (PRM + N16): within-AUC .836 vs PRM .824 (`results/prm_boost_socratic_diagnosis_v1/`).
- Threshold-moved PRM: 68.26 at PRISM's TPR; the best global threshold gives 68.31 (Step 291; a label-chosen
  oracle).
- These rows are mandatory: they separate "fusion adds information" from "the decision rule moved".

---

## 3. Tier 3: CONTEXT (cite; not directly comparable)

| paper | why only context |
|---|---|
| Socratic-PRMBench, Hard2Verify, PRMBench (the benchmarks themselves) | define the protocols; see §4 |
| Hidden Error Awareness (2605.09502v1; ICML-26 MI workshop submission) **A** | supervised linear probe; trace-level AUROC .95 (.79 from the first step); datasets not named in the abstract |
| HSRM (2608.30841v1; EMNLP 2026) **A** | hidden-state reader trained with outcome labels; response ranking |
| Uncertainty Localization via Embedding Perturbations (2602.02427v4) **V** | training-free white-box step localization. MATH/BBH/GPQA, top-10%/20% hit rate of the first wrong step (GPT-5.x labels). Not on PRMBench, but it is the closest label-free **localization** method; its baselines (NLL/entropy max/avg, CCP, TokenSAR) are telemetry-like |
| DeltaBench paper (2502.19361v3) **V** | long-CoT section-level benchmark. PRM threshold is label-free μ−σ; Qwen-PRM-7B section F1 29.22. A possible later transfer panel |
| BetaPRM (2605.15529), Distributional PRM (2605.06785), Uncertainty Calibration of PRMs (2506.09338; NeurIPS 2025) **A** | PRM calibration for BoN / adaptive compute; trained; no step-level PRMBench numbers in the abstracts |
| QD stress tests (2608.08008) **A** | Qwen-PRM-7B aggregation exploits (mean pooling); robustness context |
| Reward Under Attack (2603.06621), Out of Distribution Out of Luck (EACL 2026) (project notes, **U**) | PRMs as fluency detectors; motivates telemetry |
| Temporal Consistency (2503.14495) **A**, Reasoning Jury (2608.12585) **A**, Pessimistic Verification (2511.21522) **A**, OSVerify (2610.18089) **A** | multi-call LLM verification; no PRMBench numbers |
| TIPS, StepWiser, ThinkPRM (2504.16828v5), GenPRM | generative PRMs; possible future L-SML voters |
| CPMI (2604.10660; ACL 2026) **A**, LLaDA-PRM (2609.22700) **A**, Cliff (2609.02817) **A**, Verifiable Counterfactual Supervision (2605.02395) **A** | cite PRMBench; no PRMBench results visible in their abstracts |
| How Far Can Unsupervised RLVR Scale (2603.08660; ICLR 2026) (project notes) | the confidence–correctness ceiling of intrinsic signals; explains the Socratic failure |

---

## 4. Benchmarks (PRMBench-like) and their protocols

| benchmark | size | errors | label | official metric | PRM threshold in the paper | status for us |
|---|---|---|---|---|---|---|
| **PRMBench** (2501.03124v5, ACL 2025) **V** | 6,216 problems, 83,456 steps; 9 classes (NR 758, NCL 758, ES 757, SC 758, DC 757, CI 757, PS 756, DR 750, MS 165) | injected | per step | PRMScore = w1·F1_neg + w2·F1 (w = 0.5 in the official code; the weights are not stated in the paper text) | each model's own (0.5 for Qwen) | **done**, parity |
| PRMBench-STEM (same paper) **V** for counts | physics 1,619, chemistry 1,543, biology 2,342 | injected | per step | PRMScore | same | data availability **U** (not visible in the HF preview); a cheap same-metric transfer if released; needs inference |
| **Socratic-PRMBench** (2505.23474v1) **V** | 2,995 paths, about 8.7 steps; 6 patterns, 20 subtypes | injected by GPT-4o into Socratic traces (Qwen2.5-72B LoRA) | per step | PRMScore, w1 = w2 = 0.5 | not stated | **done**: native parity; boosting fails the pre-registered rule |
| **Hard2Verify** (2510.13744v1; ACL 2026 **U**) **V** | 200 responses, 1,860 steps (58% correct); 80 olympiad problems, 2023–25 | natural (GPT-5, Gemini-2.5-Pro, Claude Sonnet 4) | per step, strict (no error carried forward) | Balanced Acc and Balanced F1 for step, response and ErrorID | **tuned on 100 random responses** (sweep 0.1–0.9, maximize the harmonic mean of the 3 Balanced F1s), so it uses labels | **not run**: 12-row timing only; 2/12 exceed the 4,096-token context for the Qwen family |
| MedPRMBench (2604.17282v2) **A** | 6,500 questions, 13,000 chains, 113,910 step labels | 14 clinical error types | per step | PRMScore + strict first-error localization | — | deferred (Omri, 2026-09-24) |
| LSR-Ben (2605.01203v3; the title changed between versions, **U**) **V** | 3,595 instances, 9 sci/logic subdomains | natural | first-error index | ProcessBench-style F1 | — | out of the PRMScore family |
| DeltaBench (2502.19361v3) **V** | 1,236 long CoTs | natural | section | section macro-F1 | label-free μ−σ | possible later |
| ProcessBench | 3,400 | natural | first error | F1 (harmonic) | — | **out of the thesis focus** (Omri) |

---

## 5. Matrix: competitor × benchmark × metric

Legend:
- **T**: table-to-table today (their published number vs ours under the same metric).
- **C**: needs a new CPU computation on existing scores.
- **C✓**: CPU computation already done.
- **I**: needs new inference or generation.
- **—**: not applicable or impossible.

| competitor | PRMBench PRMScore + 9 cat. | PRMBench FPR/TPR @matched | PRMBench pooled AUROC + kept/risk (CROP) | PRMBench redundancy acc. (SLIDER) | PRMBench macro-F1, CV threshold (ARES) | Socratic PRMScore | Hard2Verify step BalF1 | Hard2Verify ErrorID F1 | BoN accuracy (MATH) |
|---|---|---|---|---|---|---|---|---|---|
| PRISM | **T ✓** | **C✓** | — | — | — | — (no checkpoint) | — | — | — (their BoN subsample is not reproducible) |
| CROP | — | — | **C✓** (approx. subset) | — | — | C (their calibration on our scores) | I | — | — |
| SLIDER | — | — | — | **C✓** (approx.) | — | — | — | — | — |
| ARES | approx. only | — | — | — | C (our arms) | — | — | — | — |
| Optimal Aggregation | — | — | — | — | — | — | — | — | **I** |
| VerifySteer | — | — | — | — | — | — | — | I | — |
| Trained PRMs (§2.2) | **T** | C (Qwen family only; we have their scores) | C | C | C | T (paper rows) | T (paper rows) + I (our arms) | T + I | — |
| LLM critics (§2.3) | **T** | — | — | — | — | **T** | T + I | T + I | — |
| Multi-PRM majority / ensembles | **C✓** (prm_voters: no-go) | C | C | C | C | C (5 PRMs on Socratic exist) | I | I | — |
| FUSE / Weaver / DS-style transplant | C | C | C | C | C | C | I | I | I |
| Supervised LR ceiling | C✓ | C | C✓ | C | C | C✓ (Socratic diagnosis) | I | I | — |
| Threshold-moved PRM | **C✓** | **C✓** | C✓ (quantile-grid bridge) | C✓ | C | C | I | I | — |

"Our arms on Hard2Verify" always needs inference: about 0.03–0.4 GPU-h per PRM for 200 rows (measured), and it is
gated by Omri's condition, which Socratic did not meet.

---

## 6. Positioning recommendation (stage 1)

1. **Three to five papers define the position.**
   - PRISM: same base PRM, same official metric, the trained alternative.
   - CROP (EMNLP 2026 Oral): same PRM, a utility metric at certified risk, a supervised LR-on-telemetry proxy.
   - The PRMBench leaderboard of trained PRMs and critics: Qwen-72B 68.2, PathFinder 67.7, R-PRM 66.8,
     ActPRM-X 66.7, Gemini-2-thinking 68.8.
   - The label-free ensemble line (FUSE / Weaver / ScalePRM), cited as the method family, with a CPU
     transplant at step level.
   - SLIDER and ARES are secondary: one category, under-specified protocols.
2. **Supportable claim vs PRISM.** Post-hoc label-free fusion reaches PRISM's PRMScore without retraining or
   new labels. **Do not claim "beats PRISM" from 68.25 vs 68.0.** At PRISM's TPR, the threshold-moved
   untouched PRM already reaches 68.26 / FPR 47.70. Only LSML_CONT, whose channels were chosen with labels,
   is better at the matched point. Report matched-TPR FPR and a threshold-free metric.
3. **Strongest current claim: vs CROP.** Label-free fused arms keep 23.0–25.1% of the prefix at risk ≤ 5%,
   vs 21.8% for supervised Trace+PRM. To make it paper-grade:
   - use the exact CROP subset and splits, from their code;
   - show the grid bridge row: Direct PRM 4.8% → 17.2% on a quantile grid;
   - fit the fusion on training traces only, or disclose transductive fitting.
4. **"Cheap PRMs become competitive" claim.** MathShepherd 47.0 → 63.7 and RLHFlow 54.4 → 63.5 cross the
   ReasonEval and Skywork rows. This needs a native-rule-matched or threshold-free companion, because Q80 vs
   0.5 confounds it.
5. **Every table needs an access column:** PRM trained with labels / fusion label-free / threshold
   label-calibrated (CROP, Hard2Verify, ARES, SLIDER) / retrained (PRISM, PathFinder, R-PRM, ActPRM). Our
   "label-free" covers only the fusion step.
6. **Main risk: generalization.** On Socratic (same metric) the boosting fails the pre-registered rule.
   - Qwen-7B: native 0.681 vs best boosted 0.674.
   - Qwen-72B: 0.720 vs 0.673.
   The paper must report this. The defensible framing is conditional: telemetry helps when it is aligned
   with the errors. This is consistent with the unsupervised-RLVR "confidence ceiling".
7. **Novelty statement, as far as verified.** No paper found does label-free step-level fusion of a PRM with
   gray-box telemetry and reports PRMBench.
   - FUSE and Weaver are response-level.
   - ScalePRM aggregates only to make training labels.
   - VerifySteer, HSRM and Hidden Error Awareness use labels.
   - ARES is training-free but uses many LLM calls and loses to the PRM on PRMBench in its own appendix.
   This is a search result, not proof. The search was Semantic Scholar citers of PRMBench (136),
   Socratic (3) and Hard2Verify (22), plus web search.
8. **Metrics the paper must report.**
   - PRMBench: official PRMScore with the 9 categories and FPR/TPR/TNR (PRISM parity); within-answer AUC;
     pooled AUROC plus CROP kept/risk at α = 0.05.
   - Socratic: PRMScore + within-AUC, negative result included.
   - Optional: Hard2Verify step-level Balanced F1 + ErrorID, only if Omri lifts the gate.
   - PRMBench-STEM or MedPRMBench as a same-metric transfer, if inference is approved.
9. **Baselines reviewers will ask for:**
   - threshold-moved PRM;
   - multi-PRM majority / ensemble (we already have a no-go);
   - LLM-critic self-consistency (no published PRMBench number; would need inference);
   - a FUSE/Weaver-style aggregator at step level (CPU);
   - the supervised LR ceiling;
   - position/length controls.

## 7. Not verified / gaps

- The PDFs were not read directly. All arXiv numbers come from HTML via a summarizing fetch tool and must be
  re-checked before they go into the paper.
- Venues not confirmed:
  - Hard2Verify, ACL 2026: from an ACL Anthology search hit; arXiv lists none.
  - PathFinder, EMNLP Findings 2025: from a search hit.
  - Socratic-PRMBench: no venue found.
- ThinkPRM 64.3 and GenPRM 50.5 on PRMBench: only as quoted by PRISM; the original source was not found.
- The Skywork-PRM-7B discrepancy (65.1 vs 36.2) is unexplained.
- PRMBench PRMScore weights are not given in the paper text that was read. The 0.5/0.5 comes from the official
  code (our port) and is stated explicitly in Socratic-PRMBench.
- ARES's PRMBench subset size is not stated in the paper.
- SLIDER's accuracy protocol is not stated; no code release was found.
- Hard2Verify camera-ready (ACL) numbers were not compared with arXiv v1.
- PRMBench-STEM data release was not confirmed.
- MR-Ben / MR-GSM8K / MR-MATH were not checked in this session.
- Several Tier-3 entries are abstract-only (**A**). Their bodies may contain PRMBench tables.

## Sources (opened 2026-10-21)
arxiv.org/abs + /html for: 2606.09078, 2605.30085, 2610.00571, 2501.03124, 2505.23474, 2510.13744,
2507.12948, 2605.20745, 2602.02427, 2510.13918, 2505.19706, 2504.10559, 2503.21295, 2506.18203, 2512.03244,
2605.01203, 2604.17282, 2604.17957, 2502.19361, 2605.15529, 2605.06785, 2609.22700, 2608.08008, 2605.02395,
2609.02817, 2604.10660, 2604.02450, 2605.09502, 2608.30841, 2503.14495, 2608.12585, 2511.21522, 2610.18089,
2506.09338, 2510.20603, 2509.24460, 2503.19877, 2504.16828.
GitHub: Aakriti05/PRISM (empty), SalesforceAIResearch/Hard2Verify, ssmisya/PRMBench.
HF: hitsmy/PRMBench_Preview.
Semantic Scholar citations API (raw JSON saved next to this file: `prmbench_citations.json`,
`cit_2505.23474.json`, `cit_2510.13744.json`).
Project files: `results/prmbench_paper_metric_parity_v1/README.md`,
`results/prm_boosting_generalization_v1/run_v1/eval_lsml4/README.md`, `results/prm_boost_external_v1/README.md`,
`papers/digests/{fuse-..., unsupervised-process-reward-models}.md`, `spectral_utils/prmbench.py`,
`spectral_utils/external_generalization/evaluation.py`.
