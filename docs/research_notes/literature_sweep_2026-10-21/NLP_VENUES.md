# Literature sweep: top NLP venues (ACL/EMNLP/NAACL 2025, ACL 2026, EMNLP 2026)

Date of sweep: 2026-10-21. Scope: step-level verification and PRMs, label-free step-error detection from LLM-internal signals, training-free verifiers on step-labelled benchmarks, and new step-labelled benchmarks. Papers on the "already known" list were left out of the main table. Venue updates for known papers are in a separate section at the end.

## How venues were checked

- **ACL Anthology pages** (aclanthology.org) are the evidence for ACL 2025, EMNLP 2025, NAACL 2025 and ACL 2026 papers.
- **EMNLP 2026** (24–29 Oct 2026) has no Anthology volume yet. For those papers the evidence is the official detailed program sheet, which is linked from https://2026.emnlp.org/program/ (Google Sheet "EMNLP 26 Author Presenter Schedule and Times"). Paper IDs are given as `NNNN-MAIN` or `NNNN-FIND` (Findings). The arXiv comment is cited too where one exists.
- **NAACL 2025**: no relevant paper was found after several targeted searches.

Read levels:
- **FULL**: body and tables read, either through the arXiv HTML or through text I extracted from the Anthology PDF.
- **ABS**: abstract or Anthology landing page only.
- **CARD**: only a dataset card was found.

All numbers are as the paper reports them, with the table or figure number.

## Main table

| # | title | arXiv id | venue + year + track | venue link | read level | method (1 line) | labels used | benchmarks + metric | PRMBench? ProcessBench? Socratic? Hard2Verify? (headline) | relevance | why (1 line) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Benchmarking Fine-Grained Error Detection in Multimodal Reasoning (PRMBench-V + **BR2-PRM**) | none found | ACL 2026, Main (Long) | https://aclanthology.org/2026.acl-long.2068/ | FULL (PDF text: abstract, Sec. 4, App. C–D) | BR2-PRM is a Dawid-Skene-style Bayesian reliability model over a frozen MLLM-PRM's 9 per-error-type scores. Scores are modelled as categorical draws given a latent error state with Dirichlet priors (Alg. 1–2). The trace score is the product over k of P(z=0 \| s) | supervised calibration: reliability is fitted on PRMBench-V ground-truth labels, and no PRM is retrained | PRMBench-V (907 queries, 9 error types, 8,163 cases) plus Best-of-N on MMK12 | none of the four. Best MPRMs reach about 30% error-identification accuracy (abstract). BR2-PRM gains up to +4.8% BoN over simple averaging and +1.2–1.8% over weighted scoring (Fig. 6) | **direct competitor (closest cousin)** | Same idea as ours (Dawid-Skene/latent-class reliability weighting of a frozen PRM's channels, no retraining), but labelled calibration, conditional independence and trace-level BoN only. App. C also flags the k lowest PRM steps with k taken from the ground-truth error count, which is an oracle-k protocol |
| 2 | Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens (AutoMeco + MIRA) | 2506.08410 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.171/ | FULL (PDF text + arXiv HTML) | Benchmarks training-free step "lenses" (MaxProb, PPL, entropy, ΔEntropy, CoE-R/C hidden-state). MIRA is a training-free Markov adjustment Q_i = s_i + γ·max V(S_{i+1}) with softmax over steps | none for the method. Evaluation step labels are Qwen2.5-Math-PRM-7B scores thresholded at θ | GSM8K (250), MATH500, MinervaMATH, with 3 LLMs. Metrics: AUROC/AUPR/FPR95 against PRM pseudo-labels, plus BoN | none of the four. Qwen2.5-7B GSM8K entropy AUROC 60.62 → 71.90 with MIRA (Table 2). MIRA helps 68.5% of AUROC configurations (N=54). Step-level κ between two PRMs is only 0.33 (Qwen2.5-7B, consistency table, Sec. 4.4) | **direct competitor** | The closest NLP-venue work using label-free entropy, perplexity and hidden-state step signals. It treats the PRM as ground truth instead of fusing it, and has no PRMBench or ProcessBench evaluation |
| 3 | EST-PRM: Stress-Testing Process Reward Models Before They Become Load-Bearing | 2606.00437 | EMNLP 2026, Main (program 2115-MAIN) | program sheet via https://2026.emnlp.org/program/ | FULL (arXiv HTML) | Label-preserving perturbations (step inflation, dependency-aware reordering, confidence markers) applied to 5 PRMs. Also tests training-free defenses, including min(PRM, zero-shot LLM critic) | none (evaluation study) | 4,687 chains from MATH-500, GSM8K and PRMBench (945 regenerated PRMBench chains, Table 8). Metrics: Pearson ρ, inflation rate, flagging AUROC | PRMBench problems are used but no PRMScore is reported. Qwen2.5-Math-PRM step-inflation rate 47.6±4.3% (Table 5); baseline ρ 0.434 (Table 4). min(PRM, critic) ensemble flagging AUROC 0.72 on Math-Shepherd (Table 6) | **direct competitor (partial) / context** | Measures biases of the same PRMs we boost, and its best defense is a training-free fusion of the PRM with a second signal |
| 4 | Scaling Evaluation-Time Compute with Reasoning Models as Evaluators | 2503.19877 | ACL 2026, Findings | https://aclanthology.org/2026.findings-acl.2102/ | FULL (Findings PDF, Table 1) | Prompted reasoning models (QwQ-32B, R1-Distill) judge each step separately ("multi-step process evaluation"), with self-consistency over 8 CoTs | none (training-free prompting) | ProcessBench F1. BoN reranking uses a process+outcome mix | ProcessBench: Qwen2.5-Math-PRM-72B 78.3 and -7B 73.5. QwQ-32B multi-step 79.3. R1-Distill-Qwen-32B multi-step + SC 82.8 (Table 1). PRMBench deliberately excluded as "synthetically perturbed" (App. A) | baseline | The strongest training-free verifier on ProcessBench at an NLP venue. It costs many generated CoTs per step, whereas we use one teacher-forced pass |
| 5 | ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of LLMs | 2511.06209 | ACL 2026, Main (Long) | https://aclanthology.org/2026.acl-long.536/ | FULL (arXiv HTML v3) | A probe of under 10M parameters on the frozen generator's hidden states, or on attention plus top-K logits, predicts step correctness | trains a model (probe). Labels come from DeepSeek-R1, or from the target LLM labelling itself | Self-built step-labelled test sets (MATH, GSM8K, ProofNet, NaturalPlan, StrategyQA, ScienceQA). Step PR-AUC plus BoN and beam search | none of the four (cites ProcessBench and PRMBench but does not evaluate on them). Step PR-AUC .604 (self-annotated hidden-state probe) vs .565 for Qwen2.5-Math-PRM-7B (Table 1, Qwen3-8B). Includes UQ baselines MaxProb, entropy, PPL, CCP, P(True) | baseline | Shows the generator's own internal signals match off-the-shelf PRMs. It is supervised and replaces the PRM rather than fusing with it |
| 6 | Dissecting Failure Dynamics in Large Language Model Reasoning (GUARD) | 2604.14528 | ACL 2026, Main (Long) | https://aclanthology.org/2026.acl-long.401/ | FULL (arXiv HTML) | Error onsets (valid→invalid segment) coincide with bursts of segment-mean token entropy. GUARD triggers branching at the q=0.9 entropy quantile and keeps the lowest-entropy branch | none for GUARD. Analysis uses Gemini-3-Pro segment labels with human checks | AIME24/25, AMC23, MATH500, Minerva, LCB, OlympiadBench, GPQA. Pass@1 only, with no localization metric | none of the four. Over 85% of onsets fall in the first 30% of the trace (Fig. 2). Invalid segments have higher entropy, p<0.001 (Fig. 4). QwQ-32B 71.3 vs 61.8 base (Table 1) | context | Independent support that segment-level entropy marks the first error. It never tests localization and never combines entropy with a PRM |
| 7 | Streaming Hallucination Detection in Long Chain-of-Thought Reasoning | 2601.02170 | ACL 2026, Findings | https://aclanthology.org/2026.findings-acl.1064/ | FULL (arXiv HTML, first ~2/3) | Single-layer hidden-state probe per step (exponential token weighting) plus a cumulative prefix-level state predictor | supervised (step labels from Claude Sonnet 4.5; 96% agreement on a 5% human sample) | Own data from MuSiQue and BBH (10k+ traces, 200k+ steps). Metrics: AUC/ACC/F1 and 8 dynamic metrics | none of the four. Step AUC 87.83 / 86.70 / 93.27 for Llama-3.1-8B / Qwen2.5-7B / R1-Distill-8B (Table 2). No PRM baseline | baseline / context | Step-level hallucination localization in CoT from internal states, but supervised, not math-PRM benchmarks, and no PRM comparison |
| 8 | HALT: Hallucination Assessment via Log-probs as Time series | 2602.02888 | EMNLP 2026, Main (program 1082-MAIN) | program sheet via https://2026.emnlp.org/program/ | FULL (arXiv HTML) | Teacher-forced top-20 log-probs plus 5 entropy and rank features per token, fed as a time series to a 5M-parameter Bi-GRU | trains a model (response-level binary labels) | HUB (10 clusters incl. GSM8K, MATH and AQuA reasoning from CriticBench). Metric: macro-F1 | none of the four. HALT-L overall macro-F1 67.0 vs Lettuce 64.0. Aggregate PPL and entropy baselines score 32.9–48.0 (Table 4) | context | Its input is the closest to our gray-box telemetry (teacher-forced top-k log-prob sequences), but it is supervised and response-level, not step-level |
| 9 | R-PRM: Reasoning-Driven Process Reward Modeling | 2503.21295 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.679/ | FULL (arXiv HTML, Tables 1–2) | Generative PRM trained by SFT then DPO on LLaMA-3.3-70B evaluation rationales. Averages 10 sampled evaluations per step at inference | trains a model | ProcessBench F1, PRMBench PRMScore, BoN | **PRMBench PRMScore 66.8 (DPO) / 64.9 (SFT) vs 65.5 for Qwen2.5-Math-PRM-7B (Table 2)**. ProcessBench 70.4 vs 73.5 (Table 1) | baseline | One of the few NLP-venue papers giving PRMScore with a Qwen2.5-Math-PRM-7B reference row. Its comparison numbers can go straight into our table |
| 10 | The Lessons of Developing Process Reward Models in Mathematical Reasoning | 2501.07301 | ACL 2025, Findings | https://aclanthology.org/2025.findings-acl.547/ | FULL (arXiv HTML) | Qwen2.5-Math-PRM-7B/72B, trained on hard MC labels with consensus filtering by a Qwen2.5-72B LLM judge | trains a model | ProcessBench F1, BoN | ProcessBench avg F1: 7B 73.5, 72B 78.3 (Table 7). PRMBench not evaluated | baseline | Origin of the frozen PRM we boost. It documents PRM drift toward outcome checking (minimum score often at the last step) |
| 11 | The Bidirectional Process Reward Model (BiPRM) | 2508.01682 | ACL 2026, Main (Long) | https://aclanthology.org/2026.acl-long.572/ | FULL (arXiv HTML) | Adds a right-to-left (reversed-prompt) PRM stream to the usual left-to-right one and fuses them with a learned sigmoid gate | trains a model (both streams and the gate). The "w/o Dynamic" static average still uses trained streams | ProcessBench F1 (its only step-level benchmark). BoN on GSM-Plus and MATH500 | ProcessBench avg F1 30.2 vs 24.6 for L2R (Table 3, Qwen2.5-Math-1.5B on Math-Shepherd). The claimed 37.7% relative step gain does not match the Table 3 averages | context | Fuses two views of one PRM at the step level, but supervised and with small PRMs. It shows a second view adds signal |
| 12 | Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction (MASC) | 2510.14319 | ACL 2026, Findings | https://aclanthology.org/2026.findings-acl.1168/ | ABS | Unsupervised step-level anomaly score from next-step embedding reconstruction error plus a prototype prior of normal steps | none (unsupervised) | Who&When step-level error detection. Metric: AUC-ROC | none of the four. Up to +7.8% AUC-ROC in the without-ground-truth setting (abstract) | context | One of the few NLP-venue label-free step-level error detectors, though in the agent domain |
| 13 | Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier (FlexiVe) | 2505.11966 | ACL 2026, Main (Long) | https://aclanthology.org/2026.acl-long.2190/ | ABS | Generative verifier with fast and slow modes, where GRPO trains the fast mode. Likelihood probing detects when a solution is complete | trains a model | ProcessBench, AIME24 | ProcessBench: claims open-source SOTA above GenPRM-32B with about 2.3× fewer TFLOPS and 15× less training data (abstract; no number read). AIME24 83.3% | baseline | A recent trained generative-verifier reference on ProcessBench |
| 14 | Dyve: Thinking Fast and Slow for Dynamic Process Verification | 2502.11157 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.1136/ | FULL (arXiv HTML, Table 1, Sec. 3.3) | R1-Distill-Qwen-14B is fine-tuned on step labels that combine MC rollouts with consensus filtering by DeepSeek-V3 | trains a model | ProcessBench F1 | ProcessBench F1 68.5 / 58.3 / 49.0 / 47.2 on GSM8K / MATH / Olympiad / Omni (Table 1) | baseline | Trained-verifier reference row |
| 15 | LLMs cannot spot math errors, even when allowed to peek into the solution | 2509.01395 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.553/ | ABS | First-error-step localization by prompting, with and without the reference solution. Proposes writing a corrected intermediate solution first | none (prompting) | VtG and PRM800K. Metric: first-error accuracy | none of the four | context | Shows strong LLMs fail at first-error localization even with the reference solution |
| 16 | Chain-of-Thought Prompting Obscures Hallucination Cues in LLMs: An Empirical Evaluation | 2506.17088 | EMNLP 2025, Findings | https://aclanthology.org/2025.findings-emnlp.67/ | ABS | Shows CoT changes internal states and token-probability distributions and weakens mainstream hallucination detectors | none (study) | Several hallucination benchmarks; detection accuracy and score distributions | none of the four | context | A warning for our telemetry channel: probability and entropy cues may weaken on CoT text, so their effect inside the fusion should be measured |
| 17 | Unsupervised Hallucination Detection by Inspecting Reasoning Processes (IRIS) | 2509.10004 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.1124/ | FULL (arXiv HTML) | MLP probe on verification-response embeddings, trained with the model's own uncertainty (entropy or verbalized confidence) as soft pseudo-labels | weak (self-generated pseudo-labels, no gold labels) | True-False, HaluEval2, HELM. Metric: accuracy. Statement-level | none of the four. Average accuracy 90.38 / 70.57 / 68.43 (Table 1) | context | A label-free recipe that bootstraps from uncertainty. It works per statement, not per step |
| 18 | GRACE: Step-Level Benchmark for Faithful Reasoning over Context | 2606.16151 | EMNLP 2026, Main (program 4696-MAIN; arXiv comment "Accepted at EMNLP Main 2026") | program sheet via https://2026.emnlp.org/program/ | FULL (arXiv HTML) | New human-annotated step-level faithfulness benchmark with 2 tracks (Inference, Grounding) and 4 error categories each | n/a (benchmark). Test split is human-labelled, train split labelled by LLM consensus | MuSiQue, 2Wiki, ReClor, LogiQA traces from 10 generators. 7,352 traces / 31,656 steps; test 437 traces / 2,044 steps (Table 2). Metrics: Step F1, Category F1 | none of the four. Best Step F1 81.46 (Gemini-3.1-Pro, Table 3). No PRM or entropy baseline | context (new benchmark) | A non-math step-labelled set if we need transfer beyond math |
| 19 | SHARP: Span-level Hallucination Annotation for Reasoning Paths in Generated Mathematical Solutions | none found | EMNLP 2026, Findings (program 1938-FIND) | program sheet via https://2026.emnlp.org/program/ | CARD (HF dataset `ZaandaTeika/SHARP`; matched to the paper by name only) | Math step- and span-level hallucination labels on Mathstral-7B solutions, with `<HAL>` spans | n/a (benchmark). LLM-annotated (gpt-oss-120b), per-step κ 0.689 with an independent annotator | SHARP-GSM8K 7,559 / 840 and SHARP-Math 10,021 / 1,114 (train/test) | none of the four | context (new benchmark) | A math step-labelled set with natural (non-perturbed) errors. It could serve as an extra external test. Paper text not yet available |
| 20 | ToolPRMBench: Evaluating and Advancing Process Reward Models for Tool-using Agents | 2601.12294 | ACL 2026, Findings | https://aclanthology.org/2026.findings-acl.602/ | ABS | Step-level PRM benchmark from agent trajectories (correct vs plausible-wrong action), with labels verified by multiple LLMs | n/a (benchmark) | Tool-use step pairs | none of the four | context (new benchmark) | Step-labelled benchmark outside math |
| 21 | MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification | 2503.12505 | ACL 2025, Findings | https://aclanthology.org/2025.findings-acl.1112/ | ABS | Multimodal PRM benchmark with three tasks: step correctness, answer aggregation and reasoning-process search | n/a (benchmark) | MPBench | none of the four | context (new benchmark) | Multimodal analogue of ProcessBench and PRMBench |
| 22 | ViLBench: A Suite for Vision-Language Process Reward Modeling | 2503.20271 | EMNLP 2025, Main | https://aclanthology.org/2025.emnlp-main.344/ | ABS | Vision-language PRM benchmark plus the ViLPRM model | n/a / trains a model | ViLBench | none of the four | context (new benchmark) | Multimodal step-reward benchmark |
| 23 | Verifying the Steps of Deductive Reasoning Chains | none found | ACL 2025, Findings | https://aclanthology.org/2025.findings-acl.25/ | ABS | Hand-annotated soundness of each deduction step. Compares LLM, symbolic and neuro-symbolic (VANESSA) verifiers | n/a (benchmark) | Own logical-deduction step dataset | none of the four | context (new benchmark) | Step-labelled logic benchmark |
| 24 | FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language Model Mathematical Reasoning | 2410.06304 | EMNLP 2025, Findings | https://aclanthology.org/2025.findings-emnlp.228/ | ABS | Six type-specific discriminative PRMs trained on automatically synthesised step-level hallucinations | weak (synthetic labels) / trains a model | GSM8K and MATH: step hallucination-type detection and verification (BoN) | none of the four | context | Weakly supervised step-level hallucination detection in math |

## Which benchmarks do these papers share? (counts over the 24 rows above)

| benchmark | count | papers (row #) |
|---|---|---|
| **ProcessBench** (F1) | **6** | 4, 9, 10, 11, 13, 14 (all are trained PRMs/verifiers or prompted reasoning models) |
| **PRMBench** (official PRMScore) | **1** | 9 (R-PRM). Row 3 uses PRMBench problems with regenerated chains and no PRMScore. Row 4 explicitly excludes PRMBench |
| **Socratic-PRMBench** | **0** | none |
| **Hard2Verify** | **0** | none (only Hard2Verify itself, which is on the known list) |
| PRM800K as a test set | 1 | 15 (row 5 trains on PRM800K problems) |
| Self-built step labels on GSM8K/MATH-derived traces (PRM, LLM or synthetic labels) | 5 | 2, 3, 5, 19, 24 |
| Best-of-N or final-answer accuracy as the outcome | 7 | 1, 2, 4, 5, 9, 10, 11 |
| Pass@1 on AIME/MATH-style sets (no step metric) | 2 | 6, 13 |
| Multi-hop QA step labels (MuSiQue etc.) | 2 | 7, 18 |
| Multimodal step benchmarks | 3 | 1, 21, 22 |
| Agent step benchmarks (Who&When, ToolPRMBench) | 2 | 12, 20 |
| Response- or statement-level hallucination sets | 3 | 8, 16, 17 |

Takeaway: ProcessBench is the shared standard. Only one NLP-venue paper outside our known list reports official PRMScore, and it is a trained PRM. None reports Socratic-PRMBench or Hard2Verify. The label-free internal-signal step detectors (rows 2, 6, 12) all evaluate on their own labels and never on PRMBench or ProcessBench.

## Venue updates for papers already on the known list (verified)

| known paper | venue found | evidence |
|---|---|---|
| ProcessBench (2412.06559) | ACL 2025, Main (Long) | https://aclanthology.org/2025.acl-long.50/ |
| DeltaBench (2502.19361) | ACL 2025, Main (Long), pp. 18468–18489 | https://aclanthology.org/2025.acl-long.905/ |
| Hard2Verify (2510.13744) | ACL 2026, Main (Long), pp. 22502–22517 | https://aclanthology.org/2026.acl-long.1031/ |
| Temporal Consistency (2503.14495) | EMNLP 2025, **Findings**, pp. 22114–22129 | https://aclanthology.org/2025.findings-emnlp.1205/ |
| PathFinder-PRM (2505.19706) | EMNLP 2025, **Findings**, pp. 14940–14954 | https://aclanthology.org/2025.findings-emnlp.807/ |
| ScalePRM (2512.03244) | EMNLP 2026, Main (program 4293-MAIN) | official program sheet |
| CROP (2605.30085) | EMNLP 2026, Main, confirmed (program 5410-MAIN; arXiv comment says Oral) | program sheet + arXiv |
| HSRM (2608.30841) | EMNLP 2026, Main, confirmed (program 5293-MAIN) | program sheet |
| CPMI (2604.10660) | ACL 2026, Main, confirmed | https://aclanthology.org/2026.acl-long.1744/ |
| ARES (2507.12948) | EMNLP 2025, Main, confirmed (2025.emnlp-main.382). It also has a NeurIPS 2025 listing | Anthology preview / neurips.cc |

Known papers with a non-NLP venue only (outside scope; noted so they are not searched again):
- FUSE 2604.18547: ICML 2026 poster.
- Weaver 2506.18203: NeurIPS 2025.
- PRM uncertainty calibration 2506.09338: NeurIPS 2025 poster.
- Reward Under Attack 2603.06621: ICML 2026 poster.
- ThinkPRM 2504.16828: ICML 2026 virtual poster page.
- 2610.00571 (titled "Interpreting Reasoning of LLMs via Partial Information Decomposition" on arXiv): ICLR 2026 workshop, per its arXiv comment. **Check this ID**: it is listed as "SLIDER" on our known list, but arXiv gives this title.

No NLP-venue evidence was found for these known IDs:
- They were not in the EMNLP 2026 program scan: PRISM 2606.09078, uPRM 2605.10158, VerifySteer 2605.20745, Socratic-PRMBench 2505.23474, GR-Ben 2605.01203, MedPRMBench 2604.17282, 2605.15529, 2605.06785, 2608.12585, 2511.21522, 2610.18089, 2609.22700, 2609.02817.
- Two scan caveats: the sheet lists presenters only, and I scanned it by keyword through a fetch tool. A missing title is therefore not proof that the paper was rejected.

## Seen but excluded, or worth a later look

- **EMNLP 2026 titles in the program with no arXiv version found (read once the Anthology is posted):**
  - "SCOPE: Stepwise Calibration of Reasoning Confidence with Outcome-Grounded Prefix Evidence" (8952-MAIN)
  - "When Does Reasoning Age? Survival Analysis of Step-Level Error Hazard in LLM Chains" (6208-MAIN)
  - "MathDebugger: A Benchmark for Type-Aware Error Detection in Synthetic Math Data" (1611-MAIN)
  - "FIVER: Adaptive Fusion of Formal and Informal Verification for Mathematical Reasoning in LLMs" (2650-FIND)
  - "Target-Local Verifier Choice in Best-of-K Reasoning Selection" (1451-FIND)
  - Of these, SCOPE and "When Does Reasoning Age" look the most relevant.
- **EMNLP 2026, trained-PRM papers (titles only):**
  - The Weakest Link Tells It All (1265-MAIN, 2606.27739)
  - CePRM (7382-FIND)
  - ARMOR-PRM (4887-FIND)
  - SCRIBE (2162-MAIN)
  - Process Rewarding via Solution-Space Alignment (6156-MAIN)
- **Tangential, excluded:**
  - Final-answer or trace-level only: One-Token Verification (EMNLP 2026 Main, 2603.01025); ReFeri (EMNLP 2026 Main, 2506.17251); CLUE (ACL 2026, 2026.acl-long.788); LLM Reasoning as Trajectories (ACL 2026).
  - Trace pruning: STEP (Findings ACL 2026, 2601.09093).
  - RL credit assignment: Step Potential (Findings ACL 2026).
  - Trained PRMs on new data sources: PRMs Meet Planning (ACL 2026, 2604.17957); FoVer (Findings ACL 2026).
- **Relevant but with no NLP venue found:**
  - Stepwise Confidence Attribution 2605.19228 (ICML 2026)
  - Step-level hidden-state transport geometry 2605.13772 (arXiv only; evaluates on ProcessBench)
  - NCV 2510.02816 (training-free, ProcessBench; arXiv only)
  - LLMs as a Jury 2607.10139 (arXiv only)
  - EDU-PRM 2503.22233 (arXiv only)
  - Uncertainty-aware step-wise verification 2502.11250 (ICLR 2025 workshop)
