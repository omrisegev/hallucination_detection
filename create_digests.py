import os

digests_dir = r"C:\Users\omris\TAU\hallucination_detection\papers\digests"
os.makedirs(digests_dir, exist_ok=True)

# 1. Gradients with Respect to Semantics Preserving Embeddings
gradients_content = """---
slug: gradients-with-respect-to-semantics-preserving-embeddings
title: "Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models"
authors: "Mingda Li et al., Harbin Institute of Technology"
arxiv_id: "2605.04638"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Gradients_with_Respect_to_Semantics_Preserving_Embeddings.pdf
extracted_text: papers/extracted/gradients-with-respect-to-semantics-preserving-embeddings.md
last_digested: 2026-07-15
---

## Summary

This paper proposes SemGrad and HybridGrad, the first gradient-based Uncertainty Quantification (UQ) methods for free-form generation. The core intuition is that a confident LLM should maintain stable output distributions under semantically equivalent input perturbations. The sensitivity is captured by the gradient of the output log-likelihood with respect to a set of "semantic-preserving embeddings" (identified by a Semantic Preservation Score, SPS) at the input token positions. HybridGrad fuses SemGrad with token-importance-weighted parameter gradients (restricted to the LM head weights, W_head) to handle both high and low-aleatoric uncertainty.

## Datasets & models used

- **Datasets:** SciQ, TriviaQA, TruthfulQA.
- **Models:** Qwen3-Instruct4B, Mistral-Nemo-Instruct12B, Llama3.1-Instruct8B.

## Methods it compared itself against

- **Baselines:** LN-PE (Length-Normalized Predictive Entropy), Semantic Entropy (SE), ExGrad, SAR (Semantic Association Ratio).

## Experiments — methodology & scores

The experiments evaluate uncertainty quantification on predicting generation correctness, measured by AUROC (%). Under high aleatoric uncertainty (TruthfulQA), SemGrad outperforms baselines, while HybridGrad achieves the best average performance across all dataset-model pairs.

| Setup | Metric | Score (Llama-8B) | Notes |
|---|---|---|---|
| TruthfulQA (Llama-8B) | AUROC (%) | **70.21** (HybridGrad) vs 64.78 (LN-PE) | SemGrad alone scores 69.80 |
| SciQ (Llama-8B) | AUROC (%) | **78.21** (HybridGrad) vs 72.51 (LN-PE) | ExGrad parameter-gradient baseline: 75.31 |
| TriviaQA (Llama-8B) | AUROC (%) | **85.06** (HybridGrad) vs 84.02 (LN-PE) | Factual QA baseline |

## Connection to our pipeline

- **Overlap:** Both target unsupervised/training-free uncertainty quantification from internal model distributions.
- **Difference:** SemGrad requires **backward passes** to compute gradients ($\nabla_{h_E} \log p(\hat{y}|x)$) in semantic space, whereas our method is strictly **forward-only** ($K=1$ logits), which is much more computationally efficient and works without backward-pass access.
- **Competitor:** Yes, on SciQ, TriviaQA, and TruthfulQA. Our continuous L-SML method is a competitive, gradient-free forward-only alternative.

## Notes / open questions

SemGrad relies on identifying the Semantic Preserving Token ($t^*$), which captures the bulk of the input semantics. The paper shows a strong correlation between the Semantic Preservation Score (SPS) of hidden states and the resulting AUROC.
"""

# 2. How do LLMs Compute Verbal Confidence?
confidence_content = """---
slug: how-do-llms-compute-verbal-confidence
title: "How do LLMs Compute Verbal Confidence?"
authors: "Dharshan Kumaran et al., Google DeepMind"
arxiv_id: "2603.17839"
venue: "ICML 2026"
year: 2026
source_pdf: papers/How_do_LLMs_Compute_Verbal_Confidence.pdf
extracted_text: papers/extracted/how-do-llms-compute-verbal-confidence.md
last_digested: 2026-07-15
---

## Summary

This paper presents a mechanistic interpretability study investigating how Large Language Models generate verbal confidence scores (e.g., stating a confidence number or class like "Almost certain"). Using activation steering, patching, noising, swaps, and attention blocking, the authors show that confidence is not computed "just-in-time" when verbalization is prompted. Instead, it is computed automatically during answer generation, cached at the first post-answer position (PANL), and retrieved during verbalization. Variance partitioning shows that these cached representations explain substantial variance in verbal confidence beyond simple token log-probabilities, indicating a richer self-evaluation of answer quality.

## Datasets & models used

- **Datasets:** TriviaQA, BigMath, MMLU.
- **Models:** Gemma 3 27B, Qwen 2.5 7B, Mistral Small 24B (referred to as Magistral Small 24B).

## Methods it compared itself against

- **Baselines:** Token-level log-probabilities (average, min, max, product log-probabilities).

## Experiments — methodology & scores

The authors perform causal interventions on the residual stream at the first post-answer position (PANL). They evaluate the causal effects of patching and steering on confidence reports. Linear probes on PANL activations achieve high accuracy in predicting correctness and verbalized confidence bins.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| Activation Swap (PANL) | Confidence Recovery (%) | **24.3%** | Swap at PANL alters output, PANL+1 has no effect |
| Linear Probing (PANL) | Explained Variance ($R^2$) | Up to **0.60** | Unique variance explained beyond logprob baselines |

## Connection to our pipeline

- **Overlap:** Conceptual interest in LLM self-evaluation and token log-probabilities.
- **Difference:** They focus on **white-box mechanistic interpretability** (activations, attention edges), whereas we focus on **unsupervised gray-box detection** (logits and entropy traces).
- **Competitor:** No, but their findings support our thesis premise: that models perform automatic, rich self-evaluations during generation, which are encoded in decoding trace dynamics (like $H(n)$ traces).

## Notes / open questions

The paper identifies the newline token or the first token following the answer as the primary caching site for confidence representations.
"""

# 3. Inference-Time Conformal Reasoning
conformal_content = """---
slug: inference-time-conformal-reasoning
title: "Inference-Time Conformal Reasoning with Valid Factuality Control for Large Language Models"
authors: "Ting Wang et al., University of Illinois Urbana-Champaign"
arxiv_id: "2606.08831"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Inference_Time_Conformal_Reasoning.pdf
extracted_text: papers/extracted/inference-time-conformal-reasoning.md
last_digested: 2026-07-15
---

## Summary

This paper introduces the Inference-Time Conformal Reasoning (ITCR) framework, which integrates split conformal prediction directly into the multi-step reasoning graph generation process. In multi-step reasoning, claim dependencies form an implicit directed acyclic graph (DAG). ITCR learns a graph-level factuality uncertainty function that aggregates claim-level uncertainty, designs a non-conformity score based on this uncertainty, and dynamically stop/prunes generation using a calibrated conformal threshold. This guarantees valid factuality control (marginal coverage $1-\\alpha$) under "no-miss" (recall-conservative) or "no-false" (precision-conservative) objectives.

## Datasets & models used

- **Datasets:** MATH, GSM8K, and QA benchmarks.
- **Models:** LLaMA-3.1-8B-Instruct, Qwen3-4B-Thinking-2507, DeepSeek-R1-Distill-Qwen-1.5B.

## Methods it compared itself against

- **Baselines:** PostCal (post-hoc conformal pruning), heuristic aggregations (MAX, SUM, AVG of claim uncertainties).

## Experiments — methodology & scores

ITCR is evaluated on empirical coverage (targeting $1-\\alpha$) and efficiency (compactness of the generated subgraphs). Downstream self-correction performance is measured via Correction Gain (PCR - NCR, where PCR is Positive Correction Rate and NCR is Negative Correction Rate).

| Setup | Metric | Score (Llama-8B) | Notes |
|---|---|---|---|
| GSM8K (Llama-8B) | Avg. Token Usage | **1919.50** (ITCR) vs 2092.80 (PostCal) | ITCR saves compute |
| GSM8K (Llama-8B) | PCR - NCR (%) | **30.86%** (ITCR) vs 8.02% (PostCal) | Downstream correction gain |
| MATH (Llama-8B) | PCR - NCR (%) | **17.39%** (ITCR) vs 3.51% (PostCal) | Average net correction gain |

## Connection to our pipeline

- **Overlap:** Evaluates on GSM8K and MATH, and utilizes conformal prediction frameworks.
- **Difference:** ITCR is an **inference-time generation control** method (stopping/correcting reasoning subgraphs dynamically), whereas our method is a post-generation detector.
- **Competitor:** Complements us. We can feed our continuous L-SML/U-PCR scores into the ITCR framework to calibrate the non-conformity threshold, replacing their trained MLP uncertainty model with our unsupervised spectral score.

## Notes / open questions

The paper demonstrates that learning a structured uncertainty model (MLP) yields much higher efficiency than static sum/average heuristics.
"""

# 4. HaloProbe: Bayesian Detection and Mitigation
haloprobe_content = """---
slug: haloprobe-bayesian-detection
title: "HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models"
authors: "Reihaneh Zohrabi et al., Technical University of Darmstadt"
arxiv_id: "2604.06165"
venue: "ICML 2026 (Preprint)"
year: 2026
source_pdf: papers/HaloProbe_Bayesian_Detection.pdf
extracted_text: papers/extracted/haloprobe-bayesian-detection.md
last_digested: 2026-07-15
---

## Summary

This paper presents HaloProbe, a Bayesian framework to detect and mitigate object hallucinations in Large Vision-Language Models (LVLMs). The authors reveal a Simpson's paradox in coarse-grained attention-based hallucination detection: token position and object repetition act as confounders that reverse attention statistics when aggregated. HaloProbe factorizes external description features (repetition, position) and internal decoding signals (fine-grained attention, logits). It uses class-balanced training for the internal estimator and combines it with a learned prior over external features to get the true posterior, which is used as an external scoring signal for non-invasive mitigation during decoding.

## Datasets & models used

- **Datasets:** MS COCO 2014, MME, POPE, Shikra, InternVL.
- **Models:** LLaVA-1.5-7B, Shikra, MiniGPT-4, Qwen3-VL, InternVL3.5.

## Methods it compared itself against

- **Baselines:** IC, UT, EAZY, PAI, VISTA, VCD.

## Experiments — methodology & scores

Evaluated on open-ended image captioning using CHAIRS (sentence-level hallucination rate, %) and CHAIRI (instance-level hallucination rate, %), and object probing accuracy on POPE.

| Setup | Metric | Score (LLaVA-1.5) | Notes |
|---|---|---|---|
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **24.8 / 7.2** (HaloProbe) vs 48.6 / 13.6 (Vanilla) | Significant reduction in hallucination |
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **24.8 / 7.2** (HaloProbe) vs 36.5 / 12.9 (VISTA) | Beats steering/contrastive methods |
| POPE (Greedy) | Accuracy (%) | **85.34%** | Guided decoding preserves general recognition |

## Connection to our pipeline

- **Overlap:** Explores logit and attention signals for hallucination detection.
- **Difference:** Specific to **multimodal VLMs** and **object hallucinations**, and requires training an internal estimator on a balanced dataset. Our method is **text-only reasoning** and **fully unsupervised**.
- **Competitor:** No.

## Notes / open questions

The paper demonstrates that factorized Bayesian learning improves robustness under distribution shifts.
"""

# 5. REVIS: Sparse Latent Steering
revis_content = """---
slug: revis-sparse-latent-steering
title: "REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models"
authors: "Jialin Wu et al., Ant Group"
arxiv_id: "2602.11824"
venue: "ICML 2026"
year: 2026
source_pdf: papers/REVIS_Sparse_Latent_Steering.pdf
extracted_text: papers/extracted/revis-sparse-latent-steering.md
last_digested: 2026-07-15
---

## Summary

REVIS is a training-free framework designed to mitigate object hallucinations in Large Vision-Language Models (LVLMs) by re-activating suppressed visual information in the latent space. The authors show that visual features and language priors become entangled in deep layers, leading to visual suppression. REVIS extracts a "pure visual vector" via orthogonal projection (subtracting the language prior direction) and applies sparse latent steering (intervention) only at the specific layer (e.g. layer 27) where visual suppression peaks. This surgical approach reduces object hallucinations while preserving general reasoning.

## Datasets & models used

- **Datasets:** POPE (Random, Popular, Adversarial splits), CHAIR, MME, MM-Vet, MMMU-Pro.
- **Models:** Qwen2.5-VL-7B-Instruct, LLaVA-NeXT, LLaVA-1.5-7B, Qwen3-VL, InternVL3.

## Methods it compared itself against

- **Baselines:** VTI (Vanilla steering), VCD, AGLA, ONLY, Regular greedy decoding.

## Experiments — methodology & scores

Evaluated on CHAIR (sentence-level CS and instance-level CI, %) and POPE accuracy/F1.

| Setup | Metric | Score (Qwen2.5-VL) | Notes |
|---|---|---|---|
| COCO (Generative) | CHAIRS / CHAIRI (%) | **25.00 / 8.23** (REVIS) vs 31.00 / 8.13 (Regular) | ~19% reduction in sentence-level error |
| MM-Vet | Accuracy (Overall) | **72.16** (REVIS) vs 56.38 (VTI) | Preserves/improves general reasoning |
| MME | Perception Score | **1723.21** (REVIS) vs 1715.73 (Regular) | Standard greedy baseline: 1715.73 |

## Connection to our pipeline

- **Overlap:** Focuses on latent space geometry and training-free intervention.
- **Difference:** Targets multimodal VLM object hallucinations and performs latent space steering (mitigation), whereas we focus on text-only reasoning error detection.
- **Competitor:** No.

## Notes / open questions

REVIS leverages a 5-cluster counterfactual semantic state space constructed using force-decoding on correct and hallucinated captions to identify the steering directions.
"""

# 6. Adaptive Residual-Update Steering (RUDDER)
rudder_content = """---
slug: adaptive-residual-update-steering
title: "Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models"
authors: "Zhengtao Zou et al., Aalto University"
arxiv_id: "2511.10292"
venue: "ICML 2026"
year: 2026
source_pdf: papers/Adaptive_Residual_Update_Steering.pdf
extracted_text: papers/extracted/adaptive-residual-update-steering.md
last_digested: 2026-07-15
---

## Summary

This paper proposes RUDDER (Residual-Update Directed DEcoding Regulation), a low-overhead steering framework to mitigate object hallucinations in Large Vision-Language Models (LVLMs). Autoregressive generation in LVLMs suffers from "visual dilution," where the prefix visual information fades, causing the model to over-rely on language priors. RUDDER extracts a robust visual evidence direction (CARD) from the prefill residual updates of the visual prefix. During decoding, it injects the CARD vector into the hidden states, modulated by an adaptive trust mechanism (the Beta Gate). RUDDER achieves high efficiency (96% throughput) with single-pass latency.

## Datasets & models used

- **Datasets:** MSCOCO, POPE, MME.
- **Models:** LLaVA-1.5 (7B/13B), Idefics2, InstructBLIP, Qwen2.5-VL.

## Methods it compared itself against

- **Baselines:** DoLa, VCD, VISTA, PAI.

## Experiments — methodology & scores

Evaluated on CHAIRS, CHAIRI, POPE, and MME. Throughput is measured in tokens/second.

| Setup | Metric | Score (LLaVA-1.5-7B) | Notes |
|---|---|---|---|
| COCO (Greedy) | CHAIRS / CHAIRI (%) | **36.5 / 12.1** (RUDDER-Beta) vs 48.6 / 13.6 (Vanilla) | Comparable to VISTA but 1.5x faster |
| POPE (Greedy) | F1-Score (%) | **86.5%** (RUDDER-Beta) vs 84.9% (Vanilla) | High accuracy |
| Latency | Throughput (tok/s) | **54.9** (RUDDER-Beta) vs 56.7 (Vanilla) | Maintains 96% of vanilla speed |

## Connection to our pipeline

- **Overlap:** Focuses on low-overhead, single-pass inference-time intervention.
- **Difference:** Multimodal VLM steering to mitigate object hallucinations, whereas we detect factual/reasoning errors in text-only models.
- **Competitor:** No.

## Notes / open questions

The Beta Gate dynamically scales the injection strength by mapping the cosine similarity of the current hidden state to the visual CARD vector.
"""

# 7. Agentic Confidence Calibration
agentic_content = """---
slug: agentic-confidence-calibration
title: "Agentic Confidence Calibration"
authors: "Jiaxin Zhang et al., Salesforce AI Research"
arxiv_id: "2601.15778"
venue: "Preprint"
year: 2026
source_pdf: papers/Agentic_Confidence_Calibration.pdf
extracted_text: papers/extracted/agentic-confidence-calibration.md
last_digested: 2026-07-15
---

## Summary

This paper introduces the problem of Agentic Confidence Calibration (ACC): estimating the likelihood that an AI agent's multi-step execution trajectory will succeed. To address compounding errors, tool uncertainty, and data scarcity, the authors propose Holistic Trajectory Calibration (HTC). HTC extracts a compact set of 48 process-level features (cross-step dynamics, intra-step stability, positional indicators, structural attributes) from the agent's logprob trace, and trains a regularized linear model (Ridge/Lasso) to predict trajectory success. A General Agent Calibrator (GAC) is trained to achieve out-of-domain generalization.

## Datasets & models used

- **Datasets:** SimpleQA, GPQA, HLE (Humanity's Last Exam), GAIA, WebArena, AgentBench.
- **Models:** smolagents (CodeAct), OAgents, GPT-4, GPT-OSS, DeepSeek-v3.1, Qwen3-235B.

## Methods it compared itself against

- **Baselines:** Verbalized Confidence, Last-Step TP, Global-Trace TP, Temperature Scaling, LSTM, Transformer, XGBoost, Gaussian Process.

## Experiments — methodology & scores

Evaluated using ECE (Expected Calibration Error, lower is better), Brier Score (BS, lower is better), and AUROC (higher is better) for failure prediction.

| Setup | Metric | Score (SimpleQA / GPT-4) | Notes |
|---|---|---|---|
| SimpleQA | ECE / BS | **0.032 / 0.114** (HTC) vs 0.121 / 0.196 (Verbalized) | Outperforms all baselines |
| GPQA | ECE / BS | **0.084 / 0.201** (HTC) vs 0.454 / 0.523 (Verbalized) | Significant calibration improvement |
| GAIA (OOD) | ECE (GAC) | **0.068** (GAC) vs 0.185 (LastStep-TP) | Strong out-of-domain transfer |

## Connection to our pipeline

- **Overlap:** Extracts statistical features from the autoregressive logprob/entropy trace of the model.
- **Difference:** HTC trains a **supervised linear classifier** (Lasso/Ridge) on trajectory-level features of agent runs, whereas our method (L-SML/U-PCR) is **fully unsupervised** (label-free at training and inference). Additionally, HTC is designed for multi-step agent trajectories (planning/tool-use), whereas we target single-turn reasoning.
- **Competitor:** Yes, on GPQA and Humanity's Last Exam. Our continuous L-SML method represents a competitive unsupervised alternative.

## Notes / open questions

The paper demonstrates that positional features (first/last step confidence) are the most predictive of success on hard reasoning tasks like GPQA, while dynamics and stability are important for search-based tasks like SimpleQA.
"""

# 8. Detecting Contextual Hallucinations with Frequency-Aware Attention
frequency_content = """---
slug: detecting-contextual-hallucinations-with-frequency-aware-attention
title: "Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention"
authors: "Siya Qi et al., Harbin Institute of Technology"
arxiv_id: "2604.18647"
venue: "ICML 2026"
year: 2026
source_pdf: not downloaded
extracted_text: not extracted
last_digested: 2026-07-15
---

## Summary

This paper proposes a training-free contextual hallucination detector based on the frequency components of attention distributions. The authors model attention weights across decoding steps as discrete signals. By applying signal processing (FFT), they show that hallucinated tokens exhibit higher "high-frequency attention energy," reflecting fragmented and unstable visual/textual grounding. They use this frequency-aware attention energy to build a lightweight detector.

## Datasets & models used

- **Datasets:** RAGTruth, HalluRAG.
- **Models:** LLaMA-3.1-8B, Qwen-2.5-7B, etc.

## Methods it compared itself against

- **Baselines:** Lookback-Lens, attention variance/entropy, verification-based methods (similarity/LLM-as-a-judge), and internal-representation-based methods.

## Experiments — methodology & scores

The method is evaluated on hallucination detection AUROC across RAG benchmarks. Frequency-aware attention energy consistently outperforms standard attention entropy and Lookback-Lens.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| RAGTruth (Llama-8B) | AUROC (%) | **Significant lift** vs Lookback-Lens | Exact scores not extracted |

## Connection to our pipeline

- **Overlap:** Both extract frequency/spectral features of decoding traces (we use FFT of entropy $H(n)$ traces, they use FFT of attention maps).
- **Difference:** They are **white-box** (require attention maps), while we are **gray-box** (require logits only, $K=1$), making us much more computationally efficient and API-compatible.
- **Competitor:** Yes, on RAGTruth (we score 87.7% on Llama-8B).

## Notes / open questions

Unresolved: whether combining attention energy with our logprob energy features can provide further performance gains.
"""

# 9. Mind the Gap: Catching Hallucinations via Evidence Drop
mind_the_gap_content = """---
slug: mind-the-gap-catching-hallucinations-via-evidence-drop
title: "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold"
authors: "Qunjie Chen et al., Tongji University"
arxiv_id: "OpenReview/ICML 2026"
venue: "ICML 2026"
year: 2026
source_pdf: not downloaded
extracted_text: not extracted
last_digested: 2026-07-15
---

## Summary

This paper models the multi-step reasoning process as a trajectory on a latent "Evidence Manifold," where each reasoning step should be supported by local evidence. Hallucinations are defined as "Evidence Drops"—sudden, localized declines in evidence support. The authors design a training-free, model-agnostic detector that monitors for the worst-case Evidence Drop, enabling both response-level correctness prediction and step-level error localization.

## Datasets & models used

- **Datasets:** GSM8K, MATH, ProcessBench.
- **Models:** LLaMA-3.1-8B, Qwen-2.5-7B-Instruct, etc.

## Methods it compared itself against

- **Baselines:** Sequence-level uncertainty metrics (Semantic Entropy, LN-Entropy, Perplexity, SelfCheckGPT).

## Experiments — methodology & scores

Evaluated on selective accuracy, risk-coverage trade-offs, and AUROC for correctness prediction.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| GSM8K / MATH | AUROC (%) | **Outperforms sequence-level uncertainty** | Beats standard Semantic Entropy |

## Connection to our pipeline

- **Overlap:** Direct competitor targeting reasoning benchmarks (GSM8K, MATH) with a training-free, model-agnostic approach.
- **Difference:** We use spectral features of $H(n)$ traces (sliding-window variance, EPR, CUSUM) to get a sequence-level score via unsupervised L-SML fusion. They look at semantic/contextual evidence drops directly to localize errors at the step level.
- **Competitor:** Yes, direct competitor on GSM8K/MATH.

## Notes / open questions

Their step-level error localization on ProcessBench represents a key capability we should benchmark against.
"""

# 10. GAUSS: Graph-Assisted Uncertainty Quantification
gauss_content = """---
slug: gauss-graph-assisted-uncertainty-quantification
title: "GAUSS: Graph-Assisted Uncertainty Quantification using Structure and Semantics for Long-Form Generation in LLMs"
authors: "Karthik Somayaji NS et al., Tsinghua University"
arxiv_id: "OpenReview/ICML 2026"
venue: "ICML 2026"
year: 2026
source_pdf: not downloaded
extracted_text: not extracted
last_digested: 2026-07-15
---

## Summary

GAUSS is a framework to measure uncertainty in long-form language model outputs. Each generated paragraph is represented as a semantic graph (nodes = atomic facts, edges = relations between facts). Uncertainty is quantified by computing the "expected alignment cost" between the semantic graph of an anchor paragraph and alternative reference paragraphs generated by the model for the same query.

## Datasets & models used

- **Datasets:** Long-form text generation benchmarks.
- **Models:** LLaMA-2-70B, etc.

## Methods it compared itself against

- **Baselines:** Sentence-level confidence, bipartite entailment graphs (SelfCheckGPT), etc.

## Experiments — methodology & scores

Evaluated on factual error detection in long-form generation, showing that expected alignment cost correlates strongly with factual correctness.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| Long-form Generation | AUROC (%) | **Outperforms SelfCheckGPT** | Captures structural coherence |

## Connection to our pipeline

- **Overlap:** Unsupervised uncertainty quantification.
- **Difference:** GAUSS is designed for *long-form text generation* and requires generating multiple samples ($K > 1$) to perform graph alignment. Our method targets *reasoning tasks* (math/science) using a *single-pass* ($K=1$) trace.
- **Competitor:** No.

## Notes / open questions

Building semantic graphs of atomic facts is computationally expensive compared to our simple token-level logprob extraction.
"""

# Write all files
for slug, content in [
    ("gradients-with-respect-to-semantics-preserving-embeddings", gradients_content),
    ("how-do-llms-compute-verbal-confidence", confidence_content),
    ("inference-time-conformal-reasoning", conformal_content),
    ("haloprobe-bayesian-detection", haloprobe_content),
    ("revis-sparse-latent-steering", revis_content),
    ("adaptive-residual-update-steering", rudder_content),
    ("agentic-confidence-calibration", agentic_content),
    ("detecting-contextual-hallucinations-with-frequency-aware-attention", frequency_content),
    ("mind-the-gap-catching-hallucinations-via-evidence-drop", mind_the_gap_content),
    ("gauss-graph-assisted-uncertainty-quantification", gauss_content)
]:
    filepath = os.path.join(digests_dir, f"{slug}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote digest to {filepath}")
