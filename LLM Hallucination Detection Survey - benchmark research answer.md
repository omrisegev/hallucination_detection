# **Mechanistic and Information-Theoretic Frontiers in Hallucination Detection for Multi-Step Reasoning Models**

The evolution of Large Language Models (LLMs) into specialized Large Reasoning Models (LRMs) has introduced a profound paradigm shift in the nature of generative errors.1 While early iterations of these models primarily struggled with factual retrieval errors—often termed "flat hallucinations"—contemporary reasoning models exhibit a more deceptive and structural failure mode known as reasoning hallucination.1 In these instances, the model generates a logically coherent reasoning trace that appears persuasive to the user but is ultimately grounded in fabricated premises or invalid logical transitions.1 As these models are increasingly integrated into high-stakes environments such as financial analysis, medical diagnostics, and symbolic mathematics, the development of robust, real-time detection mechanisms has become a central focus of the research community.4 This report provides an exhaustive survey of the current state of the art in hallucination detection specifically tailored to multi-step reasoning tasks, covering methods published between 2023 and 2025 that provide quantitative detection scores and uncertainty metrics.

## **The Ontology of Reasoning Hallucination: Mechanistic Foundations and Pattern Identification**

The characterization of hallucinations in reasoning tasks requires a departure from traditional overlap-based metrics like ROUGE or BLEU, which have been shown to correlate poorly with the actual truthfulness of complex chains of thought.7 Instead, recent research focuses on the mechanistic interpretability of internal thinking patterns.1 One of the most influential frameworks in this domain identifies two primary patterns of reasoning failure: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps.1

Reasoning Hallucination Detection (RHD) methodologies frequently utilize what is termed a "Reasoning Score".1 This score is derived by measuring the divergence between logits obtained from projecting late layers of a model into the vocabulary space.1 The underlying theory suggests that genuine deep reasoning involves meaningful transformations in later-layer representations, whereas shallow pattern-matching—a common precursor to hallucination—resembles the statistical distributions of the pre-training data.1 When a model enters a hallucinatory state, its internal state transitions from a structured thinking manifold to a chaotic state characterized by high epistemic uncertainty.8

The paradox of modern reasoning models is that as their capability to handle complex tasks increases, their propensity for "confident hallucinations" often scales alongside performance gains.10 This is particularly evident in models optimized through outcome-based reinforcement learning, which can learn to produce "persuasive" reasoning traces that satisfy a reward function without actually maintaining logical integrity.1 Consequently, the ability to output a continuous detection score based on internal signals is critical for distinguishing between a model that is "thinking" and one that is merely "simulating thought".1

## **Information-Theoretic Trajectories: Entropy Dynamics as a Diagnostic for Logic Inconsistency**

Entropy-based detection has transitioned from static, sequence-level aggregation to dynamic, trajectory-level analysis.12 Traditional uncertainty estimation often collapses the entire generation into a single mean entropy value, which frequently fails to capture the local instability indicative of a logical slip.8

### **The Entropy Dynamics Instability Score (EDIS)**

The Entropy Dynamics Instability Score (EDIS) represents a significant advancement in unsupervised hallucination detection.12 By analyzing the token-level entropy trajectory, EDIS identifies specific "instability patterns" that distinguish correct reasoning from plausible-sounding errors.12 Research indicates that incorrect solutions exhibit two typical failure modes: burst spikes and peak-valley (rebound) spikes.12 A burst spike occurs when entropy rises steadily over consecutive tokens as the model becomes progressively confused.12 A peak-valley spike involves a sharp drop to a local minimum (false confidence) followed by a violent rebound into uncertainty.8

Empirical analysis across multiple architectures and temperatures reveals that incorrect responses exhibit ![][image1] to ![][image2] more entropy fluctuations than correct ones.12 This suggests that the evolution of uncertainty is an intrinsic property of the reasoning process.12 EDIS provides a diagnostic signal that can be utilized at inference time to select between candidate solutions or to flag a response for manual review, achieving high separation between truth and fabrication.12

### **Stepwise Informativeness and SPREG**

The Stepwise Informativeness Assumption (SIA) provides a theoretical grounding for why these entropy trajectories are so informative.13 SIA posits that a successful reasoning chain accumulates information about the true answer as generation progresses, causing the conditional answer entropy to act as a progress variable.13 When the entropy fails to descend or exhibits sudden "spikes," it serves as a leading indicator that the model has entered a region of high epistemic uncertainty.8

Building on this, the Structured Plan-guided Real-time Entropy Gating (SPREG) framework implements an adaptive dual-threshold mechanism.8 SPREG monitors the real-time entropy ![][image3] and triggers an intervention only when a spike is detected—defined as ![][image3] exceeding both a moving average threshold and an absolute floor ![][image4] (often set to 2.0).8 This method is particularly effective for high-stakes competition mathematics, where a single logical error at a "decision fork" can lead to total system failure.8

## **Spectral Topology of Attention: Identifying Thermodynamic Shifts in Model Fabrication**

A novel frontier in hallucination detection involves the spectral analysis of internal attention maps, treating the LLM's transformer blocks as dynamic graphs.7 This approach moves beyond hidden state magnitudes to examine the structural and functional properties of the model’s internal communication flow.17

### **Laplacian Eigenvalues and Graph Spectra**

In this paradigm, the attention matrix is viewed as a weighted adjacency matrix of a directed graph.17 By computing the eigenvalues of the Laplacian matrix derived from these attention maps, researchers can detect disruptions in information flow.17 Correct reasoning is characterized by stable "eigen-structures," whereas hallucinations produce diffuse, chaotic patterns.7 The Laplacian Eigenvalues (LapEigvals) method uses the top-k eigenvalues as features for a logistic regression probe, achieving state-of-the-art performance among white-box detectors.17

### **The "Loud Liar" Phenomenon**

The application of spectral analysis has revealed what researchers call the "Loud Liar" phenomenon.9 For models like Llama 3.1-8B, hallucinatory states are "spectrally catastrophic," meaning the model's internal attention mechanisms effectively collapse into noise when it produces incorrect information.9 A single spectral feature, such as Llama L26 "Smoothness," can catch nearly 98.2% of hallucinations with a simple threshold.9 This suggests that hallucination is not merely the selection of an incorrect token but a "thermodynamic state change" where the model's attention energy disperses across modes rather than concentrating on coherent grounding.9

## **Semantic Consistency and Distributional Exploration: The Impact of Test-Time Compute on Uncertainty**

For black-box models where internal states are inaccessible, semantic consistency—measured through multiple sampling—remains a primary detection strategy.15 However, recent evaluations of frontier reasoning models like DeepSeek-R1 suggest that the reliability of these metrics is intrinsically tied to the "reasoning budget" or test-time compute allocated to the task.15

### **Semantic Entropy and its Evolution**

Semantic Entropy (SE) approximates the model's predictive distribution by Monte-Carlo sampling multiple answers and clustering them into semantically equivalent classes using an external Natural Language Inference (NLI) model.15 The Shannon entropy of this semantic distribution serves as the uncertainty score.22 While SE is well-calibrated, its performance on complex reasoning tasks like those in MATH-500 or AIME is significantly influenced by the length of the reasoning chain.15

Detailed analysis indicates that verbalized confidence (where the model explicitly states its certainty) is initially random (AUROC 0.56) but approaches the reliability of Semantic Entropy (AUROC 0.88) once the model is granted a reasoning budget of 3,500 tokens or more for mathematical items.15 This highlights that for reasoning models, uncertainty estimation is an active process of exploring the generative space rather than a simple readout of a latent variable.15

### **Reasoning-Explanation Symmetry (RES)**

To overcome the high inference cost of Semantic Entropy, which typically requires 10 or more samples, the Reasoning-Explanation Symmetry (RES) framework proposes a single-sample or triple-sample consistency check.26 RES is based on the hypothesis that a reliable answer should be "symmetric": the forward reasoning path (Question ![][image5] Answer) should semantically entail the backward explanation path (Answer ![][image5] Explanation).26 Using an NLI model to score this symmetry provides a continuous uncertainty metric that is particularly effective for multistep arithmetic and logic problems, often matching the AUROC of sampling-based methods with 70% fewer tokens.26

## **Comparative Benchmarking of Detection Efficacy across Mathematical and Symbolic Reasoning Tasks**

The following table presents a comprehensive survey of papers from 2023 to 2025 that evaluate hallucination detection specifically on reasoning benchmarks. The results are sorted by dataset and then by AUROC in descending order.

| Dataset | Method Name | Paper Title | Venue / ID | Signal Type | Evaluated Models | Metric (Score) | Superv. |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **AIME 2024** | Gnosis | Gnosis: Intrinsic Self-Verification | HF 2024 | Hidden States / Attention | LLaMA-2/3, Qwen | Baseline Accuracy Recovery | Superv. | 27 |
| **AIME 2024** | CAPO | Calibration-Aware Policy Optimization | arXiv 2025 | Advantage Estimation | Qwen2.5-Math-7B | AUROC (0.78) | Superv.\* | 28 |
| **AIME 2025** | SPREG | Real-time Entropy Gating | arXiv 2025 | Entropy Spikes | LLaMA-3-8B | Accuracy Gain (+20.0%) | Unsup. | 8 |
| **AIME 2025** | CAPO | Calibration-Aware Policy Optimization | arXiv 2025 | Advantage Estimation | Qwen2.5-Math-7B | AUROC (0.79) | Superv.\* | 28 |
| **GPQA-Diamond** | FINCH-ZK | Leveraging Prompting Variations | EMNLP 2025 | Cross-Model Consistency | Llama-4, Claude-4 | Accuracy Gain (+9.0%) | Unsup. | 30 |
| **GSM8K** | LapEigvals | Spectral Features of Attention Maps | EMNLP 2025 | Laplacian Eigenvalues | GPT-2, LLaMA | AUROC (0.826) | Superv. | 20 |
| **GSM8K** | VC (3.5k tokens) | Source of Uncertainty in DeepSeek | ACL 2025 | Verbalized Confidence | DeepSeek-R1-32B | AUROC (0.80) | Unsup. | 15 |
| **GSM8K** | HalluCodeDetector | Sampling Consensus Verification | ASE 2026 | Syntactic Consistency | (various) | AUROC (0.76) | Unsup. | 25 |
| **GSM8K** | Sem. Entropy | Phase Transitions in Hallucination | arXiv 2024 | Clustering (NLI) | Llama-3-8B | AUROC (0.70) | Unsup. | 21 |
| **MATH-500** ⭐ | **Gnosis** | **Gnosis: Intrinsic Self-Verification** | **HF 2024** | **Hidden States** | **LLaMA, Qwen** | **Acc. Gain (+12 pts)** | **Superv.** | 27 |
| **MATH-500** ⭐ | **EDIS** | **Entropy Dynamics Instability Score** | **arXiv 2025** | **Entropy Trajectories** | **Mistral, LLaMA-3** | **High Separation** | **Unsup.** | 12 |
| **MATH-500** ⭐ | **RHD** | **Mechanistic Perspective on LRMs** | **ICLR 2026** | **Logit Divergence** | **LRMs (unspecified)** | **CV Score 0.15 vs 0.24** | **Unsup.** | 1 |
| **MATH-500** ⭐ | **CAPO** | **Calibration-Aware Policy Optimization** | **arXiv 2025** | **Logistic AUC Surrogate** | **Qwen2.5-Math-7B** | **High Calibration** | \**Superv.* \*\* | 28 |
| **Multistep Arith** | RES (mean) | Reasoning-Explanation Symmetry | ICLR 2026 | NLI Symmetry | Qwen3-8B | AUROC (0.996) | Unsup. | 26 |
| **Multistep Arith** | RES (penalized) | Reasoning-Explanation Symmetry | ICLR 2026 | NLI Symmetry | Llama3-8B | AUROC (0.605) | Unsup. | 26 |
| **TriviaQA** | Sem. Entropy | Source of Uncertainty in DeepSeek | ACL 2025 | Semantic Entropy | DeepSeek-R1-32B | AUROC (0.88) | Unsup. | 15 |
| **TriviaQA** | RES (mean) | Reasoning-Explanation Symmetry | ICLR 2026 | NLI Symmetry | GPT-4o-mini | AUROC (0.800) | Unsup. | 26 |
| **StrategyQA** | RES (penalized) | Reasoning-Explanation Symmetry | ICLR 2026 | NLI Symmetry | GPT-4o-mini | AUROC (0.658) | Unsup. | 26 |

\* *Note: Methods like CAPO incorporate detection-related signals (AUC-consistent surrogates) during training but report inference-time AUC improvements as their primary result for calibration/hallucination mitigation.*

## **Supervised Internal State Probing and the Transition to Real-Time Guardrails**

While unsupervised methods offer broad generalization, supervised probing methods represent the current upper bound for detection accuracy on specific reasoning datasets.32 These techniques rely on the "Hidden State Awareness" of models—the fact that a model’s internal activations often encode whether a statement is true even if the model ultimately outputs a hallucination.27

### **Probing Late-Layer Activations**

Supervised probes typically involve training a linear classifier or a lightweight neural network (e.g., a two-layer MLP) on the hidden states of the transformer’s late layers.33 Methods like PALE and the Contrastive Mahalanobis Score (CM Score) model the distributions of truthful vs. hallucinated data in the activation space.36 By using matrix decomposition (e.g., SVD) to project these activations into a low-dimensional subspace, detectors can identify "loud liars" with high precision.9

Research into entity-focused probes has further refined this process.32 By weighting the KL divergence of token probabilities toward high-value tokens—such as numeric values, dates, and technical entities—probes can achieve a 3.0% to 9.0% absolute gain in AUROC over baseline internal probes.32 This suggests that the signal for hallucination is not uniformly distributed throughout a reasoning chain but is concentrated at critical "grounding tokens".32

### **The Integration of Context Adherence**

A critical third-order insight from recent literature is the role of context adherence as a secondary signal.6 Some models exhibit "parametric memory drift," where they ignore provided context in favor of pre-trained knowledge.7 Detectors that incorporate a "Context Adherence Signal"—measuring the inverse stress of the model weighted by context length—can better distinguish between a model that is reasoning from evidence and one that is hallucinating from its prior distribution.32

## **Synthesis of Emerging Trends and the Road to Reliable Reasoning Agents**

The survey of papers between 2023 and 2025 reveals several converging trends that define the current state of the art in reasoning hallucination detection.

1. **Inference-Time Scaling is Non-Negotiable**: For high-fidelity uncertainty estimation in mathematical and scientific tasks, models must be allowed to "reason about their reasoning".15 The dramatic rise in AUROC (from 0.56 to 0.88) when increasing reasoning tokens confirms that latent signals are insufficient for complex tasks without explicit exploration of the generative space.15  
2. **From Magnitudes to Topology**: The shift from measuring logit or embedding magnitudes to analyzing the topology of attention maps (via graph spectral analysis) represents a fundamental change in how we understand model internal states.9 Hallucination is increasingly viewed as a disruption in the coherence of the information flow across the transformer blocks.17  
3. **Internal Symmetry as a Lightweight Proxy**: While multi-sampling (Semantic Entropy) provides well-calibrated scores, the emergence of Reasoning-Explanation Symmetry (RES) offers a computationally efficient alternative for real-time monitoring.26 This reflects a broader trend toward building "self-aware" agents that can verify their own reasoning chains before final output.37  
4. **The Critical Importance of MATH-500 ⭐**: Across the literature, the MATH-500 benchmark has emerged as a rigorous differentiator for hallucination detection methods.1 Methods that perform well on simpler benchmarks like TriviaQA often fail to generalize to the symbolic complexity of MATH-500, making it the premier target for future research in symbolic reasoning reliability.1

In conclusion, the landscape of hallucination detection for reasoning models is rapidly maturing, moving away from simple post-hoc checks toward integrated, real-time diagnostic systems that leverage entropy trajectories, spectral topology, and semantic symmetry.9 As researchers continue to probe the mechanistic boundaries between pattern-matching and deep reasoning, these methods will become foundational for the safe deployment of LLM-based agents in expert-level domains.1

#### **Works cited**

1. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective \- OpenReview, accessed April 27, 2026, [https://openreview.net/pdf?id=PTbH6uKwhm](https://openreview.net/pdf?id=PTbH6uKwhm)  
2. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective \- arXiv, accessed April 27, 2026, [https://arxiv.org/pdf/2505.12886](https://arxiv.org/pdf/2505.12886)  
3. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2505.12886v1](https://arxiv.org/html/2505.12886v1)  
4. Contents \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2508.01781v1](https://arxiv.org/html/2508.01781v1)  
5. UC Santa Barbara \- eScholarship.org, accessed April 27, 2026, [https://escholarship.org/content/qt6nn947w5/qt6nn947w5.pdf](https://escholarship.org/content/qt6nn947w5/qt6nn947w5.pdf)  
6. FinReflectKG \- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2603.20252v1](https://arxiv.org/html/2603.20252v1)  
7. EdinburghNLP/awesome-hallucination-detection \- GitHub, accessed April 27, 2026, [https://github.com/EdinburghNLP/awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection)  
8. SPREG: Structured Plan Repair with Entropy-Guided Test-Time Intervention for Large Language Model Reasoning \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2604.17884v1](https://arxiv.org/html/2604.17884v1)  
9. Spectral Guardrails for Agents in the Wild: Detecting Tool Use Hallucinations via Attention Topology \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2602.08082v1](https://arxiv.org/html/2602.08082v1)  
10. The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2510.22977v2](https://arxiv.org/html/2510.22977v2)  
11. The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2510.22977v1](https://arxiv.org/html/2510.22977v1)  
12. EDIS: Diagnosing LLM Reasoning via Entropy Dynamics \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2602.01288v1](https://arxiv.org/html/2602.01288v1)  
13. The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs? \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2604.06192v1](https://arxiv.org/html/2604.06192v1)  
14. The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs? \- arXiv, accessed April 27, 2026, [https://arxiv.org/pdf/2604.06192](https://arxiv.org/pdf/2604.06192)  
15. Read Your Own Mind: Reasoning Helps Surface Self-Confidence Signals in LLMs \- ACL Anthology, accessed April 27, 2026, [https://aclanthology.org/2025.uncertainlp-main.21.pdf](https://aclanthology.org/2025.uncertainlp-main.21.pdf)  
16. From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2604.09459v1](https://arxiv.org/html/2604.09459v1)  
17. Hallucination Detection in LLMs Using Spectral Features of Attention Maps \- ACL Anthology, accessed April 27, 2026, [https://aclanthology.org/2025.emnlp-main.1239.pdf](https://aclanthology.org/2025.emnlp-main.1239.pdf)  
18. Spectral Analysis of Attention Patterns \- Emergent Mind, accessed April 27, 2026, [https://www.emergentmind.com/topics/spectral-analysis-of-attention-patterns](https://www.emergentmind.com/topics/spectral-analysis-of-attention-patterns)  
19. Attention Sinks as Internal Signals for Hallucination ... \- Bytez, accessed April 27, 2026, [https://bytez.com/docs/arxiv/2604.10697/paper](https://bytez.com/docs/arxiv/2604.10697/paper)  
20. Hallucination Detection in LLMs Using Spectral Features of Attention, accessed April 27, 2026, [https://www.wizwand.com/paper/68fa7c3e1e209131cea8d1e9](https://www.wizwand.com/paper/68fa7c3e1e209131cea8d1e9)  
21. (PDF) Hallucination as an Entropy-Driven Phase Transition \- ResearchGate, accessed April 27, 2026, [https://www.researchgate.net/publication/399475999\_Hallucination\_as\_an\_Entropy-Driven\_Phase\_Transition](https://www.researchgate.net/publication/399475999_Hallucination_as_an_Entropy-Driven_Phase_Transition)  
22. Read Your Own Mind: Reasoning Helps Surface Self-Confidence Signals in LLMs \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2505.23845v2](https://arxiv.org/html/2505.23845v2)  
23. Self-Consistency Hallucination Detection \- Emergent Mind, accessed April 27, 2026, [https://www.emergentmind.com/topics/self-consistency-based-hallucination-detection](https://www.emergentmind.com/topics/self-consistency-based-hallucination-detection)  
24. Reasoning & Semantic Scores in AI \- Emergent Mind, accessed April 27, 2026, [https://www.emergentmind.com/topics/reasoning-and-semantic-scores](https://www.emergentmind.com/topics/reasoning-and-semantic-scores)  
25. Hallucination detection in LLM code generation: A sampling-based consensus verification approach \- ResearchGate, accessed April 27, 2026, [https://www.researchgate.net/publication/403124782\_Hallucination\_detection\_in\_LLM\_code\_generation\_A\_sampling-based\_consensus\_verification\_approach](https://www.researchgate.net/publication/403124782_Hallucination_detection_in_LLM_code_generation_A_sampling-based_consensus_verification_approach)  
26. UNCERTAINTY QUANTIFICATION VIA REASON ... \- OpenReview, accessed April 27, 2026, [https://openreview.net/pdf/483ba313c6d8182349983051b6fbe4b6de01a966.pdf](https://openreview.net/pdf/483ba313c6d8182349983051b6fbe4b6de01a966.pdf)  
27. Daily Papers \- Hugging Face, accessed April 27, 2026, [https://huggingface.co/papers?q=hidden-state%20awareness](https://huggingface.co/papers?q=hidden-state+awareness)  
28. Calibration-Aware Policy Optimization for Reasoning LLMs \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2604.12632v1](https://arxiv.org/html/2604.12632v1)  
29. \[Literature Review\] Calibration-Aware Policy Optimization for Reasoning LLMs, accessed April 27, 2026, [https://www.themoonlight.io/review/calibration-aware-policy-optimization-for-reasoning-llms](https://www.themoonlight.io/review/calibration-aware-policy-optimization-for-reasoning-llms)  
30. Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency \- ACL Anthology, accessed April 27, 2026, [https://aclanthology.org/2025.emnlp-industry.139.pdf](https://aclanthology.org/2025.emnlp-industry.139.pdf)  
31. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation \- ResearchGate, accessed April 27, 2026, [https://www.researchgate.net/publication/401456333\_Is\_Your\_Code\_Generated\_by\_ChatGPT\_Really\_Correct\_Rigorous\_Evaluation\_of\_Large\_Language\_Models\_for\_Code\_Generation](https://www.researchgate.net/publication/401456333_Is_Your_Code_Generated_by_ChatGPT_Really_Correct_Rigorous_Evaluation_of_Large_Language_Models_for_Code_Generation)  
32. Case Study: Predictive Coding and Information Bottleneck for Hallucination Detection in Large Language Models \- arXiv, accessed April 27, 2026, [https://arxiv.org/html/2601.15652v1](https://arxiv.org/html/2601.15652v1)  
33. AI Hallucinations Are Getting Smarter — Here's How to Catch Them in Real-Time (Even in Agentic AI Systems, 2026\) | by Yash Mishra | Medium, accessed April 27, 2026, [https://medium.com/@yash.mishra0501/ai-hallucinations-are-getting-smarter-heres-how-to-catch-them-in-real-time-even-in-agentic-3d75a9fc1ab3](https://medium.com/@yash.mishra0501/ai-hallucinations-are-getting-smarter-heres-how-to-catch-them-in-real-time-even-in-agentic-3d75a9fc1ab3)  
34. Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models | Request PDF \- ResearchGate, accessed April 27, 2026, [https://www.researchgate.net/publication/384209997\_Unsupervised\_Real-Time\_Hallucination\_Detection\_based\_on\_the\_Internal\_States\_of\_Large\_Language\_Models](https://www.researchgate.net/publication/384209997_Unsupervised_Real-Time_Hallucination_Detection_based_on_the_Internal_States_of_Large_Language_Models)  
35. QUERY-LEVEL UNCERTAINTY IN LARGE LANGUAGE MODELS \- Fabian Suchanek, accessed April 27, 2026, [https://suchanek.name/work/publications/iclr-2026.pdf](https://suchanek.name/work/publications/iclr-2026.pdf)  
36. HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection, accessed April 27, 2026, [https://www.researchgate.net/publication/397203607\_HaloScope\_Harnessing\_Unlabeled\_LLM\_Generations\_for\_Hallucination\_Detection](https://www.researchgate.net/publication/397203607_HaloScope_Harnessing_Unlabeled_LLM_Generations_for_Hallucination_Detection)  
37. Daily Papers \- Hugging Face, accessed April 27, 2026, [https://huggingface.co/papers?q=self-verification%20procedure](https://huggingface.co/papers?q=self-verification+procedure)  
38. Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning \- AAAI Publications, accessed April 27, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/41061/45022](https://ojs.aaai.org/index.php/AAAI/article/view/41061/45022)  
39. LLM Benchmarks Compared: MMLU, HumanEval, GSM8K and More (2026), accessed April 27, 2026, [https://www.lxt.ai/blog/llm-benchmarks/](https://www.lxt.ai/blog/llm-benchmarks/)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABhElEQVR4Xu2VTytEURjGX8lCyiRKotRgYYWUoig1GwsWFjY+gZWiyEKNL2BhJfkCUjbKSlmwsvYFLGRlo4iFP8/Tca7jnXPunXuHjc6vfjXzvveeeTp3zntFIpFUmmAHbNaNv6QPzutigC54Ay/hQcC55OoGGIVH8A6+Sf0By/AefgTkWgvJ1b/EptQfcBGu6SKYgGew06m1wzbnuw8+kUzyBByGvapWgidiQrqMwQvYr+qWFriuiz7yBNTw0OxK+Iem4JXUhmS4fTH3Z9JIwGl4Dbt1w0GHtOE2kisyKBqwFZ7CPd3wYEMOiQm3JTlGVdGAk/AVLutGgBn4DKuSIxwpGrAK32FF1X1wtzkjj8Xs5ODPdjpFAnJ8nMMnOK56GhuO/zkeCp5238EJkhZwAK5K7bzqgbeSHVCHs9iQmfCmbbikG/K9S3xDHKreCHyU9IAMxwPBEeQbJwwZ3EUuysX1q4q7wt0hXHQHvoh5g7hwYD98yc8+ZuGK+MNZ+PQikci/4hNC+Uji5NA/4QAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAACIUlEQVR4Xu2VMUiVURiGX0mh0FySqEiM1ECKRASjhEBwCCKI1saGSISgoHCzIXBwEgeRFlvCalSQELxoOLi0VIi1OEhDYOgQ4qC+r9859/7/4fzdKzoY/C88cO93zj3nved833eAXLmOjy6SSbJGPpPbpCo1I1uad5d8JevkF3ni4keiy2SRdLrv42SXPC/OyFYNGSY/yXUXuwMzedNPOqwewQxNk5Okl+yQVXI+MS9UI/mB0u+8uskrci4Rqye1ie8xNYQBrxPkLEqb9MMMvyPVflJED2HzXocDEXWQOdIUDjjpJiq5sX2tOLIW83oDMzhEJsgU7HfzsJwOdQuW3+G6MjeGCnL2PlkmH2AnWk66WhlUYdxwMW32HlYwsTVCk97ci+KMMtI16wSWyJVgLFQBZvAj0qnwwMUHE7GkvMlWmLkBWJpVrGvkD8pfsz/Bl0H8novPIrsw1Mb+wv7EgcxJqqZvsE2eBWNJKe/+ZbBA6tJD+zoFa2VKJZ1kS3o4LS3wiWzDFpZOkwXYJiMuFtNjHNygN6ecU1F0IV44RbWTTaRPS71PPVAxmfBqJk9R6lc+FcI24w2Gfy405+VNRqUfzcB61AUX00ugRp2sROWSckobq71IKgx9VkGdcTFtPEp+k6suJmkfFYR6XaydyGTmKbaR77Br7SMbsKfvUmKOFtXrsAWrUi+9FjL+BXbabxF/5npQ/n0OUyVXrlz/vfYAwz1uYsezpX8AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAZCAYAAAA14t7uAAABTUlEQVR4Xu2UPUsDQRCGX1FBMI0oIiSVRUwpWAkWFjY2FmInpE0nIU3SCYH8gVQiopWdYGVnYS+IhfZ2kh9gIeLH+2ZuJY6rd3qQKg88zc7t7O3MsMCIYVOg+/TQuUkn6F4kJlMZp/N0lT7RKi3RKTpGZ2mF3tE3Wk/imWnSjl9M2KLv9AJ2YGZ05TNYghhdWGId/ieK9IEuunWhHlzRF7r2NZTOBqx+J/jepFP6TO/pXNiQFdVWV92JeJDEdICamRk1Q03RH8c4giWu+UAaqusjrMaeGXoNG8MVF9PIHcPGMUoYJU2FR8mUVMl1yCDb9JJOu/VPwig1fAB2fcXUVF9f7ftp7vvXuEF8lCbpOSzxrost0x4srudAI9lnnb7CNg1apkuwTT6m77UvcIv43Ofm1/rmIdRXyX39/43eljZt0QUXy40aryd3xBD5AOFtSLnmDJpCAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAZCAYAAABHLbxYAAACKklEQVR4Xu2WO2sVURSFlxhFfKIRGwOiRTQgNrYWFmlsrOwEAyaooISQxtKAEVFsDCmCCKbxBYKVhQghqKRIRC30N0gIlilC0Lg+9xwmOY7C9cY7CHfBx9yZfeay99mPM1Jbbf1f2m6umXsZp0yHGayw9f98s8XaaPaZr2bRnDNdZovZYDrNEfPJfDdDiuBq04q5kT8sdFphf6EIoDaRYhzBoSqNKexXc0Ortd98MYdygyLN02bZnFhrar16zYJ5oF8b56FZMp/N3vRCXaI2J82ZCiYUacdhmqs20Rw0CbtapfsKRy/mhgZFCTXViNQl9Umd5tpt5hRj63hma0Q7zVsznBsaURo9dH4unMNJnMXpZrRDMbP/Wmn0VIl0Y6PJVtfnJrOneJYOjJRWbNxzTdqV3fMOBwnXdKj8USx4rxg9ufjj5wpHz2a2UfPE3DV3zIB5pwjslrls3iicOF+sf6lwmMwxjyfNU3Pb9Ok3M/yk+aZwIsF46jaHzXxmA9bzHuI7gEyw0wSUGvK6wjnS/MocNBcUjTqjKJ8exbcCQb1WOI/+yWFCJqZUTgoa8YPKhsOZjyqDGDcjhY0d5TnjLk0SJkLljjar5Eg6yTixZlUeCFfMM7NZEQS2Y2ZbYWcdz9JJd1TxX8m+biLtI8VvGoJaS3XMDj5S1Cq7x05RBpTFgWINO/m4sKNL5qbKMlg3EfnqLt6qtaOH39RpUt71BJMP/6ZGV1tt1aEf3WZkkdI+M6sAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAeElEQVR4XmNgGAWjgGpADYiL0QXJBSJAvBCIBdElyAXVQOyBLkgukAfijVCaKsAMiHcAsQKaOIMAEEuSiFWBeCsQTwZiPgYKACsDxJAEIGZElSId+AFxJwMVDOIE4gUMEG9SDDSBeDoQs6BLkANA4cWNLjgKhhsAAL6gCkpRliz2AAAAAElFTkSuQmCC>