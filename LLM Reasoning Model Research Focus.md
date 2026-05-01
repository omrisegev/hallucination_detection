# **The Statistical Mechanics of Neural Reasoning: Anomaly Detection, Logit Manipulation, and the Probabilistic Steering of Reasoning-Based Architectures**

The architectural shift in large-scale language modeling has transitioned from the production of fluent surface-level text to the simulation of structured, multi-step deliberation. This evolution is fundamentally a transition from "System 1" heuristic associations to "System 2" algorithmic reasoning. Central to this transformation is the mathematical characterization of the model's output space, specifically the distribution of logits and the resulting token probabilities. For applied mathematicians and computer scientists, the ability to calculate, interpret, and manipulate these distributions has become the primary mechanism for ensuring reliability, detecting anomalies, and steering the generation process in high-stakes environments. The study of these processes involves a rigorous application of information theory, conformal prediction, and entropy-based optimization, creating a new frontier in the statistical nature of machine intelligence.

## **The Micro-Statistics of Autoregressive Generation**

The generation process of a modern transformer-based model is an iterative sampling from a high-dimensional probability distribution. At each temporal step, the model projects a latent representation onto a vocabulary space, producing a vector of logits ![][image1]. The conversion of these logits into a probability distribution ![][image2] via the softmax function ![][image3] serves as the fundamental interface between the model’s internal states and its external outputs.1 While this process appears deterministic under a temperature setting of ![][image4], empirical evidence suggests that hardware-level parallelization introduces subtle nondeterministic fluctuations in the logit values. These variations are most pronounced for tokens with probabilities in the mid-range (0.1 to 0.9), where floating-point summation order and atomic operations in Graphic Processing Units (GPUs) can induce shifts that occasionally change the rank-order of candidate tokens.2

This inherent nondeterminism complicates the task of reproducible reasoning. For mathematicians specializing in anomaly detection, these fluctuations represent a form of "noise" that must be distinguished from meaningful signal drift. The quantification of this noise—using metrics such as the standard deviation and range of token probability variations across repeated runs—provides a baseline for identifying anomalies.2 When a model's output deviates significantly from this expected statistical variance, it often indicates an underlying failure in the reasoning chain or a drift into out-of-distribution semantic space.

| Model Variant | Avg. Logit Variation (T=0) | Critical Prob. Range | GPU Sensitivity (A100) | Reproducibility Score |
| :---- | :---- | :---- | :---- | :---- |
| Llama 3.1-8B | ![][image5] | 0.2 \- 0.7 | High | 94.2% |
| Qwen 2.5-7B | ![][image6] | 0.1 \- 0.8 | Medium | 95.8% |
| GPT-4o | ![][image7] | 0.15 \- 0.85 | Low | 97.1% |
| Phi-3-Mini | ![][image8] | 0.1 \- 0.9 | High | 92.5% |

The table above illustrates the relationship between model architecture, scale, and logit stability. Smaller models often exhibit higher sensitivity to hardware-level parallelization, which can propagate through a Chain-of-Thought (CoT) trace, potentially leading to divergent reasoning paths.2

## **Entropy Dynamics and Uncertainty Quantification**

Entropy, in its various mathematical forms, has emerged as the most potent diagnostic signal for monitoring the integrity of LLM generation. Token-level Shannon entropy, defined as ![][image9], quantifies the predictive uncertainty at a specific step.4 However, recent research has expanded this to include semantic and structural entropy, which capture dependencies beyond the immediate token.4 Semantic entropy is particularly vital for detecting "hidden" uncertainty; it is calculated by clustering multiple generated answers into equivalence classes based on mutual entailment, measuring uncertainty over the *meaning* of the response rather than its surface form.4

For reasoning models, entropy dynamics serve as a window into the model's internal confidence. A sudden spike in entropy often marks a "transition point" where the model departs from verifiable factual grounding and begins to confabulate.7 Conversely, an "entropy collapse" in deep layers—often caused by the removal of nonlinearities like LayerNorm—can lead to sharply peaked distributions that undermine diversity and destabilize the reasoning process.4

### **Multi-Faceted Entropy Metrics in Reasoning**

The application of entropy to Chain-of-Thought traces requires a hierarchical approach. Structural entropy is used to aggregate uncertainty across node-link diagrams or reasoning graphs, where each fact ![][image10] is assigned an information value based on its conditional probability within the context of the preceding chain.3 This allows for the identification of specific "weak links" in a derivation.

| Entropy Type | Mathematical Basis | Diagnostic Utility | Primary Application |
| :---- | :---- | :---- | :---- |
| **Token-level Shannon** | **![][image11]** | Immediate step uncertainty | Hallucination detection 4 |
| **Semantic Entropy** | Clustering via entailment | Uncertainty over meaning | Open-ended QA 5 |
| **Structural Entropy** | Graph node uncertainty | Reasoning chain integrity | Logic verification 3 |
| **Kernel Language (KLE)** | von Neumann entropy | Semantic dependence | Distributed representations 4 |

The implications of these metrics are profound for anomaly detection. In clinical or financial workflows, systems can set entropy thresholds to trigger additional verification or defer to a human expert when the "inner doubts" of the model exceed a specified limit.1

## **The Unreasonable Effectiveness of Entropy Minimization**

One of the most significant recent discoveries in the manipulation of reasoning models is the role of Entropy Minimization (EM).8 Research by Hao Peng and Jiawei Han indicates that many pretrained LLMs possess "underappreciated reasoning capabilities" that can be elicited simply by concentrating probability mass on the most confident outputs.8 This technique, which requires no labeled data, substantially improves performance on complex math, physics, and coding tasks.8

The methodology of EM manifests in three frameworks:

1. **EM-FT (Fine-tuning)**: Minimizing token-level entropy on unlabeled outputs, effectively forcing the model to commit more strongly to its own predictions during post-training.9  
2. **EM-RL (Reinforcement Learning)**: Using negative entropy as the sole reward function. This objective has been shown to produce models (like Qwen-2.5-EM-RL) that perform as well as those trained on tens of thousands of labeled examples with traditional RL algorithms like GRPO or RLOO.8  
3. **EM-INF (Inference-time)**: Adjusting logits during the decoding process itself to minimize entropy. This process treats logits as parameters and uses gradient descent to sharpen the distribution before sampling, making the output more deterministic and concise.9

| Framework | Benchmark (Math) | Gain over Base | Efficiency vs. Self-Consistency |
| :---- | :---- | :---- | :---- |
| **Qwen-7B (Base)** | 43.8% | \- | \- |
| **EM-FT (N=1)** | 67.2% | \+23.4% | 1.0x |
| **EM-RL (N=4)** | 70.8% | \+27.0% | 13.1x (FLOPs) |
| **EM-INF (N=1)** | 68.5% | \+24.7% | 0.33x (3x faster) |

The success of EM-INF on the SciCode benchmark—matching or exceeding proprietary models like GPT-4o—suggests that the limitation of many current models is not a lack of knowledge, but a failure to efficiently navigate their own internal probability space.9 By steering the model toward its most confident modes, EM-INF effectively acts as a "thinking" mechanism that bypasses the need for extensive textual rollouts in certain domains.

## **Inference-Time Scaling and the o1 Paradigm**

The emergence of "o1-style" models has codified the "inference-time scaling law," which posits that performance improves monotonically as the model is given more compute budget during the generation phase.10 This budget is typically expended through "thinking tokens" that form a hidden or visible Chain-of-Thought.12 The statistical nature of this scaling is driven by two main mechanisms: search and learning.

### **Search-Based Scaling**

Models like o1 use a combination of tree search and backtracking to explore the solution space. Unlike standard autoregressive models that commit to each token permanently, reasoning models can "look ahead" and revise their internal trajectory if the probability of reaching a successful outcome (as predicted by a Process Reward Model) begins to drop.12 This mirrors the Monte Carlo Tree Search (MCTS) strategies used in game-playing AIs like AlphaGo.11

In this context, the model’s generation process is no longer a linear sequence but a branching tree of probabilities. The "deliberate reasoning" of the model is mathematically equivalent to increasing the depth and breadth of the search over the token distribution.11 This has led to the discovery that for certain tasks, particularly in STEM, the ceiling for AI performance is significantly higher than previously thought, as the model can overcome initial "System 1" errors through "System 2" deliberation.12

### **Process Reward Models (PRMs) and Verification**

The key to effective inference scaling is the ability to judge intermediate steps. Process Reward Models (PRMs) are trained to assign a correctness score to each reasoning step in a CoT trace.11 This allows the decoding algorithm to prune low-probability branches of the reasoning tree. For applied mathematicians, the challenge lies in training these PRMs without expensive human step-by-step annotation. Techniques such as "Math-Shepherd" or "K-STaR" automatically generate step-level supervision by using final answer correctness and frequency analysis to backtrack and reward the preceding reasoning steps.14

| Scaling Method | Computational Target | Core Statistical Signal | Emergent Behavior |
| :---- | :---- | :---- | :---- |
| **Chain-of-Thought** | Token Count | Sequential coherence | Step-by-step logic 11 |
| **Majority Voting** | Sample Count (N) | Answer frequency | Consensus accuracy 16 |
| **Tree Search** | Search Depth | PRM/Logit scores | Backtracking/Correction 12 |
| **Best-of-N** | Parallel Samples | Reranker/Verifier output | Path optimization 14 |

The integration of these methods creates a "reinforced cycle" where automated search identifies high-quality reasoning trajectories, which are then fed back into the model to improve its baseline reasoning capability.11

## **Conformal Prediction: Rigorous Guarantees for Open-Ended Generation**

In high-stakes applications, heuristic confidence scores are insufficient. Conformal Prediction (CP) has emerged as a critical mathematical framework for providing distribution-free, finite-sample guarantees on the correctness of LLM outputs.17 Pioneered by researchers like Anastasios Angelopoulos and Stephen Bates, CP constructs prediction sets ![][image12] that are guaranteed to contain the ground-truth answer ![][image13] with a user-specified probability ![][image14].18

The application of CP to LLMs is challenging because the output space (all possible text sequences) is infinite. To address this, "Conformal Factuality" uses a back-off algorithm.19 If the model's most specific response (e.g., "The revenue was $1,371.50") falls below the confidence threshold calibrated on a hold-out set, the algorithm reverts to a less specific but more likely correct statement (e.g., "The revenue was between $1,300 and $1,400").19 This "entailment-based" coverage ensures that the model remains "honestly uncertain" rather than overconfidently wrong.

### **Conformal Reasoning in Multi-Turn Settings**

A significant limitation of standard split conformal prediction is its inability to handle the "heuristic bias" of interactive trajectories. In multi-turn settings, the model's own uncertainty should guide its future decisions (e.g., asking a clarifying question). "Conformal Reasoning" addresses this by adapting ACI (Adaptive Conformal Inference) methods to the multi-turn setup, allowing the model to leverage its prediction sets to decide whether to seek more information or return a prediction.20 This framework has been shown to achieve theoretical coverage guarantees while improving exploration efficiency in tasks like medical diagnosis and twenty questions.20

| Uncertainty Strategy | Mathematical Tool | Statistical Guarantee | Trade-off |
| :---- | :---- | :---- | :---- |
| **Selective Classification** | Thresholding $P(y | x)$ | None (Heuristic) |
| **Conformal Prediction** | Calibration Sets | Distribution-free coverage | Larger prediction sets 18 |
| **Conformal Arbitrage** | Primary/Guardian Split | Risk budget adherence | Latency (2x model calls) 23 |
| **PAC Reasoning** | Upper Confidence Bounds | Performance loss control | Conservative compute usage 24 |

The work of Shubhendu Trivedi and colleagues further extends this by developing "Contextualized Sequence Likelihood," which enhances confidence scores for natural language generation by considering the full context of the derivation.25 These tools allow practitioners to transform black-box models into controllable systems with provable guardrails.23

## **Anomaly Detection Mechanisms: Internal and External**

Anomaly detection in the context of LLMs is a dual-natured problem. First, models are used as powerful feature transformers to detect anomalies in complex datasets. Second, the model's own generation must be monitored for internal anomalies (hallucinations).

### **Industrial and Log Anomaly Detection**

LLMs excel at identifying patterns in unstructured or semi-structured data like system logs or time-series. The Reasoning based Anomaly Detection Framework (RADF) utilizes a technique called mSelect to automate algorithm selection and hyper-parameter tuning for real-time detection on massive datasets.26 In industrial quality control, the AnomalyCoT dataset provides a multimodal framework where models generate reasoning traces to localize defects in images, achieving up to 94% accuracy after fine-tuning.27

A persistent challenge in this domain is the "normality bias." Because LLMs are trained on vast corpora of typical data, they often exhibit a bias toward interpreting anomalous events as normal.28 To counter this, frameworks like Chain-of-Anomaly Thought (CoAT) introduce "criminal-biased" or "defect-focused" layers in the reasoning process to force the model to explore "anomalous" hypotheses explicitly.28

### **Internal Anomaly Detection: Token-Guard and Probes**

Detecting hallucinations *during* the generation process requires monitoring the model's hidden states and logit distributions. "Token-Guard" is a modular solution that performs self-checking decoding.29 It calculates a hybrid score for each candidate token by balancing semantic consistency (using the mean hidden state of prior tokens) and the model's assigned token probability.29 If a token's score fails to meet a specific threshold, it is pruned, and the fragment is regenerated.29

| Detection Method | Signal Used | Latency Impact | Target Anomaly |
| :---- | :---- | :---- | :---- |
| **Linear Probing** | Intermediate activations | Low (0.01s) | Entity-level hallucination 5 |
| **Semantic Entropy** | Clustering (N samples) | High (N x Gen) | Claim-level falsehood 4 |
| **Token-Guard** | Hidden states \+ Logits | Moderate | Step-wise inconsistency 29 |
| **Logit Lens** | Projected activations | Very Low | Early prediction errors 30 |

Research using "hallucination probes" has shown that lightweight linear classifiers trained on intermediate layers can predict output correctness with 75-80% accuracy from the first output token alone.5 This enables "early auditing" of reasoning paths before the model spends excessive compute on a flawed derivation.

## **Adaptive Steering and Stopping Heuristics**

Manipulating the reasoning process is not merely about increasing compute, but about *optimizing* its allocation. This involves steering the logit distribution toward higher-quality states or terminating the reasoning chain once a solution has been reached.

### **Logit Steering and Feynman-Kac Potentials**

Inference-time steering often involves "tilting" the model's distribution toward high-reward samples. Feynman-Kac (FK) steering uses a particle-based sampler to explore the distribution of plausible solutions, scoring "particles" (parallel generation paths) based on "potential functions" derived from intermediate rewards.31 This allows the system to guide the model using arbitrary reward functions—differentiable or otherwise—to find "rare events" that represent high-quality solutions.31

### **LEASH and the Economics of Thinking**

While "thinking longer" improves accuracy, it is computationally expensive. The Logit-Entropy Adaptive Stopping Heuristic (LEASH) provides a training-free way to halt rationale generation.32 By monitoring the slope of token-level entropy and the improvement in the top-logit margin, LEASH can detect when a reasoning state has stabilized. Across benchmarks like GSM8K, this method has been shown to reduce token generation by up to 35% with minimal accuracy loss, effectively finding the point of diminishing returns in the inference-scaling curve.32

### **Soft Thinking and Continuous Concept Spaces**

Traditional Chain-of-Thought is restricted by the discrete nature of language tokens. "Soft Thinking" addresses this bottleneck by generating "soft concept tokens" in a continuous latent space—created by a probability-weighted mixture of token embeddings.33 This allows the model to explore multiple related meanings simultaneously, implicitly tracking various reasoning paths in parallel before converging on a discrete answer.14 This method has been shown to improve pass@1 accuracy while reducing token usage, as a single "soft" token can encapsulate the meaning of several discrete tokens.33

## **Leading Applied Math Professors and Research Labs**

The research landscape for the statistical steering of LLMs is centered in several key academic and industrial clusters. These researchers bridge the gap between abstract probability theory and the practical engineering of large-scale models.

### **University of California, Berkeley**

* **Anastasios Angelopoulos**: A pioneer in conformal prediction and risk control. His work on "Conformal Arbitrage" 23 and distribution-free guarantees 18 provides the mathematical framework for building safe, multi-model reasoning systems.  
* **Dan Klein**: Focusing on the interpretability of model internals. His lab has produced foundational work on "Logit Lens" and "Tuned Lens" techniques for monitoring the evolution of model predictions across layers.30  
* **Gireeja Ranade**: Collaborates on benchmarking advanced mathematical reasoning and understanding the "reasoning illusion," where models may succeed through pattern matching rather than genuine insight.34

### **Massachusetts Institute of Technology (MIT)**

* **Stephen Bates**: A central figure in the MIT Probabilistic Computing Project. His research on "Conformal Reasoning" 20 and interactive uncertainty quantification 35 focuses on ensuring that LLM agents are transparent and trustworthy in human-computer interactions.  
* **Shubhendu Trivedi**: Specializes in applied machine learning and computational mathematics. His work on conformal prediction for general deep neural networks and "Generating with Confidence" 25 is critical for the reliable deployment of LLMs in healthcare and science automation.

### **University of Illinois Urbana-Champaign (UIUC)**

* **Hao Peng**: An expert in entropy management and reasoning dynamics. His discovery of the "Unreasonable Effectiveness of Entropy Minimization" 8 has redefined how researchers view the latent reasoning capabilities of pretrained models. His work on process rewards 37 and reasoning tree structures 38 focuses on data-efficient alignment.  
* **Jiawei Han**: A world leader in data mining and knowledge discovery. His lab explores "Uncertainty Aware Knowledge-Graph Reasoning" (UAG) 39 and the integration of LLMs with structured knowledge for grounded anomaly detection.40

### **Industrial Leaders and Pioneers**

* **Noam Brown (OpenAI)**: A legend in deep reinforcement learning (pioneer of Libratus and Pluribus for poker). His current work at OpenAI on Project Strawberry (o1) focuses on the "System 2" scaling of inference compute through search and self-correction.12  
* **Eric Zelikman (Stanford/OpenAI)**: Developer of the STaR (Self-Taught Reasoner) and Quiet-STaR paradigms.15 His research focuses on how models can teach themselves to reason from unstructured text by generating unstated rationales.

## **Statistical Divergence and the "Reasoning Illusion"**

As models scale in inference compute, a critical question arises: is the model performing genuine mathematical reasoning, or is it exhibiting a "reasoning illusion" driven by sophisticated pattern matching?34 Statistical benchmarks such as StatEval and Stat-Reasoning-Bench have been developed to test this.42 Unlike elementary arithmetic, advanced statistics requires rigorous proof generation and conceptual differentiation (e.g., Type I vs. Type II error).42

Experiments show that fine-tuned 7B models can reach the level of a statistics student in basic tasks but struggle with novel or reworded problems, suggesting a high degree of "template overfitting".42 The use of entropy metrics allows researchers to quantify this gap; models often exhibit higher entropy and lower consistency on problems that require genuine "System 2" insight compared to those that can be solved via retrieval.43

| Reasoning Domain | Benchmark Accuracy | Primary Failure Mode | Statistical Indicator |
| :---- | :---- | :---- | :---- |
| **Elementary Math** | 90% \+ | Calculation error | Low entropy, wrong answer 44 |
| **Olympiad Math** | 15% \- 50% | Logic breakdown | High entropy at step ![][image15] 11 |
| **Medical Statistics** | 51% | Conceptual confusion | Factual logit drift 42 |
| **Industrial AD** | 59% | Normality bias | Compressed logit range 28 |

The detection of these failure modes is increasingly performed by "LLM-as-a-judge" systems. However, to ensure these judges are themselves reliable, applied mathematicians are now applying conformal prediction to the *evaluator's* output, creating a recursive chain of statistical guarantees.45

## **The Future of Statistical Steering and AGI**

The path toward Artificial General Intelligence (AGI) is increasingly viewed through the lens of Richard Sutton's "Bitter Lesson": that general methods that leverage compute (search and learning) are ultimately more effective than those that leverage human knowledge. The "Reasoning Model" represents the first large-scale instantiation of this in the language domain.

The future of the field will likely center on the following developments:

1. **Continuous Reasoning Spaces**: Moving beyond discrete tokens to "soft" thinking paths that allow for richer, differentiable optimization of reasoning trajectories.14  
2. **Autonomous Anomaly Correction**: Systems like Token-Guard that detect and fix internal reasoning anomalies in real-time, enabling "streaming" reliability monitoring.5  
3. **Provable Safety Guardrails**: The widespread adoption of conformal risk control to bound the frequency of harmful or incorrect model behaviors with statistical rigor.23  
4. **Inference Scaling Efficiency**: Algorithms like LEASH that dynamically adjust the "thinking time" to the difficulty of the problem, maximizing the utility of every flop.32

In summary, the ability to calculate and manipulate the statistical nature of LLM generation is the primary lever for advancing artificial reasoning. By treating the model's output as a dynamic probability landscape that can be tilted, pruned, and calibrated, applied mathematicians are building the essential infrastructure for trustworthy, reasoning-capable AI. The ongoing research at labs in Berkeley, MIT, and UIUC remains the vanguard of this effort, ensuring that as models "think" longer, they also think more reliably.

#### **Works cited**

1. Token Probabilities to Mitigate Large Language Models Overconfidence in Answering Medical Questions: Quantitative Study \- PMC, accessed April 19, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12396779/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12396779/)  
2. Beyond Reproducibility: Token Probabilities Expose Large Language Model Nondeterminism \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2601.06118v1](https://arxiv.org/html/2601.06118v1)  
3. When the Chain Breaks: Interactive Diagnosis of LLM Chain-of-Thought Reasoning Errors, accessed April 19, 2026, [https://arxiv.org/html/2603.21286v2](https://arxiv.org/html/2603.21286v2)  
4. Entropy Dynamics in LLMs: Metrics & Implications \- Emergent Mind, accessed April 19, 2026, [https://www.emergentmind.com/topics/entropy-dynamics-in-llms](https://www.emergentmind.com/topics/entropy-dynamics-in-llms)  
5. Real-Time Detection of Hallucinated Entities in Long-Form Generation, accessed April 19, 2026, [https://www.hallucination-probes.com/](https://www.hallucination-probes.com/)  
6. Detecting LLM Hallucinations at Generation Time with UQLM | by Dylan Bouchard \- Medium, accessed April 19, 2026, [https://medium.com/cvs-health-tech-blog/detecting-llm-hallucinations-at-generation-time-with-uqlm-cd749d2338ec](https://medium.com/cvs-health-tech-blog/detecting-llm-hallucinations-at-generation-time-with-uqlm-cd749d2338ec)  
7. Entropy-Based Inference Scaling: A Novel Approach to Prevent Hallucinations and Enhance LLM Reasoning | by Mo Meskarian | Medium, accessed April 19, 2026, [https://medium.com/@m.a.meskarian/entropy-based-inference-scaling-a-novel-approach-to-prevent-hallucinations-and-enhance-llm-dfb108331108](https://medium.com/@m.a.meskarian/entropy-based-inference-scaling-a-novel-approach-to-prevent-hallucinations-and-enhance-llm-dfb108331108)  
8. The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning | OpenReview, accessed April 19, 2026, [https://openreview.net/forum?id=UfFTBEsLgI\&referrer=%5Bthe%20profile%20of%20Jiawei%20Han%5D(%2Fprofile%3Fid%3D\~Jiawei\_Han1)](https://openreview.net/forum?id=UfFTBEsLgI&referrer=%5Bthe+profile+of+Jiawei+Han%5D\(/profile?id%3D~Jiawei_Han1\))  
9. The Unreasonable Effectiveness of Entropy Minimization in LLM ..., accessed April 19, 2026, [https://openreview.net/forum?id=UfFTBEsLgI](https://openreview.net/forum?id=UfFTBEsLgI)  
10. Neural scaling law \- Wikipedia, accessed April 19, 2026, [https://en.wikipedia.org/wiki/Neural\_scaling\_law](https://en.wikipedia.org/wiki/Neural_scaling_law)  
11. Towards Large Reasoning Models: A Survey on Scaling LLM Reasoning Capabilities \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2501.09686v2](https://arxiv.org/html/2501.09686v2)  
12. Noam Brown and Team on Teaching LLMs to Reason \- Sequoia Capital, accessed April 19, 2026, [https://sequoiacap.com/podcast/training-data-noam-brown/](https://sequoiacap.com/podcast/training-data-noam-brown/)  
13. A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2501.02497v3](https://arxiv.org/html/2501.02497v3)  
14. Towards Inference-time Scaling for Continuous Space Reasoning \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2510.12167v1](https://arxiv.org/html/2510.12167v1)  
15. STaR: Bootstrapping Reasoning with Reasoning | Request PDF \- ResearchGate, accessed April 19, 2026, [https://www.researchgate.net/publication/401451990\_STaR\_Bootstrapping\_Reasoning\_with\_Reasoning](https://www.researchgate.net/publication/401451990_STaR_Bootstrapping_Reasoning_with_Reasoning)  
16. OptScale: Probabilistic Optimality for Inference-time Scaling, accessed April 19, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/40661/44622](https://ojs.aaai.org/index.php/AAAI/article/view/40661/44622)  
17. ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2407.00499v3](https://arxiv.org/html/2407.00499v3)  
18. Distribution-free, Risk-controlling Prediction Sets | Request PDF \- ResearchGate, accessed April 19, 2026, [https://www.researchgate.net/publication/357462027\_Distribution-free\_Risk-controlling\_Prediction\_Sets](https://www.researchgate.net/publication/357462027_Distribution-free_Risk-controlling_Prediction_Sets)  
19. Language Models with Conformal Factuality Guarantees \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2402.10978v1](https://arxiv.org/html/2402.10978v1)  
20. CONFORMAL REASONING: UNCERTAINTY ESTIMATION IN INTERACTIVE ENVIRONMENTS \- OpenReview, accessed April 19, 2026, [https://openreview.net/pdf/b6dc068fce2a35722d1ada14eb4c6486d4fc8604.pdf](https://openreview.net/pdf/b6dc068fce2a35722d1ada14eb4c6486d4fc8604.pdf)  
21. Conformal Reasoning: Uncertainty Estimation in Interactive Environments \- NeurIPS 2026, accessed April 19, 2026, [https://neurips.cc/virtual/2024/107878](https://neurips.cc/virtual/2024/107878)  
22. NeurIPS Poster Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models, accessed April 19, 2026, [https://neurips.cc/virtual/2025/poster/117004](https://neurips.cc/virtual/2025/poster/117004)  
23. Hao Zeng, accessed April 19, 2026, [https://zenghao-stat.github.io/](https://zenghao-stat.github.io/)  
24. Shubhendu Trivedi, accessed April 19, 2026, [https://shubhendu-trivedi.org/](https://shubhendu-trivedi.org/)  
25. Reasoning-based Anomaly Detection Framework: A Real-time, Scalable, and Automated Approach to Anomaly Detection Across Domains \- Apple Machine Learning Research, accessed April 19, 2026, [https://machinelearning.apple.com/research/reasoning-based-anomaly](https://machinelearning.apple.com/research/reasoning-based-anomaly)  
26. NeurIPS Poster AnomalyCoT: A Multi-Scenario Chain-of-Thought Dataset for Multimodal Large Language Models, accessed April 19, 2026, [https://neurips.cc/virtual/2025/poster/121641](https://neurips.cc/virtual/2025/poster/121641)  
27. Chain-of-Anomaly Thoughts with Large Vision-Language Models \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2512.20417v1](https://arxiv.org/html/2512.20417v1)  
28. Token-Guard: Towards Token-Level Hallucination Control via ... \- arXiv, accessed April 19, 2026, [https://arxiv.org/abs/2601.21969](https://arxiv.org/abs/2601.21969)  
29. LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2510.04013v1](https://arxiv.org/html/2510.04013v1)  
30. A General Framework for Inference-time Scaling and Steering of Diffusion Models, accessed April 19, 2026, [https://icml.cc/virtual/2025/poster/45673](https://icml.cc/virtual/2025/poster/45673)  
31. Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning \- Reddit, accessed April 19, 2026, [https://www.reddit.com/r/singularity/comments/1orvj0h/logitentropy\_adaptive\_stopping\_heuristic\_for/](https://www.reddit.com/r/singularity/comments/1orvj0h/logitentropy_adaptive_stopping_heuristic_for/)  
32. Daily Papers \- Hugging Face, accessed April 19, 2026, [https://huggingface.co/papers?q=thinking-token%20acknowledgment](https://huggingface.co/papers?q=thinking-token+acknowledgment)  
33. Benchmarking LLMs on Advanced Mathematical Reasoning \- EECS, accessed April 19, 2026, [https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-121.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-121.pdf)  
34. ICML Poster Position: Uncertainty Quantification Needs Reassessment for Large Language Model Agents, accessed April 19, 2026, [https://icml.cc/virtual/2025/poster/40147](https://icml.cc/virtual/2025/poster/40147)  
35. Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness \- ACL Anthology, accessed April 19, 2026, [https://aclanthology.org/2024.acl-long.283.pdf](https://aclanthology.org/2024.acl-long.283.pdf)  
36. Hao Peng \- Illinois Experts, accessed April 19, 2026, [https://experts.illinois.edu/en/persons/hao-peng/](https://experts.illinois.edu/en/persons/hao-peng/)  
37. SCHEDULING YOUR LLM REINFORCEMENT LEARN- ING WITH REASONING TREES \- OpenReview, accessed April 19, 2026, [https://openreview.net/pdf/156efed035a3eb8f8914a33059dd680054d5e73d.pdf](https://openreview.net/pdf/156efed035a3eb8f8914a33059dd680054d5e73d.pdf)  
38. Towards Trustworthy Knowledge Graph Reasoning: An Uncertainty Aware Perspective \- AAAI Publications, accessed April 19, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/33353/35508](https://ojs.aaai.org/index.php/AAAI/article/view/33353/35508)  
39. LogPPO: A Log-Based Anomaly Detector Aided with Proximal Policy Optimization Algorithms \- MDPI, accessed April 19, 2026, [https://www.mdpi.com/2624-6511/9/1/5](https://www.mdpi.com/2624-6511/9/1/5)  
40. Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2403.09629v1](https://arxiv.org/html/2403.09629v1)  
41. Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2601.14479v1](https://arxiv.org/html/2601.14479v1)  
42. StatEval: A Comprehensive Benchmark for Large Language Models in Statistics \- arXiv, accessed April 19, 2026, [https://arxiv.org/html/2510.09517v1](https://arxiv.org/html/2510.09517v1)  
43. The CompMath-MCQ Dataset: Are LLMs Ready for Higher-Level Math? \- arXiv.org, accessed April 19, 2026, [https://arxiv.org/html/2603.03334v1](https://arxiv.org/html/2603.03334v1)  
44. Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction \- ACL Anthology, accessed April 19, 2026, [https://aclanthology.org/2025.emnlp-main.569.pdf](https://aclanthology.org/2025.emnlp-main.569.pdf)  
45. Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction, accessed April 19, 2026, [https://arxiv.org/html/2509.18658v1](https://arxiv.org/html/2509.18658v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAZCAYAAACPQVaOAAAB4klEQVR4Xu2XzSsFURiHX0mRr7IjkpKVjyRssKMs2FiyFRtryYb8A2InZWVlbUGSvY2kFCkkFooNQvn4/TozvHPMjLnjTl23eerpznnPXPe8c8685xBJ+Vd8KFdV/EnF84obOG3F6mC/FcsL9uC6FTuw2jlDJVwJcR6Wfd39kyV4ptqTsES1c5JaOOhcX6s4Z60PPot5MDZD8NG5LoILqi9nqRYzcHKhO8Cm83nqiRra5LsQ7euOJGmAr+KtkK7F6r4guFSDkj10PtnPh6IpF/Mb3bDR6kuELbhsBzMkLFl3Ridgqe5wYLLbdjAMrnt+6Q0WwBGJVr4X4YAdjEFQsnwPWYRYdI5VXMNxc8yRmIPjsBCOwQcxlTAKfu9RHHSyL2J+n3/bfRWmnL6sUgV37WAArKCjdjAm9sz2wmEx7yEnIBFY0fxKvB8d8A5e/mKUJWYny+8cOe1srR4P55LZhpzkzBLWDM5wu5jDQtbgGVPPQJNES/zKDsTEL1lyImZcaxJ9xYVyL2afcqmA76odxoyYyv1Xgg4VrMbutnKr4rHYga2wU7wHgXp90y9siEk6Dpy1Zjgr5rjIpHlc7BKzOxAWqB5YI2ZrjD3DLeqaTzHunslB6v8jMz1BpaSkpKSk5BOfbdJpKxYn9MYAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAAAuUlEQVR4XmNgGAXCQDwLD/ZEKMUNngLxfyDuRBITB+LzUHF5JHEM8AmIfwOxDZo4PwNE8xw0cRQAUrAViDnQJRggciCMFWgyQCRd0CWAgJsBIvcWXQIGWhkgCrDZmsMAkTNHl4CB50D8D10QCPQYIBpBBuAEMD+h4y1AzIykDgPIMEAUpqNLEANAgQTSbIwuQQwARQ9IMyO6BDEAFFBf0QWJAZwMEFvXoEvgA1wMmCELwjzIikbByAUAz1IutmDxgncAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAZCAYAAACYY8ZHAAABTklEQVR4XmNgGAWjYBTgAiFEYgcg5oBoGVzAGIj/A/ECBoRjb0DFIoBYBkp/hIrpg3XRFxwD4ocMELf9AuIaVGkGhgNArIAm9o8B4mBkIMkAMYgbTZzWoBmIn6GJgdwGCnw4eILMYYAkF5CiPWjiME/QG4DcUo4mdpgByd06QOyBkAMDFwaIRhCNDHyB+BOaGK2BIAPELSC7kcFCqDhOsJUBomAwZGBYfsXlCZxJG5QfQJgc8J4BYjixeCdEG04ASxW4PMGDJg4GsPwAio3BAGwYyPCEEgNEshVdYoAAoeSE1RMgx4MkQZ4ZDABUIuLzBFbwnAGPJBGA2nkCBEDq0IvYAwx48i1IA8gjgwlcZ8DMo1+BeAKywDQgngvE7xgQIQSq6GYhKxpAAEr3IDepQPkJQPwXLjuEACMQTwTiR0AcgyY3CkbBKBgFo4A2AAAjKGdWaEyahwAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAXCAYAAAC1Szf+AAACC0lEQVR4Xu2WMUskQRCFn6CgaGJ0CBeaKBgZKZidoclxYGJgJhhoaGIi/gE1EhHkAsHMQE4MDJY7MDG9QxEEFTEUBA0U1KtHTWNvbU/NoC4ozAeP3a6u6uk3U9uzQEVFRcXHYEDUaYMeP20ghwPRueiH6EE0Xz+dy1fRs2hGtIL82i3RDXR9fm7UTzfQBl130E54lDG7KLoysTIXaoHm9Zq4rZ0Q3UdjwvF3E4s5ReM6hZQxy0XnTOyP6NLELFPQWoutfULjPjahhodMnIyLptEEs93QRcdMnHUpIzGHSOfEta3Zd3szOWbc7o/teyLqQRPMcjHPrHdA3ME3y9qw6TyzbNeYI6hh1+yq6CIhHhg2RjGffINvtsvEY9ienlnW9mXf88zyUAyMQluYuGY30GjIM8t8QpOvNcv5IrOhc8qY/Qt9qsQ1m8db29gzW9TGrC1q42B2BHpqB5piNiyaZ9aDG03lxLU07JmtZWP+JNYi8bTm/HY2LkWRWZLaTA26AY91pM3WUF/LHLuPcEP4+koROu5dnyzhCfjLxNiiSyb2CH0tBL5AN9QexYit3RP9i8bkTLQj6jDxgGt2GPpXzOp3IkYxPxBaLfwTmoQai6Eh5tgnuYuXG9CCdG1414ZrhjcA81OwbW+hOdfZ+F3hhZehJ3V8UJShX3Qs2odfuwBdf9ZOVFRUVHx6/gOLSLwrPK0+1AAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAXCAYAAAC1Szf+AAACQUlEQVR4Xu2Wz6tPURTF1yuKKKH8KAMDE6aiGDOQTGRooAzMmDGRkn8AIwP1MlBmBiIDgxtlYkqkFJKZgYEB8mOvt89+b3/XO/fc68eAup9avbf32fucs7733B/AxMTExH/KDU308Nj0xnTM9MV0fna4l22mH6bTpqvo771l+gifn3/nZ4cX2AOf66zpafl/ey4YYozZS6b3kuNCuyWnzMHrdkhee4+bPqeYMD6a4hXwvkMp96HkuM4oxpjlhOck98j0TnLKKXivor3fsXwfN+GG95V4M3yuT4sVwJOS25pyTXQRZT18wiOSZ1/NSCY2o+TeuGL6YzJmPu9vp2llir+iPn8vQ2Z53Fpm10g+w6tQ20zu5VVpmX0l+WAvfPywDpBrprcV8YGhOYr15ADaZtdKPsPj2TLLXl6tllk+FGt5aouMLTKP5YZaZllPaPJ3zcamlNwbJ2es2VXw03ABPr5hdrjNnx7jltmhY8zeoWOsZjO34TWbdKCPIbOxmT6zLbjRWk3upeGW2U7ymTh1D3SgjyGzpLaZDn5PtriOutkOs72s0X3ED8LXF+lKfDIKsGSWT/1R6CI1npvuSo5H9LLkvplepjjejbzPMtp73/QsxeS16Y5pdYk7+FwHS0z4RcZc/vhYYD/8U0z1sJKjWB/EUYsvoRNwYxkaYo1eyXtY+gHmUO+Nd22sGW8A1gfrSm5XifkkZsz9/3W48BX4k5qfd78CN/gCfm+1ei/C5z+jAwkeY9Zwro0yNjExMfHv8hMKCsOcjnEAzQAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAXCAYAAAC1Szf+AAACMUlEQVR4Xu2WP0hXURTHzw8aEnNoKsFwcdHVyea2yKEcGwQFnRRcXBwU50CbGgJxEBraomhw+FHQ0ixGEKSIi4M4OBSUnm/n3Tzv+7t/fpaL8D7w5fe7555z3/2+d9+7V6ShoaHhmrLJgQSfVHuqCdVP1VK9O8mA6kw1p3ou6dpXqhOx8fG7Ue/+w7jql2pfbEzobi2jQDdmV1WHFMOFRinGtMTyhijOtU9VP1wboP3Yte+ptl0bwDjGukHxJN2YxYCLFPuoOqAYMyNWy3Dtb+mcx5aY4bGqjX6MNfU3w24GYgsuloUvwtwWG/ARxcPFc3yWeI6vxVOJ3Uy0EQ/zm63aWMoBrA6fU6SUGAZMme2luOdU8mZR21/9T5n9RnHPsljOA4rLC7EXm4UPBscg5AMMlDN7i+IeLM+cWdQOV/9TZvFRTIH+Yw6CDek0lDOLfACT/2oW/SWzYeVc1uygWP8T7sjxv8s4Z7a0jFFbWsYxsy2xvofcUaJkNkwmZTYHJhrL8bUwnDPbpjjAlsPbWVeUzILYZNpi72SOlxI325Z6LXJ4HuGGYPvyHEn9IIGHcWVbD9hVvaUYlugaxXDHv7r2HbEJ33QxwLXvVTuuDb6r3qh6XAyHimmxU1bQa4kcbu5LPSnoQyQGIT8QllpYOpNixjwwhBx+ku/k4gbgXZuUztqw14Zrhh0A+YFnVSymPpd3JeDC62JfahzvLsOI6ovYk8nVroiNP88dDQ0NDdeecy4EwPe7z3G+AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAXCAYAAAC1Szf+AAACSklEQVR4Xu2WsUtcQRDGR6KQoBEEQQSFICnUNiBELG3TJHZa2KWLpY1FJP+A2mgR0BSBFOkkwcLiULBJYyOKIJggFrEQQYsENJnP3Xk3b27f7nlGiPB+8HE3szN7++3b2zuikpKSknvKB5soYIv1nTXG+s2ayQ8X0sP6w3rDWqDi3k+sM3Lz43U5P5zxhNx8H1krrBM9mKIes+9YxyaHD3xmcpYmcnVPTd72TrB+qRggfmlyQ+R6W3yMzUPcnVUkqMcsJpw2uU3WkclZXpPrtdjeK6pdB54cDD9XOcz1VsWvfK5Z5aLYD7F0kJvwhcmjL2RE843CNboXCw1tJmLkZX0jPu7LKhogZRbHLWa21eQ1FxQ3i14cwZjZAx9/9nE/65z108eP/HiOJdaPgHBh2ByEejBKcbNtJq/B8YyZRe+Af19kFpciwCvijazCfddD81/fbtZQzCzqAUw2ahbjocXoXjk59ZodzyqqN729AAu57TGOmU0dY/SmjrGY3fFxZ1bh+pH7onJRUmZlMUVmY8jTsOheWXCR2YqPcTvbzZVe2ZAkKbMgtJgKue9kjPcUNluhfC9q7DpkQ/DzBeTueJxV3MGTBbtUOyGO6JzJXbL2VdxFbjEPVQ7Y3jVyx1RzyFql/G2LufATJPT5XM13dpjcXzEr3G42B6FekB2USSfJGdPAEGrsk/xK1Q1oonCv/NbKZ8pTRL1G/kQ88GOnrMVcxT8Ck8+Tu6lx5d+EQdYea53ivbPk5p+yA4p2cvNss3rNWElJScn/y1+qDMfe4I2h+QAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAAAZCAYAAADHXotLAAAC1ElEQVR4Xu2ZO6gTQRSGj6igKIgPvPhofBU+wEIUfJQqWiiCjWBpqyg2olUaSxuxEBuxsriKrWgTsNPCSmwUriCIiFoJivg4H7PjPTmZ3ewmmJu57Ac/yfw7O9ndmXNmJivS0tLS0vIfeOGNTLmhWuTN3NivOunNTNmkOu/N3Hjtjcz55o2cILy73sycjoRIyRLCe7M3M2ep6rE3cyHr8K7gvTc8b1R/SnTb1BsnhHXHm8on1SUJ17bA+GsLb7nxmrBKwvnXVC+l2cpuheqD6onqufFvSWhzjfHgsOqY8/7xVXVOtU61SzVdfEcLTb0yfAcO0oVw2kAIa8LbckW1VcIN0taUO4Y3DBslnLvYeJR5cHX4UnzGDohwrZS3Gy9CB/ZxQLXblG+qjpvyXJIK67IbB+o3GdWWH9IfjbR/z3kpiMjrxfffEq7N8k61xHnwQPoHXB/cUKo3xw0pwN+YZaaQhYdx13l1YCXHw99jPFJh3Q6JxHZ8KqJDUpyRsMeqhAYnZSeZihCID8tuFhmBeHuNVxfa8dFG5+CddX4V1OUcO6/BjCtHBkYIkeEvrA6c00SjzCHAQ6cdmwbI9XgMpoPFZ13ICp+dd1/SD7eKVDtEgI+YSHIOsdBjr7w5h2yQ/rwOJ6R34MR1/c+ibG/0oYS6drL2cJx7j8QIPG282M4d43neqp45jxVhispVVoQfjJPTpJDah8RczYJkteq76qmEDmFVc3W2qnyUUPey8SwrJRznd7aplql+qS7aSjLbjh0IniMSjtPGDglLZ+bCFGXpuAdGXpMQHQdVO/Wjqi2mvF56J+YIqyCWxCli3oedqn3mWIquNxxsE05J73Lck/VOnWjoerMhhyTsr1Kk8n4VtUb2ADqS8X9ZMOq/vam0FyE62HvVgZSERqXqerJglPchnFeWx8n1dEid0Uo7j7w5BPPifQgMuwOfNObFG8OWlpYW5S/AbqbPe9W6RAAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAVCAYAAAApZJKFAAADMElEQVR4Xu2ZTahNURTH/0KRr0RK4V2UEjIQBmQgBhIKA2WoKBkZMHRLBmYyEgMZGPgoycTAYJeSGCgpEvXIR5JEKORj/Vt7vbvPvuece/Z95z7P6/7q37l7rX3222edvdfa9z6gT58+fUYVX0V/RCdEu0t0RHRJ9Mv3N81Gn8pMRStw4yJfGZuh95yPHT3A5seF8d+zC/owz2NHB6ZA7xsJHMZIsMktaOCYLlLYI1oeG3uAwxgKNrHtOhA7RgEOYyzYDLIFvA4mit6Kzvn2WtH7oJ2CQ36wmcoGoXN+LZqe8SqzRE9EP0S3RZNEn6D3zAn6GUtEn6HptacwjXASTCszIl8q7/yV430Qrfbt46I3/nNVHNqDvRV6OrLCzivbm4Z6tAr5eN/mM1mQuRDmebvBMe6JZqK+RVdIeDo5FflSeeyvHOtFYGcwUh/EoT3YLOiXIxvbT4P2XeiLDuHfbkY2Y6HoEPRImzrHrrDTCdXIuiqzQrQdrRWyKPBt8za+WIMPx+1bhEM22AwKx1gZ2Mgqb6efxC+6yGacgc5rL4JgTxa9rCgOwP4p2OnkbOxIJDNpz1VvmxDYrol2Bu0Yh2ywD0PHmBvYiAX7gG87tO8I+rl6y7gPTTMjAoMdbsduyZv0T9H3yNYJh2zQ7CWGO4ZYsLl7yHzRb9ED6ML5Bp1TJzgGv1X3HBbJuipxPGkrUCxchKmDBenhUI98HLLBtnHi3cA27baTL4qmtdyVyEt9PWENhp86DJu0C2wsaleC9lF/Zb+ynwsc2tMB5xnbuGNOBu0m9ITCvqZ90ONfEbZr4hdZKwNITx08UoWFLsQmfdBfqY2ZHgqL2WBsDOB9r6D1h5/DPL3B20xsh6wLfLGKFhXTDE8wd2JHXfALCFdAKjdQfB63SZfB7X5TtD521IClmrwd04D68qD9dGysEwaaAU+BFb2s0FWZdBOth74Q2OvAjod58CXn+ex8vTR21AVTR9HqzKMBreqcVDPjURaLHkH9z0THsu4MrBEfkf6LY1WYvzmPZWgtph3etsU6efaLvnjfdbSnpGHDvGU5rBulvKR/yQLocZD/COFvJZX4C/Fw6ysxr2TzAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAAB/klEQVR4Xu2WPyhHURTHj1CEJMVCJgMpk0EZkfJnIoOyK1kMYjMos0kWMUsWJRlkkoXBYLAoZZBkMCB/zrd77+9d5533fvf3x2/Q71Pfevece+87775zz71EZcrkTQVrltUrHQFcsuqkMRvdrGfWt6d56xtkDdhnn0/WsjTmAMZXS6NGLZmA3lnNwjfN2ifjrxG+Y9aQsOVKO5mFSaWDTACb0uHxwPoSthbFli/3lPKxbgWvpUOwwzoUNrQ3hC1fRlhP0uh4JBNktpxYI5OTPhjXJWz5gjTCfFXS0WMd59KhUK+01Ukto6xb1gKZfv2sC9aubWvjYI9VCPwuOOQKhYDJMFYDubrltV8p2hgYk/QH0G9GM2JAg3QEgA9LCvKAfq8U+i3ZZ6TNiufzuaOoXwYXJIpxGpUULz3jlBykD4JFP62+StQgsZtCXoSvbxW2Pgobi+CSclCCMhcLEuUDEzRJhwd2vVaekFNpQbqNtsf68OyYb85r+6g56X7FiXRYcNLgBVo6pO1uFHj3AXg+jVy0TWaXa2BMbHeDNjLOF4p+aSOZ/BgjPUAHxmm79I3VyVplrZNZIcyD833RPktcndR8GaZYN6wrCi9JR5R84gxTdLtBAJNkNmASE2SOxqKDlS/m2Y2C/yecUcrFIJCgW1ChlOw+WQglv5mX+Xf8ALL8bt2LpwzJAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAuUlEQVR4XmNgGAUDBpYD8R0gZkUTL0bmXGGAKDgAxFeRxG2A+D8QM4I4gkDcCpX4B8RboWwQALF/I/HBQJoBohtEwwBI4xokPhhEMCBZAwUgfhESHwzuAvFzJL4SA0ShOJIYGIAEYW4FgSqoGAZ4C8RLoWyQ9SBFIFswADcDRBKEQeEJotNRVDBAwlAfia/DAFHIiSQGBucYIEEBA1PQ+HAA0n0MytaE8vkR0ggQx4Bw3ykG1LAcMgAA8ron45zXLqgAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAABB0lEQVR4XmNgGAWjYBQMCcANxA+B+ACa+KAFkkD8H4hnQOkDKLJDBIw6nN5goB0uxABxwywgroKy5ZHkOZDYKGAgHW7CALGfH0ksAioGAh5AbIwkhwIGyuGMDBC7p6CJi0DFNYH4GZocCiDF4dFA/IhIfAKqBxfQYYDY7YImzgMVtwXi22hyKIAUh1MLgNLtVgaI3aCQR5cDiX8BYmE0ORQwEA4H1SOgig+WltEBSLwSXRAdDITDQaAViP+hCwKBOwPETaBkiQFA0QEqfi4xQBSB8AaoGD3BXyBOhbK1gfgbENsD8RUgXg/EjVDxQQmUgTiEAZIpkYEDA2oxOQpGwSgYBaOAgQEATfhBUD3D0yoAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAWCAYAAAASEbZeAAAAqUlEQVR4XmNgGNygF4j/Q7EfmhwKUGaAKNJEl0AG0QwQRXjBVSB+iy6IDkCmTEIXRAYiDKjuqQHi1QwQ0+EgnQHhnsVAzA/Ey5HEwGApVAAkCVIEAluhYnDwFSowBVkQGTAyQBQkAvFsKNseRQUQGEMleKH8wwwQk0FAH0ozzGFAtfshEJ+Gsh/ABEE6kQPxLhAfgLIfwQT/AnExjAMEMgyIyGZFEh80AAAzbyifstCHVAAAAABJRU5ErkJggg==>