# **State of the Art in Chain-of-Thought and Agentic Hallucination Detection: A Multiview Uncertainty Perspective**

The landscape of large language model reliability has transitioned from the analysis of static, single-turn outputs to the evaluation of dynamic, multi-step reasoning trajectories. As models like Falcon-3-10B and Phi-4 increasingly rely on Chain-of-Thought (CoT) processes and agentic scaffolding to solve complex tasks, the failure modes of these systems have become more structural and less transparent. Hallucination in this context is no longer a simple factual fabrication but an evolving latent state that propagates across reasoning steps, often masked by an appearance of internal coherence and high linguistic confidence.1 Detecting these failures without ground-truth labels requires a sophisticated understanding of uncertainty dynamics. The following report provides an exhaustive survey of current methodologies for detecting hallucinations in reasoning traces and agentic flows, focusing on signals extractable in gray-box environments through unsupervised multiview fusion.

## **1\. The Evolutionary Dynamics of Hallucination in Reasoning Traces**

The emergence of reasoning-centric large language models has redefined the definition of hallucination. Traditional metrics that measure token-level surprisal or answer-level consistency are often inadequate for models that generate long-form intermediate logic before reaching a conclusion. In these settings, hallucination is best understood as a latent state that develops over time.1

### **1.1 From Point-wise Errors to Latent State Transitions**

Recent research suggests that a reasoning model does not simply "make a mistake" at a single token but rather enters a "hallucinated regime".1 This state reflects the degree to which a reasoning trajectory has been dominated by an incorrect or insufficiently supported prefix.1 Because of the autoregressive nature of these models, each subsequent reasoning step inherits the context of the previous ones. A minor grounding error in the second or third step of a ten-step chain can irreversibly bias the model's internal belief state, leading to what is termed a "Spiral of Hallucination".3

The transition into this hallucinated state is often subtle. Intermediate steps may remain locally plausible and internally consistent, yet the cumulative prefix-level hallucination signal tracks a global deviation from the factual foundation.4 This implies that a reliable detection framework must distinguish between step-level judgments—local observations of correctness—and prefix-level states, which represent the global evolution of the reasoning process.1

### **1.2 The Confidence Masking Effect of Reasoning**

A critical challenge for hallucination detection is the "double-edged" effect of Chain-of-Thought prompting. While generating reasoning steps generally improves task performance and reduces the absolute frequency of errors, it simultaneously obscures the cues used for detection.5 Cheng et al. (2025) demonstrate that CoT reasoning significantly affects the model's internal states and token probability distributions, essentially "semantically amplifying" the model's internal confidence in its output.6

Even when a reasoning model produces an incorrect answer, it tends to generate the erroneous tokens with high probability. This phenomenon, known as "confidence masking," results in flattened token-entropy distributions that make it difficult to distinguish correct answers from hallucinations using probability-based metrics alone.6 The model effectively "convinces itself" of the wrong logic path, creating a trade-off where improved task performance is accompanied by reduced hallucination detectability.6

## **2\. Agentic Flow Validation and Error Propagation**

Agentic LLM pipelines, which coordinate multiple multi-step subtasks and tool calls, amplify the risks associated with hallucination. In an agentic setting, a failure is no longer just a textual error; it is a system-level failure involving wrong plans, incorrect tool parameters, or unsupported business actions.9

### **2.1 Cascading Failures and the Curse of Recursion**

The primary bottleneck to reliability in multi-step agent workflows is the cascading failure problem. Research from 2025 confirms that error propagation from early mistakes into later stages is the most common failure pattern in failed agent trajectories.11 Unlike traditional software that might throw an exception when encountering bad input, LLM agents often treat malformed or hallucinated tool outputs as valid premises and continue reasoning.3

This leads to a phenomenon called the "Curse of Recursion," where grounding errors propagate through the context window, biasing all subsequent planning towards an irreversible failure state.3 In multi-agent systems, this is further amplified by transitive trust chains: Agent A generates a hallucinated fact, which Agent B accepts without verification, leading the entire system into a self-referential loop of uncertain information.11

### **2.2 Hierarchical Failures in Tool Use and Planning**

Agent-specific hallucinations are categorized by where they occur in the execution flow. Tool Calling Hallucinations (TCH) encompass errors in invocation structure, arguments, and execution handling.14 These range from explicitly malformed calls detectable by schema validators to semantically plausible but incorrect calls that evade surface checks.14

| Failure Mode | Description | Detection Challenge |
| :---- | :---- | :---- |
| **Planning Hallucination** | Generation of invalid, circular, or logically impossible sequences of actions. | Requires understanding of task-specific logic and environment constraints. 15 |
| **Tool Parameter Fabrication** | Inventing arguments or values that do not exist in the retrieved context. | Often appears confident and semantically relevant. 14 |
| **Goal Drift** | Gradual loss of the original objective during multi-turn interactions. | Requires long-horizon memory and intent tracking. 10 |
| **Citation/URL Fabrication** | Generating links or references that look valid but are non-resolving. | Rates can reach 13% even in advanced research agents. 17 |

## **3\. Uncertainty Signals from reasoning traces**

In the gray-box setting, the goal is to extract scalar uncertainty signals from the Reasoning-Trace without needing weight access. Current state-of-the-art methods rely on the temporal evolution and structural properties of entropy along the trace.18

### **3.1 Entropy Trajectory Monotonicity**

One of the most informative signals for reasoning correctness is the "shape" of uncertainty dynamics. Zhao et al. (2026) introduced the concept of entropy-trajectory monotonicity, which asks how the model's uncertainty over the final answer evolves step by step.19

A trajectory is considered monotone if the per-step answer-distribution entropy decreases at nearly every step, reflecting a consistent reduction in uncertainty as the model approaches the solution.20 The scalar coherence—the total entropy drop from the start to the end—is often unpredictive, but the *consistency* of the descent is highly correlated with accuracy.19 For numeric tasks, monotone chains achieve significantly higher accuracy than non-monotone ones, with accuracy gaps reaching over 20 percentage points.19

### **3.2 Entropy Dynamics Instability Score (EDIS)**

Beyond monotonicity, the instability of token-level entropy evolution provides a diagnostic lens on reasoning failure. Erroneous reasoning is characterized by two distinct patterns: "burst spikes" and "peak-valley rebounds".18

* **Burst Spikes**: These occur when entropy rises steadily over consecutive reasoning tokens, indicating that the model is becoming progressively confused.18  
* **Peak-Valley Spikes**: These are sharp rebounds following transient confidence, suggesting the model has realized a logical inconsistency in its hallucinated path and is attempting to course-correct with new fabrications.18

The Entropy Dynamics Instability Score (EDIS) quantifies these fluctuations, serving as a more effective diagnostic signal for inference-time selection than mean entropy or self-certainty.18

### **3.3 First-Token Branching Uncertainty**

The first token of each reasoning step often acts as a critical fork where the model chooses between multiple plausible reasoning branches.24 Research has shown that these branching tokens exhibit higher entropy and lower log-probabilities than subsequent tokens in the same step.24 Monitoring the entropy of the first token following newline delimiters or step markers provides a low-cost signal of logical commitment. If the model selects a path at a point of high branching entropy, it is significantly more likely to be a guess that leads to downstream hallucination.24

## **4\. Signal Inventory: extractable Uncertainty Signals in Gray-Box Settings**

The following table summarizes the signals extractable from standard generate() interfaces that can be used as "views" in a multiview fusion framework.

| Signal Name | Computation Method | Compute Cost | Orthogonality to EPR |
| :---- | :---- | :---- | :---- |
| **Trajectory Monotonicity** | Shannon entropy of answers sampled at step boundaries. | Moderate (requires ![][image1] completions/step). | High; captures process convergence rather than token surprisal. 19 |
| **EDIS (Instability)** | Rolling window detection of entropy "bursts" and "rebounds." | Low (post-processing of greedy log-probs). | High; sensitive to false-confidence cycles that EPR averages out. 18 |
| **Branching Entropy** | Entropy of the predictive distribution at step-start markers. | Very Low (single token log-prob extraction). | Moderate; correlates with difficulty but decorrelated from trace fluency. 24 |
| **Semantic Energy** | Boltzmann-inspired energy over NLI-clustered samples. | Moderate (requires sampling and NLI check). | High; captures inherent model confidence better than post-softmax SE. 26 |
| **Reasoning-Answer Alignment** | NLI-based consistency check: Does step ![][image2] entail the final answer? | Moderate (requires second model/API call). | High; detects "rationalized" hallucinations with low terminal entropy. 27 |
| **Token Constraint Bound** | Max hidden state perturbation before the top token flips. | High (requires noise injection via forward passes). | High; measures robustness rather than probability. 28 |
| **Chain Disloyalty Score** | Measuring the degree to which a model self-corrects or hedges internally. | Moderate (linguistic analysis of the trace). | High; reflects meta-cognitive state rather than factual density. 30 |

## **5\. Multiview and Ensemble Approaches to reasoning-trace Uncertainty**

The core requirement for algorithms like Nadler spectral fusion is the existence of multiple uncertainty "views" that predict the same underlying correctness signal but make errors independently.32

### **5.1 Uncertainty Deconvolution and Aggregation**

Recent frameworks like RACE (Reasoning and Answer Consistency Evaluation) demonstrate the power of joint evaluation. RACE decomposes hallucination detection into four components: reasoning consistency across multiple sampled traces, answer uncertainty via semantic entropy, semantic alignment between reasoning and answers, and the internal coherence of the logic chain.27 This integrated approach enables the detection of hallucinations that previous methods overlook, such as "lucky" or rationalized correct answers derived from flawed logic.33

### **5.2 The Theoretical Argument for Decorrelation**

A foundational question for multiview fusion is why reasoning-trace uncertainty should be decorrelated from final-answer EPR. Theoretical analysis using the "Token Constraint Bound" (![][image3]TCB) suggests that token probabilities are often an artifact of softmax normalization and may mask underlying prediction instability.29

A high-probability terminal token can arise from relative normalization even if the originating internal state is brittle. Conversely, the uncertainty trajectory of the trace measures the model’s "feeling of knowing" and "feeling of error" during the construction of the logic.34 Because the trace is a temporal process where each step provides a new opportunity for error, its uncertainty profile captures the *stochastic instability* of the reasoning path, while the final-answer EPR captures the *surprisal* of the final state.18 These are fundamentally different lenses on the same failure mode: factual untethering.

## **6\. Key Papers to Read (2021–2025)**

The following papers provide the technical and theoretical bedrock for the multiview analysis of reasoning and agentic reliability.

1. **Cheng et al. (2025). "Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models."**  
   * *Contribution*: Proves that reasoning-enhanced inference reshapes the landscape of hallucination detection, making signals more subtle and "masking" overconfidence.6  
   * *Relevance*: Establishes why terminal-token metrics like EPR are insufficient and must be augmented by process-level views.  
2. **Zhao et al. (2026). "Entropy Trajectory Shape Predicts LLM Reasoning Reliability."**  
   * *Contribution*: Introduces entropy-trajectory monotonicity as a primary triage signal for single-chain reasoning reliability.19  
   * *Relevance*: Provides a concrete, compute-efficient scalar signal for multiview fusion that focuses on uncertainty dynamics.  
3. **Duan et al. (2025). "EDIS: Diagnosing LLM Reasoning via Entropy Dynamics."**  
   * *Contribution*: Formalizes "burst" and "rebound" patterns in entropy as fundamental properties of reasoning failure.18  
   * *Relevance*: Identifies a behaviorally grounded signal that targets "false confidence" cycles common in Falcon-3 and Phi-4 models.  
4. **Lu et al. (2026). "Streaming Hallucination Detection in Long Chain-of-Thought Reasoning."**  
   * *Contribution*: Models hallucination as an evolving latent state and proposes continuous tracking of prefix-level judgments.1  
   * *Relevance*: Supports the implementation of detectors that monitor the reasoning process in real-time before the final answer is committed.  
5. **Duan et al. (2025). "UProp: Information-Theoretic Framework for Uncertainty Propagation in Sequential Decision-Making."**  
   * *Contribution*: Decomposes sequential uncertainty into intrinsic and extrinsic parts to analyze error inheritance in agentic chains.35  
   * *Relevance*: Offers a principled method for aggregating step-level reliability into a final system-level confidence score.  
6. **Ke et al. (2025). "RACE: Reasoning and Answer Consistency Evaluation."**  
   * *Contribution*: A black-box framework that jointly captures intra-sample inconsistencies in reasoning traces and answer consistency.27  
   * *Relevance*: Demonstrates that alignment between the logic chain and the conclusion is a critical, independent view of factuality.  
7. **Farquhar et al. (2024). "Detecting Hallucinations in Large Language Models Using Semantic Entropy."**  
   * *Contribution*: Groundbreaking Nature paper on clustering-based entropy for quantifying conceptual rather than lexical uncertainty.37  
   * *Relevance*: The benchmark approach for behavioral uncertainty views in any fusion framework.  
8. **Chen et al. (2025). "Semantic Energy: Penultimate Layer Uncertainty for Hallucination Detection."**  
   * *Contribution*: Proposes using pen-ultimate logits to capture inherent confidence that is often lost during softmax normalization.26  
   * *Relevance*: While technically white-box, its focus on "energy-based" uncertainty provides a theoretical bridge for more robust gray-box proxies.  
9. **Wang et al. (2026). "AgentHallu: A Comprehensive Benchmark for Automated Hallucination Attribution."**  
   * *Contribution*: Introduces the first large-scale benchmark for multi-step agent trajectories with a granular taxonomy of failure types.15  
   * *Relevance*: Essential for the rigorous evaluation of agentic hallucination detectors across diverse tasks.  
10. \*\* Cohen-Inger et al. (2025). "Token Constraint Bound (TCB) for Prediction Robustness."\*\*  
    * *Contribution*: Mathematically quantifies how sensitive a model's prediction is to internal state perturbations.28  
    * *Relevance*: Provides the theoretical justification for "stability" metrics that look beyond probability to ensure a model's answer is well-grounded.

## **7\. Recommended Approach for Nadler Spectral Fusion**

Given the thesis constraints—gray-box, single model, unsupervised, 10s forward pass—the most promising new "view" to add to the existing configuration is the **Step-Boundary Uncertainty Trajectory (SBUT)**.

### **7.1 Signal Definition and Extraction**

The SBUT signal targets the decision points *between* steps, rather than the content *within* steps. The reasoning tokens within a step are often high-probability because they follow a local narrative logic.24 However, the transitions (e.g., at tokens such as "Step 2:", "Therefore,", or newline characters) represent critical junctures where the model must choose a new logical direction from its world knowledge.25

**Extraction Process**:

1. Generate a greedy CoT trace at ![][image4] and record the log-probabilities.  
2. Segment the trace into ![][image5] steps using delimiters (e.g., \\n\\n or explicit "Step k:" markers).20  
3. Identify the first non-indentation token of each new step, ![][image6].  
4. Compute the scalar Shannon entropy of the predictive distribution at each ![][image6], using the top-K log-probs:  
   ![][image7]  
5. Formulate the view as the **Monotonicity Violation Count** across the steps:  
   ![][image8]  
   where ![][image9] is a noise tolerance threshold.19

### **7.2 Why it fits the Nadler Framework**

This signal satisfies the two core requirements of Nadler spectral fusion:

1. **Low Pairwise Correlation**: Terminal-token EPR measures the *fluency* and *surprisal* of the output, which is often low even in incorrect answers due to rationalization.7 SBUT monotonicity measures the *convergence* of the model's logic. These have been shown to have near-zero Spearman correlation (![][image10]) in pilot studies, meaning they offer genuinely independent information about the model's internal confusion.19  
2. **Common Target Correctness**: Both signals aim to identify if the model has drifted away from the facts. A high violation count in SBUT suggests the model is "guessing" its way through steps, which strongly predicts final answer incorrectness.18

## **8\. Gaps in Current Literature**

A thesis on multiview fusion is positioned to fill several critical gaps in the existing research body.

### **8.1 The Independence-Reliability Trade-off**

While it is known that Nadler fusion requires independent views, the literature lacks a systematic study of the Spearman ![][image11] covariance matrix across all known gray-box signals. The MSc finding that prompt-template variation violates independence is a significant observation that is currently underexplored.40 A thesis could provide the first formal "Map of Orthogonal Uncertainties," identifying which combinations of behavioral and internal signals maximize the spectral gain.32

### **8.2 Hallucination Recovery Mechanisms**

Current research models hallucination as an irreversible spiral.3 However, anecdotal evidence suggests that stronger models sometimes "self-correct" during long thinking cycles.41 There is a gap in defining uncertainty-based markers for **productive exploration** vs. **terminal logic drift**. Developing a signal that distinguishes an entropy spike that *resolves* (discovery) from one that *sustains* (failure) would be a novel contribution to step-level aggregation strategies.18

### **8.3 Gray-Box Proxies for White-Box Stability**

Metrics like ![][image3]TCB provide deep insights into prediction robustness but are computationally expensive.29 There is a gap in the development of "API-compatible proxies"—for example, can we use **Temperature-Scaled Logit Variance** as a high-fidelity surrogate for hidden-state sensitivity? Validating these proxies would democratize advanced reliability checks for developers using Falcon-3 or other hosted models.

## **9\. Conclusion: Towards an Integrated Uncertainty Framework**

The current state of the art in hallucination detection highlights a shift from checking *what* the model said to monitoring *how* it arrived there. The Chain-of-Thought reasoning process is a rich source of structural signals, from the monotonicity of entropy trajectories to characteristic instability patterns like burst spikes.18 In agentic workflows, the critical failure mode is the propagation of uncertainty across tool-use and planning stages, requiring a hierarchical approach to validation.11

For a gray-box, unsupervised MSc setting, the integration of trajectory-based views into a spectral fusion framework represents the most promising pathway to exceeding the performance of single-view metrics. By leveraging the low correlation between process-level convergence (monotone descent) and outcome-level fluency (EPR), the Nadler framework can effectively "see through" the confidence masking effect of reasoning-centric models.6 Future research should focus on the refinement of these process-aware views and the development of robust benchmarks that challenge models to maintain grounding across long-horizon trajectories.15

### **Benchmark Reference for CoT Evaluation**

| Benchmark | Focus Area | reasoning-trace Requirement |
| :---- | :---- | :---- |
| **AgentHallu** | Attribution of multi-step agent failures. | High; requires tracing planning and tool decisions. 15 |
| **MATH-500** | Monotonicity and convergence in math. | High; standard for entropy trajectory analysis. 19 |
| **HALLUCINOGEN** | Implicit reasoning in vision-language tasks. | Moderate; focuses on overriding language priors. 45 |
| **RAGTruth** | Clinical and legal document summarization. | High; evaluates grounding in long, noisy context. 15 |
| **AA-Omniscience** | Honesty and the "feeling of knowing." | Moderate; penalizes guesses and rewards refusal. 43 |
| **GSM8K** | Simple numeric logic paths. | Low; good for testing basic monotonicity. 19 |

#### **Works cited**

1. Streaming Hallucination Detection in Long Chain-of-Thought Reasoning \- arXiv, accessed April 14, 2026, [https://arxiv.org/pdf/2601.02170](https://arxiv.org/pdf/2601.02170)  
2. Streaming Hallucination Detection in Long Chain-of-Thought Reasoning \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2601.02170v1](https://arxiv.org/html/2601.02170v1)  
3. Agentic Uncertainty Quantification \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2601.15703v1](https://arxiv.org/html/2601.15703v1)  
4. Streaming Hallucination Detection in Long Chain-of-Thought Reasoning \- OpenReview, accessed April 14, 2026, [https://openreview.net/pdf/d1cb22c349d7c36d56c25efeaf576a857f1d2cb3.pdf](https://openreview.net/pdf/d1cb22c349d7c36d56c25efeaf576a857f1d2cb3.pdf)  
5. Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation \- ACL Anthology, accessed April 14, 2026, [https://aclanthology.org/2025.findings-emnlp.67/](https://aclanthology.org/2025.findings-emnlp.67/)  
6. Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2506.17088v3](https://arxiv.org/html/2506.17088v3)  
7. Chain-of-Thought Prompting Obscures Hallucination Cues in ... \- arXiv, accessed April 14, 2026, [https://arxiv.org/abs/2506.17088](https://arxiv.org/abs/2506.17088)  
8. Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models: An Empirical Evaluation | Request PDF \- ResearchGate, accessed April 14, 2026, [https://www.researchgate.net/publication/397418978\_Chain-of-Thought\_Prompting\_Obscures\_Hallucination\_Cues\_in\_Large\_Language\_Models\_An\_Empirical\_Evaluation](https://www.researchgate.net/publication/397418978_Chain-of-Thought_Prompting_Obscures_Hallucination_Cues_in_Large_Language_Models_An_Empirical_Evaluation)  
9. AGENTIC AI IN THE WILD: FROM HALLUCINATIONS TO RELIABLE AUTONOMY \- OpenReview, accessed April 14, 2026, [https://openreview.net/pdf?id=3ieIcGKqrV](https://openreview.net/pdf?id=3ieIcGKqrV)  
10. AI Hallucinations Are Getting Smarter — Here's How to Catch Them in Real-Time (Even in Agentic AI Systems, 2026\) | by Yash Mishra | Feb, 2026 | Medium, accessed April 14, 2026, [https://medium.com/@yash.mishra0501/ai-hallucinations-are-getting-smarter-heres-how-to-catch-them-in-real-time-even-in-agentic-3d75a9fc1ab3](https://medium.com/@yash.mishra0501/ai-hallucinations-are-getting-smarter-heres-how-to-catch-them-in-real-time-even-in-agentic-3d75a9fc1ab3)  
11. What Is Toolchaining? Solving LLM Tool Orchestration Challenges | Future AGI Blog, accessed April 14, 2026, [https://futureagi.com/blog/llm-tool-chaining-cascading-failures-production/](https://futureagi.com/blog/llm-tool-chaining-cascading-failures-production/)  
12. Addressing one of the Biggest Misunderstandings in AI | by Devansh \- Medium, accessed April 14, 2026, [https://machine-learning-made-simple.medium.com/addressing-one-of-the-biggest-misunderstandings-in-ai-4d6278213a46](https://machine-learning-made-simple.medium.com/addressing-one-of-the-biggest-misunderstandings-in-ai-4d6278213a46)  
13. Marshall's Monday Morning ML — Archive 001 \- Medium, accessed April 14, 2026, [https://medium.com/@jung.marshall/marshalls-monday-morning-ml-7af6a0d2b77f](https://medium.com/@jung.marshall/marshalls-monday-morning-ml-7af6a0d2b77f)  
14. Tool Execution Hallucination in LLM-based Agents: A Unified Taxonomy with Detection, Mitigation, and Future Directions \- TechRxiv, accessed April 14, 2026, [https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177219979.94060974](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177219979.94060974)  
15. AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2601.06818v1](https://arxiv.org/html/2601.06818v1)  
16. AI Hallucination Detection Tools: W\&B Weave & Comet \- AIMultiple, accessed April 14, 2026, [https://aimultiple.com/ai-hallucination-detection](https://aimultiple.com/ai-hallucination-detection)  
17. Detecting and Correcting Reference Hallucinations in Commercial LLMs and Deep Research Agents \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2604.03173v1](https://arxiv.org/html/2604.03173v1)  
18. EDIS: Diagnosing LLM Reasoning via Entropy Dynamics \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2602.01288v1](https://arxiv.org/html/2602.01288v1)  
19. Entropy Trajectory Shape Predicts LLM Reasoning Reliability: A Diagnostic Study of Uncertainty Dynamics in Chain-of-Thought \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2603.18940v2](https://arxiv.org/html/2603.18940v2)  
20. Entropy Trajectory Shape Predicts LLM Reasoning Reliability: A Diagnostic Study of Uncertainty Dynamics in Chain-of-Thought \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2603.18940](https://arxiv.org/html/2603.18940)  
21. InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning \- arXiv, accessed April 14, 2026, [https://arxiv.org/pdf/2603.17310](https://arxiv.org/pdf/2603.17310)  
22. Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought \- arXiv, accessed April 14, 2026, [https://arxiv.org/pdf/2603.18940](https://arxiv.org/pdf/2603.18940)  
23. (PDF) EDIS: Diagnosing LLM Reasoning via Entropy Dynamics \- ResearchGate, accessed April 14, 2026, [https://www.researchgate.net/publication/400369693\_EDIS\_Diagnosing\_LLM\_Reasoning\_via\_Entropy\_Dynamics](https://www.researchgate.net/publication/400369693_EDIS_Diagnosing_LLM_Reasoning_via_Entropy_Dynamics)  
24. On the Step Length Confounding in LLM Reasoning Data Selection \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2604.06834v1](https://arxiv.org/html/2604.06834v1)  
25. The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs? \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2604.06192v1](https://arxiv.org/html/2604.06192v1)  
26. Semantic Energy: Detecting LLM Hallucination Beyond Entropy \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2508.14496v3](https://arxiv.org/html/2508.14496v3)  
27. Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models, accessed April 14, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/40624/44585](https://ojs.aaai.org/index.php/AAAI/article/view/40624/44585)  
28. Beyond Confidence: The Rhythms of Reasoning in Generative Models \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2602.10816v1](https://arxiv.org/html/2602.10816v1)  
29. BEYOND CONFIDENCE: THE RHYTHMS OF REASON- ING IN GENERATIVE MODELS \- OpenReview, accessed April 14, 2026, [https://openreview.net/pdf/e024b49ce71ffbef35352650495506409ef1cf14.pdf](https://openreview.net/pdf/e024b49ce71ffbef35352650495506409ef1cf14.pdf)  
30. Reasoning-Driven Hallucination in AI Models \- Emergent Mind, accessed April 14, 2026, [https://www.emergentmind.com/topics/reasoning-driven-hallucination](https://www.emergentmind.com/topics/reasoning-driven-hallucination)  
31. Structural Inducements for Hallucination in Large Language Models (V4.1): Cross-Ecosystem Evidence for the False-Correction Loop and the Systemic Suppression of Novel Thought Including Appendices AH: Replicated Failure Modes, Ω-Level Experiment, Identity Slot Collapse, Cross-Ecosystem Validation, and Governance Architecture An Output-Only Case Study from Extended Human-AI Dialogue Structural Preface \- ResearchGate, accessed April 14, 2026, [https://www.researchgate.net/publication/397988600\_Structural\_Inducements\_for\_Hallucination\_in\_Large\_Language\_Models\_V41\_Cross-Ecosystem\_Evidence\_for\_the\_False-Correction\_Loop\_and\_the\_Systemic\_Suppression\_of\_Novel\_Thought\_Including\_Appendices\_A-H\_Repl](https://www.researchgate.net/publication/397988600_Structural_Inducements_for_Hallucination_in_Large_Language_Models_V41_Cross-Ecosystem_Evidence_for_the_False-Correction_Loop_and_the_Systemic_Suppression_of_Novel_Thought_Including_Appendices_A-H_Repl)  
32. Limits of Self-Correction in LLMs: An Information-Theoretic Analysis of Correlated Errors \- Preprints.org, accessed April 14, 2026, [https://www.preprints.org/frontend/manuscript/39eedf90ab65aa6c4e3d4498868bc56a/download\_pub](https://www.preprints.org/frontend/manuscript/39eedf90ab65aa6c4e3d4498868bc56a/download_pub)  
33. Joint Evaluation of Answer and Reasoning Consistency for Hallucination Detection in Large Reasoning Models \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2506.04832v2](https://arxiv.org/html/2506.04832v2)  
34. ICML Poster Position: LLMs Need a Bayesian Meta-Reasoning Framework for More Robust and Generalizable Reasoning \- ICML 2026, accessed April 14, 2026, [https://icml.cc/virtual/2025/poster/40142](https://icml.cc/virtual/2025/poster/40142)  
35. UPROP: INVESTIGATING THE UNCERTAINTY PROPA- GATION OF LLMS IN MULTI-STEP DECISION-MAKING \- OpenReview, accessed April 14, 2026, [https://openreview.net/pdf?id=NnlelrGapm](https://openreview.net/pdf?id=NnlelrGapm)  
36. Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2602.05073v2](https://arxiv.org/html/2602.05073v2)  
37. Detecting hallucinations in large language models using semantic entropy \- OATML, accessed April 14, 2026, [https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting\_hallucinations\_2024.html](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)  
38. LLM Uncertainty Estimation Methods \- Emergent Mind, accessed April 14, 2026, [https://www.emergentmind.com/topics/llm-uncertainty-estimation-methods](https://www.emergentmind.com/topics/llm-uncertainty-estimation-methods)  
39. Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy, accessed April 14, 2026, [https://openreview.net/forum?id=cGLqQfS5wH](https://openreview.net/forum?id=cGLqQfS5wH)  
40. Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models \- AAAI Publications, accessed April 14, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/41155/45116](https://ojs.aaai.org/index.php/AAAI/article/view/41155/45116)  
41. CoT Reasoning Models – Which One Reigns Supreme in 2025? \- Composio, accessed April 14, 2026, [https://composio.dev/content/cot-reasoning-models-which-one-reigns-supreme-in-2025](https://composio.dev/content/cot-reasoning-models-which-one-reigns-supreme-in-2025)  
42. Video Reasoning without Training | OpenReview, accessed April 14, 2026, [https://openreview.net/forum?id=TbLryV1dyT](https://openreview.net/forum?id=TbLryV1dyT)  
43. AI Hallucination Rates & Benchmarks in 2026 with References | Suprmind, accessed April 14, 2026, [https://suprmind.ai/hub/ai-hallucination-rates-and-benchmarks/](https://suprmind.ai/hub/ai-hallucination-rates-and-benchmarks/)  
44. AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents, accessed April 14, 2026, [https://openreview.net/forum?id=qtQkhE9zqF](https://openreview.net/forum?id=qtQkhE9zqF)  
45. HALLUCINOGEN: Benchmarking Hallucination in Implicit Reasoning within Large Vision Language Models \- ACL Anthology, accessed April 14, 2026, [https://aclanthology.org/2025.uncertainlp-main.10.pdf](https://aclanthology.org/2025.uncertainlp-main.10.pdf)  
46. FREAK: A Fine-grained Hallucination Evaluation Benchmark for Advanced MLLMs, accessed April 14, 2026, [https://openreview.net/forum?id=YeagC09j2K](https://openreview.net/forum?id=YeagC09j2K)  
47. Learning to Reason for Hallucination Span Detection \- Apple Machine Learning Research, accessed April 14, 2026, [https://machinelearning.apple.com/research/hallucination-span-detection](https://machinelearning.apple.com/research/hallucination-span-detection)  
48. ToW: Thoughts of Words Improve Reasoning in Large Language Models \- arXiv, accessed April 14, 2026, [https://arxiv.org/html/2410.16235v1](https://arxiv.org/html/2410.16235v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAWCAYAAAC/kK73AAABM0lEQVR4Xu2VMUvCQRjG36ghMDBCkigQ3ZocwiFICZwcWqTNr+Dk4iYufZIIl9ZoqCGcbHJqDCQCIamtQBvqeXrfPx5/UnSIg7gf/JD3OU/e8+44kUAgEAj8FzZj9fYvGWvmq7HcG1X4Cq/hFqzDJzixLAn78Bm+wREs/sz0CJu6hE34BR9gy8a4IGbncM+yXTiAd1bP43RJyzptMSqwJNrIEOacsYZo49yBiH3R3blyMq+wmVuYsHpNdCc+YCH6EqiJLqbtZF5hM2dOnYaPokcnZRkX04FjeGiZd9g4j03EEfyEF3DFsqzoBe2J3o0Ty2fB31zGrk5bHJ5r95gQHpN3eOBkvMBcDBeVkemF9caxaFMu9zL9ZyPy8AXe2Lh3+KDEHxU+NhuxjKzDHfsMBAJ/xDdNKUFJvIrvSAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAXCAYAAADZTWX7AAAA10lEQVR4Xs3RPwuBURQG8Ecog2SQyGCxyYQsRruPYWdQZr6AwWQ1KCMzRqNBWZSUb4BF4Tndf+79BO9Tv97uPafuPe8Fop093ehBS4r7ZZUuLehL/aBmE4NqelM7qNnk6ERXKvkllya9aEWJoGYjxQ919DpDZ7rYDvhHFaDuN4AaxEYuvKGKbpDvMWySxZbWVNZ70lg3DTK+NIk7jaloiiYy2f9UI7hjWvqLHg3NgplCTSonzMzmHP5fntCTqlDviDTtKO96UIN67AM1zGbWll2SlAo3o5IfuRoplnbGG6wAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAaCAYAAABl03YlAAAA1UlEQVR4XmNgGHDAB8S+QBwAxMxocmAgD8THgHgpED8E4l9AbI+iAgiKgDgByuZmgCjUh8tCAUjRciBmQZdABjZA/AmI/dAl0EEOEP8H4j3oEiAA8kkkED8C4n8MEIUYYCcDRLcMA8I0EWQF/EB8BYjFoHwlIH4OxMZwFUCQAcQRSHweID7AgKRIE4jfAjEHTIABEkYgq+HWgVR/hUtDgDQQPwBiRpiAIhA/gXGgAGT1N2QBkOoJQKwA5YM8cQgqhgJAbnjGAInYu0DcCcSsKCpGAUEAAD8DIMJyxI87AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAZCAYAAABuKkPfAAAB0ElEQVR4Xu2WPyhFURzHf2Lwd1MogwxKbCYyKDbZKGWjMBtNFoPVKCWTRElYZLhF2Ew2BpNBEmUw+PP99rv3Ovc8788R975651OfXu93zrnvnu+957wj4vF4PLnUwDUHp3VYJtTBTlhlNxShEfbCWrshghd9h/vyPdEn+Akv4Qxchadh7UiHpc4wvIKB6KRKoUH03l/gNnyEK6JhxjCZPbMQwsneiwZksgnnrNp/wwm3wDHR+wrCWjH64RtctuoMhfWYbnhgFkKiJ26+Plw2W3DEqKWJawicLPtPWXU+RNZjuL6XzAJoEu20aNX5wzuiwWWBawiBaH+OM4muw6WSF07yQ7J74vlwDeFOCofQbNVjuOtuwF3R198Fjh2A4w6OSpEnYvDXIbRZ9Rimcw0X7IYyILUQ+uArHLQbyoDUQoh2zrzrJUNcQwikcAg/XoN7APcCdnA9kZF6eCg6vlQfYA8Hl4BrCOui/e1zDZc66znwPDAh+q/Ag0R1srksmBe9+QvYbrURtvHkOxR+Z1DH8BZ2hDV+8nvibBRthPYTirQPGlkQvQG2gSTfiGd4Jsm1zrDO4Q2cDT9PYKvRpyLgW80NfxJ2ye+Wusfj8Xg8lcYXPlqKYe2KKfIAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABD0lEQVR4Xu2SsUoDQRRFrxALixAEwcIqZQqjYKXgB9hYBItA/iJfkG9IUkiaQJo0doqIgrWdZfqQ1g9IEfRe3gzOzG42JKTcAweWeW/vzJtdoGRb7ugXnTubcRm39ImOAi+jDsc5faCP9JdOaSWon9EO/aQrOqAnQT3imL7SG1hYL6oCB3QMCyykAQtSoIJm9DSo+43UV4h26rvnBSys/V/GFf2ABRaiEH/sISzojR65tXCjtYRjCb2sEIUpdKexPBpLQRpTY73QatSRQziWRxetC1dY3kYZdOx3epEWYL+Agp5pKy5lSe8npA4b7Qcb7ueQftMJrSU1zzU2jKWGJezo3m7UYegL3qeLJSX75A/RdzUL5u12HAAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAZCAYAAACYY8ZHAAACMElEQVR4Xu2WP0hXURTHv6JlYn80hQYDMUKJwAT/gCANQZBIDjk0NDq4pKhDQdDs1KI0hYqDuAhO4qCDIdjQqjhIVODS0BI1BdX3y3mXdz0+fz6334vfFz7w3jn3nXvvOefd94CKKqrov9dVcp/UeUdR1EK+JOi6kBogv8k6ueR8hdFL8pc8944iaBS2eM/leFBR9IcskyrvKBPVeEOWlP0xbywDqVN+ku/e4VVPfpFu7yihDrJP7nnHOZUnjpK7441et8hH0ugdJfQMNnmzd5xTZ8VRG62SWe/wGiaLSN+Hm6Q6dWdKQVeQs1dL6Kw4N8gn2GZLSoH6YZt4SuYin+wbpJd8Rlp2BZYt6AXZJNdgcVT+K6SNTJCHZIRMkx+wOHdwMo6Xjn51SRdsLXpWMU6oh3wg72FZ0UKCHsPeF016G2m1fAt8g1VUUlZD+TW50MQ6tvVMHyxO3lY6IFOw3yF9lPUOZ0pla/BGWGvtwk4vZTtswh/HR7Cxkhb1JPKFxXhpoz5OrNBKGrOH48nNrXYyCcvADNmGZVMHQDiOVZ3rZAtphpStu6ST1MIWc5j4YqlN4jheivMV1nZK0CBsrqyxp2qIrMHKPo80E9qINrVAWhObNqH+fQer2FvyIPFpodvJdSz9p8VxVJE3sA+v5l5K7iXNqcq9JhcSW27pAWXSqwnHTzBdq43CBPFvi2wXo/sgPePjSI8SVNl4wVljy1KqxivYqVZYqe/HvbGiIugf8kRaBJihEKEAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAZCAYAAACIA4ibAAACg0lEQVR4Xu2XO2hUQRSGf/GBwXcivlAQQUENpAgiYiEIgiI2PkAQtBAULAUVqyRFChuLpBB8FBaiYqFgpxaLhYUKVmKhKdJoIRZaCAqS/D/nLpk57sw0ahaZDz5298zc3TvnzpyZBSqVSuX/YzHtpwt9Q4G5dC3dRee4Ns9SusEHu41FdIx+o/fpF3qF9oSdEuyh7+kbep2+pTuiHjMo0ZP0tm/oJvRUf9BRF1eCFM8xj36im4LYVlhC7zbtYjldBZtJU/iLCVniAx3QTeTQwHWTJ1z8bBPPocG/oiuC2ErYLPGJavPHE6JpfIF+h335c7ot6hEz7AOOFux7Drm4Piuu5ZRCfVqw2tNG7xXTtQeCeJtiQlSEdtJ79CP9TLdHPWKuwdbrMboe9oSVnHFYwfIM+4BDazqXED3xFJeQT4hmmSebkGX0Fz3iGzIc9oEGJXYv7Afbfo16dKaUEO0eKUoJUbsnmxBd8JDO9w0ZOs2CkHWwwWh9l+qH6JqEhBeGbg76pNA+PkJf04PID/y0Dzi6JiH6Id1Mas9OoSWjmqHqrn3/J32JdGEd8gFHC/mEhIP1lIqq/05RTEinrSnHU8RJ1K5zEZYkFdyQXrrfxTw3YTfpC+D5Jp5jALbFhoV3NZ2AnUW0bD3JhIh9sKcc1hDtODkGfcBxFbaU7sCWUwk90SewQWxsYnrV58fNZ6HB6yT7iC4I4toULmPmyK73imlsHp1UlZAHyMy8d/QDvQVbBrvj5n+Ctu8XsPs407w+o2uCPhqATp9+Jp2Dzc4b9Gjz/iTi/zSqJWGdDP0NXdgHW0I64s4WKsyafcfpFpT/pIXo/lUvTiF/kKtUKpVKpTL7TAMTR5tW7hBsRwAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFUAAAAZCAYAAABAb2JNAAADUElEQVR4Xu2YTahNURTHl1DkK5GPfL0MSJQkykcpIRIDyUcGyoAyVC8xUjIwIDFQIgyEMlAykcENGaBMmDEgUiQlFPLx/719trfPvmefe1+d+3o951f/3jtr73fvOf+z1jrrPLOampqamprWjJYWSCPihRYMl2ZKy+OFAiZJq6Tp0tBobVAxSjotfZauSx+l49LIcFOC7eb2N6QL0j2pK1gPWSK9lS5LT6UH+eXBwzLpu3QsimMy8TLGSY+l8UFsjfRbOhLElkqfpPPZMRXRkP74DVVC+o+JgwWwb0gcrAjM4+J2RfF9WbwMDLwhDQti88xlrjebbL9lzmj2e1ZIJ4PjJJPN3YV2oKdw0uiHdCq/nGODtDEOVkTD3DlsiuIcE6c1pDhorpRDpkqvzLWShdJi6av0xtw1t81Yc3e8IX2RrpkrjRR8+DNzjZ2TWC/dkXZac/MmC85IK6N4VWBAmakTo3gIhqZM9ZnpM57MXSS9N2c4SYRvTWDcXemRNCGL+X5BCaSgfxXdNUqlW/pmvVn80Ir3VkUrUzEpRZmp/jNZ53dMJnk8tBviTW2N9GdhRxDjzj43l/Ipplm+D8Uwosw3107ahQzngtqVH5v6y9S4/Km8n9LsINYD5r2LFugjpHeZqYAJl6Qn5hr23NxqHtpEqxmQE37dpl5ab4/utKlUJb/ft/xD2ffa8OHVA07ftvyw7NMaw1OQiWelm9I5c/3mlyV6jLm+1PTlFdGwclPLHrxlDyoMwzjfUxuW/yxvavy9PZv5YA8lfbUgHrPF3MMshExjlttj+aGbDGWg7ksr6AvMjpwvFx9yIIuXwVQSJ5WvVJKKVujN88ceXgR4drD/H/RFvvSo9TZbLp6n+iy/KQHZnJoOaAuU5hVr3RaqgOxh8qAldGUxfnLMfOnBQK73RBADKoy3Ks9F6YO5Z4Jns7l9h7JjKpV9JFEOypGTITPpF7zeMValSnggQ5UwZbyQ9mY/mWqmBHsoa14r1wYx4HWWjKN38sBmZFqd2+GSbr+5fYfN+YWh3KgcLFIi/AHjVKfKs7+gQihVjJljBaNOCTOkreYqkCxMgU+7pXVW8FJBD6GXdGog/y/x/zToy92sacE2c/2hpqampqZmwPEXu9bIrhwEJ3gAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAu0lEQVR4XmNgGAW0BjxALIYuiAzUgPg4ED8C4hNAHA7EjCgqgMAFiJ8DcT4QM0PFfgKxJVwFFDwD4ulAzIIk9h+IJyHxwcZ/BWJjZEEGiMKFyAJ+QNyKLAAE3AwQhenIgiBFIMXIQBOInwCxIkwAFBQHgFgHJgAF5UCcgSygxADxbQySGCgE3gExK5IY2EqQW0DhtgWKQSEQiawIZu1bZEFsAGbtATRxDODJALEWPWgwQCIDxER9dImhBgBnnh/2TCJJ+QAAAABJRU5ErkJggg==>