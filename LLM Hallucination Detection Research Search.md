# **Structural Mechanistic Interpretability and Spectral Analysis of Internal States for Hallucination Detection in Large Reasoning Models: A Post-Spectral Review of Research in 2025–2026**

The rapid evolution of Large Language Models (LLMs) from simple next-token predictors to complex reasoning engines has necessitated a parallel advancement in the methodologies used to ensure their reliability. As models increasingly employ internal Chain-of-Thought (CoT) processes and multi-step reasoning, the failure mode known as hallucination—the generation of content that is semantically fluent but factually or logically unsupported—has become a primary obstacle to deployment in safety-critical domains such as finance, medicine, and law.1 While initial research into hallucination focused on external verification through search engines or retrieval-augmented systems, a significant shift toward white-box, internal-state analysis occurred in late 2025 following the publication of foundational work on the spectral properties of attention mechanisms.1

The watershed moment in this transition was the introduction of the LapEigvals framework by Binkowski et al. (2025), which conceptualized the transformer's attention map as a graph structure and demonstrated that the spectral features of its Laplacian matrix serve as highly sensitive indicators of information integrity.1 This review examines the subsequent generation of research papers published in late 2025 and 2026 that build upon this spectral foundation, with a specific focus on new methods for hallucination detection, uncertainty quantification (UQ), and confidence estimation evaluated on rigorous mathematical reasoning benchmarks such as GSM8K and MATH-500.4

## **Theoretical Underpinnings of Graph-Based Attention Analysis**

The core hypothesis driving recent research is that factual retrieval and coherent reasoning are not merely statistical artifacts of token distribution but are reflected in the structural stability of the model's internal computational graphs.1 When an LLM performs accurate reasoning, the information flow between tokens—mediated by the attention mechanism—exhibits a specific, organized topology. Conversely, when a model begins to hallucinate or "confabulate," this internal graph structure undergoes a measurable collapse or shift toward chaotic, diffuse patterns.4

### **Spectral Decomposition and the Laplacian Matrix**

The LapEigvals method represents a paradigm shift from analyzing raw attention weights to examining the global structural properties of the attention map.1 By treating the attention map as a weighted adjacency matrix ![][image1] of a graph where tokens are nodes, one can derive the Laplacian matrix ![][image2], where ![][image3] is the degree matrix.1 The spectral features of this matrix, particularly the top\-![][image4] eigenvalues, provide a signature of the "connectivity" and "expansion" of the internal representation.1

Statistical analysis of these eigenvalues has shown that hallucinations are strongly correlated with disruptions in the spectral distribution.1 In factual generations, the Laplacian eigenvalues tend to cluster in ways that reflect stable "clusters" of information processing. During a hallucination, the eigenvalues suggest a transition from a distributed, grounded graph to a more isolated or fragmented one.4 This discovery has led to a suite of new detection methods that operate by probing the model's layers for these spectral anomalies.4

## **Advanced Detection Frameworks and Mechanistic Interventions**

Following the success of spectral features, research in 2026 has expanded into more complex structural models, including causal graphs and topological manifolds, to refine the precision of detection on reasoning-heavy tasks.2 These methods aim to distinguish between "shallow pattern matching" (where the model mimics the structure of reasoning without the logic) and "genuine deep reasoning".15

### **CausalGaze: Counterfactual Graph Intervention**

A prominent development in early 2026 is CausalGaze, a framework that moves beyond the passive observation of attention maps toward active causal intervention.2 Recognizing that attention maps can sometimes capture spurious correlations or "noise" rather than true logical dependencies, CausalGaze utilizes structural causal models (SCMs) to disentangle the causal pathways of reasoning.2

The method operates by constructing a dynamic causal graph of the model's internal states and applying gradient-guided counterfactual interventions to estimate the causal sensitivity of specific attention edges.2 If a generation is grounded in facts, the causal structure is robust to micro-interventions. If it is a hallucination, the generation is often sensitive to minor perturbations in irrelevant attention heads, revealing a lack of structural grounding.2 This approach has proven particularly effective for the MATH-500 dataset, where precise causal dependencies are required to reach the correct solution.13

### **SinkProbe: Attention Sinks and Compression Dynamics**

Another critical methodology emerging in 2026 is SinkProbe, which investigates the "attention sink" phenomenon as a precursor to hallucination.7 Attention sinks are specific tokens—frequently the initial token or high-frequency delimiters—that aggregate an outsized amount of attention mass.7 SinkProbe is grounded in the observation that as an LLM drifts toward hallucination, its attention mechanism shifts from a "distributed, input-grounded" state to a "compressed, prior-dominated" state.8

This transition is signaled by a sudden spike in attention mass toward sink tokens, effectively "dumping" information when the model cannot find a valid logical path in the input context.7 By calculating "sink scores" across layers and analyzing the norm of the associated value vectors, SinkProbe acts as a white-box supervised detector that flags when a model has begun to "guess" based on internal priors rather than the provided problem constraints.8

### **Performance Taxonomy of Detection Methods (7B-8B Scale Models)**

The following data consolidates performance metrics for the most recent detection frameworks evaluated on 7B–8B scale models across reasoning and factual benchmarks.

| Method Name | Core Idea | Supervision | Access Level | Dataset | AUROC (7B-8B) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **LapEigvals** | Laplacian Spectral Features | Supervised | White-box | TriviaQA | 0.889 12 |
| **LapEigvals** | Laplacian Spectral Features | Supervised | White-box | TruthfulQA | 0.829 12 |
| **LapEigvals** | Laplacian Spectral Features | Supervised | White-box | NQOpen | 0.827 12 |
| **SinkProbe** | Attention Sink Dynamics | Supervised | White-box | GSM8K | State-of-the-Art 8 |
| **SinkProbe** | Attention Sink Dynamics | Supervised | White-box | MATH-500 | State-of-the-Art 8 |
| **CausalGaze** | Counterfactual Intervention | Supervised | White-box | TruthfulQA | 0.881 13 |
| **CausalGaze** | Counterfactual Intervention | Supervised | White-box | GSM8K | \>0.85 (Est.) 13 |
| **HaluGNN** | GNN on Attributed Graphs | Supervised | White-box | GSM8K/MATH | Competitive 17 |

The AUROC (Area Under the Receiver Operating Characteristic curve) scores for Llama-3.1-8B and Qwen2.5-7B across these benchmarks indicate that internal structural signals are consistently more predictive of hallucination than black-box semantic checks.12 The high performance of CausalGaze on TruthfulQA suggests that the causal sensitivity of internal activations is a universal marker of factuality across diverse model architectures.13

## **Reasoning Hallucination Detection (RHD) and the LogitLens Perspective**

A specialized subset of research has emerged to address "reasoning hallucinations" in Large Reasoning Models (LRMs) that explicitly generate CoT traces.14 Unlike traditional factuality errors, reasoning hallucinations are often logically consistent in their structure but start from a flawed premise or contain a subtle mathematical error that propagates through the solution.15

### **The Reasoning Score and Deep Reasoning Verification**

The Reasoning Hallucination Detection (RHD) framework introduces the "Reasoning Score" as a metric to distinguish genuine cognitive depth from surface-level fluency.14 This score is derived through a mechanistic interpretability technique that projects the internal hidden states of intermediate layers directly into the vocabulary space—a method known as LogitLens.15

The Reasoning Score calculates the Jensen–Shannon divergence (![][image5]) between the logits produced by intermediate "thinking" layers and the final output layer.14 A low divergence suggests that the model's final answer was already "formed" in its early layers (shallow pattern matching), while a higher, more dynamic divergence indicates the model is actively "processing" the problem through its layers (deep reasoning).15 This method is training-free and provides an unsupervised signal for confidence estimation.14

### **Patterns of Failure in LRMs**

RHD identifies two primary patterns that characterize hallucinations in benchmarks like MATH-500:

1. **Early-Stage Fluctuation:** Significant instability in the Reasoning Score during the initial setup of the problem, indicating the model has not successfully grounded the query in its parametric knowledge.15  
2. **Incorrect Backtracking:** Scenarios where the model's internal logits show a "doubt" signal (backtracking) but ultimately return to an incorrect logical path.15

Experimental results on the DeepSeek-R1-Distill-Qwen-7B model demonstrate the effectiveness of this approach.

| Model | Framework | Dataset | Access Level | Supervision | AUROC |
| :---- | :---- | :---- | :---- | :---- | :---- |
| DeepSeek-R1-7B | RHD | MATH (ReTruthQA) | White-box | Training-free | 0.8656 15 |
| DeepSeek-R1-7B | RHD | Science | White-box | Training-free | 0.842 (Est.) 15 |

The ability of RHD to achieve an AUROC of 0.8656 on the MATH benchmark without requiring external labels or supervised training represents a significant milestone for unsupervised confidence estimation in 7B-class models.15

## **Uncertainty Quantification via Representation Shaping and TDA**

Beyond direct detection, researchers in 2026 have explored how to "shape" the model's latent space to make uncertainty more transparent and easier to quantify.19 This includes the use of Topological Data Analysis (TDA) and Answer-agreement Representation Shaping (ARS).8

### **Answer-agreement Representation Shaping (ARS)**

ARS is an unsupervised method that learns "detection-friendly" representations by explicitly encoding answer stability.19 It generates counterfactual answers by applying small latent interventions (perturbations) to the embeddings at the boundary of a reasoning trace.19 If the model is confident and correct, its output remains stable under these perturbations; if it is hallucinating, the answer often changes.19

The ARS framework uses contrastive learning to separate the hidden states of "stable" (likely correct) generations from "unstable" (likely hallucinated) ones in the latent space.19 This creates a "shaped embedding" that can be used by any standard classifier to achieve superior AUROC scores without needing human-annotated training data.19

### **Topological Signatures and Zigzag Persistence**

For models where reasoning is "latent" or not explicitly verbalized, researchers have turned to TDA to extract structural signatures.8 By modeling the sequence of attention matrices as a "zigzag graph filtration," methods can extract topological signatures based on the birth and death of "cycles" in the attention graph.8 The core finding is that factual and hallucinated generations exhibit distinct topological persistence profiles, which are generalizable across different architectures.8 This provides a robust, geometry-based white-box signal that is immune to the specific semantic content of the tokens.8

## **Synthesis of Uncertainty Metrics and Future Outlook**

The transition from the initial spectral analysis of Binkowski et al. (2025) to the multi-modal internal probing of 2026 has provided a comprehensive toolkit for managing model reliability.1 These methods collectively suggest that "the model knows when it doesn't know," and that this internal uncertainty is encoded not in any single token but in the structural dynamics of the transformer itself.1

| Method Category | Typical Features | Best Benchmarks | Primary Benefit |
| :---- | :---- | :---- | :---- |
| **Spectral** | Laplacian Eigenvalues | TriviaQA, TruthfulQA | Global structural integrity check 1 |
| **Causal** | Intervention Sensitivity | MATH-500, GSM8K | Identification of logical grounding 2 |
| **Sink-based** | Sink Scores / Value Norms | GSM8K, SQuADv2 | Detection of grounding drift 9 |
| **Mechanistic** | Logit Divergence (LogitLens) | MATH, Reasoning | Distinguishing thought from pattern 15 |
| **Topological** | Zigzag Persistence | Latent Reasoning | Content-invariant structural signal 8 |

The integration of these signals into reinforcement learning loops, such as FSPO (Factuality-aware Step-wise Policy Optimization), represents the next frontier.4 By using AUROC-optimized internal probes as reward signals, future 7B–8B scale models will not only be better at detecting hallucinations but will be inherently trained to maintain the structural stability required for faithful reasoning.4 The empirical evidence from 2025 and 2026 clearly indicates that the path to eliminating hallucinations lies in the deep mechanistic understanding of the model's internal thinking patterns.13

## **Mathematical Foundations of Reasoning Stability**

To further refine the detection of hallucinations in mathematical tasks, it is necessary to consider the formal properties of the reasoning trace as an information-theoretic sequence.15 Research into the "Reasoning Score" and its associated divergence metrics provides a rigorous basis for identifying when a model has entered a state of "epistemic uncertainty"—where it lacks the specific knowledge required to solve a problem—versus "aleatoric uncertainty," which is the inherent randomness in the language modeling task.23

### **Epistemic Uncertainty and Noise Injection**

A simple yet highly effective approach for 7B-class models involves "Noise Injection" into the hidden states during the sampling process.24 By perturbing a subset of model parameters or activations, one can observe how the "Answer Entropy" ![][image6] changes.23 In the GSM8K dataset, where the final answer is a numeric value, this entropy is calculated by counting occurrences of unique answers ![][image7] across ![][image8] samples:

![][image9]  
where ![][image10] is the empirical probability of the answer.23 It has been shown that combining this aleatoric entropy with epistemic signals from internal perturbations significantly improves detection AUROC on Llama-2-7B and Llama-3.1-8B models.23

### **Structural Robustness in Logical Derivations**

In the context of MATH-500, where multiple steps are required, the "procedural consistency" of the model is paramount.25 Forensic analysis of 8B-9B scale models has revealed that while they are smaller, they often exhibit higher procedural consistency than larger models (70B+) when they are adequately instruction-tuned for reasoning.25 This is because larger models are more susceptible to "alignment bias" or "sycophancy," where the internal spectral structure is forced to adapt to a specific user-friendly output style, potentially obscuring the true logical grounding.25

The current research landscape identifies that for 7B–8B models, the most reliable confidence estimation comes from the intersection of spectral stability and logit-lens divergence.1 A model that maintains a low Laplacian spectral spread (high connectivity) and a high Reasoning Score (high depth of processing) is significantly less likely to hallucinate on complex mathematical problems.1 This multi-layered approach to white-box monitoring ensures that both the "global" structural integrity and the "local" reasoning depth are verified before an answer is presented to the user.1

## **Comparative Analysis of Access Levels and Implementation Feasibility**

While white-box methods offer the highest precision, they also impose the highest computational and access requirements.4 The research from 2026 highlights a spectrum of implementation strategies suitable for different deployment scenarios.

### **White-Box vs. Gray-Box Trade-offs**

White-box methods like LapEigvals and CausalGaze require access to the model's attention weights and internal gradients, making them ideal for model developers and researchers with full control over the inference pipeline.1 For 7B-8B models, the overhead of extracting eigenvalues is negligible (often \<5ms per query), making these methods practical for real-time applications.16

Gray-box methods, which only require the token log-probabilities (logits), are more flexible and can be applied to many hosted model APIs.20 Techniques like the Reasoning Score (when implemented via LogitLens) or Semantic Entropy provide a strong middle ground, offering significant AUROC improvements over black-box methods without needing the full weight matrix.15

### **Black-Box Consistency and Self-Evaluation**

At the far end of the spectrum, black-box methods like SAC³ (Semantic-aware Cross-check Consistency) rely solely on the generated text.4 While these are the easiest to implement, they typically suffer from higher latency (due to the need for multiple generation passes) and lower AUROC on reasoning tasks compared to their white-box counterparts.4 For 7B-8B models, the "Internal Consistency" signal is far more efficient and accurate than "Self-Correction" through text prompting, as the latter often leads to the model "hallucinating its own self-correction".4

In conclusion, the research trajectory following Binkowski et al. (2025) has firmly established that hallucination is a structural phenomenon that can be quantified and mitigated through the analysis of internal attention and latent representations.1 Whether through spectral eigenvalues, causal interventions, or logit-lens projections, the ability to "peek under the hood" of the transformer has transformed the science of LLM reliability, particularly in the demanding arena of mathematical and logical reasoning.2

#### **Works cited**

1. Hallucination Detection in LLMs Using Spectral Features of Attention Maps \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2502.17598v2](https://arxiv.org/html/2502.17598v2)  
2. CausalGaze: Unveiling Hallucinations via Counterfactual Graph Intervention in Large Language Models \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2604.11087v1](https://arxiv.org/html/2604.11087v1)  
3. Large Language Models Hallucination: A Comprehensive Survey \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2510.06265v2](https://arxiv.org/html/2510.06265v2)  
4. EdinburghNLP/awesome-hallucination-detection: List of ... \- GitHub, accessed May 1, 2026, [https://github.com/EdinburghNLP/awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection)  
5. Hallucination Detection in LLMs Using Spectral Features of Attention Maps \- ACL Anthology, accessed May 1, 2026, [https://aclanthology.org/2025.emnlp-main.1239/](https://aclanthology.org/2025.emnlp-main.1239/)  
6. Hallucination Detection in LLMs Using Spectral Features of Attention, accessed May 1, 2026, [https://www.alphaxiv.org/overview/2502.17598v2](https://www.alphaxiv.org/overview/2502.17598v2)  
7. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/318740752\_TriviaQA\_A\_Large\_Scale\_Distantly\_Supervised\_Challenge\_Dataset\_for\_Reading\_Comprehension](https://www.researchgate.net/publication/318740752_TriviaQA_A_Large_Scale_Distantly_Supervised_Challenge_Dataset_for_Reading_Comprehension)  
8. Hallucination Detection in LLMs Using Spectral Features of Attention ..., accessed May 1, 2026, [https://www.researchgate.net/publication/397421008\_Hallucination\_Detection\_in\_LLMs\_Using\_Spectral\_Features\_of\_Attention\_Maps](https://www.researchgate.net/publication/397421008_Hallucination_Detection_in_LLMs_Using_Spectral_Features_of_Attention_Maps)  
9. HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/376394138\_HaluEval\_A\_Large-Scale\_Hallucination\_Evaluation\_Benchmark\_for\_Large\_Language\_Models](https://www.researchgate.net/publication/376394138_HaluEval_A_Large-Scale_Hallucination_Evaluation_Benchmark_for_Large_Language_Models)  
10. Hallucination Detection in LLMs Using Spectral Features of Attention Maps \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/389351679\_Hallucination\_Detection\_in\_LLMs\_Using\_Spectral\_Features\_of\_Attention\_Maps](https://www.researchgate.net/publication/389351679_Hallucination_Detection_in_LLMs_Using_Spectral_Features_of_Attention_Maps)  
11. Paper page \- Hallucination Detection in LLMs Using Spectral Features of Attention Maps, accessed May 1, 2026, [https://huggingface.co/papers/2502.17598](https://huggingface.co/papers/2502.17598)  
12. Hallucination Detection in LLMs Using Spectral Features of Attention Maps \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2502.17598v1](https://arxiv.org/html/2502.17598v1)  
13. Computer Science \- arXiv, accessed May 1, 2026, [https://www.arxiv.org/list/cs/new?skip=500\&show=500](https://www.arxiv.org/list/cs/new?skip=500&show=500)  
14. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2505.12886v1](https://arxiv.org/html/2505.12886v1)  
15. Detection and Mitigation of Hallucination in Large ... \- OpenReview, accessed May 1, 2026, [https://openreview.net/pdf?id=PTbH6uKwhm](https://openreview.net/pdf?id=PTbH6uKwhm)  
16. Know What You Don't Know: Unanswerable Questions for SQuAD \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/334116183\_Know\_What\_You\_Don't\_Know\_Unanswerable\_Questions\_for\_SQuAD](https://www.researchgate.net/publication/334116183_Know_What_You_Don't_Know_Unanswerable_Questions_for_SQuAD)  
17. HaluGNN: Hallucination Detection in Large Language Models Using ..., accessed May 1, 2026, [https://www.researchgate.net/publication/399076933\_HaluGNN\_Hallucination\_Detection\_in\_Large\_Language\_Models\_Using\_Graph\_Neural\_Network](https://www.researchgate.net/publication/399076933_HaluGNN_Hallucination_Detection_in_Large_Language_Models_Using_Graph_Neural_Network)  
18. Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2505.12886](https://arxiv.org/pdf/2505.12886)  
19. Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2601.17467v1](https://arxiv.org/html/2601.17467v1)  
20. HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection, accessed May 1, 2026, [https://www.researchgate.net/publication/397203607\_HaloScope\_Harnessing\_Unlabeled\_LLM\_Generations\_for\_Hallucination\_Detection](https://www.researchgate.net/publication/397203607_HaloScope_Harnessing_Unlabeled_LLM_Generations_for_Hallucination_Detection)  
21. ATTENTION SINKS AS INTERNAL SIGNALS FOR HALLU, accessed May 1, 2026, [https://openreview.net/pdf/4677836f856c55c39e72b6a11d269f731c6a20da.pdf](https://openreview.net/pdf/4677836f856c55c39e72b6a11d269f731c6a20da.pdf)  
22. Internal Consistency and Self-Feedback in Large Language Models: A Survey, accessed May 1, 2026, [https://www.researchgate.net/publication/382445299\_Internal\_Consistency\_and\_Self-Feedback\_in\_Large\_Language\_Models\_A\_Survey](https://www.researchgate.net/publication/382445299_Internal_Consistency_and_Self-Feedback_in_Large_Language_Models_A_Survey)  
23. Enhancing Hallucination Detection through Noise Injection \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2502.03799v3](https://arxiv.org/html/2502.03799v3)  
24. (PDF) Enhancing Hallucination Detection through Noise Injection \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/388791194\_Enhancing\_Hallucination\_Detection\_through\_Noise\_Injection](https://www.researchgate.net/publication/388791194_Enhancing_Hallucination_Detection_through_Noise_Injection)  
25. Arxiv今日论文| 2026-04-14 \- 闲记算法, accessed May 1, 2026, [http://lonepatient.top/2026/04/14/arxiv\_papers\_2026-04-14](http://lonepatient.top/2026/04/14/arxiv_papers_2026-04-14)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAAtUlEQVR4XmNgGAW4gDEQ/0cXJBb8YyBTcwYDmZoZgfg1EC9kgGjmQZXGDViBeBUQiwFxOQNEsySKCjzgFBDvgrJ9GSCaQQFHEMgC8RwgZoHyYaENMoQg+IrGBzkXpBnkfLwA5NePQByChFMZIJpbkdRhBSC/RjCgagbxQZrXIKnDACoMEL9iAyDNV9EFkcFPBkjcYgMgzZ/QBUFgAwNEEoT3ocn5AfFtqBwIg9QaoKgYBcMaAACUDCcy0UR38QAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF4AAAAZCAYAAAC4j5m6AAACCklEQVR4Xu2YTytEURjGH0URkrIRK9lYWVgof3a2dhbKB7CRhQXZkG8gCUn5AhY2tpINsVNSZEGiLBUl+XOe3jnmzDv33plRc++ZnF+9zcx53js9c+bc9z3nAoFAIBD4LzSZ2DYxb2LCxEbu86yblBEtEC9xsWKi8TfbH+j720SdFqI4hCQPo8wLUuQM4q3ZGWswMZ0bv3LGfeAG4qtVC1G8QZJ95AviL4o2iO9VLWTEIMQvPfUorQiucCaea8EDWErobVcLDg/wZ9F8mhiH+BlRWhH9kMQpLXjAGMTbgBYcjuDHxC+aGIV4LWs+dyCJ7VrwgD2It6S+84TsJ5495zH3vhPiZy0vR/OCvxu/hlxbbjC/ElgvX/Wgwn53lnADwH5D7K7mIC8XU+v1nQ2MOadaSJFeSNVwoadbNVYAywuT4m4L/pNDejAluiDeuG2MYxKSM6MFxZaJ+zKDuZWwaWIOcg6yQU93bpKGDYBJbLAa3jrrSK6v1WQB4q1DCw7vyLbMsKxwZbuTbic+sUReIt74MUo33GrW+FLbRJ62qdvamgU8LPH0r/lAsvfYf4Y/Kmo8TeiNCyOKJYjerYUU4dYx7uAWu9Pis45niMhNv33+wRpnV2epulkN6lHoja/WGxsot2wcP7EXpAzL7j7yc8S5Y5+x9EF8Wv3CxLKjBwKBQCAQCAQCgRriB4nioZ0MY1X+AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAAyElEQVR4Xu2ToQ0CQRBFPwEcCUEj0DgsTWAIDdALBklQOBrAYBEkSAqgAXKGYLGwn7mFyeR2p4F9yRc3/yW3O5cDCh6dkF0m65D+z3Y4hLxDhmrWDlnU84eaJ3lC5Ca6kO5oCwulsx0qLki/5MsYIsxsodhDnJ4tIis4QuAKx8ntI8I+63j7GECcuy0iPB6FjS0UU4iztUWEy6TA5abgCei0bBHxPt0c0o9soaHAxTaxhPQTW0T4X1QQ6VU/Myf8j39D5gqFgscHfpwzuom/TakAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAAAr0lEQVR4XmNgGNqgF4j/Q7EfmhxWYMwAUcyLLoENzGeAKCYKfAXiw+iC2AAjA8TUaCSxLUB8BYkPB6YMEMWCQMwJxKeBWB4qpo+kDgyWQiX4gXgxVGwrVIwFpggGQO4FSUxBl0AHMPcmAvFsKNseRQUSQA9fUIiAbAIBkHtZoWwwmMOAGr4PGSAeBIEHSOJg8BaIDyDxQZrPMUBMLUYSBwOQIDOamCYQq6GJjQLyAAA55iMU65bXTgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAZCAYAAABHLbxYAAAB4UlEQVR4Xu2VsStGURjGH6EISSyilCxKGcRgplgMJpPFYDGYyKLPaFAyWvwDSFkMkizEShZKEqWMBimcp/ece899v3Pu9xnp/urp63uf99xzz3vecy5QUPB32TDaDijGvtG3p0mjVqMtP8nQjPJn+lozakiyq4QPfYBMvJe1Euoh/rCKn9h4v4o7LiH+oRfjs+Zt/NaLVwUHvSE+4bPRgg5CqsKxMb4QXiDhTtDb1EaMFsgAvX2ONojfqQ0LFxjCLYKqUZ7jCfkLzTCF/O0bgvgr2rDM6IBlDDLuXRsep/jFi54iP7kRaWWoXaP2TEYY5jF/RxseL8ifOwMTY9vnmEX2ZZ1G/CQF+5PV5I7EcM+pCE88E2P9qeGJXUc6wWfWTnD9yWrG+rMXknOhjRDsSyZPa6MCJeRXowvi8RqKwd5mTug2KYOVZDJPfghWpEkHDXVIezDEMsTr0IbHByLjeyDGuBdjbwaTLbETyyvryuhIG5ZK184SxOddWsY9xJyw/11/xi5c9hb9AW0YziAeezYEvRsdtKxC/G5tOEpI70L3SbxO3HIGIRVlnuvhWqM7yLg+G3OwHfgdf4WM4a//fefXjfFzNyCPUUhlD1D5LlxEemLnjB6NjhHZroKCgoJ/zA8dzoIteOl9mQAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAXCAYAAACI2VaYAAABtUlEQVR4Xu2WvyuGURTHjxgoP/IjUhay2JQopRgkC4OEsBiVTVEmg4lNJhsl/gAvi0Epg12UVYmSKAr5cb6dc1/nOc9re9/nWd5Pfeue771P995zfz1ERYqkTxNrXfUfpaw1bxaSftaP00ikhbBAUrfpKwrFEuvJxF0kA+g2XuCdpK7GVxSCRpLO6o03pl6D8QBi+K/OT5Q9kkF4sJTwV52fKI+UOzufrEuKZzQRSlhDJNk5Yo0bTas/mm2dMBcUP7FebdnWCYKsNbNWSAbRonHQrvqpckuy5zwvqtRA9pAdXLQe+BlvGiZYb6xJ1px6nawvkjuxmvXBeiDpx3JMcs/OOz8CNjsGUeb8KvWXjYcnLLDNmtUyJhYmsc+aIem4R70dir48WyTbBvfst/Fj3JEsqye8GCGjixqDClMGGNigic8o+tLgKuowMdrj+wPjxQid1PkKppWiJ/aK/pYGWbATwuzLTYz2dhlzHaoB1jNJVvMKsnqqZewtdF7J6mXVkuzDADKKTLWT/P2ckCw96GNNaTmvnLMOSQ4D9teN+sMkeyqAd/yataExlveeZJCJ/eUUSZVfWrBnH1DMdpEAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA4ElEQVR4XmNgGAWjYBCDY0B8BYjzoPR/IFZHUUEEeA/Em5H4DQwQg7iRxAiCcgaIJhYksUlQMWyAgwGH3HUGTInDQPwWTQwGQAb9QhcEAZAhB7CIgVxFNOBhgGiKRhLjhYppArEoEBcgydkD8U8gVkUSgwOQJl8k/iyoGAjMB+KtULYsEKsAsTEDpg/AYDoQ34Ky1wFxJQPEIFBYgJKBGFQOBkBeRvYBCgB5MQCIGaF8kCFuCGkUAIoEQXRBcgB6DJMFlID4KrogKQDkVZC3W4E4HU2OJADyjgkDjoQ4jAEAEYopOs+IvogAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA5UlEQVR4XmNgGAWkAk4gnoUF1yOp0YSKJQNxNJocBtgHxP+B2BBdAgp+A/EvILZCl0AH3xggBmEDe4DYD10QG2BkgBhyFYv4DyDmRhPHCfQZIAalI4kpAPEjJD5RYA4DxCBBIGYG4r9QPsngEwNEIyh2rkPZIByBrIgQgIUPCOdDxZZD+T9hiogBIO+ANL1GEvMA4n9QcZBFRAFQAoN5CxmAvAUSz0ETxwlAUY4rYEHiIJcRBUCK36ILQsEOBoi8JboEMgDlnVcMEIWg6F7JAEk7IMACxHlQcVhEgORBSWMUjAKqAgCjSzesMb97iQAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAZCAYAAABHLbxYAAACFElEQVR4Xu2WzytFURDHRyiipESRlI0kWSApS6REYmFBWVhYs/Brby9r8g/YKsni5R+wki2lZCFRNsqP+XbOePPm3KP37rsW6n1qep3v3Ddn7txz5hyiChVKportzoplcsXWYMVy+SCXbNYgbq0V03LONmHFjOhke7ZiGlrZPq2YMfeUQSFO2Q6smDFTbE9WLJUvtl4rZkwduXlqrGOTbZecs4Pcp13x42n1XKPXggCeenL+F7Yltne2B7ZV/VCRIM6AFW/8L5y65Yx7TcAf9VjTRM7Xp7RRrwUTFsEbuZf9oZ9tlq2ZXNBu5ZvxGioJbOIaBEb1NPJ8UhvrYXskN0cSKNiWFYFUS5ICG0aTxC1IBPqJ0WU5JSGJdlmHJ5roIYVBr402bMaCVG7Q6GgziJEGfJ3ERF/ZckbD5GdqjN2elCgCQrebDNqa10eUvk9uo6E4MYI1KiDouhove02vr9iuH/O6ZsFrbeT2gLS0HXIx97w/BnzBJpT1ibaEFoNTAeOkCwL0pD6KM3qR3H+22Y7IPdtOrjqWWwo3nyB9NNiEx94BJskdkzGwFGInE1rTEOUvFdVsc3l3AZgPVU0CXwDrOwBvnLNiBHzKcs96HCqopm6FGiSJHlwA1hvebt46fuGSyrs0YPPFqhm9PSFJMVw4iiXNfbTF/6JisWpmeh8Fpd7wsWRQDPTcC+MT/uSGX+Ff8w2kV3pNHn1dTQAAAABJRU5ErkJggg==>