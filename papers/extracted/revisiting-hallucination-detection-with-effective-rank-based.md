---
source_pdf: papers/Revisiting Hallucination Detection with Effective Rank-based Uncertainty.pdf
slug: revisiting-hallucination-detection-with-effective-rank-based
pages: 24
extracted_on: 2026-07-13
---

# Revisiting Hallucination Detection with Effective Rank-based Uncertainty

## Page 1

REVISITING HALLUCINATION DETECTION WITH
EFFECTIVE RANK-BASED UNCERTAINTY
Rui Wang1
Zeming Wei1
Guanzhang Yue1
Meng Sun1
1Peking University
ABSTRACT
Detecting hallucinations in large language models (LLMs) remains a fundamental
challenge for their trustworthy deployment. Going beyond basic uncertainty-driven
hallucination detection frameworks, we propose a simple yet powerful method
that quantifies uncertainty by measuring the effective rank of hidden states de-
rived from multiple model outputs and different layers. Grounded in the spectral
analysis of representations, our approach provides interpretable insights into the
model’s internal reasoning process through semantic variations, while requiring no
extra knowledge or additional modules, thus offering a combination of theoretical
elegance and practical efficiency. Meanwhile, we theoretically demonstrate the
necessity of quantifying uncertainty both internally (representations of a single
response) and externally (different responses), providing a justification for using
representations among different layers and responses from LLMs to detect hallu-
cinations. Extensive experiments demonstrate that our method effectively detects
hallucinations and generalizes robustly across various scenarios, contributing to a
new paradigm of hallucination detection for LLM truthfulness.
1
INTRODUCTION
The advent of Large Language Models (LLMs) has catalyzed a paradigm shift across artificial
intelligence, enabling remarkable capabilities in generative and reasoning tasks. From sophisticated
dialogue to complex code generation, their prowess is undeniable (Achiam et al., 2023; Team et al.,
2023). However, their reliability continues to suffer from hallucinations (Huang et al., 2025), where
models produce outputs that are fluent and contextually plausible, but factually incorrect. Unlike
simple errors, hallucinations are particularly insidious because they are often indistinguishable from
trustworthy responses, making them especially dangerous in high-stakes domains such as healthcare
and scientific discovery. The issue arises from the probabilistic nature of next-token prediction:
LLMs optimize for linguistic plausibility rather than factual accuracy, and without explicit grounding,
minor biases or gaps in knowledge can cascade into confident yet misleading narratives. As a result,
hallucinations not only limit practical deployment but also raise fundamental questions about the
epistemic reliability of generative models.
Despite various attempts to mitigate hallucinations through architectural modifications and training
heuristics, the issue remains far from solved. The challenge of mitigating hallucinations is intrinsically
tied to the problem of uncertainty quantification (UQ). Ideally, an LLM should not only provide an
answer but also communicate how confident it is. If a model could reliably estimate its epistemic
uncertainty, it could abstain or defer when unsure, thereby reducing hallucinations (Kendall & Gal,
2017; Abbasi Yadkori et al., 2024). Although a large body of research has explored UQ in traditional
machine learning, existing methods such as Monte Carlo dropout (Gal & Ghahramani, 2016) or deep
ensembles (Lakshminarayanan et al., 2017) are computationally impractical for billion-parameter
LLMs. Similarly, approaches based on retrieval augmentation or auxiliary calibration modules add
system complexity and latency. This highlights the pressing need for lightweight, self-contained, and
scalable UQ techniques tailored to the architecture and deployment constraints of modern LLMs.
In this paper, we propose an internally interpretable solution for UQ that operates purely from
the internal state of the model. Our hypothesis is that the internal representations of LLMs contain
rich but underutilized information. These representations not only reveal the model’s reasoning
1
arXiv:2510.08389v1  [cs.AI]  9 Oct 2025

## Page 2

process, enhancing internal interpretability, but their sampling probability distributions also capture
different reasoning paths, which in turn manifest as semantic divergences. Intuitively, since LLMs
inherently involve stochasticity during forward propagation, small probabilistic perturbations can
lead to noticeable shifts in their generation process. When a model lacks sufficient knowledge or
reasoning ability, these perturbations can cause its internal representations to diverge semantically
during layer-wise forward passes and eventually surface as hallucinations (Huang et al., 2025). In
contrast, a confident and well-grounded model maintains robust and constrained hidden trajectories
despite such perturbations, leading to consistent and reliable output. Thus, representational divergence
provides a natural signal for quantifying uncertainty and identifying hallucination-prone generations.
(a) The correct answers with low uncertainty
(b) The hallucinated answers with high uncertainty
Figure 1: Examples of detecting hallucinations using Effective Rank-based Uncertainty.
To formalize this intuition, we leverage the notion of effective rank (Roy & Vetterli, 2007) of
the matrix of embedding vectors as a measure of uncertainty. Effective rank serves as a smooth
measure of the divergence among vectors within a matrix, and is employed in our method to detect
semantic variations in different embedding vectors. Intuitively, a low effective rank indicates that the
activation energy is concentrated in a few directions, suggesting a confident, deterministic internal
state (Figure 1a). By contrast, a high effective rank signals that the energy is spread diffusely,
indicative of uncertainty, confusion, and a higher likelihood of hallucination (Figure 1b).
We extensively validate our hypothesis across multiple benchmark datasets and model architectures.
Our experiments demonstrate that effective rank is a highly predictive signal for hallucination detec-
tion, consistently outperforming or matching the performance of established strong baselines while
being significantly efficient and scalable. Additionally, we demonstrate how aleatoric uncertainty
(the inherent stochasticity of LLMs) progressively amplifies and obscures epistemic uncertainty
(uncertainty in the knowledge and capabilities encoded by the model’s parameters) within the model’s
internal representations during a single sequence generation without multiple samples. This finding
provides both the theoretical justification for the necessity of semantic sampling through multiple
responses and the foundational rationale for its effectiveness.
Our contributions are threefold:
• A Novel Spectral Perspective on Uncertainty. We propose and empirically validate the
Effective Rank-based Uncertainty as an efficient, robust, and model-intrinsic measure of
uncertainty for LLMs.
• A lightweight, Training-Free Detector. We introduce a rapid hallucination detection
method that does not require additional tools, fine-tuning, or external knowledge.
• Practical and Theoretical Analysis. We conducted experiments across different scenarios,
demonstrating the consistent effectiveness of our method and providing interpretable insights
for the quantitative analysis of uncertainty within the model’s internal representations.
2

## Page 3

2
RELATED WORK
2.1
HALLUCINATION DETECTION
Hallucination in LLMs has been widely studied from multiple perspectives. A major line of work
attributes hallucinations to various forms of uncertainty inherent in the model. Specifically, hal-
lucinations often arise due to (i) insufficient or incomplete knowledge and reasoning capabilities,
(ii) excessive stochasticity in the generation process, and (iii) low faithfulness to the provided
context. These uncertainty-related factors account for a large proportion of hallucination cases in
practice (Huang et al., 2025; Zhang et al., 2025; Tonmoy et al., 2024). However, hallucinations can
also occur even when model uncertainty is low, for example, when the model’s internal knowledge
itself is systematically incorrect or outdated. Such cases typically require techniques like knowledge
editing or targeted fine-tuning for correction (Huang et al., 2024).
To correct and reduce hallucinations in LLMs, we first need highly accurate hallucination detection
methods. Methods for detecting hallucinations in LLMs generally fall into three main categories.
Retrieval-based methods verify generated content against external knowledge sources or retrieved
evidence (Lewis et al., 2020). Self-verification approaches such as P_False (Kadavath et al., 2022)
and SelfCheckGPT (Manakul et al., 2023) prompt the model to evaluate its own generations for
consistency or correctness. Supervised detection involves training classifiers on labeled datasets to
identify hallucinated content (Arteaga et al., 2024). While often effective, these methods typically
depend on external tools or costly annotations, and may introduce latency for real-time applications.
These limitations motivate the exploration of intrinsic, uncertainty-based detection paradigms.
2.2
UNCERTAINTY QUANTIFICATION FOR HALLUCINATION DETECTION
Uncertainty estimation has emerged as a prominent approach for hallucination detection in LLMs.
The core premise is that a model’s internal confidence can indicate output veracity. Initial approaches
used token-level or sequence-level probability distributions, such as the Length-normalized En-
tropy (Malinin & Gales, 2020), which normalizes entropy by the number of tokens to mitigate the
impact of varying sequence lengths. However, these methods capture lexical rather than semantic
uncertainty. As models can be lexically confident yet semantically incorrect (Kuhn et al., 2023),
recent benchmarks (Ye et al., 2024; Nado et al., 2021) have spurred exploration of semantic properties
and internal representations for improved detection.
Based on semantic analysis, Kuhn et al. (2023); Farquhar et al. (2024) introduced Semantic Entropy
(SE) and Discrete Semantic Entropy (DSE) to address limitations of token-level metrics. SE and
DSE cluster generations by semantic equivalence using natural language inference (NLI) models,
then compute entropy over the distribution of semantic categories. Note that while DSE directly
estimates the probability of each semantic category using the frequency of answers in each cluster,
SE further incorporates a token-level probability adjustment. High SE indicates semantic uncertainty,
effectively capturing meaning-level variations. Meanwhile, the INSIDE framework proposed by Chen
et al. (2024) measures the model’s internal uncertainty by extracting the internal embedding vectors
of the model, computing their covariance matrix, and calculating the determinant of the covariance
matrix through eigenvalue decomposition to obtain the Eigenscore. This Eigenscore approximates
differential entropy, with higher scores indicating internal inconsistency. These methods each delve
deeply into the uncertainty about semantic variances or internal representations, but the relationship
between internal representations and external semantic information remains largely unexplored.
3
METHODOLOGY
In this section, we propose a novel uncertainty-based approach to hallucination detection in LLMs.
Our primary motivation is to additionally leverage the internal representations of the model to measure
its uncertainty, and we adopt the effective rank from spectral analysis as our main method, which
integrates the characteristics of smoothness and clear practical interpretability. This section introduces
the procedure and the underlying theoretical justification.
3

## Page 4

3.1
CONSTRUCTING THE EMBEDDING MATRIX
For each query q, we first construct the representation matrix extracted from different layers and
responses to calculate the effective rank. To this end, we sample m1 responses and extract embeddings
from m2 layers per response, resulting in m = m1 × m2 embedding vectors ⃗a1, ⃗a2, · · · , ⃗
am ∈Rn.
These vectors are concatenated into a representation matrix:
A = [ ⃗a1, ⃗a2, · · · , ⃗
am] ∈Rn×m
(1)
where each ⃗ai corresponds to the embedding of a particular response at a specific layer. Follow-
ing Skean et al. (2025), who find that intermediate layers of an LLM strike the best balance between
preserving useful information and removing noise, while the final layer often overfits the training
objective and eliminates generalization, we therefore extract the embedding vector from the exact
middle hidden layer for every response, both in our method and the Eigenscore baseline.
3.2
SPECTRAL ANALYSIS
Next, we want to quantify the dispersion of the column vectors of this matrix, and the notion of
singular values offers a perfectly suited mathematical tool for this, since singular values measure how
much the matrix stretches or compresses vectors along certain directions in space. We then compute
the singular value decomposition (SVD) of the matrix:
A = UΣV ⊤
(2)
where Σ = diag(σ1, σ2, . . . , σm) contains the singular values in descending order.
To quantify the dispersion of vectors across different directions using the entropy of the singular
values, we normalize the singular value spectrum into a probability distribution:
pi =
σi
Pm
j=1 σj
,
i = 1, . . . , m
(3)
3.3
ENTROPY-BASED MEASURE AND EFFECTIVE RANK
We consider the Shannon entropy of the singular value distribution as our measure of uncertainty for
the query q:
H = −
m
X
i=1
pi ln pi
(4)
Using entropy to quantify uncertainty is common, yet our approach rests on an exceptionally clear
and elegant mathematical foundation. The quantity exp(H) is the effective rank of the matrix A,
which measures the “effective linear independence” of the vectors in matrix A (Roy & Vetterli, 2007).
Intuitively, exp(H) corresponds to the effective number of distinct semantic modes represented in
the embedding vectors. Hence, a larger entropy indicates higher uncertainty in the model’s outputs.
Through Jensen’s inequality, it can be readily shown that the effective rank of a matrix equals its true
rank if and only if the matrix has a rank of 1 or its column vectors are pairwise orthogonal and of equal
length (corresponding to the cases of minimum and maximum uncertainty, respectively). In all other
cases, the effective rank is strictly less than the true rank and varies continuously with the dispersion
of the vectors. Therefore, effective rank smoothly encodes how divergent the column vectors are.
In practice, semantically similar embedding vectors are often similar yet not perfectly aligned, so a
continuous effective rank captures their consistency far better than the discrete true rank. Moreover,
the effective rank here possesses a clear and practical interpretation as the “effective number of
semantic categories”, providing us with an excellent tool for further analyzing and enhancing the
internal interpretability of LLM models (Zhuo et al., 2023).
We summarize the above procedure for computing the effective rank in Algorithm 1. In the following
section, we will empirically validate the effectiveness, robustness and generality of this method.
4

## Page 5

Algorithm 1: Effective Rank-based Uncertainty Computation
Input: Query q, number of responses m1, number of layers per response m2, LLM (white-box)
Output: Effective rank exp(H)
for i : 1 →m1 (in parallel) do
Sample response ri from the LLM using query q;
Extract the hidden state hi corresponding to the last generated token of response ri;
for j : 1 →m2 (in parallel) do
Extract embedding vector aij ∈Rn from the corresponding position of hi
end
end
Form embedding matrix A = [a11, . . . , a1m2, . . . , am1m2] ∈Rn×m, where m = m1 × m2;
Compute SVD: A = UΣV ⊤, with singular values {σ1, . . . , σm};
Normalize singular values: pi = σi/ Pm
j=1 σj for i = 1, . . . , m;
Compute entropy: H = −Pm
i=1 pi ln pi;
return exp(H)
4
EXPERIMENTS
4.1
EXPERIMENTAL SETUP
Datasets. Our dataset selection is designed to rigorously evaluate hallucination detection across
a spectrum of knowledge and reasoning domains. TriviaQA (Joshi et al., 2017) serves as a broad
open-domain knowledge benchmark. In contrast, Natural Questions (NQ) (Kwiatkowski et al., 2019)
and BioASQ (Tsatsaronis et al., 2015) probe deeper into specialized knowledge within natural science
and healthcare domains where LLM hallucinations can be particularly misleading and harmful. To
test foundational competencies beyond factual recall, we also include SQuAD (Rajpurkar et al., 2016)
for context understanding and text inference.
Models. For our backbone language models, we utilize three widely recognized open-weight models:
Llama-2-7b-chat (Touvron et al., 2023), Llama-2-13b-chat, and Mistral-7B-v0.1 (Jiang et al., 2023)
to evaluate our methods across models of varying scales and distinct architectural lineages.
Evaluation Metrics. To determine whether a model’s generation contains hallucinations, we use
ROUGE-L (Lin, 2004), an n-gram based metric that computes the longest common subsequence
between the generation and the ground-truth answer. A generation is considered correct if its ROUGE-
L score ≥0.5; otherwise, it is flagged as a hallucination. The validity of this annotation method is
discussed in Appendix F. To evaluate the overall performance of the hallucination detection methods
themselves, we use the Area Under the Receiver Operating Characteristic curve (AUROC) (Filos
et al., 2019). AUROC is chosen as it provides a comprehensive measure that is agnostic to the
absolute scale of the uncertainty scores produced by different detectors. This makes it an ideal metric
for comparing the ability to rank hallucinated examples higher than non-hallucinated ones.
Baselines. We benchmark our proposed method against recent strong baselines: Semantic Entropy,
Discrete Semantic Entropy (Farquhar et al., 2024), and Eigenscore (Chen et al., 2024). We have also
considered these representative baselines: P_False (Kadavath et al., 2022), and Length-Normalized
Entropy (Malinin & Gales, 2020). Detailed introductions to these baselines can be found in Section 2.
Implementation Details. All experiments were conducted on a virtual GPU (vGPU) with 48GB
of memory. We use a temperature of 1.0 for sampling and generate N = 10 answers per input. For
embedding extraction, we utilize the hidden state of the last token from the exact middle layer of the
model. Finally, in the Eigenscore calculation, we follow the recommendation of Chen et al. (2024)
and set the small regularization term to α = 0.001.
4.2
MAIN RESULTS
The overall performance comparison of our Effective Rank (ER) method against five baselines is
summarized in Table 1. The results, measured in AUROC across three models and five diverse
datasets, demonstrate the efficacy and generality of our approach in detecting hallucinations.
5

## Page 6

Model
Dataset
ER (Ours)
ES
PF
DSE
LNE
SE
Llama-7b
TriviaQA
0.7877
0.7802
0.6625
0.7758
0.6947
0.7750
SQuAD
0.7212
0.7199
0.6594
0.7172
0.6505
0.7181
BioASQ
0.8447
0.8433
0.7321
0.8437
0.4703
0.8425
NQ
0.7029
0.7056
0.6584
0.6984
0.6495
0.7001
Average
0.7641
0.7623
0.6781
0.7588
0.6163
0.7589
Llama-13b
TriviaQA
0.7407
0.7342
0.7379
0.7331
0.6930
0.7390
SQuAD
0.7259
0.7239
0.6945
0.7381
0.6886
0.7350
BioASQ
0.8234
0.8215
0.7980
0.7951
0.5610
0.7988
NQ
0.7284
0.7270
0.6841
0.7234
0.6866
0.7258
Average
0.7546
0.7517
0.7286
0.7474
0.6573
0.7497
Mistral-7b
TriviaQA
0.7634
0.7579
0.7239
0.7575
0.6519
0.7553
SQuAD
0.7208
0.7120
0.6333
0.7227
0.6223
0.7238
BioASQ
0.8563
0.8513
0.7173
0.8507
0.5955
0.8513
NQ
0.7658
0.7627
0.7523
0.7678
0.6770
0.7662
Average
0.7766
0.7710
0.7067
0.7747
0.6367
0.7742
Table 1: Overall comparison of ER with baselines, where ER denotes Effective Rank, ES denotes
Eigenscore, PF denotes P_False, DSE denotes Discrete Semantic Entropy, LNE denotes Length-
Normalized Entropy, and SE denotes Semantic Entropy. All numbers in the table are AUROC scores;
values closer to 1 indicate stronger hallucination-detection ability.
Overall Effectiveness. Our method achieves the highest AUROC score in 8 out of 12 evaluation
scenarios, establishing a consistent and strong performance baseline. Notably, this superiority is not
confined to a specific model or dataset but is observed across various scales (7b and 13b parameters)
and architectures (Llama and Mistral). This indicates that the principle of analyzing representation
space stability through effective rank is a generalizable strategy for hallucination detection. Even in
cases where ER does not rank first, its performance remains highly competitive (e.g., on NQ dataset).
We also note that our method’s overall margin over the baselines is larger on Llama-2-13B-chat than
on Llama-2-7B-chat, suggesting its potential scales with model size.
Analysis of Shortcomings. A primary limitation of our method is observed in its unstable perfor-
mance on the SQuAD dataset. This suggests that for tasks requiring complex textual reasoning and
comprehension, rather than pure factual recall, the relationship between internal representations and
uncertainty may be less consistent. In such scenarios, SE and DSE methods, which directly measure
semantic variation across multiple sampled answers, appear to capture uncertainty more reliably.
Nevertheless, Effective Rank remains highly competitive compared to all methods except SE and
DSE on SQuAD dataset. Our ablation study further analyzes the interplay between the roles of
different hidden layers and the requirements of tasks across various datasets.
Comparative Analysis of Baselines. The results also shed light on the relative performance of
existing methods. The Eigenscore method, apart from showing a noticeable gap compared to
Effective Rank on the TriviaQA dataset, remains relatively close to our approach and occasionally
even surpasses it by a narrow margin. This is because both methods utilize the representations from
the model’s internal hidden layers to measure uncertainty. However, Eigenscore is an approximation
of differential entropy, whereas Effective Rank directly yields an intuitive and effective uncertainty
indicator, the “effective number of distinct semantic categories”. This gives Effective Rank an
overall advantage over Eigenscore. Although SE and DSE methods generally perform better than our
approach on SQuAD, they require an auxiliary NLI model and additional time for semantic reasoning
(Table 2). Additionally, while P_False and LNE methods are relatively unstable, they occasionally
deliver relatively strong performance.
In conclusion, the main results strongly support the effectiveness of our Effective Rank approach. Its
dominant performance on TriviaQA and BioASQ tasks highlights its value as a tool for detecting
factual hallucinations, a critical challenge for modern LLMs.
6

## Page 7

Methods
Effective
Internally interpretable
Semantic Information
Time Efficiency
ES
✓
✓
✗
9.5s
SE, DSE
✓
✗
✓
11.7s
PF
✗
✗
✗
10.1s
LNE
✗
✗
✗
9.6s
ER (ours)
✓
✓
✓
9.5s
Table 2: Qualitative comparison of hallucination detection methods. In terms of time efficiency,
the numbers in the table indicate the average time spent on each question in the experiment where
each method generated results for 3 models × 4 datasets under the parameter settings of the main
experiment. Our method took an average of 9.5 seconds, which is essentially the same as the time
required to naturally generate answers without hallucination detection. The detailed time complexity
analysis of our ER method can be found in Appendix F.
4.3
ABLATION STUDIES
N
Dataset
M1
M5
L1
L5
Best Baseline
10
TriviaQA
0.7877
0.7700
0.7809
0.7629
0.7802 (ES)
SQuAD
0.7212
0.7217
0.7191
0.7190
0.7199 (ES)
BioASQ
0.8447
0.8461
0.8481
0.8472
0.8437 (DSE)
NQ
0.7029
0.7038
0.7036
0.7049
0.7056 (ES)
Average
0.7641
0.7604
0.7629
0.7585
0.7623 (ES)
15
TriviaQA
0.7862
0.7902
0.7870
0.7807
0.7825 (ES)
SQuAD
0.7407
0.7407
0.7391
0.7395
0.7391 (ES)
BioASQ
0.8715
0.8690
0.8684
0.8678
0.8682 (ES)
NQ
0.7182
0.7166
0.7174
0.7188
0.7206 (ES)
Average
0.7792
0.7780
0.7767
0.7791
0.7776 (ES)
20
TriviaQA
0.7786
0.7780
0.7729
0.7702
0.7743 (ES)
SQuAD
0.7596
0.7604
0.7599
0.7610
0.7588 (DSE)
BioASQ
0.8645
0.8625
0.8638
0.8620
0.8628 (ES)
NQ
0.7277
0.7292
0.7274
0.7278
0.7284 (ES)
Average
0.7826
0.7825
0.7810
0.7803
0.7809 (ES)
Table 3: Ablation studies on Llama-2-7b-chat, where N refers to the number of generations, M1
refers to extracting the exact middle hidden layer for each answer in the ER method (i.e., the ER
method used in the main experiment), M5 refers to extracting the middle five layers, L1 refers to
extracting the last layer, and L5 refers to extracting the last five layers. Ablation studies on other
models and complete experimental data can be found in Appendix D.
Number of generations and the use of different hidden layers. Our ablation studies on Llama-
2-7b-chat reveal that the performance of the ER method is robust to the number of generations,
since it maintains a consistent overall advantage over the baselines across different values of N.
Moreover, no single layer extraction strategy consistently outperforms all others, indicating a complex
interaction between the task domain and the representational properties of different model layers.
Notably, although M5, L1, and L5 hidden layer selection strategies yield results comparable to M1
and can complement M1 to a certain extent, they are slightly inferior overall and exhibit marginally
less stability (somewhat analogous to how DSE is generally slightly weaker than SE). Furthermore,
increasing the number of generations (N) does not yield uniform improvements, implying a task-
dependent optimum exists for this hyperparameter.
Temperature. According to Table 4, we observe that the hallucination detection performance is
generally optimal when the temperature is set around 0.5 and 1.0. Under these conditions, our
Effective Rank method also demonstrates a significant overall advantage. Excessively high or low
temperatures substantially undermine uncertainty-based hallucination detection methods, while the
Self-Verification-based P_False method unexpectedly exhibits a notable relative advantage in such
7

## Page 8

TriviaQA
SQuAD
BioASQ
NQ
Average
t=0.1
Best ER
0.6782 (L1)
0.6466 (M5)
0.7416 (L5)
0.6841 (L1)
0.6768 (L1)
Best Baseline
0.6695 (PF)
0.6413 (PF)
0.7720 (PF)
0.7308 (PF)
0.7034 (PF)
t=0.5
Best ER
0.7628 (M5)
0.7401 (L5)
0.8452 (L5)
0.7724 (L1)
0.7771 (L1)
Best Baseline
0.7350 (ES)
0.7330 (ES)
0.8378 (ES)
0.7641 (ES)
0.7675 (ES)
t=1.0
Best ER
0.7651 (L1)
0.7208 (M1)
0.8584 (M5)
0.7696 (M5)
0.7769 (M5)
Best Baseline
0.7579 (ES)
0.7268 (SE)
0.8513 (ES)
0.7678 (DSE)
0.7747 (DSE)
t=2.0
Best ER
0.6685 (M5)
0.5754 (M1)
0.7329 (M5)
0.6118 (M1)
0.6437 (M1)
Best Baseline
0.7084 (PF)
0.5698 (ES)
0.7244 (SE)
0.7006 (PF)
0.6640 (PF)
Table 4: Ablation studies on Mistral-7B-v0.1 about temperature, where t denotes the temperature, and
Best ER refers to the Effective Rank that performed the best among the four hidden vector selection
methods (M1, M5, L1, L5). Complete experimental data can be found in Appendix D.
scenarios, albeit with remaining instability. Moreover, the detailed data show that the L1 and L5
methods are more robust to low temperatures, while the M1 and M5 methods are more robust to high
temperatures.
5
INTERNAL INTERPRETABILITY: THE DECOMPOSITION AND QUANTITATIVE
ANALYSIS OF UNCERTAINTY
5.1
EMPIRICAL MOTIVATION
While multi-sample methods are effective for hallucination detection, we still want to explore the real-
time detection without multiple samples. On Llama-2-7b-chat, we use a sliding window to measure
the effective rank of the internal representations across every three consecutive layers and then take
the average. The final results vary slightly between different responses, but generally fluctuate around
1.9. This indicates that semantic changes occur within the model during the reasoning process, and
these shifts exhibit a certain degree of similarity but little variability between different responses.
This finding is broadly consistent with conclusions drawn from using cosine similarity of internal
representations to assess internal consistency (Min et al., 2024). However, the experiments revealed
that this form of internal uncertainty showed only a weak correlation with hallucinations (overall
AUROC ≈0.57, barely exceeding the random chance). This unsuccessful attempt prompted us to
reconsider: What truly constitutes effective internal uncertainty for hallucination detection?
According to the perspective of Depeweg et al. (2018), the uncertainty in the predictions of Deep
Neural Networks can be decomposed into two primary types: aleatoric uncertainty, which is inherent
in the data and model architecture due to its ambiguity or inherent stochasticity, and epistemic
uncertainty, which stems from the model’s lack of knowledge or ability. The latter is considered
particularly critical for detecting hallucinations. We subsequently argue, however, that within any
single sampled response, the information signal pertaining to epistemic uncertainty is largely masked
by aleatoric uncertainty. Furthermore, this aleatoric uncertainty can propagate and become amplified
through the model’s reasoning process. Consequently, when the LLM’s internal knowledge is
insufficient to produce a determinate result amidst this prevailing uncertainty noise (i.e., when a
hallucination occurs), the generation and reasoning processes of LLMs often involve probabilistic
distributions with obvious uncertainty, which ultimately manifest as semantic divergences across
different sampled answers (Manakul et al., 2023; Huang et al., 2025).
5.2
PRELIMINARIES AND PROBLEM SETUP
Consider an autoregressive language model parameterized by θ. Given an input prompt q, the model
generates a sequence of tokens (y1, y2, . . . , yT ) and a corresponding sequence of hidden states
(h1, h2, . . . , hL), where ht ∈Rd is the hidden state at step t. The generation process is defined as:
yt ∼p(yt|ht−1; θ),
ht = f(ht−1, yt; θ)
where f is a deterministic, non-linear transformation, and p is the model’s output distribution.
8

## Page 9

To frame our discussion, we adapt the Bayesian deep learning framework (Kendall & Gal, 2017;
Depeweg et al., 2018) to reason about the hidden states. We consider the expected total variance of the
hidden state ht across the data generation process, which can be decomposed into two components:
Var(ht)
| {z }
Total Uncertainty
= Eθ[Var(ht|θ)]
|
{z
}
Aleatoric Uncertainty
+ Varθ(E[ht|θ])
|
{z
}
Epistemic Uncertainty
.
(5)
Since in modern large language models, parameters are typically fixed at a point estimate, this
variance decomposition relies on a Bayesian treatment of model parameters, such as deep ensembles,
Monte Carlo dropout, or variational Inference. We build upon two common observations from prior
work, noting that they are tendencies rather than absolute laws:
Peaked Parameter Posterior: In many well-trained LLMs, the parameters are often found in a
region of a peaked posterior distribution p(θ|D) (Izmailov et al., 2018; Fort et al., 2019). This
suggests that the parameter covariance Σθ = Cov(θ) is often small, though not negligible, especially
for under-regularized or smaller models.
Representational Expansion: Each transformer layer actively transforms and enriches its input, cre-
ating overcomplete representations in high dimensions (Sanford et al., 2023). This process inherently
amplifies minor input variations, causing stochasticity to accumulate across layers (Schoenholz et al.,
2016). However, this expansive potential is also constrained by key stabilizing mechanisms within
the Transformer architecture, such as residual connections and layer normalization.
5.3
POTENTIAL DOMINANCE OF ALEATORIC UNCERTAINTY
The autoregressive process can be viewed as a Markov chain where sampling noise may accumulate
through the network’s dynamics. We analyze the variance at step t for a fixed θ to gain intuition.
Lemma 1 (Variance Propagation with Expansion Property). For a fixed parameter θ, the variance of
the hidden state ht can be bounded from below recursively. Assuming the transformation f exhibits
representational expansion in deep Transformers (Wang et al., 2022; Schoenholz et al., 2016), the
conditional variance satisfies:
Eθ[Var(ht|θ)] ≥EθEht−1

|Jy(ht−1)|2
F · Var(yt|ht−1; θ)

+ ∆nonlin
(6)
where Jy = ∂f
∂y is the Jacobian of the transformation with regard to the input token embedding, | · |F
denotes the Frobenius norm, and ∆nonlin is a small term capturing higher-order effects.
Lemma 2 (Epistemic Uncertainty Bound). The epistemic variance can be bounded by:
Varθ(E[ht|θ]) ≤∥Gt∥2
F · Tr(Σθ) + ϵt
(7)
where Gt = ∂E[ht|θ]
∂θ
is the sensitivity matrix of the expected hidden state trajectory to parameters,
Σθ is the covariance of the parameter posterior, and ϵt is a non-linearity term in parameter space.
For detailed derivation and further discussion, please refer to Appendix C. Based on these lemmas, it
is straightforward to obtain the final conclusion, noting its conditional nature.
Proposition 1. For an autoregressive language model exhibiting a sufficiently peaked parame-
ter posterior and representational expansion over t steps, the aleatoric uncertainty in its hidden
representations may dominate the epistemic uncertainty:
Eθ[Var(ht|θ)] ≫Varθ(E[ht|θ]).
(8)
5.4
BRIDGING THEORY AND METHOD: THE NECESSITY OF A UNIFIED APPROACH
Our theoretical analysis underscores a critical challenge: the predominance of aleatoric uncertainty
during a single forward pass without multiple samples obscures the epistemic uncertainty crucial
for hallucination detection. With multiple samples, we can more easily probe the different paths
the model considers, approaching the full picture of its internal probability distribution. This lets
us externalize that distribution as semantic divergence, yielding richer and more visible uncertainty
information for hallucination detection. However, this raises the question of how to best quantify
this uncertainty. Relying solely on external metrics, such as the semantic entropy of final outputs,
9

## Page 10

discards the rich information within the model’s internal representations. Conversely, methods
that focus purely on internal statistics, such as the Eigenscore which approximates the differential
entropy of internal probability distribution, lack clear and interpretable associations with external
semantic information. Meanwhile, our proposed method provides exactly such a unified view. By
applying effective rank to hidden state embeddings collected from multiple sampled responses, it
simultaneously (i) captures the “effective number of semantic categories” emerging externally across
generations and (ii) encodes the dispersion of internal representations that reflect the probabilistic
dynamics of the model.
In this way, Effective Rank-based Uncertainty bridges the gap between external semantic variation
and internal interpretability. The effective rank method elegantly captures how the model’s internal
representations continuously and interpretably transform into external semantic divergences. There-
fore, we believe that further employing the effective rank method to study the external real-world
meanings of internal representations is a highly promising direction.
6
CONCLUSION
In this work, we introduce the concept of Effective Rank-based Uncertainty, a spectral analysis
tool, and apply it to the internal representations of LLMs, thereby proposing a simple and efficient
method for hallucination detection. Through comprehensive experiments and ablation studies, we
demonstrated the effectiveness of our approach. The practical interpretation of effective rank as
the “effective number of distinct semantic categories in hidden representations” provides a clear
and motivated basis for its application. Furthermore, by decomposing and quantitatively analyzing
internal uncertainty, we provide in-depth theoretical modeling and analysis of the propagation of
internal uncertainty noise in LLMs and how to leverage such uncertainty for hallucination detection.
We believe that further exploration of effective rank as an elegant and powerful theoretical tool holds
significant promise for enhancing the interpretability and reliability of AI systems.
ETHICS STATEMENT
Our overarching goal is to contribute to the development of safer AI systems by providing users
with tools to better interpret the confidence and reliability of the output of the language model. In
principle, such mechanisms could mitigate several risks associated with LLMs based on foundation
models, including the production of misleading or harmful content in sensitive domains such as
medicine. Nevertheless, this promise also carries potential downsides: systematic errors in uncertainty
estimation, or poor communication of uncertainty, could foster misplaced trust and lead to unintended
harms. Although our work advances methodological understanding and introduces new approaches
for uncertainty quantification in LLM, we emphasize that deployment should be preceded by a careful,
context-specific evaluation to ensure that the communicated uncertainty genuinely empowers users
rather than creating confusion or false assurance.
REPRODUCIBILITY STATEMENT
Due to the high computational demands of experimentation with large foundation models, most
prior work on uncertainty in LLM has depended on proprietary systems, costly fine-tuning pro-
cedures, or labor-intensive human evaluation, which can limit accessibility for many academic
researchers. In contrast, our approach leverages recently released, openly available models and
constructs a fully open-source pipeline for uncertainty quantification in LLM. Importantly, our
method requires neither fine-tuning nor additional training of foundation models, and can be applied
directly to pre-trained models in an “off-the-shelf” manner. We hope this design lowers barri-
ers for future research in the academic community and facilitates straightforward replication of
our experiments. Our open-source code is available at https://github.com/1240148048/
Effective-Rank-based-Uncertainty.
10

## Page 11

REFERENCES
Yasin Abbasi Yadkori, Ilja Kuzborskij, András György, and Csaba Szepesvari. To believe or not to
believe your llm: Iterative prompting for estimating epistemic uncertainty. Advances in Neural
Information Processing Systems, 37:58077–58117, 2024.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774, 2023.
Gabriel Y Arteaga, Thomas B Schön, and Nicolas Pielawski. Hallucination detection in llms: Fast
and memory-efficient fine-tuned models. arXiv preprint arXiv:2409.02976, 2024.
Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. Inside:
Llms’ internal states retain the power of hallucination detection. arXiv preprint arXiv:2402.03744,
2024.
Stefan Depeweg, Jose-Miguel Hernandez-Lobato, Finale Doshi-Velez, and Steffen Udluft. Decom-
position of uncertainty in bayesian deep learning for efficient and risk-sensitive learning. In
International conference on machine learning, pp. 1184–1193. PMLR, 2018.
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large
language models using semantic entropy. Nature, 630(8017):625–630, 2024.
Angelos Filos, Sebastian Farquhar, Aidan N Gomez, Tim GJ Rudner, Zachary Kenton, Lewis Smith,
Milad Alizadeh, Arnoud de Kroon, and Yarin Gal. Benchmarking bayesian deep learning with
diabetic retinopathy diagnosis. Preprint at https://arxiv. org/abs/1912.10481, 2019.
Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan. Deep ensembles: A loss landscape perspec-
tive. arXiv preprint arXiv:1912.02757, 2019.
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model
uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059.
PMLR, 2016.
Baixiang Huang, Canyu Chen, Xiongxiao Xu, Ali Payani, and Kai Shu. Can knowledge editing really
correct hallucinations? arXiv preprint arXiv:2410.16251, 2024.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information
Systems, 43(2):1–55, 2025.
Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Av-
eraging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407,
2018.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas
Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.
org/abs/2310.06825.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly)
know what they know. arXiv preprint arXiv:2207.05221, 2022.
Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer
vision? Advances in neural information processing systems, 30, 2017.
11

## Page 12

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664, 2023.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a
benchmark for question answering research. Transactions of the Association for Computational
Linguistics, 7:453–466, 2019.
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. Advances in neural information processing systems,
30, 2017.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:
9459–9474, 2020.
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization
branches out, pp. 74–81, 2004.
Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction.
arXiv preprint arXiv:2002.07650, 2020.
Potsawee Manakul, Adian Liusie, and Mark JF Gales. Selfcheckgpt: Zero-resource black-box
hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896,
2023.
Nay Myat Min, Long H Pham, Yige Li, and Jun Sun. Crow: Eliminating backdoors from large
language models via internal consistency regularization. arXiv preprint arXiv:2411.12768, 2024.
Zachary Nado, Neil Band, Mark Collier, Josip Djolonga, Michael W Dusenberry, Sebastian Farquhar,
Qixuan Feng, Angelos Filos, Marton Havasi, Rodolphe Jenatton, et al. Uncertainty baselines:
Benchmarks for uncertainty & robustness in deep learning. arXiv preprint arXiv:2106.04015,
2021.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for
machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.
Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007
15th European signal processing conference, pp. 606–610. IEEE, 2007.
Clayton Sanford, Daniel J Hsu, and Matus Telgarsky. Representational strengths and limitations of
transformers. Advances in Neural Information Processing Systems, 36:36677–36707, 2023.
Samuel S Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information
propagation. arXiv preprint arXiv:1611.01232, 2016.
Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid
Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. arXiv
preprint arXiv:2502.02013, 2025.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut,
Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly
capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
SMTI Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava Das.
A comprehensive survey of hallucination mitigation techniques in large language models. arXiv
preprint arXiv:2401.01313, 6, 2024.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
12

## Page 13

George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas, Matthias Zschunke,
Michael R Alvers, Dirk Weissenborn, Anastasia Krithara, Sergios Petridis, Dimitris Polychronopou-
los, et al. An overview of the bioasq large-scale biomedical semantic indexing and question
answering competition. BMC bioinformatics, 16(1):138, 2015.
Peihao Wang, Wenqing Zheng, Tianlong Chen, and Zhangyang Wang. Anti-oversmoothing in deep
vision transformers via the fourier domain analysis: From theory to practice. arXiv preprint
arXiv:2203.05962, 2022.
Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue Wang, Derek Wong, Emine Yilmaz, Shuming
Shi, and Zhaopeng Tu. Benchmarking llms via uncertainty quantification. Advances in Neural
Information Processing Systems, 37:15356–15385, 2024.
Yuji Zhang, Sha Li, Cheng Qian, Jiateng Liu, Pengfei Yu, Chi Han, Yi R Fung, Kathleen McKeown,
Chengxiang Zhai, Manling Li, et al. The law of knowledge overshadowing: Towards understanding,
predicting, and preventing llm hallucination. arXiv preprint arXiv:2502.16143, 2025.
Zhijian Zhuo, Yifei Wang, Jinwen Ma, and Yisen Wang. Towards a unified theoretical understanding
of non-contrastive learning via rank differential mechanism. arXiv preprint arXiv:2303.02387,
2023.
13

## Page 14

A
THE USE OF LARGE LANGUAGE MODELS (LLMS)
In preparing this manuscript, we made limited use of LLMs solely to aid in polishing the writing.
Specifically, LLMs were used for improving grammar, clarity, and style of exposition. All conceptual
development, technical contributions, theoretical results, and experimental analyses were conceived
and carried out entirely by the authors. The scientific content, data analysis, and conclusions remain
exclusively the responsibility of the authors.
B
MORE THEORETICAL BACKGROUND ON EFFECTIVE RANK
A key theoretical perspective motivating our approach comes from the information-theoretic inter-
pretation of Shannon entropy. Given a categorical distribution p = (p1, p2, . . . , pC), the Shannon
entropy is defined as H(p)
=
−PC
i=1 pi log pi. While H is commonly viewed as a measure
of uncertainty, it also directly encodes the notion of an effective number of categories, defined as
Neff(p) = exp
 H(p)

. This quantity represents the number of outcomes in a uniform distribution
that would yield the same level of uncertainty as p. For instance,
p = [0.8, 0.1, 0.1]
⇒
H(p) ≈0.639, Neff ≈1.89,
whereas
p = [0.3, 0.3, 0.4]
⇒
H(p) ≈1.089, Neff ≈2.97.
Although both are distributions over three categories, the first case exhibits a stronger bias towards
a single outcome, hence the small probabilities 0.1 are more likely to represent noise or modeling
errors rather than genuine equiprobable alternatives, whereas the second distribution is closer to
true ambiguity among three plausible options. This principle is precisely the motivation behind the
notion of effective rank, introduced in (Roy & Vetterli, 2007). Analogous to the effective number of
categories, effective rank measures the effective dimensionality of the matrix, which in our method
corresponds to the number of equivalent semantic categories represented by the embedding vectors.
As shown in Figure 2, we can intuitively see how the effective rank continuously measures the degree
of divergence of matrix vectors.
Moreover, the effective rank employs the singular value distribution to calculate the Shannon entropy
because the singular values of a matrix inherently capture information about the direction and
magnitude distribution of its internal vectors. The more uniform the singular value distribution, the
more dispersed the linearly independent vector groups within the matrix are; conversely, the more
concentrated they are:
• If all column vectors share the same direction, then A has rank one, and there is only a
single non-zero singular value, thus H = 0, exp(H) = 1, implying complete certainty.
• If all column vectors have the same length and are pairwise orthogonal, then all singular
values are equal. In this case, H = −Σm
i=1
1
m ln( 1
m) = ln m, exp(H) = m, which
corresponds to maximal uncertainty.
We prove that the effective rank of a matrix is always less than or equal to its true rank, and the equality
holds if and only if the two conditions above are satisfied. Let A be a matrix with singular values
σ1, σ2, . . . , σn where n is the minimum dimension of A. The normalized singular values are defined
as pi = σi/∥σ∥1, where ∥σ∥1 = Pn
i=1 σi. The entropy of A is given by H(A) = −Pn
i=1 pi log pi
(with 0 log 0 = 0), and the entropy-based effective rank is erank(A) = exp(H(A)). The true rank is
r = rank(A), the number of non-zero singular values.
To prove erank(A) ≤r, we apply Jensen’s inequality to the concave function log x:
H(A) =
X
i:pi>0
pi log
 1
pi

≤log

X
i:pi>0
pi · 1
pi

= log r
(9)
where the sum is over the r indices with pi > 0. Thus, erank(A) = exp(H(A)) ≤exp(log r) = r.
Equality holds if and only if all non-zero singular values are equal, since log x is strictly concave and
equality in Jensen’s inequality requires that all 1/pi are equal for pi > 0, meaning pi = 1/r for each
non-zero singular value.
14

## Page 15

(a) a rank-one matrix
(b) a rank-two matrix
(c) a rank-three matrix
(d) a rank-three matrix
Figure 2: Visualization of Effective Ranks
C
PROOFS AND DISCUSSION
This appendix provides the detailed proofs for the lemmas and proposition stated in the main text.
C.1
PROOF OF LEMMA 1
(Variance Propagation with Expansion Property) For a fixed parameter θ, the variance of the hidden
state ht can be bounded from below recursively. Assuming the transformation f exhibits representa-
tional expansion in deep Transformers, the conditional variance satisfies:
Eθ[Var(ht|θ)] ≥EθEht−1

|Jy(ht−1)|2
F · Var(yt|ht−1; θ)

+ ∆nonlin
(10)
where Jy = ∂f
∂y is the Jacobian of the transformation with regard to the input token embedding, | · |F
denotes the Frobenius norm, and ∆nonlin is a non-negative term capturing higher-order effects that
typically amplify variance.
Proof. We analyze the evolution of the variance through the deterministic function ht
=
f(ht−1, yt; θ). Our goal is to find a lower bound for Var(ht|θ).
We begin by applying the law of total variance conditional on θ:
Var(ht|θ) = Eht−1[Var(ht|ht−1, θ)] + Varht−1(E[ht|ht−1, θ])
(11)
We now analyze each term separately.
1. Analysis of Eht−1[Var(ht|ht−1, θ)]:
15

## Page 16

Given ht−1, the randomness in ht comes solely from yt. Let us define the conditional mean of the
next token:
µy(ht−1) = E[yt|ht−1; θ]
We perform a first-order Taylor expansion of the function f around the point (ht−1, µy(ht−1)):
ht = f(ht−1, yt; θ) = f(ht−1, µy; θ) + Jy(ht−1) · (yt −µy) + R(ht−1, yt)
(12)
where Jy(ht−1) = ∂f(ht−1,y;θ)
∂y

y=µy is the Jacobian matrix, R(ht−1, yt) is the Taylor remainder
term, which encapsulates all higher-order derivatives.
Taking the conditional expectation E[·|ht−1, θ] of Eq. 12:
E[ht|ht−1, θ] = f(ht−1, µy; θ) + Jy(ht−1) · E[(yt −µy)|ht−1, θ]
|
{z
}
=0
+E[R|ht−1, θ]
= f(ht−1, µy; θ) + E[R|ht−1, θ]
The conditional variance Var(ht|ht−1, θ) is the variance of the right-hand side of Eq. 12 around
this conditional mean. Neglecting the covariance between the linear and remainder terms, we can
approximate:
Var(ht|ht−1, θ) ≳Var (Jy(ht−1)(yt −µy) | ht−1, θ) + Var(R|ht−1, θ)
(13)
The first term is the variance of the linear approximation. Assuming Jy is approximately constant
over the variation of yt (because the internal transformations of well-trained LLMs often exhibit a
certain degree of consistency in generating different sequences), this variance can be expressed as:
Var (Jy(yt −µy) | ht−1, θ) = Jy · Var(yt|ht−1; θ) · J⊤
y
The total variance of this linear term is the trace of this covariance matrix:
Tr
 Jy · Var(yt|ht−1; θ) · J⊤
y

= Tr
 J⊤
y Jy · Var(yt|ht−1; θ)

= |Jy|2
F · Var(yt|ht−1; θ)
The second term in Eq. 13, Var(R|ht−1, θ), is the variance of the remainder. For common modern
activation functions which are smooth and exhibit local linearity, and under the influence of layer
normalization which constrains inputs, the remainder term R is typically small. We denote this small
contribution as δnonlin(ht−1). Crucially, for expansive transformations (Sanford et al., 2023; Wang
et al., 2022), the higher-order terms often amplify variance rather than suppress it. Moreover, Jy
itself contains a large amount of parameter information. Therefore, the Frobenius norm |Jy(ht−1)|F ,
which represents the square root of the sum of the squares of the matrix elements, is often much
greater than 1.
Thus, we can write:
Var(ht|ht−1, θ) ≳|Jy(ht−1)|2
F · Var(yt|ht −1; θ) + δnonlin(ht−1)
(14)
Taking the expectation of Eq. 14 with respect to ht−1 and θ gives the first term of our final bound:
EθEht−1[Var(ht|ht−1, θ)] ≥EθEht−1

|Jy(ht−1)|2
F · Var(yt|ht−1; θ)

+ ∆(1)
nonlin
(15)
where ∆(1)
nonlin = EθEht−1[δnonlin(ht−1)].
C.2
PROOF OF LEMMA 2
[Epistemic Uncertainty Bound] The epistemic variance can be bounded by:
Varθ(E[ht|θ]) ≤|Gt|2
F · Tr(Σθ) + ϵt
Proof. Let µθ = E[θ] be the mean of the parameter posterior distribution and δ = θ −µθ be the
zero-mean perturbation vector with covariance Σθ = E[δδ⊤].
Consider the expected hidden state µh(θ) = E[ht|θ] as a function of the parameters. We perform a
first-order Taylor expansion around µθ:
µh(θ) = µh(µθ + δ) ≈µh(µθ) + Gt · δ
(16)
16

## Page 17

where Gt = ∂µh(θ)
∂θ

θ=µθ
, Gt is the Jacobian matrix describing the sensitivity of the expected hidden
state to parameter changes. The variance of µh(θ) is then:
Varθ(E[ht|θ]) = Varθ(µh(θ)) ≈Varθ(Gt · δ) = E[(Gtδ)(Gtδ)⊤] = GtE[δδ⊤]G⊤
t
= GtΣθG⊤
t
(17)
To find the total variance, we take the trace of this covariance matrix:
Tr(Varθ(µh(θ))) ≈Tr(GtΣθG⊤
t ) = Tr(G⊤
t GtΣθ)
(18)
Applying the Cauchy-Schwarz inequality for the trace (von Neumann’s trace inequality), we get:
Tr(G⊤
t GtΣθ) ≤
q
Tr((G⊤
t Gt)2) · Tr(Σ2
θ)
(a) ≤Tr(G⊤
t Gt) · Tr(Σθ)
(b)
(19)
Inequality (a) is the Cauchy-Schwarz application. Inequality (b) holds because: Tr(A2) ≤(Tr(A))2
for a positive semidefinite matrix A. Noting that Tr(G⊤
t Gt) = |Gt|2
F , we arrive at:
Tr(Varθ(µh(θ))) ≲|Gt|2
F · Tr(Σθ)
The term ϵt is introduced to account for the error in the first-order Taylor approximation, which
includes the effects of higher-order derivatives. This error is typically small if the function µh(θ) is
approximately linear in θ in the region of high posterior probability, an assumption linked to a peaked
posterior.
C.3
PROOF OF PROPOSITION 1
For an autoregressive language model exhibiting a sufficiently peaked parameter posterior and
representational expansion over t steps, the aleatoric uncertainty in its hidden representations may
dominate the epistemic uncertainty:
Eθ[Var(ht|θ)] ≫Varθ(E[ht|θ])
(20)
Proof. Let at = Eθ[Var(ht|θ)] (aleatoric) and et = Varθ(E[ht|θ]) (epistemic).
From Lemma 1, the aleatoric term follows a recursive inequality:
at ≳EθEht−1

|Jy(ht−1)|2
F · Var(yt|ht−1; θ)

Under the representational expansion assumption, the expected Jacobian norm E[|Jy|2
F ] is large
(≫1). Furthermore, for most tokens in a generation, the predictive distribution p(yt|ht−1; θ) has
high entropy, meaning Var(yt|ht−1; θ) is also significant. The product of these two large factors is a
large positive quantity at each step t.
From Lemma 2, the epistemic term is bounded:
et ≤|Gt|2
F · Tr(Σθ) + ϵt
Under the peaked parameter posterior assumption, Tr(Σθ) is small. The sensitivity |Gt|2
F of the
expected hidden state to parameters depends on the model’s stability. In a well-trained robust
transformer with residual connections and layer normalization, the expected representations are often
stable with respect to small parameter perturbations, restricting |Gt|2
F to a smaller range. The error
term ϵt from the linear approximation is also small due to the peaked posterior. Consequently, et
remains bounded by a relatively small constant as t increases.
Therefore, for each step t, the recursive accumulation of sampling noise through expansive transfor-
mations causes at to become much larger than the bounded epistemic term et, leading to the stated
dominance: at ≫et.
C.4
DISCUSSION OF LIMITATIONS
It is important to emphasize that the above analysis is heuristic rather than a strict universal proof.
Several limitations apply. First, the variance decomposition in Eq. 5 is borrowed from Bayesian
17

## Page 18

deep learning for model outputs, but here it is applied to hidden states of autoregressive LLMs.
This approach is heuristic and requires us to define the distribution function of LLM parameters in
additional ways (although in practical applications, the parameters of LLMs are often given). Second,
Lemma 1 relies on first-order Taylor expansions; the neglected higher-order term ∆nonlin may be
non-negligible in some cases. Finally, the bound on epistemic uncertainty in Lemma 2 assumes a
concentrated posterior distribution and smooth sensitivity to parameters. While reasonable for large
pretrained models, this may not hold for small or under-regularized ones. Therefore, the dominance
of aleatoric over epistemic uncertainty in Proposition 1 should be viewed as a qualitative tendency
under certain conditions, not a universal law. We highlight these caveats to clarify that our theoretical
motivation is primarily heuristic, aiming to explain why single-pass-based hallucination is often
unreliable and provide interpretable insights for future works.
D
ADDITIONAL EXPERIMENTAL DATA
Here are the complete data from our ablation study on the number of generations, hidden-layer
selection strategy, and temperature. It offers a finer-grained view of how the information encoded in
hidden-layer representations interacts with these hyper-parameters.
N
Dataset
M1
M5
L1
L5
ES
PF
DSE
LNE
SE
10
TriviaQA
0.7877
0.7700
0.7809
0.7629
0.7802
0.6625
0.7758
0.6947
0.7750
SQuAD
0.7212
0.7217
0.7191
0.7190
0.7199
0.6594
0.7172
0.6505
0.7181
BioASQ
0.8447
0.8461
0.8481
0.8472
0.8433
0.7321
0.8437
0.4703
0.8425
NQ
0.7029
0.7038
0.7036
0.7049
0.7056
0.6584
0.6984
0.6495
0.7001
Average
0.7641
0.7604
0.7629
0.7585
0.7623
0.6781
0.7588
0.6163
0.7589
15
TriviaQA
0.7862
0.7902
0.787
0.7807
0.7825
0.6579
0.7813
0.7211
0.7768
SQuAD
0.7407
0.7407
0.7391
0.7395
0.7391
0.7148
0.7349
0.6070
0.7381
BioASQ
0.8715
0.8690
0.8684
0.8678
0.8682
0.7496
0.8648
0.4748
0.8615
NQ
0.7182
0.7166
0.7174
0.7188
0.7206
0.6598
0.7148
0.6823
0.7157
Average
0.7792
0.7791
0.7780
0.7767
0.7776
0.6955
0.7740
0.6213
0.7730
20
TriviaQA
0.7786
0.7780
0.7729
0.7702
0.7743
0.6636
0.7671
0.7180
0.7698
SQuAD
0.7596
0.7604
0.7599
0.7610
0.7579
0.6791
0.7588
0.6583
0.7581
BioASQ
0.8645
0.8625
0.8638
0.8620
0.8628
0.7517
0.8564
0.4570
0.8585
NQ
0.7277
0.7292
0.7274
0.7278
0.7284
0.6551
0.7243
0.6599
0.7256
Average
0.7826
0.7825
0.7810
0.7803
0.7809
0.6874
0.7767
0.6233
0.7780
Table 5: Ablation studies on Llama-2-7b-chat, where N refers to the number of generations, M1
refers to extracting the exact middle hidden layer for each answer in the ER method, M5 refers to
extracting the middle five layers, L1 refers to extracting the last layer, and L5 refers to extracting the
last five layers, ES denotes Eigenscore, PF denotes P_False, DSE denotes Discrete Semantic Entropy,
LNE denotes Length-Normalized Entropy, and SE denotes Semantic Entropy. All numbers in the
table are AUROC scores; values closer to 1 indicate stronger hallucination-detection ability
E
CASE STUDY
Below, we provide several specific examples from the main experiment. Please refer to Appendix F
for the discussion.
18

## Page 19

N
Dataset
M1
M5
L1
L5
ES
PF
DSE
LNE
SE
10
TriviaQA
0.7407
0.7516
0.7302
0.7495
0.7342
0.7379
0.7331
0.6930
0.7390
SQuAD
0.7259
0.7273
0.7241
0.7228
0.7239
0.6945
0.7381
0.6886
0.7350
BioASQ
0.8234
0.8277
0.8318
0.8324
0.8215
0.7980
0.7951
0.5610
0.7988
NQ
0.7284
0.7290
0.7286
0.7352
0.7270
0.6841
0.7234
0.6866
0.7258
Average
0.7546
0.7589
0.7537
0.7600
0.7517
0.7286
0.7474
0.6573
0.7497
15
TriviaQA
0.7763
0.7672
0.7681
0.7781
0.7654
0.7249
0.7590
0.6446
0.7577
SQuAD
0.7289
0.7289
0.7271
0.7285
0.7264
0.6747
0.7393
0.6823
0.7379
BioASQ
0.8358
0.8313
0.8341
0.8313
0.8328
0.7490
0.8224
0.4874
0.8148
NQ
0.7015
0.6996
0.7045
0.7054
0.7038
0.6902
0.6999
0.6452
0.6936
Average
0.7606
0.7560
0.7577
0.7601
0.7571
0.7097
0.7567
0.6149
0.7523
20
TriviaQA
0.7439
0.7553
0.7446
0.7339
0.7246
0.7294
0.7413
0.6953
0.7429
SQuAD
0.7331
0.7306
0.7293
0.7291
0.7299
0.6589
0.7382
0.6874
0.7360
BioASQ
0.8409
0.8454
0.8411
0.8461
0.8413
0.7864
0.8279
0.5939
0.8299
NQ
0.7178
0.7155
0.7193
0.7185
0.7175
0.6428
0.7149
0.6880
0.7120
Average
0.7589
0.7617
0.7586
0.7569
0.7533
0.7044
0.7556
0.6662
0.7552
Table 6: Ablation studies on Llama-2-13b-chat about the number of generations and hidden layer
vector extraction strategy
N
Dataset
M1
M5
L1
L5
ES
PF
DSE
LNE
SE
10
TriviaQA
0.7634
0.7650
0.7651
0.7637
0.7579
0.7239
0.7575
0.6519
0.7553
SQuAD
0.7208
0.7144
0.7073
0.7080
0.7120
0.6333
0.7227
0.6223
0.7238
BioASQ
0.8563
0.8584
0.8506
0.8475
0.8513
0.7173
0.8507
0.5955
0.8513
NQ
0.7658
0.7696
0.7614
0.7611
0.7627
0.7523
0.7678
0.6770
0.7662
Average
0.7766
0.7769
0.7711
0.7701
0.7710
0.7067
0.7747
0.6367
0.7742
15
TriviaQA
0.7727
0.7767
0.7727
0.7720
0.7706
0.7265
0.7660
0.7058
0.7676
SQuAD
0.6952
0.7064
0.6936
0.6907
0.6866
0.6047
0.7041
0.6529
0.6991
BioASQ
0.8586
0.8548
0.8539
0.8557
0.8557
0.6866
0.8551
0.6129
0.8560
NQ
0.7859
0.7861
0.7886
0.7896
0.7826
0.7717
0.7826
0.6975
0.7828
Average
0.7781
0.7810
0.7772
0.7770
0.7739
0.6974
0.7770
0.6673
0.7764
20
TriviaQA
0.7677
0.7672
0.7599
0.7688
0.7644
0.7663
0.7569
0.6861
0.7623
SQuAD
0.7433
0.7480
0.7413
0.7384
0.7371
0.5995
0.7559
0.6782
0.7533
BioASQ
0.8670
0.8662
0.8630
0.8631
0.8641
0.7594
0.8617
0.5854
0.8645
NQ
0.7754
0.7749
0.7723
0.7724
0.7695
0.7709
0.7736
0.7046
0.7712
Average
0.7884
0.7891
0.7841
0.7857
0.7838
0.7240
0.7870
0.6636
0.7878
Table 7: Ablation studies on Mistral-7B-v0.1 about the number of generations and hidden layer vector
extraction strategy
E.1
TRIVIAQA DATASET ON LLAMA-2-7B-CHAT
Question: Who was the first man sent into space, in 1961?
Correct Answer: [’gagarin’]
LLM’s Answer: yuri gagarin
Accuracy: 1.0
Sampled Responses: [’yuri gagarin’, ’yuri gagarin’, ’yuri gagarin’, ’yuri gagarin’, ’yuri
gagarin’, ’yuri gagarin’, ’yuri gagarin’, ’yuri gagarin’, ’yuri gagarin’]
Singular Values: [76.35235595703125, 2.6761877219491637e-14, 1.1108339471383637e-
15, 2.146262724714404e-30, 3.9155441760212304e-31, 0.0, 0.0, 0.0, 0.0, 0.0]
Effective Rank: 1.0000000000000002
Eigenscore: -61.72965268471336
19

## Page 20

t
Dataset
M1
M5
L1
L5
ES
PF
DSE
LNE
SE
0.1
TriviaQA
0.6514
0.6660
0.6782
0.6503
0.6200
0.6695
0.6263
0.5391
0.6466
SQuAD
0.6451
0.6466
0.6118
0.5971
0.6038
0.6413
0.6125
0.5506
0.6333
BioASQ
0.7336
0.6834
0.7268
0.7416
0.6923
0.7720
0.7085
0.4401
0.7352
NQ
0.6771
0.6665
0.6841
0.6664
0.6673
0.7308
0.6700
0.6231
0.6822
Average
0.6768
0.6656
0.6752
0.6639
0.6459
0.7034
0.6543
0.5382
0.6743
0.5
TriviaQA
0.7438
0.7628
0.7518
0.7492
0.7350
0.6806
0.7280
0.5880
0.7154
SQuAD
0.7364
0.7391
0.7392
0.7401
0.7330
0.6551
0.7308
0.6104
0.7318
BioASQ
0.8432
0.8374
0.8449
0.8452
0.8378
0.7320
0.8355
0.5167
0.8293
NQ
0.7679
0.7671
0.7724
0.7678
0.7641
0.7516
0.7643
0.6356
0.7629
Average
0.7728
0.7766
0.7771
0.7756
0.7675
0.7048
0.7647
0.5877
0.7599
1.0
TriviaQA
0.7634
0.7650
0.7651
0.7637
0.7579
0.7239
0.7575
0.6519
0.7553
SQuAD
0.7208
0.7144
0.7073
0.7080
0.7120
0.6333
0.7227
0.6223
0.7238
BioASQ
0.8563
0.8584
0.8506
0.8475
0.8513
0.7173
0.8507
0.5955
0.8513
NQ
0.7658
0.7696
0.7614
0.7611
0.7627
0.7523
0.7678
0.6770
0.7662
Average
0.7766
0.7769
0.7711
0.7701
0.7710
0.7067
0.7747
0.6367
0.7742
2.0
TriviaQA
0.6679
0.6685
0.6651
0.6662
0.6545
0.7084
0.6643
0.6234
0.6666
SQuAD
0.5754
0.5502
0.5593
0.5407
0.5698
0.5603
0.5213
0.5633
0.5592
BioASQ
0.7280
0.7329
0.7106
0.7238
0.6998
0.6867
0.7243
0.5534
0.7244
NQ
0.6034
0.6031
0.6118
0.6069
0.5788
0.7006
0.5996
0.5580
0.5906
Average
0.6437
0.6387
0.6367
0.6344
0.6257
0.6640
0.6274
0.5745
0.6352
Table 8: Ablation studies on Mistral-7B-v0.1 about temperature, where t denotes temperature
Question: September 9, 1969 saw what made an official language of Canada?
Correct Answer: [’french’]
LLM’s Answer: french
Accuracy: 1.0
Sampled Responses: [’Quebec’, ’french’, ’french’, ’Quebec’, ’constitution’, ’Quebec’,
’french’, ’french’, ’french’]
Singular Values:
[60.25969314575195, 41.63536071777344, 24.346080780029297,
1.6957731741170864e-14,
5.443248018391789e-15,
1.198083634661477e-15,
8.686304874642721e-16,
1.4343240353264575e-16,
8.151879937535818e-32,
2.8225723498131095e-32]
Effective Rank: 2.818618681563689
Eigenscore: -51.25870849393463
Question: In which part of the human body would you find the Sphenoid bone?
Correct Answer: [’skull’]
LLM’s Answer: brain
Accuracy: 0.0
Sampled Responses: [’head’, ’brain’, ’skull’, ’skull’, ’skull’, ’head’, ’brain’, ’skull’, ’skull’]
Singular Values:
[58.72682571411133, 34.81084442138672, 29.228200912475586,
7.700909307212667e-15,
7.15308397231237e-15,
2.269143032179147e-15,
3.835939579384829e-16,
3.338613852606579e-30,
1.4536969531393071e-31,
7.850294372204882e-33]
Effective Rank: 2.8627889069115606
Eigenscore: -51.30254888163167
20

## Page 21

E.2
SQUAD DATASET ON MISTRAL-7B-V0.1
Question: Who did Frick remove from the police force?
Correct Answer: [’anyone he suspected of being a republican’]
LLM’s Answer: the corrupt
Accuracy: 0.0
Sampled Responses: [’the corrupt’, ’the corrupt’, ’the corrupt’, ’the corrupt’, ’the corrupt’,
’the corrupt’, ’the corrupt’, ’the corrupt’, ’the corrupt’]
Singular Values: [1139.242431640625, 3.5298562575496184e-13, 1.458547028271827e-
14,
3.75700543160097e-29,
3.410722249588845e-30,
2.802596928649634e-45,
1.401298464324817e-45, 0.0, 0.0, 0.0]
Effective Rank: 1.0000000000000004
Eigenscore: -64.89193565768241
Question: Who verbally attacked Steve Stone?
Correct Answer: [’Kent Mercker’]
LLM’s Answer: Steve Stone
Accuracy: 0.0
Sampled Responses: [’Joe Morgan’, ’his boss’, ’John Kruk’, ’Steve Stone’, ’Steve Stone’,
’his boss’, ’John Madden’, ’his boss’, ’his boss’]
Singular Values:
[1180.1435546875,
552.9315185546875,
154.78468322753906,
16.52665138244629, 6.953697204589844, 1.193859735463404e-13, 1.553780128377754e-
14, 6.809435505831249e-16, 1.553959034971104e-17, 1.1361135973578757e-30]
Effective Rank: 2.513272471125542
Eigenscore: -61.32968780733988
Question: In 2002 what act granted full British citizenship to the citizens of the islands?
Correct Answer: [’British Overseas Territories Act 2002’]
LLM’s Answer: British Overseas Territories Act
Accuracy: 1.0
Sampled Responses: [’British Overseas Territories Act’, ’British Overseas Territories
Act’, ’British Overseas Territories Act, ’British Overseas Territories Act’, ’British Overseas
Territories Act’, ’British Nationality Act’, ’British Overseas Territories Act’, ’British Overseas
Territories Act’, ’British Overseas Territories Act’]
Singular Values: [1153.8829345703125, 328.6450500488281, 7.30688702902868e-14,
2.118430780068855e-30, 1.2843073833195104e-33, 0.0, 0.0, 0.0, 0.0, 0.0]
Effective Rank: 1.6972759552279857
Eigenscore: -63.12688798174471
21

## Page 22

E.3
BIOASQ DATASET ON LLAMA-2-13B-CHAT
Question: Which are the thyroid hormone analogs utilized in human studies?
Correct Answer: [’TRIAC’]
LLM’s Answer: Liothyronine (T3) and levothyroxine (T4)
Accuracy: 0.0
Sampled Responses: [’triiodothyronine (T3) and levothyroxine (T4)’, ’Liothyronine (T3)
and levothyroxine (T4)’, ’Liothyronine (T3) and levothyroxine (T4)’, ’Liothyronine (T3) and
levothyroxine (T4)’, ’Liothyronine (T3) and levothyroxine (T4)’, ’Liothyronine (T3) and
levothyroxine (T4)’, ’liothyronine and levothyroxine’, ’Liothyronine (T3) and levothyroxine
(T4)’, ’liothyronine and levothyroxine’]
Singular Values:
[218.68751525878906, 62.00069046020508, 15.469388961791992,
3.4699248902941024e-15,
3.3113897569630145e-15,
4.50245604514668e-31,
1.978491468987644e-31, 2.0052295279776766e-32, 0.0, 0.0]
Effective Rank: 2.0248367530554368
Eigenscore: -53.52278438073435
Question: Which genes are involved in patient response to warfarin?
Correct Answer: [’CYP2C9’, ’VKORC1’, ’ORM1’, ’CYP4F2’, ’EPHX1’, ’CYP2C18’,
’CYP2C19’, ’CYP3A5’, ’protein S’, ’clotting factor V’, ’PROC’, ’GGCX’]
LLM’s Answer: VKORC1 and CYP2C9
Accuracy: 1.0
Sampled Responses: [’VKORC1 and CYP2C9’, ’VKORC1 and CYP2C9’, ’CYP2C9
and VKORC1’, ’VKORC1 and CYP2C9’, ’CYP2C9, VKORC1, and CYP3A4’, ’CYP2C9,
VKORC1, and CYP3A4’, ’VKORC1 and CYP2C9’, ’VKORC1 and CYP2C9’, ’CYP2C9
and VKORC1’]
Singular Values:
[217.03753662109375, 100.93067932128906, 6.290593147277832,
4.6974177206424855e-15,
2.6008424048733518e-15,
1.8994920306458578e-16,
1.1646331124343226e-16, 8.127036452178364e-32, 1.1876486869033015e-33, 0.0]
Effective Rank: 2.0309090152250144
Eigenscore: -53.3001540843245
Question: List symptoms of the IFAP syndrome.
Correct Answer: [’follicular ichthyosis’, ’atrichia’, ’photophobia’]
LLM’s Answer: intellectual disability, fibrosis, alopecia, and pigmentary changes.
Accuracy: 0.0
Sampled Responses: [’ichthyosis, facial dysmorphism, arthrogryposis, and pul’, ’intellectual
disability, fibrosis, alopecia, and pigmentary changes’, ’Intellectual disability, FGFR3 muta-
tion, cardiac defects, and pul’, ’intellectual disability, fibrosis, ataxia, and pulmonary disease’,
’ichthyosis, ataxia, pulmonary dysfunction, and immunodefic’, ’intellectual disability, fibrosis,
ataxia, and seizures’, ’intellectual disability, fibrosis, alopecia, and pigmentation abnormali-
ties’, ’ichthyosis, facial dysmorphism, arthrogryposis, and pul’, ’intellectual disability, fragile
bones, anxiety, and pulmonary issues’]
Singular Values:
[122.30469512939453, 114.34846496582031, 82.5096206665039,
71.02500915527344, 61.05084991455078, 55.45585250854492, 55.206974029541016,
45.03531265258789, 5.794018268585205, 1.2034170623171584e-14]
Effective Rank: 7.804594598381604
Eigenscore: -22.963745969679774
22

## Page 23

E.4
NQ DATASET ON MISTRAL-7B-V0.1
Question: last time east won nba all star game
Correct Answer: [’2014’]
LLM’s Answer: 2006
Accuracy: 0.0
Sampled Responses: [’2004’, ’2001’, ’2001’, ’2013’, ’2013’, ’2013’, ’2013’, ’2004’, ’2013’]
Singular Values:
[1097.43212890625,
585.4111938476562,
355.2393798828125,
2.6302599906921387,
1.0910121140935217e-13,
7.080334175945183e-14,
6.622470177493692e-14,
9.815065427076775e-15,
4.458403975289517e-16,
3.7646645390169164e-30]
Effective Rank: 2.731143238800789
Eigenscore: -61.13279244771307
Question: who was the temple of vesta built for?
Correct Answer: [’Vesta’]
LLM’s Answer: Vesta
Accuracy: 1.0
Sampled Responses: [’Vesta’, ’the goddess vesta’, ’Vesta’, ’the goddess vesta’, ’the goddess
vesta’, ’Vesta’, ’Vesta’, ’Vesta’, ’Vesta’]
Singular Values:
[1252.1671752929688, 372.3441772460938, 2.068291691343857e-
13,
1.5302303423957858e-13,
1.352673246172607e-14,
2.020799899278259e-29,
9.934199431454563e-30, 4.8484518987236446e-30, 0.0, 0.0]
Effective Rank: 1.7131135330533338
Eigenscore: -62.060661269023285
Question: what percentage of the world died in ww2?
Correct Answer: [’about 3%’]
LLM’s Answer: 3%
Accuracy: 1.0
Sampled Responses: [’3%’, ’3%’, ’3%’, ’3%’, ’3%’, ’3%’, ’3%’, ’3%’, ’3%’]
Singular Values: [1136.5360107421875, 3.5605334869695526e-13, 1.574521085343991e-
14, 2.644209128265101e-29, 1.6524366075384805e-30, 2.802596928649634e-45, 0.0, 0.0,
0.0, 0.0]
Effective Rank: 1.0000000000000004
Eigenscore: -64.94262853905146
F
FURTHER DISCUSSION ON THE EXPERIMENTS
F.1
RELATIONSHIP BETWEEN HALLUCINATION AND UNCERTAINTY
From the above examples, we observe that hallucinations in LLMs are generally positively correlated
with their uncertainty. However, counterexamples also exist, such as hallucinated answers with high
confidence and correct answers with low confidence. We argue that this phenomenon mainly stems
from the diverse causes of hallucination: (i) erroneous internal knowledge, (ii) insufficient internal
knowledge and reasoning capability, (iii) excessive stochasticity, and (iv) lack of faithfulness during
the generation process. Hallucinations caused by the latter three can often be detected via uncertainty
estimation. In contrast, hallucinations due to erroneous internal knowledge are usually associated
with low uncertainty, making them difficult to capture with uncertainty-based methods. Therefore,
we recommend combining our uncertainty-based approach with knowledge editing and retrieval
augmentation techniques to eliminate internal knowledge errors, thereby reducing highly consistent
hallucinations and further enhancing AI trustworthiness.
23

## Page 24

F.2
RELIABILITY AND LIMITATIONS OF ROUGE-L FOR HALLUCINATION ANNOTATION
Based on the case analysis, we find that ROUGE-L is a reliable proxy for hallucination annotation in
our experiments, particularly when the dataset contains abundant correct answers and these answers
are relatively short. This is because ROUGE-L, by computing the longest common subsequence,
effectively checks whether key tokens from the ground-truth answer appear in the model output,
similar to keyword-based grading in short-answer exams. In contrast, semantic vector similarity is
more prone to misjudgment, as small variations such as additional predicates or modifiers can cause
embeddings to be judged as semantically different. Moreover, some questions in our dataset have
multiple semantically distinct correct answers, and LLMs may generate multiple answers simulta-
neously, making semantic similarity metrics less suitable. Another potentially effective annotation
method is to determine whether the LLM output and the ground truth are in a bidirectional entailment
relationship within context. However, this requires introducing an auxiliary NLI model, which greatly
increases annotation complexity. In summary, ROUGE-L aligns well with the characteristics and
needs of our datasets and experiments, providing satisfactory annotation reliability. Nevertheless, for
more complex scenarios such as multi-turn QA or long-form generation, we recommend adopting
more robust hallucination annotation methods.
F.3
TIME COMPLEXITY ANALYSIS OF EFFECTIVE RANK
Although our method requires performing singular value decomposition (SVD) on a matrix composed
of multiple high-dimensional vectors, modern libraries such as numpy.linalg.svd() make
the computation highly efficient. The runtime analysis in the main text (Table 2) shows that the
additional computation time of our Effective Rank method is negligible compared to the time required
for answer generation. Moreover, its time cost is substantially lower than that of P_False (which
introduces extra prompt-based verification) and Semantic Entropy (which requires semantic analysis).
For an m × n matrix with m < n, the time complexity of SVD is O(m2n). In our main experiments,
we compute matrices of size 10 × 4096 or 50 × 4096, where such complexity is entirely acceptable
in practice.
24
