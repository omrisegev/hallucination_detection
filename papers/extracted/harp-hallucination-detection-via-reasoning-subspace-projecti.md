---
source_pdf: papers/HARP Hallucination Detection via Reasoning Subspace Projection.pdf
slug: harp-hallucination-detection-via-reasoning-subspace-projecti
pages: 19
extracted_on: 2026-07-13
---

# HARP Hallucination Detection via Reasoning Subspace Projection

## Page 1

HARP: HALLUCINATION DETECTION VIA REASONING
SUBSPACE PROJECTION
Junjie Hu, Gang Tu∗, ShengYu Cheng, Jinxin Li, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan
School of Computer Science and Technology
Huazhong University of Science and Technology
Wuhan, China
{hujunjie,tugang}@hust.edu.cn
ABSTRACT
Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical
decision-making. Although existing hallucination detection methods have improved accuracy, they
still struggle with disentangling semantic and reasoning information and maintaining robustness.
To address these challenges, we propose HARP (HAllucination detection via Reasoning subspace
Projection), a novel hallucination detection framework. HARP establishes that the hidden state space
of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace,
where the former encodes linguistic expression and the latter captures internal reasoning processes.
Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by
applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the
semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis
vectors of the reasoning subspace, and the resulting projections are then used as input features for
hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the
feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness.
Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection
performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous
best method by 7.5%.
Keywords Hallucination detection · Subspace · Projection · SVD
1
Introduction
Reasoning
Answer
Transform 
to 
Semantic
Embedding
Decoder
Unembedding
× 𝑙
Reasoning
Answer
Question
Question
Humans
LLMs
Transform 
to 
Semantic
Figure 1: Comparison of the “Reasoning →Ex-
pression” behavior between humans and LLMs
Large Language Models (LLMs) have recently demonstrated remark-
able generative capabilities and broad applicability across various
natural language processing tasks [1, 2, 3]. However, hallucina-
tions—i.e., model-generated information inconsistent with objec-
tive facts—remain a major obstacle to their deployment in critical
decision-making scenarios [4, 5]. Consequently, efficiently and
accurately detecting hallucinations during LLMs generation has
become a pressing challenge.
From a cognitive perspective, the hallucination behavior of LLMs
is to some extent similar to human’s “nonsense” behavior. When
answering complex questions, humans typically follow a “Reasoning
→Expression” process: they first perform internal reasoning and
then express part of the thought outcomes in language [6]. Therefore,
although assessing the veracity of the answer is challenging when
based solely on linguistic output, it can be substantially improved
∗Corresponding Author
arXiv:2509.11536v2  [cs.CL]  5 Dec 2025

## Page 2

HARP: Hallucination Detection via Reasoning Subspace Projection
Decoder
𝑡𝑔𝑒𝑛∈𝒯
× 𝑙
ℎ0 ∈𝒮𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
ℎ𝑙∈𝒮𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐⊕𝒮𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
ℎ𝑙,𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
ℎ𝑙,𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
+
=
Reasoning
Transform 
to 
Semantic
𝑊𝑢𝑛𝑒𝑚𝑏
𝑺𝒄𝒐𝒓𝒆
Calculate 
Hallucination
Score
𝑔𝜃
Where is the capital of the United States?
The capital of the United States is Washington !
The capital of the United States is   Shanghai   !
<0.01 <0.01
……
<0.01
0.73
<0.01
<0.01
0.02
𝑔𝜃
𝑔𝜃
Accept
Hallucination
Figure 2: Illustration of the proposed HARP framework for hallucination detection. HARP separates the reasoning
information hl,Reasoning from the hidden state hl to compute token-level hallucination scores, with the maximum score
taken as the hallucination score of the entire response.
by observing the complete reasoning process [7]. By analogy, achieving high-precision hallucination detection in LLMs
requires placing greater emphasis on the reasoning information encoded within the hidden states, rather than primarily
on the semantic content of the outputs.
Inspired by this cognitive insight, we propose a novel hallucination detection framework, HARP (HAllucination
detection via Reasoning subspace Projection). Specifically, HARP decomposes the hidden state space into a direct sum
of the semantic subspace and the reasoning subspace. The semantic subspace captures the linguistic information of
the generated content, while the reasoning subspace reveals the model’s internal reasoning process. As illustrated in
Figure 1, comparing humans and LLMs “Reasoning →Expression” behaviors reveals that LLMs discard reasoning
information in the Unembedding layer while compressing semantic information into generated tokens. This suggests
that the Unembedding layer inherently distinguishes between semantic and reasoning information. Based on this, we
perform Singular Value Decomposition (SVD) on the parameter matrix of the Unembedding layer to identify the basis
vectors of the semantic subspace, which dominates token prediction, as well as those of the reasoning subspace, which
is orthogonal to the semantic subspace.
Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace and uses the resulting projections
as input features for hallucination detection in LLMs. Since the reasoning subspace basis vectors account for only about
5% of the hidden state dimension, the input features are highly concentrated in reasoning information while largely
eliminating noise. This allows HARP to achieve strong robustness while maintaining high detection accuracy. The
main contributions of this work are:
• We establish that the hidden state space of LLMs can be decomposed into a direct sum structure composed of
a semantic subspace and a reasoning subspace.
• We verify that the Unembedding layer has the capability to distinguish between the semantic subspace and the
reasoning subspace. Furthermore, by performing SVD on the parameters of the Unembedding layer, the basis
vectors that span the semantic subspace and the reasoning subspace are identified.
• We introduce a novel approach that explicitly constructs input features by projecting hidden states onto the
basis vectors of the reasoning subspace. This projection drastically reduces the feature dimensionality to about
5% of the original, suppresses most noise, and achieves highly accurate hallucination detection in LLMs.
2
Related Work
Mechanistic interpretability of LLMs. Research on mechanistic interpretability mainly focuses on two aspects: model
parameters and hidden states. For the former, several works analyze weight matrices to uncover structural properties
and interactions among modules. For instance, Merullo et al.[8] and Cheng et al.[9] employ SVD to characterize
attention-head structures and investigate their roles in downstream tasks. However, such approaches remain largely
at the structural level, offering limited semantic interpretability. To address this, recent work has shifted toward
analyzing hidden states directly, probing the predictive relationship between intermediate representations and model
outputs[10, 11, 12, 13, 14].
Hallucination detection. The success of probing methods has motivated researchers to adopt similar ideas in
hallucination detection[15, 16, 17]. For instance, HaloScope[18] leverages unlabeled embeddings and applies SVD to
identify key subspace directions, followed by probing to link these directions to hallucinations. Yet, probing-based
2

## Page 3

HARP: Hallucination Detection via Reasoning Subspace Projection
methods often rely on predefined supervised labels, making them less generalizable when feature dimensions are large
or category priors are incomplete. Another line of work approaches hallucination detection from the perspective of
output consistency. EigenScore[19] quantifies semantic agreement through covariance eigenvalues, while Farquhar et
al.[20] utilize clustering and semantic entropy to detect hallucinations. These methods are effective in practice but may
suffer from misclassification due to their inability to exploit internal reasoning information.
Different from these approaches, our method explicitly separates semantic and reasoning subspaces, and projects hidden
states onto the basis vectors of the reasoning subspace to construct compact and interpretable features for hallucination
detection.
3
Preliminaries
In this section, we first formulate a mathematical model to characterize the hallucination behavior of LLMs. Then, we
analyze how the hidden state space evolves across decoder layers during generation, and subsequently decompose it into
the direct sum of the semantic subspace and the reasoning subspace. This theoretical framework forms the foundation
of HARP and provides essential support for hallucination detection via reasoning subspace projection.
3.1
Mathematical Modeling of LLMs’ Hallucination
To model LLMs’ hallucination mathematically, we first define the knowledge set known to the LLMs. Given an input
sequence x and its reference answer y∗, the LLMs generate multiple responses γ = {y1, y2, . . . , ys} for x. If any
generated response closely matches the reference answer, the knowledge about x is considered known to the LLMs,
denoted as known(x) = 1. Formally:
known(x) =
1,
∃y ∈γ, sim(y, y∗) > λ
0,
otherwise
(1)
where sim(y, y∗) measures the similarity between y and y∗, and λ is a similarity threshold. Let Xknown = {x |
known(x) = 1} denote the set of all inputs whose knowledge is known to the LLMs. For each x ∈Xknown, let
y = LLMs(x) denote the response generated by the LLMs. The hallucination indicator G(y | x) is then defined as:
G(y | x) =
1,
sim(y, y∗) ≤λ
0,
otherwise
(2)
When G(y | x) = 1, the QA pair [x, y] exhibits hallucination.
3.2
Direct Sum Decomposition of Hidden State Space
Let the token vocabulary be T . For LLMs with l decoder layers, an input token t ∈T is mapped by the embedding
layer to an initial hidden state h0 containing purely semantic information. As the hidden states propagate through
successive layers, semantic and reasoning information are progressively integrated into their representations. Finally,
the Unembedding layer projects only the semantic component to generate the output token tgen ∈T . Thus, the final
hidden state hl simultaneously encodes: (1) Semantic prediction information: To accurately generate the next token,
hl must retain sufficient semantic features. These features are primarily captured by the parameter matrix Wunemb of
the Unembedding layer and play a dominant role in predicting the next token. (2) Reasoning trajectory information:
To support multi-step reasoning and intermediate state computation, hl also encodes intermediate reasoning information
that does not directly affect the output. This information is typically not explicitly captured by Wunemb and exerts
minimal influence on the final output.
Denote the hidden state space at layer l as Hl. To disentangle these two signals, we decompose Hl into the direct sum
of two orthogonal subspaces:
Hl = SSemantic ⊕SReasoning
(3)
where SSemantic and SReasoning represent the semantic and reasoning subspaces, respectively. The final hidden state
hl ∈Hl is projected to token logits by the Unembedding layer:
logits = Wunemb · hl
(4)
where Wunemb denotes the Unembedding parameters. Let hl,Semantic and hl,Reasoning denote the components of hl
in the semantic and reasoning subspaces, with hl,Semantic exerting primary influence on the logits for token prediction,
while hl,Reasoning encodes the model’s reasoning processes.
3

## Page 4

HARP: Hallucination Detection via Reasoning Subspace Projection
To empirically validate the existence and functional role of the reasoning subspace, we design a Reasoning Patch
experiment in Appendix E. This experiment demonstrates that the reasoning subspace SReasoning indeed captures
critical intermediate reasoning information by showing that patching reasoning components from correct solutions can
effectively rectify erroneous reasoning trajectories while preserving semantic coherence.
4
Method
In this section, we detail the proposed HARP framework for hallucination detection, as illustrated in Figure 2. First, in
subsection 4.1, we validate the Unembedding layer’s capability to effectively disentangle the semantic and reasoning
subspaces. Then, in subsection 4.2 and subsection 4.3, we present a practical strategy for subspace decomposition.
Finally, in subsection 4.4, we introduce the HARP algorithm, which performs hallucination detection based on reasoning
subspace projection.
4.1
Subspace Decomposer — Unembedding Layer
Embedding
𝑡∈𝒯
Decoder
Unembedding
𝑡𝑔𝑒𝑛∈𝒯
× 𝑙
ℎ0 ∈𝒮𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
ℎ𝑙∈𝒮𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐⊕𝒮𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
ℎ𝑙,𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
ℎ𝑙,𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
+
=
×
Reasoning
Figure 3: Flow of semantic and rea-
soning information within LLMs
hidden states.
As shown in Figure 3, during token generation, the Unembedding layer of LLMs
compresses only the semantic information hl,Semantic in hidden states into the
generated tokens, filtering out the reasoning information hl,Reasoning used in
intermediate computations. Therefore, by analyzing the basis vectors that interact
with the Unembedding layer parameters Wunemb, we can determine the mathe-
matical representations of the semantic subspace and its orthogonal reasoning
subspace.
Based on the properties of the semantic and reasoning subspaces, their interactions
with Wunemb can be defined as:
Wunemb · SSemantic ≈Wunemb · Hl
(5)
Wunemb · SReasoning ≈0
(6)
In other words, SSemantic aligns with the principal acting directions of Wunemb,
while the orthogonal SReasoning contributes negligibly to prediction scores. In
subsection 5.3, we demonstrate the validity of our definitions for these subspace
properties, laying the foundation for subsequently identifying the subspace basis
vectors.
4.2
Determination of Subspace Basis Vectors via SVD
Given that the Unembedding layer can filter reasoning information, we first perform SVD on its parameter matrix
Wunemb. By analyzing which hidden state components interact with Wunemb, we identify the basis vectors of the
semantic and reasoning subspaces. As shown in Equation 7, we decompose Wunemb via SVD:
Wunemb = UΣV ⊤=
Xd
i=1 uiσiv⊤
i
(7)
where U ∈R∥T ∥×∥T ∥, Σ ∈R∥T ∥×d, V ∈Rd×d, ∥ui∥= ∥vi∥= 1, and the singular values in Σ are sorted in
descending order σ1 ≥σ2 ≥· · · ≥σk > σk+1 = σk+2 = · · · = σd = 0.
For any hidden state h = Pd
i=1 aivi ∈Rd, its interaction with Wunemb is expressed as:
Wunemb · h =
Xd
i=1 uiσiv⊤
i · aivi =
Xd
i=1(σiai)ui
(8)
Since the vectors ui are mutually orthogonal, it follows that Wunemb · h = 0 if and only if Pd
i=1 |σiai| = 0, in
which case the vector h is filtered out by the Unembedding layer. In other words, h belongs to the reasoning subspace
SReasoning if and only if all singular values corresponding to non-zero ai vanish. Accordingly, we define an orthogonal
basis for the reasoning subspace as VR = {vi | σi = 0}, while the remaining directions VS = {vi | σi > 0} constitute
the semantic subspace SSemantic. Since σi>k = 0, the semantic and reasoning subspaces can be expressed as:
SSemantic = Span ({v1, v2, . . . , vk})
(9)
SReasoning = Span ({vk+1, vk+2, . . . , vd})
(10)
4

## Page 5

HARP: Hallucination Detection via Reasoning Subspace Projection
Let ai = v⊤
i hl denote the projection coefficients of the hidden state hl onto the basis vectors. Then the components
in the semantic and reasoning subspaces are hl,Semantic = Pk
i=1 aivi and hl,Reasoning = Pd
i=k+1 aivi, respectively,
with interactions with Wunemb given by:
Wunemb · hl,Semantic =
Xk
i=1 σi(aiui) = Wunemb · hl
(11)
Wunemb · hl,Reasoning =
Xd
i=k+1 σi(aiui) = 0
(12)
This partitioning of the hidden state space aligns precisely with the definitions of semantic and reasoning subspaces in
Equation 5 and Equation 6, and provides a theoretical basis for constructing low-rank approximation-based subspaces
in real models.
4.3
Construction of Semantic and Reasoning Subspaces via Low-Rank Approximation
While the method described in subsection 4.2 can ideally construct the semantic and reasoning subspaces, in practice,
the condition σ = 0 for singular values rarely holds. To address this, we perform a rank-k approximation of Wunemb,
extracting the k most representative semantic directions from its row space to define the semantic subspace under
realistic conditions, and determine the reasoning subspace using orthogonal relationships.
Specifically, based on Equation 7, for any k < rank(Wunemb), the Eckart–Young–Mirsky theorem [21, 22] gives the
best rank-k approximation Wk of Wunemb under the Frobenius norm as:
Wk = arg min
rank(A)≤k
∥Wunemb −A∥F =
Xk
i=1 uiσiv⊤
i
(13)
To ensure that this approximation does not significantly degrade prediction accuracy, the following information-
preservation condition should hold:
∥Wunemb −Wk∥F =
rXd
i=k+1 σ2
i ≪
rXk
i=1 σ2
i
(14)
This condition implies that Wk retains the majority of Wunemb’s information in the Frobenius norm, i.e., the first k
singular values account for most of the total energy.
Figure 4a illustrates the singular value distribution of the Unembedding layer parameters. We observe that the trailing
5% of singular values are markedly smaller than the others, and the information loss associated with these minor
singular values can be safely ignored. Accordingly, we set k = d × 95%. By analyzing Wk and incorporating it into
Equation 9 and Equation 10, we derive the corresponding subspace representations. Denoting the basis of the reasoning
subspace as VR = [vk+1, vk+2, . . . , vd] ∈Rd×(d−k), the projection of hidden states hl onto the reasoning subspace is:
projR (hl) = V ⊤
R · hl
(15)
In subsection 5.3, we experimentally demonstrate that replacing Wunemb with Wk in the token prediction task introduces
negligible error. This finding provides the basis for subsequently using projR (hl) as the input feature to construct the
hallucination detector.
4.4
Hallucination Detection via Reasoning subspace Projection
As shown in Figure 4b, universal representations of hidden states are extracted from different layers of the LLMs and
projected onto the basis V = [VS, VR]. We observe that shallow hidden states primarily enhance information in the
semantic subspace, while deep hidden states exhibit stronger features in the reasoning subspace. This observation
is consistent with our definitions of the two subspaces. Based on this, we propose a novel hallucination detection
framework—HARP, which detects hallucinations using projections of hidden states onto the reasoning subspace.
During training, HARP employs a beam search strategy to generate multiple candidate answers γ = {y1, y2, . . . , ys}
for a given question x, and annotates whether each candidate contains hallucinations. For a QA pair [x, y] composed of
n tokens, HARP computes the projection of each token’s hidden state onto the reasoning subspace and calculates its
hallucination score. The maximum score among all tokens is taken as the hallucination score of the QA pair:
gθ (y|x) = max
1≤i≤n gθ

projR

h(i)
l

(16)
5

## Page 6

HARP: Hallucination Detection via Reasoning Subspace Projection
0
1000
2000
3000
4000
100
101
102
0
1000
2000
3000
4000
100
101
102
Singular values
Index
Qwen-2.5-7B-Instruct
LLaMA-3.1-8B
Singular values
Index
(a)
Semantic
Reasoning
-0.4
-0.2
0.0
0.2
0.4
0.6
-0.4
-0.2
0.0
0.2
0.4
0.6
Projection
Layer 01
Layer 02
Layer 03
Projection
Layer l - 2
Layer l - 1
Layer l
(b)
Figure 4: (a) Singular value distributions of Wunemb after SVD, with hidden state dimensions of 3584 for Qwen-2.5-
7B-Instruct and 4096 for LLaMA-3.1-8B. (b) Projections of hidden states onto the basis vectors of the semantic and
reasoning subspaces across layers, where the first row shows the first three layers and the second row shows the last
three layers. Further details are provided in Appendix B.
where θ denotes the parameters of the hallucination detector. gθ

projR

h(i)
l

represents the hallucination score
of the i-th token, and gθ (y|x) ∈[0, 1] is the score for the entire QA pair. We optimize the detector using the Binary
Cross-Entropy Loss [23]:
L = −flag · log(gθ) −(1 −flag) · log(1 −gθ)
(17)
where flag ∈{0, 1} indicates whether the QA pair [x, y] contains hallucinations. Minimizing this loss trains a
hallucination detector bG:
bG(y|x) = I [gθ(y|x) > α]
(18)
where α ∈[0, 1] is the detection threshold. When bG(y|x) = 1, the QA pair is considered hallucinated. Beam search
is used only during training to construct diverse supervision samples, whereas during testing, bG relies solely on the
projection of a single sampled answer onto SReasoning.
As shown in Figure 2, for the question “Where is the capital of the United States?”, the hallucinated answer “The capital
of the United States is Shanghai!” assigns a hallucination score of 0.73 to the token “Shanghai”, whereas all tokens in
the correct answer “The capital of the United States is Washington!” have scores below 0.01. This demonstrates the
effectiveness of bG.
5
Experiments
In this section, we first describe the experimental setup and demonstrate HARP’s advantages over other hallucination
detection methods across multiple models and datasets. We then analyze the validity of our proposed direct-sum
decomposition of the hidden state space and the necessity of the projection operation, followed by an evaluation of the
detection performance under varying reasoning subspace dimensions and hallucination score thresholds. Finally, we
discuss HARP’s cross-dataset generalization capability.
5.1
Experimental Setup
Datasets and models. Our experiments cover four generative question answering (QA) tasks, including three open-
domain dialogue QA datasets—NQ Open[24], TruthfulQA[25] (generation task), and TriviaQA[26]—and one reading
comprehension dataset, TyDiQA-GP (English)[27]. To assess the effectiveness and generality of our proposed
framework, we conduct evaluations using two widely adopted open-source foundation models: Qwen-2.5-7B-Instruct[1]
and LLaMA-3.1-8B[2]. More dataset and inference details are provided in Appendix A.
Evaluation Metrics. AUROC (area under the ROC curve) is employed as the primary evaluation metric. AUROC
measures a binary classifier’s ability to distinguish positive and negative samples across different thresholds, ranging
from 0 to 1, with higher values indicating stronger discriminative power. AUROC equal to 1 indicates perfect
classification, while a value of 0.5 corresponds to random guessing.
Baseline Methods.
HARP is compared with several mainstream hallucination detection methods, including
Perplexity[28], LN-Entropy[29], Semantic Entropy[20], Lexical Similarity[30], EigenScore[19], and HaloScope[18].
6

## Page 7

HARP: Hallucination Detection via Reasoning Subspace Projection
Table 1: Main result. Comparison of different methods on hallucination detection performance across multiple datasets.
All values are AUROC percentages. “Single” indicates whether multiple samplings are required for hallucination
detection.
Models
Methods
Single
NQ Open
TruthfulQA
TriviaQA
TyDiQA
Qwen-2.5-7B-Instruct
Perplexity
✓
76.5
64.4
83.1
30.5
LN-Entropy
77.7
63.6
80.2
47.1
Semantic Entropy
77.7
60.0
76.1
68.6
Lexical Similarity
77.8
63.9
76.9
60.3
EigenScore
78.9
63.8
76.2
74.8
HaloScope
✓
60.7
63.0
85.3
69.0
HARP(Ours)
✓
84.0
88.1
92.8
88.4
LLaMA-3.1-8B
Perplexity
✓
50.3
71.4
76.3
53.4
LN-Entropy
52.7
62.5
55.8
48.8
Semantic Entropy
60.7
59.4
68.7
62.2
Lexical Similarity
60.9
49.1
71.0
69.5
EigenScore
56.7
45.3
69.1
82.4
HaloScope
✓
62.7
70.6
76.2
53.3
HARP(Ours)
✓
89.4
88.5
92.9
86.6
Correctness Measurement. Following Chen et al.[19], correctness is determined based on ROUGE-L and semantic
similarity between generated and reference answers. Semantic similarity is computed using the BLEURT model[31, 17].
An answer is considered correct if its ROUGE-L score exceeds 0.7 or its semantic similarity exceeds 0.5.
5.2
Main Results
Table 1 summarizes the AUROC scores (in %) of various hallucination detection methods across four QA datasets,
using Qwen-2.5-7B-Instruct and LLaMA-3.1-8B as backbone models. Several key findings emerge from these results.
(1) HARP consistently outperforms all baseline methods across all datasets and models, often by a significant margin.
For instance, on TriviaQA, HARP achieves AUROC scores of 92.8% on Qwen and 92.9% on LLaMA, yielding
improvements of +7.5% and +16.6%, respectively, over the second-best method, demonstrating its robustness and
scalability across architectures and data characteristics. (2) Baseline methods such as Perplexity and HaloScope
perform competitively on simpler datasets like TriviaQA, where answers are often limited to one or two tokens, but
their performance deteriorates sharply on more complex datasets such as TyDiQA, which contains long contexts and
accompanying documents. In contrast, HARP maintains high AUROC scores of 88.4% on Qwen and 86.6% on LLaMA
in these challenging settings, highlighting its ability to handle reasoning-intensive and context-rich inputs. (3) Sampling-
based methods, such as Semantic Entropy, Lexical Similarity, and EigenScore, incur higher computational costs but still
fail to achieve comparable performance, whereas HARP’s single-pass approach provides both superior efficiency and
accuracy. In addition, Table 2 reports the number of known and unknown questions for Qwen-2.5-7B-Instruct across the
four datasets, reflecting the model’s varying answering capabilities on these benchmarks. Collectively, these findings
validate the effectiveness, robustness, and practical utility of HARP for hallucination detection in diverse QA scenarios.
Table 2: Distribution of known and unknown questions across four QA datasets. A question is classified as Known
if the model’s knowledge state contains the correct answer according to the criterion in Equation 1, and as Unknown if
none of the 10 candidate responses contain the correct answer.
Dataset
Known
Unknown
TruthfulQA
636
181
TyDiQA
402
38
TriviaQA
6225
3735
NQ-open
293
3317
5.3
More Analysis
Rationality of Direct Sum Decomposition in Hidden State Space. To validate this direct sum decomposition, we
conduct a comparative experiment: removing the reasoning subspace components of hidden states and examining their
7

## Page 8

HARP: Hallucination Detection via Reasoning Subspace Projection
32
64
128
196
256
0
2
4
6
8
10
12
14
16
18
20
Reasoning Dimension
Rank
 25%~75%
 Range within 1.5IQR
 Median Line
(a)
32
64
128
192
256
512
1024
70
80
90
100
AUROC (%)
Reasoning Dimension
 NQ Open - Qwen
 TruthfulQA - Qwen
 NQ Open - LLaMa
 TruthfulQA - LLaMa
(b)
Figure 5: (a) Greedy token rankings in logits′ under different reasoning subspace dimensions. (b) Effect of reasoning
subspace dimension on hallucination detection performance.
effect on token prediction scores and rankings. Mathematically, this operation can be formulated as:
logits′ = Wk · hl = Wunemb · hl,Semantic
(19)
As shown in Figure 5a, computing token prediction scores using Equation 19 instead of the original logits maintains
the top rankings of greedily generated tokens. This result aligns with our theoretical design: the hidden state space
can be decomposed into semantic and reasoning subspaces, and token prediction is mainly influenced by the semantic
subspace component hl,Semantic.This experiment confirms that the proposed direct sum decomposition exhibits clear
representational disentanglement and functional partitioning, providing theoretical support for building hallucination
detection models based on the reasoning subspace.
Ablation Study. We tested the importance of projecting hidden states onto the reasoning subspace by comparing
hallucination detection performance under different projection strategies. “HARP (w/o)” denotes completely removing
the projection, while retaining hidden state features of the same dimensionality as full HARP; “HARP (random)”
denotes randomly selecting a set of bases from the projection basis V = {v1, v2, . . . , vd} for projection. The results
in Table 3 show that both removing the projection and using random projection significantly degrade hallucination
detection performance, confirming the necessity of projecting hidden states onto the reasoning subspace.
Table 3: Hallucination detection performance under different projection strategies
Methods
Qwen-2.5-7B-Instruct
LLaMA-3.1-8B
NQ Open
TruthfulQA
NQ Open
TruthfulQA
HARP (w/o)
62.9
70.7
70.4
73.5
HARP (random)
67.6
68.6
59.5
75.8
HARP
84.0
88.1
89.4
88.5
Impact of Reasoning Subspace Dimension on Hallucination Detection. The reasoning subspace dimension affects
hallucination detection in two ways: (1) its influence on logits scores: when the dimension is too large, Equation 14
gradually breaks down, which impairs the model’s next-token prediction capability; (2) its effect on detection accuracy
and generalization: increasing the dimension may improve training accuracy but also increases the risk of overfitting,
reducing generalization. We evaluated dimensions from 32 to 1024 using Qwen-2.5-7B-Instruct and LLaMA-3.1-8B
models. As shown in the Figure 5b, a dimension of 256 yields the best performance. This dimension accounts for only
about 5% of the original hidden state dimensionality, preserving sufficient reasoning information while filtering most
redundant noise, satisfying the information-preservation constraint in Equation 14.
Selection of Hallucination Score Threshold. In practice, it is necessary to set a hallucination score threshold α
so that bG(y|x) produces a clear binary decision. As shown in Figure 6a and Figure 6b, when α is between 0.2
and 0.8, both detection accuracy and F1 score remain high, indicating a substantial separation between normal and
8

## Page 9

HARP: Hallucination Detection via Reasoning Subspace Projection
0.0
0.2
0.4
0.6
0.8
1.0
30
40
50
60
70
80
90
100
Accuracy  (%)
Thresholds
 NQ Open
 TruthfulQA
 TriviaQA
 TyDiQA
(a)
0.0
0.2
0.4
0.6
0.8
1.0
30
40
50
60
70
80
90
100
F1 (%)
Thresholds
 NQ Open
 TruthfulQA
 TriviaQA
 TyDiQA
(b)
Figure 6: (a) Effect of hallucination score threshold on detection accuracy. (b) Effect of hallucination score threshold
on detection F1 score.
hallucinated answers under bG. To align with common expectations for a binary classifier, we set α = 0.5, where
bG(y|x) = I [gθ(y|x) > 0.5].
Robustness Experiments. To apply HARP in real-world scenarios, we examined its performance under distribution
shifts between training and test sets. We trained the hallucination detector on a source dataset s and evaluated it on
different target datasets t. Figure 7 shows that HARP generalizes well across multiple target datasets. Notably, when
trained on TriviaQA, its accuracy on NQ Open is nearly identical to directly training on NQ Open, demonstrating
HARP’s strong robustness and cross-distribution adaptability.
89.4
77.9
91.7
81.6
76.2
88.5
91.9
81.2
89.2
84.9
92.9
80.7
82.0
80.9
91.8
86.6
NQ Open (t)
TruthfulQA (t)
TriviaQA (t)
TyDiQA (t)
TyDiQA (s)
TriviaQA (s)
TruthfulQA (s)
NQ Open (s)
76.2
79.5
82.9
86.2
89.6
92.9
Figure 7: Cross-dataset generalization. “(s)” indicates the source dataset used for training the hallucination detector;
“(t)” indicates the target dataset.
6
Conclusion
In this study, we introduced HARP, a novel hallucination detection method that leverages only reasoning information
as input features, achieving high detection accuracy while maintaining strong robustness. First, we showed that
the hidden state space admits a direct-sum decomposition into a semantic subspace and a reasoning subspace, and
that the Unembedding layer can effectively separate these two components. Building on this, we applied singular
value decomposition to the parameters of the Unembedding layer and, following the Eckart–Young–Mirsky theorem,
approximated Wunemb with its best rank-k approximation Wk. Setting k = d×95%, we identified basis vectors for both
the semantic and reasoning subspaces that align with empirical observations. Furthermore, we empirically validated
9

## Page 10

HARP: Hallucination Detection via Reasoning Subspace Projection
that the reasoning subspace effectively captures intermediate reasoning information through the Reasoning Patch
experiment detailed in Appendix E. Finally, HARP constructs an accurate and efficient hallucination detector by using
the projections of hidden states in the reasoning subspace as input features. Experiments show that HARP significantly
outperforms existing mainstream hallucination detection methods and maintains robustness under distribution shifts
across datasets. In addition, we present a proof-of-concept demonstration of hallucination mitigation using our
framework in Appendix D and aim to inspire future research in this direction.
References
[1] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei
Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
[2] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint
arXiv:2407.21783, 2024.
[3] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and
Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024.
[4] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto,
and Pascale Fung. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1–38,
2023.
[5] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025.
[6] P. N. Johnson-Laird. Mental models: towards a cognitive science of language, inference, and consciousness.
Harvard University Press, USA, 1986.
[7] Michael C Frank and Noah D Goodman.
Predicting pragmatic reasoning in language games.
Science,
336(6084):998–998, 2012.
[8] Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Talking heads: Understanding inter-layer communication in
transformer language models. Advances in Neural Information Processing Systems, 37:61372–61418, 2024.
[9] Pei Cheng, Xiayang Shi, and Yinlin Li. Enhancing translation ability of large language models by leveraging
task-related layers. In Proceedings of the 2024 Joint International Conference on Computational Linguistics,
Language Resources and Evaluation (LREC-COLING 2024), pages 6110–6121, 2024.
[10] Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding
neurons in a haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610, 2023.
[11] Ang Lv, Yuhan Chen, Kaiyi Zhang, Yulong Wang, Lifeng Liu, Ji-Rong Wen, Jian Xie, and Rui Yan. Interpreting
key mechanisms of factual recall in transformer-based language models. arXiv preprint arXiv:2403.19521, 2024.
[12] Tianjie Ju, Weiwei Sun, Wei Du, Xinwei Yuan, Zhaochun Ren, and Gongshen Liu. How large language models
encode context knowledge? a layer-wise probing study. In Nicoletta Calzolari, Min-Yen Kan, Veronique
Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International
Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages
8235–8246, Torino, Italia, May 2024. ELRA and ICCL.
[13] Linyang He, Peili Chen, Ercong Nie, Yuanning Li, and Jonathan R. Brennan. Decoding probing: Revealing
internal linguistic structures in neural language models using minimal pairs. In Nicoletta Calzolari, Min-Yen Kan,
Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint
International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING
2024, 20-25 May, 2024, Torino, Italy, pages 4488–4497. ELRA and ICCL, 2024.
[14] Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao, Kai
Mei, Yanda Meng, Kaize Ding, Fan Yang, Mengnan Du, and Yongfeng Zhang. Exploring concept depth: How
large language models acquire knowledge and concept at different layers? In Owen Rambow, Leo Wanner,
Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the
31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24,
2025, pages 558–573. Association for Computational Linguistics, 2025.
[15] Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model
representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023.
10

## Page 11

HARP: Hallucination Detection via Reasoning Subspace Projection
[16] Lennart Bürger, Fred A Hamprecht, and Boaz Nadler. Truth is universal: Robust detection of lies in llms. Advances
in Neural Information Processing Systems, 37:138393–138431, 2024.
[17] Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li. How to steer LLM latents for
hallucination detection? In ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The
Next Frontier in Reliable AI, 2025.
[18] Xuefeng Du, Chaowei Xiao, and Sharon Li. Haloscope: Harnessing unlabeled LLM generations for hallucination
detection. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak,
and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural
Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.
[19] Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. Inside: Llms’ internal
states retain the power of hallucination detection. arXiv preprint arXiv:2402.03744, 2024.
[20] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language
models using semantic entropy. Nat., 630(8017):625–630, 2024.
[21] Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika,
1(3):211–218, 1936.
[22] Michael Greenacre, Patrick JF Groenen, Trevor Hastie, Alfonso Iodice d’Enza, Angelos Markos, and Elena
Tuzhilina. Principal component analysis. Nature Reviews Methods Primers, 2(1):100, 2022.
[23] Ian J. Goodfellow, Yoshua Bengio, and Aaron C. Courville. Deep Learning. Adaptive computation and machine
learning. MIT Press, 2016.
[24] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey,
Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark
for question answering research. Trans. Assoc. Comput. Linguistics, 7:452–466, 2019.
[25] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In
Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27,
2022, pages 3214–3252. Association for Computational Linguistics, 2022.
[26] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 -
August 4, Volume 1: Long Papers, pages 1601–1611. Association for Computational Linguistics, 2017.
[27] Jonathan H. Clark, Jennimaria Palomaki, Vitaly Nikolaev, Eunsol Choi, Dan Garrette, Michael Collins, and
Tom Kwiatkowski. Tydi QA: A benchmark for information-seeking question answering in typologically diverse
languages. Trans. Assoc. Comput. Linguistics, 8:454–470, 2020.
[28] Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J
Liu. Out-of-distribution detection and selective generation for conditional language models. In The Eleventh
International Conference on Learning Representations, 2023.
[29] Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International
Conference on Learning Representations, 2021.
[30] Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang. Towards collaborative neural-symbolic graph semantic parsing via
uncertainty. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association
for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 4160–4173. Association for
Computational Linguistics, 2022.
[31] Thibault Sellam, Dipanjan Das, and Ankur P. Parikh. BLEURT: learning robust metrics for text generation. In Dan
Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pages 7881–7892. Association
for Computational Linguistics, 2020.
[32] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing
systems, 35:24824–24837, 2022.
[33] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,
Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve
math word problems. arXiv preprint arXiv:2110.14168, 2021.
11

## Page 12

HARP: Hallucination Detection via Reasoning Subspace Projection
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
12

## Page 13

HARP: Hallucination Detection via Reasoning Subspace Projection
Appendix
A
Datasets and Implementation Details
Input prompts. In our experiments, datasets were categorized based on whether additional supporting information is
provided. For datasets without context, including NQ-Open, TruthfulQA, and TriviaQA, we used prompts that contain
only the question. Specifically, the prompt format is:
Prompts for datasets without context
Q: {question}
A:
For datasets with context, including TyDiQA, the prompt includes both the task description and the relevant context:
Prompts for datasets with context
Concisely answer the following question based on the information in the given passage: 
Passage:  {context}
Q: {question}
A:
Implementation details. Using the formulations in subsection 3.1, we select LLMs’ known knowledge set Xknown =
{x | known(x) = 1} and unknown knowledge set Xunknown = {x | known(x) = 0}. 75% of Xknown is used
for training, while the remaining 25%, together with Xunknown, is used to test the hallucination detector on unseen
data. For dataset questions, the temperature is set to 0.5, and beam search is used to generate 10 answer paths per
question. The hallucination detector G is a two-layer MLP with hidden dimension 1024 and ReLU activation. Training
is conducted for 50 epochs with the Adam optimizer, initial learning rate 1e-4, cosine decay, batch size 128, and weight
decay 3e-4.
B
Extracting a Universal Representation via Uncentered PCA
Given a collection of n hidden vectors {h(i)}n
i=1 from LLMs, each of dimension d, we arrange them into a matrix:
M =


(h(1))⊤
...
(h(n))⊤

∈Rn×d
(20)
From an energy-maximization perspective, the “universal representation” of these hidden vectors can be interpreted as
their dominant direction of variation in the feature space. To extract this direction, we perform SVD:
M = U ′Σ′V ′⊤
(21)
where U ′ ∈Rn×n, Σ′ = diag(σ′
1, · · · , σ′
d) ∈Rn×d, V ′⊤= [v′
1, · · · , v′
d] ∈Rd×d, and the singular values satisfy
σ′
1 ≥σ′
2 ≥· · · ≥0. The dominant right singular vector v1 provides the principal direction of the row space of M,
which is equivalent to the first principal component in uncentered Principal Component Analysis (PCA). We define the
universal representation direction as:
ˆh = v′
1 ∈Rd
(22)
By collecting n hidden states from the i-th layer, we can derive the corresponding universal representation ˆhi following
the steps above. Projecting it onto the basis vectors V = [VS, VR] ∈Rd×d yields the projections of the i-th layer’s
hidden state onto the semantic and reasoning subspaces:
proj

ˆhi

= V ⊤· ˆhi
(23)
13

## Page 14

HARP: Hallucination Detection via Reasoning Subspace Projection
In Figure 4b, we normalize the lengths of proj

ˆhi

and visualize the projections of the universal representations
of hidden states from the first three and last three layers of the Qwen-2.5-7B-Instruct model onto the semantic and
reasoning subspaces. We observe that shallow layer vectors are primarily represented in the semantic subspace, while
deep layer vectors are more concentrated in the reasoning subspace.
C
Analysis of Layer-wise Contributions in LLMs
Although our previous analysis has characterized the hidden states after processing through multiple decoder layers, it
remains important to understand the individual contributions of each layer and how they differ. To this end, we define
the contribution of the i-th decoder layer as dhi = hi −hi−1, and, following the method described in Appendix B,
compute the universal representation direction ˆ
dhi. Since singular vectors obtained via SVD can have arbitrary signs,
we compute the absolute cosine similarity between ˆdhi and ˆdhj to measure the similarity between the universal
representations of the increments of the layers i and j.
Layer-01
Layer-02
Layer-03
Layer-04
Layer-05
Layer-06
Layer-07
Layer-08
Layer-09
Layer-10
Layer-11
Layer-12
Layer-13
Layer-14
Layer-15
Layer-16
Layer-17
Layer-18
Layer-19
Layer-20
Layer-21
Layer-22
Layer-23
Layer-24
Layer-01
Layer-02
Layer-03
Layer-04
Layer-05
Layer-06
Layer-07
Layer-08
Layer-09
Layer-10
Layer-11
Layer-12
Layer-13
Layer-14
Layer-15
Layer-16
Layer-17
Layer-18
Layer-19
Layer-20
Layer-21
Layer-22
Layer-23
Layer-24
0.0
0.2
0.4
0.6
0.8
1.0
Figure 8: Similarity between universal representation directions of layer-wise increments
Figure 8 illustrates the cosine similarity between the universal representation directions of layer-wise increments in
the Qwen-2.5-0.5B-Instruct model. We observe that the first six layers behave in a broadly similar manner; however,
the first two layers are relatively independent of the remaining ones, while layers 3, 4, and 6 exhibit almost identical
directions. Interestingly, the direction around layer 22 is remarkably similar to that of layers 3, 4, and 6. We hypothesize
that the first two layers primarily parse the shallow semantic structure of the input, layers 3, 4, and 6 encode this
information into an internal representation space (a “language” specific to the LLM), the intermediate layers perform
14

## Page 15

HARP: Hallucination Detection via Reasoning Subspace Projection
various reasoning operations over this representation, and layer 22 decodes it back into a human-interpretable semantic
space before unembedding.
Based on this observation, we argue that mitigating hallucinations—especially those induced by suboptimal training
patterns or aggressive answer-generation strategies—requires intervention in the decoding behavior around layer 22.
Following this intuition, Appendix D provides a demonstration of hallucination mitigation, with the goal of inspiring
future research in this direction.
D
Mitigating Hallucinations by Modifying Reasoning Subspace Components
Based on our previous analysis of LLM behavior, we explore mitigating hallucinations by intervening on the components
of hidden states within the reasoning subspace. To this end, we introduce a fictitious city, “Epsilon”, and pose the
question to the LLM: “The capital of Epsilon is ?”. We then investigate the effect of removing the reasoning subspace
components from hidden states at different layers and observe the resulting impact on the LLM’s outputs.
Table 4: Impact of interventions across layers and reasoning subspace dimensions on hallucination mitigation for the
query “The capital of Epsilon is ?”. Green responses indicate a correct mitigation of hallucinations.
Intervened
Layer
Reasoning
Subspace
Dimension
Output
None
None
Epsilon’s capital is likely **Kaiyuan**, which is the capital city of the Kingdom of
Kaiyuan in the Eastern Regions.
layer-01
8
Epsilon’s capital is likely to be the city or town where its government and administrative
center is located.
16
Epsilon’s capital is likely to be the city or town where its government and administrative
center is located.
32
Epsilon’s capital is likely to be the city or town where its government and administrative
center is located.
64
The capital of Epsilon is Elea.
layer-02
8
Epsilon is the capital city of the planet Alpha.
16
I apologize, but I don’t have enough context to determine the specific name or location
of the capital city in question.
32
I apologize, but I’m not able to determine the capital city of Epsilon as it appears to be
a fictional planet or alternate universe.
64
Epsilon is the capital city of the European Union (EU).
Layer-22
8
I’m sorry, but I don’t have enough context to accurately answer your question about
the capital city of Epsilon.
16
I’m sorry, but I need more context to accurately answer your question.
32
I’m sorry for any misunderstanding earlier.
64
Epsilon is currently not specified in my knowledge base for now.
Layer-23
8
I’m sorry, but I don’t have enough context to accurately answer your question about
the capital city of Epsilon.
16
I’m sorry, but I need more information to accurately answer your question.
32
Epsilon is currently not in my knowledge base as I am an AI language model created
by Alibaba Cloud based on publicly available information...
64
Epsilon is currently unknown due to lack of information about its current status in
relation to other planets in our solar system or neighboring celestial bodies...
Table 4 presents the outputs of the LLM under interventions in various layers and with different subspace dimensions
of reasoning. We observe that interventions in shallow layers, such as layers 1 and 2, produce limited improvement,
whereas interventions at deeper layers, such as layers 22 and 23, lead the LLM to explicitly acknowledge its lack of
knowledge about the fictitious city “Epsilon” and refuse to answer. This phenomenon aligns with our earlier analysis of
the behavior of LLMs. We hope that this hallucination-mitigation demo can inspire further research in this direction.
15

## Page 16

HARP: Hallucination Detection via Reasoning Subspace Projection
E
Verification of Reasoning Information in the Reasoning Subspace
To verify that the components of hidden states lying in the reasoning subspace indeed encode internal reasoning
information, we design a controlled experiment consisting of three input conditions (Figure 9). These conditions isolate
the effect of the reasoning subspace while keeping all other factors unchanged.
Embedding
Unembedding
× 𝑙
Question
Decoder
ℎ∙,𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
𝑁𝑜𝑟𝑚𝑎𝑙
ℎ∙,𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
𝑁𝑜𝑟𝑚𝑎𝑙+
Wrong  answer
Embedding
Unembedding
× 𝑙
CoT + Question
Decoder
ℎ∙,𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
𝐶𝑜𝑇
ℎ∙,𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
𝐶𝑜𝑇
+
Embedding
Unembedding
× 𝑙
Question
Decoder
ℎ∙,𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐
𝑁𝑜𝑟𝑚𝑎𝑙+
Reasoning steps
+ 
Correct answer
ℎ∙,𝑅𝑒𝑎𝑠𝑜𝑛𝑖𝑛𝑔
𝐶𝑜𝑇
Replace
Maintain
Reasoning steps
+ 
Correct answer
(A) Normal:
(B) CoT:
(C) Reasoning Patch:
Figure 9: Experimental design illustrating the three conditions used to verify the causal role of the reasoning
subspace. (A)Normal: the model receives only the question and produces an incorrect answer. (B)CoT: a chain-of-
thought is prepended, enabling multi-step reasoning and a correct answer. (C)Reasoning Patch: no CoT is provided, but
the reasoning-subspace components of hidden states at all layers are replaced with those from the CoT run, causing the
model to generate reasoning steps and arrive at the correct answer.
(A) Normal: direct question input.
In the first condition, we feed the model only the question without any chain-of-
thought (CoT) guidance. The model typically produces an incorrect answer. Let the hidden state be
hNormal
·
= hNormal
·,Semantic + hNormal
·,Reasoning.
(B) CoT: prepend chain-of-thought.
In the second condition, we prepend a chain-of-thought[32] to the input. The
model now first generates intermediate reasoning steps and then outputs the correct answer. The hidden state is
hCoT
·
= hCoT
·,Semantic + hCoT
·,Reasoning.
(C) Reasoning Patch: replace reasoning components at all relevant layers.
The third condition serves as the key
causal intervention. The input text is identical to condition (A); however, at every decoder layer that contributes to the
representation of a token, we replace only the reasoning-subspace component of the hidden state with the corresponding
component extracted from condition (B). Formally, for all layers along the forward-pass trajectory of token t, we apply:
hP atch
·
= hNormal
·,Semantic + hCoT
·,Reasoning.
Thus, semantic information is preserved at every layer, while the reasoning components across all intermediate layers
are substituted with those from the CoT run. This ensures that the patched forward pass follows the CoT reasoning
trajectory throughout the entire decoder stack.
Key result.
We evaluate the effectiveness of the proposed Reasoning Patch on mathematical reasoning benchmarks
such as GSM8K[33], using both few-shot CoT and zero-shot CoT to extract the reasoning-subspace components.
Figure 10 presents the Qwen2.5-7B-Instruct outputs under the three conditions (A)–(C) when the reasoning components
of condition (C) are derived from few-shot CoT, with the full prompts shown in Table 5. We observe that, even though
condition (C) receives no CoT text in the input, injecting the CoT-derived reasoning-subspace components reliably
triggers the model to follow a “reason-then-answer” generation pattern. As a result, the model transitions from an
incorrect answer in (A) to a correct, multi-step reasoning process in (C), demonstrating that the patched reasoning
trajectory causally determines the emergence of correct step-by-step reasoning.
16

## Page 17

HARP: Hallucination Detection via Reasoning Subspace Projection
(A) Normal
Input:
Q: Mishka bought 3 pairs of shorts, 3 
pairs of pants, and 3 pairs of shoes. One 
pair of shorts costs $16.50. One pair of 
pants costs $22.50 and one pair of shoes 
costs $42. How many dollars did 
Mishka spend on all the clothing items?
A: 
LLM Output: 
247.5
(B) CoT
Input:
(…few-shot CoT prompt…)
Q: Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts 
costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many 
dollars did Mishka spend on all the clothing items?
A: 
LLM Output: 
1. Calculate the cost for each type of clothing:
   - Shorts: 3 × $16.50 = $49.50
   - Pants: 3 × $22.50 = $67.50
   - Shoes: 3 × $42 = $126
2. Add up the costs to find the total amount spent:
   - Total cost = $49.50 + $67.50 + $126 = $243.00
Therefore, Mishka spent $243.00 on all the clothing items.
(C) Patch
Input:
Q: Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and 
one pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?
A: 
LLM Output: 
1. Calculate the total cost for each type of clothing item:
   - Shorts: 3 * $16.5 = $49.5
   - Pants] 3 * $22.5 = $67.5
   - Shoes] 3 * $42 = $126
2. Add up the total cost for all types of clothing items:
   - Total cost = $49.5 + $67.5 + $126 = $243
Therefore, Mishka spent a total of $243 on all the clothing items.
Figure 10: Reasoning Patch experiment using few-shot chain-of-thought supervision.
Table 5: few-shot chain-of-thought prompt.
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How 
many bolts in total does it take?
A: A robe needs 2 bolts of blue fiber.
The amount of white fiber needed is half of the blue fiber.
Half of 2 bolts is 1 bolt of white fiber.
The total bolts needed is the sum of blue and white fiber.
2 bolts plus 1 bolt equals 3 bolts.
Therefore, the final answer is 3.
(…Input…)
Figure 11 shows the corresponding results when the reasoning components are extracted from zero-shot CoT. Remark-
ably, even though condition (C) does not contain the zero-shot instruction (e.g., “Answer the following question step
by step to the best of your ability.”), the patched model nonetheless produces a coherent step-by-step reasoning chain
before giving the final answer. Interestingly, in this setting the original CoT run in condition (B) makes an arithmetic
mistake and outputs an incorrect final answer; however, condition (C)—which inherits only the reasoning-subspace
components rather than the explicit token sequence—does not reproduce this error and instead produces the correct
result. This highlights that the reasoning subspace captures the structural reasoning trajectory without being constrained
by the semantic information in the CoT prompt.
17

## Page 18

HARP: Hallucination Detection via Reasoning Subspace Projection
(A) Normal
Input:
Q: Charlie wants to sell beeswax 
candles. For every pound of beeswax, he 
can make 10 tapered candles. One 
pound of beeswax and the wicks cost 
$10.00 in supplies. If he sells each 
candle for $2.00 each, what is his net 
profit if he makes and sells 20 candles?
A: 
LLM Output: 
3.6
(B) CoT
Input:
Answer the following question step by step to the best of your ability.
Q: Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 
tapered candles. One pound of beeswax and the wicks cost $10.00 in supplies. If he sells each 
candle for $2.00 each, what is his net profit if he makes and sells 20 candles?
A: 
LLM Output: 
1. First, let's calculate the total cost of supplies for making 20 candles:
   We know that one pound of beeswax and a wick cost $11.00.\n To find out how many pounds of 
beeswax are needed for 20 candles, we divide the number of candles by the number of candles per pound: 
20 / 10 = 2 pounds.\n Therefore, the total cost of supplies is 2 * $11.00 = $22.00
2. Next, let's calculate the total revenue from selling 20 candles:
   We know that each candle is sold for $2.00. \n Therefore, the total revenue is 20 * $2.00 = $44.00
3. Finally, let's calculate the net profit:
   Net profit is total revenue minus total cost of supplies.\n Net profit = $44.00 - $22.00 = $22.00
Therefore, Charlie's net profit if he makes and sells 20 candles is $22.00.
(C) Patch
Input:
Q: Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 tapered candles. One pound of beeswax and the wicks 
cost $10.00 in supplies. If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?
A: 
LLM Output: 
1. Calculate the total cost of supplies for making candles:
   Total cost = 20  candles × $1.00  per candle = $20.00
2. Calculate the total revenue from selling the candles:
   Total revenue = 20  candles × $2.00  per candle = $40.00
3. Calculate the net profit by subtracting the total cost from the total revenue:
   Net profit = $40.00 - $20.00 = $20.00
Therefore, the net profit for Charlie if he makes and sells 20 candles is: $20.00
Figure 11: Reasoning Patch experiment using zero-shot chain-of-thought prompting.
Together, these results provide compelling evidence that the reasoning subspace encodes causally meaningful internal
reasoning information, and that injecting its components is sufficient to induce coherent multi-step reasoning even in
the absence of explicit CoT prompting.
F
Computational Complexity of SVD
To construct the reasoning subspace, we perform singular value decomposition (SVD) on a matrix M ∈Rn×d, where n
denotes the vocabulary size and d is the dimensionality of the hidden representation. In typical large language models,
the matrix is tall and skinny with n ≫d (e.g., for Qwen2.5-7B, n = 152,064 and d = 3,584). The computational
complexity of SVD depends on these matrix dimensions as well as whether a full or truncated decomposition is applied.
Time Complexity.
For a full SVD on an n × d matrix, the time complexity is
O
 min(nd2, n2d)

.
Since the vocabulary size is typically much larger than the hidden dimension, the dominant term becomes
O(nd2),
which makes full SVD computationally expensive in practice. For truncated SVD that retains only the top-k singular
directions, the complexity reduces to
O(ndk),
particularly when using iterative or randomized SVD algorithms. Such approximations are crucial for scaling to
vocabularies of realistic size.
18

## Page 19

HARP: Hallucination Detection via Reasoning Subspace Projection
Table 6: SVD computation cost on the unembedding layer using an H100 80GB GPU. We report wall-clock time,
memory required during the SVD computation, and the additional memory by SVD.
Model
Unembedding
Shape
Time
Peak
Memory
Extra
Memory
Qwen2.5-7B-Instruct
152,064 × 3,584
1.30s
8.37GB
0.02GB
LLaMA-3.1-8B
128,256 × 4,096
1.60s
8.08GB
0.03GB
Qwen2.5-72B-Instruct
152,064 × 8,192
9.83s
19.22GB
0.13GB
Qwen3-235B-A22B-
Instruct-2507-FP8 (MoE)
151,936 × 4,096
1.70s
9.55GB
0.03GB
Space Complexity.
Storing the matrix M requires
O(n2 + nd + d2)
memory. The truncated singular vectors U ∈Rn×k and V ∈Rd×k introduce an additional
O((n + d)k)
space overhead. Because n is very large in modern LLMs, the memory is dominated by storing U.
SVD Resource Consumption.
To quantify the practical resource requirements of performing SVD on the unem-
bedding layer, Table 6 summarizes the wall-clock time, the peak memory consumption during the SVD computation,
and the additional memory introduced by truncated SVD across several representative models. The evaluation covers
models of different scales—including Qwen2.5-7B-Instruct, LLaMA-3.1-8B, Qwen2.5-72B-Instruct, and the MoE
model Qwen3-235B-A22B-Instruct-2507-FP8 [34]—using an H100 80GB GPU.
Singular Value Distribution in Larger Models.
Figure 12 shows the singular value distribution of the unembedding
layers for larger models, including Qwen2.5-72B-Instruct and Qwen3-235B-A22B-Instruct-2507-FP8 (MoE). The
trend of singular value decay is consistent with that observed for Qwen2.5-7B-Instruct and LLaMA-3.1-8B (Figure 4a),
indicating that our method can be directly applied to larger models.
0
2000
4000
6000
8000
10-1
100
101
102
0
2000
4000
6000
8000
100
101
102
Singular values
Index
Qwen-2.5-72B-Instruct
Qwen3-235B-A22B-Instruct-2507-FP8
Singular values
Index
Figure 12: Singular value distributions of Wunemb after SVD, with hidden state dimensions of 8192 for Qwen2.5-72B-
Instruct and 4096 for Qwen3-235B-A22B-Instruct-2507-FP8 (MoE).
19
