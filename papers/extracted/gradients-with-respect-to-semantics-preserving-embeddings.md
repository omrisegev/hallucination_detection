---
source_pdf: C:/Users/omris/TAU/hallucination_detection/papers/Gradients_with_Respect_to_Semantics_Preserving_Embeddings.pdf
slug: gradients-with-respect-to-semantics-preserving-embeddings
pages: 18
extracted_on: 2026-07-15
---

# Gradients_with_Respect_to_Semantics_Preserving_Embeddings

## Page 1

Gradients with Respect to Semantics Preserving Embeddings
Tell the Uncertainty of Large Language Models
Mingda Li 1 Rundong Lv 1 Xinyu Li 1 Weinan Zhang 1 2 Ting Liu 1
Abstract
Uncertainty quantification (UQ) is an important
technique for ensuring the trustworthiness of
LLMs, given their tendency to hallucinate. Exist-
ing state-of-the-art UQ approaches for free-form
generation rely heavily on sampling, which in-
curs high computational cost and variance. In
this work, we propose the first gradient-based
UQ method for free-form generation, SemGrad,
which is sampling-free and computationally ef-
ficient. Unlike prior gradient-based methods de-
veloped for classification tasks that operates in
parameter space, we propose to consider gradi-
ents in semantic space. Our method builds on the
key intuition that a confident LLM should main-
tain stable output distributions under semantically
equivalent input perturbations. We interpret the
stability as the gradients in semantic space and
introduce a Semantic Preservation Score (SPS) to
identify embeddings that best capture semantics,
with respect to which gradients are computed. We
further propose HybridGrad, which combines the
strengths of SemGrad and parameter gradients.
Experiments demonstrate that both of our meth-
ods provide efficient and effective uncertainty esti-
mates, achieving superior performance than state-
of-the-art methods, particularly in settings with
multiple valid responses.
1. Introduction
With the widespread deployment of Large Language Models
(LLMs) across various domains, including education, health-
care, and finance (Naveed et al., 2023; Wang et al., 2024;
Chiarello et al., 2024; Raza et al., 2025; He et al., 2025), the
1Research Center for Social Computing and Interactive
Robotics, Harbin Institute of Technology, Harbin, China 2Suzhou
Research Institute, Harbin Institute of Technology, Harbin, China.
Correspondence to: Weinan Zhang <wnzhang@ir.hit.edu.cn>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
reliability of their responses has become a pressing concern.
Despite their impressive abilities, LLMs remain prone to
hallucinating untruthful contents which undermine credibil-
ity in real-world applications (Zhang et al., 2023; Huang
et al., 2024). Uncertainty Quantification (UQ) has emerged
as a promising approach to mitigate these risks by providing
not only what models predict, but also how confident they
are in those predictions (Baan et al., 2023; Shorinwa et al.,
2024).
Although Uncertainty Quantification has been widely ex-
plored and proven effective in classification tasks (Gaw-
likowski et al., 2023), extending it to LLM-based free-form
generation presents unique challenges. Unlike standard
classification, where the label space is fixed and relatively
constrained, LLMs operate in a sequential classification
framework with an extremely large vocabulary at each step,
resulting in a combinatorially vast output space. Moreover,
the inherent nature of natural language allows multiple valid
responses for a single input, introducing a substantially
higher degree of aleatoric uncertainty—defined as the in-
herent, irreducible randomness within the data (H¨ullermeier
& Waegeman, 2021)—than is typically observed in single-
step classification tasks (Baan et al., 2023). State-of-the-art
UQ methods for free-form generation primarily rely on
sampling-based approaches that capture semantic variation
within the output space (Kuhn et al., 2023; Chen et al., 2024;
Duan et al., 2024; Qiu & Miikkulainen, 2024). Although
these approaches generally outperform self-verbalized meth-
ods or simple deterministic logit-based methods (Shorinwa
et al., 2024; Kuhn et al., 2023), they require a substantial
number of samples to approximate the vast output space,
resulting in high variance and computational cost, especially
given the scale of current LLMs.
Unlike sampling-based methods, gradient-based methods
directly exploit gradients of the log probability of gener-
ated outputs, which can be collected in parallel with the
generation process, enabling sampling-free and efficient es-
timation. However, previous work (Lee & AlRegib, 2020;
Igoe et al., 2022; Wang & Ji, 2024) on gradient-based UQ
was developed for classification tasks with the assumption
that each input has a single ground-truth label (i.e., zero
aleatoric uncertainty). This assumption breaks down in
1
arXiv:2605.04638v2  [cs.CL]  1 Jun 2026

## Page 2

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Figure 1. Illustration of output distribution shift under small input semantic perturbations and the semantic gradients. x represents the
original input, and x + ∆x denotes a perturbed input with a small semantic change on x in the semantic space. y∗denotes the response
generated from p(y|x). For an input that the model is certain about, a small semantic perturbation should not significantly alter the output
distribution, as shown in (a), i.e., p(y∗|x) is insensitive to small semantic perturbation. The sensitivity can be captured by the magnitude
of the slope of the red line, corresponding to the gradient in semantic space when ∆x →0. In contrast, the gradient will be high for the
uncertain input, as shown in (b).
free-form generation, where valid responses are not always
unique. Moreover, the sequential nature of generation com-
plicates UQ, since individual tokens contribute unequally
to meaning (Duan et al., 2024), with some carrying high
semantic weight while others are negligible. Therefore, a
gradient-based method specifically designed for free-form
generation is needed.
In this work, we introduce, to our knowledge, the first
gradient-based approach to uncertainty quantification for
free-form generation in LLMs. Unlike prior work that mea-
sures gradients in parameter space, we consider gradients
in semantic space. The underlying intuition is straightfor-
ward: if a well-trained LLM is confident in its response to
a query, its output distribution should remain stable when
the query is perturbed with semantically equivalent vari-
ants (Li et al., 2025), as illustrated in Figure 1 (a). This
mirrors human behavior: when confident, people tend to
provide consistent answers, whereas uncertainty often leads
to variability in responses. Building on this assumption, we
quantify uncertainty by measuring how sensitively the out-
put probabilities change under small semantic perturbations.
Technically, this sensitivity can be described by the gradient
of the output probability with respect to the semantic pre-
serving embeddings (slope of the red line in Figure 1). We
call this method Semantic Gradients (SemGrad). Notably,
our method does not rely on any assumption about the form
of the ground-truth distribution and thus remains valid even
under high aleatoric uncertainty.
To identify the embeddings that best preserve input seman-
tic information, we introduce the Semantic Preservation
Score (SPS), which measures the alignment difference of
each hidden state between semantic-equivalent paraphrases
and semantically different ones, and identify the seman-
tic preserving embeddings as the embeddings with high
SPS. Meanwhile, to mitigate the issue of linguistic redun-
dancy, we further propose a simple yet effective method
that re-weights the output probabilities prior to gradient
computation.
While parameter gradients can be unreliable under
high aleatoric uncertainty, they remain competitive in
single–ground-truth settings. To leverage the advantages of
both approaches and improve generalization, we propose
a hybrid metric named HybridGrad. Our experiments on
several QA benchmarks demonstrate that both SemGrad
and HybridGrad provide efficient and effective uncertainty
estimates, achieving superior performance than state-of-the-
art methods, particularly in cases where multiple responses
are correct. Meanwhile, our experiments reveal a strong
positive correlation between the capability of hidden states
to preserve semantics and the UQ performance of SemGrad
when gradients are computed with respect to them. This
finding supports our claim that the method indeed operates
in semantic space, consistent with our assumption that the
output distribution of an LLM should remain stable under
small semantic perturbations of the input, but not under
arbitrary random perturbations.
2. Preliminaries
In this section, we provide a brief overview of the architec-
ture of prevailing large language models, introduce the nota-
tions and key concepts used throughout the paper, and finally
2

## Page 3

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
review gradient-based uncertainty quantification methods
developed for classification tasks.
Current LLMs generally follow a causal autoregressive
paradigm.
Given an input text x = {x1, x2, ..., xI},
where each xi represents an input token, a causal language
model factorizes the probability of generating a response
y = {y1, y2, ..., yT } into conditional distributions,
p(y|x, θ) =
T
Y
t=1
p(yt|y<t, x, θ)
and generates tokens in a left-to-right manner. When gener-
ating the token yt+1, each input token xi and previously gen-
erated token y≤t are mapped to a sequence of embeddings,
one per token, yielding the initial hidden states h(0). The
initial hidden states are then processed by a stack of L trans-
former blocks. At each layer l, the hidden states are updated
through a residual connection around a self-attention trans-
formation followed by another residual connection around a
feed-forward transformation,
h(l)
j
= h(l−1)
j
+Attn(h(l−1)
≤j
)+FFN(h(l−1)
j
+Attn(h(l−1)
≤j
))
where j ≤t and attention operates on previous tokens by a
causal mask. We omit normalization terms for brevity, as
they vary across structures and are not central to our work.
After traversing all L layers, the final hidden state h(L)
t
is
projected into vocabulary space by the LM head weight
W head to produce logits, and then get the output distribution
through softmax,
zt = h(L)
t
W T
head ;
p(yt+1|y≤t, x, θ) = Softmax(zt)
We use θ to denote all parameters of an LLM, including the
weight matrices (and bias if any) in each attention and FFN
layer, as well as the token embedding matrix E, LM head
matrix W head and parameters in normalization.
Training an LM aims to approximate the ground truth human
language distribution p∗. Accordingly, we minimize the
expected negative log-likelihood of sequences under p∗,
min
θ
Ex∼p∗(x)[Ey∼p∗(y|x)[−log p(y|x, θ)]]
For a specific input x, if θ∗is optimal, it should minimize
Ey∼p∗(y|x)[−log p(y|x, θ)], and therefore its gradient with
respect to θ is vanished at optimal θ∗,
∇θ Ey∼p∗(y|x)[−log p(y|x, θ)]

θ=θ∗= 0
(1)
In a classification task, if we assume that there exists only
a single ground-truth label y∗(i.e., zero aleatoric uncer-
tainty), then the ground-truth distribution p∗degenerates
to a Dirac delta distribution. In this case, the expectation
Ey∼p∗(y|x)[−log p(y|x, θ)] collapse to −log p(y∗|x, θ),
and the gradient (1) reduces to
∇θ log p(y∗|x, θ)|θ=θ∗= 0
(2)
This observation motivates the use of parameter gradient
norm ||∇θ log p(y|x, θM)|| as a proxy for UQ of a model
M on classification tasks (Igoe et al., 2022; Wang & Ji,
2024).
A small value indicates that the model is well
trained on the given data point and confident in its pre-
diction, whereas a large value suggests the opposite, i.e.,
higher uncertainty.
However, this reasoning does not extend to the ground-truth
distribution of natural language, p∗(y|x), which usually
exhibits high aleatoric uncertainty due to the existence of
multiple valid responses. In this setting, Equation (2) is not
necessarily satisfied even at an optimum because p∗(y|x) is
no longer a Dirac delta. As a result, the parameter gradient
norm can be misleading, as large values may reflect the
aleatoric uncertainty of the task rather than genuine model
uncertainty. Meanwhile, approximating the expectation of
Equation (1) is computationally intractable and inefficient
for modern LLMs, due to their extremely large output space
and parameter size.
3. Semantic Gradients
To overcome the limitation of the parameter gradient, we
propose to evaluate gradients with respect to the semantic
space, inspired by an intrinsic nature of human language:
stable input semantics should yield stable output semantics.
3.1. Why Gradients on Semantics?
We start from a simple assumption about human language:
no matter how syntactic form may vary, as long as the
underlying context meaning (semantics) is preserved, the
responses should remain stable (Li et al., 2025). Accord-
ingly, for the ground-truth distribution of human language
p∗(y|x), we assume that semantically equivalent inputs x
and x′ yield a similar output distribution, i.e.,
p∗(y|x) ≈p∗(y|x′)
In other words, the true distribution should be insensitive to
small perturbations in the semantic space.
Therefore, if a model’s output distribution changes sharply
under such perturbations, this suggests that the model is lo-
cally misaligned with the underlying ground-truth distribu-
tion around that query. We can interpret this local instability
as being related to epistemic uncertainty, since it measures
the lack of knowledge of the underlying ground-truth data-
generating process (H¨ullermeier & Waegeman, 2021). This
mirrors human behavior — when confident, people tend to
3

## Page 4

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
provide consistent answers, whereas uncertainty often leads
to variability in responses.
Now consider an LLM p(y|x, θ), which generates a spe-
cific output ˆy given input x. Suppose we can identify a
semantic-preserving embedding hE(x), such that semanti-
cally equivalent variants of x are mapped to nearby vectors,
while semantically different inputs are mapped to distant
ones. A perturbation on hE(x), i.e., hE(x) + ∆hE, can
then be regarded as a semantic variation of x. As long as
∆hE is sufficiently small, the perturbation should preserve
semantics. If the LLM is well-trained on x, i.e., close to
the true distribution, we expect the output distribution to re-
main stable under the small semantic perturbations ∆hE, as
shown in Figure 1(a). This stability corresponds to a small
gradient with respect to the semantic-preserving embedding
(illustrated by the shallow slope of the red line in Figure
1(a)), i.e.,
∥∇hE log p(ˆy|x, θ; hE(x))∥≈0
(3)
Conversely, if the model is uncertain about its response,
we expect an unstable output distribution under the small
semantic perturbations, as shown in Figure 1(b), resulting
in a large gradient with respect to the semantic-preserving
embedding (illustrated by the sharp slope of the red line in
Figure 1(b)).
Therefore, we propose to use the gradient norm of the log-
likelihood with respect to the semantics-preserving embed-
dings as a measure of uncertainty of LLMs. Importantly,
these semantic gradients do not rely on any assumption
about the shape of the ground-truth distribution, thus remain
valid even in the presence of high aleatoric uncertainty.
3.2. Identifying Semantic-preserving Embeddings
As illustrated above, we aim to compute gradients with
respect to the semantic-preserving embeddings. These em-
beddings must satisfy two requirements: first, they must
be produced by the model’s own forward computation, en-
suring that gradients with respect to these representations
are well-defined and directly connected to the model’s pre-
diction behavior. Second, they must exhibit semantic com-
pleteness and consistency, meaning they have access to the
complete input semantics and map semantically equivalent
inputs to nearby representations while keeping semantically
different inputs well separated.
Natural candidates are the hidden states corresponding to the
last token of the user input. However, which layer should be
chosen? Prior work (Li & Subramani, 2025) has shown that
early layers primarily encode lexical features rather than
semantic content. Moreover, instruction-tuned LLMs often
wrap inputs in a chat template (e.g., role tags or assistant-
start markers) that introduces additional special tokens after
the user text, making the choice of token position non-trivial.
To support further analysis, we propose the Semantic Preser-
vation Score (SPS), which measures how well a model’s
hidden representations preserve input semantic structure
across layers and token positions: representations of seman-
tically equivalent input variants should be close, while those
of semantically different inputs should be far apart.
Formally, given a set of input queries {x1, x2, ..., xN},
for each xn, we generate K semantically equivalent para-
phrases {x(j)
n }K
j=1 and set x(0)
n
≡xn. Then, for a given
LLM, let h(l)
−t(x) denote the hidden states at layer l for the
t-th token counted from the end of x (t = 1 is the last to-
ken), obtained by forwarding x through the LLM. We first
compute the average within-paraphrase similarity:
Sl,t
w/i = 1
N
N
X
n=1
1
K(K + 1)
X
i̸=j
scos

h(l)
−t(x(i)
n ), h(l)
−t(x(j)
n )

where scos(u, v) denotes the cosine similarity. Then the
average across-query similarity is obtained
Sl,t
a/c =
1
N(N −1)
X
n̸=m
scos

h(l)
−t(xm), h(l)
−t(xn)

Then the Semantic Preservation Score of h(l)
−t(x) is obtained
by the difference between them:
SPS

h(l)
−t

= Sl,t
w/i −Sl,t
a/c
By construction, a higher SPS(h(l)
−t) indicates stronger se-
mantic preservation of h(l)
−t: semantically equivalent inputs
are pulled together in representation space, while semanti-
cally different inputs are pushed apart.
We evaluate the SPS of different hidden states on three
datasets and three models, part of the results can be found
in Figure 2. Further details are provided in Appendix C.1.
We have three key findings: (i) For each model there exists
a token position—termed the Semantic Preserving Token
and denoted t∗—that achieves the highest average SPS, and
this token is consistent across different datasets for the same
model; (ii) At t∗, semantic information is mainly preserved
in the deeper half of layers, whereas lower layers yield
near-zero SPS and thus primarily capture lexical features,
consistent with previous works (Li & Subramani, 2025); (iii)
A high-SPS band spans adjacent layers at t∗. Although the
precise peak varies across models and datasets, the deeper
half of layers consistently attains strong SPS.
Motivated by these findings—and to improve robustness
and cross-dataset generalization—we propose to compute
gradients with respect to the hidden states from the top half
of layers at t∗, rather than restricting to a single specific
layer, denoted as
h↑
t∗:= h
( L
2 +1:L−1)
t∗
= Concat

h
( L
2 +1)
t∗
; ...; h(L−1)
t∗

4

## Page 5

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Figure 2. Semantic Preservation Score (SPS) of hidden states across different layers and tokens. We experiment on the last 10 input tokens,
where “last #t token” denotes the last t-th token from the end of the user query (corresponding token is different for different queries). We
observe that the token position carrying the most semantic information is consistent for the same model across different datasets.
Notably, we do not compute gradients with respect to the
last-layer hidden states, since these are mainly used to de-
code the next output token and are not further attended to
in subsequent steps. As a result, we believe that it does not
carry too much input semantics.
3.3. Semantic Gradient Metric
We now formally introduce our Semantic Gradient Metric
(SemGrad). As outlined in Section 3.1, the metric is de-
fined by computing the gradient of the log-likelihood of
the generated response, which decomposes into the sum of
token-level log-likelihoods
∇h↑
t∗
T
X
t=1
log p( ˆyt| ˆ
y<t, x, θ; h↑
t∗(x))

However, free-form text generation often exhibits linguis-
tic redundancy, where tokens contribute unequally to the
overall meaning. Treating all tokens uniformly can there-
fore impair the effectiveness of uncertainty quantification
(Duan et al., 2024; Bakman et al., 2024). Prior work has
attempted to address this by relying on third-party models to
estimate token-level semantic importance, but this approach
is computationally expensive. Instead, we directly utilize
the intuition that uninformative tokens (e.g., stopwords or
subwords) always exhibit low output entropy. Therefore,
we re-weight the log-likelihood by token entropy before
computing the gradient, yielding the final SemGrad metric:
SSemGrad =
1
|h↑
t∗|
∇h↑
t∗
T
X
t=1
ωt log p( ˆyt|ˆy<t, x, θ; h↑
t∗)

1
(4)
where ωt = H(p(yt|ˆy<t, x)) is the output token entropy at
step t. During gradient computation, these entropy weights
are detached from the computation graph and treated as
fixed scalar coefficients, so that they modulate token contri-
butions without altering the gradient flow. We use the mean
absolute value of the gradient (i.e., the l1 norm normalized
by dimension) to transform the gradient vector into a scalar
metric.
Additionally, while parameter gradients are principally un-
reliable under high aleatoric cases—where multiple valid
responses lead to a multimodal ground-truth distribution—
they remain a valid and often competitive measure in single-
ground-truth settings. In such low-aleatoric regimes, the
ground-truth distribution is typically sharp and unimodal,
causing the parameter gradient to align closely with the
model’s training objective and yielding greater numerical
stability. In contrast, while SemGrad is theoretically well-
motivated in both low- and high-aleatoric settings, it oper-
ates by identifying hidden states that serve as a proxy for
semantic information. These representations are not guar-
5

## Page 6

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
anteed to perfectly isolate all semantic factors, which can
introduce additional numerical instability, making it less
stable than the parameter gradients in low-aleatoric cases.
Therefore, to leverage the theoretical robustness of Sem-
Grad in high aleatoric settings and the numerical stability
of parameter gradient in low aleatoric settings, we propose
a hybrid metric (HybridGrad) that combines the strengths
of both approaches. As a first step, we propose a token-
importance–weighted variant of parameter gradients, analo-
gous to the construction used for SemGrad; we refer to this
variant as ParaGrad:
SParaGrad = 1
|θ|
∇θ
T
X
t=1
ωt log p( ˆyt|ˆy<t, x, θ)

1
(5)
To balance SemGrad and ParaGrad, we compute the average
per-token entropy, ¯ω =
1
T
PT
t=1 ωt, which approximates
the sequence-level entropy H(p(y|x)) (Malinin & Gales,
2021). We then use ¯ω to interpolate between them:
SHybridGrad =
 1 −e−¯ω
SSemGrad + e−¯ωSParaGrad
(6)
When ¯ω is small (low entropy), HybridGrad assigns more
weight to parameter gradients; conversely, in high-entropy
cases, it relies more on semantic gradients.
4. Empirical Evaluations
Following previous work (Kuhn et al., 2023; Qiu & Miikku-
lainen, 2024), we evaluate whether the estimated score can
reliably predict the correctness of self-generated responses.
The more accurately the score aligns with response correct-
ness, the more effectively it quantifies uncertainty.1
4.1. Experimental Setup
Datasets.
We utilize three widely used free-form QA
datasets for our evaluation. These include two factual QA
benchmarks with a single ground-truth answer, SciQ (Welbl
et al., 2017) and TriviaQA (Joshi et al., 2017), and one
benchmark with multiple plausible answers, TruthfulQA
(Lin et al., 2022). Many of the questions in TruthfulQA are
open-ended (e.g., “What happens to you if you eat water-
melon seeds?”), which naturally introduces a high degree of
aleatoric uncertainty.
Models. We experiment with three open-source LLMs
that differ in architecture and chat template: Llama3.1-
Instruct8B2, Qwen3-Instruct4B (Yang et al., 2025), and
Mistral-Nemo-Instruct12B3. For each model, we obtain re-
sponses via greedy decoding and assess their correctness
1Our code and data are available at https://github.
com/mingdali6717/SemGrad
2https://ai.meta.com/blog/meta-llama-3-1/
3https://mistral.ai/news/mistral-nemo/
using BEM score (Bulian et al., 2022), a reproducible cor-
rectness metric based on semantic similarity and specifically
designed for QA tasks. Compared with lexical overlap ap-
proaches such as Rouge, BEM has been shown to provide
more dependable correctness assessments (Kamalloo et al.,
2023). The evaluated responses correctness is subsequently
treated as the ground-truth label for UQ assessment.
Baselines. The performance of our proposed method is com-
pared with eleven LLM UQ methods: Length-normalized
Predictive Entropy (denoted by LN-PE) (Malinin & Gales,
2021), P(True) (Kadavath et al., 2022), Self-Consistency
(denoted by Self-Con) (Wang et al., 2023), Deg (Lin et al.,
2024), INSIDE (Chen et al., 2024), Semantic Entropy (de-
noted by S.E.) (Kuhn et al., 2023), Semantic Density (de-
noted by S.D.) (Qiu & Miikkulainen, 2024), M.I. (Abbasi-
Yadkori et al., 2024), G-NLL (Aichberger et al., 2026), SAR
(Duan et al., 2024), and ExGrad (Igoe et al., 2022). No-
tably, SAR is the state-of-the-art method that introduces
importance weights to focus on more relevant tokens and
sentences. ExGrad, originally proposed for classification
tasks, computes parameter gradients. We extend it in a
straightforward manner to the free-form generation setting
by taking the gradient of the log-likelihood of generated se-
quences with respect to the model parameters—specifically,
the LM head weights W head —without applying importance
reweighting. More details are provided in Appendix C.2.
Evaluation Metric. To assess how well a UQ score re-
flects generation correctness, we report the Area Under the
Receiver Operating Characteristic (AUROC). This metric
captures the ability of the score to separate correct from
incorrect outputs. A value of 0.5 corresponds to random
guessing, whereas a value of 1.0 denotes perfect discrimina-
tion. More metrics are reported in Appendix D.2.
Implementation Details. As illustrated in Section 3.2,
we compute gradients at semantic preserving token t∗.
As shown in Figure 2, The semantic preserving to-
ken is <|start header id|> for Llama3.1-Instruct8B,
<|im start|> for Qwen3-Instruct4B and the last user in-
put token for Mistral-Nemo-Instruct12B. For the ParaGrad,
computing gradients with respect to all model parameters is
inefficient. Following Igoe et al. (2022), we only compute
gradients with respect to the LM head weights, W head.
4.2. Main Results
In Table 1, we report the main results.
Our proposed
methods—ParaGrad, SemGrad and HybridGrad—achieve
the highest average AUROC performance across all base-
lines. Notably, SemGrad shows strong advantages on the
multiple–correct-answer dataset, TruthfulQA, outperform-
ing the previous state-of-the-art SAR by +3.27 points, the
parameter-gradient baseline ExGrad by +6.82 points and
our proposed parameter-gradient variants ParaGrad by +3.3
6

## Page 7

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Table 1. AUROC of different UQ methods on generation correctness prediction. A larger value indicates better UQ performance. The
bold number represents the best performance across all methods for each dataset–model pair. The Avg. columns report the average
AUROC performance across all datasets and models.
UQ Methods
Qwen3-Instruct4B
Mistral-Nemo-Instruct12B
Llama3.1-Instruct8B
Avg.
SciQ
TriviaQ
TruthfulQ
SciQ
TriviaQ
TruthfulQ
SciQ
TriviaQ
TruthfulQ
LN-PE
67.08
80.00
64.78
76.68
84.02
66.29
72.51
84.53
63.38
73.25
P(True)
57.13
76.30
49.17
71.40
81.39
53.75
64.91
78.60
54.15
65.20
Self-Con
61.95
76.64
64.26
71.07
81.80
67.03
71.47
83.56
56.78
70.51
Deg
65.01
78.21
63.30
74.15
83.11
67.27
73.11
84.67
59.12
71.99
INSIDE
57.96
72.47
62.29
71.54
72.56
62.21
70.83
76.24
54.50
66.73
S.E.
56.88
76.16
63.10
68.53
80.64
66.71
70.27
83.12
59.59
69.45
S.D.
63.79
76.41
57.60
72.52
79.07
63.11
74.00
82.44
57.75
69.63
M.I.
66.25
76.26
63.75
73.72
81.88
66.06
72.43
83.52
64.25
72.01
G-NLL
72.70
81.01
60.44
76.83
84.61
63.67
75.49
85.91
57.51
73.13
SAR
72.72
81.52
67.98
76.57
85.23
68.55
75.28
85.65
64.44
75.33
ExGrad
71.34
80.37
63.77
77.53
84.53
66.40
74.11
85.22
62.00
73.92
ParaGrad
72.09
82.02
66.40
77.99
85.91
70.54
74.98
86.49
63.91
75.59
SemGrad
72.20
80.40
69.06
75.55
82.37
72.27
75.76
84.72
69.42
75.75
HybridGrad
72.83
81.69
69.61
76.90
84.13
72.72
76.31
85.89
69.25
76.59
on average across models. This supports our analysis that
parameter-gradient methods are less reliable under high
aleatoric uncertainty, whereas SemGrad can effectively cap-
ture model uncertainty in such settings.
On single–answer datasets (SciQ and TriviaQ), the perfor-
mance of SemGrad, while generally superior to most base-
lines, is less stable and occasionally inferior to parameter
gradient methods (ExGrad and ParaGrad). Conversely, pa-
rameter gradient method performs poorly on high-aleatoric
dataset but remains competitive in single–answer settings.
We attribute this to its direct alignment with the model’s
training objective in the single-answer setting and the ad-
ditional numerical instability introduced by SemGrad’s
semantic-proxy representations, as discussed in Section 3.3.
By combining the strengths of both approaches, our pro-
posed HybridGrad metric delivers consistently superior and
more stable performance in most settings, achieving the best
overall AUROC.
4.3. Importance of Semantic-Preserving Embeddings
To validate the importance of identifying the semantic-
preserving embeddings, we compute the correctness pre-
diction performance of SemGrad with respect to different
hidden states across layers and tokens. Specifically, we re-
place the h↑
t∗in Equation (4) with h(l)
−t for each layer l and
the last t-th tokens. We then compare the resulting AUROC
scores with the corresponding SPS scores for each hidden
state. The results are shown in Figure 3.
We observe a clear correlation between SPS and AUROC:
hidden states with higher SPS (better capturing input se-
mantic structure) yield stronger UQ performance with Sem-
Grad, whereas states with low SPS lead to weaker perfor-
mance. This finding underscores the necessity of identi-
fying semantic-preserving embeddings when computing
SemGrad. Meanwhile, the strong correlation suggests that
the performance of SemGrad is dependent on the semantic-
preserving capability of the hidden states on which it op-
erates, i.e., whether the hidden representations preserve
semantic structure effectively. This observation is consis-
tent with our core motivation that the output distribution of
an LLM should be relatively stable under small semantic-
preserving perturbations for confident inputs, rather than
under arbitrary random perturbations.
4.4. Ablation Study
We perform an ablation study on three components of Sem-
Grad: (1) the choice of norm, (2) the importance reweight-
ing mechanism, and (3) the embeddings (determined by
layer and token positions) with respect to which gradients
are computed. The results are presented in Table 2. There
are several findings. First, the l1-norm performs slightly
better than the l2-norm, though the difference is negligible.
Second, our proposed entropy weight ωt consistently im-
proves performance over methods without it, highlighting
its effectiveness at addressing linguistic redundancy. Third,
for the embeddings, those from the Semantic Preserving
Token t∗consistently outperform those from the last token.
This is consistent with the discussion in Section 4.3 and
the observation in Section 3.2 that the Semantic Preserving
Token captures most of the input semantics. However, when
varying the layer spans at t∗, performance differs, aligning
7

## Page 8

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Figure 3. Comparison of SemGrad UQ performance (AUROC) and semantic preservation capability (SPS) of different hidden states
across layers and tokens. Experiments are conducted on the last 5 input tokens of Llama3.1-Instruct8B and Qwen3-Instruct4B. A strong
correlation is observed: hidden states with higher semantic preservation capability yield better SemGrad performance.
Table 2. AUROC results of ablation study on SemGrad. We ablate
Equation (4) in three ways: (1) replacing l1 norm with l2 norm;
(2) removing the entropy weight ωt; (3) substituing the semantic
preserving embeddings h↑
t∗with embeddings from different layers
l and different token position t. t = −1 denotes the last token of
input.
Qwen3-Instruct4B
Llama3.1-Instruct8B
SciQ
TriviaQ
TruthfulQ
SciQ
TriviaQ
TruthfulQ
Proposed Method
SemGrad
72.20
80.40
69.06
75.76
84.72
69.42
Norm Function
SemGrad - l2 norm
72.07
80.59
68.65
75.73
84.82
69.42
Reweighting
SemGrad w/o ωt
71.39
76.83
67.79
74.19
81.28
68.98
Layer Span, t = t∗
l = L −1
72.47
79.92
69.45
74.64
84.55
68.13
l = L −4
72.30
78.65
70.09
75.68
84.30
69.46
l =
 2L
3

:(L−1)
72.43
79.57
69.23
75.67
84.26
69.34
l = 1 : (L −1)
71.60
80.48
66.60
75.37
85.36
67.41
Token Choice, l =
 L
2

:(L−1)
t = −1
70.49
79.30
63.35
74.28
83.94
69.07
with our observation in Section 3.2 that the peak span of
high SPS region varies across models and datasets. Among
these choices, our implementation (using hidden states from
the top half of layers) achieves the most stable performance.
5. Related Work
Gradient-based UQ Methods. Gradient-based approaches
estimate uncertainty from gradient information, and prior
works were developed for classification tasks. Lee & Al-
Regib (2020) firstly proposed to use the gradient as a mea-
sure of uncertainty and measured the gradient of the KL
divergence between the predicted label distribution and a
uniform prior. Igoe et al. (2022) proposed ExGrad, which
computes gradients of the log-likelihood of the predicted
class. Wang & Ji (2024) further extended ExGrad by weight-
ing gradients across classes and layers. However, many of
these methods require work on the whole prediction space,
which is infeasible for LLMs given the intractable output
space. Moreover, they assume a single ground-truth label,
which is problematic in free-form generation where multiple
plausible outputs exist.
UQ for Free-form Generation. Existing unsupervised
UQ methods for free-form generation can be grouped into
four categories (Shorinwa et al., 2024): (i) token-level UQ,
such as average log probability; (ii) self-verbalized UQ
(Kadavath et al., 2022; Tian et al., 2023), where the model is
prompted to report its own uncertainty; (iii) sampling-based
UQ (Kuhn et al., 2023; Duan et al., 2024; Lin et al., 2024;
Qiu & Miikkulainen, 2024), which estimates uncertainty
by measuring semantic similarity across sampled outputs;
and (iv) test-time augmentation-based UQ (Abbasi-Yadkori
et al., 2024), which derives uncertainty by perturbing the
input prompts. Among these, sampling-based methods have
achieved state-of-the-art performance (Kuhn et al., 2023;
Qiu & Miikkulainen, 2024), but their reliance on sampling
leads to high variance and significant computational cost.
6. Conclusion
In this work, we introduced the first gradient-based method,
SemGrad, for uncertainty quantification in free-form gener-
8

## Page 9

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
ation with LLMs. By leveraging the Semantic Preservation
Score to identify semantics-preserving embeddings and re-
weighting outputs to mitigate linguistic redundancy, our
method provides efficient and effective estimates of uncer-
tainty. We further proposed HybridGrad, combining se-
mantic and parameter gradients for improved generalization.
Experiments on QA benchmarks show that both methods
outperform state-of-the-art approaches, especially in cases
with multiple valid responses, highlighting semantic gradi-
ents as a promising direction for reliable UQ in LLMs.
Acknowledgements
We would like to thank all the anonymous reviewers for
their insightful comments. We thank the HIT SCIR-DT
group members for their valuable discussions and insightful
feedback. This work is supported by the National Science
and Technology Major Project (No. 2025ZD1606200 and
Sub-project No. 2025ZD1606203) and the National Natural
Science Foundation of China (No. 92470205).
Impact Statement
This paper presents work whose goal is to advance the field
of the reliability of modern LLMs. There are many potential
societal consequences of our work, none of which we feel
must be specifically highlighted here.
References
Abbasi-Yadkori, Y., Kuzborskij, I., Gy¨orgy, A., and
Szepesv´ari, C. To believe or not to believe your LLM:
iterative prompting for estimating epistemic uncertainty.
In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Pa-
quet, U., Tomczak, J. M., and Zhang, C. (eds.), Advances
in Neural Information Processing Systems 38: Annual
Conference on Neural Information Processing Systems
2024, NeurIPS 2024, Vancouver, BC, Canada, December
10 - 15, 2024, 2024.
Aichberger, L., Schweighofer, K., and Hochreiter, S.
Rethinking uncertainty estimation in llms: A princi-
pled single-sequence measure. In The Fourteenth In-
ternational Conference on Learning Representations,
ICLR 2026, 2026. URL https://arxiv.org/abs/
2412.15176.
Baan, J., Daheim, N., Ilia, E., Ulmer, D., Li, H., Fern´andez,
R., Plank, B., Sennrich, R., Zerva, C., and Aziz, W.
Uncertainty in natural language generation: From the-
ory to applications. CoRR, abs/2307.15703, 2023. doi:
10.48550/ARXIV.2307.15703. URL https://doi.
org/10.48550/arXiv.2307.15703.
Bakman, Y. F., Yaldiz, D. N., Buyukates, B., Tao, C.,
Dimitriadis, D., and Avestimehr, S. MARS: meaning-
aware response scoring for uncertainty estimation in
generative llms.
In Ku, L., Martins, A., and Sriku-
mar, V. (eds.), Proceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2024, Bangkok, Thai-
land, August 11-16, 2024, pp. 7752–7767. Association
for Computational Linguistics, 2024. doi: 10.18653/
V1/2024.ACL-LONG.419. URL https://doi.org/
10.18653/v1/2024.acl-long.419.
Bulian, J., Buck, C., Gajewski, W., B¨orschinger, B.,
and Schuster, T.
Tomayto, tomahto. beyond token-
level answer equivalence for question answering eval-
uation. In Goldberg, Y., Kozareva, Z., and Zhang, Y.
(eds.), Proceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing, EMNLP
2022, Abu Dhabi, United Arab Emirates, December
7-11, 2022, pp. 291–305. Association for Computa-
tional Linguistics, 2022.
doi:
10.18653/V1/2022.
EMNLP-MAIN.20.
URL https://doi.org/10.
18653/v1/2022.emnlp-main.20.
Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu,
Z., and Ye, J. INSIDE: llms’ internal states retain the
power of hallucination detection. In The Twelfth Inter-
national Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024. OpenReview.net,
2024. URL https://openreview.net/forum?
id=Zj12nzlQbz.
Chiarello, F., Giordano, V., Spada, I., Barandoni, S., and
Fantoni, G.
Future applications of generative large
language models: A data-driven case study on chatgpt.
Technovation, 133:103002, 2024.
ISSN 0166-4972.
doi: https://doi.org/10.1016/j.technovation.2024.103002.
URL
https://www.sciencedirect.com/
science/article/pii/S016649722400052X.
Duan, J., Cheng, H., Wang, S., Zavalny, A., Wang, C.,
Xu, R., Kailkhura, B., and Xu, K. Shifting attention to
relevance: Towards the predictive uncertainty quantifi-
cation of free-form large language models. In Ku, L.,
Martins, A., and Srikumar, V. (eds.), Proceedings of the
62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), ACL 2024,
Bangkok, Thailand, August 11-16, 2024, pp. 5050–5063.
Association for Computational Linguistics, 2024. doi: 10.
18653/V1/2024.ACL-LONG.276. URL https://doi.
org/10.18653/v1/2024.acl-long.276.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting
hallucinations in large language models using semantic
entropy. Nat., 630(8017):625–630, 2024. doi: 10.1038/
S41586-024-07421-0. URL https://doi.org/10.
1038/s41586-024-07421-0.
9

## Page 10

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Gawlikowski, J., Tassi, C. R. N., Ali, M., Lee, J., Humt,
M., Feng, J., Kruspe, A. M., Triebel, R., Jung, P.,
Roscher, R., Shahzad, M., Yang, W., Bamler, R., and
Zhu, X.
A survey of uncertainty in deep neural net-
works. Artif. Intell. Rev., 56(S1):1513–1589, 2023. doi:
10.1007/S10462-023-10562-9. URL https://doi.
org/10.1007/s10462-023-10562-9.
He, F., Lai, H., Liu, J., Wang, B., Chen, H., Liu, H.,
and Zhang, C. Solving mathematical problems using
large language models: A survey.
Data Intell., 7(4):
907–946, 2025. doi: 10.3724/2096-7004.DI.2025.0064.
URL https://doi.org/10.3724/2096-7004.
di.2025.0064.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang,
H., Chen, Q., Peng, W., Feng, X., Qin, B., and Liu,
T. A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions.
ACM Transactions on Information Systems, November
2024. ISSN 1558-2868. doi: 10.1145/3703155. URL
http://dx.doi.org/10.1145/3703155.
H¨ullermeier, E. and Waegeman, W. Aleatoric and epis-
temic uncertainty in machine learning: an introduction to
concepts and methods. Mach. Learn., 110(3):457–506,
2021. doi: 10.1007/S10994-021-05946-3. URL https:
//doi.org/10.1007/s10994-021-05946-3.
Igoe,
C.,
Chung,
Y.,
Char,
I.,
and Schneider,
J.
How useful are gradients for OOD detection really?
CoRR, abs/2205.10439, 2022. doi: 10.48550/ARXIV.
2205.10439. URL https://doi.org/10.48550/
arXiv.2205.10439.
Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. Trivi-
aqa: A large scale distantly supervised challenge dataset
for reading comprehension. In Barzilay, R. and Kan, M.
(eds.), Proceedings of the 55th Annual Meeting of the
Association for Computational Linguistics, ACL 2017,
Vancouver, Canada, July 30 - August 4, Volume 1: Long
Papers, pp. 1601–1611. Association for Computational
Linguistics, 2017. doi: 10.18653/V1/P17-1147. URL
https://doi.org/10.18653/v1/P17-1147.
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain,
D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma,
N., Tran-Johnson, E., Johnston, S., Showk, S. E., Jones,
A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman,
S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J.,
Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson,
C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph,
N., Mann, B., McCandlish, S., Olah, C., and Kaplan,
J.
Language models (mostly) know what they know.
CoRR, abs/2207.05221, 2022. doi: 10.48550/ARXIV.
2207.05221. URL https://doi.org/10.48550/
arXiv.2207.05221.
Kamalloo, E., Dziri, N., Clarke, C. L. A., and Rafiei, D.
Evaluating open-domain question answering in the era
of large language models. In Rogers, A., Boyd-Graber,
J. L., and Okazaki, N. (eds.), Proceedings of the 61st
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2023, Toronto,
Canada, July 9-14, 2023, pp. 5591–5606. Association
for Computational Linguistics, 2023. doi: 10.18653/
V1/2023.ACL-LONG.307. URL https://doi.org/
10.18653/v1/2023.acl-long.307.
Kuhn, L., Gal, Y., and Farquhar, S.
Semantic uncer-
tainty: Linguistic invariances for uncertainty estimation
in natural language generation. In The Eleventh Inter-
national Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net,
2023. URL https://openreview.net/forum?
id=VD-AYtP0dve.
Lee, J. and AlRegib, G. Gradients as a measure of uncer-
tainty in neural networks. In IEEE International Confer-
ence on Image Processing, ICIP 2020, Abu Dhabi, United
Arab Emirates, October 25-28, 2020, pp. 2416–2420.
IEEE, 2020.
doi: 10.1109/ICIP40778.2020.9190679.
URL https://doi.org/10.1109/ICIP40778.
2020.9190679.
Li, M. and Subramani, N. Model internal sleuthing: Finding
lexical identity and inflectional morphology in modern
language models. CoRR, abs/2506.02132, 2025. doi:
10.48550/ARXIV.2506.02132. URL https://doi.
org/10.48550/arXiv.2506.02132.
Li, M., Li, X., Zhang, W., and Ma, L. ESI: epistemic un-
certainty quantification via semantic-preserving interven-
tion for large language models. CoRR, abs/2510.13103,
2025. doi: 10.48550/ARXIV.2510.13103. URL https:
//doi.org/10.48550/arXiv.2510.13103.
Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring how
models mimic human falsehoods. In Muresan, S., Nakov,
P., and Villavicencio, A. (eds.), Proceedings of the 60th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2022, Dublin,
Ireland, May 22-27, 2022, pp. 3214–3252. Association
for Computational Linguistics, 2022. doi: 10.18653/
V1/2022.ACL-LONG.229. URL https://doi.org/
10.18653/v1/2022.acl-long.229.
Lin, Z., Trivedi, S., and Sun, J. Generating with confi-
dence: Uncertainty quantification for black-box large
language models.
Trans. Mach. Learn. Res., 2024,
2024. URL https://openreview.net/forum?
id=DWkJCSxKU5.
Malinin, A. and Gales, M. J. F. Uncertainty estimation in
autoregressive structured prediction. In 9th International
10

## Page 11

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Conference on Learning Representations, ICLR 2021,
Virtual Event, Austria, May 3-7, 2021. OpenReview.net,
2021. URL https://openreview.net/forum?
id=jN5y-zb5Q7m.
Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh,
P. W., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H.
Factscore: Fine-grained atomic evaluation of factual
precision in long form text generation.
In Bouamor,
H., Pino, J., and Bali, K. (eds.), Proceedings of the
2023 Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore, December
6-10, 2023, pp. 12076–12100. Association for Com-
putational Linguistics, 2023. doi: 10.18653/V1/2023.
EMNLP-MAIN.741. URL https://doi.org/10.
18653/v1/2023.emnlp-main.741.
Mohri, C. and Hashimoto, T.
Language models with
conformal factuality guarantees. In Forty-first Interna-
tional Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024. OpenReview.net,
2024. URL https://openreview.net/forum?
id=uYISs2tpwP.
Naveed, H., Khan, A. U., Qiu, S., Saqib, M., An-
war, S., Usman, M., Barnes, N., and Mian, A.
A
comprehensive overview of large language models.
CoRR, abs/2307.06435, 2023. doi: 10.48550/ARXIV.
2307.06435. URL https://doi.org/10.48550/
arXiv.2307.06435.
Qiu, X. and Miikkulainen, R. Semantic density: Uncertainty
quantification for large language models through confi-
dence measurement in semantic space. In Globersons, A.,
Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak,
J. M., and Zhang, C. (eds.), Advances in Neural Infor-
mation Processing Systems 38: Annual Conference on
Neural Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15, 2024,
2024.
Raza, M., Jahangir, Z., Riaz, M. B., Saeed, M. J., and Sattar,
M. A. Industrial applications of large language models.
Scientific Reports, 15(1):13755, 2025. ISSN 2045-2322.
doi: 10.1038/s41598-025-98483-1. URL https://
doi.org/10.1038/s41598-025-98483-1.
Shorinwa, O., Mei, Z., Lidard, J., Ren, A. Z., and Ma-
jumdar, A. A survey on uncertainty quantification of
large language models: Taxonomy, open research chal-
lenges, and future directions. CoRR, abs/2412.05563,
2024. doi: 10.48550/ARXIV.2412.05563. URL https:
//doi.org/10.48550/arXiv.2412.05563.
Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R.,
Yao, H., Finn, C., and Manning, C. D. Just ask for calibra-
tion: Strategies for eliciting calibrated confidence scores
from language models fine-tuned with human feedback.
In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2023, Singapore,
December 6-10, 2023, pp. 5433–5442. Association for
Computational Linguistics, 2023. doi: 10.18653/V1/
2023.EMNLP-MAIN.330. URL https://doi.org/
10.18653/v1/2023.emnlp-main.330.
Wang, H. and Ji, Q. Epistemic uncertainty quantification for
pretrained neural networks. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, CVPR 2024,
Seattle, WA, USA, June 16-22, 2024, pp. 11052–11061.
IEEE, 2024.
doi: 10.1109/CVPR52733.2024.01051.
URL https://doi.org/10.1109/CVPR52733.
2024.01051.
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J.,
Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z.,
and Wen, J. A survey on large language model based au-
tonomous agents. Frontiers Comput. Sci., 18(6):186345,
2024. doi: 10.1007/S11704-024-40231-1. URL https:
//doi.org/10.1007/s11704-024-40231-1.
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi,
E. H., Narang, S., Chowdhery, A., and Zhou, D.
Self-consistency improves chain of thought reason-
ing in language models.
In The Eleventh Interna-
tional Conference on Learning Representations, ICLR
2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net,
2023. URL https://openreview.net/forum?
id=1PL1NIMMrw.
Welbl, J., Liu, N. F., and Gardner, M.
Crowdsourcing
multiple choice science questions. In Derczynski, L.,
Xu, W., Ritter, A., and Baldwin, T. (eds.), Proceed-
ings of the 3rd Workshop on Noisy User-generated Text,
NUT@EMNLP 2017, Copenhagen, Denmark, Septem-
ber 7, 2017, pp. 94–106. Association for Computational
Linguistics, 2017. doi: 10.18653/V1/W17-4413. URL
https://doi.org/10.18653/v1/w17-4413.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D.,
Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H.,
Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J.,
Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L.,
Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang,
P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T.,
Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren,
X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu,
Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z.
Qwen3 technical report. CoRR, abs/2505.09388, 2025.
doi: 10.48550/ARXIV.2505.09388. URL https://
doi.org/10.48550/arXiv.2505.09388.
11

## Page 12

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang,
X., Zhao, E., Zhang, Y., Chen, Y., Wang, L., Luu,
A. T., Bi, W., Shi, F., and Shi, S.
Siren’s song in
the AI ocean: A survey on hallucination in large lan-
guage models.
CoRR, abs/2309.01219, 2023.
doi:
10.48550/ARXIV.2309.01219. URL https://doi.
org/10.48550/arXiv.2309.01219.
12

## Page 13

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
A. Limitation
Our approach works in a white-box setting, meaning it requires access to both the model’s gradients and internal weights.
Such access is generally unavailable for closed-source APIs. Nevertheless, when applied to open-source models, our
methods prove to be highly competitive.
In addition, our work primarily targets claim-level predictions (i.e., short answers) as our baselines did. Performance may
decline on long-form outputs, where gradient signals can be diluted across numerous correct and less informative tokens.
However, claim-level evaluation is widely adopted as a building block for long-form assessment methods, since longer
responses are often segmented into individual claims before evaluation (Min et al., 2023; Mohri & Hashimoto, 2024).
Consequently, our approach can be integrated into long-form pipelines, and its efficiency and accuracy make it a valuable
and competitive component.
B. Efficiency Analysis
Computation Efficiency. To demonstrate the computation efficiency of our method, we evaluate the average per-example
runtime, as shown in Table 3. Both of our proposed gradient-based methods, SemGrad and HybridGrad, consistently run
faster than the sampling-based baselines by a large margin.
We observe that computing parameter gradients (i.e., the difference between HybridGrad and SemGrad runtime) is nearly ten
times faster than computing SemGrad. This discrepancy mainly arises from implementation constraints in the transformers
library4. When using torch.autograd.grad5, the input must remain within the computation graph of the output loss. Although
hidden states produced by the framework do participate in the loss computation, indexing them directly results in sub-tensors
that are no longer tracked in the loss graph. Consequently, we are forced to compute gradients with respect to all hidden
states in the input sequence rather than one positions in later steps, which introduces substantial computational overhead.
This also accounts for the slower runtime of SemGrad on SciQ compared to TruthfulQA, as SciQ queries are typically
longer, even though the answers are shorter.
For our current purposes, the existing implementation is sufficiently efficient. Nevertheless, we emphasize that SemGrad in
principle could be made considerably faster with targeted engineering optimizations.
Memory Efficiency. Our method requires a single forward and backward pass through the model, which does incur
additional memory overhead for storing activations, similar to a standard training step. Concretely, the memory scales as
O(L · T · D), where L is the number of layers, T the sequence length, and D the hidden size. In principle, the dependence
on T can be further reduced since gradients are only required at a small number of token positions.
In contrast, while sampling-based methods do not require storing backward activations, they require K independent forward
passes with K generated outputs. This process requires caching the key-value (KV) pairs, resulting in a memory scaling
as K · O(L · T · D). Many methods additionally store per-sample embeddings (Chen et al., 2024) or similarity structures
(Kuhn et al., 2023) and, in some cases, rely on auxiliary models for semantic comparison (Kuhn et al., 2023; Duan et al.,
2024; Qiu & Miikkulainen, 2024). As a result, their memory grows with the number of samples K, and in many cases,
includes additional storage for other operations.
C. Implementation Details
C.1. Semantic Preservation Score Implementation Details
We evaluate our proposed Semantic Preservation Score (SPS) on three datasets—TriviaQA, SciQ, TruthfulQA—and three
models: Qwen3-Instruct4B, Mistral-Nemo-Instruct12B, Llama3.1-Instruct8B. The full results are shown in Figure 4. For each
query in each dataset, we prompt DeepSeek API6 to generate five paraphrases. Each query and its paraphrases are then
passed through each model to obtain the corresponding hidden states at all layers and token positions.
To validate the quality of our generated paraphrases, we conduct a small-scale validation on TruthfulQA to assess how well
the generated paraphrases preserve semantic meaning. We evaluate semantic consistency using two independent methods:
4https://huggingface.co/docs/transformers/index
5https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
6https://api-docs.deepseek.com/
13

## Page 14

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Table 3. Average runtime per example (in seconds), measured with Llama3.1-Instruct8B on a single NVIDIA A100 80GB GPU under
single-batch inference. All methods are evaluated under the same experimental conditions as in the main results. “+” denoted the
additional runtime needed compared to pure generation.
UQ methods
SciQ
TriviaQ
TruthfulQA
Pure Generation
0.2088
0.2089
0.1467
SemGrad
+0.2506
+0.2577
+0.1979
HybridGrad
+0.2780
+0.2878
+0.2287
SAR
+0.3632
+0.4715
+0.5542
Semantic Entropy
+0.3754
+0.5093
+0.5790
Semantic Density
+1.6502
+1.7173
+1.8917
(i) an NLI-based judge (DeBERTa-large trained on MNLI), where we assign a score of 1 if the paraphrase is classified
as entailment, and 0 otherwise; and (ii) an LLM-based judge (Llama3-Instruct-70B), where we prompt the model with a
Yes/No question regarding semantic equivalence, assigning 1 if the response contains “Yes”. The NLI-based judge yields a
consistency score of 90.08, and the LLM-based judge achieves 98.72, indicating that our paraphrase generation process
reliably preserves the original semantic meaning.”
C.2. Baseline Implementation Details
In this section, we provide an overview of the baseline methods used in our work along with their implementation settings.
Length-Normalized Predictive Entropy (Malinin & Gales, 2021). LN-PE estimates entropy in the output space through
Monte Carlo sampling, where sentence log-probabilities are normalized by length. Since the original work employed an
ensemble of models, we instead follow the configuration from (Kadavath et al., 2022), generating 10 samples at temperature
1.0.
P(True) (Kadavath et al., 2022). P(True) directly prompts the model to judge the correctness of its own responses, and the
probability assigned to the label “True” is taken as the uncertainty score. We adopt the same prompt template provided in
the original paper.
Self-Consistency (Wang et al., 2023). Self-Consistency computes the uncertainty score based on the fraction of sampled
responses that are semantically equivalent to the greedy-decoded output. Following prior work, we generate 10 responses
using temperature 0.7 and top-p 1.0. Semantic equivalence is assessed using the Deberta-large model7 trained on MNLI.
Deg (Lin et al., 2024). Deg applies spectral clustering to the similarity matrix of sampled responses and derives the
uncertainty score from the degree matrix, which essentially corresponds to the average pairwise similarity. The experimental
setup follows that of Self-Consistency.
INSIDE (Chen et al., 2024). INSIDE quantifies uncertainty by analyzing the variability in semantic embeddings of sampled
outputs via eigenvalues. In line with the original configuration, we set the sampling parameters to temperature 0.5, top-p
0.99, top-k 5, and generate 10 responses. The sentence embedding is taken as the final token embedding from a middle layer
of the model.
Semantic Entropy (Kuhn et al., 2023). Semantic Entropy accounts for semantic equivalence by clustering outputs with
similar meaning, then computing entropy across the clusters. We adopt the journal version (Farquhar et al., 2024), which
samples 10 generations at temperature 1.0. Semantic similarity is measured using the same function as in Self-Consistency.
Semantic Density (Qiu & Miikkulainen, 2024). Semantic Density uses kernel density estimation with Epanechnikov kernel
to estimate out probability density with sampled responses. The uncertainty score is derived from the probability assigned
by this estimated density. We follow the configuration from the original paper, which samples 10 responses with diverse
beam search with diversity penalty 1.0 and beams group 10, renormalize the token output probability with temperature
0.1, evaluate the semantic similarity (distance in their words) with the same similarity function identical to that of the
self-consistency method, then follow Algorithm 1 in the original paper to calculate the semantic density scores.
7deberta-large
14

## Page 15

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Figure 4. Semantic Preservation Score (SPS) of hidden states across different layers and tokens. We experiment on the last 10 input tokens,
where “last #t token” denotes the last t-th token from the end of the user query (corresponding token is different for different queries). We
observe that the token position carrying the most semantic information is consistent for the same model across different datasets.
M.I. (Abbasi-Yadkori et al., 2024). M.I. assumes that outputs sampled from the same query are independent, and evaluates
uncertainty via mutual information between them. We implement Algorithm 3 from the original paper: 10 responses are
sampled at temperature 0.9, answers are clustered with F1 matching (probabilities aggregated when F1 > 0.25), and the
uncertainty is computed from the mutual information of two independently prompted responses (n = 2) with stabilization
parameters γ1 = 0 and γ2 = 0.
G-NLL (Aichberger et al., 2026). G-NLL is a simple sampling-free method that directly evaluates the negative log-likelihood
probability of the most likely output sequence.
SAR (Duan et al., 2024). SAR, the current state-of-the-art baseline, refines uncertainty estimation by applying importance
weighting to prioritize informative tokens and sentences. In line with the original configuration, we sample five generations
for instructed LLMs and temperature to 1.0. We utilize Cross-Encoder-Roberta-Large 8 to evaluate token importance and
sentence importance as the original paper did.
ExGrad (Igoe et al., 2022). ExGrad is designed for classification model which computes the empirical expectation of
gradients of the log-likelihood of prediction labels with respect to the output layer weights (weights used for producing
8cross-encoder/stsb-roberta-large
15

## Page 16

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
prediction logits). For large language models, this expectation is impractical because it requires integrating over the entire
response space, and even sampling-based approximations are inefficient. To make it feasible, we compute the gradient of the
log-likelihood for the generated responses directly.
C.3. Datasets
TriviaQA (Joshi et al., 2017). TriviaQA contains factual question-answer pairs collected from trivia and quiz league
websites. Each question is associated with a single semantically correct ground-truth answer. For our experiments, we use
the test split of the open-domain setting, which includes 11,313 examples.
SciQ (Welbl et al., 2017). SciQ is composed of science exam questions spanning subjects such as chemistry, physics, and
biology. Similar to TriviaQA, each question has a single ground-truth answer in meaning. Following Duan et al. (2024), we
evaluate on the validation split, which consists of 1,000 questions.
TruthfulQA (Lin et al., 2022). TruthfulQA includes 817 questions across 38 categories, many of which are designed to
expose misconceptions or false beliefs. Many of these questions are open-ended, such as ”What happens to you if you eat
watermelon seeds?”, which naturally introduce higher levels of aleatoric uncertainty. An example is given in the below box.
Experiments are performed on the entire set of 817 examples.
Query: What is the spiciest part of a chili pepper?
Valid Responses:
• The spiciest part of a chili pepper is the pith
•The spiciest part of a chili pepper is the ribs
•The spiciest part of a chili pepper is the placenta
•The spiciest part of a chili pepper is the membrane
•The spiciest part of a chili pepper is the capsaicin glands
•It’s a common misconception that the spiciest part of a chili pepper is the seeds. It’s actually the pith
C.4. Prompt Templates
Template for Question Answering:
{query} represents the placeholder to insert query.
Please directly answer the following question with one or few words:
{query}
D. Additional Experiments
D.1. LLM-as-a-Judge for Correctness Evaluation
We choose BEM (Bulian et al., 2022) as the primary correctness evaluation metric because it is reproducible, cost-free,
and computationally lightweight, and prior work has shown that it is effective and consistent with human annotation for
evaluating short-form QA (Kamalloo et al., 2023).
Since LLM-as-a-judge, while more computationally and economically expensive, is generally considered a finer-grained
evaluation approach, we additionally conduct experiments using an LLM-based correctness evaluator (via the DeepSeek
API9) on the same generations produced by Llama3.1-8B-Instruct (see Table 4). The resulting rankings and relative
performance trends under BEM and LLM-as-a-judge are highly consistent, and our proposed methods continue to achieve
superior performance under both metrics. These results indicate that our main conclusions are robust to the choice of
correctness metric.
9https://api-docs.deepseek.com/
16

## Page 17

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Table 4. AUROC Comparison between LLM-as-a-Judge and BEM as Correctness Evaluation Metrics.
UQ Methods
SciQ
TriviaQA
TruthfulQA
Avg.
LLM
BEM
LLM
BEM
LLM
BEM
LLM
BEM
LN-PE
73.23
72.51
86.10
84.53
57.26
63.38
72.20
73.47
S.E.
74.06
70.27
85.92
83.12
58.33
59.59
72.77
70.99
S.D.
75.73
74.00
84.39
82.44
53.59
57.75
71.24
71.40
M.I.
73.57
72.43
84.60
83.52
55.01
64.25
71.06
73.40
G-NLL
74.53
75.49
87.06
85.91
54.33
57.51
71.97
72.97
SAR
76.76
75.28
86.82
85.65
59.07
64.44
74.22
75.12
ExGrad
73.87
74.11
86.35
85.22
57.27
62.00
72.50
73.78
ParaGrad
75.31
74.98
87.68
86.49
58.48
63.91
73.82
75.13
SemGrad
77.42
75.76
85.76
84.72
65.97
69.42
76.38
76.63
HybridGrad
77.76
76.31
87.03
85.89
65.35
69.25
76.71
77.15
Table 5. AURC of different UQ methods on generation correctness prediction. A smaller value indicates better UQ performance. The
bold number represents the best performance across all methods for each dataset–model pair. The Avg. columns report the average AURC
performance across all datasets and models.
UQ Methods
Qwen3-Instruct4B
Mistral-Nemo-Instruct12B
Llama3.1-Instruct8B
Avg.
SciQ
TriviaQ
TruthfulQ
SciQ
TriviaQ
TruthfulQ
SciQ
TriviaQ
TruthfulQ
LN-PE
26.90
33.69
47.74
23.84
13.27
50.97
24.16
12.64
55.78
32.11
P(True)
33.48
36.51
57.24
27.04
14.21
58.22
28.38
15.10
61.84
36.89
Self-Con
30.29
37.32
44.87
30.68
15.82
48.56
25.72
14.92
61.37
34.39
Deg
30.28
35.91
48.17
26.81
14.70
49.52
24.78
13.44
62.69
34.03
INSIDE
33.72
42.41
44.61
30.64
18.51
51.97
24.03
16.48
60.43
35.87
S.E.
35.01
40.34
45.97
34.10
17.77
49.39
29.02
15.47
60.76
36.43
S.D.
30.83
36.48
52.08
27.21
16.34
51.65
23.64
14.02
62.38
34.96
M.I.
26.98
37.34
46.56
26.73
14.69
46.20
25.71
14.19
54.03
32.49
G-NLL
21.29
30.54
45.93
23.03
12.14
49.67
21.54
11.71
59.08
30.55
SAR
21.72
30.84
42.70
22.92
12.00
47.49
21.76
11.77
54.35
29.50
ExGrad
21.83
30.73
44.45
22.96
12.20
48.96
22.09
11.93
57.15
30.25
ParaGrad
21.67
30.14
43.42
22.56
11.72
46.16
21.88
11.51
56.15
29.47
SemGrad
21.66
30.77
41.74
23.34
12.96
44.86
21.45
11.91
51.99
28.97
HybridGrad
21.44
30.27
41.58
22.72
12.27
44.54
21.18
11.50
52.35
28.65
D.2. Additional Experiments on More Evaluation Metrics
We provide additional experimental results using AURC (Area Under the Risk–Coverage Curve, Table 5) as the evaluation
metric.
The results under AURC are consistent with the conclusions drawn from AUROC: our proposed methods achieve the best
average performance across baselines. Parameter-gradient methods (ExGrad and ParaGrad) perform well in single–ground-
truth settings, where SemGrad also achieves comparable results. In high-aleatoric settings, SemGrad remains stable while
parameter-gradient methods degrade substantially, further supporting our analysis.
D.3. Additional Ablation Study on HybridGrad Balancing Weight e−¯ω
The upper panel of Figure 5 shows the histogram of the average per-token entropy ¯ω from Llama3.1-Instruct8B outputs.
TruthfulQA exhibits a broad entropy distribution with fewer extremely low-entropy samples, reflecting the inherently high
aleatoric nature of many of its prompts. In contrast, SciQ and TriviaQA produce predominantly low-entropy responses,
consistent with their single-answer factoid-style questions. This supports using ¯ω as a practical proxy for the sharpness of
the model’s ground-truth distribution.
17

## Page 18

Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
Figure 5. Upper: The upper panels show the histogram of the average per-token entropy ¯ω of responses generated by Llama3.1-Instruct8B
on TruthfulQA, SciQ, and TriviaQA (left to right). The darker blue histogram corresponds to ¯ω for correct generations, while the lighter
blue histogram corresponds to ¯ω for all generations. The two vertical dashed lines indicate the 50th and 75th percentiles of the ¯ω
distribution for correct generations. Lower: The lower panels plot the AUROC performance with varying ¯ω scaling coefficient τ and the
ParaGrad scaling coefficient β for the same three datasets, aligned column-wise with the upper panels.
The balancing weight α = e−¯ω reflects the sharpness of the predictive distribution and scales the value range to [0, 1].
To provide a complete picture of how the balancing weight influence the performance of HybridGrad, we introduce two
additional hyperparameters: a scaling coefficient τ and a ParaGrad scaling coefficient β as follows:
¯SHybridGrad = (1 −ατ)SSemGrad + βατSParaGrad
We redefine the weight as ατ = e−¯
ω
τ , where smaller τ causes ατ to decay rapidly as entropy increases—biasing HybridGrad
toward SemGrad—whereas larger τ slows the decay and emphasizes ParaGrad. The coefficient β compensates for magnitude
differences between the two gradient types and directly modulates HybridGrad’s reliance on ParaGrad.
The influence of τ and β is shown in the lower panel of Figure 5, and the results are consistent with the above analysis: when
τ is extremely small, the performance of HybridGrad converges to that of SemGrad, while it approaches the performance of
ParaGrad as τ increases. Similarly, HybridGrad leans more toward ParaGrad when β is larger.
Although both SciQ and TriviaQA exhibit low-entropy patterns, SciQ has more overconfident erroneous predictions, as
indicated by the larger discrepancy between the two histograms in the low-entropy region. TriviaQA, by contrast, has
far fewer confident wrong answers, meaning that sharpness is more predictive of correctness than in SciQ. Consequently,
ParaGrad—which directly measures distribution sharpness—tends to behave more stably and achieves better empirical
performance on TriviaQA, as illustrated by the dashed line in the lower panel. In contrast, SemGrad, which is independent
of the sharpness of the model’s predictive distribution, performs significantly better on TruthfulQA, where multiple valid
answers exist and correctness is less coupled to predictive sharpness. For SciQ, which lies between these two extremes,
ParaGrad and SemGrad achieve comparable performance. Interestingly, on a mixed dataset such as SciQ, appropriately
combining SemGrad and ParaGrad can further boost performance, as shown in the middle figure of the lower panel.
Generally, when choosing
E. The Use of Large Language Models
LLMs are used to polish the language of some parts of our original content and to generate parts of simple, repetitive, and
non-novel code, such as plotting.
18
