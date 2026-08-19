---
source_pdf: papers/Beyond Next Token Probabilities Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions.pdf
slug: beyond-next-token-probabilities-learnable-fast-detection-of
pages: 39
extracted_on: 2026-07-21
---

# Beyond Next Token Probabilities Learnable, Fast Detection of Hallucinations and Data Contamination on LLM Output Distributions

## Page 1

Beyond Next Token Probabilities: Learnable, Fast Detection of
Hallucinations and Data Contamination on LLM Output
Distributions
Guy Bar-Shalom*
Fabrizio Frasca*
Technion
Technion
guybs99@gmail.com
fabriziof@campus.technion.ac.il
Derek Lim
Yoav Gelberg
Yftah Ziser
Ran El-Yaniv
Gal Chechik
MIT CSAIL
Technion
Nvidia
Technion,
Nvidia
Bar-Ilan University,
Nvidia
Haggai Maron
Technion,
Nvidia
Abstract
The automated detection of hallucinations and training data contamination is pivotal to the
safe deployment of Large Language Models (LLMs). These tasks are particularly challenging
in settings where no access to model internals is available. Current approaches in this setup
typically leverage only the probabilities of actual tokens in the text, relying on simple task-specific
heuristics. Crucially, they overlook the information contained in the full sequence of next-token
probability distributions. We propose to go beyond hand-crafted decision rules by learning
directly from the complete observable output of LLMs — consisting not only of next-token
probabilities, but also the full sequence of next-token distributions. We refer to this as the
LLM Output Signature (LOS), and treat it as a reference data type for detecting hallucinations
and data contamination. To that end, we introduce LOS-Net, a lightweight attention-based
architecture trained on an efficient encoding of the LOS, which can provably approximate
a broad class of existing techniques for both tasks. Empirically, LOS-Net achieves superior
performance across diverse benchmarks and LLMs, while maintaining extremely low detection
latency. Furthermore, it demonstrates promising transfer capabilities across datasets and LLMs.
Full code is available at https://github.com/BarSGuy/Beyond-next-token-probabilities.
1
Introduction
As the remarkable capabilities of LLMs continue to drive their expanding range of applications,
detecting hallucinations [46, 30, 19, 22, 42], and training data contamination [7, 43, 58] becomes
increasingly important to their reliable deployment and responsible use. Specifically, the tasks of
Hallucination and Data Contamination Detection (resp. HD, DCD) relate to determining whether
∗Equal contribution.
1
arXiv:2503.14043v3  [cs.LG]  30 Sep 2025

## Page 2

A big mouse
LLM
What does the 
cat chase?
Query token 
probabilities
0,2
0,4
0,6
0,8
0,2
0,4
0,6
0,8
What
…
…
does
…
0,125
0,25
0,375
0,5
 chase?
…
…
…
…
…
…
TDS (query)
ATP (query)
TDS (resp.)
0,15
0,3
0,45
0,6
0,2
0,4
0,6
0,8
…
…
mouse
A
…
…
does
the
A
big
mouse
…
…
…
…
<EOS>
big
0,5
1
1,5
2
…
…
Token Vocabulary
Sampled 
response token 
probabilities
ATP (resp.)
LLM Output Signature (LOS)
does
the
An
you
…
do
…
…
…
A
Token Vocabulary
Figure 1: Left: The LLM processes the input “What does the cat chase?” and generates the output
“A big mouse”. Right: The corresponding query/response Token Distribution Sequences (TDS) and
Actual Token Probabilities (ATP), together constituting the LLM Output Signature (LOS). We
propose to detect instances of hallucinations and data contamination by learning directly over this
unified data representation, beyond task specific heuristics operating on partial information thereof.
an LLM is fabricating information or is providing an incorrect answer to a user question, or whether
it has been exposed to specific training data, such as copyrighted material.
Prominent methods to tackle HD include probing techniques which, although effective, require
restrictive white-box access to model internals [4, 37, 18, 17, 41], such as its hidden states. On both
HD and DCD, gray-box methods relax these assumptions by operating only on LLM outputs, thus
finding application to a broader set of models. These approaches [16, 24, 49, 20] typically extract
simple features on the sequence of token probabilities, a vector we term Actual Token Probabilities
(ATP). However, these methods, often based on heuristics, overlook the information contained in the
complete next-token probability distributions generated over the token vocabulary at each generation
step – we term this matrix the Token Distribution Sequence (TDS), see Figure 1. Importantly,
this limitation can mask distinctive patterns in the model’s text generation process, including its
confidence or uncertainty, known to correlate with its correctness [25, 13]. This aspect is evident even
at the level of a single generation step. Consider, e.g., an LLM generating a token with probability 0.5
in two scenarios: in one case, the remaining next-token probability mass is concentrated on a single
alternative (0.5, 0.5, 0, ..., 0), while in the other it is spread across many tokens: (0.5, 0.01, ..., 0.01).
See Figure 12 in Section F for an illustration of a similar case. Yet ATP-based approaches would
treat them identically. Similarly, an ATP value of 0.1 at a certain time step could indicate either
high uncertainty (if it is the highest probability in a diffused distribution) or strong evidence against
the token (if it is a low-ranking probability in a peaked distribution). A recent promising approach
[58] used some TDS information using heuristics, but a principled framework to utilize this data
source is still lacking.
Our approach. We argue that a successful gray-box detection approach should leverage both
ATP and TDS, together forming what we term the LLM Output Signature (LOS) (Figure 1) –
the complete observable representation of an LLM in the gray-box setup. Instead of relying on
heuristics, we treat LOS as a sequential, high-dimensional and structured data modality on which
we apply principled deep learning techniques. We propose LOS-Net, an efficient attention-based
model1 operating on an effective encoding of ATP, TDS, and their interactions. We prove that
LOS-Net can approximate a broad class of functions applied to the LOS, subsuming many recent
approaches [16, 24, 49, 20, 43, 58]. Our comprehensive empirical study on DCD and HD demonstrates
1Our model features around 1M parameters.
2

## Page 3

a substantial performance gap between using the complete LOS and relying solely on the ATP.
Notably, LOS-Net improves over all considered baselines across both tasks, often by a significant
margin. Crucially, our architecture is extremely efficient, with detection times of ∼10−5s per
instance. This makes it a compelling approach for applications such as on-line error detection for
guided-generation, as opposed to previously proposed popular methods based on multiple LLM
prompting or generations, such as Semantic Entropy (SE) and P(True) [25, 24]. LOS-Net also
exhibits promising dataset-level transfer and strong cross-model generalization, the latter suggesting
its viable application to real-world tasks such as copyright-infringement detection over closed-source
LLMs (see, e.g., our results on the BookMIA benchmark [43] in Section 5.2). Last, we show LOS-Net
retains strong performance also when processing a small subset of the TDS, expressed in terms of
the number of top-scoring output probabilities at each generation step. This extends its successful
application to LLMs with restricted API access, such as GPT models [36].
Contributions. (1) We introduce LOS as a suitable “data type” for the detection of hallucinations
and data contamination, and develop LOS-Net, an effective and efficient learning framework for that.
(2) We show this framework unifies and generalizes previous approaches, and demonstrate it achieves
superior performance across models, datasets, and tasks. (3) We find that LOS-Net exhibits strong
empirical evidence for cross-model generalization and promising cross-dataset transfer abilities.
2
Related Work
We review background and related work on DCD and HD, focusing on studies leveraging logits or
output probabilities. Given the breadth of research, we highlight the most relevant works for our
setup and refer interested readers toC for further details on these tasks.
Data Contamination Detection (DCD). This task consists in identifying text passages an LLM
has likely seen during training, or memorized. This is crucial for ensuring fair benchmarking of LLMs,
guiding dataset curation, and auditing potential copyright infringement. Early methods leveraged
model loss [55, 9], assuming that models overfit their training data. Later refinements introduced
reference models – independent LLMs trained on disjoint datasets from a similar distribution –
comparing their scores with the target model [10, 11]. However, this approach requires access
to a well-matched reference model with similar architecture, which is often impractical in real-
world settings. Recently, [43] introduced Min-K%, which flags an input as contaminated if the
log probability of its bottom K tokens exceeds a predefined threshold. Building on this approach,
[58] proposed Min-K%++, which refines contamination detection by calibrating the next-token
log-likelihood using the mean and standard deviation of log-likelihoods across all candidate tokens
in the vocabulary.
Hallucination Detection (HD). This task has been studied to enable selective intervention,
allowing LLMs to prevent fabricated outputs only when necessary [45, 56, 48]. Recently, [37, 3]
showed that training a classifier on top of LLMs’ hidden states is highly effective for hallucination
detection. However, these methods operate under the white-box assumption, requiring full access to
the model’s internal states. In contrast, our paper explores a more constrained (gray-box) setting, of
particular interest especially when targeting closed-source LLMs with restricted API access.
Output probability-Based Analysis. Previous works showed that using log probabilities or
raw logits as decision thresholds can be effective for various tasks, including HD in LLMs [16, 49],
correctness self-evaluation [24], uncertainty estimation [20], and zero shot learning [2]. However,
these approaches often rely on naive handcrafted thresholding. Other works [51, 35] rely on linear
classifiers over features extracted from LLM outputs aiming at tackling adjacent tasks, such as
3

## Page 4

detecting machine-generated text [52] but overlook the full TDS, limiting contextual understanding.
3
Learning on LLM Output Signatures
In this section, we define the LLM Output Signature (LOS) and introduce LOS-Net, a novel
architecture specifically designed to process LOS.
3.1
LLM Output Signatures (LOS)
Let f denote a pretrained LLM, and ⃗s a text input to f consisting of n tokens. When queried with
⃗s, f produces outputs Xs = f(⃗s), i.e., a matrix in Rn×V of next-token probabilities for each token in
⃗s, where V is the size of the token vocabulary. We define ⃗g to be the LLM response to ⃗s, consisting
of m tokens generated using f’s outputs in Xg ∈Rm×V (and Xs). We refer to Xs or Xg as Token
Distribution Sequences (TDS). See Figure 11, Section E. We also define ps ∈Rn, pg ∈Rm, vectors
which holds the probabilities associated with the actual tokens appearing in ⃗s,⃗g respectively. We
denote these as the Actual Token Probabilities (ATP). Specifically, (ps)i := Xi,v where v ∈{1, . . . , V }
is the token used in the i + 1 place in the sequence ⃗s and similarly for ⃗g (see Figure 1). We call the
pairs (Xs, ps) or (Xg, pg) the LLM Output Signature (LOS). For DCD, we analyze input sequences
using (Xs, ps) as our interest lies in how the model processes the input text ⃗s. For HD, we use
(Xg, pg) as we need to make predictions on the model’s response. We may use (X, p) if the distinction
between the tasks is irrelevant, and use N as the sequence length.
Problem Statement. LOS elements, along with their associated annotations depending on the
task of interest, can be gathered into datasets D = {
 (X, p)i, yi

}ℓ
i=1 where supervised learning
problems can be instantiated. Our goal in this paper is to propose a neural architecture that can
effectively utilize the complete LOS to solve tasks such as DCD, HD, or any other classification
problem thereon.
3.2
LOS-Net
Learning from LOS data objects presents inherent challenges, particularly related to their encoding.
Next, we detail these challenges and introduce our LOS-Net approach, illustrated in full in Section D
and Figure 10. What follows is a detailed explanation of each of its components.
Preprocessing the token distribution sequences. Utilizing X may pose significant challenges
due to three key factors. (1) Complexity: The vocabulary tensor can be extremely large in real-world
scenarios. For instance, Liang et al. [29] (XLM-V) reported a vocabulary size of 1M tokens, which,
for a small batch of documents and popular context sizes, would already entail processing a tensor
of tens (or hundreds) of GBs. (2) Transferability: Vocabulary size and order may significantly vary
between LLMs, something which can complicate transfer learning – e.g., training on one LLM and
applying on another with a different vocabulary size; (3) Limited Access: As already mentioned, in
certain LLMs, such as those released by OpenAI, the output tensor X is only partially accessible,
with APIs only exposing a small number of the top (log-)probs. To tackle these challenges, we
propose selecting, for each row of X, a fixed number of elements. Specifically, we preprocess X by
sorting each row independently and selecting the top K probabilities, as follows:
X′ = row-sort(X):,:K,
(1)
resulting in X′ ∈RN×K. This approach not only reduces computational complexity but also provides
a standardized representation independent of the vocabulary size (for an appropriate choice of K).
4

## Page 5

Later, in Section 5, we will show how even small values of K can capture most of the TDS probability
mass and enable strong empirical performance.
Encoding the ATP. The tensor X′ provides a comprehensive description of the LLM’s output,
but does not explicitly encode an important source of information: the probability p of the actual
tokens appearing in the sequence, i.e, the ATP. While these values are technically present in the
TDS (since it contains the full distribution), they are not directly distinguishable from the other
token probabilities in the vocabulary. Thus, we do also include ATPs as separate inputs to our
architecture and further complement these probabilities with additional information which allows us
to contextualize them with respect to the whole TDS. Specifically, we argue that valuable information
is encoded in the rank 2 (position) of the ATP within the sorted TDS. This information reveals the
“gap” between the actual token and the token the model would most likely expect to find instead.
We encode the rank in a way to make this feature more amenable for learning: we apply scaling to
a closed interval and transform it with specific parameters, obtaining RE(X, p). More details are
found in Appendix B.4.
Architecture. Given the preprocessed TDS X′ and the rank encoding RE(X, p), we first linearly
project X′ via W ∈RK×K′, concatenate it with RE(X, p), and then feed it to an encoder-only
transformer module T with learnable positional encodings, operating in the sequence dimension [50]:
hθ(X, p) = T

X′W
 RE(X, p)

.
(2)
Here, θ includes all model’s parameters, ∥denotes concatenation on the feature dimension. Finally,
we pool over the [CLS] token and obtain output scores via a linear layer. The resulting model,
LOS-Net, is trained with binary cross-entropy loss.
4
Generalization of Previous Approaches
As already mentioned, prior research has introduced various gray-box, methods for HD and DCD
based on LLM’s output probabilities [16, 24, 49, 20]. In what follows, we propose a general framework
to unify these diverse techniques, and show that this can be captured by our LOS-Net method,
shedding light on its flexibility.
Motivating example: Min-K% Shi et al. [43]. Min-K%, a prominent, recent method for DCD,
makes predictions on an input text ⃗s based on a score R calculated as the average of the smallest
K% log-probs: R(⃗s) =
1
|M|
P
i∈M log(pi), with M = {i | pi < perc(p, K)} being the set of token
indices whose probabilities are in the first K-th percentile of p. We note that it is instructive to
rewrite the scoring equation as:
R(⃗s) =
|⃗s|
X
i=1
token-wise score
z
}|
{
log(pi)
 K
100 · |⃗s|

· I
  confidence
z}|{
pi
<
adaptive threshold
z
}|
{
perc(p, K)

|
{z
}
gating
.
(3)
This highlights a general pattern: that of computing a global score by aggregating token-wise
values meeting a (dynamic) “acceptance” condition, a form of “gating”. To unify the aforementioned
baselines under a common framework, we formalize this pattern via a family of functions (see next).
Gated Scoring Functions (GSFs). We define the family of Gated Scoring Functions (GSF) as the
set of functions scoring LOSs by aggregating token-wise scores across the input sequence whenever
2The rank of the i-th token is defined as: ri(X, p) = PV
v=1 I(Xi,v > pi), where I(·) is the indicator function.
5

## Page 6

their confidence values exceed a (possibly adaptive) threshold. GSFs are described in terms of the
following components: (1) A confidence function κ : RN×k × RN →RN that assigns confidence
values to each token in the sequence; (2) A threshold function T : RN×k × RN →R that determines
an acceptance criterion; and (3) A weight function g : RN×k × RN →RN that assigns importance
scores to tokens. Given a LOS (X, p), a GSF computes a global score R(X, p) as follows:
F(X, p)i =
(
g(X′, p)i,
if κ(X′, p)i ≥T(X′, p),
0,
otherwise,
,
R(X, p) =
N
X
i=1
F(X, p)i,
(4)
where X′ is the sorted version of X, as per Equation (1). The family of GSF is flexible enough to
capture previously proposed gray-box methods, as we show in the following:
Proposition 1 (GSFs capture known baselines). Let B be the set of scoring functions implemented
by the Min/Max/Mean aggregated probability methods [16, 24, 49, 20] for HD, as well as Loss [55],
the MinK% [43] and MinK%++ [58] methods for DCD. For any scoring function f ∈B, there exists
a choice of functions κ, T, g such that the GSF R in Equation (4), implements f.
It is easy to see, e.g., how MinK% is implemented as a GSF3. Refer to Section A for more details on
how other baselines are implemented.
LOS-Net can approximate GSFs and implement known baselines. As the following result
shows, our LOS-Net can, in fact, approximate virtually all GSFs of interest; intuitively, there exist
sets of parameters such that it evaluates “arbitrarily close” to the target GSFs.
Proposition 2 (LOS-Net can approximate Equation (4)). Assume maximal possible vocabulary
size Vmax and context size Nmax. Let X × M ⊆RNmax×Vmax × RNmax represent a compact subset in
the LOS. For any measurable κ : X × M →RNmax, measurable T : X × M →R, measurable and
integrable weight function g : X × M →RNmax, and for any ϵ > 0, there exists a set of parameters θ
such that our model hθ : X × M →R satisfies ∥hθ −R∥L1 < ϵ where ∥·∥L1 denotes the L1 norm.
To prove this result, we build on existing universality results on approximating continuous
functions with Transformers [57], showing that our (generally non-continuous) target functions can
be approximated by continuous functions. Importantly, Proposition 2 implies that, as long as the
LOS space of interest lies within a compact domain4, our model can approximate the general GSF in
Equation (4) of LOSs for any LLM under mild conditions on κ, T, and g. Note that Proposition 2
cannot be generally extended to L∞due to the discontinuity of GSFs. The practical relevance
of Proposition 2, is underscored by the following: [Approximation of Baselines by LOS-Net] Our
architecture, as defined in Equation (2), can arbitrarily well approximate, in the L1 sense, any
of the baseline methods in B when operating on context and token-vocabulary of, resp., maximal
sizes Nmax and Vmax.
The above states that well-established, successful baselines (see class B in
Proposition 1) can be approximated by LOS-Net. All proofs are enclosed in Section A.
3For a sequence length of N, it suffices to choose: T(X′, p) = −perc(p, K) = −
 sort(p)⌈K
100 ·N⌉

,
κ(X′, p) =
−p,
g(X′, p) =
log p
⌈K
100 ·N⌉.
4This is inherently satisfied when using probabilities; or via clamping in the case of logits or log-probs.
6

## Page 7

Method
HotpotQA
IMDB
Movies
HotpotQA
IMDB
Movies
Mistral-7b-instruct
Llama3-8b-instruct
Logits-mean
61.00 ± 0.20
57.00 ± 0.60
63.00 ± 0.50
65.00 ± 0.20
59.00 ± 1.70
75.00 ± 0.50
Logits-min
61.00 ± 0.30
52.00 ± 0.70
66.00 ± 0.80
67.00 ± 0.80
55.00 ± 1.60
71.00 ± 0.50
Logits-max
53.00 ± 0.80
47.00 ± 0.40
54.00 ± 0.40
59.00 ± 0.50
51.00 ± 0.90
67.00 ± 0.30
Probas-mean
63.00 ± 0.30
54.00 ± 0.80
61.00 ± 0.20
61.00 ± 0.20
73.00 ± 1.50
73.00 ± 0.60
Probas-min
58.00 ± 0.30
51.00 ± 1.00
60.00 ± 0.80
60.00 ± 0.40
57.00 ± 1.60
65.00 ± 0.40
Probas-max
50.00 ± 0.50
48.00 ± 0.40
51.00 ± 0.50
56.00 ± 0.50
49.00 ± 0.80
64.00 ± 0.60
P(True)
54.00 ± 0.60
62.00 ± 0.90
62.00 ± 0.50
55.00 ± 0.50
60.00 ± 0.60
66.00 ± 0.40
Semantic Entropy
67.66 ± 0.55
62.44 ± 0.81
70.24 ± 0.68
65.58 ± 0.53
74.96 ± 1.00
72.27 ± 0.65
ATP+R-MLP
68.92 ± 0.24
90.70 ± 0.50
66.04 ± 0.13
64.50 ± 0.75
88.68 ± 0.30
73.25 ± 0.15
ATP+R-Transf.
69.70 ± 0.39
89.64 ± 1.08
67.92 ± 0.98
66.72 ± 0.39
85.46 ± 1.14
75.89 ± 1.07
LOS-Net
72.92 ± 0.45
94.73 ± 0.58
72.20 ± 0.66
72.60 ± 0.34
90.57 ± 0.28
77.43 ± 0.66
Act. Probe (incomp.†)
73.00 ± 0.60
92.00 ± 1.00
72.00 ± 0.50
77.00 ± 0.50
81.00 ± 1.40
78.00 ± 0.40
Table 1: Test AUCs for HD over Mis-7b and L-3-8b (bold:
best method, underlined:
second best).
orange
indi-
cates baselines requiring additional prompting/generations.
† Activation Probes , included as reference, are incomparable
as they access model internals.
IMDB
Movies
HotpotQA
Target
IMDB
Movies
HotpotQA
Source
94.73 *
 ± 0.58
62.65 *
 ± 0.27
69.24 *
 ± 1.56
92.78 *
 ± 0.73
72.20 *
 ± 0.66
69.53
 ± 0.12
92.92 *
 ± 0.44
67.74 *
 ± 0.70
72.92 *
 ± 0.45
Mis-7b
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
L-3-8b
Mis-7b
Q-2.5-7b
Target
L-3-8b
Mis-7b
Q-2.5-7b
Source
90.57 *
 ± 0.28
89.54 *
 ± 2.85
86.84 *
 ± 0.14
82.24 *
 ± 2.39
94.73 *
 ± 0.58
87.22 *
 ± 0.43
87.88 *
 ± 0.72
89.58
 ± 0.89
88.18 *
 ± 0.88
IMDB
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
Figure 2: Transfer Test AUC to vary-
ing datasets (top, Mis-7b) and LLMs
(bottom, IMDB fixed).
5
Experiments
We assess various aspects of learning with LOS via the following questions: (1) Is learning on LOS
an effective approach for DCD and HD? Does it outperform baselines? And how important is X, i.e.,
the TDS, in the pair (X, p), often overlooked? (Sections 5.1 and 5.2); (2) Does our model exhibit
transfer capabilities across LLMs and datasets? (Section 5.3); (3) What is the practical runtime of
our approach and how robust is it w.r.t. K (Sections 5.4 and 5.5)?
In the following, we present our main results, and refer to Section B for additional results and
details.
General setup. Our experiments focus on the two tasks of DCD and HD; in all results presented
next, hyperparameter K is always set to 1, 000, while its impact is discussed in Section 5.5. Aligning
with prior work, we use datasets and LLMs from [43, 58] for DCD and [37] for HD, where we also
experiment with an additional LLM (Qwen-2.5-7b-Instruct [53]). Further details are in subsequent
sections. As performance metric, we use the area under the ROC curve (AUC), a standard metric in
this domain [37, 43, 58]. We run each experiment with three different random seeds (when applicable)
and report the mean along with the standard deviation of the results. All LOS-Net experiments were
conducted using PyTorch [38] on a single NVIDIA L-40 GPU.
Newly introduced learning-based baselines. In addition to task-specific baselines, we also
introduce two novel learning-based baselines to appreciate the contribution of the TDS: ATP+R-MLP,
ATP+R-Transf.. Specifically, we ablate information about the TDS and only process the ATP
and rank information with, resp., an MLP or Transformer backbone. Formal definitions are in
Section B.4.
5.1
Hallucination Detection
Datasets and LLMs. We adopt datasets from Orgad et al. [37], following the same setup: the
objective is to predict whether an LLM-generated response to a given input prompt is correct or
not. We choose three datasets spanning various domains and tasks: HotpotQA without context [54],
IMDB sentiment analysis [32], roles in Movies [37]. Details regarding the annotation process, splits
7

## Page 8

and dataset sizes are in Section B.5.1. As the target LLMs, coherently with Orgad et al. [37], we use
Mistral-7b-instruct-v0.2 [23] (Mis-7b) and LlaMa3-8b-instruct [47] (L-3-8b), and further experiment
with Qwen-2.5-7b-Instruct [53] (Q-2.5-7b).
HD Baselines.
1. Aggregated probabilities/logits [16, 24, 49, 20]. They simply operate mean/max/min pooling
over the ATP to score LLM confidence for error detection. We refer to them as Logit/Probas-
mean/min/max.
2. P(True) – Kadavath et al. [24] found that LLMs show reasonable calibration in assessing
their own correctness via additional querying.
3. Semantic Entropy [13, 25]: a popular technique resorting to additional generations and an
auxiliary entailment model to assess the uncertainty of LLM’s responses at a semantic level,
argued, in turn, to be predictive for correctness. Note that both this method and the above
P(True) require additional prompting and/or generations, making their detection latency
orders of magnitudes higher than other methods in comparison, refer to Section 5.4.
4. Activation Probes [37, 3, 4] are linear classifiers fitted over the LLM’s internal activations.
They operate in the more restrictive white-box setup, thus not directly comparable to LOS-Net.
Still, they constitute relevant performance references. We probe the last generated token at
the layer maximizing validation performance.
Results.
Table 1 presents a comprehensive summary of results on Mis-7b and L-3-8b. These
clearly demonstrate that LOS-Net outperforms all gray-box baselines across all six dataset/LLM
combinations, often by a significant margin. These also include P(True) and Semantic Entropy, which
use auxiliary prompts or generations. We highlight how, on the IMDB dataset, LOS-Net achieves an
AUC improvement of around 31 units over the best of these baselines for Mis-7b and 17 over the
best baseline for L-3-8b. Intriguingly, we note how LOS-Net outperforms even white-box Activation
Probes in 2 out of 6 combinations, while performing similarly in 3 of them. Our results also indicate
that ATP learning-based baselines consistently underperform compared to LOS-Net, underscoring
the critical role of the TDS, X. Our ATP-based learnable baselines still outperform non-learnable
probability-based methods in most cases, suggesting that a learning approach relying exclusively on
ATP can still be a viable solution in certain scenarios. Results on Q-2.5-7b are consistent with the
above findings, and are deferred to Section B.8.
5.2
Data Contamination Detection
DCD is often framed as a Membership Inference Attack (MIA) [44, 33, 43]. A DC dataset D =
{qi, yi}ℓ
i=1 contains ℓtext samples, where qi represents the text and yi, the target, indicates whether
it was part of the training data or not.
Datasets and LLMs.
We use three datasets to assess DCD, specifically: WikiMIA-32 and
WikiMIA-64 [43] (excerpts from Wikipedia articles), as well as BookMIA [43] (excerpts from books).
Henceforth, due to space limitations, we will only discuss details and results related to the latter,
while referring readers to Section B.9 for the former. In BookMIA, positive members correspond to
books known to be well memorized by certain OpenAI models [12], or otherwise known to (partly)
be in the pretraining corpus of other open-source LLMs [1]. Non-members include excerpts from
books released after 2023, necessarily absent from the pretraining corpus of the last ones. This
dataset allows us to test LOS-Net in a realistic scenario akin to copyright-infringement detection.
8

## Page 9

Method / LLM
P-6.9b
P-12b
L-13b
L-30b
Loss
67.40
76.27
76.23
89.18
MinK
68.78
77.32
75.36
89.61
MinK++
66.73
71.76
72.87
80.60
Zlib
50.01
60.84
61.94
80.83
Lowercase
74.97
81.64
67.80
82.18
Ref
89.52
91.93
84.58
94.93
ATP+R-MLP
56.31 ± 1.48
57.18 ± 1.06
66.60 ± 1.05
83.89 ± 0.41
ATP+R-Transf.
79.59 ± 0.61
74.77 ± 0.57
74.65 ± 0.79
87.62 ± 0.68
LOS-Net
90.71 ± 0.90
89.43 ± 0.59
91.02 ± 0.15
95.60 ± 0.41
Table 2: Test AUCs on BookMIA. ‘P’: Pythia, ‘L’:
LlaMa-1 (bold: best, underlined: second best, pink :
reference-based).
P-6.9b
P-12b
L-13b
L-30b
Test
P-6.9b
P-12b
L-13b
L-30b
Train
90.71 *
± 0.9
87.36
± 1.4
78.70
± 1.4
77.28
± 1.9
89.86 *
± 0.21
89.43
± 0.59
83.36
± 1.4
85.06
± 3.5
78.31
± 2.2
78.26
± 1.4
91.03 *
± 0.15
95.60 *
± 0.33
86.27
± 0.63
83.61
± 1.5
90.48 *
± 0.56
95.60 *
± 0.41
77.5
80.0
82.5
85.0
87.5
90.0
92.5
95.0
Figure 3: BookMIA zero-shot AUC –
bold, ∗: outperforms, resp., ref-free and
ref-based.
In particular, contrary to previous works, we propose a novel split that ensures all excerpts from
the same book always appear either in the training or test split (and never in both). Details are
enclosed in Section B.5.2. We attack LLMs considered in [1]: LlaMa-13b/30b [47] (L-13b/30b),
Pythia-6.9b/12b[5] (P-6.9b/12b).
DCD Baselines. The Loss approach [55] directly uses the loss value as the detection score. The
Reference ( Ref ) method [10] calibrates the target LLM’s perplexity leveraging a similar reference
model known or supposed not to have memorized text of interest5. Both Zlib and Lowercase [10]
are also reference-based methods: they utilize zlib compression entropy and lowercased text perplexity
as reference for normalization. Lastly, Min-K% [43] and Min-K%++ [58] are reference-free methods,
which examine token probabilities and average a subset of the minimum token scores, or a function
thereof, over the input.
For these baselines, we select their hyperparameters by maximizing
performance on the validation set(s).
Results. Refer to Table 2 for results on BookMIA. LOS-Net attains exceptional results, largely
surpassing other reference-free approaches. Among these last ones, ours is the only method that can
match or outperform even the reference-LLM-based baselines. Importantly, (even partial) access to
the TDS reveals crucial to obtain such strong reference-free performance: our ATP-based learnable
methods – which only process features for the actual sequence tokens – incur indeed significant
performance drops. As for WikiMIA, while full results are enclosed in Section B.9 (Table 6), we
point out how LOS-Net consistently surpasses all baselines across all combinations of LLMs and
datasets. We also report the second-best method is MinK%++, followed by MinK%, consistent with
the findings in [58].
5.3
Generalization to Other LLMs and Datasets
Zero-shot Cross-LLM Generalization in DCD. We assess our model’s ability to detect DC
in target LLMs that were unseen during training. Using the BookMIA benchmark and the setup
described in Section 5.2, we evaluate our model directly across different LLMs without any fine-tuning.
This setup is particularly relevant in DCD scenarios where contamination information is not yet
available for newly released LLMs. The results are presented in the heatmap shown in Figure 3.
We observe strong transferability: in 10/12 cases, our model achieves the best performance among
reference-free approaches, highlighted in bold in Figure 3. Interestingly, in 3/12 cases, LOS-Net
5For example for Pythia-12b, a valid reference LLM would be the smaller Pythia-70M.
9

## Page 10

(which is reference-free) even surpasses reference-based baselines, as indicated via a superscript of ∗.
We also observe particularly strong transfer across differently sized LLM architectures within the
same family and highlight the surprising positive transfer from the largest LlaMa to Pythia models.
Transfer Learning across LLMs and Datasets for HD. Although LOS-Net delivered non-trivial
generalization, its zero-shot application on HD was not sufficient to surpass the simpler probability-
based techniques. This led us to investigate LOS-Net capabilities in a transfer learning setting.
Specifically, we fix an LLM/dataset combination and fine-tune the corresponding pretrained LOS-Net
either on the remaining LLMs for the same dataset, or the remaining datasets for the same LLM.
All Test AUCs of our fine-tuned LOS-Net’s are in Figures 4 and 5, Section B.6, while we report here
two representative plots (see Figure 2). On these heatmaps, superscript ‘∗’ indicates the fine-tuned
LOS-Net is better than a counterpart trained from scratch in the same setting – testing for successful
transfer; bold indicates it outperforms the best non-learnable probability-based method.
Discussion. First, LOS-Net exhibits solid transferability in both scenarios. The finetuned models
consistently outperform their counterparts trained from scratch: 16/18 cases in both the cross-LLM
(Figure 4, Section B.6) and cross-dataset setups (Figure 5, Section B.6) – see ‘∗’ on the off-diagonal
entries. This highlights a generally positive transfer of LOS-Net’s learned representations across
datasets and LLMs, and underscores the suitability of LOS as a data type in capturing generalizable
patterns for HD. Second, from a practical perspective, we find that LOS-Net outperforms the best
probability-based baseline in 15/18 cases for both the cross-LLM (Figure 4) and cross-dataset
(Figure 5) scenarios – see bold on the off-diagonal entries. Focusing on the IMDB dataset, when
training on L-3-8b and testing on Mis-7b (Figure 2 (bottom)), our model substantially gains around
27 AUC units over the best probability-based baseline. This result underscores the possibility of
transferring across LLMs. A similar trend is observed in the cross-dataset setup (Figure 2 (top)): on
Mis-7b, when training on HotpotQA or Movies and testing on IMDB, our model achieves a notable
improvement of around 30 AUC units compared to the best baseline).
5.4
Run-Time Analysis
To empirically assess the efficiency of our approach, we ran a comprehensive set of training and
inference timings, reported in full in Section B.10 and discussed in the following. The results
clearly show LOS-Net features an extremely contained detection latency: ∼10−5s per inference
fwd-pass. Training is also efficient, typically completing in under one hour on a single NVIDIA
L-40 GPU, and often taking significantly shorter. To contextualize this computational efficiency
w.r.t. methods relying on multiple prompting/generations [37, 25], we measured the detection
latency of Semantic Entropy (SE) [13, 25], a pioneering method for HD. SE involves generating 10
additional responses and their semantic clustering, operated by checking mutual entailment with
an auxiliary language model. On average, we measured 7.14 ± 2.97 seconds per sample detection
on Movies/L-3-8b and 7.55 ± 1.70 seconds on Movies/Mis-7b. This is five orders of magnitude
slower than LOS-Net, making the latter preferable for both accuracy and latency. Results on other
LLM/dataset combinations are reported in Section B.10.
5.5
The value of K and restricted TDS access
We conclude this section by presenting results on the impact of the parameter K, as defined in Equa-
tion (1). In particular, we slide K in {10, 50, 100, 500, 1000} and discuss here results for the Mis-7B
LLM on the HotpotQA dataset, see Table 3. For each of the above values, we measure two quantities:
the average probability mass captured, i.e.,
PN
n=1(
PV
v=1(X′
n,v))/N, and the corresponding performance of
LOS-Net.
10

## Page 11

Table 3:
Ablation study on K in
LOS-Net.
Average Probability Mass
(APM) and Test AUC for varying K on
Mis-7B - HotpotQA.
K
APM (%)
Test AUC
10
99.49
71.82 ± 0.15
50
99.80
71.87 ± 0.24
100
99.85
72.34 ± 0.20
500
99.99
72.67 ± 0.32
1000
99.99
72.92 ± 0.45
We observe that the former exceeds 0.99 for all considered
K’s, indicating that even values as small as K = 10 are
often sufficient to convey most of the information in the full
TDS. In terms of performance, Test AUC tends to improve
as K values increase, though with diminishing returns
beyond K ≥100. As expected, the value K = 1000 tends
to deliver the best performance overall, this confirmed by
results on the other LLM-dataset combinations reported in
Section B.7. Given its extremely contained run-times (see
above), this value appears to hit a sweet-spot optimizing
performance and complexity. Most notably, however, even
with K = 10, LOS-Net outperforms all baselines and matches the white-box Activation Probe,
highlighting the practical effectiveness of LOS-Net even in API-limited settings such as in GPT
models – at the time of writing, exposing only the top K = 20 output logits.
6
Conclusions
We proposed LOS-Net, an efficient method to detect data contamination and hallucinations in LLMs
by leveraging their output signatures (LOS), defined as the union of Token Distribution Sequences
(TDS) and Actual Token Probabilities (ATP). LOS-Net consists of a lightweight attention-based
model operating on an effective encoding of the LOS. We proved it unifies and extends existing gray-
box methods under a general framework, and experimentally showed it outperforms state-of-the-art
gray-box methods across datasets and LLMs. It also exhibited promising generalization and transfer
capabilities of LOS-Net, both across datasets and across LLMs. Our framework could be applied to
other tasks, such as detecting LLM-generated content. Additional sources of information can also be
incorporated, e.g., in the absence of latency constraints, it can be interesting to include “exact-token”
flags as proposed by [37]. Last, the LOS can be extended to account for multiple prompting [25].
11

## Page 12

Acknowledgements
The authors are grateful to Beatrice Bevilacqua for insightful discussions. G.B. is supported by
the Jacobs Qualcomm PhD Fellowship. F.F. conducted this work supported by an Aly Kaufman
and an Andrew and Erna Finci Viterbi Post-Doctoral Fellowship. Y.G. is supported by the UKRI
Engineering and Physical Sciences Research Council (EPSRC) CDT in Autonomous and Intelligent
Machines and Systems (grant reference EP/S024050/1). H.M. is a Robert J. Shillman Fellow and is
supported by the Israel Science Foundation through a personal grant (ISF 264/23) and an equipment
grant (ISF 532/23). D.L. is funded by an NSF Graduate Fellowship. Research was also supported by
the Israeli Ministry of Science, Israel-Singapore binational grant 207606. F.F. is extremely grateful
to the members of the “Eva Project”, whose support he immensely appreciates.
References
[1] Sagiv Antebi, Edan Habler, Asaf Shabtai, and Yuval Elovici. Tag&tab: Pretraining data
detection in large language models using keyword-based membership inference attack. arXiv
preprint arXiv:2501.08454, 2025.
[2] Yuval Atzmon and Gal Chechik. Adaptive confidence smoothing for generalized zero-shot
learning.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 11671–11680, 2019.
[3] Amos Azaria and Tom Mitchell. The internal state of an llm knows when it’s lying, 2023. URL
https://arxiv.org/abs/2304.13734.
[4] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational
Linguistics, 48(1):207–219, 2022.
[5] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien,
Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward
Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In
International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.
[6] Lukas Biewald. Experiment tracking with weights and biases, 2020. URL https://www.wandb.
com/. Software available from wandb.com.
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler,
Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray,
Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,
and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Had-
sell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, vol-
ume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.
cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
[8] Meng Cao, Yue Dong, Jingyi He, and Jackie Chi Kit Cheung. Learning with rejection for
abstractive text summarization. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors,
Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages
9768–9780, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational
12

## Page 13

Linguistics. doi: 10.18653/v1/2022.emnlp-main.663. URL https://aclanthology.org/2022.
emnlp-main.663/.
[9] Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer:
Evaluating and testing unintended memorization in neural networks. In 28th USENIX security
symposium (USENIX security 19), pages 267–284, 2019.
[10] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine
Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data
from large language models. In 30th USENIX Security Symposium (USENIX Security 21),
pages 2633–2650, 2021.
[11] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer.
Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and
Privacy (SP), pages 1897–1914. IEEE, 2022.
[12] Kent Chang, Mackenzie Cramer, Sandeep Soni, and David Bamman. Speak, memory: An
archaeology of books known to ChatGPT/GPT-4. In Houda Bouamor, Juan Pino, and Kalika
Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Pro-
cessing, pages 7312–7327, Singapore, December 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.emnlp-main.453. URL https://aclanthology.org/2023.emnlp-main.
453/.
[13] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in
large language models using semantic entropy. Nature, 630(8017):625–630, 2024.
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997, 2023.
[15] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces.
arXiv preprint arXiv:2312.00752, 2023.
[16] Nuno M Guerreiro, Elena Voita, and André FT Martins. Looking for a needle in a haystack:
A comprehensive study of hallucinations in neural machine translation.
arXiv preprint
arXiv:2208.05309, 2022.
[17] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. arXiv
preprint arXiv:1909.03368, 2019.
[18] John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word
representations. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings
of the 2019 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4129–
4138, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi:
10.18653/v1/N19-1419. URL https://aclanthology.org/N19-1419/.
[19] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems, 2023.
13

## Page 14

[20] Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu,
and Lei Ma. Look before you leap: An exploratory study of uncertainty measurement for large
language models. arXiv preprint arXiv:2307.10236, 2023.
[21] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models. Journal of Machine Learning Research, 24(251):
1–43, 2023.
[22] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM Computing Surveys, 55(12):1–38, 2023.
[23] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
[24] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez,
Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language
models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.
[25] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances
for uncertainty estimation in natural language generation, 2023. URL https://arxiv.org/
abs/2302.09664.
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing
Systems, 33:9459–9474, 2020.
[27] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-
time intervention: Eliciting truthful answers from a language model. Advances in Neural
Information Processing Systems, 36, 2024.
[28] Yucheng Li, Yunhao Guo, Frank Guerin, and Chenghua Lin. An open-source data contamination
report for large language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen,
editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 528–541,
Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/
v1/2024.findings-emnlp.30. URL https://aclanthology.org/2024.findings-emnlp.30/.
[29] Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke
Zettlemoyer, and Madian Khabsa. Xlm-v: Overcoming the vocabulary bottleneck in multilingual
masked language models. arXiv preprint arXiv:2301.10472, 2023.
[30] Tianyu Liu, Yizhe Zhang, Chris Brockett, Yi Mao, Zhifang Sui, Weizhu Chen, and Bill Dolan.
A token-level reference-free hallucination detection benchmark for free-form text generation.
arXiv preprint arXiv:2104.08704, 2021.
[31] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
[32] Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher
Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting
14

## Page 15

of the association for computational linguistics: Human language technologies, pages 142–150,
2011.
[33] Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya
Sachan, and Taylor Berg-Kirkpatrick. Membership inference attacks against language models
via neighbourhood comparison. arXiv preprint arXiv:2305.18462, 2023.
[34] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and
factuality in abstractive summarization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel
Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational
Linguistics, pages 1906–1919, Online, July 2020. Association for Computational Linguistics. doi:
10.18653/v1/2020.acl-main.173. URL https://aclanthology.org/2020.acl-main.173/.
[35] Edoardo Mosca, Shreyash Agarwal, Javier Rando Ramírez, and Georg Groh.
“that is a
suspicious reaction!”: Interpreting logits variation to detect NLP adversarial attacks.
In
Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 7806–7816, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.acl-long.538. URL https://aclanthology.org/2022.acl-long.538/.
[36] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red
Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad
Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher
Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman,
Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann,
Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis,
Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey
Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux,
Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila
Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix,
Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson,
Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan
Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy,
Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey,
Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost
Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin,
Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali,
Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook
Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel
Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen
Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel
Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez,
Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv
Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKin-
ney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob
Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa,
Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro
Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang,
15

## Page 16

Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista
Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam
Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto,
Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power,
Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond,
Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario
Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr,
John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker,
Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama,
Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie
Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin
Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón
Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang,
Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter
Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel
Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao
Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang,
Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret
Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.
[37] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and
Yonatan Belinkov. Llms know more than they show: On the intrinsic representation of llm
hallucinations. arXiv preprint arXiv:2410.02707, 2024.
[38] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
style, high-performance deep learning library. Advances in neural information processing systems,
32, 2019.
[39] Yifu Qiu, Yftah Ziser, Anna Korhonen, Edoardo Ponti, and Shay Cohen. Detecting and
mitigating hallucinations in multilingual summarisation. In Houda Bouamor, Juan Pino, and
Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Lan-
guage Processing, pages 8914–8932, Singapore, December 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.emnlp-main.551. URL https://aclanthology.org/2023.
emnlp-main.551/.
[40] Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo M Ponti, and Shay B Cohen. Spec-
tral editing of activations for large language model alignment. arXiv preprint arXiv:2405.09719,
2024.
[41] Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, and Skyler Speakman. Weakly
supervised detection of hallucinations in llm activations. arXiv preprint arXiv:2312.02798, 2023.
[42] Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, SM Tonmoy, Aman
Chadha, Amit P Sheth, and Amitava Das. The troubling emergence of hallucination in large
language models–an extensive definition, quantification, and prescriptive remediations. arXiv
preprint arXiv:2310.04988, 2023.
[43] Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi
Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. arXiv
preprint arXiv:2310.16789, 2023.
16

## Page 17

[44] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference
attacks against machine learning models. In 2017 IEEE symposium on security and privacy
(SP), pages 3–18. IEEE, 2017.
[45] Ben Snyder, Marius Moisescu, and Muhammad Bilal Zafar. On early detection of hallucinations
in factual question answering.
In Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, pages 2721–2732, 2024.
[46] SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha, and Amitava
Das. A comprehensive survey of hallucination mitigation techniques in large language models.
arXiv preprint arXiv:2401.01313, 2024.
[47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timo-
thée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open
and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
[48] Simon Valentin, Jinmiao Fu, Gianluca Detommaso, Shaoyuan Xu, Giovanni Zappella, and
Bryan Wang. Cost-effective hallucination detection for llms. arXiv preprint arXiv:2407.21424,
2024.
[49] Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, and Dong Yu. A stitch in
time saves nine: Detecting and mitigating hallucinations of llms by validating low-confidence
generation. arXiv preprint arXiv:2307.03987, 2023.
[50] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems,
2017.
[51] Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text
ghostwritten by large language models. In Kevin Duh, Helena Gomez, and Steven Bethard,
editors, Proceedings of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages
1702–1717, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi:
10.18653/v1/2024.naacl-long.95. URL https://aclanthology.org/2024.naacl-long.95/.
[52] Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Llmdet: A third
party large language models generated text detection tool. arXiv preprint arXiv:2305.15004,
2023.
[53] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong
Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou,
Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li,
Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie
Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong
Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan,
Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
[54] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. arXiv preprint arXiv:1809.09600, 2018.
17

## Page 18

[55] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine
learning: Analyzing the connection to overfitting.
In 2018 IEEE 31st computer security
foundations symposium (CSF), pages 268–282. IEEE, 2018.
[56] Fan Yin, Jayanth Srinivasa, and Kai-Wei Chang. Characterizing truthfulness in large language
model generations with local intrinsic dimension. arXiv preprint arXiv:2402.18048, 2024.
[57] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar.
Are transformers universal approximators of sequence-to-sequence functions? arXiv preprint
arXiv:1912.10077, 2019.
[58] Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank
Yang, and Hai Li. Min-k%++: Improved baseline for detecting pre-training data from large
language models. arXiv preprint arXiv:2404.02936, 2024.
[59] Zheng Zhao, Emilio Monti, Jens Lehmann, and Haytham Assem.
Enhancing contextual
understanding in large language models through contrastive decoding. In Kevin Duh, He-
lena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (Volume 1:
Long Papers), pages 4225–4237, Mexico City, Mexico, June 2024.
Association for Computational Linguistics.
doi: 10.18653/v1/2024.naacl-long.237.
URL
https://aclanthology.org/2024.naacl-long.237/.
18

## Page 19

A
Proofs
Proposition 2 (LOS-Net can approximate Equation (4)). Assume maximal possible vocabulary
size Vmax and context size Nmax. Let X × M ⊆RNmax×Vmax × RNmax represent a compact subset in
the LOS. For any measurable κ : X × M →RNmax, measurable T : X × M →R, measurable and
integrable weight function g : X × M →RNmax, and for any ϵ > 0, there exists a set of parameters θ
such that our model hθ : X × M →R satisfies ∥hθ −R∥L1 < ϵ where ∥·∥L1 denotes the L1 norm.
Proof. We define D := X × M. Recall that the target function we want to approximate is the gated
scoring function R as defined in Equation (4), which can be written as follows:
R(x) =
Nmax
X
i=1
I(κ(x)i ≥T(x)) · g(x)i,
(5)
for x ∈D.
Define f(1) : D →RNmax to be the components of the sum in Equation (5):
f(1)(x)i = I(κ(x)i ≥T(x)) · g(x)i.
(6)
It follows that R(x) = PNmax
i=1
f(1)(x)i.
Step 1: We begin by selecting K = Vmax as a hyperparameter6 and initializing the parameters
p1, p2, and W as follows:
p1 = 0,
(7)
p2 = 1,
(8)
W = IK×K.
(9)
As a result, the input to the transformer encoder in our architecture (see Equation (2)) becomes
X′||p ∈RNmax×(Vmax+1).
This simplifies our architecture in Equation (2) to:
hθ(X, p) = T (X′||p).
(10)
Step 2: f (1) ∈L1(D). Define the L1(D) norm for a field F : D →Rn2 as:
∥F∥L1 =
Z
x∈D
∥F(x)∥1 dx =
Z
x∈D
n2
X
i=1
|F(x)i| dx
=
n2
X
i=1
Z
x∈D
|F(x)i| dx =
n2
X
i=1
∥F(x)i∥L1 ,
(11)
where ∥v∥1 = Pn2
i=1 |vi| is the l1 norm of the vector v.
Next, observe that f(1) ∈L1(D). To see this, first note that f(1) is measurable. The indicator
function is measurable because the indicator set is the preimage of the measurable function κ(x)−T(x)
on the closed set [0, ∞). Thus, f (1), being a product of measurable functions, is measurable. Next,
we show that the L1 norm is finite. This is true because f(1) is a product of the integrable function
g and the bounded function 1 on the compact domain D.
Step 3: Approximating f(1) by a continuous field ˜f(1). We need to approximate the field
f(1) : D →RNmax by a continuous field, so that we can apply existing results on approximating
continuous functions with Transformers. We state the following Lemma, saying the continuous fields
are dense in L1(D).
6For LLMs with a vocabulary size smaller than Vmax, appropriate padding can be applied.
19

## Page 20

Lemma 1. For any g ∈L1(D) and any ϵ > 0, there exists a continuous ˜g ∈L1(D) such that
∥g −˜g∥L1 < ϵ.
Proof. Consider the coordinate functions gi : D →R. Since continuous functions are dense in L1
for scalar valued functions, we can choose continuous ˜gi such that ∥g −˜g∥L1 < ϵ/N. Thus, letting
˜g(x) = [g1(x), . . . , gN(x)] ∈RN, it holds that ∥g −˜g∥= PN
i=1 ∥gi −˜gi∥< ϵ.
Thus, we can choose a function ˜f(1) such that,
f(1) −˜f(1) <
ϵ
2Nmax
.
(12)
Step 4: Approximating the continuous field ˜f(1) by a transformer model h(1)
θ . We
start by restating the following from [57] in our context,
Theorem 1. Let 1 ≤p < ∞and ϵ > 0, then for any given f ∈FCD, where FCD is the set of
all continuous functions that map a compact domain in Rn×d to Rn×d, there exists a Transformer
network (with positional encodings) g : Rn×d →Rn×d such that we have ∥f −g∥Lp ≤ϵ.
To apply this theorem in our context, we observe that in our case d := Vmax + 1 and n := Nmax
for the input space, and the domain D ⊆RNmax×(Vmax+1) is compact. Thus ˜f(1) ∈FCD (note that
the output space dimension in our case is RNmax×1 instead of RNmax×d, but this can be handled using
zero-padding). Using p = 1, it holds that there exists a transformer h(1)
θ
s.t.,
h(1)
θ
−˜f(1) <
ϵ
2Nmax .
Step 5: Pooling. Our model concludes with a [CLS] token pooling mechanism, which is
equivalent in expressiveness to the standard sum pooling method. Thus, assuming that the final
layer of our model is given by h(1)
θ (x), our model can be written as follows,
hθ(x) =
Nmax
X
i=1

h(1)
θ (x)i

.
(13)
Step 6: Approximating the objective function. Intuitively, hθ(x) approximates R(x)
because h(1)
θ (x)i approximates f(1)(x)i.
We demonstrate this as follows.
∥hθ −R∥L1 =

Nmax
X
i=1

h(1)
θ;i

−
Nmax
X
i=1
f(1)
i

L1
(14)
≤
Nmax
X
i=1
h(1)
θ;i −f(1)
i

(15)
=
Nmax
X
i=1
h(1)
θ;i + ( ˜f(1)
i
−˜f(1)
i
) −f(1)
i

(16)
≤
Nmax
X
i=1
h(1)
θ;i −˜f(1)
i
 +
Nmax
X
i=1
 ˜f(1)
i
−f(1)
i

(17)
We applied the triangle inequality to obtain the two inequalities. Next, note that for a field F :
Rn1 →Rn2, the L1 norm of any coordinate function is less than the L1 norm of F: ∥Fj∥L1 ≤∥F∥L1
20

## Page 21

for any j ∈{1, . . . , n2}. This can be seen directly from the definition of the L1 norm of F. Combining
this with our choices of ˜f and hθ shows that:
N
X
i=1
h(1)
θ;i −˜f(1)
i
 +
Nmax
X
i=1
 ˜f(1)
i
−f(1)
i

(18)
<
Nmax
X
i=1
ϵ
2Nmax
+
Nmax
X
i=1
ϵ
2Nmax
(19)
= ϵ.
(20)
In total, this means that ∥hθ −R∥L1 < ϵ, so we are done.
Proposition 1 (GSFs capture known baselines). Let B be the set of scoring functions implemented
by the Min/Max/Mean aggregated probability methods [16, 24, 49, 20] for HD, as well as Loss [55],
the MinK% [43] and MinK%++ [58] methods for DCD. For any scoring function f ∈B, there exists
a choice of functions κ, T, g such that the GSF R in Equation (4), implements f.
Proof. We will prove the Proposition by defining, for each baseline, the functions implementing
components κ, T, g, assuming no ties in the ATP values p.
Mean Aggregated Probability. This baseline simply outputs the mean across the ATPs p.
The following selection of functions implements it as a GFS:
κ(X′, p) = 1
T(X′, p) = 0
g(X′, p) = 1
N p
Min Aggregated Probability outputs the min value across the ATPs p. The following
selection of functions implements it as a GFS:
κ(X′, p) = −p
T(X′, p) = −min(p)
g(X′, p) = p
Max Aggregated Probability outputs the max value across the ATPs p. We simply pick:
κ(X′, p) = p
T(X′, p) = max(p)
g(X′, p) = p
MinK%. Please refer to Section 4.
MinK%++. Let ¯p = log(p)−µ
σ
, be the normalized version of p, with:
µi = EX′
i[log(X′
i)] =
V
X
v=1
X′
i,v · log(X′
i,v),
σi =
q
EX′
i[(log(X′
i) −µi)2]
=
v
u
u
t
V
X
v=1
X′
i,v ·
 log(X′
i,v) −µi
2,
(21)
Where X′ is given from Equation (1).
The baseline is implemented by setting:
T(X′, p) = −perc(¯p, K) = −
 sort(¯p)⌈K
100 ·N⌉

,
κ(X′, p) = −¯p,
g(X′, p) =
¯p
 K
100 · N
.
21

## Page 22

Loss as a Privacy Proxy [55]. This method uses the model’s negated loss as a proxy for
contamination, which can be defined as the average of the log ATPs. The method can thus be
implemented with:
κ(X′, p) = 1,
T(X′, p) = 0,
g(X′, p) = 1
N log(p).
(22)
[Approximation of Baselines by LOS-Net] Our architecture, as defined in Equation (2), can
arbitrarily well approximate, in the L1 sense, any of the baseline methods in B when operating on
context and token-vocabulary of, resp., maximal sizes Nmax and Vmax.
Proof. To prove Section 4, it suffices to show the following. First (i), that the baselines can be
implemented as in Equation (4), given their sequence length and vocabulary size satisfy, N ≤
Nmax,
V ≤Vmax, where values in the inputs for indices larger than N, V are ‘padded’ with e.g.,
−1. Second (ii), that their implementations are realized with κ, T, and g which are all measurable,
and with g also integrable.
(i) Let us slightly modify the implementations provided in the Proof for Proposition 1 to correctly
account for padding values. Let us conveniently define:
α : R →R,
α(x) = 1 −ReLU(−x) =
(
1
x ≥0
1 + x
x < 0
Neff =
Nmax
X
i=1
α(pi)
Veff =
Vmax
X
v=1
α(X1,v)
(23)
as well as the following function, which will help us ‘manipulate’ the padding value in order not to
interfere with the effective computations required by baselines:
β : R →R,
β(x; M, f) =
(
f(x)
x ≥0
M
x = −1 , M > 0.
(24)
Mean Aggregated Probability.
κ(X′, p) = 1
T(X′, p) = 0
g(X′, p) =
1
Neff
p ◦α(p),
where ◦denotes the hadamard (element-wise) product.
Min Aggregated Probability.
κ(X′, p) = −β(p)
T(X′, p) = −min(β(p))
g(X′, p) = p
M = 2,
f ≡id.
Max Aggregated Probability.
κ(X′, p) = p
T(X′, p) = max(p)
g(X′, p) = p
22

## Page 23

MinK%.
κ(X′, p) = −β(p)
T(X′, p) = −
 sort(β(p))⌈K
100 ·Neff⌉

g(X′, p) = log(β(p))
 K
100 · Neff

M = 2,
f ≡id.
where the note the application of β inside the log prevents negative inputs.
MinK%++. Before illustrating how this baseline is implemented, we note the following. In
order for the normalization of log-probs to be well-defined, it is required that: (1) µ is finite, (2) the
denominator is greater than 0. As for (1), we note that null probability values (Xi,v = 0) would be
problematic, as they would cause the log function to output −∞. We assume, in this case, that all
probability values lie in [ϵ1, 1], with ϵ1 being a small value such that 0 < ϵ1 < 1. Regarding (2), we
see that the problematic situation would occur in cases where the probability distribution is uniform.
We assume to handle this case by adding a small positive constant ϵ2 > 0 in the denominator, so
that the normalization would take the form: ¯p = log(p)−µ
σ+ϵ2
.
Under these assumptions, we define the following β functions:
β1 = β(·; 2, id.)
βi
2 = β(·; −2 log(ϵ1)
ϵ2
, fi),
fi(x) =
log(x) −µi
 K
100 · Neff

σi + ϵ2
where we note that −2 log(ϵ1)
ϵ2
upper-bounds all the possible values that can be attained by fi’s under
our assumptions.
At this point, we observe that the values µi, σi can be correctly obtained as follows, in a way
that is not influenced by our padding scheme:
µi =
X
v
α(X′
i,v) · X′
i,v log
 β1(X′
iv)

(25)
σi =
sX
v
α(X′
i,v) · X′
i,v (log(β1(X′)i,v) −µi)2
(26)
At this point, let β2(p)i = βi
2(pi). We set:
κ(X′, p) = −β2(p)
T(X′, p) = −
 sort(β2(p))⌈K
100 ·Neff⌉

g(X′, p) =
β2(p)
 K
100 · Neff

and note that the K-th percentile in T is correctly computed despite the padding values due to the
specific choice of M in β2’s.
23

## Page 24

Loss as a Privacy Proxy [55].
κ(X′, p) = 1,
T(X′, p) = 0,
g(X′, p) =
1
Neff
log(p).
(27)
(ii) We now proceed to show that the implementations above are obtained via measurable
functions κ, T, and a measurable and integrable function g, which completes the proof.
Step 1: Consider a fixed sequence length N′ ∈[Nmax] and a fixed vocabulary size V ∈[Vmax].
When restricted to these parameters, all relevant functions are continuous. This follows from the
fact that each function, when restricted in this manner, is composed of continuous functions.
Step 2: The input domain for each combination of sequence length N′ ∈[Nmax] and vocabulary
size V ∈[Vmax] forms a compact set, and the union of all of this domains is also compact (as a finite
union of compact sets). Moreover, for any two distinct pairs (N1, V1) and (N2, V2), if either N1 ̸= N2
or V1 ̸= V2, then the corresponding domains are disjoint.
In most of our cases of interest, this follows from the fact that probabilities lie within [0, 1] and
that padding is represented by −1. In other cases, e.g., the application of β, the sets might be
different, but remain disjoint and compact.
Thus, by the following lemma, all functions κ, T, g for all baselines are continuous, completing
the proof.
Lemma 2. Let X be a subset of a metric space, which is compact, and can be expressed as a finite
disjoint union of compact subsets Xi indexed by a finite set I, i.e.,
X =
G
i∈I
Xi.
Suppose a function f : X →Rn is defined such that for each i ∈I, there is a continuous function
g(i) : Xi →Rn
satisfying f|Xi = g(i). Then, f is continuous on X.
The finite disjoint union of compact subsets correspond to all possible sequence lengths (N′ ∈
Nmax) and vocabulary sizes (V ′ ∈Vmax). Below we provide the proof for Lemma 2.
Proof. Consider any point x ∈X, and let (x(m)) be a sequence converging to x, in X. We need to
show that
f(x(m)) →f(x)
as m →∞.
Since X is a finite disjoint union of compact subsets Xi, there exists an index i∗such that x ∈Xi∗.
Since the subsets Xi are disjoint and compact, there exists a positive minimum separation
distance between distinct subsets, defined as,
δ∗= 1
2 min
i̸=j
inf
x∈Xi,y∈Xj ∥x −y∥.
Since each Xi is compact and the index set is finite7, this minimum distance is well-defined and
strictly positive.
Because x(m) →x, there exists an integer M such that for all m > M, we have
∥x(m) −x∥< δ∗.
7https://proofwiki.org/wiki/Distance_between_Disjoint_Compact_Set_and_Closed_Set_in_Metric_Space_
is_Positive#google_vignette
24

## Page 25

By the definition of δ∗, this ensures that for sufficiently large m, the sequence x(m) remains in Xi∗,
i.e., x(m) ∈Xi∗for all m > M.
Since f coincides with g(i∗) on Xi∗, we have
f(x(m)) = g(i∗)(x(m)),
for all m > M.
By assumption, g(i∗) is continuous on Xi∗, so
g(i∗)(x(m)) →g(i∗)(x)
as m →∞.
Since f(x) = g(i∗)(x), it follows that
f(x(m)) →f(x),
which proves that f is continuous at x. Since x was arbitrary, f is continuous on X.
B
Extended Experimental Section
B.1
Experimental Details
Our experiments were conducted using the PyTorch [38] framework (License: BSD), using a single
NVIDIA L-40 GPU for all experiments regarding LOS-Net. We use a fixed batch size of 64 for
all the tasks and datasets, and a fixed value of 8 heads (except for the Movies[37] dataset) in our
light-weight transformer encoder for LOS-Net. Hyperparameter tuning was performed utilizing the
Weight and Biases framework [6] – see Table 4.
B.2
HyperParameters
In this section, we detail the hyperparameter search conducted for our experiments. We use the
same hyperparameter grid for our main model, LOS-Net, and our proposed learning-based baselines,
namely, ATP+R-MLP, ATP+R-Transf.. Additionally, we note that for a given dataset, we
maintained the same grid search approach for all LLMs’ LOSs that we have trained on. The
hyperparameter search configurations for all datasets are presented in Table 4. The grid search
optimizes for the AUC calculated on the validation set.
Table 4: Hyperparameter search grid for LOS-Net.
Dataset
Num. layers
Learning rate
Embedding size
Epochs
Dropout
Weight Decay
HotpotQA
{1, 2}
{0.0001}
{128, 256}
{300}
{0, 0.3}
{0, 0.001}
IMDB
{1, 2}
{0.0001}
{128, 256}
{300}
{0, 0.3}
{0, 0.001}
Movies
{1, 2}
{0.0001}
{128, 256}
{300, 500}
{0.0, 0.3, 0.5}
{0, , 0.001, 0.005}
WikiMIA (32/64)
{1, 2}
{0.0001}
{128, 256}
{100, 500, 1000}
{0, 0.3}
{0, 0.001}
BookMIA
{1, 2}
{0.0001}
{64, 128}
{500}
{0, 0.3, 0.5}
{0, 0.001}
B.3
Optimizers and Schedulers
For all datasets we employ the AdamW optimizer [31] paired with a Linear scheduler, using a warm
up of 10% of the epochs. We apply an early stopping criterion if there is no improvement in validation
performance for 30 consecutive epochs.
25

## Page 26

B.4
Our Baselines and Rank Encoding
Rank Encoding. We construct the following learnable rank encoding 8,
RE(X, p) = p ⊙rscaled · w1 + p · w2,
(28)
where ⊙is the hadamard product, and w1, w2 are learnable parameters in Rd. As a result, RE(X, p)
is in RN×d. Importantly, the multiplication by p makes sure that the rank encoding and the TDS
are in similar scales, especially when using log probabilities or logits.
Our baselines. Below we present our additional learnable baselines. ATP+R-Transf is imple-
mented as described in Equation (2), but without incorporating the TDS (X), as follows:
hθ(X, p) = T (RE(X, p)) ,
(29)
where T represents an encoder-only transformer architecture [50]. ATP+R-MLP is similar to
ATP+R-Transf. but replaces the transformer with an MLP. Formally:
hθ(X, p) = MLP (RE(X, p)) ,
(30)
B.5
Dataset Description
B.5.1
Datasets for Hallucination Detection
In this section, we provide an overview of the three datasets used in our hallucination detection
analysis; we mostly follow the framework given in [37] in constructing the datasets. Our aim was to
ensure coverage of a wide variety of tasks, required reasoning skills, and dataset diversity. For each
dataset, we highlight its unique contributions and how it complements the others.
For all datasets, we used a consistent split of 10,000 training samples and 10,000 test samples.
1. HotpotQA [54] (License: CC-BY-SA-4.0): This dataset is specifically designed for multi-hop
question answering and includes diverse questions that require reasoning across multiple pieces
of information. Each entry comprises supporting Wikipedia documents that aid in answering
the questions. For our analysis, we utilized the “without context” setting, where questions are
posed directly. This setup demands both factual knowledge and reasoning skills to generate
accurate answers.
2. Movies [37] (License: MIT): This dataset checks for factual accuracy in scenarios regarding
movies. LLMs are asked, in particular, who was the actor/actress playing a specific role in a
movie of interest. This dataset contains 7857 test samples.
3. IMDB (originally released with no known license by Maas et al. [32]): This dataset contains
movie reviews designed for sentiment classification tasks. Following the approach outlined
in [37], we applied a one-shot prompt to guide the large language model (LLM) in using the
predefined sentiment labels effectively.
Annotation collection for HD. Specifically, following [37], the dataset D = {(qi, zi)}ℓ
i=1
contains ℓquestion-answer pairs, where qi are questions and zi are ground-truth answers. For each
qi, the model generates a response ˆzi, with predicted answers {ˆzi}ℓ
i=1. The LOS for each response,
{(X, p)i}ℓ
i=1, is saved. Correctness labels yi ∈{0, 1} are assigned by comparing ˆzi to zi, resulting in
the error-detection dataset {(X, p)i, yi}ℓ
i=1.
LLMs. We consider the following LLMs for our experiments on HD:
8For the Wiki-MIA dataset, we used a lookup table for Rank Encoding, where the index corresponds to ri and the
value is an embedding.
26

## Page 27

1. Mistral-7b-instruct-v0.2 [23] (License: Apache-2.0). Referred to as Mis-7b in the main text
and accessed through the Hugging Face interface at https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.3.
2. Llama-3-8b-Instruct [47] (License: Llama-39). Referred to as L-3-8b in the main text
and accessed through the Hugging Face interface at https://huggingface.co/meta-llama/
Meta-Llama-3-8B-Instruct3.
3. Qwen-2.5-7b-Instruct (License: Apache-2.0): Referred to as Q-2.5-7b in the main text
and accessed through the Hugging Face interface at https://huggingface.co/Qwen/Qwen2.
5-7B-Instruct.
B.5.2
Datasets for Data Contamination Detection
BookMIA. [43] The original BookMIA data have been obtained from the Hugging Face dataset
swj0419/BookMIA10, accessed via the Hugging Face python datasets API (License: MIT). The
dataset totals 9, 870 excerpts from a total of 100 books, of which 50 are labeled as members (positives)
and 50 are labeled as non-members (negatives).
Throughout all experiments on BookMIA, including the evaluation of baselines, we process only
the first 128 words from each excerpt, originally 512-word long. This expedient allowed for faster
LLM inference and lighter data storage at the time of dataset creation, i.e., the extraction and saving
of LLM outputs.
As no standard split is available for this dataset, we proceed by randomly forming training and
test sets in the proportions of, resp., 80% and 20%. To ensure that all excerpts from the same book
are in either one of the two sets (and never in both), we first separate books into two separate lists
based on their label, shuffle the obtained lists using a random seed of 42, and then, for each of
the two lists, take the first 80% of books as training books, and the remaining 20% as test books.
Training and test sets are obtained by taking the corresponding excerpts from, respectively, training
and test books. After this, we verified that the obtained sets are both approximately class-balanced
(≈50% of excerpts in both the training and test sets are labeled as positives).
In the case of the reference-based baseline, we consider the smallest-sized available counterparts
for the respectively attacked LLMs, namely: Pythia 70M for Pythia models and Llama-1 7B for
Llama models. All LLMs are accessed through the Hugging Face python interface, specifically:
EleutherAI/pythia-70m, EleutherAI/pythia-{6.9,12}b11 and huggyllama/llama-{7,13,30}b12
(License: Llama13).
WikiMIA. WikiMIA[43] (License: MIT) is the first benchmark for pre-training data detection,
comprising texts from Wikipedia events. The distinction between training and non-training data is
determined by timestamps. WikiMIA organizes data into splits based on sentence length, enabling
fine-grained evaluation. It also considers two settings: original and paraphrased. The original setting
evaluates the detection of verbatim training texts, while the paraphrased setting, where training texts
are rewritten using ChatGPT, assesses detection on paraphrased inputs. In this paper, we consider
the original (non-paraphrased) split and focus on the 32 and 64 split sizes, as they contain the
9https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE
10https://huggingface.co/datasets/swj0419/BookMIA.
11https://huggingface.co/EleutherAI/pythia-70m
(License:
Apache-2.0),
https://huggingface.co/
EleutherAI/pythia-6.9b, https://huggingface.co/EleutherAI/pythia-12b.
12https://huggingface.co/huggyllama/llama-7b, https://huggingface.co/huggyllama/llama-13b, https://
huggingface.co/huggyllama/llama-30b.
13https://huggingface.co/huggyllama/llama-13b/blob/main/LICENSE, https://huggingface.co/huggyllama/
llama-30b/blob/main/LICENSE
27

## Page 28

L-3-8b
Mis-7b
Q-2.5-7b
Target
L-3-8b
Mis-7b
Q-2.5-7b
Source
90.57 *
 ± 0.28
89.54 *
 ± 2.85
86.84 *
 ± 0.14
82.24 *
 ± 2.39
94.73 *
 ± 0.58
87.22 *
 ± 0.43
87.88 *
 ± 0.72
89.58
 ± 0.89
88.18 *
 ± 0.88
IMDB
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
L-3-8b
Mis-7b
Q-2.5-7b
Target
L-3-8b
Mis-7b
Q-2.5-7b
Source
72.60 *
 ± 0.34
67.87
 ± 0.04
71.93 *
 ± 0.12
63.62 *
 ± 0.09
72.92 *
 ± 0.45
71.46 *
 ± 0.92
67.76 *
 ± 0.53
70.04 *
 ± 1.45
73.70 *
 ± 1.21
HotpotQA
0.50
0.55
0.60
0.65
0.70
L-3-8b
Mis-7b
Q-2.5-7b
Target
L-3-8b
Mis-7b
Q-2.5-7b
Source
77.43 *
 ± 0.66
63.26 *
 ± 0.59
87.60 *
 ± 0.24
76.07 *
 ± 0.81
72.20 *
 ± 0.66
88.08 *
 ± 0.60
75.13 *
 ± 0.20
60.58 *
 ± 0.87
88.00 *
 ± 0.39
Movies
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
Figure 4: Cross-LLM transfer Test AUCs (cols: source LLMs, rows: target LLMs). Bold: finetuning
LOS-Net outperforms baselines, ∗: it outperforms the same LOS-Net trained from scratch.
largest number of samples, approximately 750 and 550, respectively. On top of the LLMs attacked
in BookMIA, here we also attack Mamba-1.4b (License: Apache-2.0), accessed via the Hugging Face
interface (https://huggingface.co/state-spaces/mamba-1.4b).
B.6
Extended Transferability Experiments
Setup. We fine-tune the LOS-Net given target LLM/dataset (depending on the task) for 10 epochs
—significantly fewer than the number of epochs used in our standard training protocol. Notably, this
fine-tuning process takes less than one minute in practice, on a single NVIDIA L-40 GPU.
To evaluate the effectiveness of fine-tuning, we benchmark the resulting model against two base-
lines. First, to assess knowledge transfer, we compare it with a LOS-Net model trained from scratch
under the same 10-epoch setup. Second, we compare against the strongest known probability-based
baselines with comparable detection latency—specifically, the strongest among Logits-mean/min/max
and Probas-mean/min/max (see Table 1). This comparison is essential: generalization scores above
0.5 AUC are only meaningful if they outperform non-learnable baselines that rely purely on prob-
abilities or logits. We note that we exclude P(true) and Semantic Entropy baselines from this
assessment, as they incur significantly higher latency (five order of magnitude higher than LOS-Net)
and are thus not directly comparable to LOS-Net ˙These methods require additional generation or
prompting to detect hallucinations. For a more detailed analysis, please refer to Section 5.4.
Comprehensive results are shown in Figure 4 (cross-LLM generalization) and Figure 5 (cross-
dataset generalization).
In the heatmaps, a superscript ‘∗’ indicates that the fine-tuned LOS-Net outperforms its scratch-
trained counterpart in the same setting—evidence of successful transfer. Bold entries denote cases
where it surpasses the best non-learnable probability-based method.
28

## Page 29

IMDB
Movies
HotpotQA
Target
IMDB
Movies
HotpotQA
Source
94.73 *
 ± 0.58
62.65 *
 ± 0.27
69.24 *
 ± 1.56
92.78 *
 ± 0.73
72.20 *
 ± 0.66
69.53
 ± 0.12
92.92 *
 ± 0.44
67.74 *
 ± 0.70
72.92 *
 ± 0.45
Mis-7b
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
IMDB
Movies
HotpotQA
Target
IMDB
Movies
HotpotQA
Source
90.57 *
 ± 0.28
67.41
 ± 3.27
63.54 *
 ± 0.39
75.37 *
 ± 4.91
77.43 *
 ± 0.66
67.02 *
 ± 0.23
88.79 *
 ± 0.43
77.19 *
 ± 0.27
72.60 *
 ± 0.34
L-3-8b
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
IMDB
Movies
HotpotQA
Target
IMDB
Movies
HotpotQA
Source
88.18 *
 ± 0.88
88.12 *
 ± 0.27
70.80 *
 ± 0.38
86.72 *
 ± 0.15
88.00 *
 ± 0.39
68.44 *
 ± 0.34
88.20 *
 ± 0.08
89.59 *
 ± 0.24
73.70 *
 ± 1.21
Q-2.5-7b
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
Figure 5: Cross-dataset transfer Test AUCs (cols: source data, rows: target data). Bold: finetuning
LOS-Net outperforms baselines, ∗: it outperforms the same LOS-Net trained from scratch.
B.7
Ablation Study
Existing methods often overlook a critical aspect of LOS. Specifically, they primarily rely on the
ATP, p, while neglecting the TDS, X. In this subsection, we conduct an ablation study to evaluate
the significance of the TDS in general, as well as its size, namely the hyperparameter K introduced
in Equation (1).
The Role of the TDS (X). As a case study, we examine both the DCD task on the BookMIA
dataset and the HD task across the three datasets: HotpotQA, IMDB, and Movies. Figures 6 and 7
report a close-up comparison between LOS-Net and our two proposed baselines, which explicitly
neglect the TDS, namely, ATP+R-Transf. and ATP+R-MLP. These plots consistently show
how the best-performing model is LOS-Net. In many cases, LOS-Net outperforms the alternatives by
a significant margin, indicating that the information encoded in the TDS (X) is crucial for both
tasks. Regarding the two ATP-based baselines, we report that ATP+R-Transf. obtains better
performance than ATP+R-MLP in 8 out of 12 cases, but these improvements do not seem to follow
a clear pattern across LLMs and datasets. The only exception is BookMIA, on which the former
architecture outperformed the latter across all the four attacked LLMs.
The hyperparameter K. To evaluate the impact of the hyperparameter K introduced in
Equation (1), we conduct a comprehensive case study focusing on the task of HD.
We experiment with various values of K, specifically K ∈{10, 50, 100, 500, 1000}, and trained
the same selected model whose results are reported in Table 1 for K = 1000. The corresponding
Test AUCs are presented in Figure 8.
From the reported bar plots, we do observe that performances either weakly increase with K (see,
e.g., Movies for Q-2.5-7b or HotpotQA on L-3-8b), or stay approximately constant (see, e.g., IMDB
on Mis-7b). In any case, the performance difference w.r.t. our default setting K = 1000 remains
contained. This is a valuable feature, as it unlocks the effective application of LOS-Net even on non
29

## Page 30

fully open LLMs such as the most recent models released by OpenAI14.
To complement Figure 8, we present Figure 9, which shows the average probability mass captured
for each value of K, computed as
PN
n=1(
PV
v=1(X′
n,v))/N. Across all LLM/dataset combinations and for
every K ∈{10, 50, 100, 500, 1000}, the captured probability mass exceeds 91%. This helps explain
why even small values such as K = 10 perform well, as observed in Figure 8.
Table 5: Test AUC scores for HD on Qwen-2.5-7b-Instruct (Q-2.5-7b). The best-performing method
is in bold, and the second best is underlined.
Method
HotpotQA
IMDB
Movies
Q-2.5-7b
Logits-mean
66.2
74.8
71.3
Logits-min
59.8
72.1
42.1
Logits-max
60.4
60.7
65.1
Probas-mean
67.5
74.6
74.2
Probas-min
54.4
65.4
44.7
Probas-max
61.8
50.1
72.9
ATP+R-MLP
71.38 ± 0.28
84.69 ± 0.37
78.06 ± 0.45
ATP+R-Transf.
69.34 ± 2.04
87.73 ± 0.03
77.37 ± 3.13
LOS-Net
73.71 ± 1.21
88.19 ± 0.88
88.00 ± 0.39
B.8
Results For Hallucination Detection for Qwen-2.5-7b
Table 5 reports results on our three considered HD datasets over LLM Qwen-2.5-7b-Instruct (Q-2.5-
7b) [53]. We can see LOS-Net outperforms all non-learnable output-based baselines by large margin,
as well as our learnable baselines ATP+R-Transf. and ATP+R-MLP.
B.9
Results On The WikiMIA Dataset
Table 6: Comparison of AUC over four different LLMs, on DCD, over the discussed baselines methods.
The best-performing method is in bold, and the second best is underlined.
Reference-based
approaches are shaded in pink.
Dataset →
WikiMIA - 32
WikiMIA - 64
LLM →
P-6.9b
L-13b
L-30b
M-1.4b
P-6.9b
L-13b
L-30b
M-1.4b
Loss
63.82 ±2.22
67.45 ±1.57
69.37 ±2.66
60.89 ±1.35
60.59 ±3.50
63.68 ±5.57
66.18 ±4.64
58.46 ±3.69
MinK
66.39 ±2.56
68.08 ±1.45
70.02 ±2.92
63.27 ±1.85
65.07 ±1.80
66.24 ±5.01
68.45 ±4.11
62.46 ±2.75
MinK++
70.60 ±3.58
84.93 ±1.76
84.46 ±1.43
67.06 ±2.78
71.82 ±3.73
85.66 ±2.25
85.02 ±2.79
67.24 ±4.06
Zlib
64.35 ±3.46
67.70 ±2.25
69.81 ±3.17
62.07 ±3.35
62.59 ±3.38
65.40 ±5.35
67.61 ±4.21
60.59 ±3.73
Lowercase
62.09 ±4.22
64.03 ±6.97
64.31 ±5.18
60.59 ±3.24
58.34 ±4.21
62.63 ±5.05
61.54 ±7.81
57.03 ±2.83
Ref
63.45 ±6.03
57.77 ±5.94
63.55 ±6.69
62.05 ±5.43
62.35 ±4.84
63.07 ±5.09
68.94 ±5.83
60.29 ±4.62
LOS-Net
76.98 ±3.36
93.46 ±1.31
93.76 ±1.56
71.04 ±9.07
76.00 ±5.48
87.86 ±3.73
93.04 ±2.51
79.39 ±2.61
The WikiMIA-32 and -64 datasets contain excerpts from Wikipedia articles, consisting of, resp.,
32 and 64 words. The distinction between contaminated and uncontaminated data is determined by
timestamps. As in [43, 58], we attack Mamba-1.4b [15] (M-1.4b), LlaMa-13b/30b [47] (L-13b/30b),
Pythia-6.9b [5] (P-6.9b).
14At the time of writing, OpenAI’s API only gives access to a maximum of 20 top scoring logprobs (https:
//platform.openai.com/docs/api-reference/completions/create, accessed May 2025.
30

## Page 31

Since WikiMIA does not provide an official training split and our method requires labeled
data, we perform 5-fold cross-validation with training, validation, and testing splits15 and rerun
all baselines under the same protocol for a fair comparison. Results are reported as the mean and
standard deviation across folds. For these datasets only, setting the hyperparameter K = 1000
(recall Equation (1)) led to suboptimal performance in preliminary experiments, thus, we set
K = “Full-Vocabulary”.
As shown in Table 6, LOS-Net consistently surpasses all baseline methods across all eight
combinations of LLMs and datasets. Notably, for L-30b, our model achieves an AUC score that
is more than 8 points higher than the best-performing baseline, MinK%++ for both datasets,
demonstrating a substantial improvement. Similarly, for P-6.9b, our model maintains a steady
advantage of approximately 5 AUC for both datasets, further underscoring its robustness. Overall,
the second-best method is MinK%++, followed by MinK%, consistent with the findings of [58].
B.10
Empirical Run-Times
In Table 7, we report the wall-clock training times (for the best selected model based on the held-out
validation set) and single-example detection times for LOS-Net
for all experiments presented in this
paper.
C
Additional Tasks Background
In this section, we provide some additional background and motivation for the DCD and HD tasks.
Data Contamination Detection.. Large-scale pre-training of LLMs typically involves crawling
vast amounts of online data, a common practice to meet their substantial data requirements. However,
this approach risks exposing models to evaluation datasets, potentially compromising our ability to
assess their generalization performance accurately [7], or, taking a different perspective, can pose
legal and ethical issues when models are accidentally exposed to copyrighted or sensitive data during
training. This phenomenon is typically referred to as Data Contamination. Recently, Li et al. [28]
demonstrated that LLMs from the widely used LLaMA [47] and Mistral [23] model families exhibit
significant data contamination, particularly concerning frequently used evaluation datasets.
Hallucination Detection.. LLMs’ tendency to generate untrustworthy outputs, commonly known
as "hallucinations," remains a significant challenge to their widespread adoption in real-world
applications [46]. To address this issue, various hallucination mitigation techniques have been
proposed, including retrieval-augmented generation [26, 21, 14], customized fine-tuning [34, 8, 39],
and, inference-time manipulation [27, 40, 59], to name a few. However, applying these methods to all
user-LLM interactions can be computationally expensive. As a more targeted approach, hallucination
detection has been explored to enable selective intervention only when necessary.
General Considerations on Annotations..
We consider access to a set of annotations y’s,
which we naturally associate with the corresponding LOS elements. These encode ground-truth
labels pertaining to problems of interest, e.g., whether the input text ⃗s is in the pretraining corpus
of f, or whether f hallucinated when generating ⃗g from prompt ⃗s. Collecting these annotations
is generally possible, and various strategies could be adopted. For example, for DCD, labels can
be gathered with collaborative efforts testing for text memorization, as studied e.g. in [12]. We
also note that annotations are immediately (and trivially) available for open-source LLMs with
15We use { 3
5, 1
5, 1
5} as the ratios for training, validation, and testing, respectively.
31

## Page 32

Table 7: Comparison of training and detection times of our model LOS-Net
across all the DC
settings explored in our paper, as well as HD settings for Mis-7b and L-3-8b. All are measured on a
single NVIDIA L-40 GPU.
Task
Target
LLM
Dataset
Training Time [h =
hours, m = minutes, s =
seconds]
Detection Time (Mean
± Std) [seconds]
HD
Mis-7b
HotpotQA
9m 19s
3.32 × 10−5 ± 1.20 × 10−5
s
IMDB
16m 8s
4.05 × 10−5 ± 1.83 × 10−5
s
Movies
17m 50s
1.95 × 10−5 ± 7.24 × 10−6
s
L-3-8b
HotpotQA
6m 39s
2.38 × 10−5 ± 7.18 × 10−6
s
IMDB
4m 23s
3.37 × 10−5 ± 1.53 × 10−5
s
Movies
11m 34s
3.05 × 10−5 ± 1.21 × 10−5
s
DCD
L-13b
WikiMIA-32
33m 6s
4.13 × 10−5 ± 1.67 × 10−6
s
WikiMIA-64
2m 7s
2.67 × 10−5 ± 1.12 × 10−5
s
BookMIA
7m 32s
3.67 × 10−5 ± 8.65 × 10−6
s
L-30b
WikiMIA-32
28m 40s
4.05 × 10−5 ± 3.10 × 10−6
s
WikiMIA-64
5m 8s
4.96 × 10−5 ± 2.54 × 10−5
s
BookMIA
16m 38s
3.94 × 10−5 ± 1.42 × 10−5
s
P-6.9
WikiMIA-32
24m 55s
2.91 × 10−5 ± 4.41 × 10−6
s
WikiMIA-64
26m 13s
3.18 × 10−5 ± 1.56 × 10−5
s
BookMIA
18m 23s
2.86 × 10−5 ± 6.16 × 10−6
s
P-12b
BookMIA
19m 49s
4.07 × 10−5 ± 4.89 × 10−6
s
M-1.4b
WikiMIA-32
1h 6m 18s
3.87 × 10−5 ± 1.27 × 10−6
s
WikiMIA-64
1h 16m 51s
3.12 × 10−5 ± 1.42 × 10−5
s
disclosed pretraining corpora such as Pythia [5]. As we demonstrated in Section 5, models trained on
annotations available for one LLM can, in some cases, be transferred and applied to another LLM.
For HD, ground-truth labels can be collected by providing the target LLM with inputs prompting
for completion or question answering on known facts and/or reasoning tasks. Hallucinations or error
annotations are derived by comparing the consistency of the model’s response with known, factually
32

## Page 33

true, or logically correct answers. For further details, refer to Section B.5.1.
33

## Page 34

D
LOS-Net Visualization
In Figure 10 we provide a visualization of our architecture, LOS-Net .
Row Sort
Rank Encoding
Concat
Transformer
Slice
X
(X, p)
Figure 10: A visualization of LOS-Net .
E
LLM Processing Pipeline
In Figure 11 we provide a visualization of the LLM processing pipeline.
X ⃗s
Input sequence 
(prompt)
⃗s
V
n
LLM  f
m
V
X ⃗g
⃗g
Output sequence 
(response)
Figure 11: LLM processing pipeline. Token sequence ⃗s is processed by an LLM f, generating full
TDSs Xs, Xg for input ⃗s and response ⃗g.
F
Importance of TDS Illustration
We demonstrate the importance of the TDS tensor through the following example, see Figure 12
34

## Page 35

Figure 12: Illustrative example of the importance of the TDS.
35

## Page 36

ATP+R-MLP
ATP+R-Transf.
LOS-Net
65.0
67.5
70.0
72.5
Test AUC
HotpotQA, L-3-8b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
70
72
Test AUC
HotpotQA, Mis-7b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
68
70
72
74
Test AUC
HotpotQA, Q-2.5-7b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
84
86
88
90
Test AUC
IMDB, L-3-8b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
88
90
92
94
Test AUC
IMDB, Mis-7b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
84
86
88
Test AUC
IMDB, Q-2.5-7b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
74
76
78
Test AUC
Movies, L-3-8b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
66
68
70
72
Test AUC
Movies, Mis-7b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
75
80
85
Test AUC
Movies, Q-2.5-7b
Figure 6: Ablation study evaluating the role of the TDS (X) and the ATP (p) on our HD setups,
including datasets HotpotQA, IMDB, Movies, and LLMs L-3-8b, Mis-7b, Q-2.5-7b.
36

## Page 37

ATP+R-MLP
ATP+R-Transf.
LOS-Net
60
80
Test AUC
BookMIA, P-6.9b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
60
80
Test AUC
BookMIA, P-12b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
70
80
90
Test AUC
BookMIA, L-13b
ATP+R-MLP
ATP+R-Transf.
LOS-Net
85
90
95
Test AUC
BookMIA, L-30b
Figure 7: Ablation study evaluating the role of the TDS (X) and the ATP (p) on BookMIA for
Pythia and Llama-1 LLMs.
37

## Page 38

10
50
100
500
1000
Top-K
0.7625
0.7650
0.7675
0.7700
0.7725
0.7750
0.7775
0.7800
0.7825
Test AUC
Movies, L-3-8b
10
50
100
500
1000
Top-K
0.68
0.69
0.70
0.71
0.72
0.73
Test AUC
Movies, Mis-7b
10
50
100
500
1000
Top-K
0.84
0.85
0.86
0.87
0.88
Test AUC
Movies, Q-2.5-7b
10
50
100
500
1000
Top-K
0.69
0.70
0.71
0.72
0.73
Test AUC
HotpotQA, L-3-8b
10
50
100
500
1000
Top-K
0.7150
0.7175
0.7200
0.7225
0.7250
0.7275
0.7300
0.7325
0.7350
Test AUC
HotpotQA, Mis-7b
10
50
100
500
1000
Top-K
0.71
0.72
0.73
0.74
0.75
0.76
Test AUC
HotpotQA, Q-2.5-7b
10
50
100
500
1000
Top-K
0.80
0.82
0.84
0.86
0.88
0.90
0.92
Test AUC
IMDB, L-3-8b
10
50
100
500
1000
Top-K
0.925
0.930
0.935
0.940
0.945
0.950
0.955
Test AUC
IMDB, Mis-7b
10
50
100
500
1000
Top-K
0.865
0.870
0.875
0.880
0.885
0.890
0.895
Test AUC
IMDB, Q-2.5-7b
Figure 8: Ablation study analyzing the effect of the hyperparameter K introduced in Equation (1).
38

## Page 39

Figure 9: Probability mass (y-axis) as a function of K (x-axis) for each LLM/dataset combination
considered in the HD study.
39
