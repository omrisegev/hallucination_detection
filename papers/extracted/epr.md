---
source_pdf: papers/EPR.pdf
slug: epr
pages: 8
extracted_on: 2026-07-13
---

# EPR

## Page 1

Learned Hallucination Detection in Black-Box LLMs using
Token-level Entropy Production Rate
Charles Moslonka
Artefact Research Center
Paris, France
MICS, CentraleSupélec, Université Paris-Saclay
Gif-sur-Yvette, France
{name.surname}@artefact.com
Hicham Randrianarivo
Artefact Research Center
Paris, France
MICS, CentraleSupélec, Université Paris-Saclay
Gif-sur-Yvette, France
{name.surname}@artefact.com
Arthur Garnier
Ardian
Paris, France
{name.surname}@ardian.com
Emmanuel Malherbe
Artefact Research Center
Paris, France
{name.surname}@artefact.com
Abstract
Hallucinations in Large Language Model (LLM) outputs for Ques-
tion Answering (QA) tasks can critically undermine their real-world
reliability. This paper introduces a methodology for robust, one-
shot hallucination detection, specifically designed for scenarios
with limited data access, such as interacting with black-box LLM
APIs that typically expose only a few top candidate log-probabilities
per token. Our approach derives uncertainty indicators directly
from these readily available log-probabilities generated during non-
greedy decoding. We first derive an Entropy Production Rate (EPR)
that offers baseline performance, later augmented with supervised
learning. Our learned model leverages the entropic contributions
of the accessible top-ranked tokens within a single generated se-
quence, without multiple re-runs per query. Evaluated across di-
verse QA datasets and multiple LLMs, this estimator significantly
improves token-level hallucination detection over state-of-the-art
methods. Crucially, high performance is demonstrated using only
the typically small set of available log-probabilities (e.g., top-10
per token), confirming its practical efficiency and suitability for
API-constrained deployments. This work provides a lightweight
technique to enhance the trustworthiness of LLM responses, at the
token level, after a single generation pass, for QA and Retrieval-
Augmented Generation (RAG) systems. Our experiments confirmed
the performance of our method against existing approaches on pub-
lic dataset as well as for a financial framework analyzing annual
company reports.
1
Introduction
Large Language Models (LLMs) have demonstrated remarkable ca-
pabilities across a wide range of tasks [1–3]. However, a significant
and persistent challenge is their propensity to generate hallucina-
tions [4, 5]. These are defined as LLM-generated content that is
factually incorrect, nonsensical, or unfaithful to a provided source,
despite often appearing plausible and coherent [6, 7]. The impact
of hallucinations can be severe, particularly in safety-critical ap-
plications such as medicine or infrastructure engineering [8, 9],
leading to the propagation of misinformation and erosion of user
trust. Therefore, detecting and quantifying the uncertainty associ-
ated with LLM outputs is paramount to building trust and enabling
responsible deployment.
The practical deployment of Uncertainty Quantification (UQ) [10]
and hallucination detection methods is often constrained by lim-
ited access to LLM internals. These limitations are especially true
when interacting with proprietary models through APIs, which
may expose only a small number of top-𝐾log-probabilities per
token (where 𝐾might be around 151). This black-box setting re-
stricts the applicability of techniques requiring access to full logits,
hidden states, or architectural modification [11]. Moreover, many
real-world applications require one-shot (or “single-round” [12])
detection—the ability to assess the reliability of a single generated
sequence without the need for multiple, often costly, model infer-
ences for the same input query to measure output variability.
This paper introduces a UQ methodology rooted in information
theory, tailored for these black-box, one-shot scenarios. Our ap-
proach leverages the accessible log-probabilities to derive entropic
measures of model hesitation during the generation process. We
define and evaluate a metric termed the Entropy Production Rate
(EPR) of a generated sequence. We show that EPR, calculated as the
average entropy of the token probability distributions across the
entire sequence, serves as an initial, unsupervised estimator of the
degree of hesitation exhibited by the model during inference.
Our second contribution is a supervised learning model that
adapts entropy measures to a dataset of annotations. By training
on the entropic contributions of top-ranked tokens across the se-
quence, we learn an estimator that more accurately distinguishes
between faithful and hallucinatory responses. This estimator can
also highlight high-uncertainty tokens in a generated sequence, as
displayed in Figure 1.
We detail the entropic framework, its theoretical and practical
underpinnings under limited log-probability access, and its empir-
ical validation. We showcase its potential as a practical tool for
improving the reliability of LLM systems, using RAG in a financial
scenario as an example. To facilitate adoption and future research,
we release our methods (EPR and WEPR) as an open-source Python
package available at https://github.com/artefactory/artefactual/.
1The maximum value is 𝐾= 20 for the OpenAI API at the time of writing.
arXiv:2509.04492v2  [cs.CL]  20 Jan 2026

## Page 2

Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe
Em
erson
’ s
reportednet
salesforthe
fiscal
year
2 0 2 3
were
$ 1 8 .
5
billion .
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Hallucination Score
Chatbot: Emerson’s reported net sales
for the fiscal year 2023 were $18.5 billion.
Without Reference Context
Em
erson
’ s
reportednet
salesforthe
fiscal
year
2 0 2 3
were
$ 1 5 .
2
billion .
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Hallucination Score
Chatbot: Emerson’s reported net sales
for the fiscal year 2023 were $15.2 billion.
With Reference Context
Figure 1: Illustration of token-level uncertainty detection in a financial RAG pipeline.
2
Related Work
UQ in Large Language Models. The primary goal of UQ in LLMs
is to enable them to signal their confidence — to “know what they
do not know” [13, 14]. LLM uncertainty arises from traditional
aleatoric (data-inherent) and epistemic (model-based) sources [15],
and additionally from their generative characteristics like diver-
gent multi-step reasoning paths and decoding stochasticity [12].
Traditional neural network UQ approaches, such as training modifi-
cations for aleatoric uncertainty [16] or dropout-based methods for
epistemic uncertainty [17, 18], are often ill-suited or prohibitively
costly for LLMs due to training expense or architectural differences.
Developing UQ techniques tailored to LLM intricacies remains an
active research area [19].
Information-theoretic entropy has emerged as a cornerstone for
assessing LLM uncertainty and hallucinations, based on the hy-
pothesis that higher output distribution entropy often correlates
with a greater likelihood of error [20]. Researchers have employed
diverse entropy measures to capture different facets of LLM uncer-
tainty [21]. Predictive entropy [22] assesses overall output distribu-
tion uncertainty. Semantic Entropy [23] quantifies uncertainty over
semantic meaning by clustering multiple generated responses, with
Discrete Semantic Entropy [24, 25] using frequency-based cluster
probabilities. Answer Entropy [26], common in QA, measures vari-
ability in complete answer strings from multiple trials, sometimes
enhanced with noise injection. Particularly relevant to our work,
token-level entropy [13] measures per-step uncertainty from the
next-token probability distribution and has been used to predict
errors in QA and mathematical reasoning [11, 27].
Entropy-driven detection techniques and challenges. Direct ap-
plication of entropy scores, often via thresholding, is a common
approach for hallucination detection or calibration [11, 13, 23].
More sophisticated techniques include Semantic Entropy Probes
(SEPs) [28]—learned models predicting Semantic Entropy from
single-pass hidden states for efficiency—or Shifting Attention to
Relevance (SAR) [29], which weights tokens by relevance. However,
entropy-based methods face challenges like high-certainty halluci-
nations [30], where LLMs produce low-entropy incorrect outputs,
necessitating strategies beyond simple thresholds. Moreover, the
computational cost of multi-sample methods (e.g., for Answer or
Semantic Entropy) can be prohibitive, highlighting the need for
efficient single-shot techniques such as SEPs [28], Logit-induced
Token Uncertainty (LogTokU) [31], and ours.
3
Theoretical Framework for Entropic
Uncertainty
Our approach to estimate uncertainty in LLM generations is rooted
in information theory [32, 33]. The core idea is to derive an entropic
score reflecting the model’s hesitation when generating tokens.
Unlike methods requiring multiple model runs to assess output
variability, our method is a one-shot analysis, leveraging the proba-
bilistic information from a single generated sequence.
3.1
Entropy computation during token
generation.
For a standard classification task with 𝑁𝑐classes, given an input
𝑞, a model outputs probabilities 𝑝𝑖= 𝑝(class𝑖|𝑞) for each class
𝑖∈{1, . . . , 𝑁𝑐}. The uncertainty can be quantified by the Shannon
entropy (in bits) of this distribution:
𝐻(𝑞) = −
𝑁𝑐
∑︁
𝑖=1
𝑝𝑖log2(𝑝𝑖) ≥0.
(1)
This entropy 𝐻(𝑞) is minimized to zero when the distribution
is sharply peaked (high certainty) and maximized at 𝐻max(𝑞) =
log2(𝑁𝑐) for a uniform distribution (maximum uncertainty).
Extending this to LLM sequence generation, let 𝑉= {𝑣𝑖}|𝑉|
𝑖=1 de-
note the model’s vocabulary, with size2 |𝑉|. For a given input query
𝑞, the LLM generates a sequence of tokens T = {𝑡1,𝑡2, . . . ,𝑡|T|}
(∀𝑗,𝑡𝑗∈𝑉). The generated sequence T depends on 𝑞, but we omit
2e.g., |𝑉|
=
217
=
131,072 for some contemporary models, such as
Mistral-Small-3.1-24B [34].

## Page 3

Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate
this dependency for clarity. At each generation step 𝑗, the LLM
internally computes a probability distribution {𝑝(𝑣𝑖| 𝑞,𝑡<𝑗)}|𝑉|
𝑖=1
for the next token 𝑡𝑗, conditioned on the preceding tokens 𝑡<𝑗:=
{𝑡1, . . . ,𝑡𝑗−1} and the input 𝑞. The complete entropy for the 𝑗-th
token’s distribution is:
𝐻(𝑞,𝑡<𝑗) = −
|𝑉|
∑︁
𝑖=1
𝑝(𝑣𝑖| 𝑞,𝑡<𝑗) log2 𝑝(𝑣𝑖| 𝑞,𝑡<𝑗).
(2)
Note that this quantity does not depend on the sampled token 𝑡𝑗.
In black-box scenarios involving proprietary LLMs accessed via
APIs, only the log-probabilities (or probabilities) for a small number,
denoted by 𝐾, of top-ranked candidate tokens are exposed (e.g.,
𝐾≤20). Thus, we estimate the per-token entropy based on these top
𝐾probabilities. Let 𝑟: {1, . . . , |𝑉|} →{1, . . . , |𝑉|} be the ranking
operator on the vocabulary indices, meaning 𝑟(𝑘) is the index 𝑖of
the 𝑘-th token in decreasing order of 𝑝(𝑣𝑖| 𝑞,𝑡<𝑗). The estimator
can then be written as:
˜𝐻𝐾(𝑞,𝑡<𝑗) = −
𝐾
∑︁
𝑘=1
𝑝𝑟(𝑘),𝑗log2
 𝑝𝑟(𝑘),𝑗
 ,
(3)
where 𝑝𝑟(𝑘),𝑗:= 𝑝(𝑣𝑟(𝑘) |𝑞,𝑡<𝑗) is the probability of the 𝑘-th ranked
token at generation step 𝑗.
3.2
Sufficiency of Top-𝐾Log-Probabilities for
Entropy Estimation
A critical question is whether ˜𝐻𝐾(𝑞,𝑡<𝑗), calculated from only the
top 𝐾probabilities, is a reliable proxy for the true entropy 𝐻(𝑞,𝑡<𝑗).
The discrepancy arises from the missing tail of the distribution:
Δ𝐻𝐾(𝑞,𝑡<𝑗) = 𝐻(𝑞,𝑡<𝑗) −˜𝐻𝐾(𝑞,𝑡<𝑗) = −
|𝑉|
∑︁
𝑘=𝐾+1
𝑝𝑟(𝑘),𝑗log2 𝑝𝑟(𝑘),𝑗.
(4)
We can establish an upper bound for this missing entropy by assum-
ing the remaining probability mass 𝑄𝐾(𝑞,𝑡<𝑗) = 1 −Í𝐾
𝑘=1 𝑝𝑟(𝑘),𝑗
is distributed uniformly among the remaining tokens considered
sampleable. When the LLM generation uses a cutoff parameter
𝐾samp ≪|𝑉| (e.g., 𝐾samp = 50), probabilities beyond 𝐾samp are set
to zero. In that case:
Δ𝐻𝐾,max(𝑞,𝑡<𝑗) = −𝑄𝐾(𝑞,𝑡<𝑗)

log2(𝑄𝐾(𝑞,𝑡<𝑗)) −log2(𝐾samp−𝐾)

(5)
The quality of ˜𝐻𝐾(𝑞,𝑡<𝑗) as an estimator of the entropy within
the relevant token pool (either full vocabulary or 𝐾samp limited) can
be assessed by comparing ˜𝐻𝐾(𝑞,𝑡<𝑗) and Δ𝐻𝐾,max(𝑞,𝑡<𝑗); a large
ratio ˜𝐻𝐾/Δ𝐻𝐾,max suggests that ˜𝐻𝐾(𝑞,𝑡<𝑗) captures most of the
relevant uncertainty.
Empirically, we observe a concentration in the log-probabilities
of lower-ranked tokens (e.g., for ranks 𝑖≈10 −15), as illustrated in
Figure 2. Differences between successive 𝑝𝑟(𝑘),𝑗tend to diminish for
larger 𝑘within the accessible 𝐾. When combined with 𝐾samp sam-
pling, this supports the idea that ˜𝐻𝐾(𝑞,𝑡<𝑗) captures the dominant
portion of the uncertainty within the set of practically sampleable
tokens, especially if 𝐾is reasonably close to 𝐾samp.
3.3
Influence of Sampling Temperature
In practice, at 𝑗-th generation step, a LLM first predicts raw log-
its 𝑙𝑗,𝑖and converts them to probabilities 𝑝𝑗,𝑖by a softmax func-
tion with a temperature parameter 𝑇samp, such that: 𝑝𝑗,𝑖(𝑇samp) =
𝑒𝑙𝑗,𝑖/𝑇samp/Í|𝑉|
𝑚=1 𝑒𝑙𝑗,𝑚/𝑇samp.
The per-token information entropy 𝐻(𝑞,𝑡<𝑗;𝑇samp) thus depends
on 𝑇samp, verifying:
• As 𝑇samp →0+, 𝑝𝑗,𝑖(𝑇samp) approaches 1 for the token with
the highest logit (assuming a unique maximum) and 0 for
others. Consequently, 𝐻(𝑞,𝑡<𝑗;𝑇samp) →0, reflecting high
certainty.
• As 𝑇samp →∞, 𝑝𝑗,𝑖(𝑇samp) approaches 1/|𝑉| for all tokens,
leading to a uniform distribution. Therefore, the entropy
𝐻(𝑞,𝑡<𝑗;𝑇samp) →log2 |𝑉|, its maximum possible value,
reflecting maximum uncertainty.
However, a critical distinction exists between the distribution used
for sampling tokens and the log-probabilities returned by modern
inference engines. Systems like vllm, or the OpenAI API, typically
expose log-probabilities computed from raw logits (equivalent to
fixing 𝑇= 1 for calculation), regardless of the 𝑇samp used to select
the token 𝑡𝑗. Consequently, while 𝑇samp dictates the trajectory of
the generated sequence T, our entropic features (defined in the
following sections) measure the model’s intrinsic hesitation based
on its unscaled raw distribution. This decoupling ensures that our
estimator remains mathematically well-defined and stable even if
the user employs conservative decoding settings (e.g., low 𝑇samp).
3.4
Entropy Production Rate (EPR) of the
sequence
Based on the per-token entropy estimate ˜𝐻𝐾(𝑞,𝑡<𝑗), we define
the Entropy Production Rate (EPR) for a generated sequence T
of length |T | in response to query 𝑞. It is the average estimated
entropy over the top 𝐾accessible log-probabilities, across all tokens
in the sequence:
EPR𝐾(𝑞, T) =
1
|T |
| T|
∑︁
𝑗=1
˜𝐻𝐾(𝑞,𝑡<𝑗)
(6)
= −1
|T |
| T|
∑︁
𝑗=1
𝐾
∑︁
𝑘=1
𝑝𝑟(𝑘),𝑗log2(𝑝𝑟(𝑘),𝑗).
This EPR𝐾may serve as a global measure of hesitation or uncer-
tainty for the generated sequence, given the black-box constraints
(see Fig. 3) In the following, we improve its alignment with ground-
truth annotations via supervised learning.
3.5
Supervision on Entropic Contributions by
Rank
We consider a dataset D where each entry is a query 𝑞, the gener-
ated sequence T, and an annotation 𝑌∈{0, 1} indicating whether
T is a correct (therefore considered non-hallucinatory) answer to
𝑞. Note that this annotation depends on the LLM used to generate
T. A realistic dataset size is on the order of hundreds to thousands
of entries.
To build a more nuanced supervised detector, we define features
based on the entropic contributions of tokens at specific ranks. For

## Page 4

Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe
1
5
10
15
Rank k
25
20
15
10
5
0
Log Probability
(a) Log-Probability vs. Rank k
1
5
10
15
Cutoff K
10
9
10
7
10
5
10
3
10
1
Missing Mass QK (Log Scale)
(b) Missing Mass QK vs. Cutoff K
Figure 2: Analysis of probability mass concentration on 180k tokens generated by Mistral-Small-3.1-24B (𝑇samp = 1.0). (a)
Distribution of log-probabilities for the token at rank 𝑘, showing a rapid decay in likelihood. (b) Distribution of the missing
probability mass 𝑄𝐾(log scale) when truncating at cutoff 𝐾. The median 𝑄𝐾drops below 10−4 by 𝐾= 10, validating the Top-𝐾
approximation.
0
0.2
0.4
0.6
0.8
1
1.2
1.4
1.6
1.8
0
1
2
3
4
Entropy Production Rate
Density
Distribution of EPR by Judgment
Non-hallucinated (µ = 0.183)
Hallucinated (µ = 0.367)
Figure 3: Distributions of EPR scores on Falcon-10B re-
sponses. The separation between the Hallucinated (red) and
Non-hallucinated (green) distributions illustrates the dis-
criminative power of the metric.
the 𝑗-th token generation and given 𝑘∈{1, . . . , 𝐾}, the entropic
contribution of the 𝑘-th ranked token is
𝑠𝑘,𝑗= −𝑝𝑟(𝑘),𝑗log2(𝑝𝑟(𝑘),𝑗).
Note that the actually sampled token at step 𝑗may not be the
highest-ranked token (𝑘= 1), especially with high 𝑇samp. Our es-
timation leverages the probabilities of the tokens at fixed ranks
𝑘, regardless of which token was sampled. We consider weighing
each term of ˜𝐻𝐾(𝑞,𝑡<𝑗) (Eq. (3)) according to its rank 𝑘:
𝑆𝛽(𝑞,𝑡<𝑗) = 𝛽0 +
𝐾
∑︁
𝑘=1
𝛽𝑘𝑠𝑘,𝑗,
(7)
where 𝛽∈R𝐾+1 are weights that aim to adapt the entropy
measure and are learned using the annotations of D, as described
in the following. This token-wise quantity 𝑆𝛽(𝑞,𝑡<𝑗) is close to
˜𝐻𝐾(𝑞,𝑡<𝑗), but can be negative.
We aggregate the token-wise scores to learn the parameters 𝛽
at the sequence level. A simple average of 𝑆𝛽
 𝑞,𝑡<𝑗
 across the
sequence captures the overall model hesitation. However, a single
moment of high uncertainty (a "spike") can often be indicative of a
hallucination, even if the rest of the sequence is generated confi-
dently, and in such case 𝑠𝑘,𝑗values will be high for the uncertain
generation, even at high ranks 𝑘. To capture these critical tokens,
we also consider the maximum entropic contribution 𝑠𝑘,𝑗for each
rank across the sequence. This leads to our final scoring function,
the Weighted Entropy Production Rate (WEPR):
WEPR𝛽,𝛾(𝑞, T) =
1
|T |
| T|
∑︁
𝑗=1
𝑆𝛽(𝑞,𝑡<𝑗) +
𝐾
∑︁
𝑘=1
𝛾𝑘max
𝑗
𝑠𝑘,𝑗
(8)
where 𝛾∈R𝐾are parameters to be learned along with 𝛽by maxi-
mizing on the annotations of the dataset D:
max
𝛽,𝛾
∑︁
𝑞,T,𝑌∈D
𝑌log

𝜎 WEPR𝛽,𝛾(𝑞, T)
(9)
+(1 −𝑌) log

𝜎 1−WEPR𝛽,𝛾(𝑞, T)
where 𝜎is the sigmoid function. This corresponds to a logistic re-
gression loss, without regularization, on the averaged contributions
𝑆𝛽(𝑞,𝑡<𝑗) across tokens and the maximal 𝑠𝑘,𝑗values. This way, the
signal provided by the entropy along the sequence is aligned to the
ground truth annotations.

## Page 5

Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate
Beyond this supervision, 𝜎 WEPR𝛽,𝛾(𝑞, T) ∈[0, 1] can be in-
terpreted as a confidence score for the generated sequence T to be
valid, with no hallucination. In particular, contrary to EPR𝐾(𝑞), this
quantity is scaled and can easily be displayed as a normalized score
to the user. Moreover, while the annotations are at the sequence
level, we can also consider a token-level measure, computed on the
entropic contributions features of each token:
𝜎 𝑆𝛽
 𝑞,𝑡<𝑗
  ∈[0, 1].
(10)
This score can be interpreted as the localized risk of hallucination
associated with the 𝑗-th token, allowing users to identify specific
parts of a less reliable response. A significant advantage of this
token-level score is its suitability for online estimation: it can be
computed and displayed in real-time as the sequence is generated,
token by token, without waiting for the full output T.
4
Experiments
4.1
Experimental Setup
4.1.1
Models and Datasets. 3 We evaluated our methods on a suite
of contemporary instruction-finetuned LLMs, including Mistral-
Small-3.1-24B-Instruct-2503 [34], Falcon3-10B-Instruct [35],
Phi-4 [36], and Ministral-8B-Instruct-2410 [37]. For Question
Answering (QA) tasks, we primarily utilized the TriviaQA [38]
(Wikipedia domain) and WebQuestions [39] datasets. We also ex-
plored the applicability of our entropic measures in a Retrieval-
Augmented Generation (RAG) [40] context using a specialized
dataset based on financial reports from the ArGiMi-Ardian Finance
dataset [41] to observe how entropy varies with the provision of
relevant context (we detail this particular application below). We
focused on models generating relatively short answers, typical of
non-extended reasoning QA.
4.1.2
Answer Generation and Log-probability Extraction. LLMs were
served using the vllm inference library [42]. To ensure variabil-
ity and access to meaningful probability distributions for entropy
calculation, we used non-greedy decoding with 𝑇samp = 1.0. The
top_p parameter was set to 1.0, while the sampling cutoff was set
to 𝐾samp = 50. This truncation allows us to compute the theoretical
upper bound on missing entropy defined in Eq. (5). For each gen-
erated token, we extracted the accessible top-𝐾log-probabilities
(typically 𝐾≤15 or 𝐾≤20 depending on the API/model).
4.1.3
Hallucination Annotation Protocol. Generated answers for
the QA tasks were labeled as hallucinated or non-hallucinated (as
displayed in Figure 3) using an LLM-as-a Judge [43–46] approach,
employing Gemma-3-12b-it [47]. The judge LLM semantically com-
pared the generated answer to the ground-truth answer (and any
available aliases). If both answers were judged similar, the gen-
erated answer was labeled as non-hallucinated. This automated
annotation is scalable and suitable for one-shot analysis. However it
may mislabel random correct guesses with high uncertainty or fail
to flag confident but incorrect statements (high-certainty hallucina-
tions). To validate its reliability, we conducted a human evaluation.
A group of 15 annotators, comprising Ph.D. and graduate students
in Machine Learning from our laboratory, hand-labeled a sample
3Code and data to reproduce the experiments presented in this paper are available at
https://github.com/artefactory/artefactual/releases/tag/ECIR2026.
set of 1,333 output/ground-truth pairs. The human-machine agree-
ment was exceptionally high: they matched on 1,275 instances
(95.7%), yielding a Cohen’s Kappa of 𝜅= 0.898 (see Table 2). This
strong alignment confirms that the automated labels used for our
large-scale training and evaluation are reliable proxies for human
judgment.
4.1.4
RAG Uncertainty Analysis. Beyond general QA, we explored
the applicability of our entropic methods for analyzing outputs
in RAG pipelines. The premise is that uncertainty, as captured by
EPR or WEPR, might increase when an LLM generates an answer
without sufficient supporting context. We demonstrated this capa-
bility using the ArGiMi-Ardian Finance 10k dataset [41]. This
open-source resource comprises 52,000 financial annual reports,
written in English and extracted from their original PDF format,
covering the period from the late 90s to 2024. For our evaluation, we
focused on a specific 81-page report from this corpus, from which
we manually constructed 50 question/answer/source-page triplets.
Answers were generated both with and without the retrieval of the
relevant context pages. The task was framed as detecting “missing
context” hallucinations—instances where the model hallucinates
facts due to the specific absence of the retrieved information—based
solely on the entropic signature of the answer.
4.2
Evaluation Protocol
4.2.1
Metrics. The primary metrics used to evaluate the perfor-
mance of our hallucination detection methods are the Area Under
the Precision-Recall Curve (PR-AUC) and the Area Under the Re-
ceiver Operating Characteristic Curve (ROC-AUC). As both metrics
led to identical conclusions, we report only ROC-AUC scores for
conciseness.
4.2.2
Comparison to existing methods. We benchmark our pro-
posed methods, EPR (unsupervised) and WEPR (supervised), against
two state-of-the-art black-box detection techniques:
• SelfCheckGPT [11]: A multi-shot method that measures the
consistency across several answers generated for the same
query. We used the BertScore variant with 10 generated
samples.
• HalluDetect [19]: A single-shot method that, similar to our
approach, engineers features from model probabilities to
train a detector.
This selection allows for a direct comparison with a powerful multi-
shot approach (SelfCheckGPT) and a closely related single-shot
competitor (HalluDetect), for which we used the same number of
accessible log-probabilities (K=15) to ensure a fair comparison.
4.2.3
Training and Evaluation Strategy. The supervised logistic re-
gression model have been trained only on TriviaQA data (generated
sequences and their labels), as being the only dataset with anno-
tation. For evaluating these methods on TriviaQA, we generated
training and testing splits by grouping the data points by their
original query. This process ensures that no data points originat-
ing from the same query appear in the training and testing sets
for a given fold, mitigating potential data leakage and overesti-
mating generalization performance. To ensure stability and robust
performance estimation, particularly given potential variability in

## Page 6

Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe
Dataset
Model
Unsupervised
Supervised
SCGPT [11]
EPR
(ours)
HalluDetect [19]
WEPR
(ours)
TriviaQA [38]
(Hallucination
Detection)
Mistral-Small-24B
79.0
74.6
78.7
82.0
Falcon-3-10B
70.1
75.4
79.0
84.1
Phi-4 (14.7B)
71.4
78.2
83.8
85.4
Ministral-8B-2410
81.1
81.4
86.1
85.8
WebQuestions [39]
(Hallucination
Detection
Generalization)
Mistral-Small-24B
59.3
62.5
62.8
64.8
Falcon-3-10B
65.8
68.2
69.3
73.2
Phi-4 (14.7B)
65.0
65.2
66.3
66.6
Ministral-8B-2410
66.2
65.4
71.6
72.6
ArGiMi-Ardian [41]
(Missing Context
Detection)
Mistral-Small-24B
64.2
81.0
81.1
82.8
Falcon-3-10B
67.6
82.8
87.0
89.3
Phi-4 (14.7B)
79.7
91.0
88.5
93.6
Ministral-8B-2410
65.9
73.1
81.6
81.8
Table 1: ROC-AUC of compared methods across different LLMs, using 𝐾= 15 accessible log-probabilities, on: (i) hallucination
detection on TriviaQA; (ii) generalization to WebQuestions (supervised models trained on TriviaQA); and (iii) missing context
detection on ArGiMi-Ardian (also trained on TriviaQA).
smaller datasets or specific model-dataset pairings, we employed
bootstrapping on 1000 iterations during model evaluation.
4.3
Results
4.3.1
Hallucination Classification Performance. Table 1 shows re-
sults on TriviaQA for hallucination detection. The EPR baseline
demonstrates notable discriminative ability, with ROC-AUC scores
ranging from 74.6 (Mistral-Small-3.1-24B) to 81.4 (Ministral-
8B-Instruct-2410). For a similar computational overhead as other
single-shot methods (around 80 ± 20 µs per score) WEPR consis-
tently matches or outperforms EPR, SelfCheckGPT, and HalluDe-
tect across all LLMs. Note that multi-shot methods are multiple
orders of magnitude more computationally costly (at least 10s for
SelfCheckGPT). We also verified the robustness of our approach
to sampling parameters; for instance, on Falcon-3-10B, reducing
the temperature to 𝑇samp = 0.6 resulted in a negligible change in
ROC-AUC (< 1.0 point difference). Since the log-probabilities used
for WEPR features are derived from raw logits, this result confirms
that our entropic signal remains highly discriminative even when
applied to the more deterministic generation trajectories typical of
conservative decoding. Regarding the generalization on WebQues-
tions (Table 1), we observe that although performance is generally
lower on this dataset, WEPR maintains its advantage across LLMs.
The improved classification performance of the WEPR model
is further illustrated by the sample Precision-Recall curves for
Falcon-3-10B on TriviaQA (Figure 5), where the WEPR curve con-
sistently dominates that of the EPR baseline and the HalluDetect
[19] method.
4.3.2
RAG uncertainty. Table 1 presents the performance of com-
pared methods (trained on the general TriviaQA dataset for su-
pervised ones) for the missing context detection task. The results
show that the WEPR model significantly outperforms the other
2
4
6
8
10
12
14
0.68
0.7
0.72
0.74
0.76
0.78
0.8
0.82
0.84
Number of token considered K
Mean ROC-AUC
Mean ROC-AUC vs Number of tokens considered K
WEPRK
EPRK
Figure 4: Mean ROC-AUC with respect to 𝐾for Falcon-3 on
TriviaQA.
methods in this domain-specific RAG scenario. This indicates that
an entropic detector trained on a general QA dataset can transfer,
to a notable extent, to specialized domains for identifying responses
generated with insufficient context. Such a mechanism can be read-
ily applied in RAG pipelines to flag outputs where context retrieval
may have been inadequate or where the model is otherwise strug-
gling, thereby triggering deeper information retrieval or alerting
the user.
4.3.3
Effect of the Number of Accessible Log-probabilities (𝐾). We
investigated the impact of 𝐾on hallucination detection perfor-
mance. Figure 4 displays the evolution of the mean ROC-AUC for

## Page 7

Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate
Task annotated
LLM-as-a-Judge
Sentence-level
Token-level
Samples
1,333
312
334
Agreement (%)
95.7
77.8
78.1
Table 2: Human agreement with (i) LLM annotations validating labels in Section 4.3, and (ii) WEPR scores at sentence and token
granularity, measuring adoption potential.
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
Recall
Precision
Precision-Recall Curves Comparison
WEPR (AUC = 0.75)
EPR (AUC = 0.61)
HalluDetect (AUC = 0.69)
Figure 5: Precision-Recall curves of 3 methods for Falcon-3
on TriviaQA.
Falcon-3-10B on TriviaQA as 𝐾varies. The results indicate that
performance generally improves with increasing ranked token con-
tributions up to a certain point. Notably, for several models, the
ROC-AUC tends to plateau when 𝐾reaches approximately 8 to
10. This suggests that incorporating entropic contributions beyond
the top ∼10 ranks may yield diminishing returns for hallucination
detection. This finding has practical significance, implying that
effective detection can be achieved even when API access is limited
to a relatively small number of top logprobs, thereby reducing data
requirements and potentially API costs.
4.3.4
Human adoption analysis. We wanted to measure the per-
ceived downstream quality of our WEPR scoring methods, at the
two possible granularities: sentence-wise (𝜎 WEPR𝛽,𝛾(𝑞, T) and
token-wise (Eq. 10). At least three independent annotators (from
the same pool of researchers) rated each sample as acceptable or
not (mean 3.76). The adoption score is then computed based on
the majority vote, with results compiled in Table 2. The high level
of satisfaction for token scores (78.1%) indicates that the learned
coefficients 𝛽𝑘from our WEPR model can be used to compute a
weighted entropic contribution for each token 𝑗in a sequence, thus
offering a finer-grained view of uncertainty hotspots within the
generated text (see text colors of Figure 1).
5
Limitations
While our methods demonstrate promising results for one-shot,
black-box hallucination detection, several limitations warrant ac-
knowledgment. First, our experiments were conducted on mid-sized
LLMs (approximately 8B–24B parameters). Performance on bigger
models (e.g., 100B+ parameters or closed-source APIs like GPT-
5) remains to be tested. Second, the current study focuses on QA
tasks that yield relatively short, factual answers. We did not study
tasks requiring extensive multi-step reasoning or lengthy outputs,
where the aggregated entropy signal might become less distinct
or different types of uncertainty could dominate. Moreover, the
method may produce false positives by flagging model hesitation
related to stylistic choices or sentence formulation, as opposed to
uncertainty about the factual content of the answer. This effect was
observed to be more pronounced at the beginning of generated se-
quences. Finally, our approach, like many other uncertainty-based
methods, is inherently limited in its ability to detect "high-certainty
hallucinations" [30]—instances where the model generates incor-
rect information with low output entropy (i.e., high confidence)
due to memorization errors. If an LLM has strongly learned erro-
neous facts during its training, entropic measures based on output
probabilities may not flag such confident fabrications.
6
Conclusion
We addressed one-shot hallucination detection in black-box LLMs
by introducing a methodology rooted in information-theoretic prin-
ciples, leveraging only the top-𝐾token log-probabilities typically
exposed by APIs. Our primary contribution is a supervised ap-
proach that utilizes features representing individual top-ranked
tokens mean and maximal entropic contributions. Experiments
across QA datasets and multiple LLMs show that the learned es-
timator outperforms a baseline Entropy Production Rate (EPR).
Notably, strong detection performance is achieved even with as
few as 𝐾≤10 accessible log-probabilities, underscoring practical
efficiency under API constraints. We also showcased utility in a
specialized financial setting for RAG and highlighted token-level
uncertainty visualization.
This work provides a practical technique for enhancing the trust-
worthiness of LLM responses in both general QA and RAG from a
single generation pass. Future work may extend this approach to
more complex reasoning tasks and study the interpretability of the
learned entropic weights.
Acknowledgments
This work was done as part of the ArGiMi project, funded by
BPIFrance under the France2030 national effort towards numerical

## Page 8

Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, and Emmanuel Malherbe
common goods. CM wishes to thank Antoine Delaby for prelim-
inary discussions on RAG applications, along with Gabriel Trier,
Lucas Tramonte and Rayane Bouaita for their help on the question
dataset. HR also wishes to thank Yves-Roland Kouayip and Mathieu
Crochet from Artefact for their help in creating the annotation tool.
References
[1] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton
Bakhtin, Yuxiang Wu, and Alexander Miller.
Language Models as Knowl-
edge Bases?
In Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 2463–2473, Hong Kong,
China, 2019. Association for Computational Linguistics.
[2] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian
Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H.
Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William
Fedus. Emergent Abilities of Large Language Models, October 2022.
[3] OpenAI et. al. GPT-4 Technical Report, March 2024.
[4] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate
Saenko. Object Hallucination in Image Captioning, March 2019.
[5] S. Baker and T. Kanade.
Hallucinating faces.
In
Proceedings
Fourth IEEE International Conference
on
Automatic Face
and
Gesture Recognition (Cat. No. PR00580), pages 83–88, March 2000.
[6] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Yejin Bang, Delong Chen, Wenliang Dai, Ho Shu Chan, Andrea Madotto, and
Pascale Fung. Survey of Hallucination in Natural Language Generation. ACM
Computing Surveys, 55(12):1–38, December 2023.
[7] Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha,
Priyatham Kattakinda, and Soheil Feizi. Llm-check: Investigating detection of
hallucinations in large language models. In A. Globerson, L. Mackey, D. Bel-
grave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural
Information Processing Systems, volume 37, pages 34188–34216. Curran Asso-
ciates, Inc., 2024.
[8] Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Chanwoo Park, Mingyu
Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, et al. Medical
hallucinations in foundation models and their impact on healthcare. arXiv
preprint arXiv:2503.05777, 2025.
[9] Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, Li Zhang,
Zhongqi Li, and Yuchi Ma. Exploring and Evaluating Hallucinations in LLM-
Powered Code Generation, May 2024.
[10] T. J. Sullivan. Introduction to Uncertainty Quantification. Springer, December
2015.
[11] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. SelfCheckGPT: Zero-
Resource Black-Box Hallucination Detection for Generative Large Language
Models, October 2023.
[12] Xiaoou Liu, Tiejin Chen, Longchao Da, Chacha Chen, Zhen Lin, and Hua Wei. Un-
certainty Quantification and Confidence Calibration in Large Language Models:
A Survey, March 2025.
[13] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain,
Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-
Johnson, et al. Language models (mostly) know what they know. arXiv preprint
arXiv:2207.05221, 2022.
[14] Amos Azaria and Tom Mitchell. The Internal State of an LLM Knows When It’s
Lying, October 2023.
[15] Yarin Gal. Uncertainty in Deep Learning. PhD thesis, University of Cambridge,
2016.
[16] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in
machine learning: An introduction to concepts and methods. Machine learning,
110(3):457–506, 2021.
[17] Janis Postels, Francesco Ferroni, Huseyin Coskun, Nassir Navab, and Federico
Tombari. Sampling-free epistemic uncertainty estimation using approximated
variance propagation. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pages 2931–2940, 2019.
[18] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation:
Representing model uncertainty in deep learning. In International Conference
on Machine Learning, pages 1050–1059. PMLR, 2016.
[19] Ernesto Quevedo, Jorge Yero Salazar, Rachel Koerner, Pablo Rivas, and Tomas
Cerny. Detecting hallucinations in large language model generation: A to-
ken probability approach. In World Congress in Computer Science, Computer
Engineering & Applied Computing, pages 154–173. Springer, 2024.
[20] C. E. Shannon. A mathematical theory of communication. The Bell System
Technical Journal, 27(3):379–423, July 1948.
[21] Marylou Gabrié, Andre Manoel, Clément Luneau, jean barbier, Nicolas
Macris, Florent Krzakala, and Lenka Zdeborová.
Entropy and mu-
tual information in models of deep neural networks.
In Advances in
Neural Information Processing Systems, volume 31. Curran Associates, Inc.,
2018.
[22] Wonbin Kweon, Sanghwan Jang, SeongKu Kang, and Hwanjo Yu.
Uncer-
tainty Quantification and Decomposition for LLM-based Recommendation. In
Proceedings of the ACM on Web Conference 2025, pages 4889–4901, April 2025.
[23] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic Uncertainty: Lin-
guistic Invariances for Uncertainty Estimation in Natural Language Generation,
April 2023.
[24] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting halluci-
nations in large language models using semantic entropy. Nature, 630(8017):625–
630, June 2024.
[25] Nicola Cecere, Andrea Bacciu, Ignacio Fernández Tobías, and Amin Mantrach.
Monte Carlo Temperature: A robust sampling strategy for LLM’s uncertainty
quantification methods, February 2025.
[26] Litian Liu, Reza Pourreza, Sunny Panchal, Apratim Bhattacharyya, Yao Qin, and
Roland Memisevic. Enhancing Hallucination Detection through Noise Injection,
February 2025.
[27] Eojin Joo, Young-Jun Lee, and Ho-Jin Choi. Entropy-based Sentence-level Halluci-
nation Score in Large Language Models. In 2025 IEEE International Conference
on Big Data and Smart Computing (BigComp), pages 77–78, February 2025.
[28] Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and
Yarin Gal. Semantic Entropy Probes: Robust and Cheap Hallucination Detection
in LLMs, June 2024.
[29] Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing
Xu, Bhavya Kailkhura, and Kaidi Xu. Shifting Attention to Relevance: Towards
the Predictive Uncertainty Quantification of Free-Form Large Language Models,
May 2024.
[30] Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky, and Yonatan Belinkov.
Trust Me, I’m Wrong: High-Certainty Hallucinations in LLMs, February 2025.
[31] Huan Ma, Jingdong Chen, Joey Tianyi Zhou, Guangyu Wang, and Changqing
Zhang. Estimating LLM Uncertainty with Evidence, May 2025.
[32] Leon Brillouin. Science and Information Theory. Courier Corporation, July
2013.
[33] Marc Mezard and Andrea Montanari. Information, physics, and computation.
Oxford University Press, 2009.
[34] Mistral Small 3.1 | Mistral AI. https://mistral.ai/news/mistral-small-3-1.
[35] TII
Falcon-LLM
Team.
The
falcon
3
family
of
open
models.
https://huggingface.co/blog/falcon3, December 2024.
[36] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya
Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauff-
mann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.
[37] Un Ministral, des Ministraux | Mistral AI. https://mistral.ai/fr/news/ministraux.
[38] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A
Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension,
May 2017.
[39] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic Parsing
on Freebase from Question-Answer Pairs. In David Yarowsky, Timothy Baldwin,
Anna Korhonen, Karen Livescu, and Steven Bethard, editors, Proceedings of the
2013 Conference on Empirical Methods in Natural Language Processing, pages
1533–1544, Seattle, Washington, USA, October 2013. Association for Computa-
tional Linguistics.
[40] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks, April 2021.
[41] Hicham Randrianarivo, Charles Moslonka, Arthur Garnier, and Emmanuel Mal-
herbe. The ArGiMi datasets. https://huggingface.co/datasets/artefactory/Argimi-
Ardian-Finance-10k-text, 2024.
[42] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica.
Efficient
memory management for large language model serving with PagedAttention.
In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems
Principles, 2023.
[43] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena, December 2023.
[44] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo
Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large Language Models are not Fair
Evaluators, August 2023.
[45] Vaibhav Adlakha, Parishad BehnamGhader, Xing Han Lu, Nicholas Meade,
and Siva Reddy.
Evaluating Correctness and Faithfulness of Instruction-
Following Models for Question Answering. Transactions of the Association
for Computational Linguistics, 12:681–699, May 2024.
[46] Hossein A. Rahmani, Emine Yilmaz, Nick Craswell, Bhaskar Mitra, Paul Thomas,
Charles L. A. Clarke, Mohammad Aliannejadi, Clemencia Siro, and Guglielmo
Faggioli. LLMJudge: LLMs for Relevance Judgments, August 2024.
[47] Gemma et. al. Team. Gemma 3 Technical Report, March 2025.
