---
source_pdf: papers/Zero-source LLM Hallucination Detection with Human-like Criteria Probing.pdf
slug: zero-source-llm-hallucination-detection-with-human-like-crit
pages: 30
extracted_on: 2026-07-13
---

# Zero-source LLM Hallucination Detection with Human-like Criteria Probing

## Page 1

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Jiahao Yang * 1 2 Shuhai Zhang * 1 3 Hailong Kang * 1 Feng Liu 2 Qi Chen 4 Mingkui Tan 1 3
Abstract
Large language models (LLMs) often halluci-
nate by generating factually incorrect or unfaith-
ful content, posing significant risks to their safe
use.
Detecting such hallucinations is particu-
larly challenging under the zero-source constraint,
where no model internals or external references
are available, and detection must rely solely on
the textual query–answer pair. In this paper, we
propose Human-like Criteria Probing for Hal-
lucination Detection (HCPD), a paradigm that
emulates the multi-faceted reasoning of human
evaluators.
Its core is a Human-like Criteria
Probing (HCP) mechanism, in which a LLM
agent adaptively decomposes its judgment into
a weighted set of interpretable criteria and aggre-
gates criterion-specific scores into a final truth-
fulness measure. To achieve this adaptive ca-
pability, we introduce a reward-based alignment
scheme using only weak supervision from seman-
tic consistency. At inference, we employ a multi-
sampling aggregation strategy to ensure robust
decisions while preserving full interpretability.
We further provide theoretical analysis supporting
the reliability of our approach. Extensive experi-
ments show that HCPD consistently outperforms
state-of-the-art baselines, offering an effective
and explainable solution for zero-source hallu-
cination detection. Code is available at https:
//github.com/TRISKEL10N/HCPD.
1. Introduction
Large language models (LLMs) have rapidly advanced and
are increasingly deployed across a broad range of appli-
cations, including information retrieval (Zhu et al., 2025),
*Equal contribution
1South China University of Tech-
nology
2The University of Melbourne
3Pazhou Laboratory
4Australian Institute for Machine Learning, Adelaide Uni-
versity.
Correspondence
to:
Mingkui
Tan
<mingkui-
tan@scut.edu.cn>, Qi Chen <qi.chen04@adelaide.edu.au>, Feng
Liu <fengliu.ml@gmail.com>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
decision support (Chiang et al., 2024; Chen et al., 2024b;
Ma et al., 2025), and domain-specific assistance in health-
care (Benary et al., 2023; Vrdoljak et al., 2025), finance (Yu
et al., 2024), and education (Neumann et al., 2024). Never-
theless, their practical use is constrained by hallucinations,
where LLMs generate responses that are factually incorrect,
ungrounded, or unfaithful to the user’s intent, posing signifi-
cant risks in safety-critical settings. Consequently, reliable
hallucination detection has become essential for the safe
and trustworthy deployment of LLM-based assistants.
A critical challenge is that practical hallucination detection
frequently operates under a strict zero-source constraint
(Fang et al., 2025; Yang et al., 2025b). In prevalent open-
world scenarios, the auditing process is entirely decoupled
from the generation. For instance, third-party auditors (e.g.,
social media platforms, news agencies) must evaluate mas-
sive user-uploaded texts without knowing the underlying
source LLMs. Similar situations also occur when the vast
majority of end-users interact with LLMs through web in-
terfaces (e.g., ChatGPT1, Gemini2 and Claude3) or browser
extensions, where plain text is the sole accessible output.
Consequently, commercial APIs, internal states, and auxil-
iary resources (e.g., external knowledge bases) are typically
unavailable. Under such realistic restrictions, robust detec-
tion must rely solely on the observed query–answer pair.
Unfortunately, most existing approaches are not directly ap-
plicable under the aforementioned constraint. While effec-
tive in the traditional knowledge-based hallucination detec-
tion, retrieval-augmented or fact-verification methods (Sem-
nani et al., 2023; Hu et al., 2024; Chen et al., 2024c) require
access to web or knowledge resources, whose availability
and reliability are difficult to guarantee. Avoiding exter-
nal references, confidence-based and metric-based methods
(Malinin & Gales, 2021; Kuhn et al., 2023; Park et al., 2025)
mainly depend on model internals, which are unattainable
for black-box or commercial systems. Self-supervised or
consistency-based methods (Kadavath et al., 2022; Manakul
et al., 2023) typically employ static, task-agnostic heuris-
tics, limiting their ability to capture the precise, context-
dependent judgments across diverse domains. Moreover,
1https://chatgpt.com
2https://gemini.google.com/app
3https://claude.ai/new
1
arXiv:2606.12900v1  [cs.AI]  11 Jun 2026

## Page 2

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
most detectors provide only binary labels or scalar scores,
offering limited interpretability and diagnostic insight.
In contrast, human experts rarely judge a response using a
single monolithic criterion. They instead decompose evalu-
ation into multiple dimensions, adapt their relative weights
to the context, and provide evidence-grounded judgment.
This observation motivates a general zero-source detection
paradigm that emulates human-style evaluative reasoning.
In this paper, we propose Human-like Criteria Probing for
Zero-source Hallucination Detection (HCPD). Its core is
an Human-like Criteria Probing (HCP) mechanism, which
enables a pre-trained LLM agent to evaluate responses
through a transparent, multi-step process (Section 4.3). For
each query–answer pair, the agent first adaptively generates
a set of fine-grained criteria (e.g., factual accuracy, logical
consistency) and their context-aware importance weights,
then scores the text against each criterion, and finally ag-
gregates these scores into an overall truthfulness measure,
effectively emulating the nuanced, multi-perspective reason-
ing of a human expert. To enable this adaptive judgment
capability, we introduce a reward-based alignment training
scheme, which leverages weak supervision from semantic
consistency to teach the agent how to decompose and weight
criteria without needing ground-truth hallucination labels
(Section 4.4). At inference, we apply a multi-sampling ag-
gregation strategy to reduce variance from the stochastic
generation process, performing K independent HCP eval-
uations per instance and averaging the results for a robust
final decision (Section 4.5). We further provide a statistical
characterization of both training and inference behaviors,
and derive a threshold-free performance characterization by
bounding the ranking error probability (Section 4.6). Ex-
tensive experiments demonstrate that HCPD outperforms
existing state-of-the-art methods, validating the effective-
ness under the zero-source hallucination detection.
Our contributions are summarized as follows:
• An adaptive, multi-criteria probing framework for zero-
source detection: We are the first to explicitly formalize
hallucination detection under zero-source constraint,
and propose Human-like Criteria Probing mechanism,
which reframes detection as a process of context-aware
criteria generation, weighting, and aggregation. By en-
abling an agent to adaptively decompose its judgment
into interpretable dimensions, our method emulates the
nuanced, multi-faceted reasoning of human evaluators,
moving beyond monolithic scoring paradigms.
• Weakly-supervised alignment training without ground-
truth labels: To achieve reliable adaptive judgment in
the scoring agent, we introduce a reward-based align-
ment training scheme using only weak supervision
derived from semantic consistency. This method ef-
fectively teaches the agent to identify, weight, and
score relevant criteria without requiring any annotated
hallucination data, making it uniquely suited for the
practical zero-source constraint.
• A stable and interpretable inference strategy with theo-
retical guarantees: We design a multi-sampling aggre-
gation strategy during inference to mitigate generation
variance, enhancing decision robustness while preserv-
ing full interpretability through the generated criteria
and weights. Furthermore, we develop a theoretical
analysis, providing formal insights into the feasibility
and reliability of our probing-based approach.
2. Preliminaries
Group Relative Policy Optimization. Group Relative Pol-
icy Optimization (GRPO) is a reinforcement learning al-
gorithm designed for the stable and efficient fine-tuning of
LLMs. To obviate the need for an explicit value function
commonly required in policy gradient methods (Schulman
et al., 2017), GRPO adopts the average reward of multiple
sampled outputs conditioned on the same prompt as an im-
plicit baseline. Concretely, for each input x, the current pol-
icy fθ samples a group of outputs {Y1, . . . , YG} and assigns
each output a critic-based scalar reward r(Yg). Rather than
optimizing the absolute reward, GRPO constructs a group-
relative advantage Ag = r(Yg) −1
G
PG
j=1 r(Yj) within the
group, which normalizes rewards within the group and in-
herently reduces variance of the gradient estimates. The
policy model is then optimized to increase the likelihood
of outputs with above-average rewards, while constraining
updates to stay close to a reference policy f0 (typically the
initialization) to preserve generation quality. The objective
for a single group is defined as:
J (θ)= 1
G
PG
g=1[fθ(Yg|x)
f0(Yg|x)Ag]−β · DKL(fθ(·|x)||f0(·|x)),
(1)
where β is a hyperparameter controlling the strength of the
Kullback–Leibler divergence (Csisz´ar, 1975) penalty. For
simplicity, we reuse J (θ) to denote the objective aggregated
over the distribution when used in theoretical bounds.
By evaluating and comparing multiple outputs for the same
prompt, GRPO obtains a robust, context-aware learning sig-
nal and achieves more stable convergence compared to per-
sample absolute reward optimization. This property makes
it particularly suitable for aligning our scoring agent, where
rewards are dense yet require precise calibration across di-
verse evaluation properties.
3. Related Work
Hallucination Detection. Although truthfulness is a funda-
mental requirement of language generation, large language
2

## Page 3

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Factual 
Verification
Logical 
Consistency
Semantic 
Accuracy
Social 
Fairness
Timeline 
Verification
Multi
Sampling
Q & A
<Criteria ��: (Weight ��)>;
<Criteria �� : (Weight ��)>;
……
<Analysis>
<Score: ��
 � >
<Criteria ��: (Weight ��)>;
<Criteria �� : (Weight ��)>;
……
<Analysis>
<Score: ��
 � >
<Criteria ��: (Weight ��)>;
<Criteria �� : (Weight ��)>;
……
<Analysis>
<Score: ��
 � >
……
Aggregate
Reward 
Function
Rolling
Out
Online Update
Final Score
[ � / 10 ]
Human Judgement
Human-like Criteria Probing Mechanism
Inference
GRPO
Agent ��
……
Figure 1. Overview of the proposed HCPD. Given a query–answer pair (q, a), the agent instantiates a set of specific criteria {ci}m
i=1
and corresponding importance weights {wi}m
i=1. The criterion-level partial scores {si}m
i=1 are subsequently produced and aggregated
into an overall truthfulness measure sp. During GRPO training, we fine-tune the agent by maximizing the score-alignment reward that
encourages the predicted sp to match weak supervision labels derived from consistency-based similarity measures. At inference time, we
invoke the agent K times and aggregate the resulting scores to obtain a reliable decision ¯s.
models (LLMs) still frequently produce outputs that are
factually incorrect or contextually inconsistent, commonly
referred to as hallucination. Consequently, hallucination
detection (Lin et al., 2022; Azaria & Mitchell, 2023; Kuhn
et al., 2023; Ren et al., 2023; Manakul et al., 2023; Zhang
et al., 2023; Lin et al., 2024; Chen et al., 2024a; Du et al.,
2024; Park et al., 2025) has emerged as a central research
focus for the safe and reliable deployments.
A prevailing research paradigm attributes LLM hallucina-
tions to predictive uncertainty. Probability-based methods
quantify such uncertainty via metrics including Perplexity
(Ren et al., 2023), Length-Normalized Entropy (Malinin
& Gales, 2021), and Semantic Entropy (Kuhn et al., 2023).
In contrast, consistency-based methods evaluate agreement
across multiple sampled responses either through similarity
metrics such as BERTScore (Zhang et al., 2019), ROUGE
(Lin et al., 2024), natural language inference, and prompt-
based self-consistency verification (Manakul et al., 2023), or
via spectral analysis of response covariance matrices such as
eigenvalue decomposition (Chen et al., 2024a). Verbalized-
based methods elicit confidence signals by prompting mod-
els to express uncertainty explicitly in natural language,
through verbalized uncertainty (Lin et al., 2022) or self-
evaluation (Kadavath et al., 2022) mechanisms. Neverthe-
less, these uncertainty-driven signals remain limited as hal-
lucinations frequently occur even in high-confidence gener-
ations. Whereas multi-sample evaluation partially mitigates
this issue, it incurs substantial computational overhead.
A complementary perspective infers textual truthfulness by
exploiting internal model states. For instance, CCS (Burns
et al., 2022) extracts latent knowledge from activation pat-
terns, SAPLMA (Azaria & Mitchell, 2023) trains classifiers
directly on hidden representations, HaloScope (Du et al.,
2024) identifies hallucination-related subspaces via singular
value decomposition, and TSV (Park et al., 2025) intro-
duces learnable steering vectors to adapt latent features for
improved separability. Despite their effectiveness, these
methods necessitate access to the internal model represen-
tations, thereby limiting their applicability in real-world
deployment scenarios where the underlying model architec-
ture and training data provenance remain undisclosed.
4. Method
4.1. Problem Definition
LLM Hallucination. In the context of text generation, hal-
lucination refers to contents that are either factually inconsis-
tent with established knowledge or contextually unfaithful
to the given source or query intent (Huang et al., 2025).
Such errors typically arise from the model’s inherent uncer-
tainty, limitations in its learned knowledge, or failures in
reasoning. Characterizing and probing these failure modes
is therefore essential for the reliable deployment of LLMs.
Hallucination Detection under Zero-source Constraint.
The goal of hallucination detection is to assess whether a
model-generated response exhibits factual inconsistencies
or contextual unfaithfulness (Ji et al., 2023; Farquhar et al.,
2024). Formally, let Q denote the space of user queries and
A the space of model outputs. For a query q ∈Q and a
generated answer a ∈A, the detection model f produces a
binary label y ∈{0, 1} indicating whether the response con-
stitutes hallucinations. In this work, we focus on the zero-
source constraint: the detector f cannot access the LLM’s
internal states (e.g., token probabilities) or output distribu-
tion during inference. Furthermore, no external knowledge
bases or reference texts are available. It must operate solely
on the textual pair (q, a), making the task challenging yet
aligned with practical black-box deployment.
Challenges in Hallucination Detection under Zero-
source Constraint. Developing an effective detector under
this constraint faces several challenges. First, without access
3

## Page 4

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Table 1. The structured and interpretable output from the agent.
Component
Example
Q & A
Qes: In which country were the 1948 Winter
Olympics held?
Ans: Norway.
Specific
Criteria
1. Factual Grounding (Weight: 60%)
2. Temporal Consistency (Weight: 20%)
3. Semantic Precision (Weight: 20%)
Analysis
- Factual Grounding: The 1948 Winter Oly-
mpics were held in St. Moritz, Switzerland, n-
ot Norway. This is a clear factual error.
- Temporal Consistency: The response does
not specify a year, so there are no temporal is-
sues directly in the text.
- Semantic Precision: The response is vague
and does not provide the correct information,
leading to semantic distortion as the user is n-
ot given accurate details.
Score (1∼10)
1
to model-internal confidence signals (e.g., token logits), the
detector must rely entirely on the semantic content, requir-
ing a deep understanding to identify subtle inconsistencies.
Second, the absence of reference sources or external knowl-
edge eliminates straightforward fact-checking, forcing the
detector to reason intrinsically about plausibility. Third, hal-
lucinations are heterogeneous; they may manifest as factual
errors, logical fallacies or semantic misalignments, demand-
ing a flexible, multi-perspective evaluation rather than a
single static criterion. Finally, for trust and usability, deci-
sions must be interpretable, clarifying why a text is deemed
hallucinatory. These motivate the need for an adaptive, in-
terpretable, and purely text-based probing mechanism.
4.2. Motivations and Method Overview
Motivations. When assessing the correctness of a text, Hu-
man evaluators naturally engage in holistic and multifaceted
reasoning. They do not apply a single, rigid rule but dynam-
ically consider various dimensions, e.g., factual accuracy,
logical soundness, temporal consistency, and contextual
faithfulness, and intuitively weigh their importance based on
the specific content. This context-dependent, multi-criteria
weighting yields two practical advantages: i) adaptivity —
the ability to focus on the most diagnostic checks for each in-
stance, improving detection precision; and ii) interpretabil-
ity — the ability to explain negative judgments by pointing
to specific violated criteria. These observations motivate
us to design a detection paradigm that emulates the human
evaluative process: decomposing the holistic judgment into
a set of interpretable criteria, dynamically determining their
relevance (weights) based on the input, and synthesizing a
final decision through a transparent aggregation.
Method Overview. Motivated by the above analysis, we
propose Human-like Criteria Probing for Zero-source Hal-
lucination Detection (HCPD), which employs a Human-like
Criteria Probing (HCP) mechanism as its core (Figure 1).
Specifically, given an input pair (q, a), HCP adaptively uses
an LLM agent to generate a set of interpretable evaluation
criteria along with their relative importance weights and out-
put an aggregate truthfulness score (Section 4.3). To achieve
this capability into the agent, we devise a Reward-Based
Alignment Training scheme using weak supervision from
semantic consistency, teaching the agent to decompose and
weight criteria without ground-truth labels (Section 4.4). To
reduce the generation randomness, we introduce a Multi-
Sampling Aggregation strategy by performing K indepen-
dent probes per instance and averaging the resulting scores
as the final detection statistic; this multi-sampling aggrega-
tion stabilizes the decision while preserving interpretability
(Section 4.5). We also provide theoretical analysis justifying
the feasibility of our probing approach (Section 4.6).
4.3. Human-like Criteria Probing Mechanism
Inspired by the multi-faceted nature of human evaluation,
we develop Human-like Criteria Probing (HCP), a struc-
tured reasoning framework that enables a pre-trained LLM
to function as an adaptive scoring agent for hallucination
detection, moving beyond conventional monolithic scoring.
Human-like Criteria Probing. Given the query-answer
pair (q, a) under evaluation, the agent fθ produces inter-
pretable, criterion-wise assessments. Specifically, it first
adaptively generates a set of fine-grained specific crite-
ria {ci}m
i=1 derived from predefined General Evaluation
Criteria set C in broader dimensions such as factual and
logical coherence. The agent then autonomously assigns
context-sensitive importance weights {wi}m
i=1 to these spe-
cific criteria, emphasizing, for instance, temporal accuracy
for historical queries and logical soundness for scientific
explanations. Finally, it predicts a partial score si for each
weighted criterion and aggregates them into an overall truth-
fulness measure sp, thereby mimicking the contextual and
multi-faceted nature of human judgment. The operation of
the agent can be formalized as:
sp = Pm
i=1wi · si,
{(ci, wi, si)}m
i=1 ←fθ(q, a; C), (2)
where m is the number of specific criteria, the weight wi ≥
0 and P
i wi = 1, the scores are integers in {1, . . . , 10},
and the General Evaluation Criteria is given by:
C = {Factual, Logical, Semantic, Temporal, Social}. (3)
Structured and Interpretable Output. To ensure trans-
parency and facilitate downstream processing, the agent is
constrained to generate outputs in a strictly structured format
(Table 1). For each specific criterion, the output explicitly
reports its assigned weight, the corresponding evidence that
supports or contradicts the response, and the associated par-
tial score. These components are subsequently aggregated
4

## Page 5

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
into an overall score via weighted synthesis. This design
renders full interpretability of the detector’s decisions and
enables fine-grained attribution of hallucinated content.
4.4. Reward-based Alignment Training
To equip the agent fθ with precise evaluative capabilities,
we train it with a reward-based alignment paradigm using
Group Relative Policy Optimization (GRPO) (Shao et al.,
2024). This aligns the agent’s adaptive probing and scoring
behavior with a weak supervision signal derived from se-
mantic consistency, eliminating the reliance on expensive
human-annotated hallucination labels.
Training Data Construction and Labeling. Our training
data is constructed from standard question-answer (QA)
benchmarks that pair queries with human-verified reference
answers. Specifically, for each query q sourced from es-
tablished datasets (e.g., TriviaQA (Joshi et al., 2017)), a
corresponding ground-truth answer ˆa is available. We fur-
ther leverage an auxiliary LLM (e.g.., the test target model)
to generate a set of candidate responses {a(n)}N
n=1 for the
same query, intentionally sampling outputs that span a spec-
trum from factually correct to clearly hallucinated.
Following common practice in hallucination detection
benchmarks, we adopt widely used consistency metrics
(e.g., BLEURT (Sellam et al., 2020)) in natural language
processing (NLP) as a source of weak supervision label-
ing. Concretely, for each generated answer a(n), we com-
pute its semantic consistency to the reference ˆa, yielding
sim(ˆa, a(n)) ∈[0, 1]. Consistent with prior work (Du et al.,
2024), responses with a consistency score above 0.5 are
treated as relatively faithful, whereas those below are la-
beled as hallucinated. To obtain a graded supervision signal,
we translate this continuous score to a discrete 1–10 scale:
s(n)
l
= clip(⌊10 · sim(ˆa, a(n))⌉, 1, 10),
(4)
where the reference answer ˆa is assigned the maximum
score of 10. This procedure yields a weakly supervised
dataset D = {(q, {(ˆa, 10)} ∪{(a(n), s(n)
l
)}N
n=1)} that cap-
tures gradual degrees of hallucination severity and enables
the agent to learn fine-grained distinctions.
Score-alignment Reward Function. To encourage the
agent towards producing scores consistent with the weak
supervision labels, we introduce a score-alignment reward
function. Given an input pair (q, a), the agent performs
human-like criteria probing and outputs a structured eval-
uation containing a predicted score sp ∈{1, . . . , 10}. The
reward function assigns a scalar reward r ∈[0, 1] by directly
comparing the predicted score sp with the label sl:
r =



1 −|sp −sl|
9
,
if the output is well-formed,
0,
otherwise,
(5)
and optimize according to Eqn. (1).
The reward attains its maximum value r = 1 when the pre-
diction perfectly matches the label (sp = sl) and decreases
linearly with the absolute deviation between them. Notably,
the score-alignment reward implicitly enforces structural
constraints on the agent’s output: any deviation from the
prescribed format that prevents reliable extraction of a valid
score results in r = 0; and the inherent KL (Csisz´ar, 1975)
regularization of GRPO in Eqn. (1) limits deviation from
the initial policy (Shao et al., 2024).
Differentiable Scoring v.s. Binary Classification. The
adoption of a differentiable scoring scheme rather than bi-
nary “True/False” classification is a central design choice.
1) The graded formulation better reflects the inherently
varying degrees of hallucination severity, which ranges
from minor factual imprecision to fully fabricated content.
By modeling hallucination on a fine-grained scale, the agent
can capture subtle deviations that binary labels necessarily
collapse. 2) The reward defined in Eq. (5) induces an
informative policy optimization signal. By penalizing
prediction errors proportionally to their magnitude, it yields
a denser reward landscape than binary rewards, which only
indicate correctness without guiding the magnitude or direc-
tion of improvement. 3) Scalar scoring enables flexible
decision-making at inference time. Unlike binary classi-
fiers with fixed operating points, our agent’s scores can be
thresholded at varying levels to accommodate different pre-
cision–recall trade-offs without retraining. Collectively, this
scoring-alignment reward formulation not only facilitates
effective learning under weak supervision but also endows
the detector with calibrated, adaptable judgment capabilities
essential for effective hallucination detection.
4.5. Robust Inference via Multi-sampling Aggregation
Due to the inherent stochasticity of language model gener-
ation, a single evaluation from the agent may exhibit non-
negligible variance. To reduce this randomness and improve
decision reliability, we employ a multi-sampling aggrega-
tion strategy during inference.
Formally, the trained agent fθ is independently invoked
K times for the same input (q, a) pair, yielding a set of
structured evaluations together with corresponding synthe-
sized final scores {s(k)
p }K
k=1. We obtain a robust estimate
of response truthfulness by aggregating these scores via
arithmetic averaging:
¯s = 1
K
PK
k=1s(k)
p .
(6)
This aggregation suppresses stochastic fluctuations in in-
dividual evaluations, resulting in more reliable and sta-
ble detection outcomes. Experimental results indicate that
5

## Page 6

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
our method achieves substantially stronger detection per-
formance than SelfCKGPT (Manakul et al., 2023) while
incurring a similar time cost.
4.6. Theoretical Analysis
To establish a theoretical foundation for HCPD, we provide
guarantees for both its training and inference stages. Our
analysis yields three core results: (i) Theorem 1 ensures that
GRPO training anchors the learned scoring behavior to the
weak supervision signal in expectation; (ii) Proposition 1
justifies multi-sampling aggregation as a variance-reduction
mechanism via a concentration bound; and (iii) Corollary
1 derives a threshold-free performance characterization by
bounding the ranking error probability.
We first provide theoretical guarantees for the rationality
and effectiveness of the training and inference framework.
Theorem 1. (Expectation alignment under training) Let
x denote an input and sl(x) its corresponding weak label.
Consider the ℓ1-risk of predicting score s: Rx(s) ≜|s −
sl(x)|, whose minimizer is s = sl(x). Let Y ∼fθ(· | x)
be a stochastic well-formed generation of the agent, and
let Sθ(x) = sp(x, Y ) denote the parsed score. Define the
conditional mean score as µθ(x) ≜E

Sθ(x) | x

, we have
Ex

|µθ(x) −sl(x)|

≤J ′(θ),
(7)
where J ′(θ) is affine-equivalent to the GRPO objective
J (θ) defined in Eqn. (1).
Theorem 1 formalizes a training-time alignment guarantee:
optimizing the KL-regularized GRPO objective forces the
expected parsed scores µθ(x) toward the weak supervision
label sl(x) in expectation over the training distribution. This
result links the GRPO objective to stable scoring behavior
and constitutes the training-side foundation of our end-to-
end detection analysis.
While characterizing training-time alignment, inference in-
troduces an additional uncertainty: each evaluation produces
a random score Sθ(x). Proposition 1 provides Hoeffding’s
concentration bound for K parsed scores.
Proposition 1. (Multi-sampling concentration) Fix an
inference input x. Suppose the K well-formed generations
Y1, . . . , YK
i.i.d.
∼fθ(· | x). For each k ∈{1, . . . , K}, define
the parsed score S(k)
θ (x) ≜sp(x, Yk) ∈{1, . . . , 10}, and
let ¯s(x) ≜
1
K
PK
k=1 S(k)
θ (x) denote the aggregated score
in Eqn. (6). Then for any bias threshold u > 0,
P
 |¯s(x)−E[Sθ(x) | x]| ≥u | x

≤2 exp

−
2Ku2
(10 −1)2

.
(8)
Due to the requirement of domain knowledge reserves or ex-
pert intervention, a rigorous human-level annotation of hal-
lucination severity s⋆is almost inaccessible. Accordingly,
we treat the similarity-based label sl as a weak proxy and
model it as: sl(x) = g(s⋆(x)) + ϵ(x), where g is a mono-
tone non-decreasing mapping, and ϵ captures conditional
bias arising from stochasticity and systematic knowledge
discrepancies. For the detection process based on this, we
have the following derivations.
Corollary 1. (Ranking error decomposition) Let x denote
an input and sl(x) its corresponding weak label. Assume
the proxy bias is uniformly bounded: |ϵ(x)| ≤bmax,
∀x.
Let Y1, . . . , YK
i.i.d.
∼
fθ(· | x) be well-formed genera-
tions, and define S(k)
θ (x) and ¯s(x) as in Proposition 1.
Let x+ and x−denote independent draws from the dis-
tributions of true (non-hallucinated) and hallucinated in-
puts, respectively. Define the ranking error probability
Erank ≜P
 ¯s(x+) ≤¯s(x−)

. Then for any ∆> bmax,
Erank ≤P
 g(s⋆(x+)) −g(s⋆(x−)) ≤2∆

|
{z
}
intrinsic separability
+
4 J ′(θ)
∆−bmax
|
{z
}
training alignment
+ 4 exp

−
2K
(10 −1)2
∆−bmax
2
2
|
{z
}
multi-sampling concentration
,
where J ′(θ) is affine-equivalent to the GRPO objective
J (θ) defined in Eqn. (1).
Corollary 1 establishes a decomposable upper bound on
the detector’s ranking error probability. This bound makes
explicit the principal factors that affect the detection per-
formance. Beyond the intrinsic separability of the data,
it crucially shows that a smaller training alignment loss
J ′(θ) and a larger inference sampling size K both con-
tribute to a reduction of the error bound. This provides
theoretical support for our design choices: optimizing the
GRPO objective effectively aligns the agent’s scores with
the proxy supervision, while the multi-sampling aggregation
strategy suppresses inference variance at an exponential rate.
Consequently, the proposed training and inference compo-
nents jointly ensure reliable detection. We refer readers to
Appendix A for more details.
5. Experiments
5.1. Experimental Settings
Datasets. We conduct the experiments on four standard
QA datasets: TriviaQA (Joshi et al., 2017), SciQ (Welbl
et al., 2017), NQ Open (Kwiatkowski et al., 2019), and
CoQA (Reddy et al., 2019). For NQ Open, we use all
3, 610 samples from the validation set. From TriviaQA, we
randomly sample 3, 310 test pairs while 3, 000 training pairs
are drawn from SciQ and CoQA. Each dataset is split into
training and test sets with a 3 : 1 ratio, and all reported
results are averaged over 5 independent random splits. All
6

## Page 7

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Table 2. Comparisons with hallucination detection baselines on different datasets for LLaMA-3.1-8b and Qwen-3-8b in terms of AUROC
(%), where ♣denotes methods trained on fully labeled datasets.
Model
Method
TriviaQA
SciQ
NQ Open
CoQA
Avg.
Llama-3.1-8b
LN-Entropy (Malinin & Gales, 2021)
73.62±2.20
62.69±2.73
52.36±1.53
74.52±1.86
65.80
Self-evaluation (Kadavath et al., 2022)
56.07±1.92
54.12±1.82
59.83±3.68
62.51±1.42
58.13
CCS (Burns et al., 2022)
78.20±1.89
58.85±3.03
55.50±1.89
68.98±2.04
65.38
SelfCKGPT (Manakul et al., 2023)
74.58±1.90
59.68±2.45
62.13±2.60
70.61±2.10
66.75
Perplexity (Ren et al., 2023)
80.62±2.62
66.12±2.85
57.92±2.60
81.41±2.21
71.52
SAPLMA♣(Azaria & Mitchell, 2023)
78.51±3.16
85.63±0.96
76.23±0.82
71.58±1.35
77.99
Semantic Entropy (Kuhn et al., 2023)
78.71±3.09
77.81±3.17
61.04±4.29
75.26±4.63
73.21
Lexical Similarity (Lin et al., 2024)
77.96±2.03
67.09±2.05
62.85±1.57
77.53±0.90
71.36
EigenScore (Chen et al., 2024a)
51.35±1.23
51.52±0.82
52.17±2.13
52.00±1.19
51.76
HaloScope♣(Du et al., 2024)
58.19±5.79
69.04±6.36
63.38±3.02
72.11±4.96
65.68
TAD ♣(Vazhentsev et al., 2025)
72.01±1.03
66.75±1.96
68.88±2.56
74.86±2.10
70.63
TSV♣(Park et al., 2025)
79.78±3.36
80.01±1.17
70.17±1.41
69.31±6.75
74.82
HCPD (Ours)
86.25±1.08
86.04±2.25
90.38±3.58
90.07±2.58
88.19
Qwen-3-8b
LN-Entropy (Malinin & Gales, 2021)
64.66±2.55
61.22±4.44
68.70±4.11
54.28±3.51
62.22
Self-evaluation (Kadavath et al., 2022)
54.74±3.59
51.36±0.36
60.02±1.18
57.01±1.44
55.78
CCS (Burns et al., 2022)
57.51±1.97
57.96±3.03
58.63±5.53
55.10±4.58
57.30
SelfCKGPT (Manakul et al., 2023)
77.12±2.73
66.78±2.40
80.51±4.53
67.57±2.30
73.00
Perplexity (Ren et al., 2023)
59.65±2.42
59.27±4.59
67.81±4.32
54.06±3.63
60.20
SAPLMA♣(Azaria & Mitchell, 2023)
78.11±1.09
86.63±1.53
72.86±1.20
80.28±1.40
79.47
Semantic Entropy (Kuhn et al., 2023)
81.74±2.78
70.84±5.41
75.10±1.97
70.59±5.09
74.57
Lexical Similarity (Lin et al., 2024)
52.33±1.44
53.61±2.77
56.75±4.30
59.13±2.32
55.45
EigenScore (Chen et al., 2024a)
52.60±2.03
52.85±2.57
56.39±3.26
52.79±0.37
53.66
HaloScope♣(Du et al., 2024)
58.21±3.99
74.98±3.98
57.25±1.50
62.18±4.49
63.16
TAD ♣(Vazhentsev et al., 2025)
71.27±1.90
55.98±4.37
76.82±3.77
66.94±0.58
67.75
TSV♣(Park et al., 2025)
73.42±3.89
78.77±0.94
61.38±3.43
68.40±4.92
70.49
HCPD (Ours)
93.69±0.89
92.63±2.90
87.35±6.22
84.80±1.01
89.62
model responses are generated using greedy decoding. More
details on dataset are shown in Appendix C.1.
Baselines. We compare our method against various base-
lines from multiple technical paradigms for hallucination
detection. 1) Logit-based approaches: Perplexity (Ren et al.,
2023), Length Normalized Entropy (LN-entropy) (Malinin
& Gales, 2021) and Semantic Entropy (Kuhn et al., 2023);
2) Consistency-based approaches: Lexical Similarity (Lin
et al., 2024), SelfCKGPT (Manakul et al., 2023) and Eigen-
Score (Chen et al., 2024a); 3) Verbalized-confidence ap-
proach: Self-evaluation (Kadavath et al., 2022); 4) Internal
state-based approaches: Contrast-Consistent Search (CCS)
(Burns et al., 2022), SAPLMA (Azaria & Mitchell, 2023),
HaloScope (Du et al., 2024), TAD (Vazhentsev et al., 2025)
and TSV (Park et al., 2025).
Implementation Details. We compare our method with
baselines on 7 LLMs, i.e., LLaMA-3.1-8b, LLaMA-3.1-
70b (Dubey et al., 2024), LLaMA-2-7b, LLaMA-2-13b
(Touvron et al., 2023b), Qwen-3-8b (Yang et al., 2025a),
Qwen-2.5-7b, and Qwen-2.5-14b (Yang et al., 2024a). For
our method, we adopt a Qwen-2.5-7b as the agent and train
it according to the Open-R1 implementation4. Through the
optimization parameters in our method, we set lr = 2 ×
10−4 on LLaMA-3.1-8b and lr = 1 × 10−4 on Qwen-3-8b.
For the coefficient β that controls the DKL strength, we set
β = 0.05 on TriviaQA, SciQ with Qwen-3-8b and TriviaQA
4https://github.com/huggingface/open-r1
with LLaMA-3.1-8b, while β = 0.04 for all others. More
details are shown in Appendix C.2.
Evaluation. Following Farquhar et al. (2024), we evaluate
detection performance using the Area Under the Receiver
Operating Characteristic Curve (AUROC). As a proxy for
ground-truth factuality, we employ BLEURT (Sellam et al.,
2020) to measure semantic consistency between a generated
answer and its reference. Generated answers with a consis-
tency score exceeding 0.5 are regarded as relatively faithful,
whereas those below are labeled as hallucinated.
5.2. Comparisons with Baselines
Results on LLaMA-3.1-8b. As reported in Table 2, most
training-free methods exhibit limited detection performance
(about 51.76% to 73.21% on average). While some meth-
ods display moderate separability on simple benchmarks
(e.g., TriviaQA, CoQA), their performance drops signifi-
cantly on more challenging datasets. In contrast, training-
based methods maintain stronger and more consistent per-
formance across datasets, such as SAPLMA (77.99%) and
TSV (74.82%). Notably, our method attains the best overall
results while relying solely on (q, a) inputs, with an av-
erage AUROC of 88.19%, which exceeds the second-best
method by 10.20%. Specifically, HCPD improves AUROC
by 5.63% on TriviaQA, by 14.15% on NQ Open, and by
8.66% on CoQA. We attribute these gains to the agent that
fully considers criteria from multiple perspectives, enabling
7

## Page 8

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Table 3. Comparisons with training-based baselines across target models on TriviaQA in terms of AUROC (%).
Source Model
Method
Target Model
LLaMA-3.1-8b
LLaMA-3.1-70b
LLaMA-2-7b
LLaMA-2-13b
Qwen-3-8b
Qwen-2.5-7b
Qwen-2.5-14b
LLaMA-3.1-8b
SAPLMA (Azaria & Mitchell, 2023)
78.51±3.16
77.63±2.63
78.13±2.65
78.25±1.83
71.63±2.15
69.70±2.13
63.77±2.28
HaloScope (Du et al., 2024)
58.19±6.10
86.50±5.56
82.68±2.27
90.88±1.85
54.99±1.89
68.37±3.66
68.33±3.77
TSV (Park et al., 2025)
79.78±3.36
81.29±10.23
82.85±3.90
88.06±4.23
59.89±8.38
77.07±5.82
61.17±7.98
HCPD (Ours)
86.25±1.08
86.87±0.98
90.74±0.52
93.43±1.33
78.89±2.22
88.84±1.02
83.34±0.86
Qwen-3-8b
SAPLMA (Azaria & Mitchell, 2023)
78.22±2.64
78.14±1.64
80.33±1.22
79.50±1.28
78.11±1.09
71.22±1.16
71.73±2.18
HaloScope (Du et al., 2024)
53.89±0.78
57.97±1.21
55.47±9.50
56.73±9.89
58.21±3.99
57.64±1.53
78.65±5.95
TSV (Park et al., 2025)
54.14±3.26
58.85±5.70
60.31±6.89
64.04±8.75
73.42±3.89
57.08±3.75
65.77±4.82
HCPD (Ours)
87.09±1.32
89.73±2.63
90.84±0.76
93.95±1.07
93.69±0.89
89.74±0.77
87.94±1.19
86.25 78.99 83.46 85.47
85.27
85.29
84.44
86.04
84.26
72.84
87.85
90.38
73.83
81.46
82.26
90.07
50
60
70
80
90
100
NQ Open
TriviaQA
SciQ
CoQA
TriviaQA SciQ NQ Open CoQA
Source
Target
Figure 2. Cross-dataset AUROCs of HCPD on LLaMA-3.1-8b.
Table 4. Controlled empirical decomposition of HCPD.
Method
AUROC
Self-evaluation (Kadavath et al., 2022)
56.07±1.92
HCPD (Ours, Pre-RL)
66.54±0.59
HCPD (Ours, Post-RL)
86.25±1.08
fine-grained identification of hallucinated content.
Results on Qwen-3-8b. The results on Qwen-3-8b exhibit
similar characteristics: the training-free and training-based
methods achieve average AUROCs of 53.66%−74.57% and
63.16%−79.47%, respectively. In comparison, our HCPD
again achieves consistent improvements outperforming the
second-best method by 10.15% on average. Taken together
with the results on LLaMA-3.1-8b across datasets, the effec-
tiveness and robustness of HCPD under diverse backbone
models and benchmarks are significantly underscored.
Transferability across target models. A key advantage of
HCPD is its model-agnostic design: it operates in the natural
language space, which is universally shared across LLMs,
thereby achieving superior transferability across diverse tar-
get models. To demonstrate the advantages of HCPD, we
further compare it with other training-based baselines under
a transfer setting where the detector is evaluated on out-
puts generated by target models from different families and
scales. Results in Table 3 show that when applied to unseen
target models, competing baselines such as HaloScope and
TSV often suffer noticeable performance degradation due
to distributional shifts in features of the proxy model. In
contrast, HCPD maintains stable and excellent performance
across heterogeneous target models, which undoubtedly
better meet the needs of practical deployment in open, real-
world settings. More results on the other 3 datasets are
0.2
0.4
0.6
0.8
1.0
Iteration Step / Total Step
30
40
50
60
70
80
90
AUROC
TriviaQA-D
TriviaQA-B
CoQA-D
CoQA-B
Figure 3. Impact of reward design, where “-D” denotes differen-
tiable scoring reward and “-B” denotes binary scoring reward.
Table 5. Impact of sampling size K.
Dataset
1
2
5
10
TriviaQA
85.21±1.22
85.71±0.86
86.25±1.08
86.25±0.98
SciQ
85.06±2.33
85.47±2.12
86.04±2.25
86.14±1.89
NQ Open
86.89±1.60
89.26±2.46
90.38±3.58
90.41±2.46
CoQA
88.60±4.12
89.37±3.34
90.07±2.58
89.85±2.66
Inf. Time (s)
0.2349
0.4675
1.1313
2.2807
shown in Appendix D.1.
Transferability across data distributions. Furthermore,
the proposed Human-like Criteria Probing mechanism en-
ables our method to assess queries with varying distributions
and formats in a flexible manner. To evaluate distributional
transfer, we test HCPD under cross-dataset settings. As
shown in Figure 2, HCPD exhibits amazing generalization
when transferred to TriviaQA, SciQ, and NQ Open, preserv-
ing high performance. We observe a modest degradation
on CoQA, which we attribute to substantial differences in
the underlying QA formulation and interaction mode. Addi-
tional results on Qwen-3-8b are shown in Appendix D.2.
5.3. Ablation Studies
Controlled empirical decomposition. To isolate the contri-
butions of our proposed human-like criteria probing mech-
anism and weakly-supervised alignment training, we con-
ducted a controlled empirical decomposition using standard
Self-evaluation (Kadavath et al., 2022) as a baseline. As
detailed in Table 4, HCPD outperforms the baselines by
Human-like Criteria Probing itself, yielding an AUROC
improvement of 10.47%. This performance is further aug-
mented by our GRPO alignment framework, which delivers
a substantial subsequent AUROC gain of 19.71%.
8

## Page 9

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Table 6. Comparisons with baselines under alternative metrics on
TriviaQA for LLaMA-3.1-8b.
Method
Metric
ROUGE
DeepSeek
LN-Entropy (Malinin & Gales, 2021)
77.70±1.39
52.69±1.19
CCS (Burns et al., 2022)
82.92±1.64
52.52±2.12
SelfCKGPT (Manakul et al., 2023)
73.92±1.26
71.38±0.90
Perplexity (Ren et al., 2023)
85.49±1.13
53.82±2.86
SAPLMA (Azaria & Mitchell, 2023)
87.22±1.13
79.99±0.92
Lexical Similarity (Lin et al., 2024)
82.40±0.90
53.63±1.69
HaloScope (Du et al., 2024)
72.99±4.61
57.98±4.96
TSV (Park et al., 2025)
78.46±6.15
73.31±5.63
HCPD (Ours)
89.17±0.42
80.42±0.39
Impact of reward design. We discussed the advantages of
differentiable scoring over binary scoring in Section 4.4, and
here we further validate this claim empirically. As shown in
Figure 3, replacing the graded reward with a binary “0/1”
signal not only downgrades the evaluation to binary classifi-
cation, but also yields a significant reduction in performance
(from 86.25% to 79.06% on TriviaQA and from 90.07% to
51.75% on CoQA). We attribute this degradation to the lack
of severity information, which makes the agent more prone
to misclassifying samples near the decision threshold.
Impact of sampling size. According to the Corollary 1,
the ranking error decreases as the number of samples K
increases. The results in Table 5 provide quantitative sup-
port for this trend. Larger K yields progressively improved
accuracy and inference stability, but meanwhile incurs a lin-
ear increase in computational cost. To balance performance
and efficiency, we set K = 5 in experiments. Additional
performance and consumption comparison with baselines
are detailed in Appendix D.4. Notably, HCPD matches the
speed of light metrics and outpaces consistency-based meth-
ods. Moreover, its model-agnostic design yields greater
computational savings when evaluating outputs from large
target models (e.g., LLaMA-2-13B, Qwen-2.5-14B, and
LLaMA-3.1-70B).
Beyond the BLEURT metric. It is notable that BLEURT
is an option, not a dependency. We conducte GRPO us-
ing alternative signals (e.g., ROUGE (Lin et al., 2024),
and DeepSeek-V3 as a Judge (Liu et al., 2024)). Results
in Table 6 confirm that as an evaluation metric provided,
HCPD seamlessly aligns its behavior to the reward signal
and achieves strong performance. The detailed implementa-
tion of DeepSeek-V3 is shown in Appendix C.2.
6. Conclusion
In this paper, we propose Human-like Criteria Probing
(HCP), an interpretable paradigm that achieves hallucination
detection under the zero-source setting. We instantiate HCP-
Detector as an LLM-based scoring agent that decomposes
evaluation into context-relevant criteria, assigns adaptive
weights, and produces evidence-grounded analyses along
with an overall score. The agent is further aligned via GRPO
using a dense score-alignment reward derived from weak
proxy supervision, which is based on semantic consistency
signals on QA benchmarks. It employs multi-sampling ag-
gregation at inference time to suppress stochastic variance
and yield stable predictions. Both theoretical analysis and
extensive experiments across diverse datasets and model
architectures demonstrate the effectiveness of our HCPD in
identifying hallucinated content.
Acknowledgments
This work was partially supported by the Joint Funds of
the National Natural Science Foundation of China (Grant
No.U24A20327), Key-Area Research and Development Pro-
gram Guangdong Province 2018B010107001, and TCL Sci-
ence and Technology Innovation Fund, China. Jiahao Yang
is supported by the China Scholarship Council (CSC) under
Grant No. 202506150018.
Impact Statement
This work proposes Human-like Criteria Probing for Hal-
lucination Detection (HCPD), a zero-source framework for
detecting factual and logical inconsistencies in LLM outputs.
By operating solely on the observed query–response pair,
HCPD is intended to improve reliability and transparency
in practical black-box deployment settings, with particular
relevance to safety-critical domains such as healthcare and
education. Furthermore, HCPD provides inherent explain-
ability by outputting the specific criteria and weights used
for each decision, which can aid developers in model de-
bugging and offer end-users transparent justifications. We
believe this research contributes positively to the develop-
ment of more accountable and transparent AI systems.
Author Contributions
Jiahao Yang, Shuhai Zhang, and Hailong Kang contributed
equally to this work. Feng Liu, Qi Chen, and Mingkui Tan
are the corresponding authors. Jiahao Yang and Shuhai
Zhang conceived the main idea and designed the proposed
method. Jiahao Yang and Hailong Kang conducted the
experiments and performed the main empirical analysis. Ji-
ahao Yang, Shuhai Zhang, and Hailong Kang contributed to
manuscript writing, result organization, and paper revision.
Feng Liu, Qi Chen, and Mingkui Tan supervised the project,
provided guidance on methodology and experiments, and
revised the manuscript.
References
Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y.,
Lebr´on, F., and Sanghai, S. Gqa: Training generalized
9

## Page 10

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
multi-query transformer models from multi-head check-
points. In The 2023 Conference on Empirical Methods in
Natural Language Processing, 2023.
Azaria, A. and Mitchell, T. The internal state of an llm
knows when it’s lying. In The 2023 Conference on Em-
pirical Methods in Natural Language Processing, 2023.
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan,
Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical
report. arXiv preprint arXiv:2309.16609, 2023.
Benary, M., Wang, X. D., Schmidt, M., Soll, D., Hilfenhaus,
G., Nassir, M., Sigler, C., Kn¨odler, M., Keller, U., Beule,
D., et al. Leveraging large language models for decision
support in personalized oncology. JAMA Network Open,
6(11):e2343689–e2343689, 2023.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
Advances in Neural Information Processing Systems, 33:
1877–1901, 2020.
Burns, C., Ye, H., Klein, D., and Steinhardt, J. Discovering
latent knowledge in language models without supervision.
In The Eleventh International Conference on Learning
Representations, 2022.
Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z.,
and Ye, J. INSIDE: LLMs’ internal states retain the power
of hallucination detection. In The Twelfth International
Conference on Learning Representations, 2024a.
Chen, Q., Zhang, B., Wang, G., and Wu, Q. Weak-eval-
strong: Evaluating and eliciting lateral thinking of llms
with situation puzzles. Advances in Neural Information
Processing Systems, 37:79642–79665, 2024b.
Chen, X., Song, D., Gui, H., Wang, C., Zhang, N., Jiang,
Y., Huang, F., Lyu, C., Zhang, D., and Chen, H. Factchd:
benchmarking fact-conflicting hallucination detection. In
Proceedings of the Thirty-Third International Joint Con-
ference on Artificial Intelligence, 2024c.
Chen, Y., You, Z., Zhang, S., Li, H., Li, Y., Wang, Y.,
and Tan, M. Core context aware transformers for long
context language modeling. In Forty-second International
Conference on Machine Learning, 2025.
Chiang, C.-W., Lu, Z., Li, Z., and Yin, M. Enhancing
ai-assisted group decision making through llm-powered
devil’s advocate. In Proceedings of the 29th International
Conference on Intelligent User Interfaces, pp. 103–119,
2024.
Csisz´ar, I. I-divergence geometry of probability distributions
and minimization problems. The annals of probability,
pp. 146–158, 1975.
Du, X., Xiao, C., and Li, S. Haloscope: Harnessing un-
labeled llm generations for hallucination detection. Ad-
vances in Neural Information Processing Systems, 2024.
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,
A., et al. The llama 3 herd of models. arXiv e-prints, pp.
arXiv–2407, 2024.
Fang, X., Huang, Z., Tian, Z., Fang, M., Pan, Z., Fang, Q.,
Wen, Z., Pan, H., and Li, D. Zero-resource hallucina-
tion detection for text generation via graph-based contex-
tual knowledge triples modeling. In Proceedings of the
AAAI Conference on Artificial Intelligence, volume 39,
pp. 23868–23877, 2025.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting
hallucinations in large language models using semantic
entropy. Nature, 630(8017):625–630, 2024.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., et al. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783, 2024.
Guo, Y., Liu, J., Li, M., Cheng, D., Tang, X., Sui, D., Liu, Q.,
Chen, X., and Zhao, K. Vtg-llm: Integrating timestamp
knowledge into video llms for enhanced video temporal
grounding. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 39, pp. 3302–3310, 2025.
Hu, X., Ru, D., Qiu, L., Guo, Q., Zhang, T., Xu, Y., Luo,
Y., Liu, P., Zhang, Y., and Zhang, Z. Knowledge-centric
hallucination detection. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language Pro-
cessing, 2024.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H.,
Chen, Q., Peng, W., Feng, X., Qin, B., et al. A survey on
hallucination in large language models: Principles, taxon-
omy, challenges, and open questions. ACM Transactions
on Information Systems, 43(2):1–55, 2025.
Hunter, D. R. Mm algorithms for generalized bradley-terry
models. The annals of statistics, 32(1):384–406, 2004.
Jensen, J. L. W. V.
Sur les fonctions convexes et les
in´egalit´es entre les valeurs moyennes. Acta mathematica,
30(1):175–193, 1906.
Ji, J., Wang, H., Wu, C., Ma, Y., Sun, X., and Ji, R. Jm3d &
jm3d-llm: Elevating 3d representation with joint multi-
modal cues. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2024.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E.,
Bang, Y. J., Madotto, A., and Fung, P. Survey of halluci-
nation in natural language generation. ACM computing
surveys, 55(12):1–38, 2023.
10

## Page 11

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. TriviaQA:
A large scale distantly supervised challenge dataset for
reading comprehension. In Proceedings of the Annual
Meeting of the Association for Computational Linguistics,
pp. 1601–1611, 2017.
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain,
D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma,
N., Tran-Johnson, E., et al. Language models (mostly)
know what they know. arXiv preprint arXiv:2207.05221,
2022.
Kuhn, L., Gal, Y., and Farquhar, S. Semantic uncertainty:
Linguistic invariances for uncertainty estimation in nat-
ural language generation. In The Eleventh International
Conference on Learning Representations, 2023.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M.,
Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and
Petrov, S. Natural questions: A benchmark for question
answering research. Transactions of the Association for
Computational Linguistics, 7:452–466, 2019.
Li, L., Xu, C., Wu, W., Zhao, Y., Zhao, X., and Tao, C.
Zero-resource knowledge-grounded dialogue generation.
Advances in Neural Information Processing Systems, 33:
8475–8485, 2020.
Lin, S., Hilton, J., and Evans, O.
Teaching models
to express their uncertainty in words. arXiv preprint
arXiv:2205.14334, 2022.
Lin, Z., Trivedi, S., and Sun, J. Generating with confidence:
Uncertainty quantification for black-box large language
models. Transactions on Machine Learning Research,
2024. ISSN 2835-8856.
Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao,
C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437, 2024.
Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., Li, P., Liu, Y.,
and Wu, Y. Inference-time scaling for generalist reward
modeling. arXiv preprint arXiv:2504.02495, 2025.
Ma, S., Chen, Q., Wang, X., Zheng, C., Peng, Z., Yin,
M., and Ma, X. Towards human-ai deliberation: Design
and evaluation of llm-empowered deliberative ai for ai-
assisted decision-making. In Proceedings of the 2025 CHI
Conference on Human Factors in Computing Systems, pp.
1–23, 2025.
Malinin, A. and Gales, M. Uncertainty estimation in autore-
gressive structured prediction. In International Confer-
ence on Learning Representations, 2021.
Manakul, P., Liusie, A., and Gales, M. J. F. Selfcheck-
gpt: Zero-resource black-box hallucination detection for
generative large language models, 2023.
Neumann, A. T., Yin, Y., Sowe, S., Decker, S., and Jarke, M.
An llm-driven chatbot in higher education for databases
and information systems. IEEE Transactions on Educa-
tion, 2024.
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A.,
et al. Training language models to follow instructions
with human feedback. Advances in neural information
processing systems, 35:27730–27744, 2022.
Park, S., Du, X., Yeh, M.-H., Wang, H., and Li, Y. Steer
llm latents for hallucination detection. In Forty-second
International Conference on Machine Learning, 2025.
Qian, S., Chen, W., Bai, M., Zhou, X., Tu, Z., and Li,
L. E. Affordancellm: Grounding affordance from vi-
sion language models. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR) Workshops, pp. 7587–7597, June 2024.
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D.,
Ermon, S., and Finn, C. Direct preference optimiza-
tion: Your language model is secretly a reward model.
Advances in neural information processing systems, 36:
53728–53741, 2023.
Reddy, S., Chen, D., and Manning, C. D. Coqa: A conversa-
tional question answering challenge. Transactions of the
Association for Computational Linguistics, 7:249–266,
2019.
Ren, J., Luo, J., Zhao, Y., Krishna, K., Saleh, M., Lakshmi-
narayanan, B., and Liu, P. J. Out-of-distribution detection
and selective generation for conditional language models.
In The Eleventh International Conference on Learning
Representations, 2023.
Ross, S. M. Introduction to probability and statistics for
engineers and scientists. Academic press, 2020.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347, 2017.
Sellam, T., Das, D., and Parikh, A. BLEURT: Learning
robust metrics for text generation. In Proceedings of the
Annual Meeting of the Association for Computational
Linguistics, pp. 7881–7892, 2020.
Semnani, S., Yao, V., Zhang, H., and Lam, M. Wikichat:
Stopping the hallucination of large language model chat-
bots by few-shot grounding on wikipedia. In Findings
of the association for computational linguistics: EMNLP
2023, 2023.
11

## Page 12

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang,
H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Push-
ing the limits of mathematical reasoning in open language
models. arXiv preprint arXiv:2402.03300, 2024.
Shazeer, N.
Glu variants improve transformer.
arXiv
preprint arXiv:2002.05202, 2020.
Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
Q., Hinton, G., and Dean, J. Outrageously large neural
networks: The sparsely-gated mixture-of-experts layer.
arXiv preprint arXiv:1701.06538, 2017.
Team, Q.
Qwen2 technical report.
arXiv preprint
arXiv:2407.10671, 2, 2024.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models. arXiv preprint arXiv:2302.13971, 2023a.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models. arXiv preprint arXiv:2307.09288,
2023b.
Vazhentsev, A., Fadeeva, E., Xing, R., Kuzmin, G.,
Lazichny, I., Panchenko, A., Nakov, P., Baldwin, T.,
Panov, M., and Shelmanov, A. Unconditional truthful-
ness: Learning unconditional uncertainty of large lan-
guage models. In Proceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pp.
35661–35682, 2025.
Vrdoljak, J., Boban, Z., Vilovi´c, M., Kumri´c, M., and Boˇzi´c,
J. A review of large language models in medical educa-
tion, clinical decision support, and healthcare administra-
tion. In Healthcare, volume 13, pp. 603. MDPI, 2025.
Wang, B., Yue, X., and Sun, H. Can chatgpt defend its
belief in truth? evaluating llm reasoning via debate. In
The 2023 Conference on Empirical Methods in Natural
Language Processing, 2023.
Wang, Q., Wang, Z., Su, Y., Tong, H., and Song, Y. Re-
thinking the bounds of llm reasoning: Are multi-agent
discussions the key? In Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 6106–6131, 2024.
Welbl, J., Liu, N. F., and Gardner, M.
Crowdsourcing
multiple choice science questions. In Proceedings of
the Workshop on Noisy User-generated Text, pp. 94–106,
2017.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu,
B., Gao, C., Huang, C., Lv, C., et al. Qwen2.5 technical
report. arXiv preprint arXiv:2412.15115, 2024a.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report. arXiv preprint arXiv:2505.09388, 2025a.
Yang, B., Al Mamun, M. A., Zhang, J. M., and Uddin,
G.
Hallucination detection in large language models
with metamorphic relations. Proceedings of the ACM on
Software Engineering, 2(FSE):425–445, 2025b.
Yang, Y., Zhou, T., Li, K., Tao, D., Li, L., Shen, L., He, X.,
Jiang, J., and Shi, Y. Embodied multi-modal agent trained
by an llm from a parallel textworld. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 26275–26285, June 2024b.
Yu, Y., Yao, Z., Li, H., Deng, Z., Jiang, Y., Cao, Y., Chen,
Z., Suchow, J., Cui, Z., Liu, R., et al. Fincon: A syn-
thesized llm multi-agent system with conceptual verbal
reinforcement for enhanced financial decision making.
Advances in Neural Information Processing Systems, 37:
137010–137045, 2024.
Zhang, B. and Sennrich, R. Root mean square layer nor-
malization. Advances in neural information processing
systems, 32, 2019.
Zhang, S., You, Z., Chen, Y., Wen, Z., Wang, Q., Qiu,
Z., Li, Y., and Tan, M. Curse of high dimensionality
issue in transformer for long context modeling. In Forty-
second International Conference on Machine Learning,
pp. 76600–76624. PMLR, 2025.
Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi,
Y. Bertscore: Evaluating text generation with bert. In
International Conference on Learning Representations,
2019.
Zhang, T., Qiu, L., Guo, Q., Deng, C., Zhang, Y., Zhang, Z.,
Zhou, C., Wang, X., and Fu, L. Enhancing uncertainty-
based hallucination detection with stronger focus. arXiv
preprint arXiv:2311.13230, 2023.
Zhao, C., Song, Y., Chen, J., Rong, K., Feng, H., Zhang,
G., Ji, S., Wang, J., Ding, E., and Sun, Y. Octopus: A
multi-modal llm with parallel recognition and sequential
understanding. Advances in Neural Information Process-
ing Systems, 37:90009–90029, 2024.
Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z.,
Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., et al.
Judging llm-as-a-judge with mt-bench and chatbot arena.
In Proceedings of the 37th International Conference on
Neural Information Processing Systems, 2023.
Zhu, Y., Yuan, H., Wang, S., Liu, J., Liu, W., Deng, C.,
Chen, H., Liu, Z., Dou, Z., and Wen, J.-R. Large lan-
guage models for information retrieval: A survey. ACM
Transactions on Information Systems, 44(1):1–54, 2025.
12

## Page 13

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
APPENDIX
Contents
A Theoretical Analysis
14
A.1
Setup and Notations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
A.2
Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
A.3
Proof of Corollary 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
B
More Related Work
17
C More Details for Experiment Settings
18
C.1
More Details on Datasets
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18
C.2
More Details on Implementation
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18
D More Experiment Results
23
D.1
More Results on Transferability across Target Models
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
D.2
More Results on Transferability across Data Distributions
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
D.3
Impact of Generation Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
D.4
Performance and Consumption Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
D.5
Generalization to Long-form Generation Task
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
E
Future Directions
25
F
Examples
26
13

## Page 14

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
A. Theoretical Analysis
A.1. Setup and Notations
Let x = (q, a) be an input of hallucination detection under the zero-source constraint, where a can be generated by
any source LLM. The scoring agent fθ produces a structured and interpretable evaluation whose predicted score is an
integer random variable sp(x) ∈{1, . . . , 10} in the well-formed regime. During training, each query q has a ground-truth
reference answer ˆa within dataset, and an auxiliary LLM generates candidate answers {a(n)}N
n=1. Weak supervision labels
s(n)
l
∈{1, . . . , 10} are computed from a semantic consistency metric sim(ˆa, a(n)) (based on BLEURT (Sellam et al., 2020)).
The agent is optimized with GRPO using the score-alignment reward given by Eqn. (5). During inference, we invoke the
agent K times and aggregate ¯s according to Eqn. (6).
A.2. Proof of Theorem 1
Theorem 1 (Expectation alignment under training) Let x denote an input and sl(x) its corresponding weak label. Consider
the ℓ1-risk of predicting score s:
Rx(s) ≜|s −sl(x)|,
whose minimizer is s = sl(x). Let Y ∼fθ(· | x) be a stochastic well-formed generation of the agent, and let Sθ(x) =
sp(x, Y ) denote the parsed score. Define the conditional mean score:
µθ(x) ≜E

Sθ(x) | x

.
Then we have:
Ex

|µθ(x) −sl(x)|

≤J ′(θ),
where the expectation is taken over the training distribution of x, and J ′(θ) is affine-equivalent to the GRPO objective
J (θ) defined in Eqn. (1).
Proof. Given a fixed input x, since sl(x) is deterministic, we have
|µθ(x) −sl(x)| = |E

Sθ(x) | x

−sl(x)| = |E

Sθ(x) −sl(x) | x

|.
By Jensen’s inequality (Jensen, 1906),
|E

Sθ(x) −sl(x) | x

| ≤E

|Sθ(x) −sl(x)| | x

.
Taking expectation over the training distribution of x yields
Ex

|µθ(x) −sl(x)|

≤ExE

|Sθ(x) −sl(x)| | x

= Ex,Y

|sp(x, Y ) −sl(x)|

.
(9)
By the score-alignment reward in Eqn. (5)), minimizing the KL-regularized GRPO objective in Eqn. (1) is equivalent up to
affine constants to minimizing the nonnegative objective
J ′(θ) = ExEY ∼fθ(·|x)

|sp(x, Y ) −sl(x)|

+ λ · ExDKL
 fθ(·|x)||f0(·|x)

= Ex,Y

|sp(x, Y ) −sl(x)|

+ λ · ExDKL
 fθ(·|x)||f0(·|x)

.
(10)
where λ ∝β. Since the non-negativity of the KL regularizer, we have
Ex,Y

|sp(x, Y ) −sl(x)|

≤J ′(θ).
Combining the two inequalities Eqn. (9) and Eqn. (10) completes the proof.
14

## Page 15

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
A.3. Proof of Corollary 1
Proposition 1 (Multi-sampling concentration) Fix an inference input x. Suppose the K well-formed generations
Y1, . . . , YK
i.i.d.
∼fθ(· | x). For each k ∈{1, . . . , K}, define the parsed score
S(k)
θ (x) ≜sp(x, Yk) ∈{1, . . . , 10},
and let ¯s(x) ≜1
K
PK
k=1 S(k)
θ (x) denote the aggregated score in Eqn. (6). Then for any bias threshold u > 0,
P
 |¯s(x) −E[Sθ(x) | x]| ≥u | x

≤2 exp

−
2Ku2
(10 −1)2

.
Proof. Conditional on the input x, the random variables {S(k)
θ (x)}K
k=1 are i.i.d and each are bounded in [1, 10]. Let
µθ(x) ≜E

Sθ(x) | x

. By Hoeffding’s inequality for bounded i.i.d variables, we have
P
 |¯s(x) −µθ(x)| ≥u | x

≤2 exp

−
2Ku2
(10 −1)2

.
which is exactly the claim.
Corollary 1 (Ranking error decomposition) Let x denote an input and sl(x) its corresponding weak label. Assume the
proxy bias is uniformly bounded:
|ϵ(x)| ≤bmax,
∀x.
Let Y1, . . . , YK
i.i.d.
∼fθ(· | x) be well-formed generations, and define S(k)
θ (x) and ¯s(x) as in Proposition 1. Let x+ and x−
denote independent draws from the distributions of true (non-hallucinated) and hallucinated inputs, respectively. Define the
ranking error probability:
Erank ≜P
 ¯s(x+) ≤¯s(x−)

.
Then for any ∆> bmax,
Erank ≤P
 g(s⋆(x+)) −g(s⋆(x−)) ≤2∆

|
{z
}
intrinsic separability
+
4 J ′(θ)
∆−bmax
|
{z
}
training alignment
+ 4 exp

−
2K
(10 −1)2
∆−bmax
2
2
|
{z
}
multi-sampling concentration
,
where J ′(θ) is affine-equivalent to the GRPO objective J (θ) defined in Eqn. (1).
Proof. Let ∆> bmax and define δ ≜∆−bmax
2
> 0. Consider the event
A ≜{g(s⋆(x+)) −g(s⋆(x−)) ≤2∆}.
On the complement Ac, we have g(s⋆(x+)) −g(s⋆(x−)) ≥2∆.
If in addition scores are close to their truthfulness scores |¯s(x) −g(s⋆(x))| ≤∆, then
¯s(x+) ≥g(s⋆(x+)) −∆≥g(s⋆(x−)) + ∆≥¯s(x−)
which means order-preserving. Therefore, the ranking error event {¯s(x+) ≤¯s(x−)} is contained in
A ∪{|¯s(x+) −g(s⋆(x+))| > ∆} ∪{|¯s(x−) −g(s⋆(x−))| > ∆}
Taking probability and using the union bound gives
Erank ≤P(A) + P(|¯s(x+) −g(s⋆(x+))| > ∆) + P(|¯s(x−) −g(s⋆(x−))| > ∆).
(11)
For a fixed input x, the proxy bias satisfies |ϵ(x)| ≤bmax, so
|¯s(x) −g(s⋆(x))| ≤|¯s(x) −sl(x)| + |sl(x) −g(s⋆(x))| ≤|¯s(x) −sl(x)| + bmax.
15

## Page 16

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Thus |¯s(x) −g(s⋆(x))| > ∆implies |¯s(x) −sl(x)| ≥∆−bmax = 2δ.
Meanwhile, we have
|¯s(x) −sl(x)| ≤|¯s(x) −µθ(x)| + |µθ(x) −sl(x)|,
hence |¯s(x) −sl(x)| > 2δ implies
|¯s(x) −µθ(x)| > δ or |µθ(x) −sl(x)| > δ,
so again by union bound, we have
P(|¯s(x) −g(s⋆(x))| > ∆) ≤P(|¯s(x) −µθ(x)| > δ) + P(|µθ(x) −sl(x)| > δ).
(12)
Conditional on x, Proposition 1 yields
P
 |¯s(x) −µθ(x)| ≥δ | x

≤2 exp

−
2Kδ2
(10 −1)2

.
Taking expectation over x preserves the same upper bound:
P
 |¯s(x) −µθ(x)| ≥δ

≤2 exp

−
2Kδ2
(10 −1)2

.
(13)
Moreover, by Markov’s inequality (Ross, 2020),
P(|µθ(x) −sl(x)| > δ) ≤Ex

|µθ(x) −sl(x)|

δ
.
Using Theorem 1, we have Ex

|µθ(x) −sl(x)|

≤J ′(θ), hence
P(|µθ(x) −sl(x)| > δ) ≤J ′(θ)
δ
.
(14)
By combining Eqn. (12)-(14), we have for any ∆> bmax and δ = ∆−bmax
2
,
P(|¯s(x) −g(s⋆(x))| > ∆) ≤J ′(θ)
δ
+ 2 exp

−
2Kδ2
(10 −1)2

.
(15)
Apply Eqn. (15) to x+ and x−in Eqn. (11), we have
Erank ≤P(A) + P(|¯s(x+) −g(s⋆(x+))| > ∆) + P(|¯s(x−) −g(s⋆(x−))| > ∆)
= P(A) + 2
J ′(θ)
δ
+ 2 exp
 −
2Kδ2
(10 −1)2

= P
 g(s⋆(x+)) −g(s⋆(x−)) ≤2∆

+
4 J ′(θ)
∆−bmax
+ 4 exp

−
2K
(10 −1)2
∆−bmax
2
2
,
(16)
by recalling δ = ∆−bmax
2
, which is exactly the desired decomposition.
16

## Page 17

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
B. More Related Work
Large Language Models (LLMs). LLMs have become central to modern natural language processing, demonstrating strong
capabilities in reasoning (Brown et al., 2020; Wang et al., 2023; 2024; Chen et al., 2025; Zhang et al., 2025), knowledge
grounding (Li et al., 2020; Qian et al., 2024; Guo et al., 2025), and multi-modal understanding (Zhao et al., 2024; Yang
et al., 2024b; Ji et al., 2024). Among open-source ecosystems, two influential model families are LLaMA (Touvron et al.,
2023a;b; Dubey et al., 2024; Grattafiori et al., 2024) and Qwen (Bai et al., 2023; Team, 2024; Yang et al., 2024a; 2025a).
The LLaMA series, developed by Meta, has been instrumental in advancing open-weight LLMs. LLaMA (Touvron et al.,
2023a) demonstrated that competitive models can be trained on publicly available corpora, incorporating architectural
components such as RMSNorm (Zhang & Sennrich, 2019), SwiGLU activations (Shazeer, 2020), and Rotary Positional
Embeddings (RoPE). LLaMA 2 (Touvron et al., 2023b) scaled pre-training data, introduced RLHF-tuned conversational
variants, and adopted a more permissive license. LLaMA 3 (Dubey et al., 2024) further scaled model capacity (up to 70B
parameters) and multilingual data, together with an improved tokenizer and stronger instruction-following performance. The
most recent iteration, LLaMA 3.1 (Grattafiori et al., 2024), extends the context window to 128k tokens via enhanced RoPE
scaling, enlarges the pre-training corpus to over 15 trillion tokens, and integrates advanced post-training techniques such
as large-scale Direct Preference Optimization (DPO) (Rafailov et al., 2023) and emergent tool-use capabilities, achieving
state-of-the-art results among open models and approaching parity with leading proprietary systems.
The Qwen series, first released in 2023 (Bai et al., 2023), emphasizes scalability and practical deployment, offering models
across a wide range of parameter sizes. Qwen-1.55 and Qwen-2 (Team, 2024) extend context length to 128k tokens across
the family and incorporate Grouped Query Attention (GQA) (Ainslie et al., 2023) to improve inference efficiency, leading
to substantial performance gains. Qwen2.5 (Yang et al., 2024a) further introduces domain-specialized variants for coding
and mathematics, forming a flexible generalist–specialist design. The most recent Qwen3 (Yang et al., 2025a) proposes a
Hybrid-Reasoning paradigm that dynamically switches between a deliberate Thinking mode for complex reasoning and a
lightweight Non-Thinking mode for rapid responses, enabling explicit trade-offs between computational cost and accuracy.
Its large-scale Mixture-of-Experts (MoE) (Shazeer et al., 2017) variants achieve competitive state-of-the-art performance,
while smaller dense models retain strong parameter efficiency.
Reinforcement Learning. Reinforcement learning (RL) has demonstrated effectiveness in improving the reasoning ability
of LLMs by aligning their behaviors with human preferences or task-specific objectives (Ouyang et al., 2022).
Similar to conventional machine learning, Supervised Fine-Tuning (SFT) trains LLMs by maximizing the log-likelihood on
high-quality, task-specific demonstration data. While SFT effectively transfers knowledge, it is inherently constrained by
the quality and coverage of the demonstrations and cannot directly optimize non-differentiable or long-horizon objectives.
Serving as the subsequent step, Proximal Policy Optimization (PPO) (Schulman et al., 2017) introduces a reward model
trained from human preference comparisons and maximizes this reward while constraining policy updates to maintain
stability. This PPO-based Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022) enables models like
InstructGPT (Ouyang et al., 2022) and ChatGPT6, substantially improving instruction-following and helpfulness. However,
it is computationally intensive and operationally complex, with the requirement of joint training and balancing of a policy
model, a reward model, and a value function. Direct Preference Optimization (DPO) (Rafailov et al., 2023) emerged
as a simplified and more stable alternative, bypassing explicit reward modeling by deriving a closed-form objective that
directly optimizes the policy using pairwise preference data. By reformulating the RL objective as a supervised loss, DPO
significantly reduces training complexity compared to PPO. Nonetheless, its reliance on the Bradley–Terry (Hunter, 2004)
preference model makes it less amenable to scenarios involving dense, scalar reward signals, such as the score-alignment
supervision adopted in this work. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) is a recent advancement
designed to improve stability and efficiency in LLM alignment. It operates by sampling a group of candidate outputs for
each prompt, assigning rewards for each, and optimizing the policy by increasing the relative likelihood of higher-reward
outputs within the group. This group-relative ranking objective provides a stable and efficient learning signal, making GRPO
particularly well-suited for weakly supervised, scalar reward settings.
5https://qwenlm.github.io/blog/qwen1.5/
6https://openai.com/blog/chatgpt
17

## Page 18

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
C. More Details for Experiment Settings
C.1. More Details on Datasets
TriviaQA (Joshi et al., 2017) is a large-scale reading comprehension benchmark comprising 95, 956 question-answer
pairs, each supplemented with an average of approximately 6 supporting evidence documents, yielding 662, 659 associated
documents in total sourced from Wikipedia and the Web. Since questions are authored independently of the subsequent
evidence retrieval, the dataset exhibits substantial lexical and syntactic divergence between queries and supporting evidence,
and frequently requires multi-sentence reasoning to answer accurately.
SciQ (Welbl et al., 2017) is a science question answering benchmark designed to evaluate domain knowledge and reasoning.
It comprises 13, 679 manually collected scientific exam questions spanning physics, chemistry, biology, and related subjects,
split into 11, 679 training, 1, 000 validation, and 1, 000 test instances. Each question is multiple-choice with 4 answer
options, and most are accompanied by an additional paragraph that supports the correct answer. SciQ also provides a
direct-answer variant in which distractors are removed to enable reading-comprehension-style evaluation.
NQ Open (Kwiatkowski et al., 2019) is an open-domain question answering benchmark derived from Natural Questions
corpus, comprising anonymized real-world Google queries paired with brief answers, all of which can be answered through
content from the English Wikipedia. The dataset contains 91, 535 question–answer pairs, including 87, 925 training instances
and 3, 610 validation instances. To ensure concise and standardized responses, examples with answer lengths exceeding five
words are discarded, yielding a high-quality benchmark for open-domain QA evaluation.
CoQA (Reddy et al., 2019) is a large-scale benchmark for conversational question answering, which aims to measure a
model’s ability to understand a given passage and answer a series of interdependent questions that appear in a conversation.
The dataset contains over 127, 000 questions with answers collected from more than 8, 000 conversations. Each conversation
is collected by pairing two crowd workers to interact over a passage through multi-turn question and answer exchanges.
CoQA captures challenging phenomena that are not present in standard reading comprehension benchmarks, including
coreference and pragmatic reasoning.
Wikipedia7 is a large-scale multilingual dataset comprising cleaned articles from Wikipedia dumps across 320+ languages.
It includes 61.6 million+ rows of text, with each entry containing an article’s ID, URL, title, and markdown-stripped content.
The dataset supports tasks like text generation and masked-language modeling, with one subset per language and a single
training split, making it a foundational resource for multilingual NLP research.
C.2. More Details on Implementation
C.2.1. IMPLEMENTATION DETAILS ON BASELINES.
We implement baselines spanning 4 paradigms, largely following the official implementations when available.
(1) Logit-based methods. We implement Perplexity (Ren et al., 2023) using the public codebase8, which adopts sequence-
level perplexity as the detection score; and LN-Entropy (Malinin & Gales, 2021) based on the same codebase8, which
computes entropy with length normalization; and Semantic Entropy (Kuhn et al., 2023) using the official repository9, which
clusters semantically equivalent generations before entropy estimation.
(2) Consistency-based methods. We implement Lexical Similarity (Lin et al., 2024) following the codebase8, using ROUGE-
based similarity to measure agreement across multiple sampled responses; and SelfCheckGPT (Manakul et al., 2023) using
the same codebase8, which aggregates multiple similarity metrics (e.g., BERTScore) for self-consistency evaluation; and
EigenScore (Chen et al., 2024a) following the codebase8, which evaluates semantic consistency via eigenvalue statistics of
the response covariance matrix in embedding space.
(3) Verbalized-confidence methods. We implement Self-evaluation (Kadavath et al., 2022) following the codebase9, which
prompts the model to estimate the probability that its answer is correct.
(4) Internal-state methods. We implement CCS (Burns et al., 2022) using the official codebase10, which extracts latent
7https://dumps.wikimedia.org
8https://github.com/D2I-ai/eigenscore
9https://github.com/jlko/semantic_uncertainty
10https://github.com/collin-burns/discovering_latent_knowledge
18

## Page 19

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
knowledge signals from model activations; and SAPLMA (Azaria & Mitchell, 2023) with the provided repository11, which
trains auxiliary classifiers on hidden representations; and HaloScope (Du et al., 2024) using the official implementation12,
which applies SVD to identify hallucination-relevant subspaces; and TSV (Park et al., 2025) following the codebase13, which
learns steering vectors to reshape latent features for improved separability.
C.2.2. IMPLEMENTATION DETAILS ON OUR METHOD.
In our main experiments, we instantiate the agent fθ with Qwen-2.5-7b (Yang et al., 2024a) to assess the generated answer
a conditioned on the query q. To elicit expert-like behavior and implement Human-like Criteria Probing, we follow the
Generalist Reward Model design in SPCT (Liu et al., 2025) and format each input pair (q, a) using the template below:
    Hallucination detection prompt template (PART I)
You are a skilled little expert at detecting hallucinations in AI responses. You should 
evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last round is the User's query) and multiple 
responses from the Assistant, you need to refer to the [General Evaluation Criteria] to 
score the responses for **hallucination risk**. Based on the general evaluation criteria, 
state potential other specific criteria to the query, the weights of different criteria, and 
then provide an overall comprehensive score upon them.
Each score is an integer between 1 and 10, where a **higher score indicates lower 
hallucination risk and higher reliability** (i.e., the response is more factually grounded, 
logically sound, and appropriate). For example:
- Score 1: Severe hallucination; largely fabricated or misleading
- Score 6: Moderate reliability; contains noticeable but non-critical hallucinations
- Score 10: No hallucination; fully accurate, consistent, and appropriate
#### General Evaluation Criteria ####
1. Factual Verification
   - Extract all factual claims and validate against scientific consensus, historical records, 
or authoritative sources.
   - Fully Grounded (9–10 points): All claims are verifiable, accurate, and aligned with 
authoritative knowledge.
   - Mostly Accurate (6–8 points): Minor inaccuracies (e.g., slight statistic deviation <5%, 
or non-critical outdated info), but core facts are correct.
   - Partially Grounded (3–5 points): Noticeable factual errors (e.g., wrong person/event, 
>5% statistical error, or mention of fictional entities as real).
   - Severely Fabricated (1–2 points): Multiple false claims, completely invented 
narratives, or reliance on non-existent sources.
2. Logical Consistency
   - Map argument dependencies and flag contradictions or reasoning flaws.
   - Fully Consistent (9–10 points): Reasoning is sound, premises support conclusions, no 
contradictions.
   - Generally Consistent (6–8 points): Minor logical gaps or weak inferences, but overall 
argument holds.
   - Partially Consistent (3–5 points): Clear issues such as self-contradiction, circular 
reasoning, or unsupported conclusions.
   - Critically Inconsistent (1–2 points): Multiple contradictions, paradoxical statements, 
or reasoning that collapses under basic scrutiny.
3. Semantic Accuracy
   - Identify term misuse, ambiguity, or information loss that distorts meaning.
The prompt specifies a General Evaluation Criteria set C that includes “Factual Verification”, “Logical Consistency”,
“Semantic Accuracy”, “Social Fairness”, and “Timeline Verification”, together with their associated scoring guidelines. It
11https://github.com/ivanrozhd/anlp-project
12https://github.com/deeplearning-wisc/haloscope
13https://github.com/deeplearning-wisc/tsv
19

## Page 20

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
further enforces a strictly structured and interpretable output schema to facilitate downstream aggregation.
Notably, we employ distinct sampling strategies (temperature T, nucleus sampling parameter top-p, and repetition penalty
RP) for the training and test phases to serve different objectives. In particular, we set T = top-p = RP = 1.0 to endow fθ
with greater flexibility to explore the evaluation space more broadly and learn robust assessment behaviors; and we adopt
a lower-entropy (more deterministic) setting with T = 0.9, top-p = 0.9, RP = 1.1 to obtain higher-confidence and more
stable evaluations in inference.
    Hallucination detection prompt template (PART II)
 - Precise & Clear (9–10 points): Terms used correctly; meaning is unambiguous and 
fully preserved.
   - Mostly Clear (6–8 points): Occasional vague phrasing or mild over-abstraction, but 
core message remains intact.
   - Partially Accurate (3–5 points): Misused technical terms, misleading analogies, or 
significant information loss affecting understanding.
   - Highly Distorted (1–2 points): Language so ambiguous, abstract, or erroneous that it 
conveys false or incomprehensible meaning.
4. Social Fairness
   - Detect stereotypes, discrimination, or culturally insensitive content.
   - Fully Inclusive (9–10 points): Language is respectful, neutral, culturally aware, and 
free of bias.
   - Generally Appropriate (6–8 points): Minor generalizations or slightly outdated 
phrasing, but not offensive.
   - Partially Biased (3–5 points): Clear stereotypes, gender/region/culture-based 
assumptions, or insensitive framing.
   - Highly Offensive (1–2 points): Promotes discrimination, harmful generalizations, or 
dehumanizing language.
5. Timeline Verification
   - Annotate time-related claims and validate temporal logic and recency.
   - Temporally Sound (9–10 points): All time references are accurate, logically ordered, 
and appropriately current.
   - Mostly Accurate (6–8 points): Minor recency issues (e.g., citing 6-year-old data in a 
stable domain) or slightly vague timeline.
   - Partially Accurate (3–5 points): Confused chronology, treating future speculation as 
fact, or using obsolete info (>5 years) in critical contexts.
   - Severely Anachronistic (1–2 points): Asserts future events as past, creates impossible 
timelines, or mixes historical eras illogically (e.g., “Einstein used smartphones”).
#### Conversation Context ####
{conversation context & query}
#### Responses to be Scored ####
[The Begin of Response i]
{the i-th response}
[The End of Response i]
#### Output Format Requirements ####
Specific Criteria: <Other criteria and weights>
Analysis: <Comparison based on criteria>
Scores: \boxed{x, x} (for multiple responses)
For reward-based alignment with GRPO (Shao et al., 2024), we adopt the Open-R1 implementation14 and conduct training
on 2 NVIDIA A800 GPUs. We use a per-device batch size of 10. Through the optimization parameters in RL, the learning
rate is set to lr = 2 × 10−4 on LLaMA-3.1-8b for all 4 datasets, while lr = 1 × 10−4 on Qwen-3-8b. For the coefficient β
that controls the strength of DKL, we set β = 0.05 on TriviaQA with LLaMA-3.1-8b and Qwen-3-8b as well as SciQ with
Qwen-3-8b, while β = 0.04 for all others. All reported results are averaged over 5 independent random splits.
14https://github.com/huggingface/open-r1
20

## Page 21

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
C.2.3. TRAINING DATA CONSTRUCTION AND LABELING.
Following Kuhn et al. (2023), we construct hallucination detection datasets from standard question-answer (QA) benchmarks.
Specifically, for traditional QA datasets that do not require additional reference context (e.g., TriviaQA (Joshi et al., 2017),
SciQ (Welbl et al., 2017), and NQ Open (Kwiatkowski et al., 2019)), we generate candidate answers for each question q
using the following prompt template:
    Prompt template for closed-source QA dataset
Answer the question concisely:
Q: {question}
A:
For open-source QA datasets with auxiliary context (e.g., CoQA (Reddy et al., 2019)), we adopt the following prompt:
    Prompt template for open-source QA dataset
Based on the following context, answer the question concisely:
Context: {context}
Q: {question}
A:
We also employ different sampling strategies for the training and test splits. In the training split, we aim to generate a set
of responses {a(n)}N
n=1 that spans a broad spectrum from factually correct to clearly hallucinated, providing fine-grained
supervision for reinforcement learning. We therefore adopt a higher-entropy sampling strategy with T = 0.5, top-p =
1.0, RP = 1.0, and set N = 9, yielding 10 answers per query in total when including the ground-truth reference ˆa for
training. In the test split, we generate a single response per query using greedy decoding with 5 beam search, matching the
evaluation protocol in Kuhn et al. (2023). Notably, although this procedure may appear superficially similar to the agent
setting, it differs fundamentally in both its target (the generator rather than the evaluation agent) and its purpose (producing
diverse candidate answers rather than learning reliable scoring behavior).
For labeling, we assign the reference answer ˆa the maximum score sl = 10 and adopt BLEURT (Sellam et al., 2020) to
score the generated answers. The resulting graded score is derived from Eqn. (4). For binary evaluation on the test set, we
further threshold the BLEURT-based similarity: answers with the similarity above 0.5 are labeled as True, and those below
as Hallucinated. It should be emphasized that the reference answer ˆa is only used offline to construct labels during training
and never accessed at inference; at test time, the detector takes only (q, a) where a can be produced by any LLM.
In ablation experiments that using alternative signals, we adopt Deepseek-V3 (Liu et al., 2024) to evaluate the truthfulness
of generated content, following the LLM-as-a-judge (Zheng et al., 2023) paradigm. Specifically, we assess the truthfulness
of LLM-generated responses by verifying their semantic equivalence to the provided gold standard answers. The input
prompt for this evaluation on QA datasets is as:
Prompt template for QA dataset
You are tasked with evaluating whether a generated answer is truthful compared to
a provided list of gold standard answers. Use a 10-point scoring scale (1-2: Compl-
etely inconsistent/hallucinated, 3-4: Mostly inconsistent with minor, 5-6: Partially
consistent, mixed quality, 7-8: Mostly consistent with minor errors, 9-10: Complet-
ely consistent, accurate, comprehensive) to score the generated content. Provide a
lear justification for your scoring.
Response Format:
- Score: [1/2/3/4/5/6/7/8/9/10]
- Justification: [Explain briefly why the answer is correct or incorrect.]
Question: {question}
Gold Standard Answers: {reference_answers}
Generated Answer: {generated_answer}
And the input prompt Wikipedia dataset is as:
21

## Page 22

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Prompt template for Wikipedia dataset
You are tasked with evaluating whether a generated Wikipedia passage is consistent
with the provided real Wikipedia passage. Use a 10-point scoring scale (1-2: Compl-
etely inconsistent/hallucinated, 3-4: Mostly inconsistent with minor, 5-6: Partially c-
onsistent, mixed quality, 7-8: Mostly consistent with minor errors, 9-10: Completely
consistent, accurate, comprehensive) to score the generated content. Provide a clear
justification for your scoring.
Response Format:
- Score: [1/2/3/4/5/6/7/8/9/10]
- Justification: [Explain briefly why the answer is correct or incorrect.]
Title: {question}
Real Wikipedia passage: {reference_answers}
Generated Wikipedia passage: {generated_answer}
C.2.4. PSEUDO CODE OF HCPD
Algorithm 1 Reward-based Alignment Training of HCPD
Input: Dataset D = {(qi, {(a(n)
i
, s(n)
i
)})}, group size G,
general criteria set C, initial agent fθ ←f0, η;
Form detection input x(n)
i
←(qi, a(n)
i
);
for x(n)
i
in D do
Sample a group of evaluations {Yg}G
g=1 ∼fθ(· | x; C);
for g = 1, 2, . . . , G do
Parse predicted score s(g)
p
←sp(x, Yg) ∈{1, . . . , 10};
Compute reward rg via Eqn. (5);
Compute advantages Ag ←rg −1
G
PG
j=1 rj;
end for
Compute GRPO objective via Eqn. (1);
θ ←θ + η∇θJx(θ).
end for
Output: f ∗
θ
Algorithm 2 Detection via Multi-sampling Aggregation
Input: Scoring agent f ∗
θ , general criteria set C,
input x = (q, a), K;
Sample K evaluations {Yk}K
k=1 ∼f ∗
θ (· | x; C);
for k = 1, 2, . . . , K do
Parse score s(k)
p
←sp(x, Yk) ∈{1, . . . , 10};
end for
Aggregate scores via Eqn. (6);
Output: Truthfulness Score ¯s
22

## Page 23

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
D. More Experiment Results
D.1. More Results on Transferability across Target Models
To further demonstrate the advantages of HCPD’s model-agnostic design, we provide cross-target evaluation on the remaining
3 datasets. Results in Table 7-9 show that our method maintains consistently strong performance across heterogeneous
target models. Notably, when faced with earlier-generation models (e.g., LLaMA-2 and Qwen-2.5), detection performance
further improves, likely because such models produce less fluent and less realistic outputs, making hallucinations easier to
distinguish. Overall, these results indicate that HCPD is undoubtedly well-suited for detecting hallucinations from complex
input sources, which is common in real-world deployment.
Table 7. Comparisons with training-based baselines across target models on SciQ in terms of AUROC (%).
Source Model
Method
Target Model
LLaMA-3.1-8b
LLaMA-2-7b
LLaMA-2-13b
Qwen-3-8b
Qwen-2.5-7b
Qwen-2.5-14b
LLaMA-3.1-8b
SAPLMA (Azaria & Mitchell, 2023)
85.63±0.96
79.77±2.45
70.96±2.64
79.05±2.61
74.65±1.88
63.16±1.83
HaloScope (Du et al., 2024)
69.04±6.36
75.64±9.33
78.95±8.82
62.87±10.34
66.15±11.19
61.65±8.32
TSV (Park et al., 2025)
80.01±1.17
68.77±10.12
61.86±7.32
70.46±9.86
64.56±7.16
56.10±2.27
HCPD (Ours)
86.04±2.25
92.47±1.44
95.48±1.26
91.89±1.07
89.92±2.08
78.20±7.42
Qwen-3-8b
SAPLMA (Azaria & Mitchell, 2023)
85.21±1.71
84.11±0.93
79.88±1.65
86.63±1.53
77.08±1.66
68.39±3.73
HaloScope (Du et al., 2024)
53.60±4.11
58.31±8.59
57.42±7.30
74.98±4.19
65.18±10.56
59.16±7.85
TSV (Park et al., 2025)
63.67±3.91
57.88±6.40
56.89±5.85
78.77±0.94
60.67±2.00
59.06±3.88
HCPD (Ours)
85.39±2.70
90.51±1.28
96.63±0.86
92.63±2.90
92.57±3.14
84.21±4.75
Table 8. Comparisons with training-based baselines across target models on NQ Open in terms of AUROC (%).
Source Model
Method
Target Model
LLaMA-3.1-8b
LLaMA-2-7b
LLaMA-2-13b
Qwen-3-8b
Qwen-2.5-7b
Qwen-2.5-14b
LLaMA-3.1-8b
SAPLMA (Azaria & Mitchell, 2023)
76.23±0.82
66.77±1.43
65.55±2.04
68.27±2.34
63.18±1.53
63.82±1.43
HaloScope (Du et al., 2024)
63.38±3.02
74.35±9.93
62.58±7.66
63.68±11.04
59.10±7.32
60.66±4.97
TSV (Park et al., 2025)
70.17±1.47
58.42±3.87
58.20±3.31
61.00±1.57
60.43±3.68
57.24±2.98
HCPD (Ours)
90.38±3.58
92.94±0.75
90.42±1.92
92.45±1.77
85.88±6.03
78.62±6.69
Qwen-3-8b
SAPLMA (Azaria & Mitchell, 2023)
70.55±3.11
66.85±0.67
61.85±1.50
72.86±1.20
68.52±1.95
68.87±2.20
HaloScope (Du et al., 2024)
62.83±9.95
64.84±12.85
55.49±4.94
57.25±1.50
62.00±7.79
58.60±6.65
TSV (Park et al., 2025)
53.94±4.57
57.30±3.92
56.10±3.84
61.38±3.43
57.91±3.73
55.56±2.53
HCPD (Ours)
79.42±3.52
91.60±1.41
93.04±0.98
87.35±6.22
85.70±1.58
60.09±5.91
Table 9. Comparisons with training-based baselines across target models on CoQA in terms of AUROC (%).
Source Model
Method
Target Model
LLaMA-3.1-8b
LLaMA-2-7b
LLaMA-2-13b
Qwen-3-8b
Qwen-2.5-7b
Qwen-2.5-14b
LLaMA-3.1-8b
SAPLMA (Azaria & Mitchell, 2023)
71.58±1.35
76.81±1.18
72.62±0.62
69.07±3.24
68.11±1.32
59.84±1.42
HaloScope (Du et al., 2024)
72.11±4.96
58.10±6.07
62.53±7.26
57.75±2.67
67.05±6.23
53.53±2.05
TSV (Park et al., 2025)
69.31±6.75
53.59±3.55
56.85±3.20
56.82±4.28
59.10±3.59
52.91±1.90
HCPD (Ours)
90.07±2.58
89.12±1.24
89.38±2.02
76.43±3.03
84.19±3.39
60.76±6.24
Qwen-3-8b
SAPLMA (Azaria & Mitchell, 2023)
69.63±1.55
81.52±1.13
77.99±1.40
80.28±1.40
72.91±2.42
73.00±1.40
HaloScope (Du et al., 2024)
74.46±8.04
65.96±12.78
69.45±7.78
62.18±4.49
64.07±10.56
54.10±2.21
TSV (Park et al., 2025)
61.22±10.15
58.82±7.59
59.81±7.30
68.40±4.92
58.63±7.01
56.79±4.45
HCPD (Ours)
88.43±3.50
86.12±0.83
88.09±1.61
84.80±1.01
83.14±0.52
73.26±1.25
23

## Page 24

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
D.2. More Results on Transferability across Data Distributions
We also evaluate cross-dataset evaluation transfer on Qwen-3-8b. As shown in Figure 4, HCPD exhibits strong generalization
under distribution shift, similar to that on LLaMA-3.1-8b. This indicates that by adaptively proposing criteria that fit
the problem domain and format, HCPD is widely applicable across diverse hallucination detection scenarios. Notably,
the agent trained on CoQA even outperforms its in-domain counterpart when transferred to the other three datasets. We
hypothesize that CoQA’s contextual setting provides richer supervision, which may improve knowledge coverage and
contextual reasoning, thereby enhancing the agent’s discriminative capability.
93.69 94.33 93.43 79.25
90.49
82.67
94.12
92.63
79.98
91.07
91.24
87.35
95.91
79.71
73.75
84.80
50
60
70
80
90
100
NQ Open
TriviaQA
SciQ
CoQA
TriviaQA SciQ NQ Open CoQA
Source
Target
Figure 4. Cross-dataset AUROCs of HCPD on Qwen-3-8b.
D.3. Impact of Generation Strategy
As mentioned above, we use different sampling strategies for the training and test phases to serve different objectives. To
seek the most effective strategies for inference-time generation, we evaluate 5 sampling strategies with an increasing degree
of freedom. Intuitively, higher freedom degree can broaden exploration across diverse perspectives, but it also reduces
controllability and may yield unstable or less reliable evaluations (e.g., T = 1.3, top-p = 0.98, RP = 1.0); conversely,
overly deterministic sampling can constrain the assessment, potentially biasing it toward a single criterion or reasoning
pattern (e.g., T = 0.5, top-p = 0.8, RP = 1.3). Results in Table 10 indicate that HCPD can maintain excellent reasoning
ability across strategies with diverse degree of freedom, which further supports the effectiveness of our alignment procedure.
Table 10. Impact of generation strategy.
Dataset
Strategy (T | top-p | RP)
0.5 | 0.8 | 1.3
0.7 | 0.85 | 1.2
0.9 | 0.9 | 1.1
1.1 | 0.95 | 1.05
1.3 | 0.98 | 1.0
TriviaQA
85.51±1.38
85.90±1.44
86.25±1.08
86.03±1.36
82.80±2.01
SciQ
85.59±0.44
86.62±2.03
86.04±2.25
85.94±1.95
85.36±2.48
NQ Open
88.94±2.23
90.09±2.13
90.38±3.58
90.08±2.17
87.30±3.00
CoQA
89.44±2.98
89.83±2.76
90.07±2.58
89.85±2.67
88.14±4.04
D.4. Performance and Consumption Comparison
In light of the practical deployment of HCPD, we analyze its computational overhead from 2 perspectives: i) Inference
time: As shown in Table 5, the multi-sampling aggregation yields substantial performance gains at the expense of additional
computation. However, K = 5 is not strictly necessary, since HCPD already outperforms all baselines at K = 1, reducing
latency to 0.2349s per sample; ii) Inference VRAM: Since most methods require LLM to extract internal states or probabilities,
HCPD does not consume additional VRAM during inference. Moreover, its fixed 7B footprint is more efficient when
evaluating large-scale LLMs. A detailed performance–consumption comparison is provided in Table 11, demonstrating that
HCPD attains efficiency comparable to lightweight metrics while surpassing consistency-based methods.
D.5. Generalization to Long-form Generation Task
Beyond factual hallucinations in short-form QA settings, we further evaluate the detection of faithful hallucinations in a
Wikipedia article continuation task. To assess the generalization capability of HCPD in long-form generation, we directly
apply the agent trained on TriviaQA to a Wikipedia continuation dataset. Notably, we employ DeepSeek-V3 as the evaluation
metric (Liu et al., 2024), since most continuation passages exceed the 512-token limit of BLEURT. Detailed descriptions of
24

## Page 25

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
the dataset and prompting strategy are provided in Appendix C.1 and C.2, respectively. As shown in Table 12, As shown in
Table 12, HCPD consistently outperforms existing baselines, demonstrating robust scalability to long-form generation tasks.
Table 11. Comparisons with baselines in terms of performance, inference time and GPU memory.
Method
AUROC (%) ↑
Inf. Time (s) ↓
GPU Mem. (MiB) ↓
LN-Entropy (Malinin & Gales, 2021)
73.62±2.20
0.1383
17185
CCS (Burns et al., 2022)
78.20±1.89
1.1600
31498
SelfCKGPT (Manakul et al., 2023)
74.58±1.90
6.5310
17540
Perplexity (Ren et al., 2023)
80.62±2.62
0.1349
17185
SAPLMA (Azaria & Mitchell, 2023)
78.51±3.16
0.5341
16019
Semantic Entropy (Kuhn et al., 2023)
78.71±3.09
14.3026
17185
Lexical Similarity (Lin et al., 2024)
77.96±2.03
36.5646
17185
EigenScore (Chen et al., 2024a)
51.35±1.23
37.7550
17453
HaloScope (Du et al., 2024)
58.19±5.79
0.6012
16757
TSV (Park et al., 2025)
79.78±3.36
0.0934
23592
HCPD (Ours, K = 1)
85.21±1.22
0.2349
18513
HCPD (Ours, K = 5)
86.25±1.08
1.1313
19051
Table 12. Cross-dataset transfer performance on Wikipedia, using TriviaQA as the source dataset for training-based baselines.
Method
AUROC
LN-Entropy (Malinin & Gales, 2021)
73.02±1.76
CCS (Burns et al., 2022)
70.20±2.28
SelfCKGPT (Manakul et al., 2023)
72.37±2.03
Perplexity (Ren et al., 2023)
74.19±2.07
HaloScope (Du et al., 2024)
71.74±1.30
TSV (Park et al., 2025)
65.07±1.27
HCPD (Ours)
74.36±1.37
E. Future Directions
While HCPD provides an effective zero-source hallucination detector, several promising extensions warrant further investi-
gation. Future work includes leveraging HCPD as a reward model within LLM training to explicitly suppress hallucinations,
extending the framework to multimodal generation (e.g., image-text or video-text) by incorporating modality-aware evalua-
tion criteria, and generalizing human-like criteria probing mechanism to broader evaluation tasks such as safety, helpfulness,
or style compliance. Furthermore, scaling HCPD to long-form generation, multi-turn dialogue, and tool-augmented systems
will likely require hierarchical or claim-level scoring, as well as dialogue-aware consistency modeling. Collectively, these
directions illustrate a principled path toward establishing HCP as a general, interpretable evaluation paradigm for increasingly
complex and multimodal generation settings.
25

## Page 26

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
F. Examples
Q: 
A:
What colour is the liqueur Galliano?
Galliano is green.
### Specific Criteria:
1. **Factual Grounding**: Verify if the statement about the color of 
Galliano is accurate.
2. **Logical Coherence**: Ensure there are no logical inconsistencies 
within the response.
3. **Semantic Precision**: Check if the use of terminology and phrasing 
is clear and precise.
4. **Temporal Consistency**: Not applicable here since the question 
does not involve any time-specific details.
5. **Social Fairness**: Not applicable here since the question does not 
involve social aspects or cultural elements.
### Analysis:
- **Response 1**:
  - **Factual Grounding**: Galliano is actually amber-colored, often 
described as a golden or honey-brown hue. The response incorrectly 
states that it is green, which is a factual error.
  - **Logical Coherence**: There are no logical inconsistencies in this 
brief response.
  - **Semantic Precision**: The statement uses incorrect terminology 
("green" instead of "amber") which affects precision.
  - **Temporal Consistency**: Not relevant in this case.
  - **Social Fairness**: Not relevant in this case.
### Scores:
\boxed{3};
GT:
Score:
0.2847
3
Q: 
A:
Soul Patch, Reed, Lincoln and Monkey Tail are 
all types of what?
Beards.
#### Specific Criteria ####
1. Factual Grounding (70%)
2. Logical Coherence (15%)
3. Semantic Precision (10%)
4. Temporal Consistency (5%)
5. Social Fairness (5%)
#### Analysis ####
**Response 1: "Beards."**
- **Factual Grounding:** Correct classification, though somewhat brief. 
It mentions beard types without providing additional context or details 
about their characteristics or uses. This minor detail can contribute to a 
fully grounded score.
- **Logical Coherence:** The statement is clear and directly answers the 
question without any logical inconsistencies.
- **Semantic Precision:** The term "beards" is precise in this context, 
but the brevity might suggest a lack of comprehensive explanation.
- **Temporal Consistency:** Not applicable here since the information 
provided does not involve time-specific claims.
- **Social Fairness:** Neutral and unbiased terminology is used, which 
is good.
Overall, the response is accurate and direct while being concise.
#### Scores ####
\boxed{8};
GT:
Score:
0.8165
8
Q: 
A:
Which late English singer was born Ronald 
William Wycherley in April 1940? 
Tom Jones.
#### Specific Criteria ####
1. Factual Grounding (70%)
2. Logical Coherence (15%)
3. Semantic Precision (10%)
4. Temporal Consistency (5%)
5. Social Fairness (5%)
#### Analysis ####
Response 1: "Tom Jones."
- **Factual Grounding**: Incorrect. The actual name of the singer 
who was born Ronald William Wycherley in April 1940 is Roddy 
Woomble, not Tom Jones. This is a severe factual error, leading to a 
low score.
- **Logical Coherence**: Not applicable since there is only one 
statement.
- **Semantic Precision**: The term "Tom Jones" is not misused 
per se, but it is semantically imprecise because it is incorrect.
- **Temporal Consistency**: Not applicable here.
- **Social Fairness**: Not applicable.
#### Scores ####
\boxed{2};
GT:
Score:
0.2246
2
Q: 
A:
Which US singer/musician played Leather 
Tuscadero in the television show ‘Happy Days’?
Suzi Quatro.
### Specific Criteria:
1. **Factual Grounding**: Validate if the claim about who played 
Leather Tuscadero in "Happy Days" is accurate.
2. **Temporal Consistency**: Ensure there are no anachronisms related 
to the character appearance and the actual history of the show.
3. **Social Fairness**: Check for any biased or inappropriate language 
regarding the artist.
4. **Logical Coherence**: Verify that the statement makes sense in the 
context provided.
### Analysis:
- **Response 1 ("Suzi Quatro")**
  - **Factual Grounding**: Suzi Quatro did indeed play the character of 
Leather Tuscadero in "Happy Days." This is well-documented in pop 
culture and music history.
  - **Temporal Consistency**: There are no anachronisms here since the 
character appeared during the series run of "Happy Days," which 
spanned from 1974 to 1977.
  - **Social Fairness**: The response uses her name without any biases 
or cultural insensitivities.
  - **Logical Coherence**: The answer directly addresses the question 
asked without any internal contradictions.
### Scores:
\boxed{9};
GT:
Score:
0.9335
9
Figure 5. Visualizations of successful detections via HCPD on TriviaQA.
26

## Page 27

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Q: 
A:
What is the name of the two metalloids in the 
carbon group called?
The two metalloids in the carbon group are boron 
and silicon.
### Specific Criteria
1. Factual Accuracy
2. Relevance to the Query
3. Clarity and Brevity
### Analysis
**Response 1:** "The two metalloids in the carbon group are boron and 
silicon."
- **Factual Accuracy:** This statement is incorrect. While silicon is 
indeed a metalloid in Group 14 (carbon group), boron is also considered 
a metalloid but is not typically grouped alongside silicon in this manner. 
The response conflates boron with the carbon group when it actually 
belongs to Group 13. Therefore, it does not meet the required accuracy 
standards.
- **Relevance to the Query:** The response addresses the query directly 
and provides relevant information about both metalloids within the 
carbon group. However, due to the factual error, the relevance is 
compromised because the information provided is not entirely accurate.
- **Clarity and Brevity:** The response is concise and clear but fails in 
providing accurate details.
### Scores
\boxed{2};
GT:
Score:
0.1481
2
#### Specific Criteria
- **Factual Grounding**: (70%)
- **Logical Coherence**: (15%)
- **Semantic Precision**: (10%)
- **Temporal Consistency**: (5%)
#### Analysis
**Response 1:**
- **Factual Grounding**: Correctly identifies "organelle" as the term for 
a structure within the cytoplasm that performs a specific job in the cell, 
which aligns perfectly with biological definitions. This receives full 
marks here.
- **Logical Coherence**: The answer directly addresses the question 
without any contradiction or unsupported conclusion. It meets this 
criterion.
- **Semantic Precision**: The term "organelle" is used precisely, and the 
concept is clearly conveyed without ambiguity. It scores well here.
- **Temporal Consistency**: The concept of organelles has been known 
since the early 20th century and does not have any temporal 
inconsistency.
Considering these factors, the response is highly accurate and reliable.
#### Scores
\boxed{8};
GT:
Score:
0.8483
8
Q: 
A:
What is the term for a structure within the 
cytoplasm that performs a specific job in the cell?
organelle.
Q: 
A:
Producing light without a high temperature is 
called?
Producing light without a high temperature is 
called incandescence.
### Specific Criteria
1. Factual Accuracy (70%)
2. Logical Coherence (15%)
3. Semantic Precision (10%)
4. Temporal Consistency (5%)
### Analysis
Response 1:
1. **Factual Accuracy**: The statement "producing light without a high 
temperature is called incandescence" is incorrect. Incandescence refers to 
producing light through heat emission. The correct term for producing 
light without a high temperature is "phosphorescence." Hence, this 
response has severe factual inaccuracy.
2. **Logical Coherence**: There are no major logical inconsistencies or 
unsupported conclusions within the statement, though the incorrect 
terminology does impact its overall coherence.
3. **Semantic Precision**: The use of "incandescence" instead of 
"phosphorescence" results in semantic distortion, as these two 
phenomena describe distinctly different processes.
4. **Temporal Consistency**: This criterion does not apply here since 
there are no time-related claims.
### Scores
\boxed{1}
GT:
Score:
0.1079
1
### Specific Criteria
1. Factual Grounding (40%)
2. Logical Coherence (20%)
3. Semantic Precision (20%)
4. Temporal Consistency (10%)
5. Social Fairness (10%)
### Analysis
**Response 1: "Hydrogen"**
1. **Factual Grounding**: Correct. Zinc reacts with hydrochloric acid to 
produce hydrogen gas. This statement aligns accurately with chemical 
principles.
2. **Logical Coherence**: The response directly answers the question 
without any contradictions or unsupported conclusions. However, it lacks 
additional relevant details about the reaction (like stating the equation or 
mentioning chlorine gas produced along with hydrogen).
3. **Semantic Precision**: The term "hydrogen" is used precisely and 
clearly. There is no ambiguity or miscommunication in this response.
4. **Temporal Consistency**: The response does not involve any 
temporal elements, making this criterion irrelevant here.
5. **Social Fairness**: The response is neutral and does not include any 
cultural insensitivity, stereotyping, or bias.
### Scores
\boxed{7}
GT:
Score:
0.7595
7
Q: 
A:
Zinc reacting with hydrochloric acid produces 
bubbles of which gas?
Hydrogen
Figure 6. Visualizations of successful detections via HCPD on SciQ.
27

## Page 28

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Q: 
A:
who made possible the first reproduction of the 
human voice on the phonograph
Thomas Edison
### Specific Criteria:
1. **Factual Grounding**: 70%
   - Verifying if the claim about Thomas Edison is correct.
2. **Temporal Consistency**: 15%
   - Ensuring there are no time-related inconsistencies.
3. **Social Fairness**: 15%
   - Checking for any cultural insensitivity or biases in the response.
### Analysis:
- **Response 1**: "Thomas Edison"
  - **Factual Grounding**: This is likely partially grounded since 
Thomas Edison did reproduce the human voice on the phonograph, 
making it partially accurate. However, the answer lacks specificity and 
fails to mention the actual contribution or year. Full accuracy would 
include mentioning the achievement along with the name (which may 
lead to further elaboration). Weighted at 9.
  - **Temporal Consistency**: The response does not contain any 
temporal inaccuracies. It’s simply incomplete. Therefore, this criterion 
scores fully consistent.
  - **Social Fairness**: There is nothing socially biased in mentioning 
just one person without qualifications, hence this score receives full 
consistency.
  
Overall Comprehensive Score: \boxed{10}
GT:
Score:
0.9241
10
Q: 
A:
h o s t  o f  t h e  l a t e  s h o w  w h o  w a s  o n c e  a 
correspondent for the daily show
stephen colbert.
#### Specific Criteria:
- Factual Accuracy (40%)
- Logical Coherence (20%)
- Semantic Precision (20%)
- Temporal Consistency (10%)
### Analysis:
#### Response 1: 'stephen colbert.'
1. **Factual Accuracy**: Stephen Colbert is indeed a host of a late-night 
show and has previously been a correspondent for The Daily Show. This 
is a well-established fact.
2. **Logical Coherence**: The statement is straightforward and logical, 
without any contradictions.
3. **Semantic Precision**: The term "Stephen Colbert" is clearly 
defined and used precisely. However, there could be semantic precision 
issues regarding whether he is still hosting the late night show or if the 
question asked about his previous role.
4. **Temporal Consistency**: There is no explicit mention of when this 
was true, which might leave room for uncertainty unless the user 
specifically requested a past or present status.
#### Scores:
\boxed{7}
GT:
Score:
0.7586
7
Q: 
A:
disagreements involving slavery and states' rights 
were two of the main causes of
the American Civil War.
### Specific Criteria
1. Factual Grounding (75% weight)
2. Logical Coherence (20% weight)
3. Semantic Precision (5% weight)
### Analysis
#### Response 1: "the American Civil War."
- **Factual Grounding:** Correct. The statement about 
disagreements involving slavery and states' rights being one 
cause of the American Civil War is generally accurate. This 
includes both direct facts ("states' rights" was indeed a key 
issue) and contextual details.
- **Logical Coherence:** Perfect. The premise directly 
supports the conclusion without any internal inconsistencies.
- **Semantic Precision:** Excellent. The wording is clear and 
unambiguous.
Overall, this response scores very well across all three criteria.
### Scores
\boxed{10}
GT:
Score:
0.9362
10
Q: 
A:
who was elected president in mexico in 2000
Vicente Fox
### Specific Criteria
- **Factual Grounding**: Verify if the statement about the Mexican 
presidential election in 2000 is true. 
- **Temporal Consistency**: Ensure the date mentioned aligns with 
historical record.
- **Social Fairness**: Check for any biases or cultural insensitivity.
### Analysis
**Response 1: "Vicente Fox"**
- **Factual Grounding**: Correct. Vicente Fox was indeed elected 
President of Mexico in 2000.
- **Temporal Consistency**: Correct. The event occurred in 2000 which 
is accurate.
- **Social Fairness**: Neutral. There is no bias or cultural insensitivity 
present here.
Given the high accuracy and relevance to the question:
- **Factual Grounding**: 10/10
- **Temporal Consistency**: 10/10
- **Social Fairness**: 10/10
### Scores
\boxed{10};
GT:
Score:
0.9895
10
Figure 7. Visualizations of successful detections via HCPD on NQOpen.
28

## Page 29

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Q: 
A:
{context}
Q: who walked in first
Phineas Duge
### Specific Criteria:
1. Factual Grounding (Weight: 50%)
   - Who walked into the room first?
2. Logical Coherence (Weight: 25%)
   - Does the response align with the narrative provided?
3. Semantic Precision (Weight: 20%)
   - Are there any misinterpretations or imprecisions in the statement?
4. Temporal Consistency (Weight: 5%)
### Analysis:
- **Factual Grounding:** The response does not mention Virginia 
entering the room before anyone else. It directly states that "Phineas 
Duge" walked in first, which contradicts the given context where it 
clearly mentions that Virginia and Guy enter together.
- **Logical Coherence:** The response does not logically follow the 
sequence of events described in the context. The context shows that 
Virginia and Guy enter the room together, and Phineas Duge enters after 
they do.
- **Semantic Precision:** There is a clear semantic error here as the 
response does not accurately represent the sequence of entries.
- **Temporal Consistency:** This criterion does not apply since the 
context itself is not temporally inconsistent regarding who entered the 
room first.
### Scores:
\boxed{1};
GT:
Score:
0.0564
1
Q: 
A:
{context}
Q: who did harry meet
george purvis
### Specific Criteria
1. **Factual Grounding (60%)**: Verify if the response aligns with the 
provided context.
2. **Logical Coherence (20%)**: Ensure the response does not contain 
any contradictions or unsupported conclusions.
3. **Semantic Precision (15%)**: Check for any misuses of terminology 
or clarity issues.
4. **Temporal Consistency (5%)**: Since the text is set in a historical 
context, ensure there are no anachronisms.
### Analysis
**Response 1**
- **Factual Grounding**: The response is fully grounded. It accurately 
states that Harry met George Purvis, which matches the context provided.
- **Logical Coherence**: There are no contradictions or unsupported 
conclusions in this short response.
- **Semantic Precision**: The response is precise and clear. The use of 
terms like "met" and "George Purvis" is appropriate and clear.
- **Temporal Consistency**: The response is temporally sound, fitting 
within the established timeline without any anachronisms.
### Scores
\boxed{9}
GT:
Score:
0.8950
9
Q: 
A:
{context}
Q: What city hit 88,859 population in 2011?
Bath
### Specific Criteria:
1. **Factual Grounding (50%)**: Verify if the provided information 
about the population of Bath in 2011 aligns with the given context.
2. **Temporal Consistency (20%)**: Ensure that there are no 
anachronisms regarding the timing of the population data.
3. **Semantic Precision (20%)**: Check if the wording is clear and 
precise without any misinterpretations.
4. **Logical Coherence (10%)**: Assess whether the response logically 
follows from the context provided.
### Analysis:
- **Response 1**: [' Bath\n']
  - **Factual Grounding**: The response directly mentions Bath, which 
matches the context provided. This is fully grounded as the context 
clearly states that Bath had a population of 88,859 in 2011.
  - **Temporal Consistency**: The response does not introduce any 
temporal inconsistencies since it simply reiterates the given year.
  - **Semantic Precision**: The answer is clear and precise, just stating 
the city name without additional information that could potentially 
distort the meaning.
  - **Logical Coherence**: The response is entirely coherent with the 
question asked and the provided context.
### Scores:
\boxed{8}
GT:
Score:
0.7753
8
Q: 
A:
{context}
Q: Where was the Excommunication published?
In the Duomo.
### Specific Criteria
1. **Factual Grounding (Weight: 70%)**: The response should 
accurately state the location of the Excommunication publication.
2. **Temporal Consistency (Weight: 20%)**: Ensure that the statement 
does not introduce anachronisms or temporal inconsistencies.
3. **Semantic Precision (Weight: 10%)**: Verify that the wording is 
clear and precise without ambiguities.
### Analysis
**Response 1**: "In the Duomo."
- **Factual Grounding**: This response directly answers the question 
correctly according to the text provided ("Romola went to witness the 
scene, that the resistance it inspired might invigorate that sympathy with 
Savonarola which was one source of her strength. It was in memorable 
contrast with the scene she had been accustomed to witness there." 
implies the Excommunication was indeed published in the Duomo.)
- **Temporal Consistency**: There are no temporal inconsistencies in 
this brief response.
- **Semantic Precision**: The wording is simple and precise, without 
any ambiguity.
### Scores
\boxed{9};
GT:
Score:
0.8341
9
Figure 8. Visualizations of successful detections via HCPD on CoQA.
29

## Page 30

Zero-source LLM Hallucination Detection with Human-like Criteria Probing
Q: 
A:
Poopdeck Pappy is which fictional character’s 
father?
Popeye the Sailor Man.
#### Specific Criteria:
1. Factual Grounding (50% weight)
2. Logical Coherence (25% weight)
3. Semantic Precision (20% weight)
4. Temporal Consistency (5% weight)
5. Social Fairness (5% weight)
#### Analysis:
- **Response 1:**
  - **Factual Grounding:** Incorrect claim about Poopdeck Pappy being 
Popeye the Sailor Man's father. This is a clear factual error since the 
context asks specifically about which fictional character's father 
Poopdeck Pappy is, and Popeye does not have this relation. The correct 
answer would be Wimpy from the Popeye comics.
  - **Logical Coherence:** The statement is factually incorrect but 
doesn't introduce any logical contradictions within itself.
  - **Semantic Precision:** "Popeye the Sailor Man" is spelled out 
incorrectly as "Popeye the Sailor Man" instead of "Popeye," but the 
semantic precision isn't significantly affected by this minor issue.
  - **Temporal Consistency:** Not applicable here as there are no time-
related claims.
  - **Social Fairness:** The response is respectful and neutral, hence 
socially fair.
#### Scores:
\boxed{3};
GT:
Score:
0.9181
3
(a) TriviaQA
Q: 
A:
Structures like the human tail bone and whale 
pelvis are called what?
vestigial structures.
#### Specific Criteria ####
1. **Factual Grounding**: Correct terminology related to evolutionary 
biology and anatomy.
2. **Logical Coherence**: Proper mapping of the concept of vestigial 
structures.
3. **Semantic Precision**: Accuracy in the use of the term "vestigial 
structures."
4. **Temporal Consistency**: No concerns as this is about current 
biological terminology.
#### Analysis ####
- **Response 1**: "vestigial structures."
  - **Factual Grounding (7)**: The term "vestigial structures" accurately 
describes the anatomical components that have lost their original 
function during evolution. It aligns well with established biological 
definitions.
  - **Logical Coherence (8)**: The answer directly addresses the user’s 
question without introducing any contradictions or unsupported claims.
  - **Semantic Precision (9)**: The term "vestigial structures" is used 
correctly, conveying the precise meaning requested.
  - **Temporal Consistency (10)**: There are no temporal issues as this 
is a standard term in biology.
#### Scores ####
\boxed{4}
GT:
Score:
0.8961
4
(b) SciQ
#### Specific Criteria ####
1. Factual Grounding (5 points)
2. Temporal Consistency (5 points)
#### Analysis ####
Response 1: "in the 1940s"
1. **Factual Grounding**: The vaccine for whooping cough 
(pertussis) was developed much earlier than the 1940s. The 
first effective pertussis vaccines were licensed in the United 
States in 1926. Therefore, this statement is inaccurate, leading 
to a low score here.
2. **Temporal Consistency**: Since the response mentions a 
timeframe that does not align with the actual development of 
the pertussis vaccine, there is a clear discrepancy in temporal 
accuracy.
#### Scores ####
\boxed{4}; 
GT:
Score:
0.8205
4
Q: 
A:
when did they start vaccinating for whooping 
cough
in the 1940s
(c) NQ Open
Q: 
A:
{context}
Q: What-all's his name?
His name is Franz Bussman.
### Specific Criteria
1. **Factual Grounding** (40%)
   - Validity of the name provided.
2. **Logical Coherence** (30%)
   - Consistency of the information with the narrative provided.
3. **Temporal Consistency** (20%)
   - Accuracy of any time-related claims within the context.
4. **Social Fairness** (10%)
   - Respectful and unbiased presentation of names and identities.
### Analysis
**Response 1:**
- **Factual Grounding**: The name "Franz Bussman" matches the character 
introduced in the narrative. This criterion scores high because it does not 
introduce any new or incorrect information about the brother’s identity. Score: 
9.
- **Logical Coherence**: The response is directly related to the question asked 
("What-all's his name?") without introducing any additional information or 
contradictions. It is logically consistent with the narrative. Score: 9.
- **Temporal Consistency**: There are no temporal inconsistencies as the 
name provided fits the context without referencing any specific timeframe 
incorrectly. Score: 10.
- **Social Fairness**: The name "Franz Bussman" is presented neutrally and 
respectfully, avoiding any biased or discriminatory language. Score: 10.
### Scores
\boxed{2}
GT:
Score:
0.7127
2
(d) CoQA
Figure 9. Visualizations of failed detections via HCPD.
30
