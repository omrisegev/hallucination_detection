---
source_pdf: papers/Unsupervised Process Reward Models.pdf
slug: unsupervised-process-reward-models
pages: 25
extracted_on: 2026-08-10
---

# Unsupervised Process Reward Models

## Page 1

Unsupervised Process Reward Models
Artyom Gadetsky∗
Maxim Kodryan∗
Siba Smarak Panigrahi
Hang Guo
Maria Brbic
Swiss Federal Institute of Technology (EPFL)
Abstract
Process Reward Models (PRMs) are a powerful mechanism for steering large lan-
guage model reasoning by providing fine-grained, step-level supervision. However,
this effectiveness comes at a significant cost: PRMs require expert annotations for
every reasoning step, making them costly and difficult to scale. Here, we propose
a method for training unsupervised PRMs (uPRM) that requires no human super-
vision, neither at the level of step-by-step annotations nor through ground-truth
verification of final answers. The key idea behind our approach is to define a
scoring function, derived from LLM next-token probabilities, that jointly assesses
candidate positions of first erroneous steps across a batch of reasoning trajectories.
We demonstrate the effectiveness of uPRM across diverse scenarios: (i) uPRM
achieves up to 15% absolute accuracy improvements over the LLM-as-a-Judge in
identifying first erroneous steps on the ProcessBench dataset; (ii) as a verifier for
test-time scaling, uPRM performs comparably to supervised PRMs and outper-
forms the majority voting baseline by up to 6.9%, and (iii) when used as a reward
signal in reinforcement learning, uPRM enables more robust policy optimization
throughout training compared to a supervised PRM trained using ground-truth la-
bels. Overall, our results open a path toward scalable reward modeling for complex
reasoning tasks.
1
Introduction
Improvements in the step-by-step reasoning abilities of large language models (LLMs) have become
a cornerstone for their recent success in domains such as mathematics and programming [1–4]. In
order to incentivize or steer the reasoning process in LLMs, one needs to evaluate their correctness.
The basic approach to achieve this is by computing a single score for the whole reasoning trajectory
(e.g., verifying only the final answer of a solution) using Outcome Reward Models (ORMs) [5, 4, 6].
However, using such sparse and crude feedback, especially for long chains of thought, is extremely
ineffective and can lead to false positives, reinforcing incorrect reasoning traces that ultimately result
in formally correct answers [7].
In contrast, Process Reward Models (PRMs) [8] were introduced to produce dense step-wise scores
that can guide the reasoning process more gradually. Naturally, such finer control over reasoning leads
to improved results in both test-time scaling (TTS) [9] and reinforcement learning (RL) [10]. Despite
their overall advantage over ORMs, PRMs have a significant limitation: they require meticulously
labeled training data containing step-by-step annotated reasoning trajectories. To address this problem,
numerous frameworks have been developed to infer step-level labels from ground truth final answers
based on brute-force Monte Carlo estimations [11, 12] or implicit process reward modeling [13, 14].
However, these approaches still rely heavily on the availability of ground truth answers in the data
∗Equal contribution
Preprint.
arXiv:2605.10158v1  [cs.LG]  11 May 2026

## Page 2

or access to external verifiers, and are often highly computationally demanding, which limits their
general applicability.
In this work, we present an approach for training fully unsupervised Process Reward Models (uPRMs)
that requires neither step-level annotations nor ground-truth verification of final answers. Our key
insight is that LLMs, through their next-token probabilities, implicitly encode judgments about the
correctness of reasoning steps. Specifically, we construct sequences that interleave reasoning steps
with correctness markers, and extract the probabilities an LLM assigns to these markers to define
the scoring function that measures how plausible a given error position is. By evaluating multiple
trajectories jointly rather than independently, we leverage the in-context learning capabilities of
LLMs to obtain more reliable assessments. We then train uPRM to optimize this joint score via RL,
effectively distilling the LLM’s evaluation capability into a dedicated process reward model.
We demonstrate the effectiveness of our uPRM through a diverse set of experiments:
• We show that uPRM effectively identifies the positions of first erroneous steps, achieving up
to 15% absolute accuracy improvements over the LLM-as-a-Judge baseline. Remarkably,
uPRM achieves the largest gains on the most challenging datasets such as Olympiad-
Bench [15] and Omni-Math [16].
• In test-time scaling experiments, we show that uPRM outperforms the majority voting
baseline by up to 6.9% absolute gains when verifying 256 generations of Llama-3.2-1B-
Instruct [17]. Moreover, although being fully unsupervised, uPRM is competitive with
various supervised PRMs trained with step-level human-labeled annotations on Best-of-8
selection.
• We show that uPRM can be used as a reward signal for RL. Surprisingly, compared to a
supervised PRM trained with ground-truth labels, which is prone to rapid reward hacking,
uPRM supports more robust policy optimization across training runs. Although it does not
fully eliminate reward hacking, we observe that such failures arise less frequently and tend
to be less severe, yielding superior final performance across multiple policy models. For
example, uPRM yields a 4% accuracy gain for Qwen2.5-Math-1.5B [18] over training with
a verifiable outcome reward.
2
Related Work
Process Reward Models from Outcome Labels. Since manually obtaining granular annotations
can be laborious and expensive [8], a variety of approaches have emerged to take advantage of
the available outcome labels to obtain process supervision for training PRMs. For example, Math-
Shepherd [11] proposed an automatic process annotation procedure that assigns a label to each step
based on its potential to lead to a correct final answer. Similar automated annotation techniques
were proposed in subsequent works [12, 19–21]. Nevertheless, such techniques only complement
the labeling corresponding to actual step correctness [22], and require significant computational
resources for Monte Carlo rollouts.
An alternative approach is based on implicit process reward modeling, in which a PRM is learned
directly from the outcome rewards, without relying on explicitly annotated reasoning steps. In partic-
ular, Yuan et al. [13] and Cui et al. [14] develop this idea by introducing a special parameterization of
an ORM that allows for interpreting its partial responses as Q-values required for deriving implicit
process rewards. Other works suggest leveraging ORM outputs to provide step-wise feedback for
training PRMs by computing a relative confidence change [23], introducing a modified Bradley-Terry
objective [24], or by adopting buffering probabilities to reduce label noise [25].
While these methods reduce the need for step-level annotations, they remain dependent on access
to the ground-truth outcome labels either for assessing Monte Carlo rollouts or for training the
underlying ORM. In contrast, our approach eliminates this requirement entirely, training PRMs
without any supervision at either step-level annotations or outcome labels.
LLM-as-a-Judge paradigm. Large language models have been employed as automatic evaluators in
various complex tasks due to their ability to process diverse data types and provide flexible assessment,
eliminating the need for expert annotations. Prominent instances include MT-Bench and Chatbot
Arena [26], as well as AlpacaEval [27], which use strong LLMs to perform pairwise comparisons
of candidate responses and aggregate win-rates. GPTScore [28] uses the generation likelihood of
2

## Page 3

candidate text given an instruction as a quality measure. G-Eval [29] prompts an LLM to output
discrete scores and uses token probabilities over score tokens to compute a weighted average, yielding
more continuous and stable evaluations.
Most existing LLM-as-a-Judge pipelines operate by prompting an LLM to generate an explicit verdict,
and then parsing the generated text into a discrete label or score. Viewed through this lens, our method
can be seen as an instantiation of the LLM-as-a-Judge paradigm, but instead of sampling a judgment,
it employs raw next-token probabilities to define a scoring function that measures how plausible
a given solution is. Furthermore, while prior work primarily leveraged LLM judges for offline
evaluation and model selection, we convert the judge’s probabilistic assessment into an optimization
objective that provides direct supervision for training PRMs.
Test-time Scaling with Process Reward Models. Test-time scaling (TTS) involves allocating
additional compute resources to an LLM during inference to enhance task performance [3, 9, 12, 30].
This paradigm includes a sampling strategy to generate diverse candidate answers and a method to
select the final response, typically using a reward model [31]. Common sampling strategies include (i)
Best-of-N [31], where N independent answers are generated and scored, and the answer with highest
aggregated score is selected, (ii) Beam Search [9], in which intermediate nodes within each beam are
retained or discarded using scores from reward model, and (iii) Diverse Verifier Tree Search (DVTS)
[32], which constructs multiple, independent beam search trees to increase response diversity. In
addition to these approaches, majority voting is a reward-model-free method that selects the most
frequent answer.
One major concern with PRMs in TTS is the effective use of the assigned rewards to select the
final response. Current selection methods do not achieve similar performance to the pass@N
metric, where a single-correct answer is sufficient, and have led to recent exploration on improving
PRMs [33, 22, 34, 35]. In our work, we observe that uPRM performs on par with existing supervised
counterparts despite being fully unsupervised.
Reinforcement Learning with Process Reward Models. RL has been widely adopted to incentivize
reasoning abilities in LLMs, particularly to solve mathematical problems [4, 36]. Most popular
frameworks assign a sparse outcome reward for the entire response generated by the policy model. A
more desirable option would be to introduce dense intermediate rewards into the reasoning process
so that learning becomes more effective [11, 37, 14]. One of the key challenges in applying PRMs
to RL is reward hacking, where the policy learns to exploit spurious patterns in the reward model
rather than genuinely improving reasoning quality [38, 4]. Existing work has focused on algorithmic
mitigations, such as min-form credit assignment [10], but reward hacking is generally considered
inevitable when relying solely on PRM rewards. In our experiments, we find that uPRM exhibits
better robustness to reward hacking than a supervised PRM trained on the same dataset.
3
Background
3.1
Supervised Process Reward Models
Let τ = (x, y) be a solution trajectory consisting of a problem x and a sequence of reasoning steps
y = (y1, . . . , yT ) tackling it. We use the prefix notation y1:t = (y1, . . . , yt) and write τ≤t := (x, y1:t)
for the partial trajectory up to step t. A parametrized process reward model rθ(ct|τ≤t) defines a
distribution over step-correctness labels ct ∈{0, 1}, where ct = 1 indicates that step yt is correct2
and θ refers to trainable parameters.
In practice, training a PRM requires a labeled dataset D where each solution trajectory τ is paired
with the corresponding ground truth label jgt that indicates the position of the first erroneous step3.
Given such labeled dataset, PRM is usually trained with the maximum likelihood objective:
max
θ
E(τ,jgt)∼D log pθ(j = jgt|τ),
(1)
2In the literature, PRMs are sometimes defined as models that behave like value functions, estimating the probability that
a partial trajectory will eventually yield a correct final answer rather than stepwise correctness. In this paper, we focus on
step-level correctness as defined above.
3We follow such definition as the meaning of step’s correctness may become ambiguous after the first erroneous step.
3

## Page 4

with the log-likelihood log pθ(j|τ) defined as:
log pθ(j|τ) :=1[j ≤T] · log rθ(cj=0|τ≤j) +
X
t<j
log rθ(ct=1|τ≤t),
(2)
where the random variable j ∈{1, . . . , T, T + 1} represents the position of the first erroneous step
in τ, with j = T + 1 indicating no error, and 1[·] corresponds to Iverson bracket.
3.2
Large Language Models as Scoring Functions
Pre-trained large language models (LLMs) can be repurposed to define scoring functions for down-
stream tasks by leveraging their next-token probabilities. In particular, given an LLM and a suitably
constructed prompt, one can measure the plausibility of candidate solutions by examining and
combining probabilities the model assigns to specific tokens.
For example, consider the task of verifying a biographical claim about Albert Einstein. Given the
template T = “Albert Einstein won [award] in [year] for [contribution]” with candidates filled in,
we can extract and sum probabilities at each position to define the score for a candidate triplet.
More generally, extracting and blending probabilities at arbitrary positions within a templated
sequence allows defining complex scoring functions S(a; T ) that assess the plausibility of answers a.
Intuitively, such scoring functions measure consistency with the knowledge acquired by the LLM
during the pre-training stage.
Given such a score S(a; T ), a policy πθ can be trained to produce the most plausible answers via
reinforcement learning:
max
θ
ET ∼D,a∼πθS(a; T ).
(3)
In the following section, we build on this principle to construct a score for training PRMs without
access to ground-truth labels jgt.
4
Unsupervised Process Reward Models
Our goal is to train a PRM without relying on the curated labels jgt. The key idea is to define
a scoring function derived from LLM next-token probabilities, which measures how plausible a
candidate position of the first erroneous step is in a given trajectory. Subsequently, we train uPRM by
optimizing this score, eliminating the need for any expert annotations.
4.1
Scoring First Erroneous Position with LLMs
Consider a trajectory τ = (x, y1, . . . , yT ) and a candidate position of the first erroneous step
j ∈{1, . . . , T + 1}. To define the scoring function, we interleave reasoning steps with correctness
labels, marking steps y1, . . . , yj−1 as correct and step yj as incorrect, resulting into a sequence:
s(τ, j) = [x, y1, +, . . . , yj−1, +, yj, -],
(4)
where “+” and “-” denote correct and incorrect labels respectively. The special case j = T + 1 (no
error) corresponds to all steps marked as correct:
s(τ, T + 1) = [x, y1, +, . . . , yj−1, +, yT , +].
(5)
We feed the constructed sequence to an LLM and extract the next-token probabilities LLM assigns to
each label to define the scoring function S(j; s) as follows:
S(j; s) :=1[j ≤T] · log p-
j +
X
t<j
log p+
t,
(6)
where p+
t and p-
t denote the LLM’s next-token probabilities of generating the label tokens “+” and “-”
after yt, respectively, renormalized over {+, -}.
4

## Page 5

4.2
Scoring Multiple Trajectories at Once
The score in Eq (6) can be viewed as an instance of the LLM-as-a-Judge paradigm [26, 29, 28].
Recent works have shown that LLMs produce more reliable judgments when evaluating multiple
instances jointly rather than independently, whether through comparative ranking [39], batched
evaluation [40], or sequential in-context learning [41]. Motivated by this, we extend our score to joint
assessment of positions of first erroneous steps j1, . . . , jN, jn ∈{1, . . . , Tn + 1} for a batch of N
trajectories τ1, . . . , τN, τn = (x, y1, . . . , yTn).
To jointly score a batch of trajectories, we concatenate marked sequences s(τn, jn) together, obtain-
ing:
s1:N = [s(τ1, j1), . . . , s(τN, jN)].
(7)
Subsequently, the resulted sequence is fed to the LLM and the joint score is defined as:
S(j1:N; s1:N) = 1
N
N
X
n=1

1[jn ≤Tn] · log p-
n,jn +
X
t<jn
log p+
n,t

,
(8)
where p+
n,t and p-
n,t now denote the LLM’s next-token probabilities of generating the corresponding
label tokens for step t in trajectory τn, conditioned on all preceding tokens in s1:n, and renormalized
over {+, -} as before. It is worth noting that in this formulation, the score for a trajectory τn is
computed given the previous trajectories τ1, . . . , τn−1 along with their candidate labels j1, . . . , jn−1
as in-context examples. In practice, we observed a failure mode induced by this in-context learning
effect. In particular, the joint score can become spuriously large for configurations in which all
trajectories share the same label jn, regardless of the actual error positions. We describe a simple
correction that mitigates this effect in Appendix A.
4.3
Training PRM via Optimizing Joint Score
We parameterize PRM rθ(ct|τ≤t) by applying LoRA [42] to the same LLM used for computing the
joint score. Noteworthy, this parametrization can be seen as an instantiation of self-training, in which
a model trains by obtaining training signal from itself [43, 44, 41]. We follow recent best practices in
model architectures to define PRMs [22]. In particular, given a trajectory τ = (x, y1, . . . , yT ), we
construct a sequence by interleaving each reasoning step with a special token [*]:
[x, y1, [*], y2, [*], . . . , yT , [*]],
(9)
where the embedding of [*] is trainable. We process this sequence with the LLM and extract the
last-layer hidden state zt at each [*] token position following step yt.
To obtain step-level correctness probabilities, we replace the language modeling head with a two-layer
MLP with ReLU activation that projects each hidden state to two logits:
lt = MLP(zt) ∈R2,
(10)
which are converted to probabilities via softmax:
rθ(ct = 1|τ≤t) =
exp((lt)1)
exp((lt)0) + exp((lt)1), rθ(ct = 0|τ≤t) = 1 −rθ(ct = 1|τ≤t).
(11)
The distribution over the position of the first erroneous step pθ(j|τ) is then defined as in Equation (2).
We train pθ by optimizing the following entropy-regularized objective [45]:
max
θ
E{τn}N
n=1∼D
"
Ejn∼pθ(·|τn)
h
S(j1:N)
i
+ γ
N
N
X
n=1
H(pθ(·|τn))
#
,
(12)
where H(·) denotes Shannon entropy that prevents pθ from premature convergence, and γ corresponds
to the regularization strength. We set γ by monitoring the training curves and choosing the value
that prevents collapse of rθ throughout the training. We study the effect of γ on the optimization in
Appendix D.1.
Efficient Optimization. We develop a custom gradient estimator inspired by the actor-critic frame-
work [46] to enable efficient optimization of the objective (12). In particular, on 8 H200 GPUs, uPRM
5

## Page 6

training via our custom RL takes ≈5.5 hours, compared to ≈4.25 hours for supervised PRM trained
via SFT on the same data and architecture, highlighting that the additional computational overhead
is negligible relative to the expert labeling effort it removes. It is important to emphasize that joint
scoring is used only during uPRM training. At test time, the trained uPRM processes trajectories
independently, reflecting any existing PRM inference with no additional context length requirements.
Thus, the overhead is a one-time training cost, not an inference cost. The details on the estimator are
provided in Appendix B. Furthermore, rather than treating N as the hyperparameter, we design a
principled trajectory packing strategy that maximizes GPU memory utilization and ensures stable
signal-to-noise ratio throughout training. We provide the details of this strategy in Appendix C.2.
5
Experiments
Method Instantiation. We employ Qwen2.5-14B-Instruct [47] to calculate the joint score in Eq (8)
and instantiate the PRM rθ in Eq (11). It is important to emphasize that Qwen2.5-14B-Instruct’s post-
training didn’t involve training on any step-level correctness labels of reasoning chains, thus, keeping
our setup fully unsupervised with respect to these labels. We train uPRM on the PRM800K dataset [8],
using only the reasoning trajectories without any correctness labels. The detailed description of the
experimental setup and the implementation details are provided in Appendix C.
We evaluate uPRM along three dimensions. In Section 5.1, we directly assess its ability to detect step-
level errors on the ProcessBench benchmark [48]. In Section 5.2, we use uPRM as a verifier coupled
with various test-time scaling approaches, measuring its ability to successfully guide inference.
Last but not least, in Section 5.3, we use uPRM as a reward signal for reinforcement learning,
demonstrating that it can effectively guide policy optimization.
5.1
ProcessBench
We first evaluate the ability of uPRM to identify the position of the first erroneous step in reasoning
trajectories as the most direct evaluation protocol. We employ ProcessBench [48], a benchmark specif-
ically designed to evaluate process reward models on step-level error detection. ProcessBench contains
reasoning trajectories generated by various LLMs across four mathematical reasoning datasets of
increasing difficulty: GSM8K [5], MATH [49], OlympiadBench [15], and Omni-MATH [16]. Each
trajectory is annotated with the position of the first erroneous step, or marked as fully correct if no
errors are present.
Following Zheng et al. [48], we report three metrics: (i) accuracy on erroneous trajectories, measuring
how often the model correctly identifies the first mistake in trajectories that contain errors; (ii) accuracy
on correct trajectories, measuring how often the model correctly concludes that a trajectory is error-
free; and (iii) F1 score computed as the harmonic mean of the two accuracies, which serves as
the primary aggregated metric. We report F1 scores in Table 1 and provide the full breakdown in
Table D1.
We compare uPRM against LLM-as-a-Judge, which uses the same base model to score each trajectory
independently. Given a trajectory τ, the baseline predicts the first erroneous position as ˆj =
arg maxj∈{1,...,T +1} S(j; s), where S(j; s) is defined in Equation (6). This baseline shares the same
prompt template, parametrization over the position of the first erroneous step, and base model as
our method. Consequently, this controlled setup ensures that the improvements directly reflect the
benefits of joint scoring via in-context learning.
Table 1: Results on the ProcessBench dataset (F1 score).
ProcessBench
GSM8K
MATH
OlympiadBench
Omni-MATH
LLM-as-a-Judge
49.8
42.8
29.4
26.6
uPRM (ours)
58.3
52.6
42.7
39.8
As shown in Table 1, uPRM consistently outperforms the LLM-as-a-Judge baseline across all four
datasets by a large margin. The improvements are particularly pronounced on the more challenging
benchmarks: uPRM achieves a 13% absolute improvement on OlympiadBench and 13% on Omni-
6

## Page 7

MATH. This suggests that training the PRM to optimize the joint score is especially beneficial when
the underlying reasoning is more complex and the LLM’s independent judgments are less reliable.
These results confirm that our unsupervised training procedure successfully distills the evaluation
capability of the LLM into a dedicated process reward model.
5.2
Scaling Test-Time Compute with uPRM
We next evaluate the utility of uPRM in the test-time compute scaling paradigm [8, 9]. We use
Best-of-N and DVTS sampling strategies that require rewards from uPRM to guide and select the
final response, and compare with majority voting as a baseline.
Experimental Setup. We evaluate a range of instruction-tuned LLMs across different parameter
scales, including Qwen2.5-Instruct series (1.5B, 7B, 14B) [47], Llama-3.2-1B-Instruct, and Llama-
3.1-8B-Instruct [17] to generate candidate responses. We set the generation temperature to 0.7 and
use nucleus sampling with a cumulative probability threshold of 0.8. We define the test-time compute
budget in terms of the number of independent generations, which we scale as powers of 2, up to
256 candidate answers per question. We assess the performance using accuracy on three standard
mathematical benchmarks: MATH-500 [49], MinervaMath [50], and Olympiad Bench [15]. Finally,
since PRM assigns step-level scores, we use the last-step score as the overall score for a candidate
answer [9]. We conduct an ablation study using an alternative aggregation method, where the product
of step-wise scores is considered [8, 22], as detailed in Appendix C.3. Our results indicate that the
last-step score marginally outperforms the product score. For all TTS experiments with uPRM, we
always run three independent seeds and report the mean performance.
20
21
22
23
24
25
26
27
28
30
40
50
MATH500
Accuracy (%)
Llama-3.2-1B-Instruct
20
21
22
23
24
25
26
27
28
50
55
60
65
70
Llama-3.1-8B-Instruct
20
21
22
23
24
25
26
27
28
55
60
65
70
Qwen2.5-1.5B-Instruct
20
21
22
23
24
25
26
27
28
70.0
72.5
75.0
77.5
80.0
82.5
Qwen2.5-7B-Instruct
20
21
22
23
24
25
26
27
28
76
78
80
82
84
Qwen2.5-14B-Instruct
20
21
22
23
24
25
26
27
28
7.5
10.0
12.5
15.0
17.5
20.0
MinervaMath
Accuracy (%)
20
21
22
23
24
25
26
27
28
25.0
27.5
30.0
32.5
35.0
37.5
20
21
22
23
24
25
26
27
28
20
25
30
20
21
22
23
24
25
26
27
28
35
40
45
50
20
21
22
23
24
25
26
27
28
46
48
50
52
54
56
20
21
22
23
24
25
26
27
28
# Generations
10
15
20
OlympiadBench
Accuracy (%)
20
21
22
23
24
25
26
27
28
# Generations
20
25
30
35
20
21
22
23
24
25
26
27
28
# Generations
20
25
30
35
20
21
22
23
24
25
26
27
28
# Generations
37.5
40.0
42.5
45.0
47.5
20
21
22
23
24
25
26
27
28
# Generations
40.0
42.5
45.0
47.5
50.0
Best-of-N
Majority
DVTS
Figure 1: Accuracy of LLMs across different scales on MATH-500, MinervaMath, and Olympiad-
Bench with different test-time scaling approaches based on uPRM. Majority voting is an uPRM-
independent baseline. Results are reported across three seeds.
Results. The results in Figure 1 show that uPRM assigns meaningful rewards to candidate responses,
leading to performance improvements as the test-time compute budget increases. Notably, in Llama-
3.2-1B-Instruct, the average accuracy across three benchmarks jumps from 14.6% (with 1 candidate
response) to 31.7% (with 256 candidate responses), an absolute improvement of 17.1%. In contrast,
the impact is weaker in larger LLMs, where majority voting acts as a strong baseline. Furthermore,
the performance improvements depend on the sampling strategy, the model, and its size. For
instance, DVTS with uPRM significantly outperforms other sampling strategies in smaller models,
leading to 6.9% and 4.4% over majority voting for Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct,
respectively, and 2.8% and 1.5% improvements over Best-of-N. However, the performance degrades
with DVTS for larger policy LLMs. Similar findings about dependence on sampling strategies and
non-generalization of PRMs have also been studied previously with supervised PRMs [22, 30].
7

## Page 8

We next compare uPRM with several supervised PRMs, trained with ground truth step-by-step
annotations or with annotations obtained via credit assignment from ground truth final answers
(Table 2). In particular, we use Math-Shepherd-PRM-7B [11], RLHFlow-PRMs [51], Skywork-PRM-
7B [52], Qwen2.5-7B-Math-Instruct based PRMs [22] and Implicit PRM [13]. Additionally, to have
a controlled setup, we trained supervised PRM model on the PRM800K dataset with benchmark step
labels; otherwise, we used the same setup as for the uPRM. We refer to the resulting supervised PRM
as sPRM. We use Qwen2.5-7B-Math-Instruct [18] as policy and generate 8 answers per question. We
compare Best-of-8 against pass@8.
Table 2: Performance comparison between supervised PRMs and our uPRM on Best-of-8 strategy for
generations from Qwen2.5-Math-7B-Instruct. Results are averaged across three random seeds.
PRM
MATH-500
Minerva
Olympiad
Avg.
Math
Bench
pass@8 (Upper Bound)
91.5±0.4
55.5±0.3
60.3±0.8
69.1
Math-Shepherd-PRM-7B
86.8±0.9
47.3±1.2
47.1±0.1
60.4
RLHFlow-PRM-Mistral-8B
86.6±1.1
46.9±1.5
46.4±0.4
60.0
RLHFlow-PRM-Deepseek-8B
86.8±1.1
47.2±1.9
45.9±1.0
60.0
Skywork-PRM-7B
87.4±0.6
46.6±0.8
48.4±0.5
60.8
Qwen2.5-Math-7B-PRM800K
87.1±0.6
47.1±0.4
46.9±0.7
60.4
Qwen2.5-Math-PRM-7B
87.0±0.9
47.2±0.2
47.7±0.4
60.6
Implicit PRM (CE)
86.3±0.6
47.4±1.2
46.6±0.3
60.1
Implicit PRM (DPO)
86.5±0.7
47.2±1.0
46.4±0.4
60.0
sPRM
86.3±1.1
46.7±0.6
47.1±0.1
60.0
uPRM
86.5±0.5
46.7±2.2
47.1±0.4
60.1
Across three datasets, we find that remarkably, uPRM is competitive with supervised PRMs, including
those initialized from specialized versions for mathematical tasks, despite being fully unsupervised,
and initialized from a generic instruction-tuned LLM (i.e., Qwen2.5-14B-Instruct).
5.3
Reinforcement Learning with uPRM
Next, we explore whether uPRM obtained with our framework can be used as a reward source for
reinforcement learning (RL). To this end, we adopt the state-of-the-art PURE framework [10] that
incorporates dense rewards from PRMs into RL via min-form credit assignment. For a group of
responses generated by a policy, PURE computes return values as an approximate minimum of
per-step rewards and propagates them to per-token advantages, allowing PRM outputs to be naturally
embedded into any policy gradient RL framework.
Experimental Setup. We adopt the codebase4 and experimental setup of Cheng et al. [10] and
perform RL fine-tuning of Qwen2.5 policy models [47, 18]: Qwen2.5-7B, Qwen2.5-Math-7B, and
Qwen2.5-Math-1.5B. Following Cheng et al. [10], we use RLOO [53] as a training algorithm and a
subset of hard (level 3–5) problems from the MATH dataset [49] as training data. While the principled
min-form credit assignment and other algorithmic adjustments introduced in PURE help delay reward
hacking (RH), Cheng et al. [10] argue that it is still inevitable when relying solely on PRM rewards.
Therefore, as a preventive measure, the authors propose mixing per-step rewards with a standard
outcome verifiable reward (VR) for a subset of the data to introduce an auxiliary ground-truth signal
that will prevent overfitting on PRM. In line with these recommendations, we consider three training
scenarios: (i) using VR only, (ii) using PRM rewards only, and (iii) using PRM + VR for 10% of the
data, as in Cheng et al. [10]. We keep hyperparameters the same for all training runs and vary only
the reward source. Details can be found in Appendix C.4.
Results. We compare the performance of the RL-trained policies on MATH-500, MinervaMath, and
Olympiad Bench in Table 3. Remarkably, uPRM is comparable or superior to VR and sPRM in
terms of performance of the learned policies. For example, Qwen2.5-Math-1.5B trained with uPRM
achieves a 4-point average accuracy gain across the three benchmarks compared with the same model
trained using the ground-truth verifiable reward.
4https://github.com/CJReinforce/PURE
8

## Page 9

Table 3: Accuracy on mathematical benchmarks after RL training with different reward sources.
Entries report mean ± sample standard deviation across seeds. Cells shaded in green correspond
to runs with our uPRM. Rows marked with † were evaluated at the last checkpoint before reward
hacking, i.e., prior to the end of training for at least one seed. Qwen2.5-Math-7B with sPRM and
Qwen2.5-7B with sPRM only could not be evaluated due to rapid reward hacking.
Model
Reward
MATH-500
Minerva Math
OlympiadBench
Qwen2.5-7B
VR
74.1±0.8
34.2±1.0
34.8±1.0
sPRM + VR†
75.4±0.0
29.4±4.7
36.9±0.4
uPRM†
73.2±0.4
35.0±1.3
37.5±1.1
uPRM + VR†
73.2±1.4
35.8±0.6
35.7±1.9
Qwen2.5-Math-7B
VR
80.1±0.8
35.9±0.4
41.8±0.4
uPRM
82.9±0.4
37.9±1.0
42.1±1.3
uPRM + VR
82.1±1.3
36.3±2.2
43.8±0.2
Qwen2.5-Math-1.5B
VR
70.0±0.4
26.0±0.2
33.5±1.0
sPRM†
74.7±0.3
27.8±0.6
35.0±1.0
sPRM + VR†
74.4±1.2
28.7±1.3
36.3±0.6
uPRM
73.5±1.2
31.8±0.8
36.6±0.6
uPRM + VR
74.3±0.5
31.5±0.8
35.8±0.5
Interestingly, although Cheng et al. [10] report that RH is inevitable for PRMs, we were able to
successfully complete training of Qwen2.5-Math models using just rewards from uPRM and observed
no signs of RH. In contrast, sPRM, which is the same PRM but trained via standard SFT, turned
out to be highly susceptible to hacking: training with sPRM collapsed either almost immediately
(< 50 iterations) for Qwen2.5-Math-7B or after several hundreds of training iterations for the smaller
1.5B model. For the Qwen2.5-7B base model, where both PRMs succumbed to RH before training
completed, we observe qualitatively different hacking behaviors in the learned policy under uPRM
versus sPRM. We provide a detailed analysis of this phenomenon in Appendix D.3.
6
Conclusion and Limitations
In this work, we propose a fully unsupervised approach for training PRMs that requires neither step-
level annotations nor ground-truth verification of final answers. Our experiments demonstrate that
our unsupervised PRM is competitive to supervised PRMs trained with expert annotations, thus, the
marginal cost of obtaining step-wise guidance for new domains, model families, or inference/training
pipelines can be significantly reduced. Notable experimental evidence is that strong downstream
utility does not require perfect localization of erroneous steps. Indeed, while our uPRM may lag
behind state-of-the-art supervised PRMs on error localization benchmarks such as ProcessBench,
it remains competitive in settings where PRMs actually provide value. In particular, these include
serving as verifiers in test-time scaling and as a reward source in reinforcement learning. In general,
our results reinforce the view that the direct accuracy of a reward model is an incomplete proxy for
downstream utility, consistent with recent work [54] that finds that the most accurate reward models
are not necessarily the most effective teachers.
Limitations. As we justify in Section 5.1, joint scoring is the crucial component for training a strong
PRM. Since it requires an LLM with sufficient context length to process concatenated trajectory
batches and sufficient capability to produce meaningful correctness judgments, it limits the choice
of base models. Both limitations can be mitigated by decoupling the scoring LLM from the PRM
backbone, allowing usage of more capable model to provide training signal while a smaller model
serves as the final PRM. Moreover, as context windows and capabilities of open-source LLMs
continue to grow, these constraints will naturally relax.
Beyond introducing the first fully unsupervised PRM training method, our paper identifies a practically
important robustness phenomenon (Section 5.3) that had not been highlighted before. Although we
make an effort to unravel the source of this robustness (Appendix D.3.1), fully characterizing its
precise origin is an important next step, and we see this as a promising direction opened by our work
for the broader community.
9

## Page 10

References
[1] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, et al. Towards Reasoning
Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models. arXiv
preprint arXiv:2503.09567, 2025.
[2] Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, et al. From
System 1 to System 2: A Survey of Reasoning Large Language Models. arXiv preprint
arXiv:2502.17419, 2025.
[3] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, et al. Chain-of-Thought
Prompting Elicits Reasoning in Large Language Models. Neural Information Processing
Systems, 2022.
[4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, et al. DeepSeek-R1
Incentivizes Reasoning in LLMs through Reinforcement Learning. Nature, 2025.
[5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, et al. Training
Verifiers to Solve Math Word Problems. OpenAI Technical Report, 2021.
[6] Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, et al. Kimi k1.5: Scaling
Reinforcement Learning with LLMs. arXiv preprint arXiv:2501.12599, 2025.
[7] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, et al. Solv-
ing Math Word Problems with Process- and Outcome-based Feedback.
arXiv preprint
arXiv:2211.14275, 2022.
[8] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, et al. Let’s
Verify Step by Step. In International Conference on Learning Representations, 2024.
[9] Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM Test-Time Com-
pute Optimally Can be More Effective than Scaling Parameters for Reasoning. In International
Conference on Learning Representations, 2025.
[10] Jie Cheng, Gang Xiong, Ruixi Qiao, Lijun Li, Chao Guo, et al. Stop Summation: Min-Form
Credit Assignment Is All Process Reward Model Needs for Reasoning. In Neural Information
Processing Systems, 2025.
[11] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, et al. Math-Shepherd: Verify and
Reinforce LLMs Step-by-step without Human Annotations. In Association for Computational
Linguistics, 2024.
[12] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, et al. Improve Math-
ematical Reasoning in Language Models by Automated Process Supervision. arXiv preprint
arXiv:2406.06592, 2024.
[13] Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, et al. Free Process Rewards without
Process Labels. In International Conference on Machine Learning, 2025.
[14] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, et al. Process Reinforcement
Through Implicit Rewards. arXiv preprint arXiv:2502.01456, 2025.
[15] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, et al. OlympiadBench: A
Challenging Benchmark for Pomoting AGI with Olympiad-level Bilingual Multimodal Scientific
Problems. In Association for Computational Linguistics, 2024.
[16] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, et al. Omni-MATH: A Univer-
sal Olympiad Level Mathematic Benchmark for Large Language Models. In International
Conference on Learning Representations, 2025.
[17] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al.
The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.
10

## Page 11

[18] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, et al. Qwen2.5-Math Tech-
nical Report: Toward Mathematical Expert Model via Self-Improvement. arXiv preprint
arXiv:2409.12122, 2024.
[19] Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. AlphaMath Almost Zero: Process
Supervision without Process. Neural Information Processing Systems, 2024.
[20] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, et al. URSA: Understand-
ing and Verifying Chain-of-thought Reasoning in Multimodal Mathematics. arXiv preprint
arXiv:2501.04686, 2025.
[21] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy,
et al. VinePPO: Refining Credit Assignment in RL Training of LLMs. In International
Conference on Machine Learning, 2025.
[22] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, et al. The Lessons of
Developing Process Reward Models in Mathematical Reasoning. In Association for Computa-
tional Linguistics, 2025.
[23] Jianqiao Lu, Zhiyang Dou, Hongru Wang, Zeyu Cao, Jianbo Dai, et al. AutoPSV: Automated
Process-Supervised Verifier. Neural Information Processing Systems, 2024.
[24] Bin Xie, Bingbing Xu, Yige Yuan, Shengmao Zhu, and Huawei Shen. From Outcomes to
Processes: Guiding PRM Learning from ORM for Inference-Time Alignment. In Association
for Computational Linguistics, 2025.
[25] Lin Sun, Chuang Liu, Xiaofeng Ma, Tao Yang, Weijia Lu, and Ning Wu. FreePRM: Training Pro-
cess Reward Models Without Ground Truth Process Labels. arXiv preprint arXiv:2506.03570,
2025.
[26] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wuand, et al. Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Neural Information Processing Systems,
2023.
[27] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. Length-Controlled
AlpacaEval: A Simple Way to Debias Automatic Evaluators. In Conference on Language
Modeling, 2024.
[28] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. GPTScore: Evaluate as You Desire.
In Conference of the North American Chapter of the Association for Computational Linguistics,
2024.
[29] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, et al. G-Eval: NLG Evaluation
using GPT-4 with Better Human Alignment. In Conference on Empirical Methods in Natural
Language Processing, 2023.
[30] Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, et al. Can 1B LLM Surpass 405b
LLM? Rethinking Compute-Optimal Test-Time Scaling. arXiv preprint arXiv:2502.06703,
2025.
[31] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, et al. Large
Language Monkeys: Scaling Inference Compute with Repeated Sampling. arXiv preprint
arXiv:2407.21787, 2024.
[32] Edward Beeching, Lewis Tunstall, and Sasha Rush.
Scaling Test-Time Compute
with Open Models, 2024.
URL https://huggingface.co/spaces/HuggingFaceH4/
blogpost-scaling-test-time-compute.
[33] Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, et al. GenPRM: Scaling
Test-Time Compute of Process Reward Models via Generative Reasoning. arXiv preprint
arXiv:2504.00891, 2025.
[34] Zhangyue Yin, Qiushi Sun, Zhiyuan Zeng, Qinyuan Cheng, Xipeng Qiu, et al. Dynamic and
Generalizable Process Reward Modeling. In Association for Computational Linguistics, 2025.
11

## Page 12

[35] Jiarui Yao, Ruida Wang, and Tong Zhang. PRL: Process Reward Learning Improves LLMs’
Reasoning Ability and Broadens the Reasoning Boundary. arXiv preprint arXiv:2601.10201,
2026.
[36] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, et al. Understanding
R1-Zero-Like Training: A Critical Perspective. In Conference on Language Modeling, 2025.
[37] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, et al. Rewarding
Progress: Scaling Automated Process Verifiers for LLM Reasoning. In International Conference
on Learning Representations, 2025.
[38] Leo Gao, John Schulman, and Jacob Hilton. Scaling Laws for Reward Model Overoptimization.
In International Conference on Machine Learning, 2023.
[39] Kyle Corbitt, Saumya Gandhi, Angky William, Andie Jones, Brad Hilton, et al. RULER:
Relative Universal LLM-Elicited Rewards. OpenPipe Blog, 2025.
[40] Anton Korikov, Pan Du, Scott Sanner, and Navid Rekabsaz. Batched Self-Consistency Improves
LLM Relevance Assessment and Ranking. In Conference on Empirical Methods in Natural
Language Processing, 2025.
[41] Artyom Gadetsky, Andrei Atanov, Yulun Jiang, Zhitong Gao, Ghazal Hosseini Mighan, et al.
Large (Vision) Language Models are Unsupervised In-Context Learners. In International
Conference on Learning Representations, 2025.
[42] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, et al. LoRA:
Low-Rank Adaptation of Large Language Models. In International Conference on Learning
Representations, 2022.
[43] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Self-Taught Reasoner
Bootstrapping Reasoning With Reasoning. Neural Information Processing Systems, 2022.
[44] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, et al. Beyond
Human Data: Scaling Self-Training for Problem-Solving with Language Models. Transactions
on Machine Learning Research, 2024.
[45] Brian Ziebart. Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal
Entropy. Carnegie Mellon University, 2010.
[46] Vijay Konda and John Tsitsiklis. Actor-Critic Algorithms. In Neural Information Processing
Systems, 1999.
[47] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. Qwen2.5 Technical
Report. arXiv preprint arXiv:2412.15115, 2024.
[48] Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, et al. ProcessBench:
Identifying Process Errors in Mathematical Reasoning. In Association for Computational
Linguistics, 2025.
[49] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, et al. Measuring
Mathematical Problem Solving With the MATH Dataset. In Neural Information Processing
Systems, 2021.
[50] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, et al.
Solving Quantitative Reasoning Problems with Language Models. Neural Information Process-
ing Systems, 2022.
[51] Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, et al. RLHF Workflow: From
Reward Modeling to Online RLHF. Transactions on Machine Learning Research, 2024.
[52] Jujie He, Jiacai Liu, Chris Yuhao Liu, Riu Yan, Chaojie Wang, et al. Skywork Open Reasoner 1
Technical Report. arXiv preprint arXiv:2505.22312, 2025.
12

## Page 13

[53] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, et al. Back to
Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in
LLMs. In Association for Computational Linguistics, 2024.
[54] Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D. Lee, et al. What Makes
a Reward Model a Good Teacher? An Optimization Perspective. In Neural Information
Processing Systems, 2025.
[55] Boyuan Dong, Juechu Feng, Driss Guessous, Yanbo Liang, and Horace He. FlexAttention:
A Programming Model for Generating Fused Attention Variants. In Conference on Machine
Learning and Systems, 2025.
[56] Ronald J. Williams. Simple Statistical Gradient-following Algorithms for Connectionist Rein-
forcement Learning. Machine Learning, 1992.
[57] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International
Conference on Learning Representations, 2019.
[58] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, et al. HybridFlow: A
Flexible and Efficient RLHF Framework. In European Conference on Computer Systems, 2025.
[59] Lisa Alazraki, Maximilian Mozes, Jon Ander Campos, Tan Yi-Chern, Marek Rei, et al. No
Need for Explanations: LLMs Can Implicitly Learn from Mistakes In-context. Conference on
Empirical Methods in Natural Language Processing, 2025.
[60] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, et al. SFT Memorizes,
RL Generalizes: A Comparative Study of Foundation Model Post-training. In International
Conference on Machine Learning, 2025.
[61] Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. RL’s Razor: Why Online Reinforcement
Learning Forgets Less. arXiv preprint arXiv:2509.04259, 2025.
A
Score Correction to Mitigate Degenerate Solutions
In our preliminary experiments, we observed that, although, the score usually assigns higher values to
configurations of j1, . . . , jN that are close to ground truth as desired, it also encourages degenerate
solutions due to in-context learning pathologies. We observed at least 2 degenerate solutions: setting
each jn = 1 or setting each jn = Tn + 1, corresponding to the first step labeled as erroneous
in each trajectory and the absence of erroneous steps in each trajectory, respectively. We add the
correction term to our joint score, excluding the aforementioned configurations. In particular, let
Wn = 1 + log(√Tn + 1). Intuitively, Wn measures the amount of surprise when observing the
realized value of jn. Let Sfirst(j1:N) = PN
n=1 Wn · 1[jn = 1], Slast(j1:N) = PN
n=1 Wn · 1[jn =
Tn + 1], and Scorner(j1:N) = Sfirst(j1:N) + Slast(j1:N). Let Smax = PN
n=1 Wn and B = (1 −ρ)Smax
for ρ ∈(0, 1), where B, intuitively, signifies the allowed surprise budget. Consequently, ρ defines the
amount of non-corner values of jn’s that are allowed to happen in the group j1, . . . , jN. As a result,
our correction term is defined as:
Scorrection(j1:N) = −max(0, Scorner(j1:N) −B),
(13)
where we found ρ = 0.25 is the good default choice. It corresponds to the budget B = 0.75 · Smax,
thus, the regularization is inactive unless more than 75% of the batch predictions collapse to the
corner categories. This makes it a weak safeguard against severe collapse rather than a strong bias
against general corner predictions.
We also conducted additional training runs with different values of ρ to directly assess the robustness
of our correction term. The results in Figure A1 show no significant differences in either pθ(·|τ)
entropy or model performance.
Consequently, we can redefine our joint score as follows:
S(j1:N) = S(j1:N; s1:N) + 1
N Scorrection(j1:N),
where S(j1:N; s1:N) is defined in Equation (8).
13

## Page 14

0
200
400
600
800
1000
Step
0.00
0.25
0.50
0.75
1.00
1.25
1.50
1.75
Average Entropy (p (
))
0
200
400
600
800
1000
Step
0.0
0.1
0.2
0.3
0.4
Average F1
= 0.1
= 0.25
= 0.5
Figure A1: Ablation of the non-corner budget in the correction term.
B
Optimization Details
B.1
Gradient Estimator
Recall our entropy-regularized objective from Equation (12):
max
θ
E{τn}N
n=1∼D
"
Ejn∼pθ(·|τn)S(j1:N)
i
+ γ
N
N
X
n=1
H(pθ(·|τn))
#
.
(14)
Given that the gradients for the entropy term H(pθ(·|τn)) can be easily computed using standard
automatic differentiation engines, it is only required to derive the unbiased gradient estimator for the
first term:
J (θ) = E{τn}N
n=1∼DEjn∼pθ(·|τn)S(j1:N) = E{τn}N
n=1∼DEjn∼pθ(·|τn)
N
X
n=1
S(jn|j<n),
(15)
where, for notational brevity, we write S(j1:N) = PN
n=1 S(jn|j<n) since our joint score admits
autoregressive factorization. Let’s introduce the following variables:
Gm =
N
X
n=m
S(jn|j<n), GN+1 = 0,
(16)
bimm
m (j<m) = Ejm∼pθ(·|τm)S(jm|j<m) =
Tm+1
X
j=1
S(jm = j|j<m)pθ(jm = j|τm),
(17)
where one can note that it is possible to efficiently compute bimm
m
for all m in a single forward pass of
an LLM using custom FlexAttention [55] masks. The gradient estimator, inspired by the actor-critic
framework [46], is defined as follows:
∇θJ (θ) =
N
X
m=1
"h
S(jm|j<m) −bimm
m (j<m)
i
+ (Gm+1 −Vϕ(j<m))
#
∇θ log pθ(jm|τm),
(18)
where Vϕ(j<m) is a trainable critic neural network parametrized by ϕ, and is allowed to depend
on j<m without introducing bias into the estimator. The gradient estimator has the form of REIN-
FORCE [56] with the baseline for variance reduction that directly computes the optimal immediate
part bimm
m (j<m) and employs the neural network critic Vϕ(j<m) to estimate future returns Gm+1.
To reduce the variance of the estimator even further, the critic is trained to approximate the returns
Gm+1:
Lcritic(ϕ) =
1
N −1
N−1
X
m=1
(Gm+1 −Vϕ(j<m))2.
(19)
In the following section, we describe the architecture of the critic network.
14

## Page 15

B.2
Critic Architecture
The critic Vϕ(j<m) must estimate future returns Gm+1 = PN
n=m+1 S(jn|j<n) given the history
of sampled positions j<m. We design the critic architecture with two considerations in mind: (i)
avoiding additional LLM forward passes by reusing hidden states already computed during the
joint score and PRM calculation, and (ii) leveraging privileged information about future trajectories
τm+1, . . . , τN to facilitate estimation of future returns.
Specifically, we extract two types of hidden representations:
• From the joint score computation (8), we collect the last-layer hidden state at the final token
of each marked sequence s(τn, jn), denoted hn ∈Rd. These representations encode the
history of trajectories and their sampled positions, i.e., hn = hn(j≤n). We additionally
define h0 as the hidden state at the end of the system prompt, before any trajectory is
processed.
• From the PRM forward pass, we extract the hidden state at the final special token for each
trajectory τn, denoted gn ∈Rd. These representations serve as trajectory embeddings and
constitute privileged information available only during training.
The critic combines these representations using cross-attention. Let H = [h0, . . . , hN−1] ∈RN×d
and G = [g1, . . . , gN] ∈RN×d. To compute Vϕ(j<m), we first obtain a contextualized representa-
tion of future trajectories:
Qm = Wqhm−1,
(20)
K = GWk,
V = GWv,
(21)
αm = softmax
KQm
√
d

,
(22)
Cm =
N
X
n=1
(αm)nVn,
(23)
where Wq, Wk, Wv ∈Rd×d are learnable parameters. The contextualized representation Cm ag-
gregates information about all trajectories, weighted by their relevance to the current history hm−1.
Finally, the critic value is computed as:
Vϕ(j<m) = MLP([hm−1; Cm]),
(24)
where [·; ·] denotes concatenation and MLP : R2d →R is a two-layer network with GELU activation
between the layers. Intuitively, hm−1 provide the critic with representations conditioned on j<m,
while contextualized representations Cm equip the critic with the privileged information about future
trajectories, enabling accurate prediction of Gm+1. In practice, we employ multi-head cross-attention
with 8 heads and project the hidden states from dimension d to a hidden dimension of 1024. The
attention output is passed through an output projection Wo, followed by dropout (with probability 0.1)
and layer normalization. The final value is computed by concatenating the normalized history hm−1
with the attention context Cm and passing through a two-layer MLP with GELU activation. We
apply layer normalization to H and G before the cross-attention, and do not backpropagate gradients
through the hidden states. The learnable parameters are ϕ = {Wq, Wk, Wv, Wo, MLP}.
C
Experimental and Implementation Details
C.1
Prompt Templates
We use Qwen2.5-14B-Instruct as our LLM, leveraging its chat format with system and user/assistant
turns. Below, we describe the prompt templates used for the joint score and PRM computation.
System prompt. We use the following system prompt during both the joint score and PRM calcula-
tions:
15

## Page 16

System Prompt
You are a strict mathematical reasoning judge.
Your task is to evaluate one individual reasoning step of a math problem at a
time.
- If the step is mathematically correct, respond with ‘+‘.
- If the step is mathematically incorrect or logically flawed, respond with
‘-‘.
- Do not provide any explanation, comment, or feedback - only respond with ‘+‘
or ‘-‘, and nothing else.
- Each input is either a single reasoning step or a new problem followed by
its first reasoning step.
In both cases, evaluate only the validity of the
reasoning step.
- For each new problem, once you determine that a step is incorrect, you must
consider all subsequent steps for that problem to also be incorrect, and
respond with ‘-‘ for them as well.
Your response must only be one of these two symbols:
‘+‘ or ‘-‘.
Conversation structure. We format each trajectory as a multi-turn conversation where the user
provides reasoning steps and the assistant responds with markers. Figure C1 illustrates the template
structure.
<|im_start|>system
[System prompt as above] <|im_end|>
<|im_start|>user
[Problem x] [Step y1] <|im_end|>
<|im_start|>assistant
[Marker] <|im_end|>
<|im_start|>user
[Step y2] <|im_end|>
<|im_start|>assistant
[Marker] <|im_end|>
...
Figure C1: Prompt template structure. For the joint score, markers are +/- tokens, and multiple
trajectories are concatenated sequentially, with each new problem introduced in the user turn. The
conversation for trajectory τn terminates at step jn with marker - (or continues through all steps
with + if jn = Tn + 1). For the PRM, markers are the special token [*] and all steps are included.
Furthermore, each trajectory is processed independently.
C.2
Hyperparameters for Training PRM
We apply LoRA [42] to an LLM, attaching low-rank adapters to all linear layers in the transformer.
We set LoRA rank to 64, scaling factor α = 32, and disable both bias terms and dropout. The
trainable parameters consist of the LoRA adapters, the embedding of the special token [*], and the
two-layer MLP that projects hidden states to step-level probabilities.
We train with the AdamW optimizer [57] using a constant learning rate of 10−5 for 1000 gradient
updates across 8 H200 GPUs with 8 gradient accumulation steps. Each device at each gradient
accumulation step processes a single batch of trajectories, resulting in effective batch size of 64.
Rather than fixing the number of trajectories N per batch, we pack trajectories in random order on
each GPU such that the total number of reasoning steps equals 80, resulting in approximately N = 13
trajectories on average. This number corresponds to the maximal that fits into GPU memory without
16

## Page 17

causing out-of-memory errors in our setting. If the last trajectory in a batch exceeds the remaining
budget, we truncate it. This remains valid since our objective is to identify the position of the first
mistake, and truncation only removes later steps. Importantly, fixing the total number of steps rather
than fixing N is a deliberate design choice. Indeed, since our gradient estimator (Appendix B.1)
operates at the level of individual step predictions, fixing N would result in a variable number of
steps across batches due to different trajectory lengths, leading to fluctuating signal-to-noise ratio
(SNR) in the gradients. By fixing the total step count instead, each batch contributes a consistent
number of step-level predictions to the objective function, ensuring stable SNR throughout training.
C.3
Test-time Scaling with PRM
Test-time scaling (TTS) requires a score for each candidate answer, which is used to guide and select
the final response. These scores are provided with a reward model. To compute the score for a
candidate answer, we need to define an aggregation function for the step-wise rewards assigned by a
PRM. Common aggregation functions include (i) last, where the reward for the last step is assigned
to the entire answer, (ii) product, where the product of all step-level scores is used, and (iii) min,
where the minimum step-level reward is the final reward. Multiple studies have investigated the
impact of these aggregation methods [8, 9, 22] and have found that this impact is PRM-dependent.
Further, Lightman et al. [8] shows that the difference between min and product is minor. Therefore,
we perform ablation with two strategies, last and product, using the Best-of-8 sampling strategy, and
observe that last marginally outperforms product in our uPRM; thus, we use last aggregation for all
TTS experiments in Section 5.2. We adopt search-and-learn codebase5 for our test-time scaling
experiments.
Table C1: Performance comparison between last and product aggregation in the Best-of-8 strategy
across different LLMs with our uPRM.
Policy
Agg.
MATH
Minerva
Olympiad
Avg.
500
Math
Bench
Qwen2.5-1.5B-Instruct
last
67.1±0.6
26.7±1.4
28.3±0.4
40.7
product
68.1±0.6
25.6±1.1
28.0±0.6
40.6
Qwen2.5-7B-Instruct
last
81.1±0.1
47.2±0.8
44.6±0.4
57.6
product
81.1±0.6
47.5±1.1
44.9±0.8
57.8
Llama-3.2-1B-Instruct
last
46.6±0.7
14.0±1.6
13.4±1.5
24.7
product
45.3±0.9
12.7±1.3
13.1±0.9
23.7
Llama-3.1-8B-Instruct
last
64.5±0.6
34.8±1.1
27.1±1.2
42.1
product
63.5±1.8
35.5±1.5
27.2±0.3
42.1
C.4
Reinforcement Learning with PRM
To perform RL with dense PRM rewards, we integrated our PRMs into an open-source PURE [10]
implementation based on VeRL [58]. We used the same hyperparameters as suggested in the original
work, detailed in Table C2. For each reward–policy combination, we conduct three independent runs
with different random seeds.
For the PRM + VR setting, we did not vary the coefficients before the PRM term and the VR term
and set both to 1, which results in a plain sum of the two terms. We also disabled the curriculum
learning option available in the latest PURE implementation to avoid any confounding factors and
stay as close to the original method as possible.
During evaluation, we apply greedy decoding (sampling temperature 0). In Table 3, the results are
aggregated over the last available non-degenerate models i.e., either at the end of training or at the
last saved checkpoint before RH. Qwen2.5-7B with sPRM + VR is reported over 2 seeds because one
trial failed before reaching the first checkpoint.
5https://github.com/huggingface/search-and-learn
17

## Page 18

Table C2: Training hyperparameters and optimization settings for RL experiments. For Qwen2.5-
Math models, we set a smaller generation length of 4096 due to the limited context.
Hyperparameter
Value
Epochs
4 (532 iterations)
Learning rate
10−6 (constant)
Prompt batch size
64
Group size (responses per prompt)
8
Mini-batch size
512
Maximum generation length (tokens)
8192
Sampling temperature
1.0
KL coefficient
10−3
PURE transform temperature
0.1
Save interval (iterations)
50
D
Additional Results
D.1
Ablation of Entropy Regularization Strength
We study the effect of the entropy regularization strength γ on the optimization dynamics. Figure D1
shows the average entropy H(pθ(·|τ)) and average joint score S(j1:N) throughout training for three
values of γ ∈{30, 31, 32}.
When γ is too small (γ = 1), the entropy collapses rapidly, dropping to near zero by step 400. This
premature collapse indicates that pθ converges to near-deterministic predictions early in training,
losing the ability to explore alternative positions of the first erroneous step. While this leads to the
highest joint scores, the resulting PRM may overfit to spurious patterns in the scoring function rather
than learning robust error detection. Conversely, when γ is too large (γ = 9), pθ stays nearly uniform
over candidate positions, preventing it from exploiting the signal in the joint score. The intermediate
value (γ = 3) provides a favorable trade-off between exploration and exploitation. The entropy
decreases gradually, allowing the model to concentrate probability mass on plausible error positions
while maintaining sufficient exploration to avoid premature convergence. Based on this analysis, we
use γ = 3 for all experiments reported in the main paper.
0
200
400
600
800
1000
Step
0.00
0.25
0.50
0.75
1.00
1.25
1.50
1.75
Average Entropy (p (
))
0
200
400
600
800
1000
Step
10
8
6
4
2
Average Joint Score (j1 : N)
= 30
= 31
= 32
Figure D1: Ablation of the entropy regularization strength γ. Low values cause premature entropy
collapse and overoptimization of the joint score, while γ = 3 keeps unsupervised PRM pθ(·|τ)
non-collapsed throughout the training, balancing exploration and exploitation.
D.2
Complete Results on ProcessBench
Table D1 presents the complete breakdown of results on the ProcessBench benchmark [48], including
accuracy on erroneous trajectories (Err.), accuracy on correct trajectories (Corr.), and their harmonic
mean (F1). We additionally report results on the the PRM800K dataset used for training, which serves
18

## Page 19

as a sanity check since our unsupervised PRM is expected to perform better than LLM-as-a-Judge
given it is trained on these trajectories.
The results indicate that our unsupervised PRM consistently outperforms the LLM-as-a-Judge
baseline across all datasets and metrics, achieving absolute F1 improvements ranging from +8%
on the GSM8K dataset to +14% on the OlympiadBench dataset. This confirms that optimizing the
joint score successfully distills the LLM’s evaluation capability into a more effective process reward
model.
Accuracy on Erroneous Trajectories. Detecting errors in flawed reasoning is generally more
challenging than recognizing correct solutions. The LLM-as-a-Judge baseline achieves relatively low
accuracy on erroneous trajectories, with performance degrading on harder benchmarks. Our unsu-
pervised PRM substantially improves error detection, with the largest gains on the most challenging
datasets: +15% on the Omni-MATH dataset and +13% on the OlympiadBench dataset.
Table D1: Full results on the ProcessBench dataset. We report accuracy on erroneous trajectories
(Err.), accuracy on correct trajectories (Corr.), and their aggregation via F1 score.
PRM800K
GSM8K
MATH
OlympiadBench
Omni-MATH
Err.
Corr.
F1
Err.
Corr.
F1
Err.
Corr.
F1
Err.
Corr.
F1
Err.
Corr.
F1
LLM-as-a-Judge
0.25
0.57
0.34
0.37
0.75
0.50
0.33
0.61
0.43
0.22
0.46
0.29
0.19
0.44
0.27
uPRM (ours)
0.33
0.65
0.43
0.44
0.89
0.58
0.41
0.72
0.53
0.35
0.55
0.43
0.34
0.48
0.40
Improvement
+8%
+8%
+9%
+7%
+14%
+8%
+8%
+11%
+10%
+13%
+9%
+14%
+15%
+4%
+13%
D.3
Reinforcement Learning with PRM
In this section, we provide additional analysis and results obtained in our RL experiments with PRM
as a reward source.
D.3.1
Reward Hacking Analysis
As discussed in the main text, although reward hacking can occur during training with both uPRM
and sPRM, it manifests differently for the two PRMs.
In Figure D2, we report both average length of the generated responses and KL divergence to the
reference policy for Qwen2.5-7B trained with uPRM or sPRM rewards only. One can notice that
the uPRM-trained model not only experiences RH substantially later, but also stays closer to the
reference policy and produces lengthier responses.
Deeper analysis reveals that RH induced by sPRM can be attributed to Case 3 (0 steps) according
to Cheng et al. [10], i.e., the policy learns to output empty or nonsensical responses that are nonetheless
highly awarded by the PRM. Meanwhile, training Qwen2.5-7B with uPRM eventually results in Case
2 (1 step) RH, i.e., the policy outputs a single reasoning step and stops generation. As an example,
consider the following input prompt: “What is the sum of the value(s) of n for which |2n −7| = 3?
Please reason step by step with steps separated by "\n\n" and put your final answer within \boxed{}”.
After the policy trained with sPRM collapses around iteration 30, as evidenced by a sharp drop
in the response length and increase in the KL divergence, it starts producing empty or extra short
responses (e.g., “\n\n”), completely neglecting the asked question. In contrast, RH of uPRM, which
happens around the 100th training iteration, also results in a significantly reduced response length,
but the policy output remains sensible: “To solve the equation |2n −7| = 3, we need to consider
the definition of the absolute value function, which leads to two possible cases: 2n −7 = 3 and
2n−7 = −3”. We put further discussion as well as additional plots and examples in Appendix D.3.2.
It is of interest to characterize the mechanisms underlying robustness of uPRM to trivial RH. Since
uPRM and sPRM differ only in their training procedure, while other factors like dataset or model
type are fixed, the observed effect must stem from our method. There are two potential causes: (i)
the favorable effects of unsupervised learning that prevent PRM from overfitting on specific labeling
patterns in the data [59] or (ii) an implicit bias in the learning method itself, i.e., RL vs. SFT [60, 61].
To distinguish between these possibilities, we trained an additional PRM with SFT on the PRM800K
dataset, replacing the ground-truth labels with per-step labels generated by uPRM. This ablation,
therefore, varies one axis at a time: it preserves the SFT training procedure used for sPRM while
replacing the labeling pattern with that learned by uPRM. We then used the resulting uPRM-SFT as
19

## Page 20

the reward model for RL training of Qwen2.5-7B and found that trivial RH disappeared. The policy
still hacked the reward, but instead of producing trivial responses, it generated a single long reasoning
step containing the full solution (see example in Figure D3). Such behavior closely matches Case 2
RH in the taxonomy of Cheng et al. [10], and differs conceptually from the Case 3 RH observed with
sPRM. These results suggest that susceptibility to different types of RH is inherited primarily from
the data-labeling pattern rather than from the training procedure itself.
0
50
100
150
200
250
Iteration
0
100
200
300
400
500
600
700
Response length (tokens)
uPRM
sPRM
Resp. length
KL to ref.
10
3
10
2
10
1
100
101
KL to reference model
Figure D2: Average response length in tokens and KL divergence to the reference policy for Qwen2.5-
7B trained with uPRM and sPRM as the reward source, respectively.
uPRM-SFT Generation Example
Prompt:
What is the largest whole number value of n that makes the following
inequality true?
1
3 + n
7 < 1.
Please reason step by step with steps separated
by "\n\n" and put your final answer within \boxed{}.
Completion:
We start with the given inequality:
1
3 +
n
7
<
1. First, we aim
to isolate the term involving n.
To do this, we subtract
1
3 from both sides
of the inequality:
n
7
<
1 −
1
3. Next, we need to simplify the right-hand
side.
We express 1 as a fraction with a denominator of 3:
1
=
3
3. Thus, we
have 1 −1
3 =
3
3 −1
3 =
2
3. The inequality now becomes
n
7
<
2
3. To eliminate the
fractions, we cross-multiply.
This step is valid because both
n
7 and
2
3 are
positive (assuming n is a positive integer):
n · 3
<
2 · 7. Simplifying both
sides, we get 3n
<
14. Next, we solve for n by dividing both sides of the
inequality by 3:
n < 14
3 . We then convert
14
3
to a decimal to better understand
the value:
14
3
≈4.6667. This tells us that n must be less than 4.6667.
Since
n is a whole number, the largest possible value for n is 4.
To confirm that
n
=
4 satisfies the original inequality, we substitute n
=
4 back into the
inequality:
1
3 + 4
7 < 1. We find a common denominator for the fractions
1
3 and
4
7.
The least common multiple of 3 and 7 is 21.
Thus, we rewrite the fractions
with a denominator of 21:
1
3
=
7
21 and
4
7
=
12
21. Adding these fractions, we
get
7
21 +
12
21
=
19
21. Therefore, the inequality becomes
19
21
<
1, which is true.
Consequently, n = 4 indeed satisfies the original inequality.
To ensure that
n
=
4 is the largest possible value, we check n
=
5.
Substituting n
=
5
into the inequality, we get
1
3 +
5
7
<
1. Again, we find a common denominator
of 21 for the fractions
1
3 and
5
7:
1
3 =
7
21 and
5
7 =
15
21. Adding these fractions,
we obtain
7
21 +
15
21
=
22
21. Therefore, the inequality becomes
22
21
<
1, which is
false.
Thus, n
=
5 does not satisfy the original inequality.
Consequently,
the largest whole number value of n that makes the inequality true is indeed
4.
Therefore, the final answer is
4 .
Figure D3: When training with uPRM-SFT, policy learns to hack the reward by generating the
complete solution in a single step.
20

## Page 21

D.3.2
Further Analysis and More Results
Figures D4, D5, and D6 depict panels of metrics for all three RL-training runs (with different random
seeds) of Qwen2.5 models. As stated in the main text, we consider two process reward models, uPRM
and sPRM, and the following reward options: (i) only VR, (ii) only PRM rewards, (iii) PRM rewards
+ VR on 10% of the data. We plot the following metrics (averaged over responses in the batch):
1. Accumulated PRM reward. Mathematically, this is the PURE return value for the first
step in the response, computed according to equation (6) in Cheng et al. [10]; effectively,
it approximates the minimum of per-step PRM-emitted rewards for a given response. By
definition, verifiable reward is not taken into account when computing this value, therefore
we do not plot it for the VR run. Since uPRM and sPRM yield different reward models, it
should be noted that their accumulated rewards are not directly comparable.
2. Response length. Amount of tokens in the response generated by the model for a given
input prompt.
3. KL to reference model. Kullback–Leibler divergence between the current policy and the
reference policy computed over response tokens. Reference policy is defined by the model
at initialization (zero-shot policy).
Analysis. As evidenced from the plots, Qwen2.5-Math models could be successfully trained using
uPRM both with and without VR. Training with sPRM, even in combination with VR, resulted in
reward hacking for all considered models. This can be noticed by sharp transitions in all metrics:
sudden increase in the reward and KL and drop in the response length. We found that, after RH occurs,
each model trained with sPRM (+ VR) converges to the same degenerate behavior of producing empty
or extremely short responses, like “\n\n”, which is highly rewarded by sPRM (∼0.65 accumulated
reward).
During RL training of Qwen2.5-7B, both uPRM and sPRM were exposed to RH. However, as
discussed earlier, the types of their RH differ substantially: training with sPRM results in the most
trivial Case 3 (0 step) RH according to Cheng et al. [10], while RH in uPRM can be attributed to
Case 2 (1 step).
It can be observed that occasionally, when training with the sPRM reward, another RH transition
occurs at a later stage of training: note, e.g., a drop in KL around iteration 300 for sPRM-trained
Qwen2.5-7B in Figure D4a (solid red line). Still, the type of RH is not changed essentially. The model
switches from giving empty-like responses to trivially continuing the input prompt, for example, as
follows: “You do not need to box your final answer if it is a variable or an expression” or “If you
need to use variables or special functions, define them clearly”.6
At the same time, reward hacking that happens in training Qwen2.5-7B with uPRM or uPRM + VR
is milder. The hacking behavior does not change throughout training and consists in outputting a
single but correct reasoning step, which consequentially receives a high reward from the PRM. Some
examples are provided in Table D2.
Interestingly, combining PRM rewards with VR neither necessarily improves final performance
nor delays the onset of reward hacking. For instance, for Qwen2.5-Math-1.5B in Figure D4c, RH
occurred earlier with sPRM + VR than with sPRM. Potentially, changing the weights before reward
terms or increasing the proportion of data for which VR is calculated could have an impact.
6Model inputs are created by appending a question from the dataset with the standard instruction: “Please reason step by
step with steps separated by "\n\n" and put your final answer within \boxed{}”.
21

## Page 22

Table D2: Qwen2.5-7B post-reward hacking examples for training with uPRM rewards. Sample
generations with high PRM reward that exhibit “solution-introduction” behavior: beginning a correct
derivation but not completing it.
Iter.
Question
Model generation
PRM
reward
100
A parabola with equation y = x2 + bx + c passes
through (−1, −11) and (3, 17). What is c?
“To find the value of c in the equation y
=
x2 + bx + c, we start by using the given points
(−1, −11) and (3, 17) that lie on the parabola.”
0.98
150
What is the sum of the value(s) of n for which
|2n −7| = 3?
“To solve the equation |2n −7| = 3, we need to
consider the definition of the absolute value func-
tion, which leads to two possible cases: 2n−7 = 3
and 2n −7 = −3.”
0.99
200
Find the least integer value of x for which 2|x| +
7 < 17.
“To find the least integer value of x, we start by
isolating the absolute value term. We begin with
the inequality: 2|x| + 7 < 17.”
0.97
300
Using side lengths 2, 3, 5, 7, 11, how many trian-
gles with exactly two equal sides can be formed?
“To form a triangle, the triangle inequality must be
satisfied, which states that the sum of the lengths of
any two sides must be greater than the length of the
remaining side.”
0.94
400
For how many n ∈{2, 3, . . . , 100} is 235236n a
multiple of 7?
“We start by expressing the base-n number
235236n in base-10. The number 235236n can be
expanded as 2n5 + 3n4 + 5n3 + 2n2 + 3n + 6.”
0.99
500
Solve for c: c−23
2
= 2c+5
7
.
“We start by eliminating the denominators through
cross-multiplication.
The given equation is
c−23
2
= 2c+5
7
.”
0.98
22

## Page 23

VR
uPRM
sPRM
uPRM + VR
sPRM + VR
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(a) Qwen2.5-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(b) Qwen2.5-Math-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(c) Qwen2.5-Math-1.5B
Figure D4: Training run (seed 1). Accumulated PRM reward, response length in tokens, and
KL divergence to the reference policy for Qwen2.5 models trained with different reward sources.
Accumulated reward is computed according to eq. (6) in Cheng et al. [10], and represents an
approximate minimum of per-step PRM rewards for a given response (VR is not taken into account).
Note that formally uPRM and sPRM reward values are incomparable due to different reward models.
23

## Page 24

VR
uPRM
sPRM
uPRM + VR
sPRM + VR
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(a) Qwen2.5-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(b) Qwen2.5-Math-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(c) Qwen2.5-Math-1.5B
Figure D5: Training run (seed 2). Accumulated PRM reward, response length in tokens, and
KL divergence to the reference policy for Qwen2.5 models trained with different reward sources.
Accumulated reward is computed according to eq. (6) in Cheng et al. [10], and represents an
approximate minimum of per-step PRM rewards for a given response (VR is not taken into account).
Note that formally uPRM and sPRM reward values are incomparable due to different reward models.
24

## Page 25

VR
uPRM
sPRM
uPRM + VR
sPRM + VR
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(a) Qwen2.5-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(b) Qwen2.5-Math-7B
0
100
200
300
400
500
Iteration
0.6
0.4
0.2
0.0
0.2
0.4
0.6
0.8
1.0
PRM reward (accum.)
0
100
200
300
400
500
Iteration
0
200
400
600
800
Response length (tokens)
0
100
200
300
400
500
Iteration
10
3
10
2
10
1
100
101
KL to reference model
(c) Qwen2.5-Math-1.5B
Figure D6: Training run (seed 3). Accumulated PRM reward, response length in tokens, and
KL divergence to the reference policy for Qwen2.5 models trained with different reward sources.
Accumulated reward is computed according to eq. (6) in Cheng et al. [10], and represents an
approximate minimum of per-step PRM rewards for a given response (VR is not taken into account).
Note that formally uPRM and sPRM reward values are incomparable due to different reward models.
25
