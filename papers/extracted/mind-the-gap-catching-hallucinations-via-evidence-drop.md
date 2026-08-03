---
source_pdf: papers/Mind the Gap -  Catching Hallucinations via Evidence Drop on the Reasoning.pdf
slug: mind-the-gap-catching-hallucinations-via-evidence-drop
pages: 25
extracted_on: 2026-08-03
---

# Mind the Gap -  Catching Hallucinations via Evidence Drop on the Reasoning

## Page 1

Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning
Manifold
Qunjie Chen 1 Yufei Chen 1 † Xiaodong Yue 2 Linye Li 1
Abstract
Large Language Models (LLMs) show strong
reasoning abilities, yet their reliability is hin-
dered by hallucinations, where ﬂuent reason-
ing becomes factually or logically incorrect.
Most existing uncertainty-based detectors rely on
sequence-level averaging, which ignores the step-
wise dynamics of reasoning and often misclas-
siﬁes hard-but-correct or easy-but-wrong sam-
ples.
We propose a dynamic perspective that
models reasoning as a trajectory on a latent Ev-
idence Manifold, where each step is supported
by local evidence.
Hallucinations are charac-
terized as Evidence Drops, i.e., sudden declines
in local evidence support that indicate topolog-
ical deviations from this manifold.
Based on
this insight, we design a training-free and model-
agnostic detector that identiﬁes hallucinations
via the worst-case Evidence Drop and enables
step-level error localization.
Experiments on
GSM8K, MATH, and ProcessBench show con-
sistent improvements over sequence-level uncer-
tainty baselines in selective accuracy and riskcov-
erage trade-offs.
1. Introduction
Large Language Models (LLMs) have demonstrated re-
markable capabilities in complex reasoning tasks through
Chain-of-Thought (Wei et al., 2022).
Nevertheless, the
reliability of the reasoning process is substantially under-
mined by the problem of hallucinations (Ji et al., 2023;
Turpin et al., 2023; Cheng et al., 2025; Shi et al., 2026),
in which models produce reasoning that is linguistically co-
herent yet factually incorrect. To address this challenge,
recent studies have explored a wide range of signals for hal-
1School of Computer Science and Technology, Tongji Univer-
sity 2Artiﬁcial Intelligence Institute, Shanghai University. Corre-
spondence to: Yufei Chen <yufeichen@tongji.edu.cn>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
Reliable reasoning steps
with smooth manifold
Hard but Correct
Easy but Wrong
Hard but Correct: high 
average uncertainty, no drop
Reasoning Steps Manifold
Uncertainty Level
Easy but Wrong: low average 
uncertainty with sudden drop
Avg: 0.5 > 𝜏!"#
Avg:  0.48 < 𝜏!"#
𝜏!"#
Figure 1. Dynamics of LLMs on the proposed latent evidence
manifold. Previous methods rely on sequence-level average uncer-
tainty, which incorrectly ﬂags hard but correct samples as halluci-
nations because their averaged uncertainty exceeds the threshold
τavg, while overlooking easy but wrong samples whose average un-
certainty is low despite exhibiting sharp uncertainty transitions.
lucination detection, including internal hidden states (Yang
et al., 2026b; 2025c;d), attention patterns, and output proba-
bility distributions (Bazarova et al., 2025; Kim et al., 2025;
Zhang et al., 2025; Sriramanan et al., 2024; Orgad et al.,
2024). Among them, a predominant line of research fo-
cuses on logit-based uncertainty estimation as a principled
signal for hallucination detection, leveraging the models in-
trinsic probabilistic structure to identify unreliable genera-
tions (Malinin & Gales, 2020; Farquhar et al., 2024).
However,
existing uncertainty-based hallucination de-
tection methods predominantly adopt an average-over-
sequence paradigm, in which uncertainty is represented as
a single sequence-level score obtained by aggregating un-
certainty signals across all tokens or reasoning steps, rather
than being modeled as a structured, step-wise process. For
instance, Length-Normalized Scoring (LN-S) (Malinin &
Gales, 2020), Semantic Entropy (Farquhar et al., 2024), and
LogTokU (Ma et al., 2025) quantify uncertainty by summa-
rizing token-level or semantic-level statistics over the en-
tire output sequence into a single scalar measure. Such
sequence-level aggregation inevitably obscures the ﬁne-
grained dynamics of uncertainty evolution during multi-
step reasoning and masks how errors emerge and propagate
across individual steps (Lightman et al., 2023).
1

## Page 2

Mind the Gap: Catching Hallucinations via Evidence Drop
In this work, we propose a paradigm shift from static ag-
gregation to dynamic process monitoring. We posit that
a valid reasoning process can be viewed as a trajectory
evolving on a coherent Evidence Manifold, where each
intermediate step is supported by a latent evidence state
that provides the internal support structure of reasoning:
it grounds each step, maintains logical consistency across
transitions, and ensures the continuity of the reasoning tra-
jectory.
When a hallucination occurs, this continuity is
disrupted. The model transitions into a region where the
inferred reasoning is no longer sufﬁciently supported by
coherent evidence, leading to a breakdown in the underly-
ing evidence structure. Consequently, a hallucination is not
merely a low-probability event or an instance of high un-
certainty, but rather a topological deviation from the man-
ifold. We theoretically characterize such deviations as an
Evidence Drop, deﬁned as a sudden and sharp decline in
the models local evidence support between consecutive rea-
soning steps. Speciﬁcally, we quantify local evidence mass
using token-level uncertainty signals and detect hallucina-
tions by identifying their abrupt deviations, rather than ag-
gregating uncertainty in an average manner over the entire
sequence. To this end, we further integrate our metric with
a hypothesis testing framework, which enables us to derive
a statistically grounded decision threshold. This framework
provides ﬁnite-sample guarantees and strictly controls the
rate of accepted hallucinations (Type I error) below a user-
speciﬁed signiﬁcant level, ensuring reliability in practical
deployment. Our main contributions are summarized as fol-
lows:
• We model multi-step reasoning as a trajectory on
an Evidence Manifold and characterize hallucinations
as topological deviations, identiﬁed by sudden local
drops in evidence support.
• We propose a training-free, model-agnostic hallucina-
tion detector based on the worst-case Evidence Drop,
compatible with various uncertainty measures and ca-
pable of step-level error localization.
• Experiments on GSM8K, MATH, and ProcessBench
demonstrate consistent improvements over sequence-
level uncertainty baselines in selective accuracy and
riskcoverage trade-offs.
2. Related Work
2.1. Uncertainty Estimation in LLMs
Quantifying uncertainty in LLMs is a central problem for
building reliable and trustworthy systems.
Existing ap-
proaches can be broadly categorized into logits-based and
consistency-based methods. Logits-based methods derive
uncertainty directly from the models output distribution,
such as token probabilities, entropy, or related scoring func-
tions (Malinin & Gales, 2020; Farquhar et al., 2024; Ma
et al., 2025). These approaches are attractive due to their
simplicity and their tight connection to the probabilistic
formulation of language models. Consistency-based meth-
ods, on the other hand, measure uncertainty through vari-
ability across multiple sampled generations, such as Self-
Consistency (Wang et al., 2022) and its extensions, which
interpret disagreement among sampled reasoning paths or
answers as a signal of unreliability.
However, most ex-
isting uncertainty estimation methods ultimately summa-
rize uncertainty into a single scalar score at the sequence
level, typically by averaging or aggregating token- or step-
wise statistics. While effective for coarse-grained reliabil-
ity assessment, such static formulations overlook how un-
certainty evolves during multi-step reasoning and cannot
reveal localized failures within the reasoning process.
2.2. Process Monitoring and Step-level Veriﬁcation
Beyond token-level supervision, a growing body of work
has emphasized the importance of monitoring the reason-
ing process itself. Process Reward Models (PRMs) (Light-
man et al., 2023) train separate veriﬁers to score the cor-
rectness of each intermediate step in a reasoning chain,
and have demonstrated strong empirical performance in im-
proving reasoning reliability. Related approaches include
training step-level critics or veriﬁers for mathematical and
logical reasoning (Cobbe et al., 2021b; Lightman et al.,
2023; Wang et al., 2024; Luo et al., 2024), as well as meth-
ods that analyze internal activations or hidden states to de-
tect errors (Azaria & Mitchell, 2023). Despite their effec-
tiveness, these methods typically require step-level annota-
tions and additional supervised training, which can be ex-
pensive and difﬁcult to scale across domains and tasks.
3. The Proposed Method
We formalize the reasoning process as a two-level stochas-
tic system consisting of latent evidence states and observ-
able uncertainty signals.
The latent level captures the
abstract evidence supporting each underlying reasoning
step, while the observable level corresponds to uncertainty-
related quantities derived from the models output distribu-
tion. This distinction allows us to deﬁne hallucinations as
topological violations in the latent evidence space that man-
ifest as statistical anomalies in the observation space.
3.1. Markov Dynamics on the Evidence Manifold
Latent Evidence Manifold.
We assume the reasoning
process is driven by a sequence of latent evidence states,
E = (E1, E2, . . . , ET ), where each Ei ∈M lies on a low-
dimensional manifold M, referred to as the Evidence Man-
ifold. The variable Ei represents the abstract internal evi-
2

## Page 3

Mind the Gap: Catching Hallucinations via Evidence Drop
dence support associated with the i-th reasoning step and
T denotes the total number of reasoning steps (or tokens)
in the sequence. This assumption is motivated by the man-
ifold hypothesis, which posits that high-dimensional struc-
tured data concentrate near low-dimensional manifolds. In
our context, M captures the intrinsic structural constraints
that characterize coherent and logically consistent reason-
ing trajectories. We model the evolution of evidence states
as a Markov process on M,
Ptrue(E1, . . . , ET ) = P(E1)
T −1
Y
i=1
Ptrue(Ei+1 | Ei),
(1)
which implies that valid reasoning corresponds to a lo-
cally smooth transmission of evidence along the manifold.
While Ei is latent and unobservable, in practice the models
belief at step t is expressed through a predictive distribution
with a ﬁnite effective support, induced by decoding strate-
gies such as top-K sampling.
Observation Model: Uncertainty Signals.
The latent
evidence states {Ei} are not directly observable.
In-
stead, we observe a sequence of uncertainty signals U =
(U1, U2, . . . , UT ), where each Ui is computed from the
models output distribution at step i. Concretely, Ui may
correspond to token-level negative log-likelihood, entropy,
or other logit-derived uncertainty measures. We treat all
such quantities uniformly as observable realizations of an
underlying evidence state. We assume that the uncertainty
signal Ui is generated from the latent evidence state Ei
through an observation model
Ui = g(Ei) + εi,
(2)
where g(·) is an unknown (possibly nonlinear) observation
function and εi represents observation noise. Unlike exist-
ing methods that operate directly on {Ui}, we treat uncer-
tainty as an indirect and noisy observation of an underlying
latent evidence process.
Locality of Evidence. Inspired by the ﬁndings of (Prys-
tawski et al., 2023) that the effectiveness of reasoning
comes from the local statistical structure of the training
data, we propose that reasoning is fundamentally enabled
by the local structure of evidence. In many real-world sce-
narios, training data does not contain all variables jointly
but only presents them in overlapping local neighbors. For
example, one subset of observations may capture the rela-
tionship Weather →Road Condition →Commuting Time,
while another subset contains Road Condition →Trafﬁc Ac-
cident →Rescue Time. Although Weather and Rescue Time
never co-occur in the training data, their relationship can
be inferred by chaining accurate local inferences through
the shared variable Road Condition. This illustrates that
global reasoning emerges from the composition of reliable
local transitions. We refer to this principle as the locality of
evidence: the training data Ptrain only constrains local tran-
sitions between adjacent evidence states, while long-range
reasoning is achieved by chaining these locally consistent
transitions along a trajectory. Based on this principle, we
characterize the training distribution Ptrain through a local
observation constraint, which enforces that the model only
observe and learn transitions between adjacent evidence
states on the latent evidence manifold.
This constraint
transfers the topological structure of the manifold directly
onto the observation space.
Assumption 1 (Locality of Transitions). For any two non-
adjacent evidence states Ei and Ej with |i −j| > 1, the
training data contains no direct transitions between them:
Ptrain(Uj | Ui) = 0.
(3)
Assumption 2 (Local Consistency). For adjacent evidence
states Ei and Ei+1, the observed transition distribution pre-
serves the true local transition structure on the evidence
manifold:
Ptrain(Ui+1 | Ui) ∝Ptrue(Ei+1 | Ei).
(4)
3.2. Evidence Drop
Modern large language models explicitly regulate the en-
tropy of their predictive distributions to balance exploration
and exploitation, prevent premature mode collapse, and en-
sure training stability. Such entropy control mechanisms
have been widely adopted in recent LLMs, including tem-
perature scheduling, entropy penalties, and KL-regularized
objectives (e.g., in pretraining or RL stages) (Yang et al.,
2025a;b; He et al., 2025; Cui et al., 2025). Accordingly,
their learning objective can be abstracted into a generic
entropy-regularized risk minimization form:
R(q) = EPtrain[−log q(U)]
|
{z
}
data ﬁtting
+λ EUnif(V)[−log q(U)]
|
{z
}
entropy regularization
. (5)
where q(U) denotes the models predictive distribution over
the uncertainty observation U, Unif(V) is the uniform dis-
tribution over the vocabulary (or token space) V, and λ > 0
controls the strength of entropy regularization. This for-
mulation captures the trade-off between ﬁtting the training
data and maintaining sufﬁcient predictive entropy.
Under this setting, we derive the following theorem charac-
terizing the model behavior under topological violations of
the evidence manifold.
Theorem 3.1. Let q∗be the optimal estimator minimizing
the risk R(q). For any attempt to transition between two
non-adjacent evidence states (Ei, Ek) with |i −k| > 1
3

## Page 4

Mind the Gap: Catching Hallucinations via Evidence Drop
(i.e., an off-manifold or topological deviation), the optimal
predictive distribution over the uncertainty observation de-
generates to the uniform distribution:
q∗(U | Ei) = Unif(V) = 1
|V|.
(6)
Consequently, the corresponding evidence measure col-
lapses to its theoretical minimum:
Ehallucination ≈log
 K
|V|

≪0,
(7)
where K denotes the top-K sampling during inference.
The full proof is provided in Appendix B.
Intuitively, Theorem 3.1 states that due to Assumption 1,
only transitions between adjacent evidence states are sup-
ported by empirical co-occurrence statistics in the training
data Ptrain. For such adjacent pairs, the data-ﬁtting term
pulls q toward the true conditional distribution, and the en-
tropy regularization term pushes q toward the uniform dis-
tribution; the optimal solution for adjacent pairs is there-
fore a convex combination of the two, yielding a smoothed
but still informative predictive distribution. In contrast, for
non-adjacent evidence states (Ei, Ej) with |i −j| > 1,
no joint observations are available under the locality con-
straint. Consequently, the data-ﬁtting term vanishes, and
the risk reduces solely to the entropy regularization term.
Minimizing this term forces the optimal predictor to de-
generate to the uniform distribution Unif(V), which rep-
resents complete uncertainty and the absence of supporting
evidence.
Therefore,
an off-manifold transition is intrinsically
mapped to a state of maximal entropy and minimal evi-
dential support. This reveals a fundamental asymmetry be-
tween valid reasoning and hallucination:
• On-Manifold (Valid Reasoning). For adjacent tran-
sitions, Ptrain(Ei+1 | Ei) is nonzero, and the data-
ﬁtting term dominates the risk, yielding a concen-
trated predictive distribution and stable evidence, i.e.,
Evalid ≈0.
• Off-Manifold (Hallucination).
For non-adjacent
transitions, Ptrain vanishes due to locality. The risk is
governed solely by entropy regularization, forcing the
predictor toward the uniform distribution over V and
causing evidence collapse.
As a result, a pronounced Evidence Drop emerges:
∆= Ehallucination −Evalid ≈log K −log |V| ≪0. (8)
This result establishes that any off-manifold transition nec-
essarily induces a negative evidence drop, i.e., ∆< 0.
Hence, hallucinations are not identiﬁed by low absolute ev-
idence values, but by the occurrence of evidence decreases.
Accordingly, we only track positions in the reasoning tra-
jectory where evidence drops appear.
3.3. Tracking the Evidence Drop
Guided by Theorem 3.1, our objective is not merely to
quantify static uncertainty, but to detect the dynamic phe-
nomenon we term the Evidence Drop. This corresponds to
a sharp, non-smooth disruption of the topological support
of the latent evidence manifold during the reasoning pro-
cess. Since the latent evidence states E are not directly ob-
servable, an Evidence Drop cannot be measured explicitly.
Instead, leveraging Assumption 2, we utilize observable un-
certainty dynamics as an empirical proxy for the latent evi-
dence manifold. Speciﬁcally, we construct a surrogate mea-
sure based on the models predictive distribution and track
its temporal evolution along the reasoning trajectory.
1. Entropy as an Empirical Proxy for Latent Evidence.
To make the latent evidence Et tractable, we adopt Shan-
non entropy (Shannon, 1948) as our primary uncertainty
measure, deﬁning its negative as a proxy for evidence.
While alternative metrics exist (e.g., LogTokU (Ma et al.,
2025), LN-S (Malinin & Gales, 2020)), they are often
heuristic and lack direct alignment with the training ob-
jective in Eq. 5. In contrast, Shannon entropy is explic-
itly coupled with the entropy regularization term in the risk
function, admitting a principled interpretation within our
theoretical framework.
We deﬁne the empirical evidence ˆEi at step i using the
next-token predictive distribution. To mitigate the inﬂuence
of the long-tail vocabulary noise, we compute the entropy
over a renormalized top-K distribution. Let V(t)
K be the set
of K tokens with the highest probabilities under P(·|y<t).
The restricted distribution ˜P is deﬁned as:
˜P(v | y<i) =
P(v | y<i)
P
v∈V(t)
K P(v | y<i).
(9)
Accordingly, the empirical evidence proxy is:
ˆEi := −H( ˜P(·|y<i)) =
X
v∈V(t)
K
˜P(v|y<i) log ˜P(v|y<i).
(10)
By deﬁning ˆEi as the negative entropy, we align it with
the physical interpretation of evidence: higher values in-
dicate stronger evidential support (a concentrated distribu-
tion), while lower values signify a collapse in certainty.
2. Identifying the Evidence Drop. Our core premise is
that valid reasoning preserves a smooth evolution of evi-
dence, whereas hallucinations manifest as abrupt structural
collapses. To capture this behavior, we analyze the local
4

## Page 5

Mind the Gap: Catching Hallucinations via Evidence Drop
gradient of a smoothed evidence trajectory. We ﬁrst com-
pute a smoothed sequence ˜E = { ˜E1, . . . , ˜ET } via an ex-
ponential moving average (EMA) over the raw proxy se-
quence ˆE. For each time step j > 1, the evidence ﬂux is
deﬁned as:
∆j = ˜Ej −˜Ej−1.
(11)
An Evidence Drop is characterized by a signiﬁcant nega-
tive gradient, ∆j ≪0, indicating a sudden loss of probabil-
ity mass concentration. To isolate these deleterious events,
we consider the set of negative ﬂuctuations D−= {∆j |
∆j < 0}. Let ∆(1) ≤∆(2) ≤. . . denote the elements of
D−sorted in ascending order of magnitude. The ﬁnal risk
score ϕ(x, y) is formulated by aggregating the M most se-
vere drops:
ϕ(x, y) = −1
M
M
X
j=1
∆(j),
(12)
where M is a hyperparameter (we set M = 5 in our exper-
iments). A higher risk score indicates a more pronounced
structural violation of the evidence manifold, suggesting a
higher likelihood that the reasoning process has deviated
into a hallucinated state.
4. Experiments
In this section, we conduct extensive experiments to evalu-
ate the effectiveness, reliability, and interpretability of our
proposed framework. It is worth noting that our framework
operates in a strictly training-free method, requiring no gra-
dient updates or auxiliary reward models. Speciﬁcally, we
aim to answer the following Research Questions (RQs):
• RQ1
(Selective
Performance):
Does
the
de-
tection of dynamic Evidence Drops yield supe-
rior selective prediction performance and a better
accuracy-rejection trade-off compared to conventional
sequence-averaging uncertainty baselines?
• RQ2 (Theoretical Alignment): Is the efﬁcacy of the
Evidence Drop framework inherently tied to Shannon
entropy due to its alignment with the optimization
evidence manifold, and does it outperform heuristic
uncertainty measures that lack such theoretical cou-
pling?
• RQ3 (Error Localization): Can the sharp gradients
in the evidence trajectory faithfully pinpoint the spe-
ciﬁc step where a reasoning deviation (hallucination)
occurs within complex Chain-of-Thought traces?
Datasets. We evaluate our framework across two distinct
granularities of reasoning:
• Sequence-level
Evaluation:
We
ﬁrst
utilize
GSM8K (Cobbe et al., 2021a) and MATH (Hendrycks
et al., 2021) to assess the model’s ability to detect
overall reasoning failure. These benchmarks require
multi-step logical derivation, where we compute
the aggregate uncertainty (or maximum evidence
drop) over the entire trajectory to perform selective
prediction on the ﬁnal answer.
• Step-level Evaluation: We further employ Process-
Bench (Zheng et al., 2025), a specialized diagnos-
tic benchmark encompassing tasks from GSM8K,
MATH, OlympiadBench (He et al., 2024) and Omni-
MATH (Gao et al., 2024) which provides human-
annotated labels for each individual reasoning step,
identifying exactly where a logical fallacy occurs.
This allows us to evaluate whether the Evidence Drop
precisely aligns with the speciﬁc onset of reasoning er-
rors, moving beyond mere sequence-level correlation
to true stepwise localization.
Implementation Details. We employ the Qwen3-4B and
8B models (Yang et al., 2025a) as our primary reasoning
backbones. All inference experiments are conducted on a
cluster of four NVIDIA A100 GPUs (40GB). During the
generation process, we utilize greedy decoding with tem-
perature τ = 0, and set the nucleus sampling threshold to
p = 0.95. The top-K vocabulary constraint is ﬁxed at K =
20, which directly corresponds to the hyperparameter K
utilized in the evidence estimation in Eq. 10. For the step-
level diagnostics on ProcessBench, we adopt a teacher-
forcing (or forced decoding) approach to extract the predic-
tive distributions along the reasoning traces. Speciﬁcally,
given a pre-deﬁned reasoning chain, we perform a single
forward pass to compute the logit distributions for each to-
ken transition. This allows us to accurately map the step-
wise Evidence Drop to the speciﬁc human-annotated log-
ical segments, ensuring that the uncertainty dynamics are
calculated over the exact reasoning path intended for eval-
uation. Our baselines consist of several uncertainty mea-
sures, including Shannon Entropy, LN-S (Malinin & Gales,
2020), and LogTokU (Ma et al., 2025). The detailed com-
putation procedures are provided in Appendix D.
Selective Prediction and Calibration. To evaluate the re-
liability of the uncertainty estimates, we adopt a selective
prediction (rejection) framework. A reasoning trajectory
is deemed unreliable and subsequently rejected if its risk
score ϕ(x, y) (or the baseline uncertainty) exceeds a de-
cision threshold ˆτ. To determine a statistically principled
threshold, we formulate the task as a hypothesis testing
problem under the Neyman-Pearson paradigm. We ﬁrst par-
tition the test data into two disjoint subsets of equal size: a
Calibration Set (Dcal), used to compute ˆτ, and an Evalua-
tion Set (Deval), used for reporting ﬁnal performance. On
Dcal, we compute the distribution of risk scores for samples
where the model produces incorrect ﬁnal answers. Given a
5

## Page 6

Mind the Gap: Catching Hallucinations via Evidence Drop
pre-deﬁned signiﬁcance level α ∈(0, 1), we set ˆτ as the
(1 −α)-quantile of the risk distribution. This procedure
ensures that the rejection mechanism provides a controlled
level of error detection, allowing for a rigorous compari-
son between our Evidence Drop metric and baseline uncer-
tainty measures across various operating points.The whole
process can be seen in Appendix C.
Metrics.
We utilize a multi-dimensional evaluation suite
to assess the effectiveness of our framework:
• Selective Accuracy (Acc): This measures the model’s
reliability on the subset of queries it chooses to answer.
Speciﬁcally, for a given threshold ˆτ, let Daccepted =
{(x, y) ∈D | ϕ(x, y) ≤ˆτ}. Selective Accuracy is
deﬁned as:
Acc = |Daccepted ∩Dcorrect|
|Daccepted|
.
(13)
• Area Under the Risk-Coverage Curve (AURC): To
evaluate the quality of the uncertainty ranking with-
out dependence on a speciﬁc threshold ˆτ, we report
the AURC (Geifman & El-Yaniv, 2017). The Risk-
Coverage curve plots the error rate (Risk) against the
fraction of samples accepted (Coverage).
A lower
AURC indicates that the risk score effectively assigns
higher values to incorrect predictions, allowing for a
better trade-off between performance and abstention.
• Step-level Localization Accuracy (SLA): Evaluated
speciﬁcally on ProcessBench, this metric assesses the
model’s ability to pinpoint the exact onset of reason-
ing failure. We deﬁne a successful localization if the
ﬁrst signiﬁcant Evidence Drop (i.e., the ﬁrst step t
where ∆t exceeds a step-wise threshold) matches the
human-annotated ﬁrst erroneous step in the CoT trace.
This measures the diagnostic precision of our method
in complex, multi-step derivations.
4.1. Comparison with Sequence-Averaging Baselines
(RQ1)
We evaluate the efﬁcacy of the Evidence Drop across mul-
tiple model scales (4B, 8B) and mathematical reasoning
benchmarks. The results, summarized in Table 1, demon-
strate a consistent and signiﬁcant advantage in utilizing dy-
namic drops over static sequence averages for selective pre-
diction. The experimental results conﬁrm that monitoring
the Evidence Drop is substantially more effective at identi-
fying reasoning errors than traditional sequence-averaging
methods.
• Signiﬁcant Gains in Selective Accuracy: Across al-
most all conﬁgurations, the “Drop” variant of a metric
outperforms its “Avg” or "Base" counterpart. For in-
stance, on the MATH dataset with Qwen3-8B (α =
0.05), the Shannon Drop achieves a selective accu-
racy of 88.26%, representing a remarkable +18.02%
improvement over Shannon Avg (70.24%).
• Rescuing Weak Baselines: The most dramatic im-
provement is observed in LogTokU. On GSM8K
(α = 0.05), LogTokU Avg fails to provide a mean-
ingful decision threshold (yielding near 0% accuracy
in early iterations), whereas LogTokU Drop recovers
this signal to achieve 85.57% accuracy. This suggests
that the absolute magnitude of evidence is often less
informative than the sudden loss of evidence during a
reasoning step.
• Robustness Across Complexity: While the baseline
accuracy on MATH is signiﬁcantly lower (∼59%)
than on GSM8K (∼91%), the Evidence Drop main-
tains high selective accuracy (often > 80%). This in-
dicates that the mechanism effectively ﬁlters out hal-
lucinated derivations even when the underlying task
difﬁculty increases.
To understand why our method works, we analyze the un-
certainty distributions in Figure 2. Separability, The his-
togram for our method (Figure 2f shows a more distinct
separation between valid (green) and hallucinated (red) re-
sponses compared to Shannon Entropy (Figure 2c). This
conﬁrms our theoretical posit: hallucinations manifest as
distinct drops rather than just generically high entropy.
Further, as demonstrated in Table 2, the proposed Evidence
Drop mechanism consistently achieves lower AURC val-
ues than static sequence-averaging baselines, conﬁrming
its superior capacity for threshold-free error ranking across
diverse model scales and task complexities. In particular,
Shannon Drop emerges as the most potent error detec-
tor, notably reducing the AURC on the MATH benchmark
for Qwen3-8B from 288.8 to 190.0, a signiﬁcant improve-
ment that underscores the diagnostic value of monitoring
dynamic manifold collapses over absolute uncertainty lev-
els. This trend is further mirrored in the LogTokU Drop
variant, which substantially mitigates the poor ranking per-
formance of its static counterpartfor instance, lowering the
MATH-4B AURC from 481.3 to 299.1thereby validating
our premise that sudden disruptions in evidential support
are more indicative of reasoning deviations than cumula-
tive probability densities. Collectively, these results signify
that aligning uncertainty dynamics with the model’s opti-
mization manifold provides a robust and scalable frame-
work for reliable selective prediction in multi-step logical
derivations.
6

## Page 7

Mind the Gap: Catching Hallucinations via Evidence Drop
Table 1. The accuracy performance (%) on GSM8k and MATH datasets. Subscripts denote the performance Drop (∆) relative to the
respective baseline (red for gain, green for loss).
Model
Dataset
Pretrained
LN-S
LN-S Drop
LogTokU Avg
LogTokU Drop
Shannon Avg
Shannon Drop
α
Qwen3-4B
GSM8K
87.63 ± 0.31
95.45
83.33−12.12
0
85.57+85.57
92.31
100+7.69
0.05
87.63 ± 0.31
90.70
89.83−0.87
44.44
87.70+43.26
91.53
100+8.47
0.10
87.63 ± 0.31
91.39
87.99−3.40
65.48
89.17+23.69
90.87
90.51−0.36
0.50
MATH
59.24 ± 0.21
73.99
82.17+8.18
17.19
67.35+50.16
71.43
84.94+13.51
0.05
59.24 ± 0.21
74.94
75.76+0.82
24.11
74.59+50.48
74.72
83.85+9.13
0.10
59.24 ± 0.21
69.17
63.14−6.03
57.04
72.44+15.40
68.76
72.56+3.80
0.50
Qwen3-8B
Gsm8k
91.07 ± 0.26
98.68
94.44−4.24
0
96.40+96.40
98.57
95.51−3.06
0.05
91.07 ± 0.26
96.95
93.48−3.47
66.67
92.73+26.06
95.75
96.99+1.24
0.10
91.07 ± 0.26
95.01
90.13−4.88
80.15
92.27+12.12
97.06
93.72−3.34
0.50
MATH
59.24 ± 0.21
69.79
86.94+17.15
30.00
81.55+51.55
70.24
88.26+18.02
0.05
59.24 ± 0.21
70.85
84.69+13.84
42.52
81.12+38.60
70.72
85.81+15.09
0.10
59.24 ± 0.21
72.81
71.84−0.97
63.41
74.91+11.50
72.28
77.13+4.85
0.50
Qwen3.5-27B
MATH
76.00
70.37
86.59+10.59
68.83
93.14+17.14
62.50
92.92+16.92
0.05
Table 2.
Comparison of AURC (×1000) across different un-
certainty estimation methods on GSM8K and MATH datasets.
Lower is better.
Model
Dataset
LN-S
LogTokU
Shannon
Base
Drop
Avg.
Drop
Avg.
Drop
Qwen3-4B
GSM8K
82.7
125.7
224.2
117.2
81.1
77.1
MATH
302.4
327.3
481.3
299.1
303.1
239.1
Qwen3-8B
GSM8K
45.4
92.8
155.8
73.8
41.9
48.3
MATH
283.2
246.2
402.8
233.7
288.8
190.0
4.2. Synergy with Shannon Entropy (RQ2)
The empirical experiments also support our theoretical in-
tuition that the Evidence Drop is most potent when paired
with Shannon Entropy, which is explicitly aligned with
the model’s optimization manifold.
• Optimal Performance with Shannon Entropy: In
the Qwen3-4B GSM8K trials (α = 0.05 and 0.10),
Shannon Drop achieves a perfect or near-perfect se-
lective accuracy of 100%. No other uncertainty proxy
reaches this ceiling. This validates the premise that
because Shannon entropy is coupled with the entropy
regularization in the training objective (Eq. 5), its
ﬂuctuations are the most reliable indicators of on-
manifold versus off-manifold transitions.
• Comparison with Heuristic Proxies: While Log-
TokU Drop shows massive gains, its absolute perfor-
mance often lags behind Shannon Drop.
Similarly,
LN-S Drop frequently shows performance degrada-
tion (noted by green subscripts) or marginal gains.
This is likely because LN-S focuses only on the prob-
ability of the single most likely token, whereas Shan-
non Entropy considers the entire distribution, captur-
ing the subtle probability mass shifts that precede a
Table 3.
Step-level Localization Accuracy (%) on different
datasets across multiple model scales.
Model
Dataset
LN-S
LogTokU
Shannon
Avg
Drop
Avg
Drop
Avg
Drop
Qwen3-4B
GSM8K
31.24
27.94
28.76
23.19
27.94
43.42
MATH
28.39
24.17
21.77
23.93
24.17
32.03
OlympiadBench
26.22
24.93
28.57
22.62
24.95
43.06
Omni-MATH
23.21
23.60
26.87
22.96
23.67
38.04
Qwen3-8B
GSM8K
32.34
34.14
27.87
29.45
27.66
46.11
MATH
26.91
25.71
21.77
20.27
24.62
32.90
OlympiadBench
26.48
26.29
18.74
18.75
26.30
41.52
Omni-MATH
23.35
23.33
21.10
21.15
23.40
37.04
structural collapse in reasoning.
• Consistency across Conﬁdence Levels:
Shannon
Drop remains the most stable metric as the rejection
threshold α varies.
Even at α = 0.50, where a
larger portion of the distribution is accepted, Shan-
non Drop consistently provides positive gains on the
MATH dataset, whereas LN-S Drop often falls below
its own static baseline.
4.3. Stepwise Error Localization Ability (RQ3)
We further evaluate stepwise error localization using
Step-level Localization Accuracy (SLA), which assesses
whether an uncertainty signal can precisely identify erro-
neous reasoning steps, rather than merely detecting unre-
liable samples at the sequence level.
As shown in Ta-
ble 3, the Drop-based criterion consistently outperforms the
sequence-level Avg criterion across all datasets and uncer-
tainty measures on Qwen3-8B. Notably, Shannon entropy
combined with Evidence Drop achieves the best perfor-
mance on every benchmark. For instance, SLA improves
from 27.66% to 46.11% on GSM8K, from 24.62% to
32.90% on MATH, from 26.30% to 41.52% on Olympiad-
7

## Page 8

Mind the Gap: Catching Hallucinations via Evidence Drop
(a) LN-S Average
(b) LogTokU Average
(c) Shannon Average
(d) LN-S Drop
(e) LogTokU Drop
(f) Shannon Drop
Figure 2. Comparison of Uncertainty Distributions. The top row displays sequence-level aggregation methods (Average), while the
bottom row displays our proposed step-level monitoring methods (Drop). The Drop metrics exhibit a clearer separation between correct
and hallucinated responses.
Bench, and from 23.40% to 37.04% on Omni-MATH.
These results indicate that abrupt uncertainty transitions
provide a much stronger signal for step-level error local-
ization than the absolute magnitude of uncertainty.
In contrast, the Avg-based criterion shows limited discrim-
inative power.
While high average uncertainty often re-
ﬂects task difﬁculty, it fails to capture the precise moment
when the reasoning trajectory deviates from the evidence
manifold, thereby diluting critical local anomalies.
The
superior performance of the Drop-based methods conﬁrms
our hypothesis that hallucinations are characterized by non-
smooth disruptions in the uncertainty trajectory, and that
effective error localization requires dynamic process moni-
toring rather than static uncertainty aggregation.
Further, Figure 3 provides a ﬁne-grained comparison be-
tween Entropy-Avg and Entropy-Drop. Entropy-Avg ex-
hibits substantial overlap between correct and erroneous
steps, showing that averaged entropy cannot reliably distin-
guish local reasoning failures. In contrast, Entropy-Drop
achieves clear separation: erroneous steps correspond to
pronounced entropy drops, while correct steps maintain
smooth and stable uncertainty trajectories.
This demon-
strates that it is the dynamic change in uncertainty, rather
than its static value, that serves as an effective indicator for
precise step-level error localization, further validating the
Evidence Manifold perspective.
(a) GSM8K
(b) OlympiadBench
Figure 3.
Comparison of reasoning-correct and reasoning-
incorrect steps on the GSM8K and OlympiadBench datasets
using Qwen3-8B, evaluated by entropy-based uncertainty with
sequence-level averaging (Entropy-Avg) and stepwise Evidence
Drop (Entropy-Drop). The blue dashed line indicates the decision
threshold, obtained on the calibration set with signiﬁcance level
α = 0.1.
8

## Page 9

Mind the Gap: Catching Hallucinations via Evidence Drop
5. Conclusion
In this paper, we introduced a dynamic perspective on hal-
lucination detection by modeling reasoning as a trajectory
on an Evidence Manifold. We demonstrated that halluci-
nations are not merely instances of high uncertainty but are
characterized by Evidence Drops. Extensive experiments
on benchmarks such as GSM8K and MATH show that our
training-free, model-agnostic detector signiﬁcantly outper-
forms sequence-level averaging baselines in both selective
accuracy and step-level error localization. By integrating a
hypothesis testing framework, we provide formal statistical
guarantees to ensuring reliable deployment in complex rea-
soning tasks. Finally, the Evidence Drop mechanism offers
an interpretable pathway toward monitoring and securing
the integrity of multi-step reasoning in LLMs.
6. Limitation
Despite the effectiveness of our Evidence Drop mechanism
in identifying unreliable reasoning steps, a primary limi-
tation of this work is its current focus on passive detection
rather than active intervention.Given our method’s high pre-
cision in step-level error localization, future research could
extend this signal into an online mitigation system.
Acknowledgements
This work was supported by the National Natural Science
Foundation of China ( Nos. 62472315, 62476165).
Impact Statement
This paper presents work whose goal is to advance the ﬁeld
of Machine Learning, speciﬁcally in enhancing the relia-
bility and safety of Large Language Models (LLMs) dur-
ing complex reasoning tasks. By introducing the Evidence
Drop mechanism and providing rigorous statistical guaran-
tees under a hypothesis testing framework, our method fa-
cilitates the safe deployment of LLMs in high-stakes do-
mainssuch as medical diagnosis, ﬁnancial analysis, and le-
gal reasoningwhere logical hallucinations can lead to se-
vere real-world consequences. Furthermore, as a training-
free and model-agnostic approach, our framework signiﬁ-
cantly reduces the computational overhead and carbon foot-
print typically associated with training specialized process
reward models or veriﬁers.
We believe that improving
the transparency and interpretability of AI systems through
step-level error localization fundamentally aligns with the
broader societal goals of developing trustworthy, responsi-
ble, and aligned Artiﬁcial Intelligence. There are no obvi-
ous adverse ethical or societal consequences expected from
this work.
References
Angelopoulos, A., Bates, S., Malik, J., and Jordan, M. I.
Uncertainty sets for image classiﬁers using conformal
prediction. arXiv preprint arXiv:2009.14193, 2020.
Angelopoulos, A. N. and Bates, S. Conformal prediction:
A gentle introduction. Foundations and Trends in Ma-
chine Learning, 16(4):494–591, 2023.
Azaria, A. and Mitchell, T. The internal state of an llm
knows when it’s lying. arXiv preprint arXiv:2304.13734,
2023.
Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani,
R. J. Predictive inference with the jackknife+. The An-
nals of Statistics, 49(1):486–507, 2021.
Bates, S., Angelopoulos, A., Lei, L., Malik, J., and Jordan,
M. I. Distribution-free, risk-controlling prediction sets.
Journal of the ACM (JACM), 68(6):1–34, 2021.
Bazarova, A., Yugay, A., Shulga, A., Ermilova, A.,
Volodichev, A., Polev, K., Belikova, J., Parchiev, R.,
Simakov, D., Savchenko, M., et al.
Hallucination de-
tection in llms with topological divergence on attention
graphs. arXiv preprint arXiv:2504.10063, 2025.
Cheng, J., Su, T., Yuan, J., He, G., Liu, J., Tao, X., Xie,
J., and Li, H. Chain-of-thought prompting obscures hal-
lucination cues in large language models: An empirical
evaluation. In Findings of ACL: EMNLP 2025, pp. 1272–
1305, 2025.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun,
H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J.,
Nakano, R., Hesse, C., and Schulman, J.
Training
veriﬁers to solve math word problems. arXiv preprint
arXiv:2110.14168, 2021a.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., et al. Training veriﬁers to solve math word problems.
arXiv preprint arXiv:2110.14168, 2021b.
Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y.,
Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy
mechanism of reinforcement learning for reasoning lan-
guage models. arXiv preprint arXiv:2505.22617, 2025.
Dhillon, G. S., Deligiannidis, G., and Rainforth, T.
On
the expected size of conformal prediction sets. In Inter-
national Conference on Artiﬁcial Intelligence and Statis-
tics, pp. 1549–1557. PMLR, 2024.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting
hallucinations in large language models using semantic
entropy. Nature, 630(8017):625–630, 2024.
9

## Page 10

Mind the Gap: Catching Hallucinations via Evidence Drop
Gao, B., Song, F., Yang, Z., Cai, Z., Miao, Y., Dong, Q., Li,
L., Ma, C., Chen, L., Xu, R., et al. Omni-math: A univer-
sal olympiad level mathematic benchmark for large lan-
guage models. arXiv preprint arXiv:2410.07985, 2024.
Geifman, Y. and El-Yaniv, R. Selective classiﬁcation for
deep neural networks. Advances in neural information
processing systems, 30, 2017.
He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu,
J., Han, X., Huang, Y., Zhang, Y., Liu, J., Qi, L., Liu,
Z., and Sun, M. OlympiadBench: A challenging bench-
mark for promoting AGI with olympiad-level bilingual
multimodal scientiﬁc problems. In Ku, L.-W., Martins,
A., and Srikumar, V. (eds.), Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 3828–3850,
Bangkok, Thailand, August 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.acl-long.
211. URL https://aclanthology.org/2024.
acl-long.211/.
He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P.,
Zhang, X., Zhang, F., Xu, J., Shen, W., et al.
Sky-
work open reasoner 1 technical report. arXiv preprint
arXiv:2505.22312, 2025.
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J.
Measuring
mathematical problem solving with the math dataset.
arXiv preprint arXiv:2103.03874, 2021.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E.,
Bang, Y. J., Madotto, A., and Fung, P. Survey of halluci-
nation in natural language generation. ACM computing
surveys, 55(12):1–38, 2023.
Kim, H., Lamb, T. A., Bibi, A., Torr, P., and Gal,
Y. Detecting LLM hallucination through layer-wise in-
formation deﬁciency: Analysis of ambiguous prompts
and unanswerable questions.
In Christodoulopou-
los, C., Chakraborty, T., Rose, C., and Peng, V.
(eds.), Proceedings of the 2025 Conference on Em-
pirical Methods in Natural Language Processing, pp.
32310–32322, Suzhou, China, November 2025. As-
sociation for Computational Linguistics.
ISBN 979-
8-89176-332-6.
doi:
10.18653/v1/2025.emnlp-main.
1644. URL https://aclanthology.org/2025.
emnlp-main.1644/.
Lee, D. D., Pham, P., Largman, Y., and Ng, A. Advances
in neural information processing systems 22. Tech Rep,
2009.
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,
B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and
Cobbe, K. Let’s verify step by step. In The Twelfth Inter-
national Conference on Learning Representations, 2023.
Liu, W., Chen, Y., and Yue, X. Building trust in decision
with conformalized multi-view deep classiﬁcation.
In
Proceedings of the 32nd ACM International Conference
on Multimedia, pp. 7278–7287, 2024.
Liu, W., Chen, Y., and Yue, X. Enhancing multi-view clas-
siﬁcation reliability with adaptive rejection. In Proceed-
ings of the AAAI Conference on Artiﬁcial Intelligence,
volume 39, pp. 18969–18977, 2025a.
Liu, W., Chen, Y., and Yue, X.
Enhancing testing-time
robustness for trusted multi-view classiﬁcation in the
wild. In Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pp. 15508–15517, 2025b.
Luo, L., Liu, Y., Liu, R., Phatale, S., Guo, M., Lara, H., Li,
Y., Shu, L., Zhu, Y., Meng, L., et al. Improve mathemat-
ical reasoning in language models by automated process
supervision. arXiv preprint arXiv:2406.06592, 2024.
Ma, H., Chen, J., Zhou, J. T., Wang, G., and Zhang, C.
Estimating llm uncertainty with evidence. arXiv preprint
arXiv:2502.00290, 2025.
Malinin, A. and Gales, M.
Uncertainty estimation in
autoregressive structured prediction.
arXiv preprint
arXiv:2002.07650, 2020.
Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor,
I., Kotek, H., and Belinkov, Y. Llms know more than
they show: On the intrinsic representation of llm halluci-
nations. arXiv preprint arXiv:2410.02707, 2024.
Prystawski, B., Li, M., and Goodman, N. Why think step
by step? reasoning emerges from the locality of experi-
ence. Advances in Neural Information Processing Sys-
tems, 36:70926–70947, 2023.
Shannon, C. E. A mathematical theory of communication.
The Bell system technical journal, 27(3):379–423, 1948.
Shi, J., Yue, X., Liu, W., Chen, Y., and Dong, F. Not all
inconsistency is equal: Decomposing lvlm uncertainty
into belief divergence and belief conﬂict. In Proceed-
ings of the AAAI Conference on Artiﬁcial Intelligence,
volume 40, pp. 25339–25347, 2026.
Sriramanan, G., Bharti, S., Sadasivan, V. S., Saha, S.,
Kattakinda, P., and Feizi, S.
LLM-check: Investigat-
ing detection of hallucinations in large language mod-
els. In The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024. URL https:
//openreview.net/forum?id=LYx4w3CAgy.
Tong, X., Feng, Y., and Zhao, A. A survey on neyman-
pearson classiﬁcation and suggestions for future re-
search. Wiley Interdisciplinary Reviews: Computational
Statistics, 8(2):64–81, 2016.
10

## Page 11

Mind the Gap: Catching Hallucinations via Evidence Drop
Tong, X., Feng, Y., and Li, J. J. Neyman-pearson classiﬁca-
tion algorithms and np receiver operating characteristics.
Science advances, 4(2):eaao1659, 2018.
Turpin, M., Michael, J., Perez, E., and Bowman, S. Lan-
guage models don’t always say what they think: Un-
faithful explanations in chain-of-thought prompting. Ad-
vances in Neural Information Processing Systems, 36:
74952–74965, 2023.
Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D.,
Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce
llms step-by-step without human annotations.
In Pro-
ceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers),
pp. 9426–9439, 2024.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang,
S., Chowdhery, A., and Zhou, D. Self-consistency im-
proves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2022.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F.,
Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought
prompting elicits reasoning in large language models.
Advances in neural information processing systems, 35:
24824–24837, 2022.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 techni-
cal report. arXiv preprint arXiv:2505.09388, 2025a.
Yang, K., Xu, X., Chen, Y., Liu, W., Lyu, J., Lin,
Z., Ye, D., and Yang, S.
Entropic:
Towards sta-
ble long-term training of llms via entropy stabiliza-
tion with proportional-integral control. arXiv preprint
arXiv:2511.15248, 2025b.
Yang, X., Lu, J., and Yu, E. Adapting multi-modal large
language model to concept drift from pre-training on-
wards. In The Thirteenth International Conference on
Learning Representations, 2025c.
URL https://
openreview.net/forum?id=b20VK2GnSs.
Yang, X., Xu, L., Li, H., and Zhang, S. One leaf reveals
the season: Occlusion-based contrastive learning with
semantic-aware views for efﬁcient visual representation.
In International Conference on Machine Learning, pp.
71425–71440. PMLR, 2025d.
Yang, X., Lu, J., and Yu, E.
Walking the tightrope:
Autonomous disentangling beneﬁcial and detrimental
drifts in non-stationary custom-tuning. Advances in neu-
ral information processing systems, 38:116167–116193,
2026a.
Yang, X., Yu, E., Duan, W., and Lu, J. Turning drift into
constraint: Robust reasoning alignment in non-stationary
multi-stream environments. In Forty-third International
Conference on Machine Learning, 2026b. URL https:
//openreview.net/forum?id=jgebUtw1lA.
Zhang, F., Yu, P., Yi, B., Zhang, B., Li, T., and Liu, Z.
Prompt-guided internal states for hallucination detection
of large language models. In Proceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 21806–21818,
2025.
Zheng, C., Zhang, Z., Zhang, B., Lin, R., Lu, K., Yu, B.,
Liu, D., Zhou, J., and Lin, J. Processbench: Identifying
process errors in mathematical reasoning. In Proceed-
ings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
1009–1024, 2025.
11

## Page 12

Mind the Gap: Catching Hallucinations via Evidence Drop
A. Symbols
Table 4. Summary of Notations and Deﬁnitions used in this paper.
Symbol
Description & Physical Meaning
Manifold & Theoretical Framework
M
The low-dimensional Evidence Manifold representing valid reasoning trajectories.
Ei
Latent Evidence State at step i. Represents the intrinsic logical support.
Ui
Observed Uncertainty Vector (e.g., logits) at step i. The tangible observation of Ei.
Ptrue
The True Distribution on the manifold (Ideal Logic).
Ptrue(v)
The probability of generating a speciﬁc token v on the ideal manifold. Note: The speciﬁc
geometry is determined by the chosen uncertainty metric.
Ptrain
The Training Distribution learned by the model.
∆j
Evidence Drop. The quantitative decline of ˜Ej relative to ˜Ej−1, indicating a hallucination.
Model Inputs & Calculation
y<t
Token History (Context). The sequence of tokens generated prior to step t.
p(v | y<t)
Next-Token Probability. The model’s predicted probability for token v given history y<t.
ˆEi
Calculated Evidence Value at step i (e.g., Shannon Entropy). It is a scalar realization of Ei.
˜Ei
Smoothed Evidence Value. The EMA-smoothed version of ˆEi to mitigate noise.
V
The model’s complete vocabulary size.
Evaluation & Datasets
ϕ(x, y)
Sequence-Level Uncertainty Score. A function (like LN-S) evaluating the quality of the entire
response y.
D
The dataset containing correct/valid reasoning samples (Positive samples).
D−
The Hallucination Dataset containing incorrect/fabricated samples (Negative samples).
K
Hyperparameter. Top-K candidates used for evidence approximation (e.g., K = 20).
B. Theoretical analysis
In this section,we will provide a detailed proof of our theoretical framework.
B.1. Problem Setup
Evidence Manifold and Factorization
We assume that the reasoning process of an LLM is driven by a series of latent Evidence States E = (E1, E2, . . . , ET ),,
which exist on a low-dimensional data manifold M. Here, Ei can not only represent an abstract logical step, but also be
speciﬁcally quantitied by different uncertainty measures.
We assume that the true evidence distribution deﬁned on the manifold satisﬁes the Markov Property. That is, the joint
distribution of the entire evidence chain can be factorized into the product of local conditional probabilities:
Ptrue(E1, . . . , EN) = P(E1)
N−1
Y
t=1
Ptrue(Et+1 | Et)
(14)
This formula implies that a valid reasoning process is a local transmission of evidence on the manifold, where the evidence
state Et+1 at each step only depends on the preceding step Et.
Observation Model: Indices and Uncertainty Measurement. To capture these latent evidence states, we deﬁne the
observation data as serialized index-observation pairs. For each step in the reasoning, the model outputs a tuple (i, Ui),
where i is the step index and Ui is the corresponding vector (e.g., logits or probability distribution) from which uncertainty
is measured. The latent evidence state Ei can be derived from Ui, i.e., Ei = f(Ui).
12

## Page 13

Mind the Gap: Catching Hallucinations via Evidence Drop
Local Observation Assumptions. We deﬁne the distributional characteristics of the training data Ptrain through the follow-
ing two topological assumptions:
Assumption 1 (Locality of Transitions). For any two non-adjacent evidence states Ei and Ej with |i−j| > 1, the training
data contains no direct transitions between them:
Ptrain(Uj | Ui) = 0.
(15)
This implies that a direct skip across the manifold constitutes an Out-of-Distribution (OOD) event.
Assumption 2 (Local Consistency). For adjacent evidence states Ei and Ei+1, the observed transition distribution pre-
serves the true local transition structure on the evidence manifold:
Ptrain(Ui+1 | Ui) ∝Ptrue(Ei+1 | Ei).
(16)
This implies that the model learns to faithfully approximate the local evidence transmission logic.
B.2. Problem Transformation
Proposition 1. Let αdata ≥0, αreg ≥0. Deﬁne the risk function R(q) as the weighted sum of a data ﬁtting term and a
regularization term:
R(q) = αdataEPtrain(U)[−log q(U)] + αregEU(U)[−log q(U)]
(17)
where U is the observed values vector, and U is the uniform distribution. Then, the optimal distribution q∗= arg minq R(q)
that minimizes the risk satisﬁes:
q∗(U) =
αdata
αdata + αreg
Ptrain(U) +
αreg
αdata + αreg
U(U)
(18)
Remark: The risk function formulated in Eq. (5) of the main text corresponds to a special case of this general objective
where we set the weights to αdata = 1 and αreg = λ.
Proof. We assume the probability distribution is discrete (deﬁned on a ﬁnite vocabulary V). To solve for the extremum
under the constraint P
U q(U) = 1, we construct the Lagrangian function:
L(q, λ0) = −αdata
X
U
Ptrain(U) log q(U) −αreg
X
U
U(U) log q(U) + λ0
 X
U
q(U) −1
!
(19)
The ﬁrst-order conditions with respect to q(U) and λ0 are:
∂L
∂q(U) = −αdata
Ptrain(U)
q(U)
−αreg
U(U)
q(U) + λ0 = 0
(20)
∂L
∂λ0
=
X
U
q(U) −1 = 0
(21)
From Eq. (20), we obtain the expression for q(U):
q(U) = αdataPtrain(U) + αregU(U)
λ0
(22)
To solve for the normalization constant λ0, we substitute the above expression into the normalization constraint (21):
X
U
αdataPtrain(U) + αregU(U)
λ0
= 1
(23)
Rearranging gives:
1
λ0




αdata
X
U
Ptrain(U)
|
{z
}
1
+αreg
X
U
U(U)
|
{z
}
1




= 1
(24)
13

## Page 14

Mind the Gap: Catching Hallucinations via Evidence Drop
Since Ptrain and U are both valid probability distributions, their sums are 1, thus yielding:
λ0 = αdata + αreg
(25)
Substituting λ0 back into the expression for q(U) yields the stated conclusion.
□
B.3. Theorem Statement
Theorem 1: Evidence Collapse from Manifold Deviation
Preliminaries Let U be the uniform distribution deﬁned on the vocabulary V. Let Ptrain be the joint distribution over
evidence state indices i and observed values U deﬁned by the “Local Observation Constraint” in Section B.1. Let
H(p, q) denote the cross-entropy between distributions p and q. We consider the following risk function with entropy
regularization:
R(q) = H(Ptrain, q) + λH(U, q)
(26)
where λ > 0 is the regularization coefﬁcient.
Theorem Conclusion The optimal estimator q∗= arg minq R(q) that minimizes the risk satisﬁes the following properties
and exhibits a signiﬁcant “Evidence Drop”:
• For all pairs of adjacent evidence states (Ei, Ei+1) on the manifold:
The optimal predictive distribution q∗(U | Ei) is a weighted mixture of the true manifold distribution and the
uniform distribution:
q∗(U | Ei) = αPtrue(U | Ei) + (1 −α)U(U)
(27)
where α ∈(0, 1) is a weight close to 1. In this case, the observed evidence quantity E remains in a high-
conﬁdence interval:
E(q∗) ≈E(Ptrue) ≈0
(28)
• For all pairs of non-adjacent evidence states (Ei, Ek) (where |i −k| > 1):
The optimal predictive distribution q∗(U | Ei) strictly degenerates to the uniform distribution:
q∗(U | Ei) = U(U) = 1
|V|
(29)
In this case, the observed evidence quantity E undergoes a drastic collapse, reaching the theoretical lower bound:
E(q∗) = log
 K
|V|

≪0
(30)
Corollary Therefore, when the model attempts to perform a non-adjacent jump, the variation in evidence value (i.e., the
Evidence Drop) ∆is mathematically bounded as:
∆= Ehallucination −Evalid ≈log
 K
|V|

≪0
(31)
Proof.
1. Orthogonal Decomposition of Risk
Using the Law of Iterated Expectations, we decompose the total risk R(q) into a sum over evidence state transition pairs
(i, k). According to the Local Observation Constraint deﬁned in Section B.1, we partition the summation domain into
adjacent (Zadj) and non-adjacent (Znon) sets:
R(q) = EPtrain[−log q] + λEU[−log q]
(32)
=
X
(i,k)∈Zadj

Ptrain(i, k)H(Ptrue, q) + λH(U, q)
|
{z
}
Term I: Visible Transitions


+
X
(i,k)∈Znon

0 · H(·, q) + λH(U, q)
|
{z
}
Term II: Invisible Transitions


(33)
14

## Page 15

Mind the Gap: Catching Hallucinations via Evidence Drop
Note that in Term II, since Ptrain(i, k) = 0, the data ﬁtting term vanishes, and the risk is constituted solely by the regular-
ization term.
2. Extremal Analysis
We solve for the optimal estimator q∗= arg minq R(q) for the two cases described above.
Case A: On-Manifold Adjacent Transitions ((i, k) ∈Zadj)
In this case, the local risk function is r(q) = αH(Ptrue, q)+λH(U, q). Constructing the Lagrangian L = r(q)+γ(P q−1)
and solving for ∇qL = 0:
−αPtrue(U)
q(U)
−λU(U)
q(U) + γ = 0 =⇒q∗(U) ∝αPtrue(U) + λU(U)
(34)
Since α ≫λ on the training manifold, the optimal solution is dominated by the true distribution:
q∗(U | Ei) ≈Ptrue(U | Ei)
(35)
Analyzing the ﬁrst term (Term I) of the risk decomposition in Eq. (32) (i.e., the adjacent transitions), and substituting the
derived approximation q∗≈Ptrue into the evidence deﬁnition, we obtain:
Evalid = log


X
v∈Top-K(q∗)
q∗(v)

≈log


X
v∈Top-K(Ptrue)
Ptrue(v)


(36)
Since valid reasoning steps on the manifold typically exhibit low uncertainty (i.e., the probability mass is highly concen-
trated on a few correct tokens), the cumulative probability of the top-K candidates approaches 1:
X
v∈Top-K(Ptrue)
Ptrue(v) ≈1
=⇒
Evalid ≈log(1) = 0
(37)
Case B: Off-Manifold Non-Adjacent Jumps ((i, k) ∈Znon)
Here, the data term weight α = 0, and the local risk function degenerates to r(q) = λH(U, q). The optimal solution must
minimize the cross-entropy with the uniform distribution:
∇q[λH(U, q)] = 0 =⇒q∗(U | Ei) = U(U) = 1
|V|
(38)
This represents a perfectly ﬂat distribution. The Top-K evidence Ehallucination is thus:
Ehallucination = log


K
X
j=1
1
|V|

= log
 K
|V|

(39)
3. Evidence Drop Bound
Deﬁning the evidence variation ∆= Ehallucination −Evalid, and based on the results above:
∆= log
 K
|V|

−Evalid
(40)
≤log
 K
|V|

−(1 −ϵ) ≈log K −log |V| ≪0
(41)
where |V| is the vocabulary size (typically > 104) and K is a constant. Therefore, ∆is a signiﬁcant negative value.
□
15

## Page 16

Mind the Gap: Catching Hallucinations via Evidence Drop
C. Hypothesis Testing
In this part,we will elaborate in detail on how to obtain statistically guaranteed thresholds through hypothesis testing.
C.1. Statistical Guarantees in Machine Learning
Reliable deployment of machine learning systems often requires rigorous statistical guarantees beyond empirical perfor-
mance. Conformal prediction (Angelopoulos & Bates, 2023) provides distribution-free guarantees on coverage by con-
structing prediction sets, while risk control frameworks (Bates et al., 2021) bound expected loss or error under speciﬁed
constraints. These approaches have been widely adopted in classiﬁcation and multi-view classiﬁcation (Angelopoulos et al.,
2020; Bates et al., 2021; Liu et al., 2025a;b; 2024), regression (Lee et al., 2009; Barber et al., 2021; Yang et al., 2026a),
and selective prediction settings (Geifman & El-Yaniv, 2017; Dhillon et al., 2024). In parallel, hypothesis testing offers
a principled mechanism for controlling error rates in binary decision problems, such as the False Positive Rate or Type I
error, with ﬁnite-sample guarantees (Tong et al., 2016). In this work, we formulate hallucination detection as a hypothesis
testing problem, where accepting a hallucinated reasoning trace corresponds to a Type I error. By integrating our Evidence
Drop statistic with a NeymanPearson style decision rule, we obtain explicit ﬁnite-sample control over the false acceptance
rate.
C.2. Hypothesis Testing
We deploy the derived risk metric ϕ(x, y) ,which quantiﬁes the severity of evidence drops,to construct a binary decision
rule. Unlike empirical thresholds, we formulate this as a hypothesis testing problem under the Neyman-Pearson paradigm
to provide a ﬁnite-sample statistical guarantee against hallucinations:
H0 : The reasoning chain deviates from M (Hallucination).
H1 : The reasoning chain is valid on M (Reliable).
(42)
We aim to control the Type I Error (accepting a hallucination) at a user-speciﬁed signiﬁcance level α (e.g., α = 0.05).
Calibration and Statistical Bound. To determine the critical decision threshold τ, we utilize a calibration set of known
hallucinations, Dcal = {(xi, yi)|yi is incorrect}. We compute the risk scores Scal = {ϕ(xi, yi)} for these failure cases.
Since higher ϕ values indicate severe evidence drops (high risk), a hallucination is falsely accepted if its risk score falls
below the threshold τ. We seek the smallest τ such that the probability of this error is bounded:
P(ϕ(x, y) ≤τ | H0 is true) ≤α
(43)
To ensure this bound holds for ﬁnite samples without assuming a parametric distribution (e.g., Gaussian), we employ the
concentration inequality based on the Binomial tail. Motivated by (Tong et al., 2018),we select ˆτ as the (1−α)-th quantile
of the empirical distribution of Scal, adjusted for ﬁnite sample size Ncal to satisfy:
Ncal
X
j=k
Ncal
j

(1 −α)jαNcal−j ≤δ
(44)
where δ is the conﬁdence level of the guarantee itself. The Decision Rule. With the calibrated threshold ˆτ, the ﬁnal
deployment policy is:
D(y) =
(
Accept
if ϕ(x, y) ≤ˆτ
(Safe Manifold Trajectory)
Reject
if ϕ(x, y) > ˆτ
(Detected Deviation)
(45)
D(y) is equivalent to Decision(y). This mechanism ensures that, with probability 1 −δ, the rate of accepting hallucinations
is strictly upper-bounded by α, providing a rigorous safety certiﬁcate for the LLM’s reasoning process.
16

## Page 17

Mind the Gap: Catching Hallucinations via Evidence Drop
D. Baselines
To ensure a rigorous comparison, we implement the following uncertainty baselines:
• Shannon Entropy: Following common practice for high-cardinality vocabularies, we compute a computationally
efﬁcient approximation using the top-K logits (K = 20). For each step t, we re-normalize the probabilities of the
top-K logits to form a local distribution ˜P(v|y<t). The step-wise entropy is
Hi = −
X
v∈VK
˜P(v|y<i) log ˜P(v|y<i).
(46)
The sequence-level uncertainty score is the mean entropy over the trajectory of length T: ϕShannon = 1
T
PT
i=1 Hi.
• LogTokU (Ma et al., 2025): This baseline measures the Evidence Mass Mt, deﬁned as the sum of log-probabilities
for the top-K candidates:
Mi =
X
v∈VK
log P(v|y<i).
(47)
High mass indicates a concentration of conﬁdence among the top candidates. The ﬁnal risk score is the negative
average mass: ϕLogTokU = −1
T
PT
i=1 Mi.
• LN-S (Malinin & Gales, 2020): This baseline utilizes the Length-Normalized Scoring of the generated sequence.
Under greedy decoding (sampling temperature τ = 0), the model selects the token yt with the maximum likelihood
at each step. We compute the average negative log-probability:
ϕLN-S = −1
T
T
X
i=1
log P(v|y<i),
(48)
where P(v|y<i) represents the probability of the token actually generated at step t.
17

## Page 18

Mind the Gap: Catching Hallucinations via Evidence Drop
E. Supplementary experiments
E.1. Hyperparameter Robustness
The Table 5 below present our ablation of our drop methods on the MATH dataset with Qwen3-8B. It is important to note
that the numbers reported in the tables represent the selective accuracy (Acc). These results are achieved by rejecting
unreliable reasoning trajectories using a statistically guaranteed decision threshold, which is rigorously derived via our
hypothesis testing framework at a signiﬁcance level of α = 0.05. In each experiment, we isolate the effect of a single
hyperparameter by ﬁxing the other two to our default conﬁguration (i.e., Max Drops M = 5, EMA span = 5, and Top-
K = 20).
For a direct comparison, we include our strongest baseline (Shannon Avg, 70.24%) in our paper to demonstrate the robust-
ness of our Drop method. This serves as the reference baseline for all ablation results below.
Table 5. Ablation study of Evidence Drop on the MATH dataset (Qwen3-8B). The reference baseline (Shannon Avg) achieves a selective
accuracy of 70.24%. Default settings are M = 5, EMA span = 5, and K = 20.
(a) Effect of Max Drops M (EMA span = 5)
M
LN-S Drop
LogTokU Drop
Shannon Drop
1
84.58
73.40
87.26
3
85.77
81.56
87.92
5 (Ours)
86.94
81.55
88.26
10
85.93
83.33
90.75
All
55.43
65.66
74.81
(b) Effect of EMA Span (M = 5)
EMA Span
LN-S Drop
LogTokU Drop
Shannon Drop
1
84.58
73.40
87.26
3
85.77
81.56
87.92
5 (Ours)
86.94
81.55
88.26
10
85.93
83.33
90.75
All
55.43
65.66
74.81
(c) Effect of Top-K (EMA span = 5, M = 5)
Top-K
LN-S Drop
K = 2
86.05
K = 5
85.42
K = 10
87.64
K = 20 (Ours)
86.94
Since K has minimal impact, we use LN-S Drop as a representative example. Following Ma et al. (Ma et al., 2025), we
default to K = 20. Table 5(c) demonstrates high robustness across K ∈[2, 20], proving our method captures intrinsic
dynamics without sensitive tuning.
The results demonstrate exceptional hyperparameter robustness. Across M, EMA span, and Top-K, performance remains
highly stable within broad, non-extreme windows. Degradation occurs only in extreme settings that inherently violate our
theoretical premises (e.g., M →All dilutes local signals via global averaging).
18

## Page 19

Mind the Gap: Catching Hallucinations via Evidence Drop
E.2. Comparison with other methods
To compare against generation-based uncertainty methods like Semantic Entropy, we evaluated Self-Consistency (SC),
which shares the exact same foundational mechanism: sampling multiple reasoning paths to measure output consis-
tency/entropy. We tested SC with N = 5, 10, and 15 sampling paths on the MATH dataset. For clarity, the SC results
represent the ﬁnal accuracy after majority voting, which serves as a direct comparison against the baseline accuracy of the
pretrained model with zero voting overhead.
Table 6. Performance comparison of Self-Consistency (SC) on the MATH dataset. The results indicate the accuracy (%) of answers after
majority voting.
Voting Paths (N)
Qwen3-8B
Qwen3-4B
Pretrained
+ SC
Pretrained
+ SC
N = 5
66.12
69.76 (+3.64)
57.92
63.72 (+5.80)
N = 10
66.40
70.72 (+4.32)
58.56
65.36 (+6.80)
N = 15
65.96
70.56 (+4.60)
57.88
65.04 (+7.16)
Analysis: As shown in Table 6, while multi-path methods like SC incur a 15× computational overhead for marginal
gains (e.g., reaching 70.56% on Qwen3-8B), our single-path Shannon Drop method avoids these massive generation costs
entirely. With just a single forward pass, it delivers a dramatic 10–20% absolute gain, reaching an impressive 88.26%.
E.3. Generalization to Daily Life Topics
To demonstrate how our method detects daily life topic hallucinations, we conducted new experiments on the ProofWriter
dataset.
ProofWriter is a standard benchmark for deductive logical reasoning constructed entirely from natural language statements
about everyday entities and scenarios. It perfectly simulates daily-life factual logic.
The results are presented in Table 7. We evaluate the Area Under the Risk-Coverage curve (AURC, where lower is better)
on the ProofWriter dataset using Qwen3-8B, covering both Open World Assumption (OWA) and Closed World Assumption
(CWA) at depth-3. The experimental settings remain consistent with our main paper.
Table 7. AURC (lower is better) on the ProofWriter dataset (Qwen3-8B) for Open World Assumption (OWA) and Closed World As-
sumption (CWA) at depth-3. The best results for each setting are highlighted in bold.
Setting
Aggregation
LN-S
LogTokU
Shannon
OWA (Depth-3)
Average
711.82
685.96
717.17
Drop (Ours)
801.13
648.28
589.02
CWA (Depth-3)
Average
496.73
579.03
484.33
Drop (Ours)
729.19
550.80
418.96
Analysis: As shown in Table 7, our Evidence Drop method utilizing Shannon Entropy as the uncertainty measure achieves
the best performance. It signiﬁcantly outperforms the global averaging baselines by yielding the lowest AURC scores
under both OWA and CWA settings, demonstrating its strong generalization capabilities in daily-life factual logic.
19

## Page 20

Mind the Gap: Catching Hallucinations via Evidence Drop
F. Case Analysis
Case Description. "There are 6 periods in the day for a normal student but John has to take 2 extra classes. Each
class is 40 minutes long. He goes to class for 5 days a week. He then spends 1/16 of his weekly minutes each on
Saturday and Sunday as extra learning time. How many hours a week does he spend learning?"
The problem from GSM8K dataset requires calculating weekly study hours, explicitly stating that "John has to take 2 extra
classes." However, the model hallucinates by ignoring this condition, assuming a standard schedule.
Forensic Detection. As visualized in Figure 4, our method, based on Shannon Entropy, detects this reasoning error through
distinct Evidence Drops:
• The Root Cause (∆≈−0.35): The ﬁrst signiﬁcant drop occurs precisely at the token "each" in the phrase "He has
6 periods a day, [each]...". This drop signals the model’s internal conﬂict or attention failure regarding the "2 extra
classes" constraint, ﬂagging the exact moment the premise was distorted.
• Error Propagation (∆≈−0.45): As the reasoning proceeds based on the ﬂawed premise (6 × 40 instead of 8 × 40),
a secondary, deeper drop is triggered at the calculation unit "per". This conﬁrms that the initial uncertainty has
propagated into a conﬁdent-sounding yet factually incorrect calculation.
This case study reﬂects that our Worst-Drop metric does not merely ﬂag the ﬁnal wrong answer, but acts as a good tool to
trace where the hallucination happens, distinguishing between the origin of the error and its downstream consequences.
Furthermore, we’ll analyze more on MATH dataset literally, in which the difﬁculty level varys from 1 to 4:
20

## Page 21

Mind the Gap: Catching Hallucinations via Evidence Drop
Figure 4. Traceability of Reasoning Collapse. A visualization of evidence drops where the model fails to account for “2 extra classes.”
Our method effectively localizes the failure genealogy: (1) Root Cause (Token “each”, ∆≈−0.35): The initial drop ﬂags the precise
moment of attention conﬂict where the premise was overlooked. (2) Error Propagation (Token “per”, ∆≈−0.45): As the model
proceeds with the wrong calculation (6 × 40 instead of 8 × 40), uncertainty spikes at the calculation unit. (3) Final Collapse (Token
“Therefore”): The drop at the concluding connector reﬂects the model’s implicit uncertainty in the derived outcome.
21

## Page 22

Mind the Gap: Catching Hallucinations via Evidence Drop
Example Model Output & Metrics of MATH
Question:
Zach has three bags and a bunch of pencils to be placed into the bags. He is told to place the greatest number of
pencils possible into each of the three bags while also keeping the number of pencils in each bag the same. What is
the greatest number of pencils he could have left over?
Generated Response (with Evidence Drops):
To maximize [the ▼-44.91] number of [pencils ▼-44.63] in each bag, we
need to find the greatest common [divisor ▼-39.45] (G[CD ▼-60.08]) of the
number of pencils.
However, the problem doesn’t specify the total number
of pencils.
Therefore, we cannot determine a unique answer.
The answer is
\[boxed ▼-101.85]{0}
Results Comparison:
• Pred Answer: 0
• GT Answer: 2
• Difﬁculty Level: 1 (Easy)
Raw Curve Data (metric_type: LogTokU Drop):
[ -100.2, -104.3, -149.2, -175.4, -195.5, -240.2, -215.0, -225.4, -237.2, -219.7,
-190.2, -167.2, -180.4, -155.8, -172.3, -159.4, -173.6, -213.0, -219.1, -231.9,
-292.0, -291.4, -268.1, -241.6, -212.7, -224.8, -245.2, -204.3, -184.9, -222.1,
-202.3, -196.9, -183.6, -199.3, -193.6, -201.6, -209.4, -231.5, -241.4, -280.6,
-244.0, -190.5, -218.6, -191.1, -178.5, -180.0, -197.0, -193.9, -209.3, -194.4,
-168.2, -164.6, -175.1, -165.5, -267.3, -263.3, -225.8, -222.7 ]
Explanation:
Significant evidence drops at “divisor” (∇
−
39.45) and “CD” (∇
−
60.08)
effectively signal the model’s hesitation as it attempts to apply an invalid
mathematical template (GCD). Crucially, the most dramatic drop at “boxed”
(∇
−
101.85) quantifies the distributional conflict between the model’s
internal admission of ignorance (“cannot determine”) and the external
constraint to output an answer, successfully flagging the final forced
output “0” as highly unreliable.
22

## Page 23

Mind the Gap: Catching Hallucinations via Evidence Drop
Example Model Output & Metrics of MATH
Question:
February 1, 2008 is a Friday. The last Friday of February 2008 will be February x, 2008. If 2008 is a leap year,
what is the value of x?
Generated Response (with Evidence Drops):
February has [29 ▼-94.83] [days ▼-97.10] in a leap year.
Since February
1 is a Friday, we can count the number of days from February 1 to February
29 to find the last Friday.
There are [28 ▼-91.11] days from February 1
to February 28, which is exactly 4 weeks.
Therefore, February 28 is also
a Friday.
The last Friday is February 28 because February 29 is a [Saturday
▼-118.01].
The answer is \[boxed ▼-99.29]{28}
Results Comparison:
• Pred Answer: 28
• GT Answer: 29
• Difﬁculty Level: 2 (A Little Bit Easy)
Raw Curve Data (metric_type: LogTokU Drop):
[ -101.3, -131.5, -168.6, -263.4, -360.5, -346.9, -305.3, -279.2, -297.8, -309.6,
-290.6, -246.5, -202.7, -192.5, -206.5, -258.6, -291.3, -298.6, -315.0, -338.5,
-337.2, -266.4, -221.5, -198.0, -187.0, -200.5, -233.3, -261.6, -230.6, -239.7,
-265.2, -315.3, -308.6, -303.3, -284.4, -313.3, -358.6, -449.7, -380.5, -302.1,
-272.2, -255.4, -259.9, -296.6, -277.3, -248.4, -203.2, -239.0, -250.9, -295.6,
-360.7, -365.0, -308.4, -296.9, -323.8, -441.8, -395.0, -373.8, -381.3, -371.2,
-407.9, -497.1, -411.3, -339.8, -309.3, -294.5, -289.5, -344.4, -353.3, -311.0,
-285.6, -244.8, -274.6, -269.3, -299.7, -364.8, -446.5, -423.2, -384.7, -383.9,
-382.8, -384.8, -362.0, -316.1, -247.9, -242.6, -267.9, -252.6, -235.9, -265.3,
-313.3, -355.4, -322.2, -251.2, -237.8, -262.0, -331.4, -388.2, -375.5, -343.0,
-320.5, -338.1, -313.4, -287.7, -252.7, -264.9, -301.7, -288.8, -388.0, -410.7,
-448.9, -473.4, -398.8 ]
Explanation:
This case illustrates a classic “Concept Conflation” failure.
The reasoning
error originates at “28” (∇
−
91.11), where the model confuses the duration
(28 days) with the date (February 28th).
This initial logical slip triggers
a cascade of failures:
the subsequent massive drop at “Saturday” (∇−118.01)
reflects the inevitable contradiction with the leap year premise, and the
final drop at “boxed” (∇
−
99.29) signals the model’s high uncertainty when
forcing the incorrect conclusion.
23

## Page 24

Mind the Gap: Catching Hallucinations via Evidence Drop
Example Model Output & Metrics of MATH
Question:
Solve the congruence 5n ≡8 mod 17. (Give an answer between 0 and 16.)
Generated Response (with Evidence Drops):
We need to find the [multiplicative ▼-91.57] inverse of 5 modulo 17.
Let’s
try some numbers.
5*1=5, 5*2=10, 5*3=15, 5*4=20 ≡3, 5*5=25 ≡8.
So the
inverse of 5 modulo 17 is [5 ▼-98.44].
Therefore, multiplying both sides of
the congruence by 5 gives n ≡5*8 = 40 ≡[6 ▼-94.95] mod 17.
The answer is
\[boxed ▼-81.69]{6}
Results Comparison:
• Pred Answer: 6
• GT Answer: 5
• Difﬁculty Level: 3 (Medium)
Raw Curve Data (metric_type: LogTokU Drop):
[ -97.8, -103.2, -152.9, -169.7, -171.5, -167.8, -259.3, -291.2, -293.3, -293.9,
-322.2, -289.5, -303.5, -339.9, -418.2, -339.7, -261.2, -208.9, -226.4, -190.7,
-161.4, -176.1, -161.4, -166.9, -161.1, -224.1, -200.5, -176.1, -187.7, -228.3,
-209.8, -223.1, -245.9, -280.8, -282.4, -306.5, -319.4, -343.4, -324.1, -304.7,
-332.3, -339.9, -331.3, -345.6, -351.8, -363.8, -329.8, -306.5, -343.7, -343.9,
-329.8, -334.9, -336.1, -354.0, -286.3, -288.1, -264.0, -263.8, -282.9, -300.5,
-300.7, -315.2, -315.0, -335.4, -342.4, -345.3, -293.8, -253.7, -207.7, -204.2,
-224.2, -250.5, -279.9, -308.1, -290.3, -311.9, -362.5, -460.9, -431.8, -393.9,
-340.9, -278.8, -248.0, -210.6, -227.0, -209.8, -247.5, -303.4, -297.4, -289.8,
-291.0, -354.5, -402.4, -355.2, -336.8, -337.5, -307.5, -257.8, -273.9, -365.6,
-344.4, -310.8, -321.4, -324.5, -307.0, -321.9, -317.0, -350.5, -340.7, -414.9,
-376.7, -370.6, -317.1, -319.8, -358.8, -453.8, -406.9, -353.5, -311.4, -305.7,
-334.7, -317.8, -399.5, -414.6, -393.0, -340.6 ]
Explanation:
This case demonstrates a distinct “Logic Leap”.
The model performs valid
arithmetic steps (5 × 5
=
25
≡
8) but then makes a baseless assertion that
the inverse is 5 (whereas the true inverse is 7).
The massive evidence drop
at this exact token (∇
−
98.44) signals the model’s awareness of the logical
disconnect.
The subsequent drops at the final answer (∇
−
94.95, ∇
−
81.69)
reflect the propagated uncertainty stemming from this initial fallacy.
24

## Page 25

Mind the Gap: Catching Hallucinations via Evidence Drop
Example Model Output & Metrics of MATH
Problem Description:
Recall that if b is a residue modulo m, then the modular inverse of b is the residue c for which bc ≡1 (mod m).
The table below displays the inverses for the ﬁrst 9 positive residues modulo 47.
b
1
2
3
4
5
6
7
8
9
Inverse of b
1
24
16
12
19
8
27
6
21
Question: Find the modular inverse of 35 (mod 47). Express your answer as an integer from 0 to 46, inclusive.
Generated Response (with Evidence Drops):
Since 35 is congruent to −12 [ ▼-0.18]
(mod 47), we can find the inverse of
−12 [ ▼-0.21] modulo 47.
The inverse of −12 is the negative of the inverse
of 12, so we need to find the inverse of 12 [. ▼-0.25].
From the table, the
inverse of 12 is 12.
Therefore, the inverse of −12 is −12, so the inverse
of 35 is −12 (mod 47).
[ Since ▼-0.35] −12 ≡35 (mod 47), the [ inverse ▼-0.19]
of 35 is 35.
The answer is \boxed{35}
Results Comparison:
• Pred Answer: 35
• GT Answer: 43
• Difﬁculty Level: 4 (A Little Bit Difficult)
Raw Curve Data (metric_type: LN-S Drop):
[ -0.6, -0.7, -0.6, -0.5, -0.5, -0.5, -0.4, -0.4, -0.4, -0.3, -0.3, -0.3, -0.3, -0.3,
-0.3, -0.3, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
-0.1, -0.0, -0.0, -0.0, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.0, -0.0, -0.1, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.0, -0.1, -0.1, -0.1, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.1, -0.1,
-0.0, -0.0, -0.0, -0.0, -0.1, -0.1, -0.2, -0.2, -0.2, -0.1, -0.1, -0.2, -0.2, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.1, -0.1, -0.1,
-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
-0.1, -0.1, -0.1 ]
Explanation:
The failure sequence initiates at Index 30 ("Trigger"), where the model
misreads the table to hallucinate inv(12)
=
12; this contradicts latent
mathematical constraints (122
≡
3
̸=
1), causing an immediate evidence drop
(∇≈−0.62).
This error subsequently propagates to inv(−12) = −12 at Index 68,
sustaining an off-manifold trajectory (∇
≈
−0.75).
The process culminates
in complete semantic degeneration at Index 147 ("Final Collapse"), where the
mathematically impossible conclusion inv(35)
=
35 coincides with a flatlined
evidence score (∇
≈
−0.58), confirming the irreversibility of the reasoning
failure.
25
